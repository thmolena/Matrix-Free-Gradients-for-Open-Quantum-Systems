"""Matrix-free learning on site-resolved 51-ion Néel-state observations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .field_data import load_neel_site_imbalance
from .tangent import augmented_tangent_action


N_IONS = 51
STATE_DIMENSION = 2 * N_IONS


def chain_laplacian(vector: np.ndarray) -> np.ndarray:
    """Apply the open-chain graph Laplacian without storing a matrix."""

    value = np.asarray(vector)
    if value.shape != (N_IONS,):
        raise ValueError(f"expected a length-{N_IONS} vector")
    result = np.empty_like(value)
    result[0] = value[1] - value[0]
    result[-1] = value[-2] - value[-1]
    result[1:-1] = value[:-2] + value[2:] - 2.0 * value[1:-1]
    return result


def generator_matvec(parameters: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Apply a damped, coupled 51-site oscillator generator."""

    omega, gamma, coupling = np.asarray(parameters, dtype=float)
    if gamma < 0.0 or coupling < 0.0:
        raise ValueError("damping and coupling must be nonnegative")
    value = np.asarray(vector)
    if value.shape != (STATE_DIMENSION,):
        raise ValueError(f"expected a length-{STATE_DIMENSION} state")
    x = value[:N_IONS]
    y = value[N_IONS:]
    return np.concatenate(
        (
            -gamma * x - omega * y + coupling * chain_laplacian(x),
            omega * x - gamma * y + coupling * chain_laplacian(y),
        )
    )


def derivative_matvec(direction: int, vector: np.ndarray) -> np.ndarray:
    """Apply one parameter derivative of :func:`generator_matvec`."""

    value = np.asarray(vector)
    x = value[:N_IONS]
    y = value[N_IONS:]
    if direction == 0:
        return np.concatenate((-y, x))
    if direction == 1:
        return -value
    if direction == 2:
        return np.concatenate((chain_laplacian(x), chain_laplacian(y)))
    raise ValueError("direction must be 0, 1, or 2")


def augmented_log_norm_bound(parameters: np.ndarray, direction: int) -> float:
    """Return a structural Euclidean logarithmic-norm upper bound.

    The base generator has logarithmic norm at most ``-gamma`` because the
    rotation is skew symmetric and the chain Laplacian is negative
    semidefinite.  For ``[[A,E],[0,A]]`` the block Hermitian part adds at most
    ``||E||_2/2``.  The derivative norms are 1, 1, and at most 4.
    """

    _, gamma, _ = np.asarray(parameters, dtype=float)
    direction_norm = (1.0, 1.0, 4.0)[direction]
    return float(-gamma + 0.5 * direction_norm)


@dataclass(frozen=True)
class SitePrediction:
    prediction: np.ndarray
    jacobian: np.ndarray
    largest_truncation_estimate: float
    generator_applications: int
    derivative_applications: int


def predict_and_jacobian(
    parameters: np.ndarray,
    times: np.ndarray,
    initial_x: np.ndarray,
    *,
    steps: int = 28,
) -> SitePrediction:
    """Predict every ion and differentiate by matrix-free Arnoldi actions."""

    params = np.asarray(parameters, dtype=float)
    time = np.asarray(times, dtype=float)
    initial = np.concatenate(
        (np.asarray(initial_x, dtype=float), np.zeros(N_IONS, dtype=float))
    )
    prediction = np.empty((time.size, N_IONS), dtype=float)
    jacobian = np.empty((time.size, N_IONS, 3), dtype=float)
    counters = {"generator": 0, "derivative": 0}
    largest_truncation_estimate = 0.0

    def apply_generator(vector: np.ndarray) -> np.ndarray:
        counters["generator"] += 1
        return generator_matvec(params, vector)

    for row, instant in enumerate(time):
        for direction in range(3):
            def apply_derivative(
                vector: np.ndarray, selected: int = direction
            ) -> np.ndarray:
                counters["derivative"] += 1
                return derivative_matvec(selected, vector)

            tangent = augmented_tangent_action(
                apply_generator,
                apply_derivative,
                initial,
                float(instant),
                steps,
                augmented_log_norm_bound=augmented_log_norm_bound(
                    params, direction
                ),
            )
            if direction == 0:
                prediction[row] = tangent.state[:N_IONS].real
            jacobian[row, :, direction] = tangent.tangent[:N_IONS].real
            largest_truncation_estimate = max(
                largest_truncation_estimate,
                float(tangent.truncation_bound_estimate),
            )

    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("matrix-free site prediction is nonfinite")
    return SitePrediction(
        prediction=prediction,
        jacobian=jacobian,
        largest_truncation_estimate=largest_truncation_estimate,
        generator_applications=counters["generator"],
        derivative_applications=counters["derivative"],
    )


def _loss_and_gradient(
    parameters: np.ndarray,
    times: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
    initial_x: np.ndarray,
    *,
    coupling_fixed: float | None = None,
) -> tuple[float, np.ndarray]:
    if coupling_fixed is None:
        full = np.asarray(parameters, dtype=float)
    else:
        full = np.array([parameters[0], parameters[1], coupling_fixed])
    result = predict_and_jacobian(full, times, initial_x)
    residual = (result.prediction - observed) / scale
    loss = 0.5 * float(np.mean(residual**2))
    gradient_full = np.einsum(
        "ti,tij->j", residual / scale, result.jacobian
    ) / residual.size
    if coupling_fixed is None:
        return loss, gradient_full
    return loss, gradient_full[:2]


def fit_site_resolved_dynamics() -> dict:
    """Fit coupled and independent-site models with a held-out time split."""

    times, observed, standard_error = load_neel_site_imbalance()
    initial = observed[0]
    train = np.arange(1, 8)
    test = np.arange(8, times.size)
    scale = np.maximum(standard_error, 0.02)

    baseline = minimize(
        lambda value: _loss_and_gradient(
            value,
            times[train],
            observed[train],
            scale[train],
            initial,
            coupling_fixed=0.0,
        ),
        np.array([0.72, 0.28]),
        jac=True,
        method="L-BFGS-B",
        bounds=((0.0, 2.0), (0.0, 2.0)),
        options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 120},
    )
    coupled = minimize(
        lambda value: _loss_and_gradient(
            value,
            times[train],
            observed[train],
            scale[train],
            initial,
        ),
        np.array([baseline.x[0], baseline.x[1], 0.03]),
        jac=True,
        method="L-BFGS-B",
        bounds=((0.0, 2.0), (0.0, 2.0), (0.0, 1.0)),
        options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 160},
    )

    baseline_full = np.array([baseline.x[0], baseline.x[1], 0.0])
    baseline_prediction = predict_and_jacobian(
        baseline_full, times, initial
    ).prediction
    coupled_result = predict_and_jacobian(coupled.x, times, initial)
    coupled_prediction = coupled_result.prediction

    def rmse(prediction: np.ndarray, indices: np.ndarray) -> float:
        return float(
            np.sqrt(np.mean((prediction[indices] - observed[indices]) ** 2))
        )

    return {
        "source_doi": "10.5281/zenodo.17750419",
        "source_license": "CC-BY-4.0",
        "records": int(observed.size),
        "ions": int(observed.shape[1]),
        "time_points": int(observed.shape[0]),
        "train_time_points": int(train.size),
        "test_time_points": int(test.size),
        "matrix_free": True,
        "dense_generator_materialized": False,
        "state_dimension": STATE_DIMENSION,
        "krylov_dimension": 28,
        "baseline_parameters": baseline_full.tolist(),
        "coupled_parameters": coupled.x.tolist(),
        "baseline_train_rmse": rmse(baseline_prediction, train),
        "baseline_test_rmse": rmse(baseline_prediction, test),
        "coupled_train_rmse": rmse(coupled_prediction, train),
        "coupled_test_rmse": rmse(coupled_prediction, test),
        "baseline_converged": bool(baseline.success),
        "coupled_converged": bool(coupled.success),
        "largest_truncation_bound_estimate": (
            coupled_result.largest_truncation_estimate
        ),
        "generator_applications_per_full_evaluation": (
            coupled_result.generator_applications
        ),
        "derivative_applications_per_full_evaluation": (
            coupled_result.derivative_applications
        ),
        "dense_augmented_entries_per_parameter": (2 * STATE_DIMENSION) ** 2,
        "model_scope": (
            "site-resolved effective coupled oscillator; not a microscopic "
            "51-ion Lindbladian"
        ),
    }


def _dense_generator(parameters: np.ndarray) -> np.ndarray:
    eye = np.eye(STATE_DIMENSION)
    return np.column_stack(
        [generator_matvec(parameters, eye[:, column]) for column in range(STATE_DIMENSION)]
    )


def _dense_direction(direction: int) -> np.ndarray:
    eye = np.eye(STATE_DIMENSION)
    return np.column_stack(
        [derivative_matvec(direction, eye[:, column]) for column in range(STATE_DIMENSION)]
    )


def dense_reference(
    parameters: np.ndarray, times: np.ndarray, initial_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Dense block-Fréchet reference used only for validation and timing."""

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        matrix = _dense_generator(parameters)
        initial = np.concatenate((initial_x, np.zeros(N_IONS)))
        prediction = np.empty((len(times), N_IONS))
        jacobian = np.empty((len(times), N_IONS, 3))
        for row, instant in enumerate(times):
            for direction in range(3):
                derivative = _dense_direction(direction)
                zeros = np.zeros_like(matrix)
                augmented = np.block([[matrix, derivative], [zeros, matrix]])
                start = np.concatenate((np.zeros_like(initial), initial))
                action = expm(float(instant) * augmented) @ start
                if direction == 0:
                    prediction[row] = action[STATE_DIMENSION:][:N_IONS]
                jacobian[row, :, direction] = action[:STATE_DIMENSION][:N_IONS]
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("dense site reference is nonfinite")
    return prediction, jacobian


def benchmark_site_gradient(parameters: np.ndarray, initial_x: np.ndarray) -> dict:
    """Compare matrix-free and dense reference kernels at three real times."""

    selected_times = np.array([2.0, 6.0, 10.0])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        start = perf_counter()
        matrix_free = predict_and_jacobian(parameters, selected_times, initial_x)
        matrix_free_seconds = perf_counter() - start
        start = perf_counter()
        dense_prediction, dense_jacobian = dense_reference(
            parameters, selected_times, initial_x
        )
        dense_seconds = perf_counter() - start
    prediction_error = float(
        np.linalg.norm(matrix_free.prediction - dense_prediction)
        / np.linalg.norm(dense_prediction)
    )
    jacobian_error = float(
        np.linalg.norm(matrix_free.jacobian - dense_jacobian)
        / np.linalg.norm(dense_jacobian)
    )
    if not all(
        np.isfinite(value)
        for value in (
            matrix_free_seconds,
            dense_seconds,
            prediction_error,
            jacobian_error,
        )
    ):
        raise FloatingPointError("site-gradient benchmark is nonfinite")
    return {
        "times_ms": selected_times.tolist(),
        "matrix_free_seconds": matrix_free_seconds,
        "dense_seconds": dense_seconds,
        "dense_over_matrix_free_time": dense_seconds / matrix_free_seconds,
        "prediction_relative_error": prediction_error,
        "jacobian_relative_error": jacobian_error,
    }
