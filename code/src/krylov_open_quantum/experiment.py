"""Reproduce dense-validation and experimental-data results on a CPU."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from .arnoldi import dense_log_norm_bound, expm_action
from .field_data import load_neel_imbalance
from .models import augmented_matrix, damped_rotation, damped_rotation_derivatives
from .site_learning import fit_site_resolved_dynamics
from .tangent import augmented_tangent_action


def _predict_and_gradient(parameters: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = damped_rotation(parameters)
    derivatives = damped_rotation_derivatives()
    initial = np.array([1.0, 0.0])
    prediction = np.empty(time.size)
    jacobian = np.empty((time.size, 2))
    largest_truncation_estimate = 0.0
    for row, instant in enumerate(time):
        for column, direction in enumerate(derivatives):
            augmented = augmented_matrix(matrix, direction)
            tangent = augmented_tangent_action(
                lambda vector, a=matrix: a @ vector,
                lambda vector, e=direction: e @ vector,
                initial,
                float(instant),
                steps=4,
                augmented_log_norm_bound=dense_log_norm_bound(augmented),
            )
            prediction[row] = tangent.state[0].real
            jacobian[row, column] = tangent.tangent[0].real
            largest_truncation_estimate = max(
                largest_truncation_estimate,
                tangent.truncation_bound_estimate,
            )
    return prediction, jacobian, largest_truncation_estimate


def fit_experimental_imbalance() -> dict:
    """Fit an effective damped mode to the public 51-ion data summary."""

    time, observed, standard_error = load_neel_imbalance()
    amplitude = float(observed[0])
    scaled = observed / amplitude
    weights = 1.0 / np.maximum(standard_error / amplitude, 5.0e-3)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        prediction, jacobian, _ = _predict_and_gradient(parameters, time)
        residual = (prediction - scaled) * weights
        loss = 0.5 * float(np.mean(residual**2))
        gradient = jacobian.T @ (residual * weights) / residual.size
        return loss, gradient

    fitted = minimize(
        lambda x: objective(x),
        np.array([0.72, 0.28]),
        jac=True,
        method="L-BFGS-B",
        bounds=((0.0, 2.0), (0.0, 2.0)),
        options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 250},
    )
    prediction, _, truncation_estimate = _predict_and_gradient(fitted.x, time)
    prediction *= amplitude
    rmse = float(np.sqrt(np.mean((prediction - observed) ** 2)))
    return {
        "source_doi": "10.5281/zenodo.17750419",
        "source_license": "CC-BY-4.0",
        "ions": 51,
        "shots_per_time": 1000,
        "time_points": int(time.size),
        "fitted_angular_frequency_per_ms": float(fitted.x[0]),
        "fitted_damping_rate_per_ms": float(fitted.x[1]),
        "rmse_staggered_imbalance": rmse,
        "optimizer_converged": bool(fitted.success),
        "largest_full_dimension_tangent_truncation_estimate": float(
            truncation_estimate
        ),
        "model_scope": "effective two-coordinate damped mode, not a microscopic 51-ion identification",
    }


def validate_truncation_estimator(
    seed: int = 260724,
    *,
    size: int = 18,
    time: float = 0.8,
    steps: tuple[int, ...] = (4, 6, 8),
) -> dict:
    """Validate the bound for a nonnormal stable matrix and several dimensions."""

    if size < 2:
        raise ValueError("size must be at least two")
    if time < 0.0:
        raise ValueError("time must be nonnegative")
    if not steps or any(step < 1 or step > size for step in steps):
        raise ValueError("steps must contain integers between one and size")
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(size, size))
    matrix = 0.09 * raw - 1.5 * np.eye(size)
    initial = rng.normal(size=size)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        exact = expm(time * matrix) @ initial
    if not np.all(np.isfinite(exact)):
        raise FloatingPointError("dense validation reference is nonfinite")
    log_bound = dense_log_norm_bound(matrix)
    cases = []
    for dimension in steps:
        action = expm_action(
            lambda vector: matrix @ vector,
            initial,
            time,
            dimension,
            log_norm_bound=log_bound,
        )
        error = float(np.linalg.norm(action.value - exact))
        cases.append(
            {
                "steps": dimension,
                "error_2norm": error,
                "truncation_bound_estimate": action.truncation_bound_estimate,
                "coverage_ratio": action.truncation_bound_estimate
                / max(error, np.finfo(float).tiny),
                "observed_coverage": bool(error <= action.truncation_bound_estimate),
            }
        )
    return {
        "dimension": size,
        "time": time,
        "log_norm_bound": log_bound,
        "all_cases_observed_covered": all(
            case["observed_coverage"] for case in cases
        ),
        "cases": cases,
    }


def validate_action_stress(seed: int = 731, cases: int = 64) -> dict:
    """Dense-reference stress test away from the roundoff floor."""

    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(cases):
        size = 12
        time = float(rng.uniform(0.25, 1.1))
        matrix = (
            rng.uniform(0.04, 0.12) * rng.normal(size=(size, size))
            - rng.uniform(0.4, 1.4) * np.eye(size)
        )
        initial = rng.normal(size=size)
        action = expm_action(
            lambda vector, a=matrix: a @ vector,
            initial,
            time,
            4,
            log_norm_bound=dense_log_norm_bound(matrix),
        )
        reference = expm(time * matrix) @ initial
        error = float(np.linalg.norm(action.value - reference))
        ratios.append(
            action.truncation_bound_estimate
            / max(error, np.finfo(float).tiny)
        )
    return {
        "seed": seed,
        "cases": cases,
        "dimension": 12,
        "krylov_dimension": 4,
        "observed_failures": int(sum(ratio < 1.0 for ratio in ratios)),
        "minimum_estimate_to_error_ratio": float(min(ratios)),
        "maximum_estimate_to_error_ratio": float(max(ratios)),
        "reference": "scipy.linalg.expm",
    }


def validate_tangent_stress(seed: int = 732, cases: int = 32) -> dict:
    """Independent Fréchet-reference stress test for the augmented tangent."""

    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(cases):
        size = 8
        time = float(rng.uniform(0.25, 0.9))
        matrix = (
            rng.uniform(0.04, 0.1) * rng.normal(size=(size, size))
            - rng.uniform(0.5, 1.2) * np.eye(size)
        )
        direction = 0.05 * rng.normal(size=(size, size))
        initial = rng.normal(size=size)
        block = np.block(
            [[matrix, direction], [np.zeros_like(matrix), matrix]]
        )
        action = augmented_tangent_action(
            lambda vector, a=matrix: a @ vector,
            lambda vector, e=direction: e @ vector,
            initial,
            time,
            4,
            augmented_log_norm_bound=dense_log_norm_bound(block),
        )
        reference_state, reference_frechet = expm_frechet(
            time * matrix, time * direction, compute_expm=True
        )
        state_error = np.linalg.norm(action.state - reference_state @ initial)
        tangent_error = np.linalg.norm(
            action.tangent - reference_frechet @ initial
        )
        error = float(np.hypot(state_error, tangent_error))
        ratios.append(
            action.truncation_bound_estimate
            / max(error, np.finfo(float).tiny)
        )
    return {
        "seed": seed,
        "cases": cases,
        "state_dimension": 8,
        "augmented_dimension": 16,
        "krylov_dimension": 4,
        "observed_failures": int(sum(ratio < 1.0 for ratio in ratios)),
        "minimum_estimate_to_error_ratio": float(min(ratios)),
        "maximum_estimate_to_error_ratio": float(max(ratios)),
        "reference": "scipy.linalg.expm_frechet",
    }


def validate_near_breakdown() -> dict:
    """Adversarial regression for tolerance-triggered Arnoldi termination."""

    delta = 1.0e-8
    matrix = np.diag([1.0, 1.0 + delta])
    initial = np.array([1.0, 1.0]) / np.sqrt(2.0)
    time = 0.8
    action = expm_action(
        lambda vector: matrix @ vector,
        initial,
        time,
        2,
        log_norm_bound=dense_log_norm_bound(matrix),
        breakdown_tolerance=1.0e-6,
    )
    error = float(np.linalg.norm(action.value - expm(time * matrix) @ initial))
    return {
        "matrix_eigenvalue_gap": delta,
        "breakdown_tolerance": 1.0e-6,
        "basis_dimension": action.basis_dimension,
        "retained_residual_norm": action.residual_norm,
        "error_2norm": error,
        "truncation_bound_estimate": action.truncation_bound_estimate,
        "observed_coverage": bool(error <= action.truncation_bound_estimate),
    }


def generate_results(
    *,
    validation_seed: int = 260724,
    validation_size: int = 18,
    validation_time: float = 0.8,
    validation_steps: tuple[int, ...] = (4, 6, 8),
) -> dict:
    return {
        "schema_version": 2,
        "truncation_estimator_validation": validate_truncation_estimator(
            validation_seed,
            size=validation_size,
            time=validation_time,
            steps=validation_steps,
        ),
        "stress_validation": {
            "generic_actions": validate_action_stress(),
            "augmented_tangents": validate_tangent_stress(),
            "near_breakdown_regression": validate_near_breakdown(),
        },
        "experimental_fit": fit_experimental_imbalance(),
        "site_resolved_learning": fit_site_resolved_dynamics(),
        "claims": {
            "matrix_free_algorithm": True,
            "cpu_only": True,
            "experimental_data_used": True,
            "site_resolved_records_used": 561,
            "microscopic_51ion_model_claimed": False,
            "theorem_scope": (
                "exact-arithmetic Euclidean Arnoldi truncation bound given a "
                "valid logarithmic-norm upper bound"
            ),
            "floating_point_certificate_claimed": False,
        },
    }


def canonical_bytes(results: dict) -> bytes:
    def normalize(value):
        if isinstance(value, float):
            return float(f"{value:.5g}")
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return (
        json.dumps(normalize(results), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def locked_results_bytes() -> bytes:
    path = resources.files("krylov_open_quantum").joinpath("results/results.json")
    return path.read_bytes()


def reproduce(
    output: Path | None = None,
    verify: bool = False,
    *,
    validation_seed: int = 260724,
    validation_size: int = 18,
    validation_time: float = 0.8,
    validation_steps: tuple[int, ...] = (4, 6, 8),
) -> dict:
    results = generate_results(
        validation_seed=validation_seed,
        validation_size=validation_size,
        validation_time=validation_time,
        validation_steps=validation_steps,
    )
    rendered = canonical_bytes(results)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(rendered)
    if verify:
        locked = locked_results_bytes()
        if rendered != locked:
            raise RuntimeError(
                "recomputed results differ from the locked package artifact: "
                f"computed={hashlib.sha256(rendered).hexdigest()} "
                f"locked={hashlib.sha256(locked).hexdigest()}"
            )
    return results
