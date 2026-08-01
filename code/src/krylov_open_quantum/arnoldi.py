"""Matrix-free Arnoldi exponential actions and defect-based error estimates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from math import factorial

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

Array = NDArray[np.complex128]
MatVec = Callable[[Array], Array]


def _finite_public_result(function):
    """Contain spurious Accelerate flags, then reject any nonfinite result."""

    @wraps(function)
    def guarded(*args, **kwargs):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            result = function(*args, **kwargs)
        if isinstance(result, ExponentialAction):
            finite = (
                np.all(np.isfinite(result.value))
                and np.isfinite(result.truncation_bound_estimate)
                and np.isfinite(result.roundoff_scale)
                and np.isfinite(result.residual_norm)
            )
            if not finite:
                raise FloatingPointError("exponential action is nonfinite")
        return result

    return guarded


@dataclass(frozen=True)
class ArnoldiDecomposition:
    """Finite Arnoldi relation ``A V = V H + r e_m^T``."""

    basis: Array
    projected: Array
    initial_norm: float
    residual_vector: Array

    @property
    def dimension(self) -> int:
        return int(self.projected.shape[0])

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residual_vector))


@dataclass(frozen=True)
class ExponentialAction:
    """Projected action and diagnostics for the exact-arithmetic theorem.

    ``truncation_bound_estimate`` evaluates the rigorous defect bound derived
    for an exact Arnoldi relation.  Ordinary NumPy/SciPy arithmetic is not
    interval arithmetic, so the returned floating-point number is deliberately
    not labelled a finite-precision certificate.  ``roundoff_scale`` is a
    diagnostic scale, not a proved roundoff bound.
    """

    value: Array
    truncation_bound_estimate: float
    roundoff_scale: float
    residual_norm: float
    basis_dimension: int


def arnoldi(
    matvec: MatVec,
    initial: NDArray[np.generic],
    steps: int,
    *,
    breakdown_tolerance: float = 1.0e-14,
) -> ArnoldiDecomposition:
    """Build an Arnoldi decomposition using two-pass orthogonalization."""

    vector = np.asarray(initial, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("initial must be a nonempty vector")
    if steps < 1:
        raise ValueError("steps must be positive")
    beta = float(np.linalg.norm(vector))
    if beta == 0.0:
        raise ValueError("initial must be nonzero")

    maximum = min(int(steps), vector.size)
    basis = np.zeros((vector.size, maximum + 1), dtype=np.complex128)
    hessenberg = np.zeros((maximum + 1, maximum), dtype=np.complex128)
    basis[:, 0] = vector / beta
    achieved = maximum
    residual = np.zeros_like(vector)

    for column in range(maximum):
        work = np.asarray(matvec(basis[:, column]), dtype=np.complex128)
        if work.shape != vector.shape:
            raise ValueError("matvec returned a vector with the wrong shape")
        for _ in range(2):
            coefficients = basis[:, : column + 1].conj().T @ work
            hessenberg[: column + 1, column] += coefficients
            work -= basis[:, : column + 1] @ coefficients
        next_norm = float(np.linalg.norm(work))
        hessenberg[column + 1, column] = next_norm
        if next_norm <= breakdown_tolerance:
            achieved = column + 1
            # A tolerance-triggered stop is not an exact happy breakdown.
            # Retaining ``work`` preserves the computed Arnoldi defect and
            # prevents a false zero estimate when 0 < next_norm <= tolerance.
            residual = work
            break
        residual = work
        if column + 1 < maximum:
            basis[:, column + 1] = work / next_norm

    return ArnoldiDecomposition(
        basis=basis[:, :achieved],
        projected=hessenberg[:achieved, :achieved],
        initial_norm=beta,
        residual_vector=residual,
    )


def logarithmic_norm(matrix: NDArray[np.generic]) -> float:
    """Return the Euclidean logarithmic norm ``lambda_max((A+A*)/2)``."""

    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    hermitian_part = 0.5 * (array + array.conj().T)
    return float(np.linalg.eigvalsh(hermitian_part)[-1])


def _defect_integral_upper(
    projected: Array,
    initial_norm: float,
    time: float,
    full_log_norm: float,
    *,
    subintervals: int = 256,
    taylor_order: int = 8,
) -> float:
    """Upper-bound the scalar Arnoldi-defect integral.

    On each time cell a Taylor majorant bounds the final coordinate of
    ``exp(sH)e1``. This retains the decisive small final-coordinate factor that
    is lost in the much coarser bound ``||exp(sH)||``.
    """

    if time == 0.0:
        return 0.0
    width = time / subintervals
    half = 0.5 * width
    dimension = projected.shape[0]
    coordinate = np.zeros(dimension, dtype=np.complex128)
    coordinate[0] = initial_norm
    midpoint_coordinate = expm(half * projected) @ coordinate
    step = expm(width * projected)
    matrix_norm = float(np.linalg.norm(projected, 2))
    projected_log_norm = logarithmic_norm(projected)
    factorial_accumulator = 1.0
    total = 0.0
    for index in range(subintervals):
        midpoint = (index + 0.5) * width
        derivative = midpoint_coordinate
        scalar_upper = abs(derivative[-1])
        power = 1.0
        for order in range(1, taylor_order + 1):
            derivative = projected @ derivative
            power *= half
            factorial_accumulator *= order
            scalar_upper += abs(derivative[-1]) * power / factorial_accumulator
        factorial_accumulator = 1.0
        remainder_power = half ** (taylor_order + 1)
        remainder_factorial = float(factorial(taylor_order + 1))
        local_endpoint = (
            midpoint + half if projected_log_norm >= 0.0 else midpoint - half
        )
        remainder = (
            initial_norm
            * matrix_norm ** (taylor_order + 1)
            * np.exp(local_endpoint * projected_log_norm)
            * remainder_power
            / remainder_factorial
        )
        scalar_upper += float(remainder)
        semigroup_endpoint = (
            midpoint - half if full_log_norm >= 0.0 else midpoint + half
        )
        semigroup_upper = np.exp((time - semigroup_endpoint) * full_log_norm)
        total += width * semigroup_upper * scalar_upper
        midpoint_coordinate = step @ midpoint_coordinate
    return float(total)


@_finite_public_result
def expm_action(
    matvec: MatVec,
    initial: NDArray[np.generic],
    time: float,
    steps: int,
    *,
    log_norm_bound: float,
    breakdown_tolerance: float = 1.0e-14,
) -> ExponentialAction:
    """Approximate ``exp(time*A) initial`` and estimate truncation error.

    ``log_norm_bound`` must upper-bound the Euclidean logarithmic norm of the
    full operator. In exact arithmetic the returned defect expression is a
    rigorous upper bound by ``||exp(t A)||_2 <= exp(t log_norm_bound)``.  The
    implementation uses ordinary floating-point linear algebra and therefore
    does not claim a rigorous enclosure of roundoff or callback error.
    """

    if time < 0.0:
        raise ValueError("time must be nonnegative")
    decomposition = arnoldi(
        matvec,
        initial,
        steps,
        breakdown_tolerance=breakdown_tolerance,
    )
    e1 = np.zeros(decomposition.dimension, dtype=np.complex128)
    e1[0] = decomposition.initial_norm
    coordinates = expm(time * decomposition.projected) @ e1
    value = decomposition.basis @ coordinates
    truncation_bound = decomposition.residual_norm * _defect_integral_upper(
        decomposition.projected,
        decomposition.initial_norm,
        float(time),
        float(log_norm_bound),
    )
    roundoff_scale = (
        np.finfo(float).eps
        * decomposition.initial_norm
        * np.exp(time * max(float(log_norm_bound), 0.0))
    )
    return ExponentialAction(
        value=value,
        truncation_bound_estimate=float(max(0.0, truncation_bound)),
        roundoff_scale=float(roundoff_scale),
        residual_norm=decomposition.residual_norm,
        basis_dimension=decomposition.dimension,
    )


def dense_log_norm_bound(matrix: NDArray[np.generic]) -> float:
    """Convenience helper for validation-sized dense operators."""

    return logarithmic_norm(matrix)
