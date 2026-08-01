"""Conditional least-squares gradient and descent certificates.

The functions in this module are algebraic: callers must supply valid bounds
on observation and tangent errors.  A floating-point Arnoldi estimate is not
silently promoted to such a bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GradientComponentEstimate:
    """Approximate real gradient component and a conditional error bound."""

    value: float
    error_bound: float


@dataclass(frozen=True)
class DescentDecision:
    """Decision for the direction opposite an approximate gradient."""

    certified: bool
    gradient_norm: float
    error_norm_bound: float
    directional_derivative_upper_bound: float


def least_squares_gradient_component(
    approximate_residual: NDArray[np.generic],
    approximate_tangent: NDArray[np.generic],
    *,
    residual_error_bound: float,
    tangent_error_bound: float,
) -> GradientComponentEstimate:
    """Bound one component of ``Re(r* z)`` from normwise input bounds.

    If ``||r-r_hat|| <= eps_r`` and ``||z-z_hat|| <= eps_z``, then

    ``|Re(r* z) - Re(r_hat* z_hat)|``
    ``<= ||r_hat|| eps_z + ||z_hat|| eps_r + eps_r eps_z``.
    """

    residual = np.asarray(approximate_residual, dtype=np.complex128)
    tangent = np.asarray(approximate_tangent, dtype=np.complex128)
    if residual.ndim != 1 or tangent.ndim != 1 or residual.shape != tangent.shape:
        raise ValueError("residual and tangent must be vectors of equal shape")
    eps_r = float(residual_error_bound)
    eps_z = float(tangent_error_bound)
    if not np.isfinite(eps_r) or not np.isfinite(eps_z) or eps_r < 0 or eps_z < 0:
        raise ValueError("error bounds must be finite and nonnegative")
    value = float(np.real(np.vdot(residual, tangent)))
    bound = (
        float(np.linalg.norm(residual)) * eps_z
        + float(np.linalg.norm(tangent)) * eps_r
        + eps_r * eps_z
    )
    return GradientComponentEstimate(value=value, error_bound=float(bound))


def certify_negative_approximate_gradient(
    approximate_gradient: NDArray[np.generic],
    component_error_bounds: NDArray[np.generic],
) -> DescentDecision:
    """Certify ``-g_hat`` as a descent direction for every admissible gradient.

    The test is strict: ``||g_hat||_2 > ||delta||_2``.  Equality is not enough
    to guarantee a negative directional derivative.
    """

    gradient = np.asarray(approximate_gradient, dtype=float)
    bounds = np.asarray(component_error_bounds, dtype=float)
    if gradient.ndim != 1 or bounds.ndim != 1 or gradient.shape != bounds.shape:
        raise ValueError("gradient and component bounds must be equal-length vectors")
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(bounds)):
        raise ValueError("gradient and bounds must be finite")
    if np.any(bounds < 0.0):
        raise ValueError("component bounds must be nonnegative")
    gradient_norm = float(np.linalg.norm(gradient))
    error_norm = float(np.linalg.norm(bounds))
    certified = bool(gradient_norm > error_norm)
    directional_upper = -gradient_norm * (gradient_norm - error_norm)
    return DescentDecision(
        certified=certified,
        gradient_norm=gradient_norm,
        error_norm_bound=error_norm,
        directional_derivative_upper_bound=float(directional_upper),
    )
