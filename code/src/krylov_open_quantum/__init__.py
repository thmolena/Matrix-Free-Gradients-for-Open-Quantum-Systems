"""Auditable Krylov actions and gradients for linear open dynamics."""

from .arnoldi import (
    ArnoldiDecomposition,
    ExponentialAction,
    arnoldi,
    expm_action,
    logarithmic_norm,
)
from .field_data import load_neel_imbalance, load_neel_site_imbalance
from .decisions import (
    DescentDecision,
    GradientComponentEstimate,
    certify_negative_approximate_gradient,
    least_squares_gradient_component,
)
from .hermitian import LanczosDecomposition, lanczos
from .models import damped_rotation, damped_rotation_derivatives
from .projected import (
    directional_derivative,
    lift_projected_sensitivity,
    projected_quadratic,
    projected_sensitivity,
)
from .site_learning import (
    benchmark_site_gradient,
    fit_site_resolved_dynamics,
    predict_and_jacobian,
)
from .tangent import TangentAction, augmented_tangent_action

__all__ = [
    "ArnoldiDecomposition",
    "DescentDecision",
    "ExponentialAction",
    "GradientComponentEstimate",
    "LanczosDecomposition",
    "TangentAction",
    "arnoldi",
    "augmented_tangent_action",
    "benchmark_site_gradient",
    "certify_negative_approximate_gradient",
    "damped_rotation",
    "damped_rotation_derivatives",
    "directional_derivative",
    "expm_action",
    "fit_site_resolved_dynamics",
    "lanczos",
    "least_squares_gradient_component",
    "load_neel_imbalance",
    "load_neel_site_imbalance",
    "logarithmic_norm",
    "lift_projected_sensitivity",
    "projected_quadratic",
    "projected_sensitivity",
    "predict_and_jacobian",
]

__version__ = "1.1.0"
