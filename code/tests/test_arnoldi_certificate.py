import numpy as np
from scipy.linalg import expm

from krylov_open_quantum import arnoldi, expm_action, logarithmic_norm


def test_nonnormal_exponential_error_is_covered():
    rng = np.random.default_rng(41)
    matrix = 0.08 * rng.normal(size=(15, 15)) - 1.2 * np.eye(15)
    initial = rng.normal(size=15)
    time = 0.7
    action = expm_action(
        lambda vector: matrix @ vector,
        initial,
        time,
        7,
        log_norm_bound=logarithmic_norm(matrix),
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        reference = expm(time * matrix) @ initial
    assert np.all(np.isfinite(reference))
    error = np.linalg.norm(action.value - reference)
    assert error <= action.truncation_bound_estimate
    assert action.basis_dimension == 7


def test_full_dimension_breakdown_has_zero_residual():
    matrix = np.array([[-0.3, -0.8], [0.8, -0.3]])
    initial = np.array([1.0, 0.0])
    action = expm_action(
        lambda vector: matrix @ vector,
        initial,
        1.3,
        2,
        log_norm_bound=logarithmic_norm(matrix),
    )
    np.testing.assert_allclose(
        action.value, expm(1.3 * matrix) @ initial, rtol=1e-13, atol=1e-13
    )
    assert action.truncation_bound_estimate < 1e-12
    assert action.roundoff_scale > 0.0


def test_tolerance_stop_retains_nonzero_arnoldi_defect():
    """Regression: near breakdown must not be reported as exact breakdown."""

    delta = 1.0e-8
    matrix = np.diag([1.0, 1.0 + delta])
    initial = np.array([1.0, 1.0]) / np.sqrt(2.0)
    decomposition = arnoldi(
        lambda vector: matrix @ vector,
        initial,
        steps=2,
        breakdown_tolerance=1.0e-6,
    )
    assert decomposition.dimension == 1
    assert 0.0 < decomposition.residual_norm < 1.0e-6

    action = expm_action(
        lambda vector: matrix @ vector,
        initial,
        time=0.8,
        steps=2,
        log_norm_bound=logarithmic_norm(matrix),
        breakdown_tolerance=1.0e-6,
    )
    error = np.linalg.norm(action.value - expm(0.8 * matrix) @ initial)
    assert action.residual_norm > 0.0
    assert action.truncation_bound_estimate > 0.0
    assert error <= action.truncation_bound_estimate
