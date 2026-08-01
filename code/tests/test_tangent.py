import numpy as np
from scipy.linalg import expm, expm_frechet

from krylov_open_quantum import augmented_tangent_action, logarithmic_norm
from krylov_open_quantum.models import augmented_matrix


def test_augmented_tangent_matches_finite_difference():
    matrix = np.array([[-0.27, -0.91], [0.91, -0.27]])
    direction = np.array([[0.0, -1.0], [1.0, 0.0]])
    initial = np.array([1.0, 0.0])
    time = 1.7
    augmented = augmented_matrix(matrix, direction)
    result = augmented_tangent_action(
        lambda vector: matrix @ vector,
        lambda vector: direction @ vector,
        initial,
        time,
        4,
        augmented_log_norm_bound=logarithmic_norm(augmented),
    )
    epsilon = 1e-6
    finite = (
        expm(time * (matrix + epsilon * direction)) @ initial
        - expm(time * (matrix - epsilon * direction)) @ initial
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(result.state, expm(time * matrix) @ initial, atol=1e-12)
    np.testing.assert_allclose(result.tangent, finite, rtol=2e-9, atol=2e-9)
    assert result.truncation_bound_estimate < 1e-11


def test_nonnormal_tangent_matches_independent_expm_frechet_reference():
    rng = np.random.default_rng(20260731)
    matrix = 0.12 * rng.normal(size=(6, 6)) - 0.9 * np.eye(6)
    direction = 0.08 * rng.normal(size=(6, 6))
    initial = rng.normal(size=6)
    time = 0.65
    block = np.block(
        [[matrix, direction], [np.zeros_like(matrix), matrix]]
    )
    result = augmented_tangent_action(
        lambda vector: matrix @ vector,
        lambda vector: direction @ vector,
        initial,
        time,
        8,
        augmented_log_norm_bound=logarithmic_norm(block),
    )
    reference_state, reference_frechet = expm_frechet(
        time * matrix, time * direction, compute_expm=True
    )
    state_error = np.linalg.norm(result.state - reference_state @ initial)
    tangent_error = np.linalg.norm(result.tangent - reference_frechet @ initial)
    joint_error = np.hypot(state_error, tangent_error)
    assert joint_error <= result.truncation_bound_estimate
    np.testing.assert_allclose(
        result.tangent, reference_frechet @ initial, rtol=2e-5, atol=2e-7
    )
