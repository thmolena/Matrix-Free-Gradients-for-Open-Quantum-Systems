import numpy as np
from scipy.linalg import expm

from krylov_open_quantum import (
    directional_derivative,
    lanczos,
    lift_projected_sensitivity,
    projected_quadratic,
    projected_sensitivity,
)


def _symmetric_random_matrix(rng: np.random.Generator, size: int) -> np.ndarray:
    raw = rng.normal(size=(size, size))
    return 0.5 * (raw + raw.T)


def test_full_krylov_value_and_directional_gradient_match_dense_reference():
    rng = np.random.default_rng(7)
    size = 7
    matrix = _symmetric_random_matrix(rng, size)
    direction = _symmetric_random_matrix(rng, size)
    initial = rng.normal(size=size)

    decomposition = lanczos(
        lambda vector: matrix @ vector,
        initial,
        size,
        reorthogonalize=True,
    )

    value = projected_quadratic(decomposition, np.exp)
    dense_value = float(initial @ expm(matrix) @ initial)
    np.testing.assert_allclose(value, dense_value, rtol=2e-11, atol=2e-11)

    small_sensitivity = projected_sensitivity(
        decomposition,
        np.exp,
        np.exp,
    )
    matrix_gradient = lift_projected_sensitivity(
        decomposition,
        small_sensitivity,
    )
    gradient = directional_derivative(matrix_gradient, direction)

    epsilon = 1e-6
    plus = float(initial @ expm(matrix + epsilon * direction) @ initial)
    minus = float(initial @ expm(matrix - epsilon * direction) @ initial)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(
        gradient,
        finite_difference,
        rtol=2e-7,
        atol=2e-7,
    )


def test_truncated_lanczos_returns_hermitian_projection_and_residual():
    rng = np.random.default_rng(11)
    matrix = _symmetric_random_matrix(rng, 12)
    initial = rng.normal(size=12)

    decomposition = lanczos(
        lambda vector: matrix @ vector,
        initial,
        4,
    )

    assert decomposition.dimension == 4
    assert decomposition.basis.shape == (12, 4)
    assert decomposition.residual_coefficient > 0
    np.testing.assert_allclose(
        decomposition.projected,
        decomposition.projected.T,
        rtol=0,
        atol=1e-14,
    )
