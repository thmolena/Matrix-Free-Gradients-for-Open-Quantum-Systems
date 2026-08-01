import numpy as np

from krylov_open_quantum import (
    certify_negative_approximate_gradient,
    least_squares_gradient_component,
)


def test_gradient_component_bound_covers_adversarial_perturbations():
    rng = np.random.default_rng(731)
    residual_hat = rng.normal(size=9) + 1j * rng.normal(size=9)
    tangent_hat = rng.normal(size=9) + 1j * rng.normal(size=9)
    eps_r = 0.07
    eps_z = 0.04
    estimate = least_squares_gradient_component(
        residual_hat,
        tangent_hat,
        residual_error_bound=eps_r,
        tangent_error_bound=eps_z,
    )
    for _ in range(500):
        dr = rng.normal(size=9) + 1j * rng.normal(size=9)
        dz = rng.normal(size=9) + 1j * rng.normal(size=9)
        dr *= eps_r * rng.random() / np.linalg.norm(dr)
        dz *= eps_z * rng.random() / np.linalg.norm(dz)
        exact = float(np.real(np.vdot(residual_hat + dr, tangent_hat + dz)))
        assert abs(exact - estimate.value) <= estimate.error_bound * (1 + 1e-13)


def test_descent_decision_is_strict_and_covers_worst_case_gradient():
    gradient_hat = np.array([3.0, 4.0])
    bounds = np.array([0.6, 0.8])
    decision = certify_negative_approximate_gradient(gradient_hat, bounds)
    assert decision.certified
    # The component-wise worst admissible perturbation aligned against g_hat.
    exact_gradient = gradient_hat - bounds
    directional_derivative = exact_gradient @ (-gradient_hat)
    assert directional_derivative <= decision.directional_derivative_upper_bound
    assert decision.directional_derivative_upper_bound < 0.0

    boundary = certify_negative_approximate_gradient(
        np.array([1.0]), np.array([1.0])
    )
    assert not boundary.certified
    assert boundary.directional_derivative_upper_bound == 0.0
