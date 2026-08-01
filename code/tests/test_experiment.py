from krylov_open_quantum.experiment import (
    fit_experimental_imbalance,
    validate_action_stress,
    validate_near_breakdown,
    validate_tangent_stress,
    validate_truncation_estimator,
)


def test_all_locked_truncation_cases_have_observed_coverage():
    validation = validate_truncation_estimator()
    assert validation["all_cases_observed_covered"]
    assert len(validation["cases"]) == 3


def test_validation_configuration_is_public():
    validation = validate_truncation_estimator(
        17, size=10, time=0.25, steps=(3, 5)
    )
    assert validation["dimension"] == 10
    assert validation["time"] == 0.25
    assert [case["steps"] for case in validation["cases"]] == [3, 5]


def test_dense_reference_stress_fixtures_have_observed_coverage():
    action = validate_action_stress()
    tangent = validate_tangent_stress()
    assert action["observed_failures"] == 0
    assert tangent["observed_failures"] == 0
    assert action["minimum_estimate_to_error_ratio"] > 1.0
    assert tangent["minimum_estimate_to_error_ratio"] > 1.0


def test_near_breakdown_regression_is_nonzero_and_covered():
    result = validate_near_breakdown()
    assert result["basis_dimension"] == 1
    assert result["retained_residual_norm"] > 0.0
    assert result["truncation_bound_estimate"] > 0.0
    assert result["observed_coverage"]


def test_experimental_fit_is_scoped_and_converges():
    result = fit_experimental_imbalance()
    assert result["optimizer_converged"]
    assert result["ions"] == 51
    assert result["shots_per_time"] == 1000
    assert result["rmse_staggered_imbalance"] < 0.08
    assert "not a microscopic" in result["model_scope"]
