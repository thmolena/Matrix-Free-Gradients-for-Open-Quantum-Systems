import numpy as np

from krylov_open_quantum.field_data import load_neel_site_imbalance
from krylov_open_quantum.site_learning import (
    dense_reference,
    fit_site_resolved_dynamics,
    predict_and_jacobian,
)


def test_site_resolved_archive_summary_has_expected_shape():
    time, observed, standard_error = load_neel_site_imbalance()
    assert time.shape == (11,)
    assert observed.shape == standard_error.shape == (11, 51)
    assert np.all(standard_error > 0.0)


def test_matrix_free_site_tangent_matches_dense_reference():
    time, observed, _ = load_neel_site_imbalance()
    parameters = np.array([0.7366811, 0.2352389, 0.2449596])
    selected = time[[2]]
    matrix_free = predict_and_jacobian(parameters, selected, observed[0])
    dense_prediction, dense_jacobian = dense_reference(
        parameters, selected, observed[0]
    )
    np.testing.assert_allclose(
        matrix_free.prediction, dense_prediction, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        matrix_free.jacobian, dense_jacobian, rtol=1e-8, atol=1e-10
    )
    assert matrix_free.largest_truncation_estimate < 1e-4


def test_site_fit_is_matrix_free_and_honestly_held_out():
    result = fit_site_resolved_dynamics()
    assert result["matrix_free"]
    assert not result["dense_generator_materialized"]
    assert result["records"] == 561
    assert result["train_time_points"] == 7
    assert result["test_time_points"] == 3
    assert result["coupled_converged"]
    assert result["coupled_test_rmse"] > result["baseline_test_rmse"]
