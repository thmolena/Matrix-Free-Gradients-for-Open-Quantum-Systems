"""Determinism and coverage checks for the figure-data builders.

These exercise the numbers behind the manuscript figures without importing
matplotlib, so they run in the base test environment.  Plot rendering itself is
covered by ``build_all`` under the optional ``figures`` extra.
"""

import numpy as np

from krylov_open_quantum.figures import field_curve, tangent_convergence_sweep


def test_tangent_sweep_shows_truncation_then_roundoff_regimes():
    data = tangent_convergence_sweep()
    rows = data["rows"]
    assert rows[0]["m"] == 2 and rows[-1]["m"] == 24
    assert all(row["observed_coverage"] for row in rows[:5])
    assert any(not row["observed_coverage"] for row in rows[5:])
    assert rows[4]["joint_error"] < 1.0e-11


def test_tangent_sweep_matches_independent_reference_fixture():
    rows = tangent_convergence_sweep()["rows"]
    row = {item["m"]: item for item in rows}
    np.testing.assert_allclose(row[4]["joint_error"], 1.161475e-3, rtol=1e-5)
    np.testing.assert_allclose(
        row[8]["truncation_bound_estimate"], 5.464084e-9, rtol=1e-5
    )


def test_field_curve_reports_the_locked_fit():
    data = field_curve()
    np.testing.assert_allclose(data["omega"], 0.748438, rtol=1e-4)
    np.testing.assert_allclose(data["gamma"], 0.259592, rtol=1e-4)
    assert data["rmse"] < 0.08
