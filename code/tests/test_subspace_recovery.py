import numpy as np

from krylov_open_quantum.recovery_study import run_study
from krylov_open_quantum.subspace_recovery import ResidualSubspace, contiguous_masks, periodic_masks


def test_exact_subspace_recovery_without_noise():
    coordinate = np.linspace(-1.0, 1.0, 12)
    basis = np.column_stack((coordinate, coordinate**2 - np.mean(coordinate**2)))
    profiles = np.vstack([1.0 + basis @ np.array([a, 0.2 - a]) for a in np.linspace(-0.4, 0.4, 6)])
    model = ResidualSubspace.fit(profiles, 2)
    target = 0.3 + model.basis @ np.array([0.8, -0.2])
    observed = np.ones(12, dtype=bool)
    observed[[2, 7]] = False
    recovered = model.recover(target, observed, np.full(12, 0.1))
    assert np.allclose(recovered, target, atol=1.0e-12)


def test_mask_contract():
    assert all(mask.sum() in (10, 11) for _, mask in periodic_masks())
    assert all(mask.sum() == 10 for _, mask in contiguous_masks())


def test_locked_protocol_improves_pooled_confirmation():
    result = run_study(repeats=3)
    pooled = result["confirmation"]["pooled"]
    assert result["protocol"]["selected_rank"] == 4
    assert pooled["residual_subspace_rmse"] < pooled["observed_mean_rmse"]
    assert pooled["paired_cases"] == 30
