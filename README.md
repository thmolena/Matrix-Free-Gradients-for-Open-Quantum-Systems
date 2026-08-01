# Variance-Weighted Residual Subspaces for Missing Ion-Resolved Readouts

This repository contains the manuscript, attributed 51-ion measurements, and
the complete development/confirmation replay. The method centers each complete
development profile, retains four spatial residual directions selected by
leave-one-time-out reconstruction, and solves a variance-weighted least-squares
problem on the sites available in an incomplete profile.

The confirmation set consists of times 8--10 ms and two registered outage
families. Pooled missing-site RMSE is 0.06270, compared with 0.08435 for the
observed-site mean, 0.12071 for linear interpolation, and 0.12424 for
nearest-site fill. The method wins 25 of 30 paired cases; the five losses are
retained in the paper and result file.

## Reproduction

```bash
python -m pip install "./code[test,figures]"
ion-subspace-study --output code/results/recovery_study.json --verify
ion-subspace-figures
python -m pytest -q code/tests
```

The packaged measurements derive from Zenodo DOI
`10.5281/zenodo.17750419`, released under CC BY 4.0. The source reconstruction
script and byte checksums remain in `code/scripts/` and `code/data/`.

## Contents

- `main.tex` and `main.pdf`: manuscript source and rendered paper.
- `code/src/krylov_open_quantum/subspace_recovery.py`: reconstruction method.
- `code/src/krylov_open_quantum/recovery_study.py`: locked split and metrics.
- `code/src/krylov_open_quantum/recovery_figures.py`: four evidence figures.
- `code/results/recovery_study.json`: case-level results and timing records.
- `code/tests/`: exact-recovery, protocol, source, and numerical tests.

The recovery theorem is deterministic. The measured improvement is restricted
to the stated archive, masks, and confirmation split.
