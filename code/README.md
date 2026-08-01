# Variance-Weighted Residual-Subspace Recovery

The primary study is reproduced with `ion-subspace-study --verify`; figures
are regenerated with `ion-subspace-figures`. The earlier Arnoldi and Fréchet
action modules remain tested numerical components, while the manuscript now
centers on recovery of missing 51-ion readouts from authenticated measurements.

Install and verify the CPU package from the repository root:

```bash
python -m pip install "./code[test,figures]"
krylov-open-quantum-reproduce --verify
python -m pytest -q code/tests
krylov-open-quantum-figures \
  --output code/manuscript_assets/figures \
  --json-output code/results/figure_results.json
```

`augmented_tangent_action` accepts matrix-vector callbacks and returns a state,
one directional tangent, a retained Arnoldi residual, a floating-point
evaluation of the exact-arithmetic truncation bound, and a diagnostic roundoff
scale. It does **not** claim a rigorous finite-precision certificate.

`least_squares_gradient_component` propagates caller-supplied valid state and
tangent error bounds into a gradient-component bound.
`certify_negative_approximate_gradient` then applies the strict descent test
proved in the manuscript.

The packaged CSVs derive under CC BY 4.0 from Zenodo DOI
`10.5281/zenodo.17750419`. Rebuild and verify them from the original archive:

```bash
python code/scripts/fetch_and_build_neel.py
```

Standard wrapper commands remain available:

```bash
python code/scripts/download_data.py
python code/scripts/reproduce.py
python code/scripts/make_figures.py
python code/scripts/validate_release.py
```
