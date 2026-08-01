# Data provenance

**Dataset:** Public 51-ion Néel-dynamics measurements and metadata from Zenodo record 17750419.

**Original source:** https://zenodo.org/records/17750419

**Split:** A site-held-out coupled model is evaluated separately from the fitted local model.

**Integrity:** The reconstruction script downloads the original archive and verifies source and derived hashes.

Run from the repository root:

```bash
python -m pip install -e code
python code/scripts/download_data.py
```

Downloaded third-party files remain governed by the source terms documented in
`THIRD_PARTY.md`. When redistribution is not explicit, the package fetches
the data into an external cache instead of committing the source bytes.
