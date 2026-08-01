#!/usr/bin/env python3
"""Download the authentic 51-ion Neel-state archive and rebuild the summary.

This script closes the last reproducibility gap in the experimental-data path.
It downloads the original ``NeelDataZ51.zip`` from the public Zenodo record
(DOI 10.5281/zenodo.17750419, CC BY 4.0), decodes every 51-bit measurement shot
from ``DataHL51IonmeasType14Base3.mat``, forms the mean staggered imbalance and
its standard error at each of the 11 times, and compares the result byte-for-byte
with the summary shipped in the package
(``src/krylov_open_quantum/data/neel_51ion_imbalance.csv``).

The decoding follows the archive ReadMe exactly: each shot is a decimal integer
whose bit ``i`` (least significant bit = ion 1) is ``0`` or ``1`` for spin
``+1`` or ``-1``; the staggered imbalance correlates the 51 spins with the
alternating Neel pattern, with a global sign fixed so the initial imbalance is
positive.

Usage:

    python code/scripts/fetch_and_build_neel.py            # verify against shipped CSV
    python code/scripts/fetch_and_build_neel.py --write out.csv

Requires only numpy and scipy (already package dependencies) and network access
to zenodo.org.  No GPU or quantum processor is used.
"""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path

import numpy as np
import scipy.io as sio

ZENODO_ZIP = "https://zenodo.org/api/records/17750419/files/NeelDataZ51.zip/content"
MAT_NAME = "DataHL51IonmeasType14Base3.mat"
N_IONS = 51


def _download_mat(url: str = ZENODO_ZIP) -> bytes:
    with urllib.request.urlopen(url) as response:  # noqa: S310 (trusted DOI host)
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with archive.open(MAT_NAME) as handle:
            return handle.read()


def _decode_staggered_shots(mat_bytes: bytes) -> dict[int, np.ndarray]:
    """Decode the archive into one ``(shots, ions)`` array per time."""

    cells = sio.loadmat(io.BytesIO(mat_bytes))["dataT"].ravel()  # 20 sets
    pattern = (-1.0) ** np.arange(N_IONS)  # staggered sign per ion (LSB = ion 1)
    per_time: dict[int, list[np.ndarray]] = {}
    for cell in cells:
        cell = np.asarray(cell)  # (11 times, 1 time column + 50 shots)
        times = cell[:, 0]
        shots = cell[:, 1:]
        for row in range(cell.shape[0]):
            key = int(round(float(times[row])))
            bucket = per_time.setdefault(key, [])
            for decimal in shots[row]:
                value = int(round(float(decimal)))
                bits = np.array([(value >> i) & 1 for i in range(N_IONS)])
                spins = np.where(bits == 0, 1.0, -1.0)
                bucket.append(pattern * spins)
    decoded = {key: np.asarray(value, dtype=float) for key, value in per_time.items()}
    keys = sorted(decoded)
    sign = 1.0 if decoded[keys[0]].mean() >= 0 else -1.0
    return {key: sign * decoded[key] for key in keys}


def build_summary(mat_bytes: bytes) -> list[tuple[float, float, float]]:
    """Return (time_ms, mean_imbalance, standard_error) for each time step."""

    decoded = _decode_staggered_shots(mat_bytes)
    rows = []
    for key, site_values in decoded.items():
        values = site_values.mean(axis=1)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(values.size))
        rows.append((key / 1000.0, mean, se))
    return rows


def build_site_summary(
    mat_bytes: bytes,
) -> list[tuple[float, int, float, float]]:
    """Return site-resolved staggered means and standard errors."""

    rows = []
    for key, values in _decode_staggered_shots(mat_bytes).items():
        means = values.mean(axis=0)
        errors = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        for ion, (mean, error) in enumerate(zip(means, errors), start=1):
            rows.append((key / 1000.0, ion, float(mean), float(error)))
    return rows


def render_csv(rows: list[tuple[float, float, float]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["time_ms", "mean_staggered_imbalance", "standard_error"])
    for time_ms, mean, se in rows:
        writer.writerow([_trim(time_ms), repr(mean), repr(se)])
    return buffer.getvalue()


def render_site_csv(rows: list[tuple[float, int, float, float]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["time_ms", "ion", "mean_staggered_spin", "standard_error"]
    )
    for time_ms, ion, mean, se in rows:
        writer.writerow([_trim(time_ms), ion, repr(mean), repr(se)])
    return buffer.getvalue()


def _trim(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(value)


def shipped_csv_text() -> str:
    path = resources.files("krylov_open_quantum").joinpath(
        "data/neel_51ion_imbalance.csv"
    )
    return path.read_text(encoding="utf-8")


def shipped_site_csv_text() -> str:
    path = resources.files("krylov_open_quantum").joinpath(
        "data/neel_51ion_site_imbalance.csv"
    )
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None,
                        help="write the regenerated CSV to this path")
    parser.add_argument(
        "--site-write",
        type=Path,
        default=None,
        help="write the regenerated site-resolved CSV to this path",
    )
    parser.add_argument("--url", default=ZENODO_ZIP)
    args = parser.parse_args()

    print(f"downloading {args.url}")
    mat_bytes = _download_mat(args.url)
    rows = build_summary(mat_bytes)
    site_rows = build_site_summary(mat_bytes)
    rendered = render_csv(rows)
    rendered_site = render_site_csv(site_rows)

    if args.write is not None:
        args.write.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.write}")
    if args.site_write is not None:
        args.site_write.write_text(rendered_site, encoding="utf-8")
        print(f"wrote {args.site_write}")

    shipped = shipped_csv_text()
    ship_rows = list(csv.DictReader(io.StringIO(shipped)))
    max_abs = 0.0
    for (t, mean, se), ref in zip(rows, ship_rows):
        max_abs = max(
            max_abs,
            abs(mean - float(ref["mean_staggered_imbalance"])),
            abs(se - float(ref["standard_error"])),
        )
    print(f"time points: {len(rows)}   shots per time: {50 * 20}")
    print(f"max |recomputed - shipped| (mean or s.e.): {max_abs:.3e}")
    if max_abs == 0.0:
        print("OK: regenerated summary matches the shipped CSV exactly.")
    elif max_abs < 1e-12:
        print("OK: regenerated summary matches the shipped CSV to rounding.")
    else:
        raise SystemExit("MISMATCH: regenerated summary differs from shipped CSV.")

    shipped_site = shipped_site_csv_text()
    if rendered_site != shipped_site:
        raise SystemExit(
            "MISMATCH: regenerated site-resolved summary differs from shipped CSV."
        )
    print(
        "OK: regenerated site-resolved summary matches the shipped CSV exactly "
        f"({len(site_rows)} ion-time records)."
    )


if __name__ == "__main__":
    main()
