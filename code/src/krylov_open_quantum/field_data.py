"""Loader for the attributed 51-ion experimental Néel-state summary."""

from __future__ import annotations

import csv
from importlib import resources

import numpy as np


def load_neel_imbalance() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time in ms, mean staggered imbalance, and standard error."""

    path = resources.files("krylov_open_quantum").joinpath(
        "data/neel_51ion_imbalance.csv"
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    time = np.array([float(row["time_ms"]) for row in rows])
    mean = np.array([float(row["mean_staggered_imbalance"]) for row in rows])
    standard_error = np.array([float(row["standard_error"]) for row in rows])
    return time, mean, standard_error


def load_neel_site_imbalance() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time and ``(time, ion)`` site-resolved means and errors."""

    path = resources.files("krylov_open_quantum").joinpath(
        "data/neel_51ion_site_imbalance.csv"
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = sorted({float(row["time_ms"]) for row in rows})
    ions = sorted({int(row["ion"]) for row in rows})
    time_index = {value: index for index, value in enumerate(times)}
    ion_index = {value: index for index, value in enumerate(ions)}
    mean = np.empty((len(times), len(ions)), dtype=float)
    standard_error = np.empty_like(mean)
    for row in rows:
        i = time_index[float(row["time_ms"])]
        j = ion_index[int(row["ion"])]
        mean[i, j] = float(row["mean_staggered_spin"])
        standard_error[i, j] = float(row["standard_error"])
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(standard_error)):
        raise ValueError("site-resolved data contain nonfinite values")
    return np.asarray(times), mean, standard_error
