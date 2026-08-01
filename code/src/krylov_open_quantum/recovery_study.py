"""Locked development/confirmation study for missing ion readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from .field_data import load_neel_site_imbalance
from .subspace_recovery import (
    ResidualSubspace,
    contiguous_masks,
    linear_interpolation,
    nearest_fill,
    periodic_masks,
)


def _rmse(prediction: np.ndarray, truth: np.ndarray, missing: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction[missing] - truth[missing]) ** 2)))


def _evaluate(times: np.ndarray, values: np.ndarray, errors: np.ndarray, rows: range, rank: int, *, leave_one_out: bool) -> dict:
    families = {"periodic": periodic_masks(values.shape[1]), "contiguous": contiguous_masks(values.shape[1])}
    records = []
    for time_index in rows:
        training = values[:8]
        if leave_one_out:
            training = values[[index for index in range(8) if index != time_index]]
        model = ResidualSubspace.fit(training, rank)
        for family, masks in families.items():
            for name, missing in masks:
                observed = ~missing
                proposed = model.recover(values[time_index], observed, errors[time_index])
                mean = np.full(values.shape[1], values[time_index, observed].mean())
                interpolation = linear_interpolation(values[time_index], observed)
                nearest = nearest_fill(values[time_index], observed)
                records.append({
                    "time_ms": float(times[time_index]), "family": family, "mask": name,
                    "missing_sites": int(missing.sum()),
                    "residual_subspace_rmse": _rmse(proposed, values[time_index], missing),
                    "observed_mean_rmse": _rmse(mean, values[time_index], missing),
                    "linear_interpolation_rmse": _rmse(interpolation, values[time_index], missing),
                    "nearest_fill_rmse": _rmse(nearest, values[time_index], missing),
                })
    pooled = {}
    for method in ("residual_subspace", "observed_mean", "linear_interpolation", "nearest_fill"):
        key = f"{method}_rmse"
        pooled[key] = float(np.sqrt(np.mean([record[key] ** 2 for record in records])))
    strongest = min(pooled[key] for key in pooled if key != "residual_subspace_rmse")
    pooled["squared_error_reduction_vs_strongest_simple"] = float(1.0 - pooled["residual_subspace_rmse"] ** 2 / strongest**2)
    pooled["paired_wins_vs_observed_mean"] = int(sum(record["residual_subspace_rmse"] < record["observed_mean_rmse"] for record in records))
    pooled["paired_cases"] = len(records)
    return {"pooled": pooled, "records": records}


def run_study(repeats: int = 101) -> dict:
    times, values, errors = load_neel_site_imbalance()
    ranks = list(range(1, 8))
    development_curve = []
    for rank in ranks:
        result = _evaluate(times, values, errors, range(1, 8), rank, leave_one_out=True)
        development_curve.append({"rank": rank, "rmse": result["pooled"]["residual_subspace_rmse"]})
    selected_rank = min(development_curve, key=lambda row: (row["rmse"], row["rank"]))["rank"]
    development = _evaluate(times, values, errors, range(1, 8), selected_rank, leave_one_out=True)
    confirmation = _evaluate(times, values, errors, range(8, 11), selected_rank, leave_one_out=False)

    model = ResidualSubspace.fit(values[:8], selected_rank)
    missing = periodic_masks(values.shape[1])[0][1]
    observed = ~missing
    runtime = []
    for rank in ranks:
        candidate = ResidualSubspace.fit(values[:8], rank)
        samples = []
        for _ in range(repeats):
            start = perf_counter_ns()
            candidate.recover(values[10], observed, errors[10])
            samples.append(perf_counter_ns() - start)
        runtime.append({"rank": rank, "median_microseconds": float(np.median(samples) / 1000.0), "stored_basis_scalars": int(values.shape[1] * rank)})

    example_missing = contiguous_masks(values.shape[1])[2][1]
    example_observed = ~example_missing
    example_prediction = model.recover(values[10], example_observed, errors[10])
    return {
        "schema_version": 1,
        "source": {"doi": "10.5281/zenodo.17750419", "license": "CC-BY-4.0", "ions": 51, "time_points": 11, "records": 561, "shots_per_time": 1000},
        "protocol": {"development_times_ms": times[1:8].tolist(), "confirmation_times_ms": times[8:11].tolist(), "candidate_ranks": ranks, "selected_rank": selected_rank, "ridge": 0.0, "mask_families": ["periodic one-in-five", "five contiguous ten-site outages"]},
        "development_curve": development_curve,
        "development": development,
        "confirmation": confirmation,
        "singular_values": model.singular_values.tolist(),
        "runtime": {"repeats": repeats, "rows": runtime},
        "example": {"time_ms": float(times[10]), "site": list(range(1, 52)), "observed": example_observed.tolist(), "truth": values[10].tolist(), "prediction": example_prediction.tolist(), "standard_error": errors[10].tolist()},
    }


def semantic_view(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "runtime"}


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("code/results/recovery_study.json"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = run_study()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.verify and args.output.exists():
        old = json.loads(args.output.read_text())
        if semantic_view(old) != semantic_view(result):
            raise SystemExit("semantic results differ from the locked artifact")
    args.output.write_bytes(_canonical(result))
    print("semantic_sha256", hashlib.sha256(_canonical(semantic_view(result))).hexdigest())
    print(json.dumps(result["confirmation"]["pooled"], indent=2))


if __name__ == "__main__":
    main()
