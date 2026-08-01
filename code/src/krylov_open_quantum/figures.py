"""Evidence builders for the four manuscript line figures.

The synthetic problem isolates numerical truncation and is checked against
SciPy's independent dense ``expm_frechet`` implementation.  The remaining
figures use the attributed 51-ion archive packaged with the project.  Runtime
values are measured medians and are intentionally not byte locked.
"""

from __future__ import annotations

from pathlib import Path
from platform import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.linalg import expm_frechet

from .arnoldi import logarithmic_norm
from .experiment import fit_experimental_imbalance
from .field_data import load_neel_imbalance, load_neel_site_imbalance
from .site_learning import dense_reference, fit_site_resolved_dynamics, predict_and_jacobian
from .tangent import augmented_tangent_action

_INK = "#1b1b1b"
_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_GREY = "#777777"


def _style() -> None:
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": _INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
        }
    )


def tangent_convergence_sweep(
    *,
    seed: int = 260731,
    size: int = 24,
    time: float = 0.9,
    steps: tuple[int, ...] = tuple(range(2, 25, 2)),
) -> dict:
    """Joint state--tangent convergence against ``expm_frechet``."""

    rng = np.random.default_rng(seed)
    matrix = 0.07 * rng.normal(size=(size, size)) - np.eye(size)
    direction = 0.05 * rng.normal(size=(size, size))
    initial = rng.normal(size=size)
    augmented = np.block(
        [[matrix, direction], [np.zeros_like(matrix), matrix]]
    )
    dense_state, dense_frechet = expm_frechet(
        time * matrix, time * direction, compute_expm=True
    )
    reference_state = dense_state @ initial
    reference_tangent = dense_frechet @ initial

    rows: list[dict] = []
    for dimension in steps:
        result = augmented_tangent_action(
            lambda vector: matrix @ vector,
            lambda vector: direction @ vector,
            initial,
            time,
            dimension,
            augmented_log_norm_bound=logarithmic_norm(augmented),
        )
        state_error = float(np.linalg.norm(result.state - reference_state))
        tangent_error = float(np.linalg.norm(result.tangent - reference_tangent))
        joint_error = float(np.hypot(state_error, tangent_error))
        rows.append(
            {
                "m": int(dimension),
                "state_error": state_error,
                "tangent_error": tangent_error,
                "joint_error": joint_error,
                "truncation_bound_estimate": float(
                    result.truncation_bound_estimate
                ),
                "roundoff_scale": float(result.roundoff_scale),
                "observed_coverage": bool(
                    joint_error <= result.truncation_bound_estimate
                ),
            }
        )
    return {
        "seed": seed,
        "size": size,
        "time": time,
        "reference": "scipy.linalg.expm_frechet",
        "rows": rows,
    }


def field_curve(points: int = 400) -> dict:
    """Aggregate authentic-data curve and effective damped-mode fit."""

    time, observed, standard_error = load_neel_imbalance()
    fit = fit_experimental_imbalance()
    omega = fit["fitted_angular_frequency_per_ms"]
    gamma = fit["fitted_damping_rate_per_ms"]
    amplitude = float(observed[0])
    grid = np.linspace(float(time[0]), float(time[-1]), points)
    curve = amplitude * np.exp(-gamma * grid) * np.cos(omega * grid)
    return {
        "time": time.tolist(),
        "observed": observed.tolist(),
        "standard_error": standard_error.tolist(),
        "grid": grid.tolist(),
        "curve": curve.tolist(),
        "omega": omega,
        "gamma": gamma,
        "rmse": fit["rmse_staggered_imbalance"],
    }


def realdata_krylov_sweep(
    *,
    steps: tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20, 24, 28),
    repeats: int = 7,
    parameters: np.ndarray | None = None,
) -> dict:
    """Accuracy and measured runtime on the 10-ms 51-ion observation state."""

    times, observed, _ = load_neel_site_imbalance()
    if parameters is None:
        parameters = np.asarray(
            fit_site_resolved_dynamics()["coupled_parameters"], dtype=float
        )
    selected = times[[-1]]
    dense_prediction, dense_jacobian = dense_reference(
        parameters, selected, observed[0]
    )
    dense_durations = []
    for _ in range(repeats):
        start = perf_counter()
        dense_reference(parameters, selected, observed[0])
        dense_durations.append(perf_counter() - start)
    rows: list[dict] = []
    for dimension in steps:
        # One untimed call warms imports and allocators for this dimension.
        result = predict_and_jacobian(
            parameters, selected, observed[0], steps=dimension
        )
        durations = []
        for _ in range(repeats):
            start = perf_counter()
            result = predict_and_jacobian(
                parameters, selected, observed[0], steps=dimension
            )
            durations.append(perf_counter() - start)
        prediction_error = float(
            np.linalg.norm(result.prediction - dense_prediction)
            / np.linalg.norm(dense_prediction)
        )
        jacobian_error = float(
            np.linalg.norm(result.jacobian - dense_jacobian)
            / np.linalg.norm(dense_jacobian)
        )
        rows.append(
            {
                "m": int(dimension),
                "prediction_relative_error": prediction_error,
                "jacobian_relative_error": jacobian_error,
                "truncation_bound_estimate": float(
                    result.largest_truncation_estimate
                ),
                "median_runtime_seconds": float(np.median(durations)),
                "minimum_runtime_seconds": float(np.min(durations)),
            }
        )
    return {
        "source_doi": "10.5281/zenodo.17750419",
        "time_ms": float(selected[0]),
        "state_dimension": 102,
        "parameters": np.asarray(parameters).tolist(),
        "repeats": repeats,
        "dense_median_runtime_seconds": float(np.median(dense_durations)),
        "dense_minimum_runtime_seconds": float(np.min(dense_durations)),
        "platform": platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "rows": rows,
    }


def site_holdout_curve(parameters: dict | None = None) -> dict:
    """Held-out 10-ms site profile for two effective models."""

    times, observed, standard_error = load_neel_site_imbalance()
    fit = fit_site_resolved_dynamics() if parameters is None else parameters
    baseline = predict_and_jacobian(
        np.asarray(fit["baseline_parameters"]), times, observed[0]
    ).prediction
    coupled = predict_and_jacobian(
        np.asarray(fit["coupled_parameters"]), times, observed[0]
    ).prediction
    held_out = times.size - 1
    return {
        "source_doi": "10.5281/zenodo.17750419",
        "ions": list(range(1, observed.shape[1] + 1)),
        "time_ms": float(times[held_out]),
        "observed": observed[held_out].tolist(),
        "standard_error": standard_error[held_out].tolist(),
        "baseline": baseline[held_out].tolist(),
        "coupled": coupled[held_out].tolist(),
        "baseline_test_rmse": fit["baseline_test_rmse"],
        "coupled_test_rmse": fit["coupled_test_rmse"],
    }


def _plot_tangent(data: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = data["rows"]
    m = [row["m"] for row in rows]
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.semilogy(
        m,
        [row["joint_error"] for row in rows],
        "-o",
        color=_BLUE,
        ms=3.0,
        label="joint error vs. dense Fréchet reference",
    )
    ax.semilogy(
        m,
        [row["truncation_bound_estimate"] for row in rows],
        "--s",
        color=_ORANGE,
        ms=3.0,
        label="computed exact-arithmetic bound",
    )
    ax.semilogy(
        m,
        [row["roundoff_scale"] for row in rows],
        ":",
        color=_GREY,
        label="one-unit roundoff scale (diagnostic)",
    )
    ax.set_xlabel("augmented Arnoldi dimension $m$")
    ax.set_ylabel("Euclidean error or estimate")
    ax.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def _plot_field(data: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.plot(data["grid"], data["curve"], color=_ORANGE, label="damped-mode fit")
    ax.errorbar(
        data["time"],
        data["observed"],
        yerr=data["standard_error"],
        fmt="o",
        color=_INK,
        ms=3.2,
        capsize=2,
        lw=0.8,
        label="51-ion shot averages",
    )
    ax.axhline(0.0, color="#bbbbbb", lw=0.6)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("staggered imbalance")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def _plot_realdata_krylov(data: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = data["rows"]
    m = [row["m"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.25), sharex=True)
    axes[0].semilogy(
        m,
        [row["prediction_relative_error"] for row in rows],
        "-o",
        color=_BLUE,
        ms=3.0,
        label="state",
    )
    axes[0].semilogy(
        m,
        [row["jacobian_relative_error"] for row in rows],
        "--s",
        color=_ORANGE,
        ms=3.0,
        label="three-parameter Jacobian",
    )
    axes[0].set_ylabel("relative error")
    axes[0].legend(loc="upper right")
    axes[1].plot(
        m,
        [1.0e3 * row["median_runtime_seconds"] for row in rows],
        "-o",
        color=_GREEN,
        ms=3.0,
        label="matrix-free tangent",
    )
    axes[1].axhline(
        1.0e3 * data["dense_median_runtime_seconds"],
        color=_GREY,
        linestyle="--",
        label="dense block Fréchet reference",
    )
    axes[1].set_xlabel("augmented Arnoldi dimension $m$")
    axes[1].set_ylabel("median runtime (ms)")
    axes[1].legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def _plot_site_holdout(data: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.errorbar(
        data["ions"],
        data["observed"],
        yerr=data["standard_error"],
        fmt="o",
        color=_INK,
        ms=2.1,
        capsize=1.4,
        lw=0.6,
        label="held-out shot averages",
    )
    ax.plot(data["ions"], data["baseline"], "--", color=_GREY, label="uncoupled")
    ax.plot(data["ions"], data["coupled"], "-", color=_BLUE, label="coupled")
    ax.set_xlabel("ion index")
    ax.set_ylabel("staggered spin at 10 ms")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def build_all(output_dir: Path) -> dict:
    """Regenerate the four manuscript figures and return their source data."""

    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    tangent = tangent_convergence_sweep()
    field = field_curve()
    fit = fit_site_resolved_dynamics()
    realdata = realdata_krylov_sweep(
        parameters=np.asarray(fit["coupled_parameters"])
    )
    holdout = site_holdout_curve(fit)
    _plot_tangent(tangent, output_dir / "fig_tangent_convergence.pdf")
    _plot_realdata_krylov(realdata, output_dir / "fig_realdata_krylov.pdf")
    _plot_field(field, output_dir / "fig_field_fit.pdf")
    _plot_site_holdout(holdout, output_dir / "fig_site_holdout.pdf")
    return {
        "tangent_convergence": tangent,
        "realdata_krylov": realdata,
        "field_fit": {
            "omega": field["omega"],
            "gamma": field["gamma"],
            "rmse": field["rmse"],
        },
        "site_holdout": holdout,
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Regenerate manuscript line figures.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("code/manuscript_assets/figures"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional machine-readable snapshot, including measured runtimes",
    )
    args = parser.parse_args()
    summary = build_all(args.output)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
