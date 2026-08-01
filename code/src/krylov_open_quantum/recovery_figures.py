"""Publication figures for the registered ion-readout recovery study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .field_data import load_neel_site_imbalance
from .recovery_study import _evaluate


def _style() -> None:
    import matplotlib
    matplotlib.rcParams.update({
        "font.size": 8.2, "axes.labelsize": 8.2, "axes.titlesize": 8.2,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.3, "ytick.labelsize": 7.3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#dddddd", "grid.linewidth": 0.5, "legend.frameon": False,
        "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight",
    })


def make_figures(study_path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    _style()
    result = json.loads(study_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    blue, orange, green, ink = "#0072B2", "#D55E00", "#009E73", "#222222"

    # Development selection and post-lock confirmation sensitivity.
    times, values, errors = load_neel_site_imbalance()
    ranks = [row["rank"] for row in result["development_curve"]]
    development = [row["rmse"] for row in result["development_curve"]]
    confirmation = [_evaluate(times, values, errors, range(8, 11), rank, leave_one_out=False)["pooled"]["residual_subspace_rmse"] for rank in ranks]
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    ax.plot(ranks, development, "-o", color=blue, ms=3.2, label="development LOEO")
    ax.plot(ranks, confirmation, "--s", color=orange, ms=3.0, label="confirmation (post-lock)")
    ax.axvline(result["protocol"]["selected_rank"], color=ink, ls=":", label="selected rank")
    ax.set(xlabel="residual-subspace rank", ylabel="pooled missing-site RMSE", xticks=ranks)
    ax.legend()
    fig.savefig(output / "rank_selection.pdf"); fig.savefig(output / "rank_selection.png", dpi=220); plt.close(fig)

    # Every confirmation case, sorted by the registered baseline error.
    records = result["confirmation"]["records"]
    records = sorted(records, key=lambda row: row["observed_mean_rmse"])
    index = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.plot(index, [row["observed_mean_rmse"] for row in records], "-o", color=ink, ms=2.7, label="observed-site mean")
    ax.plot(index, [row["residual_subspace_rmse"] for row in records], "-o", color=blue, ms=2.7, label="residual subspace")
    losses = [i for i, row in enumerate(records) if row["residual_subspace_rmse"] >= row["observed_mean_rmse"]]
    ax.scatter(losses, [records[i]["residual_subspace_rmse"] for i in losses], marker="x", color=orange, s=22, zorder=4, label="method loss")
    ax.set(xlabel="confirmation mask-time case (sorted)", ylabel="missing-site RMSE")
    ax.legend()
    fig.savefig(output / "confirmation_cases.pdf"); fig.savefig(output / "confirmation_cases.png", dpi=220); plt.close(fig)

    # A contiguous ten-ion outage at the final time.
    example = result["example"]
    site = np.asarray(example["site"]); truth = np.asarray(example["truth"]); prediction = np.asarray(example["prediction"]); observed = np.asarray(example["observed"], dtype=bool); se = np.asarray(example["standard_error"])
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.errorbar(site[observed], truth[observed], yerr=se[observed], fmt="o", color=ink, ms=2.8, capsize=1.4, label="available readouts")
    ax.errorbar(site[~observed], truth[~observed], yerr=se[~observed], fmt="s", mfc="white", color=orange, ms=3.4, capsize=1.4, label="withheld readouts")
    ax.plot(site, prediction, color=blue, label="reconstruction")
    ax.axhline(0, color="#aaaaaa", lw=.6)
    ax.set(xlabel="ion index", ylabel="staggered spin at 10 ms")
    ax.legend(ncol=1)
    fig.savefig(output / "outage_profile.pdf"); fig.savefig(output / "outage_profile.png", dpi=220); plt.close(fig)

    # Measured recovery cost and explicit storage count.
    rows = result["runtime"]["rows"]
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    ax.plot([row["rank"] for row in rows], [row["median_microseconds"] for row in rows], "-o", color=green, ms=3.2)
    ax.set(xlabel="residual-subspace rank", ylabel="median recovery time (microseconds)", xticks=ranks)
    secondary = ax.twinx(); secondary.spines["right"].set_visible(True)
    secondary.plot([row["rank"] for row in rows], [row["stored_basis_scalars"] for row in rows], "--s", color=orange, ms=3.0)
    secondary.set_ylabel("stored basis scalars")
    fig.savefig(output / "runtime_storage.pdf"); fig.savefig(output / "runtime_storage.png", dpi=220); plt.close(fig)

    # Wide paired view used in the manuscript: statistical and computational selection.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    axes[0].plot(ranks, development, "-o", color=blue, ms=3.2, label="development LOEO")
    axes[0].plot(ranks, confirmation, "--s", color=orange, ms=3.0, label="confirmation (post-lock)")
    axes[0].axvline(result["protocol"]["selected_rank"], color=ink, ls=":", label="selected rank")
    axes[0].set(xlabel="residual-subspace rank", ylabel="pooled missing-site RMSE", xticks=ranks)
    axes[0].legend()
    axes[0].text(-0.15, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].plot([row["rank"] for row in rows], [row["median_microseconds"] for row in rows], "-o", color=green, ms=3.2)
    axes[1].set(xlabel="residual-subspace rank", ylabel="median recovery time (microseconds)", xticks=ranks)
    paired = axes[1].twinx(); paired.spines["right"].set_visible(True)
    paired.plot([row["rank"] for row in rows], [row["stored_basis_scalars"] for row in rows], "--s", color=orange, ms=3.0)
    paired.set_ylabel("stored basis scalars")
    axes[1].text(-0.15, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.subplots_adjust(wspace=0.34)
    fig.savefig(output / "selection_cost.pdf"); fig.savefig(output / "selection_cost.png", dpi=220); plt.close(fig)

    # Wide paired confirmation view used in the manuscript.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    axes[0].plot(index, [row["observed_mean_rmse"] for row in records], "-o", color=ink, ms=2.7, label="observed-site mean")
    axes[0].plot(index, [row["residual_subspace_rmse"] for row in records], "-o", color=blue, ms=2.7, label="residual subspace")
    axes[0].scatter(losses, [records[i]["residual_subspace_rmse"] for i in losses], marker="x", color=orange, s=22, zorder=4, label="method loss")
    axes[0].set(xlabel="confirmation mask-time case (sorted)", ylabel="missing-site RMSE")
    axes[0].legend(); axes[0].text(-0.15, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].errorbar(site[observed], truth[observed], yerr=se[observed], fmt="o", color=ink, ms=2.8, capsize=1.4, label="available readouts")
    axes[1].errorbar(site[~observed], truth[~observed], yerr=se[~observed], fmt="s", mfc="white", color=orange, ms=3.4, capsize=1.4, label="withheld readouts")
    axes[1].plot(site, prediction, color=blue, label="reconstruction")
    axes[1].axhline(0, color="#aaaaaa", lw=.6)
    axes[1].set(xlabel="ion index", ylabel="staggered spin at 10 ms")
    axes[1].legend(); axes[1].text(-0.15, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold")
    fig.subplots_adjust(wspace=0.30)
    fig.savefig(output / "confirmation_overview.pdf"); fig.savefig(output / "confirmation_overview.png", dpi=220); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=Path("code/results/recovery_study.json"))
    parser.add_argument("--output", type=Path, default=Path("code/manuscript_assets/figures/recovery"))
    args = parser.parse_args(); make_figures(args.study, args.output)


if __name__ == "__main__":
    main()
