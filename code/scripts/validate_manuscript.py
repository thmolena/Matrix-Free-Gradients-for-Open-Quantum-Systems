#!/usr/bin/env python3
"""Validate the residual-subspace manuscript and evidence contract."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"


def pdf_pages(path: Path) -> int:
    executable = shutil.which("pdfinfo")
    if executable is None:
        raise RuntimeError("pdfinfo is required")
    text = subprocess.run([executable, str(path)], check=True, capture_output=True, text=True).stdout
    return int(re.search(r"^Pages:\s+(\d+)", text, re.MULTILINE).group(1))


def main() -> int:
    failures, passes = [], []
    def check(condition: bool, message: str) -> None:
        (passes if condition else failures).append(message)

    required = [
        ROOT / "README.md", ROOT / "index.html", ROOT / "main.tex", ROOT / "main.pdf",
        CODE / "results/recovery_study.json",
        CODE / "src/krylov_open_quantum/subspace_recovery.py",
        CODE / "src/krylov_open_quantum/recovery_study.py",
        CODE / "src/krylov_open_quantum/recovery_figures.py",
        CODE / "src/krylov_open_quantum/data/neel_51ion_site_imbalance.csv",
    ]
    for path in required:
        check(path.is_file(), f"required file: {path.relative_to(ROOT)}")

    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    pages = pdf_pages(ROOT / "main.pdf")
    check(3 <= pages <= 6, f"rendered page count is 3--6 (found {pages})")
    check(
        tex.startswith(r"\documentclass[10pt,letterpaper,twoside]{article}"),
        "one-column preprint contract",
    )
    check(len(re.findall(r"\\begin\{theorem\}", tex)) == 2, "two proved theorems")
    check(len(re.findall(r"\\begin\{proof\}", tex)) == 2, "two theorem proofs")
    check(len(re.findall(r"\\begin\{table\*?\}", tex)) == 2, "two evidence tables")
    included = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    allowed = {
        "code/manuscript_assets/figures/recovery/selection_cost.pdf",
        "code/manuscript_assets/figures/recovery/confirmation_overview.pdf",
    }
    check(included == allowed, "exact two-figure evidence set")
    for relative in allowed:
        check((ROOT / relative).is_file(), f"figure exists: {relative}")

    result = json.loads((CODE / "results/recovery_study.json").read_text())
    protocol = result["protocol"]
    pooled = result["confirmation"]["pooled"]
    check(result["schema_version"] == 1, "results schema version 1")
    check(protocol["selected_rank"] == 4, "development-selected rank is four")
    check(protocol["confirmation_times_ms"] == [8.0, 9.0, 10.0], "confirmation times remain fixed")
    check(pooled["paired_cases"] == 30, "thirty registered confirmation cases")
    check(pooled["paired_wins_vs_observed_mean"] == 25, "paired wins and losses retained")
    check(pooled["residual_subspace_rmse"] < pooled["observed_mean_rmse"], "pooled confirmation improvement")
    check(len(result["confirmation"]["records"]) == 30, "case-level evidence retained")
    check(all(row["median_microseconds"] > 0 for row in result["runtime"]["rows"]), "positive measured runtimes")
    check([row["stored_basis_scalars"] for row in result["runtime"]["rows"]] == [51*r for r in range(1, 8)], "exact basis storage path")

    public = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in ("main.tex", "README.md", "index.html"))
    check(not re.search(r"\b(?:AI|agent|agentic|ChatGPT)\b", public, re.IGNORECASE), "prohibited standalone words absent")
    check("?" not in public, "question punctuation absent")
    check("10.5281/zenodo.17750419" in public, "authenticated source identified")
    check("25 of 30" in public and "five" in public.lower(), "confirmation losses disclosed")

    for message in passes:
        print("PASS", message)
    for message in failures:
        print("FAIL", message)
    print(f"SUMMARY passes={len(passes)} failures={len(failures)} pages={pages}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
