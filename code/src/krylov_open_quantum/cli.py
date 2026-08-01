"""Command-line interface for the reproducibility artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .experiment import canonical_bytes, reproduce
from .field_data import load_neel_site_imbalance
from .site_learning import benchmark_site_gradient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute error-controlled open-quantum results on a CPU."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-seed", type=int, default=260724)
    parser.add_argument("--validation-size", type=int, default=18)
    parser.add_argument("--validation-time", type=float, default=0.8)
    parser.add_argument(
        "--validation-steps",
        type=int,
        nargs="+",
        default=(4, 6, 8),
        metavar="M",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="also time matrix-free and dense kernels on three real-data times",
    )
    args = parser.parse_args()
    results = reproduce(
        args.output,
        args.verify,
        validation_seed=args.validation_seed,
        validation_size=args.validation_size,
        validation_time=args.validation_time,
        validation_steps=tuple(args.validation_steps),
    )
    rendered = canonical_bytes(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"sha256={hashlib.sha256(rendered).hexdigest()}")
    if args.benchmark:
        _, observed, _ = load_neel_site_imbalance()
        parameters = results["site_resolved_learning"]["coupled_parameters"]
        benchmark = benchmark_site_gradient(parameters, observed[0])
        print(json.dumps({"benchmark": benchmark}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
