#!/usr/bin/env python3
"""Run a deterministic CPU smoke check of the Hermitian baseline."""

from __future__ import annotations

import json

import numpy as np
from scipy.linalg import expm

from krylov_open_quantum import (
    directional_derivative,
    lanczos,
    lift_projected_sensitivity,
    projected_quadratic,
    projected_sensitivity,
)


def symmetric(rng: np.random.Generator, size: int) -> np.ndarray:
    raw = rng.normal(size=(size, size))
    return 0.5 * (raw + raw.T)


def main() -> None:
    rng = np.random.default_rng(0)
    size = 8
    matrix = symmetric(rng, size)
    direction = symmetric(rng, size)
    initial = rng.normal(size=size)

    decomposition = lanczos(
        lambda vector: matrix @ vector,
        initial,
        size,
        reorthogonalize=True,
    )
    value = projected_quadratic(decomposition, np.exp)
    dense_value = float(initial @ expm(matrix) @ initial)

    small_gradient = projected_sensitivity(
        decomposition,
        np.exp,
        np.exp,
    )
    matrix_gradient = lift_projected_sensitivity(
        decomposition,
        small_gradient,
    )
    gradient = directional_derivative(matrix_gradient, direction)

    epsilon = 1e-6
    plus = float(initial @ expm(matrix + epsilon * direction) @ initial)
    minus = float(initial @ expm(matrix - epsilon * direction) @ initial)
    finite_difference = (plus - minus) / (2.0 * epsilon)

    summary = {
        "dimension": size,
        "krylov_dimension": decomposition.dimension,
        "residual_coefficient": decomposition.residual_coefficient,
        "value_absolute_error": abs(value - dense_value),
        "gradient_absolute_error": abs(gradient - finite_difference),
        "status": "ok",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
