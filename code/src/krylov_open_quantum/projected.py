"""Projected scalar values and sensitivities for Hermitian matrix functions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .hermitian import Array, LanczosDecomposition

ScalarFunction = Callable[[NDArray[np.floating]], NDArray[np.floating]]


def _divided_difference(
    eigenvalues: NDArray[np.floating],
    function: ScalarFunction,
    derivative: ScalarFunction,
    *,
    collision_tolerance: float = 1e-12,
) -> NDArray[np.floating]:
    values = np.asarray(function(eigenvalues), dtype=float)
    if values.shape != eigenvalues.shape:
        raise ValueError("function must preserve the eigenvalue array shape")

    size = eigenvalues.size
    divided = np.empty((size, size), dtype=float)
    for row in range(size):
        for column in range(size):
            left = eigenvalues[row]
            right = eigenvalues[column]
            scale = 1.0 + max(abs(left), abs(right))
            if abs(left - right) <= collision_tolerance * scale:
                midpoint = np.asarray([(left + right) / 2.0])
                divided[row, column] = float(
                    np.asarray(derivative(midpoint))[0]
                )
            else:
                divided[row, column] = (
                    values[row] - values[column]
                ) / (left - right)
    return divided


def projected_quadratic(
    decomposition: LanczosDecomposition,
    function: ScalarFunction,
) -> float:
    """Evaluate the projected approximation to u* f(A) u."""

    eigenvalues, eigenvectors = np.linalg.eigh(decomposition.projected)
    first_coordinates = eigenvectors.conj().T[:, 0]
    values = np.asarray(function(eigenvalues), dtype=float)
    estimate = decomposition.initial_norm**2 * np.vdot(
        first_coordinates, values * first_coordinates
    )
    return float(np.real(estimate))


def projected_sensitivity(
    decomposition: LanczosDecomposition,
    function: ScalarFunction,
    derivative: ScalarFunction,
) -> Array:
    """Return the small projected sensitivity matrix G."""

    eigenvalues, eigenvectors = np.linalg.eigh(decomposition.projected)
    first_coordinates = eigenvectors.conj().T[:, 0]
    divided = _divided_difference(eigenvalues, function, derivative)
    weights = np.outer(first_coordinates, first_coordinates.conj())
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        sensitivity = (
            decomposition.initial_norm**2
            * eigenvectors
            @ (weights * divided)
            @ eigenvectors.conj().T
        )
        sensitivity = 0.5 * (sensitivity + sensitivity.conj().T)
    if not np.all(np.isfinite(sensitivity)):
        raise FloatingPointError("projected sensitivity is nonfinite")
    return sensitivity


def lift_projected_sensitivity(
    decomposition: LanczosDecomposition,
    sensitivity: Array,
) -> Array:
    """Lift G to the dense matrix V G V* for small validation problems."""

    expected = (decomposition.dimension, decomposition.dimension)
    if sensitivity.shape != expected:
        raise ValueError(f"sensitivity must have shape {expected}")
    basis = decomposition.basis
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        lifted = basis @ sensitivity @ basis.conj().T
    if not np.all(np.isfinite(lifted)):
        raise FloatingPointError("lifted sensitivity is nonfinite")
    return lifted


def directional_derivative(matrix_gradient: Array, direction: Array) -> float:
    """Contract a Hermitian matrix sensitivity with a Hermitian direction."""

    if matrix_gradient.shape != direction.shape:
        raise ValueError("matrix_gradient and direction must have equal shape")
    return float(np.real(np.vdot(matrix_gradient, direction)))
