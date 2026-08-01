"""Small open-system models used by the CPU validation artifact."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def damped_rotation(parameters: NDArray[np.generic]) -> np.ndarray:
    """Return a two-coordinate effective open-system generator.

    Parameters are angular frequency and damping rate, both in inverse time.
    """

    omega, gamma = np.asarray(parameters, dtype=float)
    if gamma < 0.0:
        raise ValueError("damping rate must be nonnegative")
    return np.array([[-gamma, -omega], [omega, -gamma]], dtype=float)


def damped_rotation_derivatives() -> tuple[np.ndarray, np.ndarray]:
    """Return derivatives with respect to frequency and damping."""

    return (
        np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float),
        -np.eye(2, dtype=float),
    )


def augmented_matrix(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Assemble a validation-sized augmented tangent generator."""

    zeros = np.zeros_like(matrix)
    return np.block([[matrix, direction], [zeros, matrix]])
