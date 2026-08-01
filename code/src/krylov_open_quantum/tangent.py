"""Augmented matrix-free tangent actions for parameter derivatives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .arnoldi import Array, ExponentialAction, expm_action

MatVec = Callable[[Array], Array]


@dataclass(frozen=True)
class TangentAction:
    """State and directional derivative extracted from an augmented action."""

    state: Array
    tangent: Array
    truncation_bound_estimate: float
    roundoff_scale: float
    residual_norm: float
    basis_dimension: int


def augmented_tangent_action(
    matvec: MatVec,
    derivative_matvec: MatVec,
    initial: NDArray[np.generic],
    time: float,
    steps: int,
    *,
    augmented_log_norm_bound: float,
) -> TangentAction:
    """Approximate a state and one Fréchet directional derivative.

    The augmented operator is ``[[A, E], [0, A]]`` and the starting vector is
    ``[0, b]``. Its exponential contains both ``exp(tA)b`` and
    ``D exp(tA)[E] b`` without reverse differentiation through Arnoldi.
    """

    vector = np.asarray(initial, dtype=np.complex128)
    if vector.ndim != 1:
        raise ValueError("initial must be a vector")
    size = vector.size

    def augmented(vector2: Array) -> Array:
        tangent = vector2[:size]
        state = vector2[size:]
        return np.concatenate(
            (
                np.asarray(matvec(tangent)) + np.asarray(derivative_matvec(state)),
                np.asarray(matvec(state)),
            )
        )

    start = np.concatenate((np.zeros_like(vector), vector))
    action: ExponentialAction = expm_action(
        augmented,
        start,
        time,
        steps,
        log_norm_bound=augmented_log_norm_bound,
    )
    return TangentAction(
        state=action.value[size:],
        tangent=action.value[:size],
        truncation_bound_estimate=action.truncation_bound_estimate,
        roundoff_scale=action.roundoff_scale,
        residual_norm=action.residual_norm,
        basis_dimension=action.basis_dimension,
    )
