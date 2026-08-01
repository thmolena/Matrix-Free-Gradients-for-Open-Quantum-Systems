"""Auditable Lanczos primitives for the Hermitian reproduction baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.generic]
MatVec = Callable[[Array], Array]


@dataclass(frozen=True)
class LanczosDecomposition:
    """Result of a finite Hermitian Lanczos pass."""

    basis: Array
    projected: Array
    initial_norm: float
    residual_coefficient: float
    residual_vector: Array | None

    @property
    def dimension(self) -> int:
        """Return the achieved Krylov dimension."""

        return self.projected.shape[0]


def lanczos(
    matvec: MatVec,
    initial: Array,
    steps: int,
    *,
    reorthogonalize: bool = False,
    breakdown_tolerance: float = 1e-14,
    hermitian_tolerance: float = 1e-10,
) -> LanczosDecomposition:
    """Run a Hermitian Lanczos iteration using matrix-vector products.

    The final residual coefficient is the beta associated with the unretained
    next Lanczos vector. Full two-pass reorthogonalization is optional and is
    intended for validation rather than as the final research algorithm.
    """

    vector = np.asarray(initial)
    if vector.ndim != 1:
        raise ValueError("initial must be a one-dimensional vector")
    if steps < 1:
        raise ValueError("steps must be positive")
    if breakdown_tolerance < 0:
        raise ValueError("breakdown_tolerance must be nonnegative")

    initial_norm = float(np.linalg.norm(vector))
    if initial_norm == 0:
        raise ValueError("initial must be nonzero")

    dtype = np.result_type(vector.dtype, np.float64)
    vector = vector.astype(dtype, copy=False)
    current = vector / initial_norm
    previous = np.zeros_like(current)
    previous_beta = 0.0

    basis_vectors: list[Array] = []
    diagonal: list[complex | float] = []
    off_diagonal: list[float] = []
    residual_coefficient = 0.0
    residual_vector: Array | None = None

    for index in range(steps):
        basis_vectors.append(current.copy())
        work = np.asarray(matvec(current), dtype=dtype)
        if work.shape != current.shape:
            raise ValueError("matvec returned a vector with the wrong shape")

        if index:
            work = work - previous_beta * previous

        alpha = np.vdot(current, work)
        imaginary_scale = hermitian_tolerance * (1.0 + abs(alpha.real))
        if abs(alpha.imag) > imaginary_scale:
            raise ValueError(
                "matvec does not appear Hermitian: Lanczos diagonal is complex"
            )
        alpha = float(alpha.real)
        work = work - alpha * current

        if reorthogonalize:
            basis = np.column_stack(basis_vectors)
            for _ in range(2):
                work = work - basis @ (basis.conj().T @ work)

        beta = float(np.linalg.norm(work))
        diagonal.append(alpha)

        is_last_step = index + 1 == steps
        is_breakdown = beta <= breakdown_tolerance
        if is_last_step or is_breakdown:
            residual_coefficient = beta
            if not is_breakdown:
                residual_vector = work / beta
            break

        off_diagonal.append(beta)
        previous, current = current, work / beta
        previous_beta = beta

    basis = np.column_stack(basis_vectors)
    dimension = len(diagonal)
    projected = np.diag(np.asarray(diagonal, dtype=dtype))
    if dimension > 1:
        off = np.asarray(off_diagonal[: dimension - 1], dtype=dtype)
        projected += np.diag(off, 1) + np.diag(off.conj(), -1)

    return LanczosDecomposition(
        basis=basis,
        projected=projected,
        initial_norm=initial_norm,
        residual_coefficient=residual_coefficient,
        residual_vector=residual_vector,
    )
