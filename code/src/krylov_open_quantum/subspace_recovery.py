"""Variance-weighted residual-subspace recovery for missing site readouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualSubspace:
    """Mean-augmented spatial basis learned from complete development profiles."""

    basis: np.ndarray
    singular_values: np.ndarray

    @classmethod
    def fit(cls, profiles: np.ndarray, rank: int) -> "ResidualSubspace":
        data = np.asarray(profiles, dtype=float)
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError("profiles must be a two-dimensional array with at least two rows")
        if rank < 1 or rank > min(data.shape):
            raise ValueError("rank is outside the available residual subspace")
        residual = data - data.mean(axis=1, keepdims=True)
        _, singular_values, right = np.linalg.svd(residual, full_matrices=False)
        return cls(basis=right[:rank].T.copy(), singular_values=singular_values.copy())

    @property
    def design(self) -> np.ndarray:
        return np.column_stack((np.ones(self.basis.shape[0]), self.basis))

    def recover(
        self,
        values: np.ndarray,
        observed: np.ndarray,
        standard_error: np.ndarray,
        *,
        ridge: float = 0.0,
    ) -> np.ndarray:
        """Recover a profile by weighted least squares on observed coordinates."""

        values = np.asarray(values, dtype=float)
        observed = np.asarray(observed, dtype=bool)
        standard_error = np.asarray(standard_error, dtype=float)
        n = self.basis.shape[0]
        if values.shape != (n,) or observed.shape != (n,) or standard_error.shape != (n,):
            raise ValueError("values, observed, and standard_error must match the site count")
        if ridge < 0.0 or observed.sum() < self.basis.shape[1] + 1:
            raise ValueError("ridge must be nonnegative and enough sites must be observed")
        design = self.design
        weight = 1.0 / np.maximum(standard_error[observed], 1.0e-2)
        weighted_design = design[observed] * weight[:, None]
        normal = weighted_design.T @ weighted_design
        normal[1:, 1:] += ridge * np.eye(self.basis.shape[1])
        rhs = weighted_design.T @ (values[observed] * weight)
        coefficient = np.linalg.solve(normal, rhs)
        return design @ coefficient


def periodic_masks(sites: int = 51, period: int = 5) -> list[tuple[str, np.ndarray]]:
    if sites < period or period < 2:
        raise ValueError("sites must be at least the period, and period at least two")
    coordinate = np.arange(sites)
    return [(f"periodic-{offset}", coordinate % period == offset) for offset in range(period)]


def contiguous_masks(sites: int = 51, width: int = 10) -> list[tuple[str, np.ndarray]]:
    if sites < width or width < 1:
        raise ValueError("width must lie between one and the number of sites")
    starts = np.linspace(0, sites - width, 5).round().astype(int)
    masks = []
    for start in starts:
        mask = np.zeros(sites, dtype=bool)
        mask[start : start + width] = True
        masks.append((f"block-{start + 1}", mask))
    return masks


def linear_interpolation(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    coordinates = np.arange(values.size)
    return np.interp(coordinates, coordinates[observed], values[observed])


def nearest_fill(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    kept = np.flatnonzero(observed)
    nearest = np.abs(np.arange(values.size)[:, None] - kept).argmin(axis=1)
    return values[kept[nearest]]
