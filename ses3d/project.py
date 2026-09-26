"""Gaussian projection of patch scores onto query points."""
from __future__ import annotations

from typing import Optional

import numpy as np


def gaussian_project(
    points: np.ndarray,
    centers: np.ndarray,
    scores: np.ndarray,
    sigma: float = 1.0,
    valid: Optional[np.ndarray] = None,
    include_prefactor: bool = True,
) -> np.ndarray:
    """Project slot scores to dense points with a Gaussian kernel.

    a(x) = c * sum_i exp(-||x - z_i||^2 / (2 sigma^2)) s_i
               / sum_i exp(-||x - z_i||^2 / (2 sigma^2))

    with c = 1 / (sqrt(2 pi) sigma) when ``include_prefactor`` is True.

    Parameters
    ----------
    points : [N, 3] query points
    centers : [K, 3] scored slot centres
    scores : [K] patch scores (NaN entries are excluded from both sums)
    sigma : kernel bandwidth in the registered frame
    valid : optional [K] mask; combined with finite scores
    include_prefactor : keep the implementation prefactor (not a probability)

    Returns
    -------
    [N] projected anomaly evidence. Slots with empty support leave NaN.
    """
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if centers.shape[0] != scores.shape[0]:
        raise ValueError("centers and scores must have the same length")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool).reshape(-1)
        if valid.shape[0] != scores.shape[0]:
            raise ValueError("valid mask must match scores")
    else:
        valid = np.ones(scores.shape[0], dtype=bool)

    live = valid & np.isfinite(scores)
    out = np.full(points.shape[0], np.nan, dtype=np.float64)
    if not np.any(live):
        return out

    z = centers[live]
    s = scores[live]
    # [N, K] squared distances
    diff = points[:, None, :] - z[None, :, :]
    sq = np.einsum("nkd,nkd->nk", diff, diff)
    w = np.exp(-sq / (2.0 * sigma * sigma))
    denom = w.sum(axis=1)
    numer = w @ s
    ok = denom > 0
    vals = np.full(points.shape[0], np.nan, dtype=np.float64)
    vals[ok] = numer[ok] / denom[ok]
    if include_prefactor:
        vals = vals / (np.sqrt(2.0 * np.pi) * sigma)
    out[:] = vals
    return out
