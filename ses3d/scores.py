"""Few-template score estimators on registered same-slot normal features.

All estimators consume the same geometry: L2-normalised query features and
L2-normalised valid same-slot normal references. Higher scores mean more
anomalous. Undefined slots are NaN (never silently zero) unless a caller
explicitly requests the legacy zero-fallback convention.

Estimators
----------
SES / EXEMPLAR_MIN
    min_t (1 - cos(q, r_t))                         -- Spatial Exemplar Scoring
MEAN_DISTANCE
    mean_t (1 - cos(q, r_t))
CENTROID_DISTANCE
    1 - cos(q, normalise(mean_t normalise(r_t)))
RAW_EFFECT
    |mean(D) - mean(P)| / sqrt(v_pool)              -- pooled Cohen-style effect
SHRUNK_EFFECT
    |mean(D) - mu_P| / sqrt(v_tilde)                -- variance-shrunk effect
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .memory import unit_rows, valid_feature_mask

NU0 = 2.0
CENTROID_MIN_NORM = 1e-12
VAR_FLOOR_RATIO = 1e-12

EXEMPLAR_MIN = "ses"
MEAN_DISTANCE = "mean"
CENTROID_DISTANCE = "centroid"
RAW_EFFECT = "effect"
SHRUNK_EFFECT = "shrink"
COMMON_DEFINED_SES = "ses_common"
ESTIMATOR_NAMES = (
    EXEMPLAR_MIN,
    MEAN_DISTANCE,
    CENTROID_DISTANCE,
    RAW_EFFECT,
    SHRUNK_EFFECT,
    COMMON_DEFINED_SES,
)


def slot_geometry(
    query: np.ndarray,
    bank: np.ndarray,
    slots: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cosine-distance geometry on selected slots.

    Parameters
    ----------
    query : [S, d]
    bank : [S, T, d]
    slots : optional index array into ``S``. Defaults to all slots.

    Returns
    -------
    dist : [K, T]  1 - cos(q, r_t), NaN where the reference is invalid
    q_hat : [K, d]
    r_hat : [K, T, d]
    ok : [K, T] valid-reference mask
    """
    query = np.asarray(query, dtype=np.float64)
    bank = np.asarray(bank, dtype=np.float64)
    if slots is None:
        slots = np.arange(query.shape[0], dtype=np.int64)
    else:
        slots = np.asarray(slots, dtype=np.int64).reshape(-1)

    q = query[slots]
    refs = bank[slots]
    ok = valid_feature_mask(refs)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    q_ok = qn.reshape(-1) > 0
    q_hat = np.full_like(q, np.nan)
    q_hat[q_ok] = q[q_ok] / qn[q_ok]
    rn = np.linalg.norm(refs, axis=2, keepdims=True)
    r_hat = np.full_like(refs, np.nan)
    live = ok[..., None]
    r_hat = np.where(live, refs / np.where(rn > 0, rn, 1.0), np.nan)
    dots = np.einsum("kd,ktd->kt", q_hat, r_hat)
    dist = np.where(ok, 1.0 - dots, np.nan)
    return dist, q_hat, r_hat, ok


def ses_scores(
    query: np.ndarray,
    bank: np.ndarray,
    slots: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Spatial Exemplar Scoring: nearest valid same-slot normal distance."""
    dist, _, _, ok = slot_geometry(query, bank, slots)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmin(dist, axis=1) if dist.size else np.full(dist.shape[0], np.nan)
    return np.where(ok.any(axis=1), out, np.nan)


def mean_scores(
    query: np.ndarray,
    bank: np.ndarray,
    slots: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mean query--reference cosine distance."""
    dist, _, _, ok = slot_geometry(query, bank, slots)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(dist, axis=1) if dist.size else np.full(dist.shape[0], np.nan)
    return np.where(ok.any(axis=1), out, np.nan)


def centroid_scores(
    query: np.ndarray,
    bank: np.ndarray,
    slots: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Distance to the renormalised mean of unit same-slot references."""
    _, q_hat, r_hat, ok = slot_geometry(query, bank, slots)
    k = ok.shape[0]
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    m = ok.sum(axis=1)
    summed = np.where(ok[..., None], r_hat, 0.0).sum(axis=1)
    mean_vec = summed / np.where(m > 0, m, 1)[:, None]
    norm = np.linalg.norm(mean_vec, axis=1)
    live = (norm >= CENTROID_MIN_NORM) & np.isfinite(q_hat).all(axis=1)
    proto = np.full_like(mean_vec, np.nan)
    proto[live] = mean_vec[live] / norm[live, None]
    cos = np.einsum("kd,kd->k", q_hat, proto)
    out = np.full(k, np.nan)
    out[live] = 1.0 - cos[live]
    return out


def _pairwise_ref_stats(
    r_hat: np.ndarray, ok: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (n_pairs, mean_pair_dist, var_pair_dist) per slot."""
    k, t, _ = r_hat.shape
    pairs = [(i, j) for i in range(t) for j in range(i + 1, t)]
    if not pairs:
        return (
            np.zeros(k, dtype=np.int64),
            np.full(k, np.nan),
            np.full(k, np.nan),
        )
    pi = np.array([p[0] for p in pairs], dtype=np.int64)
    pj = np.array([p[1] for p in pairs], dtype=np.int64)
    both = ok[:, pi] & ok[:, pj]
    gram = np.einsum("ktd,kud->ktu", r_hat, r_hat, optimize=True)
    pdist = np.where(both, 1.0 - gram[:, pi, pj], np.nan)
    n = both.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nansum(pdist, axis=1) / np.where(n > 0, n, 1)
        sq = np.where(both, (pdist - mu[:, None]) ** 2, 0.0)
        var = np.where(n >= 2, sq.sum(axis=1) / np.maximum(n - 1, 1), np.nan)
    mu = np.where(n > 0, mu, np.nan)
    return n, mu, var


def raw_effect_scores(
    query: np.ndarray,
    bank: np.ndarray,
    slots: Optional[np.ndarray] = None,
    zero_fallback: bool = False,
) -> np.ndarray:
    """Pooled Cohen-style effect on scalar cosine distances.

    Requires at least three valid references and positive finite pooled
    variance. If ``zero_fallback`` is True, undefined slots become 0.0 (legacy
    deployed convention); otherwise they stay NaN.
    """
    dist, _, r_hat, ok = slot_geometry(query, bank, slots)
    k = ok.shape[0]
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    nd = ok.sum(axis=1)
    d = np.where(ok, dist, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_d = np.nansum(d, axis=1) / np.where(nd > 0, nd, 1)
        sq_d = np.where(ok, (d - mean_d[:, None]) ** 2, 0.0)
        var_d = np.where(nd >= 2, sq_d.sum(axis=1) / np.maximum(nd - 1, 1), np.nan)
        n_p, mean_p, var_p = _pairwise_ref_stats(r_hat, ok)
        # pooled variance with sample sizes |D|=nd, |P|=n_p
        numer = (np.maximum(nd - 1, 0) * np.where(np.isfinite(var_d), var_d, 0.0)
                 + np.maximum(n_p - 1, 0) * np.where(np.isfinite(var_p), var_p, 0.0))
        denom = nd + n_p - 2
        v_pool = np.where(denom > 0, numer / np.maximum(denom, 1), np.nan)
        score = np.abs(mean_d - mean_p) / np.sqrt(v_pool)

    defined = (nd >= 3) & (n_p >= 1) & np.isfinite(score) & (v_pool > 0)
    # classical pooled form also needs both sample variances when both sides exist
    defined &= np.isfinite(mean_d) & np.isfinite(mean_p)
    out = np.full(k, np.nan)
    out[defined] = score[defined]
    if zero_fallback:
        out = np.where(defined, score, 0.0)
    return out


def normal_pair_prior(bank: np.ndarray) -> Tuple[float, float]:
    """Normal-only prior ``(mu0, v0)`` from pairwise co-located template distances.

    Built from the normal bank only (all slots). Never uses queries or labels.
    """
    bank = np.asarray(bank, dtype=np.float64)
    pairs = []
    for s in range(bank.shape[0]):
        refs = bank[s]
        keep = valid_feature_mask(refs)
        if int(keep.sum()) < 2:
            continue
        u = unit_rows(refs[keep])
        for i in range(u.shape[0]):
            for j in range(i + 1, u.shape[0]):
                pairs.append(1.0 - float(np.dot(u[i], u[j])))
    if len(pairs) < 2:
        raise ValueError("normal bank yields too few pairs for a global prior")
    a = np.asarray(pairs, dtype=np.float64)
    mu0 = float(a.mean())
    v0 = float(np.var(a, ddof=1))
    if not (np.isfinite(v0) and v0 > 0):
        raise ValueError(f"prior variance must be positive and finite, got {v0}")
    return mu0, v0


def shrunk_effect_scores(
    query: np.ndarray,
    bank: np.ndarray,
    slots: Optional[np.ndarray] = None,
    mu0: Optional[float] = None,
    v0: Optional[float] = None,
    nu0: float = NU0,
) -> np.ndarray:
    """Variance-shrunk effect size; finite whenever the slot has a reference.

    ``v_tilde = (nu_P v_P + nu_D v_D + nu0 v0) / (nu_P + nu_D + nu0)`` with
    ``nu_D = max(|D|-1, 0)``, ``nu_P = max(|P|-1, 0)``. Missing sample variances
    fall back to the normal-only prior ``v0``.
    """
    if mu0 is None or v0 is None:
        mu0, v0 = normal_pair_prior(bank)
    dist, _, r_hat, ok = slot_geometry(query, bank, slots)
    k = ok.shape[0]
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    nd = ok.sum(axis=1)
    d = np.where(ok, dist, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_d = np.nansum(d, axis=1) / np.where(nd > 0, nd, 1)
        sq_d = np.where(ok, (d - mean_d[:, None]) ** 2, 0.0)
        var_d = np.where(nd >= 2, sq_d.sum(axis=1) / np.maximum(nd - 1, 1), v0)
        n_p, mean_p_raw, var_p_raw = _pairwise_ref_stats(r_hat, ok)
        mean_p = np.where(n_p > 0, mean_p_raw, mu0)
        var_p = np.where(n_p >= 2, var_p_raw, v0)
        nu_p = np.maximum(n_p - 1, 0)
        nu_d = np.maximum(nd - 1, 0)
        v_tilde = (nu_p * var_p + nu_d * var_d + nu0 * v0) / (nu_p + nu_d + nu0)
        score = np.abs(mean_d - mean_p) / np.sqrt(np.maximum(v_tilde, VAR_FLOOR_RATIO * v0))
    return np.where(nd > 0, score, np.nan)


def score_estimators(
    query: np.ndarray,
    bank: np.ndarray,
    slots: Optional[np.ndarray] = None,
    include_zero_fallback: bool = True,
) -> Dict[str, np.ndarray]:
    """All paper estimators on the same slots.

    Keys: ``ses``, ``mean``, ``centroid``, ``effect``, ``shrink``, ``ses_common``.
    ``ses_common`` is SES restricted to the raw-effect defined support (the
    common-defined control that removes coverage as an explanation).
    """
    effect = raw_effect_scores(query, bank, slots, zero_fallback=False)
    ses = ses_scores(query, bank, slots)
    out = {
        EXEMPLAR_MIN: ses,
        MEAN_DISTANCE: mean_scores(query, bank, slots),
        CENTROID_DISTANCE: centroid_scores(query, bank, slots),
        RAW_EFFECT: effect,
        SHRUNK_EFFECT: shrunk_effect_scores(query, bank, slots),
        COMMON_DEFINED_SES: np.where(np.isfinite(effect), ses, np.nan),
    }
    if include_zero_fallback:
        out["effect_legacy0"] = np.where(np.isfinite(effect), effect, 0.0)
    return out
