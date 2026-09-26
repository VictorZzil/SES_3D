"""Point-level metrics used in the paper (category-macro AUROC / AP)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC with mid-ranks for ties. NaNs are dropped."""
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    pos = labels > 0
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    sum_pos_ranks = ranks[pos].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary average precision (area under the precision-recall step curve)."""
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    n_pos = int((labels > 0).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order] > 0
    tp = np.cumsum(sorted_labels)
    precision = tp / np.arange(1, scores.size + 1)
    return float(precision[sorted_labels].sum() / n_pos)


def category_macro(
    labels: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    names: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Macro-average AUROC/AP over categories (equal category weight).

    Each category contributes one concatenated point set, matching the paper's
    evaluation unit. Points within a category are not treated as independent
    replicates for uncertainty; this helper only aggregates metrics.
    """
    if len(labels) != len(scores):
        raise ValueError("labels/scores length mismatch")
    aurocs: List[float] = []
    aps: List[float] = []
    per: Dict[str, Tuple[float, float]] = {}
    for i, (y, s) in enumerate(zip(labels, scores)):
        a = roc_auc(y, s)
        p = average_precision(y, s)
        aurocs.append(a)
        aps.append(p)
        if names is not None:
            per[names[i]] = (a, p)
    out = {
        "P_AUROC": float(np.nanmean(aurocs)) if aurocs else float("nan"),
        "P_AP": float(np.nanmean(aps)) if aps else float("nan"),
        "n_categories": float(len(labels)),
    }
    if names is not None:
        out["per_category"] = per  # type: ignore[assignment]
    return out
