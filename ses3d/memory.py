"""Registered local support: location-indexed normal exemplar memory.

A slot ``i`` is a registered local region (not a raw point index). For each slot
the bank stores one feature per available normal template, or NaN if that
template has no valid feature at the slot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


def unit_rows(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """L2-normalise rows; zero-norm rows become NaN."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    out = np.full_like(x, np.nan)
    live = n.reshape(-1) > eps
    out[live] = x[live] / n[live]
    return out


def valid_feature_mask(bank: np.ndarray) -> np.ndarray:
    """True where a reference vector is finite and nonzero.

    ``bank`` is ``[..., d]``. Returns a boolean mask over the leading axes.
    """
    bank = np.asarray(bank, dtype=np.float64)
    finite = np.isfinite(bank).all(axis=-1)
    nonzero = np.linalg.norm(bank, axis=-1) > 0
    return finite & nonzero


@dataclass
class SlotMemory:
    """Location-indexed memory of same-slot normal features.

    Attributes
    ----------
    query : [S, d]
        Finite query patch features (invalid rows may be NaN).
    bank : [S, T, d]
        Normal template features at matching slots. Missing entries are NaN.
    centers : [S, 3] or None
        Optional registered point centres used by Gaussian projection.
    """

    query: np.ndarray
    bank: np.ndarray
    centers: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.query = np.asarray(self.query, dtype=np.float64)
        self.bank = np.asarray(self.bank, dtype=np.float64)
        if self.query.ndim != 2:
            raise ValueError("query must be [S, d]")
        if self.bank.ndim != 3 or self.bank.shape[0] != self.query.shape[0]:
            raise ValueError("bank must be [S, T, d] aligned with query slots")
        if self.bank.shape[2] != self.query.shape[1]:
            raise ValueError("query/bank feature dimension mismatch")
        if self.centers is not None:
            self.centers = np.asarray(self.centers, dtype=np.float64)
            if self.centers.shape[0] != self.query.shape[0]:
                raise ValueError("centers must align with slots")

    @property
    def n_slots(self) -> int:
        return int(self.query.shape[0])

    @property
    def n_templates(self) -> int:
        return int(self.bank.shape[1])

    @classmethod
    def from_templates(
        cls,
        query: np.ndarray,
        templates: Sequence[np.ndarray],
        centers: Optional[np.ndarray] = None,
    ) -> "SlotMemory":
        """Stack per-template ``[S, d]`` features into a ``[S, T, d]`` bank."""
        if len(templates) == 0:
            raise ValueError("need at least one template")
        bank = np.stack([np.asarray(t, dtype=np.float64) for t in templates], axis=1)
        return cls(query=query, bank=bank, centers=centers)

    def reference_mask(self) -> np.ndarray:
        """``[S, T]`` valid-reference mask (finite and nonzero)."""
        return valid_feature_mask(self.bank)

    def query_mask(self) -> np.ndarray:
        """``[S]`` valid-query mask (finite and nonzero)."""
        return valid_feature_mask(self.query)

    def shared_slots(self, require_all_templates: bool = False) -> np.ndarray:
        """Slot indices eligible for scoring.

        Default: finite query and at least one valid same-slot reference.
        If ``require_all_templates``: every native template must be valid, so a
        smaller subset cannot silently widen availability.
        """
        q_ok = self.query_mask()
        ref_ok = self.reference_mask()
        if require_all_templates:
            ref_support = ref_ok.all(axis=1)
        else:
            ref_support = ref_ok.any(axis=1)
        return np.flatnonzero(q_ok & ref_support)


def shared_slot_mask(
    query: np.ndarray,
    bank: np.ndarray,
    require_all_templates: bool = False,
) -> np.ndarray:
    """Boolean ``[S]`` mask of scoreable slots (see :meth:`SlotMemory.shared_slots`)."""
    mem = SlotMemory(query=query, bank=bank)
    mask = np.zeros(mem.n_slots, dtype=bool)
    mask[mem.shared_slots(require_all_templates=require_all_templates)] = True
    return mask


def freeze_subset_mask(full_bank: np.ndarray) -> np.ndarray:
    """Native-bank availability used to freeze the slot mask across subsets."""
    return valid_feature_mask(np.asarray(full_bank, dtype=np.float64)).all(axis=-1)
