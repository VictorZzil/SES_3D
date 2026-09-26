"""Synthetic end-to-end demo: registered slots -> SES map -> point projection.

No external datasets. Useful to verify the public API on a clean machine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ses3d import (
    SlotMemory,
    category_macro,
    gaussian_project,
    score_estimators,
)


def make_synthetic(n_slots: int = 64, n_templates: int = 3, d: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    # normal prototypes per slot
    proto = rng.normal(size=(n_slots, d))
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    templates = []
    for t in range(n_templates):
        noise = 0.05 * rng.normal(size=(n_slots, d))
        feat = proto + noise
        feat /= np.linalg.norm(feat, axis=1, keepdims=True)
        # one missing reference at the first slot of template 0
        if t == 0:
            feat[0] = np.nan
        templates.append(feat)
    query = proto + 0.05 * rng.normal(size=(n_slots, d))
    # inject a defect on slots 20:24
    query[20:24] += 1.5 * rng.normal(size=(4, d))
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    centers = rng.uniform(-1, 1, size=(n_slots, 3))
    labels_slot = np.zeros(n_slots, dtype=np.int64)
    labels_slot[20:24] = 1
    return query, templates, centers, labels_slot


def main() -> None:
    query, templates, centers, labels = make_synthetic()
    mem = SlotMemory.from_templates(query, templates, centers=centers)
    shared = mem.shared_slots()
    print(f"slots={mem.n_slots} templates={mem.n_templates} shared={shared.size}")

    cols = score_estimators(mem.query, mem.bank, slots=shared)
    for name, vec in cols.items():
        defined = int(np.isfinite(vec).sum())
        print(f"  {name:16s} defined={defined}/{vec.size}  "
              f"mean={np.nanmean(vec):.4f}")

    ses = cols["ses"]
    # project onto a dense grid of synthetic points near the centres
    pts = centers[shared] + 0.01 * np.random.default_rng(1).normal(size=(shared.size, 3))
    dense = gaussian_project(pts, centers[shared], ses, sigma=1.0, include_prefactor=True)
    print(f"projected points: finite={np.isfinite(dense).sum()}/{dense.size}")

    # toy category-macro on one category
    y = labels[shared]
    metrics = category_macro([y], [ses], names=["synthetic"])
    print(f"P_AUROC={metrics['P_AUROC']:.4f}  P_AP={metrics['P_AP']:.4f}")


if __name__ == "__main__":
    main()
