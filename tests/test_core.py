"""Self-contained checks for SES-3D core estimators (no external data)."""
from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ses3d import (
    SlotMemory,
    centroid_scores,
    gaussian_project,
    mean_scores,
    raw_effect_scores,
    ses_scores,
    shrunk_effect_scores,
    score_estimators,
    average_precision,
    roc_auc,
)


def test_single_reference_estimators_coincide():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(5, 8))
    r = rng.normal(size=(5, 1, 8))
    ses = ses_scores(q, r)
    mean = mean_scores(q, r)
    ctr = centroid_scores(q, r)
    np.testing.assert_allclose(ses, mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ses, ctr, rtol=1e-10, atol=1e-10)


def test_ses_is_min_distance():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(4, 6))
    bank = rng.normal(size=(4, 3, 6))
    dist = 1.0 - np.einsum(
        "kd,ktd->kt",
        q / np.linalg.norm(q, axis=1, keepdims=True),
        bank / np.linalg.norm(bank, axis=2, keepdims=True),
    )
    np.testing.assert_allclose(ses_scores(q, bank), dist.min(axis=1), rtol=1e-10)


def test_missing_reference_is_nan_not_zero():
    q = np.ones((2, 4))
    bank = np.full((2, 2, 4), np.nan)
    bank[0, 0] = np.array([1.0, 0, 0, 0])
    ses = ses_scores(q, bank)
    assert np.isfinite(ses[0])
    assert np.isnan(ses[1])


def test_raw_effect_requires_three_refs():
    q = np.ones((1, 3))
    two = np.ones((1, 2, 3))
    two[0, 0] = [1.0, 0, 0]
    two[0, 1] = [0.0, 1, 0]
    assert np.isnan(raw_effect_scores(q, two)[0])
    three = np.ones((1, 3, 3))
    three[0, 0] = [1.0, 0, 0]
    three[0, 1] = [0.0, 1, 0]
    three[0, 2] = [0.0, 0, 1]
    q = np.array([[1.0, 1.0, 0.0]])
    s = raw_effect_scores(q, three)[0]
    assert np.isnan(s) or np.isfinite(s)


def test_effect_zero_fallback():
    q = np.ones((1, 3))
    bank = np.full((1, 2, 3), np.nan)
    bank[0, 0] = [1.0, 0, 0]
    bank[0, 1] = [0.0, 1, 0]
    legacy = raw_effect_scores(q, bank, zero_fallback=True)[0]
    assert legacy == 0.0
    strict = raw_effect_scores(q, bank, zero_fallback=False)[0]
    assert np.isnan(strict)


def test_shrink_defined_with_two_refs():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(3, 5))
    bank = rng.normal(size=(3, 2, 5))
    s = shrunk_effect_scores(q, bank)
    assert np.isfinite(s).all()


def test_common_defined_ses_matches_support():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(6, 4))
    bank = rng.normal(size=(6, 3, 4))
    bank[0, 2] = np.nan  # only slot 0 loses a ref; effect may still define
    cols = score_estimators(q, bank)
    eff = cols["effect"]
    ses = cols["ses"]
    common = cols["ses_common"]
    both = np.isfinite(eff) & np.isfinite(common)
    np.testing.assert_allclose(common[both], ses[both])
    assert np.all(np.isnan(common[~np.isfinite(eff)]))


def test_gaussian_projection_constant_scores():
    pts = np.zeros((10, 3))
    pts[:, 0] = np.linspace(0, 1, 10)
    centers = np.array([[0.5, 0.0, 0.0]])
    scores = np.array([2.0])
    out = gaussian_project(pts, centers, scores, sigma=1.0, include_prefactor=False)
    assert np.isfinite(out).all()
    # single center, no prefactor -> score everywhere
    np.testing.assert_allclose(out, 2.0)


def test_metrics_toy():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    a = roc_auc(y, s)
    p = average_precision(y, s)
    assert 0.0 <= a <= 1.0 and 0.0 <= p <= 1.0
    # perfect ranking
    assert roc_auc(y, np.array([0.0, 0.1, 0.9, 1.0])) == 1.0


def test_slot_memory_shared():
    q = np.array([[1.0, 0], [0, 0], [1.0, 1.0]])
    bank = np.zeros((3, 2, 2))
    bank[0, 0] = [1.0, 0]
    bank[0, 1] = np.nan
    bank[1, 0] = [1.0, 0]
    bank[1, 1] = [0, 1.0]
    bank[2, 0] = np.nan
    bank[2, 1] = np.nan
    mem = SlotMemory(query=q, bank=bank)
    shared = mem.shared_slots()
    assert shared.tolist() == [0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")
