"""Measure EFF rank reversals on normal/defect centers in identical Real3D slots."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np


OUTCOMES = ("both_right", "corrected_by_sesad", "reversed_by_sesad", "both_reversed", "tied")
NUMERATOR = "corrected_with_smaller_effect_numerator"
TIE_TOL = 1e-12


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values).all(axis=-1)
    values = np.where(finite[..., None], values, 0)
    norm = np.linalg.norm(values, axis=-1)
    good = finite & (norm > 0)
    return values / np.where(good, norm, 1)[..., None], good


def bank_setup(features, template_indices):
    bank, valid = normalize(features)
    valid &= np.asarray(template_indices) >= 0
    n_refs = bank.shape[1]
    signature = valid @ (2 ** np.arange(n_refs))
    support = valid.sum(axis=1).astype(np.uint8)
    parts = []
    for bits in sorted(set(signature) - {0}):
        idx = np.flatnonzero(signature == bits)
        ids = np.flatnonzero((bits >> np.arange(n_refs)) & 1)
        memory = bank[idx][:, ids]
        pairs = None
        if len(ids) >= 3:
            ij = np.asarray(list(combinations(range(len(ids)), 2)))
            pairs = 1 - np.einsum("spd,spd->sp", memory[:, ij[:, 0]], memory[:, ij[:, 1]])
        parts.append((idx, memory, pairs))
    return support, parts


def score_query(query, centers, point_labels, setup):
    q, valid = normalize(query)
    support, parts = setup
    cs = np.asarray(centers)
    labels = np.asarray(point_labels).reshape(-1)
    present = valid & (cs >= 0) & (cs < len(labels))
    target = np.full(len(cs), -1, dtype=np.int8)
    target[present] = labels[cs[present]] > 0
    out = np.full((len(cs), 3), np.nan)
    for idx, memory, pairs in parts:
        qdist = 1 - np.einsum("sd,srd->sr", q[idx], memory)
        out[idx, 0] = qdist.min(axis=1)
        if pairs is not None:
            nq, npair = qdist.shape[1], pairs.shape[1]
            vp = ((nq - 1) * qdist.var(axis=1, ddof=1) + (npair - 1) * pairs.var(axis=1, ddof=1)) / (nq + npair - 2)
            eff = np.divide(abs(qdist.mean(axis=1) - pairs.mean(axis=1)), np.sqrt(vp),
                            out=np.full(len(idx), np.nan), where=vp > 0)
            out[idx, 1:] = np.column_stack((eff, vp))
    out[~present] = np.nan
    return target, out


def save_chunk(args, files, repo, category, token, names, cache, cfg, manifest):
    from e62.scoring import official_patch_scores
    from e68.slot_memory import common_slot_set, slot_nn_scores
    from e90_real.score import build_context

    path = args.output / "slots" / category / f"{token}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with np.load(cache, allow_pickle=False) as z:
        ctx = build_context(category, z, cfg)
        assert ctx.bank_sha256 == manifest["bank_sha256"] and ctx.template_ids == manifest["template_ids"]
        refs = sorted(k[:-3] for k in z.files if k.startswith("n") and k.endswith("_id"))
        assert [z[f"{k}_id"].item() for k in refs] == list(ctx.template_ids)
        setup = bank_setup(ctx.ff, np.stack([z[f"{k}_cf"] for k in refs], axis=1))
        ids = {str(z[k].item()): k[:-3] for k in z.files if k.startswith("t") and k.endswith("_id")}
        for name in names:
            g = ids[name]
            cf = z[f"{g}_cf"]
            query = z[f"{g}_ff"]
            labels = z[f"{g}_gt"].reshape(-1)
            target, scores = score_query(query, cf, labels, setup)
            with np.load(repo / "results/e90_real/per_sample" / category / name, allow_pickle=False) as old:
                assert np.array_equal(labels, old["gt"]) and str(old["bank_sha256"].item()) == ctx.bank_sha256
                assert len(z[f"{g}_reg"]) == int(old["n_points"])
                assert abs(float(z[f"{g}_obj"]) - float(old["object_native"])) <= 1e-6
                S, _ = common_slot_set(query, ctx.ff, np.flatnonzero(cf >= 0), ctx.perm)
                assert len(S) == int(old["n_common_slots"])
                np.testing.assert_allclose(scores[S, 0], slot_nn_scores(query, ctx.ff, S), atol=1e-12, rtol=0)
                k = np.flatnonzero(np.isfinite(scores[:, 1]))
                np.testing.assert_allclose(scores[k, 1], official_patch_scores(ctx.ostats, query[k], k),
                                           atol=1e-8, rtol=1e-8)
                records.append((target, scores, int(old["label"]), str(old["object_id"].item()),
                                bool(old["is_cut"]), bool(old["is_copy"]), name))
    y, v, obj_label, obj, cut, copy, names = zip(*records)
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, target=y, scores=v, obj_label=obj_label, obj=obj, cut=cut, copy=copy,
                            names=names, support=setup[0], bank=ctx.bank_sha256)
    path.with_suffix(".tmp").replace(path)
    print(json.dumps({"category": category, "chunk": token, "scans": len(records),
                      "cached_input": str(cache) in files}), flush=True)


def compute(args):
    sys.path.insert(0, str(args.replay_helpers.resolve()))
    import torch
    from sesad_full_real_calibration import chunk_plan, environment, extract_one, safe_remove

    root, repo, official, _, cfg = environment(args)
    torch.set_num_threads(2)
    files = set()
    if args.cache_inventory:
        files = {line.strip() for line in args.cache_inventory.read_text().splitlines() if line.strip()}
    cache_files = defaultdict(list)
    for line in sorted(files):
        p = Path(line)
        if p.exists():
            cache_files[p.name].append(p)
    paths = args.output / "slots"
    paths.mkdir(parents=True, exist_ok=True)
    categories = [args.category] if args.category else sorted(p.name for p in (repo / "results/e90_real/per_sample").iterdir() if p.is_dir())
    assert len(categories) == (1 if args.category else 12)

    def locate(cls, token, names):
        for candidate in cache_files[f"features_Real3D_AD_{cls}_SMALL.npz"]:
            try:
                with np.load(candidate, allow_pickle=False) as cache:
                    stems = {str(cache[k].item()) for k in cache.files if k.startswith("t") and k.endswith("_id")}
                # Prefer the exact original token. Cache from a different run
                # cannot be substituted just because it has IDs in common.
                if token in candidate.parts and set(names).issubset(stems):
                    return candidate
            except (ValueError, OSError):
                continue
        return None

    for cls in categories:
        plan = chunk_plan(repo, cls)
        if args.pilot_chunks:
            plan = plan[:args.pilot_chunks]
        for token, manifest, names in plan:
            if (paths / cls / f"{token}.npz").exists():
                continue
            cached = locate(cls, token, names)
            if cached:
                save_chunk(args, files, repo, cls, token, names, cached, cfg, manifest)
                continue
            cache, view_cfg = extract_one(args, repo, official, cfg, cls, token, manifest)
            save_chunk(args, files, repo, cls, token, names, cache, view_cfg, manifest)
            safe_remove(cache.parent, args.scratch)
            safe_remove(Path(view_cfg["cossad_data_root"]), args.scratch)
        for path in (args.scratch / "stage" / "train").glob(f"real3dad_{cls}_*.npz"):
            path.unlink()


def average_precision(labels, scores):
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
    tp = labels[order].cumsum()[ends]
    return float(np.sum(tp / (ends + 1) * np.diff(tp, prepend=0) / tp[-1]))


def count_pairs(values, result):
    neg, pos = values
    if not len(neg) or not len(pos):
        return
    se, ee = (pos[:, None, :2] - neg[None, :, :2]).transpose(2, 0, 1)
    tie = (abs(se) <= TIE_TOL) | (abs(ee) <= TIE_TOL)
    corrected = (se > TIE_TOL) & (ee < -TIE_TOL)
    counts = (np.count_nonzero((se > TIE_TOL) & (ee > TIE_TOL)),
              np.count_nonzero(corrected),
              np.count_nonzero((se < -TIE_TOL) & (ee > TIE_TOL)),
              np.count_nonzero((se < -TIE_TOL) & (ee < -TIE_TOL)), np.count_nonzero(tie))
    assert sum(counts) == len(neg) * len(pos)
    for key, value in zip(OUTCOMES, counts):
        result[key] += int(value)
    pos_numerator = pos[:, 1] * np.sqrt(pos[:, 2])
    neg_numerator = neg[:, 1] * np.sqrt(neg[:, 2])
    result[NUMERATOR] += int(np.count_nonzero(corrected & (pos_numerator[:, None] + TIE_TOL < neg_numerator[None, :])))


def summarize(output, allow_partial=False):
    totals = defaultdict(lambda: Counter({k: 0 for k in (*OUTCOMES, NUMERATOR)}))
    per_class = []
    seen = set()
    for category in sorted((output / "slots").iterdir()):
        chunks = [dict(np.load(p, allow_pickle=False)) for p in sorted(category.glob("*.npz"))]
        assert len({chunk["bank"].item() for chunk in chunks}) == 1
        support = chunks[0]["support"]
        assert all(np.array_equal(support, c["support"]) for c in chunks)
        combined = {k: np.concatenate([p[k] for p in chunks]) for k in
                    ("target", "scores", "obj_label", "obj", "cut", "copy", "names")}
        before = len(seen)
        seen.update(combined["names"].tolist())
        assert len(seen) - before == len(combined["names"])
        # Preserve original scans as the primary unit. The secondary cropped
        # stratum is scored separately, and cut/copy variants never pool as units.
        for cut, scope in ((False, "uncropped"), (True, "cropped")):
            keep = (~combined["copy"]) & (combined["cut"] == cut)
            targets = combined["target"][keep]
            scores = combined["scores"][keep]
            positive_objects = combined["obj_label"][keep] > 0
            objects = combined["obj"][keep]
            for s in range(5):
                counts = Counter({k: 0 for k in (*OUTCOMES, NUMERATOR)})
                point_values, point_labels = [], []
                paired_slots = 0
                for slot in np.flatnonzero(support == s):
                    score, target = scores[:, slot], targets[:, slot]
                    valid = np.isfinite(score[:, :2]).all(axis=1)
                    neg = valid & (~positive_objects) & (target == 0)
                    pos = valid & positive_objects & (target == 1)
                    if not neg.any() or not pos.any():
                        continue
                    paired_slots += 1
                    # Count strict same-slot contrasts only across different
                    # source-object IDs, even when official labels share a group.
                    for obj in np.unique(objects[pos]):
                        count_pairs((score[neg & (objects != obj)], score[pos & (objects == obj)]), counts)
                    choose = neg | pos
                    point_values.append(score[choose, :2])
                    point_labels.append(target[choose])
                totals[(scope, s)].update(counts)
                points = np.concatenate(point_values) if point_values else None
                yy = np.concatenate(point_labels) if point_labels else None
                per_class.append({"category": category.name, "scope": scope, "actual_refs": s,
                                  "scans": int(keep.sum()), "reference_slots": int((support == s).sum()),
                                  "query_centers": int((targets[:, support == s] >= 0).sum()),
                                  "labeled_defect_centers": int((targets[:, support == s] == 1).sum()),
                                  "slots_with_both_labels": paired_slots, "pair_counts": dict(counts),
                                  "distinct_scored_centers": len(yy) if yy is not None else 0,
                                  "center_ap": {m: average_precision(yy, p) for m, p in zip(("SESAD", "EFF"), points.T)}
                                  if points is not None else None})
    if not allow_partial:
        assert len(seen) == 1172 and len({r["category"] for r in per_class}) == 12
    result = {"verified_accepted_scans": len(seen), "exploratory_benchmark_reanalysis": True,
              "tolerance": TIE_TOL, "negative_sources": "accepted normal scans only",
              "positive_sources": "annotated defect centers from accepted anomaly scans",
              "pairing": "distinct source objects at the exact same category slot with an identical reference bank, "
              "positive pooled variance for both queries; copies excluded; uncropped and cropped evaluated separately",
              "center_ap": "AP on labeled query centers in slots where both labels are represented; no point projection",
              "effect_numerator": "absolute difference between the query-reference mean and the reference-pair mean",
              "counts": [{"scope": scope, "actual_refs": s, "pairs": sum(c[k] for k in OUTCOMES), **c}
                         for (scope, s), c in sorted(totals.items())], "by_category": per_class}
    temp = output / "slot_reversals.tmp"
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output / "slot_reversals.json")
    print(json.dumps({k: v for k, v in result.items() if k != "by_category"}), flush=True)


def self_check():
    refs = np.array([[[1, 0], [1, 0], [1, 0], [-1, 0]]], float)
    setup = bank_setup(refs, np.zeros((1, 4), int))
    _, good = score_query([[1, 0]], [0], [0], setup)
    _, anomaly = score_query([[0, 1]], [0], [1], setup)
    np.testing.assert_allclose(good[0], [0, np.sqrt(2) / 3, 9 / 8])
    np.testing.assert_allclose(anomaly[0], [1, 0, 3 / 4])
    result = Counter()
    count_pairs((good, anomaly), result)
    assert result["corrected_by_sesad"] == sum(result[k] for k in OUTCOMES) == result[NUMERATOR] == 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("compute", "summarize", "self-check"))
    p.add_argument("output", nargs="?", type=Path)
    p.add_argument("--scratch", type=Path)
    p.add_argument("--replay-helpers", type=Path)
    p.add_argument("--cache-inventory", type=Path)
    p.add_argument("--category")
    p.add_argument("--pilot-chunks", type=int, default=0)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    self_check()
    if args.mode == "compute":
        import fcntl

        assert args.replay_helpers and args.scratch and args.scratch.resolve().parent == Path("/dev/shm")
        assert args.category and args.category.isalpha()
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"{args.category}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            compute(args)
    if args.mode == "summarize":
        summarize(args.output, args.allow_partial)


if __name__ == "__main__":
    main()
