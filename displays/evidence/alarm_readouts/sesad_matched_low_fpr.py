"""Compare three frozen Real3D readouts at equal FPR and under held-out crop shift."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from sesad_alarm_readout_comparison import ARMS, alarm_threshold, write_csv
from sesad_alarm_thresholds import digest, group_key, metrics, read_csv, scan_key, write_json


CAPS = (0.01, 0.05, 0.10)


def roc_at_caps(score, target, weights):
    order = np.argsort(-score, kind="stable")
    end = np.r_[np.flatnonzero(np.diff(score[order])), len(order) - 1]
    pos = np.r_[0, np.cumsum(weights[order] * target[order])[end]]
    neg = np.r_[0, np.cumsum(weights[order] * (1 - target[order]))[end]]
    if not pos[-1] or not neg[-1]:
        return None
    fpr, tpr = neg / neg[-1], pos / pos[-1]
    # Use the rightmost point at a repeated FPR; interpolate to the first point
    # above the target. Direct empirical thresholds use the lower endpoint.
    index = np.searchsorted(fpr, CAPS, side="right") - 1
    fraction = (np.asarray(CAPS) - fpr[index]) / (fpr[index + 1] - fpr[index])
    return np.stack((tpr[index], tpr[index] + fraction * (tpr[index + 1] - tpr[index]), fpr[index]))


def matched_roc(rows, scores, repeats=2000):
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["category"]].append(row)
    rng = np.random.default_rng(20261117)
    estimates, sampled, detail = [], [], []
    for category, members in sorted(by_class.items()):
        group_index = {key: i for i, key in enumerate(sorted(set(map(group_key, members))))}
        tags = np.array([group_index[group_key(row)] for row in members])
        n_groups = len(group_index)
        y = np.array([int(row["label"]) for row in members])
        values = np.array([scores[scan_key(row)] for row in members]).T
        arms = np.array([roc_at_caps(score, y, np.ones(len(y), int)) for score in values])
        estimates.append(arms)
        boot = []
        until = 0
        while len(boot) < repeats:
            weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups))[tags]
            if (weights @ y) and (weights @ (1 - y)):
                boot.append([roc_at_caps(score, y, weights)[1] for score in values])
            until += 1
            if until > repeats * 4:
                raise ValueError(f"Too many single-label resamples in {category}")
        sampled.append(np.array(boot))
        for method, result in zip(ARMS, arms):
            for col, fpr in enumerate(CAPS):
                detail.append({"category": category, "scored_scans": len(y), "object_groups": n_groups,
                               "normal_scans": int((y == 0).sum()), "readout": method, "target_fpr": fpr,
                               "empirical_recall_at_cap": result[0, col], "equal_fpr_recall": result[1, col],
                               "empirical_actual_fpr": result[2, col]})
    macro = np.mean(estimates, axis=0)
    bootstrap = np.mean(sampled, axis=0)
    outcomes = {}
    for col, fpr in enumerate(CAPS):
        interval = {}
        for i, name in enumerate(ARMS[:2]):
            gaps = bootstrap[:, 2, col] - bootstrap[:, i, col]
            interval[name] = {"recall_gain": macro[2, 1, col] - macro[i, 1, col],
                              "group_bootstrap_percentile_95": np.quantile(gaps, [0.025, 0.975]).tolist()}
        outcomes[str(fpr)] = {"by_readout": {name: {"empirical_recall_at_cap": macro[i, 0, col],
                                                 "equal_fpr_recall": macro[i, 1, col],
                                                 "empirical_actual_fpr": macro[i, 2, col]}
                                             for i, name in enumerate(ARMS)},
                              "spatial_minus_comparator": interval}
    return outcomes, detail


def frozen_crop_shift(manifest, scores):
    calibration = defaultdict(list)
    family = defaultdict(dict)
    fixed_heldout = [r for r in manifest if r["split"] == "evaluation"]
    copies = 0
    for row in manifest:
        variant = row["variant"]
        if row["split"] == "calibration" and variant == "good" and row["status"] == "scored":
            assert row["label"] == "0"
            calibration[row["category"]].append(scores[scan_key(row)])
        if row["split"] != "evaluation" or variant.endswith("_copy"):
            continue
        kind = "cropped" if variant.endswith("_cut") else "uncropped"
        entry = family[(*group_key(row), row["label"], variant.removesuffix("_cut"))]
        assert kind not in entry, (row, entry)
        entry[kind] = row
    assert len(calibration) == 12 and set(map(group_key, fixed_heldout)).isdisjoint(
        group_key(row) for row in manifest if row["split"] == "calibration")
    pairs = [pair for pair in family.values() if set(pair) == {"uncropped", "cropped"}]
    cut = [p["cropped"] for p in pairs]
    raw = [p["uncropped"] for p in pairs]
    scopes = {"uncropped_heldout": raw, "cropped_heldout": cut}
    records = []
    for index, name in enumerate(ARMS):
        thresholds = {c: alarm_threshold(sorted(x[index] for x in v)) for c, v in calibration.items()}
        for scope, all_rows in scopes.items():
            accepted = [r for r in all_rows if r["status"] == "scored"]
            alarms = [scores[scan_key(r)][index] > thresholds[r["category"]] for r in accepted]
            result = metrics(all_rows, accepted, sum(r["label"] == "0" and a for r, a in zip(accepted, alarms)),
                             sum(r["label"] == "1" and not a for r, a in zip(accepted, alarms)))
            records.append({"scope": scope, "readout": name, **result})
    return {"uncropped_normal_calibration": sum(map(len, calibration.values())),
            "calibration_per_class": {c: len(v) for c, v in sorted(calibration.items())},
            "matched_heldout_variant_pairs": len(pairs), "metrics": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split", "scores", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    split = read_csv(args.split)
    records = read_csv(args.scores)
    scores = {scan_key(r): [float(r[m]) for m in ARMS] for r in records}
    accepted = [r for r in split if r["status"] == "scored"]
    assert len(split) == 1206 and len(accepted) == len(scores) == len(records) == 1172
    assert set(scores) == set(map(scan_key, accepted))
    assert {scan_key(r): (r["label"], r["official_file"]) for r in accepted} == {
        scan_key(r): (r["label"], r["official_file"]) for r in records}
    assert np.isfinite(np.array(list(scores.values()))).all()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"source_sha256": {p.name: digest(p) for p in (args.split, args.scores)},
              "retrospective_roc": "Macro-average of class ROC recall evaluated at common actual FPRs. "
              "Linear interpolation uses adjacent empirical operating points. The empirical cap reports the "
              "highest observed recall without crossing the cap. Evaluation labels define both curves.",
              "bootstrap": "2000 paired category-stratified object-group resamples, conditional on accepted scans; "
              "same weights across all three arms. Descriptive percentile intervals.",
              "crop_shift": "Fit thresholds using accepted uncropped normal scans in the original calibration "
              "groups; evaluate matching uncropped and cropped variants of separate held-out objects. "
              "The paired scan variants come from the same benchmark and are not a new physical acquisition batch."}
    by_class = []
    for scope, subset in (("all_accepted", accepted), ("holdout_accepted", [r for r in accepted if r["split"] == "evaluation"])):
        result[scope], detail = matched_roc(subset, scores)
        by_class.extend({"scope": scope, **row} for row in detail)
    result["heldout_crop_shift"] = frozen_crop_shift(split, scores)
    write_json(args.output / "matched_low_fpr.json", result)
    write_csv(args.output / "matched_low_fpr_by_class.csv", by_class)


if __name__ == "__main__":
    main()
