"""Replay the fixed Real3D split and repeated object-group threshold splits."""

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import mean, median
import timeit

from sesad_alarm_thresholds import digest, group_key, metrics, read_csv, scan_key, write_json


ARMS = ("COSSAD_NATIVE", "POINT_MAX", "SPATIAL_EVIDENCE")
N_SPLITS = 200
SALT = "SESAD-TASE-v26-20261117-alarm-repeat-v1"
ENDPOINTS = ("scored_fpr", "scored_miss_rate", "normal_review_rate", "defect_auto_pass_rate")


def export_scores(snapshot_path, baseline_path, output):
    snapshot = json.loads(snapshot_path.read_bytes())
    archive = json.loads(baseline_path.read_bytes())
    assert snapshot["run_complete"] and snapshot["protocol"]["canonical_template_graph"]
    arm = snapshot["protocol"]["metrics"].index("SESAD")
    canonical = snapshot["protocol"]["heads"].index("CANONICAL_EEF")
    peak = snapshot["protocol"]["heads"].index("POINT_MAX")
    rows = []
    for cls, rec in sorted(snapshot["classes"].items()):
        old = {s["id"]: s for s in archive[cls]["samples"]}
        assert len(old) == len(rec["samples"]) and set(old) == {s["id"] for s in rec["samples"]}
        for scan in sorted(rec["samples"], key=lambda s: s["id"]):
            b = old[scan["id"]]
            assert (b["label"], b["points"]) == (scan["label"], scan["points"])
            s = b["object_scores"]
            npair = scan["object_scores"][arm]
            assert abs(s["V22_EEF"]-npair[canonical]) <= 1e-8
            assert abs(s["POINT_MAX"]-npair[peak]) <= 2e-8
            rows.append({"category": cls, "staged_id": scan["id"], "label": scan["label"],
                         "official_file": b["official_file"], "COSSAD_NATIVE": s["COSSAD_NATIVE"],
                         "POINT_MAX": s["POINT_MAX"], "SPATIAL_EVIDENCE": s["V22_EEF"]})
    assert len(rows) == 1172
    write_csv(output, rows)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def grouped_pool(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    eligible = defaultdict(list)
    for key, members in groups.items():
        if all(row["label"] == "0" for row in members):
            eligible[key[0]].append(key)
    assert len(eligible) == 12
    return eligible


def repeat_groups(eligible, index):
    selected = set()
    for keys in eligible.values():
        ranked = sorted(keys, key=lambda k: hashlib.sha256(
            (SALT + f"/{index:03d}/" + "/".join(k)).encode()).hexdigest())
        selected.update(ranked[:len(keys)//2])
    return selected


def alarm_threshold(ordered_scores):
    n = len(ordered_scores)
    return ordered_scores[n - n // 20 - 1]


def evaluate(rows, scores, selected):
    calibration = [r for r in rows if group_key(r) in selected]
    evaluation = [r for r in rows if group_key(r) not in selected]
    assert {group_key(r) for r in calibration} == selected
    assert selected.isdisjoint(map(group_key, evaluation))
    assert all(r["label"] == "0" for r in calibration)
    cal_scores = defaultdict(list)
    for row in calibration:
        if row["status"] == "scored":
            cal_scores[row["category"]].append(scores[scan_key(row)])
    accepted = [r for r in evaluation if r["status"] == "scored"]
    normal = [r for r in accepted if r["label"] == "0"]
    anomaly = [r for r in accepted if r["label"] == "1"]
    assert len(cal_scores) == 12 and len(anomaly) == 593
    assert sum(row["label"] == "1" for row in evaluation) == 602
    output, thresholds, decisions = {}, {}, {}
    for i, name in enumerate(ARMS):
        vals = {c: sorted(s[i] for s in samples) for c, samples in cal_scores.items()}
        limits = {c: alarm_threshold(v) for c, v in vals.items()}
        flagged = lambda r: scores[scan_key(r)][i] > limits[r["category"]]
        output[name] = metrics(evaluation, accepted, sum(flagged(r) for r in normal),
                               sum(not flagged(r) for r in anomaly))
        thresholds[name] = {c: {"threshold": t, "accepted_calibration": len(vals[c]),
                                "calibration_alarms": sum(x > t for x in vals[c])}
                            for c, t in limits.items()}
        decisions[name] = {scan_key(r): ("alarm" if flagged(r) else "pass") for r in accepted}
    info = {"calibration_scans": len(calibration), "calibration_scored": sum(map(len, cal_scores.values())),
            "calibration_groups": len(selected), "evaluation_scans": len(evaluation)}
    return output, thresholds, decisions, info


def quantile(values, q):
    values = sorted(values)
    lo = math.floor((len(values)-1)*q)
    fraction = (len(values)-1)*q-lo
    return values[lo]*(1-fraction)+values[min(lo+1, len(values)-1)]*fraction


def describe(values):
    return {"mean": mean(values), "q10": quantile(values, .1), "q90": quantile(values, .9)}


def replay(split_path, score_path, output):
    rows = read_csv(split_path)
    data = read_csv(score_path)
    zipped = {scan_key(r): r for r in data}
    assert len(rows) == 1206 and len(zipped) == len(data) == 1172
    assert set(zipped) == {scan_key(r) for r in rows if r["status"] == "scored"}
    for row in rows:
        if scan_key(row) in zipped:
            scored = zipped[scan_key(row)]
            assert (scored["label"], scored["official_file"]) == (row["label"], row["official_file"])
    scores = {k: [float(r[a]) for a in ARMS] for k, r in zipped.items()}
    assert all(math.isfinite(value) for items in scores.values() for value in items)
    eligible = grouped_pool(rows)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"split_sha256": digest(split_path), "scores_sha256": digest(score_path),
                "salt": SALT, "repeats": N_SPLITS, "readouts": ARMS,
                "threshold": "Per class and readout, ascending rank n-floor(n/20) from accepted calibration normals; alarm if strictly above.",
                "split": "Within each class, rank normal-only object groups by salted SHA-256 and hold out the first floor(n/2) for calibration. Keep all scan variants together and all groups with defects in evaluation.",
                "summary": "Scan-micro rates per split, equally averaged across repeated splits; q10/q90 are descriptive quantiles across reused objects, not uncertainty intervals."}
    write_json(output/"readout_protocol.json", protocol)
    original = {group_key(r) for r in rows if r["split"] == "calibration"}
    fixed, limits, decisions, info = evaluate(rows, scores, original)
    assert [fixed[m]["false_alarms"] for m in ARMS] == [23, 23, 29]
    assert [fixed[m]["misses"] for m in ARMS] == [127, 172, 110]
    assert info == {"calibration_scans": 290, "calibration_scored": 280, "calibration_groups": 145, "evaluation_scans": 916}
    fixed_rows = [{**r, **{m: decisions[m].get(scan_key(r), "reject") for m in ARMS}}
                  for r in rows if r["split"] == "evaluation"]
    write_csv(output/"readout_fixed_decisions.csv", fixed_rows)
    write_json(output/"readout_fixed_thresholds.json", limits)
    repeated = []
    memberships = []
    repeats = set()
    for i in range(N_SPLITS):
        partition = repeat_groups(eligible, i)
        canonical = frozenset(partition)
        assert canonical not in repeats and canonical != original
        repeats.add(canonical)
        result, _, _, _ = evaluate(rows, scores, partition)
        for method in ARMS:
            repeated.append({"repeat": i, "readout": method, **result[method]})
        memberships.extend({"repeat": i, "category": c, "object_id": obj}
                           for c, obj in sorted(partition))
    write_csv(output/"readout_repeat_metrics.csv", repeated)
    write_csv(output/"readout_repeat_calibration_groups.csv", memberships)
    method_rows = {m: [r for r in repeated if r["readout"] == m] for m in ARMS}
    parts = {m: {k: describe([r[k] for r in method_rows[m]]) for k in ENDPOINTS} for m in ARMS}
    changes = {}
    for alt in ARMS[:2]:
        changes[alt] = {}
        for k in ENDPOINTS:
            values = [s[k]-o[k] for s, o in zip(method_rows["SPATIAL_EVIDENCE"], method_rows[alt])]
            changes[alt][k] = {**describe(values), "spatial_lower": sum(v < 0 for v in values),
                               "spatial_higher": sum(v > 0 for v in values)}
    write_json(output/"readout_summary.json", {"protocol": protocol, "fixed_counts": fixed,
               "fixed_cohort": info, "repeat_summary": parts, "paired_spatial_minus_other": changes,
               "repeat_rejects": {label: describe([r[label] for r in method_rows[ARMS[0]]])
                                  for label in ("normal_rejects", "anomaly_rejects")}})
    print(json.dumps({"fixed": fixed, "repeated": parts, "paired": changes}, indent=2))


def measure_alarm_overhead(split_path, score_path, output):
    rows = read_csv(split_path)
    scores = {scan_key(r): r for r in read_csv(score_path)}
    repeats, batches = 21, 1000
    results = {}
    for arm in ARMS:
        calibration = defaultdict(list)
        evaluation = []
        for row in rows:
            if row["status"] != "scored":
                continue
            category = row["category"]
            score = float(scores[scan_key(row)][arm])
            if row["split"] == "calibration":
                calibration[category].append(score)
            elif row["split"] == "evaluation":
                evaluation.append((category, score))
        assert len(calibration) == 12 and sum(map(len, calibration.values())) == 280
        assert len(evaluation) == 892

        def fit_thresholds():
            return {c: alarm_threshold(sorted(samples)) for c, samples in calibration.items()}

        limits = fit_thresholds()

        def apply_thresholds():
            return [s > limits[c] for c, s in evaluation]

        fit_thresholds()
        apply_thresholds()
        fits = timeit.repeat(fit_thresholds, repeat=repeats, number=batches)
        applications = timeit.repeat(apply_thresholds, repeat=repeats, number=batches)
        results[arm] = {"fit_ms_all_classes": median(fits) * 1e3 / batches,
                        "apply_ns_per_scored_scan": median(applications) * 1e9 / batches / len(evaluation)}
    write_json(output, {"platform": platform.system(), "python_version": platform.python_version(),
                        "repetitions": repeats, "batches_per_repetition": batches,
                        "calibration_scores": 280, "scored_evaluation_scans": 892,
                        "timed_scope": "In-memory threshold fit and batch comparison after scoring and registration. Excludes I/O.",
                        "arms": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode, inputs in (("export-scores", ["snapshot", "baseline"]), ("replay", ["split", "scores"]),
                         ("time-overhead", ["split", "scores"])):
        sub = modes.add_parser(mode)
        for key in inputs:
            sub.add_argument(key, type=Path)
        sub.add_argument("output", type=Path)
    a = parser.parse_args()
    if a.mode == "export-scores":
        export_scores(a.snapshot, a.baseline, a.output)
    elif a.mode == "replay":
        replay(a.split, a.scores, a.output)
    else:
        measure_alarm_overhead(a.split, a.scores, a.output)
