"""Evaluate frozen class-specific Real3D alarms on disjoint normal scan groups."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

SALT = "SESAD-TASE-20261117-real3d-alarm-v1"
METHODS = ("SESAD", "DIAG_COSINE")
RULE = "Per method and class, ascending rank n-floor(n/20) of accepted calibration-normal scan scores. Alarm iff score > threshold."


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def scan_key(row):
    return row["category"], row["staged_id"]


def group_key(row):
    return row["category"], row["object_id"]


def freeze(inventory_path, reject_path, directory):
    inventory = read_csv(inventory_path)
    rejects = read_csv(reject_path)
    unique = {scan_key(row): row for row in inventory}
    assert len(unique) == len(inventory) == 1206
    assert len({row["category"] for row in inventory}) == 12
    assert Counter((r["label"], r["status"]) for r in inventory) == {
        ("0", "scored"): 579, ("0", "rejected"): 25, ("1", "scored"): 593, ("1", "rejected"): 9}
    assert len(rejects) == 34 and {
        (r["class"], r["staged_id"], r["official_file"], r["label"]) for r in rejects
    } == {(r["category"], r["staged_id"], r["official_file"], r["label"]) for r in inventory
          if r["status"] == "rejected"}

    groups = defaultdict(set)
    for row in inventory:
        groups[group_key(row)].add(row["label"])
    selected = set()
    eligible = defaultdict(list)
    for key, labels in groups.items():
        if labels == {"0"}:
            eligible[key[0]].append(key)
    for keys in eligible.values():
        ranked = sorted(keys, key=lambda k: hashlib.sha256(
            (SALT + "/" + "/".join(k)).encode()).hexdigest())
        selected.update(ranked[:len(ranked)//2])
    assert len(eligible) == 12 and selected
    manifest = sorted(inventory, key=scan_key)
    for row in manifest:
        row["split"] = "calibration" if group_key(row) in selected else "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "split.csv"
    protocol_path = directory / "protocol.json"
    if destination.exists() or protocol_path.exists():
        raise FileExistsError("Frozen trial already exists. Use its recorded split.")
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    write_json(protocol_path, {
        "source_sha256": {inventory_path.name: digest(inventory_path), reject_path.name: digest(reject_path)},
        "split_sha256": digest(destination),
        "salt": SALT, "target_calibration_fpr": 0.05, "threshold_rule": RULE,
        "partition": "First half (rounded down) of normal-only object IDs per class sorted by salted SHA-256; grouping includes all scan variants.",
        "mixed_label_groups_retained_for_evaluation": [k for k, labels in sorted(groups.items()) if len(labels) > 1],
        "evaluation_primary": "Scan-micro FPR and miss rate among registration-accepted scans. Report registration rejects by label among all evaluation inputs.",
        "routing": "Alarm or failed registration requests review. Otherwise the scan automatically passes. No downstream human performance is assumed.",
        "status": "Retrospective threshold holdout within an existing benchmark. Scorers were frozen before this experiment.",
        "methods": list(METHODS), "head": "CANONICAL_EEF",
    })
    print(Counter((row["split"], row["label"], row["status"]) for row in manifest))


def metrics(rows, accepted, normal_alarms, defect_passes):
    normal_inputs = sum(r["label"] == "0" for r in rows)
    anomaly_inputs = len(rows) - normal_inputs
    normal_scored = sum(r["label"] == "0" for r in accepted)
    anomaly_scored = len(accepted) - normal_scored
    normal_rejects = normal_inputs - normal_scored
    anomaly_rejects = anomaly_inputs - anomaly_scored
    ratio = lambda x, n: x/n if n else None
    return {
        "normal_inputs": normal_inputs, "anomaly_inputs": anomaly_inputs,
        "normal_scored": normal_scored, "anomaly_scored": anomaly_scored,
        "normal_rejects": normal_rejects, "anomaly_rejects": anomaly_rejects,
        "false_alarms": normal_alarms, "misses": defect_passes,
        "scored_fpr": ratio(normal_alarms, normal_scored),
        "scored_miss_rate": ratio(defect_passes, anomaly_scored),
        "normal_review_rate": ratio(normal_alarms + normal_rejects, normal_inputs),
        "defect_auto_pass_rate": ratio(defect_passes, anomaly_inputs),
        "normal_reject_rate": ratio(normal_rejects, normal_inputs),
        "anomaly_reject_rate": ratio(anomaly_rejects, anomaly_inputs),
    }


def evaluate(snapshot_path, old_path, directory):
    protocol = json.loads((directory/"protocol.json").read_bytes())
    split = directory/"split.csv"
    assert digest(split) == protocol["split_sha256"] and protocol["threshold_rule"] == RULE
    manifest = read_csv(split)
    snapshot = json.loads(snapshot_path.read_bytes())
    older = json.loads(old_path.read_bytes())
    ref = snapshot["protocol"]
    assert snapshot["run_complete"] and ref["canonical_template_graph"] and ref["normal_only_metric_fitting"]
    baseline = {(category, scan["id"]): scan for category in snapshot["classes"]
                for scan in older[category]["samples"]}
    all_scores = {(category, scan["id"]): scan for category, record in snapshot["classes"].items()
                  for scan in record["samples"]}
    assert len(all_scores) == 1172 and set(all_scores) == set(baseline) == {
        scan_key(r) for r in manifest if r["status"] == "scored"}
    head = ref["heads"].index("CANONICAL_EEF")
    indices = {m: ref["metrics"].index(m) for m in METHODS}
    for row in manifest:
        key = scan_key(row)
        if key in all_scores:
            s, b = all_scores[key], baseline[key]
            assert b["official_file"] == row["official_file"]
            assert s["label"] == b["label"] == int(row["label"]) and s["points"] == b["points"]
            assert math.isclose(s["object_scores"][indices["SESAD"]][head],
                                b["object_scores"]["V22_EEF"], rel_tol=0, abs_tol=1e-8)
            assert all(math.isfinite(s["object_scores"][i][head]) for i in indices.values())
    calib = [r for r in manifest if r["split"] == "calibration"]
    evaluation = [r for r in manifest if r["split"] == "evaluation"]
    assert {group_key(r) for r in calib}.isdisjoint(group_key(r) for r in evaluation)
    assert all(r["label"] == "0" for r in calib)
    scored_calib = [r for r in calib if r["status"] == "scored"]
    scored_eval = [r for r in evaluation if r["status"] == "scored"]
    categories = sorted(snapshot["classes"])
    thresholds = {}
    for method, idx in indices.items():
        thresholds[method] = {}
        for category in categories:
            samples = [r for r in scored_calib if r["category"] == category]
            scores = sorted(all_scores[scan_key(r)]["object_scores"][idx][head] for r in samples)
            k = len(scores) - len(scores)//20
            assert k >= 1
            threshold = scores[k-1]
            thresholds[method][category] = {"score": threshold, "n": len(scores), "rank_1_based": k,
                                            "above_threshold": sum(v > threshold for v in scores)}
    threshold_path = directory/"thresholds.json"
    write_json(threshold_path, {"rule": RULE, "thresholds": thresholds,
                                "split_sha256": digest(split), "score_sha256": digest(snapshot_path)})
    thresholds = json.loads(threshold_path.read_bytes())["thresholds"]
    decisions = {
        method: {scan_key(row): all_scores[scan_key(row)]["object_scores"][i][head] >
                thresholds[method][row["category"]]["score"] for row in scored_eval}
        for method, i in indices.items()
    }
    output = {}
    with (directory/"decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluation[0]) + list(METHODS))
        writer.writeheader()
        for row in evaluation:
            key = scan_key(row)
            writer.writerow({**row, **{m: ("reject" if key not in calls else "alarm" if calls[key] else "pass")
                                       for m, calls in decisions.items()}})
    for method, calls in decisions.items():
        subsets = {"all": evaluation,
                   "original": [r for r in evaluation if not r["variant"].endswith(("_cut", "_copy"))],
                   "cropped": [r for r in evaluation if r["variant"].endswith("_cut")],
                   "copies": [r for r in evaluation if r["variant"].endswith("_copy")]}
        subsets.update({c: [r for r in evaluation if r["category"] == c] for c in categories})
        output[method] = {}
        for name, subset in subsets.items():
            accepted = [r for r in subset if r["status"] == "scored"]
            output[method][name] = metrics(
                subset, accepted,
                sum(calls[scan_key(r)] for r in accepted if r["label"] == "0"),
                sum(not calls[scan_key(r)] for r in accepted if r["label"] == "1"))
        output[method]["macro"] = {k: mean(output[method][c][k] for c in categories)
                                   for k in ("scored_fpr", "scored_miss_rate")}
    write_json(directory/"results.json", {
        "calibration_inputs": len(calib), "calibration_scored": len(scored_calib),
        "calibration_rejected": len(calib)-len(scored_calib),
        "calibration_groups": len({group_key(r) for r in calib}),
        "evaluation_inputs": len(evaluation), "evaluation_scored": len(scored_eval),
        "evaluation_normal_groups": len({group_key(r) for r in evaluation if r["label"] == "0"}),
        "evaluation_anomaly_groups": len({group_key(r) for r in evaluation if r["label"] == "1"}),
        "threshold_sha256": digest(threshold_path), "split_sha256": digest(split),
        "baseline_archive_sha256": digest(old_path), "methods": output})
    for method in METHODS:
        print(method, json.dumps(output[method]["all"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode, sources in (("freeze", ("inventory", "rejects")), ("evaluate", ("snapshot", "baseline"))):
        options = modes.add_parser(mode)
        for source in sources:
            options.add_argument(source, type=Path)
        options.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.inventory, args.rejects, args.output)
    else:
        evaluate(args.snapshot, args.baseline, args.output)
