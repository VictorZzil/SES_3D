"""v29 M1: paired full-bank category-bootstrap intervals.

Protocol matches D1 (REPORT_CN.md): 95% paired category bootstrap, 10,000
draws, seed=2026, percentile method. Cohorts follow the main manuscript:
  Original40 T=4 : 39 original classes from D1 common50 T=4 + shelf0
                   (m=2) unchanged at its T=2 subset average.
  All52 T=4      : 50 eligible classes at T=4 + cap1/shelf0 (m=2) unchanged
                   at their T=2 subset averages (main Sec. IV-F convention).
  Real3D full    : e94 real_per_class_detail, slot_nn minus official, 12 classes.
Cross-check: COMMON50 T=4 intervals must reproduce the server D1 metrics.csv.
"""
import csv, io, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

srv = json.load(open(os.path.join(DATA, "server_table_sources_v19.json"), encoding="utf-8"))
audit = json.load(open(os.path.join(DATA, "shape_cohort_audit_v17.json"), encoding="utf-8"))
e94 = json.load(open(os.path.join(DATA, "e94_original_records.json"), encoding="utf-8"))

# ---- D1 per-category values ----
pc = {}
for r in csv.DictReader(io.StringIO(srv["PASDF-ses-d1/results/ses_d1/per_category.csv"])):
    if r["value"].strip():
        pc[(r["dataset"], int(r["T"]), r["method"], r["metric"], r["class"])] = float(r["value"])

manifest = list(csv.DictReader(open(os.path.join(DATA, "original40_class_manifest.csv"), encoding="utf-8")))
key = "class" if manifest and "class" in manifest[0] else ("category" if manifest and "category" in manifest[0] else None)
orig40 = [r[key] for r in manifest] if key else []
if not orig40:
    orig40 = audit["categories"]["original40"]
all52 = audit["categories"]["all52"]
common50 = audit["categories"]["COMMON50"]
excluded = audit["COMMON50_excludes"]  # cap1, shelf0 (m=2)
assert set(orig40) <= set(all52), "original40 must be subset of all52"
assert set(all52) == set(common50) | set(excluded)
assert "shelf0" in orig40 and "cap1" not in orig40
print(f"cohort check: orig40={len(orig40)}, common50={len(common50)}, all52={len(all52)}, "
      f"excluded={excluded}, orig40∩common50={len(set(orig40)&set(common50))}")

# T=2 subset averages for the two m=2 categories (from v17 audit, all 52 classes)
t2 = {}
for r in audit["category_results"]:
    if r["T"] == 2:
        t2[(r["method"], r["category"])] = r

METHODS = {"SESAD": "nearest-exemplar", "MEAN": "mean", "CEN": "centroid"}
METRICS = ["P-AUROC", "P-AP", "O-AUROC"]
AUDIT_METRIC = {"P-AUROC": "P_AUROC", "P-AP": "P_AP", "O-AUROC": "O_AUROC"}

def cohort_values(cohort, metric):
    """Return {method: np.array over categories} at full bank (T=4 convention)."""
    if cohort in ("Original40", "All52"):
        classes = orig40 if cohort == "Original40" else all52
        out = {}
        for m, mm in METHODS.items():
            vals = []
            for c in classes:
                if c in excluded:
                    vals.append(t2[(mm, c)][AUDIT_METRIC[metric]])
                else:
                    vals.append(pc[("shapenet_common50", 4, mm, metric, c)])
            out[m] = np.array(vals)
        return out, classes
    if cohort == "Common50":
        classes = common50
        return {m: np.array([pc[("shapenet_common50", 4, mm, metric, c)] for c in classes])
                for m, mm in METHODS.items()}, classes
    raise ValueError(cohort)

# ---- bootstrap (percentile, paired categories) ----
def paired_ci(diff, seed=2026, draws=10000, rng_kind="default_rng"):
    n = len(diff)
    if rng_kind == "default_rng":
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(draws, n))
    else:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=(draws, n))
    boots = diff[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

# ---- cross-check against server D1 COMMON50 T=4 ----
server = {}
for r in csv.DictReader(io.StringIO(srv["PASDF-ses-d1/results/ses_d1/metrics.csv"])):
    if r["dataset"] == "shapenet_common50" and "T=4" in r["notes"] and r["value"]:
        server[(r["method"], r["metric"])] = (float(r["value"]), float(r["ci_low"]), float(r["ci_high"]))

print("=== cross-check: COMMON50 T=4 (server D1 vs local recompute) ===")
best_kind = None
for kind in ("default_rng", "RandomState"):
    max_err = 0.0
    for metric in METRICS:
        vals, classes = cohort_values("Common50", metric)
        for comp in ("MEAN", "CEN"):
            key = f"nearest-exemplar_minus_{'mean' if comp=='MEAN' else 'centroid'}"
            sv, slo, shi = server[(key, metric)]
            diff = vals["SESAD"] - vals[comp]
            lo, hi = paired_ci(diff, rng_kind=kind)
            err = max(abs(lo - slo), abs(hi - shi))
            max_err = max(max_err, err)
    print(f"  rng={kind}: max CI endpoint error vs server = {max_err:.5f}")
    if best_kind is None or max_err < best_err:
        best_kind, best_err = kind, max_err
print(f"  -> using rng={best_kind} (max err {best_err:.5f})")

# ---- macro verification against Table II(b) / S1 / S14 ----
print("\n=== macro verification (full bank) ===")
expected = {
    ("Original40", "SESAD"): {"P-AUROC": 0.986, "P-AP": 0.714, "O-AUROC": 0.969},
    ("Original40", "MEAN"):  {"P-AUROC": 0.981, "P-AP": 0.701, "O-AUROC": 0.954},
    ("Original40", "CEN"):   {"P-AUROC": 0.984, "P-AP": 0.709, "O-AUROC": 0.959},
    ("All52", "SESAD"): {"P-AUROC": 0.982, "P-AP": 0.656, "O-AUROC": 0.949},
    ("All52", "MEAN"):  {"P-AUROC": 0.973, "P-AP": 0.638, "O-AUROC": 0.922},
    ("All52", "CEN"):   {"P-AUROC": 0.978, "P-AP": 0.646, "O-AUROC": 0.930},
}
ok = True
for (cohort, m), exp in expected.items():
    for metric, ev in exp.items():
        vals, _ = cohort_values(cohort, metric)
        got = float(vals[m].mean())
        flag = "OK" if abs(got - ev) < 0.0005 else "MISMATCH"
        if flag != "OK":
            ok = False
        print(f"  {cohort:10s} {m:5s} {metric:8s} computed={got:.4f} table={ev:.3f} {flag}")
assert ok, "macro reconstruction failed - stop"

# ---- Real3D full-bank paired (e94, 12 classes) ----
rows = list(csv.DictReader(io.StringIO(e94["real_per_class_detail.csv"])))
print("\n=== Real3D e94 detail columns ===")
print(" ", list(rows[0].keys()))
ckey = "class" if "class" in rows[0] else "category"
real = {}
for metric, col_off, col_ses in [
    ("P-AUROC", "P_AUROC_official", "P_AUROC_slot_nn"),
    ("P-AP", "P_AP_official", "P_AP_slot_nn"),
]:
    off = np.array([float(r[col_off]) for r in rows])
    ses = np.array([float(r[col_ses]) for r in rows])
    real[metric] = (off, ses, [r[ckey] for r in rows])
    print(f"  {metric}: native macro={off.mean():.4f} SESAD macro={ses.mean():.4f} n={len(rows)}")

# ---- compute all paired intervals ----
results = []
for cohort in ("Original40", "All52"):
    for metric in ("P-AUROC", "P-AP", "O-AUROC"):
        vals, classes = cohort_values(cohort, metric)
        for comp in ("CEN", "MEAN"):
            diff = vals["SESAD"] - vals[comp]
            lo, hi = paired_ci(diff, rng_kind=best_kind)
            results.append({
                "cohort": cohort, "budget": "full (T=4 convention)", "metric": metric,
                "comparison": f"SESAD_minus_{comp}", "n_categories": len(classes),
                "mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
                "significant_positive": lo > 0,
            })
for metric, (off, ses, classes) in real.items():
    diff = ses - off
    lo, hi = paired_ci(diff, rng_kind=best_kind)
    results.append({
        "cohort": "Real3D12", "budget": "full (per-category m_c)", "metric": metric,
        "comparison": "SESAD_minus_native_COSSAD", "n_categories": len(classes),
        "mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
        "significant_positive": lo > 0,
    })

print("\n=== v29 paired full-bank intervals (10k draws, seed=2026, percentile) ===")
for r in results:
    sig = "SIG+" if r["significant_positive"] else ("SIG-" if r["ci_high"] < 0 else "ns")
    print(f"  {r['cohort']:10s} {r['metric']:8s} {r['comparison']:28s} "
          f"{r['mean_diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] {sig}")

out = {
    "protocol": "95% paired category bootstrap, 10000 draws, seed=2026, percentile; matches D1 REPORT_CN.md",
    "rng_kind": best_kind,
    "crosscheck_max_endpoint_error_vs_server_common50_T4": best_err,
    "cohort_convention": "Full bank = eligible classes at T=4 + m=2 classes (cap1, shelf0) unchanged at T=2 subset averages; Original40 uses 39 eligible original classes + shelf0, All52 uses all 50 eligible classes + cap1/shelf0; Real3D = e94 per-category full bank (slot_nn vs official), 12 classes",
    "sources": ["server_table_sources_v19.json (D1 per_category.csv, metrics.csv)",
                 "shape_cohort_audit_v17.json (category_results T=2, cohort lists)",
                 "original40_class_manifest.csv", "e94_original_records.json (real_per_class_detail.csv)"],
    "results": results,
}
with open(os.path.join(DATA, "fullbank_paired_v29.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
with open(os.path.join(DATA, "fullbank_paired_v29.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print("\nwrote data/fullbank_paired_v29.json and .csv")