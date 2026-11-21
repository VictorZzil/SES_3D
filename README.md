# SESAD and SES-3D code

## Displays manuscript code and evidence

The [Displays reproducibility package](displays/) contains the aligned-descriptor
scorer, frozen-threshold and same-slot ranking replays, and paired full-bank
evidence. Run the README instructions in that directory to check the archived
results.

## Earlier SES-3D scoring release

Minimal, dependency-light implementation of **Spatial Exemplar Scoring** for
few-template 3D anomaly localization, prepared for double-blind review.

This package contains **only** the scoring estimators, registered slot memory
helpers, Gaussian point projection, and point metrics used in the paper.
It does **not** include the encoder, registration pipeline, dataset caches, or
training code.

## Contents

```
ses3d/
  memory.py    # location-indexed same-slot normal memory
  scores.py    # SES, mean, centroid, raw effect, shrinkage, common-defined SES
  project.py   # Gaussian projection of patch scores to points
  metrics.py   # point AUROC / AP and category-macro aggregation
examples/
  demo_synthetic.py
tests/
  test_core.py
```

## Method (one line per estimator)

Given L2-normalised query feature \(q\) and valid same-slot normal features
\(\{r_t\}\) with cosine distance \(d(q,r)=1-\cos(q,r)\):

| key | formula |
|---|---|
| `ses` | \(\min_t d(q,r_t)\) |
| `mean` | \(\mathrm{mean}_t\, d(q,r_t)\) |
| `centroid` | \(1-\cos(q,\ \mathrm{norm}(\mathrm{mean}_t \mathrm{norm}(r_t)))\) |
| `effect` | \(\lvert \bar D-\bar P\rvert/\sqrt{v_{\mathrm{pool}}}\) (needs \(\ge 3\) refs) |
| `shrink` | shrunk-effect variance with normal-only prior \((\mu_0,v_0),\ \nu_0=2\) |
| `ses_common` | SES restricted to the raw-effect defined support |

Undefined slots are **NaN** (excluded from projection), never silently zero.
The optional legacy zero-fallback for `effect` is exposed as `effect_legacy0`.

## Install / run

```bash
# Python >= 3.9, only NumPy required
pip install -r requirements.txt

# unit checks
python tests/test_core.py

# synthetic API demo
python examples/demo_synthetic.py
```

## Minimal usage

```python
import numpy as np
from ses3d import SlotMemory, score_estimators, gaussian_project

# query features [S, d], one [S, d] array per normal template
query = ...
templates = [t0, t1, t2]          # each [S, d], NaN = missing
centers = ...                     # optional [S, 3]

mem = SlotMemory.from_templates(query, templates, centers=centers)
slots = mem.shared_slots()        # finite query and >=1 valid reference
scores = score_estimators(mem.query, mem.bank, slots=slots)

point_scores = gaussian_project(
    points,                       # [N, 3]
    mem.centers[slots],
    scores["ses"],                # Spatial Exemplar Scoring
    sigma=1.0,
)
```

## Evaluation convention (paper)

Point AUROC and AP are computed over concatenated points within each category,
then macro-averaged across categories. Template-subset metrics are averaged
inside a category before any paired bootstrap. See `ses3d.metrics.category_macro`.

## Scope and limitations

- Encoder features are assumed to be produced upstream (frozen in the paper).
- Registration is assumed to have produced a shared frame; this package only
  stores same-slot features.
- Scores are ranking evidence, not calibrated defect probabilities.
- The raw effect statistic is a descriptive few-sample score; shared endpoints
  of pairwise reference distances are dependent.

## Anonymous release notes

- No author names, institutions, or identifying URLs.
- No dataset paths, server mounts, or machine-specific configuration.
- No training checkpoints or non-redistributable assets.
- Public baselines (e.g. COSSAD, Template3D-AD) are referenced in the paper only.

## License

Released under the MIT License for review and research use.
