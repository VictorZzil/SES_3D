# SESAD for aligned 3D anomaly localization

This package contains SESAD scoring from registered points and aligned descriptors, together with the evidence and replay scripts for the reported Real3D-AD alarm and ranking analyses and the paired full-bank comparisons. It accompanies the Displays manuscript. The project repository is [SES_3D](https://github.com/VictorZzil/SES_3D).

## Setup and replay

Use Python 3.12 with the NumPy version in requirements.txt.

    python -m pip install -r requirements.txt
    python reproduce.py

The replay runs on a CPU. It compares eight alarm and ROC output files with the saved results, checks the same-slot ranking audit on 1,172 accepted scans, and recomputes 14 paired benchmark intervals. Temporary outputs are kept separate from the archived values.

| Manuscript result | Archived input or result |
| --- | --- |
| Fixed alarm thresholds and grouped repeats | [Scores](evidence/alarm_readouts/readout_scores.csv), [split](evidence/alarm_readouts/fixed_split.csv), [summary](evidence/alarm_readouts/readout_summary.json) |
| Equal false-alarm rate and crop stress test | [ROC and crop results](evidence/alarm_readouts/matched_low_fpr.json) |
| Same-slot ranking reversals at positive variance | [Ranking summary](evidence/slot_reversals/slot_reversals.json), [scored slots](evidence/slot_reversals/slot_scores.zip) |
| Paired full-bank comparisons | [Paired intervals](evidence/fullbank_paired/data/fullbank_paired_v29.csv) |
| ShapeNet matched spatial search and measured Figure 2(c) cases | [Spatial comparisons](evidence/shapenet_matched_spatial.csv), [case distances](evidence/shapenet_motivation_cases.json) |

Replay starts from the archived scores and category results. Upstream registration, descriptor extraction, checkpoint training, and the end-to-end runs behind the main benchmark table require the original data and pipelines.

## Score one registered scan

    python sesad.py aligned_scan.npz scores.npz

The input NumPy archive must contain these arrays.

| Name | Shape and meaning |
| --- | --- |
| points | (N, 3) registered query points in the order expected by the caller |
| query | (S, D) query descriptors at template slots |
| bank | (S, T, D) normal descriptors at the same slots |
| point_indices | (S,) integer query-point correspondence for each slot, with an out-of-range index marking a missing match |
| canonical_centers | (S, 3) coordinates of the slots on the base template |
| reference_valid | Optional (S, T) mask for valid template correspondences |
| point_slot_mask | Optional (S,) mask to apply the archived point-projection support |

Local scores use the closest valid normal exemplar at each slot and Gaussian projection with sigma 1 in cache coordinates by default. The optional projection mask can reproduce the point-map support chosen by the host pipeline. The spatial readout uses slots with valid descriptors from the full bank and needs at least two templates. It returns an unavailable spatial score if that set is empty. The point-map maximum and spatial object score are reported as separate outputs. This standalone implementation uses CPU float64. The paper's timing and point scores were measured with the host implementation and its GPU projection kernel.

## Upstream dependencies and data

- [COSSAD](https://github.com/alextarvo/cossad) supplies the upstream registration, descriptor extractor, and frozen checkpoints for Real3D-AD and ShapeNet. Obtain these from the original repository under its terms.
- [Real3D-AD](https://github.com/M-3LAB/Real3D-AD) and [Anomaly-ShapeNet](https://github.com/Chopper-233/Anomaly-ShapeNet) provide the benchmark scans. The Real3D-AD and Anomaly-ShapeNet project pages describe their dataset licenses.
- [IEC3D-AD](https://doi.org/10.1109/TCSVT.2026.3723149) is used for the local-scoring representation control. Use the dataset and descriptor pipeline from the cited providers.

See [THIRD_PARTY.md](THIRD_PARTY.md) for the provenance and distribution scope of this package.
