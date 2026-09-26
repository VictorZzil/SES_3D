"""SES-3D: Spatial Exemplar Scoring for few-template 3D anomaly localization.

Core scoring and projection utilities only. Encoder, registration, and dataset
pipelines are intentionally out of scope; this package implements the score
estimators and Gaussian point projection described in the paper.
"""
from .memory import SlotMemory, shared_slot_mask
from .project import gaussian_project
from .scores import (
    centroid_scores,
    mean_scores,
    raw_effect_scores,
    ses_scores,
    shrunk_effect_scores,
    slot_geometry,
    score_estimators,
)
from .metrics import average_precision, category_macro, roc_auc

__all__ = [
    "SlotMemory",
    "shared_slot_mask",
    "gaussian_project",
    "ses_scores",
    "mean_scores",
    "centroid_scores",
    "raw_effect_scores",
    "shrunk_effect_scores",
    "slot_geometry",
    "score_estimators",
    "roc_auc",
    "average_precision",
    "category_macro",
]

__version__ = "0.1.0"
