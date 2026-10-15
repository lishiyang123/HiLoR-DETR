"""Nearest-neighbour post-processing baselines for frozen CDW1630 outputs."""

from .candidate_adapter import CandidateSet, candidates_from_record
from .gilg_iou_aware import ConditionalBetaCalibrator, gilg_scores
from .soft_nms import soft_nms_scores

__all__ = [
    "CandidateSet",
    "ConditionalBetaCalibrator",
    "candidates_from_record",
    "gilg_scores",
    "soft_nms_scores",
]
