"""Matched VA/VAC teacher study components for S4.3-PI2B-T."""

from .matched_teacher import (
    MatchLossWeights,
    MatchedTeacher,
    StrictFieldStore,
    build_matched_pair,
    clip_active_grad_norm_,
    matched_teacher_loss,
)

__all__ = [
    "MatchLossWeights",
    "MatchedTeacher",
    "StrictFieldStore",
    "build_matched_pair",
    "clip_active_grad_norm_",
    "matched_teacher_loss",
]
