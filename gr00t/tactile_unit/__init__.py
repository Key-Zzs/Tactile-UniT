"""Tactile-UniT milestone contracts, adaptors, and compatibility helpers."""

from .contact_adapter import ContactCodebookAdaptor, build_contact_adaptor

from .compatibility import (
    active_set_jaccard,
    code_frequency,
    codebook_usage,
    deterministic_contact_subset,
    effective_rank,
    jensen_shannon_divergence,
    parameter_digest,
    quantization_metrics,
    quantize_with_stage_diagnostics,
)
from .causal_contact_contract import (
    ContactBridgeBatch,
    ContactMode,
    ContactTransitionTarget,
    CurrentContactContext,
    FutureContactLeakageError,
    PredictedContactTransition,
    VisionTransitionTarget,
    reject_future_oracles,
    runtime_contact_batch,
)
from .continuous_contact_bridge import (
    CausalContactGate,
    TokenSetCrossAttentionBridge,
    TwoTowerContinuousProjector,
)

__all__ = [
    "ContactCodebookAdaptor",
    "active_set_jaccard",
    "code_frequency",
    "codebook_usage",
    "deterministic_contact_subset",
    "effective_rank",
    "jensen_shannon_divergence",
    "parameter_digest",
    "quantization_metrics",
    "quantize_with_stage_diagnostics",
    "build_contact_adaptor",
    "ContactBridgeBatch",
    "ContactMode",
    "ContactTransitionTarget",
    "CurrentContactContext",
    "FutureContactLeakageError",
    "PredictedContactTransition",
    "VisionTransitionTarget",
    "reject_future_oracles",
    "runtime_contact_batch",
    "CausalContactGate",
    "TokenSetCrossAttentionBridge",
    "TwoTowerContinuousProjector",
]

from .paired_contract import (
    CANONICAL_HORIZON,
    TREX_EMBODIMENT_ID,
    TREX_EMBODIMENT_TAG,
    VIDEO_KEY,
)
