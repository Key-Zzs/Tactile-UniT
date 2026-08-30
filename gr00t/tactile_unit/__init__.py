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
from .vac_transition_contract import (
    ActionTransitionTarget,
    FutureOracleLeakageError,
    ModalityAvailability,
    OfflineVACTransitionTeachers,
    OnlineCausalContext,
    PredictedOrPlannedActionTransition,
    TransitionAnchor,
    VACContractError,
    reject_online_oracles,
    validate_integrated_manifest_row,
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
    "ActionTransitionTarget",
    "FutureOracleLeakageError",
    "ModalityAvailability",
    "OfflineVACTransitionTeachers",
    "OnlineCausalContext",
    "PredictedOrPlannedActionTransition",
    "TransitionAnchor",
    "VACContractError",
    "reject_online_oracles",
    "validate_integrated_manifest_row",
]

from .paired_contract import (
    CANONICAL_HORIZON,
    TREX_EMBODIMENT_ID,
    TREX_EMBODIMENT_TAG,
    VIDEO_KEY,
)
