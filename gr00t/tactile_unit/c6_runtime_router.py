"""Canonical M3 runtime availability route, excluding C5 diagnostics."""

from __future__ import annotations

from enum import Enum


class C6RuntimeMode(str, Enum):
    FULL_AH = "FULL_AH"
    FALLBACK_A = "FALLBACK_A"
    ABSTAIN_NO_ACTION = "ABSTAIN_NO_ACTION"


def route_c6_availability(*, action_available: bool, contact_context_available: bool, vision_available: bool) -> C6RuntimeMode:
    """Route only explicit availability metadata; Vision never changes M3 runtime."""
    if any(type(value) is not bool for value in (action_available, contact_context_available, vision_available)):
        raise TypeError("availability values must be explicit bools")
    if not action_available:
        return C6RuntimeMode.ABSTAIN_NO_ACTION
    return C6RuntimeMode.FULL_AH if contact_context_available else C6RuntimeMode.FALLBACK_A
