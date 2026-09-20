from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/simulation/launch_s4_3_pi2n_development.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "launch_s4_3_pi2n_development", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_scope_and_session_are_frozen() -> None:
    module = load_module()
    assert module.MODELS == ("B_HVA", "B2", "B_VAC_V")
    assert module.TOTAL_OUTCOMES == 90
    assert module.SESSION == "s43_pi2n_dev"


def test_launcher_is_persistent_non_overwriting_and_does_not_chain_final() -> None:
    source = SCRIPT.read_text()
    assert "tmux" in source
    assert "development_heartbeat.json" in source
    assert "refusing to overwrite or duplicate PI2N DEV evaluation" in source
    assert '"automatic_final_launch": False' in source
    assert "final evaluation remains manually blocked" in source
    assert "eligible[:3]" in source
