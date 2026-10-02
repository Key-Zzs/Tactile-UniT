"""Frozen identities and path resolution for Track A.

Tracked code contains no machine-private absolute paths.  Every local path is
resolved from Prompt1's ignored ``pi2b_workspace.json`` and is checked before
use.  Track B is deliberately absent from all input resolvers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any


MODEL_ORDER = ("B0", "B_VA27", "B1", "B_HVA", "B2")
NEW_SEEDS = (43, 44)
BASE_SHA = "8f39aed123adf0a8b7241e75472e678355444e5e"
EXPECTED_BRANCH = "develop/pi2b-policy"


@dataclass(frozen=True)
class ModelRecipe:
    model_id: str
    mode: str
    lambda_phys: float
    online_contact_history: bool
    target: str | None


RECIPES = {
    "B0": ModelRecipe("B0", "NONE", 0.0, False, None),
    "B_VA27": ModelRecipe(
        "B_VA27", "VA_PHYSICAL_AUX", 0.03077957631925596, False, "VA27"
    ),
    "B1": ModelRecipe("B1", "CONTACT_STATE_TOKENS", 0.0, True, None),
    "B_HVA": ModelRecipe(
        "B_HVA",
        "CONTACT_STATE_TOKENS_VA_PHYSICAL_AUX",
        0.026468189597253295,
        True,
        "VA27",
    ),
    "B2": ModelRecipe(
        "B2",
        "CONTACT_STATE_TOKENS_PHYSICAL_AUX",
        0.020927851827513076,
        True,
        "VAC_CONTACT27",
    ),
}

SEED42_CHECKPOINT_HASHES = {
    "B0": "1e7a6ace5d69a988a8b258e1c56a24d88b077580a05b27be3f510df9ac3864f3",
    "B_VA27": "14e6ea9b8a759426610ec185b5f960e9681d14cd2b7d9fbf5918c578039c52cc",
    "B1": "04b59cbc3491bf4e88dc75558a08a94e5588e2fbe398d413e1766a87c7ab6f4f",
    "B_HVA": "82469f1d48b09f1c6152019512dba82f0f899f07bb0aab76bb58c0e0ba1e3a8b",
    "B2": "86c4908533ca24da06ae66f943e3605b5ccbae455441290ca8fb35514359e949",
}

SIDECAR_HASHES = {
    "contact": "833db9ddb4d37534bf38a7ed0b214fee2f000bb4e489d3fa9507b4e7bca5bd8e",
    "va27": "7d51a23672273ec3ea46f947080c0c3ea66db55332522b199caab00ad4fc2126",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Workspace:
    root: Path
    main_root: Path
    common_git_dir: Path
    nas_experiment_root: Path
    write_root: Path
    coordination_contract: Path

    @classmethod
    def load(cls, root: Path | None = None) -> "Workspace":
        root = (root or repository_root()).resolve()
        config_path = root / ".local/config/pi2b_workspace.json"
        payload: dict[str, Any] = json.loads(config_path.read_text())
        required = {
            "track": "policy",
            "expected_branch": EXPECTED_BRANCH,
            "base_sha": BASE_SHA,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise RuntimeError(f"workspace {key} mismatch: {payload.get(key)!r} != {expected!r}")
        if Path(payload["wt_root"]).resolve() != root:
            raise RuntimeError("workspace config points at a different worktree")
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip()
        if branch != EXPECTED_BRANCH:
            raise RuntimeError(f"wrong branch: {branch}")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=root, check=False
        )
        if ancestor.returncode:
            raise RuntimeError("Prompt1 BASE is not an ancestor of current HEAD")
        workspace = cls(
            root=root,
            main_root=Path(payload["main_root"]).resolve(),
            common_git_dir=Path(payload["common_git_dir"]).resolve(),
            nas_experiment_root=Path(payload["nas_experiment_root"]).resolve(),
            write_root=Path(payload["write_root"]).resolve(),
            coordination_contract=Path(payload["coordination_contract_path"]).resolve(),
        )
        expected_write = workspace.nas_experiment_root / "simulation/s4_3_pi2b_policy"
        if workspace.write_root != expected_write.resolve():
            raise RuntimeError("policy write root differs from the Prompt1 namespace")
        teacher_name = "s4_3_pi2b_teacher"
        if teacher_name in workspace.write_root.parts:
            raise RuntimeError("policy write root resolves into Track B")
        return workspace

    @property
    def history_artifacts(self) -> Path:
        return self.main_root / ".local/artifacts"

    @property
    def history_datasets(self) -> Path:
        return self.main_root / ".local/datasets"

    @property
    def history_external(self) -> Path:
        return self.main_root / ".local/external"

    @property
    def history_cache(self) -> Path:
        return self.main_root / ".local/cache"

    @property
    def openpi_root(self) -> Path:
        return self.history_external / "simulation/s4_3_pi1/openpi"

    @property
    def dataset_root(self) -> Path:
        return (
            self.history_external
            / "s4_3_pi0/datasets/DexJoCo-Datasets-LeRobot/"
            "dexjoco_lerobot_datasets/pinch_tongs"
        )

    @property
    def base_params(self) -> Path:
        return self.history_external / "s4_3_pi0/models/DexJoCo-Pi05/pi05_base/params"

    @property
    def assets(self) -> Path:
        return self.history_cache / "simulation/s4_3_pi0/assets"

    @property
    def contact_sidecar(self) -> Path:
        return self.history_datasets / "simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"

    @property
    def va27_sidecar(self) -> Path:
        return self.history_datasets / "simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"

    def checkpoint_seed42(self, model_id: str) -> Path:
        relative = {
            "B0": "simulation/s4_3_pi0/training/pinch_tongs/s43_pi0_official_seed42/29999",
            "B_VA27": "simulation/s4_3_pi2n/runs/B_VA27/pinch_tongs/s43_pi2n_b_va27_seed42/29999",
            "B1": "simulation/s4_3_pi1/training/pinch_tongs/s43_pi1b_contact_tokens_seed42/29999",
            "B_HVA": "simulation/s4_3_pi2m/bhva/pinch_tongs/s43_pi2m_bhva_seed42/29999",
            "B2": "simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42/29999",
        }[model_id]
        return self.nas_experiment_root / relative

    def run_root(self, model_id: str, seed: int) -> Path:
        if model_id not in RECIPES or seed not in NEW_SEEDS:
            raise ValueError((model_id, seed))
        return self.write_root / "experiments/runs" / model_id / f"seed{seed}"

    def run_id(self, model_id: str, seed: int) -> str:
        return f"seed{seed}_{model_id.lower()}"

    def experiment_name(self, model_id: str, seed: int) -> str:
        return f"s43_pi2b_{model_id.lower()}_seed{seed}"

    def final_checkpoint(self, model_id: str, seed: int) -> Path:
        return self.run_root(model_id, seed) / "pinch_tongs" / self.experiment_name(model_id, seed) / "29999"
