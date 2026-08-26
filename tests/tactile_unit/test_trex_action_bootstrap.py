from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.model.tokenizer.action_branch_decoder import ActionDecoder, ActionDecoderConfig
from gr00t.model.tokenizer.action_branch_encoder import ActionEncoder, ActionEncoderConfig
from gr00t.tactile_unit.paired_contract import pad_trex_state_action, validate_episode_splits
from gr00t.tactile_unit.trex_action_bootstrap import (
    EXPANDED_CATEGORY_CAPACITY,
    GR1_EMBODIMENT_ID,
    RELEASED_CATEGORY_CAPACITY,
    TREX_EMBODIMENT_ID,
    UNUSED_EMBODIMENT_ID,
    FrozenActionOnlyProjection,
    TReXActionBootstrap,
    expand_category_tensor,
    load_bootstrap_checkpoint,
    save_bootstrap_checkpoint,
)
from gr00t.tactile_unit.trex_action_data import (
    ActionWindow,
    EpisodeActionPointer,
    action_activity,
    deterministic_windows,
    different_episode_indices,
)
from scripts.tactile_unit.train_trex_action_bootstrap import validation_selection_key
from scripts.tactile_unit.evaluate_trex_action_bootstrap import atomic_json as evaluation_atomic_json


ROOT = Path(__file__).resolve().parents[2]


class FakeReleasedSource:
    def __init__(self) -> None:
        hidden_size = 64
        encoder_cfg = {
            "action_dim": 128,
            "state_dim": 128,
            "action_horizon": 16,
            "hidden_size": hidden_size,
            "query_num": 8,
            "num_conv_layers": 2,
            "conv_kernel_size": 3,
            "conv_stride": 2,
            "use_dilation": False,
            "downsample_target_len": None,
            "max_num_embodiments": 30,
            "dropout": 0.0,
            "m_former_cfg": {
                "hidden_size": hidden_size,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "intermediate_size": 128,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "query_num": 8,
                "input_hidden_size": hidden_size,
            },
        }
        decoder_cfg = {
            "action_dim": 128,
            "action_horizon": 16,
            "hidden_size": hidden_size,
            "query_num": 8,
            "num_conv_layers": 2,
            "conv_kernel_size": 3,
            "upsample_stride": 2,
            "use_dilation": False,
            "max_num_embodiments": 30,
            "dropout": 0.0,
            "m_former_cfg": {
                "hidden_size": hidden_size,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "intermediate_size": 128,
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "query_num": 8,
                "input_hidden_size": hidden_size,
            },
        }
        self.config = {
            "action_horizon": 16,
            "action_dim": 128,
            "state_dim": 128,
            "query_num": 8,
            "hidden_size": hidden_size,
            "action_encoder_cfg": encoder_cfg,
            "action_decoder_cfg": decoder_cfg,
            "vq_cfg": {"e_dim": 32, "n_e": 128},
        }
        torch.manual_seed(17)
        encoder = ActionEncoder(ActionEncoderConfig(**encoder_cfg))
        decoder = ActionDecoder(ActionDecoderConfig(**decoder_cfg))
        projection = FrozenActionOnlyProjection(hidden_size, 8, 32)
        self.values = {
            **{f"action_branch.{name}": value for name, value in encoder.state_dict().items()},
            **{f"action_decoder.{name}": value for name, value in decoder.state_dict().items()},
        }
        for name, value in projection.state_dict().items():
            source_name = f"fusion.{name}" if name.startswith(("align_action.", "shared_projection.")) else name
            self.values[source_name] = value
        self.identity = {"config_sha256": "fake-config", "index_sha256": "fake-index"}

    def tensor(self, key: str) -> torch.Tensor:
        return self.values[key].clone()


@pytest.fixture()
def fake_source() -> FakeReleasedSource:
    return FakeReleasedSource()


def test_explicit_trex_registration_does_not_alias_gr1() -> None:
    assert EmbodimentTag.TREX.value == "trex"
    assert EMBODIMENT_TAG_MAPPING["trex"] == TREX_EMBODIMENT_ID == 31
    assert EMBODIMENT_TAG_MAPPING["gr1"] == GR1_EMBODIMENT_ID == 24
    assert TREX_EMBODIMENT_ID != GR1_EMBODIMENT_ID
    assert UNUSED_EMBODIMENT_ID == 30
    assert UNUSED_EMBODIMENT_ID not in EMBODIMENT_TAG_MAPPING.values()


def test_category_expansion_preserves_every_old_row_and_gr1_output() -> None:
    torch.manual_seed(2)
    released = torch.randn(RELEASED_CATEGORY_CAPACITY, 5, 7)
    trex = torch.randn(1, 5, 7)
    expanded = expand_category_tensor(released, name="toy.W", trex_row=trex, seed=9)
    assert expanded.shape == (EXPANDED_CATEGORY_CAPACITY, 5, 7)
    assert torch.equal(expanded[:RELEASED_CATEGORY_CAPACITY], released)
    assert torch.equal(expanded[TREX_EMBODIMENT_ID : TREX_EMBODIMENT_ID + 1], trex)
    value = torch.randn(3, 5)
    before = value @ released[GR1_EMBODIMENT_ID]
    after = value @ expanded[GR1_EMBODIMENT_ID]
    assert torch.equal(before, after)


def test_category_expansion_rejects_wrong_bounds() -> None:
    with pytest.raises(ValueError):
        expand_category_tensor(torch.zeros(29, 2), name="x.W", trex_row=torch.zeros(1, 2))
    with pytest.raises(ValueError):
        expand_category_tensor(torch.zeros(30, 2), name="x.W", trex_row=torch.zeros(2, 2))


def test_raw_58_to_canonical_128_masks_and_order() -> None:
    state = np.arange(58, dtype=np.float32)
    action = np.arange(16 * 58, dtype=np.float32).reshape(16, 58)
    result = pad_trex_state_action(state, action)
    assert result["state"].shape == (128,)
    assert result["action"].shape == (16, 128)
    assert result["state_mask"].sum() == 58
    assert result["action_mask"].sum() == 16 * 58
    np.testing.assert_array_equal(result["state"][:58], state)
    np.testing.assert_array_equal(result["action"][:, :58], action)
    assert not result["state_mask"][58:].any()
    assert not result["action_mask"][:, 58:].any()


def test_t_to_t_plus_15_window_has_no_off_by_one() -> None:
    pointer = EpisodeActionPointer(0, 16, 0, 0, 0, 16, "reach")
    split = {"train": [0], "val": [1], "test": [2]}
    pointers = [
        pointer,
        EpisodeActionPointer(1, 17, 0, 0, 16, 33, "reach"),
        EpisodeActionPointer(2, 18, 0, 0, 33, 51, "reach"),
    ]
    windows = deterministic_windows(
        pointers, split, limits={"train": None, "val": None, "test": None}
    )
    assert [item.anchor_frame for item in windows["train"]] == [0]
    assert [item.anchor_frame for item in windows["val"]] == [0, 1]
    assert [item.anchor_frame for item in windows["test"]] == [0, 1, 2]
    assert windows["test"][-1].anchor_frame + 15 == 17


def test_frozen_episode_split_leakage_is_rejected() -> None:
    assert validate_episode_splits({"train": [0], "val": [1], "test": [2]}) == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    with pytest.raises(ValueError, match="leakage"):
        validate_episode_splits({"train": [0], "val": [0], "test": [2]})


def test_action_activity_labels_have_required_semantics() -> None:
    action = np.zeros((2, 16, 58), dtype=np.float32)
    action[0, :, :7] = np.arange(16)[:, None]
    action[1, :, 36:] = np.arange(16)[:, None]
    labels = action_activity(action)
    assert labels["active_side"].tolist() == [0, 1]
    assert labels["arm_vs_hand"].tolist() == [0, 1]
    assert np.all(labels["magnitude"] > 0)


def test_different_episode_control_never_reuses_episode() -> None:
    class Cache:
        episode_id = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)

        def __len__(self) -> int:
            return len(self.episode_id)

    cache = Cache()
    source = np.arange(len(cache), dtype=np.int64)
    negative = different_episode_indices(cache, source)
    assert np.all(cache.episode_id[source] != cache.episode_id[negative])


def test_compact_forward_shape_bounds_and_determinism(fake_source: FakeReleasedSource) -> None:
    model = TReXActionBootstrap(fake_source, initialization="old_rows_mean", seed=3).eval()
    state = torch.randn(2, 128)
    action = torch.randn(2, 16, 128)
    embodiment = torch.full((2,), TREX_EMBODIMENT_ID, dtype=torch.long)
    first = model(state, action, embodiment)
    second = model(state, action, embodiment)
    assert first["z_action"].shape == (2, 8, 32)
    assert first["prediction"].shape == (2, 16, 128)
    assert torch.isfinite(first["z_action"]).all()
    assert torch.equal(first["z_action"], second["z_action"])
    with pytest.raises(ValueError, match="only explicit T-Rex"):
        model(state, action, torch.full((2,), GR1_EMBODIMENT_ID))
    with pytest.raises(IndexError, match="outside expanded"):
        model(state, action, torch.full((2,), EXPANDED_CATEGORY_CAPACITY))


def test_gradient_routes_only_to_trex_category_parameters(fake_source: FakeReleasedSource) -> None:
    model = TReXActionBootstrap(fake_source, initialization="old_rows_mean", seed=4)
    model.train()
    state = torch.randn(2, 128)
    action = torch.randn(2, 16, 128)
    embodiment = torch.full((2,), TREX_EMBODIMENT_ID, dtype=torch.long)
    output = model(state, action, embodiment)
    output["prediction"].square().mean().backward()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    with_grad = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert with_grad == trainable
    assert trainable
    assert all(
        name.startswith(("action_encoder.action_conv_encoder.", "action_encoder.state_encoder.", "action_decoder.resnet_decoder."))
        for name in trainable
    )
    assert all(parameter.grad is None for parameter in model.action_path.parameters())
    assert all(parameter.grad is None for parameter in model.action_encoder.m_former.parameters())
    assert all(parameter.grad is None for parameter in model.action_decoder.m_former.parameters())


def test_overlay_checkpoint_cold_reload(fake_source: FakeReleasedSource, tmp_path: Path) -> None:
    model = TReXActionBootstrap(fake_source, initialization="old_rows_mean", seed=5).eval()
    state = torch.randn(1, 128)
    action = torch.randn(1, 16, 128)
    embodiment = torch.full((1,), TREX_EMBODIMENT_ID, dtype=torch.long)
    expected = model(state, action, embodiment)["z_action"]
    path = tmp_path / "bootstrap.pt"
    digest = save_bootstrap_checkpoint(path, model, metadata={"selection_split": "validation"})
    assert len(digest) == 64
    reloaded, metadata = load_bootstrap_checkpoint(path, fake_source)
    reloaded.eval()
    actual = reloaded(state, action, embodiment)["z_action"]
    assert torch.equal(expected, actual)
    assert metadata == {"selection_split": "validation"}


def test_a2_adapter_is_trex_owned_and_cold_reloadable(
    fake_source: FakeReleasedSource, tmp_path: Path
) -> None:
    model = TReXActionBootstrap(
        fake_source,
        initialization="old_rows_mean",
        seed=6,
        enable_a2_adapter=True,
    )
    state = torch.randn(2, 128)
    action = torch.randn(2, 16, 128)
    embodiment = torch.full((2,), TREX_EMBODIMENT_ID, dtype=torch.long)
    model(state, action, embodiment)["prediction"].square().mean().backward()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert {
        "a2_adapter.norm.weight",
        "a2_adapter.norm.bias",
        "a2_adapter.proj.weight",
        "a2_adapter.proj.bias",
    } <= trainable
    assert all(parameter.grad is None for parameter in model.action_path.parameters())
    assert all(parameter.grad is None for parameter in model.action_encoder.m_former.parameters())
    assert all(parameter.grad is None for parameter in model.action_decoder.m_former.parameters())

    model.eval()
    expected = model(state, action, embodiment)["z_action"]
    path = tmp_path / "a2-bootstrap.pt"
    save_bootstrap_checkpoint(path, model, metadata={"selection_split": "validation"})
    reloaded, _ = load_bootstrap_checkpoint(path, fake_source)
    assert reloaded.train_stage == "A2"
    reloaded.eval()
    actual = reloaded(state, action, embodiment)["z_action"]
    assert torch.equal(expected, actual)


def test_a2_validation_selection_prefers_complete_gate_pass() -> None:
    acceptance = {"normalized_mse_max": 1.0, "minimum_temporal_loss_ratio": 1.05}
    failing_temporal = {
        "reversed_ratio_to_correct": 1.01,
        "shuffled_ratio_to_correct": 1.02,
        "different_episode_ratio_to_correct": 4.0,
    }
    passing_temporal = {
        "reversed_ratio_to_correct": 1.06,
        "shuffled_ratio_to_correct": 1.07,
        "different_episode_ratio_to_correct": 4.0,
    }
    failing_key, failing_gate, failing_shortfall = validation_selection_key(
        stage="A2",
        reconstruction_mse=0.1,
        temporal=failing_temporal,
        acceptance=acceptance,
    )
    passing_key, passing_gate, passing_shortfall = validation_selection_key(
        stage="A2",
        reconstruction_mse=0.9,
        temporal=passing_temporal,
        acceptance=acceptance,
    )
    assert not failing_gate and failing_shortfall > 0
    assert passing_gate and passing_shortfall == 0
    assert passing_key < failing_key


def test_evaluation_json_accepts_numpy_scalars(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    evaluation_atomic_json(
        path,
        {"gate": np.bool_(True), "score": np.float32(0.25), "count": np.int64(4)},
    )
    assert json.loads(path.read_text()) == {"gate": True, "score": 0.25, "count": 4}


def test_track_a_config_has_no_private_paths() -> None:
    path = ROOT / "configs/tactile_unit/s3_3_action_bootstrap.json"
    value = path.read_text()
    json.loads(value)
    config = json.loads(value)
    assert config["gpu"]["allowed_physical"] == [1, 2, 3]
    assert config["gpu"]["forbidden_physical"] == [0]
    assert "/" + "home/" not in value
    assert "/" + "mnt/" not in value
    assert "Author" + "ization" not in value
    assert "Bear" + "er " not in value
