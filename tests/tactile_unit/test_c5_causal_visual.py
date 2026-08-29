import inspect

import numpy as np
import pytest
import torch

from gr00t.tactile_unit.c4_availability_conditioning import ContactFallbackPredictor
from gr00t.tactile_unit.c5_causal_visual import (
    CausalFrameSelection,
    CausalVisualEncoder,
    CausalVisionSubstituter,
    DirectCausalContactPredictor,
    ModularCausalContactPredictor,
    VisualSupport,
)
from scripts.tactile_unit.train_c5_causal_fallback import past_time_indices


def test_current_and_history_indices_are_exactly_causal_and_episode_local():
    current = CausalFrameSelection.create(VisualSupport.CURRENT_FRAME, 4, 20, 100)
    history = CausalFrameSelection.create(VisualSupport.CAUSAL_HISTORY_8, 4, 20, 100)
    assert current.frame_indices == (20,)
    assert history.frame_indices == tuple(range(13, 21))
    assert max(history.frame_indices) == history.anchor_t
    with pytest.raises(ValueError, match="episode boundary"):
        CausalFrameSelection.create(VisualSupport.CAUSAL_HISTORY_8, 4, 6, 100)


def test_wrong_time_control_is_past_only_even_at_episode_boundaries():
    episode = np.asarray([1, 1, 1, 2, 2])
    anchor = np.asarray([7, 9, 11, 8, 10])
    rows = past_time_indices(episode, anchor)
    assert np.array_equal(rows, np.asarray([0, 0, 1, 3, 3]))
    assert np.all(episode[rows] == episode)
    assert np.all(anchor[rows] <= anchor)


@pytest.mark.parametrize("support,frames", [(VisualSupport.CURRENT_FRAME, 1), (VisualSupport.CAUSAL_HISTORY_8, 8)])
def test_causal_visual_encoder_is_bounded_and_outputs_eight_shared_slots(support, frames):
    model = CausalVisualEncoder(support)
    value = torch.randn(3, frames, 8, 32)
    assert model(value).shape == (3, 8, 32)
    assert model.layers <= 2 and model.heads <= 4 and model.mlp_width <= 128


def test_frozen_visual_input_must_be_stop_gradient():
    model = CausalVisualEncoder(VisualSupport.CURRENT_FRAME)
    with pytest.raises(ValueError, match="stop-gradient"):
        model(torch.randn(2, 1, 8, 32, requires_grad=True))


def test_direct_predictor_interface_has_no_h_true_uv_contact_or_pair_id():
    parameters = set(inspect.signature(DirectCausalContactPredictor.forward).parameters)
    assert parameters == {"self", "c_v", "u_a"}
    assert parameters.isdisjoint({"h_current", "h_future", "u_v", "u_c", "z_c", "r_c_priv", "pair_id"})
    prediction, visual = DirectCausalContactPredictor(visual_head=True)(torch.randn(2, 8, 32), torch.randn(2, 8, 32))
    assert prediction.shape == visual.shape == (2, 8, 32)


def test_modular_predictor_accepts_true_uv_only_as_external_loss_target_not_input():
    substituter = CausalVisionSubstituter()
    frozen = ContactFallbackPredictor("VA")
    model = ModularCausalContactPredictor(substituter, frozen)
    prediction, u_hat_v = model(torch.randn(2, 8, 32), torch.randn(2, 8, 32))
    assert prediction.shape == u_hat_v.shape == (2, 8, 32)
    assert set(inspect.signature(model.forward).parameters) == {"c_v", "u_a"}
    assert all(not parameter.requires_grad for parameter in model.frozen_f_va.parameters())
    model.train()
    assert model.training is True and model.substituter.training is True
    assert model.frozen_f_va.training is False


def test_only_c5_modules_receive_gradients():
    visual = CausalVisualEncoder(VisualSupport.CURRENT_FRAME)
    substituter = CausalVisionSubstituter()
    frozen = ContactFallbackPredictor("VA")
    model = ModularCausalContactPredictor(substituter, frozen)
    prediction, _ = model(visual(torch.randn(2, 1, 8, 32)), torch.randn(2, 8, 32))
    prediction.square().mean().backward()
    assert any(parameter.grad is not None for parameter in visual.parameters())
    assert any(parameter.grad is not None for parameter in substituter.parameters())
    assert all(parameter.grad is None for parameter in frozen.parameters())
