import numpy as np
import pytest
import torch

from gr00t.simulation.s4_3_act import (
    CausalACTPolicy,
    CausalRawTactileEncoder,
    PolicyNormalization,
)
from gr00t.simulation.s4_3_runtime import (
    ActionChunkQueue,
    CausalHistoryBuffer,
    mapped_force_metrics,
)


def normalization() -> PolicyNormalization:
    return PolicyNormalization(
        proprio_mean=np.zeros(22, dtype=np.float32),
        proprio_std=np.ones(22, dtype=np.float32),
        action_min=-np.ones(22, dtype=np.float32),
        action_max=np.ones(22, dtype=np.float32),
        tactile_mean=np.zeros(30, dtype=np.float32),
        tactile_std=np.ones(30, dtype=np.float32),
    )


@pytest.mark.parametrize("variant", ("P0", "P1", "P2", "P3"))
def test_act_variants_produce_exact_bounded_chunk(variant: str) -> None:
    torch.manual_seed(0)
    model = CausalACTPolicy(variant, normalization()).eval()
    kwargs = {}
    if variant == "P1":
        kwargs["tactile_history"] = torch.zeros(2, 26, 30)
    if variant in {"P2", "P3"}:
        kwargs["contact_state"] = torch.zeros(2, 256)
    with torch.inference_mode():
        first = model(torch.zeros(2, 8, 32), torch.zeros(2, 22), **kwargs)
        second = model(torch.zeros(2, 8, 32), torch.zeros(2, 22), **kwargs)
    assert first["physical_action"].shape == (2, 27, 22)
    assert torch.isfinite(first["physical_action"]).all()
    assert torch.all(first["physical_action"] >= -1.0)
    assert torch.all(first["physical_action"] <= 1.0)
    assert torch.equal(first["physical_action"], second["physical_action"])


def test_p2_p3_inference_architecture_and_initialization_match() -> None:
    torch.manual_seed(7)
    p2 = CausalACTPolicy("P2", normalization())
    torch.manual_seed(7)
    p3 = CausalACTPolicy("P3", normalization())
    assert p2.trainable_parameter_count == p3.trainable_parameter_count
    assert p2.state_dict().keys() == p3.state_dict().keys()
    for name in p2.state_dict():
        assert torch.equal(p2.state_dict()[name], p3.state_dict()[name]), name


def test_raw_tactile_encoder_is_under_frozen_budget() -> None:
    encoder = CausalRawTactileEncoder()
    assert encoder.parameter_count == 63552
    assert encoder.parameter_count <= 100000
    assert encoder(torch.zeros(3, 26, 30)).shape == (3, 256)


def test_act_training_loss_and_deterministic_prior() -> None:
    torch.manual_seed(0)
    model = CausalACTPolicy("P0", normalization())
    target = torch.zeros(2, 27, 22)
    output = model(
        torch.zeros(2, 8, 32),
        torch.zeros(2, 22),
        target_action=target,
        sample_posterior=False,
    )
    losses = model.act_loss(output, target)
    assert set(losses) == {"action_l1", "kl", "act_total"}
    losses["act_total"].backward()
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0
    model.eval()
    with torch.inference_mode():
        inference = model(torch.zeros(2, 8, 32), torch.zeros(2, 22))
    assert torch.equal(inference["posterior_mu"], torch.zeros_like(inference["posterior_mu"]))


def test_causal_history_warmup_and_no_future_provenance() -> None:
    history = CausalHistoryBuffer("episode")
    for step in range(26):
        history.append(step, np.full(30, step, dtype=np.float32))
    observation = history.observation(
        np.zeros((16, 16, 3), dtype=np.uint8), np.zeros(22, dtype=np.float32)
    )
    assert observation.tactile_history.shape == (26, 30)
    assert observation.current_control_step == 25
    assert observation.provenance[2].source_min_step == 0
    assert observation.provenance[2].source_max_step == 25
    with pytest.raises(ValueError, match="consecutive"):
        history.append(27, np.zeros(30, dtype=np.float32))


def test_action_queue_executes_five_then_requires_replan() -> None:
    queue = ActionChunkQueue("episode")
    plan = np.arange(27 * 22, dtype=np.float32).reshape(27, 22)
    provenance = queue.set_plan(plan, 25)
    assert provenance.role == "PLAN"
    for index in range(5):
        assert np.array_equal(queue.pop(), plan[index])
    assert queue.needs_replan
    with pytest.raises(RuntimeError, match="fresh plan"):
        queue.pop()


def test_mapped_force_metrics_use_frozen_five_region_schema() -> None:
    tactile = np.zeros((3, 30), dtype=np.float32)
    tactile[1, 0] = 1.0
    tactile[1, 1] = 2.0
    tactile[1, 2] = 3.0
    metrics = mapped_force_metrics(tactile)
    assert metrics["free_to_contact"] == 1
    assert metrics["contact_to_free"] == 1
    assert metrics["peak_normal_force"] == 2.0
    assert metrics["integrated_normal_force"] == pytest.approx(0.04)
    assert metrics["peak_tangential_force"] == 3.0
