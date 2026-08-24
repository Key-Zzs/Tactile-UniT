from pathlib import Path

import pytest
import torch

from gr00t.contact_dynamics.teacher import load_frozen_teacher, parameter_digest


def test_accepted_teacher_loads_frozen_when_checkpoint_is_available():
    checkpoint = Path(".local/experiments/tactile_teacher/s1_teacher/best.pt")
    if not checkpoint.is_file():
        pytest.skip("accepted local S1 Teacher checkpoint is unavailable")
    teacher, identity = load_frozen_teacher(checkpoint)
    before = parameter_digest(teacher)
    with torch.inference_mode():
        first = teacher.encode(torch.zeros(2, 16, 60))
        second = teacher.encode(torch.zeros(2, 16, 60))
    assert identity["latent_dim"] == 256
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert torch.equal(first, second)
    assert parameter_digest(teacher) == before
