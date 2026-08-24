import os
import unittest
from unittest.mock import patch

from scripts.groot_tokenizer_train import (
    configure_local_torch_load_resume_compat,
    configure_single_gpu_visibility,
)


class SingleGpuVisibilityTest(unittest.TestCase):
    def test_defaults_to_zero_when_no_gpu_is_selected(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_single_gpu_visibility()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_preserves_explicit_gpu_selection(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}, clear=True):
            configure_single_gpu_visibility()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_trusted_local_resume_bypass_is_explicit_and_scoped(self):
        import torch
        import transformers.trainer as trainer_module

        original = trainer_module.check_torch_load_is_safe
        try:
            with patch.dict(
                os.environ,
                {"GR00T_ALLOW_TRUSTED_LOCAL_TORCH_LOAD": "1"},
                clear=False,
            ), patch.object(torch, "__version__", "2.5.1"):
                self.assertTrue(configure_local_torch_load_resume_compat(True))
                self.assertIsNone(trainer_module.check_torch_load_is_safe())
        finally:
            trainer_module.check_torch_load_is_safe = original
