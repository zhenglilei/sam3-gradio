"""CPU-only checks for optional GE2 adapter loading."""

import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from sam3_demo.model_adapter import LoRALinear, load_ge2_adapter
from sam3_demo.model_worker import Sam3WorkerRuntime


def small_model():
    model = nn.Module()
    model.backbone = nn.Linear(3, 3)
    model.inst_interactive_predictor = nn.Module()
    tracker = model.inst_interactive_predictor.model = nn.Module()
    tracker.sam_mask_decoder = nn.Module()
    tracker.sam_mask_decoder.transformer = nn.Module()
    tracker.sam_mask_decoder.transformer.q_proj = nn.Linear(3, 4)
    tracker.sam_mask_decoder.head = nn.Linear(4, 1)
    tracker.sam_mask_decoder.dynamic_multimask_via_stability = True
    tracker.sam_prompt_encoder = nn.Linear(2, 3)
    return model


class ModelAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "adapter.pt"
        self.model = small_model()
        reference = copy.deepcopy(self.model)
        tracker = reference.inst_interactive_predictor.model
        transformer = tracker.sam_mask_decoder.transformer
        transformer.q_proj = LoRALinear(transformer.q_proj, 2)
        with torch.no_grad():
            tracker.sam_mask_decoder.head.weight.fill_(0.75)
            tracker.sam_prompt_encoder.weight.fill_(0.25)
        self.reference = reference
        self.payload = {
            "format": "ge2_layout_lora_v1",
            "config": {"rank": 2, "heads": True, "prompt": False},
            "step": 700,
            "preprocessing": {"dynamic_multimask_via_stability": False},
            "decoder": tracker.sam_mask_decoder.state_dict(),
            "prompt_encoder": tracker.sam_prompt_encoder.state_dict(),
        }

    def save_and_load(self):
        torch.save(self.payload, self.path)
        return load_ge2_adapter(self.model, str(self.path))

    def test_load_matches_training_formula_and_preserves_backbone(self):
        backbone = copy.deepcopy(self.model.backbone.state_dict())
        info = self.save_and_load()
        tracker = self.model.inst_interactive_predictor.model
        projection = tracker.sam_mask_decoder.transformer.q_proj
        reference = self.reference.inst_interactive_predictor.model
        x = torch.randn(2, 3)
        expected = projection.base(x) + 2 * projection.lora_b(projection.lora_a(x.float()))
        torch.testing.assert_close(projection(x), expected)
        for name in ("sam_mask_decoder", "sam_prompt_encoder"):
            for key, value in getattr(reference, name).state_dict().items():
                torch.testing.assert_close(getattr(tracker, name).state_dict()[key], value)
        for key, value in backbone.items():
            torch.testing.assert_close(self.model.backbone.state_dict()[key], value)
        self.assertFalse(tracker.sam_mask_decoder.dynamic_multimask_via_stability)
        self.assertFalse(self.model.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))
        self.assertEqual(info["projections"], 1)
        self.assertEqual(info["step"], 700)
        self.assertEqual(info["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_wrong_format_is_rejected(self):
        self.payload["format"] = "another_adapter"
        with self.assertRaisesRegex(ValueError, "ge2_layout_lora_v1"):
            self.save_and_load()

    def test_invalid_rank_is_rejected(self):
        self.payload["config"]["rank"] = 0
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.save_and_load()

    def test_missing_decoder_tensor_is_rejected(self):
        del self.payload["decoder"]["head.weight"]
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.save_and_load()

    def test_unexpected_decoder_tensor_is_rejected(self):
        self.payload["decoder"]["unrelated.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            self.save_and_load()

    def test_wrong_shape_is_rejected(self):
        self.payload["decoder"]["head.weight"] = torch.ones(2, 4)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            self.save_and_load()

    def test_missing_prompt_encoder_tensor_is_rejected(self):
        del self.payload["prompt_encoder"]["weight"]
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.save_and_load()

    def test_absent_checkpoint_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            load_ge2_adapter(self.model, str(self.path))

    def worker_load(self, adapter_path):
        self.model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size = (4, 4)
        runtime = Sam3WorkerRuntime(device="cpu")
        runtime._import_runtime = Mock()
        runtime._torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        runtime._checkpoint_paths = Mock(return_value=(self.path, self.path))
        builder = Mock(return_value=self.model)
        modules = {
            "sam3.model.sam3_image_processor": SimpleNamespace(Sam3Processor=Mock()),
            "sam3.model_builder": SimpleNamespace(build_sam3_image_model=builder),
        }
        torch.save(self.payload, self.path)
        with patch.dict(sys.modules, modules), patch.dict(
            os.environ, {"SAM3_ADAPTER_CHECKPOINT": adapter_path}
        ):
            result = runtime.load()
            self.assertEqual(runtime.load(), result)
        builder.assert_called_once()
        self.assertTrue(builder.call_args.kwargs["enable_inst_interactivity"])
        return result

    def test_worker_without_adapter_keeps_base_model(self):
        result = self.worker_load("")
        self.assertNotIn("adapter", result)
        projection = self.model.inst_interactive_predictor.model.sam_mask_decoder.transformer.q_proj
        self.assertIsInstance(projection, nn.Linear)

    def test_worker_loads_configured_adapter_once(self):
        result = self.worker_load(str(self.path))
        self.assertEqual(result["adapter"]["format"], "ge2_layout_lora_v1")
        self.assertEqual(result["adapter"]["rank"], 2)

    def test_base_predictions_keep_multimask_output(self):
        runtime = Sam3WorkerRuntime(device="cpu")
        runtime._capabilities = {"mask_input_size": [4, 4]}
        self.assertTrue(runtime._common_inst_kwargs({})["multimask_output"])

    def test_ge2_predictions_select_trained_single_mask_head(self):
        runtime = Sam3WorkerRuntime(device="cpu")
        runtime._capabilities = {
            "mask_input_size": [4, 4], "adapter": {"format": "ge2_layout_lora_v1"}
        }
        self.assertFalse(runtime._common_inst_kwargs({})["multimask_output"])


if __name__ == "__main__":
    unittest.main()
