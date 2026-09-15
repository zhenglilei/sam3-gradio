import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3_demo.lama_runtime import LamaRuntime, _prepare_image_and_mask


class IdentityModel:
    def __call__(self, image, mask):
        return image


class LamaRuntimeTests(unittest.TestCase):
    def test_prepare_pads_to_modulo_eight_and_binarizes_mask(self):
        image = Image.new("RGB", (37, 29), "white")
        mask = np.zeros((29, 37), dtype=np.uint8)
        mask[2:5, 3:7] = 127
        image_tensor, mask_tensor, original_size = _prepare_image_and_mask(image, mask)
        self.assertEqual(tuple(image_tensor.shape), (1, 3, 32, 40))
        self.assertEqual(tuple(mask_tensor.shape), (1, 1, 32, 40))
        self.assertEqual(original_size, (37, 29))
        self.assertEqual(set(torch.unique(mask_tensor).tolist()), {0.0, 1.0})

    def test_output_is_cropped_back_to_original_size(self):
        runtime = LamaRuntime(Path(tempfile.gettempdir()) / "unused-big-lama.pt")
        runtime._model = IdentityModel()
        image = Image.new("RGB", (37, 29), (30, 40, 50))
        mask = np.zeros((29, 37), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        result = runtime.inpaint(image, mask)
        self.assertEqual(result.size, image.size)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(image))

    def test_empty_mask_does_not_load_model(self):
        runtime = LamaRuntime(Path(tempfile.gettempdir()) / "missing-big-lama.pt")
        with self.assertRaisesRegex(ValueError, "修复区域为空"):
            runtime.inpaint(Image.new("RGB", (8, 8)), Image.new("L", (8, 8)))


if __name__ == "__main__":
    unittest.main()
