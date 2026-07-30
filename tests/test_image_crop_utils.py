import unittest

import numpy as np
from PIL import Image

from image_crop_utils import crop_pil_image, normalize_crop_box, whole_image_crop_box


class ImageCropUtilsTests(unittest.TestCase):
    def test_normalize_crop_box_clamps_and_uses_exclusive_end(self):
        self.assertEqual(
            normalize_crop_box((9.8, 8.2), (-2.0, 2.1), 10, 9),
            (0, 2, 10, 9),
        )

    def test_normalize_crop_box_rejects_small_or_nonfinite_gesture(self):
        with self.assertRaisesRegex(ValueError, "at least 4x4"):
            normalize_crop_box((1, 1), (4, 4), 20, 20)
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_crop_box((1, 1), (float("nan"), 8), 20, 20)

    def test_crop_preserves_authoritative_pixels(self):
        array = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        cropped = crop_pil_image(Image.fromarray(array), (2, 1, 8, 7))
        np.testing.assert_array_equal(np.asarray(cropped), array[1:7, 2:8])

    def test_whole_image_box(self):
        self.assertEqual(whole_image_crop_box(11, 7), (0, 0, 11, 7))


if __name__ == "__main__":
    unittest.main()
