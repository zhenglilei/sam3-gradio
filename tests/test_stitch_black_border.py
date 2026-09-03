from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from sam3_demo.stitch_black_border import trim_black_border, trim_black_borders


def _content(height: int = 48, width: int = 64) -> np.ndarray:
    rng = np.random.RandomState(123)
    return rng.randint(70, 220, size=(height, width, 3), dtype=np.uint8)


class StitchBlackBorderTest(unittest.TestCase):
    def test_four_side_black_bars_are_trimmed(self):
        original = _content()
        framed = np.zeros_like(original)
        framed[3:-4, 5:-6] = original[3:-4, 5:-6]

        result, record = trim_black_border(Image.fromarray(framed, mode="RGB"))

        self.assertTrue(record["applied"])
        self.assertEqual(record["crop_bbox"], [5, 3, 58, 44])
        self.assertEqual(record["trim"], {"left": 5, "top": 3, "right": 6, "bottom": 4})
        np.testing.assert_array_equal(np.asarray(result), original[3:-4, 5:-6])

    def test_jpeg_near_black_border_is_trimmed(self):
        original = _content(50, 70)
        framed = np.full_like(original, 18)
        framed[4:-4, 3:-5] = original[4:-4, 3:-5]
        buffer = io.BytesIO()
        Image.fromarray(framed, mode="RGB").save(buffer, format="JPEG", quality=75)
        jpeg = Image.open(io.BytesIO(buffer.getvalue()))

        result, record = trim_black_border(jpeg)

        self.assertTrue(record["applied"])
        self.assertLess(result.width, jpeg.width)
        self.assertLess(result.height, jpeg.height)
        self.assertEqual(result.size, (62, 42))

    def test_internal_black_line_is_not_trimmed(self):
        source = _content()
        source[23, 2:-2] = 0
        image = Image.fromarray(source, mode="RGB")

        result, record = trim_black_border(image)

        self.assertFalse(record["applied"])
        self.assertEqual(record["reason"], "no_black_border")
        self.assertEqual(result.size, image.size)
        np.testing.assert_array_equal(np.asarray(result), source)

    def test_edge_local_dark_structure_is_not_trimmed(self):
        source = _content()
        source[:8, :20] = 0
        image = Image.fromarray(source, mode="RGB")

        result, record = trim_black_border(image)

        self.assertFalse(record["applied"])
        self.assertEqual(record["reason"], "no_black_border")
        np.testing.assert_array_equal(np.asarray(result), source)

    def test_all_black_falls_back_without_change(self):
        source = np.zeros((24, 32, 3), dtype=np.uint8)
        result, record = trim_black_border(Image.fromarray(source, mode="RGB"))

        self.assertFalse(record["applied"])
        self.assertEqual(record["reason"], "all_black")
        np.testing.assert_array_equal(np.asarray(result), source)

    def test_overlarge_single_side_candidate_falls_back(self):
        source = _content(40, 60)
        source[:, :30] = 0
        image = Image.fromarray(source, mode="RGB")

        result, record = trim_black_border(image)

        self.assertFalse(record["applied"])
        self.assertEqual(record["reason"], "border_crop_too_large")
        self.assertEqual(record["candidate_trim"][0], 30)
        np.testing.assert_array_equal(np.asarray(result), source)

    def test_full_width_dark_header_is_not_mistaken_for_a_border(self):
        source = _content(40, 60)
        source[:10, :] = 0
        image = Image.fromarray(source, mode="RGB")

        result, record = trim_black_border(image)

        self.assertFalse(record["applied"])
        self.assertEqual(record["reason"], "border_crop_too_large")
        np.testing.assert_array_equal(np.asarray(result), source)

    def test_no_border_is_bitwise_equivalent_and_batch_warnings_are_stable(self):
        source = _content(18, 22)
        image = Image.fromarray(source, mode="RGB")

        result, record = trim_black_border(image)
        batch, records, warnings = trim_black_borders([image, image])

        self.assertFalse(record["applied"])
        self.assertEqual(record["reason"], "no_black_border")
        self.assertEqual(result.tobytes(), image.tobytes())
        self.assertEqual(len(records), 2)
        self.assertEqual(warnings, [])
        self.assertEqual(batch[0].tobytes(), image.tobytes())


if __name__ == "__main__":
    unittest.main()
