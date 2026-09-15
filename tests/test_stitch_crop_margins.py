import unittest
import numpy as np
import gradio as gr
from PIL import Image
from sam3_demo.stitch_callbacks import crop_tile_margins


class CropMarginsTest(unittest.TestCase):
    def test_four_sides_preserve_original(self):
        a = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        image = Image.fromarray(a)
        out = crop_tile_margins([image], 2, 3, 4, 5)[0]
        np.testing.assert_array_equal(np.asarray(out), a[2:17, 4:25])
        np.testing.assert_array_equal(np.asarray(image), a)

    def test_invalid_values_and_extents(self):
        for value in (-1, 1.5, float('nan'), float('inf'), None, 20):
            with self.subTest(value=value), self.assertRaises(gr.Error):
                crop_tile_margins([Image.new('RGB', (30, 20))], top=value)

    def test_all_tiles_validated(self):
        with self.assertRaises(gr.Error):
            crop_tile_margins([Image.new('RGB', (30, 20)), Image.new('RGB', (5, 5))], left=5)

    def test_zero_is_identity(self):
        self.assertEqual(crop_tile_margins([Image.new('RGB', (30, 20))])[0].size, (30, 20))
