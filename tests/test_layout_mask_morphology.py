import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from sam3_demo import app as demo_module


class LayoutMaskMorphologyTest(unittest.TestCase):
    def test_zero_preserves_binary_mask(self):
        mask = np.zeros((11, 11), dtype=bool)
        mask[3:8, 4:7] = True

        result = demo_module._apply_layout_mask_morphology(mask, 0)

        self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(result, mask)
        self.assertIsNot(result, mask)

    def test_positive_pixels_dilate_by_requested_radius(self):
        mask = np.zeros((15, 15), dtype=bool)
        mask[7, 7] = True
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        expected = cv2.dilate(
            mask.astype(np.uint8),
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

        result = demo_module._apply_layout_mask_morphology(mask, 2)

        np.testing.assert_array_equal(result, expected)
        self.assertGreater(int(result.sum()), int(mask.sum()))

    def test_negative_pixels_erode_by_requested_radius(self):
        mask = np.zeros((15, 15), dtype=bool)
        mask[3:12, 3:12] = True
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        expected = cv2.erode(
            mask.astype(np.uint8),
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

        result = demo_module._apply_layout_mask_morphology(mask, -2)

        np.testing.assert_array_equal(result, expected)
        self.assertLess(int(result.sum()), int(mask.sum()))
    def test_agent_draft_matches_authoritative_pipeline_pixel_for_pixel(self):
        rgb = np.full((40, 50, 3), 255, dtype=np.uint8)
        rgb[8:32, 10:40] = (255, 0, 0)
        rgb[18:22, 20:24] = (255, 255, 255)
        image = Image.fromarray(rgb)
        params = {
            "threshold": 12,
            "invert": False,
            "open_kernel": 0,
            "close_kernel": 3,
            "min_component_area": 0,
            "region_mode": "all",
            "morph_pixels": 1,
        }

        draft_image, draft_mask, draft_contours, draft_params = (
            demo_module._compute_layout_mask_draft(
                image,
                params["threshold"],
                params["invert"],
                params["open_kernel"],
                params["close_kernel"],
                params["min_component_area"],
                params["region_mode"],
                params["morph_pixels"],
            )
        )
        expected_image, expected_mask = demo_module._binarize_layout_image(
            image, 12, False, 0, 3, 1
        )
        expected_mask = demo_module._filter_layout_components(expected_mask, 0, "all")
        expected_contours = demo_module._layout_mask_contours(expected_mask)

        np.testing.assert_array_equal(draft_mask, expected_mask)
        self.assertEqual(draft_contours, expected_contours)
        self.assertEqual(draft_params, params)
        self.assertEqual(draft_image.size, expected_image.size)

    def test_pixels_are_clamped_to_ui_limit(self):
        self.assertEqual(
            demo_module._normalize_layout_morph_pixels(999),
            demo_module._LAYOUT_MASK_MORPH_LIMIT_PX,
        )
        self.assertEqual(
            demo_module._normalize_layout_morph_pixels(-999),
            -demo_module._LAYOUT_MASK_MORPH_LIMIT_PX,
        )

    def test_generation_passes_pixels_and_persists_normalized_value(self):
        image = Image.new("RGB", (12, 10), (255, 255, 255))
        mask = np.zeros((10, 12), dtype=bool)
        mask[2:8, 3:9] = True
        saved_state = {
            "layout_id": "layout-test",
            "session_id": "session-test",
            "source_mask_pixel_sha256": "mask-hash",
        }

        with (
            mock.patch.object(
                demo_module,
                "_binarize_layout_image",
                return_value=(image, mask),
            ) as binarize,
            mock.patch.object(
                demo_module,
                "_filter_layout_components",
                side_effect=lambda value, *_args: value,
            ),
            mock.patch.object(demo_module, "_layout_mask_contours", return_value=[]),
            mock.patch.object(
                demo_module,
                "_save_layout_mask_files",
                return_value=(saved_state, "mask.png", "contours.json", image),
            ) as save_files,
            mock.patch.object(demo_module, "_layout_editor_payload", return_value={}),
        ):
            result = demo_module._run_layout_mask_page(
                {"session_id": "session-test"},
                {},
                image,
                12,
                False,
                0,
                0,
                0,
                "all",
                -3,
            )

        self.assertEqual(result[0]["layout_id"], "layout-test")
        binarize.assert_called_once_with(image, 12, False, 0, 0, -3)
        params = save_files.call_args.args[-1]
        self.assertEqual(params["morph_pixels"], -3)


if __name__ == "__main__":
    unittest.main()
