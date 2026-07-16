import ast
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _load_overlay_namespace():
    source = (ROOT / "sam3_gradio_demo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    required_functions = {
        "_active_instances",
        "_is_pcs_mode",
        "_is_pvs_manual_mode",
        "_is_layout_mask_mode",
        "_is_pvs_pool_mode",
        "_pvs_pending_bbox_records",
        "_overlay",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in required_functions
    ]
    namespace = {
        "np": np,
        "cv2": cv2,
        "Image": Image,
        "MODE_PCS": "PCS Auto",
        "MODE_PVS": "PVS Manual",
        "MODE_LAYOUT": "Layout Mask",
        "_layout_cache_get": lambda *_args, **_kwargs: None,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT / "sam3_gradio_demo.py"), "exec"), namespace)
    return namespace


class PcsPvsOverlayTest(unittest.TestCase):
    def setUp(self):
        self.base_rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
        self.base_image = Image.fromarray(self.base_rgb)
        self.image_state = {"image_id": "overlay-test-image", "width": 64, "height": 64}
        self.namespace = _load_overlay_namespace()
        self.namespace["_workspace"] = lambda _image_state: {"image": self.base_image}
        self.overlay = self.namespace["_overlay"]

    def test_pvs_prompt_bbox_remains_on_left_after_batch_segmentation(self):
        predicted_mask = np.zeros((64, 64), dtype=bool)
        predicted_mask[35:50, 35:50] = True
        pcs_state = {"instances": {}}
        pvs_state = {
            "active_instance_id": 1,
            "pending_boxes": [],
            "pending_bbox_records": [],
            "instances": {
                1: {
                    "id": 1,
                    "status": "draft",
                    "mask_fullres_bool": predicted_mask,
                    "box_xyxy_px": [35.0, 35.0, 49.0, 49.0],
                    "prompt_history": [
                        {
                            "op": "create_from_pending_bbox",
                            "box_xyxy_px": [5.0, 6.0, 24.0, 25.0],
                        }
                    ],
                }
            },
        }

        left_image = np.asarray(
            self.overlay(
                self.image_state,
                pcs_state,
                pvs_state,
                "PVS Manual",
                show_instances=False,
            )
        )
        right_image = np.asarray(
            self.overlay(
                self.image_state,
                pcs_state,
                pvs_state,
                "PVS Manual",
                show_instances=True,
                show_interaction_prompts=False,
            )
        )

        self.assertFalse(np.array_equal(left_image[6, 14], self.base_rgb[6, 14]))
        np.testing.assert_array_equal(left_image[42, 42], self.base_rgb[42, 42])
        np.testing.assert_array_equal(right_image[6, 14], self.base_rgb[6, 14])

    def test_pcs_mask_overlay_has_strong_green_contrast(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[16:48, 16:48] = True
        pcs_state = {
            "instances": {
                1: {
                    "id": 1,
                    "status": "draft",
                    "mask_fullres_bool": mask,
                    "box_xyxy_px": [16.0, 16.0, 47.0, 47.0],
                }
            }
        }
        pvs_state = {"instances": {}, "pending_boxes": [], "pending_bbox_records": []}

        result = np.asarray(
            self.overlay(
                self.image_state,
                pcs_state,
                pvs_state,
                "PCS Auto",
                show_instances=True,
                show_interaction_prompts=False,
            )
        )
        center = result[36, 36]
        green_contrast = int(center[1]) - max(int(center[0]), int(center[2]))
        self.assertGreaterEqual(green_contrast, 40)
        np.testing.assert_array_equal(result[4, 4], self.base_rgb[4, 4])


if __name__ == "__main__":
    unittest.main()
