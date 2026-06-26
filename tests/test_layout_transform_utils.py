import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import layout_transform_utils as tx


def _l_mask():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:25, 6:10] = 1
    mask[21:25, 6:24] = 1
    return mask


def _transform(rotation_deg=0.0):
    return tx.make_layout_transform_v2(
        session_id="s1",
        layout_id="layout1",
        image_id="image1",
        target_size=(300, 300),
        source_mask=_l_mask(),
        center_x=100.0,
        center_y=200.0,
        pivot_xy=(10.0, 20.0),
        scale=2.0,
        rotation_deg=rotation_deg,
        preview_alpha=0.35,
        revision=3,
        source_mask_pixel_sha256="maskhash",
        target_image_sha256="imagehash",
    )


class LayoutTransformUtilsTest(unittest.TestCase):
    def test_affine_maps_pivot_to_center_and_inverse(self):
        transform = _transform(rotation_deg=0.0)
        matrix = tx.build_layout_affine_matrix(transform)

        self.assertTrue(np.allclose(tx.apply_affine_to_point((10.0, 20.0), matrix), (100.0, 200.0)))
        self.assertTrue(np.allclose(tx.inverse_transform_point((100.0, 200.0), transform), (10.0, 20.0)))

    def test_clockwise_positive_rotation_in_y_down_coordinates(self):
        plus_90 = _transform(rotation_deg=90.0)
        minus_90 = _transform(rotation_deg=-90.0)

        self.assertTrue(np.allclose(tx.apply_affine_to_point((11.0, 20.0), tx.build_layout_affine_matrix(plus_90)), (100.0, 202.0)))
        self.assertTrue(np.allclose(tx.apply_affine_to_point((11.0, 20.0), tx.build_layout_affine_matrix(minus_90)), (100.0, 198.0)))

    def test_warp_layout_mask_is_deterministic(self):
        transform = _transform(rotation_deg=90.0)
        matrix = tx.build_layout_affine_matrix(transform)
        first = tx.warp_layout_mask(_l_mask(), matrix, (300, 300))
        second = tx.warp_layout_mask(_l_mask(), matrix, (300, 300))

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(int(first.sum()), int(second.sum()))

    def test_pixel_hash_ignores_png_encoding_but_file_hash_does_not(self):
        mask = (_l_mask() * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            first = f"{tmp}/first.png"
            second = f"{tmp}/second.png"
            Image.fromarray(mask).save(first, compress_level=0)
            Image.fromarray(mask).save(second, compress_level=9)

            self.assertEqual(tx.mask_pixel_sha256(mask), tx.mask_pixel_sha256(np.asarray(Image.open(second))))
            self.assertNotEqual(tx.file_sha256(first), tx.file_sha256(second))

    def test_migrate_v1_transform_preserves_old_pivot_target(self):
        mask = _l_mask()
        old_state = {"tx": 7.0, "ty": -3.0, "scale": 1.4, "rotation_deg": 30.0, "preview_alpha": 0.5}
        migrated = tx.migrate_layout_transform_v1_to_v2(
            old_state,
            mask,
            target_size=(300, 200),
            session_id="s1",
            layout_id="layout1",
            image_id="image1",
            target_image_sha256="imagehash",
        )

        bbox = tx.foreground_bbox_xyxy(mask)
        pivot = tx.pivot_from_bbox_xyxy(bbox)
        old_matrix = tx.old_cv2_layout_matrix(old_state, mask.shape, (300, 200))
        self.assertTrue(
            np.allclose(
                tx.apply_affine_to_point(pivot, tx.build_layout_affine_matrix(migrated)),
                tx.apply_affine_to_point(pivot, old_matrix),
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
