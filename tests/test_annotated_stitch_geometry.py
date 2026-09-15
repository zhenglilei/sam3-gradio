import unittest

import numpy as np
from PIL import Image

from sam3_demo.annotated_stitch_geometry import compose_annotated_mosaic, crop_annotations
from sam3_demo.stitch_workflow import export_mosaic


def image(value, size=(6, 6)):
    return Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8))


def annotation(instance_id, mask):
    return {"id": instance_id, "category_name": "defect", "mask": np.asarray(mask, dtype=bool), "provenance": {"source": "test"}}


class AnnotatedStitchGeometryTests(unittest.TestCase):
    def test_zero_rotation_matches_image_export_for_every_layout(self):
        cases = (
            ("horizontal", [image(1), image(2)], [(0, 0), (4, 0)]),
            ("vertical", [image(1), image(2)], [(0, 0), (0, 4)]),
            ("grid_2x2", [image(i) for i in range(4)], [(0, 0), (4, 0), (0, 4), (4, 4)]),
            ("grid_2xn", [image(i) for i in range(4)], [(0, 0), (4, 0), (0, 4), (4, 4)]),
        )
        for layout, images, shifts in cases:
            annotations = [[annotation(i, np.ones((6, 6), dtype=bool))] for i in range(len(images))]
            mosaic, instances, manifest = compose_annotated_mosaic(images, annotations, shifts, layout=layout, blend=False, rotations=[0] * len(images))
            expected, _ = export_mosaic(images, shifts, layout=layout, blend=False, rotations=[0] * len(images))
            self.assertTrue(np.array_equal(np.asarray(mosaic), np.asarray(expected)), layout)
            self.assertEqual(mosaic.size, tuple(manifest["canvas_size"]))
            self.assertTrue(all(item["mask"].shape == (mosaic.height, mosaic.width) for item in instances))

    def test_negative_coordinates_and_source_provenance(self):
        images = [image(1), image(2)]
        annotations = [[annotation("a", np.ones((6, 6)))], [annotation("b", np.ones((6, 6)))]]
        mosaic, instances, manifest = compose_annotated_mosaic(images, annotations, [(-3, -2), (1, -2)], blend=True)
        self.assertEqual(mosaic.size, (10, 6))
        self.assertEqual(manifest["tiles"][0]["origin_xy"], [0, 0])
        self.assertEqual(instances[0]["provenance"]["source_tile_index"], 0)
        self.assertEqual(instances[1]["source_instance_id"], "b")

    def test_zero_rotation_blend_matches_image_export_for_every_layout(self):
        cases = (
            ("horizontal", [image(1), image(2)], [(0, 0), (4, 0)]),
            ("vertical", [image(1), image(2)], [(0, 0), (0, 4)]),
            ("grid_2x2", [image(i) for i in range(4)], [(0, 0), (4, 0), (0, 4), (4, 4)]),
            ("grid_2xn", [image(i) for i in range(4)], [(0, 0), (4, 0), (0, 4), (4, 4)]),
        )
        for layout, images, shifts in cases:
            annotations = [[annotation(i, np.ones((6, 6), dtype=bool))] for i in range(len(images))]
            mosaic, _instances, _manifest = compose_annotated_mosaic(images, annotations, shifts, layout=layout, blend=True, rotations=[0] * len(images))
            expected, _ = export_mosaic(images, shifts, layout=layout, blend=True, rotations=[0] * len(images))
            self.assertTrue(np.array_equal(np.asarray(mosaic), np.asarray(expected)), layout)

    def test_nearest_rotation_preserves_center_hole_at_90_and_arbitrary_angles(self):
        mask = np.ones((7, 7), dtype=bool)
        mask[2:5, 2:5] = False
        for angle in (90, 31.0):
            mosaic, instances, _ = compose_annotated_mosaic([image(1, (7, 7))], [[annotation("hole", mask)]], [(0, 0)], rotations=[angle])
            output = instances[0]["mask"]
            self.assertFalse(output[mosaic.height // 2, mosaic.width // 2], angle)
            self.assertLess(output.sum(), 49, angle)

    def test_blend_keeps_overlap_but_nonblend_assigns_ownership(self):
        full = np.ones((6, 6), dtype=bool)
        _mosaic, instances, _manifest = compose_annotated_mosaic([image(1), image(2)], [[annotation("left", full)], [annotation("right", full)]], [(0, 0), (2, 0)], blend=True)
        self.assertEqual(len(instances), 2)
        self.assertTrue((instances[0]["mask"] & instances[1]["mask"]).any())
        _mosaic, instances, _manifest = compose_annotated_mosaic([image(1), image(2)], [[annotation("left", full)], [annotation("right", full)]], [(0, 0), (2, 0)], blend=False)
        self.assertFalse((instances[0]["mask"] & instances[1]["mask"]).any())
        _mosaic, instances, manifest = compose_annotated_mosaic([image(1), image(2)], [[annotation("first", full)], [annotation("second", full)]], [(0, 0), (0, 0)], blend=False, rotations=[18, 18])
        self.assertEqual([item["source_instance_id"] for item in instances], ["second"])
        self.assertEqual(manifest["dropped_instances"][0]["reason"], "covered_by_later_tile")

    def test_periodic_crop_and_completely_empty_crop(self):
        x = np.arange(64, dtype=np.uint8)
        row = ((x // 4) % 2 * 255).astype(np.uint8)
        tiled = np.tile(row, (64, 1))
        source = Image.fromarray(np.stack([tiled, tiled, tiled], axis=-1), mode="RGB")
        mosaic, instances, manifest = compose_annotated_mosaic([source], [[annotation("periodic", np.ones((64, 64)))]], [(0, 0)], crop_periodic=True)
        self.assertEqual(instances[0]["mask"].shape, (mosaic.height, mosaic.width))
        self.assertEqual(mosaic.size, tuple(np.subtract(manifest["crop_bbox_xyxy"][2:], manifest["crop_bbox_xyxy"][:2])))
        kept, dropped = crop_annotations([annotation("empty", np.ones((3, 3)))], (0, 0, 0, 0))
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0]["reason"], "crop_empty")


if __name__ == "__main__":
    unittest.main()
