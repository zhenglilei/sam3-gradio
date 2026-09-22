import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from sam3_demo import app, batch_workspace as batch
from sam3_demo import model_runtime


class BatchParallelPcsTests(unittest.TestCase):
    def setUp(self):
        # Keep serialization and restore real; durable flushes have dedicated tests.
        self.enterContext(mock.patch("os.fsync"))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runtime_patch = mock.patch.object(app, "runtime_dir", Path(self.tmp.name))
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.session = {"session_id": "a" * 32}
        self.state = batch.owned_batch({}, self.session)
        self.paths = []
        for index in range(6):
            path = Path(self.tmp.name) / f"tile{index}.png"
            Image.new("RGB", (24, 20), (index * 20, 30, 50)).save(path)
            self.paths.append(str(path))
        batch.add_images(self.state, self.paths)

    def _save_inputs(self):
        boxes = []
        for index, item in enumerate(self.state["items"]):
            values = list(batch.restore_item(app, self.state, item, self.session))
            if index == 1:
                values[0]["pending_crop_bbox_xyxy"] = [2, 3, 20, 18]
                cropped = app._apply_source_crop(
                    values[0], values[6], self.session, values[5]
                )
                values[:5] = [cropped[0], cropped[3], cropped[4], cropped[5], cropped[6]]
            width, height = app._workspace(values[1])["image"].size
            box = [index + 1, 2, min(width - 1, index + 9), min(height - 1, 12)]
            boxes.append(box)
            values[2]["positive_boxes"] = [box]
            values[6] = "PCS Auto"
            batch.save_snapshot(app, self.state, *values)
        return boxes

    @staticmethod
    def _prediction(payload):
        width, height = payload["image"].size
        mask = np.ones((1, height, width), dtype=bool)
        return {
            "masks": mask,
            "scores": np.asarray([0.9], dtype=np.float32),
            "boxes": np.asarray([[1, 1, width - 1, height - 1]], dtype=np.float32),
            "probs": None,
        }

    def _start(self, text=""):
        ids = [item["id"] for item in self.state["items"]]
        self.state.update(running=True, pending=ids[:], batch_text=text, batch_threshold=0.4)
        return ids

    def test_step_processes_four_then_two_in_order_with_each_image_prompts_and_crop(self):
        boxes = self._save_inputs()
        ids = self._start()
        calls = []

        def predict(payloads):
            calls.append(payloads)
            return {
                "items": [{"prediction": self._prediction(item), "error": None}
                          for item in payloads],
                "concurrency": min(4, len(payloads)), "fallback": False, "reason": "",
            }

        with mock.patch.object(model_runtime, "_predict_pcs_batch", side_effect=predict,
                               create=True):
            first = batch.process_pcs_chunk(app, self.state, self.session)
            self.assertEqual(self.state["pending"], ids[4:])
            second = batch.process_pcs_chunk(app, self.state, self.session)

        self.assertEqual([len(call) for call in calls], [4, 2])
        self.assertEqual([first["last_id"], second["last_id"]], [ids[3], ids[5]])
        self.assertEqual(self.state["active_id"], ids[5])
        self.assertEqual([item["status"] for item in self.state["items"]], ["done"] * 6)
        for index, payload in enumerate(calls[0] + calls[1]):
            image_size = payload["image"].size
            self.assertEqual(payload["positive_boxes_cxcywh"], [
                app._xyxy_to_cxcywh_norm(boxes[index], *image_size)
            ])
            self.assertEqual(payload["negative_boxes_cxcywh"], [])
            self.assertEqual(payload["threshold"], 0.4)
            self.assertEqual(payload["text"], "")
        self.assertEqual(calls[0][1]["image"].size, (18, 15))

    def test_negative_boxes_stay_with_their_image_across_chunks(self):
        boxes = []
        for index, item in enumerate(self.state["items"]):
            values = list(batch.restore_item(app, self.state, item, self.session))
            box = [index + 1, 9, index + 9, 19]
            boxes.append(box)
            values[2]["positive_boxes"] = []
            values[2]["negative_boxes"] = [box]
            values[6] = "PCS Auto"
            batch.save_snapshot(app, self.state, *values)
        self._start(text="shared prompt")
        seen = []

        def predict(payloads):
            seen.extend(payload["negative_boxes_cxcywh"] for payload in payloads)
            return {
                "items": [{"prediction": self._prediction(item), "error": None}
                          for item in payloads],
                "concurrency": min(4, len(payloads)), "fallback": False, "reason": "",
            }

        with mock.patch.object(model_runtime, "_predict_pcs_batch", side_effect=predict):
            batch.process_pcs_chunk(app, self.state, self.session)
            batch.process_pcs_chunk(app, self.state, self.session)

        self.assertEqual(seen, [[app._xyxy_to_cxcywh_norm(box, 24, 20)]
                                for box in boxes])
        self.assertEqual([item["status"] for item in self.state["items"]], ["done"] * 6)

    def test_missing_prompt_and_backend_error_fail_independently_and_keep_old_pcs(self):
        boxes = self._save_inputs()
        ids = self._start()
        originals = []
        for index in (0, 1):
            item = self.state["items"][index]
            values = list(batch.restore_item(app, self.state, item, self.session))
            mask = np.zeros((20, 24), dtype=bool)
            mask[3:10, 4:12] = True
            values[2]["instances"] = {
                7: app._make_inst(7, "pcs", mask, [4, 3, 12, 10], 0.7, history=[])
            }
            values[2]["next_instance_id"] = 8
            if index == 0:
                values[2]["positive_boxes"] = []
            batch.save_snapshot(app, self.state, *values)
            originals.append(deepcopy(item["snapshot"]["pcs"]["instances"][7]["mask_fullres_bool"]))
        calls = []

        def predict(payloads):
            calls.append(payloads)
            results = [{"prediction": None, "error": "runtime unavailable"}]
            results.extend({"prediction": self._prediction(item), "error": None}
                           for item in payloads[1:])
            return {"items": results, "concurrency": 2, "fallback": True,
                    "reason": "worker unavailable"}

        with mock.patch.object(model_runtime, "_predict_pcs_batch", side_effect=predict,
                               create=True):
            batch.process_pcs_chunk(app, self.state, self.session)

        self.assertEqual(len(calls[0]), 3)
        self.assertEqual([item["status"] for item in self.state["items"][:4]],
                         ["failed", "failed", "done", "done"])
        self.assertIn("正样本框", self.state["items"][0]["error"])
        self.assertIn("runtime unavailable", self.state["items"][1]["error"])
        for index in (0, 1):
            restored = self.state["items"][index]["snapshot"]["pcs"]["instances"][7]
            np.testing.assert_array_equal(restored["mask_fullres_bool"], originals[index])
        self.assertEqual(self.state["items"][2]["snapshot"]["pcs"]["instances"][1]["id"], 1)
        self.assertEqual(self.state["pending"], ids[4:])

    def test_owner_and_cancel_are_checked_at_chunk_boundaries(self):
        self._save_inputs()
        ids = self._start()
        predict = mock.Mock(side_effect=lambda payloads: {
            "items": [{"prediction": self._prediction(item), "error": None}
                      for item in payloads],
            "concurrency": 4, "fallback": False, "reason": "",
        })
        with mock.patch.object(model_runtime, "_predict_pcs_batch", predict, create=True):
            before = list(self.state["pending"])
            batch.process_pcs_chunk(app, self.state, {"session_id": "b" * 32})
            self.assertEqual(self.state["pending"], before)
            self.assertEqual(predict.call_count, 0)

            batch.process_pcs_chunk(app, self.state, self.session)
            self.assertEqual(self.state["pending"], ids[4:])
            self.state.update(running=False, pending=[])
            batch.process_pcs_chunk(app, self.state, self.session)

        self.assertEqual(predict.call_count, 1)
        self.assertEqual(self.state["active_id"], ids[3])


if __name__ == "__main__":
    unittest.main()
