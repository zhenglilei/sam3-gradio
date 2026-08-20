import unittest

from sam3_demo.pcs_pvs_callbacks import _clear_pcs_instances_impl


class ClearPcsInstancesTest(unittest.TestCase):
    def setUp(self):
        self.image_state = {"image_id": "image-1"}
        self.pvs_state = {"instances": {7: {"id": 7}}}
        self.view_calls = []
        self.deps = {
            "_is_pcs_mode": lambda mode: mode == "PCS Auto",
            "_reset_pcs_predictions": self._reset_predictions,
            "_view": self._view,
        }

    @staticmethod
    def _reset_predictions(state):
        state["instances"] = {}
        state["next_instance_id"] = 1

    def _view(self, image_state, pcs_state, pvs_state, mode, info):
        self.view_calls.append((image_state, pcs_state, pvs_state, mode, info))
        return ("view",)

    def test_clears_instances_and_preserves_prompts(self):
        pcs_state = {
            "text_prompt": "metal",
            "positive_boxes": [[1, 2, 3, 4]],
            "negative_boxes": [[5, 6, 7, 8]],
            "instances": {1: {"id": 1}, 2: {"id": 2}},
            "next_instance_id": 3,
        }

        result = _clear_pcs_instances_impl(
            self.deps,
            self.image_state,
            pcs_state,
            self.pvs_state,
            "PCS Auto",
        )

        self.assertIs(result[0], pcs_state)
        self.assertEqual(pcs_state["instances"], {})
        self.assertEqual(pcs_state["next_instance_id"], 1)
        self.assertEqual(pcs_state["text_prompt"], "metal")
        self.assertEqual(pcs_state["positive_boxes"], [[1, 2, 3, 4]])
        self.assertEqual(pcs_state["negative_boxes"], [[5, 6, 7, 8]])
        self.assertEqual(self.pvs_state["instances"], {7: {"id": 7}})
        self.assertIn("\u5df2\u6e05\u7a7a 2 \u4e2a PCS \u5b9e\u4f8b", self.view_calls[-1][-1])

    def test_rejects_non_pcs_mode_without_changing_state(self):
        pcs_state = {"instances": {1: {"id": 1}}, "next_instance_id": 2}

        _clear_pcs_instances_impl(
            self.deps,
            self.image_state,
            pcs_state,
            self.pvs_state,
            "PVS Manual",
        )

        self.assertEqual(pcs_state["instances"], {1: {"id": 1}})
        self.assertEqual(pcs_state["next_instance_id"], 2)
        self.assertIn("\u4ec5 PCS Auto \u6a21\u5f0f", self.view_calls[-1][-1])


if __name__ == "__main__":
    unittest.main()
