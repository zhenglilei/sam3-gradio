"""Rendered configuration contract for the multi-image workspace."""
import unittest
from sam3_demo import app

class BatchUIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = app.create_demo()
        cls.config = cls.demo.get_config_file()

    def test_batch_controls_are_present(self):
        labels = {c.get("props", {}).get("value") for c in self.config["components"]
                  if c["type"] == "button"}
        for value in ("保存并下一张", "批量 PCS", "送往拼接"):
            self.assertIn(value, labels)

    def test_canvas_anchor_ids_remain_unique(self):
        ids = [c.get("props", {}).get("elem_id") for c in self.config["components"]]
        for name in ("source_input_image", "input_image", "el-workspace"):
            self.assertEqual(ids.count(name), 1)

    def _ancestors(self, target):
        def find(node, path):
            if node.get("id") == target:
                return path
            for child in node.get("children", []):
                found = find(child, [*path, node.get("id")])
                if found is not None:
                    return found
        ids = find(self.config["layout"], [])
        by_id = {c["id"]: c for c in self.config["components"]}
        return [by_id[i] for i in ids if i in by_id]

    def test_business_tabs_are_explicit(self):
        labels = {c["props"].get("label") for c in self.config["components"] if c["type"] == "tabitem"}
        self.assertTrue({"分割标注", "裁剪原图", "模板匹配", "导出与记录"} <= labels)

    def test_primary_pcs_and_template_controls_are_not_accordion_children(self):
        targets = ("文本提示（可选）", "用于模板匹配的 PVS 实例（可多选）")
        for label in targets:
            component = next(c for c in self.config["components"] if c.get("props", {}).get("label") == label)
            parents = self._ancestors(component["id"])
            self.assertNotIn("accordion", [c["type"] for c in parents])
        template = next(c for c in self.config["components"] if c.get("props", {}).get("elem_id") == "template_match_preview")
        tabs = [c["props"].get("label") for c in self._ancestors(template["id"]) if c["type"] == "tabitem"]
        self.assertIn("模板匹配", tabs)

    def test_gallery_uses_gradio6_integer_columns(self):
        gallery = next(c for c in self.config["components"] if c.get("props", {}).get("elem_id") == "el-image-gallery")
        self.assertIsInstance(gallery["props"]["columns"], int)

    def test_canvas_view_controls_are_interactive_without_backend_events(self):
        for anchor in ("el-canvas-view", "repair-canvas-view"):
            component = next(c for c in self.config["components"] if c["props"].get("elem_id") == anchor)
            self.assertTrue(component["props"]["interactive"])
            self.assertEqual(component["props"]["value"], "auto")
            self.assertEqual([choice[1] for choice in component["props"]["choices"]],
                             ["auto", "source", "result", "compare"])

    def test_repair_actions_are_above_canvas_and_disabled_until_ready(self):
        for label in ("修复当前图片", "自动标记并批量修复", "送往智能图像分割"):
            button = next(c for c in self.config["components"] if c["type"] == "button" and c["props"].get("value") == label)
            parents = {c["props"].get("elem_id") for c in self._ancestors(button["id"])}
            self.assertIn("repair-actions", parents)
            self.assertNotIn("repair-tools", parents)
            self.assertFalse(button["props"]["interactive"])

    def test_programmatic_pvs_selection_does_not_trigger_user_action(self):
        event = next(d for d in self.config["dependencies"] if d.get("api_name") == "_set_active_pvs")
        self.assertEqual([target[1] for target in event["targets"]], ["input"])

    def test_stitch_editor_and_result_have_explicit_tabs(self):
        for anchor, label in (("stitch_preview_canvas", "拼接编辑"), ("stitch_mosaic_preview", "结果预览")):
            component = next(c for c in self.config["components"] if c["props"].get("elem_id") == anchor)
            parents = self._ancestors(component["id"])
            self.assertIn(label, [c["props"].get("label") for c in parents if c["type"] == "tabitem"])

if __name__ == "__main__":
    unittest.main()
