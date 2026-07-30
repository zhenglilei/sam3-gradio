from __future__ import annotations

import inspect
import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sam3_gradio_demo as demo


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "refactor_gradio_config_snapshot.json"

_COMPONENT_PROP_KEYS = (
    "label",
    "value",
    "visible",
    "interactive",
    "elem_id",
    "id",
    "choices",
    "minimum",
    "maximum",
    "step",
    "lines",
    "height",
    "show_label",
    "sources",
)

_DEPENDENCY_KEYS = (
    "api_name",
    "backend_fn",
    "queue",
    "batch",
    "max_batch_size",
    "cancels",
    "trigger_after",
    "trigger_only_on_success",
    "scroll_to_output",
    "show_progress",
    "concurrency_limit",
    "concurrency_id",
)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _canonical_layout(node, component_index):
    if not isinstance(node, dict):
        return node
    result = {}
    if "id" in node:
        result["component"] = component_index.get(node["id"], "root")
    if "children" in node:
        result["children"] = [
            _canonical_layout(child, component_index)
            for child in node.get("children") or []
        ]
    return result


def canonical_config(config):
    components = config["components"]
    component_index = {
        component["id"]: index for index, component in enumerate(components)
    }
    canonical_components = []
    for component in components:
        props = component.get("props") or {}
        canonical_components.append(
            {
                "type": component.get("type"),
                "props": {
                    key: _json_value(props[key])
                    for key in _COMPONENT_PROP_KEYS
                    if key in props
                },
            }
        )

    canonical_dependencies = []
    for dependency in config["dependencies"]:
        item = {
            "targets": [
                [None if target[0] is None else component_index[target[0]], target[1]]
                for target in dependency.get("targets") or []
            ],
            "inputs": [component_index[value] for value in dependency["inputs"]],
            "outputs": [component_index[value] for value in dependency["outputs"]],
        }
        for key in _DEPENDENCY_KEYS:
            if key in dependency:
                item[key] = _json_value(dependency[key])
        canonical_dependencies.append(item)

    return {
        "component_count": len(components),
        "dependency_count": len(config["dependencies"]),
        "components": canonical_components,
        "layout": _canonical_layout(config["layout"], component_index),
        "dependencies": canonical_dependencies,
    }


def _component_semantic_name(component):
    props = component.get("props") or {}
    return props.get("elem_id") or props.get("label") or component.get("type")


class RefactorCompatibilityContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = demo.create_demo()
        cls.config = cls.app.config
        cls.components = cls.config["components"]
        cls.component_by_id = {
            component["id"]: component for component in cls.components
        }
        cls.dependencies = {
            dependency.get("api_name"): dependency
            for dependency in cls.config["dependencies"]
            if dependency.get("api_name")
        }

    def test_gradio_config_semantic_snapshot(self):
        actual = canonical_config(self.config)
        if os.environ.get("UPDATE_REFACTOR_CONFIG_SNAPSHOT") == "1":
            FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE_PATH.write_text(
                json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        self.assertTrue(FIXTURE_PATH.is_file(), "missing refactor config snapshot")
        expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)

    def test_common_output_order_is_frozen(self):
        dependency = self.dependencies["_run_pcs"]
        output_names = [
            _component_semantic_name(self.component_by_id[component_id])
            for component_id in dependency["outputs"][-8:]
        ]
        self.assertEqual(
            output_names,
            [
                "input_image",
                "分割结果",
                "分析报告",
                "PCS 实例",
                "PVS 实例",
                "当前 PVS 实例",
                "interaction-info",
                "markdown",
            ],
        )

    def test_workspace_initialization_output_contract(self):
        result = demo._init_workspace_with_layout_editor(
            None,
            demo.MODE_PVS,
            {"session_id": "contract-session"},
            demo._new_layout_state("contract-session"),
        )
        self.assertEqual(len(result), 16)
        self.assertEqual(
            [type(result[index]).__name__ for index in range(4)],
            ["dict", "dict", "dict", "dict"],
        )
        self.assertIsNone(result[14])
        self.assertIsInstance(result[15], dict)

        upload_dependency = next(
            dependency
            for dependency in self.config["dependencies"]
            if dependency.get("api_name") == "_source_upload_workspace"
            and dependency.get("targets")
            and dependency["targets"][0][1] == "upload"
        )
        self.assertEqual(len(upload_dependency["outputs"]), 19)
        workspace_outputs = upload_dependency["outputs"][3:]
        workspace_names = [
            _component_semantic_name(self.component_by_id[component_id])
            for component_id in workspace_outputs
        ]
        self.assertEqual(
            workspace_names,
            [
                "state",
                "state",
                "state",
                "state",
                "PCS bbox 列表",
                "PVS 待生成 bbox 列表",
                "input_image",
                "分割结果",
                "分析报告",
                "PCS 实例",
                "PVS 实例",
                "当前 PVS 实例",
                "interaction-info",
                "markdown",
                "下载结果包（PNG + masks + JSON）",
                "layout_transform_editor",
            ],
        )

    def test_mode_switch_contract(self):
        result = demo._switch_mode_with_layout_editor(
            demo.MODE_LAYOUT,
            {"image_id": None, "width": 0, "height": 0},
            demo._new_pcs_state(),
            demo._new_pvs_state(),
            demo._new_layout_state("contract-session"),
        )
        self.assertEqual(len(result), 28)
        dependency = self.dependencies["_switch_mode_with_layout_editor"]
        self.assertEqual((len(dependency["inputs"]), len(dependency["outputs"])), (5, 28))

    def test_state_schema_contract(self):
        self.assertEqual(
            set(demo._new_source_image_state()),
            {
                "session_id", "source_image_id", "source_image_sha256",
                "source_width", "source_height", "source_revision",
                "crop_bbox_xyxy", "pending_crop_bbox_xyxy",
                "workspace_image_id", "workspace_hash",
            },
        )
        self.assertEqual(
            set(demo._new_pcs_state()),
            {
                "text_prompt", "positive_boxes", "negative_boxes",
                "bbox_history", "bbox_records", "next_bbox_id",
                "instances", "next_instance_id",
            },
        )
        self.assertEqual(
            set(demo._new_pvs_state()),
            {
                "instances", "active_instance_id", "next_instance_id",
                "pending_boxes", "pending_bbox_records",
                "next_pending_bbox_id",
            },
        )
        self.assertEqual(
            set(demo._new_template_match_state()),
            {
                "schema_version", "source_image_id", "workspace_image_id",
                "active_instance_id", "result",
            },
        )
        self.assertEqual(set(demo._new_session_state()), {"session_id"})
        self.assertEqual(
            set(demo._new_prompt_state()),
            {"bbox_start", "last_bbox", "last_point", "polygon_points", "bbox_role"},
        )
        self.assertEqual(
            set(demo._new_layout_region_state()),
            {
                "session_id", "layout_id", "source_mask_hash",
                "regions_revision", "next_region_id", "selected_region_id",
            },
        )
        self.assertEqual(
            set(demo._new_layout_state("contract-session")),
            {
                "transform_version", "session_id", "layout_id", "image_id",
                "enabled", "region_mode", "revision", "center_x", "center_y",
                "pivot_x", "pivot_y", "tx", "ty", "scale", "rotation_deg",
                "preview_alpha", "source_width", "source_height",
                "source_mask_pixel_sha256", "source_mask_file_sha256",
                "target_image_sha256", "matrix_2x3", "prompt_mask_scope",
                "prompt_class_label", "prompt_labels", "prompt_group_transforms",
                "prompt_active_group_id", "prompt_selection_signature",
                "prompt_transform_set_revision", "prompt_regions_revision",
                "prompt_region_ids",
            },
        )
        image_state = demo._init_workspace_with_layout_editor(
            None,
            demo.MODE_PVS,
            {"session_id": "contract-session"},
            demo._new_layout_state("contract-session"),
        )[0]
        self.assertEqual(
            set(image_state),
            {
                "image_id", "width", "height", "session_id",
                "target_image_sha256", "interaction_revision",
            },
        )

    def test_callback_signature_contract(self):
        expected = {
            "_run_pcs": 6,
            "_pvs_point_prompt": 7,
            "_layout_point_refine": 8,
            "_create_pvs_from_layout_selection": 14,
            "_source_upload_workspace": 4,
            "_apply_source_crop": 4,
            "_run_template_matching": 7,
        }
        for name, parameter_count in expected.items():
            with self.subTest(name=name):
                function = getattr(demo, name)
                self.assertEqual(len(inspect.signature(function).parameters), parameter_count)

    def test_custom_component_event_contract(self):
        component_type = {
            component["id"]: component.get("type") for component in self.components
        }
        events_by_type = {}
        for dependency in self.config["dependencies"]:
            for component_id, event_name in dependency.get("targets") or []:
                events_by_type.setdefault(component_type.get(component_id), set()).add(
                    event_name
                )
        self.assertIn("input", events_by_type["imagegestureoverlay"])
        self.assertIn("input", events_by_type["layoutregionannotator"])
        self.assertIn("change", events_by_type["layouttransformeditor"])

    def test_dependency_concurrency_and_public_api_contract(self):
        for name in (
            "_run_layout_mask_page",
            "_clear_current_layout_mask",
            "_run_pcs",
            "_create_pvs_from_layout_selection",
            "_run_template_matching",
        ):
            self.assertIn(name, self.dependencies)
        stateful = [
            block_fn
            for block_fn in self.app.fns.values()
            if block_fn.concurrency_id == "image-prepost-state"
        ]
        self.assertTrue(stateful)
        self.assertTrue(
            all(block_fn.concurrency_limit in (1, "1") for block_fn in stateful)
        )


if __name__ == "__main__":
    unittest.main()
