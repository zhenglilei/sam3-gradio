"""Feedback persistence and segmentation export callbacks."""

from __future__ import annotations

import json
import time
import uuid

import cv2
import numpy as np

import layout_region_utils as _layout_regions
import layout_transform_utils as _layout_tx


def _history_json_impl(_deps, history):
    rows = []
    for item in history:
        row = {"op": item.get("op"), "prompt": item.get("prompt"), "box_xyxy_px": item.get("box_xyxy_px"), "candidate_scores": item.get("candidate_scores")}
        if item.get("before"):
            row["before"] = {"box_xyxy_px": item["before"].get("box_xyxy_px"), "score": item["before"].get("score"), "status": item["before"].get("status")}
        if item.get("after"):
            row["after"] = {"box_xyxy_px": item["after"].get("box_xyxy_px"), "score": item["after"].get("score"), "status": item["after"].get("status")}
        rows.append({k: v for k, v in row.items() if v is not None})
    return rows


def _latest_layout_prompt_from_instances_impl(_deps, instances):
    for item in reversed(list(instances or [])):
        for hist in reversed(item.get("prompt_history", []) or []):
            prompt = hist.get("prompt") or {}
            if hist.get("op") in {"create_from_layout_mask", "refine_with_layout_mask"} or prompt.get("type") == "layout_mask":
                return prompt
    return None


def _reconstruct_frozen_layout_prompt_mask_impl(_deps, layout_prompt):
    _LAYOUT_PROMPT_SCOPE_FULL = _deps['_LAYOUT_PROMPT_SCOPE_FULL']
    _LAYOUT_REGION_STORE = _deps['_LAYOUT_REGION_STORE']
    if not isinstance(layout_prompt, dict):
        raise ValueError("layout prompt metadata 无效")
    session_id = layout_prompt.get("session_id")
    layout_id = layout_prompt.get("layout_id")
    source_mask_hash = layout_prompt.get("source_mask_pixel_sha256")
    target_width = int(layout_prompt.get("target_width") or 0)
    target_height = int(layout_prompt.get("target_height") or 0)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("layout prompt 目标尺寸无效")
    matrix = np.asarray(layout_prompt.get("matrix_2x3"), dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("layout prompt 缺少有效的 frozen affine matrix")

    scope = layout_prompt.get("mask_scope")
    details = {"reconstruction_status": "ok"}
    if not scope or scope == _LAYOUT_PROMPT_SCOPE_FULL:
        source_mask, _ = _LAYOUT_REGION_STORE.load_source_mask(
            session_id,
            layout_id,
            source_mask_hash,
        )
        prompt_mask = source_mask
        details["reconstruction_method"] = "frozen_full_mask"
    elif scope == "region":
        document, source_mask = _LAYOUT_REGION_STORE.load_document(
            session_id,
            layout_id,
            source_mask_hash,
        )
        details["current_regions_revision"] = int(
            document.get("regions_revision") or 0
        )
        try:
            region_id = int(layout_prompt.get("region_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("layout Region prompt 缺少有效 region_id") from exc
        record = next(
            (
                item
                for item in document.get("regions", [])
                if int(item.get("region_id") or 0) == region_id
            ),
            None,
        )
        if record is None:
            raise ValueError(f"layout Region R{region_id} 不存在")
        prompt_mask = _layout_regions.decode_binary_mask(
            record.get("mask_rle"),
            source_mask.shape,
        )
        actual_hash = _layout_regions.mask_pixel_sha256(
            prompt_mask.astype(np.uint8)
        )
        expected_hash = layout_prompt.get("region_mask_pixel_sha256")
        if not expected_hash or actual_hash != str(expected_hash):
            raise ValueError(f"layout Region R{region_id} mask hash 不匹配")
        prompt_label = layout_prompt.get("label")
        if prompt_label is not None:
            if _layout_regions.region_label(record) != str(prompt_label):
                raise ValueError(f"layout Region R{region_id} Label 与实例记录不一致")
        else:
            prompt_class = layout_prompt.get("class_label")
            if (
                prompt_class is not None
                and record.get("class_label") != prompt_class
            ):
                raise ValueError(f"layout Region R{region_id} 历史类别与实例记录不一致")
        details.update(
            {
                "reconstruction_method": "frozen_region_rle",
                "region_id": region_id,
                "region_deleted_at": record.get("deleted_at"),
            }
        )
    else:
        raise ValueError(f"未知 layout prompt mask_scope: {scope}")

    transformed = _layout_tx.warp_layout_mask(
        prompt_mask,
        matrix,
        (target_width, target_height),
    )
    return transformed, details


def _write_feedback_layout_artifacts_impl(_deps, sample_dir, layout_prompt):
    _reconstruct_frozen_layout_prompt_mask = _deps['_reconstruct_frozen_layout_prompt_mask']
    if not layout_prompt:
        return {}
    transform_path = sample_dir / "layout_transform.json"
    layout_id = layout_prompt.get("layout_id")
    transformed_mask_path = None
    reconstruction = {}
    try:
        transformed_mask, reconstruction = (
            _reconstruct_frozen_layout_prompt_mask(layout_prompt)
        )
        transformed_mask_path = sample_dir / "layout_transformed_mask.png"
        written = cv2.imwrite(
            str(transformed_mask_path),
            np.asarray(transformed_mask, dtype=np.uint8) * 255,
        )
        if not written:
            raise OSError("cannot write reconstructed layout prompt mask")
    except Exception as exc:
        transformed_mask_path = None
        reconstruction = {
            "reconstruction_status": "unavailable",
            "reconstruction_error": str(exc),
        }

    payload = {
        "layout_prompt": layout_prompt,
        "layout_id": layout_id,
        "has_cached_transformed_mask": False,
        "has_reconstructed_transformed_mask": (
            transformed_mask_path is not None
        ),
        "layout_transformed_mask_file": (
            str(transformed_mask_path)
            if transformed_mask_path is not None
            else None
        ),
        **reconstruction,
    }
    with transform_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {
        "layout_transform_file": str(transform_path),
        "layout_transformed_mask_file": (
            str(transformed_mask_path)
            if transformed_mask_path is not None
            else None
        ),
    }


def _submit_feedback_impl(_deps, image_state, pcs_state, pvs_state, mode, rating, feedback_tags, feedback_comment):
    _FEEDBACK_WRITE_LOCK = _deps['_FEEDBACK_WRITE_LOCK']
    _active_instances = _deps['_active_instances']
    _history_json = _deps['_history_json']
    _is_pcs_mode = _deps['_is_pcs_mode']
    _is_pvs_pool_mode = _deps['_is_pvs_pool_mode']
    _latest_layout_prompt_from_instances = _deps['_latest_layout_prompt_from_instances']
    _result_image = _deps['_result_image']
    _view = _deps['_view']
    _workspace = _deps['_workspace']
    _write_feedback_layout_artifacts = _deps['_write_feedback_layout_artifacts']
    runtime_feedback_dir = _deps['runtime_feedback_dir']
    try:
        if _is_pvs_pool_mode(mode):
            active_id = pvs_state.get("active_instance_id")
            if active_id is None:
                raise ValueError("请先选择一个 active PVS instance")
            inst = pvs_state.get("instances", {}).get(int(active_id))
            if inst is None or inst.get("status") == "deleted":
                raise ValueError("当前 active PVS instance 不存在或已删除")
            feedback_instances = [inst]
            feedback_target = "active_pvs_instance"
        elif _is_pcs_mode(mode):
            feedback_instances = _active_instances(pcs_state)
            if not feedback_instances:
                raise ValueError("请先运行 PCS 并生成至少一个 PCS instance")
            inst = None
            feedback_target = "pcs_instance_pool"
        else:
            raise ValueError(f"不支持的 feedback 模式: {mode}")

        ws = _workspace(image_state)
        image = ws["image"]
        feedback_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        sample_dir = runtime_feedback_dir / "samples" / feedback_id
        sample_dir.mkdir(parents=True, exist_ok=False)

        image_path = sample_dir / "image.png"
        overlay_path = sample_dir / "overlay.png"
        mask_path = sample_dir / "mask.png"
        npz_path = sample_dir / "mask.npz"
        feedback_path = sample_dir / "feedback.json"

        image.save(image_path)
        overlay = _result_image(image_state, pcs_state, pvs_state, mode)
        if overlay is not None:
            overlay.save(overlay_path)

        masks = [np.asarray(item["mask_fullres_bool"]).astype(bool) for item in feedback_instances]
        mask_stack = np.stack([mask.astype(np.uint8) for mask in masks], axis=0)
        mask_preview = np.any(mask_stack.astype(bool), axis=0).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_preview * 255)
        pvs_logits_values = [item.get("pvs_lowres_logits") for item in feedback_instances if item.get("pvs_lowres_logits") is not None]
        pcs_prob_values = [item.get("pcs_fullres_prob") for item in feedback_instances if item.get("pcs_fullres_prob") is not None]
        np.savez_compressed(
            npz_path,
            mask_fullres_uint8=mask_stack,
            pvs_lowres_logits=np.stack([np.asarray(v, dtype=np.float32) for v in pvs_logits_values], axis=0) if pvs_logits_values else np.empty((0,), dtype=np.float32),
            pcs_fullres_prob=np.stack([np.asarray(v, dtype=np.float32) for v in pcs_prob_values], axis=0) if pcs_prob_values else np.empty((0,), dtype=np.float32),
        )
        instance_rows = [
            {
                "instance_id": int(item["id"]),
                "source": item.get("source"),
                "status": item.get("status"),
                "score": float(item.get("score", 0.0)),
                "bbox_xyxy_px": [float(v) for v in item.get("box_xyxy_px", [])],
                "prompt_history": _history_json(item.get("prompt_history", [])),
            }
            for item in feedback_instances
        ]
        layout_artifacts = _write_feedback_layout_artifacts(sample_dir, _latest_layout_prompt_from_instances(feedback_instances))

        payload = {
            "feedback_id": feedback_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "target": feedback_target,
            "rating": rating,
            "tags": feedback_tags or [],
            "comment": feedback_comment or "",
            "image_id": image_state.get("image_id"),
            "image_size": [int(image.width), int(image.height)],
            "instance_id": int(inst["id"]) if inst is not None else None,
            "source": inst.get("source") if inst is not None else "pcs",
            "status": inst.get("status") if inst is not None else None,
            "score": float(inst.get("score", 0.0)) if inst is not None else None,
            "bbox_xyxy_px": [float(v) for v in inst.get("box_xyxy_px", [])] if inst is not None else None,
            "prompt_history": _history_json(inst.get("prompt_history", [])) if inst is not None else [],
            "instance_count": len(feedback_instances),
            "instances": instance_rows,
            "image_file": str(image_path),
            "overlay_file": str(overlay_path) if overlay is not None else None,
            "mask_file": str(mask_path),
            "mask_npz_file": str(npz_path),
            "layout_transform_file": layout_artifacts.get("layout_transform_file"),
            "layout_transformed_mask_file": layout_artifacts.get("layout_transformed_mask_file"),
            "branch": "Zhengqiyuan/PVS-demo",
        }

        with feedback_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with _FEEDBACK_WRITE_LOCK:
            with (runtime_feedback_dir / "feedback.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        info = f"反馈已保存: {feedback_id}"
    except Exception as exc:
        info = f"反馈保存失败: {exc}"
    return _view(image_state, pcs_state, pvs_state, mode, info)


def _export_pool_impl(_deps, image_state, pcs_state, pvs_state, mode, pool_name, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    _active_instances = _deps['_active_instances']
    _history_json = _deps['_history_json']
    _overlay = _deps['_overlay']
    _publish_segmentation_zip = _deps['_publish_segmentation_zip']
    _workspace = _deps['_workspace']
    compare_with_coco = _deps['compare_with_coco']
    create_prediction_coco_json = _deps['create_prediction_coco_json']
    mask_to_polygons = _deps['mask_to_polygons']
    runtime_export_dir = _deps['runtime_export_dir']
    try:
        ws = _workspace(image_state)
        image = ws["image"]
        pool = pcs_state if pool_name == "pcs" else pvs_state
        instances = _active_instances(pool)
        if not instances:
            raise ValueError(f"No active {pool_name.upper()} instance")
        export_id = f"{pool_name}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        export_dir = runtime_export_dir / export_id
        mask_dir = export_dir / "masks"
        export_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(exist_ok=True)
        _overlay(image_state, pcs_state, pvs_state, mode).save(export_dir / "overlay.png")
        masks, scores, predictions = [], [], []
        coco_annotation_extras = []
        for inst in instances:
            mask = np.asarray(inst["mask_fullres_bool"]).astype(bool)
            masks.append(mask)
            scores.append(float(inst.get("score", 1.0)))
            mask_path = mask_dir / f"{pool_name}_{inst['id']:03d}.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            mask_file = str(mask_path.relative_to(export_dir))
            bbox_xyxy = [float(v) for v in inst.get("box_xyxy_px", [])]
            predictions.append({"id": int(inst["id"]), "source": inst.get("source"), "status": inst.get("status"), "score": float(inst.get("score", 0.0)), "bbox_xyxy": bbox_xyxy, "mask_file": mask_file, "final_contour_polygon": mask_to_polygons(mask), "prompt_history": _history_json(inst.get("prompt_history", []))})
            coco_annotation_extras.append({"instance_id": int(inst["id"]), "source": inst.get("source"), "status": inst.get("status"), "mask_file": mask_file, "bbox_xyxy": bbox_xyxy})
        metrics = compare_with_coco(masks, scores, coco_dataset, coco_image_name.strip() if coco_image_name else "", coco_split, pcs_state.get("text_prompt", "") if pool_name == "pcs" else "", image.width, image.height, coco_eval_scope, annotation_json_file)
        with (export_dir / "prediction.json").open("w", encoding="utf-8") as f:
            json.dump({"export_id": export_id, "pool": pool_name, "image": {"width": image.width, "height": image.height}, "predictions": predictions, "metrics": metrics}, f, ensure_ascii=False, indent=2)
        with (export_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        coco_payload = create_prediction_coco_json(
            masks,
            scores,
            image.width,
            image.height,
            image_file_name=coco_image_name.strip() if coco_image_name else "source_image",
            category_name=f"{pool_name}_object",
            export_id=export_id,
            annotation_extras=coco_annotation_extras,
        )
        with (export_dir / "coco_masks.json").open("w", encoding="utf-8") as f:
            json.dump(coco_payload, f, ensure_ascii=False, indent=2)
        zip_path = _publish_segmentation_zip(export_dir, f"{export_id}.zip")
        info = f"Exported {len(instances)} {pool_name.upper()} instances: {zip_path}"
        if metrics.get("summary_lines"):
            info += "\n" + "\n".join(metrics["summary_lines"])
        return str(zip_path), info
    except Exception as exc:
        return None, f"Export failed: {exc}"


def _export_pcs_impl(_deps, image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    _export_pool = _deps['_export_pool']
    _view = _deps['_view']
    path, info = _export_pool(image_state, pcs_state, pvs_state, mode, "pcs", coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file)
    return path, *_view(image_state, pcs_state, pvs_state, mode, info)


def _export_pvs_impl(_deps, image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file):
    _export_pool = _deps['_export_pool']
    _view = _deps['_view']
    path, info = _export_pool(image_state, pcs_state, pvs_state, mode, "pvs", coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file)
    return path, *_view(image_state, pcs_state, pvs_state, mode, info)
