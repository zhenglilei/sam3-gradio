"""Session-owned multi-image workbench, using existing segmentation callbacks."""
from copy import deepcopy
from pathlib import Path
import json
import uuid
import numpy as np
from PIL import Image
import gradio as gr
from .session_guard import guard_callback
from .session_cleanup import validate_server_session_id

MAX_IMAGES = 32
MAX_PIXELS = 32_000_000


def owned_batch(batch, session):
    sid = validate_server_session_id(session["session_id"])
    if not batch or batch.get("session_id") != sid:
        return {"session_id": sid, "items": [], "active_id": None,
                "pending": [], "running": False}
    return batch


def instance_count(item):
    snap = item.get("snapshot") or {}
    pool = snap.get("pcs") if snap.get("mode") == "PCS Auto" else snap.get("pvs")
    return sum(not x.get("deleted") for x in (pool or {}).get("instances", {}).values())


def queue_display(batch, selected):
    gallery, choices = [], []
    for index, item in enumerate(batch["items"]):
        thumb = item["original"].copy()
        thumb.thumbnail((150, 100))
        count = instance_count(item)
        status = {"pending": "待处理", "done": "已标注", "failed": "失败"}.get(item["status"], item["status"])
        short_name = item["name"] if len(item["name"]) <= 28 else item["name"][:16] + "..." + item["name"][-9:]
        title = f"{index+1}. {count} 实例 · {status} | {short_name}"
        gallery.append((thumb, title))
        choices.append((title, item["id"]))
    valid = {i["id"] for i in batch["items"]}
    return gallery, gr.update(choices=choices, value=[x for x in (selected or []) if x in valid])


def add_images(batch, files):
    items = []
    pixels = sum(i["original"].width * i["original"].height for i in batch["items"])
    for path in files or []:
        with Image.open(path) as source:
            pixels += source.width * source.height
            if pixels > MAX_PIXELS or len(batch["items"]) + len(items) >= MAX_IMAGES:
                raise ValueError(f"每批最多 {MAX_IMAGES} 张、总计 {MAX_PIXELS//1000000} 百万像素")
            image = source.convert("RGB").copy()
        items.append({"id": uuid.uuid4().hex, "name": Path(path).name,
                      "original": image, "snapshot": None, "status": "pending", "error": ""})
    batch["items"].extend(items)
    return items


def save_snapshot(app, batch, source, image, pcs, pvs, prompt, layout, mode, tool, text, threshold):
    item = next((x for x in batch["items"] if x["id"] == batch.get("active_id")), None)
    if item is None or not image or not image.get("image_id"):
        return False
    if (item.get("live_image_id") != image["image_id"]
            and item.get("live_source_id") != source.get("source_image_id")):
        return False
    item["original"] = app._source_image_cache_get(source)
    item["snapshot"] = {
        "source": deepcopy(source), "image": app._workspace(image)["image"].copy(),
        "pcs": deepcopy(pcs), "pvs": deepcopy(pvs), "prompt": deepcopy(prompt),
        "layout": deepcopy(layout), "mode": mode, "tool": tool,
        "text": text, "threshold": threshold,
    }
    if instance_count(item):
        item["status"] = "done"
    folder = app.runtime_dir / "batch_workspace" / validate_server_session_id(batch["session_id"]) / item["id"]
    folder.mkdir(parents=True, exist_ok=True)
    item["original"].save(folder / "source.png")
    item["snapshot"]["image"].save(folder / "image.png")
    # Durable masks/metadata are safe data, never pickled model or browser objects.
    masks, metadata = {}, []
    for pool_name, pool in (("pcs", pcs), ("pvs", pvs)):
        for key, inst in (pool or {}).get("instances", {}).items():
            mask = inst.get("mask_fullres_bool")
            if mask is None or inst.get("deleted"):
                continue
            mask_key = f"{pool_name}_{key}"
            masks[mask_key] = np.asarray(mask, dtype=bool)
            metadata.append({"id": key, "pool": pool_name,
                             "category": inst.get("category_name"), "mask": mask_key})
    np.savez_compressed(folder / "masks.npz", **masks)
    (folder / "metadata.json").write_text(json.dumps({
        "name": item["name"], "crop": source.get("crop_bbox_xyxy"),
        "mode": mode, "instances": metadata}, ensure_ascii=False), encoding="utf-8")
    return True


def restore_item(app, batch, item, session, inherit_config=None):
    snap = item.get("snapshot") or {}
    inherit_config = {} if snap else (inherit_config or {})
    mode = snap.get("mode", inherit_config.get("mode", "PVS Manual"))
    layout = app._new_layout_state(session["session_id"])
    layout.update(deepcopy(snap.get("layout") or {}))
    layout["session_id"] = session["session_id"]
    loaded = list(app._source_upload_workspace(item["original"], mode, session, layout))
    source, image, pcs, pvs, prompt = loaded[0], loaded[3], loaded[4], loaded[5], loaded[6]
    saved_source = snap.get("source") or {}
    crop = saved_source.get("crop_bbox_xyxy")
    if crop and list(crop) != [0, 0, item["original"].width, item["original"].height]:
        source["pending_crop_bbox_xyxy"] = list(crop)
        loaded = list(app._apply_source_crop(source, mode, session, layout))
        source, image, pcs, pvs, prompt = loaded[0], loaded[3], loaded[4], loaded[5], loaded[6]
    if snap:
        pcs, pvs, prompt = deepcopy(snap["pcs"]), deepcopy(snap["pvs"]), deepcopy(snap["prompt"])
        # Browser event sequence numbers are scoped to the newly-created image.
        for key in list(prompt):
            if key.startswith("last_") or key == "bbox_start":
                prompt[key] = None
    item["live_image_id"] = image["image_id"]
    item["live_source_id"] = source["source_image_id"]
    batch["active_id"] = item["id"]
    return (source, image, pcs, pvs, prompt, layout, mode,
            snap.get("tool", inherit_config.get("tool", "bbox")),
            snap.get("text", inherit_config.get("text", "")),
            snap.get("threshold", inherit_config.get("threshold", 0.4)))


def selected_for_run(batch, selected, retry=False):
    selected = set(selected or [])
    return [i["id"] for i in batch["items"] if i["id"] in selected and
            (i["status"] == "failed" if retry else i["status"] != "done")]


def pcs_input_for_batch(pcs, text):
    """Use this tile's visual prompts, never another tile's coordinates."""
    if not str(text or "").strip() and not (pcs or {}).get("positive_boxes"):
        raise ValueError("本图没有正样本框；请给本图画框，或填写共用文本提示")
    return deepcopy(pcs)


def send_tiles(app, batch, selected, stitch, live_values=None):
    from .annotated_stitch import change_queue
    from .annotated_stitch_io import make_tile
    tiles = list(stitch.get("saved_tiles") or [])
    selected = set(selected or [])
    count = 0
    for item in batch["items"]:
        if item["id"] not in selected:
            continue
        snap = item.get("snapshot")
        if live_values is not None and item["id"] == batch.get("active_id"):
            source, image, pcs, pvs, _prompt, _layout, mode, *_ = live_values
            source_id = (source or {}).get("source_image_id")
            if source_id and source_id == item.get("live_source_id") and image and image.get("image_id") and (
                item.get("live_image_id") == image["image_id"]
                or (source or {}).get("workspace_image_id") == image["image_id"]
            ):
                # make_tile detaches only final masks; probability maps and
                # durable workspace archives are unnecessary for this transfer.
                snap = {"source": source, "image": app._workspace(image)["image"],
                        "pcs": pcs, "pvs": pvs, "mode": mode}
        if not snap:
            continue
        pool_name = "pcs" if snap["mode"] == "PCS Auto" else "pvs"
        pool = snap[pool_name]
        instances = []
        for inst in app._active_instances(pool):
            data = {"id": inst["id"], "mask": inst["mask_fullres_bool"],
                    "category_name": inst.get("category_name") or (pool.get("text_prompt") if pool_name == "pcs" else "") or f"{pool_name}_object",
                    "provenance": {"pool": pool_name, "history": app._history_json(inst.get("prompt_history") or [])}}
            if not inst.get("score_missing") and inst.get("score") is not None:
                data["score"] = float(inst["score"])
            instances.append(data)
        if not instances and item["status"] != "done":
            continue
        tile = make_tile(snap["image"], item["name"], instances, tile_id=item["id"],
                         provenance={"batch_item_id": item["id"], "crop": snap["source"].get("crop_bbox_xyxy")})
        tiles = [old for old in tiles if old["tile_id"] != item["id"]]
        tiles.append(tile)
        count += 1
    if not count:
        raise ValueError("请勾选已标注图片")
    return change_queue(stitch, tiles), count


def remove_images(batch, image_ids):
    if batch.get("running"):
        raise ValueError("请先取消批处理，再删除图片")
    targets = set(image_ids or [])
    items = batch["items"]
    removed = [item for item in items if item["id"] in targets]
    if not removed:
        return 0, False
    old_active = batch.get("active_id")
    old_index = next((i for i, item in enumerate(items) if item["id"] == old_active), 0)
    remaining = [item for item in items if item["id"] not in targets]
    active_removed = old_active in targets
    batch["items"] = remaining
    if active_removed:
        batch["active_id"] = remaining[min(old_index, len(remaining) - 1)]["id"] if remaining else None
    valid = {item["id"] for item in remaining}
    batch["selected_ids"] = [key for key in batch.get("selected_ids", []) if key in valid]
    batch["pending"] = [key for key in batch.get("pending", []) if key in valid]
    # Published annotations and stitch tiles are independent saved copies.
    return len(removed), active_removed


def bind_batch_workspace(state_refs, image_refs, stitch_refs, callbacks, app):
    s, r, t = state_refs, image_refs, stitch_refs
    timer = gr.Timer(0.5, active=False)
    shared = [r.batch_state, s.session_state, s.source_image_state, s.image_state,
              s.pcs_state, s.pvs_state, s.prompt_state, s.layout_state,
              r.mode, r.click_tool, r.text_prompt, r.confidence_threshold,
              r.batch_selection, r.batch_upload, t.stitch_state]
    common = [r.image_upload, r.result_image, r.analysis_report, r.pcs_summary,
              r.pvs_summary, r.active_pvs, r.interaction_info, r.pvs_pending_count]
    controls = [r.batch_upload, r.batch_prev_btn, r.batch_next_btn, r.batch_save_btn,
                r.batch_delete_current_btn, r.batch_delete_selected_btn,
                r.batch_run_btn, r.batch_retry_btn, r.batch_stitch_btn,
                r.batch_select_all_btn, r.batch_select_none_btn, r.batch_selection,
                r.source_image_upload, r.apply_crop_btn, r.use_full_image_btn,
                r.run_pcs_btn, r.create_pvs_batch_btn, r.finish_polygon_btn,
                r.pvs_point_btn, r.layout_point_btn, r.create_from_layout_btn,
                r.delete_active_pvs_btn, r.clear_pvs_btn, r.clear_pcs_instances_btn,
                r.mode, r.click_tool, r.text_prompt, r.confidence_threshold,
                r.clear_prompt_btn, r.workspace_gesture_overlay, r.source_crop_overlay]
    # Custom gesture components do not have interactive props; disable through payload.
    controls = [c for c in controls if c not in (r.workspace_gesture_overlay, r.source_crop_overlay)]
    queue_gallery = t.annotated_gallery
    queue_selection = t.annotated_selection
    mode_components = [
        s.prompt_state, s.bbox_payload, s.point_payload, s.polygon_payload,
        r.click_tool, r.finish_polygon_btn, r.pcs_bbox_tools, r.pcs_panel,
        r.pvs_panel, r.pvs_action_panel, r.analysis_report_panel, r.pvs_layout_panel,
        r.layout_transform_panel, r.pvs_bbox_prompt_panel, r.pvs_point_prompt_panel,
        r.pvs_polygon_prompt_panel, r.pcs_bbox_selector, r.pvs_pending_bbox_selector,
        r.layout_point_refine_panel, *common, r.layout_editor,
    ]
    outputs = list(dict.fromkeys([
        r.batch_state, r.batch_gallery, r.batch_selection, r.batch_status, timer,
        s.source_image_state, s.image_state, s.pcs_state, s.pvs_state,
        s.prompt_state, s.layout_state, r.source_image_upload, r.source_crop_overlay,
        r.source_crop_status, r.workspace_gesture_overlay, r.pcs_bbox_selector,
        r.pvs_pending_bbox_selector, *common, r.export_file, r.layout_editor,
        s.bbox_payload, s.polygon_payload, s.point_payload, r.mode, r.click_tool,
        r.text_prompt, r.confidence_threshold, r.batch_cancel_btn, *controls,
        t.stitch_state, queue_gallery, queue_selection,
        s.main_tabs, s.template_match_state, r.template_match_preview,
        r.template_match_file, r.template_match_status, *mode_components,
    ]))

    def execute(action, batch, session_state, source_image_state, image_state,
                pcs_state, pvs_state, prompt_state, layout_state, mode, click_tool,
                text, threshold, selected, files, stitch_state, index=None):
        batch = owned_batch(batch, session_state)
        if action != "selection":
            selected = batch.get("selected_ids", selected)
        values = (source_image_state, image_state, pcs_state, pvs_state, prompt_state,
                  layout_state, mode, click_tool, text, threshold)
        changed = False
        extra = {}
        message = ""
        if batch.get("running") and action not in ("step", "cancel"):
            return {r.batch_status: "批处理进行中；可取消后继续编辑"}
        had_active = any(item["id"] == batch.get("active_id") for item in batch["items"])
        try:
            if action not in ("step", "cancel", "selection", "stitch", "delete_current", "delete_selected"):
                saved = save_snapshot(app, batch, *values)
                if action in ("prev", "next", "select") and had_active and not saved:
                    raise ValueError("\u5f53\u524d\u56fe\u7247\u4fdd\u5b58\u5931\u8d25\uff0c\u672a\u5207\u6362")
                if action == "save" and not saved:
                    raise ValueError("\u5f53\u524d\u56fe\u7247\u4fdd\u5b58\u5931")
            if action == "upload":
                added = add_images(batch, files)
                selected = list(dict.fromkeys([*(selected or []), *(i["id"] for i in added)]))
                if added:
                    values = restore_item(app, batch, added[0], session_state)
                    changed = True
                message = f"已添加 {len(added)} 张图片"
            elif action in ("delete_current", "delete_selected"):
                targets = [batch.get("active_id")] if action == "delete_current" else selected
                count, active_removed = remove_images(batch, targets)
                selected = batch.get("selected_ids", [])
                message = f"已删除 {count} 张图片；当前队列共 {len(batch['items'])} 张" if count else "请先选择要删除的图片"
                if active_removed:
                    target = next((item for item in batch["items"] if item["id"] == batch["active_id"]), None)
                    if target is not None:
                        values = restore_item(app, batch, target, session_state)
                    else:
                        layout = app._new_layout_state(session_state["session_id"])
                        loaded = app._source_upload_workspace(None, mode, session_state, layout)
                        values = (loaded[0], loaded[3], loaded[4], loaded[5], loaded[6],
                                  layout, mode, click_tool, text, threshold)
                    changed = True
            elif action in ("prev", "next", "select"):
                ids = [i["id"] for i in batch["items"]]
                current = ids.index(batch['active_id']) if batch['active_id'] in ids else (-1 if action == 'next' else 0)
                dest = int(index) if action == "select" else current + (-1 if action == "prev" else 1)
                if ids:
                    dest = min(max(dest, 0), len(ids)-1)
                    target = batch["items"][dest]
                    inherit_config = None
                    if action == "next" and not target.get("snapshot"):
                        inherit_config = {
                            "mode": mode, "tool": click_tool, "text": text,
                            "threshold": threshold,
                        }
                    values = restore_item(app, batch, target, session_state,
                                          inherit_config=inherit_config)
                    changed = True
                    message = f"{dest+1} / {len(ids)} · {batch['items'][dest]['name']}"
            elif action in ("select_all", "select_none", "selection"):
                if action != "selection":
                    selected = [i["id"] for i in batch["items"]] if action == "select_all" else []
                message = f"已选择 {len(selected)} 张图片"
            elif action in ("run", "retry"):
                ids = selected_for_run(batch, selected, action == "retry")
                if not ids:
                    raise ValueError("请勾选待处理图片" if action == "run" else "没有勾选失败图片")
                batch.update(pending=ids, running=True, batch_text=text, batch_threshold=threshold)
                message = f"批量 PCS 已开始，共 {len(ids)} 张；文本为空时使用各图自己的正样本框"
                gr.Info(message)
            elif action == "cancel":
                batch.update(pending=[], running=False)
                message = "批处理已取消，已完成结果保留"
            elif action == "step":
                if batch.get("running") and batch.get("pending"):
                    tile_id = batch["pending"].pop(0)
                    item = next(i for i in batch["items"] if i["id"] == tile_id)
                    values = restore_item(app, batch, item, session_state)
                    source, image, pcs, pvs, prompt, layout, _, tool, _, _ = values
                    prior = deepcopy(pcs)
                    try:
                        new_pcs = pcs_input_for_batch(pcs, batch["batch_text"])
                        result = app._run_pcs(image, new_pcs, pvs, "PCS Auto",
                                              batch["batch_text"], batch["batch_threshold"])
                        if str(result[-2]).startswith("PCS failed"):
                            raise ValueError(str(result[-2]))
                        pcs = result[0]
                        item.update(status="done", error="")
                    except Exception as exc:
                        pcs = prior
                        item.update(status="failed", error=str(exc))
                    if not item.get("error"):
                        values = (source, image, pcs, pvs, prompt, layout, "PCS Auto",
                                  tool, batch["batch_text"], batch["batch_threshold"])
                    save_snapshot(app, batch, *values)
                    if item.get("error"):
                        item["status"] = "failed"
                    changed = True
                    status_text = "失败" if item.get("error") else "已完成"
                    message = f"{item['name']} · {status_text} · 剩余 {len(batch['pending'])}"
                    if item.get("error"):
                        message += f" · {item['error']}"
                        gr.Warning(message)
                if not batch.get("pending"):
                    batch["running"] = False
            elif action == "stitch":
                gr.Info("正在将选中图片的标注送往拼接")
                stitch_state, count = send_tiles(app, batch, selected, stitch_state, live_values=values)
                from .annotated_stitch import queue_view
                gallery, selection = queue_view(stitch_state)
                extra.update({t.stitch_state: stitch_state, queue_gallery: gallery,
                              queue_selection: selection, s.main_tabs: gr.update(selected="tab_stitch")})
                message = f"已送往拼接 {count} 张"
            else:
                message = "当前图片已保存"
        except Exception as exc:
            message = str(exc)
            gr.Warning(message)
            if action == "step":
                batch["running"] = False
        valid_ids = {i["id"] for i in batch["items"]}
        selected = [i for i in (selected or []) if i in valid_ids]
        batch["selected_ids"] = selected
        gallery, choice = queue_display(batch, selected)
        result = {r.batch_state: batch, r.batch_gallery: gallery,
                  r.batch_selection: choice, r.batch_status: message,
                  timer: gr.update(active=batch.get("running", False)),
                  r.batch_cancel_btn: gr.update(interactive=batch.get("running", False)), **extra}
        for c in controls:
            result[c] = gr.update(interactive=not batch.get("running", False))
        result[r.batch_selection] = {**choice, **gr.update(interactive=not batch.get("running", False))}
        if action == "upload":
            result[r.batch_upload] = gr.update(value=None, interactive=not batch.get("running", False))
        if changed:
            source, image, pcs, pvs, prompt, layout, mode, tool, text, threshold = values
            mode_values = app._switch_mode_with_layout_editor(mode, image, pcs, pvs, layout)
            result.update(dict(zip(mode_components, mode_values)))
            result.update(dict(zip([s.source_image_state, s.image_state, s.pcs_state,
                                    s.pvs_state, s.prompt_state, s.layout_state], values[:6])))
            result.update(dict(zip(common, app._view(image, pcs, pvs, mode, message, prompt, layout))))
            result.update({
                r.source_image_upload: gr.update(
                    value=app._source_image_cache_get(source) if source.get("source_image_id") else None,
                    interactive=not batch["running"]),
                r.source_crop_overlay: app._source_gesture_payload(source),
                r.source_crop_status: message,
                r.workspace_gesture_overlay: app._workspace_gesture_payload(image, mode, tool),
                r.pcs_bbox_selector: app._pcs_bbox_choices(pcs),
                r.pvs_pending_bbox_selector: app._pvs_pending_bbox_choices(pvs),
                r.export_file: None, r.layout_editor: app._layout_editor_empty(image),
                s.bbox_payload: "", s.polygon_payload: "", s.point_payload: "",
                r.mode: gr.update(value=mode, interactive=not batch["running"]),
                r.click_tool: {**result.get(r.click_tool, {}), **gr.update(value=tool, interactive=not batch["running"])},
                r.text_prompt: gr.update(value=text, interactive=not batch["running"]),
                r.confidence_threshold: gr.update(value=threshold, interactive=not batch["running"]),
                s.template_match_state: app._new_template_match_state(session_state["session_id"]),
                r.template_match_preview: None,
                r.template_match_file: None, r.template_match_status: "",
            })
        if batch.get("running"):
            result[r.workspace_gesture_overlay] = {"server_view": {"enabled": False}, "client_intent": {}}
            result[r.source_crop_overlay] = {"server_view": {"enabled": False}, "client_intent": {}}
        elif not changed and image_state and image_state.get("image_id"):
            result[r.workspace_gesture_overlay] = app._workspace_gesture_payload(image_state, mode, click_tool)
            result[r.source_crop_overlay] = app._source_gesture_payload(source_image_state)
        return result

    def ordered(result, components=outputs):
        return tuple(result.get(component, gr.skip()) for component in components)

    def wrap(action, components=outputs):
        def callback(batch, session_state, source_image_state, image_state, pcs_state,
                     pvs_state, prompt_state, layout_state, mode, click_tool, text,
                     threshold, selected, files, stitch_state):
            return ordered(execute(action, batch, session_state, source_image_state, image_state,
                           pcs_state, pvs_state, prompt_state, layout_state, mode,
                           click_tool, text, threshold, selected, files, stitch_state), components)
        callback.__name__ = "batch_" + action
        return guard_callback(callback, registry=app._SESSION_REGISTRY,
                              trusted_proxy_cidrs=app._SESSION_TRUSTED_PROXY_CIDRS,
                              recovery_factory=app._session_recovery_states)

    options = dict(inputs=shared, outputs=outputs, concurrency_limit=1,
                   concurrency_id="image-prepost-state", show_progress="hidden", api_visibility="private")
    upload_callback = wrap("upload")
    r.batch_upload.upload(upload_callback, **options)
    r.batch_selection.input(wrap("selection"), **options)
    for component, action in ((r.batch_prev_btn,"prev"),(r.batch_next_btn,"next"),
                              (r.batch_delete_current_btn,"delete_current"),
                              (r.batch_delete_selected_btn,"delete_selected"),
                              (r.batch_save_btn,"save"),(r.batch_run_btn,"run"),
                              (r.batch_retry_btn,"retry"),(r.batch_cancel_btn,"cancel"),
                              (r.batch_select_all_btn,"select_all"),
                              (r.batch_select_none_btn,"select_none")):
        component.click(wrap(action), **options)
    # Handoff must not rebuild the source image and gesture components while hiding them.
    handoff_outputs = [r.batch_state, r.batch_status, t.stitch_state,
                       queue_gallery, queue_selection, s.main_tabs]
    r.batch_stitch_btn.click(wrap("stitch", handoff_outputs),
                            **{**options, "outputs": handoff_outputs})
    timer.tick(wrap("step"), **options)

    def batch_select(batch, session_state, source_image_state, image_state, pcs_state,
               pvs_state, prompt_state, layout_state, mode, click_tool, text,
               threshold, selected, files, stitch_state, evt: gr.SelectData):
        index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        return ordered(execute("select", batch, session_state, source_image_state, image_state,
                       pcs_state, pvs_state, prompt_state, layout_state, mode, click_tool,
                       text, threshold, selected, files, stitch_state, index))
    r.batch_gallery.select(guard_callback(batch_select, registry=app._SESSION_REGISTRY,
                            trusted_proxy_cidrs=app._SESSION_TRUSTED_PROXY_CIDRS,
                            recovery_factory=app._session_recovery_states), **options)

    def bind_external_upload(event, files_component):
        external_inputs = [
            files_component if component is r.batch_upload else component
            for component in shared
        ]
        # Hidden custom editors can enter a Svelte update loop during a cross-tab restore.
        hidden_custom_outputs = {s.main_tabs, r.source_crop_overlay, r.workspace_gesture_overlay, r.layout_editor}
        external_outputs = [
            component for component in outputs
            if component not in hidden_custom_outputs
        ]
        external_upload_callback = wrap("upload", external_outputs)
        restore_event = event.success(
            fn=external_upload_callback,
            inputs=external_inputs,
            outputs=external_outputs,
            concurrency_limit=1,
            concurrency_id="image-prepost-state",
            show_progress="hidden",
            api_visibility="private",
        )

        def batch_refresh_workspace_overlay(
            session_state,
            image_state,
            mode,
            click_tool,
        ):
            del session_state
            return app._workspace_gesture_payload(image_state, mode, click_tool)

        workspace_event = restore_event.success(
            fn=guard_callback(
                batch_refresh_workspace_overlay,
                registry=app._SESSION_REGISTRY,
                trusted_proxy_cidrs=app._SESSION_TRUSTED_PROXY_CIDRS,
                recovery_factory=app._session_recovery_states,
            ),
            inputs=[s.session_state, s.image_state, r.mode, r.click_tool],
            outputs=[r.workspace_gesture_overlay],
            concurrency_limit=1,
            concurrency_id="image-prepost-state",
            show_progress="hidden",
            api_visibility="private",
        )

        def batch_refresh_source_overlay(session_state, source_image_state):
            del session_state
            return app._source_gesture_payload(source_image_state)

        source_event = workspace_event.success(
            fn=guard_callback(
                batch_refresh_source_overlay,
                registry=app._SESSION_REGISTRY,
                trusted_proxy_cidrs=app._SESSION_TRUSTED_PROXY_CIDRS,
                recovery_factory=app._session_recovery_states,
            ),
            inputs=[s.session_state, s.source_image_state],
            outputs=[r.source_crop_overlay],
            concurrency_limit=1,
            concurrency_id="image-prepost-state",
            show_progress="hidden",
            api_visibility="private",
        )

        def batch_refresh_layout_editor(session_state, image_state, mode):
            del session_state
            if mode != "Layout Mask":
                return gr.skip()
            return app._layout_editor_empty(image_state)

        layout_event = source_event.success(
            fn=guard_callback(
                batch_refresh_layout_editor,
                registry=app._SESSION_REGISTRY,
                trusted_proxy_cidrs=app._SESSION_TRUSTED_PROXY_CIDRS,
                recovery_factory=app._session_recovery_states,
            ),
            inputs=[s.session_state, s.image_state, r.mode],
            outputs=[r.layout_editor],
            concurrency_limit=1,
            concurrency_id="image-prepost-state",
            show_progress="hidden",
            api_visibility="private",
        )
        return layout_event

    return bind_external_upload
