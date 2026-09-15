"""Event bindings for the responsive EL UI-repair workspace."""

from __future__ import annotations

import time
from typing import Any

import gradio as gr

from sam3_demo.lama_runtime import get_lama_runtime
from sam3_demo.session_guard import guard_callback
from sam3_demo import ui_repair_core as core


def bind_repair_events(
    *,
    state_refs,
    repair_refs,
    app,
    runtime_root,
    model_path,
    bind_batch_upload,
):
    state_component = state_refs.repair_state
    session_component = state_refs.session_state

    def guarded(function):
        return guard_callback(
            function,
            registry=app._SESSION_REGISTRY,
            trusted_proxy_cidrs=app._SESSION_TRUSTED_PROXY_CIDRS,
            recovery_factory=app._session_recovery_states,
        )

    def selected_ids(state, selected, *, default_active=False):
        valid = {item["id"] for item in state.get("items", [])}
        values = [image_id for image_id in (selected or []) if image_id in valid]
        if not values and default_active and state.get("active_id") in valid:
            values = [state["active_id"]]
        state["selected_ids"] = values
        return values

    def view(state, selected, tool, brush_size, alpha, message):
        item = core.active_item(state)
        gallery, choices = core.queue_view(state)
        valid_selection = selected_ids(state, selected)
        selection_update = gr.update(choices=choices, value=valid_selection)
        if item is None:
            position = "尚未添加图片"
        else:
            index = next(
                index
                for index, candidate in enumerate(state["items"], 1)
                if candidate["id"] == item["id"]
            )
            position = f"{index} / {len(state['items'])} · {item['name']} · {item['status']}"
        return (
            state,
            gallery,
            selection_update,
            core.editor_payload(
                item,
                tool=tool,
                brush_size=brush_size,
                overlay_alpha=alpha,
            ),
            core.result_image(item),
            message,
            position,
        )

    common_outputs = [
        state_component,
        repair_refs.repair_gallery,
        repair_refs.repair_selection,
        repair_refs.repair_editor,
        repair_refs.repair_result,
        repair_refs.repair_status,
        repair_refs.repair_position,
    ]

    def upload(
        repair_state,
        session_state,
        files,
        selected,
        tool,
        brush_size,
        alpha,
    ):
        state = core.owned_repair_state(repair_state, session_state)
        try:
            added = core.add_uploaded_images(state, files, runtime_root)
            message = f"已添加 {len(added)} 张图片；当前队列共 {len(state['items'])} 张"
        except Exception as exc:
            message = str(exc)
            gr.Warning(message)
        return (*view(state, state.get("selected_ids", selected), tool, brush_size, alpha, message), None)

    repair_refs.repair_upload.upload(
        guarded(upload),
        inputs=[
            state_component,
            session_component,
            repair_refs.repair_upload,
            repair_refs.repair_selection,
            repair_refs.repair_tool,
            repair_refs.repair_brush_size,
            repair_refs.repair_alpha,
        ],
        outputs=[*common_outputs, repair_refs.repair_upload],
        concurrency_limit=1,
        concurrency_id="ui-repair-state",
        show_progress="hidden",
        api_visibility="private",
    )

    def select_gallery(
        repair_state,
        session_state,
        selected,
        tool,
        brush_size,
        alpha,
        evt: gr.SelectData,
    ):
        state = core.owned_repair_state(repair_state, session_state)
        index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        try:
            item = core.select_item(state, int(index))
            message = f"已切换到 {item['name']}"
        except Exception as exc:
            message = str(exc)
            gr.Warning(message)
        return view(state, selected, tool, brush_size, alpha, message)

    repair_refs.repair_gallery.select(
        guarded(select_gallery),
        inputs=[
            state_component,
            session_component,
            repair_refs.repair_selection,
            repair_refs.repair_tool,
            repair_refs.repair_brush_size,
            repair_refs.repair_alpha,
        ],
        outputs=common_outputs,
        concurrency_limit=1,
        concurrency_id="ui-repair-state",
        show_progress="hidden",
        api_visibility="private",
    )

    def move(
        delta,
        repair_state,
        session_state,
        selected,
        tool,
        brush_size,
        alpha,
    ):
        state = core.owned_repair_state(repair_state, session_state)
        try:
            item = core.move_active(state, delta)
            message = f"已切换到 {item['name']}"
        except Exception as exc:
            message = str(exc)
            gr.Warning(message)
        return view(state, selected, tool, brush_size, alpha, message)

    def previous(repair_state, session_state, selected, tool, brush_size, alpha):
        return move(-1, repair_state, session_state, selected, tool, brush_size, alpha)

    def following(repair_state, session_state, selected, tool, brush_size, alpha):
        return move(1, repair_state, session_state, selected, tool, brush_size, alpha)

    navigation_inputs = [
        state_component,
        session_component,
        repair_refs.repair_selection,
        repair_refs.repair_tool,
        repair_refs.repair_brush_size,
        repair_refs.repair_alpha,
    ]
    for button, callback in (
        (repair_refs.repair_prev_btn, previous),
        (repair_refs.repair_next_btn, following),
    ):
        button.click(
            guarded(callback),
            inputs=navigation_inputs,
            outputs=common_outputs,
            concurrency_limit=1,
            concurrency_id="ui-repair-state",
            show_progress="hidden",
            api_visibility="private",
        )

    def set_selection(repair_state, session_state, selected):
        state = core.owned_repair_state(repair_state, session_state)
        values = selected_ids(state, selected)
        return state, f"已选择 {len(values)} 张图片"

    repair_refs.repair_selection.input(
        guarded(set_selection),
        inputs=[state_component, session_component, repair_refs.repair_selection],
        outputs=[state_component, repair_refs.repair_status],
        concurrency_limit=1,
        concurrency_id="ui-repair-state",
        show_progress="hidden",
        api_visibility="private",
    )

    def select_all(repair_state, session_state):
        state = core.owned_repair_state(repair_state, session_state)
        values = [item["id"] for item in state.get("items", [])]
        state["selected_ids"] = values
        _gallery, choices = core.queue_view(state)
        return state, gr.update(choices=choices, value=values), f"已选择 {len(values)} 张图片"

    repair_refs.repair_select_all_btn.click(
        guarded(select_all),
        inputs=[state_component, session_component],
        outputs=[state_component, repair_refs.repair_selection, repair_refs.repair_status],
        concurrency_limit=1,
        concurrency_id="ui-repair-state",
        show_progress="hidden",
        api_visibility="private",
    )

    def remove_current(
        repair_state,
        session_state,
        selected,
        tool,
        brush_size,
        alpha,
    ):
        state = core.owned_repair_state(repair_state, session_state)
        item = core.active_item(state)
        if item is None:
            message = "当前没有可删除的图片，请先添加图片"
        else:
            removed = core.remove_repair_items(state, [item["id"]])
            if removed:
                message = f"已删除当前图片；剩余 {len(state['items'])} 张"
            else:
                message = "当前图片已不在队列，未删除任何内容；请刷新后重试"
        return view(
            state,
            selected,
            tool,
            brush_size,
            alpha,
            message,
        )

    def remove_selected(
        repair_state,
        session_state,
        selected,
        tool,
        brush_size,
        alpha,
    ):
        state = core.owned_repair_state(repair_state, session_state)
        values = selected_ids(state, selected)
        if not values:
            message = "未选择图片，未删除任何内容；请先选择图片"
        else:
            removed = core.remove_repair_items(state, values)
            if removed:
                message = f"已删除选中图片 {len(removed)} 张；剩余 {len(state['items'])} 张"
            else:
                message = "选中的图片已不在当前队列，未删除任何内容；请重新选择"
        return view(
            state,
            state.get("selected_ids", selected),
            tool,
            brush_size,
            alpha,
            message,
        )

    for button, callback in (
        (repair_refs.repair_remove_current_btn, remove_current),
        (repair_refs.repair_remove_selected_btn, remove_selected),
    ):
        button.click(
            guarded(callback),
            inputs=navigation_inputs,
            outputs=common_outputs,
            concurrency_limit=1,
            concurrency_id="ui-repair-state",
            show_progress="hidden",
            api_visibility="private",
        )

    def editor_changed(repair_state, session_state, editor_value):
        state = core.owned_repair_state(repair_state, session_state)
        message = "当前修复区域已保存"
        try:
            core.commit_editor_value(state, editor_value)
        except Exception as exc:
            message = str(exc)
            gr.Warning(message)
        item = core.active_item(state)
        gallery, _choices = core.queue_view(state)
        position = (
            f"{next((index for index, candidate in enumerate(state['items'], 1) if candidate['id'] == item['id']), 0)}"
            f" / {len(state['items'])} · {item['name']} · {item['status']}"
            if item
            else "尚未添加图片"
        )
        return state, gallery, core.result_image(item), message, position

    repair_refs.repair_editor.change(
        guarded(editor_changed),
        inputs=[state_component, session_component, repair_refs.repair_editor],
        outputs=[
            state_component,
            repair_refs.repair_gallery,
            repair_refs.repair_result,
            repair_refs.repair_status,
            repair_refs.repair_position,
        ],
        concurrency_limit=1,
        concurrency_id="ui-repair-state",
        show_progress="hidden",
        api_visibility="private",
    )

    def update_settings(
        repair_state,
        session_state,
        tool,
        brush_size,
        alpha,
    ):
        state = core.owned_repair_state(repair_state, session_state)
        return core.editor_payload(
            core.active_item(state),
            tool=tool,
            brush_size=brush_size,
            overlay_alpha=alpha,
        )

    for setting in (
        repair_refs.repair_tool,
        repair_refs.repair_brush_size,
        repair_refs.repair_alpha,
    ):
        setting.input(
            guarded(update_settings),
            inputs=[
                state_component,
                session_component,
                repair_refs.repair_tool,
                repair_refs.repair_brush_size,
                repair_refs.repair_alpha,
            ],
            outputs=[repair_refs.repair_editor],
            concurrency_limit=1,
            concurrency_id="ui-repair-state",
            show_progress="hidden",
            api_visibility="private",
        )

    detection_inputs = [
        state_component,
        session_component,
        repair_refs.repair_selection,
        repair_refs.repair_tool,
        repair_refs.repair_brush_size,
        repair_refs.repair_alpha,
        repair_refs.detect_saturation,
        repair_refs.detect_value,
        repair_refs.detect_min_area,
        repair_refs.detect_padding,
        repair_refs.detect_merge,
    ]

    def detection_callback(colors, batch):
        def detect(
            repair_state,
            session_state,
            selected,
            tool,
            brush_size,
            alpha,
            saturation,
            value,
            min_area,
            padding,
            merge_distance,
        ):
            state = core.owned_repair_state(repair_state, session_state)
            targets = selected_ids(state, selected, default_active=True) if batch else [state.get("active_id")]
            targets = [target for target in targets if target]
            try:
                if not targets:
                    raise ValueError("当前没有待修复图片")
                affected, regions = core.apply_color_detection(
                    state,
                    targets,
                    colors,
                    saturation=int(saturation),
                    value=int(value),
                    min_area=int(min_area),
                    padding=int(padding),
                    merge_distance=int(merge_distance),
                )
                message = f"已在 {affected} 张图片中加入 {regions} 个颜色区域"
            except Exception as exc:
                message = str(exc)
                gr.Warning(message)
            return view(state, selected, tool, brush_size, alpha, message)

        detect.__name__ = "repair_detect_" + "_".join(colors) + ("_batch" if batch else "")
        return detect

    for button, colors, batch in (
        (repair_refs.detect_red_btn, ("red",), False),
        (repair_refs.detect_yellow_btn, ("yellow",), False),
        (repair_refs.detect_both_btn, ("red", "yellow"), False),
        (repair_refs.detect_selected_btn, ("red", "yellow"), True),
    ):
        button.click(
            guarded(detection_callback(colors, batch)),
            inputs=detection_inputs,
            outputs=common_outputs,
            concurrency_limit=1,
            concurrency_id="ui-repair-state",
            show_progress="hidden",
            api_visibility="private",
        )

    def clear_mask(
        repair_state,
        session_state,
        selected,
        tool,
        brush_size,
        alpha,
    ):
        state = core.owned_repair_state(repair_state, session_state)
        try:
            core.clear_active_mask(state)
            message = "已清空当前图片的修复区域"
        except Exception as exc:
            message = str(exc)
            gr.Warning(message)
        return view(state, selected, tool, brush_size, alpha, message)

    repair_refs.clear_mask_btn.click(
        guarded(clear_mask),
        inputs=navigation_inputs,
        outputs=common_outputs,
        concurrency_limit=1,
        concurrency_id="ui-repair-state",
        show_progress="hidden",
        api_visibility="private",
    )

    repair_inputs = [
        state_component,
        session_component,
        repair_refs.repair_selection,
        repair_refs.repair_editor,
        repair_refs.repair_tool,
        repair_refs.repair_brush_size,
        repair_refs.repair_alpha,
        repair_refs.detect_saturation,
        repair_refs.detect_value,
        repair_refs.detect_min_area,
        repair_refs.detect_padding,
        repair_refs.detect_merge,
    ]

    def repair_callback(batch):
        def run(
            repair_state,
            session_state,
            selected,
            editor_value,
            tool,
            brush_size,
            alpha,
            saturation,
            value,
            min_area,
            padding,
            merge_distance,
        ):
            state = core.owned_repair_state(repair_state, session_state)
            try:
                if core.active_item(state) is not None and isinstance(editor_value, dict):
                    core.commit_editor_value(state, editor_value)
                targets = (
                    selected_ids(state, selected, default_active=True)
                    if batch
                    else [state.get("active_id")]
                )
                targets = [target for target in targets if target]
                detected_images = 0
                detected_regions = 0
                if batch:
                    detected_images, detected_regions = core.apply_color_detection(
                        state,
                        targets,
                        ("red", "yellow"),
                        saturation=int(saturation),
                        value=int(value),
                        min_area=int(min_area),
                        padding=int(padding),
                        merge_distance=int(merge_distance),
                    )
                if not targets:
                    raise ValueError("当前没有待修复图片")
                completed, failures = core.repair_items(
                    state,
                    targets,
                    get_lama_runtime(model_path),
                )
                message = f"修复完成：{completed}/{len(targets)} 张"
                if batch:
                    message = (
                        f"自动检测：{detected_images} 张、{detected_regions} 个红黄区域；{message}"
                    )
                if failures:
                    message += "；" + "；".join(failures[:3])
                    gr.Warning(message)
                elif completed:
                    gr.Info(message)
            except Exception as exc:
                message = str(exc)
                gr.Warning(message)
            return view(state, selected, tool, brush_size, alpha, message)

        run.__name__ = "repair_selected" if batch else "repair_current"
        return run

    for button, batch in (
        (repair_refs.repair_current_btn, False),
        (repair_refs.repair_selected_btn, True),
    ):
        button.click(
            guarded(repair_callback(batch)),
            inputs=repair_inputs,
            outputs=common_outputs,
            concurrency_limit=1,
            concurrency_id="ui-repair-model",
            show_progress="full",
            show_progress_on=[repair_refs.repair_result],
            api_visibility="private",
        )

    def send_to_segmentation(
        repair_state,
        session_state,
        selected,
        editor_value,
    ):
        state = core.owned_repair_state(repair_state, session_state)
        try:
            if core.active_item(state) is not None and isinstance(editor_value, dict):
                core.commit_editor_value(state, editor_value)
            targets = selected_ids(state, selected, default_active=True)
            paths = core.repaired_paths(state, targets)
            message = f"正在将 {len(paths)} 张修复结果送往智能图像分割"
            gr.Info(message)
            return state, message, paths
        except Exception as exc:
            message = str(exc)
            gr.Warning(message)
            return state, message, []

    repair_refs.send_repaired_btn.click(
        fn=None,
        inputs=[],
        outputs=[],
        js="""() => {
            const tabs = Array.from(document.querySelectorAll('#main_tabs [role=\"tab\"]'));
            if (tabs[1] instanceof HTMLElement) requestAnimationFrame(() => tabs[1].click());
            return [];
        }""",
        queue=False,
        api_visibility="private",
    )

    send_event = repair_refs.send_repaired_btn.click(
        guarded(send_to_segmentation),
        inputs=[
            state_component,
            session_component,
            repair_refs.repair_selection,
            repair_refs.repair_editor,
        ],
        outputs=[
            state_component,
            repair_refs.repair_status,
            state_refs.repair_handoff_paths,
        ],
        concurrency_limit=1,
        concurrency_id="ui-repair-state",
        show_progress="hidden",
        api_visibility="private",
    )
    def wait_for_segmentation_tab():
        # Let Gradio mount the custom canvases before restoring their state.
        time.sleep(0.5)

    settle_event = send_event.success(
        fn=wait_for_segmentation_tab,
        inputs=[],
        outputs=[],
        concurrency_limit=8,
        concurrency_id="image-prepost-state",
        show_progress="hidden",
        api_visibility="private",
    )
    bind_batch_upload(settle_event, state_refs.repair_handoff_paths)

__all__ = ["bind_repair_events"]
