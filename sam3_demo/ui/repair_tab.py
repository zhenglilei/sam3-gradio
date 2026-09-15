"""EL screenshot UI-repair tab construction."""

from __future__ import annotations

import gradio as gr

from .refs import ComponentRefs


def build_repair_tab(
    *,
    RepairMaskEditor,
    editor_import_error,
    empty_editor_payload,
):
    with gr.TabItem("图像修复", id="tab_repair", render_children=True) as repair_tab:
        with gr.Column(elem_id="repair-workspace"):
            with gr.Row(elem_id="repair-main-row"):
                with gr.Column(elem_id="repair-library", min_width=220):
                    repair_upload = gr.File(
                        label="添加待修复图片",
                        file_count="multiple",
                        type="filepath",
                        file_types=["image"],
                        height=104,
                    )
                    repair_gallery = gr.Gallery(
                        label="图片队列",
                        columns=1,
                        height=392,
                        allow_preview=False,
                        buttons=[],
                        interactive=False,
                        object_fit="contain",
                        show_label=False,
                        elem_id="repair-gallery",
                    )
                    with gr.Row(elem_classes="repair-nav-row"):
                        repair_prev_btn = gr.Button("上一张", size="sm")
                        repair_next_btn = gr.Button("下一张", size="sm")
                    repair_position = gr.Markdown("尚未添加图片", elem_id="repair-position")
                    with gr.Accordion("批量范围", open=False):
                        repair_select_all_btn = gr.Button("全选", size="sm")
                        repair_selection = gr.Dropdown(
                            choices=[],
                            value=[],
                            multiselect=True,
                            label="选中的图片",
                        )

                with gr.Column(elem_id="repair-canvas-column", min_width=360):
                    with gr.Row(elem_id="repair-canvas-pair"):
                        with gr.Column(min_width=280, elem_classes="repair-canvas-pane"):
                            gr.Markdown("#### 修复区域")
                            if RepairMaskEditor is not None:
                                repair_editor = RepairMaskEditor(
                                    value=empty_editor_payload(),
                                    label="修复区域编辑器",
                                    show_label=False,
                                    height="clamp(360px, 55vh, 560px)",
                                    elem_id="repair-mask-editor",
                                )
                            else:
                                gr.Markdown(
                                    f"修复画布组件不可用：{editor_import_error}",
                                    elem_classes="repair-inline-warning",
                                )
                                repair_editor = gr.JSON(
                                    value=empty_editor_payload(),
                                    visible=False,
                                )
                        with gr.Column(min_width=280, elem_classes="repair-canvas-pane"):
                            gr.Markdown("#### 修复结果")
                            repair_result = gr.Image(
                                type="pil",
                                label="修复结果",
                                show_label=False,
                                interactive=False,
                                height="clamp(360px, 55vh, 560px)",
                                elem_id="repair-result",
                            )

                with gr.Column(elem_id="repair-tools", min_width=260):
                    with gr.Group(elem_classes="repair-tool-section"):
                        gr.Markdown("#### 自动标记")
                        with gr.Row():
                            detect_red_btn = gr.Button("红色", variant="secondary")
                            detect_yellow_btn = gr.Button("黄色", variant="secondary")
                        detect_both_btn = gr.Button("红色 + 黄色", variant="primary")
                        detect_selected_btn = gr.Button("批量标记选中图片")
                        with gr.Accordion("检测参数", open=False):
                            detect_saturation = gr.Slider(
                                minimum=20,
                                maximum=255,
                                value=80,
                                step=1,
                                label="饱和度阈值",
                            )
                            detect_value = gr.Slider(
                                minimum=20,
                                maximum=255,
                                value=80,
                                step=1,
                                label="亮度阈值",
                            )
                            detect_min_area = gr.Number(
                                value=20,
                                minimum=1,
                                maximum=10000,
                                precision=0,
                                label="最小面积",
                            )
                            detect_padding = gr.Number(
                                value=5,
                                minimum=0,
                                maximum=100,
                                precision=0,
                                label="矩形外扩",
                            )
                            detect_merge = gr.Number(
                                value=10,
                                minimum=0,
                                maximum=100,
                                precision=0,
                                label="合并距离",
                            )

                    with gr.Group(elem_classes="repair-tool-section"):
                        gr.Markdown("#### 手动修正")
                        repair_tool = gr.Radio(
                            choices=[
                                ("画笔", "brush"),
                                ("擦除", "eraser"),
                                ("矩形添加", "rect_add"),
                                ("矩形删除", "rect_erase"),
                            ],
                            value="brush",
                            label="工具",
                            show_label=False,
                            elem_classes="repair-tool-radio",
                        )
                        repair_brush_size = gr.Slider(
                            minimum=1,
                            maximum=120,
                            value=20,
                            step=1,
                            label="笔刷大小",
                        )
                        repair_alpha = gr.Slider(
                            minimum=0.05,
                            maximum=0.9,
                            value=0.45,
                            step=0.05,
                            label="标记透明度",
                        )
                        clear_mask_btn = gr.Button("清空当前标记", variant="secondary")

                    with gr.Group(elem_classes="repair-tool-section"):
                        gr.Markdown("#### 生成结果")
                        repair_current_btn = gr.Button("修复当前图片", variant="primary")
                        repair_selected_btn = gr.Button("批量修复选中图片")
                        send_repaired_btn = gr.Button("送往智能图像分割", variant="primary")

            repair_status = gr.Markdown(
                "添加图片后，先自动标记红黄区域，再用画布修正。",
                elem_id="repair-status",
            )

    return ComponentRefs(
        repair_tab=repair_tab,
        repair_upload=repair_upload,
        repair_gallery=repair_gallery,
        repair_prev_btn=repair_prev_btn,
        repair_next_btn=repair_next_btn,
        repair_position=repair_position,
        repair_select_all_btn=repair_select_all_btn,
        repair_selection=repair_selection,
        repair_editor=repair_editor,
        repair_result=repair_result,
        detect_red_btn=detect_red_btn,
        detect_yellow_btn=detect_yellow_btn,
        detect_both_btn=detect_both_btn,
        detect_selected_btn=detect_selected_btn,
        detect_saturation=detect_saturation,
        detect_value=detect_value,
        detect_min_area=detect_min_area,
        detect_padding=detect_padding,
        detect_merge=detect_merge,
        repair_tool=repair_tool,
        repair_brush_size=repair_brush_size,
        repair_alpha=repair_alpha,
        clear_mask_btn=clear_mask_btn,
        repair_current_btn=repair_current_btn,
        repair_selected_btn=repair_selected_btn,
        send_repaired_btn=send_repaired_btn,
        repair_status=repair_status,
    )


__all__ = ["build_repair_tab"]
