"""Gradio construction for the standalone tile-stitching workflow."""

import gradio as gr

from .refs import ComponentRefs


def build_stitch_tab(
    *,
    StitchPreviewCanvas,
    _stitch_canvas_import_error,
    ImageGestureOverlay,
    _image_gesture_overlay_import_error,
    empty_canvas,
    empty_crop_overlay,
    layout_choices,
):
    with gr.TabItem("周期拼接", id="tab_stitch"):
        gr.Markdown(
            "### 周期拼接",
            elem_classes="stitch-intro",
        )
        stitch_state = gr.State(value={})

        with gr.Row(elem_classes="stitch-main-row"):
            with gr.Column(scale=1, min_width=280, elem_classes="stitch-control-column"):
                with gr.Column(elem_classes="stitch-sidebar"):
                    with gr.Tabs(elem_classes="stitch-control-tabs"):
                        with gr.TabItem("图片", id="stitch_images"):
                            gr.Markdown("分块图片", elem_classes="stitch-group-label")
                            stitch_files = gr.File(
                                label="分块图片",
                                show_label=False,
                                file_count="multiple",
                                file_types=["image"],
                                type="filepath",
                                height=150,
                                elem_classes="stitch-upload",
                            )
                            with gr.Accordion("小图标注队列", open=False):
                                annotated_files = gr.File(label="图片、COCO / LabelMe JSON 或标注 ZIP",
                                    file_count="multiple", type="filepath",
                                    file_types=["image", ".json", ".zip"], height=100)
                                annotated_import_btn = gr.Button("导入标注文件")
                                annotated_gallery = gr.Gallery(label="已保存的小图", columns=2, height=145,
                                    object_fit="contain", interactive=False)
                                annotated_selection = gr.Dropdown(label="当前队列小图", choices=[])
                                with gr.Row():
                                    annotated_up = gr.Button("↑", min_width=40, size="sm")
                                    annotated_down = gr.Button("↓", min_width=40, size="sm")
                                    annotated_remove = gr.Button("移除", min_width=60, size="sm")
                                annotated_load = gr.Button("加载标注队列", variant="primary")
                                annotated_status = gr.Markdown("")
                            stitch_layout = gr.Dropdown(
                                choices=layout_choices,
                                value="horizontal",
                                label="排列方式",
                            )
                            gr.Markdown("边缘裁剪（px）", elem_classes="stitch-group-label")
                            with gr.Row(elem_classes="stitch-crop-fields"):
                                stitch_crop_top = gr.Number(value=0, minimum=0, precision=0, label="上")
                                stitch_crop_bottom = gr.Number(value=0, minimum=0, precision=0, label="下")
                                stitch_crop_left = gr.Number(value=0, minimum=0, precision=0, label="左")
                                stitch_crop_right = gr.Number(value=0, minimum=0, precision=0, label="右")
                            stitch_remove_black_border = gr.Checkbox(
                                value=True,
                                label="加载时自动去黑边",
                            )
                            stitch_load_btn = gr.Button(
                                "加载到拼接画布",
                                variant="primary",
                                elem_classes="stitch-action-btn",
                            )

                        with gr.TabItem("对齐", id="stitch_alignment"):
                            stitch_align_btn = gr.Button(
                                "自动对齐",
                                variant="primary",
                                elem_classes="stitch-action-btn",
                            )
                            gr.Markdown("选中块位置与旋转", elem_classes="stitch-group-label")
                            with gr.Row(elem_classes="stitch-position-fields"):
                                stitch_dx = gr.Number(value=0, precision=0, label="x（px）")
                                stitch_dy = gr.Number(value=0, precision=0, label="y（px）")
                                stitch_rotation = gr.Number(
                                    value=0,
                                    minimum=-180,
                                    maximum=180,
                                    step=0.1,
                                    precision=2,
                                    label="旋转（°）",
                                )
                            stitch_apply_xy_btn = gr.Button(
                                "应用位置与旋转",
                                size="sm",
                                elem_classes="stitch-action-btn stitch-action-btn-quiet",
                            )
                            annotated_visible = gr.Checkbox(value=True, label="显示标注")
                            annotated_alpha = gr.Slider(0, 1, value=.35, step=.05, label="标注透明度")
                            gr.Markdown("画布交互", elem_classes="stitch-group-label")
                            stitch_nudge_step = gr.Radio(
                                choices=[("1 px", 1), ("5 px", 5), ("10 px", 10)],
                                value=1,
                                label="方向键步长",
                                elem_classes="stitch-segmented",
                            )
                            with gr.Row(elem_classes="stitch-toggle-row"):
                                stitch_diff_mode = gr.Checkbox(value=False, label="差分闪边")
                                stitch_show_loupe = gr.Checkbox(value=True, label="光标放大镜")

                        with gr.TabItem("导出", id="stitch_export"):
                            gr.Markdown("输出设置", elem_classes="stitch-group-label")
                            with gr.Row(elem_classes="stitch-toggle-row"):
                                stitch_blend = gr.Checkbox(value=True, label="接缝融合")
                                stitch_crop_periodic = gr.Checkbox(value=False, label="裁完整周期")
                            stitch_export_btn = gr.Button(
                                "生成拼接结果",
                                variant="primary",
                                elem_classes="stitch-action-btn",
                            )
                            stitch_handoff_confirm = gr.Checkbox(value=False,
                                label="确认替换当前工作区图片与实例")
                            stitch_handoff_btn = gr.Button(
                                "导入到智能图像分割",
                                variant="secondary",
                                interactive=False,
                                elem_classes="stitch-action-btn stitch-action-btn-quiet",
                            )

                stitch_status = gr.Markdown(
                    "请先上传一组分块图。",
                    elem_classes="stitch-status-card",
                )
                stitch_handoff_status = gr.Markdown(
                    "",
                    elem_classes="stitch-handoff-status",
                )

            with gr.Column(scale=3, min_width=0, elem_classes="stitch-preview-column"):
                with gr.Group(elem_classes="stitch-canvas-card"):
                    gr.Markdown("#### 拼接画布")
                    if StitchPreviewCanvas is not None:
                        stitch_canvas = StitchPreviewCanvas(
                            value=empty_canvas,
                            label="分块预览",
                            show_label=False,
                            height=560,
                            elem_id="stitch_preview_canvas",
                        )
                    else:
                        gr.Markdown(f"分块画布组件不可用：{_stitch_canvas_import_error}")
                        stitch_canvas = gr.JSON(value=empty_canvas, label="stitch canvas payload")

                with gr.Group(elem_classes="stitch-result-card"):
                    gr.Markdown("#### 导出预览")
                    stitch_mosaic_preview = gr.Image(
                        type="pil",
                        label="拼接结果",
                        interactive=False,
                        height=240,
                        visible=True,
                        elem_id="stitch_mosaic_preview",
                    )
                    if ImageGestureOverlay is not None:
                        stitch_mosaic_crop_overlay = ImageGestureOverlay(
                            value=empty_crop_overlay,
                            label="拼接结果矩形截图手势",
                            show_label=False,
                            target_elem_id="stitch_mosaic_preview",
                            height=1,
                            elem_classes="gesture-overlay-anchor",
                        )
                    else:
                        gr.Markdown(f"矩形截图组件不可用：{_image_gesture_overlay_import_error}")
                        stitch_mosaic_crop_overlay = gr.JSON(
                            value=empty_crop_overlay,
                            visible=False,
                        )
                    stitch_restore_full_btn = gr.Button(
                        "恢复完整拼接图",
                        size="sm",
                        interactive=False,
                    )
                    gr.Markdown(
                        "生成后可在预览上拖拽长方形截图；截图会成为下载和导入工作区的当前结果。",
                        elem_classes="stitch-canvas-help",
                    )
                    stitch_mosaic_file = gr.File(
                        label="下载拼接结果（PNG / 标注 ZIP）",
                        interactive=False,
                        visible=True,
                        elem_id="stitch_download",
                    )

    return ComponentRefs(
        annotated_files=annotated_files, annotated_import_btn=annotated_import_btn,
        annotated_gallery=annotated_gallery, annotated_selection=annotated_selection,
        annotated_up=annotated_up, annotated_down=annotated_down, annotated_remove=annotated_remove,
        annotated_load=annotated_load, annotated_status=annotated_status,
        annotated_visible=annotated_visible, annotated_alpha=annotated_alpha,
        stitch_handoff_confirm=stitch_handoff_confirm,
        stitch_state=stitch_state,
        stitch_files=stitch_files,
        stitch_layout=stitch_layout,
        stitch_remove_black_border=stitch_remove_black_border,
        stitch_crop_top=stitch_crop_top,
        stitch_crop_bottom=stitch_crop_bottom,
        stitch_crop_left=stitch_crop_left,
        stitch_crop_right=stitch_crop_right,
        stitch_load_btn=stitch_load_btn,
        stitch_align_btn=stitch_align_btn,
        stitch_nudge_step=stitch_nudge_step,
        stitch_dx=stitch_dx,
        stitch_dy=stitch_dy,
        stitch_rotation=stitch_rotation,
        stitch_apply_xy_btn=stitch_apply_xy_btn,
        stitch_diff_mode=stitch_diff_mode,
        stitch_show_loupe=stitch_show_loupe,
        stitch_blend=stitch_blend,
        stitch_crop_periodic=stitch_crop_periodic,
        stitch_export_btn=stitch_export_btn,
        stitch_status=stitch_status,
        stitch_mosaic_preview=stitch_mosaic_preview,
        stitch_mosaic_crop_overlay=stitch_mosaic_crop_overlay,
        stitch_restore_full_btn=stitch_restore_full_btn,
        stitch_mosaic_file=stitch_mosaic_file,
        stitch_canvas=stitch_canvas,
        stitch_handoff_btn=stitch_handoff_btn,
        stitch_handoff_status=stitch_handoff_status,
    )
