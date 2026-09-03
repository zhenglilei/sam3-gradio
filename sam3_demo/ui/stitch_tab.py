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
            "### 周期拼接\n"
            "上传已去除界面 overlay 的相邻分块，自动对齐后可拖动微调，最后生成整图并写入智能图像分割工作区。",
            elem_classes="stitch-intro",
        )
        stitch_state = gr.State(value={})

        with gr.Row(elem_classes="stitch-main-row"):
            with gr.Column(scale=1, min_width=320, elem_classes="stitch-control-column"):
                with gr.Group(elem_classes="stitch-card"):
                    gr.Markdown("#### 1 · 上传分块")
                    stitch_files = gr.File(
                        label="选择同一组相邻图片",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    stitch_layout = gr.Dropdown(
                        choices=layout_choices,
                        value="horizontal",
                        label="排列方式",
                        info="横/纵按文件顺序；网格按上排从左到右，再按下排从左到右。",
                    )
                    stitch_remove_black_border = gr.Checkbox(
                        value=True,
                        label="加载时自动去黑边",
                        info="仅裁除与图片四周连续相连的近黑边；不会删除画面内部黑线。修改后请重新加载。",
                    )
                    stitch_load_btn = gr.Button("加载到拼接画布", variant="secondary")

                with gr.Group(elem_classes="stitch-card"):
                    gr.Markdown("#### 2 · 自动对齐与微调")
                    stitch_align_btn = gr.Button("自动对齐", variant="primary")
                    with gr.Row():
                        stitch_dx = gr.Number(value=0, precision=0, label="选中块 x（px）")
                        stitch_dy = gr.Number(value=0, precision=0, label="选中块 y（px）")
                        stitch_rotation = gr.Number(
                            value=0,
                            minimum=-180,
                            maximum=180,
                            step=0.1,
                            precision=2,
                            label="选中块旋转（°）",
                        )
                    stitch_apply_xy_btn = gr.Button("应用位置与旋转", size="sm")
                    stitch_nudge_step = gr.Radio(
                        choices=[("1 px", 1), ("5 px", 5), ("10 px", 10)],
                        value=1,
                        label="方向键步长",
                    )
                    with gr.Row():
                        stitch_diff_mode = gr.Checkbox(value=False, label="差分闪边")
                        stitch_show_loupe = gr.Checkbox(value=True, label="光标放大镜")

                with gr.Group(elem_classes="stitch-card"):
                    gr.Markdown("#### 3 · 生成与交接")
                    with gr.Row():
                        stitch_blend = gr.Checkbox(value=True, label="接缝融合")
                        stitch_crop_periodic = gr.Checkbox(value=False, label="裁完整周期")
                    stitch_export_btn = gr.Button("生成拼接结果", variant="primary")
                    stitch_handoff_btn = gr.Button(
                        "导入到智能图像分割 · 上传与裁剪",
                        variant="secondary",
                        interactive=False,
                    )

                stitch_status = gr.Markdown(
                    "请先上传一组分块图。",
                    elem_classes="stitch-status-card",
                )
                stitch_handoff_status = gr.Markdown(
                    "",
                    elem_classes="stitch-handoff-status",
                )

            with gr.Column(scale=2, min_width=520, elem_classes="stitch-preview-column"):
                with gr.Group(elem_classes="stitch-canvas-card"):
                    gr.Markdown("#### 拼接画布")
                    gr.Markdown(
                        "单击选择分块；拖动或使用方向键调整位置；拖动选中框右上角圆形手柄可绕中心旋转。交互期间只在浏览器本地更新。",
                        elem_classes="stitch-canvas-help",
                    )
                    if StitchPreviewCanvas is not None:
                        stitch_canvas = StitchPreviewCanvas(
                            value=empty_canvas,
                            label="分块预览",
                            show_label=False,
                            height=500,
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
                        label="下载拼接 PNG",
                        interactive=False,
                        visible=False,
                    )

    return ComponentRefs(
        stitch_state=stitch_state,
        stitch_files=stitch_files,
        stitch_layout=stitch_layout,
        stitch_remove_black_border=stitch_remove_black_border,
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
