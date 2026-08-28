"""Stitch tiles then align layout overlay on the mosaic."""

import gradio as gr

from .refs import ComponentRefs


def build_stitch_tab(
    *,
    StitchPreviewCanvas,
    _stitch_canvas_import_error,
    LayoutTransformEditor,
    _layout_editor_import_error,
    empty_canvas,
    empty_layout_editor,
    layout_choices,
):
    with gr.TabItem("分块拼接与版图对齐", id="tab_stitch_layout"):
        gr.Markdown(
            "### 分块拼接与版图对齐\n"
            "先自动对齐分块并手动微调，再把拼好的整图当作底图，用半透明版图层对齐。"
            "分块应已去掉红黄 UI overlay。"
        )
        stitch_state = gr.State(value={})
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("#### 1 拼接微调")
                stitch_files = gr.File(
                    label="上传分块图（一组）",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                stitch_layout = gr.Dropdown(
                    choices=layout_choices,
                    value="horizontal",
                    label="布局",
                )
                with gr.Row():
                    stitch_load_btn = gr.Button("加载分组", variant="secondary")
                    stitch_align_btn = gr.Button("自动对齐当前组", variant="primary")
                stitch_nudge_step = gr.Radio(
                    choices=[("1 px", 1), ("5 px", 5), ("10 px", 10)],
                    value=1,
                    label="方向键步长",
                )
                with gr.Row():
                    stitch_dx = gr.Number(value=0, precision=0, label="选中块 x")
                    stitch_dy = gr.Number(value=0, precision=0, label="选中块 y")
                stitch_apply_xy_btn = gr.Button("应用 x/y", size="sm")
                stitch_diff_mode = gr.Checkbox(value=False, label="差分闪边")
                stitch_show_loupe = gr.Checkbox(value=True, label="光标放大镜")
                stitch_blend = gr.Checkbox(value=True, label="接缝融合")
                stitch_crop_periodic = gr.Checkbox(value=False, label="裁完整周期")
                stitch_export_btn = gr.Button("生成拼接结果", variant="primary")
                stitch_status = gr.Markdown("请上传一组分块图。")
                stitch_mosaic_preview = gr.Image(
                    type="pil",
                    label="拼接结果",
                    interactive=False,
                    height=180,
                )
                stitch_mosaic_file = gr.File(label="下载拼接 PNG", interactive=False)
            with gr.Column(scale=3):
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
        with gr.Group() as stitch_step2_group:
            gr.Markdown("#### 2 版图对齐（需先有拼接结果）")
            with gr.Row():
                stitch_use_layout_btn = gr.Button("使用当前已保存版图 mask", variant="secondary")
                stitch_handoff_btn = gr.Button("写入智能图像分割工作区", variant="primary")
            stitch_step2_status = gr.Markdown("生成拼接结果后即可在底图上对齐版图。")
            if LayoutTransformEditor is not None:
                stitch_layout_editor = LayoutTransformEditor(
                    value=empty_layout_editor,
                    label="拼接底图上的版图",
                    show_label=False,
                    height=520,
                    elem_id="stitch_layout_transform_editor",
                )
            else:
                gr.Markdown(f"版图编辑器不可用：{_layout_editor_import_error}")
                stitch_layout_editor = gr.JSON(value=empty_layout_editor, visible=False)
        stitch_handoff_status = gr.Markdown("")
    return ComponentRefs(
        stitch_state=stitch_state,
        stitch_files=stitch_files,
        stitch_layout=stitch_layout,
        stitch_load_btn=stitch_load_btn,
        stitch_align_btn=stitch_align_btn,
        stitch_nudge_step=stitch_nudge_step,
        stitch_dx=stitch_dx,
        stitch_dy=stitch_dy,
        stitch_apply_xy_btn=stitch_apply_xy_btn,
        stitch_diff_mode=stitch_diff_mode,
        stitch_show_loupe=stitch_show_loupe,
        stitch_blend=stitch_blend,
        stitch_crop_periodic=stitch_crop_periodic,
        stitch_export_btn=stitch_export_btn,
        stitch_status=stitch_status,
        stitch_mosaic_preview=stitch_mosaic_preview,
        stitch_mosaic_file=stitch_mosaic_file,
        stitch_canvas=stitch_canvas,
        stitch_step2_group=stitch_step2_group,
        stitch_use_layout_btn=stitch_use_layout_btn,
        stitch_handoff_btn=stitch_handoff_btn,
        stitch_step2_status=stitch_step2_status,
        stitch_layout_editor=stitch_layout_editor,
        stitch_handoff_status=stitch_handoff_status,
    )
