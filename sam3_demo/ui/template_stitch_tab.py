"""Gradio construction for hole-pattern template stitching."""

import gradio as gr

from .refs import ComponentRefs


def build_template_stitch_tab():
    with gr.TabItem("模板拼接", id="tab_template_stitch", visible=False):
        gr.Markdown(
            "### 模板拼接\n"
            "面向周期孔阵分块：检测孔心与周期，2×2 自动判定角点排列，其他网格按文件顺序配准拼接。",
            elem_classes="stitch-intro",
        )
        template_stitch_state = gr.State(value={})

        with gr.Row(elem_classes="stitch-main-row"):
            with gr.Column(scale=1, min_width=320, elem_classes="stitch-control-column"):
                with gr.Group(elem_classes="stitch-card"):
                    gr.Markdown("#### 1 · 上传同组分块")
                    template_stitch_files = gr.File(
                        label="选择一个网格组的图片",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    with gr.Row():
                        template_stitch_rows = gr.Number(
                            value=2,
                            minimum=1,
                            maximum=6,
                            precision=0,
                            label="行数",
                        )
                        template_stitch_cols = gr.Number(
                            value=2,
                            minimum=1,
                            maximum=6,
                            precision=0,
                            label="列数",
                        )
                    gr.Markdown(
                        "2×2 会自动判断 TL/TR/BL/BR；其他网格按文件名顺序先行后列。",
                        elem_classes="stitch-canvas-help",
                    )
                    template_stitch_run_btn = gr.Button("运行模板拼接", variant="primary")

                with gr.Group(elem_classes="stitch-card"):
                    gr.Markdown("#### 2 · 导出与交接")
                    template_stitch_handoff_btn = gr.Button(
                        "导入到智能图像分割 · 上传与裁剪",
                        variant="secondary",
                        interactive=False,
                    )
                    template_stitch_file = gr.File(
                        label="下载模板拼接 PNG",
                        interactive=False,
                        visible=False,
                    )

                template_stitch_status = gr.Markdown(
                    "请上传与网格数量一致的一组图片。",
                    elem_classes="stitch-status-card",
                )
                template_stitch_handoff_status = gr.Markdown("")

            with gr.Column(scale=2, min_width=520, elem_classes="stitch-preview-column"):
                with gr.Group(elem_classes="stitch-result-card"):
                    gr.Markdown("#### 模板拼接结果")
                    template_stitch_preview = gr.Image(
                        type="pil",
                        label="模板拼接预览",
                        show_label=False,
                        interactive=False,
                        height=520,
                    )
                with gr.Accordion("配准诊断", open=False):
                    template_stitch_meta = gr.JSON(label="孔周期、布局方法与分块位置")

    return ComponentRefs(
        template_stitch_state=template_stitch_state,
        template_stitch_files=template_stitch_files,
        template_stitch_rows=template_stitch_rows,
        template_stitch_cols=template_stitch_cols,
        template_stitch_run_btn=template_stitch_run_btn,
        template_stitch_preview=template_stitch_preview,
        template_stitch_meta=template_stitch_meta,
        template_stitch_file=template_stitch_file,
        template_stitch_status=template_stitch_status,
        template_stitch_handoff_btn=template_stitch_handoff_btn,
        template_stitch_handoff_status=template_stitch_handoff_status,
    )
