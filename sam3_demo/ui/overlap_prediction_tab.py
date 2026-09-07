"""Read-only UI for persisted O3 layer-overlap predictions."""

import gradio as gr

from .refs import ComponentRefs


def build_overlap_prediction_tab(
    *,
    authoring_builder=None,
    authoring_kwargs=None,
):
    with gr.TabItem("叠层预测", id="tab_overlap_prediction") as tab_item:
        gr.Markdown(
            "### 叠层预测\n"
            "选择已制作的主版图（如 O3、X3、C3）和当前工艺层，查看下一层位置与两层交叠区域。",
            elem_classes="stitch-intro",
        )
        with gr.Row(elem_classes="stitch-main-row"):
            with gr.Column(
                scale=1,
                min_width=320,
                elem_classes="stitch-control-column",
            ):
                with gr.Group(elem_classes="stitch-card"):
                    gr.Markdown("#### 选择版图")
                    layout_selector = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="主版图",
                        info="数据来自后端叠层版图库。",
                    )
                    refresh_btn = gr.Button("刷新版图库", variant="secondary", size="sm")
                with gr.Group(elem_classes="stitch-card"):
                    gr.Markdown("#### 选择当前层")
                    layer_selector = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="当前层",
                        info="按制作时保存的顺序，自动选择其下一层。",
                    )
                    gr.Markdown(
                        "🔵 当前层　🟣 下一层　🟡 交叠区域",
                        elem_classes="stitch-canvas-help",
                    )
                report = gr.Markdown(
                    "正在读取叠层版图库……",
                    elem_classes="stitch-status-card",
                )

            with gr.Column(
                scale=2,
                min_width=520,
                elem_classes="stitch-preview-column",
            ):
                with gr.Group(elem_classes="stitch-result-card"):
                    gr.Markdown("#### 下一层交叠位置")
                    preview = gr.Image(
                        type="pil",
                        label="叠层预测结果",
                        show_label=False,
                        interactive=False,
                        height=600,
                        elem_id="overlap_prediction_preview",
                    )

        authoring_refs = None
        if authoring_builder is not None:
            authoring_refs = authoring_builder(**(authoring_kwargs or {}))

    return ComponentRefs(
        tab=tab_item,
        layout_selector=layout_selector,
        layer_selector=layer_selector,
        refresh_btn=refresh_btn,
        preview=preview,
        report=report,
        authoring=authoring_refs,
    )


__all__ = ["build_overlap_prediction_tab"]
