"""Optional manual authoring panel for persisted overlap layouts."""

from __future__ import annotations

from typing import Any

import gradio as gr

from sam3_demo import overlap_authoring_callbacks as callbacks
from sam3_demo.overlap_view_callbacks import refresh_overlap_view

from .refs import ComponentRefs


_EMPTY_EDITOR = {
    "enabled": False,
    "status": "请先上传整体版图并添加层标注。",
}


def build_overlap_authoring_panel(*, LayoutTransformEditor, import_error=None):
    """Build the optional editor without changing the read-only viewer."""
    with gr.Accordion(
        "版图制作（内部工具）",
        open=False,
        elem_id="overlap_authoring_panel",
    ):
        draft_state = gr.State(None)
        gr.Markdown(
            "输入主版图名称，上传一张主版图图片；每份 JSON 标注就是一个子层。"
            "JSON 会直接显示为 mask 并叠加到主版图，列表顺序就是叠层顺序。"
            "拖动标注可平移，滚轮缩放，拖动右上圆点旋转。",
            elem_classes="stitch-canvas-help",
        )
        with gr.Group(elem_classes="stitch-card"):
            gr.Markdown("#### 已保存版图")
            saved_layout_selector = gr.Dropdown(
                choices=[],
                value=None,
                label="版图库",
                info="制作数据独立保存；关闭本模块不会影响上方只读展示。",
            )
            with gr.Row():
                refresh_saved_btn = gr.Button(
                    "刷新", variant="secondary", size="sm"
                )
                load_saved_btn = gr.Button(
                    "加载编辑", variant="secondary", size="sm"
                )
                new_draft_btn = gr.Button(
                    "新建", variant="secondary", size="sm"
                )
                delete_saved_btn = gr.Button(
                    "删除当前编辑版图", variant="stop", size="sm"
                )

        layout_name = gr.Textbox(
            label="版图名称",
            placeholder="例如：O3-A1 标准叠层",
            max_lines=1,
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300):
                base_image = gr.Image(
                    type="pil",
                    label="主版图图片",
                    sources=["upload", "clipboard"],
                    interactive=True,
                    height=320,
                )
            with gr.Column(scale=1, min_width=300):
                layer_annotation_json = gr.File(
                    label="新增子层标注（LabelMe / 单图 COCO JSON）",
                    file_types=[".json"],
                    type="filepath",
                )
                layer_mask_preview = gr.Image(
                    type="pil",
                    label="该层标注 mask 预览",
                    interactive=False,
                    height=220,
                )
                layer_type = gr.Dropdown(
                    choices=["BSM", "ACT", "GE1", "GE2"],
                    value="ACT",
                    label="子层分类",
                    info="可直接输入当前主版图实际使用的自定义层分类。",
                    allow_custom_value=True,
                )
                add_layer_btn = gr.Button(
                    "添加这一层", variant="primary"
                )

        with gr.Group(elem_classes="stitch-card"):
            gr.Markdown("#### 叠层顺序与当前编辑层")
            layer_selector = gr.Dropdown(
                choices=[],
                value=None,
                label="当前层（顺序即叠层顺序）",
            )
            remove_layer_btn = gr.Button(
                "删除当前层", variant="secondary", size="sm"
            )

        if LayoutTransformEditor is not None:
            editor = LayoutTransformEditor(
                value=dict(_EMPTY_EDITOR),
                label="叠层位置编辑器",
                show_label=False,
                height=620,
                elem_id="overlap_authoring_editor",
            )
        else:
            gr.Markdown(
                f"叠层拖拽编辑器不可用：{import_error}",
                elem_classes="stitch-status-card",
            )
            editor = gr.JSON(value=dict(_EMPTY_EDITOR), visible=False)

        save_btn = gr.Button("保存版图", variant="primary")
        status = gr.Markdown(
            "制作模块正在初始化……",
            elem_classes="stitch-status-card",
        )

    return ComponentRefs(
        draft_state=draft_state,
        saved_layout_selector=saved_layout_selector,
        refresh_saved_btn=refresh_saved_btn,
        load_saved_btn=load_saved_btn,
        new_draft_btn=new_draft_btn,
        delete_saved_btn=delete_saved_btn,
        layout_name=layout_name,
        base_image=base_image,
        layer_annotation_json=layer_annotation_json,
        layer_mask_preview=layer_mask_preview,
        layer_type=layer_type,
        add_layer_btn=add_layer_btn,
        layer_selector=layer_selector,
        remove_layer_btn=remove_layer_btn,
        editor=editor,
        save_btn=save_btn,
        status=status,
    )


def bind_overlap_authoring_events(
    *,
    demo: gr.Blocks,
    refs: Any,
    viewer_refs: Any,
) -> None:
    """Bind authoring separately from all PVS/session callbacks."""
    event_kwargs = {
        "queue": True,
        "show_progress": "minimal",
        "concurrency_limit": 1,
        "concurrency_id": "overlap-authoring-write",
        "api_visibility": "private",
    }
    view_outputs = [
        refs.draft_state,
        refs.layout_name,
        refs.base_image,
        refs.layer_selector,
        refs.editor,
        refs.status,
    ]
    persist_outputs = [*view_outputs, refs.saved_layout_selector]

    init_event = demo.load(
        fn=callbacks.new_authoring_draft,
        inputs=None,
        outputs=view_outputs,
        **event_kwargs,
    )
    init_event.success(
        fn=callbacks.refresh_authoring_layouts,
        inputs=[refs.saved_layout_selector],
        outputs=[refs.saved_layout_selector, refs.status],
        **event_kwargs,
    )
    refs.new_draft_btn.click(
        fn=callbacks.new_authoring_draft,
        inputs=None,
        outputs=view_outputs,
        **event_kwargs,
    )
    refs.refresh_saved_btn.click(
        fn=callbacks.refresh_authoring_layouts,
        inputs=[refs.saved_layout_selector],
        outputs=[refs.saved_layout_selector, refs.status],
        **event_kwargs,
    )
    refs.load_saved_btn.click(
        fn=callbacks.load_authoring_layout,
        inputs=[refs.saved_layout_selector, refs.draft_state],
        outputs=persist_outputs,
        **event_kwargs,
    )
    refs.layout_name.blur(
        fn=callbacks.set_authoring_name,
        inputs=[refs.layout_name, refs.draft_state],
        outputs=view_outputs,
        **event_kwargs,
    )
    refs.base_image.input(
        fn=callbacks.set_authoring_base_image,
        inputs=[refs.base_image, refs.draft_state],
        outputs=view_outputs,
        **event_kwargs,
    )
    refs.layer_annotation_json.change(
        fn=callbacks.inspect_authoring_annotation,
        inputs=[refs.layer_annotation_json],
        outputs=[refs.layer_mask_preview, refs.status],
        **event_kwargs,
    )
    refs.add_layer_btn.click(
        fn=callbacks.add_authoring_layer,
        inputs=[
            refs.layer_type,
            refs.layer_annotation_json,
            refs.draft_state,
        ],
        outputs=view_outputs,
        **event_kwargs,
    )
    refs.layer_selector.input(
        fn=callbacks.select_authoring_layer,
        inputs=[refs.layer_selector, refs.draft_state],
        outputs=view_outputs,
        **event_kwargs,
    )
    refs.remove_layer_btn.click(
        fn=callbacks.remove_authoring_layer,
        inputs=[refs.layer_selector, refs.draft_state],
        outputs=view_outputs,
        **event_kwargs,
    )
    refs.editor.change(
        fn=callbacks.apply_authoring_editor,
        inputs=[refs.editor, refs.draft_state],
        outputs=[refs.draft_state, refs.editor, refs.status],
        queue=True,
        show_progress="hidden",
        concurrency_limit=1,
        concurrency_id="overlap-authoring-write",
        api_visibility="private",
    )
    save_event = refs.save_btn.click(
        fn=callbacks.save_authoring_layout,
        inputs=[refs.draft_state, refs.layout_name, refs.editor],
        outputs=persist_outputs,
        **event_kwargs,
    )
    delete_event = refs.delete_saved_btn.click(
        fn=callbacks.delete_authoring_layout,
        inputs=[refs.draft_state],
        outputs=persist_outputs,
        **event_kwargs,
    )
    for event in (save_event, delete_event):
        event.then(
            fn=refresh_overlap_view,
            inputs=[refs.saved_layout_selector],
            outputs=[
                viewer_refs.layout_selector,
                viewer_refs.layer_selector,
                viewer_refs.preview,
                viewer_refs.report,
            ],
            queue=True,
            show_progress="minimal",
            concurrency_limit=2,
            concurrency_id="overlap-read-view",
            api_visibility="private",
        )


__all__ = [
    "bind_overlap_authoring_events",
    "build_overlap_authoring_panel",
]
