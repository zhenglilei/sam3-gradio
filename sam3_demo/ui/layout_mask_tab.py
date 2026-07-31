"""Layout screenshot mask tab component construction."""

import gradio as gr

from .refs import ComponentRefs


def build_layout_mask_tab(
    *,
    LayoutRegionAnnotator,
    _layout_region_annotator_import_error,
    _layout_region_editor_empty,
    _LAYOUT_MASK_MORPH_LIMIT_PX,
):
    with gr.TabItem("版图截图转掩码", id="tab_layout_mask"):
        gr.Markdown("### 版图截图转二值 mask")
        gr.Markdown("binary mask 是唯一权威数据；contour 仅用于预览和导出。左右两侧预览使用相同高度。")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 上传版图截图")
                layout_input = gr.Image(type="numpy", label="上传版图截图", show_label=False, sources=["upload", "clipboard"], height=430)
                layout_threshold = gr.Slider(minimum=0, maximum=255, value=12, step=1, label="threshold（色彩/饱和度阈值）")
                layout_invert = gr.Checkbox(value=False, label="invert（反转前景/背景）")
                with gr.Row():
                    layout_open_kernel = gr.Slider(minimum=0, maximum=31, value=0, step=1, label="open kernel")
                    layout_close_kernel = gr.Slider(minimum=0, maximum=31, value=0, step=1, label="close kernel")
                layout_morph_pixels = gr.Slider(
                    minimum=-_LAYOUT_MASK_MORPH_LIMIT_PX,
                    maximum=_LAYOUT_MASK_MORPH_LIMIT_PX,
                    value=0,
                    step=1,
                    label="膨胀/腐蚀像素（正数膨胀，负数腐蚀）",
                )
                layout_min_area = gr.Number(value=0, precision=0, label="min component area")
                layout_region_mode = gr.Radio(
                    choices=[("全部区域", "all"), ("最大连通区域", "largest")],
                    value="all",
                    label="区域模式",
                    elem_classes="mode-radio",
                )
                run_layout_mask_btn = gr.Button("生成并保存当前版图 mask", variant="primary")
                save_layout_mask_btn = gr.Button("保存为当前版图 mask", variant="secondary")
                clear_layout_mask_btn = gr.Button("清除当前版图", variant="secondary")
                layout_info = gr.Textbox(label="处理信息", lines=8, interactive=False)
                with gr.Row():
                    layout_mask_file = gr.File(label="下载 mask PNG", interactive=False)
                    layout_contour_file = gr.File(label="下载 contour JSON", interactive=False)
            with gr.Column(scale=1):
                layout_source_preview = gr.Image(type="pil", label="原图预览", show_label=False, visible=False)
                gr.Markdown("#### mask 与 contour 预览")
                with gr.Row(elem_classes="layout-preview-pager"):
                    with gr.Column(elem_classes="layout-preview-page"):
                        layout_mask_preview = gr.Image(type="pil", label="binary mask 预览", show_label=False, height=430)
                        gr.Markdown("**binary mask 预览**")
                    with gr.Column(elem_classes="layout-preview-page"):
                        layout_overlay_preview = gr.Image(type="pil", label="contour overlay", show_label=False, height=430)
                        gr.Markdown("**contour overlay**")
                gr.Markdown("左右滑动或拖动下方滚动条切换预览。")
                with gr.Accordion("AI Mask \u4fee\u590d\u52a9\u624b", open=False):
                    gr.Markdown(
                        "VLM only selects backend-generated candidates. "
                        "Drafts stay in memory until you apply parameters and use the existing generate button."
                    )
                    layout_agent_consent = gr.Checkbox(
                        value=False,
                        label="I confirm sending this image thumbnail and candidate sheet to Qwen3.5-122B",
                    )
                    layout_agent_profile = gr.Radio(
                        choices=["Auto", "ACT", "GE1", "GE2"],
                        value="Auto",
                        label="Profile",
                    )
                    layout_agent_action_auto = gr.State("\u81ea\u52a8\u5206\u6790")
                    layout_agent_action_fill = gr.State("\u586b\u8865\u51f9\u5751")
                    layout_agent_action_bridge = gr.State("\u51cf\u5c11\u7c98\u8fde")
                    layout_agent_action_thicken = gr.State("\u6574\u4f53\u52a0\u7c97")
                    layout_agent_action_holes = gr.State("\u4fdd\u7559\u5b54\u6d1e")
                    with gr.Row():
                        layout_agent_auto_btn = gr.Button("\u81ea\u52a8\u5206\u6790")
                        layout_agent_fill_btn = gr.Button("\u586b\u8865\u51f9\u5751")
                        layout_agent_bridge_btn = gr.Button("\u51cf\u5c11\u7c98\u8fde")
                    with gr.Row():
                        layout_agent_thicken_btn = gr.Button("\u6574\u4f53\u52a0\u7c97")
                        layout_agent_holes_btn = gr.Button("\u4fdd\u7559\u5b54\u6d1e")
                        layout_agent_undo_btn = gr.Button("\u64a4\u56de")
                        layout_agent_reset_btn = gr.Button("\u91cd\u7f6e")
                    layout_agent_chatbot = gr.Chatbot(
                        label="\u591a\u8f6e\u5bf9\u8bdd",
                        height=280,
                    )
                    with gr.Row():
                        layout_agent_prompt = gr.Textbox(
                            label="\u4fee\u590d\u53cd\u9988",
                            placeholder="\u4f8b\u5982\uff1a\u518d\u586b\u4e00\u70b9\uff0c\u4f46\u4e0d\u8981\u53d8\u7c97",
                            max_lines=3,
                            scale=5,
                        )
                        layout_agent_send_btn = gr.Button("\u53d1\u9001", variant="primary", scale=1)
                    with gr.Row():
                        layout_agent_draft_preview = gr.Image(
                            type="pil",
                            label="Agent Draft delta: red added, blue removed, yellow risk",
                            height=330,
                        )
                        layout_agent_candidate_preview = gr.Image(
                            type="pil",
                            label="Candidate comparison",
                            height=330,
                        )
                    layout_agent_diff = gr.Markdown("No active Agent Draft.")
                    layout_agent_status = gr.Textbox(
                        label="Model / tokens / cost / manual review",
                        interactive=False,
                        lines=3,
                    )
                    layout_agent_apply_btn = gr.Button(
                        "\u5e94\u7528\u63a8\u8350\u53c2\u6570",
                        variant="primary",
                        interactive=False,
                    )
                gr.Markdown("### Label Annotation Layer")
                gr.Markdown("黄色表示未保存 Draft；绿色表示已保存 Label。一个套索对应一个独立 Label；Label 可选留空并自动按序号命名。")
                if LayoutRegionAnnotator is not None:
                    layout_region_annotator = LayoutRegionAnnotator(
                        value=_layout_region_editor_empty(),
                        label="版图 Label 套索标注器",
                        show_label=False,
                        height=520,
                        elem_id="layout_region_annotator",
                    )
                else:
                    gr.Markdown(f"Label 套索组件不可用。错误：{_layout_region_annotator_import_error}")
                    layout_region_annotator = gr.JSON(
                        value=_layout_region_editor_empty(),
                        label="layout Region payload",
                        visible=False,
                    )
                layout_region_label = gr.Textbox(
                    label="Label（可选）",
                    value="",
                    placeholder="留空将自动命名为 Label 1、Label 2……",
                    max_lines=1,
                )
                save_layout_region_btn = gr.Button("保存当前 Draft Label", variant="primary", interactive=False)
                layout_region_selector = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="活动 Label",
                    interactive=True,
                )
                delete_layout_region_btn = gr.Button("软删除选中 Label", variant="secondary", interactive=False)
                layout_region_status = gr.Textbox(label="Label 状态", lines=4, interactive=False)
                export_layout_regions_btn = gr.Button(
                    "导出当前 Label 标注",
                    variant="secondary",
                )
                layout_region_export_file = gr.File(
                    label="下载 Label 标注包",
                    interactive=False,
                )
    return ComponentRefs(
        layout_input=layout_input,
        layout_threshold=layout_threshold,
        layout_invert=layout_invert,
        layout_open_kernel=layout_open_kernel,
        layout_close_kernel=layout_close_kernel,
        layout_morph_pixels=layout_morph_pixels,
        layout_min_area=layout_min_area,
        layout_region_mode=layout_region_mode,
        run_layout_mask_btn=run_layout_mask_btn,
        save_layout_mask_btn=save_layout_mask_btn,
        clear_layout_mask_btn=clear_layout_mask_btn,
        layout_info=layout_info,
        layout_mask_file=layout_mask_file,
        layout_contour_file=layout_contour_file,
        layout_source_preview=layout_source_preview,
        layout_mask_preview=layout_mask_preview,
        layout_agent_consent=layout_agent_consent,
        layout_agent_profile=layout_agent_profile,
        layout_agent_action_auto=layout_agent_action_auto,
        layout_agent_action_fill=layout_agent_action_fill,
        layout_agent_action_bridge=layout_agent_action_bridge,
        layout_agent_action_thicken=layout_agent_action_thicken,
        layout_agent_action_holes=layout_agent_action_holes,
        layout_agent_auto_btn=layout_agent_auto_btn,
        layout_agent_fill_btn=layout_agent_fill_btn,
        layout_agent_bridge_btn=layout_agent_bridge_btn,
        layout_agent_thicken_btn=layout_agent_thicken_btn,
        layout_agent_holes_btn=layout_agent_holes_btn,
        layout_agent_undo_btn=layout_agent_undo_btn,
        layout_agent_reset_btn=layout_agent_reset_btn,
        layout_agent_chatbot=layout_agent_chatbot,
        layout_agent_prompt=layout_agent_prompt,
        layout_agent_send_btn=layout_agent_send_btn,
        layout_agent_draft_preview=layout_agent_draft_preview,
        layout_agent_candidate_preview=layout_agent_candidate_preview,
        layout_agent_diff=layout_agent_diff,
        layout_agent_status=layout_agent_status,
        layout_agent_apply_btn=layout_agent_apply_btn,
        layout_overlay_preview=layout_overlay_preview,
        layout_region_annotator=layout_region_annotator,
        layout_region_label=layout_region_label,
        save_layout_region_btn=save_layout_region_btn,
        layout_region_selector=layout_region_selector,
        delete_layout_region_btn=delete_layout_region_btn,
        layout_region_status=layout_region_status,
        export_layout_regions_btn=export_layout_regions_btn,
        layout_region_export_file=layout_region_export_file,
    )
