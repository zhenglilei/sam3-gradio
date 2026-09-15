"""Image segmentation tab component construction."""

import gradio as gr

from .refs import ComponentRefs


def build_image_tab(
    *,
    ImageGestureOverlay,
    _image_gesture_overlay_import_error,
    _source_gesture_payload,
    _new_source_image_state,
    _workspace_gesture_payload,
    LayoutTransformEditor,
    _layout_editor_import_error,
    _layout_editor_empty,
    _LAYOUT_PROMPT_SCOPE_FULL,
    coco_dataset_choices,
    default_coco_dataset,
    coco_eval_scope_overlap,
    coco_eval_scope_full,
):
    with gr.TabItem("智能图像分割", id="tab_image", render_children=True) as image_tab:
        batch_state = gr.State({"items": [], "active_id": None})
        with gr.Column(elem_id="el-workspace"):
            with gr.Column(elem_id="el-image-list"):
                with gr.Row(elem_id="el-library-row"):
                    batch_upload = gr.File(
                        label="添加图片", file_count="multiple", type="filepath",
                        file_types=["image"], height=120, scale=1, min_width=160,
                    )
                    batch_gallery = gr.Gallery(
                        label="图片列表", columns=6, height=104,
                        allow_preview=False, buttons=[], interactive=False,
                        object_fit="contain", show_label=False, scale=5, min_width=220,
                        elem_id="el-image-gallery",
                    )
                with gr.Accordion("图片选择与删除", open=False):
                    with gr.Row(elem_classes="el-selection-actions"):
                        batch_select_all_btn = gr.Button("全选", size="sm", min_width=80)
                        batch_select_none_btn = gr.Button("取消全选", size="sm", min_width=80)
                        batch_delete_current_btn = gr.Button("删除当前图片", size="sm", min_width=100)
                        batch_delete_selected_btn = gr.Button("删除选中图片", size="sm", min_width=100)
                    batch_selection = gr.CheckboxGroup(
                        choices=[], label="处理范围", elem_id="el-batch-selection",
                    )
            batch_status = gr.Markdown("", elem_id="el-batch-status")
            with gr.Row(elem_id="el-workspace-footer"):
                batch_prev_btn = gr.Button("上一张", size="sm", min_width=80)
                batch_save_btn = gr.Button("保存", size="sm", min_width=80)
                batch_next_btn = gr.Button("保存并下一张", variant="primary", size="sm", min_width=120)
                batch_run_btn = gr.Button("批量 PCS", size="sm", min_width=100)
                batch_retry_btn = gr.Button("重试失败", size="sm", min_width=100)
                batch_cancel_btn = gr.Button("取消批处理", size="sm", interactive=False, min_width=110)
                batch_stitch_btn = gr.Button("送往拼接", variant="primary", size="sm", min_width=100)
            with gr.Tabs(elem_id="el-workflow-tabs") as workflow_tabs:
                with gr.Tab("分割标注", id="segment", elem_classes="el-workflow-page"):
                    mode = gr.Radio(
                        choices=[("PCS 自动分割", "PCS Auto"), ("PVS 手动标注", "PVS Manual"), ("版图 Mask", "Layout Mask")],
                        value="PVS Manual",
                        label="分割方式", show_label=False, container=False,
                        elem_classes="mode-radio",
                    )
                    with gr.Row(elem_id="el-workspace-body"):
                        with gr.Column(scale=3, min_width=360, elem_id="el-workspace-center") as workspace_center:
                            with gr.Row(elem_classes="el-image-pair"):
                                with gr.Column(min_width=180):
                                    gr.Markdown("### 原始图像（点击进行交互）")
                                    image_upload = gr.Image(type="numpy", label="原始图像", show_label=False, interactive=False, elem_id="input_image", height=400)
                                    if ImageGestureOverlay is not None:
                                        workspace_gesture_overlay = ImageGestureOverlay(
                                            value=_workspace_gesture_payload({}, "PVS Manual", "bbox"),
                                            label="分割交互手势",
                                            show_label=False,
                                            target_elem_id="input_image",
                                            height=1,
                                            elem_classes="gesture-overlay-anchor",
                                        )
                                    else:
                                        workspace_gesture_overlay = gr.JSON(
                                            value=_workspace_gesture_payload({}, "PVS Manual", "bbox"),
                                            visible=False,
                                        )
                                with gr.Column(min_width=180):
                                    gr.Markdown("### 分割结果")
                                    result_image = gr.Image(type="numpy", label="分割结果", show_label=False, interactive=False, height=400, elem_classes="el-result-image")
                            with gr.Group(visible=False) as layout_transform_panel:
                                gr.Markdown("### 修改变形版图")
                                if LayoutTransformEditor is not None:
                                    layout_editor = LayoutTransformEditor(value=_layout_editor_empty(), label="\u7248\u56fe\u4ea4\u4e92\u7f16\u8f91\u5668", show_label=False, height=440, elem_id="layout_transform_editor")
                                else:
                                    gr.Markdown(f"版图 Canvas 编辑器组件不可用；仍可使用数值控件。错误：{_layout_editor_import_error}")
                                    layout_editor = gr.JSON(value=_layout_editor_empty(), label="layout transform payload", visible=False)
                                layout_enabled = gr.Checkbox(value=False, label="显示/启用版图 overlay")
                                gr.Markdown("多选 Label 时，下方数值只对应 Canvas 中当前激活的 Label。")
                                with gr.Row():
                                    layout_tx = gr.Number(value=0.0, label="水平偏移 tx")
                                    layout_ty = gr.Number(value=0.0, label="垂直偏移 ty")
                                with gr.Row():
                                    layout_scale = gr.Slider(minimum=0.1, maximum=20.0, value=1.0, step=0.01, label="缩放 scale")
                                    layout_rotation = gr.Slider(minimum=-180.0, maximum=180.0, value=0.0, step=1.0, label="旋转 rotation")
                                layout_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.35, step=0.05, label="透明度 alpha")
                                with gr.Row():
                                    reset_layout_btn = gr.Button("重置", variant="secondary")
                                    update_layout_preview_btn = gr.Button("更新预览", variant="primary")
                                create_from_layout_btn = gr.Button("用所选版图 mask / Label 创建实例", variant="primary")
                                layout_pvs_info = gr.Textbox(label="版图提示状态", lines=5, interactive=False)
                        with gr.Column(scale=1, min_width=280, elem_id="el-workspace-tools") as workspace_tools:
                            with gr.Group():
                                gr.Markdown("### \u4ea4\u4e92\u6a21\u5f0f")
                                click_tool = gr.Radio(
                                    choices=[("\u70b9\u63d0\u793a (Point)", "point"), ("\u6846\u63d0\u793a (Box)", "bbox"), ("\u591a\u8fb9\u5f62Mask (Polygon)", "polygon")],
                                    value="bbox",
                                    label="\u9009\u62e9\u6a21\u5f0f",
                                    show_label=False,
                                    elem_classes="mode-radio",
                                )
                                with gr.Group(visible=False) as pcs_bbox_tools:
                                    pcs_bbox_kind = gr.Radio(
                                        choices=[("\u6b63\u6837\u672c bbox", "Positive exemplar"), ("\u8d1f\u6837\u672c bbox", "Negative exemplar")],
                                        value="Positive exemplar",
                                        label="PCS bbox \u6837\u672c\u7c7b\u578b",
                                        elem_classes="mode-radio",
                                    )
                                    pcs_bbox_selector = gr.Dropdown(choices=[], label="PCS bbox \u5217\u8868", interactive=True)
                                    delete_selected_pcs_bbox_btn = gr.Button("\u5220\u9664\u9009\u4e2d PCS bbox", size="sm", variant="secondary")
                                with gr.Row():
                                    clear_prompt_btn = gr.Button("\u6e05\u7a7a\u63d0\u793a (Clear Prompts)", size="sm", variant="secondary")
                                interaction_info = gr.Markdown("\u70b9\u51fb\u56fe\u50cf\u5f00\u59cb\u6dfb\u52a0\u63d0\u793a...", elem_id="interaction-info")

                            with gr.Accordion("点提示修缮", open=False, visible=False) as layout_point_refine_panel:
                                layout_point_kind = gr.Radio(
                                    choices=[("正向点", "positive"), ("负向点", "negative")],
                                    value="positive",
                                    label="点类型",
                                    elem_classes="mode-radio",
                                )
                                layout_point_btn = gr.Button("应用点提示", variant="primary")
                            with gr.Group(visible=False) as pcs_panel:
                                gr.Markdown("### PCS Auto \u81ea\u52a8\u6982\u5ff5\u5206\u5272")
                                text_prompt = gr.Textbox(label="文本提示（可选）", placeholder="例如：circle", lines=1)
                                confidence_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="\u7f6e\u4fe1\u5ea6\u9608\u503c (Confidence)")
                                run_pcs_btn = gr.Button("\u5f00\u59cb PCS \u5206\u5272", variant="primary")
                                with gr.Row():
                                    clear_pcs_instances_btn = gr.Button("\u6e05\u7a7a PCS \u5b9e\u4f8b", variant="secondary")
                                    export_pcs_btn = gr.Button("\u5bfc\u51fa PCS")
                                with gr.Accordion("PCS 实例明细", open=False):
                                    pcs_summary = gr.Textbox(label="PCS \u5b9e\u4f8b", lines=6, interactive=False)
                            with gr.Group(visible=True) as pvs_panel:
                                gr.Markdown("### PVS Manual \u624b\u52a8\u5b9e\u4f8b\u5206\u5272")
                                with gr.Group(visible=True) as pvs_bbox_prompt_panel:
                                    gr.Markdown("#### BBox prompt")
                                    pvs_pending_count = gr.Markdown("\u5f85\u751f\u6210 bbox \u6570\u91cf: 0")
                                    pvs_pending_bbox_selector = gr.Dropdown(choices=[], label="PVS \u5f85\u751f\u6210 bbox \u5217\u8868", interactive=True)
                                    create_pvs_batch_btn = gr.Button("\u6279\u91cf\u751f\u6210 PVS \u5b9e\u4f8b", variant="primary")
                                    with gr.Row():
                                        delete_selected_pending_bbox_btn = gr.Button("\u5220\u9664\u9009\u4e2d\u5f85\u751f\u6210 bbox", size="sm", variant="secondary")
                                        clear_pending_bbox_btn = gr.Button("\u6e05\u7a7a\u5f85\u751f\u6210 bbox", size="sm", variant="secondary")
                                with gr.Group(visible=False) as pvs_point_prompt_panel:
                                    gr.Markdown("#### Point prompt")
                                    pvs_point_kind = gr.Radio(
                                        choices=[("正向点", "positive"), ("负向点", "negative")],
                                        value="positive",
                                        label="点类型",
                                        elem_classes="mode-radio",
                                    )
                                    pvs_point_btn = gr.Button("应用点提示", variant="primary")
                                with gr.Group(visible=False) as pvs_polygon_prompt_panel:
                                    gr.Markdown("#### Polygon prompt")
                                    polygon_action = gr.Radio(
                                        choices=[("\u521b\u5efa\u65b0 PVS \u5b9e\u4f8b", "create"), ("\u7cbe\u4fee\u5f53\u524d PVS \u5b9e\u4f8b", "refine")],
                                        value="create",
                                        label="\u591a\u8fb9\u5f62\u52a8\u4f5c",
                                        elem_classes="mode-radio",
                                    )
                                    finish_polygon_btn = gr.Button("\u5b8c\u6210\u591a\u8fb9\u5f62\u5bf9\u8c61", variant="primary", elem_classes="polygon-finish-btn")
                                    with gr.Accordion("高级 Polygon 融合方式", open=False):
                                        polygon_combine_mode = gr.Radio(
                                            choices=[("Replace \u91cd\u65b0\u5b9a\u4e49\u5b9e\u4f8b", "replace"), ("Blend \u4e0e\u65e7 mask \u878d\u5408", "blend"), ("Union \u8865\u5145\u533a\u57df", "union"), ("Intersect \u9650\u5236\u8303\u56f4", "intersect")],
                                            value="replace",
                                            label="\u591a\u8fb9\u5f62\u878d\u5408\u65b9\u5f0f",
                                            elem_classes="mode-radio",
                                        )
                                        gr.Markdown(
                                            "**\u4ee5\u4e0a\u56db\u79cd\u90fd\u662f Positive Polygon \u7684\u878d\u5408\u65b9\u5f0f\uff0c\u4e0d\u5305\u542b negative prompt\u3002**  \n"
                                            "- Replace \u91cd\u65b0\u5b9a\u4e49\u5b9e\u4f8b\uff1a\u7528\u5f53\u524d polygon \u4f5c\u4e3a\u5b8c\u6574 mask prompt\u3002  \n"
                                            "- Blend \u4e0e\u65e7 mask \u878d\u5408\uff1a\u65e7 logits \u548c polygon logits \u5171\u540c\u5f71\u54cd\u7ed3\u679c\u3002  \n"
                                            "- Union \u8865\u5145\u533a\u57df\uff1a\u4fdd\u7559\u65e7 mask\uff0c\u5e76\u52a0\u5165 polygon \u533a\u57df\u3002  \n"
                                            "- Intersect \u9650\u5236\u8303\u56f4\uff1a\u5c06\u7ed3\u679c\u9650\u5236\u5728 polygon \u8303\u56f4\u5185\u3002"
                                        )
                                pvs_summary = gr.Textbox(label="PVS 实例", lines=6, interactive=False, visible=False)
                            with gr.Group(visible=False) as pvs_layout_panel:
                                gr.Markdown("### PVS 版图 mask 提示")
                                gr.Markdown("上传或选择二值版图 mask；多选 Label 后，每个 Label 可在右侧独立拖动、缩放和旋转。")
                                with gr.Row():
                                    use_current_layout_btn = gr.Button("使用当前已保存版图 mask", variant="secondary")
                                    load_layout_binary_btn = gr.Button("载入二值 mask PNG", variant="secondary")
                                layout_prompt_mask_selector = gr.CheckboxGroup(
                                    choices=[("全部版图 mask", _LAYOUT_PROMPT_SCOPE_FULL)],
                                    value=[_LAYOUT_PROMPT_SCOPE_FULL],
                                    label="版图 mask 选择（可多选 Label）",
                                    interactive=False,
                                    elem_id="layout_prompt_mask_selector",
                                )
                                gr.Markdown("#### 直接上传二值 mask PNG")
                                layout_binary_upload = gr.Image(type="numpy", label="直接上传二值 mask PNG", show_label=False, sources=["upload", "clipboard"])
                            with gr.Group(visible=True) as pvs_action_panel:
                                gr.Markdown("### PVS 实例操作")
                                active_pvs = gr.Dropdown(choices=[], label="\u5f53\u524d PVS \u5b9e\u4f8b")
                                with gr.Row():
                                    delete_active_pvs_btn = gr.Button("删除当前实例", variant="secondary")
                                    clear_pvs_btn = gr.Button("清空所有实例", variant="secondary")
                                export_pvs_btn = gr.Button("\u5bfc\u51fa PVS")
                with gr.Tab("裁剪原图", id="crop", elem_classes="el-workflow-page", render_children=True):
                    source_image_upload = gr.Image(
                        type="pil",
                        label="完整原图",
                        show_label=False,
                        sources=["upload", "clipboard"],
                        interactive=False,
                        elem_id="source_input_image",
                        elem_classes="aligned-prepost-preview",
                        height=440,
                    )
                    if ImageGestureOverlay is not None:
                        source_crop_overlay = ImageGestureOverlay(
                            value=_source_gesture_payload(_new_source_image_state()),
                            label="完整原图裁剪手势",
                            show_label=False,
                            target_elem_id="source_input_image",
                            height=1,
                            elem_classes="gesture-overlay-anchor",
                        )
                    else:
                        gr.Markdown(f"图像手势组件不可用。错误：{_image_gesture_overlay_import_error}")
                        source_crop_overlay = gr.JSON(
                            value=_source_gesture_payload(_new_source_image_state()),
                            visible=False,
                        )
                    with gr.Row():
                        apply_crop_btn = gr.Button("应用裁剪", variant="primary", interactive=False)
                        use_full_image_btn = gr.Button("使用整图", variant="secondary", interactive=False)
                    source_crop_status = gr.Markdown("页面初始化中，请稍候…")
                with gr.Tab("模板匹配", id="template", elem_classes="el-workflow-page") as template_tab:
                    with gr.Row(elem_id="el-template-workspace"):
                        with gr.Column(scale=3, min_width=360):
                            template_match_preview = gr.Image(
                                type="pil",
                                label="完整原图模板匹配结果",
                                show_label=False,
                                interactive=False,
                                height=440,
                                elem_id="template_match_preview",
                                elem_classes="aligned-prepost-preview",
                            )
                        with gr.Column(scale=1, min_width=280):
                            with gr.Group(elem_classes="template-instance-card"):
                                gr.Markdown("#### 选择模板实例")
                                template_instance_selector = gr.CheckboxGroup(
                                    choices=[],
                                    label="用于模板匹配的 PVS 实例（可多选）",
                                    info="每个选中的 PVS 实例都会生成独立的衍生结果组。",
                                    elem_id="template_instance_selector",
                                )
                                refresh_template_instances_btn = gr.Button(
                                    "刷新实例列表",
                                    variant="secondary",
                                    size="sm",
                                )
                            with gr.Accordion("模板匹配可选参数", open=False):
                                match_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.01, label="matchThreshold")
                                expand_threshold = gr.Number(value=20, minimum=0, precision=0, label="expandThreshold (px)")
                                nms_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="nmsThreshold")
                            run_template_match_btn = gr.Button("开始模板匹配", variant="primary")
                            template_match_status = gr.Markdown("请先完成智能分割并选择当前 PVS 实例")
                            template_match_selection = gr.CheckboxGroup(
                                choices=[],
                                label="保存衍生结果组",
                                info="每个 PVS 实例对应一个结果组，可选择 Ax、Bx 或同时选择多个组。",
                                elem_id="template_match_selection",
                            )
                            export_template_selection_btn = gr.Button(
                                "生成所选衍生组下载包",
                                variant="secondary",
                            )
                            template_match_file = gr.File(label="下载模板匹配结果包", interactive=False)
                with gr.Tab("导出与记录", id="export", elem_classes="el-workflow-page"):
                    with gr.Column():
                        export_file = gr.File(label="下载结果包（PNG + masks + JSON）", interactive=False)
                        with gr.Accordion("\u5bfc\u51fa\u4e0e COCO \u91cf\u5316", open=False):
                            coco_dataset = gr.Dropdown(choices=coco_dataset_choices, value=default_coco_dataset, label="\u6307\u6807\u6570\u636e\u96c6")
                            coco_image_name = gr.Textbox(label="COCO image file_name\uff08\u53ef\u9009\uff09", lines=1)
                            coco_split = gr.Radio(choices=["auto", "val", "train", "test"], value="auto", label="\u6807\u6ce8 split")
                            coco_eval_scope = gr.Radio(choices=[coco_eval_scope_overlap, coco_eval_scope_full], value=coco_eval_scope_overlap, label="\u8bc4\u4f30\u8303\u56f4")
                            annotation_json_file = gr.File(label="\u4e0a\u4f20 O3/LabelMe-like JSON \u6807\u6ce8\uff08\u4f18\u5148\u4e8e COCO lookup\uff09", file_types=[".json"], type="filepath")
                        with gr.Accordion("小图标注与拼接", open=False):
                            stitch_tile_name = gr.Textbox(label="小图名称", placeholder="可留空")
                            save_stitch_tile_btn = gr.Button("保存标注并加入拼接", variant="primary")
                            update_stitch_tile_btn = gr.Button("更新拼接队列中当前小图")
                            saved_tile_file = gr.File(label="下载小图标注包", interactive=False)
                            save_tile_status = gr.Markdown("")
                        with gr.Group(visible=True) as analysis_report_panel:
                            analysis_report = gr.Textbox(label="分析报告", interactive=False, lines=8)
                        with gr.Accordion("结果反馈（PCS 结果 / PVS 当前实例，用于 RL 数据收集）", open=False):
                            feedback_rating = gr.Radio(
                                choices=[("好", "good"), ("及格", "pass"), ("差", "bad")],
                                value="pass",
                                label="结果质量",
                                elem_classes="mode-radio",
                            )
                            feedback_tags = gr.CheckboxGroup(
                                choices=["毛边", "空缺", "漏检", "误检", "边界偏移", "多分/粘连", "polygon 不贴合", "其他"],
                                label="问题标签",
                            )
                            feedback_comment = gr.Textbox(label="备注", lines=3, placeholder="可选：描述这次生成的问题或可用性")
                            submit_feedback_btn = gr.Button("提交反馈", variant="primary")
    return ComponentRefs(
        image_tab=image_tab, workspace_center=workspace_center, workspace_tools=workspace_tools,
        batch_state=batch_state, batch_upload=batch_upload, batch_gallery=batch_gallery,
        batch_selection=batch_selection, batch_status=batch_status,
        batch_select_all_btn=batch_select_all_btn, batch_select_none_btn=batch_select_none_btn,
        batch_delete_current_btn=batch_delete_current_btn, batch_delete_selected_btn=batch_delete_selected_btn,
        workflow_tabs=workflow_tabs, template_tab=template_tab,
        batch_prev_btn=batch_prev_btn, batch_save_btn=batch_save_btn, batch_next_btn=batch_next_btn,
        batch_run_btn=batch_run_btn, batch_retry_btn=batch_retry_btn,
        batch_cancel_btn=batch_cancel_btn, batch_stitch_btn=batch_stitch_btn,
        save_stitch_tile_btn=save_stitch_tile_btn,
        update_stitch_tile_btn=update_stitch_tile_btn,
        stitch_tile_name=stitch_tile_name,
        saved_tile_file=saved_tile_file,
        save_tile_status=save_tile_status,
        source_image_upload=source_image_upload,
        source_crop_overlay=source_crop_overlay,
        apply_crop_btn=apply_crop_btn,
        use_full_image_btn=use_full_image_btn,
        source_crop_status=source_crop_status,
        match_threshold=match_threshold,
        expand_threshold=expand_threshold,
        nms_threshold=nms_threshold,
        template_instance_selector=template_instance_selector,
        refresh_template_instances_btn=refresh_template_instances_btn,
        template_match_preview=template_match_preview,
        run_template_match_btn=run_template_match_btn,
        template_match_status=template_match_status,
        template_match_selection=template_match_selection,
        export_template_selection_btn=export_template_selection_btn,
        template_match_file=template_match_file,
        mode=mode,
        image_upload=image_upload,
        workspace_gesture_overlay=workspace_gesture_overlay,
        click_tool=click_tool,
        pcs_bbox_tools=pcs_bbox_tools,
        pcs_bbox_kind=pcs_bbox_kind,
        pcs_bbox_selector=pcs_bbox_selector,
        delete_selected_pcs_bbox_btn=delete_selected_pcs_bbox_btn,
        clear_prompt_btn=clear_prompt_btn,
        interaction_info=interaction_info,
        layout_point_refine_panel=layout_point_refine_panel,
        layout_point_kind=layout_point_kind,
        layout_point_btn=layout_point_btn,
        pcs_panel=pcs_panel,
        text_prompt=text_prompt,
        confidence_threshold=confidence_threshold,
        run_pcs_btn=run_pcs_btn,
        clear_pcs_instances_btn=clear_pcs_instances_btn,
        export_pcs_btn=export_pcs_btn,
        pcs_summary=pcs_summary,
        pvs_panel=pvs_panel,
        pvs_bbox_prompt_panel=pvs_bbox_prompt_panel,
        pvs_pending_count=pvs_pending_count,
        pvs_pending_bbox_selector=pvs_pending_bbox_selector,
        create_pvs_batch_btn=create_pvs_batch_btn,
        delete_selected_pending_bbox_btn=delete_selected_pending_bbox_btn,
        clear_pending_bbox_btn=clear_pending_bbox_btn,
        pvs_point_prompt_panel=pvs_point_prompt_panel,
        pvs_point_kind=pvs_point_kind,
        pvs_point_btn=pvs_point_btn,
        pvs_polygon_prompt_panel=pvs_polygon_prompt_panel,
        polygon_action=polygon_action,
        finish_polygon_btn=finish_polygon_btn,
        polygon_combine_mode=polygon_combine_mode,
        pvs_summary=pvs_summary,
        pvs_layout_panel=pvs_layout_panel,
        use_current_layout_btn=use_current_layout_btn,
        load_layout_binary_btn=load_layout_binary_btn,
        layout_prompt_mask_selector=layout_prompt_mask_selector,
        layout_binary_upload=layout_binary_upload,
        coco_dataset=coco_dataset,
        coco_image_name=coco_image_name,
        coco_split=coco_split,
        coco_eval_scope=coco_eval_scope,
        annotation_json_file=annotation_json_file,
        result_image=result_image,
        analysis_report_panel=analysis_report_panel,
        analysis_report=analysis_report,
        layout_transform_panel=layout_transform_panel,
        layout_editor=layout_editor,
        layout_enabled=layout_enabled,
        layout_tx=layout_tx,
        layout_ty=layout_ty,
        layout_scale=layout_scale,
        layout_rotation=layout_rotation,
        layout_alpha=layout_alpha,
        reset_layout_btn=reset_layout_btn,
        update_layout_preview_btn=update_layout_preview_btn,
        create_from_layout_btn=create_from_layout_btn,
        layout_pvs_info=layout_pvs_info,
        pvs_action_panel=pvs_action_panel,
        active_pvs=active_pvs,
        delete_active_pvs_btn=delete_active_pvs_btn,
        clear_pvs_btn=clear_pvs_btn,
        export_pvs_btn=export_pvs_btn,
        export_file=export_file,
        feedback_rating=feedback_rating,
        feedback_tags=feedback_tags,
        feedback_comment=feedback_comment,
        submit_feedback_btn=submit_feedback_btn,
    )
