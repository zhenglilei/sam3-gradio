"""Gradio event wiring kept separate from component construction."""


def bind_demo_events(*, state_refs, image_refs, layout_refs, callbacks):
    """Bind the existing demo events without changing callback contracts."""
    session_state = state_refs.session_state
    image_state = state_refs.image_state
    source_image_state = state_refs.source_image_state
    pcs_state = state_refs.pcs_state
    pvs_state = state_refs.pvs_state
    template_match_state = state_refs.template_match_state
    prompt_state = state_refs.prompt_state
    layout_state = state_refs.layout_state
    layout_mask_agent_state = state_refs.layout_mask_agent_state
    layout_region_state = state_refs.layout_region_state
    bbox_payload = state_refs.bbox_payload
    polygon_payload = state_refs.polygon_payload
    point_payload = state_refs.point_payload

    source_image_upload = image_refs.source_image_upload
    source_crop_overlay = image_refs.source_crop_overlay
    apply_crop_btn = image_refs.apply_crop_btn
    use_full_image_btn = image_refs.use_full_image_btn
    source_crop_status = image_refs.source_crop_status
    match_threshold = image_refs.match_threshold
    expand_threshold = image_refs.expand_threshold
    nms_threshold = image_refs.nms_threshold
    template_match_preview = image_refs.template_match_preview
    run_template_match_btn = image_refs.run_template_match_btn
    template_match_status = image_refs.template_match_status
    template_match_file = image_refs.template_match_file
    mode = image_refs.mode
    image_upload = image_refs.image_upload
    workspace_gesture_overlay = image_refs.workspace_gesture_overlay
    click_tool = image_refs.click_tool
    pcs_bbox_tools = image_refs.pcs_bbox_tools
    pcs_bbox_kind = image_refs.pcs_bbox_kind
    pcs_bbox_selector = image_refs.pcs_bbox_selector
    delete_selected_pcs_bbox_btn = image_refs.delete_selected_pcs_bbox_btn
    clear_prompt_btn = image_refs.clear_prompt_btn
    interaction_info = image_refs.interaction_info
    layout_point_refine_panel = image_refs.layout_point_refine_panel
    layout_point_kind = image_refs.layout_point_kind
    layout_point_btn = image_refs.layout_point_btn
    pcs_panel = image_refs.pcs_panel
    text_prompt = image_refs.text_prompt
    confidence_threshold = image_refs.confidence_threshold
    run_pcs_btn = image_refs.run_pcs_btn
    export_pcs_btn = image_refs.export_pcs_btn
    pcs_summary = image_refs.pcs_summary
    pvs_panel = image_refs.pvs_panel
    pvs_bbox_prompt_panel = image_refs.pvs_bbox_prompt_panel
    pvs_pending_count = image_refs.pvs_pending_count
    pvs_pending_bbox_selector = image_refs.pvs_pending_bbox_selector
    create_pvs_batch_btn = image_refs.create_pvs_batch_btn
    delete_selected_pending_bbox_btn = image_refs.delete_selected_pending_bbox_btn
    clear_pending_bbox_btn = image_refs.clear_pending_bbox_btn
    pvs_point_prompt_panel = image_refs.pvs_point_prompt_panel
    pvs_point_kind = image_refs.pvs_point_kind
    pvs_point_btn = image_refs.pvs_point_btn
    pvs_polygon_prompt_panel = image_refs.pvs_polygon_prompt_panel
    polygon_action = image_refs.polygon_action
    finish_polygon_btn = image_refs.finish_polygon_btn
    polygon_combine_mode = image_refs.polygon_combine_mode
    pvs_summary = image_refs.pvs_summary
    pvs_layout_panel = image_refs.pvs_layout_panel
    use_current_layout_btn = image_refs.use_current_layout_btn
    load_layout_binary_btn = image_refs.load_layout_binary_btn
    layout_prompt_mask_selector = image_refs.layout_prompt_mask_selector
    layout_binary_upload = image_refs.layout_binary_upload
    coco_dataset = image_refs.coco_dataset
    coco_image_name = image_refs.coco_image_name
    coco_split = image_refs.coco_split
    coco_eval_scope = image_refs.coco_eval_scope
    annotation_json_file = image_refs.annotation_json_file
    result_image = image_refs.result_image
    analysis_report_panel = image_refs.analysis_report_panel
    analysis_report = image_refs.analysis_report
    layout_transform_panel = image_refs.layout_transform_panel
    layout_editor = image_refs.layout_editor
    layout_enabled = image_refs.layout_enabled
    layout_tx = image_refs.layout_tx
    layout_ty = image_refs.layout_ty
    layout_scale = image_refs.layout_scale
    layout_rotation = image_refs.layout_rotation
    layout_alpha = image_refs.layout_alpha
    reset_layout_btn = image_refs.reset_layout_btn
    update_layout_preview_btn = image_refs.update_layout_preview_btn
    create_from_layout_btn = image_refs.create_from_layout_btn
    layout_pvs_info = image_refs.layout_pvs_info
    pvs_action_panel = image_refs.pvs_action_panel
    active_pvs = image_refs.active_pvs
    undo_pvs_btn = image_refs.undo_pvs_btn
    delete_pvs_btn = image_refs.delete_pvs_btn
    accept_pvs_btn = image_refs.accept_pvs_btn
    export_pvs_btn = image_refs.export_pvs_btn
    export_file = image_refs.export_file
    feedback_rating = image_refs.feedback_rating
    feedback_tags = image_refs.feedback_tags
    feedback_comment = image_refs.feedback_comment
    submit_feedback_btn = image_refs.submit_feedback_btn

    layout_input = layout_refs.layout_input
    layout_threshold = layout_refs.layout_threshold
    layout_invert = layout_refs.layout_invert
    layout_open_kernel = layout_refs.layout_open_kernel
    layout_close_kernel = layout_refs.layout_close_kernel
    layout_morph_pixels = layout_refs.layout_morph_pixels
    layout_min_area = layout_refs.layout_min_area
    layout_region_mode = layout_refs.layout_region_mode
    run_layout_mask_btn = layout_refs.run_layout_mask_btn
    save_layout_mask_btn = layout_refs.save_layout_mask_btn
    clear_layout_mask_btn = layout_refs.clear_layout_mask_btn
    layout_info = layout_refs.layout_info
    layout_mask_file = layout_refs.layout_mask_file
    layout_contour_file = layout_refs.layout_contour_file
    layout_source_preview = layout_refs.layout_source_preview
    layout_mask_preview = layout_refs.layout_mask_preview
    layout_overlay_preview = layout_refs.layout_overlay_preview
    layout_region_annotator = layout_refs.layout_region_annotator
    layout_region_label = layout_refs.layout_region_label
    save_layout_region_btn = layout_refs.save_layout_region_btn
    layout_agent_consent = layout_refs.layout_agent_consent
    layout_agent_profile = layout_refs.layout_agent_profile
    layout_agent_action_auto = layout_refs.layout_agent_action_auto
    layout_agent_action_fill = layout_refs.layout_agent_action_fill
    layout_agent_action_bridge = layout_refs.layout_agent_action_bridge
    layout_agent_action_thicken = layout_refs.layout_agent_action_thicken
    layout_agent_action_holes = layout_refs.layout_agent_action_holes
    layout_agent_auto_btn = layout_refs.layout_agent_auto_btn
    layout_agent_fill_btn = layout_refs.layout_agent_fill_btn
    layout_agent_bridge_btn = layout_refs.layout_agent_bridge_btn
    layout_agent_thicken_btn = layout_refs.layout_agent_thicken_btn
    layout_agent_holes_btn = layout_refs.layout_agent_holes_btn
    layout_agent_undo_btn = layout_refs.layout_agent_undo_btn
    layout_agent_reset_btn = layout_refs.layout_agent_reset_btn
    layout_agent_chatbot = layout_refs.layout_agent_chatbot
    layout_agent_prompt = layout_refs.layout_agent_prompt
    layout_agent_send_btn = layout_refs.layout_agent_send_btn
    layout_agent_draft_preview = layout_refs.layout_agent_draft_preview
    layout_agent_candidate_preview = layout_refs.layout_agent_candidate_preview
    layout_agent_diff = layout_refs.layout_agent_diff
    layout_agent_status = layout_refs.layout_agent_status
    layout_agent_apply_btn = layout_refs.layout_agent_apply_btn
    layout_region_selector = layout_refs.layout_region_selector
    delete_layout_region_btn = layout_refs.delete_layout_region_btn
    layout_region_status = layout_refs.layout_region_status
    export_layout_regions_btn = layout_refs.export_layout_regions_btn
    layout_region_export_file = layout_refs.layout_region_export_file

    _run_layout_mask_page_with_downloads = callbacks["_run_layout_mask_page_with_downloads"]
    _layout_mask_agent_reset_callback = callbacks["_layout_mask_agent_reset_callback"]
    _layout_mask_agent_consent_callback = callbacks["_layout_mask_agent_consent_callback"]
    _layout_mask_agent_run_callback = callbacks["_layout_mask_agent_run_callback"]
    _layout_mask_agent_undo_callback = callbacks["_layout_mask_agent_undo_callback"]
    _layout_mask_agent_apply_callback = callbacks["_layout_mask_agent_apply_callback"]
    _layout_mask_agent_mark_saved_callback = callbacks["_layout_mask_agent_mark_saved_callback"]
    _load_layout_region_context = callbacks["_load_layout_region_context"]
    _reset_layout_prompt_selection = callbacks["_reset_layout_prompt_selection"]
    _save_current_layout_mask = callbacks["_save_current_layout_mask"]
    _clear_current_layout_mask_with_prompt_epoch = callbacks["_clear_current_layout_mask_with_prompt_epoch"]
    _clear_layout_region_context = callbacks["_clear_layout_region_context"]
    _preview_layout_region = callbacks["_preview_layout_region"]
    _save_layout_region = callbacks["_save_layout_region"]
    _select_layout_region = callbacks["_select_layout_region"]
    _delete_layout_region = callbacks["_delete_layout_region"]
    _export_layout_regions = callbacks["_export_layout_regions"]
    _use_current_layout_mask = callbacks["_use_current_layout_mask"]
    _load_layout_prompt_choices = callbacks["_load_layout_prompt_choices"]
    _load_layout_binary_mask_png = callbacks["_load_layout_binary_mask_png"]
    _select_layout_prompt_mask = callbacks["_select_layout_prompt_mask"]
    _update_layout_preview_with_groups = callbacks["_update_layout_preview_with_groups"]
    _sync_layout_controls_from_editor_with_prompt_epoch = callbacks["_sync_layout_controls_from_editor_with_prompt_epoch"]
    _reset_layout_controls_with_prompt_epoch = callbacks["_reset_layout_controls_with_prompt_epoch"]
    _create_pvs_from_layout_selection = callbacks["_create_pvs_from_layout_selection"]
    _clear_template_match_outputs = callbacks["_clear_template_match_outputs"]
    _source_upload_workspace = callbacks["_source_upload_workspace"]
    _record_source_crop_gesture = callbacks["_record_source_crop_gesture"]
    _apply_source_crop = callbacks["_apply_source_crop"]
    _use_full_source_image = callbacks["_use_full_source_image"]
    _clear_pending_point_payload = callbacks["_clear_pending_point_payload"]
    _clear_bbox_polygon_payloads = callbacks["_clear_bbox_polygon_payloads"]
    _workspace_gesture_payload = callbacks["_workspace_gesture_payload"]
    _workspace_gesture_input = callbacks["_workspace_gesture_input"]
    _run_template_matching = callbacks["_run_template_matching"]
    _finish_native_polygon = callbacks["_finish_native_polygon"]
    _clear_prompt_selection = callbacks["_clear_prompt_selection"]
    _switch_mode_with_layout_editor = callbacks["_switch_mode_with_layout_editor"]
    _switch_click_tool = callbacks["_switch_click_tool"]
    _delete_selected_pcs_bbox = callbacks["_delete_selected_pcs_bbox"]
    _run_pcs = callbacks["_run_pcs"]
    _create_pvs_from_pending_boxes = callbacks["_create_pvs_from_pending_boxes"]
    _delete_selected_pending_pvs_bbox = callbacks["_delete_selected_pending_pvs_bbox"]
    _clear_pending_pvs_boxes = callbacks["_clear_pending_pvs_boxes"]
    _pvs_point_prompt = callbacks["_pvs_point_prompt"]
    _layout_point_refine = callbacks["_layout_point_refine"]
    _set_active_pvs = callbacks["_set_active_pvs"]
    _undo_pvs = callbacks["_undo_pvs"]
    _delete_pvs = callbacks["_delete_pvs"]
    _accept_pvs = callbacks["_accept_pvs"]
    _export_pcs = callbacks["_export_pcs"]
    _export_pvs = callbacks["_export_pvs"]
    _submit_feedback = callbacks["_submit_feedback"]

    layout_agent_control_inputs = [
        layout_threshold,
        layout_invert,
        layout_open_kernel,
        layout_close_kernel,
        layout_min_area,
        layout_region_mode,
        layout_morph_pixels,
    ]
    layout_agent_run_outputs = [
        layout_mask_agent_state,
        layout_agent_chatbot,
        layout_agent_draft_preview,
        layout_agent_candidate_preview,
        layout_agent_diff,
        layout_agent_status,
        layout_agent_apply_btn,
        layout_agent_prompt,
    ]
    layout_agent_reset_outputs = [
        layout_mask_agent_state,
        layout_agent_consent,
        layout_agent_chatbot,
        layout_agent_draft_preview,
        layout_agent_candidate_preview,
        layout_agent_diff,
        layout_agent_status,
        layout_agent_apply_btn,
        layout_agent_prompt,
    ]

    layout_input.change(
        fn=_layout_mask_agent_reset_callback,
        inputs=[session_state, layout_input, layout_agent_profile],
        outputs=layout_agent_reset_outputs,
        concurrency_limit=1,
        concurrency_id="layout-mask-agent-local",
    )
    layout_agent_profile.change(
        fn=_layout_mask_agent_reset_callback,
        inputs=[session_state, layout_input, layout_agent_profile],
        outputs=layout_agent_reset_outputs,
        concurrency_limit=1,
        concurrency_id="layout-mask-agent-local",
    )
    layout_agent_reset_btn.click(
        fn=_layout_mask_agent_reset_callback,
        inputs=[session_state, layout_input, layout_agent_profile],
        outputs=layout_agent_reset_outputs,
        concurrency_limit=1,
        concurrency_id="layout-mask-agent-local",
    )

    layout_agent_consent.change(
        fn=_layout_mask_agent_consent_callback,
        inputs=[
            session_state,
            layout_mask_agent_state,
            layout_input,
            layout_agent_consent,
            layout_agent_profile,
        ],
        outputs=[layout_mask_agent_state],
        queue=False,
    )

    def bind_layout_agent_request(trigger, message_component):
        trigger.click(
            fn=_layout_mask_agent_run_callback,
            inputs=[
                session_state,
                layout_mask_agent_state,
                layout_input,
                layout_agent_consent,
                layout_agent_profile,
                message_component,
                *layout_agent_control_inputs,
            ],
            outputs=layout_agent_run_outputs,
            concurrency_limit=1,
            concurrency_id="layout-mask-vlm",
        )

    for trigger, message_component in (
        (layout_agent_auto_btn, layout_agent_action_auto),
        (layout_agent_fill_btn, layout_agent_action_fill),
        (layout_agent_bridge_btn, layout_agent_action_bridge),
        (layout_agent_thicken_btn, layout_agent_action_thicken),
        (layout_agent_holes_btn, layout_agent_action_holes),
    ):
        bind_layout_agent_request(trigger, message_component)

    layout_agent_send_btn.click(
        fn=_layout_mask_agent_run_callback,
        inputs=[
            session_state,
            layout_mask_agent_state,
            layout_input,
            layout_agent_consent,
            layout_agent_profile,
            layout_agent_prompt,
            *layout_agent_control_inputs,
        ],
        outputs=layout_agent_run_outputs,
        concurrency_limit=1,
        concurrency_id="layout-mask-vlm",
        api_name="_layout_mask_agent_run",
    )
    layout_agent_prompt.submit(
        fn=_layout_mask_agent_run_callback,
        inputs=[
            session_state,
            layout_mask_agent_state,
            layout_input,
            layout_agent_consent,
            layout_agent_profile,
            layout_agent_prompt,
            *layout_agent_control_inputs,
        ],
        outputs=layout_agent_run_outputs,
        concurrency_limit=1,
        concurrency_id="layout-mask-vlm",
    )
    layout_agent_undo_btn.click(
        fn=_layout_mask_agent_undo_callback,
        inputs=[layout_mask_agent_state, layout_input],
        outputs=[
            layout_mask_agent_state,
            layout_agent_chatbot,
            layout_agent_draft_preview,
            layout_agent_diff,
            layout_agent_status,
            layout_agent_apply_btn,
        ],
        concurrency_limit=1,
        concurrency_id="layout-mask-agent-local",
    )
    layout_agent_apply_btn.click(
        fn=_layout_mask_agent_apply_callback,
        inputs=[layout_mask_agent_state],
        outputs=[
            layout_mask_agent_state,
            layout_threshold,
            layout_invert,
            layout_open_kernel,
            layout_close_kernel,
            layout_min_area,
            layout_region_mode,
            layout_morph_pixels,
            layout_agent_diff,
            layout_agent_status,
            layout_agent_apply_btn,
        ],
        concurrency_limit=1,
        concurrency_id="layout-mask-agent-local",
    )

    run_layout_mask_event = run_layout_mask_btn.click(
        fn=_run_layout_mask_page_with_downloads,
        inputs=[session_state, image_state, layout_input, layout_threshold, layout_invert, layout_open_kernel, layout_close_kernel, layout_min_area, layout_region_mode, layout_morph_pixels],
        outputs=[layout_state, layout_editor, layout_source_preview, layout_mask_preview, layout_overlay_preview, layout_mask_file, layout_contour_file, layout_info],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
        api_name="_run_layout_mask_page",
    )
    run_layout_mask_event.success(
        fn=_layout_mask_agent_mark_saved_callback,
        inputs=[
            layout_mask_agent_state,
            layout_state,
            *layout_agent_control_inputs,
        ],
        outputs=[
            layout_mask_agent_state,
            layout_agent_diff,
            layout_agent_status,
            layout_agent_apply_btn,
        ],
        concurrency_limit=1,
        concurrency_id="layout-mask-agent-local",
    )
    run_layout_region_event = run_layout_mask_event.then(
        fn=_load_layout_region_context,
        inputs=[layout_state],
        outputs=[layout_region_state, layout_region_annotator, layout_region_label, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
        concurrency_limit=1,
    )
    run_layout_region_event.then(
        fn=_reset_layout_prompt_selection,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    save_layout_mask_btn.click(
        fn=_save_current_layout_mask,
        inputs=[layout_state],
        outputs=[layout_mask_file, layout_contour_file, layout_info],
        concurrency_limit=1,
    )
    clear_layout_mask_event = clear_layout_mask_btn.click(
        fn=_clear_current_layout_mask_with_prompt_epoch,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_source_preview, layout_mask_preview, layout_overlay_preview, layout_mask_file, layout_contour_file, layout_info],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
        api_name="_clear_current_layout_mask",
    )
    clear_layout_region_event = clear_layout_mask_event.then(
        fn=_clear_layout_region_context,
        inputs=[layout_state],
        outputs=[layout_region_state, layout_region_annotator, layout_region_label, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
        concurrency_limit=1,
    )
    clear_layout_region_event.then(
        fn=_reset_layout_prompt_selection,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    layout_region_annotator.input(
        fn=_preview_layout_region,
        inputs=[layout_state, layout_region_state, layout_region_annotator],
        outputs=[layout_region_state, layout_region_annotator, save_layout_region_btn, layout_region_status],
        concurrency_limit=1,
    )
    save_layout_region_event = save_layout_region_btn.click(
        fn=_save_layout_region,
        inputs=[layout_state, layout_region_state, layout_region_annotator, layout_region_label],
        outputs=[layout_region_state, layout_region_annotator, layout_region_label, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
        concurrency_limit=1,
    )
    save_layout_region_event.then(
        fn=_reset_layout_prompt_selection,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    layout_region_selector.input(
        fn=_select_layout_region,
        inputs=[layout_state, layout_region_state, layout_region_selector],
        outputs=[layout_region_state, layout_region_annotator, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
        concurrency_limit=1,
    )
    delete_layout_region_event = delete_layout_region_btn.click(
        fn=_delete_layout_region,
        inputs=[layout_state, layout_region_state, layout_region_annotator, layout_region_selector],
        outputs=[layout_region_state, layout_region_annotator, layout_region_label, layout_region_selector, save_layout_region_btn, delete_layout_region_btn, layout_region_status],
        concurrency_limit=1,
    )
    delete_layout_region_event.then(
        fn=_reset_layout_prompt_selection,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )

    export_layout_regions_btn.click(
        fn=_export_layout_regions,
        inputs=[layout_state, layout_region_state],
        outputs=[layout_region_export_file, layout_region_status],
        concurrency_limit=1,
    )

    common = [image_upload, result_image, analysis_report, pcs_summary, pvs_summary, active_pvs, interaction_info, pvs_pending_count]
    use_current_layout_event = use_current_layout_btn.click(
        fn=_use_current_layout_mask,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_pvs_info],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    use_current_layout_event.then(
        fn=_load_layout_prompt_choices,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector, layout_pvs_info],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    load_layout_binary_event = load_layout_binary_btn.click(
        fn=_load_layout_binary_mask_png,
        inputs=[session_state, image_state, layout_binary_upload, layout_region_mode],
        outputs=[layout_state, layout_editor, layout_pvs_info],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    load_layout_binary_event.then(
        fn=_reset_layout_prompt_selection,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    layout_prompt_mask_selector.input(
        fn=_select_layout_prompt_mask,
        inputs=[image_state, layout_state, layout_prompt_mask_selector],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector, layout_pvs_info, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    update_layout_preview_btn.click(
        fn=_update_layout_preview_with_groups,
        inputs=[image_state, pcs_state, pvs_state, mode, layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor],
        outputs=[layout_state, image_upload, layout_editor, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_pvs_info],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    layout_editor.change(
        fn=_sync_layout_controls_from_editor_with_prompt_epoch,
        inputs=[layout_state, layout_editor],
        outputs=[layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_pvs_info],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    reset_layout_btn.click(
        fn=_reset_layout_controls_with_prompt_epoch,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor, layout_pvs_info],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    create_from_layout_event = create_from_layout_btn.click(
        fn=_create_pvs_from_layout_selection,
        inputs=[image_state, pcs_state, pvs_state, mode, layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor, layout_prompt_mask_selector],
        outputs=[pvs_state, layout_state, layout_editor, layout_pvs_info, *common],
        show_progress_on=[result_image],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    create_from_layout_event.then(
        fn=_clear_template_match_outputs,
        inputs=None,
        outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )

    workspace_init_outputs = [
        image_state,
        pcs_state,
        pvs_state,
        prompt_state,
        pcs_bbox_selector,
        pvs_pending_bbox_selector,
        *common,
        export_file,
        layout_editor,
    ]
    source_image_event = source_image_upload.upload(
        fn=_source_upload_workspace,
        inputs=[source_image_upload, mode, session_state, layout_state],
        outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    source_clear_event = source_image_upload.clear(
        fn=_source_upload_workspace,
        inputs=[source_image_upload, mode, session_state, layout_state],
        outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    source_crop_overlay.input(
        fn=_record_source_crop_gesture,
        inputs=[source_image_state, source_crop_overlay],
        outputs=[source_image_state, source_crop_overlay, source_crop_status],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    apply_crop_event = apply_crop_btn.click(
        fn=_apply_source_crop,
        inputs=[source_image_state, mode, session_state, layout_state],
        outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    use_full_image_event = use_full_image_btn.click(
        fn=_use_full_source_image,
        inputs=[source_image_state, mode, session_state, layout_state],
        outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    for workspace_event in (source_image_event, source_clear_event, apply_crop_event, use_full_image_event):
        workspace_event.then(fn=_clear_pending_point_payload, inputs=None, outputs=[point_payload], concurrency_limit=1, concurrency_id="image-prepost-state")
        workspace_event.then(fn=_clear_bbox_polygon_payloads, inputs=None, outputs=[bbox_payload, polygon_payload], concurrency_limit=1, concurrency_id="image-prepost-state")
        workspace_event.then(
            fn=_reset_layout_prompt_selection,
            inputs=[image_state, layout_state],
            outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
            concurrency_limit=1,
            concurrency_id="image-prepost-state",
        )
        workspace_event.then(
            fn=_workspace_gesture_payload,
            inputs=[image_state, mode, click_tool],
            outputs=[workspace_gesture_overlay],
            concurrency_limit=1,
            concurrency_id="image-prepost-state",
        )
        workspace_event.then(
            fn=_clear_template_match_outputs,
            inputs=None,
            outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
            concurrency_limit=1,
            concurrency_id="image-prepost-state",
        )
    workspace_gesture_overlay.input(
        fn=_workspace_gesture_input,
        inputs=[image_state, pcs_state, pvs_state, mode, click_tool, pcs_bbox_kind, prompt_state, workspace_gesture_overlay],
        outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, pcs_bbox_selector, pvs_pending_bbox_selector, *common, workspace_gesture_overlay],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    run_template_match_btn.click(
        fn=_run_template_matching,
        inputs=[source_image_state, image_state, pvs_state, mode, match_threshold, expand_threshold, nms_threshold],
        outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
        concurrency_limit=1,
        concurrency_id="image-prepost-state",
    )
    for template_parameter in (match_threshold, expand_threshold, nms_threshold):
        template_parameter.change(
            fn=_clear_template_match_outputs,
            inputs=None,
            outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
            concurrency_limit=1,
            concurrency_id="image-prepost-state",
        )
    finish_polygon_event = finish_polygon_btn.click(fn=_finish_native_polygon, inputs=[image_state, prompt_state, pcs_state, pvs_state, mode, polygon_action, polygon_combine_mode], outputs=[prompt_state, polygon_payload, pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1, concurrency_id="image-prepost-state")
    clear_prompt_btn.click(fn=_clear_prompt_selection, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, pcs_bbox_selector, pvs_pending_bbox_selector, text_prompt, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    mode_event = mode.change(fn=_switch_mode_with_layout_editor, inputs=[mode, image_state, pcs_state, pvs_state, layout_state], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, click_tool, finish_polygon_btn, pcs_bbox_tools, pcs_panel, pvs_panel, pvs_action_panel, analysis_report_panel, pvs_layout_panel, layout_transform_panel, pvs_bbox_prompt_panel, pvs_point_prompt_panel, pvs_polygon_prompt_panel, pcs_bbox_selector, pvs_pending_bbox_selector, layout_point_refine_panel, *common, layout_editor], concurrency_limit=1, concurrency_id="image-prepost-state")
    mode_event.then(fn=_workspace_gesture_payload, inputs=[image_state, mode, click_tool], outputs=[workspace_gesture_overlay], concurrency_limit=1, concurrency_id="image-prepost-state")
    click_tool_event = click_tool.change(fn=_switch_click_tool, inputs=[click_tool, mode], outputs=[pvs_bbox_prompt_panel, pvs_point_prompt_panel, pvs_polygon_prompt_panel], concurrency_limit=1)
    click_tool_event.then(fn=_workspace_gesture_payload, inputs=[image_state, mode, click_tool], outputs=[workspace_gesture_overlay], concurrency_limit=1, concurrency_id="image-prepost-state")
    delete_selected_pcs_bbox_btn.click(fn=_delete_selected_pcs_bbox, inputs=[image_state, pcs_state, pvs_state, mode, pcs_bbox_selector], outputs=[pcs_state, pcs_bbox_selector, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    run_pcs_btn.click(fn=_run_pcs, inputs=[image_state, pcs_state, pvs_state, mode, text_prompt, confidence_threshold], outputs=[pcs_state, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    create_pvs_batch_event = create_pvs_batch_btn.click(fn=_create_pvs_from_pending_boxes, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, pvs_pending_bbox_selector, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1, concurrency_id="image-prepost-state")
    delete_selected_pending_bbox_btn.click(fn=_delete_selected_pending_pvs_bbox, inputs=[image_state, pcs_state, pvs_state, mode, pvs_pending_bbox_selector], outputs=[pvs_state, pvs_pending_bbox_selector, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    clear_pending_bbox_btn.click(fn=_clear_pending_pvs_boxes, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, pvs_pending_bbox_selector, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    pvs_point_event = pvs_point_btn.click(fn=_pvs_point_prompt, inputs=[image_state, pcs_state, pvs_state, mode, point_payload, pvs_point_kind], outputs=[pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1, concurrency_id="image-prepost-state")
    layout_point_event = layout_point_btn.click(fn=_layout_point_refine, inputs=[image_state, pcs_state, pvs_state, mode, point_payload, layout_point_kind, prompt_state], outputs=[prompt_state, point_payload, pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=1, concurrency_id="image-prepost-state")
    active_pvs_event = active_pvs.change(fn=_set_active_pvs, inputs=[image_state, pcs_state, pvs_state, mode, active_pvs], outputs=[pvs_state, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    undo_pvs_event = undo_pvs_btn.click(fn=_undo_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    delete_pvs_event = delete_pvs_btn.click(fn=_delete_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    accept_pvs_event = accept_pvs_btn.click(fn=_accept_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    for invalidating_event in (
        mode_event,
        finish_polygon_event,
        create_pvs_batch_event,
        pvs_point_event,
        layout_point_event,
        active_pvs_event,
        undo_pvs_event,
        delete_pvs_event,
        accept_pvs_event,
    ):
        invalidating_event.then(
            fn=_clear_template_match_outputs,
            inputs=None,
            outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
            concurrency_limit=1,
            concurrency_id="image-prepost-state",
        )
    export_pcs_btn.click(fn=_export_pcs, inputs=[image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file], outputs=[export_file, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    export_pvs_btn.click(fn=_export_pvs, inputs=[image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file], outputs=[export_file, *common], concurrency_limit=1, concurrency_id="image-prepost-state")
    submit_feedback_btn.click(fn=_submit_feedback, inputs=[image_state, pcs_state, pvs_state, mode, feedback_rating, feedback_tags, feedback_comment], outputs=common, concurrency_limit=1, concurrency_id="image-prepost-state")
