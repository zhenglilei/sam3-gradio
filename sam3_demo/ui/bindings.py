"""Gradio event wiring kept separate from component construction."""


_MULTI_USER_CONCURRENCY_LIMIT = 8


def bind_demo_events(
    *,
    state_refs,
    image_refs,
    layout_refs,
    callbacks,
    stitch_refs=None,
    template_stitch_refs=None,
):
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
    template_instance_selector = image_refs.template_instance_selector
    refresh_template_instances_btn = image_refs.refresh_template_instances_btn
    template_match_preview = image_refs.template_match_preview
    run_template_match_btn = image_refs.run_template_match_btn
    template_match_status = image_refs.template_match_status
    template_match_selection = image_refs.template_match_selection
    export_template_selection_btn = image_refs.export_template_selection_btn
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
    clear_pcs_instances_btn = image_refs.clear_pcs_instances_btn
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
    delete_active_pvs_btn = image_refs.delete_active_pvs_btn
    clear_pvs_btn = image_refs.clear_pvs_btn
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
    layout_agent_pending_message = layout_refs.layout_agent_pending_message
    layout_agent_chatbot = layout_refs.layout_agent_chatbot
    layout_agent_prompt = layout_refs.layout_agent_prompt
    layout_agent_send_btn = layout_refs.layout_agent_send_btn
    layout_agent_apply_command = layout_refs.layout_agent_apply_command
    layout_agent_apply_btn = layout_refs.layout_agent_apply_btn
    layout_agent_draft_preview = layout_refs.layout_agent_draft_preview
    layout_agent_candidate_preview = layout_refs.layout_agent_candidate_preview
    layout_agent_diff = layout_refs.layout_agent_diff
    layout_agent_status = layout_refs.layout_agent_status
    layout_region_selector = layout_refs.layout_region_selector
    delete_layout_region_btn = layout_refs.delete_layout_region_btn
    layout_region_status = layout_refs.layout_region_status
    export_layout_regions_btn = layout_refs.export_layout_regions_btn
    layout_region_export_file = layout_refs.layout_region_export_file

    _run_layout_mask_page_with_downloads = callbacks["_run_layout_mask_page_with_downloads"]
    _layout_mask_agent_reset_callback = callbacks["_layout_mask_agent_reset_callback"]
    _layout_mask_agent_prepare_upload_callback = callbacks["_layout_mask_agent_prepare_upload_callback"]
    _layout_mask_agent_run_callback = callbacks["_layout_mask_agent_run_callback"]
    _layout_mask_agent_begin_chat_callback = callbacks["_layout_mask_agent_begin_chat_callback"]
    _layout_mask_agent_chat_callback = callbacks["_layout_mask_agent_chat_callback"]
    _layout_mask_agent_applied_preview_callback = callbacks["_layout_mask_agent_applied_preview_callback"]
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
    _template_instance_choices = callbacks["_template_instance_choices"]
    _preview_template_instance = callbacks["_preview_template_instance"]
    _template_match_selection_choices = callbacks["_template_match_selection_choices"]
    _export_template_match_selection = callbacks["_export_template_match_selection"]
    _finish_native_polygon = callbacks["_finish_native_polygon"]
    _clear_prompt_selection = callbacks["_clear_prompt_selection"]
    _switch_mode_with_layout_editor = callbacks["_switch_mode_with_layout_editor"]
    _switch_click_tool = callbacks["_switch_click_tool"]
    _delete_selected_pcs_bbox = callbacks["_delete_selected_pcs_bbox"]
    _clear_pcs_instances = callbacks["_clear_pcs_instances"]
    _run_pcs = callbacks["_run_pcs"]
    _create_pvs_from_pending_boxes = callbacks["_create_pvs_from_pending_boxes"]
    _delete_selected_pending_pvs_bbox = callbacks["_delete_selected_pending_pvs_bbox"]
    _clear_pending_pvs_boxes = callbacks["_clear_pending_pvs_boxes"]
    _pvs_point_prompt = callbacks["_pvs_point_prompt"]
    _layout_point_refine = callbacks["_layout_point_refine"]
    _set_active_pvs = callbacks["_set_active_pvs"]
    _delete_active_pvs = callbacks["_delete_active_pvs"]
    _clear_pvs = callbacks["_clear_pvs"]
    _export_pcs = callbacks["_export_pcs"]
    _export_pvs = callbacks["_export_pvs"]
    _submit_feedback = callbacks["_submit_feedback"]
    _load_stitch_tiles = callbacks["_load_stitch_tiles"]
    _apply_stitch_layout = callbacks["_apply_stitch_layout"]
    _auto_align_stitch = callbacks["_auto_align_stitch"]
    _stitch_canvas_changed = callbacks["_stitch_canvas_changed"]
    _apply_stitch_xy = callbacks["_apply_stitch_xy"]
    _apply_stitch_options = callbacks["_apply_stitch_options"]
    _apply_stitch_export_options = callbacks["_apply_stitch_export_options"]
    _generate_stitch_mosaic = callbacks["_generate_stitch_mosaic"]
    _crop_stitch_mosaic = callbacks["_crop_stitch_mosaic"]
    _restore_stitch_mosaic = callbacks["_restore_stitch_mosaic"]
    _handoff_stitch_mosaic = callbacks["_handoff_stitch_mosaic"]
    _stitch_handoff_source_image = callbacks["_stitch_handoff_source_image"]
    _stitch_handoff_status = callbacks["_stitch_handoff_status"]
    _run_template_stitch = callbacks["_run_template_stitch"]
    _handoff_template_stitch = callbacks["_handoff_template_stitch"]
    _template_stitch_handoff_source_image = callbacks["_template_stitch_handoff_source_image"]
    _template_stitch_handoff_status = callbacks["_template_stitch_handoff_status"]

    layout_agent_control_inputs = [
        layout_threshold,
        layout_invert,
        layout_open_kernel,
        layout_close_kernel,
        layout_min_area,
        layout_region_mode,
        layout_morph_pixels,
    ]
    layout_agent_reset_outputs = [
        layout_mask_agent_state,
        layout_agent_consent,
        layout_agent_chatbot,
        layout_agent_draft_preview,
        layout_agent_candidate_preview,
        layout_agent_diff,
        layout_agent_status,
        layout_agent_pending_message,
        layout_agent_prompt,
        layout_agent_send_btn,
    ]
    layout_agent_chat_outputs = [
        layout_mask_agent_state,
        layout_agent_chatbot,
        layout_agent_draft_preview,
        layout_agent_candidate_preview,
        layout_agent_diff,
        layout_agent_status,
        layout_agent_consent,
        layout_agent_pending_message,
        layout_agent_prompt,
        layout_agent_send_btn,
        *layout_agent_control_inputs,
    ]

    def bind_layout_agent_result(begin_event, api_name=None):
        chat_event = begin_event.then(
            fn=_layout_mask_agent_chat_callback,
            inputs=[
                session_state,
                layout_mask_agent_state,
                layout_input,
                layout_agent_consent,
                layout_agent_profile,
                layout_agent_pending_message,
                layout_agent_chatbot,
                *layout_agent_control_inputs,
            ],
            outputs=layout_agent_chat_outputs,
            concurrency_limit=1,
            concurrency_id="layout-mask-vlm",
            show_progress="hidden",
            api_name=api_name,
        )
        chat_event.then(
            fn=_layout_mask_agent_applied_preview_callback,
            inputs=[
                session_state,
                layout_mask_agent_state,
                layout_input,
            ],
            outputs=[
                layout_mask_preview,
                layout_overlay_preview,
            ],
            concurrency_limit=1,
            concurrency_id="layout-mask-vlm",
            show_progress="hidden",
        )
        return chat_event

    def bind_layout_agent_chat(event_method, message_input, api_name=None):
        begin_event = event_method(
            fn=_layout_mask_agent_begin_chat_callback,
            inputs=[layout_agent_chatbot, message_input],
            outputs=[
                layout_agent_chatbot,
                layout_agent_pending_message,
                layout_agent_prompt,
                layout_agent_status,
                layout_agent_send_btn,
            ],
            queue=False,
            show_progress="hidden",
        )
        return bind_layout_agent_result(begin_event, api_name=api_name)

    layout_agent_upload_event = layout_input.change(
        fn=_layout_mask_agent_prepare_upload_callback,
        inputs=[session_state, layout_input, layout_agent_profile],
        outputs=layout_agent_reset_outputs,
        concurrency_limit=1,
        concurrency_id="layout-mask-vlm",
        show_progress="hidden",
    )
    bind_layout_agent_result(layout_agent_upload_event)

    bind_layout_agent_chat(
        layout_agent_send_btn.click,
        layout_agent_prompt,
        api_name="_layout_mask_agent_run",
    )
    bind_layout_agent_chat(layout_agent_prompt.submit, layout_agent_prompt)
    bind_layout_agent_chat(
        layout_agent_apply_btn.click,
        layout_agent_apply_command,
    )

    run_layout_mask_event = run_layout_mask_btn.click(
        fn=_run_layout_mask_page_with_downloads,
        inputs=[session_state, image_state, layout_input, layout_threshold, layout_invert, layout_open_kernel, layout_close_kernel, layout_min_area, layout_region_mode, layout_morph_pixels],
        outputs=[layout_state, layout_editor, layout_source_preview, layout_mask_preview, layout_overlay_preview, layout_mask_file, layout_contour_file, layout_info],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
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
        ],
        concurrency_limit=1,
        concurrency_id="layout-mask-vlm",
        show_progress="hidden",
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
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
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
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
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
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
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
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
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
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
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
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    use_current_layout_event.then(
        fn=_load_layout_prompt_choices,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector, layout_pvs_info],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    load_layout_binary_event = load_layout_binary_btn.click(
        fn=_load_layout_binary_mask_png,
        inputs=[session_state, image_state, layout_binary_upload, layout_region_mode],
        outputs=[layout_state, layout_editor, layout_pvs_info],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    load_layout_binary_event.then(
        fn=_reset_layout_prompt_selection,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    layout_prompt_mask_selector.input(
        fn=_select_layout_prompt_mask,
        inputs=[image_state, layout_state, layout_prompt_mask_selector],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector, layout_pvs_info, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    update_layout_preview_btn.click(
        fn=_update_layout_preview_with_groups,
        inputs=[image_state, pcs_state, pvs_state, mode, layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor],
        outputs=[layout_state, image_upload, layout_editor, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_pvs_info],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    layout_editor.change(
        fn=_sync_layout_controls_from_editor_with_prompt_epoch,
        inputs=[layout_state, layout_editor],
        outputs=[layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_pvs_info],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    reset_layout_btn.click(
        fn=_reset_layout_controls_with_prompt_epoch,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor, layout_pvs_info],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    create_from_layout_event = create_from_layout_btn.click(
        fn=_create_pvs_from_layout_selection,
        inputs=[image_state, pcs_state, pvs_state, mode, layout_state, layout_enabled, layout_tx, layout_ty, layout_scale, layout_rotation, layout_alpha, layout_editor, layout_prompt_mask_selector],
        outputs=[pvs_state, layout_state, layout_editor, layout_pvs_info, *common],
        show_progress_on=[result_image],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    create_from_layout_event.then(
        fn=_clear_template_match_outputs,
        inputs=[template_match_state],
        outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
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
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    source_clear_event = source_image_upload.clear(
        fn=_source_upload_workspace,
        inputs=[source_image_upload, mode, session_state, layout_state],
        outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    source_crop_overlay.input(
        fn=_record_source_crop_gesture,
        inputs=[source_image_state, source_crop_overlay],
        outputs=[source_image_state, source_crop_overlay, source_crop_status],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    apply_crop_event = apply_crop_btn.click(
        fn=_apply_source_crop,
        inputs=[source_image_state, mode, session_state, layout_state],
        outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    use_full_image_event = use_full_image_btn.click(
        fn=_use_full_source_image,
        inputs=[source_image_state, mode, session_state, layout_state],
        outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    for workspace_event in (source_image_event, source_clear_event, apply_crop_event, use_full_image_event):
        workspace_event.then(fn=_clear_pending_point_payload, inputs=None, outputs=[point_payload], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
        workspace_event.then(fn=_clear_bbox_polygon_payloads, inputs=None, outputs=[bbox_payload, polygon_payload], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
        workspace_event.then(
            fn=_reset_layout_prompt_selection,
            inputs=[image_state, layout_state],
            outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state",
        )
        workspace_event.then(
            fn=_workspace_gesture_payload,
            inputs=[image_state, mode, click_tool],
            outputs=[workspace_gesture_overlay],
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state",
        )
        workspace_event.then(
            fn=_clear_template_match_outputs,
            inputs=[template_match_state],
            outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state",
        )
    workspace_gesture_overlay.input(
        fn=_workspace_gesture_input,
        inputs=[image_state, pcs_state, pvs_state, mode, click_tool, pcs_bbox_kind, prompt_state, workspace_gesture_overlay],
        outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, pcs_bbox_selector, pvs_pending_bbox_selector, *common, workspace_gesture_overlay],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    image_refs.template_tab.select(
        fn=_template_instance_choices,
        inputs=[pvs_state],
        outputs=[template_instance_selector],
        queue=False,
        show_progress="hidden",
    )
    refresh_template_instances_btn.click(
        fn=_template_instance_choices,
        inputs=[pvs_state],
        outputs=[template_instance_selector],
        queue=False,
        show_progress="hidden",
    )
    template_instance_event = template_instance_selector.change(
        fn=_preview_template_instance,
        inputs=[source_image_state, image_state, pvs_state, template_instance_selector],
        outputs=[pvs_state, template_match_state, template_match_preview, template_match_file, template_match_status],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    template_instance_event.then(
        fn=_template_match_selection_choices,
        inputs=[template_match_state],
        outputs=[template_match_selection],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    run_template_match_event = run_template_match_btn.click(
        fn=_run_template_matching,
        inputs=[source_image_state, image_state, pvs_state, mode, match_threshold, expand_threshold, nms_threshold],
        outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    run_template_match_event.then(
        fn=_template_match_selection_choices,
        inputs=[template_match_state],
        outputs=[template_match_selection],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    export_template_selection_btn.click(
        fn=_export_template_match_selection,
        inputs=[source_image_state, image_state, pvs_state, template_match_state, template_match_selection],
        outputs=[template_match_file, template_match_status],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    for template_parameter in (match_threshold, expand_threshold, nms_threshold):
        clear_event = template_parameter.change(
            fn=_clear_template_match_outputs,
            inputs=[template_match_state],
            outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state",
        )
        clear_event.then(
            fn=_template_match_selection_choices,
            inputs=[template_match_state],
            outputs=[template_match_selection],
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state",
        )
    finish_polygon_event = finish_polygon_btn.click(fn=_finish_native_polygon, inputs=[image_state, prompt_state, pcs_state, pvs_state, mode, polygon_action, polygon_combine_mode], outputs=[prompt_state, polygon_payload, pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    clear_prompt_btn.click(fn=_clear_prompt_selection, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, pcs_state, pvs_state, pcs_bbox_selector, pvs_pending_bbox_selector, text_prompt, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    mode_event = mode.input(fn=_switch_mode_with_layout_editor, inputs=[mode, image_state, pcs_state, pvs_state, layout_state], outputs=[prompt_state, bbox_payload, point_payload, polygon_payload, click_tool, finish_polygon_btn, pcs_bbox_tools, pcs_panel, pvs_panel, pvs_action_panel, analysis_report_panel, pvs_layout_panel, layout_transform_panel, pvs_bbox_prompt_panel, pvs_point_prompt_panel, pvs_polygon_prompt_panel, pcs_bbox_selector, pvs_pending_bbox_selector, layout_point_refine_panel, *common, layout_editor], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    mode_event.then(fn=_workspace_gesture_payload, inputs=[image_state, mode, click_tool], outputs=[workspace_gesture_overlay], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    click_tool_event = click_tool.change(fn=_switch_click_tool, inputs=[click_tool, mode], outputs=[pvs_bbox_prompt_panel, pvs_point_prompt_panel, pvs_polygon_prompt_panel], concurrency_limit=1)
    click_tool_event.then(fn=_workspace_gesture_payload, inputs=[image_state, mode, click_tool], outputs=[workspace_gesture_overlay], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    delete_selected_pcs_bbox_btn.click(fn=_delete_selected_pcs_bbox, inputs=[image_state, pcs_state, pvs_state, mode, pcs_bbox_selector], outputs=[pcs_state, pcs_bbox_selector, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    clear_pcs_instances_btn.click(fn=_clear_pcs_instances, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pcs_state, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    run_pcs_btn.click(fn=_run_pcs, inputs=[image_state, pcs_state, pvs_state, mode, text_prompt, confidence_threshold], outputs=[pcs_state, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    create_pvs_batch_event = create_pvs_batch_btn.click(fn=_create_pvs_from_pending_boxes, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, pvs_pending_bbox_selector, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    delete_selected_pending_bbox_btn.click(fn=_delete_selected_pending_pvs_bbox, inputs=[image_state, pcs_state, pvs_state, mode, pvs_pending_bbox_selector], outputs=[pvs_state, pvs_pending_bbox_selector, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    clear_pending_bbox_btn.click(fn=_clear_pending_pvs_boxes, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, pvs_pending_bbox_selector, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    pvs_point_event = pvs_point_btn.click(fn=_pvs_point_prompt, inputs=[image_state, pcs_state, pvs_state, mode, point_payload, pvs_point_kind], outputs=[pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    layout_point_event = layout_point_btn.click(fn=_layout_point_refine, inputs=[image_state, pcs_state, pvs_state, mode, point_payload, layout_point_kind, prompt_state], outputs=[prompt_state, point_payload, pvs_state, *common], show_progress_on=[result_image, analysis_report], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    active_pvs_event = active_pvs.change(fn=_set_active_pvs, inputs=[image_state, pcs_state, pvs_state, mode, active_pvs], outputs=[pvs_state, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    delete_active_pvs_event = delete_active_pvs_btn.click(fn=_delete_active_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    clear_pvs_event = clear_pvs_btn.click(fn=_clear_pvs, inputs=[image_state, pcs_state, pvs_state, mode], outputs=[pvs_state, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    for invalidating_event in (
        mode_event,
        finish_polygon_event,
        create_pvs_batch_event,
        pvs_point_event,
        layout_point_event,
        active_pvs_event,
        delete_active_pvs_event,
        clear_pvs_event,
    ):
        clear_event = invalidating_event.then(
            fn=_clear_template_match_outputs,
            inputs=[template_match_state],
            outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state",
        )
        sync_event = clear_event.then(
            fn=_template_instance_choices,
            inputs=[pvs_state],
            outputs=[template_instance_selector],
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state",
        )
        sync_event.then(
            fn=_template_match_selection_choices,
            inputs=[template_match_state],
            outputs=[template_match_selection],
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state",
        )
    pcs_export_event = export_pcs_btn.click(fn=_export_pcs, inputs=[image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file], outputs=[export_file, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    pvs_export_event = export_pvs_btn.click(fn=_export_pvs, inputs=[image_state, pcs_state, pvs_state, mode, coco_dataset, coco_image_name, coco_split, coco_eval_scope, annotation_json_file], outputs=[export_file, *common], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    for event in (pcs_export_event, pvs_export_event):
        event.then(
            fn=lambda: {"__type__": "update", "selected": "export"},
            inputs=[], outputs=[image_refs.workflow_tabs], queue=False,
            show_progress="hidden", api_visibility="private",
        )
    submit_feedback_btn.click(fn=_submit_feedback, inputs=[image_state, pcs_state, pvs_state, mode, feedback_rating, feedback_tags, feedback_comment], outputs=common, concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")

    if stitch_refs is None:
        return
    stitch_state = stitch_refs.stitch_state
    stitch_files = stitch_refs.stitch_files
    stitch_layout = stitch_refs.stitch_layout
    stitch_remove_black_border = stitch_refs.stitch_remove_black_border
    stitch_load_btn = stitch_refs.stitch_load_btn
    stitch_align_btn = stitch_refs.stitch_align_btn
    stitch_nudge_step = stitch_refs.stitch_nudge_step
    stitch_dx = stitch_refs.stitch_dx
    stitch_dy = stitch_refs.stitch_dy
    stitch_rotation = stitch_refs.stitch_rotation
    stitch_apply_xy_btn = stitch_refs.stitch_apply_xy_btn
    stitch_diff_mode = stitch_refs.stitch_diff_mode
    stitch_show_loupe = stitch_refs.stitch_show_loupe
    stitch_blend = stitch_refs.stitch_blend
    stitch_crop_periodic = stitch_refs.stitch_crop_periodic
    stitch_export_btn = stitch_refs.stitch_export_btn
    stitch_status = stitch_refs.stitch_status
    stitch_mosaic_preview = stitch_refs.stitch_mosaic_preview
    stitch_mosaic_crop_overlay = stitch_refs.stitch_mosaic_crop_overlay
    stitch_restore_full_btn = stitch_refs.stitch_restore_full_btn
    stitch_mosaic_file = stitch_refs.stitch_mosaic_file
    stitch_canvas = stitch_refs.stitch_canvas
    stitch_handoff_btn = stitch_refs.stitch_handoff_btn
    stitch_handoff_status = stitch_refs.stitch_handoff_status

    queue_outputs = [
        stitch_state, stitch_refs.annotated_gallery, stitch_refs.annotated_selection,
        stitch_refs.annotated_status, image_refs.saved_tile_file,
        stitch_mosaic_preview, stitch_mosaic_file, stitch_handoff_btn, stitch_handoff_status,
        stitch_status,
    ]
    save_inputs = [image_state, pcs_state, pvs_state, mode, stitch_state,
                   image_refs.stitch_tile_name, stitch_refs.annotated_selection]
    for button, name in ((image_refs.save_stitch_tile_btn, "_save_stitch_tile"),
                         (image_refs.update_stitch_tile_btn, "_update_stitch_tile")):
        event = button.click(fn=callbacks[name], inputs=save_inputs, outputs=queue_outputs,
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
        event.success(fn=lambda message: message, inputs=[stitch_refs.annotated_status],
                      outputs=[image_refs.save_tile_status])
    stitch_refs.annotated_import_btn.click(
        fn=callbacks["_import_stitch_annotations"],
        inputs=[stitch_refs.annotated_files, stitch_state], outputs=queue_outputs,
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    for button, name in ((stitch_refs.annotated_up, "_stitch_queue_up"),
                         (stitch_refs.annotated_down, "_stitch_queue_down"),
                         (stitch_refs.annotated_remove, "_stitch_queue_remove")):
        button.click(fn=callbacks[name], inputs=[stitch_state, stitch_refs.annotated_selection],
            outputs=queue_outputs, concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state")
    stitch_refs.annotated_load.click(
        fn=callbacks["_load_annotated_stitch"],
        inputs=[stitch_state, stitch_layout, stitch_nudge_step, stitch_diff_mode, stitch_show_loupe,
                stitch_blend, stitch_crop_periodic, stitch_remove_black_border,
                stitch_refs.stitch_crop_top, stitch_refs.stitch_crop_bottom,
                stitch_refs.stitch_crop_left, stitch_refs.stitch_crop_right],
        outputs=[stitch_state, stitch_canvas, stitch_dx, stitch_dy, stitch_status,
                 stitch_mosaic_preview, stitch_mosaic_file, stitch_handoff_btn,
                 stitch_handoff_status, stitch_rotation],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT, concurrency_id="image-prepost-state")
    for component in (stitch_refs.annotated_visible, stitch_refs.annotated_alpha):
        component.input(fn=callbacks["_stitch_annotation_display"],
            inputs=[stitch_refs.annotated_visible, stitch_refs.annotated_alpha, stitch_state],
            outputs=[stitch_state, stitch_canvas], concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state")

    stitch_load_btn.click(
        fn=_load_stitch_tiles,
        inputs=[
            stitch_files,
            stitch_layout,
            stitch_state,
            stitch_nudge_step,
            stitch_diff_mode,
            stitch_show_loupe,
            stitch_blend,
            stitch_crop_periodic,
            stitch_remove_black_border,
            stitch_refs.stitch_crop_top,
            stitch_refs.stitch_crop_bottom,
            stitch_refs.stitch_crop_left,
            stitch_refs.stitch_crop_right,
        ],
        outputs=[
            stitch_state,
            stitch_canvas,
            stitch_dx,
            stitch_dy,
            stitch_status,
            stitch_mosaic_preview,
            stitch_mosaic_file,
            stitch_handoff_btn,
            stitch_handoff_status,
            stitch_rotation,
        ],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    stitch_layout.input(
        fn=_apply_stitch_layout,
        inputs=[stitch_layout, stitch_state],
        outputs=[
            stitch_state,
            stitch_layout,
            stitch_canvas,
            stitch_dx,
            stitch_dy,
            stitch_status,
            stitch_mosaic_preview,
            stitch_mosaic_file,
            stitch_handoff_btn,
            stitch_handoff_status,
            stitch_rotation,
        ],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    stitch_align_btn.click(
        fn=_auto_align_stitch,
        inputs=[stitch_state, stitch_layout, stitch_nudge_step, stitch_diff_mode, stitch_show_loupe],
        outputs=[
            stitch_state,
            stitch_canvas,
            stitch_dx,
            stitch_dy,
            stitch_status,
            stitch_mosaic_preview,
            stitch_mosaic_file,
            stitch_handoff_btn,
            stitch_handoff_status,
            stitch_rotation,
        ],
        show_progress_on=[stitch_canvas],
        concurrency_limit=2,
        concurrency_id="stitch-auto-align",
    )
    stitch_canvas.change(
        fn=_stitch_canvas_changed,
        inputs=[stitch_canvas, stitch_state],
        outputs=[
            stitch_state,
            stitch_dx,
            stitch_dy,
            stitch_status,
            stitch_mosaic_preview,
            stitch_mosaic_file,
            stitch_handoff_btn,
            stitch_handoff_status,
            stitch_rotation,
        ],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    stitch_apply_xy_btn.click(
        fn=_apply_stitch_xy,
        inputs=[stitch_dx, stitch_dy, stitch_rotation, stitch_state],
        outputs=[
            stitch_state,
            stitch_canvas,
            stitch_dx,
            stitch_dy,
            stitch_status,
            stitch_mosaic_preview,
            stitch_mosaic_file,
            stitch_handoff_btn,
            stitch_handoff_status,
            stitch_rotation,
        ],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    for option in (stitch_nudge_step, stitch_diff_mode, stitch_show_loupe):
        option.change(
            fn=_apply_stitch_options,
            inputs=[stitch_nudge_step, stitch_diff_mode, stitch_show_loupe, stitch_state],
            outputs=[stitch_state, stitch_canvas, stitch_status],
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state",
        )
    for option in (stitch_blend, stitch_crop_periodic):
        option.change(
            fn=_apply_stitch_export_options,
            inputs=[stitch_blend, stitch_crop_periodic, stitch_state],
            outputs=[
                stitch_state,
                stitch_status,
                stitch_mosaic_preview,
                stitch_mosaic_file,
                stitch_handoff_btn,
                stitch_handoff_status,
            ],
            concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
            concurrency_id="image-prepost-state",
        )
    stitch_generate_event = stitch_export_btn.click(
        fn=_generate_stitch_mosaic,
        inputs=[stitch_state, stitch_blend, stitch_crop_periodic],
        outputs=[
            stitch_state,
            stitch_mosaic_preview,
            stitch_mosaic_file,
            stitch_status,
            stitch_handoff_btn,
            stitch_handoff_status,
            stitch_mosaic_crop_overlay,
        ],
        show_progress_on=[stitch_canvas],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    stitch_mosaic_crop_overlay.input(
        fn=_crop_stitch_mosaic,
        inputs=[stitch_mosaic_crop_overlay, stitch_state],
        outputs=[
            stitch_state,
            stitch_mosaic_preview,
            stitch_mosaic_file,
            stitch_status,
            stitch_handoff_btn,
            stitch_handoff_status,
            stitch_mosaic_crop_overlay,
            stitch_restore_full_btn,
        ],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    stitch_restore_full_btn.click(
        fn=_restore_stitch_mosaic,
        inputs=[stitch_state],
        outputs=[
            stitch_state,
            stitch_mosaic_preview,
            stitch_mosaic_file,
            stitch_status,
            stitch_handoff_btn,
            stitch_handoff_status,
            stitch_mosaic_crop_overlay,
            stitch_restore_full_btn,
        ],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    handoff_event = stitch_handoff_btn.click(
        fn=_handoff_stitch_mosaic,
        inputs=[stitch_state, mode, session_state, layout_state, stitch_refs.stitch_handoff_confirm],
        outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    handoff_event.success(
        fn=callbacks["_apply_stitch_instances"],
        inputs=[stitch_state, image_state, pcs_state, pvs_state, mode],
        outputs=[pvs_state, mode, *common],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    handoff_event.success(
        fn=lambda: False, inputs=None, outputs=[stitch_refs.stitch_handoff_confirm],
    )
    handoff_event.success(
        fn=_stitch_handoff_source_image,
        inputs=[stitch_state],
        outputs=[source_image_upload],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    handoff_event.success(
        fn=_stitch_handoff_status,
        inputs=[source_crop_status],
        outputs=[stitch_handoff_status],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    handoff_event.success(
        fn=_clear_pending_point_payload,
        inputs=None,
        outputs=[point_payload],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    handoff_event.success(
        fn=_clear_bbox_polygon_payloads,
        inputs=None,
        outputs=[bbox_payload, polygon_payload],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    handoff_event.success(
        fn=_reset_layout_prompt_selection,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    handoff_event.success(
        fn=_workspace_gesture_payload,
        inputs=[image_state, mode, click_tool],
        outputs=[workspace_gesture_overlay],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    handoff_event.success(
        fn=_clear_template_match_outputs,
        inputs=[template_match_state],
        outputs=[template_match_state, template_match_preview, template_match_file, template_match_status],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )

    if template_stitch_refs is None:
        return
    template_stitch_state = template_stitch_refs.template_stitch_state
    template_stitch_files = template_stitch_refs.template_stitch_files
    template_stitch_rows = template_stitch_refs.template_stitch_rows
    template_stitch_cols = template_stitch_refs.template_stitch_cols
    template_stitch_run_btn = template_stitch_refs.template_stitch_run_btn
    template_stitch_preview = template_stitch_refs.template_stitch_preview
    template_stitch_meta = template_stitch_refs.template_stitch_meta
    template_stitch_file = template_stitch_refs.template_stitch_file
    template_stitch_status = template_stitch_refs.template_stitch_status
    template_stitch_handoff_btn = template_stitch_refs.template_stitch_handoff_btn
    template_stitch_handoff_status = template_stitch_refs.template_stitch_handoff_status

    template_stitch_run_btn.click(
        fn=_run_template_stitch,
        inputs=[
            template_stitch_files,
            template_stitch_rows,
            template_stitch_cols,
            template_stitch_state,
        ],
        outputs=[
            template_stitch_state,
            template_stitch_preview,
            template_stitch_meta,
            template_stitch_file,
            template_stitch_status,
            template_stitch_handoff_btn,
            template_stitch_handoff_status,
        ],
        show_progress_on=[template_stitch_preview],
        concurrency_limit=2,
        concurrency_id="template-stitch",
    )
    template_handoff_event = template_stitch_handoff_btn.click(
        fn=_handoff_template_stitch,
        inputs=[template_stitch_state, mode, session_state, layout_state],
        outputs=[source_image_state, source_crop_overlay, source_crop_status, *workspace_init_outputs],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    template_handoff_event.success(
        fn=_template_stitch_handoff_source_image,
        inputs=[template_stitch_state],
        outputs=[source_image_upload],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    template_handoff_event.success(
        fn=_template_stitch_handoff_status,
        inputs=[source_crop_status],
        outputs=[template_stitch_handoff_status],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    template_handoff_event.success(
        fn=_clear_pending_point_payload,
        inputs=None,
        outputs=[point_payload],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    template_handoff_event.success(
        fn=_clear_bbox_polygon_payloads,
        inputs=None,
        outputs=[bbox_payload, polygon_payload],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    template_handoff_event.success(
        fn=_reset_layout_prompt_selection,
        inputs=[image_state, layout_state],
        outputs=[layout_state, layout_editor, layout_prompt_mask_selector],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    template_handoff_event.success(
        fn=_workspace_gesture_payload,
        inputs=[image_state, mode, click_tool],
        outputs=[workspace_gesture_overlay],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
    template_handoff_event.success(
        fn=_clear_template_match_outputs,
        inputs=[template_match_state],
        outputs=[
            template_match_state,
            template_match_preview,
            template_match_file,
            template_match_status,
        ],
        concurrency_limit=_MULTI_USER_CONCURRENCY_LIMIT,
        concurrency_id="image-prepost-state",
    )
