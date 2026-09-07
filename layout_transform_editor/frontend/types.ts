import type { LoadingStatus } from "@gradio/statustracker";

export interface LayoutTransform {
	transform_version?: number;
	session_id?: string;
	layout_id?: string;
	image_id?: string;
	revision?: number;
	center_x: number;
	center_y: number;
	pivot_x: number;
	pivot_y: number;
	scale: number;
	rotation_deg: number;
	preview_alpha?: number;
	source_mask_pixel_sha256?: string;
	target_image_sha256?: string;
	origin?: string;
}

export type LayoutTransformGroupId = string | number;

export interface LayoutTransformGroupView {
	group_id: LayoutTransformGroupId;
	label: string;
	region_ids: number[];
	mask_image: string | null;
	foreground_bbox_xyxy: number[];
	group_mask_pixel_sha256: string;
}

export interface LayoutTransformGroupServerView {
	selection_signature: string;
	regions_revision: number;
	allow_reorder?: boolean;
	groups: LayoutTransformGroupView[];
}

export interface LayoutTransformGroupIntentEntry {
	group_id: LayoutTransformGroupId;
	transform: LayoutTransform;
}

export interface LayoutTransformGroupIntent {
	selection_signature: string;
	transform_set_revision: number;
	editor_session_id?: string;
	document_epoch?: string;
	client_revision?: number;
	payload_hash?: string;
	active_group_id: LayoutTransformGroupId | null;
	changed_group_ids?: LayoutTransformGroupId[];
	ordered_group_ids?: LayoutTransformGroupId[];
	transforms: LayoutTransformGroupIntentEntry[];
}

export interface LayoutTransformValue {
	payload_kind?: "snapshot" | "intent" | "ack" | "ack_error" | "alignment_proposal";
	editor_session_id?: string;
	document_epoch?: string;
	server_ack_revision?: number;
	payload_hash?: string;
	enabled?: boolean;
	base_image?: string | null;
	mask_image?: string | null;
	transform?: LayoutTransform | null;
	transform_mode?: string;
	group_view?: LayoutTransformGroupServerView | null;
	group_intent?: LayoutTransformGroupIntent | null;
	target_width?: number;
	target_height?: number;
	source_width?: number;
	source_height?: number;
	foreground_bbox_xyxy?: number[] | null;
	status?: string;
}

export interface LayoutTransformEditorProps {
	value: null | LayoutTransformValue;
	height: number | string;
}

export interface LayoutTransformEditorEvents {
	change: never;
	clear_status: LoadingStatus;
}
