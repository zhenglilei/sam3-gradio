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

export interface LayoutTransformValue {
	enabled?: boolean;
	base_image?: string | null;
	mask_image?: string | null;
	transform?: LayoutTransform | null;
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
