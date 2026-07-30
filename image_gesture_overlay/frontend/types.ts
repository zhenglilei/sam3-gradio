import type { LoadingStatus } from "@gradio/statustracker";

export type Gesture = "click" | "drag";
export type Interaction = "auto" | "workspace" | "crop" | "drag" | "bbox" | "click" | "point" | "polygon" | "disabled";
export type PointTuple = [number, number];

export interface GestureServerView {
	enabled?: boolean;
	natural_width?: number;
	natural_height?: number;
	image_id?: string;
	image_sha256?: string;
	revision?: number;
	interaction?: Interaction | string;
	selection_state?: "draft" | "applied" | string;
}

export interface GestureClientIntent {
	gesture?: Gesture | "";
	start_xy?: PointTuple | number[];
	end_xy?: PointTuple | number[];
	expected_revision?: number | null;
	image_id?: string;
	image_sha256?: string;
}

export interface ImageGestureOverlayValue {
	server_view?: GestureServerView;
	client_intent?: GestureClientIntent;
}

export interface ImageGestureOverlayProps {
	value: ImageGestureOverlayValue | null;
	height: number | string;
	target_elem_id: string;
}

export interface ImageGestureOverlayEvents {
	input: never;
	clear_status: LoadingStatus;
}
