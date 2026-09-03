import type { LoadingStatus } from "@gradio/statustracker";

export interface StitchTile {
	index: number;
	image?: string | null;
	x: number;
	y: number;
	width: number;
	height: number;
	/** Clockwise rotation around the tile center, normalized to (-180, 180]. */
	rotation_deg: number;
}

export interface StitchPreviewValue {
	tiles?: StitchTile[];
	selected?: number;
	nudge_step?: number;
	diff_mode?: boolean;
	show_loupe?: boolean;
	drag_gain?: number;
	status?: string;
}

export interface StitchPreviewCanvasProps {
	value: null | StitchPreviewValue;
	height: number | string;
}

export interface StitchPreviewCanvasEvents {
	change: never;
	clear_status: LoadingStatus;
}
