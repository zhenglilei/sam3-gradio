import type { LoadingStatus } from "@gradio/statustracker";

export type ToolMode = "browse" | "lasso";

export interface RegionSummary {
	region_id: number;
	label?: string;
	/** Legacy fields kept only for reading server payloads created before label unification. */
	class_label?: string;
	name?: string;
	area: number;
}

export function regionSummaryLabel(region: RegionSummary): string {
	const label = String(region.label || "").trim();
	if (label) return label;

	const classLabel = String(region.class_label || "").trim();
	const name = String(region.name || "").trim();
	if (classLabel && name && classLabel !== name) return `${classLabel} / ${name}`;
	return name || classLabel || `R${region.region_id}`;
}

export interface RegionServerView {
	enabled?: boolean;
	source_image?: string | null;
	source_mask_image?: string | null;
	saved_region_overlay_image?: string | null;
	draft_region_overlay_image?: string | null;
	natural_width?: number;
	natural_height?: number;
	regions_revision?: number;
	selected_region_id?: number | null;
	regions?: RegionSummary[];
	status?: string;
}

export interface RegionClientIntent {
	tool_mode?: ToolMode;
	lasso_polygon?: number[][];
	expected_regions_revision?: number | null;
	session_id?: string;
	layout_id?: string;
	source_mask_hash?: string;
}

export interface LayoutRegionAnnotatorValue {
	server_view?: RegionServerView;
	client_intent?: RegionClientIntent;
}

export interface LayoutRegionAnnotatorProps {
	value: LayoutRegionAnnotatorValue | null;
	height: number | string;
}

export interface LayoutRegionAnnotatorEvents {
	input: never;
	clear_status: LoadingStatus;
}
