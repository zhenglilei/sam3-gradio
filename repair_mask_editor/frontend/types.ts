import type { LoadingStatus } from "@gradio/statustracker";

export type RepairTool = "brush" | "eraser" | "rect_add" | "rect_erase";

export interface RepairMaskEditorValue {
  image_id?: string;
  revision?: number;
  source_width?: number;
  source_height?: number;
  base_image?: string | null;
  mask_png?: string | null;
  preview_alpha?: number;
  tool?: RepairTool;
  brush_size?: number;
  status?: string;
}

export interface RepairMaskEditorProps {
  value: null | RepairMaskEditorValue;
  height: number | string;
}

export interface RepairMaskEditorEvents {
  change: never;
  clear_status: LoadingStatus;
}
