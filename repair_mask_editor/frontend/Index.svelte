<svelte:options accessors={true} />

<script lang="ts">
  import { Block } from "@gradio/atoms";
  import { StatusTracker } from "@gradio/statustracker";
  import { Gradio } from "@gradio/utils";
  import { onDestroy, onMount } from "svelte";
  import type {
    RepairMaskEditorEvents,
    RepairMaskEditorProps,
    RepairMaskEditorValue,
    RepairTool,
  } from "./types";

  type Point = { x: number; y: number };
  type Rect = { x: number; y: number; width: number; height: number };
  type CanvasSnapshot = { mask: ImageData; tint: ImageData };

  const props = $props();
  const gradio = new Gradio<RepairMaskEditorEvents, RepairMaskEditorProps>(props);
  const MASK_COLOR = "#f0445d";
  const DEFAULT_ALPHA = 0.45;
  const DEFAULT_BRUSH_SIZE = 24;
  const MIN_BRUSH_SIZE = 1;
  const MAX_BRUSH_SIZE = 512;

  let surfaceEl: HTMLDivElement;
  let canvasEl: HTMLCanvasElement;
  let localValue = $state<RepairMaskEditorValue>({});
  let baseImg: HTMLImageElement | null = null;
  let maskImg: HTMLImageElement | null = null;
  let maskCanvas: HTMLCanvasElement | null = null;
  let tintCanvas: HTMLCanvasElement | null = null;
  let displayCssWidth = 1;
  let displayCssHeight = 1;
  let resolvedSourceWidth = 0;
  let resolvedSourceHeight = 0;
  let loadedInputSignature = "";
  let lastSignature = "";
  let imageLoadGeneration = 0;
  let activePointerId: number | null = null;
  let pointerStart: Point | null = null;
  let pointerLast: Point | null = null;
  let pointerSnapshot: CanvasSnapshot | null = null;
  let pointerDirty = false;
  let draftRect: Rect | null = null;
  let resizeObserver: ResizeObserver | null = null;
  let resizeFrame: number | null = null;

  function cloneValue(value: RepairMaskEditorValue | null | undefined): RepairMaskEditorValue {
    return JSON.parse(JSON.stringify(value || {})) as RepairMaskEditorValue;
  }

  function finiteNumber(value: unknown, fallback: number): number {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? numberValue : fallback;
  }

  function clamp(value: number, low: number, high: number): number {
    return Math.max(low, Math.min(high, value));
  }

  function positiveInteger(value: unknown, fallback: number): number {
    const numberValue = Math.round(finiteNumber(value, fallback));
    if (numberValue <= 0) return fallback;
    return Math.max(1, numberValue);
  }

  function heightStyle(value: number | string | undefined): string {
    if (typeof value === "number") return String(value) + "px";
    return value || "520px";
  }

  function activeTool(): RepairTool {
    const value = localValue.tool;
    return value === "eraser" || value === "rect_add" || value === "rect_erase" ? value : "brush";
  }

  function previewAlpha(): number {
    return clamp(finiteNumber(localValue.preview_alpha, DEFAULT_ALPHA), 0, 1);
  }

  function brushSize(): number {
    return clamp(
      finiteNumber(localValue.brush_size, DEFAULT_BRUSH_SIZE),
      MIN_BRUSH_SIZE,
      MAX_BRUSH_SIZE,
    );
  }

  function baseSource(): string | null {
    return typeof localValue.base_image === "string" && localValue.base_image
      ? localValue.base_image
      : null;
  }

  function maskSource(): string | null {
    for (const value of [localValue.mask_png]) {
      if (typeof value === "string" && value.startsWith("data:image/png;base64,")) return value;
    }
    return null;
  }

  function sourceDimension(field: "source_width" | "source_height"): number {
    const explicit = positiveInteger(localValue[field], 0);
    if (explicit > 0) return explicit;
    const resolved = field === "source_width" ? resolvedSourceWidth : resolvedSourceHeight;
    if (resolved > 0) return resolved;
    const image = baseImg || maskImg;
    if (!image) return 1;
    return field === "source_width"
      ? Math.max(1, image.naturalWidth)
      : Math.max(1, image.naturalHeight);
  }

  function sourceSize(): { width: number; height: number } {
    resolveSourceDimensions();
    return {
      width: sourceDimension("source_width"),
      height: sourceDimension("source_height"),
    };
  }

  function resolveSourceDimensions(): void {
    const explicitWidth = positiveInteger(localValue.source_width, 0);
    const explicitHeight = positiveInteger(localValue.source_height, 0);
    if (explicitWidth > 0) resolvedSourceWidth = explicitWidth;
    if (explicitHeight > 0) resolvedSourceHeight = explicitHeight;
    const image = baseImg || maskImg;
    if (!image) return;
    if (resolvedSourceWidth <= 0) resolvedSourceWidth = Math.max(1, image.naturalWidth);
    if (resolvedSourceHeight <= 0) resolvedSourceHeight = Math.max(1, image.naturalHeight);
  }

  function inputSignature(value: RepairMaskEditorValue): string {
    return JSON.stringify([
      value.image_id || "",
      value.revision || 0,
      value.base_image || "",
      value.mask_png || "",
      value.source_width || 0,
      value.source_height || 0,
    ]);
  }

  function loadImage(
    source: string | null,
    generation: number,
    callback: (image: HTMLImageElement | null) => void,
  ): void {
    if (!source) {
      if (generation === imageLoadGeneration) callback(null);
      return;
    }
    const image = new Image();
    image.onload = () => {
      if (generation === imageLoadGeneration) callback(image);
    };
    image.onerror = () => {
      if (generation === imageLoadGeneration) callback(null);
    };
    image.src = source;
  }

  function createCanvas(width: number, height: number): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }

  function buildMaskCanvases(image: HTMLImageElement | null): void {
    const size = sourceSize();
    const nextMask = createCanvas(size.width, size.height);
    const nextTint = createCanvas(size.width, size.height);
    const maskContext = nextMask.getContext("2d", { willReadFrequently: true });
    const tintContext = nextTint.getContext("2d", { willReadFrequently: true });
    if (!maskContext || !tintContext) return;
    maskContext.imageSmoothingEnabled = false;
    tintContext.imageSmoothingEnabled = false;
    if (image) maskContext.drawImage(image, 0, 0, size.width, size.height);
    const input = maskContext.getImageData(0, 0, size.width, size.height);
    const maskOutput = maskContext.createImageData(size.width, size.height);
    const tintOutput = tintContext.createImageData(size.width, size.height);
    const color = parseColor(MASK_COLOR);
    for (let index = 0; index < input.data.length; index += 4) {
      const luminance = Math.max(input.data[index], input.data[index + 1], input.data[index + 2]);
      const alpha = Math.round((input.data[index + 3] * luminance) / 255);
      maskOutput.data[index] = 255;
      maskOutput.data[index + 1] = 255;
      maskOutput.data[index + 2] = 255;
      maskOutput.data[index + 3] = alpha;
      tintOutput.data[index] = color[0];
      tintOutput.data[index + 1] = color[1];
      tintOutput.data[index + 2] = color[2];
      tintOutput.data[index + 3] = alpha;
    }
    maskContext.putImageData(maskOutput, 0, 0);
    tintContext.putImageData(tintOutput, 0, 0);
    maskCanvas = nextMask;
    tintCanvas = nextTint;
  }

  function ensureCanvases(): void {
    const size = sourceSize();
    if (
      maskCanvas &&
      tintCanvas &&
      maskCanvas.width === size.width &&
      maskCanvas.height === size.height &&
      tintCanvas.width === size.width &&
      tintCanvas.height === size.height
    ) {
      return;
    }
    buildMaskCanvases(maskImg);
  }

  function parseColor(value: string): [number, number, number] {
    const match = value.match(/^#([0-9a-f]{6})$/i);
    if (!match) return [240, 68, 93];
    return [
      Number.parseInt(match[1].slice(0, 2), 16),
      Number.parseInt(match[1].slice(2, 4), 16),
      Number.parseInt(match[1].slice(4, 6), 16),
    ];
  }

  function editContexts(): CanvasRenderingContext2D[] {
    ensureCanvases();
    const contexts: CanvasRenderingContext2D[] = [];
    for (const canvas of [maskCanvas, tintCanvas]) {
      const context = canvas?.getContext("2d", { willReadFrequently: true });
      if (context) contexts.push(context);
    }
    return contexts;
  }

  function drawStroke(start: Point, end: Point): void {
    const tool = activeTool();
    const erasing = tool === "eraser";
    for (const context of editContexts()) {
      context.save();
      context.globalCompositeOperation = erasing ? "destination-out" : "source-over";
      context.strokeStyle = context.canvas === maskCanvas ? "#ffffff" : MASK_COLOR;
      context.lineWidth = brushSize();
      context.lineCap = "round";
      context.lineJoin = "round";
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
      context.restore();
    }
  }

  function normalizeRect(start: Point, end: Point): Rect {
    const left = Math.min(start.x, end.x);
    const top = Math.min(start.y, end.y);
    return {
      x: left,
      y: top,
      width: Math.max(1, Math.abs(end.x - start.x)),
      height: Math.max(1, Math.abs(end.y - start.y)),
    };
  }

  function drawRectangle(rect: Rect): void {
    const erasing = activeTool() === "rect_erase";
    for (const context of editContexts()) {
      context.save();
      context.globalCompositeOperation = erasing ? "destination-out" : "source-over";
      context.fillStyle = context.canvas === maskCanvas ? "#ffffff" : MASK_COLOR;
      context.fillRect(rect.x, rect.y, rect.width, rect.height);
      context.restore();
    }
  }

  function cloneSnapshot(): CanvasSnapshot | null {
    ensureCanvases();
    const maskContext = maskCanvas?.getContext("2d", { willReadFrequently: true });
    const tintContext = tintCanvas?.getContext("2d", { willReadFrequently: true });
    if (!maskContext || !tintContext || !maskCanvas || !tintCanvas) return null;
    return {
      mask: maskContext.getImageData(0, 0, maskCanvas.width, maskCanvas.height),
      tint: tintContext.getImageData(0, 0, tintCanvas.width, tintCanvas.height),
    };
  }

  function restoreSnapshot(snapshot: CanvasSnapshot | null): void {
    if (!snapshot || !maskCanvas || !tintCanvas) return;
    maskCanvas.getContext("2d")?.putImageData(snapshot.mask, 0, 0);
    tintCanvas.getContext("2d")?.putImageData(snapshot.tint, 0, 0);
  }

  function surfaceSize(): { width: number; height: number } {
    const rect = surfaceEl?.getBoundingClientRect();
    return {
      width: Math.max(1, rect?.width || 1),
      height: Math.max(1, rect?.height || 1),
    };
  }

  function containRect(width: number, height: number): {
    left: number;
    top: number;
    width: number;
    height: number;
    scale: number;
  } {
    const size = sourceSize();
    const scale = Math.min(width / size.width, height / size.height);
    const displayWidth = size.width * scale;
    const displayHeight = size.height * scale;
    return {
      left: (width - displayWidth) / 2,
      top: (height - displayHeight) / 2,
      width: displayWidth,
      height: displayHeight,
      scale,
    };
  }

  function eventToImage(event: PointerEvent, clampOutside: boolean): Point | null {
    if (!surfaceEl) return null;
    const surfaceRect = surfaceEl.getBoundingClientRect();
    const rect = containRect(surfaceRect.width, surfaceRect.height);
    const localX = event.clientX - surfaceRect.left;
    const localY = event.clientY - surfaceRect.top;
    const inside =
      localX >= rect.left &&
      localX <= rect.left + rect.width &&
      localY >= rect.top &&
      localY <= rect.top + rect.height;
    if (!inside && !clampOutside) return null;
    const size = sourceSize();
    return {
      x: clamp((localX - rect.left) / rect.scale, 0, size.width),
      y: clamp((localY - rect.top) / rect.scale, 0, size.height),
    };
  }

  function resizeDisplayCanvas(): void {
    if (!canvasEl) return;
    const size = surfaceSize();
    const width = Math.max(1, Math.round(size.width));
    const height = Math.max(1, Math.round(size.height));
    if (canvasEl.width !== width || canvasEl.height !== height) {
      canvasEl.width = width;
      canvasEl.height = height;
    }
    displayCssWidth = size.width;
    displayCssHeight = size.height;
  }

  function scheduleResizeDraw(): void {
    if (resizeFrame !== null) return;
    resizeFrame = requestAnimationFrame(() => {
      resizeFrame = null;
      resizeDisplayCanvas();
      draw();
    });
  }

  function drawChecker(context: CanvasRenderingContext2D, width: number, height: number): void {
    context.fillStyle = "#f1f5f9";
    context.fillRect(0, 0, width, height);
    context.fillStyle = "#e2e8f0";
    const cell = 16;
    for (let y = 0; y < height; y += cell) {
      for (let x = 0; x < width; x += cell) {
        if ((x / cell + y / cell) % 2 === 0) context.fillRect(x, y, cell, cell);
      }
    }
  }

  function drawDraft(context: CanvasRenderingContext2D, rect: Rect, imageRect: ReturnType<typeof containRect>): void {
    const x = imageRect.left + rect.x * imageRect.scale;
    const y = imageRect.top + rect.y * imageRect.scale;
    const width = rect.width * imageRect.scale;
    const height = rect.height * imageRect.scale;
    context.save();
    context.fillStyle = activeTool() === "rect_erase" ? "rgba(37, 99, 235, 0.16)" : "rgba(240, 68, 93, 0.18)";
    context.strokeStyle = activeTool() === "rect_erase" ? "#2563eb" : MASK_COLOR;
    context.lineWidth = 2;
    context.setLineDash([6, 4]);
    context.fillRect(x, y, width, height);
    context.strokeRect(x, y, width, height);
    context.restore();
  }

  function draw(): void {
    if (!canvasEl) return;
    resizeDisplayCanvas();
    ensureCanvases();
    const context = canvasEl.getContext("2d");
    if (!context) return;
    const width = displayCssWidth;
    const height = displayCssHeight;
    context.clearRect(0, 0, width, height);
    drawChecker(context, width, height);
    const imageRect = containRect(width, height);
    if (baseImg) {
      context.save();
      context.imageSmoothingEnabled = true;
      context.drawImage(baseImg, imageRect.left, imageRect.top, imageRect.width, imageRect.height);
      context.restore();
    }
    if (tintCanvas) {
      context.save();
      context.imageSmoothingEnabled = false;
      context.globalAlpha = previewAlpha();
      context.drawImage(tintCanvas, imageRect.left, imageRect.top, imageRect.width, imageRect.height);
      context.restore();
    }
    if (draftRect) drawDraft(context, draftRect, imageRect);
  }

  function publishValue(status: string): void {
    ensureCanvases();
    const size = sourceSize();
    const revision = Math.max(0, Math.trunc(finiteNumber(localValue.revision, 0))) + 1;
    const outbound: RepairMaskEditorValue = {
      ...localValue,
      image_id: String(localValue.image_id || ""),
      revision,
      source_width: size.width,
      source_height: size.height,
      base_image: baseSource(),
      mask_png: maskCanvas ? maskCanvas.toDataURL("image/png") : null,
      preview_alpha: previewAlpha(),
      tool: activeTool(),
      brush_size: brushSize(),
      status,
    };
    localValue = outbound;
    gradio.props.value = outbound;
    lastSignature = JSON.stringify(outbound);
    gradio.dispatch("change");
    draw();
  }

  function commitValue(status = "mask updated"): void {
    publishValue(status);
  }

  function ingestValue(value: RepairMaskEditorValue | null): void {
    const next = cloneValue(value);
    localValue = next;
    const nextInputSignature = inputSignature(next);
    if (nextInputSignature === loadedInputSignature) {
      draw();
      return;
    }
    loadedInputSignature = nextInputSignature;
    imageLoadGeneration += 1;
    baseImg = null;
    maskImg = null;
    maskCanvas = null;
    tintCanvas = null;
    resolvedSourceWidth = positiveInteger(next.source_width, 0);
    resolvedSourceHeight = positiveInteger(next.source_height, 0);
    const generation = imageLoadGeneration;
    const nextBaseSource = baseSource();
    const nextMaskSource = maskSource();
    loadImage(nextBaseSource, generation, (image) => {
      baseImg = image;
      resolveSourceDimensions();
      ensureCanvases();
      draw();
    });
    loadImage(nextMaskSource, generation, (image) => {
      maskImg = image;
      resolveSourceDimensions();
      buildMaskCanvases(maskImg);
      draw();
    });
    if (!nextBaseSource && !nextMaskSource) {
      ensureCanvases();
      draw();
    }
  }

  function onPointerDown(event: PointerEvent): void {
    if (event.button !== 0 || activePointerId !== null) return;
    const point = eventToImage(event, false);
    if (!point) return;
    ensureCanvases();
    event.preventDefault();
    activePointerId = event.pointerId;
    pointerStart = point;
    pointerLast = point;
    pointerSnapshot = cloneSnapshot();
    pointerDirty = false;
    draftRect = null;
    surfaceEl.setPointerCapture(event.pointerId);
    if (activeTool() === "brush" || activeTool() === "eraser") {
      drawStroke(point, point);
      pointerDirty = true;
      draw();
    }
  }

  function onPointerMove(event: PointerEvent): void {
    if (event.pointerId !== activePointerId || !pointerStart || !pointerLast) return;
    event.preventDefault();
    const point = eventToImage(event, true);
    if (!point) return;
    if (activeTool() === "brush" || activeTool() === "eraser") {
      drawStroke(pointerLast, point);
      pointerDirty = true;
      pointerLast = point;
    } else {
      draftRect = normalizeRect(pointerStart, point);
      pointerDirty = true;
    }
    draw();
  }

  function releasePointer(): void {
    const pointerId = activePointerId;
    activePointerId = null;
    if (pointerId !== null && surfaceEl?.hasPointerCapture(pointerId)) {
      surfaceEl.releasePointerCapture(pointerId);
    }
    pointerStart = null;
    pointerLast = null;
    pointerSnapshot = null;
    pointerDirty = false;
    draftRect = null;
  }

  function onPointerUp(event: PointerEvent): void {
    if (event.pointerId !== activePointerId || !pointerStart) return;
    event.preventDefault();
    const point = eventToImage(event, true) || pointerLast || pointerStart;
    if (activeTool() === "rect_add" || activeTool() === "rect_erase") {
      drawRectangle(normalizeRect(pointerStart, point));
    }
    const changed = pointerDirty;
    releasePointer();
    if (changed) commitValue();
    else draw();
  }

  function onPointerCancel(): void {
    if (activePointerId === null) return;
    restoreSnapshot(pointerSnapshot);
    releasePointer();
    draw();
  }

  function onWindowBlur(): void {
    onPointerCancel();
  }

  function observeSurface(node: HTMLDivElement): { destroy: () => void } {
    resizeObserver = new ResizeObserver(scheduleResizeDraw);
    resizeObserver.observe(node);
    scheduleResizeDraw();
    return {
      destroy: () => {
        resizeObserver?.disconnect();
        resizeObserver = null;
      },
    };
  }

  function surfaceStyle(): string {
    return "height:" + heightStyle(gradio.props.height);
  }

  $effect(() => {
    const signature = JSON.stringify(gradio.props.value || null);
    if (signature !== lastSignature) {
      lastSignature = signature;
      ingestValue(gradio.props.value);
    }
  });

  onMount(() => {
    window.addEventListener("blur", onWindowBlur);
    resizeDisplayCanvas();
    draw();
    return () => window.removeEventListener("blur", onWindowBlur);
  });

  onDestroy(() => {
    imageLoadGeneration += 1;
    if (resizeFrame !== null) cancelAnimationFrame(resizeFrame);
    resizeObserver?.disconnect();
  });
</script>

<Block
  visible={gradio.shared.visible}
  variant="solid"
  border_mode="base"
  padding={false}
  elem_id={gradio.shared.elem_id}
  elem_classes={gradio.shared.elem_classes}
  allow_overflow={false}
  container={gradio.shared.container}
  scale={gradio.shared.scale}
  min_width={gradio.shared.min_width}
>
  <StatusTracker
    autoscroll={gradio.shared.autoscroll}
    i18n={gradio.i18n}
    {...gradio.shared.loading_status}
    on_clear_status={() => gradio.dispatch("clear_status", gradio.shared.loading_status)}
  />
  <div class="repair-editor" style={surfaceStyle()}>
    <div
      class="surface"
      bind:this={surfaceEl}
      use:observeSurface
      role="application"
      aria-label="Repair mask canvas"
      onpointerdown={onPointerDown}
      onpointermove={onPointerMove}
      onpointerup={onPointerUp}
      onpointercancel={onPointerCancel}
      onlostpointercapture={onPointerCancel}
    >
      <canvas bind:this={canvasEl}></canvas>
    </div>
  </div>
</Block>

<style>
  .repair-editor {
    box-sizing: border-box;
    width: 100%;
    min-width: 0;
    max-width: 100%;
    overflow: hidden;
    background: #f8fafc;
  }

  .surface {
    position: relative;
    box-sizing: border-box;
    height: 100%;
    width: 100%;
    min-width: 0;
    min-height: 180px;
    max-width: 100%;
    overflow: hidden;
    border: 1px solid #cbd5e1;
    background: #e2e8f0;
    touch-action: none;
    user-select: none;
  }

  canvas {
    display: block;
    width: 100%;
    height: 100%;
    max-width: 100%;
    min-width: 0;
    min-height: 0;
    touch-action: none;
  }

</style>
