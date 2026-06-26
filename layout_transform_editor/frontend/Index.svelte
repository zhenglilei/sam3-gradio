<svelte:options accessors={true} />

<script lang="ts">
	import type { LayoutTransformEditorEvents, LayoutTransformEditorProps, LayoutTransform, LayoutTransformValue } from "./types";
	import { Gradio } from "@gradio/utils";
	import { Block } from "@gradio/atoms";
	import { StatusTracker } from "@gradio/statustracker";
	import { onDestroy } from "svelte";

	const props = $props();
	const MAX_CANVAS_SIDE = 2048;
	const gradio = new Gradio<LayoutTransformEditorEvents, LayoutTransformEditorProps>(props);

	let canvasEl: HTMLCanvasElement;
	let baseImg: HTMLImageElement | null = null;
	let maskImg: HTMLImageElement | null = null;
	let maskCanvas: HTMLCanvasElement | null = null;
	let tintCanvas: HTMLCanvasElement | null = null;
	let baseReady = $state(false);
	let maskReady = $state(false);
	let statusText = $state("等待版图 mask");
	let localValue = $state<LayoutTransformValue>({ enabled: false });
	let transform = $state<LayoutTransform>(defaultTransform());
	let dragging = $state(false);
	let rotating = $state(false);
	let cursorStyle = $state("crosshair");
	let lastSignature = "";
	let dragStart = { x: 0, y: 0, center_x: 0, center_y: 0 };
	let rotateStart = { angle: 0, rotation: 0 };
	let changeSyncTimer: ReturnType<typeof setTimeout> | null = null;

	function defaultTransform(): LayoutTransform {
		return {
			transform_version: 2,
			revision: 0,
			center_x: 0,
			center_y: 0,
			pivot_x: 0,
			pivot_y: 0,
			scale: 1,
			rotation_deg: 0,
			preview_alpha: 0.35,
		};
	}

	function cloneValue(value: LayoutTransformValue | null | undefined): LayoutTransformValue {
		return JSON.parse(JSON.stringify(value || { enabled: false }));
	}

	function heightStyle(value: number | string | undefined): string {
		if (typeof value === "number") return `${value}px`;
		return value || "520px";
	}

	function clamp(value: number, lo: number, hi: number): number {
		return Math.max(lo, Math.min(hi, value));
	}

	function normalizeRotation(value: number): number {
		let out = ((value + 180) % 360 + 360) % 360 - 180;
		if (out === -180) out = 180;
		return out;
	}

	function targetWidth(): number {
		return Math.max(1, Number(localValue.target_width || baseImg?.naturalWidth || 1));
	}

	function targetHeight(): number {
		return Math.max(1, Number(localValue.target_height || baseImg?.naturalHeight || 1));
	}

	function sourceWidth(): number {
		return Math.max(1, Number(localValue.source_width || maskImg?.naturalWidth || 1));
	}

	function sourceHeight(): number {
		return Math.max(1, Number(localValue.source_height || maskImg?.naturalHeight || 1));
	}

	function canvasRenderScale(): number {
		return Math.min(1, MAX_CANVAS_SIDE / Math.max(targetWidth(), targetHeight()));
	}

	function foregroundBbox(): number[] {
		const bbox = localValue.foreground_bbox_xyxy;
		if (Array.isArray(bbox) && bbox.length >= 4) return bbox.map(Number);
		return [0, 0, sourceWidth() - 1, sourceHeight() - 1];
	}

	function loadImage(src: string | null | undefined, onload: (img: HTMLImageElement | null) => void): void {
		if (!src) {
			onload(null);
			return;
		}
		const img = new Image();
		img.onload = () => onload(img);
		img.onerror = () => onload(null);
		img.src = src;
	}

	function rebuildMaskCanvases(): void {
		if (!maskImg) {
			maskCanvas = null;
			tintCanvas = null;
			return;
		}
		const sw = sourceWidth();
		const sh = sourceHeight();
		maskCanvas = document.createElement("canvas");
		maskCanvas.width = sw;
		maskCanvas.height = sh;
		const rawCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
		if (!rawCtx) return;
		rawCtx.imageSmoothingEnabled = false;
		rawCtx.drawImage(maskImg, 0, 0, sw, sh);
		const raw = rawCtx.getImageData(0, 0, sw, sh);
		tintCanvas = document.createElement("canvas");
		tintCanvas.width = sw;
		tintCanvas.height = sh;
		const tintCtx = tintCanvas.getContext("2d");
		if (!tintCtx) return;
		const tinted = tintCtx.createImageData(sw, sh);
		for (let i = 0; i < raw.data.length; i += 4) {
			const lum = Math.max(raw.data[i], raw.data[i + 1], raw.data[i + 2]);
			if (raw.data[i + 3] > 0 && lum >= 128) {
				tinted.data[i] = 0;
				tinted.data[i + 1] = 255;
				tinted.data[i + 2] = 120;
				tinted.data[i + 3] = 255;
			}
		}
		tintCtx.putImageData(tinted, 0, 0);
	}

	function clearQueuedSync(): void {
		if (changeSyncTimer) {
			clearTimeout(changeSyncTimer);
			changeSyncTimer = null;
		}
	}

	function ingestValue(value: LayoutTransformValue | null): void {
		clearQueuedSync();
		localValue = cloneValue(value);
		transform = { ...defaultTransform(), ...(localValue.transform || {}) };
		statusText = localValue.status || "编辑器已加载";
		baseReady = false;
		maskReady = false;
		loadImage(localValue.base_image, (img) => {
			baseImg = img;
			baseReady = !!img;
			draw();
		});
		loadImage(localValue.mask_image, (img) => {
			maskImg = img;
			maskReady = !!img;
			rebuildMaskCanvases();
			draw();
		});
	}

	$effect(() => {
		const signature = JSON.stringify(gradio.props.value || null);
		if (signature !== lastSignature) {
			lastSignature = signature;
			ingestValue(gradio.props.value);
		}
	});

	onDestroy(() => {
		clearQueuedSync();
	});

	function matrix(t: LayoutTransform): [number, number, number, number, number, number] {
		const theta = (Number(t.rotation_deg || 0) * Math.PI) / 180;
		const s = Number(t.scale || 1);
		const cosT = Math.cos(theta);
		const sinT = Math.sin(theta);
		const a = s * cosT;
		const b = s * sinT;
		const e = Number(t.center_x || 0) - a * Number(t.pivot_x || 0) + b * Number(t.pivot_y || 0);
		const f = Number(t.center_y || 0) - b * Number(t.pivot_x || 0) - a * Number(t.pivot_y || 0);
		return [a, b, -b, a, e, f];
	}

	function sourceToTarget(x: number, y: number, t: LayoutTransform = transform): { x: number; y: number } {
		const [a, b, c, d, e, f] = matrix(t);
		return { x: a * x + c * y + e, y: b * x + d * y + f };
	}

	function targetToSource(x: number, y: number, t: LayoutTransform = transform): { x: number; y: number } {
		const [a, b, c, d, e, f] = matrix(t);
		const det = a * d - b * c;
		if (Math.abs(det) < 1e-9) return { x: -1, y: -1 };
		const dx = x - e;
		const dy = y - f;
		return { x: (d * dx - c * dy) / det, y: (-b * dx + a * dy) / det };
	}

	function eventToTarget(evt: PointerEvent | WheelEvent): { x: number; y: number } {
		const rect = canvasEl.getBoundingClientRect();
		return {
			x: ((evt.clientX - rect.left) / Math.max(1, rect.width)) * targetWidth(),
			y: ((evt.clientY - rect.top) / Math.max(1, rect.height)) * targetHeight(),
		};
	}

	function maskForegroundAt(sourceX: number, sourceY: number): boolean {
		if (!maskCanvas) return false;
		const x = Math.round(sourceX);
		const y = Math.round(sourceY);
		if (x < 0 || y < 0 || x >= maskCanvas.width || y >= maskCanvas.height) return false;
		const ctx = maskCanvas.getContext("2d", { willReadFrequently: true });
		if (!ctx) return false;
		const data = ctx.getImageData(x, y, 1, 1).data;
		return data[3] > 0 && Math.max(data[0], data[1], data[2]) >= 128;
	}

	function hitMask(targetX: number, targetY: number): boolean {
		const p = targetToSource(targetX, targetY);
		return maskForegroundAt(p.x, p.y);
	}

	function bboxCorners(): { x: number; y: number }[] {
		const [x1, y1, x2, y2] = foregroundBbox();
		return [sourceToTarget(x1, y1), sourceToTarget(x2, y1), sourceToTarget(x2, y2), sourceToTarget(x1, y2)];
	}

	function rotateHandleCenter(): { x: number; y: number } {
		const corners = bboxCorners();
		let corner = corners[0];
		for (const p of corners) {
			if (p.y < corner.y || (Math.abs(p.y - corner.y) < 1e-6 && p.x > corner.x)) corner = p;
		}
		const angle = (Number(transform.rotation_deg || 0) * Math.PI) / 180;
		const ux = Math.cos(angle - Math.PI / 4);
		const uy = Math.sin(angle - Math.PI / 4);
		return { x: corner.x + ux * 34, y: corner.y + uy * 34 };
	}

	function handleRadius(): number {
		const rect = canvasEl?.getBoundingClientRect();
		if (!rect) return 14;
		return Math.max(8, (14 * targetWidth()) / Math.max(1, rect.width));
	}

	function hitRotateHandle(targetX: number, targetY: number): boolean {
		const h = rotateHandleCenter();
		const r = handleRadius();
		return Math.hypot(targetX - h.x, targetY - h.y) <= r;
	}

	function bumpRevision(origin: string): void {
		transform = {
			...transform,
			revision: Number(transform.revision || 0) + 1,
			origin,
			scale: clamp(Number(transform.scale || 1), 0.01, 20),
			rotation_deg: normalizeRotation(Number(transform.rotation_deg || 0)),
		};
	}

	function syncValue(origin: string, status: string, dispatchChange = true): void {
		bumpRevision(origin);
		localValue = {
			...localValue,
			enabled: true,
			transform: { ...transform },
			status,
		};
		statusText = status;
		gradio.props.value = localValue;
		lastSignature = JSON.stringify(localValue);
		if (dispatchChange) {
			clearQueuedSync();
			gradio.dispatch("change");
		}
		draw();
	}

	function queueSyncValue(origin: string, status: string, delay = 140): void {
		syncValue(origin, status, false);
		clearQueuedSync();
		changeSyncTimer = setTimeout(() => {
			changeSyncTimer = null;
			gradio.dispatch("change");
		}, delay);
	}

	function draw(): void {
		if (!canvasEl) return;
		const tw = targetWidth();
		const th = targetHeight();
		const rs = canvasRenderScale();
		canvasEl.width = Math.max(1, Math.round(tw * rs));
		canvasEl.height = Math.max(1, Math.round(th * rs));
		const ctx = canvasEl.getContext("2d");
		if (!ctx) return;
		ctx.setTransform(rs, 0, 0, rs, 0, 0);
		ctx.clearRect(0, 0, tw, th);
		if (baseImg && baseReady) {
			ctx.imageSmoothingEnabled = true;
			ctx.drawImage(baseImg, 0, 0, tw, th);
		} else {
			ctx.fillStyle = "#f8fafc";
			ctx.fillRect(0, 0, tw, th);
			ctx.fillStyle = "#64748b";
			ctx.font = "18px sans-serif";
			ctx.fillText("请先上传图像", 24, 42);
		}
		if (tintCanvas && maskReady && localValue.enabled !== false) {
			ctx.save();
			ctx.globalAlpha = clamp(Number(transform.preview_alpha || 0.35), 0, 1);
			const [a, b, c, d, e, f] = matrix(transform);
			ctx.setTransform(rs * a, rs * b, rs * c, rs * d, rs * e, rs * f);
			ctx.imageSmoothingEnabled = false;
			ctx.drawImage(tintCanvas, 0, 0, sourceWidth(), sourceHeight());
			ctx.restore();
			const corners = bboxCorners();
			ctx.save();
			ctx.lineJoin = "round";
			ctx.lineWidth = Math.max(2.5, tw / 700);
			ctx.strokeStyle = "rgba(0,0,0,0.82)";
			ctx.beginPath();
			ctx.moveTo(corners[0].x, corners[0].y);
			for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i].x, corners[i].y);
			ctx.closePath();
			ctx.stroke();
			ctx.lineWidth = Math.max(1.8, tw / 1000);
			ctx.strokeStyle = "#00ff66";
			ctx.stroke();
			const h = rotateHandleCenter();
			const top = corners.reduce((best, p) => (p.y < best.y || (Math.abs(p.y - best.y) < 1e-6 && p.x > best.x) ? p : best), corners[0]);
			ctx.strokeStyle = "#0f172a";
			ctx.lineWidth = Math.max(2, tw / 900);
			ctx.beginPath();
			ctx.moveTo(top.x, top.y);
			ctx.lineTo(h.x, h.y);
			ctx.stroke();
			ctx.fillStyle = rotating ? "#ffb000" : "#ffffff";
			ctx.strokeStyle = "#00ff66";
			ctx.lineWidth = Math.max(2, tw / 900);
			ctx.beginPath();
			ctx.arc(h.x, h.y, handleRadius(), 0, Math.PI * 2);
			ctx.fill();
			ctx.stroke();
			ctx.restore();
		}
	}

	function onPointerDown(evt: PointerEvent): void {
		if (evt.button !== 0) return;
		if (!localValue.enabled || !maskReady || !maskCanvas) {
			statusText = "请先启用并加载版图 mask";
			draw();
			return;
		}
		clearQueuedSync();
		const p = eventToTarget(evt);
		if (hitRotateHandle(p.x, p.y)) {
			cursorStyle = "grabbing";
			rotating = true;
			rotateStart = { angle: Math.atan2(p.y - transform.center_y, p.x - transform.center_x) * 180 / Math.PI, rotation: Number(transform.rotation_deg || 0) };
			canvasEl.setPointerCapture(evt.pointerId);
			return;
		}
		if (!hitMask(p.x, p.y)) {
			statusText = "请点中版图 mask 前景后拖动";
			draw();
			return;
		}
		cursorStyle = "grabbing";
		dragging = true;
		dragStart = { x: p.x, y: p.y, center_x: Number(transform.center_x || 0), center_y: Number(transform.center_y || 0) };
		canvasEl.setPointerCapture(evt.pointerId);
	}

	function updateCursor(p: { x: number; y: number }): void {
		if (!localValue.enabled || !maskReady || !maskCanvas) {
			cursorStyle = "not-allowed";
			return;
		}
		if (hitRotateHandle(p.x, p.y) || hitMask(p.x, p.y)) {
			cursorStyle = "grab";
			return;
		}
		cursorStyle = "crosshair";
	}

	function onPointerMove(evt: PointerEvent): void {
		const p = eventToTarget(evt);
		if (!dragging && !rotating) {
			updateCursor(p);
			return;
		}
		cursorStyle = "grabbing";
		if (dragging) {
			transform = { ...transform, center_x: dragStart.center_x + p.x - dragStart.x, center_y: dragStart.center_y + p.y - dragStart.y };
			statusText = "正在拖动；松开后同步变换";
		} else if (rotating) {
			const angle = Math.atan2(p.y - transform.center_y, p.x - transform.center_x) * 180 / Math.PI;
			transform = { ...transform, rotation_deg: normalizeRotation(rotateStart.rotation + angle - rotateStart.angle) };
			statusText = "正在旋转；松开后同步变换";
		}
		draw();
	}

	function onPointerLeave(): void {
		if (!dragging && !rotating) cursorStyle = "crosshair";
	}

	function onPointerUp(evt: PointerEvent): void {
		if (dragging || rotating) {
			dragging = false;
			rotating = false;
			try { canvasEl.releasePointerCapture(evt.pointerId); } catch (_) {}
			updateCursor(eventToTarget(evt));
			syncValue("canvas", "Canvas 变换已同步；点击更新预览或创建实例后使用后端权威 mask");
		}
	}

	function onWheel(evt: WheelEvent): void {
		if (!localValue.enabled || !maskReady) return;
		evt.preventDefault();
		const p = eventToTarget(evt);
		const before = targetToSource(p.x, p.y);
		const factor = Math.exp(-evt.deltaY * 0.001);
		const nextScale = clamp(Number(transform.scale || 1) * factor, 0.01, 20);
		let next = { ...transform, scale: nextScale };
		const afterTarget = sourceToTarget(before.x, before.y, next);
		next = { ...next, center_x: Number(next.center_x || 0) + p.x - afterTarget.x, center_y: Number(next.center_y || 0) + p.y - afterTarget.y };
		transform = next;
		queueSyncValue("canvas", `滚轮缩放已同步: scale=${nextScale.toFixed(3)}`);
	}

	function resetTransform(): void {
		transform = { ...transform, center_x: targetWidth() / 2, center_y: targetHeight() / 2, scale: 1, rotation_deg: 0 };
		syncValue("reset", "重置：已居中，scale=1，rotation=0");
	}

	function centerTransform(): void {
		transform = { ...transform, center_x: targetWidth() / 2, center_y: targetHeight() / 2 };
		syncValue("center", "居中：保留缩放和旋转");
	}

	function fitTransform(): void {
		const [x1, y1, x2, y2] = foregroundBbox();
		const bw = Math.max(1, x2 - x1 + 1);
		const bh = Math.max(1, y2 - y1 + 1);
		const scale = clamp(Math.min(targetWidth() / bw, targetHeight() / bh) * 0.9, 0.01, 20);
		transform = { ...transform, center_x: targetWidth() / 2, center_y: targetHeight() / 2, scale };
		syncValue("fit", `适配：scale=${scale.toFixed(3)}`);
	}

	function bringIntoView(): void {
		const corners = bboxCorners();
		const minX = Math.min(...corners.map((p) => p.x));
		const maxX = Math.max(...corners.map((p) => p.x));
		const minY = Math.min(...corners.map((p) => p.y));
		const maxY = Math.max(...corners.map((p) => p.y));
		let dx = 0;
		let dy = 0;
		const pad = Math.max(20, targetWidth() * 0.03);
		if (maxX < pad) dx = pad - maxX;
		else if (minX > targetWidth() - pad) dx = targetWidth() - pad - minX;
		if (maxY < pad) dy = pad - maxY;
		else if (minY > targetHeight() - pad) dy = targetHeight() - pad - minY;
		if (dx === 0 && dy === 0) {
			statusText = "版图已经在视野内";
			return;
		}
		transform = { ...transform, center_x: Number(transform.center_x || 0) + dx, center_y: Number(transform.center_y || 0) + dy };
		syncValue("bring_into_view", "已找回到视野内");
	}
</script>

<Block
	visible={gradio.shared.visible}
	variant="solid"
	border_mode={dragging || rotating ? "focus" : "base"}
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
	<div class="layout-editor" style={`min-height:${heightStyle(gradio.props.height)}`}>
		<div class="toolbar">
			<button type="button" on:click={resetTransform}>重置</button>
			<button type="button" on:click={centerTransform}>居中</button>
			<button type="button" on:click={fitTransform}>适配</button>
			<button type="button" on:click={bringIntoView}>找回视野</button>
		</div>
		<div class="canvas-wrap">
			<canvas
				bind:this={canvasEl}
				on:pointerdown={onPointerDown}
				on:pointermove={onPointerMove}
				on:pointerup={onPointerUp}
				on:pointercancel={onPointerUp}
				on:pointerleave={onPointerLeave}
				on:wheel={onWheel}
				style={`cursor:${cursorStyle}`}
			></canvas>
		</div>
		<div class="status">{statusText}</div>
	</div>
</Block>

<style>
	.layout-editor {
		display: flex;
		flex-direction: column;
		gap: 8px;
		background: #f8fafc;
		padding: 10px;
	}
	.toolbar {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 8px;
	}
	.toolbar button {
		border: 1px solid #cbd5e1;
		background: #ffffff;
		border-radius: 6px;
		padding: 7px 8px;
		font-weight: 600;
		color: #0f172a;
		cursor: pointer;
	}
	.toolbar button:hover {
		border-color: #2563eb;
		color: #1d4ed8;
	}
	.canvas-wrap {
		flex: 1;
		min-height: 280px;
		overflow: auto;
		border: 1px solid #cbd5e1;
		background: #0f172a;
	}
	canvas {
		display: block;
		width: 100%;
		height: auto;
		user-select: none;
		touch-action: none;
	}
	.status {
		font-size: 12px;
		line-height: 1.35;
		color: #475569;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 6px;
		padding: 6px 8px;
		min-height: 28px;
	}
</style>
