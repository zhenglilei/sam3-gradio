<svelte:options accessors={true} />

<script lang="ts">
	import { Block } from "@gradio/atoms";
	import { StatusTracker } from "@gradio/statustracker";
	import { Gradio } from "@gradio/utils";
	import type {
		LayoutRegionAnnotatorEvents,
		LayoutRegionAnnotatorProps,
		LayoutRegionAnnotatorValue,
		RegionClientIntent,
		ToolMode,
	} from "./types";

	const props = $props();
	const gradio = new Gradio<LayoutRegionAnnotatorEvents, LayoutRegionAnnotatorProps>(props);
	const MAX_CANVAS_SIDE = 2048;
	const MAX_LASSO_POINTS = 4096;
	const SAMPLE_DISTANCE_CSS_PX = 3;

	type Point = { x: number; y: number };

	let canvasEl: HTMLCanvasElement;
	let sourceImg: HTMLImageElement | null = null;
	let sourceMaskImg: HTMLImageElement | null = null;
	let sourceMaskTint: HTMLCanvasElement | null = null;
	let savedOverlayImg: HTMLImageElement | null = null;
	let draftOverlayImg: HTMLImageElement | null = null;
	let localValue = $state<LayoutRegionAnnotatorValue>({});
	let toolMode = $state<ToolMode>("browse");
	let statusText = $state("请先生成版图 binary mask");
	let drawing = $state(false);
	let openDraft = $state(false);
	let capped = $state(false);
	let points = $state<Point[]>([]);
	let activePointerId: number | null = null;
	let lastClientPoint: Point | null = null;
	let pointerStartNatural: Point | null = null;
	let freehandGesture = false;
	let lastSignature = "";
	let imageLoadGeneration = 0;
	let loadedIdentity = "";
	let activeDraftIdentity: string | null = null;
	let sourceLoadDone = false;
	let maskLoadDone = false;
	let imagesReady = $state(false);

	function cloneValue(value: LayoutRegionAnnotatorValue | null | undefined): LayoutRegionAnnotatorValue {
		return JSON.parse(JSON.stringify(value || {}));
	}

	function heightStyle(value: number | string | undefined): string {
		if (typeof value === "number") return `${value}px`;
		return value || "520px";
	}

	function clamp(value: number, low: number, high: number): number {
		return Math.max(low, Math.min(high, value));
	}

	function serverView() {
		return localValue.server_view || {};
	}

	function clientIntent(): RegionClientIntent {
		return localValue.client_intent || {};
	}

	function layoutIdentity(intent: RegionClientIntent = clientIntent()): string {
		const view = serverView();
		return JSON.stringify([
			intent.session_id || "",
			intent.layout_id || "",
			intent.source_mask_hash || "",
			Number(view.natural_width || 0),
			Number(view.natural_height || 0),
		]);
	}

	function naturalWidth(): number {
		return Math.max(1, Number(serverView().natural_width || sourceImg?.naturalWidth || 1));
	}

	function naturalHeight(): number {
		return Math.max(1, Number(serverView().natural_height || sourceImg?.naturalHeight || 1));
	}

	function renderScale(): number {
		return Math.min(1, MAX_CANVAS_SIDE / Math.max(naturalWidth(), naturalHeight()));
	}

	function loadImage(
		url: string | null | undefined,
		generation: number,
		callback: (image: HTMLImageElement | null) => void,
	): void {
		const finish = (image: HTMLImageElement | null) => {
			if (generation === imageLoadGeneration) callback(image);
		};
		if (!url) {
			finish(null);
			return;
		}
		const image = new Image();
		image.onload = () => finish(image);
		image.onerror = () => finish(null);
		image.src = url;
	}

	function refreshImageReadiness(generation: number, payloadStatus: string): void {
		if (generation !== imageLoadGeneration) return;
		imagesReady = sourceLoadDone && maskLoadDone && sourceImg !== null && sourceMaskImg !== null;
		if (sourceLoadDone && maskLoadDone) {
			statusText = imagesReady
				? payloadStatus
				: "当前 Layout 图像加载失败，套索已禁用";
		}
		draw();
	}

	function rebuildSourceMaskTint(): void {
		if (!sourceMaskImg) {
			sourceMaskTint = null;
			return;
		}
		const width = naturalWidth();
		const height = naturalHeight();
		const rawCanvas = document.createElement("canvas");
		rawCanvas.width = width;
		rawCanvas.height = height;
		const rawContext = rawCanvas.getContext("2d", { willReadFrequently: true });
		if (!rawContext) return;
		rawContext.imageSmoothingEnabled = false;
		rawContext.drawImage(sourceMaskImg, 0, 0, width, height);
		const raw = rawContext.getImageData(0, 0, width, height);
		const tint = document.createElement("canvas");
		tint.width = width;
		tint.height = height;
		const context = tint.getContext("2d");
		if (!context) return;
		const output = context.createImageData(width, height);
		for (let index = 0; index < raw.data.length; index += 4) {
			const luminance = Math.max(raw.data[index], raw.data[index + 1], raw.data[index + 2]);
			if (raw.data[index + 3] > 0 && luminance >= 128) {
				output.data[index] = 45;
				output.data[index + 1] = 160;
				output.data[index + 2] = 255;
				output.data[index + 3] = 80;
			}
		}
		context.putImageData(output, 0, 0);
		sourceMaskTint = tint;
	}

	function ingestValue(value: LayoutRegionAnnotatorValue | null): void {
		localValue = cloneValue(value);
		const intent = clientIntent();
		const view = serverView();
		const generation = ++imageLoadGeneration;
		const nextIdentity = layoutIdentity(intent);
		const identityChanged = nextIdentity !== loadedIdentity;
		loadedIdentity = nextIdentity;
		const payloadStatus = view.status || "Region 标注器已加载";

		toolMode = intent.tool_mode === "lasso" ? "lasso" : "browse";
		points = Array.isArray(intent.lasso_polygon)
			? intent.lasso_polygon
					.filter((point) => Array.isArray(point) && point.length === 2)
					.map((point) => ({ x: Number(point[0]), y: Number(point[1]) }))
			: [];
		if (activePointerId !== null && canvasEl?.hasPointerCapture(activePointerId)) {
			canvasEl.releasePointerCapture(activePointerId);
		}
		drawing = false;
		openDraft = false;
		capped = false;
		activePointerId = null;
		activeDraftIdentity = null;
		lastClientPoint = null;
		pointerStartNatural = null;
		freehandGesture = false;
		savedOverlayImg = null;
		draftOverlayImg = null;

		if (identityChanged || view.enabled !== true) {
			sourceImg = null;
			sourceMaskImg = null;
			sourceMaskTint = null;
			sourceLoadDone = false;
			maskLoadDone = false;
			imagesReady = false;
		} else {
			sourceLoadDone = sourceImg !== null;
			maskLoadDone = sourceMaskImg !== null;
			imagesReady = sourceLoadDone && maskLoadDone;
		}

		if (view.enabled !== true) {
			statusText = payloadStatus;
			draw();
			return;
		}
		statusText = imagesReady ? payloadStatus : "正在加载当前 Layout 图像…";
		draw();

		loadImage(view.source_image, generation, (image) => {
			sourceImg = image;
			sourceLoadDone = true;
			refreshImageReadiness(generation, payloadStatus);
		});
		loadImage(view.source_mask_image, generation, (image) => {
			sourceMaskImg = image;
			maskLoadDone = true;
			rebuildSourceMaskTint();
			refreshImageReadiness(generation, payloadStatus);
		});
		loadImage(view.saved_region_overlay_image, generation, (image) => {
			savedOverlayImg = image;
			draw();
		});
		loadImage(view.draft_region_overlay_image, generation, (image) => {
			draftOverlayImg = image;
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

	function updateBrowserValue(dispatchChange = false): void {
		localValue = {
			...localValue,
			client_intent: {
				...clientIntent(),
				tool_mode: toolMode,
				lasso_polygon: points.map((point) => [point.x, point.y]),
			},
		};
		gradio.props.value = localValue;
		lastSignature = JSON.stringify(localValue);
		if (dispatchChange) gradio.dispatch("input");
	}

	function setToolMode(mode: ToolMode): void {
		if (drawing || openDraft) cancelDraft("绘制已取消");
		toolMode = mode;
		statusText =
			mode === "lasso"
				? "按住拖动绘制自由线，松开后可继续点击添加直线段；点击完成后统一闭合"
				: "浏览模式：Canvas 只读";
		updateBrowserValue(false);
		draw();
	}

	function eventToNatural(event: PointerEvent): Point {
		const rect = canvasEl.getBoundingClientRect();
		return {
			x: clamp(((event.clientX - rect.left) / Math.max(1, rect.width)) * naturalWidth(), 0, naturalWidth() - 1),
			y: clamp(((event.clientY - rect.top) / Math.max(1, rect.height)) * naturalHeight(), 0, naturalHeight() - 1),
		};
	}

	function clearServerDraftLocally(): void {
		localValue = {
			...localValue,
			server_view: { ...serverView(), draft_region_overlay_image: "" },
		};
		draftOverlayImg = null;
	}

	function onPointerDown(event: PointerEvent): void {
		if (event.button !== 0 || toolMode !== "lasso" || serverView().enabled !== true) return;
		if (!imagesReady) {
			statusText = "当前 Layout 图像尚未加载完成，不能开始套索";
			draw();
			return;
		}
		event.preventDefault();
		if (openDraft && activeDraftIdentity !== loadedIdentity) {
			cancelDraft("Layout 已切换，Draft 已清除");
			return;
		}
		const point = eventToNatural(event);
		if (!openDraft) {
			clearServerDraftLocally();
			points = [point];
			activeDraftIdentity = loadedIdentity;
			capped = false;
		}
		drawing = true;
		freehandGesture = false;
		activePointerId = event.pointerId;
		pointerStartNatural = point;
		lastClientPoint = { x: event.clientX, y: event.clientY };
		canvasEl.setPointerCapture(event.pointerId);
		statusText = openDraft ? "松开添加直线顶点，或继续拖动绘制自由线条" : "正在绘制 Draft";
		updateBrowserValue(false);
		draw();
	}

	function onPointerMove(event: PointerEvent): void {
		if (!drawing || event.pointerId !== activePointerId || !lastClientPoint) return;
		if (activeDraftIdentity !== loadedIdentity) {
			cancelDraft("Layout 已切换，Draft 已清除");
			return;
		}
		event.preventDefault();
		const cssDistance = Math.hypot(event.clientX - lastClientPoint.x, event.clientY - lastClientPoint.y);
		if (cssDistance < SAMPLE_DISTANCE_CSS_PX) return;
		const point = eventToNatural(event);
		if (!freehandGesture) {
			freehandGesture = true;
			if (openDraft && pointerStartNatural) {
				const last = points[points.length - 1];
				if (
					(!last || last.x !== pointerStartNatural.x || last.y !== pointerStartNatural.y) &&
					points.length < MAX_LASSO_POINTS
				) {
					points = [...points, pointerStartNatural];
				}
			}
			openDraft = false;
		}
		if (points.length < MAX_LASSO_POINTS) {
			points = [...points, point];
		} else {
			points = [...points.slice(0, -1), point];
			capped = true;
		}
		lastClientPoint = { x: event.clientX, y: event.clientY };
		statusText = capped ? `已达到 ${MAX_LASSO_POINTS} 点上限` : `Draft 点数：${points.length}`;
		draw();
	}

	function appendFinalPoint(event: PointerEvent): void {
		const point = eventToNatural(event);
		const last = points[points.length - 1];
		if (last && last.x === point.x && last.y === point.y) return;
		if (points.length < MAX_LASSO_POINTS) points = [...points, point];
		else {
			points = [...points.slice(0, -1), point];
			capped = true;
		}
	}

	function finishPointer(event: PointerEvent, keepDraftIdentity = false): void {
		if (activePointerId !== null && canvasEl.hasPointerCapture(activePointerId)) {
			canvasEl.releasePointerCapture(activePointerId);
		}
		activePointerId = null;
		if (!keepDraftIdentity) activeDraftIdentity = null;
		lastClientPoint = null;
		pointerStartNatural = null;
		drawing = false;
		freehandGesture = false;
		event.preventDefault();
	}

	function uniquePointCount(): number {
		return new Set(points.map((point) => `${point.x.toFixed(4)},${point.y.toFixed(4)}`)).size;
	}

	function finishDraft(): void {
		if (!openDraft) return;
		if (activeDraftIdentity !== loadedIdentity) {
			cancelDraft("Layout 已切换，Draft 已清除");
			return;
		}
		if (points.length < 3 || uniquePointCount() < 3) {
			statusText = "套索至少需要 3 个不同点";
			draw();
			return;
		}
		openDraft = false;
		activeDraftIdentity = null;
		statusText = "正在生成权威 Draft 交集预览…";
		updateBrowserValue(true);
		draw();
	}

	function onPointerUp(event: PointerEvent): void {
		if (!drawing || event.pointerId !== activePointerId) return;
		if (activeDraftIdentity !== loadedIdentity) {
			cancelDraft("Layout 已切换，Draft 已清除");
			return;
		}
		if (freehandGesture) {
			appendFinalPoint(event);
			finishPointer(event, true);
			openDraft = true;
			statusText = capped
				? `已达到 ${MAX_LASSO_POINTS} 点上限，请点击“完成套索”`
				: `自由线段已保留；可继续拖动或点击添加直线段，最后点击“完成套索”`;
			updateBrowserValue(false);
			draw();
			return;
		}

		const continuingOpenDraft = openDraft;
		if (continuingOpenDraft) appendFinalPoint(event);
		finishPointer(event, true);
		openDraft = true;
		statusText = capped
			? `已达到 ${MAX_LASSO_POINTS} 点上限，请完成套索`
			: `Draft 点数：${points.length}；可继续拖动或点击添加直线段，最后点击“完成套索”`;
		updateBrowserValue(false);
		draw();
	}

	function cancelDraft(message = "Draft 已清除"): void {
		if (activePointerId !== null && canvasEl?.hasPointerCapture(activePointerId)) {
			canvasEl.releasePointerCapture(activePointerId);
		}
		activePointerId = null;
		activeDraftIdentity = null;
		lastClientPoint = null;
		pointerStartNatural = null;
		drawing = false;
		openDraft = false;
		freehandGesture = false;
		capped = false;
		points = [];
		clearServerDraftLocally();
		statusText = message;
		updateBrowserValue(false);
		draw();
	}

	function onPointerCancel(event: PointerEvent): void {
		if (event.pointerId !== activePointerId) return;
		cancelDraft("指针操作已取消，Draft 已清除");
	}

	function onWindowBlur(): void {
		if (drawing || openDraft) cancelDraft("窗口失焦，Draft 已清除");
	}

	function draw(): void {
		if (!canvasEl) return;
		const width = naturalWidth();
		const height = naturalHeight();
		const scale = renderScale();
		canvasEl.width = Math.max(1, Math.round(width * scale));
		canvasEl.height = Math.max(1, Math.round(height * scale));
		const context = canvasEl.getContext("2d");
		if (!context) return;
		context.setTransform(scale, 0, 0, scale, 0, 0);
		context.clearRect(0, 0, width, height);
		if (sourceImg) {
			context.imageSmoothingEnabled = true;
			context.drawImage(sourceImg, 0, 0, width, height);
		} else {
			context.fillStyle = "#f8fafc";
			context.fillRect(0, 0, width, height);
			context.fillStyle = "#64748b";
			context.font = "18px sans-serif";
			context.fillText(
				serverView().enabled === true ? "正在加载当前 Layout 图像…" : "请先生成版图 binary mask",
				24,
				42,
			);
		}
		context.imageSmoothingEnabled = false;
		if (sourceMaskTint) context.drawImage(sourceMaskTint, 0, 0, width, height);
		if (savedOverlayImg) context.drawImage(savedOverlayImg, 0, 0, width, height);
		if (draftOverlayImg) context.drawImage(draftOverlayImg, 0, 0, width, height);
		if (points.length > 0) {
			context.save();
			context.beginPath();
			context.moveTo(points[0].x, points[0].y);
			for (let index = 1; index < points.length; index += 1) context.lineTo(points[index].x, points[index].y);
			if (!drawing && !openDraft && points.length >= 3) context.closePath();
			context.setLineDash([8, 5]);
			context.lineWidth = Math.max(2, width / 700);
			context.strokeStyle = "rgba(255, 220, 0, 0.98)";
			context.stroke();
			if (openDraft) {
				context.setLineDash([]);
				context.fillStyle = "rgba(255, 220, 0, 0.98)";
				const radius = Math.max(2.5, width / 500);
				for (const point of points) {
					context.beginPath();
					context.arc(point.x, point.y, radius, 0, Math.PI * 2);
					context.fill();
				}
			}
			context.restore();
		}
	}
</script>

<svelte:window on:blur={onWindowBlur} />

<Block
	visible={gradio.shared.visible}
	variant="solid"
	border_mode={drawing ? "focus" : "base"}
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
	<div class="region-annotator" style={`min-height:${heightStyle(gradio.props.height)}`}>
		<div class="toolbar" role="group" aria-label="Region annotation tool">
			<button type="button" class:active={toolMode === "browse"} on:click={() => setToolMode("browse")}>浏览</button>
			<button type="button" class:active={toolMode === "lasso"} on:click={() => setToolMode("lasso")}>套索选择</button>
			<button type="button" class="finish" disabled={!openDraft || uniquePointCount() < 3} on:click={finishDraft}>完成套索</button>
			<button type="button" class="clear" on:click={() => cancelDraft()}>清除 Draft</button>
		</div>
		<div class="legend">
			<span><i class="draft"></i>黄色：Draft</span>
			<span><i class="saved"></i>绿色：Saved Region</span>
		</div>
		<div class="canvas-wrap">
			<canvas
				bind:this={canvasEl}
				on:pointerdown={onPointerDown}
				on:pointermove={onPointerMove}
				on:pointerup={onPointerUp}
				on:pointercancel={onPointerCancel}
				style={`cursor:${toolMode === "lasso" ? "crosshair" : "default"}`}
			></canvas>
		</div>
		<div class="status">{statusText} · 点数 {points.length}/{MAX_LASSO_POINTS}</div>
	</div>
</Block>

<style>
	.region-annotator {
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
	.toolbar button.active {
		border-color: #d4a900;
		background: #fff7cc;
		color: #715600;
	}
	.toolbar button.clear {
		color: #9f1239;
	}
	.toolbar button.finish {
		color: #166534;
	}
	.toolbar button:disabled {
		cursor: not-allowed;
		opacity: 0.45;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 14px;
		font-size: 12px;
		color: #475569;
	}
	.legend span {
		display: inline-flex;
		align-items: center;
		gap: 5px;
	}
	.legend i {
		display: inline-block;
		width: 12px;
		height: 12px;
		border-radius: 3px;
	}
	.legend .draft {
		background: rgba(255, 210, 0, 0.75);
	}
	.legend .saved {
		background: rgba(35, 200, 85, 0.6);
	}
	.canvas-wrap {
		display: flex;
		align-items: center;
		justify-content: center;
		min-height: 260px;
		background: #0f172a;
		border: 1px solid #cbd5e1;
		border-radius: 8px;
		overflow: hidden;
	}
	canvas {
		display: block;
		width: auto;
		height: auto;
		max-width: 100%;
		max-height: 680px;
		touch-action: none;
		user-select: none;
	}
	.status {
		min-height: 20px;
		font-size: 13px;
		color: #334155;
	}
</style>
