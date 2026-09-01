<svelte:options accessors={true} />

<script lang="ts">
	import type {
		StitchPreviewCanvasEvents,
		StitchPreviewCanvasProps,
		StitchPreviewValue,
		StitchTile,
	} from "./types";
	import { Gradio } from "@gradio/utils";
	import { Block } from "@gradio/atoms";
	import { StatusTracker } from "@gradio/statustracker";
	import { onDestroy, onMount } from "svelte";

	interface TileRuntime {
		tile: StitchTile;
		image: HTMLImageElement | null;
		ready: boolean;
		baseX: number;
		baseY: number;
	}

	interface UndoSnapshot {
		tiles: { index: number; x: number; y: number }[];
		selected: number;
	}

	const props = $props();
	const DEADZONE_PX = 4;
	const LOUPE_RADIUS = 70;
	const LOUPE_SCALE = 3;
	const UNDO_LIMIT = 30;
	const WHEEL_ZOOM_SPEED = 0.001;
	const DEFAULT_DRAG_GAIN = 1.0;
	const PRECISE_DRAG_GAIN = 0.25;
	const gradio = new Gradio<StitchPreviewCanvasEvents, StitchPreviewCanvasProps>(props);

	let canvasEl: HTMLCanvasElement;
	let wrapEl: HTMLDivElement;
	let localValue = $state<StitchPreviewValue>({ tiles: [], selected: 0 });
	let tileRuntimes = $state<TileRuntime[]>([]);
	let statusText = $state("点击画布以启用键盘");
	let focused = $state(false);
	let cursorStyle = $state("crosshair");
	let viewPanX = $state(0);
	let viewPanY = $state(0);
	let viewZoom = $state(1);
	let spaceDown = $state(false);
	let panning = $state(false);
	let dragging = $state(false);
	let dragTileIndex = $state(-1);
	let dragOriginX = 0;
	let dragOriginY = 0;
	let dragGhostX = 0;
	let dragGhostY = 0;
	let pointerDownX = 0;
	let pointerDownY = 0;
	let pointerLastX = 0;
	let pointerLastY = 0;
	let dragMoved = false;
	let dragShift = false;
	let pointerIdActive = -1;
	let pointerDownSelected = 0;
	let loupeX = 0;
	let loupeY = 0;
	let showLoupePointer = false;
	let lastSignature = "";
	let imageLoadGeneration = 0;
	let changeSyncTimer: ReturnType<typeof setTimeout> | null = null;
	let undoStack: UndoSnapshot[] = [];
	let undoIndex = -1;

	function heightStyle(value: number | string | undefined): string {
		if (typeof value === "number") return String(value) + "px";
		return value || "520px";
	}

	function finiteNumber(value: unknown, fallback: number): number {
		const numberValue = Number(value);
		return Number.isFinite(numberValue) ? numberValue : fallback;
	}

	function clamp(value: number, lo: number, hi: number): number {
		return Math.max(lo, Math.min(hi, value));
	}

	function cloneValue(value: StitchPreviewValue | null | undefined): StitchPreviewValue {
		return JSON.parse(JSON.stringify(value || { tiles: [], selected: 0 }));
	}

	function normalizeTile(tile: StitchTile): StitchTile {
		return {
			index: Math.trunc(finiteNumber(tile.index, 0)),
			image: tile.image ?? null,
			x: finiteNumber(tile.x, 0),
			y: finiteNumber(tile.y, 0),
			width: Math.max(1, finiteNumber(tile.width, 1)),
			height: Math.max(1, finiteNumber(tile.height, 1)),
		};
	}

	function selectedIndex(): number {
		return Math.trunc(finiteNumber(localValue.selected, 0));
	}

	function nudgeStep(): number {
		const step = finiteNumber(localValue.nudge_step, 1);
		return step > 0 ? step : 1;
	}

	function dragGain(): number {
		const gain = finiteNumber(localValue.drag_gain, DEFAULT_DRAG_GAIN);
		return gain > 0 ? gain : DEFAULT_DRAG_GAIN;
	}

	function preciseDragGain(): number {
		return Math.min(dragGain(), PRECISE_DRAG_GAIN);
	}

	function diffMode(): boolean {
		return !!localValue.diff_mode;
	}

	function showLoupe(): boolean {
		return localValue.show_loupe !== false;
	}

	function selectedTile(): StitchTile | null {
		const idx = selectedIndex();
		for (const runtime of tileRuntimes) {
			if (runtime.tile.index === idx) return runtime.tile;
		}
		return tileRuntimes[0]?.tile ?? null;
	}

	function selectedOffset(): { dx: number; dy: number } {
		const tile = selectedTile();
		if (!tile) return { dx: 0, dy: 0 };
		const runtime = tileRuntimes.find((item) => item.tile.index === tile.index);
		if (!runtime) return { dx: 0, dy: 0 };
		return {
			dx: Math.round(tile.x - runtime.baseX),
			dy: Math.round(tile.y - runtime.baseY),
		};
	}

	function currentDragGain(): number {
		if (dragging && dragShift) return preciseDragGain();
		return dragGain();
	}

	function canvasSize(): { width: number; height: number } {
		const rect = canvasEl?.getBoundingClientRect();
		return {
			width: Math.max(1, rect?.width || 1),
			height: Math.max(1, rect?.height || 1),
		};
	}

	function screenToWorld(sx: number, sy: number): { x: number; y: number } {
		return {
			x: (sx - viewPanX) / viewZoom,
			y: (sy - viewPanY) / viewZoom,
		};
	}

	function worldToScreen(wx: number, wy: number): { x: number; y: number } {
		return {
			x: wx * viewZoom + viewPanX,
			y: wy * viewZoom + viewPanY,
		};
	}

	function eventToScreen(evt: MouseEvent | PointerEvent | WheelEvent): { x: number; y: number } {
		const rect = canvasEl.getBoundingClientRect();
		return {
			x: evt.clientX - rect.left,
			y: evt.clientY - rect.top,
		};
	}

	function tileAlpha(index: number, activeDrag: boolean): number {
		if (activeDrag) return 90 / 255;
		if (index === 0) return 1;
		return 140 / 255;
	}

	function loadImage(
		src: string | null | undefined,
		generation: number,
		onload: (img: HTMLImageElement | null) => void,
	): void {
		if (!src) {
			if (generation === imageLoadGeneration) onload(null);
			return;
		}
		const img = new Image();
		img.onload = () => {
			if (generation === imageLoadGeneration) onload(img);
		};
		img.onerror = () => {
			if (generation === imageLoadGeneration) onload(null);
		};
		img.src = src;
	}

	function clearQueuedSync(): void {
		if (changeSyncTimer) {
			clearTimeout(changeSyncTimer);
			changeSyncTimer = null;
		}
	}

	function snapshotTiles(): UndoSnapshot {
		return {
			tiles: tileRuntimes.map((runtime) => ({
				index: runtime.tile.index,
				x: runtime.tile.x,
				y: runtime.tile.y,
			})),
			selected: selectedIndex(),
		};
	}

	function snapshotsEqual(a: UndoSnapshot, b: UndoSnapshot): boolean {
		return (
			a.selected === b.selected &&
			a.tiles.length === b.tiles.length &&
			a.tiles.every((tile, i) => {
				const other = b.tiles[i];
				return (
					tile.index === other.index &&
					Math.abs(tile.x - other.x) < 0.01 &&
					Math.abs(tile.y - other.y) < 0.01
				);
			})
		);
	}

	function beginMutation(): void {
		const snap = snapshotTiles();
		if (undoIndex >= 0 && snapshotsEqual(undoStack[undoIndex], snap)) return;
		undoStack = undoStack.slice(0, undoIndex + 1);
		undoStack.push(snap);
		if (undoStack.length > UNDO_LIMIT) undoStack.shift();
		undoIndex = undoStack.length - 1;
	}

	function endMutation(): void {
		const snap = snapshotTiles();
		if (undoIndex >= 0 && snapshotsEqual(undoStack[undoIndex], snap)) return;
		undoStack = undoStack.slice(0, undoIndex + 1);
		undoStack.push(snap);
		if (undoStack.length > UNDO_LIMIT) undoStack.shift();
		undoIndex = undoStack.length - 1;
	}

	function applySnapshot(snap: UndoSnapshot): void {
		for (const item of snap.tiles) {
			const runtime = tileRuntimes.find((entry) => entry.tile.index === item.index);
			if (runtime) {
				runtime.tile.x = item.x;
				runtime.tile.y = item.y;
			}
		}
		localValue = { ...localValue, selected: snap.selected };
		tileRuntimes = [...tileRuntimes];
	}

	function fitView(): void {
		if (!tileRuntimes.length) {
			viewPanX = 20;
			viewPanY = 20;
			viewZoom = 1;
			return;
		}
		let minX = Infinity;
		let minY = Infinity;
		let maxX = -Infinity;
		let maxY = -Infinity;
		for (const runtime of tileRuntimes) {
			const tile = runtime.tile;
			minX = Math.min(minX, tile.x);
			minY = Math.min(minY, tile.y);
			maxX = Math.max(maxX, tile.x + tile.width);
			maxY = Math.max(maxY, tile.y + tile.height);
		}
		const pad = 40;
		const worldW = Math.max(1, maxX - minX);
		const worldH = Math.max(1, maxY - minY);
		const { width, height } = canvasSize();
		const zoom = clamp(Math.min((width - pad * 2) / worldW, (height - pad * 2) / worldH), 0.05, 8);
		viewZoom = zoom;
		viewPanX = (width - (minX + maxX) * zoom) / 2;
		viewPanY = (height - (minY + maxY) * zoom) / 2;
	}

	function ingestValue(value: StitchPreviewValue | null): void {
		clearQueuedSync();
		imageLoadGeneration += 1;
		const generation = imageLoadGeneration;
		const prevByIndex = new Map(
			tileRuntimes.map((runtime) => [runtime.tile.index, runtime]),
		);
		const incoming = cloneValue(value);
		const tiles = Array.isArray(incoming.tiles)
			? incoming.tiles.map((tile) => {
					const normalized = normalizeTile(tile);
					if (!normalized.image) {
						const previous = prevByIndex.get(normalized.index);
						if (previous?.tile.image) normalized.image = previous.tile.image;
					}
					return normalized;
				})
			: [];
		localValue = {
			...incoming,
			tiles,
			selected: tiles.length ? Math.trunc(finiteNumber(incoming.selected, tiles[0].index)) : 0,
			nudge_step: finiteNumber(incoming.nudge_step, 1),
			diff_mode: !!incoming.diff_mode,
			show_loupe: incoming.show_loupe !== false,
			drag_gain: finiteNumber(incoming.drag_gain, DEFAULT_DRAG_GAIN),
			status: incoming.status || "",
		};
		dragging = false;
		dragTileIndex = -1;
		panning = false;
		undoStack = [];
		undoIndex = -1;
		const runtimes: TileRuntime[] = tiles.map((tile) => ({
			tile: { ...tile },
			image: null,
			ready: false,
			baseX: tile.x,
			baseY: tile.y,
		}));
		tileRuntimes = runtimes;
		if (runtimes.length) endMutation();
		statusText = localValue.status || (tiles.length ? "点击画布以启用键盘" : "等待 tile 数据");
		for (let i = 0; i < runtimes.length; i++) {
			const runtime = runtimes[i];
			loadImage(runtime.tile.image, generation, (img) => {
				if (generation !== imageLoadGeneration) return;
				const current = tileRuntimes[i];
				if (!current || current.tile.index !== runtime.tile.index) return;
				current.image = img;
				current.ready = !!img;
				tileRuntimes = [...tileRuntimes];
				draw();
			});
		}
		requestAnimationFrame(() => fitView());
	}

	$effect(() => {
		const signature = JSON.stringify(gradio.props.value || null);
		if (signature !== lastSignature) {
			lastSignature = signature;
			ingestValue(gradio.props.value);
		}
	});

	onDestroy(() => {
		imageLoadGeneration += 1;
		clearQueuedSync();
	});

	onMount(() => {
		window.addEventListener("blur", onWindowBlur);
		return () => window.removeEventListener("blur", onWindowBlur);
	});

	function publishClientValue(status?: string): void {
		const outbound: StitchPreviewValue = {
			...localValue,
			tiles: tileRuntimes.map((runtime) => ({ ...runtime.tile })),
			selected: selectedIndex(),
			status: status ?? localValue.status ?? "",
		};
		gradio.props.value = outbound;
		lastSignature = JSON.stringify(outbound);
	}

	function syncValue(status: string, dispatchChange = true): void {
		localValue = { ...localValue, status };
		statusText = status;
		publishClientValue(status);
		if (dispatchChange) {
			clearQueuedSync();
			gradio.dispatch("change");
		}
		draw();
	}

	function queueSyncValue(status: string, delay = 140): void {
		syncValue(status, false);
		clearQueuedSync();
		changeSyncTimer = setTimeout(() => {
			changeSyncTimer = null;
			gradio.dispatch("change");
		}, delay);
	}

	function hitTile(worldX: number, worldY: number): number {
		for (let i = tileRuntimes.length - 1; i >= 0; i--) {
			const tile = tileRuntimes[i].tile;
			if (
				worldX >= tile.x &&
				worldX <= tile.x + tile.width &&
				worldY >= tile.y &&
				worldY <= tile.y + tile.height
			) {
				return tile.index;
			}
		}
		return -1;
	}

	function setSelected(index: number, status?: string): void {
		if (index < 0) return;
		localValue = { ...localValue, selected: index };
		statusText = status || "已选择 tile " + String(index);
		draw();
	}

	function nudgeSelected(dx: number, dy: number): void {
		const tile = selectedTile();
		if (!tile) return;
		beginMutation();
		tile.x += dx;
		tile.y += dy;
		tileRuntimes = [...tileRuntimes];
		endMutation();
		const offset = selectedOffset();
		queueSyncValue(
			"微调 tile " + String(tile.index) + " → dx=" + String(offset.dx) + " dy=" + String(offset.dy),
		);
	}

	function zoomAroundScreen(sx: number, sy: number, factor: number): void {
		const before = screenToWorld(sx, sy);
		viewZoom = clamp(viewZoom * factor, 0.05, 16);
		const after = worldToScreen(before.x, before.y);
		viewPanX += sx - after.x;
		viewPanY += sy - after.y;
		draw();
	}

	function releasePointerCapture(pointerId: number): void {
		if (!canvasEl || pointerId < 0) return;
		try {
			if (canvasEl.hasPointerCapture(pointerId)) canvasEl.releasePointerCapture(pointerId);
		} catch {
			// capture may already be released
		}
	}

	function cancelPointerInteraction(status = "已取消拖动"): void {
		const hadInteraction = panning || pointerIdActive >= 0 || dragTileIndex >= 0;
		const activePointerId = pointerIdActive;
		const runtime = tileRuntimes.find((entry) => entry.tile.index === dragTileIndex);
		if (runtime && (dragging || dragMoved)) {
			runtime.tile.x = dragOriginX;
			runtime.tile.y = dragOriginY;
			tileRuntimes = [...tileRuntimes];
		}
		if (hadInteraction) clearQueuedSync();
		localValue = { ...localValue, selected: pointerDownSelected };
		dragging = false;
		dragTileIndex = -1;
		dragMoved = false;
		panning = false;
		pointerIdActive = -1;
		dragShift = false;
		releasePointerCapture(activePointerId);
		if (hadInteraction) {
			statusText = status;
			publishClientValue(status);
			draw();
		}
	}

	function onWindowBlur(): void {
		cancelPointerInteraction("窗口失焦，已取消拖动");
	}

	function drawLoupe(ctx: CanvasRenderingContext2D): void {
		if (!showLoupe() || !showLoupePointer) return;
		const { width, height } = canvasSize();
		const cx = clamp(loupeX, LOUPE_RADIUS + 2, width - LOUPE_RADIUS - 2);
		const cy = clamp(loupeY, LOUPE_RADIUS + 2, height - LOUPE_RADIUS - 2);
		const centerWorld = screenToWorld(cx, cy);
		ctx.save();
		ctx.beginPath();
		ctx.arc(cx, cy, LOUPE_RADIUS, 0, Math.PI * 2);
		ctx.clip();
		ctx.fillStyle = "#0f172a";
		ctx.fillRect(cx - LOUPE_RADIUS, cy - LOUPE_RADIUS, LOUPE_RADIUS * 2, LOUPE_RADIUS * 2);
		ctx.translate(cx, cy);
		ctx.scale(LOUPE_SCALE * viewZoom, LOUPE_SCALE * viewZoom);
		// Match the main canvas transform around the world point under the cursor.
		// The previous screen-space translation applied pan twice after zooming.
		ctx.translate(-centerWorld.x, -centerWorld.y);
		for (const runtime of tileRuntimes) {
			if (!runtime.ready || !runtime.image) continue;
			const tile = runtime.tile;
			const activeDrag = dragging && dragTileIndex === tile.index;
			ctx.globalAlpha = tileAlpha(tile.index, activeDrag);
			if (diffMode() && tile.index !== 0) ctx.globalCompositeOperation = "difference";
			else ctx.globalCompositeOperation = "source-over";
			ctx.drawImage(runtime.image, tile.x, tile.y, tile.width, tile.height);
		}
		ctx.restore();
		ctx.save();
		ctx.beginPath();
		ctx.arc(cx, cy, LOUPE_RADIUS, 0, Math.PI * 2);
		ctx.strokeStyle = "rgba(255,255,255,0.9)";
		ctx.lineWidth = 2;
		ctx.stroke();
		ctx.strokeStyle = "rgba(15,23,42,0.85)";
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(cx - 8, cy);
		ctx.lineTo(cx + 8, cy);
		ctx.moveTo(cx, cy - 8);
		ctx.lineTo(cx, cy + 8);
		ctx.stroke();
		ctx.restore();
	}

	function drawHud(ctx: CanvasRenderingContext2D): void {
		const tile = selectedTile();
		const offset = selectedOffset();
		const lines = [
			"选中 #" + String(selectedIndex()),
			"dx " + String(offset.dx) + "  dy " + String(offset.dy),
			"zoom " + viewZoom.toFixed(2) + "  gain " + currentDragGain().toFixed(2),
		];
		ctx.save();
		ctx.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
		const pad = 8;
		const lineHeight = 16;
		const boxW = Math.max(...lines.map((line) => ctx.measureText(line).width)) + pad * 2;
		const boxH = lines.length * lineHeight + pad;
		ctx.fillStyle = "rgba(15,23,42,0.82)";
		ctx.fillRect(10, 10, boxW, boxH);
		ctx.fillStyle = "#e2e8f0";
		lines.forEach((line, i) => {
			ctx.fillText(line, 10 + pad, 10 + pad + (i + 1) * lineHeight - 4);
		});
		if (tile) {
			const tl = worldToScreen(tile.x, tile.y);
			const br = worldToScreen(tile.x + tile.width, tile.y + tile.height);
			ctx.strokeStyle = "#22d3ee";
			ctx.lineWidth = 2;
			ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
		}
		ctx.restore();
	}

	function draw(): void {
		if (!canvasEl) return;
		const dpr = window.devicePixelRatio || 1;
		const { width, height } = canvasSize();
		canvasEl.width = Math.max(1, Math.round(width * dpr));
		canvasEl.height = Math.max(1, Math.round(height * dpr));
		const ctx = canvasEl.getContext("2d");
		if (!ctx) return;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, width, height);
		ctx.fillStyle = "#0f172a";
		ctx.fillRect(0, 0, width, height);

		if (!tileRuntimes.length) {
			ctx.fillStyle = "#94a3b8";
			ctx.font = "16px sans-serif";
			ctx.fillText("等待 tile 数据", 24, 40);
			return;
		}

		ctx.save();
		ctx.translate(viewPanX, viewPanY);
		ctx.scale(viewZoom, viewZoom);
		for (const runtime of tileRuntimes) {
			if (!runtime.ready || !runtime.image) continue;
			const tile = runtime.tile;
			const activeDrag = dragging && dragTileIndex === tile.index;
			ctx.globalAlpha = tileAlpha(tile.index, activeDrag);
			if (diffMode() && tile.index !== 0) ctx.globalCompositeOperation = "difference";
			else ctx.globalCompositeOperation = "source-over";
			ctx.drawImage(runtime.image, tile.x, tile.y, tile.width, tile.height);
		}
		ctx.restore();

		if (dragging && dragTileIndex >= 0) {
			const runtime = tileRuntimes.find((entry) => entry.tile.index === dragTileIndex);
			if (runtime) {
				const tl = worldToScreen(dragGhostX, dragGhostY);
				const br = worldToScreen(
					dragGhostX + runtime.tile.width,
					dragGhostY + runtime.tile.height,
				);
				ctx.save();
				ctx.setLineDash([6, 4]);
				ctx.strokeStyle = "rgba(250,204,21,0.95)";
				ctx.lineWidth = 2;
				ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
				ctx.restore();
			}
		}

		drawHud(ctx);
		drawLoupe(ctx);
	}

	function onCanvasFocus(): void {
		focused = true;
		statusText = "键盘已接管";
	}

	function onCanvasBlur(): void {
		cancelPointerInteraction();
		focused = false;
		spaceDown = false;
		if (!dragging && !panning) cursorStyle = "crosshair";
		statusText = localValue.status || "点击画布以启用键盘";
	}

	function onCanvasClick(): void {
		canvasEl.focus();
	}

	function onKeyDown(evt: KeyboardEvent): void {
		if (!focused) return;
		const key = evt.key;
		const lower = key.toLowerCase();
		const controlKeys = new Set([
			"ArrowUp",
			"ArrowDown",
			"ArrowLeft",
			"ArrowRight",
			"w",
			"a",
			"s",
			"d",
			" ",
			"+",
			"=",
			"-",
			"_",
			"Escape",
			"z",
		]);
		if (controlKeys.has(key) || controlKeys.has(lower) || (evt.ctrlKey && lower === "z")) {
			evt.preventDefault();
		}

		if (key === " " || key === "Spacebar") {
			spaceDown = true;
			cursorStyle = "grab";
			return;
		}
		if (key === "Escape") {
			if (dragging || panning || pointerIdActive >= 0) {
				cancelPointerInteraction("已取消拖动");
			}
			return;
		}
		if (evt.ctrlKey && lower === "z") {
			if (undoIndex > 0) {
				undoIndex -= 1;
				applySnapshot(undoStack[undoIndex]);
				queueSyncValue("撤销到步骤 " + String(undoIndex + 1));
			}
			return;
		}
		if (key === "+" || key === "=") {
			const center = canvasSize();
			zoomAroundScreen(center.width / 2, center.height / 2, 1.15);
			return;
		}
		if (key === "-" || key === "_") {
			const center = canvasSize();
			zoomAroundScreen(center.width / 2, center.height / 2, 1 / 1.15);
			return;
		}

		let dx = 0;
		let dy = 0;
		if (key === "ArrowLeft" || lower === "a") dx = -1;
		if (key === "ArrowRight" || lower === "d") dx = 1;
		if (key === "ArrowUp" || lower === "w") dy = -1;
		if (key === "ArrowDown" || lower === "s") dy = 1;
		if (dx !== 0 || dy !== 0) {
			const step = nudgeStep() * (evt.shiftKey ? 10 : 1);
			nudgeSelected(dx * step, dy * step);
		}
	}

	function onKeyUp(evt: KeyboardEvent): void {
		if (!focused) return;
		if (evt.key === " " || evt.key === "Spacebar") {
			spaceDown = false;
			if (!panning) cursorStyle = dragging ? "grabbing" : "crosshair";
		}
	}

	function onPointerDown(evt: PointerEvent): void {
		if (!canvasEl) return;
		clearQueuedSync();
		const screen = eventToScreen(evt);
		pointerDownX = screen.x;
		pointerDownY = screen.y;
		pointerLastX = screen.x;
		pointerLastY = screen.y;
		loupeX = screen.x;
		loupeY = screen.y;
		showLoupePointer = true;
		dragShift = evt.shiftKey;
		pointerDownSelected = selectedIndex();

		if (evt.button === 1 || (evt.button === 0 && spaceDown)) {
			panning = true;
			pointerIdActive = evt.pointerId;
			cursorStyle = "grabbing";
			canvasEl.setPointerCapture(evt.pointerId);
			return;
		}
		if (evt.button !== 0) return;

		const world = screenToWorld(screen.x, screen.y);
		const hit = hitTile(world.x, world.y);
		if (hit >= 0) {
			setSelected(hit);
			const runtime = tileRuntimes.find((entry) => entry.tile.index === hit);
			if (runtime) {
				dragTileIndex = hit;
				dragOriginX = runtime.tile.x;
				dragOriginY = runtime.tile.y;
				dragGhostX = runtime.tile.x;
				dragGhostY = runtime.tile.y;
				dragging = false;
				dragMoved = false;
				pointerIdActive = evt.pointerId;
				canvasEl.setPointerCapture(evt.pointerId);
			}
		} else {
			dragTileIndex = -1;
		}
		draw();
	}

	function onPointerMove(evt: PointerEvent): void {
		const screen = eventToScreen(evt);
		loupeX = screen.x;
		loupeY = screen.y;
		showLoupePointer = true;

		if (panning) {
			viewPanX += screen.x - pointerLastX;
			viewPanY += screen.y - pointerLastY;
			pointerLastX = screen.x;
			pointerLastY = screen.y;
			cursorStyle = "grabbing";
			draw();
			return;
		}

		if (pointerIdActive === evt.pointerId && dragTileIndex >= 0) {
			const dist = Math.hypot(screen.x - pointerDownX, screen.y - pointerDownY);
			const runtime = tileRuntimes.find((entry) => entry.tile.index === dragTileIndex);
			if (!runtime) return;
			if (!dragging && dist >= DEADZONE_PX) {
				dragging = true;
				beginMutation();
				cursorStyle = "grabbing";
			}
			if (dragging) {
				dragShift = evt.shiftKey;
				const gain = evt.shiftKey ? preciseDragGain() : dragGain();
				const dx = ((screen.x - pointerLastX) / viewZoom) * gain;
				const dy = ((screen.y - pointerLastY) / viewZoom) * gain;
				runtime.tile.x += dx;
				runtime.tile.y += dy;
				tileRuntimes = [...tileRuntimes];
				dragMoved = true;
				const offset = selectedOffset();
				statusText =
					"拖动 tile " +
					String(dragTileIndex) +
					"  dx=" +
					String(offset.dx) +
					" dy=" +
					String(offset.dy);
				localValue = { ...localValue, status: statusText };
				publishClientValue(statusText);
			}
			pointerLastX = screen.x;
			pointerLastY = screen.y;
			draw();
			return;
		}

		if (spaceDown) cursorStyle = "grab";
		else cursorStyle = "crosshair";
		draw();
	}

	function onPointerUp(evt: PointerEvent): void {
		if (panning) {
			panning = false;
			releasePointerCapture(evt.pointerId);
			cursorStyle = spaceDown ? "grab" : "crosshair";
			pointerIdActive = -1;
			draw();
			return;
		}

		if (pointerIdActive === evt.pointerId && dragTileIndex >= 0) {
			const screen = eventToScreen(evt);
			const dist = Math.hypot(screen.x - pointerDownX, screen.y - pointerDownY);
			if (!dragging && dist < DEADZONE_PX) {
				setSelected(dragTileIndex, "已选择 tile " + String(dragTileIndex));
				syncValue("已选择 tile " + String(dragTileIndex));
			} else if (dragging && dragMoved) {
				endMutation();
				const offset = selectedOffset();
				syncValue(
					"tile " + String(dragTileIndex) + " 对齐 dx=" + String(offset.dx) + " dy=" + String(offset.dy),
				);
			}
			dragging = false;
			dragTileIndex = -1;
			dragMoved = false;
			pointerIdActive = -1;
			releasePointerCapture(evt.pointerId);
			cursorStyle = spaceDown ? "grab" : "crosshair";
			draw();
		}
	}

	function onPointerCancel(): void {
		cancelPointerInteraction();
	}

	function onPointerLeave(): void {
		showLoupePointer = false;
		if (!panning && !dragging) cursorStyle = spaceDown ? "grab" : "crosshair";
		draw();
	}

	function onWheel(evt: WheelEvent): void {
		evt.preventDefault();
		const screen = eventToScreen(evt);
		const factor = Math.exp(-evt.deltaY * (evt.ctrlKey ? WHEEL_ZOOM_SPEED * 4 : WHEEL_ZOOM_SPEED));
		zoomAroundScreen(screen.x, screen.y, factor);
	}
</script>

<Block
	visible={gradio.shared.visible}
	variant="solid"
	border_mode={focused ? "focus" : "base"}
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
	<div class="stitch-preview" style={"height:" + heightStyle(gradio.props.height)}>
		<div class="canvas-wrap" bind:this={wrapEl}>
			<canvas
				bind:this={canvasEl}
				tabindex="0"
				role="application"
				aria-label="tile stitch preview canvas"
				style={"cursor:" + cursorStyle}
				class:focused
				onfocus={onCanvasFocus}
				onblur={onCanvasBlur}
				onclick={onCanvasClick}
				onkeydown={onKeyDown}
				onkeyup={onKeyUp}
				onpointerdown={onPointerDown}
				onpointermove={onPointerMove}
				onpointerup={onPointerUp}
				onpointercancel={onPointerCancel}
				onpointerleave={onPointerLeave}
				onwheel={onWheel}
			></canvas>
		</div>
		<div class="shortcut-bar">
			↑↓←→ / WASD 步进 · Shift 10x · 普通拖动 1×原图倍率 · Shift+拖精准 0.25× · Space平移 · +/-缩放 · Esc取消 · Ctrl+Z撤销
		</div>
		<div class="status">{statusText}</div>
	</div>
</Block>

<style>
	.stitch-preview {
		display: flex;
		flex-direction: column;
		gap: 8px;
		background: #f8fafc;
		padding: 10px;
	}
	.canvas-wrap {
		flex: 1 1 auto;
		min-height: 240px;
		overflow: hidden;
		border: 1px solid #cbd5e1;
		background: #0f172a;
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
		min-height: 280px;
		outline: none;
		user-select: none;
		touch-action: none;
	}
	canvas.focused {
		box-shadow: inset 0 0 0 2px #38bdf8;
	}
	.shortcut-bar {
		font-size: 11px;
		line-height: 1.35;
		color: #334155;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 6px;
		padding: 6px 8px;
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
