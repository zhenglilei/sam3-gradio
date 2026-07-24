<svelte:options accessors={true} />

<script lang="ts">
	import type {
		LayoutTransform,
		LayoutTransformEditorEvents,
		LayoutTransformEditorProps,
		LayoutTransformGroupId,
		LayoutTransformGroupView,
		LayoutTransformValue,
	} from "./types";
	import { Gradio } from "@gradio/utils";
	import { Block } from "@gradio/atoms";
	import { StatusTracker } from "@gradio/statustracker";
	import { onDestroy } from "svelte";

	interface GroupRuntime {
		view: LayoutTransformGroupView;
		image: HTMLImageElement | null;
		maskCanvas: HTMLCanvasElement | null;
		tintCanvas: HTMLCanvasElement | null;
		ready: boolean;
		color: [number, number, number];
	}

	const props = $props();
	const MAX_CANVAS_SIDE = 2048;
	const WHEEL_ZOOM_SPEED = 0.00035;
	const GROUP_COLORS: [number, number, number][] = [
		[0, 255, 120],
		[0, 188, 255],
		[255, 179, 0],
		[213, 94, 255],
		[255, 82, 82],
		[75, 222, 196],
	];
	const gradio = new Gradio<LayoutTransformEditorEvents, LayoutTransformEditorProps>(props);

	let canvasEl: HTMLCanvasElement;
	let baseImg: HTMLImageElement | null = null;
	let maskImg: HTMLImageElement | null = null;
	let maskCanvas: HTMLCanvasElement | null = null;
	let tintCanvas: HTMLCanvasElement | null = null;
	let groupRuntimeByKey = new Map<string, GroupRuntime>();
	let groupTransformByKey = new Map<string, LayoutTransform>();
	let baseReady = $state(false);
	let maskReady = $state(false);
	let statusText = $state("等待版图 mask");
	let localValue = $state<LayoutTransformValue>({ enabled: false });
	let transform = $state<LayoutTransform>(defaultTransform());
	let activeGroupKey = $state("");
	let dragging = $state(false);
	let rotating = $state(false);
	let cursorStyle = $state("crosshair");
	let lastSignature = "";
	let imageLoadGeneration = 0;
	let dragStart = { x: 0, y: 0, center_x: 0, center_y: 0 };
	let rotateStart = { angle: 0, rotation: 0 };
	let changeSyncTimer: ReturnType<typeof setTimeout> | null = null;
	let dirtyGroupKeys = new Set<string>();

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

	function normalizeTransform(value: LayoutTransform | null | undefined): LayoutTransform {
		const merged = { ...defaultTransform(), ...(value || {}) };
		return {
			...merged,
			center_x: finiteNumber(merged.center_x, 0),
			center_y: finiteNumber(merged.center_y, 0),
			pivot_x: finiteNumber(merged.pivot_x, 0),
			pivot_y: finiteNumber(merged.pivot_y, 0),
			scale: clamp(finiteNumber(merged.scale, 1), 0.01, 20),
			rotation_deg: normalizeRotation(finiteNumber(merged.rotation_deg, 0)),
			preview_alpha: clamp(finiteNumber(merged.preview_alpha, 0.35), 0, 1),
			revision: Math.max(0, Math.trunc(finiteNumber(merged.revision, 0))),
		};
	}

	function finiteNumber(value: unknown, fallback: number): number {
		const numberValue = Number(value);
		return Number.isFinite(numberValue) ? numberValue : fallback;
	}

	function heightStyle(value: number | string | undefined): string {
		if (typeof value === "number") return String(value) + "px";
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

	function groupKey(groupId: LayoutTransformGroupId): string {
		return typeof groupId + ":" + String(groupId);
	}

	function groupViews(): LayoutTransformGroupView[] {
		if (!isGroupMode()) return [];
		return Array.isArray(localValue.group_view?.groups) ? localValue.group_view.groups : [];
	}

	function isGroupMode(): boolean {
		return localValue.transform_mode === "label_groups" && !!localValue.group_view;
	}

	function activeGroupView(): LayoutTransformGroupView | null {
		for (const group of groupViews()) {
			if (groupKey(group.group_id) === activeGroupKey) return group;
		}
		return null;
	}

	function activeGroupLabel(): string {
		const view = activeGroupView();
		return view ? String(view.label || "Label " + String(view.group_id)) : "";
	}

	function targetWidth(): number {
		return Math.max(1, Number(localValue.target_width || baseImg?.naturalWidth || 1));
	}

	function targetHeight(): number {
		return Math.max(1, Number(localValue.target_height || baseImg?.naturalHeight || 1));
	}

	function sourceWidth(): number {
		const groupImage = groupRuntimeByKey.get(activeGroupKey)?.image;
		return Math.max(1, Number(localValue.source_width || maskImg?.naturalWidth || groupImage?.naturalWidth || 1));
	}

	function sourceHeight(): number {
		const groupImage = groupRuntimeByKey.get(activeGroupKey)?.image;
		return Math.max(1, Number(localValue.source_height || maskImg?.naturalHeight || groupImage?.naturalHeight || 1));
	}

	function canvasRenderScale(): number {
		return Math.min(1, MAX_CANVAS_SIDE / Math.max(targetWidth(), targetHeight()));
	}

	function foregroundBbox(key = activeGroupKey): number[] {
		if (isGroupMode()) {
			const bbox = groupRuntimeByKey.get(key)?.view.foreground_bbox_xyxy;
			if (Array.isArray(bbox) && bbox.length >= 4) return bbox.slice(0, 4).map(Number);
		}
		const bbox = localValue.foreground_bbox_xyxy;
		if (Array.isArray(bbox) && bbox.length >= 4) return bbox.slice(0, 4).map(Number);
		return [0, 0, sourceWidth() - 1, sourceHeight() - 1];
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

	function buildMaskCanvases(
		image: HTMLImageElement,
		color: [number, number, number],
	): { mask: HTMLCanvasElement; tint: HTMLCanvasElement } | null {
		const sw = Math.max(1, Number(localValue.source_width || image.naturalWidth || 1));
		const sh = Math.max(1, Number(localValue.source_height || image.naturalHeight || 1));
		const rawCanvas = document.createElement("canvas");
		rawCanvas.width = sw;
		rawCanvas.height = sh;
		const rawCtx = rawCanvas.getContext("2d", { willReadFrequently: true });
		if (!rawCtx) return null;
		rawCtx.imageSmoothingEnabled = false;
		rawCtx.drawImage(image, 0, 0, sw, sh);
		const raw = rawCtx.getImageData(0, 0, sw, sh);
		const coloredCanvas = document.createElement("canvas");
		coloredCanvas.width = sw;
		coloredCanvas.height = sh;
		const tintCtx = coloredCanvas.getContext("2d");
		if (!tintCtx) return null;
		const tinted = tintCtx.createImageData(sw, sh);
		for (let i = 0; i < raw.data.length; i += 4) {
			const lum = Math.max(raw.data[i], raw.data[i + 1], raw.data[i + 2]);
			if (raw.data[i + 3] > 0 && lum >= 128) {
				tinted.data[i] = color[0];
				tinted.data[i + 1] = color[1];
				tinted.data[i + 2] = color[2];
				tinted.data[i + 3] = 255;
			}
		}
		tintCtx.putImageData(tinted, 0, 0);
		return { mask: rawCanvas, tint: coloredCanvas };
	}

	function rebuildLegacyMaskCanvases(): void {
		if (!maskImg) {
			maskCanvas = null;
			tintCanvas = null;
			return;
		}
		const built = buildMaskCanvases(maskImg, GROUP_COLORS[0]);
		maskCanvas = built?.mask || null;
		tintCanvas = built?.tint || null;
	}

	function rebuildGroupMaskCanvases(runtime: GroupRuntime): void {
		if (!runtime.image) {
			runtime.maskCanvas = null;
			runtime.tintCanvas = null;
			runtime.ready = false;
			return;
		}
		const built = buildMaskCanvases(runtime.image, runtime.color);
		runtime.maskCanvas = built?.mask || null;
		runtime.tintCanvas = built?.tint || null;
		runtime.ready = !!built;
	}

	function clearQueuedSync(): void {
		if (changeSyncTimer) {
			clearTimeout(changeSyncTimer);
			changeSyncTimer = null;
		}
	}

	function ingestGroupValue(generation: number): void {
		const view = localValue.group_view;
		const groups = Array.isArray(view?.groups) ? view.groups : [];
		const intent = localValue.group_intent;
		const intentMatches = !!view && !!intent && String(intent.selection_signature || "") === String(view.selection_signature || "");
		const incomingByKey = new Map<string, LayoutTransform>();
		if (intentMatches && Array.isArray(intent?.transforms)) {
			for (const item of intent.transforms) {
				if (!item || item.group_id === undefined || !item.transform) continue;
				incomingByKey.set(groupKey(item.group_id), normalizeTransform(item.transform));
			}
		}
		const fallback = normalizeTransform(localValue.transform);
		const nextRuntimes = new Map<string, GroupRuntime>();
		const nextTransforms = new Map<string, LayoutTransform>();
		const seen = new Set<string>();
		for (let index = 0; index < groups.length; index++) {
			const group = groups[index];
			if (!group || group.group_id === undefined || group.group_id === null) continue;
			const key = groupKey(group.group_id);
			if (seen.has(key)) continue;
			seen.add(key);
			const runtime: GroupRuntime = {
				view: group,
				image: null,
				maskCanvas: null,
				tintCanvas: null,
				ready: false,
				color: GROUP_COLORS[index % GROUP_COLORS.length],
			};
			nextRuntimes.set(key, runtime);
			nextTransforms.set(key, normalizeTransform(incomingByKey.get(key) || fallback));
		}
		groupRuntimeByKey = nextRuntimes;
		groupTransformByKey = nextTransforms;
		dirtyGroupKeys = new Set<string>();
		const requestedActiveKey = intentMatches && intent?.active_group_id !== undefined && intent.active_group_id !== null
			? groupKey(intent.active_group_id)
			: "";
		activeGroupKey = nextRuntimes.has(requestedActiveKey) ? requestedActiveKey : (nextRuntimes.keys().next().value || "");
		transform = normalizeTransform(groupTransformByKey.get(activeGroupKey) || fallback);
		if (activeGroupKey) groupTransformByKey.set(activeGroupKey, transform);
		maskImg = null;
		maskCanvas = null;
		tintCanvas = null;
		maskReady = false;
		for (const [key, runtime] of nextRuntimes) {
			loadImage(runtime.view.mask_image, generation, (img) => {
				const current = groupRuntimeByKey.get(key);
				if (!current || current !== runtime) return;
				current.image = img;
				rebuildGroupMaskCanvases(current);
				draw();
			});
		}
		statusText = localValue.status || (
			groups.length > 0
				? "已加载 " + String(groups.length) + " 个 Label；当前：" + activeGroupLabel()
				: "当前选择没有可编辑 Label"
		);
	}

	function ingestValue(value: LayoutTransformValue | null): void {
		clearQueuedSync();
		imageLoadGeneration += 1;
		const generation = imageLoadGeneration;
		localValue = cloneValue(value);
		baseReady = false;
		groupRuntimeByKey = new Map();
		groupTransformByKey = new Map();
		activeGroupKey = "";
		dragging = false;
		rotating = false;
		loadImage(localValue.base_image, generation, (img) => {
			baseImg = img;
			baseReady = !!img;
			draw();
		});
		if (isGroupMode()) {
			ingestGroupValue(generation);
		} else {
			transform = normalizeTransform(localValue.transform);
			statusText = localValue.status || "编辑器已加载";
			maskReady = false;
			loadImage(localValue.mask_image, generation, (img) => {
				maskImg = img;
				maskReady = !!img;
				rebuildLegacyMaskCanvases();
				draw();
			});
		}
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

	function maskForegroundAt(canvas: HTMLCanvasElement | null, sourceX: number, sourceY: number): boolean {
		if (!canvas) return false;
		const x = Math.round(sourceX);
		const y = Math.round(sourceY);
		if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return false;
		const ctx = canvas.getContext("2d", { willReadFrequently: true });
		if (!ctx) return false;
		const data = ctx.getImageData(x, y, 1, 1).data;
		return data[3] > 0 && Math.max(data[0], data[1], data[2]) >= 128;
	}

	function transformForKey(key: string): LayoutTransform {
		if (!isGroupMode() || key === activeGroupKey) return transform;
		return groupTransformByKey.get(key) || transform;
	}

	function setActiveTransform(next: LayoutTransform): void {
		transform = normalizeTransform(next);
		if (isGroupMode() && activeGroupKey) groupTransformByKey.set(activeGroupKey, transform);
	}

	function hitLegacyMask(targetX: number, targetY: number): boolean {
		const p = targetToSource(targetX, targetY, transform);
		return maskForegroundAt(maskCanvas, p.x, p.y);
	}

	function hitGroupMask(key: string, targetX: number, targetY: number): boolean {
		const runtime = groupRuntimeByKey.get(key);
		if (!runtime?.ready) return false;
		const p = targetToSource(targetX, targetY, transformForKey(key));
		return maskForegroundAt(runtime.maskCanvas, p.x, p.y);
	}

	function hitGroupAt(targetX: number, targetY: number): string {
		if (activeGroupKey && hitGroupMask(activeGroupKey, targetX, targetY)) return activeGroupKey;
		const keys = groupViews().map((group) => groupKey(group.group_id)).reverse();
		for (const key of keys) {
			if (key !== activeGroupKey && hitGroupMask(key, targetX, targetY)) return key;
		}
		return "";
	}

	function bboxCorners(key = activeGroupKey, t = transformForKey(key)): { x: number; y: number }[] {
		const [x1, y1, x2, y2] = foregroundBbox(key);
		return [
			sourceToTarget(x1, y1, t),
			sourceToTarget(x2, y1, t),
			sourceToTarget(x2, y2, t),
			sourceToTarget(x1, y2, t),
		];
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
		if (isGroupMode() && !activeGroupKey) return false;
		const h = rotateHandleCenter();
		const r = handleRadius();
		return Math.hypot(targetX - h.x, targetY - h.y) <= r;
	}

	function bumpActiveTransformRevision(origin: string): void {
		const changedGroupKey = isGroupMode() ? activeGroupKey : "";
		setActiveTransform({
			...transform,
			revision: Number(transform.revision || 0) + 1,
			origin,
			scale: clamp(Number(transform.scale || 1), 0.01, 20),
			rotation_deg: normalizeRotation(Number(transform.rotation_deg || 0)),
		});
		if (changedGroupKey) dirtyGroupKeys.add(changedGroupKey);
	}

	function serializeGroupTransforms(): { group_id: LayoutTransformGroupId; transform: LayoutTransform }[] {
		const entries: { group_id: LayoutTransformGroupId; transform: LayoutTransform }[] = [];
		for (const group of groupViews()) {
			const key = groupKey(group.group_id);
			entries.push({
				group_id: group.group_id,
				transform: { ...normalizeTransform(transformForKey(key)) },
			});
		}
		return entries;
	}

	function publishClientIntent(): void {
		const outbound: LayoutTransformValue = {
			enabled: localValue.enabled,
			transform: { ...normalizeTransform(transform) },
			target_width: localValue.target_width,
			target_height: localValue.target_height,
		};
		if (isGroupMode()) {
			outbound.transform_mode = "label_groups";
			outbound.group_intent = localValue.group_intent
				? JSON.parse(JSON.stringify(localValue.group_intent))
				: null;
		}
		gradio.props.value = outbound;
		lastSignature = JSON.stringify(outbound);
	}

	function publishGroupIntent(origin: string, status: string, bumpTransform: boolean, dispatchChange = true): void {
		if (!isGroupMode()) return;
		if (bumpTransform) bumpActiveTransformRevision(origin);
		const view = localValue.group_view;
		const activeView = activeGroupView();
		const changedGroupIds = groupViews()
			.filter((group) => dirtyGroupKeys.has(groupKey(group.group_id)))
			.map((group) => group.group_id);
		const priorRevision = Number(localValue.group_intent?.transform_set_revision || 0);
		localValue = {
			...localValue,
			group_intent: {
				selection_signature: String(view?.selection_signature || ""),
				transform_set_revision: priorRevision + 1,
				active_group_id: activeView?.group_id ?? null,
				changed_group_ids: changedGroupIds,
				transforms: serializeGroupTransforms(),
			},
		};
		statusText = status;
		publishClientIntent();
		if (dispatchChange) {
			clearQueuedSync();
			gradio.dispatch("change");
		}
		draw();
	}

	function syncValue(origin: string, status: string, dispatchChange = true): void {
		if (isGroupMode()) {
			publishGroupIntent(origin, status, true, dispatchChange);
			return;
		}
		bumpActiveTransformRevision(origin);
		localValue = {
			...localValue,
			enabled: true,
			transform: { ...transform },
			status,
		};
		statusText = status;
		publishClientIntent();
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

	function drawBase(ctx: CanvasRenderingContext2D, tw: number, th: number): void {
		if (baseImg && baseReady) {
			ctx.imageSmoothingEnabled = true;
			ctx.drawImage(baseImg, 0, 0, tw, th);
			return;
		}
		ctx.fillStyle = "#f8fafc";
		ctx.fillRect(0, 0, tw, th);
		ctx.fillStyle = "#64748b";
		ctx.font = "18px sans-serif";
		ctx.fillText("请先上传图像", 24, 42);
	}

	function drawTint(
		ctx: CanvasRenderingContext2D,
		rs: number,
		tinted: HTMLCanvasElement,
		t: LayoutTransform,
	): void {
		ctx.save();
		ctx.globalAlpha = clamp(Number(t.preview_alpha ?? 0.35), 0, 1);
		const [a, b, c, d, e, f] = matrix(t);
		ctx.setTransform(rs * a, rs * b, rs * c, rs * d, rs * e, rs * f);
		ctx.imageSmoothingEnabled = false;
		ctx.drawImage(tinted, 0, 0, tinted.width, tinted.height);
		ctx.restore();
	}

	function drawActiveControls(ctx: CanvasRenderingContext2D, tw: number): void {
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
		const top = corners.reduce(
			(best, p) => (p.y < best.y || (Math.abs(p.y - best.y) < 1e-6 && p.x > best.x) ? p : best),
			corners[0],
		);
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

	function drawGroupLabel(ctx: CanvasRenderingContext2D, key: string, active: boolean): void {
		const runtime = groupRuntimeByKey.get(key);
		if (!runtime) return;
		const corners = bboxCorners(key, transformForKey(key));
		const left = Math.min(...corners.map((point) => point.x));
		const top = Math.min(...corners.map((point) => point.y));
		const label = String(runtime.view.label || "Label " + String(runtime.view.group_id));
		ctx.save();
		ctx.font = "600 13px sans-serif";
		const width = ctx.measureText(label).width + 12;
		const x = clamp(left, 2, Math.max(2, targetWidth() - width - 2));
		const y = clamp(top - 23, 2, Math.max(2, targetHeight() - 22));
		ctx.fillStyle = active ? "rgba(15,23,42,0.92)" : "rgba(51,65,85,0.76)";
		ctx.fillRect(x, y, width, 20);
		ctx.fillStyle = "#ffffff";
		ctx.fillText(label, x + 6, y + 14);
		ctx.restore();
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
		drawBase(ctx, tw, th);
		if (localValue.enabled === false) return;
		if (isGroupMode()) {
			const keys = groupViews().map((group) => groupKey(group.group_id));
			const drawOrder = keys.filter((key) => key !== activeGroupKey);
			if (activeGroupKey) drawOrder.push(activeGroupKey);
			for (const key of drawOrder) {
				const runtime = groupRuntimeByKey.get(key);
				if (runtime?.ready && runtime.tintCanvas) {
					drawTint(ctx, rs, runtime.tintCanvas, transformForKey(key));
				}
			}
			ctx.setTransform(rs, 0, 0, rs, 0, 0);
			for (const key of drawOrder) {
				if (groupRuntimeByKey.get(key)?.ready) drawGroupLabel(ctx, key, key === activeGroupKey);
			}
			if (activeGroupKey && groupRuntimeByKey.get(activeGroupKey)?.ready) drawActiveControls(ctx, tw);
			return;
		}
		if (tintCanvas && maskReady) {
			drawTint(ctx, rs, tintCanvas, transform);
			ctx.setTransform(rs, 0, 0, rs, 0, 0);
			drawActiveControls(ctx, tw);
		}
	}

	function hasEditableMask(): boolean {
		if (!localValue.enabled) return false;
		if (isGroupMode()) return !!activeGroupKey && !!groupRuntimeByKey.get(activeGroupKey)?.ready;
		return maskReady && !!maskCanvas;
	}

	function setActiveGroupLocal(key: string): boolean {
		if (!isGroupMode() || !groupRuntimeByKey.has(key) || key === activeGroupKey) return false;
		if (activeGroupKey) groupTransformByKey.set(activeGroupKey, normalizeTransform(transform));
		activeGroupKey = key;
		transform = normalizeTransform(groupTransformByKey.get(key));
		groupTransformByKey.set(key, transform);
		statusText = "当前 Label：" + activeGroupLabel();
		draw();
		return true;
	}

	function onActiveSelect(evt: Event): void {
		const key = (evt.currentTarget as HTMLSelectElement).value;
		if (setActiveGroupLocal(key)) {
			publishGroupIntent("select_group", "已选择 Label：" + activeGroupLabel(), false, true);
		}
	}

	function onPointerDown(evt: PointerEvent): void {
		if (evt.button !== 0) return;
		if (!hasEditableMask()) {
			statusText = "请先启用并加载版图 mask";
			draw();
			return;
		}
		clearQueuedSync();
		const p = eventToTarget(evt);
		if (hitRotateHandle(p.x, p.y)) {
			cursorStyle = "grabbing";
			rotating = true;
			rotateStart = {
				angle: Math.atan2(p.y - transform.center_y, p.x - transform.center_x) * 180 / Math.PI,
				rotation: Number(transform.rotation_deg || 0),
			};
			canvasEl.setPointerCapture(evt.pointerId);
			return;
		}
		if (isGroupMode()) {
			const hitKey = hitGroupAt(p.x, p.y);
			if (!hitKey) {
				statusText = "请点中任一 Label mask 前景后拖动";
				draw();
				return;
			}
			setActiveGroupLocal(hitKey);
		} else if (!hitLegacyMask(p.x, p.y)) {
			statusText = "请点中版图 mask 前景后拖动";
			draw();
			return;
		}
		cursorStyle = "grabbing";
		dragging = true;
		dragStart = {
			x: p.x,
			y: p.y,
			center_x: Number(transform.center_x || 0),
			center_y: Number(transform.center_y || 0),
		};
		canvasEl.setPointerCapture(evt.pointerId);
	}

	function updateCursor(p: { x: number; y: number }): void {
		if (!hasEditableMask()) {
			cursorStyle = "not-allowed";
			return;
		}
		const hit = isGroupMode() ? !!hitGroupAt(p.x, p.y) : hitLegacyMask(p.x, p.y);
		if (hitRotateHandle(p.x, p.y) || hit) {
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
			setActiveTransform({
				...transform,
				center_x: dragStart.center_x + p.x - dragStart.x,
				center_y: dragStart.center_y + p.y - dragStart.y,
			});
			statusText = "正在拖动 " + (isGroupMode() ? activeGroupLabel() : "版图") + "；松开后同步变换";
		} else if (rotating) {
			const angle = Math.atan2(p.y - transform.center_y, p.x - transform.center_x) * 180 / Math.PI;
			setActiveTransform({
				...transform,
				rotation_deg: normalizeRotation(rotateStart.rotation + angle - rotateStart.angle),
			});
			statusText = "正在旋转 " + (isGroupMode() ? activeGroupLabel() : "版图") + "；松开后同步变换";
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
			try {
				canvasEl.releasePointerCapture(evt.pointerId);
			} catch (_) {
				// Pointer capture may already have been released by the browser.
			}
			updateCursor(eventToTarget(evt));
			const subject = isGroupMode() ? activeGroupLabel() : "Canvas";
			syncValue("canvas", subject + " 变换已同步；点击更新预览或创建实例后使用后端权威 mask");
		}
	}

	function onWheel(evt: WheelEvent): void {
		if (!hasEditableMask()) return;
		evt.preventDefault();
		const p = eventToTarget(evt);
		if (isGroupMode()) {
			const hitKey = hitGroupAt(p.x, p.y);
			if (hitKey) setActiveGroupLocal(hitKey);
		}
		const before = targetToSource(p.x, p.y, transform);
		const factor = Math.exp(-evt.deltaY * WHEEL_ZOOM_SPEED);
		const nextScale = clamp(Number(transform.scale || 1) * factor, 0.01, 20);
		let next = { ...transform, scale: nextScale };
		const afterTarget = sourceToTarget(before.x, before.y, next);
		next = {
			...next,
			center_x: Number(next.center_x || 0) + p.x - afterTarget.x,
			center_y: Number(next.center_y || 0) + p.y - afterTarget.y,
		};
		setActiveTransform(next);
		queueSyncValue("canvas", "滚轮缩放已同步: scale=" + nextScale.toFixed(3));
	}

	function resetTransform(): void {
		if (!hasEditableMask()) return;
		setActiveTransform({
			...transform,
			center_x: targetWidth() / 2,
			center_y: targetHeight() / 2,
			scale: 1,
			rotation_deg: 0,
		});
		syncValue("reset", "居中归一 " + (isGroupMode() ? activeGroupLabel() : "版图") + "：scale=1，rotation=0");
	}

	function centerTransform(): void {
		if (!hasEditableMask()) return;
		setActiveTransform({ ...transform, center_x: targetWidth() / 2, center_y: targetHeight() / 2 });
		syncValue("center", "居中 " + (isGroupMode() ? activeGroupLabel() : "版图") + "：保留缩放和旋转");
	}

	function fitTransform(): void {
		if (!hasEditableMask()) return;
		const [x1, y1, x2, y2] = foregroundBbox();
		const bw = Math.max(1, x2 - x1 + 1);
		const bh = Math.max(1, y2 - y1 + 1);
		const scale = clamp(Math.min(targetWidth() / bw, targetHeight() / bh) * 0.9, 0.01, 20);
		setActiveTransform({ ...transform, center_x: targetWidth() / 2, center_y: targetHeight() / 2, scale });
		syncValue("fit", "适配 " + (isGroupMode() ? activeGroupLabel() : "版图") + "：scale=" + scale.toFixed(3));
	}

	function bringIntoView(): void {
		if (!hasEditableMask()) return;
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
			statusText = (isGroupMode() ? activeGroupLabel() : "版图") + " 已经在视野内";
			return;
		}
		setActiveTransform({
			...transform,
			center_x: Number(transform.center_x || 0) + dx,
			center_y: Number(transform.center_y || 0) + dy,
		});
		syncValue("bring_into_view", "已找回 " + (isGroupMode() ? activeGroupLabel() : "版图") + " 到视野内");
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
	<div class="layout-editor" style={"min-height:" + heightStyle(gradio.props.height)}>
		{#if isGroupMode()}
			<div class="group-picker">
				<label for="layout-active-label">当前 Label</label>
				<select id="layout-active-label" value={activeGroupKey} on:change={onActiveSelect}>
					{#each groupViews() as group}
						<option value={groupKey(group.group_id)}>
							{group.label || "Label " + String(group.group_id)}
						</option>
					{/each}
				</select>
			</div>
		{/if}
		<div class="toolbar">
			<button type="button" on:click={resetTransform} disabled={!hasEditableMask()}>居中归一</button>
			<button type="button" on:click={centerTransform} disabled={!hasEditableMask()}>居中</button>
			<button type="button" on:click={fitTransform} disabled={!hasEditableMask()}>适配</button>
			<button type="button" on:click={bringIntoView} disabled={!hasEditableMask()}>找回视野</button>
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
				style={"cursor:" + cursorStyle}
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
	.group-picker {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		align-items: center;
		gap: 8px;
	}
	.group-picker label {
		font-size: 12px;
		font-weight: 700;
		color: #334155;
	}
	.group-picker select {
		min-width: 0;
		border: 1px solid #94a3b8;
		border-radius: 6px;
		background: #ffffff;
		color: #0f172a;
		padding: 7px 9px;
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
	.toolbar button:hover:not(:disabled) {
		border-color: #2563eb;
		color: #1d4ed8;
	}
	.toolbar button:disabled {
		cursor: not-allowed;
		opacity: 0.48;
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
