<svelte:options accessors={true} />

<script lang="ts">
	import type {
		LayoutTransform,
		LayoutTransformAlignmentCandidate,
		LayoutTransformEditorEvents,
		LayoutTransformEditorProps,
		LayoutTransformGroupId,
		LayoutTransformGroupIntent,
		LayoutTransformGroupView,
		LayoutTransformValue,
	} from "./types";
	import { Gradio } from "@gradio/utils";
	import { Block } from "@gradio/atoms";
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
	const WHEEL_ZOOM_LOG_PER_PIXEL = Math.log(1.01) / 100;
	const EDIT_SYNC_DELAY_MS = 250;
	const EDIT_SYNC_TIMEOUT_MS = 10000;
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
	let overlapCanvas: HTMLCanvasElement | null = null;
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
	let syncAckTimer: ReturnType<typeof setTimeout> | null = null;
	let syncInFlightRevision: number | null = null;
	let syncRetryRevision: number | null = null;
	let clientRevision = 0;
	let serverAckRevision = 0;
	let editorSessionId = "";
	let documentEpoch = "";
	let lastAckSignature = "";
	let dirtyGroupKeys = new Set<string>();
	let draggedGroupKey = $state("");
	let dragOverGroupKey = $state("");
	let dragOverPosition = $state<"before" | "after">("before");
	let suppressGroupClick = $state(false);
	let displayMode = $state<"align" | "overlap">("align");
	let hiddenGroupKeys = $state(new Set<string>());
	let explicitVisibleGroupKeys = $state(new Set<string>());
	let soloGroupKey = $state("");
	let alignmentProposal = $state<LayoutTransformValue | null>(null);
	let manuallyTouchedGroupKeys = new Set<string>();
	let viewScale = $state(1);
	let viewPanX = $state(0);
	let viewPanY = $state(0);
	let panning = $state(false);
	let spacePressed = $state(false);
	let panStart = { client_x: 0, client_y: 0, pan_x: 0, pan_y: 0 };
	let drawFrame: number | null = null;
	type NumericField = "x" | "y" | "scale" | "rotation" | "alpha";
	let editingNumericField = $state<NumericField | "">("");
	let numericDraft = $state<Record<NumericField, string>>({
		x: "0",
		y: "0",
		scale: "100",
		rotation: "0",
		alpha: "22",
	});

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

	function canReorderGroups(): boolean {
		return isGroupMode() && localValue.group_view?.allow_reorder === true;
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

	function nextBusinessGroupKey(): string {
		const groups = groupViews();
		const index = groups.findIndex((group) => groupKey(group.group_id) === activeGroupKey);
		if (index < 0 || index + 1 >= groups.length) return "";
		return groupKey(groups[index + 1].group_id);
	}

	function defaultGroupVisible(key: string): boolean {
		if (soloGroupKey) return key === soloGroupKey;
		if (key === activeGroupKey) return true;
		return displayMode === "overlap" && key === nextBusinessGroupKey();
	}

	function isGroupVisible(key: string): boolean {
		if (hiddenGroupKeys.has(key)) return false;
		if (explicitVisibleGroupKeys.has(key)) return true;
		return defaultGroupVisible(key);
	}

	function visibleGroupKeys(): string[] {
		return groupViews()
			.map((group) => groupKey(group.group_id))
			.filter((key) => isGroupVisible(key));
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

	function clearSyncAckTimer(): void {
		if (syncAckTimer) {
			clearTimeout(syncAckTimer);
			syncAckTimer = null;
		}
	}

	function hashIntent(value: unknown): string {
		const text = JSON.stringify(value);
		let hash = 0x811c9dc5;
		for (let index = 0; index < text.length; index++) {
			hash ^= text.charCodeAt(index);
			hash = Math.imul(hash, 0x01000193);
		}
		return "fnv1a32:" + (hash >>> 0).toString(16).padStart(8, "0");
	}

	function requestDraw(): void {
		if (drawFrame !== null) return;
		drawFrame = requestAnimationFrame(() => {
			drawFrame = null;
			draw();
		});
	}

	function refreshNumericDraft(): void {
		if (editingNumericField) return;
		numericDraft = {
			x: finiteNumber(transform.center_x, 0).toFixed(2).replace(/\.00$/, ""),
			y: finiteNumber(transform.center_y, 0).toFixed(2).replace(/\.00$/, ""),
			scale: (finiteNumber(transform.scale, 1) * 100).toFixed(3).replace(/0+$/, "").replace(/\.$/, ""),
			rotation: finiteNumber(transform.rotation_deg, 0).toFixed(3).replace(/0+$/, "").replace(/\.$/, ""),
			alpha: (finiteNumber(transform.preview_alpha, 0.22) * 100).toFixed(1).replace(/\.0$/, ""),
		};
	}

	function restoreGroupOrder(
		groups: LayoutTransformGroupView[],
		orderedGroupIds: LayoutTransformGroupId[] | undefined,
	): LayoutTransformGroupView[] {
		if (!Array.isArray(orderedGroupIds) || orderedGroupIds.length !== groups.length) return groups;
		const groupsByKey = new Map<string, LayoutTransformGroupView>();
		for (const group of groups) {
			if (!group || group.group_id === undefined || group.group_id === null) return groups;
			const key = groupKey(group.group_id);
			if (groupsByKey.has(key)) return groups;
			groupsByKey.set(key, group);
		}
		const orderedGroups: LayoutTransformGroupView[] = [];
		const seen = new Set<string>();
		for (const groupId of orderedGroupIds) {
			if (typeof groupId !== "string" && typeof groupId !== "number") return groups;
			const key = groupKey(groupId);
			const group = groupsByKey.get(key);
			if (!group || seen.has(key)) return groups;
			seen.add(key);
			orderedGroups.push(group);
		}
		return orderedGroups.length === groups.length ? orderedGroups : groups;
	}


	function ingestGroupValue(generation: number): void {
		const view = localValue.group_view;
		const serverGroups = Array.isArray(view?.groups) ? view.groups : [];
		const intent = localValue.group_intent;
		const intentMatches = !!view && !!intent && String(intent.selection_signature || "") === String(view.selection_signature || "");
		editorSessionId = String(localValue.editor_session_id || intent?.editor_session_id || "");
		documentEpoch = String(localValue.document_epoch || intent?.document_epoch || view?.selection_signature || "");
		clientRevision = Math.max(0, Math.trunc(finiteNumber(intent?.client_revision ?? intent?.transform_set_revision, 0)));
		serverAckRevision = clientRevision;
		syncInFlightRevision = null;
		clearSyncAckTimer();
		const groups = view?.allow_reorder === true && intentMatches
			? restoreGroupOrder(serverGroups, intent?.ordered_group_ids)
			: serverGroups;
		if (groups !== serverGroups && view) {
			localValue = { ...localValue, group_view: { ...view, groups } };
		}
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
		manuallyTouchedGroupKeys = new Set<string>();
		alignmentProposal = null;
		const requestedActiveKey = intentMatches && intent?.active_group_id !== undefined && intent.active_group_id !== null
			? groupKey(intent.active_group_id)
			: "";
		activeGroupKey = nextRuntimes.has(requestedActiveKey) ? requestedActiveKey : (nextRuntimes.keys().next().value || "");
		transform = normalizeTransform(groupTransformByKey.get(activeGroupKey) || fallback);
		if (activeGroupKey) groupTransformByKey.set(activeGroupKey, transform);
		refreshNumericDraft();
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
		clearSyncAckTimer();
		syncInFlightRevision = null;
		syncRetryRevision = null;
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
			refreshNumericDraft();
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

	function proposalCandidates(): LayoutTransformAlignmentCandidate[] {
		return Array.isArray(alignmentProposal?.candidates)
			? alignmentProposal.candidates
			: [];
	}

	function applyAlignmentCandidate(
		candidate: LayoutTransformAlignmentCandidate,
		automatic = false,
	): void {
		if (!alignmentProposal || alignmentProposal.target_layer_id === undefined) return;
		const key = groupKey(alignmentProposal.target_layer_id);
		if (!groupTransformByKey.has(key)) return;
		if (
			automatic
			&& (
				manuallyTouchedGroupKeys.has(key)
				|| finiteNumber(
					transformForKey(key).revision,
					0,
				) !== finiteNumber(alignmentProposal.start_transform_revision, 0)
			)
		) {
			statusText = "粗对齐结果已到达，但检测到手动修改，未覆盖当前变换";
			publishClientIntent();
			return;
		}
		setActiveGroupLocal(key);
		const current = normalizeTransform(transformForKey(key));
		setActiveTransform({
			...current,
			...candidate.transform,
			preview_alpha: current.preview_alpha,
		}, false);
		alignmentProposal = null;
		queueSyncValue(
			"boundary_alignment",
			(automatic ? "已自动应用 " : "已应用 ")
				+ String(candidate.candidate_id || "候选")
				+ "；停止 250ms 后同步",
		);
	}

	function handleAlignmentProposal(value: LayoutTransformValue): void {
		const proposalSession = String(value.editor_session_id || "");
		const proposalEpoch = String(value.document_epoch || "");
		if (
			(proposalSession && proposalSession !== editorSessionId)
			|| (proposalEpoch && proposalEpoch !== documentEpoch)
		) return;
		alignmentProposal = cloneValue(value);
		statusText = String(value.status || "粗对齐候选已生成");
		publishClientIntent();
		const candidates = Array.isArray(value.candidates) ? value.candidates : [];
		if (value.auto_apply && candidates.length) {
			applyAlignmentCandidate(candidates[0], true);
		} else {
			requestDraw();
		}
	}

	function handleServerAck(value: LayoutTransformValue): void {
		const ackSession = String(value.editor_session_id || "");
		const ackEpoch = String(value.document_epoch || "");
		if (
			(ackSession && ackSession !== editorSessionId)
			|| (ackEpoch && ackEpoch !== documentEpoch)
		) {
			return;
		}
		clearSyncAckTimer();
		if (value.payload_kind === "ack_error") {
			syncInFlightRevision = null;
			statusText = String(value.status || "同步失败；本地修改仍保留");
			publishClientIntent();
			draw();
			return;
		}
		const ackRevision = Math.max(0, Math.trunc(finiteNumber(value.server_ack_revision, 0)));
		serverAckRevision = Math.max(serverAckRevision, ackRevision);
		syncRetryRevision = null;
		if (syncInFlightRevision !== null && ackRevision >= syncInFlightRevision) {
			syncInFlightRevision = null;
		}
		statusText = clientRevision > serverAckRevision ? "本地修改待同步" : "已同步";
		publishClientIntent();
		if (clientRevision > serverAckRevision) scheduleIntentDispatch(0);
		draw();
	}

	$effect(() => {
		const incoming = gradio.props.value || null;
		const signature = JSON.stringify(incoming);
		if (incoming?.payload_kind === "ack" || incoming?.payload_kind === "ack_error") {
			if (signature !== lastAckSignature) {
				lastAckSignature = signature;
				handleServerAck(incoming);
			}
			return;
		}
		if (incoming?.payload_kind === "alignment_proposal") {
			if (signature !== lastAckSignature) {
				lastAckSignature = signature;
				handleAlignmentProposal(incoming);
			}
			return;
		}
		if (signature !== lastSignature) {
			lastSignature = signature;
			ingestValue(incoming);
		}
	});

	onDestroy(() => {
		imageLoadGeneration += 1;
		clearQueuedSync();
		clearSyncAckTimer();
		if (drawFrame !== null) cancelAnimationFrame(drawFrame);
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
		const displayX = ((evt.clientX - rect.left) / Math.max(1, rect.width)) * targetWidth();
		const displayY = ((evt.clientY - rect.top) / Math.max(1, rect.height)) * targetHeight();
		return {
			x: (displayX - viewPanX) / Math.max(0.05, viewScale),
			y: (displayY - viewPanY) / Math.max(0.05, viewScale),
		};
	}

	function viewTransform(ctx: CanvasRenderingContext2D, rs: number): void {
		ctx.setTransform(
			rs * viewScale,
			0,
			0,
			rs * viewScale,
			rs * viewPanX,
			rs * viewPanY,
		);
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

	function setActiveTransform(next: LayoutTransform, manual = true): void {
		transform = normalizeTransform(next);
		if (isGroupMode() && activeGroupKey) {
			groupTransformByKey.set(activeGroupKey, transform);
			if (manual) manuallyTouchedGroupKeys.add(activeGroupKey);
		}
		refreshNumericDraft();
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
		const visible = new Set(visibleGroupKeys());
		if (
			activeGroupKey
			&& visible.has(activeGroupKey)
			&& hitGroupMask(activeGroupKey, targetX, targetY)
		) return activeGroupKey;
		const keys = groupViews().map((group) => groupKey(group.group_id)).reverse();
		for (const key of keys) {
			if (
				key !== activeGroupKey
				&& visible.has(key)
				&& hitGroupMask(key, targetX, targetY)
			) return key;
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
		const offset = 34 / Math.max(0.05, viewScale);
		return { x: corner.x + ux * offset, y: corner.y + uy * offset };
	}

	function handleRadius(): number {
		const rect = canvasEl?.getBoundingClientRect();
		if (!rect) return 14 / viewScale;
		return Math.max(
			8 / viewScale,
			(14 * targetWidth()) / Math.max(1, rect.width) / viewScale,
		);
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
			payload_kind: "intent",
			editor_session_id: editorSessionId,
			document_epoch: documentEpoch,
			payload_hash: String(localValue.group_intent?.payload_hash || ""),
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

	function dispatchLatestIntent(): void {
		if (!isGroupMode()) {
			gradio.dispatch("change");
			return;
		}
		const intent = localValue.group_intent;
		const revision = Math.max(0, Math.trunc(finiteNumber(intent?.client_revision, 0)));
		if (!intent || revision <= serverAckRevision || syncInFlightRevision !== null) return;
		clearSyncAckTimer();
		syncInFlightRevision = revision;
		statusText = "同步中…";
		publishClientIntent();
		gradio.dispatch("change");
		syncAckTimer = setTimeout(() => {
			if (syncInFlightRevision !== revision) return;
			syncInFlightRevision = null;
			const retry = syncRetryRevision !== revision;
			syncRetryRevision = revision;
			statusText = retry ? "同步超时，正在重试；本地修改仍保留" : "同步失败；可继续编辑或直接保存重试";
			publishClientIntent();
			draw();
			if (retry) scheduleIntentDispatch(1000);
		}, EDIT_SYNC_TIMEOUT_MS);
	}

	function scheduleIntentDispatch(delay = EDIT_SYNC_DELAY_MS): void {
		clearQueuedSync();
		changeSyncTimer = setTimeout(() => {
			changeSyncTimer = null;
			dispatchLatestIntent();
		}, Math.max(0, delay));
	}

	function publishGroupIntent(origin: string, status: string, bumpTransform: boolean, dispatchChange = true): void {
		if (!isGroupMode()) return;
		if (bumpTransform) bumpActiveTransformRevision(origin);
		const view = localValue.group_view;
		const activeView = activeGroupView();
		const changedGroupIds = groupViews()
			.filter((group) => dirtyGroupKeys.has(groupKey(group.group_id)))
			.map((group) => group.group_id);
		const nextRevision = clientRevision + 1;
		const nextIntent: LayoutTransformGroupIntent = {
			selection_signature: String(view?.selection_signature || documentEpoch),
			transform_set_revision: nextRevision,
			editor_session_id: editorSessionId,
			document_epoch: documentEpoch,
			client_revision: nextRevision,
			payload_hash: "",
			active_group_id: activeView?.group_id ?? null,
			changed_group_ids: changedGroupIds,
			...(canReorderGroups() ? { ordered_group_ids: groupViews().map((group) => group.group_id) } : {}),
			transforms: serializeGroupTransforms(),
		};
		nextIntent.payload_hash = hashIntent({ ...nextIntent, payload_hash: "" });
		clientRevision = nextRevision;
		syncRetryRevision = null;
		localValue = {
			...localValue,
			payload_kind: "intent",
			editor_session_id: editorSessionId,
			document_epoch: documentEpoch,
			payload_hash: nextIntent.payload_hash,
			group_intent: nextIntent,
		};
		statusText = status + "（本地修改）";
		publishClientIntent();
		if (dispatchChange) scheduleIntentDispatch(0);
		requestDraw();
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
		requestDraw();
	}

	function queueSyncValue(origin: string, status: string, delay = EDIT_SYNC_DELAY_MS): void {
		if (isGroupMode()) {
			syncValue(origin, status, false);
			scheduleIntentDispatch(delay);
			return;
		}
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
		alpha = clamp(Number(t.preview_alpha ?? 0.35), 0, 1),
	): void {
		ctx.save();
		ctx.globalAlpha = alpha;
		const [a, b, c, d, e, f] = matrix(t);
		ctx.setTransform(
			rs * viewScale * a,
			rs * viewScale * b,
			rs * viewScale * c,
			rs * viewScale * d,
			rs * (viewScale * e + viewPanX),
			rs * (viewScale * f + viewPanY),
		);
		ctx.imageSmoothingEnabled = false;
		ctx.drawImage(tinted, 0, 0, tinted.width, tinted.height);
		ctx.restore();
	}

	function drawOverlapPreview(
		ctx: CanvasRenderingContext2D,
		rs: number,
		firstKey: string,
		secondKey: string,
	): void {
		const first = groupRuntimeByKey.get(firstKey);
		const second = groupRuntimeByKey.get(secondKey);
		if (!first?.tintCanvas || !second?.tintCanvas || !first.ready || !second.ready) return;
		const overlay = overlapCanvas || document.createElement("canvas");
		overlapCanvas = overlay;
		if (overlay.width !== canvasEl.width) overlay.width = canvasEl.width;
		if (overlay.height !== canvasEl.height) overlay.height = canvasEl.height;
		const overlapCtx = overlay.getContext("2d");
		if (!overlapCtx) return;
		overlapCtx.setTransform(1, 0, 0, 1, 0, 0);
		overlapCtx.globalCompositeOperation = "source-over";
		overlapCtx.clearRect(0, 0, overlay.width, overlay.height);
		drawTint(overlapCtx, rs, first.tintCanvas, transformForKey(firstKey), 1);
		overlapCtx.globalCompositeOperation = "destination-in";
		drawTint(overlapCtx, rs, second.tintCanvas, transformForKey(secondKey), 1);
		overlapCtx.globalCompositeOperation = "source-in";
		overlapCtx.setTransform(1, 0, 0, 1, 0, 0);
		overlapCtx.fillStyle = "#ff3fa4";
		overlapCtx.fillRect(0, 0, overlay.width, overlay.height);
		overlapCtx.globalCompositeOperation = "source-atop";
		overlapCtx.strokeStyle = "rgba(255,255,255,0.78)";
		overlapCtx.lineWidth = 1.5;
		for (let offset = -overlay.height; offset < overlay.width; offset += 14) {
			overlapCtx.beginPath();
			overlapCtx.moveTo(offset, 0);
			overlapCtx.lineTo(offset + overlay.height, overlay.height);
			overlapCtx.stroke();
		}
		ctx.save();
		ctx.setTransform(1, 0, 0, 1, 0, 0);
		ctx.globalAlpha = 0.62;
		ctx.drawImage(overlay, 0, 0);
		ctx.restore();
	}

	function drawActiveControls(ctx: CanvasRenderingContext2D, tw: number): void {
		const corners = bboxCorners();
		ctx.save();
		ctx.lineJoin = "round";
		ctx.lineWidth = Math.max(2.5, tw / 700) / viewScale;
		ctx.strokeStyle = "rgba(0,0,0,0.82)";
		ctx.beginPath();
		ctx.moveTo(corners[0].x, corners[0].y);
		for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i].x, corners[i].y);
		ctx.closePath();
		ctx.stroke();
		ctx.lineWidth = Math.max(1.8, tw / 1000) / viewScale;
		ctx.strokeStyle = "#00ff66";
		ctx.stroke();
		const h = rotateHandleCenter();
		const top = corners.reduce(
			(best, point) => (
				point.y < best.y || (Math.abs(point.y - best.y) < 1e-6 && point.x > best.x)
					? point
					: best
			),
			corners[0],
		);
		ctx.strokeStyle = "#0f172a";
		ctx.lineWidth = Math.max(2, tw / 900) / viewScale;
		ctx.beginPath();
		ctx.moveTo(top.x, top.y);
		ctx.lineTo(h.x, h.y);
		ctx.stroke();
		ctx.fillStyle = rotating ? "#ffb000" : "#ffffff";
		ctx.strokeStyle = "#00ff66";
		ctx.lineWidth = Math.max(2, tw / 900) / viewScale;
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
		const inverseView = 1 / Math.max(0.05, viewScale);
		ctx.save();
		ctx.font = "600 " + String(13 * inverseView) + "px sans-serif";
		const width = ctx.measureText(label).width + 12 * inverseView;
		const x = clamp(left, 2, Math.max(2, targetWidth() - width - 2));
		const y = clamp(top - 23 * inverseView, 2, Math.max(2, targetHeight() - 22 * inverseView));
		ctx.fillStyle = active ? "rgba(15,23,42,0.92)" : "rgba(51,65,85,0.62)";
		ctx.fillRect(x, y, width, 20 * inverseView);
		ctx.fillStyle = "#ffffff";
		ctx.fillText(label, x + 6 * inverseView, y + 14 * inverseView);
		ctx.restore();
	}

	function draw(): void {
		if (!canvasEl) return;
		const tw = targetWidth();
		const th = targetHeight();
		const rs = canvasRenderScale();
		const renderWidth = Math.max(1, Math.round(tw * rs));
		const renderHeight = Math.max(1, Math.round(th * rs));
		if (canvasEl.width !== renderWidth) canvasEl.width = renderWidth;
		if (canvasEl.height !== renderHeight) canvasEl.height = renderHeight;
		const ctx = canvasEl.getContext("2d");
		if (!ctx) return;
		ctx.setTransform(1, 0, 0, 1, 0, 0);
		ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
		viewTransform(ctx, rs);
		drawBase(ctx, tw, th);
		if (localValue.enabled === false) return;
		if (isGroupMode()) {
			const keys = visibleGroupKeys();
			const drawOrder = keys.filter((key) => key !== activeGroupKey);
			if (keys.includes(activeGroupKey)) drawOrder.push(activeGroupKey);
			const nextKey = nextBusinessGroupKey();
			for (const key of drawOrder) {
				const runtime = groupRuntimeByKey.get(key);
				if (runtime?.ready && runtime.tintCanvas) {
					const alpha = key === activeGroupKey
						? clamp(Number(transformForKey(key).preview_alpha ?? 0.22), 0, 1)
						: (key === nextKey ? 0.10 : 0.08);
					drawTint(ctx, rs, runtime.tintCanvas, transformForKey(key), alpha);
				}
			}
			if (
				displayMode === "overlap"
				&& activeGroupKey
				&& nextKey
				&& keys.includes(activeGroupKey)
				&& keys.includes(nextKey)
			) {
				drawOverlapPreview(ctx, rs, activeGroupKey, nextKey);
			}
			viewTransform(ctx, rs);
			for (const key of drawOrder) {
				if (groupRuntimeByKey.get(key)?.ready) drawGroupLabel(ctx, key, key === activeGroupKey);
			}
			if (
				activeGroupKey
				&& keys.includes(activeGroupKey)
				&& groupRuntimeByKey.get(activeGroupKey)?.ready
			) drawActiveControls(ctx, tw);
			return;
		}
		if (tintCanvas && maskReady) {
			drawTint(ctx, rs, tintCanvas, transform);
			viewTransform(ctx, rs);
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
		refreshNumericDraft();
		statusText = "当前 Label：" + activeGroupLabel();
		requestDraw();
		return true;
	}

	function setDisplayMode(mode: "align" | "overlap"): void {
		displayMode = mode;
		statusText = mode === "align"
			? "对齐模式：默认只显示当前层"
			: "交叠预览：当前层突出、下一业务层淡显";
		requestDraw();
	}

	function toggleGroupVisibility(evt: MouseEvent, key: string): void {
		evt.stopPropagation();
		const nextHidden = new Set(hiddenGroupKeys);
		const nextExplicit = new Set(explicitVisibleGroupKeys);
		if (isGroupVisible(key)) {
			nextHidden.add(key);
			nextExplicit.delete(key);
		} else {
			nextHidden.delete(key);
			nextExplicit.add(key);
		}
		hiddenGroupKeys = nextHidden;
		explicitVisibleGroupKeys = nextExplicit;
		statusText = "显示设置仅保存在浏览器，不改变保存数据";
		requestDraw();
	}

	function toggleSoloGroup(evt: MouseEvent, key: string): void {
		evt.stopPropagation();
		soloGroupKey = soloGroupKey === key ? "" : key;
		statusText = soloGroupKey ? "已单独显示一层" : "已退出单层显示";
		requestDraw();
	}

	function resetView(): void {
		viewScale = 1;
		viewPanX = 0;
		viewPanY = 0;
		statusText = "画布视图已复位（不影响保存数据）";
		requestDraw();
	}

	function zoomView(factor: number): void {
		const next = clamp(viewScale * factor, 0.2, 8);
		const displayX = targetWidth() / 2;
		const displayY = targetHeight() / 2;
		const targetX = (displayX - viewPanX) / viewScale;
		const targetY = (displayY - viewPanY) / viewScale;
		viewPanX = displayX - targetX * next;
		viewPanY = displayY - targetY * next;
		viewScale = next;
		statusText = "画布视图 " + Math.round(viewScale * 100) + "%（仅本地）";
		requestDraw();
	}

	function numericTransformValue(field: NumericField, raw: number): LayoutTransform {
		if (field === "x") return { ...transform, center_x: raw };
		if (field === "y") return { ...transform, center_y: raw };
		if (field === "scale") return { ...transform, scale: clamp(raw / 100, 0.01, 20) };
		if (field === "rotation") return { ...transform, rotation_deg: normalizeRotation(raw) };
		return { ...transform, preview_alpha: clamp(raw / 100, 0, 1) };
	}

	function commitNumericField(field: NumericField): void {
		const raw = Number(numericDraft[field]);
		editingNumericField = "";
		if (!Number.isFinite(raw)) {
			refreshNumericDraft();
			return;
		}
		const before = normalizeTransform(transform);
		const next = normalizeTransform(numericTransformValue(field, raw));
		const unchanged = (
			Math.abs(next.center_x - before.center_x) < 1e-9
			&& Math.abs(next.center_y - before.center_y) < 1e-9
			&& Math.abs(next.scale - before.scale) < 1e-12
			&& Math.abs(next.rotation_deg - before.rotation_deg) < 1e-9
			&& Math.abs(next.preview_alpha - before.preview_alpha) < 1e-12
		);
		setActiveTransform(next);
		if (unchanged) {
			requestDraw();
			return;
		}
		queueSyncValue("numeric", activeGroupLabel() + " 参数已更新");
	}

	function onNumericInput(evt: Event, field: NumericField): void {
		numericDraft = {
			...numericDraft,
			[field]: (evt.currentTarget as HTMLInputElement).value,
		};
	}

	function numericStep(field: NumericField, evt: KeyboardEvent): number {
		const direction = evt.key === "ArrowUp" ? 1 : -1;
		const regular = field === "x" || field === "y" ? 1 : (field === "alpha" ? 1 : 0.1);
		const fine = field === "x" || field === "y" ? 0.1 : (field === "alpha" ? 0.1 : 0.01);
		const fast = field === "x" || field === "y" ? 10 : (field === "alpha" ? 5 : 1);
		return direction * (evt.shiftKey ? fine : (evt.altKey ? fast : regular));
	}

	function onNumericKeydown(evt: KeyboardEvent, field: NumericField): void {
		if (evt.key === "Escape") {
			evt.preventDefault();
			editingNumericField = "";
			refreshNumericDraft();
			(evt.currentTarget as HTMLInputElement).blur();
			return;
		}
		if (evt.key === "Enter") {
			evt.preventDefault();
			commitNumericField(field);
			(evt.currentTarget as HTMLInputElement).blur();
			return;
		}
		if (evt.key === "ArrowUp" || evt.key === "ArrowDown") {
			evt.preventDefault();
			const current = finiteNumber(numericDraft[field], 0);
			numericDraft = {
				...numericDraft,
				[field]: String(current + numericStep(field, evt)),
			};
			commitNumericField(field);
		}
	}

	function onWindowKeydown(evt: KeyboardEvent): void {
		if (
			evt.code === "Space"
			&& !(evt.target instanceof HTMLInputElement)
			&& !(evt.target instanceof HTMLTextAreaElement)
		) {
			spacePressed = true;
		}
	}

	function onWindowKeyup(evt: KeyboardEvent): void {
		if (evt.code === "Space") spacePressed = false;
	}

	function clearGroupDragState(): void {
		draggedGroupKey = "";
		dragOverGroupKey = "";
		dragOverPosition = "before";
	}

	function onGroupDragStart(evt: DragEvent, key: string): void {
		if (!canReorderGroups()) return;
		draggedGroupKey = key;
		dragOverGroupKey = "";
		dragOverPosition = "before";
		suppressGroupClick = false;
		if (evt.dataTransfer) {
			evt.dataTransfer.effectAllowed = "move";
			evt.dataTransfer.setData("text/plain", key);
		}
	}

	function onGroupDragOver(evt: DragEvent, key: string): void {
		if (!canReorderGroups() || !draggedGroupKey) return;
		evt.preventDefault();
		if (evt.dataTransfer) evt.dataTransfer.dropEffect = "move";
		const target = evt.currentTarget as HTMLElement;
		const rect = target.getBoundingClientRect();
		dragOverGroupKey = key;
		dragOverPosition = evt.clientY < rect.top + rect.height / 2 ? "before" : "after";
	}

	function onGroupDrop(evt: DragEvent, key: string): void {
		if (!canReorderGroups()) return;
		evt.preventDefault();
		const sourceKey = draggedGroupKey || evt.dataTransfer?.getData("text/plain") || "";
		const position = dragOverGroupKey === key ? dragOverPosition : "before";
		const changed = reorderGroupViews(sourceKey, key, position);
		clearGroupDragState();
		if (changed) {
			suppressGroupClick = true;
			publishGroupIntent("reorder_groups", "已调整叠层顺序；拖动调整叠层顺序", false, true);
			setTimeout(() => {
				suppressGroupClick = false;
			}, 0);
		}
	}

	function onGroupDragEnd(): void {
		clearGroupDragState();
	}

	function reorderGroupViews(sourceKey: string, targetKey: string, position: "before" | "after"): boolean {
		if (!canReorderGroups() || sourceKey === targetKey) return false;
		const view = localValue.group_view;
		const groups = groupViews();
		const sourceIndex = groups.findIndex((group) => groupKey(group.group_id) === sourceKey);
		const targetIndex = groups.findIndex((group) => groupKey(group.group_id) === targetKey);
		if (!view || sourceIndex < 0 || targetIndex < 0) return false;
		const nextGroups = groups.slice();
		const [moving] = nextGroups.splice(sourceIndex, 1);
		if (!moving) return false;
		const nextTargetIndex = nextGroups.findIndex((group) => groupKey(group.group_id) === targetKey);
		if (nextTargetIndex < 0) return false;
		const insertAt = nextTargetIndex + (position === "after" ? 1 : 0);
		nextGroups.splice(insertAt, 0, moving);
		const changed = nextGroups.some((group, index) => groupKey(group.group_id) !== groupKey(groups[index].group_id));
		if (!changed) return false;
		localValue = { ...localValue, group_view: { ...view, groups: nextGroups } };
		return true;
	}

	function onGroupRowClick(evt: MouseEvent, key: string): void {
		if (suppressGroupClick) {
			suppressGroupClick = false;
			return;
		}
		const target = evt.target as HTMLElement | null;
		if (target?.closest(".group-drag-handle")) return;
		if (setActiveGroupLocal(key)) {
			publishGroupIntent("select_group", "已选择 Label：" + activeGroupLabel(), false, true);
		}
	}

	function onActiveSelect(evt: Event): void {
		const key = (evt.currentTarget as HTMLSelectElement).value;
		if (setActiveGroupLocal(key)) {
			publishGroupIntent("select_group", "已选择 Label：" + activeGroupLabel(), false, true);
		}
	}

	function beginCanvasPan(evt: PointerEvent): void {
		panning = true;
		cursorStyle = "grabbing";
		panStart = {
			client_x: evt.clientX,
			client_y: evt.clientY,
			pan_x: viewPanX,
			pan_y: viewPanY,
		};
		canvasEl.setPointerCapture(evt.pointerId);
	}

	function onPointerDown(evt: PointerEvent): void {
		if (evt.button !== 0) return;
		clearQueuedSync();
		const p = eventToTarget(evt);
		if (spacePressed) {
			beginCanvasPan(evt);
			return;
		}
		if (hasEditableMask() && hitRotateHandle(p.x, p.y)) {
			cursorStyle = "grabbing";
			rotating = true;
			rotateStart = {
				angle: Math.atan2(p.y - transform.center_y, p.x - transform.center_x) * 180 / Math.PI,
				rotation: Number(transform.rotation_deg || 0),
			};
			canvasEl.setPointerCapture(evt.pointerId);
			return;
		}
		let hit = false;
		if (hasEditableMask() && isGroupMode()) {
			const hitKey = hitGroupAt(p.x, p.y);
			if (hitKey) {
				setActiveGroupLocal(hitKey);
				hit = true;
			}
		} else if (hasEditableMask()) {
			hit = hitLegacyMask(p.x, p.y);
		}
		if (!hit) {
			beginCanvasPan(evt);
			statusText = "正在平移画布视图（仅本地）";
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
		if (spacePressed) {
			cursorStyle = "grab";
			return;
		}
		if (!hasEditableMask()) {
			cursorStyle = "grab";
			return;
		}
		const hit = isGroupMode() ? !!hitGroupAt(p.x, p.y) : hitLegacyMask(p.x, p.y);
		cursorStyle = hitRotateHandle(p.x, p.y) || hit ? "grab" : "grab";
	}

	function onPointerMove(evt: PointerEvent): void {
		if (panning) {
			const rect = canvasEl.getBoundingClientRect();
			viewPanX = panStart.pan_x + (
				(evt.clientX - panStart.client_x) * targetWidth() / Math.max(1, rect.width)
			);
			viewPanY = panStart.pan_y + (
				(evt.clientY - panStart.client_y) * targetHeight() / Math.max(1, rect.height)
			);
			statusText = "正在平移画布视图（仅本地）";
			requestDraw();
			return;
		}
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
			statusText = "正在拖动 " + (isGroupMode() ? activeGroupLabel() : "版图") + "（本地）";
		} else if (rotating) {
			const angle = Math.atan2(p.y - transform.center_y, p.x - transform.center_x) * 180 / Math.PI;
			setActiveTransform({
				...transform,
				rotation_deg: normalizeRotation(rotateStart.rotation + angle - rotateStart.angle),
			});
			statusText = "正在旋转 " + (isGroupMode() ? activeGroupLabel() : "版图") + "（本地）";
		}
		requestDraw();
	}

	function onPointerLeave(): void {
		if (!dragging && !rotating && !panning) cursorStyle = "grab";
	}

	function onPointerUp(evt: PointerEvent): void {
		if (panning) {
			panning = false;
			try {
				canvasEl.releasePointerCapture(evt.pointerId);
			} catch (_) {
				// Pointer capture may already have been released by the browser.
			}
			statusText = "画布视图已调整（仅本地，不影响保存数据）";
			updateCursor(eventToTarget(evt));
			return;
		}
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
			const status = subject + " 已在本地更新，停止操作 250ms 后静默同步";
			if (isGroupMode()) {
				queueSyncValue("canvas", status);
			} else {
				syncValue("canvas", status);
			}
		}
	}

	function wheelPixels(evt: WheelEvent): number {
		if (evt.deltaMode === WheelEvent.DOM_DELTA_LINE) return evt.deltaY * 16;
		if (evt.deltaMode === WheelEvent.DOM_DELTA_PAGE) {
			return evt.deltaY * Math.max(1, canvasEl.getBoundingClientRect().height);
		}
		return evt.deltaY;
	}

	function onWheel(evt: WheelEvent): void {
		evt.preventDefault();
		const rect = canvasEl.getBoundingClientRect();
		const displayX = ((evt.clientX - rect.left) / Math.max(1, rect.width)) * targetWidth();
		const displayY = ((evt.clientY - rect.top) / Math.max(1, rect.height)) * targetHeight();
		if (evt.ctrlKey || evt.metaKey) {
			const targetX = (displayX - viewPanX) / viewScale;
			const targetY = (displayY - viewPanY) / viewScale;
			const nextViewScale = clamp(
				viewScale * Math.exp(-wheelPixels(evt) * 0.0015),
				0.2,
				8,
			);
			viewPanX = displayX - targetX * nextViewScale;
			viewPanY = displayY - targetY * nextViewScale;
			viewScale = nextViewScale;
			statusText = "画布视图 " + Math.round(viewScale * 100) + "%（仅本地）";
			requestDraw();
			return;
		}
		if (!hasEditableMask()) return;
		const p = eventToTarget(evt);
		if (isGroupMode()) {
			const hitKey = hitGroupAt(p.x, p.y);
			if (hitKey) setActiveGroupLocal(hitKey);
		}
		const before = targetToSource(p.x, p.y, transform);
		const modifier = evt.shiftKey ? 0.1 : (evt.altKey ? 5 : 1);
		const factor = Math.exp(-wheelPixels(evt) * WHEEL_ZOOM_LOG_PER_PIXEL * modifier);
		const nextScale = clamp(Number(transform.scale || 1) * factor, 0.01, 20);
		let next = { ...transform, scale: nextScale };
		const afterTarget = sourceToTarget(before.x, before.y, next);
		next = {
			...next,
			center_x: Number(next.center_x || 0) + p.x - afterTarget.x,
			center_y: Number(next.center_y || 0) + p.y - afterTarget.y,
		};
		setActiveTransform(next);
		queueSyncValue(
			"canvas",
			"本地缩放 " + (nextScale * 100).toFixed(2) + "%；停止 250ms 后同步",
		);
		requestDraw();
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

<svelte:window on:keydown={onWindowKeydown} on:keyup={onWindowKeyup} />

<Block
	visible={gradio.shared.visible}
	variant="solid"
	border_mode={dragging || rotating || panning ? "focus" : "base"}
	padding={false}
	elem_id={gradio.shared.elem_id}
	elem_classes={gradio.shared.elem_classes}
	allow_overflow={false}
	container={gradio.shared.container}
	scale={gradio.shared.scale}
	min_width={gradio.shared.min_width}
>
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
				<div class="mode-switch" aria-label="显示模式">
					<button
						type="button"
						class:active={displayMode === "align"}
						on:click={() => setDisplayMode("align")}
					>对齐模式</button>
					<button
						type="button"
						class:active={displayMode === "overlap"}
						on:click={() => setDisplayMode("overlap")}
					>交叠预览</button>
				</div>
				{#if canReorderGroups()}
					<div class="group-reorder" aria-label="调整 Label 叠层顺序">
						<div class="group-reorder-hint">拖动排序；眼睛与单层显示仅影响浏览器预览</div>
						<div class="group-reorder-list">
							{#each groupViews() as group, index}
								<div
									class="group-reorder-item"
									class:active={groupKey(group.group_id) === activeGroupKey}
									class:drop-before={dragOverGroupKey === groupKey(group.group_id) && dragOverPosition === "before"}
									class:drop-after={dragOverGroupKey === groupKey(group.group_id) && dragOverPosition === "after"}
									role="button"
									tabindex="0"
									on:click={(evt) => onGroupRowClick(evt, groupKey(group.group_id))}
									on:keydown={(evt) => {
										if (evt.key === "Enter" || evt.key === " ") {
											evt.preventDefault();
											onGroupRowClick(evt as unknown as MouseEvent, groupKey(group.group_id));
										}
									}}
									on:dragover={(evt) => onGroupDragOver(evt, groupKey(group.group_id))}
									on:drop={(evt) => onGroupDrop(evt, groupKey(group.group_id))}
								>
									<span
										class="group-drag-handle"
										draggable={true}
										title="拖动调整叠层顺序"
										aria-label="拖动调整叠层顺序"
										on:dragstart={(evt) => onGroupDragStart(evt, groupKey(group.group_id))}
										on:dragend={onGroupDragEnd}
									>⠿</span>
									<span class="group-order">{index + 1}</span>
									<span class="group-reorder-label">
										{group.label || "Label " + String(group.group_id)}
										{#if groupKey(group.group_id) === activeGroupKey}
											<span class="current-badge">当前</span>
										{/if}
									</span>
									<button
										type="button"
										class="visibility-button"
										title={isGroupVisible(groupKey(group.group_id)) ? "隐藏此层" : "显示此层"}
										on:click={(evt) => toggleGroupVisibility(evt, groupKey(group.group_id))}
									>{isGroupVisible(groupKey(group.group_id)) ? "👁" : "—"}</button>
									<button
										type="button"
										class="solo-button"
										class:active={soloGroupKey === groupKey(group.group_id)}
										title="只显示此层"
										on:click={(evt) => toggleSoloGroup(evt, groupKey(group.group_id))}
									>S</button>
								</div>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		{/if}
		<div class="toolbar">
			<button type="button" on:click={resetTransform} disabled={!hasEditableMask()}>居中归一</button>
			<button type="button" on:click={centerTransform} disabled={!hasEditableMask()}>居中</button>
			<button type="button" on:click={fitTransform} disabled={!hasEditableMask()}>适配</button>
			<button type="button" on:click={bringIntoView} disabled={!hasEditableMask()}>找回标注</button>
			<button type="button" on:click={() => zoomView(0.8)}>画布 −</button>
			<button type="button" on:click={resetView}>{Math.round(viewScale * 100)}%</button>
			<button type="button" on:click={() => zoomView(1.25)}>画布 ＋</button>
		</div>
		{#if hasEditableMask()}
			<div class="numeric-toolbar" aria-label="当前层精细参数">
				{#each [
					["x", "X", "px"],
					["y", "Y", "px"],
					["scale", "缩放", "%"],
					["rotation", "旋转", "°"],
					["alpha", "透明度", "%"],
				] as item}
					<label>
						<span>{item[1]}</span>
						<input
							type="number"
							value={numericDraft[item[0] as NumericField]}
							on:focus={() => editingNumericField = item[0] as NumericField}
							on:input={(evt) => onNumericInput(evt, item[0] as NumericField)}
							on:keydown={(evt) => onNumericKeydown(evt, item[0] as NumericField)}
							on:blur={() => commitNumericField(item[0] as NumericField)}
						/>
						<small>{item[2]}</small>
					</label>
				{/each}
			</div>
		{/if}
		{#if alignmentProposal}
			<div class="alignment-proposal" class:ambiguous={alignmentProposal.ambiguous === true}>
				<div class="proposal-copy">
					<strong>
						{alignmentProposal.ambiguous ? "存在周期歧义" : "粗对齐建议"}
					</strong>
					<span>{alignmentProposal.status}</span>
				</div>
				{#if proposalCandidates().length}
					<div class="proposal-actions">
						{#each proposalCandidates() as candidate}
							<button
								type="button"
								on:click={() => applyAlignmentCandidate(candidate)}
							>
								应用 {candidate.candidate_id}
								· 支持率 {Math.round(finiteNumber(candidate.metrics?.central_support, 0) * 100)}%
								· 分数 {finiteNumber(candidate.metrics?.score, 0).toFixed(3)}
							</button>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
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
		{#if isGroupMode() && displayMode === "overlap" && nextBusinessGroupKey()}
			<div class="overlap-legend">
				<span class="legend-current">当前：{activeGroupLabel()}</span>
				<span class="legend-next">下一层：{groupRuntimeByKey.get(nextBusinessGroupKey())?.view.label}</span>
				<span class="legend-overlap">{activeGroupLabel()} × {groupRuntimeByKey.get(nextBusinessGroupKey())?.view.label}</span>
			</div>
		{/if}
		<div class="status">
			<span class:dirty={clientRevision > serverAckRevision}>
				{clientRevision > serverAckRevision
					? (syncInFlightRevision === null ? "本地修改" : "同步中")
					: "已同步"}
			</span>
			<span>{statusText}</span>
		</div>
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
	.group-reorder {
		grid-column: 1 / -1;
		display: flex;
		flex-direction: column;
		gap: 4px;
		min-width: 0;
	}
	.group-reorder-hint {
		font-size: 11px;
		color: #64748b;
	}
	.group-reorder-list {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.group-reorder-item {
		display: grid;
		grid-template-columns: auto auto minmax(0, 1fr) auto auto;
		align-items: center;
		gap: 5px;
		min-width: 120px;
		max-width: 100%;
		border: 1px solid #cbd5e1;
		border-radius: 6px;
		background: #ffffff;
		color: #0f172a;
		padding: 5px 8px 5px 6px;
		text-align: left;
		font: inherit;
		cursor: pointer;
		user-select: none;
	}

	.group-reorder-item.active {
		background: #e0f2fe;
		border-color: #0284c7;
		color: #0c4a6e;
	}

	.group-reorder-item.drop-before {
		box-shadow: inset 3px 0 #2563eb;
	}
	.group-reorder-item.drop-after {
		box-shadow: inset -3px 0 #2563eb;
	}

	.group-drag-handle {
		font-size: 16px;
		cursor: grab;
		user-select: none;
		touch-action: none;
	}
	.group-drag-handle:active {
		cursor: grabbing;
	}

	.group-order {
		font-size: 11px;
		font-weight: 700;
		color: #64748b;
		min-width: 16px;
		text-align: center;
	}

	.group-reorder-label {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.toolbar {
		display: grid;
		grid-template-columns: repeat(7, minmax(0, 1fr));
		gap: 6px;
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
	.mode-switch {
		grid-column: 1 / -1;
		display: inline-flex;
		gap: 4px;
		padding: 3px;
		border-radius: 8px;
		background: #e2e8f0;
		width: fit-content;
	}
	.mode-switch button {
		border: 0;
		border-radius: 6px;
		background: transparent;
		color: #475569;
		font-weight: 700;
		padding: 5px 12px;
		cursor: pointer;
	}
	.mode-switch button.active {
		background: #ffffff;
		color: #0f766e;
		box-shadow: 0 1px 3px rgba(15, 23, 42, 0.14);
	}
	.current-badge {
		margin-left: 5px;
		border-radius: 999px;
		background: #0284c7;
		color: #ffffff;
		font-size: 10px;
		font-weight: 700;
		padding: 1px 5px;
	}
	.visibility-button,
	.solo-button {
		width: 28px;
		height: 26px;
		border: 1px solid #cbd5e1;
		border-radius: 5px;
		background: #ffffff;
		color: #475569;
		font-weight: 700;
		cursor: pointer;
	}
	.solo-button.active {
		border-color: #7c3aed;
		background: #ede9fe;
		color: #6d28d9;
	}
	.numeric-toolbar {
		display: grid;
		grid-template-columns: repeat(5, minmax(92px, 1fr));
		gap: 6px;
	}
	.numeric-toolbar label {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		align-items: center;
		gap: 5px;
		border: 1px solid #dbe4ef;
		border-radius: 7px;
		background: #ffffff;
		padding: 4px 6px;
		font-size: 11px;
		font-weight: 700;
		color: #334155;
	}
	.numeric-toolbar input {
		min-width: 0;
		width: 100%;
		border: 0;
		outline: 0;
		background: transparent;
		text-align: right;
		font-variant-numeric: tabular-nums;
		color: #0f172a;
	}
	.numeric-toolbar small {
		font-size: 10px;
		color: #94a3b8;
	}
	.alignment-proposal {
		display: grid;
		grid-template-columns: minmax(180px, 0.8fr) minmax(0, 1.2fr);
		gap: 8px;
		align-items: center;
		border: 1px solid #93c5fd;
		border-radius: 8px;
		background: #eff6ff;
		padding: 7px 9px;
	}
	.alignment-proposal.ambiguous {
		border-color: #fbbf24;
		background: #fffbeb;
	}
	.proposal-copy {
		display: flex;
		flex-direction: column;
		gap: 2px;
		font-size: 11px;
		color: #475569;
	}
	.proposal-copy strong {
		color: #1e3a8a;
		font-size: 12px;
	}
	.alignment-proposal.ambiguous .proposal-copy strong {
		color: #92400e;
	}
	.proposal-actions {
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: 5px;
	}
	.proposal-actions button {
		border: 1px solid #60a5fa;
		border-radius: 6px;
		background: #ffffff;
		color: #1d4ed8;
		font-size: 11px;
		font-weight: 700;
		padding: 5px 8px;
		cursor: pointer;
	}
	.proposal-actions button:hover {
		background: #dbeafe;
	}
	.overlap-legend {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		font-size: 11px;
		font-weight: 700;
	}
	.overlap-legend span {
		border-radius: 999px;
		padding: 3px 8px;
	}
	.legend-current {
		background: rgba(0, 255, 120, 0.18);
		color: #047857;
	}
	.legend-next {
		border: 1px dashed #0284c7;
		background: rgba(0, 188, 255, 0.08);
		color: #0369a1;
	}
	.legend-overlap {
		background: repeating-linear-gradient(
			135deg,
			#fce7f3,
			#fce7f3 4px,
			#fbcfe8 4px,
			#fbcfe8 8px
		);
		color: #be185d;
	}
	.status {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 12px;
		line-height: 1.35;
		color: #475569;
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 6px;
		padding: 6px 8px;
		min-height: 28px;
	}
	.status > span:first-child {
		flex: none;
		border-radius: 999px;
		background: #dcfce7;
		color: #166534;
		font-size: 10px;
		font-weight: 800;
		padding: 2px 7px;
	}
	.status > span:first-child.dirty {
		background: #fef3c7;
		color: #92400e;
	}
	@media (max-width: 760px) {
		.toolbar {
			grid-template-columns: repeat(4, minmax(0, 1fr));
		}
		.numeric-toolbar {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
</style>
