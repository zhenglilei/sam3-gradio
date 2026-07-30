<svelte:options accessors={true} />

<script lang="ts">
	import { Block } from "@gradio/atoms";
	import { StatusTracker } from "@gradio/statustracker";
	import { Gradio } from "@gradio/utils";
	import { onMount } from "svelte";
	import type {
		Gesture,
		GestureClientIntent,
		ImageGestureOverlayEvents,
		ImageGestureOverlayProps,
		ImageGestureOverlayValue,
		Interaction,
		PointTuple,
	} from "./types";

	const props = $props();
	const gradio = new Gradio<ImageGestureOverlayEvents, ImageGestureOverlayProps>(props);
	const MIN_DRAG_NATURAL_PX = 4;
	const IMAGE_TRANSITION_WAIT_MS = 500;

	type Point = { x: number; y: number };
	type Rect = { left: number; top: number; width: number; height: number };

	let surfaceEl: HTMLDivElement;
	let localValue = $state<ImageGestureOverlayValue>({});
	let surfaceWidth = $state(1);
	let surfaceHeight = $state(1);
	let surfaceLeft = $state(0);
	let surfaceTop = $state(0);
	let targetReady = $state(false);
	let activePointerId: number | null = null;
	let activePointerType = "mouse";
	let pointerStartClient: Point | null = null;
	let gestureStart = $state<Point | null>(null);
	let gestureEnd = $state<Point | null>(null);
	let committedStart = $state<Point | null>(null);
	let committedEnd = $state<Point | null>(null);
	let previousCommittedStart: Point | null = null;
	let previousCommittedEnd: Point | null = null;
	let maxCssDistance = 0;
	let lastSignature = "";
	let loadedImageIdentity = "";
	let loadedViewIdentity = "";
	let mounted = false;
	let observedTargetId = "";
	let targetHost: HTMLElement | null = null;
	let targetImage: HTMLImageElement | null = null;
	let targetResizeObserver: ResizeObserver | null = null;
	let targetMutationObserver: MutationObserver | null = null;
	let retryTimer: ReturnType<typeof setTimeout> | null = null;
	let transitionTimer: ReturnType<typeof setTimeout> | null = null;
	let targetImageGeneration = 0;
	let waitingPreviousSource = "";
	let boundImageIdentity = "";
	let boundImageSource = "";

	function cloneValue(value: ImageGestureOverlayValue | null | undefined): ImageGestureOverlayValue {
		return JSON.parse(JSON.stringify(value || {})) as ImageGestureOverlayValue;
	}

	function heightStyle(value: number | string | undefined): string {
		if (typeof value === "number") return `${value}px`;
		return value || "320px";
	}

	function targetElemId(): string {
		return String(gradio.props.target_elem_id || "").replace(/^#/, "").trim();
	}

	function serverView() {
		return localValue.server_view || {};
	}

	function interaction(): Interaction {
		const value = String(serverView().interaction || "auto").toLowerCase();
		if (
			value === "workspace" ||
			value === "crop" ||
			value === "drag" ||
			value === "bbox" ||
			value === "click" ||
			value === "point" ||
			value === "polygon" ||
			value === "disabled"
		) {
			return value;
		}
		return "auto";
	}

	function clickDistanceCssPx(pointerType: string): number {
		if (pointerType === "touch") return 12;
		if (pointerType === "pen") return 7;
		return 4;
	}

	function minDragCssPx(pointerType: string): number {
		if (pointerType === "touch") return 12;
		if (pointerType === "pen") return 7;
		return 5;
	}

	function selectionState(): string {
		return String(serverView().selection_state || "").toLowerCase();
	}

	function clickAllowed(): boolean {
		return ["auto", "workspace", "click", "point", "polygon"].includes(interaction());
	}

	function dragAllowed(): boolean {
		return ["auto", "workspace", "crop", "drag", "bbox"].includes(interaction());
	}

	function naturalWidth(): number {
		const value = Number(serverView().natural_width || 0);
		return Number.isFinite(value) && value > 0 ? value : 0;
	}

	function naturalHeight(): number {
		const value = Number(serverView().natural_height || 0);
		return Number.isFinite(value) && value > 0 ? value : 0;
	}

	function enabled(): boolean {
		return (
			serverView().enabled === true &&
			interaction() !== "disabled" &&
			(!targetElemId() || targetReady) &&
			naturalWidth() > 0 &&
			naturalHeight() > 0
		);
	}

	function imageIdentitySignature(): string {
		const view = serverView();
		return JSON.stringify([
			view.image_id || "",
			view.image_sha256 || "",
			naturalWidth(),
			naturalHeight(),
		]);
	}

	function viewIdentitySignature(): string {
		const view = serverView();
		return JSON.stringify([
			view.enabled === true,
			imageIdentitySignature(),
			Number(view.revision || 0),
			interaction(),
		]);
	}

	function tuplePoint(value: number[] | undefined): Point | null {
		if (!Array.isArray(value) || value.length !== 2) return null;
		const x = Number(value[0]);
		const y = Number(value[1]);
		return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
	}

	function releasePointer(): void {
		const pointerId = activePointerId;
		activePointerId = null;
		if (pointerId !== null && surfaceEl?.hasPointerCapture(pointerId)) {
			surfaceEl.releasePointerCapture(pointerId);
		}
		activePointerType = "mouse";
		pointerStartClient = null;
		gestureStart = null;
		gestureEnd = null;
		previousCommittedStart = null;
		previousCommittedEnd = null;
		maxCssDistance = 0;
	}

	function clearTransitionTimer(): void {
		if (transitionTimer !== null) clearTimeout(transitionTimer);
		transitionTimer = null;
	}

	function releaseTransitionWait(expectedIdentity: string): void {
		transitionTimer = null;
		if (expectedIdentity !== loadedImageIdentity || !waitingPreviousSource) return;
		waitingPreviousSource = "";
		if (targetImage) requestTargetActivation(targetImage);
	}

	function beginImageTransition(previousSource: string): void {
		clearTransitionTimer();
		targetReady = false;
		waitingPreviousSource = previousSource;
		boundImageIdentity = "";
		boundImageSource = "";
		targetImageGeneration += 1;
		if (previousSource) {
			const expectedIdentity = loadedImageIdentity;
			transitionTimer = setTimeout(
				() => releaseTransitionWait(expectedIdentity),
				IMAGE_TRANSITION_WAIT_MS,
			);
		}
	}

	function ingestValue(value: ImageGestureOverlayValue | null): void {
		const previousSource = boundImageSource || targetSource(targetImage);
		const next = cloneValue(value);
		localValue = next;
		const nextImageIdentity = imageIdentitySignature();
		const nextViewIdentity = viewIdentitySignature();
		const imageChanged = nextImageIdentity !== loadedImageIdentity;
		const viewChanged = nextViewIdentity !== loadedViewIdentity;
		if (viewChanged) {
			releasePointer();
		}
		loadedViewIdentity = nextViewIdentity;
		if (imageChanged) {
			loadedImageIdentity = nextImageIdentity;
			beginImageTransition(previousSource);
		}
		const intent = next.client_intent || {};
		if (intent.gesture === "drag") {
			committedStart = tuplePoint(intent.start_xy);
			committedEnd = tuplePoint(intent.end_xy);
		} else {
			committedStart = null;
			committedEnd = null;
		}
		if (mounted && targetElemId()) {
			if (imageChanged) bindTargetImage();
			else measureTarget();
		}
	}

	$effect(() => {
		const signature = JSON.stringify(gradio.props.value || null);
		if (signature !== lastSignature) {
			lastSignature = signature;
			ingestValue(gradio.props.value);
		}
	});

	$effect(() => {
		const nextTargetId = targetElemId();
		if (nextTargetId !== observedTargetId) {
			observedTargetId = nextTargetId;
			if (mounted) connectTarget();
		}
	});

	function observeSize(node: HTMLDivElement) {
		const update = () => {
			if (targetElemId()) return;
			const rect = node.getBoundingClientRect();
			surfaceWidth = Math.max(1, rect.width);
			surfaceHeight = Math.max(1, rect.height);
		};
		const observer = new ResizeObserver(update);
		observer.observe(node);
		update();
		return { destroy: () => observer.disconnect() };
	}

	function targetSource(image: HTMLImageElement | null): string {
		return String(image?.currentSrc || image?.src || "");
	}

	function clearTargetObservers(): void {
		clearTransitionTimer();
		if (retryTimer !== null) clearTimeout(retryTimer);
		retryTimer = null;
		targetResizeObserver?.disconnect();
		targetResizeObserver = null;
		targetMutationObserver?.disconnect();
		targetMutationObserver = null;
		targetImage?.removeEventListener("load", handleTargetImageLoad);
		targetHost = null;
		targetImage = null;
		targetReady = false;
		targetImageGeneration += 1;
		waitingPreviousSource = "";
		boundImageIdentity = "";
		boundImageSource = "";
	}

	function objectPositionOffset(freeSpace: number, token: string | undefined): number {
		if (!token || token === "center") return freeSpace / 2;
		if (token === "left" || token === "top") return 0;
		if (token === "right" || token === "bottom") return freeSpace;
		if (token.endsWith("%")) {
			const ratio = Number.parseFloat(token) / 100;
			return Number.isFinite(ratio) ? freeSpace * ratio : freeSpace / 2;
		}
		const pixels = Number.parseFloat(token);
		return Number.isFinite(pixels) ? pixels : freeSpace / 2;
	}

	function measureTarget(): void {
		const source = targetSource(targetImage);
		if (
			!targetImage ||
			!targetElemId() ||
			boundImageIdentity !== loadedImageIdentity ||
			!source ||
			source !== boundImageSource
		) {
			targetReady = false;
			return;
		}
		const imageElementRect = targetImage.getBoundingClientRect();
		const width = naturalWidth();
		const height = naturalHeight();
		if (imageElementRect.width <= 0 || imageElementRect.height <= 0 || width <= 0 || height <= 0) {
			targetReady = false;
			return;
		}
		const scale = Math.min(imageElementRect.width / width, imageElementRect.height / height);
		const renderedWidth = width * scale;
		const renderedHeight = height * scale;
		const position = getComputedStyle(targetImage).objectPosition.split(/\s+/);
		surfaceLeft = imageElementRect.left + objectPositionOffset(imageElementRect.width - renderedWidth, position[0]);
		surfaceTop =
			imageElementRect.top + objectPositionOffset(imageElementRect.height - renderedHeight, position[1] || position[0]);
		surfaceWidth = renderedWidth;
		surfaceHeight = renderedHeight;
		targetReady = renderedWidth > 0 && renderedHeight > 0;
	}

	async function activateTargetImage(
		image: HTMLImageElement,
		identity: string,
		source: string,
		generation: number,
	): Promise<void> {
		if (!image.complete || image.naturalWidth <= 0) return;
		try {
			await image.decode();
		} catch {
			if (!image.complete || image.naturalWidth <= 0) return;
		}
		if (
			generation !== targetImageGeneration ||
			identity !== loadedImageIdentity ||
			image !== targetImage ||
			source !== targetSource(image)
		) return;
		boundImageIdentity = identity;
		boundImageSource = source;
		clearTransitionTimer();
		waitingPreviousSource = "";
		measureTarget();
	}

	function requestTargetActivation(image: HTMLImageElement): void {
		const source = targetSource(image);
		if (!source) {
			targetReady = false;
			return;
		}
		if (waitingPreviousSource && source === waitingPreviousSource) {
			targetReady = false;
			return;
		}
		clearTransitionTimer();
		waitingPreviousSource = "";
		const generation = ++targetImageGeneration;
		void activateTargetImage(image, loadedImageIdentity, source, generation);
	}

	function handleTargetImageLoad(): void {
		if (targetImage) requestTargetActivation(targetImage);
	}

	function bindTargetImage(): void {
		if (!targetHost) return;
		const images = Array.from(targetHost.querySelectorAll("img"));
		const image = images.sort((a, b) => {
			const aRect = a.getBoundingClientRect();
			const bRect = b.getBoundingClientRect();
			return bRect.width * bRect.height - aRect.width * aRect.height;
		})[0] || null;
		if (image !== targetImage) {
			targetImage?.removeEventListener("load", handleTargetImageLoad);
			targetImage = image;
			targetImage?.addEventListener("load", handleTargetImageLoad);
			targetResizeObserver?.disconnect();
			targetResizeObserver = new ResizeObserver(measureTarget);
			targetResizeObserver.observe(targetHost);
			if (targetImage) targetResizeObserver.observe(targetImage);
		}
		if (!targetImage) {
			targetReady = false;
			boundImageIdentity = "";
			boundImageSource = "";
			return;
		}
		requestTargetActivation(targetImage);
	}

	function connectTarget(): void {
		clearTargetObservers();
		const id = targetElemId();
		if (!id) return;
		targetHost = document.getElementById(id);
		if (!targetHost) {
			retryTimer = setTimeout(connectTarget, 100);
			return;
		}
		targetMutationObserver = new MutationObserver(bindTargetImage);
		targetMutationObserver.observe(targetHost, {
			childList: true,
			subtree: true,
			attributes: true,
			attributeFilter: ["src", "srcset", "style", "class"],
		});
		bindTargetImage();
	}

	onMount(() => {
		mounted = true;
		observedTargetId = targetElemId();
		connectTarget();
		window.addEventListener("resize", measureTarget);
		window.addEventListener("scroll", measureTarget, true);
		window.visualViewport?.addEventListener("resize", measureTarget);
		window.visualViewport?.addEventListener("scroll", measureTarget);
		return () => {
			mounted = false;
			window.removeEventListener("resize", measureTarget);
			window.removeEventListener("scroll", measureTarget, true);
			window.visualViewport?.removeEventListener("resize", measureTarget);
			window.visualViewport?.removeEventListener("scroll", measureTarget);
			clearTargetObservers();
		};
	});

	function containRect(): Rect {
		if (targetElemId()) return { left: 0, top: 0, width: surfaceWidth, height: surfaceHeight };
		const width = naturalWidth();
		const height = naturalHeight();
		if (width <= 0 || height <= 0) return { left: 0, top: 0, width: 0, height: 0 };
		const scale = Math.min(surfaceWidth / width, surfaceHeight / height);
		const displayWidth = width * scale;
		const displayHeight = height * scale;
		return {
			left: (surfaceWidth - displayWidth) / 2,
			top: (surfaceHeight - displayHeight) / 2,
			width: displayWidth,
			height: displayHeight,
		};
	}

	function clamp(value: number, low: number, high: number): number {
		return Math.max(low, Math.min(high, value));
	}

	function eventToNatural(event: PointerEvent, clampOutside: boolean): Point | null {
		const surfaceRect = surfaceEl.getBoundingClientRect();
		const imageRect = containRect();
		const localX = event.clientX - surfaceRect.left;
		const localY = event.clientY - surfaceRect.top;
		const inside =
			localX >= imageRect.left &&
			localX <= imageRect.left + imageRect.width &&
			localY >= imageRect.top &&
			localY <= imageRect.top + imageRect.height;
		if (!inside && !clampOutside) return null;
		if (imageRect.width <= 0 || imageRect.height <= 0) return null;
		return {
			x: clamp(((localX - imageRect.left) / imageRect.width) * naturalWidth(), 0, naturalWidth()),
			y: clamp(((localY - imageRect.top) / imageRect.height) * naturalHeight(), 0, naturalHeight()),
		};
	}

	function pointTuple(point: Point): PointTuple {
		return [Number(point.x.toFixed(4)), Number(point.y.toFixed(4))];
	}

	function publishGesture(gesture: Gesture, start: Point, end: Point): void {
		const view = serverView();
		const intent: GestureClientIntent = {
			gesture,
			start_xy: pointTuple(start),
			end_xy: pointTuple(end),
			expected_revision: Number.isInteger(view.revision) ? Number(view.revision) : null,
			image_id: String(view.image_id || ""),
			image_sha256: String(view.image_sha256 || ""),
		};
		localValue = { ...localValue, client_intent: intent };
		const outbound: ImageGestureOverlayValue = { client_intent: { ...intent } };
		gradio.props.value = outbound;
		lastSignature = JSON.stringify(outbound);
		gradio.dispatch("input");
	}

	function onPointerDown(event: PointerEvent): void {
		if (event.button !== 0 || !enabled() || activePointerId !== null) return;
		const point = eventToNatural(event, false);
		if (!point) return;
		event.preventDefault();
		previousCommittedStart = committedStart ? { ...committedStart } : null;
		previousCommittedEnd = committedEnd ? { ...committedEnd } : null;
		gestureStart = point;
		gestureEnd = point;
		pointerStartClient = { x: event.clientX, y: event.clientY };
		maxCssDistance = 0;
		activePointerId = event.pointerId;
		activePointerType = event.pointerType || "mouse";
		surfaceEl.setPointerCapture(event.pointerId);
	}

	function onPointerMove(event: PointerEvent): void {
		if (event.pointerId !== activePointerId || !gestureStart || !pointerStartClient) return;
		event.preventDefault();
		const point = eventToNatural(event, true);
		if (!point) return;
		maxCssDistance = Math.max(
			maxCssDistance,
			Math.hypot(event.clientX - pointerStartClient.x, event.clientY - pointerStartClient.y),
		);
		if (dragAllowed()) gestureEnd = point;
	}

	function restorePreviousRectangle(): void {
		committedStart = previousCommittedStart;
		committedEnd = previousCommittedEnd;
	}

	function finishPointer(event?: PointerEvent): void {
		const pointerId = activePointerId;
		activePointerId = null;
		if (pointerId !== null && surfaceEl?.hasPointerCapture(pointerId)) {
			surfaceEl.releasePointerCapture(pointerId);
		}
		activePointerType = "mouse";
		pointerStartClient = null;
		gestureStart = null;
		gestureEnd = null;
		previousCommittedStart = null;
		previousCommittedEnd = null;
		maxCssDistance = 0;
		if (event) event.preventDefault();
	}

	function onPointerUp(event: PointerEvent): void {
		if (event.pointerId !== activePointerId || !gestureStart || !pointerStartClient) return;
		const start = { ...gestureStart };
		const end = eventToNatural(event, true) || { ...gestureEnd! };
		maxCssDistance = Math.max(
			maxCssDistance,
			Math.hypot(event.clientX - pointerStartClient.x, event.clientY - pointerStartClient.y),
		);

		const pointerType = activePointerType;
		let gesture: Gesture | null = null;
		let publishStart = start;
		let publishEnd = end;
		if (clickAllowed() && maxCssDistance <= clickDistanceCssPx(pointerType)) {
			gesture = "click";
			publishStart = end;
			publishEnd = end;
			restorePreviousRectangle();
		} else if (
			dragAllowed() &&
			maxCssDistance >= minDragCssPx(pointerType) &&
			Math.abs(end.x - start.x) >= MIN_DRAG_NATURAL_PX &&
			Math.abs(end.y - start.y) >= MIN_DRAG_NATURAL_PX
		) {
			gesture = "drag";
			committedStart = start;
			committedEnd = end;
		} else {
			restorePreviousRectangle();
		}
		finishPointer(event);
		if (gesture) publishGesture(gesture, publishStart, publishEnd);
	}

	function cancelActiveGesture(): void {
		if (activePointerId === null) return;
		restorePreviousRectangle();
		finishPointer();
	}

	function onPointerCancel(event: PointerEvent): void {
		if (event.pointerId !== activePointerId) return;
		cancelActiveGesture();
	}

	function onLostPointerCapture(event: PointerEvent): void {
		if (event.pointerId !== activePointerId) return;
		cancelActiveGesture();
	}

	function onWindowBlur(): void {
		cancelActiveGesture();
	}

	function rectangleStyle(): string | null {
		const start = gestureStart && dragAllowed() ? gestureStart : committedStart;
		const end = gestureEnd && dragAllowed() ? gestureEnd : committedEnd;
		if (!start || !end) return null;
		const rect = containRect();
		if (rect.width <= 0 || rect.height <= 0) return null;
		const left = rect.left + (Math.min(start.x, end.x) / naturalWidth()) * rect.width;
		const top = rect.top + (Math.min(start.y, end.y) / naturalHeight()) * rect.height;
		const width = (Math.abs(end.x - start.x) / naturalWidth()) * rect.width;
		const height = (Math.abs(end.y - start.y) / naturalHeight()) * rect.height;
		return `left:${left}px;top:${top}px;width:${width}px;height:${height}px`;
	}

	function rectangleIsDraft(): boolean {
		return gestureStart !== null || selectionState() !== "applied";
	}

	function cursorStyle(): string {
		if (!enabled()) return "default";
		return dragAllowed() ? "crosshair" : "pointer";
	}

	function surfaceStyle(): string {
		const cursor = cursorStyle();
		if (targetElemId()) {
			return `position:fixed;left:${surfaceLeft}px;top:${surfaceTop}px;width:${surfaceWidth}px;height:${surfaceHeight}px;cursor:${cursor}`;
		}
		return `position:relative;height:${heightStyle(gradio.props.height)};cursor:${cursor}`;
	}
</script>

<svelte:window on:blur={onWindowBlur} />

<Block
	visible={gradio.shared.visible}
	variant="solid"
	border_mode="none"
	padding={false}
	elem_id={gradio.shared.elem_id}
	elem_classes={gradio.shared.elem_classes}
	allow_overflow={true}
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
	<div
		bind:this={surfaceEl}
		use:observeSize
		class="gesture-surface"
		class:disabled={!enabled()}
		style={surfaceStyle()}
		role="application"
		aria-label="Image gesture overlay"
		on:pointerdown={onPointerDown}
		on:pointermove={onPointerMove}
		on:pointerup={onPointerUp}
		on:pointercancel={onPointerCancel}
		on:lostpointercapture={onLostPointerCapture}
	>
		{#if rectangleStyle()}
			<div
				class="selection-rectangle"
				class:draft-selection={rectangleIsDraft()}
				class:applied-selection={!rectangleIsDraft()}
				style={rectangleStyle()}
			></div>
		{/if}
	</div>
</Block>

<style>
	.gesture-surface {
		box-sizing: border-box;
		width: 100%;
		min-width: 1px;
		min-height: 1px;
		background: transparent;
		overflow: hidden;
		touch-action: none;
		user-select: none;
		z-index: 20;
	}
	.gesture-surface.disabled {
		pointer-events: none;
	}
	.selection-rectangle {
		position: absolute;
		box-sizing: border-box;
		box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.65);
		pointer-events: none;
	}
	.selection-rectangle.draft-selection {
		border: 2px dashed #facc15;
		background: rgba(250, 204, 21, 0.14);
	}
	.selection-rectangle.applied-selection {
		border: 2px solid #22c55e;
		background: rgba(34, 197, 94, 0.1);
	}
</style>
