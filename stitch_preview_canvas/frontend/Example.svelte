<script lang="ts">
	import type { StitchPreviewValue } from "./types";

	export let value: StitchPreviewValue | null;
	export let type: "gallery" | "table";
	export let selected = false;
</script>

{#if value && value.tiles && value.tiles.length > 0}
	<div
		class="container"
		class:table={type === "table"}
		class:gallery={type === "gallery"}
		class:selected
	>
		<span class="label">tile {value.selected ?? 0}</span>
	</div>
{/if}

<style>
	.container {
		display: flex;
		align-items: center;
		justify-content: center;
		border: 2px solid var(--border-color-primary);
		border-radius: var(--radius-lg);
		background: #0f172a;
		color: #e2e8f0;
		font-size: 12px;
		font-weight: 600;
	}

	.container.selected {
		border-color: var(--border-color-accent);
	}

	.container.table {
		width: var(--size-20);
		height: var(--size-20);
	}

	.container.gallery {
		height: var(--size-20);
		max-height: var(--size-20);
		min-width: var(--size-20);
	}
</style>
