<script lang="ts">
	import PvsEditor from "./PvsEditor.svelte";
	import OverlapEditor from "./OverlapEditor.svelte";

	const componentProps = $props();
</script>

{#if componentProps.props?.editor_mode === "overlap"}
	<OverlapEditor {...componentProps} />
{:else}
	<PvsEditor {...componentProps} />
{/if}
