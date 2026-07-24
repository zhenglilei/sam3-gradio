from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from gradio.components.base import Component
from gradio.i18n import I18nData


class LayoutTransformEditor(Component):
    """Canvas editor whose value is a JSON payload with image URLs and transform metadata."""

    EVENTS = [Events.change]
    data_model = None

    def __init__(
        self,
        value: dict | str | None = None,
        *,
        label: str | I18nData | None = None,
        every: Timer | float | None = None,
        inputs: Component | Sequence[Component] | set[Component] | None = None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        interactive: bool | None = None,
        visible: bool | Literal["hidden"] = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | tuple[int | str, ...] | None = None,
        preserved_by_key: list[str] | str | None = "value",
        height: int | str = 520,
    ):
        self.height = height
        super().__init__(
            label=label,
            every=every,
            inputs=inputs,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            key=key,
            preserved_by_key=preserved_by_key,
            value=value,
        )

    @staticmethod
    def _unwrap(payload: Any) -> Any:
        return payload.root if isinstance(payload, JsonData) else payload

    @staticmethod
    def _sanitize_transform(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        return {
            key: value[key]
            for key in _TRANSFORM_INTENT_FIELDS
            if key in value
        }

    def preprocess(self, payload: Any) -> dict[str, Any]:
        raw = self._unwrap(payload)
        if not isinstance(raw, dict):
            return {}

        result: dict[str, Any] = {}
        for field in ("enabled", "target_width", "target_height"):
            if field in raw:
                result[field] = raw[field]
        transform = self._sanitize_transform(raw.get("transform"))
        if transform is not None:
            result["transform"] = transform

        if raw.get("transform_mode") != "label_groups":
            return result
        result["transform_mode"] = "label_groups"
        intent = raw.get("group_intent")
        if not isinstance(intent, dict):
            return result
        transforms = []
        for item in intent.get("transforms") or []:
            if not isinstance(item, dict) or "group_id" not in item:
                continue
            sanitized = self._sanitize_transform(item.get("transform"))
            if sanitized is None:
                continue
            transforms.append(
                {
                    "group_id": item["group_id"],
                    "transform": sanitized,
                }
            )
        sanitized_intent = {
            "selection_signature": intent.get("selection_signature"),
            "transform_set_revision": intent.get("transform_set_revision"),
            "active_group_id": intent.get("active_group_id"),
            "transforms": transforms,
        }
        changed_group_ids = intent.get("changed_group_ids")
        if isinstance(changed_group_ids, list):
            sanitized_intent["changed_group_ids"] = [
                value
                for value in changed_group_ids
                if isinstance(value, (str, int))
                and not isinstance(value, bool)
                and value != ""
            ]
        result["group_intent"] = sanitized_intent
        return result

    def postprocess(self, value: dict | list | str | None) -> JsonData | None:
        if value is None:
            return None
        if isinstance(value, str):
            return JsonData(root=json.loads(value))
        if isinstance(value, (dict, list)):
            return JsonData(root=value)
        return JsonData(root={"value": value})

    def api_info(self) -> dict[str, Any]:
        return {"type": {}, "description": "layout transform editor json payload"}

    def example_payload(self) -> Any:
        return {"enabled": False, "status": "no layout mask loaded"}

    def example_value(self) -> Any:
        return {"enabled": False, "status": "no layout mask loaded"}
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component

    
    def change(self,
        fn: Callable[..., Any] | None = None,
        inputs: Block | Sequence[Block] | set[Block] | None = None,
        outputs: Block | Sequence[Block] | None = None,
        api_name: str | None = None,
        scroll_to_output: bool = False,
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        show_progress_on: Component | Sequence[Component] | None = None,
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = None,
        every: Timer | float | None = None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
        js: str | Literal[True] | None = None,
        concurrency_limit: int | None | Literal["default"] = "default",
        concurrency_id: str | None = None,
        api_visibility: Literal["public", "private", "undocumented"] = "public",
        key: int | str | tuple[int | str, ...] | None = None,
        api_description: str | None | Literal[False] = None,
        validator: Callable[..., Any] | None = None,
    
        ) -> Dependency:
        """
        Parameters:
            fn: the function to call when this event is triggered. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: list of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
            outputs: list of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
            api_name: defines how the endpoint appears in the API docs. Can be a string or None. If set to a string, the endpoint will be exposed in the API docs with the given name. If None (default), the name of the function will be used as the API endpoint.
            scroll_to_output: if True, will scroll to output component on completion
            show_progress: how to show the progress animation while event is running: "full" shows a spinner which covers the output component area as well as a runtime display in the upper right corner, "minimal" only shows the runtime display, "hidden" shows no progress animation at all
            show_progress_on: Component or list of components to show the progress animation on. If None, will show the progress animation on all of the output components.
            queue: if True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
            batch: if True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
            preprocess: if False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
            postprocess: if False, will not run postprocessing of component data before returning 'fn' output to the browser.
            cancels: a list of other events to cancel when this listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
            every: continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.
            trigger_mode: if "once" (default for all events except `.change()`) would not allow any submissions while an event is pending. If set to "multiple", unlimited submissions are allowed while pending, and "always_last" (default for `.change()` and `.key_up()` events) would allow a second submission after the pending event is complete.
            js: optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
            concurrency_limit: if set, this is the maximum number of this event that can be running simultaneously. Can be set to None to mean no concurrency_limit (any number of this event can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `Blocks.queue()`, which itself is 1 by default).
            concurrency_id: if set, this is the id of the concurrency group. Events with the same concurrency_id will be limited by the lowest set concurrency_limit.
            api_visibility: controls the visibility and accessibility of this endpoint. Can be "public" (shown in API docs and callable by clients), "private" (hidden from API docs and not callable by the Gradio client libraries), or "undocumented" (hidden from API docs but callable by clients and via gr.load). If fn is None, api_visibility will automatically be set to "private".
            key: A unique key for this event listener to be used in @gr.render(). If set, this value identifies an event as identical across re-renders when the key is identical.
            api_description: Description of the API endpoint. Can be a string, None, or False. If set to a string, the endpoint will be exposed in the API docs with the given description. If None, the function's docstring will be used as the API endpoint description. If False, then no description will be displayed in the API docs.
            validator: Optional validation function to run before the main function. If provided, this function will be executed first with queue=False, and only if it completes successfully will the main function be called. The validator receives the same inputs as the main function.
        
        """
        ...
