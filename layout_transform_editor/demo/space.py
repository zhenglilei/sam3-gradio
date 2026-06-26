
import gradio as gr
from app import demo as app
import os

_docs = {'LayoutTransformEditor': {'description': 'Canvas editor whose value is a JSON payload with image URLs and transform metadata.', 'members': {'__init__': {'value': {'type': 'dict | str | None', 'default': 'value = None', 'description': None}, 'label': {'type': 'str | I18nData | None', 'default': 'value = None', 'description': None}, 'every': {'type': 'Timer | float | None', 'default': 'value = None', 'description': None}, 'inputs': {'type': 'Component | Sequence[Component] | set[Component] | None', 'default': 'value = None', 'description': None}, 'show_label': {'type': 'bool | None', 'default': 'value = None', 'description': None}, 'container': {'type': 'bool', 'default': 'value = True', 'description': None}, 'scale': {'type': 'int | None', 'default': 'value = None', 'description': None}, 'min_width': {'type': 'int', 'default': 'value = 160', 'description': None}, 'interactive': {'type': 'bool | None', 'default': 'value = None', 'description': None}, 'visible': {'type': "bool | Literal['hidden']", 'default': 'value = True', 'description': None}, 'elem_id': {'type': 'str | None', 'default': 'value = None', 'description': None}, 'elem_classes': {'type': 'list[str] | str | None', 'default': 'value = None', 'description': None}, 'render': {'type': 'bool', 'default': 'value = True', 'description': None}, 'key': {'type': 'int | str | tuple[int | str, ...] | None', 'default': 'value = None', 'description': None}, 'preserved_by_key': {'type': 'list[str] | str | None', 'default': 'value = "value"', 'description': None}, 'height': {'type': 'int | str', 'default': 'value = 520', 'description': None}}, 'postprocess': {'value': {'type': 'dict| list| str| None', 'description': "The output data received by the component from the user's function in the backend."}}, 'preprocess': {'return': {'type': 'dict| list| None', 'description': "The preprocessed input data sent to the user's function in the backend."}, 'value': None}}, 'events': {'change': {'type': None, 'default': None, 'description': 'Triggered when the value of the LayoutTransformEditor changes either because of user input (e.g. a user types in a textbox) OR because of a function update (e.g. an image receives a value from the output of an event trigger). See `.input()` for a listener that is only triggered by user input.'}}}, '__meta__': {'additional_interfaces': {}, 'user_fn_refs': {'LayoutTransformEditor': []}}}

abs_path = os.path.join(os.path.dirname(__file__), "css.css")

with gr.Blocks(
    css=abs_path,
    theme=gr.themes.Default(
        font_mono=[
            gr.themes.GoogleFont("Inconsolata"),
            "monospace",
        ],
    ),
) as demo:
    gr.Markdown(
"""
# `gradio_layout_transform_editor`

<div style="display: flex; gap: 7px;">
<img alt="Static Badge" src="https://img.shields.io/badge/version%20-%200.0.1%20-%20orange">
</div>

Gradio canvas editor for SAM3 layout-mask transform prompts
""", elem_classes=["md-custom"], header_links=True)
    app.render()
    gr.Markdown(
"""
## Installation

```bash
pip install gradio_layout_transform_editor
```

## Usage

```python

import gradio as gr
from gradio_layout_transform_editor import LayoutTransformEditor


example = LayoutTransformEditor().example_value()

demo = gr.Interface(
    lambda x:x,
    LayoutTransformEditor(),  # interactive version of your component
    LayoutTransformEditor(),  # static version of your component
    # examples=[[example]],  # uncomment this line to view the "example version" of your component
)


if __name__ == "__main__":
    demo.launch()

```
""", elem_classes=["md-custom"], header_links=True)


    gr.Markdown("""
## `LayoutTransformEditor`

### Initialization
""", elem_classes=["md-custom"], header_links=True)

    gr.ParamViewer(value=_docs["LayoutTransformEditor"]["members"]["__init__"], linkify=[])


    gr.Markdown("### Events")
    gr.ParamViewer(value=_docs["LayoutTransformEditor"]["events"], linkify=['Event'])




    gr.Markdown("""

### User function

The impact on the users predict function varies depending on whether the component is used as an input or output for an event (or both).

- When used as an Input, the component only impacts the input signature of the user function.
- When used as an output, the component only impacts the return signature of the user function.

The code snippet below is accurate in cases where the component is used as both an input and an output.

- **As input:** Is passed, the preprocessed input data sent to the user's function in the backend.
- **As output:** Should return, the output data received by the component from the user's function in the backend.

 ```python
def predict(
    value: dict| list| None
) -> dict| list| str| None:
    return value
```
""", elem_classes=["md-custom", "LayoutTransformEditor-user-fn"], header_links=True)




    demo.load(None, js=r"""function() {
    const refs = {};
    const user_fn_refs = {
          LayoutTransformEditor: [], };
    requestAnimationFrame(() => {

        Object.entries(user_fn_refs).forEach(([key, refs]) => {
            if (refs.length > 0) {
                const el = document.querySelector(`.${key}-user-fn`);
                if (!el) return;
                refs.forEach(ref => {
                    el.innerHTML = el.innerHTML.replace(
                        new RegExp("\\b"+ref+"\\b", "g"),
                        `<a href="#h-${ref.toLowerCase()}">${ref}</a>`
                    );
                })
            }
        })

        Object.entries(refs).forEach(([key, refs]) => {
            if (refs.length > 0) {
                const el = document.querySelector(`.${key}`);
                if (!el) return;
                refs.forEach(ref => {
                    el.innerHTML = el.innerHTML.replace(
                        new RegExp("\\b"+ref+"\\b", "g"),
                        `<a href="#h-${ref.toLowerCase()}">${ref}</a>`
                    );
                })
            }
        })
    })
}

""")

demo.launch()
