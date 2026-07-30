
# `gradio_image_gesture_overlay`
<img alt=" Static Badge src=https://img.shields.io/badge/version%20-%200.0.1%20-%20orange>

Transparent click and drag gesture overlay for Gradio images

## Installation

```bash
pip install gradio_image_gesture_overlay
```

## Usage

```python
import gradio as gr
from gradio_image_gesture_overlay import ImageGestureOverlay


example = ImageGestureOverlay().example_value()

demo = gr.Interface(
    lambda value: value,
    ImageGestureOverlay(value=example, interactive=True),
    gr.JSON(),
)


if __name__ == "__main__":
    demo.launch()

```

## `ImageGestureOverlay`

### Initialization

<table>
<thead>
<tr>
<th align="left">name</th>
<th align="left" style="width: 25%;">type</th>
<th align="left">default</th>
<th align="left">description</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><code>value</code></td>
<td align="left" style="width: 25%;">

```python
dict | str | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>label</code></td>
<td align="left" style="width: 25%;">

```python
str | I18nData | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>every</code></td>
<td align="left" style="width: 25%;">

```python
Timer | float | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>inputs</code></td>
<td align="left" style="width: 25%;">

```python
Component | Sequence[Component] | set[Component] | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>show_label</code></td>
<td align="left" style="width: 25%;">

```python
bool | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>container</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>value = False</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>scale</code></td>
<td align="left" style="width: 25%;">

```python
int | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>min_width</code></td>
<td align="left" style="width: 25%;">

```python
int
```

</td>
<td align="left"><code>value = 160</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>interactive</code></td>
<td align="left" style="width: 25%;">

```python
bool | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>visible</code></td>
<td align="left" style="width: 25%;">

```python
bool | Literal['hidden']
```

</td>
<td align="left"><code>value = True</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>elem_id</code></td>
<td align="left" style="width: 25%;">

```python
str | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>elem_classes</code></td>
<td align="left" style="width: 25%;">

```python
list[str] | str | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>render</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>value = True</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>key</code></td>
<td align="left" style="width: 25%;">

```python
int | str | tuple[int | str, ...] | None
```

</td>
<td align="left"><code>value = None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>preserved_by_key</code></td>
<td align="left" style="width: 25%;">

```python
list[str] | str | None
```

</td>
<td align="left"><code>value = "value"</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>height</code></td>
<td align="left" style="width: 25%;">

```python
int | str
```

</td>
<td align="left"><code>value = 320</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>target_elem_id</code></td>
<td align="left" style="width: 25%;">

```python
str
```

</td>
<td align="left"><code>value = ""</code></td>
<td align="left">None</td>
</tr>
</tbody></table>


### Events

| name | description |
|:-----|:------------|
| `input` | This listener is triggered when the user changes the value of the ImageGestureOverlay. |



### User function

The impact on the users predict function varies depending on whether the component is used as an input or output for an event (or both).

- When used as an Input, the component only impacts the input signature of the user function.
- When used as an output, the component only impacts the return signature of the user function.

The code snippet below is accurate in cases where the component is used as both an input and an output.

- **As input:** Should return, the output data received by the component from the user's function in the backend.

 ```python
 def predict(
     value: dict[str,typing.Any]
 ) -> dict| list| str| None:
     return value
 ```
