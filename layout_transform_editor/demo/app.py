
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
