
import gradio as gr
from gradio_stitch_preview_canvas import StitchPreviewCanvas


demo = gr.Interface(
    lambda x: x,
    StitchPreviewCanvas(),
    StitchPreviewCanvas(),
)


if __name__ == "__main__":
    demo.launch()
