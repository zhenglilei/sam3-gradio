"""Shared Gradio theme and stylesheet for the demo."""

import gradio as gr


CUSTOM_CSS = """
.container { max-width: 1200px; margin: auto; padding-top: 10px; }
h1 { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2d3748; margin: 0 0 8px; }
.description { text-align: center; font-size: 1.1em; color: #4a5568; margin: 0 0 16px; }
#main_tabs { margin-top: -32px; }
.gr-button-primary { background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); border: none; }
.gr-box { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
#interaction-info { font-weight: bold; color: #2b6cb0; text-align: center; background-color: #ebf8ff; padding: 10px; border-radius: 5px; border: 1px solid #bee3f8; }
.hidden-payload { display: none !important; }
.mode-radio .wrap { display: flex; width: 100%; gap: 10px; }
.mode-radio .wrap label { flex: 1; justify-content: center; text-align: center; }
.sam3-panel textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.gesture-overlay-anchor { min-height: 0 !important; height: 0 !important; overflow: visible !important; }
.image-prepost-row { align-items: stretch !important; }
.image-prepost-column { height: 100%; }
.polygon-finish-btn button {
    width: 100%;
    min-height: 42px;
    font-weight: 700;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
}
.layout-preview-pager {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 12px;
    overflow-x: auto !important;
    overscroll-behavior-x: contain;
    scroll-behavior: smooth;
    scroll-snap-type: x mandatory;
    scrollbar-gutter: stable;
    padding-bottom: 8px;
}
.layout-preview-page {
    flex: 0 0 100% !important;
    min-width: 100% !important;
    scroll-snap-align: start;
    scroll-snap-stop: always;
}
"""


def build_theme():
    return gr.themes.Soft(primary_hue="blue", secondary_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
