"""Shared Gradio theme and stylesheet for the demo."""

import gradio as gr


CUSTOM_CSS = """
.container { max-width: 1200px; margin: auto; padding-top: 10px; }
h1 { text-align: center; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2d3748; margin: 0 0 8px; }
.description { text-align: center; font-size: 1.1em; color: #4a5568; margin: 0 0 16px; }
#main_tabs { margin-top: -32px; }
#sam3_model_top_bar {
    align-items: center !important;
    gap: 20px;
    min-height: 116px;
    padding: 10px 16px 18px;
}
.sam3-model-controls,
.sam3-model-spacer {
    justify-content: center !important;
}
.sam3-model-controls-row {
    align-items: center !important;
    flex-wrap: nowrap !important;
    gap: 14px !important;
}
#sam3_model_status {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    width: auto !important;
}
#sam3_model_status .prose,
#sam3_model_status > div {
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.sam3-model-status {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 34px;
    color: #334155;
    font-size: 15px;
    font-weight: 600;
    line-height: 1.35;
}
.sam3-model-status-light {
    position: relative;
    display: inline-block;
    width: 11px;
    height: 11px;
    flex: 0 0 11px;
    border-radius: 50%;
    background: var(--sam3-status-colour);
    box-shadow:
        0 0 0 3px color-mix(in srgb, var(--sam3-status-colour) 16%, transparent),
        0 0 10px 2px color-mix(in srgb, var(--sam3-status-colour) 45%, transparent);
}
.sam3-model-status-light::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.85);
}
.sam3-model-status-text {
    min-width: 0;
    overflow-wrap: anywhere;
}
#sam3_model_start {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 88px !important;
    max-width: 104px !important;
    min-height: 36px !important;
    height: 36px !important;
    padding: 0 14px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 650 !important;
    box-shadow: 0 2px 7px rgba(37, 99, 235, 0.12) !important;
}
.sam3-model-heading h1,
.sam3-model-heading .description {
    width: 100%;
}
@media (max-width: 900px) {
    #sam3_model_top_bar {
        gap: 8px;
        padding-inline: 4px;
    }
    .sam3-model-spacer {
        display: none !important;
    }
    .sam3-model-controls,
    .sam3-model-heading {
        min-width: 100% !important;
    }
    .sam3-model-controls-row {
        justify-content: center !important;
    }
}
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
.stitch-intro {
    margin-bottom: 6px !important;
}
.stitch-main-row {
    align-items: flex-start !important;
    gap: 16px !important;
}
.stitch-control-column,
.stitch-preview-column {
    gap: 12px !important;
}
.stitch-card,
.stitch-canvas-card,
.stitch-result-card {
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    background: rgba(255, 255, 255, 0.92) !important;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05) !important;
    padding: 14px !important;
}
.stitch-card h4,
.stitch-canvas-card h4,
.stitch-result-card h4 {
    margin: 0 0 8px !important;
    color: #1e3a5f !important;
}
.stitch-step-shell {
    overflow: hidden !important;
}
.stitch-step-pager {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 12px !important;
    overflow-x: auto !important;
    overscroll-behavior-x: contain;
    scroll-behavior: smooth;
    scroll-snap-type: x mandatory;
    scrollbar-gutter: stable;
    padding-bottom: 8px;
}
.stitch-step-page {
    flex: 0 0 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    scroll-snap-align: start;
    scroll-snap-stop: always;
}
.stitch-canvas-help {
    color: #64748b !important;
    font-size: 0.9rem !important;
    margin-top: -4px !important;
}
.stitch-status-card {
    border-left: 4px solid #3b82f6 !important;
    border-radius: 10px !important;
    background: #eff6ff !important;
    color: #1e3a5f !important;
    padding: 10px 12px !important;
    min-height: 44px !important;
}
.stitch-status-card .prose,
.stitch-handoff-status .prose {
    margin: 0 !important;
}
.stitch-handoff-status {
    color: #166534 !important;
    font-weight: 600 !important;
}
#stitch_preview_canvas {
    overflow: hidden !important;
    border-radius: 10px !important;
}
@media (max-width: 900px) {
    .stitch-main-row {
        flex-direction: column !important;
    }
    .stitch-control-column,
    .stitch-preview-column {
        min-width: 100% !important;
        width: 100% !important;
    }
    #stitch_preview_canvas {
        min-height: 380px !important;
    }
}
"""


def build_theme():
    return gr.themes.Soft(primary_hue="blue", secondary_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
