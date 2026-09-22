"""Shared Gradio theme and stylesheet for the demo."""

import gradio as gr


CUSTOM_CSS = """
.gradio-container > .main { margin: auto; padding-top: 10px; }
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
.gradio-container > .main:has(.stitch-main-row) {
    max-width: 1440px !important;
    padding-inline: 16px !important;
}
.stitch-main-row .form {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}
.stitch-intro {
    margin-bottom: 6px !important;
}
.stitch-main-row {
    display: grid !important;
    grid-template-columns: 288px minmax(0, 1fr);
    align-items: flex-start !important;
    gap: 14px !important;
}
.stitch-main-row > .stitch-control-column,
.stitch-main-row > .stitch-preview-column {
    min-width: 0 !important;
    width: 100% !important;
}
.stitch-control-column .block,
.stitch-control-column .form {
    min-width: 0 !important;
}
.stitch-control-column label,
.stitch-control-column button,
.stitch-control-column input,
.stitch-control-column .prose {
    font-size: 13px !important;
}
.stitch-control-column,
.stitch-preview-column {
    gap: 12px !important;
}
.stitch-card,
.stitch-canvas-card,
.stitch-result-card {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #fff !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    padding: 10px !important;
}
.stitch-card h4,
.stitch-canvas-card h4,
.stitch-result-card h4 {
    margin: 0 0 8px !important;
    color: #1e3a5f !important;
    font-size: 13px !important;
    background: transparent !important;
}
.stitch-canvas-help {
    color: #64748b !important;
    font-size: 0.9rem !important;
    margin-top: -4px !important;
}
.stitch-status-card {
    max-height: 150px;
    overflow-y: auto;
    border-left: 4px solid #3b82f6 !important;
    border-radius: 8px !important;
    background: #eff6ff !important;
    color: #1e3a5f !important;
    padding: 8px 10px !important;
    min-height: 40px !important;
    font-size: 12px !important;
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
@media (max-width: 640px) {
    .stitch-main-row {
        grid-template-columns: minmax(0, 1fr);
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
/* 周期拼接侧栏：外壳 */
.stitch-sidebar {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #fff !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    padding: 10px !important;
    gap: 0 !important;
}
.stitch-control-column .container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
}
.stitch-control-column .block {
    width: 100% !important;
    box-sizing: border-box;
}
.stitch-control-column .form {
    border-radius: 0 !important;
    padding: 0 !important;
    gap: 10px !important;
}

/* 周期拼接侧栏：图片 / 对齐 / 导出 分段标签栏 */
/* Gradio 6.x (verified on 6.14): internal tab/form/upload wrappers.
   Recheck these selectors with browser QA whenever Gradio is upgraded. */
.stitch-control-tabs,
.stitch-control-tabs .tabitem {
    min-width: 0 !important;
    width: 100% !important;
    padding: 0 !important;
    border: 0 !important;
}
.stitch-control-tabs > .tab-wrapper {
    border: 0 !important;
    margin-bottom: 12px !important;
}
.stitch-control-tabs .tab-container[role="tablist"] {
    display: flex !important;
    gap: 2px;
    padding: 3px;
    border: 0 !important;
    border-radius: 8px;
    background: #f1f5f9;
}
.stitch-control-tabs .tab-container[role="tablist"] button {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    padding: 6px 4px !important;
    border: 0 !important;
    border-radius: 6px !important;
    background: transparent !important;
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    white-space: nowrap;
}
.stitch-control-tabs .tab-container[role="tablist"] button:hover {
    color: #334155 !important;
}
/* 分段控件不需要 Gradio 默认的选中下划线 */
.stitch-control-tabs .tab-container[role="tablist"] button::after {
    display: none !important;
}
.stitch-control-tabs .tab-container[role="tablist"] button[aria-selected="true"] {
    background: #fff !important;
    color: #1d4ed8 !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12) !important;
}
.stitch-control-tabs .tabitem > .column {
    gap: 10px !important;
}

/* 周期拼接侧栏：分组小标题与字段标签 */
.stitch-group-label {
    margin: 0 !important;
    overflow: visible !important;
}
.stitch-group-label p {
    margin: 0 !important;
    color: #94a3b8 !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
}
.stitch-control-column label span,
.stitch-control-column [data-testid="block-info"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    color: #475569 !important;
    font-size: 12px !important;
}

/* 周期拼接侧栏：分块图片上传列表 */
.stitch-upload {
    border: 1px dashed #cbd5e1 !important;
    border-radius: 8px !important;
    background: #f8fafc !important;
    overflow: auto !important;
}
/* Gradio 给空态拖放区设了 min-height: 240px，会撑出外层滚动条 */
.stitch-upload .wrap:not([data-testid="status-tracker"]) {
    min-height: 0 !important;
    padding-top: 0 !important;
}
.stitch-upload table {
    table-layout: fixed !important;
    width: 100% !important;
}
.stitch-upload td:first-child {
    width: auto !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.stitch-upload td:nth-child(2) {
    display: none;
}
.stitch-upload td:last-child {
    width: 26px !important;
}

/* 周期拼接侧栏：边缘裁剪 4 列，位置与旋转 2 列 */
.stitch-crop-fields,
.stitch-position-fields {
    display: block !important;
}
.stitch-crop-fields > .form {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
    width: 100% !important;
    gap: 6px !important;
}
.stitch-position-fields > .form {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    width: 100% !important;
    gap: 8px !important;
}
.stitch-position-fields > .form > :last-child {
    grid-column: 1 / -1;
}
.stitch-crop-fields .block,
.stitch-position-fields .block,
.stitch-crop-fields label.block,
.stitch-position-fields label.block {
    display: flex !important;
    flex-direction: column;
    gap: 3px;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.stitch-crop-fields input,
.stitch-position-fields input {
    width: 100% !important;
    min-width: 0 !important;
    height: 30px !important;
    padding: 0 6px !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    background: #fff !important;
    box-shadow: none !important;
    box-sizing: border-box;
    text-align: center;
    -moz-appearance: textfield;
}
.stitch-crop-fields input:focus,
.stitch-position-fields input:focus {
    border-color: #3b82f6 !important;
    outline: none !important;
}
.stitch-crop-fields input::-webkit-inner-spin-button,
.stitch-crop-fields input::-webkit-outer-spin-button,
.stitch-position-fields input::-webkit-inner-spin-button,
.stitch-position-fields input::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
}
.stitch-crop-fields [data-testid="block-info"] {
    text-align: center;
    font-size: 11px !important;
}

/* 周期拼接侧栏：步长分段控件、开关与主操作按钮 */
.stitch-segmented .wrap:not([data-testid="status-tracker"]) {
    display: flex !important;
    width: 100%;
    gap: 6px;
}
.stitch-segmented .wrap:not([data-testid="status-tracker"]) > label {
    flex: 1 1 0;
    min-width: 0;
    justify-content: center;
    text-align: center;
    white-space: nowrap;
    padding-inline: 6px !important;
}
.stitch-segmented .wrap:not([data-testid="status-tracker"]) > label span {
    white-space: nowrap;
}
.stitch-toggle-row {
    gap: 8px !important;
}
.stitch-action-btn,
.stitch-action-btn button {
    width: 100% !important;
    min-height: 36px !important;
    border-radius: 8px !important;
    font-weight: 650 !important;
}
.stitch-action-btn-quiet,
.stitch-action-btn-quiet button {
    min-height: 32px !important;
    font-weight: 600 !important;
}
"""


# Gradio 6.x wrappers: recheck tab, form and upload selectors on upgrades.
CUSTOM_CSS += """
.gradio-container > .main:has(#el-workspace) {
    max-width: 1800px !important;
    padding: 8px 20px !important;
}
.gradio-container :has(> #sam3_model_top_bar) { gap: 6px !important; }
#sam3_model_top_bar {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center !important;
    min-height: 44px;
    padding: 0;
    gap: 12px;
}
.sam3-model-controls { grid-column: 2; grid-row: 1; min-width: 0 !important; }
.sam3-model-heading { grid-column: 1; grid-row: 1; min-width: 0 !important; }
.sam3-model-spacer, .sam3-model-heading .description { display: none !important; }
.sam3-model-heading h1 { font-size: 20px; line-height: 1.3; text-align: left; margin: 0; }
.sam3-model-controls-row { gap: 10px !important; }
.sam3-model-status { font-size: 13px; min-height: 32px; }
#main_tabs { margin-top: 0; }
#main_tabs > .tab-wrapper { margin-bottom: 6px; }
#main_tabs > .tab-wrapper button { min-height: 34px; padding-block: 6px; }
#el-workspace, #repair-workspace { gap: 8px; padding: 6px 0 16px; letter-spacing: 0; }
#el-workspace .row, #el-workspace .column, #el-workspace .form,
#repair-workspace .row, #repair-workspace .column, #repair-workspace .form {
    min-width: 0 !important;
}
#el-workspace .gr-group, #el-workspace .form,
#repair-workspace .gr-group, #repair-workspace .form,
#el-workspace .styler, #repair-workspace .styler {
    background: transparent !important;
    box-shadow: none !important;
    border: 0 !important;
}
#el-workspace h3, #el-workspace h4, #repair-workspace h4 {
    font-size: 14px !important; margin: 0 0 6px !important;
    background: transparent !important;
}
#el-workspace .prose, #repair-workspace .prose { background: transparent; font-size: 13px; }
#el-workspace button, #repair-workspace button,
#el-workspace label, #repair-workspace label,
#el-workspace textarea, #repair-workspace textarea {
    font-size: 13px; letter-spacing: 0;
}
#el-workspace label > span, #repair-workspace label > span { background: transparent !important; }
#el-workspace button, #repair-workspace button { min-width: 0 !important; min-height: 32px; }
#el-workspace button:disabled, #repair-workspace button:disabled { cursor: not-allowed; }
#el-image-list, #repair-library {
    border: 0; box-shadow: none; border-radius: 0; padding: 0; gap: 6px;
}
#el-library-row, .repair-upload-row { gap: 10px; align-items: start; }
.el-compact-upload { min-height: 76px !important; }
.el-compact-upload .wrap:not([data-testid="status-tracker"]) { min-height: 0 !important; }
.el-compact-upload .upload-container { min-height: 0 !important; padding: 4px !important; }
.el-compact-upload .upload-container svg { width: 18px; height: 18px; }
.el-compact-upload > [data-testid="block-label"] { display: none; }
.el-compact-upload [aria-dropeffect] > .wrap { gap: 2px; padding: 6px; font-size: 12px; }
.el-compact-upload .icon-wrap { width: 18px; height: 18px; }
.el-compact-upload .or { display: none; }
#el-image-gallery, #repair-gallery { height: 76px !important; min-height: 0 !important; border: 0; box-shadow: none; }
#el-image-gallery .grid-wrap, #repair-gallery .grid-wrap { padding: 4px !important; overflow: hidden !important; }
#el-image-gallery .grid-container, #repair-gallery .grid-container {
    display: grid !important;
    grid-template-columns: none !important;
    grid-template-rows: 52px !important;
    height: 68px;
    grid-auto-flow: column;
    grid-auto-columns: 112px;
    gap: 6px;
    overflow-x: auto;
    overflow-y: hidden;
}
#el-image-gallery .gallery-item, #repair-gallery .gallery-item {
    height: 52px !important; min-width: 0; aspect-ratio: auto !important;
    border-radius: 4px;
}
#el-image-gallery button, #repair-gallery button { border-radius: 4px; }
#el-image-gallery .caption-label, #repair-gallery .caption-label { font-size: 11px; }
#el-batch-selection .wrap { display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 6px; }
#el-batch-selection label { min-width: 0; padding: 6px 8px; }
.el-selection-actions { justify-content: flex-start; }
.el-selection-actions button { flex: 0 0 auto !important; padding-inline: 12px; white-space: nowrap; }
#el-batch-status, #repair-position, #repair-status {
    padding: 5px 8px; background: #eef6f1; border-left: 3px solid #34865b;
}
#el-batch-status:not(:has(p,ul,ol)) { display: none; }
#el-batch-status p, #repair-position p, #repair-status p { margin: 0; overflow-wrap: anywhere; }
#el-workspace-footer, #repair-actions {
    position: sticky; top: 0; z-index: 20;
    margin: 0; padding: 6px 0; gap: 6px;
    background: var(--body-background-fill, #f8f9fa);
    border-block: 1px solid #dfe3e8; flex-wrap: wrap;
}
#el-workspace-footer button { flex: 1 1 84px; min-height: 32px; box-shadow: none; white-space: nowrap; }
#repair-actions > button { flex: 1 1 150px; min-height: 34px; }
#el-workflow-tabs > .tab-container[role="tablist"] {
    display: flex; flex-wrap: wrap; gap: 4px; overflow: visible;
}
#el-workflow-tabs > .tab-container[role="tablist"] button { min-height: 32px; font-size: 14px; padding-block: 5px; }
.el-workflow-page { padding: 6px 0 !important; border: 0 !important; }
.el-canvas-toolbar { align-items: center; gap: 10px !important; }
#el-canvas-view, #repair-canvas-view { flex: 0 0 auto; min-width: 230px !important; }
#el-canvas-view .wrap, #repair-canvas-view .wrap {
    display: flex; flex-wrap: nowrap; gap: 2px; padding: 2px;
    background: #eef1f4; border-radius: 6px;
}
#el-canvas-view label, #repair-canvas-view label {
    flex: 1; justify-content: center; min-width: 0; white-space: nowrap;
    padding: 5px 8px; font-size: 12px;
}
#el-canvas-view input, #repair-canvas-view input {
    position: absolute; width: 1px; height: 1px; opacity: 0;
}
#el-canvas-view label:focus-within, #repair-canvas-view label:focus-within { outline: 2px solid #2563eb; }
#el-workspace-body, #el-template-workspace {
    display: grid !important; grid-template-columns: minmax(0,1fr) 280px;
    gap: 16px; align-items: start; height: auto; overflow: visible;
}
#el-workspace-center, #el-workspace-tools { min-height: 0; max-height: none; overflow: visible; }
#el-workspace-center { position: sticky; top: 52px; align-self: start; }
#el-workspace-tools { gap: 10px; padding-left: 12px; border-left: 1px solid #dfe3e8; }
#el-workspace-tools > * { flex-shrink: 0; }
#el-workspace-tools .row { flex-wrap: wrap; gap: 6px; }
#el-workspace-tools button { white-space: normal; }
#el-workspace-tools .mode-radio label { padding: 6px 5px; min-width: 0; }
#el-workspace .mode-radio .wrap { flex-wrap: wrap; gap: 4px; }
#el-workspace .mode-radio .wrap label { flex: 1 1 92px; white-space: normal; padding-block: 6px; }
#el-workspace-tools #interaction-info { padding: 5px 8px; text-align: left; font-weight: 400; }
#el-pcs-settings { order: -1; }
.el-image-pair { display: grid !important; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 10px; }
#input_image, .el-result-image { height: clamp(320px, 52vh, 520px) !important; min-height: 0 !important; }
#source_input_image, #template_match_preview { height: clamp(320px, 55vh, 560px) !important; min-height: 0 !important; }
#el-workspace img { object-fit: contain; max-width: 100%; }
#el-workspace textarea { min-width: 0; }
#el-workspace:has(#el-canvas-view input[value="source"]:checked) .el-image-pair,
#el-workspace:has(#el-canvas-view input[value="result"]:checked) .el-image-pair {
    grid-template-columns: minmax(0,1fr);
}
#el-workspace:has(#el-canvas-view input[value="source"]:checked) #el-result-pane,
#el-workspace:has(#el-canvas-view input[value="result"]:checked) #el-source-pane { display: none !important; }
.repair-nav-row { gap: 6px; align-items: center; }
.repair-nav-row > button { flex: 0 0 auto !important; padding-inline: 12px; white-space: nowrap; }
#repair-position { flex: 1 1 220px; }
#repair-main-row {
    display: grid !important; grid-template-columns: minmax(0,1fr) 280px;
    gap: 16px; align-items: start; min-width: 0;
}
#repair-main-row > * { min-width: 0 !important; }
#repair-canvas-column { position: sticky; top: 56px; align-self: start; }
#repair-canvas-pair { display: grid !important; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 10px; }
.repair-canvas-pane { min-width: 0; gap: 6px; }
#repair-mask-editor, #repair-result {
    width: 100%; height: clamp(280px, 48vh, 480px) !important;
    min-height: 280px !important; background: #f1f3f5;
    border: 1px solid #cad1da; border-radius: 4px; overflow: hidden;
}
#repair-result img { width: 100%; height: 100%; object-fit: contain; }
#repair-tools { gap: 10px; padding-left: 12px; border-left: 1px solid #dfe3e8; }
.repair-tool-section {
    border: 0 !important; border-bottom: 1px solid #dfe3e8 !important;
    border-radius: 0 !important; box-shadow: none !important; padding: 0 0 8px !important;
}
.repair-tool-section:last-child { border-bottom: 0 !important; }
.repair-tool-section button { white-space: normal; }
.repair-tool-radio .wrap { display: grid !important; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 4px; }
.repair-tool-radio label { min-width: 0 !important; padding: 6px !important; white-space: normal; }
.repair-inline-warning { color: #a12622; border-left: 3px solid #c43c35; padding: 8px 10px; }
#repair-workspace:has(#repair-canvas-view input[value="source"]:checked) #repair-canvas-pair,
#repair-workspace:has(#repair-canvas-view input[value="result"]:checked) #repair-canvas-pair { grid-template-columns: minmax(0,1fr); }
#repair-workspace:has(#repair-canvas-view input[value="source"]:checked) #repair-result-pane,
#repair-workspace:has(#repair-canvas-view input[value="result"]:checked) #repair-source-pane { display: none !important; }
.stitch-sidebar, #stitch_preview_canvas { border-radius: 6px !important; }
.stitch-canvas-card, .stitch-result-card {
    padding: 8px 0 !important; border: 0 !important;
    border-radius: 0 !important; box-shadow: none !important; background: transparent !important;
}
#stitch-preview-tabs { min-width: 0; }
#stitch_mosaic_preview { height: clamp(300px, 55vh, 560px) !important; }
@media (max-width: 1200px) {
    #el-workspace:has(#el-canvas-view input[value="auto"]:checked) .el-image-pair,
    #repair-workspace:has(#repair-canvas-view input[value="auto"]:checked) #repair-canvas-pair { grid-template-columns: minmax(0,1fr); }
    #el-workspace:has(#el-canvas-view input[value="auto"]:checked) #el-result-pane,
    #repair-workspace:has(#repair-canvas-view input[value="auto"]:checked) #repair-result-pane { display: none !important; }
}
@media (max-width: 800px) {
    .gradio-container > .main:has(#el-workspace) { padding-inline: 12px !important; }
    #sam3_model_top_bar { gap: 6px; }
    .sam3-model-heading h1 { font-size: 17px; }
    .sam3-model-status { font-size: 12px; }
    #el-workspace-body, #el-template-workspace, #repair-main-row { grid-template-columns: minmax(0,1fr); }
    #el-workspace-tools, #repair-tools { padding-left: 0; border-left: 0; border-top: 1px solid #dfe3e8; padding-top: 10px; }
    #el-workspace-center, #repair-canvas-column { position: static; }
    #el-library-row > *, .repair-upload-row > * { min-width: 0 !important; }
    #el-workspace-footer button { flex-basis: 90px; }
    #repair-actions > button { flex-basis: 140px; }
    #el-canvas-view, #repair-canvas-view { min-width: 210px !important; }
}
@media (max-width: 500px) {
    #sam3_model_top_bar { grid-template-columns: minmax(0,1fr); }
    .sam3-model-controls { grid-column: 1; grid-row: 2; }
    .sam3-model-controls-row { justify-content: flex-start !important; }
    #el-library-row, .repair-upload-row { display: grid !important; grid-template-columns: 120px minmax(0,1fr); }
    #el-workspace-footer, #repair-actions { position: static; }
    #el-workflow-tabs > .tab-container[role="tablist"] button { flex: 1 1 40%; }
}
"""
def build_theme():
    return gr.themes.Soft(primary_hue="blue", secondary_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
