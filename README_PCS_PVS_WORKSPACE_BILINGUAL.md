# SAM3 PCS/PVS Workspace (SAM3 PCS/PVS 单工作台)

<div align="center">

# SAM3 PCS/PVS Workspace

**Gradio-native SAM3 image segmentation workspace**

**基于 Gradio 原生交互的 SAM3 图像分割工作台**

</div>

---

## Project Overview 项目简介

This version provides one image workspace with two modes:

本版本提供一个图像工作台，保留两个模式：

- **PCS Auto 自动概念分割**: text prompt + positive/negative bbox exemplar.
- **PVS Manual 手动实例分割**: bbox instance creation + positive point / positive polygon refinement.



---

## Interaction 交互方式

Upload and interact on the same component:

上传和交互都在同一个组件中完成：

```text
Image / workspace (upload and click here)
```

There is no second workspace image.

不会再生成第二张 workspace 图片。

Choose **Gradio click tool**:

选择 **Gradio click tool**：

| Tool 工具 | Usage 用法 |
| --- | --- |
| Positive point | Click one point. 单击一个点。 |
| BBox two-click | Click first bbox corner, then opposite corner. 先点 bbox 第一个角，再点对角。 |
| Polygon vertex | Click polygon vertices, then click Finish polygon. 连续点击 polygon 顶点，然后点击 Finish polygon。 |

---

## PCS Auto 自动概念分割

1. Switch to **PCS Auto**.
2. Enter a text prompt if needed.
3. Use **BBox two-click** on the image.
4. Choose **Positive exemplar** or **Negative exemplar**.
5. Click **Add selected bbox to PCS exemplars**.
6. Click **Run PCS**.

PCS only uses grounding. It does not call PVS `predict_inst`.

PCS 只走 grounding，不调用 PVS `predict_inst`。

---

## PVS Manual 手动实例分割

### Bbox instance bbox 创建实例

1. Switch to **PVS Manual**.
2. Use **BBox two-click** on the image.
3. Click **Create PVS instance from bbox**.

This creates the active PVS instance and saves low-resolution logits.

这会创建 active PVS instance，并保存 low-res logits。

### Positive point 正向点提示

1. Select **Positive point**.
2. Click one point on the image.
3. Click **Create/refine with positive point**.

If an active PVS instance exists, the point refines it. Otherwise, it creates a point-based PVS instance.

如果已有 active PVS instance，point 会精修该实例；否则会创建基于点的 PVS instance。

### Positive polygon 正向多边形精修

1. Create a PVS bbox instance first.
2. Keep it selected in **Active PVS instance**.
3. Select **Polygon vertex**.
4. Click polygon vertices on the same image.
5. Click **Finish polygon / refine active PVS**.

In PVS mode, finishing polygon directly refines the active PVS instance.

在 PVS 模式下，完成 polygon 会直接精修当前 active PVS instance。

If there is no active PVS instance, the app will ask you to create or select a bbox instance first.

如果没有 active PVS instance，系统会要求先创建或选择 bbox instance。

---

## Evaluation 标注评估

Uploaded O3/LabelMe-like JSON takes priority over COCO lookup.

上传 O3/LabelMe-like JSON 时，优先使用上传 JSON，不走 COCO lookup。

Supported shapes 支持：

- `polygon`
- `rectangle`
- `linestrip`

If JSON size differs from the image size, shapes are scaled and warnings are written to metrics.

如果 JSON 尺寸与当前图片尺寸不一致，会自动缩放 shape，并在 metrics 中写入 warning。

---

## Validation 验证

```bash
cd /data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/sam3-pvs-workspace
/data/zhengqiyuan/miniforge3/envs/sam3/bin/python -m py_compile sam3_gradio_demo.py
git diff --check -- sam3_gradio_demo.py README_PCS_PVS_WORKSPACE.md README_PCS_PVS_WORKSPACE_BILINGUAL.md
```
