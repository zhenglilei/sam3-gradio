# SAM3 PCS/PVS Workspace

This branch provides a Gradio-native single-image workspace for SAM3 image segmentation.

## Modes

- **PCS Auto**: concept segmentation with text prompt and positive/negative bbox exemplars.
- **PVS Manual**: manual instance segmentation with bbox, positive point, and positive polygon refinement.


## How To Use

1. Upload an image in **Image / workspace (upload and click here)**.
2. The uploaded image becomes the workspace directly. There is no second workspace image.
3. Choose a **Gradio click tool**:
   - **Positive point**: click one point on the image.
   - **BBox two-click**: click the first bbox corner, then click the opposite corner.
   - **Polygon vertex**: click polygon vertices, then click **Finish polygon / refine active PVS**.

## PCS Auto

1. Switch mode to **PCS Auto**.
2. Optionally enter a text prompt.
3. Select **BBox two-click** and click two corners on the image.
4. Choose **Positive exemplar** or **Negative exemplar**.
5. Click **Add selected bbox to PCS exemplars**.
6. Click **Run PCS**.

PCS uses grounding only. It does not use PVS `predict_inst`.

## PVS Manual

### Bbox Instance

1. Switch mode to **PVS Manual**.
2. Select **BBox two-click**.
3. Click two bbox corners on the image.
4. Click **Create PVS instance from bbox**.

This creates an active PVS instance and stores its low-resolution logits.

### Positive Point

1. Select **Positive point**.
2. Click one point on the image.
3. Click **Create/refine with positive point**.

If an active PVS instance exists, the point refines that instance. If no active instance exists, it creates a new PVS point instance.

### Positive Polygon Linked To Active Bbox Instance

1. Create a PVS instance from bbox first.
2. Keep that instance selected in **Active PVS instance**.
3. Select **Polygon vertex**.
4. Click polygon vertices on the same image.
5. Click **Finish polygon / refine active PVS**.

In PVS mode, finishing a polygon now directly refines the active PVS instance. If no active instance exists, the app asks you to create or select a PVS bbox instance first.

## Evaluation

Uploaded O3/LabelMe-like JSON still takes priority over COCO lookup.

Supported JSON shapes:

- `polygon`
- `rectangle`
- `linestrip`

If JSON image size differs from the current image, shapes are scaled and warnings are written to metrics.

## Validation

```bash
cd /data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/sam3-pvs-workspace
/data/zhengqiyuan/miniforge3/envs/sam3/bin/python -m py_compile sam3_gradio_demo.py
git diff --check -- sam3_gradio_demo.py README_PCS_PVS_WORKSPACE.md README_PCS_PVS_WORKSPACE_BILINGUAL.md
```
