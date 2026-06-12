# SAM3 PCS/PVS Interactive Workspace

<div align="center">

# SAM3 PCS/PVS Interactive Workspace
# SAM3 交互式 PCS/PVS 分割工作台

**A Gradio workspace for SAM3 concept segmentation and manual instance refinement.**
**基于 SAM3 的单图像工作台，支持自动概念分割与人工实例精修。**

[功能概览](#功能概览--features) • [安装启动](#安装启动--setup) • [UI 使用说明](#ui-使用说明--ui-guide) • [导出与评估](#导出与评估--export-and-evaluation)

</div>

---

## 项目简介 / Overview

本项目是基于 SAM3 的 Gradio 图像分割工作台，当前分支聚焦 **单图像 PCS/PVS 交互分割**。

This project provides a Gradio-based SAM3 image segmentation workspace focused on **single-image PCS/PVS workflows**.

当前 UI 将图像上传、提示绘制、分割结果、分析报告和导出能力整合到一个工作台中，主要支持两种模式：

- **PCS Auto 自动概念分割**：通过文本提示和正/负 bbox 样本，让 SAM3 自动找出某个概念的所有实例。
- **PVS Manual 手动实例分割**：由业务人员手动选择目标位置，再用 bbox、point 或 polygon mask prompt 创建/精修实例。
- **Feedback 结果反馈**：对 active PVS 实例记录好/及格/差、问题标签和备注，用于后续 RL / 偏好数据收集。

The current branch intentionally focuses on image segmentation. The video tab is kept only as a reminder; video tracking should use the original demo branch.

---

## 功能概览 / Features

### 1. 单工作台 / Single Workspace

- 左侧显示原始图像，并直接在同一张图上添加交互提示。
- 右侧显示分割结果，并在 PVS 模式下显示实例操作区。
- 顶部 `功能模式` 用于切换 `PCS Auto` 和 `PVS Manual`。
- 切换模式时会重置当前交互提示，避免 PCS/PVS prompt 串用。

### 2. PCS Auto 自动概念分割

PCS 用于“找全某个概念的所有实例”。

PCS is for automatic concept segmentation: find all instances of a concept.

支持输入：

- `文本提示 (Text Prompt)`：例如 `ACT-1`、`red apple`、`yellow bus`。
- `正样本 bbox`：框一个正确目标，告诉模型“这种是我要的”。
- `负样本 bbox`：框一个误检或不需要的对象，告诉模型“这种不要”。
- `置信度阈值 (Confidence)`：过滤低置信度预测。

PCS 模式下，交互工具只保留 `框提示 (Box)`。当前实现采用 **两次点击成框**：先点 bbox 起点，再点对角点完成 bbox。

In PCS mode, only `Box` prompting is exposed. A box is created by two clicks: start corner and opposite corner.

### 3. PVS Manual 手动实例分割

PVS 用于“人工指定目标，再由模型生成或精修实例 mask”。

PVS is for manual instance segmentation and refinement.

支持输入：

- `框提示 (Box)`：两次点击成框，加入待生成 bbox 队列。
- `点提示 (Point)`：点击一个正向点，再用按钮创建或精修 PVS 实例。
- `多边形Mask (Polygon)`：点击多个顶点形成 polygon mask prompt，可创建新实例或精修当前实例。

PVS 模式下，bbox 不会立刻生成实例，而是先进入 `待生成 bbox` 队列。点击 `批量生成 PVS 实例` 后，才会调用 SAM3 PVS 路径生成实例。

In PVS mode, boxes are first collected as pending boxes. Click `批量生成 PVS 实例` to create PVS instances from them.

### 4. PVS 多边形操作 / Polygon Actions

PVS polygon 支持两类动作：

- `创建新 PVS 实例`：直接用当前 polygon 作为 mask prompt 创建新实例。
- `精修当前 PVS 实例`：用当前 polygon 精修下拉框选中的 PVS 实例。

Polygon supports two actions:

- `Create new PVS instance`: use the polygon mask prompt to create a new instance.
- `Refine active PVS instance`: refine the currently selected PVS instance.

### 5. 多边形融合方式 / Polygon Combine Modes

当使用 polygon 精修当前实例时，可以选择融合方式：

| 模式 | 中文说明 | English |
| --- | --- | --- |
| `Replace` | 重新定义实例，用当前 polygon 作为完整 mask prompt | Replace the full instance with the current polygon prompt |
| `Blend` | 与旧 mask 融合，旧 logits 和 polygon logits 共同影响结果 | Blend previous logits and polygon logits |
| `Union` | 补充区域，保留旧 mask 并加入 polygon 区域 | Add polygon area while preserving the previous mask |
| `Intersect` | 限制范围，将结果限制在 polygon 范围内 | Restrict the result inside the polygon region |

默认推荐使用 `Replace`，适合业务人员画完整目标轮廓的场景。

### 6. 结果反馈 / Feedback

Feedback 用于收集 PVS 分割结果的人工质量判断。首版只支持当前 active PVS instance；没有 active PVS instance 时不会写入空反馈。

支持内容：

- `结果质量`：好 / 及格 / 差。
- `问题标签`：毛边、空缺、漏检、误检、边界偏移、多分/粘连、polygon 不贴合、其他。
- `备注`：记录具体问题或可用性说明。

---

## 安装启动 / Setup

### 环境要求 / Requirements

- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+ compatible GPU
- Gradio
- OpenCV
- SAM3 checkpoint and vocabulary files

### 创建环境 / Create Environment

```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 安装项目依赖 / Install Dependencies

```bash
git clone https://github.com/zhenglilei/sam3-gradio.git
cd sam3-gradio
pip install -e .
pip install gradio opencv-python matplotlib
```

如果项目中已有 `requirements.txt`，也可以使用：

```bash
pip install -r requirements.txt
```

### 模型文件 / Model Files

启动前请确认以下文件存在：

```text
models/sam3.pt
assets/bpe_simple_vocab_16e6.txt.gz
```

远端服务器上可以使用软链接，例如：

```bash
mkdir -p models
ln -s /data/zhengqiyuan/.cache/modelscope/hub/models/facebook/sam3/sam3.pt models/sam3.pt
```

### 启动服务 / Run

```bash
python sam3_gradio_demo.py
```

默认监听：

```text
http://localhost:7890
```

服务器部署时，代码中使用：

```text
server_name = 0.0.0.0
server_port = 7890
```

如果需要临时测试其他端口，可以用独立启动脚本或修改 `server_port`。

---

## UI 使用说明 / UI Guide

### 1. 上传图像 / Upload Image

1. 打开 Web 页面。
2. 在左侧原图区域点击上传，或拖入图片。
3. 上传完成后，左侧显示可交互原图，右侧预留同尺寸分割结果区域。

After upload, the original image becomes the interaction canvas, and the result area is reserved on the right.

---

### 2. PCS Auto 使用流程

PCS 适合自动找全某类目标。

#### 操作步骤

1. 在顶部 `功能模式` 选择 `PCS Auto 自动概念分割`。
2. 在 `文本提示 (Text Prompt)` 输入目标概念，例如：

   ```text
   ACT-1
   red apple
   yellow school bus
   ```

3. 如需要样本提示，在 `PCS bbox 样本类型` 中选择：

   - `正样本 bbox`：正确目标样本。
   - `负样本 bbox`：不想要的对象样本。

4. 在原图上使用 `框提示 (Box)`：

   - 第一次点击 bbox 起点。
   - 第二次点击 bbox 对角点。
   - 完成后 bbox 会自动加入 PCS 样本，不需要额外点击“添加 bbox”。

5. 如点错 bbox，点击 `撤销最近 PCS bbox`。
6. 调整 `置信度阈值 (Confidence)`。
7. 点击 `开始 PCS 分割`。
8. 右侧显示分割结果，左侧/报告区显示 PCS 实例数量和 score。
9. 点击 `导出 PCS` 导出结果包。

#### 注意事项

- PCS 模式不使用 polygon。
- PCS 模式不暴露 point prompt。
- 不允许只用负样本 bbox 运行 PCS；至少需要文本提示或正样本 bbox。

---

### 3. PVS Manual 使用流程

PVS 适合业务人员手动指定目标，然后让 SAM3 生成或精修实例 mask。

#### A. 用 bbox 批量创建 PVS 实例

1. 顶部 `功能模式` 选择 `PVS Manual 手动实例分割`。
2. `交互模式` 选择 `框提示 (Box)`。
3. 在原图上两次点击成框：

   - 第一次点击 bbox 起点。
   - 第二次点击 bbox 对角点。

4. 每完成一个 bbox，都会进入 `待生成 bbox` 队列。
5. 可以连续框选多个目标。
6. 如点错：

   - `撤销最近待生成 bbox`：只撤销最后一个待生成 bbox。
   - `清空待生成 bbox`：只清空 pending queue，不删除已生成实例。

7. 点击 `批量生成 PVS 实例`。
8. 系统会为每个 pending bbox 创建一个 PVS 实例，并自动选择最后一个作为当前实例。

#### B. 用正向点创建或精修 PVS 实例

1. `交互模式` 选择 `点提示 (Point)`。
2. 在原图上点击目标区域。
3. 点击 `用正向点创建/精修`。
4. 如果当前没有 active PVS 实例，会创建新实例。
5. 如果已经选择了 active PVS 实例，会用该点精修当前实例。

#### C. 用 polygon 创建 PVS 实例

1. `交互模式` 选择 `多边形Mask (Polygon)`。
2. 在图像上依次点击 polygon 顶点。
3. 在右侧 `PVS 实例操作` 中选择：

   ```text
   多边形动作 = 创建新 PVS 实例
   ```

4. 点击 `完成多边形对象`。
5. 系统会将 polygon 转成 SAM3 PVS 的 mask prompt，并创建新 PVS 实例。

#### D. 用 polygon 精修当前 PVS 实例

1. 在 `当前 PVS 实例` 下拉框选择要精修的实例。
2. `交互模式` 选择 `多边形Mask (Polygon)`。
3. 在图像上点击 polygon 顶点。
4. 在右侧选择：

   ```text
   多边形动作 = 精修当前 PVS 实例
   多边形融合方式 = Replace / Blend / Union / Intersect
   ```

5. 点击 `完成多边形对象`。
6. 当前 PVS 实例会被替换为精修后的 mask。

#### E. PVS 实例管理

- `当前 PVS 实例`：选择要精修、确认或删除的实例。
- `撤销`：撤销当前实例最近一次 refine 操作。
- `删除`：将当前实例标记为 deleted。
- `确认`：将当前实例标记为 accepted。
- `清空草稿 PVS 实例`：只删除未确认的 draft 实例，保留 accepted 实例。
- `导出 PVS`：导出当前有效 PVS 实例。

---

### 4. Feedback 使用流程

Feedback 适合在生成 PVS 实例后记录人工质量判断，用于后续 RL / 偏好优化数据收集。

1. 选择 `PVS Manual 手动实例分割`。
2. 通过 bbox、point 或 polygon 创建 PVS 实例。
3. 在 `当前 PVS 实例` 下拉框中选择需要评价的 active instance。
4. 打开右侧 `结果反馈（用于 RL 数据收集）` 面板。
5. 选择 `结果质量`：好 / 及格 / 差。
6. 可选勾选问题标签并填写备注。
7. 点击 `提交反馈`。

反馈会写入：

```text
.runtime/feedback/feedback.jsonl
.runtime/feedback/samples/<feedback_id>/
```

单条 sample 目录包含：

```text
image.png
overlay.png
mask.png
mask.npz
feedback.json
```

`mask.npz` 中字段固定为：

```text
mask_fullres_uint8
pvs_lowres_logits
pcs_fullres_prob
```

---

## 分析报告字段说明 / Report Fields

分析报告会根据当前模式显示不同内容。

### PCS 报告

- `正样本 bbox`：当前 PCS positive exemplar 数量。
- `负样本 bbox`：当前 PCS negative exemplar 数量。
- `PCS 实例`：当前 PCS 输出实例数量。
- `score`：模型输出的候选置信度/质量估计。

### PVS 报告

- `PVS 实例`：当前有效 PVS 实例数量。
- `待生成 bbox`：还没有批量生成的 pending bbox 数量。
- `当前实例`：当前 active PVS instance id。
- `草稿 / draft`：表示实例尚未点击 `确认`。
- `已确认 / accepted`：表示实例已确认，可作为正式导出对象。
- `score`：SAM3 返回的候选 mask 质量/置信估计，不等同于人工质检分数。

---

## 导出与评估 / Export and Evaluation

打开 `导出与 COCO 量化` 面板后，可以选择评估数据来源。

### 1. COCO Lookup

如果不上传 JSON，系统会按：

```text
COCO image file_name + split
```

去数据集中查找 annotation。

支持配置：

- `指标数据集`
- `COCO image file_name`
- `标注 split`: `auto` / `val` / `train` / `test`
- `评估范围`

### 2. O3 / LabelMe-like JSON 上传

如果上传了 O3/LabelMe-like JSON，则优先使用上传 JSON，而不是 COCO lookup。

Uploaded JSON has priority over COCO lookup.

支持字段：

```json
{
  "imageWidth": 1234,
  "imageHeight": 987,
  "shapes": [
    {
      "label": "ACT-1",
      "shape_type": "polygon",
      "points": [[10, 20], [100, 20], [100, 80]]
    }
  ]
}
```

支持的 `shape_type`：

- `polygon`
- `rectangle`
- `linestrip`

说明：

- `linestrip` 会自动闭合为 polygon。
- 点数少于 3 的 shape 会跳过并给出 warning。
- 上传 JSON 的 `imageWidth/imageHeight` 和当前图像不一致时，系统会按比例缩放 shape，并在评估结果中记录 warning。

### 3. 导出结果包

点击 `导出 PCS` 或 `导出 PVS` 后，会生成 zip 包，通常包含：

```text
overlay.png
prediction.json
metrics.json
coco_masks.json
masks/*.png
```

`prediction.json` 中会记录：

- instance id
- source
- status
- score
- bbox_xyxy
- mask file
- final contour polygon
- COCO RLE mask in `coco_masks.json`
- prompt history

---

## 推荐工作流 / Recommended Workflows

### 场景 A：自动找全概念实例

```text
上传图片
→ 选择 PCS Auto
→ 输入 text prompt
→ 可选：添加正/负 bbox 样本
→ 开始 PCS 分割
→ 检查结果
→ 导出 PCS
```

### 场景 B：人工框选所有目标

```text
上传图片
→ 选择 PVS Manual
→ 框提示连续添加多个 bbox
→ 批量生成 PVS 实例
→ 逐个检查实例
→ 确认 accepted
→ 导出 PVS
```

### 场景 C：多边形精修复杂边界

```text
上传图片
→ 选择 PVS Manual
→ bbox 或 point 创建初始实例
→ 选择当前 PVS 实例
→ polygon 画复杂边界
→ 选择 refine + combine mode
→ 完成多边形对象
→ 确认实例
→ 导出 PVS
```

### 场景 D：只用 polygon mask prompt 创建实例

```text
上传图片
→ 选择 PVS Manual
→ 选择多边形Mask
→ 点击 polygon 顶点
→ 多边形动作选择“创建新 PVS 实例”
→ 完成多边形对象
→ 确认实例
```

---

## 注意事项 / Notes

- 当前分支重点是图像分割，视频目标跟踪请使用原始 demo 分支。
- PCS 和 PVS 是不同语义：PCS 走概念 grounding，PVS 走实例交互分割。
- PCS 不使用 point/polygon；PVS 支持 point/bbox/polygon。
- PVS 中 `清空待生成 bbox` 不会删除已生成实例。
- `清空草稿 PVS 实例` 只删除 draft 实例，不删除 accepted 实例。
- Feedback 首版只记录 active PVS instance；没有 active PVS instance 时不会写入空记录。
- 高分辨率图片和大量实例会占用更多显存和内存。
- 首次启动需要加载 SAM3 模型，可能需要等待一段时间。

---

## Development Notes

This branch keeps most changes inside:

```text
sam3_gradio_demo.py
```

The model source under `sam3/model/*` is not modified by this UI workspace branch.

---

<div align="center">

Powered by SAM3 Model
Built for interactive industrial annotation and segmentation workflows

</div>
