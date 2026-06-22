# SAM3 PCS/PVS Interactive Workspace

<div align="center">

# SAM3 PCS/PVS/版图 Mask 交互式分割工作台

**基于 SAM3 的单图像 Gradio 工作台：PCS 自动概念分割、PVS 手动实例分割、版图 mask 提示分割。**

[功能概览](#功能概览--features) | [安装启动](#安装启动--setup) | [UI 使用说明](#ui-使用说明--ui-guide) | [导出与反馈](#导出与反馈--export-and-feedback)

</div>

---

## 项目简介 / Overview

本分支聚焦 **单图像 SAM3 交互分割**，不是视频跟踪主分支。当前 UI 把图像上传、交互提示、分割结果、实例管理、版图 mask 提示、反馈采集和导出评估放在同一个 Gradio 工作台中。

当前智能图像分割页有三个功能模式：

- **PCS Auto 自动概念分割**：用 text prompt 和正/负 bbox 样本找全某个概念的所有实例。
- **PVS Manual 手动实例分割**：用 point、bbox、polygon 创建或精修实例 mask。
- **版图 mask 提示分割**：用二值版图 mask 作为 SAM3 PVS mask prompt 创建实例。

视频目标跟踪不是本分支主功能，如需视频能力请使用原始 demo 或专门分支。

---

## 功能概览 / Features

### 1. 单工作台 / Single Workspace

- 顶部 `功能模式` 切换 PCS / PVS / 版图 mask。
- 左侧是 `原始图像（点击进行交互）`，右侧是 `分割结果`，两侧保留同尺寸区域方便对比。
- 切换模式会清理当前临时 prompt，避免 PCS/PVS/版图 prompt 串用；已生成实例不会自动删除。
- 原图侧保留当前交互标记，例如 PCS bbox、PVS pending bbox、point 和 polygon 顶点。

### 2. PCS Auto 自动概念分割

PCS 用于“找全某个概念的所有实例”。

支持输入：

- `文本提示 (Text Prompt)`：例如 `ACT-1`、`a cat`、`red apple`。
- `正样本 bbox`：框一个正确目标，告诉模型“这种是我要的”。
- `负样本 bbox`：框一个误检或不需要的对象，告诉模型“这种不要”。
- `置信度阈值 (Confidence)`：过滤低置信度预测。
- `PCS bbox 列表`：按稳定 ID 删除任意已标注 bbox。

PCS 模式只显示 `框提示 (Box)`，不显示 point/polygon。bbox 通过两次点击创建：先点起点，再点对角点。完成后会按当前 `PCS bbox 样本类型` 自动加入正样本或负样本。

### 3. PVS Manual 手动实例分割

PVS 用于“人工指定目标，再由 SAM3 生成或精修实例 mask”。

支持输入：

- `点提示 (Point)`：用正向点创建或精修当前 PVS 实例。
- `框提示 (Box)`：两次点击成框，进入待生成 bbox 队列。
- `多边形Mask (Polygon)`：点击多个顶点形成 polygon mask prompt，可创建新实例或精修当前实例。
- `PVS 待生成 bbox 列表`：按稳定 ID 删除任意待生成 bbox。
- `清空提示 (Clear Prompts)`：清空临时点、框、polygon 和 pending bbox，不删除已生成 PVS 实例。

PVS bbox 不会立刻生成实例，需要点击 `批量生成 PVS 实例` 后才会调用 SAM3 PVS 路径创建实例。

### 4. 版图 mask 提示分割

该模式把二值版图 mask 转成 PVS mask prompt，底层仍使用 PVS instance pool。

支持：

- 使用当前已保存的版图 mask。
- 载入或直接上传二值 mask PNG。
- 调整数值变换参数：`tx`、`ty`、`scale`、`rotation`、`alpha`。
- `scale` 最大支持到 `20`。
- `更新预览`：在原图上查看版图 overlay。
- `用版图创建实例`：调用 `predict_inst(mask_input=...)` 创建 PVS 实例。

当前已删除 `用版图精修当前实例` 按钮；版图模式只负责从版图 mask 创建实例。

### 5. 版图截图转掩码

`版图截图转掩码` Tab 用于把版图截图转换成二值 mask 和 contour overlay。

参数说明：

- `threshold (色彩/饱和/灰阈值)`：控制前景提取阈值。
- `invert (反转前景/背景)`：交换前景和背景。
- `open kernel`：开运算核大小，用于去除小噪点。
- `close kernel`：闭运算核大小，用于连接断裂区域。
- `min component area`：过滤小连通域。
- `区域模式`：选择保留区域策略。

右侧只显示 `binary mask 预览` 和 `contour overlay`。图片内部不再画说明文字，说明用 Markdown 标题显示。

---

## 安装启动 / Setup

### 环境要求

- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+ compatible GPU
- Gradio
- OpenCV
- SAM3 checkpoint and vocabulary files

### 创建环境

```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 安装依赖

```bash
git clone https://github.com/zhenglilei/sam3-gradio.git
cd sam3-gradio
pip install -e .
pip install gradio opencv-python matplotlib
```

如果项目中已有 `requirements.txt`：

```bash
pip install -r requirements.txt
```

### 模型文件

启动前确认：

```text
models/sam3.pt
assets/bpe_simple_vocab_16e6.txt.gz
```

远端服务器可用软链接：

```bash
mkdir -p models
ln -s /data/zhengqiyuan/.cache/modelscope/hub/models/facebook/sam3/sam3.pt models/sam3.pt
```

### 启动服务

```bash
python sam3_gradio_demo.py
```

默认端口通常是 `7890`；PVS-demo 测试服务常用 `7891`。

---

## UI 使用说明 / UI Guide

### 上传图像

1. 打开 Web 页面。
2. 在左侧上传图片。
3. 上传完成后，左侧显示可交互原图，右侧预留同尺寸分割结果。

### PCS Auto 操作流程

1. `功能模式` 选择 `PCS Auto 自动概念分割`。
2. 输入 `文本提示 (Text Prompt)`，例如 `ACT-1` 或 `a cat`。
3. 在 `PCS bbox 样本类型` 选择 `正样本 bbox` 或 `负样本 bbox`。
4. 在原图上两次点击成框，bbox 会自动加入对应样本列表。
5. 如需删除某个 bbox，在 `PCS bbox 列表` 选择对应 ID，点击 `删除选中 PCS bbox`。
6. 如需全部重来，点击 `清空提示 (Clear Prompts)`。
7. 调整 `置信度阈值 (Confidence)`。
8. 点击 `开始 PCS 分割`。
9. 检查右侧结果和分析报告。
10. 点击 `导出 PCS` 下载结果包。

注意：PCS 不允许 negative-only 运行，至少需要 text prompt 或正样本 bbox。

### PVS Manual 操作流程

#### bbox 批量创建实例

1. `功能模式` 选择 `PVS Manual 手动实例分割`。
2. `交互模式` 选择 `框提示 (Box)`。
3. 在原图上两次点击成框，bbox 会进入待生成队列。
4. 在 `PVS 待生成 bbox 列表` 可选择任意 ID 删除。
5. `清空待生成 bbox` 只清空 pending bbox，不删除已生成实例。
6. 点击 `批量生成 PVS 实例`，系统会为 pending bbox 创建实例。

#### point 创建或精修实例

1. `交互模式` 选择 `点提示 (Point)`。
2. 点击目标区域。
3. 点击 `用正向点创建/精修`。
4. 无 active instance 时创建新实例；已有 active instance 时精修当前实例。

#### polygon 创建或精修实例

1. `交互模式` 选择 `多边形Mask (Polygon)`。
2. 依次点击 polygon 顶点。
3. 选择 `多边形动作`：`创建新 PVS 实例` 或 `精修当前 PVS 实例`。
4. 如果选择精修，先在 `当前 PVS 实例` 选择目标实例。
5. 点击 `完成多边形对象`。

多边形融合方式：

| 模式 | 说明 |
| --- | --- |
| `Replace` | 重新定义实例，用当前 polygon 作为完整 mask prompt |
| `Blend` | 与旧 mask 融合，旧 logits 和 polygon logits 共同影响结果 |
| `Union` | 补充区域，保留旧 mask 并加入 polygon 区域 |
| `Intersect` | 限制范围，将结果限制在 polygon 范围内 |

#### PVS 实例管理

- `当前 PVS 实例`：选择要操作的实例。
- `撤销上一个实例`：删除最近创建的 PVS 实例。
- `清空实例`：清空当前 PVS 实例。
- `确认`：将当前实例标记为 accepted。
- `导出 PVS`：导出有效 PVS 实例。

### 版图 mask 提示分割流程

1. `功能模式` 选择 `版图 mask 提示分割`。
2. 准备版图 mask：先在 `版图截图转掩码` Tab 生成并保存，或直接上传二值 mask PNG。
3. 在右侧 `修改变形版图` 面板调整 `tx`、`ty`、`scale`、`rotation`、`alpha`。
4. 勾选 `显示/启用版图 overlay`。
5. 点击 `更新预览`。
6. 位置合适后点击 `用版图创建实例`。
7. 在 PVS 实例区确认或导出。

---

## 导出与反馈 / Export and Feedback

### Feedback

反馈写入：

```text
.runtime/feedback/feedback.jsonl
.runtime/feedback/samples/<feedback_id>/
```

单条 sample 包含：

```text
image.png
overlay.png
mask.png
mask.npz
feedback.json
```

`mask.npz` 字段固定为：

```text
mask_fullres_uint8
pvs_lowres_logits
pcs_fullres_prob
```

### COCO / LabelMe-like 评估

- 不上传 JSON 时，系统按 `COCO image file_name + split` 查找 annotation。
- 上传 O3/LabelMe-like JSON 时，上传 JSON 优先于 COCO lookup。
- 支持 `polygon`、`rectangle`、`linestrip`。
- `linestrip` 会自动闭合。
- JSON 图像尺寸不一致时，会按比例缩放 shapes 并记录 warning。

### 导出结果包

点击 `导出 PCS` 或 `导出 PVS` 后生成 zip，通常包含：

```text
overlay.png
prediction.json
metrics.json
coco_masks.json
masks/*.png
```

`coco_masks.json` 使用 COCO RLE 保存 mask，可保留空洞；不要只依赖外轮廓 polygon 表达 mask。

---

## 推荐工作流 / Recommended Workflows

### 自动找全概念实例

```text
上传图片 -> PCS Auto -> 输入 text prompt -> 可选添加正/负 bbox -> 开始 PCS 分割 -> 检查结果 -> 导出 PCS
```

### 人工标注多个目标

```text
上传图片 -> PVS Manual -> 框提示添加 pending bbox -> 批量生成 PVS 实例 -> 逐个确认 -> 导出 PVS
```

### 多边形精修复杂边界

```text
上传图片 -> PVS Manual -> 创建初始实例 -> 选择 active PVS instance -> polygon 精修 -> 确认 -> 导出 PVS
```

### 版图 mask 创建实例

```text
上传图片 -> 版图 mask 提示分割 -> 载入或生成二值版图 mask -> 调整变换 -> 更新预览 -> 用版图创建实例 -> 确认/导出 PVS
```

---

## 开发与自检 / Development Checks

远端 worktree：

```text
/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/sam3-pvs-workspace
```

常用检查：

```bash
cd /data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/sam3-pvs-workspace
/data/zhengqiyuan/miniforge3/envs/sam3/bin/python -m py_compile sam3_gradio_demo.py
git diff --check -- sam3_gradio_demo.py README.md
git status --short --branch
```

本分支主要改动集中在：

```text
sam3_gradio_demo.py
README.md
TODO.md
```

不应修改 SAM3 官方模型源码。

---

## 当前限制 / Known Limits

- 版图 mask 模式当前只保留“用版图创建实例”，不提供“用版图精修当前实例”。
- Feedback 用于 RL / 偏好数据收集，不等同于正式质检系统。
- Gradio State 暂存实例和 mask，长时间多用户并发需要迁移到 server-side cache。
- 高分辨率图像和大量实例会增加显存与内存压力。
- 当前分支不以视频跟踪为主。

---

<div align="center">

Powered by SAM3 Model
Built for interactive industrial annotation and segmentation workflows

</div>
