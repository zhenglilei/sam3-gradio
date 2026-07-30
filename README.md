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
- `版图 mask 选择`：`全部版图 mask` 保持原有单实例行为；也可以多选一个或多个已保存的独立 Label。
- 每个选中 Label 都是独立 Canvas 图层，可分别拖动、缩放和旋转；点击某个 Label 后，下方 `tx`、`ty`、`scale`、`rotation`、`alpha` 数值控件只显示并修改当前 Label。
- 模型推理会按 Label 对应的 `region_id` 升序逐条解码权威 RLE；每个 Label 独立调用一次 `predict_inst(mask_input=...)` 并创建一个 PVS instance，不做 Label 并集推理。
- 在 Canvas 编辑器里直接调整版图 mask：点中完整 mask 或某个 Label 前景后拖动，滚轮缩放，用旋转手柄旋转。
- 使用 `重置`、`居中`、`适配`、`找回视野` 快速恢复或定位版图。
- 保留数值变换参数：`tx`、`ty`、`scale`、`rotation`、`alpha`，用于精确输入和回退。
- `scale` 最大支持到 `20`。
- `更新预览`：后端用 `warpAffine()` 生成权威 overlay，并校正 Canvas 显示。
- `用版图创建实例`：提交并冻结当前 transform。完整版图调用一次 `predict_inst(mask_input=...)`；Label 多选模式按各 Label 自己的 transform 分别调用一次，并在全部成功且 Region/transform/PVS state 未变化后原子写入整批实例。
- `点提示修缮`：选择版图创建的 active PVS instance，点击左图记录单个正向点或负向点，再手动应用。

每轮点修缮只使用当前实例上一轮保存的 low-res logits 和本次单点，成功后保存新的 logits 供下一轮继续使用。没有 active 版图实例、实例来源不符或缺少 logits 时都会拒绝，不会退回原始版图二值 mask，也不会创建 point-only 实例。

### 5. 版图截图转掩码

`版图截图转掩码` Tab 用于把版图截图转换成二值 mask 和 contour overlay。

参数说明：

- `threshold (色彩/饱和/灰阈值)`：控制前景提取阈值。
- `invert (反转前景/背景)`：交换前景和背景。
- `open kernel`：开运算核大小，用于去除小噪点。
- `close kernel`：闭运算核大小，用于连接断裂区域。
- `膨胀/腐蚀像素`：`0` 不处理，正数按对应像素半径膨胀前景，负数按绝对值像素半径腐蚀前景；范围为 `-31…31`。
- `min component area`：过滤小连通域。
- `区域模式`：选择保留区域策略。

右侧先显示 `binary mask 预览` 和 `contour overlay`，其下方是独立的 `Region Annotation Layer`：

1. 切换到 `套索选择`：按住鼠标左键拖动可绘制自由线条，松开后 Draft 保持开放；随后可继续拖动补自由线，或逐点点击添加直线边。至少有 3 个不同点后点击 `完成套索`，此时才会闭合并生成黄色 `Draft` 预览。
2. 可填写一个唯一 Label 名称；名称可留空，保存时会按 Region ID 自动生成 `Label N`。
3. 点击 `保存当前 Draft Region` 后，服务端会从原始套索和 source binary mask 重新计算交集，一个套索保存成一个 Label，Draft 变为绿色 Saved Region。
4. 活动 Region 可在下拉列表中选择并软删除；软删除保留历史 RLE 和 metadata，但不再显示在列表或 overlay。

新套索只保存尚未被活动 Region 覆盖的 source-mask 像素，已经标注的重叠部分会自动忽略。软删除后的 Region 不再占用这些像素，因此其区域可被后续新套索重新标注。

新数据不再使用“大类别 + 区域/模块名称”层级，每条活动 Region 只有一个面向用户的唯一 `label`。历史文档中的 `class_label` 和 `name` 只在恢复时做兼容读取，不再作为新建、筛选或推理的业务层级。Region 的压缩 COCO RLE 是唯一权威几何，持久化到 `.runtime/layout_regions/<session>/<layout_id>/regions.json`。

Region 标注与管理控件只属于 `版图截图转掩码` Tab，不显示在 PCS 或 PVS Manual 页面。已保存的活动 Label 可传递到 Layout Mask 模式多选，并作为彼此独立的 mask prompt 创建 PVS 实例；RLE、area 和 bbox 等权威数据始终只在服务端读取，不发送到浏览器。

下载采用显式导出边界：PCS/PVS 结果包、版图 mask/contour 和 Region 标注包会复制或打包到随机 ID 的 `public_downloads/` 子目录；`.runtime/layout_masks`、`.runtime/layout_regions`、反馈、源码、配置和模型文件不允许直接下载。Region 导出包由服务端重新校验当前 session、layout、source-mask hash 和 revision，包含 `source_mask.png`、`region_label_index.png`、`labels.json`、`regions.json`、`manifest.json`，以及每个活动 Label 对应的 `label_masks/label_<index>_R<region_id>.png`。
每张 Label mask 都是独立的 8-bit 灰度二值图，`0` 为背景、`255` 为该 Label 前景；软删除 Label 不导出。`labels.json` 通过 `mask_file` 指向对应文件，`region_label_index.png` 则保留为 `uint16` 汇总索引图，值 `0` 表示背景或尚未标注的 source-mask 前景。公开副本在启动及后续导出时清理，Gradio 下载缓存每小时扫描并清理超过 24 小时的文件。


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
2. 在功能模式上方的“上传与裁剪”区域上传完整原图。
3. 上传后默认直接使用整图；如需局部分割，在完整原图上拖出矩形，再点击“应用裁剪”。
4. “使用整图”可恢复完整原图工作区。
5. 下方“原始图像”只负责 point、bbox 和 polygon 交互，不再承担上传。

裁剪图作为统一 SAM3 工作图；服务端保留完整原图及 crop provenance，用于模板匹配时映射回完整图坐标。

### PCS Auto 操作流程

1. `功能模式` 选择 `PCS Auto 自动概念分割`。
2. 输入 `文本提示 (Text Prompt)`，例如 `ACT-1` 或 `a cat`。
3. 在 `PCS bbox 样本类型` 选择 `正样本 bbox` 或 `负样本 bbox`。
4. 在原图上按住左键拖出矩形，bbox 会自动加入对应样本列表。
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
3. 在原图上按住左键拖出矩形，bbox 会进入待生成队列。
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

### 完整原图模板匹配

1. 先通过 PVS Manual 或 Layout Mask 创建分割实例，并选择当前 PVS seed。
2. 点击顶部“开始模板匹配”；默认参数为 matchThreshold=0.7、expandThreshold=20 px、nmsThreshold=0.3。
3. 结果在完整原图坐标中独立预览和导出，不写入 PVS instance pool，也不再次调用 SAM3。

Blocker 固定为：seed 始终排除自身，其他 accepted PVS 阻止重复匹配，draft/deleted 不阻止候选。导出包包含完整原图、seed mask、overlay、matches JSON 及每个结果独立的 0/255 PNG mask。


### 版图 mask 提示分割流程

1. `功能模式` 选择 `版图 mask 提示分割`。
2. 准备版图 mask：先在 `版图截图转掩码` Tab 生成并保存，或直接上传二值 mask PNG。
3. 在 Canvas 编辑器中调整版图：点中白色 mask 前景后拖动，滚轮缩放，用右上角旋转手柄旋转。
4. 如果版图移出视野，可以点击 `找回视野`；如果比例不合适，可以用 `适配`；需要重新开始则用 `重置`。
5. 右侧数值控件会同步 `tx`、`ty`、`scale`、`rotation`、`alpha`，也可以直接输入数值微调。
6. 点击 `使用当前已保存版图 mask` 加载选择器；选择 `全部版图 mask`，或多选一个或多个独立 Label。
7. 勾选 `显示/启用版图 overlay` 后点击 `更新预览`，后端会生成权威 overlay。
8. Label 多选时先在 Canvas 点中需要编辑的 Label，再拖动、缩放或旋转；下方数值控件始终对应当前 Label，其他 Label 的 transform 保持不变。
9. 位置合适后点击 `用版图创建实例`。完整版图创建一个实例；Label 多选模式中每个选中 Label 创建一个实例，整批最后一个实例自动成为 active。
10. 在 PVS 实例区选择需要修缮的版图实例。
11. 展开左侧 `点提示修缮`，点击原图记录待应用点，新点击会覆盖尚未应用的旧点。
12. 选择正向点补区域或负向点删区域，点击 `应用点提示`。
13. 成功后待应用点被清空并保存新 logits；可重复步骤 11–12 继续多轮修缮。
14. 在 PVS 实例区确认或导出。

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
上传图片 -> 版图 mask 提示分割 -> 载入或生成二值版图 mask -> 选择全部 mask 或多选独立 Label -> 分别拖动/缩放/旋转各 Label -> 更新预览 -> 按完整 mask 或逐 Label 创建实例 -> 确认/导出 PVS
```

---

## 可复现离线 PCS/PVS 评测

离线评测分为“生成固定预测产物”和“纯离线复评”两步：

1. `run_pvs_bbox_label_eval.py`、`run_pvs_bbox_grouped_eval.py` 和
   `run_pcs_o3_grouped_eval.py` 必须写入一个全新的空输出目录。
2. 正常推理完成后，每个预测 mask 都会单独保存，并由
   `run_manifest.json` 固定 selected/prediction ID、文件 hash 和顺序。
3. `run_mask_iou95_eval.py` 只读取 manifest 中列出的 mask；不会加载
   SAM3、不会重新推理，也不会扫描 `groups/*/prediction.json`。
4. 旧版 `fef70b7` 生成且没有 `run_manifest.json` 的目录必须重新生成，
   不能继续作为可信评测输入。

当前 T4 权威标注为 COCO：

```text
/data/zhengqiyuan/ADC_contour/datasets/T4/original_size/<layer>/annotations/instances_all.json
```

T4 的细分类名称读取 `annotation.original_label`，缺失时才回退到 COCO
category。O3/T4 的 annotation、image、category、尺寸和 SHA-256 都会在复评
时重新核对。COCO 中声明的 canonical image 缺失时会明确失败，不再递归选择
同名可视化副本。

示例：

```bash
python scripts/run_pvs_bbox_grouped_eval.py --datasets t4 --dry-run --out-dir /tmp/pvs_t4_check
python scripts/run_pcs_o3_grouped_eval.py --split-order train --dry-run --confidence 0.73 --max-groups 1 --out-dir /tmp/pcs_o3_check
python scripts/run_mask_iou95_eval.py --pvs-dir <completed-pvs-run> --pcs-dir <completed-pcs-run> --out-dir <new-empty-dir>
```


## 开发与自检 / Development Checks

远端 worktree：

```text
/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/sam3-pvs-workspace
```

常用检查：

```bash
cd /data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/sam3-pvs-workspace
/data/zhengqiyuan/miniforge3/envs/sam3/bin/python -m py_compile sam3_gradio_demo.py public_download_utils.py layout_transform_utils.py layout_region_utils.py layout_transform_editor/backend/gradio_layout_transform_editor/layouttransformeditor.py layout_region_annotator/backend/gradio_layout_region_annotator/layoutregionannotator.py
/data/zhengqiyuan/miniforge3/envs/sam3/bin/python tests/test_layout_transform_utils.py
/data/zhengqiyuan/miniforge3/envs/sam3/bin/python -m unittest tests.test_public_download_security tests.test_layout_region_utils tests.test_layout_region_annotator_component tests.test_layout_region_callbacks -v
cd layout_transform_editor
/data/zhengqiyuan/miniforge3/envs/sam3/bin/gradio cc build --python-path /data/zhengqiyuan/miniforge3/envs/sam3/bin/python --no-generate-docs
cd ..
cd layout_region_annotator
/data/zhengqiyuan/miniforge3/envs/sam3/bin/gradio cc build --python-path /data/zhengqiyuan/miniforge3/envs/sam3/bin/python --no-generate-docs
cd ..
git diff --check -- sam3_gradio_demo.py README.md public_download_utils.py layout_region_utils.py layout_region_annotator tests
git status --short --branch
```

本分支主要改动集中在：

```text
sam3_gradio_demo.py
README.md
TODO.md
DEMO_PROGRESS_LOG.md
layout_transform_utils.py
layout_transform_editor/
tests/test_layout_transform_utils.py
layout_region_utils.py
layout_region_annotator/
tests/test_layout_region_utils.py
tests/test_layout_region_annotator_component.py
tests/test_layout_region_callbacks.py
```

不应修改 SAM3 官方模型源码。

---

## 当前限制 / Known Limits

- 版图点提示修缮当前每次只提交一个点，不提供多点批量、reference IoU 候选选择或硬裁剪。
- Layout Label 多选批量采用全有或全无；一个保存 Label 即使含多个连通组件也只创建一个 PVS instance，不做 Label 并集推理。
- Feedback 用于 RL / 偏好数据收集，不等同于正式质检系统。
- Gradio State 暂存实例和 mask，长时间多用户并发需要迁移到 server-side cache。
- 高分辨率图像和大量实例会增加显存与内存压力。
- 当前分支不以视频跟踪为主。

---

<div align="center">

Powered by SAM3 Model
Built for interactive industrial annotation and segmentation workflows

</div>
