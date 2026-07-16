# 周期性图像模板匹配后端

## 变更范围

2026-07-14 新增独立的周期性图像模板匹配能力。本次只实现后端函数、命令行入口和测试，未接入 `sam3_gradio_demo.py`，也不会影响当前运行中的 WebUI 服务。

新增文件：

- `periodic_template_matching.py`：模板匹配核心与 `smart_annotation()` 兼容入口。
- `scripts/run_periodic_template_matching.py`：JSON 请求到 JSON 结果的端到端命令。
- `tests/test_periodic_template_matching.py`：合成周期图、已有标注去重、空结果和 CLI 测试。
- `PERIODIC_TEMPLATE_MATCHING_BACKEND.md`：本说明与变更记录。

## 实现方式

1. 读取种子多边形并按原参考实现截断为整数坐标。
2. 计算多边形水平外接框，按 `expandThreshold` 向四周扩边并截取矩形 ROI。
3. 使用 `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)` 在整图搜索相同模板。
4. 将种子模板位置和 `allSegmentation` 中的已有标注作为阻塞框。
5. 对候选执行确定性的贪心 NMS；已有标注优先阻止重复补标，不使用“分数 2”魔法值。
6. 对保留位置只平移种子多边形，不旋转、缩放，也不重新分割边界。

新轮廓仍遵循：

```text
P_new = P_seed - template_top_left + matched_top_left
```

## Python 接口

兼容参考函数的参数结构：

```python
from periodic_template_matching import smart_annotation

matches = smart_annotation(
    {
        "picPath": "/path/to/periodic.png",
        "label": "part",
        "matchThreshold": 0.7,
        "expandThreshold": 20,
        "nmsThreshold": 0.3,
        "segmentation": [[110, 210], [140, 210], [140, 250]],
        "allSegmentation": [],
    }
)
```

每个结果包含：

```json
{
  "segmentation": [[520, 320], [550, 320], [550, 360]],
  "label": "part",
  "matchScore": 0.94
}
```

也可以直接调用 `match_periodic_instances()`，传入已加载的 NumPy 图像，避免重复读盘。

## 端到端命令

请求 JSON 使用上述参数结构，然后运行：

```bash
cd /data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/sam3-pvs-workspace
/data/zhengqiyuan/miniforge3/envs/sam3/bin/python \
  scripts/run_periodic_template_matching.py \
  --request /path/to/request.json \
  --output /path/to/matches.json
```

输出文件是匹配结果数组。命令不会启动模型、CUDA 或 Gradio。

## 与参考实现的兼容和修正

- 保留原参数名、矩形 ROI、归一化相关系数、已有标注去重和多边形平移语义。
- NMS 的候选分数阈值直接使用 `matchThreshold`，不再被硬编码 `0.5` 二次截断。
- 空候选直接返回空数组，不依赖 `NMSBoxes(...).flatten()`。
- 已有标注作为显式 blocker，不再依赖人为设置 `score = 2`。
- 对缺失参数、越界/退化多边形、非法阈值和无灰度变化模板给出明确错误。

## 适用边界

适合尺寸、方向和外观一致的周期性部件。当前实现只支持二维平移；旋转、缩放、透视变化、明显背景变化或需要重新贴合真实边界的场景不在本次范围内。

## 验证

```bash
cd /data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/sam3-pvs-workspace
/data/zhengqiyuan/miniforge3/envs/sam3/bin/python -m py_compile \
  periodic_template_matching.py \
  scripts/run_periodic_template_matching.py \
  tests/test_periodic_template_matching.py
/data/zhengqiyuan/miniforge3/envs/sam3/bin/python -m unittest \
  tests.test_periodic_template_matching
```
