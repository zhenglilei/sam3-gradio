## Summary

将现有 Gradio 图像分割交互从“只有正样本提示”扩展为同时支持：
- positive point / negative point prompts
- positive bbox / negative bbox prompts

实现方式是在界面增加“提示极性（正/负）”选择，并在后端把正/负提示分别解析后传入 `Sam3Processor.add_geometric_prompt(box, label, state)`，其中 `label=True` 为正框，`label=False` 为负框。

## Current State Analysis

### 代码入口
- 交互与推理主要在 [sam3_gradio_demo.py](file:///data/zhenglilei/sam3-gradio/sam3_gradio_demo.py)：
  - `handle_image_click(...)`：点击图像后把点/框坐标追加到隐藏文本框，并在图像上画红点/绿框。
  - `segment_image(...)`：解析 `point_prompt` 与 `box_prompt`（格式为 `x,y;x,y` / `x1,y1,x2,y2;...`），并调用 `image_predictor.add_geometric_prompt(box, True, state)`；目前所有几何提示都固定为正样本（True）。
- `Sam3Processor.add_geometric_prompt` 原生支持正/负框：label 为 bool，True=positive，False=negative（见 [sam3_image_processor.py](file:///data/zhenglilei/sam3-gradio/sam3/model/sam3_image_processor.py)）。

### 现有限制
- 点提示当前是用“以点击点为中心的小框”近似实现，并且全部按正样本处理。
- 框提示同样全部按正样本处理。

## Proposed Changes

### 1) UI：增加提示极性开关 + 分离正/负提示存储
文件： [sam3_gradio_demo.py](file:///data/zhenglilei/sam3-gradio/sam3_gradio_demo.py)

- 在“交互模式”区域新增一个 Radio（或 Segmented 风格的 Radio）：
  - `prompt_polarity`: 选项为 `Positive` / `Negative`，默认 `Positive`
- 将当前隐藏的两个文本框：
  - `point_prompt`
  - `box_prompt`
  替换/扩展为 4 个隐藏文本框（仍保持隐藏，仅做状态存储）：
  - `pos_point_prompt`
  - `neg_point_prompt`
  - `pos_box_prompt`
  - `neg_box_prompt`

### 2) 交互：点击逻辑支持正/负点与正/负框
文件： [sam3_gradio_demo.py](file:///data/zhenglilei/sam3-gradio/sam3_gradio_demo.py)

- 修改 `handle_image_click` 的输入/输出签名，使其接收并更新 4 个提示字符串：
  - point 模式：根据 `prompt_polarity` 把 `x,y` 追加到 `pos_point_prompt` 或 `neg_point_prompt`
  - box 模式：两次点击完成一个框，按 `prompt_polarity` 追加到 `pos_box_prompt` 或 `neg_box_prompt`
- 可视化区分正/负（不新增注释）：
  - 正点：保持现有红色
  - 负点：使用另一种高对比颜色（例如青色/紫色）
  - 正框：保持现有绿色
  - 负框：使用另一种高对比颜色（例如橙色/红色）
- 为避免“第一次点下去后用户切换正/负导致框的 label 不一致”，把 `click_state` 从 `[x,y]` 扩展为包含起点与当时 label 的结构，例如：
  - `{"start":[x,y], "label": True/False}`

### 3) 推理：解析正/负点与框，并传入正确 label
文件： [sam3_gradio_demo.py](file:///data/zhenglilei/sam3-gradio/sam3_gradio_demo.py)

- 修改 `segment_image` 形参为：
  - `pos_point_prompt, neg_point_prompt, pos_box_prompt, neg_box_prompt`
  - 保留 `text_prompt` 与 `confidence_threshold` 行为
- 校验逻辑更新：当且仅当 `text_prompt` 为空且 4 个提示都为空时提示“请提供至少一种提示”
- 点提示处理：
  - 分别解析正点/负点列表
  - 仍使用“点->小框”的近似策略（与当前一致）
  - 对正点调用 `add_geometric_prompt(box, True, state)`，对负点调用 `add_geometric_prompt(box, False, state)`
- 框提示处理：
  - 分别解析正框/负框列表（`xmin,ymin,xmax,ymax`）
  - 转为归一化 `cx,cy,w,h`
  - 对正框调用 `add_geometric_prompt(box, True, state)`，对负框调用 `add_geometric_prompt(box, False, state)`

### 4) 清空/示例按钮：同步到新状态字段
文件： [sam3_gradio_demo.py](file:///data/zhenglilei/sam3-gradio/sam3_gradio_demo.py)

- `clear_prompts`：一次性清空 4 个提示字符串并重置 `click_state`
- `example_point_btn`：写入 `pos_point_prompt`（默认示例点为正样本）

## Assumptions & Decisions

- 采用“提示极性 Radio（Positive/Negative）”作为交互方式；不依赖鼠标右键或键盘修饰键（Gradio 的 `Image.select` 事件不稳定/不可控地提供鼠标按钮信息）。
- 点提示仍按“点附近小框”实现（符合现有实现与 `Sam3Processor.add_geometric_prompt` 的接口形态）。
- 不改动视频跟踪页逻辑。

## Verification

- 启动 Gradio Demo 后在图像页验证：
  - Positive 模式下点击添加点/框：写入正提示字段，显示对应颜色
  - Negative 模式下点击添加点/框：写入负提示字段，显示对应颜色
  - 混合添加正/负点与正/负框后点击分割：不报错，且 `add_geometric_prompt` 的 label 与期望一致（负提示应能抑制不想要的区域）
  - “清空提示”能同时清空正/负提示并恢复原图

