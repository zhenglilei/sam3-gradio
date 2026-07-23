# PVS Demo 变更历史

## 1. 文档范围

- 分支：`Zhengqiyuan/PVS-demo`
- 远端工作区：`/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/sam3-pvs-workspace`
- 基线：`fef70b7`（2026-07-13，`Add grouped PCS and PVS evaluation scripts`）
- 记录截止：2026-07-23
- 范围：基线之后的正式提交，以及本文件所在提交整合的 Layout Region/PVS 与预览改动。
- `DEMO_PROGRESS_LOG.md` 是独立工作日志，不属于本变更历史，也不纳入本次提交。

## 2. 总体架构演进

当前 WebUI 保持三类职责边界：

1. **PCS Auto**：文本和正向 bbox 驱动自动概念分割。
2. **PVS Manual**：bbox、点、polygon、mask 等人工实例提示。
3. **Layout Mask**：版图 binary mask 对齐、按完整 mask 或保存 Region 创建 PVS 实例，并支持基于上一轮 low-res logits 的单点修缮。

版图截图页单独承担：

```text
截图生成 binary mask
-> Region 套索划分
-> 类别/名称标注
-> Region RLE 持久化
```

关键数据约束：

- source binary mask 是版图几何的基础事实。
- Region 的压缩 COCO RLE 是 Region 几何的唯一权威数据。
- contour、类别并集和前端 overlay 只用于展示，不能替代服务端权威 mask。
- PCS、PVS Manual 与 Layout Mask 保持独立 UI callback；`common = [...]` 的成员和顺序保持稳定。
- 不修改 SAM3 模型源码。

## 3. 已提交历史（fef70b7 之后）

| 提交 | 日期 | 主要变化 |
|---|---|---|
| `b1573ef` | 2026-07-14 | 新增 Layout Region 持久化基础：类别配置、COCO RLE、metadata、原子写盘、软删除和后端测试。 |
| `4e18373` | 2026-07-14 | 新增独立 `LayoutRegionAnnotator` Custom Component、Svelte 前端、构建产物和组件测试。 |
| `2aff218` | 2026-07-14 | 将 Region 标注层接入版图截图转掩码页面，加入类别、可选名称、保存、选择、软删除和导出流程。 |
| `545234e` | 2026-07-14 | 保存新 Region 时排除已被活动 Region 覆盖的像素，避免活动区域重叠。 |
| `b61f0a5` | 2026-07-16 | 新增独立 periodic template matching 后端、CLI、文档和测试。 |
| `a3ccf98` | 2026-07-16 | 收紧 Gradio 下载边界，将可下载产物复制到受控 public 目录，并阻止直接访问运行时、源码、配置和模型路径。 |
| `033b87c` | 2026-07-16 | 重构离线 PCS/PVS 分组评测，补充可复现 artifact、IoU95 pipeline、collector 和测试。 |
| `2f2f0ca` | 2026-07-16 | 将套索 Draft 绑定到 session/layout/source-mask identity，防止切图或异步加载时提交旧轨迹。 |
| `6b5d47f` | 2026-07-16 | 恢复 Region 历史时拒绝活动 Region 重叠，避免加载不一致文档。 |
| `260ddce` | 2026-07-16 | 使用跨进程锁串行化同一 Layout 的 Region 写入，强化 revision 与原子持久化。 |
| `1712848` | 2026-07-16 | 限制 periodic template matching 候选规模和资源占用。 |
| `e1f25e5` | 2026-07-16 | 隔离不同 session workspace，并使 PVS 批量创建采用原子提交语义。 |
| `1407f8f` | 2026-07-16 | 为 workspace、实例和相关内存状态增加容量/生命周期边界。 |
| `cb85912` | 2026-07-16 | 恢复 PVS 分割后左图 prompt 框，并增强 PCS mask overlay 的可见度。 |
| `b42d59e` | 2026-07-17 | 新增 Layout Mask 独立点提示修缮：仅修缮版图实例，逐轮使用上一轮 low-res logits 与本次单点。 |

## 4. 2026-07-23 集成改动

### 4.1 套索支持自由线与直线混合修饰

旧行为在鼠标松开时立即闭合，无法继续修边。新交互改为开放 Draft：

1. 按住拖动绘制自由线段。
2. 松开后保留开放轨迹，不请求 Python。
3. 可以继续拖动补自由线，也可以逐点点击添加直线顶点。
4. 至少有 3 个不同点后点击“完成套索”。
5. 只有显式完成时才闭合，并且只触发一次后端 Draft 预览请求。

仍保留 3 CSS px 采样、4096 点上限、pointer capture、identity 校验以及 cancel/blur 清理。

### 4.2 Layout Region 按模块创建 PVS 实例

Layout Mask 面板新增“版图 mask 选择”：

- **全部版图 mask**：保持原有完整 binary mask 单实例行为。
- **类别：class_label**：选择该类别的所有活动 Region。

类别模式的语义：

- Canvas 显示该类别 Region 的并集，仅用于整体预览。
- 推理时按 `region_id` 升序逐条解码每个 Region 的权威 RLE。
- 每个保存 Region 单独 warp、单独生成 mask logits、单独调用一次 `predict_inst(mask_input=...)`。
- 一个 Region 即使包含多个连通组件，也只创建一个 PVS instance。
- 软删除 Region 不进入类别预览或推理；历史类别即使不在当前类别配置中仍可使用。
- SAM3 输出不硬裁剪回套索，后续可继续使用正负点修缮。

### 4.3 批量提交一致性

同类多 Region 创建采用全有或全无：

- 预测前冻结 Region revision、Region mask fingerprint、transform matrix/revision、target image hash、选择 epoch 与 PVS commit token。
- 所有预测先保存在内存中，不立即修改 `pvs_state`。
- 任一 Region 预测失败，或预测期间 Region、选择、transform、目标图片、PVS state 发生变化，整批结果均不提交。
- 全部通过后一次性分配连续 PVS ID，最后一个新实例成为 active。
- prompt history 记录唯一 `region_id`、类别、批次索引、Region hash、revision 与 frozen affine matrix。

Feedback 导出不再依赖易失的 Layout cache，而是根据冻结 metadata 从完整 source mask 或唯一 Region RLE 重建当时的 transformed prompt mask；hash 不匹配时明确标记不可恢复，不回退到其他 mask。

### 4.4 版图预览横向分页

版图截图转掩码页面中的 `binary mask 预览` 与 `contour overlay` 已合并到同一横向 scroll-snap 容器：

- 一次显示一个完整宽度页面。
- 支持触控/触控板横向滑动，以及拖动底部滚动条。
- 两个原 `gr.Image` 和生成/清除 callback 的输出顺序不变。
- 仅改变展示层，不改变 binary mask 或 contour 的生成逻辑。

## 5. 主要持久化与运行时目录

| 路径 | 用途 |
|---|---|
| `.runtime/layout_masks/<session>/<layout_id>/` | source mask、版图身份和相关文件。 |
| `.runtime/layout_regions/<session>/<layout_id>/regions.json` | Region RLE、类别、名称、revision、ID 和软删除历史。 |
| `.runtime/exports/` | 服务端导出中间结果。 |
| `public_downloads/` | 允许由 Gradio 提供下载的受控副本。 |
| `.runtime/logs/` | 服务日志。 |

## 6. 回归与验证

本次集成在远端 Linux 环境完成以下验证：

- `gradio cc build --no-generate-docs --python-path /data/zhengqiyuan/miniforge3/envs/sam3/bin/python`：Custom Component 前端与 Python package 构建通过。
- `python -m unittest discover -s tests -p 'test_*.py'`：107 项测试通过。
- `python -m py_compile ...`：修改的 Python 模块通过语法检查。
- `create_demo()`/配置测试：Layout Region 控件仍位于版图页；横向预览配置包含 1 个滑轨和 2 个页面。
- `git diff --check`：提交前检查通过。
- 服务重启仅作用于 7890；7891 保持关闭。

## 7. 当前限制

- Layout 点修缮每次只提交一个正点或负点。
- Region 类别批次不做同类并集单次推理，每个活动 Region 始终对应一个独立实例。
- Region 内多个连通组件不会拆成多个实例。
- 不做 reference IoU 候选选择、硬裁剪或多点批量修缮。
- 本文记录源码和可复现行为，不记录易变化的服务 PID。
