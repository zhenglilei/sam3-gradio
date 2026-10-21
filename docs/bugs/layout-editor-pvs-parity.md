# 版图 mask 编辑器沿用 PVS-demo

撤回 `2b4f16f` 的自动选择全部 Label、工具栏、同步确认及相关界面改动。
“智能图像分割”中的版图 mask 页面沿用 `Zhengqiyuan/PVS-demo` 的操作逻辑。

`layout_transform_editor/frontend/PvsEditor.svelte` 直接取自 PVS-demo
提交 `3534e78` 的 `layout_transform_editor/frontend/Index.svelte`，内容相同。
版图页面及其绑定、mask 回调、prompt 回调也与该提交一致。
保留 PVS 原有的完整 mask / Label 选择、当前 Label、拖动、滚轮缩放、
旋转和四个画布按钮，不另行设计交互。

共享组件默认使用 `editor_mode="pvs"`。“叠层预测”页显式使用
`editor_mode="overlap"`，其前端位于 `OverlapEditor.svelte`，保持撤回前已有的
叠层编辑功能。入口组件只选择对应实现，不创建第二份 Gradio 组件状态。
这样叠层排序、交叠预览和同步协议不会影响 PVS 版图编辑器。

两个实现共用既有后端的数据筛选和静态资源构建，不增加依赖。
构建时必须让 `PYTHONPATH` 指向当前工作目录的组件 backend，避免共享环境中的
editable 安装将产物写入另一分支。

复验时对照 7892 与待发布版本：载入相同图像及版图，检查默认完整 mask、
多选 Label、当前 Label 切换、逐个拖动、滚轮缩放和生成实例的操作一致；
同时检查 764×884、390×844 视口，并确认叠层页仍使用自己的编辑器。
