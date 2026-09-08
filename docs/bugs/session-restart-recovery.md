# Bug 单：服务重启后旧页面保留假状态并触发 SessionGuardError

- 严重级别：P0
- 类型：会话生命周期、状态一致性、可恢复性
- 状态：修复中；P0/P1 已在独立 bugfix 分支实现，待跨分支回归和部署验证
- 影响范围：需要同步评估并修复 PVS-demo、sam3-el-mask、overlap-prediction 三个分支
- 现场服务：7891

## 1. 问题摘要

服务进程重启后，浏览器页面没有重新建立服务端会话。浏览器仍保留旧的
上传文件列表和控件值，但新进程中的会话注册表、Gradio gr.State 和业务对象
已经清空。用户继续点击周期拼接或其他状态型控件时，旧页面携带的状态无法通过
会话守卫，于是显示：

~~~
stitch_state has no server session id
~~~

同一操作关联的多个回调会分别抛出相同错误，最终在页面上产生多条红色 Error。
这不是一次普通拼接算法失败，而是“前端可见状态”和“后端权威状态”失去一致性
后的恢复路径缺失。

## 2. 现场证据

以下证据来自 7891 页面现场观察和运行记录：

1. 7891 在新进程启动后，用户只等待了几分钟就再次操作；默认会话空闲阈值为
   3600 秒，故该现象不能解释为正常的空闲 TTL 清理。
2. 周期拼接页面仍显示之前上传的四个 shot_*.bmp 文件名，说明浏览器侧的
   DOM/组件值没有同步清空。
3. 拼接画布和相关控件显示 Error，右侧错误面板重复出现三条：

   ~~~
   stitch_state has no server session id
   ~~~

4. 当前代码把会话注册表创建在应用进程内，并在启动时默认生成新的 HMAC secret：
   [session_runtime.py:135](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/session_runtime.py:135)
   [session_runtime.py:168](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/session_runtime.py:168)
5. 当前默认参数为 idle=3600、清理检查间隔 5 秒、最大会话数 128：
   [config.py:67](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/config.py:67)
   [config.py:71](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/config.py:71)

因此当前证据支持的直接结论是：服务重启导致进程内会话丢失，旧页面没有重连
握手；不能把这次现象归因于 3600 秒 TTL。

## 3. 当前正常会话隔离链路

现有设计在同一个 Python 进程生命周期内的隔离目标是明确的。正常首次加载
链路如下：

1. 浏览器加载页面后，demo.load() 调用 _bootstrap_session()：
   [app.py:4349](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/app.py:4349)
2. _bootstrap_session() 从 Gradio Request 取得 session_hash 和客户端地址，
   经过可信代理规则处理后调用 SessionRegistry.bind()：
   [app.py:4108](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/app.py:4108)
3. SessionRegistry 以 session_hash 摘要和规范化客户端 IP 为索引，生成新的
   session_id、generation、owner_token 及对应摘要；已有同一身份会话则
   更新 last_seen，不会把不同身份合并。
4. _session_state_bundle() 把同一会话身份写入所有业务状态，包括图像、PCS、
   PVS、模板匹配、提示、版图、版图区域、版图 Mask Agent、周期拼接和模板拼接：
   [app.py:4074](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/app.py:4074)
5. 业务回调由 guard_callback() 包装。它要求 state 或 *_state 是带 session_id
   的映射，然后校验 session_id、generation、owner_token、session_hash 摘要
   和客户端 IP，再取得该会话的 lease 后才执行原业务函数：
   [session_guard.py:128](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/session_guard.py:128)
   [session_guard.py:229](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/session_guard.py:229)
6. 周期拼接状态的初始值是空字典，只有会话 bootstrap 成功后才会替换为带
   session_id 和 owner_token 的状态：
   [stitch_tab.py:24](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/ui/stitch_tab.py:24)
   [stitch_callbacks.py:42](Z:/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/bugfix-session-recovery/sam3_demo/stitch_callbacks.py:42)

这条链路的安全含义是：状态不带完整身份时宁可拒绝，也不能猜测用户身份或
执行可能写入其他工作区的操作。问题不在于该拒绝本身，而在于缺少失效后的
可恢复路径和用户可理解的反馈。

## 4. 故障链

~~~
T0 旧进程拥有 SessionRegistry、Gradio State、上传对象和拼接状态
  ↓
T1 7891 重启；新的 Python 进程重新创建注册表和随机 HMAC secret
  ↓
T2 旧进程内的 _records、_keys、PIL/临时文件关联和 gr.State 不再存在
  ↓
T3 浏览器连接到新进程，但页面 DOM 仍显示旧文件名和旧控件值
  ↓
T4 当前页面没有 server epoch/heartbeat/reconnect handshake，
   demo.load() 不会因为后台进程替换而自动再次执行
  ↓
T5 用户触发周期拼接或其他状态型回调；后端收到空状态或没有 session_id 的状态
  ↓
T6 guard_callback() 在业务函数执行前抛出 SessionGuardError
  ↓
T7 多个回调各自处理失败，出现重复 Error；没有统一恢复入口
~~~

如果是 TTL 或容量淘汰，故障链的前两步不同：服务仍在运行，清理器删除的是
某一条会话记录；但在两种情况下，前端都必须先发现状态失效，再安全地重建会话，
不能继续把旧身份当成有效身份。

## 5. 影响范围

### 5.1 三个分支

本问题位于会话注册、会话守卫和全局 Gradio 状态的共用链路，不能只在周期拼接
页面打补丁。需要在以下三个分支分别验证并保持修复一致：

- PVS-demo
- sam3-el-mask
- overlap-prediction

各分支可以有不同 UI 或业务提交，但不得让某一分支绕过会话所有权校验。

### 5.2 所有有状态工作流

_session_state_bundle() 当前覆盖的状态包括：

- image_state
- source_image_state
- pcs_state
- pvs_state
- template_match_state
- prompt_state
- layout_state
- layout_region_state
- layout_mask_agent_state
- stitch_state
- template_stitch_state

因此风险覆盖智能图像分割、PCS/PVS、提示交互、版图及区域/Mask Agent、
周期拼接、模板拼接等所有依赖这些 gr.State 的工作流。具体表现可能分别是
上传/裁剪不能继续、提示或实例状态消失、布局编辑失效、拼接画布报错；不能只
以 stitch_state 的错误为孤立问题。

### 5.3 用户影响

- 重启后用户无法判断页面是否仍然有效，重复点击只会累积错误提示。
- 未持久化的上传文件、排序、裁剪、位移、旋转、提示和实例状态需要重新操作。
- 多用户同时使用时，若通过“按 IP 直接恢复”修复，可能把两个用户错误地合并；
  这是比当前报错更严重的数据隔离事故。
- 多标签页可能共享浏览器可见资源但拥有不同 Gradio session_hash，简单使用
  localStorage 或 IP 作为唯一恢复键会造成串状态风险。

## 6. 不可破坏的安全约束

修复必须改善恢复体验，但不得削弱当前隔离边界：

1. 不得把浏览器提交的旧业务字典直接换绑给新用户或新会话。
2. 不得仅凭客户端 IP 恢复状态。IP 只能作为当前部署中已有的身份校验输入，
   不能作为持久恢复凭据。
3. 继续校验 owner_token、session_hash（及其摘要）、客户端 IP、generation
   和状态 schema；旧 token 在进程重启后不可直接被新进程接受。
4. 服务重启后应创建新的服务端会话。只有在服务端能够验证稳定、不可伪造的
   workspace 恢复凭据，并从服务端持久化存储取回数据时，才允许恢复业务草稿。
5. 恢复过程中旧请求、迟到响应和旧标签页不能写入新会话；状态 epoch/revision
   不匹配必须丢弃并给出可解释结果。
6. 断网、TTL 失效和容量淘汰都要走同一类安全失效路径，不能为了避免红框而把
   校验异常吞掉后继续执行。
7. 错误展示应统一、去重并可恢复，但不能把 owner token、完整 session hash、
   文件真实路径或内部 secret 返回给浏览器。

## 7. 分阶段修复目标与验收

### P0：自动发现失效、握手和安全重绑

P0 的目标是让页面在服务重启/连接重建后自愈到一个合法状态，且不伪造旧
业务状态。验收条件：

1. 服务提供轻量的 server_epoch 或等价握手信息；页面首次加载和重新连接时
   都执行握手，而不是只依赖一次 demo.load()。
2. 检测到 epoch 变化、状态校验失败或连接重建后，前端在一个明确的心跳/重连
   周期内将受影响控件置为不可操作，停止继续发送旧状态请求。
3. 服务端创建新的合法会话；旧 session_id、generation、owner_token、旧业务
   状态和迟到响应均不能被直接接受。
4. 当前没有持久化草稿可恢复时，页面清理或明确标记旧文件列表，显示一条去重的
   友好提示，例如“服务已重启，当前工作区需要重新加载”；不能留下“看似有
   文件但实际无法操作”的假状态。
5. 一次失效事件最多显示一条用户提示；关联回调不再各自弹出相同红色 Python
   异常。日志保留可检索的内部错误码，例如 SESSION_EXPIRED、
   SERVER_RESTARTED，但不泄漏敏感身份字段。
6. 在智能图像分割、周期拼接、模板拼接三个代表流程中，服务重启后刷新/重连
   能重新建立会话，页面可继续上传新输入；全程不绕过会话守卫。
7. P0 必须有自动化测试覆盖：epoch 变化、握手重绑、旧 ACK/旧状态丢弃、
   错误去重和安全边界；还要用真实浏览器验证控件禁用、单条提示和重新上传。

P0 不承诺恢复服务重启前的业务数据；没有持久化数据时，安全的行为是建立
新会话并明确告知用户需要重新加载。

### P1：周期拼接草稿恢复

P1 在 P0 的合法会话基础上增加可控的持久化恢复。验收条件：

1. 上传后将输入文件复制到会话/workspace 的服务端持久目录，不能只保存
   Gradio 临时路径或 Python PIL 对象引用。
2. 防抖保存周期拼接的文件 hash/元数据、顺序、布局、选中项、平移、旋转、
   裁剪参数、黑边处理选项、revision 和必要的标注/导出关系；派生的预览图
   可以在恢复后重建。
3. 保存采用不可变 revision 或等价 compare-and-swap；服务重启、网络重连、
   多标签页迟到写入不能覆盖较新的草稿。
4. P1 恢复只能读取服务端验证过的 workspace 记录。新会话恢复后重新生成当前
   session_id/owner_token，不能把旧 owner token 写回客户端作为有效凭据。
5. 明确区分内存会话 TTL、会话容量淘汰和草稿保留期。TTL/容量清理内存对象后，
   在保留期内仍可通过合法恢复流程取回草稿；过期草稿应给出“已过期”的明确
   状态，而不是 no server session id。
6. P1 至少用 O3_test 周期拼接走完“上传→排序/变换→保存→进程重启→恢复→
   继续编辑→导出”链路，并检查恢复前后 revision、文件 hash 和变换数值一致。

## 8. 测试矩阵

| 场景 | 测试设置 | 预期用户行为 | 预期安全结果 |
|---|---|---|---|
| 首次加载 | 新无痕页打开任一分支 | 握手完成后控件可操作 | 新建唯一 session_id；状态全部带当前 owner |
| 正常空闲未超时 | 连续操作后等待小于 3600 秒 | 页面保持可用 | last_seen 正常刷新；无重绑、无数据变化 |
| 服务重启 | 上传并编辑后重启对应服务，保留原标签页 | 自动检测；控件暂时禁用；只显示一条恢复提示；P0 可重新上传 | 新 epoch/新会话；旧状态/token/迟到请求全部拒绝或丢弃 |
| 断网重连 | 浏览器与服务断开后恢复，进程不重启 | 页面显示连接恢复状态，握手后继续；未实现 P1 时明确提示需重新加载 | 不能仅凭旧前端字典恢复；旧请求不写新会话 |
| TTL 失效 | 将测试 TTL 缩短，等待 sweeper 清理后操作 | 统一“会话已过期”提示和恢复入口 | 被清理记录不可再 lease；草稿按独立保留期处理 |
| 容量淘汰 | max_sessions 设小，创建超过上限的空闲会话 | 被淘汰页得到单条可理解提示 | 只淘汰无 in-flight 会话；不能影响其他用户或错误串绑 |
| 双用户隔离 | 两个不同 session_hash/用户同时上传不同文件并操作 | 各自只看到和修改自己的工作区 | owner token、hash、IP 校验保留；A 的状态不能被 B 接受 |
| 多标签页 | 同一浏览器开两个标签，分别加载/编辑；再重启服务 | 每个标签都执行握手；冲突时提示并按 revision 处理 | 不把 IP 或 localStorage 当唯一用户；旧标签不能覆盖新标签 |
| 重启中有请求 | 在上传/拼接或保存进行时重启进程 | 请求失败后进入统一恢复状态，可重试 | 不产生半写入、半更新或跨 generation 的状态 |
| P1 草稿恢复 | 保存周期拼接后重启、TTL/容量清理内存 | 在保留期内恢复同一文件 hash、顺序和变换 | 新会话读取服务端草稿；不恢复旧 owner token |

所有矩阵项都要在三个分支执行最小代表测试；对会话共用层的修复应有一套
可复用的单元/集成测试，而不能只验证 7891 的一个页面。

## 9. 跨分支合并顺序

建议将修复拆成可回溯的独立提交，并按以下顺序推广：

1. 在专用 bugfix 分支先完成会话握手、epoch/失效错误模型和通用测试；不要把
   分支专属的 UI 或业务改动混入公共提交。
2. 以 PVS-demo 作为第一套集成和回归基线，完成 P0 后运行会话测试、全量
   unittest、create_demo() 和真实浏览器重启/重连检查。
3. 将已经审核的公共提交按 git cherry-pick -x（或等价可追踪 merge）带入
   sam3-el-mask，解决只属于该分支的冲突后重复全部检查。
4. 再带入 overlap-prediction；最后分别集成 P1 的周期拼接持久化，避免把
   未验证的业务恢复逻辑同时复制到三个分支。
5. 每个分支合并前确认工作区干净、没有覆盖用户未提交文件；每个服务的部署和
   重启由主 Agent 按端口/PID/工作目录核对后单独执行。本 bug 单子任务不执行
   提交、推送或服务操作。

如果三个分支没有共同的干净祖先，公共提交仍应保持最小化，并使用
git cherry-pick -x 保留来源；不要把一个分支的整棵业务历史强行 merge 到另一个
分支。

## 10. 回滚说明

- 回滚按各分支实际应用的提交逆序执行，优先使用独立 git revert <commit>，
  不使用可能覆盖用户工作区的 reset --hard 或 checkout --。
- P0/P1 提交必须保留旧状态字段的兼容读取或明确迁移策略；回滚应用层代码时，
  不能让已经写入的草稿数据变成不可读。
- 若仅回滚 P1，应保留 P0 的安全失效提示；若必须回滚 P0，旧页面在服务重启
  后应至少安全失败，不能恢复成未经校验的状态直通。
- 回滚后重新运行会话专项测试、全量测试和浏览器最小流程，并记录实际服务
  版本、端口和 PID。不要通过切换分支来掩盖未完成的回滚。

## 11. 当前结论

当前确认的是“进程内会话隔离严格，但服务重启/断网/会话淘汰后的恢复链路
缺失”。目前尚未实现自动握手、自动安全重绑、统一友好错误提示或周期拼接草稿
持久化；本单中的 P0/P1 仅为待实施目标和验收标准。
