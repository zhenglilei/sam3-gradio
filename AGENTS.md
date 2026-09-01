# AGENTS.md

Project-specific instructions for Codex when working on this SAM3 Gradio demo.

- This project must be edited under `/data/zhengqiyuan/sam3-gradio` on the remote server.
- Do not modify `/data/zhenglilei/sam3-gradio` or any model/source files under `/data/zhenglilei` for this project.
- Run Python, CUDA, demo, and validation commands on the remote Linux host, not against a Windows-mounted copy.
- Keep changes surgical and verify with at least a syntax/import-level check before reporting completion.

## 任务委派与验收

- “边界清晰任务”必须同时满足：输入与输出固定、文件所有权互斥、无需跨模块设计判断、验收命令明确，且失败影响局部。
- 满足上述条件的任务优先交给 `luna-worker`；不满足时由主 Agent 负责，或先拆分为满足条件的子任务。
- 每次委派必须明确文件所有权、任务范围和验收条件；子代理不得修改所有权之外的文件。
- 子代理不得提交、push、切换分支或重启服务，除非用户明确要求。
- 主 Agent 必须最终阅读实际 diff，重新执行委派任务的验收命令，并承担集成后的完整质检与回归验证。
