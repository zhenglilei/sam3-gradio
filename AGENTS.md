# AGENTS.md

Project-specific instructions for Codex when working on this SAM3 Gradio demo.

- This project must be edited under `/data/zhengqiyuan/sam3-gradio` on the remote server.
- Do not modify `/data/zhenglilei/sam3-gradio` or any model/source files under `/data/zhenglilei` for this project.
- Run Python, CUDA, demo, and validation commands on the remote Linux host, not against a Windows-mounted copy.
- Keep changes surgical and verify with at least a syntax/import-level check before reporting completion.

## Task Ownership and Delegation

- Use `luna-worker` for bounded subtasks. This role is configured as
  `gpt-5.6-luna` with reasoning effort `max` (the user's "luna-max").
  Do not silently substitute another model if the role is unavailable.
- Before delegating, identify the main agent's next critical-path action.
  Keep tightly coupled design, ambiguous requirements, integration, and urgent
  blocking work with the main agent. Delegate useful independent work that can
  progress alongside it; do not delegate trivial edits just to create an agent.
- Suitable subtasks include a focused regression test, an isolated UI component,
  a scoped sample analysis, or a read-only review with explicit acceptance criteria.
- Every assignment must specify the remote host, exact worktree, input files,
  allowed write set, forbidden operations, expected output, and verification
  commands. Give only the context needed for that task.
- Agents share remote files. Use disjoint write sets, preserve others' changes,
  and never assume a subagent has an isolated remote checkout.
- Subagents must not commit, push, change branches, install shared dependencies,
  or stop/restart services unless explicitly authorized for that operation.
- After dispatch, the main agent continues non-overlapping work. Reuse an agent
  for related follow-ups; wait only when its result is needed for integration.
  Do not duplicate its analysis or repeatedly poll unchanged status.
- Give substantial subtasks a check-in budget. On stalled progress or repeated
  failure, request the evidence and blocker, then re-scope or stop the agent
  before taking over its write set. Do not wait indefinitely or duplicate edits.
- Subagent delivery must list changed files, commands run, actual results,
  limitations, and remaining work. Report blocked/incomplete work honestly.
- The main agent reviews the actual diff, reruns focused acceptance checks,
  integrates changes, and owns final regression testing and delivery.

## Remote Scope and Service Safety

- Verify the actual branch, HEAD, dirty files, and applicable AGENTS.md before
  editing. Work only in the selected worktree; do not modify sibling worktrees.
- For EL_Mask tasks, use
  `/data/zhengqiyuan/sam3-gradio/.runtime/codex-worktrees/sam3-el-mask`.
  Do not modify PVS-demo merely because it is a related or newer branch.
- Before restarting any service, verify the requested host/port, listener PID,
  process working directory, interpreter, and launch command. Never identify
  a service from its port number alone or stop unrelated listeners.
- Preserve the deployment's model paths, local custom-component imports, and
  Python/CUDA environment. Do not reinstall or upgrade shared dependencies
  as an incidental part of a UI or algorithm change.
- Preserve existing tracked changes and untracked artifacts. No blanket reset,
  cleanup, commit, or synchronization of unrelated files.

## Verification and Completion

- UI changes require real browser interaction and screenshots, not only HTTP
  200 or a config snapshot. Exercise the changed controls and primary workflow.
  Check the user's viewport and a narrow viewport for clipping and overflow;
  confirm Gradio's rendered wrappers, not only the intended CSS structure.
- Algorithm changes require reproducible real-sample before/after results and
  focused regressions. Distinguish fallback output, visual continuity, similarity
  scores, and independently verified physical registration.
- Scale regression tests to risk. Update snapshots only after reviewing their
  semantic changes; never update them simply to suppress an unexpected failure.
- Report test counts accurately, including skips/failures. Distinguish files
  edited, commits made/pushed, service restarted, and live behavior verified.
- Final reports must state what remains unverified. Do not claim a completed
  independent analysis when the agent only confirmed tools or input availability.
