# PVS-demo TODO

## P1 SESSION-COOKIE-001

- [ ] Replace IP authorization with a signed deployment-specific cookie owner.
  - Shared identity is committed once as `b5f21f1` on `Zhengqiyuan/PVS-demo`;
    `EL_Mask` inherits it through the PVS branch history.
  - EL regression: 567 total, 566 passed and 1 opt-in paid VLM smoke skipped;
    EL-specific coverage is integrated as `714f1db`.
  - PVS regression: 428 total, 427 passed and 1 opt-in paid VLM smoke skipped;
    integrated as `b5f21f1` plus `b41271d`.
  - Browser acceptance remains partial because upload-dependent UI workflows
    and browser error presentation are not accepted.
  - Deployment pending; no formal service has been restarted.
  - Specification: `docs/bugs/session-cookie-owner.md`.

后续计划记录在 `Zhengqiyuan/PVS-demo` worktree 中，优先围绕 PVS 标注质量反馈扩展和版图 prompt 能力扩展。

- [ ] 扩展版图级 feedback 数据集整理
  - 当前首版 feedback 已支持 PCS 结果池和 active PVS instance 的好 / 及格 / 差质量记录。
  - 后续可扩展为整张版图、多实例批量反馈和导出后复核反馈，用于构建更完整的 RL / 偏好优化数据集。

- [ ] 增加版图导入作为 polygon prompt 的功能
  - 支持导入外部版图。
  - 版图在工作台中支持旋转、缩放、移动。
  - 将调整后的版图轮廓转换为 polygon / mask prompt，用于创建或精修 PVS 实例。
