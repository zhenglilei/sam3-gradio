# Chat keyword routing

Interpret the latest message before comparing candidates.

| User wording | Intended candidate direction |
|---|---|
| 自动分析、看看这张图 | Compare the current candidate and profile priors |
| 再填一点、填补凹坑、坑洼 | Prefer a nearby close candidate; do not substitute dilation |
| 减少粘连、分开、断开 | Prefer less close or a safer separating candidate |
| 整体加粗、加粗、变粗 | Prefer a +1 morph candidate |
| 保留孔洞、保留方孔 | Prefer close=0 and no positive morph |
| 不要加粗、不要变粗、保持线宽 | Reject the global-thickening interpretation |

The host handles `应用推荐参数`, `撤回`, and `重置` locally before any VLM request. Never claim that those commands changed or saved mask pixels.
