# SESSION-COOKIE-001: signed cookie owner replaces IP authorization

- Priority: P1
- Scope: shared session identity and transport guard for PVS-demo and EL_Mask
- Overall status: OPEN; one branch passing does not close this bug
- Current stage: shared implementation is under test on `codex/bugfix-session-cookie`

## Problem

The existing registry binds a Gradio `session_hash` to the client IP. A proxy,
VPN, or NAT route change can therefore reject the same browser, while different
browsers behind one address do not have a durable owner boundary. IP remains
useful diagnostic data, but it must not authorize access.

This bug is separate from `session-restart-recovery.md`. That bug handles stale
browser state after restart and safe draft recovery. SESSION-COOKIE-001 changes
the identity used by that recovery path; it does not add full workspace
persistence.

## Required contract

1. A Starlette `SessionMiddleware` cookie contains only a random owner id of at
   least 32 random bytes, schema, and deployment id. It is signed, not encrypted.
2. The cookie is deployment-specific, host-only, `HttpOnly`, `SameSite=Lax`,
   `Path=/`, and expires after 24 hours. HTTP deployment must explicitly opt out
   of `Secure`; a stolen HTTP cookie can still be replayed.
3. Each deployment has a different persistent secret outside the release tree,
   mode `0600`. A missing, short, or permissive secret fails startup; startup
   never silently generates one.
4. Authorization uses `(owner_id, session_hash)`. `client_ip` is retained only
   as a redacted diagnostic field. Trusted-proxy parsing does not grant access.
5. Gradio state remains keyed by `session_hash`, but every business, queue,
   stream, event-query, and cancel request must validate the hash-to-owner claim
   before reading or changing state.
6. Missing, invalid, expired, or cross-deployment cookies are rejected on
   business routes. Only an ordinary first `GET /` may issue an owner cookie.
7. State schema and resume namespace are upgraded. Old IP-derived state is not
   automatically claimed or deleted. Resume ids derive from the persistent
   secret, deployment, owner, and page hash.
8. Writes require an exact configured `Origin`; when Origin is absent, an exact
   Referer origin is required. Requests with neither are rejected.
9. Logs contain error codes and digests only, never complete cookies, tokens, or
   key material. Multipart uploads and SSE are not fully buffered by the guard.

## Current evidence

| Branch/stage | Commit | Tests | Deployment |
|---|---|---|---|
| Shared identity layer | `e2f0976` | 63 shared session tests and 8 real FastAPI/Gradio HTTP tests pass with the PVS adapter | not deployed |
| PVS-demo integration | `b5f21f1` plus `b41271d` | 428 total: 427 passed, 1 opt-in paid VLM smoke skipped | not deployed |
| EL_Mask integration | `4a4ca28` plus `923eadd` | 566 total: 565 passed, 1 opt-in paid VLM smoke skipped | not deployed |

The HTTP suite covers cookie issuance and rejection, IP change for the same
owner/hash, different-owner denial before registry mutation, queue SSE, event
query, cancel, exact Origin/Referer, and a guard-decorated callback. The branch
## Browser acceptance matrix

| Check | Actual result |
|---|---|
| PVS and EL normal viewport | Edge rendered both pages; expected workflow tabs were visible and selectable |
| Narrow viewport | PVS and EL rendered in the requested 390 x 844 viewport without observed overlap (reported CSS viewport 355 x 767) |
| Same-browser tabs and isolated browser context | Two Edge PVS tabs and one isolated IAB PVS page loaded; HttpOnly owner values and per-tab hashes were not read by the UI tool |
| Upload and business handoffs | Not accepted: the documented file-chooser operation did not return, so no upload-dependent UI workflow is claimed |
| 401/403 user message and valid-state retention | Covered by HTTP tests for status/state safety; browser presentation remains pending |
| IP change | Controlled HTTP client-IP change passed; no real VPN or network switch was authorized or performed |
| Model chain | Not run; model remained unloaded and model behavior is outside this bug |

Screenshots were returned inline by the browser tool, which did not expose a
persistent screenshot path. The bug remains OPEN pending the unaccepted browser
rows above and an explicitly approved deployment.

## Boundaries

- Keep HTTP for this release; HTTPS, login accounts, and complete workspace
  persistence are not part of this bug.
- Existing random `public_download` links are not owner-private ACLs. This bug
  does not claim to make every download private.
- Model/CUDA/PyTorch/Gradio and the shared Python environment are unchanged.
- PVS and EL use distinct cookie names, deployment ids, and secret files.

## Deployment gate

Before release, identify each service by port, PID, cwd, interpreter, and full
command, then provision its private persistent key and exact public origin.
Uvicorn must run one worker without reload. Restarting 7890, 7891, 7893 or any
Aliyun service requires separate explicit approval.
