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
| PVS baseline on bugfix worktree | pending integration commit | full discovery: 428 passed, 1 opt-in paid VLM smoke skipped | not deployed |
| PVS-demo integration | pending | dirty target worktree not yet touched | not deployed |
| EL_Mask integration | pending; shared commit must be cherry-picked with `-x` | EL-specific regression pending | not deployed |

The HTTP suite covers cookie issuance and rejection, IP change for the same
owner/hash, different-owner denial before registry mutation, queue SSE, event
query, cancel, exact Origin/Referer, and a guard-decorated callback. The branch
still requires PVS integration, EL integration and regression, and real browser
two-owner/two-tab acceptance before closure.

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
