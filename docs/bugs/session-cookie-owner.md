# Browser sessions without a deployment signing key

## Runtime contract

Only GET `/` can establish a browser identity. The Cookie contains a 256-bit
cryptographically random bearer token; a bounded in-process registry stores its
SHA-256 digest, an independent random owner id, deployment id and activity time.
No Cookie payload, signing middleware, persistent signing key or key mount is used.

The Cookie is host-only, HttpOnly, SameSite=Lax, Path=/ and has a 24-hour sliding
idle lifetime. Successful authenticated requests (including UI polling) refresh
both server activity and Cookie Max-Age. This measures network activity, not mouse
activity. Use distinct Cookie names and registries per service: Cookies have no
port isolation. Exact Origin/Referer checks still apply to writes. HTTP requires
both `SAM3_SESSION_COOKIE_SECURE=0` and `SAM3_ALLOW_INSECURE_COOKIE=1`.

Required configuration:

- `SAM3_SESSION_DEPLOYMENT_ID`: unique deployment identifier.
- `SAM3_SESSION_COOKIE_NAME`: unique Cookie name for the service.
- `SAM3_ALLOWED_ORIGINS`: comma-separated exact browser-facing origins.
- `SAM3_SESSION_COOKIE_SECURE`: `1` for HTTPS; explicit opt-in above for HTTP.

`SAM3_SESSION_SECRET_FILE` is retired. If inherited, its presence produces one
warning per process. Its value is not logged, and its file is never opened.
The old signing dependency and deployment instructions have been retired.

## Ownership and cleanup

Keep `(owner_id, session_hash)` and queue `event_id` checks on callbacks, state,
SSE and cancellation. Uploads and file routes also require a registered Cookie;
Gradio retains its file access controls. Public export links retain their existing
sharing behavior; a random public export URL is not a private per-user ACL.

The credential limit equals the claim limit, `max(session_capacity * 4, 128)`.
Only expired credentials proven inactive are reclaimed; full capacity returns
503 `SESSION_CREDENTIAL_CAPACITY` with Retry-After. Active credentials are never
evicted to admit a new browser. Each in-flight request pins its credential through
completion, error or disconnect. Ownership probes retain queued/running work;
unknown liveness fails closed. Token and owner ids are never reassigned.
An anonymous visitor can occupy a slot until expiration: this bounded registry
is not an ingress rate limiter or an account quota system.

State validation and logging each generate their own independent in-memory HMAC
material. Neither uses deployment configuration or writes that material to disk.
Logs use process-scoped pseudonyms and never raw credentials. These protections
remain production code. Application startup uses one worker and no auto-reload.

## Restart and verification

Restart invalidates every previous Cookie and internal state proof. Business
requests receive 401 `SESSION_COOKIE_REQUIRED` with a reopen-homepage message;
GET `/` creates a new identity. Unsaved work and cross-restart draft recovery are
not promised. Same-process TTL recovery continues to validate owner/generation.

Regression coverage lives in `tests/test_session_credentials.py`, session HTTP,
HTTP lifecycle, runtime, guard, recovery and observability suites. Test sources
are maintained in Git but excluded from production images and deployment ZIPs.
PVS is the shared integration base; deployment acceptance runs on EL and overlap.
