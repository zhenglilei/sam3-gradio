# SESSION-LIFECYCLE-002: recover coherent state and bound transport claims

- Scope: shared backend of PVS-demo, EL_Mask, and overlap-prediction.
- Status: backend fix verified on PVS; descendant integration follows the policy below. Not deployed.
- Base: `43bd1729d2ae84858637aecafc7a4451d7a75fe5`, the common ancestor of all three current branches.

## Reproduced failures

1. An interrupted JSON control request repeatedly returns `http.disconnect`;
   the body reader previously continued forever.
2. Unknown URLs could allocate permanent hash claims. Capacity exhaustion was
   incorrectly reported as an owner mismatch, and event claims had no cleanup.
3. Expired callbacks containing only thin business states did not enter recovery.
4. Recovery replaced local callback arguments but not all persisted Gradio
   States. Upload followed by crop could then mix old and new server identities.

## Recovery contract

- Cookie owner and page hash remain the authorization identity. IP is diagnostic.
- Authenticate the old HMAC owner token even if the registry record has expired;
  a resume id by itself never authorizes recovery.
- Recover a fresh state bundle under the new session's operation lease and write
  it to every registered owned Gradio State. Already recovered current states
  remain intact when another queued old request arrives. Such superseded
  requests return Gradio's all-output skip without executing their old action;
  an old delete/crop must not mutate the new workspace.
- Extension states, including the EL batch queue, reset using their component
  defaults. Discard expired in-memory image/prompt payloads instead of assigning
  them a new owner token. The existing validated stitch-draft store is the only
  persistence restoration source in this change.
- Gradio 6 supplies the owning Blocks through LocalContext. The recovery hook
  is attached to that Blocks instance, never to a process-global user list.
- Foreign owners, other page hashes, forged tokens, and conflicting identities
  continue to fail closed.

## Acceptance and Git policy

- Use fake clocks, signed synthetic cookies, and isolated Gradio/TestClient
  sessions; do not expire or cancel live user sessions for testing.
- Verify interrupted requests, claim pressure and reclamation, thin-state
  recovery, consecutive upload/crop, extension queues, IP changes, and denial
  of cross-owner/cross-page access. Existing UI config must remain unchanged.
- Make one shared bugfix commit on PVS. Before each descendant integration,
  check ancestry and skip branches that already contain the fix. Do not
  cherry-pick duplicate copies, rebase shared history, or force-push.
- Preserve unrelated tracked/untracked work. No model/environment changes,
  service restarts, remote pushes, or Aliyun deployment are included.

## Transport lifecycle

- Interrupted JSON reads stop immediately. Invalid JSON returns 400, an oversized
  body 413, and exhausted claim capacity 503 with `Retry-After: 5` instead of an
  owner-mismatch error. Missing JSON content types cannot bypass request leases.
- Only recognized Gradio transport routes allocate claims. Unmatched URLs do
  not allocate; a new claim for a failed dynamic route is released only when
  its backend state is proven unreachable.
- Hash/event claims have bounded capacities and a one-hour idle TTL. Before
  releasing an expired claim, invalidate retained Gradio states, completed
  events, and old pending messages. Never transfer a readable old state to a
  newly claiming browser. Active HTTP requests, queued/running jobs, and resumable
  iterators pin their ownership; uncertain cleanup retains the claim.
- Completed event records expire independently while their workspace stays
  active. Reclaiming them must not clear that workspace's current image/state.
- Cleanup runs on transport activity/capacity pressure, not in a new background
  thread. The Gradio 6 adapter must be rechecked when upgrading Gradio.

## Verified evidence

- PVS full runner: 502 tests, 501 passed, 1 opt-in paid VLM test skipped.
- HTTP targeted suite: 23 passed, including signed Starlette cookies and real
  Gradio StateHolder/Queue lifecycle tests.
- New recovery suite: 8 passed, including actual Gradio `process_api` upload,
  crop gesture and crop application after expiry. Synthetic images, no model.
- Syntax checks and scoped whitespace checks passed. Existing Gradio event-loop
  ResourceWarnings remain; this change does not modify dependency versions.
- Logs: `.runtime/session-lifecycle-bugfix-20260922/` under the repository root.
- Live browser/VPN switching and service deployment are not claimed by these
  automated checks. UI component definitions and configuration were unchanged.
