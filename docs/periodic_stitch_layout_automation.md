# Periodic Stitching: Layout Constraints and Automation

Scope: EL_Mask worktree only; PVS-demo and port 7890 are not part of this change.

## Root cause

The initial horizontal arrangement was correct. Auto-alignment used the general
translation matcher, whose phase/correlation seeds were not constrained to the
selected acquisition direction. Repeated circles admitted a vertical shift
(-4, -226) between tiles 2 and 3 in a horizontal chain. Accumulating that shift
moved all later tiles into another row.

## Layout contracts

- 1xN: match successive right/left edges, positive x progression; limit both
  neighbor and whole-chain y drift.
- Nx1: match successive bottom/top edges, positive y progression; limit both
  neighbor and whole-chain x drift.
- 2x2: retain multiple edge candidates and choose a consistent four-edge loop.
- 2xN: row-major order, jointly select candidates across adjacent grid cells,
  then solve all selected edge displacements together rather than averaging
  inconsistent paths greedily.
- Never silently reorder uploads. An incorrect acquisition order is not fixed
  by changing the user's selected layout.
- This acquisition model assumes overlap no greater than half a tile and
  cross-axis drift no greater than 10% of the shorter cross dimension.
  Higher overlap, large rotation, perspective changes, or a different scanning
  pattern require a different model rather than relaxed unconstrained matching.
- Missing evidence retains the regular layout with an explicit warning. This
  is a fallback, not a claim of successful image registration.
- Close-scoring periodic aliases in chains receive an ambiguity warning;
  the score gap is a heuristic, not a calibrated probability.

## Automation assessment

Current stages are upload, optional pixel-margin crop, connected exterior-black
trim, layout initialization, automatic translation matching, interactive preview,
and export. The four sample images have a bottom white strip; explicit bottom
crop handles it without classifying bright interior pixels as removable borders.

Remaining steps needed before claiming unattended physical registration:

1. Validate acquisition order with capture metadata where available.
2. Measure candidate ambiguity and overlap coverage, not NCC alone.
3. Add rotation estimation only with independent tests for the capture geometry.
4. Use a quality gate to separate accepted registration from uncertain fallback.
5. Validate seams on raw unrepaired texture as well as inpainted repeated patterns.
6. Keep an audit of shifts, crop margins, candidate scores and fallback reasons.

Export blending softens seams but cannot repair wrong geometry. Non-blended export
uses layout-specific overlap cuts. Neither is evidence that periodic phase aliases
were resolved. A repeated pattern without unique features or capture metadata can
have multiple equally plausible translations; zero manual adjustment cannot be
guaranteed for such inputs.

## Verified on 2026-09-08

- Full discovery: 428 tests run, 427 passed, 1 designed skip.
- Syntax checks passed for both matching modules.
- Live EL_Mask port 7891 restarted; PID 3735355, expected worktree/interpreter.
- Port 7890 retained PID 3686240.
- Browser: reverse upload order, bottom crop 8, horizontal automatic alignment
  produced a single row and exported 2006x583 without manual adjustment.
- Browser: vertical mode retained a single column.
- Browser: two-row mode with four real tiles retained a grid, loop residual 3px.
- Six-tile 2x3, odd five-tile grids, low-texture fallback and cross-cell candidate
  selection were verified by backend regression tests, not six-file browser QA.
- Changes remain uncommitted and unpushed.
- Source-focused whitespace checks pass. Full-worktree diff check reports
  trailing whitespace in existing/generated custom-component .pyi files;
  those unrelated generated edits were preserved.
- Physical ground-truth registration, large-angle rotation, perspective changes,
  and arbitrary upload-order recovery remain unverified.

## Verification approach

Use known-position random-texture crops for horizontal, vertical and two-row grids;
test low-texture inputs, empty/single input, and conflicting candidate paths.
Run the four supplied BMP files in both original and reversed upload order.
Verify the actual 7891 browser upload -> alignment -> export path after restart,
and check that 7890 retains its original process.
