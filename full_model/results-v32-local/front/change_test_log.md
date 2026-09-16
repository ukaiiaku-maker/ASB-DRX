# V32 front workstream change and test log

## Scope and immutable inputs

- V31 continuation source: `c643afe00b4ea6f7f30e021675d7648ecdb6422c`.
- V31 output root: `/Users/sdillon/HPC3/local-results/asb-drx-v31-front-continuation-20260916`.
- No V30/V31 checkpoint, status, or archived evidence file was modified.
- Equal-state and mobility-off controls were not rerun. The V32 screen records
  the checksum of the qualified V30 production-control evidence.

## V31 manager audit

The detached immutable-source resume manager remained live while
`launch_status.json` already said `COMPLETED_WITH_TERMINALS`. Case status and
the process tree, rather than that premature aggregate marker, are therefore
authoritative until all ten selected cases close. Existing non-front driver
processes were left untouched.

The manager subsequently closed all ten selected cases and corrected the
aggregate marker to `COMPLETED`. Seven cases reached their requested horizon;
three retained raw `PASSED_TERMINAL / FRONT_COMPONENT_SPLIT` classifications.

## The 64-grid +1e-6 terminal

The V31 step-135 event was not an ordinary physical split. The additional
closed contour enclosed `0.01661257561158891` cell squared and had length
`0.6651820269372192` cells while the main active-window front remained intact.
It is below one primal cell in area and four cell edges in length.

V32 excludes such an unresolved closed, non-winding loop from component
ownership but retains it in the cut-cell receiver fraction and material sweep.
The filter is therefore diagnostic, not a deletion of material. A loop becomes
explicit as soon as either its area or length is resolved. With nucleation
disabled, a resolved disconnected loop is classified
`UNAUTHORIZED_PHASE_ISLAND` and remains fail-closed.

## Ordinary reconnection and event-step discriminator

Explicit component split/merge support uses rectangular assignment: the best
matched child retains the parent identity, unmatched children receive fresh
identities, retired parents cannot be reused, and global cut-cell sweep remains
the transaction authority. Every supported event is recorded as
`SUPPORTED_FRONT_COMPONENT_SPLIT` or `SUPPORTED_FRONT_COMPONENT_MERGE`.

The production replay from V31 step 135 passed the original sub-cell event and
an ordinary split, then reached `PAIR_IDENTITY_LOST` at step 161 with a jump
from 73 to 165 diagnostic ray crossings. An exact-off topology-backtracking
control reproduces fail-closed behavior. With backtracking enabled, 25.390625%
of step 161 is admissible and conservative, but the following step has no
admissible prefix down to `2^-12`. This is unresolved diffuse fragmentation,
not a single oversized front event, so the case is not promoted.

Both 128-grid near-equal cases stopped symmetrically at step 137 in V31. Their
extra closed loops are not sub-cell artifacts: they enclose about 2.42 and
1.90 cells squared and have roughly 201-cell contour lengths, with ray counts
jumping from 117 to 319. V32 replay retains both as
`UNAUTHORIZED_PHASE_ISLAND` at step 138. Raw V31 classifications remain
unchanged in the evidence ledger.

## Bounded kinetics screen

The 32-grid three-step screen varied attempt frequency by 0.25x and 4x,
activation enthalpy from 0.30 to 0.40 eV, and transfer/storage partitioning.
Favorable and reversed directions, the center label swap, and center +/-1e-6
near-equal cases were run. All six candidates preserved direction and closure,
but all accepted the complete available phase trial. The short response is
geometry-limited and cannot identify a speed optimum. The unmodified center is
retained only as the neutral anchor; it is not promoted to a long run.

## Tests and replays

- Focused V31/V32 topology suite: 21 passed.
- Manufactured checks cover sub-cell conservation, resolved forbidden island,
  label symmetry, checkpoint serialization, supported split/merge identity,
  exact cut-cell balance, topology backtracking, and exact-off behavior.
- Replay roots:
  - `/Users/sdillon/HPC3/local-results/asb-drx-v32-front-terminal-replay3-20260916`
  - `/Users/sdillon/HPC3/local-results/asb-drx-v32-front-backtrack-one-step-20260916`
  - `/Users/sdillon/HPC3/local-results/asb-drx-v32-front-backtrack-replay-20260916`
- Kinetics screen root:
  `/Users/sdillon/HPC3/local-results/asb-drx-v32-front-kinetics-screen-20260916`.

The repository-wide suite must be invoked with `PYTHONPATH=src`; the first
plain invocation failed during collection because the source-layout package
was not on the Python path and did not execute tests.

## Common Mura/front integration audit

The nominal combined configuration is structurally unintegrated. Common Mura
forces one grain and frozen KWC evolution, while the later sparse-front
transaction reconstructs only `rp/rm` and scalar forest/wall reservoirs. It
does not atomically update signed forest/wall arrays, beta/Nye/alignment,
junction state, and the sparse-front reservoirs. A flag-only relaxation would
therefore create two owners.

Production now fails fast before mode rewrites whenever both paths are
requested. The proposed `v32_existing_boundary_common_state` contract is
declared but unavailable until one signed common-state transaction exists.
Its future contract forbids nucleation, embryos, relabeling, and every label
allocation path. Common-only and front-only modes retain exact-off behavior.
