# V64 execution record

V64 was adopted at `2026-10-12T19:57:02Z`; its fixed local deadline is
`2026-10-13T07:57:02Z`. The directive and V63 assessment SHA-256 identities are
`6b8670783c2e209d5bda577f37f3d47be971cdfee203e2e8943d2c3eea4a0f02` and
`760c056acff749f41d7b665148b3de8831154f5230d5d0b1eec264ca13cb7f8a`.

The startup audit verified clean local/remote V63 head `bd48abe`, the V63
delivery and source archives, and all retained n64 checkpoint hashes. Step 264
is correctly stored in the separate exact-state directory. No DRX/ASB solver
was active; unrelated long-running calculations were left untouched. The
isolated V64 branch is `exp/full-v64-production-accuracy-20261012` and evidence
root is `/Users/sdillon/HPC3/campaigns/asb-drx-v64-20261012`.

The initial implementation introduces a pure complete-interval map and a
rollback-safe step-doubling controller. One complete interval is compared with
two complete half intervals from the same immutable parent and prescribed load
history. Only the finer path can be published. Accuracy is evaluated separately
for support transfer, plastic distortion, signed content, exact Nye content,
temperature, stress, instantaneous power/slip/speed, and integrated work/heat.
Existing positivity, contour-CFL, donor, and energy gates remain admissibility
checks rather than being relabeled as error estimates.

Controller state is checkpointed with the complete material/runtime state.
Rejected trials cannot publish heat, work, counters, first-passage histories,
or elapsed time. A rejected interval reuses its already-computed first half as
the exact coarse candidate at the reduced duration. Step growth is bounded and
requires two consecutive comfortably resolved accepted intervals; no formal
order is assumed across active-set changes.

