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

## Executed accuracy cases and common-origin continuation

The late n64 step-275 suffix was executed at both 2% and 1% working
tolerances.  The 2% branch used four accepted intervals and two rejected
accuracy trials; the 1% branch used eight accepted intervals and one rejected
trial.  Both had zero physical-admissibility failures.  Their endpoint
differences were 0.000355% in temperature, 0.1058% in plastic power, 0.00202%
in exact Nye, 0.000906% in plastic distortion, 0.0000817% in supports, and
0.00164% in stress.  These are suffix comparisons with an inherited V63
prefix, not common-origin convergence certificates.

Fresh n48 and n64 analytic-origin files were generated and independently
replayed for one legacy-sized step.  Every physical array, state digest, load,
clock, and ledger was bitwise identical to the ordinary initializer.  The n48
2% production trajectory then advanced from the analytic origin.  Its
controller reduced 6.25 ns to 3.125, 1.5625, 0.78125, 0.390625, and finally
0.1953125 ns as rate-change fields became active.  No tolerance was weakened.
The retained common-origin lineage is a sequence of checksum-bound numerical
transitions because scalar front closure was repaired during execution; each
transition restarts an accepted complete state and declares unchanged physical
kernels.

The common-origin calculation subsequently crossed 0.816388 microseconds.
After exact front-product caching, a 0.30517578125 ns production trial reached
1.003573 times the declared accuracy tolerance and was rejected without
publication.  Its already-computed first half was the exact coarse candidate
for the retry; the 0.152587890625 ns fine path passed at 0.496808 times the
tolerance.  The following interval also passed and the bounded controller then
regrew to 0.190734863281 ns.  This is an on-trajectory rollback/continuation
event, rather than only a manufactured controller test.  All 19 state,
endpoint-rate, rate-change, and integral metrics remain active.
The four-grain prepared network is still hardening at this horizon.  Strict ASB
is false through adaptive interval 179: there is no stress softening, no
qualifying conjunctive episode, no matched control at the new accepted times,
and no localization-refinement certificate.  This is an early-horizon result,
not a lifetime rejection.

## Scalar front closure and exact optimizations

The complete joint front affinity has a piecewise active set.  A damped/secant
fixed point can therefore cycle even though the scalar force/rate residual is
bracketed.  Production now uses a positive safeguarded bracket, closes at the
unchanged independently re-evaluated `rtol=1e-10`, `atol=1e-12` criterion,
handles adjacent floating-point endpoints explicitly, and repeats closure if
the publication reconstruction crosses an active-set ulp.  A zero-request
point is treated as the physically stationary solution: it publishes zero
front extent, work, and heat rather than manufacturing a direction.

Two exact cost repairs were also exercised.  Parent complete energy is reused
within an immutable front-force solve, and virtual energy prices are marked
deliberately unbound so they cannot be published and need not serialize/hash
entire trial states.  Final realized transactions remain state-hash bound.
Handing a stagnant fixed-point cycle to the safeguarded solve after four rather
than eight non-improving iterations reduced one measured interval from 25.23 s
to 20.59 s.  Its largest physical-field relative L2 change was
2.50e-11, attributable to scalar root arithmetic ordering; load and the public
closure were unchanged.

## Restart, diagnostics, and physical interpretation

A source-frozen replay from adaptive checkpoint 112 through checkpoint 120
crossed the interval-118 rejected/accepted accuracy decision.  All 98 physical
arrays, the state digest, load, clocks, and runtime ledgers reproduced bitwise.
The audit exposed only an over-counted legacy serialized `step` label after
restart; current checkpoints now store the authoritative global adaptive
interval.  Postprocessing likewise indexes adaptive data by filename/controller
identity while retaining the old stored step as provenance.

Native-node frames with fixed color limits cover the common-origin history
through interval 179.  They show the prepared four-owner geometry, interface-
correlated temperature and Nye structure, and a broad high plastic-power field;
they do not show a qualified narrow persistent shear band.  The corrected V63
attribution remains projection-sensitive, while native unprojected rate replay
is exact.  The late profile quantity formerly called FWHM is recorded as a
multi-peak half-height envelope, and the component track is right-censored.
Directed feasible front probes at the V63 DRX state are downhill for 11 of 12
directions and for both senses on five of six interfaces, with virtual-slope
variation below 0.04%.  This supports complete-functional prepared-boundary
redistribution, not spontaneous grain birth or a material-specific DRX claim.

The current local solver is the only campaign-heavy process.  It is continuing
the n48 common-origin trajectory toward 1.25 microseconds under source
`b579a16`; checkpoints are retained every eight accepted intervals.  Exact
request-invariant front-product caching reduced a representative complete
proposal from 23.77 s to 15.54 s while preserving every physical array, state
digest, load, clock, and ledger bitwise.  Checkpoint 269 has SHA-256
`f73f0d9e15df9ff3f4de36e86500a3b474ee54c838c0369bdc61e8755dc56200`.
No HPC, cloud, or paid execution has been used.

At adaptive interval 318 the pointwise-only scalar solver encountered a
sign-changing active-cone residual whose bracket had collapsed to adjacent
floating-point numbers. Halving admitted one 11.92 ps interval but the next
call exhausted the 6.10 ps minimum; no invalid trial was published. The repair
treats only an adjacent-float sign bracket as a set-valued complementarity
closure, independently reconstructs that inclusion, and still requires the
unchanged pointwise tolerance away from the non-single-valued limit.

From checksum-bound checkpoint 317, the repaired complete 190.735 ps proposal
passes at 0.739912 times tolerance. Its coarse and first-half maps close
pointwise; the second half crosses the certified adjacent bracket
`[0.9999878552028691, 0.9999878552028693]` and then independently recovers a
pointwise publication value. The bracket residuals are `+1.64950e-10` and
`-1.57876e-10`. Promoted checkpoint 318 has SHA-256
`e67f4f9acc81c9d6df54c47f93aecd8a0a770a3745c25106384a5bfa05d4209b`.
Production continues under the original tolerances.
