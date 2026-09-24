# V53 closure addendum

## Attained calculations

The n128 loading history completed interval 128 at 62.5 microseconds.  Its
endpoint engineering shear is 0.0325, engineering plastic shear is
0.006710348524342243, mean shear stress is 2.297857946 GPa, and mean
temperature is 1104.347474899 K.  All available incremental and cumulative
first-law checks pass.  The observed coarse stress maximum is interval 107;
the exact peak location was not refined and is not attributed to DRX or ASB.

The n192 companion completed interval 52 at 25.390625 microseconds from the
verified interval-32 prefix.  Its post-32 accepted segment wall time is
23257.273 seconds.  Checkpoint SHA-256 is
`0c009a7b1b7e38b56b0050a23e6a53ff43bf354ec2b49d3251e56e0eba59dd53`.
Every interval filled its requested clock and the interval-52 first-law check
passes.

The controller then failed only because it sought the retained n128
interval-52 checkpoint in the V52 continuation directory, whose physical files
begin after the inherited V49 prefix.  The correct V49 checkpoint has SHA-256
`6c38c6a4ac4cad833b10e1420c7f8b734e622c8a1f6cf5dbb7fe55d51feb9470`.
Using that checkpoint with the attributable V52 full-history manifest recovers
the exact common-state comparison; all preconditions pass.

At interval 52, n128/n192 increment differences are 0.000309% for mean stress,
0.002353% for plastic shear, 0.002470% for mean temperature, and 0.000654% for
peak temperature.  The remaining common Fourier band (modes 33--63) has larger
relative differences but very small total norm: its fine-grid RMS fractions
are 0.00118% for beta-p, 0.289% for family Nye, 0.00145% for temperature rise,
and 0.556% for ordered-plus line.  Mean response is strongly supported through
interval 52; this is not a blanket full-field convergence certificate.

## Unfinished versus invalid

- The quarter-step and neighboring-window peak discrimination did not run.
  The result is `NEAR_FLOW_ATTAINED_EXACT_MICROSCOPIC_PEAK_UNRESOLVED`, not a
  failed trajectory.
- The prepared full-tensor rectangle calculation did not run.  Its mechanics
  remain focused-test qualified, not physical DRX evidence.
- The current-source ASB pair did not run.  Its state is
  `BUDGET_DEFERRED_PREPARED_NOT_EXECUTED`; this is not a negative ASB result.
- Existing-boundary migration and conservative defect processing are
  implemented but were disabled in the bulk/rectangle cohort.  Intragranular
  recognition/promotion remains unintegrated and therefore has no physical
  negative result in V53.

## Execution corrections

The mechanism runner now supports a declared exact intervention-start
checkpoint.  A short full-feedback prefix is to be generated once and both ASB
members fork from its byte-identical physical state.  Reuse checks require
source, case configuration, parent hash, checkpoint hash, accepted fields, and
record/checkpoint step agreement.  The manager checks the original campaign
clock before preflight, estimates only unfinished work after partial progress,
and retains thermal/mechanical validity-limited prefixes without converting
them into controller failures.

The V53 ASB analysis uses actual checkpoint times and strains, audits the
causal intervention, tracks periodic connected support and same-structure
overlap for at least one microsecond, and separates computational completion,
hard validity, causal comparability, candidate localization, refinement, and
strict-ASB claims.  Strict ASB remains false without grid and timestep
qualification.

## Next production mechanism work

The next heavy allocation goes first to the exact-common-state ASB causal pair.
The existing-boundary DRX task is independently specified in
`v53_existing_boundary_drx_next.json`: a resolved eight-cell interface,
35%-line lower-defect child, continued deformation, zero physical migration
pressure, conservative front processing, and staged physical horizons selected
from measured kinetics.  Required outputs are physical displacement and
transformed volume, processed/boundary line, child defect reduction, stored
energy advantage, capacity limits, and rehardening.  This is a prepared-grain
growth test, not spontaneous nucleation.

