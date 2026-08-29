# DD organization-observable audit

Updated: 2026-08-29. Scope is read-only inspection of completed archived
single-glider outputs. Files from currently running DD jobs are excluded until
they reach a terminal state and acquire immutable checksums.

## Question

Can the existing DD output constrain a continuum source for junction storage,
wall organization, lattice rotation, or a Burgers-compatible boundary?

## Located observables

The reduced EXP-floor histories record condition metadata, event time/step,
contact and forest IDs, mobile-line ID, contact position, neighbor IDs,
load-bearing arm lengths, force and local stress measures, barrier/rate/hazard,
swept area, aggregate event counts, waiting-time statistics, Fano statistics,
and same-step/window clustering proxies.

The native ExaDiS persistent-contact outputs additionally record changing
contact geometry and sparse before/after force redistribution. The completed
audit used previously has only 11 next-audit survivor comparisons for 192
releases in its highest-density case; the lower-density cases provide none.

## Availability matrix

| Required organization observable | Reduced single glider | Completed native contact audit | Usable for a wall law? |
|---|---|---|---|
| event time and waiting distribution | yes | yes | event kinetics only |
| contact/event spatial coordinate | one glider coordinate | contact/node geometry | insufficient for a 2-D/3-D wall structure factor |
| event amplitude or swept area | yes | yes | plastic intermittency proxy |
| Burgers family and sign of every participant | no resolved reaction inventory | network geometry exists but exported event audit does not provide the required participant-resolved before/after balance | no |
| junction type creation/destruction | no | collision/contact records are not a validated junction-reaction ledger | no |
| mobile/forest/GND change per event | aggregate contact counts only | no closed reservoir decomposition | no |
| Nye tensor/lattice curvature | no | no continuum coarse graining or orientation field | no |
| signed pair correlations/structure factor | no | not exported at the required cadence | no |
| wall alignment and persistence | no | no accepted wall identity | no |
| lattice orientation/plastic spin | no | no | no |
| local elastic-energy change | force-work/barrier proxies | native force work at contacts | useful locally, not a wall-energy calibration |
| causal multi-lag transfer kernel | no | audit cadence/graph churn leaves it unidentified | no |
| temperature/rate/seed coverage | several generic fixtures | limited | structural comparison only |

## Decision

The existing data support statements about contact residence, mean event rate,
waiting-time dispersion, multi-event clustering, local force work, and swept
plastic activity. They do **not** identify the sign of a multi-hit contribution
to junction storage, a wall-conversion rate, a nonlocal stress-transfer kernel,
an orientation source, or a Frank--Bilby-compatible boundary inventory.

Consequently:

1. no DD-to-wall coupling is promoted;
2. no locked collective closure can pass v2 Gate 2 from these archives;
3. the independent net EXP-floor model remains a zero-transfer baseline only;
4. any mean-preserving shot-noise closure remains an explicitly uncalibrated
   ablation and cannot create a grain or wall Boolean;
5. production Gates G/H are blocked until suitable completed evidence exists.

## Minimum targeted DD export

A future qualifying campaign must, at every reaction/event and at a registered
sub-relaxation cadence, save participant IDs, oriented Burgers vectors, line
character, position, junction type, line length, mobile/forest classification,
before/after forces and energy, changing contact graph, and a consistent volume
mapping. It must support signed pair correlations, structure factors, Nye/GND
coarse graining, wall alignment, multi-lag causal kernels, seed uncertainty,
and held-out validation. Event definitions, censoring, dead time, and all units
must be immutable in the artifact.

This is a scientific no-go, not a request to retrofit missing information from
hit order or tune it against phase-field outcomes.
