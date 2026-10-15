# DRX–ASB–static RX campaign charter

Status: standing scientific mission. This document is the user-facing mission,
not an assertion that any goal is already achieved.

## Mission

Develop one physically defensible simulation framework for dislocation
evolution, recrystallization, mechanics, and heat. It must explain dynamic
recrystallization (DRX), adiabatic shear localization (ASB), and static
recrystallization (SRX) from a retained deformed state, including
rate/temperature-dependent competition and switching. Produce and discriminate
physical models; do not merely produce grain-like pictures or passing numerical
checks.

## Overarching standing requirement

**C00 — One-law DRX/ASB response family and dynamic coexistence.** One
immutable Arrhenius Peierls--Taylor--Hall--Petch framework, including the
EXP-floor flow law and one documented physical parameterization within each
declared material/phase, geometry, thermal environment, and processing window,
must endogenously produce a lower-rate DRX-favored regime, a higher-rate
ASB-favored regime, and an overlap regime in which physically identified
recrystallized material develops inside an actively deforming shear band. The
response must follow from local kinetics, structural evolution, mechanics, and
heat transport. Rate-dependent mechanism selectors and case-by-case parameter
fits are forbidden.

## Stable goal identifiers

| ID | Permanent goal | Evidence needed |
|---|---|---|
| G01 | One common physical framework | Mechanics, dislocation content, lattice orientation, boundary motion, recovery, and heat evolve through consistent state/energy/kinetic laws. Dynamic and static protocols do not silently substitute constitutive models. |
| G02 | Existing-boundary DRX | Sustained consumption of deformed material by a lower-defect region during deformation, with physical boundary displacement, processed-material lineage, defect reduction, and evolving growth/recession competition. Raw grain-label counts or accepted front calls are insufficient. |
| G03 | Physically qualified ASB | Thermally coupled, spatially resolved shear localization with an appropriate matched causal control, relevant history/persistence, source-resolved heat and power, and numerical uncertainty small enough for the claim. A scalar stress peak is insufficient. |
| G04 | Fixed-law rate/temperature competition, switching, and spatial coexistence | Within a declared material/phase, geometry, thermal environment, and processing window, one immutable Arrhenius deformation framework and one documented physical parameterization predict a lower-rate DRX-favored regime, a higher-rate ASB-favored regime, and an overlap regime where physically identified recrystallized material develops inside actively deforming shear bands. Explain each response through evolving local state and competing physical timescales. Evaluate both mechanisms independently at every condition. A rate-based selector, per-case fit, or static RX substituted for dynamic coexistence is not acceptable. |
| G05 | Endogenous initiation of a recrystallizing region | A physically accessible defect/subgrain process creates or develops a viable growing region without a hand-inserted strain-free grain or an arbitrary nucleation double-well. Distinguish development of an inherited subgrain from formation out of a less organized defect structure. Grain count need not increase in a coalescence route. |
| G06 | Static RX from retained deformation | After a documented unload/hold/anneal protocol, inherited dislocations/subgrains supply the driving state for initiation and growth under the same physical laws, with no renewed deformation requirement and no artificial reset of density, orientation, boundary content, or history. |

These goals cannot be replaced by easier proxies. Additional passing tests, a
new checkpoint, a favorable local derivative, a stress maximum, or a
successfully archived run may support a goal but do not fulfill it by
themselves.

## Fixed-law coexistence constraint

The common constitutive law retains Arrhenius EXP-floor kinetics, Peierls
resistance, Taylor forest resistance, distinct Hall--Petch boundary-nucleation
and pile-up processes, activation entropy, physical site measures, event strain
or swept area, and hazard/residence-time rules. A loading rate may alter the
local trajectory and exposure time; it may not enable or disable a mechanism,
select another parameter set, or directly assign a response label. Both DRX
and ASB observables must be evaluated independently at every rate and
temperature.

Regime selection must be explained using deformation, diffusion, migration,
formation, recovery, and heat-transport timescales together with the evolving
local stress, temperature, signed dislocation content, boundary structure, and
stored energy. Rate comparisons use a fixed thermal-boundary law and common
strain landmarks unless the declared scientific question explicitly changes
the processing history.

Dynamic coexistence requires recrystallized-material lineage, defect
processing, and boundary motion inside a simultaneously active, spatially
qualified shear band. Post-load or static growth of a retained state is SRX
evidence and cannot satisfy this requirement. A band that persists under a
strong bath at an endpoint shows that heat is not necessary for that endpoint;
it does not show that heat was irrelevant to onset or history.

## What may evolve

Constitutive hypotheses, numerical algorithms, parameter uncertainty ranges,
operational classifiers, and candidate mechanisms may be revised when evidence
warrants. Version each revision, state its reason, and preserve earlier
source/results/classifications. A historical threshold is not a material law;
neither may it be lowered retrospectively merely to relabel an unsuccessful run.

Keep physical parameter exploration separate from numerical refinement.
Compare rate/temperature trends within a fixed hypothesis; compare different
hypotheses explicitly. A calibration shared by multiple protocols is
permissible. Case-by-case fitting to manufacture the desired switch is not.

## Physical standards

Every candidate identifies its state variables, complete energy, accessible
motions/reactions, rate law, parameter provenance, and discriminating
observation. Enforce conservation of matter and Burgers content with declared
sources/sinks, compatible kinematics, and correct work/heat accounting.
Dislocation line length may change by physical generation, stretching,
annihilation, or export; ledger those changes rather than imposing an
unphysical universal line-length conservation.

Do not double-count a boundary's constituent-dislocation energy as both bulk
line energy and intrinsic boundary energy. Do not convert unsigned absorption
automatically into chosen lattice rotation. A representation-only grain-label
handoff may not reset the physical state or add unpriced interface energy.

Differentiate thermodynamic instability, driven kinetic instability, activated
structural rearrangement, recovery, selective growth, and representation
changes. Deterministic dissipative evolution and stochastic activated paths
require different appropriate thermodynamic checks; energy conservation
applies to both.

## Agentic operating principle

Scientific non-observation selects the next experiment. A hard-invalid result
blocks use of that result, not all research. Resolve or isolate its cause while
independent physical work continues. An unavailable parameter calls for a
documented uncertainty study, not a hidden fitted value or immediate campaign
termination.

Use small analytical/discrete/continuum calculations to discriminate physics
and then carry credible mechanisms into the common production framework. Do
not let either endless infrastructure work or permanently disconnected toy
models become the campaign.

## Persistent state across cycles

Maintain one compact `campaign_state.json` (or equivalent existing file)
containing the six goal statuses, evidence/source references, currently active
hypotheses, unresolved questions, resource ownership, and the next executable
action. Every task links to one or more G identifiers and states what physical
uncertainty it resolves.

Maintain one concise append-only hypothesis/decision record with: prediction,
experiment, result, limitations, and next branch. Integrate these into existing
campaign infrastructure where possible. Do not create another manager
hierarchy solely for bookkeeping.

At every handoff or context reset, read this charter and the current state
first. State which physical capability or conclusion changed. Preserve
completed milestones and negative findings; do not repeatedly rediscover them
from the entire historical archive.

## Execution authority

This charter persists across allocations but does not authorize unlimited
compute. Each activation specifies its duration and resource ceiling. Within an
active allocation, ordinary intermediate successes, failures, commits, or
completed individual runs do not require renewed approval. At the deadline,
checkpoint safely and leave an executable continuation. Do not imply that a
shell monitor performs open-ended scientific reasoning when no reasoning agent
is active.

### Executable continuation rule

During an authorized work cycle, execute a useful in-scope next task rather
than merely recording it. The operating loop is research, derive, implement,
test, interpret, select the next discriminating experiment, and execute it. A
valid negative result selects the next experiment; a passing test or commit
selects the physical integration test. Failed conservation, kinematic, work,
or heat invariants block acceptance of the affected result but trigger
diagnosis and repair rather than a documentation-only stop. After reconnection,
recover the compact state and resume its next executable action without asking
for renewed permission for the same authorized scope.
