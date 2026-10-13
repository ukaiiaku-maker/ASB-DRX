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

## Stable goal identifiers

| ID | Permanent goal | Evidence needed |
|---|---|---|
| G01 | One common physical framework | Mechanics, dislocation content, lattice orientation, boundary motion, recovery, and heat evolve through consistent state/energy/kinetic laws. Dynamic and static protocols do not silently substitute constitutive models. |
| G02 | Existing-boundary DRX | Sustained consumption of deformed material by a lower-defect region during deformation, with physical boundary displacement, processed-material lineage, defect reduction, and evolving growth/recession competition. Raw grain-label counts or accepted front calls are insufficient. |
| G03 | Physically qualified ASB | Thermally coupled, spatially resolved shear localization with an appropriate matched causal control, relevant history/persistence, source-resolved heat and power, and numerical uncertainty small enough for the claim. A scalar stress peak is insufficient. |
| G04 | Rate/temperature response and mechanism switching | A fixed documented constitutive model and parameterization predict differing DRX/ASB/recovery regimes as loading rate, temperature, and declared processing history vary. Select material-appropriate experimental trends; do not impose a universal monotonic switch. Coexistence is allowed and need not occur at every condition. |
| G05 | Endogenous initiation of a recrystallizing region | A physically accessible defect/subgrain process creates or develops a viable growing region without a hand-inserted strain-free grain or an arbitrary nucleation double-well. Distinguish development of an inherited subgrain from formation out of a less organized defect structure. Grain count need not increase in a coalescence route. |
| G06 | Static RX from retained deformation | After a documented unload/hold/anneal protocol, inherited dislocations/subgrains supply the driving state for initiation and growth under the same physical laws, with no renewed deformation requirement and no artificial reset of density, orientation, boundary content, or history. |

These goals cannot be replaced by easier proxies. Additional passing tests, a
new checkpoint, a favorable local derivative, a stress maximum, or a
successfully archived run may support a goal but do not fulfill it by
themselves.

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
