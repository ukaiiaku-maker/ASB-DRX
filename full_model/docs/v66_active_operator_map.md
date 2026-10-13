# V66 active production operator map

Audit date: 2026-10-13 UTC

Production path audited:

`run_v64_adaptive_production._context` -> `CoupledMultigrainContext` ->
`advance_complete_multigrain_interval` ->
`advance_energy_qualified_mechanics` + `advance_multigrain_front_interval`.

This map distinguishes a variable that exists in a data structure from a
mechanism that is enabled in the V65 production checkpoint and from a mechanism
that was demonstrably exercised. Diagnostic and legacy modules do not count as
production mechanisms.

| Operator or state | Represented | Enabled in V65 production | Exercised in frozen V65 endpoint | Evidence / limitation |
|---|---:|---:|---:|---|
| EXP-floor Arrhenius glide (Peierls channel) | yes | yes | yes | `common_tensorial_wall.py`: thermal glide law and accepted mechanics increments |
| Taylor resistance from forest, wall, and junction densities | yes | yes | yes | `taylor_resistance_Pa`; retained as immutable mechanics |
| Signed four-family mobile/forest/wall populations | yes | yes | yes | `CommonWallState`; populated endpoint arrays |
| Conservative signed bulk transport and Mura/Nye update | yes | yes | yes | common wall residual and mechanics ledger |
| Plastic-spin lattice rotation | yes | yes | yes | orientation rate is the antisymmetric plastic-distortion rate |
| Moving-front transmission, absorption, and neutral sink | yes | yes | yes | `PhysicalTransferLaw(0.55, 0.08, 0.02)` and six nonzero endpoint interfaces |
| Persistent interface plus/minus/junction inventory | yes | yes | yes | `InterfaceComponentState.boundary_*`; priced as interface line energy |
| Persistent residual boundary Burgers tensor | yes | yes | yes | `boundary_burgers_m1`; nonzero at the endpoint |
| Hall--Petch stationary-boundary source/transmission barrier | no | no | no | present only in the quarantined legacy v34 model, not this production call graph |
| GB/TJ source or sink independent of front sweep | no | no | no | transfer is invoked only for swept material |
| Boundary structural content -> misorientation/rotation | no | no | no | interface content does not enter either owner's orientation equation |
| Boundary reactions/redistribution independent of front sweep | no | no | no | persistent interface inventories have no reaction-rate operator |
| Selective subgrain-boundary migration from local stored-energy affinity | partial | yes for pre-existing owner fronts | not a new-network event | the front can move existing interfaces but cannot create an internal interface |
| Misorientation redistribution across a boundary network | no | no | no | no conservative structural-dislocation transfer between interfaces |
| New internal boundary creation | diagnostic only | no | no | `common_subgrain_recognition.py` recognizes a plateau and proposes neutral handoff; it does not alter production ownership |
| Static zero-rate recovery | yes | conditionally | not yet discriminated in V66 | fixed-strain holds still execute thermal recovery/reaction terms; unloading needs an explicit loading path |
| Dynamic grain/topology allocation | fixed-owner restriction | no | no | owner count and interface graph remain fixed |

## Decision

The frozen V65 endpoint demonstrates a coupled Arrhenius mechanics, heat, and
moving-front model, but it cannot produce absorption--rotation DRX. It stores
signed interface content without converting the compatible structural part into
misorientation, and it cannot redistribute that content through a subgrain
network. Likewise, it cannot test the proposed GB/TJ-only source hypothesis in
the active production architecture because boundaries are sources/sinks only
when swept by a moving front.

V66 therefore needs three separable additions, in this order:

1. an energy-complete structural-boundary state that splits compatible
   misorientation-carrying content from extrinsic stored content;
2. a conservative absorption/redistribution reaction whose orientation change
   follows the Frank--Bilby/low-angle relation rather than imposed rotation;
3. a stationary boundary/TJ Arrhenius source/sink operator with a seeded random
   bulk-source control using the same barrier family and energy ledger.

The intrinsic boundary energy, compatible structural content, and extrinsic
line energy must be declared once each. In particular, compatible content used
to determine a Read--Shockley boundary energy must not also be charged the full
extrinsic line energy.
