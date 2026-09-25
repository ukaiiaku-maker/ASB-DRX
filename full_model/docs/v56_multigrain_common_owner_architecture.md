# V56 multi-grain common-owner production architecture

## Exact production starting point

The qualified state is `CommonFrontState` in
`full_model/production/common_front_state.py`. It owns one
`SparseFrontState`, three fixed material histories (`parent`, `child`, and
`wake`), their optional reservoir/alignment moments, boundary inventory, an
interface Nye correction, and one complete ledger. `reconstruct_common`
support-weights those three owners and takes Curl only after reconstructing
plastic distortion, so the interface product-rule term is explicit.

`complete_front_energy.py` evaluates a copied pair candidate and publishes it
only after complete Helmholtz, dissipation, export, and first-law acceptance.
The monolithic driver stores exactly one `common_front_state`, advertises
`common_front_single_owner=True`, and performs one pair transaction before
publishing the reconstructed mixture back to plastic, Nye, temperature, and
phase fields. These are architectural limits, not configuration omissions.

The legacy multi-grain tensor path is not a valid base: it has no complete
physical energy ownership. The multi-grain extension must generalize this
pair transaction rather than route around it.

## Proposed state

Introduce a separate `MultiGrainCommonState` while retaining the pair type as
a qualified adapter and regression oracle:

- `grain_ids`: immutable persistent IDs, never constitutive signs;
- `supports[g,x,y]`: nonnegative and cellwise normalized;
- `owners[g]`: the existing `CommonWallState` plus reservoir/alignment moments
  for each physical grain;
- `history_owners`: explicit recovered-wake histories where required, keyed by
  origin and receiving grain instead of one global wake;
- `interfaces[(g,h,component_id)]`: orientation-independent stable edge IDs,
  winding/component geometry, first-passage/revisit support, boundary line and
  junction inventory, event exposure, and cumulative clock/work/heat;
- one global material-capacity ledger and one complete energy ledger.

Shared temperature may remain one reconstructed field. Extensive defect,
boundary, and history content must stay attached to its owner and may not be
copied into all grains.

## Reconstruction invariants

For every cell, `sum_g supports[g] == 1` within a representation tolerance.
Support-weighted owner reservoirs reconstruct each common extensive inventory
once. Plastic distortion is reconstructed before Curl; the difference between
the resulting Nye tensor and support-weighted owner Nye is the declared
interface product-rule contribution. Intrinsic HAGB energy and plastic excess
are charged separately and exactly once.

Checkpoint state must include supports, IDs, every owner/history reservoir,
interface identities and winding, donor capacity, first-passage/revisit maps,
event exposures, and all ledgers. A restart that lacks any authoritative item
is migration/incomplete evidence, not exact restart.

## Joint boundary transaction

At each physical step:

1. Build all incident pair proposals from one immutable pre-step state.
2. Resolve donor competition cellwise with a constrained capacity allocation.
   No two pair candidates may spend the same material or recovery inventory.
3. Recompute every accepted candidate's actual nonlinear contour sweep after
   capacity limitation; reprice energy and kinetics from that final geometry.
4. Commit geometry, owner transfer, line/Burgers content, kinematics,
   intrinsic/excess boundary energy, dissipation, heat/export, exposure, and
   elapsed time atomically.
5. Reconstruct the common state once and run complete energy, material, line,
   Burgers, Nye, and clock audits before publication.

Conservative substepping is acceptable initially. Label permutation must leave
the physical result invariant, and scheduling artifacts must decrease under
subdivision. A zero-motion fixture cannot qualify the transaction.

## Minimum execution ladder for the next allocation

1. Nontrivial three-grain initialization with normalized supports, pure cores,
   two incident boundaries, one junction, and distinct orientations/densities.
2. Pair-limit identity against the existing qualified state.
3. Simultaneous competing transfers with donor-capacity conservation,
   label-permutation tests, subdivision convergence, rollback, and exact
   restart.
4. A zero-pressure continued-deformation trajectory with plasticity, recovery,
   heat, two actual moving boundaries, and complete energy.
5. Matched front-disabled and thermal controls from the same physical origin.
6. Promotion to a small polycrystal only after the three-grain production
   trajectory passes; never substitute the legacy eight-grain route.

The first code slice should therefore be the multi-owner state/reconstruction
and pair-limit adapter together, followed immediately by a nonzero two-edge
joint transaction. A schema without that transaction is not a campaign
endpoint.
