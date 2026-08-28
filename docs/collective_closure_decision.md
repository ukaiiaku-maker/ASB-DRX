# Continuous collective-closure decision

## Evidence boundary

The complete reduced single-glider histories show increasing clustering with
density: inter-event-step CV rises from 0.996 to 1.080 and the multi-hit event
fraction from 0.00918 to 0.0981 between the audited low- and high-density cases.
Those are simulated structural observations, not continuum parameter data. The
native persistent-contact audit is more causal but too sparse: only the highest
density case has next-audit survivor comparisons (11 samples from 192 releases),
and its one-step centered branching proxy is zero. Thus no event-transfer
amplitude, memory time, spatial kernel, or contact rearm law is identifiable.

This caution agrees with primary research. Dislocation activity can exhibit
temporal triggering attributed to stress redistribution
([Weiss and Miguel, 2003](https://arxiv.org/abs/cond-mat/0309277)), while
interacting dislocations below depinning show enhanced intermittency and a
modified correlation length
([Li, Picu, and Weiss, 2010](https://doi.org/10.1103/PhysRevE.82.022107)).
Simulations of thermally activated glide through obstacle fields also report a
smooth-to-jerky correlated transition whose threshold depends on stress,
temperature, and obstacle statistics—not density alone
([Xu and Picu, 2018](https://arxiv.org/abs/1807.09893)). Glissile junctions have
a dual role as mobile carriers and hardening structures, so “transparent” does
not imply mechanically irrelevant
([Wang et al., 2024](https://doi.org/10.1016/j.actamat.2024.119748)).

## Compared continuous representations

Three zero-dimensional ablations now have explicit limiting tests:

1. A sequential-hit/Erlang renewal chain. Its one-hit limit is Poisson and its
   mean completion rate can be held equal to the independent rate, but its wait
   CV is `1/sqrt(m)`. It becomes less dispersed as hit order increases, opposite
   to the audited high-density CV above one. It is rejected as a stand-alone
   explanation of the observed clustering.
2. A contact activation/rearm process. A finite transparent/rearm interval
   saturates the completion flux and also gives CV at or below one. It is useful
   bookkeeping for a measured contact residence/reset time but cannot generate
   clustering by itself.
3. Exponential-memory shot-noise self-excitation. This is the Markovian
   continuous state corresponding to an exponential Hawkes kernel. Its
   zero-kick limit is exactly the independent process, and its branching ratio
   and stability threshold are explicit. Hawkes and Oakes established the
   cluster/branching representation of self-exciting processes
   ([1974](https://doi.org/10.2307/3212693)). This is the only tested alternative
   capable of overdispersion and event clustering, but the required signed
   transfer, memory time, and spatial kernel are precisely the quantities absent
   from the current native audit.

## Decision

No collective closure is promoted into the production constitutive equations.
The independent net EXP-floor law remains active. The shot-noise representation
is retained as the preferred future ablation because it has an exact independent
limit, a continuous state, and a falsifiable stability threshold. It may be
coupled only after higher-cadence histories establish causal parentage, signed
stress transfer, relaxation/rearm time, and spatial support under held-out
conditions. It must modify kinetics or energy partition continuously and must
never be wired directly to a grain-birth Boolean.

This negative promotion decision is itself a result: adding a convenient
multi-hit multiplier now would manufacture the requested boundary from
unidentified parameters and would violate the evidence constraint.
