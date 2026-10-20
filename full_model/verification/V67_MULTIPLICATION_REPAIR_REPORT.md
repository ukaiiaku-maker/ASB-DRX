# V67 multiplication normalization and accepted-throughput repair

## Scope

This repair preserves the common Arrhenius EXP-floor Peierls--Taylor--Hall--
Petch mechanics and the Hall--Petch event measure. It changes neither a rate
switch nor an ASB-specific parameter. The strict no-multiplication model is now
a comparator rather than the primary GB-origin model.

## Recovered law

The executable v34 reference, not a dimensional guess, defines the unsigned
Kocks--Mecking storage rate on family \(s\) as

\[
  \dot\rho^{\rm raw}_{\rm store,s}
  = k_1\sqrt{\rho_{{\rm eff},s}}\,|\dot\gamma_s|,
  \qquad k_1=5.0\times10^8\ {\rm m}^{-1}.
\]

The production signed representation adds half of that rate to each forest
sign. Thus the Burgers-vector sum is unchanged while unsigned stored line rises
by the reference rate. There is no additional \(1/b\): \(k_1\),
\(\sqrt\rho\), and \(|\dot\gamma|\) already give units of
\({\rm m}^{-2}{\rm s}^{-1}\). The saved value also gives the useful
dimensionless check \(k_1b=0.124\) for \(b=2.48\times10^{-10}\) m.

The recovered obstacle argument is

\[
 \rho_{{\rm eff},s} = 0.65\rho_{{\rm forest},s}
 +0.35\sum_r\rho_{{\rm forest},r}
 +0.10\rho_{{\rm mobile},s}
 +0.35\sum_r\rho_{{\rm wall},r}
 +0.10\left|\sum_r(\rho^+_{{\rm mobile},r}
                         -\rho^-_{{\rm mobile},r})\right|.
\]

The archived diffuse GB-shell contribution is intentionally not copied: the
common-owner model already carries stationary interface line explicitly, so
including the shell again would double count boundary content as bulk line.

The accepted rate is bounded *before* integration by the available stored-work
fraction,

\[
  (\mu^+_{\rm forest,s}+\mu^-_{\rm forest,s})
  \dot\rho^+_{\rm store,s}
  \le 0.1\sum_r |\tau_r|\,|\dot\gamma_r|.
\]

Raw and accepted line increments, the active budget fraction, and signed and
absolute slip increments are accumulated over every accepted internal solver
step. Restart files serialize the law, coefficient, work fraction, obstacle
weights, and destination reservoir.

## Reference and regression evidence

| Quantity | Legacy compatibility | Recovered production | Strict comparator |
|---|---:|---:|---:|
| Schema | `legacy_v21_mobile_pair_v1` | `v34_km_forest_storage_v1` | `disabled` |
| Coefficient (m^-1) | 10 | 5.0e8 | 0 |
| Effective source density | within-family mobile + forest + wall | equation above | 0 |
| Per-sign multiplicity | 1/2 | 1/2 | 0 |
| Destination | mobile +/- | forest +/- | none |
| Stored-work fraction | legacy transport-power bound | 0.1 | 0 |
| Runtime origin | backward-compatible checkpoint fallback | explicit driver selection, then serialized | explicit driver selection |

The Burgers magnitude is 2.48e-10 m. Slip in the source law is the accepted
crystallographic family slip, not imposed macroscopic shear; no Taylor-factor
conversion is applied. Owner support is applied once when family/cell increments
are accumulated, and represented cell volume is applied once when converting
density increments to physical line.

- `full_model/reference_sources/v34/drx_var_v34_candidate_drx_asb_sweep.py`
  contains `KM_k1=5.0e8`, `stor_raw=KM_k1*sqrt(rf)*abs(gdot)`, the 0.1 stored
  work fraction, and `rho_state_store_to_forest=True`.
- Its saved run configuration does not override those defaults.
- Git history shows that coefficient 10 entered V21 as a generic verification
  default, without a material normalization derivation.
- Manufactured scalar integration converges to
  \([\sqrt{\rho_0}+k_1\gamma/2]^2\).
- Tests verify sign symmetry, exact obstacle weights, forest routing, work-cap
  activation, exact zero behavior, restart schema inheritance, and inherited
  front transmission.

The state audit is machine-readable in
`v67_multiplication_normalization_routing_audit.json`. On the saved ad400,
ad500, and ad600 states, the recovered-to-legacy accepted-line ratios are
\(6.06\times10^7\), \(6.35\times10^7\), and \(6.62\times10^7\). The
strict comparator accepts exactly zero line, and every probe consumes the full
physical interval.

## Production interpretation

The repaired model stores line in the forest reservoir; it does not create a
glissile mobile reservoir by relabeling stored content. Consequently, a
positive multiplication result can preserve or redistribute accepted flow only
through the coupled stress, slip, reaction, and boundary-transport trajectory.
It cannot manufacture carrier throughput directly.

The matched production comparison starts from the immutable ad400 common
checkpoint. It is therefore a causal law intervention on an evolved state,
not a claim that the preceding history used the repaired law. A separate replay
from the saved zero-load anchor has now been executed and is reported below.
The first completed repaired run accidentally inherited
the old driver default of 0.55 front transmission; it is retained only as a
boundary-law ablation. The driver now inherits the serialized 0.25 value, and
the accepted matched repaired/legacy/disabled set all uses 0.25.

At the matched 0.13064-shear endpoint, the repaired trajectory has accepted
\(5.00898\times10^{-6}\) m of new forest line, 98.033% of its raw proposal.
Only 0.2502% of cell-time is budget active at the maximum. The normalization
repair is therefore not suppressed by the work cap. It retains 0.18127%
accepted plastic/imposed power, versus 0.035887% for both the legacy and strict
zero paths: a factor of 5.051. Its final mean forest density is 1.559 times the
disabled comparator, while its mean mobile density is only 1.007 times larger;
the response depends on the spatial owner/family trajectory rather than on a
scalar mobile-inventory increase.

The repaired path's final connected band occurs at shear 0.12120, compared
with 0.10376 for both legacy and disabled paths. None is banded at the endpoint,
none softens after its peak stress, and all peak at the terminal load. Grain 20
still grows by 0.08104% in the repaired path and intersects active bands, but
this is existing-boundary RX/band overlap, not qualified ASB. All three paths
pass the hard absorption, Mura, first-law, and grain-identity checks.

The strict comparator accepts exactly zero multiplication. The legacy path
accepts only \(1.276\times10^{-15}\) m and is numerically indistinguishable
from disabled at the endpoint. The decision is
`NORMALIZATION_REPAIR_EXTENDS_ACCEPTED_FLOW_AND_BAND_PERSISTENCE_BUT_STORAGE_ONLY_MULTIPLICATION_STILL_ARRESTS_WITHOUT_ASB_SOFTENING`.

The production-kernel low-density fixture also proves that retained forest with
zero mobile line gives zero slip and zero multiplication. This is intentional:
the repaired storage law does not relabel all forest as an operable pinned
source. A persistent-source extension, if needed after the fresh-origin replay,
must carry its own formation, survival, operation, and destruction state.

The *evolved-state intervention* stops at its preregistered 0.13064 endpoint.
At that point the response is almost elastic and monotonically hardening near
9 GPa; extending that causal fork toward 0.20 would enter an unqualified
high-stress regime without testing an ASB feedback loop.

## Fresh-origin corrected history

The corrected history begins at the saved zero-load three-grain anchor. Its
first 33 accepted intervals use 20 s^-1 for 132 microseconds and reach shear
0.00264. One rejected macro trial is rolled back and replaced by two complete
half intervals; all accepted clocks, work, heat, source opportunity, and load
are therefore preserved. At this prefix endpoint, owner 20 has grown 1.35279%
and all hard balances pass. A matched 2 s^-1, 10-microsecond loaded continuation
then grows owner 20 by 0.08512%; its no-reset 10-microsecond static continuation
grows the same owner by 0.08262%. This establishes retained-state static RX but
also shows that this short lower-rate loaded window is dominated by the same
stored-energy/front kinetics rather than by deformation-enhanced dynamic RX.

The high-rate branch applies the single mean-zero 19.375 K, 0.6-micrometre
thermal strip once, then advances at 80000 s^-1 with 2 ns accepted intervals.
Six checkpoint-linked 200-interval segments and one 34-interval terminal
segment reach the preregistered shear 0.20008. Their checkpoint hashes and all
physics-bearing configuration values match at the seams; only the parent path,
segment length, and one-time seed metadata differ. Across this fresh high-rate
history:

- accepted multiplication is 8.42362e-5 m, 83.5617% of the raw proposal;
- accumulated domain-mean absolute family slip is 0.191611;
- endpoint plastic/imposed power is 0.226051, rather than the 0.0018127 of the
  evolved-state intervention;
- endpoint mean stress is 7.31180 GPa, mean temperature is 1106.69 K, and the
  temperature range is 78.042 K;
- endpoint mean mobile and forest densities are 1.06426e14 and 8.02520e15
  m^-2, respectively; and
- owner 20 grows 0.22258% during the high-rate portion, but none of its growth
  lies inside a qualified band because no connected physical band occurs.

The production path develops first a compact hot spot and then a broad
horizontal high-power layer. At its closest approach to the fixed topology
criterion, the largest above-2x-mean component spans half the domain but
contains only 30 cells, below the preregistered 32-cell minimum. It later
fragments. By shear 0.20008 the reconstructed continuous power field has
inverse participation 0.89448, 3.29 and 2.99 micrometre Gaussian FWHM axes in
the 4.8 micrometre domain, and power--temperature correlation -0.0270. The
fixed 2x-mean topology contains only a four-cell component. This is a broad
response, not a thresholded band hidden by the 32-cell cutoff. The endpoint is
monotonic-hardening with no softening. Thus the complete fresh result is
classified
`FRESH_REPAIRED_STORAGE_MULTIPLICATION_REMAINS_FLOWING_WITHOUT_QUALIFIED_TERMINAL_ASB`.
It is a successful carrier-throughput repair and an ASB negative through the
registered 0.20 horizon, not a combined DRX-inside-ASB result.

The significant accepted flow rules out a persistent pinned-source extension
as the immediate next repair: storage-only multiplication no longer starves on
this trajectory. The next discriminator changes only the orientation of the
same mean-zero thermal strip. The y-normal strip intersects the inherited
horizontal grain-boundary geometry, whereas the completed x-normal strip did
not. This matched test probes localization/front geometry without fitting
multiplication, transmission, source density, or any rate-dependent switch.
