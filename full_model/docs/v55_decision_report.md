# V55 independent full-model completion report

## Decision

V55 produces a functioning, restartable **existing-boundary DRX model coupled to the common Mura/plastic/thermal state**.  It does not yet produce strict ASB under the inherited criteria.  The scientifically correct combined classification is:

`VALID_GENERIC_EXISTING_BOUNDARY_DRX_WITH_COUPLED_THERMOMECHANICAL_RESPONSE; STRICT_ASB_NOT_OBSERVED`

This is a generic kinetic hypothesis, not a material calibration.  Nucleation and direct grain-label allocation were disabled, external front pressure was zero, and the DD collective/multi-hit extension remained disabled.

## Production repairs

Four localized production defects were found and repaired rather than bypassed:

1. The restart guard now uses the same 256-epsilon representation invariant as its producing common-state map.
2. The rate-complete phase envelope is confined to the declared active boundary, so it cannot move an unowned periodic interface.
3. Topological swept area is distributed over nonsaturated diffuse material capacity.  This removes the one-cell rasterization deadlock while retaining topology as the authoritative signed area and excluding width relaxation.
4. Post-front GND/GB increments are committed to phase owners at the end of the same physical step.  The segmented and uninterrupted step-60 trajectories are now bitwise identical in all 17 audited fields and five ledger/metadata records.

The nominal polycrystal runner also silently collapsed `poly_n=8` to one grain because common-Mura qualification mode is intentionally single-grain outside the integrated front.  V55 made the multi-grain reference path explicit and recorded it in the initialization hash.  The rejected one-grain run remains preserved.

## Existing-boundary DRX qualification

The zero-pressure high-mobility hypothesis (`10^12 s^-1`) reaches a topology-defined endpoint when the tracked interface leaves its active window:

| Grid | Terminal step | Applied strain | Net transformed material | Mean / peak T | Terminal reason |
|---:|---:|---:|---:|---:|---|
| 32 | 102 | 1.03% | 28.79% | 942.4 / 944.9 K | `PAIR_LEFT_ACTIVE_WINDOW` |
| 64 | 110 | 1.11% | 30.08% | 944.9 / 948.2 K | `PAIR_LEFT_ACTIVE_WINDOW` |
| 128 | 118 | 1.19% | 31.03% | 946.9 / 951.1 K | `PAIR_LEFT_ACTIVE_WINDOW` |

Successive transformed-fraction differences are 4.28% and 3.16%, below the provisional 5% spatial criterion.  At n64 the front-disabled control has zero sweep and zero transformed fraction on the same clock.  The enabled run processes 5.968e-3 m of line and sweeps 1.492e-20 m3 before its topology endpoint.  The maximum complete-transaction first-law residual is 1.35e-26 J.  The freeze-flow control reaches the same 30.08% transformation, while feedback lowers terminal stress by 36.4 MPa; front disablement raises it by another 8.34 MPa.

This qualifies substantial prepared-boundary migration and conservative material processing.  It does not qualify spontaneous intragranular DRX: the actual evolved-state recognizer found no qualified intragranular boundary, and no production nucleation event was enabled.

## Coupled DRX/ASB competition

An intermediate front attempt frequency (`10^10 s^-1`) keeps the boundary active over a useful deformation horizon.  At 30,000 s^-1 and 15.01% strain it produces:

- 4.19% net transformed material;
- stress 2.543 GPa;
- mean / peak temperature 1056.5 / 1181.1 K;
- 124.7 K peak-minus-mean contrast;
- plastic-power participation 0.856.

The trajectory initially concentrates (participation 0.667 at 5.01%) but then broadens.  It therefore demonstrates simultaneous DRX, plastic flow, Mura evolution, heat generation, and thermal feedback, but not persistent narrow localization.

At 100,000 s^-1 and 5.01% strain, the matched 1.5 micrometre feedback/control pair gives 0.496% DRX, 46.4 K thermal contrast, and power participation 0.662.  Feedback lowers stress by 56.3 MPa and participation by 0.0266 relative to frozen-flow.  A 0.75 micrometre particle is slightly broader (participation 0.674), and reducing conductivity tenfold changes participation by only -0.00049 at this n64/time horizon.  These are valid negative discriminators; no screening snapshot satisfies the inherited participation <=0.25 and contrast >=50 K conjunction, so persistence/refinement gates are not invoked.

The retained single-crystal V54/V55 continuation remains a complementary result through 35% strain: causal thermal feedback is strong, but the response is broad or nonpersistent rather than strict ASB.

## Polycrystal reference

The repaired reference run contains eight physical grains, anisotropic orientations, approximately 29% GB support, compatible GND evolution, GB transmission/recovery, finite loading, and heat through 5.01% strain.  Feedback versus frozen-flow gives 628.38 versus 630.42 MPa and participation 0.4758 versus 0.4770.

This branch is qualitative only.  It uses the earlier multi-grain tensor path, for which the production record declares `all_channels_available=false`; its legacy cumulative first-law ledger does not close.  It cannot qualify ASB or replace the common-owner bicrystal evidence.

## Claim boundary and next scientific work

- Existing-boundary DRX is functioning, causal, conservative, restart exact, and spatially convergent for a generic mobility hypothesis.
- Strict ASB is not observed in the tested single-crystal, bicrystal, or polycrystal screens.
- Spontaneous intragranular grain birth remains unqualified.
- A production multi-grain common-owner Mura/front state is still missing; the legacy polycrystal path is not energy-qualified.
- The next high-value architecture task is generalizing common phase ownership from one pair to multiple grains while retaining the complete energy ledger.  ASB work should then use resolved thermal lengths and inherited criteria, not lower thresholds or background subtraction.

Machine-readable evidence is in `v55_integrated_decision.json`; the principal plots are `v55_integrated_history.png` and `v55_high_rate_endpoint_fields.png` in the external campaign tree recorded by the V55 manifest.

The final canonical regression passes 885 tests in 374.59 s.
