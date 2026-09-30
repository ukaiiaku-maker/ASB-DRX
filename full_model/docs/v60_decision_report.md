# V60 decision report: local front heat and topology-aware ASB audit

## Decision

V60 repairs two confirmed limitations in the V59 production/evidence path:
front-mobility heat is now deposited on the realized cellwise swept measure,
and periodic component morphology distinguishes contractible structures from
rank-one and rank-two winding topology.  The repaired production path passes
its conservation tests and a matched same-parent ablation demonstrates a
spatial effect at fixed total heat.

The corrected n32 four-grain physical/all-Arrhenius-frozen pair remains a
negative strict-ASB result through 3.0 microseconds.  Prepared-boundary
migration and causal thermomechanical differences remain valid, but the power
candidate is underresolved, short-lived under the registered conjunction, and
not co-located with either mechanical or front heat over a qualifying episode.

## Production repairs

For every accepted front subevent, the deposited cell energy is

`Q_j = sum_e p_e * accepted_fraction[e,j] * cell_volume`.

The same corrected pair pressure used for mobility dissipation supplies this
field.  Its spatial normalization is corrected only for floating reduction
error, and its integral is the independently computed dissipation.  A labeled
`uniform_ablation` mode is retained for matched diagnostics.  Conduction, not
an instantaneous whole-domain redistribution, subsequently moves heat.

The morphology audit now lifts contractible periodic components to a common
cover.  Rank-one winding bands use a homology-derived periodic transverse
coordinate; rank-two winding networks are explicitly reported as networks.
Temporal association is constrained by elapsed time and a declared maximum
component speed, while unrestricted translation remains a nonqualifying
diagnostic.

The instantaneous budget separately exposes plastic power, glide drag,
mobile/forest recovery, neutral annihilation, junction, wall exchange,
ordering, reversible defect-energy transport divergence, conduction, bath
exchange, and local thermal storage.  Accepted-interval front and mechanical
heat fields are stored as supplementary checkpoint diagnostics and are not
mixed with instantaneous rates.

## Same-parent heat fork

The V59 four-grain physical checkpoint at 3.0 microseconds
(`329b74f...`) was continued twice for 50 ns using source `954c399`.  The two
children differ only in local versus uniform front-heat deposition.

- At 3.025 microseconds, total generated heat differs by only
  `4.04e-27 J` and supports by `1.12e-13`, while the local-minus-uniform
  temperature field spans `5.70 K`.
- At 3.050 microseconds, feedback increases the temperature-field span to
  `11.31 K`.
- Both children accept every event with the same parent, source, physical
  configuration, clock, and loading coordinate.

This is an operator/provenance discrimination, not an ASB or convergence
qualification.

## Corrected-source matched pair

The physical and all-Arrhenius-frozen V59 n32 checkpoints at 2.5
microseconds were each continued prospectively with local front heat to 3.0
microseconds using source `c545d35`.  Past uniformly deposited heat was not
relocated.  Both members complete 20 macro steps, accept 60 front subevents,
and have zero rejected events.

Across the 11 saved common states:

- matched physical-minus-frozen temperature excess on the associated power
  component peaks at `48.969 K` and ends at `16.842 K`;
- the power-component width ranges from `59.86` to `222.35 nm`, below the
  unchanged `625 nm` two-interface-width screen;
- aspect ranges from `1.763` to `3.561` and is not sufficient to repair the
  width and causal-excess failures;
- instantaneous mechanical heat/power component overlap is zero at 10 of 11
  states and `0.364` once;
- accepted-interval front-heat/power overlap ranges from `0` to `0.0171`;
- endpoint true preceding-peak softening is `13.98%`, below the registered
  `20%` screen; and
- no saved state, and therefore no one-microsecond episode, satisfies the
  full strict-ASB conjunction.

At 2.5 microseconds the original four-grain component is contractible, so the
topology repair does not change its reported `61.6 nm` width or `4.09` aspect.
That result remains an underresolved precursor rather than a winding-analysis
artifact.

## Scope and next evidence

V60 does not establish strict ASB, spontaneous grain birth, or material
calibration.  It does establish a locally resolved front heat operator and a
decision-grade negative n32 corrected window.  The next useful calculations
are fixed-domain n48 physical/frozen companions from attributable initial
states, followed only if useful by n64, and the bounded single-crystal
thermal/sign-balanced defect perturbation discriminator.  Those calculations
must preserve the current physical scales and must not tune classification
thresholds.

