# v10 existing-HAGB SIBM common-boundary model

## State and functional

The SIBM path migrates an existing pair of phase labels.  It never samples an
orientation or allocates a bulk label.  The advancing phase inherits its
orientation and lineage.  Its defect state uses the v9 virgin-parent,
active-child, recovered-wake, current-fraction, and maximum-swept-fraction
representation.

An argmax-label contact is insufficient evidence for an existing boundary in
a diffuse multiphase field.  Both phases must have independent `eta>=0.8` pure
cores and share a resolved diffuse pair band.  The deterministic experiment
then freezes phase evolution outside one declared pair-local circular window.
Its seed radius must exceed two `sqrt(kappa_eta/W_eta)` interface widths.

For a graph fixture with height `h(x)` and represented thickness `t`, the
declared free energy is

\[
F=t\left[\gamma\int\sqrt{1+h_x^2}\,dx
 -(\Delta\psi-P_{FB})\int h\,dx\right].
\]

Its negative variational derivative gives

\[
P_n=\Delta\psi-\gamma\kappa-P_{FB}-P_{drag}\quad [\mathrm{Pa}].
\]

The full 2-D implementation uses the same phase functional derivative.  Its
stored-energy difference is line tension `[J m-1]` times parent-minus-child
line density `[m-2]`.  The physical GND/Frank--Bilby opposition is also priced
at line tension and inserted as a frozen-state child energy density during an
accepted operator-split phase step.  Quadratic alpha/GB constraint penalties
are reported separately and do not enter migration or heat.

For a circular 2-D bulge the stationary condition is

\[
R_c=\frac{\gamma}{\Delta\psi-P_{FB}-P_{drag}}.
\]

Mobility changes the time scale, not this threshold.

## Kinetics

The signed direction comes only from the common pressure.  Its magnitude is
multiplied by the symmetric EXP-floor activation fraction with

\[
\Delta G^*_{GB}=\Delta H^*_{\mathrm{EXP-floor},GB}-T\Delta S^*_{GB}.
\]

Forward and reverse steps therefore use one law and one entropy contribution.

## Content and energy ledger

Newly swept material alone is partitioned into compatible child transmission,
signed boundary storage, neutral-pair annihilation, and declared sink/escape.
The line ledger is

\[
L_p=L_c+L_{GB}+L_{ann}+L_{sink}+R_L.
\]

`R_L` must close to roundoff; signed Burgers change must be zero.  Line energy
released by annihilation is the front heat channel.  Retreat changes current
phase fraction but not maximum swept fraction, so re-advance cannot process
the same material twice.  Sparse and full defect fields are canonicalized at
every physical step and serialized together for exact restart.

Bulge amplitude and neck width are measured on the four-connected excess-child
component seeded at the declared boundary point, relative to the pre-bulge
field and restricted to the original parent support.  Area-equivalent normal
velocity is the increment of that excess area divided by neck width and time.

The first HPC3 run predates the resolved-core/pair-window rule.  It is retained
as `SIBM_VALIDITY_FAILURE`: an unresolved argmax phase was allowed to consume
the domain while the seeded component did not advance.  It is not SIBM growth
evidence.  The corrected pair-scoped implementation is locally qualified; no
second HPC job was launched in v10.

The deterministic fixture record is
`full_model/verification/v10_sibm_local_fixtures.json`; the short full-model
record is `full_model/verification/v10_sibm_full_model_local.json`.
