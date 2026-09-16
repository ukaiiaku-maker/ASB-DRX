# V33 n128 near-equal front diagnosis

## Decision

The step-137/138 events are **grid-scale phase-representation breakdowns**.
They are not physical phase islands, a periodic-seam identity error, or evidence
of a genuine near-equal thermodynamic instability. Historical V31
`FRONT_COMPONENT_SPLIT` and V32 `UNAUTHORIZED_PHASE_ISLAND` labels remain
unaltered; this is the scientific interpretation of those conservative
terminals.

No long front run is authorized from these states.

## What the rejected trial actually contains

The positive and negative `1e-6` cases are symmetric. One Allen-Cahn trial
changes the sign of 104 and 102 pixels, respectively. The new sign occupies a
periodic-seam-spanning filament only one cell in radius (`0.078125 um`) and
roughly 101 cells long. Its maximum phase difference is only `0.001086` or
`0.000895`; it has zero pixels with `|eta_child-eta_parent| >= 0.9` and
therefore no pure phase core.

The graph encloses 2.421 and 1.896 cell-squared while reporting contour lengths
of 201.67 and 200.84 cells. This combination is the signature of a long,
one-cell zero-level filament, not a compact island. Periodic unioning correctly
identifies each seam-spanning filament as one component, so seam bookkeeping is
not the cause.

Only the zero isovalue changes topology: each trial adds two zero-level contour
components, while the component counts at `+/-0.2`, `+/-0.5`, and `+/-0.8`
remain unchanged. Neighboring isovalues therefore do not form a nested phase
interface around the alleged island.

## Resolution, thickness, and spectra

The accepted n128 main interface has median normal thickness about 9.21 cells,
or `0.719 um`; the n64 field has about 5.24 cells, or `0.819 um`. The broad
interface is resolved, but the terminal filament remains exactly one cell in
radius. The n64 raw terminal was instead a sub-cell loop of 0.0166 cell-squared
and 0.665-cell length (equivalent physical radius `0.0114 um`) and is removed
from ownership by the V32 sub-cell filter. The n128 morphology is not a
grid-convergent continuation of that loop.

High-wavenumber power is small but rises in both rejected trials:

- negative case: `0.001188 -> 0.001257`;
- positive case: `0.000845 -> 0.000885`.

The change is localized at the zero crossing rather than a broadband bulk
mode. The diagnostic figure shows the one-row sign changes and the unchanged
neighboring isovalues directly.

## Energy and state-coupling evidence

The discrete interface-plus-bulk phase energy increases in the rejected trial,
by 2.59% in the negative case and 1.59% in the positive case. The candidate is
therefore not a resolved downhill phase instability. The production adapter
rejects it atomically: accepted eta is bitwise identical to pretrial eta and
all material ledgers remain unchanged.

These continuation cases already use the S3 isolation stage, which disables
ordinary transport, storage, recovery, nucleation, relabeling, and collective
organization. Reversing the minute density contrast reverses the location but
reproduces the same pathology. Changing the bulk barrier by 0.5x or 2x does not
remove it. A stored-energy-off restart is intentionally prohibited by the
sparse-front common-variational contract, so that ablation is recorded as
unavailable rather than bypassing model invariants.

## Compact controls

- Baseline: terminal at step 138.
- Half timestep for two steps and quarter timestep for four steps reach the
  same physical interval and still terminate at steps 139 and 141. Smaller
  timesteps delay the crossing in step count but do not remove it.
- Half capillarity terminates; doubling `kappa_eta` prevents a terminal over
  the matched interval and accepts a bounded phase update. This strong
  capillarity sensitivity identifies a representation/discretization issue.
  The doubled-capillarity case is only a bounded structural alternative; it is
  not a calibrated repair and is not promoted into production.
- Halving or doubling `W_eta` does not prevent the terminal.
- Freezing phase evolution is an exact negative control: no phase change and
  no terminal.

## Effect of V32 filtering

Topology extraction is demonstrably read-only: eta hashes before and after the
filtered and unfiltered queries are identical, as are the cut-cell fractions.
For the n128 trials, the area/length filter removes nothing because the
seam-spanning filament has large graph length. V32 filtering changes only
diagnostic component ownership for the n64 sub-cell loop; it never alters the
physical phase field or swept material.

The next scientific repair should therefore regularize or reject zero-level
filaments using a phase-core/inradius representation criterion tied to the
resolved interface width. It should not reinterpret these terminals as grain
nucleation or relax the fail-closed topology gate.
