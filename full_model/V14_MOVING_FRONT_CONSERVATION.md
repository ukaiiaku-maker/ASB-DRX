# V14 moving-front conservation model

## State ownership and units

Each material state carries nonnegative positive and negative mobile densities
`rho+_a`, `rho-_a` and forest density `rhoF_a` in m^-2 for every slip family,
plus wall density `rhoW` in m^-2. Parent, child, and wake values are intensive
states owned only on their phase support. The canonical checkpoint-v6
fractions are `c`, the current child fraction, and `s`, the maximum
historically swept fraction. The recovered-wake fraction is derived:

`w = s-c`, with `0 <= c <= s <= 1`.

The reconstructed physical field is

`q = (1-s) q_parent + c q_child + (s-c) q_wake`.

The three weights are nonnegative and sum exactly to one. Absent support owns
exactly zero extensive content. In source, `current_child_fraction` is the
explicit alias for legacy `chi`, and `maximum_swept_fraction` is the explicit
alias for legacy `processed_max`. Historical schema-v3--v5 checkpoints migrate
by precisely those mappings; `cleanup_max` is retained only as a reaction-
history compatibility field. Schema v6 writes both canonical names and
byte-identical legacy aliases, rejects conflicting aliases, and derives wake
support as `maximum_swept_fraction-current_child_fraction`.
The intrinsic HAGB energy `gamma(theta)` is a phase-field property and is never
stored in the excess boundary-dislocation reservoir.

## Cellwise constrained transfer

For a newly swept fraction `dchi`, the available parent line content of sign
`s` and family `a` is `L^s_a = dchi V rho^s_a`, where `V` is represented cell
volume in m^3. A declared transmission fraction first transfers like-signed
content to the child. Opposite signs in the remainder may annihilate only in
equal pairs. Residual signed content is stored as excess boundary content or,
only in the labeled ablation, removed through an explicit signed sink.

For every accepted cell the implementation enforces

`L_parent = L_child + L_boundary + L_annihilation + L_sink`

in metres of line and enforces the signed Burgers balance family by family.
The line-energy release is `Gamma_line * (L_annihilation + L_sink)` in joules
and the heat ledger receives exactly the same amount. Numerical compatibility
penalties and intrinsic HAGB energy are excluded from both ledgers.

If finite excess-boundary capacity makes the requested conversion infeasible,
the admissible fraction `alpha_max` is returned and the cell is left unchanged;
there is no clipping or conservation exception. Accepted phase change, swept
volume, and every content ledger use the same accepted fraction.

## Profile versus sweep

Diffuse profile relaxation changes neither `m` nor any front ledger. Only the
signed normal-contour crossing supplied to `advance_front` processes virgin
material. Retreat converts child support to wake support; re-advance through
processed material does not delete content twice. Thus stationary profiles,
width relaxation at fixed contour, and oscillatory revisits have zero new
processing ledger unless a contour reaches virgin parent material.

## Qualification convention

Pinned-cap motion is classified by amplitude relative to the pinned chord and
mean interface. Neutral stationarity is declared as less than 0.02 grid cell.
Continuous versus segmented restart uses a maximum field-scaled difference of
1e-12, which admits floating-point FFT/reduction-order roundoff but no physical
trajectory tolerance. The raw `local_normal_pressure_Pa` output is a
flat-interface diagnostic; the pinned continuation sign is the applied pressure
minus physical drag minus the declared neutral continuation pressure.
