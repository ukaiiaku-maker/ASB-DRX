# V67 carrier-exchange continuation evidence index

Updated: 2026-10-19T19:18:00Z

This index is intentionally compact. It identifies the retained evidence used
to develop the arrival-limited candidate without copying the historical result
trees.

## Frozen parent states

- `v67_poly_rx_asb_seed19p375_ad_400/checkpoint_latest.npz`, load 0.06664.
- `v67_poly_rx_asb_seed19p375_ad_500/checkpoint_latest.npz`, load 0.08264.
- `v67_poly_rx_asb_seed19p375_ad_600/checkpoint_latest.npz`, load 0.09864.
- The matched long-horizon classification reaches 0.13064 and remains
  `SUSTAINED_SOURCE_LIMITED_ARREST_AFTER_PRIOR_RX_BAND`.

## Accepted-flow reconciliation

Machine record:
`full_model/verification/v67_accepted_carrier_budget_ad400_ad600.json`.

- Raw versus accepted interval-mean plastic powers are 60.267/60.443,
  14.533/14.856, and 0.033936/0.043335 TW/m3 on ad400, ad500, and ad600.
  The accepted mechanical map consumes the full 2 ns interval in all three
  probes. The positivity limiter therefore does not explain the flow collapse.
- Mobile physical line falls from 1.71894e-7 m to 2.38831e-8 m. More
  importantly, the high-power owner/family populations are selectively
  depleted: the large family-2/3 owner-10 and family-3 owner-30 carriers at
  ad400 are nearly absent by ad600. Surviving line is poorly correlated with
  resolved glide power.
- Across ad400--ad500 and ad500--ad600, lattice/interface absorption accounts
  for 101.53% and 101.46% of net mobile loss. Hall--Petch creation replaces
  only 0.0401% and 0.295% of those losses. The grouped mechanical reactions
  return a small amount and close the stated balance identity.

## Minimal candidate

The existing complete Arrhenius reaction clock is retained, including the
Hall--Petch event measure. For glide-supplied lattice capture only, it is put in
series with the actual signed normal arrival clock. In diffuse form,

`contact = dx |grad eta|`, `H_arr = contact |v| dt |s.n| / dx`,

and the reaction and arrival exposures combine harmonically before competing
for their common donor. This is the finite-volume representation of
`rho max(v.n,0)` and cancels the numerical `dx` in the surface-to-volume
conversion. It is not a fitted transparency probability. Zero normal arrival
gives zero new lattice capture; already stored interface content remains in
its explicit reservoir and retains its separate structural/reaction paths.

On immutable ad400/ad500/ad600 probes, the candidate remains downhill, closes
line/Burgers/Mura/first-law audits, and reduces newly captured line to 0.113%,
0.0918%, and 0.589% of the former reaction-only debit. These atomic values do
not establish a production response; the immediate next experiment is the
exact-parent ad400 coupled continuation against the retained baseline.

## Exact-parent production discriminator

Machine record:
`full_model/verification/v67_arrival_exchange_candidate_ad400_ad600.json`.

- From the exact ad400 parent to ad600, the candidate retains a
  plastic/imposed power fraction of `0.0460981`; the reaction-only baseline is
  `6.50172e-5`. This is a `709.0`-fold recovery of accepted flow with every
  absorption, Mura, first-law, and grain-identity hard audit passing.
- Candidate mobile line at ad600 is `1.315` times the baseline. Cumulative
  interface capture is only 4.5% lower, demonstrating that signed arrival
  preserves the power-carrying family/order correlation rather than merely
  inflating scalar density.
- The identical candidate preserves positive existing-boundary grain-20 motion
  at 2/s during both a resolved dynamic continuation (`3.8515e-5` relative
  area) and its no-reset zero-rate continuation (`3.8458e-5`).
- This is not yet ASB. The adiabatic endpoint has a connected band for only two
  terminal intervals and zero post-peak softening. The classified result is
  `CARRIER_COLLAPSE_PREVENTED_RX_PRESERVED_ASB_NOT_YET_QUALIFIED`; the next
  discriminator is the identical matched strong-bath continuation.

## Claim boundary

The candidate changes only the supply measure of lattice-to-interface capture.
It does not alter Arrhenius flow, Peierls or Taylor resistance, Hall--Petch
nucleation, source multiplicity, front transmission, thermal parameters, or
plastic history. Its coupled high-rate, lower-rate dynamic-RX, and no-reset
static consequences pass on the tested resolved continuations. The matched
bath comparison, persistent localization, and softening criteria remain open.
