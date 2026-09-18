# V41 complete-dissipation front law and declared Nye transfer

## Direction–energy consistency

For each signed geometric direction (d\in\{A\to B,B\to A\}), production first
constructs the complete common-state endpoint without publication.  Its
available change is

\[
\Delta\mathcal A_d=F(q_d)-F(q_0)-W_d+E_{\mathrm{sink},d},
\]

where the retained zero-applied-work holds have (W_d=0).  The energy contains
the declared defect, signed-junction, boundary-excess, recoverable-elastic,
phase-local, and phase-gradient terms.  Thermal energy, generated heat, and
exports remain in the separate first-law ledger.

The EXP-floor transition-state coefficient is unchanged:

\[
k_d^*=a_d\nu_0\exp(\Delta S^*/k_B)
\exp[-\Delta H^*_{\mathrm{EXP-floor}}(|\Delta\mathcal A_d|/V_e)/(k_BT)].
\]

The selected deterministic activity is

\[
r_d=k_d^*\,\max\!\left[1-\exp\!\left(
\frac{\Delta\mathcal A_d}{k_BT}\right),0\right],
\qquad
v_{A\to B}=\ell_e(r_{A\to B}-r_{B\to A}).
\]

Thus every nonzero directional activity owns a nonpositive complete affinity.
The rule is zero continuously at equilibrium, retains the mechanism-specific
EXP-floor barrier and signed entropy exactly once, and does not use a fitted
mobility multiplier.  The complete finite-candidate guard remains mandatory.
The V40 `legacy_independent_metropolis` traffic-difference rule remains an
explicit comparator, but is no longer the I3 or monolithic production default.

The two outgoing transactions are not called a microscopic reverse pair when
irreversible transmission, boundary storage, annihilation, or sink processing
differs.  Consequently stochastic detailed-balance labels are not applied to
the selected deterministic law.

## Declared support-gradient Nye term

For material owners (a) with support weights (w_a), production reconstructs

\[
\bar\beta^p=\sum_a w_a\beta_a^p,
\qquad
\alpha=-\operatorname{Curl}\bar\beta^p.
\]

Its declared split is

\[
\alpha=\underbrace{\sum_a w_a(-\operatorname{Curl}\beta_a^p)}_{\text{bulk}}
+\underbrace{\sum_a[-\nabla w_a\times\beta_a^p]}_{\text{interface}}
+\epsilon_h.
\]

`CommonFrontState.interface_nye_m1` already owns the discrete support-gradient
term.  The V40 stage diagnostic compared curl-Nye only with bulk reservoir
first moments and therefore mislabeled the declared interface contribution as
a transfer discrepancy.  V41 compares curl-Nye with bulk moments plus the
stored interface term and reports the remaining representation residual,
strong norms, physical integrals, circuits, support width, and common Fourier
modes.  No target-Nye reconstruction, filtering, or projection changes the
authoritative state.

The trace ordered inventory is already present after the first Mura half-step,
before the front transaction.  Its near-(h^3) scaling is therefore a
near-extinction ordering/tolerance diagnostic, not line created by the front.
