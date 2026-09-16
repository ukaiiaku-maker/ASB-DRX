# V30 Workstream C decision

Workstream C is `FAILED_SCIENTIFIC`; its primary classification is
`ASB_NUMERICAL_CONSTRAINT_CONTAMINATES_PHYSICAL_LEDGER`. No resolved ASB or
HPC calculation is authorized.

The typed seven-channel ledger is now part of the accepted production update,
diagnostic CSV, terminal result, and exact checkpoint schema.  It separately
records physical line/correlation, elastic, phase/interface,
GB/disconnection, and thermal energy; numerical compatibility penalties and
multiplier work remain outside physical energy and heat.  Dissipation is never
obtained from a first-law remainder.

Six channels have independent production affinity/extent ownership in the
legacy path: plastic drag, extent-limited mobile/forest recovery, realized
neutral-pair annihilation, realized boundary recovery, Fourier conduction,
and declared bath sinks.  The common-wall reaction path additionally exposes
independent junction affinity-times-extent dissipation.  The default legacy
junction transfer does not declare that affinity, so it is explicitly marked
unavailable; a silent zero is forbidden. Its accepted law is an irreversible,
stress/activity-scaled donor fraction. It has no chemical potential, reverse
rate, or conjugate junction state in the legacy free energy. Consequently an
affinity cannot be derived from the existing law without inventing new
physics. The common-wall affinity may not be copied onto this different state
law.

The 32/64/128 two-step production smokes give relative first-law residuals
0.631, 0.0395, and 1.089, respectively, versus the 0.05 gate.  The isolated
64-square pass is not converged and cannot qualify the grid sequence. The 32-square
homogeneous control remains exactly homogeneous in density and temperature,
but its relative residual is 1.065. Thus closure failure is not cured by grid
refinement or removal of spatial heterogeneity.

The new accepted-suboperator audit identifies the first dominant contaminant.
Mechanical loading contributes approximately +0.299 MJ/m3 at every grid,
consistent with about +0.302 MJ/m3 external work. Storage/recovery contributes
approximately -0.614 MJ/m3. In contrast, `orientation_boundary_sources`
increases physical stored energy by +1.045, +0.707, and +0.527 MJ/m3 while the
numerical compatibility penalty changes by -1.31e22, -4.01e22, and -7.77e22
J/m3. The orientation rate is driven directly by `dFdpsi` from the numerical
`A_alpha/A_GB` compatibility penalty. This is numerical constraint work on a
physical state and cannot be counted as heat or dissipation.

The secondary grid-dependent term is `phase_front_topology`: -0.021, -0.186,
and -0.543 MJ/m3 over 32/64/128. It explains the accidental 3.95% residual at
64 square and rejects that isolated pass as nonconverged.

A 32-square three-step run and a 2+1 segmented run are bitwise identical for
`rho`, signed populations, temperature, boundary content, plastic slip/strain,
total strain, and the cumulative ledger JSON.

Verification: 30 focused tests passed; the full canonical suite passed 509
tests in 62.47 seconds. Source checkpoint:
`5b20eeaff3ae8234b0bc7c42d536478be20eb83c`.

The next admissible repair is to run this ledger on the authoritative common
Mura/wall production path, where junction affinity is declared, and to make
each phase/line-storage operator commit physical energy, work, and heat in the
same accepted transaction.  A closure residual must remain an error measure,
never a constitutive channel.
