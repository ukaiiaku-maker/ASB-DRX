# V30 Workstream C decision

Workstream C is `FAILED_SCIENTIFIC`; no resolved ASB or HPC calculation is
authorized.

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
unavailable at `drx_full_v34_recovery.py:8578`; a silent zero is forbidden.

The 32/64/128 two-step production smokes give relative first-law residuals
0.731, 0.512, and 0.728, respectively, versus the 0.05 gate.  The 32-square
homogeneous control remains exactly homogeneous in density and temperature,
but its relative residual is 0.904.  Thus closure failure is not cured by grid
refinement or removal of spatial heterogeneity.  Its immediate cause is that
legacy phase/line-storage operators and work partitions are not yet accepted
as atomic energy transactions; the residual is diagnostic only and is not
converted into dissipation.

A 32-square three-step run and a 2+1 segmented run are bitwise identical for
`rho`, signed populations, temperature, boundary content, plastic slip/strain,
total strain, and the cumulative ledger JSON.

Verification: 30 focused tests passed; the full canonical suite passed 509
tests in 60.92 seconds.  Source checkpoint:
`4f99528e7f82c4a8886eed9ccacd2500a44abb67`.

The next admissible repair is to run this ledger on the authoritative common
Mura/wall production path, where junction affinity is declared, and to make
each phase/line-storage operator commit physical energy, work, and heat in the
same accepted transaction.  A closure residual must remain an error measure,
never a constitutive channel.
