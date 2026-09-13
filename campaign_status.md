# Campaign status

Updated: 2026-09-12 (America/Los_Angeles)

## Directive v5 transition: full v34 production model restored

### Directive v6 hazard-measure checkpoint

- The production candidate path now has an opt-in area-integrated physical
  hazard measure. Local creation rate has units `s^-1` per physical site; a
  declared areal site density converts it to `m^-2 s^-1`, and physical area and
  time integration give dimensionless expected event count. Grid cells are not
  treated as independent sites.
- Its single global Poisson clock carries exposure through GB translation,
  swept support, and Eulerian cell changes. Thresholds redraw only after a
  completed event; discarded/newly initialized/transferred exposure and every
  redraw cause are explicit. A bounded multi-event loop processes all crossings
  up to its declared work limit and refuses further physical advancement while
  an event is deferred.
- Uniform 16/32/64 grids give the same total exposure; translated support loses
  none. A forced full-driver fixture reached `Lambda_tot=1.686451534478067`,
  `P(N>=1)=0.814824551763979`, realized three events, created three embryo
  records, and retained exactly three grain labels. It had three event redraws,
  zero other redraws, and zero discarded exposure.
- Continuous four-step and exact 2+2 trajectories agree bitwise across 120
  authoritative checkpoint fields, including the independent hazard RNG.
  With the new path disabled, the production driver remains bitwise identical
  to commit `3aef225` across all 108 common authoritative fields.
- These are forced invariant fixtures, not a selected site density or creation
  parameterization. Atomic promotion and offline reweighting were subsequently
  completed below before staging the first v6 HPC3 scientific job.
- Route B precursor creation is now explicit: the stochastic event pays one
  EXP-floor kinetic free barrier and creates the minimum resolved precursor;
  the classical circular free energy supplies feasibility, critical-radius,
  and subsequent growth/shrinkage only. Candidate viability uses the kinetic
  free barrier, so the classical barrier is not paid a second time.
- Atomic phase promotion is connected to the production trajectory. A trial is
  built on copied fields and commits phase simplex, conservative line transfer,
  heat, provenance, embryo status, and grain lineage together only when total
  event free energy is nonincreasing. Capacity, slot, phase, growth, or energy
  failures leave every caller-owned field unchanged.
- The forced Route-B fixture promotes at step 3, preserves signed Burgers
  density exactly, closes physical line length to `2.17e-19 m`, closes event
  energy exactly, and does not call the newly allocated label a physical DRX
  grain. Restarts immediately before, at, and after promotion are bitwise exact
  across 123 authoritative checkpoint fields.
- Offline reweighting used 25 saved no-trigger full-field states through strain
  0.5368. With fixed `H0=1.2 eV`, attempt frequency `1e6/s`, zero entropy,
  unity activity prefactor, and no mechanics rerun, site densities `2e10`,
  `2e11`, and `6e11 m^-2` give total exposures `0.10294`, `1.02942`, and
  `3.08826`. The preregistered central row has event probability `0.64279`;
  dominant exposure ends near strain 0.437.
- The complete local suite passes 281 tests. The central HPC3 case is staged
  with baseline growth/phase energetics; none of the extreme forced-fixture
  mobility, interface, compatibility, or orientation parameters are carried
  into it.

- Production development has moved to isolated branch
  `exp/full-v34-recovery-v1`; the reduced endpoint is preserved at `fb464e1`
  and tag `reduced-v3-final-validity-stop-20260912`.
- Exact v32--v34 drivers, runners, analysis scripts, archived configurations,
  evidence roots, representative results, and prior verified HPC3 archives are
  recorded in `full_model/provenance_manifest.json`.
- The supplied v34 baseline first fails at hazard exposure: it has nonzero
  eligible sites and rates, but `max(H/E)=8.98e-6`, hence zero raw stochastic
  events, candidates, promotions, or physical grain births.
- The baseline checkpoint omits all candidate arrays.  Exact candidate restart
  is therefore a confirmed implementation defect, not an untested concern.
- The reference-restoration commit changed no full-model physics.  Subsequent
  work is confined to the separate production copy.
- Patch A is now locally staged in the separate production copy.  All-zero
  mechanism entropies reproduce 38 common immutable-v34 checkpoint fields
  bit-for-bit on a two-step 16x16 calculation.  Unit fixtures verify positive
  and negative entropy, prefactor identifiability, the EXP-floor exponent
  restriction, and explicit negative-barrier drag/rejection.
- Exact continuation exposed and repaired omissions in candidate arrays,
  `sigma_bar`, physical/global time, activity memory, and structural reference
  densities.  A 1+1 segmented trajectory now matches the continuous two-step
  trajectory bit-for-bit across all 27 compared authoritative fields.
- Checkpoints are now atomically published and are triggered by both physical
  progress and wall-clock interval.  The complete inherited-plus-full-model
  suite passes 255 tests locally.
- Matched HPC3 full-model regressions from clean source `44076d2` completed and
  were fetched. Job `55977649` (`asb_only`, 30,000/s, seed 42) completed in
  1,343 s and reproduces the immutable v32 peak stress, maximum temperature,
  and top-5% plastic-rate fraction to `3.74e-16`, `6.75e-15`, and `9.41e-15`,
  respectively. The full ASB-like reference response is therefore preserved;
  strict ASB classification remains pending a matched isothermal control and
  width refinement.
- Job `55977650` (`drx_isothermal`, 1,000/s, seed 42) completed in 1,341 s but
  reached strain 0.4369 rather than the required 0.5 because adaptive accepted
  timesteps were smaller than the nominal strain increment. It contains 352
  eligible sites and a peak hazard rate of `3.40e-2/s`, but only
  `max(H/E)=1.54e-6`: zero raw triggers, candidates, promotions, or births.
  The provisional diagnosis is hazard exposure/integration, not candidate
  persistence or label promotion. Per the physical-horizon rule, the current
  classification is `INCONCLUSIVE_INSUFFICIENT_HORIZON` until an exact
  checkpoint continuation crosses strain 0.5.
- Exact continuation job `55978063` completed from the verified step-4999
  checkpoint (`3fc4977b...`) and crossed the required horizon at strain 0.5368.
  All output and source checksums pass. It still has zero candidates, but the
  v34 diagnostic did not separately count raw `H>=E` triggers. Since comoving
  GB and swept-cell updates can redraw thresholds and reset hazard, a final
  `H/E<1` cannot prove that no earlier raw trigger occurred. The horizon is now
  sufficient, but the first-failure-stage claim remains unqualified pending a
  counter-instrumented no-retuning rerun.
- The production copy now checkpoints cumulative raw and viable-trigger counts
  and writes both to diagnostics. A forced-trigger 16x16 fixture produced four
  triggers by step 1 and 32 by step 2; continuous and exact-restart trajectories
  match bitwise across 48 authoritative checkpoint fields (only `P_json`, which
  records restart/run-length controls, differs).
- No-retuning raw-trigger audit job `55978313` completed from clean commit
  `86d7e96` in 1,568 s and was fetched with all checksums verified. At strain
  0.5368 it records exactly zero raw triggers, viable triggers, candidates,
  promotions, and births despite 352 eligible sites and a peak hazard rate of
  `3.40e-2/s`. The baseline first failing stage is now definitively hazard
  integration before stochastic triggering. It is an integrated regression,
  not a DRX mechanism result or predictive validation.
- A long-segment comparison exposed another exact-restart defect hidden by the
  earlier 1+1-step smoke test: numerical Arrhenius-potential tables and their
  temperature cache were not checkpointed. Rebuilding them at restart changed
  potential-update hysteresis and eventually the full fields. The production
  copy now serializes/restores all 58 numerical potential members and rejects
  legacy checkpoints missing them by default. A 20-step continuous versus
  10+10 segmented fixture is bitwise exact across all 106 authoritative fields.
- The current complete suite passes 268 tests. Staged Patch B--F components are
  isolated from the production trajectory: stateful embryo identity/dynamics,
  deterministic diffuse embryo support, full-field environment coupling,
  strict physical-grain recognition, and conjunctive matched-control ASB
  classification. None is yet promoted to an integrated scientific claim.
- Every scientific output checksum in both fetched runs passed. Their generated
  inventories contain one invalid self-entry because the old runner hashed the
  inventory while writing it; this provenance-only defect is recorded and fixed
  for future jobs by commit `09d20cb`.
- A selectively back-ported, full-model-only Patch B/C component now carries
  persistent embryo identity, lineage, RNG state, radius/free-energy history,
  signed-entropy EXP-floor mobility, continuous growth/shrinkage, contact and
  phase-support observables, and an exact serialization ledger. It is now
  coupled to the authoritative v34 temperature, stored-energy, compatibility,
  GB, wall, and GND fields. The immutable-v34 path remains disabled by default.
- A forced 16x16 integration fixture generated 436 raw viable triggers and
  accepted three persistent embryo records while leaving the allocated grain
  count exactly unchanged at three. Continuous four-step evolution and exact
  2+2 restart agree bitwise across all 108 authoritative checkpoint fields,
  including embryo histories, the physical-grain tracker, RNG, and all 58
  Arrhenius-potential members. The
  fixture is deliberately nonphysical: its embryos shrink below the resolved
  radius and retire, so it establishes coupling/restart/no-label invariants,
  not DRX.
- Embryo creation now has a separate opt-in EXP-floor kinetic enthalpy driven
  by favorable bulk pressure, with activation entropy entering once. The
  classical circular-nucleus balance remains the thermodynamic feasibility and
  resolved-radius calculation. This option is off for immutable-v34 regression.
- The complete local suite passes 269 tests. No new HPC3 calculation was
  submitted because the next scientific run must first include the Patch-E
  promotion/physical-grain contract; jobs `55932457` and `55950433` are
  unrelated and remain untouched.
- Patch E recognition is now connected diagnostically to the full trajectory.
  It reports resolved labels, persistent matrix grains, physical DRX grains,
  rejected/retired labels, and recrystallized area separately and checkpoints
  the tracker exactly. It cannot allocate a label; promotion remains disabled
  until its line-content/Burgers-content transfer can be closed explicitly.
- The promotion transfer prerequisite is now implemented as a full-model-only
  operator. It lowers core density by removing only sign-neutral mobile pairs
  plus unsigned forest/wall content, transfers every removed line increment to
  a resolved boundary shell, preserves each signed mobile population exactly,
  and rejects insufficient shell capacity rather than clipping. Three focused
  tests close line and Burgers content and exercise the capacity failure. The
  operator is not yet called by the production trajectory, so label allocation
  remains disabled on the stateful path.

## Current decision: instantaneous junction-friction closure rejected

### Current decision-grade checkpoint

- The hard consistency repair is locally complete: physical glide velocity is
  separated from event frequency; all 1-D content ledgers have physical units;
  the integrated control is now explicitly small-strain antiplane; lattice
  rotation follows total minus plastic spin; rotated slip tensors are used
  consistently for stress, kinematics, and work; and stored energy separately
  exposes mobile, junction, wall, and total-signed long-range contributions.
- The smallest signed-vector/topology model now has eight signed mobile
  species, explicit glissile/sessile junction products, a reaction incidence
  matrix, exact Frank-rule conservation, detailed-balance forward/reverse
  rates, junction-dependent forest friction inside the EXP-floor stress scale,
  and a complete isothermal reaction--transport Fourier symbol.
- Its bounded local screen finds no robust interior finite-wavenumber branch.
  Fifteen of sixteen cases are damped; the only positive case selects the
  shortest admitted continuum wavelength and peaks below continuum validity
  when extended. The closure is rejected without a nonlinear or HPC run.
- The canonical local suite passes 236 tests. Gate B0 remains the only passed
  physical-development classification. Wall mechanism, Gate B1, DRX, ASB,
  integrated scientific, and predictive claims remain false. The next branch
  is delayed junction memory plus a higher-order line-orientation state.
- Local topology result `output/v3_topology_dispersion.json` was regenerated
  from source `3f4712c`; SHA-256
  `cbb4103631d291eae783e5bf0a303a27d6b1098205585c7c82d9ec76a3a931e0`.
  No HPC job was submitted because this inexpensive local dispersion analysis
  rejects the closure before a nonlinear campaign is scientifically allowed.
- The immediate delayed-memory escalation adds `m_dot=(p-m)/tau` without a
  spatial length. Its 324-case dispersion screen finds 27 positive interior
  modes across 800--1000 K, two stresses, multiple diffusivities, and multiple
  relaxation times, with wavelengths `0.275--1.02 micrometers`. The region
  appears only at the largest screened forest strength, so it is a candidate
  for nonlinear falsification rather than a supported wall mechanism. The
  current suite passes 237 tests; no grain/phase allocation is enabled.
- Memory dispersion result `output/v3_memory_dispersion.json` is tied to source
  `6260a4b`; SHA-256
  `273696218d058455852466df189be6fa38162a1210b9ac902674aad099957942`.
- A nonlinear frozen-state implementation now evolves the same delayed-memory
  equations used by the Fourier Jacobian. An eigenvector-seeded 128-cell fast
  pilot reached exactly two predicted e-folds in `0.019734 s` of physical time;
  measured log amplification was `2.01850`. Local wall time was `21.7 s`,
  projecting about `108 s` to ten e-folds before nonlinear slowdown. This is an
  initial-slope and cost verification only and is explicitly classified
  `INSUFFICIENT_PHYSICAL_AMPLIFICATION_HORIZON`. The suite passes 239 tests.
- Corrected HPC3 pilot `20260913T021834Z-70d91f5-nlmem2e`, Slurm `55976150`,
  passed 17 targeted tests and completed two e-folds in 16 allocation seconds
  on `hpc3-14-06`. Measured log amplification was `2.0184975`; the projected
  solver cost to ten e-folds is 62.3 seconds. Local/HPC scientific outputs
  agree within `2.11e-9` relative. Packaging-only job `55976135` ran no
  equations and is retained as a failed provenance record.
- Fast physical-broadband hold `20260913T022148Z-bc2ecd7-nlmem12e`, Slurm
  `55976198`, reached `G=12` in `0.1184 s` physical time. The selected modal
  log amplification was `12.187`, closely following the frozen prediction.
  This establishes finite-amplitude growth from broadband noise, not a wall:
  no saturation, persistence, orientation jump, or refinement claim is made.
- Exact-checkpoint continuation `55976255` reached a last valid `G=12.912`.
  Total-density CV was 0.562, junction CV 1.519, and one mobile population
  peaked at 7.47 times its mean. The next implicit interval violated
  nonnegativity, so the run stopped without clipping. Classification is
  `HARD_NONNEGATIVITY_VALIDITY_STOP_AFTER_HORIZON`; physical Gate B remains
  false and the current closure is not eligible for grid/domain wall tests.

### Preserved campaign chronology

- The Mission-v3 pre-integration hard-invariant repair is locally complete.
  The production EXP-floor exponent is restricted to `n >= 1`; the new
  correlation flux is the exact discrete gradient flow of a declared
  logarithmic/correlation energy; its random-field energy derivative,
  nonpositive dissipation, exhausted-family floor behavior, and periodic
  balances pass.
- Nye compatibility now uses total signed mobile, locked, and wall content.
  Sign-preserving reservoir transfer leaves incompatibility unchanged, every
  positivity correction is ledgered, and accepted integrated steps require
  zero clipping. The face-slip reconstruction and adjoint traction projection
  close plastic work to roundoff.
- A four-family integrated Arrhenius--CDD state now owns mobile/locked/
  wall signed populations, authoritative face slip, cellwise plastic
  distortion, lattice rotation, and temperature. Physical and variational
  fluxes share one density/slip update; small-strain kinematics and external/elastic/plastic/correlation/
  line/heat ledgers share the same accepted interval. Homogeneous reduction,
  unloaded correlation-energy release, exact restart, no-label behavior, and
  energy closure pass. Bounded Arrhenius pair multiplication, annihilation,
  locking, unlocking, and wall capture now execute in the same interval and
  close separate line/Burgers ledgers. This historical checkpoint passed 226
  tests; the current repaired suite passes 236.
- Polygonization ordering is now reversible and selected by an explicit convex
  disordered/ordered wall free energy. Its energy decrease is ledgered, and an
  independent Frank--Bilby check now compares externally supplied kinematic
  rotation with wall Burgers content. This remains an isolated fixture.
- The IMEX logarithmic-diffusion step removes the demonstrated explicit CFL
  restriction without clipping or filtering. A factorized local screen over
  entropy sign, positive backstress/diffusion ratio, density, grid, domain,
  load state, physical-noise seed, and noise correlation length finds every
  eight-population mode damped through Nyquist and every nonlinear perturbation
  reduced. This rejects the present positive-energy, constant-mobility 1-D
  family as a spontaneous wall mechanism and selects state-dependent friction
  plus cross-family coupling as the next branch.
- Classification remains conservative: numerical integration is supported,
  but wall-mechanism, integrated scientific, DRX, ASB, and predictive claims
  remain false. The compact cross-platform verification of the earlier
  no-growth decision completed as job `55972295`. Archived job `55965192` was
  not rerun.
- First cross-platform attempt `20260912T231157Z-790035d-d2bb94`, Slurm
  `55972269`, is retained as a one-second packaging failure. The staged
  Arrhenius regression imported `asb_drx.analytical`, which was absent from the
  input manifest; no scientific calculation failed. Partial diagnostics were
  fetched, and the missing preserved dependency was added without changing
  equations or parameters.
- Corrected run `20260912T231254Z-a445735-14ca34`, Slurm `55972295`, passed
  all 37 targeted tests and the factorized screen in two seconds on
  `hpc3-14-07`. The verified result archive SHA-256 is
  `ccf8833252c84c31ca08feeff02837571ddeb949a019688872f55259754f4161`.
  Local and HPC classifications are identical and the maximum relative
  numeric difference is `1.075e-14`. Both platforms reject the positive-energy
  constant-mobility 1-D family as a wall mechanism while preserving it as a
  verified transport/relaxation operator.
- Primary vector-CDD literature rules out promoting the scalar locked-density
  fixture as the cross-family production law. The next implementation will
  carry signed vector density plus explicit junction-point/topology incidence
  so Frank's rule, line continuity, and total Nye content survive glissile and
  sessile reactions. State-dependent friction is a companion hypothesis; old
  DD multi-hit histories do not calibrate either closure.
- Mission v3 was adopted on 2026-09-12. Gates now classify claims and branch
  scientific work; they stop only use of results that fail hard invariants.
  The preserved Gate B1 no-go remains authoritative while isolated Arrhenius,
  CDD-flux, polygonization, phase, and ASB workstreams may continue.
- A versioned v3 Arrhenius kernel now separates positive EXP-floor activation
  enthalpy from independent signed activation entropy, detects negative free
  barriers, preserves forward-minus-reverse symmetry and nonnegative
  dissipation, and explicitly exposes the constant-entropy/attempt-frequency
  degeneracy. Five entropy/enthalpy families are staged for comparison.
- Two conservative staggered correlation-flux candidates are implemented:
  arithmetic face density and a logarithmic-mean discrete-chain-rule form.
  Both close periodic population balances, damp the Nyquist mode, and converge
  to the continuum low-mode symbol. A physical-spectrum perturbation generator
  replaces fixed box-mode seeding and is exactly restriction-consistent across
  same-domain grid refinements. These are isolated fixtures; full Gate B1
  kinematic integration remains unqualified.
- The isolated v3 polygonization fixture now carries mobile and wall-resolved
  signed content, distinct Arrhenius capture/climb/order channels, conservative
  sign-preserving capture, pairwise climb annihilation, explicit released line
  energy, and a simple-tilt Frank--Bilby angle derived from wall excess. It has
  no label-allocation surface and cannot yet claim an integrated LAGB or DRX
  event.
- Mission-v3 comparison run `20260912T180135Z-556e986-552929`, Slurm
  `55965192`, completed 25 targeted tests and the comparison in 7 seconds on
  `hpc3-14-06`. The fetched archive SHA-256 is
  `0b6f5ed2c9f8945d70e626aef69e93c691f655308711e375c8c42dbcaf3ed4d1`.
  Local and HPC decisions are identical; floating diagnostics agree to
  roundoff. The result reports `fixture_passed=true`,
  `numerical_verification_passed=true`, and correctly keeps
  `mechanism_supported=false`, `integrated_scientific_claim_supported=false`,
  and `predictive_validation_supported=false`.
- A follow-on staggered kinematics fixture now updates cell-centered signed
  populations and face-centered plastic slip from the identical signed line
  flux. It preserves `div(gamma_face)+b*kappa=0` algebraically, closes each
  periodic population balance, retains positivity under its declared step,
  and supplies a second-order face-to-cell slip reconstruction for future
  Gate-A calls. This establishes the coupling layout but does not yet replace
  the failed integrated Gate B1 solver.

- Gate B1 driven CDD development now has a modular spatial implementation and
  thirteen passing core tests. Every cell is dynamically coupled to Gate A stress,
  `F^p`, lattice orientation, MRSSP geometry, family slip/density evolution,
  and temperature. Positive/negative densities move through separate
  density-weighted fluxes; pair sources and locked transfers close signed and
  total-content ledgers. No wavelength, phase field, or grain-label state is
  present.
- The published strict one-dimensional `k_y=0` long-range projection has
  `T(k)=0`. Under that projection the full finite-differenced active-family,
  Nye-compatible Gate-A operator has a finite fastest mode 6 and the nonlinear
  128-point reference forms a GND-rich mode-10 precursor. The result does not
  establish a physical wall: the predeclared extended convergence and domain
  tests fail, so `physical_CDD_wall_gate_passed=false`.
- Local Gate B1 audit from pushed source `1df2973` records
  `fixture_passed=true`, `scientific_gate_passed=false`, and
  `physical_CDD_wall_gate_passed=false` in
  `output/gate_B1_driven_cdd.json` (SHA-256
  `dac88171a1d422c5c17d183d9fa3b2ab8147d67ae3846ff79c350f5b9a1f05ca`).
  The complete repository suite passes 188 tests. The result deliberately
  defers the domain/density/temperature robustness matrix after the full
  operator failure, and records no HPC3 submission.
- Subsequent derivation identified that the provisional nonzero `-i/k`
  long-range kernel contradicted the declared one-dimensional `k_y=0` reduction,
  for which the published `T(k)` is zero. Setting that term to zero—not fitting
  a wavelength—restores a full-compatible finite fastest mode 6 with growth
  `4.45e7 s^-1`. With resolution-invariant Fourier noise and mode 6 removed,
  both 64 and 128 points select nonlinear mode 10 (1.6 micrometers). Amplitude
  convergence still needs the 128/256 check. The extended single-job bundle is
  staged to test seeds, 13/16/19 micrometer domains, three densities,
  temperature/rate variation, wall width, unload behavior, and convergence.
- First extended attempt `20260912T004703Z-38c9807-77d57a`, Slurm `55949623`,
  passed all 37 staged tests and completed the expensive matrix, then failed
  during reverse unloading because a physically exhausted family reached
  exactly zero while the Gate A adapter required strictly positive density.
  Its partial archive checksum is verified and it contains no scientific
  result JSON. The adapter now evaluates an exhausted family at the already
  declared numerical density floor without altering the nonnegative physical
  populations; unload domain stops are also serialized rather than fatal.
- Corrected extended attempt `20260912T010453Z-9614a4e-d7ffb9`, Slurm
  `55949776`, passed 38 tests and completed the full numerical and unload
  matrix, but failed after plotting when its final provenance code called
  `git rev-parse` inside the intentionally git-free source archive. Its partial
  result archive and plots are checksum-verified; no JSON decision was emitted.
  Provenance now uses the source SHA embedded by the runner in `HPC3_RUN_ID`
  and cannot abort result serialization.
- Final single-job verification `20260912T011600Z-2c0bf13-edff97`, Slurm
  `55949943`, completed in 9 minutes 14 seconds on `hpc3-14-03`; all 38 staged
  tests passed and the fetched archive SHA-256 is
  `8d13be16d5b9f20cd78db0b9492d62efef502988cad0cda668555dc33a14b2ef`.
  Its machine result records `fixture_passed=true` and
  `scientific_gate_passed=false`. Timestep, three-seed, density-similitude,
  thermomechanical, unload, line-balance, and no-label checks pass. The
  128-to-256 grid check fails: wavelength changes 10%, GND/total changes 95%,
  total-density contrast changes 84%, orientation-gradient RMS changes 96%,
  and wall FWHM changes 620%; the 256-point field is classified as diffuse GND
  polarization. The 13/16/19 micrometer domains retain mode 10, making the
  apparent wavelength proportional to box length and failing the domain CV
  check. Gate C remains blocked; no polygonization or grain-label mechanism
  has been enabled.
- Postmortem linearization at 256 points finds modes 32--127 strongly damped at
  strains 0.005, 0.010, and 0.015, so the fine-grid mode-127 peak is not a
  predicted ultraviolet instability of the homogeneous continuum equations.
  The domain fixture also seeds identical Fourier mode numbers rather than an
  identical physical random field, biasing it toward box-scaled wavelengths.
  A trial two-state Rusanov face flux removed the algebraic odd/even null but
  became severely CFL-stiff by strain 0.011 and was rejected/reverted. The next
  Gate B1 revision must jointly define a continuum-consistent staggered
  correlation flux and a physical-domain-consistent perturbation ensemble;
  neither a numerical filter nor a fitted wavelength is accepted.

- Gate B0, not physical Gate B, is passed. The modular signed-transport state carries
  positive and negative mobile and locked populations for all four BCC
  Burgers families, signed Orowan flux, compatible plastic distortion and
  lattice rotation, a verified Nye tensor, and separately ledgered
  multiplication, annihilation, and locking. The DD collective scale is fixed
  to zero and nonzero baseline values are rejected.
- Gate B locally passes 16 targeted tests and the complete regression suite
  passes 176 tests. HPC3 run `20260910T203608Z-a4ac913-1fd2e3`, Slurm job
  `55923138`, passed the 16 tests in 3.171 seconds, completed in 10 seconds on
  `hpc3-14-02`, and was fetched with verified checksums.
- The imposed signed-polarization spinodal selects its prescribed `2.000 micrometer` mode
  in a `16 micrometer` periodic fixture. The wave has zero scalar
  total-density contrast, GND RMS `7.5997e13 m^-2`, GND/total ratio 0.03958,
  and a nonzero orientation gradient. A strong balanced total-density band has
  zero Nye content and is correctly rejected.
- Maximum accepted changes are 0.1521% for timestep refinement and
  `3.422e-7%` for 128-to-256 grid refinement. Local/HPC primary fields agree
  to `1.5e-13` or better on global scale; every boolean decision is identical.
  Scientific review on 2026-09-11 rejected the original physical-wall claim:
  the negative signed-density quadratic prescribes spontaneous instability,
  total density is fixed, the wavelength is an input, and the dynamics are not
  coupled to Gate A loading. Gate B1 must replace these features with driven
  positive/negative transport before polygonization can begin.

- Campaign execution resumed on 2026-09-08 from clean commit `111278e`.
  The substantive Gate A model now implements the published Bertin BCC-Ta
  multiplicative kinematics, four MRSSP pencil-glide modes, Orowan/drag-limited
  flow, augmented Kocks--Mecking density evolution, plastic spin, and distinct
  inactive-family relaxation. Its parameters are quarantined reference values,
  not production calibration.
- Nine Gate A tests pass: published stable/unstable orientation behavior,
  attractor direction, Burgers-family redistribution, inactive-family loss,
  exact no-spin/no-relaxation ablations, separate stress effects, bitwise
  restart, validity stops, and final timestep refinement. The complete local
  suite passes 160 tests.
- Substantive Gate A is passed on HPC3: run
  `20260909T044301Z-9ed6550-896a31`, Slurm job `55843151`, completed all nine
  targeted tests in 31 seconds and was fetched with verified checksums. The
  reference response uses the published BCC-Ta parameterization only inside
  its quarantine; it does not calibrate the production DRX/ASB model.
- Cross-platform comparison preserves every scientific decision. Seven of
  eight trajectories agree to about `1e-12`; near-symmetric `[101]` tension
  transiently differs by 0.592% in stress, 0.791% in density, and 0.00143
  degree in attractor angle, then reconverges below `5e-8` relative in final
  stress/density. This is within the declared 5% tolerance and is recorded,
  not hidden by output rounding.

- The v2 mission/physics addendum was adopted on 2026-08-29 from
  `CODEX_INDEPENDENT_DD_PF_DRX_ASB_CAMPAIGN_v2.md`, SHA-256
  `37142ee8029b4f461cbdbfa326c58632a7d4fd2988ff951af3ccef5e0d9dc2da`.
  It reclassifies the scalar/two-reservoir implementation below as a verified
  baseline rather than the production DRX architecture.
- Read-only DD organization audit finds event timing, contact, force, barrier,
  and clustering information, but no complete Burgers-sign/family, reaction,
  Nye/GND, wall-structure, or lattice-rotation observables. No DD-to-wall law
  or locked collective closure is scientifically identifiable.
- The five required addendum design documents and four quarantined A--D
  analytical/kinematic fixtures are now present. The fixtures test invariants
  without claiming the scientific gates: each deliberately reports
  `scientific_gate_passed=false` until its missing calibration/benchmark exists.
- The full local regression suite passes 151 tests after adding these fixtures.
- HPC3 preflight `20260829T115858Z-3a619e1-0b0f90`, Slurm `55649903`,
  completed four addendum fixture tests in one second on `hpc3-22-05` and was
  fetched with verified checksums. All fixtures pass their declared invariants;
  all four scientific gates were correctly false at that preflight stage. Gate
  A was subsequently developed and passed by the substantive Bertin model and
  HPC3 run above. No array or production sweep was launched for the preflight.

- Continuum flow now uses forward-minus-unloaded-reverse EXP-floor kinetics and
  a matrix-free backward-Euler antiplane solve. The two old stiff/unresolved
  high-rate cases reach 0.9 shear with no timestep halving.
- The full frozen-time finite-wavenumber operator is a verified 5 by 5 system
  for plastic shear, temperature, two density reservoirs, and binary order,
  including antiplane orientation, both storage-cap branches, phase heat, and
  recovery.
- Governing-equation analysis proved that storage alone cannot create a
  temperature/rate post-peak boundary. A single generic Arrhenius recovery law
  was therefore constrained by two declared arbitrary neutral points without
  retuning the DDD-derived flow parameters.
- The nonlinear near-boundary refinement passes the provisional 5% gate:
  0.873% maximum final timestep change and 0.00124% maximum 24-to-32-grid
  change. The condition remains nonlocalized, so band onset/width convergence
  is still not established.
- Sequential-hit and rearming-contact closures are underdispersed and cannot
  reproduce the audited high-density event CV above one. Shot-noise
  self-excitation remains a future ablation, not production physics, because
  its causal transfer and memory parameters are unidentified.
- A checkpointed physical embryo/orientation gate is implemented. Phase labels
  cannot become DRX without a promoted embryo that is distinct, supercritical,
  beyond the zero-excess radius, persistent, positively driven, and supported
  by a resolved pure phase field.
- The complete local suite currently passes 147 tests.
- Active v2 HPC3 run `20260828T232914Z-f5889ab-7987e9` / job `55646836`
  completed and fetched with verified checksums. All 27 sparse-matrix conditions
  reached 0.9 shear in 1,000 steps with no halving or unresolved point; none
  localized because the minimum active plastic fraction remained 0.999975.
  Maximum temperature was 1270.40 K and maximum matched-control excess was
  268.36 K. The preceding 26-second job `55646831` was cancelled before its
  first record to correct an obsolete output schema and is not used.

The 2026-08-27 DD-data no-go is superseded by the clarified scope: DD will not parameterize the model, and old programs/data are context only. Their inventory and HPC3 reproductions remain an audit record but are not new-model evidence or gates.

- Repository remote verified as `https://github.com/ukaiiaku-maker/ASB-DRX.git`.
- Isolated branch/worktree created from remote `main` at `a5dd798096e3896f319d314e8e4c60f5b277e589`.
- Source evidence root located at `/Users/sdillon/DRX-ASB` (33,358 files before filtering; about 20 GB).
- Six supplied files located at the evidence-root top level with upload suffixes normalized away.
- Legacy v32, v33 controls, and v34 sources/results located under `recrysyallization_PF-2D/shear_banding`.
- HPC3 aliases and runner verified. Existing unrelated local and Slurm campaigns were observed and left untouched.
- An EXP-floor barrier, independent-node rate law, inverse, and closed-form rate--temperature strength peak have been derived in `analytical_strength_derivation.md`.
- A material-agnostic analytical kernel and verification tests have been added under `src/asb_drx/` and `tests/`; small verification runs may execute locally and extended campaigns use HPC3.
- HPC3 run `20260828T120911Z-9d9e7c4-1bdf8a` (job `55637582`) passed all five analytical tests and was fetched with verified checksums.
- HPC3 identifiability run `20260828T131741Z-9caa154-ffbe2c` (job `55637767`) passed seven tests with verified retrieval. Strength-only peaks expose the exact scale compensation; independent peak density restores the tested five-parameter local rank.
- Literature motivates, but does not parameterize, a collective transparent-node hypothesis based on stress-transfer branching and multi-hit shot-noise memory.
- A 2026-08-28 primary-literature refresh confirms that repeat-pass conditioning is plausible and that glissile/shearable junctions may still harden. It does not provide a transferable multi-hit factor, stress-transfer kernel, or reset time; the collective extension remains outside the baseline.
- Complete single-glider Taylor DDD context was located in `/Users/sdillon/Taylor_DDD` and `/Users/sdillon/Taylor_DDD_arrhenius_native`. Commit `fb7610b` contains a passing native ExaDiS persistent-contact gate; later EXP-floor campaigns contain event histories across density and temperature. These are now structural evidence, not parameter sources.
- HPC3 structural run `20260828T130729Z-1a147e0-710710` (job `55637740`) passed both collective-context tests with verified retrieval. Higher density has substantially more multi-hit clustering, but all sampled native one-step contact operators have zero spectral-radius proxy and only 11 redistribution samples exist. No causal collective law or production parameter is established.
- HPC3 thermodynamic run `20260828T132308Z-07d589a-1e7eac` (job `55637784`) passed six material-agnostic tests with verified retrieval: discrete variational consistency, monotone relaxation, conservative reservoir transfer, exact work partition, nucleus-limit signs, and range rejection.
- HPC3 spatial run `20260828T132810Z-a4a0bf0-5c7bfe` (job `55637801`) passed nine tests with verified retrieval. The diffuse 2-D nucleus shrinks/grows on the correct sides of the derived critical radius, final grid/timestep changes are 0.384%/0.0021%, and the complete current limited state restarts bitwise exactly.
- HPC3 material-point run `20260828T133324Z-6920914-866ee2` (job `55637814`) passed ten tests with verified retrieval. Finite elastic loading, EXP-floor plastic flow, stored line energy, and residual heat close the incremental work ledger exactly without a tuned heat fraction.
- HPC3 shear-layer run `20260828T133943Z-ae2fdf9-d7f5b0` (job `55637821`) passed fourteen tests with verified retrieval. The common-stress layer reduces to the material point, conserves/damps heat correctly, closes both ledgers, and restarts exactly. An earlier unstable explicit-diffusion run was rejected and corrected by enforcing the Fourier bound.
- HPC3 grain-metric run `20260828T134840Z-5e8fabc-192eaf` (job `55637844`) passed ten tests with verified retrieval. Allocated labels, periodic topology components, resolved support, physical grains, and promoted recrystallized grains are distinct metrics; promotion requires resolved persistent support, lineage, and symmetry-reduced misorientation, while retirement preserves provenance. The preceding packaging-only failure omitted an imported module and executed no grain test.
- HPC3 multi-order run `20260828T135602Z-d1de48a-eb7ef1` (job `55637866`) passed all seventeen multi-order/grain tests with verified retrieval. The projected Onsager evolution preserves the pointwise simplex to roundoff, decreases the declared free energy, leaves an exactly pure parent unchanged, is label-permutation symmetric, recovers the analytical circular-nucleus growth signs, restarts exactly for its complete current state, and feeds the tracker without allocating labels.
- HPC3 stored-energy coupling run `20260828T140503Z-8fb6236-7dab82` (job `55637879`) passed fourteen tests with verified retrieval. The grain-growth driving force is the explicit line-energy/density difference; common density offsets do not alter binary dynamics; a pure parent neither resets nor heats; and every free-energy decrement closes between stored energy, interface/order energy, and heat.
- HPC3 candidate-decision run `20260828T141129Z-e01a0ce-969f27` (job `55637888`) passed six tests with verified retrieval. The classical cylindrical nucleus has the analytical stationary barrier at `R_c` and zero excess at `2R_c`; the Poisson event probability has the required temperature/driving monotonicity; and resolution, subcriticality, misorientation, and external-draw rejection remain distinct. The kernel never allocates a label.
- HPC3 coupled-ledger run `20260828T141957Z-c9708bb-f4aa93` (job `55637907`) passed all twenty-five material-point, phase, stored-energy, and coupled tests with verified retrieval. The binary aggregate reduces exactly to each isolated limit, uses one accepted interval, preserves a zero child, restarts exactly, and closes external work across elastic, stored, interface/order, and both heat channels without double counting. An earlier run exposed an over-strict temperature-roundoff gate and is retained as failed evidence.
- HPC3 spatial-coupled run `20260828T142706Z-3224ef1-bdf10f` (job `55638019`) passed fifteen tests with verified retrieval. Local temperature, two density fields, and two order parameters share a periodic 2-D grid under common stress; conduction, global energy closure, zero-child invariance, isolated-limit reduction, and exact restart pass.
- HPC3 localization-metric run `20260828T143238Z-ca7ab69-74cc1d` (job `55638898`) passed all six tests with verified retrieval. Classification now requires simultaneous plastic concentration, matched-control temperature excess, post-peak softening, resolved finite width, persistence, and joint onset/width refinement. The preceding run `20260828T143111Z-d608655-a7aea7` (job `55638861`) is retained as failed evidence: its negative fixture accidentally contained the required three consecutive qualifying states; the classifier behaved correctly and the fixture was corrected.
- HPC3 mechanism-ladder run `20260828T144000Z-c9dc4ab-7b33f9` (job `55640278`) passed seventeen tests with verified retrieval. Six declared common-equation variants plus thermal cases' isothermal twins preserve phase-disabled fields, route isothermal heat to an explicit bath, and close their ledgers. All six are nonlocalizing controls: minimum active fraction is about `0.907`, maximum matched-control temperature excess is `0.125 K`, and maximum softening is zero.
- HPC3 stability run `20260828T144457Z-535a0ff-662a0e` passed ten tests with verified retrieval. The marker-owning job is `55640502`; job `55640458` is an explicitly retained duplicate caused by a silent staging receipt during a `/pub` metadata stall. Analytical EXP-floor tangents, the finite-wavenumber thermal/storage Jacobian, conduction shift, nonlinear finite-difference closure, and impossible-storage rejection pass. The generic state has a positive `~4.807 s^-1` density-storage mode but strongly damped thermal diagonals; it is not thermal-ASB evidence.
- The user authorized reuse of the complete single-glider DDD constants as a generic fixture and an arbitrary analytical boundary, without matching a materials class. HPC3 run `20260828T151810Z-d31c6a4-d3af25` (job `55641106`) passed fourteen exact-mapping and upstream tests with verified retrieval.
- The arbitrary boundary is preregistered as `rho=rho_peak(T, rate)` from the independent EXP-floor law. At the DDD rate it spans `4.4117e15` to `8.5383e15 m^-2` over 1050 to 850 K. The driver's hard-coded `1e18 m^-2` field is excluded; the observed monotone DDD response through `3e16 m^-2` remains a structural mismatch, not a fitted correction.
- HPC3 run `20260828T152809Z-db81077-68e54d` (job `55641308`) passed all fifteen boundary and mapping tests with verified retrieval. The frozen surface covers 850--1050 K and `4.5`, `450`, and `45000 s^-1`; future spatial cases are preregistered at density ratios `0.5`, `1`, and `2` relative to the peak.
- The first boundary-spatial preflight `20260828T153642Z-c48cad2-f5273f` (job `55641810`) failed before simulation because one upstream test dependency was omitted from staging and an exact-equality assertion rejected a `~3e-16` relative roundoff difference. It is retained as failed packaging evidence; no equation or case definition changed.
- Corrected single-job smoke `20260828T154200Z-9d7ed90-e57dd1` (job `55642217`) passed 21 tests and fetched with verified checksums. At 950 K, `45000 s^-1`, and `rho/rho_peak=1`, 16²/32² final stress and maximum-temperature changes were `5.04e-9` and `1.10e-6`, below the 5% provisional target. The case reached 0.09 shear and `26.74 K` matched-control excess but remained spatially uniform (`f_q~0.999999`) and nonlocalizing.
- HPC3 run `20260828T155312Z-266dda8-d6b092` (job `55642641`) passed six isolated periodic-antiplane tests with verified retrieval. The Fourier projection recovers uniform common stress, relaxes longitudinal modes, produces equilibrated transverse/diagonal redistribution, and closes the exact midpoint work identity to `7.45e-9 J m^-3` (`~3.7e-15` relative). It is not yet constitutively coupled.
- Preceding antiplane run ID `20260828T154906Z-266dda8-f8fc4a` is retained as failed orchestration evidence: jobs `55642418` and `55642419` were both created after a metadata-stalled receipt and collided while finalizing the same remote path. All six tests passed in the shared log, but no scientific result is accepted from that run.
- First-pass legacy audit and candidate thermodynamic architecture drafted.
- Full evidence inventory completed: 33,358 files, 21,536,785,369 bytes, no hash errors.
- Campaign-specific HPC3 smoke job `55633650` completed, fetched, and checksum-verified.
- Exact-source HPC3 legacy controls completed and were fetched with verified checksums: v32 job `55633674`, v33 job `55633691`, and v34 job `55633694`.
- v32 reproduces all finite numerical diagnostics within `atol=1e-10`, `rtol=1e-9`; 18 degenerate zero-variance correlations differ only as `NaN` versus roundoff near `1e-17`.
- v33 reproduces the false-grain mechanism but not its exact trajectory: 165 hazard births and 177 allocated labels versus 12 unchanged topology components. These are rejected as physical grains.
- v34 produces zero active, new, or promotable candidates and zero births throughout. This reproduces the stored zero-candidate failure, not the requested candidate-without-promotion premise. Its detailed trajectory is not numerically reproducible from the supplied source/configuration.

## Active scientific boundaries

1. The analytical kernel and first optimizer-identifiability gate pass on HPC3. Physical fitting must not estimate both stress scale and attempt rate from strength peaks alone; it requires independent peak density or an authoritative fixed scale.
2. No materials-class calibration is sought in the present generic campaign; the authorized DDD fixture must not be described as a physical calibration.
3. The collective extension is an ablation, not baseline physics. Existing DDD histories show density-dependent clustering but do not resolve causal parentage or a nonzero feedback operator; higher-cadence evidence is required before reconsideration.
4. The periodic antiplane operator is now integrated with net EXP-floor flow, storage, recovery, heat, and phase evolution. The model is generic and isotropic; physical boundaries and material-scaled anisotropic properties are outside the present scope.
5. The full finite-wavenumber linearization and nonlinear timestep/grid refinement pass, but the v2 matrix supplies no candidate band. Onset and width convergence therefore cannot yet be claimed.
6. A stateful physical embryo gate and energy-release ledger pass, but automatic stochastic embryo sampling and phase-field allocation are absent because their attempt prefactor and orientation distribution are unconstrained.
7. The strict ASB classifier has evaluated all 27 coupled v2 trajectories. Every trajectory fails plastic concentration, so the result is a verified no-localization baseline rather than an ASB boundary.
8. The generic recovery parameters define an arbitrary analytical screen only. They must not be interpreted as material properties or a calibrated DRX/ASB transition.
9. A collective closure remains an ablation. Existing event histories cannot identify a signed stress-transfer kernel, memory time, or causal parentage.
10. A predictive campaign would still require a defensible finite-wavelength localization mechanism, physical embryo-rate inputs, uncertainty propagation, and external validation.

## Remaining interpretation limits

The generic campaign may proceed without selecting Fe, Cr, or another materials class. A predictive materials claim would still require an authoritative dataset. “Post-peak collective candidate” is deliberately broader than a particular transparent-junction mechanism; a specific transfer kernel and reset law remain research questions rather than boundary inputs.
