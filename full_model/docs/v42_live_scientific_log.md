# V42 live scientific log

- Restart audit: branch `exp/full-v34-recovery-v1` was clean at local and
  remote `4b6da0d`; scientific regression source `25d4906` remains distinct.
  No local campaign process was active.  The literal `hpc3` hostname did not
  resolve; the configured alias is `uci-hpc3`.  The corrected scheduler audit
  showed no live jobs, retained completed job `56151298`, and left unrelated
  failed job `56132213` untouched.  Compact and sub-30-minute matched work was
  placed locally.
- The first retained-state legacy topology increment reproduced the hard
  failure: source offset `0.08146`, line continuity `0.08072`, and an
  ordered-gradient increase of about `0.023596 J/m` relative to its matched
  control.
- Commit `ee4b805` replaced production publication of unrepresented local
  reorientation/junction sources with an atomic, complete-energy-guarded
  reservoir conversion.  Reorientation now fails closed pending persistent
  endpoint/swept-surface geometry.
- The repaired retained-state increment closes source offset and line
  continuity at `1.09e-14` and `9.05e-15`.  Its nonthermal state is bitwise
  equal to the control; released energy produces at most `0.01204 K` local
  heating.  It creates no qualified boundary.
- The 5% orientation extrema are adjacent cells at `+5.664 deg` and
  `-4.116 deg`.  No coordinate wrapping is present; p95-p05 is `0.1132 deg`.
  This is retained as an underresolved local dipolar extremum, not a LAGB.
- A same-internal-state front comparison retains zero publications at mean
  shear `0.01`.  A prepared zero-mean-shear counterfactual admits four of
  sixteen tested signed trials.  This confirms that the stationary front is
  controlled by elastic mismatch in the represented history, not a universal
  absence of defect-energy drive.  It is not labeled a free unloading path.
- Matched n128/n192 calculations were launched locally for the original
  `62.5 us` physical horizon.  Both use source commit `ee4b805` and eight
  `7.8125 us` common macros.
- The matched 62.5-us calculations completed with every common clock closed.
  Wall times were 442 s (n128) and 949 s (n192).  The n128/n192 differences
  are 38.06% in curl-Nye RMS, 50.22% in maximum tensor norm, 29.53% in the L1
  tensor-norm integral, and 70.60% in ordered line.  The mode-24 band differs
  by 40.89%; the nearly full common coefficient square differs by 132.31%.
  Classification remains `UNRESOLVED_SPATIAL_SCALE`.  The V41 one-macro
  7.31% value was a short-window result, not convergence at this horizon.
- The repaired 5.01% continuation was locally profiled for 114 s.  It reached
  step 22,620 and strain 5.000604% with hard invariants intact, then stopped
  on an exact signal checkpoint because projected local runtime exceeded two
  hours.  Pushed source `17f63ee`, the checkpoint, and matching checksums were
  staged for HPC3 job `56166765`; no array or broad sweep was submitted.
