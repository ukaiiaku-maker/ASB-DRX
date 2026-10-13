# V65 restart and reproduction commands

These commands identify the actual V65 production method.  Run them from a
checkout containing production source commit
`6debbe0813ca0c365abcf473e5106cb90dbdd75a`.  A restart output directory must
be new; never point `--out` at the preserved evidence directory.

## Continue the promoted production trajectory

The recorded 1.035 microsecond parent is
`runs/candidate_to1p035us_0637685/checkpoint_adaptive_002644.npz` in the V65
delivery.  Its SHA-256 is
`7672cb556586bbbf69547910e6239c41ef2ddb2bebeead9760741a27ab30f34e`.

```sh
PYTHONPATH=src:. python -m full_model.analysis.run_v64_adaptive_production \
  --out <new-output-directory> \
  --resume <delivery-root>/runs/candidate_to1p035us_0637685/checkpoint_adaptive_002644.npz \
  --expected-resume-sha256 7672cb556586bbbf69547910e6239c41ef2ddb2bebeead9760741a27ab30f34e \
  --target-time 1.25e-6 \
  --minimum-step 6.103515625e-12 \
  --maximum-step 6.25e-9 \
  --checkpoint-every 64 \
  --source-commit 6debbe0813ca0c365abcf473e5106cb90dbdd75a \
  --numerical-transition v65_durable_long_run \
  --warm-start-front-factor \
  --proposal-policy error_responsive \
  --compact-stdout
```

The unchanged working contract performs a complete coarse path and two
complete half paths, publishes only the fine path, and accepts only when every
registered metric is below its declared 2% relative/absolute threshold.

## Reproduce the fixed-proposal comparator

Use the reconstructed V64 terminal checkpoint (SHA-256
`2bbf987b6e6dbbf603c518686b90812a91c821715874bd580a0367dbeb1b1667`),
restore its ordinary pre-horizon proposal, omit both V65 optimization flags,
and target 1.035 microseconds:

```sh
PYTHONPATH=src:. python -m full_model.analysis.run_v64_adaptive_production \
  --out <new-reference-output-directory> \
  --resume <delivery-root>/reconstructed_v64/checkpoint_adaptive_002585.npz \
  --expected-resume-sha256 2bbf987b6e6dbbf603c518686b90812a91c821715874bd580a0367dbeb1b1667 \
  --target-time 1.035e-6 \
  --restart-proposed-step 7.450580596913911e-11 \
  --minimum-step 6.103515625e-12 \
  --maximum-step 6.25e-9 \
  --checkpoint-every 8 \
  --source-commit f82d3a61d3fbeb189786c473fcd29f6c482a2ac7 \
  --numerical-transition v65_terminal_horizon_restore
```

The comparator and promoted calculation share the same physical parent,
physical coefficients, spatial grid, load, and error tolerances.  Their
1.030--1.035 microsecond work--precision comparison is recorded in
`analysis/work_precision_1p030_to1p035us.json`.
