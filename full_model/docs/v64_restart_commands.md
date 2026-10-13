# V64 exact restart commands

The terminal physical/numerical source is `b579a1623361a93e68ae26aec27edbefc9a91a3d`.
Run source-bound restarts from a worktree whose checked-out `HEAD` is exactly
that commit, for example the retained detached worktree
`/Users/sdillon/HPC3/worktrees/asb-drx-v64-final-source-b579a16`.

To reproduce the last accepted interval and normal terminal result:

```bash
cd /Users/sdillon/HPC3/worktrees/asb-drx-v64-final-source-b579a16
PYTHONPATH=src:. python -m full_model.analysis.run_v64_adaptive_production \
  --out /Users/sdillon/HPC3/campaigns/asb-drx-v64-20261012/n48_origin_final_allocation_to1p03us_b579a16 \
  --resume /Users/sdillon/HPC3/campaigns/asb-drx-v64-20261012/n48_origin_terminal_from2456_b579a16/checkpoint_adaptive_002457.npz \
  --expected-resume-sha256 1a66604c813bff2d044910c12daa66bf3fbfb8dc7da3fd0b33ae4c3d166cf165 \
  --target-time 1.03e-6 \
  --minimum-step 6.103515625e-12 --maximum-step 6.25e-9 \
  --checkpoint-every 1 --source-commit b579a16
```

To continue from the terminal checkpoint, choose a new physical target time
greater than `1.0300000000000003e-6` and use:

```bash
cd /Users/sdillon/HPC3/worktrees/asb-drx-v64-final-source-b579a16
PYTHONPATH=src:. python -m full_model.analysis.run_v64_adaptive_production \
  --out <new-output-directory> \
  --resume /Users/sdillon/HPC3/campaigns/asb-drx-v64-20261012/n48_origin_final_allocation_to1p03us_b579a16/checkpoint_adaptive_002585.npz \
  --expected-resume-sha256 2bbf987b6e6dbbf603c518686b90812a91c821715874bd580a0367dbeb1b1667 \
  --target-time <seconds> \
  --minimum-step 6.103515625e-12 --maximum-step 6.25e-9 \
  --checkpoint-every 8 --source-commit b579a16
```

The exact target shortened the final interval; the checkpoint retains a
proposed step of `2.418041251250174e-11 s`.
Do not relabel the fixed prepared four-grain network as spontaneous nucleation,
and do not treat the 1.0300-microsecond endpoint as a strict-ASB lifetime test.
