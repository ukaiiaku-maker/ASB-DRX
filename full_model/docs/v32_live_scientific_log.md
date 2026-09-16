# V32 live scientific log

## 2026-09-16 restart audit

- Local and remote canonical branch both resolve to V31 evidence checkpoint `8df28e8e3b47eea9f04c154db13f2df468c2c096`; the canonical worktree is clean.
- Immutable V31 calculation source remains `c643afe00b4ea6f7f30e021675d7648ecdb6422c`.
- V30/V31 raw evidence and decision records are frozen.
- Unrelated Slurm job `55950433` is running and remains untouched; no V32 Slurm job exists at restart.
- Front continuation audit: six normal completions, one conservative `FRONT_COMPONENT_SPLIT` terminal, two running 128-square cases, and two near-equal 128-square cases not yet started. Exact-equal and mobility-off controls will not be replayed.
- The 64-square positive near-equal terminal occurs at step 135 after 135 accepted transactions; its cumulative maximum line-closure magnitude is `1.21e-19 m` and signed-Burgers closure is exact. The rejected topology event itself commits no heat, line, or signed content.
- ASB anchor audit: all four 128-square cases are live at step 2600 of 5000 with 27 restart files per case after about 2.4 hours. The anchor is therefore adequately checkpointed and is projected to close within the local campaign horizon.
- Three independent scientific loops are active: topology-event/front kinetics, Mura work-budget repair, and physical-ASB classification.
