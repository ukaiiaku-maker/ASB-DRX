# V31 Workstream A change and test log

Base checkpoint: `67c8b11`. V30 replay source: fetched archive from source
`675d74183baf2043ad0f7c055fe6e3370435ae65`. The archive was read only.

## Production changes

- Added a periodic marching-triangle contour graph with persistent component
  IDs, winding, interface length, periodic centroid, endpoint/closed status,
  active-window relation, and receiver-oriented tangent/normal.
- Replaced ray-crossing ownership in the coupled production adapter with
  pair-local `phi = eta_B - eta_A` component matching by winding,
  sampled-contour overlap, periodic distance, and length change.
- Replaced ray-displacement sweep with inverse-tanh cut-cell receiver area.
  Ray crossings remain diagnostic fields only.
- Added named fail-closed split, merge, consumption, annihilation,
  active-window, and identity-loss terminals. A terminal rolls phase and
  physical state back, records both generic and topology-specific JSON, and
  writes an immediate restart checkpoint.
- Persisted component topology in the coupled-front checkpoint and added
  explicit migration for old V1 checkpoints lacking topology state.

## Tests

- `python -m py_compile` on the topology, production adapter, and production
  driver: pass.
- Focused front suite: 80 passed, 0 failed in 20.13 s.
- New V31 manufactured/production suite alone: 15 passed, 0 failed.
- Existing V30 coupled-front regression: 31 passed, 0 failed.
- Full repository test boundary (`PYTHONPATH=src python -m pytest -q tests`):
  599 passed, 0 failed in 197.25 s.

## Representative failed-checkpoint replay

The V30 64 and 128 favorable cases had both failed with
`ValueError: front topology changed: crossing counts differ`. Each fetched
step-0 checkpoint was replayed locally through step 40 with the V31 production
owner.

- 64 grid: return code 0; 41/41 front proposals accepted; one persistent
  component; zero topology events; maximum line closure
  `2.875e-20 m`.
- 128 grid: return code 0; 4 accepted and 37 thermodynamic-direction
  rejections; one persistent component; zero topology events; maximum line
  closure `8.027e-20 m`.

Thus the original ray-count exception is removed at both resolutions. This is
a local hard-gate pass, not a claim that the unrun V31 production long horizon
has qualified.
