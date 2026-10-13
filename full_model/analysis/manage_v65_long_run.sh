#!/usr/bin/env bash
set -uo pipefail

solver_pid="${1:?solver PID required}"
campaign="/Users/sdillon/HPC3/campaigns/asb-drx-v65-20261013"
worktree="/Users/sdillon/HPC3/worktrees/asb-drx-full-v65-20261013"
run="$campaign/runs/production_1p035_to1p25us_6debbe0"
candidate="$campaign/runs/candidate_to1p035us_0637685"
reconstructed="$campaign/reconstructed_v64"
analysis="$campaign/analysis"
logs="$campaign/logs"
mkdir -p "$analysis" "$logs"

while kill -0 "$solver_pid" 2>/dev/null; do
  sleep 60
done

cd "$worktree" || exit 2
if [[ ! -s "$run/result.json" ]]; then
  CAMPAIGN="$campaign" RUN="$run" PID_VALUE="$solver_pid" python - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
payload = {
    "schema": "asb-drx-v65-finalizer-status-v1",
    "status": "SOLVER_ENDED_WITHOUT_RESULT",
    "solver_pid": int(os.environ["PID_VALUE"]),
    "run": os.environ["RUN"],
    "observed_utc": datetime.now(timezone.utc).isoformat(),
}
Path(os.environ["CAMPAIGN"], "finalizer_status.json").write_text(
    json.dumps(payload, indent=2)+"\n")
PY
  exit 3
fi

PYTHONPATH=src:. python -m full_model.analysis.summarize_v64_adaptive_history \
  --run "$run" --output "$analysis/production_to1p25us_summary.json" \
  > "$logs/finalize_summary.log" 2>&1
summary_status=$?

PYTHONPATH=src:. python -m full_model.analysis.merge_adaptive_histories \
  --history "$reconstructed/histories/terminal_through2184.json" \
  --history "$reconstructed/histories/terminal_extension.json" \
  --history "$reconstructed/histories/final_allocation.json" \
  --history "$candidate/history.json" --history "$run/history.json" \
  --output "$analysis/merged_controlled_history.json" \
  > "$logs/finalize_merge.log" 2>&1
merge_status=$?

PYTHONPATH=src:. python -m full_model.analysis.postprocess_v65_controlled_episode \
  --run "$reconstructed" --run "$candidate" --run "$run" \
  --history-file "$reconstructed/histories/terminal_through2184.json" \
  --history-file "$reconstructed/histories/terminal_extension.json" \
  --history-file "$reconstructed/histories/final_allocation.json" \
  --output "$analysis/final_asb_through1p25us.json" \
  > "$logs/finalize_asb.log" 2>&1
asb_status=$?

terminal_checkpoint="$(python - <<'PY'
import json
print(json.load(open('/Users/sdillon/HPC3/campaigns/asb-drx-v65-20261013/runs/production_1p035_to1p25us_6debbe0/result.json'))['checkpoint'])
PY
)"

if [[ -z "$terminal_checkpoint" || ! -s "$terminal_checkpoint" ]]; then
  terminal_checkpoint_status=1
else
  terminal_checkpoint_status=0
fi

PYTHONPATH=src:. python -m full_model.analysis.analyze_v65_grain_selection \
  --origin "$reconstructed/trajectory_checkpoints/n48_analytic_origin_bab4c95.npz" \
  --checkpoint "$reconstructed/trajectory_checkpoints/checkpoint_adaptive_000454.npz" \
  --checkpoint "$reconstructed/trajectory_checkpoints/checkpoint_adaptive_000886.npz" \
  --checkpoint "$reconstructed/checkpoint_adaptive_002585.npz" \
  --checkpoint "$candidate/checkpoint_adaptive_002644.npz" \
  --checkpoint "$terminal_checkpoint" \
  --history "$reconstructed/histories/terminal_through2184.json" \
  --history "$reconstructed/histories/terminal_extension.json" \
  --history "$reconstructed/histories/final_allocation.json" \
  --history "$candidate/history.json" --history "$run/history.json" \
  --joint-front-diagnostics \
  --output "$analysis/final_grain20_selection.json" \
  > "$logs/finalize_grain.log" 2>&1
grain_status=$?

PYTHONPATH=src:. python -m full_model.analysis.render_v64_controlled_fields \
  --run "$reconstructed" --run "$candidate" --run "$run" \
  --output-dir "$campaign/visuals" --maximum-frames 24 \
  > "$logs/finalize_render.log" 2>&1
render_status=$?

PYTHONPATH=src:. python -m pytest -q \
  tests/test_v64_coupled_interval_control.py \
  tests/test_v64_complete_multigrain_interval.py \
  tests/test_v64_profile_tracking.py tests/test_v64_rate_attribution.py \
  tests/test_full_model_v58_multigrain_production.py \
  tests/test_v59_physical_asb.py tests/test_v65_grain_response_metrics.py \
  tests/test_v65_merge_adaptive_histories.py tests/test_v65_delivery.py \
  > "$logs/focused_tests_final.log" 2>&1
focused_status=$?

PYTHONPATH=src:. python -m pytest -q \
  > "$logs/canonical_tests_final.log" 2>&1
canonical_status=$?

CAMPAIGN="$campaign" RUN="$run" PID_VALUE="$solver_pid" \
SUMMARY_STATUS="$summary_status" MERGE_STATUS="$merge_status" \
ASB_STATUS="$asb_status" TERMINAL_CHECKPOINT_STATUS="$terminal_checkpoint_status" \
GRAIN_STATUS="$grain_status" RENDER_STATUS="$render_status" \
FOCUSED_STATUS="$focused_status" CANONICAL_STATUS="$canonical_status" \
python - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
campaign = Path(os.environ["CAMPAIGN"])
files = [
    campaign/"analysis/production_to1p25us_summary.json",
    campaign/"analysis/merged_controlled_history.json",
    campaign/"analysis/final_asb_through1p25us.json",
    campaign/"analysis/final_grain20_selection.json",
    campaign/"visuals/render_manifest.json",
    campaign/"runs/production_1p035_to1p25us_6debbe0/result.json",
]
inventory = {}
for path in files:
    if path.is_file():
        inventory[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
step_exit_codes = {
    "summary": int(os.environ["SUMMARY_STATUS"]),
    "merge": int(os.environ["MERGE_STATUS"]),
    "asb": int(os.environ["ASB_STATUS"]),
    "terminal_checkpoint": int(os.environ["TERMINAL_CHECKPOINT_STATUS"]),
    "grain": int(os.environ["GRAIN_STATUS"]),
    "render": int(os.environ["RENDER_STATUS"]),
    "focused_tests": int(os.environ["FOCUSED_STATUS"]),
    "canonical_tests": int(os.environ["CANONICAL_STATUS"]),
}
all_steps_passed = all(code == 0 for code in step_exit_codes.values())
all_artifacts_present = len(inventory) == len(files)
payload = {
    "schema": "asb-drx-v65-finalizer-status-v1",
    "status": ("COMPLETE" if all_steps_passed and all_artifacts_present
               else "POSTPROCESS_OR_TEST_FAILURE"),
    "solver_pid": int(os.environ["PID_VALUE"]),
    "run": os.environ["RUN"],
    "focused_test_exit_code": int(os.environ["FOCUSED_STATUS"]),
    "canonical_test_exit_code": int(os.environ["CANONICAL_STATUS"]),
    "step_exit_codes": step_exit_codes,
    "all_required_artifacts_present": all_artifacts_present,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "inventory": inventory,
}
(campaign/"finalizer_status.json").write_text(json.dumps(payload, indent=2)+"\n")
PY
