#!/usr/bin/env bash
set -uo pipefail

solver_pid="${1:?solver PID required}"
campaign="/Users/sdillon/HPC3/campaigns/asb-drx-v65-20261013"
worktree="/Users/sdillon/HPC3/worktrees/asb-drx-full-v65-20261013"
run="$campaign/runs/production_1p035_to1p25us_6debbe0"
live="$campaign/analysis/live_monitor"
status="$campaign/repository_sync_status.json"

while kill -0 "$solver_pid" 2>/dev/null; do
  sleep 60
done

ready=0
for _ in $(seq 1 30); do
  if [[ -s "$run/result.json" && -s "$live/live_checkpoint_latest.json" && \
        -s "$live/component_identity_monitor_status.json" && \
        -s "$live/peak_forecast_monitor_status.json" ]] && \
     python - "$run/result.json" "$live/live_checkpoint_latest.json" \
       "$live/component_identity_monitor_status.json" \
       "$live/peak_forecast_monitor_status.json" <<'PY'
import json, sys
result, latest, identity, peak = (
    json.load(open(path)) for path in sys.argv[1:])
raise SystemExit(0 if (
    identity.get("status") == "COMPLETE"
    and peak.get("status") == "COMPLETE"
    and latest.get("checkpoint_sha256") == result.get("checkpoint_sha256")
) else 1)
PY
  then
    ready=1
    break
  fi
  sleep 10
done

cd "$worktree" || exit 2
sync_code=1; test_code=1; commit_code=1; push_code=1
if [[ "$ready" -eq 1 && -z "$(git status --porcelain)" ]]; then
  python -m full_model.analysis.sync_v65_terminal_repository \
    --manifest full_model/verification/v65_campaign_manifest.json \
    --execution-record full_model/docs/v65_execution_record.md \
    --result "$run/result.json" \
    --latest-classification "$live/live_checkpoint_latest.json" \
    --identity-status "$live/component_identity_monitor_status.json" \
    --peak-forecast-status "$live/peak_forecast_monitor_status.json"
  sync_code=$?
  python -m pytest -q tests/test_v65_terminal_repository_sync.py
  test_code=$?
  git diff --check
  if [[ "$sync_code" -eq 0 && "$test_code" -eq 0 ]]; then
    git add full_model/verification/v65_campaign_manifest.json \
      full_model/docs/v65_execution_record.md
    git commit -m "record V65 terminal production provenance"
    commit_code=$?
    if [[ "$commit_code" -eq 0 ]]; then
      git push origin exp/full-v65-efficient-controlled-evolution-20261013
      push_code=$?
    fi
  fi
fi

CAMPAIGN="$campaign" READY="$ready" SYNC_CODE="$sync_code" \
TEST_CODE="$test_code" COMMIT_CODE="$commit_code" PUSH_CODE="$push_code" \
python - <<'PY'
import json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
campaign = Path(os.environ["CAMPAIGN"])
codes = {name: int(os.environ[name]) for name in (
    "SYNC_CODE", "TEST_CODE", "COMMIT_CODE", "PUSH_CODE")}
payload = {
    "schema": "asb-drx-v65-repository-sync-status-v1",
    "status": ("COMPLETE" if int(os.environ["READY"]) == 1
               and all(value == 0 for value in codes.values()) else "FAILED"),
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "terminal_evidence_ready": bool(int(os.environ["READY"])),
    "step_exit_codes": codes,
    "repository_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
}
temporary = campaign/"repository_sync_status.json.tmp"
temporary.write_text(json.dumps(payload, indent=2)+"\n")
temporary.replace(campaign/"repository_sync_status.json")
PY
