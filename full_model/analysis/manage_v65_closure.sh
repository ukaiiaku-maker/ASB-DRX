#!/usr/bin/env bash
set -uo pipefail

finalizer_pid="${1:?finalizer PID required}"
campaign="/Users/sdillon/HPC3/campaigns/asb-drx-v65-20261013"
worktree="/Users/sdillon/HPC3/worktrees/asb-drx-full-v65-20261013"
analysis="$campaign/analysis"
logs="$campaign/logs"
status="$campaign/closure_status.json"
mkdir -p "$analysis" "$logs"

write_failure() {
  STATUS_VALUE="$1" STATUS_PATH="$status" python - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ["STATUS_PATH"])
temporary = path.with_name(path.name+".tmp")
temporary.write_text(json.dumps({"status": os.environ["STATUS_VALUE"]})+"\n")
temporary.replace(path)
PY
}

while kill -0 "$finalizer_pid" 2>/dev/null; do
  sleep 60
done

cd "$worktree" || exit 2
if [[ -n "$(git status --porcelain)" ]]; then
  write_failure "DIRTY_WORKTREE_PREVENTED_CLOSURE"
  exit 3
fi
if [[ ! -s "$campaign/finalizer_status.json" ]] || \
   [[ "$(python -c 'import json; print(json.load(open("/Users/sdillon/HPC3/campaigns/asb-drx-v65-20261013/finalizer_status.json"))["status"])')" != "COMPLETE" ]]; then
  write_failure "FINALIZER_NOT_COMPLETE"
  exit 4
fi

PYTHONPATH=src:. python -m full_model.analysis.write_v65_decision \
  --campaign "$campaign" \
  --json "$analysis/v65_final_decision.json" \
  --markdown "$analysis/v65_final_decision.md" \
  > "$logs/write_v65_decision.log" 2>&1
decision_status=$?

evidence_commit="$(git rev-parse HEAD)"
archive="$campaign/v65_definitive_delivery.zip"
PYTHONPATH=src:. python -m full_model.analysis.build_v65_delivery \
  --campaign "$campaign" --worktree "$worktree" --output "$archive" \
  --production-commit 6debbe0813ca0c365abcf473e5106cb90dbdd75a \
  --evidence-commit "$evidence_commit" \
  > "$logs/build_v65_delivery.log" 2>&1
archive_status=$?

CAMPAIGN="$campaign" EVIDENCE_COMMIT="$evidence_commit" \
DECISION_STATUS="$decision_status" ARCHIVE_STATUS="$archive_status" \
python - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
campaign = Path(os.environ["CAMPAIGN"])
archive = campaign/"v65_definitive_delivery.zip"
decision = campaign/"analysis/v65_final_decision.json"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
steps = {
    "decision": int(os.environ["DECISION_STATUS"]),
    "archive": int(os.environ["ARCHIVE_STATUS"]),
}
payload = {
    "schema": "asb-drx-v65-closure-status-v1",
    "status": ("COMPLETE" if all(value == 0 for value in steps.values())
               and archive.is_file() and decision.is_file() else "FAILED"),
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "evidence_source_commit": os.environ["EVIDENCE_COMMIT"],
    "step_exit_codes": steps,
    "decision_sha256": digest(decision),
    "archive": str(archive.resolve()),
    "archive_sha256": digest(archive),
    "archive_bytes": archive.stat().st_size if archive.is_file() else None,
}
temporary = campaign/"closure_status.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
temporary.replace(campaign/"closure_status.json")
PY
