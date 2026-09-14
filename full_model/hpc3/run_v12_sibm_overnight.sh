#!/usr/bin/env bash
set -uo pipefail

export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHON=python DRIVER=drx_full_v34_recovery.py
source_checkpoint_128="../../.hpc3/inputs/v9_event_restart_006900.npz"
source_checkpoint_192="../../.hpc3/inputs/v12_event_restart_006900_192.npz"
source_checkpoint_256="../../.hpc3/inputs/v12_event_restart_006900_256.npz"
campaign_start=$(date +%s)
launch_deadline=$((campaign_start + 40800))
mkdir -p output/v12-cases output/v12-status

case_table=$(mktemp "${TMPDIR:-/tmp}/v12-cases.XXXXXX")
trap 'rm -f "$case_table"' EXIT
python ../hpc3/write_v12_case_table.py "$case_table"

run_case() {
  row="$1"
  IFS='|' read -r case_id grid steps rate temp radius window mobility drag checkpoint tier <<< "$row"
  case_dir="output/v12-cases/$case_id"
  status_dir="output/v12-status/$case_id"
  mkdir -p "$case_dir" "$status_dir"
  start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  export DRX_OUTDIR="$case_dir" NX="$grid" NY="$grid" NSTEPS="$steps"
  export RATE="$rate" T0="$temp" SIBM_INITIAL_BULGE_RADIUS_UM="$radius"
  export SIBM_ACTIVE_WINDOW_RADIUS_UM="$window" SIBM_MOBILITY_MULTIPLIER="$mobility"
  export SIBM_PHYSICAL_DRAG_PRESSURE_PA="$drag" RESTART_FILE="$checkpoint"
  export RESTART_RESET_CLOCK=true BRANCH=drx_isothermal
  export USE_STATEFUL_EMBRYOS=false USE_ATOMIC_STATEFUL_PROMOTION=false
  export USE_EXPF_EMBRYO_CREATION=false USE_SPARSE_COMMON_FRONT_STATE=true
  export USE_SIBM_EXISTING_BOUNDARY=true STORED_ENERGY_COUPLING_MODE=common_variational
  export DISABLE_NEW_STOCHASTIC_CREATION_AFTER_RESTART=true
  export DIAG_INTERVAL=100 SAVE_INTERVAL=250 RESTART_INTERVAL=250
  export RESTART_WALLCLOCK_INTERVAL_S=600 SAVE_MAIN_PANELS=false
  resolved="$status_dir/resolved.env"
  env | LC_ALL=C sort > "$resolved"
  set +e
  bash run_full_v34_recovery.sh >"$status_dir/stdout.log" 2>"$status_dir/stderr.log"
  application_exit=$?
  set -e
  end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  CASE_ID="$case_id" TIER="$tier" GRID="$grid" STEPS="$steps" RATE_VALUE="$rate" \
    TEMP_VALUE="$temp" RADIUS="$radius" WINDOW="$window" MOBILITY="$mobility" \
    DRAG="$drag" CHECKPOINT="$checkpoint" START_UTC="$start_utc" END_UTC="$end_utc" \
    APPLICATION_EXIT="$application_exit" SCIENTIFIC_SOURCE_SHA="${SCIENTIFIC_SOURCE_SHA:-unknown}" \
    python ../hpc3/finalize_v12_case.py "$case_dir" "$status_dir/final.json"
  (cd "$status_dir" && find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256)
  (cd "$case_dir" && find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256)
  return 0
}

running=0
pids=()
ids=()
remaining=()
while IFS= read -r row; do
  now=$(date +%s)
  if (( now >= launch_deadline )); then
    remaining+=("$row")
    continue
  fi
  run_case "$row" &
  pids+=("$!")
  ids+=("${row%%|*}")
  running=$((running+1))
  if (( running == 2 )); then
    wait "${pids[0]}" || true
    wait "${pids[1]}" || true
    pids=(); ids=(); running=0
  fi
done < "$case_table"
for pid in "${pids[@]}"; do wait "$pid" || true; done
printf '%s\n' "${remaining[@]}" > output/v12-status/remaining_cases.txt
SCIENTIFIC_SOURCE_SHA="${SCIENTIFIC_SOURCE_SHA:-unknown}" \
  python ../hpc3/summarize_v12_overnight.py output/v12-cases output/v12-status \
  output/overnight_manifest.json output/v12_comparison.csv output/v12_comparison.json
(cd output && find . -type f ! -name output_inventory.sha256 -print0 | sort -z | xargs -0 sha256sum > output_inventory.sha256)
