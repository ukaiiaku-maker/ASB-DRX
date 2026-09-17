#!/usr/bin/env bash
# Wait locally for the V37/V38 conduction array to release its slots, then run
# the selected current-source physical front hold on the local workstation.
set -Eeuo pipefail

repo=/Users/sdillon/HPC3/worktrees/asb-drx-full-v34-recovery
output_root=/Users/sdillon/HPC3/local-results/asb-drx-v38-front-physical
conduction_job=56129085
login_host=${HPC3_LOGIN_HOST:-uci-hpc3}
poll_seconds=${V38_FRONT_POLL_SECONDS:-300}
source_sha=$(git -C "$repo" rev-parse HEAD)

while ssh -o BatchMode=yes -o ConnectTimeout=10 "$login_host" \
  "squeue -h -j '$conduction_job' -o %T" | grep -q .; do
  sleep "$poll_seconds"
done

case_root="$output_root/$source_sha/n64_shape_n_low_hold_30ms"
mkdir -p "$case_root"
cd "$repo"
export V37_FRONT_SOURCE_SHA="$source_sha"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec python full_model/analysis/run_v36_recurrent_physical_response.py \
  --output-dir "$case_root" \
  --protocol hold \
  --grid 64 \
  --intervals 30 \
  --dt-s 1e-3 \
  --proposal-fraction 0.0625 \
  --disable-mura \
  --checkpoint-every 5 \
  --front-exp-n 1.0
