#!/usr/bin/env bash
#SBATCH --job-name=v42-topology-bc4b2bd
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -Eeuo pipefail
module load anaconda/2025.12

: "${V42_SOURCE_ARCHIVE:?source archive required}"
: "${V42_SOURCE_SHA:?source SHA required}"
: "${V42_SEED_CHECKPOINT:?seed checkpoint required}"
: "${V42_SEED_SHA256:?seed checksum required}"
: "${V42_RUN_ROOT:?persistent run root required}"

scratch="${TMPDIR:?}/v42-topology-${SLURM_JOB_ID}"
source_root="$scratch/source"
case_dir="$V42_RUN_ROOT/mechanical_heterogeneity_topology_on_n64"
mkdir -p "$source_root" "$case_dir" "$V42_RUN_ROOT/logs"
tar -xzf "$V42_SOURCE_ARCHIVE" -C "$source_root"
actual=$(sha256sum "$V42_SEED_CHECKPOINT" | awk '{print $1}')
test "$actual" = "$V42_SEED_SHA256"
if ! find "$case_dir" -maxdepth 1 -name 'checkpoint_step_*.npz' | grep -q .; then
  cp "$V42_SEED_CHECKPOINT" \
    "$case_dir/checkpoint_step_000022620_strain_0.05000604.npz"
fi

export PYTHONPATH="$source_root:$source_root/src"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export V37_SOURCE_SHA="$V42_SOURCE_SHA"

solver_pid=""
stop_requested() {
  test -z "$solver_pid" || kill -TERM "$solver_pid" 2>/dev/null || true
  test -z "$solver_pid" || wait "$solver_pid" 2>/dev/null || true
  exit 99
}
trap stop_requested USR1 TERM

python "$source_root/full_model/analysis/run_v30_mura_tier_b1_case.py" \
  --case-dir "$case_dir" --grid 64 --condition mechanical_heterogeneity \
  --seed 42 --length-m 1e-5 --temperature-K 1100 --strain-rate-s 10000 \
  --initial-strain .01 --target-strain .0501 --trial-dt-s 2e-9 \
  --progress-checkpoint-strain .000025 --wall-checkpoint-s 600 \
  --history-interval 25 --max-wall-s 13800 \
  --mura-work-budget-mode energy_limited_feasible_extents \
  --ordering-integration-method v40_stiff_dispatch \
  --topology-route-enabled \
  >"$V42_RUN_ROOT/logs/solver.out" 2>"$V42_RUN_ROOT/logs/solver.err" &
solver_pid=$!
wait "$solver_pid"
