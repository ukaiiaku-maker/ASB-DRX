#!/usr/bin/env bash
set -euo pipefail

case_name="${CASE_NAME:?CASE_NAME must be asb_rate30000_seed42 or drx_iso_rate1000_seed42}"
export MPLBACKEND=Agg
export PYTHON=python
export DRIVER=drx_full_v34_recovery.py
export DRX_OUTDIR="output/${case_name}"
export GLIDE_ACTIVATION_ENTROPY_KB=0.0
export RECOVERY_ACTIVATION_ENTROPY_KB=0.0
export CLIMB_ACTIVATION_ENTROPY_KB=0.0
export WALL_ACTIVATION_ENTROPY_KB=0.0
export GB_SOURCE_ACTIVATION_ENTROPY_KB=0.0
export GB_TRANSMISSION_ACTIVATION_ENTROPY_KB=0.0
export EMBRYO_ACTIVATION_ENTROPY_KB=0.0
export BOUNDARY_ACTIVATION_ENTROPY_KB=0.0

case "$case_name" in
  asb_rate30000_seed42)
    export BRANCH=asb_only RATE=30000 NSTEPS=5000 DT_STRAIN_STEP=1.0e-4
    export T0=1100.0 POLY_SEED=42 NUC_SEED=271870
    ;;
  drx_iso_rate1000_seed42)
    export BRANCH=drx_isothermal RATE=1000 TARGET_STRAIN=0.5 DT_STRAIN_STEP=1.0e-4
    export T0=1100.0 POLY_SEED=42 NUC_SEED=271870
    ;;
  *)
    echo "unknown CASE_NAME=$case_name" >&2
    exit 2
    ;;
esac

python -m py_compile drx_full_v34_recovery.py arrhenius_kinetics.py
bash run_full_v34_recovery.sh
find "$DRX_OUTDIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$DRX_OUTDIR/output_inventory.sha256"
