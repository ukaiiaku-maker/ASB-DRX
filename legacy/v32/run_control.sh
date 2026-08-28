#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=Agg
export PYTHON=python
export DRIVER=drx_var_v32_gb_transmission_processzone_asb_sweep.py
export DRX_OUTDIR=output/v32_rate30000_seed42
export RATE=30000
export NSTEPS=5000
export DT_STRAIN_STEP=1.0e-4
export ACTIVITY_MODE=crystallographic_local
export POLY_SEED=42
export NUC_SEED=271870
bash run_v32_gb_transmission_processzone_asb_v3.sh
