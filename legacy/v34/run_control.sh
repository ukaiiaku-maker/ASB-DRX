#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=Agg
export PYTHON=python
export DRIVER=drx_var_v34_candidate_drx_asb_sweep.py
export DRX_OUTDIR=output/v34_coupled_rate1000_seed42
export BRANCH=coupled
export RATE=1000
export TARGET_STRAIN=0.5
export DT_STRAIN_STEP=1.0e-4
export ACTIVITY_MODE=crystallographic_local
export POLY_SEED=42
export NUC_SEED=271870
bash run_v34_candidate_drx_asb.sh
