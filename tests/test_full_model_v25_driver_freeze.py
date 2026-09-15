import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_production_exact_all_defect_freeze_changes_only_phase_support(tmp_path):
    root = Path(__file__).resolve().parents[1]
    production = root/"full_model"/"production"
    parameters = {
        "Nx": 16, "Ny": 16, "poly_n": 4, "nSteps": 2,
        "diag_interval": 1, "save_interval": 1000, "restart_interval": 1,
        "plot_interval": 1000, "write_field_npz": False,
        "save_main_panels": False, "save_signed_panels": False,
        "disable_nucleation": True, "use_hazard_nucleation": False,
        "sibm_all_defects_frozen": True, "freeze_orientation": True,
    }
    environment = dict(os.environ, DRX_OUTDIR=str(tmp_path),
                       DRX_PARAMS=json.dumps(parameters), MPLBACKEND="Agg")
    subprocess.run([sys.executable, "drx_full_v34_recovery.py"],
                   cwd=production, env=environment, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                   text=True, timeout=60)
    paths = sorted(tmp_path.glob("drx_v25_restart_*.npz"))
    assert len(paths) == 2
    frozen = ("rho", "rp", "rm", "rho_forest", "rho_wall", "rho_GB",
              "T", "psi_plastic", "gamma_slip", "eps_p", "E_tot",
              "sigma_bar", "collective_activity_memory")
    with np.load(paths[0], allow_pickle=True) as first, np.load(
            paths[1], allow_pickle=True) as second:
        for name in frozen:
            np.testing.assert_array_equal(first[name], second[name])
        assert not np.array_equal(first["eta"], second["eta"])
