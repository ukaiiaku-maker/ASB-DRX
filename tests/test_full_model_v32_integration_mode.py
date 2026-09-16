import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from full_model.production.integration_modes import (
    MODE, common_front_integration_requested,
    validate_common_front_integration)


def _combined():
    return {
        "v31_asb_common_mura_ledger": True,
        "use_sparse_common_front_state": True,
        "use_sibm_existing_boundary": True,
        "disable_nucleation": True,
        "use_hazard_nucleation": False,
        "use_stateful_embryos": False,
        "use_component_relabel": False,
    }


@pytest.mark.parametrize("parameters", [
    {"v31_asb_common_mura_ledger": True},
    {"use_sparse_common_front_state": True,
     "use_sibm_existing_boundary": True},
    {},
])
def test_integration_gate_is_exact_off_and_does_not_mutate_single_modes(parameters):
    before = copy.deepcopy(parameters)
    assert validate_common_front_integration(parameters) == "INACTIVE_EXACT_OFF"
    assert parameters == before
    assert not common_front_integration_requested(parameters)


def test_nominal_common_front_now_fails_fast_instead_of_silently_freezing_front():
    parameters = _combined()
    with pytest.raises(ValueError, match="UNSUPPORTED_COMMON_FRONT_DUAL_OWNER"):
        validate_common_front_integration(parameters)
    parameters[MODE] = True
    with pytest.raises(ValueError, match="V32_COMMON_FRONT_ADAPTER_UNAVAILABLE"):
        validate_common_front_integration(parameters)


def test_future_adapter_contract_forbids_every_label_creation_path():
    parameters = _combined(); parameters[MODE] = True
    assert (validate_common_front_integration(parameters, adapter_active=True)
            == "ACTIVE_UNIFIED_EXISTING_BOUNDARY")
    for key, bad_value in (
            ("disable_nucleation", False), ("use_hazard_nucleation", True),
            ("use_stateful_embryos", True), ("use_component_relabel", True)):
        invalid = dict(parameters, **{key: bad_value})
        with pytest.raises(ValueError, match="LABEL_CREATION_FORBIDDEN"):
            validate_common_front_integration(invalid, adapter_active=True)


def test_production_driver_calls_gate_before_common_mode_rewrites(tmp_path):
    root = Path(__file__).resolve().parents[1]
    production = root/"full_model"/"production"
    parameters = {
        "v31_asb_common_mura_ledger": True,
        "use_sparse_common_front_state": True,
        "use_sibm_existing_boundary": True,
    }
    environment = dict(
        os.environ, DRX_OUTDIR=str(tmp_path), MPLBACKEND="Agg",
        DRX_PARAMS=json.dumps(parameters, separators=(",", ":")))
    completed = subprocess.run(
        [sys.executable, "drx_full_v34_recovery.py"], cwd=production,
        env=environment, text=True, capture_output=True, timeout=30)
    assert completed.returncode != 0
    assert "UNSUPPORTED_COMMON_FRONT_DUAL_OWNER" in (
        completed.stdout+completed.stderr)
