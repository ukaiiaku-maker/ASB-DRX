#!/usr/bin/env python3
"""Non-model deterministic numerical/staging smoke; execute only on HPC3."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(*args: str) -> str:
    result = subprocess.run(args, check=False, text=True, capture_output=True)
    return (result.stdout or result.stderr).strip()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    output = Path("output")
    output.mkdir(exist_ok=True)
    intervals = 100_000
    dx = 1.0 / intervals
    total = 0.5 * (1.0 + math.exp(-1.0))
    for index in range(1, intervals):
        total += math.exp(-index * dx)
    integral = total * dx
    exact = 1.0 - math.exp(-1.0)
    abs_error = abs(integral - exact)
    tolerance = 1.0e-10
    passed = abs_error < tolerance
    source_files = [Path("smoke.py"), Path("run.sh")]
    payload = {
        "schema": "asb-drx-environment-smoke/v1",
        "run_id": os.environ.get("HPC3_RUN_ID"),
        "parent_campaign_id": "asb-drx-independent-20260827",
        "purpose": "environment_and_staging_only_not_physical_model",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "slurm": {key: value for key, value in os.environ.items() if key.startswith("SLURM_")},
        "modules": command("bash", "-lc", "module -t list 2>&1 || true"),
        "environment": {
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "hpc3_work_dir": os.environ.get("HPC3_WORK_DIR"),
            "hpc3_output_dir": os.environ.get("HPC3_OUTPUT_DIR")
        },
        "source": {str(path): file_hash(path) for path in source_files},
        "resolved_configuration_SI": {
            "test_function": "exp(-x)",
            "integration_domain": [0.0, 1.0],
            "intervals": intervals,
            "dx": dx,
            "absolute_tolerance": tolerance,
            "units": "dimensionless"
        },
        "result": {
            "integral": integral,
            "exact": exact,
            "absolute_error": abs_error,
            "passed": passed
        },
        "dd_closure": {"version": None, "sha256": None, "status": "not_used_gate0_smoke"},
        "rng": "none"
    }
    result_path = output / "smoke_result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output / "output_inventory.sha256").write_text(f"{file_hash(result_path)}  smoke_result.json\n")
    print(json.dumps(payload["result"], sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
