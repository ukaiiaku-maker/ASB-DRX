#!/usr/bin/env python3
"""Recovery-capable V52 continuation and companion-grid controller."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


SCHEMA = "asb-drx/v52/completion-manager/v2"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def process_identity(pid):
    """Return a PID-reuse-resistant process identity, or None."""
    result = subprocess.run(
        ["ps", "-p", str(int(pid)), "-o", "pid=", "-o", "ppid=",
         "-o", "lstart=", "-o", "command="], capture_output=True, text=True)
    if result.returncode or not result.stdout.strip():
        return None
    fields = result.stdout.strip().split(None, 7)
    if len(fields) < 8:
        return None
    cwd = None
    lsof = subprocess.run(
        ["lsof", "-a", "-p", str(int(pid)), "-d", "cwd", "-Fn"],
        capture_output=True, text=True)
    for line in lsof.stdout.splitlines():
        if line.startswith("n"):
            cwd = line[1:]
    return {"pid": int(fields[0]), "ppid": int(fields[1]),
            "start_identity": " ".join(fields[2:7]), "command": fields[7],
            "cwd": cwd}


def same_identity(record, live):
    return bool(live and int(record.get("pid", -1)) == live["pid"]
                and record.get("start_identity") == live["start_identity"])


def _option(argv, name):
    for index, token in enumerate(argv):
        if token == name and index+1 < len(argv):
            return argv[index+1]
        if token.startswith(name+"="):
            return token.split("=", 1)[1]
    return None


def command_owns_output(command, cwd, output_root):
    """Recognize direct-script and ``python -m`` continuation invocations."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    direct = any(Path(token).name == "run_v49_physical_continuation.py"
                 for token in argv)
    module = any(token.endswith("run_v49_physical_continuation")
                 for token in argv)
    raw = _option(argv, "--output-dir")
    if not (direct or module) or raw is None:
        return False
    path = Path(raw)
    if not path.is_absolute():
        if not cwd:
            return False
        path = Path(cwd)/path
    return path.resolve() == Path(output_root).resolve()


def matching_processes(output_root, identities=None):
    if identities is None:
        listing = subprocess.run(
            ["ps", "-axo", "pid=,command="], capture_output=True, text=True,
            check=True).stdout
        candidate_pids = []
        for line in listing.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) == 2 and "run_v49_physical_continuation" in fields[1]:
                candidate_pids.append(int(fields[0]))
        identities = [process_identity(pid) for pid in candidate_pids]
    return [row for row in identities if row and command_owns_output(
        row["command"], row.get("cwd"), output_root)]


def inspect_manifest(output_root, expected_intervals, identities=None):
    """Classify a continuation prefix without changing it."""
    path = Path(output_root)/"run_manifest.json"
    owners = matching_processes(output_root, identities)
    if not path.exists():
        return {"classification": ("HEALTHY_RUNNING" if owners else
                                    "UNKNOWN_OR_CORRUPT"), "owners": owners}
    try:
        manifest = json.loads(path.read_text())
        checkpoint = Path(manifest["latest_checkpoint"])
        checksum_ok = checkpoint.exists() and digest(checkpoint) == manifest[
            "latest_checkpoint_sha256"]
        prefix_ok = (checksum_ok and int(manifest["completed_intervals"]) >= 0
                     and len(manifest.get("records", []))
                     >= int(manifest["completed_intervals"]))
    except Exception as error:
        return {"classification": "UNKNOWN_OR_CORRUPT", "owners": owners,
                "reason": f"{type(error).__name__}: {error}"}
    status = manifest.get("status")
    completed = int(manifest["completed_intervals"])
    if status == "COMPLETE" and completed >= int(expected_intervals) and prefix_ok:
        classification = "VALID_COMPLETED"
    elif status == "COMPLETE" and completed < int(expected_intervals) and prefix_ok:
        classification = "VALID_COMPLETED_PREFIX"
    elif status in ("PHYSICAL_TERMINAL", "EARLY_PHYSICAL_TERMINAL") and prefix_ok:
        classification = "VALID_PHYSICAL_TERMINAL"
    elif owners and prefix_ok:
        classification = "HEALTHY_RUNNING"
    elif status == "RUNNING" and prefix_ok:
        classification = "RESTARTABLE_INFRASTRUCTURE_INTERRUPTION"
    elif prefix_ok:
        classification = "NUMERICAL_FAILURE_WITH_VALID_PREFIX"
    else:
        classification = "UNKNOWN_OR_CORRUPT"
    return {"classification": classification, "owners": owners,
            "manifest": manifest, "manifest_path": str(path.resolve()),
            "checkpoint_checksum_verified": checksum_ok}


class LogicalLock:
    def __init__(self, path):
        self.path = Path(path)

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        identity = process_identity(os.getpid())
        payload = {"schema": SCHEMA+"/lock", "created_utc": utc(),
                   "manager": identity}
        while True:
            try:
                descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, (json.dumps(payload)+"\n").encode())
                os.close(descriptor); return payload
            except FileExistsError:
                try:
                    old = json.loads(self.path.read_text())
                except Exception as error:
                    raise RuntimeError("logical manager lock is corrupt") from error
                owner = old.get("manager", {})
                if same_identity(owner, process_identity(owner.get("pid", -1))):
                    raise RuntimeError("logical V52 output already has a live manager")
                stale = self.path.with_name(
                    self.path.name+".stale."+datetime.now().strftime("%Y%m%dT%H%M%S"))
                os.replace(self.path, stale)

    def release(self):
        try:
            current = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if same_identity(current.get("manager", {}), process_identity(os.getpid())):
            self.path.unlink(missing_ok=True)


def new_or_resumed_state(path, primary_output, source):
    path = Path(path); primary = str(Path(primary_output).resolve())
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("schema") != SCHEMA or state.get("primary_output") != primary:
            raise RuntimeError("manager journal does not own this logical output")
        state["resume_count"] = int(state.get("resume_count", 0))+1
        state["resumed_utc"] = utc()
        history = list(state.get("manager_source_history", []))
        history.append(state.get("manager_source_sha"))
        state["manager_source_history"] = list(dict.fromkeys(
            value for value in history if value))
        state["manager_source_sha"] = source
    else:
        state = {"schema": SCHEMA, "created_utc": utc(),
                 "primary_output": primary, "manager_source_sha": source,
                 "resume_count": 0, "state": "STARTING", "history": [],
                 "stages": {}}
    state["manager"] = process_identity(os.getpid())
    # A recovered controller retains prior failures as history rather than as
    # the current terminal status.
    if state.get("state") == "COMPLETE":
        state.pop("failure", None)
        state.pop("failed_utc", None)
    return state


def run_short(command, cwd, log):
    started = time.perf_counter()
    with Path(log).open("a") as stream:
        stream.write("COMMAND "+json.dumps(command)+"\n"); stream.flush()
        result = subprocess.run(command, cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, text=True,
                                env={**os.environ, "PYTHONPATH": "src:."})
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}")
    return time.perf_counter()-started


def persist(state_path, state, status=None):
    if status is not None:
        state["state"] = status
    state["updated_utc"] = utc(); atomic_json(state_path, state)


def wait_or_run_continuation(*, output, target, command, cwd, state,
                             state_path, stage, log, poll_s,
                             infrastructure_retries=1):
    """Resume, attach to, or start one continuation stage exactly once."""
    output = Path(output); retries = 0
    while True:
        audit = inspect_manifest(output, target)
        state["stages"][stage] = {**state["stages"].get(stage, {}),
                                  "target_intervals": target,
                                  "last_classification": audit["classification"]}
        persist(state_path, state, "RUNNING_"+stage)
        if audit["classification"] in ("VALID_COMPLETED",
                                       "VALID_PHYSICAL_TERMINAL"):
            return audit
        if audit["classification"] == "HEALTHY_RUNNING":
            time.sleep(poll_s); continue
        actual = list(command)
        if audit["classification"] in (
                "RESTARTABLE_INFRASTRUCTURE_INTERRUPTION",
                "VALID_COMPLETED_PREFIX"):
            if retries >= infrastructure_retries:
                if audit["classification"] == "RESTARTABLE_INFRASTRUCTURE_INTERRUPTION":
                    raise RuntimeError(f"{stage}: infrastructure retry limit reached")
            manifest = audit["manifest"]
            if "--initial-checkpoint" in actual:
                index = actual.index("--initial-checkpoint")
                actual[index:index+2] = ["--restart", manifest["latest_checkpoint"]]
            elif "--restart" in actual:
                index = actual.index("--restart")
                actual[index+1] = manifest["latest_checkpoint"]
            if audit["classification"] == "RESTARTABLE_INFRASTRUCTURE_INTERRUPTION":
                retries += 1
        elif audit["classification"] != "UNKNOWN_OR_CORRUPT":
            raise RuntimeError(f"{stage}: {audit['classification']}")
        elif (output/"run_manifest.json").exists():
            raise RuntimeError(f"{stage}: corrupt manifest refuses replacement")
        output.mkdir(parents=True, exist_ok=True)
        with Path(log).open("a") as stream:
            stream.write("CHILD "+json.dumps(actual)+"\n"); stream.flush()
            child = subprocess.Popen(
                actual, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                text=True, env={**os.environ, "PYTHONPATH": "src:."},
                start_new_session=True)
        state["stages"][stage].update({
            "child": process_identity(child.pid), "executable": actual[0],
            "argv": actual, "cwd": str(Path(cwd).resolve()),
            "started_utc": utc(), "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()})
        persist(state_path, state)
        returncode = child.wait(); final = inspect_manifest(output, target)
        state["stages"][stage]["child_returncode"] = returncode
        state["stages"][stage]["last_classification"] = final["classification"]
        persist(state_path, state)
        if (returncode and final["classification"] ==
                "RESTARTABLE_INFRASTRUCTURE_INTERRUPTION"
                and retries < infrastructure_retries):
            continue
        if final["classification"] not in ("VALID_COMPLETED",
                                            "VALID_PHYSICAL_TERMINAL"):
            if final.get("checkpoint_checksum_verified"):
                final["classification"] = "NUMERICAL_FAILURE_WITH_VALID_PREFIX"
            raise RuntimeError(f"{stage}: child terminal {final['classification']}")
        return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-output", type=Path, required=True)
    parser.add_argument("--primary-worktree", type=Path, required=True)
    parser.add_argument("--manager-state", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--n192-targets", default="1,8,16,32")
    parser.add_argument("--n192-wall-budget-s", type=float, default=8*3600.0)
    args = parser.parse_args()
    root = Path.cwd().resolve(); verification = root/"full_model/verification"
    log = verification/"v52_completion_manager_v2.log"
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    lock_path = (args.lock_path or
                 args.primary_output.parent.parent/".v52-completion-manager.lock")
    lock = LogicalLock(lock_path); lock.acquire()
    state = new_or_resumed_state(args.manager_state, args.primary_output, source)
    persist(args.manager_state, state)
    try:
        primary_manifest = args.primary_output/"run_manifest.json"
        while True:
            audit = inspect_manifest(args.primary_output, 104)
            state["stages"]["N128_CONTINUATION"] = {
                "target_intervals": 104,
                "last_classification": audit["classification"],
                "owners": audit.get("owners", [])}
            persist(args.manager_state, state, "WAITING_FOR_N128_CONTINUATION")
            if audit["classification"] in ("VALID_COMPLETED",
                                           "VALID_PHYSICAL_TERMINAL"):
                break
            if audit["classification"] == "HEALTHY_RUNNING":
                time.sleep(args.poll_s); continue
            if audit["classification"] == "RESTARTABLE_INFRASTRUCTURE_INTERRUPTION":
                manifest = audit["manifest"]
                command = [sys.executable,
                    "full_model/analysis/run_v49_physical_continuation.py",
                    "--restart", manifest["latest_checkpoint"], "--output-dir",
                    str(args.primary_output.resolve()), "--grid", "128",
                    "--intervals", "104", "--dt-s", str(manifest["macro_dt_s"]),
                    "--protocol", manifest["protocol"], "--strain-rate-s",
                    str(manifest["strain_rate_s"])]
                wait_or_run_continuation(
                    output=args.primary_output, target=104, command=command,
                    cwd=args.primary_worktree, state=state,
                    state_path=args.manager_state, stage="N128_CONTINUATION",
                    log=log, poll_s=args.poll_s)
                break
            raise RuntimeError(f"primary audit failed: {audit['classification']}")

        continuation = verification/"v52_continuation.json"
        plot = verification/"v52_continuation.png"
        if not continuation.exists():
            elapsed = run_short([
                sys.executable, "full_model/analysis/run_v52_continuation_analysis.py",
                "--manifest", str(primary_manifest), "--output", str(continuation),
                "--plot", str(plot)], root, log)
            state["history"].append({"stage": "N128_POSTPROCESS", "wall_s": elapsed})
        persist(args.manager_state, state, "N128_POSTPROCESSED")

        retained_root = Path(
            "/Users/sdillon/HPC3/worktrees/asb-drx-full-v49-20260921/"
            "full_model/production/results-local/v49-physical/n128")
        spatial_root = root/"full_model/production/results-local/v52-spatial/n192"
        initial = spatial_root/"initial.npz"
        init_audit = verification/"v52_spatial_initialization.json"
        if not initial.exists() or not init_audit.exists():
            elapsed = run_short([
                sys.executable,
                "full_model/analysis/run_v52_analytic_spatial_initialization.py",
                "--retained-n128", str(retained_root/"initial.npz"),
                "--companion-checkpoint", str(initial), "--evidence",
                str(init_audit)], root, log)
            state["history"].append({"stage": "ANALYTIC_N192_INITIAL",
                                     "wall_s": elapsed})
        targets = sorted(set(int(value) for value in args.n192_targets.split(",")))
        n192_started = time.perf_counter(); last_target = 0
        for target in targets:
            loading = spatial_root/"loading"
            if last_target:
                manifest = json.loads((loading/"run_manifest.json").read_text())
                samples = [segment["wall_seconds"] for row in manifest["records"]
                           for segment in row.get("segments", [])]
                observed = (sum(samples)/len(samples)) if samples else None
                elapsed = time.perf_counter()-n192_started
                projected = (None if observed is None else
                             observed*max(target-last_target, 0))
                if (elapsed >= args.n192_wall_budget_s or
                        projected is not None
                        and elapsed+projected > args.n192_wall_budget_s):
                    state["history"].append({
                        "stage": "N192_BUDGET_TERMINAL",
                        "last_target": last_target,
                        "next_target_not_started": target,
                        "elapsed_wall_s": elapsed,
                        "observed_mean_segment_wall_s": observed,
                        "projected_increment_wall_s": projected,
                        "budget_s": args.n192_wall_budget_s})
                    break
            command = [sys.executable,
                "full_model/analysis/run_v49_physical_continuation.py",
                "--initial-checkpoint", str(initial), "--output-dir", str(loading),
                "--grid", "192", "--intervals", str(target), "--dt-s",
                "4.8828125e-7", "--protocol", "continued_deformation",
                "--strain-rate-s", "100"]
            result = wait_or_run_continuation(
                output=loading, target=target, command=command, cwd=root,
                state=state, state_path=args.manager_state,
                stage=f"N192_TO_{target:03d}", log=log, poll_s=args.poll_s)
            last_target = int(result["manifest"]["completed_intervals"])
            n128_checkpoint = retained_root/"loading"/f"checkpoint_{last_target:06d}.npz"
            if n128_checkpoint.exists():
                comparison = verification/f"v52_spatial_comparison_{last_target:03d}.json"
                run_short([
                    sys.executable, "full_model/analysis/run_v52_spatial_comparison.py",
                    "--n128-checkpoint", str(n128_checkpoint), "--n128-manifest",
                    str(retained_root/"loading/run_manifest.json"),
                    "--n192-checkpoint", result["manifest"]["latest_checkpoint"],
                    "--n192-manifest", result["manifest_path"],
                    "--initialization-audit", str(init_audit), "--output",
                    str(comparison)], root, log)
        persist(args.manager_state, state, "RUNNING_CANONICAL_REGRESSION")
        run_short([sys.executable, "-m", "pytest", "-q", "tests"], root, log)
        state["state"] = "COMPLETE"; state["completed_utc"] = utc()
        state.pop("failure", None); state.pop("failed_utc", None)
        state["outputs"] = {name: {"path": str(path.resolve()),
                                          "sha256": digest(path)}
            for name, path in {"primary_manifest": primary_manifest,
                               "continuation": continuation,
                               "continuation_plot": plot,
                               "spatial_initialization": init_audit}.items()}
        persist(args.manager_state, state)
    except Exception as error:
        state["state"] = "FAILED_CONTROLLER"
        state["failure"] = f"{type(error).__name__}: {error}"
        state["failed_utc"] = utc(); persist(args.manager_state, state)
        raise
    finally:
        lock.release()


if __name__ == "__main__":
    main()
