#!/usr/bin/env python3
"""
Plot GB slip-transmission diagnostics over ASB/DRX fields from v30-v32 restart files.

Usage:
  python3 plot_v32_gb_transmission_overlay.py results_v32_gbtrans_processzone_30k_long \
      --steps 0,1000,3000,5000,6750

Outputs:
  <result_dir>/gbtrans_overlay_diagnostics/gbtrans_overlay_XXXXXX.png
  <result_dir>/gbtrans_overlay_diagnostics/gbtrans_timeseries.png

The overlay recomputes the reduced 2-D Arrhenius GB transmission proxy from saved
restart data: grain labels, grain orientations, plastic orientation, temperature, and
run parameters. It is intentionally diagnostic only; it does not modify state.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None

KB_EV = 8.617333262145e-5


def angle_wrap(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def angle_wrap_pi(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap orientation/slip-direction differences modulo pi."""
    return (np.asarray(x) + 0.5 * np.pi) % np.pi - 0.5 * np.pi


def load_params(z: np.lib.npyio.NpzFile) -> Dict:
    if "P_json" in z.files:
        raw = z["P_json"]
        try:
            if np.ndim(raw) == 0:
                return json.loads(str(raw.item()))
            return json.loads(str(raw))
        except Exception:
            pass
    return {}


def infer_step(path: Path) -> int:
    m = re.search(r"_(\d{6})\.npz$", path.name)
    if m:
        return int(m.group(1))
    return -1


def find_restart_files(result_dir: Path) -> List[Path]:
    files = sorted(result_dir.glob("drx_v25_restart_*.npz"), key=infer_step)
    if not files:
        files = sorted(result_dir.glob("drx_v*_restart_*.npz"), key=infer_step)
    return files


def choose_files(files: List[Path], steps: Optional[List[int]], every: Optional[int], max_files: int) -> List[Path]:
    if not files:
        return []
    if steps:
        out = []
        available = np.array([infer_step(p) for p in files])
        for s in steps:
            idx = int(np.argmin(np.abs(available - s)))
            out.append(files[idx])
        # preserve order, remove duplicates
        seen = set(); uniq = []
        for p in out:
            if p not in seen:
                seen.add(p); uniq.append(p)
        return uniq
    if every and every > 0:
        out = [p for p in files if infer_step(p) % every == 0]
        if files[-1] not in out:
            out.append(files[-1])
        return out[:max_files]
    if len(files) <= max_files:
        return files
    idx = np.linspace(0, len(files) - 1, max_files).round().astype(int)
    return [files[i] for i in idx]


def label_gb_support(lab: np.ndarray, sigma_px: float = 1.25) -> np.ndarray:
    """Make a diffuse GB support from hard labels, robust to restarts without gb_mask."""
    lab = np.asarray(lab, dtype=int)
    edge = np.zeros_like(lab, dtype=float)
    for shift in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        edge |= (np.roll(lab, shift, axis=(0, 1)) != lab)
    edge = edge.astype(float)
    if ndimage is not None and sigma_px > 0:
        gb = ndimage.gaussian_filter(edge, sigma_px, mode="wrap")
        if np.nanmax(gb) > 0:
            gb = gb / np.nanmax(gb)
        return np.clip(gb, 0, 1)
    return edge


def recompute_gb_transmission(z: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    P = load_params(z)
    lab = np.asarray(z["lab"], dtype=int)
    T = np.asarray(z["T"], dtype=float)
    nx, ny = lab.shape

    # Orientations. Prefer hard grain orientations + plastic residual orientation.
    if "psi_gv" in z.files:
        psi_gv = np.asarray(z["psi_gv"], dtype=float)
        psi = psi_gv[np.clip(lab, 0, len(psi_gv) - 1)]
    elif "psi_lat" in z.files:
        psi = np.asarray(z["psi_lat"], dtype=float)
    else:
        psi = np.zeros_like(T)
    if bool(P.get("gb_trans_include_plastic_orientation", True)) and "psi_plastic" in z.files:
        psi = psi + np.asarray(z["psi_plastic"], dtype=float)
    psi = angle_wrap(psi)

    gb = label_gb_support(lab, sigma_px=float(P.get("gb_diag_sigma_px", 1.25)))
    active_core = gb > float(P.get("gb_hp_min_gb_support", 0.25))

    slip_angles = np.deg2rad(np.asarray(P.get("slip_angles_deg", [25.0, -65.0]), dtype=float))
    nslip = int(P.get("nSlip", len(slip_angles)))
    if slip_angles.size < nslip:
        slip_angles = np.resize(slip_angles, nslip)

    Gmis = float(P.get("gb_trans_misorientation_barrier_eV", 0.80))
    Gres = float(P.get("gb_trans_residual_barrier_eV", 1.20))
    pwr = max(float(P.get("gb_trans_barrier_power", 2.0)), 1.0e-12)
    mis_ref = max(np.deg2rad(float(P.get("gb_trans_mis_ref_deg", 30.0))), 1.0e-12)
    bres_ref = max(float(P.get("gb_trans_bres_ref", 0.75)), 1.0e-12)
    use_worst = bool(P.get("gb_trans_use_neighbor_worst", True))
    outgoing_mode = str(P.get("gb_trans_outgoing_mode", "same_index")).lower()
    coupling = np.clip(float(P.get("gb_trans_gdot_coupling", 1.0)), 0.0, 1.0)
    fmin = float(P.get("gb_trans_min_factor", 1.0e-6))

    shp = (nx, ny, nslip)
    best_bar = np.zeros(shp, dtype=float)
    best_mis = np.zeros(shp, dtype=float)
    best_mp = np.ones(shp, dtype=float)
    best_bres = np.zeros(shp, dtype=float)
    have = np.zeros(shp, dtype=bool)

    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        lab_n = np.roll(lab, (di, dj), (0, 1))
        psi_n = np.roll(psi, (di, dj), (0, 1))
        is_gb_face = (lab_n != lab) & active_core
        if not np.any(is_gb_face):
            continue
        dpsi = np.abs(angle_wrap(psi - psi_n))
        mis_term = (np.sin(0.5 * np.minimum(dpsi, np.pi)) / max(np.sin(0.5 * mis_ref), 1.0e-12)) ** pwr
        for ss in range(nslip):
            th_in = psi + slip_angles[ss]
            out_list: Iterable[int]
            if outgoing_mode in ("same", "same_index", "same_slip"):
                out_list = [ss]
            else:
                out_list = range(nslip)
            local_best_bar = None
            local_best_mp = None
            local_best_bres = None
            for tt in out_list:
                th_out = psi_n + slip_angles[tt % nslip]
                c = np.abs(np.cos(angle_wrap_pi(th_in - th_out)))
                mp = np.clip(c * c, 0.0, 1.0)
                bres = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - c)))
                bar = Gmis * mis_term + Gres * (bres / bres_ref) ** pwr
                if local_best_bar is None:
                    local_best_bar, local_best_mp, local_best_bres = bar, mp, bres
                else:
                    better = bar < local_best_bar
                    local_best_bar = np.where(better, bar, local_best_bar)
                    local_best_mp = np.where(better, mp, local_best_mp)
                    local_best_bres = np.where(better, bres, local_best_bres)
            if use_worst:
                replace = is_gb_face & ((~have[:, :, ss]) | (local_best_bar > best_bar[:, :, ss]))
            else:
                replace = is_gb_face & ((~have[:, :, ss]) | (local_best_bar < best_bar[:, :, ss]))
            best_bar[:, :, ss] = np.where(replace, local_best_bar, best_bar[:, :, ss])
            best_mp[:, :, ss] = np.where(replace, local_best_mp, best_mp[:, :, ss])
            best_bres[:, :, ss] = np.where(replace, local_best_bres, best_bres[:, :, ss])
            best_mis[:, :, ss] = np.where(replace, np.rad2deg(dpsi), best_mis[:, :, ss])
            have[:, :, ss] |= (is_gb_face & replace)

    kBT = KB_EV * np.maximum(T, 1.0)
    fac = np.exp(np.clip(-best_bar / np.maximum(kBT[:, :, None], 1.0e-12), -700.0, 0.0))
    fac = np.clip(fac, fmin, 1.0)
    fac_eff = 1.0 - coupling * gb[:, :, None] * (1.0 - fac)
    fac_eff = np.where(have, fac_eff, 1.0)

    return dict(
        gb=gb,
        factor_min=np.min(fac_eff, axis=2),
        factor_mean=np.mean(fac_eff, axis=2),
        barrier_max=np.max(best_bar, axis=2),
        barrier_min=np.min(best_bar, axis=2),
        bres_max=np.max(best_bres, axis=2),
        bres_mean=np.mean(best_bres, axis=2),
        mprime_min=np.min(best_mp, axis=2),
        mprime_mean=np.mean(best_mp, axis=2),
        mis_max=np.max(best_mis, axis=2),
        have=np.any(have, axis=2),
    )


def masked(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(arr, dtype=float, copy=True)
    out[~mask] = np.nan
    return out


def plot_one(result_dir: Path, file: Path, out_dir: Path) -> None:
    z = np.load(file, allow_pickle=True)
    step = int(z["step"]) if "step" in z.files else infer_step(file)
    T = np.asarray(z["T"], dtype=float)
    rho = np.asarray(z["rho"], dtype=float)
    lab = np.asarray(z["lab"], dtype=int)
    psi_lat = np.asarray(z["psi_lat"], dtype=float) if "psi_lat" in z.files else np.zeros_like(T)
    psi_deg = np.rad2deg(psi_lat)
    tr = recompute_gb_transmission(z)
    gb = tr["gb"] > 0.25
    have = tr["have"] & gb

    fig, ax = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    im = ax[0, 0].pcolormesh(T.T, shading="auto")
    ax[0, 0].contour(gb.T.astype(float), levels=[0.5], linewidths=0.6)
    ax[0, 0].set_title(f"T (K), step {step}")
    fig.colorbar(im, ax=ax[0, 0])

    im = ax[0, 1].pcolormesh(T.T, shading="auto")
    ax[0, 1].contour(masked(tr["factor_min"], have).T, levels=[0.3, 0.5, 0.7], linewidths=0.9)
    ax[0, 1].contour(masked(tr["bres_max"], have).T, levels=[0.25, 0.5, 0.75], linewidths=0.6, linestyles="dashed")
    ax[0, 1].set_title("T with f_trans contours; dashed b_res")
    fig.colorbar(im, ax=ax[0, 1])

    im = ax[0, 2].pcolormesh(masked(tr["factor_min"], have).T, shading="auto", vmin=0, vmax=1)
    ax[0, 2].set_title("GB min transmission factor")
    fig.colorbar(im, ax=ax[0, 2])

    im = ax[1, 0].pcolormesh(masked(tr["barrier_max"], have).T, shading="auto")
    ax[1, 0].set_title("GB max transmission barrier (eV)")
    fig.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].pcolormesh(masked(tr["bres_max"], have).T, shading="auto", vmin=0, vmax=max(1.0, np.nanmax(masked(tr["bres_max"], have))))
    ax[1, 1].set_title(r"GB max $b_{res}/b$")
    fig.colorbar(im, ax=ax[1, 1])

    im = ax[1, 2].pcolormesh(psi_deg.T, shading="auto")
    ax[1, 2].contour(gb.T.astype(float), levels=[0.5], linewidths=0.6)
    ax[1, 2].set_title("psi_lat (deg) + GB")
    fig.colorbar(im, ax=ax[1, 2])

    for a in ax.ravel():
        a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])

    out_path = out_dir / f"gbtrans_overlay_{step:06d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timeseries(result_dir: Path, out_dir: Path) -> None:
    csv = result_dir / "drx_v25_restart_asb_diagnostics.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    x = df["eps_pct"] if "eps_pct" in df.columns else df["step"]
    xlabel = "strain (%)" if "eps_pct" in df.columns else "step"
    fig, ax = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

    def plot_col(axis, cols, title, ylabel=None):
        for c in cols:
            if c in df.columns:
                axis.plot(x, df[c], label=c)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        if ylabel:
            axis.set_ylabel(ylabel)
        if axis.lines:
            axis.legend(fontsize=8)

    plot_col(ax[0, 0], ["sigma_MPa"], "stress")
    plot_col(ax[0, 1], ["T_max", "T_mean", "asb_T_std"], "thermal response")
    plot_col(ax[1, 0], ["asb_rho_hot_over_cold", "asb_corr_T_logrho"], "ASB density diagnostics")
    plot_col(ax[1, 1], ["gb_trans_factor_mean", "gb_trans_factor_min"], "GB transmission factor")
    plot_col(ax[2, 0], ["gb_trans_barrier_eV_mean", "gb_trans_barrier_eV_max"], "GB barrier")
    plot_col(ax[2, 1], ["gb_trans_bres_mean", "gb_trans_bres_max", "psi_std_deg", "psi_plastic_max_deg"], "residual/rotation")

    out_path = out_dir / "gbtrans_timeseries.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--steps", default="", help="comma-separated requested steps; closest restart is used")
    ap.add_argument("--every", type=int, default=0, help="plot restarts whose step is a multiple of this value")
    ap.add_argument("--max-files", type=int, default=8)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    result_dir = args.result_dir
    out_dir = args.outdir or (result_dir / "gbtrans_overlay_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_restart_files(result_dir)
    if not files:
        raise SystemExit(f"No restart npz files found in {result_dir}")
    steps = [int(s) for s in args.steps.split(",") if s.strip()] if args.steps else None
    chosen = choose_files(files, steps, args.every or None, args.max_files)
    print(f"Found {len(files)} restarts; plotting {len(chosen)}")
    for f in chosen:
        print(f"  {f.name}")
        plot_one(result_dir, f, out_dir)
    plot_timeseries(result_dir, out_dir)
    print(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
