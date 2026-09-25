#!/usr/bin/env python3
"""Create outcome-neutral V54 ASB and existing-boundary evolution figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_asb(decision_path: Path, output: Path):
    decision=json.loads(decision_path.read_text())
    fig,axes=plt.subplots(3,2,figsize=(10,9),sharex=True)
    for name,label in (("baseline","thermal feedback"),("control","frozen-flow control")):
        rows=decision["trajectories"][name]
        time=np.asarray([r["physical_time_s"] for r in rows])*1e6
        axes[0,0].plot(time,[r["temperature_peak_minus_mean_K"] for r in rows],label=label)
        axes[0,1].plot(time,[r["work_conjugate_plastic_power"]["inverse_participation_fraction"] for r in rows],label=label)
        axes[1,0].plot(time,[r["stress_Pa"]*1e-9 for r in rows],label=label)
        axes[1,1].plot(time,[r["temperature_peak_K"] for r in rows],label=label)
        axes[2,0].plot(time,[r["absolute_shear_rate"]["maximum"] for r in rows],label=label)
        axes[2,1].plot(time,[r["work_conjugate_plastic_power"]["negative_cell_fraction"] for r in rows],label=label)
    labels=(("peak - mean temperature (K)","power participation fraction"),
            ("mean stress (GPa)","peak temperature (K)"),
            ("maximum |slip rate| (s$^{-1}$)","negative-power cell fraction"))
    for i in range(3):
        for j in range(2): axes[i,j].set_ylabel(labels[i][j]);axes[i,j].grid(alpha=.25)
    axes[2,0].set_xlabel("physical time (microseconds)");axes[2,1].set_xlabel("physical time (microseconds)")
    axes[0,0].legend();fig.tight_layout();output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180);plt.close(fig)


def plot_asb_fields(decision_path: Path, output: Path):
    decision=json.loads(decision_path.read_text())
    fig,axes=plt.subplots(2,4,figsize=(14,6),constrained_layout=True)
    keys=("T","asb_last_plastic_power_W_m3","asb_last_heat_production_W_m3","asb_last_gdot_abs")
    titles=("temperature (K)","plastic power (W m$^{-3}$)","heat production (W m$^{-3}$)","|slip rate| (s$^{-1}$)")
    for i,name in enumerate(("baseline","control")):
        checkpoint=Path(decision["trajectories"][name][-1]["checkpoint"])
        with np.load(checkpoint,allow_pickle=True) as raw:
            for j,(key,title) in enumerate(zip(keys,titles)):
                im=axes[i,j].imshow(np.asarray(raw[key]),origin="lower",cmap="viridis")
                axes[i,j].set_title(("feedback: " if i==0 else "control: ")+title)
                fig.colorbar(im,ax=axes[i,j],shrink=.75)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)


def plot_asb_common_scale_fields(decision_path: Path, output: Path):
    """Compare causal endpoints on shared scales plus signed differences."""
    decision=json.loads(decision_path.read_text())
    names=("baseline","control")
    checkpoints=[Path(decision["trajectories"][name][-1]["checkpoint"])
                 for name in names]
    keys=("T","asb_last_plastic_power_W_m3",
          "asb_last_heat_production_W_m3","asb_last_gdot_abs")
    titles=("temperature (K)","plastic power (W m$^{-3}$)",
            "heat production (W m$^{-3}$)","|slip rate| (s$^{-1}$)")
    fields={}; accumulated=[]; extent=None
    for name,checkpoint in zip(names,checkpoints):
        with np.load(checkpoint,allow_pickle=True) as raw:
            fields[name]=[np.asarray(raw[key],dtype=float) for key in keys]
            parameters=json.loads(str(raw["P_json"].item()))
            length_um=float(parameters["L_phys"])*1e6
            extent=(0.0,length_um,0.0,length_um)
        total=None; previous_time=0.0
        for row in decision["trajectories"][name]:
            with np.load(Path(row["checkpoint"]),allow_pickle=True) as raw:
                rate=np.asarray(raw["asb_last_gdot_abs"],dtype=float)
            dt=float(row["physical_time_s"])-previous_time
            total=rate*dt if total is None else total+rate*dt
            previous_time=float(row["physical_time_s"])
        accumulated.append(total)
        fields[name].append(total)
    titles=titles+("sampled accumulated |slip|",)
    fig,axes=plt.subplots(3,len(titles),figsize=(18,10),constrained_layout=True)
    for j,title in enumerate(titles):
        pair=[fields[name][j] for name in names]
        vmin=min(float(np.min(value)) for value in pair)
        vmax=max(float(np.max(value)) for value in pair)
        for i,(name,value) in enumerate(zip(names,pair)):
            image=axes[i,j].imshow(value.T,origin="lower",extent=extent,
                                   cmap="viridis",vmin=vmin,vmax=vmax)
            axes[i,j].set_title(("feedback: " if name=="baseline" else
                                 "frozen-flow: ")+title)
            fig.colorbar(image,ax=axes[i,j],shrink=.72)
        difference=pair[0]-pair[1]
        limit=max(float(np.max(np.abs(difference))),1e-300)
        image=axes[2,j].imshow(difference.T,origin="lower",extent=extent,
                               cmap="coolwarm",vmin=-limit,vmax=limit)
        axes[2,j].set_title("feedback - control: "+title)
        fig.colorbar(image,ax=axes[2,j],shrink=.72)
        axes[2,j].set_xlabel("x (micrometres)")
        for i in range(3): axes[i,j].set_ylabel("y (micrometres)")
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180);plt.close(fig)


def plot_drx(decision_path: Path, output: Path):
    decision=json.loads(decision_path.read_text())
    fig,axes=plt.subplots(3,2,figsize=(10,9),sharex=True)
    for case,label in (("misoriented_front","front enabled"),
                       ("misoriented_front_disabled","front disabled")):
        # The independently postprocessed decision may live above the execution
        # directory.  Its case record is the authoritative result address.
        result_path=Path(decision["cases"][case]["result"])
        result=json.loads(result_path.read_text())
        rows=result["records"];time=np.asarray([r["physical_time_end_s"] for r in rows])*1e6
        axes[0,0].plot(time,np.cumsum([r["accepted_contour_displacement_m"] for r in rows])*1e9,label=label)
        axes[0,1].plot(time,[r["cumulative_newly_swept_volume_m3"] for r in rows],label=label)
        axes[1,0].plot(time,[r["cumulative_processed_line_m"] for r in rows],label=label)
        axes[1,1].plot(time,[r["cumulative_boundary_stored_line_m"] for r in rows],label=label)
        axes[2,0].plot(time,[r["mean_shear_stress_Pa"]*1e-9 for r in rows],label=label)
        axes[2,1].plot(time,[r["child_owner_total_line_density_mean_m2"] for r in rows],label=label)
    labels=(("signed contour displacement (nm)","newly swept volume (m$^3$)"),
            ("processed line (m)","boundary-stored line (m)"),
            ("mean shear stress (GPa)","child-owner line density (m$^{-2}$)"))
    for i in range(3):
        for j in range(2):axes[i,j].set_ylabel(labels[i][j]);axes[i,j].grid(alpha=.25)
    axes[2,0].set_xlabel("physical time (microseconds)");axes[2,1].set_xlabel("physical time (microseconds)")
    axes[0,0].legend();fig.tight_layout();output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--asb-decision",type=Path)
    parser.add_argument("--drx-decision",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.asb_decision:
        plot_asb(args.asb_decision,args.output_dir/"v54_asb_evolution.png")
        plot_asb_fields(args.asb_decision,args.output_dir/"v54_asb_endpoint_fields.png")
        plot_asb_common_scale_fields(
            args.asb_decision,args.output_dir/"v55_asb_common_scale_fields.png")
    if args.drx_decision:
        plot_drx(args.drx_decision,args.output_dir/"v54_drx_evolution.png")


if __name__=="__main__":main()
