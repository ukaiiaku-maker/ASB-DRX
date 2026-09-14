#!/usr/bin/env python3
"""Periodically resample a full-v34 checkpoint without changing its domain."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy import ndimage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--grid", type=int, required=True)
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=True) as source:
        old = np.asarray(source["rho"]).shape
        if old[0] != old[1]:
            raise ValueError("only square production checkpoints are supported")
        ratio = args.grid/old[0]
        output = {}
        for key in source.files:
            value = np.asarray(source[key])
            if value.ndim >= 2 and value.shape[:2] == old:
                order = 0 if value.dtype.kind in "biu" else 1
                zoom = (ratio, ratio)+(1.0,)*(value.ndim-2)
                value = ndimage.zoom(
                    value, zoom, order=order, mode="grid-wrap",
                    prefilter=False, grid_mode=True)
                if source[key].dtype.kind == "b":
                    value = value.astype(bool)
                elif source[key].dtype.kind in "iu":
                    value = np.rint(value).astype(source[key].dtype)
            output[key] = value
        eta = output.get("eta")
        if eta is not None:
            output["eta"] = eta/np.maximum(np.sum(eta, axis=2, keepdims=True), 1e-300)
            output["lab"] = np.argmax(output["eta"], axis=2).astype(output["lab"].dtype)
        if "P_json" in output:
            params = json.loads(str(output["P_json"]))
            params.update(Nx=args.grid, Ny=args.grid)
            output["P_json"] = np.array(json.dumps(params, sort_keys=True))
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=args.destination.name+".", suffix=".npz", dir=args.destination.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **output)
        with np.load(temporary, allow_pickle=True) as check:
            if check["rho"].shape != (args.grid, args.grid):
                raise RuntimeError("resampled checkpoint verification failed")
        os.replace(temporary, args.destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    main()
