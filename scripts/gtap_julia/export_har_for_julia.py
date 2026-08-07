"""Export one of our aggregated GTAP datasets (basedata.har / default.prm /
sets.har) to flat CSVs the Julia port can read to build (hData, hParameters,
hSets) NamedArrays — bypassing Julia's HeaderArrayFile dependency and its raw-data
aggregation step (our data is already aggregated).

Emits <out_dir>/{data.csv, params.csv, sets.csv}:
  data.csv:   header,idx1,idx2,...,value   (lower-case Julia header names)
  params.csv: header,idx1,...,value
  sets.csv:   setname,i,member

Usage: uv run python scripts/gtap_julia/export_har_for_julia.py <dataset> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from equilibria.babel.har import read_har

# Julia header (lower) -> our HAR header (UPPER). Data + parameters.
_DATA = [
    "evfb", "evfp", "evos", "makb", "maks", "pop", "save", "vcif", "vdep",
    "vdfb", "vdfp", "vdgb", "vdgp", "vdib", "vdip", "vdpb", "vdpp", "vfob",
    "vkb", "vmfb", "vmfp", "vmgb", "vmgp", "vmib", "vmip", "vmpb", "vmpp",
    "vmsb", "vst", "vtwr", "vxsb",
]
_PARAMS = [
    "eflg", "esbc", "esbd", "esbg", "esbm", "esbq", "esbs", "esbt", "esbv",
    "etre", "etrq", "incp", "rdlt", "rflx", "subp",
]
_SETS = {
    "reg": "REG", "comm": "COMM", "acts": "ACTS", "endw": "ENDW", "marg": "MARG",
}


def _root(dataset: str) -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "datasets").is_dir():
            return p / "datasets" / dataset
    raise FileNotFoundError("datasets/ not found")


def _dump_har(har, headers: list[str], out: Path) -> None:
    with out.open("w") as f:
        for h in headers:
            ha = har.get(h.upper())
            if ha is None:
                continue
            arr = np.asarray(ha.array).astype(float)
            # etre: mobile factors carry 0.0 in our HAR, but Julia's calibration
            # does (·)^(1/etre) and 0 → division-by-zero. Julia's own sample uses a
            # tiny negative dummy (~-1e-5) for non-sluggish factors; mirror that.
            if h == "etre":
                arr = np.where(arr == 0.0, -1e-5, arr)
            # Bilateral trade self-cells: GTAPAgg records the self-trade diagonal
            # VXSB/VCIF/VMSB/VFOB[c,r,r] as ~2e-5 (a numerical-zero placeholder —
            # domestic sales live in the VD* headers, not the import matrix). Julia's
            # log-CES Armington nest keeps that cell active (δ_qxs = α_qxs≠0), calibrates
            # an inconsistent tiny share, and IPOPT stalls (esubm~7.9 → singular log
            # derivative). Native GTAP masks it via xwFlag=1 ⟺ VXSB>0; snap the sub-ε
            # placeholder to exact 0.0 so δ_qxs drops it, while market-clearing sums see
            # a valid 0 (not NaN → no "Invalid number"). Threshold 1e-3 ≫ the ~2e-5
            # placeholder and ≪ any real trade flow (millions).
            if h in ("vxsb", "vcif", "vmsb", "vfob"):
                arr = np.where(np.abs(arr) < 1e-3, 0.0, arr)
            if arr.ndim == 0:
                f.write(f"{h},{float(arr)}\n")
                continue
            for idx in np.ndindex(arr.shape):
                v = arr[idx]
                f.write(f"{h}," + ",".join(str(i) for i in idx) + f",{v}\n")


def export(dataset: str, out_dir: Path | str) -> dict[str, Path]:
    d = _root(dataset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = read_har(str(d / "basedata.har"))
    prm = read_har(str(d / "default.prm"))
    sets = read_har(str(d / "sets.har"))

    data_csv = out / "data.csv"
    par_csv = out / "params.csv"
    set_csv = out / "sets.csv"

    _dump_har(base, _DATA, data_csv)
    _dump_har(prm, _PARAMS, par_csv)
    with set_csv.open("w") as f:
        for sname, upper in _SETS.items():
            ha = sets.get(upper)
            if ha is None:
                continue
            for i, m in enumerate(np.asarray(ha.array).ravel().tolist(), 1):
                f.write(f"{sname},{i},{m}\n")
    return {"data": data_csv, "params": par_csv, "sets": set_csv}


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "gtap7_15x10"
    od = sys.argv[2] if len(sys.argv) > 2 else "."
    paths = export(ds, od)
    for k, v in paths.items():
        print(f"{k}: {v}")
