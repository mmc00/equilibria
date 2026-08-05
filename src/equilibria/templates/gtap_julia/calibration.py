"""Calibrated point for the gtap_julia port.

The Julia calibration (generate_calibration_inputs + a solve) backs out the CES
share parameters (α_*, γ_*, σ_*, ϵ_*, β_qpa, ...) and seeded quantities from the
benchmark data. Re-deriving that ~60-line share back-out in Python adds no value
and risks subtle bugs; instead the port loads Julia's OWN calibrated point (the
spec's source of truth). `dump_calibrated.jl` emits it; `load_calibrated` reads it.

A calibrated value is stored as a dict {index_tuple: value} (indexed params) or a
0-d array (scalars). The equation groups look up cells by their (set-elem) tuple,
matching Pyomo's indexing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parents[3] / "scripts" / "gtap_julia" / "dump_calibrated.jl"
_JULIA = Path.home() / ".juliaup" / "bin" / "julia"
_PKG = Path.home() / "proyectos" / "GlobalTradeAnalysisProjectModelV7.jl"


def load_calibrated(csv: Path | str) -> dict[str, Any]:
    """Parse a calibrated-point CSV into {name: {idx_tuple: val}} / scalar arrays.

    Rows: ``name,i1,i2,...,value`` (indexed) or ``name,value`` (scalar).
    """
    out: dict[str, Any] = {}
    scalars: dict[str, float] = {}
    for line in Path(csv).read_text().splitlines():
        if not line or line.startswith(">>>"):
            continue
        parts = line.split(",")
        name = parts[0]
        try:
            val = float(parts[-1])
        except ValueError:
            continue
        idx = tuple(parts[1:-1])
        if idx:
            out.setdefault(name, {})[idx] = val
        else:
            scalars[name] = val
    result: dict[str, Any] = dict(out)
    for name, v in scalars.items():
        result[name] = np.asarray(v)
    return result


def dump_and_load_calibrated(
    dataset: str = "sample",
    out_dir: Path | str | None = None,
    timeout: int = 900,
) -> Path:
    """Run the Julia calibration dumper and return the CSV path."""
    if not _JULIA.exists():
        raise FileNotFoundError(f"julia not found at {_JULIA}")
    out = Path(out_dir) if out_dir is not None else _HERE
    out.mkdir(parents=True, exist_ok=True)
    csv = (
        out
        / f"julia_{'sample' if dataset == 'sample' else Path(dataset).name}_calibrated.csv"
    )
    res = subprocess.run(
        [str(_JULIA), f"--project={_PKG}", str(_SCRIPT), dataset, str(csv)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if res.returncode != 0 or ">>> DONE" not in res.stdout:
        raise RuntimeError(
            f"Julia calibration dump failed (rc={res.returncode}).\n"
            f"stdout tail:\n{res.stdout[-2000:]}\nstderr tail:\n{res.stderr[-2000:]}"
        )
    return csv
