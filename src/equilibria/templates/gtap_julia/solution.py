"""Load Julia's full solved point and seed a Pyomo model with it.

Used by the per-equation residual tests (Tasks 6-11): a ported equation group is
validated by seeding a model to Julia's solution and asserting the constraint
residuals are ~0. dump_solution.jl emits variables (prefix 'd'), parameters
(prefix 'p') and sets (prefix 'SET_<name>').
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pyomo.environ as pyo

from .variables import build_variables

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE.parents[3] / "scripts" / "gtap_julia" / "dump_solution.jl"
_JULIA = Path.home() / ".juliaup" / "bin" / "julia"
_PKG = Path.home() / "proyectos" / "GlobalTradeAnalysisProjectModelV7.jl"


def dump_solution(
    dataset: str = "sample", out_dir: Path | str | None = None, timeout: int = 900
) -> Path:
    """Run the Julia full-solution dumper, return the CSV path."""
    if not _JULIA.exists():
        raise FileNotFoundError(f"julia not found at {_JULIA}")
    out = Path(out_dir) if out_dir is not None else _HERE
    out.mkdir(parents=True, exist_ok=True)
    slug = "sample" if dataset == "sample" else Path(dataset).name
    csv = out / f"julia_{slug}_solution.csv"
    res = subprocess.run(
        [str(_JULIA), f"--project={_PKG}", str(_SCRIPT), dataset, str(csv)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if res.returncode != 0 or ">>> DONE" not in res.stdout:
        raise RuntimeError(
            f"Julia solution dump failed (rc={res.returncode}).\n"
            f"stdout tail:\n{res.stdout[-2000:]}\nstderr tail:\n{res.stderr[-2000:]}"
        )
    return csv


def load_solution(csv: Path | str) -> dict[str, Any]:
    """Parse the dump into a structured dict.

    Returns ``{"var": {name: {idx: val}|scalar}, "par": {...}, "sets": {name: [members]}}``.
    Variables (prefix 'd') and parameters (prefix 'p') are merged into one flat
    lookup under both "all" (name -> {idx: val}) and split by kind.
    """
    var: dict[str, Any] = {}
    par: dict[str, Any] = {}
    sets: dict[str, list[str]] = {}
    # dump_solution.jl writes data with real key names (no 'd,'/'p,' prefix in the
    # value rows — the prefix arg is unused there), so we bucket by known set rows.
    for line in Path(csv).read_text().splitlines():
        if not line or line.startswith(">>>"):
            continue
        parts = line.split(",")
        name = parts[0]
        if name.startswith("SET_"):
            sets.setdefault(name[4:], []).append(parts[-1])
            continue
        try:
            val = float(parts[-1])
        except ValueError:
            continue
        idx = tuple(parts[1:-1])
        target = (
            var  # data + params share the flat namespace; equations look up by name
        )
        if idx:
            target.setdefault(name, {})[idx] = val
        else:
            target[name] = val
    return {"all": var, "par": par, "sets": sets}


def build_sets_on(model, sol: dict[str, Any]) -> dict[str, list[str]]:
    """Attach Pyomo Sets from the dumped set members and return the members dict."""
    sets = sol["sets"]
    # Julia set names are lower-case already (reg, comm, acts, endw, marg, endws...)
    return sets


def seed_model(model, sol: dict[str, Any]) -> None:
    """Build variables and seed each cell to Julia's solved value."""
    sets = sol["sets"]
    build_variables(model, sets)
    allvals = sol["all"]
    for vname in list(model.component_map(pyo.Var)):
        v = getattr(model, vname)
        data = allvals.get(vname)
        if data is None:
            continue
        if isinstance(data, dict):
            for idx in v:
                key = idx if isinstance(idx, tuple) else (idx,)
                skey = tuple(str(k) for k in key)
                val = data.get(skey)
                if val is not None and val == val:  # skip missing / NaN
                    v[idx].set_value(val)
        elif data == data:  # scalar, non-NaN
            v.set_value(float(data))
