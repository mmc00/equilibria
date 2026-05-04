"""Compare Python baseline (no shock) values vs GAMS NEOS baseline (t='base').

Reads /Users/marmol/proyectos2/equilibria/output/nus333_neos/out.gdx via gdxdump
and reports the largest absolute and relative discrepancies for the core
endogenous variables. The baseline solve uses the same closure as the shock
script, but tariff power shocks are NOT applied.
"""
from __future__ import annotations
import csv
import io
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pyomo.environ import value  # noqa: E402

GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"
OUT_GDX = ROOT / "output" / "nus333_neos" / "out.gdx"
NUS333 = Path("/Users/marmol/Downloads/10284")

CORE_VARS = [
    "gdpmp", "regY", "u", "pnum", "pwfact", "pgdpmp", "pop",
    "yi", "rsav", "savf", "kstock",
    "xp", "xs", "xmt", "pmt",
    "xa", "xd", "xm", "pa",
    "xf", "pf", "pi",
    "pe", "pm", "pd",
    "pp", "xw",
]


def load_gams_var(name: str) -> dict[tuple, float]:
    """Read a variable from out.gdx for t='base' (or only t)."""
    res = subprocess.run(
        [GDXDUMP, str(OUT_GDX), f"Symb={name}", "Format=csv"],
        capture_output=True, text=True, check=True,
    )
    reader = csv.reader(io.StringIO(res.stdout))
    header = next(reader, None)
    if header is None:
        return {}
    n = len(header)
    val_idx = n - 1
    t_idx = None
    for i, h in enumerate(header):
        if h.strip('"').lower() == "t":
            t_idx = i
            break
    out: dict[tuple, float] = {}
    for row in reader:
        if not row or len(row) < 2:
            continue
        if t_idx is not None and row[t_idx].strip('"') != "base":
            continue
        try:
            v = float(row[val_idx])
        except ValueError:
            continue
        if t_idx is None:
            key = tuple(c.strip('"') for c in row[:val_idx])
        else:
            key = tuple(c.strip('"') for i, c in enumerate(row[:val_idx]) if i != t_idx)
        out[key] = v
    return out


def solve_python_baseline():
    """Solve Python baseline (no shock applied) — replicates compare_nus333_vs_neos.main()."""
    sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
    from compare_nus333_vs_neos import _solve  # type: ignore
    from equilibria.templates.gtap.gtap_parameters import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from equilibria.templates.gtap.gtap_solver import GTAPClosureConfig

    params = GTAPParameters()
    params.load_from_har(
        basedata_path=NUS333 / "basedata.har",
        sets_path=NUS333 / "sets.har",
        default_path=NUS333 / "default.prm",
        baserate_path=NUS333 / "baserate.har",
    )
    closure = GTAPClosureConfig(if_sub=False)
    builder = GTAPModelEquations(params.sets, params, residual_region="ROW", closure=closure)
    model = builder.build_model()
    _solve(model, params, label="baseline")
    return model


def collect_python_vars(model) -> dict[str, dict[tuple, float]]:
    """Gather variable levels from solved Pyomo model."""
    from pyomo.environ import Var
    out: dict[str, dict[tuple, float]] = {}
    for var in model.component_objects(Var, active=True):
        d: dict[tuple, float] = {}
        for idx in var:
            try:
                v = value(var[idx])
            except Exception:
                continue
            if v is None:
                continue
            key = idx if isinstance(idx, tuple) else (idx,) if idx is not None else ()
            d[key] = float(v)
        out[var.local_name] = d
    return out


def normalize_key(key: tuple, var_name: str, py: bool) -> tuple:
    """Normalize keys so Python (r,i,rp) and GAMS (rp,i,r) for xw/pe etc align.

    GAMS exports xw as (rp, i, r); Python stores xw as (rp, i, r) too — let's
    just compare keys as-is and report mismatches when sets differ.
    """
    return key


def compare(name_py: str, py_vals: dict, gams_vals: dict) -> list[tuple]:
    rows = []
    keys = set(py_vals) | set(gams_vals)
    for k in keys:
        p = py_vals.get(k)
        g = gams_vals.get(k)
        if p is None or g is None:
            rows.append((k, p, g, None, None))
            continue
        diff = p - g
        if abs(g) > 1e-9:
            rel = diff / abs(g)
        else:
            rel = float("nan") if abs(p) > 1e-9 else 0.0
        rows.append((k, p, g, diff, rel))
    return rows


def main():
    print("Loading GAMS baseline (t='base') from", OUT_GDX)
    gams = {n: load_gams_var(n) for n in CORE_VARS}
    for n, d in gams.items():
        print(f"  {n}: {len(d)} keys")

    model = solve_python_baseline()
    py = collect_python_vars(model)

    print("\n=== TOP DISCREPANCIES PER VARIABLE (baseline) ===")
    print(f"{'var':>8} {'key':>25} {'py':>12} {'gams':>12} {'diff':>12} {'rel%':>10}")
    print("-" * 80)
    summary = []
    for name in CORE_VARS:
        py_d = py.get(name)
        if py_d is None:
            print(f"  ! {name}: not in Python model")
            continue
        rows = compare(name, py_d, gams[name])
        rows_with_data = [r for r in rows if r[3] is not None]
        rows_with_data.sort(key=lambda r: -abs(r[4]) if r[4] is not None and not math.isnan(r[4]) else 0)
        max_rel = max((abs(r[4]) for r in rows_with_data if r[4] is not None and not math.isnan(r[4])), default=0.0)
        summary.append((name, max_rel, len(rows_with_data), len(rows)))
        for r in rows_with_data[:3]:
            k, p, g, d, rel = r
            print(f"{name:>8} {str(k):>25} {p:>12.5g} {g:>12.5g} {d:>12.3g} {rel*100:>9.2f}%")
    print()
    print("=== SUMMARY (sorted by max rel discrepancy) ===")
    summary.sort(key=lambda x: -x[1])
    print(f"{'var':>8} {'max_rel%':>10} {'matched':>8} {'total':>8}")
    for name, mr, m, t in summary:
        print(f"{name:>8} {mr*100:>9.2f}% {m:>8} {t:>8}")


if __name__ == "__main__":
    main()
