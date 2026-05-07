"""Full variable-by-variable diff: Python GTAP 9x10 vs GAMS COMP.gdx.

Iterates every Var symbol in COMP.gdx, looks for a matching Pyomo Var on
the Python model, aligns indices (dropping the GAMS time axis t∈{base,
check, shock}), and reports counts of matching/diverging cells per
variable. Symbols that don't exist on either side, or that are constant
zero, are reported as 'skipped'.
"""
from __future__ import annotations
import argparse, csv, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

GDX = ROOT / "src/equilibria/templates/reference/gtap/data/basedata-9x10.gdx"
GAMS_COMP = ROOT / "src/equilibria/templates/reference/gtap/output/COMP.gdx"
GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump"

T_LABELS = {"base", "check", "shock"}


def list_populated_vars() -> list[str]:
    out = subprocess.run(
        [GDXDUMP, str(GAMS_COMP), "Symbols"],
        capture_output=True, text=True, check=True,
    ).stdout
    # Format: "  N  Name  Dim  Type  Records  Explanatory text"
    # parts[0]=N, parts[1]=Name, parts[2]=Dim, parts[3]=Type, parts[4]=Records
    names = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[3] != "Var":
            continue
        try:
            n_records = int(parts[4])
        except ValueError:
            continue
        if n_records > 0:
            names.append(parts[1])
    return names


def gams_levels(symbol: str) -> dict:
    """Return {tuple_keys: float_value} for all entries of `symbol`."""
    res = subprocess.run(
        [GDXDUMP, str(GAMS_COMP), "Format=csv", f"Symb={symbol}"],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return {}
    out: dict = {}
    reader = csv.reader(res.stdout.splitlines())
    next(reader, None)
    for row in reader:
        if len(row) < 2:
            continue
        *keys, val = row
        keys = tuple(k.strip('"') for k in keys)
        try:
            out[keys] = float(val)
        except ValueError:
            pass
    return out


def split_t(keys: tuple) -> tuple[tuple, str | None]:
    """If the last key is base/check/shock, peel it off as the t-label."""
    if keys and keys[-1] in T_LABELS:
        return keys[:-1], keys[-1]
    return keys, None


def get_py_var_value(py_var, key: tuple) -> float | None:
    """Fetch level from a Pyomo Var indexed by `key` (or a single scalar)."""
    from pyomo.core import value
    try:
        if not key:
            return float(value(py_var))
        if len(key) == 1:
            k0 = key[0]
            if k0 in py_var:
                return float(value(py_var[k0]))
            for kk in py_var:
                if str(kk) == k0:
                    return float(value(py_var[kk]))
            return None
        if key in py_var:
            return float(value(py_var[key]))
        for kk in py_var:
            if isinstance(kk, tuple) and len(kk) == len(key) and tuple(str(x) for x in kk) == key:
                return float(value(py_var[kk]))
        return None
    except Exception:
        return None


def find_py_var(model, name: str):
    """Try the literal name, then lowercase, then a case-insensitive scan."""
    v = getattr(model, name, None)
    if v is not None:
        return v, name
    v = getattr(model, name.lower(), None)
    if v is not None:
        return v, name.lower()
    target = name.lower()
    from pyomo.environ import Var
    for comp in model.component_objects(Var, active=True):
        if comp.name.lower() == target:
            return comp, comp.name
    return None, None


def compare_phase(model_py, gams_all: dict, t_label: str, tol_rel: float, tol_abs: float):
    """Compare GAMS values (filtered to t_label) against the Python model.

    Returns dict with: n_total, n_match, n_diverge, n_missing, max_abs, max_rel,
    worst (gams_key, py_val, gams_val, abs, rel).
    """
    n_total = 0
    n_match = 0
    n_diverge = 0
    n_missing = 0
    max_abs = 0.0
    max_rel = 0.0
    worst = None
    for full_key, g_val in gams_all.items():
        body, t = split_t(full_key)
        if t is not None and t != t_label:
            continue
        n_total += 1
        p_val = get_py_var_value(model_py, body) if body else None
        if p_val is None and not body:
            try:
                from pyomo.core import value
                p_val = float(value(model_py))
            except Exception:
                p_val = None
        if p_val is None:
            n_missing += 1
            continue
        d = p_val - g_val
        rel = abs(d) / abs(g_val) if abs(g_val) > 1e-12 else (0.0 if abs(d) < tol_abs else float("inf"))
        if abs(d) <= tol_abs or rel <= tol_rel:
            n_match += 1
        else:
            n_diverge += 1
        if abs(d) > max_abs:
            max_abs = abs(d)
        if rel != float("inf") and rel > max_rel:
            max_rel = rel
        if worst is None or abs(d) > abs(worst[3]):
            worst = (full_key, p_val, g_val, d, rel)
    return {
        "n_total": n_total, "n_match": n_match, "n_diverge": n_diverge,
        "n_missing": n_missing, "max_abs": max_abs, "max_rel": max_rel,
        "worst": worst,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["base", "shock", "both"], default="both")
    ap.add_argument("--tol-rel", type=float, default=1e-3,
                    help="Relative tolerance for match (default 0.1%%)")
    ap.add_argument("--tol-abs", type=float, default=1e-6,
                    help="Absolute tolerance fallback (default 1e-6)")
    ap.add_argument("--show-worst", action="store_true",
                    help="Print the single worst cell for each diverging symbol")
    args = ap.parse_args()

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from run_gtap import _run_path_capi_nonlinear_full, _build_gtap_contract_with_calibration

    contract = _build_gtap_contract_with_calibration("gtap_standard7_9x10")

    print("=== Python baseline 9x10 ===")
    p_b = GTAPParameters()
    p_b.load_from_gdx(GDX)
    eq_b = GTAPModelEquations(p_b.sets, p_b, contract.closure)
    m_b = eq_b.build_model()
    r_b = _run_path_capi_nonlinear_full(
        m_b, p_b, enforce_post_checks=False, strict_path_capi=False,
        closure_config=contract.closure, equation_scaling=True,
    )
    print(f"  baseline residual={r_b.get('residual'):.3e}  code={r_b.get('termination_code')}")

    m_s = None
    if args.phase != "base":
        print("\n=== Python shock 9x10 (10% imptx, rate scaling) ===")
        p_s = GTAPParameters()
        p_s.load_from_gdx(GDX)
        for k in list(p_s.taxes.imptx.keys()):
            p_s.taxes.imptx[k] = float(p_s.taxes.imptx[k]) * 1.10
        eq_s = GTAPModelEquations(
            p_s.sets, p_s, contract.closure,
            is_counterfactual=True, t0_snapshot=m_b,
        )
        m_s = eq_s.build_model()
        from pyomo.environ import Var
        from pyomo.core import value
        for comp in m_b.component_objects(Var, active=True):
            dst = getattr(m_s, comp.name, None)
            if dst is None:
                continue
            for idx in comp:
                try:
                    v = float(value(comp[idx]))
                    if dst[idx].lb is not None and v < float(dst[idx].lb): v = float(dst[idx].lb)
                    if dst[idx].ub is not None and v > float(dst[idx].ub): v = float(dst[idx].ub)
                    dst[idx].set_value(v)
                except Exception:
                    pass
        r_s = _run_path_capi_nonlinear_full(
            m_s, p_s, enforce_post_checks=False, strict_path_capi=False,
            closure_config=contract.closure, equation_scaling=True,
        )
        print(f"  shock residual={r_s.get('residual'):.3e}  code={r_s.get('termination_code')}")

    var_names = list_populated_vars()
    print(f"\nPopulated GAMS Vars in COMP.gdx: {len(var_names)}")

    phases = [("base", m_b)]
    if args.phase != "base":
        phases.append(("shock", m_s))

    for phase, m_py in phases:
        print(f"\n{'='*120}")
        print(f"PHASE: {phase}    (tol_rel={args.tol_rel}  tol_abs={args.tol_abs})")
        print(f"{'='*120}")
        print(f"{'gams_var':<14s} {'py_var':<14s} {'cells':>7s} {'match':>7s} {'diverge':>8s} {'missing':>8s} {'max_abs':>10s} {'max_rel':>10s}  status")
        print("-" * 120)
        agg = {"vars_total": 0, "vars_match_all": 0, "vars_partial": 0,
               "vars_no_py": 0, "cells_total": 0, "cells_match": 0,
               "cells_diverge": 0, "cells_missing": 0}
        diverge_details = []
        for name in var_names:
            agg["vars_total"] += 1
            gams_all = gams_levels(name)
            if not gams_all:
                continue
            py_var, py_name = find_py_var(m_py, name)
            if py_var is None:
                # Count GAMS cells for this t to know coverage gap
                n_t = sum(1 for k in gams_all if split_t(k)[1] == phase)
                if n_t == 0:
                    n_t = len(gams_all)
                print(f"{name:<14s} {'<n/a>':<14s} {n_t:>7d} {0:>7d} {0:>8d} {n_t:>8d} {'—':>10s} {'—':>10s}  no-py")
                agg["vars_no_py"] += 1
                agg["cells_total"] += n_t
                agg["cells_missing"] += n_t
                continue
            stats = compare_phase(py_var, gams_all, phase,
                                  tol_rel=args.tol_rel, tol_abs=args.tol_abs)
            agg["cells_total"] += stats["n_total"]
            agg["cells_match"] += stats["n_match"]
            agg["cells_diverge"] += stats["n_diverge"]
            agg["cells_missing"] += stats["n_missing"]
            if stats["n_total"] == 0:
                continue
            if stats["n_diverge"] == 0 and stats["n_missing"] == 0:
                agg["vars_match_all"] += 1
                status = "ok"
            else:
                agg["vars_partial"] += 1
                status = "diff" if stats["n_diverge"] else "miss"
                if stats["worst"]:
                    diverge_details.append((name, py_name, stats))
            print(f"{name:<14s} {py_name or '?':<14s} {stats['n_total']:>7d} "
                  f"{stats['n_match']:>7d} {stats['n_diverge']:>8d} "
                  f"{stats['n_missing']:>8d} {stats['max_abs']:>10.2e} "
                  f"{stats['max_rel']:>10.2e}  {status}")

        print("-" * 120)
        print(f"  Vars total:           {agg['vars_total']}")
        print(f"  Vars all-match:       {agg['vars_match_all']}")
        print(f"  Vars partial/diverge: {agg['vars_partial']}")
        print(f"  Vars not in Python:   {agg['vars_no_py']}")
        print(f"  Cells total:          {agg['cells_total']}")
        print(f"  Cells match:          {agg['cells_match']}")
        print(f"  Cells diverge:        {agg['cells_diverge']}")
        print(f"  Cells missing/no-py:  {agg['cells_missing']}")
        coverage = (agg["cells_match"] / agg["cells_total"] * 100.0) if agg["cells_total"] else 0.0
        print(f"  Match rate:           {coverage:.2f}%")

        if args.show_worst and diverge_details:
            print(f"\n  Worst diverging cell per variable (phase={phase}):")
            for name, py_name, stats in diverge_details[:30]:
                w = stats["worst"]
                if w is None:
                    continue
                key, p_val, g_val, d, rel = w
                rel_str = f"{rel*100:.3f}%" if rel != float("inf") else "inf"
                print(f"    {name:<12s} {str(key):<60s}  py={p_val:+.6e}  gams={g_val:+.6e}  Δ={d:+.3e}  rel={rel_str}")


if __name__ == "__main__":
    main()
