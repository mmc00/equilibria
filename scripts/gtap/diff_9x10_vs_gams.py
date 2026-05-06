"""Compare Python 9x10 baseline + shocked solutions vs GAMS COMP.gdx.

Reads variable levels from COMP.gdx via gdxdump CSV and matches them against
Python's solver output side by side, region by region. Reports Δ% per macro
to localize which regions / variables diverge.
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

REGIONS = ["Oceania", "EastAsia", "SEAsia", "SouthAsia", "NAmerica",
           "LatinAmer", "EU_28", "MENA", "SSA", "RestofWorld"]


def gams_levels(symbol: str, t_filter: str | None = None) -> dict:
    """Read GAMS variable levels via gdxdump CSV. Returns {key_tuple: value}."""
    res = subprocess.run(
        [GDXDUMP, str(GAMS_COMP), "Format=csv", f"Symb={symbol}"],
        capture_output=True, text=True, check=True,
    )
    out: dict = {}
    reader = csv.reader(res.stdout.splitlines())
    header = next(reader, None)
    for row in reader:
        *keys, val = row
        keys = tuple(k.strip('"') for k in keys)
        if t_filter and keys[-1] != t_filter:
            continue
        try:
            out[keys] = float(val)
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["base", "shock", "both"], default="both")
    args = ap.parse_args()

    from pyomo.core import value
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from run_gtap import _run_path_capi_nonlinear_full, _build_gtap_contract_with_calibration

    contract = _build_gtap_contract_with_calibration("gtap_standard7_9x10")

    print("=== Solving Python baseline 9x10 ===")
    p_b = GTAPParameters()
    p_b.load_from_gdx(GDX)
    eq_b = GTAPModelEquations(p_b.sets, p_b, contract.closure)
    m_b = eq_b.build_model()
    r_b = _run_path_capi_nonlinear_full(
        m_b, p_b, enforce_post_checks=False, strict_path_capi=False,
        closure_config=contract.closure, equation_scaling=True,
    )
    print(f"  baseline residual={r_b.get('residual'):.3e}  code={r_b.get('termination_code')}")

    if args.phase != "base":
        print("\n=== Solving Python shock 9x10 (10% imptx) ===")
        p_s = GTAPParameters()
        p_s.load_from_gdx(GDX)
        # Rate scaling matches tariff_sim.gms:77 `tm = tm_base * (1 + tm_shock)`.
        for k in list(p_s.taxes.imptx.keys()):
            p_s.taxes.imptx[k] = float(p_s.taxes.imptx[k]) * 1.10
        eq_s = GTAPModelEquations(
            p_s.sets, p_s, contract.closure,
            is_counterfactual=True, t0_snapshot=m_b,
        )
        m_s = eq_s.build_model()
        from pyomo.environ import Var
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

    # Compare
    macros_1d = ["regy", "yc", "yg", "yi", "rsav", "gdpmp", "rgdpmp", "pgdpmp",
                 "pabs", "pcons", "pi", "pfact", "kstock", "u", "uh", "ug", "us",
                 "facty", "ytaxTot", "phi", "phiP", "savf"]
    gams_name = {
        "regy": "regY", "ytaxTot": "ytaxTot", "phiP": "phiP",
    }

    for phase, m_py, t_label in [("base", m_b, "base")] + (
        [("shock", m_s, "shock")] if args.phase != "base" else []
    ):
        print(f"\n{'='*100}")
        print(f"PHASE: {phase} (Python vs GAMS)")
        print(f"{'='*100}")
        print(f"{'var':<10s} {'region':<14s} {'python':>12s} {'gams':>12s} {'abs_diff':>11s} {'rel_%':>9s}")
        print("-" * 80)
        for v in macros_1d:
            gname = gams_name.get(v, v)
            try:
                gams = gams_levels(gname, t_filter=t_label)
            except subprocess.CalledProcessError:
                continue
            py = getattr(m_py, v, None)
            if py is None:
                continue
            for r in REGIONS:
                # GAMS keys: 1-D vars use (r, t); uh has h: (r, h, t)
                if (r, t_label) in gams:
                    g_val = gams[(r, t_label)]
                elif (r, "hhd", t_label) in gams:
                    g_val = gams[(r, "hhd", t_label)]
                else:
                    continue
                try:
                    if r in py:
                        p_val = float(value(py[r]))
                    elif (r, "hhd") in py:
                        p_val = float(value(py[r, "hhd"]))
                    elif r in [str(k) for k in py]:
                        for k in py:
                            if str(k) == r:
                                p_val = float(value(py[k]))
                                break
                    else:
                        continue
                except Exception:
                    continue
                d = p_val - g_val
                rel = (d / g_val * 100.0) if abs(g_val) > 1e-12 else float("inf")
                mark = " " if abs(rel) < 0.5 else ("⚠️" if abs(rel) < 5 else "❌")
                print(f"{v:<10s} {r:<14s} {p_val:>+12.6f} {g_val:>+12.6f} {d:>+11.3e} {rel:>+8.2f}%  {mark}")


if __name__ == "__main__":
    main()
