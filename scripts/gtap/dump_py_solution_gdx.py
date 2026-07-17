"""Solve the Python multi-period model as NLP (in-process), then write ALL its
shock-period variable values to a GAMS .gms that only DECLARES the data and
execute_unloads it to a GDX — GAMS writes the GDX (no solve, fast). Read both GDXs
with equilibria's reader afterwards to diff."""
import sys, os
from pathlib import Path
ROOT = Path("/Users/marmol/.superset/worktrees/b14cb643-ee65-449d-b3f0-be8003b60783/scratched-stag")
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
sys.path.insert(0, str(ROOT / "src"))
_PC = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PC.exists(): sys.path.insert(0, str(_PC))
os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = "1"
os.environ["EQUILIBRIA_GTAP_NLP_NO_JACSCALE"] = "1"

from pyomo.environ import value as V, Var
from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel, PERIODS
from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

DATASET = sys.argv[1] if len(sys.argv) > 1 else "gtap7_3x3"
_bundle = {"gtap7_3x3": "out_3x3_nlp.gdx", "gtap7_5x5": "out_5x5_nlp.gdx", "gtap7_10x7": "out_10x7_nlp.gdx", "gtap7_15x10": "out_15x10_nlp.gdx"}.get(DATASET)
gdx_ref = ROOT / f"output/{DATASET}_pure_local_bundle/{_bundle}"
d = ROOT / "datasets" / DATASET
p = GTAPParameters()
p.load_from_har(basedata_path=d/"basedata.har", sets_path=d/"sets.har", default_path=d/"default.prm", baserate_path=d/"baserate.har")
rr = list(p.sets.r)[-1]
gc = GTAPClosureConfig(name="base", closure_type="MCP", capital_mobility="sluggish", fix_endowments=False, fix_taxes=False, fix_technology=False, if_sub=True, numeraire="pnum")
mp = GTAPMultiPeriodModel(p.sets, p, gc, residual_region=rr)
m = mp.build_sets(); mp.build_vars(m)
for per in PERIODS: mp.build_equations_intra(m, per)
mp.build_equations_fisher(m); m._residual_region = rr
mp.seed_all_periods(m, gdx_ref)
solve_multiperiod(m, p, gc, ref_gdx=gdx_ref, skip_base_solve=True, mute_welfare=True,
                  seed_from_prior=False, holdfix_cd=False, mode="gtap")

# collect ALL variable AND parameter values (all periods) into per-symbol dicts.
# Params are dumped with a p_ prefix so the diff can compare calibrated inputs
# (gf/aft/gf_share/and_param/...) against the GAMS GDX's own params.
from pyomo.environ import Param
syms = {}
for v in m.component_objects(Var, active=True):
    nm = v.name
    for idx in v:
        try:
            val = float(V(v[idx]))
        except Exception:
            continue
        key = idx if isinstance(idx, tuple) else (idx,)
        syms.setdefault("v_" + nm, {})[tuple(str(x) for x in key)] = val
for pcomp in m.component_objects(Param, active=True):
    nm = pcomp.name
    try:
        items = list(pcomp.items())
    except Exception:
        continue
    for idx, pv in items:
        try:
            val = float(V(pv)) if hasattr(pv, "__float__") or True else float(pv)
        except Exception:
            try:
                val = float(pv)
            except Exception:
                continue
        key = idx if isinstance(idx, tuple) else (idx,)
        syms.setdefault("p_" + nm, {})[tuple(str(x) for x in key)] = val

# write a GAMS .gms that declares each symbol as a parameter over its keys and unloads
out = ROOT / "output" / "pyshock_data.gms"
lines = ["$offlisting", "$offdigit", ""]
sym_names = []
for nm, cells in syms.items():
    if not cells:
        continue
    dim = len(next(iter(cells)))
    gnm = nm  # already prefixed v_/p_ above
    sym_names.append(gnm)
    idxset = ",".join(["*"] * dim) if dim else ""
    lines.append(f"parameter {gnm}({idxset});" if dim else f"scalar {gnm};")
    # data via assignments (robust vs domain decl)
    for keys, val in cells.items():
        if dim:
            k = ",".join(f"'{x}'" for x in keys)
            lines.append(f"{gnm}({k}) = {val:.12g};")
        else:
            lines.append(f"{gnm} = {val:.12g};")
lines.append("")
lines.append(f"execute_unload 'pyshock_sol.gdx' {', '.join(sym_names)};")
out.write_text("\n".join(lines))
print(f"wrote {out} with {len(sym_names)} symbols, {sum(len(c) for c in syms.values())} cells")
print("solved shock pft[EU_28,Land]:", syms.get("pft", {}).get(("EU_28", "Land", "shock")))
