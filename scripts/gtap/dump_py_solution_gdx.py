"""Solve the Python multi-period model as NLP (in-process), then write ALL its
shock-period variable values to a GAMS .gms that only DECLARES the data and
execute_unloads it to a GDX — GAMS writes the GDX (no solve, fast). Read both GDXs
with equilibria's reader afterwards to diff."""
from equilibria.sources import dataset_dir
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
sys.path.insert(0, str(ROOT / "src"))
from equilibria._local_refs import path_capi_src  # noqa: E402
_PC = Path(str(path_capi_src()))
if _PC.exists(): sys.path.insert(0, str(_PC))
import contextlib

from pyomo.environ import value as V, Var

# the NLP route, without the Jacobian pre-scale: what this dump was written for
NLP_ENV = {"EQUILIBRIA_GTAP_SOLVE_NLP": "1", "EQUILIBRIA_GTAP_NLP_NO_JACSCALE": "1"}
_BUNDLE = {"gtap7_3x3": "out_3x3_nlp.gdx", "gtap7_5x5": "out_5x5_nlp.gdx", "gtap7_10x7": "out_10x7_nlp.gdx", "gtap7_15x10": "out_15x10_nlp.gdx"}


@contextlib.contextmanager
def _nlp_env():
    before = {k: os.environ.get(k) for k in NLP_ENV}
    os.environ.update(NLP_ENV)
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def solve(dataset: str):
    """Seed every period from the dataset's pure local bundle GDX and solve base
    (skipped) -> check -> shock as NLP on the core driver; returns (model, codes in
    the old driver's form)."""
    from _core_run import solve_on_core

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    gdx_ref = ROOT / f"output/{dataset}_pure_local_bundle/{_BUNDLE.get(dataset)}"
    d = dataset_dir(dataset)
    p = GTAPParameters()
    p.load_from_har(basedata_path=d/"basedata.har", sets_path=d/"sets.har", default_path=d/"default.prm", baserate_path=d/"baserate.har")
    rr = list(p.sets.r)[-1]
    gc = GTAPClosureConfig(name="base", closure_type="MCP", capital_mobility="sluggish", fix_endowments=False, fix_taxes=False, fix_technology=False, if_sub=True, numeraire="pnum")
    with _nlp_env():
        m, codes, _ = solve_on_core(p, gc, rr, mode="gtap", ref_gdx=gdx_ref, seed_gdx=gdx_ref)
    return m, codes


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "gtap7_3x3"
    m, _ = solve(dataset)

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


if __name__ == "__main__":
    main()
