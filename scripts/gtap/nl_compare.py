"""GTAP .nl comparison tool — Layer 6 diagnostic.

Writes Python GTAP models to .nl, fetches GAMS CompStat .nl via NEOS CONVERT,
and diffs structure + linear Jacobian coefficients.

Usage:
    uv run python3 scripts/gtap/nl_compare.py \\
        --dataset 3x3 \\
        --phase base check shock \\
        --neos-email dracomarmol@gmail.com \\
        --out-dir /tmp/gtap_nl
"""
from __future__ import annotations
from equilibria.sources import dataset_dir
import argparse
import base64
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from equilibria.templates.gtap import GTAPParameters
from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
from equilibria.templates.gtap.gtap_solver import GTAPSolver
from _parity_json import make_violation, run_tool  # noqa: E402 — shared JSON contract

# The standalone .gms builder generates results, so it lives with the model.
from equilibria.sources import GAMS_TOOLS_DIR  # noqa: E402

sys.path.insert(0, str(GAMS_TOOLS_DIR))
from nl_standalone import (  # noqa: E402, F401 — re-exported for callers of nl_compare
    COMP_GMS,
    DATA_DIR,
    KEEP_I,
    KEEP_R,
    _HIC,
    _MANUF,
    _PRM_HAR_NAME_MAP,
    _PRM_SET_PREFIX,
    _build_getdata_replacement,
    _build_getdata_replacement_agg,
    _build_getdata_replacement_shrunk,
    _build_standalone_gms,
    _compute_keep_a,
    _convert_locally,
    _detect_prm_prefix_convention,
    _fmt_set,
    _gdx_set_elements,
    _gdxdump,
    _gdxdump_params_only,
    _muv_baskets,
    _prm_as_assignments,
    _prm_har_as_assignments,
)

GDX9 = ROOT / "data/gtap7/9x10/basedata-9x10.gdx"





_SQUARENESS_FORCED_PAIRS = [
    ("eq_xfeq", "xf"),
    ("eq_regy", "regy"),
    ("eq_pwfact", "pwfact"),
    ("eq_facty", "facty"),
    ("eq_pfact", "pfact"),
    ("eq_ytax", "ytax"),
    ("eq_pfteq", "pft"),
]
_SQUARENESS_UNCONDITIONAL_PAIRS = [
    ("eq_xft", "pft"),
]


def _shrink_sets(p: GTAPParameters, keep_r: list[str], keep_i: list[str]) -> None:
    from diff_3x3_full import _shrink_sets as _do_shrink
    _do_shrink(p, keep_r, keep_i)


def _build_python_nl(
    period: str,
    p: GTAPParameters,
    closure_config,
    out_dir: Path,
    base_model=None,
) -> Path:
    """Build Pyomo model for `period`, apply closure+patches, write .nl.

    For the shock AND check periods, `base_model` (already-built base
    ConcreteModel) is passed as `t0_snapshot` so that counterfactual equations
    reference correct base-period levels.  The t_set is ("base", <period>) for
    those, ("base",) for base.  The check is the altertax compStat
    betaCal→check transition: counterfactual with the base frozen as snapshot,
    but with NO tariff shock (imptx stays at base; that is applied only in the
    shock branch of `build_python_nls`).
    """
    is_cf = period in ("shock", "check")
    t0_snap = base_model if is_cf else None
    rres = list(p.sets.r)[-1]

    eq = GTAPModelEquations(
        p.sets,
        p,
        closure=closure_config,
        is_counterfactual=is_cf,
        t0_snapshot=t0_snap,
        residual_region=rres,
    )
    m = eq.build_model()

    # Apply same closure + fixing as _run_path_capi_nonlinear_full
    solver_helper = GTAPSolver(m, closure=closure_config, solver_name="path", params=p)
    solver_helper.apply_closure(closure_config)
    solver_helper.apply_conditional_fixing()

    from _closure_patches import apply_squareness_patches
    apply_squareness_patches(m, p, label=f"nl-write-{period}")
    solver_helper.apply_aggressive_fixing_for_mcp()

    out_path = out_dir / f"python_{period}.nl"
    print(f"  Writing {out_path.name} ...")
    m.write(str(out_path), format="nl", io_options={"symbolic_solver_labels": True})
    kb = out_path.stat().st_size // 1024
    print(f"  Written: {out_path.name} ({kb} KB)")
    return out_path, m


def build_python_nls(
    phases: list[str],
    out_dir: Path,
    closure_config,
    gdx_path: Path | None = None,
    do_shrink: bool = True,
    har_dir: Path | None = None,
) -> dict[str, Path]:
    """Build Python .nl files for each period in phases.

    For the shock period, first builds (without solving) the base model to use
    as t0_snapshot, then builds the shock model with imptx*1.10.

    gdx_path: GTAP GDX to load (uppercase symbols). Defaults to the 9x10 dataset.
    har_dir: directory with HAR files (basedata/sets/default/baserate). When given,
        load_from_har is used instead of load_from_gdx (for GTAPAgg compstat datasets).
    do_shrink: when True, shrink the loaded 9x10 to the KEEP_R x KEEP_I subset.
    """
    print("\n=== Phase 1: Python → .nl ===")
    p = GTAPParameters()
    if har_dir is not None:
        p.load_from_har(
            basedata_path=har_dir / "basedata.har",
            sets_path=har_dir / "sets.har",
            default_path=har_dir / "default.prm",
            baserate_path=har_dir / "baserate.har",
        )
    else:
        p.load_from_gdx(gdx_path or GDX9)
        if do_shrink:
            _shrink_sets(p, KEEP_R, KEEP_I)
    print(f"  subset: r={list(p.sets.r)}  i={list(p.sets.i)}  a={list(p.sets.a)}")

    import copy

    paths: dict[str, Path] = {}
    base_model = None  # cached base ConcreteModel for shock t0_snapshot

    for period in ["base", "check", "shock", "altertax"]:
        if period not in phases:
            continue

        params_to_use = p
        period_closure = closure_config

        if period == "shock":
            p_shocked = copy.deepcopy(p)
            for key in list(p_shocked.taxes.imptx.keys()):
                old = float(p_shocked.taxes.imptx[key] or 0.0)
                p_shocked.taxes.imptx[key] = old * 1.10
            params_to_use = p_shocked
            # Ensure we have a base model for t0_snapshot
            if base_model is None:
                print("  Building base model (for shock t0_snapshot) ...")
                _, base_model = _build_python_nl("base", p, closure_config, out_dir)

        elif period == "check":
            # Altertax compStat betaCal→check transition: CD elasticities
            # (esubva=1, esubd=esubm=0.95, etrae=1, omegaf=1) + altertax closure,
            # but the BASE is frozen as t0_snapshot and NO tariff shock is applied
            # (imptx stays at base). Mirrors the GAMS check .nl
            # (_build_standalone_gms phase=="check"). The altertax branch hardcodes
            # if_sub=False; we match that here so the .nl gate is comparable —
            # closure_config.if_sub (the run's choice) is not threaded through.
            from equilibria.templates.gtap.altertax import apply_altertax_elasticities
            from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
            params_to_use = apply_altertax_elasticities(copy.deepcopy(p))
            period_closure = GTAPClosureConfig(
                name="altertax",
                closure_type="MCP",
                capital_mobility="mobile",
                fix_endowments=False,
                fix_taxes=True,
                fix_technology=True,
                if_sub=False,
            )
            # Ensure we have a base model for t0_snapshot (built once, reused).
            if base_model is None:
                print("  Building base model (for check t0_snapshot) ...")
                _, base_model = _build_python_nl("base", p, closure_config, out_dir)

        elif period == "altertax":
            # Malcolm (1998) CD-elasticity closure: override elasticities,
            # use altertax closure preset. No tariff shock applied to the .nl —
            # the .nl comparison is coefficient-only (structure check).
            from equilibria.templates.gtap.altertax import apply_altertax_elasticities
            from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
            params_to_use = apply_altertax_elasticities(copy.deepcopy(p))
            period_closure = GTAPClosureConfig(
                name="altertax",
                closure_type="MCP",
                capital_mobility="mobile",
                fix_endowments=False,
                fix_taxes=True,
                fix_technology=True,
                if_sub=False,
            )

        t0 = time.perf_counter()
        nl_path, built_model = _build_python_nl(
            period, params_to_use, period_closure, out_dir,
            base_model=base_model if period in ("shock", "check") else None,
        )
        if period == "base":
            base_model = built_model  # cache for shock
        print(f"  {period}: {time.perf_counter()-t0:.1f}s")
        paths[period] = nl_path

    return paths


# ---------------------------------------------------------------------------
# Phase 2: GAMS → .nl via NEOS CONVERT
# ---------------------------------------------------------------------------

def _encode_gdx(gdx_path: Path) -> str:
    return base64.b64encode(gdx_path.read_bytes()).decode()


def _extract_nl_from_neos_output(raw: str) -> str | None:
    """Extract .nl file content from NEOS output log."""
    # NEOS CONVERT output: .nl content starts with "g3 " header
    idx = raw.find("g3 ")
    if idx != -1:
        return raw[idx:].strip()
    return None


_SETS_9X10_GAMS = """
sets
   acts           "Activities"      / a_agricultur, a_Extraction, a_ProcFood, a_TextWapp,
                                      a_LightMnfc, a_HeavyMnfc, a_Util_Cons, a_TransComm,
                                      a_OthService /
   comm           "Commodities"     / c_Crops, c_MeatLstk, c_Extraction, c_ProcFood,
                                      c_TextWapp, c_LightMnfc, c_HeavyMnfc, c_Util_Cons,
                                      c_TransComm, c_OthService /
   marg(comm)     "Margin comm"     / c_TransComm /
   reg            "Regions"         / Oceania, EastAsia, SEAsia, SouthAsia, NAmerica,
                                      LatinAmer, EU_28, MENA, SSA, RestofWorld /
   endw           "Endowments"      / Land, UnSkLab, SkLab, Capital, NatRes /
   endwf(endw)    "Fixed factors"   / NatRes /
   endwm(endw)    "Mobile factors"  / UnSkLab, SkLab, Capital /
   endws(endw)    "Sluggish factors"/ Land /
;
"""



























def _build_solve_gms_for_9x10() -> str:
    """Like _build_solve_gms but for the full 9x10 dataset.

    Uses _build_getdata_replacement_agg(9x10Dat.gdx) for the main data
    (same inlining path as NUS333 bundles that compile cleanly on NEOS 52.5.0),
    then appends the elasticity parameters from 9x10Prm.gdx via $onMulti/$offMulti
    so that esubq, esubt, esubc, etc. are available.
    """
    import re
    scripts_dir = COMP_GMS.parent
    dat_gdx = DATA_DIR / "9x10Dat.gdx"
    prm_gdx = DATA_DIR / "9x10Prm.gdx"

    # Build the standard agg getData replacement from the flow data GDX.
    getdata_base = _build_getdata_replacement_agg(dat_gdx)

    # Embed elasticity parameters from 9x10Prm.gdx BEFORE the derived-parameter
    # assignments (omegas = -etraq, sigmas = 1/esubq, ...) that use them.
    # Insert just before "parameters sigmap(r,a)" in the getdata_base text.
    # Use plain assignment style — $onMulti + data triggers $767 in NEOS GAMS 52.5.0.
    prm_assignments = _prm_as_assignments(prm_gdx, renames={"RORFLEX": "rorFlex0"})
    prm_block = (
        "\n* --- Elasticity parameters from 9x10Prm.gdx ---\n"
        + prm_assignments
        + "* END elasticity parameters\n\n"
    )
    anchor = "parameters sigmap(r,a)"
    if anchor in getdata_base:
        getdata_replacement = getdata_base.replace(anchor, prm_block + anchor, 1)
    else:
        # Fallback: append after the base
        getdata_replacement = getdata_base.rstrip() + "\n" + prm_block

    def _inline(text: str, depth: int = 0) -> str:
        if depth > 5:
            return text
        def _rep(m):
            fname = m.group(1).strip('"\'')
            if fname == "getData.gms":
                return getdata_replacement
            fpath = scripts_dir / fname
            if fpath.exists():
                return _inline(fpath.read_text(), depth + 1)
            return m.group(0)
        def _rep_bat(m):
            fname = m.group(1).strip('"\'')
            args = (m.group(2).strip().split() if m.group(2) else [])
            fpath = scripts_dir / fname
            if not fpath.exists():
                return m.group(0)
            inner = fpath.read_text()
            # Handle $setargs: parse named args, substitute %name% → value,
            # then strip the $setargs line. This correctly handles mvar.gms
            # ($setargs theMacro theYear) so %theMacro%/%theYear% are resolved.
            setargs_m = re.search(r'^\$setargs\s+(.*)', inner, re.MULTILINE | re.IGNORECASE)
            if setargs_m:
                named = setargs_m.group(1).split()
                for idx, name in enumerate(named):
                    val = args[idx] if idx < len(args) else ""
                    inner = inner.replace(f"%{name}%", val)
                inner = re.sub(r'^\$setargs\s+.*\n?', '', inner, count=1, flags=re.MULTILINE | re.IGNORECASE)
            # Also substitute positional %N% args.
            for idx, arg in enumerate(args, start=1):
                inner = inner.replace(f"%{idx}", arg)
            return _inline(inner, depth + 1)
        text = re.sub(r'^[ \t]*\$\$?include\s+("?[^"\s]+"?)', _rep, text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^[ \t]*\$\$?batinclude\s+("?[^"\s]+"?)((?:[ \t]+\S+)*)', _rep_bat, text, flags=re.IGNORECASE | re.MULTILINE)
        return text

    inlined = _inline(COMP_GMS.read_text())

    # 9x10 full dataset: hardcode correct rres/rmuv/imuv.
    rres = "NAmerica"
    rmuv = ("Oceania", "NAmerica", "EU_28")
    imuv = ("c_ProcFood", "c_TextWapp", "c_LightMnfc", "c_HeavyMnfc")
    for sname, dom, elems in (("rres", "r", (rres,)), ("rmuv", "r", rmuv), ("imuv", "i", imuv)):
        inlined = re.sub(
            rf'set\s+{sname}\({dom}\)\s*"[^"]*"\s*/\s*[^/]*?/\s*;',
            f'set {sname}({dom}) "{sname}" /\n   {", ".join(elems)}\n/ ;',
            inlined, flags=re.IGNORECASE,
        )

    # Replace placeholder shock (pnum.fx=1.5) with 10% import-tariff shock.
    inlined, n = re.subn(
        r"pnum\.fx\(tsim\)\s*=\s*1\.5\s*;",
        "imptx.fx(r,i,rp,tsim) = imptx.l(r,i,rp,tsim) * 1.10 ;",
        inlined, count=1, flags=re.IGNORECASE,
    )
    if n != 1:
        raise RuntimeError("could not find pnum.fx=1.5 shock anchor in comp.gms")

    # Redirect the solution unload to out.gdx.
    inlined = inlined.replace('"%outDir%/%simName%.gdx"', '"out.gdx"')

    header = (
        "$setGlobal simType   CompStat\n"
        "$setGlobal simName   COMP\n"
        "$setGlobal baseName  9x10\n"
        "$setGlobal inDir     .\n$setGlobal outDir    .\n"
        "$setGlobal utility   cde\n$setGlobal savfFlag  capFix\n"
        "$setGlobal ifCal     0\n$setGlobal ifSUB     0\n"
        "$setGlobal ifCSV     0\n$setGlobal ifMCP     1\n"
        "$setGlobal ifCSVAppend 0\n"
        "option mcp = path ;\n"
    )
    return header + inlined


def _build_solve_gms(agg_gdx: Path, har_dir: Path | None = None) -> str:
    """Build a self-contained comp.gms that SOLVES via MCP/PATH and writes out.gdx.

    Unlike _build_standalone_gms (CONVERT, truncated), this keeps the full
    comp.gms simulation loop + postsim (ifMCP=1), inlines the aggregated subset
    data, restricts rres/rmuv/imuv to the subset, replaces the placeholder
    pnum=1.5 shock with the 10% import-tariff shock, and unloads to out.gdx.
    har_dir: when given, elasticities from default.prm are appended to getData.
    """
    import re
    scripts_dir = COMP_GMS.parent
    getdata_replacement = _build_getdata_replacement_agg(agg_gdx, har_dir=har_dir)

    def _inline(text: str, depth: int = 0) -> str:
        if depth > 5:
            return text
        def _rep(m):
            fname = m.group(1).strip('"\'')
            if fname == "getData.gms":
                return getdata_replacement
            fpath = scripts_dir / fname
            if fpath.exists():
                return _inline(fpath.read_text(), depth + 1)
            return m.group(0)
        def _rep_bat(m):
            fname = m.group(1).strip('"\'')
            args = (m.group(2).strip().split() if m.group(2) else [])
            fpath = scripts_dir / fname
            if not fpath.exists():
                return m.group(0)
            inner = fpath.read_text()
            # Handle $setargs: parse named args, substitute %name% → value,
            # then strip the $setargs line so GAMS doesn't see it outside batinclude.
            setargs_m = re.search(r'^\$setargs\s+(.*)', inner, re.MULTILINE | re.IGNORECASE)
            if setargs_m:
                named = setargs_m.group(1).split()
                for idx, name in enumerate(named):
                    val = args[idx] if idx < len(args) else ""
                    inner = inner.replace(f"%{name}%", val)
                inner = re.sub(r'^\$setargs\s+.*\n?', '', inner, count=1, flags=re.MULTILINE | re.IGNORECASE)
            # Also substitute positional %N% args.
            for idx, arg in enumerate(args, start=1):
                inner = inner.replace(f"%{idx}", arg)
            return _inline(inner, depth + 1)
        text = re.sub(r'^[ \t]*\$\$?include\s+("?[^"\s]+"?)', _rep, text, flags=re.IGNORECASE | re.MULTILINE)
        # NOTE: batinclude args must stay on the SAME line — use [ \t] not \s,
        # else the arg group swallows the rest of the file (loop + execute_unload).
        text = re.sub(r'^[ \t]*\$\$?batinclude\s+("?[^"\s]+"?)((?:[ \t]+\S+)*)', _rep_bat, text, flags=re.IGNORECASE | re.MULTILINE)
        return text

    inlined = _inline(COMP_GMS.read_text())

    # Restrict comp.gms hardcoded rres/rmuv/imuv to the aggregate's elements.
    reg  = _gdx_set_elements(agg_gdx, "REG")
    comm = _gdx_set_elements(agg_gdx, "COMM")
    rres, rmuv, imuv = _muv_baskets(reg, comm)
    for sname, dom, elems in (("rres", "r", (rres,)), ("rmuv", "r", rmuv), ("imuv", "i", imuv)):
        inlined = re.sub(
            rf'set\s+{sname}\({dom}\)\s*"[^"]*"\s*/\s*[^/]*?/\s*;',
            f'set {sname}({dom}) "{sname}" /\n   {", ".join(elems)}\n/ ;',
            inlined, flags=re.IGNORECASE,
        )

    # Replace placeholder shock (pnum.fx=1.5) with 10% import-tariff shock.
    inlined, n = re.subn(
        r"pnum\.fx\(tsim\)\s*=\s*1\.5\s*;",
        "imptx.fx(r,i,rp,tsim) = imptx.l(r,i,rp,tsim) * 1.10 ;",
        inlined, count=1, flags=re.IGNORECASE,
    )
    if n != 1:
        raise RuntimeError("could not find pnum.fx=1.5 shock anchor in comp.gms")

    # Redirect the solution unload to out.gdx.
    inlined = inlined.replace('"%outDir%/%simName%.gdx"', '"out.gdx"')

    # Header: ifMCP=1 (real PATH solve), ifSUB=0 (Python parity), no calibration.
    header = (
        "$setGlobal simType   CompStat\n"
        "$setGlobal simName   COMP\n"
        "$setGlobal baseName  agg\n"
        "$setGlobal inDir     .\n$setGlobal outDir    .\n"
        "$setGlobal utility   cde\n$setGlobal savfFlag  capFix\n"
        "$setGlobal ifCal     0\n$setGlobal ifSUB     0\n"
        "$setGlobal ifCSV     0\n$setGlobal ifMCP     1\n"
        "option mcp = path ;\n"
    )
    return header + inlined


def _build_solve_gms_9x10() -> str:
    """Like _build_solve_gms but for the full 9x10 dataset (no aggregation GDX).

    Uses _build_getdata_replacement() (hardcoded 9x10 sets + embedded data from
    9x10Dat.gdx / 9x10Prm.gdx) instead of _build_getdata_replacement_agg.
    Applies the 10% import-tariff shock (imptx * 1.10) and unloads to out.gdx.
    """
    import re
    scripts_dir = COMP_GMS.parent
    getdata_replacement = _build_getdata_replacement()

    def _inline(text: str, depth: int = 0) -> str:
        if depth > 5:
            return text
        def _rep(m):
            fname = m.group(1).strip('"\'')
            if fname == "getData.gms":
                return getdata_replacement
            fpath = scripts_dir / fname
            if fpath.exists():
                return _inline(fpath.read_text(), depth + 1)
            return m.group(0)
        def _rep_bat(m):
            fname = m.group(1).strip('"\'')
            args = (m.group(2).strip().split() if m.group(2) else [])
            fpath = scripts_dir / fname
            if fpath.exists():
                inner = fpath.read_text()
                for idx, arg in enumerate(args, start=1):
                    inner = inner.replace(f"%{idx}", arg)
                return _inline(inner, depth + 1)
            return m.group(0)
        text = re.sub(r'^[ \t]*\$\$?include\s+("?[^"\s]+"?)', _rep, text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^[ \t]*\$\$?batinclude\s+("?[^"\s]+"?)((?:[ \t]+\S+)*)', _rep_bat, text, flags=re.IGNORECASE | re.MULTILINE)
        return text

    inlined = _inline(COMP_GMS.read_text())

    # 9x10 full: rres=NAmerica, rmuv=HIC trio, imuv=manufactures — match Python closure.
    rres = "NAmerica"
    rmuv = ("Oceania", "NAmerica", "EU_28")
    imuv = ("c_ProcFood", "c_TextWapp", "c_LightMnfc", "c_HeavyMnfc")
    for sname, dom, elems in (("rres", "r", (rres,)), ("rmuv", "r", rmuv), ("imuv", "i", imuv)):
        inlined = re.sub(
            rf'set\s+{sname}\({dom}\)\s*"[^"]*"\s*/\s*[^/]*?/\s*;',
            f'set {sname}({dom}) "{sname}" /\n   {", ".join(elems)}\n/ ;',
            inlined, flags=re.IGNORECASE,
        )

    # Replace placeholder shock (pnum.fx=1.5) with 10% import-tariff shock.
    inlined, n = re.subn(
        r"pnum\.fx\(tsim\)\s*=\s*1\.5\s*;",
        "imptx.fx(r,i,rp,tsim) = imptx.l(r,i,rp,tsim) * 1.10 ;",
        inlined, count=1, flags=re.IGNORECASE,
    )
    if n != 1:
        raise RuntimeError("could not find pnum.fx=1.5 shock anchor in comp.gms")

    # Strip dynamic-simulation-only blocks that use years(tsim) in if() conditions.
    # GAMS $149 fires when a loop variable is used as a constant in if().
    # All three blocks are ifDyn/multi-year-only; CompStat (base+shock) doesn't need them.
    for pattern in [
        # ifDyn block: contains $setargs/$macro/%theMacro% compile-time directives
        r'if\(years\(tsim\)\s+gt\s+FirstYear\s+and\s+ifDyn\s*,.*?^\s*\)\s*;',
        # lag-fixing block (ne firstYear)
        r'if\(years\(tsim\)\s+ne\s+firstYear\s*,.*?^\s*\)\s*;',
        # dynamic solve guard (gt firstYear, without ifDyn)
        r'if\(years\(tsim\)\s+gt\s+firstYear\s*,.*?^\s*\$\$endif.*?^\s*\)\s*;',
    ]:
        inlined, cnt = re.subn(pattern, '* [years(tsim) block removed — CompStat only]',
                               inlined, count=1, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE)
        if cnt:
            continue

    # Redirect the solution unload to out.gdx.
    inlined = inlined.replace('"%outDir%/%simName%.gdx"', '"out.gdx"')

    # Suppress $141 (unassigned symbol) — walras.l not assigned before display when
    # solve is skipped. Also suppress $189 (domain violation in data) for robustness.
    inlined = "$onImplicitAssign\n" + inlined

    header = (
        "$setGlobal simType   CompStat\n"
        "$setGlobal simName   COMP\n"
        "$setGlobal baseName  9x10\n"
        "$setGlobal inDir     .\n$setGlobal outDir    .\n"
        "$setGlobal utility   cde\n$setGlobal savfFlag  capFix\n"
        "$setGlobal ifCal     0\n$setGlobal ifSUB     0\n"
        "$setGlobal ifCSV     0\n$setGlobal ifMCP     1\n"
        "$setGlobal ifCSVAppend 0\n"
        "option mcp = path ;\n"
    )
    return header + inlined


def submit_and_fetch_gdx_9x10(out_dir: Path, neos_email: str) -> Path:
    """Submit a GAMS MCP/PATH solve of the full 9x10 dataset to NEOS, fetch out.gdx.

    Uses _build_solve_gms with the merged 9x10Combined.gdx (Dat+Prm), which
    uses the same inlining path as the NUS333 agg bundles that compile cleanly
    on NEOS 52.5.0. Returns path to downloaded out.gdx.
    """
    import xmlrpc.client, zipfile, io
    out_dir.mkdir(parents=True, exist_ok=True)
    solve_gms = _build_solve_gms_for_9x10()
    (out_dir / "gtap_solve_9x10.gms").write_text(solve_gms)
    print(f"  solve .gms: {len(solve_gms.splitlines())} lines (9x10 full)")

    xml_job = f"""<document>
<category>cp</category>
<solver>PATH</solver>
<inputMethod>GAMS</inputMethod>
<email>{neos_email}</email>
<model><![CDATA[
{solve_gms}
]]></model>
<wantgdx><![CDATA[yes]]></wantgdx>
<wantlog><![CDATA[yes]]></wantlog>
<comments>GTAP 9x10 full PATH solve — 10% tariff shock reference</comments>
</document>"""

    neos = xmlrpc.client.ServerProxy("https://neos-server.org:3333", allow_none=True)
    print("  Submitting 9x10 PATH solve to NEOS ...")
    (job_number, password) = neos.submitJob(xml_job)
    print(f"  Job #{job_number} submitted. Polling ...")
    status = "Waiting"
    for attempt in range(90):
        time.sleep(10)
        try:
            status = neos.getJobStatus(job_number, password)
        except Exception:
            pass
        print(f"  [{attempt*10}s] status={status}")
        if status in ("Done", "Error"):
            break

    listing = neos.getFinalResults(job_number, password)
    raw = listing.data.decode("utf-8", errors="replace") if hasattr(listing, "data") else str(listing)
    (out_dir / "neos_solve_9x10.lst").write_text(raw)
    if status == "Error":
        raise RuntimeError(f"NEOS solve #{job_number} failed. See {out_dir/'neos_solve_9x10.lst'}")

    zip_result = neos.getOutputFile(job_number, password, "solver-output.zip")
    zip_data = zip_result.data if hasattr(zip_result, "data") else bytes(zip_result)
    try:
        z = zipfile.ZipFile(io.BytesIO(zip_data))
    except zipfile.BadZipFile:
        raise RuntimeError(f"NEOS did not return a zip: {zip_data[:200]!r}")
    print(f"  solver-output.zip files: {z.namelist()}")
    gdx_name = next((n for n in z.namelist() if n.endswith(".gdx")), None)
    if gdx_name is None:
        raise RuntimeError(
            f"no .gdx in solver-output.zip. Files: {z.namelist()} — "
            f"check {out_dir/'neos_solve_9x10.lst'} for errors.")
    out_path = out_dir / "out.gdx"
    out_path.write_bytes(z.read(gdx_name))
    print(f"  Written: {out_path} ({out_path.stat().st_size // 1024} KB)")
    return out_path






def _submit_and_fetch_nl(
    standalone_gms: str,
    out_dir: Path,
    neos_email: str,
    phase: str,
) -> Path:
    """Submit one NEOS CONVERT job and return path to the downloaded .nl."""
    import xmlrpc.client, zipfile, io

    xml_job = f"""<document>
<category>application</category>
<solver>CONVERT</solver>
<inputMethod>GAMS</inputMethod>
<email>{neos_email}</email>
<model><![CDATA[
{standalone_gms}
]]></model>
<formats><![CDATA[AmplNL ampl.nl]]></formats>
<wantlog><![CDATA[yes]]></wantlog>
<comments>GTAP nl_compare Layer 6 diagnostic — {phase} period NLP CONVERT</comments>
</document>"""

    neos = xmlrpc.client.ServerProxy("https://neos-server.org:3333", allow_none=True)
    print(f"  Submitting {phase} to NEOS ...")
    try:
        (job_number, password) = neos.submitJob(xml_job)
    except Exception as e:
        raise RuntimeError(f"NEOS submission failed: {e}") from e

    print(f"  Job #{job_number} submitted. Polling ...")
    status = "Waiting"
    for attempt in range(72):  # max 12 minutes
        time.sleep(10)
        try:
            status = neos.getJobStatus(job_number, password)
        except Exception:
            pass
        print(f"  [{attempt*10}s] status={status}")
        if status in ("Done", "Error"):
            break

    listing = neos.getFinalResults(job_number, password)
    raw_lst = listing.data.decode("utf-8", errors="replace") if hasattr(listing, "data") else str(listing)
    lst_path = out_dir / f"neos_convert_{phase}.lst"
    lst_path.write_text(raw_lst)

    if status == "Error":
        raise RuntimeError(f"NEOS job #{job_number} ({phase}) failed. See {lst_path}")

    zip_result = neos.getOutputFile(job_number, password, "solver-output.zip")
    zip_data = zip_result.data if hasattr(zip_result, "data") else bytes(zip_result)

    try:
        z = zipfile.ZipFile(io.BytesIO(zip_data))
    except zipfile.BadZipFile:
        msg = zip_data.decode("utf-8", errors="replace")
        raise RuntimeError(f"NEOS ({phase}) did not return valid zip. Response: {msg[:200]}")

    zip_path = out_dir / f"neos_convert_{phase}_output.zip"
    zip_path.write_bytes(zip_data)
    print(f"  solver-output.zip: {zip_path.stat().st_size // 1024} KB  files: {z.namelist()}")

    if "ampl.nl" not in z.namelist():
        raise RuntimeError(
            f"ampl.nl not in solver-output.zip ({phase}). Available: {z.namelist()}\n"
            f"Check {lst_path} for GAMS errors."
        )

    nl_bytes = z.read("ampl.nl")
    col_bytes = z.read("ampl.col") if "ampl.col" in z.namelist() else b""
    row_bytes = z.read("ampl.row") if "ampl.row" in z.namelist() else b""

    out_path = out_dir / f"gams_{phase}.nl"
    out_path.write_bytes(nl_bytes)
    if col_bytes:
        (out_dir / f"gams_{phase}.col").write_bytes(col_bytes)
    if row_bytes:
        (out_dir / f"gams_{phase}.row").write_bytes(row_bytes)

    print(f"  Written: {out_path.name} ({out_path.stat().st_size // 1024} KB)")
    return out_path


def submit_and_fetch_gdx(agg_gdx: Path, out_dir: Path, neos_email: str,
                         har_dir: Path | None = None) -> Path:
    """Submit a GAMS MCP/PATH SOLVE of a GTAPAgg dataset to NEOS, fetch out.gdx.

    Builds the solve .gms (_build_solve_gms: ifMCP=1, PATH, 10% tariff shock,
    execute_unload 'out.gdx') and submits via category=complementarity solver=PATH.
    har_dir: when given, elasticities from default.prm are appended to getData.
    """
    import xmlrpc.client, zipfile, io
    out_dir.mkdir(parents=True, exist_ok=True)
    solve_gms = _build_solve_gms(agg_gdx, har_dir=har_dir)
    (out_dir / "gtap_solve.gms").write_text(solve_gms)
    print(f"  solve .gms: {len(solve_gms.splitlines())} lines from {agg_gdx.name}")

    # Category 'cp' (complementarity) + PATH, per NEOS docs
    # (neos-server.org/neos/solvers/cp:PATH/GAMS-help.html). wantgdx returns a
    # compressed GDX with ALL model symbols — the solved levels the residual needs.
    xml_job = f"""<document>
<category>cp</category>
<solver>PATH</solver>
<inputMethod>GAMS</inputMethod>
<email>{neos_email}</email>
<model><![CDATA[
{solve_gms}
]]></model>
<wantgdx><![CDATA[yes]]></wantgdx>
<wantlog><![CDATA[yes]]></wantlog>
<comments>GTAP nl_compare — GAMS PATH solve, wantgdx for residual reference</comments>
</document>"""

    neos = xmlrpc.client.ServerProxy("https://neos-server.org:3333", allow_none=True)
    print("  Submitting PATH solve to NEOS ...")
    (job_number, password) = neos.submitJob(xml_job)
    print(f"  Job #{job_number} submitted. Polling ...")
    status = "Waiting"
    for attempt in range(90):  # max 15 min
        time.sleep(10)
        try:
            status = neos.getJobStatus(job_number, password)
        except Exception:
            pass
        print(f"  [{attempt*10}s] status={status}")
        if status in ("Done", "Error"):
            break

    listing = neos.getFinalResults(job_number, password)
    raw = listing.data.decode("utf-8", errors="replace") if hasattr(listing, "data") else str(listing)
    (out_dir / "neos_solve.lst").write_text(raw)
    if status == "Error":
        raise RuntimeError(f"NEOS solve #{job_number} failed. See {out_dir/'neos_solve.lst'}")

    zip_result = neos.getOutputFile(job_number, password, "solver-output.zip")
    zip_data = zip_result.data if hasattr(zip_result, "data") else bytes(zip_result)
    try:
        z = zipfile.ZipFile(io.BytesIO(zip_data))
    except zipfile.BadZipFile:
        raise RuntimeError(f"NEOS solve did not return a zip: {zip_data[:200]!r}")
    print(f"  solver-output.zip files: {z.namelist()}")
    gdx_name = next((n for n in z.namelist() if n.endswith(".gdx")), None)
    if gdx_name is None:
        raise RuntimeError(
            f"no .gdx in solver-output.zip. Available: {z.namelist()} — "
            f"NEOS may not return unloaded GDX files for this solver.")
    out_path = out_dir / "out.gdx"
    out_path.write_bytes(z.read(gdx_name))
    print(f"  Written: {out_path} ({out_path.stat().st_size // 1024} KB)")
    return out_path


def fetch_gams_nl(
    out_dir: Path,
    neos_email: str,
    phases: list[str],
    shrink: bool = False,
    agg_gdx: Path | None = None,
    har_dir: Path | None = None,
    ifsub: int = 0,
    local: bool = False,
) -> dict[str, Path]:
    """Submit GTAP model to NEOS CONVERT for each requested phase.

    local: run the local GAMS CONVERT instead of NEOS (see _convert_locally).

    Returns dict phase → Path of downloaded .nl file.
    For "shock" phase, submits a separate job with tm*1.10 tariff shock applied.

    shrink: when True, the standalone .gms declares sets shrunk to the
        KEEP_R x KEEP_I subset (full 9x10 data), matching the Python
        `_shrink_sets` run — true apples-to-apples 3x3.
    """
    print("\n=== Phase 2: GAMS → .nl via NEOS CONVERT:GAMS (application category) ===")

    result: dict[str, Path] = {}
    for phase in phases:
        print(f"\n  Building standalone .gms for phase={phase} ...")
        standalone_gms = _build_standalone_gms(phase=phase, shrink=shrink, agg_gdx=agg_gdx,
                                               har_dir=har_dir, ifsub=ifsub)
        if local:
            result[phase] = _convert_locally(standalone_gms, out_dir, phase)
            continue
        standalone_path = out_dir / f"gtap_standalone_{phase}.gms"
        standalone_path.write_text(standalone_gms)
        print(f"  Standalone .gms: {len(standalone_gms.splitlines())} lines ({len(standalone_gms)//1024} KB)")
        result[phase] = _submit_and_fetch_nl(standalone_gms, out_dir, neos_email, phase)

    return result


# ---------------------------------------------------------------------------
# Phase 3: Diff
# ---------------------------------------------------------------------------

_T_PERIODS = ("base", "check", "shock")


def _raw_period(name: str) -> str | None:
    """Period index of a raw GAMS or Python name, or None if it has none.

    pmeq(EU_28,c_Food,ROW,shock) -> "shock"; pm[EU_28,Food,ROW,base] -> "base".
    """
    if not name or name[-1] not in ")]":
        return None
    last = name[:-1].rsplit(",", 1)[-1].lstrip("([")
    return last if last in _T_PERIODS else None


def _normalize_name(name: str) -> str:
    """Strip time-period suffix and unify bracket style.

    GAMS: axp(Oceania,a_agricultur,base)  →  axp[Oceania,a_agricultur]
    Python: axp[Oceania,a_agricultur,base] →  axp[Oceania,a_agricultur]
    Also strips 'eq' suffix from GAMS equation names:
      ndeq(Oceania,...) → nd[Oceania,...]  (GAMS uses {name}eq convention)
    """
    # Unify to bracket notation
    if "(" in name and name.endswith(")"):
        name = name[:-1].replace("(", "[", 1) + "]"
        name = name.replace(",", ",").replace("(", ",")

    # Strip time suffix: name[...,base] → name[...]
    for period in _T_PERIODS:
        suffix = f",{period}]"
        if name.endswith(suffix):
            name = name[: -len(suffix)] + "]"
            break
    # Strip bare period: name[base] → name
    for period in _T_PERIODS:
        if name == f"[{period}]" or name.endswith(f"[{period}]"):
            bare = f"[{period}]"
            if name == bare:
                return ""
            name = name[: -len(bare)]
            break

    name = _strip_index_prefixes(name)
    name = _strip_eq_suffix(name)
    return name


def _strip_eq_suffix(name: str) -> str:
    """``eq_xfeq[...]`` -> ``eq_xf[...]``.

    GAMS nombra sus ecuaciones ``<var>eq`` (xfeq, pfeq) y _gams_to_python_name ya
    las convierte a ``eq_<var>``.  Pero el lado Python nombra ALGUNAS de las suyas
    ``eq_<var>eq`` —conservando el sufijo GAMS DENTRO del prefijo propio— asi que
    ``eq_xfeq`` y ``eq_xf`` son la MISMA ecuacion y no casaban.

    MEDIDO (gtap7_3x3/base): afectaba a eq_xfeq/eq_pfeq/eq_pfaeq/eq_pfyeq (36
    filas cada una), eq_ytaxshreq (30), eq_xweq/eq_pmeq/eq_peeq (27)...  Solo se
    toca el sufijo del NOMBRE; los indices ya los normalizo _strip_index_prefixes.
    """
    if not name.startswith("eq_"):
        return name
    bracket = name.find("[")
    base = name[:bracket] if bracket != -1 else name
    rest = name[bracket:] if bracket != -1 else ""
    if base.endswith("eq") and len(base) > len("eq_") + 2:
        return base[:-2] + rest
    return name


# Los indices de commodity/actividad llevan prefijo de dominio en unos datasets
# (c_Food / a_Food) y no en otros (Food) — es DATASET-SPECIFIC, no universal
# (ver _PRM_SET_PREFIX).  El .nl de GAMS conserva la convencion de SU dataset,
# mientras que el modelo Python indexa siempre con el nombre desnudo, asi que
# emparejar por nombre crudo falla en TODA fila indexada por commodity o
# actividad — justo produccion, comercio y demanda Armington.
#
# MEDIDO (gtap7_3x3/base, antes de este despojado): de 797 filas GAMS solo 111
# casaban, y las 32 familias comparadas eran TODAS de cierre macro (eq_gdpmp,
# eq_yc, eq_rorg...).  Las familias con mas filas sueltas eran precisamente las
# indexadas: eq_pa (57), eq_xd (57), eq_xm (54), eq_xf (36).  Un sabotaje
# deliberado en eq_pp_rai pasaba INADVERTIDO porque esa familia no se comparaba.
#
# Normalizar a nombre desnudo en AMBOS lados hace la comparacion independiente
# de la convencion del dataset.  Solo se tocan los segmentos DENTRO de los
# corchetes: el nombre de la ecuacion nunca se toca (no hay familias c_*/a_*).
_INDEX_PREFIXES = ("c_", "a_", "f_", "r_")


def _strip_index_prefixes(name: str) -> str:
    """Quita los prefijos de dominio c_/a_/f_/r_ de los INDICES de ``name``."""
    start = name.find("[")
    if start == -1 or not name.endswith("]"):
        return name
    head, idx = name[:start], name[start + 1 : -1]
    if not idx:
        return name
    parts = []
    for part in idx.split(","):
        for pref in _INDEX_PREFIXES:
            if part.startswith(pref) and len(part) > len(pref):
                part = part[len(pref) :]
                break
        parts.append(part)
    return f"{head}[{','.join(parts)}]"


def _gams_to_python_name(name: str) -> str:
    """Convert GAMS equation/variable name convention to Python convention.

    GAMS equations: {name}eq(...)  →  eq_{name}[...]
    GAMS variables: same base name, just different bracket style
    """
    name = _normalize_name(name)
    # Check for GAMS equation suffix 'eq': ndeq[...] → eq_nd[...]
    bracket = name.find("[")
    base = name[:bracket] if bracket != -1 else name
    rest = name[bracket:] if bracket != -1 else ""
    if base.endswith("eq") and len(base) > 2:
        return "eq_" + base[:-2] + rest
    return name


def _family(name: str) -> str:
    for sep in ("[", "("):
        idx = name.find(sep)
        if idx != -1:
            return name[:idx]
    return name


def _split_name(name: str) -> tuple[str, list[str]]:
    """Split 'eq_x[USA,Svces,Svces]' -> ('eq_x', ['USA','Svces','Svces']).

    Returns (base, index_list). A bare name with no brackets -> (name, []).
    """
    for op, cl in (("[", "]"), ("(", ")")):
        i = name.find(op)
        if i != -1:
            base = name[:i]
            inner = name[i + 1:].rstrip(cl + " ")
            if inner.endswith(cl):
                inner = inner[:-1]
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            return base, parts
    return name, []


def diff_nl_models(
    py_model: "NLModel",
    gams_model: "NLModel",
    tol_rel: float = 1e-4,
    py_period: str = "base",
    cell_cap: int = 50,
) -> dict:
    """Compare two NLModel objects. Returns structured diff result.

    Both Python and GAMS names are normalized (time suffix stripped) so that
    e.g. pf[NAmerica,NatRes,a_Extraction,base] → pf[NAmerica,NatRes,a_Extraction]
    in both models before comparison.

    py_period: which time-period to extract from Python's multi-period .nl.
    For single-period models ("base" only), all entries match.
    For two-period models ("base","shock"), set py_period="shock" to compare
    only the shock-period variables/constraints against the GAMS shock .nl.
    """
    # Python: may have multiple periods; filter to py_period entries.
    # For a single-period model t_set=("base",), py_period="base" → all entries.
    # For two-period t_set=("base","shock"), py_period="shock" → only shock entries.
    def _matches_period(raw_name: str, period: str) -> bool:
        """True if the raw name contains the given period as its time index."""
        suffix = f",{period}]"
        bare = f"[{period}]"
        return raw_name.endswith(suffix) or raw_name.endswith(bare) or raw_name == period

    def _is_multi_period(model: "NLModel") -> bool:
        for i in range(min(model.n_vars, 200)):
            nm = model.var_name(i)
            for p in _T_PERIODS:
                if _matches_period(nm, p):
                    return True
        return False

    py_multi = _is_multi_period(py_model)

    py_var_map: dict[str, int] = {}
    for i in range(py_model.n_vars):
        raw = py_model.var_name(i)
        # In multi-period model, only keep entries for py_period
        if py_multi and not _matches_period(raw, py_period):
            continue
        norm = _normalize_name(raw)
        if norm:
            py_var_map.setdefault(norm, i)

    py_con_map: dict[str, int] = {}
    for i in range(py_model.n_cons):
        raw = py_model.con_name(i)
        if py_multi and not _matches_period(raw, py_period):
            continue
        norm = _normalize_name(raw)
        if norm:
            py_con_map.setdefault(norm, i)

    # GAMS names: convert to Python convention.  A GAMS check/shock .nl holds
    # every period of the model (pmeq(...,base) AND pmeq(...,shock)), and
    # stripping the period makes them collide; keep the rows/columns of
    # py_period, or the first-seen BASE row would be compared against Python's
    # shock row (tests/templates/gtap/test_nl_compare_period_filter.py).
    gams_names = [gams_model.con_name(i) for i in range(gams_model.n_cons)]
    gams_periods = {_raw_period(n) for n in gams_names} - {None}
    gams_multi = len(gams_periods) > 1
    if gams_multi and py_period not in gams_periods:
        # Filtering would keep only the period-less rows: a near-empty
        # comparison that still passes an n_common_cons > 0 guard.
        raise ValueError(
            f"py_period={py_period!r} is not in the GAMS .nl "
            f"(periods: {sorted(gams_periods)})"
        )

    def _keep_gams(raw: str) -> bool:
        return not gams_multi or _raw_period(raw) in (None, py_period)

    gams_var_map: dict[str, int] = {}
    for i in range(gams_model.n_vars):
        raw = gams_model.var_name(i)
        if not _keep_gams(raw):
            continue
        norm = _normalize_name(raw)  # strips time, unifies brackets
        if norm and norm not in gams_var_map:
            gams_var_map[norm] = i

    gams_con_map: dict[str, int] = {}
    for i in range(gams_model.n_cons):
        raw = gams_model.con_name(i)
        if not _keep_gams(raw):
            continue
        # Convert GAMS {name}eq(...) → eq_{name}[...]
        norm_raw = _normalize_name(raw)
        norm = _gams_to_python_name(raw)
        if norm and norm not in gams_con_map:
            gams_con_map[norm] = i
        # Also index by raw-normalized for presence diff
        if norm_raw and norm_raw not in gams_con_map:
            gams_con_map[norm_raw] = i

    structural = {
        "py_vars": py_model.n_vars,
        "py_cons": py_model.n_cons,
        "py_nnz": py_model.n_nonzeros,
        "gams_vars": gams_model.n_vars,
        "gams_cons": gams_model.n_cons,
        "gams_nnz": gams_model.n_nonzeros,
    }

    py_var_names = set(py_var_map)
    gams_var_names = set(gams_var_map)
    only_py_vars = sorted(py_var_names - gams_var_names)
    only_gams_vars = sorted(gams_var_names - py_var_names)

    py_con_names = set(py_con_map)
    gams_con_names = set(gams_con_map)
    only_py_cons = sorted(py_con_names - gams_con_names)
    only_gams_cons = sorted(gams_con_names - py_con_names)

    # Coefficient diff by constraint family (only constraints common to both).
    # n_common_cons==0 means the two .nl files share NO named constraint — almost
    # always because the GAMS .nl lacks .row/.col sidecars so its rows parse as
    # anonymous c[idx]. In that case a "0 diffs" result is VACUOUS (nothing was
    # compared), NOT "algebra matches"; the caller must report that distinctly.
    family_stats: dict[str, dict] = {}
    # Per-CELL Jacobian diffs (a coefficient lives at the eq↔var crossing). Each
    # entry is the FULL info Phase 2 needs to name the exact cell and judge scale:
    #   (con_name, vname, py_c, effective_gams_c, abs_diff, rel)
    # Capped per family (cell_cap) so large datasets don't blow up the JSON; the
    # real per-family count is tracked in family_stats["n_diff"] regardless.
    cell_diffs: list[tuple] = []
    cell_truncated: dict[str, int] = {}  # family -> how many were dropped by the cap
    common_cons = py_con_names & gams_con_names

    for con_name in common_cons:
        py_ci = py_con_map[con_name]
        gams_ci = gams_con_map[con_name]
        py_row = py_model.J.get(py_ci, {})
        gams_row = gams_model.J.get(gams_ci, {})

        py_coeffs = {_normalize_name(py_model.var_name(vi)): c for vi, c in py_row.items()}
        gams_coeffs = {_normalize_name(gams_model.var_name(vi)): c
                       for vi, c in gams_row.items()}

        fam = _family(con_name)
        if fam not in family_stats:
            family_stats[fam] = {"n_pairs": 0, "n_diff": 0, "max_rel": 0.0, "worst": None}
        fs = family_stats[fam]

        common_vars = set(py_coeffs) & set(gams_coeffs)

        # --- Grupo B filter: GAMS multi-period .fx fixings ---
        # GAMS solves base/check/shock jointly and fixes prior-period variables
        # with .fx, making their Jacobian coefficients zero. Python has real
        # equations for these. These are structural differences, not bugs.
        _GRUPO_B_FAMILIES = {
            "eq_pi", "eq_u", "eq_psave", "eq_pgdpmp", "eq_rgdpmp",
        }
        if fam in _GRUPO_B_FAMILIES:
            fs.setdefault("_structural_fp", "Grupo B: GAMS .fx multi-period")
            continue

        # --- Grupo C filter: known structural differences ---
        # eq_savf: savf_bar[Oceania] differs ~7.9% due to 9x10 dataset numerical gap
        # eq_ug: GAMS ugeq is trivially 0=0 (ug not computed in .nl); Python has real eq
        # eq_kstock: xft is free in GAMS (kstockeq constraint), fixed as param in Python
        _GRUPO_C_REASONS = {
            "eq_savf": "Grupo C: savf_bar dataset gap (~7.9% Oceania)",
            "eq_ug": "Grupo C: GAMS ugeq is trivial 0=0 in .nl",
            "eq_kstock": "Grupo C: xft free in GAMS, fixed param in Python",
        }
        if fam in _GRUPO_C_REASONS:
            fs.setdefault("_structural_fp", _GRUPO_C_REASONS[fam])
            continue

        # Detect global sign flip: GAMS uses "0 =e= f(x)" convention while
        # Python uses "LHS = RHS". If every common coefficient is negated by
        # the same factor (-1), the rows are mathematically identical.
        sign_flip = False
        if common_vars:
            ratios = []
            for vname in common_vars:
                gc = gams_coeffs[vname]
                pc = py_coeffs[vname]
                if abs(gc) > 1e-12 and abs(pc) > 1e-12:
                    ratios.append(pc / gc)
            if ratios and all(abs(r - ratios[0]) / max(abs(ratios[0]), 1e-10) < 1e-6 for r in ratios):
                sign_flip = abs(ratios[0] - (-1.0)) < 1e-6

        # --- Grupo A filter: Python linearizes nonlinear products ---
        # When GAMS has a non-trivial C-block (nonlinear body) for this row AND
        # a variable has gams_coeff=0 but py_coeff≠0, Python is linearizing a
        # product (e.g. chif*regy, kappaf*pf*xf). Same equation, different .nl form.
        gams_has_nonlinear = gams_ci in gams_model.has_nonlinear_body

        for vname in common_vars:
            py_c = py_coeffs[vname]
            gams_c = gams_coeffs[vname]
            # Skip linearization false positives: GAMS nonlinear body + gams_c==0 + py_c!=0
            if gams_has_nonlinear and abs(gams_c) < 1e-12 and abs(py_c) > 1e-12:
                continue
            fs["n_pairs"] += 1
            # Normalize GAMS coeff by -1 if global sign-flip detected
            effective_gams_c = -gams_c if sign_flip else gams_c
            denom = max(abs(effective_gams_c), 1e-10)
            rel = abs(py_c - effective_gams_c) / denom
            if rel > tol_rel:
                fs["n_diff"] += 1
                if rel > fs["max_rel"]:
                    fs["max_rel"] = rel
                    fs["worst"] = (con_name, vname, py_c, gams_c, rel)
                # Accumulate this Jacobian cell (capped per family). abs_diff and
                # rel are BOTH kept: a 1e-9 coeff off 100% rel is noise, a 1.0
                # coeff off 0.3 abs is the lead — Phase 2 needs both to not
                # confuse scale.
                abs_diff = abs(py_c - effective_gams_c)
                if fs["n_diff"] <= cell_cap:
                    cell_diffs.append(
                        (con_name, vname, py_c, effective_gams_c, abs_diff, rel))
                else:
                    cell_truncated[fam] = cell_truncated.get(fam, 0) + 1

    return {
        "structural": structural,
        "only_py_vars": only_py_vars,
        "only_gams_vars": only_gams_vars,
        "only_py_cons": only_py_cons,
        "only_gams_cons": only_gams_cons,
        "family_stats": family_stats,
        "cell_diffs": cell_diffs,
        "cell_truncated": cell_truncated,
        "n_common_cons": len(common_cons),
    }


def diff_bounds(
    py_model: "NLModel",
    gams_model: "NLModel",
    tol_rel: float = 1e-4,
) -> list[dict]:
    """Compare variable bounds (lb, ub) between Python and GAMS .nl files.

    Returns list of diffs: vars where Python and GAMS have meaningfully
    different lower or upper bounds. Useful for diagnosing iterloop.gms
    bound differences that affect solver behaviour but not equations.
    """
    _INF = 1e30

    # Build name → (lb, ub) maps, normalizing variable names
    def _bounds_map(model: "NLModel") -> dict[str, tuple[float, float]]:
        result = {}
        for i, (lb, ub) in enumerate(model.bounds):
            name = _normalize_name(model.var_name(i))
            if name:
                result[name] = (lb, ub)
        return result

    py_bounds = _bounds_map(py_model)
    gams_bounds = _bounds_map(gams_model)

    common = set(py_bounds) & set(gams_bounds)
    diffs = []

    for name in sorted(common):
        py_lb, py_ub = py_bounds[name]
        g_lb, g_ub = gams_bounds[name]

        # Compare lb
        lb_diff = False
        if abs(py_lb) < _INF and abs(g_lb) < _INF:
            denom = max(abs(g_lb), 1e-12)
            lb_diff = abs(py_lb - g_lb) / denom > tol_rel
        elif (py_lb < -_INF/2) != (g_lb < -_INF/2):
            lb_diff = True  # one is -inf, other isn't

        # Compare ub
        ub_diff = False
        if abs(py_ub) < _INF and abs(g_ub) < _INF:
            denom = max(abs(g_ub), 1e-12)
            ub_diff = abs(py_ub - g_ub) / denom > tol_rel
        elif (py_ub > _INF/2) != (g_ub > _INF/2):
            ub_diff = True  # one is +inf, other isn't

        if lb_diff or ub_diff:
            diffs.append({
                "var": name,
                "py_lb": py_lb, "py_ub": py_ub,
                "g_lb": g_lb, "g_ub": g_ub,
                "lb_diff": lb_diff, "ub_diff": ub_diff,
            })

    # Also report vars that are fixed in one but not the other (lb==ub)
    only_py = sorted(set(py_bounds) - set(gams_bounds))
    only_gams = sorted(set(gams_bounds) - set(py_bounds))

    return diffs, only_py, only_gams


def print_diff_report(result: dict, period: str, tol_rel: float = 1e-4) -> None:
    s = result["structural"]
    print(f"\n=== .nl COMPARISON [{period}] ===")
    print(f"  Python:  {s['py_vars']:6d} vars  {s['py_cons']:6d} cons  {s['py_nnz']:8d} nnz")
    print(f"  GAMS:    {s['gams_vars']:6d} vars  {s['gams_cons']:6d} cons  {s['gams_nnz']:8d} nnz")

    if result["only_py_vars"]:
        print(f"\n  Variables only in Python ({len(result['only_py_vars'])}):")
        for n in result["only_py_vars"][:20]:
            print(f"    {n}")

    if result["only_gams_vars"]:
        print(f"\n  Variables only in GAMS ({len(result['only_gams_vars'])}):")
        for n in result["only_gams_vars"][:20]:
            print(f"    {n}")

    if result["only_py_cons"]:
        print(f"\n  Constraints only in Python ({len(result['only_py_cons'])}):")
        for n in result["only_py_cons"][:20]:
            print(f"    {n}")

    if result["only_gams_cons"]:
        print(f"\n  Constraints only in GAMS ({len(result['only_gams_cons'])}):")
        for n in result["only_gams_cons"][:20]:
            print(f"    {n}")

    fam = result["family_stats"]
    if fam:
        print(f"\n  Coefficient diff by family (tol_rel={tol_rel:.0e}):")
        print(f"  {'family':<30s} {'pairs':>7s} {'diff':>6s} {'max_rel':>10s}  status")
        print(f"  {'-'*30} {'-'*7} {'-'*6} {'-'*10}  ------")
        for fname, fs in sorted(fam.items(), key=lambda x: -x[1]["max_rel"]):
            if fs.get("_structural_fp"):
                reason = fs["_structural_fp"]
                print(f"  {fname:<30s} {'—':>7s} {'—':>6s} {'—':>10s}  STRUCTURAL_FP ({reason})")
                continue
            worst_str = ""
            if fs["worst"] and fs["n_diff"] > 0:
                cn, vn, py_c, gams_c, rel = fs["worst"]
                worst_str = (f"  worst: {cn} var={vn} "
                             f"py={py_c:.4e} gams={gams_c:.4e}")
            status = "OK" if fs["n_diff"] == 0 else "DIFF"
            print(f"  {fname:<30s} {fs['n_pairs']:>7d} {fs['n_diff']:>6d} "
                  f"{fs['max_rel']:>10.2e}  {status}{worst_str}")
    else:
        print("\n  No common constraints found — models may have incompatible structure.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _phase_to_period(phase: str):
    """nl_compare's --phase includes 'altertax', which is off the base/check/
    shock axis. normalize_period collapses it to None; this is just explicit."""
    return phase if phase in ("base", "check", "shock") else None


def _cells_to_violations(result: dict) -> list[dict]:
    """Map per-cell Jacobian diffs to common-schema violations.

    entity names the eq↔var CROSSING (a coefficient lives there). index carries
    the sets of BOTH sides. value=rel; diff_abs and diff_rel BOTH exposed so
    Phase 2 can tell a 1e-9 coeff off 100% (noise) from a 1.0 coeff off 0.3 (lead).
    """
    viols = []
    for con_name, vname, py_c, gams_c, abs_diff, rel in result.get("cell_diffs", []):
        eq_base, eq_idx = _split_name(con_name)
        var_base, var_idx = _split_name(vname)
        v = make_violation(f"{eq_base} × {var_base}",
                           {"eq": eq_idx, "var": var_idx}, "jac_coef_diff", rel)
        v["diff_rel"] = rel
        v["diff_abs"] = abs_diff
        v["py_coeff"] = py_c
        v["gams_coeff"] = gams_c
        viols.append(v)
    return viols


def _run_nl(args) -> dict:
    """Build/parse the .nl files, diff per phase, return a common-schema dict."""
    from run_gtap import _build_gtap_contract_with_calibration  # noqa: F401
    from _parity_datasets import DATASETS
    from _nl_parser import parse_nl

    args.out_dir.mkdir(parents=True, exist_ok=True)

    global KEEP_R, KEEP_I
    if args.keep_r:
        KEEP_R = list(args.keep_r)
    if args.keep_i:
        KEEP_I = list(args.keep_i)
    custom_subset = bool(args.keep_r or args.keep_i)
    full = args.full_9x10
    gams_shrink = (args.gams_shrink or custom_subset) and not full

    ds = DATASETS.get(args.dataset) or DATASETS["9x10"]
    use_agg = ds.mode == "compstat" and ds.agg_gdx is not None
    if use_agg:
        from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
        _reg = _gdx_set_elements(ds.agg_gdx, "REG")
        _comm = _gdx_set_elements(ds.agg_gdx, "COMM")
        _rres, _rmuv, _imuv = _muv_baskets(_reg, _comm)
        closure_config = GTAPClosureConfig(if_sub=False, rmuv=_rmuv, imuv=_imuv)
        full, gams_shrink = True, False
    else:
        contract = _build_gtap_contract_with_calibration(ds.nl_contract)
        closure_update = {"if_sub": False}
        if full:
            closure_update["rmuv"], closure_update["imuv"] = ds.nl_full_muv
        elif gams_shrink:
            _rres, _rmuv, _imuv = _muv_baskets(KEEP_R, KEEP_I)
            closure_update["rmuv"] = _rmuv
            closure_update["imuv"] = _imuv
        closure_config = contract.closure.model_copy(update=closure_update)

    _har_dir = (dataset_dir(ds.name)) if use_agg else None
    if not args.skip_python:
        py_paths = build_python_nls(
            args.phase, args.out_dir, closure_config,
            gdx_path=(None if use_agg else ds.agg_gdx),
            do_shrink=not full, har_dir=_har_dir)
    else:
        py_paths = {p: args.out_dir / f"python_{p}.nl" for p in args.phase}

    gams_nl_paths: dict[str, Path] = {}
    if not args.skip_gams:
        gams_nl_paths = fetch_gams_nl(
            args.out_dir, args.neos_email, list(args.phase), shrink=gams_shrink,
            agg_gdx=(ds.agg_gdx if use_agg else None), har_dir=_har_dir,
            ifsub=args.ifsub, local=args.gams_local)
    else:
        for phase in args.phase:
            for p in (args.out_dir / f"gams_{phase}.nl",
                      args.out_dir / "gams_compstat.nl"):
                if p.exists():
                    gams_nl_paths[phase] = p
                    break

    per_phase = {}
    all_viols = []
    skipped = []
    for period in args.phase:
        py_nl = py_paths.get(period)
        gams_nl = gams_nl_paths.get(period)
        if py_nl is None or not Path(py_nl).exists():
            skipped.append({"phase": period, "reason": "python_nl_missing"})
            continue
        if gams_nl is None or not Path(gams_nl).exists():
            skipped.append({"phase": period, "reason": "gams_nl_missing"})
            continue
        py_m = parse_nl(py_nl)
        gams_m = parse_nl(gams_nl)
        result = diff_nl_models(py_m, gams_m, tol_rel=args.tol_rel,
                                py_period=period, cell_cap=args.cell_cap)
        print_diff_report(result, period, tol_rel=args.tol_rel)  # -> stderr
        viols = _cells_to_violations(result)
        for v in viols:
            v["phase"] = period
        all_viols.extend(viols)
        n_diff = sum(fs["n_diff"] for fs in result["family_stats"].values())
        per_phase[period] = {
            "n_common_cons": result["n_common_cons"],
            "comparable": result["n_common_cons"] > 0,
            "n_cell_diffs_total": n_diff,
            "n_cell_diffs_reported": len(viols),
            "cell_truncated": result["cell_truncated"],
            "structural": result["structural"],
            "only_py_vars": result["only_py_vars"][:50],
            "only_gams_vars": result["only_gams_vars"][:50],
            "only_py_cons": result["only_py_cons"][:50],
            "only_gams_cons": result["only_gams_cons"][:50],
        }

    if args.skip_python and len(gams_nl_paths) == len(args.phase):
        # Asked for the GAMS side only (regenerate_gams.py): writing it is the job.
        return dict(
            status="clean", period=None,
            headline=f"GAMS .nl written for {', '.join(args.phase)} (python side skipped)",
            violations=[], meta={"gams_nl": {p: str(v) for p, v in gams_nl_paths.items()}})

    if not per_phase and skipped:
        return dict(
            status="error", period=None,
            headline=("no phase could be diffed — " +
                      "; ".join(f"{s['phase']}:{s['reason']}" for s in skipped)),
            violations=[], meta={"error_kind": "no_phase_diffed",
                                 "skipped": skipped, "phases": list(args.phase),
                                 "skip_gams": args.skip_gams,
                                 "skip_python": args.skip_python})

    diffed = list(per_phase.keys())
    comparable_phases = [p for p, d in per_phase.items() if d["comparable"]]
    n_total = sum(p["n_cell_diffs_total"] for p in per_phase.values())
    # period: the phase if exactly one was diffed, else None (off the orchestrator
    # axis). When None the per-phase detail lives in meta.per_phase.
    period = _phase_to_period(diffed[0]) if len(diffed) == 1 else None

    # NO phase shared any named constraint → the comparison was VACUOUS (the GAMS
    # .nl has no .row/.col so its rows are anonymous c[idx]). Report error, NOT a
    # spurious "clean" — Phase 2 must not read this as "algebra matches".
    if not comparable_phases:
        return dict(
            status="error", period=period,
            headline=(f"no common named constraints in any phase "
                      f"({'/'.join(diffed)}) — GAMS .nl likely missing .row/.col "
                      f"sidecars; coefficient diff is vacuous, not 'clean'"),
            violations=[],
            meta={"error_kind": "no_common_constraints",
                  "comparison": "python_vs_gams_per_phase",
                  "phases_diffed": diffed, "phases_skipped": skipped,
                  "tol_rel": args.tol_rel, "dataset_mode": ds.mode,
                  "per_phase": per_phase})

    status = "dirty" if all_viols else "clean"
    if status == "clean":
        headline = (f"Jacobian coefficient diff ({'/'.join(comparable_phases)}): "
                    f"0 cells differ at tol_rel={args.tol_rel:g} over "
                    f"{sum(per_phase[p]['n_common_cons'] for p in comparable_phases)} "
                    f"common constraints — algebra matches; this layer does not "
                    f"explain the gap")
    else:
        worst = max(all_viols, key=lambda v: abs(v["value"]))
        headline = (f"Jacobian coefficient diff ({'/'.join(comparable_phases)}): "
                    f"{n_total} cell(s) differ > {args.tol_rel:g}; worst "
                    f"{worst['entity']} rel={worst['diff_rel']:.2e} "
                    f"abs={worst['diff_abs']:.2e}")
    return dict(
        status=status, period=period, headline=headline, violations=all_viols,
        meta={"comparison": "python_vs_gams_per_phase",
              "phases_diffed": diffed, "phases_comparable": comparable_phases,
              "phases_skipped": skipped,
              "tol_rel": args.tol_rel, "cell_cap": args.cell_cap,
              "n_cell_diffs_total": n_total, "dataset_mode": ds.mode,
              "per_phase": per_phase})


def main() -> int:
    ap = argparse.ArgumentParser(description="GTAP .nl comparison tool (Layer 6 diagnostic)")
    ap.add_argument("--fetch-gdx-9x10", action="store_true",
                    help="Submit full 9x10 PATH solve to NEOS (10%% tariff shock), "
                         "download out.gdx to --out-dir/output/gtap9x10_neos/. "
                         "Use with --neos-email.")
    ap.add_argument("--dataset", default="3x3",
                    help="3x3 / 9x10 (9x10 GDX, optionally shrunk), or a GTAPAgg "
                         "registry key with a consolidated GDX, e.g. gtap7_3x3, "
                         "gtap7_5x5, gtap7_10x7, gtap7_20x41 (sets-agnostic NEOS).")
    ap.add_argument("--phase", nargs="+", choices=["base", "check", "shock", "altertax"],
                    default=["base", "check", "shock"])
    ap.add_argument("--neos-email", default="dracomarmol@gmail.com")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/gtap_nl"))
    ap.add_argument("--skip-gams", action="store_true",
                    help="Skip NEOS submission; use existing gams_compstat.nl")
    ap.add_argument("--skip-python", action="store_true",
                    help="Skip Python model builds; use existing python_*.nl files")
    ap.add_argument("--tol-rel", type=float, default=1e-4,
                    help="Relative tolerance for coefficient diff")
    ap.add_argument("--gams-shrink", action="store_true",
                    help="Shrink the GAMS sets to the same KEEP_R x KEEP_I subset "
                         "the Python side uses (full 9x10 data, subset sets) — "
                         "true apples-to-apples 3x3 .nl comparison.")
    ap.add_argument("--full-9x10", action="store_true",
                    help="Run BOTH sides on the full 9x10 (no _shrink_sets, GAMS "
                         "default full getdata) — largest in-dataset comparison. "
                         "Overrides --gams-shrink.")
    ap.add_argument("--keep-r", nargs="+", default=None,
                    help="Override KEEP_R: regions to keep for the shrunk subset "
                         "(intermediate 9x10 sizes, e.g. 5x5). Implies --gams-shrink.")
    ap.add_argument("--keep-i", nargs="+", default=None,
                    help="Override KEEP_I: commodities to keep for the shrunk subset.")
    ap.add_argument("--cell-cap", type=int, default=50,
                    help="Max Jacobian-cell diffs reported per family in the JSON "
                         "(prevents a huge payload on large datasets; the real "
                         "per-family count is always in meta).")
    ap.add_argument("--mode", default="altertax", choices=["altertax", "gtap"],
                    help="altertax CD (default) or pure-gtap real-CES")
    ap.add_argument("--ifsub", type=int, default=0, choices=[0, 1],
                    help="ifSUB of the GAMS model (CONVERT .nl) and of the pure-gtap Python side")
    ap.add_argument("--gams-local", action="store_true",
                    help="write the GAMS .nl with the local GAMS CONVERT instead of NEOS")
    args = ap.parse_args()

    # mode=gtap UNSUPPORTED, honestly: the pure-gtap shock is multi-period — the
    # tariff wedge enters the equations via solve_multiperiod(mode="gtap") (the
    # _rebuild_eq_pmeq_shock / NatRes-anchor / ytax[mt] in-equation rebuilds, which
    # MUTATE constraints in-place at solve time). A single-period .nl write (what
    # this tool does) cannot capture those, so a gtap .nl would diff the WRONG
    # (un-shocked) equations and a "clean" would be a lie. Emit an explicit
    # mode_unsupported error rather than silently writing an altertax/un-shocked .nl.
    if args.mode == "gtap":
        def _unsupported() -> dict:
            return dict(
                status="error", period=args.phase[0] if args.phase else None,
                headline=("nl_compare does not support --mode gtap: the pure-gtap "
                          "shock is wired in solve_multiperiod (constraints rebuilt "
                          "in-place at solve time), which a single-period .nl write "
                          "cannot represent. Use seed_and_solve --mode gtap instead."),
                violations=[],
                meta={"error_kind": "mode_unsupported", "mode": "gtap",
                      "ifsub": args.ifsub})
        return run_tool("nl_compare", args.dataset, _unsupported,
                        period_hint=args.phase[0] if args.phase else None)

    # --fetch-gdx-9x10 is a NEOS side-effect (submit + download), not a diff. It
    # still emits JSON (status clean on success, error on failure) so stdout stays
    # parseable, but it is not a cascade-signal run.
    if args.fetch_gdx_9x10:
        def _fetch() -> dict:
            out = ROOT / "output" / "gtap9x10_neos"
            gdx = submit_and_fetch_gdx_9x10(out_dir=out, neos_email=args.neos_email)
            return dict(status="clean", period=None,
                        headline=f"fetched 9x10 out.gdx: {gdx}",
                        violations=[], meta={"mode": "fetch_gdx_9x10",
                                             "gdx": str(gdx)})
        return run_tool("nl_compare", args.dataset, _fetch)

    return run_tool("nl_compare", args.dataset, lambda: _run_nl(args))


if __name__ == "__main__":
    raise SystemExit(main())
