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

GDX9 = ROOT / "src/equilibria/templates/reference/gtap/data/basedata-9x10.gdx"
COMP_GMS = ROOT / "src/equilibria/templates/reference/gtap/scripts/comp.gms"
DATA_DIR = ROOT / "src/equilibria/templates/reference/gtap/data"

KEEP_R = ["Oceania", "NAmerica", "EU_28"]
KEEP_I = ["c_Crops", "c_MeatLstk", "c_Extraction", "c_TransComm"]

# comp.gms MUV-basket / residual reference sets are hardcoded for the 9x10.
# For any subset we must restrict them to kept elements (else $170 domain
# violations) and mirror them on the Python closure so eq_pmuv is comparable.
_MANUF = ("c_ProcFood", "c_TextWapp", "c_LightMnfc", "c_HeavyMnfc")
_HIC = ("Oceania", "NAmerica", "EU_28")


def _muv_baskets(keep_r: list[str], keep_i: list[str]) -> tuple[str, tuple, tuple]:
    """Return (rres, rmuv, imuv) restricted to the kept subset.

    rres  = residual region (NAmerica if kept, else last kept region)
    rmuv  = HIC regions present in keep_r (fallback: first 3 kept)
    imuv  = manufactures present in keep_i (fallback: last kept commodity)
    """
    kr, ki = set(keep_r), set(keep_i)
    rmuv = tuple(r for r in _HIC if r in kr) or tuple(keep_r[:3])
    imuv = tuple(i for i in _MANUF if i in ki) or (keep_i[-1],)
    rres = "NAmerica" if "NAmerica" in kr else keep_r[-1]
    return rres, rmuv, imuv

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

def _gdxdump_params_only(gdx_path: Path, renames: dict[str, str] | None = None,
                         universal_domains: bool = False) -> str:
    """Dump a GDX file, keeping only Parameter/Scalar blocks (strip Set blocks).

    renames: maps GDX symbol name (uppercase) → GAMS identifier name.
    universal_domains: rewrite every Parameter domain to (*,*,...). Needed for the
        _agg path: a consolidated GTAPAgg GDX reports domains that don't match the
        data labels (e.g. vdfb declared (acts,acts,reg) but data is (comm,acts,reg))
        → GAMS $170 Domain violation. Universal domains hold the data verbatim and
        the model indexes them with proper sets (standard GTAP getData practice).
    """
    import subprocess
    import re as _re
    GDXDUMP = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump")
    if not GDXDUMP.exists() or not gdx_path.exists():
        return f"* gdxdump not available or {gdx_path.name} not found"
    try:
        raw = subprocess.check_output(
            [str(GDXDUMP), str(gdx_path)],
            stderr=subprocess.DEVNULL, timeout=30
        ).decode()
    except Exception as e:
        return f"* gdxdump failed: {e}"

    lines = []
    skip_block = False
    skip_symbol: set[str] = set()
    if renames:
        # Symbols to skip entirely (value is None) or rename
        skip_symbol = {k.upper() for k, v in (renames or {}).items() if v is None}

    for line in raw.splitlines():
        if line.startswith(("$onEmpty", "$offEmpty")):
            continue
        if line.startswith("Set "):
            skip_block = True
            continue
        if skip_block:
            if line.strip().endswith("/;"):
                skip_block = False
            continue
        # Rename Parameter/Scalar header lines
        if renames and (line.startswith("Parameter ") or line.startswith("Scalar ")):
            m = _re.match(r'^(Parameter|Scalar)\s+(\w+)(\(|\s)', line)
            if m:
                sym = m.group(2).upper()
                if sym in skip_symbol:
                    skip_block = True
                    continue
                if sym in {k.upper() for k in renames}:
                    rename_to = next(v for k, v in renames.items() if k.upper() == sym)
                    line = line.replace(m.group(2), rename_to, 1)
        if universal_domains and (line.startswith("Parameter ") or line.startswith("Scalar ")):
            # Rewrite an explicit domain (d1,d2,...) to (*,*,...) of the same arity.
            line = _re.sub(
                r'^((?:Parameter|Scalar)\s+\w+)\(([^)]*)\)',
                lambda mm: mm.group(1) + "(" + ",".join("*" * len(mm.group(2).split(","))) + ")",
                line,
            )
        lines.append(line)
    return "\n".join(lines)


_PRM_HAR_NAME_MAP: dict[str, str] = {
    "ESBD": "esubd", "ESBM": "esubm", "ESBT": "esubt", "ESBC": "esubc",
    "ESBV": "esubva", "ESBQ": "esubq", "ESBG": "esubg", "ESBI": "esubi",
    "ESBS": "esubs", "ETRQ": "etraq", "ETRE": "etrae",
    "INCP": "incpar", "SUBP": "subpar", "RFLX": "rorFlex0",
}


def _prm_har_as_assignments(prm_path: Path) -> str:
    """Emit GAMS parameter assignments from a GEMPACK default.prm file.

    Reads elasticity headers (ESBD→esubd, ETRQ→etraq, etc.) and emits
    plain GAMS assignment statements. Only emits headers in _PRM_HAR_NAME_MAP.
    """
    import numpy as np
    from equilibria.babel.har.reader import read_har
    headers = read_har(str(prm_path))
    out_parts: list[str] = []
    for har_name, gams_name in _PRM_HAR_NAME_MAP.items():
        h = headers.get(har_name)
        if h is None or h.array.ndim == 0:
            continue
        dims = h.array.ndim
        dims_str = ",".join(["*"] * dims)
        out_parts.append(f"parameter {gams_name}({dims_str}) ;")
        label_lists = h.set_elements  # list of lists, one per dimension
        it = np.nditer(h.array, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            val = float(it[0])
            if val != 0.0:
                keys_fmt = ",".join(f'"{label_lists[d][i]}"' for d, i in enumerate(idx))
                out_parts.append(f"{gams_name}({keys_fmt}) = {val} ;")
            it.iternext()
    return "\n".join(out_parts)


def _prm_as_assignments(gdx_path: Path, renames: dict[str, str] | None = None) -> str:
    """Convert gdxdump Parameter blocks to plain GAMS assignment statements.

    Avoids $onMulti/$offMulti (confirmed to trigger $767 in NEOS GAMS 52.5.0).
    Output style:
        parameter etraq(*,*) ;
        etraq("a_agricultur","Oceania") = -5 ;
        ...
    """
    import subprocess, re as _re
    GDXDUMP = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump")
    if not GDXDUMP.exists() or not gdx_path.exists():
        return f"* gdxdump not available or {gdx_path.name} not found\n"
    try:
        raw = subprocess.check_output(
            [str(GDXDUMP), str(gdx_path)],
            stderr=subprocess.DEVNULL, timeout=30
        ).decode()
    except Exception as e:
        return f"* gdxdump failed: {e}\n"

    renames_upper = {k.upper(): v for k, v in (renames or {}).items()}

    out_parts: list[str] = []

    # State machine: parse each Parameter/Scalar block
    in_block = False
    skip_block = False   # Set blocks
    cur_name = ""        # GAMS-output name (after rename)
    cur_dims = 0
    cur_data: list[str] = []

    def flush_block():
        if not cur_name:
            return
        dims_str = ",".join(["*"] * cur_dims) if cur_dims else "*"
        out_parts.append(f"parameter {cur_name}({dims_str}) ;")
        for entry in cur_data:
            out_parts.append(entry)

    for line in raw.splitlines():
        stripped = line.strip()

        # Skip gdxdump directives
        if stripped.startswith(("$onEmpty", "$offEmpty")):
            continue

        # Track Set blocks (skip entirely)
        if stripped.startswith("Set "):
            if in_block:
                flush_block()
                in_block = False
                cur_data = []
            skip_block = True
            continue
        if skip_block:
            if stripped.endswith("/;") or stripped == "/;":
                skip_block = False
            continue

        # New Parameter/Scalar block
        if stripped.startswith("Parameter ") or stripped.startswith("Scalar "):
            if in_block:
                flush_block()
                cur_data = []
            m = _re.match(r'^(?:Parameter|Scalar)\s+(\w+)((?:\([^)]*\))?)', stripped)
            if m:
                sym_raw = m.group(1)
                sym_up = sym_raw.upper()
                dom_str = m.group(2)  # e.g. "(*,*)" or ""
                out_name = renames_upper.get(sym_up, sym_raw)
                dim_count = dom_str.count(",") + 1 if dom_str else 1
                cur_name = out_name
                cur_dims = dim_count
                cur_data = []
                in_block = True
            continue

        # End of block — may also contain a data entry on the same line
        if stripped == "/;" or stripped.endswith("/;"):
            # Check if there's data before the /; terminator
            data_part = stripped[:-2].rstrip().rstrip(",").rstrip()
            if data_part and in_block and not data_part.startswith("*"):
                parts = data_part.rsplit(None, 1)
                if len(parts) == 2:
                    keys_part, val = parts
                    keys = [k.strip("'") for k in keys_part.split(".")]
                    keys_fmt = ",".join(f'"{k}"' for k in keys)
                    cur_data.append(f'{cur_name}({keys_fmt}) = {val} ;')
            in_block = False
            flush_block()
            cur_data = []
            cur_name = ""
            continue

        # Data lines inside a parameter block: 'a'.'b' value,
        if in_block and stripped and not stripped.startswith("*"):
            # Strip trailing comma
            entry = stripped.rstrip(",").rstrip()
            # Parse: 'key1'.'key2' ... value  OR  'key' value
            # Split on last whitespace to get keys_part and value
            parts = entry.rsplit(None, 1)
            if len(parts) == 2:
                keys_part, val = parts
                # Convert 'k1'.'k2'.'k3' → "k1","k2","k3"
                keys = [k.strip("'") for k in keys_part.split(".")]
                keys_fmt = ",".join(f'"{k}"' for k in keys)
                cur_data.append(f'{cur_name}({keys_fmt}) = {val} ;')

    if in_block:
        flush_block()

    return "\n".join(out_parts) + "\n"


def _build_getdata_replacement() -> str:
    """Build complete getData.gms replacement — fully self-contained, no $gdxin.

    Hardcodes all 9x10 sets, aliases, and derived sets exactly as getData.gms
    would construct them. Embeds all parameter data from 9x10Dat.gdx and
    9x10Prm.gdx via gdxdump. No $gdxin or set.acts syntax used.
    """
    # getData.gms renames: $load pop0=pop (and rorFlex0=rorFlex in Prm)
    dat_dump = _gdxdump_params_only(DATA_DIR / "9x10Dat.gdx", renames={"POP": "pop0"})
    prm_dump = _gdxdump_params_only(DATA_DIR / "9x10Prm.gdx", renames={"RORFLEX": "rorFlex0"})

    return r"""
* ============================================================
* getData.gms REPLACEMENT — fully self-contained, no $gdxin
* Generated by nl_compare.py _build_getdata_replacement()
* ============================================================

* --- Primary sets (from %baseName%Sets.gdx) ---
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

* --- Standard SAM labels ---
set stdlab "Standard SAM labels" /
   TRD, hhd, gov, inv, deprY, tmg, itax, ptax, mtax, etax, vtax, vsub, dtax, ctax, bop, tot
/ ;

set findem(stdlab) "Final demand accounts" / hhd, gov, inv, tmg / ;

* --- is superset: acts + comm + endw + stdlab + reg ---
set is "SAM accounts" /
   a_agricultur, a_Extraction, a_ProcFood, a_TextWapp, a_LightMnfc, a_HeavyMnfc,
   a_Util_Cons, a_TransComm, a_OthService,
   c_Crops, c_MeatLstk, c_Extraction, c_ProcFood, c_TextWapp, c_LightMnfc,
   c_HeavyMnfc, c_Util_Cons, c_TransComm, c_OthService,
   Land, UnSkLab, SkLab, Capital, NatRes,
   TRD, hhd, gov, inv, deprY, tmg, itax, ptax, mtax, etax, vtax, vsub, dtax, ctax, bop, tot,
   Oceania, EastAsia, SEAsia, SouthAsia, NAmerica, LatinAmer, EU_28, MENA, SSA, RestofWorld
/ ;
alias(is, js) ;

* --- Derived sets ---
set aa(is) "Armington agents" / a_agricultur, a_Extraction, a_ProcFood, a_TextWapp,
                                 a_LightMnfc, a_HeavyMnfc, a_Util_Cons, a_TransComm,
                                 a_OthService, hhd, gov, inv, tmg / ;
set a(aa)  "Activities"       / a_agricultur, a_Extraction, a_ProcFood, a_TextWapp,
                                 a_LightMnfc, a_HeavyMnfc, a_Util_Cons, a_TransComm,
                                 a_OthService / ;
set i(is)  "Commodities"      / c_Crops, c_MeatLstk, c_Extraction, c_ProcFood,
                                 c_TextWapp, c_LightMnfc, c_HeavyMnfc, c_Util_Cons,
                                 c_TransComm, c_OthService / ;
alias(i, j) ;

set r(is)  "Regions" / Oceania, EastAsia, SEAsia, SouthAsia, NAmerica,
                        LatinAmer, EU_28, MENA, SSA, RestofWorld / ;
alias(r,s) ; alias(r,d) ; alias(r,rp) ;

set fp(is) "Factors"          / Land, UnSkLab, SkLab, Capital, NatRes / ;
set fnm(fp) "Non-mobile"      / NatRes / ;
set fm(fp) "Mobile factors" ;
fm(fp)$(not fnm(fp)) = yes ;

set fd(aa) "Domestic final demand agents" / hhd, gov, inv, tmg / ;
set h(fd)  "Households"   / hhd / ;
set gov(fd) "Government"  / gov / ;
set inv(fd) "Investment"  / inv / ;
set fdc(fd) "CES final demand" / gov, inv / ;
set tmg(fd) "Trade margins" / tmg / ;

alias(i0,i) ; alias(a0,a) ; alias(j0,i0) ; alias(m0,i0) ;
set mapa0(a,a0) ;
set mapi0(i,i0) ;
mapa0(a,a0)$(sameas(a,a0)) = yes ;
mapi0(i,i0)$(sameas(i,i0)) = yes ;

* --- Embedded data from 9x10Dat.gdx (POP renamed to pop0) ---
$onMulti
""" + dat_dump + """
$offMulti

* --- fbep/ftrv/ptax/tvom/check derived (getData.gms section) ---
parameter fbep(*,*,*) ; fbep(fp,a0,r) = 0 ;
parameter ftrv(*,*,*) ; ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;
parameter ptax(*,*,*) ; ptax(i0,a0,r) = makb(i0,a0,r) - maks(i0,a0,r) ;
parameter tvom(a0,r)  ; tvom(a0,r) = sum(i0, vdfp(i0,a0,r) + vmfp(i0,a0,r)) + sum(fp, evfp(fp,a0,r)) ;
parameter check(a0,r) ; check(a0,r) = 0 ;

* --- CO2 emissions (zero — no emissions GDX) ---
parameters
   mdf(i0,a0,r), mmf(i0,a0,r), mdp(i0,r), mmp(i0,r),
   mdg(i0,r), mmg(i0,r), mdi(i0,r), mmi(i0,r) ;
mdf(i0,a0,r)=0; mmf(i0,a0,r)=0; mdp(i0,r)=0; mmp(i0,r)=0;
mdg(i0,r)=0; mmg(i0,r)=0; mdi(i0,r)=0; mmi(i0,r)=0;

* --- Embedded elasticities from 9x10Prm.gdx (RORFLEX renamed to rorFlex0) ---
$onMulti
""" + prm_dump + """
$offMulti

* --- Derived parameters (getData.gms lines 344-383) ---
* pop0 and rorFlex0 come from gdxdump (renamed from POP, RORFLEX)
parameters
   sigmap(r,a), sigmand(r,a), sigmav(r,a),
   omegas(r,a), sigmas(r,i),
   eh0(r,i), bh0(r,i),
   sigmag(r), sigmai(r),
   sigmam(r,i,aa), sigmaw(r,i), sigmamg(i),
   omegaf(r,fp),
   omegax(r,i), omegaw(r,i),
   etaf(r,fp), etaff(r,fp,a),
   mdtx0(r), RoRFlag ;

sigmap(r,a)    = na ; sigmand(r,a)  = na ; sigmav(r,a)   = na ;
omegas(r,a)    = -etraq(a,r) ;
sigmas(r,i)    = inf$(esubq(i,r) eq 0) + (1/esubq(i,r))$(esubq(i,r) ne 0) ;
eh0(r,i)       = na ; bh0(r,i)      = na ;
sigmag(r)      = na ; sigmai(r)     = na ;
sigmam(r,i,aa) = na ; sigmaw(r,i)   = na ; sigmamg(i)    = na ;
loop(fp$(not fnm(fp)), loop(endwm, omegaf(r,fp)$sameas(endwm,fp) = inf ; ) ;
                        loop(endws, omegaf(r,fp)$sameas(endws,fp) = na  ; ) ; ) ;
parameter rorFlex(r,t) "Flexibility of foreign capital" ;
rorFlex(r,t) = rorFlex0(r) ;
omegax(r,i)    = inf ; omegaw(r,i)  = inf ;
etaf(r,fp)     = 0 ;   etaff(r,fp,a) = 0 ;
mdtx0(r)       = na ;

* ============================================================
* END getData.gms replacement
* ============================================================
"""


def _fmt_set(elements: list[str], max_line: int = 150) -> str:
    """Join set elements, breaking into multiple lines if needed (GAMS limit ~200 chars)."""
    result, current = [], ""
    for i, el in enumerate(elements):
        sep = ", " if i < len(elements) - 1 else ""
        candidate = (current + ", " + el) if current else el
        if len(candidate + sep) > max_line and current:
            result.append(current + ",")
            current = el
        else:
            current = candidate
    if current:
        result.append(current)
    return ("\n   ".join(result))


def _compute_keep_a(keep_i: list[str]) -> list[str]:
    """Activities producing any commodity in keep_i, from 9x10 MAKB.

    Mirrors diff_3x3_full._shrink_sets keep_a derivation. Activities are the
    'a_*' tokens, commodities the 'c_*' tokens in each MAKB record.
    """
    import subprocess, re as _re
    GDXDUMP = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump")
    raw = subprocess.check_output(
        [str(GDXDUMP), str(DATA_DIR / "9x10Dat.gdx"), "Symb=MAKB"],
        stderr=subprocess.DEVNULL, timeout=30,
    ).decode()
    keep = set(keep_i)
    keep_a: set[str] = set()
    for line in raw.splitlines():
        toks = _re.findall(r"'([^']+)'", line)
        comms = [t for t in toks if t.startswith("c_")]
        acts = [t for t in toks if t.startswith("a_")]
        if any(c in keep for c in comms):
            keep_a.update(acts)
    return sorted(keep_a)


def _build_getdata_replacement_shrunk(keep_r: list[str], keep_i: list[str]) -> str:
    """getData.gms replacement for a KEEP_R x KEEP_I subset of the 9x10.

    Strategy: keep the FULL 9x10 embedded data unchanged, but declare the
    primary/derived SETS shrunk to the subset (reg=keep_r, comm=keep_i,
    acts=keep_a, and the `is` superset built from those). Because every model
    equation ranges over the subset sets (r, i, a, ...), only the subset's data
    is ever referenced — exactly mirroring the Python `_shrink_sets` drop, which
    is itself a pure filter of the same 9x10 dicts. No data filtering needed.
    """
    keep_a = _compute_keep_a(keep_i)

    # getData.gms renames: $load pop0=pop (and rorFlex0=rorFlex in Prm) — same
    # full 9x10 dumps as the un-shrunk path.
    dat_dump = _gdxdump_params_only(DATA_DIR / "9x10Dat.gdx", renames={"POP": "pop0"})
    prm_dump = _gdxdump_params_only(DATA_DIR / "9x10Prm.gdx", renames={"RORFLEX": "rorFlex0"})

    # Standard GTAP 5-factor partition (full, as in the 9x10 path).
    endw  = ["Land", "UnSkLab", "SkLab", "Capital", "NatRes"]
    endwf = ["NatRes"]
    endwm = ["UnSkLab", "SkLab", "Capital"]
    endws = ["Land"]
    marg  = ["c_TransComm"]

    is_elems = keep_a + keep_i + endw + [
        "TRD", "hhd", "gov", "inv", "deprY", "tmg", "itax", "ptax", "mtax",
        "etax", "vtax", "vsub", "dtax", "ctax", "bop", "tot",
    ] + keep_r

    return (
        "\n* ============================================================\n"
        "* getData.gms REPLACEMENT (shrunk subset) — full 9x10 data, subset sets\n"
        f"* keep_r={keep_r}  keep_i={keep_i}  keep_a={keep_a}\n"
        "* ============================================================\n\n"
        "* --- Primary sets (SHRUNK to subset) ---\n"
        "sets\n"
        f'   acts           "Activities"      / {_fmt_set(keep_a)} /\n'
        f'   comm           "Commodities"     / {_fmt_set(keep_i)} /\n'
        f'   marg(comm)     "Margin comm"     / {_fmt_set(marg)} /\n'
        f'   reg            "Regions"         / {_fmt_set(keep_r)} /\n'
        f'   endw           "Endowments"      / {_fmt_set(endw)} /\n'
        f'   endwf(endw)    "Fixed factors"   / {_fmt_set(endwf)} /\n'
        f'   endwm(endw)    "Mobile factors"  / {_fmt_set(endwm)} /\n'
        f'   endws(endw)    "Sluggish factors"/ {_fmt_set(endws)} /\n'
        ";\n\n"
        "* --- Standard SAM labels ---\n"
        "set stdlab \"Standard SAM labels\" /\n"
        "   TRD, hhd, gov, inv, deprY, tmg, itax, ptax, mtax, etax, vtax, vsub, dtax, ctax, bop, tot\n"
        "/ ;\n\n"
        "set findem(stdlab) \"Final demand accounts\" / hhd, gov, inv, tmg / ;\n\n"
        "* --- is superset: acts + comm + endw + stdlab + reg ---\n"
        f'set is "SAM accounts" /\n   {_fmt_set(is_elems)}\n/ ;\n'
        "alias(is, js) ;\n\n"
        "* --- Derived sets ---\n"
        f'set aa(is) "Armington agents" / {_fmt_set(keep_a + ["hhd", "gov", "inv", "tmg"])} / ;\n'
        f'set a(aa)  "Activities"       / {_fmt_set(keep_a)} / ;\n'
        f'set i(is)  "Commodities"      / {_fmt_set(keep_i)} / ;\n'
        "alias(i, j) ;\n\n"
        f'set r(is)  "Regions" / {_fmt_set(keep_r)} / ;\n'
        "alias(r,s) ; alias(r,d) ; alias(r,rp) ;\n\n"
        f'set fp(is) "Factors"          / {_fmt_set(endw)} / ;\n'
        f'set fnm(fp) "Non-mobile"      / {_fmt_set(endwf)} / ;\n'
        "set fm(fp) \"Mobile factors\" ;\n"
        "fm(fp)$(not fnm(fp)) = yes ;\n\n"
        "set fd(aa) \"Domestic final demand agents\" / hhd, gov, inv, tmg / ;\n"
        "set h(fd)  \"Households\"   / hhd / ;\n"
        "set gov(fd) \"Government\"  / gov / ;\n"
        "set inv(fd) \"Investment\"  / inv / ;\n"
        "set fdc(fd) \"CES final demand\" / gov, inv / ;\n"
        "set tmg(fd) \"Trade margins\" / tmg / ;\n\n"
        "alias(i0,i) ; alias(a0,a) ; alias(j0,i0) ; alias(m0,i0) ;\n"
        "set mapa0(a,a0) ;\n"
        "set mapi0(i,i0) ;\n"
        "mapa0(a,a0)$(sameas(a,a0)) = yes ;\n"
        "mapi0(i,i0)$(sameas(i,i0)) = yes ;\n\n"
        "* --- Embedded FULL 9x10 data (POP renamed to pop0) ---\n"
        "$onMulti\n" + dat_dump + "\n$offMulti\n\n"
        "* --- fbep/ftrv/ptax/tvom/check derived (getData.gms section) ---\n"
        "parameter fbep(*,*,*) ; fbep(fp,a0,r) = 0 ;\n"
        "parameter ftrv(*,*,*) ; ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;\n"
        "parameter ptax(*,*,*) ; ptax(i0,a0,r) = makb(i0,a0,r) - maks(i0,a0,r) ;\n"
        "parameter tvom(a0,r)  ; tvom(a0,r) = sum(i0, vdfp(i0,a0,r) + vmfp(i0,a0,r)) + sum(fp, evfp(fp,a0,r)) ;\n"
        "parameter check(a0,r) ; check(a0,r) = 0 ;\n\n"
        "* --- CO2 emissions (zero — no emissions GDX) ---\n"
        "parameters\n"
        "   mdf(i0,a0,r), mmf(i0,a0,r), mdp(i0,r), mmp(i0,r),\n"
        "   mdg(i0,r), mmg(i0,r), mdi(i0,r), mmi(i0,r) ;\n"
        "mdf(i0,a0,r)=0; mmf(i0,a0,r)=0; mdp(i0,r)=0; mmp(i0,r)=0;\n"
        "mdg(i0,r)=0; mmg(i0,r)=0; mdi(i0,r)=0; mmi(i0,r)=0;\n\n"
        "* --- Embedded elasticities from 9x10Prm.gdx (RORFLEX renamed) ---\n"
        "$onMulti\n" + prm_dump + "\n$offMulti\n\n"
        "* --- Derived parameters (getData.gms lines 344-383) ---\n"
        "parameters\n"
        "   sigmap(r,a), sigmand(r,a), sigmav(r,a),\n"
        "   omegas(r,a), sigmas(r,i),\n"
        "   eh0(r,i), bh0(r,i),\n"
        "   sigmag(r), sigmai(r),\n"
        "   sigmam(r,i,aa), sigmaw(r,i), sigmamg(i),\n"
        "   omegaf(r,fp),\n"
        "   omegax(r,i), omegaw(r,i),\n"
        "   etaf(r,fp), etaff(r,fp,a),\n"
        "   mdtx0(r), RoRFlag ;\n\n"
        "sigmap(r,a)    = na ; sigmand(r,a)  = na ; sigmav(r,a)   = na ;\n"
        "omegas(r,a)    = -etraq(a,r) ;\n"
        "sigmas(r,i)    = inf$(esubq(i,r) eq 0) + (1/esubq(i,r))$(esubq(i,r) ne 0) ;\n"
        "eh0(r,i)       = na ; bh0(r,i)      = na ;\n"
        "sigmag(r)      = na ; sigmai(r)     = na ;\n"
        "sigmam(r,i,aa) = na ; sigmaw(r,i)   = na ; sigmamg(i)    = na ;\n"
        "loop(fp$(not fnm(fp)), loop(endwm, omegaf(r,fp)$sameas(endwm,fp) = inf ; ) ;\n"
        "                        loop(endws, omegaf(r,fp)$sameas(endws,fp) = na  ; ) ; ) ;\n"
        "parameter rorFlex(r,t) \"Flexibility of foreign capital\" ;\n"
        "rorFlex(r,t) = rorFlex0(r) ;\n"
        "omegax(r,i)    = inf ; omegaw(r,i)  = inf ;\n"
        "etaf(r,fp)     = 0 ;   etaff(r,fp,a) = 0 ;\n"
        "mdtx0(r)       = na ;\n\n"
        "* ============================================================\n"
        "* END getData.gms replacement (shrunk subset)\n"
        "* ============================================================\n"
    )


def _gdx_set_elements(gdx_path: Path, set_name: str) -> list[str]:
    """Read elements of a 1-dim Set from a GDX via gdxdump Symb=."""
    import subprocess, re as _re
    GDXDUMP = Path("/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump")
    raw = subprocess.check_output(
        [str(GDXDUMP), str(gdx_path), f"Symb={set_name}"],
        stderr=subprocess.DEVNULL, timeout=30,
    ).decode()
    return _re.findall(r"'([^']+)'", raw)


def _build_getdata_replacement_agg(gdx_path: Path, har_dir: Path | None = None) -> str:
    """getData.gms replacement for a pre-aggregated (balanced) subset GDX.

    Unlike the shrunk path (full 9x10 data + subset sets), this uses a single
    consolidated aggregated GDX (e.g. basedata-3x4.gdx from make_small_gdx) that
    contains ONLY the subset elements with summed/balanced flows — suitable for
    an actual MCP solve. Sets are read from the GDX so they match the data.
    har_dir: when given, elasticity parameters are read from default.prm via
        _prm_har_as_assignments (for GTAPAgg datasets whose GDX lacks elasticities).
    rorFlex0 is absent in the aggregate → defaulted to 0 (mirrors Python).
    """
    reg  = _gdx_set_elements(gdx_path, "REG")
    comm = _gdx_set_elements(gdx_path, "COMM")
    acts = _gdx_set_elements(gdx_path, "ACTS")
    endw = _gdx_set_elements(gdx_path, "ENDW") or ["Land", "UnSkLab", "SkLab", "Capital", "NatRes"]
    marg = _gdx_set_elements(gdx_path, "MARG")
    endwf = [f for f in endw if f == "NatRes"]
    endws = [f for f in endw if f == "Land"]
    endwm = [f for f in endw if f not in ("NatRes", "Land")]
    # Rename params that collide with comp.gms's own declarations (pop, rorFlex)
    # to the *0 baseline-input names getData expects — mirrors the full-9x10 path.
    # Use assignment-style (not $onMulti) — $onMulti + data triggers $767 in NEOS 52.5.0.
    dump = _prm_as_assignments(
        gdx_path, renames={"POP": "pop0", "RORFLEX": "rorFlex0"},
    )
    if har_dir is not None and (har_dir / "default.prm").exists():
        dump += "\n" + _prm_har_as_assignments(har_dir / "default.prm")
    _seen: set[str] = set()
    is_elems = []
    for _e in acts + comm + endw + [
        "TRD", "hhd", "gov", "inv", "deprY", "tmg", "itax", "ptax", "mtax",
        "etax", "vtax", "vsub", "dtax", "ctax", "bop", "tot",
    ] + reg:
        if _e not in _seen:
            is_elems.append(_e)
            _seen.add(_e)
    return (
        "\n* getData.gms REPLACEMENT (aggregated subset) -- from "
        f"{gdx_path.name}\n"
        "sets\n"
        f'   acts           "Activities"      / {_fmt_set(acts)} /\n'
        f'   comm           "Commodities"     / {_fmt_set(comm)} /\n'
        f'   marg(comm)     "Margin comm"     / {_fmt_set(marg)} /\n'
        f'   reg            "Regions"         / {_fmt_set(reg)} /\n'
        f'   endw           "Endowments"      / {_fmt_set(endw)} /\n'
        f'   endwf(endw)    "Fixed factors"   / {_fmt_set(endwf)} /\n'
        f'   endwm(endw)    "Mobile factors"  / {_fmt_set(endwm)} /\n'
        f'   endws(endw)    "Sluggish factors"/ {_fmt_set(endws)} /\n'
        ";\n\n"
        "set stdlab \"Standard SAM labels\" /\n"
        "   TRD, hhd, gov, inv, deprY, tmg, itax, ptax, mtax, etax, vtax, vsub, dtax, ctax, bop, tot\n"
        "/ ;\n\n"
        "set findem(stdlab) \"Final demand accounts\" / hhd, gov, inv, tmg / ;\n\n"
        f'set is "SAM accounts" /\n   {_fmt_set(is_elems)}\n/ ;\n'
        "alias(is, js) ;\n\n"
        f'set aa(is) "Armington agents" / {_fmt_set(acts + ["hhd", "gov", "inv", "tmg"])} / ;\n'
        f'set a(aa)  "Activities"       / {_fmt_set(acts)} / ;\n'
        f'set i(is)  "Commodities"      / {_fmt_set(comm)} / ;\n'
        "alias(i, j) ;\n\n"
        f'set r(is)  "Regions" / {_fmt_set(reg)} / ;\n'
        "alias(r,s) ; alias(r,d) ; alias(r,rp) ;\n\n"
        f'set fp(is) "Factors"          / {_fmt_set(endw)} / ;\n'
        f'set fnm(fp) "Non-mobile"      / {_fmt_set(endwf)} / ;\n'
        "set fm(fp) \"Mobile factors\" ;\n"
        "fm(fp)$(not fnm(fp)) = yes ;\n\n"
        "set fd(aa) \"Domestic final demand agents\" / hhd, gov, inv, tmg / ;\n"
        "set h(fd)  \"Households\"   / hhd / ;\n"
        "set gov(fd) \"Government\"  / gov / ;\n"
        "set inv(fd) \"Investment\"  / inv / ;\n"
        "set fdc(fd) \"CES final demand\" / gov, inv / ;\n"
        "set tmg(fd) \"Trade margins\" / tmg / ;\n\n"
        "alias(i0,i) ; alias(a0,a) ; alias(j0,i0) ; alias(m0,i0) ;\n"
        "set mapa0(a,a0) ; set mapi0(i,i0) ;\n"
        "mapa0(a,a0)$(sameas(a,a0)) = yes ;\n"
        "mapi0(i,i0)$(sameas(i,i0)) = yes ;\n\n"
        "$onImplicitAssign\n"
        + dump + "\n"
        "parameter fbep(*,*,*) ; fbep(fp,a0,r) = 0 ;\n"
        "parameter ftrv(*,*,*) ; ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;\n"
        "parameter ptax(*,*,*) ; ptax(i0,a0,r) = makb(i0,a0,r) - maks(i0,a0,r) ;\n"
        "parameter tvom(a0,r)  ; tvom(a0,r) = sum(i0, vdfp(i0,a0,r) + vmfp(i0,a0,r)) + sum(fp, evfp(fp,a0,r)) ;\n"
        "parameter check(a0,r) ; check(a0,r) = 0 ;\n\n"
        "parameters mdf(i0,a0,r), mmf(i0,a0,r), mdp(i0,r), mmp(i0,r),\n"
        "   mdg(i0,r), mmg(i0,r), mdi(i0,r), mmi(i0,r) ;\n"
        "mdf(i0,a0,r)=0; mmf(i0,a0,r)=0; mdp(i0,r)=0; mmp(i0,r)=0;\n"
        "mdg(i0,r)=0; mmg(i0,r)=0; mdi(i0,r)=0; mmi(i0,r)=0;\n\n"
        "parameters sigmap(r,a), sigmand(r,a), sigmav(r,a), omegas(r,a), sigmas(r,i),\n"
        "   eh0(r,i), bh0(r,i), sigmag(r), sigmai(r), sigmam(r,i,aa), sigmaw(r,i),\n"
        "   sigmamg(i), omegaf(r,fp), omegax(r,i), omegaw(r,i), etaf(r,fp),\n"
        "   etaff(r,fp,a), mdtx0(r), RoRFlag ;\n"
        "sigmap(r,a)=na; sigmand(r,a)=na; sigmav(r,a)=na;\n"
        "omegas(r,a) = -etraq(a,r) ;\n"
        "sigmas(r,i) = inf$(esubq(i,r) eq 0) + (1/esubq(i,r))$(esubq(i,r) ne 0) ;\n"
        "eh0(r,i)=na; bh0(r,i)=na; sigmag(r)=na; sigmai(r)=na;\n"
        "sigmam(r,i,aa)=na; sigmaw(r,i)=na; sigmamg(i)=na;\n"
        "loop(fp$(not fnm(fp)), loop(endwm, omegaf(r,fp)$sameas(endwm,fp) = inf ; ) ;\n"
        "                        loop(endws, omegaf(r,fp)$sameas(endws,fp) = na  ; ) ; ) ;\n"
        # rorFlex0 comes from the dump (RORFLEX rename) when the GDX has it; declare
        # it (*) WITHOUT overwriting so consolidated data is preserved and absent
        # datasets default to 0. Same (*) domain as the dump avoids $184 redefine.
        "parameter rorFlex0(*) ;\n"
        "parameter rorFlex(r,t) ; rorFlex(r,t) = rorFlex0(r) ;\n"
        "omegax(r,i)=inf; omegaw(r,i)=inf; etaf(r,fp)=0; etaff(r,fp,a)=0; mdtx0(r)=na;\n"
        "* END getData.gms replacement (aggregated subset)\n"
    )


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


def _build_standalone_gms(phase: str = "base", shrink: bool = False,
                          agg_gdx: Path | None = None,
                          har_dir: Path | None = None) -> str:
    """Build a self-contained .gms for NEOS submission.

    Inlines all $include files. Replaces getData.gms with embedded sets + data
    from gdxdump (making the GMS fully self-contained). Uses NLP=CONVERT so GAMS
    writes .nl instead of solving.

    getData source (precedence):
      - agg_gdx given → `_build_getdata_replacement_agg(agg_gdx)`: SETS-AGNOSTIC,
        reads sets from the aggregated/consolidated GDX (any GTAPAgg dataset) and
        restricts rres/rmuv/imuv to those elements. This is what lets nl_compare
        run on NEOS for arbitrary datasets.
      - shrink → hardcoded full-9x10 sets shrunk to KEEP_R x KEEP_I.
      - else → hardcoded full 9x10.

    phase: "base" → ts('base') only; "shock" → apply tm*1.1, solve shock period.
    """
    scripts_dir = COMP_GMS.parent
    import re
    if agg_gdx is not None:
        getdata_replacement = _build_getdata_replacement_agg(agg_gdx, har_dir=har_dir)
    elif shrink:
        getdata_replacement = _build_getdata_replacement_shrunk(KEEP_R, KEEP_I)
    else:
        getdata_replacement = _build_getdata_replacement()

    def _inline_includes(text: str, depth: int = 0) -> str:
        if depth > 5:
            return text

        def _replace(m):
            fname = m.group(1).strip('"\'')
            if fname == "getData.gms":
                return getdata_replacement
            fpath = scripts_dir / fname
            if fpath.exists():
                inner = fpath.read_text()
                return (f"* --- inlined: {fname} ---\n"
                        f"{_inline_includes(inner, depth+1)}\n"
                        f"* --- end: {fname} ---")
            return m.group(0)

        def _replace_bat(m):
            fname = m.group(1).strip('"\'')
            args_str = m.group(2).strip() if m.group(2) else ""
            args = args_str.split() if args_str else []
            fpath = scripts_dir / fname
            if fpath.exists():
                inner = fpath.read_text()
                for idx, arg in enumerate(args, start=1):
                    inner = inner.replace(f"%{idx}", arg)
                args_label = " ".join(args) if args else ""
                return (f"* --- batinlined: {fname} ---\n"
                        f"{_inline_includes(inner, depth+1)}\n"
                        f"* --- end bat: {fname} --- {args_label}")
            return m.group(0)

        text = re.sub(r'\$\$?include\s+("?[^"\s]+"?)', _replace, text, flags=re.IGNORECASE)
        text = re.sub(r'\$\$?batinclude\s+("?[^"\s]+"?)((?:\s+\S+)*)', _replace_bat, text, flags=re.IGNORECASE)
        return text

    comp_text = COMP_GMS.read_text()
    inlined = _inline_includes(comp_text)

    # comp.gms hardcodes rres/rmuv/imuv for the 9x10 aggregation; any element
    # outside the active set is a $170 domain violation that aborts CONVERT.
    # Restrict all three to the active subset so they stay valid and eq_pmuv is
    # comparable on both sides. For agg_gdx, derive the subset from the GDX sets
    # (sets-agnostic); for shrink, from KEEP_R x KEEP_I.
    _restrict = None
    if agg_gdx is not None:
        _restrict = _muv_baskets(_gdx_set_elements(agg_gdx, "REG"),
                                 _gdx_set_elements(agg_gdx, "COMM"))
    elif shrink:
        _restrict = _muv_baskets(KEEP_R, KEEP_I)
    if _restrict is not None:
        rres, rmuv, imuv = _restrict
        for sname, dom, elems in (
            ("rres", "r", (rres,)),
            ("rmuv", "r", rmuv),
            ("imuv", "i", imuv),
        ):
            inlined = re.sub(
                rf'set\s+{sname}\({dom}\)\s*"[^"]*"\s*/\s*[^/]*?/\s*;',
                f'set {sname}({dom}) "{sname}" /\n   {", ".join(elems)}\n/ ;',
                inlined, flags=re.IGNORECASE,
            )

    # Truncate at the simulation loop and inject a solve block.
    cut_marker = "Run the simulations for each time period"
    cut_idx = inlined.find(cut_marker)
    if cut_idx != -1:
        block_start = inlined.rfind("* ---", 0, cut_idx)
        if block_start == -1:
            block_start = cut_idx - 200
        before_loop = inlined[:block_start]
    else:
        before_loop = inlined

    if phase == "shock":
        solve_block = """
* ============================================================
* CONVERT-mode solve (SHOCK period): write gtap.nl — Layer 6 diagnostic
* Use NLP mode (ifMCP=0) — GAMS CONVERT cannot write .nl for MCP.
* ============================================================
rs(r) = yes ;
ts('base') = yes ;
ts('shock') = yes ;

* Solve base first to initialize variables
option limrow = 0, limcol = 0, solprint = off ;
option NLP = CONVERT;
gtap.optfile = 1 ;
solve gtap using nlp maximizing walras ;

* Apply 10% import tariff shock (mirror Python: imptx_shock = imptx_base * 1.10)
imptx.fx(r,i,rp,'shock') = imptx.l(r,i,rp,'base') * 1.10 ;

* Write shock .nl via second CONVERT solve
solve gtap using nlp maximizing walras ;
"""
    elif phase == "check":
        solve_block = """
* ============================================================
* CONVERT-mode solve (CHECK period): write gtap.nl — Layer 6 diagnostic
* Altertax check period: CD elasticities + mobile factors, NO tm shock.
* This is the compStat betaCal→check transition: same closure as altertax
* but tariffs unchanged (base period values). Mirrors Python altertax_check.
* Use NLP mode (ifMCP=0) — GAMS CONVERT cannot write .nl for MCP.
* ============================================================
rs(r) = yes ;
ts('base') = yes ;
ts('check') = yes ;

* Apply altertax elasticity overrides (parameter_altertax.gms equivalent)
esubva(a,r)  = 1 ;
esubd(i,r)   = 0.95 ;
esubm(i,r)   = 0.95 ;
etrae(fp,r)  = 1 ;
omegaf(r,fp) = 1 ;

* Recalibrate derived elasticities from the overrides
sigmav(r,a)$(va.l(r,a,'base')) = 1 ;
sigmam(r,i,aa)$(xa.l(r,i,aa,'base')) = 0.95 ;
sigmaw(r,i)$(xmt.l(r,i,'base')) = 0.95 ;

* Solve base first, then write check .nl (no tariff shock applied)
option limrow = 0, limcol = 0, solprint = off ;
option NLP = CONVERT;
gtap.optfile = 1 ;
solve gtap using nlp maximizing walras ;
"""
    elif phase == "altertax":
        solve_block = """
* ============================================================
* CONVERT-mode solve (ALTERTAX period): write gtap.nl — Layer 6 diagnostic
* Mirrors Malcolm (1998) altertax: CD elasticities (all subst. elast. = 1),
* mobile factors (omegaf=inf or large), Armington 0.95.
* Signatures from standard_gtap_7/cal.gms:
*   esubva(a,r), esubd(i,r), esubm(i,r), etrae(fp,r), omegaf(r,fp).
* Use NLP mode (ifMCP=0) — GAMS CONVERT cannot write .nl for MCP.
* ============================================================
rs(r) = yes ;
ts('base') = yes ;

* Apply altertax elasticity overrides (parameter_altertax.gms equivalent)
esubva(a,r)  = 1 ;
esubd(i,r)   = 0.95 ;
esubm(i,r)   = 0.95 ;
etrae(fp,r)  = 1 ;
omegaf(r,fp) = 1 ;

* Recalibrate derived elasticities from the overrides (mirrors cal.gms flow)
sigmav(r,a)$(va.l(r,a,'base')) = 1 ;
sigmam(r,i,aa)$(xa.l(r,i,aa,'base')) = 0.95 ;
sigmaw(r,i)$(xmt.l(r,i,'base')) = 0.95 ;

option limrow = 0, limcol = 0, solprint = off ;
option NLP = CONVERT;
gtap.optfile = 1 ;
solve gtap using nlp maximizing walras ;
"""
    else:
        solve_block = """
* ============================================================
* CONVERT-mode solve (BASE period): write gtap.nl — Layer 6 diagnostic
* Use NLP mode (ifMCP=0) — GAMS CONVERT cannot write .nl for MCP.
* ============================================================
rs(r) = yes ;
ts('base') = yes ;

option limrow = 0, limcol = 0, solprint = off ;
option NLP = CONVERT;
gtap.optfile = 1 ;
solve gtap using nlp maximizing walras ;
"""

    header = """$setGlobal simType   CompStat
$setGlobal simName   COMP
$setGlobal baseName  9x10
$setGlobal inDir     .
$setGlobal outDir    .
$setGlobal utility   cde
$setGlobal savfFlag  capFix
$setGlobal ifCal     0
$setGlobal ifSUB     0
$setGlobal ifCSV     0
$setGlobal ifMCP     0
"""
    return header + before_loop + solve_block


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
) -> dict[str, Path]:
    """Submit GTAP model to NEOS CONVERT for each requested phase.

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
        standalone_gms = _build_standalone_gms(phase=phase, shrink=shrink, agg_gdx=agg_gdx, har_dir=har_dir)
        standalone_path = out_dir / f"gtap_standalone_{phase}.gms"
        standalone_path.write_text(standalone_gms)
        print(f"  Standalone .gms: {len(standalone_gms.splitlines())} lines ({len(standalone_gms)//1024} KB)")
        result[phase] = _submit_and_fetch_nl(standalone_gms, out_dir, neos_email, phase)

    return result


# ---------------------------------------------------------------------------
# Phase 3: Diff
# ---------------------------------------------------------------------------

_T_PERIODS = ("base", "check", "shock")


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

    return name


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


def diff_nl_models(
    py_model: "NLModel",
    gams_model: "NLModel",
    tol_rel: float = 1e-4,
    py_period: str = "base",
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

    # GAMS names: convert to Python convention, keep base period only
    gams_var_map: dict[str, int] = {}
    for i in range(gams_model.n_vars):
        raw = gams_model.var_name(i)
        norm = _normalize_name(raw)  # strips time, unifies brackets
        if norm and norm not in gams_var_map:
            gams_var_map[norm] = i

    gams_con_map: dict[str, int] = {}
    for i in range(gams_model.n_cons):
        raw = gams_model.con_name(i)
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

    # Coefficient diff by constraint family (only constraints common to both)
    family_stats: dict[str, dict] = {}
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

    return {
        "structural": structural,
        "only_py_vars": only_py_vars,
        "only_gams_vars": only_gams_vars,
        "only_py_cons": only_py_cons,
        "only_gams_cons": only_gams_cons,
        "family_stats": family_stats,
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

def main() -> None:
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
    args = ap.parse_args()

    if args.fetch_gdx_9x10:
        out = ROOT / "output" / "gtap9x10_neos"
        gdx = submit_and_fetch_gdx_9x10(out_dir=out, neos_email=args.neos_email)
        print(f"Got: {gdx}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    global KEEP_R, KEEP_I
    if args.keep_r:
        KEEP_R = list(args.keep_r)
    if args.keep_i:
        KEEP_I = list(args.keep_i)
    custom_subset = bool(args.keep_r or args.keep_i)

    full = args.full_9x10
    gams_shrink = (args.gams_shrink or custom_subset) and not full
    if full:
        print("[full-9x10] BOTH sides on full 9x10 (no shrink)")
    elif gams_shrink:
        print(f"[gams-shrink] GAMS sets shrunk to KEEP_R={KEEP_R} KEEP_I={KEEP_I} "
              f"(matches Python _shrink_sets)")

    from run_gtap import _build_gtap_contract_with_calibration
    from _parity_datasets import DATASETS
    ds = DATASETS.get(args.dataset) or DATASETS["9x10"]

    # GTAPAgg dataset (comp-stat with a consolidated GDX): run BOTH sides on the
    # dataset's OWN sets, sets-agnostic. GAMS getData + Python load both come from
    # ds.agg_gdx; rres/rmuv/imuv are derived from the GDX. No 9x10 contract / shrink.
    use_agg = ds.mode == "compstat" and ds.agg_gdx is not None
    if use_agg:
        from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
        _reg = _gdx_set_elements(ds.agg_gdx, "REG")
        _comm = _gdx_set_elements(ds.agg_gdx, "COMM")
        _rres, _rmuv, _imuv = _muv_baskets(_reg, _comm)
        closure_config = GTAPClosureConfig(if_sub=False, rmuv=_rmuv, imuv=_imuv)
        full, gams_shrink = True, False
        print(f"[agg] dataset={args.dataset} from {ds.agg_gdx.name} "
              f"({len(_reg)}reg×{len(_comm)}comm, sets-agnostic NEOS)")
    else:
        # 9x10 family. "3x3" maps to the 9x10 GDX shrunk via --gams-shrink/--keep.
        contract = _build_gtap_contract_with_calibration(ds.nl_contract)
        closure_update = {"if_sub": False}
        if full:
            closure_update["rmuv"], closure_update["imuv"] = ds.nl_full_muv
        elif gams_shrink:
            _rres, _rmuv, _imuv = _muv_baskets(KEEP_R, KEEP_I)
            closure_update["rmuv"] = _rmuv
            closure_update["imuv"] = _imuv
        closure_config = contract.closure.model_copy(update=closure_update)

    # Phase 1: Python → .nl  (_shrink_sets to KEEP_R x KEEP_I unless --full-9x10)
    # compstat datasets (gtap7_*) have HAR files → load_from_har via har_dir.
    _har_dir = (ROOT / "datasets" / ds.name) if use_agg else None
    if not args.skip_python:
        py_paths = build_python_nls(
            args.phase, args.out_dir, closure_config,
            gdx_path=(None if use_agg else ds.agg_gdx),
            do_shrink=not full,
            har_dir=_har_dir)
    else:
        py_paths = {p: args.out_dir / f"python_{p}.nl" for p in args.phase}
        print(f"\n[skip-python] Using existing Python .nl files in {args.out_dir}")

    # Phase 2: GAMS → .nl via NEOS
    gams_nl_paths: dict[str, Path] = {}
    gams_phases = list(args.phase)
    if not args.skip_gams:
        gams_nl_paths = fetch_gams_nl(
            args.out_dir, args.neos_email, gams_phases, shrink=gams_shrink,
            agg_gdx=(ds.agg_gdx if use_agg else None),
            har_dir=_har_dir)
    else:
        for phase in gams_phases:
            # Support both old (gams_compstat.nl) and new (gams_base.nl / gams_shock.nl) names
            candidates = [
                args.out_dir / f"gams_{phase}.nl",
                args.out_dir / "gams_compstat.nl",  # legacy fallback for base
            ]
            for p in candidates:
                if p.exists():
                    gams_nl_paths[phase] = p
                    print(f"\n[skip-gams] Using existing {p.name} for {phase}")
                    break
            else:
                print(f"\n[skip-gams] WARNING: no gams .nl for {phase} — skipping diff")

    # Phase 3: Diff (per-phase, each Python .nl vs its GAMS counterpart)
    from _nl_parser import parse_nl
    print("\n=== Phase 3: Diff ===")
    for period in args.phase:
        py_nl = py_paths.get(period)
        if py_nl is None or not Path(py_nl).exists():
            print(f"  [{period}] Python .nl not found — skipping")
            continue
        gams_nl = gams_nl_paths.get(period)
        if gams_nl is None or not gams_nl.exists():
            print(f"  [{period}] GAMS .nl not found — skipping diff (Python .nl written)")
            continue
        py_m = parse_nl(py_nl)
        gams_m = parse_nl(gams_nl)
        print(f"  GAMS ({period}): {gams_m.n_vars} vars, {gams_m.n_cons} cons, {gams_m.n_nonzeros} nnz")
        result = diff_nl_models(py_m, gams_m, tol_rel=args.tol_rel, py_period=period)
        print_diff_report(result, period, tol_rel=args.tol_rel)

        # Bounds diff: reveals iterloop.gms lower-bound differences
        b_diffs, only_py_vars, only_gams_vars = diff_bounds(py_m, gams_m, tol_rel=args.tol_rel)
        print(f"\n  Bounds diff [{period}]: {len(b_diffs)} variables with different lb/ub")
        if b_diffs:
            print(f"  {'variable':<50s} {'py_lb':>12s} {'g_lb':>12s} {'py_ub':>12s} {'g_ub':>12s}  flags")
            print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*12} {'-'*12}  -----")
            _INF = 1e30
            def _fmt(v): return f"{v:.4e}" if abs(v) < _INF/2 else ("-inf" if v < 0 else "+inf")
            for d in b_diffs[:60]:
                flags = ("LB " if d["lb_diff"] else "   ") + ("UB" if d["ub_diff"] else "  ")
                print(f"  {d['var']:<50s} {_fmt(d['py_lb']):>12s} {_fmt(d['g_lb']):>12s} "
                      f"{_fmt(d['py_ub']):>12s} {_fmt(d['g_ub']):>12s}  {flags}")
            if len(b_diffs) > 60:
                print(f"  ... {len(b_diffs)-60} more")


if __name__ == "__main__":
    main()
