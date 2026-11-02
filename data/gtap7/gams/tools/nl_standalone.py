"""Build the standalone GAMS .gms that nl_compare and regenerate_gams run.

The GAMS model (comp.gms and its includes) with the data inlined, so CONVERT
or NEOS can run it with nothing else. Moved out of scripts/gtap/nl_compare.py
(phase 2b): it generates results, so it lives with the model it reads and
the model-folder hash every result records covers it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from equilibria.sources import DATA_ROOT, GAMS_MODEL_DIR

COMP_GMS = GAMS_MODEL_DIR / "comp.gms"
DATA_DIR = DATA_ROOT / "gtap7" / "9x10"

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


def _gdxdump(gdx_path: Path, *args: str) -> str:
    """gdxdump's text for gdx_path; an error, never a placeholder, when it can't run.

    The dumps below are the DATA of the standalone .gms: a comment in their place
    let GAMS run on the model's defaults without saying so.
    """
    exe = shutil.which("gdxdump")
    if exe is None:
        raise RuntimeError("gdxdump not on PATH (it ships with GAMS)")
    if not gdx_path.exists():
        raise FileNotFoundError(gdx_path)
    return subprocess.check_output(
        [exe, str(gdx_path), *args], stderr=subprocess.DEVNULL, timeout=30
    ).decode()


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
    import re as _re
    raw = _gdxdump(gdx_path)

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


# GEMPACK set name -> GAMS element prefix. WHETHER an inlined .gms's comm/
# acts sets carry a domain prefix on their elements (c_Agri/a_Agri) or not
# (bare Agri) is DATASET-SPECIFIC, not universal — confirmed empirically:
# gtap7_5x5's inlined `Set comm / c_Agri, ... /` carries the prefix, but
# gtap7_15x10's inlined `Set comm / Rice, ... /` does NOT (bare element
# names, even though comm and acts share the same underlying commodity
# labels there — the disambiguation lives in the i0/i alias domain
# mechanism instead, not in the element name). Passing the WRONG choice
# hard-fails GAMS Error 170 (domain violation) at compile time — confirmed
# both ways: unprefixed keys against 5x5's prefixed set, and prefixed keys
# against 15x10's bare set. There is no way to know which convention a
# given comp.gms uses without inspecting it — callers MUST pass
# `use_prefix=` explicitly (detected via _detect_prm_prefix_convention
# below), do not assume True or False.
_PRM_SET_PREFIX = {"COMM": "c_", "ACTS": "a_", "MARG": "c_"}


def _detect_prm_prefix_convention(comp_gms_text: str) -> bool:
    """Return True if this comp.gms's inlined comm/acts sets carry a c_/a_
    prefix on their elements, False if they use bare names.

    Checks the `Set comm 'Set COMM ...' / ... /` block emitted by the GDX
    inliner (gdx_to_gams_inline.py) for the FIRST element's prefix — cheap
    and reliable since a dataset is internally consistent (either every
    comm/acts element is prefixed or none are).
    """
    m = re.search(r"Set\s+comm\b[^/]*/\s*([A-Za-z0-9_]+)", comp_gms_text)
    if not m:
        return False
    return m.group(1).startswith("c_")


def _prm_har_as_assignments(prm_path: Path, use_prefix: bool = True) -> str:
    """Emit GAMS parameter assignments from a GEMPACK default.prm file.

    Reads elasticity headers (ESBD→esubd, ETRQ→etraq, etc.) and emits
    plain GAMS assignment statements. Only emits headers in _PRM_HAR_NAME_MAP.

    `use_prefix` must reflect the TARGET comp.gms's own comm/acts set
    convention (see _detect_prm_prefix_convention) — it is dataset-specific,
    not universal. Defaults to True (the historical/5x5 convention) only for
    backward compatibility with existing callers; new callers should always
    pass it explicitly.
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
        set_names = getattr(h, "set_names", None) or [None] * dims
        prefixes = [_PRM_SET_PREFIX.get(sn, "") if use_prefix else "" for sn in set_names]
        it = np.nditer(h.array, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            val = float(it[0])
            if val != 0.0:
                keys_fmt = ",".join(
                    f'"{prefixes[d]}{label_lists[d][i]}"' for d, i in enumerate(idx)
                )
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
    import re as _re
    raw = _gdxdump(gdx_path)

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
    import re as _re
    raw = _gdxdump(DATA_DIR / "9x10Dat.gdx", "Symb=MAKB")
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
    import re as _re
    raw = _gdxdump(gdx_path, f"Symb={set_name}")
    return _re.findall(r"'([^']+)'", raw)


def _fbep_header_block(fbep: str, har_dir: Path | None, dump: str) -> str:
    """The FBEP/FTRV assignments from basedata.har when fbep is "header", else ""."""
    if fbep == "zero":
        return ""
    if fbep != "header":
        raise ValueError(f"fbep must be 'header' or 'zero', not {fbep!r}")
    if har_dir is None:
        raise ValueError("fbep='header' needs har_dir (the basedata.har it injects)")
    from build_gtap7_altertax_neos_bundle import negate_fbep_assignments
    from build_gtap7_pure_local_bundle import _fbep_ftrv_har_as_assignments

    blocks = [
        negate_fbep_assignments(
            _fbep_ftrv_har_as_assignments(har_dir / "basedata.har", sym, dump)
        )
        for sym in ("FBEP", "FTRV")
    ]
    from build_gtap7_altertax_neos_bundle import ftrv_zero_where_only_subsidised

    blocks.append(ftrv_zero_where_only_subsidised(har_dir / "basedata.har", dump))
    return "".join(b + "\n" for b in blocks if b.strip())


def _build_getdata_replacement_agg(
    gdx_path: Path, har_dir: Path | None = None, fbep: str = "zero"
) -> str:
    """getData.gms replacement for a pre-aggregated (balanced) subset GDX.

    Unlike the shrunk path (full 9x10 data + subset sets), this uses a single
    consolidated aggregated GDX (e.g. basedata-3x4.gdx from make_small_gdx) that
    contains ONLY the subset elements with summed/balanced flows — suitable for
    an actual MCP solve. Sets are read from the GDX so they match the data.
    har_dir: when given, elasticity parameters are read from default.prm via
        _prm_har_as_assignments (for GTAPAgg datasets whose GDX lacks elasticities).
    rorFlex0 is absent in the aggregate → defaulted to 0 (mirrors Python).
    fbep: "zero" keeps getData.gms's fbep = 0 and ftrv = evfp - evfb; "header"
        injects har_dir/basedata.har's FBEP/FTRV right after them, as the bundle
        builders do (#187).
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
        # use_prefix DEBE reflejar la convencion de ESTE dataset, no un default.
        # Si/no llevan prefijo c_/a_ los elementos de comm/acts es dataset-
        # specific (ver _PRM_SET_PREFIX): gtap7_5x5 declara `c_Agri`, gtap7_15x10
        # declara `Rice` a pelo, y ambos son correctos.  El GDX de donde salen los
        # sets es la fuente mas directa de esa convencion, asi que la leemos de
        # ahi en vez de asumir.
        #
        # MEDIDO: sin esto, el default True inyectaba claves `c_Food` contra un
        # set declarado `Food` en gtap7_3x4 y gtap7_15x10 -> las claves no casan,
        # el denominador `sum(i0$mapi0, vdpp+vmpp)` suma 0 y GAMS aborta con
        # "division by zero" en la calibracion CDE (eh0/bh0) y en sigmaq.  Los 2
        # datasets que fallaban eran EXACTAMENTE los 2 sin prefijo; los 3 que
        # funcionaban, los 3 con prefijo.  El docstring de
        # _prm_har_as_assignments ya avisaba: "new callers should always pass it
        # explicitly" — esta era la llamada que no lo hacia.
        _use_prefix = bool(comm) and str(comm[0]).startswith("c_")
        dump += "\n" + _prm_har_as_assignments(
            har_dir / "default.prm", use_prefix=_use_prefix
        )
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
        + _fbep_header_block(fbep, har_dir, dump)
        + "parameter ptax(*,*,*) ; ptax(i0,a0,r) = makb(i0,a0,r) - maks(i0,a0,r) ;\n"
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


def _build_standalone_gms(phase: str = "base", shrink: bool = False,
                          agg_gdx: Path | None = None,
                          har_dir: Path | None = None,
                          ifsub: int = 0, fbep: str = "zero") -> str:
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
        getdata_replacement = _build_getdata_replacement_agg(agg_gdx, har_dir=har_dir, fbep=fbep)
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
    # comp.gms sets ifSUB itself and GAMS keeps the last $setGlobal, so the
    # header alone would be overridden (#23): every setting gets the value asked.
    return re.sub(r"^\$setGlobal\s+ifSUB\s+\d+\s*$", f"$setGlobal ifSUB     {ifsub}",
                  header + before_loop + solve_block, flags=re.MULTILINE)


def _convert_locally(standalone_gms: str, out_dir: Path, phase: str) -> Path:
    """Run the standalone .gms through the local GAMS CONVERT (no size limit).

    Leaves exactly gams_<phase>.{nl,row,col} and gams_<phase>.gms (the file
    CONVERT read) in out_dir; raises if GAMS fails or writes no .nl.
    """
    gams = shutil.which("gams")
    if gams is None:
        raise RuntimeError("--gams-local needs the gams binary on PATH")
    work = out_dir / f"_convert_{phase}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / "comp.gms").write_text(standalone_gms)
    (work / "convert.opt").write_text("AmplNL ampl.nl\n")
    proc = subprocess.run([gams, "comp.gms", "lo=2"], cwd=work, capture_output=True, text=True)
    nl = work / "ampl.nl"
    if proc.returncode != 0 or not nl.exists():
        raise RuntimeError(
            f"local CONVERT ({phase}) failed (rc={proc.returncode}); see {work / 'comp.lst'}"
        )
    shutil.copyfile(work / "comp.gms", out_dir / f"gams_{phase}.gms")
    for ext in ("nl", "row", "col"):
        if (work / f"ampl.{ext}").exists():
            shutil.copyfile(work / f"ampl.{ext}", out_dir / f"gams_{phase}.{ext}")
    shutil.rmtree(work)
    print(f"  Written: gams_{phase}.nl (local CONVERT)")
    return out_dir / f"gams_{phase}.nl"


def generate_nl(
    dataset: str, phases: list[str], ifsub: int, out_dir: Path, fbep: str = "zero"
) -> dict[str, Path]:
    """The GAMS .nl of each phase of ``dataset``, by local CONVERT, into out_dir.

    What ``nl_compare.py --gams-local --skip-python`` did for a gtap7 dataset:
    the aggregated v7_consolidated.gdx plus the dataset's HAR files.
    """
    from equilibria.sources import dataset_dir

    har_dir = dataset_dir(dataset)
    agg_gdx = har_dir / "v7_consolidated.gdx"
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        phase: _convert_locally(
            _build_standalone_gms(
                phase=phase, agg_gdx=agg_gdx, har_dir=har_dir, ifsub=ifsub, fbep=fbep
            ),
            out_dir,
            phase,
        )
        for phase in phases
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=generate_nl.__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--phase", nargs="+", required=True)
    ap.add_argument("--ifsub", type=int, choices=(0, 1), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fbep", choices=("header", "zero"), default="zero",
                    help="zero: official getData (default); header: inject basedata.har's "
                         "FBEP/FTRV like the bundle builders (#187)")
    args = ap.parse_args(argv)
    generate_nl(args.dataset, list(args.phase), args.ifsub, args.out_dir, args.fbep)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
