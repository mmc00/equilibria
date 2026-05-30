"""HAR → GDX bridge.

Reads one or more GEMPACK .har files and emits one or three GDX files using
the native writer.

- `write_har_to_gdx(hars, out)` — single GDX with all headers as sets/params.
- `write_nus333_gdx_bundle(har_dir, out_dir)` — emits the three files that the
  reference GAMS scripts (getData.gms) expect:
      <base>Sets.gdx, <base>Dat.gdx, <base>Prm.gdx
  with HAR-uppercase names mapped to GAMS-lowercase, and parent-set membership
  declared for the MARG/ENDM/ENDF/ENDS subsets.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from equilibria.babel.gdx.symbols import Parameter, Set, SymbolBase
from equilibria.babel.gdx.writer import write_gdx
from equilibria.babel.har.reader import read_har
from equilibria.babel.har.symbols import HeaderArray


# ── single-file API (kept stable for prior callers) ──────────────────────────


def har_to_symbols(har_paths: list[str | Path]) -> list[SymbolBase]:
    headers: dict[str, HeaderArray] = {}
    for p in har_paths:
        for name, h in read_har(str(p)).items():
            headers[name] = h

    sets: list[Set] = []
    params: list[Parameter] = []

    for name, h in headers.items():
        if _is_set_header(h):
            sets.append(_set_from_header(h))
        elif h.array.ndim >= 1 and not h.set_names:
            continue
        else:
            params.append(_parameter_from_header(h))

    return [*sets, *params]


def write_har_to_gdx(har_paths: list[str | Path], out_path: str | Path) -> None:
    write_gdx(out_path, har_to_symbols(har_paths))


# ── NUS333-style three-file bundle for GTAP getData.gms ──────────────────────


# HAR header name (uppercase) → GAMS symbol name (lowercase) used by getData.gms.
DAT_NAME_MAP: dict[str, str] = {
    "VDFB": "vdfb", "VDFP": "vdfp", "VMFB": "vmfb", "VMFP": "vmfp",
    "VDPB": "vdpb", "VDPP": "vdpp", "VMPB": "vmpb", "VMPP": "vmpp",
    "VDGB": "vdgb", "VDGP": "vdgp", "VMGB": "vmgb", "VMGP": "vmgp",
    "VDIB": "vdib", "VDIP": "vdip", "VMIB": "vmib", "VMIP": "vmip",
    "EVFB": "evfb", "EVFP": "evfp", "EVOS": "evos",
    "VXSB": "vxsb", "VFOB": "vfob", "VCIF": "vcif", "VMSB": "vmsb",
    "VST": "vst", "VTWR": "vtwr",
    "SAVE": "save", "VDEP": "vdep", "VKB": "vkb", "POP": "pop",
    "MAKS": "maks", "MAKB": "makb",
}

PRM_NAME_MAP: dict[str, str] = {
    "ESBT": "esubt", "ESBC": "esubc", "ESBV": "esubva",
    "ETRQ": "etraq", "ESBQ": "esubq",
    "INCP": "incpar", "SUBP": "subpar",
    "ESBG": "esubg", "ESBI": "esubi",
    "ESBD": "esubd", "ESBM": "esubm", "ESBS": "esubs",
    "ETRE": "etrae", "RFLX": "rorFlex",
}

# Parent set for HAR subset headers (1D string arrays). When a HAR subset has
# elements that are themselves elements of a parent set, GAMS represents it as
# `set sub(parent) /elem1, elem2/`. In our model: MARG ⊂ COMM; ENDM/ENDF/ENDS ⊂ ENDW.
SUBSET_PARENT: dict[str, str] = {
    "MARG": "comm",
    "ENDM": "endw",
    "ENDF": "endw",
    "ENDS": "endw",
}

# HAR set name (uppercase) → GAMS set name (lowercase)
SET_NAME_MAP: dict[str, str] = {
    "ACTS": "acts",
    "COMM": "comm",
    "MARG": "marg",
    "REG": "reg",
    "ENDW": "endw",
    "ENDM": "endwm",
    "ENDF": "endwf",
    "ENDS": "endws",
}

# GTAP convention used by 9x10 and getData.gms: prefix acts/comm element labels
# so they don't collide in shared UEL space (NUS333 has identical acts and comm
# names: AGR/MFG/SER). Without this, `set is / set.acts set.comm /` redefines.
ACTS_PREFIX = "a_"
COMM_PREFIX = "c_"


def _prefix_for(har_name: str) -> str:
    if har_name == "ACTS":
        return ACTS_PREFIX
    if har_name in ("COMM", "MARG"):
        return COMM_PREFIX
    return ""


def write_nus333_gdx_bundle(
    har_dir: str | Path,
    out_dir: str | Path,
    base: str = "nus333",
) -> dict[str, Path]:
    """Read NUS333 HARs and emit the three GDX files getData.gms expects.

    Returns a dict {kind: path} for sets/dat/prm.
    """
    har_dir = Path(har_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sets_har = read_har(str(har_dir / "sets.har"))
    data_har = read_har(str(har_dir / "basedata.har"))
    prm_har = read_har(str(har_dir / "default.prm"))

    # ── Sets.gdx ──
    set_syms = _build_set_symbols(sets_har)
    sets_path = out_dir / f"{base}Sets.gdx"
    write_gdx(sets_path, set_syms)

    # ── Dat.gdx ──  parameters (and the sets they depend on as domain refs)
    dat_syms = _build_param_symbols(data_har, DAT_NAME_MAP, set_syms)
    # NUS333 lacks POP — emit a zero placeholder so getData.gms's $load pop0=pop succeeds.
    if not any(getattr(s, "name", None) == "pop" for s in dat_syms):
        reg_set = next((s for s in set_syms if s.name == "reg"), None)
        if reg_set is not None:
            # Use 1.0, not 0.0: CDE utility divides yc/pop in tariff-sim block
            # (comp_nus333.gms line 2693). Matches convert_nus333_har_to_gdx_v2.py.
            zero_pop = Parameter(
                name="pop",
                dimensions=1,
                description="Population (synthetic 1.0, missing in HAR)",
                domain=["reg"],
                records=[([e[0]], 1.0) for e in reg_set.elements],
            )
            dat_syms.append(zero_pop)
    dat_path = out_dir / f"{base}Dat.gdx"
    write_gdx(dat_path, dat_syms)

    # ── Prm.gdx ──  elasticity parameters
    prm_syms = _build_param_symbols(prm_har, PRM_NAME_MAP, set_syms)
    prm_path = out_dir / f"{base}Prm.gdx"
    write_gdx(prm_path, prm_syms)

    return {"sets": sets_path, "dat": dat_path, "prm": prm_path}


def write_v7_consolidated_gdx(
    har_dir: str | Path,
    out_path: str | Path,
) -> Path:
    """Read GTAP v7 HAR triple (sets.har + basedata.har + default.prm) and emit
    one consolidated GDX consumable by `GTAPSets.load_from_gdx` +
    `GTAPParameters.load_from_gdx`.

    The v7 template auto-aliases set names (r/reg, i/comm, ...) and reads
    elasticity parameter names via `get_elasticity_parameter_name`, so the
    same SET_NAME_MAP / DAT_NAME_MAP / PRM_NAME_MAP used for nus333 apply.
    """
    har_dir = Path(har_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sets_har = read_har(str(har_dir / "sets.har"))
    data_har = read_har(str(har_dir / "basedata.har"))
    prm_har = read_har(str(har_dir / "default.prm"))

    set_syms = _build_set_symbols(sets_har)
    dat_syms = _build_param_symbols(data_har, DAT_NAME_MAP, set_syms)
    prm_syms = _build_param_symbols(prm_har, PRM_NAME_MAP, set_syms)

    # _build_param_symbols prepends set_syms in its output; drop duplicates
    # from prm_syms by filtering to the parameters only.
    dat_params = [s for s in dat_syms if s not in set_syms]
    prm_params = [s for s in prm_syms if s not in set_syms]

    all_syms = [*set_syms, *dat_params, *prm_params]
    write_gdx(out_path, all_syms)
    return out_path


def _build_set_symbols(sets_har: dict[str, HeaderArray]) -> list[Set]:
    """Build the GAMS sets that getData.gms loads from <base>Sets.gdx."""
    syms: list[Set] = []
    # Order matters: parent sets before subsets (for symbol-table indexing).
    order = ["ACTS", "COMM", "REG", "ENDW", "MARG", "ENDM", "ENDF", "ENDS"]
    for har_name in order:
        h = sets_har.get(har_name)
        if h is None:
            continue
        gams_name = SET_NAME_MAP[har_name]
        prefix = _prefix_for(har_name)
        elements = [[f"{prefix}{str(e).strip()}"] for e in h.array.tolist()]
        parent = SUBSET_PARENT.get(har_name)
        domain = [parent] if parent else ["*"]
        syms.append(
            Set(
                name=gams_name,
                dimensions=1,
                description=h.long_name.strip(),
                domain=domain,
                elements=elements,
            )
        )
    return syms


def _build_param_symbols(
    har: dict[str, HeaderArray],
    name_map: dict[str, str],
    set_symbols: list[Set],
) -> list[SymbolBase]:
    """Build params. Domains are mapped from HAR set names to GAMS lowercase."""
    set_name_to_gams = {h.upper(): SET_NAME_MAP.get(h, h.lower()) for h in SET_NAME_MAP}
    # also alias COMM→comm/acts in HAR (depends on context); we use SET_NAME_MAP as-is.

    syms: list[SymbolBase] = list(set_symbols)  # include sets so domain refs resolve

    for har_name, h in har.items():
        if har_name not in name_map:
            continue
        gname = name_map[har_name]
        if h.array.ndim == 0:
            syms.append(
                Parameter(
                    name=gname,
                    dimensions=0,
                    description=h.long_name.strip(),
                    records=[([], float(h.array.item()))],
                )
            )
            continue
        # map domain HAR names → GAMS names (lowercase, with COMM→comm but
        # in many HARs the COMM dim is actually "acts" — we trust HAR's set_names
        # but normalize aliases here)
        domain = []
        for sname in h.set_names:
            up = sname.upper()
            # COMM-shaped dims used as activity index → acts
            domain.append(set_name_to_gams.get(up, up.lower()))

        dim_prefixes = [_prefix_for(s.upper()) for s in h.set_names]
        records = _records_from_array(h, dim_prefixes=dim_prefixes)
        syms.append(
            Parameter(
                name=gname,
                dimensions=h.array.ndim,
                description=h.long_name.strip(),
                domain=domain,
                records=records,
            )
        )
    return syms


# ── shared helpers ───────────────────────────────────────────────────────────


def _is_set_header(h: HeaderArray) -> bool:
    return (
        h.array.ndim == 1
        and h.array.dtype.kind in ("U", "S", "O")
        and not h.set_names
    )


def _set_from_header(h: HeaderArray) -> Set:
    elements = [[str(e).strip()] for e in h.array.tolist()]
    return Set(
        name=h.name,
        dimensions=1,
        description=h.long_name.strip(),
        domain=["*"],
        elements=elements,
    )


def _parameter_from_header(h: HeaderArray) -> Parameter:
    if h.array.ndim == 0:
        return Parameter(
            name=h.name,
            dimensions=0,
            description=h.long_name.strip(),
            domain=[],
            records=[([], float(h.array.item()))],
        )
    return Parameter(
        name=h.name,
        dimensions=h.array.ndim,
        description=h.long_name.strip(),
        domain=list(h.set_names),
        records=_records_from_array(h),
    )


def _records_from_array(
    h: HeaderArray,
    dim_prefixes: list[str] | None = None,
) -> list[tuple[list[str], float]]:
    """Materialize all non-zero records. Sparse: GAMS treats absent keys as 0.

    If dim_prefixes is given, prefix the label at each dimension. Used so that
    acts/comm labels don't collide in shared UEL space (a_AGR vs c_AGR).
    """
    arr = np.asarray(h.array)
    records: list[tuple[list[str], float]] = []
    pref = dim_prefixes or [""] * arr.ndim
    for idx in product(*(range(s) for s in arr.shape)):
        val = float(arr[idx])
        if val == 0.0:
            continue
        keys = [f"{pref[d]}{str(h.set_elements[d][i]).strip()}" for d, i in enumerate(idx)]
        records.append((keys, val))
    return records
