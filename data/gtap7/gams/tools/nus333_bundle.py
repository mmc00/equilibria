"""nus333's HARs to the three GDX files getData.gms reads:
<base>Sets.gdx, <base>Dat.gdx, <base>Prm.gdx, with HAR-uppercase names mapped to
GAMS-lowercase and parent-set membership declared for the MARG/ENDM/ENDF/ENDS
subsets.

Moved from equilibria.babel.har_to_gdx: it decides the data the nus333 GAMS
reference runs on, so it lives with the model and the model-folder sha256 every
result records covers it. The HAR reader and the GDX writer stay in the library.
"""

from __future__ import annotations

from pathlib import Path

from equilibria.babel.gdx.symbols import Parameter, Set, SymbolBase
from equilibria.babel.gdx.writer import write_gdx
from equilibria.babel.har.reader import read_har
from equilibria.babel.har.symbols import HeaderArray
from equilibria.babel.har_to_gdx import _records_from_array

# HAR header name (uppercase) → GAMS symbol name (lowercase) used by getData.gms.
DAT_NAME_MAP: dict[str, str] = {
    "VDFB": "vdfb",
    "VDFP": "vdfp",
    "VMFB": "vmfb",
    "VMFP": "vmfp",
    "VDPB": "vdpb",
    "VDPP": "vdpp",
    "VMPB": "vmpb",
    "VMPP": "vmpp",
    "VDGB": "vdgb",
    "VDGP": "vdgp",
    "VMGB": "vmgb",
    "VMGP": "vmgp",
    "VDIB": "vdib",
    "VDIP": "vdip",
    "VMIB": "vmib",
    "VMIP": "vmip",
    "EVFB": "evfb",
    "EVFP": "evfp",
    "EVOS": "evos",
    "VXSB": "vxsb",
    "VFOB": "vfob",
    "VCIF": "vcif",
    "VMSB": "vmsb",
    "VST": "vst",
    "VTWR": "vtwr",
    "SAVE": "save",
    "VDEP": "vdep",
    "VKB": "vkb",
    "POP": "pop",
    "MAKS": "maks",
    "MAKB": "makb",
}

PRM_NAME_MAP: dict[str, str] = {
    "ESBT": "esubt",
    "ESBC": "esubc",
    "ESBV": "esubva",
    "ETRQ": "etraq",
    "ESBQ": "esubq",
    "INCP": "incpar",
    "SUBP": "subpar",
    "ESBG": "esubg",
    "ESBI": "esubi",
    "ESBD": "esubd",
    "ESBM": "esubm",
    "ESBS": "esubs",
    "ETRE": "etrae",
    "RFLX": "rorFlex",
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

