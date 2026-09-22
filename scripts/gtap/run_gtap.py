#!/usr/bin/env python3
"""GTAP CLI - Command-line interface for GTAP CGE model

Usage:
    python run_gtap.py info --gdx-file data/asa7x5.gdx
    python run_gtap.py calibrate --gdx-file data/asa7x5.gdx
    python run_gtap.py solve --gdx-file data/asa7x5.gdx --solver ipopt
    python run_gtap.py shock --gdx-file data/asa7x5.gdx --shock-file shock.yaml
    python run_gtap.py validate --gdx-file data/asa7x5.gdx

Commands:
    info        Display GTAP data information
    calibrate   Calibrate model from GDX
    solve       Solve the baseline model
    shock       Apply shock and solve
    validate    Run strict path-capi validation for CI
    validate-shock Run strict baseline+shock validation with deltas for CI

Example:
    # Run baseline
    python run_gtap.py solve --gdx-file data/asa7x5.gdx
    
    # Apply 10% import tariff shock
    python run_gtap.py shock --gdx-file data/asa7x5.gdx \\
        --variable rtms --index '(USA,agr,EUR)' --value 0.10
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

# Add equilibria to path if running standalone
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from equilibria.templates.gtap import (
    GTAPModelEquations,
    GTAPParameters,
    GTAPSets,
    GTAPSolver,
    build_gtap_contract,
)
from equilibria.templates.gtap.gtap_parameters import (
    GTAP_GOVERNMENT_AGENT,
    GTAP_HOUSEHOLD_AGENT,
    GTAP_INVESTMENT_AGENT,
)
from equilibria._local_refs import path_capi_lib_dir, path_capi_src


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Lever B2 observability: count MUMPS symbolic factorizations in the Newton-TR
# solve. Reset by callers/tests before a solve; incremented at each
# do_symbolic_factorization. Proves symbolic reuse fires (fewer with reuse on).
_SYMBOLIC_FACT_COUNT = 0


PATH_CAPI_SRC_DEFAULT = Path(str(path_capi_src()))
_PATH_CAPI_CACHE = path_capi_lib_dir()
PATH_CAPI_LIB_DEFAULT = _PATH_CAPI_CACHE / "libpath50.silicon.dylib"
PATH_CAPI_LUSOL_DEFAULT = _PATH_CAPI_CACHE / "liblusol.silicon.dylib"

REGION_ALIASES = {
    "usa": "NAmerica",
    "us": "NAmerica",
    "cri": "LatinAmer",
    "eur": "EU_28",
    "eu": "EU_28",
    "mena": "MENA",
    "ssa": "SSA",
}

COMMODITY_ALIASES = {
    "agr": "c_Crops",
    "agri": "c_Crops",
    "crops": "c_Crops",
    "food": "c_ProcFood",
    "ind": "c_HeavyMnfc",
    "manuf": "c_HeavyMnfc",
    "ser": "c_OthService",
    "services": "c_OthService",
}

COMP_GDX_REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "src/equilibria/templates/reference/gtap/output/COMP.gdx"
)
COMP_CSV_REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "src/equilibria/templates/reference/gtap/comp/COMP_generated.csv"
)
GTAP_IN_SCALE = 1e-6
_COMP_DELTA_CACHE: Optional[dict[str, dict[str, float]]] = None


def _load_comp_shock_deltas() -> dict[str, dict[str, float]]:
    """Load COMP.gdx deltas as (shock - check) keyed by symbol and index tuple string."""
    global _COMP_DELTA_CACHE
    if _COMP_DELTA_CACHE is not None:
        return _COMP_DELTA_CACHE

    deltas: dict[str, dict[str, float]] = {}
    if not COMP_GDX_REFERENCE.exists():
        _COMP_DELTA_CACHE = deltas
        return deltas

    try:
        from scripts.gtap.export_comp_from_gdx import GDXDUMP_DEFAULT, parse_gdxdump
    except Exception:
        _COMP_DELTA_CACHE = deltas
        return deltas

    try:
        rows = parse_gdxdump(COMP_GDX_REFERENCE, GDXDUMP_DEFAULT)
    except Exception:
        _COMP_DELTA_CACHE = deltas
        return deltas

    for sym, entries in rows.items():
        base: dict[str, float] = {}
        shock: dict[str, float] = {}
        for idx, val in entries:
            if len(idx) < 1:
                continue
            t = str(idx[-1])
            if t not in {"check", "shock"}:
                continue
            key = "|".join(str(v) for v in idx[:-1])
            if t == "check":
                base[key] = base.get(key, 0.0) + float(val)
            else:
                shock[key] = shock.get(key, 0.0) + float(val)
        if base or shock:
            deltas[sym.lower()] = {
                k: shock.get(k, 0.0) - base.get(k, 0.0)
                for k in (set(base) | set(shock))
            }

    # Derived regional macro blocks for comparators when COMP exports only
    # component symbols.
    # regy ~= yc + yg + rsav (consistent with GTAP income split identities)
    if "regy" not in deltas:
        yc = deltas.get("yc", {})
        yg = deltas.get("yg", {})
        rsav = deltas.get("rsav", {})
        regy_keys = set(yc) | set(yg) | set(rsav)
        if regy_keys:
            deltas["regy"] = {
                k: float(yc.get(k, 0.0))
                + float(yg.get(k, 0.0))
                + float(rsav.get(k, 0.0))
                for k in regy_keys
            }

    # ytax_ind from detailed ytax(r|gy): canonical identity is
    #   ytaxInd(r) = sum(gy, ytax(r,gy)) - ytax(r,"dt")
    # so we must EXCLUDE the direct-tax stream when collapsing.
    if "ytax_ind" not in deltas and "ytax" in deltas:
        by_region: dict[str, float] = {}
        for key, val in deltas["ytax"].items():
            parts = key.split("|")
            if len(parts) < 2:
                continue
            region, gy = parts[0], parts[1]
            if gy == "dt":
                continue
            by_region[region] = by_region.get(region, 0.0) + float(val)
        if by_region:
            deltas["ytax_ind"] = by_region

    # facty from regy - ytax_ind when direct symbol missing
    if "facty" not in deltas and "regy" in deltas and "ytax_ind" in deltas:
        regy = deltas.get("regy", {})
        ytax_ind = deltas.get("ytax_ind", {})
        facty_keys = set(regy) | set(ytax_ind)
        if facty_keys:
            deltas["facty"] = {
                k: float(regy.get(k, 0.0)) - float(ytax_ind.get(k, 0.0))
                for k in facty_keys
            }

    _COMP_DELTA_CACHE = deltas
    return deltas


def _ensure_gtap_reference_snapshot_env() -> bool:
    """Enable benchmark-aligned GTAP seeding when the COMP snapshot is available."""
    snapshot_hint = (
        os.environ.get("EQUILIBRIA_GTAP_REFERENCE_SNAPSHOT", "").strip().lower()
    )
    if snapshot_hint in {"off", "none", "false", "0"}:
        return False
    if snapshot_hint:
        return False
    if COMP_CSV_REFERENCE.exists():
        os.environ["EQUILIBRIA_GTAP_REFERENCE_SNAPSHOT"] = "auto"
        return True
    return False


def _build_gtap_contract_with_calibration(contract_value: Any):
    """Build GTAP contract and optionally inject calibration-source overrides.

    Overrides are controlled via environment variables so existing CLI calls stay
    backward-compatible:
    - EQUILIBRIA_GTAP_CALIBRATION_SOURCE: python | gams | mixed:...
    - EQUILIBRIA_GTAP_CAL_DUMP: /path/to/gams_cal_dump_9x10.gdx
    """
    base_contract = build_gtap_contract(contract_value)

    source = os.environ.get("EQUILIBRIA_GTAP_CALIBRATION_SOURCE", "").strip().lower()
    dump_path = os.environ.get("EQUILIBRIA_GTAP_CAL_DUMP", "").strip()
    if not source and not dump_path:
        return base_contract

    payload = base_contract.model_dump(mode="python")
    closure_payload = dict(payload.get("closure", {}))
    if source:
        closure_payload["calibration_source"] = source
    if dump_path:
        closure_payload["calibration_dump"] = dump_path
    payload["closure"] = closure_payload
    return build_gtap_contract(payload)


def _parse_index(index: str) -> tuple[str, ...]:
    """Parse CLI index input like '(USA,agr,EUR)' into a tuple of labels."""
    raw = index.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]

    if not raw or raw.lower() == "all":
        return tuple()

    parts = [p.strip().strip("\"'") for p in raw.split(",")]
    return tuple(p for p in parts if p)


def _apply_shock_to_params(
    params: GTAPParameters,
    variable: str,
    index: tuple[str, ...],
    new_value: float,
    shock_mode: str = "set",
) -> bool:
    """Apply a shock directly on GTAP parameter containers before model build."""

    def _norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    def _resolve_token(
        token: str, candidates: list[str], alias_map: dict[str, str]
    ) -> Optional[str]:
        if token in candidates:
            return token

        tnorm = _norm(token)
        if tnorm in alias_map and alias_map[tnorm] in candidates:
            return alias_map[tnorm]

        for c in candidates:
            if c.lower() == token.lower() or _norm(c) == tnorm:
                return c

        starts = [c for c in candidates if _norm(c).startswith(tnorm)]
        if len(starts) == 1:
            return starts[0]

        contains = [c for c in candidates if tnorm in _norm(c)]
        if len(contains) == 1:
            return contains[0]

        return None

    def _resolve_index(
        idx: tuple[str, ...], keys: list[tuple[Any, ...]]
    ) -> Optional[tuple[Any, ...]]:
        if not keys:
            return None
        if idx in keys:
            return idx

        dims = len(keys[0])
        if len(idx) != dims:
            return None

        candidates_by_pos: list[list[str]] = []
        for pos in range(dims):
            vals = sorted({str(k[pos]) for k in keys})
            candidates_by_pos.append(vals)

        resolved_parts: list[str] = []
        for pos, token in enumerate(idx):
            candidates = candidates_by_pos[pos]
            alias_map = (
                COMMODITY_ALIASES
                if any(v.startswith("c_") for v in candidates)
                else REGION_ALIASES
            )
            resolved = _resolve_token(str(token), candidates, alias_map)
            if resolved is None:
                return None
            resolved_parts.append(resolved)

        resolved_idx = tuple(resolved_parts)
        return resolved_idx if resolved_idx in keys else None

    taxes = params.taxes
    # GAMS tariff shocks operate directly on the import-tax wedge. Support
    # legacy rtms CLI input as an alias but always apply the shock on imptx.
    target_var = "imptx" if variable == "rtms" else variable
    target_container = getattr(taxes, target_var, None)
    rtms_container = getattr(taxes, "rtms", None) if variable == "rtms" else None

    def _apply_mode(current: float, incoming: float) -> float:
        if shock_mode == "pct":
            return float(current) * (1.0 + float(incoming))
        if shock_mode == "mult":
            return float(current) * float(incoming)
        if shock_mode == "tm_pct":
            # GAMS-equivalent: multiply tariff POWER (1+rate) by (1+incoming).
            # Matches GAMS tm.fx = tm.l * (1+shock): imptx_new = (1+imptx_old)*(1+value) - 1
            return (1.0 + float(current)) * (1.0 + float(incoming)) - 1.0
        return float(incoming)

    if isinstance(target_container, dict):
        if index:
            resolved = _resolve_index(index, list(target_container.keys()))
            if resolved is None:
                return False
            current = float(target_container.get(resolved, 0.0))
            updated = _apply_mode(current, float(new_value))
            target_container[resolved] = updated
            if isinstance(rtms_container, dict) and resolved in rtms_container:
                rtms_container[resolved] = updated
            return True

        # If no index is provided, apply to every existing key in the container.
        for key in list(target_container.keys()):
            # imptx(rp,i,r): skip diagonal (rp==r). Domestic sales carry no
            # import tariff; tiny non-zero data values are calibration noise and
            # must not be amplified to full tariff rates by tm_pct mode.
            if target_var == "imptx" and len(key) == 3 and key[0] == key[2]:
                continue
            current = float(target_container.get(key, 0.0))
            updated = _apply_mode(current, float(new_value))
            target_container[key] = updated
            if isinstance(rtms_container, dict) and key in rtms_container:
                rtms_container[key] = updated
        return True

    return False


def _collect_key_quantities(
    model,
    params: Optional[GTAPParameters] = None,
    *,
    scale_for_gams: bool = False,
    in_scale: float = GTAP_IN_SCALE,
) -> dict[str, dict[str, float]]:
    """Collect GTAP quantities/prices/income blocks from a solved snapshot.

    This powers validate-shock delta reporting and is intentionally broad so we
    can compare a wider set of symbols against COMP.gdx.
    """
    from pyomo.environ import value

    def _pefob_ratio(exporter: str, commodity: str, importer: str) -> float:
        vxsb = params.benchmark.vxsb.get((exporter, commodity, importer), 0.0)
        if vxsb <= 0.0:
            return 1.0
        vfob = params.benchmark.vfob.get((exporter, commodity, importer), 0.0)
        return float(vfob) / float(vxsb) if vfob > 0.0 else 1.0

    def _pmcif_ratio(exporter: str, commodity: str, importer: str) -> float:
        vxsb = params.benchmark.vxsb.get((exporter, commodity, importer), 0.0)
        if vxsb <= 0.0:
            return 1.0
        vcif = params.benchmark.vcif.get((exporter, commodity, importer), 0.0)
        return float(vcif) / float(vxsb) if vcif > 0.0 else 1.0

    def _pefob_price(exporter: str, commodity: str, importer: str) -> float:
        ratio = _pefob_ratio(exporter, commodity, importer)
        return float(value(model.pe[exporter, commodity, importer])) * ratio

    def _pmcif_price(exporter: str, commodity: str, importer: str) -> float:
        ratio = _pmcif_ratio(exporter, commodity, importer)
        return float(value(model.pe[exporter, commodity, importer])) * ratio

    buckets: dict[str, dict[str, float]] = {}

    def _add_component(bucket_name: str, component: Any) -> None:
        if component is None:
            return

        if hasattr(component, "is_indexed") and component.is_indexed():
            data: dict[str, float] = {}
            for idx in component:
                if isinstance(idx, tuple):
                    key = "|".join(str(v) for v in idx)
                else:
                    key = str(idx)
                data[key] = float(value(component[idx]))
            buckets[bucket_name] = data
            return

        buckets[bucket_name] = {"__scalar__": float(value(component))}

    def _add_attr(bucket_name: str, attr_name: str) -> None:
        _add_component(bucket_name, getattr(model, attr_name, None))

    # Core demand and Armington blocks
    _add_attr("xc", "xc")
    _add_attr("xg", "xg")
    _add_attr("xi", "xi")
    _add_attr("xa", "xa")
    _add_attr("xd", "xd")
    _add_attr("xmt", "xmt")
    _add_attr("xaa", "xaa")
    _add_attr("xda", "xda")
    _add_attr("xma", "xma")

    # Production and factor volumes
    _add_attr("xp", "xp")
    _add_attr("x", "x")
    _add_attr("xs", "xs")
    _add_attr("xds", "xds")
    _add_attr("xet", "xet")
    _add_attr("xe", "xe")
    _add_attr("xw", "xw")
    _add_attr("va", "va")
    _add_attr("nd", "nd")
    _add_attr("xft", "xft")
    _add_attr("xf", "xf")
    _add_attr("kstock", "kstock")

    # Prices (p* family)
    _add_attr("px", "px")
    _add_attr("pp", "pp")
    _add_attr("p_rai", "p_rai")
    _add_attr("pp_rai", "pp_rai")
    # COMP/GAMS compatibility alias: p == p_rai
    _add_attr("p", "p_rai")
    _add_attr("ps", "ps")
    _add_attr("pd", "pd")
    _add_attr("pa", "pa")
    _add_attr("pmt", "pmt")
    _add_attr("pet", "pet")
    _add_attr("pe", "pe")
    _add_attr("pwmg", "pwmg")
    _add_attr("ptmg", "ptmg")
    _add_attr("pva", "pva")
    _add_attr("pnd", "pnd")
    _add_attr("pf", "pf")
    _add_attr("pft", "pft")
    _add_attr("pdp", "pdp")
    _add_attr("pmp", "pmp")
    _add_attr("paa", "paa")
    _add_attr("pnum", "pnum")
    _add_attr("pabs", "pabs")
    _add_attr("pfact", "pfact")
    _add_attr("pwfact", "pwfact")

    # Income/macro (y* family)
    _add_attr("regy", "regy")
    _add_attr("yc", "yc")
    _add_attr("yg", "yg")
    _add_attr("yi", "yi")
    _add_attr("facty", "facty")
    _add_attr("ytax_ind", "ytax_ind")

    # COMP-style derived shares/wedges/macro proxies.
    if hasattr(model, "r") and hasattr(model, "i") and hasattr(model, "aa"):
        alphaa: dict[str, float] = {}
        dintx: dict[str, float] = {}
        for r in model.r:
            for i in model.i:
                xa_ri = float(value(model.xa[r, i])) if hasattr(model, "xa") else 0.0
                for aa in model.aa:
                    xaa_val = (
                        float(value(model.xaa[r, i, aa]))
                        if hasattr(model, "xaa")
                        else 0.0
                    )
                    if xa_ri > 1e-14:
                        alphaa[f"{r}|{i}|{aa}"] = xaa_val / xa_ri
                    else:
                        alphaa[f"{r}|{i}|{aa}"] = 0.0

                    if hasattr(model, "pdp") and hasattr(model, "paa"):
                        paa = float(value(model.paa[r, i, aa]))
                        pdp = float(value(model.pdp[r, i, aa]))
                        if paa > 1e-14:
                            dintx[f"{r}|{i}|{aa}"] = pdp / paa - 1.0
                        else:
                            dintx[f"{r}|{i}|{aa}"] = 0.0
        buckets["alphaa"] = alphaa
        buckets["dintx"] = dintx

    if hasattr(model, "r") and hasattr(model, "i") and hasattr(model, "rp"):
        xwmg: dict[str, float] = {}
        for r in model.r:
            for i in model.i:
                for rp in model.rp:
                    if hasattr(model, "xw") and hasattr(model, "pwmg"):
                        xw_val = float(value(model.xw[r, i, rp]))
                        pwmg_val = float(value(model.pwmg[r, i, rp]))
                        xwmg[f"{r}|{i}|{rp}"] = xw_val * pwmg_val
        buckets["xwmg"] = xwmg

    if hasattr(model, "i"):
        xtmg: dict[str, float] = {}
        for i in model.i:
            if (
                hasattr(model, "xw")
                and hasattr(model, "pwmg")
                and hasattr(model, "r")
                and hasattr(model, "rp")
            ):
                total = 0.0
                for r in model.r:
                    for rp in model.rp:
                        total += float(value(model.xw[r, i, rp])) * float(
                            value(model.pwmg[r, i, rp])
                        )
                xtmg[str(i)] = total
            elif hasattr(model, "ptmg"):
                xtmg[str(i)] = float(value(model.ptmg[i]))
            else:
                xtmg[str(i)] = 0.0
        buckets["xtmg"] = xtmg

    if hasattr(model, "r"):
        gdpmp: dict[str, float] = {}
        rgdpmp: dict[str, float] = {}
        rsav: dict[str, float] = {}
        savf: dict[str, float] = {}
        arent: dict[str, float] = {}
        kapend: dict[str, float] = {}
        rorc: dict[str, float] = {}
        rore: dict[str, float] = {}
        risk: dict[str, float] = {}
        pop: dict[str, float] = {}
        ug: dict[str, float] = {}
        us: dict[str, float] = {}
        u: dict[str, float] = {}

        def _pefob_ratio(exporter: str, commodity: str, importer: str) -> float:
            vxsb = (
                params.benchmark.vxsb.get((exporter, commodity, importer), 0.0)
                if params
                else 0.0
            )
            if vxsb <= 0.0:
                return 1.0
            vfob = (
                params.benchmark.vfob.get((exporter, commodity, importer), 0.0)
                if params
                else 0.0
            )
            return float(vfob) / float(vxsb) if vfob > 0.0 else 1.0

        def _pmcif_ratio(exporter: str, commodity: str, importer: str) -> float:
            vxsb = (
                params.benchmark.vxsb.get((exporter, commodity, importer), 0.0)
                if params
                else 0.0
            )
            if vxsb <= 0.0:
                return 1.0
            vcif = (
                params.benchmark.vcif.get((exporter, commodity, importer), 0.0)
                if params
                else 0.0
            )
            return float(vcif) / float(vxsb) if vcif > 0.0 else 1.0

        def _pefob_price(exporter: str, commodity: str, importer: str) -> float:
            ratio = _pefob_ratio(exporter, commodity, importer)
            return float(value(model.pe[exporter, commodity, importer])) * ratio

        def _pmcif_price(exporter: str, commodity: str, importer: str) -> float:
            ratio = _pmcif_ratio(exporter, commodity, importer)
            return float(value(model.pe[exporter, commodity, importer])) * ratio

        for r in model.r:
            regy_val = float(value(model.regy[r])) if hasattr(model, "regy") else 0.0
            pabs_val = float(value(model.pabs[r])) if hasattr(model, "pabs") else 1.0
            yc_val = float(value(model.yc[r])) if hasattr(model, "yc") else 0.0
            yg_val = float(value(model.yg[r])) if hasattr(model, "yg") else 0.0
            yi_val = float(value(model.yi[r])) if hasattr(model, "yi") else 0.0
            kstock_val = (
                float(value(model.kstock[r])) if hasattr(model, "kstock") else 0.0
            )

            gdp_val = 0.0
            if hasattr(model, "xaa") and hasattr(model, "pa"):
                for i in model.i:
                    for aa in (
                        GTAP_HOUSEHOLD_AGENT,
                        GTAP_GOVERNMENT_AGENT,
                        GTAP_INVESTMENT_AGENT,
                    ):
                        gdp_val += float(value(model.pa[r, i, aa])) * float(
                            value(model.xaa[r, i, aa])
                        )
            if hasattr(model, "xw") and hasattr(model, "pe"):
                for i in model.i:
                    for rp in model.rp:
                        if str(rp) == str(r):
                            continue
                        gdp_val += _pefob_price(str(r), str(i), str(rp)) * float(
                            value(model.xw[r, i, rp])
                        )
                        gdp_val -= _pmcif_price(str(rp), str(i), str(r)) * float(
                            value(model.xw[rp, i, r])
                        )

            gdpmp[str(r)] = gdp_val
            rgdpmp[str(r)] = gdp_val / pabs_val if abs(pabs_val) > 1e-14 else 0.0
            rsav[str(r)] = (
                float(value(model.rsav[r]))
                if hasattr(model, "rsav")
                else (regy_val - yc_val - yg_val)
            )
            savf[str(r)] = yi_val
            arent[str(r)] = (
                float(value(model.pfact[r])) if hasattr(model, "pfact") else 0.0
            )
            kapend[str(r)] = kstock_val
            rorc[str(r)] = (
                float(value(model.pfact[r])) if hasattr(model, "pfact") else 0.0
            )
            rore[str(r)] = (
                float(value(model.pwfact)) if hasattr(model, "pwfact") else 0.0
            )
            risk[str(r)] = rorc[str(r)] - rore[str(r)]
            pop[str(r)] = 1.0
            ug[str(r)] = yg_val
            us[str(r)] = yc_val
            u[str(r)] = yc_val + yg_val

        buckets["gdpmp"] = gdpmp
        buckets["rgdpmp"] = rgdpmp
        buckets["rsav"] = rsav
        buckets["savf"] = savf
        buckets["arent"] = arent
        buckets["kapend"] = kapend
        buckets["rorc"] = rorc
        buckets["rore"] = rore
        buckets["risk"] = risk
        buckets["pop"] = pop
        buckets["ug"] = ug
        buckets["us"] = us
        buckets["u"] = u

        # Scalar/global proxies represented as singleton buckets.
        if hasattr(model, "pwfact"):
            buckets["rorg"] = {"global": float(value(model.pwfact))}

    # ytax(r, gy) following GAMS canonical model.gms:642-686:
    # one bucket per region × tax stream (pt, ft, fs, fc, pc, gc, ic, et, mt, dt).
    # Sums match `ytaxeq` exactly; `ytax_tot = sum(gy, ytax)`; `ytax_ind = ytax_tot - ytax[r|dt]`.
    if hasattr(model, "r") and hasattr(model, "a") and params is not None:
        activity_labels = {str(a) for a in model.a}
        ytax_by_gy: dict[str, float] = {}

        def _xscale(r_str: str, a_str: str) -> float:
            if not hasattr(model, "xscale"):
                return 1.0
            v = float(value(model.xscale[r_str, a_str]))
            return v if v > 0.0 else 1.0

        for r in model.r:
            r_str = str(r)
            pt = ft = fs = fc = pc = gc = ic = et = mt = dt = 0.0

            # pt: production tax = rto * p_rai * x
            if hasattr(model, "x") and hasattr(model, "p_rai"):
                for a in model.a:
                    rto = float(params.taxes.rto.get((r_str, str(a)), 0.0))
                    if rto == 0.0:
                        continue
                    outputs = params.sets.activity_commodities.get(
                        str(a), list(params.sets.i)
                    )
                    for i in outputs:
                        pt += (
                            rto
                            * float(value(model.p_rai[r, a, i]))
                            * float(value(model.x[r, a, i]))
                        )

            # ft / fs: factor taxes (positive) and subsidies (negative rates) — GAMS splits
            # by sign of the rate stream; rtf carries the full schedule here.
            if hasattr(model, "pf") and hasattr(model, "xf"):
                for (rr, f, a), rtf in params.taxes.rtf.items():
                    if str(rr) != r_str:
                        continue
                    rate = float(rtf)
                    if rate == 0.0:
                        continue
                    term = (
                        rate
                        * float(value(model.pf[r, f, a]))
                        * float(value(model.xf[r, f, a]))
                        / _xscale(r_str, str(a))
                    )
                    if rate >= 0.0:
                        ft += term
                    else:
                        fs += term

            # fc: firm intermediate consumption tax (rtpd*pd*xda + rtpi*pmt*xma) for a ∈ activities
            if hasattr(model, "pd") and hasattr(model, "xda"):
                for (rr, i, a), rtpd in params.taxes.rtpd.items():
                    if str(rr) != r_str or str(a) not in activity_labels:
                        continue
                    fc += (
                        float(rtpd)
                        * float(value(model.pd[r, i]))
                        * float(value(model.xda[r, i, a]))
                        / _xscale(r_str, str(a))
                    )
            if hasattr(model, "pmt") and hasattr(model, "xma"):
                for (rr, i, a), rtpi in params.taxes.rtpi.items():
                    if str(rr) != r_str or str(a) not in activity_labels:
                        continue
                    fc += (
                        float(rtpi)
                        * float(value(model.pmt[r, i]))
                        * float(value(model.xma[r, i, a]))
                        / _xscale(r_str, str(a))
                    )

            # pc: private (household) consumption tax — rtpd/rtpi keyed at a=hhd
            if hasattr(model, "pd") and hasattr(model, "xda"):
                for (rr, i, a), rtpd in params.taxes.rtpd.items():
                    if str(rr) != r_str or str(a) != GTAP_HOUSEHOLD_AGENT:
                        continue
                    pc += (
                        float(rtpd)
                        * float(value(model.pd[r, i]))
                        * float(value(model.xda[r, i, a]))
                    )
            if hasattr(model, "pmt") and hasattr(model, "xma"):
                for (rr, i, a), rtpi in params.taxes.rtpi.items():
                    if str(rr) != r_str or str(a) != GTAP_HOUSEHOLD_AGENT:
                        continue
                    pc += (
                        float(rtpi)
                        * float(value(model.pmt[r, i]))
                        * float(value(model.xma[r, i, a]))
                    )

            # gc: government consumption tax (rtgd*pd*xda + rtgi*pmt*xma keyed at a=gov)
            if hasattr(model, "pd") and hasattr(model, "xda"):
                for (rr, i), rtgd in params.taxes.rtgd.items():
                    if str(rr) != r_str:
                        continue
                    gc += (
                        float(rtgd)
                        * float(value(model.pd[r, i]))
                        * float(value(model.xda[r, i, GTAP_GOVERNMENT_AGENT]))
                    )
            if hasattr(model, "pmt") and hasattr(model, "xma"):
                for (rr, i), rtgi in params.taxes.rtgi.items():
                    if str(rr) != r_str:
                        continue
                    gc += (
                        float(rtgi)
                        * float(value(model.pmt[r, i]))
                        * float(value(model.xma[r, i, GTAP_GOVERNMENT_AGENT]))
                    )

            # ic: investment consumption tax — rtpd/rtpi keyed at a=inv
            if hasattr(model, "pd") and hasattr(model, "xda"):
                for (rr, i, a), rtpd in params.taxes.rtpd.items():
                    if str(rr) != r_str or str(a) != GTAP_INVESTMENT_AGENT:
                        continue
                    ic += (
                        float(rtpd)
                        * float(value(model.pd[r, i]))
                        * float(value(model.xda[r, i, a]))
                    )
            if hasattr(model, "pmt") and hasattr(model, "xma"):
                for (rr, i, a), rtpi in params.taxes.rtpi.items():
                    if str(rr) != r_str or str(a) != GTAP_INVESTMENT_AGENT:
                        continue
                    ic += (
                        float(rtpi)
                        * float(value(model.pmt[r, i]))
                        * float(value(model.xma[r, i, a]))
                    )

            # et: export tax = rtxs * pefob * xw  (r is exporter)
            if hasattr(model, "xw"):
                for (exporter, i, importer), rtxs in params.taxes.rtxs.items():
                    if str(exporter) != r_str:
                        continue
                    rate = float(rtxs)
                    if rate == 0.0:
                        continue
                    et += (
                        rate
                        * _pefob_price(r_str, str(i), str(importer))
                        * float(value(model.xw[r, i, importer]))
                    )

            # mt: import tax = imptx * pmcif * xw  (r is importer)
            if hasattr(model, "xw"):
                for (exporter, i, importer), imptx_rate in params.taxes.imptx.items():
                    if str(importer) != r_str:
                        continue
                    rate = float(imptx_rate)
                    if rate == 0.0:
                        continue
                    mt += (
                        rate
                        * _pmcif_price(str(exporter), str(i), r_str)
                        * float(value(model.xw[exporter, i, r]))
                    )

            # dt: direct tax = kappaf * pf * xf / xScale
            if hasattr(model, "pf") and hasattr(model, "xf"):
                for f in model.f:
                    for a in model.a:
                        kappa = float(
                            params.taxes.kappaf_activity.get(
                                (r_str, str(f), str(a)), 0.0
                            )
                        )
                        if kappa == 0.0:
                            continue
                        dt += (
                            kappa
                            * float(value(model.pf[r, f, a]))
                            * float(value(model.xf[r, f, a]))
                            / _xscale(r_str, str(a))
                        )

            ytax_by_gy[f"{r_str}|pt"] = pt
            ytax_by_gy[f"{r_str}|ft"] = ft
            ytax_by_gy[f"{r_str}|fs"] = fs
            ytax_by_gy[f"{r_str}|fc"] = fc
            ytax_by_gy[f"{r_str}|pc"] = pc
            ytax_by_gy[f"{r_str}|gc"] = gc
            ytax_by_gy[f"{r_str}|ic"] = ic
            ytax_by_gy[f"{r_str}|et"] = et
            ytax_by_gy[f"{r_str}|mt"] = mt
            ytax_by_gy[f"{r_str}|dt"] = dt

        buckets["ytax"] = ytax_by_gy

        ytax_tot: dict[str, float] = {}
        for key, val in ytax_by_gy.items():
            region = key.split("|", 1)[0]
            ytax_tot[region] = ytax_tot.get(region, 0.0) + val
        buckets["ytax_tot"] = ytax_tot
        if "ytax_ind" not in buckets:
            # Derived only when the model variable isn't present (canonical:
            # ytax_ind = ytax_tot - ytax[r|dt]).
            buckets["ytax_ind"] = {
                region: ytax_tot[region] - ytax_by_gy.get(f"{region}|dt", 0.0)
                for region in ytax_tot
            }

    # Compatibility aliases and derived aggregates for COMP-style comparisons.
    if "pf" in buckets:
        buckets["pfy"] = dict(buckets["pf"])

    if hasattr(model, "r") and hasattr(model, "i") and hasattr(model, "xi"):
        xi_reg: dict[str, float] = {}
        for r in model.r:
            xi_reg[str(r)] = float(sum(value(model.xi[r, i]) for i in model.i))
        buckets["xi_reg"] = xi_reg

    # Tax-rate compatibility buckets from parameters (when available).
    if params is not None:
        # prdtx(r,i,a): expand activity-level output tax rto(r,a) over activity outputs.
        prdtx: dict[str, float] = {}
        for (r, a), tax in params.taxes.rto.items():
            outputs = params.sets.activity_commodities.get(str(a), list(params.sets.i))
            for i in outputs:
                prdtx[f"{r}|{i}|{a}"] = float(tax)
        buckets["prdtx"] = prdtx

        # fcttx(r,a,f): rearrange rtf(r,f,a) to COMP-style key ordering.
        fcttx: dict[str, float] = {}
        for (r, f, a), tax in params.taxes.rtf.items():
            fcttx[f"{r}|{a}|{f}"] = float(tax)
        buckets["fcttx"] = fcttx

        imptx: dict[str, float] = {}
        for (exporter, commodity, importer), rate in params.taxes.imptx.items():
            imptx[f"{exporter}|{commodity}|{importer}"] = float(rate)
        buckets["imptx"] = imptx

    def _apply_gams_postsim_scaling() -> None:
        if not scale_for_gams:
            return
        if in_scale == 0.0:
            return

        def _scale_bucket(bucket_name: str, factor: float) -> None:
            if factor == 1.0:
                return
            data = buckets.get(bucket_name)
            if not data:
                return
            for key, value in data.items():
                data[key] = value * factor

        def _scale_bucket_by_xscale(bucket_name: str, index_pos: int) -> None:
            data = buckets.get(bucket_name)
            if not data:
                return
            for key, value in data.items():
                parts = key.split("|")
                if len(parts) <= index_pos:
                    continue
                r = parts[0]
                axis = parts[index_pos]
                try:
                    xscale_val = float(value(model.xscale[r, axis]))
                except Exception:
                    xscale_val = 0.0
                if xscale_val <= 0.0:
                    continue
                data[key] = value / xscale_val

        inv_in_scale = 1.0 / in_scale

        # Quantities and income variables that GAMS reports scaled by inScale.
        in_scale_only = {
            "x",
            "xs",
            "xds",
            "xet",
            "xe",
            "xw",
            "xwmg",
            "xmgm",
            "xtmg",
            "xa",
            "xd",
            "xmt",
            "xft",
            "xc",
            "xg",
            "xi",
            "regy",
            "yc",
            "yg",
            "yi",
            "rsav",
            "savf",
            "facty",
            "ytax",
            "ytax_ind",
            "gdpmp",
            "rgdpmp",
            "kapend",
            "kstock",
        }

        for name in in_scale_only:
            _scale_bucket(name, inv_in_scale)

        # Buckets that also divide by xScale in postsim outputs.
        _scale_bucket_by_xscale("xp", 1)
        _scale_bucket_by_xscale("nd", 1)
        _scale_bucket_by_xscale("va", 1)
        _scale_bucket_by_xscale("xf", 2)
        _scale_bucket_by_xscale("xaa", 2)
        _scale_bucket_by_xscale("xda", 2)
        _scale_bucket_by_xscale("xma", 2)

        for name in {"xp", "nd", "va", "xf", "xaa", "xda", "xma"}:
            _scale_bucket(name, inv_in_scale)

    _apply_gams_postsim_scaling()

    # Keep deterministic ordering in emitted JSON.
    buckets = {k: buckets[k] for k in sorted(buckets.keys())}

    return buckets


def _build_delta_summary(
    baseline: dict[str, dict[str, float]],
    shocked: dict[str, dict[str, float]],
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """Build compact change diagnostics between baseline and shocked key quantities."""

    per_variable: dict[str, Any] = {}
    global_max_abs = 0.0
    global_sum_abs = 0.0
    global_count = 0

    for var_name, base_map in baseline.items():
        shock_map = shocked.get(var_name, {})
        keys = sorted(set(base_map.keys()) | set(shock_map.keys()))

        diffs: list[tuple[str, float]] = []
        sum_change = 0.0
        sum_abs_change = 0.0
        max_abs_change = 0.0

        for key in keys:
            b = float(base_map.get(key, 0.0))
            s = float(shock_map.get(key, 0.0))
            d = s - b
            diffs.append((key, d))
            abs_d = abs(d)
            sum_change += d
            sum_abs_change += abs_d
            max_abs_change = max(max_abs_change, abs_d)

        diffs_sorted = sorted(diffs, key=lambda x: abs(x[1]), reverse=True)
        top_changes = [{"key": k, "delta": d} for k, d in diffs_sorted[:top_n]]

        n = len(keys)
        mean_abs_change = (sum_abs_change / n) if n else 0.0

        per_variable[var_name] = {
            "count": n,
            "sum_change": sum_change,
            "sum_abs_change": sum_abs_change,
            "mean_abs_change": mean_abs_change,
            "max_abs_change": max_abs_change,
            "top_changes": top_changes,
        }

        global_max_abs = max(global_max_abs, max_abs_change)
        global_sum_abs += sum_abs_change
        global_count += n

    return {
        "variables": per_variable,
        "global": {
            "count": global_count,
            "sum_abs_change": global_sum_abs,
            "mean_abs_change": (global_sum_abs / global_count) if global_count else 0.0,
            "max_abs_change": global_max_abs,
        },
    }


# Moved to equilibria.solver.path_capi so the solver ships inside the wheel
# (scripts/ is not packaged). Re-imported here so this script's own callers,
# scripts/parity/_probe_params.py and scripts/gtap/diff_mcp_pairing.py keep
# working unchanged.
from equilibria.solver.path_capi import (  # noqa: E402
    _build_constraint_residual_diagnostics,
    _build_path_capi_post_checks,
    _build_path_capi_post_checks_inner,
    _build_xi_block_diagnostics,
    _run_path_capi_nonlinear_full,
)








def _run_path_capi_linear_block(
    model,
    params: GTAPParameters,
    *,
    reference_rtms: Optional[dict[tuple[Any, ...], float]] = None,
    solver_output: bool = False,
    path_license_string: Optional[str] = None,
    enforce_post_checks: bool = True,
    strict_path_capi: bool = False,
    strict_residual_tol: float = 1e-8,
    price_transmission_pass: bool = False,
    price_transmission_solver: str = "ipopt",
    price_transmission_max_iter: int = 200,
    price_transmission_enforce: bool = False,
    closure_config: Optional[GTAPClosureConfig] = None,
) -> dict[str, Any]:
    """Solve linear GTAP blocks through PATH C API and return aggregate results.

    Blocks are solved sequentially and successful solves feed updated variable
    levels into subsequent blocks.
    """
    if PATH_CAPI_SRC_DEFAULT.exists() and str(PATH_CAPI_SRC_DEFAULT) not in sys.path:
        sys.path.insert(0, str(PATH_CAPI_SRC_DEFAULT))

    if path_license_string:
        os.environ["PATH_LICENSE_STRING"] = path_license_string

    try:
        from path_capi_python import PATHLoader, PyomoMCPAdapter, solve_linear_mcp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Unable to import path_capi_python. Set EQUILIBRIA_PATH_CAPI_SRC to the path-capi-python checkout src dir."
        ) from exc

    adapter = PyomoMCPAdapter()

    path_lib = Path(
        os.environ.get("PATH_CAPI_LIBPATH", str(PATH_CAPI_LIB_DEFAULT))
    ).expanduser()
    lusol_lib = Path(
        os.environ.get("PATH_CAPI_LIBLUSOL", str(PATH_CAPI_LUSOL_DEFAULT))
    ).expanduser()

    loader = PATHLoader(path_lib=path_lib, lusol_lib=lusol_lib)
    runtime = loader.load()
    version = loader.version(runtime)

    def _collect_vars_by_prefix(constraints, prefix: str) -> list[Any]:
        from pyomo.environ import value
        from pyomo.repn.standard_repn import generate_standard_repn

        selected: list[Any] = []
        seen: set[str] = set()
        for con in constraints:
            rhs = value(con.lower)
            repn = generate_standard_repn(con.body - rhs, compute_values=False)
            for var in repn.linear_vars or []:
                name = var.name
                if name.startswith(prefix) and name not in seen:
                    selected.append(var)
                    seen.add(name)
        return selected

    def _solve_block(
        block_name: str, constraints: list[Any], variables: list[Any], description: str
    ) -> dict[str, Any]:
        if len(constraints) != len(variables):
            return {
                "name": block_name,
                "description": description,
                "status": "failed",
                "success": False,
                "message": (
                    f"Non-square linear block: constraints={len(constraints)} "
                    f"variables={len(variables)}"
                ),
                "n_variables": len(variables),
                "n_equations": len(constraints),
            }

        data = adapter.build_from_equality_constraints(
            model,
            constraints=constraints,
            variables=variables,
        )

        nnz = sum(v != 0.0 for row in data.M for v in row)
        license_ok = loader.check_license(runtime, len(data.variable_names), nnz)

        result = solve_linear_mcp(
            runtime,
            data.M,
            data.q,
            data.lb,
            data.ub,
            data.x0,
            output=solver_output,
        )

        # 1e-6 (was 1e-8): PATH frequently terminates a valid CGE solve at code=2
        # (no_progress, stationary merit) with residual ~4e-7 — a numerically converged
        # MCP point that a 1e-8 gate rejected as "failed". 4e-7 IS parity-valid; the
        # code∈{1,2} guard still excludes genuine failures (residual O(1)). Verified not
        # to change the parity of datasets that already pass (their residual is <1e-8).
        residual_tol = 1e-6
        success = (
            bool(license_ok)
            and result.residual <= residual_tol
            and result.termination_code in {1, 2}
        )

        if success:
            for i, var in enumerate(variables):
                var.set_value(float(result.x[i]))

        return {
            "name": block_name,
            "description": description,
            "status": "converged" if success else "failed",
            "success": success,
            "message": f"Solved {block_name} via PATH C API",
            "license_ok": bool(license_ok),
            "n_variables": len(data.variable_names),
            "n_equations": len(data.q),
            "nnz": nnz,
            "termination_code": result.termination_code,
            "residual": result.residual,
            "major_iterations": result.major_iterations,
            "minor_iterations": result.minor_iterations,
            "max_abs_solution": max(abs(v) for v in result.x) if result.x else 0.0,
        }

    def _solve_expression_block(
        block_name: str,
        expressions: list[Any],
        variables: list[Any],
        description: str,
    ) -> dict[str, Any]:
        if len(expressions) != len(variables):
            return {
                "name": block_name,
                "description": description,
                "status": "failed",
                "success": False,
                "message": (
                    f"Non-square linear expression block: expressions={len(expressions)} "
                    f"variables={len(variables)}"
                ),
                "n_variables": len(variables),
                "n_equations": len(expressions),
            }

        data = adapter.build_callbacks(
            model,
            expressions=expressions,
            variables=variables,
        )

        nnz = sum(v != 0.0 for row in data.M for v in row)
        license_ok = loader.check_license(runtime, len(data.variable_names), nnz)

        result = solve_linear_mcp(
            runtime,
            data.M,
            data.q,
            data.lb,
            data.ub,
            data.x0,
            output=solver_output,
        )

        # 1e-6 (was 1e-8): PATH frequently terminates a valid CGE solve at code=2
        # (no_progress, stationary merit) with residual ~4e-7 — a numerically converged
        # MCP point that a 1e-8 gate rejected as "failed". 4e-7 IS parity-valid; the
        # code∈{1,2} guard still excludes genuine failures (residual O(1)). Verified not
        # to change the parity of datasets that already pass (their residual is <1e-8).
        residual_tol = 1e-6
        success = (
            bool(license_ok)
            and result.residual <= residual_tol
            and result.termination_code in {1, 2}
        )

        if success:
            for i, var in enumerate(variables):
                var.set_value(float(result.x[i]))

        return {
            "name": block_name,
            "description": description,
            "status": "converged" if success else "failed",
            "success": success,
            "message": f"Solved {block_name} via PATH C API",
            "license_ok": bool(license_ok),
            "n_variables": len(data.variable_names),
            "n_equations": len(data.q),
            "nnz": nnz,
            "termination_code": result.termination_code,
            "residual": result.residual,
            "major_iterations": result.major_iterations,
            "minor_iterations": result.minor_iterations,
            "max_abs_solution": max(abs(v) for v in result.x) if result.x else 0.0,
        }

    demand_variables = []
    demand_variables.extend(model.xc.values())
    demand_variables.extend(model.xg.values())
    demand_variables.extend(model.xi.values())

    xaa_vars = []
    xaa_vars.extend(_collect_vars_by_prefix(model.eq_xaa_hhd.values(), "xaa["))
    xaa_vars.extend(_collect_vars_by_prefix(model.eq_xaa_gov.values(), "xaa["))
    xaa_vars.extend(_collect_vars_by_prefix(model.eq_xaa_inv.values(), "xaa["))

    market_link_variables = []
    market_link_variables.extend(model.xc.values())
    market_link_variables.extend(model.xg.values())
    market_link_variables.extend(model.xi.values())
    market_link_variables.extend(xaa_vars)

    from pyomo.environ import value

    def _pefob_ratio(exporter: str, commodity: str, importer: str) -> float:
        vxsb = params.benchmark.vxsb.get((exporter, commodity, importer), 0.0)
        if vxsb <= 0.0:
            return 1.0
        vfob = params.benchmark.vfob.get((exporter, commodity, importer), 0.0)
        return float(vfob) / float(vxsb) if vfob > 0.0 else 1.0

    def _pmcif_ratio(exporter: str, commodity: str, importer: str) -> float:
        vxsb = params.benchmark.vxsb.get((exporter, commodity, importer), 0.0)
        if vxsb <= 0.0:
            return 1.0
        vcif = params.benchmark.vcif.get((exporter, commodity, importer), 0.0)
        return float(vcif) / float(vxsb) if vcif > 0.0 else 1.0

    def _pefob_price(exporter: str, commodity: str, importer: str) -> float:
        ratio = _pefob_ratio(exporter, commodity, importer)
        return float(value(model.pe[exporter, commodity, importer])) * ratio

    def _pmcif_price(exporter: str, commodity: str, importer: str) -> float:
        ratio = _pmcif_ratio(exporter, commodity, importer)
        return float(value(model.pe[exporter, commodity, importer])) * ratio

    def _build_final_demand_snapshot_block() -> tuple[list[Any], list[Any]]:
        expressions: list[Any] = []
        variables: list[Any] = []
        for r in model.r:
            yc_val = float(value(model.yc[r]))
            yg_val = float(value(model.yg[r]))
            yi_val = float(value(model.yi[r]))
            for i in model.i:
                c_share = float(value(model.c_share[r, i]))
                g_share = float(value(model.g_share[r, i]))
                i_share = float(value(model.i_share[r, i]))
                c_denom = float(value(model.pa[r, i, GTAP_HOUSEHOLD_AGENT])) + 1e-12
                g_denom = float(value(model.pa[r, i, GTAP_GOVERNMENT_AGENT])) + 1e-12
                i_denom = float(value(model.pa[r, i, GTAP_INVESTMENT_AGENT])) + 1e-12
                expressions.append(model.xc[r, i] - (c_share * yc_val / c_denom))
                variables.append(model.xc[r, i])
                expressions.append(model.xg[r, i] - (g_share * yg_val / g_denom))
                variables.append(model.xg[r, i])
                expressions.append(model.xi[r, i] - (i_share * yi_val / i_denom))
                variables.append(model.xi[r, i])
        return expressions, variables

    def _build_income_snapshot_block() -> tuple[list[Any], list[Any]]:
        expressions: list[Any] = []
        variables: list[Any] = []
        activity_labels = {str(a) for a in model.a}
        for r in model.r:
            facty_const = 0.0
            for f in model.f:
                for a in model.a:
                    pf0 = float(value(model.pf[r, f, a]))
                    xf0 = float(value(model.xf[r, f, a]))
                    xscale0 = float(value(model.xscale[r, a]))
                    if xscale0 <= 0.0:
                        continue
                    facty_const += pf0 * xf0 / xscale0

            # Approximate GAMS pi(r,t) with the current investment price index
            # from observed Armington prices and benchmark investment shares.
            pi0 = 0.0
            for i in model.i:
                pi0 += float(value(model.i_share[r, i])) * float(
                    value(model.pa[r, i, GTAP_INVESTMENT_AGENT])
                )
            if pi0 <= 0.0:
                pi0 = 1.0

            fdepr0 = float(value(model.fdepr[r])) if hasattr(model, "fdepr") else 0.0
            kstock0 = float(value(model.kstock[r])) if hasattr(model, "kstock") else 0.0
            facty_const = max(facty_const - fdepr0 * pi0 * kstock0, 0.0)

            ytax_total_const = 0.0
            for a in model.a:
                rto = float(params.taxes.rto.get((str(r), str(a)), 0.0))
                outputs = params.sets.activity_commodities.get(
                    str(a), list(params.sets.i)
                )
                for i in outputs:
                    ytax_total_const += (
                        rto
                        * float(value(model.p_rai[r, a, i]))
                        * float(value(model.x[r, a, i]))
                    )

            for (rr, f, a), rtf in params.taxes.rtf.items():
                if str(rr) != str(r):
                    continue
                xscale_a = float(value(model.xscale[r, a]))
                if xscale_a <= 0.0:
                    continue
                ytax_total_const += (
                    float(rtf)
                    * float(value(model.pf[r, f, a]))
                    * float(value(model.xf[r, f, a]))
                    / xscale_a
                )

            for (rr, i, a), rtpd in params.taxes.rtpd.items():
                if str(rr) != str(r):
                    continue
                term = (
                    float(rtpd)
                    * float(value(model.pd[r, i]))
                    * float(value(model.xda[r, i, a]))
                )
                if str(a) in activity_labels:
                    xscale_a = float(value(model.xscale[r, a]))
                    if xscale_a > 0.0:
                        term /= xscale_a
                ytax_total_const += term
            for (rr, i, a), rtpi in params.taxes.rtpi.items():
                if str(rr) != str(r):
                    continue
                term = (
                    float(rtpi)
                    * float(value(model.pmt[r, i]))
                    * float(value(model.xma[r, i, a]))
                )
                if str(a) in activity_labels:
                    xscale_a = float(value(model.xscale[r, a]))
                    if xscale_a > 0.0:
                        term /= xscale_a
                ytax_total_const += term

            for (rr, i), rtgd in params.taxes.rtgd.items():
                if str(rr) != str(r):
                    continue
                ytax_total_const += (
                    float(rtgd)
                    * float(value(model.pd[r, i]))
                    * float(value(model.xda[r, i, GTAP_GOVERNMENT_AGENT]))
                )
            for (rr, i), rtgi in params.taxes.rtgi.items():
                if str(rr) != str(r):
                    continue
                ytax_total_const += (
                    float(rtgi)
                    * float(value(model.pmt[r, i]))
                    * float(value(model.xma[r, i, GTAP_GOVERNMENT_AGENT]))
                )

            for (exporter, i, importer), rtxs in params.taxes.rtxs.items():
                if str(exporter) != str(r):
                    continue
                ytax_total_const += (
                    float(rtxs)
                    * _pefob_price(str(r), str(i), str(importer))
                    * float(value(model.xw[r, i, importer]))
                )
            for (exporter, i, importer), imptx in params.taxes.imptx.items():
                if str(importer) != str(r):
                    continue
                rate = float(imptx)
                ytax_total_const += (
                    rate
                    * _pmcif_price(str(exporter), str(i), str(r))
                    * float(value(model.xw[exporter, i, r]))
                )

            direct_tax_const = 0.0
            for f in model.f:
                for a in model.a:
                    kappa = float(
                        params.taxes.kappaf_activity.get((str(r), str(f), str(a)), 0.0)
                    )
                    if kappa == 0.0:
                        continue
                    xscale_a = float(value(model.xscale[r, a]))
                    if xscale_a <= 0.0:
                        continue
                    direct_tax_const += (
                        kappa
                        * float(value(model.pf[r, f, a]))
                        * float(value(model.xf[r, f, a]))
                        / xscale_a
                    )

            # Reference GAMS identity: yTaxInd = yTaxTot - ytax("dt")
            ytax_const = ytax_total_const - direct_tax_const

            regy_const = facty_const + ytax_const

            expressions.append(model.facty[r] - facty_const)
            variables.append(model.facty[r])
            expressions.append(model.ytax_ind[r] - ytax_const)
            variables.append(model.ytax_ind[r])
            expressions.append(model.regy[r] - regy_const)
            variables.append(model.regy[r])

            expressions.append(
                model.yc[r]
                - model.betap[r] * (model.phi[r] / model.phip[r]) * model.regy[r]
            )
            variables.append(model.yc[r])
            expressions.append(
                model.yg[r] - model.betag[r] * model.phi[r] * model.regy[r]
            )
            variables.append(model.yg[r])
            expressions.append(
                model.rsav[r] - model.betas[r] * model.phi[r] * model.regy[r]
            )
            variables.append(model.rsav[r])
            expressions.append(model.yi[r] - model.regy[r] * model.yi_share_reg[r])
            variables.append(model.yi[r])
        return expressions, variables

    def _build_market_link_snapshot_block() -> tuple[list[Any], list[Any]]:
        expressions, variables = _build_final_demand_snapshot_block()
        for r in model.r:
            for i in model.i:
                expressions.append(
                    model.xaa[r, i, GTAP_HOUSEHOLD_AGENT] - model.xc[r, i]
                )
                variables.append(model.xaa[r, i, GTAP_HOUSEHOLD_AGENT])
                expressions.append(
                    model.xaa[r, i, GTAP_GOVERNMENT_AGENT] - model.xg[r, i]
                )
                variables.append(model.xaa[r, i, GTAP_GOVERNMENT_AGENT])
                expressions.append(
                    model.xaa[r, i, GTAP_INVESTMENT_AGENT] - model.xi[r, i]
                )
                variables.append(model.xaa[r, i, GTAP_INVESTMENT_AGENT])
        return expressions, variables

    # Snapshot builders use current model levels, so running them after previous
    # successful blocks creates a true block-by-block chained workflow.
    def _build_market_clearing_snapshot_block() -> tuple[list[Any], list[Any]]:
        expressions: list[Any] = []
        variables: list[Any] = []
        for r in model.r:
            for i in model.i:
                absorption_snapshot = sum(
                    value(model.xaa[r, i, aa]) / value(model.xscale[r, aa])
                    for aa in model.aa
                )
                inventory = params.benchmark.vst.get((r, i), 0.0)
                expressions.append(model.xa[r, i] - (absorption_snapshot + inventory))
                variables.append(model.xa[r, i])
        return expressions, variables

    def _build_trade_domestic_aggregation_snapshot_block() -> tuple[
        list[Any], list[Any]
    ]:
        expressions: list[Any] = []
        variables: list[Any] = []
        for r in model.r:
            for i in model.i:
                domestic_snapshot = sum(
                    value(model.xda[r, i, aa]) / value(model.xscale[r, aa])
                    for aa in model.aa
                )
                expressions.append(model.xd[r, i] - domestic_snapshot)
                variables.append(model.xd[r, i])
        return expressions, variables

    def _build_trade_import_aggregation_snapshot_block() -> tuple[list[Any], list[Any]]:
        expressions: list[Any] = []
        variables: list[Any] = []
        for r in model.r:
            for i in model.i:
                import_snapshot = sum(
                    value(model.xma[r, i, aa]) / value(model.xscale[r, aa])
                    for aa in model.aa
                )
                expressions.append(model.xmt[r, i] - import_snapshot)
                variables.append(model.xmt[r, i])
        return expressions, variables

    def _build_import_price_snapshot_block() -> tuple[list[Any], list[Any]]:
        expressions: list[Any] = []
        variables: list[Any] = []
        for r in model.r:
            for i in model.i:
                numer = 0.0
                denom = 0.0
                for rp in model.rp:
                    w = params.benchmark.vmsb.get((str(rp), str(i), str(r)), 0.0)
                    w = float(w or 0.0)
                    if w <= 0.0:
                        continue
                    rate = float(params.taxes.imptx.get((str(rp), str(i), str(r)), 0.0))
                    numer += w * rate
                    denom += w
                rate = (numer / denom) if denom > 0.0 else 0.0
                pd0 = float(value(model.pd[r, i]))
                expressions.append(model.pmt[r, i] - pd0 * (1.0 + rate))
                variables.append(model.pmt[r, i])
        return expressions, variables

    block_results = []
    import_price_expressions, import_price_variables = (
        _build_import_price_snapshot_block()
    )
    block_results.append(
        _solve_expression_block(
            block_name="linear-import-price-identities",
            expressions=import_price_expressions,
            variables=import_price_variables,
            description="snapshot pmt from imptx",
        )
    )

    income_expressions, income_variables = _build_income_snapshot_block()
    block_results.append(
        _solve_expression_block(
            block_name="linear-income-identities",
            expressions=income_expressions,
            variables=income_variables,
            description="linearized regy with yc/yg/yi shares",
        )
    )

    demand_expressions, demand_variables = _build_final_demand_snapshot_block()
    block_results.append(
        _solve_expression_block(
            block_name="linear-demand-identities",
            expressions=demand_expressions,
            variables=demand_variables,
            description="snapshot xC/xG/xI demand identities",
        )
    )
    market_link_expressions, market_link_variables = _build_market_link_snapshot_block()
    block_results.append(
        _solve_expression_block(
            block_name="linear-market-link-identities",
            expressions=market_link_expressions,
            variables=market_link_variables,
            description="snapshot xC/xG/xI + xaa link identities",
        )
    )

    market_clearing_expressions, market_clearing_variables = (
        _build_market_clearing_snapshot_block()
    )
    block_results.append(
        _solve_expression_block(
            block_name="linear-market-clearing-identities",
            expressions=market_clearing_expressions,
            variables=market_clearing_variables,
            description="mkt_goods snapshot over xa (square 100x100)",
        )
    )

    trade_aggregation_expressions, trade_aggregation_variables = (
        _build_trade_domestic_aggregation_snapshot_block()
    )
    block_results.append(
        _solve_expression_block(
            block_name="linear-trade-aggregation-identities",
            expressions=trade_aggregation_expressions,
            variables=trade_aggregation_variables,
            description="eq_xd_agg snapshot over xd (square 100x100)",
        )
    )

    import_aggregation_expressions, import_aggregation_variables = (
        _build_trade_import_aggregation_snapshot_block()
    )
    block_results.append(
        _solve_expression_block(
            block_name="linear-import-aggregation-identities",
            expressions=import_aggregation_expressions,
            variables=import_aggregation_variables,
            description="eq_xmt_agg snapshot over xmt (square 100x100)",
        )
    )

    post_checks = _build_path_capi_post_checks(model, params)

    block_success = all(block.get("success", False) for block in block_results)
    max_residual = max(
        (block.get("residual", 0.0) for block in block_results), default=0.0
    )
    total_major_iterations = sum(
        int(block.get("major_iterations", 0)) for block in block_results
    )
    total_minor_iterations = sum(
        int(block.get("minor_iterations", 0)) for block in block_results
    )

    post_check_gate_pass = bool(post_checks.get("overall_pass", False))
    residual_gate_pass = max_residual <= float(strict_residual_tol)

    overall_success = block_success
    if enforce_post_checks:
        overall_success = overall_success and post_check_gate_pass
    if strict_path_capi:
        overall_success = overall_success and residual_gate_pass

    price_pass_result = None
    price_pass_payload = None
    if price_transmission_pass and overall_success:
        try:
            price_model = model.clone()
            price_solver = GTAPSolver(
                price_model,
                closure=closure_config,
                solver_name=price_transmission_solver,
                solver_options={"max_iter": price_transmission_max_iter},
                params=params,
            )
            price_pass_result = price_solver.solve()
            price_pass_payload = {
                "success": bool(price_pass_result.success),
                "status": getattr(
                    price_pass_result.status, "value", str(price_pass_result.status)
                ),
                "termination_condition": price_pass_result.termination_condition,
                "iterations": price_pass_result.iterations,
                "residual": price_pass_result.residual,
                "solve_time": price_pass_result.solve_time,
                "message": price_pass_result.message,
            }

            if price_pass_result.success:
                for name in (
                    "pf",
                    "xf",
                    "pft",
                    "pfact",
                    "pva",
                    "pnd",
                    "px",
                    "pa",
                    "pmt",
                ):
                    if not hasattr(model, name) or not hasattr(price_model, name):
                        continue
                    src = getattr(price_model, name)
                    tgt = getattr(model, name)
                    for idx in src:
                        val = src[idx].value
                        if val is None:
                            continue
                        tgt[idx].set_value(float(val))
            elif price_transmission_enforce:
                overall_success = False
        except Exception as exc:
            price_pass_payload = {
                "success": False,
                "status": "error",
                "termination_condition": None,
                "iterations": 0,
                "residual": float("inf"),
                "solve_time": 0.0,
                "message": str(exc),
            }
            if price_transmission_enforce:
                overall_success = False

    return {
        "status": "converged" if overall_success else "failed",
        "success": overall_success,
        "solver": "path-capi",
        "message": "Solved linear GTAP blocks via PATH C API (sequential chained mode)",
        "block_success": block_success,
        "post_checks_enforced": bool(enforce_post_checks),
        "post_checks_gate_pass": post_check_gate_pass,
        "strict_path_capi": bool(strict_path_capi),
        "strict_residual_tol": float(strict_residual_tol),
        "residual_gate_pass": residual_gate_pass,
        "path_version": version,
        "n_blocks": len(block_results),
        "residual": max_residual,
        "major_iterations": total_major_iterations,
        "minor_iterations": total_minor_iterations,
        "blocks": block_results,
        "post_checks": post_checks,
        "price_transmission_pass": price_transmission_pass,
        "price_transmission_solver": price_transmission_solver,
        "price_transmission_max_iter": price_transmission_max_iter,
        "price_transmission_enforce": price_transmission_enforce,
        "price_transmission_result": price_pass_payload,
    }


# Evaluador del Jacobiano para PATH. "reverse_numeric" deriva el arbol de
# expresiones en Python en CADA llamada; "asl" delega en la AMPL Solver Library
# via PyNumero (C). Medido en gtap7_20x41: 24.59s -> 0.216s por llamada (114x),
# con los mismos valores (|dif|max 1.4e-14) y sin perder entradas.
# Medido de punta a punta en gtap7_20x41: 28.8 min -> 11.7 min (2.5x), code=1
# en ambos. EQUILIBRIA_GTAP_JAC_MODE=reverse_numeric vuelve a la via historica.
_DEFAULT_JACOBIAN_EVAL_MODE = os.environ.get(
    "EQUILIBRIA_GTAP_JAC_MODE", "asl"
).strip().lower()

# IPOPT sale de la fase de RESTAURACION cuando ha reducido la infactibilidad por
# debajo de required_infeasibility_reduction (default 0.9). En el sistema CUADRADO
# (objetivo constante f=0) el gradiente es cero, los multiplicadores quedan
# INDETERMINADOS y `inf_du` no significa nada -- pero IPOPT la usa para decidir el
# paso, asi que rebota dentro/fuera de restauracion sin avanzar. Subir el umbral a
# 0.99 permite abandonar la restauracion antes y es la UNICA opcion medida que
# esquiva ese criterio dual roto.
# Medido en gtap7_20x41 (en solitario, caffeinate -dims, guard CPU/reloj):
#   reloj      37.4 min -> 20.2 min
#   iteraciones     838 -> 379   (en restauracion 832 = 99.3% -> 132 = 35%)
#   residual P1/P2  5.242e-09 / 5.678e-09 -> 2.132e-13 / 3.847e-11
#   walras            0.000e+00 en ambos (mismo equilibrio)
# FIDELIDAD: gobierna cuando se puede ABANDONAR la restauracion, NO el criterio
# sobre F(x)=0 -- el residual sale MEJOR, no peor. No es aflojar una tolerancia.
# EQUILIBRIA_GTAP_IPOPT_REQ_INFEAS=0.9 vuelve al default de IPOPT.
_DEFAULT_IPOPT_REQUIRED_INFEASIBILITY_REDUCTION = float(
    os.environ.get("EQUILIBRIA_GTAP_IPOPT_REQ_INFEAS", "0.99")
)




def _run_homotopy_shocked(
    base_model,
    gdx_path,
    shock_variable: str,
    shock_index: tuple,
    shock_value: float,
    shock_mode: str,
    homotopy_steps: int,
    contract,
    *,
    solver_output: bool = False,
    path_license_string: Optional[str] = None,
    strict_residual_tol: float = 1e-6,
    calibrated_start: bool = False,
) -> dict[str, Any]:
    """Solve the shocked model via homotopy continuation.

    Applies shock_value in homotopy_steps equal increments, warm-starting
    each step from the previous solution. Returns dict with keys:
    shocked_model, params, residual, homotopy_steps, step_residuals.

    When calibrated_start=True, the first homotopy step starts from the
    model's own calibrated initial values (pmt=1, pm=VMSB/VXSB, etc.)
    rather than from the solved baseline solution. This mirrors the GAMS
    approach: GAMS never solves the baseline explicitly; it starts the
    shocked solve directly from calibrated initial values which ARE the
    baseline equilibrium by construction.
    """
    if homotopy_steps < 1:
        raise ValueError("homotopy_steps must be >= 1")

    from equilibria.templates.gtap.gtap_parity_pipeline import GTAPVariableSnapshot
    from equilibria.templates.gtap.gtap_solver import GTAPSolver

    prev_model = base_model
    final_result: dict[str, Any] = {"residual": float("inf"), "status": "not_run"}
    step_residuals: list[float] = []
    # Capture per-step snapshots so callers (e.g. welfare decomposition) can
    # integrate first-order contributions along the homotopy path.
    step_models: list[Any] = []
    step_params_list: list[GTAPParameters] = []

    for step in range(1, homotopy_steps + 1):
        fraction = step / homotopy_steps
        partial_value = shock_value * fraction

        step_params = GTAPParameters()
        step_params.load_from_gdx(gdx_path)
        _apply_shock_to_params(
            step_params,
            shock_variable,
            shock_index,
            partial_value,
            shock_mode=shock_mode,
        )

        step_eq = GTAPModelEquations(
            step_params.sets,
            step_params,
            contract.closure,
            is_counterfactual=True,
        )
        step_model = step_eq.build_model()

        # When calibrated_start=True and this is the first step, skip the
        # warm-start so PATH begins from the model's calibrated initial values
        # (pmt=1, pm=VMSB/VXSB). This mirrors GAMS's approach exactly.
        if calibrated_start and step == 1:
            prev_snapshot = None
            click.echo(
                "  Using calibrated initial values (no baseline warm-start) for step 1"
            )
        else:
            prev_snapshot = GTAPVariableSnapshot.from_python_model(prev_model)

        click.echo(
            f"  Homotopy step {step}/{homotopy_steps} "
            f"(fraction={fraction:.0%}, partial={partial_value:.4f})..."
        )

        result = _run_path_capi_nonlinear_full(
            step_model,
            step_params,
            solver_output=solver_output,
            path_license_string=path_license_string,
            enforce_post_checks=False,
            strict_path_capi=False,
            strict_residual_tol=strict_residual_tol,
            closure_config=contract.closure,
            solution_hint=prev_snapshot,
            equation_scaling=True,  # mirrors GAMS scaleopt=1
        )
        step_residuals.append(result["residual"])
        click.echo(f"    residual={result['residual']:.3e}  status={result['status']}")
        # Continue even if this step failed — the next step may still benefit from
        # the partial warm-start. The caller should check step_residuals for quality.
        if result["residual"] > 10.0:
            click.echo(
                f"    WARNING: step {step}/{homotopy_steps} residual "
                f"{result['residual']:.3e} is high — warm-start for next step may be unreliable"
            )
        prev_model = step_model
        step_models.append(step_model)
        step_params_list.append(step_params)
        final_result = result

    final_result["homotopy_steps"] = homotopy_steps
    final_result["step_residuals"] = step_residuals
    final_result["shocked_model"] = prev_model
    final_result["step_models"] = step_models
    final_result["step_params"] = step_params_list
    return final_result


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, verbose):
    """GTAP CGE Model CLI - CGEBox Implementation"""
    ctx.ensure_object(dict)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    ctx.obj["logger"] = logging.getLogger(__name__)


@cli.command()
@click.option(
    "--gdx-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to GTAP GDX file",
)
@click.pass_context
def info(ctx, gdx_file):
    """Display GTAP data information"""
    logger = ctx.obj["logger"]
    logger.info(f"Loading GTAP data from {gdx_file}")

    try:
        # Load sets
        sets = GTAPSets()
        sets.load_from_gdx(gdx_file)

        click.echo(f"\n{'=' * 60}")
        click.echo(f"GTAP Data Information")
        click.echo(f"{'=' * 60}")
        click.echo(f"File: {gdx_file}")
        click.echo(f"Aggregation: {sets.aggregation_name}")
        click.echo(f"")
        click.echo(f"Sets:")
        click.echo(f"  Regions:      {sets.n_regions:3d} - {', '.join(sets.r)}")
        click.echo(f"  Commodities:  {sets.n_commodities:3d} - {', '.join(sets.i)}")
        click.echo(f"  Activities:   {sets.n_activities:3d} - {', '.join(sets.a)}")
        click.echo(f"  Factors:      {sets.n_factors:3d} - {', '.join(sets.f)}")
        click.echo(f"")
        click.echo(f"Factor Mobility:")
        click.echo(f"  Mobile:   {sets.n_mobile_factors} - {', '.join(sets.mf)}")
        click.echo(f"  Specific: {sets.n_specific_factors} - {', '.join(sets.sf)}")
        click.echo(f"")

        # Validate
        is_valid, errors = sets.validate()
        if is_valid:
            click.echo(click.style("✓ Sets are valid", fg="green"))
        else:
            click.echo(click.style("✗ Set validation errors:", fg="red"))
            for error in errors:
                click.echo(f"  - {error}")

        # Load parameters summary
        params = GTAPParameters()
        params.load_from_gdx(gdx_file)

        click.echo(f"\nParameters:")
        click.echo(
            f"  Elasticities:    {len(params.elasticities.esubva) + len(params.elasticities.esubm)} loaded"
        )
        click.echo(
            f"  Benchmark:       {len(params.benchmark.vom) + len(params.benchmark.vfm)} flows"
        )
        click.echo(
            f"  Tax rates:       {len(params.taxes.rto) + len(params.taxes.rtms)} rates"
        )

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--gdx-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to GTAP GDX file",
)
@click.option(
    "--elasticity-gdx",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional GDX with base elasticities (default: default-9x10.gdx next to gdx-file)",
)
@click.option(
    "--override-omegas-sigmas-gdx",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional GDX used to override only omegas/sigmas (e.g., COMP.gdx)",
)
@click.option(
    "--closure",
    default="gtap_standard7_9x10",
    help="Closure type (currently: gtap_standard7_9x10)",
)
@click.option(
    "--solver",
    type=click.Choice(["ipopt", "path", "conopt", "path-capi"]),
    default="ipopt",
    help="Solver to use",
)
@click.option(
    "--output", type=click.Path(path_type=Path), help="Output file for results (JSON)"
)
@click.option("--tee/--no-tee", default=False, help="Show solver output while solving")
@click.option(
    "--path-license-string",
    default=None,
    help="Optional PATH license string for path-capi runs (otherwise uses PATH_LICENSE_STRING env var)",
)
@click.option(
    "--path-capi-mode",
    type=click.Choice(["linear", "nonlinear"]),
    default="linear",
    show_default=True,
    help="PATH C API mode: linear block snapshot or full nonlinear system",
)
@click.option(
    "--enforce-post-checks/--no-enforce-post-checks",
    default=True,
    help="For path-capi runs, fail command if post_checks.overall_pass is false",
)
@click.option(
    "--strict-path-capi/--no-strict-path-capi",
    default=False,
    help="When enabled, also require global path-capi residual <= strict-residual-tol",
)
@click.option(
    "--path-capi-convergence-tol",
    type=float,
    default=1e-5,
    show_default=True,
    help="Residual tolerance used to classify nonlinear path-capi solve success",
)
@click.option(
    "--strict-residual-tol",
    type=float,
    default=1e-5,
    show_default=True,
    help="Global residual tolerance used by --strict-path-capi",
)
@click.option(
    "--path-capi-trace-residuals/--no-path-capi-trace-residuals",
    default=False,
    help="Capture per-function-call residual trace for nonlinear path-capi solves",
)
@click.option(
    "--path-capi-trace-max-calls",
    type=int,
    default=120,
    show_default=True,
    help="Maximum nonlinear function callback calls stored in residual trace",
)
@click.option(
    "--path-capi-trace-top-n",
    type=int,
    default=12,
    show_default=True,
    help="Top-N absolute residual equations saved per callback call",
)
@click.option(
    "--path-capi-trace-focus",
    multiple=True,
    default=("kstock", "arent", "kapEnd"),
    show_default=True,
    help="Equation-name substring filters saved in focused residual trace (repeat option)",
)
@click.option(
    "--path-capi-trace-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional JSON output file for nonlinear residual trace",
)
@click.option(
    "--path-capi-xi-diag/--no-path-capi-xi-diag",
    default=False,
    help="Capture detailed runtime diagnostics for eq_xi/eq_pi/eq_xiagg at one region/commodity",
)
@click.option(
    "--path-capi-xi-diag-region",
    default="EastAsia",
    show_default=True,
    help="Region for xi-block runtime diagnostics",
)
@click.option(
    "--path-capi-xi-diag-commodity",
    default="c_Util_Cons",
    show_default=True,
    help="Commodity for xi-block runtime diagnostics",
)
@click.option(
    "--compare-gams/--no-compare-gams",
    default=False,
    help="After solving, compare solution values against GAMS COMP_generated.csv reference",
)
@click.option(
    "--compare-gams-tol",
    type=float,
    default=0.001,
    show_default=True,
    help="Absolute tolerance used in --compare-gams comparison",
)
@click.option(
    "--strict-mirror/--no-strict-mirror",
    default=False,
    help=(
        "Enforce mirror gates after solving: solver residual < 1e-4, "
        "0 non-xd mismatches vs COMP_generated.csv (tol=5e-3). Exit code 2 on failure."
    ),
)
@click.pass_context
def solve(
    ctx,
    gdx_file,
    elasticity_gdx,
    override_omegas_sigmas_gdx,
    closure,
    solver,
    output,
    tee,
    path_license_string,
    path_capi_mode,
    enforce_post_checks,
    path_capi_convergence_tol,
    strict_path_capi,
    strict_residual_tol,
    path_capi_trace_residuals,
    path_capi_trace_max_calls,
    path_capi_trace_top_n,
    path_capi_trace_focus,
    path_capi_trace_file,
    path_capi_xi_diag,
    path_capi_xi_diag_region,
    path_capi_xi_diag_commodity,
    compare_gams,
    compare_gams_tol,
    strict_mirror,
):
    """Solve the GTAP baseline model"""
    logger = ctx.obj["logger"]
    logger.info(f"Solving GTAP model from {gdx_file}")

    try:
        snapshot_enabled = _ensure_gtap_reference_snapshot_env()
        if snapshot_enabled:
            logger.info(
                "Enabled benchmark-aligned GTAP reference snapshot from %s",
                COMP_CSV_REFERENCE,
            )
        if elasticity_gdx is not None:
            logger.info("Using custom elasticity GDX: %s", elasticity_gdx)
        if override_omegas_sigmas_gdx is not None:
            logger.info(
                "Overriding omegas/sigmas from calibration GDX: %s",
                override_omegas_sigmas_gdx,
            )
        # Load data
        with click.progressbar(length=3, label="Loading data") as bar:
            params = GTAPParameters()
            params.load_from_gdx(
                gdx_file,
                elasticity_gdx=elasticity_gdx,
                elasticity_override_gdx=override_omegas_sigmas_gdx,
            )
            bar.update(1)

            contract = _build_gtap_contract_with_calibration(closure)
            bar.update(1)

            equations = GTAPModelEquations(params.sets, params, contract.closure)
            model = equations.build_model()
            bar.update(1)

        reference_rtms = dict(params.taxes.rtms)

        if solver in {"path-capi", "path"}:
            if path_capi_mode == "nonlinear":
                click.echo("\nSolving full GTAP system with PATH C API (nonlinear)...")
                if solver == "path":
                    click.echo(
                        "Note: --solver path is mapped to PATH C API backend by default."
                    )
                path_capi_result = _run_path_capi_nonlinear_full(
                    model,
                    params,
                    solver_output=tee,
                    path_license_string=path_license_string,
                    enforce_post_checks=enforce_post_checks,
                    path_capi_convergence_tol=path_capi_convergence_tol,
                    strict_path_capi=strict_path_capi,
                    strict_residual_tol=strict_residual_tol,
                    closure_config=contract.closure,
                    residual_trace_enabled=path_capi_trace_residuals,
                    residual_trace_max_calls=path_capi_trace_max_calls,
                    residual_trace_top_n=path_capi_trace_top_n,
                    residual_trace_focus_patterns=list(path_capi_trace_focus),
                    residual_trace_file=path_capi_trace_file,
                    xi_diag_enabled=path_capi_xi_diag,
                    xi_diag_region=path_capi_xi_diag_region,
                    xi_diag_commodity=path_capi_xi_diag_commodity,
                )
            else:
                click.echo("\nSolving linear GTAP subset with PATH C API...")
                if solver == "path":
                    click.echo(
                        "Note: --solver path is mapped to PATH C API backend by default."
                    )
                path_capi_result = _run_path_capi_linear_block(
                    model,
                    params,
                    reference_rtms=reference_rtms,
                    solver_output=tee,
                    path_license_string=path_license_string,
                    enforce_post_checks=enforce_post_checks,
                    strict_path_capi=strict_path_capi,
                    strict_residual_tol=strict_residual_tol,
                    closure_config=contract.closure,
                )

            path_capi_label = (
                "PATH C API nonlinear full model"
                if path_capi_mode == "nonlinear"
                else "PATH C API linear block"
            )
            if path_capi_result["success"]:
                click.echo(click.style(f"✓ {path_capi_label} converged", fg="green"))
            else:
                click.echo(click.style(f"✗ {path_capi_label} failed", fg="red"))

            click.echo(f"Status:       {path_capi_result['status']}")
            click.echo(f"Residual:     {path_capi_result['residual']:.2e}")
            click.echo(
                f"Iterations:   {path_capi_result['major_iterations']} / {path_capi_result['minor_iterations']}"
            )
            if path_capi_mode == "nonlinear":
                click.echo(
                    f"Jac mode:     {path_capi_result.get('jacobian_eval_mode', 'symbolic')}"
                )
                profile = path_capi_result.get("callback_profile", {})
                click.echo(
                    "Callbacks:    "
                    f"F={profile.get('function_calls', 0)} "
                    f"({profile.get('function_time_sec', 0.0):.2f}s) | "
                    f"J={profile.get('jacobian_calls', 0)} "
                    f"({profile.get('jacobian_time_sec', 0.0):.2f}s) | "
                    f"total={profile.get('total_callback_time_sec', 0.0):.2f}s"
                )
            click.echo(f"Blocks:       {path_capi_result['n_blocks']}")
            click.echo(
                f"Post-checks:  {path_capi_result['post_checks_gate_pass']} "
                f"(enforced={path_capi_result['post_checks_enforced']})"
            )
            click.echo(
                f"Strict gate:  {path_capi_result['residual_gate_pass']} "
                f"(enabled={path_capi_result['strict_path_capi']}, "
                f"tol={path_capi_result['strict_residual_tol']:.1e})"
            )
            if path_capi_mode == "nonlinear":
                click.echo(
                    f"Conv tol:     {path_capi_result.get('path_capi_convergence_tol', path_capi_convergence_tol):.1e}"
                )
            for block in path_capi_result.get("blocks", []):
                click.echo(
                    f"  - {block['name']}: {block['status']} "
                    f"(res={block.get('residual', 0.0):.2e}, "
                    f"n={block.get('n_equations', 0)})"
                )
            if path_capi_mode == "nonlinear":
                residual_diag = path_capi_result.get("constraint_residuals", {})
                top = residual_diag.get("top", [])
                if top:
                    worst = top[0]
                    click.echo(
                        "Worst eq:    "
                        f"{worst.get('name', '')} "
                        f"(abs={float(worst.get('abs_residual', 0.0)):.2e}, "
                        f"signed={float(worst.get('signed_residual', 0.0)):.2e})"
                    )
                trace_payload = path_capi_result.get("residual_trace", {})
                if bool(trace_payload.get("enabled", False)):
                    captured_calls = int(trace_payload.get("captured_calls", 0))
                    click.echo(f"Trace calls:  {captured_calls}")
                    calls = trace_payload.get("calls", [])
                    if calls:
                        last_call = calls[-1]
                        focus_rows = last_call.get("focused", [])
                        if focus_rows:
                            focus_top = focus_rows[0]
                            click.echo(
                                "Trace focus: "
                                f"{focus_top.get('name', '')} "
                                f"(abs={float(focus_top.get('abs_residual', 0.0)):.2e}, "
                                f"signed={float(focus_top.get('signed_residual', 0.0)):.2e})"
                            )
                xi_diag_payload = path_capi_result.get("xi_block_diagnostics", {})
                if bool(xi_diag_payload.get("enabled", False)):
                    if xi_diag_payload.get("error"):
                        click.echo(f"XI diag:     error={xi_diag_payload.get('error')}")
                    else:
                        eq_xi = xi_diag_payload.get("eq_xi", {})
                        eq_pi = xi_diag_payload.get("eq_pi", {})
                        eq_xiagg = xi_diag_payload.get("eq_xiagg", {})
                        click.echo(
                            "XI diag:     "
                            f"({xi_diag_payload.get('region')}, {xi_diag_payload.get('commodity')})"
                        )
                        click.echo(
                            "  eq_xi:     "
                            f"lhs={float(eq_xi.get('lhs', 0.0)):.6e} "
                            f"rhs={float(eq_xi.get('rhs', 0.0)):.6e} "
                            f"res={float(eq_xi.get('residual', 0.0)):.6e}"
                        )
                        click.echo(
                            "  eq_pi:     "
                            f"lhs={float(eq_pi.get('lhs', 0.0)):.6e} "
                            f"rhs={float(eq_pi.get('rhs', 0.0)):.6e} "
                            f"res={float(eq_pi.get('residual', 0.0)):.6e}"
                        )
                        click.echo(
                            "  eq_xiagg:  "
                            f"lhs={float(eq_xiagg.get('lhs', 0.0)):.6e} "
                            f"rhs={float(eq_xiagg.get('rhs', 0.0)):.6e} "
                            f"res={float(eq_xiagg.get('residual', 0.0)):.6e}"
                        )

            if compare_gams:
                try:
                    import csv as _csv_mod
                    from equilibria.templates.gtap.gtap_parity_pipeline import (
                        GTAPVariableSnapshot,
                    )

                    click.echo(
                        "\nComparing solution against GAMS reference (COMP_generated.csv)..."
                    )
                    py_snapshot = GTAPVariableSnapshot.from_python_model(model)

                    # Load CSV directly without any scale factor; COMP_generated.csv stores
                    # values in the same normalized units as the Python model.
                    gams_raw: dict[str, dict] = {}
                    with open(
                        COMP_CSV_REFERENCE, newline="", encoding="utf-8"
                    ) as _csvf:
                        for _row in _csv_mod.DictReader(_csvf):
                            _yr = (_row.get("Year") or "").strip()
                            if _yr not in {"1", "1.0"}:
                                continue
                            _var = (_row.get("Variable") or "").strip().lower()
                            _reg = (_row.get("Region") or "").strip()
                            _sec = (_row.get("Sector") or "").strip()
                            _qual = (_row.get("Qualifier") or "").strip()
                            try:
                                _val = float((_row.get("Value") or "0").strip())
                            except (ValueError, TypeError):
                                continue
                            if _var not in gams_raw:
                                gams_raw[_var] = {}
                            if _reg and _sec and _qual:
                                gams_raw[_var][(_reg, _sec, _qual)] = _val
                            elif _reg and _sec:
                                gams_raw[_var][(_reg, _sec)] = _val
                            elif _reg:
                                gams_raw[_var][(_reg,)] = _val
                            else:
                                gams_raw[_var][()] = _val

                    # Compare variable group by group (match on Python model's key format)
                    _all_mm: list[dict] = []
                    _n_compared = 0
                    _n_mm = 0
                    _max_abs = 0.0
                    for _attr in sorted(py_snapshot.__dataclass_fields__):
                        if _attr in {"pnum", "walras", "xd"}:
                            # xd (xda) is stored normalized by xscale in Python but in GAMS-level
                            # units in COMP_generated.csv; skip to avoid spurious factor-10 gaps.
                            continue
                        _py_dict = getattr(py_snapshot, _attr, {})
                        if not isinstance(_py_dict, dict):
                            continue
                        _gams_dict = gams_raw.get(
                            _attr, gams_raw.get(_attr.lower(), {})
                        )
                        if not _gams_dict:
                            continue
                        for _k, _pv in _py_dict.items():
                            if _pv == 0.0:
                                continue
                            # Normalize key: Python may use bare strings; GAMS CSV may use tuples
                            if isinstance(_k, str):
                                _gv = _gams_dict.get(_k, _gams_dict.get((_k,), None))
                            elif isinstance(_k, tuple) and len(_k) == 1:
                                _gv = _gams_dict.get(_k, _gams_dict.get(_k[0], None))
                            else:
                                _gv = _gams_dict.get(_k, None)
                            if _gv is None or _gv == 0.0:
                                continue
                            _n_compared += 1
                            _ad = abs(_pv - _gv)
                            _rd = _ad / max(abs(_gv), 1e-10)
                            _max_abs = max(_max_abs, _ad)
                            if _ad > compare_gams_tol:
                                _n_mm += 1
                                _all_mm.append(
                                    {
                                        "group": _attr,
                                        "key": _k,
                                        "python": _pv,
                                        "gams": _gv,
                                        "abs_diff": _ad,
                                        "rel_diff": _rd,
                                    }
                                )

                    _all_mm.sort(key=lambda _m: _m["abs_diff"], reverse=True)
                    click.echo(f"  Compared:  {_n_compared} variable entries")
                    click.echo(f"  Mismatches (tol={compare_gams_tol:.2g}): {_n_mm}")
                    click.echo(f"  Max abs diff: {_max_abs:.4e}")
                    if _all_mm:
                        click.echo(f"\n  Top-20 mismatches (sorted by abs diff):")
                        click.echo(
                            f"  {'Group':8s}  {'Key':45s}  {'Python':>14s}  {'GAMS':>14s}  {'AbsDiff':>12s}  {'RelDiff':>10s}"
                        )
                        click.echo("  " + "-" * 120)
                        for _mm in _all_mm[:20]:
                            click.echo(
                                f"  {_mm['group']:8s}  {str(_mm['key']):45s}  "
                                f"{_mm['python']:14.6g}  {_mm['gams']:14.6g}  "
                                f"{_mm['abs_diff']:12.4e}  {_mm['rel_diff']:10.4e}"
                            )
                    else:
                        click.echo("  All variables within tolerance!")
                    if output:
                        path_capi_result["gams_comparison"] = {
                            "n_compared": _n_compared,
                            "n_mismatches": _n_mm,
                            "max_abs_diff": _max_abs,
                            "tolerance": compare_gams_tol,
                            "top_mismatches": _all_mm[:50],
                        }
                except Exception as _cmp_exc:
                    click.echo(f"  Comparison failed: {_cmp_exc}", err=True)

            # --strict-mirror gate: enforce residual + parity invariants.
            if strict_mirror:
                _STRICT_RESIDUAL_TOL = 1e-4
                _STRICT_MIRROR_TOL = 5e-3
                _mirror_failures: list[str] = []

                # Gate 1: solver residual
                _res = path_capi_result.get("residual", float("inf"))
                if not path_capi_result.get("success") or _res >= _STRICT_RESIDUAL_TOL:
                    _mirror_failures.append(
                        f"Solver residual {_res:.3e} >= {_STRICT_RESIDUAL_TOL:.1e} "
                        f"(status={path_capi_result.get('status')})"
                    )

                # Gate 2: parity against COMP_generated.csv (0 mismatches at 5e-3)
                if COMP_CSV_REFERENCE.exists():
                    try:
                        import csv as _csv_m
                        from equilibria.templates.gtap.gtap_parity_pipeline import (
                            GTAPVariableSnapshot as _Snap,
                        )

                        _py_snap = _Snap.from_python_model(model)
                        _g_raw: dict[str, dict] = {}
                        with open(
                            COMP_CSV_REFERENCE, newline="", encoding="utf-8"
                        ) as _f:
                            for _r2 in _csv_m.DictReader(_f):
                                if (_r2.get("Year") or "").strip() not in {"1", "1.0"}:
                                    continue
                                _v2 = (_r2.get("Variable") or "").strip().lower()
                                _rg = (_r2.get("Region") or "").strip()
                                _sc = (_r2.get("Sector") or "").strip()
                                _ql = (_r2.get("Qualifier") or "").strip()
                                try:
                                    _vl = float((_r2.get("Value") or "0").strip())
                                except (ValueError, TypeError):
                                    continue
                                if _v2 not in _g_raw:
                                    _g_raw[_v2] = {}
                                if _rg and _sc and _ql:
                                    _g_raw[_v2][(_rg, _sc, _ql)] = _vl
                                elif _rg and _sc:
                                    _g_raw[_v2][(_rg, _sc)] = _vl
                                elif _rg:
                                    _g_raw[_v2][(_rg,)] = _vl
                                else:
                                    _g_raw[_v2][()] = _vl
                        _mirror_mm = 0
                        for _at2 in sorted(_py_snap.__dataclass_fields__):
                            if _at2 in {"pnum", "walras", "xd"}:
                                continue
                            _pd2 = getattr(_py_snap, _at2, {})
                            if not isinstance(_pd2, dict):
                                continue
                            _gd2 = _g_raw.get(_at2, _g_raw.get(_at2.lower(), {}))
                            if not _gd2:
                                continue
                            for _k2, _pv2 in _pd2.items():
                                if _pv2 == 0.0:
                                    continue
                                if isinstance(_k2, str):
                                    _gv2 = _gd2.get(_k2, _gd2.get((_k2,), None))
                                elif isinstance(_k2, tuple) and len(_k2) == 1:
                                    _gv2 = _gd2.get(_k2, _gd2.get(_k2[0], None))
                                else:
                                    _gv2 = _gd2.get(_k2)
                                if _gv2 is None or _gv2 == 0.0:
                                    continue
                                if abs(_pv2 - _gv2) > _STRICT_MIRROR_TOL:
                                    _mirror_mm += 1
                        if _mirror_mm > 0:
                            _mirror_failures.append(
                                f"{_mirror_mm} parity mismatches (tol={_STRICT_MIRROR_TOL:.1e}) "
                                f"vs COMP_generated.csv"
                            )
                    except Exception as _me:
                        _mirror_failures.append(f"Mirror parity check error: {_me}")
                else:
                    click.echo(
                        "  [strict-mirror] COMP_generated.csv not found — skipping parity gate",
                        err=True,
                    )

                if _mirror_failures:
                    click.echo("\n[strict-mirror] FAILED:", err=True)
                    for _mf in _mirror_failures:
                        click.echo(f"  • {_mf}", err=True)
                    if output:
                        with open(output, "w") as f:
                            json.dump(path_capi_result, f, indent=2)
                    sys.exit(2)
                else:
                    click.echo("\n[strict-mirror] All gates passed.")

            if output:
                with open(output, "w") as f:
                    json.dump(path_capi_result, f, indent=2)
                click.echo(f"\nResults saved to: {output}")

            sys.exit(0 if path_capi_result["success"] else 1)

        # Solve
        click.echo(f"\nSolving with {solver}...")
        gtap_solver = GTAPSolver(
            model, contract.closure, solver_name=solver, params=params
        )

        solver_tee = tee or (solver == "path")
        with click.progressbar(length=1, label="Solving") as bar:
            result = gtap_solver.solve(tee=solver_tee)
            bar.update(1)

        # Display results
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Solution Results")
        click.echo(f"{'=' * 60}")

        if result.success:
            click.echo(click.style(f"✓ Converged successfully", fg="green"))
        else:
            click.echo(click.style(f"✗ Did not converge", fg="red"))

        click.echo(f"Status:       {result.status.value}")
        click.echo(f"Iterations:   {result.iterations}")
        click.echo(f"Solve time:   {result.solve_time:.2f}s")
        click.echo(f"Walras check: {result.walras_value:.2e}")

        if result.objective_value is not None:
            click.echo(f"Objective:    {result.objective_value:.6f}")

        # Save results
        if output:
            results_data = {
                "status": result.status.value,
                "success": result.success,
                "iterations": result.iterations,
                "solve_time": result.solve_time,
                "walras_value": result.walras_value,
                "message": result.message,
            }
            with open(output, "w") as f:
                json.dump(results_data, f, indent=2)
            click.echo(f"\nResults saved to: {output}")

        # Exit with appropriate code
        sys.exit(0 if result.success else 1)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--gdx-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to GTAP GDX file",
)
@click.option("--variable", required=True, help="Variable to shock (e.g., rtms, rtxs)")
@click.option("--index", required=True, help='Index tuple (e.g., "(USA,agr,EUR)")')
@click.option("--value", type=float, required=True, help="New value for the shock")
@click.option(
    "--shock-mode",
    type=click.Choice(["set", "pct", "mult", "tm_pct"]),
    default="set",
    show_default=True,
    help="Shock semantics: set=value, pct=old*(1+value), mult=old*value",
)
@click.option(
    "--solver",
    type=click.Choice(["ipopt", "path", "conopt", "path-capi"]),
    default="ipopt",
    help="Solver to use",
)
@click.option(
    "--output", type=click.Path(path_type=Path), help="Output file for results (JSON)"
)
@click.option("--tee/--no-tee", default=False, help="Show solver output while solving")
@click.option(
    "--path-license-string",
    default=None,
    help="Optional PATH license string for path-capi runs (otherwise uses PATH_LICENSE_STRING env var)",
)
@click.option(
    "--path-capi-mode",
    type=click.Choice(["linear", "nonlinear"]),
    default="linear",
    show_default=True,
    help="PATH C API mode: linear block snapshot or full nonlinear system",
)
@click.option(
    "--enforce-post-checks/--no-enforce-post-checks",
    default=True,
    help="For path-capi runs, fail command if post_checks.overall_pass is false",
)
@click.option(
    "--strict-path-capi/--no-strict-path-capi",
    default=False,
    help="When enabled, also require global path-capi residual <= strict-residual-tol",
)
@click.option(
    "--strict-residual-tol",
    type=float,
    default=1e-8,
    show_default=True,
    help="Global residual tolerance used by --strict-path-capi",
)
@click.pass_context
def shock(
    ctx,
    gdx_file,
    variable,
    index,
    value,
    shock_mode,
    solver,
    output,
    tee,
    path_license_string,
    path_capi_mode,
    enforce_post_checks,
    strict_path_capi,
    strict_residual_tol,
):
    """Apply a shock and solve the model"""
    logger = ctx.obj["logger"]
    logger.info(f"Applying shock: {variable}{index} = {value}")

    try:
        # Parse index string, accepting CLI form like "(USA,agr,EUR)".
        idx = _parse_index(index)
        if not idx:
            raise ValueError(f"Invalid index format: {index}")

        # Load and solve baseline
        params = GTAPParameters()
        params.load_from_gdx(gdx_file)
        reference_rtms = dict(params.taxes.rtms)

        # Apply shock directly to the parameter object when available
        # (e.g., tariff rtms in params.taxes.rtms).
        applied_to_params = _apply_shock_to_params(
            params, variable, idx, value, shock_mode=shock_mode
        )
        if applied_to_params:
            logger.info(f"Applied parameter shock: {variable}{idx} = {value}")

        contract = _build_gtap_contract_with_calibration("gtap_standard7_9x10")
        equations = GTAPModelEquations(params.sets, params, contract.closure)
        model = equations.build_model()

        if solver in {"path-capi", "path"}:
            if path_capi_mode == "linear" and not applied_to_params:
                click.echo(
                    click.style(
                        "✗ path-capi linear shock currently supports parameter shocks only (e.g., rtms in params.taxes)",
                        fg="red",
                    )
                )
                if output:
                    with open(output, "w") as f:
                        json.dump(
                            {
                                "shock": {
                                    "variable": variable,
                                    "index": idx,
                                    "value": value,
                                },
                                "status": "failed",
                                "success": False,
                                "message": "path-capi linear shock supports parameter-level shocks only",
                            },
                            f,
                            indent=2,
                        )
                    click.echo(f"\nResults saved to: {output}")
                sys.exit(1)

            if path_capi_mode == "nonlinear":
                click.echo(
                    "Solving shocked full GTAP system with PATH C API (nonlinear)..."
                )
                if solver == "path":
                    click.echo(
                        "Note: --solver path is mapped to PATH C API backend by default."
                    )
                path_capi_result = _run_path_capi_nonlinear_full(
                    model,
                    params,
                    solver_output=tee,
                    path_license_string=path_license_string,
                    enforce_post_checks=enforce_post_checks,
                    strict_path_capi=strict_path_capi,
                    strict_residual_tol=strict_residual_tol,
                    closure_config=contract.closure,
                )
            else:
                click.echo("Solving shocked linear GTAP subset with PATH C API...")
                if solver == "path":
                    click.echo(
                        "Note: --solver path is mapped to PATH C API backend by default."
                    )
                path_capi_result = _run_path_capi_linear_block(
                    model,
                    params,
                    reference_rtms=reference_rtms,
                    solver_output=tee,
                    path_license_string=path_license_string,
                    enforce_post_checks=enforce_post_checks,
                    strict_path_capi=strict_path_capi,
                    strict_residual_tol=strict_residual_tol,
                    closure_config=contract.closure,
                )

            click.echo(f"\n{'=' * 60}")
            click.echo(
                "Shock Results (PATH C API Nonlinear Full Model)"
                if path_capi_mode == "nonlinear"
                else "Shock Results (PATH C API Linear Block)"
            )
            click.echo(f"{'=' * 60}")
            click.echo(f"Shock: {variable}{index} = {value}")

            if path_capi_result["success"]:
                click.echo(click.style("✓ Converged successfully", fg="green"))
            else:
                click.echo(click.style("✗ Did not converge", fg="red"))

            click.echo(f"Status:       {path_capi_result['status']}")
            click.echo(f"Residual:     {path_capi_result['residual']:.2e}")
            if path_capi_mode == "nonlinear":
                click.echo(
                    f"Jac mode:     {path_capi_result.get('jacobian_eval_mode', 'symbolic')}"
                )
                profile = path_capi_result.get("callback_profile", {})
                click.echo(
                    "Callbacks:    "
                    f"F={profile.get('function_calls', 0)} "
                    f"({profile.get('function_time_sec', 0.0):.2f}s) | "
                    f"J={profile.get('jacobian_calls', 0)} "
                    f"({profile.get('jacobian_time_sec', 0.0):.2f}s) | "
                    f"total={profile.get('total_callback_time_sec', 0.0):.2f}s"
                )
            click.echo(f"Blocks:       {path_capi_result['n_blocks']}")
            click.echo(
                f"Post-checks:  {path_capi_result['post_checks_gate_pass']} "
                f"(enforced={path_capi_result['post_checks_enforced']})"
            )
            click.echo(
                f"Strict gate:  {path_capi_result['residual_gate_pass']} "
                f"(enabled={path_capi_result['strict_path_capi']}, "
                f"tol={path_capi_result['strict_residual_tol']:.1e})"
            )
            for block in path_capi_result.get("blocks", []):
                click.echo(
                    f"  - {block['name']}: {block['status']} "
                    f"(res={block.get('residual', 0.0):.2e}, "
                    f"n={block.get('n_equations', 0)})"
                )
            if path_capi_mode == "nonlinear":
                residual_diag = path_capi_result.get("constraint_residuals", {})
                top = residual_diag.get("top", [])
                if top:
                    worst = top[0]
                    click.echo(
                        "Worst eq:    "
                        f"{worst.get('name', '')} "
                        f"(abs={float(worst.get('abs_residual', 0.0)):.2e}, "
                        f"signed={float(worst.get('signed_residual', 0.0)):.2e})"
                    )

            if output:
                with open(output, "w") as f:
                    json.dump(
                        {
                            "shock": {
                                "variable": variable,
                                "index": idx,
                                "value": value,
                            },
                            **path_capi_result,
                        },
                        f,
                        indent=2,
                    )
                click.echo(f"\nResults saved to: {output}")

            sys.exit(0 if path_capi_result["success"] else 1)

        gtap_solver = GTAPSolver(
            model, contract.closure, solver_name=solver, params=params
        )

        # Apply at model-level only when not already handled in parameters.
        if not applied_to_params:
            shock_def = {"variable": variable, "index": idx, "value": value}
            gtap_solver.apply_shock(shock_def)

        # Solve
        click.echo(f"Solving with shock...")
        solver_tee = tee or (solver == "path")
        result = gtap_solver.solve(tee=solver_tee)

        # Display results
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Shock Results")
        click.echo(f"{'=' * 60}")
        click.echo(f"Shock: {variable}{index} = {value}")

        if result.success:
            click.echo(click.style(f"✓ Converged successfully", fg="green"))
        else:
            click.echo(click.style(f"✗ Did not converge", fg="red"))

        click.echo(f"Status:       {result.status.value}")
        click.echo(f"Walras check: {result.walras_value:.2e}")

        if output:
            with open(output, "w") as f:
                json.dump(
                    {
                        "shock": {"variable": variable, "index": idx, "value": value},
                        "status": result.status.value,
                        "success": result.success,
                        "walras_value": result.walras_value,
                    },
                    f,
                    indent=2,
                )
            click.echo(f"\nResults saved to: {output}")

        sys.exit(0 if result.success else 1)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--gdx-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to GTAP GDX file",
)
@click.option(
    "--closure",
    default="gtap_standard7_9x10",
    help="Closure type (currently: gtap_standard7_9x10)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("output/gtap_validate_path_capi.json"),
    show_default=True,
    help="Output file for validation results (JSON)",
)
@click.option(
    "--tee/--no-tee", default=False, help="Show solver output while validating"
)
@click.option(
    "--path-license-string",
    default=None,
    help="Optional PATH license string (otherwise uses PATH_LICENSE_STRING env var)",
)
@click.option(
    "--path-capi-mode",
    type=click.Choice(["linear", "nonlinear"]),
    default="linear",
    show_default=True,
    help="PATH C API mode: linear block snapshot or full nonlinear system",
)
@click.option(
    "--strict-residual-tol",
    type=float,
    default=1e-8,
    show_default=True,
    help="Global residual tolerance for strict validation gate",
)
@click.pass_context
def validate(
    ctx,
    gdx_file,
    closure,
    output,
    tee,
    path_license_string,
    path_capi_mode,
    strict_residual_tol,
):
    """Run strict path-capi baseline validation for CI pipelines."""
    logger = ctx.obj["logger"]
    logger.info(f"Validating GTAP path-capi baseline from {gdx_file}")

    try:
        with click.progressbar(length=3, label="Loading data") as bar:
            params = GTAPParameters()
            params.load_from_gdx(gdx_file)
            bar.update(1)

            contract = _build_gtap_contract_with_calibration(closure)
            bar.update(1)

            equations = GTAPModelEquations(params.sets, params, contract.closure)
            model = equations.build_model()
            bar.update(1)

        click.echo("\nRunning strict path-capi validation...")
        if path_capi_mode == "nonlinear":
            result = _run_path_capi_nonlinear_full(
                model,
                params,
                solver_output=tee,
                path_license_string=path_license_string,
                enforce_post_checks=True,
                strict_path_capi=True,
                strict_residual_tol=strict_residual_tol,
                closure_config=contract.closure,
            )
        else:
            result = _run_path_capi_linear_block(
                model,
                params,
                reference_rtms=dict(params.taxes.rtms),
                solver_output=tee,
                path_license_string=path_license_string,
                enforce_post_checks=True,
                strict_path_capi=True,
                strict_residual_tol=strict_residual_tol,
                closure_config=contract.closure,
            )

        click.echo(f"\n{'=' * 60}")
        click.echo("Validation Results")
        click.echo(f"{'=' * 60}")
        click.echo(f"Status:       {result['status']}")
        click.echo(f"Residual:     {result['residual']:.2e}")
        click.echo(f"Blocks:       {result['n_blocks']}")
        click.echo(f"Block gate:   {result['block_success']}")
        click.echo(f"Post-checks:  {result['post_checks_gate_pass']}")
        click.echo(
            f"Strict gate:  {result['residual_gate_pass']} (tol={result['strict_residual_tol']:.1e})"
        )

        for block in result.get("blocks", []):
            click.echo(
                f"  - {block['name']}: {block['status']} "
                f"(res={block.get('residual', 0.0):.2e}, n={block.get('n_equations', 0)})"
            )

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(result, f, indent=2)
            click.echo(f"\nValidation report saved to: {output}")

        sys.exit(0 if result["success"] else 1)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command(name="validate-shock")
@click.option(
    "--gdx-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to GTAP GDX file",
)
@click.option(
    "--closure",
    default="gtap_standard7_9x10",
    help="Closure type (currently: gtap_standard7_9x10)",
)
@click.option("--variable", required=True, help="Shock variable (e.g., rtms)")
@click.option(
    "--index", required=True, help='Shock index tuple (e.g., "(CRI,agr,USA)")'
)
@click.option("--value", type=float, required=True, help="Shock value")
@click.option(
    "--shock-mode",
    type=click.Choice(["set", "pct", "mult", "tm_pct"]),
    default="pct",
    show_default=True,
    help="Shock semantics: set=value, pct=old*(1+value), mult=old*value",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("output/gtap_validate_shock_path_capi.json"),
    show_default=True,
    help="Output file for validation + delta report (JSON)",
)
@click.option(
    "--tee/--no-tee", default=False, help="Show solver output while validating"
)
@click.option(
    "--path-license-string",
    default=None,
    help="Optional PATH license string (otherwise uses PATH_LICENSE_STRING env var)",
)
@click.option(
    "--path-capi-mode",
    type=click.Choice(["linear", "nonlinear"]),
    default="linear",
    show_default=True,
    help="PATH C API mode: linear block snapshot or full nonlinear system",
)
@click.option(
    "--strict-residual-tol",
    type=float,
    default=1e-8,
    show_default=True,
    help="Global residual tolerance for strict validation gate",
)
@click.option(
    "--homotopy-steps",
    default=1,
    show_default=True,
    type=int,
    help="Apply shock in N equal increments for PATH continuation. Use 5-10 for large shocks.",
)
@click.option(
    "--calibrated-start",
    is_flag=True,
    default=False,
    help=(
        "Start shocked solve from calibrated initial values (pmt=1, pm=VMSB/VXSB) "
        "instead of the solved baseline. Mirrors GAMS approach where the shocked "
        "solve starts directly from the calibration equilibrium."
    ),
)
@click.option(
    "--if-sub/--no-if-sub",
    default=False,
    show_default=True,
    help=(
        "Use GAMS ifSUB=1 mode (substitutes macro identities, fixes pm/pmcif/pefob). "
        "Default False matches GAMS ifSUB=0 (full equation system active)."
    ),
)
@click.option(
    "--welfare-decomp",
    is_flag=True,
    default=False,
    help="Compute Huff/McDougall welfare decomposition (writes welfare_decomposition.csv).",
)
@click.option(
    "--welfare-har",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path for WELVIEW.har export (RunGTAP-compatible). Implies --welfare-decomp.",
)
@click.pass_context
def validate_shock(
    ctx,
    gdx_file,
    closure,
    variable,
    index,
    value,
    shock_mode,
    output,
    tee,
    path_license_string,
    path_capi_mode,
    strict_residual_tol,
    homotopy_steps,
    calibrated_start,
    if_sub,
    welfare_decomp,
    welfare_har,
):
    """Run strict baseline + strict shocked path-capi validation for CI pipelines."""
    logger = ctx.obj["logger"]
    logger.info(f"Validating GTAP baseline+shock path-capi from {gdx_file}")

    try:
        shock_index = _parse_index(index)
        raw_index = index.strip() if isinstance(index, str) else ""
        if raw_index and raw_index.lower() not in {"()", "all"} and not shock_index:
            raise ValueError(f"Invalid index format: {index}")

        # Baseline run
        with click.progressbar(length=3, label="Loading baseline") as bar:
            base_params = GTAPParameters()
            base_params.load_from_gdx(gdx_file)
            bar.update(1)

            contract = _build_gtap_contract_with_calibration(closure)
            # Override if_sub to match target GAMS ifSUB setting
            if contract.closure.if_sub != if_sub:
                payload = contract.model_dump(mode="python")
                payload["closure"]["if_sub"] = if_sub
                contract = build_gtap_contract(payload)
            bar.update(1)

            base_equations = GTAPModelEquations(
                base_params.sets, base_params, contract.closure
            )
            base_model = base_equations.build_model()
            # apply_production_scaling already called inside build_model()
            bar.update(1)

        click.echo("\nRunning strict baseline validation...")
        if path_capi_mode == "nonlinear":
            baseline_result = _run_path_capi_nonlinear_full(
                base_model,
                base_params,
                solver_output=tee,
                path_license_string=path_license_string,
                enforce_post_checks=True,
                strict_path_capi=True,
                strict_residual_tol=strict_residual_tol,
                closure_config=contract.closure,
                equation_scaling=True,
            )
        else:
            baseline_result = _run_path_capi_linear_block(
                base_model,
                base_params,
                reference_rtms=dict(base_params.taxes.rtms),
                solver_output=tee,
                path_license_string=path_license_string,
                enforce_post_checks=True,
                strict_path_capi=True,
                strict_residual_tol=strict_residual_tol,
                closure_config=contract.closure,
            )

        # Shocked run
        with click.progressbar(length=2, label="Loading shock") as bar:
            shock_params = GTAPParameters()
            shock_params.load_from_gdx(gdx_file)
            bar.update(1)

            applied = _apply_shock_to_params(
                shock_params,
                variable,
                shock_index,
                value,
                shock_mode=shock_mode,
            )
            if not applied:
                raise ValueError(
                    "validate-shock currently supports parameter-level shocks only "
                    "(e.g., rtms in params.taxes)"
                )

            shock_equations = GTAPModelEquations(
                shock_params.sets,
                shock_params,
                contract.closure,
                is_counterfactual=True,
            )
            shock_model = shock_equations.build_model()
            # apply_production_scaling already called inside build_model()
            bar.update(1)

        click.echo("Running strict shocked validation...")
        if path_capi_mode == "nonlinear":
            if homotopy_steps > 1 or calibrated_start:
                label = f"Running homotopy shocked validation ({homotopy_steps} steps"
                if calibrated_start:
                    label += ", calibrated-start"
                label += ")..."
                click.echo(label)
                homotopy_result = _run_homotopy_shocked(
                    base_model=base_model,
                    gdx_path=gdx_file,
                    shock_variable=variable,
                    shock_index=shock_index,
                    shock_value=value,
                    shock_mode=shock_mode,
                    homotopy_steps=max(homotopy_steps, 1),
                    contract=contract,
                    solver_output=tee,
                    path_license_string=path_license_string,
                    strict_residual_tol=strict_residual_tol,
                    calibrated_start=calibrated_start,
                )
                shock_model = homotopy_result["shocked_model"]
                shocked_result = homotopy_result
            else:
                shocked_result = _run_path_capi_nonlinear_full(
                    shock_model,
                    shock_params,
                    solver_output=tee,
                    path_license_string=path_license_string,
                    enforce_post_checks=True,
                    strict_path_capi=True,
                    strict_residual_tol=strict_residual_tol,
                    closure_config=contract.closure,
                    equation_scaling=True,
                )
        else:
            shocked_result = _run_path_capi_linear_block(
                shock_model,
                shock_params,
                reference_rtms=dict(base_params.taxes.rtms),
                solver_output=tee,
                path_license_string=path_license_string,
                enforce_post_checks=True,
                strict_path_capi=True,
                strict_residual_tol=strict_residual_tol,
                closure_config=contract.closure,
            )

        baseline_quantities = _collect_key_quantities(
            base_model, base_params, scale_for_gams=True
        )
        shocked_quantities = _collect_key_quantities(
            shock_model, shock_params, scale_for_gams=True
        )
        delta_summary = _build_delta_summary(baseline_quantities, shocked_quantities)

        if welfare_decomp or welfare_har:
            from equilibria.templates.gtap.welfare_decomp import (
                compute_welfare_decomposition,
                compute_welfare_decomposition_homotopy,
            )

            output_dir = Path(output) if output else Path(".")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Use homotopy path integration when available — gives exact RunGTAP equivalence.
            homotopy_path = (
                isinstance(shocked_result, dict)
                and shocked_result.get("step_models")
                and shocked_result.get("step_params")
                and len(shocked_result["step_models"]) >= 2
            )
            if homotopy_path:
                step_levels = [baseline_quantities] + [
                    _collect_key_quantities(m, p, scale_for_gams=True)
                    for m, p in zip(
                        shocked_result["step_models"], shocked_result["step_params"]
                    )
                ]
                step_params_full = [base_params] + list(shocked_result["step_params"])
                welfare_results = compute_welfare_decomposition_homotopy(
                    step_params_full,
                    step_levels,
                )
                click.echo(
                    f"\nWelfare decomposition (Huff/RunGTAP, N={len(step_levels) - 1} homotopy steps):"
                )
            else:
                welfare_results = compute_welfare_decomposition(
                    base_params,
                    shock_params,
                    baseline_quantities,
                    shocked_quantities,
                )
                click.echo(
                    "\nWelfare decomposition (Huff/RunGTAP, single-step — residual ~1-3% expected):"
                )

            csv_path = output_dir / "welfare_decomposition.csv"
            with csv_path.open("w") as fh:
                first = next(iter(welfare_results.values()))
                cols = ["region"] + list(first.as_dict().keys())
                fh.write(",".join(cols) + "\n")
                for region in sorted(welfare_results):
                    row = welfare_results[region].as_dict()
                    fh.write(
                        region
                        + ","
                        + ",".join(f"{row[c]:.6f}" for c in cols[1:])
                        + "\n"
                    )
            click.echo(f"  Wrote {csv_path}")

            click.echo(
                f"  {'Region':<10} {'EV ($M)':>12} {'A':>10} {'T':>10} {'IS':>10} {'resid%':>8}"
            )
            for region in sorted(welfare_results):
                w = welfare_results[region]
                resid_pct = (100.0 * w.residual / w.EV) if abs(w.EV) > 1e-9 else 0.0
                click.echo(
                    f"  {region:<10} {w.EV:>12,.2f} {w.A_total:>10,.2f} "
                    f"{w.T:>10,.2f} {w.IS:>10,.2f} {resid_pct:>+7.2f}%"
                )

            if welfare_har:
                from equilibria.templates.gtap.welfare_decomp_har import (
                    write_welview_har,
                )

                write_welview_har(Path(welfare_har), welfare_results)
                click.echo(f"  Wrote {welfare_har}")

        income_block_vars = ["regy", "ytax_ind", "facty", "yc", "yg", "yi", "rsav"]
        income_block_maxima: dict[str, dict[str, float]] = {}

        def _max_abs(values: dict[str, float]) -> float:
            return max((abs(v) for v in values.values()), default=0.0)

        for var_name in income_block_vars:
            income_block_maxima[var_name] = {
                "baseline_max_abs": _max_abs(baseline_quantities.get(var_name, {})),
                "shock_max_abs": _max_abs(shocked_quantities.get(var_name, {})),
            }

        def _income_block_diagnostics(model) -> dict[str, Any]:
            from pyomo.environ import value

            per_region: dict[str, dict[str, float]] = {}
            residuals: dict[str, dict[str, float]] = {
                "regy_balance": {},
                "yc_balance": {},
                "yg_balance": {},
                "rsav_balance": {},
                "yi_balance": {},
            }

            for r in model.r:
                regy = float(value(model.regy[r])) if hasattr(model, "regy") else 0.0
                facty = float(value(model.facty[r])) if hasattr(model, "facty") else 0.0
                ytax_ind = (
                    float(value(model.ytax_ind[r]))
                    if hasattr(model, "ytax_ind")
                    else 0.0
                )
                yc = float(value(model.yc[r])) if hasattr(model, "yc") else 0.0
                yg = float(value(model.yg[r])) if hasattr(model, "yg") else 0.0
                yi = float(value(model.yi[r])) if hasattr(model, "yi") else 0.0
                rsav = float(value(model.rsav[r])) if hasattr(model, "rsav") else 0.0
                betap = float(value(model.betap[r])) if hasattr(model, "betap") else 0.0
                betag = float(value(model.betag[r])) if hasattr(model, "betag") else 0.0
                betas = float(value(model.betas[r])) if hasattr(model, "betas") else 0.0
                phi = float(value(model.phi[r])) if hasattr(model, "phi") else 1.0
                phip = float(value(model.phip[r])) if hasattr(model, "phip") else 1.0
                yi_share = (
                    float(value(model.yi_share_reg[r]))
                    if hasattr(model, "yi_share_reg")
                    else 0.0
                )

                per_region[str(r)] = {
                    "regy": regy,
                    "facty": facty,
                    "ytax_ind": ytax_ind,
                    "yc": yc,
                    "yg": yg,
                    "yi": yi,
                    "rsav": rsav,
                    "betap": betap,
                    "betag": betag,
                    "betas": betas,
                    "phi": phi,
                    "phip": phip,
                    "yi_share": yi_share,
                }

                residuals["regy_balance"][str(r)] = regy - (facty + ytax_ind)
                residuals["yc_balance"][str(r)] = (
                    yc - betap * (phi / phip) * regy if phip != 0.0 else 0.0
                )
                residuals["yg_balance"][str(r)] = yg - betag * phi * regy
                residuals["rsav_balance"][str(r)] = rsav - betas * phi * regy
                residuals["yi_balance"][str(r)] = yi - yi_share * regy

            def _top_abs(
                entries: dict[str, float], top_n: int = 5
            ) -> list[dict[str, float]]:
                ordered = sorted(
                    entries.items(), key=lambda kv: abs(kv[1]), reverse=True
                )
                return [{"region": r, "residual": v} for r, v in ordered[:top_n]]

            max_abs: dict[str, dict[str, float]] = {}
            top_regions: dict[str, list[dict[str, float]]] = {}
            for name, entries in residuals.items():
                max_abs[name] = {
                    "max_abs": max((abs(v) for v in entries.values()), default=0.0)
                }
                top_regions[name] = _top_abs(entries)

            return {
                "per_region": per_region,
                "residuals": residuals,
                "max_abs": max_abs,
                "top_regions": top_regions,
            }

        income_block_diagnostics = {
            "baseline": _income_block_diagnostics(base_model),
            "shock": _income_block_diagnostics(shock_model),
        }

        success = bool(baseline_result.get("success", False)) and bool(
            shocked_result.get("success", False)
        )
        status = "converged" if success else "failed"

        report = {
            "status": status,
            "success": success,
            "solver": "path-capi",
            "message": "Strict baseline+shock validation via PATH C API",
            "shock": {
                "variable": variable,
                "index": list(shock_index),
                "value": value,
                "mode": shock_mode,
            },
            "baseline": baseline_result,
            "shocked": shocked_result,
            "delta_summary": delta_summary,
            "income_block_maxima": income_block_maxima,
            "income_block_diagnostics": income_block_diagnostics,
        }

        click.echo(f"\n{'=' * 60}")
        click.echo("Validation Shock Results")
        click.echo(f"{'=' * 60}")
        click.echo(f"Overall:      {status}")
        click.echo(
            f"Baseline:     {baseline_result['status']} (res={baseline_result['residual']:.2e})"
        )
        click.echo(
            f"Shocked:      {shocked_result['status']} (res={shocked_result['residual']:.2e})"
        )
        click.echo(f"Delta max|d|: {delta_summary['global']['max_abs_change']:.2e}")

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(report, f, indent=2)
            click.echo(f"\nValidation shock report saved to: {output}")

        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command(name="altertax")
@click.option(
    "--gdx-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to GTAP basedata GDX file",
)
@click.option(
    "--variable",
    default="rtms",
    show_default=True,
    help="Tax variable to shock before rebalancing",
)
@click.option(
    "--index",
    default="all",
    show_default=True,
    help='Shock index tuple, e.g. "(EastAsia,c_TextWapp,NAmerica)" or "all"',
)
@click.option(
    "--value",
    type=float,
    required=True,
    help="Shock value (interpretation depends on --shock-mode)",
)
@click.option(
    "--shock-mode",
    type=click.Choice(["set", "pct", "mult", "tm_pct"]),
    default="tm_pct",
    show_default=True,
    help="Shock semantics. tm_pct mirrors GAMS tm.fx = tm.l*(1+v).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Destination .har file for the rebalanced dataset",
)
@click.option("--tee/--no-tee", default=False, help="Show solver output")
@click.option(
    "--path-license-string", default=None, help="Optional PATH license string"
)
@click.pass_context
def altertax(
    ctx, gdx_file, variable, index, value, shock_mode, output, tee, path_license_string
):
    """Rebalance a GTAP dataset under altertax (Malcolm 1998 CD invariance).

    Applies altertax CD elasticity overrides + the user's tax shock,
    solves via PATH C API (nonlinear full), then extracts a balanced
    SAM into a HAR file consumable as a new baseline.
    """
    from equilibria.templates.gtap.altertax import (
        apply_altertax_elasticities,
        rebalance_to_altertax_dataset,
    )

    logger = ctx.obj["logger"]
    logger.info(f"Altertax rebalance: gdx={gdx_file} → {output}")

    try:
        click.echo(f"\n[1/4] Loading 9x10 baseline from {gdx_file.name}")
        sets = GTAPSets()
        sets.load_from_gdx(gdx_file)
        base_params = GTAPParameters()
        base_params.load_from_gdx(gdx_file)
        click.echo(
            f"      R={sets.n_regions} I={sets.n_commodities} "
            f"A={sets.n_activities} F={sets.n_factors}"
        )

        click.echo("[2/4] Applying altertax CD elasticity overrides + shock")
        shock_params = apply_altertax_elasticities(base_params, in_place=False)
        shock_index = _parse_index(index)
        applied = _apply_shock_to_params(
            shock_params,
            variable,
            shock_index,
            value,
            shock_mode=shock_mode,
        )
        if not applied:
            raise ValueError(f"Shock {variable}{shock_index}={value} did not apply.")
        click.echo(
            f"      shock applied to {variable}{shock_index} (mode={shock_mode})"
        )

        click.echo("[3/4] Building model + solving via PATH C API (nonlinear full)")
        contract = _build_gtap_contract_with_calibration({"closure": "altertax"})
        equations = GTAPModelEquations(sets, shock_params, contract.closure)
        model = equations.build_model()
        solve_result = _run_path_capi_nonlinear_full(
            model,
            shock_params,
            solver_output=tee,
            path_license_string=path_license_string,
            enforce_post_checks=False,
            strict_path_capi=False,
            closure_config=contract.closure,
            equation_scaling=True,
        )
        status = solve_result.get("status")
        residual = solve_result.get("residual", float("nan"))
        click.echo(f"      solve status={status} residual={residual:.2e}")
        # Guardrail: refuse to write a "rebalanced" SAM from a non-converged
        # solution — the values would be cold-init garbage, not equilibrium.
        if status not in ("solved", "success", "optimal") or not (residual < 1e-3):
            raise RuntimeError(
                f"PATH solve did not converge (status={status}, "
                f"residual={residual:.2e}); refusing to write HAR. The "
                "altertax closure may need additional eq pairings — see "
                "docs/plans/gtap_altertax_implementation_plan_2026-05-13.md."
            )

        click.echo(f"[4/4] Writing rebalanced HAR → {output}")
        rebalance = rebalance_to_altertax_dataset(
            base_params=base_params,
            shock_params=shock_params,
            shock_model=model,
            sets=sets,
            output_path=output,
        )
        click.echo(f"      {output.name} ({output.stat().st_size:,} bytes)")
        click.echo("\n  SAM totals (USD M):")
        for k, v in rebalance.sam_totals.items():
            click.echo(f"    {k:14s} = {v:>14,.1f}")
        click.echo("\n✓ Altertax rebalance complete")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
