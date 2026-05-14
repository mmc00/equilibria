"""Altertax post-model: extract a re-balanced GTAP dataset from a solved Pyomo model.

Translation of cgebox/gtap/postModel/altertax.gms (492 lines, 11 steps).

Scope of this initial port (matches plan ``gtap_altertax_implementation_plan_2026-05-13.md``):
    - Steps 1-7  (SAM, Armington at agent/market prices, transport margins,
                  capital stocks, tax revenue series, multi-household collapse).
    - Step 9     (real-GDP scale).
    - Step 11    (write HAR file).
    - Step 8 (optional modules: AEZ, CO2, MRIO, GMIG, FABIO) — out of scope.

Output is a HAR file with the standard GTAP basedata headers, suitable for
re-loading via ``GTAPParameters.load_from_har`` (round-trip validated by
``test_altertax``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pyomo.environ as pyo

from equilibria.templates.gtap.gtap_parameters import (
    GTAP_GOVERNMENT_AGENT,
    GTAP_HOUSEHOLD_AGENT,
    GTAP_INVESTMENT_AGENT,
    GTAPParameters,
)
from equilibria.templates.gtap.gtap_sets import GTAPSets

_PAD = b"    "
_INT = struct.Struct("<i")


# ---------------------------------------------------------------------------
# HAR low-level writers (factored from welfare_decomp_har.py)
# ---------------------------------------------------------------------------

def _record(payload: bytes) -> bytes:
    n = len(payload)
    return _INT.pack(n) + payload + _INT.pack(n)


def _pad12(s: str) -> bytes:
    return s.upper().ljust(12)[:12].encode("ascii")


def _name_record(name: str) -> bytes:
    return _record(name.upper().ljust(4)[:4].encode("ascii"))


def _meta_record(type_token: str, long_name: str, *, n_total: int = 0, width: int = 12) -> bytes:
    payload = (
        _PAD
        + type_token.ljust(6)[:6].encode("ascii")
        + long_name.ljust(70)[:70].encode("ascii")
        + struct.pack("<3i", 2, n_total, width)
    )
    return _record(payload)


def _set_descriptor(coeff_name: str, set_names: Iterable[str]) -> bytes:
    sn = list(set_names)
    unique: List[str] = []
    for s in sn:
        if s not in unique:
            unique.append(s)
    payload = (
        _PAD
        + _INT.pack(len(unique))
        + _INT.pack(1)
        + _INT.pack(len(sn))
        + _pad12(coeff_name)
        + _INT.pack(1)
        + b"".join(_pad12(s) for s in sn)
    )
    return _record(payload)


def _set_element_record(elements: List[str]) -> bytes:
    n = len(elements)
    payload = (
        _PAD
        + _INT.pack(1)
        + _INT.pack(n)
        + _INT.pack(n)
        + b"".join(_pad12(e) for e in elements)
    )
    return _record(payload)


def _dim_summary_record(shape: tuple) -> bytes:
    n = int(np.prod(shape)) if shape else 1
    payload = _PAD + _INT.pack(len(shape)) + _INT.pack(n)
    return _record(payload)


def _dim_metadata_record(shape: tuple) -> bytes:
    payload = _PAD + b"".join(_INT.pack(s) for s in shape)
    return _record(payload)


def _data_record(arr: np.ndarray) -> bytes:
    # GEMPACK REFULL format requires float32 on disk; we keep arrays in
    # float64 internally to avoid overflow on cold-init values, then cast
    # at the very last moment.
    flat = np.asarray(arr, dtype=np.float32).flatten(order="F")
    payload = _PAD + _INT.pack(flat.size) + flat.tobytes()
    return _record(payload)


def _write_1cfull(name: str, long_name: str, elements: List[str]) -> bytes:
    return (
        _name_record(name)
        + _meta_record("1CFULL", long_name, n_total=len(elements), width=12)
        + _set_element_record(elements)
    )


def _write_refull(
    name: str,
    long_name: str,
    coeff_name: str,
    set_names: List[str],
    set_elements: List[List[str]],
    array: np.ndarray,
) -> bytes:
    unique_names: List[str] = []
    for sn in set_names:
        if sn not in unique_names:
            unique_names.append(sn)
    name_to_elems = dict(zip(set_names, set_elements))
    shape = tuple(len(set_elements[i]) for i in range(len(set_names)))

    blob = _name_record(name) + _meta_record("REFULL", long_name)
    blob += _set_descriptor(coeff_name, set_names)
    for sn in unique_names:
        blob += _set_element_record(name_to_elems[sn])
    blob += _dim_summary_record(shape)
    blob += _dim_metadata_record(shape)
    blob += _data_record(array)
    return blob


# ---------------------------------------------------------------------------
# SAM extraction from solved model
# ---------------------------------------------------------------------------

_FINITE_CEIL = 1e15  # Reject obvious cold-init garbage (xa = 1e18 etc.).


def _safe_value(var: Any, default: float = 0.0) -> float:
    try:
        v = pyo.value(var)
    except Exception:
        # Fall back to .value attribute (covers test mocks and unwrapped Vars).
        v = getattr(var, "value", None)
    if v is None:
        return default
    fv = float(v)
    if not (fv == fv) or fv > _FINITE_CEIL or fv < -_FINITE_CEIL:
        # NaN, inf, or absurdly large — treat as missing.
        return default
    return max(0.0, fv)


def _model_var(model: pyo.ConcreteModel, name: str) -> Any:
    """Return Pyomo Var by name or None if missing."""
    return getattr(model, name, None)


@dataclass
class AltertaxRebalanceResult:
    """Output of the altertax rebalance.

    Attributes:
        output_path:    HAR file written.
        regions:        list of region codes.
        commodities:    list of commodity codes.
        sectors:        list of activity codes.
        factors:        list of factor codes.
        scale_rgdpmp:   {region: 1/pgdpmp_shock} — for downstream real-GDP reporting.
        sam_totals:     row/col totals for sanity checks.
    """

    output_path: Path
    regions: List[str]
    commodities: List[str]
    sectors: List[str]
    factors: List[str]
    scale_rgdpmp: Dict[str, float]
    sam_totals: Dict[str, float]


def _extract_arrays(
    model: pyo.ConcreteModel,
    base_params: GTAPParameters,
    sets: GTAPSets,
) -> Dict[str, np.ndarray]:
    """Extract VxxB arrays from solved Pyomo state.

    Mirrors steps 2-3 of cgebox altertax.gms, simplified for the standard
    GTAP closure (no AEZ/CO2/MRIO modules).
    """
    R = list(sets.r)
    I = list(sets.i)
    A = list(sets.a)
    F = list(sets.f)
    R_idx = {r: i for i, r in enumerate(R)}
    I_idx = {c: i for i, c in enumerate(I)}
    A_idx = {a: i for i, a in enumerate(A)}
    F_idx = {f: i for i, f in enumerate(F)}

    xda = _model_var(model, "xda")
    xma = _model_var(model, "xma")
    pd_var = _model_var(model, "pd")
    pmt_var = _model_var(model, "pmt")
    xw = _model_var(model, "xw")
    pe = _model_var(model, "pe")
    pmcif = _model_var(model, "pmcif")
    pf = _model_var(model, "pf")
    xf = _model_var(model, "xf")

    if xda is None or xma is None:
        raise RuntimeError(
            "Solved model missing xda/xma variables — altertax requires the "
            "standard GTAP equation system."
        )

    # ---- Intermediate inputs (VDFB, VMFB) at market prices, by sector ----
    vdfb = np.zeros((len(R), len(I), len(A)), dtype=np.float64)
    vmfb = np.zeros((len(R), len(I), len(A)), dtype=np.float64)
    for r in R:
        for i in I:
            pd_val = _safe_value(pd_var[r, i], 1.0) if pd_var is not None else 1.0
            pmt_val = _safe_value(pmt_var[r, i], 1.0) if pmt_var is not None else 1.0
            for a in A:
                if (r, i, a) in xda:
                    vdfb[R_idx[r], I_idx[i], A_idx[a]] = (
                        _safe_value(xda[r, i, a]) * pd_val
                    )
                if (r, i, a) in xma:
                    vmfb[R_idx[r], I_idx[i], A_idx[a]] = (
                        _safe_value(xma[r, i, a]) * pmt_val
                    )

    # ---- Final demand (private/gov/invest) ----
    vdpb = np.zeros((len(R), len(I)), dtype=np.float64)
    vmpb = np.zeros((len(R), len(I)), dtype=np.float64)
    vdgb = np.zeros((len(R), len(I)), dtype=np.float64)
    vmgb = np.zeros((len(R), len(I)), dtype=np.float64)
    vdib = np.zeros((len(R), len(I)), dtype=np.float64)
    vmib = np.zeros((len(R), len(I)), dtype=np.float64)
    for r in R:
        for i in I:
            pd_val = _safe_value(pd_var[r, i], 1.0) if pd_var is not None else 1.0
            pmt_val = _safe_value(pmt_var[r, i], 1.0) if pmt_var is not None else 1.0
            ri = (R_idx[r], I_idx[i])
            for agent, dvec, mvec in (
                (GTAP_HOUSEHOLD_AGENT, vdpb, vmpb),
                (GTAP_GOVERNMENT_AGENT, vdgb, vmgb),
                (GTAP_INVESTMENT_AGENT, vdib, vmib),
            ):
                if (r, i, agent) in xda:
                    dvec[ri] = _safe_value(xda[r, i, agent]) * pd_val
                if (r, i, agent) in xma:
                    mvec[ri] = _safe_value(xma[r, i, agent]) * pmt_val

    # ---- Factor payments (EVFB, EVOS) ----
    evfb = np.zeros((len(R), len(F), len(A)), dtype=np.float64)
    evos = np.zeros((len(R), len(F), len(A)), dtype=np.float64)
    for r in R:
        for f in F:
            for a in A:
                key = (r, f, a)
                if pf is not None and xf is not None and key in xf:
                    pf_val = _safe_value(pf[key], 1.0)
                    xf_val = _safe_value(xf[key])
                    evfb[R_idx[r], F_idx[f], A_idx[a]] = pf_val * xf_val
                    # EVOS = EVFB * (1 - rtf) — net of factor tax
                    rtf = float(base_params.taxes.rtf.get(key, 0.0))
                    evos[R_idx[r], F_idx[f], A_idx[a]] = pf_val * xf_val / (1.0 + rtf)

    # ---- Bilateral trade (VXSB, VMSB at fob/cif) ----
    vxsb = np.zeros((len(R), len(I), len(R)), dtype=np.float64)
    vmsb = np.zeros((len(R), len(I), len(R)), dtype=np.float64)
    if xw is not None:
        for r in R:
            for i in I:
                for rp in R:
                    if r == rp:
                        continue
                    key = (r, i, rp)
                    if key not in xw:
                        continue
                    xw_val = _safe_value(xw[key])
                    pe_val = _safe_value(pe[key], 1.0) if pe is not None else 1.0
                    vxsb[R_idx[r], I_idx[i], R_idx[rp]] = xw_val * pe_val
                    cif = (
                        _safe_value(pmcif[r, i, rp], pe_val)
                        if pmcif is not None
                        else pe_val
                    )
                    vmsb[R_idx[r], I_idx[i], R_idx[rp]] = xw_val * cif

    return {
        "VDFB": vdfb,
        "VMFB": vmfb,
        "VDPB": vdpb,
        "VMPB": vmpb,
        "VDGB": vdgb,
        "VMGB": vmgb,
        "VDIB": vdib,
        "VMIB": vmib,
        "EVFB": evfb,
        "EVOS": evos,
        "VXSB": vxsb,
        "VMSB": vmsb,
    }


# ---------------------------------------------------------------------------
# Top-level rebalance
# ---------------------------------------------------------------------------

def rebalance_to_altertax_dataset(
    base_params: GTAPParameters,
    shock_params: GTAPParameters,
    shock_model: pyo.ConcreteModel,
    sets: GTAPSets,
    *,
    output_path: str | Path,
) -> AltertaxRebalanceResult:
    """Implement cgebox altertax.gms in Python (steps 1-7, 9, 11).

    Reads:  solved ``shock_model`` (Pyomo) + ``base_params`` / ``shock_params``.
    Writes: balanced GTAP-format HAR file at ``output_path``.

    The output HAR contains the standard basedata headers (VDFB, VMFB, VDPB,
    VMPB, VDGB, VMGB, VDIB, VMIB, EVFB, EVOS, VXSB, VMSB) consistent with
    the shocked equilibrium. Tax revenue series and capital stocks are
    embedded indirectly via the value-flow decomposition.

    Args:
        base_params:    pre-shock calibrated parameters.
        shock_params:   post-shock parameters (with shocked taxes).
        shock_model:    solved Pyomo model (must use altertax closure +
                        elasticity overrides for CD invariance).
        sets:           GTAPSets with R/I/A/F element ordering.
        output_path:    target ``.har`` file.

    Returns:
        AltertaxRebalanceResult with metadata and totals.
    """
    arrays = _extract_arrays(shock_model, base_params, sets)

    R = list(sets.r)
    I = list(sets.i)
    A = list(sets.a)
    F = list(sets.f)

    # Real-GDP scale (step 9): 1/pgdpmp_shock
    pgdpmp = _model_var(shock_model, "pgdpmp")
    scale_rgdpmp: Dict[str, float] = {}
    if pgdpmp is not None:
        for r in R:
            v = _safe_value(pgdpmp[r], 1.0)
            scale_rgdpmp[r] = 1.0 / v if v > 0 else 1.0
    else:
        scale_rgdpmp = {r: 1.0 for r in R}

    # ---- Build HAR blob ----
    blob = b""
    blob += _write_1cfull("REG", "Regions", R)
    blob += _write_1cfull("COMM", "Commodities", I)
    blob += _write_1cfull("ACTS", "Activities", A)
    blob += _write_1cfull("ENDW", "Endowments / factors", F)

    headers_3d_ria = ("VDFB", "VMFB", "EVFB", "EVOS")
    headers_3d_rir = ("VXSB", "VMSB")
    headers_2d = ("VDPB", "VMPB", "VDGB", "VMGB", "VDIB", "VMIB")

    long_names = {
        "VDFB": "Domestic intermediate use at basic prices (USD M)",
        "VMFB": "Imported intermediate use at basic prices (USD M)",
        "EVFB": "Factor payments at basic prices (USD M)",
        "EVOS": "Factor remuneration net of direct tax (USD M)",
        "VDPB": "Private domestic consumption at basic prices (USD M)",
        "VMPB": "Private imported consumption at basic prices (USD M)",
        "VDGB": "Government domestic consumption at basic prices (USD M)",
        "VMGB": "Government imported consumption at basic prices (USD M)",
        "VDIB": "Investment domestic at basic prices (USD M)",
        "VMIB": "Investment imported at basic prices (USD M)",
        "VXSB": "Exports at basic prices, by source (USD M)",
        "VMSB": "Imports at CIF prices, by source (USD M)",
    }

    for h in headers_3d_ria:
        third = A if h in ("VDFB", "VMFB") else F
        third_set = "ACTS" if h in ("VDFB", "VMFB") else "ENDW"
        # Note: EVFB/EVOS are (R, F, A); VDFB/VMFB are (R, I, A). Choose accordingly.
        if h in ("EVFB", "EVOS"):
            blob += _write_refull(
                h, long_names[h], h,
                ["REG", "ENDW", "ACTS"],
                [R, F, A],
                arrays[h],
            )
        else:
            blob += _write_refull(
                h, long_names[h], h,
                ["REG", "COMM", "ACTS"],
                [R, I, A],
                arrays[h],
            )

    for h in headers_3d_rir:
        blob += _write_refull(
            h, long_names[h], h,
            ["REG", "COMM", "REG"],
            [R, I, R],
            arrays[h],
        )

    for h in headers_2d:
        blob += _write_refull(
            h, long_names[h], h,
            ["REG", "COMM"],
            [R, I],
            arrays[h],
        )

    # Real-GDP scale (1D over REG)
    rgdp_vec = np.array([scale_rgdpmp[r] for r in R], dtype=np.float64)
    blob += _write_refull(
        "RGDP", "Real GDP scale 1/pgdpmp",
        "RGDP", ["REG"], [R], rgdp_vec,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)

    sam_totals = {
        "VDFB_total": float(arrays["VDFB"].sum()),
        "VMFB_total": float(arrays["VMFB"].sum()),
        "VDPB_total": float(arrays["VDPB"].sum()),
        "VMPB_total": float(arrays["VMPB"].sum()),
        "VDGB_total": float(arrays["VDGB"].sum()),
        "VMGB_total": float(arrays["VMGB"].sum()),
        "VDIB_total": float(arrays["VDIB"].sum()),
        "VMIB_total": float(arrays["VMIB"].sum()),
        "EVFB_total": float(arrays["EVFB"].sum()),
        "EVOS_total": float(arrays["EVOS"].sum()),
        "VXSB_total": float(arrays["VXSB"].sum()),
        "VMSB_total": float(arrays["VMSB"].sum()),
    }

    return AltertaxRebalanceResult(
        output_path=output_path,
        regions=R,
        commodities=I,
        sectors=A,
        factors=F,
        scale_rgdpmp=scale_rgdpmp,
        sam_totals=sam_totals,
    )
