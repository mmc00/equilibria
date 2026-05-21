"""GTAP v6.2 Parameters — benchmark values, elasticities, and tax rates.

Loads the canonical v6.2 dataset format used by GEMPACK:

- ``basedata.har`` — value-flow SAM headers (VFM, VDFA, VIFA, VPA, VGA,
  VXMD, VIMS, VTWR, VKB, ...) at agent / market / world prices.
- ``GTAPPARM`` / ``Default.prm`` — elasticities and parameters (ESBD,
  ESBM, ESBT, ESBV, ETRE, INCP, SUBP, RFLX, SLUG).

Tax rates are *implicit* in v6.2 data (agent price − market price
weighted) and derived at calibration time. This module stores the raw
value flows and elasticities; tax-rate derivation lives in the model
builder (Phase 2).

Differences from `templates.gtap.gtap_parameters.GTAPParameters` (v7):

- No ``ESBC`` (intermediate bundle), ``ESBG`` (govt CES),
  ``ESBS`` (margins CES), ``ETRQ`` (MAKE CET), ``ESBQ`` (MAKE CES).
- ``ESBD``/``ESBM`` indexed by commodity only (no region dim).
- No ``tinc(e,a,r)`` (factor income tax stored on agent prices).
- No ``ENDOWFLAG`` matrix; mobility from ``SLUG`` binary.

Reference: ``runs/gtap_v62_vs_v7/notation_crosswalk.md`` §1–§10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from equilibria.templates.gtap_v62.gtap_v62_sets import GTAPv62Sets

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Elasticities
# ----------------------------------------------------------------------


@dataclass
class GTAPv62Elasticities:
    """v6.2 elasticity parameters from GTAPPARM.

    All elasticity values follow the v6.2 indexing convention:

    - ``esubd[i]`` / ``esubm[i]``: indexed by commodity only (v7 adds a
      region dimension).
    - ``esubt[j]`` / ``esubva[j]``: indexed by sector ∈ PROD_COMM
      (TRAD_COMM ∪ CGDS_COMM, so usually one extra entry for CGDS).
    - ``etrae[i]``: indexed by endowment commodity (factor).
    - ``incpar[i,r]`` / ``subpar[i,r]``: CDE parameters by (commodity, region).
    - ``rorflex[r]``: rate-of-return flexibility by region.
    """

    esubd: Dict[str, float] = field(default_factory=dict)        # ESBD(i)
    esubm: Dict[str, float] = field(default_factory=dict)        # ESBM(i)
    esubt: Dict[str, float] = field(default_factory=dict)        # ESBT(j)
    esubva: Dict[str, float] = field(default_factory=dict)       # ESBV(j)
    etrae: Dict[str, float] = field(default_factory=dict)        # ETRE(i_endw)
    incpar: Dict[Tuple[str, str], float] = field(default_factory=dict)  # INCP(i,r)
    subpar: Dict[Tuple[str, str], float] = field(default_factory=dict)  # SUBP(i,r)
    rorflex: Dict[str, float] = field(default_factory=dict)      # RFLX(r)
    slug: Dict[str, float] = field(default_factory=dict)         # SLUG(i_endw)

    def load_from_har(self, default_path: Path, sets: GTAPv62Sets) -> None:
        """Read all elasticity headers from GTAPPARM/Default.prm."""
        from equilibria.babel.har import read_har

        data = read_har(default_path)

        self.esubd = _vec1d(data, "ESBD", sets.i)
        self.esubm = _vec1d(data, "ESBM", sets.i)
        self.esubt = _vec1d(data, "ESBT", sets.prod_comm)
        self.esubva = _vec1d(data, "ESBV", sets.prod_comm)
        self.etrae = _vec1d(data, "ETRE", sets.f)
        self.rorflex = _vec1d(data, "RFLX", sets.r)
        self.slug = _vec1d(data, "SLUG", sets.f)

        self.incpar = _vec2d(data, "INCP", sets.i, sets.r)
        self.subpar = _vec2d(data, "SUBP", sets.i, sets.r)


# ----------------------------------------------------------------------
# Benchmark SAM values
# ----------------------------------------------------------------------


@dataclass
class GTAPv62BenchmarkValues:
    """v6.2 benchmark value-flow SAM from basedata.har.

    Stored as plain ``Dict[Tuple[str,...], float]`` for compatibility with
    the existing equilibria parameter API. Agent vs market vs world
    prices are kept separately — tax rates are derived from their ratios
    at calibration time.
    """

    # Factor payments (3-d)
    vfm: Dict[Tuple[str, str, str], float] = field(default_factory=dict)   # VFM(f, j, r) market
    evfa: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # EVFA(f, j, r) agent
    evoa: Dict[Tuple[str, str], float] = field(default_factory=dict)       # EVOA(f, r)

    # Intermediates — firms (3-d: commodity, sector, region)
    vdfm: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vdfa: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vifm: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vifa: Dict[Tuple[str, str, str], float] = field(default_factory=dict)

    # Final demand — household (2-d: commodity, region)
    vdpm: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vdpa: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vipm: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vipa: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Final demand — government (2-d: commodity, region)
    vdgm: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vdga: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vigm: Dict[Tuple[str, str], float] = field(default_factory=dict)
    viga: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # International trade (3-d: commodity, source, destination)
    vxmd: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # market (FOB)
    vxwd: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # world prices
    vims: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # market
    viws: Dict[Tuple[str, str, str], float] = field(default_factory=dict)  # world

    # Margins
    vst: Dict[Tuple[str, str], float] = field(default_factory=dict)        # VST(m, r)
    vtwr: Dict[Tuple[str, str, str, str], float] = field(default_factory=dict)  # VTWR(m,i,s,d)

    # Capital and savings
    vkb: Dict[str, float] = field(default_factory=dict)                    # VKB(r)
    vdep: Dict[str, float] = field(default_factory=dict)                   # VDEP(r)
    save: Dict[str, float] = field(default_factory=dict)                   # SAVE(r)

    def load_from_har(self, basedata_path: Path, sets: GTAPv62Sets) -> None:
        """Read all V** value headers from basedata.har.

        The v6.2 SAM has a known dimension order for each header. We map
        the array values into Python dicts using the actual element
        labels from the HAR's ``set_elements`` (defensive: we don't rely
        on positional alignment with ``sets.i``/``sets.r``/etc.).
        """
        from equilibria.babel.har import read_har

        data = read_har(basedata_path)

        # 3-d (factor, sector, region): VFM, EVFA
        self.vfm = _dense_to_dict(data, "VFM", expected_rank=3)
        self.evfa = _dense_to_dict(data, "EVFA", expected_rank=3)

        # 2-d (factor, region): EVOA
        self.evoa = _dense_to_dict(data, "EVOA", expected_rank=2)

        # 3-d (commodity, sector, region): firm intermediates
        for name, target in (
            ("VDFM", "vdfm"),
            ("VDFA", "vdfa"),
            ("VIFM", "vifm"),
            ("VIFA", "vifa"),
        ):
            setattr(self, target, _dense_to_dict(data, name, expected_rank=3))

        # 2-d (commodity, region): household & gov
        for name, target in (
            ("VDPM", "vdpm"),
            ("VDPA", "vdpa"),
            ("VIPM", "vipm"),
            ("VIPA", "vipa"),
            ("VDGM", "vdgm"),
            ("VDGA", "vdga"),
            ("VIGM", "vigm"),
            ("VIGA", "viga"),
        ):
            setattr(self, target, _dense_to_dict(data, name, expected_rank=2))

        # 3-d (commodity, source, destination): bilateral trade
        for name, target in (
            ("VXMD", "vxmd"),
            ("VXWD", "vxwd"),
            ("VIMS", "vims"),
            ("VIWS", "viws"),
        ):
            setattr(self, target, _dense_to_dict(data, name, expected_rank=3))

        # Margins
        self.vst = _dense_to_dict(data, "VST", expected_rank=2)
        self.vtwr = _dense_to_dict(data, "VTWR", expected_rank=4)

        # 1-d (region)
        self.vkb = _dense_to_dict(data, "VKB", expected_rank=1, as_tuple=False)
        self.vdep = _dense_to_dict(data, "VDEP", expected_rank=1, as_tuple=False)
        self.save = _dense_to_dict(data, "SAVE", expected_rank=1, as_tuple=False)


# ----------------------------------------------------------------------
# Top-level container
# ----------------------------------------------------------------------


@dataclass
class GTAPv62Parameters:
    """Top-level container for GTAP v6.2 elasticities + benchmark values.

    Tax rates and calibrated share parameters are derived later by the
    model builder (Phase 2); this container only holds the raw inputs.
    """

    elasticities: GTAPv62Elasticities = field(default_factory=GTAPv62Elasticities)
    benchmark: GTAPv62BenchmarkValues = field(default_factory=GTAPv62BenchmarkValues)

    aggregation_name: str = ""
    basedata_path: Optional[Path] = None
    default_prm_path: Optional[Path] = None

    def load_from_har(
        self,
        basedata_path: Path,
        default_prm_path: Path,
        sets: GTAPv62Sets,
    ) -> None:
        """Load both basedata.har and Default.prm (GTAPPARM)."""
        self.basedata_path = basedata_path
        self.default_prm_path = default_prm_path
        self.aggregation_name = basedata_path.parent.name or basedata_path.stem
        self.elasticities.load_from_har(default_prm_path, sets)
        self.benchmark.load_from_har(basedata_path, sets)

    def validate(self, sets: GTAPv62Sets) -> Tuple[bool, List[str]]:
        """Sanity-check that benchmark values cover the declared sets."""
        errors: List[str] = []

        if not self.benchmark.vfm:
            errors.append("VFM (factor purchases at market prices) is empty")
        if not self.benchmark.vdfa or not self.benchmark.vifa:
            errors.append("VDFA / VIFA (firm intermediates) are empty")
        if not self.benchmark.vxmd:
            errors.append("VXMD (bilateral exports) is empty")

        # Spot-check elasticities for the canonical headers
        if not self.elasticities.esubd:
            errors.append("ESBD (top Armington elasticity) is empty")
        if not self.elasticities.esubva:
            errors.append("ESBV (value-added elasticity) is empty")
        if not self.elasticities.etrae:
            errors.append("ETRE (factor mobility elasticity) is empty")

        return len(errors) == 0, errors

    def get_info(self) -> Dict[str, Any]:
        is_valid, errors = self.validate(GTAPv62Sets())  # type: ignore[arg-type]
        return {
            "aggregation": self.aggregation_name,
            "basedata": str(self.basedata_path) if self.basedata_path else None,
            "default_prm": str(self.default_prm_path) if self.default_prm_path else None,
            "version": "6.2",
            "n_benchmark_cells": {
                "vfm": len(self.benchmark.vfm),
                "vdfa": len(self.benchmark.vdfa),
                "vifa": len(self.benchmark.vifa),
                "vxmd": len(self.benchmark.vxmd),
                "vims": len(self.benchmark.vims),
                "vtwr": len(self.benchmark.vtwr),
            },
            "elasticity_keys": sorted(self.elasticities.esubd.keys()),
            "valid": is_valid,
            "errors": errors,
        }


# ----------------------------------------------------------------------
# HAR reading helpers
# ----------------------------------------------------------------------


def _vec1d(
    data: Dict[str, Any],
    header: str,
    expected_labels: List[str],
) -> Dict[str, float]:
    """Read a 1-d header into a ``label -> value`` dict.

    Falls back to positional indexing when ``set_elements`` is missing.
    """
    if header not in data:
        logger.debug("v6.2 header %s not found in HAR; returning empty dict", header)
        return {}

    arr = data[header]
    values = arr.array if hasattr(arr, "array") else arr
    flat = list(np.asarray(values).flatten())
    labels = _resolve_labels(arr, 0, expected_labels, flat)

    return {label: float(v) for label, v in zip(labels, flat) if label is not None}


def _vec2d(
    data: Dict[str, Any],
    header: str,
    expected_dim0: List[str],
    expected_dim1: List[str],
) -> Dict[Tuple[str, str], float]:
    """Read a 2-d header into a ``(label0, label1) -> value`` dict."""
    if header not in data:
        return {}

    arr = data[header]
    values = arr.array if hasattr(arr, "array") else arr
    np_values = np.asarray(values)

    labels0 = _resolve_labels(arr, 0, expected_dim0, np_values)
    labels1 = _resolve_labels(arr, 1, expected_dim1, np_values)

    out: Dict[Tuple[str, str], float] = {}
    for i, lab0 in enumerate(labels0):
        for j, lab1 in enumerate(labels1):
            if lab0 is None or lab1 is None:
                continue
            out[(lab0, lab1)] = float(np_values[i, j])
    return out


def _dense_to_dict(
    data: Dict[str, Any],
    header: str,
    *,
    expected_rank: int,
    as_tuple: bool = True,
) -> Dict[Any, float]:
    """Generic loader: dense array → dict keyed by element labels.

    Args:
        data: parsed HAR dict.
        header: header name (e.g. "VFM").
        expected_rank: number of dimensions expected in the header.
        as_tuple: when False (only for 1-d), keys are plain strings; when
            True, keys are tuples (even for 1-d, single-element tuples).
    """
    if header not in data:
        logger.debug("v6.2 header %s not found in HAR; returning empty dict", header)
        return {}

    arr = data[header]
    values = np.asarray(arr.array if hasattr(arr, "array") else arr)

    if values.ndim != expected_rank:
        logger.warning(
            "v6.2 header %s expected rank %d but got %d; reshaping",
            header,
            expected_rank,
            values.ndim,
        )

    labels_per_dim: List[List[Optional[str]]] = []
    for dim in range(values.ndim):
        size = values.shape[dim]
        dim_labels = _resolve_labels(arr, dim, [], values, size)
        labels_per_dim.append(dim_labels)

    out: Dict[Any, float] = {}
    if values.ndim == 1:
        for idx, value in enumerate(values):
            label = labels_per_dim[0][idx]
            if label is None:
                continue
            key: Any = (label,) if as_tuple else label
            out[key] = float(value)
    else:
        for index_tuple, value in np.ndenumerate(values):
            key_parts = tuple(
                labels_per_dim[d][index_tuple[d]] for d in range(values.ndim)
            )
            if any(p is None for p in key_parts):
                continue
            out[key_parts] = float(value)
    return out


def _resolve_labels(
    arr: Any,
    dim: int,
    fallback: List[str],
    values: Any,
    size_hint: Optional[int] = None,
) -> List[Optional[str]]:
    """Return label list for a given dimension, falling back to positional."""
    set_elements = getattr(arr, "set_elements", None) or []
    if dim < len(set_elements) and set_elements[dim]:
        return [str(e).strip() for e in set_elements[dim]]

    if size_hint is None:
        np_values = np.asarray(values)
        size_hint = np_values.shape[dim] if np_values.ndim > dim else len(np_values)

    if fallback and len(fallback) >= size_hint:
        return [fallback[i] for i in range(size_hint)]

    return [None] * size_hint
