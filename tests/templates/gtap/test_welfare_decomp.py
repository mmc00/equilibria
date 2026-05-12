"""Tests for GTAP welfare decomposition (Huff/RunGTAP).

Uses lightweight mock objects to avoid importing pyomo or building model
equations. All tests run in-memory with synthetic benchmark/tax dicts.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import pytest

from equilibria.templates.gtap.welfare_decomp import (
    ALLOC_BUCKETS,
    WelfareComponents,
    compute_welfare_decomposition,
    compute_welfare_decomposition_homotopy,
)


# ---------------------------------------------------------------------------
# Lightweight mocks for GTAPParameters
# ---------------------------------------------------------------------------


@dataclass
class _MockBenchmark:
    vom: Dict[Tuple[str, str], float] = field(default_factory=dict)
    vmsb: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vxsb: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vfob: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vcif: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    evfb: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    evos: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    vdpb: Dict[Tuple, float] = field(default_factory=dict)
    vdpp: Dict[Tuple, float] = field(default_factory=dict)
    vmpb: Dict[Tuple, float] = field(default_factory=dict)
    vmpp: Dict[Tuple, float] = field(default_factory=dict)
    vdgb: Dict[Tuple, float] = field(default_factory=dict)
    vdgp: Dict[Tuple, float] = field(default_factory=dict)
    vmgb: Dict[Tuple, float] = field(default_factory=dict)
    vmgp: Dict[Tuple, float] = field(default_factory=dict)
    vdib: Dict[Tuple, float] = field(default_factory=dict)
    vdip: Dict[Tuple, float] = field(default_factory=dict)
    vmib: Dict[Tuple, float] = field(default_factory=dict)
    vmip: Dict[Tuple, float] = field(default_factory=dict)


@dataclass
class _MockTaxes:
    rto: Dict[Tuple[str, str], float] = field(default_factory=dict)
    rtms: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    rtxs: Dict[Tuple[str, str, str], float] = field(default_factory=dict)


def _mock_params(regions, factors=("Labor",)):
    sets = SimpleNamespace(r=list(regions), f=list(factors))
    return SimpleNamespace(
        sets=sets,
        benchmark=_MockBenchmark(),
        taxes=_MockTaxes(),
    )


def _snapshot(**buckets):
    """Build a `_collect_key_quantities`-style snapshot from kwargs."""
    out: Dict[str, Dict[str, float]] = {}
    for bucket_name, values in buckets.items():
        out[bucket_name] = {}
        for key, val in values.items():
            if isinstance(key, tuple):
                joined = "|".join(str(k) for k in key)
            else:
                joined = str(key)
            out[bucket_name][joined] = float(val)
    return out


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


def test_alloc_buckets_eleven():
    assert len(ALLOC_BUCKETS) == 11
    assert set(ALLOC_BUCKETS) == {
        "ptax", "imptx", "exptx", "dftax", "mftax",
        "dctax", "mctax", "dgtax", "mgtax", "ditx", "mitx",
    }


def test_welfare_components_total_and_residual():
    w = WelfareComponents(T=2.0, IS=-0.5, ENDW=0.1, TECH=0.0, EV=10.0)
    w.A["ptax"] = 3.0
    w.A["imptx"] = 5.0

    assert w.A_total == pytest.approx(8.0)
    assert w.total == pytest.approx(8.0 + 2.0 - 0.5 + 0.1)
    assert w.residual == pytest.approx(10.0 - w.total)


def test_welfare_components_as_dict_has_all_keys():
    w = WelfareComponents()
    d = w.as_dict()
    for bucket in ALLOC_BUCKETS:
        assert f"A_{bucket}" in d
    for key in ("A_total", "T", "IS", "ENDW", "TECH", "total", "EV", "residual", "residual_pct_of_EV"):
        assert key in d


# ---------------------------------------------------------------------------
# Single-step decomposition
# ---------------------------------------------------------------------------


def test_imptx_bucket_credits_importer():
    """Tariff distortion is credited to the importing region."""
    base = _mock_params(["USA", "CHN"])
    base.benchmark.vmsb[("CHN", "manuf", "USA")] = 100.0
    base.taxes.rtms[("CHN", "manuf", "USA")] = 0.10  # 10% tariff

    shock = _mock_params(["USA", "CHN"])
    shock.benchmark.vmsb[("CHN", "manuf", "USA")] = 90.0  # USA imports less

    snap_base = _snapshot()
    snap_shock = _snapshot()
    out = compute_welfare_decomposition(base, shock, snap_base, snap_shock)

    # τ · Δq = 0.10 · (90 - 100) = -1.0 — efficiency LOSS to USA (less trade through wedge)
    # Wait: rtms is positive (tariff), Δvmsb negative, so contribution is negative.
    # Sign convention: A_imptx > 0 means activity moved INTO the distortion (welfare gain).
    assert out["USA"].A["imptx"] == pytest.approx(-1.0)
    assert out["CHN"].A["imptx"] == 0.0  # not the importer


def test_exptx_bucket_credits_exporter():
    base = _mock_params(["USA", "CHN"])
    base.benchmark.vxsb[("USA", "wheat", "CHN")] = 50.0
    base.taxes.rtxs[("USA", "wheat", "CHN")] = 0.05

    shock = _mock_params(["USA", "CHN"])
    shock.benchmark.vxsb[("USA", "wheat", "CHN")] = 60.0

    out = compute_welfare_decomposition(base, shock, _snapshot(), _snapshot())

    assert out["USA"].A["exptx"] == pytest.approx(0.05 * 10.0)
    assert out["CHN"].A["exptx"] == 0.0


def test_ptax_bucket_credits_producer():
    base = _mock_params(["USA"])
    base.benchmark.vom[("USA", "agri")] = 200.0
    base.taxes.rto[("USA", "agri")] = 0.03

    shock = _mock_params(["USA"])
    shock.benchmark.vom[("USA", "agri")] = 210.0

    out = compute_welfare_decomposition(base, shock, _snapshot(), _snapshot())
    assert out["USA"].A["ptax"] == pytest.approx(0.03 * 10.0)


def test_dftax_uses_evfb_evos_wedge():
    base = _mock_params(["USA"], factors=["Labor"])
    base.benchmark.evfb[("USA", "Labor", "agri")] = 100.0
    base.benchmark.evos[("USA", "Labor", "agri")] = 80.0  # 20% factor tax

    shock = _mock_params(["USA"], factors=["Labor"])
    shock.benchmark.evfb[("USA", "Labor", "agri")] = 110.0

    out = compute_welfare_decomposition(base, shock, _snapshot(), _snapshot())
    # rate = (100-80)/100 = 0.20; Δ = 10 ⇒ A_dftax = 2.0
    assert out["USA"].A["dftax"] == pytest.approx(2.0)


def test_terms_of_trade_exporter_gains_importer_loses_symmetric():
    """If USA exports more wheat at baseline pfob to CHN, USA's T rises; CHN's T falls."""
    base = _mock_params(["USA", "CHN"])
    base.benchmark.vfob[("USA", "wheat", "CHN")] = 100.0
    base.benchmark.vcif[("USA", "wheat", "CHN")] = 110.0

    snap_base = _snapshot(xw={("USA", "wheat", "CHN"): 100.0}, pnum={"__scalar__": 1.0})
    snap_shock = _snapshot(xw={("USA", "wheat", "CHN"): 110.0}, pnum={"__scalar__": 1.0})

    out = compute_welfare_decomposition(base, _mock_params(["USA", "CHN"]), snap_base, snap_shock)

    # pfob_base = 100/100 = 1.0; Δxw = 10 ⇒ USA gains +10
    assert out["USA"].T == pytest.approx(10.0)
    # pcif_base = 110/100 = 1.1; CHN loses 1.1·10 = -11
    assert out["CHN"].T == pytest.approx(-11.0)


def test_endowment_zero_when_factors_unchanged():
    base = _mock_params(["USA"], factors=["Labor"])
    snap = _snapshot(xft={("USA", "Labor"): 100.0}, pft={("USA", "Labor"): 1.0})
    out = compute_welfare_decomposition(base, _mock_params(["USA"], factors=["Labor"]), snap, snap)
    assert out["USA"].ENDW == 0.0


def test_endowment_positive_when_factor_grows():
    base = _mock_params(["USA"], factors=["Labor"])
    snap_b = _snapshot(xft={("USA", "Labor"): 100.0}, pft={("USA", "Labor"): 1.0})
    snap_s = _snapshot(xft={("USA", "Labor"): 105.0}, pft={("USA", "Labor"): 1.0})
    out = compute_welfare_decomposition(base, _mock_params(["USA"], factors=["Labor"]), snap_b, snap_s)
    assert out["USA"].ENDW == pytest.approx(5.0)


def test_tech_always_zero_in_current_impl():
    """TECH is a stub until aoreg/afe surfaced."""
    base = _mock_params(["USA"])
    out = compute_welfare_decomposition(base, _mock_params(["USA"]), _snapshot(), _snapshot())
    assert out["USA"].TECH == 0.0


def test_ev_uses_yc_base_times_delta_uh():
    base = _mock_params(["USA"])
    snap_b = _snapshot(yc={"USA": 1000.0}, uh={"USA": 1.0})
    snap_s = _snapshot(yc={"USA": 1000.0}, uh={"USA": 1.005})  # 0.5% utility gain

    out = compute_welfare_decomposition(base, _mock_params(["USA"]), snap_b, snap_s)
    # EV = 1000 · (1.005 - 1.0) = 5.0
    assert out["USA"].EV == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Homotopy decomposition
# ---------------------------------------------------------------------------


def test_homotopy_requires_at_least_two_snapshots():
    base = _mock_params(["USA"])
    with pytest.raises(ValueError, match="at least two"):
        compute_welfare_decomposition_homotopy([base], [_snapshot()])


def test_homotopy_mismatched_lengths_raises():
    base = _mock_params(["USA"])
    with pytest.raises(ValueError, match="same length"):
        compute_welfare_decomposition_homotopy([base, base], [_snapshot()])


def test_homotopy_with_two_steps_equals_single_step():
    """Two-step path with identical intermediate equals a single-step decomposition."""
    base = _mock_params(["USA"])
    base.benchmark.vom[("USA", "agri")] = 100.0
    base.taxes.rto[("USA", "agri")] = 0.10

    final = _mock_params(["USA"])
    final.benchmark.vom[("USA", "agri")] = 110.0

    single = compute_welfare_decomposition(base, final, _snapshot(), _snapshot())
    homo = compute_welfare_decomposition_homotopy([base, final], [_snapshot(), _snapshot()])

    assert homo["USA"].A["ptax"] == pytest.approx(single["USA"].A["ptax"])


def test_homotopy_sums_partial_contributions():
    """Three-step path: τ·Δq accumulates across segments."""
    p0 = _mock_params(["USA"])
    p0.benchmark.vom[("USA", "agri")] = 100.0
    p0.taxes.rto[("USA", "agri")] = 0.10

    p1 = _mock_params(["USA"])
    p1.benchmark.vom[("USA", "agri")] = 105.0
    p1.taxes.rto[("USA", "agri")] = 0.10

    p2 = _mock_params(["USA"])
    p2.benchmark.vom[("USA", "agri")] = 110.0
    p2.taxes.rto[("USA", "agri")] = 0.10

    snaps = [_snapshot(), _snapshot(), _snapshot()]
    homo = compute_welfare_decomposition_homotopy([p0, p1, p2], snaps)

    # Step 1: 0.10·(105-100) = 0.5. Step 2: 0.10·(110-105) = 0.5. Sum = 1.0.
    assert homo["USA"].A["ptax"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# HAR writer roundtrip
# ---------------------------------------------------------------------------


def test_har_writer_roundtrip():
    """WELVIEW.har round-trips through babel's native reader."""
    from equilibria.babel.har.reader import read_har
    from equilibria.templates.gtap.welfare_decomp_har import write_welview_har

    welfare = {
        "USA": WelfareComponents(EV=10.0, T=2.0, IS=-0.5),
        "CHN": WelfareComponents(EV=-5.0, T=-1.0, IS=0.3),
    }
    welfare["USA"].A["ptax"] = 1.0
    welfare["USA"].A["imptx"] = 7.0
    welfare["CHN"].A["imptx"] = -4.0

    with tempfile.NamedTemporaryFile(suffix=".har", delete=False) as f:
        path = Path(f.name)
    try:
        write_welview_har(path, welfare)
        assert path.stat().st_size > 0

        headers = read_har(path)
        assert "EVAL" in headers
        assert "ALET" in headers
        assert "ALEF" in headers
        assert "TOTE" in headers
        assert "REG" in headers

        # Set REG should contain both regions
        reg_elems = headers["REG"].array.tolist()
        assert set(reg_elems) == {"USA", "CHN"}

        # ALEF is 2D (BUCKET × REG) with 11 × 2 = 22 cells
        assert headers["ALEF"].array.shape == (11, 2)
    finally:
        if path.exists():
            path.unlink()
