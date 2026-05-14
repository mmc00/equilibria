"""Tests for GTAP altertax (Malcolm 1998 CD rebalance).

Three groups:
1. Closure preset round-trip (no model needed).
2. Elasticity overrides (in-memory parameter swap).
3. HAR rebalance round-trip with a synthetic 2-region/2-commodity solved model.

Heavy integration tests against a real solved 9x10 model live in
``scripts/gtap/`` (see plan §validation).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import pytest

from equilibria.templates.gtap import build_gtap_contract
from equilibria.templates.gtap.altertax import (
    ALTERTAX_ELASTICITY_DEFAULTS,
    apply_altertax_elasticities,
    rebalance_to_altertax_dataset,
)
from equilibria.templates.gtap.gtap_parameters import (
    GTAPElasticities,
    GTAPParameters,
)


# ---------------------------------------------------------------------------
# 1. Closure preset
# ---------------------------------------------------------------------------


def test_altertax_closure_preset_loads():
    contract = build_gtap_contract({"closure": "altertax"})
    assert contract.closure.name == "altertax"
    assert contract.closure.closure_type == "MCP"
    assert contract.closure.capital_mobility == "mobile"
    assert contract.closure.fix_endowments is True
    assert contract.closure.label is not None
    assert "altertax" in contract.closure.label.lower()


# ---------------------------------------------------------------------------
# 2. Elasticity overrides
# ---------------------------------------------------------------------------


def _params_with_synthetic_elasticities() -> GTAPParameters:
    """Build a GTAPParameters with non-trivial baseline elasticities."""
    e = GTAPElasticities()
    e.esubva.update({("USA", "agr"): 0.3, ("EU", "agr"): 0.5})
    e.esubd.update({("USA", "c1"): 2.0, ("EU", "c1"): 3.0})
    e.esubm.update({("USA", "c1"): 5.0})
    e.omegax.update({("USA", "c1"): 4.0})
    e.omegaw.update({("USA", "c1"): 6.0})
    e.omegas.update({("USA", "agr"): 1.5})
    e.sigmas.update({("USA", "c1"): 1.7})
    e.omegaf.update({("USA", "Labor"): 1.5})
    e.sigmav.update({("USA", "agr"): 0.4})
    e.sigmap.update({("USA", "agr"): 0.4})
    e.sigmand.update({("USA", "agr"): 0.4})
    e.etrae.update({"Labor": 0.7, "Capital": 0.9})
    p = GTAPParameters()
    p.elasticities = e
    return p


def test_apply_altertax_elasticities_returns_copy_by_default():
    base = _params_with_synthetic_elasticities()
    altered = apply_altertax_elasticities(base)

    # Base is untouched
    assert base.elasticities.esubva[("USA", "agr")] == 0.3
    # Altered has CD value
    assert altered.elasticities.esubva[("USA", "agr")] == 1.0
    assert altered.elasticities.esubva[("EU", "agr")] == 1.0


def test_apply_altertax_elasticities_overrides_all_containers():
    base = _params_with_synthetic_elasticities()
    p = apply_altertax_elasticities(base)
    o = ALTERTAX_ELASTICITY_DEFAULTS

    assert all(v == o.esubva for v in p.elasticities.esubva.values())
    assert all(v == o.esubd for v in p.elasticities.esubd.values())
    assert all(v == o.esubm for v in p.elasticities.esubm.values())
    assert all(v == o.omegax for v in p.elasticities.omegax.values())
    assert all(v == o.omegaw for v in p.elasticities.omegaw.values())
    assert all(v == o.omegaf for v in p.elasticities.omegaf.values())
    assert all(v == o.sigmav for v in p.elasticities.sigmav.values())
    assert all(v == o.sigmap for v in p.elasticities.sigmap.values())
    assert all(v == o.sigmand for v in p.elasticities.sigmand.values())
    assert all(v == o.omegas for v in p.elasticities.omegas.values())
    assert all(v == o.sigmas for v in p.elasticities.sigmas.values())
    assert all(v == o.etrae for v in p.elasticities.etrae.values())


def test_apply_altertax_elasticities_in_place_mutates():
    base = _params_with_synthetic_elasticities()
    same = apply_altertax_elasticities(base, in_place=True)
    assert same is base
    assert base.elasticities.esubva[("USA", "agr")] == 1.0


def test_apply_altertax_preserves_calibrated_data():
    base = _params_with_synthetic_elasticities()
    base.benchmark.evfb[("USA", "Labor", "agr")] = 100.0
    base.taxes.rtms[("USA", "c1", "EU")] = 0.05

    altered = apply_altertax_elasticities(base)

    # Benchmarks and taxes unchanged
    assert altered.benchmark.evfb[("USA", "Labor", "agr")] == 100.0
    assert altered.taxes.rtms[("USA", "c1", "EU")] == 0.05


# ---------------------------------------------------------------------------
# 3. HAR rebalance round-trip with synthetic solved model
# ---------------------------------------------------------------------------


@dataclass
class _MockVar(dict):
    """Pyomo-Var lookalike: __contains__ + .value via __getitem__."""

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        return SimpleNamespace(value=super().__getitem__(key))


@dataclass
class _MockModel:
    xda: _MockVar = field(default_factory=_MockVar)
    xma: _MockVar = field(default_factory=_MockVar)
    pd: _MockVar = field(default_factory=_MockVar)
    pmt: _MockVar = field(default_factory=_MockVar)
    xw: _MockVar = field(default_factory=_MockVar)
    pe: _MockVar = field(default_factory=_MockVar)
    pmcif: _MockVar = field(default_factory=_MockVar)
    pf: _MockVar = field(default_factory=_MockVar)
    xf: _MockVar = field(default_factory=_MockVar)
    pgdpmp: _MockVar = field(default_factory=_MockVar)


def _build_synthetic_model() -> Tuple[_MockModel, GTAPParameters, SimpleNamespace]:
    """Tiny 2-region, 2-commodity, 1-sector, 1-factor model."""
    R = ["USA", "EU"]
    I = ["c1", "c2"]
    A = ["s1"]
    F = ["Labor"]
    AGENTS = ["s1", "hhd", "gov", "inv"]

    m = _MockModel()
    for r in R:
        m.pgdpmp[r] = 1.0
        for i in I:
            m.pd[r, i] = 1.0
            m.pmt[r, i] = 1.05
            for aa in AGENTS:
                m.xda[r, i, aa] = 10.0
                m.xma[r, i, aa] = 5.0
        for rp in R:
            if r == rp:
                continue
            for i in I:
                m.xw[r, i, rp] = 4.0
                m.pe[r, i, rp] = 1.0
                m.pmcif[r, i, rp] = 1.1
        for f in F:
            for a in A:
                m.pf[r, f, a] = 1.0
                m.xf[r, f, a] = 50.0

    p = GTAPParameters()
    # Minimal tax rate so EVOS calculation has signal
    p.taxes.rtf[("USA", "Labor", "s1")] = 0.10
    p.taxes.rtf[("EU", "Labor", "s1")] = 0.05

    sets = SimpleNamespace(r=R, i=I, a=A, f=F)
    return m, p, sets


def test_rebalance_writes_har_file(tmp_path: Path):
    model, params, sets = _build_synthetic_model()
    out = tmp_path / "altertax.har"

    result = rebalance_to_altertax_dataset(
        params, params, model, sets,
        output_path=out,
    )

    assert out.exists()
    assert out.stat().st_size > 0
    assert result.regions == ["USA", "EU"]
    assert result.commodities == ["c1", "c2"]
    assert result.sectors == ["s1"]
    assert result.factors == ["Labor"]
    assert result.scale_rgdpmp == {"USA": 1.0, "EU": 1.0}


def test_rebalance_sam_totals_match_inputs(tmp_path: Path):
    model, params, sets = _build_synthetic_model()
    out = tmp_path / "altertax.har"

    result = rebalance_to_altertax_dataset(
        params, params, model, sets, output_path=out,
    )

    # 2 regions × 2 commodities × 4 agents at xda=10 with pd=1.0 gives 80 for VDFB+VxxB.
    # VDFB only includes activities (1 sector "s1"): 2 regions × 2 commodities × 10 = 40
    assert result.sam_totals["VDFB_total"] == pytest.approx(40.0)
    # VMFB: same shape, with pmt=1.05 → 2*2*5*1.05 = 21
    assert result.sam_totals["VMFB_total"] == pytest.approx(21.0)
    # VDPB: 2 regions × 2 commodities × hhd=10 × pd=1.0 = 40
    assert result.sam_totals["VDPB_total"] == pytest.approx(40.0)
    # EVFB: 2 regions × 1 factor × 1 sector × pf*xf=50 = 100
    assert result.sam_totals["EVFB_total"] == pytest.approx(100.0)
    # VXSB: 2 regions × 2 commodities × 1 partner × xw*pe=4 = 16
    assert result.sam_totals["VXSB_total"] == pytest.approx(16.0)
    # VMSB: 2 regions × 2 commodities × 1 partner × xw*pmcif=4*1.1 = 17.6
    assert result.sam_totals["VMSB_total"] == pytest.approx(17.6)


def test_rebalance_har_roundtrips_via_reader(tmp_path: Path):
    """Output HAR must be parseable by babel.har.reader."""
    from equilibria.babel.har.reader import read_har

    model, params, sets = _build_synthetic_model()
    out = tmp_path / "altertax.har"
    rebalance_to_altertax_dataset(params, params, model, sets, output_path=out)

    headers = read_har(out)
    # All standard GTAP basedata headers present
    expected = {
        "REG", "COMM", "ACTS", "ENDW",
        "VDFB", "VMFB", "VDPB", "VMPB", "VDGB", "VMGB", "VDIB", "VMIB",
        "EVFB", "EVOS", "VXSB", "VMSB", "RGDP",
    }
    assert expected.issubset(set(headers.keys()))


def test_rebalance_evos_reflects_factor_tax(tmp_path: Path):
    """EVOS = EVFB / (1 + rtf): USA has rtf=0.10, EU has rtf=0.05."""
    model, params, sets = _build_synthetic_model()
    out = tmp_path / "altertax.har"
    result = rebalance_to_altertax_dataset(
        params, params, model, sets, output_path=out,
    )

    # EVFB total = 100. EVOS = 50/(1+0.10) + 50/(1+0.05) ≈ 45.4545 + 47.6190 = 93.0736
    assert result.sam_totals["EVOS_total"] == pytest.approx(45.4545 + 47.6190, rel=1e-4)
