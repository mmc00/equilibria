"""GTAP6Parameters loads datasets/gtap6_3x3 correctly."""

from __future__ import annotations

from pathlib import Path

from equilibria.templates.gtap6.gtap6_parameters import (
    GTAP6BenchmarkValues,
    GTAP6Elasticities,
    GTAP6Parameters,
)
from equilibria.templates.gtap6.gtap6_sets import GTAP6Sets

DATASET = Path(__file__).resolve().parents[3] / "datasets" / "gtap6_3x3"


def _load_sets() -> GTAP6Sets:
    sets = GTAP6Sets()
    sets.load_from_har(DATASET / "sets.har", default_path=DATASET / "default.prm")
    return sets


def _load_params(sets: GTAP6Sets) -> GTAP6Parameters:
    params = GTAP6Parameters()
    params.load_from_har(DATASET / "basedata.har", DATASET / "default.prm", sets)
    return params


def test_load_from_har_gtap6_3x3():
    sets = _load_sets()
    params = _load_params(sets)

    is_valid, errors = params.validate(sets)
    assert is_valid, errors
    # Benchmark output value for at least one (commodity, sector, region) cell must be positive.
    assert any(v > 0 for v in params.benchmark.vdfm.values())


def test_loads_elasticities():
    sets = _load_sets()
    params = _load_params(sets)

    e = params.elasticities
    assert isinstance(e, GTAP6Elasticities)
    assert set(e.esubd.keys()) == set(sets.i)
    assert set(e.esubm.keys()) == set(sets.i)
    # ESBT/ESBV are indexed over PROD_COMM = TRAD_COMM union CGDS_COMM
    assert set(e.esubt.keys()) == set(sets.prod_comm)
    assert set(e.esubva.keys()) == set(sets.prod_comm)
    # ETRE indexed over factors, RFLX indexed over regions
    assert set(e.etrae.keys()) == set(sets.f)
    assert set(e.rorflex.keys()) == set(sets.r)


def test_loads_factor_and_trade_benchmark():
    sets = _load_sets()
    params = _load_params(sets)

    b = params.benchmark
    assert isinstance(b, GTAP6BenchmarkValues)
    # Keys are 3-tuples (factor, sector, region)
    sample_key = next(iter(b.vfm.keys()))
    assert len(sample_key) == 3

    # Bilateral trade keyed (commodity, source, destination)
    assert b.vxmd
    sample_trade_key = next(iter(b.vxmd.keys()))
    assert len(sample_trade_key) == 3

    # Region aggregates keyed by plain region string
    assert set(b.vkb.keys()) == set(sets.r)
    assert all(v >= 0 for v in b.vkb.values())
