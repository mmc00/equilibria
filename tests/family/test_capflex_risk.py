"""capFlex's ``risk`` on the family route: from the data as Pyomo (no GAMS GDX),
stamped before the cache keys."""

import copy

import pytest

pytest.importorskip("jax")

from tests.family._data import load_params  # noqa: E402
from tests.family._oracle import comp_stat_closure  # noqa: E402

DATASET = "gtap7_3x3"


def _capflex():
    return comp_stat_closure().model_copy(update={"savf_flag": "capFlex"})


def _params():
    return copy.deepcopy(load_params(DATASET))


def test_risk_equals_pyomos():
    from equilibria.family.capflex import capflex_risk
    from equilibria.templates.gtap.gtap_block_model import _calibrate_capflex_risk

    p = _params()
    regions = [str(r) for r in p.sets.r]
    risk = capflex_risk(p)
    assert set(risk) == set(regions)
    assert risk == _calibrate_capflex_risk(p, p.sets, _capflex(), regions[-1])


def test_prepare_run_needs_no_gdx(monkeypatch, tmp_path):
    """capFlex with ``ref_gdx=None`` gets past the stamp to the records."""
    from equilibria.family import disk_cache
    from equilibria.family.run_setup import prepare_run

    class Reached(Exception):
        pass

    def reached(*a, **k):
        raise Reached

    monkeypatch.setattr(disk_cache, "load_or_build", reached)
    p = _params()
    with pytest.raises(Reached):
        prepare_run(p, _capflex(), "ROW", cache_root=tmp_path, ref_gdx=None)
    assert set(p._capflex_risk) == {str(r) for r in p.sets.r}


def test_keys_split_capfix_and_capflex():
    from equilibria.family.capflex import stamp_capflex_risk
    from equilibria.family.disk_cache import cache_key

    capfix, capflex = comp_stat_closure(), _capflex()
    p = _params()
    k_fix = cache_key(p, capfix, "ROW")
    k_unstamped = cache_key(p, capflex, "ROW")
    stamp_capflex_risk(p, capfix)  # capFix: no-op
    assert not hasattr(p, "_capflex_risk")
    assert cache_key(p, capfix, "ROW") == k_fix

    q = _params()
    stamp_capflex_risk(q, capflex)
    assert set(q._capflex_risk) == {str(r) for r in q.sets.r}
    k_flex = cache_key(q, capflex, "ROW")
    assert k_flex not in {k_fix, k_unstamped}

    changed = copy.deepcopy(q)
    changed._capflex_risk = {r: v * 1.5 for r, v in q._capflex_risk.items()}
    assert cache_key(changed, capflex, "ROW") != k_flex


def test_cross_period_path_follows_the_stamp(tmp_path):
    from equilibria.family import cross_period
    from equilibria.family.capflex import stamp_capflex_risk

    fn = cross_period.cache_path
    capfix, capflex = comp_stat_closure(), _capflex()
    p, q = _params(), _params()
    stamp_capflex_risk(q, capflex)
    paths = {fn(p, capfix, "ROW", tmp_path), fn(q, capflex, "ROW", tmp_path)}
    q2 = copy.deepcopy(q)
    q2._capflex_risk = {r: v * 1.5 for r, v in q._capflex_risk.items()}
    paths.add(fn(q2, capflex, "ROW", tmp_path))
    assert len(paths) == 3


def test_stamp_refuses_a_stale_risk():
    from equilibria.family.capflex import stamp_capflex_risk

    p = _params()
    stamp_capflex_risk(p, _capflex())
    stamp_capflex_risk(p, _capflex())  # same value: fine
    p._capflex_risk = dict.fromkeys(p._capflex_risk, 2.0)
    with pytest.raises(ValueError, match="stale"):
        stamp_capflex_risk(p, _capflex())


def test_capfix_drops_a_stale_stamp_so_its_keys_do_not_move():
    from equilibria.family.capflex import stamp_capflex_risk
    from equilibria.family.disk_cache import cache_key

    capfix = comp_stat_closure()
    clean = _params()
    k_clean = cache_key(clean, capfix, "ROW")
    p = _params()
    stamp_capflex_risk(p, _capflex())  # an earlier capFlex run
    assert hasattr(p, "_capflex_risk")
    stamp_capflex_risk(p, capfix)
    assert not hasattr(p, "_capflex_risk")
    assert cache_key(p, capfix, "ROW") == k_clean
