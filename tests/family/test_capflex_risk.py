"""capFlex's ``risk`` from the GAMS GDX with no Pyomo, stamped before the cache keys."""

import copy
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax")

from tests.family._data import load_params  # noqa: E402
from tests.family._oracle import comp_stat_closure  # noqa: E402

DATASET = "gtap7_3x3"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _diff_core_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "gtap"))


@pytest.fixture(scope="module")
def gdx():
    from tests.templates.gtap.test_gtap7_gempack_parity import find_reference

    sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
    ref = find_reference(
        DATASET, "gams", form="mcp", mode="gtap", ifsub=1, fbep="header"
    ) or find_reference(
        DATASET, "gams", form="nlp", mode="gtap", ifsub=1, fbep="header"
    )
    path = None if ref is None else ref / "out.gdx"
    if path is None or not path.exists():
        pytest.skip("no GAMS reference GDX for gtap7_3x3")
    return path


def _capflex():
    return comp_stat_closure().model_copy(update={"savf_flag": "capFlex"})


def _params():
    return copy.deepcopy(load_params(DATASET))


def test_risk_equals_pyomo_helpers(gdx):
    from equilibria.family.capflex import capflex_risk
    from equilibria.templates.gtap.gtap_block_model import (
        _calibrate_capflex_risk,
        _capflex_risk_from_gdx,
    )

    p = _params()
    regions = [str(r) for r in p.sets.r]
    risk = capflex_risk(p, gdx)
    assert set(risk) == set(regions)
    assert risk == _capflex_risk_from_gdx(gdx, regions)
    rr = regions[-1]
    assert risk == _calibrate_capflex_risk(p, p.sets, _capflex(), rr, ref_gdx=gdx)


@pytest.mark.parametrize("which", ["none", "missing"])
def test_risk_needs_a_gdx(which, tmp_path):
    from equilibria.family.capflex import capflex_risk

    ref = None if which == "none" else tmp_path / "absent.gdx"
    with pytest.raises(NotImplementedError, match="rore/rorg from a GAMS GDX"):
        capflex_risk(_params(), ref)


def test_risk_needs_rore_in_the_gdx(tmp_path, monkeypatch):
    import importlib

    from equilibria.family.capflex import capflex_risk

    _diff_core = importlib.import_module("_diff_core")
    gdx = tmp_path / "out.gdx"
    gdx.write_bytes(b"")
    monkeypatch.setattr(_diff_core, "gams_levels", lambda *a, **k: {})
    with pytest.raises(NotImplementedError, match="rore/rorg from a GAMS GDX"):
        capflex_risk(_params(), gdx)


def test_prepare_run_raises_before_load_or_build(monkeypatch, tmp_path):
    from equilibria.family import disk_cache
    from equilibria.family.driver import prepare_run

    def boom(*a, **k):
        raise AssertionError("load_or_build ran")

    monkeypatch.setattr(disk_cache, "load_or_build", boom)
    p = _params()
    with pytest.raises(NotImplementedError, match="rore/rorg"):
        prepare_run(p, _capflex(), "ROW", cache_root=tmp_path, ref_gdx=None)


def test_keys_split_capfix_and_capflex(gdx):
    from equilibria.family.capflex import stamp_capflex_risk
    from equilibria.family.disk_cache import cache_key

    capfix, capflex = comp_stat_closure(), _capflex()
    p = _params()
    k_fix = cache_key(p, capfix, "ROW")
    k_unstamped = cache_key(p, capflex, "ROW")
    stamp_capflex_risk(p, capfix, gdx)  # capFix: no-op
    assert not hasattr(p, "_capflex_risk")
    assert cache_key(p, capfix, "ROW") == k_fix

    q = _params()
    stamp_capflex_risk(q, capflex, gdx)
    assert set(q._capflex_risk) == {str(r) for r in q.sets.r}
    k_flex = cache_key(q, capflex, "ROW")
    assert k_flex not in {k_fix, k_unstamped}

    changed = copy.deepcopy(q)
    changed._capflex_risk = {r: v * 1.5 for r, v in q._capflex_risk.items()}
    assert cache_key(changed, capflex, "ROW") != k_flex


def test_cross_period_path_follows_the_stamp(gdx, tmp_path):
    from equilibria.family import cross_period
    from equilibria.family.capflex import stamp_capflex_risk

    fn = cross_period.cache_path
    capfix, capflex = comp_stat_closure(), _capflex()
    p, q = _params(), _params()
    stamp_capflex_risk(q, capflex, gdx)
    paths = {fn(p, capfix, "ROW", tmp_path), fn(q, capflex, "ROW", tmp_path)}
    q2 = copy.deepcopy(q)
    q2._capflex_risk = {r: v * 1.5 for r, v in q._capflex_risk.items()}
    paths.add(fn(q2, capflex, "ROW", tmp_path))
    assert len(paths) == 3


def test_stamp_refuses_a_stale_risk(gdx):
    from equilibria.family.capflex import stamp_capflex_risk

    p = _params()
    stamp_capflex_risk(p, _capflex(), gdx)
    stamp_capflex_risk(p, _capflex(), gdx)  # same value: fine
    p._capflex_risk = dict.fromkeys(p._capflex_risk, 2.0)
    with pytest.raises(ValueError, match="stale"):
        stamp_capflex_risk(p, _capflex(), gdx)
