"""``FactorBlock.calibrate_base`` settles through the core driver (PR 3a of
gtap-on-core) and still gives main's settled seed, to 1e-12 relative (another
machine moves the golden's last bits).

The oracle (``golden/calibrate_base_main.json``) was recorded on main before the
switch by ``scripts/dev/record_calibrate_base.py``."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from tests._golden import mismatches

from equilibria.sources import has_dataset

GOLDEN = Path(__file__).parent / "golden" / "calibrate_base_main.json"
CASES = (
    ("gtap7_3x3-ifsub0", "gtap7_3x3", 0),
    ("gtap7_3x3-ifsub1", "gtap7_3x3", 1),
    ("nus333-ifsub0", "nus333", 0),
)


@pytest.fixture(autouse=True)
def _no_seed_cache(monkeypatch):
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")


def _recorder():
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "record_calibrate_base", root / "scripts/dev/record_calibrate_base.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _built(dataset: str, ifsub: int):
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    rec = _recorder()
    p = rec._params(dataset)
    rr = list(p.sets.r)[-1]
    cl = rec.closure(ifsub)
    m, _ = build_block_model(p, p.sets, cl, rr)
    return p, cl, rr, m


def _calibrate(p, cl, rr, m):
    from equilibria.blocks.gtap.factor import FactorBlock

    return FactorBlock(sets=p.sets, params=p).calibrate_base(
        p, p.sets, cl, rr, ref_gdx=None, model=m
    )


@pytest.mark.parametrize(("cid", "dataset", "ifsub"), CASES, ids=[c[0] for c in CASES])
def test_calibrate_base_equals_main(cid: str, dataset: str, ifsub: int) -> None:
    if not has_dataset(dataset):
        pytest.skip(f"{dataset} not present")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))["settle"][cid]
    seed = _calibrate(*_built(dataset, ifsub))
    got = {
        var: {repr(body): repr(float(v)) for body, v in cells.items()}
        for var, cells in seed.items()
    }
    assert mismatches(got, want) == []


def test_calibrate_base_restores_the_given_model() -> None:
    """It settles on the caller's model and gives it back as it was: values, fixed
    flags, bounds, domains, active rows, row bodies and mutable params."""
    if not has_dataset("gtap7_3x3"):
        pytest.skip("gtap7_3x3 not present")
    from equilibria.templates.gtap import model_state

    p, cl, rr, m = _built("gtap7_3x3", 0)

    def state():
        st = model_state.capture(m)
        return (
            [(id(vd), *rest) for vd, *rest in st.vars],
            [(id(cd), act, id(expr)) for cd, act, expr in st.filas],
            [(id(pd), v) for pd, v in st.params],
            [(id(c), a) for c, a in st.activos],
            sorted(st.atributos),
        )

    before = state()
    _calibrate(p, cl, rr, m)
    assert state() == before


def test_the_seed_cache_key_covers_the_core_settle_path() -> None:
    """The settle cache keys on the code of CODE_FOLDERS: every module the core
    settle runs must be in one, or a seed computed by other code is served."""
    from equilibria.backends import pyomo_periods
    from equilibria.blocks.gtap.fingerprint import CODE_FOLDERS, _package_root
    from equilibria.core import driver, steps
    from equilibria.templates.gtap import pyomo_periods as gtap_periods
    from equilibria.templates.gtap import recipe, template
    from equilibria.templates.gtap import steps as gtap_steps

    root = _package_root()
    for mod in (
        driver,
        steps,
        pyomo_periods,
        gtap_periods,
        recipe,
        template,
        gtap_steps,
    ):
        rel = Path(inspect.getfile(mod)).resolve().relative_to(root).as_posix()
        assert any(rel.startswith(f"{f}/") for f in CODE_FOLDERS), rel


def test_calibrate_base_settles_inside_the_run_scope(monkeypatch) -> None:
    """The settle flies inside ``GTAP.run_scope`` (the GC freeze of the built model
    over every period, as the old driver held it): without it the steps
    between solves walk the whole model again (minutes on 20x41)."""
    if not has_dataset("gtap7_3x3"):
        pytest.skip("gtap7_3x3 not present")
    import contextlib

    from equilibria.core import driver
    from equilibria.templates.gtap.template import GTAP

    depth = [0]
    flown_inside: list[bool] = []
    scope = GTAP.run_scope

    @contextlib.contextmanager
    def counting(self):
        depth[0] += 1
        try:
            with scope(self):
                yield
        finally:
            depth[0] -= 1

    fly = driver.fly

    def recording(*a, **k):
        flown_inside.append(depth[0] > 0)
        return fly(*a, **k)

    monkeypatch.setattr(GTAP, "run_scope", counting)
    monkeypatch.setattr(driver, "fly", recording)
    _calibrate(*_built("gtap7_3x3", 0))
    assert flown_inside == [True]
