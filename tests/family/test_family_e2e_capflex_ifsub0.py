"""capFlex with ifSUB=0 on the family route (Burfisher's closure, MAR-251): every cell
equals a live Pyomo run of the same exercise through the core driver.

The frozen photos (``family.photos``) have no capFlex ifSUB=0 variant and cannot be
regenerated (MAR-218), so the oracle is the Pyomo route itself, solved in the session
with the same solver settings. Tolerance 1e-9, as the photo e2e tests."""

from __future__ import annotations

from typing import Literal

import pytest

pytest.importorskip("jax")
pytest.importorskip("pyomo")

from tests.family._period_levels import levels, mismatches, solver_env  # noqa: E402

DATASET = "gtap7_3x3"
REL = 1e-9


def _run(
    backend: str, ifsub: int, savf: Literal["capFix", "capFlex"], cache_root=None
) -> tuple[dict, dict]:
    import copy

    from equilibria.core.exercise import Exercise
    from equilibria.family.photos import photo_closure
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP
    from tests.family._data import load_params

    p = copy.deepcopy(load_params(DATASET))
    extra = {} if cache_root is None else {"EQUILIBRIA_FAMILY_CACHE": str(cache_root)}
    with solver_env(extra):
        ex = Exercise(
            template=GTAP(
                p,
                photo_closure(ifsub, savf),
                residual_region=list(p.sets.r)[-1],
                base_calibrated=True,
                ref_gdx=None,
                backend=backend,
            ),
            recipe=gtap_static(base_calibrated=True, solve_check=True),
        )
        out = ex.run()
    codes = {k: v["code"] for k, v in out.items()}
    return codes, levels(ex.model)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("family_cache")


@pytest.fixture(scope="module")
def pyomo_ifsub0():
    return _run("pyomo", 0, "capFlex")


@pytest.fixture(scope="module")
def family_ifsub0(cache):
    return _run("family", 0, "capFlex", cache)


def test_family_capflex_ifsub0_equals_pyomo(family_ifsub0, pyomo_ifsub0):
    f_codes, f = family_ifsub0
    p_codes, p = pyomo_ifsub0
    assert f_codes["shock"] == p_codes["shock"] == "optimal"
    assert len(f) > 0
    bad = mismatches(f, p, REL)
    assert not bad, f"{len(bad)} of {len(f)} cells differ, first: {bad[:10]}"


def test_other_closures_in_the_same_process_do_not_leak(family_ifsub0, cache):
    """A capFix run and a capFlex ifSUB=1 run in the same process and cache, then
    capFlex ifSUB=0 again: it equals its first run, and ifSUB=1 is another solution
    (no record or stamp of one closure is reused by another)."""
    _, first = family_ifsub0
    _run("family", 0, "capFix", cache)
    _, ifsub1 = _run("family", 1, "capFlex", cache)
    codes, again = _run("family", 0, "capFlex", cache)
    assert codes["shock"] == "optimal"
    assert not mismatches(again, first, 1e-12)
    assert mismatches(ifsub1, first, 1e-9)
