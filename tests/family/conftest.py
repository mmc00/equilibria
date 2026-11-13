"""Slow tests are skipped by default in tests/family (opt in with ``-m slow`` or
``EQUILIBRIA_RUN_SLOW=1``). Only the items under this directory are touched, and a
single slow node id run without ``-m slow`` is skipped too."""

import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent

# The family route is the optional ``family`` extra (jax). CI installs it
# (``uv sync --dev --extra family`` in .github/actions/setup-test-env), so these tests
# run there. Without it (a plain ``uv sync --dev``) every test here is skipped; CI
# names the files one by one, which ``collect_ignore`` does not cover.
HAS_JAX = importlib.util.find_spec("jax") is not None


def _wants_slow(config: pytest.Config) -> bool:
    return (
        "slow" in (config.getoption("-m") or "")
        or os.environ.get("EQUILIBRIA_RUN_SLOW") == "1"
    )


def _honour_gdxdump_outside_path(items: list) -> None:
    """The root guard skips ``needs_gdxdump`` when ``gdxdump`` is not on PATH, but the
    code under test finds it the way ``locate_gdxdump`` does (EQUILIBRIA_GDXDUMP, PATH,
    the usual GAMS install dirs). Drop that skip when the library does find it, so a
    Mac with GAMS installed still runs these tests; CI (gdxdump job) puts it on PATH."""
    from equilibria.babel.gdx.gdxdump import locate_gdxdump

    if locate_gdxdump() is None:
        return
    for item in items:
        if item.path.is_relative_to(HERE) and item.get_closest_marker("needs_gdxdump"):
            item.own_markers = [
                m
                for m in item.own_markers
                if not (m.name == "skip" and "gdxdump" in str(m.kwargs.get("reason")))
            ]


# trylast: tests/conftest.py adds its solver skips first; this hook can then undo one.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    _honour_gdxdump_outside_path(items)
    if not HAS_JAX:
        no_jax = pytest.mark.skip(reason="needs the family extra (jax)")
        for item in items:
            if item.path.is_relative_to(HERE):
                item.add_marker(no_jax)
        return
    if _wants_slow(config):
        return
    skip = pytest.mark.skip(reason="slow: run with -m slow or EQUILIBRIA_RUN_SLOW=1")
    for item in items:
        if (
            item.path.is_relative_to(HERE)
            and item.get_closest_marker("slow") is not None
        ):
            item.add_marker(skip)


@pytest.fixture(scope="module")
def altax_run(tmp_path_factory):
    """One ``solve_altertax(nus333_dir())`` per module, shared by the Phase B tests."""
    pytest.importorskip("pyomo")
    from equilibria._local_refs import nus333_dir
    from equilibria.family.altertax.run import solve_altertax
    from tests.family._prep import _newton_mumps

    mp = pytest.MonkeyPatch()
    for k, v in {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        mp.setenv(k, v)
    mp.setenv("EQUILIBRIA_FAMILY_CACHE", str(tmp_path_factory.mktemp("altax_cache")))
    try:
        yield solve_altertax(nus333_dir(), cache_root=tmp_path_factory.mktemp("c2"))
    finally:
        mp.undo()
