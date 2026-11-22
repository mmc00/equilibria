"""The scripts moved by PR 3b of gtap-on-core run on the core driver (``Exercise``)
and solve to the same point as on main. Oracle:
``tests/scripts/golden/pr3b_scripts_main.json.gz`` (recorded on main before the move;
its recorder was deleted by PR 4)."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import pytest
from tests._golden import mismatches, model_levels

ROOT = Path(__file__).resolve().parents[2]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/gtap" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_scripts_take_no_route() -> None:
    """The core is the only route: neither script offers another."""
    for name, fn in (
        ("run_burfisher", "solve_exercise"),
        ("compare_gempack_levels", "solve"),
    ):
        assert "route" not in inspect.signature(getattr(_script(name), fn)).parameters


GOLDEN = ROOT / "tests/scripts/golden/pr3b_scripts_main.json.gz"
# A full solve on another machine (Linux CI vs the Mac that recorded the golden):
# ~6e-12 relative on normal cells, ~2e-11 absolute on residuals near zero (walras).
# The real changes this PR met were >= 1.5e-8 relative (and a sign flip).
SOLVE_REL, SOLVE_ABS = 1e-9, 1e-10
REF = "data/gtap7/gtap7_3x3/gams_mcp_gtap_ifsub{}_fbep-header/out.gdx"


def _golden() -> dict:
    import gzip
    import json

    return json.loads(gzip.decompress(GOLDEN.read_bytes()))


def _gempack_blocks(calibrated: bool):
    return _script("measure_gempack_blocks")._solve(
        "gtap7_3x3", calibrated, ROOT / REF.format(1)
    )


def _ifsub(ifsub: int):
    return _script("verify_ifsub_equivalence")._solve("gtap7_3x3", ifsub)


def _seed_and_solve():
    m, res = _script("seed_and_solve")._solve_gtap(
        "gtap7_3x3", ROOT / REF.format(0), False
    )
    return m, {k: r["code"] for k, r in res.items()}


def _dump_py():
    from pyomo.environ import SolverFactory

    if not SolverFactory("ipopt").available(exception_flag=False):  # its solve is NLP
        pytest.skip("ipopt not available")
    # as recorded: the script's own GDX path (absent here: an unseeded solve)
    return _script("dump_py_solution_gdx").solve("gtap7_3x3")


def _demo():
    spec = importlib.util.spec_from_file_location(
        "demo", ROOT / "examples/demo_gtap_pep.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gtap_tariff_demo()


# case id (as recorded) -> () -> (solved Pyomo model, codes in the old form)
RUN = {
    "gempack_blocks[gtap7_3x3-default]": lambda: _gempack_blocks(False),
    "gempack_blocks[gtap7_3x3-calibrated]": lambda: _gempack_blocks(True),
    "ifsub_equivalence[gtap7_3x3-ifsub0]": lambda: _ifsub(0),
    "ifsub_equivalence[gtap7_3x3-ifsub1]": lambda: _ifsub(1),
    "seed_and_solve_gtap[gtap7_3x3-ifsub0-shock]": _seed_and_solve,
    "dump_py[gtap7_3x3]": _dump_py,
    "demo_gtap[nus333]": _demo,
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """As the recorder: no seed cache, the scripts' imports on the path."""
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    from equilibria._local_refs import path_capi_src

    for p in (ROOT / "scripts/gtap", path_capi_src()):
        monkeypatch.syspath_prepend(str(p))


def _needs_data() -> None:
    from equilibria.babel.gdx.gdxdump import locate_gdxdump
    from equilibria.sources import has_dataset

    if not has_dataset("gtap7_3x3") or not (ROOT / REF.format(1)).exists():
        pytest.skip("gtap7_3x3 or its GAMS references not present")
    if locate_gdxdump() is None:  # the seed reads the GDX through it
        pytest.skip("gdxdump not on the PATH (comes with GAMS)")


@pytest.mark.needs_gdxdump  # the seed reads the GAMS GDX through it
@pytest.mark.parametrize("case", sorted(RUN))
def test_script_on_core_equals_main(case: str) -> None:
    _needs_data()
    want = _golden()[case]
    m, codes = RUN[case]()
    assert codes == want["codes"]
    bad = mismatches(model_levels(m), want["levels"], rel=SOLVE_REL, abs_tol=SOLVE_ABS)
    assert not bad, f"{len(bad)} cells off, first: {bad[:10]}"


@pytest.mark.needs_gdxdump  # the seed reads the GAMS GDX through it
def test_seed_and_solve_gtap_builds_twice(monkeypatch) -> None:
    """One hand build (the seed: sentinel before, residual tail) and the core's own
    build for the solve; a third build is minutes on the large datasets."""
    _needs_data()
    from equilibria.templates.gtap import gtap_block_model as gbm

    built = []
    cls = gbm.GTAPBlockMultiPeriodModel

    class Counting(cls):
        def __init__(self, *a, **k):
            built.append(1)
            super().__init__(*a, **k)

    monkeypatch.setattr(gbm, "GTAPBlockMultiPeriodModel", Counting)
    out = _script("seed_and_solve").run_seed_and_solve_gtap(
        "gtap7_3x3", ROOT / REF.format(0), "shock", 3, False
    )
    assert out["code"] == 1
    assert len(built) == 2
