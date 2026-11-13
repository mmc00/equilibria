"""Shared 3x3 ifSUB=1 two-photo capture for the period-prep tests (one solve per session)."""

from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _newton_mumps() -> dict:
    path = ROOT / "scripts" / "gtap" / "perf_budget_20x41.py"
    spec = importlib.util.spec_from_file_location("perf_budget_20x41", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NEWTON_MUMPS


def _closure_tag(ifsub: int, savf: str, mode: str) -> str:
    path = ROOT / "scripts" / "family" / "capture_prep.py"
    spec = importlib.util.spec_from_file_location("capture_prep", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.closure_tag(ifsub, savf, mode)


def reference_dir(
    dataset: str, ifsub: int = 1, savf: str = "capFix", mode: str = "gtap"
) -> Path:
    return (
        Path.home()
        / ".cache/equilibria/family/reference"
        / dataset
        / _closure_tag(ifsub, savf, mode)
    )


@cache
def capture_dir(
    dataset: str = "gtap7_3x3",
    linsolve: str = "direct",
    ifsub: int = 1,
    savf: str = "capFix",
) -> Path:
    """Photos of ``dataset``: the saved Pyomo reference when there is one (larger
    datasets), else captured once per session into a temp dir (in-process solve with
    Pyomo's ``linsolve``: a reference only matches a family run that uses the same one,
    since the stopping point of the Newton loop depends on its trajectory)."""
    import atexit
    import os
    import shutil
    import tempfile

    from equilibria.family import prep_capture
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    ref = reference_dir(dataset, ifsub, savf)
    if dataset != "gtap7_3x3" and ref.is_dir() and any(ref.glob("001_A_*.pkl")):
        return ref
    out = Path(tempfile.mkdtemp(prefix=f"prep_{dataset}_"))
    # Photos run to hundreds of MB on 15x10: never leave them behind.
    atexit.register(shutil.rmtree, out, True)
    env = {
        **_newton_mumps(),
        "EQUILIBRIA_GTAP_TR_LINSOLVE": linsolve,
        prep_capture.ENV: str(out),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    prep_capture.reset()
    try:
        _, code = _solve_shock(dataset, ifsub, savf, solve_nlp=True)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert code == 1
    return out


@cache
def recording(dataset: str = "gtap7_3x3", ifsub: int = 1):
    """The single-period recording of the route (``FamilyBackend``) the shock recipe asks
    which instrument cells a live row reads: from the family disk cache, 3x3 in a temp
    dir removed at exit."""
    import atexit
    import shutil
    import tempfile

    from equilibria.family.disk_cache import load_or_build
    from tests.family.test_prep_recipe import _params
    from tests.family.test_squared_system import _route_closure

    p = _params(dataset)
    if dataset == "gtap7_3x3":
        root = Path(tempfile.mkdtemp(prefix="records_3x3_"))
        atexit.register(shutil.rmtree, root, True)
    else:
        root = Path.home() / ".cache" / "equilibria" / "family"
    closure = _route_closure()
    if ifsub == 0:
        closure = closure.model_copy(update={"if_sub": False})
    backend, _ = load_or_build(p, closure, list(p.sets.r)[-1], root)
    return backend


@cache
def altax_gem() -> Path:
    """Burfisher's ALTAX run by GEMSIM (Gragg): ``solution.sl4``, ``updated.har``,
    ``summary.har``, ``decomp.har``, ``volume.har``. Found through ``equilibria.sources``."""
    from equilibria.sources import reference

    return reference("nus333", "gemsim", experiment="altax", method="gragg")


@cache
def altax_levels_har() -> Path:
    """sltoht's levels dump of that run (the ``.sl4`` has no ``PLEV``)."""
    from equilibria.sources import reference

    return (
        reference("nus333", "sltoht", experiment="altax", method="gragg")
        / "sl4levels.har"
    )


@cache
def altax_reference_dir() -> Path:
    """One folder with everything ``--compare`` reads: the GEMSIM run plus ``sl4levels.har``."""
    import shutil
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="altax_ref_"))
    for f in altax_gem().iterdir():
        if f.suffix in (".har", ".sl4"):
            shutil.copy(f, d / f.name)
    shutil.copy(altax_levels_har(), d / "sl4levels.har")
    return d
