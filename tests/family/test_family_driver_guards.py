"""The family driver refuses what it cannot run, by name, and stays out of the Pyomo route."""

from typing import Any

import pytest

from equilibria.family import driver

GOOD_ENV = {
    "EQUILIBRIA_GTAP_SOLVER": "scipy_newton_tr",
    "EQUILIBRIA_GTAP_SOLVE_NLP": "1",
    "EQUILIBRIA_GTAP_TR_LINSOLVE": "mumps",
}
GOOD: dict[str, Any] = dict(  # noqa: C408
    if_sub=True,
    savf_flag="capFix",
    mode="gtap",
    skip_base_solve=True,
    mute_welfare=True,
    seed_from_prior=False,
    solve_check=True,
    shock_steps=None,
    shock_method="nonlinear",
)


def run(env: dict[str, str] | None = None, **kw: Any) -> None:
    driver.check_supported(**{**GOOD, **kw}, env={**GOOD_ENV, **(env or {})})


def test_supported_config_passes():
    run()
    run(env={"EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"})


@pytest.mark.parametrize(
    "kw, needle",
    [
        ({"if_sub": False}, "ifSUB=0"),
        ({"savf_flag": "capFlex"}, "savf_flag"),
        ({"mode": "altertax"}, "mode"),
        ({"skip_base_solve": False}, "skip_base_solve"),
        ({"seed_from_prior": True}, "seed_from_prior"),
        ({"mute_welfare": False}, "mute_welfare"),
        ({"solve_check": False}, "solve_check"),
        ({"shock_steps": (0.5, 1.0)}, "shock_steps"),
        ({"shock_method": "johansen"}, "shock_method"),
    ],
)
def test_argument_refusals_name_the_setting(kw, needle):
    with pytest.raises(NotImplementedError, match=needle):
        run(**kw)


@pytest.mark.parametrize(
    "env, needle",
    [
        ({"EQUILIBRIA_GTAP_SOLVER": "scipy_krylov"}, "EQUILIBRIA_GTAP_SOLVER"),
        ({"EQUILIBRIA_GTAP_SOLVER": ""}, "EQUILIBRIA_GTAP_SOLVER"),
        ({"EQUILIBRIA_GTAP_TR_LINSOLVE": "gmres"}, "TR_LINSOLVE"),
        ({"EQUILIBRIA_GTAP_SOLVE_NLP": ""}, "SOLVE_NLP"),
        ({"EQUILIBRIA_GTAP_REF_MODEL": "x"}, "REF_MODEL"),
        ({"EQUILIBRIA_GTAP_JAC_DUMP": "1"}, "JAC_DUMP"),
        ({"EQUILIBRIA_GTAP_WARM_START": "1"}, "WARM_START"),
        ({"EQUILIBRIA_GTAP_DIRECT_DEGEN_FIX": "1"}, "DIRECT_DEGEN_FIX"),
        ({"EQUILIBRIA_GTAP_BTF_PROBE": "1"}, "BTF_PROBE"),
        ({"EQUILIBRIA_DEBUG_EXPORT_NL_CHECK": "1"}, "EQUILIBRIA_DEBUG_EXPORT_NL_CHECK"),
        # Pyomo turns it into shock_steps / takes NEOS over scipy
        ({"EQUILIBRIA_GTAP_SHOCK_CONTINUATION": "0.5,1.0"}, "SHOCK_CONTINUATION"),
        ({"EQUILIBRIA_GTAP_SOLVE_NEOS": "conopt"}, "SOLVE_NEOS"),
        # rescale the equations the Pyomo route hands to Newton
        ({"EQUILIBRIA_GTAP_BENCH_SCALE": "1"}, "BENCH_SCALE"),
        ({"EQUILIBRIA_GTAP_ROW_EQUIL": "1"}, "ROW_EQUIL"),
        ({"EQUILIBRIA_SEED_FROM_GDX": "x.gdx"}, "SEED_FROM_GDX"),
        # the family route fixes ICNTL(6)=7 by design
        ({"EQUILIBRIA_GTAP_MUMPS_ICNTL6": "0"}, "ICNTL6"),
    ],
)
def test_env_refusals_name_the_setting(env, needle):
    with pytest.raises(NotImplementedError, match=needle):
        run(env=env)


def test_icntl6_refusal_explains_the_fixed_value():
    with pytest.raises(NotImplementedError, match=r"ICNTL\(6\)=7"):
        run(env={"EQUILIBRIA_GTAP_MUMPS_ICNTL6": "0"})


def _any(x: Any) -> Any:
    return x


def test_failed_newton_has_code_2_like_pyomo(monkeypatch):
    import types

    import numpy as np

    import equilibria.solver.newton_tr as ntr

    class SS:
        x0 = np.zeros(2)

        def fscale(self):
            return np.ones(2)

        def F(self, z):
            return np.ones(2)

        def J(self, z):
            return None

    sq = _any(
        types.SimpleNamespace(
            cells={"a": (0, 0, -np.inf, np.inf), "b": (0, 0, -np.inf, np.inf)},
            col_names=["a", "b"],
            row_names=["r0", "r1"],
        )
    )
    res = types.SimpleNamespace(success=False, x=np.zeros(2), nit=3)
    monkeypatch.setattr(ntr, "newton_tr", lambda *a, **k: res)
    assert driver._newton(SS(), sq)[2] == 2
    res.success = True
    assert driver._newton(SS(), sq)[2] == 2  # residual 1.0 above the tolerance


def test_missing_solve_nlp_key_is_refused():
    env = {k: v for k, v in GOOD_ENV.items() if k != "EQUILIBRIA_GTAP_SOLVE_NLP"}
    with pytest.raises(NotImplementedError, match="SOLVE_NLP"):
        driver.check_supported(**GOOD, env=env)


def test_active_overwrite_hooks_are_refused():
    from equilibria.blocks.gtap.overwrite import overwrite

    @overwrite(object)
    def _hook(*a):
        return None

    try:
        with pytest.raises(NotImplementedError, match="overwrite"):
            run()
    finally:
        overwrite.clear()


def test_user_shocks_are_not_a_refusal_reason():
    # check_supported has no shock argument: any ShockBlock instrument goes through
    # apply_shock, as in the Pyomo route.
    import inspect

    assert not any(
        "shock" in p and p not in ("shock_steps", "shock_method")
        for p in inspect.signature(driver.check_supported).parameters
    )


def _parity_module():
    import importlib.util
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "templates/gtap/test_gtap7_gempack_parity.py"
    )
    spec = importlib.util.spec_from_file_location("_gempack_parity_mod", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_solve_shock_family_route_refuses_before_any_work(monkeypatch):
    mod = _parity_module()
    monkeypatch.delenv("EQUILIBRIA_GTAP_SOLVER", raising=False)
    with pytest.raises(NotImplementedError, match="EQUILIBRIA_GTAP_SOLVER"):
        mod._solve_shock("gtap7_3x3", 1, "capFix", solve_nlp=True, backend="family")
    with pytest.raises(NotImplementedError, match="ifSUB=0"):
        mod._solve_shock("gtap7_3x3", 0, "capFix", solve_nlp=True, backend="family")
    with pytest.raises(NotImplementedError, match="savf_flag"):
        mod._solve_shock("gtap7_3x3", 1, "capFlex", solve_nlp=True, backend="family")
    monkeypatch.setenv("EQUILIBRIA_GTAP_SOLVER", "scipy_newton_tr")
    with pytest.raises(NotImplementedError, match="SOLVE_NLP"):
        mod._solve_shock("gtap7_3x3", 1, "capFix", solve_nlp=False, backend="family")


def test_env_selects_the_family_backend(monkeypatch):
    mod = _parity_module()
    monkeypatch.setenv("EQUILIBRIA_GTAP_BACKEND", "family")
    monkeypatch.delenv("EQUILIBRIA_GTAP_SOLVER", raising=False)
    with pytest.raises(NotImplementedError, match="family backend"):
        mod._solve_shock("gtap7_3x3", 1, "capFix", solve_nlp=True)


def test_unknown_backend_is_rejected():
    mod = _parity_module()
    with pytest.raises(ValueError, match="backend"):
        mod._solve_shock("gtap7_3x3", 1, "capFix", solve_nlp=True, backend="nope")


def test_pyomo_route_runs_no_family_driver_code(monkeypatch):
    mod = _parity_module()
    monkeypatch.delenv("EQUILIBRIA_GTAP_BACKEND", raising=False)
    calls = []
    monkeypatch.setattr(driver, "check_supported", lambda **k: calls.append("check"))
    monkeypatch.setattr(
        driver, "solve_gtap_family", lambda *a, **k: calls.append("solve")
    )
    # Stop the Pyomo route right after the switch, before the HAR is read.
    import equilibria.templates.gtap as gtap

    class Stop(Exception):
        pass

    def boom(*a, **k):
        raise Stop

    monkeypatch.setattr(gtap.GTAPParameters, "load_from_har", boom)
    with pytest.raises(Stop):
        mod._solve_shock("gtap7_3x3", 1, "capFix", solve_nlp=True)
    assert calls == []


def test_family_solve_runs_under_the_gc_threshold_of_the_pyomo_route(monkeypatch):
    """``solve_gtap_family`` wraps its work in ``_gc_relaxed_threshold`` with the same
    env control (``EQUILIBRIA_GTAP_GC_THRESHOLD``) as ``solve_multiperiod``."""
    import contextlib
    import gc

    from equilibria.family import driver as drv
    from equilibria.solver import gc_tuning

    seen = []

    @contextlib.contextmanager
    def spy(threshold0):
        seen.append(("enter", threshold0, gc.get_threshold()[0]))
        yield
        seen.append(("exit",))

    monkeypatch.setattr(drv, "_gc_relaxed_threshold", spy, raising=False)
    monkeypatch.setattr(drv, "check_supported", lambda **k: None)

    def inner(*a, **k):
        seen.append(("inner",))
        return "result"

    monkeypatch.setattr(drv, "_solve_family", inner)
    monkeypatch.setenv("EQUILIBRIA_GTAP_GC_THRESHOLD", "1234")
    assert drv.solve_gtap_family(None, None, "R") == "result"
    assert seen == [("enter", 1234, gc.get_threshold()[0]), ("inner",), ("exit",)]
    # the Pyomo route's helpers are the very same objects
    from equilibria.templates.gtap import gtap_multiperiod_driver as pyomo_drv

    assert pyomo_drv._gc_relaxed_threshold is gc_tuning._gc_relaxed_threshold
    assert pyomo_drv._gc_threshold_from_env is gc_tuning._gc_threshold_from_env


def test_family_solve_freezes_the_gc_and_undoes_it(monkeypatch):
    """Like ``solve_multiperiod``: the family solve runs under ``gc.freeze()`` (same
    ``EQUILIBRIA_GTAP_GC_FREEZE`` control) and unfreezes on exit, also on error."""
    import contextlib
    import gc

    import pytest

    from equilibria.family import driver as drv
    from equilibria.solver import gc_tuning
    from equilibria.templates.gtap import gtap_multiperiod_driver as pyomo_drv

    assert pyomo_drv._gc_frozen_permanent_graph is gc_tuning._gc_frozen_permanent_graph
    assert pyomo_drv._gc_freeze_from_env is gc_tuning._gc_freeze_from_env
    monkeypatch.setattr(drv, "check_supported", lambda **k: None)
    monkeypatch.setattr(
        drv, "_gc_relaxed_threshold", lambda t: contextlib.nullcontext()
    )
    gc.unfreeze()
    monkeypatch.delenv("EQUILIBRIA_GTAP_GC_FREEZE", raising=False)
    inside = []

    def inner(*a, **k):
        inside.append(gc.get_freeze_count())
        return "result"

    monkeypatch.setattr(drv, "_solve_family", inner)
    assert drv.solve_gtap_family(None, None, "R") == "result"
    assert inside[0] > 0
    assert gc.get_freeze_count() == 0

    def boom(*a, **k):
        inside.append(gc.get_freeze_count())
        raise RuntimeError

    monkeypatch.setattr(drv, "_solve_family", boom)
    with pytest.raises(RuntimeError):
        drv.solve_gtap_family(None, None, "R")
    assert inside[1] > 0
    assert gc.get_freeze_count() == 0

    # EQUILIBRIA_GTAP_GC_FREEZE=0 turns it off
    monkeypatch.setenv("EQUILIBRIA_GTAP_GC_FREEZE", "0")
    monkeypatch.setattr(drv, "_solve_family", inner)
    drv.solve_gtap_family(None, None, "R")
    assert inside[2] == 0


def test_solve_family_freezes_what_prepare_run_loaded(monkeypatch):
    """The records/cross/MPArrays built by ``prepare_run`` are long-lived too: they
    are frozen right after it, and the outer unfreeze releases everything."""
    import gc

    import pytest

    from equilibria.family import driver as drv

    class Stop(Exception):
        pass

    keep = []
    seen = {}

    def fake_prepare(*a, **k):
        seen["before"] = gc.get_freeze_count()
        keep.append([[i] for i in range(5000)])  # young long-lived containers
        return "run", "mp"

    def fake_calibrate(mp, run):
        seen["after"] = gc.get_freeze_count()
        raise Stop

    monkeypatch.setattr(drv, "prepare_run", fake_prepare)
    monkeypatch.setattr(drv, "calibrate_base_arrays", fake_calibrate)
    monkeypatch.setattr(drv, "check_supported", lambda **k: None)
    monkeypatch.delenv("EQUILIBRIA_GTAP_GC_FREEZE", raising=False)
    gc.unfreeze()
    with pytest.raises(Stop):
        drv.solve_gtap_family(None, None, "R")
    assert seen["after"] >= seen["before"] + 5000
    assert gc.get_freeze_count() == 0

    monkeypatch.setenv("EQUILIBRIA_GTAP_GC_FREEZE", "0")
    seen.clear()
    with pytest.raises(Stop):
        drv.solve_gtap_family(None, None, "R")
    assert seen["after"] == seen["before"] == 0


def test_gc_paused_disables_and_restores():
    import gc

    from equilibria.solver.gc_tuning import _gc_paused

    assert gc.isenabled()
    with _gc_paused():
        assert not gc.isenabled()
    assert gc.isenabled()
    gc.disable()  # a caller that had it disabled keeps it disabled
    try:
        with _gc_paused():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        gc.enable()
    try:
        with _gc_paused():
            raise RuntimeError
    except RuntimeError:
        pass
    assert gc.isenabled()


def test_family_backend_load_runs_with_gc_paused(tmp_path, monkeypatch):
    import gc

    from equilibria.backends.family_backend import FamilyBackend

    b = FamilyBackend()
    for k in FamilyBackend._SAVED:
        setattr(b, k, {})
    b.save(tmp_path / "r.pkl")
    import pickle

    states = []
    real = pickle.load
    monkeypatch.setattr(
        pickle, "load", lambda fh: (states.append(gc.isenabled()), real(fh))[1]
    )
    FamilyBackend.load(tmp_path / "r.pkl")
    assert states == [False]
    assert gc.isenabled()


def _stub_prepare_run(monkeypatch, seen):
    import gc
    import types

    from equilibria.family import cross_period, disk_cache, prep_recipe, prep_square
    from equilibria.family import driver as drv
    from equilibria.templates.gtap import residual_region as rr

    keep = []

    def load_or_build(*a, **k):
        keep.append([[i] for i in range(4000)])
        return types.SimpleNamespace(sets={}, var_domains={}), "state"

    def load_cross(*a, **k):
        keep.append([[i] for i in range(4000)])
        return "cross"

    def build(*a, **k):
        seen["at_mp_build"] = gc.get_freeze_count()
        return "mp"

    params = types.SimpleNamespace(
        sets=types.SimpleNamespace(r=["a"]), set_va_subsidy_basis=lambda b: None
    )
    monkeypatch.setattr(disk_cache, "load_or_build", load_or_build)
    monkeypatch.setattr(cross_period, "load_or_record", load_cross)
    monkeypatch.setattr(drv.MPArrays, "build", staticmethod(build))
    monkeypatch.setattr(rr, "resolve_residual_region", lambda *a, **k: "R")
    monkeypatch.setattr(prep_recipe, "sp_reference_arrays", lambda *a, **k: None)
    monkeypatch.setattr(
        prep_square.ConditionalMasks, "from_params", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(drv, "_driver_closure", lambda c: c)
    monkeypatch.delenv("EQUILIBRIA_GTAP_GC_FREEZE", raising=False)
    return params, keep


def test_prepare_run_leaves_the_freeze_count_alone_by_default(monkeypatch):
    import gc

    from equilibria.family import driver as drv

    seen = {}
    params, _keep = _stub_prepare_run(monkeypatch, seen)
    gc.unfreeze()
    before = gc.get_freeze_count()
    try:
        drv.prepare_run(params, None, "R", jax_cache=False)
        assert gc.get_freeze_count() == before
        assert seen["at_mp_build"] == before
    finally:
        gc.unfreeze()


def test_solve_family_passes_freeze_from_env(monkeypatch):
    from equilibria.family import driver as drv

    class Stop(Exception):
        pass

    got = {}

    def fake_prepare(*a, **k):
        got["freeze"] = k.get("freeze")
        raise Stop

    monkeypatch.setattr(drv, "prepare_run", fake_prepare)
    monkeypatch.setattr(drv, "check_supported", lambda **k: None)
    monkeypatch.delenv("EQUILIBRIA_GTAP_GC_FREEZE", raising=False)
    import gc

    import pytest

    with pytest.raises(Stop):
        drv.solve_gtap_family(None, None, "R")
    assert got["freeze"] is True
    monkeypatch.setenv("EQUILIBRIA_GTAP_GC_FREEZE", "0")
    with pytest.raises(Stop):
        drv.solve_gtap_family(None, None, "R")
    assert got["freeze"] is False
    assert gc.get_freeze_count() == 0


def test_prepare_run_freezes_the_loaded_records_before_mp_build(monkeypatch):
    import gc
    import types

    from equilibria.family import cross_period, disk_cache, prep_recipe, prep_square
    from equilibria.family import driver as drv
    from equilibria.templates.gtap import residual_region as rr

    keep = []

    def load_or_build(*a, **k):
        keep.append([[i] for i in range(4000)])
        return types.SimpleNamespace(sets={}, var_domains={}), "state"

    def load_cross(*a, **k):
        keep.append([[i] for i in range(4000)])
        return "cross"

    seen = {}

    def build(*a, **k):
        seen["at_mp_build"] = gc.get_freeze_count()
        return "mp"

    params = types.SimpleNamespace(
        sets=types.SimpleNamespace(r=["a"]), set_va_subsidy_basis=lambda b: None
    )
    monkeypatch.setattr(disk_cache, "load_or_build", load_or_build)
    monkeypatch.setattr(cross_period, "load_or_record", load_cross)
    monkeypatch.setattr(drv.MPArrays, "build", staticmethod(build))
    monkeypatch.setattr(rr, "resolve_residual_region", lambda *a, **k: "R")
    monkeypatch.setattr(prep_recipe, "sp_reference_arrays", lambda *a, **k: None)
    monkeypatch.setattr(
        prep_square.ConditionalMasks, "from_params", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(drv, "_driver_closure", lambda c: c)
    monkeypatch.delenv("EQUILIBRIA_GTAP_GC_FREEZE", raising=False)
    gc.unfreeze()
    try:
        drv.prepare_run(params, None, "R", jax_cache=False, freeze=True)
        assert seen["at_mp_build"] >= 8000
    finally:
        gc.unfreeze()


def test_family_driver_import_does_not_load_pyomo():
    import subprocess
    import sys

    code = "import sys, equilibria.family.driver; sys.exit('pyomo' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
