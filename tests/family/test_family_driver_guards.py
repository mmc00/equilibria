"""The family route refuses what it cannot run, by name, and stays out of the Pyomo route."""

import gc
from typing import Any

import pytest

from equilibria.family import period_newton, run_setup

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
)


def run(env: dict[str, str] | None = None, **kw: Any) -> None:
    run_setup.check_supported(**{**GOOD, **kw}, env={**GOOD_ENV, **(env or {})})


def test_supported_config_passes():
    run()
    run(if_sub=False)  # ifSUB=0 is supported
    run(savf_flag="capFlex")  # capFlex with ifSUB=1 is supported
    run(savf_flag="capFlex", if_sub=False)  # and with ifSUB=0 (MAR-251)
    run(env={"EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"})


@pytest.mark.parametrize(
    "kw, needle",
    [
        ({"savf_flag": "capSFix"}, "savf_flag='capSFix'"),
        ({"savf_flag": "capShrFix"}, "savf_flag='capShrFix'"),
        ({"savf_flag": "capFixDp"}, "savf_flag='capFixDp'"),
        ({"mode": "altertax"}, "mode"),
        ({"skip_base_solve": False}, "skip_base_solve"),
        ({"seed_from_prior": True}, "seed_from_prior"),
        ({"mute_welfare": False}, "mute_welfare"),
        ({"solve_check": False}, "solve_check"),
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
    assert period_newton._newton(SS(), sq)[2] == 2
    res.success = True
    assert period_newton._newton(SS(), sq)[2] == 2  # residual 1.0 above the tolerance


def test_missing_solve_nlp_key_is_refused():
    env = {k: v for k, v in GOOD_ENV.items() if k != "EQUILIBRIA_GTAP_SOLVE_NLP"}
    with pytest.raises(NotImplementedError, match="SOLVE_NLP"):
        run_setup.check_supported(**GOOD, env=env)


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
        "shock" in p for p in inspect.signature(run_setup.check_supported).parameters
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
    with pytest.raises(NotImplementedError, match="savf_flag='capSFix'"):
        mod._solve_shock("gtap7_3x3", 1, "capSFix", solve_nlp=True, backend="family")
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
    import types

    from equilibria.family import cross_period, disk_cache, prep_square, sp_arrays
    from equilibria.family import run_setup as drv
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
        sets=types.SimpleNamespace(r=["a"]),
        set_va_subsidy_basis=lambda b: None,
        set_unit_sigma=lambda m: None,
    )
    monkeypatch.setattr(disk_cache, "load_or_build", load_or_build)
    monkeypatch.setattr(cross_period, "load_or_record", load_cross)
    monkeypatch.setattr(drv.MPArrays, "build", staticmethod(build))
    monkeypatch.setattr(rr, "resolve_residual_region", lambda *a, **k: "R")
    monkeypatch.setattr(sp_arrays, "sp_reference_arrays", lambda *a, **k: None)
    monkeypatch.setattr(
        prep_square.ConditionalMasks, "from_params", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(drv, "_driver_closure", lambda c: c)
    monkeypatch.delenv("EQUILIBRIA_GTAP_GC_FREEZE", raising=False)
    return params, keep


def test_prepare_run_leaves_the_freeze_count_alone_by_default(monkeypatch):
    from equilibria.family import run_setup as drv

    seen = {}
    params, _keep = _stub_prepare_run(monkeypatch, seen)
    # a spy, not gc.get_freeze_count(): an automatic gen-2 collection can change the count
    frozen = []
    from equilibria.family import run_setup

    real_now = run_setup._gc_freeze_now
    monkeypatch.setattr(
        run_setup, "_gc_freeze_now", lambda e=True: (frozen.append(e), real_now(e))
    )
    monkeypatch.setattr(gc, "freeze", lambda: frozen.append("gc.freeze"))
    drv.prepare_run(params, None, "R", jax_cache=False)
    assert frozen == [False]  # asked with freezing off; gc.freeze never reached


def test_prepare_run_freezes_the_loaded_records_before_mp_build(monkeypatch):
    from equilibria.family import run_setup as drv

    seen = {}
    params, _keep = _stub_prepare_run(monkeypatch, seen)
    gc.unfreeze()
    try:
        drv.prepare_run(params, None, "R", jax_cache=False, freeze=True)
        assert seen["at_mp_build"] >= 8000
    finally:
        gc.unfreeze()


def test_family_driver_import_does_not_load_pyomo():
    import subprocess
    import sys

    code = (
        "import sys, equilibria.family.run_setup, equilibria.family.period_newton; "
        "sys.exit('pyomo' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_capflex_still_refuses_auto_residual_region():
    from equilibria.templates.gtap.residual_region import resolve_residual_region
    from tests.family._oracle import comp_stat_closure

    closure = comp_stat_closure(True).model_copy(update={"savf_flag": "capFlex"})
    with pytest.raises(ValueError, match="only valid under capFix"):
        resolve_residual_region(None, None, closure, "auto")


@pytest.mark.parametrize(
    "dataset", ["gtap7_3x3", "gtap7_10x7", "gtap7_15x10", "gtap7_20x41"]
)
def test_family_and_pyomo_read_the_same_gdx(monkeypatch, dataset):
    """Both backends of ``_solve_shock`` hand the same ``ref_gdx`` on (no solve runs)."""
    import equilibria.templates.gtap as gtap
    import equilibria.templates.gtap.template as tpl

    mod = _parity_module()
    seen: dict[str, Any] = {}

    class Stop(Exception):
        pass

    def fake_gtap(p, ac, *, ref_gdx=None, backend="pyomo", **kw):
        seen[backend] = ref_gdx
        raise Stop

    class FakeParams:
        sets = type("S", (), {"r": ["A", "B"]})()

        def load_from_har(self, **kw):
            pass

    monkeypatch.setattr(tpl, "GTAP", fake_gtap)
    monkeypatch.setattr(gtap, "GTAPParameters", FakeParams)
    monkeypatch.setenv("EQUILIBRIA_GTAP_SOLVER", "scipy_newton_tr")
    monkeypatch.setenv("EQUILIBRIA_GTAP_TR_LINSOLVE", "direct")
    for backend in ("family", "pyomo"):
        with pytest.raises(Stop):
            mod._solve_shock(dataset, 1, "capFlex", solve_nlp=True, backend=backend)
    assert seen["family"] == seen["pyomo"] == mod._reference_gdx(dataset, 1)
    if seen["family"] is not None:
        name = (
            "gams_nlp_gtap_ifsub1_fbep-header"
            if dataset == "gtap7_20x41"
            else ("gams_mcp_gtap_ifsub1_fbep-header")
        )
        assert seen["family"].parent.name == name


def test_family_newton_analyses_mumps_with_the_values(monkeypatch):
    """The family route hands MUMPS the values at analysis (ICNTL(6)=7 uses them)."""
    import types

    import numpy as np

    import equilibria.solver.newton_tr as ntr

    class SS:
        x0 = np.zeros(1)

        def fscale(self):
            return np.ones(1)

        def F(self, z):
            return np.zeros(1)

    sq = types.SimpleNamespace(
        cells={"a": (0, 0, -np.inf, np.inf)}, col_names=["a"], row_names=["r0"]
    )
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        return types.SimpleNamespace(success=True, x=np.zeros(1), nit=1)

    monkeypatch.setattr(ntr, "newton_tr", fake)
    period_newton._newton(SS(), _any(sq))
    assert seen["mumps_values_at_analysis"] is True
