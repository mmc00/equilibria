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
    ],
)
def test_env_refusals_name_the_setting(env, needle):
    with pytest.raises(NotImplementedError, match=needle):
        run(env=env)


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
