"""PeriodPreparer: que receta toca a cada periodo, y las herramientas de depuracion.

El CONTENIDO de cada receta (que el modelo quede igual antes de cada solve) lo
fija test_period_prep_snapshot.py; aqui solo la seleccion y los bordes.
"""

from __future__ import annotations

import pytest
from pyomo.environ import ConcreteModel, Constraint, Var

from equilibria.templates.gtap import period_prep
from equilibria.templates.gtap.period_debug import debug_before_solve
from equilibria.templates.gtap.period_prep import PeriodPreparer


def _prep(**kw):
    kw.setdefault("mode", "gtap")
    return PeriodPreparer(
        base_closure="BASE", alt_closure="ALT", residual_region="ROW", **kw
    )


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        _prep(mode="nlp")


def test_unknown_period_is_rejected():
    with pytest.raises(ValueError, match="no recipe"):
        _prep().recipe("forecast")


@pytest.mark.parametrize(
    "mode,period,closure",
    [
        ("gtap", "base", "base"),
        ("altertax", "base", "base"),
        ("gtap", "check", "base"),  # gtap puro resuelve el check con el cierre base
        ("altertax", "check", "altertax"),
        ("gtap", "shock", "base"),
        ("altertax", "shock", "altertax"),
    ],
)
def test_each_period_and_mode_has_its_recipe(mode, period, closure):
    assert _prep(mode=mode).recipe(period).closure == closure


def test_f35_copies_the_check_from_the_base_and_does_not_solve_it():
    r = _prep(base_calibrated=True).recipe("check")
    assert r.closure is None
    assert r.steps == (period_prep._copy_from_base,)


def test_solve_check_overrides_the_f35_copy():
    assert _prep(base_calibrated=True, solve_check=True).recipe("check").closure == (
        "base"
    )


@pytest.mark.parametrize(
    "base_calibrated,solve_check,prior",
    [(False, False, "check"), (True, False, "base"), (True, True, "check")],
)
def test_shock_seeds_from_the_check_unless_it_was_copied(
    base_calibrated, solve_check, prior
):
    """F3.5 sin solve_check: el check no se resolvio, el shock parte del base."""
    p = _prep(base_calibrated=base_calibrated, solve_check=solve_check)
    assert p.seed_prior("shock") == prior
    assert p.seed_prior("check") == "base"


def test_shock_always_builds_its_own_reference_model():
    """El modelo de referencia del shock lleva los params del shock: no reusa el
    del base (el check gtap si lo reusa)."""
    for mode in ("gtap", "altertax"):
        steps = _prep(mode=mode).recipe("shock").steps
        assert period_prep._replicate_fresh_sp_reference in steps
        assert period_prep._replicate_sp_reference not in steps


def test_gtap_recipes_do_not_recalibrate_and_altertax_ones_do():
    """GAMS gtap puro calibra una sola vez (t0); altertax recalibra cada periodo."""
    for period in ("check", "shock"):
        assert period_prep._recalibrate_shares not in _prep().recipe(period).steps
        assert (
            period_prep._recalibrate_shares
            in _prep(mode="altertax").recipe(period).steps
        )


# ── Depuracion antes del solve ─────────────────────────────────────────────


def _tiny():
    m = ConcreteModel()
    m.x = Var(initialize=1.0)
    m.c = Constraint(expr=m.x == 1.0)
    return m


def test_debug_is_a_no_op_without_env(monkeypatch):
    for var in (
        "EQUILIBRIA_DEBUG_EXPORT_NL_CHECK",
        "EQUILIBRIA_DEBUG_EXPORT_NL_BASE",
        "EQUILIBRIA_DEBUG_PROBE_PF_CHECK",
    ):
        monkeypatch.delenv(var, raising=False)
    debug_before_solve(_tiny(), "check")


@pytest.mark.parametrize("period", ["base", "check", "shock"])
def test_nl_export_works_for_any_period_and_stops(period, monkeypatch, tmp_path):
    """Las herramientas eran solo del check; ahora sirven a cualquier periodo,
    con el periodo en el nombre (EQUILIBRIA_DEBUG_EXPORT_NL_CHECK de siempre)."""
    out = tmp_path / "m.nl"
    var = f"EQUILIBRIA_DEBUG_EXPORT_NL_{period.upper()}"
    monkeypatch.setenv(var, str(out))
    with pytest.raises(RuntimeError, match=var):
        debug_before_solve(_tiny(), period)
    assert out.exists()


def test_a_hook_for_another_period_does_not_fire(monkeypatch, tmp_path):
    monkeypatch.setenv("EQUILIBRIA_DEBUG_EXPORT_NL_CHECK", str(tmp_path / "m.nl"))
    debug_before_solve(_tiny(), "base")  # no para
    assert not (tmp_path / "m.nl").exists()
