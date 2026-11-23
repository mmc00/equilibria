"""The tools before a period's solve (``period_debug``), and the base's Fisher
aggregates after the settled seed, on the core recipe."""

from __future__ import annotations

import pytest
from pyomo.environ import ConcreteModel, Constraint, Var

from equilibria.templates.gtap.period_debug import debug_before_solve

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


# ── F3.5: los agregados Fisher tras la siembra del base ────────────────────


def test_base_refreshes_the_fisher_aggregates_after_the_settled_seed():
    """F3.5 re-siembra el base desde el punto asentado: los agregados Fisher
    (``mq_*``, sumas DEFINIDAS de los niveles sembrados) se recalculan, como en el
    arranque en caliente del shock. Si no, el check resuelve contra las sumas del
    punto de construccion y el shock contra las refrescadas."""
    from pyomo.environ import value
    from tests.family._core import no_solve
    from tests.templates.gtap import _period_snapshot as snap

    from equilibria.core.driver import fly_period
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.recipe import gtap_static
    from equilibria.templates.gtap.template import GTAP

    if not snap.DATASET.exists():
        pytest.skip("gtap7_3x3 dataset not present")
    p = snap._load_params()
    closure = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        numeraire="pnum",
    )
    t = GTAP(p, closure, residual_region=list(p.sets.r)[-1], base_calibrated=True)
    recipe = no_solve(gtap_static("altertax", base_calibrated=True), stop="base")
    model = t.build(recipe.names)
    m = model.pyomo
    assert m._settled_seed
    fly_period(model, recipe, "base", template=t)
    worst = (0.0, "")
    for name, (var, defs) in m._fisher_aggregates.items():
        for idx, expr in defs.items():
            r = abs(float(var[idx].value) - float(value(expr)))
            if r > worst[0]:
                worst = (r, f"{name}{idx}")
    assert worst[0] < 1e-12, f"{worst[1]}: residual {worst[0]:.3e}"
