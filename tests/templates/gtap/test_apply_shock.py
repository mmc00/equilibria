"""apply_shock: fija la celda del periodo shock de un instrumento registrado."""

import pytest


def test_factor_multiplica_solo_la_celda_shock(nus333_mp_model):
    from pyomo.environ import value

    from equilibria.templates.gtap.instruments import apply_shock

    m = nus333_mp_model
    k = ("USA", "CAPITAL")
    x = {t: float(value(m.aft[(*k, t)])) for t in ("base", "check", "shock")}
    try:
        got = apply_shock(m, "aft", k, factor=1.10)
        assert got == pytest.approx(x["shock"] * 1.10)
        assert float(value(m.aft[(*k, "shock")])) == pytest.approx(x["shock"] * 1.10)
        assert float(value(m.aft[(*k, "check")])) == x["check"]
        assert float(value(m.aft[(*k, "base")])) == x["base"]
        assert m.aft[(*k, "shock")].fixed
    finally:
        m.aft[(*k, "shock")].fix(x["shock"])


@pytest.mark.parametrize(
    ("name", "index", "kw", "msg"),
    [
        ("xft", ("USA", "CAPITAL"), {"factor": 1.1}, "not a registered instrument"),
        ("aft", ("USA", "labor"), {"factor": 1.1}, "LABOR"),
        ("aft", ("USA", "CAPITAL"), {}, "exactly one of factor/value"),
        ("aft", ("USA", "CAPITAL"), {"factor": 1.1, "value": 2.0}, "exactly one"),
        ("aft", ("USA", "CAPITAL"), {"factor": 0.0}, "must be > 0"),
    ],
)
def test_rechaza_con_mensaje_claro(nus333_mp_model, name, index, kw, msg):
    from equilibria.templates.gtap.instruments import apply_shock

    with pytest.raises(ValueError, match=msg):
        apply_shock(nus333_mp_model, name, index, **kw)
