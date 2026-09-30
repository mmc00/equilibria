"""fix_instrument_shock: fija la celda del periodo shock de un instrumento registrado."""

import pytest


def test_factor_multiplica_solo_la_celda_shock(nus333_mp_model):
    from pyomo.environ import value

    from equilibria.templates.gtap.instruments import fix_instrument_shock

    m = nus333_mp_model
    k = ("USA", "CAPITAL")
    x = {t: float(value(m.aft[(*k, t)])) for t in ("base", "check", "shock")}
    try:
        got = fix_instrument_shock(m, "aft", k, factor=1.10)
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
    from equilibria.templates.gtap.instruments import fix_instrument_shock

    with pytest.raises(ValueError, match=msg):
        fix_instrument_shock(nus333_mp_model, name, index, **kw)


@pytest.mark.parametrize(
    ("name", "index", "eq"),
    [
        ("aft", ("USA", "CAPITAL"), "eq_xfteq"),
        ("lambdava", ("USA", "SER"), "eq_va"),
        ("lambdava", ("USA", "SER"), "eq_pxeq"),
    ],
)
def test_rechaza_una_celda_que_ninguna_fila_lee(nus333_mp_model, name, index, eq):
    """Sin fila viva que lea la celda, el shock se perderia con code=1 (p.ej. aft de
    un factor con xftflag<=0: eq_xfteq no se genera). Debe fallar al fijarla."""
    from pyomo.environ import value

    from equilibria.templates.gtap.instruments import fix_instrument_shock

    m = nus333_mp_model
    idx = (*index, "shock")
    before = float(value(getattr(m, name)[idx]))
    getattr(m, eq)[idx].deactivate()
    try:
        with pytest.raises(ValueError, match="would not enter"):
            fix_instrument_shock(m, name, index, factor=1.1)
        assert float(value(getattr(m, name)[idx])) == before
    finally:
        getattr(m, eq)[idx].activate()


def test_rechaza_aft_con_xft_fijada(nus333_mp_model):
    """Con xft fijada el solver desactiva eq_xfteq y el shock de aft se pierde."""
    from equilibria.templates.gtap.instruments import fix_instrument_shock

    m = nus333_mp_model
    idx = ("USA", "CAPITAL", "shock")
    m.xft[idx].fix()
    try:
        with pytest.raises(ValueError, match="fixed"):
            fix_instrument_shock(m, "aft", ("USA", "CAPITAL"), factor=1.1)
    finally:
        m.xft[idx].unfix()


def test_factor_es_relativo_al_check_y_no_compone(nus333_mp_model):
    """factor= multiplica el valor de 'check' (benchmark), no el actual: aplicar el
    mismo shock dos veces da lo mismo que una."""
    from pyomo.environ import value

    from equilibria.templates.gtap.instruments import fix_instrument_shock

    m = nus333_mp_model
    k = ("USA", "CAPITAL")
    chk = float(value(m.aft[(*k, "check")]))
    before = float(value(m.aft[(*k, "shock")]))
    try:
        fix_instrument_shock(m, "aft", k, factor=1.10)
        got = fix_instrument_shock(m, "aft", k, factor=1.10)
        assert got == pytest.approx(chk * 1.10)
    finally:
        m.aft[(*k, "shock")].fix(before)


def test_solo_el_periodo_shock(nus333_mp_model):
    """Un shock en 'check' o 'base' no lo detecta el driver (sumaria el arancel) y
    la copia base->check lo pisaria: la API solo fija la celda 'shock'."""
    from equilibria.templates.gtap.instruments import fix_instrument_shock

    with pytest.raises(TypeError, match="period"):
        kw: dict = {"factor": 1.1, "period": "check"}
        fix_instrument_shock(nus333_mp_model, "aft", ("USA", "CAPITAL"), **kw)
