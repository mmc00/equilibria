"""The per-period CDE household share recal reproduces GAMS's alphaa.

GAMS (cal block, loop t0):
    alphaa = (xcshr/bh) * ((yc/pop)/pa)**bh * uh**(-eh*bh) / sum(j, xcshr(j)/bh(j))

With the prior period's values unchanged, the rebuilt zcons row must hold at
the GAMS zcons level. The denominator is sum(xcshr/bh), not the sum of the full
terms; the two differ once bh varies across commodities and yc/pop != 1
(gtap7_15x10 India: the old denominator left the check row 37 off).
"""

from types import SimpleNamespace

from pyomo.environ import ConcreteModel, Constraint, Var, value

from equilibria.templates.gtap.gtap_multiperiod_driver import _recalibrate_alphaa_hhd

BH = {"a": 0.5, "b": 0.9}
EH = {"a": 1.2, "b": 0.8}
XCSHR = {"a": 0.4, "b": 0.6}
POP, YC, PA, UH = 100.0, 2.0, 1.0, 1.0
PERIODS = ("base", "check")


def _gams_zcons(i: str) -> float:
    denom = sum(XCSHR[j] / BH[j] for j in BH)
    alphaa = (
        (XCSHR[i] / BH[i]) * ((YC / POP) / PA) ** BH[i] * UH ** (-EH[i] * BH[i])
    ) / denom
    return alphaa * BH[i] * PA ** BH[i] * UH ** (EH[i] * BH[i]) * (YC / POP) ** (-BH[i])


def _model() -> tuple[ConcreteModel, SimpleNamespace]:
    m = ConcreteModel()
    cells = [("r", i, t) for i in BH for t in PERIODS]
    m.zcons = Var(cells, initialize=lambda m, r, i, t: _gams_zcons(i))
    m.xcshr = Var(cells, initialize=lambda m, r, i, t: XCSHR[i])
    m.pa = Var([("r", i, "hhd", t) for i in BH for t in PERIODS], initialize=PA)
    m.uh = Var([("r", t) for t in PERIODS], initialize=UH)
    m.yc = Var([("r", t) for t in PERIODS], initialize=YC)
    m.eq_zcons = Constraint(cells, rule=lambda m, r, i, t: m.zcons[r, i, t] == 0)
    params = SimpleNamespace(
        elasticities=SimpleNamespace(
            subpar={("r", i): v for i, v in BH.items()},
            incpar={("r", i): v for i, v in EH.items()},
        ),
        benchmark=SimpleNamespace(pop={"r": POP}),
    )
    return m, params


def test_rebuilt_zcons_row_holds_at_the_gams_level():
    m, params = _model()

    assert _recalibrate_alphaa_hhd(m, params, "check", "base") == len(BH)

    for i in BH:
        row = getattr(m, f"_eq_zcons_recal_r_{i}_check")
        assert abs(value(row.body) - value(row.upper)) < 1e-12, i
