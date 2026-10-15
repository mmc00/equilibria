"""Presupuesto de trabajo del pipeline GTAP: conteos deterministas, no segundos.

El tiempo de pared depende de la maquina y del swap; estos conteos no. Cada test
fija el trabajo que hoy hace una corrida y falla con el numero exacto cuando un
cambio vuelve a meter trabajo repetido. Las regresiones que cada uno habria
atrapado (octubre 2026, 20x41 en la Mac: 9,5 min en septiembre, 12,1 en main,
8,0 tras arreglarlas):

- construcciones: calibrate_base armaba un segundo modelo multiperiodo
  identico (#119) y el modelo de referencia del base se armaba dos veces (#121).
- nodos de expresion: M_PM y su cadena se copiaban en cada fila que los usa
  (#118: 10x7 444.014 -> 295.370 nodos unicos; 20x41 -28% de memoria).
- iteraciones de Newton: el trabajo puede pasar de una fase a otra sin que
  cambie el resultado (ME9 asento la base en el build: el 20x41 paso de
  23/23/11 a 3/3/22 iteraciones base/check/shock).

Si un cambio necesita de verdad mas trabajo, se sube el presupuesto aca, en el
mismo PR, con la medicion que lo justifica.
"""

import contextlib
import io
import re

import pytest


def _quieto(fn, *a, **k):
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return fn(*a, **k)


def _contar(monkeypatch, owner, nombre, cuenta, clave):
    original = getattr(owner, nombre)

    def espia(*a, **k):
        cuenta[clave] += 1
        return original(*a, **k)

    monkeypatch.setattr(owner, nombre, espia)


@pytest.mark.needs_path
def test_una_corrida_construye_cada_modelo_lo_justo(monkeypatch):
    """3x3, base calibrada (camino de los gates): el modelo multiperiodo se arma
    UNA vez (la calibracion resuelve sobre el mismo) y el modelo de referencia de
    periodo simple DOS (base, compartido con la calibracion; y shock).

    En total el SP de bloques se arma TRES veces: una para el multiperiodo (sus
    Var y sus filas salen del mismo SP) y las dos referencias. build_vars armaba
    uno propio solo para leer las Var (20x41: ~8 % del tiempo, perfil 2026-10-14).
    """
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    import equilibria.templates.gtap.gtap_block_model as gbm
    import equilibria.templates.gtap.gtap_multiperiod_driver as drv

    # Con el seed cache, la calibracion no corre y el conteo no mide nada.
    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    monkeypatch.delenv("EQUILIBRIA_GTAP_MODEL_CACHE", raising=False)
    cuenta = {"multiperiodo": 0, "referencia": 0, "sp": 0}
    _contar(
        monkeypatch,
        gbm.GTAPBlockMultiPeriodModel,
        "build_equations_all_periods",
        cuenta,
        "multiperiodo",
    )
    _contar(monkeypatch, drv, "_build_sp_reference", cuenta, "referencia")
    _contar(monkeypatch, gbm, "build_block_single_period", cuenta, "sp")

    _, code = _quieto(_solve_shock, "gtap7_3x3", 1, "capFix", solve_nlp=True)

    assert code == 1
    assert cuenta == {"multiperiodo": 1, "referencia": 2, "sp": 3}


def _nodos_unicos(m) -> int:
    """Nodos no-hoja distintos (por objeto) en las filas activas: lo que se aloca."""
    from pyomo.environ import Constraint

    vistos: set[int] = set()
    for c in m.component_data_objects(Constraint, active=True):
        pila = [c.body]
        while pila:
            x = pila.pop()
            if not hasattr(x, "is_expression_type") or not x.is_expression_type():
                continue
            if id(x) in vistos:
                continue
            vistos.add(id(x))
            pila.extend(x.args)
    return len(vistos)


# Medido en main 1f3acd3 (2026-10-14); techo = medido + 2 %.
NODOS_MEDIDOS = {"gtap7_10x7": 295_370, "gtap7_15x10": 976_109}


@pytest.mark.parametrize("ds", sorted(NODOS_MEDIDOS))
def test_las_expresiones_no_se_inflan(ds):
    from pathlib import Path

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    d = Path(__file__).resolve().parents[3] / "datasets" / ds
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    closure = GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        if_sub=True,
        savf_flag="capFix",
        numeraire="pnum",
        va_subsidy_basis="gempack",
    )
    m, _ = _quieto(build_block_model, p, p.sets, closure, list(p.sets.r)[-1])
    techo = int(NODOS_MEDIDOS[ds] * 1.02)
    assert _nodos_unicos(m) <= techo


# Medido en main 1f3acd3 (2026-10-14), 10x7 base calibrada, igual en corridas
# repetidas: 3/3/6 con MUMPS y 3/3/14 con el directo de scipy. Las iteraciones
# dependen del solver lineal, asi que cada uno tiene su techo, con un poco de
# holgura por fase.
NEWTON_TECHO = {
    "mumps": {"base": 4, "check": 4, "shock": 8},
    "direct": {"base": 4, "check": 4, "shock": 16},
}
# La ruta libre Newton-TR (la de las cifras del paper, que usan MUMPS).
NEWTON_TR = {
    "EQUILIBRIA_GTAP_SOLVE_NLP": "1",
    "EQUILIBRIA_GTAP_SOLVER": "scipy_newton_tr",
    "EQUILIBRIA_GTAP_NLP_NO_JACSCALE": "1",
    "EQUILIBRIA_GTAP_TR_GATE": "1",
    "EQUILIBRIA_GTAP_TR_FTOL": "1e-7",
    "EQUILIBRIA_GTAP_SCIPY_MAXITER": "300",
    "EQUILIBRIA_GTAP_TR_DELTA0": "10.0",
    "EQUILIBRIA_GTAP_GMIN": "1e-9",
    "EQUILIBRIA_GTAP_TR_RELTOL": "1e-6",
}


def _linsolve() -> str:
    """MUMPS si esta (pymumps viene de conda-forge); si no, el directo de scipy.

    Pedir MUMPS sin pymumps no falla: cada iteracion registra "MUMPS FAILED" y
    el solve sigue por otra ruta que en 10x7 termina en una celda fijada negativa.
    """
    import importlib.util

    return "mumps" if importlib.util.find_spec("mumps") else "direct"


@pytest.mark.needs_path
def test_newton_no_hace_mas_iteraciones(monkeypatch):
    from tests.templates.gtap.test_gtap7_gempack_parity import _solve_shock

    monkeypatch.setenv("EQUILIBRIA_SEED_CACHE_DISABLE", "1")
    monkeypatch.delenv("EQUILIBRIA_GTAP_MODEL_CACHE", raising=False)
    for k, v in NEWTON_TR.items():
        monkeypatch.setenv(k, v)
    linsolve = _linsolve()
    monkeypatch.setenv("EQUILIBRIA_GTAP_TR_LINSOLVE", linsolve)
    techo = NEWTON_TECHO[linsolve]
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        _, code = _solve_shock("gtap7_10x7", 1, "capFix", solve_nlp=True)
    assert code == 1
    # Una linea "[nlp-square] scipy.root(...) done: ... nit=N" por resolucion.
    nits = [
        int(n)
        for n in re.findall(r"scipy\.root\([^)]*\) done: .*?nit=(\d+)", err.getvalue())
    ]
    assert len(nits) == len(techo), nits
    por_fase = dict(zip(techo, nits, strict=True))
    excedidas = {f: n for f, n in por_fase.items() if n > techo[f]}
    assert not excedidas, f"{linsolve}: iteraciones {por_fase} > techo {techo}"


# El benchmark del 20x41 (scripts/gtap/perf_budget_20x41.py) mide tiempo real y
# no corre en CI; lo que se testea aca es su veredicto.
def _veredicto():
    import importlib.util
    from pathlib import Path

    ruta = Path(__file__).resolve().parents[3] / "scripts/gtap/perf_budget_20x41.py"
    spec = importlib.util.spec_from_file_location("perf_budget_20x41", ruta)
    assert spec is not None and spec.loader is not None, ruta
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.veredicto


BASE = {"wall_min": 8.0, "peak_gb": 10.0}


@pytest.mark.parametrize(
    "medido,swap_mb,rc",
    [
        ({"code": 1, "wall_min": 8.5, "peak_gb": 10.5}, 0.0, 0),  # +6 %: dentro
        ({"code": 1, "wall_min": 9.0, "peak_gb": 10.0}, 0.0, 1),  # +12 % tiempo
        ({"code": 1, "wall_min": 8.0, "peak_gb": 11.5}, 0.0, 1),  # +15 % memoria
        ({"code": 1, "wall_min": 15.8, "peak_gb": 13.9}, 13_000.0, 2),  # con swap
        ({"code": 5, "wall_min": 3.0, "peak_gb": 9.0}, 0.0, 2),  # no convergio
    ],
)
def test_veredicto_del_benchmark(medido, swap_mb, rc):
    assert _veredicto()(medido, BASE, swap_mb)[0] == rc
