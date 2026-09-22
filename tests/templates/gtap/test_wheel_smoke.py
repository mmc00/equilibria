# tests/templates/gtap/test_wheel_smoke.py
#
# Contrato de distribucion: lo que se publica en PyPI tiene que RESOLVER, no
# solo importar.
#
# Historia: `gtap_multiperiod_driver` cargaba el solver PATH por ruta desde
# `scripts/`, que no viaja en el wheel — instalado con pip reventaba con
# FileNotFoundError en la primera linea del solve (#72). Ningun test lo vio
# porque todos corren desde el checkout, donde `scripts/` SIEMPRE existe.
#
# Estos dos tests cubren ese hueco sin necesidad de construir un wheel:
#   1. el solver se importa como MODULO del paquete (no por ruta), y
#   2. las dependencias que el solve necesita de verdad estan DECLARADAS.
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402


def test_solver_ships_inside_the_package():
    """El solver PATH tiene que ser importable como modulo del paquete.

    Si alguien lo vuelve a cargar por ruta desde scripts/, este test sigue
    pasando pero el de abajo (`_load_run_gtap`) falla: es el que ata el driver
    al modulo empaquetado.
    """
    mod = importlib.import_module("equilibria.solver.path_capi")
    assert callable(mod._run_path_capi_nonlinear_full)

    pkg_root = (
        pathlib.Path(importlib.import_module("equilibria").__file__).resolve().parent
    )
    assert pathlib.Path(mod.__file__).resolve().is_relative_to(pkg_root), (
        f"el solver vive fuera del paquete: {mod.__file__}"
    )

    # _closure_patches (cuadrado MCP) tiene que viajar con el.
    patches = importlib.import_module("equilibria.solver._closure_patches")
    assert pathlib.Path(patches.__file__).resolve().is_relative_to(pkg_root)


def test_driver_loads_the_packaged_solver():
    """`_load_run_gtap()` resuelve al modulo del paquete, no a scripts/run_gtap.py."""
    from equilibria.templates.gtap import gtap_multiperiod_driver as driver

    assert driver._load_run_gtap().__name__ == "equilibria.solver.path_capi"


def test_pynumero_imports_with_declared_deps_only():
    """PyNumero (motor del Jacobiano) tiene que importar con las deps declaradas.

    `packaging` lo declara Pyomo solo en su extra "optional", pero sin el
    PyNumero lanza DeferredImportError y el solve devuelve residual=inf. En el
    checkout entra de rebote por mypy, asi que el agujero solo se ve en un venv
    limpio — medido con `pip install` del wheel a pelo.
    """
    importlib.import_module("packaging.version")

    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP  # noqa: F401


@pytest.mark.needs_path
@pytest.mark.slow
def test_full_multiperiod_solve_end_to_end(monkeypatch):
    """Solve completo 3x3 base->check->shock, medido en las filas Fisher de `m`.

    Mismo criterio que test_multiperiod_driver: el "code" del solver NO es el
    contrato — lo es que las ecuaciones evaluen sobre `m`. Exige
    reverse_numeric por lo mismo que aquel (ASL aborta bajo ifSUB).
    """
    monkeypatch.setenv("EQUILIBRIA_GTAP_JAC_MODE", "reverse_numeric")

    from pyomo.environ import value as pyo_value
    from test_multiperiod_sets import _load_3x3_params

    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p_alt = apply_altertax_elasticities(_load_3x3_params(), in_place=False)
    mp = GTAPMultiPeriodModel(
        p_alt.sets, p_alt, None, residual_region=list(p_alt.sets.r)[-1]
    )
    m = mp.build_sets()
    mp.build_vars(m)
    for t in ("base", "check", "shock"):
        mp.build_equations_intra(m, t)
    mp.build_equations_fisher(m)

    res = solve_multiperiod(m, p_alt, None)
    assert set(res) == {"base", "check", "shock"}

    tol = 5e-2
    for r in m.r:
        con = m.eq_rgdpmp[r, "shock"]
        body = pyo_value(con.body)
        lb = pyo_value(con.lower) if con.lower is not None else None
        ub = pyo_value(con.upper) if con.upper is not None else None
        if lb is not None and ub is not None and abs(lb - ub) < 1e-15:
            resid = abs(body - lb)
        elif lb is not None:
            resid = max(0.0, lb - body)
        elif ub is not None:
            resid = max(0.0, body - ub)
        else:
            resid = 0.0
        assert resid < tol, f"eq_rgdpmp[{r!r},'shock'] residual {resid:.6g} >= {tol}"

    assert any((pyo_value(m.rgdpmp[r, "base"]) or 0) > 0 for r in m.r), (
        "rgdpmp['base'] todo cero — `m` no se resolvio"
    )
