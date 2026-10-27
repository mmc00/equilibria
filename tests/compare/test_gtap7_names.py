"""El diccionario de nombres GTAP (ADR-0009): cada nombre que equilibria comparte con
GAMS se declara una vez, y el diccionario esta COMPLETO contra los dos lados.

Oraculo independiente del codigo que se prueba:
  - el lado GAMS sale de los sidecars ``.row``/``.col`` de las fixtures que GAMS
    escribio (tests/fixtures/gtap7/<ds>/gams_<fase>.{row,col}), leidos aqui con un
    parser propio de diez lineas;
  - el lado equilibria sale de los componentes del modelo de bloques multiperiodo
    construido (Constraint/Var de Pyomo), no del diccionario.

Lo que se exige:
  1. Toda familia de ecuaciones y de variables de GAMS esta emparejada o declarada
     "solo en GAMS" con su motivo.
  2. Toda familia de equilibria esta emparejada o declarada "solo en equilibria".
  3. Una pareja tiene indices que cuadran: los de GAMS son los de equilibria mas
     los que el diccionario fija a un elemento (p. ej. ``xaceq(r,i,'hhd',t)`` es
     ``eq_xaa_hhd(r,i,t)``). Asi la regla por defecto no puede emparejar
     ``xgeq(r,t)`` con ``eq_xg(r,i,t)``, que no son la misma ecuacion.
  4. Ninguna fila del diccionario nombra una familia que no existe (filas viejas).
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/gtap7"
HAR_3X3 = ROOT / "datasets/gtap7_3x3"


def _family(line: str) -> str:
    return re.split(r"[\[(]", line.strip())[0]


def _ndims(line: str) -> int:
    m = re.search(r"\((.*)\)\s*$", line.strip())
    return len(m.group(1).split(",")) if m else 0


@cache
def _gams_families() -> dict[str, dict[str, int]]:
    """{'eq': {familia: n_indices}, 'var': {...}} de todas las fixtures GAMS."""
    out: dict[str, dict[str, int]] = {"eq": {}, "var": {}}
    files = sorted(FIXTURES.glob("*/gams_*.row")) + sorted(
        FIXTURES.glob("*/gams_*.col")
    )
    assert files, f"no hay sidecars .row/.col en {FIXTURES}"
    for f in files:
        kind = "eq" if f.suffix == ".row" else "var"
        for line in f.read_text().splitlines():
            if line.strip():
                out[kind].setdefault(_family(line), _ndims(line))
    return out


@cache
def _equilibria_families() -> dict[str, dict[str, int]]:
    """{'eq': {familia: n_indices}, 'var': {...}} del modelo de bloques 3x3."""
    from pyomo.environ import Constraint, Var

    from equilibria.templates.gtap.gtap_block_model import GTAPBlockMultiPeriodModel
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_parameters import GTAPParameters

    if not (HAR_3X3 / "basedata.har").exists():
        pytest.skip(f"no hay datos HAR en {HAR_3X3}")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=HAR_3X3 / "basedata.har",
        sets_path=HAR_3X3 / "sets.har",
        default_path=HAR_3X3 / "default.prm",
        baserate_path=HAR_3X3 / "baserate.har",
    )
    closure = GTAPClosureConfig(if_sub=False)
    mp = GTAPBlockMultiPeriodModel(
        p.sets, p, closure, residual_region=list(p.sets.r)[-1]
    )
    m = mp.build_sets()
    mp.build_vars(m)
    for period in ("base", "check", "shock"):
        mp.build_equations_intra(m, period)
    mp.build_equations_fisher(m)
    out: dict[str, dict[str, int]] = {"eq": {}, "var": {}}
    for c in m.component_objects(Constraint, active=None):
        out["eq"][c.local_name] = c.dim() if c.is_indexed() else 0
    for v in m.component_objects(Var):
        out["var"][v.local_name] = v.dim() if v.is_indexed() else 0
    return out


@pytest.fixture(scope="module")
def names():
    from equilibria.compare.names import load_names  # ty: ignore[unresolved-import]

    return load_names("gtap7")


@pytest.mark.parametrize("kind", ["eq", "var"])
def test_every_gams_family_is_accounted(names, kind):
    missing = [
        f for f in _gams_families()[kind] if names.lookup("gams", kind, f) is None
    ]
    assert not missing, (
        f"{len(missing)} familias de GAMS ({kind}) sin fila en el diccionario: {missing}"
    )


@pytest.mark.parametrize("kind", ["eq", "var"])
def test_every_equilibria_family_is_accounted(names, kind):
    fams = _equilibria_families()[kind]
    missing = [f for f in fams if names.lookup("equilibria", kind, f) is None]
    assert not missing, (
        f"{len(missing)} familias de equilibria ({kind}) sin fila en el diccionario: "
        f"{missing}"
    )


@pytest.mark.parametrize("kind", ["eq", "var"])
def test_pairs_line_up_their_indices(names, kind):
    gams, eqb = _gams_families()[kind], _equilibria_families()[kind]
    bad = []
    for f, n_gams in gams.items():
        e = names.lookup("gams", kind, f)
        if e is None or e.equilibria is None or e.equilibria not in eqb:
            continue
        if n_gams != eqb[e.equilibria] + len(e.gams_fixed):
            bad.append(
                f"{f}({n_gams}) <-> {e.equilibria}({eqb[e.equilibria]}) "
                f"+ fijos {dict(e.gams_fixed)}"
            )
    assert not bad, f"parejas cuyos indices no cuadran: {bad}"


def test_one_sided_rows_say_why(names):
    bad = [
        e
        for e in names.entries
        if (e.gams is None or e.equilibria is None) and not e.why
    ]
    assert not bad, f"filas de un solo lado sin motivo: {bad}"


def test_no_stale_rows(names):
    gams, eqb = _gams_families(), _equilibria_families()
    stale = [
        e
        for e in names.entries
        if (e.gams is not None and e.gams not in gams[e.kind])
        or (e.equilibria is not None and e.equilibria not in eqb[e.kind])
    ]
    assert not stale, f"filas que nombran una familia que no existe: {stale}"
