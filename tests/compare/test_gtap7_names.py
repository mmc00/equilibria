"""El diccionario de nombres GTAP (ADR-0009): cada nombre que equilibria comparte con
GAMS se declara una vez, y el diccionario esta COMPLETO contra los dos lados.

Oraculo independiente del codigo que se prueba:
  - el lado GAMS sale de los bloques ``Variables`` y ``Equations`` de los ``.gms``
    de las fixtures (tests/fixtures/gtap7/<ds>/comp_*_ifsub{0,1}.gms), leidos aqui
    con un parser propio. Las declaraciones no dependen de ifSUB: cubren las dos
    formulaciones, incluidas pdp/pmp/pm/pefob/... que la fixture ``.nl`` (ifSUB=1)
    no trae;
  - el lado equilibria sale de los componentes del modelo de bloques multiperiodo
    construido (Constraint/Var de Pyomo) con ifSUB=0 y con ifSUB=1, no del
    diccionario.

Lo que se exige:
  1. Toda familia de ecuaciones y de variables de GAMS tiene su fila: emparejada
     o declarada "solo en GAMS" con su motivo. No hay regla por defecto: un nombre
     igual en los dos lados tambien lleva fila.
  2. Toda familia de equilibria tiene su fila, igual.
  3. Una pareja tiene indices que cuadran: los de GAMS son los de equilibria mas
     los que el diccionario fija a un elemento (p. ej. ``xaceq(r,i,'hhd',t)`` es
     ``eq_xc(r,i,t)``), menos los que solo tiene equilibria (``walras`` es escalar
     en GAMS y va por periodo en equilibria). Asi no se puede emparejar
     ``xgeq(r,t)`` con ``eq_xg(r,i,t)``, que no son la misma ecuacion.
  4. Ninguna fila del diccionario nombra una familia que no existe (filas viejas).
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

from equilibria.sources import dataset_dir

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/gtap7"
HAR_3X3 = dataset_dir("gtap7_3x3")


_DECL = re.compile(r"(?i)^\s*(variables|equations)\b")
_ITEM = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?")


@cache
def _gams_families() -> dict[str, dict[str, int]]:
    """{'eq': {familia: n_indices}, 'var': {...}} declarados en los .gms de las fixtures."""
    out: dict[str, dict[str, int]] = {"eq": {}, "var": {}}
    files = sorted(FIXTURES.glob("*/comp_*.gms"))
    assert files, f"no hay .gms en {FIXTURES}"
    for f in files:
        kind = None
        for line in f.read_text().splitlines():
            if kind is None:
                m = _DECL.match(line)
                if m:
                    kind = "var" if m.group(1).lower() == "variables" else "eq"
                continue
            s = line.strip()
            if s.startswith(";"):
                kind = None
                continue
            if not s or s.startswith("*"):
                continue
            m = _ITEM.match(s)
            if m:
                n = len(m.group(2).split(",")) if m.group(2) else 0
                out[kind].setdefault(m.group(1), n)
            if s.endswith(";"):
                kind = None
    # Anti-vacuidad: si el parser deja de leer los bloques, que no pase en verde.
    assert len(out["eq"]) > 80 and len(out["var"]) > 120, {
        k: len(v) for k, v in out.items()
    }
    return out


@cache
def _equilibria_families() -> dict[str, dict[str, int]]:
    """{'eq': {familia: n_indices}, 'var': {...}} del modelo de bloques 3x3, ifSUB 0 y 1."""
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
    out: dict[str, dict[str, int]] = {"eq": {}, "var": {}}
    for if_sub in (False, True):
        closure = GTAPClosureConfig(if_sub=if_sub)
        mp = GTAPBlockMultiPeriodModel(
            p.sets, p, closure, residual_region=list(p.sets.r)[-1]
        )
        m = mp.build_sets()
        mp.build_vars(m)
        for period in ("base", "check", "shock"):
            mp.build_equations_intra(m, period)
        mp.build_equations_fisher(m)
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
        if n_gams + len(e.equilibria_extra) != eqb[e.equilibria] + len(e.gams_fixed):
            bad.append(
                f"{f}({n_gams}) <-> {e.equilibria}({eqb[e.equilibria]}) "
                f"+ fijos {dict(e.gams_fixed)} - extra {list(e.equilibria_extra)}"
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


# Parejas que una regla por defecto (``<x>eq`` -> ``eq_<x>``) haria mal: el
# diccionario tiene que decir la correcta. Leidas en las definiciones del .gms y
# de los bloques (handoff 2026-10-27).
_TRAPS = [
    ("eq", "xgeq", "eq_xg_agg"),  # pg*xg = yg; eq_xg es xa(r,i,'gov')
    ("eq", "xieq", "eq_xiagg"),  # pi*xi = yi; eq_xi es xa(r,i,'inv')
    ("eq", "xseq", "eq_xseq"),  # CET de la oferta; eq_xs es pseq
    ("eq", "xfteq", "eq_xfteq"),  # xft = aft*(pft/pabs)**etaf; eq_xft es otra
    ("var", "xd", "xda"),  # xd(r,i,aa); el xd(r,i) de equilibria es un agregado
    ("var", "xg", "xg_agg"),  # xg(r,t); el xg(r,i) de equilibria es xa(r,i,'gov')
    ("var", "pp", "pp_rai"),  # pp(r,a,i); el pp(r,a) de equilibria es otro
]


@pytest.mark.parametrize(("kind", "gams", "equilibria"), _TRAPS)
def test_default_rule_traps(names, kind, gams, equilibria):
    e = names.lookup("gams", kind, gams)
    assert e is not None and e.equilibria == equilibria, (gams, e)


def test_no_implicit_default_rule(names):
    """Un nombre sin fila no se resuelve solo (ADR-0009: explicito para todo)."""
    assert names.lookup("gams", "eq", "nadaeq") is None
    assert names.lookup("equilibria", "eq", "eq_nada") is None
