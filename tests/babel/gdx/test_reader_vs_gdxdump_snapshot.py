"""El lector GDX contra la foto de gdxdump de TODO el GDX (todos los campos).

Oraculos: ``<gdx>.gdxdump.json.gz``, escritos por gdxdump con
``tests/babel/gdx/oracles/gdxdump_snapshot.py`` y commiteados antes del arreglo.
No necesita GAMS para correr. Cinco GDX: el 3x3 ifsub0 y ifsub1; el NLP puro
(2.238 Eps, token 4); los Parameters de PEP2 (0,5, token 8); y los resultados de
PEP (escalares en 0 que el GDX no guarda y gdxdump imprime como 0).

Lo que medio este test el 2026-10-19 con el lector de main: los parametros volvian
vacios (buscaban su seccion por posicion, y en un GDX comprimido hay menos secciones
que simbolos); un registro con valor 0,5 o 2 (tokens 8 y 9) cortaba la seccion en
silencio; y una seccion guardada SIN comprimir dentro de un GDX comprimido (cabecera
``00`` + largo) se leia vacia (rorgeq, 0 de 2).
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import pytest

from equilibria.babel.gdx.reader import (
    get_equations,
    get_parameters,
    get_variables,
    read_equation_values,
    read_gdx,
    read_parameter_values,
    read_variable_values,
)

ROOT = Path(__file__).resolve().parents[3]
GDX = ROOT / "tests" / "fixtures" / "gtap7" / "gtap7_3x3" / "out_gtap_shock_ifsub0.gdx"
SNAP_DIR = ROOT / "tests" / "fixtures" / "gdxdump"
REF = ROOT / "src" / "equilibria" / "templates" / "reference"
CASES = {
    "gtap7_3x3_ifsub0": (GDX, GDX.with_name(GDX.name + ".gdxdump.json.gz")),
    "gtap7_3x3_ifsub1": (
        GDX.with_name("out_gtap_shock_ifsub1.gdx"),
        GDX.with_name("out_gtap_shock_ifsub1.gdx.gdxdump.json.gz"),
    ),
    "gtap7_nlp_eps": (
        ROOT / "tests" / "fixtures" / "gtap7_nlp" / "gtap7_3x3_pure_ifsub0.gdx",
        ROOT
        / "tests"
        / "fixtures"
        / "gtap7_nlp"
        / "gtap7_3x3_pure_ifsub0.gdx.gdxdump.json.gz",
    ),
    "pep2_parameters": (
        REF / "pep2" / "scripts" / "Parameters.gdx",
        SNAP_DIR / "pep2_parameters.gdx.gdxdump.json.gz",
    ),
    "pep_results": (
        REF / "pep" / "results.gdx",
        SNAP_DIR / "pep_results.gdx.gdxdump.json.gz",
    ),
}
FIELDS = ("level", "marginal", "lower", "upper", "scale")
SPECIAL = {
    "+Inf": math.inf,
    "-Inf": -math.inf,
    "Eps": 0.0,
    "NA": math.nan,
    "UNDF": math.nan,
}


def _value(text: str) -> float | None:
    """El numero que imprime gdxdump; None para un acronimo (no es un numero)."""
    if text in SPECIAL:
        return SPECIAL[text]
    try:
        return float(text)
    except ValueError:
        return None


def _same(ours: float, text: str) -> bool:
    ref = _value(text)
    if ref is None:
        return True  # acronimo: gdxdump imprime su nombre, no el codigo
    if math.isnan(ref):
        return math.isnan(ours)
    if math.isinf(ref):
        return ours == ref
    # gdxdump imprime 15 cifras significativas
    return abs(ours - ref) <= 1e-13 * max(1.0, abs(ref))


def _snap(case: str) -> dict[str, Any]:
    with gzip.open(CASES[case][1], "rt", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request: pytest.FixtureRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    return read_gdx(str(CASES[request.param][0])), _snap(request.param)


@pytest.fixture(scope="module")
def snap() -> dict[str, Any]:
    return _snap("gtap7_3x3_ifsub0")


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    return read_gdx(str(GDX))


def _compare(name: str, ours: dict[tuple[str, ...], list[float]], snap: dict) -> None:
    ref = snap[name]
    dim = ref["dim"]
    expected = {tuple(r[:dim]): r[dim:] for r in ref["records"]}
    assert ours.keys() == expected.keys(), (
        name,
        f"{len(ours)} registros vs {len(expected)} de gdxdump",
    )
    for key, texts in expected.items():
        for value, text in zip(ours[key], texts, strict=True):
            assert _same(value, text), (name, key, value, text)


def test_todos_los_parametros(case: tuple[dict, dict]) -> None:
    data, snap = case
    for p in get_parameters(data):
        values = read_parameter_values(data, p["name"])
        _compare(p["name"], {k: [v] for k, v in values.items()}, snap)


def test_todas_las_variables(case: tuple[dict, dict]) -> None:
    data, snap = case
    for v in get_variables(data):
        values = read_variable_values(data, v["name"])
        _compare(
            v["name"], {k: [r[f] for f in FIELDS] for k, r in values.items()}, snap
        )


def test_todas_las_ecuaciones(case: tuple[dict, dict]) -> None:
    data, snap = case
    for e in get_equations(data):
        values = read_equation_values(data, e["name"])
        _compare(
            e["name"], {k: [r[f] for f in FIELDS] for k, r in values.items()}, snap
        )


def test_valores_especiales_y_seccion_sin_comprimir(data: dict) -> None:
    """Los casos concretos que fallaban, nombrados."""
    assert read_parameter_values(data, "years") == {
        ("base",): 1.0,
        ("check",): 2.0,  # token 9
        ("shock",): 3.0,
    }
    assert read_parameter_values(data, "gap") == {  # seccion sin comprimir
        ("base",): 1.0,
        ("check",): 1.0,
        ("shock",): 1.0,
    }
    assert len(read_equation_values(data, "rorgeq")) == 2


def test_la_tabla_de_simbolos_lista_todo(case: tuple[dict, dict]) -> None:
    """#137: la tabla de simbolos ocupa mas de un bloque de 32768 bytes; la
    cabecera del segundo (01 + largo) quedaba metida en la descripcion de rorgeq
    y el parser se detenia en 418 de 450 (faltaban sam y 31 ecuaciones)."""
    data, snap = case
    ours = {
        s["name"]
        for s in get_parameters(data) + get_variables(data) + get_equations(data)
    }
    assert ours == set(snap)
