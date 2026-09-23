# tests/blocks/gtap/test_floors.py
#
# El piso de precios es una REGLA DEL DOMINIO compartida: los bloques que
# declaran variables de precio tienen que aplicar el mismo piso o el gate de
# paridad se mueve. Vivia copiada palabra por palabra en 6 bloques (mas una
# septima copia muerta en closure.py) con sus dos constantes al lado.
#
# Estos tests fijan la regla y ademas impiden que la copia vuelva.
import ast
import pathlib

import pytest

from equilibria.blocks.gtap.floors import (
    PRICE_FLOOR_ABS,
    PRICE_FLOOR_REL,
    price_floor,
)

GTAP_BLOCKS = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "equilibria"
    / "blocks"
    / "gtap"
)


@pytest.mark.parametrize(
    "init,esperado",
    [
        (None, PRICE_FLOOR_ABS),  # sin init
        (0.0, PRICE_FLOOR_ABS),  # init cero
        (-5.0, PRICE_FLOOR_ABS),  # init negativo
        (1e-9, PRICE_FLOOR_ABS),  # relativo por debajo del absoluto -> gana el absoluto
        (1.0, 1e-3),  # escala ~1 -> gana el relativo
        (100.0, 0.1),  # escala grande -> gana el relativo
    ],
)
def test_el_piso_es_el_maximo_de_absoluto_y_relativo(init, esperado):
    assert price_floor(init) == pytest.approx(esperado)


def test_el_piso_nunca_baja_del_absoluto():
    """Invariante que los bloques asumen al pasar el resultado como `lower`."""
    for init in (None, -1e9, 0.0, 1e-30, 1e-9, 1.0, 1e9):
        assert price_floor(init) >= PRICE_FLOOR_ABS


def test_las_constantes_son_las_del_monolito():
    assert PRICE_FLOOR_ABS == 1e-8
    assert PRICE_FLOOR_REL == 1e-3


def test_ningun_bloque_vuelve_a_copiar_la_regla():
    """Si vuelve a aparecer una copia local, este test la caza.

    Es el candado: la regla se comparte importando `floors`, no re-declarando
    `_FLOOR`/`_REL` ni redefiniendo `_price_floor` dentro del bloque.
    """
    copias = []
    for f in sorted(GTAP_BLOCKS.glob("*.py")):
        if f.name == "floors.py":
            continue
        arbol = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(arbol):
            if isinstance(n, ast.FunctionDef) and n.name.endswith("price_floor"):
                copias.append(f"{f.name}:{n.lineno} redefine {n.name}()")
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id in {"_FLOOR", "_REL"}:
                        copias.append(f"{f.name}:{n.lineno} re-declara {t.id}")
    assert not copias, (
        "la regla del piso volvio a copiarse; importala de floors.py:\n  "
        + "\n  ".join(copias)
    )


def test_los_bloques_que_aplican_piso_lo_importan():
    """Y lo hacen desde `floors`, no de otro bloque (evita re-exportar en cadena)."""
    esperados = {
        "factor.py",
        "income.py",
        "trade_armington_bilateral.py",
        "production_supply.py",
        "demand_utility.py",
    }
    importan = set()
    for f in sorted(GTAP_BLOCKS.glob("*.py")):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if (
                isinstance(n, ast.ImportFrom)
                and n.module
                and n.module.endswith("gtap.floors")
            ):
                importan.add(f.name)
    assert esperados <= importan, f"dejaron de importar floors: {esperados - importan}"


def test_ningun_bloque_reescribe_la_regla_a_mano():
    """El candado por NOMBRE no basta: `max(1e-8, 1e-3*x)` escrito a pelo lo
    esquivaba (verificado). Este caza la FORMA, se llame como se llame.

    Solo mira la expresion exacta de la regla, no cualquier 1e-8 suelto: en
    `blocks/gtap/` hay ~80 literales de esos que son otros pisos legitimos.
    """
    inline = []
    for f in sorted(GTAP_BLOCKS.glob("*.py")):
        if f.name == "floors.py":
            continue
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            # max(<1e-8>, <1e-3> * <algo>) en cualquier orden de argumentos
            if not (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "max"):
                continue
            if len(n.args) != 2:
                continue
            tiene_abs = any(
                isinstance(a, ast.Constant) and a.value == PRICE_FLOOR_ABS
                for a in n.args
            )
            tiene_rel = any(
                isinstance(a, ast.BinOp)
                and isinstance(a.op, ast.Mult)
                and any(
                    isinstance(o, ast.Constant) and o.value == PRICE_FLOOR_REL
                    for o in (a.left, a.right)
                )
                for a in n.args
            )
            if tiene_abs and tiene_rel:
                inline.append(f"{f.name}:{n.lineno} reescribe la regla a mano")
    assert not inline, (
        "la regla del piso esta escrita a mano; usa price_floor():\n  "
        + "\n  ".join(inline)
    )
