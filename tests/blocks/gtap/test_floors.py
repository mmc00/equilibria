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

from equilibria.blocks.gtap.declarations import (
    declare_price_var,
    declare_quantity_var,
)
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
    for f in sorted(GTAP_BLOCKS.rglob("*.py")):
        if f.name in ("floors.py", "declarations.py"):
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


def test_los_bloques_que_declaran_precios_pasan_por_el_declarador():
    """Quien declara un precio lo hace via `declare_price_var`, que es el unico
    sitio que aplica la regla.

    Antes este test comprobaba que los bloques IMPORTABAN `floors`, y se
    degrado: al mover los declaradores a `declarations.py`, `income` y
    `trade_armington_bilateral` dejaron de nombrar el piso en ninguna parte y
    el test seguia verde importando otra cosa. Ahora mira el uso real.
    """
    esperados = {
        "factor.py",
        "income.py",
        "trade_armington_bilateral.py",
        "production_supply.py",
        "demand_utility.py",
    }
    usan = set()
    for f in sorted(GTAP_BLOCKS.rglob("*.py")):
        if f.name == "declarations.py":
            continue
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Name) and n.id == "declare_price_var":
                usan.add(f.name)
    assert esperados <= usan, (
        f"dejaron de declarar precios via declare_price_var: {esperados - usan}"
    )


def test_ningun_bloque_reescribe_la_regla_a_mano():
    """El candado por NOMBRE no basta: `max(1e-8, 1e-3*x)` escrito a pelo lo
    esquivaba (verificado). Este caza la EXPRESION, se llame como se llame la
    funcion que la envuelve.

    ALCANCE, medido variante a variante: caza `max(...)` y `np.maximum(...)`
    con los dos literales en cualquier orden, con la multiplicacion en
    cualquier orden, y con cualquier notacion del numero (`1E-8`, `0.001`) —
    compara VALORES de constantes, no texto. Se le escapan la regla partida en
    dos sentencias y la que pasa por constantes locales con otro nombre; para
    eso esta el candado por nombre de arriba. No persigue dataflow a proposito.

    Solo mira la expresion exacta de la regla, no cualquier 1e-8 suelto: en
    `blocks/gtap/` hay ~100 literales de esos que son otros pisos legitimos.
    """
    inline = []
    for f in sorted(GTAP_BLOCKS.rglob("*.py")):
        if f.name in ("floors.py", "declarations.py"):
            continue
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            # max/np.maximum(<1e-8>, <1e-3> * <algo>), argumentos en cualquier
            # orden. `np.maximum` importa: los bloques vectorizan, asi que es la
            # forma que escribiria alguien aqui (ya se usa en demand_utility).
            if not isinstance(n, ast.Call):
                continue
            nombre = getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            if nombre not in ("max", "maximum", "fmax"):
                continue
            if len(n.args) < 2:
                continue
            tiene_abs = any(
                isinstance(a, ast.Constant) and a.value == PRICE_FLOOR_ABS
                for a in n.args
            )

            # <algo> * 1e-3 (cualquier orden) o su equivalente <algo> / 1000
            def _es_rel(a: ast.expr) -> bool:
                if not isinstance(a, ast.BinOp):
                    return False
                if isinstance(a.op, ast.Mult):
                    return any(
                        isinstance(o, ast.Constant) and o.value == PRICE_FLOOR_REL
                        for o in (a.left, a.right)
                    )
                if isinstance(a.op, ast.Div):
                    return (
                        isinstance(a.right, ast.Constant)
                        and a.right.value == 1.0 / PRICE_FLOOR_REL
                    )
                return False

            tiene_rel = any(_es_rel(a) for a in n.args)
            if tiene_abs and tiene_rel:
                inline.append(f"{f.name}:{n.lineno} reescribe la regla a mano")
    assert not inline, (
        "la regla del piso esta escrita a mano; usa price_floor():\n  "
        + "\n  ".join(inline)
    )


def test_declarar_precio_aplica_el_piso_celda_a_celda():
    """Cada celda lleva SU piso, no uno global: es lo que distingue precio de
    cantidad y la razon de que el helper viva junto a la regla."""
    import numpy as np

    variables: dict = {}
    init = np.array([1.0, 100.0, 0.0])
    declare_price_var(variables, "p", ("r",), init)
    v = variables["p"]
    assert list(np.asarray(v.lower)) == [1e-3, 0.1, PRICE_FLOOR_ABS]
    assert v.domain == "NonNegativeReals"


def test_declarar_cantidad_reproduce_las_tres_firmas_viejas():
    """Los 5 bloques tenian `_q` con TRES firmas; los defaults de aqui las
    reproducen exactamente. Si un default cambia, un dominio cambia en silencio.
    """
    import numpy as np

    init = np.array([1.0, 2.0])
    variables: dict = {}

    declare_quantity_var(variables, "sin_nada", ("r",), init)
    assert variables["sin_nada"].lower == 0.0
    assert variables["sin_nada"].domain == "NonNegativeReals"

    declare_quantity_var(variables, "con_lower", ("r",), init, lower=1e-8)
    assert variables["con_lower"].lower == 1e-8

    declare_quantity_var(
        variables, "libre", ("r",), init, lower=float("-inf"), dom="Reals"
    )
    assert variables["libre"].domain == "Reals"
    assert variables["libre"].lower == float("-inf")


def test_el_declarador_vive_en_un_solo_sitio():
    """Invariante: el cuerpo del declarador existe una sola vez.

    Los tres candados anteriores enumeraban EVASIONES (nombres `_q`/`_price`,
    `def` no-async, llamada por nombre desnudo) y los tres fallaron. Este no
    enumera: lista QUE variables se declaran a mano, con su motivo. Cualquier
    cuerpo nuevo —bajo otro nombre, en un lambda, en un `async def`— aparece
    como un nombre que no esta en la lista y el test cae.

    Un CONTEO no bastaba: sustituir una var legitima por otra dejaba el numero
    igual y el candado pasaba (verificado). Por eso la lista es de nombres.
    """
    # Vars con cota o dominio PROPIOS, que no encajan en ningun declarador.
    ESPERADOS = {
        # xft es Reals con cota 1e-8; pwfact/pfact escalares con piso 1e-3
        # (literal, ver #76); rorc/rore libres (Reals sin cota).
        # kstock y arent SI encajaban en declare_quantity_var y se migraron:
        # el comentario viejo decia "cota o dominio propios" y era falso.
        "factor.py": {"pfact", "pwfact", "rorc", "rore", "xft"},
        # dintx/mintx con lower=-0.999; xw totalmente libre.
        "trade_armington_bilateral.py": {"dintx", "mintx", "xw"},
        # pnum/pwfact piso 1e-3, walras libre, y los 6 agregados Fisher que se
        # declaran en dos bucles (mfw_* y mfr_*, Reals sin cota).
        "closure.py": {"pnum", "pwfact", "walras", "<bucle:202>", "<bucle:217>"},
        # pet piso 1e-3 (init 1.0), xet cantidad.
        "trade_cet.py": {"pet", "xet"},
    }
    reales: dict[str, set[str]] = {}
    for f in sorted(GTAP_BLOCKS.rglob("*.py")):
        if f.name in ("floors.py", "declarations.py"):
            continue
        arbol = ast.parse(f.read_text(encoding="utf-8"))
        # Un alias de import (`Variable as _Var`) esquivaba el candado
        # (verificado): resolvemos todos los nombres que apuntan a Variable.
        alias = {"Variable"}
        for n in ast.walk(arbol):
            if isinstance(n, ast.ImportFrom):
                alias.update(
                    a.asname or a.name for a in n.names if a.name == "Variable"
                )
        nombres = set()
        for n in ast.walk(arbol):
            if not (isinstance(n, ast.Call) and getattr(n.func, "id", None) in alias):
                continue
            literal = next(
                (
                    k.value.value
                    for k in n.keywords
                    if k.arg == "name" and isinstance(k.value, ast.Constant)
                ),
                None,
            )
            nombres.add(literal if literal is not None else f"<bucle:{n.lineno}>")
        if nombres:
            reales[f.name] = nombres
    assert reales == ESPERADOS, (
        "cambio el reparto de declaraciones a mano.\n"
        f"  sobran : { {k: sorted(v - ESPERADOS.get(k, set())) for k, v in reales.items() if v - ESPERADOS.get(k, set())} }\n"
        f"  faltan : { {k: sorted(ESPERADOS[k] - reales.get(k, set())) for k in ESPERADOS if ESPERADOS[k] - reales.get(k, set())} }\n"
        "Si anadiste una var con cota propia, subela aqui con su motivo; "
        "si reescribiste un declarador, usa declare_price_var/"
        "declare_quantity_var."
    )


def test_el_directorio_vigilado_existe():
    """Si `GTAP_BLOCKS` deja de resolver, `rglob` no falla: devuelve vacio y los
    candados pasan sin mirar nada. Esto convierte ese silencio en un fallo."""
    assert GTAP_BLOCKS.is_dir(), f"no resuelve: {GTAP_BLOCKS}"
    assert len(list(GTAP_BLOCKS.rglob("*.py"))) > 5
