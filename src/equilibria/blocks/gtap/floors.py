"""El piso numerico que el monolito aplica a los precios.

Es una regla del dominio, no una utilidad: el monolito no deja que un precio
baje de `max(1e-8, 1e-3 * init)`, y los seis bloques que declaran variables de
precio tienen que aplicar EL MISMO piso o el gate de paridad se mueve.

Vivia copiada palabra por palabra en los seis (mas una septima copia muerta en
`closure.py`), con sus dos constantes al lado: dieciocho declaraciones para una
sola regla. Los cuerpos coincidian, pero los docstrings ya habian divergido en
la procedencia — tres decian monolito 5295-5356 y tres 5298-5379 — que es como
empieza a derivar una copia antes de que el numero cambie.

PROCEDENCIA (medida, no deducida): el piso del monolito es UN solo sitio,
`templates/gtap/gtap_model_equations.py:4454-4458`, con sus constantes
nombradas en 4450-4451 (`MIN_QUANTITY`, `GAMS_REL_LOWER_BOUND`). Los tres
rangos que citaban las copias (5295-5356, 5298-5379 y el 5298-5385 de
`blocks/gtap/__init__.py:77`) NO contienen el piso: esa zona es construccion de
ecuaciones. Las tres referencias estaban caducadas.

DIFERENCIA CONOCIDA con el monolito, anterior a este modulo y deliberada: el
monolito hace `continue` cuando `init <= 0` y deja la variable SIN cota; los
bloques vectorizan sobre el array entero, asi que esas celdas reciben
`PRICE_FLOOR_ABS`. El gate de paridad corre verde con esa diferencia.
"""

from __future__ import annotations

import numpy as np

from equilibria.core.variables import Variable

#: Piso absoluto: ningun precio baja de aqui, ni siquiera con init<=0.
PRICE_FLOOR_ABS = 1e-8

#: Piso relativo al valor inicial, para precios cuya escala no es ~1.
PRICE_FLOOR_REL = 1e-3


def price_floor(init: float | None) -> float:
    """Piso del monolito: max(1e-8, 1e-3*init) para init>0.

    `gtap_model_equations.py:4454-4458` (_set_relative_positive_lower_bound).
    """
    if init is None or init <= 0.0:
        return PRICE_FLOOR_ABS
    return max(PRICE_FLOOR_ABS, PRICE_FLOOR_REL * float(init))


__all__ = [
    "PRICE_FLOOR_ABS",
    "PRICE_FLOOR_REL",
    "declare_price_var",
    "declare_quantity_var",
    "price_floor",
]


def declare_price_var(variables: dict, name: str, doms, init) -> None:
    """Declara una variable de PRECIO: piso relativo celda a celda.

    El `lower` sale de `price_floor` aplicado sobre el init, asi que cada celda
    lleva su propio piso. Era el mismo cuerpo de 8 lineas copiado en los cinco
    bloques que declaran precios (38 llamadas), mas una copia suelta en kapEnd.

    `otypes=[float]` NO es decorativo: sin el, `np.vectorize` levanta
    `ValueError` sobre un array vacio, y `ptmg` se declara sobre el set de
    margenes (`_price("ptmg", ("m",), np.ones(nm))`). Hoy ningun dataset tiene
    ese set vacio (3, 5, 10 y 15 elementos en los cuatro), asi que era latente,
    no un fallo vivo.
    """
    variables[name] = Variable(
        name=name,
        value=init,
        domains=tuple(doms),
        domain="NonNegativeReals",
        lower=np.vectorize(price_floor, otypes=[float])(init),
        upper=float("inf"),
    )


def declare_quantity_var(
    variables: dict,
    name: str,
    doms,
    init,
    lower: float = 0.0,
    dom: str = "NonNegativeReals",
) -> None:
    """Declara una variable de CANTIDAD/NIVEL: cota uniforme, sin piso relativo.

    Los cinco bloques tenian este mismo cuerpo con TRES firmas distintas
    (unos sin `lower`, otros sin `dom`); los defaults de aqui reproducen las
    tres exactamente, asi que ningun llamador cambia de comportamiento.
    """
    variables[name] = Variable(
        name=name,
        value=init,
        domains=tuple(doms),
        domain=dom,
        lower=lower,
        upper=float("inf"),
    )
