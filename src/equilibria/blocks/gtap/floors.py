"""El piso numerico que el monolito aplica a los precios.

Es una regla del dominio, no una utilidad: el monolito no deja que un precio
baje de `max(1e-8, 1e-3 * init)`, y los seis bloques que declaran variables de
precio tienen que aplicar EL MISMO piso o el gate de paridad se mueve.

Vivia copiada palabra por palabra en los seis (mas una septima copia muerta en
`closure.py`), con sus dos constantes al lado: dieciocho declaraciones para una
sola regla. Los cuerpos coincidian, pero los docstrings ya habian divergido en
la procedencia — tres decian monolito 5295-5356 y tres 5298-5379 — que es como
empieza a derivar una copia antes de que el numero cambie.

La procedencia buena es la que cubre AMBOS rangos: el monolito aplica el piso
en dos pasadas (declaracion de la variable y re-seed), y los dos rangos son las
dos pasadas, no dos versiones de la regla.
"""

from __future__ import annotations

#: Piso absoluto: ningun precio baja de aqui, ni siquiera con init<=0.
PRICE_FLOOR_ABS = 1e-8

#: Piso relativo al valor inicial, para precios cuya escala no es ~1.
PRICE_FLOOR_REL = 1e-3


def price_floor(init: float | None) -> float:
    """Piso de dos pasadas del monolito: max(1e-8, 1e-3*init) para init>0.

    Monolito 5295-5356 (declaracion) y 5298-5379 (re-seed).
    """
    if init is None or init <= 0.0:
        return PRICE_FLOOR_ABS
    return max(PRICE_FLOOR_ABS, PRICE_FLOOR_REL * float(init))


__all__ = ["PRICE_FLOOR_ABS", "PRICE_FLOOR_REL", "price_floor"]
