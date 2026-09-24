"""Como se declara una variable en un bloque GTAP.

Dos formas, y la diferencia es el PISO: un precio lleva piso relativo celda a
celda (`floors.price_floor` sobre su init), una cantidad lleva una cota
uniforme. Ese cuerpo de 8 lineas estaba copiado en los cinco bloques que
declaran variables —dos helpers locales cada uno, `_q` y `_price`, 10
definiciones para 78 llamadas— mas una copia suelta en `kapEnd`.

Vivio un tiempo dentro de `floors.py`, pero `declare_quantity_var` no aplica
ningun piso: estaba en un modulo llamado como la regla que no usa. Aqui el
modulo se llama por lo que hace, y `floors` vuelve a ser solo la regla.

Las vars con cota o dominio PROPIOS (xft, kstock, dintx/mintx, xw, walras, los
agregados Fisher) no pasan por aqui y se declaran a mano en su bloque;
`tests/blocks/gtap/test_floors.py` lleva la lista con el motivo de cada una.
"""

from __future__ import annotations

import numpy as np

from equilibria.blocks.gtap.floors import price_floor
from equilibria.core.variables import Variable


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


__all__ = ["declare_price_var", "declare_quantity_var"]
