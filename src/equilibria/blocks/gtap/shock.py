"""GTAP SHOCK block: los instrumentos de shock como Vars fijas.

En GAMS los instrumentos de politica y tecnologia son variables fijadas por
periodo (lambdava, imptx, prdtx, ...; model.gms) o parametros indexados por t
(aft): el shock es `x.fx(...,'shock') = v`, sin tocar ecuaciones. Aca cada
instrumento es una Var que el compositor FIJA en su valor de benchmark y registra
en ``_exogenous_instruments``. El constructor multiperiodo la copia por periodo
como referencia viva (una Var fija no se pliega a literal, a diferencia de un
Param), y el driver no libera ni re-siembra un instrumento registrado. Aplicar un
shock es ``instruments.apply_shock(m, nombre, indice, factor=...)``.

Solo el CUERPO de las ecuaciones lee el instrumento; la calibracion (shares,
semillas, krat de eq_kstock) sigue leyendo el benchmark.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from equilibria.blocks.base import Block
from equilibria.blocks.gtap import _derived_params as dp
from equilibria.core.symbolic_equations import SymbolicEquation
from equilibria.core.variables import Variable

# Nombres de componente de cada instrumento. Fuente unica para el compositor,
# build_block_model y el driver.
SHOCK_INSTRUMENTS: tuple[str, ...] = ("lambdava", "aft")


class ShockBlock(Block):
    """Instrumentos de shock (sin ecuaciones propias)."""

    name: str = "GTAP_SHOCK"
    description: str = "GTAP shock instruments as fixed Vars (GAMS x.fx per period)"
    sets: Any = None
    params: Any = None

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["r", "a", "f"]

    def setup(self, set_manager, parameters, variables) -> list[SymbolicEquation]:
        regions = list(set_manager.get("r"))
        acts = list(set_manager.get("a"))
        lva = self.params.shifts.lambdava
        # avaall -> lambdava(r,a,t) (model.gms:38, :540, :547).
        variables["lambdava"] = Variable(
            name="lambdava",
            value=np.array(
                [[float(lva.get((r, a), 1.0)) for a in acts] for r in regions]
            ),
            domains=("r", "a"),
            domain="Reals",
            lower=float("-inf"),
            upper=float("inf"),
        )
        facs = list(set_manager.get("f"))
        aft = dp.aft_data(self.params, self.sets)
        # qe -> aft(r,fm,t): Parameter indexado por t en GAMS (model.gms:290),
        # xft = aft*(pft/pabs)**etaf (model.gms:1073). Instrumento del shock de
        # dotacion; el benchmark para calibrar es aft0 (FactorBlock).
        variables["aft"] = Variable(
            name="aft",
            value=np.array(
                [[float(aft.get((r, f), 0.0)) for f in facs] for r in regions]
            ),
            domains=("r", "f"),
            domain="Reals",
            lower=float("-inf"),
            upper=float("inf"),
        )
        return []
