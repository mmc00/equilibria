"""simple_open: Hosoe's standard CGE model (gamslib stdcge), from blocks.

The goods and factors come from the SAM (``sam_path``); the book's SAM (bread,
milk, capital, labour) is the default. The model is the square system of the
.gms in levels, solved with a zero objective: at the GAMS NLP optimum it is the
same equilibrium.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyomo.environ as pyo
from pydantic import Field

from equilibria.backends.pyomo_backend import PyomoBackend
from equilibria.blocks.base import Block
from equilibria.blocks.demand import CobbDouglasHousehold
from equilibria.blocks.equilibrium import FactorMarket, GoodsMarket, Numeraire
from equilibria.blocks.institutions import ExternalSector, TaxGovernment, TaxRates
from equilibria.blocks.production import (
    CETOutput,
    CobbDouglasValueAdded,
    LeontiefComposite,
)
from equilibria.blocks.trade import ArmingtonComposite
from equilibria.core import Set
from equilibria.core.period_model import PeriodModel
from equilibria.core.recipe import Recipe
from equilibria.model import Model
from equilibria.templates.base import ModelTemplate
from equilibria.templates.simple_open.blocks import StdcgeInvestment
from equilibria.templates.simple_open.calibration import Calibration, calibrate
from equilibria.templates.simple_open.recipe import stdcge_static
from equilibria.templates.simple_open.sam import load_sam

# the tax rates are instruments: fixed variables a shock changes (ADR-0003)
INSTRUMENTS: tuple[str, ...] = ("tau_d", "tau_z", "tau_m")


def _default_sam() -> Path:
    from equilibria.sources import dataset_dir

    return dataset_dir("hosoe_2x2") / "sam.csv"


def _overwritten(blocks: Sequence[Block], overwrite: Sequence[Block]) -> list[Block]:
    """``blocks`` with each overwrite block in place of the one with its name."""
    by_name = {b.name: b for b in overwrite}
    stray = sorted(set(by_name) - {b.name for b in blocks})
    if stray:
        raise ValueError(
            f"@overwrite block(s) {', '.join(stray)} match no block of the template "
            f"({', '.join(b.name for b in blocks)})"
        )
    return [by_name.get(b.name, b) for b in blocks]


class SimpleOpenEconomy(ModelTemplate):
    """Hosoe, Gasawa and Hashimoto's standard CGE model (stdcge), from blocks."""

    name: str = Field(default="SimpleOpenEconomy")
    description: str = Field(default="Hosoe's standard CGE model (stdcge)")
    sam_path: Path = Field(default_factory=_default_sam)
    sigma: float = Field(default=2.0, gt=0, description="Armington elasticity")
    psi: float = Field(default=2.0, gt=0, description="CET elasticity")
    numeraire: str | None = Field(
        default=None,
        description="factor whose price is 1; None reads closure.yaml next to the SAM",
    )

    def numeraire_factor(self) -> str:
        """The numeraire factor: the field, or ``numeraire`` in the
        ``closure.yaml`` that sits next to the SAM (it belongs to the data)."""
        if self.numeraire is not None:
            return self.numeraire
        closure = self.sam_path.parent / "closure.yaml"
        if not closure.exists():
            raise ValueError(
                f"no numeraire: pass numeraire= or write {closure} "
                "with a line 'numeraire: <factor>'"
            )
        import yaml

        data = yaml.safe_load(closure.read_text()) or {}
        if "numeraire" not in data:
            raise ValueError(f"{closure} has no 'numeraire' entry")
        return str(data["numeraire"])

    def calibration(self) -> Calibration:
        return calibrate(load_sam(self.sam_path), sigma=self.sigma, psi=self.psi)

    def get_info(self) -> dict[str, Any]:
        """The template as configured: the SAM's goods and factors, not defaults."""
        sam = load_sam(self.sam_path)
        return {
            "name": self.name,
            "description": self.description,
            "sam": str(self.sam_path),
            "goods": list(sam.goods),
            "factors": list(sam.factors),
            "sigma": self.sigma,
            "psi": self.psi,
            "numeraire": self.numeraire_factor(),
            "blocks": [b.name for b in self.create_model().blocks],
        }

    def __repr__(self) -> str:
        sam = load_sam(self.sam_path)
        return (
            f"{self.__class__.__name__}("
            f"{len(sam.goods)} goods, {len(sam.factors)} factors)"
        )

    def create_model(
        self, calibration: Calibration | None = None, overwrite: Sequence[Block] = ()
    ) -> Model:
        cal = calibration or self.calibration()
        model = Model(name=self.name, description=self.description)
        model.add_sets(
            [
                Set(name="I", elements=cal.goods, description="goods"),
                Set(name="J", elements=cal.goods, description="sectors"),
                Set(name="F", elements=cal.factors, description="factors"),
            ]
        )
        kw: dict[str, Any] = {"values": cal.values, "start": cal.base}
        numeraire = self.numeraire_factor()
        blocks = (
            TaxRates(**kw),
            CobbDouglasValueAdded(**kw),
            LeontiefComposite(**kw),
            CETOutput(**kw),
            ArmingtonComposite(**kw),
            ExternalSector(**kw),
            CobbDouglasHousehold(**kw),
            TaxGovernment(**kw),
            StdcgeInvestment(**kw),
            GoodsMarket(**kw),
            FactorMarket(numeraire=numeraire, **kw),
            Numeraire(numeraire=numeraire, **kw),
        )
        for block in _overwritten(blocks, overwrite):
            model.add_block(block)
        return model

    def period_model(self, overwrite: Sequence[Block] = ()) -> pyo.ConcreteModel:
        """The single-period Pyomo model at the calibrated base, its tax rates
        fixed (they are the instruments a shock changes)."""
        backend = PyomoBackend()
        backend.build(self.create_model(overwrite=overwrite))
        pm = backend.pyomo_model
        assert pm is not None  # PyomoBackend.build populates pyomo_model
        for rate in INSTRUMENTS:
            cast("Any", pm.find_component(rate)).fix()
        return pm

    def default_recipe(self) -> Recipe:
        return stdcge_static()


def equivalent_variation(
    calibration: Calibration, model: PeriodModel, period: str = "shock"
) -> float:
    """Hicksian equivalent variation at base prices (stdcge lines 351-360).

    ``ep = UU / prod_i alpha_H**alpha_H``; ``EV = ep(period) - ep(base)``, read
    from a model run through an ``Exercise``.
    """
    alpha = np.asarray(calibration.values["alpha_H"], dtype=float)
    scale = float(np.prod(alpha**alpha))
    return (model.value("UU", (), period) - calibration.base["UU"]) / scale
