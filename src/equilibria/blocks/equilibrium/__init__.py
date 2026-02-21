"""Equilibrium blocks for CGE models.

This module provides market equilibrium-related equation blocks:
- Market clearing conditions
- Price normalization
"""

from __future__ import annotations

import typing
from typing import Any, TYPE_CHECKING

import numpy as np
from pydantic import Field

from equilibria.blocks.base import Block, ParameterSpec, VariableSpec
from equilibria.core.calibration_phase import CalibrationPhase
from equilibria.core.symbolic_equations import (
    SymbolicEquation,
)
from equilibria.core.parameters import Parameter
from equilibria.core.sets import SetManager
from equilibria.core.variables import Variable

if TYPE_CHECKING:
    from equilibria.core.calibration_data import CalibrationData


class MarketClearing(Block):
    """Market clearing condition block.

    Ensures supply equals demand for all commodities:
    QS[i] = QD[i] for all commodities i

    Where:
    - QS[i] = total supply of commodity i (domestic + imports)
    - QD[i] = total demand for commodity i (intermediate + final)

    Attributes:
        name: Block name (default: "MarketClearing")
    """

    name: str = Field(default="MarketClearing", description="Block name")
    description: str = Field(
        default="Market clearing conditions", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I"]  # Commodities

        self.parameters = {}

        self.variables = {
            "QS": VariableSpec(
                name="QS",
                domains=("I",),
                lower=0.0,
                description="Total commodity supply",
            ),
            "QD": VariableSpec(
                name="QD",
                domains=("I",),
                lower=0.0,
                description="Total commodity demand",
            ),
            "P": VariableSpec(
                name="P",
                domains=("I",),
                lower=0.0,
                description="Commodity price",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        """Set up the market clearing block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Create variables
        qs_vals = np.ones((n_comm,))
        variables["QS"] = Variable(
            name="QS",
            value=qs_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity supply",
        )

        qd_vals = np.ones((n_comm,))
        variables["QD"] = Variable(
            name="QD",
            value=qd_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity demand",
        )

        p_vals = np.ones((n_comm,))
        variables["P"] = Variable(
            name="P",
            value=p_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity prices",
        )

        equations = []

        # Market clearing: QS[i] = QD[i]
        class MarketClearingEq(SymbolicEquation):
            name: str = "Market_Clearing"
            domains: tuple = ("I",)
            description: str = "Commodity market clearing"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for market clearing."""
                QS = getattr(pyomo_model, "QS")
                QD = getattr(pyomo_model, "QD")

                i = indices[0]
                return QS[i] == QD[i]

        equations.append(MarketClearingEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.EQUILIBRIUM]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for market clearing."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        if mode == "sam":
            # Get data from other blocks
            trade_params = data.get_block_params("Armington")
            demand_params = data.get_block_params("LES_Consumer")

            if "QA0" in trade_params:
                QS0 = trade_params["QA0"]  # Supply
            else:
                QS0 = np.ones(n_comm)

            if "QD0" in demand_params:
                QD0 = demand_params["QD0"]  # Demand
            else:
                QD0 = QS0  # Balanced in base year

            P0 = np.ones(n_comm)  # Normalized prices

        else:  # dummy mode
            QS0 = self._get_dummy_value("QS0", (n_comm,), 1.0)
            QD0 = QS0  # Balanced
            P0 = np.ones(n_comm)

        return {
            "QS0": QS0,
            "QD0": QD0,
            "P0": P0,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "QS0" in calibrated:
            if "QS" in var_manager:
                var_manager.get("QS").value = calibrated["QS0"].copy()
        if "QD0" in calibrated:
            if "QD" in var_manager:
                var_manager.get("QD").value = calibrated["QD0"].copy()
        if "P0" in calibrated:
            if "P" in var_manager:
                var_manager.get("P").value = calibrated["P0"].copy()


class PriceNormalization(Block):
    """Price normalization block.

    Sets the numeraire price to fix the price level:
    P[numeraire] = 1

    Attributes:
        name: Block name (default: "PriceNorm")
        numeraire: Name of numeraire commodity (default: first commodity)
    """

    name: str = Field(default="PriceNorm", description="Block name")
    description: str = Field(
        default="Price normalization (numeraire)", description="Block description"
    )
    numeraire: str = Field(default="", description="Numeraire commodity name")

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["I"]  # Commodities

        self.parameters = {}

        self.variables = {
            "P": VariableSpec(
                name="P",
                domains=("I",),
                lower=0.0,
                description="Commodity price",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        """Set up the price normalization block."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Set numeraire if not specified
        if not self.numeraire:
            self.numeraire = list(commodities)[0]

        # Create variables
        p_vals = np.ones((n_comm,))
        variables["P"] = Variable(
            name="P",
            value=p_vals,
            domains=("I",),
            lower=0.0,
            description="Commodity prices",
        )

        # Fix numeraire price to 1
        numeraire_idx = list(commodities).index(self.numeraire)
        variables["P"].value[numeraire_idx] = 1.0

        equations = []

        # Price normalization: P[numeraire] = 1
        # Capture numeraire in closure to avoid scope issues
        numeraire_value = self.numeraire

        class PriceNormEq(SymbolicEquation):
            name: str = "Price_Normalization"
            domains: tuple = ()  # Scalar
            description: str = "Numeraire price normalization"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for price normalization."""
                P = getattr(pyomo_model, "P")

                # Fix the numeraire price to 1
                return P[numeraire_value] == 1.0

        equations.append(PriceNormEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.EQUILIBRIUM]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for price normalization."""
        commodities = set_manager.get("I")
        n_comm = len(commodities)

        # Get numeraire
        if not self.numeraire:
            self.numeraire = list(commodities)[0]

        # Prices are normalized to 1
        P0 = np.ones(n_comm)

        return {
            "P0": P0,
            "numeraire": self.numeraire,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "P0" in calibrated:
            if "P" in var_manager:
                var_manager.get("P").value = calibrated["P0"].copy()


class FactorMarketClearing(Block):
    """Factor market clearing block.

    Ensures factor supply equals factor demand:
    FSUP[f] = FD[f] for all factors f

    Where:
    - FSUP[f] = supply of factor f
    - FD[f] = demand for factor f

    Attributes:
        name: Block name (default: "FactorMarket")
    """

    name: str = Field(default="FactorMarket", description="Block name")
    description: str = Field(
        default="Factor market clearing", description="Block description"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize block specifications."""
        self.required_sets = ["F"]  # Factors

        self.parameters = {}

        self.variables = {
            "FSUP": VariableSpec(
                name="FSUP",
                domains=("F",),
                lower=0.0,
                description="Factor supply",
            ),
            "FD": VariableSpec(
                name="FD",
                domains=("F",),
                lower=0.0,
                description="Factor demand",
            ),
            "WF": VariableSpec(
                name="WF",
                domains=("F",),
                lower=0.0,
                description="Factor price",
            ),
        }

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        """Set up the factor market clearing block."""
        factors = set_manager.get("F")
        n_factors = len(factors)

        # Create variables
        fsup_vals = np.ones((n_factors,))
        variables["FSUP"] = Variable(
            name="FSUP",
            value=fsup_vals,
            domains=("F",),
            lower=0.0,
            description="Factor supply",
        )

        # Note: FD (factor demand) is defined by CESValueAdded block with 2D indexing (F x J)
        # We don't redefine it here to avoid dimension mismatch

        wf_vals = np.ones((n_factors,))
        variables["WF"] = Variable(
            name="WF",
            value=wf_vals,
            domains=("F",),
            lower=0.0,
            description="Factor prices",
        )

        equations = []

        # Factor market clearing: FSUP[f] = sum_j FD[f,j]
        class FactorMarketClearingEq(SymbolicEquation):
            name: str = "Factor_Market_Clearing"
            domains: tuple = ("F",)
            description: str = "Factor market clearing condition"

            def build_expression(self, pyomo_model, indices):
                """Build Pyomo expression for factor market clearing."""
                FSUP = getattr(pyomo_model, "FSUP")
                FD = getattr(pyomo_model, "FD")

                f = indices[0]
                # Sum factor demand over all sectors
                J_set = pyomo_model.J
                total_demand = sum(FD[f, j] for j in J_set)

                return FSUP[f] == total_demand

        equations.append(FactorMarketClearingEq())

        return equations

    def get_calibration_phases(self):
        """Return calibration phases for this block."""
        return [CalibrationPhase.EQUILIBRIUM]

    def _extract_calibration(self, phase, data, mode, set_manager):
        """Extract calibration data for factor market clearing."""
        factors = set_manager.get("F")
        n_factors = len(factors)

        if mode == "sam":
            # Get data from household block
            hh_params = data.get_block_params("Household")

            if "FSUP0" in hh_params:
                FSUP0 = hh_params["FSUP0"]
            else:
                FSUP0 = np.ones(n_factors)

            WF0 = np.ones(n_factors)

        else:  # dummy mode
            FSUP0 = self._get_dummy_value("FSUP0", (n_factors,), 1.0)
            WF0 = np.ones(n_factors)

        # Note: FD0 is NOT created here - it comes from CESValueAdded block
        # FactorMarketClearing only uses FD, it doesn't define the base year values
        return {
            "FSUP0": FSUP0,
            "WF0": WF0,
        }

    def _initialize_variables(self, calibrated, set_manager, var_manager):
        """Initialize variables from calibrated parameters."""
        if "FSUP0" in calibrated:
            if "FSUP" in var_manager:
                var_manager.get("FSUP").value = calibrated["FSUP0"].copy()
        if "FD0" in calibrated:
            if "FD" in var_manager:
                var_manager.get("FD").value = calibrated["FD0"].copy()
        if "WF0" in calibrated:
            if "WF" in var_manager:
                var_manager.get("WF").value = calibrated["WF0"].copy()


class PEPMacroClosureInit(Block):
    """PEP macro closure blockwise initializer/validator.

    Reconciles:
    - EQ44 (YROW),
    - EQ45 / EQ46 (SROW, CAB),
    - EQ87 (IT = savings closure),
    - EQ93 (GDP_FD identity).
    """

    name: str = Field(default="PEP_MacroClosure_Init", description="Block name")
    description: str = Field(
        default="PEP blockwise macro closure initialization and validation",
        description="Block description",
    )

    def model_post_init(self, __context: Any) -> None:
        self.required_sets = ["I"]

    def setup(
        self,
        set_manager: SetManager,
        parameters: dict[str, Parameter],
        variables: dict[str, Variable],
    ) -> list[SymbolicEquation]:
        _ = (set_manager, parameters, variables)
        return []

    @staticmethod
    def _first_map(source: dict[str, Any], *names: str) -> dict[Any, float]:
        for name in names:
            obj = source.get(name)
            if isinstance(obj, dict):
                return obj
        return {}

    @staticmethod
    def _scalar(source: dict[str, Any], name: str, default: float = 0.0) -> float:
        obj = source.get(name, default)
        try:
            return float(obj)
        except Exception:
            return float(default)

    @staticmethod
    def _set_or_blend(current: float, new: float, alpha: float) -> float:
        return (1.0 - alpha) * float(current) + alpha * float(new)

    def initialize_levels(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
        mode: str = "gams_blockwise",
    ) -> None:
        _ = mode
        I = tuple(set_manager.get("I"))
        K = tuple(set_manager.get("K")) if "K" in set_manager else tuple()
        J = tuple(set_manager.get("J")) if "J" in set_manager else tuple()
        AGD = tuple(set_manager.get("AGD")) if "AGD" in set_manager else tuple()

        alpha = max(0.0, min(1.0, self._scalar(parameters, "macro_alpha", 1.0)))

        imo0 = self._first_map(parameters, "IMO0")
        exdo0 = self._first_map(parameters, "EXDO0", "EXDO")
        kdo0 = self._first_map(parameters, "KDO0")
        lambda_rk = self._first_map(parameters, "lambda_RK")

        pwm = self._first_map(variables, "PWM")
        im = self._first_map(variables, "IM")
        r = self._first_map(variables, "R")
        kd = self._first_map(variables, "KD")
        tr = self._first_map(variables, "TR")
        pe_fob = self._first_map(variables, "PE_FOB")
        exd = self._first_map(variables, "EXD")
        sh = self._first_map(variables, "SH")
        sf = self._first_map(variables, "SF")
        pc = self._first_map(variables, "PC")
        c = self._first_map(variables, "C")
        cg = self._first_map(variables, "CG")
        inv = self._first_map(variables, "INV")
        vstk = self._first_map(variables, "VSTK")
        h_set = tuple(set_manager.get("H")) if "H" in set_manager else tuple()

        e = self._scalar(variables, "e", 1.0)
        yrow_cur = self._scalar(variables, "YROW", 0.0)
        srow_cur = self._scalar(variables, "SROW", 0.0)
        it_cur = self._scalar(variables, "IT", 0.0)
        gfcf_cur = self._scalar(variables, "GFCF", 0.0)
        gdp_fd_cur = self._scalar(variables, "GDP_FD", 0.0)
        sg = self._scalar(variables, "SG", 0.0)

        yrow_new = 0.0
        for i in I:
            if abs(imo0.get(i, 0.0)) > 1e-12:
                yrow_new += e * float(pwm.get(i, 1.0)) * float(im.get(i, 0.0))
        for k in K:
            lam = float(lambda_rk.get(("row", k), 0.0))
            for j in J:
                if abs(kdo0.get((k, j), 0.0)) > 1e-12:
                    yrow_new += lam * float(r.get((k, j), 1.0)) * float(kd.get((k, j), 0.0))
        for agd in AGD:
            yrow_new += float(tr.get(("row", agd), 0.0))

        yrow = self._set_or_blend(yrow_cur, yrow_new, alpha)
        variables["YROW"] = yrow

        srow_new = yrow
        for i in I:
            if abs(exdo0.get(i, 0.0)) > 1e-12:
                srow_new -= float(pe_fob.get(i, 0.0)) * float(exd.get(i, 0.0))
        for agd in AGD:
            srow_new -= float(tr.get((agd, "row"), 0.0))

        srow = self._set_or_blend(srow_cur, srow_new, alpha)
        variables["SROW"] = srow
        variables["CAB"] = -srow

        it_new = sum(float(v) for v in sh.values()) + sum(float(v) for v in sf.values()) + float(sg) + srow
        it = self._set_or_blend(it_cur, it_new, alpha)
        variables["IT"] = it

        stock_val = sum(float(pc.get(i, 1.0)) * float(vstk.get(i, 0.0)) for i in I)
        gfcf_new = it - stock_val
        variables["GFCF"] = self._set_or_blend(gfcf_cur, gfcf_new, alpha)

        gdp_fd_new = 0.0
        for i in I:
            cons_i = sum(float(c.get((i, h), 0.0)) for h in h_set)
            gdp_fd_new += float(pc.get(i, 0.0)) * (
                cons_i + float(cg.get(i, 0.0)) + float(inv.get(i, 0.0)) + float(vstk.get(i, 0.0))
            )
            gdp_fd_new += float(pe_fob.get(i, 0.0)) * float(exd.get(i, 0.0))
            gdp_fd_new -= float(pwm.get(i, 0.0)) * e * float(im.get(i, 0.0))
        variables["GDP_FD"] = self._set_or_blend(gdp_fd_cur, gdp_fd_new, alpha)

    def validate_initialization(
        self,
        *,
        set_manager: SetManager,
        parameters: dict[str, Any],
        variables: dict[str, Any],
    ) -> dict[str, float]:
        I = tuple(set_manager.get("I"))
        K = tuple(set_manager.get("K")) if "K" in set_manager else tuple()
        J = tuple(set_manager.get("J")) if "J" in set_manager else tuple()
        AGD = tuple(set_manager.get("AGD")) if "AGD" in set_manager else tuple()
        h_set = tuple(set_manager.get("H")) if "H" in set_manager else tuple()

        imo0 = self._first_map(parameters, "IMO0")
        exdo0 = self._first_map(parameters, "EXDO0", "EXDO")
        kdo0 = self._first_map(parameters, "KDO0")
        lambda_rk = self._first_map(parameters, "lambda_RK")

        pwm = self._first_map(variables, "PWM")
        im = self._first_map(variables, "IM")
        r = self._first_map(variables, "R")
        kd = self._first_map(variables, "KD")
        tr = self._first_map(variables, "TR")
        pe_fob = self._first_map(variables, "PE_FOB")
        exd = self._first_map(variables, "EXD")
        sh = self._first_map(variables, "SH")
        sf = self._first_map(variables, "SF")
        pc = self._first_map(variables, "PC")
        c = self._first_map(variables, "C")
        cg = self._first_map(variables, "CG")
        inv = self._first_map(variables, "INV")
        vstk = self._first_map(variables, "VSTK")

        e = self._scalar(variables, "e", 1.0)
        yrow = self._scalar(variables, "YROW", 0.0)
        srow = self._scalar(variables, "SROW", 0.0)
        cab = self._scalar(variables, "CAB", 0.0)
        it = self._scalar(variables, "IT", 0.0)
        sg = self._scalar(variables, "SG", 0.0)
        gdp_fd = self._scalar(variables, "GDP_FD", 0.0)

        yrow_rhs = 0.0
        for i in I:
            if abs(imo0.get(i, 0.0)) > 1e-12:
                yrow_rhs += e * float(pwm.get(i, 1.0)) * float(im.get(i, 0.0))
        for k in K:
            lam = float(lambda_rk.get(("row", k), 0.0))
            for j in J:
                if abs(kdo0.get((k, j), 0.0)) > 1e-12:
                    yrow_rhs += lam * float(r.get((k, j), 1.0)) * float(kd.get((k, j), 0.0))
        for agd in AGD:
            yrow_rhs += float(tr.get(("row", agd), 0.0))

        srow_rhs = yrow
        for i in I:
            if abs(exdo0.get(i, 0.0)) > 1e-12:
                srow_rhs -= float(pe_fob.get(i, 0.0)) * float(exd.get(i, 0.0))
        for agd in AGD:
            srow_rhs -= float(tr.get((agd, "row"), 0.0))

        it_rhs = sum(float(v) for v in sh.values()) + sum(float(v) for v in sf.values()) + float(sg) + srow

        gdp_fd_rhs = 0.0
        for i in I:
            cons_i = sum(float(c.get((i, h), 0.0)) for h in h_set)
            gdp_fd_rhs += float(pc.get(i, 0.0)) * (
                cons_i + float(cg.get(i, 0.0)) + float(inv.get(i, 0.0)) + float(vstk.get(i, 0.0))
            )
            gdp_fd_rhs += float(pe_fob.get(i, 0.0)) * float(exd.get(i, 0.0))
            gdp_fd_rhs -= float(pwm.get(i, 0.0)) * e * float(im.get(i, 0.0))

        return {
            "EQ44": yrow - yrow_rhs,
            "EQ45": srow - srow_rhs,
            "EQ46": srow - (-cab),
            "EQ87": it - it_rhs,
            "EQ93": gdp_fd - gdp_fd_rhs,
        }
