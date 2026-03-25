"""PEP equation overrides for the sector-CO2 extension."""

from __future__ import annotations

from equilibria.templates.pep_co2_data import carbon_unit_tax
from equilibria.templates.pep_model_equations import PEPModelEquations, PEPModelVariables


class PEPCO2ModelEquations(PEPModelEquations):
    """PEP equations with one sector-specific CO2 tax wedge."""

    def government_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        residuals = super().government_residuals(vars)
        for j in self.J:
            ttip = self.params.get("ttip", {}).get(j, 0.0)
            carbon_tax = carbon_unit_tax(self.params, vars, j)
            expected_tip = (ttip * vars.PP.get(j, 0.0) + carbon_tax) * vars.XST.get(j, 0.0)
            residuals[f"EQ39_{j}"] = vars.TIP.get(j, 0.0) - expected_tip
        return residuals

    def price_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        residuals = super().price_residuals(vars)
        for j in self.J:
            ttip = self.params.get("ttip", {}).get(j, 0.0)
            carbon_tax = carbon_unit_tax(self.params, vars, j)
            expected_pt = (1.0 + ttip) * vars.PP.get(j, 0.0) + carbon_tax
            residuals[f"EQ66_{j}"] = vars.PT.get(j, 0.0) - expected_pt
        return residuals
