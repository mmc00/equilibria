"""PEP+CO2 equation overrides — extends PEP-CRI equations with a sector carbon tax.

Hierarchy:
    PEPModelEquations
        └── PEPCRIModelEquations    (L→ROW cross-border labor in EQ44)
                └── PEPCO2ModelEquations  (CO2 tax wedge in EQ39 and EQ66)

This means all CO2 scenarios automatically include CRI cross-border labor support.
The corresponding GAMS counterpart would be PEP-1-1_v2_1_cri_co2.gms.
"""

from __future__ import annotations

from equilibria.templates.pep_co2_data import carbon_unit_tax
from equilibria.templates.pep_cri_model_equations import PEPCRIModelEquations
from equilibria.templates.pep_model_equations import PEPModelVariables


class PEPCO2ModelEquations(PEPCRIModelEquations):
    """PEP-CRI equations with one sector-specific CO2 tax wedge.

    Overrides:
    - EQ39_{j}: TIP(j) = [ttip(j)*PP(j) + carbon_unit_tax(j)] * XST(j)
    - EQ66_{j}: PT(j)  = (1 + ttip(j))*PP(j) + carbon_unit_tax(j)
    """

    def government_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """EQ39 overridden to include carbon tax in production tax identity."""
        residuals = super().government_residuals(vars)
        for j in self.J:
            ttip = self.params.get("ttip", {}).get(j, 0.0)
            carbon_tax = carbon_unit_tax(self.params, vars, j)
            expected_tip = (ttip * vars.PP.get(j, 0.0) + carbon_tax) * vars.XST.get(j, 0.0)
            residuals[f"EQ39_{j}"] = vars.TIP.get(j, 0.0) - expected_tip
        return residuals

    def price_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """EQ66 overridden to include carbon tax wedge in producer price."""
        residuals = super().price_residuals(vars)
        for j in self.J:
            ttip = self.params.get("ttip", {}).get(j, 0.0)
            carbon_tax = carbon_unit_tax(self.params, vars, j)
            expected_pt = (1.0 + ttip) * vars.PP.get(j, 0.0) + carbon_tax
            residuals[f"EQ66_{j}"] = vars.PT.get(j, 0.0) - expected_pt
        return residuals
