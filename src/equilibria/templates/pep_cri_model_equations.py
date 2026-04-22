"""PEP-CRI model equations — extends PEP standard with cross-border labor (L→ROW).

This template is for SAMs derived from ICIO/IEEM databases (e.g. Costa Rica)
that include compensation paid to non-resident employees (cross-border workers).

Hierarchy:
    PEPModelEquations  (standard GAMS PEP-1-1, resident workers only)
        └── PEPCRIModelEquations  (adds L→ROW to EQ44)
                └── PEPCO2ModelEquations  (adds sector CO2 tax wedge)
"""

from __future__ import annotations

from typing import Any

from equilibria.templates.pep_model_equations import PEPModelEquations, PEPModelVariables


class PEPCRIModelEquations(PEPModelEquations):
    """PEP equations for CRI/ICIO SAMs with cross-border labor compensation included.

    Extends EQ44 (YROW definition) to include wages paid to non-resident workers:
        YROW += SUM{l, lambda_WL('row',l) * W(l) * SUM[j, LD(l,j)]}

    The corresponding GAMS counterpart is PEP-1-1_v2_1_cri.gms.
    """

    def row_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """ROW residuals with cross-border labor compensation (EQ44 extended)."""
        residuals = super().row_residuals(vars)

        # Re-compute EQ44 adding the L→ROW labor term.
        # We re-do the full computation so we get one clean residual value.
        kdo0 = self.params.get("KDO0", {})
        ldo0 = self.params.get("LDO0", self.params.get("LDO", {}))
        imo0 = self.params.get("IMO0", {})

        yrow = 0.0
        for i in self.I:
            if abs(imo0.get(i, 0.0)) <= 1e-12:
                continue
            yrow += vars.e * vars.PWM.get(i, 1.0) * vars.IM.get(i, 0)

        for k in self.K:
            lambda_rk = self.params.get("lambda_RK", {}).get(("row", k), 0)
            for j in self.J:
                if abs(kdo0.get((k, j), 0.0)) <= 1e-12:
                    continue
                yrow += lambda_rk * vars.R.get((k, j), 1.0) * vars.KD.get((k, j), 0)

        # CRI extension: cross-border labor compensation (EQ44 extra term)
        # Corresponds to GAMS: +SUM{l, lambda_WL('row',l)*W(l)*SUM[j$LDO(l,j), LD(l,j)]}
        for l in self.L:
            lambda_wl = self.params.get("lambda_WL", {}).get(("row", l), 0.0)
            if abs(lambda_wl) <= 1e-16:
                continue
            w_l = vars.W.get(l, 1.0)
            for j in self.J:
                if abs(ldo0.get((l, j), 0.0)) <= 1e-12:
                    continue
                yrow += lambda_wl * w_l * vars.LD.get((l, j), 0)

        for agd in self.AGD:
            yrow += vars.TR.get(("row", agd), 0)

        residuals["EQ44"] = vars.YROW - yrow
        return residuals
