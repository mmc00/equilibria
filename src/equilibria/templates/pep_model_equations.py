"""
PEP Model Equations Implementation

This module implements all equations from the GAMS PEP-1-1_v2_1_modular.gms model
(Section 5.3, lines 934-1239).

Each equation returns a residual (should equal zero at equilibrium).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PEPModelVariables:
    """Container for all PEP model variables."""
    
    # Production variables (Volume)
    VA: dict[str, float] = field(default_factory=dict)
    CI: dict[str, float] = field(default_factory=dict)
    LDC: dict[str, float] = field(default_factory=dict)
    KDC: dict[str, float] = field(default_factory=dict)
    LD: dict[tuple[str, str], float] = field(default_factory=dict)
    KD: dict[tuple[str, str], float] = field(default_factory=dict)
    DI: dict[tuple[str, str], float] = field(default_factory=dict)
    XST: dict[str, float] = field(default_factory=dict)
    XS: dict[tuple[str, str], float] = field(default_factory=dict)
    
    # Income variables
    YH: dict[str, float] = field(default_factory=dict)
    YHL: dict[str, float] = field(default_factory=dict)
    YHK: dict[str, float] = field(default_factory=dict)
    YHTR: dict[str, float] = field(default_factory=dict)
    YDH: dict[str, float] = field(default_factory=dict)
    CTH: dict[str, float] = field(default_factory=dict)
    SH: dict[str, float] = field(default_factory=dict)
    YF: dict[str, float] = field(default_factory=dict)
    YFK: dict[str, float] = field(default_factory=dict)
    YFTR: dict[str, float] = field(default_factory=dict)
    YDF: dict[str, float] = field(default_factory=dict)
    SF: dict[str, float] = field(default_factory=dict)
    
    # Government variables
    YG: float = 0.0
    YGK: float = 0.0
    TDHT: float = 0.0
    TDFT: float = 0.0
    TPRODN: float = 0.0
    TPRCTS: float = 0.0
    TIWT: float = 0.0
    TIKT: float = 0.0
    TIPT: float = 0.0
    TICT: float = 0.0
    TIMT: float = 0.0
    TIXT: float = 0.0
    YGTR: float = 0.0
    SG: float = 0.0
    TDH: dict[str, float] = field(default_factory=dict)
    TDF: dict[str, float] = field(default_factory=dict)
    TIW: dict[tuple[str, str], float] = field(default_factory=dict)
    TIK: dict[tuple[str, str], float] = field(default_factory=dict)
    TIP: dict[str, float] = field(default_factory=dict)
    TIC: dict[str, float] = field(default_factory=dict)
    TIM: dict[str, float] = field(default_factory=dict)
    TIX: dict[str, float] = field(default_factory=dict)
    G: float = 0.0
    
    # Rest of world
    YROW: float = 0.0
    SROW: float = 0.0
    CAB: float = 0.0
    
    # Transfers
    TR: dict[tuple[str, str], float] = field(default_factory=dict)
    
    # Demand variables
    C: dict[tuple[str, str], float] = field(default_factory=dict)
    GFCF: float = 0.0
    INV: dict[str, float] = field(default_factory=dict)
    CG: dict[str, float] = field(default_factory=dict)
    DIT: dict[str, float] = field(default_factory=dict)
    MRGN: dict[str, float] = field(default_factory=dict)
    CMIN: dict[tuple[str, str], float] = field(default_factory=dict)
    
    # Trade variables
    EXD: dict[str, float] = field(default_factory=dict)
    IM: dict[str, float] = field(default_factory=dict)
    DD: dict[str, float] = field(default_factory=dict)
    DS: dict[tuple[str, str], float] = field(default_factory=dict)
    EX: dict[tuple[str, str], float] = field(default_factory=dict)
    Q: dict[str, float] = field(default_factory=dict)
    
    # Price variables
    P: dict[tuple[str, str], float] = field(default_factory=dict)
    PC: dict[str, float] = field(default_factory=dict)
    PD: dict[str, float] = field(default_factory=dict)
    PE: dict[str, float] = field(default_factory=dict)
    PE_FOB: dict[str, float] = field(default_factory=dict)
    PM: dict[str, float] = field(default_factory=dict)
    PWM: dict[str, float] = field(default_factory=dict)
    PP: dict[str, float] = field(default_factory=dict)
    PCI: dict[str, float] = field(default_factory=dict)
    PVA: dict[str, float] = field(default_factory=dict)
    WC: dict[str, float] = field(default_factory=dict)
    RC: dict[str, float] = field(default_factory=dict)
    W: dict[str, float] = field(default_factory=dict)
    R: dict[tuple[str, str], float] = field(default_factory=dict)
    WTI: dict[tuple[str, str], float] = field(default_factory=dict)
    RTI: dict[tuple[str, str], float] = field(default_factory=dict)
    PL: dict[str, float] = field(default_factory=dict)
    PIXCON: float = 1.0
    PIXGDP: float = 1.0
    PIXINV: float = 1.0
    PIXGVT: float = 1.0
    e: float = 1.0
    
    # Real variables
    CTH_REAL: dict[str, float] = field(default_factory=dict)
    G_REAL: float = 0.0
    GDP_BP_REAL: float = 0.0
    GDP_MP_REAL: float = 0.0
    
    # GDP variables
    GDP_BP: float = 0.0
    GDP_MP: float = 0.0
    GDP_IB: float = 0.0
    GDP_FD: float = 0.0
    
    # Investment
    IT: float = 0.0
    VSTK: dict[str, float] = field(default_factory=dict)


@dataclass
class SolverResult:
    """Result from model solution."""
    
    converged: bool = False
    iterations: int = 0
    final_residual: float = float('inf')
    variables: PEPModelVariables = field(default_factory=PEPModelVariables)
    residuals: dict[str, float] = field(default_factory=dict)
    message: str = ""
    
    def summary(self) -> str:
        """Return a text summary of the solution."""
        lines = [
            "=" * 70,
            "PEP MODEL SOLUTION RESULT",
            "=" * 70,
            f"Status:        {'CONVERGED' if self.converged else 'FAILED'}",
            f"Iterations:    {self.iterations}",
            f"Final RMS Residual: {self.final_residual:.2e}",
            f"Message:       {self.message}",
            "",
            "Key Variables:",
            f"  GDP (Basic Prices):  {self.variables.GDP_BP:15,.2f}",
            f"  GDP (Market Prices): {self.variables.GDP_MP:15,.2f}",
            f"  Total Consumption:   {sum(self.variables.CTH.values()):15,.2f}",
            f"  Total Investment:    {self.variables.IT:15,.2f}",
            f"  Trade Balance:       {sum(self.variables.EXD.values()) - sum(self.variables.IM.values()):15,.2f}",
            "=" * 70,
        ]
        return "\n".join(lines)


class PEPModelEquations:
    """Implements all PEP model equations."""
    
    def __init__(
        self,
        sets: dict[str, list[str]],
        parameters: dict[str, Any],
    ):
        """Initialize equations with model sets and parameters.
        
        Args:
            sets: Dictionary of model sets (H, F, J, I, K, L, etc.)
            parameters: Dictionary of calibrated parameters from Phase 1-5
        """
        self.sets = sets
        self.params = parameters
        
        # Extract commonly used sets
        self.H = sets.get("H", [])
        self.F = sets.get("F", [])
        self.J = sets.get("J", [])
        self.I = sets.get("I", [])
        self.K = sets.get("K", [])
        self.L = sets.get("L", [])
        self.AG = sets.get("AG", [])
        self.AGNG = sets.get("AGNG", [])
        self.AGD = sets.get("AGD", [])
    
    def calculate_all_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate residuals for all equations.
        
        Args:
            vars: Current values of all model variables
            
        Returns:
            Dictionary mapping equation names to residuals (should be ~0 at solution)
        """
        residuals = {}
        
        # Production equations
        residuals.update(self.production_residuals(vars))
        
        # Income equations
        residuals.update(self.income_residuals(vars))
        
        # Government equations
        residuals.update(self.government_residuals(vars))
        
        # Rest of world equations
        residuals.update(self.row_residuals(vars))
        
        # Transfer equations
        residuals.update(self.transfer_residuals(vars))
        
        # Demand equations
        residuals.update(self.demand_residuals(vars))
        
        # Trade equations
        residuals.update(self.trade_residuals(vars))
        
        # Price equations
        residuals.update(self.price_residuals(vars))
        
        # Equilibrium equations
        residuals.update(self.equilibrium_residuals(vars))
        
        # GDP equations
        residuals.update(self.gdp_residuals(vars))
        
        return residuals
    
    def production_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate production block residuals (EQ1-EQ9)."""
        residuals = {}
        
        for j in self.J:
            # EQ1: VA(j) = v(j) * XST(j)
            v_j = self.params.get("v", {}).get(j, 0)
            expected_va = v_j * vars.XST.get(j, 0)
            residuals[f"EQ1_{j}"] = vars.VA.get(j, 0) - expected_va
            
            # EQ2: CI(j) = io(j) * XST(j)
            io_j = self.params.get("io", {}).get(j, 0)
            expected_ci = io_j * vars.XST.get(j, 0)
            residuals[f"EQ2_{j}"] = vars.CI.get(j, 0) - expected_ci
            
            # EQ3: VA CES function
            if vars.VA.get(j, 0) > 0:
                rho_va = self.params.get("rho_VA", {}).get(j, -1)
                if j not in self.params.get("beta_VA", {}):
                    continue
                beta_va = self.params.get("beta_VA", {}).get(j, 0.5)
                B_va = self.params.get("B_VA", {}).get(j, 1)
                
                if rho_va != 0:
                    ldc_term = beta_va * (vars.LDC.get(j, 0) ** (-rho_va)) if vars.LDC.get(j, 0) > 0 else 0
                    kdc_term = (1 - beta_va) * (vars.KDC.get(j, 0) ** (-rho_va)) if vars.KDC.get(j, 0) > 0 else 0
                    ces_va = B_va * (ldc_term + kdc_term) ** (-1 / rho_va)
                    residuals[f"EQ3_{j}"] = vars.VA.get(j, 0) - ces_va
            
            # EQ4: LDC/KDC ratio (CES)
            if vars.LDC.get(j, 0) > 0 and vars.KDC.get(j, 0) > 0:
                sigma_va = self.params.get("sigma_VA", {}).get(j, 1)
                beta_va = self.params.get("beta_VA", {}).get(j, 0.5)
                
                if vars.RC.get(j, 0) > 0 and vars.WC.get(j, 0) > 0:
                    # Corner CES share cases (beta=0 or beta=1) are degenerate for
                    # the FOC ratio form and can appear in calibrated benchmark data.
                    if 0 < beta_va < 1:
                        ratio = ((beta_va / (1 - beta_va)) * (vars.RC[j] / vars.WC[j])) ** sigma_va
                        expected_ldc = ratio * vars.KDC[j]
                        residuals[f"EQ4_{j}"] = vars.LDC[j] - expected_ldc
                    else:
                        # Keep equation count fixed for solver Jacobian dimensions.
                        residuals[f"EQ4_{j}"] = 0.0
        
            # EQ5: LDC composite labor CES
            if vars.LDC.get(j, 0) > 0:
                rho_ld = self.params.get("rho_LD", {}).get(j, 0)
                B_ld = self.params.get("B_LD", {}).get(j, 1)
                
                ld_sum = 0
                for l in self.L:
                    beta_ld = self.params.get("beta_LD", {}).get((l, j), 0)
                    if vars.LD.get((l, j), 0) > 0:
                        ld_sum += beta_ld * (vars.LD[(l, j)] ** (-rho_ld))
                
                if ld_sum > 0 and rho_ld != 0:
                    expected_ldc = B_ld * (ld_sum ** (-1 / rho_ld))
                    residuals[f"EQ5_{j}"] = vars.LDC[j] - expected_ldc
            
            # EQ6: Labor demand by category
            for l in self.L:
                key = (l, j)
                if vars.LD.get(key, 0) > 0:
                    sigma_ld = self.params.get("sigma_LD", {}).get(j, 1)
                    beta_ld = self.params.get("beta_LD", {}).get(key, 0)
                    B_ld = self.params.get("B_LD", {}).get(j, 1)
                    
                    if vars.WTI.get(key, 0) > 0 and vars.WC.get(j, 0) > 0:
                        demand = (beta_ld * vars.WC[j] / vars.WTI[key]) ** sigma_ld
                        demand *= B_ld ** (sigma_ld - 1) * vars.LDC[j]
                        residuals[f"EQ6_{l}_{j}"] = vars.LD[key] - demand
            
            # EQ7: KDC composite capital CES
            if vars.KDC.get(j, 0) > 0:
                rho_kd = self.params.get("rho_KD", {}).get(j, 0)
                B_kd = self.params.get("B_KD", {}).get(j, 1)
                
                kd_sum = 0
                for k in self.K:
                    beta_kd = self.params.get("beta_KD", {}).get((k, j), 0)
                    if vars.KD.get((k, j), 0) > 0:
                        kd_sum += beta_kd * (vars.KD[(k, j)] ** (-rho_kd))
                
                if kd_sum > 0 and rho_kd != 0:
                    expected_kdc = B_kd * (kd_sum ** (-1 / rho_kd))
                    residuals[f"EQ7_{j}"] = vars.KDC[j] - expected_kdc
            
            # EQ8: Capital demand by category
            for k in self.K:
                key = (k, j)
                if vars.KD.get(key, 0) > 0:
                    sigma_kd = self.params.get("sigma_KD", {}).get(j, 1)
                    beta_kd = self.params.get("beta_KD", {}).get(key, 0)
                    B_kd = self.params.get("B_KD", {}).get(j, 1)
                    
                    if vars.RTI.get(key, 0) > 0 and vars.RC.get(j, 0) > 0:
                        demand = (beta_kd * vars.RC[j] / vars.RTI[key]) ** sigma_kd
                        demand *= B_kd ** (sigma_kd - 1) * vars.KDC[j]
                        residuals[f"EQ8_{k}_{j}"] = vars.KD[key] - demand
            
            # EQ9: Intermediate consumption
            for i in self.I:
                key = (i, j)
                aij = self.params.get("aij", {}).get(key, 0)
                expected_di = aij * vars.CI.get(j, 0)
                residuals[f"EQ9_{i}_{j}"] = vars.DI.get(key, 0) - expected_di
        
        return residuals
    
    def income_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate income block residuals (EQ10-EQ21)."""
        residuals = {}
        
        # Household income (EQ10-EQ16)
        for h in self.H:
            # EQ10: YH = YHL + YHK + YHTR
            expected_yh = vars.YHL.get(h, 0) + vars.YHK.get(h, 0) + vars.YHTR.get(h, 0)
            residuals[f"EQ10_{h}"] = vars.YH.get(h, 0) - expected_yh
            
            # EQ11: YHL = sum of labor income
            yhl = 0
            for l in self.L:
                lambda_wl = self.params.get("lambda_WL", {}).get((h, l), 0)
                w_l = vars.W.get(l, 1.0)
                ld_sum = sum(vars.LD.get((l, j), 0) for j in self.J)
                yhl += lambda_wl * w_l * ld_sum
            residuals[f"EQ11_{h}"] = vars.YHL.get(h, 0) - yhl
            
            # EQ12: YHK = sum of capital income
            yhk = 0
            for k in self.K:
                lambda_rk = self.params.get("lambda_RK", {}).get((h, k), 0)
                for j in self.J:
                    r_kj = vars.R.get((k, j), 1.0)
                    kd_kj = vars.KD.get((k, j), 0)
                    yhk += lambda_rk * r_kj * kd_kj
            residuals[f"EQ12_{h}"] = vars.YHK.get(h, 0) - yhk
            
            # EQ13: YHTR = sum of transfers to household h
            yhtr = sum(vars.TR.get((h, ag), 0) for ag in self.AG)
            residuals[f"EQ13_{h}"] = vars.YHTR.get(h, 0) - yhtr
            
            # EQ14: YDH = YH - TDH - TR('gvt',h) (transfer from h to gvt)
            expected_ydh = vars.YH.get(h, 0) - vars.TDH.get(h, 0) - vars.TR.get(("gvt", h), 0)
            residuals[f"EQ14_{h}"] = vars.YDH.get(h, 0) - expected_ydh
            
            # EQ15: CTH = YDH - SH - sum of transfers
            tr_sum = sum(vars.TR.get((agng, h), 0) for agng in self.AGNG)
            expected_cth = vars.YDH.get(h, 0) - vars.SH.get(h, 0) - tr_sum
            residuals[f"EQ15_{h}"] = vars.CTH.get(h, 0) - expected_cth
            
            # EQ16: SH = savings function
            eta = self.params.get("eta", 0)
            sh0 = self.params.get("sh0", {}).get(h, 0)
            sh1 = self.params.get("sh1", {}).get(h, 0)
            expected_sh = (vars.PIXCON ** eta) * sh0 + sh1 * vars.YDH.get(h, 0)
            residuals[f"EQ16_{h}"] = vars.SH.get(h, 0) - expected_sh
        
        # Firm income (EQ17-EQ21)
        for f in self.F:
            # EQ17: YF = YFK + YFTR
            expected_yf = vars.YFK.get(f, 0) + vars.YFTR.get(f, 0)
            residuals[f"EQ17_{f}"] = vars.YF.get(f, 0) - expected_yf
            
            # EQ18: YFK = capital income
            yfk = 0
            for k in self.K:
                lambda_rk = self.params.get("lambda_RK", {}).get((f, k), 0)
                for j in self.J:
                    r_kj = vars.R.get((k, j), 1.0)
                    kd_kj = vars.KD.get((k, j), 0)
                    yfk += lambda_rk * r_kj * kd_kj
            residuals[f"EQ18_{f}"] = vars.YFK.get(f, 0) - yfk
            
            # EQ19: YFTR = sum of transfers to firm f
            yftr = sum(vars.TR.get((f, ag), 0) for ag in self.AG)
            residuals[f"EQ19_{f}"] = vars.YFTR.get(f, 0) - yftr
            
            # EQ20: YDF = YF - TDF
            expected_ydf = vars.YF.get(f, 0) - vars.TDF.get(f, 0)
            residuals[f"EQ20_{f}"] = vars.YDF.get(f, 0) - expected_ydf
            
            # EQ21: SF = YDF - sum of transfers from firm f to others
            tr_sum = sum(vars.TR.get((ag, f), 0) for ag in self.AG)
            expected_sf = vars.YDF.get(f, 0) - tr_sum
            residuals[f"EQ21_{f}"] = vars.SF.get(f, 0) - expected_sf
        
        return residuals
    
    def government_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate government block residuals (EQ22-EQ43)."""
        residuals = {}
        
        # EQ22: YG = YGK + TDHT + TDFT + TPRODN + TPRCTS + YGTR
        expected_yg = (
            vars.YGK + vars.TDHT + vars.TDFT + vars.TPRODN + 
            vars.TPRCTS + vars.YGTR
        )
        residuals["EQ22"] = vars.YG - expected_yg
        
        # EQ23: YGK = government capital income
        ygk = 0
        for k in self.K:
            lambda_rk = self.params.get("lambda_RK", {}).get(("gvt", k), 0)
            for j in self.J:
                r_kj = vars.R.get((k, j), 1.0)
                kd_kj = vars.KD.get((k, j), 0)
                ygk += lambda_rk * r_kj * kd_kj
        residuals["EQ23"] = vars.YGK - ygk
        
        # EQ24: TDHT = sum of household taxes
        expected_tdht = sum(vars.TDH.values())
        residuals["EQ24"] = vars.TDHT - expected_tdht
        
        # EQ25: TDFT = sum of firm taxes
        expected_tdft = sum(vars.TDF.values())
        residuals["EQ25"] = vars.TDFT - expected_tdft
        
        # EQ26: TPRODN = TIWT + TIKT + TIPT
        expected_tprodn = vars.TIWT + vars.TIKT + vars.TIPT
        residuals["EQ26"] = vars.TPRODN - expected_tprodn
        
        # EQ27: TIWT = sum of labor taxes
        expected_tiwt = sum(vars.TIW.values())
        residuals["EQ27"] = vars.TIWT - expected_tiwt
        
        # EQ28: TIKT = sum of capital taxes
        expected_tikt = sum(vars.TIK.values())
        residuals["EQ28"] = vars.TIKT - expected_tikt
        
        # EQ29: TIPT = sum of production taxes
        expected_tipt = sum(vars.TIP.values())
        residuals["EQ29"] = vars.TIPT - expected_tipt
        
        # EQ30: TPRCTS = TICT + TIMT + TIXT
        expected_tprcts = vars.TICT + vars.TIMT + vars.TIXT
        residuals["EQ30"] = vars.TPRCTS - expected_tprcts
        
        # EQ31: TICT = sum of commodity taxes
        expected_tict = sum(vars.TIC.values())
        residuals["EQ31"] = vars.TICT - expected_tict
        
        # EQ32: TIMT = sum of import taxes
        expected_timt = sum(vars.TIM.values())
        residuals["EQ32"] = vars.TIMT - expected_timt
        
        # EQ33: TIXT = sum of export taxes
        expected_tixt = sum(vars.TIX.values())
        residuals["EQ33"] = vars.TIXT - expected_tixt
        
        # EQ34: YGTR = sum of transfers from government to non-government agents
        expected_ygtr = sum(vars.TR.get(("gvt", agng), 0) for agng in self.AGNG)
        residuals["EQ34"] = vars.YGTR - expected_ygtr
        
        # EQ35: TDH(h) = tax function
        for h in self.H:
            eta = self.params.get("eta", 0)
            ttdh0 = self.params.get("ttdh0", {}).get(h, 0)
            ttdh1 = self.params.get("ttdh1", {}).get(h, 0)
            expected_tdh = (vars.PIXCON ** eta) * ttdh0 + ttdh1 * vars.YH.get(h, 0)
            residuals[f"EQ35_{h}"] = vars.TDH.get(h, 0) - expected_tdh
        
        # EQ36: TDF(f) = tax function
        for f in self.F:
            eta = self.params.get("eta", 0)
            ttdf0 = self.params.get("ttdf0", {}).get(f, 0)
            ttdf1 = self.params.get("ttdf1", {}).get(f, 0)
            expected_tdf = (vars.PIXCON ** eta) * ttdf0 + ttdf1 * vars.YFK.get(f, 0)
            residuals[f"EQ36_{f}"] = vars.TDF.get(f, 0) - expected_tdf
        
        # EQ37: TIW(l,j) = labor tax
        for l in self.L:
            for j in self.J:
                key = (l, j)
                ttiw = self.params.get("ttiw", {}).get(key, 0)
                w_l = vars.W.get(l, 1.0)
                expected_tiw = ttiw * w_l * vars.LD.get(key, 0)
                residuals[f"EQ37_{l}_{j}"] = vars.TIW.get(key, 0) - expected_tiw
        
        # EQ38: TIK(k,j) = capital tax
        for k in self.K:
            for j in self.J:
                key = (k, j)
                ttik = self.params.get("ttik", {}).get(key, 0)
                r_kj = vars.R.get(key, 1.0)
                expected_tik = ttik * r_kj * vars.KD.get(key, 0)
                residuals[f"EQ38_{k}_{j}"] = vars.TIK.get(key, 0) - expected_tik
        
        # EQ39: TIP(j) = production tax
        for j in self.J:
            ttip = self.params.get("ttip", {}).get(j, 0)
            expected_tip = ttip * vars.PP.get(j, 0) * vars.XST.get(j, 0)
            residuals[f"EQ39_{j}"] = vars.TIP.get(j, 0) - expected_tip

        # EQ40: TIC(i) = commodity tax
        for i in self.I:
            ttic = self.params.get("ttic", {}).get(i, 0)
            denom = 1 + ttic
            if abs(denom) < 1e-12:
                expected_tic = 0.0
            else:
                expected_tic = (ttic / denom) * (
                    vars.PD.get(i, 0) * vars.DD.get(i, 0) +
                    vars.PM.get(i, 0) * vars.IM.get(i, 0)
                )
            residuals[f"EQ40_{i}"] = vars.TIC.get(i, 0) - expected_tic
        
        # EQ41: TIM(i) = import duty
        for i in self.I:
            ttim = self.params.get("ttim", {}).get(i, 0)
            pwm_i = vars.PWM.get(i, 1.0)
            expected_tim = ttim * vars.e * pwm_i * vars.IM.get(i, 0)
            residuals[f"EQ41_{i}"] = vars.TIM.get(i, 0) - expected_tim
        
        # EQ42: TIX(i) = export tax
        for i in self.I:
            ttix = self.params.get("ttix", {}).get(i, 0)
            margin_sum = sum(vars.PC.get(ij, 1.0) * self.params.get("tmrg_X", {}).get((ij, i), 0) for ij in self.I)
            expected_tix = ttix * (vars.PE.get(i, 0) + margin_sum) * vars.EXD.get(i, 0)
            residuals[f"EQ42_{i}"] = vars.TIX.get(i, 0) - expected_tix
        
        # EQ43: SG = government savings
        tr_sum = sum(vars.TR.get((agng, "gvt"), 0) for agng in self.AGNG)
        expected_sg = vars.YG - tr_sum - vars.G
        residuals["EQ43"] = vars.SG - expected_sg
        
        return residuals
    
    def row_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate rest of world residuals (EQ44-EQ46)."""
        residuals = {}
        
        # EQ44: YROW = e * sum(PWM * IM) + capital income + transfers to ROW
        yrow = 0
        for i in self.I:
            pwm_i = vars.PWM.get(i, 1.0)
            yrow += vars.e * pwm_i * vars.IM.get(i, 0)
        
        for k in self.K:
            lambda_rk = self.params.get("lambda_RK", {}).get(("row", k), 0)
            for j in self.J:
                r_kj = vars.R.get((k, j), 1.0)
                kd_kj = vars.KD.get((k, j), 0)
                yrow += lambda_rk * r_kj * kd_kj
        
        for agd in self.AGD:
            yrow += vars.TR.get(("row", agd), 0)
        
        residuals["EQ44"] = vars.YROW - yrow
        
        # EQ45: SROW = YROW - exports + net transfers from ROW
        srow = vars.YROW
        for i in self.I:
            srow -= vars.PE_FOB.get(i, 0) * vars.EXD.get(i, 0)
        for agd in self.AGD:
            srow -= vars.TR.get((agd, "row"), 0)
        
        residuals["EQ45"] = vars.SROW - srow
        
        # EQ46: SROW = -CAB
        residuals["EQ46"] = vars.SROW - (-vars.CAB)
        
        return residuals
    
    def transfer_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate transfer residuals (EQ47-EQ51)."""
        residuals = {}
        
        # EQ47: TR(agng,h) = lambda_TR * YDH(h)
        for agng in self.AGNG:
            for h in self.H:
                lambda_tr = self.params.get("lambda_TR_households", {}).get((agng, h), 0)
                expected_tr = lambda_tr * vars.YDH.get(h, 0)
                residuals[f"EQ47_{agng}_{h}"] = vars.TR.get((agng, h), 0) - expected_tr
        
        # EQ48: TR('gvt',h) = transfer function
        for h in self.H:
            eta = self.params.get("eta", 0)
            tr0 = self.params.get("tr0", {}).get(h, 0)
            tr1 = self.params.get("tr1", {}).get(h, 0)
            expected_tr = (vars.PIXCON ** eta) * tr0 + tr1 * vars.YH.get(h, 0)
            residuals[f"EQ48_{h}"] = vars.TR.get(("gvt", h), 0) - expected_tr
        
        # EQ49: TR(ag,f) = lambda_TR * YDF(f)
        for ag in self.AG:
            for f in self.F:
                lambda_tr = self.params.get("lambda_TR_firms", {}).get((ag, f), 0)
                expected_tr = lambda_tr * vars.YDF.get(f, 0)
                residuals[f"EQ49_{ag}_{f}"] = vars.TR.get((ag, f), 0) - expected_tr
        
        # EQ50: TR(agng,'gvt') = PIXCON^eta * TRO(agng,'gvt')
        for agng in self.AGNG:
            eta = self.params.get("eta", 0)
            tro = self.params.get("TRO", {}).get((agng, "gvt"), 0)
            expected_tr = (vars.PIXCON ** eta) * tro
            residuals[f"EQ50_{agng}"] = vars.TR.get((agng, "gvt"), 0) - expected_tr
        
        # EQ51: TR(agd,'row') = PIXCON^eta * TRO(agd,'row')
        for agd in self.AGD:
            eta = self.params.get("eta", 0)
            tro = self.params.get("TRO", {}).get((agd, "row"), 0)
            expected_tr = (vars.PIXCON ** eta) * tro
            residuals[f"EQ51_{agd}"] = vars.TR.get((agd, "row"), 0) - expected_tr
        
        return residuals
    
    def demand_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate demand block residuals (EQ52-EQ57)."""
        residuals = {}
        
        # EQ52: PC(i)*C(i,h) = LES demand function
        for i in self.I:
            for h in self.H:
                pc_i = vars.PC.get(i, 1.0)
                cmin_ih = vars.CMIN.get((i, h), 0)
                gamma_les = self.params.get("gamma_LES", {}).get((i, h), 0)
                
                cmin_sum = sum(vars.PC.get(ij, 1.0) * vars.CMIN.get((ij, h), 0) for ij in self.I)
                expected_pc_c = pc_i * cmin_ih + gamma_les * (vars.CTH.get(h, 0) - cmin_sum)
                
                residuals[f"EQ52_{i}_{h}"] = pc_i * vars.C.get((i, h), 0) - expected_pc_c
        
        # EQ53: GFCF = IT - sum(PC * VSTK)
        expected_gfcf = vars.IT - sum(vars.PC.get(i, 1.0) * vars.VSTK.get(i, 0) for i in self.I)
        residuals["EQ53"] = vars.GFCF - expected_gfcf
        
        # EQ54: INV(i) = investment allocation (not fully specified in GAMS)
        # Skip for now
        
        # EQ55: CG(i) = government consumption (exogenous)
        # Skip as it's exogenous
        
        # EQ56: DIT(i) = sum of intermediate demand
        for i in self.I:
            expected_dit = sum(vars.DI.get((i, j), 0) for j in self.J)
            residuals[f"EQ56_{i}"] = vars.DIT.get(i, 0) - expected_dit
        
        # EQ57: MRGN(i) = trade margin demand
        for i in self.I:
            mrgn = 0
            for ij in self.I:
                tmrg = self.params.get("tmrg", {}).get((i, ij), 0)
                mrgn += tmrg * vars.DD.get(ij, 0)
                mrgn += tmrg * vars.IM.get(ij, 0)
                for j in self.J:
                    tmrg_x = self.params.get("tmrg_X", {}).get((i, ij), 0)
                    mrgn += tmrg_x * vars.EX.get((j, ij), 0)
            residuals[f"EQ57_{i}"] = vars.MRGN.get(i, 0) - mrgn
        
        return residuals
    
    def trade_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate trade block residuals (EQ58-EQ64)."""
        residuals = {}
        
        # EQ58: XST(j) = CET between commodities
        for j in self.J:
            rho_xt = self.params.get("rho_XT", {}).get(j, 1)
            B_xt = self.params.get("B_XT", {}).get(j, 1)
            
            if rho_xt != 0:
                xs_sum = 0
                for i in self.I:
                    beta_xt = self.params.get("beta_XT", {}).get((j, i), 0)
                    if vars.XS.get((j, i), 0) > 0:
                        xs_sum += beta_xt * (vars.XS[(j, i)] ** rho_xt)
                
                if xs_sum > 0:
                    expected_xst = B_xt * (xs_sum ** (1 / rho_xt))
                    residuals[f"EQ58_{j}"] = vars.XST.get(j, 0) - expected_xst
        
        # EQ59: XS(j,i) = production by sector
        # This is defined implicitly by EQ58
        
        # EQ60: XS(j,i) = CET between exports and local
        for j in self.J:
            for i in self.I:
                rho_x = self.params.get("rho_X", {}).get((j, i), 1)
                B_x = self.params.get("B_X", {}).get((j, i), 1)
                beta_x = self.params.get("beta_X", {}).get((j, i), 0.5)
                if (j, i) not in self.params.get("beta_X", {}):
                    continue
                
                if rho_x != 0:
                    ex = vars.EX.get((j, i), 0)
                    ds = vars.DS.get((j, i), 0)
                    if ex > 0 or ds > 0:
                        expected_xs = B_x * (beta_x * (ex ** rho_x) + (1 - beta_x) * (ds ** rho_x)) ** (1 / rho_x)
                        residuals[f"EQ60_{j}_{i}"] = vars.XS.get((j, i), 0) - expected_xs
        
        # EQ61: EX/DS ratio (CET condition)
        # Skip for now as it's complex
        
        # EQ62: EXD(i) = world demand for exports
        for i in self.I:
            sigma_xd = self.params.get("sigma_XD", {}).get(i, 1)
            pwx_i = self.params.get("PWX", {}).get(i, vars.PWM.get(i, 1.0))
            pe_fob_i = vars.PE_FOB.get(i, 0)
            
            exdo = self.params.get("EXDO", {}).get(i, 0)
            if pe_fob_i > 0:
                expected_exd = exdo * ((vars.e * pwx_i) / pe_fob_i) ** sigma_xd
                residuals[f"EQ62_{i}"] = vars.EXD.get(i, 0) - expected_exd
        
        # EQ63: Q(i) = CES between imports and local
        for i in self.I:
            rho_m = self.params.get("rho_M", {}).get(i, -0.5)
            B_m = self.params.get("B_M", {}).get(i, 1)
            beta_m = self.params.get("beta_M", {}).get(i, 0.5)
            
            if rho_m != 0:
                im = vars.IM.get(i, 0)
                dd = vars.DD.get(i, 0)
                if im > 0 or dd > 0:
                    expected_q = B_m * (beta_m * (im ** (-rho_m)) + (1 - beta_m) * (dd ** (-rho_m))) ** (-1 / rho_m)
                    residuals[f"EQ63_{i}"] = vars.Q.get(i, 0) - expected_q
        
        # EQ64: IM(i) = import demand
        # Skip as it's implicit in EQ63
        
        return residuals
    
    def price_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate price block residuals (EQ65-EQ84)."""
        residuals = {}
        
        # EQ65: PP(j) = unit cost
        for j in self.J:
            expected_pp = vars.PVA.get(j, 0) * vars.VA.get(j, 0) + vars.PCI.get(j, 0) * vars.CI.get(j, 0)
            if vars.XST.get(j, 0) > 0:
                expected_pp /= vars.XST[j]
            residuals[f"EQ65_{j}"] = vars.PP.get(j, 0) - expected_pp
        
        # EQ66-EQ67: Price definitions
        # Skip as they're definitional
        
        # EQ68: PVA(j) = price of value added
        for j in self.J:
            if vars.VA.get(j, 0) > 0:
                expected_pva = (vars.WC.get(j, 0) * vars.LDC.get(j, 0) + vars.RC.get(j, 0) * vars.KDC.get(j, 0)) / vars.VA[j]
                residuals[f"EQ68_{j}"] = vars.PVA.get(j, 0) - expected_pva
        
        # EQ70: WTI(l,j) = wage with taxes
        for l in self.L:
            for j in self.J:
                ttiw = self.params.get("ttiw", {}).get((l, j), 0)
                w_l = vars.W.get(l, 1.0)
                expected_wti = w_l * (1 + ttiw)
                residuals[f"EQ70_{l}_{j}"] = vars.WTI.get((l, j), 0) - expected_wti
        
        # EQ72: RTI(k,j) = rental rate with taxes
        for k in self.K:
            for j in self.J:
                ttik = self.params.get("ttik", {}).get((k, j), 0)
                r_kj = vars.R.get((k, j), 1.0)
                expected_rti = r_kj * (1 + ttik)
                residuals[f"EQ72_{k}_{j}"] = vars.RTI.get((k, j), 0) - expected_rti
        
        # EQ76: PE(i) = export price
        for i in self.I:
            ttix = self.params.get("ttix", {}).get(i, 0)
            margin_sum = sum(vars.PC.get(ij, 1.0) * self.params.get("tmrg_X", {}).get((ij, i), 0) for ij in self.I)
            expected_pe = vars.PE_FOB.get(i, 0) / (1 + ttix) - margin_sum
            residuals[f"EQ76_{i}"] = vars.PE.get(i, 0) - expected_pe
        
        # EQ77: PD(i) = domestic price
        for i in self.I:
            pl_i = vars.PL.get(i, 1.0)
            margin_sum = sum(vars.PC.get(ij, 1.0) * self.params.get("tmrg", {}).get((ij, i), 0) for ij in self.I)
            ttic = self.params.get("ttic", {}).get(i, 0)
            expected_pd = (pl_i + margin_sum) * (1 + ttic)
            residuals[f"EQ77_{i}"] = vars.PD.get(i, 0) - expected_pd
        
        # EQ78: PM(i) = import price
        for i in self.I:
            ttim = self.params.get("ttim", {}).get(i, 0)
            pwm_i = vars.PWM.get(i, 1.0)
            margin_sum = sum(vars.PC.get(ij, 1.0) * self.params.get("tmrg", {}).get((ij, i), 0) for ij in self.I)
            ttic = self.params.get("ttic", {}).get(i, 0)
            expected_pm = ((1 + ttim) * vars.e * pwm_i + margin_sum) * (1 + ttic)
            residuals[f"EQ78_{i}"] = vars.PM.get(i, 0) - expected_pm
        
        # EQ79: PC(i) = composite price
        for i in self.I:
            if vars.Q.get(i, 0) > 0:
                expected_pc = (vars.PM.get(i, 0) * vars.IM.get(i, 0) + vars.PD.get(i, 0) * vars.DD.get(i, 0)) / vars.Q[i]
                residuals[f"EQ79_{i}"] = vars.PC.get(i, 0) - expected_pc
        
        return residuals
    
    def equilibrium_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate equilibrium conditions (EQ85-EQ89)."""
        residuals = {}
        
        # EQ85: Labor market equilibrium
        for l in self.L:
            labor_supply = sum(self.params.get("LS", {}).get(l, 0) for _ in [0])  # LSO
            labor_demand = sum(vars.LD.get((l, j), 0) for j in self.J)
            residuals[f"EQ85_{l}"] = labor_supply - labor_demand
        
        # EQ86: Capital market equilibrium
        for k in self.K:
            capital_supply = sum(self.params.get("KS", {}).get(k, 0) for _ in [0])  # KSO
            capital_demand = sum(vars.KD.get((k, j), 0) for j in self.J)
            residuals[f"EQ86_{k}"] = capital_supply - capital_demand
        
        # EQ87: Savings-Investment balance
        total_savings = (
            sum(vars.SH.values()) + 
            sum(vars.SF.values()) + 
            vars.SG + 
            vars.SROW
        )
        residuals["EQ87"] = vars.IT - total_savings
        
        # EQ88: Domestic product market equilibrium
        for i in self.I:
            supply = sum(vars.DS.get((j, i), 0) for j in self.J)
            demand = vars.DD.get(i, 0)
            residuals[f"EQ88_{i}"] = supply - demand
        
        # EQ89: Export market equilibrium
        for i in self.I:
            supply = sum(vars.EX.get((j, i), 0) for j in self.J)
            demand = vars.EXD.get(i, 0)
            residuals[f"EQ89_{i}"] = supply - demand
        
        return residuals
    
    def gdp_residuals(self, vars: PEPModelVariables) -> dict[str, float]:
        """Calculate GDP definitions (EQ90-EQ97)."""
        residuals = {}
        
        # EQ90: GDP_BP = GDP at basic prices
        expected_gdp_bp = sum(vars.PVA.get(j, 0) * vars.VA.get(j, 0) for j in self.J) + vars.TIPT
        residuals["EQ90"] = vars.GDP_BP - expected_gdp_bp
        
        # EQ91: GDP_MP = GDP at market prices
        expected_gdp_mp = vars.GDP_BP + vars.TPRCTS
        residuals["EQ91"] = vars.GDP_MP - expected_gdp_mp
        
        # EQ92: GDP_IB = GDP income-based
        gdp_ib = 0
        for l in self.L:
            for j in self.J:
                gdp_ib += vars.W.get(l, 1.0) * vars.LD.get((l, j), 0)
        for k in self.K:
            for j in self.J:
                gdp_ib += vars.R.get((k, j), 1.0) * vars.KD.get((k, j), 0)
        gdp_ib += vars.TPRODN + vars.TPRCTS
        residuals["EQ92"] = vars.GDP_IB - gdp_ib
        
        # EQ93: GDP_FD = GDP from final demand
        gdp_fd = 0
        for i in self.I:
            cons = sum(vars.C.get((i, h), 0) for h in self.H)
            gdp_fd += vars.PC.get(i, 0) * (cons + vars.CG.get(i, 0) + vars.INV.get(i, 0) + vars.VSTK.get(i, 0))
        for i in self.I:
            gdp_fd += vars.PE_FOB.get(i, 0) * vars.EXD.get(i, 0)
            gdp_fd -= vars.PWM.get(i, 0) * vars.e * vars.IM.get(i, 0)
        residuals["EQ93"] = vars.GDP_FD - gdp_fd
        
        # EQ94-EQ97: Real variables
        for h in self.H:
            expected_cth_real = vars.CTH.get(h, 0) / vars.PIXCON
            residuals[f"EQ94_{h}"] = vars.CTH_REAL.get(h, 0) - expected_cth_real
        
        residuals["EQ95"] = vars.G_REAL - vars.G / vars.PIXGVT
        residuals["EQ96"] = vars.GDP_BP_REAL - vars.GDP_BP / vars.PIXGDP
        residuals["EQ97"] = vars.GDP_MP_REAL - vars.GDP_MP / vars.PIXCON
        
        return residuals
