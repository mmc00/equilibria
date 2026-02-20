"""Canonical array<->variable transforms for the PEP solver."""

from __future__ import annotations

import numpy as np

from equilibria.templates.pep_model_equations import PEPModelVariables


def pep_array_to_variables(
    x: np.ndarray,
    sets: dict[str, list[str]],
    *,
    min_price: float = 0.1,
) -> PEPModelVariables:
    """Convert flat solver vector to `PEPModelVariables`.

    Ordering is intentionally aligned with historical IPOPT packing order.
    """
    vars = PEPModelVariables()
    idx = 0

    # Production variables
    for sector in sets.get("J", []):
        if idx < len(x):
            vars.WC[sector] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.RC[sector] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.PP[sector] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.PT[sector] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.PVA[sector] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.PCI[sector] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.XST[sector] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.VA[sector] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.CI[sector] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.LDC[sector] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.KDC[sector] = max(0.0, float(x[idx]))
            idx += 1

        for labor in sets.get("L", []):
            if idx < len(x):
                vars.LD[(labor, sector)] = max(0.0, float(x[idx]))
                idx += 1
            if idx < len(x):
                vars.WTI[(labor, sector)] = max(min_price, float(x[idx]))
                idx += 1

        for capital in sets.get("K", []):
            if idx < len(x):
                vars.KD[(capital, sector)] = max(0.0, float(x[idx]))
                idx += 1
            if idx < len(x):
                vars.RTI[(capital, sector)] = max(min_price, float(x[idx]))
                idx += 1
            if idx < len(x):
                vars.R[(capital, sector)] = max(min_price, float(x[idx]))
                idx += 1

        for commodity in sets.get("I", []):
            if idx < len(x):
                vars.DI[(commodity, sector)] = max(0.0, float(x[idx]))
                idx += 1
            if idx < len(x):
                vars.XS[(sector, commodity)] = max(0.0, float(x[idx]))
                idx += 1
            if idx < len(x):
                vars.DS[(sector, commodity)] = max(0.0, float(x[idx]))
                idx += 1
            if idx < len(x):
                vars.EX[(sector, commodity)] = max(0.0, float(x[idx]))
                idx += 1
            if idx < len(x):
                vars.P[(sector, commodity)] = max(min_price, float(x[idx]))
                idx += 1

    # Wages
    for labor in sets.get("L", []):
        if idx < len(x):
            vars.W[labor] = max(min_price, float(x[idx]))
            idx += 1
    for capital in sets.get("K", []):
        if idx < len(x):
            vars.RK[capital] = max(min_price, float(x[idx]))
            idx += 1

    # Price and trade variables
    for commodity in sets.get("I", []):
        if idx < len(x):
            vars.PC[commodity] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.PD[commodity] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.PM[commodity] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.PE[commodity] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.PE_FOB[commodity] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.PL[commodity] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.PWM[commodity] = max(min_price, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.IM[commodity] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.DD[commodity] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.Q[commodity] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.EXD[commodity] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.TIC[commodity] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.TIM[commodity] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.TIX[commodity] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.MRGN[commodity] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.DIT[commodity] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.INV[commodity] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.CG[commodity] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.VSTK[commodity] = float(x[idx])
            idx += 1

    # Income variables
    for household in sets.get("H", []):
        if idx < len(x):
            vars.YH[household] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.YHL[household] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.YHK[household] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.YHTR[household] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.YDH[household] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.CTH[household] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.SH[household] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.TDH[household] = float(x[idx])
            idx += 1

        for commodity in sets.get("I", []):
            if idx < len(x):
                vars.C[(commodity, household)] = max(0.0, float(x[idx]))
                idx += 1
            if idx < len(x):
                vars.CMIN[(commodity, household)] = max(0.0, float(x[idx]))
                idx += 1

    for firm in sets.get("F", []):
        if idx < len(x):
            vars.YF[firm] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.YFK[firm] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.YFTR[firm] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.YDF[firm] = max(0.0, float(x[idx]))
            idx += 1
        if idx < len(x):
            vars.SF[firm] = float(x[idx])
            idx += 1
        if idx < len(x):
            vars.TDF[firm] = float(x[idx])
            idx += 1

    # Full transfer matrix TR(ag,agj)
    for agent in sets.get("AG", []):
        for source in sets.get("AG", []):
            if idx < len(x):
                vars.TR[(agent, source)] = float(x[idx])
                idx += 1

    # Government
    if idx < len(x):
        vars.YG = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.YGK = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.TDHT = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.TDFT = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.TPRCTS = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.TPRODN = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.TIWT = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.TIKT = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.TIPT = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.TICT = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.TIMT = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.TIXT = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.YGTR = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.G = max(0.0, float(x[idx]))
        idx += 1
    if idx < len(x):
        vars.SG = float(x[idx])
        idx += 1

    # ROW
    if idx < len(x):
        vars.YROW = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.SROW = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.CAB = float(x[idx])
        idx += 1

    # Investment
    if idx < len(x):
        vars.IT = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.GFCF = float(x[idx])
        idx += 1

    # GDP
    if idx < len(x):
        vars.GDP_BP = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.GDP_MP = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.GDP_IB = float(x[idx])
        idx += 1
    if idx < len(x):
        vars.GDP_FD = float(x[idx])
        idx += 1

    # Price indices
    if idx < len(x):
        vars.PIXCON = max(min_price, float(x[idx]))
        idx += 1
    if idx < len(x):
        vars.PIXGDP = max(min_price, float(x[idx]))
        idx += 1
    if idx < len(x):
        vars.PIXGVT = max(min_price, float(x[idx]))
        idx += 1
    if idx < len(x):
        vars.PIXINV = max(min_price, float(x[idx]))
        idx += 1

    # Real variables
    for household in sets.get("H", []):
        if idx < len(x):
            vars.CTH_REAL[household] = max(0.0, float(x[idx]))
            idx += 1
    if idx < len(x):
        vars.G_REAL = max(0.0, float(x[idx]))
        idx += 1
    if idx < len(x):
        vars.GDP_BP_REAL = max(0.0, float(x[idx]))
        idx += 1
    if idx < len(x):
        vars.GDP_MP_REAL = max(0.0, float(x[idx]))
        idx += 1
    if idx < len(x):
        vars.GFCF_REAL = max(0.0, float(x[idx]))
        idx += 1
    if idx < len(x):
        vars.LEON = float(x[idx])
        idx += 1

    # Exchange rate
    if idx < len(x):
        vars.e = max(min_price, float(x[idx]))
        idx += 1

    return vars


def pep_variables_to_array(
    vars: PEPModelVariables,
    sets: dict[str, list[str]],
) -> np.ndarray:
    """Convert `PEPModelVariables` to flat solver vector."""
    values: list[float] = []

    # Production variables
    for sector in sets.get("J", []):
        values.append(vars.WC.get(sector, 1.0))
        values.append(vars.RC.get(sector, 1.0))
        values.append(vars.PP.get(sector, 1.0))
        values.append(vars.PT.get(sector, 1.0))
        values.append(vars.PVA.get(sector, 1.0))
        values.append(vars.PCI.get(sector, 1.0))
        values.append(vars.XST.get(sector, 0.0))
        values.append(vars.VA.get(sector, 0.0))
        values.append(vars.CI.get(sector, 0.0))
        values.append(vars.LDC.get(sector, 0.0))
        values.append(vars.KDC.get(sector, 0.0))

        for labor in sets.get("L", []):
            values.append(vars.LD.get((labor, sector), 0.0))
            values.append(vars.WTI.get((labor, sector), 1.0))

        for capital in sets.get("K", []):
            values.append(vars.KD.get((capital, sector), 0.0))
            values.append(vars.RTI.get((capital, sector), 1.0))
            values.append(vars.R.get((capital, sector), 1.0))

        for commodity in sets.get("I", []):
            values.append(vars.DI.get((commodity, sector), 0.0))
            values.append(vars.XS.get((sector, commodity), 0.0))
            values.append(vars.DS.get((sector, commodity), 0.0))
            values.append(vars.EX.get((sector, commodity), 0.0))
            values.append(vars.P.get((sector, commodity), 1.0))

    # Wages
    for labor in sets.get("L", []):
        values.append(vars.W.get(labor, 1.0))
    for capital in sets.get("K", []):
        values.append(vars.RK.get(capital, 1.0))

    # Price and trade variables
    for commodity in sets.get("I", []):
        values.append(vars.PC.get(commodity, 1.0))
        values.append(vars.PD.get(commodity, 1.0))
        values.append(vars.PM.get(commodity, 1.0))
        values.append(vars.PE.get(commodity, 1.0))
        values.append(vars.PE_FOB.get(commodity, 1.0))
        values.append(vars.PL.get(commodity, 1.0))
        values.append(vars.PWM.get(commodity, 1.0))
        values.append(vars.IM.get(commodity, 0.0))
        values.append(vars.DD.get(commodity, 0.0))
        values.append(vars.Q.get(commodity, 0.0))
        values.append(vars.EXD.get(commodity, 0.0))
        values.append(vars.TIC.get(commodity, 0.0))
        values.append(vars.TIM.get(commodity, 0.0))
        values.append(vars.TIX.get(commodity, 0.0))
        values.append(vars.MRGN.get(commodity, 0.0))
        values.append(vars.DIT.get(commodity, 0.0))
        values.append(vars.INV.get(commodity, 0.0))
        values.append(vars.CG.get(commodity, 0.0))
        values.append(vars.VSTK.get(commodity, 0.0))

    # Income variables
    for household in sets.get("H", []):
        values.append(vars.YH.get(household, 0.0))
        values.append(vars.YHL.get(household, 0.0))
        values.append(vars.YHK.get(household, 0.0))
        values.append(vars.YHTR.get(household, 0.0))
        values.append(vars.YDH.get(household, 0.0))
        values.append(vars.CTH.get(household, 0.0))
        values.append(vars.SH.get(household, 0.0))
        values.append(vars.TDH.get(household, 0.0))

        for commodity in sets.get("I", []):
            values.append(vars.C.get((commodity, household), 0.0))
            values.append(vars.CMIN.get((commodity, household), 0.0))

    for firm in sets.get("F", []):
        values.append(vars.YF.get(firm, 0.0))
        values.append(vars.YFK.get(firm, 0.0))
        values.append(vars.YFTR.get(firm, 0.0))
        values.append(vars.YDF.get(firm, 0.0))
        values.append(vars.SF.get(firm, 0.0))
        values.append(vars.TDF.get(firm, 0.0))

    # Full transfer matrix TR(ag,agj)
    for agent in sets.get("AG", []):
        for source in sets.get("AG", []):
            values.append(vars.TR.get((agent, source), 0.0))

    # Government
    values.append(vars.YG)
    values.append(vars.YGK)
    values.append(vars.TDHT)
    values.append(vars.TDFT)
    values.append(vars.TPRCTS)
    values.append(vars.TPRODN)
    values.append(vars.TIWT)
    values.append(vars.TIKT)
    values.append(vars.TIPT)
    values.append(vars.TICT)
    values.append(vars.TIMT)
    values.append(vars.TIXT)
    values.append(vars.YGTR)
    values.append(vars.G)
    values.append(vars.SG)

    # ROW
    values.append(vars.YROW)
    values.append(vars.SROW)
    values.append(vars.CAB)

    # Investment
    values.append(vars.IT)
    values.append(vars.GFCF)

    # GDP
    values.append(vars.GDP_BP)
    values.append(vars.GDP_MP)
    values.append(vars.GDP_IB)
    values.append(vars.GDP_FD)

    # Price indices
    values.append(vars.PIXCON)
    values.append(vars.PIXGDP)
    values.append(vars.PIXGVT)
    values.append(vars.PIXINV)

    # Real variables
    for household in sets.get("H", []):
        values.append(vars.CTH_REAL.get(household, 0.0))
    values.append(vars.G_REAL)
    values.append(vars.GDP_BP_REAL)
    values.append(vars.GDP_MP_REAL)
    values.append(vars.GFCF_REAL)
    values.append(vars.LEON)

    # Exchange rate
    values.append(vars.e)

    return np.array(values)

