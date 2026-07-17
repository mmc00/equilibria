"""PEP equation blocks as Pyomo Constraints — faithful to pep_model_equations.py.

Each `EQn` residual `f=0` becomes `lhs == rhs`. Equations are instantiated only over
their active index sets (the GAMS `$`-masks, precomputed in `idx`). Constraints are named
`eqN` (lowercase) with the GAMS index suffix in the Constraint's own index.

`attach_all_blocks(m, S, P, idx, variant) -> int` attaches every block, returns the count.
"""
from __future__ import annotations

from pyomo.environ import Constraint


def _P(P, name, *idx):
    return P.get(name, *idx)


def attach_all_blocks(m, S, P, idx, variant) -> int:
    H, F, K, L, J, I = S.H, S.F, S.K, S.L, S.J, S.I
    AG, AGNG, AGD, I1, wi = S.AG, S.AGNG, S.AGD, S.I1, S.walras_i
    LDact, KDact, XSact = idx["LDact"], idx["KDact"], idx["XSact"]
    IMact, DDact, EXDact = idx["IMact"], idx["DDact"], idx["EXDact"]
    EXact, DSact = idx["EXact"], idx["DSact"]
    p = lambda *a: _P(P, *a)  # noqa: E731
    n = [0]

    def add(name, cset, rule):
        con = Constraint(cset, rule=rule) if cset is not None else Constraint(rule=rule)
        setattr(m, name, con)
        n[0] += len(con)

    # ---------------- PRODUCTION (EQ1-EQ9) ----------------
    add("eq1", J, lambda m, j: m.VA[j] == p("v", j) * m.XST[j])
    add("eq2", J, lambda m, j: m.CI[j] == p("io", j) * m.XST[j])

    def eq3(m, j):
        rho, bv, B = p("rho_VA", j), p("beta_VA", j), p("B_VA", j)
        if B == 0 or bv in (0.0, 1.0):
            return Constraint.Skip
        return m.VA[j] == B * (bv * m.LDC[j] ** (-rho)
                               + (1 - bv) * m.KDC[j] ** (-rho)) ** (-1.0 / rho)
    add("eq3", J, eq3)

    def eq4(m, j):
        bv, sg = p("beta_VA", j), p("sigma_VA", j)
        if bv in (0.0, 1.0):
            return Constraint.Skip
        return m.LDC[j] == ((bv / (1 - bv)) * (m.RC[j] / m.WC[j])) ** sg * m.KDC[j]
    add("eq4", J, eq4)

    def eq5(m, j):
        rho, B = p("rho_LD", j), p("B_LD", j)
        if B == 0:
            return Constraint.Skip
        return m.LDC[j] == B * sum(p("beta_LD", l, j) * m.LD[l, j] ** (-rho)
                                   for l in L) ** (-1.0 / rho)
    add("eq5", J, eq5)

    def eq6(m, l, j):
        sg = p("sigma_LD", j)
        return m.LD[l, j] == (p("beta_LD", l, j) * m.WC[j] / m.WTI[l, j]) ** sg \
            * p("B_LD", j) ** (sg - 1) * m.LDC[j]
    add("eq6", LDact, eq6)

    def eq7(m, j):
        rho, B = p("rho_KD", j), p("B_KD", j)
        if B == 0 or not any(kk == j for (_, kk) in [(k, j) for k in K if (k, j) in KDact]):
            return Constraint.Skip
        return m.KDC[j] == B * sum(p("beta_KD", k, j) * m.KD[k, j] ** (-rho)
                                   for k in K if (k, j) in KDact) ** (-1.0 / rho)
    add("eq7", J, eq7)

    def eq8(m, k, j):
        sg = p("sigma_KD", j)
        return m.KD[k, j] == (p("beta_KD", k, j) * m.RC[j] / m.RTI[k, j]) ** sg \
            * p("B_KD", j) ** (sg - 1) * m.KDC[j]
    add("eq8", KDact, eq8)

    add("eq9", [(i, j) for i in I for j in J],
        lambda m, i, j: m.DI[i, j] == p("aij", i, j) * m.CI[j])

    # ---------------- INCOME (EQ10-EQ21) ----------------
    add("eq10", H, lambda m, h: m.YH[h] == m.YHL[h] + m.YHK[h] + m.YHTR[h])
    add("eq11", H, lambda m, h: m.YHL[h] == sum(
        p("lambda_WL", h, l) * m.W[l] * sum(m.LD[l, j] for j in J if (l, j) in LDact)
        for l in L))
    add("eq12", H, lambda m, h: m.YHK[h] == sum(
        p("lambda_RK", h, k) * sum(m.R[k, j] * m.KD[k, j] for j in J if (k, j) in KDact)
        for k in K))
    add("eq13", H, lambda m, h: m.YHTR[h] == sum(m.TR[h, ag] for ag in AG))
    add("eq14", H, lambda m, h: m.YDH[h] == m.YH[h] - m.TDH[h] - m.TR["gvt", h])
    add("eq15", H, lambda m, h: m.CTH[h] == m.YDH[h] - m.SH[h]
        - sum(m.TR[ag, h] for ag in AGNG))
    add("eq16", H, lambda m, h: m.SH[h] == m.PIXCON ** p("eta") * p("sh0", h)
        + p("sh1", h) * m.YDH[h])
    add("eq17", F, lambda m, f: m.YF[f] == m.YFK[f] + m.YFTR[f])
    add("eq18", F, lambda m, f: m.YFK[f] == sum(
        p("lambda_RK", f, k) * sum(m.R[k, j] * m.KD[k, j] for j in J if (k, j) in KDact)
        for k in K))
    add("eq19", F, lambda m, f: m.YFTR[f] == sum(m.TR[f, ag] for ag in AG))
    add("eq20", F, lambda m, f: m.YDF[f] == m.YF[f] - m.TDF[f])
    add("eq21", F, lambda m, f: m.SF[f] == m.YDF[f] - sum(m.TR[ag, f] for ag in AG))

    # ---------------- GOVERNMENT (EQ22-EQ43) ----------------
    add("eq22", None, lambda m: m.YG == m.YGK + m.TDHT + m.TDFT + m.TPRODN + m.TPRCTS + m.YGTR)
    add("eq23", None, lambda m: m.YGK == sum(
        p("lambda_RK", "gvt", k) * sum(m.R[k, j] * m.KD[k, j] for j in J if (k, j) in KDact)
        for k in K))
    add("eq24", None, lambda m: m.TDHT == sum(m.TDH[h] for h in H))
    add("eq25", None, lambda m: m.TDFT == sum(m.TDF[f] for f in F))
    add("eq26", None, lambda m: m.TPRODN == m.TIWT + m.TIKT + m.TIPT)
    add("eq27", None, lambda m: m.TIWT == sum(m.TIW[l, j] for (l, j) in LDact))
    add("eq28", None, lambda m: m.TIKT == sum(m.TIK[k, j] for (k, j) in KDact))
    add("eq29", None, lambda m: m.TIPT == sum(m.TIP[j] for j in J))
    add("eq30", None, lambda m: m.TPRCTS == m.TICT + m.TIMT + m.TIXT)
    add("eq31", None, lambda m: m.TICT == sum(m.TIC[i] for i in I))
    add("eq32", None, lambda m: m.TIMT == sum(m.TIM[i] for i in IMact))
    add("eq33", None, lambda m: m.TIXT == sum(m.TIX[i] for i in EXDact))
    add("eq34", None, lambda m: m.YGTR == sum(m.TR["gvt", ag] for ag in AGNG))
    add("eq35", H, lambda m, h: m.TDH[h] == m.PIXCON ** p("eta") * p("ttdh0", h)
        + p("ttdh1", h) * m.YH[h])
    add("eq36", F, lambda m, f: m.TDF[f] == m.PIXCON ** p("eta") * p("ttdf0", f)
        + p("ttdf1", f) * m.YFK[f])
    add("eq37", LDact, lambda m, l, j: m.TIW[l, j] == p("ttiw", l, j) * m.W[l] * m.LD[l, j])
    add("eq38", KDact, lambda m, k, j: m.TIK[k, j] == p("ttik", k, j) * m.R[k, j] * m.KD[k, j])
    add("eq39", J, lambda m, j: m.TIP[j] == p("ttip", j) * m.PP[j] * m.XST[j])

    def eq40(m, i):
        tc = p("ttic", i)
        dd = m.PD[i] * m.DD[i] if i in DDact else 0.0
        im = m.PM[i] * m.IM[i] if i in IMact else 0.0
        return m.TIC[i] == (tc / (1 + tc)) * (dd + im)
    add("eq40", I, eq40)
    add("eq41", IMact, lambda m, i: m.TIM[i] == p("ttim", i) * m.e * p("PWM", i) * m.IM[i])
    add("eq42", EXDact, lambda m, i: m.TIX[i] == p("ttix", i)
        * (m.PE[i] + sum(m.PC[ij] * p("tmrg_X", ij, i) for ij in I)) * m.EXD[i])
    add("eq43", None, lambda m: m.SG == m.YG - sum(m.TR[ag, "gvt"] for ag in AGNG) - m.G)

    # ---------------- REST OF WORLD (EQ44-EQ46) ----------------
    add("eq44", None, lambda m: m.YROW == sum(m.e * p("PWM", i) * m.IM[i] for i in IMact)
        + sum(p("lambda_RK", "row", k)
              * sum(m.R[k, j] * m.KD[k, j] for j in J if (k, j) in KDact) for k in K)
        + sum(m.TR["row", ag] for ag in AGD))
    add("eq45", None, lambda m: m.SROW == m.YROW - sum(m.PE_FOB[i] * m.EXD[i] for i in EXDact)
        - sum(m.TR[ag, "row"] for ag in AGD))
    add("eq46", None, lambda m: m.SROW == -m.CAB)

    # ---------------- TRANSFERS (EQ47-EQ51) ----------------
    add("eq47", [(ag, h) for ag in AGNG for h in H],
        lambda m, ag, h: m.TR[ag, h] == p("lambda_TR_households", ag, h) * m.YDH[h])
    add("eq48", H, lambda m, h: m.TR["gvt", h] == m.PIXCON ** p("eta") * p("tr0", h)
        + p("tr1", h) * m.YH[h])
    add("eq49", [(ag, f) for ag in AG for f in F],
        lambda m, ag, f: m.TR[ag, f] == p("lambda_TR_firms", ag, f) * m.YDF[f])
    add("eq50", AGNG, lambda m, ag: m.TR[ag, "gvt"] == m.PIXCON ** p("eta") * p("TRO", ag, "gvt"))
    add("eq51", AGD, lambda m, ag: m.TR[ag, "row"] == m.PIXCON ** p("eta") * p("TRO", ag, "row"))

    # ---------------- DEMAND (EQ52-EQ57) ----------------
    add("eq52", [(i, h) for i in I for h in H], lambda m, i, h:
        m.PC[i] * m.C[i, h] == m.PC[i] * m.CMIN[i, h]
        + p("gamma_LES", i, h) * (m.CTH[h] - sum(m.PC[ij] * m.CMIN[ij, h] for ij in I)))
    add("eq53", None, lambda m: m.GFCF == m.IT - sum(m.PC[i] * m.VSTK[i] for i in I))
    add("eq54", I, lambda m, i: m.PC[i] * m.INV[i] == p("gamma_INV", i) * m.GFCF)
    add("eq55", I, lambda m, i: m.PC[i] * m.CG[i] == p("gamma_GVT", i) * m.G)
    add("eq56", I, lambda m, i: m.DIT[i] == sum(m.DI[i, j] for j in J))
    add("eq57", I, lambda m, i: m.MRGN[i] == sum(
        p("tmrg", i, ij) * (m.DD[ij] if ij in DDact else 0.0)
        + p("tmrg", i, ij) * (m.IM[ij] if ij in IMact else 0.0)
        + p("tmrg_X", i, ij) * (m.EXD[ij] if ij in EXDact else 0.0) for ij in I))

    # ---------------- TRADE (EQ58-EQ64) ----------------
    def eq58(m, j):
        rho, B = p("rho_XT", j), p("B_XT", j)
        prods = [i for i in I if (j, i) in XSact]
        if B == 0 or not prods:
            return Constraint.Skip
        return m.XST[j] == B * sum(p("beta_XT", j, i) * m.XS[j, i] ** rho
                                   for i in prods) ** (1.0 / rho)
    add("eq58", J, eq58)

    def eq59(m, j, i):
        prods = [ii for ii in I if (j, ii) in XSact]
        if len(prods) <= 1:
            return Constraint.Skip
        sg, B = p("sigma_XT", j), p("B_XT", j)
        return m.XS[j, i] == m.XST[j] / B ** (1 + sg) \
            * (m.P[j, i] / (p("beta_XT", j, i) * m.PT[j])) ** sg
    add("eq59", XSact, eq59)

    def eq60(m, j, i):
        rho, B, bx = p("rho_X", j, i), p("B_X", j, i), p("beta_X", j, i)
        ex = bx * m.EX[j, i] ** rho if (j, i) in EXact else 0.0
        ds = (1 - bx) * m.DS[j, i] ** rho if (j, i) in DSact else 0.0
        if B == 0:
            return Constraint.Skip
        return m.XS[j, i] == B * (ex + ds) ** (1.0 / rho)
    add("eq60", XSact, eq60)

    def eq61(m, j, i):
        if (j, i) not in EXact or (j, i) not in DSact:
            return Constraint.Skip
        sg, bx = p("sigma_X", j, i), p("beta_X", j, i)
        return m.EX[j, i] == (((1 - bx) / bx) * (m.PE[i] / m.PL[i])) ** sg * m.DS[j, i]
    add("eq61", XSact, eq61)

    add("eq62", EXDact, lambda m, i: m.EXD[i] == p("EXDO", i)
        * (m.e * p("PWX", i) / m.PE_FOB[i]) ** p("sigma_XD"))

    def eq63(m, i):
        rho, B, bm = p("rho_M", i), p("B_M", i), p("beta_M", i)
        im = bm * m.IM[i] ** (-rho) if i in IMact else 0.0
        dd = (1 - bm) * m.DD[i] ** (-rho) if i in DDact else 0.0
        if B == 0 or (i not in IMact and i not in DDact):
            return Constraint.Skip
        return m.Q[i] == B * (im + dd) ** (-1.0 / rho)
    add("eq63", I, eq63)

    def eq64(m, i):
        if i not in IMact or i not in DDact:
            return Constraint.Skip
        sg, bm = p("sigma_M", i), p("beta_M", i)
        return m.IM[i] == ((bm / (1 - bm)) * (m.PD[i] / m.PM[i])) ** sg * m.DD[i]
    add("eq64", I, eq64)

    # ---------------- PRICES (EQ65-EQ83) ----------------
    add("eq65", J, lambda m, j: m.PP[j] * m.XST[j] == m.PVA[j] * m.VA[j] + m.PCI[j] * m.CI[j])
    add("eq66", J, lambda m, j: m.PT[j] == (1 + p("ttip", j)) * m.PP[j])
    add("eq67", J, lambda m, j: m.PCI[j] * m.CI[j] == sum(m.PC[i] * m.DI[i, j] for i in I))
    add("eq68", J, lambda m, j: m.PVA[j] * m.VA[j] == m.WC[j] * m.LDC[j] + m.RC[j] * m.KDC[j])
    add("eq70", LDact, lambda m, l, j: m.WTI[l, j] == m.W[l] * (1 + p("ttiw", l, j)))
    add("eq72", KDact, lambda m, k, j: m.RTI[k, j] == m.R[k, j] * (1 + p("ttik", k, j)))
    add("eq73", KDact, lambda m, k, j: (m.R[k, j] == m.RK[k])
        if p("kmob") else Constraint.Skip)

    def eq74(m, j, i):
        # single-product sector: P == PT
        if abs(p("XSO", j, i) - p("XSTO", j)) > 1e-9:
            return Constraint.Skip
        return m.P[j, i] == m.PT[j]
    add("eq74", XSact, eq74)

    def eq75(m, j, i):
        ex = m.PE[i] * m.EX[j, i] if (j, i) in EXact else 0.0
        ds = m.PL[i] * m.DS[j, i] if (j, i) in DSact else 0.0
        return m.P[j, i] * m.XS[j, i] == ex + ds
    add("eq75", XSact, eq75)

    add("eq76", EXDact, lambda m, i: m.PE[i] == m.PE_FOB[i] / (1 + p("ttix", i))
        - sum(m.PC[ij] * p("tmrg_X", ij, i) for ij in I))
    add("eq77", DDact, lambda m, i: m.PD[i] == (m.PL[i]
        + sum(m.PC[ij] * p("tmrg", ij, i) for ij in I)) * (1 + p("ttic", i)))
    add("eq78", IMact, lambda m, i: m.PM[i] == ((1 + p("ttim", i)) * m.e * p("PWM", i)
        + sum(m.PC[ij] * p("tmrg", ij, i) for ij in I)) * (1 + p("ttic", i)))

    def eq79(m, i):
        im = m.PM[i] * m.IM[i] if i in IMact else 0.0
        dd = m.PD[i] * m.DD[i] if i in DDact else 0.0
        return m.PC[i] * m.Q[i] == im + dd
    add("eq79", I, eq79)

    # EQ80 GDP deflator (Fisher of unit costs) — faithful to pep_model_equations.py:1069-1086.
    # unit_cur[j] = (PVA[j]*VA[j] + TIP[j]) / VA[j] ; unit_base[j] = (PVAO*VAO+TIPO)/VAO.
    # num1/den1 weight by benchmark VA (Laspeyres); num2/den2 weight by current VA (Paasche).
    # Active only where both current and benchmark VA are non-trivial (VAO mask at build).
    def eq80(m):
        jset = [j for j in J if p("VAO", j) > 1e-12]
        num1 = sum(((m.PVA[j] * m.VA[j] + m.TIP[j]) / m.VA[j]) * p("VAO", j) for j in jset)
        num2 = sum(((m.PVA[j] * m.VA[j] + m.TIP[j]) / m.VA[j]) * m.VA[j] for j in jset)
        den1 = sum(((p("PVAO", j) * p("VAO", j) + p("TIPO", j)) / p("VAO", j)) * p("VAO", j)
                   for j in jset)  # constant
        den2 = sum(((p("PVAO", j) * p("VAO", j) + p("TIPO", j)) / p("VAO", j)) * m.VA[j]
                   for j in jset)
        return m.PIXGDP == ((num1 / den1) * (num2 / den2)) ** 0.5
    add("eq80", None, eq80)

    add("eq81", None, lambda m: m.PIXCON * sum(p("PCO", i) * sum(p("CO", i, h) for h in H)
                                               for i in I)
        == sum(m.PC[i] * sum(p("CO", i, h) for h in H) for i in I))

    from pyomo.environ import log, exp
    add("eq82", None, lambda m: m.PIXINV == exp(sum(p("gamma_INV", i)
                                                    * log(m.PC[i] / p("PCO", i)) for i in I)))
    add("eq83", None, lambda m: m.PIXGVT == exp(sum(p("gamma_GVT", i)
                                                    * log(m.PC[i] / p("PCO", i)) for i in I)))

    # ---------------- EQUILIBRIUM (EQ84-EQ89, WALRAS) ----------------
    add("eq85", L, lambda m, l: m.LS[l] == sum(m.LD[l, j] for j in J))
    add("eq86", K, lambda m, k: m.KS[k] == sum(m.KD[k, j] for j in J))
    add("eq87", None, lambda m: m.IT == sum(m.SH[h] for h in H) + sum(m.SF[f] for f in F)
        + m.SG + m.SROW)
    add("eq84", I1, lambda m, i: m.Q[i] == sum(m.C[i, h] for h in H) + m.CG[i]
        + m.INV[i] + m.VSTK[i] + m.DIT[i] + m.MRGN[i])
    add("eq88", DDact, lambda m, i: sum(m.DS[j, i] for j in J if (j, i) in DSact) == m.DD[i])
    add("eq89", EXDact, lambda m, i: sum(m.EX[j, i] for j in J if (j, i) in EXact) == m.EXD[i])
    # WALRAS: LEON = excess supply of the redundant (agr) market
    add("walras", None, lambda m: m.LEON == m.Q[wi] - sum(m.C[wi, h] for h in H)
        - m.CG[wi] - m.INV[wi] - m.VSTK[wi] - m.DIT[wi] - m.MRGN[wi])

    # ---------------- GDP DEFINITIONS (EQ90-EQ98) ----------------
    add("eq90", None, lambda m: m.GDP_BP == sum(m.PVA[j] * m.VA[j] for j in J) + m.TIPT)
    add("eq91", None, lambda m: m.GDP_MP == m.GDP_BP + m.TPRCTS)
    add("eq92", None, lambda m: m.GDP_IB == sum(m.W[l] * m.LD[l, j] for (l, j) in LDact)
        + sum(m.R[k, j] * m.KD[k, j] for (k, j) in KDact) + m.TPRODN + m.TPRCTS)
    add("eq93", None, lambda m: m.GDP_FD == sum(m.PC[i] * (sum(m.C[i, h] for h in H)
        + m.CG[i] + m.INV[i] + m.VSTK[i]) for i in I)
        + sum(m.PE_FOB[i] * m.EXD[i] for i in EXDact)
        - sum(p("PWM", i) * m.e * m.IM[i] for i in IMact))
    add("eq94", H, lambda m, h: m.CTH_REAL[h] == m.CTH[h] / m.PIXCON)
    add("eq95", None, lambda m: m.G_REAL == m.G / m.PIXGVT)
    add("eq96", None, lambda m: m.GDP_BP_REAL == m.GDP_BP / m.PIXGDP)
    add("eq97", None, lambda m: m.GDP_MP_REAL == m.GDP_MP / m.PIXCON)
    add("eq98", None, lambda m: m.GFCF_REAL == m.GFCF / m.PIXINV)

    return n[0]
