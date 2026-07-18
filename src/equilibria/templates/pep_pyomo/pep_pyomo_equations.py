"""PEP CGE model as a Pyomo ConcreteModel — Vars + all ~96 EQ Constraints.

Faithful port of the cyipopt residual system (pep_model_equations.py). Each residual
`f(x)=0` becomes a Pyomo `Constraint` `lhs == rhs`. Sets/params come from PEPSets/PEPParams
(which wrap the calibrated PEPModelState). The `$`-masks of the GAMS/residual code are
resolved at construction from the `*O`/`*O0` benchmark params (an equation instance is
built only where its benchmark level is non-zero), matching the cyipopt active-set.

Builder entry point: `build_pep_model(state, variant="base", form="nlp") -> ConcreteModel`.
  variant: "base" (EQ1..EQ98 + WALRAS) | "objdef" (adds OBJDEF: OBJ==0, free OBJ)
  form:    "nlp" (min 0 over the equality system) | "mcp" (walras⊥LEON, e fixed numeraire)
"""
from __future__ import annotations

from typing import Any

from pyomo.environ import (
    ConcreteModel, Var, Constraint, Objective, Reals, NonNegativeReals,
    Param, value, minimize,
)

from .pep_pyomo_sets import PEPSets
from .pep_pyomo_parameters import PEPParams

POS = (1e-6, None)   # strictly positive (prices, quantities that must stay > 0)


def _active(p: PEPParams, mask: str, *idx) -> bool:
    """True where the benchmark mask level is non-zero (the GAMS $-guard)."""
    try:
        return abs(p.get(mask, *idx)) > 1e-12
    except KeyError:
        return False


def build_pep_model(state: Any, variant: str = "base", form: str = "nlp") -> ConcreteModel:
    S = PEPSets.from_state(state)
    P = PEPParams(state)
    m = ConcreteModel(name=f"PEP_{variant}_{form}")
    m._pep = {"sets": S, "params": P, "variant": variant, "form": form}

    # ---- python-side index lists (Pyomo Sets are implicit via rule domains) ----
    H, F, K, L, J, I = S.H, S.F, S.K, S.L, S.J, S.I
    AG, AGNG, AGD, I1 = S.AG, S.AGNG, S.AGD, S.I1
    LDact = [(l, j) for l in L for j in J if _active(P, "LDO", l, j)]
    KDact = [(k, j) for k in K for j in J if _active(P, "KDO", k, j)]
    XSact = [(j, i) for j in J for i in I if _active(P, "XSO", j, i)]
    IMact = [i for i in I if _active(P, "IMO", i)]
    DDact = [i for i in I if _active(P, "DDO", i)]
    EXDact = [i for i in I if _active(P, "EXDO", i)]
    EXact = [(j, i) for j in J for i in I if _active(P, "EXO", j, i)]
    DSact = [(j, i) for j in J for i in I if _active(P, "DSO", j, i)]
    m._pep["idx"] = dict(LDact=LDact, KDact=KDact, XSact=XSact, IMact=IMact,
                         DDact=DDact, EXDact=EXDact, EXact=EXact, DSact=DSact)

    # ================= VARIABLES (init at benchmark *O levels) =================
    # Benchmark levels missing from calibration, filled from CGE normalization:
    #   W/RK/R = 1.0 (factor prices normalized at base), TIW/TIK from tax·price·quantity.
    def _bench(name, *idx):
        try:
            v = P.get(name, *idx)
            if v not in (None, 0.0):
                return v
        except KeyError:
            pass
        if name in ("WO", "RKO", "RO", "PPO", "PTO", "PVAO", "PCIO", "WCO", "RCO",
                    "PCO", "PDO", "PMO", "PEO", "PE_FOBO", "PLO", "PO", "WTIO", "RTIO"):
            return 1.0                       # prices normalized to 1 at benchmark
        if name == "TIWO" and len(idx) == 2:
            l, j = idx
            return P.get("ttiw", l, j) * 1.0 * P.get("LDO", l, j)
        if name == "TIKO" and len(idx) == 2:
            k, j = idx
            return P.get("ttik", k, j) * 1.0 * P.get("KDO", k, j)
        return 1e-6

    def _init(name):
        def r(mm, *idx):
            return _bench(name, *idx)
        return r

    # production
    m.VA = Var(J, domain=NonNegativeReals, initialize=_init("VAO"))
    m.CI = Var(J, domain=NonNegativeReals, initialize=_init("CIO"))
    m.LDC = Var(J, domain=NonNegativeReals, initialize=_init("LDCO"))
    m.KDC = Var(J, domain=NonNegativeReals, initialize=_init("KDCO"))
    m.LD = Var(L, J, domain=NonNegativeReals, initialize=_init("LDO"))
    m.KD = Var(K, J, domain=NonNegativeReals, initialize=_init("KDO"))
    m.DI = Var(I, J, domain=NonNegativeReals, initialize=_init("DIO"))
    m.XST = Var(J, domain=NonNegativeReals, initialize=_init("XSTO"))
    m.XS = Var(J, I, domain=NonNegativeReals, initialize=_init("XSO"))
    # prices (positive)
    m.WC = Var(J, bounds=POS, initialize=_init("WCO"))
    m.RC = Var(J, bounds=POS, initialize=_init("RCO"))
    m.PP = Var(J, bounds=POS, initialize=_init("PPO"))
    m.PT = Var(J, bounds=POS, initialize=_init("PTO"))
    m.PVA = Var(J, bounds=POS, initialize=_init("PVAO"))
    m.PCI = Var(J, bounds=POS, initialize=_init("PCIO"))
    m.W = Var(L, bounds=POS, initialize=_init("WO"))
    m.RK = Var(K, bounds=POS, initialize=_init("RKO"))
    m.R = Var(K, J, bounds=POS, initialize=_init("RO"))
    m.WTI = Var(L, J, bounds=POS, initialize=_init("WTIO"))
    m.RTI = Var(K, J, bounds=POS, initialize=_init("RTIO"))
    m.P = Var(J, I, bounds=POS, initialize=_init("PO"))
    m.PC = Var(I, bounds=POS, initialize=_init("PCO"))
    m.PD = Var(I, bounds=POS, initialize=_init("PDO"))
    m.PM = Var(I, bounds=POS, initialize=_init("PMO"))
    m.PE = Var(I, bounds=POS, initialize=_init("PEO"))
    m.PE_FOB = Var(I, bounds=POS, initialize=_init("PE_FOBO"))
    m.PL = Var(I, bounds=POS, initialize=_init("PLO"))
    # trade quantities
    m.IM = Var(I, domain=NonNegativeReals, initialize=_init("IMO"))
    m.DD = Var(I, domain=NonNegativeReals, initialize=_init("DDO"))
    m.DS = Var(J, I, domain=NonNegativeReals, initialize=_init("DSO"))
    m.EX = Var(J, I, domain=NonNegativeReals, initialize=_init("EXO"))
    m.EXD = Var(I, domain=NonNegativeReals, initialize=_init("EXDO"))
    m.Q = Var(I, domain=NonNegativeReals, initialize=_init("QO"))
    # income (households)
    m.YH = Var(H, domain=NonNegativeReals, initialize=_init("YHO"))
    m.YHL = Var(H, domain=NonNegativeReals, initialize=_init("YHLO"))
    m.YHK = Var(H, domain=NonNegativeReals, initialize=_init("YHKO"))
    m.YHTR = Var(H, domain=Reals, initialize=_init("YHTRO"))
    m.YDH = Var(H, domain=NonNegativeReals, initialize=_init("YDHO"))
    m.CTH = Var(H, domain=NonNegativeReals, initialize=_init("CTHO"))
    m.SH = Var(H, domain=Reals, initialize=_init("SHO"))
    m.TDH = Var(H, domain=Reals, initialize=_init("TDHO"))
    m.C = Var(I, H, domain=NonNegativeReals, initialize=_init("CO"))
    # income (firms)
    m.YF = Var(F, domain=NonNegativeReals, initialize=_init("YFO"))
    m.YFK = Var(F, domain=NonNegativeReals, initialize=_init("YFKO"))
    m.YFTR = Var(F, domain=Reals, initialize=_init("YFTRO"))
    m.YDF = Var(F, domain=NonNegativeReals, initialize=_init("YDFO"))
    m.SF = Var(F, domain=Reals, initialize=_init("SFO"))
    m.TDF = Var(F, domain=Reals, initialize=_init("TDFO"))
    # transfers (full AG x AG)
    m.TR = Var(AG, AG, domain=Reals, initialize=_init("TRO"))
    m.TIW = Var(L, J, domain=Reals, initialize=_init("TIWO"))
    m.TIK = Var(K, J, domain=Reals, initialize=_init("TIKO"))
    m.TIP = Var(J, domain=Reals, initialize=_init("TIPO"))
    m.TIC = Var(I, domain=Reals, initialize=_init("TICO"))
    m.TIM = Var(I, domain=Reals, initialize=_init("TIMO"))
    m.TIX = Var(I, domain=Reals, initialize=_init("TIXO"))
    # government aggregates — seed from the defining sums (eq24-34) so they're seed-consistent,
    # not 0 (TIWTO/TIKTO/... aren't calibrated params).
    _tiwt = sum(_bench("TIWO", l, j) for l in L for j in J)
    _tikt = sum(_bench("TIKO", k, j) for k in K for j in J)
    _tipt = sum(_bench("TIPO", j) for j in J)
    _tict = sum(_bench("TICO", i) for i in I)
    _timt = sum(_bench("TIMO", i) for i in I)
    _tixt = sum(_bench("TIXO", i) for i in I)
    _gov_seed = {
        "TIWT": _tiwt, "TIKT": _tikt, "TIPT": _tipt, "TICT": _tict, "TIMT": _timt,
        "TIXT": _tixt, "TPRODN": _tiwt + _tikt + _tipt, "TPRCTS": _tict + _timt + _tixt,
        "TDHT": sum(_bench("TDHO", h) for h in H), "TDFT": sum(_bench("TDFO", f) for f in F),
        "YGK": _bench("YGKO"),
        "YGTR": sum(_bench("TRO", "gvt", ag) for ag in AGNG),   # eq34
        "G": _bench("GO"),
    }
    # YG (eq22) and SG (eq43) seeded from their component identities for consistency
    _gov_seed["YG"] = (_gov_seed["YGK"] + _gov_seed["TDHT"] + _gov_seed["TDFT"]
                       + _gov_seed["TPRODN"] + _gov_seed["TPRCTS"] + _gov_seed["YGTR"])
    _gov_seed["SG"] = (_gov_seed["YG"]
                       - sum(_bench("TRO", ag, "gvt") for ag in AGNG) - _gov_seed["G"])
    for nm in ("YG", "YGK", "TDHT", "TDFT", "TPRODN", "TPRCTS", "TIWT", "TIKT",
               "TIPT", "TICT", "TIMT", "TIXT", "YGTR", "SG", "G"):
        seed = _gov_seed.get(nm, 0.0)
        if seed in (0.0, 1e-6) and (nm + "O") in P:
            seed = P.get(nm + "O")
        setattr(m, nm, Var(domain=Reals, initialize=seed))
    # rest of world
    m.YROW = Var(domain=Reals, initialize=P.get("YROWO") if "YROWO" in P else 0.0)
    m.SROW = Var(domain=Reals, initialize=P.get("SROWO") if "SROWO" in P else 0.0)
    m.CAB = Var(domain=Reals, initialize=P.get("CABO") if "CABO" in P else 0.0)
    # demand
    m.GFCF = Var(domain=Reals, initialize=P.get("GFCFO") if "GFCFO" in P else 0.0)
    m.INV = Var(I, domain=Reals, initialize=_init("INVO"))
    m.CG = Var(I, domain=Reals, initialize=_init("CGO"))
    m.DIT = Var(I, domain=Reals, initialize=_init("DITO"))
    m.MRGN = Var(I, domain=Reals, initialize=_init("MRGNO"))
    m.VSTK = Var(I, domain=Reals, initialize=_init("VSTKO"))
    m.CMIN = Var(I, H, domain=NonNegativeReals, initialize=_init("CMINO"))
    # macro / price indices / real
    for nm in ("PIXCON", "PIXGDP", "PIXINV", "PIXGVT"):
        setattr(m, nm, Var(bounds=POS, initialize=1.0))
    # GDP_IB is a pure report var appearing ONLY in eq92. Seeding it from GDP_IBO leaves
    # eq92 with the calibration's 2026 imbalance (GDP_MPO 53681 ≠ GDP_IBO 51655) — an
    # infeasibility that forces IPOPT to drift the whole system. GDP_IB is free, so seed
    # it from the eq92 RHS (factor income + TPRODN + TPRCTS) → the seed is fully feasible
    # and IPOPT stays at the benchmark. (GDP_IB then reports the income-side GDP, 53681.)
    _gdp_ib_seed = (sum(_bench("WO", l) * _bench("LDO", l, j) for (l, j) in LDact)
                    + sum(_bench("RO", k, j) * _bench("KDO", k, j) for (k, j) in KDact)
                    + _gov_seed["TPRODN"] + _gov_seed["TPRCTS"])
    _macro_seed = {"GDP_BP": _bench("GDP_BPO"), "GDP_MP": _bench("GDP_MPO"),
                   "GDP_IB": _gdp_ib_seed, "GDP_FD": _bench("GDP_FDO"), "IT": _bench("ITO")}
    for nm in ("GDP_BP", "GDP_MP", "GDP_IB", "GDP_FD", "IT"):
        seed = _macro_seed.get(nm, 0.0)
        if seed in (0.0, 1e-6) and (nm + "O") in P:
            seed = P.get(nm + "O")
        setattr(m, nm, Var(domain=Reals, initialize=seed))
    m.CTH_REAL = Var(H, domain=NonNegativeReals, initialize=_init("CTHO"))
    # real vars = nominal / price-index; at benchmark PIX*=1 so seed = nominal *O level
    _real_nom = {"G_REAL": "GO", "GDP_BP_REAL": "GDP_BPO",
                 "GDP_MP_REAL": "GDP_MPO", "GFCF_REAL": "GFCFO"}
    for nm, nomO in _real_nom.items():
        setattr(m, nm, Var(domain=NonNegativeReals, initialize=_bench(nomO)))
    m.LS = Var(L, domain=NonNegativeReals, initialize=_init("LSO"))
    m.KS = Var(K, domain=NonNegativeReals, initialize=_init("KSO"))
    m.e = Var(bounds=POS, initialize=1.0)     # exchange rate / numeraire
    m.LEON = Var(domain=Reals, initialize=0.0)  # Walras slack (free)
    if variant == "objdef":
        m.OBJ = Var(domain=Reals, initialize=0.0)

    # Constraints are attached by the block builders (imported lazily to keep this
    # file navigable). Each returns the count of instantiated constraints.
    from .pep_pyomo_blocks import attach_all_blocks
    n = attach_all_blocks(m, S, P, m._pep["idx"], variant)
    m._pep["n_constraints"] = n

    # ---- closure: fix numeraire + exogenous vars at benchmark (both NLP and MCP) ----
    # Without this the homogeneous CGE system has a scaling DOF and IPOPT drifts to a
    # spurious scaled root. e is the PEP numeraire; KS/LS/PWM/PWX/CMIN/VSTK/G/CAB are
    # closure-fixed (pep_contract.py:74-89).
    m.e.fix(1.0)
    for k in K:
        m.KS[k].fix(_bench("KSO", k))
    for l in L:
        m.LS[l].fix(_bench("LSO", l))
    # PWM/PWX are constant params (world prices), not Vars — already fixed by construction.
    for i in I:
        m.VSTK[i].fix(_bench("VSTKO", i))
        for h in H:
            m.CMIN[i, h].fix(_bench("CMINO", i, h))
    m.G.fix(_bench("GO"))
    m.CAB.fix(_bench("CABO"))

    # NOTE on structural-zero prices (e.g. PD['othind'], which has no domestic supply so
    # eq77 is $-masked out): GAMS's PEP NLP/CNS does NOT pin such a price — it keeps the
    # calibrated SAM value (valPD['othind']=1.132), a benign free DOF that no equilibrium
    # value depends on. So the NLP leaves it at its seed (100% GAMS parity). The MCP form
    # DOES pin it (to 1.0) below — the square complementarity system needs a determinate
    # value for the zero-quantity cell, and 1.0 is inert. This is the ONE cell where the
    # NLP↔MCP mirror is 99.79% not 100%, and that gap is faithful, not a bug: forcing the
    # NLP to 1.0 would break GAMS parity to manufacture a mirror. (Do NOT re-add a
    # both-forms price fix — verified 2026-07-18 that GAMS keeps 1.132 here.)

    # ---- MCP square-closure: fix structurally-zero cells of matrix variables ----
    # The equations are instantiated only over their active masks (XSact/DSact/…), but the
    # Vars are declared DENSE. In the NLP a constant-0 objective absorbs the extra free DOF;
    # an MCP must be SQUARE (one eq per free var). So fix every matrix-var cell that has NO
    # equation defining it (outside its benchmark mask) at its benchmark value. This mirrors
    # the GAMS $-guard that makes those cells exogenous zeros.
    if form == "mcp":
        # TR endogenous cells = the union of the transfer equations' actual masks:
        #   EQ47 TR(agng,h) where lambda_TR_households≠0; EQ48 TR('gvt',h) all h;
        #   EQ49 TR(ag,f) where lambda_TR_firms≠0; EQ50 TR(agng,'gvt'); EQ51 TR(agd,'row').
        lam_h = P.get("lambda_TR_households") if "lambda_TR_households" in P else {}
        lam_f = P.get("lambda_TR_firms") if "lambda_TR_firms" in P else {}
        tr_active = set()
        if isinstance(lam_h, dict):
            tr_active |= set(lam_h.keys())
        if isinstance(lam_f, dict):
            tr_active |= set(lam_f.keys())
        for h in H:
            tr_active.add(("gvt", h))
        for ag in AGNG:
            tr_active.add((ag, "gvt"))
        for ag in AGD:
            tr_active.add((ag, "row"))
        for a in AG:
            for b in AG:
                if (a, b) not in tr_active:
                    m.TR[a, b].fix(_bench("TRO", a, b))
        # quantity/price matrix vars: fix cells outside the make/trade masks
        _fix_outside = [
            (m.XS, [(j, i) for j in J for i in I], XSact),
            (m.DS, [(j, i) for j in J for i in I], DSact),
            (m.EX, [(j, i) for j in J for i in I], EXact),
            (m.P,  [(j, i) for j in J for i in I], XSact),   # P defined only where sector produces i
            (m.DI, [(i, j) for i in I for j in J], None),    # DI dense (all i,j via aij) — keep
            (m.IM, [i for i in I], IMact),
            (m.EXD, [i for i in I], EXDact),
        ]
        # factor vars (R/RTI/TIK on capital-mask KDact; WTI/TIW on labor-mask LDact) and
        # trade prices (PD/DD on DDact; PM/PE/PE_FOB/EXD/TIM/TIX on IM/EXD masks) — fix the
        # cells whose defining equation wasn't instantiated by its $-mask.
        _fix_outside += [
            (m.R,   [(k, j) for k in K for j in J], KDact),
            (m.RTI, [(k, j) for k in K for j in J], KDact),
            (m.TIK, [(k, j) for k in K for j in J], KDact),
            # WTI now has EQ70 over ALL (l,j) — no cells to fix. TIW: EQ37 is LDact-masked.
            (m.TIW, [(l, j) for l in L for j in J], LDact),
            (m.PD,  [i for i in I], DDact),
            (m.DD,  [i for i in I], DDact),
            (m.PM,  [i for i in I], IMact),
            (m.TIM, [i for i in I], IMact),
            (m.PE,  [i for i in I], EXDact),
            (m.PE_FOB, [i for i in I], EXDact),
            (m.TIX, [i for i in I], EXDact),
            (m.PL,  [i for i in I], DDact),
            # factor DEMAND cells: KD/LD are dense but endogenous only on their masks
            (m.KD,  [(k, j) for k in K for j in J], KDact),
            (m.LD,  [(l, j) for l in L for j in J], LDact),
        ]
        # composite factor vars for sectors with NO capital (KDCO==0, e.g. admin):
        # KDC and RC are exogenous zeros there (GAMS masks EQ7/EQ8 by KDCO).
        _cap_sectors = [j for j in J if _active(P, "KDCO", j)]
        for j in J:
            if j not in _cap_sectors:
                try:
                    m.KDC[j].fix(0.0)
                except Exception:
                    pass
                try:
                    m.RC[j].fix(_bench("RCO", j) or 1.0)
                except Exception:
                    pass
        # CRITICAL (from the GAMS PEP reference): structurally-zero cells fix QUANTITIES
        # at 0 but PRICES at 1 — `pcomp.fx$(xcomp0=0)=1; xcomp.fx$(xcomp0=0)=0`. Fixing a
        # price at 0 makes ratio/division terms blow up and PATH diverges (7.7e15). So the
        # fill value depends on whether the var is a price.
        _PRICE_VARS = {"PD", "PM", "PE", "PE_FOB", "PL", "P", "PC", "PT", "PP", "PVA",
                       "PCI", "W", "RK", "R", "WC", "RC", "WTI", "RTI"}
        for var, allkeys, active in _fix_outside:
            if active is None:
                continue
            act = set(active)
            fill = 1.0 if var.name in _PRICE_VARS else 0.0
            for k in allkeys:
                if k not in act:
                    try:
                        var[k].fix(fill)
                    except Exception:
                        pass

    # objective / closure
    # MCP is a pure complementarity system — NO objective (the bridge/PATH pairs eqs↔vars).
    # NLP forms carry a constant/dummy objective so IPOPT solves the feasibility system.
    if form != "mcp":
        if variant == "objdef":
            m.OBJDEF = Constraint(expr=m.OBJ == 0.0)
            m.OBJECTIVE = Objective(expr=m.OBJ, sense=minimize)
        else:
            m.OBJECTIVE = Objective(expr=0.0, sense=minimize)  # pure feasibility
    return m
