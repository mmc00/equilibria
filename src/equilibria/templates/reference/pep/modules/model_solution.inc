*==============================================================================*
*                            MODEL SOLUTION MODULE                            *
*==============================================================================*
*                                                                              *
*  This module contains all model solution operations for the PEP CGE model   *
*  including variable initialization, closure rules, simulation scenarios,    *
*  and model solution commands                                                 *
*                                                                              *
*==============================================================================*

* 6 Numerical resolution

** 6.1 Variable initialisation
**  6.1.1 Volume variables
 C.l(i,h)        = CO(i,h);
 CG.l(i)         = CGO(i);
 CI.l(j)         = CIO(j);
 CMIN.l(i,h)     = CMINO(i,h);
 DD.l(i)         = DDO(i);
 DI.l(i,j)       = DIO(i,j);
 DIT.l(i)        = DITO(i);
 DS.l(j,i)       = DSO(j,i);
 EX.l(j,i)       = EXO(j,i);
 EXD.l(i)        = EXDO(i);
 IM.l(i)         = IMO(i);
 INV.l(i)        = INVO(i);
 KD.l(k,j)       = KDO(k,j);
 KDC.l(j)        = KDCO(j);
 KS.l(k)         = KSO(k);
 LD.l(l,j)       = LDO(l,j);
 LDC.l(j)        = LDCO(j);
 LS.l(l)         = LSO(l);
 MRGN.l(i)       = MRGNO(i);
 Q.l(i)          = QO(i);
 VA.l(j)         = VAO(j);
 VSTK.l(i)       = VSTKO(i);
 XS.l(j,i)       = XSO(j,i);
 XST.l(j)        = XSTO(j);

**  6.1.2 Price variables
 e.l             = eO;
 P.l(j,i)        = PO(j,i);
 PC.l(i)         = PCO(i);
 PCI.l(j)        = PCIO(j);
 PD.l(i)         = PDO(i);
 PE.l(i)         = PEO(i);
 PE_FOB.l(i)     = PE_FOBO(i);
 PIXCON.l        = PIXCONO;
 PIXGDP.l        = PIXGDPO;
 PIXGVT.l        = PIXGVTO;
 PIXINV.l        = PIXINVO;
 PL.l(i)         = PLO(i);
 PM.l(i)         = PMO(i);
 PP.l(j)         = PPO(j);
 PT.l(j)         = PTO(j);
 PVA.l(j)        = PVAO(j);
 PWM.l(i)        = PWMO(i);
 PWX.l(i)        = PWXO(i);
 R.l(k,j)        = RO(k,j);
 RC.l(j)         = RCO(j);
 RK.l(k)         = RKO(k);
 RTI.l(k,j)      = RTIO(k,j);
 W.l(l)          = WO(l);
 WC.l(j)         = WCO(j);
 WTI.l(l,j)      = WTIO(l,j);

**  6.1.3 Nominal (value) variables
 CAB.l           = CABO;
 CTH.l(h)        = CTHO(h);
 G.l             = GO;
 GDP_BP.l        = GDP_BPO;
 GDP_FD.l        = GDP_FDO;
 GDP_IB.l        = GDP_IBO;
 GDP_MP.l        = GDP_MPO;
 GFCF.l          = GFCFO;
 IT.l            = ITO;
 SF.l(f)         = SFO(f);
 SG.l            = SGO;
 SH.l(h)         = SHO(h);
 SROW.l          = SROWO;
 TDF.l(f)        = TDFO(f);
 TDFT.l          = TDFTO;
 TDH.l(h)        = TDHO(h);
 TDHT.l          = TDHTO;
 TIC.l(i)        = TICO(i);
 TICT.l          = TICTO;
 TIK.l(k,j)      = TIKO(k,j);
 TIKT.l          = TIKTO;
 TIM.l(i)        = TIMO(i);
 TIMT.l          = TIMTO;
 TIP.l(j)        = TIPO(j);
 TIPT.l          = TIPTO;
 TIW.l(l,j)      = TIWO(l,j);
 TIWT.l          = TIWTO;
 TIX.l(i)        = TIXO(i);
 TIXT.l          = TIXTO;
 TPRODNT.l       = TPRODNO;
 TPRCTS.l        = TPRCTSO;
 TR.l(ag,agj)    = TRO(ag,agj);
 YDF.l(f)        = YDFO(f);
 YDH.l(h)        = YDHO(h);
 YF.l(f)         = YFO(f);
 YFK.l(f)        = YFKO(f);
 YFTR.l(f)       = YFTRO(f);
 YG.l            = YGO;
 YGK.l           = YGKO;
 YGTR.l          = YGTRO;
 YH.l(h)         = YHO(h);
 YHK.l(h)        = YHKO(h);
 YHL.l(h)        = YHLO(h);
 YHTR.l(h)       = YHTRO(h);
 YROW.l          = YROWO;

** 6.1.4 Real (volume) variables computed from price indices
 CTH_REAL.l(h)   = CTH_REALO(h);
 G_REAL.l        = G_REALO;
 GDP_BP_REAL.l   = GDP_BP_REALO;
 GDP_MP_REAL.l   = GDP_MP_REALO;
 GFCF_REAL.l     = GFCF_REALO;

**  6.1.5 Rates and intercepts
 sh0.l(h)        = sh0O(h);
 sh1.l(h)        = sh1O(h);
 tr0.l(h)        = tr0O(h);
 tr1.l(h)        = tr1O(h);
 ttdf0.l(f)      = ttdf0O(f);
 ttdf1.l(f)      = ttdf1O(f);
 ttdh0.l(h)      = ttdh0O(h);
 ttdh1.l(h)      = ttdh1O(h);
 ttic.l(i)       = tticO(i);
 ttik.l(k,j)     = ttikO(k,j);
 ttim.l(i)       = ttimO(i);
 ttip.l(j)       = ttipO(j);
 ttiw.l(l,j)     = ttiwO(l,j);
 ttix.l(i)       = ttixO(i);

**  6.1.6  Other
 LEON.l          = 0;

** 6.2 Choice of mobile or sector-specific capital

*  If kmob=1, capital is mobile, if kmob=0, it is sector-specific
 kmob            = 1;
 RK.fx(k)$(kmob=1) = RKO(k);
 R.fx(k,j)$(kmob=0) = RO(k,j);

** 6.3 Closures
*  The numeraire is the nominal exchange rate
 e.fx            = eO;

*  This is the default closure: Government savings, capital supply,
*  world prices and transfers from the rest of the world are exogenous.
 SG.fx           = SGO;
 KS.fx(k)        = KSO(k);
 LS.fx(l)        = LSO(l);
 PWM.fx(i)       = PWMO(i);
 PWX.fx(i)       = PWXO(i);
 EXD.fx(i)       = EXDO(i);
 TR.fx(agd,'row') = TRO(agd,'row');
 TR.fx('row',agng) = TRO('row',agng);

*  Add possibility of fixed minimum consumption
 CMIN.fx(i,h)    = CMINO(i,h);

*  Fixed inventory changes
 VSTK.fx(i)      = VSTKO(i);

*  Transfers, direct tax rates, and household savings rates could instead
*  be endogenous if the government has a target for its budget balance, etc.

*  Additional closures such as fixed government consumption, fixed transfers
*  or endogenous direct tax rates can be activated by the user.

** 6.4 Simulations
$include "modules/simulations.inc"

** 6.5 Resolution
option limrow = 10000;
OPTION NLP = conopt3;
MODEL PEP11 Standard PEP static model version 2_1 /ALL/;
PEP11.HOLDFIXED=1;
SOLVE PEP11 USING CNS;

$include "RESULTS PEP 1-1.GMS"

* Export trade variables/prices for parity with Python comparers
$include "modules/export_trade_results.inc"

* Export government aggregates and per-commodity items for Python parity
$include "modules/export_government_results.inc"
 PIXINV.l        = PIXINVO;
 PL.l(i)         = PLO(i);
 PM.l(i)         = PMO(i);
 PP.l(j)         = PPO(j);
 PT.l(j)         = PTO(j);
 PVA.l(j)        = PVAO(j);
 PWM.l(i)        = PWMO(i);
 PWX.l(i)        = PWXO(i);
 R.l(k,j)        = RO(k,j);
 RC.l(j)         = RCO(j);
 RK.l(k)         = RKO(k);
 RTI.l(k,j)      = RTIO(k,j);
 W.l(l)          = WO(l);
 WC.l(j)         = WCO(j);
 WTI.l(l,j)      = WTIO(l,j);

**  6.1.3 Nominal (value) variables
 CAB.l           = CABO;
 CTH.l(h)        = CTHO(h);
 G.l             = GO;
 GDP_BP.l        = GDP_BPO;
 GDP_FD.l        = GDP_FDO;
 GDP_IB.l        = GDP_IBO;
 GDP_MP.l        = GDP_MPO;
 GFCF.l          = GFCFO;
 IT.l            = ITO;
 SF.l(f)         = SFO(f);
 SG.l            = SGO;
 SH.l(h)         = SHO(h);
 SROW.l          = SROWO;
 TDF.l(f)        = TDFO(f);
 TDFT.l          = TDFTO;
 TDH.l(h)        = TDHO(h);
 TDHT.l          = TDHTO;
 TIC.l(i)        = TICO(i);
 TICT.l          = TICTO;
 TIK.l(k,j)      = TIKO(k,j);
 TIKT.l          = TIKTO;
 TIM.l(i)        = TIMO(i);
 TIMT.l          = TIMTO;
 TIP.l(j)        = TIPO(j);
 TIPT.l          = TIPTO;
 TIW.l(l,j)      = TIWO(l,j);
 TIWT.l          = TIWTO;
 TIX.l(i)        = TIXO(i);
 TIXT.l          = TIXTO;
 TPRCTS.l        = TPRCTSO;
 TPRODN.l        = TPRODNO;
 TR.l(ag,agj)    = TRO(ag,agj);
 YDF.l(f)        = YDFO(f);
 YDH.l(h)        = YDHO(h);
 YF.l(f)         = YFO(f);
 YFK.l(f)        = YFKO(f);
 YFTR.l(f)       = YFTRO(f);
 YG.l            = YGO;
 YGK.l           = YGKO;
 YGTR.l          = YGTRO;
 YH.l(h)         = YHO(h);
 YHK.l(h)        = YHKO(h);
 YHL.l(h)        = YHLO(h);
 YHTR.l(h)       = YHTRO(h);
 YROW.l          = YROWO;

** 6.1.4 Real (volume) variables computed from price indices
 CTH_REAL.l(h)   = CTH_REALO(h);
 G_REAL.l        = G_REALO;
 GDP_BP_REAL.l   = GDP_BP_REALO;
 GDP_MP_REAL.l   = GDP_MP_REALO;
 GFCF_REAL.l     = GFCF_REALO;

**  6.1.5 Rates and intercepts
 sh0.l(h)        = sh0O(h);
 sh1.l(h)        = sh1O(h);
 tr0.l(h)        = tr0O(h);
 tr1.l(h)        = tr1O(h);
 ttdf0.l(f)      = ttdf0O(f);
 ttdf1.l(f)      = ttdf1O(f);
 ttdh0.l(h)      = ttdh0O(h);
 ttdh1.l(h)      = ttdh1O(h);
 ttic.l(i)       = tticO(i);
 ttik.l(k,j)     = ttikO(k,j);
 ttim.l(i)       = ttimO(i);
 ttip.l(j)       = ttipO(j);
 ttiw.l(l,j)     = ttiwO(l,j);
 ttix.l(i)       = ttixO(i);

**  6.1.6  Other variables
 LEON.l          = 0;

** 6.2 Choice of mobile or sector-specific capital

*  If kmob=1, capital is mobile, if kmob=0, it is sector-specific
 kmob            = 1;
 KD.fx(k,j)$(kmob eq 0)
                 = KDO(k,j);
 KS.fx(k)$kmob   = KSO(k);

** 6.3 Closures
*  The numeraire is the nominal exchange rate

*** 6.3.1 Exogenous variables (fixed values)
 e.fx            = 1;
 CAB.fx          = CABO;
 CMIN.fx(i,h)    = CMINO(i,h);
 G.fx            = GO;
 LS.fx(l)        = LSO(l);
 PWM.fx(i)       = PWMO(i);
 PWX.fx(i)       = PWXO(i);
 VSTK.fx(i)      = VSTKO(i);

*** 6.3.2 Policy parameters (fixed at benchmark values)
 sh0.fx(h)       = sh0O(h);
 sh1.fx(h)       = sh1O(h);
 tr0.fx(h)       = tr0O(h);
 tr1.fx(h)       = tr1O(h);
 ttdf0.fx(f)     = ttdf0O(f);
 ttdf1.fx(f)     = ttdf1O(f);
 ttdh0.fx(h)     = ttdh0O(h);
 ttdh1.fx(h)     = ttdh1O(h);
 ttic.fx(i)      = tticO(i);
 ttik.fx(k,j)    = ttikO(k,j);
 ttim.fx(i)      = ttimO(i);
 ttip.fx(j)      = ttipO(j);
 ttiw.fx(l,j)    = ttiwO(l,j);
 ttix.fx(i)      = ttixO(i);

** 6.4 Simulation scenarios
*  This section is modularized. See modules/simulations.inc for policy shocks.
$include "modules/simulations.inc"

** 6.5 Model solution

*** 6.5.1 Solver options
option limrow = 10000;
OPTION NLP = conopt3;

*** 6.5.2 Model definition and solution
MODEL PEP11 Standard PEP static model version 2_1 /ALL/;
PEP11.HOLDFIXED = 1;
SOLVE PEP11 USING CNS;

*** 6.5.3 Results reporting
$include "RESULTS PEP 1-1.GMS"

* Export trade variables/prices for parity with Python compare tools
$include "modules/export_trade_results.inc"
