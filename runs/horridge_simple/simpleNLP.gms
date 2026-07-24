$Title                   Simple Model
$OFFSYMLIST
$OFFSYMXREF
$EOLCOM !

* Convention: variables (and initial values of variables) are in lowercase,
* other parameters are in upper case

*======================== (1) Set Declaration ============================
* Declare and read Sets and Parameters stored in GDX
Set     ! on data file
 USR      Users
 FAC      Factors
 LOCUSR(USR) Local users
 COM(LOCUSR) Commodities
 IND(COM) Industries
 LOCFIN(LOCUSR) Local final users
 SEC(IND)      Sectors
 SRC      Sources / dom, imp / ;
Parameter    ! on data file
 USE(COM,SRC,USR) Value of inputs
 FACTOR(FAC,IND)  Primary factor costs
 SIGMA(COM)       CES dom-imp elasticity   ;
$GDXIN input.gdx
$LOADdc USR, FAC,  LOCUSR, COM, IND, LOCFIN, SEC, FACTOR, USE, SIGMA

Set     ! Additional Sets
 FIN(USR)       Final users         / Hou, Gov, Inv, Exp /
 EXP(USR)       Exports             / Exp / ;

Alias (IND,i),(COM,c),(SRC,s),(USR,u),(FAC,f),(f,ff),(LOCFIN,lf),(LOCUSR,lu),(FIN,fu);

*--- alter data to avoid problems
* USE(c,s,u) $(USE(c,s,u) eq 0) = 1E-9;  ! avoid zero problems
 FACTOR(f,i)$(FACTOR(f,i)eq 0) = 1E-9;
 SIGMA(c) $(SIGMA(c)eq 1)  = 1.0001;   !NOTE: CES sigma must not equal 1

Parameter   ! addups of base data
 VALADD0(i)       Factor costs
 COSTS0(i)        Costs
 SALES0(c,s)      Sales
 DIFF(i)          Costs - sales ;

 VALADD0(i)   = SUM[f,FACTOR(f,i)];
 COSTS0(i)    = VALADD0(i) + SUM((c,s), USE(c,s,i));
 SALES0(c,s)  = SUM(u, USE(c,s,u));
 DIFF(i)      = SALES0(i,"dom")- COSTS0(i);

DISPLAY VALADD0, COSTS0, SALES0, DIFF, FACTOR;
ABORT$(ABS{SUM[i, DIFF(i)]} GT 0.01)"!!!DATA BALANCE PROBLEM!!!";
*======================= (2) Parameter declarations ======================
Parameter   ! initial values of variables (and variables) are in lower case
 ffac0(f,i)       Factor wage shift
 pfac0(f,i)       Factor prices
 pfac_f0(i)       Factor composite prices
 xfac0(f,i)       Factor use
 xfac_i0(f)       Total value-weighted factor use
 ffac_i0(f)       Factor wage shift
 xfac_f0(i)       Total factor use by industry
 z0(i)            Industry outputs
 x0(c,s,lu)       Value of inputs
 wtot0(lf)        Nominal expenditure by local final users
 xtot0(lf)        Real expenditure by local final users
 xdem0(c,s)       Total demand for goods
 xcomp0(c,lu)     Quant: final dom-imp composites
 ptot0(lf)        Price indices for local final users
 p0(c,s)          Goods prices
 pcomp0(c,lu)     Price: Intermediate dom-imp composites for local users
 wgdpinc0         Nominal GDP from income side
 wgdpexp0         Nominal GDP from expenditure side
 delB0            Trade balance
 phi0             Exchange rate
 pfimp0(c)        Imported goods prices-foreign $
 xexp0(c)         Export demand
 ftot0(lf)        Absorption parameter
 afac0(f,i)       Factor shifter
 fqexp0(c)        Initial value of export shifter;

*-- Assign initial values
 pcomp0(c,lu) = 1;
 p0(c,s)      = 1;
 pfac_f0(i)   = 1;
 pfac0(f,i)   = 1;
 pfimp0(c)    = 1;
 ptot0(lf)    = 1;
 ffac0(f,i)   = 1;
 ffac_i0(f)   = 1;
 afac0(f,i)   = 1;
 phi0         = 1;

*use(c,s,lu)$(not round(use(c,s,lu),7)) = 0;

*--- Volumes calculations
 x0(c,s,lu)     = USE(c,s,lu)/p0(c,s);
 xcomp0(c,lu)   = SUM[s,USE(c,s,lu)]/pcomp0(c,lu);
 xexp0(c)       = USE(c,"dom","exp")/p0(c,"dom");



*--- Initial Data for use in calibration
 xfac0(f,i)  = FACTOR(f,i)/pfac0(f,i);
 xfac_f0(i)  = VALADD0(i)/pfac_f0(i);
 xfac_i0(f)  = SUM[i, FACTOR(f,i)/pfac0(f,i)];
 z0(i)       = COSTS0(i)/p0(i,"dom");
 wtot0(lf)   = SUM[(c,s), USE(c,s,lf)];
 xtot0(lf)   = wtot0(lf)/ptot0(lf);
 xdem0(c,s)  = SUM[lu, x0(c,s,lu)];
 xdem0(c,"dom") = xdem0(c,"dom") + xexp0(c);

*--- Other initial values
 fqexp0(c)   = xexp0(c)*[p0(c,"dom")/phi0]**(5) ;
 wgdpinc0    = SUM{i, SUM[f,FACTOR(f,i)]};
 delB0       = SUM[c, USE(c,"dom","exp")] - SUM[(c,u), USE(c,"imp",u)];
 wgdpexp0    = SUM[(c,s,lf), USE(c,s,lf)] + delB0;
 delB0       = delB0 / wgdpinc0;
 ftot0(lf)   = wtot0(lf)/wgdpinc0;

*======================== (3) Calibration  ===============================
Parameter   ! Calibrate CES dom-imp
 A_ARM(c,s,lu) Armington shares ;
 A_ARM(c,s,lu)$xcomp0(c,lu)  = x0(c,s,lu)/xcomp0(c,lu)
                                *[pcomp0(c,lu)/p0(c,s)]**SIGMA(c);

Parameter   ! Calibrate CES primary factor
 SIGMA_FAC(i)     Elasticity in factor use
 ALPHA_F(f,i)     Share of factor in CES value added
 RHO_F(i)         Rho parameter in  CES value added
 A_F(i)           Technology parameter in CES value added;
 SIGMA_FAC(i)   = 0.5;
 RHO_F(i)       = (1-SIGMA_FAC(i))/SIGMA_FAC(i);
 ALPHA_F(f,i)   = pfac0(F,I)*xfac0(f,i)**(1+RHO_F(i))/
                   SUM{ff,pfac0(ff,I)*xfac0(ff,i)**[1+RHO_F(i)]};
 A_F(i)         = xfac_f0(i)/SUM[f,ALPHA_F(f,i)*xfac0(f,i)**(-RHO_F(i))]**(-1/RHO_F(i));

Parameter   ! Calibrate Leontief coefficient
 VA_COEF(i)    Value added (Leontief) coefficient for industries
 INT_COEF(c,i) Intermediate (Leontief) coefficient for industries;
 VA_COEF(i)    = z0(i)/xfac_f0(i);
 INT_COEF(c,i) $xcomp0(c,i) = z0(i)/xcomp0(c,i);

Parameter   ! Calibrate Cobb-Douglas for final users
 ALPHA_LF(c,lf) CD Shares of local final users
 A_LF(lf)       CD Shift parameter for local final users ;
 ALPHA_LF(c,lf) = [xcomp0(c,lf)*pcomp0(c,lf)]/wtot0(lf);
 A_LF(lf)       = xtot0(lf)/{PROD[c,xcomp0(c,lf)**ALPHA_LF(c,lf)]}

*========================== (4) MODEL ====================================
Variable
 pcomp(c,lu)      Price of Composite dom-imp commodity
 p(c,s)           Goods' prices
 pfac_f(i)        Composite factor prices
 pfac(f,i)        Price of Factors
 ptot(lf)         Price indices for local final users
 phi              Exchange rate
 pfimp(i)         Import prices
 xcomp(c,lu)      Composite commodity
 x(c,s,lu)        Intermediate demand
 xfac_f(i)        Total factor use
 xtot(lf)         Total output of local final user
 xfac(f,i)        Factor demand
 xfac_i(f)        Total factor use
 xtot(lf)         Real expenditure of local final users
 xexp(c)          Export demand
 xdem(c,s)        Total demand for goods
 z(i)             Output of industry
 wtot(lf)         Nominal expenditure by local final users
 wgdpinc          Nominal GDP - income side
 wgdpexp          Nominal GDP expenditure side
 delB             Trade Balance
 afac(f,i)        Factor using technical change
 ffac_i(f)        Factor wage shifter
 ffac(f,i)        Factor wage shifter
 ftot(lf)         Absorption parameter for local final users
 fqexp(c)         Export shifter
 OBJ              NLP objective function ;

Equation   !  Dom-imp Block
 E_pcomp          Price of Composite dom-imp commodity
 E_xcomp1         Dom-imp Composite for industries (Leontief)
 E_xcomp2         Dom-imp Composite for local final users (Cobb-Douglas)
 E_x              Intermediate demand ;

 E_pcomp(c,lu)$xcomp0(c,lu)..  pcomp(c,lu)*xcomp(c,lu) =e= SUM[s, x(c,s,lu)*p(c,s)];

 E_xcomp1(c,i)$xcomp0(c,i)..  xcomp(c,i) =e= z(i)/INT_COEF(c,i);

 E_xcomp2(c,lf).. xcomp(c,lf)*pcomp(c,lf) =e= ALPHA_LF(c,lf)*wtot(lf);

 E_x(c,s,lu)..  x(c,s,lu)=e= A_ARM(c,s,lu)*xcomp(c,lu)*[pcomp(c,lu)/p(c,s)]**SIGMA(c);

Equation   ! Industry demands
 E_pA             Industry cost indices
 E_pfac_f         Composite factor prices
 E_xfac           Demand for factors
 E_xfac_f         Composite factor CES  ;

 E_pA(i,"dom")..  p(i,"dom")*z(i) =e= SUM[c,xcomp(c,i)*pcomp(c,i)] + xfac_f(i)*pfac_f(i);

 E_pfac_f(i)..    pfac_f(i)*xfac_f(i) =e= SUM[f, xfac(f,i)*pfac(f,i)];

 E_xfac(f,i)..    xfac(f,i)/xfac_f(i) =e= [afac(f,i)*ALPHA_F(f,i)*pfac_f(i)/pfac(f,i)]
                      **SIGMA_FAC(i)*A_F(i)**(SIGMA_FAC(i)-1);

 E_xfac_f(i)..    xfac_f(i)*VA_COEF(i) =e= z(i);

Equation   ! Final demanders
 E_wtot           Nominal expenditure by local final users
 E_ptot           Price indices for local final users
 E_xtot           Real expenditure by local final users
 E_xexp           Export demand  ;

 E_wtot(lf)..     wtot(lf) =e= ftot(lf)*wgdpinc;

 E_ptot(lf)..     ptot(lf) =e= [1/A_LF(lf)]*
                PROD{c$ALPHA_LF(c,lf),[pcomp(c,lf)/ALPHA_LF(c,lf)]
                                **ALPHA_LF(c,lf)};

 E_xtot(lf)..     xtot(lf)*ptot(lf) =e= wtot(lf);

 E_xexp(c)..      xexp(c) =e= fqexp(c)*[p(c,"dom")/phi]**(-5) ;

Equation   !  Total demand and market clearing
 E_ffac_i         Total factor use
 E_xdemA          Total demand for dom goods
 E_xdemB          Total demand for imp goods
 E_z(i)           Market clearing ;

 E_ffac_i(f)..    xfac_i(f) =e= SUM[i, xfac(f,i)];

 E_xdemA(c)..     xdem(c,"dom") =e= SUM[lu, x(c,"dom",lu)] + xexp(c);

 E_xdemB(c)..     xdem(c,"imp") =e= SUM[lu, x(c,"imp",lu)];

 E_z(i)..         z(i) =e= xdem(i,"dom");

Equation   ! Miscellaneous equations
 E_pB(i)          Import prices
 E_pfac(f,i)      Factor remuneration
 E_wgdpinc        Nominal GDP income
 E_wgdpexp        Nominal GDP expenditure
 E_delB           Trade Balance
 E_OBJ            NLP objective function      ;

 E_pB(i)..        p(i,"imp") =e= pfimp(i)*phi ;

 E_pfac(f,i)..    pfac(f,i) =e= ffac_I(f)*ffac(f,i)* ptot("hou");

 E_wgdpinc..      wgdpinc =e= SUM{i, SUM[f,xfac(f,i)*pfac(f,i)]};

 E_wgdpexp..      wgdpexp =e= SUM[(c,lf)$xcomp0(c,lf), xcomp(c,lf)*pcomp(c,lf)]
                           + SUM[c, xexp(c)*p(c,"dom") - xdem(c,"imp")*p(c,"imp")];

 E_delB..         delB*wgdpinc =e= SUM[c, xexp(c)*p(c,"dom") - xdem(c,"imp")*p(c,"imp")];

 E_OBJ..          OBJ =e= 1;

*--- Initialize levels variables
 xfac_f.l(i)      = xfac_f0(i);
 xcomp.l(c,lu)    = xcomp0(c,lu);
 x.l(c,s,lu)      = x0(c,s,lu);
 xfac.l(f,i)      = xfac0(f,i);
 xtot.l(lf)       = xtot0(lf);
 xexp.l(c)        = xexp0(c);
 xfac_i.l(f)      = xfac_i0(f);
 xdem.l(c,s)      = xdem0(c,s);
 xtot.l(lf)       = xtot0(lf);
 z.l(i)           = z0(i);
 pcomp.l(c,lu)    = pcomp0(c,lu);
 pfac_f.l(i)      = pfac_f0(i);
 pfac.l(f,i)      = pfac0(f,i);
 p.l(c,s)         = p0(c,s);
 p.l(i,s)         = p0(i,s);
 ptot.l(lf)       = ptot0(lf);
 wtot.l(lf)       = wtot0(lf);
 wgdpinc.l        = wgdpinc0;
 wgdpexp.l        = wgdpexp0;
 delB.l           = delB0;
 pfimp.l(i)       = pfimp0(i);
 phi.l            = phi0;
 fqexp.l(c)       = fqexp0(c);
 ftot.l(lf)       = ftot0(lf);
 afac.l(f,i)      = afac0(f,i);
 ffac_I.l(f)      = ffac_I0(f);
 ffac.l(f,i)      = ffac0(f,i);
 OBJ.l            = 1;

model simple /all/;
*======================= (5) Simulation Closure and Shocks ================
* Exogenous Variables in standard closure
 afac.fx(f,i)     = afac0(f,i);
 ffac.fx(f,i)     = ffac0(f,i);
 fqexp.fx(c)      = xexp0(c);
 ftot.fx(lf)      = ftot0(lf);
 pfimp.fx(i)      = pfimp0(i);
 phi.fx           = phi0;
 xfac_i.fx(f)     = xfac_i0(f);

pcomp.fx(c,lu)$(xcomp0(c,lu)=0) = 1;
xcomp.fx(c,lu)$(xcomp0(c,lu)=0) = 0;

option sysout=on;

simple.holdfixed=1;     ! assist check that n.equ = n.endogenous var
simple.iterlim=0;       ! dummy simulation to test calibration
simple.tolinfrep=1E-8;
* solve simple using mcp; ! would abort if initial values not a solution
 solve simple maximizing obj using NLP;

*--- Closure changes/Swaps
 xfac.fx("capital",i) = xfac0("capital",i);   ! fix industry capital stocks
 ffac.lo("capital",i) = -INF;
 ffac.up("capital",i) = +INF;
 ffac.l("capital",i)  = ffac0("capital",i);

 ffac_I.fx("capital") = ffac_I0("capital");   ! unfix total capital stock
 xfac_i.lo("capital") = -INF;
 xfac_i.up("capital") = +INF;
 xfac_i.l("capital")  = xfac_i0("capital");

*--- SHOCKS
afac.fx("labor","srv") = 0.90;   !Improvement in Service factor productivity

*--- Solve model
simple.reslim=19000;
simple.iterlim=200;
simple.holdfixed=1;
*solve simple using MCP;
 solve simple maximizing obj using NLP;

*======================= (6) Simulation Results ==========================
Parameter
 LGDP             Laspeyeres GDP quantity index
 PGDP             Paasche GDP quantity index
 CH_XGDPINC       Percentage change in Fisher (ideal) GDP quantity index
 CH_Z(i)          Percentage change in output of industries
 CH_X(c,s,lu)     Percentage change in intermediate demand
 CH_XFAC(f,i)     Percentage change in factor demand
 CH_PFAC(f,i)     Percentage change in price of factors
 CH_PFAC_F(i)     Percentage change in composite price of factors
 CH_P(c,s)        Percentage change in basic prices
 CH_WGDPINC       Percentage change in nominal GDP income side
 CH_WGDPEXP       Percentage change in nominal GDP expenditure side
 CH_DELB          Ordinary change in ratio nominal trade balance to GDP;

 LGDP = SUM[i,xfac_f.l(i)*pfac_f0(i)]  / SUM[i,xfac_f0(i)*pfac_f0(i)];
 PGDP = SUM[i,xfac_f.l(i)*pfac_f.l(i)] / SUM[i,xfac_f0(i)*pfac_f.l(i)];
 CH_XGDPINC = [SQRT(LGDP*PGDP)-1]*100;
 CH_Z(i)                  = (z.l(i)/z0(i)-1)*100;
 CH_X(c,s,lu) $x0(c,s,lu) = (x.l(c,s,lu)/x0(c,s,lu)-1)*100;
 CH_XFAC(f,i) $xfac0(f,i) = (xfac.l(f,i)/xfac0(f,i)-1)*100;
 CH_PFAC(f,i) $pfac0(f,i) = (pfac.l(f,i)/pfac0(f,i)-1)*100;
 CH_PFAC_F(i) $pfac_f0(i) = (pfac_f.l(i)/pfac_f0(i)-1)*100;
 CH_P(c,s)                = (p.l(c,s)/p0(c,s)-1)*100;
 CH_WGDPINC               = (wgdpinc.l/wgdpinc0-1)*100;
 CH_WGDPEXP               = (wgdpexp.l/wgdpexp0-1)*100;
 CH_DELB                  = (delB.l-delB0);

Display CH_Z,CH_XFAC,CH_PFAC,CH_PFAC_F,CH_P,CH_WGDPINC,CH_XGDPINC,CH_WGDPEXP,CH_DELB;

*---Export results to GDX
EXECUTE_UNLOAD 'RESULTS', CH_Z,CH_XFAC, CH_PFAC, CH_PFAC_F,CH_P, CH_WGDPINC,
                          CH_XGDPINC, CH_WGDPEXP,  CH_DELB;
