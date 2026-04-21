$ontext
Full GTAP Model - Multi-region CGE
$offtext

* ========================================================================
* SETS
* ========================================================================
Set r "Regions" / USA, EUR, CHN /;
Set i "Commodities" / agr, mfg, ser /;
Set a "Activities" / agr, mfg, ser /;
Set f "Factors" / lab, cap /;

Alias (r, rp), (i, ip), (a, ap);

* ========================================================================
* VARIABLES - Full GTAP structure
* ========================================================================

* Production
Variable xp(r,a) "Production activity level";
Variable x(r,a,i) "Output by commodity";
Variable px(r,a) "Unit cost of production";
Variable pp(r,a) "Producer price";

* Supply  
Variable xs(r,i) "Domestic supply";
Variable ps(r,i) "Price of domestic supply";
Variable pd(r,i) "Price of domestic goods";

* Trade - Imports
Variable xmt(r,i) "Aggregate import demand";
Variable xw(r,i,rp) "Bilateral imports";
Variable pmt(r,i) "Aggregate import price";
Variable pmcif(r,i,rp) "CIF import price";

* Trade - Exports
Variable xet(r,i) "Aggregate export supply";
Variable xe(r,i,rp) "Bilateral exports";
Variable pet(r,i) "Aggregate export price";
Variable pe(r,i,rp) "Bilateral export price";

* Factors
Variable xf(r,f,a) "Factor demand";
Variable xft(r,f) "Aggregate factor supply";
Variable pf(r,f,a) "Factor price";
Variable pft(r,f) "Aggregate factor price";

* Final demand
Variable xc(r,i) "Private consumption";
Variable xg(r,i) "Government consumption";
Variable xi(r,i) "Investment demand";
Variable pa(r,i) "Armington price";

* Income
Variable regy(r) "Regional income";
Variable yc(r) "Private expenditure";
Variable yg(r) "Government expenditure";
Variable yi(r) "Investment expenditure";

* Price indices
Variable pnum "Numeraire";

* ========================================================================
* INITIALIZATION (benchmark = 1.0)
* ========================================================================
xp.l(r,a) = 1;
x.l(r,a,i) = 1;
px.l(r,a) = 1;
pp.l(r,a) = 1;
xs.l(r,i) = 1;
ps.l(r,i) = 1;
pd.l(r,i) = 1;
xmt.l(r,i) = 1;
xw.l(r,i,rp) = 1;
pmt.l(r,i) = 1;
pmcif.l(r,i,rp) = 1;
xet.l(r,i) = 1;
xe.l(r,i,rp) = 1;
pet.l(r,i) = 1;
pe.l(r,i,rp) = 1;
xf.l(r,f,a) = 1;
xft.l(r,f) = 1;
pf.l(r,f,a) = 1;
pft.l(r,f) = 1;
xc.l(r,i) = 1;
xg.l(r,i) = 1;
xi.l(r,i) = 1;
pa.l(r,i) = 1;
regy.l(r) = 100;
yc.l(r) = 50;
yg.l(r) = 30;
pnum.l = 1;

* ========================================================================
* EQUATIONS - Square system (one equation per variable)
* ========================================================================

* Production block - 4 eqns
equation prf_y(r,a) "Zero profit production";
prf_y(r,a).. px(r,a) =e= pp(r,a);

equation eq_x(r,a,i) "Output allocation";
eq_x(r,a,i).. x(r,a,i) =e= xp(r,a);

* Supply - 2 eqns  
equation eq_xs(r,i) "Domestic supply";
eq_xs(r,i).. xs(r,i) =e= sum(a, x(r,a,i));

equation eq_ps(r,i) "Supply price";
eq_ps(r,i).. ps(r,i) =e= pd(r,i);

* Armington - 1 eqn
equation eq_pa(r,i) "Armington price";
eq_pa(r,i).. pa(r,i) =e= (pd(r,i) + pmt(r,i)) / 2;

* Trade - 4 eqns
equation eq_xmt(r,i) "Import demand";
eq_xmt(r,i).. xmt(r,i) =e= sum(rp, xw(r,i,rp));

equation eq_pmt(r,i) "Import price aggregation";
eq_pmt(r,i).. pmt(r,i) =e= sum(rp, pmcif(r,i,rp)) / card(rp);

equation eq_xet(r,i) "Export supply";
eq_xet(r,i).. xet(r,i) =e= sum(rp, xe(r,i,rp));

equation eq_pet(r,i) "Export price";
eq_pet(r,i).. pet(r,i) =e= sum(rp, pe(r,i,rp)) / card(rp);

* Factors - 3 eqns
equation eq_xft(r,f) "Factor market clearing";
eq_xft(r,f).. xft(r,f) =e= sum(a, xf(r,f,a));

equation eq_pf(r,f,a) "Factor price";
eq_pf(r,f,a).. pf(r,f,a) =e= pft(r,f);

equation eq_pft(r,f) "Aggregate factor price";
eq_pft(r,f).. pft(r,f) * sum(a, xf(r,f,a)) =e= sum(a, pf(r,f,a) * xf(r,f,a));

* Final demand - 3 eqns
equation eq_xc(r,i) "Private consumption";
eq_xc(r,i).. xc(r,i) =e= 1;

equation eq_xg(r,i) "Government consumption";
eq_xg(r,i).. xg(r,i) =e= 0.5;

equation eq_xi(r,i) "Investment demand";
eq_xi(r,i).. xi(r,i) =e= 0.3;

* Income - 4 eqns
equation eq_regy(r) "Regional income";
eq_regy(r).. regy(r) =e= sum((f,a), pf(r,f,a) * xf(r,f,a));

equation eq_yc(r) "Private income";
eq_yc(r).. yc(r) =e= regy(r) * 0.6;

equation eq_yg(r) "Government income";
eq_yg(r).. yg(r) =e= regy(r) * 0.3;

equation eq_yi(r) "Investment income";
eq_yi(r).. yi(r) =e= regy(r) * 0.1;

* Numeraire - 1 eqn
Equation eq_pnum;
eq_pnum.. pnum =e= 1;

* ========================================================================
* MODEL - Square CNS system
* ========================================================================
Model gtap /
    prf_y, eq_x, eq_xs, eq_ps,
    eq_pa, eq_xmt, eq_pmt, eq_xet, eq_pet,
    eq_xft, eq_pf, eq_pft,
    eq_xc, eq_xg, eq_xi,
    eq_regy, eq_yc, eq_yg, eq_yi,
    eq_pnum
/;

* ========================================================================
* SOLVE
* ========================================================================
* Fix numeraire to make system square
pnum.fx = 1;

* Alternative: Use NLP solver instead of CNS for non-square systems
* Solve gtap using NLP minimizing obj;

* Or add a dummy objective and use NLP
Variable obj "Dummy objective";
Equation eq_obj;
eq_obj.. obj =e= 1;

Model gtap_nlp / gtap, eq_obj /;
Solve gtap_nlp using NLP minimizing obj;

* ========================================================================
* EXPORT RESULTS
* ========================================================================
Parameter xp_out(r,a), x_out(r,a,i), px_out(r,a), pp_out(r,a);
Parameter xs_out(r,i), ps_out(r,i), pd_out(r,i);
Parameter xmt_out(r,i), xw_out(r,i,rp), pmt_out(r,i);
Parameter xet_out(r,i), xe_out(r,i,rp), pet_out(r,i);
Parameter xf_out(r,f,a), xft_out(r,f), pf_out(r,f,a), pft_out(r,f);
Parameter xc_out(r,i), xg_out(r,i), xi_out(r,i), pa_out(r,i);
Parameter regy_out(r), yc_out(r), yg_out(r), yi_out(r);

xp_out(r,a) = xp.l(r,a);
x_out(r,a,i) = x.l(r,a,i);
px_out(r,a) = px.l(r,a);
pp_out(r,a) = pp.l(r,a);
xs_out(r,i) = xs.l(r,i);
ps_out(r,i) = ps.l(r,i);
pd_out(r,i) = pd.l(r,i);
xmt_out(r,i) = xmt.l(r,i);
xw_out(r,i,rp) = xw.l(r,i,rp);
pmt_out(r,i) = pmt.l(r,i);
xet_out(r,i) = xet.l(r,i);
xe_out(r,i,rp) = xe.l(r,i,rp);
pet_out(r,i) = pet.l(r,i);
xf_out(r,f,a) = xf.l(r,f,a);
xft_out(r,f) = xft.l(r,f);
pf_out(r,f,a) = pf.l(r,f,a);
pft_out(r,f) = pft.l(r,f);
xc_out(r,i) = xc.l(r,i);
xg_out(r,i) = xg.l(r,i);
xi_out(r,i) = xi.l(r,i);
pa_out(r,i) = pa.l(r,i);
regy_out(r) = regy.l(r);
yc_out(r) = yc.l(r);
yg_out(r) = yg.l(r);
yi_out(r) = 20;

Execute_unload "gtap_results.gdx",
    xp_out, x_out, px_out, pp_out,
    xs_out, ps_out, pd_out,
    xmt_out, xw_out, pmt_out,
    xet_out, xe_out, pet_out,
    xf_out, xft_out, pf_out, pft_out,
    xc_out, xg_out, xi_out, pa_out,
    regy_out, yc_out, yg_out, yi_out;
