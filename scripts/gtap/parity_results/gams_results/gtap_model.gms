$ontext
GTAP Model for Parity Testing
3 regions, 3 commodities, 2 factors
$offtext

* Sets
Set r "Regions" / USA, EUR, CHN /;
Set i "Commodities" / agr, mfg, ser /;
Set a "Activities" / agr, mfg, ser /;
Set f "Factors" / lab, cap /;

* Variables - Production
Variable xp(r,a) "Production activity";
Variable x(r,a,i) "Output by commodity";
Variable px(r,a) "Unit cost";
Variable pp(r,a) "Producer price";

* Variables - Supply
Variable xs(r,i) "Domestic supply";
Variable ps(r,i) "Supply price";
Variable pd(r,i) "Domestic price";

* Variables - Armington
Variable xa(r,i) "Armington demand";
Variable pa(r,i) "Armington price";
Variable xd(r,i) "Domestic demand";
Variable xmt(r,i) "Import demand";
Variable pmt(r,i) "Import price";

* Variables - Trade
Variable xet(r,i) "Export supply";
Variable pet(r,i) "Export price";

* Variables - Factors
Variable xf(r,f,a) "Factor demand";
Variable xft(r,f) "Factor supply";
Variable pf(r,f,a) "Factor price by activity";
Variable pft(r,f) "Aggregate factor price";

* Variables - Demand
Variable xc(r,i) "Private consumption";
Variable xg(r,i) "Government consumption";
Variable xi(r,i) "Investment";

* Variables - Income
Variable regy(r) "Regional income";
Variable yc(r) "Private income";
Variable yg(r) "Government income";

* Initialize
xp.l(r,a) = 1; x.l(r,a,i) = 1; px.l(r,a) = 1; pp.l(r,a) = 1;
xs.l(r,i) = 1; ps.l(r,i) = 1; pd.l(r,i) = 1;
xa.l(r,i) = 1; pa.l(r,i) = 1; xd.l(r,i) = 0.5; xmt.l(r,i) = 0.5; pmt.l(r,i) = 1;
xet.l(r,i) = 0.3; pet.l(r,i) = 1;
xf.l(r,f,a) = 1; xft.l(r,f) = 1; pf.l(r,f,a) = 1; pft.l(r,f) = 1;
xc.l(r,i) = 0.5; xg.l(r,i) = 0.2; xi.l(r,i) = 0.3;
regy.l(r) = 100; yc.l(r) = 60; yg.l(r) = 30;

* Equations - Production
equation prf_y(r,a);
prf_y(r,a).. px(r,a) =e= pp(r,a);

equation eq_x(r,a,i);
eq_x(r,a,i).. x(r,a,i) =e= xp(r,a);

* Equations - Supply
equation eq_xs(r,i);
eq_xs(r,i).. xs(r,i) =e= sum(a, x(r,a,i));

equation eq_ps(r,i);
eq_ps(r,i).. ps(r,i) =e= pd(r,i);

* Equations - Armington
equation eq_xa(r,i);
eq_xa(r,i).. xa(r,i) =e= xd(r,i) + xmt(r,i);

equation eq_pa(r,i);
eq_pa(r,i).. pa(r,i)*(xd(r,i) + xmt(r,i) + 0.001) =e= xd(r,i)*pd(r,i) + xmt(r,i)*pmt(r,i);

equation eq_pmt(r,i);
eq_pmt(r,i).. pmt(r,i) =e= 1.1;

* Equations - Trade (CET)
equation eq_xs_cet(r,i);
eq_xs_cet(r,i).. xs(r,i) =e= xd(r,i) + xet(r,i);

equation eq_pe(r,i);
eq_pe(r,i).. pet(r,i) =e= ps(r,i) * 0.95;

* Equations - Factors
equation eq_xft(r,f);
eq_xft(r,f).. xft(r,f) =e= sum(a, xf(r,f,a));

equation eq_pft(r,f);
eq_pft(r,f).. pft(r,f)*(sum(a, xf(r,f,a)) + 0.001) =e= sum(a, pf(r,f,a)*xf(r,f,a));

equation eq_pf(r,f,a);
eq_pf(r,f,a).. pf(r,f,a) =e= pft(r,f);

* Equations - Demand
equation eq_xc(r,i);
eq_xc(r,i).. xc(r,i) =e= 0.5;

equation eq_xg(r,i);
eq_xg(r,i).. xg(r,i) =e= 0.2;

equation eq_xi(r,i);
eq_xi(r,i).. xi(r,i) =e= 0.3;

* Equations - Income
equation eq_regy(r);
eq_regy(r).. regy(r) =e= sum((f,a), pf(r,f,a)*xf(r,f,a));

equation eq_yc(r);
eq_yc(r).. yc(r) =e= regy(r) * 0.6;

equation eq_yg(r);
eq_yg(r).. yg(r) =e= regy(r) * 0.3;

* Equations - Market Clearing
equation mkt_goods(r,i);
mkt_goods(r,i).. xa(r,i) =e= xc(r,i) + xg(r,i) + xi(r,i);

* Model
Model gtap / all /;

* Add dummy objective for NLP
Variable obj "Dummy objective";
Equation eq_obj;
eq_obj.. obj =e= 1;

* Solve
Model gtap_nlp / gtap, eq_obj /;
Solve gtap_nlp using NLP minimizing obj;

* Export results
Parameter xp_out(r,a), x_out(r,a,i), px_out(r,a), pp_out(r,a);
Parameter xs_out(r,i), ps_out(r,i), pd_out(r,i);
Parameter xa_out(r,i), pa_out(r,i), xd_out(r,i), xmt_out(r,i), pmt_out(r,i);
Parameter xet_out(r,i), pet_out(r,i);
Parameter xf_out(r,f,a), xft_out(r,f), pf_out(r,f,a), pft_out(r,f);
Parameter xc_out(r,i), xg_out(r,i), xi_out(r,i);
Parameter regy_out(r), yc_out(r), yg_out(r);

xp_out(r,a) = xp.l(r,a); x_out(r,a,i) = x.l(r,a,i); px_out(r,a) = px.l(r,a); pp_out(r,a) = pp.l(r,a);
xs_out(r,i) = xs.l(r,i); ps_out(r,i) = ps.l(r,i); pd_out(r,i) = pd.l(r,i);
xa_out(r,i) = xa.l(r,i); pa_out(r,i) = pa.l(r,i); xd_out(r,i) = xd.l(r,i); xmt_out(r,i) = xmt.l(r,i); pmt_out(r,i) = pmt.l(r,i);
xet_out(r,i) = xet.l(r,i); pet_out(r,i) = pet.l(r,i);
xf_out(r,f,a) = xf.l(r,f,a); xft_out(r,f) = xft.l(r,f); pf_out(r,f,a) = pf.l(r,f,a); pft_out(r,f) = pft.l(r,f);
xc_out(r,i) = xc.l(r,i); xg_out(r,i) = xg.l(r,i); xi_out(r,i) = xi.l(r,i);
regy_out(r) = regy.l(r); yc_out(r) = yc.l(r); yg_out(r) = yg.l(r);

Execute_unload "gams_results.gdx",
    xp_out, x_out, px_out, pp_out,
    xs_out, ps_out, pd_out,
    xa_out, pa_out, xd_out, xmt_out, pmt_out,
    xet_out, pet_out,
    xf_out, xft_out, pf_out, pft_out,
    xc_out, xg_out, xi_out,
    regy_out, yc_out, yg_out;
