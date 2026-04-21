$ontext
GTAP Synchronized Model - Matches Python exactly
Simple structure for parity testing
$offtext

Set r / USA, EUR /;
Set i / agr, mfg /;
Set a / agr, mfg /;
Set f / lab, cap /;

* Variables
Variable xp(r,a), x(r,a,i), px(r,a), pp(r,a);
Variable xs(r,i), ps(r,i), pd(r,i);
Variable xa(r,i), pa(r,i), xd(r,i), xmt(r,i), pmt(r,i);
Variable xet(r,i), pet(r,i);
Variable xf(r,f,a), xft(r,f), pf(r,f,a), pft(r,f);
Variable xc(r,i), xg(r,i), xi(r,i);
Variable regy(r), yc(r), yg(r);
Variable obj;

* Initialize at 1.0 (benchmark)
xp.l(r,a) = 1; x.l(r,a,i) = 1; px.l(r,a) = 1; pp.l(r,a) = 1;
xs.l(r,i) = 1; ps.l(r,i) = 1; pd.l(r,i) = 1;
xa.l(r,i) = 1; pa.l(r,i) = 1; xd.l(r,i) = 0.5; xmt.l(r,i) = 0.5; pmt.l(r,i) = 1;
xet.l(r,i) = 0.3; pet.l(r,i) = 1;
xf.l(r,f,a) = 1; xft.l(r,f) = 1; pf.l(r,f,a) = 1; pft.l(r,f) = 1;
xc.l(r,i) = 0.5; xg.l(r,i) = 0.2; xi.l(r,i) = 0.3;
regy.l(r) = 1; yc.l(r) = 0.6; yg.l(r) = 0.3;

* Synchronized equations - IDENTICAL to Python

* Production
equation eq_xp(r,a); eq_xp(r,a).. xp(r,a) =e= 1;
equation eq_x(r,a,i); eq_x(r,a,i).. x(r,a,i) =e= xp(r,a);
equation eq_px(r,a); eq_px(r,a).. px(r,a) =e= pp(r,a);

* Supply
equation eq_xs(r,i); eq_xs(r,i).. xs(r,i) =e= sum(a, x(r,a,i));
equation eq_ps(r,i); eq_ps(r,i).. ps(r,i) =e= pd(r,i);

* Armington
equation eq_xa(r,i); eq_xa(r,i).. xa(r,i) =e= xd(r,i) + xmt(r,i);
equation eq_pa(r,i); eq_pa(r,i).. pa(r,i) =e= (pd(r,i) + pmt(r,i)) / 2;
equation eq_pmt(r,i); eq_pmt(r,i).. pmt(r,i) =e= 1.1;

* Trade
equation eq_xs_cet(r,i); eq_xs_cet(r,i).. xs(r,i) =e= xd(r,i) + xet(r,i);
equation eq_pe(r,i); eq_pe(r,i).. pet(r,i) =e= ps(r,i) * 0.95;

* Factors
equation eq_xft(r,f); eq_xft(r,f).. xft(r,f) =e= sum(a, xf(r,f,a));
equation eq_pf(r,f,a); eq_pf(r,f,a).. pf(r,f,a) =e= pft(r,f);
equation eq_pft(r,f); eq_pft(r,f).. pft(r,f) =e= sum(a, pf(r,f,a)) / card(a);

* Demand
equation eq_xc(r,i); eq_xc(r,i).. xc(r,i) =e= 0.5;
equation eq_xg(r,i); eq_xg(r,i).. xg(r,i) =e= 0.2;
equation eq_xi(r,i); eq_xi(r,i).. xi(r,i) =e= 0.3;

* Income
equation eq_regy(r); eq_regy(r).. regy(r) =e= sum((f,a), pf(r,f,a) * xf(r,f,a));
equation eq_yc(r); eq_yc(r).. yc(r) =e= regy(r) * 0.6;
equation eq_yg(r); eq_yg(r).. yg(r) =e= regy(r) * 0.3;

* Market clearing
equation mkt_goods(r,i); mkt_goods(r,i).. xa(r,i) =e= xc(r,i) + xg(r,i) + xi(r,i);

* Objective
equation eq_obj; eq_obj.. obj =e= 1;

Model gtap / all /;
Solve gtap using NLP minimizing obj;

* Export
Execute_unload "gams_results.gdx", xp, x, px, pp, xs, ps, pd, xa, pa, xd, xmt, pmt,
    xet, pet, xf, xft, pf, pft, xc, xg, xi, regy, yc, yg;
