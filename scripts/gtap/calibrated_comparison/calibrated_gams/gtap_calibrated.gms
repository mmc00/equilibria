$ontext
GTAP Model - Calibrated from SAM
Benchmark equilibrium: all prices = 1, quantities from SAM
$offtext

* Sets
Set r "Regions" / USA, EUR, CHN /;
Set i "Commodities" / agr, mfg, ser /;
Set a "Activities" / agr, mfg, ser /;
Set f "Factors" / lab, cap /;

* Calibrated parameters from SAM
Parameter axp(r,a) "Production shifter";
Parameter ava(r,a) "Value-added coefficient";
Parameter aio(r,a) "Intermediate coefficient";
Parameter alphad(r,i) "Domestic share";
Parameter alpham(r,i) "Import share";
Parameter alphaf(r,f,a) "Factor share in VA";
Parameter gamma_c(r,i) "Consumption share";

* Initialize parameters from calibration
axp('USA','agr') = 1.000000;
axp('USA','mfg') = 1.000000;
axp('USA','ser') = 1.000000;
axp('EUR','agr') = 1.000000;
axp('EUR','mfg') = 1.000000;
axp('EUR','ser') = 1.000000;
axp('CHN','agr') = 1.000000;
axp('CHN','mfg') = 1.000000;
axp('CHN','ser') = 1.000000;
ava('USA','agr') = 0.600000;
ava('USA','mfg') = 0.600000;
ava('USA','ser') = 0.600000;
ava('EUR','agr') = 0.600000;
ava('EUR','mfg') = 0.600000;
ava('EUR','ser') = 0.600000;
ava('CHN','agr') = 0.600000;
ava('CHN','mfg') = 0.600000;
ava('CHN','ser') = 0.600000;
aio('USA','agr') = 0.215000;
aio('USA','mfg') = 0.215000;
aio('USA','ser') = 0.215000;
aio('EUR','agr') = 0.215000;
aio('EUR','mfg') = 0.215000;
aio('EUR','ser') = 0.215000;
aio('CHN','agr') = 0.215000;
aio('CHN','mfg') = 0.215000;
aio('CHN','ser') = 0.215000;
alphad('USA','agr') = 0.481928;
alphad('USA','mfg') = 0.481928;
alphad('USA','ser') = 0.481928;
alphad('EUR','agr') = 0.481928;
alphad('EUR','mfg') = 0.481928;
alphad('EUR','ser') = 0.481928;
alphad('CHN','agr') = 0.481928;
alphad('CHN','mfg') = 0.481928;
alphad('CHN','ser') = 0.481928;
alpham('USA','agr') = 0.036145;
alpham('USA','mfg') = 0.036145;
alpham('USA','ser') = 0.036145;
alpham('EUR','agr') = 0.036145;
alpham('EUR','mfg') = 0.036145;
alpham('EUR','ser') = 0.036145;
alpham('CHN','agr') = 0.036145;
alpham('CHN','mfg') = 0.036145;
alpham('CHN','ser') = 0.036145;
alphaf('USA','lab','agr') = 0.500000;
alphaf('USA','cap','agr') = 0.500000;
alphaf('USA','lab','mfg') = 0.500000;
alphaf('USA','cap','mfg') = 0.500000;
alphaf('USA','lab','ser') = 0.500000;
alphaf('USA','cap','ser') = 0.500000;
alphaf('EUR','lab','agr') = 0.500000;
alphaf('EUR','cap','agr') = 0.500000;
alphaf('EUR','lab','mfg') = 0.500000;
alphaf('EUR','cap','mfg') = 0.500000;
alphaf('EUR','lab','ser') = 0.500000;
alphaf('EUR','cap','ser') = 0.500000;
alphaf('CHN','lab','agr') = 0.500000;
alphaf('CHN','cap','agr') = 0.500000;
alphaf('CHN','lab','mfg') = 0.500000;
alphaf('CHN','cap','mfg') = 0.500000;
alphaf('CHN','lab','ser') = 0.500000;
alphaf('CHN','cap','ser') = 0.500000;
gamma_c('USA','agr') = 0.333333;
gamma_c('USA','mfg') = 0.333333;
gamma_c('USA','ser') = 0.333333;
gamma_c('EUR','agr') = 0.333333;
gamma_c('EUR','mfg') = 0.333333;
gamma_c('EUR','ser') = 0.333333;
gamma_c('CHN','agr') = 0.333333;
gamma_c('CHN','mfg') = 0.333333;
gamma_c('CHN','ser') = 0.333333;

* Variables - all at benchmark = 1 initially
Variable xp(r,a), x(r,a,i), px(r,a), pp(r,a);
Variable xs(r,i), ps(r,i), pd(r,i);
Variable xa(r,i), pa(r,i), xd(r,i), xmt(r,i), pmt(r,i);
Variable xet(r,i), pet(r,i);
Variable xf(r,f,a), xft(r,f), pf(r,f,a), pft(r,f);
Variable xc(r,i), xg(r,i), xi(r,i);
Variable regy(r), yc(r), yg(r);
Variable obj;

* Initialize at benchmark
xp.l(r,a) = 1; x.l(r,a,i) = 1; px.l(r,a) = 1; pp.l(r,a) = 1;
xs.l(r,i) = 1; ps.l(r,i) = 1; pd.l(r,i) = 1;
xa.l(r,i) = 1; pa.l(r,i) = 1; xd.l(r,i) = 0.7; xmt.l(r,i) = 0.3; pmt.l(r,i) = 1;
xet.l(r,i) = 0.2; pet.l(r,i) = 1;
xf.l(r,f,a) = 1; xft.l(r,f) = 1; pf.l(r,f,a) = 1; pft.l(r,f) = 1;
xc.l(r,i) = 0.5; xg.l(r,i) = 0.2; xi.l(r,i) = 0.3;
regy.l(r) = 1; yc.l(r) = 0.6; yg.l(r) = 0.3;

* Equations using calibrated parameters

* Production: Leontief technology
equation eq_xp(r,a);
eq_xp(r,a).. xp(r,a) =e= axp(r,a);

equation eq_x(r,a,i);
eq_x(r,a,i).. x(r,a,i) =e= xp(r,a);

equation eq_px(r,a);
eq_px(r,a).. px(r,a) * xp(r,a) =e= ava(r,a) * xp(r,a) * sum(f, pf(r,f,a) * xf(r,f,a)) / sum(f, xf(r,f,a))
    + aio(r,a) * xp(r,a) * pa(r,a);

* Supply
equation eq_xs(r,i);
eq_xs(r,i).. xs(r,i) =e= sum(a, x(r,a,i));

equation eq_ps(r,i);
eq_ps(r,i).. ps(r,i) =e= pd(r,i);

* Armington (simplified)
equation eq_xa(r,i);
eq_xa(r,i).. xa(r,i) =e= alphad(r,i) * xd(r,i) + alpham(r,i) * xmt(r,i);

equation eq_pa(r,i);
eq_pa(r,i).. pa(r,i) =e= alphad(r,i) * pd(r,i) + alpham(r,i) * pmt(r,i);

equation eq_pmt(r,i);
eq_pmt(r,i).. pmt(r,i) =e= 1.0;

* Trade
equation eq_xs_cet(r,i);
eq_xs_cet(r,i).. xs(r,i) =e= xd(r,i) + xet(r,i);

equation eq_pet(r,i);
eq_pet(r,i).. pet(r,i) =e= ps(r,i);

* Factors
equation eq_xft(r,f);
eq_xft(r,f).. xft(r,f) =e= sum(a, xf(r,f,a));

equation eq_xf(r,f,a);
eq_xf(r,f,a).. xf(r,f,a) =e= alphaf(r,f,a) * xp(r,a) / ava(r,a);

equation eq_pf(r,f,a);
eq_pf(r,f,a).. pf(r,f,a) =e= pft(r,f);

equation eq_pft(r,f);
eq_pft(r,f).. pft(r,f) =e= sum(a, pf(r,f,a) * alphaf(r,f,a)) / sum(a, alphaf(r,f,a));

* Demand
equation eq_xc(r,i);
eq_xc(r,i).. xc(r,i) =e= gamma_c(r,i) * yc(r) / pa(r,i);

equation eq_xg(r,i);
eq_xg(r,i).. xg(r,i) =e= 0.2;

equation eq_xi(r,i);
eq_xi(r,i).. xi(r,i) =e= 0.3;

* Income
equation eq_regy(r);
eq_regy(r).. regy(r) =e= sum((f,a), pf(r,f,a) * xf(r,f,a));

equation eq_yc(r);
eq_yc(r).. yc(r) =e= 0.6 * regy(r);

equation eq_yg(r);
eq_yg(r).. yg(r) =e= 0.3 * regy(r);

* Market clearing
equation mkt_goods(r,i);
mkt_goods(r,i).. xa(r,i) =e= xc(r,i) + xg(r,i) + xi(r,i) + sum(a, x(r,a,i));

* Objective
equation eq_obj;
eq_obj.. obj =e= 1;

Model gtap / all /;
Solve gtap using NLP minimizing obj;

* Export results
Execute_unload "gams_results.gdx", xp, x, px, pp, xs, ps, pd, xa, pa, xd, xmt, pmt,
    xet, pet, xf, xft, pf, pft, xc, xg, xi, regy, yc, yg;
