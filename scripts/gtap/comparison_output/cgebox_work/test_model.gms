$ontext
Simple GTAP test model - Square CNS system
$offtext

* Define sets
Set r "Regions" / USA, EUR /;
Set i "Commodities" / agr, mfg /;
Set a "Activities" / agr, mfg /;

* Define variables (square system: 4 vars = 4 eqns)
Variable xp(r,a) "Production";
Variable ps(r,i) "Supply price";

* Initialize at benchmark
xp.l(r,a) = 1;
ps.l(r,i) = 1;

* Equations: zero profit for each activity
equation prf_y(r,a) "Zero profit";
prf_y(r,a).. xp(r,a) =e= 1;

* Equations: market clearing for each commodity
equation mkt_ps(r,i) "Market clearing";
mkt_ps(r,i).. ps(r,i) =e= 1;

Model test / prf_y, mkt_ps /;
Solve test using CNS;

* Copy variable levels to parameters for easy export
Parameter xp_out(r,a), ps_out(r,i);
xp_out(r,a) = xp.l(r,a);
ps_out(r,i) = ps.l(r,i);

* Save results as parameters
Execute_unload "results.gdx", xp_out, ps_out;
