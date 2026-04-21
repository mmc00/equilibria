* -------------------------------------------------------------------------
*
*  Tariff simulation -- 10% increase in import tariffs
*
*  Model preamble -- user options
*
* -------------------------------------------------------------------------

$setGlobal simType   CompStat
$setGlobal simName   TARIFF_SIM
$if not setGlobal baseName $setGlobal baseName  9x10
$if not setGlobal inDir $setGlobal inDir     ../data
$if not setGlobal outDir $setGlobal outDir    ../output
$setGlobal utility   cde
$setGlobal savfFlag  capFlex
$setGlobal ifCal       0
$setGlobal ifSUB       1

set
   t        "Time frame"   / base, check, shock /
   t0(t)    "Base year"    / base /
   ts(t)    "Time flag"
;
alias(t,tsim) ;

Parameter
   years(t)
   gap(t)
   FirstYear
;

years(t) = ord(t) ;
gap(t)   = 1 ;
loop(t0,
   FirstYear = years(t0) ;
) ;

ts(t) = no ;

scalar
   ifSUB       "Set to 1 to reduce model size"         / %ifSUB% /
   ifCal       "Set to 1 to calibrate dynamically"     / %ifCal% /
   ifDyn       "Set to 1 to for a dynamic scenario"    / 0 /
   ifDebug     "Set to 1 to debug calibration"         / 0 /
   inScale     "Scale for input data"                  / 1e-6 /
   xpScale     "Scale factor for output"               / 1 /
;

$include model.gms

Display
   "Loaded base data"
;

* Solve baseline to establish equilibrium
ts("check") = yes ;
Solve STD_GTAP using mcp ;

Display
   "Baseline equilibrium established"
   qgdp.l
   vgdp.l
;

* Apply 10% tariff shock to all imports
ts("shock") = yes ;

Parameter
   tm_base(i,r,s)  "Baseline tariff rate"
   tm_shock        "Tariff shock magnitude" / 0.10 /
;

* Save baseline tariffs
tm_base(i,r,s) = tm(i,r,s) ;

* Apply uniform 10% increase to all import tariffs
tm(i,r,s) = tm_base(i,r,s) * (1 + tm_shock) ;

Display
   "Applied 10% tariff shock"
   tm_base
   tm
;

* Solve with tariff shock
Solve STD_GTAP using mcp ;

Display
   "Tariff simulation completed"
   qgdp.l
   vgdp.l
;

* Export results
execute_unload "%outDir%/tariff_sim_results.gdx" ;

* Compare baseline vs shock results
Parameter
   gdp_change(r)    "GDP change from tariff shock (%)"
   welfare_change(r) "Welfare change from tariff shock"
;

gdp_change(r) = 100 * (qgdp.l(r,"shock") / qgdp.l(r,"check") - 1) ;

Display
   gdp_change
;
