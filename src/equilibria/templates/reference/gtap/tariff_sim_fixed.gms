* -------------------------------------------------------------------------
*  Tariff simulation -- 10% increase in import tariffs
*  This file includes the standard COMP model and adds a tariff shock
* -------------------------------------------------------------------------

$setGlobal ifCSV 0
$setGlobal ifCSVAppend 0

* Include the working COMP model that we know works on NEOS
$include "/Users/marmol/proyectos2/cge_babel/standard_gtap_7/comp.gms"

* -------------------------------------------------------------------------
*  Now add the tariff shock scenario
* -------------------------------------------------------------------------

* Save baseline values before shock
alias(REG,REGG);

Parameter
   qgdp_base(REG)      "Baseline real GDP"
   vgdp_base(REG)      "Baseline nominal GDP"
   evoa_base(REG)      "Baseline welfare"
;

qgdp_base(REG) = qgdp.l(REG) ;
vgdp_base(REG) = vgdp.l(REG) ;
evoa_base(REG) = evoa.l(REG) ;

Display "=== Baseline saved ===" ;

* -------------------------------------------------------------------------
*  Apply 10% uniform tariff shock to all bilateral import tariffs
* -------------------------------------------------------------------------

Parameter
   tm_original(TRAD_COMM,REG,REGG) "Original import tariff power"
   tm_shock                        / 0.10 /
;

* Save baseline tariff rates (tm is tariff power: 1+tariff_rate)
tm_original(TRAD_COMM,REG,REGG)$MSHHR(TRAD_COMM,REG,REGG) = tm.l(TRAD_COMM,REG,REGG) ;

* Apply 10% tariff increase: tm goes from (1+t) to (1+t)*(1+0.10)
* For example: if tariff was 5% (tm=1.05), new tariff is (1.05)*1.10 = 1.155 (15.5%)
tm.fx(TRAD_COMM,REG,REGG)$MSHHR(TRAD_COMM,REG,REGG) = 
    tm_original(TRAD_COMM,REG,REGG) * (1 + tm_shock) ;

Display "=== Applied 10% tariff shock ===" ;

* -------------------------------------------------------------------------
*  Solve model with tariff shock
* -------------------------------------------------------------------------

Solve STD_GTAP using mcp ;

Display "=== Tariff shock simulation completed ===" ;

* -------------------------------------------------------------------------
*  Calculate percentage changes from baseline
* -------------------------------------------------------------------------

Parameter
   qgdp_chg(REG)       "Real GDP % change"
   vgdp_chg(REG)       "Nominal GDP % change"
   evoa_chg(REG)       "Welfare % change (EV)"
   qgdp_level(REG)     "Post-shock real GDP level"
   vgdp_level(REG)     "Post-shock nominal GDP level"
   evoa_level(REG)     "Post-shock welfare level"
;

qgdp_level(REG) = qgdp.l(REG) ;
vgdp_level(REG) = vgdp.l(REG) ;
evoa_level(REG) = evoa.l(REG) ;

qgdp_chg(REG) = 100 * (qgdp_level(REG) - qgdp_base(REG)) / qgdp_base(REG) ;
vgdp_chg(REG) = 100 * (vgdp_level(REG) - vgdp_base(REG)) / vgdp_base(REG) ;
evoa_chg(REG) = 100 * (evoa_level(REG) - evoa_base(REG)) / evoa_base(REG) ;

Display
   "=== TARIFF SHOCK RESULTS ===" 
   "Baseline:"
   qgdp_base
   vgdp_base
   evoa_base
   "Post-shock levels:"
   qgdp_level
   vgdp_level
   evoa_level
   "Percentage changes:"
   qgdp_chg
   vgdp_chg
   evoa_chg
;

* -------------------------------------------------------------------------
*  Export all results to GDX
* -------------------------------------------------------------------------

execute_unload 'TARIFF_SIM_RESULTS.gdx',
   qgdp_base, vgdp_base, evoa_base,
   qgdp_level, vgdp_level, evoa_level,
   qgdp_chg, vgdp_chg, evoa_chg,
   tm_original
;

Display "=== Results exported to TARIFF_SIM_RESULTS.gdx ===" ;

