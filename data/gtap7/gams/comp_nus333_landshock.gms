* -------------------------------------------------------------------------
*  comp_nus333_landshock.gms — ROW LAND productivity shock (-30%) on NUS333.
*
*  Mirror of comp_nus333.gms but replaces the 10% tariff power shock with a
*  30% cut to ROW land supply: xft.fx(ROW, LAND, 'shock') = xft.l * 0.7.
*
*  The xft variable is already pinned by the closure (iterloop.gms re-fixes
*  xft.fx(r,fm,tsim) every period to last-period level); overwriting that
*  fix here applies the supply shock for the 'shock' period only.
*
*  Run with:
*    gams comp_nus333_landshock.gms --baseName=nus333 --inDir=... --outDir=...
* -------------------------------------------------------------------------

$setGlobal simType   CompStat
$setGlobal simName   LANDSHOCK
$if not setGlobal baseName $setGlobal baseName  nus333
$if not setGlobal inDir $setGlobal inDir     ../data
$if not setGlobal outDir $setGlobal outDir    ../output
$setGlobal utility   cde
$setGlobal savfFlag  capFix
$setGlobal ifCal       0
$setGlobal ifSUB       0
$if not setGlobal ifCSV $setGlobal ifCSV 1
$if not setGlobal ifCSVAppend $setGlobal ifCSVAppend 0

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
   $$iftheni "%simType%" == "CompStat"
      ifDyn       "Set to 1 to for a dynamic scenario"    / 0 /
   $$else
      ifDyn       "Set to 1 to for a dynamic scenario"    / 1 /
   $$endif
   ifDebug     "Set to 1 to debug calibration"         / 0 /
   inScale     "Scale for input data"                  / 1e-6 /
   xpScale     "Scale factor for output"               / 1 /
   ifCSV       "Flag for CSV file"                     / %ifCSV% /
   ifCSVAppend "Flag to append to existing CSV file"   / %ifCSVAppend% /
   ifMCP       "Set to 1 to solve using MCP"           / 1 /
   landCut     "Fraction by which to cut ROW land productivity" / 0.30 /
;

file
   csv      / "%outDir%/%simName%.csv" /
   screen   / con /
;

if(ifCSV,
   if(ifCSVAppend,
      csv.ap = 1 ;
      put csv ;
   else
      csv.ap = 0 ;
      put csv ;
      put "Variable,Region,Sector,Qualifier,Year,Value" / ;
   ) ;
   csv.pc=5 ;
   csv.nd=9 ;
) ;

file debug / "%outDir%/%simName%DBG.csv" / ;
if(ifDebug,
   put debug ;
   put "Var,Region,Sector,Qual,Year,Value" / ;
   debug.pc=5 ;
   debug.nd=9 ;
) ;

* -------------------------------------------------------------------------
*  Retrieve GTAP sets, data and parameters from NUS333 GDX files
* -------------------------------------------------------------------------

$include "getData.gms"

* -------------------------------------------------------------------------
*  NUS333-specific user definitions (identical to comp_nus333.gms)
* -------------------------------------------------------------------------

set l(fp) "Labor factors" /
   LABOR             "Labor"
/ ;

set cap(fp) "Capital factor" /
   CAPITAL           "Capital"
/ ;

set rres(r) "Residual region" /
   ROW
/ ;

set rmuv(r) "RMUV regions" /
   ROW
/ ;

set imuv(i) "IMUV commodities" /
   c_MFG
/ ;

* -------------------------------------------------------------------------
*  Load model, initialize variables and calibrate parameters
* -------------------------------------------------------------------------

$include "model.gms"
$include "cal.gms"

* -------------------------------------------------------------------------
*  Run the simulations for each time period
* -------------------------------------------------------------------------

rs(r) = yes ;
ts(t) = no ;

loop(tsim,

   ts(tsim) = yes ;

   $$include "iterloop.gms"

   if(sameas(tsim,'shock'),
*     -30% shock to ROW land supply: xft.fx(ROW, LAND, 'shock') = xft.l * 0.7
*     iterloop.gms has already fixed xft.fx(r,fm,tsim) = xft.l(r,fm,t-1) ;
*     overwriting that fix here applies the productivity shock.
      xft.fx('ROW','LAND',tsim) = xft.l('ROW','LAND',tsim) * (1 - landCut) ;
   ) ;

   options limrow = 3, limcol = 3, solprint = off, iterlim = 1000 ;

   if(years(tsim) gt firstYear,

      $$iftheni.solve "%simType%" == "CompStat"

         $$batinclude "solve.gms" gtap

      $$else.solve

         $$ifthen.calStatus %ifCal% == 1

            $$batinclude "solve.gms" dynCal

         $$else.calStatus

            $$batinclude "solve.gms" dynGTAP

         $$endif.calStatus

      $$endif.solve
   ) ;

   display walras.l ;
   put screen ;
   put / ;
   put "Walras: ", (walras.l/inScale) / ;
   putclose screen ;

   ts(tsim) = no ;
) ;

$include "postsim.gms"
execute_unload "%outDir%/%simName%.gdx" ;
