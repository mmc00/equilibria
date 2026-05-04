* -------------------------------------------------------------------------
*
*  comp_nus333.gms - Standard GTAP comparative-static run for the NUS333
*  dataset (3 sectors x 2 regions: USA, ROW; 3 mobile factors: LAND,
*  LABOR, CAPITAL). Same model logic as comp.gms; only the dataset-specific
*  hardcoded sets (l, cap, rres, rmuv, imuv) are adapted to NUS333 labels.
*
*  Run with:
*    gams comp_nus333.gms --baseName=nus333 --inDir=/path/to/nus333_gdx \
*                         --outDir=/path/to/output
*
* -------------------------------------------------------------------------

$setGlobal simType   CompStat
$setGlobal simName   COMP
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
*  NUS333-specific user definitions
* -------------------------------------------------------------------------

* Single labor factor in NUS333 (no skilled/unskilled split).
set l(fp) "Labor factors" /
   LABOR             "Labor"
/ ;

set cap(fp) "Capital factor" /
   CAPITAL           "Capital"
/ ;

* Residual region for the closure - chose ROW so USA shock effects
* remain on the active side of the SOE-style closure.
set rres(r) "Residual region" /
   ROW
/ ;

* MUV deflator: only ROW is large; use ROW + MFG as a non-empty deflator
* basket so the price index macro mqmuv does not collapse to zero.
set rmuv(r) "RMUV regions" /
   ROW
/ ;

set imuv(i) "IMUV commodities" /
   MFG
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
      pnum.fx(tsim) = 1.5 ;
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
