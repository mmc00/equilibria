* -------------------------------------------------------------------------
*
*  Standard model diagnostics
*
*  Model preamble -- user options
*
* -------------------------------------------------------------------------

$setGlobal simType   CompStat
$setGlobal simName   COMP_ALTERTAX
$setGlobal baseName  9x10
$setGlobal inDir     .
$setGlobal outDir    .
$setGlobal utility   cde
$setGlobal savfFlag  capFix
$setGlobal ifCal       0
$setGlobal ifSUB       1
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

*  CSV results go to this file

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

*  This file is optional--sometimes useful to debug model

file debug / "%outDir%/%simName%DBG.csv" / ;
if(ifDebug,
   put debug ;
   put "Var,Region,Sector,Qual,Year,Value" / ;
   debug.pc=5 ;
   debug.nd=9 ;
) ;

* -------------------------------------------------------------------------
*
*  Retrieve GTAP sets, data and parameters
*
* -------------------------------------------------------------------------

$include "getData.gms"

* -------------------------------------------------------------------------
*
*  Required user definitions
*
* -------------------------------------------------------------------------

set l(fp) "Labor factors" /
   UnSkLab           "Unskilled labor"
   SkLab             "Skilled labor"
/ ;

set cap(fp) "Capital factor" /
   Capital
/ ;

set rres(r) "Residual region" /
   NAmerica
/ ;

set rmuv(r) "RMUV regions" /
   Oceania, NAmerica, EU_28
/ ;

set imuv(i) "IMUV commodities" /
   c_ProcFood, c_TextWapp, c_LightMnfc, c_HeavyMnfc
/ ;

* -------------------------------------------------------------------------
*
*  Parameter overrides, for example factor supply elasticities,
*     output transformation elasticities
*
* -------------------------------------------------------------------------

* -------------------------------------------------------------------------
*  ALTERTAX overrides (Malcolm 1998, mirrored from
*  cgebox/scen/Parameters/parameter_altertax.gms):
*     - Cobb-Douglas value-added (esubva = 1)
*     - Armington top + second level both 0.95
*     - CET for factors: 1 (CD)
*     - Government / investment expenditure: CD (sigmag, sigmai = 1)
*     - All factors fully mobile (omegaf set on all fp, fm := all factors)
* -------------------------------------------------------------------------

esubva(a0,r) = 1.0 ;
esubd(i0,r)  = 0.95 ;
esubm(i0,r)  = 0.95 ;
etrae(fp,r)  = 1.0 ;

* Make ALL factors mobile under altertax (override the fnm/fm split).
fnm(fp)      = no ;
fm(fp)       = yes ;

* CET mobility for ALL fp set to CD (omegaf = 1).
omegaf(r,fp) = 1.0 ;

* Government / investment expenditure: CD limit (esubg, esubi = 1)
esubg(r)     = 1.0 ;
esubi(r)     = 1.0 ;

* Force ND/VA (sigmap), intermediate nest (sigmand), VA (sigmav) to CD.
sigmap(r,a)  = 1.0 ;
sigmand(r,a) = 1.0 ;
sigmav(r,a)  = 1.0 ;

* Mark altertax run for diagnostics
scalar altertax / 1 / ;

* -------------------------------------------------------------------------
*
*  Load model, initialize variables and calibrate parameters
*
* -------------------------------------------------------------------------

*  Get the model specification

$include "model.gms"

*  Initialize the model

$include "cal.gms"

* -------------------------------------------------------------------------
*
*  Run the simulations for each time period
*
* -------------------------------------------------------------------------

rs(r) = yes ;
ts(t) = no ;

loop(tsim,

   ts(tsim) = yes ;

   $$include "iterloop.gms"

*  Define the simulation specific shock

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
