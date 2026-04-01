* -------------------------------------------------------------------------
*
*  Global free trade -- with upward sloping supply curves for natural res.
*                       Elasticity overrides are around line 95
*
*  Model preamble -- user options
*
* -------------------------------------------------------------------------

$setGlobal simType   CompStat
$setGlobal simName   GFTFlxFF
$setGlobal baseName  10x10
$setGlobal inDir     .
$setGlobal outDir    .
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
   $$iftheni "%simType%" == "CompStat"
      ifDyn       "Set to 1 to for a dynamic scenario"    / 0 /
   $$else
      ifDyn       "Set to 1 to for a dynamic scenario"    / 1 /
   $$endif
   ifDebug     "Set to 1 to debug calibration"         / 0 /
   inScale     "Scale for input data"                  / 1e-6 /
   xpScale     "Scale factor for output"               / 1 /
   ifCSV       "Flag for CSV file"                     / 1 /
   ifCSVAppend "Flag to append to existing CSV file"   / 0 /
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

* >>>>> OVERRIDE NATURAL RESOURCE SUPPLY ELASTICITIES <<<<<

parameter etaff0(r,fp,a) "Natl. resource supply elasticities" /

Oceania       .NatRes.a_Extraction   2
EastAsia      .NatRes.a_Extraction   1
SEAsia        .NatRes.a_Extraction   2
SouthAsia     .NatRes.a_Extraction   1
NAmerica      .NatRes.a_Extraction   2
LatinAmer     .NatRes.a_Extraction   2
EU_28         .NatRes.a_Extraction   1
MENA          .NatRes.a_Extraction   4
SSA           .NatRes.a_Extraction   2
RestofWorld   .NatRes.a_Extraction   3

/ ;

etaff(r,fp,a) = etaff0(r,fp,a) ;

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
   c_procfood, c_textWapp, c_LightMnfc, c_HeavyMnfc
/ ;

* -------------------------------------------------------------------------
*
*  Parameter overrides, for example factor supply elasticities,
*     output transformation elasticities
*
* -------------------------------------------------------------------------


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

set iter / it1*it10 / ;

loop(tsim,

   ts(tsim) = yes ;

   $$include "iterloop.gms"

   options limrow = 0, limcol = 0, solprint = off, iterlim = 1000 ;

*  Define the simulation specific shock

   if(sameas(tsim,'shock'),
      loop(iter,
         imptx.fx(s,i,d,tsim) = imptx.l(s,i,d,tsim-1)
                              * (1 - ord(iter)/card(iter)) ;
         if(ord(iter) lt card(iter),
            $$batinclude "solve.gms" gtap
         ) ;
      ) ;
   ) ;

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
