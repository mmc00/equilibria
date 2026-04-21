* ------------------------------------------------------------------------------
* GTAP calibration dump for NEOS (self-contained GTAP script conventions)
*
* Uses GTAP standard include/data conventions so NEOS can resolve model-library
* files similarly to comp.gms submissions.
* ------------------------------------------------------------------------------

$setGlobal simType   CompStat
$setGlobal simName   CAL_DUMP
$if not setGlobal baseName $setGlobal baseName  9x10
$if not setGlobal inDir $setGlobal inDir     ../data
$if not setGlobal outDir $setGlobal outDir    ../output
$setGlobal utility   cde
$setGlobal savfFlag  capFix
$setGlobal ifCal       0
$setGlobal ifSUB       1
$setGlobal ifCSV       0
$setGlobal ifCSVAppend 0

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
   ifDyn       "Set to 1 for dynamic scenario"         / 0 /
   ifDebug     "Set to 1 to debug calibration"         / 0 /
   inScale     "Scale for input data"                  / 1e-6 /
   xpScale     "Scale factor for output"               / 1 /
   ifCSV       "Flag for CSV file"                     / %ifCSV% /
   ifCSVAppend "Flag to append to existing CSV file"   / %ifCSVAppend% /
   ifMCP       "Set to 1 to solve using MCP"           / 0 /
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

$include "getData.gms"

set l(fp) "Labor factors" /
   UnSkLab
   SkLab
/
;

set cap(fp) "Capital factor" /
   Capital
/
;

set rres(r) "Residual region" /
   NAmerica
/
;

set rmuv(r) "RMUV regions" /
   Oceania, NAmerica, EU_28
/
;

set imuv(i) "IMUV commodities" /
   c_ProcFood, c_TextWapp, c_LightMnfc, c_HeavyMnfc
/
;

$include "model.gms"
$include "cal.gms"

parameter
   diag_counts(*)
   diag_scalars(*)
;

diag_counts("and")    = sum((r,a,t0)$and(r,a,t0), 1) ;
diag_counts("ava")    = sum((r,a,t0)$ava(r,a,t0), 1) ;
diag_counts("io")     = sum((r,i,a,t0)$io(r,i,a,t0), 1) ;
diag_counts("af")     = sum((r,fp,a,t0)$af(r,fp,a,t0), 1) ;
diag_counts("gx")     = sum((r,a,i,t0)$gx(r,a,i,t0), 1) ;
diag_counts("gf")     = sum((r,fp,a,t0)$gf(r,fp,a,t0), 1) ;
diag_counts("gw")     = sum((r,i,rp,t0)$gw(r,i,rp,t0), 1) ;
diag_counts("gd")     = sum((r,i,t0)$gd(r,i,t0), 1) ;
diag_counts("ge")     = sum((r,i,t0)$ge(r,i,t0), 1) ;
diag_counts("amw")    = sum((rp,i,r,t0)$amw(rp,i,r,t0), 1) ;
diag_counts("alphaa") = sum((r,i,aa,t0)$alphaa(r,i,aa,t0), 1) ;
diag_counts("tmarg")  = sum((r,i,rp,t0)$(abs(tmarg.l(r,i,rp,t0)) > 1e-12), 1) ;
diag_counts("chipm")  = sum((r,i,rp)$(abs(chipm(r,i,rp)) > 1e-12), 1) ;
diag_counts("kappaf") = sum((r,fp,a,t0)$(abs(kappaf.l(r,fp,a,t0)) > 1e-12), 1) ;

diag_counts("xp_l")   = sum((r,a,t0)$(abs(xp.l(r,a,t0)) > 1e-12), 1) ;
diag_counts("xf_l")   = sum((r,fp,a,t0)$(abs(xf.l(r,fp,a,t0)) > 1e-12), 1) ;
diag_counts("xa_l")   = sum((r,i,aa,t0)$(abs(xa.l(r,i,aa,t0)) > 1e-12), 1) ;
diag_counts("xd_l")   = sum((r,i,aa,t0)$(abs(xd.l(r,i,aa,t0)) > 1e-12), 1) ;
diag_counts("xm_l")   = sum((r,i,aa,t0)$(abs(xm.l(r,i,aa,t0)) > 1e-12), 1) ;
diag_counts("xw_l")   = sum((r,i,rp,t0)$(abs(xw.l(r,i,rp,t0)) > 1e-12), 1) ;
diag_counts("pf_l")   = sum((r,fp,a,t0)$(abs(pf.l(r,fp,a,t0)) > 1e-12), 1) ;
diag_counts("pa_l")   = sum((r,i,aa,t0)$(abs(pa.l(r,i,aa,t0)) > 1e-12), 1) ;

diag_scalars("walras_l") = 0 ;
diag_scalars("xigbl_l")  = sum(t0, xigbl.l(t0)) ;
diag_scalars("pigbl_l")  = sum(t0, pigbl.l(t0)) ;
diag_scalars("rorg_l")   = sum(t0, rorg.l(t0)) ;

execute_unload "%outDir%/gams_cal_dump_%baseName%.gdx" ;
