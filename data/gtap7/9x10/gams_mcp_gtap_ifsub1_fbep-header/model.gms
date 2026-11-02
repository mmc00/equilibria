* -------------------------------------------------------------------------
*
*  Standard model diagnostics
*
*  Model preamble -- user options
*
* -------------------------------------------------------------------------

$setGlobal simType   CompStat
$setGlobal simName   COMP
$if not setGlobal baseName $setGlobal baseName  9x10
$if not setGlobal inDir $setGlobal inDir     ../data
$if not setGlobal outDir $setGlobal outDir    ../output
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
   csv      / "COMP.csv" /
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

file debug / "COMPDBG.csv" / ;
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
* --------------------------------------------------------------------------------------------------
*
*     Read in the GTAP SETS
*
* --------------------------------------------------------------------------------------------------


* === Inlined SETS from in.gdx ===

Set acts '' /
   a_agricultur,
   a_Extraction,
   a_ProcFood,
   a_TextWapp,
   a_LightMnfc,
   a_HeavyMnfc,
   a_Util_Cons,
   a_TransComm,
   a_OthService
/;

Set comm '' /
   c_Crops,
   c_MeatLstk,
   c_Extraction,
   c_ProcFood,
   c_TextWapp,
   c_LightMnfc,
   c_HeavyMnfc,
   c_Util_Cons,
   c_TransComm,
   c_OthService
/;

Set reg '' /
   Oceania,
   EastAsia,
   SEAsia,
   SouthAsia,
   NAmerica,
   LatinAmer,
   EU_28,
   MENA,
   SSA,
   RestofWorld
/;

Set endw '' /
   Land,
   UnSkLab,
   SkLab,
   Capital,
   NatRes
/;

Set marg '' /
   c_TransComm
/;

Set endwm '' /
   SkLab,
   UnSkLab,
   Capital
/;

Set endwf '' /
   NatRes
/;

Set endws '' /
   Land
/;

* [getData set decl stripped] sets
* [getData set decl stripped]    acts           "Activities"
* [getData set decl stripped]    comm           "Commodities"
* [getData set decl stripped]    marg(comm)     "Margin commodities"
* [getData set decl stripped]    reg            "Regions"
* [getData set decl stripped]    endw           "Endowments"
* [getData set decl stripped]    endwf(endw)    "Fixed factors"
* [getData set decl stripped]    endwm(endw)    "Mobile factors"
* [getData set decl stripped]    endws(endw)    "Sluggish factors"
* [getData set decl stripped] ;
* [stripped for NEOS inline] $gdxin "%inDir%/%BaseName%Sets.gdx"
* [stripped for NEOS inline] $load acts, comm, marg, reg, endw, endwf, endwm, endws
* [stripped for NEOS inline] $gdxin

*  CREATE THE SAM SETS

set stdlab "Standard SAM labels" /
   TRD               "Trade account"
   hhd               "Household"
   gov               "Government"
   inv               "Investment"
   deprY             "Depreciation"
   tmg               "Trade margins"
   itax              "Indirect tax"
   ptax              "Production tax"
   mtax              "Import tax"
   etax              "Export tax"
   vtax              "Taxes on factors of production"
   vsub              "Subsidies on factors of production"
   dtax              "Direct taxation"
   ctax              "Carbon tax"
   bop               "Balance of payments account"
   tot               "Total for row/column sums"
/ ;

set findem(stdlab) "Final demand accounts" /
   hhd               "Household"
   gov               "Government"
   inv               "Investment"
   tmg               "Trade margins"
/ ;

set is "SAM accounts for aggregated SAM" /

*  User-defined activities

   set.acts

*  User-defined commodities

   set.comm

*  User-defined factors

   set.endw

*  Standard SAM accounts

   set.stdlab

*  User-defined regions

   set.reg

/ ;

alias(is, js) ;

set aa(is) "Armington agents" /

   set.acts

   set.findem

/ ;

set a(aa) "Activities" /

   set.acts

/ ;

set i(is) "Commodities" /

   set.comm

/ ;
alias(i, j) ;

set r(is) "Regions" /

   set.reg

/ ;

alias(r,s) ; alias(r,d) ; alias(r,rp) ;

set fp(is)  "Factors of production" /

   set.endw

/ ;

set fnm(fp) "Non-mobile factors" ;
loop((fp,endwf)$sameas(fp,endwf),
   fnm(fp) = yes ;
) ;

set fm(fp) "Mobile factors" ;
fm(fp)$(not fnm(fp)) = yes ;


set fd(aa) "Domestic final demand agents" /

   set.findem

/ ;

set h(fd) "Households" /
   hhd               "Household"
/ ;

set gov(fd) "Government" /
   gov               "Government"
/ ;

set inv(fd) "Investment" /
   inv               "Investment"
/ ;

set fdc(fd) "Final demand accounts with CES expenditure function" /
   gov               "Government"
   inv               "Investment"
/ ;

set tmg(fd) "Domestic supply of trade margins services" /
   tmg               "Trade margins"
/ ;

alias(i0,i) ; alias(a0,a) ; alias(i,j) ; alias(j0,i0) ;

sets
   mapa0(a,a0)
   mapi0(i,i0)
;

*  No aggregation needed for this version of the model--so map a0 to a and i0 to i

mapa0(a,a) = yes ;
mapi0(i,i) = yes ;

* --------------------------------------------------------------------------------------------------
*
*     Read in the GTAP database
*
* --------------------------------------------------------------------------------------------------

parameters
   VDFB(i0, a0, r)      "Firm purchases of domestic goods at basic prices"
   VDFP(i0, a0, r)      "Firm purchases of domestic goods at purchaser prices"
   VMFB(i0, a0, r)      "Firm purchases of imported goods at basic prices"
   VMFP(i0, a0, r)      "Firm purchases of domestic goods at purchaser prices"
   VDPB(i0, r)          "Private purchases of domestic goods at basic prices"
   VDPP(i0, r)          "Private purchases of domestic goods at purchaser prices"
   VMPB(i0, r)          "Private purchases of imported goods at basic prices"
   VMPP(i0, r)          "Private purchases of domestic goods at purchaser prices"
   VDGB(i0, r)          "Government purchases of domestic goods at basic prices"
   VDGP(i0, r)          "Government purchases of domestic goods at purchaser prices"
   VMGB(i0, r)          "Government purchases of imported goods at basic prices"
   VMGP(i0, r)          "Government purchases of domestic goods at purchaser prices"
   VDIB(i0, r)          "Investment purchases of domestic goods at basic prices"
   VDIP(i0, r)          "Investment purchases of domestic goods at purchaser prices"
   VMIB(i0, r)          "Investment purchases of imported goods at basic prices"
   VMIP(i0, r)          "Investment purchases of domestic goods at purchaser prices"

   EVFB(fp, a0, r)      "Primary factor purchases at basic prices"
   EVFP(fp, a0, r)      "Primary factor purchases at purchaser prices"
   EVOS(fp, a0, r)      "Factor remuneration after income tax"

   VXSB(i0, r, rp)      "Exports at basic prices"
   VFOB(i0, r, rp)      "Exports at FOB prices"
   VCIF(i0, r, rp)      "Import at CIF prices"
   VMSB(i0, r, rp)      "Imports at basic prices"

   VST(i0, r)           "Exports of trade and transport services"
   VTWR(i0, j0, r, rp)  "Margins by margin commodity"

   SAVE(r)              "Net saving, by region"
   VDEP(r)              "Capital depreciation"
   VKB(r)               "Capital stock"
   POP0(r)              "Population"

   MAKS(i0,a0,r)        "Make matrix at supply prices"
   MAKB(i0,a0,r)        "Make matrix at basic prices (incl taxes)"
   PTAX(i0,a0,r)        "Output taxes"

   fbep(fp, a0, r)      "Factor subsidies"
   ftrv(fp, a0, r)      "Tax on factor use"
   tvom(a0,r)           "Value of output"
   check(a0,r)          "Check"
;

* Load the GTAP data at compile time so NEOS remote execution does not need
* access to the local GDX files during the execution phase.
* [stripped for NEOS inline] $gdxin "%inDir%/%BaseName%Dat.gdx"
* [stripped for NEOS inline] $load vdfb vdfp vmfb vmfp
* [stripped for NEOS inline] $load vdpb vdpp vmpb vmpp
* [stripped for NEOS inline] $load vdgb vdgp vmgb vmgp
* [stripped for NEOS inline] $load vdib vdip vmib vmip
* [stripped for NEOS inline] $load evfb evfp evos
* [stripped for NEOS inline] $load vxsb vfob vcif vmsb
* [stripped for NEOS inline] $load vst vtwr
* [stripped for NEOS inline] $load save vdep vkb pop0=pop
* [stripped for NEOS inline] $load maks makb
* [stripped for NEOS inline] $gdxin

*  !!!! MAY WANT TO FIX THIS AT SOME STAGE--THERE IS INCONSISTENCY IN THE
*        HANDLING OF FBEP and FTRV

fbep(fp,a0,r) = 0 ;
* === Inlined PARAMETER data from in.gdx ===
$onImplicitAssign
* vdfb data (900 cells)
vdfb('c_Crops','a_agricultur','Oceania') = 3747.946045 ;
vdfb('c_Crops','a_agricultur','EastAsia') = 162582.2656 ;
vdfb('c_Crops','a_agricultur','SEAsia') = 64831.82812 ;
vdfb('c_Crops','a_agricultur','SouthAsia') = 95996.64844 ;
vdfb('c_Crops','a_agricultur','NAmerica') = 50745.85547 ;
vdfb('c_Crops','a_agricultur','LatinAmer') = 40633.17578 ;
vdfb('c_Crops','a_agricultur','EU_28') = 35521.55469 ;
vdfb('c_Crops','a_agricultur','MENA') = 33805.29297 ;
vdfb('c_Crops','a_agricultur','SSA') = 15173.30957 ;
vdfb('c_Crops','a_agricultur','RestofWorld') = 19694.15039 ;
vdfb('c_Crops','a_Extraction','Oceania') = 221.7974548 ;
vdfb('c_Crops','a_Extraction','EastAsia') = 3727.342529 ;
vdfb('c_Crops','a_Extraction','SEAsia') = 2011.128662 ;
vdfb('c_Crops','a_Extraction','SouthAsia') = 59.17321777 ;
vdfb('c_Crops','a_Extraction','NAmerica') = 218.1413269 ;
vdfb('c_Crops','a_Extraction','LatinAmer') = 523.1860962 ;
vdfb('c_Crops','a_Extraction','EU_28') = 493.4533081 ;
vdfb('c_Crops','a_Extraction','MENA') = 1089.545898 ;
vdfb('c_Crops','a_Extraction','SSA') = 181.1891937 ;
vdfb('c_Crops','a_Extraction','RestofWorld') = 211.2624207 ;
vdfb('c_Crops','a_ProcFood','Oceania') = 5237.226562 ;
vdfb('c_Crops','a_ProcFood','EastAsia') = 233467.0625 ;
vdfb('c_Crops','a_ProcFood','SEAsia') = 53751.44922 ;
vdfb('c_Crops','a_ProcFood','SouthAsia') = 80290.70312 ;
vdfb('c_Crops','a_ProcFood','NAmerica') = 65400.37891 ;
vdfb('c_Crops','a_ProcFood','LatinAmer') = 92361.22656 ;
vdfb('c_Crops','a_ProcFood','EU_28') = 61727.5 ;
vdfb('c_Crops','a_ProcFood','MENA') = 32990.07812 ;
vdfb('c_Crops','a_ProcFood','SSA') = 17862.77539 ;
vdfb('c_Crops','a_ProcFood','RestofWorld') = 24755.69727 ;
vdfb('c_Crops','a_TextWapp','Oceania') = 17.05568314 ;
vdfb('c_Crops','a_TextWapp','EastAsia') = 45226.73438 ;
vdfb('c_Crops','a_TextWapp','SEAsia') = 689.6853027 ;
vdfb('c_Crops','a_TextWapp','SouthAsia') = 19039.51953 ;
vdfb('c_Crops','a_TextWapp','NAmerica') = 792.9057617 ;
vdfb('c_Crops','a_TextWapp','LatinAmer') = 4567.870605 ;
vdfb('c_Crops','a_TextWapp','EU_28') = 5076.521484 ;
vdfb('c_Crops','a_TextWapp','MENA') = 2539.617676 ;
vdfb('c_Crops','a_TextWapp','SSA') = 1714.110718 ;
vdfb('c_Crops','a_TextWapp','RestofWorld') = 482.0223083 ;
vdfb('c_Crops','a_LightMnfc','Oceania') = 53.71229553 ;
vdfb('c_Crops','a_LightMnfc','EastAsia') = 14153.25195 ;
vdfb('c_Crops','a_LightMnfc','SEAsia') = 3355.314941 ;
vdfb('c_Crops','a_LightMnfc','SouthAsia') = 832.0418701 ;
vdfb('c_Crops','a_LightMnfc','NAmerica') = 124.3728333 ;
vdfb('c_Crops','a_LightMnfc','LatinAmer') = 450.2002869 ;
vdfb('c_Crops','a_LightMnfc','EU_28') = 827.2999268 ;
vdfb('c_Crops','a_LightMnfc','MENA') = 477.2406311 ;
vdfb('c_Crops','a_LightMnfc','SSA') = 1628.617676 ;
vdfb('c_Crops','a_LightMnfc','RestofWorld') = 574.84021 ;
vdfb('c_Crops','a_HeavyMnfc','Oceania') = 125.3504562 ;
vdfb('c_Crops','a_HeavyMnfc','EastAsia') = 17574.45898 ;
vdfb('c_Crops','a_HeavyMnfc','SEAsia') = 5904.19043 ;
vdfb('c_Crops','a_HeavyMnfc','SouthAsia') = 5532.426758 ;
vdfb('c_Crops','a_HeavyMnfc','NAmerica') = 1911.028809 ;
vdfb('c_Crops','a_HeavyMnfc','LatinAmer') = 9765.496094 ;
vdfb('c_Crops','a_HeavyMnfc','EU_28') = 2123.051758 ;
vdfb('c_Crops','a_HeavyMnfc','MENA') = 743.9385986 ;
vdfb('c_Crops','a_HeavyMnfc','SSA') = 869.9567871 ;
vdfb('c_Crops','a_HeavyMnfc','RestofWorld') = 381.3477478 ;
vdfb('c_Crops','a_Util_Cons','Oceania') = 941.6238403 ;
vdfb('c_Crops','a_Util_Cons','EastAsia') = 5759.796387 ;
vdfb('c_Crops','a_Util_Cons','SEAsia') = 186.5285339 ;
vdfb('c_Crops','a_Util_Cons','SouthAsia') = 3533.357666 ;
vdfb('c_Crops','a_Util_Cons','NAmerica') = 4083.848877 ;
vdfb('c_Crops','a_Util_Cons','LatinAmer') = 276.1221619 ;
vdfb('c_Crops','a_Util_Cons','EU_28') = 3162.332275 ;
vdfb('c_Crops','a_Util_Cons','MENA') = 183.006134 ;
vdfb('c_Crops','a_Util_Cons','SSA') = 126.5093079 ;
vdfb('c_Crops','a_Util_Cons','RestofWorld') = 316.7387695 ;
vdfb('c_Crops','a_TransComm','Oceania') = 2180.503662 ;
vdfb('c_Crops','a_TransComm','EastAsia') = 37983.46875 ;
vdfb('c_Crops','a_TransComm','SEAsia') = 10528.89062 ;
vdfb('c_Crops','a_TransComm','SouthAsia') = 17434.64844 ;
vdfb('c_Crops','a_TransComm','NAmerica') = 4944.286621 ;
vdfb('c_Crops','a_TransComm','LatinAmer') = 5338.305664 ;
vdfb('c_Crops','a_TransComm','EU_28') = 11315.05469 ;
vdfb('c_Crops','a_TransComm','MENA') = 2946.374756 ;
vdfb('c_Crops','a_TransComm','SSA') = 1320.024414 ;
vdfb('c_Crops','a_TransComm','RestofWorld') = 2799.597656 ;
vdfb('c_Crops','a_OthService','Oceania') = 1928.473755 ;
vdfb('c_Crops','a_OthService','EastAsia') = 15332.75 ;
vdfb('c_Crops','a_OthService','SEAsia') = 3263.268311 ;
vdfb('c_Crops','a_OthService','SouthAsia') = 1049.137085 ;
vdfb('c_Crops','a_OthService','NAmerica') = 11399.17285 ;
vdfb('c_Crops','a_OthService','LatinAmer') = 2656.846924 ;
vdfb('c_Crops','a_OthService','EU_28') = 10221.12305 ;
vdfb('c_Crops','a_OthService','MENA') = 5922.895996 ;
vdfb('c_Crops','a_OthService','SSA') = 916.8139038 ;
vdfb('c_Crops','a_OthService','RestofWorld') = 2770.646973 ;
vdfb('c_MeatLstk','a_agricultur','Oceania') = 14347.34766 ;
vdfb('c_MeatLstk','a_agricultur','EastAsia') = 159389.7969 ;
vdfb('c_MeatLstk','a_agricultur','SEAsia') = 26608.86133 ;
vdfb('c_MeatLstk','a_agricultur','SouthAsia') = 20510.75586 ;
vdfb('c_MeatLstk','a_agricultur','NAmerica') = 141200.0156 ;
vdfb('c_MeatLstk','a_agricultur','LatinAmer') = 92942.45312 ;
vdfb('c_MeatLstk','a_agricultur','EU_28') = 90133.22656 ;
vdfb('c_MeatLstk','a_agricultur','MENA') = 29786.54883 ;
vdfb('c_MeatLstk','a_agricultur','SSA') = 22957.95508 ;
vdfb('c_MeatLstk','a_agricultur','RestofWorld') = 55728.43359 ;
vdfb('c_MeatLstk','a_Extraction','Oceania') = 41.32938385 ;
vdfb('c_MeatLstk','a_Extraction','EastAsia') = 297.8728943 ;
vdfb('c_MeatLstk','a_Extraction','SEAsia') = 55.12163925 ;
vdfb('c_MeatLstk','a_Extraction','SouthAsia') = 45.27746582 ;
vdfb('c_MeatLstk','a_Extraction','NAmerica') = 479.4245911 ;
vdfb('c_MeatLstk','a_Extraction','LatinAmer') = 260.9805298 ;
vdfb('c_MeatLstk','a_Extraction','EU_28') = 222.3279877 ;
vdfb('c_MeatLstk','a_Extraction','MENA') = 51.06682968 ;
vdfb('c_MeatLstk','a_Extraction','SSA') = 103.5401535 ;
vdfb('c_MeatLstk','a_Extraction','RestofWorld') = 171.7084656 ;
vdfb('c_MeatLstk','a_ProcFood','Oceania') = 12302.08105 ;
vdfb('c_MeatLstk','a_ProcFood','EastAsia') = 60435.28516 ;
vdfb('c_MeatLstk','a_ProcFood','SEAsia') = 5994.856934 ;
vdfb('c_MeatLstk','a_ProcFood','SouthAsia') = 8731.430664 ;
vdfb('c_MeatLstk','a_ProcFood','NAmerica') = 50382.29297 ;
vdfb('c_MeatLstk','a_ProcFood','LatinAmer') = 31220.2207 ;
vdfb('c_MeatLstk','a_ProcFood','EU_28') = 84470.96094 ;
vdfb('c_MeatLstk','a_ProcFood','MENA') = 22446.88672 ;
vdfb('c_MeatLstk','a_ProcFood','SSA') = 3643.663574 ;
vdfb('c_MeatLstk','a_ProcFood','RestofWorld') = 10668.29297 ;
vdfb('c_MeatLstk','a_TextWapp','Oceania') = 438.839447 ;
vdfb('c_MeatLstk','a_TextWapp','EastAsia') = 17085.30078 ;
vdfb('c_MeatLstk','a_TextWapp','SEAsia') = 400.0744629 ;
vdfb('c_MeatLstk','a_TextWapp','SouthAsia') = 653.149231 ;
vdfb('c_MeatLstk','a_TextWapp','NAmerica') = 989.9855347 ;
vdfb('c_MeatLstk','a_TextWapp','LatinAmer') = 1054.987183 ;
vdfb('c_MeatLstk','a_TextWapp','EU_28') = 194.2825775 ;
vdfb('c_MeatLstk','a_TextWapp','MENA') = 1998.054321 ;
vdfb('c_MeatLstk','a_TextWapp','SSA') = 309.1931152 ;
vdfb('c_MeatLstk','a_TextWapp','RestofWorld') = 852.3637695 ;
vdfb('c_MeatLstk','a_LightMnfc','Oceania') = 254.1936951 ;
vdfb('c_MeatLstk','a_LightMnfc','EastAsia') = 62372.93359 ;
vdfb('c_MeatLstk','a_LightMnfc','SEAsia') = 911.1380005 ;
vdfb('c_MeatLstk','a_LightMnfc','SouthAsia') = 10547.08008 ;
vdfb('c_MeatLstk','a_LightMnfc','NAmerica') = 2049.386719 ;
vdfb('c_MeatLstk','a_LightMnfc','LatinAmer') = 3683.966797 ;
vdfb('c_MeatLstk','a_LightMnfc','EU_28') = 951.7039795 ;
vdfb('c_MeatLstk','a_LightMnfc','MENA') = 554.7333374 ;
vdfb('c_MeatLstk','a_LightMnfc','SSA') = 693.4593506 ;
vdfb('c_MeatLstk','a_LightMnfc','RestofWorld') = 521.5037231 ;
vdfb('c_MeatLstk','a_HeavyMnfc','Oceania') = 1208.727295 ;
vdfb('c_MeatLstk','a_HeavyMnfc','EastAsia') = 21611.66797 ;
vdfb('c_MeatLstk','a_HeavyMnfc','SEAsia') = 540.510498 ;
vdfb('c_MeatLstk','a_HeavyMnfc','SouthAsia') = 835.3009644 ;
vdfb('c_MeatLstk','a_HeavyMnfc','NAmerica') = 1617.355713 ;
vdfb('c_MeatLstk','a_HeavyMnfc','LatinAmer') = 1984.729492 ;
vdfb('c_MeatLstk','a_HeavyMnfc','EU_28') = 1870.143066 ;
vdfb('c_MeatLstk','a_HeavyMnfc','MENA') = 622.3815918 ;
vdfb('c_MeatLstk','a_HeavyMnfc','SSA') = 159.1750031 ;
vdfb('c_MeatLstk','a_HeavyMnfc','RestofWorld') = 1124.252441 ;
vdfb('c_MeatLstk','a_Util_Cons','Oceania') = 137.4409485 ;
vdfb('c_MeatLstk','a_Util_Cons','EastAsia') = 12.15774059 ;
vdfb('c_MeatLstk','a_Util_Cons','SEAsia') = 2.244171858 ;
vdfb('c_MeatLstk','a_Util_Cons','SouthAsia') = 254.0172272 ;
vdfb('c_MeatLstk','a_Util_Cons','NAmerica') = 79.60671997 ;
vdfb('c_MeatLstk','a_Util_Cons','LatinAmer') = 138.1144257 ;
vdfb('c_MeatLstk','a_Util_Cons','EU_28') = 419.7877502 ;
vdfb('c_MeatLstk','a_Util_Cons','MENA') = 108.2214508 ;
vdfb('c_MeatLstk','a_Util_Cons','SSA') = 209.0470123 ;
vdfb('c_MeatLstk','a_Util_Cons','RestofWorld') = 146.5032501 ;
vdfb('c_MeatLstk','a_TransComm','Oceania') = 4828.513184 ;
vdfb('c_MeatLstk','a_TransComm','EastAsia') = 50959.01953 ;
vdfb('c_MeatLstk','a_TransComm','SEAsia') = 8350.331055 ;
vdfb('c_MeatLstk','a_TransComm','SouthAsia') = 7292.710938 ;
vdfb('c_MeatLstk','a_TransComm','NAmerica') = 13875.85352 ;
vdfb('c_MeatLstk','a_TransComm','LatinAmer') = 9892.211914 ;
vdfb('c_MeatLstk','a_TransComm','EU_28') = 18359.38867 ;
vdfb('c_MeatLstk','a_TransComm','MENA') = 3927.497559 ;
vdfb('c_MeatLstk','a_TransComm','SSA') = 1804.403564 ;
vdfb('c_MeatLstk','a_TransComm','RestofWorld') = 4531.586914 ;
vdfb('c_MeatLstk','a_OthService','Oceania') = 1739.408325 ;
vdfb('c_MeatLstk','a_OthService','EastAsia') = 6588.685059 ;
vdfb('c_MeatLstk','a_OthService','SEAsia') = 1417.445068 ;
vdfb('c_MeatLstk','a_OthService','SouthAsia') = 173.1059723 ;
vdfb('c_MeatLstk','a_OthService','NAmerica') = 31561.83203 ;
vdfb('c_MeatLstk','a_OthService','LatinAmer') = 5075.908691 ;
vdfb('c_MeatLstk','a_OthService','EU_28') = 10810.80371 ;
vdfb('c_MeatLstk','a_OthService','MENA') = 2913.481934 ;
vdfb('c_MeatLstk','a_OthService','SSA') = 1641.198975 ;
vdfb('c_MeatLstk','a_OthService','RestofWorld') = 5336.687988 ;
vdfb('c_Extraction','a_agricultur','Oceania') = 643.1380005 ;
vdfb('c_Extraction','a_agricultur','EastAsia') = 1798.797241 ;
vdfb('c_Extraction','a_agricultur','SEAsia') = 1714.599731 ;
vdfb('c_Extraction','a_agricultur','SouthAsia') = 493.3206787 ;
vdfb('c_Extraction','a_agricultur','NAmerica') = 1278.366577 ;
vdfb('c_Extraction','a_agricultur','LatinAmer') = 2805.202148 ;
vdfb('c_Extraction','a_agricultur','EU_28') = 1969.768921 ;
vdfb('c_Extraction','a_agricultur','MENA') = 301.8361816 ;
vdfb('c_Extraction','a_agricultur','SSA') = 963.8328857 ;
vdfb('c_Extraction','a_agricultur','RestofWorld') = 1512.515869 ;
vdfb('c_Extraction','a_Extraction','Oceania') = 17494.92969 ;
vdfb('c_Extraction','a_Extraction','EastAsia') = 56090.20703 ;
vdfb('c_Extraction','a_Extraction','SEAsia') = 8577.691406 ;
vdfb('c_Extraction','a_Extraction','SouthAsia') = 2759.819092 ;
vdfb('c_Extraction','a_Extraction','NAmerica') = 13841.58301 ;
vdfb('c_Extraction','a_Extraction','LatinAmer') = 19477.38281 ;
vdfb('c_Extraction','a_Extraction','EU_28') = 11263.43262 ;
vdfb('c_Extraction','a_Extraction','MENA') = 10601.53418 ;
vdfb('c_Extraction','a_Extraction','SSA') = 3072.110352 ;
vdfb('c_Extraction','a_Extraction','RestofWorld') = 22382.46289 ;
vdfb('c_Extraction','a_ProcFood','Oceania') = 1844.315674 ;
vdfb('c_Extraction','a_ProcFood','EastAsia') = 68708.94531 ;
vdfb('c_Extraction','a_ProcFood','SEAsia') = 13706.53223 ;
vdfb('c_Extraction','a_ProcFood','SouthAsia') = 3500.070557 ;
vdfb('c_Extraction','a_ProcFood','NAmerica') = 5796.866699 ;
vdfb('c_Extraction','a_ProcFood','LatinAmer') = 6973.117676 ;
vdfb('c_Extraction','a_ProcFood','EU_28') = 7210.813477 ;
vdfb('c_Extraction','a_ProcFood','MENA') = 1726.991089 ;
vdfb('c_Extraction','a_ProcFood','SSA') = 6314.240723 ;
vdfb('c_Extraction','a_ProcFood','RestofWorld') = 7057.076172 ;
vdfb('c_Extraction','a_TextWapp','Oceania') = 32.58847809 ;
vdfb('c_Extraction','a_TextWapp','EastAsia') = 1034.103516 ;
vdfb('c_Extraction','a_TextWapp','SEAsia') = 201.4160614 ;
vdfb('c_Extraction','a_TextWapp','SouthAsia') = 87.35192108 ;
vdfb('c_Extraction','a_TextWapp','NAmerica') = 352.4451904 ;
vdfb('c_Extraction','a_TextWapp','LatinAmer') = 131.6085663 ;
vdfb('c_Extraction','a_TextWapp','EU_28') = 111.6153564 ;
vdfb('c_Extraction','a_TextWapp','MENA') = 179.4017639 ;
vdfb('c_Extraction','a_TextWapp','SSA') = 36.07077789 ;
vdfb('c_Extraction','a_TextWapp','RestofWorld') = 100.6092758 ;
vdfb('c_Extraction','a_LightMnfc','Oceania') = 2907.313965 ;
vdfb('c_Extraction','a_LightMnfc','EastAsia') = 47011.11328 ;
vdfb('c_Extraction','a_LightMnfc','SEAsia') = 11980.84473 ;
vdfb('c_Extraction','a_LightMnfc','SouthAsia') = 7372.502441 ;
vdfb('c_Extraction','a_LightMnfc','NAmerica') = 27984.46289 ;
vdfb('c_Extraction','a_LightMnfc','LatinAmer') = 13026.86133 ;
vdfb('c_Extraction','a_LightMnfc','EU_28') = 32054.89453 ;
vdfb('c_Extraction','a_LightMnfc','MENA') = 2645.294189 ;
vdfb('c_Extraction','a_LightMnfc','SSA') = 3855.954346 ;
vdfb('c_Extraction','a_LightMnfc','RestofWorld') = 11810.72168 ;
vdfb('c_Extraction','a_HeavyMnfc','Oceania') = 36638.08594 ;
vdfb('c_Extraction','a_HeavyMnfc','EastAsia') = 432211.8438 ;
vdfb('c_Extraction','a_HeavyMnfc','SEAsia') = 74059.82031 ;
vdfb('c_Extraction','a_HeavyMnfc','SouthAsia') = 37453.83203 ;
vdfb('c_Extraction','a_HeavyMnfc','NAmerica') = 339625.0312 ;
vdfb('c_Extraction','a_HeavyMnfc','LatinAmer') = 140012.7344 ;
vdfb('c_Extraction','a_HeavyMnfc','EU_28') = 95242.91406 ;
vdfb('c_Extraction','a_HeavyMnfc','MENA') = 271279.9062 ;
vdfb('c_Extraction','a_HeavyMnfc','SSA') = 17824.57227 ;
vdfb('c_Extraction','a_HeavyMnfc','RestofWorld') = 250664.375 ;
vdfb('c_Extraction','a_Util_Cons','Oceania') = 11808.60059 ;
vdfb('c_Extraction','a_Util_Cons','EastAsia') = 111995.6953 ;
vdfb('c_Extraction','a_Util_Cons','SEAsia') = 19498.74219 ;
vdfb('c_Extraction','a_Util_Cons','SouthAsia') = 19916.72656 ;
vdfb('c_Extraction','a_Util_Cons','NAmerica') = 88589.00781 ;
vdfb('c_Extraction','a_Util_Cons','LatinAmer') = 11848.37793 ;
vdfb('c_Extraction','a_Util_Cons','EU_28') = 41149.5625 ;
vdfb('c_Extraction','a_Util_Cons','MENA') = 61428.36328 ;
vdfb('c_Extraction','a_Util_Cons','SSA') = 7895.030762 ;
vdfb('c_Extraction','a_Util_Cons','RestofWorld') = 61220.01562 ;
vdfb('c_Extraction','a_TransComm','Oceania') = 1522.196167 ;
vdfb('c_Extraction','a_TransComm','EastAsia') = 26004.60352 ;
vdfb('c_Extraction','a_TransComm','SEAsia') = 2870.086426 ;
vdfb('c_Extraction','a_TransComm','SouthAsia') = 1430.456787 ;
vdfb('c_Extraction','a_TransComm','NAmerica') = 4209.951172 ;
vdfb('c_Extraction','a_TransComm','LatinAmer') = 1543.014893 ;
vdfb('c_Extraction','a_TransComm','EU_28') = 5605.247559 ;
vdfb('c_Extraction','a_TransComm','MENA') = 1147.311523 ;
vdfb('c_Extraction','a_TransComm','SSA') = 584.3660889 ;
vdfb('c_Extraction','a_TransComm','RestofWorld') = 11488.78516 ;
vdfb('c_Extraction','a_OthService','Oceania') = 1219.603149 ;
vdfb('c_Extraction','a_OthService','EastAsia') = 5466.147461 ;
vdfb('c_Extraction','a_OthService','SEAsia') = 751.5771484 ;
vdfb('c_Extraction','a_OthService','SouthAsia') = 63.8975563 ;
vdfb('c_Extraction','a_OthService','NAmerica') = 4137.228027 ;
vdfb('c_Extraction','a_OthService','LatinAmer') = 1176.316162 ;
vdfb('c_Extraction','a_OthService','EU_28') = 5002.291504 ;
vdfb('c_Extraction','a_OthService','MENA') = 1747.686768 ;
vdfb('c_Extraction','a_OthService','SSA') = 654.269043 ;
vdfb('c_Extraction','a_OthService','RestofWorld') = 4751.784668 ;
vdfb('c_ProcFood','a_agricultur','Oceania') = 1529.526001 ;
vdfb('c_ProcFood','a_agricultur','EastAsia') = 111721.5234 ;
vdfb('c_ProcFood','a_agricultur','SEAsia') = 11372.59863 ;
vdfb('c_ProcFood','a_agricultur','SouthAsia') = 3555.964355 ;
vdfb('c_ProcFood','a_agricultur','NAmerica') = 41932.14844 ;
vdfb('c_ProcFood','a_agricultur','LatinAmer') = 27123.84961 ;
vdfb('c_ProcFood','a_agricultur','EU_28') = 47923.97656 ;
vdfb('c_ProcFood','a_agricultur','MENA') = 8500.251953 ;
vdfb('c_ProcFood','a_agricultur','SSA') = 5141.88623 ;
vdfb('c_ProcFood','a_agricultur','RestofWorld') = 9741.027344 ;
vdfb('c_ProcFood','a_Extraction','Oceania') = 301.4434509 ;
vdfb('c_ProcFood','a_Extraction','EastAsia') = 14959.96777 ;
vdfb('c_ProcFood','a_Extraction','SEAsia') = 2120.954346 ;
vdfb('c_ProcFood','a_Extraction','SouthAsia') = 284.421051 ;
vdfb('c_ProcFood','a_Extraction','NAmerica') = 813.3590088 ;
vdfb('c_ProcFood','a_Extraction','LatinAmer') = 2068.260742 ;
vdfb('c_ProcFood','a_Extraction','EU_28') = 1096.479858 ;
vdfb('c_ProcFood','a_Extraction','MENA') = 672.8251953 ;
vdfb('c_ProcFood','a_Extraction','SSA') = 459.2012329 ;
vdfb('c_ProcFood','a_Extraction','RestofWorld') = 3091.888916 ;
vdfb('c_ProcFood','a_ProcFood','Oceania') = 11456.55664 ;
vdfb('c_ProcFood','a_ProcFood','EastAsia') = 218383.7344 ;
vdfb('c_ProcFood','a_ProcFood','SEAsia') = 50127.98828 ;
vdfb('c_ProcFood','a_ProcFood','SouthAsia') = 20692.48828 ;
vdfb('c_ProcFood','a_ProcFood','NAmerica') = 109490.7344 ;
vdfb('c_ProcFood','a_ProcFood','LatinAmer') = 76029.46875 ;
vdfb('c_ProcFood','a_ProcFood','EU_28') = 206280.7969 ;
vdfb('c_ProcFood','a_ProcFood','MENA') = 33048.23047 ;
vdfb('c_ProcFood','a_ProcFood','SSA') = 18179.55859 ;
vdfb('c_ProcFood','a_ProcFood','RestofWorld') = 50960.17969 ;
vdfb('c_ProcFood','a_TextWapp','Oceania') = 15.50391197 ;
vdfb('c_ProcFood','a_TextWapp','EastAsia') = 3768.780518 ;
vdfb('c_ProcFood','a_TextWapp','SEAsia') = 586.0960083 ;
vdfb('c_ProcFood','a_TextWapp','SouthAsia') = 439.5722961 ;
vdfb('c_ProcFood','a_TextWapp','NAmerica') = 122.0737762 ;
vdfb('c_ProcFood','a_TextWapp','LatinAmer') = 168.7750549 ;
vdfb('c_ProcFood','a_TextWapp','EU_28') = 2097.282227 ;
vdfb('c_ProcFood','a_TextWapp','MENA') = 442.1375427 ;
vdfb('c_ProcFood','a_TextWapp','SSA') = 162.1771088 ;
vdfb('c_ProcFood','a_TextWapp','RestofWorld') = 87.96647644 ;
vdfb('c_ProcFood','a_LightMnfc','Oceania') = 130.0000916 ;
vdfb('c_ProcFood','a_LightMnfc','EastAsia') = 11047.83496 ;
vdfb('c_ProcFood','a_LightMnfc','SEAsia') = 1020.037231 ;
vdfb('c_ProcFood','a_LightMnfc','SouthAsia') = 10491.9082 ;
vdfb('c_ProcFood','a_LightMnfc','NAmerica') = 1656.870605 ;
vdfb('c_ProcFood','a_LightMnfc','LatinAmer') = 618.8469238 ;
vdfb('c_ProcFood','a_LightMnfc','EU_28') = 10761.40527 ;
vdfb('c_ProcFood','a_LightMnfc','MENA') = 281.1493835 ;
vdfb('c_ProcFood','a_LightMnfc','SSA') = 752.888855 ;
vdfb('c_ProcFood','a_LightMnfc','RestofWorld') = 681.7566528 ;
vdfb('c_ProcFood','a_HeavyMnfc','Oceania') = 448.0777588 ;
vdfb('c_ProcFood','a_HeavyMnfc','EastAsia') = 42019.76562 ;
vdfb('c_ProcFood','a_HeavyMnfc','SEAsia') = 3762.131104 ;
vdfb('c_ProcFood','a_HeavyMnfc','SouthAsia') = 1845.952393 ;
vdfb('c_ProcFood','a_HeavyMnfc','NAmerica') = 2812.286133 ;
vdfb('c_ProcFood','a_HeavyMnfc','LatinAmer') = 3007.549072 ;
vdfb('c_ProcFood','a_HeavyMnfc','EU_28') = 14729.38867 ;
vdfb('c_ProcFood','a_HeavyMnfc','MENA') = 765.0623779 ;
vdfb('c_ProcFood','a_HeavyMnfc','SSA') = 958.8529663 ;
vdfb('c_ProcFood','a_HeavyMnfc','RestofWorld') = 994.9884644 ;
vdfb('c_ProcFood','a_Util_Cons','Oceania') = 148.4550018 ;
vdfb('c_ProcFood','a_Util_Cons','EastAsia') = 7124.958984 ;
vdfb('c_ProcFood','a_Util_Cons','SEAsia') = 111.7287827 ;
vdfb('c_ProcFood','a_Util_Cons','SouthAsia') = 74.60312653 ;
vdfb('c_ProcFood','a_Util_Cons','NAmerica') = 428.8509827 ;
vdfb('c_ProcFood','a_Util_Cons','LatinAmer') = 498.6388245 ;
vdfb('c_ProcFood','a_Util_Cons','EU_28') = 2463.806396 ;
vdfb('c_ProcFood','a_Util_Cons','MENA') = 393.9165039 ;
vdfb('c_ProcFood','a_Util_Cons','SSA') = 153.9747772 ;
vdfb('c_ProcFood','a_Util_Cons','RestofWorld') = 400.7962341 ;
vdfb('c_ProcFood','a_TransComm','Oceania') = 12546.83496 ;
vdfb('c_ProcFood','a_TransComm','EastAsia') = 149765.4062 ;
vdfb('c_ProcFood','a_TransComm','SEAsia') = 16219.36914 ;
vdfb('c_ProcFood','a_TransComm','SouthAsia') = 10012.2207 ;
vdfb('c_ProcFood','a_TransComm','NAmerica') = 31041.16797 ;
vdfb('c_ProcFood','a_TransComm','LatinAmer') = 41697.22656 ;
vdfb('c_ProcFood','a_TransComm','EU_28') = 120151.2578 ;
vdfb('c_ProcFood','a_TransComm','MENA') = 10123.86816 ;
vdfb('c_ProcFood','a_TransComm','SSA') = 4866.271973 ;
vdfb('c_ProcFood','a_TransComm','RestofWorld') = 11954.63672 ;
vdfb('c_ProcFood','a_OthService','Oceania') = 4264.098633 ;
vdfb('c_ProcFood','a_OthService','EastAsia') = 46339.83203 ;
vdfb('c_ProcFood','a_OthService','SEAsia') = 4206.466309 ;
vdfb('c_ProcFood','a_OthService','SouthAsia') = 638.6053467 ;
vdfb('c_ProcFood','a_OthService','NAmerica') = 125873.0703 ;
vdfb('c_ProcFood','a_OthService','LatinAmer') = 11480.24414 ;
vdfb('c_ProcFood','a_OthService','EU_28') = 50569.49219 ;
vdfb('c_ProcFood','a_OthService','MENA') = 14528.29297 ;
vdfb('c_ProcFood','a_OthService','SSA') = 6520.793945 ;
vdfb('c_ProcFood','a_OthService','RestofWorld') = 15370.8877 ;
vdfb('c_TextWapp','a_agricultur','Oceania') = 61.1699028 ;
vdfb('c_TextWapp','a_agricultur','EastAsia') = 643.4482422 ;
vdfb('c_TextWapp','a_agricultur','SEAsia') = 121.5865555 ;
vdfb('c_TextWapp','a_agricultur','SouthAsia') = 1716.267334 ;
vdfb('c_TextWapp','a_agricultur','NAmerica') = 1584.205444 ;
vdfb('c_TextWapp','a_agricultur','LatinAmer') = 842.161377 ;
vdfb('c_TextWapp','a_agricultur','EU_28') = 487.9519348 ;
vdfb('c_TextWapp','a_agricultur','MENA') = 791.9141846 ;
vdfb('c_TextWapp','a_agricultur','SSA') = 853.3110352 ;
vdfb('c_TextWapp','a_agricultur','RestofWorld') = 243.0013885 ;
vdfb('c_TextWapp','a_Extraction','Oceania') = 188.8171844 ;
vdfb('c_TextWapp','a_Extraction','EastAsia') = 3382.124023 ;
vdfb('c_TextWapp','a_Extraction','SEAsia') = 328.3035278 ;
vdfb('c_TextWapp','a_Extraction','SouthAsia') = 599.6270752 ;
vdfb('c_TextWapp','a_Extraction','NAmerica') = 667.2670898 ;
vdfb('c_TextWapp','a_Extraction','LatinAmer') = 930.7546387 ;
vdfb('c_TextWapp','a_Extraction','EU_28') = 491.6601562 ;
vdfb('c_TextWapp','a_Extraction','MENA') = 268.3081665 ;
vdfb('c_TextWapp','a_Extraction','SSA') = 591.1124878 ;
vdfb('c_TextWapp','a_Extraction','RestofWorld') = 495.288208 ;
vdfb('c_TextWapp','a_ProcFood','Oceania') = 86.1611557 ;
vdfb('c_TextWapp','a_ProcFood','EastAsia') = 2336.577637 ;
vdfb('c_TextWapp','a_ProcFood','SEAsia') = 200.7409821 ;
vdfb('c_TextWapp','a_ProcFood','SouthAsia') = 1769.76062 ;
vdfb('c_TextWapp','a_ProcFood','NAmerica') = 544.3461304 ;
vdfb('c_TextWapp','a_ProcFood','LatinAmer') = 1269.659668 ;
vdfb('c_TextWapp','a_ProcFood','EU_28') = 928.5420532 ;
vdfb('c_TextWapp','a_ProcFood','MENA') = 644.2787476 ;
vdfb('c_TextWapp','a_ProcFood','SSA') = 294.1923523 ;
vdfb('c_TextWapp','a_ProcFood','RestofWorld') = 340.6341553 ;
vdfb('c_TextWapp','a_TextWapp','Oceania') = 921.0551147 ;
vdfb('c_TextWapp','a_TextWapp','EastAsia') = 342781.5312 ;
vdfb('c_TextWapp','a_TextWapp','SEAsia') = 29734.14844 ;
vdfb('c_TextWapp','a_TextWapp','SouthAsia') = 38505.54688 ;
vdfb('c_TextWapp','a_TextWapp','NAmerica') = 83691.16406 ;
vdfb('c_TextWapp','a_TextWapp','LatinAmer') = 39170.66406 ;
vdfb('c_TextWapp','a_TextWapp','EU_28') = 81187.97656 ;
vdfb('c_TextWapp','a_TextWapp','MENA') = 46827.98047 ;
vdfb('c_TextWapp','a_TextWapp','SSA') = 4359.197754 ;
vdfb('c_TextWapp','a_TextWapp','RestofWorld') = 8876.935547 ;
vdfb('c_TextWapp','a_LightMnfc','Oceania') = 331.6177673 ;
vdfb('c_TextWapp','a_LightMnfc','EastAsia') = 50691.99609 ;
vdfb('c_TextWapp','a_LightMnfc','SEAsia') = 2863.040039 ;
vdfb('c_TextWapp','a_LightMnfc','SouthAsia') = 909.7306519 ;
vdfb('c_TextWapp','a_LightMnfc','NAmerica') = 21343.16211 ;
vdfb('c_TextWapp','a_LightMnfc','LatinAmer') = 4689.302734 ;
vdfb('c_TextWapp','a_LightMnfc','EU_28') = 13818.57227 ;
vdfb('c_TextWapp','a_LightMnfc','MENA') = 2062.363281 ;
vdfb('c_TextWapp','a_LightMnfc','SSA') = 1357.575195 ;
vdfb('c_TextWapp','a_LightMnfc','RestofWorld') = 1675.409546 ;
vdfb('c_TextWapp','a_HeavyMnfc','Oceania') = 230.303009 ;
vdfb('c_TextWapp','a_HeavyMnfc','EastAsia') = 35220.84375 ;
vdfb('c_TextWapp','a_HeavyMnfc','SEAsia') = 3244.368408 ;
vdfb('c_TextWapp','a_HeavyMnfc','SouthAsia') = 1741.541504 ;
vdfb('c_TextWapp','a_HeavyMnfc','NAmerica') = 9224.258789 ;
vdfb('c_TextWapp','a_HeavyMnfc','LatinAmer') = 2170.04834 ;
vdfb('c_TextWapp','a_HeavyMnfc','EU_28') = 7329.769043 ;
vdfb('c_TextWapp','a_HeavyMnfc','MENA') = 2084.987549 ;
vdfb('c_TextWapp','a_HeavyMnfc','SSA') = 307.9414978 ;
vdfb('c_TextWapp','a_HeavyMnfc','RestofWorld') = 1038.86145 ;
vdfb('c_TextWapp','a_Util_Cons','Oceania') = 390.5699463 ;
vdfb('c_TextWapp','a_Util_Cons','EastAsia') = 11371.31738 ;
vdfb('c_TextWapp','a_Util_Cons','SEAsia') = 402.1499939 ;
vdfb('c_TextWapp','a_Util_Cons','SouthAsia') = 392.8338013 ;
vdfb('c_TextWapp','a_Util_Cons','NAmerica') = 4602.054199 ;
vdfb('c_TextWapp','a_Util_Cons','LatinAmer') = 764.6972046 ;
vdfb('c_TextWapp','a_Util_Cons','EU_28') = 2550.137451 ;
vdfb('c_TextWapp','a_Util_Cons','MENA') = 712.3661499 ;
vdfb('c_TextWapp','a_Util_Cons','SSA') = 240.0709534 ;
vdfb('c_TextWapp','a_Util_Cons','RestofWorld') = 710.9805908 ;
vdfb('c_TextWapp','a_TransComm','Oceania') = 874.4945068 ;
vdfb('c_TextWapp','a_TransComm','EastAsia') = 25712.71875 ;
vdfb('c_TextWapp','a_TransComm','SEAsia') = 2604.530029 ;
vdfb('c_TextWapp','a_TransComm','SouthAsia') = 1632.225708 ;
vdfb('c_TextWapp','a_TransComm','NAmerica') = 11862.39453 ;
vdfb('c_TextWapp','a_TransComm','LatinAmer') = 4393.683594 ;
vdfb('c_TextWapp','a_TransComm','EU_28') = 10733.18848 ;
vdfb('c_TextWapp','a_TransComm','MENA') = 4199.804688 ;
vdfb('c_TextWapp','a_TransComm','SSA') = 1999.874634 ;
vdfb('c_TextWapp','a_TransComm','RestofWorld') = 2327.296387 ;
vdfb('c_TextWapp','a_OthService','Oceania') = 1177.235229 ;
vdfb('c_TextWapp','a_OthService','EastAsia') = 44274.47266 ;
vdfb('c_TextWapp','a_OthService','SEAsia') = 2610.343262 ;
vdfb('c_TextWapp','a_OthService','SouthAsia') = 1258.216187 ;
vdfb('c_TextWapp','a_OthService','NAmerica') = 12620.86914 ;
vdfb('c_TextWapp','a_OthService','LatinAmer') = 6842.498535 ;
vdfb('c_TextWapp','a_OthService','EU_28') = 12510.30957 ;
vdfb('c_TextWapp','a_OthService','MENA') = 8957.020508 ;
vdfb('c_TextWapp','a_OthService','SSA') = 2413.028564 ;
vdfb('c_TextWapp','a_OthService','RestofWorld') = 3709.221436 ;
vdfb('c_LightMnfc','a_agricultur','Oceania') = 626.7124634 ;
vdfb('c_LightMnfc','a_agricultur','EastAsia') = 9034.885742 ;
vdfb('c_LightMnfc','a_agricultur','SEAsia') = 1426.23291 ;
vdfb('c_LightMnfc','a_agricultur','SouthAsia') = 702.2893066 ;
vdfb('c_LightMnfc','a_agricultur','NAmerica') = 12778.30664 ;
vdfb('c_LightMnfc','a_agricultur','LatinAmer') = 6342.509766 ;
vdfb('c_LightMnfc','a_agricultur','EU_28') = 10810.40625 ;
vdfb('c_LightMnfc','a_agricultur','MENA') = 1843.003906 ;
vdfb('c_LightMnfc','a_agricultur','SSA') = 2298.649902 ;
vdfb('c_LightMnfc','a_agricultur','RestofWorld') = 3566.614502 ;
vdfb('c_LightMnfc','a_Extraction','Oceania') = 2979.126465 ;
vdfb('c_LightMnfc','a_Extraction','EastAsia') = 28403.73633 ;
vdfb('c_LightMnfc','a_Extraction','SEAsia') = 2801.329834 ;
vdfb('c_LightMnfc','a_Extraction','SouthAsia') = 1635.676758 ;
vdfb('c_LightMnfc','a_Extraction','NAmerica') = 7353.897949 ;
vdfb('c_LightMnfc','a_Extraction','LatinAmer') = 8898.429688 ;
vdfb('c_LightMnfc','a_Extraction','EU_28') = 5949.101562 ;
vdfb('c_LightMnfc','a_Extraction','MENA') = 3475.533691 ;
vdfb('c_LightMnfc','a_Extraction','SSA') = 3692.794434 ;
vdfb('c_LightMnfc','a_Extraction','RestofWorld') = 9959.209961 ;
vdfb('c_LightMnfc','a_ProcFood','Oceania') = 3198.479004 ;
vdfb('c_LightMnfc','a_ProcFood','EastAsia') = 47138.47656 ;
vdfb('c_LightMnfc','a_ProcFood','SEAsia') = 4832.606934 ;
vdfb('c_LightMnfc','a_ProcFood','SouthAsia') = 2990.355713 ;
vdfb('c_LightMnfc','a_ProcFood','NAmerica') = 54196.38281 ;
vdfb('c_LightMnfc','a_ProcFood','LatinAmer') = 12460.96387 ;
vdfb('c_LightMnfc','a_ProcFood','EU_28') = 48053.26562 ;
vdfb('c_LightMnfc','a_ProcFood','MENA') = 4856.526367 ;
vdfb('c_LightMnfc','a_ProcFood','SSA') = 2897.302734 ;
vdfb('c_LightMnfc','a_ProcFood','RestofWorld') = 7141.435547 ;
vdfb('c_LightMnfc','a_TextWapp','Oceania') = 234.4776154 ;
vdfb('c_LightMnfc','a_TextWapp','EastAsia') = 36370.55078 ;
vdfb('c_LightMnfc','a_TextWapp','SEAsia') = 2246.589111 ;
vdfb('c_LightMnfc','a_TextWapp','SouthAsia') = 2425.213379 ;
vdfb('c_LightMnfc','a_TextWapp','NAmerica') = 5284.316895 ;
vdfb('c_LightMnfc','a_TextWapp','LatinAmer') = 2698.482666 ;
vdfb('c_LightMnfc','a_TextWapp','EU_28') = 12909.91309 ;
vdfb('c_LightMnfc','a_TextWapp','MENA') = 3243.989746 ;
vdfb('c_LightMnfc','a_TextWapp','SSA') = 930.5673218 ;
vdfb('c_LightMnfc','a_TextWapp','RestofWorld') = 1015.223755 ;
vdfb('c_LightMnfc','a_LightMnfc','Oceania') = 16700.74219 ;
vdfb('c_LightMnfc','a_LightMnfc','EastAsia') = 1138957.375 ;
vdfb('c_LightMnfc','a_LightMnfc','SEAsia') = 51628.50781 ;
vdfb('c_LightMnfc','a_LightMnfc','SouthAsia') = 36565.11328 ;
vdfb('c_LightMnfc','a_LightMnfc','NAmerica') = 493404.375 ;
vdfb('c_LightMnfc','a_LightMnfc','LatinAmer') = 101487.0781 ;
vdfb('c_LightMnfc','a_LightMnfc','EU_28') = 697688.3125 ;
vdfb('c_LightMnfc','a_LightMnfc','MENA') = 45917.67578 ;
vdfb('c_LightMnfc','a_LightMnfc','SSA') = 25760.125 ;
vdfb('c_LightMnfc','a_LightMnfc','RestofWorld') = 46805.0625 ;
vdfb('c_LightMnfc','a_HeavyMnfc','Oceania') = 6335.320312 ;
vdfb('c_LightMnfc','a_HeavyMnfc','EastAsia') = 456656.6562 ;
vdfb('c_LightMnfc','a_HeavyMnfc','SEAsia') = 21218.23633 ;
vdfb('c_LightMnfc','a_HeavyMnfc','SouthAsia') = 19289.67773 ;
vdfb('c_LightMnfc','a_HeavyMnfc','NAmerica') = 148847.8594 ;
vdfb('c_LightMnfc','a_HeavyMnfc','LatinAmer') = 36633.42969 ;
vdfb('c_LightMnfc','a_HeavyMnfc','EU_28') = 250072.8906 ;
vdfb('c_LightMnfc','a_HeavyMnfc','MENA') = 16019.53418 ;
vdfb('c_LightMnfc','a_HeavyMnfc','SSA') = 8638.387695 ;
vdfb('c_LightMnfc','a_HeavyMnfc','RestofWorld') = 20401.40039 ;
vdfb('c_LightMnfc','a_Util_Cons','Oceania') = 23025.88867 ;
vdfb('c_LightMnfc','a_Util_Cons','EastAsia') = 228615.8594 ;
vdfb('c_LightMnfc','a_Util_Cons','SEAsia') = 59217.83203 ;
vdfb('c_LightMnfc','a_Util_Cons','SouthAsia') = 33745.16797 ;
vdfb('c_LightMnfc','a_Util_Cons','NAmerica') = 241858.9531 ;
vdfb('c_LightMnfc','a_Util_Cons','LatinAmer') = 34755.51562 ;
vdfb('c_LightMnfc','a_Util_Cons','EU_28') = 162927.6875 ;
vdfb('c_LightMnfc','a_Util_Cons','MENA') = 41160.29297 ;
vdfb('c_LightMnfc','a_Util_Cons','SSA') = 10317.53613 ;
vdfb('c_LightMnfc','a_Util_Cons','RestofWorld') = 30421.99023 ;
vdfb('c_LightMnfc','a_TransComm','Oceania') = 16472.73242 ;
vdfb('c_LightMnfc','a_TransComm','EastAsia') = 148979.6562 ;
vdfb('c_LightMnfc','a_TransComm','SEAsia') = 24940.38867 ;
vdfb('c_LightMnfc','a_TransComm','SouthAsia') = 25351.31836 ;
vdfb('c_LightMnfc','a_TransComm','NAmerica') = 144365.5156 ;
vdfb('c_LightMnfc','a_TransComm','LatinAmer') = 39673.62109 ;
vdfb('c_LightMnfc','a_TransComm','EU_28') = 126388.1016 ;
vdfb('c_LightMnfc','a_TransComm','MENA') = 17098.26562 ;
vdfb('c_LightMnfc','a_TransComm','SSA') = 10982.5459 ;
vdfb('c_LightMnfc','a_TransComm','RestofWorld') = 35366.52344 ;
vdfb('c_LightMnfc','a_OthService','Oceania') = 19116.80469 ;
vdfb('c_LightMnfc','a_OthService','EastAsia') = 295566.875 ;
vdfb('c_LightMnfc','a_OthService','SEAsia') = 20898.90234 ;
vdfb('c_LightMnfc','a_OthService','SouthAsia') = 12028.6377 ;
vdfb('c_LightMnfc','a_OthService','NAmerica') = 264757.25 ;
vdfb('c_LightMnfc','a_OthService','LatinAmer') = 55822 ;
vdfb('c_LightMnfc','a_OthService','EU_28') = 255832.8438 ;
vdfb('c_LightMnfc','a_OthService','MENA') = 22647.35938 ;
vdfb('c_LightMnfc','a_OthService','SSA') = 19085.07812 ;
vdfb('c_LightMnfc','a_OthService','RestofWorld') = 31776.25195 ;
vdfb('c_HeavyMnfc','a_agricultur','Oceania') = 4985.54541 ;
vdfb('c_HeavyMnfc','a_agricultur','EastAsia') = 82839.875 ;
vdfb('c_HeavyMnfc','a_agricultur','SEAsia') = 12178.69922 ;
vdfb('c_HeavyMnfc','a_agricultur','SouthAsia') = 34176.63281 ;
vdfb('c_HeavyMnfc','a_agricultur','NAmerica') = 47416.60156 ;
vdfb('c_HeavyMnfc','a_agricultur','LatinAmer') = 47962.13281 ;
vdfb('c_HeavyMnfc','a_agricultur','EU_28') = 33971.71094 ;
vdfb('c_HeavyMnfc','a_agricultur','MENA') = 14333.38574 ;
vdfb('c_HeavyMnfc','a_agricultur','SSA') = 5281.325684 ;
vdfb('c_HeavyMnfc','a_agricultur','RestofWorld') = 13175.88867 ;
vdfb('c_HeavyMnfc','a_Extraction','Oceania') = 6546.140625 ;
vdfb('c_HeavyMnfc','a_Extraction','EastAsia') = 124466.8203 ;
vdfb('c_HeavyMnfc','a_Extraction','SEAsia') = 6615.437012 ;
vdfb('c_HeavyMnfc','a_Extraction','SouthAsia') = 6996.931152 ;
vdfb('c_HeavyMnfc','a_Extraction','NAmerica') = 34842.96094 ;
vdfb('c_HeavyMnfc','a_Extraction','LatinAmer') = 23144.19727 ;
vdfb('c_HeavyMnfc','a_Extraction','EU_28') = 15146.66699 ;
vdfb('c_HeavyMnfc','a_Extraction','MENA') = 47525.03516 ;
vdfb('c_HeavyMnfc','a_Extraction','SSA') = 12039.29004 ;
vdfb('c_HeavyMnfc','a_Extraction','RestofWorld') = 28535.6582 ;
vdfb('c_HeavyMnfc','a_ProcFood','Oceania') = 4263.40332 ;
vdfb('c_HeavyMnfc','a_ProcFood','EastAsia') = 55506.65625 ;
vdfb('c_HeavyMnfc','a_ProcFood','SEAsia') = 5630.771484 ;
vdfb('c_HeavyMnfc','a_ProcFood','SouthAsia') = 6491.674316 ;
vdfb('c_HeavyMnfc','a_ProcFood','NAmerica') = 39304.79688 ;
vdfb('c_HeavyMnfc','a_ProcFood','LatinAmer') = 19813.22852 ;
vdfb('c_HeavyMnfc','a_ProcFood','EU_28') = 41812.95312 ;
vdfb('c_HeavyMnfc','a_ProcFood','MENA') = 8362.456055 ;
vdfb('c_HeavyMnfc','a_ProcFood','SSA') = 2712.712158 ;
vdfb('c_HeavyMnfc','a_ProcFood','RestofWorld') = 6901.190918 ;
vdfb('c_HeavyMnfc','a_TextWapp','Oceania') = 290.3639221 ;
vdfb('c_HeavyMnfc','a_TextWapp','EastAsia') = 78188.66406 ;
vdfb('c_HeavyMnfc','a_TextWapp','SEAsia') = 4719.075684 ;
vdfb('c_HeavyMnfc','a_TextWapp','SouthAsia') = 11468.82812 ;
vdfb('c_HeavyMnfc','a_TextWapp','NAmerica') = 29712.46289 ;
vdfb('c_HeavyMnfc','a_TextWapp','LatinAmer') = 6714.350586 ;
vdfb('c_HeavyMnfc','a_TextWapp','EU_28') = 19543.0293 ;
vdfb('c_HeavyMnfc','a_TextWapp','MENA') = 8296.327148 ;
vdfb('c_HeavyMnfc','a_TextWapp','SSA') = 1348.597046 ;
vdfb('c_HeavyMnfc','a_TextWapp','RestofWorld') = 1270.204834 ;
vdfb('c_HeavyMnfc','a_LightMnfc','Oceania') = 13425.4502 ;
vdfb('c_HeavyMnfc','a_LightMnfc','EastAsia') = 834940.9375 ;
vdfb('c_HeavyMnfc','a_LightMnfc','SEAsia') = 53438.00781 ;
vdfb('c_HeavyMnfc','a_LightMnfc','SouthAsia') = 44917.96094 ;
vdfb('c_HeavyMnfc','a_LightMnfc','NAmerica') = 356371 ;
vdfb('c_HeavyMnfc','a_LightMnfc','LatinAmer') = 84758.375 ;
vdfb('c_HeavyMnfc','a_LightMnfc','EU_28') = 316918.1875 ;
vdfb('c_HeavyMnfc','a_LightMnfc','MENA') = 40472.91016 ;
vdfb('c_HeavyMnfc','a_LightMnfc','SSA') = 19170.12109 ;
vdfb('c_HeavyMnfc','a_LightMnfc','RestofWorld') = 35118.75391 ;
vdfb('c_HeavyMnfc','a_HeavyMnfc','Oceania') = 34277.40234 ;
vdfb('c_HeavyMnfc','a_HeavyMnfc','EastAsia') = 4263423 ;
vdfb('c_HeavyMnfc','a_HeavyMnfc','SEAsia') = 211003 ;
vdfb('c_HeavyMnfc','a_HeavyMnfc','SouthAsia') = 186971.4062 ;
vdfb('c_HeavyMnfc','a_HeavyMnfc','NAmerica') = 1054827.75 ;
vdfb('c_HeavyMnfc','a_HeavyMnfc','LatinAmer') = 256198.1406 ;
vdfb('c_HeavyMnfc','a_HeavyMnfc','EU_28') = 1089224.75 ;
vdfb('c_HeavyMnfc','a_HeavyMnfc','MENA') = 162747.5 ;
vdfb('c_HeavyMnfc','a_HeavyMnfc','SSA') = 46861.50781 ;
vdfb('c_HeavyMnfc','a_HeavyMnfc','RestofWorld') = 221399.6875 ;
vdfb('c_HeavyMnfc','a_Util_Cons','Oceania') = 29097.99805 ;
vdfb('c_HeavyMnfc','a_Util_Cons','EastAsia') = 987110.375 ;
vdfb('c_HeavyMnfc','a_Util_Cons','SEAsia') = 66259.23438 ;
vdfb('c_HeavyMnfc','a_Util_Cons','SouthAsia') = 94125.10938 ;
vdfb('c_HeavyMnfc','a_Util_Cons','NAmerica') = 309930.1875 ;
vdfb('c_HeavyMnfc','a_Util_Cons','LatinAmer') = 111314.3906 ;
vdfb('c_HeavyMnfc','a_Util_Cons','EU_28') = 343244.5938 ;
vdfb('c_HeavyMnfc','a_Util_Cons','MENA') = 159670.2656 ;
vdfb('c_HeavyMnfc','a_Util_Cons','SSA') = 19519.8418 ;
vdfb('c_HeavyMnfc','a_Util_Cons','RestofWorld') = 111896.6953 ;
vdfb('c_HeavyMnfc','a_TransComm','Oceania') = 28577.48633 ;
vdfb('c_HeavyMnfc','a_TransComm','EastAsia') = 336468.9062 ;
vdfb('c_HeavyMnfc','a_TransComm','SEAsia') = 63740.625 ;
vdfb('c_HeavyMnfc','a_TransComm','SouthAsia') = 82398.71094 ;
vdfb('c_HeavyMnfc','a_TransComm','NAmerica') = 405780.75 ;
vdfb('c_HeavyMnfc','a_TransComm','LatinAmer') = 89427.67969 ;
vdfb('c_HeavyMnfc','a_TransComm','EU_28') = 429537.4375 ;
vdfb('c_HeavyMnfc','a_TransComm','MENA') = 86126.83594 ;
vdfb('c_HeavyMnfc','a_TransComm','SSA') = 23221.41406 ;
vdfb('c_HeavyMnfc','a_TransComm','RestofWorld') = 99756.02344 ;
vdfb('c_HeavyMnfc','a_OthService','Oceania') = 14513.69238 ;
vdfb('c_HeavyMnfc','a_OthService','EastAsia') = 475545.0625 ;
vdfb('c_HeavyMnfc','a_OthService','SEAsia') = 21315.375 ;
vdfb('c_HeavyMnfc','a_OthService','SouthAsia') = 15692.17676 ;
vdfb('c_HeavyMnfc','a_OthService','NAmerica') = 315642.8125 ;
vdfb('c_HeavyMnfc','a_OthService','LatinAmer') = 50961.69531 ;
vdfb('c_HeavyMnfc','a_OthService','EU_28') = 244990.5 ;
vdfb('c_HeavyMnfc','a_OthService','MENA') = 39070.52344 ;
vdfb('c_HeavyMnfc','a_OthService','SSA') = 15265.01465 ;
vdfb('c_HeavyMnfc','a_OthService','RestofWorld') = 38877.23828 ;
vdfb('c_Util_Cons','a_agricultur','Oceania') = 2619.497314 ;
vdfb('c_Util_Cons','a_agricultur','EastAsia') = 8825.669922 ;
vdfb('c_Util_Cons','a_agricultur','SEAsia') = 1638.331665 ;
vdfb('c_Util_Cons','a_agricultur','SouthAsia') = 23931.27344 ;
vdfb('c_Util_Cons','a_agricultur','NAmerica') = 20983.29102 ;
vdfb('c_Util_Cons','a_agricultur','LatinAmer') = 6281.439941 ;
vdfb('c_Util_Cons','a_agricultur','EU_28') = 16017.4668 ;
vdfb('c_Util_Cons','a_agricultur','MENA') = 8695.84375 ;
vdfb('c_Util_Cons','a_agricultur','SSA') = 1279.027954 ;
vdfb('c_Util_Cons','a_agricultur','RestofWorld') = 11982.34375 ;
vdfb('c_Util_Cons','a_Extraction','Oceania') = 12645.67578 ;
vdfb('c_Util_Cons','a_Extraction','EastAsia') = 16935.00586 ;
vdfb('c_Util_Cons','a_Extraction','SEAsia') = 3099.071777 ;
vdfb('c_Util_Cons','a_Extraction','SouthAsia') = 5536.528809 ;
vdfb('c_Util_Cons','a_Extraction','NAmerica') = 27459.20312 ;
vdfb('c_Util_Cons','a_Extraction','LatinAmer') = 11171.56641 ;
vdfb('c_Util_Cons','a_Extraction','EU_28') = 11630.90918 ;
vdfb('c_Util_Cons','a_Extraction','MENA') = 5100.960449 ;
vdfb('c_Util_Cons','a_Extraction','SSA') = 5468.611328 ;
vdfb('c_Util_Cons','a_Extraction','RestofWorld') = 30124.32031 ;
vdfb('c_Util_Cons','a_ProcFood','Oceania') = 1670.523804 ;
vdfb('c_Util_Cons','a_ProcFood','EastAsia') = 13556.91602 ;
vdfb('c_Util_Cons','a_ProcFood','SEAsia') = 3144.860107 ;
vdfb('c_Util_Cons','a_ProcFood','SouthAsia') = 3378.519531 ;
vdfb('c_Util_Cons','a_ProcFood','NAmerica') = 15971.00098 ;
vdfb('c_Util_Cons','a_ProcFood','LatinAmer') = 5490.743164 ;
vdfb('c_Util_Cons','a_ProcFood','EU_28') = 28458.39062 ;
vdfb('c_Util_Cons','a_ProcFood','MENA') = 3637.137939 ;
vdfb('c_Util_Cons','a_ProcFood','SSA') = 856.493103 ;
vdfb('c_Util_Cons','a_ProcFood','RestofWorld') = 9770.507812 ;
vdfb('c_Util_Cons','a_TextWapp','Oceania') = 202.322403 ;
vdfb('c_Util_Cons','a_TextWapp','EastAsia') = 14803.48828 ;
vdfb('c_Util_Cons','a_TextWapp','SEAsia') = 2248.027344 ;
vdfb('c_Util_Cons','a_TextWapp','SouthAsia') = 3766.852295 ;
vdfb('c_Util_Cons','a_TextWapp','NAmerica') = 5631.739746 ;
vdfb('c_Util_Cons','a_TextWapp','LatinAmer') = 2130.476807 ;
vdfb('c_Util_Cons','a_TextWapp','EU_28') = 7800.883789 ;
vdfb('c_Util_Cons','a_TextWapp','MENA') = 3600.439453 ;
vdfb('c_Util_Cons','a_TextWapp','SSA') = 244.9682159 ;
vdfb('c_Util_Cons','a_TextWapp','RestofWorld') = 1705.964478 ;
vdfb('c_Util_Cons','a_LightMnfc','Oceania') = 2437.904785 ;
vdfb('c_Util_Cons','a_LightMnfc','EastAsia') = 46949.57422 ;
vdfb('c_Util_Cons','a_LightMnfc','SEAsia') = 6742.680176 ;
vdfb('c_Util_Cons','a_LightMnfc','SouthAsia') = 14309.33887 ;
vdfb('c_Util_Cons','a_LightMnfc','NAmerica') = 50868.03125 ;
vdfb('c_Util_Cons','a_LightMnfc','LatinAmer') = 10828.4668 ;
vdfb('c_Util_Cons','a_LightMnfc','EU_28') = 65400.23828 ;
vdfb('c_Util_Cons','a_LightMnfc','MENA') = 8307.592773 ;
vdfb('c_Util_Cons','a_LightMnfc','SSA') = 1728.988281 ;
vdfb('c_Util_Cons','a_LightMnfc','RestofWorld') = 14719.37305 ;
vdfb('c_Util_Cons','a_HeavyMnfc','Oceania') = 13071.37598 ;
vdfb('c_Util_Cons','a_HeavyMnfc','EastAsia') = 229356.5 ;
vdfb('c_Util_Cons','a_HeavyMnfc','SEAsia') = 25545.55859 ;
vdfb('c_Util_Cons','a_HeavyMnfc','SouthAsia') = 41593.57422 ;
vdfb('c_Util_Cons','a_HeavyMnfc','NAmerica') = 128371.6641 ;
vdfb('c_Util_Cons','a_HeavyMnfc','LatinAmer') = 33716.26953 ;
vdfb('c_Util_Cons','a_HeavyMnfc','EU_28') = 153968.75 ;
vdfb('c_Util_Cons','a_HeavyMnfc','MENA') = 38509.99219 ;
vdfb('c_Util_Cons','a_HeavyMnfc','SSA') = 7456.799316 ;
vdfb('c_Util_Cons','a_HeavyMnfc','RestofWorld') = 76028.45312 ;
vdfb('c_Util_Cons','a_Util_Cons','Oceania') = 97355.34375 ;
vdfb('c_Util_Cons','a_Util_Cons','EastAsia') = 86273.29688 ;
vdfb('c_Util_Cons','a_Util_Cons','SEAsia') = 45826.82031 ;
vdfb('c_Util_Cons','a_Util_Cons','SouthAsia') = 33555.83594 ;
vdfb('c_Util_Cons','a_Util_Cons','NAmerica') = 159986.5469 ;
vdfb('c_Util_Cons','a_Util_Cons','LatinAmer') = 23644.55469 ;
vdfb('c_Util_Cons','a_Util_Cons','EU_28') = 363616.4688 ;
vdfb('c_Util_Cons','a_Util_Cons','MENA') = 98967.00781 ;
vdfb('c_Util_Cons','a_Util_Cons','SSA') = 25312.3125 ;
vdfb('c_Util_Cons','a_Util_Cons','RestofWorld') = 87291.90625 ;
vdfb('c_Util_Cons','a_TransComm','Oceania') = 17100.55469 ;
vdfb('c_Util_Cons','a_TransComm','EastAsia') = 91172.46875 ;
vdfb('c_Util_Cons','a_TransComm','SEAsia') = 14277.09473 ;
vdfb('c_Util_Cons','a_TransComm','SouthAsia') = 25106.59961 ;
vdfb('c_Util_Cons','a_TransComm','NAmerica') = 173554.0156 ;
vdfb('c_Util_Cons','a_TransComm','LatinAmer') = 21232.85938 ;
vdfb('c_Util_Cons','a_TransComm','EU_28') = 108216.2344 ;
vdfb('c_Util_Cons','a_TransComm','MENA') = 23665.62305 ;
vdfb('c_Util_Cons','a_TransComm','SSA') = 5588.770508 ;
vdfb('c_Util_Cons','a_TransComm','RestofWorld') = 42433.62109 ;
vdfb('c_Util_Cons','a_OthService','Oceania') = 35674.41406 ;
vdfb('c_Util_Cons','a_OthService','EastAsia') = 188520.25 ;
vdfb('c_Util_Cons','a_OthService','SEAsia') = 32101.2207 ;
vdfb('c_Util_Cons','a_OthService','SouthAsia') = 21233.0918 ;
vdfb('c_Util_Cons','a_OthService','NAmerica') = 448282.0312 ;
vdfb('c_Util_Cons','a_OthService','LatinAmer') = 61782.32812 ;
vdfb('c_Util_Cons','a_OthService','EU_28') = 372281.5 ;
vdfb('c_Util_Cons','a_OthService','MENA') = 45446.13281 ;
vdfb('c_Util_Cons','a_OthService','SSA') = 10024.90723 ;
vdfb('c_Util_Cons','a_OthService','RestofWorld') = 73461.88281 ;
vdfb('c_TransComm','a_agricultur','Oceania') = 9050.982422 ;
vdfb('c_TransComm','a_agricultur','EastAsia') = 61774.18359 ;
vdfb('c_TransComm','a_agricultur','SEAsia') = 13922.2373 ;
vdfb('c_TransComm','a_agricultur','SouthAsia') = 52642.82422 ;
vdfb('c_TransComm','a_agricultur','NAmerica') = 61950.20703 ;
vdfb('c_TransComm','a_agricultur','LatinAmer') = 48097.17578 ;
vdfb('c_TransComm','a_agricultur','EU_28') = 23877.38477 ;
vdfb('c_TransComm','a_agricultur','MENA') = 23742.73242 ;
vdfb('c_TransComm','a_agricultur','SSA') = 32727.49805 ;
vdfb('c_TransComm','a_agricultur','RestofWorld') = 27418.91992 ;
vdfb('c_TransComm','a_Extraction','Oceania') = 14581.36621 ;
vdfb('c_TransComm','a_Extraction','EastAsia') = 62978.72656 ;
vdfb('c_TransComm','a_Extraction','SEAsia') = 12781.80859 ;
vdfb('c_TransComm','a_Extraction','SouthAsia') = 11098.26758 ;
vdfb('c_TransComm','a_Extraction','NAmerica') = 43025.82812 ;
vdfb('c_TransComm','a_Extraction','LatinAmer') = 38344.30078 ;
vdfb('c_TransComm','a_Extraction','EU_28') = 18528.69336 ;
vdfb('c_TransComm','a_Extraction','MENA') = 22239.0293 ;
vdfb('c_TransComm','a_Extraction','SSA') = 22471.7168 ;
vdfb('c_TransComm','a_Extraction','RestofWorld') = 106036.3906 ;
vdfb('c_TransComm','a_ProcFood','Oceania') = 15210.24414 ;
vdfb('c_TransComm','a_ProcFood','EastAsia') = 106642.2969 ;
vdfb('c_TransComm','a_ProcFood','SEAsia') = 31043.39844 ;
vdfb('c_TransComm','a_ProcFood','SouthAsia') = 50428.46094 ;
vdfb('c_TransComm','a_ProcFood','NAmerica') = 102438.0078 ;
vdfb('c_TransComm','a_ProcFood','LatinAmer') = 57960.49609 ;
vdfb('c_TransComm','a_ProcFood','EU_28') = 86661.94531 ;
vdfb('c_TransComm','a_ProcFood','MENA') = 21881.53906 ;
vdfb('c_TransComm','a_ProcFood','SSA') = 26163.31836 ;
vdfb('c_TransComm','a_ProcFood','RestofWorld') = 35499.72266 ;
vdfb('c_TransComm','a_TextWapp','Oceania') = 1617.430786 ;
vdfb('c_TransComm','a_TextWapp','EastAsia') = 56060.71094 ;
vdfb('c_TransComm','a_TextWapp','SEAsia') = 8560.920898 ;
vdfb('c_TransComm','a_TextWapp','SouthAsia') = 30205.1875 ;
vdfb('c_TransComm','a_TextWapp','NAmerica') = 27107.46289 ;
vdfb('c_TransComm','a_TextWapp','LatinAmer') = 14339.35352 ;
vdfb('c_TransComm','a_TextWapp','EU_28') = 29006.92188 ;
vdfb('c_TransComm','a_TextWapp','MENA') = 16119.88379 ;
vdfb('c_TransComm','a_TextWapp','SSA') = 10769.2373 ;
vdfb('c_TransComm','a_TextWapp','RestofWorld') = 3757.285645 ;
vdfb('c_TransComm','a_LightMnfc','Oceania') = 19635.59961 ;
vdfb('c_TransComm','a_LightMnfc','EastAsia') = 280268.4062 ;
vdfb('c_TransComm','a_LightMnfc','SEAsia') = 38251.50391 ;
vdfb('c_TransComm','a_LightMnfc','SouthAsia') = 38667.40234 ;
vdfb('c_TransComm','a_LightMnfc','NAmerica') = 280398.7812 ;
vdfb('c_TransComm','a_LightMnfc','LatinAmer') = 65370.57422 ;
vdfb('c_TransComm','a_LightMnfc','EU_28') = 194040.1406 ;
vdfb('c_TransComm','a_LightMnfc','MENA') = 30192.69531 ;
vdfb('c_TransComm','a_LightMnfc','SSA') = 32861.30859 ;
vdfb('c_TransComm','a_LightMnfc','RestofWorld') = 31210.01172 ;
vdfb('c_TransComm','a_HeavyMnfc','Oceania') = 23267.43945 ;
vdfb('c_TransComm','a_HeavyMnfc','EastAsia') = 685105 ;
vdfb('c_TransComm','a_HeavyMnfc','SEAsia') = 91107.1875 ;
vdfb('c_TransComm','a_HeavyMnfc','SouthAsia') = 74332.82812 ;
vdfb('c_TransComm','a_HeavyMnfc','NAmerica') = 477227.125 ;
vdfb('c_TransComm','a_HeavyMnfc','LatinAmer') = 109712.9688 ;
vdfb('c_TransComm','a_HeavyMnfc','EU_28') = 274902.2812 ;
vdfb('c_TransComm','a_HeavyMnfc','MENA') = 55068.39453 ;
vdfb('c_TransComm','a_HeavyMnfc','SSA') = 59561.04688 ;
vdfb('c_TransComm','a_HeavyMnfc','RestofWorld') = 82782.78906 ;
vdfb('c_TransComm','a_Util_Cons','Oceania') = 29898.00781 ;
vdfb('c_TransComm','a_Util_Cons','EastAsia') = 375016.7188 ;
vdfb('c_TransComm','a_Util_Cons','SEAsia') = 43263.70703 ;
vdfb('c_TransComm','a_Util_Cons','SouthAsia') = 74868.92969 ;
vdfb('c_TransComm','a_Util_Cons','NAmerica') = 243169.9844 ;
vdfb('c_TransComm','a_Util_Cons','LatinAmer') = 42760.36328 ;
vdfb('c_TransComm','a_Util_Cons','EU_28') = 107242.5625 ;
vdfb('c_TransComm','a_Util_Cons','MENA') = 40799.53906 ;
vdfb('c_TransComm','a_Util_Cons','SSA') = 11452.86914 ;
vdfb('c_TransComm','a_Util_Cons','RestofWorld') = 111015.3281 ;
vdfb('c_TransComm','a_TransComm','Oceania') = 80716.39844 ;
vdfb('c_TransComm','a_TransComm','EastAsia') = 707011.1875 ;
vdfb('c_TransComm','a_TransComm','SEAsia') = 111887.1094 ;
vdfb('c_TransComm','a_TransComm','SouthAsia') = 75135.52344 ;
vdfb('c_TransComm','a_TransComm','NAmerica') = 561736.6875 ;
vdfb('c_TransComm','a_TransComm','LatinAmer') = 182236.0938 ;
vdfb('c_TransComm','a_TransComm','EU_28') = 903565.6875 ;
vdfb('c_TransComm','a_TransComm','MENA') = 118081.4609 ;
vdfb('c_TransComm','a_TransComm','SSA') = 52160.76562 ;
vdfb('c_TransComm','a_TransComm','RestofWorld') = 202312.0781 ;
vdfb('c_TransComm','a_OthService','Oceania') = 84614.60938 ;
vdfb('c_TransComm','a_OthService','EastAsia') = 677090.875 ;
vdfb('c_TransComm','a_OthService','SEAsia') = 55663.96875 ;
vdfb('c_TransComm','a_OthService','SouthAsia') = 50629.35547 ;
vdfb('c_TransComm','a_OthService','NAmerica') = 845234 ;
vdfb('c_TransComm','a_OthService','LatinAmer') = 159137.4219 ;
vdfb('c_TransComm','a_OthService','EU_28') = 570649.375 ;
vdfb('c_TransComm','a_OthService','MENA') = 66376.9375 ;
vdfb('c_TransComm','a_OthService','SSA') = 49923.73047 ;
vdfb('c_TransComm','a_OthService','RestofWorld') = 104502.0391 ;
vdfb('c_OthService','a_agricultur','Oceania') = 11518.8252 ;
vdfb('c_OthService','a_agricultur','EastAsia') = 40640.63281 ;
vdfb('c_OthService','a_agricultur','SEAsia') = 5819.188477 ;
vdfb('c_OthService','a_agricultur','SouthAsia') = 5183.357422 ;
vdfb('c_OthService','a_agricultur','NAmerica') = 68749.5 ;
vdfb('c_OthService','a_agricultur','LatinAmer') = 16721.86328 ;
vdfb('c_OthService','a_agricultur','EU_28') = 59158.30469 ;
vdfb('c_OthService','a_agricultur','MENA') = 8869.477539 ;
vdfb('c_OthService','a_agricultur','SSA') = 4700.718262 ;
vdfb('c_OthService','a_agricultur','RestofWorld') = 7050.386719 ;
vdfb('c_OthService','a_Extraction','Oceania') = 26037.96094 ;
vdfb('c_OthService','a_Extraction','EastAsia') = 39808.69922 ;
vdfb('c_OthService','a_Extraction','SEAsia') = 7768.344727 ;
vdfb('c_OthService','a_Extraction','SouthAsia') = 4057.423584 ;
vdfb('c_OthService','a_Extraction','NAmerica') = 51410.61719 ;
vdfb('c_OthService','a_Extraction','LatinAmer') = 27641.05078 ;
vdfb('c_OthService','a_Extraction','EU_28') = 26202.9082 ;
vdfb('c_OthService','a_Extraction','MENA') = 9260.858398 ;
vdfb('c_OthService','a_Extraction','SSA') = 22046.32617 ;
vdfb('c_OthService','a_Extraction','RestofWorld') = 54072.17188 ;
vdfb('c_OthService','a_ProcFood','Oceania') = 8603.916016 ;
vdfb('c_OthService','a_ProcFood','EastAsia') = 62373.90625 ;
vdfb('c_OthService','a_ProcFood','SEAsia') = 4505.785156 ;
vdfb('c_OthService','a_ProcFood','SouthAsia') = 8502.751953 ;
vdfb('c_OthService','a_ProcFood','NAmerica') = 75373.82031 ;
vdfb('c_OthService','a_ProcFood','LatinAmer') = 23057.31641 ;
vdfb('c_OthService','a_ProcFood','EU_28') = 161190.4531 ;
vdfb('c_OthService','a_ProcFood','MENA') = 7023.847656 ;
vdfb('c_OthService','a_ProcFood','SSA') = 6580.920898 ;
vdfb('c_OthService','a_ProcFood','RestofWorld') = 12191.64551 ;
vdfb('c_OthService','a_TextWapp','Oceania') = 967.1235352 ;
vdfb('c_OthService','a_TextWapp','EastAsia') = 50570.01562 ;
vdfb('c_OthService','a_TextWapp','SEAsia') = 2408.391602 ;
vdfb('c_OthService','a_TextWapp','SouthAsia') = 21201.47656 ;
vdfb('c_OthService','a_TextWapp','NAmerica') = 27203.85547 ;
vdfb('c_OthService','a_TextWapp','LatinAmer') = 7746.94873 ;
vdfb('c_OthService','a_TextWapp','EU_28') = 44454.17578 ;
vdfb('c_OthService','a_TextWapp','MENA') = 8686.09375 ;
vdfb('c_OthService','a_TextWapp','SSA') = 2521.415771 ;
vdfb('c_OthService','a_TextWapp','RestofWorld') = 1463.967773 ;
vdfb('c_OthService','a_LightMnfc','Oceania') = 14467.31641 ;
vdfb('c_OthService','a_LightMnfc','EastAsia') = 196491.5312 ;
vdfb('c_OthService','a_LightMnfc','SEAsia') = 9567.856445 ;
vdfb('c_OthService','a_LightMnfc','SouthAsia') = 27546.75 ;
vdfb('c_OthService','a_LightMnfc','NAmerica') = 163793.5938 ;
vdfb('c_OthService','a_LightMnfc','LatinAmer') = 27968.45117 ;
vdfb('c_OthService','a_LightMnfc','EU_28') = 368275.1875 ;
vdfb('c_OthService','a_LightMnfc','MENA') = 13777.91895 ;
vdfb('c_OthService','a_LightMnfc','SSA') = 6053.545898 ;
vdfb('c_OthService','a_LightMnfc','RestofWorld') = 19912.23633 ;
vdfb('c_OthService','a_HeavyMnfc','Oceania') = 16782.96289 ;
vdfb('c_OthService','a_HeavyMnfc','EastAsia') = 563994.875 ;
vdfb('c_OthService','a_HeavyMnfc','SEAsia') = 29268.58203 ;
vdfb('c_OthService','a_HeavyMnfc','SouthAsia') = 29970.14258 ;
vdfb('c_OthService','a_HeavyMnfc','NAmerica') = 311448.7812 ;
vdfb('c_OthService','a_HeavyMnfc','LatinAmer') = 57734.28906 ;
vdfb('c_OthService','a_HeavyMnfc','EU_28') = 540680.25 ;
vdfb('c_OthService','a_HeavyMnfc','MENA') = 33043.50781 ;
vdfb('c_OthService','a_HeavyMnfc','SSA') = 15300.62598 ;
vdfb('c_OthService','a_HeavyMnfc','RestofWorld') = 49545.07422 ;
vdfb('c_OthService','a_Util_Cons','Oceania') = 62235.04297 ;
vdfb('c_OthService','a_Util_Cons','EastAsia') = 228156.8906 ;
vdfb('c_OthService','a_Util_Cons','SEAsia') = 20751.625 ;
vdfb('c_OthService','a_Util_Cons','SouthAsia') = 27253.28125 ;
vdfb('c_OthService','a_Util_Cons','NAmerica') = 276582.7188 ;
vdfb('c_OthService','a_Util_Cons','LatinAmer') = 38963.05859 ;
vdfb('c_OthService','a_Util_Cons','EU_28') = 324736.7812 ;
vdfb('c_OthService','a_Util_Cons','MENA') = 28673.12891 ;
vdfb('c_OthService','a_Util_Cons','SSA') = 11543.62207 ;
vdfb('c_OthService','a_Util_Cons','RestofWorld') = 33447.28516 ;
vdfb('c_OthService','a_TransComm','Oceania') = 97229.29688 ;
vdfb('c_OthService','a_TransComm','EastAsia') = 748386.9375 ;
vdfb('c_OthService','a_TransComm','SEAsia') = 67143.82031 ;
vdfb('c_OthService','a_TransComm','SouthAsia') = 43608.41406 ;
vdfb('c_OthService','a_TransComm','NAmerica') = 868193 ;
vdfb('c_OthService','a_TransComm','LatinAmer') = 164626.3906 ;
vdfb('c_OthService','a_TransComm','EU_28') = 815477.5 ;
vdfb('c_OthService','a_TransComm','MENA') = 82739.25 ;
vdfb('c_OthService','a_TransComm','SSA') = 40348.08984 ;
vdfb('c_OthService','a_TransComm','RestofWorld') = 114967.5625 ;
vdfb('c_OthService','a_OthService','Oceania') = 280280.8125 ;
vdfb('c_OthService','a_OthService','EastAsia') = 1208575 ;
vdfb('c_OthService','a_OthService','SEAsia') = 144945.5625 ;
vdfb('c_OthService','a_OthService','SouthAsia') = 51936.91406 ;
vdfb('c_OthService','a_OthService','NAmerica') = 2409757.75 ;
vdfb('c_OthService','a_OthService','LatinAmer') = 315059.8125 ;
vdfb('c_OthService','a_OthService','EU_28') = 2579228.25 ;
vdfb('c_OthService','a_OthService','MENA') = 147006.2656 ;
vdfb('c_OthService','a_OthService','SSA') = 90797.54688 ;
vdfb('c_OthService','a_OthService','RestofWorld') = 263717.8125 ;

* vdfp data (900 cells)
vdfp('c_Crops','a_agricultur','Oceania') = 3722.90625 ;
vdfp('c_Crops','a_agricultur','EastAsia') = 160982 ;
vdfp('c_Crops','a_agricultur','SEAsia') = 63792.99609 ;
vdfp('c_Crops','a_agricultur','SouthAsia') = 95996.64062 ;
vdfp('c_Crops','a_agricultur','NAmerica') = 48722.08594 ;
vdfp('c_Crops','a_agricultur','LatinAmer') = 40260.12891 ;
vdfp('c_Crops','a_agricultur','EU_28') = 34348.5 ;
vdfp('c_Crops','a_agricultur','MENA') = 33741.60547 ;
vdfp('c_Crops','a_agricultur','SSA') = 15169.58887 ;
vdfp('c_Crops','a_agricultur','RestofWorld') = 19445.57227 ;
vdfp('c_Crops','a_Extraction','Oceania') = 221.7974548 ;
vdfp('c_Crops','a_Extraction','EastAsia') = 3727.342529 ;
vdfp('c_Crops','a_Extraction','SEAsia') = 2011.128662 ;
vdfp('c_Crops','a_Extraction','SouthAsia') = 59.17321777 ;
vdfp('c_Crops','a_Extraction','NAmerica') = 218.1413269 ;
vdfp('c_Crops','a_Extraction','LatinAmer') = 523.1860962 ;
vdfp('c_Crops','a_Extraction','EU_28') = 493.4533386 ;
vdfp('c_Crops','a_Extraction','MENA') = 1089.545898 ;
vdfp('c_Crops','a_Extraction','SSA') = 181.1891937 ;
vdfp('c_Crops','a_Extraction','RestofWorld') = 211.2624207 ;
vdfp('c_Crops','a_ProcFood','Oceania') = 5237.226562 ;
vdfp('c_Crops','a_ProcFood','EastAsia') = 233467.0625 ;
vdfp('c_Crops','a_ProcFood','SEAsia') = 53751.44922 ;
vdfp('c_Crops','a_ProcFood','SouthAsia') = 80290.70312 ;
vdfp('c_Crops','a_ProcFood','NAmerica') = 65400.38281 ;
vdfp('c_Crops','a_ProcFood','LatinAmer') = 92361.22656 ;
vdfp('c_Crops','a_ProcFood','EU_28') = 61727.5 ;
vdfp('c_Crops','a_ProcFood','MENA') = 32990.07812 ;
vdfp('c_Crops','a_ProcFood','SSA') = 17862.77734 ;
vdfp('c_Crops','a_ProcFood','RestofWorld') = 24755.69727 ;
vdfp('c_Crops','a_TextWapp','Oceania') = 17.05568314 ;
vdfp('c_Crops','a_TextWapp','EastAsia') = 45226.73438 ;
vdfp('c_Crops','a_TextWapp','SEAsia') = 689.6853027 ;
vdfp('c_Crops','a_TextWapp','SouthAsia') = 19039.51758 ;
vdfp('c_Crops','a_TextWapp','NAmerica') = 792.9057007 ;
vdfp('c_Crops','a_TextWapp','LatinAmer') = 4567.870605 ;
vdfp('c_Crops','a_TextWapp','EU_28') = 5076.521484 ;
vdfp('c_Crops','a_TextWapp','MENA') = 2539.617676 ;
vdfp('c_Crops','a_TextWapp','SSA') = 1714.110718 ;
vdfp('c_Crops','a_TextWapp','RestofWorld') = 482.0223389 ;
vdfp('c_Crops','a_LightMnfc','Oceania') = 53.71229935 ;
vdfp('c_Crops','a_LightMnfc','EastAsia') = 14153.25195 ;
vdfp('c_Crops','a_LightMnfc','SEAsia') = 3355.315186 ;
vdfp('c_Crops','a_LightMnfc','SouthAsia') = 832.0418701 ;
vdfp('c_Crops','a_LightMnfc','NAmerica') = 124.3728333 ;
vdfp('c_Crops','a_LightMnfc','LatinAmer') = 450.2002869 ;
vdfp('c_Crops','a_LightMnfc','EU_28') = 827.2999268 ;
vdfp('c_Crops','a_LightMnfc','MENA') = 477.2406311 ;
vdfp('c_Crops','a_LightMnfc','SSA') = 1628.617676 ;
vdfp('c_Crops','a_LightMnfc','RestofWorld') = 574.84021 ;
vdfp('c_Crops','a_HeavyMnfc','Oceania') = 125.3504562 ;
vdfp('c_Crops','a_HeavyMnfc','EastAsia') = 17574.45703 ;
vdfp('c_Crops','a_HeavyMnfc','SEAsia') = 5904.19043 ;
vdfp('c_Crops','a_HeavyMnfc','SouthAsia') = 5532.426758 ;
vdfp('c_Crops','a_HeavyMnfc','NAmerica') = 1911.028809 ;
vdfp('c_Crops','a_HeavyMnfc','LatinAmer') = 9765.49707 ;
vdfp('c_Crops','a_HeavyMnfc','EU_28') = 2123.051758 ;
vdfp('c_Crops','a_HeavyMnfc','MENA') = 743.9385986 ;
vdfp('c_Crops','a_HeavyMnfc','SSA') = 869.9567871 ;
vdfp('c_Crops','a_HeavyMnfc','RestofWorld') = 381.3477478 ;
vdfp('c_Crops','a_Util_Cons','Oceania') = 941.6237793 ;
vdfp('c_Crops','a_Util_Cons','EastAsia') = 5759.796387 ;
vdfp('c_Crops','a_Util_Cons','SEAsia') = 186.5285187 ;
vdfp('c_Crops','a_Util_Cons','SouthAsia') = 3533.35791 ;
vdfp('c_Crops','a_Util_Cons','NAmerica') = 4083.849121 ;
vdfp('c_Crops','a_Util_Cons','LatinAmer') = 276.1221619 ;
vdfp('c_Crops','a_Util_Cons','EU_28') = 3162.33252 ;
vdfp('c_Crops','a_Util_Cons','MENA') = 183.006134 ;
vdfp('c_Crops','a_Util_Cons','SSA') = 126.5093079 ;
vdfp('c_Crops','a_Util_Cons','RestofWorld') = 316.7387695 ;
vdfp('c_Crops','a_TransComm','Oceania') = 2180.503662 ;
vdfp('c_Crops','a_TransComm','EastAsia') = 37983.46875 ;
vdfp('c_Crops','a_TransComm','SEAsia') = 10528.8916 ;
vdfp('c_Crops','a_TransComm','SouthAsia') = 17434.64844 ;
vdfp('c_Crops','a_TransComm','NAmerica') = 4944.286621 ;
vdfp('c_Crops','a_TransComm','LatinAmer') = 5338.305664 ;
vdfp('c_Crops','a_TransComm','EU_28') = 11315.05469 ;
vdfp('c_Crops','a_TransComm','MENA') = 2946.374756 ;
vdfp('c_Crops','a_TransComm','SSA') = 1320.024414 ;
vdfp('c_Crops','a_TransComm','RestofWorld') = 2799.597656 ;
vdfp('c_Crops','a_OthService','Oceania') = 1928.473755 ;
vdfp('c_Crops','a_OthService','EastAsia') = 15332.75 ;
vdfp('c_Crops','a_OthService','SEAsia') = 3263.268311 ;
vdfp('c_Crops','a_OthService','SouthAsia') = 1049.137085 ;
vdfp('c_Crops','a_OthService','NAmerica') = 11399.17285 ;
vdfp('c_Crops','a_OthService','LatinAmer') = 2656.846924 ;
vdfp('c_Crops','a_OthService','EU_28') = 10221.12305 ;
vdfp('c_Crops','a_OthService','MENA') = 5922.895996 ;
vdfp('c_Crops','a_OthService','SSA') = 916.8139038 ;
vdfp('c_Crops','a_OthService','RestofWorld') = 2770.646973 ;
vdfp('c_MeatLstk','a_agricultur','Oceania') = 14328.08984 ;
vdfp('c_MeatLstk','a_agricultur','EastAsia') = 158954.9062 ;
vdfp('c_MeatLstk','a_agricultur','SEAsia') = 26545.91211 ;
vdfp('c_MeatLstk','a_agricultur','SouthAsia') = 20510.75586 ;
vdfp('c_MeatLstk','a_agricultur','NAmerica') = 140997.0312 ;
vdfp('c_MeatLstk','a_agricultur','LatinAmer') = 92835.89062 ;
vdfp('c_MeatLstk','a_agricultur','EU_28') = 90023.05469 ;
vdfp('c_MeatLstk','a_agricultur','MENA') = 29780.83789 ;
vdfp('c_MeatLstk','a_agricultur','SSA') = 22953.79883 ;
vdfp('c_MeatLstk','a_agricultur','RestofWorld') = 55597.05078 ;
vdfp('c_MeatLstk','a_Extraction','Oceania') = 41.32938766 ;
vdfp('c_MeatLstk','a_Extraction','EastAsia') = 297.8728943 ;
vdfp('c_MeatLstk','a_Extraction','SEAsia') = 55.12163925 ;
vdfp('c_MeatLstk','a_Extraction','SouthAsia') = 45.27746582 ;
vdfp('c_MeatLstk','a_Extraction','NAmerica') = 479.4246216 ;
vdfp('c_MeatLstk','a_Extraction','LatinAmer') = 260.9805603 ;
vdfp('c_MeatLstk','a_Extraction','EU_28') = 222.3279877 ;
vdfp('c_MeatLstk','a_Extraction','MENA') = 51.0668335 ;
vdfp('c_MeatLstk','a_Extraction','SSA') = 103.5401535 ;
vdfp('c_MeatLstk','a_Extraction','RestofWorld') = 171.7084656 ;
vdfp('c_MeatLstk','a_ProcFood','Oceania') = 12302.08203 ;
vdfp('c_MeatLstk','a_ProcFood','EastAsia') = 60435.28516 ;
vdfp('c_MeatLstk','a_ProcFood','SEAsia') = 5994.856934 ;
vdfp('c_MeatLstk','a_ProcFood','SouthAsia') = 8731.431641 ;
vdfp('c_MeatLstk','a_ProcFood','NAmerica') = 50382.29297 ;
vdfp('c_MeatLstk','a_ProcFood','LatinAmer') = 31220.2207 ;
vdfp('c_MeatLstk','a_ProcFood','EU_28') = 84470.96094 ;
vdfp('c_MeatLstk','a_ProcFood','MENA') = 22446.88672 ;
vdfp('c_MeatLstk','a_ProcFood','SSA') = 3643.663574 ;
vdfp('c_MeatLstk','a_ProcFood','RestofWorld') = 10668.29297 ;
vdfp('c_MeatLstk','a_TextWapp','Oceania') = 438.839447 ;
vdfp('c_MeatLstk','a_TextWapp','EastAsia') = 17085.30078 ;
vdfp('c_MeatLstk','a_TextWapp','SEAsia') = 400.0744629 ;
vdfp('c_MeatLstk','a_TextWapp','SouthAsia') = 653.149231 ;
vdfp('c_MeatLstk','a_TextWapp','NAmerica') = 989.9855347 ;
vdfp('c_MeatLstk','a_TextWapp','LatinAmer') = 1054.987183 ;
vdfp('c_MeatLstk','a_TextWapp','EU_28') = 194.2825928 ;
vdfp('c_MeatLstk','a_TextWapp','MENA') = 1998.054321 ;
vdfp('c_MeatLstk','a_TextWapp','SSA') = 309.1931152 ;
vdfp('c_MeatLstk','a_TextWapp','RestofWorld') = 852.3637695 ;
vdfp('c_MeatLstk','a_LightMnfc','Oceania') = 254.1936951 ;
vdfp('c_MeatLstk','a_LightMnfc','EastAsia') = 62372.93359 ;
vdfp('c_MeatLstk','a_LightMnfc','SEAsia') = 911.1380005 ;
vdfp('c_MeatLstk','a_LightMnfc','SouthAsia') = 10547.08008 ;
vdfp('c_MeatLstk','a_LightMnfc','NAmerica') = 2049.386719 ;
vdfp('c_MeatLstk','a_LightMnfc','LatinAmer') = 3683.966797 ;
vdfp('c_MeatLstk','a_LightMnfc','EU_28') = 951.7039795 ;
vdfp('c_MeatLstk','a_LightMnfc','MENA') = 554.7333374 ;
vdfp('c_MeatLstk','a_LightMnfc','SSA') = 693.4594116 ;
vdfp('c_MeatLstk','a_LightMnfc','RestofWorld') = 521.5037231 ;
vdfp('c_MeatLstk','a_HeavyMnfc','Oceania') = 1208.727295 ;
vdfp('c_MeatLstk','a_HeavyMnfc','EastAsia') = 21611.66797 ;
vdfp('c_MeatLstk','a_HeavyMnfc','SEAsia') = 540.510498 ;
vdfp('c_MeatLstk','a_HeavyMnfc','SouthAsia') = 835.3009644 ;
vdfp('c_MeatLstk','a_HeavyMnfc','NAmerica') = 1617.355713 ;
vdfp('c_MeatLstk','a_HeavyMnfc','LatinAmer') = 1984.729492 ;
vdfp('c_MeatLstk','a_HeavyMnfc','EU_28') = 1870.143066 ;
vdfp('c_MeatLstk','a_HeavyMnfc','MENA') = 622.3815918 ;
vdfp('c_MeatLstk','a_HeavyMnfc','SSA') = 159.1750031 ;
vdfp('c_MeatLstk','a_HeavyMnfc','RestofWorld') = 1124.252441 ;
vdfp('c_MeatLstk','a_Util_Cons','Oceania') = 137.4409485 ;
vdfp('c_MeatLstk','a_Util_Cons','EastAsia') = 12.15774059 ;
vdfp('c_MeatLstk','a_Util_Cons','SEAsia') = 2.244171858 ;
vdfp('c_MeatLstk','a_Util_Cons','SouthAsia') = 254.0172272 ;
vdfp('c_MeatLstk','a_Util_Cons','NAmerica') = 79.60671997 ;
vdfp('c_MeatLstk','a_Util_Cons','LatinAmer') = 138.1144257 ;
vdfp('c_MeatLstk','a_Util_Cons','EU_28') = 419.7877502 ;
vdfp('c_MeatLstk','a_Util_Cons','MENA') = 108.2214508 ;
vdfp('c_MeatLstk','a_Util_Cons','SSA') = 209.0470123 ;
vdfp('c_MeatLstk','a_Util_Cons','RestofWorld') = 146.5032501 ;
vdfp('c_MeatLstk','a_TransComm','Oceania') = 4828.513184 ;
vdfp('c_MeatLstk','a_TransComm','EastAsia') = 50959.01953 ;
vdfp('c_MeatLstk','a_TransComm','SEAsia') = 8350.331055 ;
vdfp('c_MeatLstk','a_TransComm','SouthAsia') = 7292.710449 ;
vdfp('c_MeatLstk','a_TransComm','NAmerica') = 13875.85352 ;
vdfp('c_MeatLstk','a_TransComm','LatinAmer') = 9892.211914 ;
vdfp('c_MeatLstk','a_TransComm','EU_28') = 18359.38867 ;
vdfp('c_MeatLstk','a_TransComm','MENA') = 3927.497559 ;
vdfp('c_MeatLstk','a_TransComm','SSA') = 1804.403564 ;
vdfp('c_MeatLstk','a_TransComm','RestofWorld') = 4531.586914 ;
vdfp('c_MeatLstk','a_OthService','Oceania') = 1739.408325 ;
vdfp('c_MeatLstk','a_OthService','EastAsia') = 6588.685059 ;
vdfp('c_MeatLstk','a_OthService','SEAsia') = 1417.445068 ;
vdfp('c_MeatLstk','a_OthService','SouthAsia') = 173.1059723 ;
vdfp('c_MeatLstk','a_OthService','NAmerica') = 31561.83203 ;
vdfp('c_MeatLstk','a_OthService','LatinAmer') = 5075.908691 ;
vdfp('c_MeatLstk','a_OthService','EU_28') = 10810.80371 ;
vdfp('c_MeatLstk','a_OthService','MENA') = 2913.481934 ;
vdfp('c_MeatLstk','a_OthService','SSA') = 1641.198975 ;
vdfp('c_MeatLstk','a_OthService','RestofWorld') = 5336.687988 ;
vdfp('c_Extraction','a_agricultur','Oceania') = 639.6345825 ;
vdfp('c_Extraction','a_agricultur','EastAsia') = 1866.611572 ;
vdfp('c_Extraction','a_agricultur','SEAsia') = 1712.104858 ;
vdfp('c_Extraction','a_agricultur','SouthAsia') = 493.5296631 ;
vdfp('c_Extraction','a_agricultur','NAmerica') = 1244.604126 ;
vdfp('c_Extraction','a_agricultur','LatinAmer') = 2770.14624 ;
vdfp('c_Extraction','a_agricultur','EU_28') = 1944.726318 ;
vdfp('c_Extraction','a_agricultur','MENA') = 301.7339478 ;
vdfp('c_Extraction','a_agricultur','SSA') = 963.7654419 ;
vdfp('c_Extraction','a_agricultur','RestofWorld') = 1505.761841 ;
vdfp('c_Extraction','a_Extraction','Oceania') = 17495.53906 ;
vdfp('c_Extraction','a_Extraction','EastAsia') = 56137.87891 ;
vdfp('c_Extraction','a_Extraction','SEAsia') = 8577.868164 ;
vdfp('c_Extraction','a_Extraction','SouthAsia') = 2761.17041 ;
vdfp('c_Extraction','a_Extraction','NAmerica') = 13902.29004 ;
vdfp('c_Extraction','a_Extraction','LatinAmer') = 19506.73047 ;
vdfp('c_Extraction','a_Extraction','EU_28') = 11271.58691 ;
vdfp('c_Extraction','a_Extraction','MENA') = 10604.23438 ;
vdfp('c_Extraction','a_Extraction','SSA') = 3072.095215 ;
vdfp('c_Extraction','a_Extraction','RestofWorld') = 22771.7832 ;
vdfp('c_Extraction','a_ProcFood','Oceania') = 1847.289185 ;
vdfp('c_Extraction','a_ProcFood','EastAsia') = 68862.125 ;
vdfp('c_Extraction','a_ProcFood','SEAsia') = 13706.98047 ;
vdfp('c_Extraction','a_ProcFood','SouthAsia') = 3503.158203 ;
vdfp('c_Extraction','a_ProcFood','NAmerica') = 5798.108398 ;
vdfp('c_Extraction','a_ProcFood','LatinAmer') = 6975.477539 ;
vdfp('c_Extraction','a_ProcFood','EU_28') = 7221.185547 ;
vdfp('c_Extraction','a_ProcFood','MENA') = 1730.616211 ;
vdfp('c_Extraction','a_ProcFood','SSA') = 6314.213867 ;
vdfp('c_Extraction','a_ProcFood','RestofWorld') = 7067.790527 ;
vdfp('c_Extraction','a_TextWapp','Oceania') = 33.06276703 ;
vdfp('c_Extraction','a_TextWapp','EastAsia') = 1124.899048 ;
vdfp('c_Extraction','a_TextWapp','SEAsia') = 201.4284058 ;
vdfp('c_Extraction','a_TextWapp','SouthAsia') = 93.86181641 ;
vdfp('c_Extraction','a_TextWapp','NAmerica') = 352.4555359 ;
vdfp('c_Extraction','a_TextWapp','LatinAmer') = 133.5513306 ;
vdfp('c_Extraction','a_TextWapp','EU_28') = 112.0280609 ;
vdfp('c_Extraction','a_TextWapp','MENA') = 180.7393494 ;
vdfp('c_Extraction','a_TextWapp','SSA') = 36.09587479 ;
vdfp('c_Extraction','a_TextWapp','RestofWorld') = 100.6375961 ;
vdfp('c_Extraction','a_LightMnfc','Oceania') = 2911.22168 ;
vdfp('c_Extraction','a_LightMnfc','EastAsia') = 47213.62109 ;
vdfp('c_Extraction','a_LightMnfc','SEAsia') = 11981.88574 ;
vdfp('c_Extraction','a_LightMnfc','SouthAsia') = 7401.641113 ;
vdfp('c_Extraction','a_LightMnfc','NAmerica') = 27998.48047 ;
vdfp('c_Extraction','a_LightMnfc','LatinAmer') = 13036.68164 ;
vdfp('c_Extraction','a_LightMnfc','EU_28') = 32092.44336 ;
vdfp('c_Extraction','a_LightMnfc','MENA') = 2647.503906 ;
vdfp('c_Extraction','a_LightMnfc','SSA') = 3856.231934 ;
vdfp('c_Extraction','a_LightMnfc','RestofWorld') = 11846.73047 ;
vdfp('c_Extraction','a_HeavyMnfc','Oceania') = 36703.09375 ;
vdfp('c_Extraction','a_HeavyMnfc','EastAsia') = 434870.9062 ;
vdfp('c_Extraction','a_HeavyMnfc','SEAsia') = 74118.66406 ;
vdfp('c_Extraction','a_HeavyMnfc','SouthAsia') = 37725.69141 ;
vdfp('c_Extraction','a_HeavyMnfc','NAmerica') = 339723.75 ;
vdfp('c_Extraction','a_HeavyMnfc','LatinAmer') = 140084.9688 ;
vdfp('c_Extraction','a_HeavyMnfc','EU_28') = 95472.33594 ;
vdfp('c_Extraction','a_HeavyMnfc','MENA') = 271296.2812 ;
vdfp('c_Extraction','a_HeavyMnfc','SSA') = 17830.10938 ;
vdfp('c_Extraction','a_HeavyMnfc','RestofWorld') = 256375.6406 ;
vdfp('c_Extraction','a_Util_Cons','Oceania') = 11880.98145 ;
vdfp('c_Extraction','a_Util_Cons','EastAsia') = 121945.6797 ;
vdfp('c_Extraction','a_Util_Cons','SEAsia') = 19541.11133 ;
vdfp('c_Extraction','a_Util_Cons','SouthAsia') = 20529.94141 ;
vdfp('c_Extraction','a_Util_Cons','NAmerica') = 88606.86719 ;
vdfp('c_Extraction','a_Util_Cons','LatinAmer') = 11858.46582 ;
vdfp('c_Extraction','a_Util_Cons','EU_28') = 41040.69531 ;
vdfp('c_Extraction','a_Util_Cons','MENA') = 61829.95312 ;
vdfp('c_Extraction','a_Util_Cons','SSA') = 7898.005859 ;
vdfp('c_Extraction','a_Util_Cons','RestofWorld') = 76905.39062 ;
vdfp('c_Extraction','a_TransComm','Oceania') = 1525.836914 ;
vdfp('c_Extraction','a_TransComm','EastAsia') = 26077.96289 ;
vdfp('c_Extraction','a_TransComm','SEAsia') = 2873.027344 ;
vdfp('c_Extraction','a_TransComm','SouthAsia') = 1453.826416 ;
vdfp('c_Extraction','a_TransComm','NAmerica') = 4247.684082 ;
vdfp('c_Extraction','a_TransComm','LatinAmer') = 1554.813477 ;
vdfp('c_Extraction','a_TransComm','EU_28') = 5610.680176 ;
vdfp('c_Extraction','a_TransComm','MENA') = 1137.032959 ;
vdfp('c_Extraction','a_TransComm','SSA') = 584.2724609 ;
vdfp('c_Extraction','a_TransComm','RestofWorld') = 13000.95312 ;
vdfp('c_Extraction','a_OthService','Oceania') = 1222.561279 ;
vdfp('c_Extraction','a_OthService','EastAsia') = 5662.04541 ;
vdfp('c_Extraction','a_OthService','SEAsia') = 752.4416504 ;
vdfp('c_Extraction','a_OthService','SouthAsia') = 65.81726074 ;
vdfp('c_Extraction','a_OthService','NAmerica') = 4183.362793 ;
vdfp('c_Extraction','a_OthService','LatinAmer') = 1180.918091 ;
vdfp('c_Extraction','a_OthService','EU_28') = 5033.84082 ;
vdfp('c_Extraction','a_OthService','MENA') = 1750.082153 ;
vdfp('c_Extraction','a_OthService','SSA') = 653.755127 ;
vdfp('c_Extraction','a_OthService','RestofWorld') = 4907.316895 ;
vdfp('c_ProcFood','a_agricultur','Oceania') = 1521.724976 ;
vdfp('c_ProcFood','a_agricultur','EastAsia') = 110188.9609 ;
vdfp('c_ProcFood','a_agricultur','SEAsia') = 11337.68457 ;
vdfp('c_ProcFood','a_agricultur','SouthAsia') = 3555.964355 ;
vdfp('c_ProcFood','a_agricultur','NAmerica') = 40568.37891 ;
vdfp('c_ProcFood','a_agricultur','LatinAmer') = 26738.70508 ;
vdfp('c_ProcFood','a_agricultur','EU_28') = 46911.57031 ;
vdfp('c_ProcFood','a_agricultur','MENA') = 8473.867188 ;
vdfp('c_ProcFood','a_agricultur','SSA') = 5113.370117 ;
vdfp('c_ProcFood','a_agricultur','RestofWorld') = 9560.169922 ;
vdfp('c_ProcFood','a_Extraction','Oceania') = 301.4434204 ;
vdfp('c_ProcFood','a_Extraction','EastAsia') = 14959.96777 ;
vdfp('c_ProcFood','a_Extraction','SEAsia') = 2120.954346 ;
vdfp('c_ProcFood','a_Extraction','SouthAsia') = 284.421051 ;
vdfp('c_ProcFood','a_Extraction','NAmerica') = 813.3590088 ;
vdfp('c_ProcFood','a_Extraction','LatinAmer') = 2068.260742 ;
vdfp('c_ProcFood','a_Extraction','EU_28') = 1096.479858 ;
vdfp('c_ProcFood','a_Extraction','MENA') = 672.8251953 ;
vdfp('c_ProcFood','a_Extraction','SSA') = 459.2012329 ;
vdfp('c_ProcFood','a_Extraction','RestofWorld') = 3091.88916 ;
vdfp('c_ProcFood','a_ProcFood','Oceania') = 11456.55664 ;
vdfp('c_ProcFood','a_ProcFood','EastAsia') = 218383.7344 ;
vdfp('c_ProcFood','a_ProcFood','SEAsia') = 50127.98828 ;
vdfp('c_ProcFood','a_ProcFood','SouthAsia') = 20692.48828 ;
vdfp('c_ProcFood','a_ProcFood','NAmerica') = 109490.7344 ;
vdfp('c_ProcFood','a_ProcFood','LatinAmer') = 76029.46875 ;
vdfp('c_ProcFood','a_ProcFood','EU_28') = 206280.7969 ;
vdfp('c_ProcFood','a_ProcFood','MENA') = 33048.23047 ;
vdfp('c_ProcFood','a_ProcFood','SSA') = 18179.55859 ;
vdfp('c_ProcFood','a_ProcFood','RestofWorld') = 50960.17969 ;
vdfp('c_ProcFood','a_TextWapp','Oceania') = 15.50391293 ;
vdfp('c_ProcFood','a_TextWapp','EastAsia') = 3768.780518 ;
vdfp('c_ProcFood','a_TextWapp','SEAsia') = 586.0960083 ;
vdfp('c_ProcFood','a_TextWapp','SouthAsia') = 439.5722961 ;
vdfp('c_ProcFood','a_TextWapp','NAmerica') = 122.0737686 ;
vdfp('c_ProcFood','a_TextWapp','LatinAmer') = 168.7750549 ;
vdfp('c_ProcFood','a_TextWapp','EU_28') = 2097.282227 ;
vdfp('c_ProcFood','a_TextWapp','MENA') = 442.1375427 ;
vdfp('c_ProcFood','a_TextWapp','SSA') = 162.1770935 ;
vdfp('c_ProcFood','a_TextWapp','RestofWorld') = 87.96647644 ;
vdfp('c_ProcFood','a_LightMnfc','Oceania') = 130.0000916 ;
vdfp('c_ProcFood','a_LightMnfc','EastAsia') = 11047.83594 ;
vdfp('c_ProcFood','a_LightMnfc','SEAsia') = 1020.03717 ;
vdfp('c_ProcFood','a_LightMnfc','SouthAsia') = 10491.9082 ;
vdfp('c_ProcFood','a_LightMnfc','NAmerica') = 1656.870605 ;
vdfp('c_ProcFood','a_LightMnfc','LatinAmer') = 618.8469238 ;
vdfp('c_ProcFood','a_LightMnfc','EU_28') = 10761.40527 ;
vdfp('c_ProcFood','a_LightMnfc','MENA') = 281.1493835 ;
vdfp('c_ProcFood','a_LightMnfc','SSA') = 752.888855 ;
vdfp('c_ProcFood','a_LightMnfc','RestofWorld') = 681.7566528 ;
vdfp('c_ProcFood','a_HeavyMnfc','Oceania') = 448.0777893 ;
vdfp('c_ProcFood','a_HeavyMnfc','EastAsia') = 42019.76562 ;
vdfp('c_ProcFood','a_HeavyMnfc','SEAsia') = 3762.131104 ;
vdfp('c_ProcFood','a_HeavyMnfc','SouthAsia') = 1845.952393 ;
vdfp('c_ProcFood','a_HeavyMnfc','NAmerica') = 2812.285889 ;
vdfp('c_ProcFood','a_HeavyMnfc','LatinAmer') = 3007.549072 ;
vdfp('c_ProcFood','a_HeavyMnfc','EU_28') = 14729.38867 ;
vdfp('c_ProcFood','a_HeavyMnfc','MENA') = 765.0623169 ;
vdfp('c_ProcFood','a_HeavyMnfc','SSA') = 958.8529663 ;
vdfp('c_ProcFood','a_HeavyMnfc','RestofWorld') = 994.9884644 ;
vdfp('c_ProcFood','a_Util_Cons','Oceania') = 148.4549866 ;
vdfp('c_ProcFood','a_Util_Cons','EastAsia') = 7124.958984 ;
vdfp('c_ProcFood','a_Util_Cons','SEAsia') = 111.7287827 ;
vdfp('c_ProcFood','a_Util_Cons','SouthAsia') = 74.60312653 ;
vdfp('c_ProcFood','a_Util_Cons','NAmerica') = 428.8509827 ;
vdfp('c_ProcFood','a_Util_Cons','LatinAmer') = 498.6388245 ;
vdfp('c_ProcFood','a_Util_Cons','EU_28') = 2463.806396 ;
vdfp('c_ProcFood','a_Util_Cons','MENA') = 393.9165039 ;
vdfp('c_ProcFood','a_Util_Cons','SSA') = 153.9747925 ;
vdfp('c_ProcFood','a_Util_Cons','RestofWorld') = 400.7962341 ;
vdfp('c_ProcFood','a_TransComm','Oceania') = 12546.83594 ;
vdfp('c_ProcFood','a_TransComm','EastAsia') = 149765.4062 ;
vdfp('c_ProcFood','a_TransComm','SEAsia') = 16219.36914 ;
vdfp('c_ProcFood','a_TransComm','SouthAsia') = 10012.2207 ;
vdfp('c_ProcFood','a_TransComm','NAmerica') = 31041.16992 ;
vdfp('c_ProcFood','a_TransComm','LatinAmer') = 41697.22656 ;
vdfp('c_ProcFood','a_TransComm','EU_28') = 120151.2578 ;
vdfp('c_ProcFood','a_TransComm','MENA') = 10123.86816 ;
vdfp('c_ProcFood','a_TransComm','SSA') = 4866.271484 ;
vdfp('c_ProcFood','a_TransComm','RestofWorld') = 11954.63672 ;
vdfp('c_ProcFood','a_OthService','Oceania') = 4264.098633 ;
vdfp('c_ProcFood','a_OthService','EastAsia') = 46339.83203 ;
vdfp('c_ProcFood','a_OthService','SEAsia') = 4206.466309 ;
vdfp('c_ProcFood','a_OthService','SouthAsia') = 638.6053467 ;
vdfp('c_ProcFood','a_OthService','NAmerica') = 125873.0703 ;
vdfp('c_ProcFood','a_OthService','LatinAmer') = 11480.24414 ;
vdfp('c_ProcFood','a_OthService','EU_28') = 50569.49219 ;
vdfp('c_ProcFood','a_OthService','MENA') = 14528.29199 ;
vdfp('c_ProcFood','a_OthService','SSA') = 6520.793945 ;
vdfp('c_ProcFood','a_OthService','RestofWorld') = 15370.88574 ;
vdfp('c_TextWapp','a_agricultur','Oceania') = 60.77814484 ;
vdfp('c_TextWapp','a_agricultur','EastAsia') = 633.4502563 ;
vdfp('c_TextWapp','a_agricultur','SEAsia') = 118.6026535 ;
vdfp('c_TextWapp','a_agricultur','SouthAsia') = 1716.267334 ;
vdfp('c_TextWapp','a_agricultur','NAmerica') = 1513.839722 ;
vdfp('c_TextWapp','a_agricultur','LatinAmer') = 830.1343994 ;
vdfp('c_TextWapp','a_agricultur','EU_28') = 469.217041 ;
vdfp('c_TextWapp','a_agricultur','MENA') = 791.2514038 ;
vdfp('c_TextWapp','a_agricultur','SSA') = 849.3063965 ;
vdfp('c_TextWapp','a_agricultur','RestofWorld') = 241.0216064 ;
vdfp('c_TextWapp','a_Extraction','Oceania') = 188.8171997 ;
vdfp('c_TextWapp','a_Extraction','EastAsia') = 3382.124023 ;
vdfp('c_TextWapp','a_Extraction','SEAsia') = 328.3035278 ;
vdfp('c_TextWapp','a_Extraction','SouthAsia') = 599.6270752 ;
vdfp('c_TextWapp','a_Extraction','NAmerica') = 667.2670898 ;
vdfp('c_TextWapp','a_Extraction','LatinAmer') = 930.7546387 ;
vdfp('c_TextWapp','a_Extraction','EU_28') = 491.6601868 ;
vdfp('c_TextWapp','a_Extraction','MENA') = 268.3081665 ;
vdfp('c_TextWapp','a_Extraction','SSA') = 591.1126099 ;
vdfp('c_TextWapp','a_Extraction','RestofWorld') = 495.288208 ;
vdfp('c_TextWapp','a_ProcFood','Oceania') = 86.1611557 ;
vdfp('c_TextWapp','a_ProcFood','EastAsia') = 2336.577637 ;
vdfp('c_TextWapp','a_ProcFood','SEAsia') = 200.7409821 ;
vdfp('c_TextWapp','a_ProcFood','SouthAsia') = 1769.76062 ;
vdfp('c_TextWapp','a_ProcFood','NAmerica') = 544.3461304 ;
vdfp('c_TextWapp','a_ProcFood','LatinAmer') = 1269.659668 ;
vdfp('c_TextWapp','a_ProcFood','EU_28') = 928.5420532 ;
vdfp('c_TextWapp','a_ProcFood','MENA') = 644.2787476 ;
vdfp('c_TextWapp','a_ProcFood','SSA') = 294.1923523 ;
vdfp('c_TextWapp','a_ProcFood','RestofWorld') = 340.6341553 ;
vdfp('c_TextWapp','a_TextWapp','Oceania') = 921.0551758 ;
vdfp('c_TextWapp','a_TextWapp','EastAsia') = 342781.5 ;
vdfp('c_TextWapp','a_TextWapp','SEAsia') = 29734.14844 ;
vdfp('c_TextWapp','a_TextWapp','SouthAsia') = 38505.54688 ;
vdfp('c_TextWapp','a_TextWapp','NAmerica') = 83691.15625 ;
vdfp('c_TextWapp','a_TextWapp','LatinAmer') = 39170.66406 ;
vdfp('c_TextWapp','a_TextWapp','EU_28') = 81187.97656 ;
vdfp('c_TextWapp','a_TextWapp','MENA') = 46827.98047 ;
vdfp('c_TextWapp','a_TextWapp','SSA') = 4359.197754 ;
vdfp('c_TextWapp','a_TextWapp','RestofWorld') = 8876.935547 ;
vdfp('c_TextWapp','a_LightMnfc','Oceania') = 331.6177673 ;
vdfp('c_TextWapp','a_LightMnfc','EastAsia') = 50692 ;
vdfp('c_TextWapp','a_LightMnfc','SEAsia') = 2863.040039 ;
vdfp('c_TextWapp','a_LightMnfc','SouthAsia') = 909.7306519 ;
vdfp('c_TextWapp','a_LightMnfc','NAmerica') = 21343.16406 ;
vdfp('c_TextWapp','a_LightMnfc','LatinAmer') = 4689.302734 ;
vdfp('c_TextWapp','a_LightMnfc','EU_28') = 13818.57227 ;
vdfp('c_TextWapp','a_LightMnfc','MENA') = 2062.363281 ;
vdfp('c_TextWapp','a_LightMnfc','SSA') = 1357.575317 ;
vdfp('c_TextWapp','a_LightMnfc','RestofWorld') = 1675.409668 ;
vdfp('c_TextWapp','a_HeavyMnfc','Oceania') = 230.303009 ;
vdfp('c_TextWapp','a_HeavyMnfc','EastAsia') = 35220.84375 ;
vdfp('c_TextWapp','a_HeavyMnfc','SEAsia') = 3244.368408 ;
vdfp('c_TextWapp','a_HeavyMnfc','SouthAsia') = 1741.541626 ;
vdfp('c_TextWapp','a_HeavyMnfc','NAmerica') = 9224.258789 ;
vdfp('c_TextWapp','a_HeavyMnfc','LatinAmer') = 2170.048584 ;
vdfp('c_TextWapp','a_HeavyMnfc','EU_28') = 7329.769043 ;
vdfp('c_TextWapp','a_HeavyMnfc','MENA') = 2084.987549 ;
vdfp('c_TextWapp','a_HeavyMnfc','SSA') = 307.9414978 ;
vdfp('c_TextWapp','a_HeavyMnfc','RestofWorld') = 1038.86145 ;
vdfp('c_TextWapp','a_Util_Cons','Oceania') = 390.5699463 ;
vdfp('c_TextWapp','a_Util_Cons','EastAsia') = 11371.31641 ;
vdfp('c_TextWapp','a_Util_Cons','SEAsia') = 402.1499939 ;
vdfp('c_TextWapp','a_Util_Cons','SouthAsia') = 392.8338013 ;
vdfp('c_TextWapp','a_Util_Cons','NAmerica') = 4602.054688 ;
vdfp('c_TextWapp','a_Util_Cons','LatinAmer') = 764.6971436 ;
vdfp('c_TextWapp','a_Util_Cons','EU_28') = 2550.137451 ;
vdfp('c_TextWapp','a_Util_Cons','MENA') = 712.3661499 ;
vdfp('c_TextWapp','a_Util_Cons','SSA') = 240.0709381 ;
vdfp('c_TextWapp','a_Util_Cons','RestofWorld') = 710.9806519 ;
vdfp('c_TextWapp','a_TransComm','Oceania') = 874.4945679 ;
vdfp('c_TextWapp','a_TransComm','EastAsia') = 25712.71875 ;
vdfp('c_TextWapp','a_TransComm','SEAsia') = 2604.530029 ;
vdfp('c_TextWapp','a_TransComm','SouthAsia') = 1632.225708 ;
vdfp('c_TextWapp','a_TransComm','NAmerica') = 11862.39453 ;
vdfp('c_TextWapp','a_TransComm','LatinAmer') = 4393.683594 ;
vdfp('c_TextWapp','a_TransComm','EU_28') = 10733.18848 ;
vdfp('c_TextWapp','a_TransComm','MENA') = 4199.804688 ;
vdfp('c_TextWapp','a_TransComm','SSA') = 1999.874634 ;
vdfp('c_TextWapp','a_TransComm','RestofWorld') = 2327.296143 ;
vdfp('c_TextWapp','a_OthService','Oceania') = 1177.235229 ;
vdfp('c_TextWapp','a_OthService','EastAsia') = 44274.47656 ;
vdfp('c_TextWapp','a_OthService','SEAsia') = 2610.343262 ;
vdfp('c_TextWapp','a_OthService','SouthAsia') = 1258.216187 ;
vdfp('c_TextWapp','a_OthService','NAmerica') = 12620.86914 ;
vdfp('c_TextWapp','a_OthService','LatinAmer') = 6842.498535 ;
vdfp('c_TextWapp','a_OthService','EU_28') = 12510.30957 ;
vdfp('c_TextWapp','a_OthService','MENA') = 8957.020508 ;
vdfp('c_TextWapp','a_OthService','SSA') = 2413.028564 ;
vdfp('c_TextWapp','a_OthService','RestofWorld') = 3709.221436 ;
vdfp('c_LightMnfc','a_agricultur','Oceania') = 624.3150635 ;
vdfp('c_LightMnfc','a_agricultur','EastAsia') = 8950.857422 ;
vdfp('c_LightMnfc','a_agricultur','SEAsia') = 1414.416626 ;
vdfp('c_LightMnfc','a_agricultur','SouthAsia') = 702.2893066 ;
vdfp('c_LightMnfc','a_agricultur','NAmerica') = 12460.78516 ;
vdfp('c_LightMnfc','a_agricultur','LatinAmer') = 6295.080078 ;
vdfp('c_LightMnfc','a_agricultur','EU_28') = 10590.54785 ;
vdfp('c_LightMnfc','a_agricultur','MENA') = 1836.288696 ;
vdfp('c_LightMnfc','a_agricultur','SSA') = 2292.55835 ;
vdfp('c_LightMnfc','a_agricultur','RestofWorld') = 3533.156006 ;
vdfp('c_LightMnfc','a_Extraction','Oceania') = 2979.126465 ;
vdfp('c_LightMnfc','a_Extraction','EastAsia') = 28403.73633 ;
vdfp('c_LightMnfc','a_Extraction','SEAsia') = 2801.329834 ;
vdfp('c_LightMnfc','a_Extraction','SouthAsia') = 1635.676758 ;
vdfp('c_LightMnfc','a_Extraction','NAmerica') = 7353.897949 ;
vdfp('c_LightMnfc','a_Extraction','LatinAmer') = 8898.429688 ;
vdfp('c_LightMnfc','a_Extraction','EU_28') = 5949.101562 ;
vdfp('c_LightMnfc','a_Extraction','MENA') = 3475.533691 ;
vdfp('c_LightMnfc','a_Extraction','SSA') = 3692.797119 ;
vdfp('c_LightMnfc','a_Extraction','RestofWorld') = 9959.209961 ;
vdfp('c_LightMnfc','a_ProcFood','Oceania') = 3198.479004 ;
vdfp('c_LightMnfc','a_ProcFood','EastAsia') = 47138.47266 ;
vdfp('c_LightMnfc','a_ProcFood','SEAsia') = 4832.607422 ;
vdfp('c_LightMnfc','a_ProcFood','SouthAsia') = 2990.355713 ;
vdfp('c_LightMnfc','a_ProcFood','NAmerica') = 54196.37891 ;
vdfp('c_LightMnfc','a_ProcFood','LatinAmer') = 12460.96387 ;
vdfp('c_LightMnfc','a_ProcFood','EU_28') = 48053.26562 ;
vdfp('c_LightMnfc','a_ProcFood','MENA') = 4856.526367 ;
vdfp('c_LightMnfc','a_ProcFood','SSA') = 2897.302734 ;
vdfp('c_LightMnfc','a_ProcFood','RestofWorld') = 7141.436035 ;
vdfp('c_LightMnfc','a_TextWapp','Oceania') = 234.4776154 ;
vdfp('c_LightMnfc','a_TextWapp','EastAsia') = 36370.55078 ;
vdfp('c_LightMnfc','a_TextWapp','SEAsia') = 2246.589111 ;
vdfp('c_LightMnfc','a_TextWapp','SouthAsia') = 2425.213379 ;
vdfp('c_LightMnfc','a_TextWapp','NAmerica') = 5284.316895 ;
vdfp('c_LightMnfc','a_TextWapp','LatinAmer') = 2698.482422 ;
vdfp('c_LightMnfc','a_TextWapp','EU_28') = 12909.91211 ;
vdfp('c_LightMnfc','a_TextWapp','MENA') = 3243.989746 ;
vdfp('c_LightMnfc','a_TextWapp','SSA') = 930.5673218 ;
vdfp('c_LightMnfc','a_TextWapp','RestofWorld') = 1015.223755 ;
vdfp('c_LightMnfc','a_LightMnfc','Oceania') = 16700.74219 ;
vdfp('c_LightMnfc','a_LightMnfc','EastAsia') = 1138957.375 ;
vdfp('c_LightMnfc','a_LightMnfc','SEAsia') = 51628.51172 ;
vdfp('c_LightMnfc','a_LightMnfc','SouthAsia') = 36565.11328 ;
vdfp('c_LightMnfc','a_LightMnfc','NAmerica') = 493404.375 ;
vdfp('c_LightMnfc','a_LightMnfc','LatinAmer') = 101487.0781 ;
vdfp('c_LightMnfc','a_LightMnfc','EU_28') = 697688.3125 ;
vdfp('c_LightMnfc','a_LightMnfc','MENA') = 45917.67578 ;
vdfp('c_LightMnfc','a_LightMnfc','SSA') = 25760.125 ;
vdfp('c_LightMnfc','a_LightMnfc','RestofWorld') = 46805.0625 ;
vdfp('c_LightMnfc','a_HeavyMnfc','Oceania') = 6335.320312 ;
vdfp('c_LightMnfc','a_HeavyMnfc','EastAsia') = 456656.6562 ;
vdfp('c_LightMnfc','a_HeavyMnfc','SEAsia') = 21218.23633 ;
vdfp('c_LightMnfc','a_HeavyMnfc','SouthAsia') = 19289.67773 ;
vdfp('c_LightMnfc','a_HeavyMnfc','NAmerica') = 148847.875 ;
vdfp('c_LightMnfc','a_HeavyMnfc','LatinAmer') = 36633.42969 ;
vdfp('c_LightMnfc','a_HeavyMnfc','EU_28') = 250072.8906 ;
vdfp('c_LightMnfc','a_HeavyMnfc','MENA') = 16019.53418 ;
vdfp('c_LightMnfc','a_HeavyMnfc','SSA') = 8638.387695 ;
vdfp('c_LightMnfc','a_HeavyMnfc','RestofWorld') = 20401.40039 ;
vdfp('c_LightMnfc','a_Util_Cons','Oceania') = 23025.88867 ;
vdfp('c_LightMnfc','a_Util_Cons','EastAsia') = 228615.8594 ;
vdfp('c_LightMnfc','a_Util_Cons','SEAsia') = 59217.83203 ;
vdfp('c_LightMnfc','a_Util_Cons','SouthAsia') = 33745.17578 ;
vdfp('c_LightMnfc','a_Util_Cons','NAmerica') = 241858.9531 ;
vdfp('c_LightMnfc','a_Util_Cons','LatinAmer') = 34755.51562 ;
vdfp('c_LightMnfc','a_Util_Cons','EU_28') = 162927.6875 ;
vdfp('c_LightMnfc','a_Util_Cons','MENA') = 41160.29297 ;
vdfp('c_LightMnfc','a_Util_Cons','SSA') = 10317.53613 ;
vdfp('c_LightMnfc','a_Util_Cons','RestofWorld') = 30421.99023 ;
vdfp('c_LightMnfc','a_TransComm','Oceania') = 16472.73242 ;
vdfp('c_LightMnfc','a_TransComm','EastAsia') = 148979.6562 ;
vdfp('c_LightMnfc','a_TransComm','SEAsia') = 24940.38867 ;
vdfp('c_LightMnfc','a_TransComm','SouthAsia') = 25351.31641 ;
vdfp('c_LightMnfc','a_TransComm','NAmerica') = 144365.5156 ;
vdfp('c_LightMnfc','a_TransComm','LatinAmer') = 39673.62109 ;
vdfp('c_LightMnfc','a_TransComm','EU_28') = 126388.1016 ;
vdfp('c_LightMnfc','a_TransComm','MENA') = 17098.26562 ;
vdfp('c_LightMnfc','a_TransComm','SSA') = 10982.5459 ;
vdfp('c_LightMnfc','a_TransComm','RestofWorld') = 35366.52344 ;
vdfp('c_LightMnfc','a_OthService','Oceania') = 19116.80469 ;
vdfp('c_LightMnfc','a_OthService','EastAsia') = 295566.875 ;
vdfp('c_LightMnfc','a_OthService','SEAsia') = 20898.90234 ;
vdfp('c_LightMnfc','a_OthService','SouthAsia') = 12028.6377 ;
vdfp('c_LightMnfc','a_OthService','NAmerica') = 264757.25 ;
vdfp('c_LightMnfc','a_OthService','LatinAmer') = 55822 ;
vdfp('c_LightMnfc','a_OthService','EU_28') = 255832.8438 ;
vdfp('c_LightMnfc','a_OthService','MENA') = 22647.35938 ;
vdfp('c_LightMnfc','a_OthService','SSA') = 19085.07812 ;
vdfp('c_LightMnfc','a_OthService','RestofWorld') = 31776.25195 ;
vdfp('c_HeavyMnfc','a_agricultur','Oceania') = 5136.541504 ;
vdfp('c_HeavyMnfc','a_agricultur','EastAsia') = 83680.38281 ;
vdfp('c_HeavyMnfc','a_agricultur','SEAsia') = 11980.74316 ;
vdfp('c_HeavyMnfc','a_agricultur','SouthAsia') = 35866.20312 ;
vdfp('c_HeavyMnfc','a_agricultur','NAmerica') = 48152.27344 ;
vdfp('c_HeavyMnfc','a_agricultur','LatinAmer') = 48163.27734 ;
vdfp('c_HeavyMnfc','a_agricultur','EU_28') = 37634.35547 ;
vdfp('c_HeavyMnfc','a_agricultur','MENA') = 16802.53906 ;
vdfp('c_HeavyMnfc','a_agricultur','SSA') = 5411.22168 ;
vdfp('c_HeavyMnfc','a_agricultur','RestofWorld') = 13722.61914 ;
vdfp('c_HeavyMnfc','a_Extraction','Oceania') = 6761.062988 ;
vdfp('c_HeavyMnfc','a_Extraction','EastAsia') = 126316.6094 ;
vdfp('c_HeavyMnfc','a_Extraction','SEAsia') = 6748.337891 ;
vdfp('c_HeavyMnfc','a_Extraction','SouthAsia') = 7746.762695 ;
vdfp('c_HeavyMnfc','a_Extraction','NAmerica') = 35442.62891 ;
vdfp('c_HeavyMnfc','a_Extraction','LatinAmer') = 23544.28906 ;
vdfp('c_HeavyMnfc','a_Extraction','EU_28') = 16842.94922 ;
vdfp('c_HeavyMnfc','a_Extraction','MENA') = 47773.40234 ;
vdfp('c_HeavyMnfc','a_Extraction','SSA') = 12208.08594 ;
vdfp('c_HeavyMnfc','a_Extraction','RestofWorld') = 28957.20117 ;
vdfp('c_HeavyMnfc','a_ProcFood','Oceania') = 4285.050781 ;
vdfp('c_HeavyMnfc','a_ProcFood','EastAsia') = 56298.71094 ;
vdfp('c_HeavyMnfc','a_ProcFood','SEAsia') = 5699.072754 ;
vdfp('c_HeavyMnfc','a_ProcFood','SouthAsia') = 6506.880859 ;
vdfp('c_HeavyMnfc','a_ProcFood','NAmerica') = 39703.69141 ;
vdfp('c_HeavyMnfc','a_ProcFood','LatinAmer') = 19873.41211 ;
vdfp('c_HeavyMnfc','a_ProcFood','EU_28') = 43255.92188 ;
vdfp('c_HeavyMnfc','a_ProcFood','MENA') = 8530.300781 ;
vdfp('c_HeavyMnfc','a_ProcFood','SSA') = 2723.310059 ;
vdfp('c_HeavyMnfc','a_ProcFood','RestofWorld') = 7043.266602 ;
vdfp('c_HeavyMnfc','a_TextWapp','Oceania') = 298.4064941 ;
vdfp('c_HeavyMnfc','a_TextWapp','EastAsia') = 78345.11719 ;
vdfp('c_HeavyMnfc','a_TextWapp','SEAsia') = 4726.25 ;
vdfp('c_HeavyMnfc','a_TextWapp','SouthAsia') = 11654.16992 ;
vdfp('c_HeavyMnfc','a_TextWapp','NAmerica') = 29772.02148 ;
vdfp('c_HeavyMnfc','a_TextWapp','LatinAmer') = 6729.001465 ;
vdfp('c_HeavyMnfc','a_TextWapp','EU_28') = 19759.33398 ;
vdfp('c_HeavyMnfc','a_TextWapp','MENA') = 8300.354492 ;
vdfp('c_HeavyMnfc','a_TextWapp','SSA') = 1352.0625 ;
vdfp('c_HeavyMnfc','a_TextWapp','RestofWorld') = 1287.390137 ;
vdfp('c_HeavyMnfc','a_LightMnfc','Oceania') = 13449.00879 ;
vdfp('c_HeavyMnfc','a_LightMnfc','EastAsia') = 836474.6875 ;
vdfp('c_HeavyMnfc','a_LightMnfc','SEAsia') = 53446.63281 ;
vdfp('c_HeavyMnfc','a_LightMnfc','SouthAsia') = 45137.85938 ;
vdfp('c_HeavyMnfc','a_LightMnfc','NAmerica') = 357508.0625 ;
vdfp('c_HeavyMnfc','a_LightMnfc','LatinAmer') = 84926.8125 ;
vdfp('c_HeavyMnfc','a_LightMnfc','EU_28') = 318429.4375 ;
vdfp('c_HeavyMnfc','a_LightMnfc','MENA') = 40650.55469 ;
vdfp('c_HeavyMnfc','a_LightMnfc','SSA') = 19193.49023 ;
vdfp('c_HeavyMnfc','a_LightMnfc','RestofWorld') = 35261.28516 ;
vdfp('c_HeavyMnfc','a_HeavyMnfc','Oceania') = 34667.55859 ;
vdfp('c_HeavyMnfc','a_HeavyMnfc','EastAsia') = 4293924.5 ;
vdfp('c_HeavyMnfc','a_HeavyMnfc','SEAsia') = 211979.2656 ;
vdfp('c_HeavyMnfc','a_HeavyMnfc','SouthAsia') = 193121.2344 ;
vdfp('c_HeavyMnfc','a_HeavyMnfc','NAmerica') = 1066142.375 ;
vdfp('c_HeavyMnfc','a_HeavyMnfc','LatinAmer') = 257699.8906 ;
vdfp('c_HeavyMnfc','a_HeavyMnfc','EU_28') = 1127183.5 ;
vdfp('c_HeavyMnfc','a_HeavyMnfc','MENA') = 164810.2969 ;
vdfp('c_HeavyMnfc','a_HeavyMnfc','SSA') = 47079.10938 ;
vdfp('c_HeavyMnfc','a_HeavyMnfc','RestofWorld') = 224363.1562 ;
vdfp('c_HeavyMnfc','a_Util_Cons','Oceania') = 29554.94922 ;
vdfp('c_HeavyMnfc','a_Util_Cons','EastAsia') = 991322 ;
vdfp('c_HeavyMnfc','a_Util_Cons','SEAsia') = 66271.33594 ;
vdfp('c_HeavyMnfc','a_Util_Cons','SouthAsia') = 95167.77344 ;
vdfp('c_HeavyMnfc','a_Util_Cons','NAmerica') = 311540.0625 ;
vdfp('c_HeavyMnfc','a_Util_Cons','LatinAmer') = 112835.3828 ;
vdfp('c_HeavyMnfc','a_Util_Cons','EU_28') = 350879.5 ;
vdfp('c_HeavyMnfc','a_Util_Cons','MENA') = 159773.6562 ;
vdfp('c_HeavyMnfc','a_Util_Cons','SSA') = 19902.00781 ;
vdfp('c_HeavyMnfc','a_Util_Cons','RestofWorld') = 120873.9297 ;
vdfp('c_HeavyMnfc','a_TransComm','Oceania') = 30777.74219 ;
vdfp('c_HeavyMnfc','a_TransComm','EastAsia') = 370154.8125 ;
vdfp('c_HeavyMnfc','a_TransComm','SEAsia') = 64892.35547 ;
vdfp('c_HeavyMnfc','a_TransComm','SouthAsia') = 95482.82812 ;
vdfp('c_HeavyMnfc','a_TransComm','NAmerica') = 450295.4375 ;
vdfp('c_HeavyMnfc','a_TransComm','LatinAmer') = 95753.16406 ;
vdfp('c_HeavyMnfc','a_TransComm','EU_28') = 568576.3125 ;
vdfp('c_HeavyMnfc','a_TransComm','MENA') = 97184.70312 ;
vdfp('c_HeavyMnfc','a_TransComm','SSA') = 25153.98633 ;
vdfp('c_HeavyMnfc','a_TransComm','RestofWorld') = 110597.1562 ;
vdfp('c_HeavyMnfc','a_OthService','Oceania') = 14637.62891 ;
vdfp('c_HeavyMnfc','a_OthService','EastAsia') = 481298.2188 ;
vdfp('c_HeavyMnfc','a_OthService','SEAsia') = 21405.47656 ;
vdfp('c_HeavyMnfc','a_OthService','SouthAsia') = 15970.18457 ;
vdfp('c_HeavyMnfc','a_OthService','NAmerica') = 317315.2812 ;
vdfp('c_HeavyMnfc','a_OthService','LatinAmer') = 51214.75781 ;
vdfp('c_HeavyMnfc','a_OthService','EU_28') = 251865.2656 ;
vdfp('c_HeavyMnfc','a_OthService','MENA') = 39627.84375 ;
vdfp('c_HeavyMnfc','a_OthService','SSA') = 15419.36816 ;
vdfp('c_HeavyMnfc','a_OthService','RestofWorld') = 39331.49219 ;
vdfp('c_Util_Cons','a_agricultur','Oceania') = 2607.68457 ;
vdfp('c_Util_Cons','a_agricultur','EastAsia') = 9169.424805 ;
vdfp('c_Util_Cons','a_agricultur','SEAsia') = 1648.475098 ;
vdfp('c_Util_Cons','a_agricultur','SouthAsia') = 22062.11719 ;
vdfp('c_Util_Cons','a_agricultur','NAmerica') = 20828.44531 ;
vdfp('c_Util_Cons','a_agricultur','LatinAmer') = 6616.249023 ;
vdfp('c_Util_Cons','a_agricultur','EU_28') = 16973.09375 ;
vdfp('c_Util_Cons','a_agricultur','MENA') = 8999.963867 ;
vdfp('c_Util_Cons','a_agricultur','SSA') = 1284.76062 ;
vdfp('c_Util_Cons','a_agricultur','RestofWorld') = 12436.7832 ;
vdfp('c_Util_Cons','a_Extraction','Oceania') = 12646.27051 ;
vdfp('c_Util_Cons','a_Extraction','EastAsia') = 17606.5 ;
vdfp('c_Util_Cons','a_Extraction','SEAsia') = 3117.230713 ;
vdfp('c_Util_Cons','a_Extraction','SouthAsia') = 5345.310059 ;
vdfp('c_Util_Cons','a_Extraction','NAmerica') = 28175.27539 ;
vdfp('c_Util_Cons','a_Extraction','LatinAmer') = 11506.98438 ;
vdfp('c_Util_Cons','a_Extraction','EU_28') = 12385.23242 ;
vdfp('c_Util_Cons','a_Extraction','MENA') = 5209.258301 ;
vdfp('c_Util_Cons','a_Extraction','SSA') = 5504.289551 ;
vdfp('c_Util_Cons','a_Extraction','RestofWorld') = 31013.84961 ;
vdfp('c_Util_Cons','a_ProcFood','Oceania') = 1675.783691 ;
vdfp('c_Util_Cons','a_ProcFood','EastAsia') = 14096.52246 ;
vdfp('c_Util_Cons','a_ProcFood','SEAsia') = 3191.273682 ;
vdfp('c_Util_Cons','a_ProcFood','SouthAsia') = 3265.001465 ;
vdfp('c_Util_Cons','a_ProcFood','NAmerica') = 16258.92773 ;
vdfp('c_Util_Cons','a_ProcFood','LatinAmer') = 5832.209961 ;
vdfp('c_Util_Cons','a_ProcFood','EU_28') = 31611.58008 ;
vdfp('c_Util_Cons','a_ProcFood','MENA') = 3858.507324 ;
vdfp('c_Util_Cons','a_ProcFood','SSA') = 863.4122925 ;
vdfp('c_Util_Cons','a_ProcFood','RestofWorld') = 10135.02051 ;
vdfp('c_Util_Cons','a_TextWapp','Oceania') = 202.450943 ;
vdfp('c_Util_Cons','a_TextWapp','EastAsia') = 15322.03906 ;
vdfp('c_Util_Cons','a_TextWapp','SEAsia') = 2262.704834 ;
vdfp('c_Util_Cons','a_TextWapp','SouthAsia') = 3617.036865 ;
vdfp('c_Util_Cons','a_TextWapp','NAmerica') = 5726.613281 ;
vdfp('c_Util_Cons','a_TextWapp','LatinAmer') = 2272.927734 ;
vdfp('c_Util_Cons','a_TextWapp','EU_28') = 8638.117188 ;
vdfp('c_Util_Cons','a_TextWapp','MENA') = 4017.870117 ;
vdfp('c_Util_Cons','a_TextWapp','SSA') = 250.7481689 ;
vdfp('c_Util_Cons','a_TextWapp','RestofWorld') = 1789.243896 ;
vdfp('c_Util_Cons','a_LightMnfc','Oceania') = 2440.382812 ;
vdfp('c_Util_Cons','a_LightMnfc','EastAsia') = 48655.85156 ;
vdfp('c_Util_Cons','a_LightMnfc','SEAsia') = 6805.878418 ;
vdfp('c_Util_Cons','a_LightMnfc','SouthAsia') = 13386.6875 ;
vdfp('c_Util_Cons','a_LightMnfc','NAmerica') = 52009.9375 ;
vdfp('c_Util_Cons','a_LightMnfc','LatinAmer') = 11491.24121 ;
vdfp('c_Util_Cons','a_LightMnfc','EU_28') = 73500.82031 ;
vdfp('c_Util_Cons','a_LightMnfc','MENA') = 8736.541992 ;
vdfp('c_Util_Cons','a_LightMnfc','SSA') = 1763.487671 ;
vdfp('c_Util_Cons','a_LightMnfc','RestofWorld') = 15661.80859 ;
vdfp('c_Util_Cons','a_HeavyMnfc','Oceania') = 13075.20801 ;
vdfp('c_Util_Cons','a_HeavyMnfc','EastAsia') = 238711.9688 ;
vdfp('c_Util_Cons','a_HeavyMnfc','SEAsia') = 25818.33203 ;
vdfp('c_Util_Cons','a_HeavyMnfc','SouthAsia') = 39109.91797 ;
vdfp('c_Util_Cons','a_HeavyMnfc','NAmerica') = 131447.7969 ;
vdfp('c_Util_Cons','a_HeavyMnfc','LatinAmer') = 36231.92188 ;
vdfp('c_Util_Cons','a_HeavyMnfc','EU_28') = 175954.2188 ;
vdfp('c_Util_Cons','a_HeavyMnfc','MENA') = 40539.45312 ;
vdfp('c_Util_Cons','a_HeavyMnfc','SSA') = 7563.975586 ;
vdfp('c_Util_Cons','a_HeavyMnfc','RestofWorld') = 80438.24219 ;
vdfp('c_Util_Cons','a_Util_Cons','Oceania') = 97354.47656 ;
vdfp('c_Util_Cons','a_Util_Cons','EastAsia') = 87740.96875 ;
vdfp('c_Util_Cons','a_Util_Cons','SEAsia') = 46295.27734 ;
vdfp('c_Util_Cons','a_Util_Cons','SouthAsia') = 33074.98828 ;
vdfp('c_Util_Cons','a_Util_Cons','NAmerica') = 161705.5625 ;
vdfp('c_Util_Cons','a_Util_Cons','LatinAmer') = 24030.18164 ;
vdfp('c_Util_Cons','a_Util_Cons','EU_28') = 365710.625 ;
vdfp('c_Util_Cons','a_Util_Cons','MENA') = 99327.60938 ;
vdfp('c_Util_Cons','a_Util_Cons','SSA') = 25354.17773 ;
vdfp('c_Util_Cons','a_Util_Cons','RestofWorld') = 88388.76562 ;
vdfp('c_Util_Cons','a_TransComm','Oceania') = 17101.58203 ;
vdfp('c_Util_Cons','a_TransComm','EastAsia') = 94075.54688 ;
vdfp('c_Util_Cons','a_TransComm','SEAsia') = 14452.00195 ;
vdfp('c_Util_Cons','a_TransComm','SouthAsia') = 23802.53125 ;
vdfp('c_Util_Cons','a_TransComm','NAmerica') = 175924.7188 ;
vdfp('c_Util_Cons','a_TransComm','LatinAmer') = 22148.63672 ;
vdfp('c_Util_Cons','a_TransComm','EU_28') = 116271.2656 ;
vdfp('c_Util_Cons','a_TransComm','MENA') = 24308.25391 ;
vdfp('c_Util_Cons','a_TransComm','SSA') = 5620.356445 ;
vdfp('c_Util_Cons','a_TransComm','RestofWorld') = 43668.02734 ;
vdfp('c_Util_Cons','a_OthService','Oceania') = 35675.91797 ;
vdfp('c_Util_Cons','a_OthService','EastAsia') = 192660.1719 ;
vdfp('c_Util_Cons','a_OthService','SEAsia') = 32339.62891 ;
vdfp('c_Util_Cons','a_OthService','SouthAsia') = 20602.88672 ;
vdfp('c_Util_Cons','a_OthService','NAmerica') = 452247.8438 ;
vdfp('c_Util_Cons','a_OthService','LatinAmer') = 62909.52734 ;
vdfp('c_Util_Cons','a_OthService','EU_28') = 387545.625 ;
vdfp('c_Util_Cons','a_OthService','MENA') = 46515.94922 ;
vdfp('c_Util_Cons','a_OthService','SSA') = 10064.87598 ;
vdfp('c_Util_Cons','a_OthService','RestofWorld') = 75147.23438 ;
vdfp('c_TransComm','a_agricultur','Oceania') = 8999.668945 ;
vdfp('c_TransComm','a_agricultur','EastAsia') = 61209.75 ;
vdfp('c_TransComm','a_agricultur','SEAsia') = 13647.00098 ;
vdfp('c_TransComm','a_agricultur','SouthAsia') = 52642.82422 ;
vdfp('c_TransComm','a_agricultur','NAmerica') = 60014.78516 ;
vdfp('c_TransComm','a_agricultur','LatinAmer') = 47700.51953 ;
vdfp('c_TransComm','a_agricultur','EU_28') = 23452.78711 ;
vdfp('c_TransComm','a_agricultur','MENA') = 23700.57227 ;
vdfp('c_TransComm','a_agricultur','SSA') = 32687.55664 ;
vdfp('c_TransComm','a_agricultur','RestofWorld') = 27070.79492 ;
vdfp('c_TransComm','a_Extraction','Oceania') = 14581.36621 ;
vdfp('c_TransComm','a_Extraction','EastAsia') = 62978.72656 ;
vdfp('c_TransComm','a_Extraction','SEAsia') = 12781.80859 ;
vdfp('c_TransComm','a_Extraction','SouthAsia') = 11098.26758 ;
vdfp('c_TransComm','a_Extraction','NAmerica') = 43025.82812 ;
vdfp('c_TransComm','a_Extraction','LatinAmer') = 38344.30078 ;
vdfp('c_TransComm','a_Extraction','EU_28') = 18528.69336 ;
vdfp('c_TransComm','a_Extraction','MENA') = 22239.0293 ;
vdfp('c_TransComm','a_Extraction','SSA') = 22471.7168 ;
vdfp('c_TransComm','a_Extraction','RestofWorld') = 106036.3906 ;
vdfp('c_TransComm','a_ProcFood','Oceania') = 15210.24414 ;
vdfp('c_TransComm','a_ProcFood','EastAsia') = 106642.2891 ;
vdfp('c_TransComm','a_ProcFood','SEAsia') = 31043.39648 ;
vdfp('c_TransComm','a_ProcFood','SouthAsia') = 50428.46094 ;
vdfp('c_TransComm','a_ProcFood','NAmerica') = 102438.0078 ;
vdfp('c_TransComm','a_ProcFood','LatinAmer') = 57960.49609 ;
vdfp('c_TransComm','a_ProcFood','EU_28') = 86661.94531 ;
vdfp('c_TransComm','a_ProcFood','MENA') = 21881.53906 ;
vdfp('c_TransComm','a_ProcFood','SSA') = 26163.31836 ;
vdfp('c_TransComm','a_ProcFood','RestofWorld') = 35499.72266 ;
vdfp('c_TransComm','a_TextWapp','Oceania') = 1617.430786 ;
vdfp('c_TransComm','a_TextWapp','EastAsia') = 56060.71094 ;
vdfp('c_TransComm','a_TextWapp','SEAsia') = 8560.920898 ;
vdfp('c_TransComm','a_TextWapp','SouthAsia') = 30205.1875 ;
vdfp('c_TransComm','a_TextWapp','NAmerica') = 27107.46289 ;
vdfp('c_TransComm','a_TextWapp','LatinAmer') = 14339.35352 ;
vdfp('c_TransComm','a_TextWapp','EU_28') = 29006.92188 ;
vdfp('c_TransComm','a_TextWapp','MENA') = 16119.88379 ;
vdfp('c_TransComm','a_TextWapp','SSA') = 10769.2373 ;
vdfp('c_TransComm','a_TextWapp','RestofWorld') = 3757.285889 ;
vdfp('c_TransComm','a_LightMnfc','Oceania') = 19635.59961 ;
vdfp('c_TransComm','a_LightMnfc','EastAsia') = 280268.4062 ;
vdfp('c_TransComm','a_LightMnfc','SEAsia') = 38251.50391 ;
vdfp('c_TransComm','a_LightMnfc','SouthAsia') = 38667.40234 ;
vdfp('c_TransComm','a_LightMnfc','NAmerica') = 280398.8125 ;
vdfp('c_TransComm','a_LightMnfc','LatinAmer') = 65370.57422 ;
vdfp('c_TransComm','a_LightMnfc','EU_28') = 194040.1406 ;
vdfp('c_TransComm','a_LightMnfc','MENA') = 30192.69531 ;
vdfp('c_TransComm','a_LightMnfc','SSA') = 32861.30859 ;
vdfp('c_TransComm','a_LightMnfc','RestofWorld') = 31210.01172 ;
vdfp('c_TransComm','a_HeavyMnfc','Oceania') = 23267.43945 ;
vdfp('c_TransComm','a_HeavyMnfc','EastAsia') = 685105 ;
vdfp('c_TransComm','a_HeavyMnfc','SEAsia') = 91107.1875 ;
vdfp('c_TransComm','a_HeavyMnfc','SouthAsia') = 74332.82812 ;
vdfp('c_TransComm','a_HeavyMnfc','NAmerica') = 477227.125 ;
vdfp('c_TransComm','a_HeavyMnfc','LatinAmer') = 109712.9688 ;
vdfp('c_TransComm','a_HeavyMnfc','EU_28') = 274902.2812 ;
vdfp('c_TransComm','a_HeavyMnfc','MENA') = 55068.39453 ;
vdfp('c_TransComm','a_HeavyMnfc','SSA') = 59561.04688 ;
vdfp('c_TransComm','a_HeavyMnfc','RestofWorld') = 82782.78906 ;
vdfp('c_TransComm','a_Util_Cons','Oceania') = 29898.00781 ;
vdfp('c_TransComm','a_Util_Cons','EastAsia') = 375016.6875 ;
vdfp('c_TransComm','a_Util_Cons','SEAsia') = 43263.70703 ;
vdfp('c_TransComm','a_Util_Cons','SouthAsia') = 74868.92969 ;
vdfp('c_TransComm','a_Util_Cons','NAmerica') = 243169.9688 ;
vdfp('c_TransComm','a_Util_Cons','LatinAmer') = 42760.36328 ;
vdfp('c_TransComm','a_Util_Cons','EU_28') = 107242.5625 ;
vdfp('c_TransComm','a_Util_Cons','MENA') = 40799.53516 ;
vdfp('c_TransComm','a_Util_Cons','SSA') = 11452.86914 ;
vdfp('c_TransComm','a_Util_Cons','RestofWorld') = 111015.3281 ;
vdfp('c_TransComm','a_TransComm','Oceania') = 80716.39844 ;
vdfp('c_TransComm','a_TransComm','EastAsia') = 707011.1875 ;
vdfp('c_TransComm','a_TransComm','SEAsia') = 111887.1094 ;
vdfp('c_TransComm','a_TransComm','SouthAsia') = 75135.52344 ;
vdfp('c_TransComm','a_TransComm','NAmerica') = 561736.625 ;
vdfp('c_TransComm','a_TransComm','LatinAmer') = 182236.0938 ;
vdfp('c_TransComm','a_TransComm','EU_28') = 903565.75 ;
vdfp('c_TransComm','a_TransComm','MENA') = 118081.4609 ;
vdfp('c_TransComm','a_TransComm','SSA') = 52160.76562 ;
vdfp('c_TransComm','a_TransComm','RestofWorld') = 202312.0781 ;
vdfp('c_TransComm','a_OthService','Oceania') = 84614.60938 ;
vdfp('c_TransComm','a_OthService','EastAsia') = 677090.875 ;
vdfp('c_TransComm','a_OthService','SEAsia') = 55663.96875 ;
vdfp('c_TransComm','a_OthService','SouthAsia') = 50629.35547 ;
vdfp('c_TransComm','a_OthService','NAmerica') = 845234 ;
vdfp('c_TransComm','a_OthService','LatinAmer') = 159137.4219 ;
vdfp('c_TransComm','a_OthService','EU_28') = 570649.375 ;
vdfp('c_TransComm','a_OthService','MENA') = 66376.9375 ;
vdfp('c_TransComm','a_OthService','SSA') = 49923.73047 ;
vdfp('c_TransComm','a_OthService','RestofWorld') = 104502.0312 ;
vdfp('c_OthService','a_agricultur','Oceania') = 11462.24707 ;
vdfp('c_OthService','a_agricultur','EastAsia') = 40224.73438 ;
vdfp('c_OthService','a_agricultur','SEAsia') = 5638.609863 ;
vdfp('c_OthService','a_agricultur','SouthAsia') = 5183.357422 ;
vdfp('c_OthService','a_agricultur','NAmerica') = 66410.44531 ;
vdfp('c_OthService','a_agricultur','LatinAmer') = 16618.65625 ;
vdfp('c_OthService','a_agricultur','EU_28') = 57801.60547 ;
vdfp('c_OthService','a_agricultur','MENA') = 8836.791992 ;
vdfp('c_OthService','a_agricultur','SSA') = 4692.121582 ;
vdfp('c_OthService','a_agricultur','RestofWorld') = 6969.60791 ;
vdfp('c_OthService','a_Extraction','Oceania') = 26037.96289 ;
vdfp('c_OthService','a_Extraction','EastAsia') = 39808.69922 ;
vdfp('c_OthService','a_Extraction','SEAsia') = 7768.344238 ;
vdfp('c_OthService','a_Extraction','SouthAsia') = 4057.423584 ;
vdfp('c_OthService','a_Extraction','NAmerica') = 51410.61719 ;
vdfp('c_OthService','a_Extraction','LatinAmer') = 27641.05078 ;
vdfp('c_OthService','a_Extraction','EU_28') = 26202.9082 ;
vdfp('c_OthService','a_Extraction','MENA') = 9260.858398 ;
vdfp('c_OthService','a_Extraction','SSA') = 22046.32617 ;
vdfp('c_OthService','a_Extraction','RestofWorld') = 54072.17188 ;
vdfp('c_OthService','a_ProcFood','Oceania') = 8603.916016 ;
vdfp('c_OthService','a_ProcFood','EastAsia') = 62373.90625 ;
vdfp('c_OthService','a_ProcFood','SEAsia') = 4505.785156 ;
vdfp('c_OthService','a_ProcFood','SouthAsia') = 8502.751953 ;
vdfp('c_OthService','a_ProcFood','NAmerica') = 75373.82031 ;
vdfp('c_OthService','a_ProcFood','LatinAmer') = 23057.31641 ;
vdfp('c_OthService','a_ProcFood','EU_28') = 161190.4531 ;
vdfp('c_OthService','a_ProcFood','MENA') = 7023.847656 ;
vdfp('c_OthService','a_ProcFood','SSA') = 6580.920898 ;
vdfp('c_OthService','a_ProcFood','RestofWorld') = 12191.64648 ;
vdfp('c_OthService','a_TextWapp','Oceania') = 967.1234741 ;
vdfp('c_OthService','a_TextWapp','EastAsia') = 50570.01562 ;
vdfp('c_OthService','a_TextWapp','SEAsia') = 2408.391602 ;
vdfp('c_OthService','a_TextWapp','SouthAsia') = 21201.47656 ;
vdfp('c_OthService','a_TextWapp','NAmerica') = 27203.85547 ;
vdfp('c_OthService','a_TextWapp','LatinAmer') = 7746.94873 ;
vdfp('c_OthService','a_TextWapp','EU_28') = 44454.17578 ;
vdfp('c_OthService','a_TextWapp','MENA') = 8686.09375 ;
vdfp('c_OthService','a_TextWapp','SSA') = 2521.415771 ;
vdfp('c_OthService','a_TextWapp','RestofWorld') = 1463.967773 ;
vdfp('c_OthService','a_LightMnfc','Oceania') = 14467.31641 ;
vdfp('c_OthService','a_LightMnfc','EastAsia') = 196491.5312 ;
vdfp('c_OthService','a_LightMnfc','SEAsia') = 9567.856445 ;
vdfp('c_OthService','a_LightMnfc','SouthAsia') = 27546.75 ;
vdfp('c_OthService','a_LightMnfc','NAmerica') = 163793.5781 ;
vdfp('c_OthService','a_LightMnfc','LatinAmer') = 27968.45117 ;
vdfp('c_OthService','a_LightMnfc','EU_28') = 368275.1562 ;
vdfp('c_OthService','a_LightMnfc','MENA') = 13777.91895 ;
vdfp('c_OthService','a_LightMnfc','SSA') = 6053.545898 ;
vdfp('c_OthService','a_LightMnfc','RestofWorld') = 19912.23633 ;
vdfp('c_OthService','a_HeavyMnfc','Oceania') = 16782.96289 ;
vdfp('c_OthService','a_HeavyMnfc','EastAsia') = 563994.8125 ;
vdfp('c_OthService','a_HeavyMnfc','SEAsia') = 29268.58203 ;
vdfp('c_OthService','a_HeavyMnfc','SouthAsia') = 29970.14258 ;
vdfp('c_OthService','a_HeavyMnfc','NAmerica') = 311448.7812 ;
vdfp('c_OthService','a_HeavyMnfc','LatinAmer') = 57734.28906 ;
vdfp('c_OthService','a_HeavyMnfc','EU_28') = 540680.25 ;
vdfp('c_OthService','a_HeavyMnfc','MENA') = 33043.50781 ;
vdfp('c_OthService','a_HeavyMnfc','SSA') = 15300.62598 ;
vdfp('c_OthService','a_HeavyMnfc','RestofWorld') = 49545.07422 ;
vdfp('c_OthService','a_Util_Cons','Oceania') = 62235.04297 ;
vdfp('c_OthService','a_Util_Cons','EastAsia') = 228156.9062 ;
vdfp('c_OthService','a_Util_Cons','SEAsia') = 20751.625 ;
vdfp('c_OthService','a_Util_Cons','SouthAsia') = 27253.28125 ;
vdfp('c_OthService','a_Util_Cons','NAmerica') = 276582.7188 ;
vdfp('c_OthService','a_Util_Cons','LatinAmer') = 38963.05859 ;
vdfp('c_OthService','a_Util_Cons','EU_28') = 324736.7812 ;
vdfp('c_OthService','a_Util_Cons','MENA') = 28673.13086 ;
vdfp('c_OthService','a_Util_Cons','SSA') = 11543.62109 ;
vdfp('c_OthService','a_Util_Cons','RestofWorld') = 33447.28516 ;
vdfp('c_OthService','a_TransComm','Oceania') = 97229.29688 ;
vdfp('c_OthService','a_TransComm','EastAsia') = 748386.9375 ;
vdfp('c_OthService','a_TransComm','SEAsia') = 67143.82031 ;
vdfp('c_OthService','a_TransComm','SouthAsia') = 43608.41406 ;
vdfp('c_OthService','a_TransComm','NAmerica') = 868193 ;
vdfp('c_OthService','a_TransComm','LatinAmer') = 164626.4062 ;
vdfp('c_OthService','a_TransComm','EU_28') = 815477.5 ;
vdfp('c_OthService','a_TransComm','MENA') = 82739.25 ;
vdfp('c_OthService','a_TransComm','SSA') = 40348.08594 ;
vdfp('c_OthService','a_TransComm','RestofWorld') = 114967.5625 ;
vdfp('c_OthService','a_OthService','Oceania') = 280280.8125 ;
vdfp('c_OthService','a_OthService','EastAsia') = 1208575 ;
vdfp('c_OthService','a_OthService','SEAsia') = 144945.5625 ;
vdfp('c_OthService','a_OthService','SouthAsia') = 51936.91406 ;
vdfp('c_OthService','a_OthService','NAmerica') = 2409757.75 ;
vdfp('c_OthService','a_OthService','LatinAmer') = 315059.8125 ;
vdfp('c_OthService','a_OthService','EU_28') = 2579228.25 ;
vdfp('c_OthService','a_OthService','MENA') = 147006.2812 ;
vdfp('c_OthService','a_OthService','SSA') = 90797.55469 ;
vdfp('c_OthService','a_OthService','RestofWorld') = 263717.8125 ;

* vmfb data (900 cells)
vmfb('c_Crops','a_agricultur','Oceania') = 410.9914551 ;
vmfb('c_Crops','a_agricultur','EastAsia') = 9021.801758 ;
vmfb('c_Crops','a_agricultur','SEAsia') = 2716.877197 ;
vmfb('c_Crops','a_agricultur','SouthAsia') = 2030.407715 ;
vmfb('c_Crops','a_agricultur','NAmerica') = 4078.754395 ;
vmfb('c_Crops','a_agricultur','LatinAmer') = 3207.329102 ;
vmfb('c_Crops','a_agricultur','EU_28') = 14946.16992 ;
vmfb('c_Crops','a_agricultur','MENA') = 8915.383789 ;
vmfb('c_Crops','a_agricultur','SSA') = 1772.997681 ;
vmfb('c_Crops','a_agricultur','RestofWorld') = 5940.894531 ;
vmfb('c_Crops','a_Extraction','Oceania') = 13.49488258 ;
vmfb('c_Crops','a_Extraction','EastAsia') = 355.0088196 ;
vmfb('c_Crops','a_Extraction','SEAsia') = 17.47990036 ;
vmfb('c_Crops','a_Extraction','SouthAsia') = 9.732661247 ;
vmfb('c_Crops','a_Extraction','NAmerica') = 79.10799408 ;
vmfb('c_Crops','a_Extraction','LatinAmer') = 93.07389069 ;
vmfb('c_Crops','a_Extraction','EU_28') = 256.9627075 ;
vmfb('c_Crops','a_Extraction','MENA') = 432.3305359 ;
vmfb('c_Crops','a_Extraction','SSA') = 43.92572021 ;
vmfb('c_Crops','a_Extraction','RestofWorld') = 41.04860687 ;
vmfb('c_Crops','a_ProcFood','Oceania') = 1252.597412 ;
vmfb('c_Crops','a_ProcFood','EastAsia') = 69168.57031 ;
vmfb('c_Crops','a_ProcFood','SEAsia') = 13558.72168 ;
vmfb('c_Crops','a_ProcFood','SouthAsia') = 3600.448486 ;
vmfb('c_Crops','a_ProcFood','NAmerica') = 24891.03906 ;
vmfb('c_Crops','a_ProcFood','LatinAmer') = 10132.99707 ;
vmfb('c_Crops','a_ProcFood','EU_28') = 49277.76953 ;
vmfb('c_Crops','a_ProcFood','MENA') = 19960.12891 ;
vmfb('c_Crops','a_ProcFood','SSA') = 3798.704346 ;
vmfb('c_Crops','a_ProcFood','RestofWorld') = 11463.35547 ;
vmfb('c_Crops','a_TextWapp','Oceania') = 3.917861938 ;
vmfb('c_Crops','a_TextWapp','EastAsia') = 13333.10352 ;
vmfb('c_Crops','a_TextWapp','SEAsia') = 3346.662109 ;
vmfb('c_Crops','a_TextWapp','SouthAsia') = 2452.241699 ;
vmfb('c_Crops','a_TextWapp','NAmerica') = 647.3112183 ;
vmfb('c_Crops','a_TextWapp','LatinAmer') = 577.057373 ;
vmfb('c_Crops','a_TextWapp','EU_28') = 2472.567871 ;
vmfb('c_Crops','a_TextWapp','MENA') = 2850.927002 ;
vmfb('c_Crops','a_TextWapp','SSA') = 231.4829102 ;
vmfb('c_Crops','a_TextWapp','RestofWorld') = 290.6422729 ;
vmfb('c_Crops','a_LightMnfc','Oceania') = 3.998417854 ;
vmfb('c_Crops','a_LightMnfc','EastAsia') = 1503.028442 ;
vmfb('c_Crops','a_LightMnfc','SEAsia') = 312.1921692 ;
vmfb('c_Crops','a_LightMnfc','SouthAsia') = 87.03527069 ;
vmfb('c_Crops','a_LightMnfc','NAmerica') = 145.541748 ;
vmfb('c_Crops','a_LightMnfc','LatinAmer') = 45.45517731 ;
vmfb('c_Crops','a_LightMnfc','EU_28') = 437.2658997 ;
vmfb('c_Crops','a_LightMnfc','MENA') = 768.5335083 ;
vmfb('c_Crops','a_LightMnfc','SSA') = 418.559967 ;
vmfb('c_Crops','a_LightMnfc','RestofWorld') = 136.0675354 ;
vmfb('c_Crops','a_HeavyMnfc','Oceania') = 67.37784576 ;
vmfb('c_Crops','a_HeavyMnfc','EastAsia') = 8441.8125 ;
vmfb('c_Crops','a_HeavyMnfc','SEAsia') = 1823.900879 ;
vmfb('c_Crops','a_HeavyMnfc','SouthAsia') = 242.9030762 ;
vmfb('c_Crops','a_HeavyMnfc','NAmerica') = 1447.677368 ;
vmfb('c_Crops','a_HeavyMnfc','LatinAmer') = 210.9395905 ;
vmfb('c_Crops','a_HeavyMnfc','EU_28') = 3270.857178 ;
vmfb('c_Crops','a_HeavyMnfc','MENA') = 954.0062256 ;
vmfb('c_Crops','a_HeavyMnfc','SSA') = 346.2543335 ;
vmfb('c_Crops','a_HeavyMnfc','RestofWorld') = 200.1822052 ;
vmfb('c_Crops','a_Util_Cons','Oceania') = 11.06129742 ;
vmfb('c_Crops','a_Util_Cons','EastAsia') = 354.0631714 ;
vmfb('c_Crops','a_Util_Cons','SEAsia') = 7.733784676 ;
vmfb('c_Crops','a_Util_Cons','SouthAsia') = 46.38639069 ;
vmfb('c_Crops','a_Util_Cons','NAmerica') = 1175.654297 ;
vmfb('c_Crops','a_Util_Cons','LatinAmer') = 29.82790184 ;
vmfb('c_Crops','a_Util_Cons','EU_28') = 1122.868042 ;
vmfb('c_Crops','a_Util_Cons','MENA') = 97.76766968 ;
vmfb('c_Crops','a_Util_Cons','SSA') = 14.90891933 ;
vmfb('c_Crops','a_Util_Cons','RestofWorld') = 75.24523163 ;
vmfb('c_Crops','a_TransComm','Oceania') = 276.0535278 ;
vmfb('c_Crops','a_TransComm','EastAsia') = 4704.043457 ;
vmfb('c_Crops','a_TransComm','SEAsia') = 1517.036743 ;
vmfb('c_Crops','a_TransComm','SouthAsia') = 409.5995483 ;
vmfb('c_Crops','a_TransComm','NAmerica') = 1346.616455 ;
vmfb('c_Crops','a_TransComm','LatinAmer') = 663.8964233 ;
vmfb('c_Crops','a_TransComm','EU_28') = 5598.101074 ;
vmfb('c_Crops','a_TransComm','MENA') = 2311.76123 ;
vmfb('c_Crops','a_TransComm','SSA') = 336.4663391 ;
vmfb('c_Crops','a_TransComm','RestofWorld') = 1115.618652 ;
vmfb('c_Crops','a_OthService','Oceania') = 130.0715485 ;
vmfb('c_Crops','a_OthService','EastAsia') = 2187.540527 ;
vmfb('c_Crops','a_OthService','SEAsia') = 186.5507202 ;
vmfb('c_Crops','a_OthService','SouthAsia') = 99.133461 ;
vmfb('c_Crops','a_OthService','NAmerica') = 5350.258301 ;
vmfb('c_Crops','a_OthService','LatinAmer') = 308.6897888 ;
vmfb('c_Crops','a_OthService','EU_28') = 5810.605469 ;
vmfb('c_Crops','a_OthService','MENA') = 1716.804932 ;
vmfb('c_Crops','a_OthService','SSA') = 350.8140564 ;
vmfb('c_Crops','a_OthService','RestofWorld') = 1428.519653 ;
vmfb('c_MeatLstk','a_agricultur','Oceania') = 134.4498749 ;
vmfb('c_MeatLstk','a_agricultur','EastAsia') = 3435.191162 ;
vmfb('c_MeatLstk','a_agricultur','SEAsia') = 580.3636475 ;
vmfb('c_MeatLstk','a_agricultur','SouthAsia') = 157.0627594 ;
vmfb('c_MeatLstk','a_agricultur','NAmerica') = 6654.958984 ;
vmfb('c_MeatLstk','a_agricultur','LatinAmer') = 1276.972046 ;
vmfb('c_MeatLstk','a_agricultur','EU_28') = 13427.08008 ;
vmfb('c_MeatLstk','a_agricultur','MENA') = 4219.07959 ;
vmfb('c_MeatLstk','a_agricultur','SSA') = 444.9822083 ;
vmfb('c_MeatLstk','a_agricultur','RestofWorld') = 2926.318848 ;
vmfb('c_MeatLstk','a_Extraction','Oceania') = 1.406763554 ;
vmfb('c_MeatLstk','a_Extraction','EastAsia') = 14.43925571 ;
vmfb('c_MeatLstk','a_Extraction','SEAsia') = 17.77701187 ;
vmfb('c_MeatLstk','a_Extraction','SouthAsia') = 2.179727554 ;
vmfb('c_MeatLstk','a_Extraction','NAmerica') = 67.42585754 ;
vmfb('c_MeatLstk','a_Extraction','LatinAmer') = 4.660620213 ;
vmfb('c_MeatLstk','a_Extraction','EU_28') = 75.28479004 ;
vmfb('c_MeatLstk','a_Extraction','MENA') = 25.19254494 ;
vmfb('c_MeatLstk','a_Extraction','SSA') = 9.413448334 ;
vmfb('c_MeatLstk','a_Extraction','RestofWorld') = 80.66946411 ;
vmfb('c_MeatLstk','a_ProcFood','Oceania') = 159.1802063 ;
vmfb('c_MeatLstk','a_ProcFood','EastAsia') = 7694.527832 ;
vmfb('c_MeatLstk','a_ProcFood','SEAsia') = 776.8787842 ;
vmfb('c_MeatLstk','a_ProcFood','SouthAsia') = 40.93994141 ;
vmfb('c_MeatLstk','a_ProcFood','NAmerica') = 1336.799927 ;
vmfb('c_MeatLstk','a_ProcFood','LatinAmer') = 257.628418 ;
vmfb('c_MeatLstk','a_ProcFood','EU_28') = 7591.763672 ;
vmfb('c_MeatLstk','a_ProcFood','MENA') = 1129.327148 ;
vmfb('c_MeatLstk','a_ProcFood','SSA') = 141.4814606 ;
vmfb('c_MeatLstk','a_ProcFood','RestofWorld') = 427.3331909 ;
vmfb('c_MeatLstk','a_TextWapp','Oceania') = 6.699580193 ;
vmfb('c_MeatLstk','a_TextWapp','EastAsia') = 3834.720703 ;
vmfb('c_MeatLstk','a_TextWapp','SEAsia') = 123.5802002 ;
vmfb('c_MeatLstk','a_TextWapp','SouthAsia') = 23.85804176 ;
vmfb('c_MeatLstk','a_TextWapp','NAmerica') = 356.1141663 ;
vmfb('c_MeatLstk','a_TextWapp','LatinAmer') = 110.1925888 ;
vmfb('c_MeatLstk','a_TextWapp','EU_28') = 794.1949463 ;
vmfb('c_MeatLstk','a_TextWapp','MENA') = 651.1525879 ;
vmfb('c_MeatLstk','a_TextWapp','SSA') = 92.17822266 ;
vmfb('c_MeatLstk','a_TextWapp','RestofWorld') = 107.59198 ;
vmfb('c_MeatLstk','a_LightMnfc','Oceania') = 6.61125803 ;
vmfb('c_MeatLstk','a_LightMnfc','EastAsia') = 5819.554688 ;
vmfb('c_MeatLstk','a_LightMnfc','SEAsia') = 831.0637817 ;
vmfb('c_MeatLstk','a_LightMnfc','SouthAsia') = 136.2564697 ;
vmfb('c_MeatLstk','a_LightMnfc','NAmerica') = 864.1254883 ;
vmfb('c_MeatLstk','a_LightMnfc','LatinAmer') = 205.3309937 ;
vmfb('c_MeatLstk','a_LightMnfc','EU_28') = 1036.860718 ;
vmfb('c_MeatLstk','a_LightMnfc','MENA') = 866.7556763 ;
vmfb('c_MeatLstk','a_LightMnfc','SSA') = 61.72673035 ;
vmfb('c_MeatLstk','a_LightMnfc','RestofWorld') = 251.4047699 ;
vmfb('c_MeatLstk','a_HeavyMnfc','Oceania') = 15.74754524 ;
vmfb('c_MeatLstk','a_HeavyMnfc','EastAsia') = 1071.388428 ;
vmfb('c_MeatLstk','a_HeavyMnfc','SEAsia') = 97.34387207 ;
vmfb('c_MeatLstk','a_HeavyMnfc','SouthAsia') = 15.44259834 ;
vmfb('c_MeatLstk','a_HeavyMnfc','NAmerica') = 457.4449158 ;
vmfb('c_MeatLstk','a_HeavyMnfc','LatinAmer') = 70.44506073 ;
vmfb('c_MeatLstk','a_HeavyMnfc','EU_28') = 593.5616455 ;
vmfb('c_MeatLstk','a_HeavyMnfc','MENA') = 285.5663757 ;
vmfb('c_MeatLstk','a_HeavyMnfc','SSA') = 20.10767174 ;
vmfb('c_MeatLstk','a_HeavyMnfc','RestofWorld') = 130.4233093 ;
vmfb('c_MeatLstk','a_Util_Cons','Oceania') = 2.641120195 ;
vmfb('c_MeatLstk','a_Util_Cons','EastAsia') = 5.154036999 ;
vmfb('c_MeatLstk','a_Util_Cons','SEAsia') = 1.429030061 ;
vmfb('c_MeatLstk','a_Util_Cons','SouthAsia') = 4.495933533 ;
vmfb('c_MeatLstk','a_Util_Cons','NAmerica') = 14.04214573 ;
vmfb('c_MeatLstk','a_Util_Cons','LatinAmer') = 19.77370834 ;
vmfb('c_MeatLstk','a_Util_Cons','EU_28') = 59.07350922 ;
vmfb('c_MeatLstk','a_Util_Cons','MENA') = 20.56485748 ;
vmfb('c_MeatLstk','a_Util_Cons','SSA') = 7.408382893 ;
vmfb('c_MeatLstk','a_Util_Cons','RestofWorld') = 18.68823433 ;
vmfb('c_MeatLstk','a_TransComm','Oceania') = 235.7287903 ;
vmfb('c_MeatLstk','a_TransComm','EastAsia') = 13058.85449 ;
vmfb('c_MeatLstk','a_TransComm','SEAsia') = 603.861084 ;
vmfb('c_MeatLstk','a_TransComm','SouthAsia') = 71.39276886 ;
vmfb('c_MeatLstk','a_TransComm','NAmerica') = 1253.297241 ;
vmfb('c_MeatLstk','a_TransComm','LatinAmer') = 1356.671509 ;
vmfb('c_MeatLstk','a_TransComm','EU_28') = 5131.33252 ;
vmfb('c_MeatLstk','a_TransComm','MENA') = 1359.821167 ;
vmfb('c_MeatLstk','a_TransComm','SSA') = 329.8420715 ;
vmfb('c_MeatLstk','a_TransComm','RestofWorld') = 883.1560669 ;
vmfb('c_MeatLstk','a_OthService','Oceania') = 98.39219666 ;
vmfb('c_MeatLstk','a_OthService','EastAsia') = 1164.52002 ;
vmfb('c_MeatLstk','a_OthService','SEAsia') = 105.7642365 ;
vmfb('c_MeatLstk','a_OthService','SouthAsia') = 82.35015106 ;
vmfb('c_MeatLstk','a_OthService','NAmerica') = 872.3966675 ;
vmfb('c_MeatLstk','a_OthService','LatinAmer') = 259.9705811 ;
vmfb('c_MeatLstk','a_OthService','EU_28') = 3467.586426 ;
vmfb('c_MeatLstk','a_OthService','MENA') = 610.3719482 ;
vmfb('c_MeatLstk','a_OthService','SSA') = 203.8705597 ;
vmfb('c_MeatLstk','a_OthService','RestofWorld') = 710.887146 ;
vmfb('c_Extraction','a_agricultur','Oceania') = 29.10751724 ;
vmfb('c_Extraction','a_agricultur','EastAsia') = 239.1485901 ;
vmfb('c_Extraction','a_agricultur','SEAsia') = 29.81288528 ;
vmfb('c_Extraction','a_agricultur','SouthAsia') = 16.59950066 ;
vmfb('c_Extraction','a_agricultur','NAmerica') = 174.5667877 ;
vmfb('c_Extraction','a_agricultur','LatinAmer') = 91.60399628 ;
vmfb('c_Extraction','a_agricultur','EU_28') = 2228.701904 ;
vmfb('c_Extraction','a_agricultur','MENA') = 41.90459442 ;
vmfb('c_Extraction','a_agricultur','SSA') = 44.89680862 ;
vmfb('c_Extraction','a_agricultur','RestofWorld') = 308.1869202 ;
vmfb('c_Extraction','a_Extraction','Oceania') = 55.85098648 ;
vmfb('c_Extraction','a_Extraction','EastAsia') = 19064.32422 ;
vmfb('c_Extraction','a_Extraction','SEAsia') = 317.2098694 ;
vmfb('c_Extraction','a_Extraction','SouthAsia') = 229.5932617 ;
vmfb('c_Extraction','a_Extraction','NAmerica') = 2927.567383 ;
vmfb('c_Extraction','a_Extraction','LatinAmer') = 1326.688843 ;
vmfb('c_Extraction','a_Extraction','EU_28') = 6662.005859 ;
vmfb('c_Extraction','a_Extraction','MENA') = 1523.895386 ;
vmfb('c_Extraction','a_Extraction','SSA') = 440.8300476 ;
vmfb('c_Extraction','a_Extraction','RestofWorld') = 2228.493896 ;
vmfb('c_Extraction','a_ProcFood','Oceania') = 77.62145996 ;
vmfb('c_Extraction','a_ProcFood','EastAsia') = 2805.107666 ;
vmfb('c_Extraction','a_ProcFood','SEAsia') = 550.1598511 ;
vmfb('c_Extraction','a_ProcFood','SouthAsia') = 164.1078339 ;
vmfb('c_Extraction','a_ProcFood','NAmerica') = 2024.035522 ;
vmfb('c_Extraction','a_ProcFood','LatinAmer') = 168.1605682 ;
vmfb('c_Extraction','a_ProcFood','EU_28') = 8055.737793 ;
vmfb('c_Extraction','a_ProcFood','MENA') = 627.8652344 ;
vmfb('c_Extraction','a_ProcFood','SSA') = 55.48390961 ;
vmfb('c_Extraction','a_ProcFood','RestofWorld') = 596.2597046 ;
vmfb('c_Extraction','a_TextWapp','Oceania') = 1.429281592 ;
vmfb('c_Extraction','a_TextWapp','EastAsia') = 305.5892639 ;
vmfb('c_Extraction','a_TextWapp','SEAsia') = 45.52474594 ;
vmfb('c_Extraction','a_TextWapp','SouthAsia') = 104.7737656 ;
vmfb('c_Extraction','a_TextWapp','NAmerica') = 28.3374424 ;
vmfb('c_Extraction','a_TextWapp','LatinAmer') = 28.39341545 ;
vmfb('c_Extraction','a_TextWapp','EU_28') = 756.142395 ;
vmfb('c_Extraction','a_TextWapp','MENA') = 265.7185974 ;
vmfb('c_Extraction','a_TextWapp','SSA') = 1.948303223 ;
vmfb('c_Extraction','a_TextWapp','RestofWorld') = 37.30546188 ;
vmfb('c_Extraction','a_LightMnfc','Oceania') = 86.94788361 ;
vmfb('c_Extraction','a_LightMnfc','EastAsia') = 13313.94238 ;
vmfb('c_Extraction','a_LightMnfc','SEAsia') = 662.3861084 ;
vmfb('c_Extraction','a_LightMnfc','SouthAsia') = 5926.709961 ;
vmfb('c_Extraction','a_LightMnfc','NAmerica') = 1713.873657 ;
vmfb('c_Extraction','a_LightMnfc','LatinAmer') = 524.6052246 ;
vmfb('c_Extraction','a_LightMnfc','EU_28') = 15608.95215 ;
vmfb('c_Extraction','a_LightMnfc','MENA') = 3931.893799 ;
vmfb('c_Extraction','a_LightMnfc','SSA') = 157.9904327 ;
vmfb('c_Extraction','a_LightMnfc','RestofWorld') = 898.7989502 ;
vmfb('c_Extraction','a_HeavyMnfc','Oceania') = 18534.63086 ;
vmfb('c_Extraction','a_HeavyMnfc','EastAsia') = 730931.0625 ;
vmfb('c_Extraction','a_HeavyMnfc','SEAsia') = 101527.6484 ;
vmfb('c_Extraction','a_HeavyMnfc','SouthAsia') = 155308.0469 ;
vmfb('c_Extraction','a_HeavyMnfc','NAmerica') = 366023.3125 ;
vmfb('c_Extraction','a_HeavyMnfc','LatinAmer') = 44831.80859 ;
vmfb('c_Extraction','a_HeavyMnfc','EU_28') = 497444.4375 ;
vmfb('c_Extraction','a_HeavyMnfc','MENA') = 44293.59766 ;
vmfb('c_Extraction','a_HeavyMnfc','SSA') = 25919.13281 ;
vmfb('c_Extraction','a_HeavyMnfc','RestofWorld') = 36135.21875 ;
vmfb('c_Extraction','a_Util_Cons','Oceania') = 90.11577606 ;
vmfb('c_Extraction','a_Util_Cons','EastAsia') = 105225.7422 ;
vmfb('c_Extraction','a_Util_Cons','SEAsia') = 7947.894531 ;
vmfb('c_Extraction','a_Util_Cons','SouthAsia') = 32615.42578 ;
vmfb('c_Extraction','a_Util_Cons','NAmerica') = 9074.892578 ;
vmfb('c_Extraction','a_Util_Cons','LatinAmer') = 5031.09375 ;
vmfb('c_Extraction','a_Util_Cons','EU_28') = 89082.25781 ;
vmfb('c_Extraction','a_Util_Cons','MENA') = 24112.59766 ;
vmfb('c_Extraction','a_Util_Cons','SSA') = 1032.557373 ;
vmfb('c_Extraction','a_Util_Cons','RestofWorld') = 16849.77539 ;
vmfb('c_Extraction','a_TransComm','Oceania') = 73.00930786 ;
vmfb('c_Extraction','a_TransComm','EastAsia') = 10713.58789 ;
vmfb('c_Extraction','a_TransComm','SEAsia') = 392.1062622 ;
vmfb('c_Extraction','a_TransComm','SouthAsia') = 360.7737732 ;
vmfb('c_Extraction','a_TransComm','NAmerica') = 932.4447632 ;
vmfb('c_Extraction','a_TransComm','LatinAmer') = 398.3613586 ;
vmfb('c_Extraction','a_TransComm','EU_28') = 12291.78809 ;
vmfb('c_Extraction','a_TransComm','MENA') = 1293.45874 ;
vmfb('c_Extraction','a_TransComm','SSA') = 64.81999969 ;
vmfb('c_Extraction','a_TransComm','RestofWorld') = 2372.320068 ;
vmfb('c_Extraction','a_OthService','Oceania') = 75.37521362 ;
vmfb('c_Extraction','a_OthService','EastAsia') = 2618.262207 ;
vmfb('c_Extraction','a_OthService','SEAsia') = 69.81736755 ;
vmfb('c_Extraction','a_OthService','SouthAsia') = 25.47101402 ;
vmfb('c_Extraction','a_OthService','NAmerica') = 1253.724487 ;
vmfb('c_Extraction','a_OthService','LatinAmer') = 78.16928101 ;
vmfb('c_Extraction','a_OthService','EU_28') = 12749.20508 ;
vmfb('c_Extraction','a_OthService','MENA') = 121.8572922 ;
vmfb('c_Extraction','a_OthService','SSA') = 136.6882477 ;
vmfb('c_Extraction','a_OthService','RestofWorld') = 390.2554932 ;
vmfb('c_ProcFood','a_agricultur','Oceania') = 180.2363586 ;
vmfb('c_ProcFood','a_agricultur','EastAsia') = 3715.793701 ;
vmfb('c_ProcFood','a_agricultur','SEAsia') = 2312.137695 ;
vmfb('c_ProcFood','a_agricultur','SouthAsia') = 1975.90979 ;
vmfb('c_ProcFood','a_agricultur','NAmerica') = 2259.853271 ;
vmfb('c_ProcFood','a_agricultur','LatinAmer') = 2355.604004 ;
vmfb('c_ProcFood','a_agricultur','EU_28') = 18435.17773 ;
vmfb('c_ProcFood','a_agricultur','MENA') = 3135.915527 ;
vmfb('c_ProcFood','a_agricultur','SSA') = 982.2347412 ;
vmfb('c_ProcFood','a_agricultur','RestofWorld') = 1842.313843 ;
vmfb('c_ProcFood','a_Extraction','Oceania') = 55.88679504 ;
vmfb('c_ProcFood','a_Extraction','EastAsia') = 602.2800293 ;
vmfb('c_ProcFood','a_Extraction','SEAsia') = 237.9932861 ;
vmfb('c_ProcFood','a_Extraction','SouthAsia') = 30.84228706 ;
vmfb('c_ProcFood','a_Extraction','NAmerica') = 85.12388611 ;
vmfb('c_ProcFood','a_Extraction','LatinAmer') = 189.8962555 ;
vmfb('c_ProcFood','a_Extraction','EU_28') = 415.9726257 ;
vmfb('c_ProcFood','a_Extraction','MENA') = 144.6561432 ;
vmfb('c_ProcFood','a_Extraction','SSA') = 115.1981659 ;
vmfb('c_ProcFood','a_Extraction','RestofWorld') = 228.6036682 ;
vmfb('c_ProcFood','a_ProcFood','Oceania') = 1988.021118 ;
vmfb('c_ProcFood','a_ProcFood','EastAsia') = 34634.42969 ;
vmfb('c_ProcFood','a_ProcFood','SEAsia') = 17886.73633 ;
vmfb('c_ProcFood','a_ProcFood','SouthAsia') = 4372.502441 ;
vmfb('c_ProcFood','a_ProcFood','NAmerica') = 29643.10156 ;
vmfb('c_ProcFood','a_ProcFood','LatinAmer') = 10394.18066 ;
vmfb('c_ProcFood','a_ProcFood','EU_28') = 65337.03125 ;
vmfb('c_ProcFood','a_ProcFood','MENA') = 14275.95605 ;
vmfb('c_ProcFood','a_ProcFood','SSA') = 4960.196289 ;
vmfb('c_ProcFood','a_ProcFood','RestofWorld') = 11326.02246 ;
vmfb('c_ProcFood','a_TextWapp','Oceania') = 2.388509035 ;
vmfb('c_ProcFood','a_TextWapp','EastAsia') = 138.5906219 ;
vmfb('c_ProcFood','a_TextWapp','SEAsia') = 33.01974106 ;
vmfb('c_ProcFood','a_TextWapp','SouthAsia') = 34.20780182 ;
vmfb('c_ProcFood','a_TextWapp','NAmerica') = 20.06487846 ;
vmfb('c_ProcFood','a_TextWapp','LatinAmer') = 16.34663773 ;
vmfb('c_ProcFood','a_TextWapp','EU_28') = 529.5247192 ;
vmfb('c_ProcFood','a_TextWapp','MENA') = 106.7258301 ;
vmfb('c_ProcFood','a_TextWapp','SSA') = 71.97602844 ;
vmfb('c_ProcFood','a_TextWapp','RestofWorld') = 58.25073624 ;
vmfb('c_ProcFood','a_LightMnfc','Oceania') = 98.2117157 ;
vmfb('c_ProcFood','a_LightMnfc','EastAsia') = 567.8295288 ;
vmfb('c_ProcFood','a_LightMnfc','SEAsia') = 55.11190033 ;
vmfb('c_ProcFood','a_LightMnfc','SouthAsia') = 68.19954681 ;
vmfb('c_ProcFood','a_LightMnfc','NAmerica') = 435.1402588 ;
vmfb('c_ProcFood','a_LightMnfc','LatinAmer') = 105.9268951 ;
vmfb('c_ProcFood','a_LightMnfc','EU_28') = 2894.309082 ;
vmfb('c_ProcFood','a_LightMnfc','MENA') = 137.4673767 ;
vmfb('c_ProcFood','a_LightMnfc','SSA') = 140.488327 ;
vmfb('c_ProcFood','a_LightMnfc','RestofWorld') = 140.416214 ;
vmfb('c_ProcFood','a_HeavyMnfc','Oceania') = 252.9959717 ;
vmfb('c_ProcFood','a_HeavyMnfc','EastAsia') = 3333.168701 ;
vmfb('c_ProcFood','a_HeavyMnfc','SEAsia') = 1660.486328 ;
vmfb('c_ProcFood','a_HeavyMnfc','SouthAsia') = 449.4811096 ;
vmfb('c_ProcFood','a_HeavyMnfc','NAmerica') = 2287.076904 ;
vmfb('c_ProcFood','a_HeavyMnfc','LatinAmer') = 870.6882935 ;
vmfb('c_ProcFood','a_HeavyMnfc','EU_28') = 4994.755859 ;
vmfb('c_ProcFood','a_HeavyMnfc','MENA') = 416.5029602 ;
vmfb('c_ProcFood','a_HeavyMnfc','SSA') = 380.7593079 ;
vmfb('c_ProcFood','a_HeavyMnfc','RestofWorld') = 282.4179382 ;
vmfb('c_ProcFood','a_Util_Cons','Oceania') = 30.43476868 ;
vmfb('c_ProcFood','a_Util_Cons','EastAsia') = 153.3267822 ;
vmfb('c_ProcFood','a_Util_Cons','SEAsia') = 31.71809196 ;
vmfb('c_ProcFood','a_Util_Cons','SouthAsia') = 2.81204915 ;
vmfb('c_ProcFood','a_Util_Cons','NAmerica') = 55.26042175 ;
vmfb('c_ProcFood','a_Util_Cons','LatinAmer') = 79.05483246 ;
vmfb('c_ProcFood','a_Util_Cons','EU_28') = 797.9203491 ;
vmfb('c_ProcFood','a_Util_Cons','MENA') = 37.01213455 ;
vmfb('c_ProcFood','a_Util_Cons','SSA') = 63.30392075 ;
vmfb('c_ProcFood','a_Util_Cons','RestofWorld') = 77.11165619 ;
vmfb('c_ProcFood','a_TransComm','Oceania') = 1742.478149 ;
vmfb('c_ProcFood','a_TransComm','EastAsia') = 14373.41797 ;
vmfb('c_ProcFood','a_TransComm','SEAsia') = 3389.097412 ;
vmfb('c_ProcFood','a_TransComm','SouthAsia') = 1278.574463 ;
vmfb('c_ProcFood','a_TransComm','NAmerica') = 5266.912598 ;
vmfb('c_ProcFood','a_TransComm','LatinAmer') = 2978.890137 ;
vmfb('c_ProcFood','a_TransComm','EU_28') = 31990.40625 ;
vmfb('c_ProcFood','a_TransComm','MENA') = 7043.688965 ;
vmfb('c_ProcFood','a_TransComm','SSA') = 2635.373047 ;
vmfb('c_ProcFood','a_TransComm','RestofWorld') = 2574.240723 ;
vmfb('c_ProcFood','a_OthService','Oceania') = 831.0930786 ;
vmfb('c_ProcFood','a_OthService','EastAsia') = 2391.27124 ;
vmfb('c_ProcFood','a_OthService','SEAsia') = 774.5455933 ;
vmfb('c_ProcFood','a_OthService','SouthAsia') = 67.40852356 ;
vmfb('c_ProcFood','a_OthService','NAmerica') = 12248.32324 ;
vmfb('c_ProcFood','a_OthService','LatinAmer') = 939.7966309 ;
vmfb('c_ProcFood','a_OthService','EU_28') = 14875.2627 ;
vmfb('c_ProcFood','a_OthService','MENA') = 3054.703125 ;
vmfb('c_ProcFood','a_OthService','SSA') = 1712.6698 ;
vmfb('c_ProcFood','a_OthService','RestofWorld') = 2849.303711 ;
vmfb('c_TextWapp','a_agricultur','Oceania') = 66.33325958 ;
vmfb('c_TextWapp','a_agricultur','EastAsia') = 53.59847641 ;
vmfb('c_TextWapp','a_agricultur','SEAsia') = 61.01171112 ;
vmfb('c_TextWapp','a_agricultur','SouthAsia') = 235.6318817 ;
vmfb('c_TextWapp','a_agricultur','NAmerica') = 842.7241211 ;
vmfb('c_TextWapp','a_agricultur','LatinAmer') = 367.6818848 ;
vmfb('c_TextWapp','a_agricultur','EU_28') = 552.3334351 ;
vmfb('c_TextWapp','a_agricultur','MENA') = 734.9699707 ;
vmfb('c_TextWapp','a_agricultur','SSA') = 727.2885742 ;
vmfb('c_TextWapp','a_agricultur','RestofWorld') = 204.6123962 ;
vmfb('c_TextWapp','a_Extraction','Oceania') = 115.7049637 ;
vmfb('c_TextWapp','a_Extraction','EastAsia') = 346.7378845 ;
vmfb('c_TextWapp','a_Extraction','SEAsia') = 50.70137787 ;
vmfb('c_TextWapp','a_Extraction','SouthAsia') = 58.11943054 ;
vmfb('c_TextWapp','a_Extraction','NAmerica') = 426.6512451 ;
vmfb('c_TextWapp','a_Extraction','LatinAmer') = 129.5910645 ;
vmfb('c_TextWapp','a_Extraction','EU_28') = 503.5301819 ;
vmfb('c_TextWapp','a_Extraction','MENA') = 219.494812 ;
vmfb('c_TextWapp','a_Extraction','SSA') = 292.0785828 ;
vmfb('c_TextWapp','a_Extraction','RestofWorld') = 305.5086365 ;
vmfb('c_TextWapp','a_ProcFood','Oceania') = 122.7268753 ;
vmfb('c_TextWapp','a_ProcFood','EastAsia') = 281.2229004 ;
vmfb('c_TextWapp','a_ProcFood','SEAsia') = 135.9411011 ;
vmfb('c_TextWapp','a_ProcFood','SouthAsia') = 611.9154053 ;
vmfb('c_TextWapp','a_ProcFood','NAmerica') = 133.0549774 ;
vmfb('c_TextWapp','a_ProcFood','LatinAmer') = 207.0367737 ;
vmfb('c_TextWapp','a_ProcFood','EU_28') = 903.0578003 ;
vmfb('c_TextWapp','a_ProcFood','MENA') = 369.0718994 ;
vmfb('c_TextWapp','a_ProcFood','SSA') = 210.1535187 ;
vmfb('c_TextWapp','a_ProcFood','RestofWorld') = 242.1328278 ;
vmfb('c_TextWapp','a_TextWapp','Oceania') = 1016.994629 ;
vmfb('c_TextWapp','a_TextWapp','EastAsia') = 47632.21094 ;
vmfb('c_TextWapp','a_TextWapp','SEAsia') = 25927.2207 ;
vmfb('c_TextWapp','a_TextWapp','SouthAsia') = 11620.67188 ;
vmfb('c_TextWapp','a_TextWapp','NAmerica') = 29325.11523 ;
vmfb('c_TextWapp','a_TextWapp','LatinAmer') = 14875.90723 ;
vmfb('c_TextWapp','a_TextWapp','EU_28') = 58542.53516 ;
vmfb('c_TextWapp','a_TextWapp','MENA') = 16318.03613 ;
vmfb('c_TextWapp','a_TextWapp','SSA') = 3124.038574 ;
vmfb('c_TextWapp','a_TextWapp','RestofWorld') = 7184.775879 ;
vmfb('c_TextWapp','a_LightMnfc','Oceania') = 430.640564 ;
vmfb('c_TextWapp','a_LightMnfc','EastAsia') = 4688.494629 ;
vmfb('c_TextWapp','a_LightMnfc','SEAsia') = 3533.339355 ;
vmfb('c_TextWapp','a_LightMnfc','SouthAsia') = 140.9259491 ;
vmfb('c_TextWapp','a_LightMnfc','NAmerica') = 13000.62695 ;
vmfb('c_TextWapp','a_LightMnfc','LatinAmer') = 1593.942993 ;
vmfb('c_TextWapp','a_LightMnfc','EU_28') = 11626.72266 ;
vmfb('c_TextWapp','a_LightMnfc','MENA') = 879.4038086 ;
vmfb('c_TextWapp','a_LightMnfc','SSA') = 886.8568726 ;
vmfb('c_TextWapp','a_LightMnfc','RestofWorld') = 1477.804565 ;
vmfb('c_TextWapp','a_HeavyMnfc','Oceania') = 285.339325 ;
vmfb('c_TextWapp','a_HeavyMnfc','EastAsia') = 3013.72998 ;
vmfb('c_TextWapp','a_HeavyMnfc','SEAsia') = 860.8842773 ;
vmfb('c_TextWapp','a_HeavyMnfc','SouthAsia') = 170.7772675 ;
vmfb('c_TextWapp','a_HeavyMnfc','NAmerica') = 3319.988525 ;
vmfb('c_TextWapp','a_HeavyMnfc','LatinAmer') = 916.8098145 ;
vmfb('c_TextWapp','a_HeavyMnfc','EU_28') = 6736.916016 ;
vmfb('c_TextWapp','a_HeavyMnfc','MENA') = 2735.699219 ;
vmfb('c_TextWapp','a_HeavyMnfc','SSA') = 325.4897461 ;
vmfb('c_TextWapp','a_HeavyMnfc','RestofWorld') = 1552.505493 ;
vmfb('c_TextWapp','a_Util_Cons','Oceania') = 283.1362915 ;
vmfb('c_TextWapp','a_Util_Cons','EastAsia') = 1055.837524 ;
vmfb('c_TextWapp','a_Util_Cons','SEAsia') = 191.5922546 ;
vmfb('c_TextWapp','a_Util_Cons','SouthAsia') = 24.12039948 ;
vmfb('c_TextWapp','a_Util_Cons','NAmerica') = 1567.494141 ;
vmfb('c_TextWapp','a_Util_Cons','LatinAmer') = 159.4113312 ;
vmfb('c_TextWapp','a_Util_Cons','EU_28') = 2761.334717 ;
vmfb('c_TextWapp','a_Util_Cons','MENA') = 749.1322021 ;
vmfb('c_TextWapp','a_Util_Cons','SSA') = 188.3923645 ;
vmfb('c_TextWapp','a_Util_Cons','RestofWorld') = 626.1069946 ;
vmfb('c_TextWapp','a_TransComm','Oceania') = 858.6398926 ;
vmfb('c_TextWapp','a_TransComm','EastAsia') = 1886.143677 ;
vmfb('c_TextWapp','a_TransComm','SEAsia') = 813.1248169 ;
vmfb('c_TextWapp','a_TransComm','SouthAsia') = 124.4251938 ;
vmfb('c_TextWapp','a_TransComm','NAmerica') = 4167.498047 ;
vmfb('c_TextWapp','a_TransComm','LatinAmer') = 790.6846924 ;
vmfb('c_TextWapp','a_TransComm','EU_28') = 7925.134766 ;
vmfb('c_TextWapp','a_TransComm','MENA') = 3994.284668 ;
vmfb('c_TextWapp','a_TransComm','SSA') = 802.0488281 ;
vmfb('c_TextWapp','a_TransComm','RestofWorld') = 1337.01355 ;
vmfb('c_TextWapp','a_OthService','Oceania') = 1466.895386 ;
vmfb('c_TextWapp','a_OthService','EastAsia') = 2783.579834 ;
vmfb('c_TextWapp','a_OthService','SEAsia') = 627.4266968 ;
vmfb('c_TextWapp','a_OthService','SouthAsia') = 418.2997131 ;
vmfb('c_TextWapp','a_OthService','NAmerica') = 4189.97998 ;
vmfb('c_TextWapp','a_OthService','LatinAmer') = 740.7105103 ;
vmfb('c_TextWapp','a_OthService','EU_28') = 11266.26562 ;
vmfb('c_TextWapp','a_OthService','MENA') = 3501.617676 ;
vmfb('c_TextWapp','a_OthService','SSA') = 1480.223389 ;
vmfb('c_TextWapp','a_OthService','RestofWorld') = 2190.790527 ;
vmfb('c_LightMnfc','a_agricultur','Oceania') = 199.6208344 ;
vmfb('c_LightMnfc','a_agricultur','EastAsia') = 525.6530762 ;
vmfb('c_LightMnfc','a_agricultur','SEAsia') = 319.1559753 ;
vmfb('c_LightMnfc','a_agricultur','SouthAsia') = 165.2107086 ;
vmfb('c_LightMnfc','a_agricultur','NAmerica') = 1222.022339 ;
vmfb('c_LightMnfc','a_agricultur','LatinAmer') = 1139.095581 ;
vmfb('c_LightMnfc','a_agricultur','EU_28') = 4436.138672 ;
vmfb('c_LightMnfc','a_agricultur','MENA') = 1370.362671 ;
vmfb('c_LightMnfc','a_agricultur','SSA') = 2376.219727 ;
vmfb('c_LightMnfc','a_agricultur','RestofWorld') = 1723.492676 ;
vmfb('c_LightMnfc','a_Extraction','Oceania') = 868.9053955 ;
vmfb('c_LightMnfc','a_Extraction','EastAsia') = 2900.815918 ;
vmfb('c_LightMnfc','a_Extraction','SEAsia') = 1163.38623 ;
vmfb('c_LightMnfc','a_Extraction','SouthAsia') = 623.1765747 ;
vmfb('c_LightMnfc','a_Extraction','NAmerica') = 2565.970947 ;
vmfb('c_LightMnfc','a_Extraction','LatinAmer') = 1508.644409 ;
vmfb('c_LightMnfc','a_Extraction','EU_28') = 3209.179932 ;
vmfb('c_LightMnfc','a_Extraction','MENA') = 2512.130371 ;
vmfb('c_LightMnfc','a_Extraction','SSA') = 3357.464111 ;
vmfb('c_LightMnfc','a_Extraction','RestofWorld') = 3990.243164 ;
vmfb('c_LightMnfc','a_ProcFood','Oceania') = 652.3693237 ;
vmfb('c_LightMnfc','a_ProcFood','EastAsia') = 2689.939941 ;
vmfb('c_LightMnfc','a_ProcFood','SEAsia') = 3112.551514 ;
vmfb('c_LightMnfc','a_ProcFood','SouthAsia') = 905.1005249 ;
vmfb('c_LightMnfc','a_ProcFood','NAmerica') = 3102.640625 ;
vmfb('c_LightMnfc','a_ProcFood','LatinAmer') = 2002.199463 ;
vmfb('c_LightMnfc','a_ProcFood','EU_28') = 14508.79492 ;
vmfb('c_LightMnfc','a_ProcFood','MENA') = 2777.245117 ;
vmfb('c_LightMnfc','a_ProcFood','SSA') = 1522.623779 ;
vmfb('c_LightMnfc','a_ProcFood','RestofWorld') = 2540.468262 ;
vmfb('c_LightMnfc','a_TextWapp','Oceania') = 153.697525 ;
vmfb('c_LightMnfc','a_TextWapp','EastAsia') = 2977.060547 ;
vmfb('c_LightMnfc','a_TextWapp','SEAsia') = 843.9711914 ;
vmfb('c_LightMnfc','a_TextWapp','SouthAsia') = 837.5515137 ;
vmfb('c_LightMnfc','a_TextWapp','NAmerica') = 2005.63208 ;
vmfb('c_LightMnfc','a_TextWapp','LatinAmer') = 967.9093018 ;
vmfb('c_LightMnfc','a_TextWapp','EU_28') = 5126.060059 ;
vmfb('c_LightMnfc','a_TextWapp','MENA') = 1163.231812 ;
vmfb('c_LightMnfc','a_TextWapp','SSA') = 409.3229065 ;
vmfb('c_LightMnfc','a_TextWapp','RestofWorld') = 894.1315918 ;
vmfb('c_LightMnfc','a_LightMnfc','Oceania') = 6951.849609 ;
vmfb('c_LightMnfc','a_LightMnfc','EastAsia') = 121505.1094 ;
vmfb('c_LightMnfc','a_LightMnfc','SEAsia') = 42388.76562 ;
vmfb('c_LightMnfc','a_LightMnfc','SouthAsia') = 11400.29102 ;
vmfb('c_LightMnfc','a_LightMnfc','NAmerica') = 170034.7031 ;
vmfb('c_LightMnfc','a_LightMnfc','LatinAmer') = 37914.29688 ;
vmfb('c_LightMnfc','a_LightMnfc','EU_28') = 357893.6562 ;
vmfb('c_LightMnfc','a_LightMnfc','MENA') = 39660.19141 ;
vmfb('c_LightMnfc','a_LightMnfc','SSA') = 12127.47852 ;
vmfb('c_LightMnfc','a_LightMnfc','RestofWorld') = 28521.98242 ;
vmfb('c_LightMnfc','a_HeavyMnfc','Oceania') = 1439.333008 ;
vmfb('c_LightMnfc','a_HeavyMnfc','EastAsia') = 28776.33203 ;
vmfb('c_LightMnfc','a_HeavyMnfc','SEAsia') = 9029.981445 ;
vmfb('c_LightMnfc','a_HeavyMnfc','SouthAsia') = 3531.760742 ;
vmfb('c_LightMnfc','a_HeavyMnfc','NAmerica') = 21450.28125 ;
vmfb('c_LightMnfc','a_HeavyMnfc','LatinAmer') = 6501.553711 ;
vmfb('c_LightMnfc','a_HeavyMnfc','EU_28') = 80664.49219 ;
vmfb('c_LightMnfc','a_HeavyMnfc','MENA') = 8579.621094 ;
vmfb('c_LightMnfc','a_HeavyMnfc','SSA') = 2854.983887 ;
vmfb('c_LightMnfc','a_HeavyMnfc','RestofWorld') = 14102.1123 ;
vmfb('c_LightMnfc','a_Util_Cons','Oceania') = 5866.34375 ;
vmfb('c_LightMnfc','a_Util_Cons','EastAsia') = 22453.61914 ;
vmfb('c_LightMnfc','a_Util_Cons','SEAsia') = 10074.06934 ;
vmfb('c_LightMnfc','a_Util_Cons','SouthAsia') = 4463.611328 ;
vmfb('c_LightMnfc','a_Util_Cons','NAmerica') = 35948.19141 ;
vmfb('c_LightMnfc','a_Util_Cons','LatinAmer') = 8381.492188 ;
vmfb('c_LightMnfc','a_Util_Cons','EU_28') = 64413.78516 ;
vmfb('c_LightMnfc','a_Util_Cons','MENA') = 33327.95312 ;
vmfb('c_LightMnfc','a_Util_Cons','SSA') = 4686.83252 ;
vmfb('c_LightMnfc','a_Util_Cons','RestofWorld') = 16891.15625 ;
vmfb('c_LightMnfc','a_TransComm','Oceania') = 6465.046387 ;
vmfb('c_LightMnfc','a_TransComm','EastAsia') = 20047.41211 ;
vmfb('c_LightMnfc','a_TransComm','SEAsia') = 6781.985352 ;
vmfb('c_LightMnfc','a_TransComm','SouthAsia') = 7715.334961 ;
vmfb('c_LightMnfc','a_TransComm','NAmerica') = 32354.66016 ;
vmfb('c_LightMnfc','a_TransComm','LatinAmer') = 9214.943359 ;
vmfb('c_LightMnfc','a_TransComm','EU_28') = 68587.82812 ;
vmfb('c_LightMnfc','a_TransComm','MENA') = 17204.75391 ;
vmfb('c_LightMnfc','a_TransComm','SSA') = 8447.582031 ;
vmfb('c_LightMnfc','a_TransComm','RestofWorld') = 21089.35352 ;
vmfb('c_LightMnfc','a_OthService','Oceania') = 6501.289062 ;
vmfb('c_LightMnfc','a_OthService','EastAsia') = 25818.20312 ;
vmfb('c_LightMnfc','a_OthService','SEAsia') = 6571.991211 ;
vmfb('c_LightMnfc','a_OthService','SouthAsia') = 2180.152588 ;
vmfb('c_LightMnfc','a_OthService','NAmerica') = 37841.25 ;
vmfb('c_LightMnfc','a_OthService','LatinAmer') = 8235.217773 ;
vmfb('c_LightMnfc','a_OthService','EU_28') = 95115.42188 ;
vmfb('c_LightMnfc','a_OthService','MENA') = 12180.58789 ;
vmfb('c_LightMnfc','a_OthService','SSA') = 9507.818359 ;
vmfb('c_LightMnfc','a_OthService','RestofWorld') = 13153.03027 ;
vmfb('c_HeavyMnfc','a_agricultur','Oceania') = 3412.290039 ;
vmfb('c_HeavyMnfc','a_agricultur','EastAsia') = 12789.20117 ;
vmfb('c_HeavyMnfc','a_agricultur','SEAsia') = 9324.162109 ;
vmfb('c_HeavyMnfc','a_agricultur','SouthAsia') = 10677.29785 ;
vmfb('c_HeavyMnfc','a_agricultur','NAmerica') = 17029.27734 ;
vmfb('c_HeavyMnfc','a_agricultur','LatinAmer') = 12620.92871 ;
vmfb('c_HeavyMnfc','a_agricultur','EU_28') = 28124.43359 ;
vmfb('c_HeavyMnfc','a_agricultur','MENA') = 10098.52344 ;
vmfb('c_HeavyMnfc','a_agricultur','SSA') = 9814.894531 ;
vmfb('c_HeavyMnfc','a_agricultur','RestofWorld') = 9830.333008 ;
vmfb('c_HeavyMnfc','a_Extraction','Oceania') = 6500.602051 ;
vmfb('c_HeavyMnfc','a_Extraction','EastAsia') = 17982.64648 ;
vmfb('c_HeavyMnfc','a_Extraction','SEAsia') = 9001.755859 ;
vmfb('c_HeavyMnfc','a_Extraction','SouthAsia') = 3183.175049 ;
vmfb('c_HeavyMnfc','a_Extraction','NAmerica') = 20790.51172 ;
vmfb('c_HeavyMnfc','a_Extraction','LatinAmer') = 11016.33105 ;
vmfb('c_HeavyMnfc','a_Extraction','EU_28') = 12254.77637 ;
vmfb('c_HeavyMnfc','a_Extraction','MENA') = 60815.71484 ;
vmfb('c_HeavyMnfc','a_Extraction','SSA') = 11083.71875 ;
vmfb('c_HeavyMnfc','a_Extraction','RestofWorld') = 13768.99609 ;
vmfb('c_HeavyMnfc','a_ProcFood','Oceania') = 1881.253418 ;
vmfb('c_HeavyMnfc','a_ProcFood','EastAsia') = 8038.23584 ;
vmfb('c_HeavyMnfc','a_ProcFood','SEAsia') = 5392.081055 ;
vmfb('c_HeavyMnfc','a_ProcFood','SouthAsia') = 3824.436523 ;
vmfb('c_HeavyMnfc','a_ProcFood','NAmerica') = 11808.58105 ;
vmfb('c_HeavyMnfc','a_ProcFood','LatinAmer') = 6721.757324 ;
vmfb('c_HeavyMnfc','a_ProcFood','EU_28') = 36441.70703 ;
vmfb('c_HeavyMnfc','a_ProcFood','MENA') = 5498.215332 ;
vmfb('c_HeavyMnfc','a_ProcFood','SSA') = 2564.916748 ;
vmfb('c_HeavyMnfc','a_ProcFood','RestofWorld') = 6213.09375 ;
vmfb('c_HeavyMnfc','a_TextWapp','Oceania') = 312.6651917 ;
vmfb('c_HeavyMnfc','a_TextWapp','EastAsia') = 17390.5 ;
vmfb('c_HeavyMnfc','a_TextWapp','SEAsia') = 7440.202637 ;
vmfb('c_HeavyMnfc','a_TextWapp','SouthAsia') = 9051.520508 ;
vmfb('c_HeavyMnfc','a_TextWapp','NAmerica') = 9994.288086 ;
vmfb('c_HeavyMnfc','a_TextWapp','LatinAmer') = 4331.640137 ;
vmfb('c_HeavyMnfc','a_TextWapp','EU_28') = 21258.66602 ;
vmfb('c_HeavyMnfc','a_TextWapp','MENA') = 9242.720703 ;
vmfb('c_HeavyMnfc','a_TextWapp','SSA') = 1151.632935 ;
vmfb('c_HeavyMnfc','a_TextWapp','RestofWorld') = 1822.452515 ;
vmfb('c_HeavyMnfc','a_LightMnfc','Oceania') = 11016.03223 ;
vmfb('c_HeavyMnfc','a_LightMnfc','EastAsia') = 125795.2188 ;
vmfb('c_HeavyMnfc','a_LightMnfc','SEAsia') = 59485.76172 ;
vmfb('c_HeavyMnfc','a_LightMnfc','SouthAsia') = 34025.42969 ;
vmfb('c_HeavyMnfc','a_LightMnfc','NAmerica') = 123003.375 ;
vmfb('c_HeavyMnfc','a_LightMnfc','LatinAmer') = 29780.20508 ;
vmfb('c_HeavyMnfc','a_LightMnfc','EU_28') = 333484.5938 ;
vmfb('c_HeavyMnfc','a_LightMnfc','MENA') = 42640.42188 ;
vmfb('c_HeavyMnfc','a_LightMnfc','SSA') = 8157.971191 ;
vmfb('c_HeavyMnfc','a_LightMnfc','RestofWorld') = 26919.1582 ;
vmfb('c_HeavyMnfc','a_HeavyMnfc','Oceania') = 24124.69336 ;
vmfb('c_HeavyMnfc','a_HeavyMnfc','EastAsia') = 922392.25 ;
vmfb('c_HeavyMnfc','a_HeavyMnfc','SEAsia') = 279085.1875 ;
vmfb('c_HeavyMnfc','a_HeavyMnfc','SouthAsia') = 99803.19531 ;
vmfb('c_HeavyMnfc','a_HeavyMnfc','NAmerica') = 473951.0312 ;
vmfb('c_HeavyMnfc','a_HeavyMnfc','LatinAmer') = 121869.1719 ;
vmfb('c_HeavyMnfc','a_HeavyMnfc','EU_28') = 1010025.875 ;
vmfb('c_HeavyMnfc','a_HeavyMnfc','MENA') = 114756.8984 ;
vmfb('c_HeavyMnfc','a_HeavyMnfc','SSA') = 27098.13672 ;
vmfb('c_HeavyMnfc','a_HeavyMnfc','RestofWorld') = 161716.7656 ;
vmfb('c_HeavyMnfc','a_Util_Cons','Oceania') = 12219.66602 ;
vmfb('c_HeavyMnfc','a_Util_Cons','EastAsia') = 76361.14062 ;
vmfb('c_HeavyMnfc','a_Util_Cons','SEAsia') = 44207.96484 ;
vmfb('c_HeavyMnfc','a_Util_Cons','SouthAsia') = 23347.98438 ;
vmfb('c_HeavyMnfc','a_Util_Cons','NAmerica') = 91250.78906 ;
vmfb('c_HeavyMnfc','a_Util_Cons','LatinAmer') = 33065.45703 ;
vmfb('c_HeavyMnfc','a_Util_Cons','EU_28') = 201655 ;
vmfb('c_HeavyMnfc','a_Util_Cons','MENA') = 69091.69531 ;
vmfb('c_HeavyMnfc','a_Util_Cons','SSA') = 16767.29688 ;
vmfb('c_HeavyMnfc','a_Util_Cons','RestofWorld') = 38508.42969 ;
vmfb('c_HeavyMnfc','a_TransComm','Oceania') = 20104 ;
vmfb('c_HeavyMnfc','a_TransComm','EastAsia') = 72847.83594 ;
vmfb('c_HeavyMnfc','a_TransComm','SEAsia') = 56763.18359 ;
vmfb('c_HeavyMnfc','a_TransComm','SouthAsia') = 19604.76953 ;
vmfb('c_HeavyMnfc','a_TransComm','NAmerica') = 100628.0625 ;
vmfb('c_HeavyMnfc','a_TransComm','LatinAmer') = 37248.60547 ;
vmfb('c_HeavyMnfc','a_TransComm','EU_28') = 222428.6562 ;
vmfb('c_HeavyMnfc','a_TransComm','MENA') = 37074.25781 ;
vmfb('c_HeavyMnfc','a_TransComm','SSA') = 23140.02344 ;
vmfb('c_HeavyMnfc','a_TransComm','RestofWorld') = 40867.88281 ;
vmfb('c_HeavyMnfc','a_OthService','Oceania') = 20289.1543 ;
vmfb('c_HeavyMnfc','a_OthService','EastAsia') = 99821.26562 ;
vmfb('c_HeavyMnfc','a_OthService','SEAsia') = 22192.44141 ;
vmfb('c_HeavyMnfc','a_OthService','SouthAsia') = 6309.049316 ;
vmfb('c_HeavyMnfc','a_OthService','NAmerica') = 138074.0625 ;
vmfb('c_HeavyMnfc','a_OthService','LatinAmer') = 18513.04688 ;
vmfb('c_HeavyMnfc','a_OthService','EU_28') = 237371.3125 ;
vmfb('c_HeavyMnfc','a_OthService','MENA') = 26245.00391 ;
vmfb('c_HeavyMnfc','a_OthService','SSA') = 12485.13379 ;
vmfb('c_HeavyMnfc','a_OthService','RestofWorld') = 31865.88477 ;
vmfb('c_Util_Cons','a_agricultur','Oceania') = 10.98040676 ;
vmfb('c_Util_Cons','a_agricultur','EastAsia') = 15.51138496 ;
vmfb('c_Util_Cons','a_agricultur','SEAsia') = 18.30466652 ;
vmfb('c_Util_Cons','a_agricultur','SouthAsia') = 96.61144257 ;
vmfb('c_Util_Cons','a_agricultur','NAmerica') = 87.78820038 ;
vmfb('c_Util_Cons','a_agricultur','LatinAmer') = 45.25820923 ;
vmfb('c_Util_Cons','a_agricultur','EU_28') = 1256.404175 ;
vmfb('c_Util_Cons','a_agricultur','MENA') = 213.7208099 ;
vmfb('c_Util_Cons','a_agricultur','SSA') = 47.79948044 ;
vmfb('c_Util_Cons','a_agricultur','RestofWorld') = 453.98703 ;
vmfb('c_Util_Cons','a_Extraction','Oceania') = 72.61755371 ;
vmfb('c_Util_Cons','a_Extraction','EastAsia') = 43.4908638 ;
vmfb('c_Util_Cons','a_Extraction','SEAsia') = 274.5045166 ;
vmfb('c_Util_Cons','a_Extraction','SouthAsia') = 64.5680542 ;
vmfb('c_Util_Cons','a_Extraction','NAmerica') = 436.0178833 ;
vmfb('c_Util_Cons','a_Extraction','LatinAmer') = 335.6769104 ;
vmfb('c_Util_Cons','a_Extraction','EU_28') = 625.7515869 ;
vmfb('c_Util_Cons','a_Extraction','MENA') = 529.7523804 ;
vmfb('c_Util_Cons','a_Extraction','SSA') = 292.8496399 ;
vmfb('c_Util_Cons','a_Extraction','RestofWorld') = 1776.230957 ;
vmfb('c_Util_Cons','a_ProcFood','Oceania') = 25.84289551 ;
vmfb('c_Util_Cons','a_ProcFood','EastAsia') = 21.21903801 ;
vmfb('c_Util_Cons','a_ProcFood','SEAsia') = 25.71649742 ;
vmfb('c_Util_Cons','a_ProcFood','SouthAsia') = 18.32940674 ;
vmfb('c_Util_Cons','a_ProcFood','NAmerica') = 60.33438873 ;
vmfb('c_Util_Cons','a_ProcFood','LatinAmer') = 87.18622589 ;
vmfb('c_Util_Cons','a_ProcFood','EU_28') = 1682.598877 ;
vmfb('c_Util_Cons','a_ProcFood','MENA') = 450.8607483 ;
vmfb('c_Util_Cons','a_ProcFood','SSA') = 36.86710739 ;
vmfb('c_Util_Cons','a_ProcFood','RestofWorld') = 164.1123352 ;
vmfb('c_Util_Cons','a_TextWapp','Oceania') = 1.076112747 ;
vmfb('c_Util_Cons','a_TextWapp','EastAsia') = 88.98374176 ;
vmfb('c_Util_Cons','a_TextWapp','SEAsia') = 111.0166016 ;
vmfb('c_Util_Cons','a_TextWapp','SouthAsia') = 232.8853149 ;
vmfb('c_Util_Cons','a_TextWapp','NAmerica') = 18.75982475 ;
vmfb('c_Util_Cons','a_TextWapp','LatinAmer') = 21.88328171 ;
vmfb('c_Util_Cons','a_TextWapp','EU_28') = 314.7430725 ;
vmfb('c_Util_Cons','a_TextWapp','MENA') = 139.5011444 ;
vmfb('c_Util_Cons','a_TextWapp','SSA') = 38.16614532 ;
vmfb('c_Util_Cons','a_TextWapp','RestofWorld') = 31.53237724 ;
vmfb('c_Util_Cons','a_LightMnfc','Oceania') = 10.01731682 ;
vmfb('c_Util_Cons','a_LightMnfc','EastAsia') = 100.5463715 ;
vmfb('c_Util_Cons','a_LightMnfc','SEAsia') = 84.32733154 ;
vmfb('c_Util_Cons','a_LightMnfc','SouthAsia') = 35.10617828 ;
vmfb('c_Util_Cons','a_LightMnfc','NAmerica') = 321.203949 ;
vmfb('c_Util_Cons','a_LightMnfc','LatinAmer') = 91.37747955 ;
vmfb('c_Util_Cons','a_LightMnfc','EU_28') = 3118.682373 ;
vmfb('c_Util_Cons','a_LightMnfc','MENA') = 974.1399536 ;
vmfb('c_Util_Cons','a_LightMnfc','SSA') = 75.28288269 ;
vmfb('c_Util_Cons','a_LightMnfc','RestofWorld') = 375.4502869 ;
vmfb('c_Util_Cons','a_HeavyMnfc','Oceania') = 75.31126404 ;
vmfb('c_Util_Cons','a_HeavyMnfc','EastAsia') = 366.8891296 ;
vmfb('c_Util_Cons','a_HeavyMnfc','SEAsia') = 834.218811 ;
vmfb('c_Util_Cons','a_HeavyMnfc','SouthAsia') = 144.3707123 ;
vmfb('c_Util_Cons','a_HeavyMnfc','NAmerica') = 764.3430786 ;
vmfb('c_Util_Cons','a_HeavyMnfc','LatinAmer') = 491.4882507 ;
vmfb('c_Util_Cons','a_HeavyMnfc','EU_28') = 9805.728516 ;
vmfb('c_Util_Cons','a_HeavyMnfc','MENA') = 1769.894409 ;
vmfb('c_Util_Cons','a_HeavyMnfc','SSA') = 498.116272 ;
vmfb('c_Util_Cons','a_HeavyMnfc','RestofWorld') = 2614.067383 ;
vmfb('c_Util_Cons','a_Util_Cons','Oceania') = 142.0058746 ;
vmfb('c_Util_Cons','a_Util_Cons','EastAsia') = 811.1520996 ;
vmfb('c_Util_Cons','a_Util_Cons','SEAsia') = 2169.19043 ;
vmfb('c_Util_Cons','a_Util_Cons','SouthAsia') = 378.4008789 ;
vmfb('c_Util_Cons','a_Util_Cons','NAmerica') = 266.7277222 ;
vmfb('c_Util_Cons','a_Util_Cons','LatinAmer') = 3048.012939 ;
vmfb('c_Util_Cons','a_Util_Cons','EU_28') = 6879.925781 ;
vmfb('c_Util_Cons','a_Util_Cons','MENA') = 8548.066406 ;
vmfb('c_Util_Cons','a_Util_Cons','SSA') = 919.6744385 ;
vmfb('c_Util_Cons','a_Util_Cons','RestofWorld') = 2073.083496 ;
vmfb('c_Util_Cons','a_TransComm','Oceania') = 73.8769455 ;
vmfb('c_Util_Cons','a_TransComm','EastAsia') = 568.3583374 ;
vmfb('c_Util_Cons','a_TransComm','SEAsia') = 102.5735703 ;
vmfb('c_Util_Cons','a_TransComm','SouthAsia') = 54.18649673 ;
vmfb('c_Util_Cons','a_TransComm','NAmerica') = 802.3128052 ;
vmfb('c_Util_Cons','a_TransComm','LatinAmer') = 115.0047455 ;
vmfb('c_Util_Cons','a_TransComm','EU_28') = 3947.870361 ;
vmfb('c_Util_Cons','a_TransComm','MENA') = 925.2675781 ;
vmfb('c_Util_Cons','a_TransComm','SSA') = 254.4065094 ;
vmfb('c_Util_Cons','a_TransComm','RestofWorld') = 1700.571533 ;
vmfb('c_Util_Cons','a_OthService','Oceania') = 363.7159729 ;
vmfb('c_Util_Cons','a_OthService','EastAsia') = 3779.319092 ;
vmfb('c_Util_Cons','a_OthService','SEAsia') = 1096.154541 ;
vmfb('c_Util_Cons','a_OthService','SouthAsia') = 116.9484329 ;
vmfb('c_Util_Cons','a_OthService','NAmerica') = 2044.005981 ;
vmfb('c_Util_Cons','a_OthService','LatinAmer') = 318.7377625 ;
vmfb('c_Util_Cons','a_OthService','EU_28') = 12169.35938 ;
vmfb('c_Util_Cons','a_OthService','MENA') = 1295.912598 ;
vmfb('c_Util_Cons','a_OthService','SSA') = 539.4472656 ;
vmfb('c_Util_Cons','a_OthService','RestofWorld') = 2834.917969 ;
vmfb('c_TransComm','a_agricultur','Oceania') = 110.417923 ;
vmfb('c_TransComm','a_agricultur','EastAsia') = 1253.33606 ;
vmfb('c_TransComm','a_agricultur','SEAsia') = 246.0002136 ;
vmfb('c_TransComm','a_agricultur','SouthAsia') = 542.9604492 ;
vmfb('c_TransComm','a_agricultur','NAmerica') = 346.836853 ;
vmfb('c_TransComm','a_agricultur','LatinAmer') = 609.4765015 ;
vmfb('c_TransComm','a_agricultur','EU_28') = 1665.403931 ;
vmfb('c_TransComm','a_agricultur','MENA') = 467.8283081 ;
vmfb('c_TransComm','a_agricultur','SSA') = 877.5483398 ;
vmfb('c_TransComm','a_agricultur','RestofWorld') = 484.3108215 ;
vmfb('c_TransComm','a_Extraction','Oceania') = 1004.068176 ;
vmfb('c_TransComm','a_Extraction','EastAsia') = 2561.106689 ;
vmfb('c_TransComm','a_Extraction','SEAsia') = 613.7062378 ;
vmfb('c_TransComm','a_Extraction','SouthAsia') = 182.3853302 ;
vmfb('c_TransComm','a_Extraction','NAmerica') = 909.8754272 ;
vmfb('c_TransComm','a_Extraction','LatinAmer') = 648.0304565 ;
vmfb('c_TransComm','a_Extraction','EU_28') = 4047.753174 ;
vmfb('c_TransComm','a_Extraction','MENA') = 1511.112915 ;
vmfb('c_TransComm','a_Extraction','SSA') = 1732.644043 ;
vmfb('c_TransComm','a_Extraction','RestofWorld') = 2725.279785 ;
vmfb('c_TransComm','a_ProcFood','Oceania') = 143.3628998 ;
vmfb('c_TransComm','a_ProcFood','EastAsia') = 1947.625977 ;
vmfb('c_TransComm','a_ProcFood','SEAsia') = 498.919342 ;
vmfb('c_TransComm','a_ProcFood','SouthAsia') = 444.6357117 ;
vmfb('c_TransComm','a_ProcFood','NAmerica') = 749.4780884 ;
vmfb('c_TransComm','a_ProcFood','LatinAmer') = 1179.17749 ;
vmfb('c_TransComm','a_ProcFood','EU_28') = 8150.260254 ;
vmfb('c_TransComm','a_ProcFood','MENA') = 491.2651062 ;
vmfb('c_TransComm','a_ProcFood','SSA') = 721.1970825 ;
vmfb('c_TransComm','a_ProcFood','RestofWorld') = 564.1265259 ;
vmfb('c_TransComm','a_TextWapp','Oceania') = 14.4516964 ;
vmfb('c_TransComm','a_TextWapp','EastAsia') = 2274.126221 ;
vmfb('c_TransComm','a_TextWapp','SEAsia') = 257.8248596 ;
vmfb('c_TransComm','a_TextWapp','SouthAsia') = 375.5444946 ;
vmfb('c_TransComm','a_TextWapp','NAmerica') = 292.499939 ;
vmfb('c_TransComm','a_TextWapp','LatinAmer') = 213.7783508 ;
vmfb('c_TransComm','a_TextWapp','EU_28') = 3661.892334 ;
vmfb('c_TransComm','a_TextWapp','MENA') = 275.2165527 ;
vmfb('c_TransComm','a_TextWapp','SSA') = 278.0074158 ;
vmfb('c_TransComm','a_TextWapp','RestofWorld') = 150.3536072 ;
vmfb('c_TransComm','a_LightMnfc','Oceania') = 249.7813416 ;
vmfb('c_TransComm','a_LightMnfc','EastAsia') = 6826.622559 ;
vmfb('c_TransComm','a_LightMnfc','SEAsia') = 1117.403198 ;
vmfb('c_TransComm','a_LightMnfc','SouthAsia') = 634.0684204 ;
vmfb('c_TransComm','a_LightMnfc','NAmerica') = 2786.959473 ;
vmfb('c_TransComm','a_LightMnfc','LatinAmer') = 1047.250244 ;
vmfb('c_TransComm','a_LightMnfc','EU_28') = 25950.82422 ;
vmfb('c_TransComm','a_LightMnfc','MENA') = 778.5722656 ;
vmfb('c_TransComm','a_LightMnfc','SSA') = 971.364502 ;
vmfb('c_TransComm','a_LightMnfc','RestofWorld') = 1207.922607 ;
vmfb('c_TransComm','a_HeavyMnfc','Oceania') = 311.0870361 ;
vmfb('c_TransComm','a_HeavyMnfc','EastAsia') = 20227.53516 ;
vmfb('c_TransComm','a_HeavyMnfc','SEAsia') = 2976.983643 ;
vmfb('c_TransComm','a_HeavyMnfc','SouthAsia') = 973.8325806 ;
vmfb('c_TransComm','a_HeavyMnfc','NAmerica') = 4822.946289 ;
vmfb('c_TransComm','a_HeavyMnfc','LatinAmer') = 2293.954834 ;
vmfb('c_TransComm','a_HeavyMnfc','EU_28') = 50810.80078 ;
vmfb('c_TransComm','a_HeavyMnfc','MENA') = 1766.868774 ;
vmfb('c_TransComm','a_HeavyMnfc','SSA') = 1822.917969 ;
vmfb('c_TransComm','a_HeavyMnfc','RestofWorld') = 2192.214844 ;
vmfb('c_TransComm','a_Util_Cons','Oceania') = 334.3233337 ;
vmfb('c_TransComm','a_Util_Cons','EastAsia') = 7690.887207 ;
vmfb('c_TransComm','a_Util_Cons','SEAsia') = 1022.815186 ;
vmfb('c_TransComm','a_Util_Cons','SouthAsia') = 779.1357422 ;
vmfb('c_TransComm','a_Util_Cons','NAmerica') = 3878.960938 ;
vmfb('c_TransComm','a_Util_Cons','LatinAmer') = 599.9361572 ;
vmfb('c_TransComm','a_Util_Cons','EU_28') = 9036.358398 ;
vmfb('c_TransComm','a_Util_Cons','MENA') = 3378.817871 ;
vmfb('c_TransComm','a_Util_Cons','SSA') = 575.0618286 ;
vmfb('c_TransComm','a_Util_Cons','RestofWorld') = 3947.446777 ;
vmfb('c_TransComm','a_TransComm','Oceania') = 3643.257324 ;
vmfb('c_TransComm','a_TransComm','EastAsia') = 66696.44531 ;
vmfb('c_TransComm','a_TransComm','SEAsia') = 23803.78125 ;
vmfb('c_TransComm','a_TransComm','SouthAsia') = 2723.082764 ;
vmfb('c_TransComm','a_TransComm','NAmerica') = 45481.99219 ;
vmfb('c_TransComm','a_TransComm','LatinAmer') = 13763.67871 ;
vmfb('c_TransComm','a_TransComm','EU_28') = 150068.1406 ;
vmfb('c_TransComm','a_TransComm','MENA') = 18315.40039 ;
vmfb('c_TransComm','a_TransComm','SSA') = 4219.631836 ;
vmfb('c_TransComm','a_TransComm','RestofWorld') = 25969.84766 ;
vmfb('c_TransComm','a_OthService','Oceania') = 4726.853027 ;
vmfb('c_TransComm','a_OthService','EastAsia') = 19232.16016 ;
vmfb('c_TransComm','a_OthService','SEAsia') = 9144.78418 ;
vmfb('c_TransComm','a_OthService','SouthAsia') = 1869.75708 ;
vmfb('c_TransComm','a_OthService','NAmerica') = 58912.71484 ;
vmfb('c_TransComm','a_OthService','LatinAmer') = 2796.953613 ;
vmfb('c_TransComm','a_OthService','EU_28') = 81630.85938 ;
vmfb('c_TransComm','a_OthService','MENA') = 7096.234375 ;
vmfb('c_TransComm','a_OthService','SSA') = 3276.389648 ;
vmfb('c_TransComm','a_OthService','RestofWorld') = 4241.24707 ;
vmfb('c_OthService','a_agricultur','Oceania') = 316.3271484 ;
vmfb('c_OthService','a_agricultur','EastAsia') = 1591.979492 ;
vmfb('c_OthService','a_agricultur','SEAsia') = 191.7732086 ;
vmfb('c_OthService','a_agricultur','SouthAsia') = 440.0583496 ;
vmfb('c_OthService','a_agricultur','NAmerica') = 1853.057007 ;
vmfb('c_OthService','a_agricultur','LatinAmer') = 597.0435791 ;
vmfb('c_OthService','a_agricultur','EU_28') = 5615.425781 ;
vmfb('c_OthService','a_agricultur','MENA') = 1295.581421 ;
vmfb('c_OthService','a_agricultur','SSA') = 435.093811 ;
vmfb('c_OthService','a_agricultur','RestofWorld') = 613.786499 ;
vmfb('c_OthService','a_Extraction','Oceania') = 983.4888306 ;
vmfb('c_OthService','a_Extraction','EastAsia') = 1668.594727 ;
vmfb('c_OthService','a_Extraction','SEAsia') = 641.1038208 ;
vmfb('c_OthService','a_Extraction','SouthAsia') = 532.6539917 ;
vmfb('c_OthService','a_Extraction','NAmerica') = 3529.929688 ;
vmfb('c_OthService','a_Extraction','LatinAmer') = 4054.148193 ;
vmfb('c_OthService','a_Extraction','EU_28') = 2942.002197 ;
vmfb('c_OthService','a_Extraction','MENA') = 8428.52832 ;
vmfb('c_OthService','a_Extraction','SSA') = 4328.69873 ;
vmfb('c_OthService','a_Extraction','RestofWorld') = 5412.917969 ;
vmfb('c_OthService','a_ProcFood','Oceania') = 304.1353149 ;
vmfb('c_OthService','a_ProcFood','EastAsia') = 2394.13501 ;
vmfb('c_OthService','a_ProcFood','SEAsia') = 324.1337891 ;
vmfb('c_OthService','a_ProcFood','SouthAsia') = 1194.966919 ;
vmfb('c_OthService','a_ProcFood','NAmerica') = 3582.988037 ;
vmfb('c_OthService','a_ProcFood','LatinAmer') = 1389.328369 ;
vmfb('c_OthService','a_ProcFood','EU_28') = 16099.23242 ;
vmfb('c_OthService','a_ProcFood','MENA') = 1191.106201 ;
vmfb('c_OthService','a_ProcFood','SSA') = 732.43396 ;
vmfb('c_OthService','a_ProcFood','RestofWorld') = 1070.845581 ;
vmfb('c_OthService','a_TextWapp','Oceania') = 37.29088593 ;
vmfb('c_OthService','a_TextWapp','EastAsia') = 2109.021973 ;
vmfb('c_OthService','a_TextWapp','SEAsia') = 236.0016937 ;
vmfb('c_OthService','a_TextWapp','SouthAsia') = 3205.574219 ;
vmfb('c_OthService','a_TextWapp','NAmerica') = 558.5511475 ;
vmfb('c_OthService','a_TextWapp','LatinAmer') = 377.1375732 ;
vmfb('c_OthService','a_TextWapp','EU_28') = 3850.76123 ;
vmfb('c_OthService','a_TextWapp','MENA') = 1478.220947 ;
vmfb('c_OthService','a_TextWapp','SSA') = 175.357132 ;
vmfb('c_OthService','a_TextWapp','RestofWorld') = 115.9647751 ;
vmfb('c_OthService','a_LightMnfc','Oceania') = 542.3000488 ;
vmfb('c_OthService','a_LightMnfc','EastAsia') = 9545.749023 ;
vmfb('c_OthService','a_LightMnfc','SEAsia') = 816.7984619 ;
vmfb('c_OthService','a_LightMnfc','SouthAsia') = 3088.746094 ;
vmfb('c_OthService','a_LightMnfc','NAmerica') = 9387.587891 ;
vmfb('c_OthService','a_LightMnfc','LatinAmer') = 1808.911133 ;
vmfb('c_OthService','a_LightMnfc','EU_28') = 36200.71484 ;
vmfb('c_OthService','a_LightMnfc','MENA') = 3997.092529 ;
vmfb('c_OthService','a_LightMnfc','SSA') = 508.4765625 ;
vmfb('c_OthService','a_LightMnfc','RestofWorld') = 1727.183838 ;
vmfb('c_OthService','a_HeavyMnfc','Oceania') = 524.8574219 ;
vmfb('c_OthService','a_HeavyMnfc','EastAsia') = 24863.86133 ;
vmfb('c_OthService','a_HeavyMnfc','SEAsia') = 9991.740234 ;
vmfb('c_OthService','a_HeavyMnfc','SouthAsia') = 5217.031738 ;
vmfb('c_OthService','a_HeavyMnfc','NAmerica') = 19271.85547 ;
vmfb('c_OthService','a_HeavyMnfc','LatinAmer') = 3383.075439 ;
vmfb('c_OthService','a_HeavyMnfc','EU_28') = 67949.35938 ;
vmfb('c_OthService','a_HeavyMnfc','MENA') = 8852.551758 ;
vmfb('c_OthService','a_HeavyMnfc','SSA') = 1063.630615 ;
vmfb('c_OthService','a_HeavyMnfc','RestofWorld') = 3490.965332 ;
vmfb('c_OthService','a_Util_Cons','Oceania') = 1332.564819 ;
vmfb('c_OthService','a_Util_Cons','EastAsia') = 9141.079102 ;
vmfb('c_OthService','a_Util_Cons','SEAsia') = 5494.052734 ;
vmfb('c_OthService','a_Util_Cons','SouthAsia') = 1874.40979 ;
vmfb('c_OthService','a_Util_Cons','NAmerica') = 17932.58789 ;
vmfb('c_OthService','a_Util_Cons','LatinAmer') = 2004.430176 ;
vmfb('c_OthService','a_Util_Cons','EU_28') = 31287.37305 ;
vmfb('c_OthService','a_Util_Cons','MENA') = 8339.208008 ;
vmfb('c_OthService','a_Util_Cons','SSA') = 2087.615234 ;
vmfb('c_OthService','a_Util_Cons','RestofWorld') = 3068.402832 ;
vmfb('c_OthService','a_TransComm','Oceania') = 3698.924072 ;
vmfb('c_OthService','a_TransComm','EastAsia') = 36355.32031 ;
vmfb('c_OthService','a_TransComm','SEAsia') = 10248.21094 ;
vmfb('c_OthService','a_TransComm','SouthAsia') = 6833.327148 ;
vmfb('c_OthService','a_TransComm','NAmerica') = 53368.61719 ;
vmfb('c_OthService','a_TransComm','LatinAmer') = 10691.6543 ;
vmfb('c_OthService','a_TransComm','EU_28') = 86609.98438 ;
vmfb('c_OthService','a_TransComm','MENA') = 12470.00977 ;
vmfb('c_OthService','a_TransComm','SSA') = 5248.005371 ;
vmfb('c_OthService','a_TransComm','RestofWorld') = 11145.79395 ;
vmfb('c_OthService','a_OthService','Oceania') = 10794.20801 ;
vmfb('c_OthService','a_OthService','EastAsia') = 57412.15234 ;
vmfb('c_OthService','a_OthService','SEAsia') = 24520.78125 ;
vmfb('c_OthService','a_OthService','SouthAsia') = 11908.76562 ;
vmfb('c_OthService','a_OthService','NAmerica') = 130576.7266 ;
vmfb('c_OthService','a_OthService','LatinAmer') = 17923.03516 ;
vmfb('c_OthService','a_OthService','EU_28') = 301089 ;
vmfb('c_OthService','a_OthService','MENA') = 17658.5625 ;
vmfb('c_OthService','a_OthService','SSA') = 8289.605469 ;
vmfb('c_OthService','a_OthService','RestofWorld') = 34579.28906 ;

* vmfp data (900 cells)
vmfp('c_Crops','a_agricultur','Oceania') = 409.9712219 ;
vmfp('c_Crops','a_agricultur','EastAsia') = 8913.141602 ;
vmfp('c_Crops','a_agricultur','SEAsia') = 2689.907471 ;
vmfp('c_Crops','a_agricultur','SouthAsia') = 2030.407593 ;
vmfp('c_Crops','a_agricultur','NAmerica') = 3880.711914 ;
vmfp('c_Crops','a_agricultur','LatinAmer') = 3199.217041 ;
vmfp('c_Crops','a_agricultur','EU_28') = 14517.29297 ;
vmfp('c_Crops','a_agricultur','MENA') = 8899.974609 ;
vmfp('c_Crops','a_agricultur','SSA') = 1772.067993 ;
vmfp('c_Crops','a_agricultur','RestofWorld') = 5829.54541 ;
vmfp('c_Crops','a_Extraction','Oceania') = 13.49488258 ;
vmfp('c_Crops','a_Extraction','EastAsia') = 355.0088196 ;
vmfp('c_Crops','a_Extraction','SEAsia') = 17.47990227 ;
vmfp('c_Crops','a_Extraction','SouthAsia') = 9.732661247 ;
vmfp('c_Crops','a_Extraction','NAmerica') = 79.10799408 ;
vmfp('c_Crops','a_Extraction','LatinAmer') = 93.07389069 ;
vmfp('c_Crops','a_Extraction','EU_28') = 256.9627075 ;
vmfp('c_Crops','a_Extraction','MENA') = 432.3305359 ;
vmfp('c_Crops','a_Extraction','SSA') = 43.92572784 ;
vmfp('c_Crops','a_Extraction','RestofWorld') = 41.04860687 ;
vmfp('c_Crops','a_ProcFood','Oceania') = 1252.597412 ;
vmfp('c_Crops','a_ProcFood','EastAsia') = 69168.57031 ;
vmfp('c_Crops','a_ProcFood','SEAsia') = 13558.72266 ;
vmfp('c_Crops','a_ProcFood','SouthAsia') = 3600.44873 ;
vmfp('c_Crops','a_ProcFood','NAmerica') = 24891.03906 ;
vmfp('c_Crops','a_ProcFood','LatinAmer') = 10132.99707 ;
vmfp('c_Crops','a_ProcFood','EU_28') = 49277.76953 ;
vmfp('c_Crops','a_ProcFood','MENA') = 19960.12891 ;
vmfp('c_Crops','a_ProcFood','SSA') = 3798.70459 ;
vmfp('c_Crops','a_ProcFood','RestofWorld') = 11463.35547 ;
vmfp('c_Crops','a_TextWapp','Oceania') = 3.917861938 ;
vmfp('c_Crops','a_TextWapp','EastAsia') = 13333.10352 ;
vmfp('c_Crops','a_TextWapp','SEAsia') = 3346.662109 ;
vmfp('c_Crops','a_TextWapp','SouthAsia') = 2452.241699 ;
vmfp('c_Crops','a_TextWapp','NAmerica') = 647.3112793 ;
vmfp('c_Crops','a_TextWapp','LatinAmer') = 577.057373 ;
vmfp('c_Crops','a_TextWapp','EU_28') = 2472.567627 ;
vmfp('c_Crops','a_TextWapp','MENA') = 2850.927002 ;
vmfp('c_Crops','a_TextWapp','SSA') = 231.4829102 ;
vmfp('c_Crops','a_TextWapp','RestofWorld') = 290.6423035 ;
vmfp('c_Crops','a_LightMnfc','Oceania') = 3.998417854 ;
vmfp('c_Crops','a_LightMnfc','EastAsia') = 1503.02832 ;
vmfp('c_Crops','a_LightMnfc','SEAsia') = 312.1921387 ;
vmfp('c_Crops','a_LightMnfc','SouthAsia') = 87.03527069 ;
vmfp('c_Crops','a_LightMnfc','NAmerica') = 145.541748 ;
vmfp('c_Crops','a_LightMnfc','LatinAmer') = 45.45517731 ;
vmfp('c_Crops','a_LightMnfc','EU_28') = 437.2658997 ;
vmfp('c_Crops','a_LightMnfc','MENA') = 768.5335083 ;
vmfp('c_Crops','a_LightMnfc','SSA') = 418.559967 ;
vmfp('c_Crops','a_LightMnfc','RestofWorld') = 136.0675354 ;
vmfp('c_Crops','a_HeavyMnfc','Oceania') = 67.37784576 ;
vmfp('c_Crops','a_HeavyMnfc','EastAsia') = 8441.8125 ;
vmfp('c_Crops','a_HeavyMnfc','SEAsia') = 1823.900879 ;
vmfp('c_Crops','a_HeavyMnfc','SouthAsia') = 242.9030914 ;
vmfp('c_Crops','a_HeavyMnfc','NAmerica') = 1447.677368 ;
vmfp('c_Crops','a_HeavyMnfc','LatinAmer') = 210.9399109 ;
vmfp('c_Crops','a_HeavyMnfc','EU_28') = 3270.857178 ;
vmfp('c_Crops','a_HeavyMnfc','MENA') = 954.0062256 ;
vmfp('c_Crops','a_HeavyMnfc','SSA') = 346.254364 ;
vmfp('c_Crops','a_HeavyMnfc','RestofWorld') = 200.1822052 ;
vmfp('c_Crops','a_Util_Cons','Oceania') = 11.06129742 ;
vmfp('c_Crops','a_Util_Cons','EastAsia') = 354.0631714 ;
vmfp('c_Crops','a_Util_Cons','SEAsia') = 7.733785152 ;
vmfp('c_Crops','a_Util_Cons','SouthAsia') = 46.38638306 ;
vmfp('c_Crops','a_Util_Cons','NAmerica') = 1175.654297 ;
vmfp('c_Crops','a_Util_Cons','LatinAmer') = 29.82790184 ;
vmfp('c_Crops','a_Util_Cons','EU_28') = 1122.868164 ;
vmfp('c_Crops','a_Util_Cons','MENA') = 97.76766968 ;
vmfp('c_Crops','a_Util_Cons','SSA') = 14.90895176 ;
vmfp('c_Crops','a_Util_Cons','RestofWorld') = 75.24523163 ;
vmfp('c_Crops','a_TransComm','Oceania') = 276.0535278 ;
vmfp('c_Crops','a_TransComm','EastAsia') = 4704.043457 ;
vmfp('c_Crops','a_TransComm','SEAsia') = 1517.036743 ;
vmfp('c_Crops','a_TransComm','SouthAsia') = 409.5995178 ;
vmfp('c_Crops','a_TransComm','NAmerica') = 1346.616455 ;
vmfp('c_Crops','a_TransComm','LatinAmer') = 663.8968506 ;
vmfp('c_Crops','a_TransComm','EU_28') = 5598.101074 ;
vmfp('c_Crops','a_TransComm','MENA') = 2311.76123 ;
vmfp('c_Crops','a_TransComm','SSA') = 336.4665222 ;
vmfp('c_Crops','a_TransComm','RestofWorld') = 1115.61853 ;
vmfp('c_Crops','a_OthService','Oceania') = 130.0715485 ;
vmfp('c_Crops','a_OthService','EastAsia') = 2187.540283 ;
vmfp('c_Crops','a_OthService','SEAsia') = 186.5507202 ;
vmfp('c_Crops','a_OthService','SouthAsia') = 99.133461 ;
vmfp('c_Crops','a_OthService','NAmerica') = 5350.258789 ;
vmfp('c_Crops','a_OthService','LatinAmer') = 308.6899414 ;
vmfp('c_Crops','a_OthService','EU_28') = 5810.605469 ;
vmfp('c_Crops','a_OthService','MENA') = 1716.804932 ;
vmfp('c_Crops','a_OthService','SSA') = 350.8143311 ;
vmfp('c_Crops','a_OthService','RestofWorld') = 1428.519653 ;
vmfp('c_MeatLstk','a_agricultur','Oceania') = 133.9465027 ;
vmfp('c_MeatLstk','a_agricultur','EastAsia') = 3414.39917 ;
vmfp('c_MeatLstk','a_agricultur','SEAsia') = 575.0391235 ;
vmfp('c_MeatLstk','a_agricultur','SouthAsia') = 157.0627441 ;
vmfp('c_MeatLstk','a_agricultur','NAmerica') = 6636.769531 ;
vmfp('c_MeatLstk','a_agricultur','LatinAmer') = 1276.408325 ;
vmfp('c_MeatLstk','a_agricultur','EU_28') = 13421.55762 ;
vmfp('c_MeatLstk','a_agricultur','MENA') = 4218.247559 ;
vmfp('c_MeatLstk','a_agricultur','SSA') = 444.9457703 ;
vmfp('c_MeatLstk','a_agricultur','RestofWorld') = 2922.253174 ;
vmfp('c_MeatLstk','a_Extraction','Oceania') = 1.406763554 ;
vmfp('c_MeatLstk','a_Extraction','EastAsia') = 14.43925571 ;
vmfp('c_MeatLstk','a_Extraction','SEAsia') = 17.77701187 ;
vmfp('c_MeatLstk','a_Extraction','SouthAsia') = 2.179729462 ;
vmfp('c_MeatLstk','a_Extraction','NAmerica') = 67.42585754 ;
vmfp('c_MeatLstk','a_Extraction','LatinAmer') = 4.660620689 ;
vmfp('c_MeatLstk','a_Extraction','EU_28') = 75.28478241 ;
vmfp('c_MeatLstk','a_Extraction','MENA') = 25.19254684 ;
vmfp('c_MeatLstk','a_Extraction','SSA') = 9.413448334 ;
vmfp('c_MeatLstk','a_Extraction','RestofWorld') = 80.66946411 ;
vmfp('c_MeatLstk','a_ProcFood','Oceania') = 159.180191 ;
vmfp('c_MeatLstk','a_ProcFood','EastAsia') = 7694.527832 ;
vmfp('c_MeatLstk','a_ProcFood','SEAsia') = 776.8787842 ;
vmfp('c_MeatLstk','a_ProcFood','SouthAsia') = 40.93994141 ;
vmfp('c_MeatLstk','a_ProcFood','NAmerica') = 1336.799927 ;
vmfp('c_MeatLstk','a_ProcFood','LatinAmer') = 257.628418 ;
vmfp('c_MeatLstk','a_ProcFood','EU_28') = 7591.763672 ;
vmfp('c_MeatLstk','a_ProcFood','MENA') = 1129.327026 ;
vmfp('c_MeatLstk','a_ProcFood','SSA') = 141.4814606 ;
vmfp('c_MeatLstk','a_ProcFood','RestofWorld') = 427.3331909 ;
vmfp('c_MeatLstk','a_TextWapp','Oceania') = 6.699580193 ;
vmfp('c_MeatLstk','a_TextWapp','EastAsia') = 3834.720703 ;
vmfp('c_MeatLstk','a_TextWapp','SEAsia') = 123.5802002 ;
vmfp('c_MeatLstk','a_TextWapp','SouthAsia') = 23.8580513 ;
vmfp('c_MeatLstk','a_TextWapp','NAmerica') = 356.1141663 ;
vmfp('c_MeatLstk','a_TextWapp','LatinAmer') = 110.1925812 ;
vmfp('c_MeatLstk','a_TextWapp','EU_28') = 794.1949463 ;
vmfp('c_MeatLstk','a_TextWapp','MENA') = 651.1525879 ;
vmfp('c_MeatLstk','a_TextWapp','SSA') = 92.17822266 ;
vmfp('c_MeatLstk','a_TextWapp','RestofWorld') = 107.59198 ;
vmfp('c_MeatLstk','a_LightMnfc','Oceania') = 6.61125803 ;
vmfp('c_MeatLstk','a_LightMnfc','EastAsia') = 5819.554688 ;
vmfp('c_MeatLstk','a_LightMnfc','SEAsia') = 831.0637817 ;
vmfp('c_MeatLstk','a_LightMnfc','SouthAsia') = 136.2564697 ;
vmfp('c_MeatLstk','a_LightMnfc','NAmerica') = 864.1254883 ;
vmfp('c_MeatLstk','a_LightMnfc','LatinAmer') = 205.3310089 ;
vmfp('c_MeatLstk','a_LightMnfc','EU_28') = 1036.860718 ;
vmfp('c_MeatLstk','a_LightMnfc','MENA') = 866.7557373 ;
vmfp('c_MeatLstk','a_LightMnfc','SSA') = 61.72673035 ;
vmfp('c_MeatLstk','a_LightMnfc','RestofWorld') = 251.4047699 ;
vmfp('c_MeatLstk','a_HeavyMnfc','Oceania') = 15.74754524 ;
vmfp('c_MeatLstk','a_HeavyMnfc','EastAsia') = 1071.388428 ;
vmfp('c_MeatLstk','a_HeavyMnfc','SEAsia') = 97.34387207 ;
vmfp('c_MeatLstk','a_HeavyMnfc','SouthAsia') = 15.44259834 ;
vmfp('c_MeatLstk','a_HeavyMnfc','NAmerica') = 457.4449158 ;
vmfp('c_MeatLstk','a_HeavyMnfc','LatinAmer') = 70.44506073 ;
vmfp('c_MeatLstk','a_HeavyMnfc','EU_28') = 593.5616455 ;
vmfp('c_MeatLstk','a_HeavyMnfc','MENA') = 285.5663757 ;
vmfp('c_MeatLstk','a_HeavyMnfc','SSA') = 20.10767174 ;
vmfp('c_MeatLstk','a_HeavyMnfc','RestofWorld') = 130.4233093 ;
vmfp('c_MeatLstk','a_Util_Cons','Oceania') = 2.641120195 ;
vmfp('c_MeatLstk','a_Util_Cons','EastAsia') = 5.154036999 ;
vmfp('c_MeatLstk','a_Util_Cons','SEAsia') = 1.429030061 ;
vmfp('c_MeatLstk','a_Util_Cons','SouthAsia') = 4.495933056 ;
vmfp('c_MeatLstk','a_Util_Cons','NAmerica') = 14.04214573 ;
vmfp('c_MeatLstk','a_Util_Cons','LatinAmer') = 19.77370834 ;
vmfp('c_MeatLstk','a_Util_Cons','EU_28') = 59.07350922 ;
vmfp('c_MeatLstk','a_Util_Cons','MENA') = 20.56485748 ;
vmfp('c_MeatLstk','a_Util_Cons','SSA') = 7.408383369 ;
vmfp('c_MeatLstk','a_Util_Cons','RestofWorld') = 18.68823433 ;
vmfp('c_MeatLstk','a_TransComm','Oceania') = 235.7287903 ;
vmfp('c_MeatLstk','a_TransComm','EastAsia') = 13058.85449 ;
vmfp('c_MeatLstk','a_TransComm','SEAsia') = 603.861084 ;
vmfp('c_MeatLstk','a_TransComm','SouthAsia') = 71.39277649 ;
vmfp('c_MeatLstk','a_TransComm','NAmerica') = 1253.297363 ;
vmfp('c_MeatLstk','a_TransComm','LatinAmer') = 1356.671509 ;
vmfp('c_MeatLstk','a_TransComm','EU_28') = 5131.33252 ;
vmfp('c_MeatLstk','a_TransComm','MENA') = 1359.821411 ;
vmfp('c_MeatLstk','a_TransComm','SSA') = 329.8420715 ;
vmfp('c_MeatLstk','a_TransComm','RestofWorld') = 883.1560669 ;
vmfp('c_MeatLstk','a_OthService','Oceania') = 98.39219666 ;
vmfp('c_MeatLstk','a_OthService','EastAsia') = 1164.519897 ;
vmfp('c_MeatLstk','a_OthService','SEAsia') = 105.7642365 ;
vmfp('c_MeatLstk','a_OthService','SouthAsia') = 82.35014343 ;
vmfp('c_MeatLstk','a_OthService','NAmerica') = 872.3966675 ;
vmfp('c_MeatLstk','a_OthService','LatinAmer') = 259.9705811 ;
vmfp('c_MeatLstk','a_OthService','EU_28') = 3467.586426 ;
vmfp('c_MeatLstk','a_OthService','MENA') = 610.3718872 ;
vmfp('c_MeatLstk','a_OthService','SSA') = 203.870575 ;
vmfp('c_MeatLstk','a_OthService','RestofWorld') = 710.887146 ;
vmfp('c_Extraction','a_agricultur','Oceania') = 29.05676079 ;
vmfp('c_Extraction','a_agricultur','EastAsia') = 259.145874 ;
vmfp('c_Extraction','a_agricultur','SEAsia') = 29.80190659 ;
vmfp('c_Extraction','a_agricultur','SouthAsia') = 17.14781761 ;
vmfp('c_Extraction','a_agricultur','NAmerica') = 172.7495575 ;
vmfp('c_Extraction','a_agricultur','LatinAmer') = 90.85786438 ;
vmfp('c_Extraction','a_agricultur','EU_28') = 2252.513428 ;
vmfp('c_Extraction','a_agricultur','MENA') = 43.79303741 ;
vmfp('c_Extraction','a_agricultur','SSA') = 44.91720581 ;
vmfp('c_Extraction','a_agricultur','RestofWorld') = 326.7561646 ;
vmfp('c_Extraction','a_Extraction','Oceania') = 55.85814285 ;
vmfp('c_Extraction','a_Extraction','EastAsia') = 19075.55078 ;
vmfp('c_Extraction','a_Extraction','SEAsia') = 317.2108765 ;
vmfp('c_Extraction','a_Extraction','SouthAsia') = 235.1899109 ;
vmfp('c_Extraction','a_Extraction','NAmerica') = 2943.508057 ;
vmfp('c_Extraction','a_Extraction','LatinAmer') = 1336.92395 ;
vmfp('c_Extraction','a_Extraction','EU_28') = 6695.48584 ;
vmfp('c_Extraction','a_Extraction','MENA') = 1539.72522 ;
vmfp('c_Extraction','a_Extraction','SSA') = 440.8325195 ;
vmfp('c_Extraction','a_Extraction','RestofWorld') = 2231.925781 ;
vmfp('c_Extraction','a_ProcFood','Oceania') = 77.67798615 ;
vmfp('c_Extraction','a_ProcFood','EastAsia') = 2911.652832 ;
vmfp('c_Extraction','a_ProcFood','SEAsia') = 550.1733398 ;
vmfp('c_Extraction','a_ProcFood','SouthAsia') = 173.7280121 ;
vmfp('c_Extraction','a_ProcFood','NAmerica') = 2024.397949 ;
vmfp('c_Extraction','a_ProcFood','LatinAmer') = 168.7303619 ;
vmfp('c_Extraction','a_ProcFood','EU_28') = 8425.195312 ;
vmfp('c_Extraction','a_ProcFood','MENA') = 723.2919922 ;
vmfp('c_Extraction','a_ProcFood','SSA') = 55.49636459 ;
vmfp('c_Extraction','a_ProcFood','RestofWorld') = 619.0639038 ;
vmfp('c_Extraction','a_TextWapp','Oceania') = 1.476010799 ;
vmfp('c_Extraction','a_TextWapp','EastAsia') = 344.1925354 ;
vmfp('c_Extraction','a_TextWapp','SEAsia') = 45.5292511 ;
vmfp('c_Extraction','a_TextWapp','SouthAsia') = 116.607048 ;
vmfp('c_Extraction','a_TextWapp','NAmerica') = 28.41861534 ;
vmfp('c_Extraction','a_TextWapp','LatinAmer') = 29.47272873 ;
vmfp('c_Extraction','a_TextWapp','EU_28') = 825.036377 ;
vmfp('c_Extraction','a_TextWapp','MENA') = 319.8063965 ;
vmfp('c_Extraction','a_TextWapp','SSA') = 1.951301813 ;
vmfp('c_Extraction','a_TextWapp','RestofWorld') = 39.28739166 ;
vmfp('c_Extraction','a_LightMnfc','Oceania') = 87.31706238 ;
vmfp('c_Extraction','a_LightMnfc','EastAsia') = 13510.30957 ;
vmfp('c_Extraction','a_LightMnfc','SEAsia') = 662.4810791 ;
vmfp('c_Extraction','a_LightMnfc','SouthAsia') = 5999.528809 ;
vmfp('c_Extraction','a_LightMnfc','NAmerica') = 1716.416138 ;
vmfp('c_Extraction','a_LightMnfc','LatinAmer') = 533.7213135 ;
vmfp('c_Extraction','a_LightMnfc','EU_28') = 16223.84863 ;
vmfp('c_Extraction','a_LightMnfc','MENA') = 3995.427246 ;
vmfp('c_Extraction','a_LightMnfc','SSA') = 158.0197296 ;
vmfp('c_Extraction','a_LightMnfc','RestofWorld') = 920.635376 ;
vmfp('c_Extraction','a_HeavyMnfc','Oceania') = 18535.5 ;
vmfp('c_Extraction','a_HeavyMnfc','EastAsia') = 732765.375 ;
vmfp('c_Extraction','a_HeavyMnfc','SEAsia') = 101637.5781 ;
vmfp('c_Extraction','a_HeavyMnfc','SouthAsia') = 156181.625 ;
vmfp('c_Extraction','a_HeavyMnfc','NAmerica') = 366050.75 ;
vmfp('c_Extraction','a_HeavyMnfc','LatinAmer') = 44938.10938 ;
vmfp('c_Extraction','a_HeavyMnfc','EU_28') = 500021.9375 ;
vmfp('c_Extraction','a_HeavyMnfc','MENA') = 44887.79688 ;
vmfp('c_Extraction','a_HeavyMnfc','SSA') = 25921.58789 ;
vmfp('c_Extraction','a_HeavyMnfc','RestofWorld') = 36265.71094 ;
vmfp('c_Extraction','a_Util_Cons','Oceania') = 91.13179779 ;
vmfp('c_Extraction','a_Util_Cons','EastAsia') = 109841.8438 ;
vmfp('c_Extraction','a_Util_Cons','SEAsia') = 8159.15625 ;
vmfp('c_Extraction','a_Util_Cons','SouthAsia') = 33980.625 ;
vmfp('c_Extraction','a_Util_Cons','NAmerica') = 9069.27832 ;
vmfp('c_Extraction','a_Util_Cons','LatinAmer') = 5034.443848 ;
vmfp('c_Extraction','a_Util_Cons','EU_28') = 88835.90625 ;
vmfp('c_Extraction','a_Util_Cons','MENA') = 24630.62891 ;
vmfp('c_Extraction','a_Util_Cons','SSA') = 1039.381104 ;
vmfp('c_Extraction','a_Util_Cons','RestofWorld') = 17050.91602 ;
vmfp('c_Extraction','a_TransComm','Oceania') = 73.32513428 ;
vmfp('c_Extraction','a_TransComm','EastAsia') = 11728.82715 ;
vmfp('c_Extraction','a_TransComm','SEAsia') = 392.1959229 ;
vmfp('c_Extraction','a_TransComm','SouthAsia') = 400.7587585 ;
vmfp('c_Extraction','a_TransComm','NAmerica') = 940.6528931 ;
vmfp('c_Extraction','a_TransComm','LatinAmer') = 445.5630188 ;
vmfp('c_Extraction','a_TransComm','EU_28') = 13159.17676 ;
vmfp('c_Extraction','a_TransComm','MENA') = 1549.162476 ;
vmfp('c_Extraction','a_TransComm','SSA') = 64.87799072 ;
vmfp('c_Extraction','a_TransComm','RestofWorld') = 2473.944092 ;
vmfp('c_Extraction','a_OthService','Oceania') = 75.54494476 ;
vmfp('c_Extraction','a_OthService','EastAsia') = 2836.939697 ;
vmfp('c_Extraction','a_OthService','SEAsia') = 70.94681549 ;
vmfp('c_Extraction','a_OthService','SouthAsia') = 28.58157158 ;
vmfp('c_Extraction','a_OthService','NAmerica') = 1257.847412 ;
vmfp('c_Extraction','a_OthService','LatinAmer') = 80.38851929 ;
vmfp('c_Extraction','a_OthService','EU_28') = 13735.30566 ;
vmfp('c_Extraction','a_OthService','MENA') = 131.1264496 ;
vmfp('c_Extraction','a_OthService','SSA') = 136.7644806 ;
vmfp('c_Extraction','a_OthService','RestofWorld') = 423.8627625 ;
vmfp('c_ProcFood','a_agricultur','Oceania') = 179.7180786 ;
vmfp('c_ProcFood','a_agricultur','EastAsia') = 3674.47168 ;
vmfp('c_ProcFood','a_agricultur','SEAsia') = 2306.555664 ;
vmfp('c_ProcFood','a_agricultur','SouthAsia') = 1975.909912 ;
vmfp('c_ProcFood','a_agricultur','NAmerica') = 2218.774902 ;
vmfp('c_ProcFood','a_agricultur','LatinAmer') = 2347.090332 ;
vmfp('c_ProcFood','a_agricultur','EU_28') = 18127.67773 ;
vmfp('c_ProcFood','a_agricultur','MENA') = 3133.542725 ;
vmfp('c_ProcFood','a_agricultur','SSA') = 981.2189941 ;
vmfp('c_ProcFood','a_agricultur','RestofWorld') = 1812.922363 ;
vmfp('c_ProcFood','a_Extraction','Oceania') = 55.88679504 ;
vmfp('c_ProcFood','a_Extraction','EastAsia') = 602.2800293 ;
vmfp('c_ProcFood','a_Extraction','SEAsia') = 237.9932861 ;
vmfp('c_ProcFood','a_Extraction','SouthAsia') = 30.84228516 ;
vmfp('c_ProcFood','a_Extraction','NAmerica') = 85.12388611 ;
vmfp('c_ProcFood','a_Extraction','LatinAmer') = 189.8962555 ;
vmfp('c_ProcFood','a_Extraction','EU_28') = 415.9726257 ;
vmfp('c_ProcFood','a_Extraction','MENA') = 144.6561432 ;
vmfp('c_ProcFood','a_Extraction','SSA') = 115.1981659 ;
vmfp('c_ProcFood','a_Extraction','RestofWorld') = 228.6036682 ;
vmfp('c_ProcFood','a_ProcFood','Oceania') = 1988.021118 ;
vmfp('c_ProcFood','a_ProcFood','EastAsia') = 34634.42969 ;
vmfp('c_ProcFood','a_ProcFood','SEAsia') = 17886.73633 ;
vmfp('c_ProcFood','a_ProcFood','SouthAsia') = 4372.502441 ;
vmfp('c_ProcFood','a_ProcFood','NAmerica') = 29643.10156 ;
vmfp('c_ProcFood','a_ProcFood','LatinAmer') = 10394.18066 ;
vmfp('c_ProcFood','a_ProcFood','EU_28') = 65337.03125 ;
vmfp('c_ProcFood','a_ProcFood','MENA') = 14275.95508 ;
vmfp('c_ProcFood','a_ProcFood','SSA') = 4960.197754 ;
vmfp('c_ProcFood','a_ProcFood','RestofWorld') = 11326.02246 ;
vmfp('c_ProcFood','a_TextWapp','Oceania') = 2.388509035 ;
vmfp('c_ProcFood','a_TextWapp','EastAsia') = 138.5906067 ;
vmfp('c_ProcFood','a_TextWapp','SEAsia') = 33.01973724 ;
vmfp('c_ProcFood','a_TextWapp','SouthAsia') = 34.20780182 ;
vmfp('c_ProcFood','a_TextWapp','NAmerica') = 20.06487846 ;
vmfp('c_ProcFood','a_TextWapp','LatinAmer') = 16.34663582 ;
vmfp('c_ProcFood','a_TextWapp','EU_28') = 529.5247192 ;
vmfp('c_ProcFood','a_TextWapp','MENA') = 106.7258301 ;
vmfp('c_ProcFood','a_TextWapp','SSA') = 71.97603607 ;
vmfp('c_ProcFood','a_TextWapp','RestofWorld') = 58.25073624 ;
vmfp('c_ProcFood','a_LightMnfc','Oceania') = 98.2117157 ;
vmfp('c_ProcFood','a_LightMnfc','EastAsia') = 567.8294678 ;
vmfp('c_ProcFood','a_LightMnfc','SEAsia') = 55.11190033 ;
vmfp('c_ProcFood','a_LightMnfc','SouthAsia') = 68.19953918 ;
vmfp('c_ProcFood','a_LightMnfc','NAmerica') = 435.1402588 ;
vmfp('c_ProcFood','a_LightMnfc','LatinAmer') = 105.9268875 ;
vmfp('c_ProcFood','a_LightMnfc','EU_28') = 2894.309082 ;
vmfp('c_ProcFood','a_LightMnfc','MENA') = 137.4673767 ;
vmfp('c_ProcFood','a_LightMnfc','SSA') = 140.488327 ;
vmfp('c_ProcFood','a_LightMnfc','RestofWorld') = 140.416214 ;
vmfp('c_ProcFood','a_HeavyMnfc','Oceania') = 252.9959717 ;
vmfp('c_ProcFood','a_HeavyMnfc','EastAsia') = 3333.168457 ;
vmfp('c_ProcFood','a_HeavyMnfc','SEAsia') = 1660.486328 ;
vmfp('c_ProcFood','a_HeavyMnfc','SouthAsia') = 449.4811096 ;
vmfp('c_ProcFood','a_HeavyMnfc','NAmerica') = 2287.077148 ;
vmfp('c_ProcFood','a_HeavyMnfc','LatinAmer') = 870.6881714 ;
vmfp('c_ProcFood','a_HeavyMnfc','EU_28') = 4994.755859 ;
vmfp('c_ProcFood','a_HeavyMnfc','MENA') = 416.5029602 ;
vmfp('c_ProcFood','a_HeavyMnfc','SSA') = 380.7593079 ;
vmfp('c_ProcFood','a_HeavyMnfc','RestofWorld') = 282.4179382 ;
vmfp('c_ProcFood','a_Util_Cons','Oceania') = 30.43476868 ;
vmfp('c_ProcFood','a_Util_Cons','EastAsia') = 153.3267975 ;
vmfp('c_ProcFood','a_Util_Cons','SEAsia') = 31.71809196 ;
vmfp('c_ProcFood','a_Util_Cons','SouthAsia') = 2.81204915 ;
vmfp('c_ProcFood','a_Util_Cons','NAmerica') = 55.26041794 ;
vmfp('c_ProcFood','a_Util_Cons','LatinAmer') = 79.05483246 ;
vmfp('c_ProcFood','a_Util_Cons','EU_28') = 797.9203491 ;
vmfp('c_ProcFood','a_Util_Cons','MENA') = 37.01213455 ;
vmfp('c_ProcFood','a_Util_Cons','SSA') = 63.30392456 ;
vmfp('c_ProcFood','a_Util_Cons','RestofWorld') = 77.11166382 ;
vmfp('c_ProcFood','a_TransComm','Oceania') = 1742.478149 ;
vmfp('c_ProcFood','a_TransComm','EastAsia') = 14373.41797 ;
vmfp('c_ProcFood','a_TransComm','SEAsia') = 3389.097412 ;
vmfp('c_ProcFood','a_TransComm','SouthAsia') = 1278.574585 ;
vmfp('c_ProcFood','a_TransComm','NAmerica') = 5266.913086 ;
vmfp('c_ProcFood','a_TransComm','LatinAmer') = 2978.889893 ;
vmfp('c_ProcFood','a_TransComm','EU_28') = 31990.40625 ;
vmfp('c_ProcFood','a_TransComm','MENA') = 7043.688965 ;
vmfp('c_ProcFood','a_TransComm','SSA') = 2635.373047 ;
vmfp('c_ProcFood','a_TransComm','RestofWorld') = 2574.240723 ;
vmfp('c_ProcFood','a_OthService','Oceania') = 831.0930786 ;
vmfp('c_ProcFood','a_OthService','EastAsia') = 2391.27124 ;
vmfp('c_ProcFood','a_OthService','SEAsia') = 774.5455933 ;
vmfp('c_ProcFood','a_OthService','SouthAsia') = 67.40852356 ;
vmfp('c_ProcFood','a_OthService','NAmerica') = 12248.32324 ;
vmfp('c_ProcFood','a_OthService','LatinAmer') = 939.7966309 ;
vmfp('c_ProcFood','a_OthService','EU_28') = 14875.2627 ;
vmfp('c_ProcFood','a_OthService','MENA') = 3054.703125 ;
vmfp('c_ProcFood','a_OthService','SSA') = 1712.669678 ;
vmfp('c_ProcFood','a_OthService','RestofWorld') = 2849.303711 ;
vmfp('c_TextWapp','a_agricultur','Oceania') = 65.98021698 ;
vmfp('c_TextWapp','a_agricultur','EastAsia') = 52.86979675 ;
vmfp('c_TextWapp','a_agricultur','SEAsia') = 60.33182907 ;
vmfp('c_TextWapp','a_agricultur','SouthAsia') = 235.6318817 ;
vmfp('c_TextWapp','a_agricultur','NAmerica') = 806.9970093 ;
vmfp('c_TextWapp','a_agricultur','LatinAmer') = 367.3488159 ;
vmfp('c_TextWapp','a_agricultur','EU_28') = 538.8872681 ;
vmfp('c_TextWapp','a_agricultur','MENA') = 734.9037476 ;
vmfp('c_TextWapp','a_agricultur','SSA') = 726.4871826 ;
vmfp('c_TextWapp','a_agricultur','RestofWorld') = 203.8461609 ;
vmfp('c_TextWapp','a_Extraction','Oceania') = 115.7049637 ;
vmfp('c_TextWapp','a_Extraction','EastAsia') = 346.7378845 ;
vmfp('c_TextWapp','a_Extraction','SEAsia') = 50.70137405 ;
vmfp('c_TextWapp','a_Extraction','SouthAsia') = 58.11943054 ;
vmfp('c_TextWapp','a_Extraction','NAmerica') = 426.6512451 ;
vmfp('c_TextWapp','a_Extraction','LatinAmer') = 129.5910645 ;
vmfp('c_TextWapp','a_Extraction','EU_28') = 503.5301819 ;
vmfp('c_TextWapp','a_Extraction','MENA') = 219.494812 ;
vmfp('c_TextWapp','a_Extraction','SSA') = 292.0785828 ;
vmfp('c_TextWapp','a_Extraction','RestofWorld') = 305.5086365 ;
vmfp('c_TextWapp','a_ProcFood','Oceania') = 122.7268753 ;
vmfp('c_TextWapp','a_ProcFood','EastAsia') = 281.2229004 ;
vmfp('c_TextWapp','a_ProcFood','SEAsia') = 135.9411011 ;
vmfp('c_TextWapp','a_ProcFood','SouthAsia') = 611.9154053 ;
vmfp('c_TextWapp','a_ProcFood','NAmerica') = 133.0549774 ;
vmfp('c_TextWapp','a_ProcFood','LatinAmer') = 207.0367737 ;
vmfp('c_TextWapp','a_ProcFood','EU_28') = 903.0578003 ;
vmfp('c_TextWapp','a_ProcFood','MENA') = 369.0718994 ;
vmfp('c_TextWapp','a_ProcFood','SSA') = 210.1535187 ;
vmfp('c_TextWapp','a_ProcFood','RestofWorld') = 242.1328278 ;
vmfp('c_TextWapp','a_TextWapp','Oceania') = 1016.994629 ;
vmfp('c_TextWapp','a_TextWapp','EastAsia') = 47632.21094 ;
vmfp('c_TextWapp','a_TextWapp','SEAsia') = 25927.22266 ;
vmfp('c_TextWapp','a_TextWapp','SouthAsia') = 11620.67188 ;
vmfp('c_TextWapp','a_TextWapp','NAmerica') = 29325.11523 ;
vmfp('c_TextWapp','a_TextWapp','LatinAmer') = 14875.90723 ;
vmfp('c_TextWapp','a_TextWapp','EU_28') = 58542.53516 ;
vmfp('c_TextWapp','a_TextWapp','MENA') = 16318.03613 ;
vmfp('c_TextWapp','a_TextWapp','SSA') = 3124.038574 ;
vmfp('c_TextWapp','a_TextWapp','RestofWorld') = 7184.775879 ;
vmfp('c_TextWapp','a_LightMnfc','Oceania') = 430.640564 ;
vmfp('c_TextWapp','a_LightMnfc','EastAsia') = 4688.494141 ;
vmfp('c_TextWapp','a_LightMnfc','SEAsia') = 3533.339355 ;
vmfp('c_TextWapp','a_LightMnfc','SouthAsia') = 140.9259491 ;
vmfp('c_TextWapp','a_LightMnfc','NAmerica') = 13000.62695 ;
vmfp('c_TextWapp','a_LightMnfc','LatinAmer') = 1593.942993 ;
vmfp('c_TextWapp','a_LightMnfc','EU_28') = 11626.72266 ;
vmfp('c_TextWapp','a_LightMnfc','MENA') = 879.4038086 ;
vmfp('c_TextWapp','a_LightMnfc','SSA') = 886.8568726 ;
vmfp('c_TextWapp','a_LightMnfc','RestofWorld') = 1477.804565 ;
vmfp('c_TextWapp','a_HeavyMnfc','Oceania') = 285.339325 ;
vmfp('c_TextWapp','a_HeavyMnfc','EastAsia') = 3013.72998 ;
vmfp('c_TextWapp','a_HeavyMnfc','SEAsia') = 860.8842773 ;
vmfp('c_TextWapp','a_HeavyMnfc','SouthAsia') = 170.7772675 ;
vmfp('c_TextWapp','a_HeavyMnfc','NAmerica') = 3319.988525 ;
vmfp('c_TextWapp','a_HeavyMnfc','LatinAmer') = 916.8098145 ;
vmfp('c_TextWapp','a_HeavyMnfc','EU_28') = 6736.915527 ;
vmfp('c_TextWapp','a_HeavyMnfc','MENA') = 2735.699219 ;
vmfp('c_TextWapp','a_HeavyMnfc','SSA') = 325.4897461 ;
vmfp('c_TextWapp','a_HeavyMnfc','RestofWorld') = 1552.505493 ;
vmfp('c_TextWapp','a_Util_Cons','Oceania') = 283.1362915 ;
vmfp('c_TextWapp','a_Util_Cons','EastAsia') = 1055.837524 ;
vmfp('c_TextWapp','a_Util_Cons','SEAsia') = 191.5922546 ;
vmfp('c_TextWapp','a_Util_Cons','SouthAsia') = 24.12040138 ;
vmfp('c_TextWapp','a_Util_Cons','NAmerica') = 1567.494141 ;
vmfp('c_TextWapp','a_Util_Cons','LatinAmer') = 159.4113312 ;
vmfp('c_TextWapp','a_Util_Cons','EU_28') = 2761.334473 ;
vmfp('c_TextWapp','a_Util_Cons','MENA') = 749.1322021 ;
vmfp('c_TextWapp','a_Util_Cons','SSA') = 188.3923645 ;
vmfp('c_TextWapp','a_Util_Cons','RestofWorld') = 626.1070557 ;
vmfp('c_TextWapp','a_TransComm','Oceania') = 858.6398315 ;
vmfp('c_TextWapp','a_TransComm','EastAsia') = 1886.143799 ;
vmfp('c_TextWapp','a_TransComm','SEAsia') = 813.1248169 ;
vmfp('c_TextWapp','a_TransComm','SouthAsia') = 124.4251709 ;
vmfp('c_TextWapp','a_TransComm','NAmerica') = 4167.498047 ;
vmfp('c_TextWapp','a_TransComm','LatinAmer') = 790.6846924 ;
vmfp('c_TextWapp','a_TransComm','EU_28') = 7925.134766 ;
vmfp('c_TextWapp','a_TransComm','MENA') = 3994.284668 ;
vmfp('c_TextWapp','a_TransComm','SSA') = 802.0488281 ;
vmfp('c_TextWapp','a_TransComm','RestofWorld') = 1337.013428 ;
vmfp('c_TextWapp','a_OthService','Oceania') = 1466.895508 ;
vmfp('c_TextWapp','a_OthService','EastAsia') = 2783.579834 ;
vmfp('c_TextWapp','a_OthService','SEAsia') = 627.4266968 ;
vmfp('c_TextWapp','a_OthService','SouthAsia') = 418.2997437 ;
vmfp('c_TextWapp','a_OthService','NAmerica') = 4189.97998 ;
vmfp('c_TextWapp','a_OthService','LatinAmer') = 740.7105103 ;
vmfp('c_TextWapp','a_OthService','EU_28') = 11266.26562 ;
vmfp('c_TextWapp','a_OthService','MENA') = 3501.617432 ;
vmfp('c_TextWapp','a_OthService','SSA') = 1480.223389 ;
vmfp('c_TextWapp','a_OthService','RestofWorld') = 2190.790527 ;
vmfp('c_LightMnfc','a_agricultur','Oceania') = 198.7162933 ;
vmfp('c_LightMnfc','a_agricultur','EastAsia') = 522.3776855 ;
vmfp('c_LightMnfc','a_agricultur','SEAsia') = 317.5440369 ;
vmfp('c_LightMnfc','a_agricultur','SouthAsia') = 165.2107086 ;
vmfp('c_LightMnfc','a_agricultur','NAmerica') = 1159.945679 ;
vmfp('c_LightMnfc','a_agricultur','LatinAmer') = 1136.056152 ;
vmfp('c_LightMnfc','a_agricultur','EU_28') = 4336.963379 ;
vmfp('c_LightMnfc','a_agricultur','MENA') = 1367.567261 ;
vmfp('c_LightMnfc','a_agricultur','SSA') = 2374.000977 ;
vmfp('c_LightMnfc','a_agricultur','RestofWorld') = 1707.864624 ;
vmfp('c_LightMnfc','a_Extraction','Oceania') = 868.9053955 ;
vmfp('c_LightMnfc','a_Extraction','EastAsia') = 2900.815674 ;
vmfp('c_LightMnfc','a_Extraction','SEAsia') = 1163.38623 ;
vmfp('c_LightMnfc','a_Extraction','SouthAsia') = 623.1765137 ;
vmfp('c_LightMnfc','a_Extraction','NAmerica') = 2565.970947 ;
vmfp('c_LightMnfc','a_Extraction','LatinAmer') = 1508.644409 ;
vmfp('c_LightMnfc','a_Extraction','EU_28') = 3209.179932 ;
vmfp('c_LightMnfc','a_Extraction','MENA') = 2512.130371 ;
vmfp('c_LightMnfc','a_Extraction','SSA') = 3357.464111 ;
vmfp('c_LightMnfc','a_Extraction','RestofWorld') = 3990.243164 ;
vmfp('c_LightMnfc','a_ProcFood','Oceania') = 652.3693237 ;
vmfp('c_LightMnfc','a_ProcFood','EastAsia') = 2689.939941 ;
vmfp('c_LightMnfc','a_ProcFood','SEAsia') = 3112.55127 ;
vmfp('c_LightMnfc','a_ProcFood','SouthAsia') = 905.1005859 ;
vmfp('c_LightMnfc','a_ProcFood','NAmerica') = 3102.640625 ;
vmfp('c_LightMnfc','a_ProcFood','LatinAmer') = 2002.199463 ;
vmfp('c_LightMnfc','a_ProcFood','EU_28') = 14508.79492 ;
vmfp('c_LightMnfc','a_ProcFood','MENA') = 2777.244873 ;
vmfp('c_LightMnfc','a_ProcFood','SSA') = 1522.623779 ;
vmfp('c_LightMnfc','a_ProcFood','RestofWorld') = 2540.468262 ;
vmfp('c_LightMnfc','a_TextWapp','Oceania') = 153.697525 ;
vmfp('c_LightMnfc','a_TextWapp','EastAsia') = 2977.060547 ;
vmfp('c_LightMnfc','a_TextWapp','SEAsia') = 843.9711914 ;
vmfp('c_LightMnfc','a_TextWapp','SouthAsia') = 837.5515747 ;
vmfp('c_LightMnfc','a_TextWapp','NAmerica') = 2005.632202 ;
vmfp('c_LightMnfc','a_TextWapp','LatinAmer') = 967.9093018 ;
vmfp('c_LightMnfc','a_TextWapp','EU_28') = 5126.060059 ;
vmfp('c_LightMnfc','a_TextWapp','MENA') = 1163.231812 ;
vmfp('c_LightMnfc','a_TextWapp','SSA') = 409.3229065 ;
vmfp('c_LightMnfc','a_TextWapp','RestofWorld') = 894.1315918 ;
vmfp('c_LightMnfc','a_LightMnfc','Oceania') = 6951.849609 ;
vmfp('c_LightMnfc','a_LightMnfc','EastAsia') = 121505.1094 ;
vmfp('c_LightMnfc','a_LightMnfc','SEAsia') = 42388.76562 ;
vmfp('c_LightMnfc','a_LightMnfc','SouthAsia') = 11400.29004 ;
vmfp('c_LightMnfc','a_LightMnfc','NAmerica') = 170034.7031 ;
vmfp('c_LightMnfc','a_LightMnfc','LatinAmer') = 37914.29688 ;
vmfp('c_LightMnfc','a_LightMnfc','EU_28') = 357893.6562 ;
vmfp('c_LightMnfc','a_LightMnfc','MENA') = 39660.19141 ;
vmfp('c_LightMnfc','a_LightMnfc','SSA') = 12127.47852 ;
vmfp('c_LightMnfc','a_LightMnfc','RestofWorld') = 28521.98242 ;
vmfp('c_LightMnfc','a_HeavyMnfc','Oceania') = 1439.333008 ;
vmfp('c_LightMnfc','a_HeavyMnfc','EastAsia') = 28776.33008 ;
vmfp('c_LightMnfc','a_HeavyMnfc','SEAsia') = 9029.982422 ;
vmfp('c_LightMnfc','a_HeavyMnfc','SouthAsia') = 3531.760986 ;
vmfp('c_LightMnfc','a_HeavyMnfc','NAmerica') = 21450.28125 ;
vmfp('c_LightMnfc','a_HeavyMnfc','LatinAmer') = 6501.553711 ;
vmfp('c_LightMnfc','a_HeavyMnfc','EU_28') = 80664.49219 ;
vmfp('c_LightMnfc','a_HeavyMnfc','MENA') = 8579.621094 ;
vmfp('c_LightMnfc','a_HeavyMnfc','SSA') = 2854.983887 ;
vmfp('c_LightMnfc','a_HeavyMnfc','RestofWorld') = 14102.11133 ;
vmfp('c_LightMnfc','a_Util_Cons','Oceania') = 5866.34375 ;
vmfp('c_LightMnfc','a_Util_Cons','EastAsia') = 22453.61914 ;
vmfp('c_LightMnfc','a_Util_Cons','SEAsia') = 10074.06934 ;
vmfp('c_LightMnfc','a_Util_Cons','SouthAsia') = 4463.61084 ;
vmfp('c_LightMnfc','a_Util_Cons','NAmerica') = 35948.19141 ;
vmfp('c_LightMnfc','a_Util_Cons','LatinAmer') = 8381.493164 ;
vmfp('c_LightMnfc','a_Util_Cons','EU_28') = 64413.78516 ;
vmfp('c_LightMnfc','a_Util_Cons','MENA') = 33327.95312 ;
vmfp('c_LightMnfc','a_Util_Cons','SSA') = 4686.833008 ;
vmfp('c_LightMnfc','a_Util_Cons','RestofWorld') = 16891.15625 ;
vmfp('c_LightMnfc','a_TransComm','Oceania') = 6465.046387 ;
vmfp('c_LightMnfc','a_TransComm','EastAsia') = 20047.41211 ;
vmfp('c_LightMnfc','a_TransComm','SEAsia') = 6781.985352 ;
vmfp('c_LightMnfc','a_TransComm','SouthAsia') = 7715.334473 ;
vmfp('c_LightMnfc','a_TransComm','NAmerica') = 32354.66211 ;
vmfp('c_LightMnfc','a_TransComm','LatinAmer') = 9214.943359 ;
vmfp('c_LightMnfc','a_TransComm','EU_28') = 68587.82812 ;
vmfp('c_LightMnfc','a_TransComm','MENA') = 17204.75391 ;
vmfp('c_LightMnfc','a_TransComm','SSA') = 8447.582031 ;
vmfp('c_LightMnfc','a_TransComm','RestofWorld') = 21089.35352 ;
vmfp('c_LightMnfc','a_OthService','Oceania') = 6501.289062 ;
vmfp('c_LightMnfc','a_OthService','EastAsia') = 25818.20312 ;
vmfp('c_LightMnfc','a_OthService','SEAsia') = 6571.991211 ;
vmfp('c_LightMnfc','a_OthService','SouthAsia') = 2180.152588 ;
vmfp('c_LightMnfc','a_OthService','NAmerica') = 37841.25391 ;
vmfp('c_LightMnfc','a_OthService','LatinAmer') = 8235.217773 ;
vmfp('c_LightMnfc','a_OthService','EU_28') = 95115.42188 ;
vmfp('c_LightMnfc','a_OthService','MENA') = 12180.58789 ;
vmfp('c_LightMnfc','a_OthService','SSA') = 9507.818359 ;
vmfp('c_LightMnfc','a_OthService','RestofWorld') = 13153.03027 ;
vmfp('c_HeavyMnfc','a_agricultur','Oceania') = 3458.02124 ;
vmfp('c_HeavyMnfc','a_agricultur','EastAsia') = 12773.15039 ;
vmfp('c_HeavyMnfc','a_agricultur','SEAsia') = 9239.560547 ;
vmfp('c_HeavyMnfc','a_agricultur','SouthAsia') = 10856.89746 ;
vmfp('c_HeavyMnfc','a_agricultur','NAmerica') = 16368.74414 ;
vmfp('c_HeavyMnfc','a_agricultur','LatinAmer') = 12644.17383 ;
vmfp('c_HeavyMnfc','a_agricultur','EU_28') = 28995.2207 ;
vmfp('c_HeavyMnfc','a_agricultur','MENA') = 11476.01758 ;
vmfp('c_HeavyMnfc','a_agricultur','SSA') = 9846.773438 ;
vmfp('c_HeavyMnfc','a_agricultur','RestofWorld') = 9939.676758 ;
vmfp('c_HeavyMnfc','a_Extraction','Oceania') = 6588.928711 ;
vmfp('c_HeavyMnfc','a_Extraction','EastAsia') = 18111.73438 ;
vmfp('c_HeavyMnfc','a_Extraction','SEAsia') = 8961.31543 ;
vmfp('c_HeavyMnfc','a_Extraction','SouthAsia') = 3269.152588 ;
vmfp('c_HeavyMnfc','a_Extraction','NAmerica') = 20879.9082 ;
vmfp('c_HeavyMnfc','a_Extraction','LatinAmer') = 11173.30762 ;
vmfp('c_HeavyMnfc','a_Extraction','EU_28') = 12965.10059 ;
vmfp('c_HeavyMnfc','a_Extraction','MENA') = 60975.35156 ;
vmfp('c_HeavyMnfc','a_Extraction','SSA') = 11147.16504 ;
vmfp('c_HeavyMnfc','a_Extraction','RestofWorld') = 14003.8584 ;
vmfp('c_HeavyMnfc','a_ProcFood','Oceania') = 1887.179688 ;
vmfp('c_HeavyMnfc','a_ProcFood','EastAsia') = 8124.709473 ;
vmfp('c_HeavyMnfc','a_ProcFood','SEAsia') = 5364.878418 ;
vmfp('c_HeavyMnfc','a_ProcFood','SouthAsia') = 3828.72998 ;
vmfp('c_HeavyMnfc','a_ProcFood','NAmerica') = 11853.20996 ;
vmfp('c_HeavyMnfc','a_ProcFood','LatinAmer') = 6745.339355 ;
vmfp('c_HeavyMnfc','a_ProcFood','EU_28') = 36957.01172 ;
vmfp('c_HeavyMnfc','a_ProcFood','MENA') = 5656.736816 ;
vmfp('c_HeavyMnfc','a_ProcFood','SSA') = 2570.988281 ;
vmfp('c_HeavyMnfc','a_ProcFood','RestofWorld') = 6316.654785 ;
vmfp('c_HeavyMnfc','a_TextWapp','Oceania') = 313.4644165 ;
vmfp('c_HeavyMnfc','a_TextWapp','EastAsia') = 17413.09766 ;
vmfp('c_HeavyMnfc','a_TextWapp','SEAsia') = 7467.585938 ;
vmfp('c_HeavyMnfc','a_TextWapp','SouthAsia') = 9068.036133 ;
vmfp('c_HeavyMnfc','a_TextWapp','NAmerica') = 10005.46582 ;
vmfp('c_HeavyMnfc','a_TextWapp','LatinAmer') = 4337.744141 ;
vmfp('c_HeavyMnfc','a_TextWapp','EU_28') = 21341.09766 ;
vmfp('c_HeavyMnfc','a_TextWapp','MENA') = 9285.973633 ;
vmfp('c_HeavyMnfc','a_TextWapp','SSA') = 1154.377441 ;
vmfp('c_HeavyMnfc','a_TextWapp','RestofWorld') = 1834.410889 ;
vmfp('c_HeavyMnfc','a_LightMnfc','Oceania') = 11023.2334 ;
vmfp('c_HeavyMnfc','a_LightMnfc','EastAsia') = 125989.8672 ;
vmfp('c_HeavyMnfc','a_LightMnfc','SEAsia') = 59446.875 ;
vmfp('c_HeavyMnfc','a_LightMnfc','SouthAsia') = 34050.53906 ;
vmfp('c_HeavyMnfc','a_LightMnfc','NAmerica') = 123172.8516 ;
vmfp('c_HeavyMnfc','a_LightMnfc','LatinAmer') = 29867.42578 ;
vmfp('c_HeavyMnfc','a_LightMnfc','EU_28') = 333952.8438 ;
vmfp('c_HeavyMnfc','a_LightMnfc','MENA') = 42837.53906 ;
vmfp('c_HeavyMnfc','a_LightMnfc','SSA') = 8174.021973 ;
vmfp('c_HeavyMnfc','a_LightMnfc','RestofWorld') = 27020.17969 ;
vmfp('c_HeavyMnfc','a_HeavyMnfc','Oceania') = 24337.36133 ;
vmfp('c_HeavyMnfc','a_HeavyMnfc','EastAsia') = 931272 ;
vmfp('c_HeavyMnfc','a_HeavyMnfc','SEAsia') = 278808.25 ;
vmfp('c_HeavyMnfc','a_HeavyMnfc','SouthAsia') = 100288.1406 ;
vmfp('c_HeavyMnfc','a_HeavyMnfc','NAmerica') = 477851.6562 ;
vmfp('c_HeavyMnfc','a_HeavyMnfc','LatinAmer') = 123090.7266 ;
vmfp('c_HeavyMnfc','a_HeavyMnfc','EU_28') = 1024147.438 ;
vmfp('c_HeavyMnfc','a_HeavyMnfc','MENA') = 116774.2266 ;
vmfp('c_HeavyMnfc','a_HeavyMnfc','SSA') = 27191.51758 ;
vmfp('c_HeavyMnfc','a_HeavyMnfc','RestofWorld') = 163428.1094 ;
vmfp('c_HeavyMnfc','a_Util_Cons','Oceania') = 12378.01465 ;
vmfp('c_HeavyMnfc','a_Util_Cons','EastAsia') = 76694.60156 ;
vmfp('c_HeavyMnfc','a_Util_Cons','SEAsia') = 44230.70703 ;
vmfp('c_HeavyMnfc','a_Util_Cons','SouthAsia') = 23556.98242 ;
vmfp('c_HeavyMnfc','a_Util_Cons','NAmerica') = 91392.89844 ;
vmfp('c_HeavyMnfc','a_Util_Cons','LatinAmer') = 33411.43359 ;
vmfp('c_HeavyMnfc','a_Util_Cons','EU_28') = 203369.4062 ;
vmfp('c_HeavyMnfc','a_Util_Cons','MENA') = 69181.72656 ;
vmfp('c_HeavyMnfc','a_Util_Cons','SSA') = 17092.37891 ;
vmfp('c_HeavyMnfc','a_Util_Cons','RestofWorld') = 39028.03125 ;
vmfp('c_HeavyMnfc','a_TransComm','Oceania') = 20966.85742 ;
vmfp('c_HeavyMnfc','a_TransComm','EastAsia') = 76341.21094 ;
vmfp('c_HeavyMnfc','a_TransComm','SEAsia') = 56426.60547 ;
vmfp('c_HeavyMnfc','a_TransComm','SouthAsia') = 20603.79297 ;
vmfp('c_HeavyMnfc','a_TransComm','NAmerica') = 106893.3516 ;
vmfp('c_HeavyMnfc','a_TransComm','LatinAmer') = 39692.51172 ;
vmfp('c_HeavyMnfc','a_TransComm','EU_28') = 278054.4062 ;
vmfp('c_HeavyMnfc','a_TransComm','MENA') = 44161.03125 ;
vmfp('c_HeavyMnfc','a_TransComm','SSA') = 23761.71875 ;
vmfp('c_HeavyMnfc','a_TransComm','RestofWorld') = 49260.70703 ;
vmfp('c_HeavyMnfc','a_OthService','Oceania') = 20328.13477 ;
vmfp('c_HeavyMnfc','a_OthService','EastAsia') = 100241.0156 ;
vmfp('c_HeavyMnfc','a_OthService','SEAsia') = 22179.58789 ;
vmfp('c_HeavyMnfc','a_OthService','SouthAsia') = 6338.827637 ;
vmfp('c_HeavyMnfc','a_OthService','NAmerica') = 138305.9219 ;
vmfp('c_HeavyMnfc','a_OthService','LatinAmer') = 18579.23828 ;
vmfp('c_HeavyMnfc','a_OthService','EU_28') = 239456.625 ;
vmfp('c_HeavyMnfc','a_OthService','MENA') = 26560.98633 ;
vmfp('c_HeavyMnfc','a_OthService','SSA') = 12536.58301 ;
vmfp('c_HeavyMnfc','a_OthService','RestofWorld') = 32343.85156 ;
vmfp('c_Util_Cons','a_agricultur','Oceania') = 10.9321785 ;
vmfp('c_Util_Cons','a_agricultur','EastAsia') = 17.33044052 ;
vmfp('c_Util_Cons','a_agricultur','SEAsia') = 18.29068756 ;
vmfp('c_Util_Cons','a_agricultur','SouthAsia') = 96.7307663 ;
vmfp('c_Util_Cons','a_agricultur','NAmerica') = 89.87813568 ;
vmfp('c_Util_Cons','a_agricultur','LatinAmer') = 48.91152191 ;
vmfp('c_Util_Cons','a_agricultur','EU_28') = 1381.819092 ;
vmfp('c_Util_Cons','a_agricultur','MENA') = 233.6154175 ;
vmfp('c_Util_Cons','a_agricultur','SSA') = 47.83044815 ;
vmfp('c_Util_Cons','a_agricultur','RestofWorld') = 458.3400879 ;
vmfp('c_Util_Cons','a_Extraction','Oceania') = 72.61758423 ;
vmfp('c_Util_Cons','a_Extraction','EastAsia') = 48.11830521 ;
vmfp('c_Util_Cons','a_Extraction','SEAsia') = 274.5952759 ;
vmfp('c_Util_Cons','a_Extraction','SouthAsia') = 64.61261749 ;
vmfp('c_Util_Cons','a_Extraction','NAmerica') = 447.5466919 ;
vmfp('c_Util_Cons','a_Extraction','LatinAmer') = 345.3678894 ;
vmfp('c_Util_Cons','a_Extraction','EU_28') = 670.0858765 ;
vmfp('c_Util_Cons','a_Extraction','MENA') = 539.946167 ;
vmfp('c_Util_Cons','a_Extraction','SSA') = 292.7575378 ;
vmfp('c_Util_Cons','a_Extraction','RestofWorld') = 1786.815918 ;
vmfp('c_Util_Cons','a_ProcFood','Oceania') = 25.84300995 ;
vmfp('c_Util_Cons','a_ProcFood','EastAsia') = 23.01737404 ;
vmfp('c_Util_Cons','a_ProcFood','SEAsia') = 25.88068962 ;
vmfp('c_Util_Cons','a_ProcFood','SouthAsia') = 18.5581398 ;
vmfp('c_Util_Cons','a_ProcFood','NAmerica') = 61.1064415 ;
vmfp('c_Util_Cons','a_ProcFood','LatinAmer') = 93.80300903 ;
vmfp('c_Util_Cons','a_ProcFood','EU_28') = 1923.57666 ;
vmfp('c_Util_Cons','a_ProcFood','MENA') = 470.2545471 ;
vmfp('c_Util_Cons','a_ProcFood','SSA') = 37.16766739 ;
vmfp('c_Util_Cons','a_ProcFood','RestofWorld') = 175.7453308 ;
vmfp('c_Util_Cons','a_TextWapp','Oceania') = 1.076451421 ;
vmfp('c_Util_Cons','a_TextWapp','EastAsia') = 92.7914505 ;
vmfp('c_Util_Cons','a_TextWapp','SEAsia') = 111.0332718 ;
vmfp('c_Util_Cons','a_TextWapp','SouthAsia') = 233.0809631 ;
vmfp('c_Util_Cons','a_TextWapp','NAmerica') = 19.12753296 ;
vmfp('c_Util_Cons','a_TextWapp','LatinAmer') = 22.75564384 ;
vmfp('c_Util_Cons','a_TextWapp','EU_28') = 365.3594971 ;
vmfp('c_Util_Cons','a_TextWapp','MENA') = 147.2321777 ;
vmfp('c_Util_Cons','a_TextWapp','SSA') = 38.14147186 ;
vmfp('c_Util_Cons','a_TextWapp','RestofWorld') = 33.54384613 ;
vmfp('c_Util_Cons','a_LightMnfc','Oceania') = 10.01785088 ;
vmfp('c_Util_Cons','a_LightMnfc','EastAsia') = 107.4809723 ;
vmfp('c_Util_Cons','a_LightMnfc','SEAsia') = 84.53599548 ;
vmfp('c_Util_Cons','a_LightMnfc','SouthAsia') = 35.94439697 ;
vmfp('c_Util_Cons','a_LightMnfc','NAmerica') = 331.289856 ;
vmfp('c_Util_Cons','a_LightMnfc','LatinAmer') = 98.87835693 ;
vmfp('c_Util_Cons','a_LightMnfc','EU_28') = 3616.736816 ;
vmfp('c_Util_Cons','a_LightMnfc','MENA') = 1013.910339 ;
vmfp('c_Util_Cons','a_LightMnfc','SSA') = 75.3465271 ;
vmfp('c_Util_Cons','a_LightMnfc','RestofWorld') = 400.9945679 ;
vmfp('c_Util_Cons','a_HeavyMnfc','Oceania') = 75.31261444 ;
vmfp('c_Util_Cons','a_HeavyMnfc','EastAsia') = 396.1621094 ;
vmfp('c_Util_Cons','a_HeavyMnfc','SEAsia') = 837.6237793 ;
vmfp('c_Util_Cons','a_HeavyMnfc','SouthAsia') = 147.099884 ;
vmfp('c_Util_Cons','a_HeavyMnfc','NAmerica') = 789.4065552 ;
vmfp('c_Util_Cons','a_HeavyMnfc','LatinAmer') = 532.3734131 ;
vmfp('c_Util_Cons','a_HeavyMnfc','EU_28') = 11216.29004 ;
vmfp('c_Util_Cons','a_HeavyMnfc','MENA') = 2149.295654 ;
vmfp('c_Util_Cons','a_HeavyMnfc','SSA') = 496.4232483 ;
vmfp('c_Util_Cons','a_HeavyMnfc','RestofWorld') = 2782.406494 ;
vmfp('c_Util_Cons','a_Util_Cons','Oceania') = 142.0762329 ;
vmfp('c_Util_Cons','a_Util_Cons','EastAsia') = 848.0397949 ;
vmfp('c_Util_Cons','a_Util_Cons','SEAsia') = 2184.700928 ;
vmfp('c_Util_Cons','a_Util_Cons','SouthAsia') = 380.8345642 ;
vmfp('c_Util_Cons','a_Util_Cons','NAmerica') = 267.484314 ;
vmfp('c_Util_Cons','a_Util_Cons','LatinAmer') = 3309.762451 ;
vmfp('c_Util_Cons','a_Util_Cons','EU_28') = 7011.887207 ;
vmfp('c_Util_Cons','a_Util_Cons','MENA') = 8935.62793 ;
vmfp('c_Util_Cons','a_Util_Cons','SSA') = 918.2032471 ;
vmfp('c_Util_Cons','a_Util_Cons','RestofWorld') = 2127.494141 ;
vmfp('c_Util_Cons','a_TransComm','Oceania') = 73.87876129 ;
vmfp('c_Util_Cons','a_TransComm','EastAsia') = 584.6291504 ;
vmfp('c_Util_Cons','a_TransComm','SEAsia') = 102.7212982 ;
vmfp('c_Util_Cons','a_TransComm','SouthAsia') = 54.51414108 ;
vmfp('c_Util_Cons','a_TransComm','NAmerica') = 819.4098511 ;
vmfp('c_Util_Cons','a_TransComm','LatinAmer') = 126.0393448 ;
vmfp('c_Util_Cons','a_TransComm','EU_28') = 4433.058594 ;
vmfp('c_Util_Cons','a_TransComm','MENA') = 1036.642212 ;
vmfp('c_Util_Cons','a_TransComm','SSA') = 253.3065491 ;
vmfp('c_Util_Cons','a_TransComm','RestofWorld') = 1769.334229 ;
vmfp('c_Util_Cons','a_OthService','Oceania') = 363.7172852 ;
vmfp('c_Util_Cons','a_OthService','EastAsia') = 3796.96167 ;
vmfp('c_Util_Cons','a_OthService','SEAsia') = 1096.702515 ;
vmfp('c_Util_Cons','a_OthService','SouthAsia') = 116.8390503 ;
vmfp('c_Util_Cons','a_OthService','NAmerica') = 2066.34375 ;
vmfp('c_Util_Cons','a_OthService','LatinAmer') = 336.3238525 ;
vmfp('c_Util_Cons','a_OthService','EU_28') = 13069.16211 ;
vmfp('c_Util_Cons','a_OthService','MENA') = 1348.529663 ;
vmfp('c_Util_Cons','a_OthService','SSA') = 539.0219116 ;
vmfp('c_Util_Cons','a_OthService','RestofWorld') = 2968.461914 ;
vmfp('c_TransComm','a_agricultur','Oceania') = 109.9680328 ;
vmfp('c_TransComm','a_agricultur','EastAsia') = 1242.156372 ;
vmfp('c_TransComm','a_agricultur','SEAsia') = 243.3941956 ;
vmfp('c_TransComm','a_agricultur','SouthAsia') = 542.9604492 ;
vmfp('c_TransComm','a_agricultur','NAmerica') = 341.661438 ;
vmfp('c_TransComm','a_agricultur','LatinAmer') = 609.3401489 ;
vmfp('c_TransComm','a_agricultur','EU_28') = 1638.410645 ;
vmfp('c_TransComm','a_agricultur','MENA') = 467.5371094 ;
vmfp('c_TransComm','a_agricultur','SSA') = 875.678894 ;
vmfp('c_TransComm','a_agricultur','RestofWorld') = 478.7336731 ;
vmfp('c_TransComm','a_Extraction','Oceania') = 1004.068176 ;
vmfp('c_TransComm','a_Extraction','EastAsia') = 2561.106689 ;
vmfp('c_TransComm','a_Extraction','SEAsia') = 613.7075806 ;
vmfp('c_TransComm','a_Extraction','SouthAsia') = 182.3853149 ;
vmfp('c_TransComm','a_Extraction','NAmerica') = 909.8754272 ;
vmfp('c_TransComm','a_Extraction','LatinAmer') = 648.0304565 ;
vmfp('c_TransComm','a_Extraction','EU_28') = 4047.753174 ;
vmfp('c_TransComm','a_Extraction','MENA') = 1511.112915 ;
vmfp('c_TransComm','a_Extraction','SSA') = 1732.644043 ;
vmfp('c_TransComm','a_Extraction','RestofWorld') = 2725.279785 ;
vmfp('c_TransComm','a_ProcFood','Oceania') = 143.3628998 ;
vmfp('c_TransComm','a_ProcFood','EastAsia') = 1947.626099 ;
vmfp('c_TransComm','a_ProcFood','SEAsia') = 498.9193726 ;
vmfp('c_TransComm','a_ProcFood','SouthAsia') = 444.6357117 ;
vmfp('c_TransComm','a_ProcFood','NAmerica') = 749.4780884 ;
vmfp('c_TransComm','a_ProcFood','LatinAmer') = 1179.17749 ;
vmfp('c_TransComm','a_ProcFood','EU_28') = 8150.260254 ;
vmfp('c_TransComm','a_ProcFood','MENA') = 491.2650757 ;
vmfp('c_TransComm','a_ProcFood','SSA') = 721.1970825 ;
vmfp('c_TransComm','a_ProcFood','RestofWorld') = 564.1265259 ;
vmfp('c_TransComm','a_TextWapp','Oceania') = 14.45169735 ;
vmfp('c_TransComm','a_TextWapp','EastAsia') = 2274.126221 ;
vmfp('c_TransComm','a_TextWapp','SEAsia') = 257.8248596 ;
vmfp('c_TransComm','a_TextWapp','SouthAsia') = 375.5444946 ;
vmfp('c_TransComm','a_TextWapp','NAmerica') = 292.4999695 ;
vmfp('c_TransComm','a_TextWapp','LatinAmer') = 213.7783508 ;
vmfp('c_TransComm','a_TextWapp','EU_28') = 3661.892334 ;
vmfp('c_TransComm','a_TextWapp','MENA') = 275.2165527 ;
vmfp('c_TransComm','a_TextWapp','SSA') = 278.0074158 ;
vmfp('c_TransComm','a_TextWapp','RestofWorld') = 150.3536072 ;
vmfp('c_TransComm','a_LightMnfc','Oceania') = 249.7813416 ;
vmfp('c_TransComm','a_LightMnfc','EastAsia') = 6826.62207 ;
vmfp('c_TransComm','a_LightMnfc','SEAsia') = 1117.403198 ;
vmfp('c_TransComm','a_LightMnfc','SouthAsia') = 634.0684204 ;
vmfp('c_TransComm','a_LightMnfc','NAmerica') = 2786.959473 ;
vmfp('c_TransComm','a_LightMnfc','LatinAmer') = 1047.250244 ;
vmfp('c_TransComm','a_LightMnfc','EU_28') = 25950.82422 ;
vmfp('c_TransComm','a_LightMnfc','MENA') = 778.5722656 ;
vmfp('c_TransComm','a_LightMnfc','SSA') = 971.3644409 ;
vmfp('c_TransComm','a_LightMnfc','RestofWorld') = 1207.922607 ;
vmfp('c_TransComm','a_HeavyMnfc','Oceania') = 311.0870667 ;
vmfp('c_TransComm','a_HeavyMnfc','EastAsia') = 20227.53516 ;
vmfp('c_TransComm','a_HeavyMnfc','SEAsia') = 2976.983643 ;
vmfp('c_TransComm','a_HeavyMnfc','SouthAsia') = 973.8325806 ;
vmfp('c_TransComm','a_HeavyMnfc','NAmerica') = 4822.946289 ;
vmfp('c_TransComm','a_HeavyMnfc','LatinAmer') = 2293.954834 ;
vmfp('c_TransComm','a_HeavyMnfc','EU_28') = 50810.80078 ;
vmfp('c_TransComm','a_HeavyMnfc','MENA') = 1766.868652 ;
vmfp('c_TransComm','a_HeavyMnfc','SSA') = 1822.917969 ;
vmfp('c_TransComm','a_HeavyMnfc','RestofWorld') = 2192.214844 ;
vmfp('c_TransComm','a_Util_Cons','Oceania') = 334.3233337 ;
vmfp('c_TransComm','a_Util_Cons','EastAsia') = 7690.886719 ;
vmfp('c_TransComm','a_Util_Cons','SEAsia') = 1022.815186 ;
vmfp('c_TransComm','a_Util_Cons','SouthAsia') = 779.1358032 ;
vmfp('c_TransComm','a_Util_Cons','NAmerica') = 3878.960938 ;
vmfp('c_TransComm','a_Util_Cons','LatinAmer') = 599.9361572 ;
vmfp('c_TransComm','a_Util_Cons','EU_28') = 9036.358398 ;
vmfp('c_TransComm','a_Util_Cons','MENA') = 3378.817871 ;
vmfp('c_TransComm','a_Util_Cons','SSA') = 575.0617676 ;
vmfp('c_TransComm','a_Util_Cons','RestofWorld') = 3947.446777 ;
vmfp('c_TransComm','a_TransComm','Oceania') = 3643.257324 ;
vmfp('c_TransComm','a_TransComm','EastAsia') = 66696.44531 ;
vmfp('c_TransComm','a_TransComm','SEAsia') = 23803.78125 ;
vmfp('c_TransComm','a_TransComm','SouthAsia') = 2723.082764 ;
vmfp('c_TransComm','a_TransComm','NAmerica') = 45481.99219 ;
vmfp('c_TransComm','a_TransComm','LatinAmer') = 13763.67871 ;
vmfp('c_TransComm','a_TransComm','EU_28') = 150068.1406 ;
vmfp('c_TransComm','a_TransComm','MENA') = 18315.40039 ;
vmfp('c_TransComm','a_TransComm','SSA') = 4219.631836 ;
vmfp('c_TransComm','a_TransComm','RestofWorld') = 25969.84766 ;
vmfp('c_TransComm','a_OthService','Oceania') = 4726.853027 ;
vmfp('c_TransComm','a_OthService','EastAsia') = 19232.16016 ;
vmfp('c_TransComm','a_OthService','SEAsia') = 9144.78418 ;
vmfp('c_TransComm','a_OthService','SouthAsia') = 1869.75708 ;
vmfp('c_TransComm','a_OthService','NAmerica') = 58912.71484 ;
vmfp('c_TransComm','a_OthService','LatinAmer') = 2796.953613 ;
vmfp('c_TransComm','a_OthService','EU_28') = 81630.85938 ;
vmfp('c_TransComm','a_OthService','MENA') = 7096.234375 ;
vmfp('c_TransComm','a_OthService','SSA') = 3276.389648 ;
vmfp('c_TransComm','a_OthService','RestofWorld') = 4241.24707 ;
vmfp('c_OthService','a_agricultur','Oceania') = 314.8374634 ;
vmfp('c_OthService','a_agricultur','EastAsia') = 1573.297729 ;
vmfp('c_OthService','a_agricultur','SEAsia') = 187.9542694 ;
vmfp('c_OthService','a_agricultur','SouthAsia') = 440.0583496 ;
vmfp('c_OthService','a_agricultur','NAmerica') = 1798.809204 ;
vmfp('c_OthService','a_agricultur','LatinAmer') = 595.3381348 ;
vmfp('c_OthService','a_agricultur','EU_28') = 5515.470215 ;
vmfp('c_OthService','a_agricultur','MENA') = 1293.119141 ;
vmfp('c_OthService','a_agricultur','SSA') = 434.8308105 ;
vmfp('c_OthService','a_agricultur','RestofWorld') = 607.2830811 ;
vmfp('c_OthService','a_Extraction','Oceania') = 983.4888306 ;
vmfp('c_OthService','a_Extraction','EastAsia') = 1668.594727 ;
vmfp('c_OthService','a_Extraction','SEAsia') = 641.1037598 ;
vmfp('c_OthService','a_Extraction','SouthAsia') = 532.6539917 ;
vmfp('c_OthService','a_Extraction','NAmerica') = 3529.929688 ;
vmfp('c_OthService','a_Extraction','LatinAmer') = 4054.148438 ;
vmfp('c_OthService','a_Extraction','EU_28') = 2942.002441 ;
vmfp('c_OthService','a_Extraction','MENA') = 8428.529297 ;
vmfp('c_OthService','a_Extraction','SSA') = 4328.698242 ;
vmfp('c_OthService','a_Extraction','RestofWorld') = 5412.917969 ;
vmfp('c_OthService','a_ProcFood','Oceania') = 304.1353149 ;
vmfp('c_OthService','a_ProcFood','EastAsia') = 2394.13501 ;
vmfp('c_OthService','a_ProcFood','SEAsia') = 324.1337891 ;
vmfp('c_OthService','a_ProcFood','SouthAsia') = 1194.966919 ;
vmfp('c_OthService','a_ProcFood','NAmerica') = 3582.988037 ;
vmfp('c_OthService','a_ProcFood','LatinAmer') = 1389.328369 ;
vmfp('c_OthService','a_ProcFood','EU_28') = 16099.23242 ;
vmfp('c_OthService','a_ProcFood','MENA') = 1191.106201 ;
vmfp('c_OthService','a_ProcFood','SSA') = 732.43396 ;
vmfp('c_OthService','a_ProcFood','RestofWorld') = 1070.845581 ;
vmfp('c_OthService','a_TextWapp','Oceania') = 37.29088593 ;
vmfp('c_OthService','a_TextWapp','EastAsia') = 2109.021973 ;
vmfp('c_OthService','a_TextWapp','SEAsia') = 236.0016937 ;
vmfp('c_OthService','a_TextWapp','SouthAsia') = 3205.574219 ;
vmfp('c_OthService','a_TextWapp','NAmerica') = 558.5511475 ;
vmfp('c_OthService','a_TextWapp','LatinAmer') = 377.1375732 ;
vmfp('c_OthService','a_TextWapp','EU_28') = 3850.76123 ;
vmfp('c_OthService','a_TextWapp','MENA') = 1478.220947 ;
vmfp('c_OthService','a_TextWapp','SSA') = 175.357132 ;
vmfp('c_OthService','a_TextWapp','RestofWorld') = 115.9647751 ;
vmfp('c_OthService','a_LightMnfc','Oceania') = 542.3000488 ;
vmfp('c_OthService','a_LightMnfc','EastAsia') = 9545.749023 ;
vmfp('c_OthService','a_LightMnfc','SEAsia') = 816.7984619 ;
vmfp('c_OthService','a_LightMnfc','SouthAsia') = 3088.746094 ;
vmfp('c_OthService','a_LightMnfc','NAmerica') = 9387.587891 ;
vmfp('c_OthService','a_LightMnfc','LatinAmer') = 1808.911133 ;
vmfp('c_OthService','a_LightMnfc','EU_28') = 36200.71484 ;
vmfp('c_OthService','a_LightMnfc','MENA') = 3997.092529 ;
vmfp('c_OthService','a_LightMnfc','SSA') = 508.4765625 ;
vmfp('c_OthService','a_LightMnfc','RestofWorld') = 1727.183838 ;
vmfp('c_OthService','a_HeavyMnfc','Oceania') = 524.8574219 ;
vmfp('c_OthService','a_HeavyMnfc','EastAsia') = 24863.85938 ;
vmfp('c_OthService','a_HeavyMnfc','SEAsia') = 9991.741211 ;
vmfp('c_OthService','a_HeavyMnfc','SouthAsia') = 5217.031738 ;
vmfp('c_OthService','a_HeavyMnfc','NAmerica') = 19271.85547 ;
vmfp('c_OthService','a_HeavyMnfc','LatinAmer') = 3383.075439 ;
vmfp('c_OthService','a_HeavyMnfc','EU_28') = 67949.35938 ;
vmfp('c_OthService','a_HeavyMnfc','MENA') = 8852.551758 ;
vmfp('c_OthService','a_HeavyMnfc','SSA') = 1063.630615 ;
vmfp('c_OthService','a_HeavyMnfc','RestofWorld') = 3490.965332 ;
vmfp('c_OthService','a_Util_Cons','Oceania') = 1332.564819 ;
vmfp('c_OthService','a_Util_Cons','EastAsia') = 9141.079102 ;
vmfp('c_OthService','a_Util_Cons','SEAsia') = 5494.052734 ;
vmfp('c_OthService','a_Util_Cons','SouthAsia') = 1874.40979 ;
vmfp('c_OthService','a_Util_Cons','NAmerica') = 17932.58789 ;
vmfp('c_OthService','a_Util_Cons','LatinAmer') = 2004.430176 ;
vmfp('c_OthService','a_Util_Cons','EU_28') = 31287.37305 ;
vmfp('c_OthService','a_Util_Cons','MENA') = 8339.208984 ;
vmfp('c_OthService','a_Util_Cons','SSA') = 2087.615234 ;
vmfp('c_OthService','a_Util_Cons','RestofWorld') = 3068.402588 ;
vmfp('c_OthService','a_TransComm','Oceania') = 3698.924072 ;
vmfp('c_OthService','a_TransComm','EastAsia') = 36355.32031 ;
vmfp('c_OthService','a_TransComm','SEAsia') = 10248.21094 ;
vmfp('c_OthService','a_TransComm','SouthAsia') = 6833.327148 ;
vmfp('c_OthService','a_TransComm','NAmerica') = 53368.61719 ;
vmfp('c_OthService','a_TransComm','LatinAmer') = 10691.6543 ;
vmfp('c_OthService','a_TransComm','EU_28') = 86609.98438 ;
vmfp('c_OthService','a_TransComm','MENA') = 12470.00977 ;
vmfp('c_OthService','a_TransComm','SSA') = 5248.005371 ;
vmfp('c_OthService','a_TransComm','RestofWorld') = 11145.79395 ;
vmfp('c_OthService','a_OthService','Oceania') = 10794.20801 ;
vmfp('c_OthService','a_OthService','EastAsia') = 57412.14844 ;
vmfp('c_OthService','a_OthService','SEAsia') = 24520.78125 ;
vmfp('c_OthService','a_OthService','SouthAsia') = 11908.76562 ;
vmfp('c_OthService','a_OthService','NAmerica') = 130576.7266 ;
vmfp('c_OthService','a_OthService','LatinAmer') = 17923.03516 ;
vmfp('c_OthService','a_OthService','EU_28') = 301089 ;
vmfp('c_OthService','a_OthService','MENA') = 17658.5625 ;
vmfp('c_OthService','a_OthService','SSA') = 8289.605469 ;
vmfp('c_OthService','a_OthService','RestofWorld') = 34579.28906 ;

* makb data (100 cells)
makb('c_Crops','a_agricultur','Oceania') = 35783.14062 ;
makb('c_Crops','a_agricultur','EastAsia') = 707528.1875 ;
makb('c_Crops','a_agricultur','SEAsia') = 268391.8125 ;
makb('c_Crops','a_agricultur','SouthAsia') = 467304.5938 ;
makb('c_Crops','a_agricultur','NAmerica') = 306241.5625 ;
makb('c_Crops','a_agricultur','LatinAmer') = 320385.2812 ;
makb('c_Crops','a_agricultur','EU_28') = 284651.625 ;
makb('c_Crops','a_agricultur','MENA') = 224854.0625 ;
makb('c_Crops','a_agricultur','SSA') = 263322.5625 ;
makb('c_Crops','a_agricultur','RestofWorld') = 122257.4141 ;
makb('c_MeatLstk','a_agricultur','Oceania') = 61168.10938 ;
makb('c_MeatLstk','a_agricultur','EastAsia') = 619780.125 ;
makb('c_MeatLstk','a_agricultur','SEAsia') = 89533.13281 ;
makb('c_MeatLstk','a_agricultur','SouthAsia') = 152625.4062 ;
makb('c_MeatLstk','a_agricultur','NAmerica') = 443853.5938 ;
makb('c_MeatLstk','a_agricultur','LatinAmer') = 295333.0625 ;
makb('c_MeatLstk','a_agricultur','EU_28') = 464652.0938 ;
makb('c_MeatLstk','a_agricultur','MENA') = 139204.5 ;
makb('c_MeatLstk','a_agricultur','SSA') = 103534.5938 ;
makb('c_MeatLstk','a_agricultur','RestofWorld') = 179520.8438 ;
makb('c_Extraction','a_Extraction','Oceania') = 219719.5625 ;
makb('c_Extraction','a_Extraction','EastAsia') = 822869.5 ;
makb('c_Extraction','a_Extraction','SEAsia') = 260976.2344 ;
makb('c_Extraction','a_Extraction','SouthAsia') = 125471.9453 ;
makb('c_Extraction','a_Extraction','NAmerica') = 658813 ;
makb('c_Extraction','a_Extraction','LatinAmer') = 413803.875 ;
makb('c_Extraction','a_Extraction','EU_28') = 302671.3438 ;
makb('c_Extraction','a_Extraction','MENA') = 1197566.75 ;
makb('c_Extraction','a_Extraction','SSA') = 307017.9062 ;
makb('c_Extraction','a_Extraction','RestofWorld') = 766122.6875 ;
makb('c_ProcFood','a_ProcFood','Oceania') = 99504.35938 ;
makb('c_ProcFood','a_ProcFood','EastAsia') = 1314232.5 ;
makb('c_ProcFood','a_ProcFood','SEAsia') = 296117.5625 ;
makb('c_ProcFood','a_ProcFood','SouthAsia') = 275570.5312 ;
makb('c_ProcFood','a_ProcFood','NAmerica') = 902711.0625 ;
makb('c_ProcFood','a_ProcFood','LatinAmer') = 503925.3438 ;
makb('c_ProcFood','a_ProcFood','EU_28') = 1437640.125 ;
makb('c_ProcFood','a_ProcFood','MENA') = 264420.0312 ;
makb('c_ProcFood','a_ProcFood','SSA') = 143965.8281 ;
makb('c_ProcFood','a_ProcFood','RestofWorld') = 273825 ;
makb('c_TextWapp','a_TextWapp','Oceania') = 11295.97949 ;
makb('c_TextWapp','a_TextWapp','EastAsia') = 939961.0625 ;
makb('c_TextWapp','a_TextWapp','SEAsia') = 125586.4141 ;
makb('c_TextWapp','a_TextWapp','SouthAsia') = 204795.7656 ;
makb('c_TextWapp','a_TextWapp','NAmerica') = 332630.1562 ;
makb('c_TextWapp','a_TextWapp','LatinAmer') = 162252.5469 ;
makb('c_TextWapp','a_TextWapp','EU_28') = 486424.5625 ;
makb('c_TextWapp','a_TextWapp','MENA') = 185141.9375 ;
makb('c_TextWapp','a_TextWapp','SSA') = 40388.71484 ;
makb('c_TextWapp','a_TextWapp','RestofWorld') = 43628.55078 ;
makb('c_LightMnfc','a_LightMnfc','Oceania') = 144404.8281 ;
makb('c_LightMnfc','a_LightMnfc','EastAsia') = 3913877.5 ;
makb('c_LightMnfc','a_LightMnfc','SEAsia') = 410461.3125 ;
makb('c_LightMnfc','a_LightMnfc','SouthAsia') = 342028.8125 ;
makb('c_LightMnfc','a_LightMnfc','NAmerica') = 2725212.5 ;
makb('c_LightMnfc','a_LightMnfc','LatinAmer') = 610960.0625 ;
makb('c_LightMnfc','a_LightMnfc','EU_28') = 3740711.5 ;
makb('c_LightMnfc','a_LightMnfc','MENA') = 376181.6875 ;
makb('c_LightMnfc','a_LightMnfc','SSA') = 166871.8906 ;
makb('c_LightMnfc','a_LightMnfc','RestofWorld') = 337023.375 ;
makb('c_HeavyMnfc','a_HeavyMnfc','Oceania') = 238532.2812 ;
makb('c_HeavyMnfc','a_HeavyMnfc','EastAsia') = 10755674 ;
makb('c_HeavyMnfc','a_HeavyMnfc','SEAsia') = 1149335.25 ;
makb('c_HeavyMnfc','a_HeavyMnfc','SouthAsia') = 810438.9375 ;
makb('c_HeavyMnfc','a_HeavyMnfc','NAmerica') = 4862524 ;
makb('c_HeavyMnfc','a_HeavyMnfc','LatinAmer') = 1181596.375 ;
makb('c_HeavyMnfc','a_HeavyMnfc','EU_28') = 6233647 ;
makb('c_HeavyMnfc','a_HeavyMnfc','MENA') = 1154324 ;
makb('c_HeavyMnfc','a_HeavyMnfc','SSA') = 298965.3125 ;
makb('c_HeavyMnfc','a_HeavyMnfc','RestofWorld') = 1209737.125 ;
makb('c_Util_Cons','a_Util_Cons','Oceania') = 416360.8438 ;
makb('c_Util_Cons','a_Util_Cons','EastAsia') = 3391842.5 ;
makb('c_Util_Cons','a_Util_Cons','SEAsia') = 492244.25 ;
makb('c_Util_Cons','a_Util_Cons','SouthAsia') = 573027 ;
makb('c_Util_Cons','a_Util_Cons','NAmerica') = 2948970.5 ;
makb('c_Util_Cons','a_Util_Cons','LatinAmer') = 688540.4375 ;
makb('c_Util_Cons','a_Util_Cons','EU_28') = 3172166 ;
makb('c_Util_Cons','a_Util_Cons','MENA') = 910209.625 ;
makb('c_Util_Cons','a_Util_Cons','SSA') = 202448.0156 ;
makb('c_Util_Cons','a_Util_Cons','RestofWorld') = 917832.3125 ;
makb('c_TransComm','a_TransComm','Oceania') = 578947.5625 ;
makb('c_TransComm','a_TransComm','EastAsia') = 5364752.5 ;
makb('c_TransComm','a_TransComm','SEAsia') = 853048.25 ;
makb('c_TransComm','a_TransComm','SouthAsia') = 860094.1875 ;
makb('c_TransComm','a_TransComm','NAmerica') = 5746750 ;
makb('c_TransComm','a_TransComm','LatinAmer') = 1570832.5 ;
makb('c_TransComm','a_TransComm','EU_28') = 5394882.5 ;
makb('c_TransComm','a_TransComm','MENA') = 1031421.812 ;
makb('c_TransComm','a_TransComm','SSA') = 428336.5 ;
makb('c_TransComm','a_TransComm','RestofWorld') = 1433487.5 ;
makb('c_OthService','a_OthService','Oceania') = 1237249.5 ;
makb('c_OthService','a_OthService','EastAsia') = 8719106 ;
makb('c_OthService','a_OthService','SEAsia') = 914694.9375 ;
makb('c_OthService','a_OthService','SouthAsia') = 779242.6875 ;
makb('c_OthService','a_OthService','NAmerica') = 14511170 ;
makb('c_OthService','a_OthService','LatinAmer') = 2499774.25 ;
makb('c_OthService','a_OthService','EU_28') = 13058007 ;
makb('c_OthService','a_OthService','MENA') = 1443460.625 ;
makb('c_OthService','a_OthService','SSA') = 613457.5 ;
makb('c_OthService','a_OthService','RestofWorld') = 1789031.625 ;

* maks data (100 cells)
maks('c_Crops','a_agricultur','Oceania') = 35776.43359 ;
maks('c_Crops','a_agricultur','EastAsia') = 708224.8125 ;
maks('c_Crops','a_agricultur','SEAsia') = 268433.9688 ;
maks('c_Crops','a_agricultur','SouthAsia') = 490609.5938 ;
maks('c_Crops','a_agricultur','NAmerica') = 306342.875 ;
maks('c_Crops','a_agricultur','LatinAmer') = 318563.5 ;
maks('c_Crops','a_agricultur','EU_28') = 284195.375 ;
maks('c_Crops','a_agricultur','MENA') = 224286.6875 ;
maks('c_Crops','a_agricultur','SSA') = 263522.3125 ;
maks('c_Crops','a_agricultur','RestofWorld') = 121656.75 ;
maks('c_MeatLstk','a_agricultur','Oceania') = 61047.35938 ;
maks('c_MeatLstk','a_agricultur','EastAsia') = 620909.5 ;
maks('c_MeatLstk','a_agricultur','SEAsia') = 89480.84375 ;
maks('c_MeatLstk','a_agricultur','SouthAsia') = 150394.6875 ;
maks('c_MeatLstk','a_agricultur','NAmerica') = 442027.7188 ;
maks('c_MeatLstk','a_agricultur','LatinAmer') = 292747.125 ;
maks('c_MeatLstk','a_agricultur','EU_28') = 455960.0312 ;
maks('c_MeatLstk','a_agricultur','MENA') = 138503.1406 ;
maks('c_MeatLstk','a_agricultur','SSA') = 102296.6328 ;
maks('c_MeatLstk','a_agricultur','RestofWorld') = 179422.1562 ;
maks('c_Extraction','a_Extraction','Oceania') = 218412.3594 ;
maks('c_Extraction','a_Extraction','EastAsia') = 821779.0625 ;
maks('c_Extraction','a_Extraction','SEAsia') = 258052.375 ;
maks('c_Extraction','a_Extraction','SouthAsia') = 123962.2734 ;
maks('c_Extraction','a_Extraction','NAmerica') = 601070.25 ;
maks('c_Extraction','a_Extraction','LatinAmer') = 409313.6562 ;
maks('c_Extraction','a_Extraction','EU_28') = 298331.4688 ;
maks('c_Extraction','a_Extraction','MENA') = 1190599.875 ;
maks('c_Extraction','a_Extraction','SSA') = 305530.7812 ;
maks('c_Extraction','a_Extraction','RestofWorld') = 753252.875 ;
maks('c_ProcFood','a_ProcFood','Oceania') = 98884.86719 ;
maks('c_ProcFood','a_ProcFood','EastAsia') = 1277568.75 ;
maks('c_ProcFood','a_ProcFood','SEAsia') = 295948.5 ;
maks('c_ProcFood','a_ProcFood','SouthAsia') = 268182.1875 ;
maks('c_ProcFood','a_ProcFood','NAmerica') = 896149.75 ;
maks('c_ProcFood','a_ProcFood','LatinAmer') = 496979.1562 ;
maks('c_ProcFood','a_ProcFood','EU_28') = 1352297.5 ;
maks('c_ProcFood','a_ProcFood','MENA') = 262856 ;
maks('c_ProcFood','a_ProcFood','SSA') = 143426.4375 ;
maks('c_ProcFood','a_ProcFood','RestofWorld') = 272025.9375 ;
maks('c_TextWapp','a_TextWapp','Oceania') = 11212.62793 ;
maks('c_TextWapp','a_TextWapp','EastAsia') = 938226.1875 ;
maks('c_TextWapp','a_TextWapp','SEAsia') = 125472.8203 ;
maks('c_TextWapp','a_TextWapp','SouthAsia') = 203339.75 ;
maks('c_TextWapp','a_TextWapp','NAmerica') = 330075.0625 ;
maks('c_TextWapp','a_TextWapp','LatinAmer') = 160769.125 ;
maks('c_TextWapp','a_TextWapp','EU_28') = 449200.875 ;
maks('c_TextWapp','a_TextWapp','MENA') = 184340.7031 ;
maks('c_TextWapp','a_TextWapp','SSA') = 40271.76172 ;
maks('c_TextWapp','a_TextWapp','RestofWorld') = 43120.93359 ;
maks('c_LightMnfc','a_LightMnfc','Oceania') = 143190.7656 ;
maks('c_LightMnfc','a_LightMnfc','EastAsia') = 3896531.75 ;
maks('c_LightMnfc','a_LightMnfc','SEAsia') = 410059.9375 ;
maks('c_LightMnfc','a_LightMnfc','SouthAsia') = 337211.6875 ;
maks('c_LightMnfc','a_LightMnfc','NAmerica') = 2707609.5 ;
maks('c_LightMnfc','a_LightMnfc','LatinAmer') = 601927.625 ;
maks('c_LightMnfc','a_LightMnfc','EU_28') = 3617432.5 ;
maks('c_LightMnfc','a_LightMnfc','MENA') = 374755.2188 ;
maks('c_LightMnfc','a_LightMnfc','SSA') = 166289.5469 ;
maks('c_LightMnfc','a_LightMnfc','RestofWorld') = 335965.4062 ;
maks('c_HeavyMnfc','a_HeavyMnfc','Oceania') = 236993.5156 ;
maks('c_HeavyMnfc','a_HeavyMnfc','EastAsia') = 10648941 ;
maks('c_HeavyMnfc','a_HeavyMnfc','SEAsia') = 1152180.25 ;
maks('c_HeavyMnfc','a_HeavyMnfc','SouthAsia') = 799790.1875 ;
maks('c_HeavyMnfc','a_HeavyMnfc','NAmerica') = 4814379 ;
maks('c_HeavyMnfc','a_HeavyMnfc','LatinAmer') = 1174674.75 ;
maks('c_HeavyMnfc','a_HeavyMnfc','EU_28') = 5829068.5 ;
maks('c_HeavyMnfc','a_HeavyMnfc','MENA') = 1151436.125 ;
maks('c_HeavyMnfc','a_HeavyMnfc','SSA') = 294293.25 ;
maks('c_HeavyMnfc','a_HeavyMnfc','RestofWorld') = 1203844 ;
maks('c_Util_Cons','a_Util_Cons','Oceania') = 412876.9688 ;
maks('c_Util_Cons','a_Util_Cons','EastAsia') = 3376593.5 ;
maks('c_Util_Cons','a_Util_Cons','SEAsia') = 492064.9688 ;
maks('c_Util_Cons','a_Util_Cons','SouthAsia') = 576067.375 ;
maks('c_Util_Cons','a_Util_Cons','NAmerica') = 2905612.5 ;
maks('c_Util_Cons','a_Util_Cons','LatinAmer') = 681501.8125 ;
maks('c_Util_Cons','a_Util_Cons','EU_28') = 3087445.25 ;
maks('c_Util_Cons','a_Util_Cons','MENA') = 904247.375 ;
maks('c_Util_Cons','a_Util_Cons','SSA') = 200372.1094 ;
maks('c_Util_Cons','a_Util_Cons','RestofWorld') = 912070.75 ;
maks('c_TransComm','a_TransComm','Oceania') = 571014.0625 ;
maks('c_TransComm','a_TransComm','EastAsia') = 5334474.5 ;
maks('c_TransComm','a_TransComm','SEAsia') = 851262.625 ;
maks('c_TransComm','a_TransComm','SouthAsia') = 855722.8125 ;
maks('c_TransComm','a_TransComm','NAmerica') = 5627552.5 ;
maks('c_TransComm','a_TransComm','LatinAmer') = 1530643.625 ;
maks('c_TransComm','a_TransComm','EU_28') = 5342793 ;
maks('c_TransComm','a_TransComm','MENA') = 1028527.75 ;
maks('c_TransComm','a_TransComm','SSA') = 423415.875 ;
maks('c_TransComm','a_TransComm','RestofWorld') = 1430725 ;
maks('c_OthService','a_OthService','Oceania') = 1209136.625 ;
maks('c_OthService','a_OthService','EastAsia') = 8632127 ;
maks('c_OthService','a_OthService','SEAsia') = 911508.25 ;
maks('c_OthService','a_OthService','SouthAsia') = 775464.625 ;
maks('c_OthService','a_OthService','NAmerica') = 14038279 ;
maks('c_OthService','a_OthService','LatinAmer') = 2487628.25 ;
maks('c_OthService','a_OthService','EU_28') = 12918810 ;
maks('c_OthService','a_OthService','MENA') = 1437290.875 ;
maks('c_OthService','a_OthService','SSA') = 606050.375 ;
maks('c_OthService','a_OthService','RestofWorld') = 1782121.875 ;

* evfb data (290 cells)
evfb('Land','a_agricultur','Oceania') = 7369.601074 ;
evfb('Land','a_agricultur','EastAsia') = 211169.2812 ;
evfb('Land','a_agricultur','SEAsia') = 92604.85156 ;
evfb('Land','a_agricultur','SouthAsia') = 153384.1875 ;
evfb('Land','a_agricultur','NAmerica') = 66172.50781 ;
evfb('Land','a_agricultur','LatinAmer') = 58093.97266 ;
evfb('Land','a_agricultur','EU_28') = 65391.40625 ;
evfb('Land','a_agricultur','MENA') = 20565.90039 ;
evfb('Land','a_agricultur','SSA') = 28765.73242 ;
evfb('Land','a_agricultur','RestofWorld') = 30123.78125 ;
evfb('UnSkLab','a_agricultur','Oceania') = 9930.836914 ;
evfb('UnSkLab','a_agricultur','EastAsia') = 355532.125 ;
evfb('UnSkLab','a_agricultur','SEAsia') = 82939.40625 ;
evfb('UnSkLab','a_agricultur','SouthAsia') = 145172.9062 ;
evfb('UnSkLab','a_agricultur','NAmerica') = 59619.64062 ;
evfb('UnSkLab','a_agricultur','LatinAmer') = 79796.83594 ;
evfb('UnSkLab','a_agricultur','EU_28') = 123831.2891 ;
evfb('UnSkLab','a_agricultur','MENA') = 96430.10156 ;
evfb('UnSkLab','a_agricultur','SSA') = 142434.6719 ;
evfb('UnSkLab','a_agricultur','RestofWorld') = 50919.58594 ;
evfb('UnSkLab','a_Extraction','Oceania') = 9172.714844 ;
evfb('UnSkLab','a_Extraction','EastAsia') = 144467.7812 ;
evfb('UnSkLab','a_Extraction','SEAsia') = 28538.64258 ;
evfb('UnSkLab','a_Extraction','SouthAsia') = 24399.19922 ;
evfb('UnSkLab','a_Extraction','NAmerica') = 36512.42969 ;
evfb('UnSkLab','a_Extraction','LatinAmer') = 23539.55859 ;
evfb('UnSkLab','a_Extraction','EU_28') = 19410.89844 ;
evfb('UnSkLab','a_Extraction','MENA') = 32351.89062 ;
evfb('UnSkLab','a_Extraction','SSA') = 24520.66797 ;
evfb('UnSkLab','a_Extraction','RestofWorld') = 30185.625 ;
evfb('UnSkLab','a_ProcFood','Oceania') = 7115.373535 ;
evfb('UnSkLab','a_ProcFood','EastAsia') = 91300.04688 ;
evfb('UnSkLab','a_ProcFood','SEAsia') = 25148.49414 ;
evfb('UnSkLab','a_ProcFood','SouthAsia') = 20875.36328 ;
evfb('UnSkLab','a_ProcFood','NAmerica') = 80107.73438 ;
evfb('UnSkLab','a_ProcFood','LatinAmer') = 26079.49219 ;
evfb('UnSkLab','a_ProcFood','EU_28') = 78288.02344 ;
evfb('UnSkLab','a_ProcFood','MENA') = 17624.12109 ;
evfb('UnSkLab','a_ProcFood','SSA') = 9436.861328 ;
evfb('UnSkLab','a_ProcFood','RestofWorld') = 17537.02148 ;
evfb('UnSkLab','a_TextWapp','Oceania') = 1173.755005 ;
evfb('UnSkLab','a_TextWapp','EastAsia') = 104614.4531 ;
evfb('UnSkLab','a_TextWapp','SEAsia') = 10868.58398 ;
evfb('UnSkLab','a_TextWapp','SouthAsia') = 20743.50195 ;
evfb('UnSkLab','a_TextWapp','NAmerica') = 42924.57812 ;
evfb('UnSkLab','a_TextWapp','LatinAmer') = 13934.24414 ;
evfb('UnSkLab','a_TextWapp','EU_28') = 36401.25 ;
evfb('UnSkLab','a_TextWapp','MENA') = 13132.09961 ;
evfb('UnSkLab','a_TextWapp','SSA') = 3668.302979 ;
evfb('UnSkLab','a_TextWapp','RestofWorld') = 3990.070312 ;
evfb('UnSkLab','a_LightMnfc','Oceania') = 14635.03613 ;
evfb('UnSkLab','a_LightMnfc','EastAsia') = 354543.9375 ;
evfb('UnSkLab','a_LightMnfc','SEAsia') = 34505.44141 ;
evfb('UnSkLab','a_LightMnfc','SouthAsia') = 41328.95703 ;
evfb('UnSkLab','a_LightMnfc','NAmerica') = 377784.25 ;
evfb('UnSkLab','a_LightMnfc','LatinAmer') = 42582.97266 ;
evfb('UnSkLab','a_LightMnfc','EU_28') = 256751.8594 ;
evfb('UnSkLab','a_LightMnfc','MENA') = 32309.95508 ;
evfb('UnSkLab','a_LightMnfc','SSA') = 10901.38867 ;
evfb('UnSkLab','a_LightMnfc','RestofWorld') = 38372.79297 ;
evfb('UnSkLab','a_HeavyMnfc','Oceania') = 15010.05957 ;
evfb('UnSkLab','a_HeavyMnfc','EastAsia') = 816193.8125 ;
evfb('UnSkLab','a_HeavyMnfc','SEAsia') = 58782.47656 ;
evfb('UnSkLab','a_HeavyMnfc','SouthAsia') = 36030.64844 ;
evfb('UnSkLab','a_HeavyMnfc','NAmerica') = 507698.75 ;
evfb('UnSkLab','a_HeavyMnfc','LatinAmer') = 53450.13672 ;
evfb('UnSkLab','a_HeavyMnfc','EU_28') = 346970.8438 ;
evfb('UnSkLab','a_HeavyMnfc','MENA') = 57421.80469 ;
evfb('UnSkLab','a_HeavyMnfc','SSA') = 15024.30859 ;
evfb('UnSkLab','a_HeavyMnfc','RestofWorld') = 75567.07031 ;
evfb('UnSkLab','a_Util_Cons','Oceania') = 63256.91797 ;
evfb('UnSkLab','a_Util_Cons','EastAsia') = 592787.8125 ;
evfb('UnSkLab','a_Util_Cons','SEAsia') = 66048.61719 ;
evfb('UnSkLab','a_Util_Cons','SouthAsia') = 90648.25781 ;
evfb('UnSkLab','a_Util_Cons','NAmerica') = 664249 ;
evfb('UnSkLab','a_Util_Cons','LatinAmer') = 120167.9219 ;
evfb('UnSkLab','a_Util_Cons','EU_28') = 339518.5312 ;
evfb('UnSkLab','a_Util_Cons','MENA') = 99562.89844 ;
evfb('UnSkLab','a_Util_Cons','SSA') = 20415.68359 ;
evfb('UnSkLab','a_Util_Cons','RestofWorld') = 109003.4766 ;
evfb('UnSkLab','a_TransComm','Oceania') = 98654.63281 ;
evfb('UnSkLab','a_TransComm','EastAsia') = 896363.0625 ;
evfb('UnSkLab','a_TransComm','SEAsia') = 129632.875 ;
evfb('UnSkLab','a_TransComm','SouthAsia') = 152234.1562 ;
evfb('UnSkLab','a_TransComm','NAmerica') = 1049096.75 ;
evfb('UnSkLab','a_TransComm','LatinAmer') = 261515.5312 ;
evfb('UnSkLab','a_TransComm','EU_28') = 398097.7188 ;
evfb('UnSkLab','a_TransComm','MENA') = 113495.125 ;
evfb('UnSkLab','a_TransComm','SSA') = 61193.21094 ;
evfb('UnSkLab','a_TransComm','RestofWorld') = 141317.1562 ;
evfb('UnSkLab','a_OthService','Oceania') = 96479.90625 ;
evfb('UnSkLab','a_OthService','EastAsia') = 953887.75 ;
evfb('UnSkLab','a_OthService','SEAsia') = 80394.77344 ;
evfb('UnSkLab','a_OthService','SouthAsia') = 53232.30859 ;
evfb('UnSkLab','a_OthService','NAmerica') = 1369530.75 ;
evfb('UnSkLab','a_OthService','LatinAmer') = 339888.9375 ;
evfb('UnSkLab','a_OthService','EU_28') = 975554.375 ;
evfb('UnSkLab','a_OthService','MENA') = 177216.2188 ;
evfb('UnSkLab','a_OthService','SSA') = 65548.45312 ;
evfb('UnSkLab','a_OthService','RestofWorld') = 132456.6562 ;
evfb('SkLab','a_agricultur','Oceania') = 13153.28809 ;
evfb('SkLab','a_agricultur','EastAsia') = 24890.48633 ;
evfb('SkLab','a_agricultur','SEAsia') = 5096.808594 ;
evfb('SkLab','a_agricultur','SouthAsia') = 3106.857666 ;
evfb('SkLab','a_agricultur','NAmerica') = 65087.16406 ;
evfb('SkLab','a_agricultur','LatinAmer') = 23343.83594 ;
evfb('SkLab','a_agricultur','EU_28') = 44342.62891 ;
evfb('SkLab','a_agricultur','MENA') = 4891.472656 ;
evfb('SkLab','a_agricultur','SSA') = 16727.06641 ;
evfb('SkLab','a_agricultur','RestofWorld') = 5268.644531 ;
evfb('SkLab','a_Extraction','Oceania') = 11348.60156 ;
evfb('SkLab','a_Extraction','EastAsia') = 12300.27832 ;
evfb('SkLab','a_Extraction','SEAsia') = 8843.916992 ;
evfb('SkLab','a_Extraction','SouthAsia') = 1190.8479 ;
evfb('SkLab','a_Extraction','NAmerica') = 35941.04688 ;
evfb('SkLab','a_Extraction','LatinAmer') = 21366.11523 ;
evfb('SkLab','a_Extraction','EU_28') = 13022.62207 ;
evfb('SkLab','a_Extraction','MENA') = 38810.90234 ;
evfb('SkLab','a_Extraction','SSA') = 10235.13574 ;
evfb('SkLab','a_Extraction','RestofWorld') = 20019.35352 ;
evfb('SkLab','a_ProcFood','Oceania') = 7371.807129 ;
evfb('SkLab','a_ProcFood','EastAsia') = 29077.82617 ;
evfb('SkLab','a_ProcFood','SEAsia') = 8154.8125 ;
evfb('SkLab','a_ProcFood','SouthAsia') = 4084.06543 ;
evfb('SkLab','a_ProcFood','NAmerica') = 49528.26562 ;
evfb('SkLab','a_ProcFood','LatinAmer') = 33866.14844 ;
evfb('SkLab','a_ProcFood','EU_28') = 84784.03125 ;
evfb('SkLab','a_ProcFood','MENA') = 7763.168457 ;
evfb('SkLab','a_ProcFood','SSA') = 8610.685547 ;
evfb('SkLab','a_ProcFood','RestofWorld') = 10138.75 ;
evfb('SkLab','a_TextWapp','Oceania') = 1218.902832 ;
evfb('SkLab','a_TextWapp','EastAsia') = 27356.92969 ;
evfb('SkLab','a_TextWapp','SEAsia') = 3791.776367 ;
evfb('SkLab','a_TextWapp','SouthAsia') = 5161.087891 ;
evfb('SkLab','a_TextWapp','NAmerica') = 27128.00586 ;
evfb('SkLab','a_TextWapp','LatinAmer') = 19581.01367 ;
evfb('SkLab','a_TextWapp','EU_28') = 38006.35938 ;
evfb('SkLab','a_TextWapp','MENA') = 6904.578125 ;
evfb('SkLab','a_TextWapp','SSA') = 3425.343262 ;
evfb('SkLab','a_TextWapp','RestofWorld') = 2334.974854 ;
evfb('SkLab','a_LightMnfc','Oceania') = 15127.37695 ;
evfb('SkLab','a_LightMnfc','EastAsia') = 121198.4375 ;
evfb('SkLab','a_LightMnfc','SEAsia') = 14758.44434 ;
evfb('SkLab','a_LightMnfc','SouthAsia') = 7587.55127 ;
evfb('SkLab','a_LightMnfc','NAmerica') = 247154.3594 ;
evfb('SkLab','a_LightMnfc','LatinAmer') = 62897.16406 ;
evfb('SkLab','a_LightMnfc','EU_28') = 275108.0938 ;
evfb('SkLab','a_LightMnfc','MENA') = 12456.72852 ;
evfb('SkLab','a_LightMnfc','SSA') = 10563.20605 ;
evfb('SkLab','a_LightMnfc','RestofWorld') = 26297.99805 ;
evfb('SkLab','a_HeavyMnfc','Oceania') = 15518.40332 ;
evfb('SkLab','a_HeavyMnfc','EastAsia') = 244194.0469 ;
evfb('SkLab','a_HeavyMnfc','SEAsia') = 35746.0625 ;
evfb('SkLab','a_HeavyMnfc','SouthAsia') = 6603.109375 ;
evfb('SkLab','a_HeavyMnfc','NAmerica') = 319834 ;
evfb('SkLab','a_HeavyMnfc','LatinAmer') = 81986.39844 ;
evfb('SkLab','a_HeavyMnfc','EU_28') = 367171.875 ;
evfb('SkLab','a_HeavyMnfc','MENA') = 21580.94336 ;
evfb('SkLab','a_HeavyMnfc','SSA') = 12849.23242 ;
evfb('SkLab','a_HeavyMnfc','RestofWorld') = 48919.93359 ;
evfb('SkLab','a_Util_Cons','Oceania') = 32984.51172 ;
evfb('SkLab','a_Util_Cons','EastAsia') = 108909.4219 ;
evfb('SkLab','a_Util_Cons','SEAsia') = 22801.5918 ;
evfb('SkLab','a_Util_Cons','SouthAsia') = 51157.88672 ;
evfb('SkLab','a_Util_Cons','NAmerica') = 234379.4062 ;
evfb('SkLab','a_Util_Cons','LatinAmer') = 33712.19141 ;
evfb('SkLab','a_Util_Cons','EU_28') = 154407.625 ;
evfb('SkLab','a_Util_Cons','MENA') = 36582.61719 ;
evfb('SkLab','a_Util_Cons','SSA') = 10871.02734 ;
evfb('SkLab','a_Util_Cons','RestofWorld') = 52590.62891 ;
evfb('SkLab','a_TransComm','Oceania') = 86943.77344 ;
evfb('SkLab','a_TransComm','EastAsia') = 439797.3438 ;
evfb('SkLab','a_TransComm','SEAsia') = 51965.10156 ;
evfb('SkLab','a_TransComm','SouthAsia') = 68481.75781 ;
evfb('SkLab','a_TransComm','NAmerica') = 808185.625 ;
evfb('SkLab','a_TransComm','LatinAmer') = 153688.8594 ;
evfb('SkLab','a_TransComm','EU_28') = 455406.3438 ;
evfb('SkLab','a_TransComm','MENA') = 73497.85938 ;
evfb('SkLab','a_TransComm','SSA') = 62462.73438 ;
evfb('SkLab','a_TransComm','RestofWorld') = 96213.25781 ;
evfb('SkLab','a_OthService','Oceania') = 300018.5312 ;
evfb('SkLab','a_OthService','EastAsia') = 1781804.625 ;
evfb('SkLab','a_OthService','SEAsia') = 184427.3281 ;
evfb('SkLab','a_OthService','SouthAsia') = 288395.1562 ;
evfb('SkLab','a_OthService','NAmerica') = 4236516 ;
evfb('SkLab','a_OthService','LatinAmer') = 585989.125 ;
evfb('SkLab','a_OthService','EU_28') = 2251345.5 ;
evfb('SkLab','a_OthService','MENA') = 285659.4062 ;
evfb('SkLab','a_OthService','SSA') = 152040.9844 ;
evfb('SkLab','a_OthService','RestofWorld') = 453074.8125 ;
evfb('Capital','a_agricultur','Oceania') = 12180.70117 ;
evfb('Capital','a_agricultur','EastAsia') = 111232.4375 ;
evfb('Capital','a_agricultur','SEAsia') = 21994.08398 ;
evfb('Capital','a_agricultur','SouthAsia') = 83298.01562 ;
evfb('Capital','a_agricultur','NAmerica') = 87451.79688 ;
evfb('Capital','a_agricultur','LatinAmer') = 125451.3359 ;
evfb('Capital','a_agricultur','EU_28') = 115637.1797 ;
evfb('Capital','a_agricultur','MENA') = 67595.75781 ;
evfb('Capital','a_agricultur','SSA') = 49308.86328 ;
evfb('Capital','a_agricultur','RestofWorld') = 31298.54883 ;
evfb('Capital','a_Extraction','Oceania') = 74038.61719 ;
evfb('Capital','a_Extraction','EastAsia') = 137725.6875 ;
evfb('Capital','a_Extraction','SEAsia') = 106607.3047 ;
evfb('Capital','a_Extraction','SouthAsia') = 38469.20703 ;
evfb('Capital','a_Extraction','NAmerica') = 173471.0469 ;
evfb('Capital','a_Extraction','LatinAmer') = 133560.0781 ;
evfb('Capital','a_Extraction','EU_28') = 78766.5 ;
evfb('Capital','a_Extraction','MENA') = 608239.0625 ;
evfb('Capital','a_Extraction','SSA') = 107145.0391 ;
evfb('Capital','a_Extraction','RestofWorld') = 230020.125 ;
evfb('Capital','a_ProcFood','Oceania') = 12996.37402 ;
evfb('Capital','a_ProcFood','EastAsia') = 146719.0156 ;
evfb('Capital','a_ProcFood','SEAsia') = 46634.24609 ;
evfb('Capital','a_ProcFood','SouthAsia') = 41242.62109 ;
evfb('Capital','a_ProcFood','NAmerica') = 147115.8438 ;
evfb('Capital','a_ProcFood','LatinAmer') = 68018.67188 ;
evfb('Capital','a_ProcFood','EU_28') = 181818.8438 ;
evfb('Capital','a_ProcFood','MENA') = 49995.08203 ;
evfb('Capital','a_ProcFood','SSA') = 23816.79883 ;
evfb('Capital','a_ProcFood','RestofWorld') = 36135.83203 ;
evfb('Capital','a_TextWapp','Oceania') = 2368.095947 ;
evfb('Capital','a_TextWapp','EastAsia') = 65222.45703 ;
evfb('Capital','a_TextWapp','SEAsia') = 20272.57031 ;
evfb('Capital','a_TextWapp','SouthAsia') = 21371.92969 ;
evfb('Capital','a_TextWapp','NAmerica') = 24706.43945 ;
evfb('Capital','a_TextWapp','LatinAmer') = 20969.41797 ;
evfb('Capital','a_TextWapp','EU_28') = 45671.24609 ;
evfb('Capital','a_TextWapp','MENA') = 37522.54297 ;
evfb('Capital','a_TextWapp','SSA') = 4719.23291 ;
evfb('Capital','a_TextWapp','RestofWorld') = 4671.597656 ;
evfb('Capital','a_LightMnfc','Oceania') = 21891.65625 ;
evfb('Capital','a_LightMnfc','EastAsia') = 406775.2812 ;
evfb('Capital','a_LightMnfc','SEAsia') = 70657.44531 ;
evfb('Capital','a_LightMnfc','SouthAsia') = 41061.44531 ;
evfb('Capital','a_LightMnfc','NAmerica') = 259999.0469 ;
evfb('Capital','a_LightMnfc','LatinAmer') = 89911.24219 ;
evfb('Capital','a_LightMnfc','EU_28') = 365679.8125 ;
evfb('Capital','a_LightMnfc','MENA') = 80791.74219 ;
evfb('Capital','a_LightMnfc','SSA') = 25752.95117 ;
evfb('Capital','a_LightMnfc','RestofWorld') = 28031.38672 ;
evfb('Capital','a_HeavyMnfc','Oceania') = 25872.91992 ;
evfb('Capital','a_HeavyMnfc','EastAsia') = 976630.5625 ;
evfb('Capital','a_HeavyMnfc','SEAsia') = 180657.2031 ;
evfb('Capital','a_HeavyMnfc','SouthAsia') = 86309.6875 ;
evfb('Capital','a_HeavyMnfc','NAmerica') = 464238.25 ;
evfb('Capital','a_HeavyMnfc','LatinAmer') = 175928.7656 ;
evfb('Capital','a_HeavyMnfc','EU_28') = 578934.125 ;
evfb('Capital','a_HeavyMnfc','MENA') = 281931.4062 ;
evfb('Capital','a_HeavyMnfc','SSA') = 45774.06641 ;
evfb('Capital','a_HeavyMnfc','RestofWorld') = 103238.1016 ;
evfb('Capital','a_Util_Cons','Oceania') = 35818.51953 ;
evfb('Capital','a_Util_Cons','EastAsia') = 337381.4375 ;
evfb('Capital','a_Util_Cons','SEAsia') = 74138.85938 ;
evfb('Capital','a_Util_Cons','SouthAsia') = 79739.32812 ;
evfb('Capital','a_Util_Cons','NAmerica') = 368728.0625 ;
evfb('Capital','a_Util_Cons','LatinAmer') = 181209.5312 ;
evfb('Capital','a_Util_Cons','EU_28') = 614663.625 ;
evfb('Capital','a_Util_Cons','MENA') = 162540.9062 ;
evfb('Capital','a_Util_Cons','SSA') = 52937.65234 ;
evfb('Capital','a_Util_Cons','RestofWorld') = 160394.9375 ;
evfb('Capital','a_TransComm','Oceania') = 73722.17188 ;
evfb('Capital','a_TransComm','EastAsia') = 1231670 ;
evfb('Capital','a_TransComm','SEAsia') = 237897.3281 ;
evfb('Capital','a_TransComm','SouthAsia') = 292423.3125 ;
evfb('Capital','a_TransComm','NAmerica') = 949999.75 ;
evfb('Capital','a_TransComm','LatinAmer') = 387293.5312 ;
evfb('Capital','a_TransComm','EU_28') = 823554.4375 ;
evfb('Capital','a_TransComm','MENA') = 347860.4688 ;
evfb('Capital','a_TransComm','SSA') = 96593.02344 ;
evfb('Capital','a_TransComm','RestofWorld') = 466256.9375 ;
evfb('Capital','a_OthService','Oceania') = 298865.4375 ;
evfb('Capital','a_OthService','EastAsia') = 2395665.75 ;
evfb('Capital','a_OthService','SEAsia') = 288789.125 ;
evfb('Capital','a_OthService','SouthAsia') = 255301.5781 ;
evfb('Capital','a_OthService','NAmerica') = 2650531.5 ;
evfb('Capital','a_OthService','LatinAmer') = 670236.25 ;
evfb('Capital','a_OthService','EU_28') = 3451036 ;
evfb('Capital','a_OthService','MENA') = 485779.25 ;
evfb('Capital','a_OthService','SSA') = 136659.7656 ;
evfb('Capital','a_OthService','RestofWorld') = 396512.5938 ;
evfb('NatRes','a_Extraction','Oceania') = 28930.05273 ;
evfb('NatRes','a_Extraction','EastAsia') = 125906.5234 ;
evfb('NatRes','a_Extraction','SEAsia') = 53949.05859 ;
evfb('NatRes','a_Extraction','SouthAsia') = 21039.64648 ;
evfb('NatRes','a_Extraction','NAmerica') = 121577.7656 ;
evfb('NatRes','a_Extraction','LatinAmer') = 71037.27344 ;
evfb('NatRes','a_Extraction','EU_28') = 46445.40234 ;
evfb('NatRes','a_Extraction','MENA') = 306366.9375 ;
evfb('NatRes','a_Extraction','SSA') = 64919.78906 ;
evfb('NatRes','a_Extraction','RestofWorld') = 166033.1719 ;

* evfp data (290 cells)
evfp('Land','a_agricultur','Oceania') = 7443.19873 ;
evfp('Land','a_agricultur','EastAsia') = 181700.5312 ;
evfp('Land','a_agricultur','SEAsia') = 93244.8125 ;
evfp('Land','a_agricultur','SouthAsia') = 153658.75 ;
evfp('Land','a_agricultur','NAmerica') = 55379.95312 ;
evfp('Land','a_agricultur','LatinAmer') = 58788.05859 ;
evfp('Land','a_agricultur','EU_28') = 40522.625 ;
evfp('Land','a_agricultur','MENA') = 20125.80859 ;
evfp('Land','a_agricultur','SSA') = 29235.11719 ;
evfp('Land','a_agricultur','RestofWorld') = 29220.04297 ;
evfp('UnSkLab','a_agricultur','Oceania') = 9976.518555 ;
evfp('UnSkLab','a_agricultur','EastAsia') = 353161.8125 ;
evfp('UnSkLab','a_agricultur','SEAsia') = 84042.70312 ;
evfp('UnSkLab','a_agricultur','SouthAsia') = 145516.6406 ;
evfp('UnSkLab','a_agricultur','NAmerica') = 63709.97656 ;
evfp('UnSkLab','a_agricultur','LatinAmer') = 90121.03125 ;
evfp('UnSkLab','a_agricultur','EU_28') = 143875.9531 ;
evfp('UnSkLab','a_agricultur','MENA') = 105450.3672 ;
evfp('UnSkLab','a_agricultur','SSA') = 158888.4688 ;
evfp('UnSkLab','a_agricultur','RestofWorld') = 62344.94922 ;
evfp('UnSkLab','a_Extraction','Oceania') = 9561.549805 ;
evfp('UnSkLab','a_Extraction','EastAsia') = 145574.3438 ;
evfp('UnSkLab','a_Extraction','SEAsia') = 28952.74805 ;
evfp('UnSkLab','a_Extraction','SouthAsia') = 24440.01758 ;
evfp('UnSkLab','a_Extraction','NAmerica') = 41747.85547 ;
evfp('UnSkLab','a_Extraction','LatinAmer') = 26557.07812 ;
evfp('UnSkLab','a_Extraction','EU_28') = 26888.31641 ;
evfp('UnSkLab','a_Extraction','MENA') = 39873.26562 ;
evfp('UnSkLab','a_Extraction','SSA') = 27119.72266 ;
evfp('UnSkLab','a_Extraction','RestofWorld') = 37133.89453 ;
evfp('UnSkLab','a_ProcFood','Oceania') = 7392.422852 ;
evfp('UnSkLab','a_ProcFood','EastAsia') = 96903.04688 ;
evfp('UnSkLab','a_ProcFood','SEAsia') = 25517.26562 ;
evfp('UnSkLab','a_ProcFood','SouthAsia') = 20901.00391 ;
evfp('UnSkLab','a_ProcFood','NAmerica') = 91296.14844 ;
evfp('UnSkLab','a_ProcFood','LatinAmer') = 29213.22461 ;
evfp('UnSkLab','a_ProcFood','EU_28') = 109134.5625 ;
evfp('UnSkLab','a_ProcFood','MENA') = 19863.44727 ;
evfp('UnSkLab','a_ProcFood','SSA') = 9802.431641 ;
evfp('UnSkLab','a_ProcFood','RestofWorld') = 21824.04102 ;
evfp('UnSkLab','a_TextWapp','Oceania') = 1218.591797 ;
evfp('UnSkLab','a_TextWapp','EastAsia') = 107036.625 ;
evfp('UnSkLab','a_TextWapp','SEAsia') = 11033.78125 ;
evfp('UnSkLab','a_TextWapp','SouthAsia') = 20856.34766 ;
evfp('UnSkLab','a_TextWapp','NAmerica') = 49172.78125 ;
evfp('UnSkLab','a_TextWapp','LatinAmer') = 15866.26562 ;
evfp('UnSkLab','a_TextWapp','EU_28') = 49763.92578 ;
evfp('UnSkLab','a_TextWapp','MENA') = 14234.69531 ;
evfp('UnSkLab','a_TextWapp','SSA') = 3823.484863 ;
evfp('UnSkLab','a_TextWapp','RestofWorld') = 4985.827637 ;
evfp('UnSkLab','a_LightMnfc','Oceania') = 15212.8125 ;
evfp('UnSkLab','a_LightMnfc','EastAsia') = 377113.0625 ;
evfp('UnSkLab','a_LightMnfc','SEAsia') = 35011.42969 ;
evfp('UnSkLab','a_LightMnfc','SouthAsia') = 41373.65625 ;
evfp('UnSkLab','a_LightMnfc','NAmerica') = 433514.7812 ;
evfp('UnSkLab','a_LightMnfc','LatinAmer') = 48921.76953 ;
evfp('UnSkLab','a_LightMnfc','EU_28') = 360997.0938 ;
evfp('UnSkLab','a_LightMnfc','MENA') = 38401.94922 ;
evfp('UnSkLab','a_LightMnfc','SSA') = 11386.00195 ;
evfp('UnSkLab','a_LightMnfc','RestofWorld') = 48176.41797 ;
evfp('UnSkLab','a_HeavyMnfc','Oceania') = 15613.82812 ;
evfp('UnSkLab','a_HeavyMnfc','EastAsia') = 856309.6875 ;
evfp('UnSkLab','a_HeavyMnfc','SEAsia') = 59609.49609 ;
evfp('UnSkLab','a_HeavyMnfc','SouthAsia') = 36061.53516 ;
evfp('UnSkLab','a_HeavyMnfc','NAmerica') = 581075.625 ;
evfp('UnSkLab','a_HeavyMnfc','LatinAmer') = 61460.85547 ;
evfp('UnSkLab','a_HeavyMnfc','EU_28') = 487295.875 ;
evfp('UnSkLab','a_HeavyMnfc','MENA') = 68351.53906 ;
evfp('UnSkLab','a_HeavyMnfc','SSA') = 15613.59961 ;
evfp('UnSkLab','a_HeavyMnfc','RestofWorld') = 94501.29688 ;
evfp('UnSkLab','a_Util_Cons','Oceania') = 65827.98438 ;
evfp('UnSkLab','a_Util_Cons','EastAsia') = 631010 ;
evfp('UnSkLab','a_Util_Cons','SEAsia') = 66971.01562 ;
evfp('UnSkLab','a_Util_Cons','SouthAsia') = 90769.71094 ;
evfp('UnSkLab','a_Util_Cons','NAmerica') = 762325.4375 ;
evfp('UnSkLab','a_Util_Cons','LatinAmer') = 140451.4219 ;
evfp('UnSkLab','a_Util_Cons','EU_28') = 476537.3125 ;
evfp('UnSkLab','a_Util_Cons','MENA') = 115584.9297 ;
evfp('UnSkLab','a_Util_Cons','SSA') = 21326.10742 ;
evfp('UnSkLab','a_Util_Cons','RestofWorld') = 136740.9062 ;
evfp('UnSkLab','a_TransComm','Oceania') = 102718.5469 ;
evfp('UnSkLab','a_TransComm','EastAsia') = 979869.125 ;
evfp('UnSkLab','a_TransComm','SEAsia') = 131550.8281 ;
evfp('UnSkLab','a_TransComm','SouthAsia') = 152787.4375 ;
evfp('UnSkLab','a_TransComm','NAmerica') = 1203030.625 ;
evfp('UnSkLab','a_TransComm','LatinAmer') = 307100.9375 ;
evfp('UnSkLab','a_TransComm','EU_28') = 533630.5625 ;
evfp('UnSkLab','a_TransComm','MENA') = 126101.7266 ;
evfp('UnSkLab','a_TransComm','SSA') = 65831.19531 ;
evfp('UnSkLab','a_TransComm','RestofWorld') = 175664.5 ;
evfp('UnSkLab','a_OthService','Oceania') = 100348.2812 ;
evfp('UnSkLab','a_OthService','EastAsia') = 1031252 ;
evfp('UnSkLab','a_OthService','SEAsia') = 81458.92188 ;
evfp('UnSkLab','a_OthService','SouthAsia') = 53376.01953 ;
evfp('UnSkLab','a_OthService','NAmerica') = 1572906.5 ;
evfp('UnSkLab','a_OthService','LatinAmer') = 402134.9062 ;
evfp('UnSkLab','a_OthService','EU_28') = 1358460.75 ;
evfp('UnSkLab','a_OthService','MENA') = 197526.0781 ;
evfp('UnSkLab','a_OthService','SSA') = 69361.49219 ;
evfp('UnSkLab','a_OthService','RestofWorld') = 166556.5938 ;
evfp('SkLab','a_agricultur','Oceania') = 13195.84961 ;
evfp('SkLab','a_agricultur','EastAsia') = 24932.20508 ;
evfp('SkLab','a_agricultur','SEAsia') = 5138.71875 ;
evfp('SkLab','a_agricultur','SouthAsia') = 3113.384766 ;
evfp('SkLab','a_agricultur','NAmerica') = 69055.32812 ;
evfp('SkLab','a_agricultur','LatinAmer') = 26548.04102 ;
evfp('SkLab','a_agricultur','EU_28') = 52602.10547 ;
evfp('SkLab','a_agricultur','MENA') = 5222.761719 ;
evfp('SkLab','a_agricultur','SSA') = 18717.58594 ;
evfp('SkLab','a_agricultur','RestofWorld') = 6429.194824 ;
evfp('SkLab','a_Extraction','Oceania') = 11832.22363 ;
evfp('SkLab','a_Extraction','EastAsia') = 12503.1709 ;
evfp('SkLab','a_Extraction','SEAsia') = 8959.34082 ;
evfp('SkLab','a_Extraction','SouthAsia') = 1195.026611 ;
evfp('SkLab','a_Extraction','NAmerica') = 41274.98438 ;
evfp('SkLab','a_Extraction','LatinAmer') = 23594.08789 ;
evfp('SkLab','a_Extraction','EU_28') = 17719.56836 ;
evfp('SkLab','a_Extraction','MENA') = 49752.70312 ;
evfp('SkLab','a_Extraction','SSA') = 11313.2666 ;
evfp('SkLab','a_Extraction','RestofWorld') = 25375.99609 ;
evfp('SkLab','a_ProcFood','Oceania') = 7656.244629 ;
evfp('SkLab','a_ProcFood','EastAsia') = 32135.49219 ;
evfp('SkLab','a_ProcFood','SEAsia') = 8253.868164 ;
evfp('SkLab','a_ProcFood','SouthAsia') = 4091.183838 ;
evfp('SkLab','a_ProcFood','NAmerica') = 56914.11328 ;
evfp('SkLab','a_ProcFood','LatinAmer') = 39433.125 ;
evfp('SkLab','a_ProcFood','EU_28') = 117011.7734 ;
evfp('SkLab','a_ProcFood','MENA') = 8500.331055 ;
evfp('SkLab','a_ProcFood','SSA') = 9165.512695 ;
evfp('SkLab','a_ProcFood','RestofWorld') = 12798.59375 ;
evfp('SkLab','a_TextWapp','Oceania') = 1264.959717 ;
evfp('SkLab','a_TextWapp','EastAsia') = 28954.74023 ;
evfp('SkLab','a_TextWapp','SEAsia') = 3843.487549 ;
evfp('SkLab','a_TextWapp','SouthAsia') = 5204.886719 ;
evfp('SkLab','a_TextWapp','NAmerica') = 31207.04688 ;
evfp('SkLab','a_TextWapp','LatinAmer') = 23281.71484 ;
evfp('SkLab','a_TextWapp','EU_28') = 52076.59766 ;
evfp('SkLab','a_TextWapp','MENA') = 7288.602539 ;
evfp('SkLab','a_TextWapp','SSA') = 3678.726318 ;
evfp('SkLab','a_TextWapp','RestofWorld') = 2957.499756 ;
evfp('SkLab','a_LightMnfc','Oceania') = 15720.24414 ;
evfp('SkLab','a_LightMnfc','EastAsia') = 133419.7969 ;
evfp('SkLab','a_LightMnfc','SEAsia') = 14944.54785 ;
evfp('SkLab','a_LightMnfc','SouthAsia') = 7599.630371 ;
evfp('SkLab','a_LightMnfc','NAmerica') = 284484.2812 ;
evfp('SkLab','a_LightMnfc','LatinAmer') = 75074.10938 ;
evfp('SkLab','a_LightMnfc','EU_28') = 383886.75 ;
evfp('SkLab','a_LightMnfc','MENA') = 14375.00781 ;
evfp('SkLab','a_LightMnfc','SSA') = 11285.47754 ;
evfp('SkLab','a_LightMnfc','RestofWorld') = 33268.48438 ;
evfp('SkLab','a_HeavyMnfc','Oceania') = 16138.05469 ;
evfp('SkLab','a_HeavyMnfc','EastAsia') = 265081.4375 ;
evfp('SkLab','a_HeavyMnfc','SEAsia') = 36167.80859 ;
evfp('SkLab','a_HeavyMnfc','SouthAsia') = 6610.187988 ;
evfp('SkLab','a_HeavyMnfc','NAmerica') = 367859.5938 ;
evfp('SkLab','a_HeavyMnfc','LatinAmer') = 97667.23438 ;
evfp('SkLab','a_HeavyMnfc','EU_28') = 512202.3438 ;
evfp('SkLab','a_HeavyMnfc','MENA') = 24951.60156 ;
evfp('SkLab','a_HeavyMnfc','SSA') = 13445.06152 ;
evfp('SkLab','a_HeavyMnfc','RestofWorld') = 61982.69922 ;
evfp('SkLab','a_Util_Cons','Oceania') = 34316.07812 ;
evfp('SkLab','a_Util_Cons','EastAsia') = 119013.9844 ;
evfp('SkLab','a_Util_Cons','SEAsia') = 23080.49414 ;
evfp('SkLab','a_Util_Cons','SouthAsia') = 51217.30859 ;
evfp('SkLab','a_Util_Cons','NAmerica') = 269586.3125 ;
evfp('SkLab','a_Util_Cons','LatinAmer') = 37809.72656 ;
evfp('SkLab','a_Util_Cons','EU_28') = 214207.2188 ;
evfp('SkLab','a_Util_Cons','MENA') = 42597.47266 ;
evfp('SkLab','a_Util_Cons','SSA') = 11394.46484 ;
evfp('SkLab','a_Util_Cons','RestofWorld') = 66445.67188 ;
evfp('SkLab','a_TransComm','Oceania') = 90390.01562 ;
evfp('SkLab','a_TransComm','EastAsia') = 500691.5625 ;
evfp('SkLab','a_TransComm','SEAsia') = 52541.65234 ;
evfp('SkLab','a_TransComm','SouthAsia') = 68657.67969 ;
evfp('SkLab','a_TransComm','NAmerica') = 929694.5 ;
evfp('SkLab','a_TransComm','LatinAmer') = 184198.6406 ;
evfp('SkLab','a_TransComm','EU_28') = 620070.8125 ;
evfp('SkLab','a_TransComm','MENA') = 80085.03906 ;
evfp('SkLab','a_TransComm','SSA') = 68603.77344 ;
evfp('SkLab','a_TransComm','RestofWorld') = 120391.8281 ;
evfp('SkLab','a_OthService','Oceania') = 312152.875 ;
evfp('SkLab','a_OthService','EastAsia') = 1968111.125 ;
evfp('SkLab','a_OthService','SEAsia') = 186741.8281 ;
evfp('SkLab','a_OthService','SouthAsia') = 288952.4688 ;
evfp('SkLab','a_OthService','NAmerica') = 4867470.5 ;
evfp('SkLab','a_OthService','LatinAmer') = 685661.8125 ;
evfp('SkLab','a_OthService','EU_28') = 3127421.5 ;
evfp('SkLab','a_OthService','MENA') = 317974.7812 ;
evfp('SkLab','a_OthService','SSA') = 162389.3125 ;
evfp('SkLab','a_OthService','RestofWorld') = 569455.8125 ;
evfp('Capital','a_agricultur','Oceania') = 12193.4873 ;
evfp('Capital','a_agricultur','EastAsia') = 101036.4219 ;
evfp('Capital','a_agricultur','SEAsia') = 21983.63867 ;
evfp('Capital','a_agricultur','SouthAsia') = 83467.54688 ;
evfp('Capital','a_agricultur','NAmerica') = 85837.60156 ;
evfp('Capital','a_agricultur','LatinAmer') = 124709.9531 ;
evfp('Capital','a_agricultur','EU_28') = 92279.46875 ;
evfp('Capital','a_agricultur','MENA') = 66857.10156 ;
evfp('Capital','a_agricultur','SSA') = 50010.96875 ;
evfp('Capital','a_agricultur','RestofWorld') = 28714.95508 ;
evfp('Capital','a_Extraction','Oceania') = 76232.51562 ;
evfp('Capital','a_Extraction','EastAsia') = 138199.3125 ;
evfp('Capital','a_Extraction','SEAsia') = 107252.0469 ;
evfp('Capital','a_Extraction','SouthAsia') = 38585.67578 ;
evfp('Capital','a_Extraction','NAmerica') = 179053.0312 ;
evfp('Capital','a_Extraction','LatinAmer') = 134736.4219 ;
evfp('Capital','a_Extraction','EU_28') = 80848.07812 ;
evfp('Capital','a_Extraction','MENA') = 614254.8125 ;
evfp('Capital','a_Extraction','SSA') = 108949.3438 ;
evfp('Capital','a_Extraction','RestofWorld') = 234145.1875 ;
evfp('Capital','a_ProcFood','Oceania') = 13320.31055 ;
evfp('Capital','a_ProcFood','EastAsia') = 148625.6562 ;
evfp('Capital','a_ProcFood','SEAsia') = 46888.39062 ;
evfp('Capital','a_ProcFood','SouthAsia') = 41317.5 ;
evfp('Capital','a_ProcFood','NAmerica') = 150974.9688 ;
evfp('Capital','a_ProcFood','LatinAmer') = 68721.94531 ;
evfp('Capital','a_ProcFood','EU_28') = 185575.2812 ;
evfp('Capital','a_ProcFood','MENA') = 50437.01953 ;
evfp('Capital','a_ProcFood','SSA') = 24185.38281 ;
evfp('Capital','a_ProcFood','RestofWorld') = 36853.85938 ;
evfp('Capital','a_TextWapp','Oceania') = 2432.212158 ;
evfp('Capital','a_TextWapp','EastAsia') = 65430.24609 ;
evfp('Capital','a_TextWapp','SEAsia') = 20386.83008 ;
evfp('Capital','a_TextWapp','SouthAsia') = 21476.41211 ;
evfp('Capital','a_TextWapp','NAmerica') = 25394.08008 ;
evfp('Capital','a_TextWapp','LatinAmer') = 21210.27734 ;
evfp('Capital','a_TextWapp','EU_28') = 46414.78906 ;
evfp('Capital','a_TextWapp','MENA') = 37864.16406 ;
evfp('Capital','a_TextWapp','SSA') = 4787.910156 ;
evfp('Capital','a_TextWapp','RestofWorld') = 4755.617676 ;
evfp('Capital','a_LightMnfc','Oceania') = 22479.94922 ;
evfp('Capital','a_LightMnfc','EastAsia') = 409607.25 ;
evfp('Capital','a_LightMnfc','SEAsia') = 71023.60156 ;
evfp('Capital','a_LightMnfc','SouthAsia') = 41110.66406 ;
evfp('Capital','a_LightMnfc','NAmerica') = 267448.0312 ;
evfp('Capital','a_LightMnfc','LatinAmer') = 90987.46875 ;
evfp('Capital','a_LightMnfc','EU_28') = 372329.4688 ;
evfp('Capital','a_LightMnfc','MENA') = 81744.97656 ;
evfp('Capital','a_LightMnfc','SSA') = 26174.99023 ;
evfp('Capital','a_LightMnfc','RestofWorld') = 28565.25781 ;
evfp('Capital','a_HeavyMnfc','Oceania') = 26552.67383 ;
evfp('Capital','a_HeavyMnfc','EastAsia') = 983697.625 ;
evfp('Capital','a_HeavyMnfc','SEAsia') = 181716.7344 ;
evfp('Capital','a_HeavyMnfc','SouthAsia') = 86395.64844 ;
evfp('Capital','a_HeavyMnfc','NAmerica') = 477292.1875 ;
evfp('Capital','a_HeavyMnfc','LatinAmer') = 177712.6719 ;
evfp('Capital','a_HeavyMnfc','EU_28') = 588845.875 ;
evfp('Capital','a_HeavyMnfc','MENA') = 285737.0625 ;
evfp('Capital','a_HeavyMnfc','SSA') = 46541.74219 ;
evfp('Capital','a_HeavyMnfc','RestofWorld') = 105487.1719 ;
evfp('Capital','a_Util_Cons','Oceania') = 36693.76172 ;
evfp('Capital','a_Util_Cons','EastAsia') = 341264.7188 ;
evfp('Capital','a_Util_Cons','SEAsia') = 74571.9375 ;
evfp('Capital','a_Util_Cons','SouthAsia') = 80072.03906 ;
evfp('Capital','a_Util_Cons','NAmerica') = 379740.375 ;
evfp('Capital','a_Util_Cons','LatinAmer') = 183330.5469 ;
evfp('Capital','a_Util_Cons','EU_28') = 626870.9375 ;
evfp('Capital','a_Util_Cons','MENA') = 164404.8281 ;
evfp('Capital','a_Util_Cons','SSA') = 53780.22266 ;
evfp('Capital','a_Util_Cons','RestofWorld') = 163345.9062 ;
evfp('Capital','a_TransComm','Oceania') = 75617.35156 ;
evfp('Capital','a_TransComm','EastAsia') = 1249030.625 ;
evfp('Capital','a_TransComm','SEAsia') = 239199.7031 ;
evfp('Capital','a_TransComm','SouthAsia') = 292856.625 ;
evfp('Capital','a_TransComm','NAmerica') = 976447.625 ;
evfp('Capital','a_TransComm','LatinAmer') = 392305.4062 ;
evfp('Capital','a_TransComm','EU_28') = 841085.75 ;
evfp('Capital','a_TransComm','MENA') = 351147.9375 ;
evfp('Capital','a_TransComm','SSA') = 98041.46875 ;
evfp('Capital','a_TransComm','RestofWorld') = 475524.25 ;
evfp('Capital','a_OthService','Oceania') = 306661.7812 ;
evfp('Capital','a_OthService','EastAsia') = 2441510.25 ;
evfp('Capital','a_OthService','SEAsia') = 290524.9375 ;
evfp('Capital','a_OthService','SouthAsia') = 255673.1406 ;
evfp('Capital','a_OthService','NAmerica') = 2731330 ;
evfp('Capital','a_OthService','LatinAmer') = 678251.1875 ;
evfp('Capital','a_OthService','EU_28') = 3519144.5 ;
evfp('Capital','a_OthService','MENA') = 491684.3125 ;
evfp('Capital','a_OthService','SSA') = 138829.625 ;
evfp('Capital','a_OthService','RestofWorld') = 404650.5938 ;
evfp('NatRes','a_Extraction','Oceania') = 29771 ;
evfp('NatRes','a_Extraction','EastAsia') = 126198.4062 ;
evfp('NatRes','a_Extraction','SEAsia') = 54282.53516 ;
evfp('NatRes','a_Extraction','SouthAsia') = 21100.40039 ;
evfp('NatRes','a_Extraction','NAmerica') = 125570.6172 ;
evfp('NatRes','a_Extraction','LatinAmer') = 71717.46094 ;
evfp('NatRes','a_Extraction','EU_28') = 47609.75391 ;
evfp('NatRes','a_Extraction','MENA') = 309746.5625 ;
evfp('NatRes','a_Extraction','SSA') = 66057.91406 ;
evfp('NatRes','a_Extraction','RestofWorld') = 169010.1875 ;

* evos data (290 cells)
evos('Land','a_agricultur','Oceania') = 6518.476562 ;
evos('Land','a_agricultur','EastAsia') = 207140.4844 ;
evos('Land','a_agricultur','SEAsia') = 84980.70312 ;
evos('Land','a_agricultur','SouthAsia') = 147646.9688 ;
evos('Land','a_agricultur','NAmerica') = 60761.26172 ;
evos('Land','a_agricultur','LatinAmer') = 53607.46094 ;
evos('Land','a_agricultur','EU_28') = 61089.39453 ;
evos('Land','a_agricultur','MENA') = 18939.2832 ;
evos('Land','a_agricultur','SSA') = 26987.47266 ;
evos('Land','a_agricultur','RestofWorld') = 28383.19336 ;
evos('UnSkLab','a_agricultur','Oceania') = 7405.304199 ;
evos('UnSkLab','a_agricultur','EastAsia') = 347653.7812 ;
evos('UnSkLab','a_agricultur','SEAsia') = 78128.24219 ;
evos('UnSkLab','a_agricultur','SouthAsia') = 140193.25 ;
evos('UnSkLab','a_agricultur','NAmerica') = 49546.60938 ;
evos('UnSkLab','a_agricultur','LatinAmer') = 77142.90625 ;
evos('UnSkLab','a_agricultur','EU_28') = 94513.625 ;
evos('UnSkLab','a_agricultur','MENA') = 83023.17188 ;
evos('UnSkLab','a_agricultur','SSA') = 122202.625 ;
evos('UnSkLab','a_agricultur','RestofWorld') = 39709.60547 ;
evos('UnSkLab','a_Extraction','Oceania') = 6962.502441 ;
evos('UnSkLab','a_Extraction','EastAsia') = 142187.4688 ;
evos('UnSkLab','a_Extraction','SEAsia') = 27172.04883 ;
evos('UnSkLab','a_Extraction','SouthAsia') = 23606.81641 ;
evos('UnSkLab','a_Extraction','NAmerica') = 28957.13477 ;
evos('UnSkLab','a_Extraction','LatinAmer') = 22918.87695 ;
evos('UnSkLab','a_Extraction','EU_28') = 14757.45312 ;
evos('UnSkLab','a_Extraction','MENA') = 24289.59961 ;
evos('UnSkLab','a_Extraction','SSA') = 20292.89062 ;
evos('UnSkLab','a_Extraction','RestofWorld') = 23861.33984 ;
evos('UnSkLab','a_ProcFood','Oceania') = 5356.359375 ;
evos('UnSkLab','a_ProcFood','EastAsia') = 86920.64062 ;
evos('UnSkLab','a_ProcFood','SEAsia') = 23998.7207 ;
evos('UnSkLab','a_ProcFood','SouthAsia') = 20235.55664 ;
evos('UnSkLab','a_ProcFood','NAmerica') = 64827.35156 ;
evos('UnSkLab','a_ProcFood','LatinAmer') = 25062.18359 ;
evos('UnSkLab','a_ProcFood','EU_28') = 58604.69922 ;
evos('UnSkLab','a_ProcFood','MENA') = 14316.36133 ;
evos('UnSkLab','a_ProcFood','SSA') = 8069.681641 ;
evos('UnSkLab','a_ProcFood','RestofWorld') = 13718.50879 ;
evos('UnSkLab','a_TextWapp','Oceania') = 881.8324585 ;
evos('UnSkLab','a_TextWapp','EastAsia') = 101823.6641 ;
evos('UnSkLab','a_TextWapp','SEAsia') = 10342.9502 ;
evos('UnSkLab','a_TextWapp','SouthAsia') = 19569.15234 ;
evos('UnSkLab','a_TextWapp','NAmerica') = 34216.30859 ;
evos('UnSkLab','a_TextWapp','LatinAmer') = 13453.71582 ;
evos('UnSkLab','a_TextWapp','EU_28') = 26901.46484 ;
evos('UnSkLab','a_TextWapp','MENA') = 11069.41406 ;
evos('UnSkLab','a_TextWapp','SSA') = 3079.637939 ;
evos('UnSkLab','a_TextWapp','RestofWorld') = 3150.619873 ;
evos('UnSkLab','a_LightMnfc','Oceania') = 11036.74316 ;
evos('UnSkLab','a_LightMnfc','EastAsia') = 336084.0312 ;
evos('UnSkLab','a_LightMnfc','SEAsia') = 33160.69141 ;
evos('UnSkLab','a_LightMnfc','SouthAsia') = 40061.58984 ;
evos('UnSkLab','a_LightMnfc','NAmerica') = 299272.5625 ;
evos('UnSkLab','a_LightMnfc','LatinAmer') = 40994.54297 ;
evos('UnSkLab','a_LightMnfc','EU_28') = 190001.875 ;
evos('UnSkLab','a_LightMnfc','MENA') = 24989.79492 ;
evos('UnSkLab','a_LightMnfc','SSA') = 8978.943359 ;
evos('UnSkLab','a_LightMnfc','RestofWorld') = 29674.43164 ;
evos('UnSkLab','a_HeavyMnfc','Oceania') = 11326.20801 ;
evos('UnSkLab','a_HeavyMnfc','EastAsia') = 781208.4375 ;
evos('UnSkLab','a_HeavyMnfc','SEAsia') = 56438.17188 ;
evos('UnSkLab','a_HeavyMnfc','SouthAsia') = 34983.71094 ;
evos('UnSkLab','a_HeavyMnfc','NAmerica') = 405765.1562 ;
evos('UnSkLab','a_HeavyMnfc','LatinAmer') = 51697.40625 ;
evos('UnSkLab','a_HeavyMnfc','EU_28') = 255254.0781 ;
evos('UnSkLab','a_HeavyMnfc','MENA') = 44235.26172 ;
evos('UnSkLab','a_HeavyMnfc','SSA') = 12358.37012 ;
evos('UnSkLab','a_HeavyMnfc','RestofWorld') = 59301.78906 ;
evos('UnSkLab','a_Util_Cons','Oceania') = 47888.81641 ;
evos('UnSkLab','a_Util_Cons','EastAsia') = 562372.5625 ;
evos('UnSkLab','a_Util_Cons','SEAsia') = 63625.10547 ;
evos('UnSkLab','a_Util_Cons','SouthAsia') = 87838.03906 ;
evos('UnSkLab','a_Util_Cons','NAmerica') = 524987.3125 ;
evos('UnSkLab','a_Util_Cons','LatinAmer') = 117109.6719 ;
evos('UnSkLab','a_Util_Cons','EU_28') = 252985.6719 ;
evos('UnSkLab','a_Util_Cons','MENA') = 78735.4375 ;
evos('UnSkLab','a_Util_Cons','SSA') = 17258.80859 ;
evos('UnSkLab','a_Util_Cons','RestofWorld') = 84751.625 ;
evos('UnSkLab','a_TransComm','Oceania') = 74719.03906 ;
evos('UnSkLab','a_TransComm','EastAsia') = 832673.6875 ;
evos('UnSkLab','a_TransComm','SEAsia') = 124154.5859 ;
evos('UnSkLab','a_TransComm','SouthAsia') = 145408.8281 ;
evos('UnSkLab','a_TransComm','NAmerica') = 831245.75 ;
evos('UnSkLab','a_TransComm','LatinAmer') = 255678.2031 ;
evos('UnSkLab','a_TransComm','EU_28') = 300010.6875 ;
evos('UnSkLab','a_TransComm','MENA') = 92029.35156 ;
evos('UnSkLab','a_TransComm','SSA') = 51441.89453 ;
evos('UnSkLab','a_TransComm','RestofWorld') = 110457.6172 ;
evos('UnSkLab','a_OthService','Oceania') = 72831.90625 ;
evos('UnSkLab','a_OthService','EastAsia') = 892896.5 ;
evos('UnSkLab','a_OthService','SEAsia') = 77014.48438 ;
evos('UnSkLab','a_OthService','SouthAsia') = 51237.67578 ;
evos('UnSkLab','a_OthService','NAmerica') = 1080900.375 ;
evos('UnSkLab','a_OthService','LatinAmer') = 330155.5 ;
evos('UnSkLab','a_OthService','EU_28') = 721028.1875 ;
evos('UnSkLab','a_OthService','MENA') = 144443.1562 ;
evos('UnSkLab','a_OthService','SSA') = 54329.77734 ;
evos('UnSkLab','a_OthService','RestofWorld') = 101779.1641 ;
evos('SkLab','a_agricultur','Oceania') = 9807.902344 ;
evos('SkLab','a_agricultur','EastAsia') = 24240.06641 ;
evos('SkLab','a_agricultur','SEAsia') = 4794.024902 ;
evos('SkLab','a_agricultur','SouthAsia') = 3003.993164 ;
evos('SkLab','a_agricultur','NAmerica') = 51141.04297 ;
evos('SkLab','a_agricultur','LatinAmer') = 22564.94141 ;
evos('SkLab','a_agricultur','EU_28') = 32489.60156 ;
evos('SkLab','a_agricultur','MENA') = 4464.293945 ;
evos('SkLab','a_agricultur','SSA') = 14127.08105 ;
evos('SkLab','a_agricultur','RestofWorld') = 4067.258545 ;
evos('SkLab','a_Extraction','Oceania') = 8617.610352 ;
evos('SkLab','a_Extraction','EastAsia') = 12022.22363 ;
evos('SkLab','a_Extraction','SEAsia') = 8411.643555 ;
evos('SkLab','a_Extraction','SouthAsia') = 1139.823853 ;
evos('SkLab','a_Extraction','NAmerica') = 28210.13086 ;
evos('SkLab','a_Extraction','LatinAmer') = 20709.24023 ;
evos('SkLab','a_Extraction','EU_28') = 9606.495117 ;
evos('SkLab','a_Extraction','MENA') = 27862.38672 ;
evos('SkLab','a_Extraction','SSA') = 8373.530273 ;
evos('SkLab','a_Extraction','RestofWorld') = 14822.28418 ;
evos('SkLab','a_ProcFood','Oceania') = 5545.244629 ;
evos('SkLab','a_ProcFood','EastAsia') = 26859.26953 ;
evos('SkLab','a_ProcFood','SEAsia') = 7731.380371 ;
evos('SkLab','a_ProcFood','SouthAsia') = 3949.145508 ;
evos('SkLab','a_ProcFood','NAmerica') = 38889.32812 ;
evos('SkLab','a_ProcFood','LatinAmer') = 32956.6875 ;
evos('SkLab','a_ProcFood','EU_28') = 62865.82812 ;
evos('SkLab','a_ProcFood','MENA') = 6642.035645 ;
evos('SkLab','a_ProcFood','SSA') = 7220.539062 ;
evos('SkLab','a_ProcFood','RestofWorld') = 7740.537109 ;
evos('SkLab','a_TextWapp','Oceania') = 914.9506836 ;
evos('SkLab','a_TextWapp','EastAsia') = 25859.40039 ;
evos('SkLab','a_TextWapp','SEAsia') = 3587.465332 ;
evos('SkLab','a_TextWapp','SouthAsia') = 4769.995605 ;
evos('SkLab','a_TextWapp','NAmerica') = 21279.52148 ;
evos('SkLab','a_TextWapp','LatinAmer') = 19168.78711 ;
evos('SkLab','a_TextWapp','EU_28') = 27737.78711 ;
evos('SkLab','a_TextWapp','MENA') = 6165.095703 ;
evos('SkLab','a_TextWapp','SSA') = 2847.173584 ;
evos('SkLab','a_TextWapp','RestofWorld') = 1812.003662 ;
evos('SkLab','a_LightMnfc','Oceania') = 11401.0127 ;
evos('SkLab','a_LightMnfc','EastAsia') = 111726.8203 ;
evos('SkLab','a_LightMnfc','SEAsia') = 14233.58594 ;
evos('SkLab','a_LightMnfc','SouthAsia') = 7333.580566 ;
evos('SkLab','a_LightMnfc','NAmerica') = 193166.125 ;
evos('SkLab','a_LightMnfc','LatinAmer') = 61424.33984 ;
evos('SkLab','a_LightMnfc','EU_28') = 202063.0781 ;
evos('SkLab','a_LightMnfc','MENA') = 10096.31641 ;
evos('SkLab','a_LightMnfc','SSA') = 8729.951172 ;
evos('SkLab','a_LightMnfc','RestofWorld') = 20028.625 ;
evos('SkLab','a_HeavyMnfc','Oceania') = 11702.45996 ;
evos('SkLab','a_HeavyMnfc','EastAsia') = 227663.0156 ;
evos('SkLab','a_HeavyMnfc','SEAsia') = 34698.08594 ;
evos('SkLab','a_HeavyMnfc','SouthAsia') = 6406.160156 ;
evos('SkLab','a_HeavyMnfc','NAmerica') = 250901.0469 ;
evos('SkLab','a_HeavyMnfc','LatinAmer') = 80315.79688 ;
evos('SkLab','a_HeavyMnfc','EU_28') = 267882.1875 ;
evos('SkLab','a_HeavyMnfc','MENA') = 17413.40625 ;
evos('SkLab','a_HeavyMnfc','SSA') = 10571.50586 ;
evos('SkLab','a_HeavyMnfc','RestofWorld') = 37799.33984 ;
evos('SkLab','a_Util_Cons','Oceania') = 24956.64453 ;
evos('SkLab','a_Util_Cons','EastAsia') = 101196.9609 ;
evos('SkLab','a_Util_Cons','SEAsia') = 22083.3125 ;
evos('SkLab','a_Util_Cons','SouthAsia') = 49592.83984 ;
evos('SkLab','a_Util_Cons','NAmerica') = 183758.2344 ;
evos('SkLab','a_Util_Cons','LatinAmer') = 32871.30859 ;
evos('SkLab','a_Util_Cons','EU_28') = 114763.5938 ;
evos('SkLab','a_Util_Cons','MENA') = 29170.05859 ;
evos('SkLab','a_Util_Cons','SSA') = 9352.858398 ;
evos('SkLab','a_Util_Cons','RestofWorld') = 40724.32422 ;
evos('SkLab','a_TransComm','Oceania') = 65629.51562 ;
evos('SkLab','a_TransComm','EastAsia') = 397131.0312 ;
evos('SkLab','a_TransComm','SEAsia') = 50032.92969 ;
evos('SkLab','a_TransComm','SouthAsia') = 65950.66406 ;
evos('SkLab','a_TransComm','NAmerica') = 633632.6875 ;
evos('SkLab','a_TransComm','LatinAmer') = 150359.6094 ;
evos('SkLab','a_TransComm','EU_28') = 340023.0625 ;
evos('SkLab','a_TransComm','MENA') = 62769.73828 ;
evos('SkLab','a_TransComm','SSA') = 52448.41406 ;
evos('SkLab','a_TransComm','RestofWorld') = 74895.67969 ;
evos('SkLab','a_OthService','Oceania') = 226631.9531 ;
evos('SkLab','a_OthService','EastAsia') = 1645114 ;
evos('SkLab','a_OthService','SEAsia') = 177548.0938 ;
evos('SkLab','a_OthService','SouthAsia') = 278791.6562 ;
evos('SkLab','a_OthService','NAmerica') = 3338910.5 ;
evos('SkLab','a_OthService','LatinAmer') = 569561.875 ;
evos('SkLab','a_OthService','EU_28') = 1662054 ;
evos('SkLab','a_OthService','MENA') = 238715.7188 ;
evos('SkLab','a_OthService','SSA') = 125205.1172 ;
evos('SkLab','a_OthService','RestofWorld') = 347116.0312 ;
evos('Capital','a_agricultur','Oceania') = 10743.9043 ;
evos('Capital','a_agricultur','EastAsia') = 107976.9922 ;
evos('Capital','a_agricultur','SEAsia') = 20295.73828 ;
evos('Capital','a_agricultur','SouthAsia') = 80103.08594 ;
evos('Capital','a_agricultur','NAmerica') = 80307.40625 ;
evos('Capital','a_agricultur','LatinAmer') = 112862.7969 ;
evos('Capital','a_agricultur','EU_28') = 107519.0859 ;
evos('Capital','a_agricultur','MENA') = 62247.02344 ;
evos('Capital','a_agricultur','SSA') = 46171.74219 ;
evos('Capital','a_agricultur','RestofWorld') = 29549.84375 ;
evos('Capital','a_Extraction','Oceania') = 65422.34766 ;
evos('Capital','a_Extraction','EastAsia') = 135125.9375 ;
evos('Capital','a_Extraction','SEAsia') = 97046.08594 ;
evos('Capital','a_Extraction','SouthAsia') = 36988.14453 ;
evos('Capital','a_Extraction','NAmerica') = 159372.2969 ;
evos('Capital','a_Extraction','LatinAmer') = 122994.6719 ;
evos('Capital','a_Extraction','EU_28') = 71289.35938 ;
evos('Capital','a_Extraction','MENA') = 570064.875 ;
evos('Capital','a_Extraction','SSA') = 100804.3906 ;
evos('Capital','a_Extraction','RestofWorld') = 216994.0781 ;
evos('Capital','a_ProcFood','Oceania') = 11475.30469 ;
evos('Capital','a_ProcFood','EastAsia') = 140301.0625 ;
evos('Capital','a_ProcFood','SEAsia') = 42938.57031 ;
evos('Capital','a_ProcFood','SouthAsia') = 39624.72266 ;
evos('Capital','a_ProcFood','NAmerica') = 135046.9062 ;
evos('Capital','a_ProcFood','LatinAmer') = 63099.52734 ;
evos('Capital','a_ProcFood','EU_28') = 168797.4375 ;
evos('Capital','a_ProcFood','MENA') = 46708.55469 ;
evos('Capital','a_ProcFood','SSA') = 22256.09961 ;
evos('Capital','a_ProcFood','RestofWorld') = 34112.88672 ;
evos('Capital','a_TextWapp','Oceania') = 2091.259033 ;
evos('Capital','a_TextWapp','EastAsia') = 63858.76172 ;
evos('Capital','a_TextWapp','SEAsia') = 18726.10352 ;
evos('Capital','a_TextWapp','SouthAsia') = 20514.10742 ;
evos('Capital','a_TextWapp','NAmerica') = 22683.83594 ;
evos('Capital','a_TextWapp','LatinAmer') = 19300.72266 ;
evos('Capital','a_TextWapp','EU_28') = 42291.08203 ;
evos('Capital','a_TextWapp','MENA') = 34769.82031 ;
evos('Capital','a_TextWapp','SSA') = 4433.695801 ;
evos('Capital','a_TextWapp','RestofWorld') = 4397.222656 ;
evos('Capital','a_LightMnfc','Oceania') = 19332 ;
evos('Capital','a_LightMnfc','EastAsia') = 394020.4062 ;
evos('Capital','a_LightMnfc','SEAsia') = 65060.75 ;
evos('Capital','a_LightMnfc','SouthAsia') = 39527.95312 ;
evos('Capital','a_LightMnfc','NAmerica') = 238746.5 ;
evos('Capital','a_LightMnfc','LatinAmer') = 81966 ;
evos('Capital','a_LightMnfc','EU_28') = 340728.3438 ;
evos('Capital','a_LightMnfc','MENA') = 75530.84375 ;
evos('Capital','a_LightMnfc','SSA') = 23817.56055 ;
evos('Capital','a_LightMnfc','RestofWorld') = 26413.44922 ;
evos('Capital','a_HeavyMnfc','Oceania') = 22870.02539 ;
evos('Capital','a_HeavyMnfc','EastAsia') = 944919.8125 ;
evos('Capital','a_HeavyMnfc','SEAsia') = 169219.8281 ;
evos('Capital','a_HeavyMnfc','SouthAsia') = 83148.79688 ;
evos('Capital','a_HeavyMnfc','NAmerica') = 426251.0938 ;
evos('Capital','a_HeavyMnfc','LatinAmer') = 161105.1719 ;
evos('Capital','a_HeavyMnfc','EU_28') = 540841.4375 ;
evos('Capital','a_HeavyMnfc','MENA') = 268621.3125 ;
evos('Capital','a_HeavyMnfc','SSA') = 42012.125 ;
evos('Capital','a_HeavyMnfc','RestofWorld') = 97685.24219 ;
evos('Capital','a_Util_Cons','Oceania') = 31616.39258 ;
evos('Capital','a_Util_Cons','EastAsia') = 323510.7812 ;
evos('Capital','a_Util_Cons','SEAsia') = 68134.70312 ;
evos('Capital','a_Util_Cons','SouthAsia') = 76537.125 ;
evos('Capital','a_Util_Cons','NAmerica') = 338628.6562 ;
evos('Capital','a_Util_Cons','LatinAmer') = 164990.4062 ;
evos('Capital','a_Util_Cons','EU_28') = 568340.875 ;
evos('Capital','a_Util_Cons','MENA') = 150813.0625 ;
evos('Capital','a_Util_Cons','SSA') = 49551.98438 ;
evos('Capital','a_Util_Cons','RestofWorld') = 151181.9844 ;
evos('Capital','a_TransComm','Oceania') = 65113.49219 ;
evos('Capital','a_TransComm','EastAsia') = 1172947.625 ;
evos('Capital','a_TransComm','SEAsia') = 222586.6562 ;
evos('Capital','a_TransComm','SouthAsia') = 281407.6562 ;
evos('Capital','a_TransComm','NAmerica') = 872229.4375 ;
evos('Capital','a_TransComm','LatinAmer') = 354594.875 ;
evos('Capital','a_TransComm','EU_28') = 763340.75 ;
evos('Capital','a_TransComm','MENA') = 323718.4062 ;
evos('Capital','a_TransComm','SSA') = 89293.72656 ;
evos('Capital','a_TransComm','RestofWorld') = 440554.0625 ;
evos('Capital','a_OthService','Oceania') = 263977 ;
evos('Capital','a_OthService','EastAsia') = 2250839 ;
evos('Capital','a_OthService','SEAsia') = 267814.9375 ;
evos('Capital','a_OthService','SouthAsia') = 245692.3594 ;
evos('Capital','a_OthService','NAmerica') = 2434297 ;
evos('Capital','a_OthService','LatinAmer') = 610065.5625 ;
evos('Capital','a_OthService','EU_28') = 3210643.75 ;
evos('Capital','a_OthService','MENA') = 453665.625 ;
evos('Capital','a_OthService','SSA') = 125552.0312 ;
evos('Capital','a_OthService','RestofWorld') = 374522.2188 ;
evos('NatRes','a_Extraction','Oceania') = 25563.0918 ;
evos('NatRes','a_Extraction','EastAsia') = 123809.6484 ;
evos('NatRes','a_Extraction','SEAsia') = 49055.55078 ;
evos('NatRes','a_Extraction','SouthAsia') = 20226.23633 ;
evos('NatRes','a_Extraction','NAmerica') = 111702.7344 ;
evos('NatRes','a_Extraction','LatinAmer') = 65177.91406 ;
evos('NatRes','a_Extraction','EU_28') = 42433.12109 ;
evos('NatRes','a_Extraction','MENA') = 287531.5938 ;
evos('NatRes','a_Extraction','SSA') = 61136.97266 ;
evos('NatRes','a_Extraction','RestofWorld') = 156499.5312 ;

* vdpb data (100 cells)
vdpb('c_Crops','Oceania') = 4871.338867 ;
vdpb('c_Crops','EastAsia') = 156118.125 ;
vdpb('c_Crops','SEAsia') = 91202.40625 ;
vdpb('c_Crops','SouthAsia') = 219131.8281 ;
vdpb('c_Crops','NAmerica') = 57875.55078 ;
vdpb('c_Crops','LatinAmer') = 70532.30469 ;
vdpb('c_Crops','EU_28') = 59780.69141 ;
vdpb('c_Crops','MENA') = 119480.7891 ;
vdpb('c_Crops','SSA') = 198857.0469 ;
vdpb('c_Crops','RestofWorld') = 47956.82422 ;
vdpb('c_MeatLstk','Oceania') = 6869.920898 ;
vdpb('c_MeatLstk','EastAsia') = 180799.4062 ;
vdpb('c_MeatLstk','SEAsia') = 37057.93359 ;
vdpb('c_MeatLstk','SouthAsia') = 98048.80469 ;
vdpb('c_MeatLstk','NAmerica') = 167819.875 ;
vdpb('c_MeatLstk','LatinAmer') = 110546.5547 ;
vdpb('c_MeatLstk','EU_28') = 170576.2188 ;
vdpb('c_MeatLstk','MENA') = 69473.67188 ;
vdpb('c_MeatLstk','SSA') = 67272.77344 ;
vdpb('c_MeatLstk','RestofWorld') = 94014.86719 ;
vdpb('c_Extraction','Oceania') = 1096.581055 ;
vdpb('c_Extraction','EastAsia') = 46991.74609 ;
vdpb('c_Extraction','SEAsia') = 25876.46484 ;
vdpb('c_Extraction','SouthAsia') = 36835.92188 ;
vdpb('c_Extraction','NAmerica') = 12816.68848 ;
vdpb('c_Extraction','LatinAmer') = 11986.63965 ;
vdpb('c_Extraction','EU_28') = 18446.94922 ;
vdpb('c_Extraction','MENA') = 11495.99609 ;
vdpb('c_Extraction','SSA') = 20842.58203 ;
vdpb('c_Extraction','RestofWorld') = 5814.928711 ;
vdpb('c_ProcFood','Oceania') = 40129.44922 ;
vdpb('c_ProcFood','EastAsia') = 660476.1875 ;
vdpb('c_ProcFood','SEAsia') = 115191.6641 ;
vdpb('c_ProcFood','SouthAsia') = 208389.125 ;
vdpb('c_ProcFood','NAmerica') = 502988.4062 ;
vdpb('c_ProcFood','LatinAmer') = 248254.75 ;
vdpb('c_ProcFood','EU_28') = 655506.75 ;
vdpb('c_ProcFood','MENA') = 165165.6719 ;
vdpb('c_ProcFood','SSA') = 90714.04688 ;
vdpb('c_ProcFood','RestofWorld') = 140457.7656 ;
vdpb('c_TextWapp','Oceania') = 4578.036621 ;
vdpb('c_TextWapp','EastAsia') = 145116.25 ;
vdpb('c_TextWapp','SEAsia') = 27748.56641 ;
vdpb('c_TextWapp','SouthAsia') = 78834.83594 ;
vdpb('c_TextWapp','NAmerica') = 151514.0625 ;
vdpb('c_TextWapp','LatinAmer') = 81179.35156 ;
vdpb('c_TextWapp','EU_28') = 190421 ;
vdpb('c_TextWapp','MENA') = 68912.75 ;
vdpb('c_TextWapp','SSA') = 23637.17383 ;
vdpb('c_TextWapp','RestofWorld') = 14508.74316 ;
vdpb('c_LightMnfc','Oceania') = 22968.05273 ;
vdpb('c_LightMnfc','EastAsia') = 287222.625 ;
vdpb('c_LightMnfc','SEAsia') = 64970.91406 ;
vdpb('c_LightMnfc','SouthAsia') = 55997.82812 ;
vdpb('c_LightMnfc','NAmerica') = 467371.0312 ;
vdpb('c_LightMnfc','LatinAmer') = 157073.7656 ;
vdpb('c_LightMnfc','EU_28') = 587115.375 ;
vdpb('c_LightMnfc','MENA') = 71995.84375 ;
vdpb('c_LightMnfc','SSA') = 36404.78906 ;
vdpb('c_LightMnfc','RestofWorld') = 50067.21875 ;
vdpb('c_HeavyMnfc','Oceania') = 18531.42578 ;
vdpb('c_HeavyMnfc','EastAsia') = 390011.7812 ;
vdpb('c_HeavyMnfc','SEAsia') = 77900.42188 ;
vdpb('c_HeavyMnfc','SouthAsia') = 80203.57031 ;
vdpb('c_HeavyMnfc','NAmerica') = 556723.625 ;
vdpb('c_HeavyMnfc','LatinAmer') = 158935.2031 ;
vdpb('c_HeavyMnfc','EU_28') = 439712.75 ;
vdpb('c_HeavyMnfc','MENA') = 131250.875 ;
vdpb('c_HeavyMnfc','SSA') = 37246.16016 ;
vdpb('c_HeavyMnfc','RestofWorld') = 73597.42188 ;
vdpb('c_Util_Cons','Oceania') = 19507.875 ;
vdpb('c_Util_Cons','EastAsia') = 174421.5625 ;
vdpb('c_Util_Cons','SEAsia') = 32035.45117 ;
vdpb('c_Util_Cons','SouthAsia') = 40498.22656 ;
vdpb('c_Util_Cons','NAmerica') = 235843.1094 ;
vdpb('c_Util_Cons','LatinAmer') = 51893.3125 ;
vdpb('c_Util_Cons','EU_28') = 271720.4062 ;
vdpb('c_Util_Cons','MENA') = 99037.82812 ;
vdpb('c_Util_Cons','SSA') = 13866.83594 ;
vdpb('c_Util_Cons','RestofWorld') = 122540.9453 ;
vdpb('c_TransComm','Oceania') = 230843.6406 ;
vdpb('c_TransComm','EastAsia') = 1690568.75 ;
vdpb('c_TransComm','SEAsia') = 266874.4375 ;
vdpb('c_TransComm','SouthAsia') = 315827.6562 ;
vdpb('c_TransComm','NAmerica') = 2614043.25 ;
vdpb('c_TransComm','LatinAmer') = 720397.5625 ;
vdpb('c_TransComm','EU_28') = 2214110.25 ;
vdpb('c_TransComm','MENA') = 421047.3125 ;
vdpb('c_TransComm','SSA') = 86257.32812 ;
vdpb('c_TransComm','RestofWorld') = 553347.4375 ;
vdpb('c_OthService','Oceania') = 375498.1562 ;
vdpb('c_OthService','EastAsia') = 2641395.5 ;
vdpb('c_OthService','SEAsia') = 304547.5312 ;
vdpb('c_OthService','SouthAsia') = 263922.375 ;
vdpb('c_OthService','NAmerica') = 6625411 ;
vdpb('c_OthService','LatinAmer') = 922883.875 ;
vdpb('c_OthService','EU_28') = 3020722 ;
vdpb('c_OthService','MENA') = 476343.0625 ;
vdpb('c_OthService','SSA') = 179079.7344 ;
vdpb('c_OthService','RestofWorld') = 451319.7188 ;

* vdpp data (100 cells)
vdpp('c_Crops','Oceania') = 5104.905762 ;
vdpp('c_Crops','EastAsia') = 159400.4844 ;
vdpp('c_Crops','SEAsia') = 92378.47656 ;
vdpp('c_Crops','SouthAsia') = 212019.1094 ;
vdpp('c_Crops','NAmerica') = 60070.32812 ;
vdpp('c_Crops','LatinAmer') = 76472.82031 ;
vdpp('c_Crops','EU_28') = 60780.91797 ;
vdpp('c_Crops','MENA') = 120983.6562 ;
vdpp('c_Crops','SSA') = 199591.125 ;
vdpp('c_Crops','RestofWorld') = 49540.56641 ;
vdpp('c_MeatLstk','Oceania') = 7164.399902 ;
vdpp('c_MeatLstk','EastAsia') = 185283.3438 ;
vdpp('c_MeatLstk','SEAsia') = 37656.92188 ;
vdpp('c_MeatLstk','SouthAsia') = 98370.3125 ;
vdpp('c_MeatLstk','NAmerica') = 172052.2031 ;
vdpp('c_MeatLstk','LatinAmer') = 118723.6016 ;
vdpp('c_MeatLstk','EU_28') = 190410.7812 ;
vdpp('c_MeatLstk','MENA') = 71469.29688 ;
vdpp('c_MeatLstk','SSA') = 68719.04688 ;
vdpp('c_MeatLstk','RestofWorld') = 111908.3203 ;
vdpp('c_Extraction','Oceania') = 1195.068481 ;
vdpp('c_Extraction','EastAsia') = 48245.24219 ;
vdpp('c_Extraction','SEAsia') = 26453.06445 ;
vdpp('c_Extraction','SouthAsia') = 36628.58594 ;
vdpp('c_Extraction','NAmerica') = 13283.69238 ;
vdpp('c_Extraction','LatinAmer') = 12300.54004 ;
vdpp('c_Extraction','EU_28') = 21043.87305 ;
vdpp('c_Extraction','MENA') = 11934.91406 ;
vdpp('c_Extraction','SSA') = 21516.85742 ;
vdpp('c_Extraction','RestofWorld') = 6151.283691 ;
vdpp('c_ProcFood','Oceania') = 52302.07031 ;
vdpp('c_ProcFood','EastAsia') = 725297.375 ;
vdpp('c_ProcFood','SEAsia') = 130557.7266 ;
vdpp('c_ProcFood','SouthAsia') = 217412.375 ;
vdpp('c_ProcFood','NAmerica') = 576211.125 ;
vdpp('c_ProcFood','LatinAmer') = 290982.125 ;
vdpp('c_ProcFood','EU_28') = 774398.9375 ;
vdpp('c_ProcFood','MENA') = 180600.6406 ;
vdpp('c_ProcFood','SSA') = 99701.80469 ;
vdpp('c_ProcFood','RestofWorld') = 174064.9375 ;
vdpp('c_TextWapp','Oceania') = 6123.921387 ;
vdpp('c_TextWapp','EastAsia') = 152450.7656 ;
vdpp('c_TextWapp','SEAsia') = 28398.37891 ;
vdpp('c_TextWapp','SouthAsia') = 80206 ;
vdpp('c_TextWapp','NAmerica') = 168646.75 ;
vdpp('c_TextWapp','LatinAmer') = 89773.625 ;
vdpp('c_TextWapp','EU_28') = 220423.5 ;
vdpp('c_TextWapp','MENA') = 75732.71875 ;
vdpp('c_TextWapp','SSA') = 24916.03516 ;
vdpp('c_TextWapp','RestofWorld') = 15596.79883 ;
vdpp('c_LightMnfc','Oceania') = 26441.74414 ;
vdpp('c_LightMnfc','EastAsia') = 306858 ;
vdpp('c_LightMnfc','SEAsia') = 68683.65625 ;
vdpp('c_LightMnfc','SouthAsia') = 57652.01562 ;
vdpp('c_LightMnfc','NAmerica') = 503449.5625 ;
vdpp('c_LightMnfc','LatinAmer') = 179908.5938 ;
vdpp('c_LightMnfc','EU_28') = 682627.5 ;
vdpp('c_LightMnfc','MENA') = 76765.30469 ;
vdpp('c_LightMnfc','SSA') = 37935.03516 ;
vdpp('c_LightMnfc','RestofWorld') = 59547.96875 ;
vdpp('c_HeavyMnfc','Oceania') = 27089.98047 ;
vdpp('c_HeavyMnfc','EastAsia') = 467307.75 ;
vdpp('c_HeavyMnfc','SEAsia') = 82026.74219 ;
vdpp('c_HeavyMnfc','SouthAsia') = 100210.2422 ;
vdpp('c_HeavyMnfc','NAmerica') = 635846.875 ;
vdpp('c_HeavyMnfc','LatinAmer') = 203339.75 ;
vdpp('c_HeavyMnfc','EU_28') = 640293.375 ;
vdpp('c_HeavyMnfc','MENA') = 143054.7969 ;
vdpp('c_HeavyMnfc','SSA') = 41813.72266 ;
vdpp('c_HeavyMnfc','RestofWorld') = 93313.47656 ;
vdpp('c_Util_Cons','Oceania') = 20545.19727 ;
vdpp('c_Util_Cons','EastAsia') = 187415.3906 ;
vdpp('c_Util_Cons','SEAsia') = 33690.85938 ;
vdpp('c_Util_Cons','SouthAsia') = 36988.92969 ;
vdpp('c_Util_Cons','NAmerica') = 243238.5156 ;
vdpp('c_Util_Cons','LatinAmer') = 61426.99609 ;
vdpp('c_Util_Cons','EU_28') = 382174.75 ;
vdpp('c_Util_Cons','MENA') = 102367.0703 ;
vdpp('c_Util_Cons','SSA') = 14197.32715 ;
vdpp('c_Util_Cons','RestofWorld') = 128646.8984 ;
vdpp('c_TransComm','Oceania') = 243118.3281 ;
vdpp('c_TransComm','EastAsia') = 1829925.375 ;
vdpp('c_TransComm','SEAsia') = 279364.625 ;
vdpp('c_TransComm','SouthAsia') = 315257.6562 ;
vdpp('c_TransComm','NAmerica') = 2648066.75 ;
vdpp('c_TransComm','LatinAmer') = 760559 ;
vdpp('c_TransComm','EU_28') = 2260198 ;
vdpp('c_TransComm','MENA') = 435356.0938 ;
vdpp('c_TransComm','SSA') = 87646.24219 ;
vdpp('c_TransComm','RestofWorld') = 567040.8125 ;
vdpp('c_OthService','Oceania') = 391831.4688 ;
vdpp('c_OthService','EastAsia') = 2731845.25 ;
vdpp('c_OthService','SEAsia') = 315203.1875 ;
vdpp('c_OthService','SouthAsia') = 266161.3438 ;
vdpp('c_OthService','NAmerica') = 6677009 ;
vdpp('c_OthService','LatinAmer') = 952885.625 ;
vdpp('c_OthService','EU_28') = 3328315.75 ;
vdpp('c_OthService','MENA') = 488042.4375 ;
vdpp('c_OthService','SSA') = 185211.5781 ;
vdpp('c_OthService','RestofWorld') = 450371.3438 ;

* vmpb data (100 cells)
vmpb('c_Crops','Oceania') = 955.3606567 ;
vmpb('c_Crops','EastAsia') = 15030.08496 ;
vmpb('c_Crops','SEAsia') = 11506.87012 ;
vmpb('c_Crops','SouthAsia') = 8330.5 ;
vmpb('c_Crops','NAmerica') = 25151.43164 ;
vmpb('c_Crops','LatinAmer') = 6798.863281 ;
vmpb('c_Crops','EU_28') = 57466.83203 ;
vmpb('c_Crops','MENA') = 29800.6543 ;
vmpb('c_Crops','SSA') = 14236.47656 ;
vmpb('c_Crops','RestofWorld') = 7952.050781 ;
vmpb('c_MeatLstk','Oceania') = 1129.153198 ;
vmpb('c_MeatLstk','EastAsia') = 9630.585938 ;
vmpb('c_MeatLstk','SEAsia') = 3365.401367 ;
vmpb('c_MeatLstk','SouthAsia') = 571.3659668 ;
vmpb('c_MeatLstk','NAmerica') = 9222.40918 ;
vmpb('c_MeatLstk','LatinAmer') = 3965.386719 ;
vmpb('c_MeatLstk','EU_28') = 50358.23438 ;
vmpb('c_MeatLstk','MENA') = 7720.400879 ;
vmpb('c_MeatLstk','SSA') = 3418.243164 ;
vmpb('c_MeatLstk','RestofWorld') = 11890.56641 ;
vmpb('c_Extraction','Oceania') = 72.91025543 ;
vmpb('c_Extraction','EastAsia') = 6465.561523 ;
vmpb('c_Extraction','SEAsia') = 637.031189 ;
vmpb('c_Extraction','SouthAsia') = 1685.953857 ;
vmpb('c_Extraction','NAmerica') = 804.2432861 ;
vmpb('c_Extraction','LatinAmer') = 376.5127563 ;
vmpb('c_Extraction','EU_28') = 18075.58594 ;
vmpb('c_Extraction','MENA') = 1756.3302 ;
vmpb('c_Extraction','SSA') = 635.0917358 ;
vmpb('c_Extraction','RestofWorld') = 5476.001953 ;
vmpb('c_ProcFood','Oceania') = 8482.163086 ;
vmpb('c_ProcFood','EastAsia') = 55411.75 ;
vmpb('c_ProcFood','SEAsia') = 26600.54883 ;
vmpb('c_ProcFood','SouthAsia') = 22854.93945 ;
vmpb('c_ProcFood','NAmerica') = 70086.25 ;
vmpb('c_ProcFood','LatinAmer') = 25407.66406 ;
vmpb('c_ProcFood','EU_28') = 173726.9531 ;
vmpb('c_ProcFood','MENA') = 45213.89062 ;
vmpb('c_ProcFood','SSA') = 29786.20508 ;
vmpb('c_ProcFood','RestofWorld') = 32455.8418 ;
vmpb('c_TextWapp','Oceania') = 6146.778809 ;
vmpb('c_TextWapp','EastAsia') = 47385.5625 ;
vmpb('c_TextWapp','SEAsia') = 9500.305664 ;
vmpb('c_TextWapp','SouthAsia') = 8671.655273 ;
vmpb('c_TextWapp','NAmerica') = 105253.1484 ;
vmpb('c_TextWapp','LatinAmer') = 18669.35156 ;
vmpb('c_TextWapp','EU_28') = 180360.5469 ;
vmpb('c_TextWapp','MENA') = 31322.25195 ;
vmpb('c_TextWapp','SSA') = 12245.10059 ;
vmpb('c_TextWapp','RestofWorld') = 33250.92578 ;
vmpb('c_LightMnfc','Oceania') = 24753.49219 ;
vmpb('c_LightMnfc','EastAsia') = 61423.95312 ;
vmpb('c_LightMnfc','SEAsia') = 22180.55078 ;
vmpb('c_LightMnfc','SouthAsia') = 8840.245117 ;
vmpb('c_LightMnfc','NAmerica') = 274207.5312 ;
vmpb('c_LightMnfc','LatinAmer') = 50438.42578 ;
vmpb('c_LightMnfc','EU_28') = 405058.7812 ;
vmpb('c_LightMnfc','MENA') = 77520.02344 ;
vmpb('c_LightMnfc','SSA') = 29229.76172 ;
vmpb('c_LightMnfc','RestofWorld') = 73883.89844 ;
vmpb('c_HeavyMnfc','Oceania') = 16790.41406 ;
vmpb('c_HeavyMnfc','EastAsia') = 116013.2344 ;
vmpb('c_HeavyMnfc','SEAsia') = 48744.17969 ;
vmpb('c_HeavyMnfc','SouthAsia') = 23699.03906 ;
vmpb('c_HeavyMnfc','NAmerica') = 257847.5938 ;
vmpb('c_HeavyMnfc','LatinAmer') = 63808.81641 ;
vmpb('c_HeavyMnfc','EU_28') = 366610.7188 ;
vmpb('c_HeavyMnfc','MENA') = 96047.90625 ;
vmpb('c_HeavyMnfc','SSA') = 39612.78516 ;
vmpb('c_HeavyMnfc','RestofWorld') = 67542.50781 ;
vmpb('c_Util_Cons','Oceania') = 726.5334473 ;
vmpb('c_Util_Cons','EastAsia') = 1239.09314 ;
vmpb('c_Util_Cons','SEAsia') = 1400.714844 ;
vmpb('c_Util_Cons','SouthAsia') = 137.5133057 ;
vmpb('c_Util_Cons','NAmerica') = 2626.999512 ;
vmpb('c_Util_Cons','LatinAmer') = 492.6765442 ;
vmpb('c_Util_Cons','EU_28') = 16456.95898 ;
vmpb('c_Util_Cons','MENA') = 3653.73999 ;
vmpb('c_Util_Cons','SSA') = 432.2212524 ;
vmpb('c_Util_Cons','RestofWorld') = 2989.920166 ;
vmpb('c_TransComm','Oceania') = 13736.13574 ;
vmpb('c_TransComm','EastAsia') = 78733.85938 ;
vmpb('c_TransComm','SEAsia') = 22652.69922 ;
vmpb('c_TransComm','SouthAsia') = 15384.10449 ;
vmpb('c_TransComm','NAmerica') = 51627.73828 ;
vmpb('c_TransComm','LatinAmer') = 21721.85742 ;
vmpb('c_TransComm','EU_28') = 244598.5312 ;
vmpb('c_TransComm','MENA') = 30340.6582 ;
vmpb('c_TransComm','SSA') = 10117.9375 ;
vmpb('c_TransComm','RestofWorld') = 28829.85156 ;
vmpb('c_OthService','Oceania') = 6727.678223 ;
vmpb('c_OthService','EastAsia') = 27705.52148 ;
vmpb('c_OthService','SEAsia') = 22956.70508 ;
vmpb('c_OthService','SouthAsia') = 5929.82959 ;
vmpb('c_OthService','NAmerica') = 58900.69531 ;
vmpb('c_OthService','LatinAmer') = 10285.39941 ;
vmpb('c_OthService','EU_28') = 197759.3125 ;
vmpb('c_OthService','MENA') = 19277.41797 ;
vmpb('c_OthService','SSA') = 8559.955078 ;
vmpb('c_OthService','RestofWorld') = 25653.88867 ;

* vmpp data (100 cells)
vmpp('c_Crops','Oceania') = 1005.742188 ;
vmpp('c_Crops','EastAsia') = 16918.70508 ;
vmpp('c_Crops','SEAsia') = 12035.80371 ;
vmpp('c_Crops','SouthAsia') = 8799.610352 ;
vmpp('c_Crops','NAmerica') = 26180.87305 ;
vmpp('c_Crops','LatinAmer') = 7046.174805 ;
vmpp('c_Crops','EU_28') = 57119.69922 ;
vmpp('c_Crops','MENA') = 29181.90039 ;
vmpp('c_Crops','SSA') = 14885.14551 ;
vmpp('c_Crops','RestofWorld') = 8215.472656 ;
vmpp('c_MeatLstk','Oceania') = 1207.344604 ;
vmpp('c_MeatLstk','EastAsia') = 10706.38379 ;
vmpp('c_MeatLstk','SEAsia') = 3529.508057 ;
vmpp('c_MeatLstk','SouthAsia') = 596.9522705 ;
vmpp('c_MeatLstk','NAmerica') = 9495.174805 ;
vmpp('c_MeatLstk','LatinAmer') = 4317.251465 ;
vmpp('c_MeatLstk','EU_28') = 57904.81641 ;
vmpp('c_MeatLstk','MENA') = 7835.993164 ;
vmpp('c_MeatLstk','SSA') = 3581.955566 ;
vmpp('c_MeatLstk','RestofWorld') = 14823.55078 ;
vmpp('c_Extraction','Oceania') = 76.97658539 ;
vmpp('c_Extraction','EastAsia') = 7813.495605 ;
vmpp('c_Extraction','SEAsia') = 663.546814 ;
vmpp('c_Extraction','SouthAsia') = 1808.842651 ;
vmpp('c_Extraction','NAmerica') = 827.7698364 ;
vmpp('c_Extraction','LatinAmer') = 384.4559021 ;
vmpp('c_Extraction','EU_28') = 26736.94727 ;
vmpp('c_Extraction','MENA') = 2007.77771 ;
vmpp('c_Extraction','SSA') = 649.1640015 ;
vmpp('c_Extraction','RestofWorld') = 7010.532227 ;
vmpp('c_ProcFood','Oceania') = 11589.15527 ;
vmpp('c_ProcFood','EastAsia') = 63232.09766 ;
vmpp('c_ProcFood','SEAsia') = 30459.8418 ;
vmpp('c_ProcFood','SouthAsia') = 25101.78516 ;
vmpp('c_ProcFood','NAmerica') = 80109.77344 ;
vmpp('c_ProcFood','LatinAmer') = 28915.44922 ;
vmpp('c_ProcFood','EU_28') = 204402.1406 ;
vmpp('c_ProcFood','MENA') = 47078.57031 ;
vmpp('c_ProcFood','SSA') = 31929.91406 ;
vmpp('c_ProcFood','RestofWorld') = 37491.20703 ;
vmpp('c_TextWapp','Oceania') = 8235.498047 ;
vmpp('c_TextWapp','EastAsia') = 50619.27344 ;
vmpp('c_TextWapp','SEAsia') = 10125.2666 ;
vmpp('c_TextWapp','SouthAsia') = 9304.680664 ;
vmpp('c_TextWapp','NAmerica') = 118172.4766 ;
vmpp('c_TextWapp','LatinAmer') = 20545.8418 ;
vmpp('c_TextWapp','EU_28') = 209012.8125 ;
vmpp('c_TextWapp','MENA') = 33231.80469 ;
vmpp('c_TextWapp','SSA') = 14445.18262 ;
vmpp('c_TextWapp','RestofWorld') = 35379.63672 ;
vmpp('c_LightMnfc','Oceania') = 28041.83203 ;
vmpp('c_LightMnfc','EastAsia') = 68195.875 ;
vmpp('c_LightMnfc','SEAsia') = 25488.75195 ;
vmpp('c_LightMnfc','SouthAsia') = 9989.142578 ;
vmpp('c_LightMnfc','NAmerica') = 302670.2812 ;
vmpp('c_LightMnfc','LatinAmer') = 56847.53516 ;
vmpp('c_LightMnfc','EU_28') = 474859.1875 ;
vmpp('c_LightMnfc','MENA') = 83622.89062 ;
vmpp('c_LightMnfc','SSA') = 33337.80078 ;
vmpp('c_LightMnfc','RestofWorld') = 83595.63281 ;
vmpp('c_HeavyMnfc','Oceania') = 20710.68945 ;
vmpp('c_HeavyMnfc','EastAsia') = 131963.0469 ;
vmpp('c_HeavyMnfc','SEAsia') = 52078.55078 ;
vmpp('c_HeavyMnfc','SouthAsia') = 28345.1582 ;
vmpp('c_HeavyMnfc','NAmerica') = 286429.6875 ;
vmpp('c_HeavyMnfc','LatinAmer') = 75568.76562 ;
vmpp('c_HeavyMnfc','EU_28') = 491057.3125 ;
vmpp('c_HeavyMnfc','MENA') = 105204.1562 ;
vmpp('c_HeavyMnfc','SSA') = 43191.875 ;
vmpp('c_HeavyMnfc','RestofWorld') = 82473.67188 ;
vmpp('c_Util_Cons','Oceania') = 726.807251 ;
vmpp('c_Util_Cons','EastAsia') = 1391.075928 ;
vmpp('c_Util_Cons','SEAsia') = 1409.385498 ;
vmpp('c_Util_Cons','SouthAsia') = 132.2489777 ;
vmpp('c_Util_Cons','NAmerica') = 2664.403564 ;
vmpp('c_Util_Cons','LatinAmer') = 542.6152344 ;
vmpp('c_Util_Cons','EU_28') = 26203.16211 ;
vmpp('c_Util_Cons','MENA') = 4405.96582 ;
vmpp('c_Util_Cons','SSA') = 421.621521 ;
vmpp('c_Util_Cons','RestofWorld') = 3498.919189 ;
vmpp('c_TransComm','Oceania') = 14629.15625 ;
vmpp('c_TransComm','EastAsia') = 82370.57812 ;
vmpp('c_TransComm','SEAsia') = 22990.47656 ;
vmpp('c_TransComm','SouthAsia') = 15439.84375 ;
vmpp('c_TransComm','NAmerica') = 52164.84375 ;
vmpp('c_TransComm','LatinAmer') = 22773.87109 ;
vmpp('c_TransComm','EU_28') = 255060.5 ;
vmpp('c_TransComm','MENA') = 32359.73242 ;
vmpp('c_TransComm','SSA') = 10163.27441 ;
vmpp('c_TransComm','RestofWorld') = 29291.2832 ;
vmpp('c_OthService','Oceania') = 7312.991211 ;
vmpp('c_OthService','EastAsia') = 29193.08789 ;
vmpp('c_OthService','SEAsia') = 23275.47266 ;
vmpp('c_OthService','SouthAsia') = 5981.064453 ;
vmpp('c_OthService','NAmerica') = 59544.01172 ;
vmpp('c_OthService','LatinAmer') = 10643.46484 ;
vmpp('c_OthService','EU_28') = 220690.2031 ;
vmpp('c_OthService','MENA') = 20461.90234 ;
vmpp('c_OthService','SSA') = 8597.827148 ;
vmpp('c_OthService','RestofWorld') = 24116.07227 ;

* vdgb data (100 cells)
vdgb('c_Crops','Oceania') = 36.35664749 ;
vdgb('c_Crops','EastAsia') = 74.49730682 ;
vdgb('c_Crops','SEAsia') = 200.4946136 ;
vdgb('c_Crops','SouthAsia') = 1325.534424 ;
vdgb('c_Crops','NAmerica') = 106.7968445 ;
vdgb('c_Crops','LatinAmer') = 16.31310272 ;
vdgb('c_Crops','EU_28') = 353.8447876 ;
vdgb('c_Crops','MENA') = 926.0446777 ;
vdgb('c_Crops','SSA') = 191.8226013 ;
vdgb('c_Crops','RestofWorld') = 796.1400146 ;
vdgb('c_MeatLstk','Oceania') = 24.14238548 ;
vdgb('c_MeatLstk','EastAsia') = 50.23033142 ;
vdgb('c_MeatLstk','SEAsia') = 42.54290009 ;
vdgb('c_MeatLstk','SouthAsia') = 969.4197998 ;
vdgb('c_MeatLstk','NAmerica') = 43.25560379 ;
vdgb('c_MeatLstk','LatinAmer') = 21.15439606 ;
vdgb('c_MeatLstk','EU_28') = 55.6676445 ;
vdgb('c_MeatLstk','MENA') = 271.4714966 ;
vdgb('c_MeatLstk','SSA') = 2.687572956 ;
vdgb('c_MeatLstk','RestofWorld') = 1338.125977 ;
vdgb('c_Extraction','Oceania') = 417.4411621 ;
vdgb('c_Extraction','EastAsia') = 14.97671223 ;
vdgb('c_Extraction','SEAsia') = 36.36848068 ;
vdgb('c_Extraction','SouthAsia') = 94.89933777 ;
vdgb('c_Extraction','NAmerica') = 13.6767931 ;
vdgb('c_Extraction','LatinAmer') = 2.442579031 ;
vdgb('c_Extraction','EU_28') = 39.55942535 ;
vdgb('c_Extraction','MENA') = 243.4008789 ;
vdgb('c_Extraction','SSA') = 21.56903648 ;
vdgb('c_Extraction','RestofWorld') = 115.1735916 ;
vdgb('c_ProcFood','Oceania') = 3516.783691 ;
vdgb('c_ProcFood','EastAsia') = 805.2585449 ;
vdgb('c_ProcFood','SEAsia') = 409.9667053 ;
vdgb('c_ProcFood','SouthAsia') = 2257.072266 ;
vdgb('c_ProcFood','NAmerica') = 1114.393188 ;
vdgb('c_ProcFood','LatinAmer') = 525.9945679 ;
vdgb('c_ProcFood','EU_28') = 1387.706665 ;
vdgb('c_ProcFood','MENA') = 2127.211426 ;
vdgb('c_ProcFood','SSA') = 45.85132599 ;
vdgb('c_ProcFood','RestofWorld') = 608.1523438 ;
vdgb('c_TextWapp','Oceania') = 0.3030329645 ;
vdgb('c_TextWapp','EastAsia') = 632.9326782 ;
vdgb('c_TextWapp','SEAsia') = 41.00069427 ;
vdgb('c_TextWapp','SouthAsia') = 1972.317749 ;
vdgb('c_TextWapp','NAmerica') = 889.0980835 ;
vdgb('c_TextWapp','LatinAmer') = 311.1191711 ;
vdgb('c_TextWapp','EU_28') = 1100.811035 ;
vdgb('c_TextWapp','MENA') = 2001.96814 ;
vdgb('c_TextWapp','SSA') = 29.82575607 ;
vdgb('c_TextWapp','RestofWorld') = 482.7776184 ;
vdgb('c_LightMnfc','Oceania') = 13.93743229 ;
vdgb('c_LightMnfc','EastAsia') = 250.5587463 ;
vdgb('c_LightMnfc','SEAsia') = 1686.951172 ;
vdgb('c_LightMnfc','SouthAsia') = 6046.359375 ;
vdgb('c_LightMnfc','NAmerica') = 341.1736755 ;
vdgb('c_LightMnfc','LatinAmer') = 137.2872314 ;
vdgb('c_LightMnfc','EU_28') = 141.87677 ;
vdgb('c_LightMnfc','MENA') = 20466.34766 ;
vdgb('c_LightMnfc','SSA') = 53.7965126 ;
vdgb('c_LightMnfc','RestofWorld') = 785.2805176 ;
vdgb('c_HeavyMnfc','Oceania') = 239.3684387 ;
vdgb('c_HeavyMnfc','EastAsia') = 646.4690552 ;
vdgb('c_HeavyMnfc','SEAsia') = 526.6027832 ;
vdgb('c_HeavyMnfc','SouthAsia') = 5507.393066 ;
vdgb('c_HeavyMnfc','NAmerica') = 619.2070923 ;
vdgb('c_HeavyMnfc','LatinAmer') = 1923.125488 ;
vdgb('c_HeavyMnfc','EU_28') = 23372.85938 ;
vdgb('c_HeavyMnfc','MENA') = 12060.96191 ;
vdgb('c_HeavyMnfc','SSA') = 176.4337616 ;
vdgb('c_HeavyMnfc','RestofWorld') = 494.3577271 ;
vdgb('c_Util_Cons','Oceania') = 1236.496704 ;
vdgb('c_Util_Cons','EastAsia') = 530.0713501 ;
vdgb('c_Util_Cons','SEAsia') = 713.0939331 ;
vdgb('c_Util_Cons','SouthAsia') = 13916.57324 ;
vdgb('c_Util_Cons','NAmerica') = 8.471951485 ;
vdgb('c_Util_Cons','LatinAmer') = 120.3606567 ;
vdgb('c_Util_Cons','EU_28') = 373.2020569 ;
vdgb('c_Util_Cons','MENA') = 1533.230103 ;
vdgb('c_Util_Cons','SSA') = 239.2920837 ;
vdgb('c_Util_Cons','RestofWorld') = 199.3787231 ;
vdgb('c_TransComm','Oceania') = 15578.41895 ;
vdgb('c_TransComm','EastAsia') = 21910.20508 ;
vdgb('c_TransComm','SEAsia') = 2312.744629 ;
vdgb('c_TransComm','SouthAsia') = 12937.77148 ;
vdgb('c_TransComm','NAmerica') = 1947.592041 ;
vdgb('c_TransComm','LatinAmer') = 2069.169189 ;
vdgb('c_TransComm','EU_28') = 45015.02344 ;
vdgb('c_TransComm','MENA') = 13796.78027 ;
vdgb('c_TransComm','SSA') = 2186.246582 ;
vdgb('c_TransComm','RestofWorld') = 10070.69336 ;
vdgb('c_OthService','Oceania') = 266635.3125 ;
vdgb('c_OthService','EastAsia') = 2389068.5 ;
vdgb('c_OthService','SEAsia') = 226420.4219 ;
vdgb('c_OthService','SouthAsia') = 216251.375 ;
vdgb('c_OthService','NAmerica') = 3074352 ;
vdgb('c_OthService','LatinAmer') = 820648.5 ;
vdgb('c_OthService','EU_28') = 3726238.5 ;
vdgb('c_OthService','MENA') = 503341.875 ;
vdgb('c_OthService','SSA') = 205502.4844 ;
vdgb('c_OthService','RestofWorld') = 627356.0625 ;

* vdgp data (100 cells)
vdgp('c_Crops','Oceania') = 36.35671234 ;
vdgp('c_Crops','EastAsia') = 74.49730682 ;
vdgp('c_Crops','SEAsia') = 204.4811554 ;
vdgp('c_Crops','SouthAsia') = 1222.532593 ;
vdgp('c_Crops','NAmerica') = 106.3236694 ;
vdgp('c_Crops','LatinAmer') = 16.31310272 ;
vdgp('c_Crops','EU_28') = 358.7796021 ;
vdgp('c_Crops','MENA') = 926.6462402 ;
vdgp('c_Crops','SSA') = 195.7763214 ;
vdgp('c_Crops','RestofWorld') = 801.6503906 ;
vdgp('c_MeatLstk','Oceania') = 24.14318466 ;
vdgp('c_MeatLstk','EastAsia') = 50.23033142 ;
vdgp('c_MeatLstk','SEAsia') = 43.38977051 ;
vdgp('c_MeatLstk','SouthAsia') = 975.5653076 ;
vdgp('c_MeatLstk','NAmerica') = 43.25560379 ;
vdgp('c_MeatLstk','LatinAmer') = 21.15439606 ;
vdgp('c_MeatLstk','EU_28') = 55.67551041 ;
vdgp('c_MeatLstk','MENA') = 271.4962769 ;
vdgp('c_MeatLstk','SSA') = 2.745131731 ;
vdgp('c_MeatLstk','RestofWorld') = 1340.176147 ;
vdgp('c_Extraction','Oceania') = 417.4411621 ;
vdgp('c_Extraction','EastAsia') = 14.98153591 ;
vdgp('c_Extraction','SEAsia') = 38.24414825 ;
vdgp('c_Extraction','SouthAsia') = 94.89933777 ;
vdgp('c_Extraction','NAmerica') = 13.6767931 ;
vdgp('c_Extraction','LatinAmer') = 2.442591429 ;
vdgp('c_Extraction','EU_28') = 39.70827484 ;
vdgp('c_Extraction','MENA') = 243.4188995 ;
vdgp('c_Extraction','SSA') = 21.75220108 ;
vdgp('c_Extraction','RestofWorld') = 115.8375702 ;
vdgp('c_ProcFood','Oceania') = 3516.783936 ;
vdgp('c_ProcFood','EastAsia') = 805.2602539 ;
vdgp('c_ProcFood','SEAsia') = 411.4887085 ;
vdgp('c_ProcFood','SouthAsia') = 2351.383789 ;
vdgp('c_ProcFood','NAmerica') = 1114.393188 ;
vdgp('c_ProcFood','LatinAmer') = 554.4360962 ;
vdgp('c_ProcFood','EU_28') = 1388.356201 ;
vdgp('c_ProcFood','MENA') = 2129.142822 ;
vdgp('c_ProcFood','SSA') = 46.18078995 ;
vdgp('c_ProcFood','RestofWorld') = 611.7781982 ;
vdgp('c_TextWapp','Oceania') = 0.3033930957 ;
vdgp('c_TextWapp','EastAsia') = 632.9326782 ;
vdgp('c_TextWapp','SEAsia') = 41.27198792 ;
vdgp('c_TextWapp','SouthAsia') = 2017.539917 ;
vdgp('c_TextWapp','NAmerica') = 889.0980835 ;
vdgp('c_TextWapp','LatinAmer') = 314.16745 ;
vdgp('c_TextWapp','EU_28') = 1101.395386 ;
vdgp('c_TextWapp','MENA') = 2044.98584 ;
vdgp('c_TextWapp','SSA') = 30.05760193 ;
vdgp('c_TextWapp','RestofWorld') = 492.3314514 ;
vdgp('c_LightMnfc','Oceania') = 13.93853283 ;
vdgp('c_LightMnfc','EastAsia') = 251.9402618 ;
vdgp('c_LightMnfc','SEAsia') = 1716.290771 ;
vdgp('c_LightMnfc','SouthAsia') = 6411.030762 ;
vdgp('c_LightMnfc','NAmerica') = 341.1736755 ;
vdgp('c_LightMnfc','LatinAmer') = 139.9626617 ;
vdgp('c_LightMnfc','EU_28') = 142.6074982 ;
vdgp('c_LightMnfc','MENA') = 20534.47656 ;
vdgp('c_LightMnfc','SSA') = 54.96845245 ;
vdgp('c_LightMnfc','RestofWorld') = 799.9395752 ;
vdgp('c_HeavyMnfc','Oceania') = 244.8863678 ;
vdgp('c_HeavyMnfc','EastAsia') = 660.9736328 ;
vdgp('c_HeavyMnfc','SEAsia') = 543.9000854 ;
vdgp('c_HeavyMnfc','SouthAsia') = 5860.630859 ;
vdgp('c_HeavyMnfc','NAmerica') = 619.2070923 ;
vdgp('c_HeavyMnfc','LatinAmer') = 1931.145508 ;
vdgp('c_HeavyMnfc','EU_28') = 23376.9707 ;
vdgp('c_HeavyMnfc','MENA') = 12089.00293 ;
vdgp('c_HeavyMnfc','SSA') = 180.3073883 ;
vdgp('c_HeavyMnfc','RestofWorld') = 519.8869629 ;
vdgp('c_Util_Cons','Oceania') = 1257.020874 ;
vdgp('c_Util_Cons','EastAsia') = 530.1115723 ;
vdgp('c_Util_Cons','SEAsia') = 760.4713745 ;
vdgp('c_Util_Cons','SouthAsia') = 13905.69434 ;
vdgp('c_Util_Cons','NAmerica') = 8.471951485 ;
vdgp('c_Util_Cons','LatinAmer') = 140.1851501 ;
vdgp('c_Util_Cons','EU_28') = 373.2038574 ;
vdgp('c_Util_Cons','MENA') = 1535.04895 ;
vdgp('c_Util_Cons','SSA') = 255.4048004 ;
vdgp('c_Util_Cons','RestofWorld') = 201.9232483 ;
vdgp('c_TransComm','Oceania') = 15594.31445 ;
vdgp('c_TransComm','EastAsia') = 23039.90039 ;
vdgp('c_TransComm','SEAsia') = 2370.055908 ;
vdgp('c_TransComm','SouthAsia') = 12799.00195 ;
vdgp('c_TransComm','NAmerica') = 1947.592041 ;
vdgp('c_TransComm','LatinAmer') = 2138.40918 ;
vdgp('c_TransComm','EU_28') = 45018.16016 ;
vdgp('c_TransComm','MENA') = 13812.37988 ;
vdgp('c_TransComm','SSA') = 2190.523926 ;
vdgp('c_TransComm','RestofWorld') = 10084.11035 ;
vdgp('c_OthService','Oceania') = 267568.1562 ;
vdgp('c_OthService','EastAsia') = 2400953 ;
vdgp('c_OthService','SEAsia') = 226389.25 ;
vdgp('c_OthService','SouthAsia') = 217291.1094 ;
vdgp('c_OthService','NAmerica') = 3074352 ;
vdgp('c_OthService','LatinAmer') = 823097.1875 ;
vdgp('c_OthService','EU_28') = 3726243.25 ;
vdgp('c_OthService','MENA') = 504831.0625 ;
vdgp('c_OthService','SSA') = 206408.7812 ;
vdgp('c_OthService','RestofWorld') = 632113.3125 ;

* vmgb data (100 cells)
vmgb('c_Crops','Oceania') = 5.259678841 ;
vmgb('c_Crops','EastAsia') = 24.05498123 ;
vmgb('c_Crops','SEAsia') = 30.12493896 ;
vmgb('c_Crops','SouthAsia') = 14.09971046 ;
vmgb('c_Crops','NAmerica') = 1.511160612 ;
vmgb('c_Crops','LatinAmer') = 4.279137611 ;
vmgb('c_Crops','EU_28') = 213.4137268 ;
vmgb('c_Crops','MENA') = 355.159668 ;
vmgb('c_Crops','SSA') = 52.93864441 ;
vmgb('c_Crops','RestofWorld') = 19.67363358 ;
vmgb('c_MeatLstk','Oceania') = 1.257036805 ;
vmgb('c_MeatLstk','EastAsia') = 0.4825249016 ;
vmgb('c_MeatLstk','SEAsia') = 2.318826437 ;
vmgb('c_MeatLstk','SouthAsia') = 15.60474968 ;
vmgb('c_MeatLstk','NAmerica') = 0.1751246154 ;
vmgb('c_MeatLstk','LatinAmer') = 1.741819024 ;
vmgb('c_MeatLstk','EU_28') = 7.474952221 ;
vmgb('c_MeatLstk','MENA') = 188.6117859 ;
vmgb('c_MeatLstk','SSA') = 2.759335756 ;
vmgb('c_MeatLstk','RestofWorld') = 7.452277184 ;
vmgb('c_Extraction','Oceania') = 0.01286348049 ;
vmgb('c_Extraction','EastAsia') = 0.6921343207 ;
vmgb('c_Extraction','SEAsia') = 7.264962673 ;
vmgb('c_Extraction','SouthAsia') = 0.1606973261 ;
vmgb('c_Extraction','NAmerica') = 0.07826551795 ;
vmgb('c_Extraction','LatinAmer') = 0.01205283403 ;
vmgb('c_Extraction','EU_28') = 1.341089606 ;
vmgb('c_Extraction','MENA') = 273.2036743 ;
vmgb('c_Extraction','SSA') = 2.316338301 ;
vmgb('c_Extraction','RestofWorld') = 14.68784904 ;
vmgb('c_ProcFood','Oceania') = 1617.559814 ;
vmgb('c_ProcFood','EastAsia') = 17.89033127 ;
vmgb('c_ProcFood','SEAsia') = 6.177992344 ;
vmgb('c_ProcFood','SouthAsia') = 190.4613495 ;
vmgb('c_ProcFood','NAmerica') = 34.82213211 ;
vmgb('c_ProcFood','LatinAmer') = 63.58196259 ;
vmgb('c_ProcFood','EU_28') = 43.93666077 ;
vmgb('c_ProcFood','MENA') = 680.4909058 ;
vmgb('c_ProcFood','SSA') = 69.32720184 ;
vmgb('c_ProcFood','RestofWorld') = 20.91249657 ;
vmgb('c_TextWapp','Oceania') = 0.1133544743 ;
vmgb('c_TextWapp','EastAsia') = 33.14259338 ;
vmgb('c_TextWapp','SEAsia') = 1.260044336 ;
vmgb('c_TextWapp','SouthAsia') = 166.747879 ;
vmgb('c_TextWapp','NAmerica') = 30.13649559 ;
vmgb('c_TextWapp','LatinAmer') = 15.20519161 ;
vmgb('c_TextWapp','EU_28') = 61.90269089 ;
vmgb('c_TextWapp','MENA') = 1243.697754 ;
vmgb('c_TextWapp','SSA') = 6.266278744 ;
vmgb('c_TextWapp','RestofWorld') = 117.8314896 ;
vmgb('c_LightMnfc','Oceania') = 0.4383155406 ;
vmgb('c_LightMnfc','EastAsia') = 30.5396862 ;
vmgb('c_LightMnfc','SEAsia') = 75.75062561 ;
vmgb('c_LightMnfc','SouthAsia') = 774.8763428 ;
vmgb('c_LightMnfc','NAmerica') = 21.31020355 ;
vmgb('c_LightMnfc','LatinAmer') = 620.5578613 ;
vmgb('c_LightMnfc','EU_28') = 248.0542297 ;
vmgb('c_LightMnfc','MENA') = 23784.97656 ;
vmgb('c_LightMnfc','SSA') = 45.18103409 ;
vmgb('c_LightMnfc','RestofWorld') = 147.126709 ;
vmgb('c_HeavyMnfc','Oceania') = 388.4746094 ;
vmgb('c_HeavyMnfc','EastAsia') = 710.0758667 ;
vmgb('c_HeavyMnfc','SEAsia') = 257.5366516 ;
vmgb('c_HeavyMnfc','SouthAsia') = 2056.450928 ;
vmgb('c_HeavyMnfc','NAmerica') = 1056.421875 ;
vmgb('c_HeavyMnfc','LatinAmer') = 584.7304688 ;
vmgb('c_HeavyMnfc','EU_28') = 29509.77148 ;
vmgb('c_HeavyMnfc','MENA') = 14514.24219 ;
vmgb('c_HeavyMnfc','SSA') = 371.1512756 ;
vmgb('c_HeavyMnfc','RestofWorld') = 1564.926514 ;
vmgb('c_Util_Cons','Oceania') = 0.08369284123 ;
vmgb('c_Util_Cons','EastAsia') = 13.06372356 ;
vmgb('c_Util_Cons','SEAsia') = 10.32177067 ;
vmgb('c_Util_Cons','SouthAsia') = 42.88286591 ;
vmgb('c_Util_Cons','NAmerica') = 0.01976360939 ;
vmgb('c_Util_Cons','LatinAmer') = 6.991494656 ;
vmgb('c_Util_Cons','EU_28') = 3.915912867 ;
vmgb('c_Util_Cons','MENA') = 52.91032028 ;
vmgb('c_Util_Cons','SSA') = 30.24280739 ;
vmgb('c_Util_Cons','RestofWorld') = 5.580267429 ;
vmgb('c_TransComm','Oceania') = 125.4253845 ;
vmgb('c_TransComm','EastAsia') = 692.6333008 ;
vmgb('c_TransComm','SEAsia') = 17.12079048 ;
vmgb('c_TransComm','SouthAsia') = 328.7103271 ;
vmgb('c_TransComm','NAmerica') = 62.88634491 ;
vmgb('c_TransComm','LatinAmer') = 45.76760101 ;
vmgb('c_TransComm','EU_28') = 3140.614014 ;
vmgb('c_TransComm','MENA') = 6287.291016 ;
vmgb('c_TransComm','SSA') = 1175.86438 ;
vmgb('c_TransComm','RestofWorld') = 285.8727112 ;
vmgb('c_OthService','Oceania') = 520.9078369 ;
vmgb('c_OthService','EastAsia') = 10041.7832 ;
vmgb('c_OthService','SEAsia') = 2454.423584 ;
vmgb('c_OthService','SouthAsia') = 4000.23999 ;
vmgb('c_OthService','NAmerica') = 2044.214478 ;
vmgb('c_OthService','LatinAmer') = 3987.324463 ;
vmgb('c_OthService','EU_28') = 56297.75391 ;
vmgb('c_OthService','MENA') = 34231.66797 ;
vmgb('c_OthService','SSA') = 8465.493164 ;
vmgb('c_OthService','RestofWorld') = 5637.752441 ;

* vmgp data (100 cells)
vmgp('c_Crops','Oceania') = 5.622738361 ;
vmgp('c_Crops','EastAsia') = 24.05498123 ;
vmgp('c_Crops','SEAsia') = 30.99768257 ;
vmgp('c_Crops','SouthAsia') = 14.70432472 ;
vmgp('c_Crops','NAmerica') = 1.511160612 ;
vmgp('c_Crops','LatinAmer') = 4.279138088 ;
vmgp('c_Crops','EU_28') = 202.3208466 ;
vmgp('c_Crops','MENA') = 355.3268738 ;
vmgp('c_Crops','SSA') = 53.59663773 ;
vmgp('c_Crops','RestofWorld') = 19.80768585 ;
vmgp('c_MeatLstk','Oceania') = 1.329437613 ;
vmgp('c_MeatLstk','EastAsia') = 0.4825249016 ;
vmgp('c_MeatLstk','SEAsia') = 2.432996988 ;
vmgp('c_MeatLstk','SouthAsia') = 16.40976143 ;
vmgp('c_MeatLstk','NAmerica') = 0.1751246154 ;
vmgp('c_MeatLstk','LatinAmer') = 1.741819024 ;
vmgp('c_MeatLstk','EU_28') = 7.915271282 ;
vmgp('c_MeatLstk','MENA') = 188.6517487 ;
vmgp('c_MeatLstk','SSA') = 3.04836154 ;
vmgp('c_MeatLstk','RestofWorld') = 7.57704401 ;
vmgp('c_Extraction','Oceania') = 0.01365573518 ;
vmgp('c_Extraction','EastAsia') = 0.6927003264 ;
vmgp('c_Extraction','SEAsia') = 7.734649658 ;
vmgp('c_Extraction','SouthAsia') = 0.1762948483 ;
vmgp('c_Extraction','NAmerica') = 0.07826551795 ;
vmgp('c_Extraction','LatinAmer') = 0.01205418073 ;
vmgp('c_Extraction','EU_28') = 1.351316094 ;
vmgp('c_Extraction','MENA') = 274.2384949 ;
vmgp('c_Extraction','SSA') = 2.408565521 ;
vmgp('c_Extraction','RestofWorld') = 14.69723988 ;
vmgp('c_ProcFood','Oceania') = 1617.623047 ;
vmgp('c_ProcFood','EastAsia') = 17.89302063 ;
vmgp('c_ProcFood','SEAsia') = 6.477556705 ;
vmgp('c_ProcFood','SouthAsia') = 200.079071 ;
vmgp('c_ProcFood','NAmerica') = 34.82213211 ;
vmgp('c_ProcFood','LatinAmer') = 63.7980423 ;
vmgp('c_ProcFood','EU_28') = 44.04983139 ;
vmgp('c_ProcFood','MENA') = 681.0241699 ;
vmgp('c_ProcFood','SSA') = 71.1568222 ;
vmgp('c_ProcFood','RestofWorld') = 21.34942627 ;
vmgp('c_TextWapp','Oceania') = 0.1186960787 ;
vmgp('c_TextWapp','EastAsia') = 33.14271545 ;
vmgp('c_TextWapp','SEAsia') = 1.270493746 ;
vmgp('c_TextWapp','SouthAsia') = 183.6413116 ;
vmgp('c_TextWapp','NAmerica') = 30.13649559 ;
vmgp('c_TextWapp','LatinAmer') = 15.36805725 ;
vmgp('c_TextWapp','EU_28') = 61.93459702 ;
vmgp('c_TextWapp','MENA') = 1243.842529 ;
vmgp('c_TextWapp','SSA') = 6.339251518 ;
vmgp('c_TextWapp','RestofWorld') = 123.9492035 ;
vmgp('c_LightMnfc','Oceania') = 0.4607716799 ;
vmgp('c_LightMnfc','EastAsia') = 30.57922745 ;
vmgp('c_LightMnfc','SEAsia') = 81.61870575 ;
vmgp('c_LightMnfc','SouthAsia') = 985.4976196 ;
vmgp('c_LightMnfc','NAmerica') = 21.31020355 ;
vmgp('c_LightMnfc','LatinAmer') = 665.824646 ;
vmgp('c_LightMnfc','EU_28') = 252.8082123 ;
vmgp('c_LightMnfc','MENA') = 23821.05664 ;
vmgp('c_LightMnfc','SSA') = 47.8572731 ;
vmgp('c_LightMnfc','RestofWorld') = 149.1654663 ;
vmgp('c_HeavyMnfc','Oceania') = 397.5142212 ;
vmgp('c_HeavyMnfc','EastAsia') = 712.5292969 ;
vmgp('c_HeavyMnfc','SEAsia') = 265.4073181 ;
vmgp('c_HeavyMnfc','SouthAsia') = 2393.003418 ;
vmgp('c_HeavyMnfc','NAmerica') = 1056.421875 ;
vmgp('c_HeavyMnfc','LatinAmer') = 606.4261475 ;
vmgp('c_HeavyMnfc','EU_28') = 30115.35938 ;
vmgp('c_HeavyMnfc','MENA') = 14430.55664 ;
vmgp('c_HeavyMnfc','SSA') = 385.3009644 ;
vmgp('c_HeavyMnfc','RestofWorld') = 1595.446411 ;
vmgp('c_Util_Cons','Oceania') = 0.08477212489 ;
vmgp('c_Util_Cons','EastAsia') = 13.23717976 ;
vmgp('c_Util_Cons','SEAsia') = 10.39082146 ;
vmgp('c_Util_Cons','SouthAsia') = 42.96527863 ;
vmgp('c_Util_Cons','NAmerica') = 0.01976360939 ;
vmgp('c_Util_Cons','LatinAmer') = 6.991495132 ;
vmgp('c_Util_Cons','EU_28') = 4.267457008 ;
vmgp('c_Util_Cons','MENA') = 55.87149048 ;
vmgp('c_Util_Cons','SSA') = 34.29779816 ;
vmgp('c_Util_Cons','RestofWorld') = 6.160098553 ;
vmgp('c_TransComm','Oceania') = 128.3091583 ;
vmgp('c_TransComm','EastAsia') = 800.0516968 ;
vmgp('c_TransComm','SEAsia') = 19.38676071 ;
vmgp('c_TransComm','SouthAsia') = 328.7409973 ;
vmgp('c_TransComm','NAmerica') = 62.88634491 ;
vmgp('c_TransComm','LatinAmer') = 45.76893997 ;
vmgp('c_TransComm','EU_28') = 3098.592773 ;
vmgp('c_TransComm','MENA') = 6449.761719 ;
vmgp('c_TransComm','SSA') = 1323.501831 ;
vmgp('c_TransComm','RestofWorld') = 284.0617371 ;
vmgp('c_OthService','Oceania') = 533.2050781 ;
vmgp('c_OthService','EastAsia') = 10941.99805 ;
vmgp('c_OthService','SEAsia') = 2516.617676 ;
vmgp('c_OthService','SouthAsia') = 4059.791748 ;
vmgp('c_OthService','NAmerica') = 2044.214478 ;
vmgp('c_OthService','LatinAmer') = 3991.114502 ;
vmgp('c_OthService','EU_28') = 56653.77344 ;
vmgp('c_OthService','MENA') = 34660.33203 ;
vmgp('c_OthService','SSA') = 9133.455078 ;
vmgp('c_OthService','RestofWorld') = 5679.307129 ;

* vdib data (100 cells)
vdib('c_Crops','Oceania') = 552.3233643 ;
vdib('c_Crops','EastAsia') = 443.3145142 ;
vdib('c_Crops','SEAsia') = 4815.713867 ;
vdib('c_Crops','SouthAsia') = 328.6303711 ;
vdib('c_Crops','NAmerica') = 32.71006012 ;
vdib('c_Crops','LatinAmer') = 6129.564453 ;
vdib('c_Crops','EU_28') = 1151.526855 ;
vdib('c_Crops','MENA') = 4510.087891 ;
vdib('c_Crops','SSA') = 44.88806534 ;
vdib('c_Crops','RestofWorld') = 2854.27417 ;
vdib('c_MeatLstk','Oceania') = 550.7305908 ;
vdib('c_MeatLstk','EastAsia') = 53488.62109 ;
vdib('c_MeatLstk','SEAsia') = 4131.293945 ;
vdib('c_MeatLstk','SouthAsia') = 1189.143677 ;
vdib('c_MeatLstk','NAmerica') = 827.3233643 ;
vdib('c_MeatLstk','LatinAmer') = 14698.91895 ;
vdib('c_MeatLstk','EU_28') = 3216.005127 ;
vdib('c_MeatLstk','MENA') = 3957.044922 ;
vdib('c_MeatLstk','SSA') = 2269.453613 ;
vdib('c_MeatLstk','RestofWorld') = 3009.87915 ;
vdib('c_Extraction','Oceania') = 2167.71167 ;
vdib('c_Extraction','EastAsia') = 469.161377 ;
vdib('c_Extraction','SEAsia') = 1595.371704 ;
vdib('c_Extraction','SouthAsia') = 786.310791 ;
vdib('c_Extraction','NAmerica') = 9007.075195 ;
vdib('c_Extraction','LatinAmer') = 1310.101074 ;
vdib('c_Extraction','EU_28') = 698.4799194 ;
vdib('c_Extraction','MENA') = 1149.817993 ;
vdib('c_Extraction','SSA') = 1152.003784 ;
vdib('c_Extraction','RestofWorld') = 823.4624023 ;
vdib('c_ProcFood','Oceania') = 0.2275110483 ;
vdib('c_ProcFood','EastAsia') = 15.33538818 ;
vdib('c_ProcFood','SEAsia') = 1930.469238 ;
vdib('c_ProcFood','SouthAsia') = 78.44683838 ;
vdib('c_ProcFood','NAmerica') = 292.8720398 ;
vdib('c_ProcFood','LatinAmer') = 1572.836548 ;
vdib('c_ProcFood','EU_28') = 3.461614609 ;
vdib('c_ProcFood','MENA') = 2842.333984 ;
vdib('c_ProcFood','SSA') = 1090.00354 ;
vdib('c_ProcFood','RestofWorld') = 1296.966309 ;
vdib('c_TextWapp','Oceania') = 458.8385315 ;
vdib('c_TextWapp','EastAsia') = 2992.902588 ;
vdib('c_TextWapp','SEAsia') = 562.3046265 ;
vdib('c_TextWapp','SouthAsia') = 252.3429718 ;
vdib('c_TextWapp','NAmerica') = 4750.186035 ;
vdib('c_TextWapp','LatinAmer') = 526.7179565 ;
vdib('c_TextWapp','EU_28') = 727.5792236 ;
vdib('c_TextWapp','MENA') = 1399.595581 ;
vdib('c_TextWapp','SSA') = 114.8966675 ;
vdib('c_TextWapp','RestofWorld') = 320.3352966 ;
vdib('c_LightMnfc','Oceania') = 15815.06152 ;
vdib('c_LightMnfc','EastAsia') = 478722.5 ;
vdib('c_LightMnfc','SEAsia') = 33067.75391 ;
vdib('c_LightMnfc','SouthAsia') = 80019.52344 ;
vdib('c_LightMnfc','NAmerica') = 365862.75 ;
vdib('c_LightMnfc','LatinAmer') = 80329.80469 ;
vdib('c_LightMnfc','EU_28') = 185340.8125 ;
vdib('c_LightMnfc','MENA') = 48509.55078 ;
vdib('c_LightMnfc','SSA') = 22006.70312 ;
vdib('c_LightMnfc','RestofWorld') = 29788.63086 ;
vdib('c_HeavyMnfc','Oceania') = 17012.49023 ;
vdib('c_HeavyMnfc','EastAsia') = 832163.375 ;
vdib('c_HeavyMnfc','SEAsia') = 20985.30273 ;
vdib('c_HeavyMnfc','SouthAsia') = 117625.4688 ;
vdib('c_HeavyMnfc','NAmerica') = 494476.2188 ;
vdib('c_HeavyMnfc','LatinAmer') = 95020.01562 ;
vdib('c_HeavyMnfc','EU_28') = 325274.9062 ;
vdib('c_HeavyMnfc','MENA') = 69270.09375 ;
vdib('c_HeavyMnfc','SSA') = 27419.72461 ;
vdib('c_HeavyMnfc','RestofWorld') = 76148.85156 ;
vdib('c_Util_Cons','Oceania') = 211750.8438 ;
vdib('c_Util_Cons','EastAsia') = 2486585.5 ;
vdib('c_Util_Cons','SEAsia') = 317719.0312 ;
vdib('c_Util_Cons','SouthAsia') = 344434.625 ;
vdib('c_Util_Cons','NAmerica') = 1665374 ;
vdib('c_Util_Cons','LatinAmer') = 456081.5312 ;
vdib('c_Util_Cons','EU_28') = 1679847.25 ;
vdib('c_Util_Cons','MENA') = 557962.6875 ;
vdib('c_Util_Cons','SSA') = 127020.1328 ;
vdib('c_Util_Cons','RestofWorld') = 422578.625 ;
vdib('c_TransComm','Oceania') = 23881.63867 ;
vdib('c_TransComm','EastAsia') = 221098.2656 ;
vdib('c_TransComm','SEAsia') = 47942.60156 ;
vdib('c_TransComm','SouthAsia') = 35380.1875 ;
vdib('c_TransComm','NAmerica') = 297900.4062 ;
vdib('c_TransComm','LatinAmer') = 65680.67969 ;
vdib('c_TransComm','EU_28') = 19730.35938 ;
vdib('c_TransComm','MENA') = 68346.24219 ;
vdib('c_TransComm','SSA') = 17069.01172 ;
vdib('c_TransComm','RestofWorld') = 36697.09766 ;
vdib('c_OthService','Oceania') = 47592.48438 ;
vdib('c_OthService','EastAsia') = 384083 ;
vdib('c_OthService','SEAsia') = 16588.81445 ;
vdib('c_OthService','SouthAsia') = 9967.03418 ;
vdib('c_OthService','NAmerica') = 194645.5469 ;
vdib('c_OthService','LatinAmer') = 26338.16602 ;
vdib('c_OthService','EU_28') = 496214.4688 ;
vdib('c_OthService','MENA') = 35304.26953 ;
vdib('c_OthService','SSA') = 9115.420898 ;
vdib('c_OthService','RestofWorld') = 55323.80078 ;

* vdip data (100 cells)
vdip('c_Crops','Oceania') = 552.605835 ;
vdip('c_Crops','EastAsia') = 443.5791321 ;
vdip('c_Crops','SEAsia') = 4852.722168 ;
vdip('c_Crops','SouthAsia') = 328.6044006 ;
vdip('c_Crops','NAmerica') = 32.71001053 ;
vdip('c_Crops','LatinAmer') = 6394.221191 ;
vdip('c_Crops','EU_28') = 1151.582153 ;
vdip('c_Crops','MENA') = 4505.496582 ;
vdip('c_Crops','SSA') = 45.42737579 ;
vdip('c_Crops','RestofWorld') = 2868.977295 ;
vdip('c_MeatLstk','Oceania') = 552.2866821 ;
vdip('c_MeatLstk','EastAsia') = 53506.58203 ;
vdip('c_MeatLstk','SEAsia') = 4187.045898 ;
vdip('c_MeatLstk','SouthAsia') = 1200.59375 ;
vdip('c_MeatLstk','NAmerica') = 827.3233032 ;
vdip('c_MeatLstk','LatinAmer') = 15823.61523 ;
vdip('c_MeatLstk','EU_28') = 3216.419189 ;
vdip('c_MeatLstk','MENA') = 3969.251953 ;
vdip('c_MeatLstk','SSA') = 2278.043701 ;
vdip('c_MeatLstk','RestofWorld') = 3027.096191 ;
vdip('c_Extraction','Oceania') = 2167.71167 ;
vdip('c_Extraction','EastAsia') = 472.2492676 ;
vdip('c_Extraction','SEAsia') = 1615.659546 ;
vdip('c_Extraction','SouthAsia') = 788.9370117 ;
vdip('c_Extraction','NAmerica') = 9007.076172 ;
vdip('c_Extraction','LatinAmer') = 1344.252563 ;
vdip('c_Extraction','EU_28') = 700.196167 ;
vdip('c_Extraction','MENA') = 1152.328857 ;
vdip('c_Extraction','SSA') = 1153.817139 ;
vdip('c_Extraction','RestofWorld') = 824.3641357 ;
vdip('c_ProcFood','Oceania') = 0.2275110483 ;
vdip('c_ProcFood','EastAsia') = 15.33538818 ;
vdip('c_ProcFood','SEAsia') = 1933.687256 ;
vdip('c_ProcFood','SouthAsia') = 82.39084625 ;
vdip('c_ProcFood','NAmerica') = 292.7999573 ;
vdip('c_ProcFood','LatinAmer') = 1572.836548 ;
vdip('c_ProcFood','EU_28') = 3.461614609 ;
vdip('c_ProcFood','MENA') = 2869.38916 ;
vdip('c_ProcFood','SSA') = 1099.226318 ;
vdip('c_ProcFood','RestofWorld') = 1296.983032 ;
vdip('c_TextWapp','Oceania') = 494.6755371 ;
vdip('c_TextWapp','EastAsia') = 3001.995361 ;
vdip('c_TextWapp','SEAsia') = 563.0925903 ;
vdip('c_TextWapp','SouthAsia') = 250.091629 ;
vdip('c_TextWapp','NAmerica') = 4883.085938 ;
vdip('c_TextWapp','LatinAmer') = 527.1398926 ;
vdip('c_TextWapp','EU_28') = 727.5792236 ;
vdip('c_TextWapp','MENA') = 1435.600952 ;
vdip('c_TextWapp','SSA') = 124.5654907 ;
vdip('c_TextWapp','RestofWorld') = 329.0890503 ;
vdip('c_LightMnfc','Oceania') = 16449.35156 ;
vdip('c_LightMnfc','EastAsia') = 508013.125 ;
vdip('c_LightMnfc','SEAsia') = 34446.08594 ;
vdip('c_LightMnfc','SouthAsia') = 86499.92188 ;
vdip('c_LightMnfc','NAmerica') = 375206.1875 ;
vdip('c_LightMnfc','LatinAmer') = 89358.64062 ;
vdip('c_LightMnfc','EU_28') = 185377.0938 ;
vdip('c_LightMnfc','MENA') = 50573.11719 ;
vdip('c_LightMnfc','SSA') = 23133.1582 ;
vdip('c_LightMnfc','RestofWorld') = 30509.45703 ;
vdip('c_HeavyMnfc','Oceania') = 17324.65039 ;
vdip('c_HeavyMnfc','EastAsia') = 864771.625 ;
vdip('c_HeavyMnfc','SEAsia') = 21312.29102 ;
vdip('c_HeavyMnfc','SouthAsia') = 130024.0859 ;
vdip('c_HeavyMnfc','NAmerica') = 499467.25 ;
vdip('c_HeavyMnfc','LatinAmer') = 106951.9453 ;
vdip('c_HeavyMnfc','EU_28') = 325331.4062 ;
vdip('c_HeavyMnfc','MENA') = 69810.15625 ;
vdip('c_HeavyMnfc','SSA') = 30589.79102 ;
vdip('c_HeavyMnfc','RestofWorld') = 79264.4375 ;
vdip('c_Util_Cons','Oceania') = 231873.5312 ;
vdip('c_Util_Cons','EastAsia') = 2595303.25 ;
vdip('c_Util_Cons','SEAsia') = 322481.2188 ;
vdip('c_Util_Cons','SouthAsia') = 344379.6875 ;
vdip('c_Util_Cons','NAmerica') = 1676424 ;
vdip('c_Util_Cons','LatinAmer') = 467185.25 ;
vdip('c_Util_Cons','EU_28') = 1679849.125 ;
vdip('c_Util_Cons','MENA') = 564117.3125 ;
vdip('c_Util_Cons','SSA') = 128916.4922 ;
vdip('c_Util_Cons','RestofWorld') = 443854.8125 ;
vdip('c_TransComm','Oceania') = 23975.50391 ;
vdip('c_TransComm','EastAsia') = 237171.1406 ;
vdip('c_TransComm','SEAsia') = 49039.65625 ;
vdip('c_TransComm','SouthAsia') = 35181.44531 ;
vdip('c_TransComm','NAmerica') = 298420.1875 ;
vdip('c_TransComm','LatinAmer') = 65746.32812 ;
vdip('c_TransComm','EU_28') = 19730.64062 ;
vdip('c_TransComm','MENA') = 68315.0625 ;
vdip('c_TransComm','SSA') = 17103.40039 ;
vdip('c_TransComm','RestofWorld') = 36364.84375 ;
vdip('c_OthService','Oceania') = 48300.89453 ;
vdip('c_OthService','EastAsia') = 412808.8438 ;
vdip('c_OthService','SEAsia') = 16689.78906 ;
vdip('c_OthService','SouthAsia') = 10145.17383 ;
vdip('c_OthService','NAmerica') = 197221.625 ;
vdip('c_OthService','LatinAmer') = 27166.06641 ;
vdip('c_OthService','EU_28') = 496214.7812 ;
vdip('c_OthService','MENA') = 35351.89062 ;
vdip('c_OthService','SSA') = 9501.285156 ;
vdip('c_OthService','RestofWorld') = 56990.69531 ;

* vmib data (100 cells)
vmib('c_Crops','Oceania') = 0.05884766951 ;
vmib('c_Crops','EastAsia') = 207.2924194 ;
vmib('c_Crops','SEAsia') = 163.4020996 ;
vmib('c_Crops','SouthAsia') = 0.1666913778 ;
vmib('c_Crops','NAmerica') = 44.41765594 ;
vmib('c_Crops','LatinAmer') = 157.0385437 ;
vmib('c_Crops','EU_28') = 744.4780273 ;
vmib('c_Crops','MENA') = 1482.627319 ;
vmib('c_Crops','SSA') = 32.00466156 ;
vmib('c_Crops','RestofWorld') = 199.690506 ;
vmib('c_MeatLstk','Oceania') = 3.091566086 ;
vmib('c_MeatLstk','EastAsia') = 993.8710327 ;
vmib('c_MeatLstk','SEAsia') = 283.0151978 ;
vmib('c_MeatLstk','SouthAsia') = 27.43815422 ;
vmib('c_MeatLstk','NAmerica') = 310.8101501 ;
vmib('c_MeatLstk','LatinAmer') = 173.0986176 ;
vmib('c_MeatLstk','EU_28') = 421.9536743 ;
vmib('c_MeatLstk','MENA') = 1038.408081 ;
vmib('c_MeatLstk','SSA') = 10.5219965 ;
vmib('c_MeatLstk','RestofWorld') = 101.7286224 ;
vmib('c_Extraction','Oceania') = 19.65583992 ;
vmib('c_Extraction','EastAsia') = 2.583641052 ;
vmib('c_Extraction','SEAsia') = 4.931149006 ;
vmib('c_Extraction','SouthAsia') = 89.49723816 ;
vmib('c_Extraction','NAmerica') = 49.07439804 ;
vmib('c_Extraction','LatinAmer') = 6.539701462 ;
vmib('c_Extraction','EU_28') = 48.67494202 ;
vmib('c_Extraction','MENA') = 333.7492981 ;
vmib('c_Extraction','SSA') = 61.65352631 ;
vmib('c_Extraction','RestofWorld') = 9.343409538 ;
vmib('c_ProcFood','Oceania') = 0.03261793032 ;
vmib('c_ProcFood','EastAsia') = 9.162628174 ;
vmib('c_ProcFood','SEAsia') = 175.7536774 ;
vmib('c_ProcFood','SouthAsia') = 1.436378717 ;
vmib('c_ProcFood','NAmerica') = 49.39813995 ;
vmib('c_ProcFood','LatinAmer') = 0.1853484362 ;
vmib('c_ProcFood','EU_28') = 1.77814126 ;
vmib('c_ProcFood','MENA') = 3164.929688 ;
vmib('c_ProcFood','SSA') = 140.334671 ;
vmib('c_ProcFood','RestofWorld') = 159.2431183 ;
vmib('c_TextWapp','Oceania') = 205.4501495 ;
vmib('c_TextWapp','EastAsia') = 1203.835815 ;
vmib('c_TextWapp','SEAsia') = 225.1712494 ;
vmib('c_TextWapp','SouthAsia') = 34.67084885 ;
vmib('c_TextWapp','NAmerica') = 964.4837036 ;
vmib('c_TextWapp','LatinAmer') = 121.7492218 ;
vmib('c_TextWapp','EU_28') = 351.2015381 ;
vmib('c_TextWapp','MENA') = 311.29953 ;
vmib('c_TextWapp','SSA') = 156.7162018 ;
vmib('c_TextWapp','RestofWorld') = 420.6361389 ;
vmib('c_LightMnfc','Oceania') = 31012.94922 ;
vmib('c_LightMnfc','EastAsia') = 68455.71094 ;
vmib('c_LightMnfc','SEAsia') = 38887.53516 ;
vmib('c_LightMnfc','SouthAsia') = 19983.35156 ;
vmib('c_LightMnfc','NAmerica') = 140227.7812 ;
vmib('c_LightMnfc','LatinAmer') = 46130.19531 ;
vmib('c_LightMnfc','EU_28') = 175626.9219 ;
vmib('c_LightMnfc','MENA') = 35513.69922 ;
vmib('c_LightMnfc','SSA') = 24849.30078 ;
vmib('c_LightMnfc','RestofWorld') = 51534.67969 ;
vmib('c_HeavyMnfc','Oceania') = 37751.75 ;
vmib('c_HeavyMnfc','EastAsia') = 282644.3438 ;
vmib('c_HeavyMnfc','SEAsia') = 115295.2891 ;
vmib('c_HeavyMnfc','SouthAsia') = 63225.45703 ;
vmib('c_HeavyMnfc','NAmerica') = 310149.6562 ;
vmib('c_HeavyMnfc','LatinAmer') = 97508 ;
vmib('c_HeavyMnfc','EU_28') = 356727.8125 ;
vmib('c_HeavyMnfc','MENA') = 100982.4219 ;
vmib('c_HeavyMnfc','SSA') = 41658.28125 ;
vmib('c_HeavyMnfc','RestofWorld') = 95427.4375 ;
vmib('c_Util_Cons','Oceania') = 384.5050049 ;
vmib('c_Util_Cons','EastAsia') = 15440.75098 ;
vmib('c_Util_Cons','SEAsia') = 2891.994385 ;
vmib('c_Util_Cons','SouthAsia') = 715.7573242 ;
vmib('c_Util_Cons','NAmerica') = 3412.257812 ;
vmib('c_Util_Cons','LatinAmer') = 1003.64679 ;
vmib('c_Util_Cons','EU_28') = 34404.01953 ;
vmib('c_Util_Cons','MENA') = 3858.216064 ;
vmib('c_Util_Cons','SSA') = 4200.250488 ;
vmib('c_Util_Cons','RestofWorld') = 13892.75195 ;
vmib('c_TransComm','Oceania') = 97.26316833 ;
vmib('c_TransComm','EastAsia') = 3282.099365 ;
vmib('c_TransComm','SEAsia') = 369.3285828 ;
vmib('c_TransComm','SouthAsia') = 299.5599365 ;
vmib('c_TransComm','NAmerica') = 521.5731812 ;
vmib('c_TransComm','LatinAmer') = 10.10368061 ;
vmib('c_TransComm','EU_28') = 501.2569275 ;
vmib('c_TransComm','MENA') = 1005.11969 ;
vmib('c_TransComm','SSA') = 348.8943176 ;
vmib('c_TransComm','RestofWorld') = 580.8533936 ;
vmib('c_OthService','Oceania') = 1173.831909 ;
vmib('c_OthService','EastAsia') = 17040.14648 ;
vmib('c_OthService','SEAsia') = 168.615097 ;
vmib('c_OthService','SouthAsia') = 5418.892578 ;
vmib('c_OthService','NAmerica') = 8533.195312 ;
vmib('c_OthService','LatinAmer') = 2423.35791 ;
vmib('c_OthService','EU_28') = 60227.45703 ;
vmib('c_OthService','MENA') = 7355.47998 ;
vmib('c_OthService','SSA') = 3040.077637 ;
vmib('c_OthService','RestofWorld') = 8809.259766 ;

* vmip data (100 cells)
vmip('c_Crops','Oceania') = 0.05884766951 ;
vmip('c_Crops','EastAsia') = 207.2924194 ;
vmip('c_Crops','SEAsia') = 164.3151245 ;
vmip('c_Crops','SouthAsia') = 0.1666913778 ;
vmip('c_Crops','NAmerica') = 44.6123085 ;
vmip('c_Crops','LatinAmer') = 161.0921478 ;
vmip('c_Crops','EU_28') = 720.1433716 ;
vmip('c_Crops','MENA') = 1486.117676 ;
vmip('c_Crops','SSA') = 34.20951843 ;
vmip('c_Crops','RestofWorld') = 200.8293915 ;
vmip('c_MeatLstk','Oceania') = 3.091624975 ;
vmip('c_MeatLstk','EastAsia') = 1202.337036 ;
vmip('c_MeatLstk','SEAsia') = 283.9414368 ;
vmip('c_MeatLstk','SouthAsia') = 29.42976952 ;
vmip('c_MeatLstk','NAmerica') = 312.4425964 ;
vmip('c_MeatLstk','LatinAmer') = 174.2372589 ;
vmip('c_MeatLstk','EU_28') = 393.9877319 ;
vmip('c_MeatLstk','MENA') = 1042.865845 ;
vmip('c_MeatLstk','SSA') = 11.65895939 ;
vmip('c_MeatLstk','RestofWorld') = 101.7377548 ;
vmip('c_Extraction','Oceania') = 19.65583992 ;
vmip('c_Extraction','EastAsia') = 2.615722895 ;
vmip('c_Extraction','SEAsia') = 4.942109108 ;
vmip('c_Extraction','SouthAsia') = 91.99105835 ;
vmip('c_Extraction','NAmerica') = 49.46346283 ;
vmip('c_Extraction','LatinAmer') = 6.585662365 ;
vmip('c_Extraction','EU_28') = 48.91973877 ;
vmip('c_Extraction','MENA') = 333.7543945 ;
vmip('c_Extraction','SSA') = 68.49838257 ;
vmip('c_Extraction','RestofWorld') = 9.370429993 ;
vmip('c_ProcFood','Oceania') = 0.03261793032 ;
vmip('c_ProcFood','EastAsia') = 9.162629128 ;
vmip('c_ProcFood','SEAsia') = 175.7536926 ;
vmip('c_ProcFood','SouthAsia') = 1.477839708 ;
vmip('c_ProcFood','NAmerica') = 49.60615921 ;
vmip('c_ProcFood','LatinAmer') = 0.1853484362 ;
vmip('c_ProcFood','EU_28') = 1.804669499 ;
vmip('c_ProcFood','MENA') = 3164.929688 ;
vmip('c_ProcFood','SSA') = 151.2703094 ;
vmip('c_ProcFood','RestofWorld') = 159.2431183 ;
vmip('c_TextWapp','Oceania') = 223.993454 ;
vmip('c_TextWapp','EastAsia') = 1221.452515 ;
vmip('c_TextWapp','SEAsia') = 227.1145782 ;
vmip('c_TextWapp','SouthAsia') = 35.85025406 ;
vmip('c_TextWapp','NAmerica') = 995.1598511 ;
vmip('c_TextWapp','LatinAmer') = 124.272934 ;
vmip('c_TextWapp','EU_28') = 361.7525024 ;
vmip('c_TextWapp','MENA') = 314.357666 ;
vmip('c_TextWapp','SSA') = 205.7200317 ;
vmip('c_TextWapp','RestofWorld') = 420.841095 ;
vmip('c_LightMnfc','Oceania') = 32821.37891 ;
vmip('c_LightMnfc','EastAsia') = 80866.29688 ;
vmip('c_LightMnfc','SEAsia') = 41168.07422 ;
vmip('c_LightMnfc','SouthAsia') = 26845.39453 ;
vmip('c_LightMnfc','NAmerica') = 144934.3438 ;
vmip('c_LightMnfc','LatinAmer') = 49661.125 ;
vmip('c_LightMnfc','EU_28') = 180312.125 ;
vmip('c_LightMnfc','MENA') = 38681.30469 ;
vmip('c_LightMnfc','SSA') = 28706.75195 ;
vmip('c_LightMnfc','RestofWorld') = 53381.29688 ;
vmip('c_HeavyMnfc','Oceania') = 38417.07812 ;
vmip('c_HeavyMnfc','EastAsia') = 322152.2188 ;
vmip('c_HeavyMnfc','SEAsia') = 118519.375 ;
vmip('c_HeavyMnfc','SouthAsia') = 75722.66406 ;
vmip('c_HeavyMnfc','NAmerica') = 314824.5312 ;
vmip('c_HeavyMnfc','LatinAmer') = 105346.3516 ;
vmip('c_HeavyMnfc','EU_28') = 364373.8125 ;
vmip('c_HeavyMnfc','MENA') = 102352.1172 ;
vmip('c_HeavyMnfc','SSA') = 43237.45312 ;
vmip('c_HeavyMnfc','RestofWorld') = 98971 ;
vmip('c_Util_Cons','Oceania') = 394.512085 ;
vmip('c_Util_Cons','EastAsia') = 15997.52734 ;
vmip('c_Util_Cons','SEAsia') = 2912.892334 ;
vmip('c_Util_Cons','SouthAsia') = 718.4580688 ;
vmip('c_Util_Cons','NAmerica') = 3417.132812 ;
vmip('c_Util_Cons','LatinAmer') = 1003.703491 ;
vmip('c_Util_Cons','EU_28') = 36153.19531 ;
vmip('c_Util_Cons','MENA') = 3858.241455 ;
vmip('c_Util_Cons','SSA') = 4201.861816 ;
vmip('c_Util_Cons','RestofWorld') = 14350.42676 ;
vmip('c_TransComm','Oceania') = 103.8053818 ;
vmip('c_TransComm','EastAsia') = 3763.659912 ;
vmip('c_TransComm','SEAsia') = 369.3376465 ;
vmip('c_TransComm','SouthAsia') = 305.3301697 ;
vmip('c_TransComm','NAmerica') = 538.3203125 ;
vmip('c_TransComm','LatinAmer') = 10.16914654 ;
vmip('c_TransComm','EU_28') = 479.0621643 ;
vmip('c_TransComm','MENA') = 1039.849976 ;
vmip('c_TransComm','SSA') = 353.7996826 ;
vmip('c_TransComm','RestofWorld') = 582.5599976 ;
vmip('c_OthService','Oceania') = 1192.833618 ;
vmip('c_OthService','EastAsia') = 18411.35156 ;
vmip('c_OthService','SEAsia') = 168.6327209 ;
vmip('c_OthService','SouthAsia') = 5543.59375 ;
vmip('c_OthService','NAmerica') = 8661.942383 ;
vmip('c_OthService','LatinAmer') = 2447.56665 ;
vmip('c_OthService','EU_28') = 61111.50781 ;
vmip('c_OthService','MENA') = 7576.077148 ;
vmip('c_OthService','SSA') = 3163.713135 ;
vmip('c_OthService','RestofWorld') = 8881.066406 ;

* vst data (10 cells)
vst('c_TransComm','Oceania') = 4106.925781 ;
vst('c_TransComm','EastAsia') = 207902.3906 ;
vst('c_TransComm','SEAsia') = 53095.21484 ;
vst('c_TransComm','SouthAsia') = 15008.07812 ;
vst('c_TransComm','NAmerica') = 50446.21484 ;
vst('c_TransComm','LatinAmer') = 15259.48633 ;
vst('c_TransComm','EU_28') = 331739.125 ;
vst('c_TransComm','MENA') = 37014.25 ;
vst('c_TransComm','SSA') = 5122.219238 ;
vst('c_TransComm','RestofWorld') = 56336.44141 ;

* vxsb data (1000 cells)
vxsb('c_Crops','Oceania','Oceania') = 857.6879883 ;
vxsb('c_Crops','Oceania','EastAsia') = 5490.546875 ;
vxsb('c_Crops','Oceania','SEAsia') = 4003.577148 ;
vxsb('c_Crops','Oceania','SouthAsia') = 957.8133545 ;
vxsb('c_Crops','Oceania','NAmerica') = 335.188324 ;
vxsb('c_Crops','Oceania','LatinAmer') = 47.52339172 ;
vxsb('c_Crops','Oceania','EU_28') = 1428.42395 ;
vxsb('c_Crops','Oceania','MENA') = 2263.045654 ;
vxsb('c_Crops','Oceania','SSA') = 434.6171875 ;
vxsb('c_Crops','Oceania','RestofWorld') = 51.01342392 ;
vxsb('c_Crops','EastAsia','Oceania') = 140.3683472 ;
vxsb('c_Crops','EastAsia','EastAsia') = 3770.420654 ;
vxsb('c_Crops','EastAsia','SEAsia') = 4214.597168 ;
vxsb('c_Crops','EastAsia','SouthAsia') = 780.0739746 ;
vxsb('c_Crops','EastAsia','NAmerica') = 1228.754395 ;
vxsb('c_Crops','EastAsia','LatinAmer') = 419.2499695 ;
vxsb('c_Crops','EastAsia','EU_28') = 1736.469727 ;
vxsb('c_Crops','EastAsia','MENA') = 1018.655334 ;
vxsb('c_Crops','EastAsia','SSA') = 781.7786865 ;
vxsb('c_Crops','EastAsia','RestofWorld') = 994.7698364 ;
vxsb('c_Crops','SEAsia','Oceania') = 533.7235107 ;
vxsb('c_Crops','SEAsia','EastAsia') = 7229.678711 ;
vxsb('c_Crops','SEAsia','SEAsia') = 5605.609375 ;
vxsb('c_Crops','SEAsia','SouthAsia') = 1874.097046 ;
vxsb('c_Crops','SEAsia','NAmerica') = 2794.952637 ;
vxsb('c_Crops','SEAsia','LatinAmer') = 160.3613434 ;
vxsb('c_Crops','SEAsia','EU_28') = 3504.646729 ;
vxsb('c_Crops','SEAsia','MENA') = 1881.696289 ;
vxsb('c_Crops','SEAsia','SSA') = 3432.992432 ;
vxsb('c_Crops','SEAsia','RestofWorld') = 633.1633911 ;
vxsb('c_Crops','SouthAsia','Oceania') = 178.7770844 ;
vxsb('c_Crops','SouthAsia','EastAsia') = 3247.812988 ;
vxsb('c_Crops','SouthAsia','SEAsia') = 3023.703125 ;
vxsb('c_Crops','SouthAsia','SouthAsia') = 3523.373535 ;
vxsb('c_Crops','SouthAsia','NAmerica') = 1336.941528 ;
vxsb('c_Crops','SouthAsia','LatinAmer') = 163.8378296 ;
vxsb('c_Crops','SouthAsia','EU_28') = 2489.459229 ;
vxsb('c_Crops','SouthAsia','MENA') = 6200.868164 ;
vxsb('c_Crops','SouthAsia','SSA') = 1588.62439 ;
vxsb('c_Crops','SouthAsia','RestofWorld') = 997.56427 ;
vxsb('c_Crops','NAmerica','Oceania') = 414.7614441 ;
vxsb('c_Crops','NAmerica','EastAsia') = 39592.67578 ;
vxsb('c_Crops','NAmerica','SEAsia') = 6186.848145 ;
vxsb('c_Crops','NAmerica','SouthAsia') = 3242.923828 ;
vxsb('c_Crops','NAmerica','NAmerica') = 27316.3457 ;
vxsb('c_Crops','NAmerica','LatinAmer') = 7709.272949 ;
vxsb('c_Crops','NAmerica','EU_28') = 7980.296387 ;
vxsb('c_Crops','NAmerica','MENA') = 10940.56543 ;
vxsb('c_Crops','NAmerica','SSA') = 4399.041992 ;
vxsb('c_Crops','NAmerica','RestofWorld') = 823.7841187 ;
vxsb('c_Crops','LatinAmer','Oceania') = 279.023468 ;
vxsb('c_Crops','LatinAmer','EastAsia') = 22874.04883 ;
vxsb('c_Crops','LatinAmer','SEAsia') = 3148.141846 ;
vxsb('c_Crops','LatinAmer','SouthAsia') = 399.9667664 ;
vxsb('c_Crops','LatinAmer','NAmerica') = 15445.34375 ;
vxsb('c_Crops','LatinAmer','LatinAmer') = 9450.181641 ;
vxsb('c_Crops','LatinAmer','EU_28') = 22078.10352 ;
vxsb('c_Crops','LatinAmer','MENA') = 8196.748047 ;
vxsb('c_Crops','LatinAmer','SSA') = 1704.201538 ;
vxsb('c_Crops','LatinAmer','RestofWorld') = 3558.922852 ;
vxsb('c_Crops','EU_28','Oceania') = 128.1779175 ;
vxsb('c_Crops','EU_28','EastAsia') = 979.5698853 ;
vxsb('c_Crops','EU_28','SEAsia') = 322.0588989 ;
vxsb('c_Crops','EU_28','SouthAsia') = 353.0032043 ;
vxsb('c_Crops','EU_28','NAmerica') = 1458.428711 ;
vxsb('c_Crops','EU_28','LatinAmer') = 667.8410034 ;
vxsb('c_Crops','EU_28','EU_28') = 70139.78125 ;
vxsb('c_Crops','EU_28','MENA') = 8884.536133 ;
vxsb('c_Crops','EU_28','SSA') = 2126.794922 ;
vxsb('c_Crops','EU_28','RestofWorld') = 7837.469238 ;
vxsb('c_Crops','MENA','Oceania') = 131.0546265 ;
vxsb('c_Crops','MENA','EastAsia') = 652.4574585 ;
vxsb('c_Crops','MENA','SEAsia') = 318.9515686 ;
vxsb('c_Crops','MENA','SouthAsia') = 575.0499878 ;
vxsb('c_Crops','MENA','NAmerica') = 878.3111572 ;
vxsb('c_Crops','MENA','LatinAmer') = 175.6295929 ;
vxsb('c_Crops','MENA','EU_28') = 6443.415039 ;
vxsb('c_Crops','MENA','MENA') = 6095.941406 ;
vxsb('c_Crops','MENA','SSA') = 543.3104248 ;
vxsb('c_Crops','MENA','RestofWorld') = 3425.030762 ;
vxsb('c_Crops','SSA','Oceania') = 88.20375061 ;
vxsb('c_Crops','SSA','EastAsia') = 2667.998535 ;
vxsb('c_Crops','SSA','SEAsia') = 1988.507812 ;
vxsb('c_Crops','SSA','SouthAsia') = 2002.924316 ;
vxsb('c_Crops','SSA','NAmerica') = 2382.590332 ;
vxsb('c_Crops','SSA','LatinAmer') = 242.6151733 ;
vxsb('c_Crops','SSA','EU_28') = 9532.144531 ;
vxsb('c_Crops','SSA','MENA') = 2072.047607 ;
vxsb('c_Crops','SSA','SSA') = 2348.8396 ;
vxsb('c_Crops','SSA','RestofWorld') = 1109.637939 ;
vxsb('c_Crops','RestofWorld','Oceania') = 10.36020947 ;
vxsb('c_Crops','RestofWorld','EastAsia') = 711.4641113 ;
vxsb('c_Crops','RestofWorld','SEAsia') = 65.27907562 ;
vxsb('c_Crops','RestofWorld','SouthAsia') = 270.0123596 ;
vxsb('c_Crops','RestofWorld','NAmerica') = 119.9371872 ;
vxsb('c_Crops','RestofWorld','LatinAmer') = 100.3630295 ;
vxsb('c_Crops','RestofWorld','EU_28') = 3631.874756 ;
vxsb('c_Crops','RestofWorld','MENA') = 9899.796875 ;
vxsb('c_Crops','RestofWorld','SSA') = 819.1544189 ;
vxsb('c_Crops','RestofWorld','RestofWorld') = 3035.632324 ;
vxsb('c_MeatLstk','Oceania','Oceania') = 607.3534546 ;
vxsb('c_MeatLstk','Oceania','EastAsia') = 8415.358398 ;
vxsb('c_MeatLstk','Oceania','SEAsia') = 1522.231812 ;
vxsb('c_MeatLstk','Oceania','SouthAsia') = 309.6993713 ;
vxsb('c_MeatLstk','Oceania','NAmerica') = 2844.744629 ;
vxsb('c_MeatLstk','Oceania','LatinAmer') = 200.1086884 ;
vxsb('c_MeatLstk','Oceania','EU_28') = 2448.843262 ;
vxsb('c_MeatLstk','Oceania','MENA') = 1249.458008 ;
vxsb('c_MeatLstk','Oceania','SSA') = 159.0433197 ;
vxsb('c_MeatLstk','Oceania','RestofWorld') = 668.5925293 ;
vxsb('c_MeatLstk','EastAsia','Oceania') = 44.17429352 ;
vxsb('c_MeatLstk','EastAsia','EastAsia') = 3209.032471 ;
vxsb('c_MeatLstk','EastAsia','SEAsia') = 524.1375732 ;
vxsb('c_MeatLstk','EastAsia','SouthAsia') = 64.92575073 ;
vxsb('c_MeatLstk','EastAsia','NAmerica') = 711.2947998 ;
vxsb('c_MeatLstk','EastAsia','LatinAmer') = 94.77544403 ;
vxsb('c_MeatLstk','EastAsia','EU_28') = 1578.191895 ;
vxsb('c_MeatLstk','EastAsia','MENA') = 212.1333313 ;
vxsb('c_MeatLstk','EastAsia','SSA') = 87.37538147 ;
vxsb('c_MeatLstk','EastAsia','RestofWorld') = 163.1923676 ;
vxsb('c_MeatLstk','SEAsia','Oceania') = 16.62125778 ;
vxsb('c_MeatLstk','SEAsia','EastAsia') = 1825.650757 ;
vxsb('c_MeatLstk','SEAsia','SEAsia') = 762.9240723 ;
vxsb('c_MeatLstk','SEAsia','SouthAsia') = 21.78966522 ;
vxsb('c_MeatLstk','SEAsia','NAmerica') = 185.9979248 ;
vxsb('c_MeatLstk','SEAsia','LatinAmer') = 9.241361618 ;
vxsb('c_MeatLstk','SEAsia','EU_28') = 1053.550171 ;
vxsb('c_MeatLstk','SEAsia','MENA') = 73.92140198 ;
vxsb('c_MeatLstk','SEAsia','SSA') = 39.79859924 ;
vxsb('c_MeatLstk','SEAsia','RestofWorld') = 31.28232765 ;
vxsb('c_MeatLstk','SouthAsia','Oceania') = 5.95156908 ;
vxsb('c_MeatLstk','SouthAsia','EastAsia') = 83.86643982 ;
vxsb('c_MeatLstk','SouthAsia','SEAsia') = 1230.145508 ;
vxsb('c_MeatLstk','SouthAsia','SouthAsia') = 75.05885315 ;
vxsb('c_MeatLstk','SouthAsia','NAmerica') = 128.8548431 ;
vxsb('c_MeatLstk','SouthAsia','LatinAmer') = 9.224197388 ;
vxsb('c_MeatLstk','SouthAsia','EU_28') = 143.3169861 ;
vxsb('c_MeatLstk','SouthAsia','MENA') = 1406.542236 ;
vxsb('c_MeatLstk','SouthAsia','SSA') = 186.2742004 ;
vxsb('c_MeatLstk','SouthAsia','RestofWorld') = 105.9741516 ;
vxsb('c_MeatLstk','NAmerica','Oceania') = 516.8505249 ;
vxsb('c_MeatLstk','NAmerica','EastAsia') = 12539.63086 ;
vxsb('c_MeatLstk','NAmerica','SEAsia') = 671.8614502 ;
vxsb('c_MeatLstk','NAmerica','SouthAsia') = 121.6927795 ;
vxsb('c_MeatLstk','NAmerica','NAmerica') = 12264.18359 ;
vxsb('c_MeatLstk','NAmerica','LatinAmer') = 1757.15564 ;
vxsb('c_MeatLstk','NAmerica','EU_28') = 1220.632568 ;
vxsb('c_MeatLstk','NAmerica','MENA') = 1177.580444 ;
vxsb('c_MeatLstk','NAmerica','SSA') = 678.7713623 ;
vxsb('c_MeatLstk','NAmerica','RestofWorld') = 1979.045288 ;
vxsb('c_MeatLstk','LatinAmer','Oceania') = 42.98696518 ;
vxsb('c_MeatLstk','LatinAmer','EastAsia') = 3796.107422 ;
vxsb('c_MeatLstk','LatinAmer','SEAsia') = 520.2775269 ;
vxsb('c_MeatLstk','LatinAmer','SouthAsia') = 60.51266098 ;
vxsb('c_MeatLstk','LatinAmer','NAmerica') = 1329.915161 ;
vxsb('c_MeatLstk','LatinAmer','LatinAmer') = 4288.722168 ;
vxsb('c_MeatLstk','LatinAmer','EU_28') = 4302.433594 ;
vxsb('c_MeatLstk','LatinAmer','MENA') = 5324.080078 ;
vxsb('c_MeatLstk','LatinAmer','SSA') = 1122.762451 ;
vxsb('c_MeatLstk','LatinAmer','RestofWorld') = 3025.052246 ;
vxsb('c_MeatLstk','EU_28','Oceania') = 407.2939758 ;
vxsb('c_MeatLstk','EU_28','EastAsia') = 5502.749023 ;
vxsb('c_MeatLstk','EU_28','SEAsia') = 572.1988525 ;
vxsb('c_MeatLstk','EU_28','SouthAsia') = 150.6558228 ;
vxsb('c_MeatLstk','EU_28','NAmerica') = 1141.633545 ;
vxsb('c_MeatLstk','EU_28','LatinAmer') = 393.4352417 ;
vxsb('c_MeatLstk','EU_28','EU_28') = 65049.58203 ;
vxsb('c_MeatLstk','EU_28','MENA') = 3284.537354 ;
vxsb('c_MeatLstk','EU_28','SSA') = 1123.639404 ;
vxsb('c_MeatLstk','EU_28','RestofWorld') = 5745.852539 ;
vxsb('c_MeatLstk','MENA','Oceania') = 13.39898872 ;
vxsb('c_MeatLstk','MENA','EastAsia') = 143.8930206 ;
vxsb('c_MeatLstk','MENA','SEAsia') = 61.90637207 ;
vxsb('c_MeatLstk','MENA','SouthAsia') = 83.0228653 ;
vxsb('c_MeatLstk','MENA','NAmerica') = 108.4807739 ;
vxsb('c_MeatLstk','MENA','LatinAmer') = 20.34193802 ;
vxsb('c_MeatLstk','MENA','EU_28') = 750.2022705 ;
vxsb('c_MeatLstk','MENA','MENA') = 1693.26709 ;
vxsb('c_MeatLstk','MENA','SSA') = 82.75591278 ;
vxsb('c_MeatLstk','MENA','RestofWorld') = 136.1815491 ;
vxsb('c_MeatLstk','SSA','Oceania') = 9.564349174 ;
vxsb('c_MeatLstk','SSA','EastAsia') = 367.7353516 ;
vxsb('c_MeatLstk','SSA','SEAsia') = 34.19633865 ;
vxsb('c_MeatLstk','SSA','SouthAsia') = 102.5603714 ;
vxsb('c_MeatLstk','SSA','NAmerica') = 89.98165894 ;
vxsb('c_MeatLstk','SSA','LatinAmer') = 13.63925076 ;
vxsb('c_MeatLstk','SSA','EU_28') = 433.0752258 ;
vxsb('c_MeatLstk','SSA','MENA') = 789.5368042 ;
vxsb('c_MeatLstk','SSA','SSA') = 548.1322632 ;
vxsb('c_MeatLstk','SSA','RestofWorld') = 79.62109375 ;
vxsb('c_MeatLstk','RestofWorld','Oceania') = 18.73529243 ;
vxsb('c_MeatLstk','RestofWorld','EastAsia') = 196.0582428 ;
vxsb('c_MeatLstk','RestofWorld','SEAsia') = 41.33714294 ;
vxsb('c_MeatLstk','RestofWorld','SouthAsia') = 14.19022942 ;
vxsb('c_MeatLstk','RestofWorld','NAmerica') = 128.8128815 ;
vxsb('c_MeatLstk','RestofWorld','LatinAmer') = 19.0430069 ;
vxsb('c_MeatLstk','RestofWorld','EU_28') = 937.2053833 ;
vxsb('c_MeatLstk','RestofWorld','MENA') = 183.6539001 ;
vxsb('c_MeatLstk','RestofWorld','SSA') = 23.57650757 ;
vxsb('c_MeatLstk','RestofWorld','RestofWorld') = 514.0194702 ;
vxsb('c_Extraction','Oceania','Oceania') = 1189.740845 ;
vxsb('c_Extraction','Oceania','EastAsia') = 115645.8828 ;
vxsb('c_Extraction','Oceania','SEAsia') = 3863.395508 ;
vxsb('c_Extraction','Oceania','SouthAsia') = 11687.2832 ;
vxsb('c_Extraction','Oceania','NAmerica') = 1949.932129 ;
vxsb('c_Extraction','Oceania','LatinAmer') = 1102.846558 ;
vxsb('c_Extraction','Oceania','EU_28') = 5129.215332 ;
vxsb('c_Extraction','Oceania','MENA') = 859.0870972 ;
vxsb('c_Extraction','Oceania','SSA') = 301.1094666 ;
vxsb('c_Extraction','Oceania','RestofWorld') = 198.629837 ;
vxsb('c_Extraction','EastAsia','Oceania') = 1837.029907 ;
vxsb('c_Extraction','EastAsia','EastAsia') = 17040.75977 ;
vxsb('c_Extraction','EastAsia','SEAsia') = 1745.759888 ;
vxsb('c_Extraction','EastAsia','SouthAsia') = 1684.704956 ;
vxsb('c_Extraction','EastAsia','NAmerica') = 941.8405151 ;
vxsb('c_Extraction','EastAsia','LatinAmer') = 93.99806213 ;
vxsb('c_Extraction','EastAsia','EU_28') = 1227.269775 ;
vxsb('c_Extraction','EastAsia','MENA') = 202.9138641 ;
vxsb('c_Extraction','EastAsia','SSA') = 83.77089691 ;
vxsb('c_Extraction','EastAsia','RestofWorld') = 214.0973969 ;
vxsb('c_Extraction','SEAsia','Oceania') = 7102.521484 ;
vxsb('c_Extraction','SEAsia','EastAsia') = 65435.12109 ;
vxsb('c_Extraction','SEAsia','SEAsia') = 14455.05859 ;
vxsb('c_Extraction','SEAsia','SouthAsia') = 8037.496582 ;
vxsb('c_Extraction','SEAsia','NAmerica') = 1753.289307 ;
vxsb('c_Extraction','SEAsia','LatinAmer') = 147.4671021 ;
vxsb('c_Extraction','SEAsia','EU_28') = 2188.511719 ;
vxsb('c_Extraction','SEAsia','MENA') = 676.5067749 ;
vxsb('c_Extraction','SEAsia','SSA') = 53.48975754 ;
vxsb('c_Extraction','SEAsia','RestofWorld') = 257.2657166 ;
vxsb('c_Extraction','SouthAsia','Oceania') = 33.91091537 ;
vxsb('c_Extraction','SouthAsia','EastAsia') = 11419.53125 ;
vxsb('c_Extraction','SouthAsia','SEAsia') = 292.9050598 ;
vxsb('c_Extraction','SouthAsia','SouthAsia') = 579.1072388 ;
vxsb('c_Extraction','SouthAsia','NAmerica') = 230.1117554 ;
vxsb('c_Extraction','SouthAsia','LatinAmer') = 38.3012352 ;
vxsb('c_Extraction','SouthAsia','EU_28') = 1311.504517 ;
vxsb('c_Extraction','SouthAsia','MENA') = 662.4644775 ;
vxsb('c_Extraction','SouthAsia','SSA') = 51.26418686 ;
vxsb('c_Extraction','SouthAsia','RestofWorld') = 57.73341751 ;
vxsb('c_Extraction','NAmerica','Oceania') = 534.2209473 ;
vxsb('c_Extraction','NAmerica','EastAsia') = 21707.61133 ;
vxsb('c_Extraction','NAmerica','SEAsia') = 575.1856689 ;
vxsb('c_Extraction','NAmerica','SouthAsia') = 2172.148193 ;
vxsb('c_Extraction','NAmerica','NAmerica') = 98830.63281 ;
vxsb('c_Extraction','NAmerica','LatinAmer') = 4025.109619 ;
vxsb('c_Extraction','NAmerica','EU_28') = 18100.98242 ;
vxsb('c_Extraction','NAmerica','MENA') = 3209.218018 ;
vxsb('c_Extraction','NAmerica','SSA') = 277.6740112 ;
vxsb('c_Extraction','NAmerica','RestofWorld') = 1727.849121 ;
vxsb('c_Extraction','LatinAmer','Oceania') = 365.3914185 ;
vxsb('c_Extraction','LatinAmer','EastAsia') = 68822.25 ;
vxsb('c_Extraction','LatinAmer','SEAsia') = 2196.509521 ;
vxsb('c_Extraction','LatinAmer','SouthAsia') = 9679.138672 ;
vxsb('c_Extraction','LatinAmer','NAmerica') = 58288.80469 ;
vxsb('c_Extraction','LatinAmer','LatinAmer') = 23818.71484 ;
vxsb('c_Extraction','LatinAmer','EU_28') = 28552.53711 ;
vxsb('c_Extraction','LatinAmer','MENA') = 5069.94043 ;
vxsb('c_Extraction','LatinAmer','SSA') = 480.3503113 ;
vxsb('c_Extraction','LatinAmer','RestofWorld') = 6236.458496 ;
vxsb('c_Extraction','EU_28','Oceania') = 134.8740692 ;
vxsb('c_Extraction','EU_28','EastAsia') = 3666.827637 ;
vxsb('c_Extraction','EU_28','SEAsia') = 496.8589172 ;
vxsb('c_Extraction','EU_28','SouthAsia') = 9436.311523 ;
vxsb('c_Extraction','EU_28','NAmerica') = 2593.459229 ;
vxsb('c_Extraction','EU_28','LatinAmer') = 830.0405273 ;
vxsb('c_Extraction','EU_28','EU_28') = 56597.45312 ;
vxsb('c_Extraction','EU_28','MENA') = 5482.632324 ;
vxsb('c_Extraction','EU_28','SSA') = 1899.906372 ;
vxsb('c_Extraction','EU_28','RestofWorld') = 2737.459717 ;
vxsb('c_Extraction','MENA','Oceania') = 3041.344971 ;
vxsb('c_Extraction','MENA','EastAsia') = 341475.5938 ;
vxsb('c_Extraction','MENA','SEAsia') = 72124.26562 ;
vxsb('c_Extraction','MENA','SouthAsia') = 109340.2734 ;
vxsb('c_Extraction','MENA','NAmerica') = 104883.0859 ;
vxsb('c_Extraction','MENA','LatinAmer') = 5730.236816 ;
vxsb('c_Extraction','MENA','EU_28') = 144164.4219 ;
vxsb('c_Extraction','MENA','MENA') = 37654.22266 ;
vxsb('c_Extraction','MENA','SSA') = 12880.85449 ;
vxsb('c_Extraction','MENA','RestofWorld') = 2324.976562 ;
vxsb('c_Extraction','SSA','Oceania') = 2214.338135 ;
vxsb('c_Extraction','SSA','EastAsia') = 67408.13281 ;
vxsb('c_Extraction','SSA','SEAsia') = 2993.486572 ;
vxsb('c_Extraction','SSA','SouthAsia') = 22677.15039 ;
vxsb('c_Extraction','SSA','NAmerica') = 75819.52344 ;
vxsb('c_Extraction','SSA','LatinAmer') = 10241.16699 ;
vxsb('c_Extraction','SSA','EU_28') = 49731.01953 ;
vxsb('c_Extraction','SSA','MENA') = 1488.817261 ;
vxsb('c_Extraction','SSA','SSA') = 9325.011719 ;
vxsb('c_Extraction','SSA','RestofWorld') = 1902.665894 ;
vxsb('c_Extraction','RestofWorld','Oceania') = 1242.827759 ;
vxsb('c_Extraction','RestofWorld','EastAsia') = 62536.57422 ;
vxsb('c_Extraction','RestofWorld','SEAsia') = 4897.588867 ;
vxsb('c_Extraction','RestofWorld','SouthAsia') = 1965.792114 ;
vxsb('c_Extraction','RestofWorld','NAmerica') = 17822.36133 ;
vxsb('c_Extraction','RestofWorld','LatinAmer') = 2438.023193 ;
vxsb('c_Extraction','RestofWorld','EU_28') = 252529.5312 ;
vxsb('c_Extraction','RestofWorld','MENA') = 17016.54492 ;
vxsb('c_Extraction','RestofWorld','SSA') = 213.9448242 ;
vxsb('c_Extraction','RestofWorld','RestofWorld') = 27717.66797 ;
vxsb('c_ProcFood','Oceania','Oceania') = 3611.03418 ;
vxsb('c_ProcFood','Oceania','EastAsia') = 6502.73584 ;
vxsb('c_ProcFood','Oceania','SEAsia') = 4550.655762 ;
vxsb('c_ProcFood','Oceania','SouthAsia') = 679.8814697 ;
vxsb('c_ProcFood','Oceania','NAmerica') = 2861.521484 ;
vxsb('c_ProcFood','Oceania','LatinAmer') = 807.9478149 ;
vxsb('c_ProcFood','Oceania','EU_28') = 2644.368408 ;
vxsb('c_ProcFood','Oceania','MENA') = 2314.969971 ;
vxsb('c_ProcFood','Oceania','SSA') = 667.6461792 ;
vxsb('c_ProcFood','Oceania','RestofWorld') = 376.641449 ;
vxsb('c_ProcFood','EastAsia','Oceania') = 1502.718872 ;
vxsb('c_ProcFood','EastAsia','EastAsia') = 17321.91797 ;
vxsb('c_ProcFood','EastAsia','SEAsia') = 6898.586914 ;
vxsb('c_ProcFood','EastAsia','SouthAsia') = 584.6611938 ;
vxsb('c_ProcFood','EastAsia','NAmerica') = 8844.964844 ;
vxsb('c_ProcFood','EastAsia','LatinAmer') = 1106.47168 ;
vxsb('c_ProcFood','EastAsia','EU_28') = 5973.013672 ;
vxsb('c_ProcFood','EastAsia','MENA') = 1952.556763 ;
vxsb('c_ProcFood','EastAsia','SSA') = 1630.104614 ;
vxsb('c_ProcFood','EastAsia','RestofWorld') = 1989.023682 ;
vxsb('c_ProcFood','SEAsia','Oceania') = 2466.391602 ;
vxsb('c_ProcFood','SEAsia','EastAsia') = 23082.7168 ;
vxsb('c_ProcFood','SEAsia','SEAsia') = 15680.21094 ;
vxsb('c_ProcFood','SEAsia','SouthAsia') = 11235.41797 ;
vxsb('c_ProcFood','SEAsia','NAmerica') = 12335.33105 ;
vxsb('c_ProcFood','SEAsia','LatinAmer') = 1362.340332 ;
vxsb('c_ProcFood','SEAsia','EU_28') = 8528.629883 ;
vxsb('c_ProcFood','SEAsia','MENA') = 7378.742676 ;
vxsb('c_ProcFood','SEAsia','SSA') = 4470.341309 ;
vxsb('c_ProcFood','SEAsia','RestofWorld') = 2517.982178 ;
vxsb('c_ProcFood','SouthAsia','Oceania') = 186.4442596 ;
vxsb('c_ProcFood','SouthAsia','EastAsia') = 2783.594971 ;
vxsb('c_ProcFood','SouthAsia','SEAsia') = 2321.296387 ;
vxsb('c_ProcFood','SouthAsia','SouthAsia') = 1975.307495 ;
vxsb('c_ProcFood','SouthAsia','NAmerica') = 2639.565918 ;
vxsb('c_ProcFood','SouthAsia','LatinAmer') = 144.4218597 ;
vxsb('c_ProcFood','SouthAsia','EU_28') = 2727.564697 ;
vxsb('c_ProcFood','SouthAsia','MENA') = 2359.476074 ;
vxsb('c_ProcFood','SouthAsia','SSA') = 1109.481812 ;
vxsb('c_ProcFood','SouthAsia','RestofWorld') = 563.0005493 ;
vxsb('c_ProcFood','NAmerica','Oceania') = 1546.806641 ;
vxsb('c_ProcFood','NAmerica','EastAsia') = 14493.78711 ;
vxsb('c_ProcFood','NAmerica','SEAsia') = 4229.942383 ;
vxsb('c_ProcFood','NAmerica','SouthAsia') = 446.8739014 ;
vxsb('c_ProcFood','NAmerica','NAmerica') = 40579.51953 ;
vxsb('c_ProcFood','NAmerica','LatinAmer') = 8097.078613 ;
vxsb('c_ProcFood','NAmerica','EU_28') = 8502.451172 ;
vxsb('c_ProcFood','NAmerica','MENA') = 3903.106201 ;
vxsb('c_ProcFood','NAmerica','SSA') = 984.8707275 ;
vxsb('c_ProcFood','NAmerica','RestofWorld') = 1360.379883 ;
vxsb('c_ProcFood','LatinAmer','Oceania') = 652.0959473 ;
vxsb('c_ProcFood','LatinAmer','EastAsia') = 10325.3418 ;
vxsb('c_ProcFood','LatinAmer','SEAsia') = 6046.696289 ;
vxsb('c_ProcFood','LatinAmer','SouthAsia') = 2622.865967 ;
vxsb('c_ProcFood','LatinAmer','NAmerica') = 12930.61719 ;
vxsb('c_ProcFood','LatinAmer','LatinAmer') = 19159.92383 ;
vxsb('c_ProcFood','LatinAmer','EU_28') = 17422.81445 ;
vxsb('c_ProcFood','LatinAmer','MENA') = 10619.8291 ;
vxsb('c_ProcFood','LatinAmer','SSA') = 8021.084473 ;
vxsb('c_ProcFood','LatinAmer','RestofWorld') = 3077.625977 ;
vxsb('c_ProcFood','EU_28','Oceania') = 3091.61377 ;
vxsb('c_ProcFood','EU_28','EastAsia') = 12838.9541 ;
vxsb('c_ProcFood','EU_28','SEAsia') = 4435.708496 ;
vxsb('c_ProcFood','EU_28','SouthAsia') = 1034.489014 ;
vxsb('c_ProcFood','EU_28','NAmerica') = 23573.82422 ;
vxsb('c_ProcFood','EU_28','LatinAmer') = 4825.026367 ;
vxsb('c_ProcFood','EU_28','EU_28') = 226261.1875 ;
vxsb('c_ProcFood','EU_28','MENA') = 17416.28516 ;
vxsb('c_ProcFood','EU_28','SSA') = 7773.343262 ;
vxsb('c_ProcFood','EU_28','RestofWorld') = 23417.89062 ;
vxsb('c_ProcFood','MENA','Oceania') = 150.6530304 ;
vxsb('c_ProcFood','MENA','EastAsia') = 964.2514648 ;
vxsb('c_ProcFood','MENA','SEAsia') = 597.0473022 ;
vxsb('c_ProcFood','MENA','SouthAsia') = 614.2488403 ;
vxsb('c_ProcFood','MENA','NAmerica') = 1247.991943 ;
vxsb('c_ProcFood','MENA','LatinAmer') = 155.6830902 ;
vxsb('c_ProcFood','MENA','EU_28') = 5488.334961 ;
vxsb('c_ProcFood','MENA','MENA') = 12943.43359 ;
vxsb('c_ProcFood','MENA','SSA') = 2408.246582 ;
vxsb('c_ProcFood','MENA','RestofWorld') = 959.1904907 ;
vxsb('c_ProcFood','SSA','Oceania') = 192.465744 ;
vxsb('c_ProcFood','SSA','EastAsia') = 927.7234497 ;
vxsb('c_ProcFood','SSA','SEAsia') = 240.7167206 ;
vxsb('c_ProcFood','SSA','SouthAsia') = 67.60327148 ;
vxsb('c_ProcFood','SSA','NAmerica') = 994.4949951 ;
vxsb('c_ProcFood','SSA','LatinAmer') = 95.97743225 ;
vxsb('c_ProcFood','SSA','EU_28') = 6345.328613 ;
vxsb('c_ProcFood','SSA','MENA') = 551.8257446 ;
vxsb('c_ProcFood','SSA','SSA') = 5110.853027 ;
vxsb('c_ProcFood','SSA','RestofWorld') = 393.3259583 ;
vxsb('c_ProcFood','RestofWorld','Oceania') = 332.203064 ;
vxsb('c_ProcFood','RestofWorld','EastAsia') = 5483.808594 ;
vxsb('c_ProcFood','RestofWorld','SEAsia') = 461.7635498 ;
vxsb('c_ProcFood','RestofWorld','SouthAsia') = 1000.542358 ;
vxsb('c_ProcFood','RestofWorld','NAmerica') = 2331.922852 ;
vxsb('c_ProcFood','RestofWorld','LatinAmer') = 567.7357788 ;
vxsb('c_ProcFood','RestofWorld','EU_28') = 13116.44238 ;
vxsb('c_ProcFood','RestofWorld','MENA') = 4367.838867 ;
vxsb('c_ProcFood','RestofWorld','SSA') = 947.3112793 ;
vxsb('c_ProcFood','RestofWorld','RestofWorld') = 9568.415039 ;
vxsb('c_TextWapp','Oceania','Oceania') = 620.7765503 ;
vxsb('c_TextWapp','Oceania','EastAsia') = 471.4954834 ;
vxsb('c_TextWapp','Oceania','SEAsia') = 155.5696716 ;
vxsb('c_TextWapp','Oceania','SouthAsia') = 142.7815857 ;
vxsb('c_TextWapp','Oceania','NAmerica') = 152.8041687 ;
vxsb('c_TextWapp','Oceania','LatinAmer') = 13.71626854 ;
vxsb('c_TextWapp','Oceania','EU_28') = 321.4785767 ;
vxsb('c_TextWapp','Oceania','MENA') = 67.31161499 ;
vxsb('c_TextWapp','Oceania','SSA') = 34.15288925 ;
vxsb('c_TextWapp','Oceania','RestofWorld') = 17.2900486 ;
vxsb('c_TextWapp','EastAsia','Oceania') = 5897.344238 ;
vxsb('c_TextWapp','EastAsia','EastAsia') = 68326.22656 ;
vxsb('c_TextWapp','EastAsia','SEAsia') = 27706.09766 ;
vxsb('c_TextWapp','EastAsia','SouthAsia') = 11559.75586 ;
vxsb('c_TextWapp','EastAsia','NAmerica') = 42348.64062 ;
vxsb('c_TextWapp','EastAsia','LatinAmer') = 15161.20312 ;
vxsb('c_TextWapp','EastAsia','EU_28') = 54985.98438 ;
vxsb('c_TextWapp','EastAsia','MENA') = 22527.93555 ;
vxsb('c_TextWapp','EastAsia','SSA') = 9391.805664 ;
vxsb('c_TextWapp','EastAsia','RestofWorld') = 16899.05078 ;
vxsb('c_TextWapp','SEAsia','Oceania') = 584.2180786 ;
vxsb('c_TextWapp','SEAsia','EastAsia') = 10427.15527 ;
vxsb('c_TextWapp','SEAsia','SEAsia') = 4193.651855 ;
vxsb('c_TextWapp','SEAsia','SouthAsia') = 1483.000854 ;
vxsb('c_TextWapp','SEAsia','NAmerica') = 21605.95898 ;
vxsb('c_TextWapp','SEAsia','LatinAmer') = 1410.737671 ;
vxsb('c_TextWapp','SEAsia','EU_28') = 10642.04492 ;
vxsb('c_TextWapp','SEAsia','MENA') = 3412.117188 ;
vxsb('c_TextWapp','SEAsia','SSA') = 575.1425781 ;
vxsb('c_TextWapp','SEAsia','RestofWorld') = 791.3001099 ;
vxsb('c_TextWapp','SouthAsia','Oceania') = 886.1143188 ;
vxsb('c_TextWapp','SouthAsia','EastAsia') = 4560.464844 ;
vxsb('c_TextWapp','SouthAsia','SEAsia') = 1204.832642 ;
vxsb('c_TextWapp','SouthAsia','SouthAsia') = 2758.088867 ;
vxsb('c_TextWapp','SouthAsia','NAmerica') = 19438.44531 ;
vxsb('c_TextWapp','SouthAsia','LatinAmer') = 2269.344971 ;
vxsb('c_TextWapp','SouthAsia','EU_28') = 32669.43945 ;
vxsb('c_TextWapp','SouthAsia','MENA') = 7635.174316 ;
vxsb('c_TextWapp','SouthAsia','SSA') = 2017.94104 ;
vxsb('c_TextWapp','SouthAsia','RestofWorld') = 1770.676758 ;
vxsb('c_TextWapp','NAmerica','Oceania') = 390.7398682 ;
vxsb('c_TextWapp','NAmerica','EastAsia') = 2963.249756 ;
vxsb('c_TextWapp','NAmerica','SEAsia') = 583.0706787 ;
vxsb('c_TextWapp','NAmerica','SouthAsia') = 306.0950317 ;
vxsb('c_TextWapp','NAmerica','NAmerica') = 15176.08691 ;
vxsb('c_TextWapp','NAmerica','LatinAmer') = 5451.807617 ;
vxsb('c_TextWapp','NAmerica','EU_28') = 2911.705078 ;
vxsb('c_TextWapp','NAmerica','MENA') = 867.3200684 ;
vxsb('c_TextWapp','NAmerica','SSA') = 252.4201202 ;
vxsb('c_TextWapp','NAmerica','RestofWorld') = 434.6053467 ;
vxsb('c_TextWapp','LatinAmer','Oceania') = 71.43756866 ;
vxsb('c_TextWapp','LatinAmer','EastAsia') = 450.2047119 ;
vxsb('c_TextWapp','LatinAmer','SEAsia') = 100.1183853 ;
vxsb('c_TextWapp','LatinAmer','SouthAsia') = 55.00000381 ;
vxsb('c_TextWapp','LatinAmer','NAmerica') = 11404.56445 ;
vxsb('c_TextWapp','LatinAmer','LatinAmer') = 5315.564453 ;
vxsb('c_TextWapp','LatinAmer','EU_28') = 1341.769897 ;
vxsb('c_TextWapp','LatinAmer','MENA') = 205.1480255 ;
vxsb('c_TextWapp','LatinAmer','SSA') = 114.2206345 ;
vxsb('c_TextWapp','LatinAmer','RestofWorld') = 103.8677368 ;
vxsb('c_TextWapp','EU_28','Oceania') = 819.2094727 ;
vxsb('c_TextWapp','EU_28','EastAsia') = 8084.403809 ;
vxsb('c_TextWapp','EU_28','SEAsia') = 1487.235107 ;
vxsb('c_TextWapp','EU_28','SouthAsia') = 981.7681885 ;
vxsb('c_TextWapp','EU_28','NAmerica') = 7925.514648 ;
vxsb('c_TextWapp','EU_28','LatinAmer') = 1766.706055 ;
vxsb('c_TextWapp','EU_28','EU_28') = 113921.3359 ;
vxsb('c_TextWapp','EU_28','MENA') = 11343.96973 ;
vxsb('c_TextWapp','EU_28','SSA') = 1499.849365 ;
vxsb('c_TextWapp','EU_28','RestofWorld') = 16307.05371 ;
vxsb('c_TextWapp','MENA','Oceania') = 181.2555237 ;
vxsb('c_TextWapp','MENA','EastAsia') = 1075.013184 ;
vxsb('c_TextWapp','MENA','SEAsia') = 268.3790894 ;
vxsb('c_TextWapp','MENA','SouthAsia') = 386.7034607 ;
vxsb('c_TextWapp','MENA','NAmerica') = 5462.172363 ;
vxsb('c_TextWapp','MENA','LatinAmer') = 431.587738 ;
vxsb('c_TextWapp','MENA','EU_28') = 28263.84375 ;
vxsb('c_TextWapp','MENA','MENA') = 6108.591309 ;
vxsb('c_TextWapp','MENA','SSA') = 664.2819214 ;
vxsb('c_TextWapp','MENA','RestofWorld') = 3436.77832 ;
vxsb('c_TextWapp','SSA','Oceania') = 52.27815628 ;
vxsb('c_TextWapp','SSA','EastAsia') = 222.9984436 ;
vxsb('c_TextWapp','SSA','SEAsia') = 70.6481781 ;
vxsb('c_TextWapp','SSA','SouthAsia') = 37.38599777 ;
vxsb('c_TextWapp','SSA','NAmerica') = 1167.877808 ;
vxsb('c_TextWapp','SSA','LatinAmer') = 49.13191986 ;
vxsb('c_TextWapp','SSA','EU_28') = 1415.650269 ;
vxsb('c_TextWapp','SSA','MENA') = 122.929718 ;
vxsb('c_TextWapp','SSA','SSA') = 979.6954956 ;
vxsb('c_TextWapp','SSA','RestofWorld') = 71.919487 ;
vxsb('c_TextWapp','RestofWorld','Oceania') = 28.63707542 ;
vxsb('c_TextWapp','RestofWorld','EastAsia') = 435.8896179 ;
vxsb('c_TextWapp','RestofWorld','SEAsia') = 88.61375427 ;
vxsb('c_TextWapp','RestofWorld','SouthAsia') = 74.67758942 ;
vxsb('c_TextWapp','RestofWorld','NAmerica') = 390.2450562 ;
vxsb('c_TextWapp','RestofWorld','LatinAmer') = 48.74290466 ;
vxsb('c_TextWapp','RestofWorld','EU_28') = 5187.274414 ;
vxsb('c_TextWapp','RestofWorld','MENA') = 742.3895874 ;
vxsb('c_TextWapp','RestofWorld','SSA') = 61.84391785 ;
vxsb('c_TextWapp','RestofWorld','RestofWorld') = 1840.752197 ;
vxsb('c_LightMnfc','Oceania','Oceania') = 3624.661377 ;
vxsb('c_LightMnfc','Oceania','EastAsia') = 3948.310547 ;
vxsb('c_LightMnfc','Oceania','SEAsia') = 1698.577515 ;
vxsb('c_LightMnfc','Oceania','SouthAsia') = 279.614502 ;
vxsb('c_LightMnfc','Oceania','NAmerica') = 2561.078857 ;
vxsb('c_LightMnfc','Oceania','LatinAmer') = 292.4028015 ;
vxsb('c_LightMnfc','Oceania','EU_28') = 2021.285034 ;
vxsb('c_LightMnfc','Oceania','MENA') = 1532.960571 ;
vxsb('c_LightMnfc','Oceania','SSA') = 502.9772644 ;
vxsb('c_LightMnfc','Oceania','RestofWorld') = 455.6325073 ;
vxsb('c_LightMnfc','EastAsia','Oceania') = 35213.38281 ;
vxsb('c_LightMnfc','EastAsia','EastAsia') = 108568.2266 ;
vxsb('c_LightMnfc','EastAsia','SEAsia') = 53532.96094 ;
vxsb('c_LightMnfc','EastAsia','SouthAsia') = 17915.95508 ;
vxsb('c_LightMnfc','EastAsia','NAmerica') = 213383.2969 ;
vxsb('c_LightMnfc','EastAsia','LatinAmer') = 46450.90625 ;
vxsb('c_LightMnfc','EastAsia','EU_28') = 147048.5312 ;
vxsb('c_LightMnfc','EastAsia','MENA') = 57313.11719 ;
vxsb('c_LightMnfc','EastAsia','SSA') = 33675.50781 ;
vxsb('c_LightMnfc','EastAsia','RestofWorld') = 44855.58594 ;
vxsb('c_LightMnfc','SEAsia','Oceania') = 8012.33252 ;
vxsb('c_LightMnfc','SEAsia','EastAsia') = 25082.12695 ;
vxsb('c_LightMnfc','SEAsia','SEAsia') = 24152.38867 ;
vxsb('c_LightMnfc','SEAsia','SouthAsia') = 4472.99707 ;
vxsb('c_LightMnfc','SEAsia','NAmerica') = 17985.24023 ;
vxsb('c_LightMnfc','SEAsia','LatinAmer') = 4317.989258 ;
vxsb('c_LightMnfc','SEAsia','EU_28') = 21991.66602 ;
vxsb('c_LightMnfc','SEAsia','MENA') = 9791.03418 ;
vxsb('c_LightMnfc','SEAsia','SSA') = 2412.023926 ;
vxsb('c_LightMnfc','SEAsia','RestofWorld') = 3307.264648 ;
vxsb('c_LightMnfc','SouthAsia','Oceania') = 795.638916 ;
vxsb('c_LightMnfc','SouthAsia','EastAsia') = 3587.044434 ;
vxsb('c_LightMnfc','SouthAsia','SEAsia') = 2991.14624 ;
vxsb('c_LightMnfc','SouthAsia','SouthAsia') = 2070.814697 ;
vxsb('c_LightMnfc','SouthAsia','NAmerica') = 12464.30078 ;
vxsb('c_LightMnfc','SouthAsia','LatinAmer') = 1427.493286 ;
vxsb('c_LightMnfc','SouthAsia','EU_28') = 13818.80273 ;
vxsb('c_LightMnfc','SouthAsia','MENA') = 23961.40234 ;
vxsb('c_LightMnfc','SouthAsia','SSA') = 3001.400146 ;
vxsb('c_LightMnfc','SouthAsia','RestofWorld') = 1113.604858 ;
vxsb('c_LightMnfc','NAmerica','Oceania') = 8234.109375 ;
vxsb('c_LightMnfc','NAmerica','EastAsia') = 59042.85938 ;
vxsb('c_LightMnfc','NAmerica','SEAsia') = 16134.11523 ;
vxsb('c_LightMnfc','NAmerica','SouthAsia') = 7314.211914 ;
vxsb('c_LightMnfc','NAmerica','NAmerica') = 271203.75 ;
vxsb('c_LightMnfc','NAmerica','LatinAmer') = 32017.36523 ;
vxsb('c_LightMnfc','NAmerica','EU_28') = 80890.57031 ;
vxsb('c_LightMnfc','NAmerica','MENA') = 26154.94141 ;
vxsb('c_LightMnfc','NAmerica','SSA') = 7985.770996 ;
vxsb('c_LightMnfc','NAmerica','RestofWorld') = 9813.068359 ;
vxsb('c_LightMnfc','LatinAmer','Oceania') = 682.9022217 ;
vxsb('c_LightMnfc','LatinAmer','EastAsia') = 8561.951172 ;
vxsb('c_LightMnfc','LatinAmer','SEAsia') = 1280.130859 ;
vxsb('c_LightMnfc','LatinAmer','SouthAsia') = 364.602478 ;
vxsb('c_LightMnfc','LatinAmer','NAmerica') = 12349.9209 ;
vxsb('c_LightMnfc','LatinAmer','LatinAmer') = 32967.8125 ;
vxsb('c_LightMnfc','LatinAmer','EU_28') = 13048.3125 ;
vxsb('c_LightMnfc','LatinAmer','MENA') = 1737.084595 ;
vxsb('c_LightMnfc','LatinAmer','SSA') = 2306.498535 ;
vxsb('c_LightMnfc','LatinAmer','RestofWorld') = 1347.962769 ;
vxsb('c_LightMnfc','EU_28','Oceania') = 16309.80957 ;
vxsb('c_LightMnfc','EU_28','EastAsia') = 97852.74219 ;
vxsb('c_LightMnfc','EU_28','SEAsia') = 20849.26367 ;
vxsb('c_LightMnfc','EU_28','SouthAsia') = 10929.43652 ;
vxsb('c_LightMnfc','EU_28','NAmerica') = 115945.2578 ;
vxsb('c_LightMnfc','EU_28','LatinAmer') = 28741.46289 ;
vxsb('c_LightMnfc','EU_28','EU_28') = 880833.25 ;
vxsb('c_LightMnfc','EU_28','MENA') = 83064.73438 ;
vxsb('c_LightMnfc','EU_28','SSA') = 22452.75781 ;
vxsb('c_LightMnfc','EU_28','RestofWorld') = 120502.0781 ;
vxsb('c_LightMnfc','MENA','Oceania') = 507.7640076 ;
vxsb('c_LightMnfc','MENA','EastAsia') = 2569.60791 ;
vxsb('c_LightMnfc','MENA','SEAsia') = 846.6627808 ;
vxsb('c_LightMnfc','MENA','SouthAsia') = 5249.425293 ;
vxsb('c_LightMnfc','MENA','NAmerica') = 13604.16992 ;
vxsb('c_LightMnfc','MENA','LatinAmer') = 1031.096924 ;
vxsb('c_LightMnfc','MENA','EU_28') = 28386.47461 ;
vxsb('c_LightMnfc','MENA','MENA') = 16853.86523 ;
vxsb('c_LightMnfc','MENA','SSA') = 3156.610352 ;
vxsb('c_LightMnfc','MENA','RestofWorld') = 6742.087891 ;
vxsb('c_LightMnfc','SSA','Oceania') = 374.4905396 ;
vxsb('c_LightMnfc','SSA','EastAsia') = 2150.90625 ;
vxsb('c_LightMnfc','SSA','SEAsia') = 778.7339478 ;
vxsb('c_LightMnfc','SSA','SouthAsia') = 300.8003235 ;
vxsb('c_LightMnfc','SSA','NAmerica') = 4473.604492 ;
vxsb('c_LightMnfc','SSA','LatinAmer') = 337.7579041 ;
vxsb('c_LightMnfc','SSA','EU_28') = 7215.873535 ;
vxsb('c_LightMnfc','SSA','MENA') = 806.8292236 ;
vxsb('c_LightMnfc','SSA','SSA') = 6976.281738 ;
vxsb('c_LightMnfc','SSA','RestofWorld') = 388.3262329 ;
vxsb('c_LightMnfc','RestofWorld','Oceania') = 366.84552 ;
vxsb('c_LightMnfc','RestofWorld','EastAsia') = 5060.149902 ;
vxsb('c_LightMnfc','RestofWorld','SEAsia') = 1278.403687 ;
vxsb('c_LightMnfc','RestofWorld','SouthAsia') = 2188.961182 ;
vxsb('c_LightMnfc','RestofWorld','NAmerica') = 5490.875488 ;
vxsb('c_LightMnfc','RestofWorld','LatinAmer') = 1362.633301 ;
vxsb('c_LightMnfc','RestofWorld','EU_28') = 32024.76172 ;
vxsb('c_LightMnfc','RestofWorld','MENA') = 5486.276855 ;
vxsb('c_LightMnfc','RestofWorld','SSA') = 662.2855225 ;
vxsb('c_LightMnfc','RestofWorld','RestofWorld') = 16007.33496 ;
vxsb('c_HeavyMnfc','Oceania','Oceania') = 10926.00488 ;
vxsb('c_HeavyMnfc','Oceania','EastAsia') = 20931.95508 ;
vxsb('c_HeavyMnfc','Oceania','SEAsia') = 11485.28418 ;
vxsb('c_HeavyMnfc','Oceania','SouthAsia') = 4749.352051 ;
vxsb('c_HeavyMnfc','Oceania','NAmerica') = 6905.460449 ;
vxsb('c_HeavyMnfc','Oceania','LatinAmer') = 1118.119873 ;
vxsb('c_HeavyMnfc','Oceania','EU_28') = 6320.167969 ;
vxsb('c_HeavyMnfc','Oceania','MENA') = 1613.489868 ;
vxsb('c_HeavyMnfc','Oceania','SSA') = 1767.101562 ;
vxsb('c_HeavyMnfc','Oceania','RestofWorld') = 954.5891724 ;
vxsb('c_HeavyMnfc','EastAsia','Oceania') = 46446.84766 ;
vxsb('c_HeavyMnfc','EastAsia','EastAsia') = 811484 ;
vxsb('c_HeavyMnfc','EastAsia','SEAsia') = 267817.0312 ;
vxsb('c_HeavyMnfc','EastAsia','SouthAsia') = 78708.10156 ;
vxsb('c_HeavyMnfc','EastAsia','NAmerica') = 461945.9375 ;
vxsb('c_HeavyMnfc','EastAsia','LatinAmer') = 82903.33594 ;
vxsb('c_HeavyMnfc','EastAsia','EU_28') = 345630.5312 ;
vxsb('c_HeavyMnfc','EastAsia','MENA') = 102176.8047 ;
vxsb('c_HeavyMnfc','EastAsia','SSA') = 35275.83984 ;
vxsb('c_HeavyMnfc','EastAsia','RestofWorld') = 61973.70312 ;
vxsb('c_HeavyMnfc','SEAsia','Oceania') = 23092.39844 ;
vxsb('c_HeavyMnfc','SEAsia','EastAsia') = 204724.375 ;
vxsb('c_HeavyMnfc','SEAsia','SEAsia') = 156203.6719 ;
vxsb('c_HeavyMnfc','SEAsia','SouthAsia') = 24347.80469 ;
vxsb('c_HeavyMnfc','SEAsia','NAmerica') = 82678.57031 ;
vxsb('c_HeavyMnfc','SEAsia','LatinAmer') = 9365.753906 ;
vxsb('c_HeavyMnfc','SEAsia','EU_28') = 70813.42969 ;
vxsb('c_HeavyMnfc','SEAsia','MENA') = 17973.8125 ;
vxsb('c_HeavyMnfc','SEAsia','SSA') = 6033.11377 ;
vxsb('c_HeavyMnfc','SEAsia','RestofWorld') = 9789.817383 ;
vxsb('c_HeavyMnfc','SouthAsia','Oceania') = 1151.938354 ;
vxsb('c_HeavyMnfc','SouthAsia','EastAsia') = 15211.12891 ;
vxsb('c_HeavyMnfc','SouthAsia','SEAsia') = 10457.625 ;
vxsb('c_HeavyMnfc','SouthAsia','SouthAsia') = 8371.325195 ;
vxsb('c_HeavyMnfc','SouthAsia','NAmerica') = 17504.71484 ;
vxsb('c_HeavyMnfc','SouthAsia','LatinAmer') = 8573.259766 ;
vxsb('c_HeavyMnfc','SouthAsia','EU_28') = 25690.10352 ;
vxsb('c_HeavyMnfc','SouthAsia','MENA') = 21458.72656 ;
vxsb('c_HeavyMnfc','SouthAsia','SSA') = 12077.39355 ;
vxsb('c_HeavyMnfc','SouthAsia','RestofWorld') = 3366.892334 ;
vxsb('c_HeavyMnfc','NAmerica','Oceania') = 25028.82227 ;
vxsb('c_HeavyMnfc','NAmerica','EastAsia') = 182583.0312 ;
vxsb('c_HeavyMnfc','NAmerica','SEAsia') = 54853.75781 ;
vxsb('c_HeavyMnfc','NAmerica','SouthAsia') = 18445.79883 ;
vxsb('c_HeavyMnfc','NAmerica','NAmerica') = 459817.4062 ;
vxsb('c_HeavyMnfc','NAmerica','LatinAmer') = 128034.6562 ;
vxsb('c_HeavyMnfc','NAmerica','EU_28') = 240090.9375 ;
vxsb('c_HeavyMnfc','NAmerica','MENA') = 51724.77344 ;
vxsb('c_HeavyMnfc','NAmerica','SSA') = 17768.65039 ;
vxsb('c_HeavyMnfc','NAmerica','RestofWorld') = 38527.96484 ;
vxsb('c_HeavyMnfc','LatinAmer','Oceania') = 1507.458862 ;
vxsb('c_HeavyMnfc','LatinAmer','EastAsia') = 34667.78906 ;
vxsb('c_HeavyMnfc','LatinAmer','SEAsia') = 5574.958984 ;
vxsb('c_HeavyMnfc','LatinAmer','SouthAsia') = 1204.704346 ;
vxsb('c_HeavyMnfc','LatinAmer','NAmerica') = 77666.88281 ;
vxsb('c_HeavyMnfc','LatinAmer','LatinAmer') = 66611.11719 ;
vxsb('c_HeavyMnfc','LatinAmer','EU_28') = 28589.63867 ;
vxsb('c_HeavyMnfc','LatinAmer','MENA') = 3676.300049 ;
vxsb('c_HeavyMnfc','LatinAmer','SSA') = 4229.492676 ;
vxsb('c_HeavyMnfc','LatinAmer','RestofWorld') = 11695.46973 ;
vxsb('c_HeavyMnfc','EU_28','Oceania') = 29388.625 ;
vxsb('c_HeavyMnfc','EU_28','EastAsia') = 213320.4375 ;
vxsb('c_HeavyMnfc','EU_28','SEAsia') = 53294.18359 ;
vxsb('c_HeavyMnfc','EU_28','SouthAsia') = 41509.65234 ;
vxsb('c_HeavyMnfc','EU_28','NAmerica') = 275185.25 ;
vxsb('c_HeavyMnfc','EU_28','LatinAmer') = 69760.22656 ;
vxsb('c_HeavyMnfc','EU_28','EU_28') = 1715428.375 ;
vxsb('c_HeavyMnfc','EU_28','MENA') = 196337.0781 ;
vxsb('c_HeavyMnfc','EU_28','SSA') = 53518.43359 ;
vxsb('c_HeavyMnfc','EU_28','RestofWorld') = 263154.6562 ;
vxsb('c_HeavyMnfc','MENA','Oceania') = 3055.361084 ;
vxsb('c_HeavyMnfc','MENA','EastAsia') = 74143.77344 ;
vxsb('c_HeavyMnfc','MENA','SEAsia') = 22686.95898 ;
vxsb('c_HeavyMnfc','MENA','SouthAsia') = 38286.52734 ;
vxsb('c_HeavyMnfc','MENA','NAmerica') = 27540.97266 ;
vxsb('c_HeavyMnfc','MENA','LatinAmer') = 8532.193359 ;
vxsb('c_HeavyMnfc','MENA','EU_28') = 86384.49219 ;
vxsb('c_HeavyMnfc','MENA','MENA') = 82764.72656 ;
vxsb('c_HeavyMnfc','MENA','SSA') = 17644.72266 ;
vxsb('c_HeavyMnfc','MENA','RestofWorld') = 14097.13086 ;
vxsb('c_HeavyMnfc','SSA','Oceania') = 511.1831665 ;
vxsb('c_HeavyMnfc','SSA','EastAsia') = 15255.2627 ;
vxsb('c_HeavyMnfc','SSA','SEAsia') = 3253.594727 ;
vxsb('c_HeavyMnfc','SSA','SouthAsia') = 8704.395508 ;
vxsb('c_HeavyMnfc','SSA','NAmerica') = 9026.160156 ;
vxsb('c_HeavyMnfc','SSA','LatinAmer') = 1131.809326 ;
vxsb('c_HeavyMnfc','SSA','EU_28') = 16480.94727 ;
vxsb('c_HeavyMnfc','SSA','MENA') = 5516.495605 ;
vxsb('c_HeavyMnfc','SSA','SSA') = 16820.04883 ;
vxsb('c_HeavyMnfc','SSA','RestofWorld') = 12003.28613 ;
vxsb('c_HeavyMnfc','RestofWorld','Oceania') = 3427.581299 ;
vxsb('c_HeavyMnfc','RestofWorld','EastAsia') = 56194.54297 ;
vxsb('c_HeavyMnfc','RestofWorld','SEAsia') = 20867.41016 ;
vxsb('c_HeavyMnfc','RestofWorld','SouthAsia') = 38152.34375 ;
vxsb('c_HeavyMnfc','RestofWorld','NAmerica') = 52589.57422 ;
vxsb('c_HeavyMnfc','RestofWorld','LatinAmer') = 13279.375 ;
vxsb('c_HeavyMnfc','RestofWorld','EU_28') = 219696.2656 ;
vxsb('c_HeavyMnfc','RestofWorld','MENA') = 44340.29688 ;
vxsb('c_HeavyMnfc','RestofWorld','SSA') = 4337.209961 ;
vxsb('c_HeavyMnfc','RestofWorld','RestofWorld') = 49680.63672 ;
vxsb('c_Util_Cons','Oceania','Oceania') = 31.07725143 ;
vxsb('c_Util_Cons','Oceania','EastAsia') = 201.0083771 ;
vxsb('c_Util_Cons','Oceania','SEAsia') = 101.6341858 ;
vxsb('c_Util_Cons','Oceania','SouthAsia') = 11.61959457 ;
vxsb('c_Util_Cons','Oceania','NAmerica') = 79.40089417 ;
vxsb('c_Util_Cons','Oceania','LatinAmer') = 49.84052277 ;
vxsb('c_Util_Cons','Oceania','EU_28') = 383.8754883 ;
vxsb('c_Util_Cons','Oceania','MENA') = 127.0182343 ;
vxsb('c_Util_Cons','Oceania','SSA') = 23.77604485 ;
vxsb('c_Util_Cons','Oceania','RestofWorld') = 78.74828339 ;
vxsb('c_Util_Cons','EastAsia','Oceania') = 366.0947571 ;
vxsb('c_Util_Cons','EastAsia','EastAsia') = 6357.323242 ;
vxsb('c_Util_Cons','EastAsia','SEAsia') = 2104.691406 ;
vxsb('c_Util_Cons','EastAsia','SouthAsia') = 451.4074097 ;
vxsb('c_Util_Cons','EastAsia','NAmerica') = 2141.272949 ;
vxsb('c_Util_Cons','EastAsia','LatinAmer') = 486.0794067 ;
vxsb('c_Util_Cons','EastAsia','EU_28') = 11890.67676 ;
vxsb('c_Util_Cons','EastAsia','MENA') = 4096.105469 ;
vxsb('c_Util_Cons','EastAsia','SSA') = 1524.092529 ;
vxsb('c_Util_Cons','EastAsia','RestofWorld') = 4494.487305 ;
vxsb('c_Util_Cons','SEAsia','Oceania') = 97.43195343 ;
vxsb('c_Util_Cons','SEAsia','EastAsia') = 1373.67334 ;
vxsb('c_Util_Cons','SEAsia','SEAsia') = 382.1662903 ;
vxsb('c_Util_Cons','SEAsia','SouthAsia') = 107.3237152 ;
vxsb('c_Util_Cons','SEAsia','NAmerica') = 393.7479858 ;
vxsb('c_Util_Cons','SEAsia','LatinAmer') = 142.2795258 ;
vxsb('c_Util_Cons','SEAsia','EU_28') = 2745.166992 ;
vxsb('c_Util_Cons','SEAsia','MENA') = 812.9971924 ;
vxsb('c_Util_Cons','SEAsia','SSA') = 236.680542 ;
vxsb('c_Util_Cons','SEAsia','RestofWorld') = 861.5371094 ;
vxsb('c_Util_Cons','SouthAsia','Oceania') = 19.71509743 ;
vxsb('c_Util_Cons','SouthAsia','EastAsia') = 237.1269989 ;
vxsb('c_Util_Cons','SouthAsia','SEAsia') = 93.98284149 ;
vxsb('c_Util_Cons','SouthAsia','SouthAsia') = 10.60282135 ;
vxsb('c_Util_Cons','SouthAsia','NAmerica') = 61.55842209 ;
vxsb('c_Util_Cons','SouthAsia','LatinAmer') = 52.26968765 ;
vxsb('c_Util_Cons','SouthAsia','EU_28') = 802.0194092 ;
vxsb('c_Util_Cons','SouthAsia','MENA') = 255.4254608 ;
vxsb('c_Util_Cons','SouthAsia','SSA') = 61.51704788 ;
vxsb('c_Util_Cons','SouthAsia','RestofWorld') = 171.6964722 ;
vxsb('c_Util_Cons','NAmerica','Oceania') = 255.6985321 ;
vxsb('c_Util_Cons','NAmerica','EastAsia') = 2842.396729 ;
vxsb('c_Util_Cons','NAmerica','SEAsia') = 829.7210083 ;
vxsb('c_Util_Cons','NAmerica','SouthAsia') = 149.27742 ;
vxsb('c_Util_Cons','NAmerica','NAmerica') = 3845.733887 ;
vxsb('c_Util_Cons','NAmerica','LatinAmer') = 357.4509888 ;
vxsb('c_Util_Cons','NAmerica','EU_28') = 3967.249756 ;
vxsb('c_Util_Cons','NAmerica','MENA') = 1951.361328 ;
vxsb('c_Util_Cons','NAmerica','SSA') = 511.8539124 ;
vxsb('c_Util_Cons','NAmerica','RestofWorld') = 1926.356934 ;
vxsb('c_Util_Cons','LatinAmer','Oceania') = 46.47725296 ;
vxsb('c_Util_Cons','LatinAmer','EastAsia') = 322.9883423 ;
vxsb('c_Util_Cons','LatinAmer','SEAsia') = 206.2909546 ;
vxsb('c_Util_Cons','LatinAmer','SouthAsia') = 54.46353149 ;
vxsb('c_Util_Cons','LatinAmer','NAmerica') = 209.4647217 ;
vxsb('c_Util_Cons','LatinAmer','LatinAmer') = 1090.684814 ;
vxsb('c_Util_Cons','LatinAmer','EU_28') = 1094.125122 ;
vxsb('c_Util_Cons','LatinAmer','MENA') = 422.6958618 ;
vxsb('c_Util_Cons','LatinAmer','SSA') = 186.4594421 ;
vxsb('c_Util_Cons','LatinAmer','RestofWorld') = 532.9012451 ;
vxsb('c_Util_Cons','EU_28','Oceania') = 525.5808716 ;
vxsb('c_Util_Cons','EU_28','EastAsia') = 7820.53125 ;
vxsb('c_Util_Cons','EU_28','SEAsia') = 3339.02002 ;
vxsb('c_Util_Cons','EU_28','SouthAsia') = 862.8706665 ;
vxsb('c_Util_Cons','EU_28','NAmerica') = 1946.275146 ;
vxsb('c_Util_Cons','EU_28','LatinAmer') = 2228.263184 ;
vxsb('c_Util_Cons','EU_28','EU_28') = 50810.94531 ;
vxsb('c_Util_Cons','EU_28','MENA') = 7784.186035 ;
vxsb('c_Util_Cons','EU_28','SSA') = 2911.449707 ;
vxsb('c_Util_Cons','EU_28','RestofWorld') = 14605.10449 ;
vxsb('c_Util_Cons','MENA','Oceania') = 315.7593079 ;
vxsb('c_Util_Cons','MENA','EastAsia') = 1619.862305 ;
vxsb('c_Util_Cons','MENA','SEAsia') = 778.6488037 ;
vxsb('c_Util_Cons','MENA','SouthAsia') = 196.4408264 ;
vxsb('c_Util_Cons','MENA','NAmerica') = 1185.112549 ;
vxsb('c_Util_Cons','MENA','LatinAmer') = 669.9071655 ;
vxsb('c_Util_Cons','MENA','EU_28') = 5639.129395 ;
vxsb('c_Util_Cons','MENA','MENA') = 2598.879639 ;
vxsb('c_Util_Cons','MENA','SSA') = 375.0569458 ;
vxsb('c_Util_Cons','MENA','RestofWorld') = 2366.30835 ;
vxsb('c_Util_Cons','SSA','Oceania') = 32.03158188 ;
vxsb('c_Util_Cons','SSA','EastAsia') = 254.3527527 ;
vxsb('c_Util_Cons','SSA','SEAsia') = 144.2131348 ;
vxsb('c_Util_Cons','SSA','SouthAsia') = 27.73508072 ;
vxsb('c_Util_Cons','SSA','NAmerica') = 159.2519226 ;
vxsb('c_Util_Cons','SSA','LatinAmer') = 124.1552811 ;
vxsb('c_Util_Cons','SSA','EU_28') = 928.7954712 ;
vxsb('c_Util_Cons','SSA','MENA') = 392.601532 ;
vxsb('c_Util_Cons','SSA','SSA') = 1045.420776 ;
vxsb('c_Util_Cons','SSA','RestofWorld') = 252.3145447 ;
vxsb('c_Util_Cons','RestofWorld','Oceania') = 196.6998596 ;
vxsb('c_Util_Cons','RestofWorld','EastAsia') = 1458.554199 ;
vxsb('c_Util_Cons','RestofWorld','SEAsia') = 1037.572632 ;
vxsb('c_Util_Cons','RestofWorld','SouthAsia') = 165.8195496 ;
vxsb('c_Util_Cons','RestofWorld','NAmerica') = 818.8847656 ;
vxsb('c_Util_Cons','RestofWorld','LatinAmer') = 856.9828491 ;
vxsb('c_Util_Cons','RestofWorld','EU_28') = 12403.9707 ;
vxsb('c_Util_Cons','RestofWorld','MENA') = 3945.540527 ;
vxsb('c_Util_Cons','RestofWorld','SSA') = 488.2766418 ;
vxsb('c_Util_Cons','RestofWorld','RestofWorld') = 3622.735107 ;
vxsb('c_TransComm','Oceania','Oceania') = 2169.616455 ;
vxsb('c_TransComm','Oceania','EastAsia') = 6539.338867 ;
vxsb('c_TransComm','Oceania','SEAsia') = 1562.600342 ;
vxsb('c_TransComm','Oceania','SouthAsia') = 348.3964844 ;
vxsb('c_TransComm','Oceania','NAmerica') = 4447.299316 ;
vxsb('c_TransComm','Oceania','LatinAmer') = 751.6378174 ;
vxsb('c_TransComm','Oceania','EU_28') = 7174.994629 ;
vxsb('c_TransComm','Oceania','MENA') = 1683.146484 ;
vxsb('c_TransComm','Oceania','SSA') = 553.6177979 ;
vxsb('c_TransComm','Oceania','RestofWorld') = 714.2366333 ;
vxsb('c_TransComm','EastAsia','Oceania') = 4212.180664 ;
vxsb('c_TransComm','EastAsia','EastAsia') = 65740.33594 ;
vxsb('c_TransComm','EastAsia','SEAsia') = 15686.9043 ;
vxsb('c_TransComm','EastAsia','SouthAsia') = 3984.506836 ;
vxsb('c_TransComm','EastAsia','NAmerica') = 34000.04688 ;
vxsb('c_TransComm','EastAsia','LatinAmer') = 6217.623047 ;
vxsb('c_TransComm','EastAsia','EU_28') = 58985.67188 ;
vxsb('c_TransComm','EastAsia','MENA') = 11513.73438 ;
vxsb('c_TransComm','EastAsia','SSA') = 4201.472656 ;
vxsb('c_TransComm','EastAsia','RestofWorld') = 6782.009766 ;
vxsb('c_TransComm','SEAsia','Oceania') = 2004.836426 ;
vxsb('c_TransComm','SEAsia','EastAsia') = 16652.25781 ;
vxsb('c_TransComm','SEAsia','SEAsia') = 2442.834961 ;
vxsb('c_TransComm','SEAsia','SouthAsia') = 1960.946777 ;
vxsb('c_TransComm','SEAsia','NAmerica') = 10710.11719 ;
vxsb('c_TransComm','SEAsia','LatinAmer') = 2438.797363 ;
vxsb('c_TransComm','SEAsia','EU_28') = 30842.5957 ;
vxsb('c_TransComm','SEAsia','MENA') = 3997.199951 ;
vxsb('c_TransComm','SEAsia','SSA') = 1551.950195 ;
vxsb('c_TransComm','SEAsia','RestofWorld') = 3739.834961 ;
vxsb('c_TransComm','SouthAsia','Oceania') = 492.8401489 ;
vxsb('c_TransComm','SouthAsia','EastAsia') = 5351.861816 ;
vxsb('c_TransComm','SouthAsia','SEAsia') = 1405.473877 ;
vxsb('c_TransComm','SouthAsia','SouthAsia') = 138.3771057 ;
vxsb('c_TransComm','SouthAsia','NAmerica') = 4213.526367 ;
vxsb('c_TransComm','SouthAsia','LatinAmer') = 716.8010864 ;
vxsb('c_TransComm','SouthAsia','EU_28') = 7466.467285 ;
vxsb('c_TransComm','SouthAsia','MENA') = 1734.245361 ;
vxsb('c_TransComm','SouthAsia','SSA') = 441.8512878 ;
vxsb('c_TransComm','SouthAsia','RestofWorld') = 970.2805176 ;
vxsb('c_TransComm','NAmerica','Oceania') = 3273.124268 ;
vxsb('c_TransComm','NAmerica','EastAsia') = 27415.13086 ;
vxsb('c_TransComm','NAmerica','SEAsia') = 6990.8125 ;
vxsb('c_TransComm','NAmerica','SouthAsia') = 3207.255859 ;
vxsb('c_TransComm','NAmerica','NAmerica') = 21350.7207 ;
vxsb('c_TransComm','NAmerica','LatinAmer') = 4646.54834 ;
vxsb('c_TransComm','NAmerica','EU_28') = 55497.03906 ;
vxsb('c_TransComm','NAmerica','MENA') = 10289.19824 ;
vxsb('c_TransComm','NAmerica','SSA') = 2744.233398 ;
vxsb('c_TransComm','NAmerica','RestofWorld') = 4709.695801 ;
vxsb('c_TransComm','LatinAmer','Oceania') = 919.934021 ;
vxsb('c_TransComm','LatinAmer','EastAsia') = 7700.701172 ;
vxsb('c_TransComm','LatinAmer','SEAsia') = 2465.787598 ;
vxsb('c_TransComm','LatinAmer','SouthAsia') = 922.2738647 ;
vxsb('c_TransComm','LatinAmer','NAmerica') = 6838.367676 ;
vxsb('c_TransComm','LatinAmer','LatinAmer') = 1695.694336 ;
vxsb('c_TransComm','LatinAmer','EU_28') = 21363.47656 ;
vxsb('c_TransComm','LatinAmer','MENA') = 3560.369385 ;
vxsb('c_TransComm','LatinAmer','SSA') = 926.9926147 ;
vxsb('c_TransComm','LatinAmer','RestofWorld') = 3073.308105 ;
vxsb('c_TransComm','EU_28','Oceania') = 8274.443359 ;
vxsb('c_TransComm','EU_28','EastAsia') = 52822.07812 ;
vxsb('c_TransComm','EU_28','SEAsia') = 22939.01562 ;
vxsb('c_TransComm','EU_28','SouthAsia') = 10831.97461 ;
vxsb('c_TransComm','EU_28','NAmerica') = 60427.17969 ;
vxsb('c_TransComm','EU_28','LatinAmer') = 22014.25 ;
vxsb('c_TransComm','EU_28','EU_28') = 316962.3125 ;
vxsb('c_TransComm','EU_28','MENA') = 28538.63281 ;
vxsb('c_TransComm','EU_28','SSA') = 12248.56934 ;
vxsb('c_TransComm','EU_28','RestofWorld') = 40754.24609 ;
vxsb('c_TransComm','MENA','Oceania') = 1772.074219 ;
vxsb('c_TransComm','MENA','EastAsia') = 15178.52441 ;
vxsb('c_TransComm','MENA','SEAsia') = 4101.641602 ;
vxsb('c_TransComm','MENA','SouthAsia') = 1782.15271 ;
vxsb('c_TransComm','MENA','NAmerica') = 16937.83984 ;
vxsb('c_TransComm','MENA','LatinAmer') = 3612.134521 ;
vxsb('c_TransComm','MENA','EU_28') = 40371.64844 ;
vxsb('c_TransComm','MENA','MENA') = 4900.626953 ;
vxsb('c_TransComm','MENA','SSA') = 1857.097778 ;
vxsb('c_TransComm','MENA','RestofWorld') = 6201.303223 ;
vxsb('c_TransComm','SSA','Oceania') = 429.4367981 ;
vxsb('c_TransComm','SSA','EastAsia') = 3114.362305 ;
vxsb('c_TransComm','SSA','SEAsia') = 998.7744751 ;
vxsb('c_TransComm','SSA','SouthAsia') = 317.2618408 ;
vxsb('c_TransComm','SSA','NAmerica') = 3187.952148 ;
vxsb('c_TransComm','SSA','LatinAmer') = 675.1118164 ;
vxsb('c_TransComm','SSA','EU_28') = 8443.737305 ;
vxsb('c_TransComm','SSA','MENA') = 1192.515625 ;
vxsb('c_TransComm','SSA','SSA') = 354.2862854 ;
vxsb('c_TransComm','SSA','RestofWorld') = 896.7719116 ;
vxsb('c_TransComm','RestofWorld','Oceania') = 947.9411621 ;
vxsb('c_TransComm','RestofWorld','EastAsia') = 10903.88184 ;
vxsb('c_TransComm','RestofWorld','SEAsia') = 4127.525391 ;
vxsb('c_TransComm','RestofWorld','SouthAsia') = 1044.633789 ;
vxsb('c_TransComm','RestofWorld','NAmerica') = 8281.378906 ;
vxsb('c_TransComm','RestofWorld','LatinAmer') = 2161.365723 ;
vxsb('c_TransComm','RestofWorld','EU_28') = 36154.76172 ;
vxsb('c_TransComm','RestofWorld','MENA') = 4304.719238 ;
vxsb('c_TransComm','RestofWorld','SSA') = 1237.386597 ;
vxsb('c_TransComm','RestofWorld','RestofWorld') = 3337.639893 ;
vxsb('c_OthService','Oceania','Oceania') = 2283.932617 ;
vxsb('c_OthService','Oceania','EastAsia') = 6401.10791 ;
vxsb('c_OthService','Oceania','SEAsia') = 2127.986084 ;
vxsb('c_OthService','Oceania','SouthAsia') = 573.2736206 ;
vxsb('c_OthService','Oceania','NAmerica') = 5652.233887 ;
vxsb('c_OthService','Oceania','LatinAmer') = 786.1976318 ;
vxsb('c_OthService','Oceania','EU_28') = 7780.802734 ;
vxsb('c_OthService','Oceania','MENA') = 2293.664307 ;
vxsb('c_OthService','Oceania','SSA') = 659.3297729 ;
vxsb('c_OthService','Oceania','RestofWorld') = 841.8104858 ;
vxsb('c_OthService','EastAsia','Oceania') = 2661.219482 ;
vxsb('c_OthService','EastAsia','EastAsia') = 24510.78125 ;
vxsb('c_OthService','EastAsia','SEAsia') = 10720.59766 ;
vxsb('c_OthService','EastAsia','SouthAsia') = 5830.672363 ;
vxsb('c_OthService','EastAsia','NAmerica') = 40831.04688 ;
vxsb('c_OthService','EastAsia','LatinAmer') = 6029.530762 ;
vxsb('c_OthService','EastAsia','EU_28') = 51103.73047 ;
vxsb('c_OthService','EastAsia','MENA') = 13700.07715 ;
vxsb('c_OthService','EastAsia','SSA') = 3753.361328 ;
vxsb('c_OthService','EastAsia','RestofWorld') = 6419.271484 ;
vxsb('c_OthService','SEAsia','Oceania') = 2375.387695 ;
vxsb('c_OthService','SEAsia','EastAsia') = 12545.82324 ;
vxsb('c_OthService','SEAsia','SEAsia') = 2450.027588 ;
vxsb('c_OthService','SEAsia','SouthAsia') = 4212.79248 ;
vxsb('c_OthService','SEAsia','NAmerica') = 11423.6748 ;
vxsb('c_OthService','SEAsia','LatinAmer') = 2735.078369 ;
vxsb('c_OthService','SEAsia','EU_28') = 30411.67188 ;
vxsb('c_OthService','SEAsia','MENA') = 4138.499512 ;
vxsb('c_OthService','SEAsia','SSA') = 1649.08313 ;
vxsb('c_OthService','SEAsia','RestofWorld') = 3016.982666 ;
vxsb('c_OthService','SouthAsia','Oceania') = 925.135498 ;
vxsb('c_OthService','SouthAsia','EastAsia') = 7902.925781 ;
vxsb('c_OthService','SouthAsia','SEAsia') = 2953.47168 ;
vxsb('c_OthService','SouthAsia','SouthAsia') = 317.1229553 ;
vxsb('c_OthService','SouthAsia','NAmerica') = 23476.5957 ;
vxsb('c_OthService','SouthAsia','LatinAmer') = 1873.734741 ;
vxsb('c_OthService','SouthAsia','EU_28') = 25990.93555 ;
vxsb('c_OthService','SouthAsia','MENA') = 3121.552734 ;
vxsb('c_OthService','SouthAsia','SSA') = 1184.272705 ;
vxsb('c_OthService','SouthAsia','RestofWorld') = 2095.656494 ;
vxsb('c_OthService','NAmerica','Oceania') = 6903.589844 ;
vxsb('c_OthService','NAmerica','EastAsia') = 61267.43359 ;
vxsb('c_OthService','NAmerica','SEAsia') = 16068.8877 ;
vxsb('c_OthService','NAmerica','SouthAsia') = 11846.02148 ;
vxsb('c_OthService','NAmerica','NAmerica') = 46485.67578 ;
vxsb('c_OthService','NAmerica','LatinAmer') = 11753.08398 ;
vxsb('c_OthService','NAmerica','EU_28') = 149312.7031 ;
vxsb('c_OthService','NAmerica','MENA') = 35297.79297 ;
vxsb('c_OthService','NAmerica','SSA') = 10327.73535 ;
vxsb('c_OthService','NAmerica','RestofWorld') = 14985.01367 ;
vxsb('c_OthService','LatinAmer','Oceania') = 767.5485229 ;
vxsb('c_OthService','LatinAmer','EastAsia') = 6080.470215 ;
vxsb('c_OthService','LatinAmer','SEAsia') = 2984.134277 ;
vxsb('c_OthService','LatinAmer','SouthAsia') = 1425.15332 ;
vxsb('c_OthService','LatinAmer','NAmerica') = 10734.77051 ;
vxsb('c_OthService','LatinAmer','LatinAmer') = 1358.890869 ;
vxsb('c_OthService','LatinAmer','EU_28') = 18873.36719 ;
vxsb('c_OthService','LatinAmer','MENA') = 4633.399902 ;
vxsb('c_OthService','LatinAmer','SSA') = 1248.408447 ;
vxsb('c_OthService','LatinAmer','RestofWorld') = 2278.50415 ;
vxsb('c_OthService','EU_28','Oceania') = 8443.917969 ;
vxsb('c_OthService','EU_28','EastAsia') = 56998.12109 ;
vxsb('c_OthService','EU_28','SEAsia') = 31403.49414 ;
vxsb('c_OthService','EU_28','SouthAsia') = 20280.83594 ;
vxsb('c_OthService','EU_28','NAmerica') = 129155.0859 ;
vxsb('c_OthService','EU_28','LatinAmer') = 27664.25195 ;
vxsb('c_OthService','EU_28','EU_28') = 491871.9062 ;
vxsb('c_OthService','EU_28','MENA') = 47819.35156 ;
vxsb('c_OthService','EU_28','SSA') = 19573.19727 ;
vxsb('c_OthService','EU_28','RestofWorld') = 62218.03125 ;
vxsb('c_OthService','MENA','Oceania') = 1445.43689 ;
vxsb('c_OthService','MENA','EastAsia') = 12839.02148 ;
vxsb('c_OthService','MENA','SEAsia') = 3465.295898 ;
vxsb('c_OthService','MENA','SouthAsia') = 2367.249756 ;
vxsb('c_OthService','MENA','NAmerica') = 18135.54883 ;
vxsb('c_OthService','MENA','LatinAmer') = 3768.030029 ;
vxsb('c_OthService','MENA','EU_28') = 34945.88281 ;
vxsb('c_OthService','MENA','MENA') = 5371.029785 ;
vxsb('c_OthService','MENA','SSA') = 1977.580933 ;
vxsb('c_OthService','MENA','RestofWorld') = 5075.946289 ;
vxsb('c_OthService','SSA','Oceania') = 308.29599 ;
vxsb('c_OthService','SSA','EastAsia') = 2903.436035 ;
vxsb('c_OthService','SSA','SEAsia') = 790.6919556 ;
vxsb('c_OthService','SSA','SouthAsia') = 361.8479004 ;
vxsb('c_OthService','SSA','NAmerica') = 4691.35791 ;
vxsb('c_OthService','SSA','LatinAmer') = 644.6809692 ;
vxsb('c_OthService','SSA','EU_28') = 6700.551758 ;
vxsb('c_OthService','SSA','MENA') = 2125.437988 ;
vxsb('c_OthService','SSA','SSA') = 492.7622986 ;
vxsb('c_OthService','SSA','RestofWorld') = 848.0107422 ;
vxsb('c_OthService','RestofWorld','Oceania') = 842.0559692 ;
vxsb('c_OthService','RestofWorld','EastAsia') = 8420.21582 ;
vxsb('c_OthService','RestofWorld','SEAsia') = 5079.750488 ;
vxsb('c_OthService','RestofWorld','SouthAsia') = 2429.520752 ;
vxsb('c_OthService','RestofWorld','NAmerica') = 18953.99609 ;
vxsb('c_OthService','RestofWorld','LatinAmer') = 2311.362549 ;
vxsb('c_OthService','RestofWorld','EU_28') = 48936.79297 ;
vxsb('c_OthService','RestofWorld','MENA') = 6074.623535 ;
vxsb('c_OthService','RestofWorld','SSA') = 2068.710938 ;
vxsb('c_OthService','RestofWorld','RestofWorld') = 3546.81543 ;

* vfob data (1000 cells)
vfob('c_Crops','Oceania','Oceania') = 857.6879883 ;
vfob('c_Crops','Oceania','EastAsia') = 5490.546875 ;
vfob('c_Crops','Oceania','SEAsia') = 4003.577148 ;
vfob('c_Crops','Oceania','SouthAsia') = 957.8133545 ;
vfob('c_Crops','Oceania','NAmerica') = 335.188324 ;
vfob('c_Crops','Oceania','LatinAmer') = 47.52339172 ;
vfob('c_Crops','Oceania','EU_28') = 1428.42395 ;
vfob('c_Crops','Oceania','MENA') = 2263.045654 ;
vfob('c_Crops','Oceania','SSA') = 434.6171875 ;
vfob('c_Crops','Oceania','RestofWorld') = 51.01342392 ;
vfob('c_Crops','EastAsia','Oceania') = 139.8350525 ;
vfob('c_Crops','EastAsia','EastAsia') = 3744.263184 ;
vfob('c_Crops','EastAsia','SEAsia') = 4213.192383 ;
vfob('c_Crops','EastAsia','SouthAsia') = 780.0725708 ;
vfob('c_Crops','EastAsia','NAmerica') = 1225.357788 ;
vfob('c_Crops','EastAsia','LatinAmer') = 419.2332153 ;
vfob('c_Crops','EastAsia','EU_28') = 1735.738037 ;
vfob('c_Crops','EastAsia','MENA') = 1018.51416 ;
vfob('c_Crops','EastAsia','SSA') = 781.7420044 ;
vfob('c_Crops','EastAsia','RestofWorld') = 994.4402466 ;
vfob('c_Crops','SEAsia','Oceania') = 533.7235107 ;
vfob('c_Crops','SEAsia','EastAsia') = 7229.678711 ;
vfob('c_Crops','SEAsia','SEAsia') = 5605.609375 ;
vfob('c_Crops','SEAsia','SouthAsia') = 1874.097046 ;
vfob('c_Crops','SEAsia','NAmerica') = 2794.952637 ;
vfob('c_Crops','SEAsia','LatinAmer') = 160.3613434 ;
vfob('c_Crops','SEAsia','EU_28') = 3504.646729 ;
vfob('c_Crops','SEAsia','MENA') = 1881.696289 ;
vfob('c_Crops','SEAsia','SSA') = 3432.992432 ;
vfob('c_Crops','SEAsia','RestofWorld') = 633.1633911 ;
vfob('c_Crops','SouthAsia','Oceania') = 178.7569122 ;
vfob('c_Crops','SouthAsia','EastAsia') = 3247.621582 ;
vfob('c_Crops','SouthAsia','SEAsia') = 3022.472168 ;
vfob('c_Crops','SouthAsia','SouthAsia') = 3515.255615 ;
vfob('c_Crops','SouthAsia','NAmerica') = 1336.687988 ;
vfob('c_Crops','SouthAsia','LatinAmer') = 163.8376923 ;
vfob('c_Crops','SouthAsia','EU_28') = 2487.262695 ;
vfob('c_Crops','SouthAsia','MENA') = 6198.697266 ;
vfob('c_Crops','SouthAsia','SSA') = 1588.599731 ;
vfob('c_Crops','SouthAsia','RestofWorld') = 995.1212769 ;
vfob('c_Crops','NAmerica','Oceania') = 414.7614441 ;
vfob('c_Crops','NAmerica','EastAsia') = 39592.66016 ;
vfob('c_Crops','NAmerica','SEAsia') = 6186.848145 ;
vfob('c_Crops','NAmerica','SouthAsia') = 3242.922607 ;
vfob('c_Crops','NAmerica','NAmerica') = 27314.78906 ;
vfob('c_Crops','NAmerica','LatinAmer') = 7699.303711 ;
vfob('c_Crops','NAmerica','EU_28') = 7975.477539 ;
vfob('c_Crops','NAmerica','MENA') = 10926.30957 ;
vfob('c_Crops','NAmerica','SSA') = 4399.038574 ;
vfob('c_Crops','NAmerica','RestofWorld') = 823.7838745 ;
vfob('c_Crops','LatinAmer','Oceania') = 279.023468 ;
vfob('c_Crops','LatinAmer','EastAsia') = 22874.04883 ;
vfob('c_Crops','LatinAmer','SEAsia') = 3148.141846 ;
vfob('c_Crops','LatinAmer','SouthAsia') = 399.9667664 ;
vfob('c_Crops','LatinAmer','NAmerica') = 15445.34375 ;
vfob('c_Crops','LatinAmer','LatinAmer') = 9450.181641 ;
vfob('c_Crops','LatinAmer','EU_28') = 22078.10352 ;
vfob('c_Crops','LatinAmer','MENA') = 8196.748047 ;
vfob('c_Crops','LatinAmer','SSA') = 1704.201538 ;
vfob('c_Crops','LatinAmer','RestofWorld') = 3558.922852 ;
vfob('c_Crops','EU_28','Oceania') = 128.0878754 ;
vfob('c_Crops','EU_28','EastAsia') = 979.0946655 ;
vfob('c_Crops','EU_28','SEAsia') = 321.4285583 ;
vfob('c_Crops','EU_28','SouthAsia') = 352.9377747 ;
vfob('c_Crops','EU_28','NAmerica') = 1455.432983 ;
vfob('c_Crops','EU_28','LatinAmer') = 667.1140747 ;
vfob('c_Crops','EU_28','EU_28') = 70138.32812 ;
vfob('c_Crops','EU_28','MENA') = 8876.798828 ;
vfob('c_Crops','EU_28','SSA') = 2126.307129 ;
vfob('c_Crops','EU_28','RestofWorld') = 7793.549316 ;
vfob('c_Crops','MENA','Oceania') = 131.0546265 ;
vfob('c_Crops','MENA','EastAsia') = 652.4574585 ;
vfob('c_Crops','MENA','SEAsia') = 318.9515686 ;
vfob('c_Crops','MENA','SouthAsia') = 575.0499878 ;
vfob('c_Crops','MENA','NAmerica') = 878.3111572 ;
vfob('c_Crops','MENA','LatinAmer') = 175.6295929 ;
vfob('c_Crops','MENA','EU_28') = 6443.415039 ;
vfob('c_Crops','MENA','MENA') = 6095.941406 ;
vfob('c_Crops','MENA','SSA') = 543.3104248 ;
vfob('c_Crops','MENA','RestofWorld') = 3425.030762 ;
vfob('c_Crops','SSA','Oceania') = 88.20375061 ;
vfob('c_Crops','SSA','EastAsia') = 2667.998535 ;
vfob('c_Crops','SSA','SEAsia') = 1988.507812 ;
vfob('c_Crops','SSA','SouthAsia') = 2002.924316 ;
vfob('c_Crops','SSA','NAmerica') = 2382.590088 ;
vfob('c_Crops','SSA','LatinAmer') = 242.6151733 ;
vfob('c_Crops','SSA','EU_28') = 9531.945312 ;
vfob('c_Crops','SSA','MENA') = 2072.047607 ;
vfob('c_Crops','SSA','SSA') = 2348.8396 ;
vfob('c_Crops','SSA','RestofWorld') = 1109.598511 ;
vfob('c_Crops','RestofWorld','Oceania') = 10.36016273 ;
vfob('c_Crops','RestofWorld','EastAsia') = 711.4573975 ;
vfob('c_Crops','RestofWorld','SEAsia') = 65.0391922 ;
vfob('c_Crops','RestofWorld','SouthAsia') = 270.0123596 ;
vfob('c_Crops','RestofWorld','NAmerica') = 119.9149475 ;
vfob('c_Crops','RestofWorld','LatinAmer') = 100.3549805 ;
vfob('c_Crops','RestofWorld','EU_28') = 3628.954346 ;
vfob('c_Crops','RestofWorld','MENA') = 9899.734375 ;
vfob('c_Crops','RestofWorld','SSA') = 819.0136719 ;
vfob('c_Crops','RestofWorld','RestofWorld') = 3035.618652 ;
vfob('c_MeatLstk','Oceania','Oceania') = 607.3534546 ;
vfob('c_MeatLstk','Oceania','EastAsia') = 8415.358398 ;
vfob('c_MeatLstk','Oceania','SEAsia') = 1522.231812 ;
vfob('c_MeatLstk','Oceania','SouthAsia') = 309.6993713 ;
vfob('c_MeatLstk','Oceania','NAmerica') = 2844.744629 ;
vfob('c_MeatLstk','Oceania','LatinAmer') = 200.1086884 ;
vfob('c_MeatLstk','Oceania','EU_28') = 2448.843262 ;
vfob('c_MeatLstk','Oceania','MENA') = 1249.458008 ;
vfob('c_MeatLstk','Oceania','SSA') = 159.0433197 ;
vfob('c_MeatLstk','Oceania','RestofWorld') = 668.5925293 ;
vfob('c_MeatLstk','EastAsia','Oceania') = 44.17465591 ;
vfob('c_MeatLstk','EastAsia','EastAsia') = 3209.034424 ;
vfob('c_MeatLstk','EastAsia','SEAsia') = 524.1376953 ;
vfob('c_MeatLstk','EastAsia','SouthAsia') = 64.92596436 ;
vfob('c_MeatLstk','EastAsia','NAmerica') = 711.2973633 ;
vfob('c_MeatLstk','EastAsia','LatinAmer') = 94.77591705 ;
vfob('c_MeatLstk','EastAsia','EU_28') = 1578.198975 ;
vfob('c_MeatLstk','EastAsia','MENA') = 212.1342163 ;
vfob('c_MeatLstk','EastAsia','SSA') = 87.37517548 ;
vfob('c_MeatLstk','EastAsia','RestofWorld') = 163.1934052 ;
vfob('c_MeatLstk','SEAsia','Oceania') = 16.62125778 ;
vfob('c_MeatLstk','SEAsia','EastAsia') = 1825.650757 ;
vfob('c_MeatLstk','SEAsia','SEAsia') = 762.9240723 ;
vfob('c_MeatLstk','SEAsia','SouthAsia') = 21.78966522 ;
vfob('c_MeatLstk','SEAsia','NAmerica') = 185.9979248 ;
vfob('c_MeatLstk','SEAsia','LatinAmer') = 9.241361618 ;
vfob('c_MeatLstk','SEAsia','EU_28') = 1053.550171 ;
vfob('c_MeatLstk','SEAsia','MENA') = 73.92140198 ;
vfob('c_MeatLstk','SEAsia','SSA') = 39.79859924 ;
vfob('c_MeatLstk','SEAsia','RestofWorld') = 31.28232765 ;
vfob('c_MeatLstk','SouthAsia','Oceania') = 5.9895854 ;
vfob('c_MeatLstk','SouthAsia','EastAsia') = 84.25678253 ;
vfob('c_MeatLstk','SouthAsia','SEAsia') = 1230.236938 ;
vfob('c_MeatLstk','SouthAsia','SouthAsia') = 75.06591797 ;
vfob('c_MeatLstk','SouthAsia','NAmerica') = 129.2500153 ;
vfob('c_MeatLstk','SouthAsia','LatinAmer') = 9.268588066 ;
vfob('c_MeatLstk','SouthAsia','EU_28') = 143.7459259 ;
vfob('c_MeatLstk','SouthAsia','MENA') = 1406.680176 ;
vfob('c_MeatLstk','SouthAsia','SSA') = 186.3073273 ;
vfob('c_MeatLstk','SouthAsia','RestofWorld') = 106.0374069 ;
vfob('c_MeatLstk','NAmerica','Oceania') = 516.8505859 ;
vfob('c_MeatLstk','NAmerica','EastAsia') = 12539.63184 ;
vfob('c_MeatLstk','NAmerica','SEAsia') = 671.8615112 ;
vfob('c_MeatLstk','NAmerica','SouthAsia') = 121.6928253 ;
vfob('c_MeatLstk','NAmerica','NAmerica') = 12264.18359 ;
vfob('c_MeatLstk','NAmerica','LatinAmer') = 1757.155762 ;
vfob('c_MeatLstk','NAmerica','EU_28') = 1220.633057 ;
vfob('c_MeatLstk','NAmerica','MENA') = 1177.580566 ;
vfob('c_MeatLstk','NAmerica','SSA') = 678.7714233 ;
vfob('c_MeatLstk','NAmerica','RestofWorld') = 1979.045288 ;
vfob('c_MeatLstk','LatinAmer','Oceania') = 42.98695755 ;
vfob('c_MeatLstk','LatinAmer','EastAsia') = 3796.107422 ;
vfob('c_MeatLstk','LatinAmer','SEAsia') = 520.2775269 ;
vfob('c_MeatLstk','LatinAmer','SouthAsia') = 60.51265335 ;
vfob('c_MeatLstk','LatinAmer','NAmerica') = 1329.915039 ;
vfob('c_MeatLstk','LatinAmer','LatinAmer') = 4288.72168 ;
vfob('c_MeatLstk','LatinAmer','EU_28') = 4302.417969 ;
vfob('c_MeatLstk','LatinAmer','MENA') = 5324.080078 ;
vfob('c_MeatLstk','LatinAmer','SSA') = 1122.762451 ;
vfob('c_MeatLstk','LatinAmer','RestofWorld') = 3025.052246 ;
vfob('c_MeatLstk','EU_28','Oceania') = 398.5938416 ;
vfob('c_MeatLstk','EU_28','EastAsia') = 5414.023438 ;
vfob('c_MeatLstk','EU_28','SEAsia') = 566.048584 ;
vfob('c_MeatLstk','EU_28','SouthAsia') = 150.4817963 ;
vfob('c_MeatLstk','EU_28','NAmerica') = 1135.482056 ;
vfob('c_MeatLstk','EU_28','LatinAmer') = 393.3152161 ;
vfob('c_MeatLstk','EU_28','EU_28') = 65035.85156 ;
vfob('c_MeatLstk','EU_28','MENA') = 3163.630127 ;
vfob('c_MeatLstk','EU_28','SSA') = 1110.924683 ;
vfob('c_MeatLstk','EU_28','RestofWorld') = 5565.346191 ;
vfob('c_MeatLstk','MENA','Oceania') = 13.39822006 ;
vfob('c_MeatLstk','MENA','EastAsia') = 143.8743896 ;
vfob('c_MeatLstk','MENA','SEAsia') = 61.90138626 ;
vfob('c_MeatLstk','MENA','SouthAsia') = 83.02220917 ;
vfob('c_MeatLstk','MENA','NAmerica') = 108.4758148 ;
vfob('c_MeatLstk','MENA','LatinAmer') = 20.33966064 ;
vfob('c_MeatLstk','MENA','EU_28') = 750.1575928 ;
vfob('c_MeatLstk','MENA','MENA') = 1693.26123 ;
vfob('c_MeatLstk','MENA','SSA') = 82.7540741 ;
vfob('c_MeatLstk','MENA','RestofWorld') = 136.1691742 ;
vfob('c_MeatLstk','SSA','Oceania') = 9.566291809 ;
vfob('c_MeatLstk','SSA','EastAsia') = 367.7507935 ;
vfob('c_MeatLstk','SSA','SEAsia') = 34.20042038 ;
vfob('c_MeatLstk','SSA','SouthAsia') = 102.5611801 ;
vfob('c_MeatLstk','SSA','NAmerica') = 89.9961853 ;
vfob('c_MeatLstk','SSA','LatinAmer') = 13.64141846 ;
vfob('c_MeatLstk','SSA','EU_28') = 433.1038818 ;
vfob('c_MeatLstk','SSA','MENA') = 789.5419922 ;
vfob('c_MeatLstk','SSA','SSA') = 548.133606 ;
vfob('c_MeatLstk','SSA','RestofWorld') = 79.62467194 ;
vfob('c_MeatLstk','RestofWorld','Oceania') = 18.74494553 ;
vfob('c_MeatLstk','RestofWorld','EastAsia') = 196.1444855 ;
vfob('c_MeatLstk','RestofWorld','SEAsia') = 41.36655045 ;
vfob('c_MeatLstk','RestofWorld','SouthAsia') = 14.19671822 ;
vfob('c_MeatLstk','RestofWorld','NAmerica') = 128.9498291 ;
vfob('c_MeatLstk','RestofWorld','LatinAmer') = 19.0758934 ;
vfob('c_MeatLstk','RestofWorld','EU_28') = 935.2359009 ;
vfob('c_MeatLstk','RestofWorld','MENA') = 183.7813568 ;
vfob('c_MeatLstk','RestofWorld','SSA') = 23.60223198 ;
vfob('c_MeatLstk','RestofWorld','RestofWorld') = 514.0353394 ;
vfob('c_Extraction','Oceania','Oceania') = 1195.493652 ;
vfob('c_Extraction','Oceania','EastAsia') = 115666.4375 ;
vfob('c_Extraction','Oceania','SEAsia') = 3867.570312 ;
vfob('c_Extraction','Oceania','SouthAsia') = 11691.52051 ;
vfob('c_Extraction','Oceania','NAmerica') = 1950.790771 ;
vfob('c_Extraction','Oceania','LatinAmer') = 1103.131714 ;
vfob('c_Extraction','Oceania','EU_28') = 5131.975586 ;
vfob('c_Extraction','Oceania','MENA') = 860.2334595 ;
vfob('c_Extraction','Oceania','SSA') = 301.757019 ;
vfob('c_Extraction','Oceania','RestofWorld') = 199.5310822 ;
vfob('c_Extraction','EastAsia','Oceania') = 1847.462158 ;
vfob('c_Extraction','EastAsia','EastAsia') = 17525.3457 ;
vfob('c_Extraction','EastAsia','SEAsia') = 1808.346924 ;
vfob('c_Extraction','EastAsia','SouthAsia') = 1737.694336 ;
vfob('c_Extraction','EastAsia','NAmerica') = 989.3400269 ;
vfob('c_Extraction','EastAsia','LatinAmer') = 99.50413513 ;
vfob('c_Extraction','EastAsia','EU_28') = 1301.929321 ;
vfob('c_Extraction','EastAsia','MENA') = 215.3435211 ;
vfob('c_Extraction','EastAsia','SSA') = 88.76267242 ;
vfob('c_Extraction','EastAsia','RestofWorld') = 221.781723 ;
vfob('c_Extraction','SEAsia','Oceania') = 7231.898926 ;
vfob('c_Extraction','SEAsia','EastAsia') = 66218.25781 ;
vfob('c_Extraction','SEAsia','SEAsia') = 14756.94629 ;
vfob('c_Extraction','SEAsia','SouthAsia') = 8135.112793 ;
vfob('c_Extraction','SEAsia','NAmerica') = 1780.827515 ;
vfob('c_Extraction','SEAsia','LatinAmer') = 149.7783051 ;
vfob('c_Extraction','SEAsia','EU_28') = 2208.474121 ;
vfob('c_Extraction','SEAsia','MENA') = 689.5847778 ;
vfob('c_Extraction','SEAsia','SSA') = 53.7468338 ;
vfob('c_Extraction','SEAsia','RestofWorld') = 260.5140686 ;
vfob('c_Extraction','SouthAsia','Oceania') = 34.65118408 ;
vfob('c_Extraction','SouthAsia','EastAsia') = 11739.7998 ;
vfob('c_Extraction','SouthAsia','SEAsia') = 296.7428894 ;
vfob('c_Extraction','SouthAsia','SouthAsia') = 565.9177246 ;
vfob('c_Extraction','SouthAsia','NAmerica') = 232.3936005 ;
vfob('c_Extraction','SouthAsia','LatinAmer') = 39.15106964 ;
vfob('c_Extraction','SouthAsia','EU_28') = 1342.369629 ;
vfob('c_Extraction','SouthAsia','MENA') = 679.2763062 ;
vfob('c_Extraction','SouthAsia','SSA') = 52.41881943 ;
vfob('c_Extraction','SouthAsia','RestofWorld') = 58.5100174 ;
vfob('c_Extraction','NAmerica','Oceania') = 533.8035889 ;
vfob('c_Extraction','NAmerica','EastAsia') = 21624.25 ;
vfob('c_Extraction','NAmerica','SEAsia') = 574.512146 ;
vfob('c_Extraction','NAmerica','SouthAsia') = 2170.053711 ;
vfob('c_Extraction','NAmerica','NAmerica') = 98830.625 ;
vfob('c_Extraction','NAmerica','LatinAmer') = 4018.219727 ;
vfob('c_Extraction','NAmerica','EU_28') = 18071.65234 ;
vfob('c_Extraction','NAmerica','MENA') = 3197.293945 ;
vfob('c_Extraction','NAmerica','SSA') = 277.6490173 ;
vfob('c_Extraction','NAmerica','RestofWorld') = 1728.963379 ;
vfob('c_Extraction','LatinAmer','Oceania') = 369.533905 ;
vfob('c_Extraction','LatinAmer','EastAsia') = 69811.35938 ;
vfob('c_Extraction','LatinAmer','SEAsia') = 2230.401367 ;
vfob('c_Extraction','LatinAmer','SouthAsia') = 9694.435547 ;
vfob('c_Extraction','LatinAmer','NAmerica') = 58320.45312 ;
vfob('c_Extraction','LatinAmer','LatinAmer') = 23895.13477 ;
vfob('c_Extraction','LatinAmer','EU_28') = 28866.29492 ;
vfob('c_Extraction','LatinAmer','MENA') = 5181.352539 ;
vfob('c_Extraction','LatinAmer','SSA') = 482.6952515 ;
vfob('c_Extraction','LatinAmer','RestofWorld') = 6238.039551 ;
vfob('c_Extraction','EU_28','Oceania') = 134.8734589 ;
vfob('c_Extraction','EU_28','EastAsia') = 3666.825439 ;
vfob('c_Extraction','EU_28','SEAsia') = 496.8589478 ;
vfob('c_Extraction','EU_28','SouthAsia') = 9436.3125 ;
vfob('c_Extraction','EU_28','NAmerica') = 2593.44751 ;
vfob('c_Extraction','EU_28','LatinAmer') = 830.0379028 ;
vfob('c_Extraction','EU_28','EU_28') = 56597.45312 ;
vfob('c_Extraction','EU_28','MENA') = 5482.632324 ;
vfob('c_Extraction','EU_28','SSA') = 1899.906372 ;
vfob('c_Extraction','EU_28','RestofWorld') = 2737.460693 ;
vfob('c_Extraction','MENA','Oceania') = 3040.057129 ;
vfob('c_Extraction','MENA','EastAsia') = 341561.8125 ;
vfob('c_Extraction','MENA','SEAsia') = 72125.03125 ;
vfob('c_Extraction','MENA','SouthAsia') = 109356.375 ;
vfob('c_Extraction','MENA','NAmerica') = 104824.8438 ;
vfob('c_Extraction','MENA','LatinAmer') = 5728.640625 ;
vfob('c_Extraction','MENA','EU_28') = 143973.2344 ;
vfob('c_Extraction','MENA','MENA') = 37592.0625 ;
vfob('c_Extraction','MENA','SSA') = 12889.25977 ;
vfob('c_Extraction','MENA','RestofWorld') = 2321.164307 ;
vfob('c_Extraction','SSA','Oceania') = 2256.40332 ;
vfob('c_Extraction','SSA','EastAsia') = 69024.29688 ;
vfob('c_Extraction','SSA','SEAsia') = 3005.983887 ;
vfob('c_Extraction','SSA','SouthAsia') = 23028.74609 ;
vfob('c_Extraction','SSA','NAmerica') = 76637.77344 ;
vfob('c_Extraction','SSA','LatinAmer') = 10294.9707 ;
vfob('c_Extraction','SSA','EU_28') = 50405.5625 ;
vfob('c_Extraction','SSA','MENA') = 1495.556274 ;
vfob('c_Extraction','SSA','SSA') = 9443.210938 ;
vfob('c_Extraction','SSA','RestofWorld') = 1929.105469 ;
vfob('c_Extraction','RestofWorld','Oceania') = 1660.235107 ;
vfob('c_Extraction','RestofWorld','EastAsia') = 82810.83594 ;
vfob('c_Extraction','RestofWorld','SEAsia') = 5939.749512 ;
vfob('c_Extraction','RestofWorld','SouthAsia') = 2071.001953 ;
vfob('c_Extraction','RestofWorld','NAmerica') = 20921.15039 ;
vfob('c_Extraction','RestofWorld','LatinAmer') = 2704.194336 ;
vfob('c_Extraction','RestofWorld','EU_28') = 321198.9375 ;
vfob('c_Extraction','RestofWorld','MENA') = 18354.03906 ;
vfob('c_Extraction','RestofWorld','SSA') = 227.7729492 ;
vfob('c_Extraction','RestofWorld','RestofWorld') = 45575.86328 ;
vfob('c_ProcFood','Oceania','Oceania') = 3611.03418 ;
vfob('c_ProcFood','Oceania','EastAsia') = 6502.73584 ;
vfob('c_ProcFood','Oceania','SEAsia') = 4550.655762 ;
vfob('c_ProcFood','Oceania','SouthAsia') = 679.8814697 ;
vfob('c_ProcFood','Oceania','NAmerica') = 2861.521484 ;
vfob('c_ProcFood','Oceania','LatinAmer') = 807.9478149 ;
vfob('c_ProcFood','Oceania','EU_28') = 2644.368408 ;
vfob('c_ProcFood','Oceania','MENA') = 2314.969971 ;
vfob('c_ProcFood','Oceania','SSA') = 667.6461792 ;
vfob('c_ProcFood','Oceania','RestofWorld') = 376.641449 ;
vfob('c_ProcFood','EastAsia','Oceania') = 1502.514282 ;
vfob('c_ProcFood','EastAsia','EastAsia') = 17315.49414 ;
vfob('c_ProcFood','EastAsia','SEAsia') = 6898.236816 ;
vfob('c_ProcFood','EastAsia','SouthAsia') = 584.6314697 ;
vfob('c_ProcFood','EastAsia','NAmerica') = 8844.349609 ;
vfob('c_ProcFood','EastAsia','LatinAmer') = 1106.4646 ;
vfob('c_ProcFood','EastAsia','EU_28') = 5972.814941 ;
vfob('c_ProcFood','EastAsia','MENA') = 1952.163208 ;
vfob('c_ProcFood','EastAsia','SSA') = 1630.007935 ;
vfob('c_ProcFood','EastAsia','RestofWorld') = 1989.016846 ;
vfob('c_ProcFood','SEAsia','Oceania') = 2466.391602 ;
vfob('c_ProcFood','SEAsia','EastAsia') = 23082.7168 ;
vfob('c_ProcFood','SEAsia','SEAsia') = 15680.21094 ;
vfob('c_ProcFood','SEAsia','SouthAsia') = 11235.41797 ;
vfob('c_ProcFood','SEAsia','NAmerica') = 12335.33105 ;
vfob('c_ProcFood','SEAsia','LatinAmer') = 1362.340332 ;
vfob('c_ProcFood','SEAsia','EU_28') = 8528.629883 ;
vfob('c_ProcFood','SEAsia','MENA') = 7378.742676 ;
vfob('c_ProcFood','SEAsia','SSA') = 4470.341309 ;
vfob('c_ProcFood','SEAsia','RestofWorld') = 2517.982178 ;
vfob('c_ProcFood','SouthAsia','Oceania') = 186.4442596 ;
vfob('c_ProcFood','SouthAsia','EastAsia') = 2783.594971 ;
vfob('c_ProcFood','SouthAsia','SEAsia') = 2321.296387 ;
vfob('c_ProcFood','SouthAsia','SouthAsia') = 1975.307495 ;
vfob('c_ProcFood','SouthAsia','NAmerica') = 2639.565918 ;
vfob('c_ProcFood','SouthAsia','LatinAmer') = 144.4218597 ;
vfob('c_ProcFood','SouthAsia','EU_28') = 2727.564697 ;
vfob('c_ProcFood','SouthAsia','MENA') = 2359.476074 ;
vfob('c_ProcFood','SouthAsia','SSA') = 1109.481812 ;
vfob('c_ProcFood','SouthAsia','RestofWorld') = 563.0005493 ;
vfob('c_ProcFood','NAmerica','Oceania') = 1546.805908 ;
vfob('c_ProcFood','NAmerica','EastAsia') = 14493.77344 ;
vfob('c_ProcFood','NAmerica','SEAsia') = 4229.92627 ;
vfob('c_ProcFood','NAmerica','SouthAsia') = 446.8729248 ;
vfob('c_ProcFood','NAmerica','NAmerica') = 40579.47656 ;
vfob('c_ProcFood','NAmerica','LatinAmer') = 8097.070312 ;
vfob('c_ProcFood','NAmerica','EU_28') = 8502.446289 ;
vfob('c_ProcFood','NAmerica','MENA') = 3903.092285 ;
vfob('c_ProcFood','NAmerica','SSA') = 984.867981 ;
vfob('c_ProcFood','NAmerica','RestofWorld') = 1360.379272 ;
vfob('c_ProcFood','LatinAmer','Oceania') = 652.0959473 ;
vfob('c_ProcFood','LatinAmer','EastAsia') = 10325.3418 ;
vfob('c_ProcFood','LatinAmer','SEAsia') = 6046.696289 ;
vfob('c_ProcFood','LatinAmer','SouthAsia') = 2622.865967 ;
vfob('c_ProcFood','LatinAmer','NAmerica') = 12930.61719 ;
vfob('c_ProcFood','LatinAmer','LatinAmer') = 19159.92383 ;
vfob('c_ProcFood','LatinAmer','EU_28') = 17422.81445 ;
vfob('c_ProcFood','LatinAmer','MENA') = 10619.8291 ;
vfob('c_ProcFood','LatinAmer','SSA') = 8021.084473 ;
vfob('c_ProcFood','LatinAmer','RestofWorld') = 3077.625977 ;
vfob('c_ProcFood','EU_28','Oceania') = 3077.376221 ;
vfob('c_ProcFood','EU_28','EastAsia') = 12681.89453 ;
vfob('c_ProcFood','EU_28','SEAsia') = 4344.393555 ;
vfob('c_ProcFood','EU_28','SouthAsia') = 1021.492981 ;
vfob('c_ProcFood','EU_28','NAmerica') = 23528.2207 ;
vfob('c_ProcFood','EU_28','LatinAmer') = 4790.538574 ;
vfob('c_ProcFood','EU_28','EU_28') = 226248.9844 ;
vfob('c_ProcFood','EU_28','MENA') = 16254.35742 ;
vfob('c_ProcFood','EU_28','SSA') = 7625.394043 ;
vfob('c_ProcFood','EU_28','RestofWorld') = 22755.93164 ;
vfob('c_ProcFood','MENA','Oceania') = 150.6530304 ;
vfob('c_ProcFood','MENA','EastAsia') = 964.2514648 ;
vfob('c_ProcFood','MENA','SEAsia') = 597.0473022 ;
vfob('c_ProcFood','MENA','SouthAsia') = 614.2488403 ;
vfob('c_ProcFood','MENA','NAmerica') = 1247.991943 ;
vfob('c_ProcFood','MENA','LatinAmer') = 155.6830902 ;
vfob('c_ProcFood','MENA','EU_28') = 5488.334961 ;
vfob('c_ProcFood','MENA','MENA') = 12943.43359 ;
vfob('c_ProcFood','MENA','SSA') = 2408.246582 ;
vfob('c_ProcFood','MENA','RestofWorld') = 959.1904907 ;
vfob('c_ProcFood','SSA','Oceania') = 192.465744 ;
vfob('c_ProcFood','SSA','EastAsia') = 927.7234497 ;
vfob('c_ProcFood','SSA','SEAsia') = 240.7167206 ;
vfob('c_ProcFood','SSA','SouthAsia') = 67.60327148 ;
vfob('c_ProcFood','SSA','NAmerica') = 994.4949951 ;
vfob('c_ProcFood','SSA','LatinAmer') = 95.97743225 ;
vfob('c_ProcFood','SSA','EU_28') = 6345.328613 ;
vfob('c_ProcFood','SSA','MENA') = 551.8257446 ;
vfob('c_ProcFood','SSA','SSA') = 5110.853027 ;
vfob('c_ProcFood','SSA','RestofWorld') = 393.3259583 ;
vfob('c_ProcFood','RestofWorld','Oceania') = 328.3638306 ;
vfob('c_ProcFood','RestofWorld','EastAsia') = 5478.37207 ;
vfob('c_ProcFood','RestofWorld','SEAsia') = 458.6554871 ;
vfob('c_ProcFood','RestofWorld','SouthAsia') = 999.9428101 ;
vfob('c_ProcFood','RestofWorld','NAmerica') = 2321.891846 ;
vfob('c_ProcFood','RestofWorld','LatinAmer') = 565.1418457 ;
vfob('c_ProcFood','RestofWorld','EU_28') = 13013.31934 ;
vfob('c_ProcFood','RestofWorld','MENA') = 4359.203125 ;
vfob('c_ProcFood','RestofWorld','SSA') = 945.8010254 ;
vfob('c_ProcFood','RestofWorld','RestofWorld') = 9562.174805 ;
vfob('c_TextWapp','Oceania','Oceania') = 621.7982178 ;
vfob('c_TextWapp','Oceania','EastAsia') = 473.4588318 ;
vfob('c_TextWapp','Oceania','SEAsia') = 155.9872284 ;
vfob('c_TextWapp','Oceania','SouthAsia') = 143.67276 ;
vfob('c_TextWapp','Oceania','NAmerica') = 153.5097504 ;
vfob('c_TextWapp','Oceania','LatinAmer') = 13.77084637 ;
vfob('c_TextWapp','Oceania','EU_28') = 323.6969299 ;
vfob('c_TextWapp','Oceania','MENA') = 67.69327545 ;
vfob('c_TextWapp','Oceania','SSA') = 34.23165512 ;
vfob('c_TextWapp','Oceania','RestofWorld') = 17.36819267 ;
vfob('c_TextWapp','EastAsia','Oceania') = 6066.476562 ;
vfob('c_TextWapp','EastAsia','EastAsia') = 69707.4375 ;
vfob('c_TextWapp','EastAsia','SEAsia') = 28342.30078 ;
vfob('c_TextWapp','EastAsia','SouthAsia') = 11945.64941 ;
vfob('c_TextWapp','EastAsia','NAmerica') = 58987.6875 ;
vfob('c_TextWapp','EastAsia','LatinAmer') = 15613.69043 ;
vfob('c_TextWapp','EastAsia','EU_28') = 64366.45703 ;
vfob('c_TextWapp','EastAsia','MENA') = 23175.00781 ;
vfob('c_TextWapp','EastAsia','SSA') = 9700.333984 ;
vfob('c_TextWapp','EastAsia','RestofWorld') = 17399.48242 ;
vfob('c_TextWapp','SEAsia','Oceania') = 586.8699341 ;
vfob('c_TextWapp','SEAsia','EastAsia') = 10458.31445 ;
vfob('c_TextWapp','SEAsia','SEAsia') = 4210.696289 ;
vfob('c_TextWapp','SEAsia','SouthAsia') = 1490.999878 ;
vfob('c_TextWapp','SEAsia','NAmerica') = 21662.92969 ;
vfob('c_TextWapp','SEAsia','LatinAmer') = 1417.456299 ;
vfob('c_TextWapp','SEAsia','EU_28') = 10675.28516 ;
vfob('c_TextWapp','SEAsia','MENA') = 3429.148682 ;
vfob('c_TextWapp','SEAsia','SSA') = 577.9711914 ;
vfob('c_TextWapp','SEAsia','RestofWorld') = 793.6313477 ;
vfob('c_TextWapp','SouthAsia','Oceania') = 885.9483643 ;
vfob('c_TextWapp','SouthAsia','EastAsia') = 4571.350586 ;
vfob('c_TextWapp','SouthAsia','SEAsia') = 1213.39502 ;
vfob('c_TextWapp','SouthAsia','SouthAsia') = 2775.943359 ;
vfob('c_TextWapp','SouthAsia','NAmerica') = 19406.82617 ;
vfob('c_TextWapp','SouthAsia','LatinAmer') = 2286.037842 ;
vfob('c_TextWapp','SouthAsia','EU_28') = 32573.64453 ;
vfob('c_TextWapp','SouthAsia','MENA') = 7680.729492 ;
vfob('c_TextWapp','SouthAsia','SSA') = 2035.38269 ;
vfob('c_TextWapp','SouthAsia','RestofWorld') = 1765.794312 ;
vfob('c_TextWapp','NAmerica','Oceania') = 391.008667 ;
vfob('c_TextWapp','NAmerica','EastAsia') = 2965.509277 ;
vfob('c_TextWapp','NAmerica','SEAsia') = 583.5131836 ;
vfob('c_TextWapp','NAmerica','SouthAsia') = 306.3420715 ;
vfob('c_TextWapp','NAmerica','NAmerica') = 15176.09863 ;
vfob('c_TextWapp','NAmerica','LatinAmer') = 5456.232422 ;
vfob('c_TextWapp','NAmerica','EU_28') = 2913.651855 ;
vfob('c_TextWapp','NAmerica','MENA') = 867.8517456 ;
vfob('c_TextWapp','NAmerica','SSA') = 252.5966492 ;
vfob('c_TextWapp','NAmerica','RestofWorld') = 434.9043579 ;
vfob('c_TextWapp','LatinAmer','Oceania') = 71.74494934 ;
vfob('c_TextWapp','LatinAmer','EastAsia') = 453.405365 ;
vfob('c_TextWapp','LatinAmer','SEAsia') = 101.1555405 ;
vfob('c_TextWapp','LatinAmer','SouthAsia') = 55.61384201 ;
vfob('c_TextWapp','LatinAmer','NAmerica') = 11427.84375 ;
vfob('c_TextWapp','LatinAmer','LatinAmer') = 5356.257812 ;
vfob('c_TextWapp','LatinAmer','EU_28') = 1347.807861 ;
vfob('c_TextWapp','LatinAmer','MENA') = 206.8310852 ;
vfob('c_TextWapp','LatinAmer','SSA') = 115.4184875 ;
vfob('c_TextWapp','LatinAmer','RestofWorld') = 104.2192841 ;
vfob('c_TextWapp','EU_28','Oceania') = 819.2094727 ;
vfob('c_TextWapp','EU_28','EastAsia') = 8084.403809 ;
vfob('c_TextWapp','EU_28','SEAsia') = 1487.235107 ;
vfob('c_TextWapp','EU_28','SouthAsia') = 981.7681885 ;
vfob('c_TextWapp','EU_28','NAmerica') = 7925.514648 ;
vfob('c_TextWapp','EU_28','LatinAmer') = 1766.706055 ;
vfob('c_TextWapp','EU_28','EU_28') = 113921.3359 ;
vfob('c_TextWapp','EU_28','MENA') = 11343.96973 ;
vfob('c_TextWapp','EU_28','SSA') = 1499.849365 ;
vfob('c_TextWapp','EU_28','RestofWorld') = 16307.05371 ;
vfob('c_TextWapp','MENA','Oceania') = 181.4058838 ;
vfob('c_TextWapp','MENA','EastAsia') = 1075.839722 ;
vfob('c_TextWapp','MENA','SEAsia') = 268.5634155 ;
vfob('c_TextWapp','MENA','SouthAsia') = 386.6010437 ;
vfob('c_TextWapp','MENA','NAmerica') = 5547.993164 ;
vfob('c_TextWapp','MENA','LatinAmer') = 432.2791443 ;
vfob('c_TextWapp','MENA','EU_28') = 28264.47266 ;
vfob('c_TextWapp','MENA','MENA') = 6110.893555 ;
vfob('c_TextWapp','MENA','SSA') = 664.463501 ;
vfob('c_TextWapp','MENA','RestofWorld') = 3436.33667 ;
vfob('c_TextWapp','SSA','Oceania') = 54.36729813 ;
vfob('c_TextWapp','SSA','EastAsia') = 234.3027954 ;
vfob('c_TextWapp','SSA','SEAsia') = 74.56367493 ;
vfob('c_TextWapp','SSA','SouthAsia') = 39.4046669 ;
vfob('c_TextWapp','SSA','NAmerica') = 1227.962402 ;
vfob('c_TextWapp','SSA','LatinAmer') = 52.25494766 ;
vfob('c_TextWapp','SSA','EU_28') = 1443.303589 ;
vfob('c_TextWapp','SSA','MENA') = 128.1373596 ;
vfob('c_TextWapp','SSA','SSA') = 1036.358154 ;
vfob('c_TextWapp','SSA','RestofWorld') = 76.11250305 ;
vfob('c_TextWapp','RestofWorld','Oceania') = 28.79694557 ;
vfob('c_TextWapp','RestofWorld','EastAsia') = 441.936676 ;
vfob('c_TextWapp','RestofWorld','SEAsia') = 89.20627594 ;
vfob('c_TextWapp','RestofWorld','SouthAsia') = 77.30461884 ;
vfob('c_TextWapp','RestofWorld','NAmerica') = 393.7702637 ;
vfob('c_TextWapp','RestofWorld','LatinAmer') = 49.38432693 ;
vfob('c_TextWapp','RestofWorld','EU_28') = 5224.221191 ;
vfob('c_TextWapp','RestofWorld','MENA') = 760.6646118 ;
vfob('c_TextWapp','RestofWorld','SSA') = 62.36700058 ;
vfob('c_TextWapp','RestofWorld','RestofWorld') = 1909.746948 ;
vfob('c_LightMnfc','Oceania','Oceania') = 3623.063721 ;
vfob('c_LightMnfc','Oceania','EastAsia') = 3947.263672 ;
vfob('c_LightMnfc','Oceania','SEAsia') = 1696.399902 ;
vfob('c_LightMnfc','Oceania','SouthAsia') = 280.8042603 ;
vfob('c_LightMnfc','Oceania','NAmerica') = 2545.902344 ;
vfob('c_LightMnfc','Oceania','LatinAmer') = 289.3599854 ;
vfob('c_LightMnfc','Oceania','EU_28') = 2020.608154 ;
vfob('c_LightMnfc','Oceania','MENA') = 1494.411865 ;
vfob('c_LightMnfc','Oceania','SSA') = 499.8693237 ;
vfob('c_LightMnfc','Oceania','RestofWorld') = 454.260437 ;
vfob('c_LightMnfc','EastAsia','Oceania') = 35590.65234 ;
vfob('c_LightMnfc','EastAsia','EastAsia') = 109985.1172 ;
vfob('c_LightMnfc','EastAsia','SEAsia') = 54226.81641 ;
vfob('c_LightMnfc','EastAsia','SouthAsia') = 18245.24609 ;
vfob('c_LightMnfc','EastAsia','NAmerica') = 217016.4531 ;
vfob('c_LightMnfc','EastAsia','LatinAmer') = 47132.24609 ;
vfob('c_LightMnfc','EastAsia','EU_28') = 149802.6562 ;
vfob('c_LightMnfc','EastAsia','MENA') = 58122.48047 ;
vfob('c_LightMnfc','EastAsia','SSA') = 34194.08203 ;
vfob('c_LightMnfc','EastAsia','RestofWorld') = 45461.07031 ;
vfob('c_LightMnfc','SEAsia','Oceania') = 8133.568359 ;
vfob('c_LightMnfc','SEAsia','EastAsia') = 25273.74219 ;
vfob('c_LightMnfc','SEAsia','SEAsia') = 24460.62109 ;
vfob('c_LightMnfc','SEAsia','SouthAsia') = 4522.596191 ;
vfob('c_LightMnfc','SEAsia','NAmerica') = 18112.71289 ;
vfob('c_LightMnfc','SEAsia','LatinAmer') = 4390.538086 ;
vfob('c_LightMnfc','SEAsia','EU_28') = 22235.26172 ;
vfob('c_LightMnfc','SEAsia','MENA') = 9920.24707 ;
vfob('c_LightMnfc','SEAsia','SSA') = 2446.131592 ;
vfob('c_LightMnfc','SEAsia','RestofWorld') = 3347.078125 ;
vfob('c_LightMnfc','SouthAsia','Oceania') = 848.0942383 ;
vfob('c_LightMnfc','SouthAsia','EastAsia') = 3721.952881 ;
vfob('c_LightMnfc','SouthAsia','SEAsia') = 3158.516846 ;
vfob('c_LightMnfc','SouthAsia','SouthAsia') = 2217.301758 ;
vfob('c_LightMnfc','SouthAsia','NAmerica') = 13029.88184 ;
vfob('c_LightMnfc','SouthAsia','LatinAmer') = 1512.740479 ;
vfob('c_LightMnfc','SouthAsia','EU_28') = 14625.80371 ;
vfob('c_LightMnfc','SouthAsia','MENA') = 24928.39648 ;
vfob('c_LightMnfc','SouthAsia','SSA') = 3268.557617 ;
vfob('c_LightMnfc','SouthAsia','RestofWorld') = 1163.295776 ;
vfob('c_LightMnfc','NAmerica','Oceania') = 8249.982422 ;
vfob('c_LightMnfc','NAmerica','EastAsia') = 59139.41406 ;
vfob('c_LightMnfc','NAmerica','SEAsia') = 16157.26855 ;
vfob('c_LightMnfc','NAmerica','SouthAsia') = 7335.530273 ;
vfob('c_LightMnfc','NAmerica','NAmerica') = 271204.0938 ;
vfob('c_LightMnfc','NAmerica','LatinAmer') = 32073.27539 ;
vfob('c_LightMnfc','NAmerica','EU_28') = 81040.83594 ;
vfob('c_LightMnfc','NAmerica','MENA') = 26217.00977 ;
vfob('c_LightMnfc','NAmerica','SSA') = 8003.946777 ;
vfob('c_LightMnfc','NAmerica','RestofWorld') = 9838.594727 ;
vfob('c_LightMnfc','LatinAmer','Oceania') = 693.2149048 ;
vfob('c_LightMnfc','LatinAmer','EastAsia') = 8761.700195 ;
vfob('c_LightMnfc','LatinAmer','SEAsia') = 1312.299438 ;
vfob('c_LightMnfc','LatinAmer','SouthAsia') = 372.7315369 ;
vfob('c_LightMnfc','LatinAmer','NAmerica') = 12679.55664 ;
vfob('c_LightMnfc','LatinAmer','LatinAmer') = 33892.85547 ;
vfob('c_LightMnfc','LatinAmer','EU_28') = 13364.24805 ;
vfob('c_LightMnfc','LatinAmer','MENA') = 1782.465576 ;
vfob('c_LightMnfc','LatinAmer','SSA') = 2361.981201 ;
vfob('c_LightMnfc','LatinAmer','RestofWorld') = 1363.639404 ;
vfob('c_LightMnfc','EU_28','Oceania') = 16309.80957 ;
vfob('c_LightMnfc','EU_28','EastAsia') = 97852.74219 ;
vfob('c_LightMnfc','EU_28','SEAsia') = 20849.26367 ;
vfob('c_LightMnfc','EU_28','SouthAsia') = 10929.43652 ;
vfob('c_LightMnfc','EU_28','NAmerica') = 115945.2578 ;
vfob('c_LightMnfc','EU_28','LatinAmer') = 28741.46289 ;
vfob('c_LightMnfc','EU_28','EU_28') = 880833.25 ;
vfob('c_LightMnfc','EU_28','MENA') = 83064.73438 ;
vfob('c_LightMnfc','EU_28','SSA') = 22452.75781 ;
vfob('c_LightMnfc','EU_28','RestofWorld') = 120502.0781 ;
vfob('c_LightMnfc','MENA','Oceania') = 508.0428162 ;
vfob('c_LightMnfc','MENA','EastAsia') = 2570.544678 ;
vfob('c_LightMnfc','MENA','SEAsia') = 847.0418091 ;
vfob('c_LightMnfc','MENA','SouthAsia') = 5249.98877 ;
vfob('c_LightMnfc','MENA','NAmerica') = 13617.21094 ;
vfob('c_LightMnfc','MENA','LatinAmer') = 1031.640747 ;
vfob('c_LightMnfc','MENA','EU_28') = 28404.23828 ;
vfob('c_LightMnfc','MENA','MENA') = 16924.61914 ;
vfob('c_LightMnfc','MENA','SSA') = 3159.633789 ;
vfob('c_LightMnfc','MENA','RestofWorld') = 6743.173828 ;
vfob('c_LightMnfc','SSA','Oceania') = 382.6640625 ;
vfob('c_LightMnfc','SSA','EastAsia') = 2194.879395 ;
vfob('c_LightMnfc','SSA','SEAsia') = 797.2085571 ;
vfob('c_LightMnfc','SSA','SouthAsia') = 309.0343628 ;
vfob('c_LightMnfc','SSA','NAmerica') = 4584.149902 ;
vfob('c_LightMnfc','SSA','LatinAmer') = 347.9335938 ;
vfob('c_LightMnfc','SSA','EU_28') = 7390.188965 ;
vfob('c_LightMnfc','SSA','MENA') = 828.1199341 ;
vfob('c_LightMnfc','SSA','SSA') = 7213.885742 ;
vfob('c_LightMnfc','SSA','RestofWorld') = 398.837677 ;
vfob('c_LightMnfc','RestofWorld','Oceania') = 368.1911926 ;
vfob('c_LightMnfc','RestofWorld','EastAsia') = 5539.549316 ;
vfob('c_LightMnfc','RestofWorld','SEAsia') = 1305.052246 ;
vfob('c_LightMnfc','RestofWorld','SouthAsia') = 2242.135498 ;
vfob('c_LightMnfc','RestofWorld','NAmerica') = 5547.327148 ;
vfob('c_LightMnfc','RestofWorld','LatinAmer') = 1379.315552 ;
vfob('c_LightMnfc','RestofWorld','EU_28') = 32565.50977 ;
vfob('c_LightMnfc','RestofWorld','MENA') = 5741.475098 ;
vfob('c_LightMnfc','RestofWorld','SSA') = 673.6758423 ;
vfob('c_LightMnfc','RestofWorld','RestofWorld') = 16534.00977 ;
vfob('c_HeavyMnfc','Oceania','Oceania') = 10967.89844 ;
vfob('c_HeavyMnfc','Oceania','EastAsia') = 20899.26367 ;
vfob('c_HeavyMnfc','Oceania','SEAsia') = 11474.97363 ;
vfob('c_HeavyMnfc','Oceania','SouthAsia') = 4749.049316 ;
vfob('c_HeavyMnfc','Oceania','NAmerica') = 6917.84082 ;
vfob('c_HeavyMnfc','Oceania','LatinAmer') = 1118.430298 ;
vfob('c_HeavyMnfc','Oceania','EU_28') = 6331.441895 ;
vfob('c_HeavyMnfc','Oceania','MENA') = 1613.69751 ;
vfob('c_HeavyMnfc','Oceania','SSA') = 1767.831665 ;
vfob('c_HeavyMnfc','Oceania','RestofWorld') = 955.1160889 ;
vfob('c_HeavyMnfc','EastAsia','Oceania') = 47182.97656 ;
vfob('c_HeavyMnfc','EastAsia','EastAsia') = 817156.8125 ;
vfob('c_HeavyMnfc','EastAsia','SEAsia') = 271722.1562 ;
vfob('c_HeavyMnfc','EastAsia','SouthAsia') = 80410.40625 ;
vfob('c_HeavyMnfc','EastAsia','NAmerica') = 468576.0312 ;
vfob('c_HeavyMnfc','EastAsia','LatinAmer') = 84540.95312 ;
vfob('c_HeavyMnfc','EastAsia','EU_28') = 350975.9688 ;
vfob('c_HeavyMnfc','EastAsia','MENA') = 103941.2969 ;
vfob('c_HeavyMnfc','EastAsia','SSA') = 36203.8125 ;
vfob('c_HeavyMnfc','EastAsia','RestofWorld') = 63253.71875 ;
vfob('c_HeavyMnfc','SEAsia','Oceania') = 23169.99023 ;
vfob('c_HeavyMnfc','SEAsia','EastAsia') = 205633.2344 ;
vfob('c_HeavyMnfc','SEAsia','SEAsia') = 157189.4688 ;
vfob('c_HeavyMnfc','SEAsia','SouthAsia') = 24428.76367 ;
vfob('c_HeavyMnfc','SEAsia','NAmerica') = 82993.74219 ;
vfob('c_HeavyMnfc','SEAsia','LatinAmer') = 9414.196289 ;
vfob('c_HeavyMnfc','SEAsia','EU_28') = 71120.59375 ;
vfob('c_HeavyMnfc','SEAsia','MENA') = 18050.05273 ;
vfob('c_HeavyMnfc','SEAsia','SSA') = 6071.635742 ;
vfob('c_HeavyMnfc','SEAsia','RestofWorld') = 9830.84668 ;
vfob('c_HeavyMnfc','SouthAsia','Oceania') = 1272.140625 ;
vfob('c_HeavyMnfc','SouthAsia','EastAsia') = 17230.86523 ;
vfob('c_HeavyMnfc','SouthAsia','SEAsia') = 11766.41309 ;
vfob('c_HeavyMnfc','SouthAsia','SouthAsia') = 9253.913086 ;
vfob('c_HeavyMnfc','SouthAsia','NAmerica') = 19497.20898 ;
vfob('c_HeavyMnfc','SouthAsia','LatinAmer') = 9901.46875 ;
vfob('c_HeavyMnfc','SouthAsia','EU_28') = 29123.80078 ;
vfob('c_HeavyMnfc','SouthAsia','MENA') = 24451.04688 ;
vfob('c_HeavyMnfc','SouthAsia','SSA') = 13675.39355 ;
vfob('c_HeavyMnfc','SouthAsia','RestofWorld') = 3686.405762 ;
vfob('c_HeavyMnfc','NAmerica','Oceania') = 25149.10938 ;
vfob('c_HeavyMnfc','NAmerica','EastAsia') = 183463.0156 ;
vfob('c_HeavyMnfc','NAmerica','SEAsia') = 55137.5625 ;
vfob('c_HeavyMnfc','NAmerica','SouthAsia') = 18526.13867 ;
vfob('c_HeavyMnfc','NAmerica','NAmerica') = 459818.5312 ;
vfob('c_HeavyMnfc','NAmerica','LatinAmer') = 128618.5547 ;
vfob('c_HeavyMnfc','NAmerica','EU_28') = 241136.9844 ;
vfob('c_HeavyMnfc','NAmerica','MENA') = 51963.40625 ;
vfob('c_HeavyMnfc','NAmerica','SSA') = 17854.48828 ;
vfob('c_HeavyMnfc','NAmerica','RestofWorld') = 38663.01953 ;
vfob('c_HeavyMnfc','LatinAmer','Oceania') = 1521.538208 ;
vfob('c_HeavyMnfc','LatinAmer','EastAsia') = 35388.56641 ;
vfob('c_HeavyMnfc','LatinAmer','SEAsia') = 5685.853027 ;
vfob('c_HeavyMnfc','LatinAmer','SouthAsia') = 1233.799438 ;
vfob('c_HeavyMnfc','LatinAmer','NAmerica') = 79312.05469 ;
vfob('c_HeavyMnfc','LatinAmer','LatinAmer') = 67945.10938 ;
vfob('c_HeavyMnfc','LatinAmer','EU_28') = 29181.93945 ;
vfob('c_HeavyMnfc','LatinAmer','MENA') = 3761.756348 ;
vfob('c_HeavyMnfc','LatinAmer','SSA') = 4315.414551 ;
vfob('c_HeavyMnfc','LatinAmer','RestofWorld') = 11776.62109 ;
vfob('c_HeavyMnfc','EU_28','Oceania') = 29388.81445 ;
vfob('c_HeavyMnfc','EU_28','EastAsia') = 213322.5781 ;
vfob('c_HeavyMnfc','EU_28','SEAsia') = 53293.88281 ;
vfob('c_HeavyMnfc','EU_28','SouthAsia') = 41510.16016 ;
vfob('c_HeavyMnfc','EU_28','NAmerica') = 275185.2812 ;
vfob('c_HeavyMnfc','EU_28','LatinAmer') = 69759.95312 ;
vfob('c_HeavyMnfc','EU_28','EU_28') = 1715428.375 ;
vfob('c_HeavyMnfc','EU_28','MENA') = 196334.4375 ;
vfob('c_HeavyMnfc','EU_28','SSA') = 53518.09766 ;
vfob('c_HeavyMnfc','EU_28','RestofWorld') = 263154.1562 ;
vfob('c_HeavyMnfc','MENA','Oceania') = 3056.772705 ;
vfob('c_HeavyMnfc','MENA','EastAsia') = 74209.3125 ;
vfob('c_HeavyMnfc','MENA','SEAsia') = 22749.14062 ;
vfob('c_HeavyMnfc','MENA','SouthAsia') = 38359.06641 ;
vfob('c_HeavyMnfc','MENA','NAmerica') = 27574.43359 ;
vfob('c_HeavyMnfc','MENA','LatinAmer') = 8546.568359 ;
vfob('c_HeavyMnfc','MENA','EU_28') = 86459.03906 ;
vfob('c_HeavyMnfc','MENA','MENA') = 83036.72656 ;
vfob('c_HeavyMnfc','MENA','SSA') = 17675.81641 ;
vfob('c_HeavyMnfc','MENA','RestofWorld') = 14120.86426 ;
vfob('c_HeavyMnfc','SSA','Oceania') = 534.9724731 ;
vfob('c_HeavyMnfc','SSA','EastAsia') = 15367.50977 ;
vfob('c_HeavyMnfc','SSA','SEAsia') = 3338.334229 ;
vfob('c_HeavyMnfc','SSA','SouthAsia') = 8753.77832 ;
vfob('c_HeavyMnfc','SSA','NAmerica') = 9223.745117 ;
vfob('c_HeavyMnfc','SSA','LatinAmer') = 1163.704102 ;
vfob('c_HeavyMnfc','SSA','EU_28') = 16914.65234 ;
vfob('c_HeavyMnfc','SSA','MENA') = 5604.098633 ;
vfob('c_HeavyMnfc','SSA','SSA') = 17732.38672 ;
vfob('c_HeavyMnfc','SSA','RestofWorld') = 12275.74316 ;
vfob('c_HeavyMnfc','RestofWorld','Oceania') = 3434.95874 ;
vfob('c_HeavyMnfc','RestofWorld','EastAsia') = 58078.75 ;
vfob('c_HeavyMnfc','RestofWorld','SEAsia') = 21114.58008 ;
vfob('c_HeavyMnfc','RestofWorld','SouthAsia') = 38397.52734 ;
vfob('c_HeavyMnfc','RestofWorld','NAmerica') = 55257.16406 ;
vfob('c_HeavyMnfc','RestofWorld','LatinAmer') = 13657.52246 ;
vfob('c_HeavyMnfc','RestofWorld','EU_28') = 229288.1094 ;
vfob('c_HeavyMnfc','RestofWorld','MENA') = 45985.52734 ;
vfob('c_HeavyMnfc','RestofWorld','SSA') = 4418.249023 ;
vfob('c_HeavyMnfc','RestofWorld','RestofWorld') = 51303.27344 ;
vfob('c_Util_Cons','Oceania','Oceania') = 31.07725143 ;
vfob('c_Util_Cons','Oceania','EastAsia') = 201.0083771 ;
vfob('c_Util_Cons','Oceania','SEAsia') = 101.6341858 ;
vfob('c_Util_Cons','Oceania','SouthAsia') = 11.61959457 ;
vfob('c_Util_Cons','Oceania','NAmerica') = 79.40089417 ;
vfob('c_Util_Cons','Oceania','LatinAmer') = 49.84052277 ;
vfob('c_Util_Cons','Oceania','EU_28') = 383.8754883 ;
vfob('c_Util_Cons','Oceania','MENA') = 127.0182343 ;
vfob('c_Util_Cons','Oceania','SSA') = 23.77604485 ;
vfob('c_Util_Cons','Oceania','RestofWorld') = 78.74828339 ;
vfob('c_Util_Cons','EastAsia','Oceania') = 366.0947571 ;
vfob('c_Util_Cons','EastAsia','EastAsia') = 6357.323242 ;
vfob('c_Util_Cons','EastAsia','SEAsia') = 2104.691406 ;
vfob('c_Util_Cons','EastAsia','SouthAsia') = 451.4074097 ;
vfob('c_Util_Cons','EastAsia','NAmerica') = 2141.272949 ;
vfob('c_Util_Cons','EastAsia','LatinAmer') = 486.0794067 ;
vfob('c_Util_Cons','EastAsia','EU_28') = 11890.67676 ;
vfob('c_Util_Cons','EastAsia','MENA') = 4096.105469 ;
vfob('c_Util_Cons','EastAsia','SSA') = 1524.092529 ;
vfob('c_Util_Cons','EastAsia','RestofWorld') = 4494.487305 ;
vfob('c_Util_Cons','SEAsia','Oceania') = 97.43195343 ;
vfob('c_Util_Cons','SEAsia','EastAsia') = 1373.67334 ;
vfob('c_Util_Cons','SEAsia','SEAsia') = 382.1662903 ;
vfob('c_Util_Cons','SEAsia','SouthAsia') = 107.3237152 ;
vfob('c_Util_Cons','SEAsia','NAmerica') = 393.7479858 ;
vfob('c_Util_Cons','SEAsia','LatinAmer') = 142.2795258 ;
vfob('c_Util_Cons','SEAsia','EU_28') = 2745.166992 ;
vfob('c_Util_Cons','SEAsia','MENA') = 812.9971924 ;
vfob('c_Util_Cons','SEAsia','SSA') = 236.680542 ;
vfob('c_Util_Cons','SEAsia','RestofWorld') = 861.5371094 ;
vfob('c_Util_Cons','SouthAsia','Oceania') = 19.71509743 ;
vfob('c_Util_Cons','SouthAsia','EastAsia') = 237.1269989 ;
vfob('c_Util_Cons','SouthAsia','SEAsia') = 93.98284149 ;
vfob('c_Util_Cons','SouthAsia','SouthAsia') = 10.60282135 ;
vfob('c_Util_Cons','SouthAsia','NAmerica') = 61.55842209 ;
vfob('c_Util_Cons','SouthAsia','LatinAmer') = 52.26968765 ;
vfob('c_Util_Cons','SouthAsia','EU_28') = 802.0194092 ;
vfob('c_Util_Cons','SouthAsia','MENA') = 255.4254608 ;
vfob('c_Util_Cons','SouthAsia','SSA') = 61.51704788 ;
vfob('c_Util_Cons','SouthAsia','RestofWorld') = 171.6964722 ;
vfob('c_Util_Cons','NAmerica','Oceania') = 255.6985321 ;
vfob('c_Util_Cons','NAmerica','EastAsia') = 2842.396729 ;
vfob('c_Util_Cons','NAmerica','SEAsia') = 829.7210083 ;
vfob('c_Util_Cons','NAmerica','SouthAsia') = 149.27742 ;
vfob('c_Util_Cons','NAmerica','NAmerica') = 3845.733887 ;
vfob('c_Util_Cons','NAmerica','LatinAmer') = 357.4509888 ;
vfob('c_Util_Cons','NAmerica','EU_28') = 3967.249756 ;
vfob('c_Util_Cons','NAmerica','MENA') = 1951.361328 ;
vfob('c_Util_Cons','NAmerica','SSA') = 511.8539124 ;
vfob('c_Util_Cons','NAmerica','RestofWorld') = 1926.356934 ;
vfob('c_Util_Cons','LatinAmer','Oceania') = 46.47725296 ;
vfob('c_Util_Cons','LatinAmer','EastAsia') = 322.9883423 ;
vfob('c_Util_Cons','LatinAmer','SEAsia') = 206.2909546 ;
vfob('c_Util_Cons','LatinAmer','SouthAsia') = 54.46353149 ;
vfob('c_Util_Cons','LatinAmer','NAmerica') = 209.4647217 ;
vfob('c_Util_Cons','LatinAmer','LatinAmer') = 1090.684814 ;
vfob('c_Util_Cons','LatinAmer','EU_28') = 1094.125122 ;
vfob('c_Util_Cons','LatinAmer','MENA') = 422.6958618 ;
vfob('c_Util_Cons','LatinAmer','SSA') = 186.4594421 ;
vfob('c_Util_Cons','LatinAmer','RestofWorld') = 532.9012451 ;
vfob('c_Util_Cons','EU_28','Oceania') = 525.5808716 ;
vfob('c_Util_Cons','EU_28','EastAsia') = 7820.53125 ;
vfob('c_Util_Cons','EU_28','SEAsia') = 3339.02002 ;
vfob('c_Util_Cons','EU_28','SouthAsia') = 862.8706665 ;
vfob('c_Util_Cons','EU_28','NAmerica') = 1946.275146 ;
vfob('c_Util_Cons','EU_28','LatinAmer') = 2228.263184 ;
vfob('c_Util_Cons','EU_28','EU_28') = 50810.94531 ;
vfob('c_Util_Cons','EU_28','MENA') = 7784.186035 ;
vfob('c_Util_Cons','EU_28','SSA') = 2911.449707 ;
vfob('c_Util_Cons','EU_28','RestofWorld') = 14605.10449 ;
vfob('c_Util_Cons','MENA','Oceania') = 315.7593079 ;
vfob('c_Util_Cons','MENA','EastAsia') = 1619.862305 ;
vfob('c_Util_Cons','MENA','SEAsia') = 778.6488037 ;
vfob('c_Util_Cons','MENA','SouthAsia') = 196.4408264 ;
vfob('c_Util_Cons','MENA','NAmerica') = 1185.112549 ;
vfob('c_Util_Cons','MENA','LatinAmer') = 669.9071655 ;
vfob('c_Util_Cons','MENA','EU_28') = 5639.129395 ;
vfob('c_Util_Cons','MENA','MENA') = 2598.879639 ;
vfob('c_Util_Cons','MENA','SSA') = 375.0569458 ;
vfob('c_Util_Cons','MENA','RestofWorld') = 2366.30835 ;
vfob('c_Util_Cons','SSA','Oceania') = 32.03158188 ;
vfob('c_Util_Cons','SSA','EastAsia') = 254.3527527 ;
vfob('c_Util_Cons','SSA','SEAsia') = 144.2131348 ;
vfob('c_Util_Cons','SSA','SouthAsia') = 27.73508072 ;
vfob('c_Util_Cons','SSA','NAmerica') = 159.2519226 ;
vfob('c_Util_Cons','SSA','LatinAmer') = 124.1552811 ;
vfob('c_Util_Cons','SSA','EU_28') = 928.7954712 ;
vfob('c_Util_Cons','SSA','MENA') = 392.601532 ;
vfob('c_Util_Cons','SSA','SSA') = 1045.420776 ;
vfob('c_Util_Cons','SSA','RestofWorld') = 252.3145447 ;
vfob('c_Util_Cons','RestofWorld','Oceania') = 196.6998596 ;
vfob('c_Util_Cons','RestofWorld','EastAsia') = 1458.554199 ;
vfob('c_Util_Cons','RestofWorld','SEAsia') = 1037.572632 ;
vfob('c_Util_Cons','RestofWorld','SouthAsia') = 165.8195496 ;
vfob('c_Util_Cons','RestofWorld','NAmerica') = 818.8847656 ;
vfob('c_Util_Cons','RestofWorld','LatinAmer') = 856.9828491 ;
vfob('c_Util_Cons','RestofWorld','EU_28') = 12403.9707 ;
vfob('c_Util_Cons','RestofWorld','MENA') = 3945.540527 ;
vfob('c_Util_Cons','RestofWorld','SSA') = 488.2766418 ;
vfob('c_Util_Cons','RestofWorld','RestofWorld') = 3622.735107 ;
vfob('c_TransComm','Oceania','Oceania') = 2169.616455 ;
vfob('c_TransComm','Oceania','EastAsia') = 6539.338867 ;
vfob('c_TransComm','Oceania','SEAsia') = 1562.600342 ;
vfob('c_TransComm','Oceania','SouthAsia') = 348.3964844 ;
vfob('c_TransComm','Oceania','NAmerica') = 4447.299316 ;
vfob('c_TransComm','Oceania','LatinAmer') = 751.6378174 ;
vfob('c_TransComm','Oceania','EU_28') = 7174.994629 ;
vfob('c_TransComm','Oceania','MENA') = 1683.146484 ;
vfob('c_TransComm','Oceania','SSA') = 553.6177979 ;
vfob('c_TransComm','Oceania','RestofWorld') = 714.2366333 ;
vfob('c_TransComm','EastAsia','Oceania') = 4212.180664 ;
vfob('c_TransComm','EastAsia','EastAsia') = 65740.33594 ;
vfob('c_TransComm','EastAsia','SEAsia') = 15686.9043 ;
vfob('c_TransComm','EastAsia','SouthAsia') = 3984.506836 ;
vfob('c_TransComm','EastAsia','NAmerica') = 34000.04688 ;
vfob('c_TransComm','EastAsia','LatinAmer') = 6217.623047 ;
vfob('c_TransComm','EastAsia','EU_28') = 58985.67188 ;
vfob('c_TransComm','EastAsia','MENA') = 11513.73438 ;
vfob('c_TransComm','EastAsia','SSA') = 4201.472656 ;
vfob('c_TransComm','EastAsia','RestofWorld') = 6782.009766 ;
vfob('c_TransComm','SEAsia','Oceania') = 2004.836426 ;
vfob('c_TransComm','SEAsia','EastAsia') = 16652.25781 ;
vfob('c_TransComm','SEAsia','SEAsia') = 2442.834961 ;
vfob('c_TransComm','SEAsia','SouthAsia') = 1960.946777 ;
vfob('c_TransComm','SEAsia','NAmerica') = 10710.11719 ;
vfob('c_TransComm','SEAsia','LatinAmer') = 2438.797363 ;
vfob('c_TransComm','SEAsia','EU_28') = 30842.5957 ;
vfob('c_TransComm','SEAsia','MENA') = 3997.199951 ;
vfob('c_TransComm','SEAsia','SSA') = 1551.950195 ;
vfob('c_TransComm','SEAsia','RestofWorld') = 3739.834961 ;
vfob('c_TransComm','SouthAsia','Oceania') = 492.8401489 ;
vfob('c_TransComm','SouthAsia','EastAsia') = 5351.861816 ;
vfob('c_TransComm','SouthAsia','SEAsia') = 1405.473877 ;
vfob('c_TransComm','SouthAsia','SouthAsia') = 138.3771057 ;
vfob('c_TransComm','SouthAsia','NAmerica') = 4213.526367 ;
vfob('c_TransComm','SouthAsia','LatinAmer') = 716.8010864 ;
vfob('c_TransComm','SouthAsia','EU_28') = 7466.467285 ;
vfob('c_TransComm','SouthAsia','MENA') = 1734.245361 ;
vfob('c_TransComm','SouthAsia','SSA') = 441.8512878 ;
vfob('c_TransComm','SouthAsia','RestofWorld') = 970.2805176 ;
vfob('c_TransComm','NAmerica','Oceania') = 3273.124268 ;
vfob('c_TransComm','NAmerica','EastAsia') = 27415.13086 ;
vfob('c_TransComm','NAmerica','SEAsia') = 6990.8125 ;
vfob('c_TransComm','NAmerica','SouthAsia') = 3207.255859 ;
vfob('c_TransComm','NAmerica','NAmerica') = 21350.7207 ;
vfob('c_TransComm','NAmerica','LatinAmer') = 4646.54834 ;
vfob('c_TransComm','NAmerica','EU_28') = 55497.03906 ;
vfob('c_TransComm','NAmerica','MENA') = 10289.19824 ;
vfob('c_TransComm','NAmerica','SSA') = 2744.233398 ;
vfob('c_TransComm','NAmerica','RestofWorld') = 4709.695801 ;
vfob('c_TransComm','LatinAmer','Oceania') = 919.934021 ;
vfob('c_TransComm','LatinAmer','EastAsia') = 7700.701172 ;
vfob('c_TransComm','LatinAmer','SEAsia') = 2465.787598 ;
vfob('c_TransComm','LatinAmer','SouthAsia') = 922.2738647 ;
vfob('c_TransComm','LatinAmer','NAmerica') = 6838.367676 ;
vfob('c_TransComm','LatinAmer','LatinAmer') = 1695.694336 ;
vfob('c_TransComm','LatinAmer','EU_28') = 21363.47656 ;
vfob('c_TransComm','LatinAmer','MENA') = 3560.369385 ;
vfob('c_TransComm','LatinAmer','SSA') = 926.9926147 ;
vfob('c_TransComm','LatinAmer','RestofWorld') = 3073.308105 ;
vfob('c_TransComm','EU_28','Oceania') = 8274.443359 ;
vfob('c_TransComm','EU_28','EastAsia') = 52822.07812 ;
vfob('c_TransComm','EU_28','SEAsia') = 22939.01562 ;
vfob('c_TransComm','EU_28','SouthAsia') = 10831.97461 ;
vfob('c_TransComm','EU_28','NAmerica') = 60427.17969 ;
vfob('c_TransComm','EU_28','LatinAmer') = 22014.25 ;
vfob('c_TransComm','EU_28','EU_28') = 316962.3125 ;
vfob('c_TransComm','EU_28','MENA') = 28538.63281 ;
vfob('c_TransComm','EU_28','SSA') = 12248.56934 ;
vfob('c_TransComm','EU_28','RestofWorld') = 40754.24609 ;
vfob('c_TransComm','MENA','Oceania') = 1772.074219 ;
vfob('c_TransComm','MENA','EastAsia') = 15178.52441 ;
vfob('c_TransComm','MENA','SEAsia') = 4101.641602 ;
vfob('c_TransComm','MENA','SouthAsia') = 1782.15271 ;
vfob('c_TransComm','MENA','NAmerica') = 16937.83984 ;
vfob('c_TransComm','MENA','LatinAmer') = 3612.134521 ;
vfob('c_TransComm','MENA','EU_28') = 40371.64844 ;
vfob('c_TransComm','MENA','MENA') = 4900.626953 ;
vfob('c_TransComm','MENA','SSA') = 1857.097778 ;
vfob('c_TransComm','MENA','RestofWorld') = 6201.303223 ;
vfob('c_TransComm','SSA','Oceania') = 429.4367981 ;
vfob('c_TransComm','SSA','EastAsia') = 3114.362305 ;
vfob('c_TransComm','SSA','SEAsia') = 998.7744751 ;
vfob('c_TransComm','SSA','SouthAsia') = 317.2618408 ;
vfob('c_TransComm','SSA','NAmerica') = 3187.952148 ;
vfob('c_TransComm','SSA','LatinAmer') = 675.1118164 ;
vfob('c_TransComm','SSA','EU_28') = 8443.737305 ;
vfob('c_TransComm','SSA','MENA') = 1192.515625 ;
vfob('c_TransComm','SSA','SSA') = 354.2862854 ;
vfob('c_TransComm','SSA','RestofWorld') = 896.7719116 ;
vfob('c_TransComm','RestofWorld','Oceania') = 947.9411621 ;
vfob('c_TransComm','RestofWorld','EastAsia') = 10903.88184 ;
vfob('c_TransComm','RestofWorld','SEAsia') = 4127.525391 ;
vfob('c_TransComm','RestofWorld','SouthAsia') = 1044.633789 ;
vfob('c_TransComm','RestofWorld','NAmerica') = 8281.378906 ;
vfob('c_TransComm','RestofWorld','LatinAmer') = 2161.365723 ;
vfob('c_TransComm','RestofWorld','EU_28') = 36154.76172 ;
vfob('c_TransComm','RestofWorld','MENA') = 4304.719238 ;
vfob('c_TransComm','RestofWorld','SSA') = 1237.386597 ;
vfob('c_TransComm','RestofWorld','RestofWorld') = 3337.639893 ;
vfob('c_OthService','Oceania','Oceania') = 2283.932617 ;
vfob('c_OthService','Oceania','EastAsia') = 6401.10791 ;
vfob('c_OthService','Oceania','SEAsia') = 2127.986084 ;
vfob('c_OthService','Oceania','SouthAsia') = 573.2736206 ;
vfob('c_OthService','Oceania','NAmerica') = 5652.233887 ;
vfob('c_OthService','Oceania','LatinAmer') = 786.1976318 ;
vfob('c_OthService','Oceania','EU_28') = 7780.802734 ;
vfob('c_OthService','Oceania','MENA') = 2293.664307 ;
vfob('c_OthService','Oceania','SSA') = 659.3297729 ;
vfob('c_OthService','Oceania','RestofWorld') = 841.8104858 ;
vfob('c_OthService','EastAsia','Oceania') = 2661.219482 ;
vfob('c_OthService','EastAsia','EastAsia') = 24510.78125 ;
vfob('c_OthService','EastAsia','SEAsia') = 10720.59766 ;
vfob('c_OthService','EastAsia','SouthAsia') = 5830.672363 ;
vfob('c_OthService','EastAsia','NAmerica') = 40831.04688 ;
vfob('c_OthService','EastAsia','LatinAmer') = 6029.530762 ;
vfob('c_OthService','EastAsia','EU_28') = 51103.73047 ;
vfob('c_OthService','EastAsia','MENA') = 13700.07715 ;
vfob('c_OthService','EastAsia','SSA') = 3753.361328 ;
vfob('c_OthService','EastAsia','RestofWorld') = 6419.271484 ;
vfob('c_OthService','SEAsia','Oceania') = 2375.387695 ;
vfob('c_OthService','SEAsia','EastAsia') = 12545.82324 ;
vfob('c_OthService','SEAsia','SEAsia') = 2450.027588 ;
vfob('c_OthService','SEAsia','SouthAsia') = 4212.79248 ;
vfob('c_OthService','SEAsia','NAmerica') = 11423.6748 ;
vfob('c_OthService','SEAsia','LatinAmer') = 2735.078369 ;
vfob('c_OthService','SEAsia','EU_28') = 30411.67188 ;
vfob('c_OthService','SEAsia','MENA') = 4138.499512 ;
vfob('c_OthService','SEAsia','SSA') = 1649.08313 ;
vfob('c_OthService','SEAsia','RestofWorld') = 3016.982666 ;
vfob('c_OthService','SouthAsia','Oceania') = 925.135498 ;
vfob('c_OthService','SouthAsia','EastAsia') = 7902.925781 ;
vfob('c_OthService','SouthAsia','SEAsia') = 2953.47168 ;
vfob('c_OthService','SouthAsia','SouthAsia') = 317.1229553 ;
vfob('c_OthService','SouthAsia','NAmerica') = 23476.5957 ;
vfob('c_OthService','SouthAsia','LatinAmer') = 1873.734741 ;
vfob('c_OthService','SouthAsia','EU_28') = 25990.93555 ;
vfob('c_OthService','SouthAsia','MENA') = 3121.552734 ;
vfob('c_OthService','SouthAsia','SSA') = 1184.272705 ;
vfob('c_OthService','SouthAsia','RestofWorld') = 2095.656494 ;
vfob('c_OthService','NAmerica','Oceania') = 6903.589844 ;
vfob('c_OthService','NAmerica','EastAsia') = 61267.43359 ;
vfob('c_OthService','NAmerica','SEAsia') = 16068.8877 ;
vfob('c_OthService','NAmerica','SouthAsia') = 11846.02148 ;
vfob('c_OthService','NAmerica','NAmerica') = 46485.67578 ;
vfob('c_OthService','NAmerica','LatinAmer') = 11753.08398 ;
vfob('c_OthService','NAmerica','EU_28') = 149312.7031 ;
vfob('c_OthService','NAmerica','MENA') = 35297.79297 ;
vfob('c_OthService','NAmerica','SSA') = 10327.73535 ;
vfob('c_OthService','NAmerica','RestofWorld') = 14985.01367 ;
vfob('c_OthService','LatinAmer','Oceania') = 767.5485229 ;
vfob('c_OthService','LatinAmer','EastAsia') = 6080.470215 ;
vfob('c_OthService','LatinAmer','SEAsia') = 2984.134277 ;
vfob('c_OthService','LatinAmer','SouthAsia') = 1425.15332 ;
vfob('c_OthService','LatinAmer','NAmerica') = 10734.77051 ;
vfob('c_OthService','LatinAmer','LatinAmer') = 1358.890869 ;
vfob('c_OthService','LatinAmer','EU_28') = 18873.36719 ;
vfob('c_OthService','LatinAmer','MENA') = 4633.399902 ;
vfob('c_OthService','LatinAmer','SSA') = 1248.408447 ;
vfob('c_OthService','LatinAmer','RestofWorld') = 2278.50415 ;
vfob('c_OthService','EU_28','Oceania') = 8443.917969 ;
vfob('c_OthService','EU_28','EastAsia') = 56998.12109 ;
vfob('c_OthService','EU_28','SEAsia') = 31403.49414 ;
vfob('c_OthService','EU_28','SouthAsia') = 20280.83594 ;
vfob('c_OthService','EU_28','NAmerica') = 129155.0859 ;
vfob('c_OthService','EU_28','LatinAmer') = 27664.25195 ;
vfob('c_OthService','EU_28','EU_28') = 491871.9062 ;
vfob('c_OthService','EU_28','MENA') = 47819.35156 ;
vfob('c_OthService','EU_28','SSA') = 19573.19727 ;
vfob('c_OthService','EU_28','RestofWorld') = 62218.03125 ;
vfob('c_OthService','MENA','Oceania') = 1445.43689 ;
vfob('c_OthService','MENA','EastAsia') = 12839.02148 ;
vfob('c_OthService','MENA','SEAsia') = 3465.295898 ;
vfob('c_OthService','MENA','SouthAsia') = 2367.249756 ;
vfob('c_OthService','MENA','NAmerica') = 18135.54883 ;
vfob('c_OthService','MENA','LatinAmer') = 3768.030029 ;
vfob('c_OthService','MENA','EU_28') = 34945.88281 ;
vfob('c_OthService','MENA','MENA') = 5371.029785 ;
vfob('c_OthService','MENA','SSA') = 1977.580933 ;
vfob('c_OthService','MENA','RestofWorld') = 5075.946289 ;
vfob('c_OthService','SSA','Oceania') = 308.29599 ;
vfob('c_OthService','SSA','EastAsia') = 2903.436035 ;
vfob('c_OthService','SSA','SEAsia') = 790.6919556 ;
vfob('c_OthService','SSA','SouthAsia') = 361.8479004 ;
vfob('c_OthService','SSA','NAmerica') = 4691.35791 ;
vfob('c_OthService','SSA','LatinAmer') = 644.6809692 ;
vfob('c_OthService','SSA','EU_28') = 6700.551758 ;
vfob('c_OthService','SSA','MENA') = 2125.437988 ;
vfob('c_OthService','SSA','SSA') = 492.7622986 ;
vfob('c_OthService','SSA','RestofWorld') = 848.0107422 ;
vfob('c_OthService','RestofWorld','Oceania') = 842.0559692 ;
vfob('c_OthService','RestofWorld','EastAsia') = 8420.21582 ;
vfob('c_OthService','RestofWorld','SEAsia') = 5079.750488 ;
vfob('c_OthService','RestofWorld','SouthAsia') = 2429.520752 ;
vfob('c_OthService','RestofWorld','NAmerica') = 18953.99609 ;
vfob('c_OthService','RestofWorld','LatinAmer') = 2311.362549 ;
vfob('c_OthService','RestofWorld','EU_28') = 48936.79297 ;
vfob('c_OthService','RestofWorld','MENA') = 6074.623535 ;
vfob('c_OthService','RestofWorld','SSA') = 2068.710938 ;
vfob('c_OthService','RestofWorld','RestofWorld') = 3546.81543 ;

* vcif data (1000 cells)
vcif('c_Crops','Oceania','Oceania') = 992.1069946 ;
vcif('c_Crops','Oceania','EastAsia') = 6136.081055 ;
vcif('c_Crops','Oceania','SEAsia') = 4387.617188 ;
vcif('c_Crops','Oceania','SouthAsia') = 1044.07666 ;
vcif('c_Crops','Oceania','NAmerica') = 385.9144592 ;
vcif('c_Crops','Oceania','LatinAmer') = 51.31932449 ;
vcif('c_Crops','Oceania','EU_28') = 1629.849731 ;
vcif('c_Crops','Oceania','MENA') = 2415.882812 ;
vcif('c_Crops','Oceania','SSA') = 457.6585083 ;
vcif('c_Crops','Oceania','RestofWorld') = 60.19305038 ;
vcif('c_Crops','EastAsia','Oceania') = 152.3969116 ;
vcif('c_Crops','EastAsia','EastAsia') = 4370.312988 ;
vcif('c_Crops','EastAsia','SEAsia') = 5067.958984 ;
vcif('c_Crops','EastAsia','SouthAsia') = 902.3258667 ;
vcif('c_Crops','EastAsia','NAmerica') = 1362.643066 ;
vcif('c_Crops','EastAsia','LatinAmer') = 470.9642944 ;
vcif('c_Crops','EastAsia','EU_28') = 1898.40332 ;
vcif('c_Crops','EastAsia','MENA') = 1142.011841 ;
vcif('c_Crops','EastAsia','SSA') = 844.6310425 ;
vcif('c_Crops','EastAsia','RestofWorld') = 1230.284668 ;
vcif('c_Crops','SEAsia','Oceania') = 594.4607544 ;
vcif('c_Crops','SEAsia','EastAsia') = 8749.767578 ;
vcif('c_Crops','SEAsia','SEAsia') = 6430.47168 ;
vcif('c_Crops','SEAsia','SouthAsia') = 2052.561523 ;
vcif('c_Crops','SEAsia','NAmerica') = 3023.987061 ;
vcif('c_Crops','SEAsia','LatinAmer') = 171.9546509 ;
vcif('c_Crops','SEAsia','EU_28') = 3761.407959 ;
vcif('c_Crops','SEAsia','MENA') = 2131.630859 ;
vcif('c_Crops','SEAsia','SSA') = 3894.45459 ;
vcif('c_Crops','SEAsia','RestofWorld') = 689.7945557 ;
vcif('c_Crops','SouthAsia','Oceania') = 196.7597961 ;
vcif('c_Crops','SouthAsia','EastAsia') = 3483.48584 ;
vcif('c_Crops','SouthAsia','SEAsia') = 3370.056152 ;
vcif('c_Crops','SouthAsia','SouthAsia') = 4045.820801 ;
vcif('c_Crops','SouthAsia','NAmerica') = 1452.3396 ;
vcif('c_Crops','SouthAsia','LatinAmer') = 178.7914734 ;
vcif('c_Crops','SouthAsia','EU_28') = 2720.454102 ;
vcif('c_Crops','SouthAsia','MENA') = 7124.632812 ;
vcif('c_Crops','SouthAsia','SSA') = 1791.977173 ;
vcif('c_Crops','SouthAsia','RestofWorld') = 1130.354248 ;
vcif('c_Crops','NAmerica','Oceania') = 478.9811707 ;
vcif('c_Crops','NAmerica','EastAsia') = 45725.59375 ;
vcif('c_Crops','NAmerica','SEAsia') = 6967.51709 ;
vcif('c_Crops','NAmerica','SouthAsia') = 3498.606201 ;
vcif('c_Crops','NAmerica','NAmerica') = 34334.22266 ;
vcif('c_Crops','NAmerica','LatinAmer') = 8535.306641 ;
vcif('c_Crops','NAmerica','EU_28') = 8229.667969 ;
vcif('c_Crops','NAmerica','MENA') = 12016.68555 ;
vcif('c_Crops','NAmerica','SSA') = 4773.899414 ;
vcif('c_Crops','NAmerica','RestofWorld') = 878.178894 ;
vcif('c_Crops','LatinAmer','Oceania') = 301.225769 ;
vcif('c_Crops','LatinAmer','EastAsia') = 27313.16406 ;
vcif('c_Crops','LatinAmer','SEAsia') = 3493.209961 ;
vcif('c_Crops','LatinAmer','SouthAsia') = 427.5668945 ;
vcif('c_Crops','LatinAmer','NAmerica') = 17982.0293 ;
vcif('c_Crops','LatinAmer','LatinAmer') = 10542.62695 ;
vcif('c_Crops','LatinAmer','EU_28') = 25203.98633 ;
vcif('c_Crops','LatinAmer','MENA') = 9031.24707 ;
vcif('c_Crops','LatinAmer','SSA') = 1788.656128 ;
vcif('c_Crops','LatinAmer','RestofWorld') = 4141.255371 ;
vcif('c_Crops','EU_28','Oceania') = 138.5540161 ;
vcif('c_Crops','EU_28','EastAsia') = 1065.071533 ;
vcif('c_Crops','EU_28','SEAsia') = 348.8464661 ;
vcif('c_Crops','EU_28','SouthAsia') = 378.2745056 ;
vcif('c_Crops','EU_28','NAmerica') = 1676.152954 ;
vcif('c_Crops','EU_28','LatinAmer') = 727.9819336 ;
vcif('c_Crops','EU_28','EU_28') = 73994.85938 ;
vcif('c_Crops','EU_28','MENA') = 9655.681641 ;
vcif('c_Crops','EU_28','SSA') = 2308.812988 ;
vcif('c_Crops','EU_28','RestofWorld') = 8834.432617 ;
vcif('c_Crops','MENA','Oceania') = 138.974884 ;
vcif('c_Crops','MENA','EastAsia') = 709.0101929 ;
vcif('c_Crops','MENA','SEAsia') = 345.34552 ;
vcif('c_Crops','MENA','SouthAsia') = 625.3825684 ;
vcif('c_Crops','MENA','NAmerica') = 969.0093384 ;
vcif('c_Crops','MENA','LatinAmer') = 184.9235229 ;
vcif('c_Crops','MENA','EU_28') = 7824.980957 ;
vcif('c_Crops','MENA','MENA') = 7614.058105 ;
vcif('c_Crops','MENA','SSA') = 612.4978027 ;
vcif('c_Crops','MENA','RestofWorld') = 4359.412598 ;
vcif('c_Crops','SSA','Oceania') = 92.86306 ;
vcif('c_Crops','SSA','EastAsia') = 2863.279541 ;
vcif('c_Crops','SSA','SEAsia') = 2136.86499 ;
vcif('c_Crops','SSA','SouthAsia') = 2137.927734 ;
vcif('c_Crops','SSA','NAmerica') = 2594.197998 ;
vcif('c_Crops','SSA','LatinAmer') = 257.2740173 ;
vcif('c_Crops','SSA','EU_28') = 10712.54785 ;
vcif('c_Crops','SSA','MENA') = 2280.793945 ;
vcif('c_Crops','SSA','SSA') = 2585.108398 ;
vcif('c_Crops','SSA','RestofWorld') = 1221.764282 ;
vcif('c_Crops','RestofWorld','Oceania') = 10.51148605 ;
vcif('c_Crops','RestofWorld','EastAsia') = 748.6572876 ;
vcif('c_Crops','RestofWorld','SEAsia') = 67.38510895 ;
vcif('c_Crops','RestofWorld','SouthAsia') = 292.3222046 ;
vcif('c_Crops','RestofWorld','NAmerica') = 125.4989929 ;
vcif('c_Crops','RestofWorld','LatinAmer') = 101.1045609 ;
vcif('c_Crops','RestofWorld','EU_28') = 3960.059082 ;
vcif('c_Crops','RestofWorld','MENA') = 10712.82324 ;
vcif('c_Crops','RestofWorld','SSA') = 872.0300293 ;
vcif('c_Crops','RestofWorld','RestofWorld') = 3636.273926 ;
vcif('c_MeatLstk','Oceania','Oceania') = 643.8560791 ;
vcif('c_MeatLstk','Oceania','EastAsia') = 8929.682617 ;
vcif('c_MeatLstk','Oceania','SEAsia') = 1638.062866 ;
vcif('c_MeatLstk','Oceania','SouthAsia') = 328.642334 ;
vcif('c_MeatLstk','Oceania','NAmerica') = 3029.532715 ;
vcif('c_MeatLstk','Oceania','LatinAmer') = 217.3272705 ;
vcif('c_MeatLstk','Oceania','EU_28') = 2580.145264 ;
vcif('c_MeatLstk','Oceania','MENA') = 1333.854126 ;
vcif('c_MeatLstk','Oceania','SSA') = 169.2742767 ;
vcif('c_MeatLstk','Oceania','RestofWorld') = 713.3197632 ;
vcif('c_MeatLstk','EastAsia','Oceania') = 45.42178345 ;
vcif('c_MeatLstk','EastAsia','EastAsia') = 3446.282227 ;
vcif('c_MeatLstk','EastAsia','SEAsia') = 546.8447876 ;
vcif('c_MeatLstk','EastAsia','SouthAsia') = 67.48873138 ;
vcif('c_MeatLstk','EastAsia','NAmerica') = 745.3659668 ;
vcif('c_MeatLstk','EastAsia','LatinAmer') = 97.49333191 ;
vcif('c_MeatLstk','EastAsia','EU_28') = 1639.041992 ;
vcif('c_MeatLstk','EastAsia','MENA') = 219.8074951 ;
vcif('c_MeatLstk','EastAsia','SSA') = 90.51722717 ;
vcif('c_MeatLstk','EastAsia','RestofWorld') = 168.7759247 ;
vcif('c_MeatLstk','SEAsia','Oceania') = 17.26716042 ;
vcif('c_MeatLstk','SEAsia','EastAsia') = 2040.153564 ;
vcif('c_MeatLstk','SEAsia','SEAsia') = 816.8880005 ;
vcif('c_MeatLstk','SEAsia','SouthAsia') = 23.42883492 ;
vcif('c_MeatLstk','SEAsia','NAmerica') = 199.6348877 ;
vcif('c_MeatLstk','SEAsia','LatinAmer') = 9.341163635 ;
vcif('c_MeatLstk','SEAsia','EU_28') = 1192.538208 ;
vcif('c_MeatLstk','SEAsia','MENA') = 79.69589233 ;
vcif('c_MeatLstk','SEAsia','SSA') = 42.32653809 ;
vcif('c_MeatLstk','SEAsia','RestofWorld') = 32.90296173 ;
vcif('c_MeatLstk','SouthAsia','Oceania') = 6.177580833 ;
vcif('c_MeatLstk','SouthAsia','EastAsia') = 89.0045929 ;
vcif('c_MeatLstk','SouthAsia','SEAsia') = 1309.084106 ;
vcif('c_MeatLstk','SouthAsia','SouthAsia') = 81.80826569 ;
vcif('c_MeatLstk','SouthAsia','NAmerica') = 137.8781586 ;
vcif('c_MeatLstk','SouthAsia','LatinAmer') = 9.366596222 ;
vcif('c_MeatLstk','SouthAsia','EU_28') = 149.7421417 ;
vcif('c_MeatLstk','SouthAsia','MENA') = 1488.973999 ;
vcif('c_MeatLstk','SouthAsia','SSA') = 198.2512207 ;
vcif('c_MeatLstk','SouthAsia','RestofWorld') = 112.1615524 ;
vcif('c_MeatLstk','NAmerica','Oceania') = 540.0287476 ;
vcif('c_MeatLstk','NAmerica','EastAsia') = 13414.94336 ;
vcif('c_MeatLstk','NAmerica','SEAsia') = 725.4830933 ;
vcif('c_MeatLstk','NAmerica','SouthAsia') = 130.141922 ;
vcif('c_MeatLstk','NAmerica','NAmerica') = 13039.51172 ;
vcif('c_MeatLstk','NAmerica','LatinAmer') = 1893.702515 ;
vcif('c_MeatLstk','NAmerica','EU_28') = 1243.093262 ;
vcif('c_MeatLstk','NAmerica','MENA') = 1275.025513 ;
vcif('c_MeatLstk','NAmerica','SSA') = 713.0420532 ;
vcif('c_MeatLstk','NAmerica','RestofWorld') = 2091.461426 ;
vcif('c_MeatLstk','LatinAmer','Oceania') = 45.51876831 ;
vcif('c_MeatLstk','LatinAmer','EastAsia') = 3961.010498 ;
vcif('c_MeatLstk','LatinAmer','SEAsia') = 544.2368774 ;
vcif('c_MeatLstk','LatinAmer','SouthAsia') = 63.49185181 ;
vcif('c_MeatLstk','LatinAmer','NAmerica') = 1412.411377 ;
vcif('c_MeatLstk','LatinAmer','LatinAmer') = 4538.038574 ;
vcif('c_MeatLstk','LatinAmer','EU_28') = 4535.825684 ;
vcif('c_MeatLstk','LatinAmer','MENA') = 5597.087402 ;
vcif('c_MeatLstk','LatinAmer','SSA') = 1181.072021 ;
vcif('c_MeatLstk','LatinAmer','RestofWorld') = 3195.658203 ;
vcif('c_MeatLstk','EU_28','Oceania') = 414.8974304 ;
vcif('c_MeatLstk','EU_28','EastAsia') = 5698.104004 ;
vcif('c_MeatLstk','EU_28','SEAsia') = 611.5287476 ;
vcif('c_MeatLstk','EU_28','SouthAsia') = 160.4988861 ;
vcif('c_MeatLstk','EU_28','NAmerica') = 1178.333496 ;
vcif('c_MeatLstk','EU_28','LatinAmer') = 412.1517334 ;
vcif('c_MeatLstk','EU_28','EU_28') = 66102.29688 ;
vcif('c_MeatLstk','EU_28','MENA') = 3308.861084 ;
vcif('c_MeatLstk','EU_28','SSA') = 1181.311768 ;
vcif('c_MeatLstk','EU_28','RestofWorld') = 5864.121582 ;
vcif('c_MeatLstk','MENA','Oceania') = 13.45483685 ;
vcif('c_MeatLstk','MENA','EastAsia') = 146.0844421 ;
vcif('c_MeatLstk','MENA','SEAsia') = 63.54631042 ;
vcif('c_MeatLstk','MENA','SouthAsia') = 88.33241272 ;
vcif('c_MeatLstk','MENA','NAmerica') = 109.7469635 ;
vcif('c_MeatLstk','MENA','LatinAmer') = 20.40985298 ;
vcif('c_MeatLstk','MENA','EU_28') = 772.5230713 ;
vcif('c_MeatLstk','MENA','MENA') = 1806.9375 ;
vcif('c_MeatLstk','MENA','SSA') = 88.2539444 ;
vcif('c_MeatLstk','MENA','RestofWorld') = 142.0113678 ;
vcif('c_MeatLstk','SSA','Oceania') = 9.650064468 ;
vcif('c_MeatLstk','SSA','EastAsia') = 385.2297058 ;
vcif('c_MeatLstk','SSA','SEAsia') = 35.010746 ;
vcif('c_MeatLstk','SSA','SouthAsia') = 108.0082321 ;
vcif('c_MeatLstk','SSA','NAmerica') = 92.90730286 ;
vcif('c_MeatLstk','SSA','LatinAmer') = 13.77873898 ;
vcif('c_MeatLstk','SSA','EU_28') = 446.3058777 ;
vcif('c_MeatLstk','SSA','MENA') = 845.1865234 ;
vcif('c_MeatLstk','SSA','SSA') = 583.5997925 ;
vcif('c_MeatLstk','SSA','RestofWorld') = 82.41765594 ;
vcif('c_MeatLstk','RestofWorld','Oceania') = 19.42339134 ;
vcif('c_MeatLstk','RestofWorld','EastAsia') = 205.7259827 ;
vcif('c_MeatLstk','RestofWorld','SEAsia') = 42.89149094 ;
vcif('c_MeatLstk','RestofWorld','SouthAsia') = 14.64772701 ;
vcif('c_MeatLstk','RestofWorld','NAmerica') = 132.9904175 ;
vcif('c_MeatLstk','RestofWorld','LatinAmer') = 19.25655937 ;
vcif('c_MeatLstk','RestofWorld','EU_28') = 960.3049316 ;
vcif('c_MeatLstk','RestofWorld','MENA') = 193.3408203 ;
vcif('c_MeatLstk','RestofWorld','SSA') = 24.1670723 ;
vcif('c_MeatLstk','RestofWorld','RestofWorld') = 546.7502441 ;
vcif('c_Extraction','Oceania','Oceania') = 1289.63147 ;
vcif('c_Extraction','Oceania','EastAsia') = 134819.5156 ;
vcif('c_Extraction','Oceania','SEAsia') = 4654.527344 ;
vcif('c_Extraction','Oceania','SouthAsia') = 14953.07812 ;
vcif('c_Extraction','Oceania','NAmerica') = 2144.832031 ;
vcif('c_Extraction','Oceania','LatinAmer') = 1236.06958 ;
vcif('c_Extraction','Oceania','EU_28') = 5686.36084 ;
vcif('c_Extraction','Oceania','MENA') = 937.3173828 ;
vcif('c_Extraction','Oceania','SSA') = 360.7489624 ;
vcif('c_Extraction','Oceania','RestofWorld') = 219.9904633 ;
vcif('c_Extraction','EastAsia','Oceania') = 1934.624512 ;
vcif('c_Extraction','EastAsia','EastAsia') = 21045.08789 ;
vcif('c_Extraction','EastAsia','SEAsia') = 2026.282104 ;
vcif('c_Extraction','EastAsia','SouthAsia') = 1858.204956 ;
vcif('c_Extraction','EastAsia','NAmerica') = 1185.054199 ;
vcif('c_Extraction','EastAsia','LatinAmer') = 118.3680038 ;
vcif('c_Extraction','EastAsia','EU_28') = 1504.211304 ;
vcif('c_Extraction','EastAsia','MENA') = 252.7928925 ;
vcif('c_Extraction','EastAsia','SSA') = 104.3816147 ;
vcif('c_Extraction','EastAsia','RestofWorld') = 252.0222168 ;
vcif('c_Extraction','SEAsia','Oceania') = 7527.130371 ;
vcif('c_Extraction','SEAsia','EastAsia') = 76035.02344 ;
vcif('c_Extraction','SEAsia','SEAsia') = 16411.43555 ;
vcif('c_Extraction','SEAsia','SouthAsia') = 10014.45605 ;
vcif('c_Extraction','SEAsia','NAmerica') = 1940.893799 ;
vcif('c_Extraction','SEAsia','LatinAmer') = 159.7043304 ;
vcif('c_Extraction','SEAsia','EU_28') = 2905.768311 ;
vcif('c_Extraction','SEAsia','MENA') = 782.0026855 ;
vcif('c_Extraction','SEAsia','SSA') = 59.34599304 ;
vcif('c_Extraction','SEAsia','RestofWorld') = 319.4360046 ;
vcif('c_Extraction','SouthAsia','Oceania') = 42.38668442 ;
vcif('c_Extraction','SouthAsia','EastAsia') = 13189.43359 ;
vcif('c_Extraction','SouthAsia','SEAsia') = 356.3277283 ;
vcif('c_Extraction','SouthAsia','SouthAsia') = 681.602478 ;
vcif('c_Extraction','SouthAsia','NAmerica') = 269.5941162 ;
vcif('c_Extraction','SouthAsia','LatinAmer') = 48.11912918 ;
vcif('c_Extraction','SouthAsia','EU_28') = 1499.40625 ;
vcif('c_Extraction','SouthAsia','MENA') = 762.8228149 ;
vcif('c_Extraction','SouthAsia','SSA') = 65.50414276 ;
vcif('c_Extraction','SouthAsia','RestofWorld') = 66.55597687 ;
vcif('c_Extraction','NAmerica','Oceania') = 638.1994019 ;
vcif('c_Extraction','NAmerica','EastAsia') = 27864.3125 ;
vcif('c_Extraction','NAmerica','SEAsia') = 740.1036987 ;
vcif('c_Extraction','NAmerica','SouthAsia') = 2502.078369 ;
vcif('c_Extraction','NAmerica','NAmerica') = 104894.2266 ;
vcif('c_Extraction','NAmerica','LatinAmer') = 4448.657227 ;
vcif('c_Extraction','NAmerica','EU_28') = 18856.07812 ;
vcif('c_Extraction','NAmerica','MENA') = 3507.842773 ;
vcif('c_Extraction','NAmerica','SSA') = 321.3175049 ;
vcif('c_Extraction','NAmerica','RestofWorld') = 1855.741821 ;
vcif('c_Extraction','LatinAmer','Oceania') = 398.6952515 ;
vcif('c_Extraction','LatinAmer','EastAsia') = 92150.67969 ;
vcif('c_Extraction','LatinAmer','SEAsia') = 2765.040039 ;
vcif('c_Extraction','LatinAmer','SouthAsia') = 12428.90039 ;
vcif('c_Extraction','LatinAmer','NAmerica') = 61228.375 ;
vcif('c_Extraction','LatinAmer','LatinAmer') = 26175.45508 ;
vcif('c_Extraction','LatinAmer','EU_28') = 36286.58203 ;
vcif('c_Extraction','LatinAmer','MENA') = 5846.509277 ;
vcif('c_Extraction','LatinAmer','SSA') = 502.9578247 ;
vcif('c_Extraction','LatinAmer','RestofWorld') = 6263.121582 ;
vcif('c_Extraction','EU_28','Oceania') = 144.820816 ;
vcif('c_Extraction','EU_28','EastAsia') = 4146.079102 ;
vcif('c_Extraction','EU_28','SEAsia') = 560.3181763 ;
vcif('c_Extraction','EU_28','SouthAsia') = 9484.038086 ;
vcif('c_Extraction','EU_28','NAmerica') = 2818.374512 ;
vcif('c_Extraction','EU_28','LatinAmer') = 910.4675903 ;
vcif('c_Extraction','EU_28','EU_28') = 60957.69531 ;
vcif('c_Extraction','EU_28','MENA') = 5943.973633 ;
vcif('c_Extraction','EU_28','SSA') = 2002.535767 ;
vcif('c_Extraction','EU_28','RestofWorld') = 3039.693359 ;
vcif('c_Extraction','MENA','Oceania') = 3167.398438 ;
vcif('c_Extraction','MENA','EastAsia') = 356380.3438 ;
vcif('c_Extraction','MENA','SEAsia') = 75045.25781 ;
vcif('c_Extraction','MENA','SouthAsia') = 115452.4219 ;
vcif('c_Extraction','MENA','NAmerica') = 108897.3516 ;
vcif('c_Extraction','MENA','LatinAmer') = 5963.664551 ;
vcif('c_Extraction','MENA','EU_28') = 149870.3281 ;
vcif('c_Extraction','MENA','MENA') = 39348.08984 ;
vcif('c_Extraction','MENA','SSA') = 13674.00879 ;
vcif('c_Extraction','MENA','RestofWorld') = 2456.5979 ;
vcif('c_Extraction','SSA','Oceania') = 2299.581299 ;
vcif('c_Extraction','SSA','EastAsia') = 74435.58594 ;
vcif('c_Extraction','SSA','SEAsia') = 3157.956787 ;
vcif('c_Extraction','SSA','SouthAsia') = 24265.63672 ;
vcif('c_Extraction','SSA','NAmerica') = 79701.49219 ;
vcif('c_Extraction','SSA','LatinAmer') = 10674.53906 ;
vcif('c_Extraction','SSA','EU_28') = 52546.57422 ;
vcif('c_Extraction','SSA','MENA') = 1603.660889 ;
vcif('c_Extraction','SSA','SSA') = 11028.80762 ;
vcif('c_Extraction','SSA','RestofWorld') = 2054.689697 ;
vcif('c_Extraction','RestofWorld','Oceania') = 1666.657959 ;
vcif('c_Extraction','RestofWorld','EastAsia') = 87310.85938 ;
vcif('c_Extraction','RestofWorld','SEAsia') = 6150.834961 ;
vcif('c_Extraction','RestofWorld','SouthAsia') = 2161.845703 ;
vcif('c_Extraction','RestofWorld','NAmerica') = 21736.47656 ;
vcif('c_Extraction','RestofWorld','LatinAmer') = 2817.719727 ;
vcif('c_Extraction','RestofWorld','EU_28') = 332753.3438 ;
vcif('c_Extraction','RestofWorld','MENA') = 19151.8125 ;
vcif('c_Extraction','RestofWorld','SSA') = 252.3052826 ;
vcif('c_Extraction','RestofWorld','RestofWorld') = 48457.78125 ;
vcif('c_ProcFood','Oceania','Oceania') = 3918.286377 ;
vcif('c_ProcFood','Oceania','EastAsia') = 6977.964844 ;
vcif('c_ProcFood','Oceania','SEAsia') = 4914.108887 ;
vcif('c_ProcFood','Oceania','SouthAsia') = 729.3468018 ;
vcif('c_ProcFood','Oceania','NAmerica') = 3048.230957 ;
vcif('c_ProcFood','Oceania','LatinAmer') = 854.6535034 ;
vcif('c_ProcFood','Oceania','EU_28') = 2846.450928 ;
vcif('c_ProcFood','Oceania','MENA') = 2447.240723 ;
vcif('c_ProcFood','Oceania','SSA') = 710.3203735 ;
vcif('c_ProcFood','Oceania','RestofWorld') = 399.5119934 ;
vcif('c_ProcFood','EastAsia','Oceania') = 1636.681396 ;
vcif('c_ProcFood','EastAsia','EastAsia') = 18697.14258 ;
vcif('c_ProcFood','EastAsia','SEAsia') = 7565.151367 ;
vcif('c_ProcFood','EastAsia','SouthAsia') = 629.8579102 ;
vcif('c_ProcFood','EastAsia','NAmerica') = 9545.265625 ;
vcif('c_ProcFood','EastAsia','LatinAmer') = 1202.185303 ;
vcif('c_ProcFood','EastAsia','EU_28') = 6446.05127 ;
vcif('c_ProcFood','EastAsia','MENA') = 2108.605713 ;
vcif('c_ProcFood','EastAsia','SSA') = 1805.25293 ;
vcif('c_ProcFood','EastAsia','RestofWorld') = 2162.932861 ;
vcif('c_ProcFood','SEAsia','Oceania') = 2715.075195 ;
vcif('c_ProcFood','SEAsia','EastAsia') = 25104.77734 ;
vcif('c_ProcFood','SEAsia','SEAsia') = 17235.64453 ;
vcif('c_ProcFood','SEAsia','SouthAsia') = 12770.38672 ;
vcif('c_ProcFood','SEAsia','NAmerica') = 13145.6123 ;
vcif('c_ProcFood','SEAsia','LatinAmer') = 1465.777588 ;
vcif('c_ProcFood','SEAsia','EU_28') = 9288.09668 ;
vcif('c_ProcFood','SEAsia','MENA') = 8011.463379 ;
vcif('c_ProcFood','SEAsia','SSA') = 4843.203613 ;
vcif('c_ProcFood','SEAsia','RestofWorld') = 2746.355957 ;
vcif('c_ProcFood','SouthAsia','Oceania') = 201.5151672 ;
vcif('c_ProcFood','SouthAsia','EastAsia') = 3020.908203 ;
vcif('c_ProcFood','SouthAsia','SEAsia') = 2540.057861 ;
vcif('c_ProcFood','SouthAsia','SouthAsia') = 2215.257324 ;
vcif('c_ProcFood','SouthAsia','NAmerica') = 2822.995605 ;
vcif('c_ProcFood','SouthAsia','LatinAmer') = 155.9632568 ;
vcif('c_ProcFood','SouthAsia','EU_28') = 2948.40625 ;
vcif('c_ProcFood','SouthAsia','MENA') = 2571.209961 ;
vcif('c_ProcFood','SouthAsia','SSA') = 1210.27478 ;
vcif('c_ProcFood','SouthAsia','RestofWorld') = 605.1375122 ;
vcif('c_ProcFood','NAmerica','Oceania') = 1671.510864 ;
vcif('c_ProcFood','NAmerica','EastAsia') = 15704.62793 ;
vcif('c_ProcFood','NAmerica','SEAsia') = 4609.520996 ;
vcif('c_ProcFood','NAmerica','SouthAsia') = 484.4703674 ;
vcif('c_ProcFood','NAmerica','NAmerica') = 44554.38281 ;
vcif('c_ProcFood','NAmerica','LatinAmer') = 8837.00293 ;
vcif('c_ProcFood','NAmerica','EU_28') = 8708.491211 ;
vcif('c_ProcFood','NAmerica','MENA') = 4245.294922 ;
vcif('c_ProcFood','NAmerica','SSA') = 1071.246704 ;
vcif('c_ProcFood','NAmerica','RestofWorld') = 1468.320679 ;
vcif('c_ProcFood','LatinAmer','Oceania') = 707.9806519 ;
vcif('c_ProcFood','LatinAmer','EastAsia') = 11210.71094 ;
vcif('c_ProcFood','LatinAmer','SEAsia') = 6538.949707 ;
vcif('c_ProcFood','LatinAmer','SouthAsia') = 2935.779053 ;
vcif('c_ProcFood','LatinAmer','NAmerica') = 14055.3291 ;
vcif('c_ProcFood','LatinAmer','LatinAmer') = 21077.91797 ;
vcif('c_ProcFood','LatinAmer','EU_28') = 18726.48633 ;
vcif('c_ProcFood','LatinAmer','MENA') = 11707.5 ;
vcif('c_ProcFood','LatinAmer','SSA') = 8702.827148 ;
vcif('c_ProcFood','LatinAmer','RestofWorld') = 3305.37085 ;
vcif('c_ProcFood','EU_28','Oceania') = 3299.022705 ;
vcif('c_ProcFood','EU_28','EastAsia') = 13606.53613 ;
vcif('c_ProcFood','EU_28','SEAsia') = 4667.397461 ;
vcif('c_ProcFood','EU_28','SouthAsia') = 1091.015503 ;
vcif('c_ProcFood','EU_28','NAmerica') = 25141.69727 ;
vcif('c_ProcFood','EU_28','LatinAmer') = 5143.818848 ;
vcif('c_ProcFood','EU_28','EU_28') = 233193.8438 ;
vcif('c_ProcFood','EU_28','MENA') = 17531.04492 ;
vcif('c_ProcFood','EU_28','SSA') = 8360.849609 ;
vcif('c_ProcFood','EU_28','RestofWorld') = 23966.32617 ;
vcif('c_ProcFood','MENA','Oceania') = 163.1560822 ;
vcif('c_ProcFood','MENA','EastAsia') = 1036.720825 ;
vcif('c_ProcFood','MENA','SEAsia') = 665.8327637 ;
vcif('c_ProcFood','MENA','SouthAsia') = 673.6491699 ;
vcif('c_ProcFood','MENA','NAmerica') = 1330.914551 ;
vcif('c_ProcFood','MENA','LatinAmer') = 165.2115784 ;
vcif('c_ProcFood','MENA','EU_28') = 5922.946777 ;
vcif('c_ProcFood','MENA','MENA') = 14279.13867 ;
vcif('c_ProcFood','MENA','SSA') = 2664.400879 ;
vcif('c_ProcFood','MENA','RestofWorld') = 1041.990112 ;
vcif('c_ProcFood','SSA','Oceania') = 207.4598541 ;
vcif('c_ProcFood','SSA','EastAsia') = 983.0687256 ;
vcif('c_ProcFood','SSA','SEAsia') = 254.5271301 ;
vcif('c_ProcFood','SSA','SouthAsia') = 70.99890137 ;
vcif('c_ProcFood','SSA','NAmerica') = 1053.123779 ;
vcif('c_ProcFood','SSA','LatinAmer') = 101.462677 ;
vcif('c_ProcFood','SSA','EU_28') = 6766.197266 ;
vcif('c_ProcFood','SSA','MENA') = 592.8356323 ;
vcif('c_ProcFood','SSA','SSA') = 5547.823242 ;
vcif('c_ProcFood','SSA','RestofWorld') = 429.2106323 ;
vcif('c_ProcFood','RestofWorld','Oceania') = 347.51297 ;
vcif('c_ProcFood','RestofWorld','EastAsia') = 5816.319336 ;
vcif('c_ProcFood','RestofWorld','SEAsia') = 487.7947693 ;
vcif('c_ProcFood','RestofWorld','SouthAsia') = 1103.602539 ;
vcif('c_ProcFood','RestofWorld','NAmerica') = 2471.063477 ;
vcif('c_ProcFood','RestofWorld','LatinAmer') = 594.7888184 ;
vcif('c_ProcFood','RestofWorld','EU_28') = 13601.48047 ;
vcif('c_ProcFood','RestofWorld','MENA') = 4705.338379 ;
vcif('c_ProcFood','RestofWorld','SSA') = 991.026062 ;
vcif('c_ProcFood','RestofWorld','RestofWorld') = 10438.66113 ;
vcif('c_TextWapp','Oceania','Oceania') = 659.8927002 ;
vcif('c_TextWapp','Oceania','EastAsia') = 495.4488525 ;
vcif('c_TextWapp','Oceania','SEAsia') = 164.3904877 ;
vcif('c_TextWapp','Oceania','SouthAsia') = 152.0117493 ;
vcif('c_TextWapp','Oceania','NAmerica') = 161.2231445 ;
vcif('c_TextWapp','Oceania','LatinAmer') = 14.28889465 ;
vcif('c_TextWapp','Oceania','EU_28') = 337.7866821 ;
vcif('c_TextWapp','Oceania','MENA') = 70.3768158 ;
vcif('c_TextWapp','Oceania','SSA') = 35.97338486 ;
vcif('c_TextWapp','Oceania','RestofWorld') = 17.99432182 ;
vcif('c_TextWapp','EastAsia','Oceania') = 6422.07666 ;
vcif('c_TextWapp','EastAsia','EastAsia') = 73834.82031 ;
vcif('c_TextWapp','EastAsia','SEAsia') = 30448.12695 ;
vcif('c_TextWapp','EastAsia','SouthAsia') = 12812.36523 ;
vcif('c_TextWapp','EastAsia','NAmerica') = 62649.45312 ;
vcif('c_TextWapp','EastAsia','LatinAmer') = 16674.08008 ;
vcif('c_TextWapp','EastAsia','EU_28') = 68128.42969 ;
vcif('c_TextWapp','EastAsia','MENA') = 24799.92188 ;
vcif('c_TextWapp','EastAsia','SSA') = 10347.91699 ;
vcif('c_TextWapp','EastAsia','RestofWorld') = 18412.49609 ;
vcif('c_TextWapp','SEAsia','Oceania') = 624.0349731 ;
vcif('c_TextWapp','SEAsia','EastAsia') = 11134.60254 ;
vcif('c_TextWapp','SEAsia','SEAsia') = 4508.45459 ;
vcif('c_TextWapp','SEAsia','SouthAsia') = 1598.042236 ;
vcif('c_TextWapp','SEAsia','NAmerica') = 22838.89258 ;
vcif('c_TextWapp','SEAsia','LatinAmer') = 1510.940186 ;
vcif('c_TextWapp','SEAsia','EU_28') = 11245.31836 ;
vcif('c_TextWapp','SEAsia','MENA') = 3664.766357 ;
vcif('c_TextWapp','SEAsia','SSA') = 614.2822266 ;
vcif('c_TextWapp','SEAsia','RestofWorld') = 834.0689087 ;
vcif('c_TextWapp','SouthAsia','Oceania') = 946.8759766 ;
vcif('c_TextWapp','SouthAsia','EastAsia') = 4891.666504 ;
vcif('c_TextWapp','SouthAsia','SEAsia') = 1300.475098 ;
vcif('c_TextWapp','SouthAsia','SouthAsia') = 2987.550781 ;
vcif('c_TextWapp','SouthAsia','NAmerica') = 20512.45508 ;
vcif('c_TextWapp','SouthAsia','LatinAmer') = 2444.332275 ;
vcif('c_TextWapp','SouthAsia','EU_28') = 34353.94141 ;
vcif('c_TextWapp','SouthAsia','MENA') = 8250.149414 ;
vcif('c_TextWapp','SouthAsia','SSA') = 2174.624023 ;
vcif('c_TextWapp','SouthAsia','RestofWorld') = 1871.553223 ;
vcif('c_TextWapp','NAmerica','Oceania') = 412.3249512 ;
vcif('c_TextWapp','NAmerica','EastAsia') = 3122.306152 ;
vcif('c_TextWapp','NAmerica','SEAsia') = 615.1200562 ;
vcif('c_TextWapp','NAmerica','SouthAsia') = 322.4178162 ;
vcif('c_TextWapp','NAmerica','NAmerica') = 16144.91016 ;
vcif('c_TextWapp','NAmerica','LatinAmer') = 5858.726562 ;
vcif('c_TextWapp','NAmerica','EU_28') = 2971.407227 ;
vcif('c_TextWapp','NAmerica','MENA') = 912.4609375 ;
vcif('c_TextWapp','NAmerica','SSA') = 265.3851929 ;
vcif('c_TextWapp','NAmerica','RestofWorld') = 449.2341003 ;
vcif('c_TextWapp','LatinAmer','Oceania') = 74.38791656 ;
vcif('c_TextWapp','LatinAmer','EastAsia') = 474.6665649 ;
vcif('c_TextWapp','LatinAmer','SEAsia') = 105.0871277 ;
vcif('c_TextWapp','LatinAmer','SouthAsia') = 58.10726166 ;
vcif('c_TextWapp','LatinAmer','NAmerica') = 12009.34375 ;
vcif('c_TextWapp','LatinAmer','LatinAmer') = 5698.59668 ;
vcif('c_TextWapp','LatinAmer','EU_28') = 1395.785156 ;
vcif('c_TextWapp','LatinAmer','MENA') = 214.0811462 ;
vcif('c_TextWapp','LatinAmer','SSA') = 118.7388306 ;
vcif('c_TextWapp','LatinAmer','RestofWorld') = 106.0248947 ;
vcif('c_TextWapp','EU_28','Oceania') = 862.1547241 ;
vcif('c_TextWapp','EU_28','EastAsia') = 8625.594727 ;
vcif('c_TextWapp','EU_28','SEAsia') = 1563.547852 ;
vcif('c_TextWapp','EU_28','SouthAsia') = 1047.968506 ;
vcif('c_TextWapp','EU_28','NAmerica') = 8333.290039 ;
vcif('c_TextWapp','EU_28','LatinAmer') = 1853.106201 ;
vcif('c_TextWapp','EU_28','EU_28') = 116778.4219 ;
vcif('c_TextWapp','EU_28','MENA') = 12055.80078 ;
vcif('c_TextWapp','EU_28','SSA') = 1582.605591 ;
vcif('c_TextWapp','EU_28','RestofWorld') = 16899.42969 ;
vcif('c_TextWapp','MENA','Oceania') = 190.3780212 ;
vcif('c_TextWapp','MENA','EastAsia') = 1125.465332 ;
vcif('c_TextWapp','MENA','SEAsia') = 279.5075989 ;
vcif('c_TextWapp','MENA','SouthAsia') = 411.9977112 ;
vcif('c_TextWapp','MENA','NAmerica') = 5841.887695 ;
vcif('c_TextWapp','MENA','LatinAmer') = 453.6681519 ;
vcif('c_TextWapp','MENA','EU_28') = 29807.97656 ;
vcif('c_TextWapp','MENA','MENA') = 6527.197754 ;
vcif('c_TextWapp','MENA','SSA') = 704.9630127 ;
vcif('c_TextWapp','MENA','RestofWorld') = 3657.498291 ;
vcif('c_TextWapp','SSA','Oceania') = 56.71712875 ;
vcif('c_TextWapp','SSA','EastAsia') = 242.9412842 ;
vcif('c_TextWapp','SSA','SEAsia') = 75.70869446 ;
vcif('c_TextWapp','SSA','SouthAsia') = 41.65351868 ;
vcif('c_TextWapp','SSA','NAmerica') = 1285.183472 ;
vcif('c_TextWapp','SSA','LatinAmer') = 53.24167252 ;
vcif('c_TextWapp','SSA','EU_28') = 1504.157349 ;
vcif('c_TextWapp','SSA','MENA') = 132.8082581 ;
vcif('c_TextWapp','SSA','SSA') = 1101.062744 ;
vcif('c_TextWapp','SSA','RestofWorld') = 77.99245453 ;
vcif('c_TextWapp','RestofWorld','Oceania') = 30.08081055 ;
vcif('c_TextWapp','RestofWorld','EastAsia') = 464.2014771 ;
vcif('c_TextWapp','RestofWorld','SEAsia') = 93.34260559 ;
vcif('c_TextWapp','RestofWorld','SouthAsia') = 81.51605988 ;
vcif('c_TextWapp','RestofWorld','NAmerica') = 411.9551392 ;
vcif('c_TextWapp','RestofWorld','LatinAmer') = 51.33949661 ;
vcif('c_TextWapp','RestofWorld','EU_28') = 5440.659668 ;
vcif('c_TextWapp','RestofWorld','MENA') = 805.9138794 ;
vcif('c_TextWapp','RestofWorld','SSA') = 66.03011322 ;
vcif('c_TextWapp','RestofWorld','RestofWorld') = 2036.305664 ;
vcif('c_LightMnfc','Oceania','Oceania') = 3922.047119 ;
vcif('c_LightMnfc','Oceania','EastAsia') = 4285.519531 ;
vcif('c_LightMnfc','Oceania','SEAsia') = 1833.453979 ;
vcif('c_LightMnfc','Oceania','SouthAsia') = 310.7367859 ;
vcif('c_LightMnfc','Oceania','NAmerica') = 2642.416992 ;
vcif('c_LightMnfc','Oceania','LatinAmer') = 299.0666504 ;
vcif('c_LightMnfc','Oceania','EU_28') = 2085.631348 ;
vcif('c_LightMnfc','Oceania','MENA') = 1539.829956 ;
vcif('c_LightMnfc','Oceania','SSA') = 526.4088745 ;
vcif('c_LightMnfc','Oceania','RestofWorld') = 460.381073 ;
vcif('c_LightMnfc','EastAsia','Oceania') = 37094.67188 ;
vcif('c_LightMnfc','EastAsia','EastAsia') = 116305.2812 ;
vcif('c_LightMnfc','EastAsia','SEAsia') = 57430.53906 ;
vcif('c_LightMnfc','EastAsia','SouthAsia') = 19475.08203 ;
vcif('c_LightMnfc','EastAsia','NAmerica') = 229782.8438 ;
vcif('c_LightMnfc','EastAsia','LatinAmer') = 49639.73438 ;
vcif('c_LightMnfc','EastAsia','EU_28') = 159284.1875 ;
vcif('c_LightMnfc','EastAsia','MENA') = 61587.55469 ;
vcif('c_LightMnfc','EastAsia','SSA') = 35679.9375 ;
vcif('c_LightMnfc','EastAsia','RestofWorld') = 47792.32422 ;
vcif('c_LightMnfc','SEAsia','Oceania') = 8687.294922 ;
vcif('c_LightMnfc','SEAsia','EastAsia') = 27305.7793 ;
vcif('c_LightMnfc','SEAsia','SEAsia') = 25998.05273 ;
vcif('c_LightMnfc','SEAsia','SouthAsia') = 4842.363281 ;
vcif('c_LightMnfc','SEAsia','NAmerica') = 19608.25586 ;
vcif('c_LightMnfc','SEAsia','LatinAmer') = 4646.523438 ;
vcif('c_LightMnfc','SEAsia','EU_28') = 23636.55469 ;
vcif('c_LightMnfc','SEAsia','MENA') = 10547.91504 ;
vcif('c_LightMnfc','SEAsia','SSA') = 2612.510986 ;
vcif('c_LightMnfc','SEAsia','RestofWorld') = 3528.503418 ;
vcif('c_LightMnfc','SouthAsia','Oceania') = 887.7250366 ;
vcif('c_LightMnfc','SouthAsia','EastAsia') = 3834.024414 ;
vcif('c_LightMnfc','SouthAsia','SEAsia') = 3256.620117 ;
vcif('c_LightMnfc','SouthAsia','SouthAsia') = 2363.501953 ;
vcif('c_LightMnfc','SouthAsia','NAmerica') = 13352.45898 ;
vcif('c_LightMnfc','SouthAsia','LatinAmer') = 1576.525635 ;
vcif('c_LightMnfc','SouthAsia','EU_28') = 15241.88086 ;
vcif('c_LightMnfc','SouthAsia','MENA') = 25228.13477 ;
vcif('c_LightMnfc','SouthAsia','SSA') = 3469.448242 ;
vcif('c_LightMnfc','SouthAsia','RestofWorld') = 1209.431641 ;
vcif('c_LightMnfc','NAmerica','Oceania') = 8582.55957 ;
vcif('c_LightMnfc','NAmerica','EastAsia') = 61932.69922 ;
vcif('c_LightMnfc','NAmerica','SEAsia') = 16665.24219 ;
vcif('c_LightMnfc','NAmerica','SouthAsia') = 7642.714355 ;
vcif('c_LightMnfc','NAmerica','NAmerica') = 286820.8125 ;
vcif('c_LightMnfc','NAmerica','LatinAmer') = 33996.74609 ;
vcif('c_LightMnfc','NAmerica','EU_28') = 81795.27344 ;
vcif('c_LightMnfc','NAmerica','MENA') = 27117.39258 ;
vcif('c_LightMnfc','NAmerica','SSA') = 8363.920898 ;
vcif('c_LightMnfc','NAmerica','RestofWorld') = 10124.22363 ;
vcif('c_LightMnfc','LatinAmer','Oceania') = 727.8182373 ;
vcif('c_LightMnfc','LatinAmer','EastAsia') = 9292.266602 ;
vcif('c_LightMnfc','LatinAmer','SEAsia') = 1374.150024 ;
vcif('c_LightMnfc','LatinAmer','SouthAsia') = 393.3861694 ;
vcif('c_LightMnfc','LatinAmer','NAmerica') = 13472.08301 ;
vcif('c_LightMnfc','LatinAmer','LatinAmer') = 35936.71094 ;
vcif('c_LightMnfc','LatinAmer','EU_28') = 14048.45312 ;
vcif('c_LightMnfc','LatinAmer','MENA') = 1892.694214 ;
vcif('c_LightMnfc','LatinAmer','SSA') = 2470.830078 ;
vcif('c_LightMnfc','LatinAmer','RestofWorld') = 1394.386108 ;
vcif('c_LightMnfc','EU_28','Oceania') = 17016.43945 ;
vcif('c_LightMnfc','EU_28','EastAsia') = 101507.2266 ;
vcif('c_LightMnfc','EU_28','SEAsia') = 21602.5293 ;
vcif('c_LightMnfc','EU_28','SouthAsia') = 11439.93164 ;
vcif('c_LightMnfc','EU_28','NAmerica') = 119448.6406 ;
vcif('c_LightMnfc','EU_28','LatinAmer') = 30043.53906 ;
vcif('c_LightMnfc','EU_28','EU_28') = 898427.1875 ;
vcif('c_LightMnfc','EU_28','MENA') = 87663.89062 ;
vcif('c_LightMnfc','EU_28','SSA') = 23761.33789 ;
vcif('c_LightMnfc','EU_28','RestofWorld') = 124850.3828 ;
vcif('c_LightMnfc','MENA','Oceania') = 521.6027832 ;
vcif('c_LightMnfc','MENA','EastAsia') = 2632.579102 ;
vcif('c_LightMnfc','MENA','SEAsia') = 871.1636963 ;
vcif('c_LightMnfc','MENA','SouthAsia') = 5377.705078 ;
vcif('c_LightMnfc','MENA','NAmerica') = 13785.19238 ;
vcif('c_LightMnfc','MENA','LatinAmer') = 1070.814209 ;
vcif('c_LightMnfc','MENA','EU_28') = 29886.87891 ;
vcif('c_LightMnfc','MENA','MENA') = 18298.92188 ;
vcif('c_LightMnfc','MENA','SSA') = 3340.433838 ;
vcif('c_LightMnfc','MENA','RestofWorld') = 7075.41748 ;
vcif('c_LightMnfc','SSA','Oceania') = 399.6300354 ;
vcif('c_LightMnfc','SSA','EastAsia') = 2319.059326 ;
vcif('c_LightMnfc','SSA','SEAsia') = 883.4864502 ;
vcif('c_LightMnfc','SSA','SouthAsia') = 334.430542 ;
vcif('c_LightMnfc','SSA','NAmerica') = 4713.425293 ;
vcif('c_LightMnfc','SSA','LatinAmer') = 361.8025818 ;
vcif('c_LightMnfc','SSA','EU_28') = 7798.375488 ;
vcif('c_LightMnfc','SSA','MENA') = 879.0528564 ;
vcif('c_LightMnfc','SSA','SSA') = 7621.883789 ;
vcif('c_LightMnfc','SSA','RestofWorld') = 409.241333 ;
vcif('c_LightMnfc','RestofWorld','Oceania') = 382.1737061 ;
vcif('c_LightMnfc','RestofWorld','EastAsia') = 6030.826172 ;
vcif('c_LightMnfc','RestofWorld','SEAsia') = 1364.99231 ;
vcif('c_LightMnfc','RestofWorld','SouthAsia') = 2340.118652 ;
vcif('c_LightMnfc','RestofWorld','NAmerica') = 5714.743164 ;
vcif('c_LightMnfc','RestofWorld','LatinAmer') = 1441.635864 ;
vcif('c_LightMnfc','RestofWorld','EU_28') = 33692.52734 ;
vcif('c_LightMnfc','RestofWorld','MENA') = 6067.353516 ;
vcif('c_LightMnfc','RestofWorld','SSA') = 706.7026367 ;
vcif('c_LightMnfc','RestofWorld','RestofWorld') = 17939.42773 ;
vcif('c_HeavyMnfc','Oceania','Oceania') = 11389.55762 ;
vcif('c_HeavyMnfc','Oceania','EastAsia') = 21563.17188 ;
vcif('c_HeavyMnfc','Oceania','SEAsia') = 11862.80273 ;
vcif('c_HeavyMnfc','Oceania','SouthAsia') = 4836.051758 ;
vcif('c_HeavyMnfc','Oceania','NAmerica') = 7135.908691 ;
vcif('c_HeavyMnfc','Oceania','LatinAmer') = 1167.259399 ;
vcif('c_HeavyMnfc','Oceania','EU_28') = 6472.588379 ;
vcif('c_HeavyMnfc','Oceania','MENA') = 1668.957642 ;
vcif('c_HeavyMnfc','Oceania','SSA') = 1839.208618 ;
vcif('c_HeavyMnfc','Oceania','RestofWorld') = 985.3304443 ;
vcif('c_HeavyMnfc','EastAsia','Oceania') = 49565.97266 ;
vcif('c_HeavyMnfc','EastAsia','EastAsia') = 849651.9375 ;
vcif('c_HeavyMnfc','EastAsia','SEAsia') = 283925.4062 ;
vcif('c_HeavyMnfc','EastAsia','SouthAsia') = 84529.8125 ;
vcif('c_HeavyMnfc','EastAsia','NAmerica') = 486607.625 ;
vcif('c_HeavyMnfc','EastAsia','LatinAmer') = 88760.69531 ;
vcif('c_HeavyMnfc','EastAsia','EU_28') = 364460.0625 ;
vcif('c_HeavyMnfc','EastAsia','MENA') = 109607.1328 ;
vcif('c_HeavyMnfc','EastAsia','SSA') = 38526.41797 ;
vcif('c_HeavyMnfc','EastAsia','RestofWorld') = 66462.51562 ;
vcif('c_HeavyMnfc','SEAsia','Oceania') = 24298.85156 ;
vcif('c_HeavyMnfc','SEAsia','EastAsia') = 213424.9375 ;
vcif('c_HeavyMnfc','SEAsia','SEAsia') = 164180.7188 ;
vcif('c_HeavyMnfc','SEAsia','SouthAsia') = 25804.69141 ;
vcif('c_HeavyMnfc','SEAsia','NAmerica') = 85377.52344 ;
vcif('c_HeavyMnfc','SEAsia','LatinAmer') = 9837.257812 ;
vcif('c_HeavyMnfc','SEAsia','EU_28') = 73334.875 ;
vcif('c_HeavyMnfc','SEAsia','MENA') = 18969.68555 ;
vcif('c_HeavyMnfc','SEAsia','SSA') = 6419.174805 ;
vcif('c_HeavyMnfc','SEAsia','RestofWorld') = 10030.20605 ;
vcif('c_HeavyMnfc','SouthAsia','Oceania') = 1369.224487 ;
vcif('c_HeavyMnfc','SouthAsia','EastAsia') = 18026.58594 ;
vcif('c_HeavyMnfc','SouthAsia','SEAsia') = 12295.94922 ;
vcif('c_HeavyMnfc','SouthAsia','SouthAsia') = 10018.72461 ;
vcif('c_HeavyMnfc','SouthAsia','NAmerica') = 20361.00586 ;
vcif('c_HeavyMnfc','SouthAsia','LatinAmer') = 10320.25195 ;
vcif('c_HeavyMnfc','SouthAsia','EU_28') = 30631.26953 ;
vcif('c_HeavyMnfc','SouthAsia','MENA') = 25860.96875 ;
vcif('c_HeavyMnfc','SouthAsia','SSA') = 14382.08789 ;
vcif('c_HeavyMnfc','SouthAsia','RestofWorld') = 3793.76001 ;
vcif('c_HeavyMnfc','NAmerica','Oceania') = 26220.88867 ;
vcif('c_HeavyMnfc','NAmerica','EastAsia') = 190448.7031 ;
vcif('c_HeavyMnfc','NAmerica','SEAsia') = 57314.51562 ;
vcif('c_HeavyMnfc','NAmerica','SouthAsia') = 19460.07812 ;
vcif('c_HeavyMnfc','NAmerica','NAmerica') = 482864.7812 ;
vcif('c_HeavyMnfc','NAmerica','LatinAmer') = 136343.0156 ;
vcif('c_HeavyMnfc','NAmerica','EU_28') = 245747.7969 ;
vcif('c_HeavyMnfc','NAmerica','MENA') = 54988.89844 ;
vcif('c_HeavyMnfc','NAmerica','SSA') = 18757.35742 ;
vcif('c_HeavyMnfc','NAmerica','RestofWorld') = 39501.45312 ;
vcif('c_HeavyMnfc','LatinAmer','Oceania') = 1568.667847 ;
vcif('c_HeavyMnfc','LatinAmer','EastAsia') = 36269.79688 ;
vcif('c_HeavyMnfc','LatinAmer','SEAsia') = 5887.791504 ;
vcif('c_HeavyMnfc','LatinAmer','SouthAsia') = 1298.421997 ;
vcif('c_HeavyMnfc','LatinAmer','NAmerica') = 82755.98438 ;
vcif('c_HeavyMnfc','LatinAmer','LatinAmer') = 72228.99219 ;
vcif('c_HeavyMnfc','LatinAmer','EU_28') = 30575.19141 ;
vcif('c_HeavyMnfc','LatinAmer','MENA') = 3983.414062 ;
vcif('c_HeavyMnfc','LatinAmer','SSA') = 4465.662598 ;
vcif('c_HeavyMnfc','LatinAmer','RestofWorld') = 11884.57324 ;
vcif('c_HeavyMnfc','EU_28','Oceania') = 30426.1875 ;
vcif('c_HeavyMnfc','EU_28','EastAsia') = 221031.4375 ;
vcif('c_HeavyMnfc','EU_28','SEAsia') = 55243.21875 ;
vcif('c_HeavyMnfc','EU_28','SouthAsia') = 43190.44141 ;
vcif('c_HeavyMnfc','EU_28','NAmerica') = 283816.4375 ;
vcif('c_HeavyMnfc','EU_28','LatinAmer') = 72801.19531 ;
vcif('c_HeavyMnfc','EU_28','EU_28') = 1747708.875 ;
vcif('c_HeavyMnfc','EU_28','MENA') = 205644.6406 ;
vcif('c_HeavyMnfc','EU_28','SSA') = 55969.53906 ;
vcif('c_HeavyMnfc','EU_28','RestofWorld') = 270866.5938 ;
vcif('c_HeavyMnfc','MENA','Oceania') = 3335.787354 ;
vcif('c_HeavyMnfc','MENA','EastAsia') = 79264.85156 ;
vcif('c_HeavyMnfc','MENA','SEAsia') = 24314.97852 ;
vcif('c_HeavyMnfc','MENA','SouthAsia') = 40906.23047 ;
vcif('c_HeavyMnfc','MENA','NAmerica') = 28839.0293 ;
vcif('c_HeavyMnfc','MENA','LatinAmer') = 9080.978516 ;
vcif('c_HeavyMnfc','MENA','EU_28') = 91365.71875 ;
vcif('c_HeavyMnfc','MENA','MENA') = 89382.08594 ;
vcif('c_HeavyMnfc','MENA','SSA') = 18665.32617 ;
vcif('c_HeavyMnfc','MENA','RestofWorld') = 14896.2002 ;
vcif('c_HeavyMnfc','SSA','Oceania') = 560.4981079 ;
vcif('c_HeavyMnfc','SSA','EastAsia') = 15705.93652 ;
vcif('c_HeavyMnfc','SSA','SEAsia') = 3491.2229 ;
vcif('c_HeavyMnfc','SSA','SouthAsia') = 9007.389648 ;
vcif('c_HeavyMnfc','SSA','NAmerica') = 9515.780273 ;
vcif('c_HeavyMnfc','SSA','LatinAmer') = 1222.290283 ;
vcif('c_HeavyMnfc','SSA','EU_28') = 17391.54492 ;
vcif('c_HeavyMnfc','SSA','MENA') = 5699.500977 ;
vcif('c_HeavyMnfc','SSA','SSA') = 19218.99414 ;
vcif('c_HeavyMnfc','SSA','RestofWorld') = 12339.79492 ;
vcif('c_HeavyMnfc','RestofWorld','Oceania') = 3516.006348 ;
vcif('c_HeavyMnfc','RestofWorld','EastAsia') = 60173.57031 ;
vcif('c_HeavyMnfc','RestofWorld','SEAsia') = 21710.65625 ;
vcif('c_HeavyMnfc','RestofWorld','SouthAsia') = 38895.42578 ;
vcif('c_HeavyMnfc','RestofWorld','NAmerica') = 57255.74219 ;
vcif('c_HeavyMnfc','RestofWorld','LatinAmer') = 14527.34082 ;
vcif('c_HeavyMnfc','RestofWorld','EU_28') = 236276.2812 ;
vcif('c_HeavyMnfc','RestofWorld','MENA') = 48096.625 ;
vcif('c_HeavyMnfc','RestofWorld','SSA') = 4651.820801 ;
vcif('c_HeavyMnfc','RestofWorld','RestofWorld') = 54284 ;
vcif('c_Util_Cons','Oceania','Oceania') = 31.07725143 ;
vcif('c_Util_Cons','Oceania','EastAsia') = 201.0083771 ;
vcif('c_Util_Cons','Oceania','SEAsia') = 101.6341858 ;
vcif('c_Util_Cons','Oceania','SouthAsia') = 11.61959457 ;
vcif('c_Util_Cons','Oceania','NAmerica') = 79.40089417 ;
vcif('c_Util_Cons','Oceania','LatinAmer') = 49.84052277 ;
vcif('c_Util_Cons','Oceania','EU_28') = 383.8754883 ;
vcif('c_Util_Cons','Oceania','MENA') = 127.0182343 ;
vcif('c_Util_Cons','Oceania','SSA') = 23.77604485 ;
vcif('c_Util_Cons','Oceania','RestofWorld') = 78.74828339 ;
vcif('c_Util_Cons','EastAsia','Oceania') = 366.0947571 ;
vcif('c_Util_Cons','EastAsia','EastAsia') = 6357.323242 ;
vcif('c_Util_Cons','EastAsia','SEAsia') = 2104.691406 ;
vcif('c_Util_Cons','EastAsia','SouthAsia') = 451.4074097 ;
vcif('c_Util_Cons','EastAsia','NAmerica') = 2141.272949 ;
vcif('c_Util_Cons','EastAsia','LatinAmer') = 486.0794067 ;
vcif('c_Util_Cons','EastAsia','EU_28') = 11890.67676 ;
vcif('c_Util_Cons','EastAsia','MENA') = 4096.105469 ;
vcif('c_Util_Cons','EastAsia','SSA') = 1524.092529 ;
vcif('c_Util_Cons','EastAsia','RestofWorld') = 4494.487305 ;
vcif('c_Util_Cons','SEAsia','Oceania') = 97.43195343 ;
vcif('c_Util_Cons','SEAsia','EastAsia') = 1373.67334 ;
vcif('c_Util_Cons','SEAsia','SEAsia') = 382.1662903 ;
vcif('c_Util_Cons','SEAsia','SouthAsia') = 107.3237152 ;
vcif('c_Util_Cons','SEAsia','NAmerica') = 393.7479858 ;
vcif('c_Util_Cons','SEAsia','LatinAmer') = 142.2795258 ;
vcif('c_Util_Cons','SEAsia','EU_28') = 2745.166992 ;
vcif('c_Util_Cons','SEAsia','MENA') = 812.9971924 ;
vcif('c_Util_Cons','SEAsia','SSA') = 236.680542 ;
vcif('c_Util_Cons','SEAsia','RestofWorld') = 861.5371094 ;
vcif('c_Util_Cons','SouthAsia','Oceania') = 19.71509743 ;
vcif('c_Util_Cons','SouthAsia','EastAsia') = 237.1269989 ;
vcif('c_Util_Cons','SouthAsia','SEAsia') = 93.98284149 ;
vcif('c_Util_Cons','SouthAsia','SouthAsia') = 10.60282135 ;
vcif('c_Util_Cons','SouthAsia','NAmerica') = 61.55842209 ;
vcif('c_Util_Cons','SouthAsia','LatinAmer') = 52.26968765 ;
vcif('c_Util_Cons','SouthAsia','EU_28') = 802.0194092 ;
vcif('c_Util_Cons','SouthAsia','MENA') = 255.4254608 ;
vcif('c_Util_Cons','SouthAsia','SSA') = 61.51704788 ;
vcif('c_Util_Cons','SouthAsia','RestofWorld') = 171.6964722 ;
vcif('c_Util_Cons','NAmerica','Oceania') = 255.6985321 ;
vcif('c_Util_Cons','NAmerica','EastAsia') = 2842.396729 ;
vcif('c_Util_Cons','NAmerica','SEAsia') = 829.7210083 ;
vcif('c_Util_Cons','NAmerica','SouthAsia') = 149.27742 ;
vcif('c_Util_Cons','NAmerica','NAmerica') = 3845.733887 ;
vcif('c_Util_Cons','NAmerica','LatinAmer') = 357.4509888 ;
vcif('c_Util_Cons','NAmerica','EU_28') = 3967.249756 ;
vcif('c_Util_Cons','NAmerica','MENA') = 1951.361328 ;
vcif('c_Util_Cons','NAmerica','SSA') = 511.8539124 ;
vcif('c_Util_Cons','NAmerica','RestofWorld') = 1926.356934 ;
vcif('c_Util_Cons','LatinAmer','Oceania') = 46.47725296 ;
vcif('c_Util_Cons','LatinAmer','EastAsia') = 322.9883423 ;
vcif('c_Util_Cons','LatinAmer','SEAsia') = 206.2909546 ;
vcif('c_Util_Cons','LatinAmer','SouthAsia') = 54.46353149 ;
vcif('c_Util_Cons','LatinAmer','NAmerica') = 209.4647217 ;
vcif('c_Util_Cons','LatinAmer','LatinAmer') = 1090.684814 ;
vcif('c_Util_Cons','LatinAmer','EU_28') = 1094.125122 ;
vcif('c_Util_Cons','LatinAmer','MENA') = 422.6958618 ;
vcif('c_Util_Cons','LatinAmer','SSA') = 186.4594421 ;
vcif('c_Util_Cons','LatinAmer','RestofWorld') = 532.9012451 ;
vcif('c_Util_Cons','EU_28','Oceania') = 525.5808716 ;
vcif('c_Util_Cons','EU_28','EastAsia') = 7820.53125 ;
vcif('c_Util_Cons','EU_28','SEAsia') = 3339.02002 ;
vcif('c_Util_Cons','EU_28','SouthAsia') = 862.8706665 ;
vcif('c_Util_Cons','EU_28','NAmerica') = 1946.275146 ;
vcif('c_Util_Cons','EU_28','LatinAmer') = 2228.263184 ;
vcif('c_Util_Cons','EU_28','EU_28') = 50810.94531 ;
vcif('c_Util_Cons','EU_28','MENA') = 7784.186035 ;
vcif('c_Util_Cons','EU_28','SSA') = 2911.449707 ;
vcif('c_Util_Cons','EU_28','RestofWorld') = 14605.10449 ;
vcif('c_Util_Cons','MENA','Oceania') = 315.7593079 ;
vcif('c_Util_Cons','MENA','EastAsia') = 1619.862305 ;
vcif('c_Util_Cons','MENA','SEAsia') = 778.6488037 ;
vcif('c_Util_Cons','MENA','SouthAsia') = 196.4408264 ;
vcif('c_Util_Cons','MENA','NAmerica') = 1185.112549 ;
vcif('c_Util_Cons','MENA','LatinAmer') = 669.9071655 ;
vcif('c_Util_Cons','MENA','EU_28') = 5639.129395 ;
vcif('c_Util_Cons','MENA','MENA') = 2598.879639 ;
vcif('c_Util_Cons','MENA','SSA') = 375.0569458 ;
vcif('c_Util_Cons','MENA','RestofWorld') = 2366.30835 ;
vcif('c_Util_Cons','SSA','Oceania') = 32.03158188 ;
vcif('c_Util_Cons','SSA','EastAsia') = 254.3527527 ;
vcif('c_Util_Cons','SSA','SEAsia') = 144.2131348 ;
vcif('c_Util_Cons','SSA','SouthAsia') = 27.73508072 ;
vcif('c_Util_Cons','SSA','NAmerica') = 159.2519226 ;
vcif('c_Util_Cons','SSA','LatinAmer') = 124.1552811 ;
vcif('c_Util_Cons','SSA','EU_28') = 928.7954712 ;
vcif('c_Util_Cons','SSA','MENA') = 392.601532 ;
vcif('c_Util_Cons','SSA','SSA') = 1045.420776 ;
vcif('c_Util_Cons','SSA','RestofWorld') = 252.3145447 ;
vcif('c_Util_Cons','RestofWorld','Oceania') = 196.6998596 ;
vcif('c_Util_Cons','RestofWorld','EastAsia') = 1458.554199 ;
vcif('c_Util_Cons','RestofWorld','SEAsia') = 1037.572632 ;
vcif('c_Util_Cons','RestofWorld','SouthAsia') = 165.8195496 ;
vcif('c_Util_Cons','RestofWorld','NAmerica') = 818.8847656 ;
vcif('c_Util_Cons','RestofWorld','LatinAmer') = 856.9828491 ;
vcif('c_Util_Cons','RestofWorld','EU_28') = 12403.9707 ;
vcif('c_Util_Cons','RestofWorld','MENA') = 3945.540527 ;
vcif('c_Util_Cons','RestofWorld','SSA') = 488.2766418 ;
vcif('c_Util_Cons','RestofWorld','RestofWorld') = 3622.735107 ;
vcif('c_TransComm','Oceania','Oceania') = 2169.616455 ;
vcif('c_TransComm','Oceania','EastAsia') = 6539.338867 ;
vcif('c_TransComm','Oceania','SEAsia') = 1562.600342 ;
vcif('c_TransComm','Oceania','SouthAsia') = 348.3964844 ;
vcif('c_TransComm','Oceania','NAmerica') = 4447.299316 ;
vcif('c_TransComm','Oceania','LatinAmer') = 751.6378174 ;
vcif('c_TransComm','Oceania','EU_28') = 7174.994629 ;
vcif('c_TransComm','Oceania','MENA') = 1683.146484 ;
vcif('c_TransComm','Oceania','SSA') = 553.6177979 ;
vcif('c_TransComm','Oceania','RestofWorld') = 714.2366333 ;
vcif('c_TransComm','EastAsia','Oceania') = 4212.180664 ;
vcif('c_TransComm','EastAsia','EastAsia') = 65740.33594 ;
vcif('c_TransComm','EastAsia','SEAsia') = 15686.9043 ;
vcif('c_TransComm','EastAsia','SouthAsia') = 3984.506836 ;
vcif('c_TransComm','EastAsia','NAmerica') = 34000.04688 ;
vcif('c_TransComm','EastAsia','LatinAmer') = 6217.623047 ;
vcif('c_TransComm','EastAsia','EU_28') = 58985.67188 ;
vcif('c_TransComm','EastAsia','MENA') = 11513.73438 ;
vcif('c_TransComm','EastAsia','SSA') = 4201.472656 ;
vcif('c_TransComm','EastAsia','RestofWorld') = 6782.009766 ;
vcif('c_TransComm','SEAsia','Oceania') = 2004.836426 ;
vcif('c_TransComm','SEAsia','EastAsia') = 16652.25781 ;
vcif('c_TransComm','SEAsia','SEAsia') = 2442.834961 ;
vcif('c_TransComm','SEAsia','SouthAsia') = 1960.946777 ;
vcif('c_TransComm','SEAsia','NAmerica') = 10710.11719 ;
vcif('c_TransComm','SEAsia','LatinAmer') = 2438.797363 ;
vcif('c_TransComm','SEAsia','EU_28') = 30842.5957 ;
vcif('c_TransComm','SEAsia','MENA') = 3997.199951 ;
vcif('c_TransComm','SEAsia','SSA') = 1551.950195 ;
vcif('c_TransComm','SEAsia','RestofWorld') = 3739.834961 ;
vcif('c_TransComm','SouthAsia','Oceania') = 492.8401489 ;
vcif('c_TransComm','SouthAsia','EastAsia') = 5351.861816 ;
vcif('c_TransComm','SouthAsia','SEAsia') = 1405.473877 ;
vcif('c_TransComm','SouthAsia','SouthAsia') = 138.3771057 ;
vcif('c_TransComm','SouthAsia','NAmerica') = 4213.526367 ;
vcif('c_TransComm','SouthAsia','LatinAmer') = 716.8010864 ;
vcif('c_TransComm','SouthAsia','EU_28') = 7466.467285 ;
vcif('c_TransComm','SouthAsia','MENA') = 1734.245361 ;
vcif('c_TransComm','SouthAsia','SSA') = 441.8512878 ;
vcif('c_TransComm','SouthAsia','RestofWorld') = 970.2805176 ;
vcif('c_TransComm','NAmerica','Oceania') = 3273.124268 ;
vcif('c_TransComm','NAmerica','EastAsia') = 27415.13086 ;
vcif('c_TransComm','NAmerica','SEAsia') = 6990.8125 ;
vcif('c_TransComm','NAmerica','SouthAsia') = 3207.255859 ;
vcif('c_TransComm','NAmerica','NAmerica') = 21350.7207 ;
vcif('c_TransComm','NAmerica','LatinAmer') = 4646.54834 ;
vcif('c_TransComm','NAmerica','EU_28') = 55497.03906 ;
vcif('c_TransComm','NAmerica','MENA') = 10289.19824 ;
vcif('c_TransComm','NAmerica','SSA') = 2744.233398 ;
vcif('c_TransComm','NAmerica','RestofWorld') = 4709.695801 ;
vcif('c_TransComm','LatinAmer','Oceania') = 919.934021 ;
vcif('c_TransComm','LatinAmer','EastAsia') = 7700.701172 ;
vcif('c_TransComm','LatinAmer','SEAsia') = 2465.787598 ;
vcif('c_TransComm','LatinAmer','SouthAsia') = 922.2738647 ;
vcif('c_TransComm','LatinAmer','NAmerica') = 6838.367676 ;
vcif('c_TransComm','LatinAmer','LatinAmer') = 1695.694336 ;
vcif('c_TransComm','LatinAmer','EU_28') = 21363.47656 ;
vcif('c_TransComm','LatinAmer','MENA') = 3560.369385 ;
vcif('c_TransComm','LatinAmer','SSA') = 926.9926147 ;
vcif('c_TransComm','LatinAmer','RestofWorld') = 3073.308105 ;
vcif('c_TransComm','EU_28','Oceania') = 8274.443359 ;
vcif('c_TransComm','EU_28','EastAsia') = 52822.07812 ;
vcif('c_TransComm','EU_28','SEAsia') = 22939.01562 ;
vcif('c_TransComm','EU_28','SouthAsia') = 10831.97461 ;
vcif('c_TransComm','EU_28','NAmerica') = 60427.17969 ;
vcif('c_TransComm','EU_28','LatinAmer') = 22014.25 ;
vcif('c_TransComm','EU_28','EU_28') = 316962.3125 ;
vcif('c_TransComm','EU_28','MENA') = 28538.63281 ;
vcif('c_TransComm','EU_28','SSA') = 12248.56934 ;
vcif('c_TransComm','EU_28','RestofWorld') = 40754.24609 ;
vcif('c_TransComm','MENA','Oceania') = 1772.074219 ;
vcif('c_TransComm','MENA','EastAsia') = 15178.52441 ;
vcif('c_TransComm','MENA','SEAsia') = 4101.641602 ;
vcif('c_TransComm','MENA','SouthAsia') = 1782.15271 ;
vcif('c_TransComm','MENA','NAmerica') = 16937.83984 ;
vcif('c_TransComm','MENA','LatinAmer') = 3612.134521 ;
vcif('c_TransComm','MENA','EU_28') = 40371.64844 ;
vcif('c_TransComm','MENA','MENA') = 4900.626953 ;
vcif('c_TransComm','MENA','SSA') = 1857.097778 ;
vcif('c_TransComm','MENA','RestofWorld') = 6201.303223 ;
vcif('c_TransComm','SSA','Oceania') = 429.4367981 ;
vcif('c_TransComm','SSA','EastAsia') = 3114.362305 ;
vcif('c_TransComm','SSA','SEAsia') = 998.7744751 ;
vcif('c_TransComm','SSA','SouthAsia') = 317.2618408 ;
vcif('c_TransComm','SSA','NAmerica') = 3187.952148 ;
vcif('c_TransComm','SSA','LatinAmer') = 675.1118164 ;
vcif('c_TransComm','SSA','EU_28') = 8443.737305 ;
vcif('c_TransComm','SSA','MENA') = 1192.515625 ;
vcif('c_TransComm','SSA','SSA') = 354.2862854 ;
vcif('c_TransComm','SSA','RestofWorld') = 896.7719116 ;
vcif('c_TransComm','RestofWorld','Oceania') = 947.9411621 ;
vcif('c_TransComm','RestofWorld','EastAsia') = 10903.88184 ;
vcif('c_TransComm','RestofWorld','SEAsia') = 4127.525391 ;
vcif('c_TransComm','RestofWorld','SouthAsia') = 1044.633789 ;
vcif('c_TransComm','RestofWorld','NAmerica') = 8281.378906 ;
vcif('c_TransComm','RestofWorld','LatinAmer') = 2161.365723 ;
vcif('c_TransComm','RestofWorld','EU_28') = 36154.76172 ;
vcif('c_TransComm','RestofWorld','MENA') = 4304.719238 ;
vcif('c_TransComm','RestofWorld','SSA') = 1237.386597 ;
vcif('c_TransComm','RestofWorld','RestofWorld') = 3337.639893 ;
vcif('c_OthService','Oceania','Oceania') = 2283.932617 ;
vcif('c_OthService','Oceania','EastAsia') = 6401.10791 ;
vcif('c_OthService','Oceania','SEAsia') = 2127.986084 ;
vcif('c_OthService','Oceania','SouthAsia') = 573.2736206 ;
vcif('c_OthService','Oceania','NAmerica') = 5652.233887 ;
vcif('c_OthService','Oceania','LatinAmer') = 786.1976318 ;
vcif('c_OthService','Oceania','EU_28') = 7780.802734 ;
vcif('c_OthService','Oceania','MENA') = 2293.664307 ;
vcif('c_OthService','Oceania','SSA') = 659.3297729 ;
vcif('c_OthService','Oceania','RestofWorld') = 841.8104858 ;
vcif('c_OthService','EastAsia','Oceania') = 2661.219482 ;
vcif('c_OthService','EastAsia','EastAsia') = 24510.78125 ;
vcif('c_OthService','EastAsia','SEAsia') = 10720.59766 ;
vcif('c_OthService','EastAsia','SouthAsia') = 5830.672363 ;
vcif('c_OthService','EastAsia','NAmerica') = 40831.04688 ;
vcif('c_OthService','EastAsia','LatinAmer') = 6029.530762 ;
vcif('c_OthService','EastAsia','EU_28') = 51103.73047 ;
vcif('c_OthService','EastAsia','MENA') = 13700.07715 ;
vcif('c_OthService','EastAsia','SSA') = 3753.361328 ;
vcif('c_OthService','EastAsia','RestofWorld') = 6419.271484 ;
vcif('c_OthService','SEAsia','Oceania') = 2375.387695 ;
vcif('c_OthService','SEAsia','EastAsia') = 12545.82324 ;
vcif('c_OthService','SEAsia','SEAsia') = 2450.027588 ;
vcif('c_OthService','SEAsia','SouthAsia') = 4212.79248 ;
vcif('c_OthService','SEAsia','NAmerica') = 11423.6748 ;
vcif('c_OthService','SEAsia','LatinAmer') = 2735.078369 ;
vcif('c_OthService','SEAsia','EU_28') = 30411.67188 ;
vcif('c_OthService','SEAsia','MENA') = 4138.499512 ;
vcif('c_OthService','SEAsia','SSA') = 1649.08313 ;
vcif('c_OthService','SEAsia','RestofWorld') = 3016.982666 ;
vcif('c_OthService','SouthAsia','Oceania') = 925.135498 ;
vcif('c_OthService','SouthAsia','EastAsia') = 7902.925781 ;
vcif('c_OthService','SouthAsia','SEAsia') = 2953.47168 ;
vcif('c_OthService','SouthAsia','SouthAsia') = 317.1229553 ;
vcif('c_OthService','SouthAsia','NAmerica') = 23476.5957 ;
vcif('c_OthService','SouthAsia','LatinAmer') = 1873.734741 ;
vcif('c_OthService','SouthAsia','EU_28') = 25990.93555 ;
vcif('c_OthService','SouthAsia','MENA') = 3121.552734 ;
vcif('c_OthService','SouthAsia','SSA') = 1184.272705 ;
vcif('c_OthService','SouthAsia','RestofWorld') = 2095.656494 ;
vcif('c_OthService','NAmerica','Oceania') = 6903.589844 ;
vcif('c_OthService','NAmerica','EastAsia') = 61267.43359 ;
vcif('c_OthService','NAmerica','SEAsia') = 16068.8877 ;
vcif('c_OthService','NAmerica','SouthAsia') = 11846.02148 ;
vcif('c_OthService','NAmerica','NAmerica') = 46485.67578 ;
vcif('c_OthService','NAmerica','LatinAmer') = 11753.08398 ;
vcif('c_OthService','NAmerica','EU_28') = 149312.7031 ;
vcif('c_OthService','NAmerica','MENA') = 35297.79297 ;
vcif('c_OthService','NAmerica','SSA') = 10327.73535 ;
vcif('c_OthService','NAmerica','RestofWorld') = 14985.01367 ;
vcif('c_OthService','LatinAmer','Oceania') = 767.5485229 ;
vcif('c_OthService','LatinAmer','EastAsia') = 6080.470215 ;
vcif('c_OthService','LatinAmer','SEAsia') = 2984.134277 ;
vcif('c_OthService','LatinAmer','SouthAsia') = 1425.15332 ;
vcif('c_OthService','LatinAmer','NAmerica') = 10734.77051 ;
vcif('c_OthService','LatinAmer','LatinAmer') = 1358.890869 ;
vcif('c_OthService','LatinAmer','EU_28') = 18873.36719 ;
vcif('c_OthService','LatinAmer','MENA') = 4633.399902 ;
vcif('c_OthService','LatinAmer','SSA') = 1248.408447 ;
vcif('c_OthService','LatinAmer','RestofWorld') = 2278.50415 ;
vcif('c_OthService','EU_28','Oceania') = 8443.917969 ;
vcif('c_OthService','EU_28','EastAsia') = 56998.12109 ;
vcif('c_OthService','EU_28','SEAsia') = 31403.49414 ;
vcif('c_OthService','EU_28','SouthAsia') = 20280.83594 ;
vcif('c_OthService','EU_28','NAmerica') = 129155.0859 ;
vcif('c_OthService','EU_28','LatinAmer') = 27664.25195 ;
vcif('c_OthService','EU_28','EU_28') = 491871.9062 ;
vcif('c_OthService','EU_28','MENA') = 47819.35156 ;
vcif('c_OthService','EU_28','SSA') = 19573.19727 ;
vcif('c_OthService','EU_28','RestofWorld') = 62218.03125 ;
vcif('c_OthService','MENA','Oceania') = 1445.43689 ;
vcif('c_OthService','MENA','EastAsia') = 12839.02148 ;
vcif('c_OthService','MENA','SEAsia') = 3465.295898 ;
vcif('c_OthService','MENA','SouthAsia') = 2367.249756 ;
vcif('c_OthService','MENA','NAmerica') = 18135.54883 ;
vcif('c_OthService','MENA','LatinAmer') = 3768.030029 ;
vcif('c_OthService','MENA','EU_28') = 34945.88281 ;
vcif('c_OthService','MENA','MENA') = 5371.029785 ;
vcif('c_OthService','MENA','SSA') = 1977.580933 ;
vcif('c_OthService','MENA','RestofWorld') = 5075.946289 ;
vcif('c_OthService','SSA','Oceania') = 308.29599 ;
vcif('c_OthService','SSA','EastAsia') = 2903.436035 ;
vcif('c_OthService','SSA','SEAsia') = 790.6919556 ;
vcif('c_OthService','SSA','SouthAsia') = 361.8479004 ;
vcif('c_OthService','SSA','NAmerica') = 4691.35791 ;
vcif('c_OthService','SSA','LatinAmer') = 644.6809692 ;
vcif('c_OthService','SSA','EU_28') = 6700.551758 ;
vcif('c_OthService','SSA','MENA') = 2125.437988 ;
vcif('c_OthService','SSA','SSA') = 492.7622986 ;
vcif('c_OthService','SSA','RestofWorld') = 848.0107422 ;
vcif('c_OthService','RestofWorld','Oceania') = 842.0559692 ;
vcif('c_OthService','RestofWorld','EastAsia') = 8420.21582 ;
vcif('c_OthService','RestofWorld','SEAsia') = 5079.750488 ;
vcif('c_OthService','RestofWorld','SouthAsia') = 2429.520752 ;
vcif('c_OthService','RestofWorld','NAmerica') = 18953.99609 ;
vcif('c_OthService','RestofWorld','LatinAmer') = 2311.362549 ;
vcif('c_OthService','RestofWorld','EU_28') = 48936.79297 ;
vcif('c_OthService','RestofWorld','MENA') = 6074.623535 ;
vcif('c_OthService','RestofWorld','SSA') = 2068.710938 ;
vcif('c_OthService','RestofWorld','RestofWorld') = 3546.81543 ;

* vmsb data (1000 cells)
vmsb('c_Crops','Oceania','Oceania') = 1012.145447 ;
vmsb('c_Crops','Oceania','EastAsia') = 6828.913086 ;
vmsb('c_Crops','Oceania','SEAsia') = 4559.614258 ;
vmsb('c_Crops','Oceania','SouthAsia') = 1152.028809 ;
vmsb('c_Crops','Oceania','NAmerica') = 389.3999329 ;
vmsb('c_Crops','Oceania','LatinAmer') = 53.590065 ;
vmsb('c_Crops','Oceania','EU_28') = 1686.741455 ;
vmsb('c_Crops','Oceania','MENA') = 2440.434326 ;
vmsb('c_Crops','Oceania','SSA') = 516.2202759 ;
vmsb('c_Crops','Oceania','RestofWorld') = 65.63729095 ;
vmsb('c_Crops','EastAsia','Oceania') = 153.9148254 ;
vmsb('c_Crops','EastAsia','EastAsia') = 6509.786621 ;
vmsb('c_Crops','EastAsia','SEAsia') = 5387.795898 ;
vmsb('c_Crops','EastAsia','SouthAsia') = 1078.723511 ;
vmsb('c_Crops','EastAsia','NAmerica') = 1389.879395 ;
vmsb('c_Crops','EastAsia','LatinAmer') = 526.2064209 ;
vmsb('c_Crops','EastAsia','EU_28') = 1998.712158 ;
vmsb('c_Crops','EastAsia','MENA') = 1270.794434 ;
vmsb('c_Crops','EastAsia','SSA') = 965.6106567 ;
vmsb('c_Crops','EastAsia','RestofWorld') = 1347.47229 ;
vmsb('c_Crops','SEAsia','Oceania') = 598.2780762 ;
vmsb('c_Crops','SEAsia','EastAsia') = 9872.158203 ;
vmsb('c_Crops','SEAsia','SEAsia') = 7322.305664 ;
vmsb('c_Crops','SEAsia','SouthAsia') = 2385.478516 ;
vmsb('c_Crops','SEAsia','NAmerica') = 3049.081543 ;
vmsb('c_Crops','SEAsia','LatinAmer') = 184.3714142 ;
vmsb('c_Crops','SEAsia','EU_28') = 3864.288818 ;
vmsb('c_Crops','SEAsia','MENA') = 2344.663574 ;
vmsb('c_Crops','SEAsia','SSA') = 4269.716309 ;
vmsb('c_Crops','SEAsia','RestofWorld') = 791.5128784 ;
vmsb('c_Crops','SouthAsia','Oceania') = 197.1062622 ;
vmsb('c_Crops','SouthAsia','EastAsia') = 4114.569824 ;
vmsb('c_Crops','SouthAsia','SEAsia') = 3642.310791 ;
vmsb('c_Crops','SouthAsia','SouthAsia') = 4369.749512 ;
vmsb('c_Crops','SouthAsia','NAmerica') = 1482.884155 ;
vmsb('c_Crops','SouthAsia','LatinAmer') = 194.6132812 ;
vmsb('c_Crops','SouthAsia','EU_28') = 2799.098389 ;
vmsb('c_Crops','SouthAsia','MENA') = 7786.577148 ;
vmsb('c_Crops','SouthAsia','SSA') = 2009.525024 ;
vmsb('c_Crops','SouthAsia','RestofWorld') = 1199.851929 ;
vmsb('c_Crops','NAmerica','Oceania') = 481.2120972 ;
vmsb('c_Crops','NAmerica','EastAsia') = 60144.25 ;
vmsb('c_Crops','NAmerica','SEAsia') = 7359.572754 ;
vmsb('c_Crops','NAmerica','SouthAsia') = 4043.277588 ;
vmsb('c_Crops','NAmerica','NAmerica') = 34396.83594 ;
vmsb('c_Crops','NAmerica','LatinAmer') = 9166.791016 ;
vmsb('c_Crops','NAmerica','EU_28') = 8487.09668 ;
vmsb('c_Crops','NAmerica','MENA') = 12726.75488 ;
vmsb('c_Crops','NAmerica','SSA') = 5064.516113 ;
vmsb('c_Crops','NAmerica','RestofWorld') = 969.2433472 ;
vmsb('c_Crops','LatinAmer','Oceania') = 301.8291931 ;
vmsb('c_Crops','LatinAmer','EastAsia') = 29905.89844 ;
vmsb('c_Crops','LatinAmer','SEAsia') = 3822.346924 ;
vmsb('c_Crops','LatinAmer','SouthAsia') = 484.4683533 ;
vmsb('c_Crops','LatinAmer','NAmerica') = 18127.68555 ;
vmsb('c_Crops','LatinAmer','LatinAmer') = 10751.62207 ;
vmsb('c_Crops','LatinAmer','EU_28') = 25778.28906 ;
vmsb('c_Crops','LatinAmer','MENA') = 9985.241211 ;
vmsb('c_Crops','LatinAmer','SSA') = 1941.005371 ;
vmsb('c_Crops','LatinAmer','RestofWorld') = 4417.928223 ;
vmsb('c_Crops','EU_28','Oceania') = 139.3562469 ;
vmsb('c_Crops','EU_28','EastAsia') = 1360.819092 ;
vmsb('c_Crops','EU_28','SEAsia') = 408.2565613 ;
vmsb('c_Crops','EU_28','SouthAsia') = 438.7893066 ;
vmsb('c_Crops','EU_28','NAmerica') = 1717.639404 ;
vmsb('c_Crops','EU_28','LatinAmer') = 768.7109985 ;
vmsb('c_Crops','EU_28','EU_28') = 74010.92969 ;
vmsb('c_Crops','EU_28','MENA') = 10512.44629 ;
vmsb('c_Crops','EU_28','SSA') = 2528.09668 ;
vmsb('c_Crops','EU_28','RestofWorld') = 10347.79688 ;
vmsb('c_Crops','MENA','Oceania') = 142.0870209 ;
vmsb('c_Crops','MENA','EastAsia') = 735.1803589 ;
vmsb('c_Crops','MENA','SEAsia') = 396.9181519 ;
vmsb('c_Crops','MENA','SouthAsia') = 713.1053467 ;
vmsb('c_Crops','MENA','NAmerica') = 1030.250854 ;
vmsb('c_Crops','MENA','LatinAmer') = 198.1359558 ;
vmsb('c_Crops','MENA','EU_28') = 8101.435547 ;
vmsb('c_Crops','MENA','MENA') = 8160.123047 ;
vmsb('c_Crops','MENA','SSA') = 679.3990479 ;
vmsb('c_Crops','MENA','RestofWorld') = 4714.517578 ;
vmsb('c_Crops','SSA','Oceania') = 93.7681427 ;
vmsb('c_Crops','SSA','EastAsia') = 3953.18457 ;
vmsb('c_Crops','SSA','SEAsia') = 2209.665283 ;
vmsb('c_Crops','SSA','SouthAsia') = 2309.179688 ;
vmsb('c_Crops','SSA','NAmerica') = 2635.209473 ;
vmsb('c_Crops','SSA','LatinAmer') = 281.2696533 ;
vmsb('c_Crops','SSA','EU_28') = 10760.63086 ;
vmsb('c_Crops','SSA','MENA') = 2448.250244 ;
vmsb('c_Crops','SSA','SSA') = 2725.835693 ;
vmsb('c_Crops','SSA','RestofWorld') = 1330.154541 ;
vmsb('c_Crops','RestofWorld','Oceania') = 10.54585075 ;
vmsb('c_Crops','RestofWorld','EastAsia') = 905.6586914 ;
vmsb('c_Crops','RestofWorld','SEAsia') = 78.76496124 ;
vmsb('c_Crops','RestofWorld','SouthAsia') = 347.8556519 ;
vmsb('c_Crops','RestofWorld','NAmerica') = 140.4584503 ;
vmsb('c_Crops','RestofWorld','LatinAmer') = 104.1374435 ;
vmsb('c_Crops','RestofWorld','EU_28') = 4130.672852 ;
vmsb('c_Crops','RestofWorld','MENA') = 11970.80273 ;
vmsb('c_Crops','RestofWorld','SSA') = 935.6115112 ;
vmsb('c_Crops','RestofWorld','RestofWorld') = 3678.873779 ;
vmsb('c_MeatLstk','Oceania','Oceania') = 670.8841553 ;
vmsb('c_MeatLstk','Oceania','EastAsia') = 11172.19629 ;
vmsb('c_MeatLstk','Oceania','SEAsia') = 1701.034058 ;
vmsb('c_MeatLstk','Oceania','SouthAsia') = 349.8945007 ;
vmsb('c_MeatLstk','Oceania','NAmerica') = 3126.846924 ;
vmsb('c_MeatLstk','Oceania','LatinAmer') = 232.1907349 ;
vmsb('c_MeatLstk','Oceania','EU_28') = 3523.261475 ;
vmsb('c_MeatLstk','Oceania','MENA') = 1476.036621 ;
vmsb('c_MeatLstk','Oceania','SSA') = 188.6508026 ;
vmsb('c_MeatLstk','Oceania','RestofWorld') = 891.3734741 ;
vmsb('c_MeatLstk','EastAsia','Oceania') = 47.33298492 ;
vmsb('c_MeatLstk','EastAsia','EastAsia') = 3652.962402 ;
vmsb('c_MeatLstk','EastAsia','SEAsia') = 573.4872437 ;
vmsb('c_MeatLstk','EastAsia','SouthAsia') = 75.77277374 ;
vmsb('c_MeatLstk','EastAsia','NAmerica') = 752.4389648 ;
vmsb('c_MeatLstk','EastAsia','LatinAmer') = 102.7839508 ;
vmsb('c_MeatLstk','EastAsia','EU_28') = 1687.029297 ;
vmsb('c_MeatLstk','EastAsia','MENA') = 235.7218933 ;
vmsb('c_MeatLstk','EastAsia','SSA') = 94.34947205 ;
vmsb('c_MeatLstk','EastAsia','RestofWorld') = 186.9283752 ;
vmsb('c_MeatLstk','SEAsia','Oceania') = 17.45967674 ;
vmsb('c_MeatLstk','SEAsia','EastAsia') = 2176.20752 ;
vmsb('c_MeatLstk','SEAsia','SEAsia') = 862.2329712 ;
vmsb('c_MeatLstk','SEAsia','SouthAsia') = 27.07761383 ;
vmsb('c_MeatLstk','SEAsia','NAmerica') = 225.8683929 ;
vmsb('c_MeatLstk','SEAsia','LatinAmer') = 9.527864456 ;
vmsb('c_MeatLstk','SEAsia','EU_28') = 1335.346802 ;
vmsb('c_MeatLstk','SEAsia','MENA') = 83.64968872 ;
vmsb('c_MeatLstk','SEAsia','SSA') = 52.82381058 ;
vmsb('c_MeatLstk','SEAsia','RestofWorld') = 38.74852371 ;
vmsb('c_MeatLstk','SouthAsia','Oceania') = 6.192296982 ;
vmsb('c_MeatLstk','SouthAsia','EastAsia') = 90.22405243 ;
vmsb('c_MeatLstk','SouthAsia','SEAsia') = 1474.654053 ;
vmsb('c_MeatLstk','SouthAsia','SouthAsia') = 87.77003479 ;
vmsb('c_MeatLstk','SouthAsia','NAmerica') = 138.4681702 ;
vmsb('c_MeatLstk','SouthAsia','LatinAmer') = 9.666842461 ;
vmsb('c_MeatLstk','SouthAsia','EU_28') = 151.6547089 ;
vmsb('c_MeatLstk','SouthAsia','MENA') = 1585.839111 ;
vmsb('c_MeatLstk','SouthAsia','SSA') = 223.2536621 ;
vmsb('c_MeatLstk','SouthAsia','RestofWorld') = 126.1120453 ;
vmsb('c_MeatLstk','NAmerica','Oceania') = 544.9553833 ;
vmsb('c_MeatLstk','NAmerica','EastAsia') = 17302.0918 ;
vmsb('c_MeatLstk','NAmerica','SEAsia') = 814.6217651 ;
vmsb('c_MeatLstk','NAmerica','SouthAsia') = 144.1173096 ;
vmsb('c_MeatLstk','NAmerica','NAmerica') = 14047.13184 ;
vmsb('c_MeatLstk','NAmerica','LatinAmer') = 2137.194092 ;
vmsb('c_MeatLstk','NAmerica','EU_28') = 1463.270874 ;
vmsb('c_MeatLstk','NAmerica','MENA') = 1345.680786 ;
vmsb('c_MeatLstk','NAmerica','SSA') = 785.390564 ;
vmsb('c_MeatLstk','NAmerica','RestofWorld') = 2880.458984 ;
vmsb('c_MeatLstk','LatinAmer','Oceania') = 45.99224091 ;
vmsb('c_MeatLstk','LatinAmer','EastAsia') = 4367.625977 ;
vmsb('c_MeatLstk','LatinAmer','SEAsia') = 560.4363403 ;
vmsb('c_MeatLstk','LatinAmer','SouthAsia') = 68.63815308 ;
vmsb('c_MeatLstk','LatinAmer','NAmerica') = 1581.283936 ;
vmsb('c_MeatLstk','LatinAmer','LatinAmer') = 4713.698242 ;
vmsb('c_MeatLstk','LatinAmer','EU_28') = 6367.041016 ;
vmsb('c_MeatLstk','LatinAmer','MENA') = 6285.923828 ;
vmsb('c_MeatLstk','LatinAmer','SSA') = 1334.130981 ;
vmsb('c_MeatLstk','LatinAmer','RestofWorld') = 4310.359375 ;
vmsb('c_MeatLstk','EU_28','Oceania') = 418.7926941 ;
vmsb('c_MeatLstk','EU_28','EastAsia') = 7151.648438 ;
vmsb('c_MeatLstk','EU_28','SEAsia') = 652.4156494 ;
vmsb('c_MeatLstk','EU_28','SouthAsia') = 173.591568 ;
vmsb('c_MeatLstk','EU_28','NAmerica') = 1199.159912 ;
vmsb('c_MeatLstk','EU_28','LatinAmer') = 442.5281982 ;
vmsb('c_MeatLstk','EU_28','EU_28') = 66170.91406 ;
vmsb('c_MeatLstk','EU_28','MENA') = 4139.003906 ;
vmsb('c_MeatLstk','EU_28','SSA') = 1336.53186 ;
vmsb('c_MeatLstk','EU_28','RestofWorld') = 8195.791992 ;
vmsb('c_MeatLstk','MENA','Oceania') = 13.47676277 ;
vmsb('c_MeatLstk','MENA','EastAsia') = 149.3258514 ;
vmsb('c_MeatLstk','MENA','SEAsia') = 69.20063782 ;
vmsb('c_MeatLstk','MENA','SouthAsia') = 92.9947052 ;
vmsb('c_MeatLstk','MENA','NAmerica') = 110.6328125 ;
vmsb('c_MeatLstk','MENA','LatinAmer') = 20.56695938 ;
vmsb('c_MeatLstk','MENA','EU_28') = 784.7366333 ;
vmsb('c_MeatLstk','MENA','MENA') = 1899.542603 ;
vmsb('c_MeatLstk','MENA','SSA') = 97.17369843 ;
vmsb('c_MeatLstk','MENA','RestofWorld') = 160.212738 ;
vmsb('c_MeatLstk','SSA','Oceania') = 9.710243225 ;
vmsb('c_MeatLstk','SSA','EastAsia') = 441.7434692 ;
vmsb('c_MeatLstk','SSA','SEAsia') = 35.84063721 ;
vmsb('c_MeatLstk','SSA','SouthAsia') = 112.9528046 ;
vmsb('c_MeatLstk','SSA','NAmerica') = 93.54225159 ;
vmsb('c_MeatLstk','SSA','LatinAmer') = 14.02079201 ;
vmsb('c_MeatLstk','SSA','EU_28') = 449.585083 ;
vmsb('c_MeatLstk','SSA','MENA') = 860.4006958 ;
vmsb('c_MeatLstk','SSA','SSA') = 604.2510376 ;
vmsb('c_MeatLstk','SSA','RestofWorld') = 188.7481232 ;
vmsb('c_MeatLstk','RestofWorld','Oceania') = 19.56282425 ;
vmsb('c_MeatLstk','RestofWorld','EastAsia') = 219.2642975 ;
vmsb('c_MeatLstk','RestofWorld','SEAsia') = 44.87368774 ;
vmsb('c_MeatLstk','RestofWorld','SouthAsia') = 15.57795334 ;
vmsb('c_MeatLstk','RestofWorld','NAmerica') = 134.6280518 ;
vmsb('c_MeatLstk','RestofWorld','LatinAmer') = 19.6953392 ;
vmsb('c_MeatLstk','RestofWorld','EU_28') = 1031.569458 ;
vmsb('c_MeatLstk','RestofWorld','MENA') = 203.4542847 ;
vmsb('c_MeatLstk','RestofWorld','SSA') = 25.97905731 ;
vmsb('c_MeatLstk','RestofWorld','RestofWorld') = 557.4868164 ;
vmsb('c_Extraction','Oceania','Oceania') = 1292.091919 ;
vmsb('c_Extraction','Oceania','EastAsia') = 134913.1562 ;
vmsb('c_Extraction','Oceania','SEAsia') = 4702.272949 ;
vmsb('c_Extraction','Oceania','SouthAsia') = 15399.42285 ;
vmsb('c_Extraction','Oceania','NAmerica') = 2149.059814 ;
vmsb('c_Extraction','Oceania','LatinAmer') = 1245.80957 ;
vmsb('c_Extraction','Oceania','EU_28') = 5689.05127 ;
vmsb('c_Extraction','Oceania','MENA') = 954.3740845 ;
vmsb('c_Extraction','Oceania','SSA') = 361.5511475 ;
vmsb('c_Extraction','Oceania','RestofWorld') = 225.7636108 ;
vmsb('c_Extraction','EastAsia','Oceania') = 1935.283447 ;
vmsb('c_Extraction','EastAsia','EastAsia') = 21276.25195 ;
vmsb('c_Extraction','EastAsia','SEAsia') = 2042.72937 ;
vmsb('c_Extraction','EastAsia','SouthAsia') = 1884.031006 ;
vmsb('c_Extraction','EastAsia','NAmerica') = 1190.653687 ;
vmsb('c_Extraction','EastAsia','LatinAmer') = 122.453392 ;
vmsb('c_Extraction','EastAsia','EU_28') = 1506.543823 ;
vmsb('c_Extraction','EastAsia','MENA') = 259.6303711 ;
vmsb('c_Extraction','EastAsia','SSA') = 108.4425201 ;
vmsb('c_Extraction','EastAsia','RestofWorld') = 266.7410278 ;
vmsb('c_Extraction','SEAsia','Oceania') = 7528.302734 ;
vmsb('c_Extraction','SEAsia','EastAsia') = 76205.59375 ;
vmsb('c_Extraction','SEAsia','SEAsia') = 16454.92578 ;
vmsb('c_Extraction','SEAsia','SouthAsia') = 10285.56445 ;
vmsb('c_Extraction','SEAsia','NAmerica') = 1942.894653 ;
vmsb('c_Extraction','SEAsia','LatinAmer') = 164.3794556 ;
vmsb('c_Extraction','SEAsia','EU_28') = 2908.319092 ;
vmsb('c_Extraction','SEAsia','MENA') = 803.9605713 ;
vmsb('c_Extraction','SEAsia','SSA') = 62.02430344 ;
vmsb('c_Extraction','SEAsia','RestofWorld') = 323.194458 ;
vmsb('c_Extraction','SouthAsia','Oceania') = 42.55395126 ;
vmsb('c_Extraction','SouthAsia','EastAsia') = 13209.8291 ;
vmsb('c_Extraction','SouthAsia','SEAsia') = 369.3856812 ;
vmsb('c_Extraction','SouthAsia','SouthAsia') = 724.8842163 ;
vmsb('c_Extraction','SouthAsia','NAmerica') = 270.0368347 ;
vmsb('c_Extraction','SouthAsia','LatinAmer') = 49.62901306 ;
vmsb('c_Extraction','SouthAsia','EU_28') = 1502.929199 ;
vmsb('c_Extraction','SouthAsia','MENA') = 779.0584717 ;
vmsb('c_Extraction','SouthAsia','SSA') = 68.51496887 ;
vmsb('c_Extraction','SouthAsia','RestofWorld') = 67.18429565 ;
vmsb('c_Extraction','NAmerica','Oceania') = 640.057251 ;
vmsb('c_Extraction','NAmerica','EastAsia') = 27941.8457 ;
vmsb('c_Extraction','NAmerica','SEAsia') = 751.4228516 ;
vmsb('c_Extraction','NAmerica','SouthAsia') = 2566.099121 ;
vmsb('c_Extraction','NAmerica','NAmerica') = 104896.8438 ;
vmsb('c_Extraction','NAmerica','LatinAmer') = 4479.817383 ;
vmsb('c_Extraction','NAmerica','EU_28') = 18887.2793 ;
vmsb('c_Extraction','NAmerica','MENA') = 3526.898926 ;
vmsb('c_Extraction','NAmerica','SSA') = 324.2833252 ;
vmsb('c_Extraction','NAmerica','RestofWorld') = 1884.96582 ;
vmsb('c_Extraction','LatinAmer','Oceania') = 398.7353516 ;
vmsb('c_Extraction','LatinAmer','EastAsia') = 92193.71094 ;
vmsb('c_Extraction','LatinAmer','SEAsia') = 2788.889404 ;
vmsb('c_Extraction','LatinAmer','SouthAsia') = 12554.40918 ;
vmsb('c_Extraction','LatinAmer','NAmerica') = 61269.99219 ;
vmsb('c_Extraction','LatinAmer','LatinAmer') = 26304.67969 ;
vmsb('c_Extraction','LatinAmer','EU_28') = 36290.33594 ;
vmsb('c_Extraction','LatinAmer','MENA') = 6017.628906 ;
vmsb('c_Extraction','LatinAmer','SSA') = 506.7384949 ;
vmsb('c_Extraction','LatinAmer','RestofWorld') = 6381.804688 ;
vmsb('c_Extraction','EU_28','Oceania') = 145.1050873 ;
vmsb('c_Extraction','EU_28','EastAsia') = 4186.612305 ;
vmsb('c_Extraction','EU_28','SEAsia') = 568.1520386 ;
vmsb('c_Extraction','EU_28','SouthAsia') = 10388.62305 ;
vmsb('c_Extraction','EU_28','NAmerica') = 2821.113281 ;
vmsb('c_Extraction','EU_28','LatinAmer') = 922.0996704 ;
vmsb('c_Extraction','EU_28','EU_28') = 60958.29297 ;
vmsb('c_Extraction','EU_28','MENA') = 6034.556152 ;
vmsb('c_Extraction','EU_28','SSA') = 2034.265869 ;
vmsb('c_Extraction','EU_28','RestofWorld') = 3066.421143 ;
vmsb('c_Extraction','MENA','Oceania') = 3167.452637 ;
vmsb('c_Extraction','MENA','EastAsia') = 359195.625 ;
vmsb('c_Extraction','MENA','SEAsia') = 75155.85156 ;
vmsb('c_Extraction','MENA','SouthAsia') = 116056.9531 ;
vmsb('c_Extraction','MENA','NAmerica') = 109005.9375 ;
vmsb('c_Extraction','MENA','LatinAmer') = 6016.584961 ;
vmsb('c_Extraction','MENA','EU_28') = 149872.9219 ;
vmsb('c_Extraction','MENA','MENA') = 39379.91406 ;
vmsb('c_Extraction','MENA','SSA') = 13720.58691 ;
vmsb('c_Extraction','MENA','RestofWorld') = 2472.301514 ;
vmsb('c_Extraction','SSA','Oceania') = 2300.398438 ;
vmsb('c_Extraction','SSA','EastAsia') = 74835.46875 ;
vmsb('c_Extraction','SSA','SEAsia') = 3191.651123 ;
vmsb('c_Extraction','SSA','SouthAsia') = 24416.39062 ;
vmsb('c_Extraction','SSA','NAmerica') = 79704.39062 ;
vmsb('c_Extraction','SSA','LatinAmer') = 10735.89941 ;
vmsb('c_Extraction','SSA','EU_28') = 52551.65625 ;
vmsb('c_Extraction','SSA','MENA') = 1628.045166 ;
vmsb('c_Extraction','SSA','SSA') = 11109.2959 ;
vmsb('c_Extraction','SSA','RestofWorld') = 2066.834473 ;
vmsb('c_Extraction','RestofWorld','Oceania') = 1666.684448 ;
vmsb('c_Extraction','RestofWorld','EastAsia') = 87727.53906 ;
vmsb('c_Extraction','RestofWorld','SEAsia') = 6166.509766 ;
vmsb('c_Extraction','RestofWorld','SouthAsia') = 2250.718994 ;
vmsb('c_Extraction','RestofWorld','NAmerica') = 21755.12695 ;
vmsb('c_Extraction','RestofWorld','LatinAmer') = 2820.591553 ;
vmsb('c_Extraction','RestofWorld','EU_28') = 332837.5312 ;
vmsb('c_Extraction','RestofWorld','MENA') = 19192.00391 ;
vmsb('c_Extraction','RestofWorld','SSA') = 257.7037354 ;
vmsb('c_Extraction','RestofWorld','RestofWorld') = 48561.4375 ;
vmsb('c_ProcFood','Oceania','Oceania') = 4067.907715 ;
vmsb('c_ProcFood','Oceania','EastAsia') = 8345.256836 ;
vmsb('c_ProcFood','Oceania','SEAsia') = 5199.632324 ;
vmsb('c_ProcFood','Oceania','SouthAsia') = 974.3950806 ;
vmsb('c_ProcFood','Oceania','NAmerica') = 3325.232422 ;
vmsb('c_ProcFood','Oceania','LatinAmer') = 1078.232422 ;
vmsb('c_ProcFood','Oceania','EU_28') = 3235.895508 ;
vmsb('c_ProcFood','Oceania','MENA') = 2722.686035 ;
vmsb('c_ProcFood','Oceania','SSA') = 795.3453369 ;
vmsb('c_ProcFood','Oceania','RestofWorld') = 452.1080017 ;
vmsb('c_ProcFood','EastAsia','Oceania') = 1682.783813 ;
vmsb('c_ProcFood','EastAsia','EastAsia') = 21545.06836 ;
vmsb('c_ProcFood','EastAsia','SEAsia') = 8361.100586 ;
vmsb('c_ProcFood','EastAsia','SouthAsia') = 794.4866333 ;
vmsb('c_ProcFood','EastAsia','NAmerica') = 9941.642578 ;
vmsb('c_ProcFood','EastAsia','LatinAmer') = 1334.043335 ;
vmsb('c_ProcFood','EastAsia','EU_28') = 7152.464355 ;
vmsb('c_ProcFood','EastAsia','MENA') = 2716.697998 ;
vmsb('c_ProcFood','EastAsia','SSA') = 2047.182129 ;
vmsb('c_ProcFood','EastAsia','RestofWorld') = 2443.789062 ;
vmsb('c_ProcFood','SEAsia','Oceania') = 2764.014893 ;
vmsb('c_ProcFood','SEAsia','EastAsia') = 27221.48242 ;
vmsb('c_ProcFood','SEAsia','SEAsia') = 18176.99414 ;
vmsb('c_ProcFood','SEAsia','SouthAsia') = 19786.71875 ;
vmsb('c_ProcFood','SEAsia','NAmerica') = 13474.91309 ;
vmsb('c_ProcFood','SEAsia','LatinAmer') = 1617.13855 ;
vmsb('c_ProcFood','SEAsia','EU_28') = 10036.59766 ;
vmsb('c_ProcFood','SEAsia','MENA') = 8903.299805 ;
vmsb('c_ProcFood','SEAsia','SSA') = 5568.654297 ;
vmsb('c_ProcFood','SEAsia','RestofWorld') = 2964.601318 ;
vmsb('c_ProcFood','SouthAsia','Oceania') = 204.9653015 ;
vmsb('c_ProcFood','SouthAsia','EastAsia') = 3224.812988 ;
vmsb('c_ProcFood','SouthAsia','SEAsia') = 2709.9104 ;
vmsb('c_ProcFood','SouthAsia','SouthAsia') = 2428.628662 ;
vmsb('c_ProcFood','SouthAsia','NAmerica') = 2854.818115 ;
vmsb('c_ProcFood','SouthAsia','LatinAmer') = 219.088623 ;
vmsb('c_ProcFood','SouthAsia','EU_28') = 3099.67334 ;
vmsb('c_ProcFood','SouthAsia','MENA') = 2909.702637 ;
vmsb('c_ProcFood','SouthAsia','SSA') = 1396.176025 ;
vmsb('c_ProcFood','SouthAsia','RestofWorld') = 671.5852661 ;
vmsb('c_ProcFood','NAmerica','Oceania') = 1697.306396 ;
vmsb('c_ProcFood','NAmerica','EastAsia') = 17808.07617 ;
vmsb('c_ProcFood','NAmerica','SEAsia') = 4939.59375 ;
vmsb('c_ProcFood','NAmerica','SouthAsia') = 609.6289673 ;
vmsb('c_ProcFood','NAmerica','NAmerica') = 46634.98047 ;
vmsb('c_ProcFood','NAmerica','LatinAmer') = 9902.621094 ;
vmsb('c_ProcFood','NAmerica','EU_28') = 9586.558594 ;
vmsb('c_ProcFood','NAmerica','MENA') = 4842.362793 ;
vmsb('c_ProcFood','NAmerica','SSA') = 1227.143555 ;
vmsb('c_ProcFood','NAmerica','RestofWorld') = 1668.062378 ;
vmsb('c_ProcFood','LatinAmer','Oceania') = 714.583313 ;
vmsb('c_ProcFood','LatinAmer','EastAsia') = 12884.91016 ;
vmsb('c_ProcFood','LatinAmer','SEAsia') = 6854.956543 ;
vmsb('c_ProcFood','LatinAmer','SouthAsia') = 3137.690918 ;
vmsb('c_ProcFood','LatinAmer','NAmerica') = 14542.76367 ;
vmsb('c_ProcFood','LatinAmer','LatinAmer') = 22366.10547 ;
vmsb('c_ProcFood','LatinAmer','EU_28') = 20190.36328 ;
vmsb('c_ProcFood','LatinAmer','MENA') = 12601.3584 ;
vmsb('c_ProcFood','LatinAmer','SSA') = 10022.9082 ;
vmsb('c_ProcFood','LatinAmer','RestofWorld') = 3861.897461 ;
vmsb('c_ProcFood','EU_28','Oceania') = 3411.62207 ;
vmsb('c_ProcFood','EU_28','EastAsia') = 15866.37598 ;
vmsb('c_ProcFood','EU_28','SEAsia') = 5377.169434 ;
vmsb('c_ProcFood','EU_28','SouthAsia') = 1583.74585 ;
vmsb('c_ProcFood','EU_28','NAmerica') = 26552.625 ;
vmsb('c_ProcFood','EU_28','LatinAmer') = 5934.890137 ;
vmsb('c_ProcFood','EU_28','EU_28') = 233386.9844 ;
vmsb('c_ProcFood','EU_28','MENA') = 21323.0625 ;
vmsb('c_ProcFood','EU_28','SSA') = 9582.614258 ;
vmsb('c_ProcFood','EU_28','RestofWorld') = 27674.26367 ;
vmsb('c_ProcFood','MENA','Oceania') = 168.3130951 ;
vmsb('c_ProcFood','MENA','EastAsia') = 1139.983887 ;
vmsb('c_ProcFood','MENA','SEAsia') = 735.6297607 ;
vmsb('c_ProcFood','MENA','SouthAsia') = 774.2523804 ;
vmsb('c_ProcFood','MENA','NAmerica') = 1395.732788 ;
vmsb('c_ProcFood','MENA','LatinAmer') = 183.0933228 ;
vmsb('c_ProcFood','MENA','EU_28') = 6238.060547 ;
vmsb('c_ProcFood','MENA','MENA') = 15093.66992 ;
vmsb('c_ProcFood','MENA','SSA') = 3043.774414 ;
vmsb('c_ProcFood','MENA','RestofWorld') = 1166.355957 ;
vmsb('c_ProcFood','SSA','Oceania') = 212.4689178 ;
vmsb('c_ProcFood','SSA','EastAsia') = 1066.48938 ;
vmsb('c_ProcFood','SSA','SEAsia') = 273.9394836 ;
vmsb('c_ProcFood','SSA','SouthAsia') = 101.9823303 ;
vmsb('c_ProcFood','SSA','NAmerica') = 1071.671875 ;
vmsb('c_ProcFood','SSA','LatinAmer') = 112.8903961 ;
vmsb('c_ProcFood','SSA','EU_28') = 6853.455078 ;
vmsb('c_ProcFood','SSA','MENA') = 732.5784912 ;
vmsb('c_ProcFood','SSA','SSA') = 6266.066895 ;
vmsb('c_ProcFood','SSA','RestofWorld') = 449.4279785 ;
vmsb('c_ProcFood','RestofWorld','Oceania') = 357.5369873 ;
vmsb('c_ProcFood','RestofWorld','EastAsia') = 6246.453125 ;
vmsb('c_ProcFood','RestofWorld','SEAsia') = 534.3963013 ;
vmsb('c_ProcFood','RestofWorld','SouthAsia') = 1135.246338 ;
vmsb('c_ProcFood','RestofWorld','NAmerica') = 2676.960205 ;
vmsb('c_ProcFood','RestofWorld','LatinAmer') = 653.7127686 ;
vmsb('c_ProcFood','RestofWorld','EU_28') = 14262.99219 ;
vmsb('c_ProcFood','RestofWorld','MENA') = 5566.520508 ;
vmsb('c_ProcFood','RestofWorld','SSA') = 1108.199219 ;
vmsb('c_ProcFood','RestofWorld','RestofWorld') = 10662.5918 ;
vmsb('c_TextWapp','Oceania','Oceania') = 671.036438 ;
vmsb('c_TextWapp','Oceania','EastAsia') = 513.7498169 ;
vmsb('c_TextWapp','Oceania','SEAsia') = 170.1005554 ;
vmsb('c_TextWapp','Oceania','SouthAsia') = 161.2661285 ;
vmsb('c_TextWapp','Oceania','NAmerica') = 168.0735321 ;
vmsb('c_TextWapp','Oceania','LatinAmer') = 16.31082535 ;
vmsb('c_TextWapp','Oceania','EU_28') = 350.6326599 ;
vmsb('c_TextWapp','Oceania','MENA') = 74.21059418 ;
vmsb('c_TextWapp','Oceania','SSA') = 41.17987442 ;
vmsb('c_TextWapp','Oceania','RestofWorld') = 19.11988831 ;
vmsb('c_TextWapp','EastAsia','Oceania') = 6965.694336 ;
vmsb('c_TextWapp','EastAsia','EastAsia') = 78242.95312 ;
vmsb('c_TextWapp','EastAsia','SEAsia') = 32781.72266 ;
vmsb('c_TextWapp','EastAsia','SouthAsia') = 14844.56543 ;
vmsb('c_TextWapp','EastAsia','NAmerica') = 69537.32812 ;
vmsb('c_TextWapp','EastAsia','LatinAmer') = 19172.55664 ;
vmsb('c_TextWapp','EastAsia','EU_28') = 75015.29688 ;
vmsb('c_TextWapp','EastAsia','MENA') = 27688.3418 ;
vmsb('c_TextWapp','EastAsia','SSA') = 12671.50684 ;
vmsb('c_TextWapp','EastAsia','RestofWorld') = 21130.43164 ;
vmsb('c_TextWapp','SEAsia','Oceania') = 652.1759033 ;
vmsb('c_TextWapp','SEAsia','EastAsia') = 11382.20996 ;
vmsb('c_TextWapp','SEAsia','SEAsia') = 4623.087402 ;
vmsb('c_TextWapp','SEAsia','SouthAsia') = 1790.883911 ;
vmsb('c_TextWapp','SEAsia','NAmerica') = 25740.5918 ;
vmsb('c_TextWapp','SEAsia','LatinAmer') = 1753.775757 ;
vmsb('c_TextWapp','SEAsia','EU_28') = 12008.06152 ;
vmsb('c_TextWapp','SEAsia','MENA') = 3893.950684 ;
vmsb('c_TextWapp','SEAsia','SSA') = 743.8192749 ;
vmsb('c_TextWapp','SEAsia','RestofWorld') = 943.3265991 ;
vmsb('c_TextWapp','SouthAsia','Oceania') = 990.5932617 ;
vmsb('c_TextWapp','SouthAsia','EastAsia') = 5077.686035 ;
vmsb('c_TextWapp','SouthAsia','SEAsia') = 1397.599121 ;
vmsb('c_TextWapp','SouthAsia','SouthAsia') = 3308.0354 ;
vmsb('c_TextWapp','SouthAsia','NAmerica') = 22552.23047 ;
vmsb('c_TextWapp','SouthAsia','LatinAmer') = 2816.437988 ;
vmsb('c_TextWapp','SouthAsia','EU_28') = 35936.17188 ;
vmsb('c_TextWapp','SouthAsia','MENA') = 8832.011719 ;
vmsb('c_TextWapp','SouthAsia','SSA') = 2622.049316 ;
vmsb('c_TextWapp','SouthAsia','RestofWorld') = 2065.861084 ;
vmsb('c_TextWapp','NAmerica','Oceania') = 432.0802612 ;
vmsb('c_TextWapp','NAmerica','EastAsia') = 3335.882568 ;
vmsb('c_TextWapp','NAmerica','SEAsia') = 656.770874 ;
vmsb('c_TextWapp','NAmerica','SouthAsia') = 356.7446899 ;
vmsb('c_TextWapp','NAmerica','NAmerica') = 16156.625 ;
vmsb('c_TextWapp','NAmerica','LatinAmer') = 6101.946777 ;
vmsb('c_TextWapp','NAmerica','EU_28') = 3180.502441 ;
vmsb('c_TextWapp','NAmerica','MENA') = 962.0218506 ;
vmsb('c_TextWapp','NAmerica','SSA') = 310.9887695 ;
vmsb('c_TextWapp','NAmerica','RestofWorld') = 484.9412231 ;
vmsb('c_TextWapp','LatinAmer','Oceania') = 78.60494232 ;
vmsb('c_TextWapp','LatinAmer','EastAsia') = 517.2318726 ;
vmsb('c_TextWapp','LatinAmer','SEAsia') = 111.4939499 ;
vmsb('c_TextWapp','LatinAmer','SouthAsia') = 63.14333725 ;
vmsb('c_TextWapp','LatinAmer','NAmerica') = 12174.81836 ;
vmsb('c_TextWapp','LatinAmer','LatinAmer') = 5922.296387 ;
vmsb('c_TextWapp','LatinAmer','EU_28') = 1415.959229 ;
vmsb('c_TextWapp','LatinAmer','MENA') = 230.7182159 ;
vmsb('c_TextWapp','LatinAmer','SSA') = 138.4530945 ;
vmsb('c_TextWapp','LatinAmer','RestofWorld') = 115.6991577 ;
vmsb('c_TextWapp','EU_28','Oceania') = 913.3493652 ;
vmsb('c_TextWapp','EU_28','EastAsia') = 9299.636719 ;
vmsb('c_TextWapp','EU_28','SEAsia') = 1702.824341 ;
vmsb('c_TextWapp','EU_28','SouthAsia') = 1152.952637 ;
vmsb('c_TextWapp','EU_28','NAmerica') = 8926.311523 ;
vmsb('c_TextWapp','EU_28','LatinAmer') = 2156.564697 ;
vmsb('c_TextWapp','EU_28','EU_28') = 116778.4219 ;
vmsb('c_TextWapp','EU_28','MENA') = 12445.09473 ;
vmsb('c_TextWapp','EU_28','SSA') = 1804.842041 ;
vmsb('c_TextWapp','EU_28','RestofWorld') = 17906.64844 ;
vmsb('c_TextWapp','MENA','Oceania') = 203.0790405 ;
vmsb('c_TextWapp','MENA','EastAsia') = 1232.275391 ;
vmsb('c_TextWapp','MENA','SEAsia') = 305.1011047 ;
vmsb('c_TextWapp','MENA','SouthAsia') = 464.7098999 ;
vmsb('c_TextWapp','MENA','NAmerica') = 6218.725586 ;
vmsb('c_TextWapp','MENA','LatinAmer') = 530.9523315 ;
vmsb('c_TextWapp','MENA','EU_28') = 29855.72656 ;
vmsb('c_TextWapp','MENA','MENA') = 7268.657715 ;
vmsb('c_TextWapp','MENA','SSA') = 814.8115845 ;
vmsb('c_TextWapp','MENA','RestofWorld') = 4114.762207 ;
vmsb('c_TextWapp','SSA','Oceania') = 60.20415115 ;
vmsb('c_TextWapp','SSA','EastAsia') = 260.3434753 ;
vmsb('c_TextWapp','SSA','SEAsia') = 78.63881683 ;
vmsb('c_TextWapp','SSA','SouthAsia') = 45.23649216 ;
vmsb('c_TextWapp','SSA','NAmerica') = 1300.791992 ;
vmsb('c_TextWapp','SSA','LatinAmer') = 58.69341278 ;
vmsb('c_TextWapp','SSA','EU_28') = 1505.933472 ;
vmsb('c_TextWapp','SSA','MENA') = 139.2583771 ;
vmsb('c_TextWapp','SSA','SSA') = 1221.132935 ;
vmsb('c_TextWapp','SSA','RestofWorld') = 82.75138855 ;
vmsb('c_TextWapp','RestofWorld','Oceania') = 31.9338913 ;
vmsb('c_TextWapp','RestofWorld','EastAsia') = 502.1434326 ;
vmsb('c_TextWapp','RestofWorld','SEAsia') = 100.6437149 ;
vmsb('c_TextWapp','RestofWorld','SouthAsia') = 90.41997528 ;
vmsb('c_TextWapp','RestofWorld','NAmerica') = 445.3875427 ;
vmsb('c_TextWapp','RestofWorld','LatinAmer') = 58.55459976 ;
vmsb('c_TextWapp','RestofWorld','EU_28') = 5544.752441 ;
vmsb('c_TextWapp','RestofWorld','MENA') = 844.6942749 ;
vmsb('c_TextWapp','RestofWorld','SSA') = 75.87306213 ;
vmsb('c_TextWapp','RestofWorld','RestofWorld') = 2047.102051 ;
vmsb('c_LightMnfc','Oceania','Oceania') = 4039.297119 ;
vmsb('c_LightMnfc','Oceania','EastAsia') = 4391.290039 ;
vmsb('c_LightMnfc','Oceania','SEAsia') = 1911.926636 ;
vmsb('c_LightMnfc','Oceania','SouthAsia') = 342.7909546 ;
vmsb('c_LightMnfc','Oceania','NAmerica') = 2649.625488 ;
vmsb('c_LightMnfc','Oceania','LatinAmer') = 317.48526 ;
vmsb('c_LightMnfc','Oceania','EU_28') = 2107.155762 ;
vmsb('c_LightMnfc','Oceania','MENA') = 1621.012573 ;
vmsb('c_LightMnfc','Oceania','SSA') = 569.2571411 ;
vmsb('c_LightMnfc','Oceania','RestofWorld') = 468.107605 ;
vmsb('c_LightMnfc','EastAsia','Oceania') = 41151.39062 ;
vmsb('c_LightMnfc','EastAsia','EastAsia') = 124048.7969 ;
vmsb('c_LightMnfc','EastAsia','SEAsia') = 63698.98828 ;
vmsb('c_LightMnfc','EastAsia','SouthAsia') = 22225.50195 ;
vmsb('c_LightMnfc','EastAsia','NAmerica') = 238209.9375 ;
vmsb('c_LightMnfc','EastAsia','LatinAmer') = 56533.3125 ;
vmsb('c_LightMnfc','EastAsia','EU_28') = 165317.3906 ;
vmsb('c_LightMnfc','EastAsia','MENA') = 68589.14844 ;
vmsb('c_LightMnfc','EastAsia','SSA') = 40872.03125 ;
vmsb('c_LightMnfc','EastAsia','RestofWorld') = 52780.39062 ;
vmsb('c_LightMnfc','SEAsia','Oceania') = 8820.294922 ;
vmsb('c_LightMnfc','SEAsia','EastAsia') = 27816.7168 ;
vmsb('c_LightMnfc','SEAsia','SEAsia') = 27368.1875 ;
vmsb('c_LightMnfc','SEAsia','SouthAsia') = 5479.595703 ;
vmsb('c_LightMnfc','SEAsia','NAmerica') = 20517.68945 ;
vmsb('c_LightMnfc','SEAsia','LatinAmer') = 5320.265137 ;
vmsb('c_LightMnfc','SEAsia','EU_28') = 24451.03125 ;
vmsb('c_LightMnfc','SEAsia','MENA') = 11254.46094 ;
vmsb('c_LightMnfc','SEAsia','SSA') = 2965.977539 ;
vmsb('c_LightMnfc','SEAsia','RestofWorld') = 3738.669434 ;
vmsb('c_LightMnfc','SouthAsia','Oceania') = 968.0926514 ;
vmsb('c_LightMnfc','SouthAsia','EastAsia') = 3917.046143 ;
vmsb('c_LightMnfc','SouthAsia','SEAsia') = 3483.65332 ;
vmsb('c_LightMnfc','SouthAsia','SouthAsia') = 2638.321777 ;
vmsb('c_LightMnfc','SouthAsia','NAmerica') = 13520.66016 ;
vmsb('c_LightMnfc','SouthAsia','LatinAmer') = 1791.958374 ;
vmsb('c_LightMnfc','SouthAsia','EU_28') = 15461.13477 ;
vmsb('c_LightMnfc','SouthAsia','MENA') = 26484.70508 ;
vmsb('c_LightMnfc','SouthAsia','SSA') = 3982.962402 ;
vmsb('c_LightMnfc','SouthAsia','RestofWorld') = 1254.364136 ;
vmsb('c_LightMnfc','NAmerica','Oceania') = 8929.230469 ;
vmsb('c_LightMnfc','NAmerica','EastAsia') = 65037.35156 ;
vmsb('c_LightMnfc','NAmerica','SEAsia') = 17161.10352 ;
vmsb('c_LightMnfc','NAmerica','SouthAsia') = 8334.694336 ;
vmsb('c_LightMnfc','NAmerica','NAmerica') = 286929.2188 ;
vmsb('c_LightMnfc','NAmerica','LatinAmer') = 35885.07422 ;
vmsb('c_LightMnfc','NAmerica','EU_28') = 83467.85938 ;
vmsb('c_LightMnfc','NAmerica','MENA') = 28225.21484 ;
vmsb('c_LightMnfc','NAmerica','SSA') = 9419.735352 ;
vmsb('c_LightMnfc','NAmerica','RestofWorld') = 10814.49707 ;
vmsb('c_LightMnfc','LatinAmer','Oceania') = 748.1294556 ;
vmsb('c_LightMnfc','LatinAmer','EastAsia') = 9452.493164 ;
vmsb('c_LightMnfc','LatinAmer','SEAsia') = 1447.669556 ;
vmsb('c_LightMnfc','LatinAmer','SouthAsia') = 428.7193298 ;
vmsb('c_LightMnfc','LatinAmer','NAmerica') = 13757.19727 ;
vmsb('c_LightMnfc','LatinAmer','LatinAmer') = 36479.61328 ;
vmsb('c_LightMnfc','LatinAmer','EU_28') = 14207.19531 ;
vmsb('c_LightMnfc','LatinAmer','MENA') = 2006.567017 ;
vmsb('c_LightMnfc','LatinAmer','SSA') = 2725.08252 ;
vmsb('c_LightMnfc','LatinAmer','RestofWorld') = 1437.005005 ;
vmsb('c_LightMnfc','EU_28','Oceania') = 18799.00781 ;
vmsb('c_LightMnfc','EU_28','EastAsia') = 111630.7109 ;
vmsb('c_LightMnfc','EU_28','SEAsia') = 23126.11719 ;
vmsb('c_LightMnfc','EU_28','SouthAsia') = 13101.95215 ;
vmsb('c_LightMnfc','EU_28','NAmerica') = 120984.5234 ;
vmsb('c_LightMnfc','EU_28','LatinAmer') = 33438.82812 ;
vmsb('c_LightMnfc','EU_28','EU_28') = 898427.1875 ;
vmsb('c_LightMnfc','EU_28','MENA') = 90670.875 ;
vmsb('c_LightMnfc','EU_28','SSA') = 26210.6875 ;
vmsb('c_LightMnfc','EU_28','RestofWorld') = 131870.2969 ;
vmsb('c_LightMnfc','MENA','Oceania') = 537.6720581 ;
vmsb('c_LightMnfc','MENA','EastAsia') = 2732.875732 ;
vmsb('c_LightMnfc','MENA','SEAsia') = 902.2540894 ;
vmsb('c_LightMnfc','MENA','SouthAsia') = 5945.648438 ;
vmsb('c_LightMnfc','MENA','NAmerica') = 13819.8916 ;
vmsb('c_LightMnfc','MENA','LatinAmer') = 1211.109863 ;
vmsb('c_LightMnfc','MENA','EU_28') = 29902.87891 ;
vmsb('c_LightMnfc','MENA','MENA') = 19516.14844 ;
vmsb('c_LightMnfc','MENA','SSA') = 3629.936035 ;
vmsb('c_LightMnfc','MENA','RestofWorld') = 7637.221191 ;
vmsb('c_LightMnfc','SSA','Oceania') = 477.5726318 ;
vmsb('c_LightMnfc','SSA','EastAsia') = 2371.616211 ;
vmsb('c_LightMnfc','SSA','SEAsia') = 907.7848511 ;
vmsb('c_LightMnfc','SSA','SouthAsia') = 359.183136 ;
vmsb('c_LightMnfc','SSA','NAmerica') = 4731.229004 ;
vmsb('c_LightMnfc','SSA','LatinAmer') = 395.1300659 ;
vmsb('c_LightMnfc','SSA','EU_28') = 7803.678223 ;
vmsb('c_LightMnfc','SSA','MENA') = 925.3195801 ;
vmsb('c_LightMnfc','SSA','SSA') = 8270.766602 ;
vmsb('c_LightMnfc','SSA','RestofWorld') = 421.25 ;
vmsb('c_LightMnfc','RestofWorld','Oceania') = 394.6512756 ;
vmsb('c_LightMnfc','RestofWorld','EastAsia') = 6205.466797 ;
vmsb('c_LightMnfc','RestofWorld','SEAsia') = 1422.019287 ;
vmsb('c_LightMnfc','RestofWorld','SouthAsia') = 2564.25415 ;
vmsb('c_LightMnfc','RestofWorld','NAmerica') = 5862.05957 ;
vmsb('c_LightMnfc','RestofWorld','LatinAmer') = 1681.76709 ;
vmsb('c_LightMnfc','RestofWorld','EU_28') = 33743.61328 ;
vmsb('c_LightMnfc','RestofWorld','MENA') = 6301.329102 ;
vmsb('c_LightMnfc','RestofWorld','SSA') = 768.1334839 ;
vmsb('c_LightMnfc','RestofWorld','RestofWorld') = 18049.89258 ;
vmsb('c_HeavyMnfc','Oceania','Oceania') = 11522.98633 ;
vmsb('c_HeavyMnfc','Oceania','EastAsia') = 21858.15039 ;
vmsb('c_HeavyMnfc','Oceania','SEAsia') = 12060.75098 ;
vmsb('c_HeavyMnfc','Oceania','SouthAsia') = 5249.358398 ;
vmsb('c_HeavyMnfc','Oceania','NAmerica') = 7162.583496 ;
vmsb('c_HeavyMnfc','Oceania','LatinAmer') = 1236.59375 ;
vmsb('c_HeavyMnfc','Oceania','EU_28') = 6533.375 ;
vmsb('c_HeavyMnfc','Oceania','MENA') = 1733.192505 ;
vmsb('c_HeavyMnfc','Oceania','SSA') = 1900.469849 ;
vmsb('c_HeavyMnfc','Oceania','RestofWorld') = 994.1199951 ;
vmsb('c_HeavyMnfc','EastAsia','Oceania') = 50672.57031 ;
vmsb('c_HeavyMnfc','EastAsia','EastAsia') = 875497.8125 ;
vmsb('c_HeavyMnfc','EastAsia','SEAsia') = 292688.2188 ;
vmsb('c_HeavyMnfc','EastAsia','SouthAsia') = 90241.0625 ;
vmsb('c_HeavyMnfc','EastAsia','NAmerica') = 492825.2188 ;
vmsb('c_HeavyMnfc','EastAsia','LatinAmer') = 95245.14844 ;
vmsb('c_HeavyMnfc','EastAsia','EU_28') = 371462.6562 ;
vmsb('c_HeavyMnfc','EastAsia','MENA') = 116425.4453 ;
vmsb('c_HeavyMnfc','EastAsia','SSA') = 41484.25391 ;
vmsb('c_HeavyMnfc','EastAsia','RestofWorld') = 69453.14844 ;
vmsb('c_HeavyMnfc','SEAsia','Oceania') = 24730.1875 ;
vmsb('c_HeavyMnfc','SEAsia','EastAsia') = 215953.2344 ;
vmsb('c_HeavyMnfc','SEAsia','SEAsia') = 167881.4531 ;
vmsb('c_HeavyMnfc','SEAsia','SouthAsia') = 27731.82227 ;
vmsb('c_HeavyMnfc','SEAsia','NAmerica') = 85829.96094 ;
vmsb('c_HeavyMnfc','SEAsia','LatinAmer') = 10546.72754 ;
vmsb('c_HeavyMnfc','SEAsia','EU_28') = 73534.21094 ;
vmsb('c_HeavyMnfc','SEAsia','MENA') = 20636.95898 ;
vmsb('c_HeavyMnfc','SEAsia','SSA') = 6864.777344 ;
vmsb('c_HeavyMnfc','SEAsia','RestofWorld') = 10227.81152 ;
vmsb('c_HeavyMnfc','SouthAsia','Oceania') = 1403.975342 ;
vmsb('c_HeavyMnfc','SouthAsia','EastAsia') = 18456.11914 ;
vmsb('c_HeavyMnfc','SouthAsia','SEAsia') = 12560.35449 ;
vmsb('c_HeavyMnfc','SouthAsia','SouthAsia') = 10812.90332 ;
vmsb('c_HeavyMnfc','SouthAsia','NAmerica') = 20548.2168 ;
vmsb('c_HeavyMnfc','SouthAsia','LatinAmer') = 10655.11133 ;
vmsb('c_HeavyMnfc','SouthAsia','EU_28') = 30692.58789 ;
vmsb('c_HeavyMnfc','SouthAsia','MENA') = 26923.68555 ;
vmsb('c_HeavyMnfc','SouthAsia','SSA') = 15286.46973 ;
vmsb('c_HeavyMnfc','SouthAsia','RestofWorld') = 3974.185303 ;
vmsb('c_HeavyMnfc','NAmerica','Oceania') = 26314.38477 ;
vmsb('c_HeavyMnfc','NAmerica','EastAsia') = 195664.8281 ;
vmsb('c_HeavyMnfc','NAmerica','SEAsia') = 58589.08594 ;
vmsb('c_HeavyMnfc','NAmerica','SouthAsia') = 20814.20703 ;
vmsb('c_HeavyMnfc','NAmerica','NAmerica') = 482985.6875 ;
vmsb('c_HeavyMnfc','NAmerica','LatinAmer') = 142416.7188 ;
vmsb('c_HeavyMnfc','NAmerica','EU_28') = 249522.5625 ;
vmsb('c_HeavyMnfc','NAmerica','MENA') = 56709.67969 ;
vmsb('c_HeavyMnfc','NAmerica','SSA') = 19886.24414 ;
vmsb('c_HeavyMnfc','NAmerica','RestofWorld') = 40160.82422 ;
vmsb('c_HeavyMnfc','LatinAmer','Oceania') = 1578.396118 ;
vmsb('c_HeavyMnfc','LatinAmer','EastAsia') = 36811.53125 ;
vmsb('c_HeavyMnfc','LatinAmer','SEAsia') = 5968.787109 ;
vmsb('c_HeavyMnfc','LatinAmer','SouthAsia') = 1386.341431 ;
vmsb('c_HeavyMnfc','LatinAmer','NAmerica') = 83204.90625 ;
vmsb('c_HeavyMnfc','LatinAmer','LatinAmer') = 73136.49219 ;
vmsb('c_HeavyMnfc','LatinAmer','EU_28') = 30706.44727 ;
vmsb('c_HeavyMnfc','LatinAmer','MENA') = 4131.098633 ;
vmsb('c_HeavyMnfc','LatinAmer','SSA') = 4728.347656 ;
vmsb('c_HeavyMnfc','LatinAmer','RestofWorld') = 11931.05762 ;
vmsb('c_HeavyMnfc','EU_28','Oceania') = 31064.89258 ;
vmsb('c_HeavyMnfc','EU_28','EastAsia') = 229613.6562 ;
vmsb('c_HeavyMnfc','EU_28','SEAsia') = 56943.33203 ;
vmsb('c_HeavyMnfc','EU_28','SouthAsia') = 46291.49609 ;
vmsb('c_HeavyMnfc','EU_28','NAmerica') = 286785.75 ;
vmsb('c_HeavyMnfc','EU_28','LatinAmer') = 77972.71875 ;
vmsb('c_HeavyMnfc','EU_28','EU_28') = 1747709 ;
vmsb('c_HeavyMnfc','EU_28','MENA') = 212495.9219 ;
vmsb('c_HeavyMnfc','EU_28','SSA') = 58824.37109 ;
vmsb('c_HeavyMnfc','EU_28','RestofWorld') = 276868.0625 ;
vmsb('c_HeavyMnfc','MENA','Oceania') = 3376.992676 ;
vmsb('c_HeavyMnfc','MENA','EastAsia') = 81202 ;
vmsb('c_HeavyMnfc','MENA','SEAsia') = 24953.29297 ;
vmsb('c_HeavyMnfc','MENA','SouthAsia') = 44049.39453 ;
vmsb('c_HeavyMnfc','MENA','NAmerica') = 29015.46289 ;
vmsb('c_HeavyMnfc','MENA','LatinAmer') = 9353.390625 ;
vmsb('c_HeavyMnfc','MENA','EU_28') = 91589.29688 ;
vmsb('c_HeavyMnfc','MENA','MENA') = 92239.80469 ;
vmsb('c_HeavyMnfc','MENA','SSA') = 19647.87695 ;
vmsb('c_HeavyMnfc','MENA','RestofWorld') = 15546.6709 ;
vmsb('c_HeavyMnfc','SSA','Oceania') = 575.4074097 ;
vmsb('c_HeavyMnfc','SSA','EastAsia') = 15850.59668 ;
vmsb('c_HeavyMnfc','SSA','SEAsia') = 3524.006836 ;
vmsb('c_HeavyMnfc','SSA','SouthAsia') = 9799.642578 ;
vmsb('c_HeavyMnfc','SSA','NAmerica') = 9532.496094 ;
vmsb('c_HeavyMnfc','SSA','LatinAmer') = 1295.252319 ;
vmsb('c_HeavyMnfc','SSA','EU_28') = 17396.83203 ;
vmsb('c_HeavyMnfc','SSA','MENA') = 5814.663574 ;
vmsb('c_HeavyMnfc','SSA','SSA') = 20408.8418 ;
vmsb('c_HeavyMnfc','SSA','RestofWorld') = 12348.99902 ;
vmsb('c_HeavyMnfc','RestofWorld','Oceania') = 3551.222656 ;
vmsb('c_HeavyMnfc','RestofWorld','EastAsia') = 61878.13672 ;
vmsb('c_HeavyMnfc','RestofWorld','SEAsia') = 22020.50195 ;
vmsb('c_HeavyMnfc','RestofWorld','SouthAsia') = 42431.54297 ;
vmsb('c_HeavyMnfc','RestofWorld','NAmerica') = 57693.33594 ;
vmsb('c_HeavyMnfc','RestofWorld','LatinAmer') = 15210.5459 ;
vmsb('c_HeavyMnfc','RestofWorld','EU_28') = 236746.4219 ;
vmsb('c_HeavyMnfc','RestofWorld','MENA') = 49897.60938 ;
vmsb('c_HeavyMnfc','RestofWorld','SSA') = 4874.290527 ;
vmsb('c_HeavyMnfc','RestofWorld','RestofWorld') = 54543.03516 ;
vmsb('c_Util_Cons','Oceania','Oceania') = 31.07725143 ;
vmsb('c_Util_Cons','Oceania','EastAsia') = 201.0098877 ;
vmsb('c_Util_Cons','Oceania','SEAsia') = 101.6341934 ;
vmsb('c_Util_Cons','Oceania','SouthAsia') = 11.61959457 ;
vmsb('c_Util_Cons','Oceania','NAmerica') = 79.43945312 ;
vmsb('c_Util_Cons','Oceania','LatinAmer') = 49.86610413 ;
vmsb('c_Util_Cons','Oceania','EU_28') = 383.8754883 ;
vmsb('c_Util_Cons','Oceania','MENA') = 127.1499786 ;
vmsb('c_Util_Cons','Oceania','SSA') = 23.77629089 ;
vmsb('c_Util_Cons','Oceania','RestofWorld') = 78.74898529 ;
vmsb('c_Util_Cons','EastAsia','Oceania') = 366.0947571 ;
vmsb('c_Util_Cons','EastAsia','EastAsia') = 6357.345215 ;
vmsb('c_Util_Cons','EastAsia','SEAsia') = 2105.017578 ;
vmsb('c_Util_Cons','EastAsia','SouthAsia') = 451.4074097 ;
vmsb('c_Util_Cons','EastAsia','NAmerica') = 2141.272949 ;
vmsb('c_Util_Cons','EastAsia','LatinAmer') = 486.0794067 ;
vmsb('c_Util_Cons','EastAsia','EU_28') = 11890.67676 ;
vmsb('c_Util_Cons','EastAsia','MENA') = 4096.105469 ;
vmsb('c_Util_Cons','EastAsia','SSA') = 1524.092529 ;
vmsb('c_Util_Cons','EastAsia','RestofWorld') = 4494.487305 ;
vmsb('c_Util_Cons','SEAsia','Oceania') = 97.43195343 ;
vmsb('c_Util_Cons','SEAsia','EastAsia') = 1373.67334 ;
vmsb('c_Util_Cons','SEAsia','SEAsia') = 382.9202271 ;
vmsb('c_Util_Cons','SEAsia','SouthAsia') = 107.3237152 ;
vmsb('c_Util_Cons','SEAsia','NAmerica') = 393.7480469 ;
vmsb('c_Util_Cons','SEAsia','LatinAmer') = 142.2795563 ;
vmsb('c_Util_Cons','SEAsia','EU_28') = 2745.166992 ;
vmsb('c_Util_Cons','SEAsia','MENA') = 812.9980469 ;
vmsb('c_Util_Cons','SEAsia','SSA') = 236.6805725 ;
vmsb('c_Util_Cons','SEAsia','RestofWorld') = 861.5371094 ;
vmsb('c_Util_Cons','SouthAsia','Oceania') = 19.71509743 ;
vmsb('c_Util_Cons','SouthAsia','EastAsia') = 237.1269989 ;
vmsb('c_Util_Cons','SouthAsia','SEAsia') = 93.98291016 ;
vmsb('c_Util_Cons','SouthAsia','SouthAsia') = 10.60282135 ;
vmsb('c_Util_Cons','SouthAsia','NAmerica') = 61.55842209 ;
vmsb('c_Util_Cons','SouthAsia','LatinAmer') = 52.26968765 ;
vmsb('c_Util_Cons','SouthAsia','EU_28') = 802.0194092 ;
vmsb('c_Util_Cons','SouthAsia','MENA') = 255.4254608 ;
vmsb('c_Util_Cons','SouthAsia','SSA') = 61.51707458 ;
vmsb('c_Util_Cons','SouthAsia','RestofWorld') = 171.6964722 ;
vmsb('c_Util_Cons','NAmerica','Oceania') = 255.6985321 ;
vmsb('c_Util_Cons','NAmerica','EastAsia') = 2842.396729 ;
vmsb('c_Util_Cons','NAmerica','SEAsia') = 829.7235107 ;
vmsb('c_Util_Cons','NAmerica','SouthAsia') = 149.27742 ;
vmsb('c_Util_Cons','NAmerica','NAmerica') = 3845.761475 ;
vmsb('c_Util_Cons','NAmerica','LatinAmer') = 357.4509888 ;
vmsb('c_Util_Cons','NAmerica','EU_28') = 3967.249756 ;
vmsb('c_Util_Cons','NAmerica','MENA') = 1951.361328 ;
vmsb('c_Util_Cons','NAmerica','SSA') = 511.8539124 ;
vmsb('c_Util_Cons','NAmerica','RestofWorld') = 1926.356934 ;
vmsb('c_Util_Cons','LatinAmer','Oceania') = 46.47725296 ;
vmsb('c_Util_Cons','LatinAmer','EastAsia') = 322.9883423 ;
vmsb('c_Util_Cons','LatinAmer','SEAsia') = 206.2914734 ;
vmsb('c_Util_Cons','LatinAmer','SouthAsia') = 54.46353149 ;
vmsb('c_Util_Cons','LatinAmer','NAmerica') = 209.4649963 ;
vmsb('c_Util_Cons','LatinAmer','LatinAmer') = 1090.684814 ;
vmsb('c_Util_Cons','LatinAmer','EU_28') = 1094.125122 ;
vmsb('c_Util_Cons','LatinAmer','MENA') = 422.6958618 ;
vmsb('c_Util_Cons','LatinAmer','SSA') = 186.4594421 ;
vmsb('c_Util_Cons','LatinAmer','RestofWorld') = 532.9012451 ;
vmsb('c_Util_Cons','EU_28','Oceania') = 525.5808716 ;
vmsb('c_Util_Cons','EU_28','EastAsia') = 7820.53125 ;
vmsb('c_Util_Cons','EU_28','SEAsia') = 3339.029053 ;
vmsb('c_Util_Cons','EU_28','SouthAsia') = 862.8706665 ;
vmsb('c_Util_Cons','EU_28','NAmerica') = 1946.275146 ;
vmsb('c_Util_Cons','EU_28','LatinAmer') = 2228.263184 ;
vmsb('c_Util_Cons','EU_28','EU_28') = 50810.94531 ;
vmsb('c_Util_Cons','EU_28','MENA') = 7784.186035 ;
vmsb('c_Util_Cons','EU_28','SSA') = 2911.449951 ;
vmsb('c_Util_Cons','EU_28','RestofWorld') = 14605.10449 ;
vmsb('c_Util_Cons','MENA','Oceania') = 315.7593079 ;
vmsb('c_Util_Cons','MENA','EastAsia') = 1619.883789 ;
vmsb('c_Util_Cons','MENA','SEAsia') = 778.6523438 ;
vmsb('c_Util_Cons','MENA','SouthAsia') = 196.4408264 ;
vmsb('c_Util_Cons','MENA','NAmerica') = 1185.114136 ;
vmsb('c_Util_Cons','MENA','LatinAmer') = 669.9084473 ;
vmsb('c_Util_Cons','MENA','EU_28') = 5639.129395 ;
vmsb('c_Util_Cons','MENA','MENA') = 2623.361084 ;
vmsb('c_Util_Cons','MENA','SSA') = 375.0581055 ;
vmsb('c_Util_Cons','MENA','RestofWorld') = 2366.308594 ;
vmsb('c_Util_Cons','SSA','Oceania') = 32.03158188 ;
vmsb('c_Util_Cons','SSA','EastAsia') = 254.3527832 ;
vmsb('c_Util_Cons','SSA','SEAsia') = 144.2133484 ;
vmsb('c_Util_Cons','SSA','SouthAsia') = 27.73508072 ;
vmsb('c_Util_Cons','SSA','NAmerica') = 159.2524719 ;
vmsb('c_Util_Cons','SSA','LatinAmer') = 124.1556854 ;
vmsb('c_Util_Cons','SSA','EU_28') = 928.7954712 ;
vmsb('c_Util_Cons','SSA','MENA') = 392.6086426 ;
vmsb('c_Util_Cons','SSA','SSA') = 1046.158813 ;
vmsb('c_Util_Cons','SSA','RestofWorld') = 252.3304749 ;
vmsb('c_Util_Cons','RestofWorld','Oceania') = 196.6998596 ;
vmsb('c_Util_Cons','RestofWorld','EastAsia') = 1459.068359 ;
vmsb('c_Util_Cons','RestofWorld','SEAsia') = 1037.573853 ;
vmsb('c_Util_Cons','RestofWorld','SouthAsia') = 165.8195496 ;
vmsb('c_Util_Cons','RestofWorld','NAmerica') = 818.8848267 ;
vmsb('c_Util_Cons','RestofWorld','LatinAmer') = 856.9828491 ;
vmsb('c_Util_Cons','RestofWorld','EU_28') = 12403.9707 ;
vmsb('c_Util_Cons','RestofWorld','MENA') = 3946.093018 ;
vmsb('c_Util_Cons','RestofWorld','SSA') = 488.2766418 ;
vmsb('c_Util_Cons','RestofWorld','RestofWorld') = 3622.735107 ;
vmsb('c_TransComm','Oceania','Oceania') = 2169.616455 ;
vmsb('c_TransComm','Oceania','EastAsia') = 6539.338867 ;
vmsb('c_TransComm','Oceania','SEAsia') = 1562.600342 ;
vmsb('c_TransComm','Oceania','SouthAsia') = 348.3964844 ;
vmsb('c_TransComm','Oceania','NAmerica') = 4447.299316 ;
vmsb('c_TransComm','Oceania','LatinAmer') = 751.6378174 ;
vmsb('c_TransComm','Oceania','EU_28') = 7174.994629 ;
vmsb('c_TransComm','Oceania','MENA') = 1683.146484 ;
vmsb('c_TransComm','Oceania','SSA') = 553.6177979 ;
vmsb('c_TransComm','Oceania','RestofWorld') = 714.2366333 ;
vmsb('c_TransComm','EastAsia','Oceania') = 4212.180664 ;
vmsb('c_TransComm','EastAsia','EastAsia') = 65740.33594 ;
vmsb('c_TransComm','EastAsia','SEAsia') = 15686.9043 ;
vmsb('c_TransComm','EastAsia','SouthAsia') = 3984.506836 ;
vmsb('c_TransComm','EastAsia','NAmerica') = 34000.04688 ;
vmsb('c_TransComm','EastAsia','LatinAmer') = 6217.623047 ;
vmsb('c_TransComm','EastAsia','EU_28') = 58985.67188 ;
vmsb('c_TransComm','EastAsia','MENA') = 11513.73438 ;
vmsb('c_TransComm','EastAsia','SSA') = 4201.472656 ;
vmsb('c_TransComm','EastAsia','RestofWorld') = 6782.009766 ;
vmsb('c_TransComm','SEAsia','Oceania') = 2004.836426 ;
vmsb('c_TransComm','SEAsia','EastAsia') = 16652.25781 ;
vmsb('c_TransComm','SEAsia','SEAsia') = 2442.834961 ;
vmsb('c_TransComm','SEAsia','SouthAsia') = 1960.946777 ;
vmsb('c_TransComm','SEAsia','NAmerica') = 10710.11719 ;
vmsb('c_TransComm','SEAsia','LatinAmer') = 2438.797363 ;
vmsb('c_TransComm','SEAsia','EU_28') = 30842.5957 ;
vmsb('c_TransComm','SEAsia','MENA') = 3997.199951 ;
vmsb('c_TransComm','SEAsia','SSA') = 1551.950195 ;
vmsb('c_TransComm','SEAsia','RestofWorld') = 3739.834961 ;
vmsb('c_TransComm','SouthAsia','Oceania') = 492.8401489 ;
vmsb('c_TransComm','SouthAsia','EastAsia') = 5351.861816 ;
vmsb('c_TransComm','SouthAsia','SEAsia') = 1405.473877 ;
vmsb('c_TransComm','SouthAsia','SouthAsia') = 138.3771057 ;
vmsb('c_TransComm','SouthAsia','NAmerica') = 4213.526367 ;
vmsb('c_TransComm','SouthAsia','LatinAmer') = 716.8010864 ;
vmsb('c_TransComm','SouthAsia','EU_28') = 7466.467285 ;
vmsb('c_TransComm','SouthAsia','MENA') = 1734.245361 ;
vmsb('c_TransComm','SouthAsia','SSA') = 441.8512878 ;
vmsb('c_TransComm','SouthAsia','RestofWorld') = 970.2805176 ;
vmsb('c_TransComm','NAmerica','Oceania') = 3273.124268 ;
vmsb('c_TransComm','NAmerica','EastAsia') = 27415.13086 ;
vmsb('c_TransComm','NAmerica','SEAsia') = 6990.8125 ;
vmsb('c_TransComm','NAmerica','SouthAsia') = 3207.255859 ;
vmsb('c_TransComm','NAmerica','NAmerica') = 21350.7207 ;
vmsb('c_TransComm','NAmerica','LatinAmer') = 4646.54834 ;
vmsb('c_TransComm','NAmerica','EU_28') = 55497.03906 ;
vmsb('c_TransComm','NAmerica','MENA') = 10289.19824 ;
vmsb('c_TransComm','NAmerica','SSA') = 2744.233398 ;
vmsb('c_TransComm','NAmerica','RestofWorld') = 4709.695801 ;
vmsb('c_TransComm','LatinAmer','Oceania') = 919.934021 ;
vmsb('c_TransComm','LatinAmer','EastAsia') = 7700.701172 ;
vmsb('c_TransComm','LatinAmer','SEAsia') = 2465.787598 ;
vmsb('c_TransComm','LatinAmer','SouthAsia') = 922.2738647 ;
vmsb('c_TransComm','LatinAmer','NAmerica') = 6838.367676 ;
vmsb('c_TransComm','LatinAmer','LatinAmer') = 1695.694336 ;
vmsb('c_TransComm','LatinAmer','EU_28') = 21363.47656 ;
vmsb('c_TransComm','LatinAmer','MENA') = 3560.369385 ;
vmsb('c_TransComm','LatinAmer','SSA') = 926.9926147 ;
vmsb('c_TransComm','LatinAmer','RestofWorld') = 3073.308105 ;
vmsb('c_TransComm','EU_28','Oceania') = 8274.443359 ;
vmsb('c_TransComm','EU_28','EastAsia') = 52822.07812 ;
vmsb('c_TransComm','EU_28','SEAsia') = 22939.01562 ;
vmsb('c_TransComm','EU_28','SouthAsia') = 10831.97461 ;
vmsb('c_TransComm','EU_28','NAmerica') = 60427.17969 ;
vmsb('c_TransComm','EU_28','LatinAmer') = 22014.25 ;
vmsb('c_TransComm','EU_28','EU_28') = 316962.3125 ;
vmsb('c_TransComm','EU_28','MENA') = 28538.63281 ;
vmsb('c_TransComm','EU_28','SSA') = 12248.56934 ;
vmsb('c_TransComm','EU_28','RestofWorld') = 40754.24609 ;
vmsb('c_TransComm','MENA','Oceania') = 1772.074219 ;
vmsb('c_TransComm','MENA','EastAsia') = 15178.52441 ;
vmsb('c_TransComm','MENA','SEAsia') = 4101.641602 ;
vmsb('c_TransComm','MENA','SouthAsia') = 1782.15271 ;
vmsb('c_TransComm','MENA','NAmerica') = 16937.83984 ;
vmsb('c_TransComm','MENA','LatinAmer') = 3612.134521 ;
vmsb('c_TransComm','MENA','EU_28') = 40371.64844 ;
vmsb('c_TransComm','MENA','MENA') = 4900.626953 ;
vmsb('c_TransComm','MENA','SSA') = 1857.097778 ;
vmsb('c_TransComm','MENA','RestofWorld') = 6201.303223 ;
vmsb('c_TransComm','SSA','Oceania') = 429.4367981 ;
vmsb('c_TransComm','SSA','EastAsia') = 3114.362305 ;
vmsb('c_TransComm','SSA','SEAsia') = 998.7744751 ;
vmsb('c_TransComm','SSA','SouthAsia') = 317.2618408 ;
vmsb('c_TransComm','SSA','NAmerica') = 3187.952148 ;
vmsb('c_TransComm','SSA','LatinAmer') = 675.1118164 ;
vmsb('c_TransComm','SSA','EU_28') = 8443.737305 ;
vmsb('c_TransComm','SSA','MENA') = 1192.515625 ;
vmsb('c_TransComm','SSA','SSA') = 354.2862854 ;
vmsb('c_TransComm','SSA','RestofWorld') = 896.7719116 ;
vmsb('c_TransComm','RestofWorld','Oceania') = 947.9411621 ;
vmsb('c_TransComm','RestofWorld','EastAsia') = 10903.88184 ;
vmsb('c_TransComm','RestofWorld','SEAsia') = 4127.525391 ;
vmsb('c_TransComm','RestofWorld','SouthAsia') = 1044.633789 ;
vmsb('c_TransComm','RestofWorld','NAmerica') = 8281.378906 ;
vmsb('c_TransComm','RestofWorld','LatinAmer') = 2161.365723 ;
vmsb('c_TransComm','RestofWorld','EU_28') = 36154.76172 ;
vmsb('c_TransComm','RestofWorld','MENA') = 4304.719238 ;
vmsb('c_TransComm','RestofWorld','SSA') = 1237.386597 ;
vmsb('c_TransComm','RestofWorld','RestofWorld') = 3337.639893 ;
vmsb('c_OthService','Oceania','Oceania') = 2283.932617 ;
vmsb('c_OthService','Oceania','EastAsia') = 6401.10791 ;
vmsb('c_OthService','Oceania','SEAsia') = 2127.986084 ;
vmsb('c_OthService','Oceania','SouthAsia') = 573.2736206 ;
vmsb('c_OthService','Oceania','NAmerica') = 5652.233887 ;
vmsb('c_OthService','Oceania','LatinAmer') = 786.1976318 ;
vmsb('c_OthService','Oceania','EU_28') = 7780.802734 ;
vmsb('c_OthService','Oceania','MENA') = 2293.664307 ;
vmsb('c_OthService','Oceania','SSA') = 659.3297729 ;
vmsb('c_OthService','Oceania','RestofWorld') = 841.8104858 ;
vmsb('c_OthService','EastAsia','Oceania') = 2661.219482 ;
vmsb('c_OthService','EastAsia','EastAsia') = 24510.78125 ;
vmsb('c_OthService','EastAsia','SEAsia') = 10720.59766 ;
vmsb('c_OthService','EastAsia','SouthAsia') = 5830.672363 ;
vmsb('c_OthService','EastAsia','NAmerica') = 40831.04688 ;
vmsb('c_OthService','EastAsia','LatinAmer') = 6029.530762 ;
vmsb('c_OthService','EastAsia','EU_28') = 51103.73047 ;
vmsb('c_OthService','EastAsia','MENA') = 13700.07715 ;
vmsb('c_OthService','EastAsia','SSA') = 3753.361328 ;
vmsb('c_OthService','EastAsia','RestofWorld') = 6419.271484 ;
vmsb('c_OthService','SEAsia','Oceania') = 2375.387695 ;
vmsb('c_OthService','SEAsia','EastAsia') = 12545.82324 ;
vmsb('c_OthService','SEAsia','SEAsia') = 2450.027588 ;
vmsb('c_OthService','SEAsia','SouthAsia') = 4212.79248 ;
vmsb('c_OthService','SEAsia','NAmerica') = 11423.6748 ;
vmsb('c_OthService','SEAsia','LatinAmer') = 2735.078369 ;
vmsb('c_OthService','SEAsia','EU_28') = 30411.67188 ;
vmsb('c_OthService','SEAsia','MENA') = 4138.499512 ;
vmsb('c_OthService','SEAsia','SSA') = 1649.08313 ;
vmsb('c_OthService','SEAsia','RestofWorld') = 3016.982666 ;
vmsb('c_OthService','SouthAsia','Oceania') = 925.135498 ;
vmsb('c_OthService','SouthAsia','EastAsia') = 7902.925781 ;
vmsb('c_OthService','SouthAsia','SEAsia') = 2953.47168 ;
vmsb('c_OthService','SouthAsia','SouthAsia') = 317.1229553 ;
vmsb('c_OthService','SouthAsia','NAmerica') = 23476.5957 ;
vmsb('c_OthService','SouthAsia','LatinAmer') = 1873.734741 ;
vmsb('c_OthService','SouthAsia','EU_28') = 25990.93555 ;
vmsb('c_OthService','SouthAsia','MENA') = 3121.552734 ;
vmsb('c_OthService','SouthAsia','SSA') = 1184.272705 ;
vmsb('c_OthService','SouthAsia','RestofWorld') = 2095.656494 ;
vmsb('c_OthService','NAmerica','Oceania') = 6903.589844 ;
vmsb('c_OthService','NAmerica','EastAsia') = 61267.43359 ;
vmsb('c_OthService','NAmerica','SEAsia') = 16068.8877 ;
vmsb('c_OthService','NAmerica','SouthAsia') = 11846.02148 ;
vmsb('c_OthService','NAmerica','NAmerica') = 46485.67578 ;
vmsb('c_OthService','NAmerica','LatinAmer') = 11753.08398 ;
vmsb('c_OthService','NAmerica','EU_28') = 149312.7031 ;
vmsb('c_OthService','NAmerica','MENA') = 35297.79297 ;
vmsb('c_OthService','NAmerica','SSA') = 10327.73535 ;
vmsb('c_OthService','NAmerica','RestofWorld') = 14985.01367 ;
vmsb('c_OthService','LatinAmer','Oceania') = 767.5485229 ;
vmsb('c_OthService','LatinAmer','EastAsia') = 6080.470215 ;
vmsb('c_OthService','LatinAmer','SEAsia') = 2984.134277 ;
vmsb('c_OthService','LatinAmer','SouthAsia') = 1425.15332 ;
vmsb('c_OthService','LatinAmer','NAmerica') = 10734.77051 ;
vmsb('c_OthService','LatinAmer','LatinAmer') = 1358.890869 ;
vmsb('c_OthService','LatinAmer','EU_28') = 18873.36719 ;
vmsb('c_OthService','LatinAmer','MENA') = 4633.399902 ;
vmsb('c_OthService','LatinAmer','SSA') = 1248.408447 ;
vmsb('c_OthService','LatinAmer','RestofWorld') = 2278.50415 ;
vmsb('c_OthService','EU_28','Oceania') = 8443.917969 ;
vmsb('c_OthService','EU_28','EastAsia') = 56998.12109 ;
vmsb('c_OthService','EU_28','SEAsia') = 31403.49414 ;
vmsb('c_OthService','EU_28','SouthAsia') = 20280.83594 ;
vmsb('c_OthService','EU_28','NAmerica') = 129155.0859 ;
vmsb('c_OthService','EU_28','LatinAmer') = 27664.25195 ;
vmsb('c_OthService','EU_28','EU_28') = 491871.9062 ;
vmsb('c_OthService','EU_28','MENA') = 47819.35156 ;
vmsb('c_OthService','EU_28','SSA') = 19573.19727 ;
vmsb('c_OthService','EU_28','RestofWorld') = 62218.03125 ;
vmsb('c_OthService','MENA','Oceania') = 1445.43689 ;
vmsb('c_OthService','MENA','EastAsia') = 12839.02148 ;
vmsb('c_OthService','MENA','SEAsia') = 3465.295898 ;
vmsb('c_OthService','MENA','SouthAsia') = 2367.249756 ;
vmsb('c_OthService','MENA','NAmerica') = 18135.54883 ;
vmsb('c_OthService','MENA','LatinAmer') = 3768.030029 ;
vmsb('c_OthService','MENA','EU_28') = 34945.88281 ;
vmsb('c_OthService','MENA','MENA') = 5371.029785 ;
vmsb('c_OthService','MENA','SSA') = 1977.580933 ;
vmsb('c_OthService','MENA','RestofWorld') = 5075.946289 ;
vmsb('c_OthService','SSA','Oceania') = 308.29599 ;
vmsb('c_OthService','SSA','EastAsia') = 2903.436035 ;
vmsb('c_OthService','SSA','SEAsia') = 790.6919556 ;
vmsb('c_OthService','SSA','SouthAsia') = 361.8479004 ;
vmsb('c_OthService','SSA','NAmerica') = 4691.35791 ;
vmsb('c_OthService','SSA','LatinAmer') = 644.6809692 ;
vmsb('c_OthService','SSA','EU_28') = 6700.551758 ;
vmsb('c_OthService','SSA','MENA') = 2125.437988 ;
vmsb('c_OthService','SSA','SSA') = 492.7622986 ;
vmsb('c_OthService','SSA','RestofWorld') = 848.0107422 ;
vmsb('c_OthService','RestofWorld','Oceania') = 842.0559692 ;
vmsb('c_OthService','RestofWorld','EastAsia') = 8420.21582 ;
vmsb('c_OthService','RestofWorld','SEAsia') = 5079.750488 ;
vmsb('c_OthService','RestofWorld','SouthAsia') = 2429.520752 ;
vmsb('c_OthService','RestofWorld','NAmerica') = 18953.99609 ;
vmsb('c_OthService','RestofWorld','LatinAmer') = 2311.362549 ;
vmsb('c_OthService','RestofWorld','EU_28') = 48936.79297 ;
vmsb('c_OthService','RestofWorld','MENA') = 6074.623535 ;
vmsb('c_OthService','RestofWorld','SSA') = 2068.710938 ;
vmsb('c_OthService','RestofWorld','RestofWorld') = 3546.81543 ;

* vtwr data (700 cells)
vtwr('c_TransComm','c_Crops','Oceania','Oceania') = 134.4190674 ;
vtwr('c_TransComm','c_Crops','Oceania','EastAsia') = 645.5335083 ;
vtwr('c_TransComm','c_Crops','Oceania','SEAsia') = 384.039093 ;
vtwr('c_TransComm','c_Crops','Oceania','SouthAsia') = 86.26319122 ;
vtwr('c_TransComm','c_Crops','Oceania','NAmerica') = 50.72628021 ;
vtwr('c_TransComm','c_Crops','Oceania','LatinAmer') = 3.795932531 ;
vtwr('c_TransComm','c_Crops','Oceania','EU_28') = 201.4256592 ;
vtwr('c_TransComm','c_Crops','Oceania','MENA') = 152.8369904 ;
vtwr('c_TransComm','c_Crops','Oceania','SSA') = 23.04143333 ;
vtwr('c_TransComm','c_Crops','Oceania','RestofWorld') = 9.179626465 ;
vtwr('c_TransComm','c_Crops','EastAsia','Oceania') = 12.56191254 ;
vtwr('c_TransComm','c_Crops','EastAsia','EastAsia') = 626.0491333 ;
vtwr('c_TransComm','c_Crops','EastAsia','SEAsia') = 854.7646484 ;
vtwr('c_TransComm','c_Crops','EastAsia','SouthAsia') = 122.253273 ;
vtwr('c_TransComm','c_Crops','EastAsia','NAmerica') = 137.2861633 ;
vtwr('c_TransComm','c_Crops','EastAsia','LatinAmer') = 51.73110962 ;
vtwr('c_TransComm','c_Crops','EastAsia','EU_28') = 162.6651611 ;
vtwr('c_TransComm','c_Crops','EastAsia','MENA') = 123.4973831 ;
vtwr('c_TransComm','c_Crops','EastAsia','SSA') = 62.88901138 ;
vtwr('c_TransComm','c_Crops','EastAsia','RestofWorld') = 235.8442078 ;
vtwr('c_TransComm','c_Crops','SEAsia','Oceania') = 60.73736572 ;
vtwr('c_TransComm','c_Crops','SEAsia','EastAsia') = 1520.088379 ;
vtwr('c_TransComm','c_Crops','SEAsia','SEAsia') = 824.8588867 ;
vtwr('c_TransComm','c_Crops','SEAsia','SouthAsia') = 178.4640656 ;
vtwr('c_TransComm','c_Crops','SEAsia','NAmerica') = 229.0344849 ;
vtwr('c_TransComm','c_Crops','SEAsia','LatinAmer') = 11.59331608 ;
vtwr('c_TransComm','c_Crops','SEAsia','EU_28') = 256.7603149 ;
vtwr('c_TransComm','c_Crops','SEAsia','MENA') = 249.9344177 ;
vtwr('c_TransComm','c_Crops','SEAsia','SSA') = 461.4612122 ;
vtwr('c_TransComm','c_Crops','SEAsia','RestofWorld') = 56.63115311 ;
vtwr('c_TransComm','c_Crops','SouthAsia','Oceania') = 18.0028801 ;
vtwr('c_TransComm','c_Crops','SouthAsia','EastAsia') = 235.8639832 ;
vtwr('c_TransComm','c_Crops','SouthAsia','SEAsia') = 347.5821838 ;
vtwr('c_TransComm','c_Crops','SouthAsia','SouthAsia') = 530.5617676 ;
vtwr('c_TransComm','c_Crops','SouthAsia','NAmerica') = 115.6518326 ;
vtwr('c_TransComm','c_Crops','SouthAsia','LatinAmer') = 14.95377731 ;
vtwr('c_TransComm','c_Crops','SouthAsia','EU_28') = 233.1908722 ;
vtwr('c_TransComm','c_Crops','SouthAsia','MENA') = 925.9355469 ;
vtwr('c_TransComm','c_Crops','SouthAsia','SSA') = 203.3773651 ;
vtwr('c_TransComm','c_Crops','SouthAsia','RestofWorld') = 135.2327881 ;
vtwr('c_TransComm','c_Crops','NAmerica','Oceania') = 64.22002411 ;
vtwr('c_TransComm','c_Crops','NAmerica','EastAsia') = 6132.930176 ;
vtwr('c_TransComm','c_Crops','NAmerica','SEAsia') = 780.6657104 ;
vtwr('c_TransComm','c_Crops','NAmerica','SouthAsia') = 255.6829224 ;
vtwr('c_TransComm','c_Crops','NAmerica','NAmerica') = 7019.450195 ;
vtwr('c_TransComm','c_Crops','NAmerica','LatinAmer') = 836.0023804 ;
vtwr('c_TransComm','c_Crops','NAmerica','EU_28') = 254.1903687 ;
vtwr('c_TransComm','c_Crops','NAmerica','MENA') = 1090.376099 ;
vtwr('c_TransComm','c_Crops','NAmerica','SSA') = 374.8605957 ;
vtwr('c_TransComm','c_Crops','NAmerica','RestofWorld') = 54.39509583 ;
vtwr('c_TransComm','c_Crops','LatinAmer','Oceania') = 22.20227432 ;
vtwr('c_TransComm','c_Crops','LatinAmer','EastAsia') = 4439.115234 ;
vtwr('c_TransComm','c_Crops','LatinAmer','SEAsia') = 345.0674133 ;
vtwr('c_TransComm','c_Crops','LatinAmer','SouthAsia') = 27.60004044 ;
vtwr('c_TransComm','c_Crops','LatinAmer','NAmerica') = 2536.69043 ;
vtwr('c_TransComm','c_Crops','LatinAmer','LatinAmer') = 1092.445068 ;
vtwr('c_TransComm','c_Crops','LatinAmer','EU_28') = 3125.881104 ;
vtwr('c_TransComm','c_Crops','LatinAmer','MENA') = 834.4991455 ;
vtwr('c_TransComm','c_Crops','LatinAmer','SSA') = 84.45435333 ;
vtwr('c_TransComm','c_Crops','LatinAmer','RestofWorld') = 582.3327026 ;
vtwr('c_TransComm','c_Crops','EU_28','Oceania') = 10.4661417 ;
vtwr('c_TransComm','c_Crops','EU_28','EastAsia') = 85.97655487 ;
vtwr('c_TransComm','c_Crops','EU_28','SEAsia') = 27.41782951 ;
vtwr('c_TransComm','c_Crops','EU_28','SouthAsia') = 25.33651733 ;
vtwr('c_TransComm','c_Crops','EU_28','NAmerica') = 220.7208252 ;
vtwr('c_TransComm','c_Crops','EU_28','LatinAmer') = 60.8678093 ;
vtwr('c_TransComm','c_Crops','EU_28','EU_28') = 3856.517578 ;
vtwr('c_TransComm','c_Crops','EU_28','MENA') = 778.8858643 ;
vtwr('c_TransComm','c_Crops','EU_28','SSA') = 182.5056763 ;
vtwr('c_TransComm','c_Crops','EU_28','RestofWorld') = 1040.882446 ;
vtwr('c_TransComm','c_Crops','MENA','Oceania') = 7.920248032 ;
vtwr('c_TransComm','c_Crops','MENA','EastAsia') = 56.55270004 ;
vtwr('c_TransComm','c_Crops','MENA','SEAsia') = 26.3938179 ;
vtwr('c_TransComm','c_Crops','MENA','SouthAsia') = 50.33245087 ;
vtwr('c_TransComm','c_Crops','MENA','NAmerica') = 90.69908142 ;
vtwr('c_TransComm','c_Crops','MENA','LatinAmer') = 9.293917656 ;
vtwr('c_TransComm','c_Crops','MENA','EU_28') = 1381.563232 ;
vtwr('c_TransComm','c_Crops','MENA','MENA') = 1518.11084 ;
vtwr('c_TransComm','c_Crops','MENA','SSA') = 69.18730164 ;
vtwr('c_TransComm','c_Crops','MENA','RestofWorld') = 934.3813477 ;
vtwr('c_TransComm','c_Crops','SSA','Oceania') = 4.659304619 ;
vtwr('c_TransComm','c_Crops','SSA','EastAsia') = 195.2805634 ;
vtwr('c_TransComm','c_Crops','SSA','SEAsia') = 148.3560944 ;
vtwr('c_TransComm','c_Crops','SSA','SouthAsia') = 135.0029907 ;
vtwr('c_TransComm','c_Crops','SSA','NAmerica') = 211.6087646 ;
vtwr('c_TransComm','c_Crops','SSA','LatinAmer') = 14.65880203 ;
vtwr('c_TransComm','c_Crops','SSA','EU_28') = 1180.602783 ;
vtwr('c_TransComm','c_Crops','SSA','MENA') = 208.7460022 ;
vtwr('c_TransComm','c_Crops','SSA','SSA') = 236.2695312 ;
vtwr('c_TransComm','c_Crops','SSA','RestofWorld') = 112.1658401 ;
vtwr('c_TransComm','c_Crops','RestofWorld','Oceania') = 0.1513227671 ;
vtwr('c_TransComm','c_Crops','RestofWorld','EastAsia') = 37.19985199 ;
vtwr('c_TransComm','c_Crops','RestofWorld','SEAsia') = 2.345897675 ;
vtwr('c_TransComm','c_Crops','RestofWorld','SouthAsia') = 22.30982018 ;
vtwr('c_TransComm','c_Crops','RestofWorld','NAmerica') = 5.584058762 ;
vtwr('c_TransComm','c_Crops','RestofWorld','LatinAmer') = 0.7495797873 ;
vtwr('c_TransComm','c_Crops','RestofWorld','EU_28') = 331.1040649 ;
vtwr('c_TransComm','c_Crops','RestofWorld','MENA') = 813.0882568 ;
vtwr('c_TransComm','c_Crops','RestofWorld','SSA') = 53.0161972 ;
vtwr('c_TransComm','c_Crops','RestofWorld','RestofWorld') = 600.6546631 ;
vtwr('c_TransComm','c_MeatLstk','Oceania','Oceania') = 36.50308228 ;
vtwr('c_TransComm','c_MeatLstk','Oceania','EastAsia') = 514.3244019 ;
vtwr('c_TransComm','c_MeatLstk','Oceania','SEAsia') = 115.8348618 ;
vtwr('c_TransComm','c_MeatLstk','Oceania','SouthAsia') = 18.94277763 ;
vtwr('c_TransComm','c_MeatLstk','Oceania','NAmerica') = 184.7874756 ;
vtwr('c_TransComm','c_MeatLstk','Oceania','LatinAmer') = 17.21829987 ;
vtwr('c_TransComm','c_MeatLstk','Oceania','EU_28') = 131.3022308 ;
vtwr('c_TransComm','c_MeatLstk','Oceania','MENA') = 84.39598846 ;
vtwr('c_TransComm','c_MeatLstk','Oceania','SSA') = 10.23093605 ;
vtwr('c_TransComm','c_MeatLstk','Oceania','RestofWorld') = 44.72723389 ;
vtwr('c_TransComm','c_MeatLstk','EastAsia','Oceania') = 1.24712944 ;
vtwr('c_TransComm','c_MeatLstk','EastAsia','EastAsia') = 237.247818 ;
vtwr('c_TransComm','c_MeatLstk','EastAsia','SEAsia') = 22.7072258 ;
vtwr('c_TransComm','c_MeatLstk','EastAsia','SouthAsia') = 2.562794447 ;
vtwr('c_TransComm','c_MeatLstk','EastAsia','NAmerica') = 34.06864929 ;
vtwr('c_TransComm','c_MeatLstk','EastAsia','LatinAmer') = 2.717411995 ;
vtwr('c_TransComm','c_MeatLstk','EastAsia','EU_28') = 60.84308624 ;
vtwr('c_TransComm','c_MeatLstk','EastAsia','MENA') = 7.673259258 ;
vtwr('c_TransComm','c_MeatLstk','EastAsia','SSA') = 3.142020464 ;
vtwr('c_TransComm','c_MeatLstk','EastAsia','RestofWorld') = 5.582521915 ;
vtwr('c_TransComm','c_MeatLstk','SEAsia','Oceania') = 0.6459020972 ;
vtwr('c_TransComm','c_MeatLstk','SEAsia','EastAsia') = 214.5027618 ;
vtwr('c_TransComm','c_MeatLstk','SEAsia','SEAsia') = 53.96387482 ;
vtwr('c_TransComm','c_MeatLstk','SEAsia','SouthAsia') = 1.639154673 ;
vtwr('c_TransComm','c_MeatLstk','SEAsia','NAmerica') = 13.63702488 ;
vtwr('c_TransComm','c_MeatLstk','SEAsia','LatinAmer') = 0.09980211407 ;
vtwr('c_TransComm','c_MeatLstk','SEAsia','EU_28') = 138.9879303 ;
vtwr('c_TransComm','c_MeatLstk','SEAsia','MENA') = 5.774458885 ;
vtwr('c_TransComm','c_MeatLstk','SEAsia','SSA') = 2.527950525 ;
vtwr('c_TransComm','c_MeatLstk','SEAsia','RestofWorld') = 1.620634079 ;
vtwr('c_TransComm','c_MeatLstk','SouthAsia','Oceania') = 0.1879956871 ;
vtwr('c_TransComm','c_MeatLstk','SouthAsia','EastAsia') = 4.747811317 ;
vtwr('c_TransComm','c_MeatLstk','SouthAsia','SEAsia') = 78.847229 ;
vtwr('c_TransComm','c_MeatLstk','SouthAsia','SouthAsia') = 6.742348194 ;
vtwr('c_TransComm','c_MeatLstk','SouthAsia','NAmerica') = 8.628139496 ;
vtwr('c_TransComm','c_MeatLstk','SouthAsia','LatinAmer') = 0.0980072394 ;
vtwr('c_TransComm','c_MeatLstk','SouthAsia','EU_28') = 5.996212959 ;
vtwr('c_TransComm','c_MeatLstk','SouthAsia','MENA') = 82.29367828 ;
vtwr('c_TransComm','c_MeatLstk','SouthAsia','SSA') = 11.94394112 ;
vtwr('c_TransComm','c_MeatLstk','SouthAsia','RestofWorld') = 6.123950005 ;
vtwr('c_TransComm','c_MeatLstk','NAmerica','Oceania') = 23.17802048 ;
vtwr('c_TransComm','c_MeatLstk','NAmerica','EastAsia') = 875.3105469 ;
vtwr('c_TransComm','c_MeatLstk','NAmerica','SEAsia') = 53.62145233 ;
vtwr('c_TransComm','c_MeatLstk','NAmerica','SouthAsia') = 8.449099541 ;
vtwr('c_TransComm','c_MeatLstk','NAmerica','NAmerica') = 775.4858398 ;
vtwr('c_TransComm','c_MeatLstk','NAmerica','LatinAmer') = 136.5470123 ;
vtwr('c_TransComm','c_MeatLstk','NAmerica','EU_28') = 22.4602375 ;
vtwr('c_TransComm','c_MeatLstk','NAmerica','MENA') = 97.44496155 ;
vtwr('c_TransComm','c_MeatLstk','NAmerica','SSA') = 34.27065659 ;
vtwr('c_TransComm','c_MeatLstk','NAmerica','RestofWorld') = 112.416069 ;
vtwr('c_TransComm','c_MeatLstk','LatinAmer','Oceania') = 2.531880856 ;
vtwr('c_TransComm','c_MeatLstk','LatinAmer','EastAsia') = 164.9031982 ;
vtwr('c_TransComm','c_MeatLstk','LatinAmer','SEAsia') = 23.95938683 ;
vtwr('c_TransComm','c_MeatLstk','LatinAmer','SouthAsia') = 2.979164839 ;
vtwr('c_TransComm','c_MeatLstk','LatinAmer','NAmerica') = 82.49649048 ;
vtwr('c_TransComm','c_MeatLstk','LatinAmer','LatinAmer') = 249.3171539 ;
vtwr('c_TransComm','c_MeatLstk','LatinAmer','EU_28') = 233.4077301 ;
vtwr('c_TransComm','c_MeatLstk','LatinAmer','MENA') = 273.0067139 ;
vtwr('c_TransComm','c_MeatLstk','LatinAmer','SSA') = 58.30953598 ;
vtwr('c_TransComm','c_MeatLstk','LatinAmer','RestofWorld') = 170.6059265 ;
vtwr('c_TransComm','c_MeatLstk','EU_28','Oceania') = 16.3035965 ;
vtwr('c_TransComm','c_MeatLstk','EU_28','EastAsia') = 284.0803528 ;
vtwr('c_TransComm','c_MeatLstk','EU_28','SEAsia') = 45.48047638 ;
vtwr('c_TransComm','c_MeatLstk','EU_28','SouthAsia') = 10.01698208 ;
vtwr('c_TransComm','c_MeatLstk','EU_28','NAmerica') = 42.8515358 ;
vtwr('c_TransComm','c_MeatLstk','EU_28','LatinAmer') = 18.83648109 ;
vtwr('c_TransComm','c_MeatLstk','EU_28','EU_28') = 1066.444092 ;
vtwr('c_TransComm','c_MeatLstk','EU_28','MENA') = 145.2308655 ;
vtwr('c_TransComm','c_MeatLstk','EU_28','SSA') = 70.38691711 ;
vtwr('c_TransComm','c_MeatLstk','EU_28','RestofWorld') = 298.7747498 ;
vtwr('c_TransComm','c_MeatLstk','MENA','Oceania') = 0.05661447719 ;
vtwr('c_TransComm','c_MeatLstk','MENA','EastAsia') = 2.210058451 ;
vtwr('c_TransComm','c_MeatLstk','MENA','SEAsia') = 1.644928932 ;
vtwr('c_TransComm','c_MeatLstk','MENA','SouthAsia') = 5.31014967 ;
vtwr('c_TransComm','c_MeatLstk','MENA','NAmerica') = 1.271152973 ;
vtwr('c_TransComm','c_MeatLstk','MENA','LatinAmer') = 0.07019328326 ;
vtwr('c_TransComm','c_MeatLstk','MENA','EU_28') = 22.36552238 ;
vtwr('c_TransComm','c_MeatLstk','MENA','MENA') = 113.6765671 ;
vtwr('c_TransComm','c_MeatLstk','MENA','SSA') = 5.499907494 ;
vtwr('c_TransComm','c_MeatLstk','MENA','RestofWorld') = 5.842210293 ;
vtwr('c_TransComm','c_MeatLstk','SSA','Oceania') = 0.08377165347 ;
vtwr('c_TransComm','c_MeatLstk','SSA','EastAsia') = 17.47891617 ;
vtwr('c_TransComm','c_MeatLstk','SSA','SEAsia') = 0.8103330135 ;
vtwr('c_TransComm','c_MeatLstk','SSA','SouthAsia') = 5.446962833 ;
vtwr('c_TransComm','c_MeatLstk','SSA','NAmerica') = 2.911120653 ;
vtwr('c_TransComm','c_MeatLstk','SSA','LatinAmer') = 0.1373200417 ;
vtwr('c_TransComm','c_MeatLstk','SSA','EU_28') = 13.20196342 ;
vtwr('c_TransComm','c_MeatLstk','SSA','MENA') = 55.64421463 ;
vtwr('c_TransComm','c_MeatLstk','SSA','SSA') = 35.46626282 ;
vtwr('c_TransComm','c_MeatLstk','SSA','RestofWorld') = 2.792983532 ;
vtwr('c_TransComm','c_MeatLstk','RestofWorld','Oceania') = 0.6784450412 ;
vtwr('c_TransComm','c_MeatLstk','RestofWorld','EastAsia') = 9.581500053 ;
vtwr('c_TransComm','c_MeatLstk','RestofWorld','SEAsia') = 1.524946451 ;
vtwr('c_TransComm','c_MeatLstk','RestofWorld','SouthAsia') = 0.4509959221 ;
vtwr('c_TransComm','c_MeatLstk','RestofWorld','NAmerica') = 4.040617466 ;
vtwr('c_TransComm','c_MeatLstk','RestofWorld','LatinAmer') = 0.1806704402 ;
vtwr('c_TransComm','c_MeatLstk','RestofWorld','EU_28') = 25.06927299 ;
vtwr('c_TransComm','c_MeatLstk','RestofWorld','MENA') = 9.559524536 ;
vtwr('c_TransComm','c_MeatLstk','RestofWorld','SSA') = 0.5648370385 ;
vtwr('c_TransComm','c_MeatLstk','RestofWorld','RestofWorld') = 32.71499252 ;
vtwr('c_TransComm','c_Extraction','Oceania','Oceania') = 94.13780212 ;
vtwr('c_TransComm','c_Extraction','Oceania','EastAsia') = 19153.07227 ;
vtwr('c_TransComm','c_Extraction','Oceania','SEAsia') = 786.9567871 ;
vtwr('c_TransComm','c_Extraction','Oceania','SouthAsia') = 3261.553467 ;
vtwr('c_TransComm','c_Extraction','Oceania','NAmerica') = 194.041275 ;
vtwr('c_TransComm','c_Extraction','Oceania','LatinAmer') = 132.937851 ;
vtwr('c_TransComm','c_Extraction','Oceania','EU_28') = 554.3853149 ;
vtwr('c_TransComm','c_Extraction','Oceania','MENA') = 77.08388519 ;
vtwr('c_TransComm','c_Extraction','Oceania','SSA') = 58.99191284 ;
vtwr('c_TransComm','c_Extraction','Oceania','RestofWorld') = 20.46026993 ;
vtwr('c_TransComm','c_Extraction','EastAsia','Oceania') = 87.16236877 ;
vtwr('c_TransComm','c_Extraction','EastAsia','EastAsia') = 3519.741211 ;
vtwr('c_TransComm','c_Extraction','EastAsia','SEAsia') = 217.9351349 ;
vtwr('c_TransComm','c_Extraction','EastAsia','SouthAsia') = 120.5086975 ;
vtwr('c_TransComm','c_Extraction','EastAsia','NAmerica') = 195.7141724 ;
vtwr('c_TransComm','c_Extraction','EastAsia','LatinAmer') = 18.8638649 ;
vtwr('c_TransComm','c_Extraction','EastAsia','EU_28') = 202.2797089 ;
vtwr('c_TransComm','c_Extraction','EastAsia','MENA') = 37.44936371 ;
vtwr('c_TransComm','c_Extraction','EastAsia','SSA') = 15.61895275 ;
vtwr('c_TransComm','c_Extraction','EastAsia','RestofWorld') = 30.24079132 ;
vtwr('c_TransComm','c_Extraction','SEAsia','Oceania') = 295.2310791 ;
vtwr('c_TransComm','c_Extraction','SEAsia','EastAsia') = 9816.766602 ;
vtwr('c_TransComm','c_Extraction','SEAsia','SEAsia') = 1654.489746 ;
vtwr('c_TransComm','c_Extraction','SEAsia','SouthAsia') = 1879.353516 ;
vtwr('c_TransComm','c_Extraction','SEAsia','NAmerica') = 160.0661926 ;
vtwr('c_TransComm','c_Extraction','SEAsia','LatinAmer') = 9.926026344 ;
vtwr('c_TransComm','c_Extraction','SEAsia','EU_28') = 697.2942505 ;
vtwr('c_TransComm','c_Extraction','SEAsia','MENA') = 92.41790771 ;
vtwr('c_TransComm','c_Extraction','SEAsia','SSA') = 5.599158287 ;
vtwr('c_TransComm','c_Extraction','SEAsia','RestofWorld') = 58.92301941 ;
vtwr('c_TransComm','c_Extraction','SouthAsia','Oceania') = 7.735498428 ;
vtwr('c_TransComm','c_Extraction','SouthAsia','EastAsia') = 1449.633911 ;
vtwr('c_TransComm','c_Extraction','SouthAsia','SEAsia') = 59.58466339 ;
vtwr('c_TransComm','c_Extraction','SouthAsia','SouthAsia') = 115.684082 ;
vtwr('c_TransComm','c_Extraction','SouthAsia','NAmerica') = 37.20053864 ;
vtwr('c_TransComm','c_Extraction','SouthAsia','LatinAmer') = 8.968050957 ;
vtwr('c_TransComm','c_Extraction','SouthAsia','EU_28') = 157.0361938 ;
vtwr('c_TransComm','c_Extraction','SouthAsia','MENA') = 83.5464859 ;
vtwr('c_TransComm','c_Extraction','SouthAsia','SSA') = 13.0853529 ;
vtwr('c_TransComm','c_Extraction','SouthAsia','RestofWorld') = 8.045982361 ;
vtwr('c_TransComm','c_Extraction','NAmerica','Oceania') = 104.3958054 ;
vtwr('c_TransComm','c_Extraction','NAmerica','EastAsia') = 6240.060547 ;
vtwr('c_TransComm','c_Extraction','NAmerica','SEAsia') = 165.5890808 ;
vtwr('c_TransComm','c_Extraction','NAmerica','SouthAsia') = 332.0239258 ;
vtwr('c_TransComm','c_Extraction','NAmerica','NAmerica') = 6063.608398 ;
vtwr('c_TransComm','c_Extraction','NAmerica','LatinAmer') = 430.437561 ;
vtwr('c_TransComm','c_Extraction','NAmerica','EU_28') = 784.4259644 ;
vtwr('c_TransComm','c_Extraction','NAmerica','MENA') = 310.5485535 ;
vtwr('c_TransComm','c_Extraction','NAmerica','SSA') = 43.66843796 ;
vtwr('c_TransComm','c_Extraction','NAmerica','RestofWorld') = 126.777832 ;
vtwr('c_TransComm','c_Extraction','LatinAmer','Oceania') = 29.16136932 ;
vtwr('c_TransComm','c_Extraction','LatinAmer','EastAsia') = 22339.31445 ;
vtwr('c_TransComm','c_Extraction','LatinAmer','SEAsia') = 534.62677 ;
vtwr('c_TransComm','c_Extraction','LatinAmer','SouthAsia') = 2734.456299 ;
vtwr('c_TransComm','c_Extraction','LatinAmer','NAmerica') = 2907.92334 ;
vtwr('c_TransComm','c_Extraction','LatinAmer','LatinAmer') = 2280.32251 ;
vtwr('c_TransComm','c_Extraction','LatinAmer','EU_28') = 7420.286621 ;
vtwr('c_TransComm','c_Extraction','LatinAmer','MENA') = 665.151062 ;
vtwr('c_TransComm','c_Extraction','LatinAmer','SSA') = 20.26258469 ;
vtwr('c_TransComm','c_Extraction','LatinAmer','RestofWorld') = 25.0826149 ;
vtwr('c_TransComm','c_Extraction','EU_28','Oceania') = 9.947365761 ;
vtwr('c_TransComm','c_Extraction','EU_28','EastAsia') = 479.2526855 ;
vtwr('c_TransComm','c_Extraction','EU_28','SEAsia') = 63.45878601 ;
vtwr('c_TransComm','c_Extraction','EU_28','SouthAsia') = 47.725914 ;
vtwr('c_TransComm','c_Extraction','EU_28','NAmerica') = 224.9269714 ;
vtwr('c_TransComm','c_Extraction','EU_28','LatinAmer') = 80.42961121 ;
vtwr('c_TransComm','c_Extraction','EU_28','EU_28') = 4360.245605 ;
vtwr('c_TransComm','c_Extraction','EU_28','MENA') = 461.3415833 ;
vtwr('c_TransComm','c_Extraction','EU_28','SSA') = 102.6293488 ;
vtwr('c_TransComm','c_Extraction','EU_28','RestofWorld') = 302.2319031 ;
vtwr('c_TransComm','c_Extraction','MENA','Oceania') = 127.3412704 ;
vtwr('c_TransComm','c_Extraction','MENA','EastAsia') = 14818.54297 ;
vtwr('c_TransComm','c_Extraction','MENA','SEAsia') = 2920.232422 ;
vtwr('c_TransComm','c_Extraction','MENA','SouthAsia') = 6096.048828 ;
vtwr('c_TransComm','c_Extraction','MENA','NAmerica') = 4072.511475 ;
vtwr('c_TransComm','c_Extraction','MENA','LatinAmer') = 235.0233002 ;
vtwr('c_TransComm','c_Extraction','MENA','EU_28') = 5897.084961 ;
vtwr('c_TransComm','c_Extraction','MENA','MENA') = 1756.021606 ;
vtwr('c_TransComm','c_Extraction','MENA','SSA') = 784.7488403 ;
vtwr('c_TransComm','c_Extraction','MENA','RestofWorld') = 135.4348602 ;
vtwr('c_TransComm','c_Extraction','SSA','Oceania') = 43.17804718 ;
vtwr('c_TransComm','c_Extraction','SSA','EastAsia') = 5411.291016 ;
vtwr('c_TransComm','c_Extraction','SSA','SEAsia') = 151.9726868 ;
vtwr('c_TransComm','c_Extraction','SSA','SouthAsia') = 1236.889404 ;
vtwr('c_TransComm','c_Extraction','SSA','NAmerica') = 3063.723145 ;
vtwr('c_TransComm','c_Extraction','SSA','LatinAmer') = 379.5687866 ;
vtwr('c_TransComm','c_Extraction','SSA','EU_28') = 2141.007324 ;
vtwr('c_TransComm','c_Extraction','SSA','MENA') = 108.1046295 ;
vtwr('c_TransComm','c_Extraction','SSA','SSA') = 1585.598877 ;
vtwr('c_TransComm','c_Extraction','SSA','RestofWorld') = 125.587822 ;
vtwr('c_TransComm','c_Extraction','RestofWorld','Oceania') = 6.422769547 ;
vtwr('c_TransComm','c_Extraction','RestofWorld','EastAsia') = 4500.025391 ;
vtwr('c_TransComm','c_Extraction','RestofWorld','SEAsia') = 211.0853577 ;
vtwr('c_TransComm','c_Extraction','RestofWorld','SouthAsia') = 90.84353638 ;
vtwr('c_TransComm','c_Extraction','RestofWorld','NAmerica') = 815.3264771 ;
vtwr('c_TransComm','c_Extraction','RestofWorld','LatinAmer') = 113.5252914 ;
vtwr('c_TransComm','c_Extraction','RestofWorld','EU_28') = 11554.33984 ;
vtwr('c_TransComm','c_Extraction','RestofWorld','MENA') = 797.7728271 ;
vtwr('c_TransComm','c_Extraction','RestofWorld','SSA') = 24.53230286 ;
vtwr('c_TransComm','c_Extraction','RestofWorld','RestofWorld') = 2881.921875 ;
vtwr('c_TransComm','c_ProcFood','Oceania','Oceania') = 307.2514648 ;
vtwr('c_TransComm','c_ProcFood','Oceania','EastAsia') = 475.2288208 ;
vtwr('c_TransComm','c_ProcFood','Oceania','SEAsia') = 363.4501038 ;
vtwr('c_TransComm','c_ProcFood','Oceania','SouthAsia') = 49.46528625 ;
vtwr('c_TransComm','c_ProcFood','Oceania','NAmerica') = 186.7101135 ;
vtwr('c_TransComm','c_ProcFood','Oceania','LatinAmer') = 46.70573807 ;
vtwr('c_TransComm','c_ProcFood','Oceania','EU_28') = 202.0822754 ;
vtwr('c_TransComm','c_ProcFood','Oceania','MENA') = 132.2709198 ;
vtwr('c_TransComm','c_ProcFood','Oceania','SSA') = 42.67424011 ;
vtwr('c_TransComm','c_ProcFood','Oceania','RestofWorld') = 22.87034416 ;
vtwr('c_TransComm','c_ProcFood','EastAsia','Oceania') = 134.1669922 ;
vtwr('c_TransComm','c_ProcFood','EastAsia','EastAsia') = 1381.64856 ;
vtwr('c_TransComm','c_ProcFood','EastAsia','SEAsia') = 666.9158325 ;
vtwr('c_TransComm','c_ProcFood','EastAsia','SouthAsia') = 45.22647095 ;
vtwr('c_TransComm','c_ProcFood','EastAsia','NAmerica') = 700.9168701 ;
vtwr('c_TransComm','c_ProcFood','EastAsia','LatinAmer') = 95.72072601 ;
vtwr('c_TransComm','c_ProcFood','EastAsia','EU_28') = 473.234375 ;
vtwr('c_TransComm','c_ProcFood','EastAsia','MENA') = 156.4423981 ;
vtwr('c_TransComm','c_ProcFood','EastAsia','SSA') = 175.2445374 ;
vtwr('c_TransComm','c_ProcFood','EastAsia','RestofWorld') = 173.9159088 ;
vtwr('c_TransComm','c_ProcFood','SEAsia','Oceania') = 248.6834106 ;
vtwr('c_TransComm','c_ProcFood','SEAsia','EastAsia') = 2022.060669 ;
vtwr('c_TransComm','c_ProcFood','SEAsia','SEAsia') = 1555.432861 ;
vtwr('c_TransComm','c_ProcFood','SEAsia','SouthAsia') = 1534.968872 ;
vtwr('c_TransComm','c_ProcFood','SEAsia','NAmerica') = 810.2817993 ;
vtwr('c_TransComm','c_ProcFood','SEAsia','LatinAmer') = 103.4373245 ;
vtwr('c_TransComm','c_ProcFood','SEAsia','EU_28') = 759.4641724 ;
vtwr('c_TransComm','c_ProcFood','SEAsia','MENA') = 632.7214355 ;
vtwr('c_TransComm','c_ProcFood','SEAsia','SSA') = 372.8618469 ;
vtwr('c_TransComm','c_ProcFood','SEAsia','RestofWorld') = 228.373642 ;
vtwr('c_TransComm','c_ProcFood','SouthAsia','Oceania') = 15.07089996 ;
vtwr('c_TransComm','c_ProcFood','SouthAsia','EastAsia') = 237.3133087 ;
vtwr('c_TransComm','c_ProcFood','SouthAsia','SEAsia') = 218.7618256 ;
vtwr('c_TransComm','c_ProcFood','SouthAsia','SouthAsia') = 239.9482574 ;
vtwr('c_TransComm','c_ProcFood','SouthAsia','NAmerica') = 183.429657 ;
vtwr('c_TransComm','c_ProcFood','SouthAsia','LatinAmer') = 11.54140091 ;
vtwr('c_TransComm','c_ProcFood','SouthAsia','EU_28') = 220.8407745 ;
vtwr('c_TransComm','c_ProcFood','SouthAsia','MENA') = 211.7338104 ;
vtwr('c_TransComm','c_ProcFood','SouthAsia','SSA') = 100.7929001 ;
vtwr('c_TransComm','c_ProcFood','SouthAsia','RestofWorld') = 42.13700485 ;
vtwr('c_TransComm','c_ProcFood','NAmerica','Oceania') = 124.7048111 ;
vtwr('c_TransComm','c_ProcFood','NAmerica','EastAsia') = 1210.854248 ;
vtwr('c_TransComm','c_ProcFood','NAmerica','SEAsia') = 379.5939026 ;
vtwr('c_TransComm','c_ProcFood','NAmerica','SouthAsia') = 37.59740448 ;
vtwr('c_TransComm','c_ProcFood','NAmerica','NAmerica') = 3974.906982 ;
vtwr('c_TransComm','c_ProcFood','NAmerica','LatinAmer') = 739.9321899 ;
vtwr('c_TransComm','c_ProcFood','NAmerica','EU_28') = 206.044754 ;
vtwr('c_TransComm','c_ProcFood','NAmerica','MENA') = 342.2026672 ;
vtwr('c_TransComm','c_ProcFood','NAmerica','SSA') = 86.37885284 ;
vtwr('c_TransComm','c_ProcFood','NAmerica','RestofWorld') = 107.9413986 ;
vtwr('c_TransComm','c_ProcFood','LatinAmer','Oceania') = 55.88465118 ;
vtwr('c_TransComm','c_ProcFood','LatinAmer','EastAsia') = 885.3692017 ;
vtwr('c_TransComm','c_ProcFood','LatinAmer','SEAsia') = 492.2524414 ;
vtwr('c_TransComm','c_ProcFood','LatinAmer','SouthAsia') = 312.9140625 ;
vtwr('c_TransComm','c_ProcFood','LatinAmer','NAmerica') = 1124.714233 ;
vtwr('c_TransComm','c_ProcFood','LatinAmer','LatinAmer') = 1917.991455 ;
vtwr('c_TransComm','c_ProcFood','LatinAmer','EU_28') = 1303.670288 ;
vtwr('c_TransComm','c_ProcFood','LatinAmer','MENA') = 1087.671875 ;
vtwr('c_TransComm','c_ProcFood','LatinAmer','SSA') = 681.7415771 ;
vtwr('c_TransComm','c_ProcFood','LatinAmer','RestofWorld') = 227.7446899 ;
vtwr('c_TransComm','c_ProcFood','EU_28','Oceania') = 221.6456451 ;
vtwr('c_TransComm','c_ProcFood','EU_28','EastAsia') = 924.6422119 ;
vtwr('c_TransComm','c_ProcFood','EU_28','SEAsia') = 323.0018005 ;
vtwr('c_TransComm','c_ProcFood','EU_28','SouthAsia') = 69.52237701 ;
vtwr('c_TransComm','c_ProcFood','EU_28','NAmerica') = 1613.479614 ;
vtwr('c_TransComm','c_ProcFood','EU_28','LatinAmer') = 353.280304 ;
vtwr('c_TransComm','c_ProcFood','EU_28','EU_28') = 6944.850586 ;
vtwr('c_TransComm','c_ProcFood','EU_28','MENA') = 1276.687378 ;
vtwr('c_TransComm','c_ProcFood','EU_28','SSA') = 735.4544678 ;
vtwr('c_TransComm','c_ProcFood','EU_28','RestofWorld') = 1210.392334 ;
vtwr('c_TransComm','c_ProcFood','MENA','Oceania') = 12.50304031 ;
vtwr('c_TransComm','c_ProcFood','MENA','EastAsia') = 72.46939087 ;
vtwr('c_TransComm','c_ProcFood','MENA','SEAsia') = 68.78567505 ;
vtwr('c_TransComm','c_ProcFood','MENA','SouthAsia') = 59.40032578 ;
vtwr('c_TransComm','c_ProcFood','MENA','NAmerica') = 82.92259216 ;
vtwr('c_TransComm','c_ProcFood','MENA','LatinAmer') = 9.528510094 ;
vtwr('c_TransComm','c_ProcFood','MENA','EU_28') = 434.6097412 ;
vtwr('c_TransComm','c_ProcFood','MENA','MENA') = 1335.704834 ;
vtwr('c_TransComm','c_ProcFood','MENA','SSA') = 256.1540527 ;
vtwr('c_TransComm','c_ProcFood','MENA','RestofWorld') = 82.79960632 ;
vtwr('c_TransComm','c_ProcFood','SSA','Oceania') = 14.99411201 ;
vtwr('c_TransComm','c_ProcFood','SSA','EastAsia') = 55.34527969 ;
vtwr('c_TransComm','c_ProcFood','SSA','SEAsia') = 13.81038952 ;
vtwr('c_TransComm','c_ProcFood','SSA','SouthAsia') = 3.395606995 ;
vtwr('c_TransComm','c_ProcFood','SSA','NAmerica') = 58.62884521 ;
vtwr('c_TransComm','c_ProcFood','SSA','LatinAmer') = 5.485240936 ;
vtwr('c_TransComm','c_ProcFood','SSA','EU_28') = 420.865387 ;
vtwr('c_TransComm','c_ProcFood','SSA','MENA') = 41.00989532 ;
vtwr('c_TransComm','c_ProcFood','SSA','SSA') = 436.9702759 ;
vtwr('c_TransComm','c_ProcFood','SSA','RestofWorld') = 35.8846817 ;
vtwr('c_TransComm','c_ProcFood','RestofWorld','Oceania') = 19.1491394 ;
vtwr('c_TransComm','c_ProcFood','RestofWorld','EastAsia') = 337.9468384 ;
vtwr('c_TransComm','c_ProcFood','RestofWorld','SEAsia') = 29.13926506 ;
vtwr('c_TransComm','c_ProcFood','RestofWorld','SouthAsia') = 103.6597824 ;
vtwr('c_TransComm','c_ProcFood','RestofWorld','NAmerica') = 149.1716614 ;
vtwr('c_TransComm','c_ProcFood','RestofWorld','LatinAmer') = 29.64699173 ;
vtwr('c_TransComm','c_ProcFood','RestofWorld','EU_28') = 588.1594238 ;
vtwr('c_TransComm','c_ProcFood','RestofWorld','MENA') = 346.1357727 ;
vtwr('c_TransComm','c_ProcFood','RestofWorld','SSA') = 45.22504425 ;
vtwr('c_TransComm','c_ProcFood','RestofWorld','RestofWorld') = 876.4853516 ;
vtwr('c_TransComm','c_TextWapp','Oceania','Oceania') = 38.09447861 ;
vtwr('c_TransComm','c_TextWapp','Oceania','EastAsia') = 21.98965645 ;
vtwr('c_TransComm','c_TextWapp','Oceania','SEAsia') = 8.403082848 ;
vtwr('c_TransComm','c_TextWapp','Oceania','SouthAsia') = 8.338934898 ;
vtwr('c_TransComm','c_TextWapp','Oceania','NAmerica') = 7.713421822 ;
vtwr('c_TransComm','c_TextWapp','Oceania','LatinAmer') = 0.5180473328 ;
vtwr('c_TransComm','c_TextWapp','Oceania','EU_28') = 14.08974361 ;
vtwr('c_TransComm','c_TextWapp','Oceania','MENA') = 2.683538675 ;
vtwr('c_TransComm','c_TextWapp','Oceania','SSA') = 1.741727471 ;
vtwr('c_TransComm','c_TextWapp','Oceania','RestofWorld') = 0.6261165738 ;
vtwr('c_TransComm','c_TextWapp','EastAsia','Oceania') = 355.5993042 ;
vtwr('c_TransComm','c_TextWapp','EastAsia','EastAsia') = 4127.391602 ;
vtwr('c_TransComm','c_TextWapp','EastAsia','SEAsia') = 2105.802734 ;
vtwr('c_TransComm','c_TextWapp','EastAsia','SouthAsia') = 866.6965332 ;
vtwr('c_TransComm','c_TextWapp','EastAsia','NAmerica') = 3661.773193 ;
vtwr('c_TransComm','c_TextWapp','EastAsia','LatinAmer') = 1060.391357 ;
vtwr('c_TransComm','c_TextWapp','EastAsia','EU_28') = 3761.942383 ;
vtwr('c_TransComm','c_TextWapp','EastAsia','MENA') = 1624.910522 ;
vtwr('c_TransComm','c_TextWapp','EastAsia','SSA') = 647.5808716 ;
vtwr('c_TransComm','c_TextWapp','EastAsia','RestofWorld') = 1013.012756 ;
vtwr('c_TransComm','c_TextWapp','SEAsia','Oceania') = 37.16503143 ;
vtwr('c_TransComm','c_TextWapp','SEAsia','EastAsia') = 676.2877808 ;
vtwr('c_TransComm','c_TextWapp','SEAsia','SEAsia') = 297.756012 ;
vtwr('c_TransComm','c_TextWapp','SEAsia','SouthAsia') = 107.0427704 ;
vtwr('c_TransComm','c_TextWapp','SEAsia','NAmerica') = 1175.964111 ;
vtwr('c_TransComm','c_TextWapp','SEAsia','LatinAmer') = 93.48394775 ;
vtwr('c_TransComm','c_TextWapp','SEAsia','EU_28') = 570.0308228 ;
vtwr('c_TransComm','c_TextWapp','SEAsia','MENA') = 235.6176147 ;
vtwr('c_TransComm','c_TextWapp','SEAsia','SSA') = 36.3108902 ;
vtwr('c_TransComm','c_TextWapp','SEAsia','RestofWorld') = 40.43737793 ;
vtwr('c_TransComm','c_TextWapp','SouthAsia','Oceania') = 60.92760086 ;
vtwr('c_TransComm','c_TextWapp','SouthAsia','EastAsia') = 320.315979 ;
vtwr('c_TransComm','c_TextWapp','SouthAsia','SEAsia') = 87.07885742 ;
vtwr('c_TransComm','c_TextWapp','SouthAsia','SouthAsia') = 211.5924988 ;
vtwr('c_TransComm','c_TextWapp','SouthAsia','NAmerica') = 1105.631348 ;
vtwr('c_TransComm','c_TextWapp','SouthAsia','LatinAmer') = 158.2949677 ;
vtwr('c_TransComm','c_TextWapp','SouthAsia','EU_28') = 1780.285156 ;
vtwr('c_TransComm','c_TextWapp','SouthAsia','MENA') = 569.4195557 ;
vtwr('c_TransComm','c_TextWapp','SouthAsia','SSA') = 139.2413025 ;
vtwr('c_TransComm','c_TextWapp','SouthAsia','RestofWorld') = 105.7578278 ;
vtwr('c_TransComm','c_TextWapp','NAmerica','Oceania') = 21.31626129 ;
vtwr('c_TransComm','c_TextWapp','NAmerica','EastAsia') = 156.7946625 ;
vtwr('c_TransComm','c_TextWapp','NAmerica','SEAsia') = 31.60693741 ;
vtwr('c_TransComm','c_TextWapp','NAmerica','SouthAsia') = 16.07570267 ;
vtwr('c_TransComm','c_TextWapp','NAmerica','NAmerica') = 968.8094482 ;
vtwr('c_TransComm','c_TextWapp','NAmerica','LatinAmer') = 402.4942627 ;
vtwr('c_TransComm','c_TextWapp','NAmerica','EU_28') = 57.75512695 ;
vtwr('c_TransComm','c_TextWapp','NAmerica','MENA') = 44.60916138 ;
vtwr('c_TransComm','c_TextWapp','NAmerica','SSA') = 12.78853893 ;
vtwr('c_TransComm','c_TextWapp','NAmerica','RestofWorld') = 14.32967854 ;
vtwr('c_TransComm','c_TextWapp','LatinAmer','Oceania') = 2.642967939 ;
vtwr('c_TransComm','c_TextWapp','LatinAmer','EastAsia') = 21.26115417 ;
vtwr('c_TransComm','c_TextWapp','LatinAmer','SEAsia') = 3.931489706 ;
vtwr('c_TransComm','c_TextWapp','LatinAmer','SouthAsia') = 2.49340868 ;
vtwr('c_TransComm','c_TextWapp','LatinAmer','NAmerica') = 581.5006104 ;
vtwr('c_TransComm','c_TextWapp','LatinAmer','LatinAmer') = 342.3372498 ;
vtwr('c_TransComm','c_TextWapp','LatinAmer','EU_28') = 47.97709656 ;
vtwr('c_TransComm','c_TextWapp','LatinAmer','MENA') = 7.250053406 ;
vtwr('c_TransComm','c_TextWapp','LatinAmer','SSA') = 3.320334435 ;
vtwr('c_TransComm','c_TextWapp','LatinAmer','RestofWorld') = 1.80554986 ;
vtwr('c_TransComm','c_TextWapp','EU_28','Oceania') = 42.94523621 ;
vtwr('c_TransComm','c_TextWapp','EU_28','EastAsia') = 541.1876831 ;
vtwr('c_TransComm','c_TextWapp','EU_28','SEAsia') = 76.31137848 ;
vtwr('c_TransComm','c_TextWapp','EU_28','SouthAsia') = 66.20001221 ;
vtwr('c_TransComm','c_TextWapp','EU_28','NAmerica') = 407.7770691 ;
vtwr('c_TransComm','c_TextWapp','EU_28','LatinAmer') = 86.40031433 ;
vtwr('c_TransComm','c_TextWapp','EU_28','EU_28') = 2857.05835 ;
vtwr('c_TransComm','c_TextWapp','EU_28','MENA') = 711.829834 ;
vtwr('c_TransComm','c_TextWapp','EU_28','SSA') = 82.75616455 ;
vtwr('c_TransComm','c_TextWapp','EU_28','RestofWorld') = 592.3740845 ;
vtwr('c_TransComm','c_TextWapp','MENA','Oceania') = 8.97214222 ;
vtwr('c_TransComm','c_TextWapp','MENA','EastAsia') = 49.62565613 ;
vtwr('c_TransComm','c_TextWapp','MENA','SEAsia') = 10.94406509 ;
vtwr('c_TransComm','c_TextWapp','MENA','SouthAsia') = 25.39663887 ;
vtwr('c_TransComm','c_TextWapp','MENA','NAmerica') = 293.8949585 ;
vtwr('c_TransComm','c_TextWapp','MENA','LatinAmer') = 21.38903236 ;
vtwr('c_TransComm','c_TextWapp','MENA','EU_28') = 1543.48877 ;
vtwr('c_TransComm','c_TextWapp','MENA','MENA') = 416.3016968 ;
vtwr('c_TransComm','c_TextWapp','MENA','SSA') = 40.49950027 ;
vtwr('c_TransComm','c_TextWapp','MENA','RestofWorld') = 221.1615295 ;
vtwr('c_TransComm','c_TextWapp','SSA','Oceania') = 2.34983182 ;
vtwr('c_TransComm','c_TextWapp','SSA','EastAsia') = 8.638480186 ;
vtwr('c_TransComm','c_TextWapp','SSA','SEAsia') = 1.145000815 ;
vtwr('c_TransComm','c_TextWapp','SSA','SouthAsia') = 2.248837709 ;
vtwr('c_TransComm','c_TextWapp','SSA','NAmerica') = 57.22114563 ;
vtwr('c_TransComm','c_TextWapp','SSA','LatinAmer') = 0.9867258668 ;
vtwr('c_TransComm','c_TextWapp','SSA','EU_28') = 60.85282135 ;
vtwr('c_TransComm','c_TextWapp','SSA','MENA') = 4.670875549 ;
vtwr('c_TransComm','c_TextWapp','SSA','SSA') = 64.70481873 ;
vtwr('c_TransComm','c_TextWapp','SSA','RestofWorld') = 1.879934549 ;
vtwr('c_TransComm','c_TextWapp','RestofWorld','Oceania') = 1.283864379 ;
vtwr('c_TransComm','c_TextWapp','RestofWorld','EastAsia') = 22.26465034 ;
vtwr('c_TransComm','c_TextWapp','RestofWorld','SEAsia') = 4.136247635 ;
vtwr('c_TransComm','c_TextWapp','RestofWorld','SouthAsia') = 4.211417198 ;
vtwr('c_TransComm','c_TextWapp','RestofWorld','NAmerica') = 18.18490028 ;
vtwr('c_TransComm','c_TextWapp','RestofWorld','LatinAmer') = 1.955175996 ;
vtwr('c_TransComm','c_TextWapp','RestofWorld','EU_28') = 216.4373474 ;
vtwr('c_TransComm','c_TextWapp','RestofWorld','MENA') = 45.24924088 ;
vtwr('c_TransComm','c_TextWapp','RestofWorld','SSA') = 3.663090944 ;
vtwr('c_TransComm','c_TextWapp','RestofWorld','RestofWorld') = 126.5587082 ;
vtwr('c_TransComm','c_LightMnfc','Oceania','Oceania') = 298.9833374 ;
vtwr('c_TransComm','c_LightMnfc','Oceania','EastAsia') = 338.2555847 ;
vtwr('c_TransComm','c_LightMnfc','Oceania','SEAsia') = 137.0535278 ;
vtwr('c_TransComm','c_LightMnfc','Oceania','SouthAsia') = 29.93248177 ;
vtwr('c_TransComm','c_LightMnfc','Oceania','NAmerica') = 96.51522827 ;
vtwr('c_TransComm','c_LightMnfc','Oceania','LatinAmer') = 9.706673622 ;
vtwr('c_TransComm','c_LightMnfc','Oceania','EU_28') = 65.02264404 ;
vtwr('c_TransComm','c_LightMnfc','Oceania','MENA') = 45.41790009 ;
vtwr('c_TransComm','c_LightMnfc','Oceania','SSA') = 26.53958702 ;
vtwr('c_TransComm','c_LightMnfc','Oceania','RestofWorld') = 6.120571613 ;
vtwr('c_TransComm','c_LightMnfc','EastAsia','Oceania') = 1504.019165 ;
vtwr('c_TransComm','c_LightMnfc','EastAsia','EastAsia') = 6320.17041 ;
vtwr('c_TransComm','c_LightMnfc','EastAsia','SEAsia') = 3203.708984 ;
vtwr('c_TransComm','c_LightMnfc','EastAsia','SouthAsia') = 1229.845703 ;
vtwr('c_TransComm','c_LightMnfc','EastAsia','NAmerica') = 12766.4043 ;
vtwr('c_TransComm','c_LightMnfc','EastAsia','LatinAmer') = 2507.48584 ;
vtwr('c_TransComm','c_LightMnfc','EastAsia','EU_28') = 9481.485352 ;
vtwr('c_TransComm','c_LightMnfc','EastAsia','MENA') = 3465.066895 ;
vtwr('c_TransComm','c_LightMnfc','EastAsia','SSA') = 1485.853882 ;
vtwr('c_TransComm','c_LightMnfc','EastAsia','RestofWorld') = 2331.251953 ;
vtwr('c_TransComm','c_LightMnfc','SEAsia','Oceania') = 553.7263794 ;
vtwr('c_TransComm','c_LightMnfc','SEAsia','EastAsia') = 2032.035156 ;
vtwr('c_TransComm','c_LightMnfc','SEAsia','SEAsia') = 1537.427612 ;
vtwr('c_TransComm','c_LightMnfc','SEAsia','SouthAsia') = 319.765686 ;
vtwr('c_TransComm','c_LightMnfc','SEAsia','NAmerica') = 1495.544678 ;
vtwr('c_TransComm','c_LightMnfc','SEAsia','LatinAmer') = 255.9853516 ;
vtwr('c_TransComm','c_LightMnfc','SEAsia','EU_28') = 1401.281494 ;
vtwr('c_TransComm','c_LightMnfc','SEAsia','MENA') = 627.6694946 ;
vtwr('c_TransComm','c_LightMnfc','SEAsia','SSA') = 166.3795013 ;
vtwr('c_TransComm','c_LightMnfc','SEAsia','RestofWorld') = 181.4272308 ;
vtwr('c_TransComm','c_LightMnfc','SouthAsia','Oceania') = 39.63079834 ;
vtwr('c_TransComm','c_LightMnfc','SouthAsia','EastAsia') = 112.064827 ;
vtwr('c_TransComm','c_LightMnfc','SouthAsia','SEAsia') = 98.10297394 ;
vtwr('c_TransComm','c_LightMnfc','SouthAsia','SouthAsia') = 146.2007141 ;
vtwr('c_TransComm','c_LightMnfc','SouthAsia','NAmerica') = 322.5776062 ;
vtwr('c_TransComm','c_LightMnfc','SouthAsia','LatinAmer') = 63.78517151 ;
vtwr('c_TransComm','c_LightMnfc','SouthAsia','EU_28') = 616.0737915 ;
vtwr('c_TransComm','c_LightMnfc','SouthAsia','MENA') = 299.7376099 ;
vtwr('c_TransComm','c_LightMnfc','SouthAsia','SSA') = 200.8905182 ;
vtwr('c_TransComm','c_LightMnfc','SouthAsia','RestofWorld') = 46.13575363 ;
vtwr('c_TransComm','c_LightMnfc','NAmerica','Oceania') = 332.5759277 ;
vtwr('c_TransComm','c_LightMnfc','NAmerica','EastAsia') = 2793.282959 ;
vtwr('c_TransComm','c_LightMnfc','NAmerica','SEAsia') = 507.9736633 ;
vtwr('c_TransComm','c_LightMnfc','NAmerica','SouthAsia') = 307.1821899 ;
vtwr('c_TransComm','c_LightMnfc','NAmerica','NAmerica') = 15617.02246 ;
vtwr('c_TransComm','c_LightMnfc','NAmerica','LatinAmer') = 1923.470459 ;
vtwr('c_TransComm','c_LightMnfc','NAmerica','EU_28') = 754.4312744 ;
vtwr('c_TransComm','c_LightMnfc','NAmerica','MENA') = 900.382019 ;
vtwr('c_TransComm','c_LightMnfc','NAmerica','SSA') = 359.9741821 ;
vtwr('c_TransComm','c_LightMnfc','NAmerica','RestofWorld') = 285.629364 ;
vtwr('c_TransComm','c_LightMnfc','LatinAmer','Oceania') = 34.60336304 ;
vtwr('c_TransComm','c_LightMnfc','LatinAmer','EastAsia') = 530.565918 ;
vtwr('c_TransComm','c_LightMnfc','LatinAmer','SEAsia') = 61.85014725 ;
vtwr('c_TransComm','c_LightMnfc','LatinAmer','SouthAsia') = 20.65457153 ;
vtwr('c_TransComm','c_LightMnfc','LatinAmer','NAmerica') = 792.5299072 ;
vtwr('c_TransComm','c_LightMnfc','LatinAmer','LatinAmer') = 2043.847656 ;
vtwr('c_TransComm','c_LightMnfc','LatinAmer','EU_28') = 684.2021484 ;
vtwr('c_TransComm','c_LightMnfc','LatinAmer','MENA') = 110.2285767 ;
vtwr('c_TransComm','c_LightMnfc','LatinAmer','SSA') = 108.849205 ;
vtwr('c_TransComm','c_LightMnfc','LatinAmer','RestofWorld') = 30.74674034 ;
vtwr('c_TransComm','c_LightMnfc','EU_28','Oceania') = 706.6298218 ;
vtwr('c_TransComm','c_LightMnfc','EU_28','EastAsia') = 3654.482666 ;
vtwr('c_TransComm','c_LightMnfc','EU_28','SEAsia') = 753.2603149 ;
vtwr('c_TransComm','c_LightMnfc','EU_28','SouthAsia') = 510.4926758 ;
vtwr('c_TransComm','c_LightMnfc','EU_28','NAmerica') = 3503.386963 ;
vtwr('c_TransComm','c_LightMnfc','EU_28','LatinAmer') = 1302.077271 ;
vtwr('c_TransComm','c_LightMnfc','EU_28','EU_28') = 17593.92578 ;
vtwr('c_TransComm','c_LightMnfc','EU_28','MENA') = 4599.15918 ;
vtwr('c_TransComm','c_LightMnfc','EU_28','SSA') = 1308.581055 ;
vtwr('c_TransComm','c_LightMnfc','EU_28','RestofWorld') = 4348.29541 ;
vtwr('c_TransComm','c_LightMnfc','MENA','Oceania') = 13.55994606 ;
vtwr('c_TransComm','c_LightMnfc','MENA','EastAsia') = 62.03187561 ;
vtwr('c_TransComm','c_LightMnfc','MENA','SEAsia') = 24.12172318 ;
vtwr('c_TransComm','c_LightMnfc','MENA','SouthAsia') = 127.7159882 ;
vtwr('c_TransComm','c_LightMnfc','MENA','NAmerica') = 167.9816742 ;
vtwr('c_TransComm','c_LightMnfc','MENA','LatinAmer') = 39.17355347 ;
vtwr('c_TransComm','c_LightMnfc','MENA','EU_28') = 1482.635742 ;
vtwr('c_TransComm','c_LightMnfc','MENA','MENA') = 1374.305054 ;
vtwr('c_TransComm','c_LightMnfc','MENA','SSA') = 180.8005371 ;
vtwr('c_TransComm','c_LightMnfc','MENA','RestofWorld') = 332.2447205 ;
vtwr('c_TransComm','c_LightMnfc','SSA','Oceania') = 16.96597481 ;
vtwr('c_TransComm','c_LightMnfc','SSA','EastAsia') = 124.1796265 ;
vtwr('c_TransComm','c_LightMnfc','SSA','SEAsia') = 86.27751923 ;
vtwr('c_TransComm','c_LightMnfc','SSA','SouthAsia') = 25.39610863 ;
vtwr('c_TransComm','c_LightMnfc','SSA','NAmerica') = 129.2754059 ;
vtwr('c_TransComm','c_LightMnfc','SSA','LatinAmer') = 13.86900425 ;
vtwr('c_TransComm','c_LightMnfc','SSA','EU_28') = 408.1833496 ;
vtwr('c_TransComm','c_LightMnfc','SSA','MENA') = 50.93305969 ;
vtwr('c_TransComm','c_LightMnfc','SSA','SSA') = 407.9977112 ;
vtwr('c_TransComm','c_LightMnfc','SSA','RestofWorld') = 10.40352345 ;
vtwr('c_TransComm','c_LightMnfc','RestofWorld','Oceania') = 13.98250484 ;
vtwr('c_TransComm','c_LightMnfc','RestofWorld','EastAsia') = 491.2780151 ;
vtwr('c_TransComm','c_LightMnfc','RestofWorld','SEAsia') = 59.93955231 ;
vtwr('c_TransComm','c_LightMnfc','RestofWorld','SouthAsia') = 97.98284149 ;
vtwr('c_TransComm','c_LightMnfc','RestofWorld','NAmerica') = 167.4163666 ;
vtwr('c_TransComm','c_LightMnfc','RestofWorld','LatinAmer') = 62.32030487 ;
vtwr('c_TransComm','c_LightMnfc','RestofWorld','EU_28') = 1127.016479 ;
vtwr('c_TransComm','c_LightMnfc','RestofWorld','MENA') = 325.8786926 ;
vtwr('c_TransComm','c_LightMnfc','RestofWorld','SSA') = 33.02674103 ;
vtwr('c_TransComm','c_LightMnfc','RestofWorld','RestofWorld') = 1405.411865 ;
vtwr('c_TransComm','c_HeavyMnfc','Oceania','Oceania') = 421.6591797 ;
vtwr('c_TransComm','c_HeavyMnfc','Oceania','EastAsia') = 663.9050903 ;
vtwr('c_TransComm','c_HeavyMnfc','Oceania','SEAsia') = 387.8289795 ;
vtwr('c_TransComm','c_HeavyMnfc','Oceania','SouthAsia') = 87.002388 ;
vtwr('c_TransComm','c_HeavyMnfc','Oceania','NAmerica') = 218.0684509 ;
vtwr('c_TransComm','c_HeavyMnfc','Oceania','LatinAmer') = 48.82910919 ;
vtwr('c_TransComm','c_HeavyMnfc','Oceania','EU_28') = 141.1456451 ;
vtwr('c_TransComm','c_HeavyMnfc','Oceania','MENA') = 55.26025772 ;
vtwr('c_TransComm','c_HeavyMnfc','Oceania','SSA') = 71.37731934 ;
vtwr('c_TransComm','c_HeavyMnfc','Oceania','RestofWorld') = 30.21434593 ;
vtwr('c_TransComm','c_HeavyMnfc','EastAsia','Oceania') = 2382.994141 ;
vtwr('c_TransComm','c_HeavyMnfc','EastAsia','EastAsia') = 32495.05078 ;
vtwr('c_TransComm','c_HeavyMnfc','EastAsia','SEAsia') = 12203.24902 ;
vtwr('c_TransComm','c_HeavyMnfc','EastAsia','SouthAsia') = 4119.28418 ;
vtwr('c_TransComm','c_HeavyMnfc','EastAsia','NAmerica') = 18031.61719 ;
vtwr('c_TransComm','c_HeavyMnfc','EastAsia','LatinAmer') = 4219.742188 ;
vtwr('c_TransComm','c_HeavyMnfc','EastAsia','EU_28') = 13484.11816 ;
vtwr('c_TransComm','c_HeavyMnfc','EastAsia','MENA') = 5665.836914 ;
vtwr('c_TransComm','c_HeavyMnfc','EastAsia','SSA') = 2322.607178 ;
vtwr('c_TransComm','c_HeavyMnfc','EastAsia','RestofWorld') = 3208.793213 ;
vtwr('c_TransComm','c_HeavyMnfc','SEAsia','Oceania') = 1128.859253 ;
vtwr('c_TransComm','c_HeavyMnfc','SEAsia','EastAsia') = 7791.658203 ;
vtwr('c_TransComm','c_HeavyMnfc','SEAsia','SEAsia') = 6991.325684 ;
vtwr('c_TransComm','c_HeavyMnfc','SEAsia','SouthAsia') = 1375.92749 ;
vtwr('c_TransComm','c_HeavyMnfc','SEAsia','NAmerica') = 2383.780762 ;
vtwr('c_TransComm','c_HeavyMnfc','SEAsia','LatinAmer') = 423.061676 ;
vtwr('c_TransComm','c_HeavyMnfc','SEAsia','EU_28') = 2214.273682 ;
vtwr('c_TransComm','c_HeavyMnfc','SEAsia','MENA') = 919.6344604 ;
vtwr('c_TransComm','c_HeavyMnfc','SEAsia','SSA') = 347.539917 ;
vtwr('c_TransComm','c_HeavyMnfc','SEAsia','RestofWorld') = 199.3588562 ;
vtwr('c_TransComm','c_HeavyMnfc','SouthAsia','Oceania') = 97.08379364 ;
vtwr('c_TransComm','c_HeavyMnfc','SouthAsia','EastAsia') = 795.7197266 ;
vtwr('c_TransComm','c_HeavyMnfc','SouthAsia','SEAsia') = 529.536438 ;
vtwr('c_TransComm','c_HeavyMnfc','SouthAsia','SouthAsia') = 764.803894 ;
vtwr('c_TransComm','c_HeavyMnfc','SouthAsia','NAmerica') = 863.796814 ;
vtwr('c_TransComm','c_HeavyMnfc','SouthAsia','LatinAmer') = 418.7842407 ;
vtwr('c_TransComm','c_HeavyMnfc','SouthAsia','EU_28') = 1507.4646 ;
vtwr('c_TransComm','c_HeavyMnfc','SouthAsia','MENA') = 1409.916626 ;
vtwr('c_TransComm','c_HeavyMnfc','SouthAsia','SSA') = 706.6940918 ;
vtwr('c_TransComm','c_HeavyMnfc','SouthAsia','RestofWorld') = 107.3543243 ;
vtwr('c_TransComm','c_HeavyMnfc','NAmerica','Oceania') = 1071.779541 ;
vtwr('c_TransComm','c_HeavyMnfc','NAmerica','EastAsia') = 6985.676758 ;
vtwr('c_TransComm','c_HeavyMnfc','NAmerica','SEAsia') = 2176.955566 ;
vtwr('c_TransComm','c_HeavyMnfc','NAmerica','SouthAsia') = 933.928772 ;
vtwr('c_TransComm','c_HeavyMnfc','NAmerica','NAmerica') = 23046.50977 ;
vtwr('c_TransComm','c_HeavyMnfc','NAmerica','LatinAmer') = 7724.46582 ;
vtwr('c_TransComm','c_HeavyMnfc','NAmerica','EU_28') = 4610.817383 ;
vtwr('c_TransComm','c_HeavyMnfc','NAmerica','MENA') = 3025.485596 ;
vtwr('c_TransComm','c_HeavyMnfc','NAmerica','SSA') = 902.8693848 ;
vtwr('c_TransComm','c_HeavyMnfc','NAmerica','RestofWorld') = 838.4298096 ;
vtwr('c_TransComm','c_HeavyMnfc','LatinAmer','Oceania') = 47.12960815 ;
vtwr('c_TransComm','c_HeavyMnfc','LatinAmer','EastAsia') = 881.2296143 ;
vtwr('c_TransComm','c_HeavyMnfc','LatinAmer','SEAsia') = 201.9386444 ;
vtwr('c_TransComm','c_HeavyMnfc','LatinAmer','SouthAsia') = 64.62201691 ;
vtwr('c_TransComm','c_HeavyMnfc','LatinAmer','NAmerica') = 3443.933105 ;
vtwr('c_TransComm','c_HeavyMnfc','LatinAmer','LatinAmer') = 4283.893066 ;
vtwr('c_TransComm','c_HeavyMnfc','LatinAmer','EU_28') = 1393.242798 ;
vtwr('c_TransComm','c_HeavyMnfc','LatinAmer','MENA') = 221.6578064 ;
vtwr('c_TransComm','c_HeavyMnfc','LatinAmer','SSA') = 150.248642 ;
vtwr('c_TransComm','c_HeavyMnfc','LatinAmer','RestofWorld') = 107.9528503 ;
vtwr('c_TransComm','c_HeavyMnfc','EU_28','Oceania') = 1037.373413 ;
vtwr('c_TransComm','c_HeavyMnfc','EU_28','EastAsia') = 7708.831055 ;
vtwr('c_TransComm','c_HeavyMnfc','EU_28','SEAsia') = 1949.332764 ;
vtwr('c_TransComm','c_HeavyMnfc','EU_28','SouthAsia') = 1680.275146 ;
vtwr('c_TransComm','c_HeavyMnfc','EU_28','NAmerica') = 8631.185547 ;
vtwr('c_TransComm','c_HeavyMnfc','EU_28','LatinAmer') = 3041.247803 ;
vtwr('c_TransComm','c_HeavyMnfc','EU_28','EU_28') = 32280.47656 ;
vtwr('c_TransComm','c_HeavyMnfc','EU_28','MENA') = 9310.204102 ;
vtwr('c_TransComm','c_HeavyMnfc','EU_28','SSA') = 2451.450195 ;
vtwr('c_TransComm','c_HeavyMnfc','EU_28','RestofWorld') = 7712.452148 ;
vtwr('c_TransComm','c_HeavyMnfc','MENA','Oceania') = 279.0144348 ;
vtwr('c_TransComm','c_HeavyMnfc','MENA','EastAsia') = 5055.532227 ;
vtwr('c_TransComm','c_HeavyMnfc','MENA','SEAsia') = 1565.837646 ;
vtwr('c_TransComm','c_HeavyMnfc','MENA','SouthAsia') = 2547.157227 ;
vtwr('c_TransComm','c_HeavyMnfc','MENA','NAmerica') = 1264.597412 ;
vtwr('c_TransComm','c_HeavyMnfc','MENA','LatinAmer') = 534.4108276 ;
vtwr('c_TransComm','c_HeavyMnfc','MENA','EU_28') = 4906.641602 ;
vtwr('c_TransComm','c_HeavyMnfc','MENA','MENA') = 6345.353027 ;
vtwr('c_TransComm','c_HeavyMnfc','MENA','SSA') = 989.5099487 ;
vtwr('c_TransComm','c_HeavyMnfc','MENA','RestofWorld') = 775.3380737 ;
vtwr('c_TransComm','c_HeavyMnfc','SSA','Oceania') = 25.52560043 ;
vtwr('c_TransComm','c_HeavyMnfc','SSA','EastAsia') = 338.4257812 ;
vtwr('c_TransComm','c_HeavyMnfc','SSA','SEAsia') = 152.8884125 ;
vtwr('c_TransComm','c_HeavyMnfc','SSA','SouthAsia') = 253.6096344 ;
vtwr('c_TransComm','c_HeavyMnfc','SSA','NAmerica') = 292.0353699 ;
vtwr('c_TransComm','c_HeavyMnfc','SSA','LatinAmer') = 58.58626938 ;
vtwr('c_TransComm','c_HeavyMnfc','SSA','EU_28') = 476.8900452 ;
vtwr('c_TransComm','c_HeavyMnfc','SSA','MENA') = 95.40235901 ;
vtwr('c_TransComm','c_HeavyMnfc','SSA','SSA') = 1486.609985 ;
vtwr('c_TransComm','c_HeavyMnfc','SSA','RestofWorld') = 64.05128479 ;
vtwr('c_TransComm','c_HeavyMnfc','RestofWorld','Oceania') = 81.04733276 ;
vtwr('c_TransComm','c_HeavyMnfc','RestofWorld','EastAsia') = 2094.812256 ;
vtwr('c_TransComm','c_HeavyMnfc','RestofWorld','SEAsia') = 596.078064 ;
vtwr('c_TransComm','c_HeavyMnfc','RestofWorld','SouthAsia') = 497.8959656 ;
vtwr('c_TransComm','c_HeavyMnfc','RestofWorld','NAmerica') = 1998.579712 ;
vtwr('c_TransComm','c_HeavyMnfc','RestofWorld','LatinAmer') = 869.8188477 ;
vtwr('c_TransComm','c_HeavyMnfc','RestofWorld','EU_28') = 6988.164551 ;
vtwr('c_TransComm','c_HeavyMnfc','RestofWorld','MENA') = 2111.098389 ;
vtwr('c_TransComm','c_HeavyMnfc','RestofWorld','SSA') = 233.572113 ;
vtwr('c_TransComm','c_HeavyMnfc','RestofWorld','RestofWorld') = 2980.727783 ;

* save data (10 cells)
save('Oceania') = 199542.4062 ;
save('EastAsia') = 3396990 ;
save('SEAsia') = 395959.8438 ;
save('SouthAsia') = 286564.1875 ;
save('NAmerica') = 513600.4062 ;
save('LatinAmer') = 315548.0938 ;
save('EU_28') = 967188.8125 ;
save('MENA') = 839884.25 ;
save('SSA') = 180493.2031 ;
save('RestofWorld') = 667770 ;

* vdep data (10 cells)
vdep('Oceania') = 229877.0938 ;
vdep('EastAsia') = 2127261.5 ;
vdep('SEAsia') = 284227.1562 ;
vdep('SouthAsia') = 221470.4375 ;
vdep('NAmerica') = 2258271.5 ;
vdep('LatinAmer') = 647166.875 ;
vdep('EU_28') = 2226819.5 ;
vdep('MENA') = 435971.7812 ;
vdep('SSA') = 117259.7656 ;
vdep('RestofWorld') = 482927.9062 ;

* vkb data (10 cells)
vkb('Oceania') = 5746928 ;
vkb('EastAsia') = 53181540 ;
vkb('SEAsia') = 7105678.5 ;
vkb('SouthAsia') = 5536761 ;
vkb('NAmerica') = 56456788 ;
vkb('LatinAmer') = 16179172 ;
vkb('EU_28') = 55670488 ;
vkb('MENA') = 10899295 ;
vkb('SSA') = 2931494.25 ;
vkb('RestofWorld') = 12073198 ;

* pop0 data (10 cells)
pop0('Oceania') = 36.82978058 ;
pop0('EastAsia') = 1580.330078 ;
pop0('SEAsia') = 601.8123779 ;
pop0('SouthAsia') = 1628.377319 ;
pop0('NAmerica') = 465.4141541 ;
pop0('LatinAmer') = 482.4779663 ;
pop0('EU_28') = 507.7632141 ;
pop0('MENA') = 460.3120422 ;
pop0('SSA') = 878.8171997 ;
pop0('RestofWorld') = 312.9768677 ;


ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;
ptax(i0,a0,r) = makb(i0,a0,r) - maks(i0,a0,r) ;

* --------------------------------------------------------------------------------------------------
*
*     Read in CO2 emissions data
*
* --------------------------------------------------------------------------------------------------

Parameters
   mdf(i0, a0, r)          "CO2 emissions from domestic intermediate demand"
   mmf(i0, a0, r)          "CO2 emissions from domestic intermediate demand"
   mdp(i0, r)              "CO2 emissions from domestic private demand"
   mmp(i0, r)              "CO2 emissions from domestic private demand"
   mdg(i0, r)              "CO2 emissions from domestic public demand"
   mmg(i0, r)              "CO2 emissions from domestic public demand"
   mdi(i0, r)              "CO2 emissions from investment demand"
   mmi(i0, r)              "CO2 emissions from investment demand"
;

$ifthen exist "%inDir%/%BaseName%Emiss.gdx"
   * [stripped for NEOS inline] $gdxin "%inDir%/%BaseName%Emiss.gdx"
   * [stripped for NEOS inline] $load mdf mmf mdp mmp mdg mmg mdi mmi
   * [stripped for NEOS inline] $gdxin
$else
   mdf(i0,a0,r) = 0 ;
   mmf(i0,a0,r) = 0 ;
   mdp(i0,r)    = 0 ;
   mmp(i0,r)    = 0 ;
   mdg(i0,r)    = 0 ;
   mmg(i0,r)    = 0 ;
   mdi(i0,r)    = 0 ;
   mmi(i0,r)    = 0 ;
$endif

* --------------------------------------------------------------------------------------------------
*
*     Read in the GTAP Parameters
*
* --------------------------------------------------------------------------------------------------

Parameters
   esubt(a0,r)       "Top level CES substitution elasticity"
   esubc(a0,r)       "ND nest CES substitution elasticity"
   esubva(a0,r)      "VA nest CES substitution elasticity"

   etraq(a0,r)       "CET make elasticity"
   esubq(i0,r)       "CES make elasticity"

   incpar(i0,r)      "CDE expansion parameter"
   subpar(i0,r)      "CDE substitution parameter"

   esubg(r)          "CES government expenditure elasticity"
   esubi(r)          "CES investment expenditure elasticity"

   esubd(i0,r)       "Top level Armington elasticity"
   esubm(i0,r)       "Second level Armington elasticity"
   esubs(i0)         "CES margin elasticity"

   etrae(fp,r)       "CET elasticity for factors"
   rorFlex0(r)       "Flexibility of foreign capital"
;

* [stripped for NEOS inline] $gdxin "%inDir%/%BaseName%Prm.gdx"
* [stripped for NEOS inline] $loadDC esubt=esubt esubc=esubc esubva=esubva
* [stripped for NEOS inline] $loadDC etraq=etraq esubq=esubq
* [stripped for NEOS inline] $loadDC incpar=incpar subpar=subpar esubg=esubg esubi=esubi
* [stripped for NEOS inline] $loadDC esubd=esubd esubm=esubm esubs=esubs
* [stripped for NEOS inline] $loadDC etrae=etrae rorFlex0=rorFlex
* [stripped for NEOS inline] $gdxin

* === Inlined PARAMETER data from in.gdx ===

$onImplicitAssign

* esubva data (90 cells)
esubva('a_agricultur','Oceania') = 0.3657608628 ;
esubva('a_agricultur','EastAsia') = 0.3657608628 ;
esubva('a_agricultur','SEAsia') = 0.3657608628 ;
esubva('a_agricultur','SouthAsia') = 0.3657608628 ;
esubva('a_agricultur','NAmerica') = 0.3657608628 ;
esubva('a_agricultur','LatinAmer') = 0.3657608628 ;
esubva('a_agricultur','EU_28') = 0.3657608628 ;
esubva('a_agricultur','MENA') = 0.3657608628 ;
esubva('a_agricultur','SSA') = 0.3657608628 ;
esubva('a_agricultur','RestofWorld') = 0.3657608628 ;
esubva('a_Extraction','Oceania') = 0.200000003 ;
esubva('a_Extraction','EastAsia') = 0.200000003 ;
esubva('a_Extraction','SEAsia') = 0.200000003 ;
esubva('a_Extraction','SouthAsia') = 0.200000003 ;
esubva('a_Extraction','NAmerica') = 0.200000003 ;
esubva('a_Extraction','LatinAmer') = 0.200000003 ;
esubva('a_Extraction','EU_28') = 0.200000003 ;
esubva('a_Extraction','MENA') = 0.200000003 ;
esubva('a_Extraction','SSA') = 0.200000003 ;
esubva('a_Extraction','RestofWorld') = 0.200000003 ;
esubva('a_ProcFood','Oceania') = 1.120000005 ;
esubva('a_ProcFood','EastAsia') = 1.120000005 ;
esubva('a_ProcFood','SEAsia') = 1.120000005 ;
esubva('a_ProcFood','SouthAsia') = 1.120000005 ;
esubva('a_ProcFood','NAmerica') = 1.120000005 ;
esubva('a_ProcFood','LatinAmer') = 1.120000005 ;
esubva('a_ProcFood','EU_28') = 1.120000005 ;
esubva('a_ProcFood','MENA') = 1.120000005 ;
esubva('a_ProcFood','SSA') = 1.120000005 ;
esubva('a_ProcFood','RestofWorld') = 1.120000005 ;
esubva('a_TextWapp','Oceania') = 1.25999999 ;
esubva('a_TextWapp','EastAsia') = 1.25999999 ;
esubva('a_TextWapp','SEAsia') = 1.25999999 ;
esubva('a_TextWapp','SouthAsia') = 1.25999999 ;
esubva('a_TextWapp','NAmerica') = 1.25999999 ;
esubva('a_TextWapp','LatinAmer') = 1.25999999 ;
esubva('a_TextWapp','EU_28') = 1.25999999 ;
esubva('a_TextWapp','MENA') = 1.25999999 ;
esubva('a_TextWapp','SSA') = 1.25999999 ;
esubva('a_TextWapp','RestofWorld') = 1.25999999 ;
esubva('a_LightMnfc','Oceania') = 1.25999999 ;
esubva('a_LightMnfc','EastAsia') = 1.25999999 ;
esubva('a_LightMnfc','SEAsia') = 1.25999999 ;
esubva('a_LightMnfc','SouthAsia') = 1.25999999 ;
esubva('a_LightMnfc','NAmerica') = 1.25999999 ;
esubva('a_LightMnfc','LatinAmer') = 1.25999999 ;
esubva('a_LightMnfc','EU_28') = 1.25999999 ;
esubva('a_LightMnfc','MENA') = 1.25999999 ;
esubva('a_LightMnfc','SSA') = 1.25999999 ;
esubva('a_LightMnfc','RestofWorld') = 1.25999999 ;
esubva('a_HeavyMnfc','Oceania') = 1.25999999 ;
esubva('a_HeavyMnfc','EastAsia') = 1.25999999 ;
esubva('a_HeavyMnfc','SEAsia') = 1.25999999 ;
esubva('a_HeavyMnfc','SouthAsia') = 1.25999999 ;
esubva('a_HeavyMnfc','NAmerica') = 1.25999999 ;
esubva('a_HeavyMnfc','LatinAmer') = 1.25999999 ;
esubva('a_HeavyMnfc','EU_28') = 1.25999999 ;
esubva('a_HeavyMnfc','MENA') = 1.25999999 ;
esubva('a_HeavyMnfc','SSA') = 1.25999999 ;
esubva('a_HeavyMnfc','RestofWorld') = 1.25999999 ;
esubva('a_Util_Cons','Oceania') = 1.367443204 ;
esubva('a_Util_Cons','EastAsia') = 1.367443204 ;
esubva('a_Util_Cons','SEAsia') = 1.367443204 ;
esubva('a_Util_Cons','SouthAsia') = 1.367443204 ;
esubva('a_Util_Cons','NAmerica') = 1.367443204 ;
esubva('a_Util_Cons','LatinAmer') = 1.367443204 ;
esubva('a_Util_Cons','EU_28') = 1.367443204 ;
esubva('a_Util_Cons','MENA') = 1.367443204 ;
esubva('a_Util_Cons','SSA') = 1.367443204 ;
esubva('a_Util_Cons','RestofWorld') = 1.367443204 ;
esubva('a_TransComm','Oceania') = 1.626793385 ;
esubva('a_TransComm','EastAsia') = 1.626793385 ;
esubva('a_TransComm','SEAsia') = 1.626793385 ;
esubva('a_TransComm','SouthAsia') = 1.626793385 ;
esubva('a_TransComm','NAmerica') = 1.626793385 ;
esubva('a_TransComm','LatinAmer') = 1.626793385 ;
esubva('a_TransComm','EU_28') = 1.626793385 ;
esubva('a_TransComm','MENA') = 1.626793385 ;
esubva('a_TransComm','SSA') = 1.626793385 ;
esubva('a_TransComm','RestofWorld') = 1.626793385 ;
esubva('a_OthService','Oceania') = 1.25999999 ;
esubva('a_OthService','EastAsia') = 1.25999999 ;
esubva('a_OthService','SEAsia') = 1.25999999 ;
esubva('a_OthService','SouthAsia') = 1.25999999 ;
esubva('a_OthService','NAmerica') = 1.25999999 ;
esubva('a_OthService','LatinAmer') = 1.25999999 ;
esubva('a_OthService','EU_28') = 1.25999999 ;
esubva('a_OthService','MENA') = 1.25999999 ;
esubva('a_OthService','SSA') = 1.25999999 ;
esubva('a_OthService','RestofWorld') = 1.25999999 ;

* etraq data (90 cells)
etraq('a_agricultur','Oceania') = -5 ;
etraq('a_agricultur','EastAsia') = -5 ;
etraq('a_agricultur','SEAsia') = -5 ;
etraq('a_agricultur','SouthAsia') = -5 ;
etraq('a_agricultur','NAmerica') = -5 ;
etraq('a_agricultur','LatinAmer') = -5 ;
etraq('a_agricultur','EU_28') = -5 ;
etraq('a_agricultur','MENA') = -5 ;
etraq('a_agricultur','SSA') = -5 ;
etraq('a_agricultur','RestofWorld') = -5 ;
etraq('a_Extraction','Oceania') = -1.000000012e-07 ;
etraq('a_Extraction','EastAsia') = -1.000000012e-07 ;
etraq('a_Extraction','SEAsia') = -1.000000012e-07 ;
etraq('a_Extraction','SouthAsia') = -1.000000012e-07 ;
etraq('a_Extraction','NAmerica') = -1.000000012e-07 ;
etraq('a_Extraction','LatinAmer') = -1.000000012e-07 ;
etraq('a_Extraction','EU_28') = -1.000000012e-07 ;
etraq('a_Extraction','MENA') = -1.000000012e-07 ;
etraq('a_Extraction','SSA') = -1.000000012e-07 ;
etraq('a_Extraction','RestofWorld') = -1.000000012e-07 ;
etraq('a_ProcFood','Oceania') = -1.000000012e-07 ;
etraq('a_ProcFood','EastAsia') = -1.000000012e-07 ;
etraq('a_ProcFood','SEAsia') = -1.000000012e-07 ;
etraq('a_ProcFood','SouthAsia') = -1.000000012e-07 ;
etraq('a_ProcFood','NAmerica') = -1.000000012e-07 ;
etraq('a_ProcFood','LatinAmer') = -1.000000012e-07 ;
etraq('a_ProcFood','EU_28') = -1.000000012e-07 ;
etraq('a_ProcFood','MENA') = -1.000000012e-07 ;
etraq('a_ProcFood','SSA') = -1.000000012e-07 ;
etraq('a_ProcFood','RestofWorld') = -1.000000012e-07 ;
etraq('a_TextWapp','Oceania') = -1.000000012e-07 ;
etraq('a_TextWapp','EastAsia') = -1.000000012e-07 ;
etraq('a_TextWapp','SEAsia') = -1.000000012e-07 ;
etraq('a_TextWapp','SouthAsia') = -1.000000012e-07 ;
etraq('a_TextWapp','NAmerica') = -1.000000012e-07 ;
etraq('a_TextWapp','LatinAmer') = -1.000000012e-07 ;
etraq('a_TextWapp','EU_28') = -1.000000012e-07 ;
etraq('a_TextWapp','MENA') = -1.000000012e-07 ;
etraq('a_TextWapp','SSA') = -1.000000012e-07 ;
etraq('a_TextWapp','RestofWorld') = -1.000000012e-07 ;
etraq('a_LightMnfc','Oceania') = -1.000000012e-07 ;
etraq('a_LightMnfc','EastAsia') = -1.000000012e-07 ;
etraq('a_LightMnfc','SEAsia') = -1.000000012e-07 ;
etraq('a_LightMnfc','SouthAsia') = -1.000000012e-07 ;
etraq('a_LightMnfc','NAmerica') = -1.000000012e-07 ;
etraq('a_LightMnfc','LatinAmer') = -1.000000012e-07 ;
etraq('a_LightMnfc','EU_28') = -1.000000012e-07 ;
etraq('a_LightMnfc','MENA') = -1.000000012e-07 ;
etraq('a_LightMnfc','SSA') = -1.000000012e-07 ;
etraq('a_LightMnfc','RestofWorld') = -1.000000012e-07 ;
etraq('a_HeavyMnfc','Oceania') = -1.000000012e-07 ;
etraq('a_HeavyMnfc','EastAsia') = -1.000000012e-07 ;
etraq('a_HeavyMnfc','SEAsia') = -1.000000012e-07 ;
etraq('a_HeavyMnfc','SouthAsia') = -1.000000012e-07 ;
etraq('a_HeavyMnfc','NAmerica') = -1.000000012e-07 ;
etraq('a_HeavyMnfc','LatinAmer') = -1.000000012e-07 ;
etraq('a_HeavyMnfc','EU_28') = -1.000000012e-07 ;
etraq('a_HeavyMnfc','MENA') = -1.000000012e-07 ;
etraq('a_HeavyMnfc','SSA') = -1.000000012e-07 ;
etraq('a_HeavyMnfc','RestofWorld') = -1.000000012e-07 ;
etraq('a_Util_Cons','Oceania') = -1.000000012e-07 ;
etraq('a_Util_Cons','EastAsia') = -1.000000012e-07 ;
etraq('a_Util_Cons','SEAsia') = -1.000000012e-07 ;
etraq('a_Util_Cons','SouthAsia') = -1.000000012e-07 ;
etraq('a_Util_Cons','NAmerica') = -1.000000012e-07 ;
etraq('a_Util_Cons','LatinAmer') = -1.000000012e-07 ;
etraq('a_Util_Cons','EU_28') = -1.000000012e-07 ;
etraq('a_Util_Cons','MENA') = -1.000000012e-07 ;
etraq('a_Util_Cons','SSA') = -1.000000012e-07 ;
etraq('a_Util_Cons','RestofWorld') = -1.000000012e-07 ;
etraq('a_TransComm','Oceania') = -1.000000012e-07 ;
etraq('a_TransComm','EastAsia') = -1.000000012e-07 ;
etraq('a_TransComm','SEAsia') = -1.000000012e-07 ;
etraq('a_TransComm','SouthAsia') = -1.000000012e-07 ;
etraq('a_TransComm','NAmerica') = -1.000000012e-07 ;
etraq('a_TransComm','LatinAmer') = -1.000000012e-07 ;
etraq('a_TransComm','EU_28') = -1.000000012e-07 ;
etraq('a_TransComm','MENA') = -1.000000012e-07 ;
etraq('a_TransComm','SSA') = -1.000000012e-07 ;
etraq('a_TransComm','RestofWorld') = -1.000000012e-07 ;
etraq('a_OthService','Oceania') = -1.000000012e-07 ;
etraq('a_OthService','EastAsia') = -1.000000012e-07 ;
etraq('a_OthService','SEAsia') = -1.000000012e-07 ;
etraq('a_OthService','SouthAsia') = -1.000000012e-07 ;
etraq('a_OthService','NAmerica') = -1.000000012e-07 ;
etraq('a_OthService','LatinAmer') = -1.000000012e-07 ;
etraq('a_OthService','EU_28') = -1.000000012e-07 ;
etraq('a_OthService','MENA') = -1.000000012e-07 ;
etraq('a_OthService','SSA') = -1.000000012e-07 ;
etraq('a_OthService','RestofWorld') = -1.000000012e-07 ;

* esubd data (100 cells)
esubd('c_Crops','Oceania') = 2.598682404 ;
esubd('c_Crops','EastAsia') = 2.598682404 ;
esubd('c_Crops','SEAsia') = 2.598682404 ;
esubd('c_Crops','SouthAsia') = 2.598682404 ;
esubd('c_Crops','NAmerica') = 2.598682404 ;
esubd('c_Crops','LatinAmer') = 2.598682404 ;
esubd('c_Crops','EU_28') = 2.598682404 ;
esubd('c_Crops','MENA') = 2.598682404 ;
esubd('c_Crops','SSA') = 2.598682404 ;
esubd('c_Crops','RestofWorld') = 2.598682404 ;
esubd('c_MeatLstk','Oceania') = 3.114979029 ;
esubd('c_MeatLstk','EastAsia') = 3.114979029 ;
esubd('c_MeatLstk','SEAsia') = 3.114979029 ;
esubd('c_MeatLstk','SouthAsia') = 3.114979029 ;
esubd('c_MeatLstk','NAmerica') = 3.114979029 ;
esubd('c_MeatLstk','LatinAmer') = 3.114979029 ;
esubd('c_MeatLstk','EU_28') = 3.114979029 ;
esubd('c_MeatLstk','MENA') = 3.114979029 ;
esubd('c_MeatLstk','SSA') = 3.114979029 ;
esubd('c_MeatLstk','RestofWorld') = 3.114979029 ;
esubd('c_Extraction','Oceania') = 4.907746792 ;
esubd('c_Extraction','EastAsia') = 4.907746792 ;
esubd('c_Extraction','SEAsia') = 4.907746792 ;
esubd('c_Extraction','SouthAsia') = 4.907746792 ;
esubd('c_Extraction','NAmerica') = 4.907746792 ;
esubd('c_Extraction','LatinAmer') = 4.907746792 ;
esubd('c_Extraction','EU_28') = 4.907746792 ;
esubd('c_Extraction','MENA') = 4.907746792 ;
esubd('c_Extraction','SSA') = 4.907746792 ;
esubd('c_Extraction','RestofWorld') = 4.907746792 ;
esubd('c_ProcFood','Oceania') = 2.150314093 ;
esubd('c_ProcFood','EastAsia') = 2.150314093 ;
esubd('c_ProcFood','SEAsia') = 2.150314093 ;
esubd('c_ProcFood','SouthAsia') = 2.150314093 ;
esubd('c_ProcFood','NAmerica') = 2.150314093 ;
esubd('c_ProcFood','LatinAmer') = 2.150314093 ;
esubd('c_ProcFood','EU_28') = 2.150314093 ;
esubd('c_ProcFood','MENA') = 2.150314093 ;
esubd('c_ProcFood','SSA') = 2.150314093 ;
esubd('c_ProcFood','RestofWorld') = 2.150314093 ;
esubd('c_TextWapp','Oceania') = 3.728585958 ;
esubd('c_TextWapp','EastAsia') = 3.728585958 ;
esubd('c_TextWapp','SEAsia') = 3.728585958 ;
esubd('c_TextWapp','SouthAsia') = 3.728585958 ;
esubd('c_TextWapp','NAmerica') = 3.728585958 ;
esubd('c_TextWapp','LatinAmer') = 3.728585958 ;
esubd('c_TextWapp','EU_28') = 3.728585958 ;
esubd('c_TextWapp','MENA') = 3.728585958 ;
esubd('c_TextWapp','SSA') = 3.728585958 ;
esubd('c_TextWapp','RestofWorld') = 3.728585958 ;
esubd('c_LightMnfc','Oceania') = 3.351011276 ;
esubd('c_LightMnfc','EastAsia') = 3.351011276 ;
esubd('c_LightMnfc','SEAsia') = 3.351011276 ;
esubd('c_LightMnfc','SouthAsia') = 3.351011276 ;
esubd('c_LightMnfc','NAmerica') = 3.351011276 ;
esubd('c_LightMnfc','LatinAmer') = 3.351011276 ;
esubd('c_LightMnfc','EU_28') = 3.351011276 ;
esubd('c_LightMnfc','MENA') = 3.351011276 ;
esubd('c_LightMnfc','SSA') = 3.351011276 ;
esubd('c_LightMnfc','RestofWorld') = 3.351011276 ;
esubd('c_HeavyMnfc','Oceania') = 3.390679836 ;
esubd('c_HeavyMnfc','EastAsia') = 3.390679836 ;
esubd('c_HeavyMnfc','SEAsia') = 3.390679836 ;
esubd('c_HeavyMnfc','SouthAsia') = 3.390679836 ;
esubd('c_HeavyMnfc','NAmerica') = 3.390679836 ;
esubd('c_HeavyMnfc','LatinAmer') = 3.390679836 ;
esubd('c_HeavyMnfc','EU_28') = 3.390679836 ;
esubd('c_HeavyMnfc','MENA') = 3.390679836 ;
esubd('c_HeavyMnfc','SSA') = 3.390679836 ;
esubd('c_HeavyMnfc','RestofWorld') = 3.390679836 ;
esubd('c_Util_Cons','Oceania') = 2.124590397 ;
esubd('c_Util_Cons','EastAsia') = 2.124590397 ;
esubd('c_Util_Cons','SEAsia') = 2.124590397 ;
esubd('c_Util_Cons','SouthAsia') = 2.124590397 ;
esubd('c_Util_Cons','NAmerica') = 2.124590397 ;
esubd('c_Util_Cons','LatinAmer') = 2.124590397 ;
esubd('c_Util_Cons','EU_28') = 2.124590397 ;
esubd('c_Util_Cons','MENA') = 2.124590397 ;
esubd('c_Util_Cons','SSA') = 2.124590397 ;
esubd('c_Util_Cons','RestofWorld') = 2.124590397 ;
esubd('c_TransComm','Oceania') = 1.899999976 ;
esubd('c_TransComm','EastAsia') = 1.899999976 ;
esubd('c_TransComm','SEAsia') = 1.899999976 ;
esubd('c_TransComm','SouthAsia') = 1.899999976 ;
esubd('c_TransComm','NAmerica') = 1.899999976 ;
esubd('c_TransComm','LatinAmer') = 1.899999976 ;
esubd('c_TransComm','EU_28') = 1.899999976 ;
esubd('c_TransComm','MENA') = 1.899999976 ;
esubd('c_TransComm','SSA') = 1.899999976 ;
esubd('c_TransComm','RestofWorld') = 1.899999976 ;
esubd('c_OthService','Oceania') = 1.899999976 ;
esubd('c_OthService','EastAsia') = 1.899999976 ;
esubd('c_OthService','SEAsia') = 1.899999976 ;
esubd('c_OthService','SouthAsia') = 1.899999976 ;
esubd('c_OthService','NAmerica') = 1.899999976 ;
esubd('c_OthService','LatinAmer') = 1.899999976 ;
esubd('c_OthService','EU_28') = 1.899999976 ;
esubd('c_OthService','MENA') = 1.899999976 ;
esubd('c_OthService','SSA') = 1.899999976 ;
esubd('c_OthService','RestofWorld') = 1.899999976 ;

* esubm data (100 cells)
esubm('c_Crops','Oceania') = 5.077477932 ;
esubm('c_Crops','EastAsia') = 5.077477932 ;
esubm('c_Crops','SEAsia') = 5.077477932 ;
esubm('c_Crops','SouthAsia') = 5.077477932 ;
esubm('c_Crops','NAmerica') = 5.077477932 ;
esubm('c_Crops','LatinAmer') = 5.077477932 ;
esubm('c_Crops','EU_28') = 5.077477932 ;
esubm('c_Crops','MENA') = 5.077477932 ;
esubm('c_Crops','SSA') = 5.077477932 ;
esubm('c_Crops','RestofWorld') = 5.077477932 ;
esubm('c_MeatLstk','Oceania') = 7.350851536 ;
esubm('c_MeatLstk','EastAsia') = 7.350851536 ;
esubm('c_MeatLstk','SEAsia') = 7.350851536 ;
esubm('c_MeatLstk','SouthAsia') = 7.350851536 ;
esubm('c_MeatLstk','NAmerica') = 7.350851536 ;
esubm('c_MeatLstk','LatinAmer') = 7.350851536 ;
esubm('c_MeatLstk','EU_28') = 7.350851536 ;
esubm('c_MeatLstk','MENA') = 7.350851536 ;
esubm('c_MeatLstk','SSA') = 7.350851536 ;
esubm('c_MeatLstk','RestofWorld') = 7.350851536 ;
esubm('c_Extraction','Oceania') = 11.3472929 ;
esubm('c_Extraction','EastAsia') = 11.3472929 ;
esubm('c_Extraction','SEAsia') = 11.3472929 ;
esubm('c_Extraction','SouthAsia') = 11.3472929 ;
esubm('c_Extraction','NAmerica') = 11.3472929 ;
esubm('c_Extraction','LatinAmer') = 11.3472929 ;
esubm('c_Extraction','EU_28') = 11.3472929 ;
esubm('c_Extraction','MENA') = 11.3472929 ;
esubm('c_Extraction','SSA') = 11.3472929 ;
esubm('c_Extraction','RestofWorld') = 11.3472929 ;
esubm('c_ProcFood','Oceania') = 4.477708817 ;
esubm('c_ProcFood','EastAsia') = 4.477708817 ;
esubm('c_ProcFood','SEAsia') = 4.477708817 ;
esubm('c_ProcFood','SouthAsia') = 4.477708817 ;
esubm('c_ProcFood','NAmerica') = 4.477708817 ;
esubm('c_ProcFood','LatinAmer') = 4.477708817 ;
esubm('c_ProcFood','EU_28') = 4.477708817 ;
esubm('c_ProcFood','MENA') = 4.477708817 ;
esubm('c_ProcFood','SSA') = 4.477708817 ;
esubm('c_ProcFood','RestofWorld') = 4.477708817 ;
esubm('c_TextWapp','Oceania') = 7.454963684 ;
esubm('c_TextWapp','EastAsia') = 7.454963684 ;
esubm('c_TextWapp','SEAsia') = 7.454963684 ;
esubm('c_TextWapp','SouthAsia') = 7.454963684 ;
esubm('c_TextWapp','NAmerica') = 7.454963684 ;
esubm('c_TextWapp','LatinAmer') = 7.454963684 ;
esubm('c_TextWapp','EU_28') = 7.454963684 ;
esubm('c_TextWapp','MENA') = 7.454963684 ;
esubm('c_TextWapp','SSA') = 7.454963684 ;
esubm('c_TextWapp','RestofWorld') = 7.454963684 ;
esubm('c_LightMnfc','Oceania') = 6.713831425 ;
esubm('c_LightMnfc','EastAsia') = 6.713831425 ;
esubm('c_LightMnfc','SEAsia') = 6.713831425 ;
esubm('c_LightMnfc','SouthAsia') = 6.713831425 ;
esubm('c_LightMnfc','NAmerica') = 6.713831425 ;
esubm('c_LightMnfc','LatinAmer') = 6.713831425 ;
esubm('c_LightMnfc','EU_28') = 6.713831425 ;
esubm('c_LightMnfc','MENA') = 6.713831425 ;
esubm('c_LightMnfc','SSA') = 6.713831425 ;
esubm('c_LightMnfc','RestofWorld') = 6.713831425 ;
esubm('c_HeavyMnfc','Oceania') = 7.179564476 ;
esubm('c_HeavyMnfc','EastAsia') = 7.179564476 ;
esubm('c_HeavyMnfc','SEAsia') = 7.179564476 ;
esubm('c_HeavyMnfc','SouthAsia') = 7.179564476 ;
esubm('c_HeavyMnfc','NAmerica') = 7.179564476 ;
esubm('c_HeavyMnfc','LatinAmer') = 7.179564476 ;
esubm('c_HeavyMnfc','EU_28') = 7.179564476 ;
esubm('c_HeavyMnfc','MENA') = 7.179564476 ;
esubm('c_HeavyMnfc','SSA') = 7.179564476 ;
esubm('c_HeavyMnfc','RestofWorld') = 7.179564476 ;
esubm('c_Util_Cons','Oceania') = 4.64574194 ;
esubm('c_Util_Cons','EastAsia') = 4.64574194 ;
esubm('c_Util_Cons','SEAsia') = 4.64574194 ;
esubm('c_Util_Cons','SouthAsia') = 4.64574194 ;
esubm('c_Util_Cons','NAmerica') = 4.64574194 ;
esubm('c_Util_Cons','LatinAmer') = 4.64574194 ;
esubm('c_Util_Cons','EU_28') = 4.64574194 ;
esubm('c_Util_Cons','MENA') = 4.64574194 ;
esubm('c_Util_Cons','SSA') = 4.64574194 ;
esubm('c_Util_Cons','RestofWorld') = 4.64574194 ;
esubm('c_TransComm','Oceania') = 3.799999952 ;
esubm('c_TransComm','EastAsia') = 3.799999952 ;
esubm('c_TransComm','SEAsia') = 3.799999952 ;
esubm('c_TransComm','SouthAsia') = 3.799999952 ;
esubm('c_TransComm','NAmerica') = 3.799999952 ;
esubm('c_TransComm','LatinAmer') = 3.799999952 ;
esubm('c_TransComm','EU_28') = 3.799999952 ;
esubm('c_TransComm','MENA') = 3.799999952 ;
esubm('c_TransComm','SSA') = 3.799999952 ;
esubm('c_TransComm','RestofWorld') = 3.799999952 ;
esubm('c_OthService','Oceania') = 3.799999952 ;
esubm('c_OthService','EastAsia') = 3.799999952 ;
esubm('c_OthService','SEAsia') = 3.799999952 ;
esubm('c_OthService','SouthAsia') = 3.799999952 ;
esubm('c_OthService','NAmerica') = 3.799999952 ;
esubm('c_OthService','LatinAmer') = 3.799999952 ;
esubm('c_OthService','EU_28') = 3.799999952 ;
esubm('c_OthService','MENA') = 3.799999952 ;
esubm('c_OthService','SSA') = 3.799999952 ;
esubm('c_OthService','RestofWorld') = 3.799999952 ;

* incpar data (100 cells)
incpar('c_Crops','Oceania') = 0.04179289937 ;
incpar('c_Crops','EastAsia') = 0.1421103328 ;
incpar('c_Crops','SEAsia') = 0.3534357548 ;
incpar('c_Crops','SouthAsia') = 0.5396416783 ;
incpar('c_Crops','NAmerica') = 0.01546779368 ;
incpar('c_Crops','LatinAmer') = 0.1053327993 ;
incpar('c_Crops','EU_28') = 0.01127303578 ;
incpar('c_Crops','MENA') = 0.2058271468 ;
incpar('c_Crops','SSA') = 0.5754452348 ;
incpar('c_Crops','RestofWorld') = 0.1368436217 ;
incpar('c_MeatLstk','Oceania') = 0.4515456855 ;
incpar('c_MeatLstk','EastAsia') = 0.5205250382 ;
incpar('c_MeatLstk','SEAsia') = 0.5894579887 ;
incpar('c_MeatLstk','SouthAsia') = 0.7081666589 ;
incpar('c_MeatLstk','NAmerica') = 0.5700877905 ;
incpar('c_MeatLstk','LatinAmer') = 0.5399753451 ;
incpar('c_MeatLstk','EU_28') = 0.5896571875 ;
incpar('c_MeatLstk','MENA') = 0.571434021 ;
incpar('c_MeatLstk','SSA') = 0.8058294654 ;
incpar('c_MeatLstk','RestofWorld') = 0.5088140368 ;
incpar('c_Extraction','Oceania') = 0.6612653732 ;
incpar('c_Extraction','EastAsia') = 0.6358565092 ;
incpar('c_Extraction','SEAsia') = 0.6011436582 ;
incpar('c_Extraction','SouthAsia') = 0.756462872 ;
incpar('c_Extraction','NAmerica') = 0.8374722004 ;
incpar('c_Extraction','LatinAmer') = 0.6907664537 ;
incpar('c_Extraction','EU_28') = 0.8761590719 ;
incpar('c_Extraction','MENA') = 0.7949383855 ;
incpar('c_Extraction','SSA') = 0.9308588505 ;
incpar('c_Extraction','RestofWorld') = 1.029101014 ;
incpar('c_ProcFood','Oceania') = 0.322021693 ;
incpar('c_ProcFood','EastAsia') = 0.4322455823 ;
incpar('c_ProcFood','SEAsia') = 0.5087119341 ;
incpar('c_ProcFood','SouthAsia') = 0.6919225454 ;
incpar('c_ProcFood','NAmerica') = 0.4487244785 ;
incpar('c_ProcFood','LatinAmer') = 0.4324302077 ;
incpar('c_ProcFood','EU_28') = 0.4715442359 ;
incpar('c_ProcFood','MENA') = 0.4652261734 ;
incpar('c_ProcFood','SSA') = 0.6857769489 ;
incpar('c_ProcFood','RestofWorld') = 0.3944467306 ;
incpar('c_TextWapp','Oceania') = 0.4071263373 ;
incpar('c_TextWapp','EastAsia') = 0.5231571198 ;
incpar('c_TextWapp','SEAsia') = 0.5396739841 ;
incpar('c_TextWapp','SouthAsia') = 0.7463091016 ;
incpar('c_TextWapp','NAmerica') = 0.5611638427 ;
incpar('c_TextWapp','LatinAmer') = 0.5296119452 ;
incpar('c_TextWapp','EU_28') = 0.5760534406 ;
incpar('c_TextWapp','MENA') = 0.5562834144 ;
incpar('c_TextWapp','SSA') = 0.7542950511 ;
incpar('c_TextWapp','RestofWorld') = 0.5083265305 ;
incpar('c_LightMnfc','Oceania') = 0.9068582654 ;
incpar('c_LightMnfc','EastAsia') = 0.8881836534 ;
incpar('c_LightMnfc','SEAsia') = 0.9194084406 ;
incpar('c_LightMnfc','SouthAsia') = 0.871062696 ;
incpar('c_LightMnfc','NAmerica') = 0.95109272 ;
incpar('c_LightMnfc','LatinAmer') = 0.9757630229 ;
incpar('c_LightMnfc','EU_28') = 0.9951548576 ;
incpar('c_LightMnfc','MENA') = 1.025807381 ;
incpar('c_LightMnfc','SSA') = 0.9889326096 ;
incpar('c_LightMnfc','RestofWorld') = 1.015355587 ;
incpar('c_HeavyMnfc','Oceania') = 0.8898109794 ;
incpar('c_HeavyMnfc','EastAsia') = 0.9506518245 ;
incpar('c_HeavyMnfc','SEAsia') = 0.9564301372 ;
incpar('c_HeavyMnfc','SouthAsia') = 0.9046379924 ;
incpar('c_HeavyMnfc','NAmerica') = 0.9611953497 ;
incpar('c_HeavyMnfc','LatinAmer') = 1.036907554 ;
incpar('c_HeavyMnfc','EU_28') = 1.014967442 ;
incpar('c_HeavyMnfc','MENA') = 1.047875166 ;
incpar('c_HeavyMnfc','SSA') = 1.004287481 ;
incpar('c_HeavyMnfc','RestofWorld') = 1.057441235 ;
incpar('c_Util_Cons','Oceania') = 0.9492480159 ;
incpar('c_Util_Cons','EastAsia') = 1.01828301 ;
incpar('c_Util_Cons','SEAsia') = 1.053247213 ;
incpar('c_Util_Cons','SouthAsia') = 1.076969862 ;
incpar('c_Util_Cons','NAmerica') = 0.9885775447 ;
incpar('c_Util_Cons','LatinAmer') = 1.077428937 ;
incpar('c_Util_Cons','EU_28') = 1.056774139 ;
incpar('c_Util_Cons','MENA') = 1.113630772 ;
incpar('c_Util_Cons','SSA') = 1.083380938 ;
incpar('c_Util_Cons','RestofWorld') = 1.117065668 ;
incpar('c_TransComm','Oceania') = 1.101302624 ;
incpar('c_TransComm','EastAsia') = 1.157223463 ;
incpar('c_TransComm','SEAsia') = 1.279706836 ;
incpar('c_TransComm','SouthAsia') = 1.358746409 ;
incpar('c_TransComm','NAmerica') = 1.068913817 ;
incpar('c_TransComm','LatinAmer') = 1.164039254 ;
incpar('c_TransComm','EU_28') = 1.122446179 ;
incpar('c_TransComm','MENA') = 1.218328357 ;
incpar('c_TransComm','SSA') = 1.53237462 ;
incpar('c_TransComm','RestofWorld') = 1.223870039 ;
incpar('c_OthService','Oceania') = 1.120303869 ;
incpar('c_OthService','EastAsia') = 1.209186316 ;
incpar('c_OthService','SEAsia') = 1.35177052 ;
incpar('c_OthService','SouthAsia') = 1.50270319 ;
incpar('c_OthService','NAmerica') = 1.080884099 ;
incpar('c_OthService','LatinAmer') = 1.239590287 ;
incpar('c_OthService','EU_28') = 1.163761973 ;
incpar('c_OthService','MENA') = 1.385303617 ;
incpar('c_OthService','SSA') = 1.540711641 ;
incpar('c_OthService','RestofWorld') = 1.21839273 ;

* subpar data (100 cells)
subpar('c_Crops','Oceania') = 0.9337511063 ;
subpar('c_Crops','EastAsia') = 0.901565671 ;
subpar('c_Crops','SEAsia') = 0.8608852029 ;
subpar('c_Crops','SouthAsia') = 0.8487433791 ;
subpar('c_Crops','NAmerica') = 0.9824842811 ;
subpar('c_Crops','LatinAmer') = 0.9259001613 ;
subpar('c_Crops','EU_28') = 0.9885696173 ;
subpar('c_Crops','MENA') = 0.8941265941 ;
subpar('c_Crops','SSA') = 0.8613839746 ;
subpar('c_Crops','RestofWorld') = 0.9133843184 ;
subpar('c_MeatLstk','Oceania') = 0.2743095756 ;
subpar('c_MeatLstk','EastAsia') = 0.614413023 ;
subpar('c_MeatLstk','SEAsia') = 0.7444689274 ;
subpar('c_MeatLstk','SouthAsia') = 0.79442662 ;
subpar('c_MeatLstk','NAmerica') = 0.2694042623 ;
subpar('c_MeatLstk','LatinAmer') = 0.6402692199 ;
subpar('c_MeatLstk','EU_28') = 0.3816928566 ;
subpar('c_MeatLstk','MENA') = 0.6538035274 ;
subpar('c_MeatLstk','SSA') = 0.7647514939 ;
subpar('c_MeatLstk','RestofWorld') = 0.5835184455 ;
subpar('c_Extraction','Oceania') = 0.1442920417 ;
subpar('c_Extraction','EastAsia') = 0.5691979527 ;
subpar('c_Extraction','SEAsia') = 0.7444922328 ;
subpar('c_Extraction','SouthAsia') = 0.7914956212 ;
subpar('c_Extraction','NAmerica') = 0.2331508696 ;
subpar('c_Extraction','LatinAmer') = 0.6231916547 ;
subpar('c_Extraction','EU_28') = 0.2753469944 ;
subpar('c_Extraction','MENA') = 0.6383680105 ;
subpar('c_Extraction','SSA') = 0.7655115128 ;
subpar('c_Extraction','RestofWorld') = 0.5313465595 ;
subpar('c_ProcFood','Oceania') = 0.258630693 ;
subpar('c_ProcFood','EastAsia') = 0.5689067245 ;
subpar('c_ProcFood','SEAsia') = 0.7706686854 ;
subpar('c_ProcFood','SouthAsia') = 0.8096511364 ;
subpar('c_ProcFood','NAmerica') = 0.3394296765 ;
subpar('c_ProcFood','LatinAmer') = 0.6889560819 ;
subpar('c_ProcFood','EU_28') = 0.3809009492 ;
subpar('c_ProcFood','MENA') = 0.6903281808 ;
subpar('c_ProcFood','SSA') = 0.7908388972 ;
subpar('c_ProcFood','RestofWorld') = 0.5639646649 ;
subpar('c_TextWapp','Oceania') = 0.2036426067 ;
subpar('c_TextWapp','EastAsia') = 0.546697855 ;
subpar('c_TextWapp','SEAsia') = 0.7252844572 ;
subpar('c_TextWapp','SouthAsia') = 0.7968374491 ;
subpar('c_TextWapp','NAmerica') = 0.2461556792 ;
subpar('c_TextWapp','LatinAmer') = 0.6412342787 ;
subpar('c_TextWapp','EU_28') = 0.3305097818 ;
subpar('c_TextWapp','MENA') = 0.6252085567 ;
subpar('c_TextWapp','SSA') = 0.764402926 ;
subpar('c_TextWapp','RestofWorld') = 0.518124342 ;
subpar('c_LightMnfc','Oceania') = 0.1541496515 ;
subpar('c_LightMnfc','EastAsia') = 0.4305266142 ;
subpar('c_LightMnfc','SEAsia') = 0.5730014443 ;
subpar('c_LightMnfc','SouthAsia') = 0.7815150023 ;
subpar('c_LightMnfc','NAmerica') = 0.1730529964 ;
subpar('c_LightMnfc','LatinAmer') = 0.5012333989 ;
subpar('c_LightMnfc','EU_28') = 0.2197080106 ;
subpar('c_LightMnfc','MENA') = 0.4041865766 ;
subpar('c_LightMnfc','SSA') = 0.704320848 ;
subpar('c_LightMnfc','RestofWorld') = 0.3464259803 ;
subpar('c_HeavyMnfc','Oceania') = 0.1213101149 ;
subpar('c_HeavyMnfc','EastAsia') = 0.3669683039 ;
subpar('c_HeavyMnfc','SEAsia') = 0.595903933 ;
subpar('c_HeavyMnfc','SouthAsia') = 0.7560237646 ;
subpar('c_HeavyMnfc','NAmerica') = 0.1738564968 ;
subpar('c_HeavyMnfc','LatinAmer') = 0.4759888947 ;
subpar('c_HeavyMnfc','EU_28') = 0.2234988511 ;
subpar('c_HeavyMnfc','MENA') = 0.4194612503 ;
subpar('c_HeavyMnfc','SSA') = 0.6963559389 ;
subpar('c_HeavyMnfc','RestofWorld') = 0.3867959082 ;
subpar('c_Util_Cons','Oceania') = 0.09594676644 ;
subpar('c_Util_Cons','EastAsia') = 0.3520166278 ;
subpar('c_Util_Cons','SEAsia') = 0.5603082776 ;
subpar('c_Util_Cons','SouthAsia') = 0.7374437451 ;
subpar('c_Util_Cons','NAmerica') = 0.1538050175 ;
subpar('c_Util_Cons','LatinAmer') = 0.4633945227 ;
subpar('c_Util_Cons','EU_28') = 0.2282612771 ;
subpar('c_Util_Cons','MENA') = 0.3984217048 ;
subpar('c_Util_Cons','SSA') = 0.6873351336 ;
subpar('c_Util_Cons','RestofWorld') = 0.4371472895 ;
subpar('c_TransComm','Oceania') = 0.09665068239 ;
subpar('c_TransComm','EastAsia') = 0.2697114944 ;
subpar('c_TransComm','SEAsia') = 0.529080689 ;
subpar('c_TransComm','SouthAsia') = 0.6834033728 ;
subpar('c_TransComm','NAmerica') = 0.1605636775 ;
subpar('c_TransComm','LatinAmer') = 0.4437009394 ;
subpar('c_TransComm','EU_28') = 0.208420068 ;
subpar('c_TransComm','MENA') = 0.4218508303 ;
subpar('c_TransComm','SSA') = 0.6207545996 ;
subpar('c_TransComm','RestofWorld') = 0.3061654568 ;
subpar('c_OthService','Oceania') = 0.09809368104 ;
subpar('c_OthService','EastAsia') = 0.2820124924 ;
subpar('c_OthService','SEAsia') = 0.5053057671 ;
subpar('c_OthService','SouthAsia') = 0.6605684161 ;
subpar('c_OthService','NAmerica') = 0.1593549252 ;
subpar('c_OthService','LatinAmer') = 0.4234178662 ;
subpar('c_OthService','EU_28') = 0.1944680959 ;
subpar('c_OthService','MENA') = 0.3841217458 ;
subpar('c_OthService','SSA') = 0.5586373806 ;
subpar('c_OthService','RestofWorld') = 0.1713339835 ;

* esubc (empty in GDX — relying on $onImplicitAssign suppression)

* esubt (empty in GDX — relying on $onImplicitAssign suppression)

* esubq (empty in GDX — relying on $onImplicitAssign suppression)

* etrae data (10 cells)
etrae('Land','Oceania') = -1 ;
etrae('Land','EastAsia') = -1 ;
etrae('Land','SEAsia') = -1 ;
etrae('Land','SouthAsia') = -1 ;
etrae('Land','NAmerica') = -1 ;
etrae('Land','LatinAmer') = -1 ;
etrae('Land','EU_28') = -1 ;
etrae('Land','MENA') = -1 ;
etrae('Land','SSA') = -1 ;
etrae('Land','RestofWorld') = -1 ;

* esubg data (10 cells)
esubg('Oceania') = 1 ;
esubg('EastAsia') = 1 ;
esubg('SEAsia') = 1 ;
esubg('SouthAsia') = 1 ;
esubg('NAmerica') = 1 ;
esubg('LatinAmer') = 1 ;
esubg('EU_28') = 1 ;
esubg('MENA') = 1 ;
esubg('SSA') = 1 ;
esubg('RestofWorld') = 1 ;

* rorFlex0 data (10 cells)
rorFlex0('Oceania') = 10 ;
rorFlex0('EastAsia') = 10 ;
rorFlex0('SEAsia') = 10 ;
rorFlex0('SouthAsia') = 10 ;
rorFlex0('NAmerica') = 10 ;
rorFlex0('LatinAmer') = 10 ;
rorFlex0('EU_28') = 10 ;
rorFlex0('MENA') = 10 ;
rorFlex0('SSA') = 10 ;
rorFlex0('RestofWorld') = 10 ;

* esubi (empty in GDX — relying on $onImplicitAssign suppression)

* esubs data (1 cells)
esubs('c_TransComm') = 1 ;
* FBEP data (injected from basedata.har)
FBEP('Land','a_agricultur','Oceania') = 106.8576279 ;
FBEP('Land','a_agricultur','EastAsia') = 30020.87109 ;
FBEP('Land','a_agricultur','NAmerica') = 12701.15625 ;
FBEP('Land','a_agricultur','LatinAmer') = 0.4535400271 ;
FBEP('Land','a_agricultur','EU_28') = 25995.41211 ;
FBEP('Land','a_agricultur','MENA') = 619.9025879 ;
FBEP('Land','a_agricultur','RestofWorld') = 1439.565796 ;
FBEP('UnSkLab','a_agricultur','Oceania') = 296.6647339 ;
FBEP('UnSkLab','a_agricultur','EastAsia') = 8728.378906 ;
FBEP('UnSkLab','a_agricultur','SEAsia') = 36.97622299 ;
FBEP('UnSkLab','a_agricultur','NAmerica') = 3515.457764 ;
FBEP('UnSkLab','a_agricultur','LatinAmer') = 80.31525421 ;
FBEP('UnSkLab','a_agricultur','EU_28') = 26830.70117 ;
FBEP('UnSkLab','a_agricultur','MENA') = 134.9829559 ;
FBEP('UnSkLab','a_agricultur','RestofWorld') = 1179.786621 ;
FBEP('SkLab','a_agricultur','Oceania') = 410.6200867 ;
FBEP('SkLab','a_agricultur','EastAsia') = 554.3601685 ;
FBEP('SkLab','a_agricultur','SEAsia') = 0.2343817055 ;
FBEP('SkLab','a_agricultur','NAmerica') = 5829.989258 ;
FBEP('SkLab','a_agricultur','LatinAmer') = 2.978908539 ;
FBEP('SkLab','a_agricultur','EU_28') = 8230.058594 ;
FBEP('SkLab','a_agricultur','MENA') = 4.427939415 ;
FBEP('SkLab','a_agricultur','RestofWorld') = 241.6553955 ;
FBEP('Capital','a_agricultur','Oceania') = 273.532135 ;
FBEP('Capital','a_agricultur','EastAsia') = 10988.04102 ;
FBEP('Capital','a_agricultur','SEAsia') = 158.8828583 ;
FBEP('Capital','a_agricultur','NAmerica') = 4165.233398 ;
FBEP('Capital','a_agricultur','LatinAmer') = 2266.219238 ;
FBEP('Capital','a_agricultur','EU_28') = 25354.54883 ;
FBEP('Capital','a_agricultur','MENA') = 1334.224609 ;
FBEP('Capital','a_agricultur','SSA') = 103.4706345 ;
FBEP('Capital','a_agricultur','RestofWorld') = 3195.649902 ;
* FTRV data (injected from basedata.har)
FTRV('Land','a_agricultur','Oceania') = 180.4554901 ;
FTRV('Land','a_agricultur','EastAsia') = 552.1333008 ;
FTRV('Land','a_agricultur','SEAsia') = 639.9677734 ;
FTRV('Land','a_agricultur','SouthAsia') = 274.5598755 ;
FTRV('Land','a_agricultur','NAmerica') = 1908.596313 ;
FTRV('Land','a_agricultur','LatinAmer') = 694.5386963 ;
FTRV('Land','a_agricultur','EU_28') = 1126.629517 ;
FTRV('Land','a_agricultur','MENA') = 179.8101654 ;
FTRV('Land','a_agricultur','SSA') = 469.3840637 ;
FTRV('Land','a_agricultur','RestofWorld') = 535.8275146 ;
FTRV('UnSkLab','a_agricultur','Oceania') = 342.347168 ;
FTRV('UnSkLab','a_agricultur','EastAsia') = 6358.059082 ;
FTRV('UnSkLab','a_agricultur','SEAsia') = 1140.278564 ;
FTRV('UnSkLab','a_agricultur','SouthAsia') = 343.7228088 ;
FTRV('UnSkLab','a_agricultur','NAmerica') = 7605.792969 ;
FTRV('UnSkLab','a_agricultur','LatinAmer') = 10404.50781 ;
FTRV('UnSkLab','a_agricultur','EU_28') = 46875.36719 ;
FTRV('UnSkLab','a_agricultur','MENA') = 9155.250977 ;
FTRV('UnSkLab','a_agricultur','SSA') = 16453.80859 ;
FTRV('UnSkLab','a_agricultur','RestofWorld') = 12605.15137 ;
FTRV('UnSkLab','a_Extraction','Oceania') = 388.835144 ;
FTRV('UnSkLab','a_Extraction','EastAsia') = 1106.565186 ;
FTRV('UnSkLab','a_Extraction','SEAsia') = 414.1051331 ;
FTRV('UnSkLab','a_Extraction','SouthAsia') = 40.82024765 ;
FTRV('UnSkLab','a_Extraction','NAmerica') = 5235.427734 ;
FTRV('UnSkLab','a_Extraction','LatinAmer') = 3017.520508 ;
FTRV('UnSkLab','a_Extraction','EU_28') = 7477.418457 ;
FTRV('UnSkLab','a_Extraction','MENA') = 7521.375 ;
FTRV('UnSkLab','a_Extraction','SSA') = 2599.054443 ;
FTRV('UnSkLab','a_Extraction','RestofWorld') = 6948.269531 ;
FTRV('UnSkLab','a_ProcFood','Oceania') = 277.0494385 ;
FTRV('UnSkLab','a_ProcFood','EastAsia') = 5603.005859 ;
FTRV('UnSkLab','a_ProcFood','SEAsia') = 368.7709961 ;
FTRV('UnSkLab','a_ProcFood','SouthAsia') = 25.64126587 ;
FTRV('UnSkLab','a_ProcFood','NAmerica') = 11188.41504 ;
FTRV('UnSkLab','a_ProcFood','LatinAmer') = 3133.733154 ;
FTRV('UnSkLab','a_ProcFood','EU_28') = 30846.54102 ;
FTRV('UnSkLab','a_ProcFood','MENA') = 2239.325439 ;
FTRV('UnSkLab','a_ProcFood','SSA') = 365.569519 ;
FTRV('UnSkLab','a_ProcFood','RestofWorld') = 4287.019043 ;
FTRV('UnSkLab','a_TextWapp','Oceania') = 44.83686447 ;
FTRV('UnSkLab','a_TextWapp','EastAsia') = 2422.167725 ;
FTRV('UnSkLab','a_TextWapp','SEAsia') = 165.1972809 ;
FTRV('UnSkLab','a_TextWapp','SouthAsia') = 112.84478 ;
FTRV('UnSkLab','a_TextWapp','NAmerica') = 6248.200684 ;
FTRV('UnSkLab','a_TextWapp','LatinAmer') = 1932.021606 ;
FTRV('UnSkLab','a_TextWapp','EU_28') = 13362.67773 ;
FTRV('UnSkLab','a_TextWapp','MENA') = 1102.596313 ;
FTRV('UnSkLab','a_TextWapp','SSA') = 155.1817169 ;
FTRV('UnSkLab','a_TextWapp','RestofWorld') = 995.7570801 ;
FTRV('UnSkLab','a_LightMnfc','Oceania') = 577.7765503 ;
FTRV('UnSkLab','a_LightMnfc','EastAsia') = 22569.13281 ;
FTRV('UnSkLab','a_LightMnfc','SEAsia') = 505.9914551 ;
FTRV('UnSkLab','a_LightMnfc','SouthAsia') = 44.69976044 ;
FTRV('UnSkLab','a_LightMnfc','NAmerica') = 55730.53516 ;
FTRV('UnSkLab','a_LightMnfc','LatinAmer') = 6338.797852 ;
FTRV('UnSkLab','a_LightMnfc','EU_28') = 104245.2344 ;
FTRV('UnSkLab','a_LightMnfc','MENA') = 6091.994141 ;
FTRV('UnSkLab','a_LightMnfc','SSA') = 484.612854 ;
FTRV('UnSkLab','a_LightMnfc','RestofWorld') = 9803.623047 ;
FTRV('UnSkLab','a_HeavyMnfc','Oceania') = 603.7680054 ;
FTRV('UnSkLab','a_HeavyMnfc','EastAsia') = 40115.85938 ;
FTRV('UnSkLab','a_HeavyMnfc','SEAsia') = 827.0192261 ;
FTRV('UnSkLab','a_HeavyMnfc','SouthAsia') = 30.88972473 ;
FTRV('UnSkLab','a_HeavyMnfc','NAmerica') = 73376.86719 ;
FTRV('UnSkLab','a_HeavyMnfc','LatinAmer') = 8010.717773 ;
FTRV('UnSkLab','a_HeavyMnfc','EU_28') = 140325.0312 ;
FTRV('UnSkLab','a_HeavyMnfc','MENA') = 10929.73438 ;
FTRV('UnSkLab','a_HeavyMnfc','SSA') = 589.2912598 ;
FTRV('UnSkLab','a_HeavyMnfc','RestofWorld') = 18934.23047 ;
FTRV('UnSkLab','a_Util_Cons','Oceania') = 2571.065186 ;
FTRV('UnSkLab','a_Util_Cons','EastAsia') = 38222.13281 ;
FTRV('UnSkLab','a_Util_Cons','SEAsia') = 922.4006348 ;
FTRV('UnSkLab','a_Util_Cons','SouthAsia') = 121.4484558 ;
FTRV('UnSkLab','a_Util_Cons','NAmerica') = 98076.45312 ;
FTRV('UnSkLab','a_Util_Cons','LatinAmer') = 20283.50195 ;
FTRV('UnSkLab','a_Util_Cons','EU_28') = 137018.7812 ;
FTRV('UnSkLab','a_Util_Cons','MENA') = 16022.02734 ;
FTRV('UnSkLab','a_Util_Cons','SSA') = 910.4245605 ;
FTRV('UnSkLab','a_Util_Cons','RestofWorld') = 27737.42188 ;
FTRV('UnSkLab','a_TransComm','Oceania') = 4063.921143 ;
FTRV('UnSkLab','a_TransComm','EastAsia') = 83506.07812 ;
FTRV('UnSkLab','a_TransComm','SEAsia') = 1917.952026 ;
FTRV('UnSkLab','a_TransComm','SouthAsia') = 553.2828369 ;
FTRV('UnSkLab','a_TransComm','NAmerica') = 153933.8594 ;
FTRV('UnSkLab','a_TransComm','LatinAmer') = 45585.39453 ;
FTRV('UnSkLab','a_TransComm','EU_28') = 135532.8281 ;
FTRV('UnSkLab','a_TransComm','MENA') = 12606.60547 ;
FTRV('UnSkLab','a_TransComm','SSA') = 4637.98291 ;
FTRV('UnSkLab','a_TransComm','RestofWorld') = 34347.35156 ;
FTRV('UnSkLab','a_OthService','Oceania') = 3868.373779 ;
FTRV('UnSkLab','a_OthService','EastAsia') = 77364.29688 ;
FTRV('UnSkLab','a_OthService','SEAsia') = 1064.144653 ;
FTRV('UnSkLab','a_OthService','SouthAsia') = 143.7115936 ;
FTRV('UnSkLab','a_OthService','NAmerica') = 203375.75 ;
FTRV('UnSkLab','a_OthService','LatinAmer') = 62245.95703 ;
FTRV('UnSkLab','a_OthService','EU_28') = 382906.4062 ;
FTRV('UnSkLab','a_OthService','MENA') = 20309.86133 ;
FTRV('UnSkLab','a_OthService','SSA') = 3813.038574 ;
FTRV('UnSkLab','a_OthService','RestofWorld') = 34099.94531 ;
FTRV('SkLab','a_agricultur','Oceania') = 453.1823425 ;
FTRV('SkLab','a_agricultur','EastAsia') = 596.0796509 ;
FTRV('SkLab','a_agricultur','SEAsia') = 42.14463425 ;
FTRV('SkLab','a_agricultur','SouthAsia') = 6.527122498 ;
FTRV('SkLab','a_agricultur','NAmerica') = 9798.155273 ;
FTRV('SkLab','a_agricultur','LatinAmer') = 3207.183838 ;
FTRV('SkLab','a_agricultur','EU_28') = 16489.53516 ;
FTRV('SkLab','a_agricultur','MENA') = 335.7174072 ;
FTRV('SkLab','a_agricultur','SSA') = 1990.519897 ;
FTRV('SkLab','a_agricultur','RestofWorld') = 1402.205688 ;
FTRV('SkLab','a_Extraction','Oceania') = 483.6216431 ;
FTRV('SkLab','a_Extraction','EastAsia') = 202.89328 ;
FTRV('SkLab','a_Extraction','SEAsia') = 115.423851 ;
FTRV('SkLab','a_Extraction','SouthAsia') = 4.178704262 ;
FTRV('SkLab','a_Extraction','NAmerica') = 5333.935059 ;
FTRV('SkLab','a_Extraction','LatinAmer') = 2227.972656 ;
FTRV('SkLab','a_Extraction','EU_28') = 4696.945801 ;
FTRV('SkLab','a_Extraction','MENA') = 10941.79785 ;
FTRV('SkLab','a_Extraction','SSA') = 1078.130981 ;
FTRV('SkLab','a_Extraction','RestofWorld') = 5356.64209 ;
FTRV('SkLab','a_ProcFood','Oceania') = 284.4375916 ;
FTRV('SkLab','a_ProcFood','EastAsia') = 3057.66626 ;
FTRV('SkLab','a_ProcFood','SEAsia') = 99.05538177 ;
FTRV('SkLab','a_ProcFood','SouthAsia') = 7.118227959 ;
FTRV('SkLab','a_ProcFood','NAmerica') = 7385.848145 ;
FTRV('SkLab','a_ProcFood','LatinAmer') = 5566.977051 ;
FTRV('SkLab','a_ProcFood','EU_28') = 32227.74219 ;
FTRV('SkLab','a_ProcFood','MENA') = 737.1625366 ;
FTRV('SkLab','a_ProcFood','SSA') = 554.8271484 ;
FTRV('SkLab','a_ProcFood','RestofWorld') = 2659.843506 ;
FTRV('SkLab','a_TextWapp','Oceania') = 46.05685425 ;
FTRV('SkLab','a_TextWapp','EastAsia') = 1597.810913 ;
FTRV('SkLab','a_TextWapp','SEAsia') = 51.71103668 ;
FTRV('SkLab','a_TextWapp','SouthAsia') = 43.79872131 ;
FTRV('SkLab','a_TextWapp','NAmerica') = 4079.039551 ;
FTRV('SkLab','a_TextWapp','LatinAmer') = 3700.700684 ;
FTRV('SkLab','a_TextWapp','EU_28') = 14070.23926 ;
FTRV('SkLab','a_TextWapp','MENA') = 384.024353 ;
FTRV('SkLab','a_TextWapp','SSA') = 253.3830872 ;
FTRV('SkLab','a_TextWapp','RestofWorld') = 622.5247803 ;
FTRV('SkLab','a_LightMnfc','Oceania') = 592.8668213 ;
FTRV('SkLab','a_LightMnfc','EastAsia') = 12221.35352 ;
FTRV('SkLab','a_LightMnfc','SEAsia') = 186.1034241 ;
FTRV('SkLab','a_LightMnfc','SouthAsia') = 12.07884121 ;
FTRV('SkLab','a_LightMnfc','NAmerica') = 37329.94141 ;
FTRV('SkLab','a_LightMnfc','LatinAmer') = 12176.94336 ;
FTRV('SkLab','a_LightMnfc','EU_28') = 108778.6562 ;
FTRV('SkLab','a_LightMnfc','MENA') = 1918.279175 ;
FTRV('SkLab','a_LightMnfc','SSA') = 722.2714844 ;
FTRV('SkLab','a_LightMnfc','RestofWorld') = 6970.486816 ;
FTRV('SkLab','a_HeavyMnfc','Oceania') = 619.6509399 ;
FTRV('SkLab','a_HeavyMnfc','EastAsia') = 20887.39258 ;
FTRV('SkLab','a_HeavyMnfc','SEAsia') = 421.7458496 ;
FTRV('SkLab','a_HeavyMnfc','SouthAsia') = 7.079072475 ;
FTRV('SkLab','a_HeavyMnfc','NAmerica') = 48025.59766 ;
FTRV('SkLab','a_HeavyMnfc','LatinAmer') = 15680.84277 ;
FTRV('SkLab','a_HeavyMnfc','EU_28') = 145030.4688 ;
FTRV('SkLab','a_HeavyMnfc','MENA') = 3370.657227 ;
FTRV('SkLab','a_HeavyMnfc','SSA') = 595.8287964 ;
FTRV('SkLab','a_HeavyMnfc','RestofWorld') = 13062.76562 ;
FTRV('SkLab','a_Util_Cons','Oceania') = 1331.566284 ;
FTRV('SkLab','a_Util_Cons','EastAsia') = 10104.55957 ;
FTRV('SkLab','a_Util_Cons','SEAsia') = 278.9020386 ;
FTRV('SkLab','a_Util_Cons','SouthAsia') = 59.41958618 ;
FTRV('SkLab','a_Util_Cons','NAmerica') = 35206.90625 ;
FTRV('SkLab','a_Util_Cons','LatinAmer') = 4097.535156 ;
FTRV('SkLab','a_Util_Cons','EU_28') = 59799.60156 ;
FTRV('SkLab','a_Util_Cons','MENA') = 6014.85791 ;
FTRV('SkLab','a_Util_Cons','SSA') = 523.4368896 ;
FTRV('SkLab','a_Util_Cons','RestofWorld') = 13855.04102 ;
FTRV('SkLab','a_TransComm','Oceania') = 3446.233643 ;
FTRV('SkLab','a_TransComm','EastAsia') = 60894.21484 ;
FTRV('SkLab','a_TransComm','SEAsia') = 576.5505981 ;
FTRV('SkLab','a_TransComm','SouthAsia') = 175.9163208 ;
FTRV('SkLab','a_TransComm','NAmerica') = 121508.8438 ;
FTRV('SkLab','a_TransComm','LatinAmer') = 30509.78516 ;
FTRV('SkLab','a_TransComm','EU_28') = 164664.4844 ;
FTRV('SkLab','a_TransComm','MENA') = 6587.175781 ;
FTRV('SkLab','a_TransComm','SSA') = 6141.036621 ;
FTRV('SkLab','a_TransComm','RestofWorld') = 24178.57422 ;
FTRV('SkLab','a_OthService','Oceania') = 12134.32617 ;
FTRV('SkLab','a_OthService','EastAsia') = 186306.5781 ;
FTRV('SkLab','a_OthService','SEAsia') = 2314.509766 ;
FTRV('SkLab','a_OthService','SouthAsia') = 557.298584 ;
FTRV('SkLab','a_OthService','NAmerica') = 630954.875 ;
FTRV('SkLab','a_OthService','LatinAmer') = 99672.71094 ;
FTRV('SkLab','a_OthService','EU_28') = 876076.0625 ;
FTRV('SkLab','a_OthService','MENA') = 32315.39453 ;
FTRV('SkLab','a_OthService','SSA') = 10348.33398 ;
FTRV('SkLab','a_OthService','RestofWorld') = 116381 ;
FTRV('Capital','a_agricultur','Oceania') = 286.3184204 ;
FTRV('Capital','a_agricultur','EastAsia') = 792.022522 ;
FTRV('Capital','a_agricultur','SEAsia') = 148.4377136 ;
FTRV('Capital','a_agricultur','SouthAsia') = 169.5282288 ;
FTRV('Capital','a_agricultur','NAmerica') = 2551.039062 ;
FTRV('Capital','a_agricultur','LatinAmer') = 1524.828613 ;
FTRV('Capital','a_agricultur','EU_28') = 1996.837524 ;
FTRV('Capital','a_agricultur','MENA') = 595.5737305 ;
FTRV('Capital','a_agricultur','SSA') = 805.5772705 ;
FTRV('Capital','a_agricultur','RestofWorld') = 612.0565186 ;
FTRV('Capital','a_Extraction','Oceania') = 2193.897705 ;
FTRV('Capital','a_Extraction','EastAsia') = 473.6182861 ;
FTRV('Capital','a_Extraction','SEAsia') = 644.7396851 ;
FTRV('Capital','a_Extraction','SouthAsia') = 116.4696655 ;
FTRV('Capital','a_Extraction','NAmerica') = 5581.984375 ;
FTRV('Capital','a_Extraction','LatinAmer') = 1176.349609 ;
FTRV('Capital','a_Extraction','EU_28') = 2081.579834 ;
FTRV('Capital','a_Extraction','MENA') = 6015.748047 ;
FTRV('Capital','a_Extraction','SSA') = 1804.301636 ;
FTRV('Capital','a_Extraction','RestofWorld') = 4125.052246 ;
FTRV('Capital','a_ProcFood','Oceania') = 323.9363098 ;
FTRV('Capital','a_ProcFood','EastAsia') = 1906.652832 ;
FTRV('Capital','a_ProcFood','SEAsia') = 254.1412659 ;
FTRV('Capital','a_ProcFood','SouthAsia') = 74.87587738 ;
FTRV('Capital','a_ProcFood','NAmerica') = 3859.126953 ;
FTRV('Capital','a_ProcFood','LatinAmer') = 703.2719116 ;
FTRV('Capital','a_ProcFood','EU_28') = 3756.424805 ;
FTRV('Capital','a_ProcFood','MENA') = 441.9403381 ;
FTRV('Capital','a_ProcFood','SSA') = 368.5838318 ;
FTRV('Capital','a_ProcFood','RestofWorld') = 718.0250244 ;
FTRV('Capital','a_TextWapp','Oceania') = 64.11643982 ;
FTRV('Capital','a_TextWapp','EastAsia') = 207.7884979 ;
FTRV('Capital','a_TextWapp','SEAsia') = 114.2589874 ;
FTRV('Capital','a_TextWapp','SouthAsia') = 104.4810867 ;
FTRV('Capital','a_TextWapp','NAmerica') = 687.6393433 ;
FTRV('Capital','a_TextWapp','LatinAmer') = 240.8592529 ;
FTRV('Capital','a_TextWapp','EU_28') = 743.5425415 ;
FTRV('Capital','a_TextWapp','MENA') = 341.6202698 ;
FTRV('Capital','a_TextWapp','SSA') = 68.67741394 ;
FTRV('Capital','a_TextWapp','RestofWorld') = 84.0198288 ;
FTRV('Capital','a_LightMnfc','Oceania') = 588.2929077 ;
FTRV('Capital','a_LightMnfc','EastAsia') = 2831.954102 ;
FTRV('Capital','a_LightMnfc','SEAsia') = 366.1488647 ;
FTRV('Capital','a_LightMnfc','SouthAsia') = 49.21936798 ;
FTRV('Capital','a_LightMnfc','NAmerica') = 7448.972656 ;
FTRV('Capital','a_LightMnfc','LatinAmer') = 1076.227173 ;
FTRV('Capital','a_LightMnfc','EU_28') = 6649.666992 ;
FTRV('Capital','a_LightMnfc','MENA') = 953.2304077 ;
FTRV('Capital','a_LightMnfc','SSA') = 422.0382996 ;
FTRV('Capital','a_LightMnfc','RestofWorld') = 533.8701172 ;
FTRV('Capital','a_HeavyMnfc','Oceania') = 679.7540283 ;
FTRV('Capital','a_HeavyMnfc','EastAsia') = 7067.104492 ;
FTRV('Capital','a_HeavyMnfc','SEAsia') = 1059.540039 ;
FTRV('Capital','a_HeavyMnfc','SouthAsia') = 85.96186066 ;
FTRV('Capital','a_HeavyMnfc','NAmerica') = 13053.95801 ;
FTRV('Capital','a_HeavyMnfc','LatinAmer') = 1783.894897 ;
FTRV('Capital','a_HeavyMnfc','EU_28') = 9911.777344 ;
FTRV('Capital','a_HeavyMnfc','MENA') = 3805.635986 ;
FTRV('Capital','a_HeavyMnfc','SSA') = 767.6757812 ;
FTRV('Capital','a_HeavyMnfc','RestofWorld') = 2249.067871 ;
FTRV('Capital','a_Util_Cons','Oceania') = 875.2405396 ;
FTRV('Capital','a_Util_Cons','EastAsia') = 3883.273438 ;
FTRV('Capital','a_Util_Cons','SEAsia') = 433.0782471 ;
FTRV('Capital','a_Util_Cons','SouthAsia') = 332.7077637 ;
FTRV('Capital','a_Util_Cons','NAmerica') = 11012.30762 ;
FTRV('Capital','a_Util_Cons','LatinAmer') = 2121.008057 ;
FTRV('Capital','a_Util_Cons','EU_28') = 12207.31836 ;
FTRV('Capital','a_Util_Cons','MENA') = 1863.912109 ;
FTRV('Capital','a_Util_Cons','SSA') = 842.5669556 ;
FTRV('Capital','a_Util_Cons','RestofWorld') = 2950.96582 ;
FTRV('Capital','a_TransComm','Oceania') = 1895.181885 ;
FTRV('Capital','a_TransComm','EastAsia') = 17360.62305 ;
FTRV('Capital','a_TransComm','SEAsia') = 1302.374023 ;
FTRV('Capital','a_TransComm','SouthAsia') = 433.2952576 ;
FTRV('Capital','a_TransComm','NAmerica') = 26447.83203 ;
FTRV('Capital','a_TransComm','LatinAmer') = 5011.879395 ;
FTRV('Capital','a_TransComm','EU_28') = 17531.31055 ;
FTRV('Capital','a_TransComm','MENA') = 3287.469238 ;
FTRV('Capital','a_TransComm','SSA') = 1448.446777 ;
FTRV('Capital','a_TransComm','RestofWorld') = 9267.255859 ;
FTRV('Capital','a_OthService','Oceania') = 7796.350098 ;
FTRV('Capital','a_OthService','EastAsia') = 45844.52734 ;
FTRV('Capital','a_OthService','SEAsia') = 1735.806274 ;
FTRV('Capital','a_OthService','SouthAsia') = 371.5693359 ;
FTRV('Capital','a_OthService','NAmerica') = 80798.44531 ;
FTRV('Capital','a_OthService','LatinAmer') = 8014.954102 ;
FTRV('Capital','a_OthService','EU_28') = 68108.4375 ;
FTRV('Capital','a_OthService','MENA') = 5905.07959 ;
FTRV('Capital','a_OthService','SSA') = 2169.860596 ;
FTRV('Capital','a_OthService','RestofWorld') = 8138.001953 ;
FTRV('NatRes','a_Extraction','Oceania') = 840.9474487 ;
FTRV('NatRes','a_Extraction','EastAsia') = 291.8811951 ;
FTRV('NatRes','a_Extraction','SEAsia') = 333.4763794 ;
FTRV('NatRes','a_Extraction','SouthAsia') = 60.75436783 ;
FTRV('NatRes','a_Extraction','NAmerica') = 3992.848145 ;
FTRV('NatRes','a_Extraction','LatinAmer') = 680.1818848 ;
FTRV('NatRes','a_Extraction','EU_28') = 1164.350952 ;
FTRV('NatRes','a_Extraction','MENA') = 3379.614014 ;
FTRV('NatRes','a_Extraction','SSA') = 1138.126587 ;
FTRV('NatRes','a_Extraction','RestofWorld') = 2977.018311 ;

* --------------------------------------------------------------------------------------------------
*
*     Declare and initialize the model parameters--overrides can be inserted before 'cal.gms'
*
* --------------------------------------------------------------------------------------------------

Parameters

*  Parameters normally sourced from GTAP

   sigmap(r,a)       "Top level CES production elasticity (ND/VA)"
   sigmand(r,a)      "CES elasticity across intermediate inputs"
   sigmav(r,a)       "CES elasticity across factors of production"

   omegas(r,a)       "Commodity supply CET elasticity"
   sigmas(r,i)       "Commodity supply CES elasticity"

   eh0(r,i)          "CDE expansion parameter"
   bh0(r,i)          "CDE substitution parameter"

   sigmag(r)         "CES government expenditure elasticity"
   sigmai(r)         "CES investment expenditure elasticity"

   sigmam(r,i,aa)    "Top level Armington elasticity"
   sigmaw(r,i)       "Second level Armington elasticity"
   sigmamg(i)        "CES expenditure elasticity for margin services exports"

   omegaf(r,fp)      "CET mobility elasticity for mobile factors"
   rorFlex(r,t)      "Flexibility of foreign capital"

*  Parameters in addition to standard GTAP model

   omegax(r,i)       "Top level output CET elasticity"
   omegaw(r,i)       "Second level export CET elasticity"

   etaf(r,fp)        "Aggregate factor supply elasticity"
   etaff(r,fp,a)     "Sector specific supply elasticity for non-mobile factors"

   mdtx0(r)          "Initial marginal tax rate"
   RoRFlag           "Capital account closure flag"
;

*  Overrides for GTAP-based parameters
*  If no overrides, parameters will be set in 'cal.gms'

sigmap(r,a)    = na ;
sigmand(r,a)   = na ;
sigmav(r,a)    = na ;

*  !!!! Explicitly assumes that these are not aggregated

omegas(r,a)    = -etraq(a, r) ;
sigmas(r,i)    = inf$(esubq(i,r) eq 0)
               + (1/esubq(i,r))$(esubq(i,r) ne 0)
               ;

eh0(r,i)       = na ;
bh0(r,i)       = na ;

sigmag(r)      = na ;
sigmai(r)      = na ;

sigmam(r,i,aa) = na ;
sigmaw(r,i)    = na ;
sigmamg(i)     = na ;

loop(fm,
   loop(endwm,
      omegaf(r,fm)$sameas(endwm,fm) = inf ;
   ) ;
   loop(endws,
      omegaf(r,fm)$sameas(endws,fm) = na ;
   ) ;
) ;
rorFlex(r,t)   = rorFlex0(r) ;

*  Other initialization -- use default GTAP assumptions

omegax(r,i)   = inf ;
omegaw(r,i)   = inf ;

etaf(r,fm)    = 0 ;
etaff(r,fp,a) = 0 ;

mdtx0(r)      = na ;

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
   RestofWorld
/ ;

set rmuv(r) "RMUV regions" /
   Oceania, EastAsia, SEAsia, SouthAsia, NAmerica, LatinAmer, EU_28, MENA, SSA, RestofWorld
/ ;

set imuv(i) "IMUV commodities" /
   c_Crops, c_MeatLstk, c_Extraction, c_ProcFood, c_TextWapp, c_LightMnfc, c_HeavyMnfc, c_Util_Cons, c_TransComm, c_OthService
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
acronym CDE, CD, capFlex, capshrFix, capFix, capSFix ;

set rs(r) "Simulation regions" ;

alias(is,js) ; alias(r,rp) ; alias(i,j) ; alias(j,jp) ; alias(m,i) ; alias(i0,j0) ; alias(m0,i0) ;

$macro M_PP(r,a,i,t)        ((p(r,a,i,t)*(1 + prdtx(r,a,i,t)))$ifSUB + (pp(r,a,i,t))$(not ifSUB))
$macro M_PDP(r,i,aa,t)      ((pd(r,i,t)*(1 + dintx(r,i,aa,t)))$ifSUB + (pdp(r,i,aa,t))$(not ifSUB))
$macro M_PMP(r,i,aa,t)      ((pmt(r,i,t)*(1 + mintx(r,i,aa,t)))$ifSUB + (pmp(r,i,aa,t))$(not ifSUB))
$macro M_XWMG(r,i,rp,t)     ((tmarg(r,i,rp,t)*xw(r,i,rp,t))$ifSUB + (xwmg(r,i,rp,t))$(not ifSUB))
$macro M_XMGM(m,r,i,rp,t)   ((amgm(m,r,i,rp)*M_XWMG(r,i,rp,t)/lambdamg(m,r,i,rp,t))$ifSUB + (xmgm(m,r,i,rp,t))$(not ifSUB))
$macro M_PWMG(r,i,rp,t)     ((sum(m, amgm(m,r,i,rp)*ptmg(m,t)/lambdamg(m,r,i,rp,t)))$ifSUB + (pwmg(r,i,rp,t))$(not ifSUB))
$macro M_PEFOB(r,i,rp,t)    (((1 + exptx(r,i,rp,t) + etax(r,i,t))*pe(r,i,rp,t))$ifSUB + (pefob(r,i,rp,t))$(not ifSUB))
$macro M_PMCIF(r,i,rp,t)    ((M_PEFOB(r,i,rp,t) + M_PWMG(r,i,rp,t)*tmarg(r,i,rp,t))$ifSUB + (pmcif(r,i,rp,t))$(not ifSUB))
$macro M_PM(r,i,rp,t)       (((1 + imptx(r,i,rp,t) + mtax(rp,i,t))*M_PMCIF(r,i,rp,t)/chipm(r,i,rp))$ifSUB + (pm(r,i,rp,t))$(not ifSUB))
$macro M_PFA(r,fp,a,t)      ((pf(r,fp,a,t)*(1 + fctts(r,fp,a,t) + fcttx(r,fp,a,t)))$ifSUB + (pfa(r,fp,a,t))$(not ifSUB))
$macro M_PFY(r,fp,a,t)      ((pf(r,fp,a,t)*(1 - kappaf(r,fp,a,t)))$(ifSUB) + (pfy(r,fp,a,t))$(not ifSUB))

sets
   gy Government tax stream revenues /
      pt       "Tax revenues from output taxes"
      fc       "Indirect tax revenues from firm consumption"
      pc       "Indirect tax revenues from private consumption"
      gc       "Indirect tax revenues from government consumption"
      ic       "Indirect tax revenues from investment consumption"
      dt       "Direct tax on factor income"
      mt       "Import tax revenues"
      et       "Export tax revenues"
      ft       "Tax revenues from factor taxes"
      fs       "Revenue costs of subsidies"
   /
;


Variables
   axp(r,a,t)           "Production frontier shifter"
   lambdand(r,a,t)      "Shifter for ND bundle"
   lambdava(r,a,t)      "Shifter for VA bundle"
   nd(r,a,t)            "Demand for aggregate intermediate (ND)"
   va(r,a,t)            "Demand for aggregate value added  (VA)"
   px(r,a,t)            "Unit cost of production"

   lambdaio(r,i,a,t)    "Shifter for intermediate demand"
   pnd(r,a,t)           "Price of ND bundle"

   lambdaf(r,fp,a,t)    "Factor specific technical change"
   xf(r,fp,a,t)         "Factor demand"
   pva(r,a,t)           "Price of value added"

   ytax(r,gy,t)         "Government tax stream revenues"
   ytaxTot(r,t)         "Total government revenues"
   ytaxInd(r,t)         "Total revenues from indirect taxes"
   factY(r,t)           "Factor income net of depreciation"
   regY(r,t)            "Regional income"

*  Allocation of national income

   phiP(r,t)            "Elasticity of private exp. wrt to private utility"
   phi(r,t)             "Elasticity of total exp. wrt to total utility"
   yc(r,t)              "Nominal private expenditure on goods and services"
   yg(r,t)              "Nominal government expenditures"
   rsav(r,t)            "Regional savings"
   uh(r,h,t)            "Private utility per capita"
   ug(r,t)              "Utility per capita from public expenditure"
   us(r,t)              "Utility per capita from savings"
   u(r,t)               "Total per capita utility"

*  Private consumption

   zcons(r,i,h,t)       "Auxiliary consumption variable"
   xcshr(r,i,h,t)       "Household budget shares"
   pcons(r,t)           "Consumer expenditure deflator"

   pg(r,t)              "Public expenditure price deflator"
   xg(r,t)              "Aggregate volume of government expenditures"

   pi(r,t)              "Investment expenditure price deflator"
   yi(r,t)              "Nominal Investment expenditures"
   xi(r,t)              "Aggregate volume of Investment expenditures"

   xa(r,i,aa,t)         "Demand for Armington good"
   xd(r,i,aa,t)         "Demand for domestic goods"
   xm(r,i,aa,t)         "Demand for imported goods"
   pdp(r,i,aa,t)        "Purchasers price of domestic goods"
   pmp(r,i,aa,t)        "Purchasers price of imported goods"
   pa(r,i,aa,t)         "Agents price of Armington good"

   xmt(r,i,t)           "Aggregate import demand"
   xw(r,i,rp,t)         "Bilateral demand for imports"
   pmt(r,i,t)           "Price of aggregate imports"

   xds(r,i,t)           "Supply of domestically produced goods"
   xet(r,i,t)           "Aggregate export supply"
   xp(r,a,t)            "Domestic output"
   xs(r,i,t)            "Domestic supply"
   ps(r,i,t)            "Price of domestic supply"
   x(r,a,i,t)           "Supply of commodity 'i' by activity 'a'"
   p(r,a,i,t)           "Pre-tax price of X"
   pp(r,a,i,t)          "Post-tax price of X"
   pe(r,i,rp,t)         "Bilateral export supply"
   pet(r,i,t)           "Price of aggregate exports"

   xwmg(r,i,rp,t)       "Demand for international trade and transport services"
   xmgm(m,r,i,rp,t)     "Demand for TT services by mode"
   pwmg(r,i,rp,t)       "Average price of TT services by node"
   xtmg(m,t)            "Global demand for TT services by mode"
   ptmg(m,t)            "Global price index of TT services by mode"

   pefob(r,i,rp,t)      "Border price of exports"
   pmcif(r,i,rp,t)      "Border price of imports"
   pm(r,i,rp,t)         "Bilateral price of imports, tariff-inclusive"

   pd(r,i,t)            "Supply of domestically produced goods"

   xft(r,fp,t)          "Aggregate supply of mobile factors"
   pf(r,fp,a,t)         "Factor price tax exclusive"
   pft(r,fp,t)          "Aggregate price of mobile factors"
   pfa(r,fp,a,t)        "Factor prices tax inclusive"
   pfy(r,fp,a,t)        "After tax factor prices"

*  Allocation of global savings

   arent(r,t)           "Aggregate rate of return to capital after tax"
   kapEnd(r,t)          "End of period capital stock"
   rorc(r,t)            "Net rate of return to capital"
   rore(r,t)            "Expected rate of return to capital"
   rorg(t)              "Global rate of return"
   chiSave(t)           "Global adjustment factor for price of savings"
   psave(r,t)           "Regional price of savings"
   xigbl(t)             "Global net investment"
   pigbl(t)             "Global price of investment"
   chiInv(r,t)          "Regional share of global net investment"
   chif(r,t)            "Share of nominal foreign savings in regional income"
   savf(r,t)            "Real foreign savings"

*  Prices and exchange rates

   pabs(r,t)            "Price of aggregate domestic absorption"
   pmuv(t)              "Price of HIC manufactured exports"
   pfact(r,t)           "Regional factor price index"
   pwfact(t)            "World factor price index"
   pnum(t)              "Model numeraire"

   walras               "Walras check"

*  Closure variables

   dintx(r,i,aa,t)      "Indirect taxes on consumption of domestic goods"
   mintx(r,i,aa,t)      "Indirect taxes on consumption of import goods"
   ytaxshr(r,gy,t)      "Indirect tax revenues as share of regional income"

   gdpmp(r,t)           "Nominal GDP at market price"
   rgdpmp(r,t)          "Real GDP at market price"
   pgdpmp(r,t)          "GDP price deflator"
   rgdppc(r,t)          "Per capita income"
   rgdppcgr(r,t)        "Change in per capita income"

*  Policy variables

   fctts(r,fp,a,t)      "Subsidies on factors of production"
   fcttx(r,fp,a,t)      "Taxes on factors of production"
   prdtx(r,a,i,t)       "Production tax"
   dtxshft(r,i,aa,t)    "Shifter on domestic indirect taxes"
   mtxshft(r,i,aa,t)    "Shifter on imported indirect taxes"
   rtxshft(r,aa,t)      "Uniform shifter on indirect taxes"
   kappaf(r,fp,a,t)     "Income tax on factor f used in activity a"
   exptx(r,i,rp,t)      "Bilateral export taxes"
   etax(r,i,t)          "Export tax shifter across destinations"
   imptx(r,i,rp,t)      "Bilateral import taxes"
   mtax(r,i,t)          "Import tax shifter uniform across sources"
   adtx(r,t)            "Direct tax schedule intercept"
   mdtx(r,t)            "Marginal rate of direct taxation"

   emid(r,i,aa,t)       "CO2 emissions from consumption of domestic goods"
   emim(r,i,aa,t)       "CO2 emissions from consumption of imported goods"

*  Technical variables

   axpsec(a,t)          "World-wide shift in production by sector"
   axpreg(r,t)          "Region-wide shift in production across sectors"
   axpall(r,a,t)        "Region and sector specific shift in production"

   andsec(a,t)          "World-wide technical shift in ND demand by sector"
   andreg(r,t)          "Region-wide technical shift in ND demand across sectors"
   andall(r,a,t)        "Region and sector specific technical shift in ND demand"

   avasec(a,t)          "World-wide technical shift in VA demand by sector"
   avareg(r,t)          "Region-wide technical shift in VA demand across sectors"
   avaall(r,a,t)        "Region and sector specific technical shift in VA demand"

   aiocom(i,t)          "World-wide technical shift in IO demand by input"
   aiosec(a,t)          "World-wide technical shift in IO demand by activity"
   aioreg(r,t)          "Region-wide tech. shift in IO demand across inputs/activities"
   aioall(r,i,a,t)      "Region/input/activity specific technical shift in IO demand"

   afecom(fp,t)         "World-wide technical shift in VA demand by factor"
   afesec(a,t)          "World-wide technical shift in VA demand by activity"
   afefac(r,fp,t)       "Region-wide tech. shift in VA demand across factors"
   afeLab(r,t)          "Region-wide labor augmenting technical shift"
   afereg(r,t)          "Region-wide tech. shift in VA demand across inputs/activities"
   afeall(r,fp,a,t)     "Region/factor/activity specific technical shift in VA demand"

   atm(m,t)             "Global tech change for mode m"
   atf(i,t)             "Global tech change for transporting good i"
   ats(r,t)             "Global tech change for transporting from region r"
   atd(r,t)             "Global tech change for transporting to region r"
   atall(m,r,i,rp,t)    "Global tech change for transporint i from r to rp using mode m"

   lambdai(r,i,t)       "Technology changes in investment expenditure function"

   lambdam(rp,i,r,t)    "Change in second-level Armington preferences"

   lambdamg(m,r,i,rp,t) "Technical change in TT demand"

   tmarg(r,i,rp,t)      "International trade and transport margin"

*  Top level utility parameters

   betap(r,t)           "Private consumption share coefficient"
   betag(r,t)           "Public consumption share coefficient"
   betas(r,t)           "Savings share coefficients"

   aug(r,t)             "Public expenditure utility shifter"
   aus(r,t)             "Savings expenditure utility shifter"
   au(r,t)              "Aggregate utility shifter"

*  Consumer demand

   bh(r,i,t)            "CDE substitution parameters"
   eh(r,i,t)            "CDE expansion parameters"
   ued(r,i,j,t)         "Uncompensated price elasticities"
   ced(r,i,j,t)         "Compensated price elasticities"
   ape(r,i,j,t)         "Allen-Uzawa price elasticities"
   incelas(r,i,t)       "Income elasticities"

*  Other

   kstock(r,t)          "Beginning of period capital stock"
   pop(r,t)             "Population"

   ev(r,h,t)            "Equivalent variation"
   cv(r,h,t)            "Compensating variation"

   gl(r,t)              "Economy-wide labor productivity parameter"
   ggdppc(r,t)          "Growth of real GDP per capita"
;

Parameters
   xscale(r,aa)         "Scale factor for IO columns"
   and(r,a,t)           "ND bundle share parameter"
   ava(r,a,t)           "VA bundle share parameter"
   sigmap(r,a)          "CES elasticity between ND and VA"

   io(r,i,a,t)          "Input output coefficient wrt ND"
   sigmand(r,a)         "CES elasticity across intermediate demand"

   af(r,fp,a,t)         "Factor shares wrt VA"
   sigmav(r,a)          "CES elasticity across factors"

   gx(r,a,i,t)          "CET share parameter for commodity supply"
   omegas(r,a)          "CET elasticity for commodity supply"

   ax(r,a,i,t)          "CES share parameter for commodity demand"
   sigmas(r,i)          "CES elasticity for commodity demand"

   auh(r,h,t)           "Utility shifter for Cobb-Douglas"

   axg(r,t)             "CES aggregate shifter for government expenditures"
   sigmag(r)            "CES elasticity in government expenditures"
   axi(r,t)             "CES aggregate shifter for investment expenditures"
   sigmai(r)            "CES elasticity in investment"

   alphaa(r,i,aa,t)     "Armington demand shift parameter"
   alphad(r,i,aa,t)     "Armington demand domestic shift parameter"
   alpham(r,i,aa,t)     "Armington demand import shift parameter"
   sigmam(r,i,aa)       "Top level Armington elasticity"

   amw(r,i,rp,t)        "Import share parameters by region of origin"
   sigmaw(r,i)          "Second level Armington elasticity"
   chipm(r,i,rp)        "Import price normalization factor"

   gd(r,i,t)            "Domestic CET share parameter"
   ge(r,i,t)            "Export CET share parameter"
   omegax(r,i)          "Top level CET elasticity"
   gw(r,i,rp,t)         "Export share parameters by region of destination"
   omegaw(r,i)          "Second level CET elasticity"

   amgm(m,r,i,rp)       "Share parameter for demand for TT services by mode"

   aft(r,fp,t)          "Aggregate factor supply shifter"
   etaf(r,fp)           "Aggregate factor supply elasticity"

   gf(r,fp,a,t)         "Sector supply share/shift parameter"
   omegaf(r,fp)         "CET elasticity of factor supply across sectors"
   etaff(r,fp,a)        "Sector specific factor supply elasticity"

   RoRflex(r,t)         "Flexibility of expected net ROR"
   fdepr(r,t)           "Fiscal depreciation rate"
   depr(r,t)            "Physical depreciation rate"
   risk(r,t)            "Regional risk factor"
   krat(r,t)            "Ratio of normalized capital stock & non-normalized capital stock"

   sigmamg(m)           "TT elasticity across suppliers"
   axmg(m,t)            "TT shift parameter"

   savfBar(r,t)         "Exogenous 'real' foreign savings flow"
   ggdppcT(r,t)         "Exogenous rate of growth of per capita GDP"

   invwgt(r,t)          "Regional share of global investment"
   savwgt(r,t)          "Regional share of global savings"

   piadd(r,l,a,t)       "Labor productivity additive sectoral shifter"
   pimlt(r,l,a,t)       "Labor productivity multiplicative sectoral shifter"

   dintx0(r,i,aa)       "Base year indirect tax on domestic consumption"
   mintx0(r,i,aa)       "Base year indirect tax on imported consumption"

   work                 "A working scalar"
   rwork(r)             "A working vector for regions"
   kron(is,js)          "Kronecker delta"
;

Parameters
   ndFlag(r,a)          "ND flags"
   vaFlag(r,a)          "VA flags"
   xpFlag(r,a)          "XP flags"
   xsFlag(r,i)          "XS flags"
   xFlag(r,a,i)         "X flags"

   xaFlag(r,i,aa)       "XA flags"
   xfFlag(r,fp,a)       "XF flags"
   xftFlag(r,fp)        "XFT flags"

   fdFlag(r,fd)         "FD flags"

   xatFlag(r,i)         "XAT flags"
   xdFlag(r,i)          "XD flags"
   xmtFlag(r,i)         "XMT flags"
   xetFlag(r,i)         "XET flags"
   xwFlag(r,i,rp)       "XW flag"
   tmgFlag(r,i,rp)      "TT flag"
   mFlag(m)             "Margin flags"
   RoRFlag              "Type of foreign savings closure"
   intxFlag(r,i,aa)     "Set to 1 for endogenous indirect sales tax, else to 0"
   afeFlag(r,fp)        "Factor or Hicks neutral tech change"
;

Equations

*  Top production nest

   ndeq(r,a,t)          "Demand for aggregate intermediate (ND)"
   vaeq(r,a,t)          "Demand for aggregate value added  (VA)"
   pxeq(r,a,t)          "Unit cost of production"

*  Intermediate demand nest

   pndeq(r,a,t)         "Price of ND bundle"
   xapeq(r,i,a,t)       "Intermediate demand food goods and services"

*  Value added demand nest

   xfeq(r,fp,a,t)       "Factor demand"
   pvaeq(r,a,t)         "Price of value added"

*  Commodity supply/demand

   xeq(r,a,i,t)         "Supply of commodity 'i' by activity 'a'"
   xpeq(r,a,t)          "Aggregate production by activity 'a'"
   ppeq(r,a,i,t)        "Post tax price of X"
   peq(r,a,i,t)         "Pre-tax price of X"
   pseq(r,i,t)          "Total domestic supply of commodity 'i'"

*  Income distribution

   ytaxeq(r,gy,t)       "Government tax revenues by stream"
   ytaxToteq(r,t)       "Total government tax revenues"
   ytaxIndeq(r,t)       "Total revenues from indirect taxes"
   factYeq(r,t)         "Factor income net of depreciation"
   regYeq(r,t)          "Regional income"

*  Top level regional expenditure decisions

   phiPeq(r,t)          "Elast. of priv. exp. wrt to priv. utility"
   phieq(r,t)           "Elast. of total exp. wrt to total utility"
   yceq(r,t)            "Determination of nominal private consumption"
   ygeq(r,t)            "Determination of government nominal expenditures"
   rsaveq(r,t)          "Determination of national savings"
   uheq(r,h,t)          "Private utility per capita"
   ugeq(r,t)            "Per capita utility from public spending"
   useq(r,t)            "Per capita utility from savings"
   ueq(r,t)             "Total utility"

*  Private demand

   zconseq(r,i,h,t)     "Auxiliary consumption variable"
   xcshreq(r,i,h,t)     "Household budget share"
   xaceq(r,i,h,t)       "Private demand for goods and services"
   pconseq(r,t)         "Household aggregate expenditure deflator"

*  Public demand

   xageq(r,i,gov,t)     "Public expenditure on Armington goods and services"
   pgeq(r,t)            "Public expenditure price deflator"
   xgeq(r,t)            "Real government expenditure"

*  Investment demand

   xaieq(r,i,inv,t)     "Investment expenditure on Armington goods and services"
   pieq(r,t)            "Investment expenditure price deflator"
   xieq(r,t)            "Nominal Investment expenditure"

*  Top level Armington demand

   pdpeq(r,i,aa,t)      "Agents price of domestic goods"
   pmpeq(r,i,aa,t)      "Agents price of import goods"
   paeq(r,i,aa,t)       "Agents composite (or Armington) price"
   xdeq(r,i,aa,t)       "Agents demand for domestic goods"
   xmeq(r,i,aa,t)       "Agents demand for import goods"

*  Second level Armington nest

   xmteq(r,i,t)         "Aggregate import demand"
   xweq(r,i,rp,t)       "Bilateral demand for imports"
   pmteq(r,i,t)         "Price of aggregate imports"

*  Allocation of domestic production

   xdseq(r,i,t)         "Supply of for domestically produced goods"
   xeteq(r,i,t)         "Aggregate export supply"
   xpeq(r,a,t)          "Domestic output"
   xeq(r,a,i,t)         "Domestic output of 'i' by activity 'a'"
   xseq(r,i,t)          "Domestic supply of good 'i'"
   peq(r,a,i,t)         "Price of X"
   peeq(r,i,rp,t)       "Bilateral export supply"
   peteq(r,i,t)         "Price of aggregate exports"

*  TT services

   xwmgeq(r,i,rp,t)     "Demand for international trade and transport services by node"
   xmgmeq(m,r,i,rp,t)   "Demand for TT services by mode and node"
   pwmgeq(r,i,rp,t)     "Price index of TT services by node"
   xtmgeq(m,t)          "Global demand for TT services by mode"
   xatmgeq(r,m,tmg,t)   "Regional supply of TT services by mode"
   ptmgeq(m,t)          "Global price index of TT services by mode"

*  Bilateral price relations

   pefobeq(r,i,rp,t)    "Border price of exports"
   pmcifeq(r,i,rp,t)    "Border price of imports"
   pmeq(r,i,rp,t)       "Bilateral price of imports, tariff-inclusive"

*  Commodity equilibrium

   pdeq(r,i,t)          "Price of domestically produced goods"
*  peeq(r,i,rp,t)       "Equilibrium for bilateral trade (substituted out)"

*  Factor supply and allocation

   xfteq(r,fp,t)        "Aggregate supply of mobile factors"
   pfeq(r,fp,a,t)       "Factor price tax exclusive"
   pfteq(r,fp,t)        "Aggregate price of mobile factors"
   pfaeq(r,fp,a,t)      "Factor prices pre-tax/subsdidies"
   pfyeq(r,fp,a,t)      "Factor prices post-tax/subsdidies"
   kstockeq(r,t)        "Beginning of period capital stock"

*  Investment determination and its allocation

   arenteq(r,t)         "Aggregate rate of return"
   kapEndeq(r,t)        "End of period capital stock"
   rorceq(r,t)          "Net rate of return to capital"
   roreeq(r,t)          "Expected rate of return to capital"
   yieq(r,t)            "Gross investment by region"
   savfeq(r,t)          "Determination of foreign savings"
   rorgeq(t)            "Global rate of return"
   chifeq(r,t)          "Share of nominal foreign savings in regional income"
   capAccteq(t)         "Capital account consistency"
   chiSaveeq(t)         "Global adjustment factor for regional price of savings"
   psaveeq(r,t)         "Price of savings"
   xigbleq(t)           "Global net investment"
   pigbleq(t)           "Price of global net investment"

*  Price indices and numeraire

   pabseq(r,t)          "Aggregate price of domestic absorption"
   pmuveq(t)            "Price index of HIC manufactured exports"
   pfacteq(r,t)         "Regional factor price index"
   pwfacteq(t)          "World factor price index"
   pnumeq(t)            "Definition of numeraire"
   walraseq             "Walras check"

*  Closure equations

   dintxeq(r,i,aa,t)    "Indirect taxes on domestic consumption"
   mintxeq(r,i,aa,t)    "Direct taxes on import consumption"
   ytaxshreq(r,gy,t)    "Indirect tax revenues as share of regional income"

   gdpmpeq(r,t)         "Nominal GDP at market price"
   rgdpmpeq(r,t)        "Real GDP at market price"
   pgdpmpeq(r,t)        "GDP price deflator"

*  Technology equations

   axpeq(r,a,t)         "Production frontier shifter"
   lambdandeq(r,a,t)    "Tech shifter for ND bundle"
   lambdavaeq(r,a,t)    "Tech shifter for VA bundle"
   lambdaioeq(r,i,a,t)  "Tech shifter for IO demand"
   lambdafeq(r,fp,a,t)  "Factor specific technical change"

   glcaleq(r,t)         "Growth of real GDP per capita"
   afealleq(r,fp,a,t)   "Growth of labor productivity"

*  Other

   uedeq(r,i,j,h,t)     "Uncompensated price elasticities"
   incelaseq(r,i,h,t)   "Income elasticities"
   cedeq(r,i,j,h,t)     "Compensated price elasticities"
   apeeq(r,i,j,h,t)     "Allen-Uzawa price elasticities"

   eveq(r,h,t)          "Equivalent variation"
   cveq(r,h,t)          "Compensating variation"
;

*  -------------------------------------------------------------------------------------------------
*
*     Section 1 -- Production
*
*  -------------------------------------------------------------------------------------------------

*  Top level nest -- (E_qint (983), no equivalent)

ndeq(r,a,t)$(rs(r) and ts(t) and ndFlag(r,a))..
   nd(r,a,t) =e= and(r,a,t)*xp(r,a,t)*(px(r,a,t)/pnd(r,a,t))**sigmap(r,a)
              *  (axp(r,a,t)*lambdand(r,a,t))**(sigmap(r,a)-1) ;

*  Demand for aggregate value added -- (E_qva (997), VADEMAND (1262))

vaeq(r,a,t)$(rs(r) and ts(t) and vaFlag(r,a))..
   va(r,a,t) =e= ava(r,a,t)*xp(r,a,t)*(px(r,a,t)/pva(r,a,t))**sigmap(r,a)
              *  (axp(r,a,t)*lambdava(r,a,t))**(sigmap(r,a)-1) ;

*  Unit cost definition (also zero profit condition) -- (E_qo (1027), ZEROPROFITS (1464))

pxeq(r,a,t)$(rs(r) and ts(t) and xpFlag(r,a))..
   px(r,a,t)**(1-sigmap(r,a)) =e= (axp(r,a,t)**(sigmap(r,a)-1))
      *  (and(r,a,t)*(pnd(r,a,t)/lambdand(r,a,t))**(1-sigmap(r,a))
      +   ava(r,a,t)*(pva(r,a,t)/lambdava(r,a,t))**(1-sigmap(r,a))) ;

*  Intermediate demand nest

*  Demand for intermediates -- (E_qfa (1052), ~INTDEMAND (1286))

xapeq(r,i,a,t)$(rs(r) and ts(t) and xaFlag(r,i,a))..
   xa(r,i,a,t) =e= io(r,i,a,t)*nd(r,a,t)*(pnd(r,a,t)/pa(r,i,a,t))**sigmand(r,a)
                *  lambdaio(r,i,a,t)**(sigmand(r,a)-1) ;

*  Price if ND bundle -- (E_pint (1071), no equivalent)

pndeq(r,a,t)$(rs(r) and ts(t) and ndFlag(r,a))..
   pnd(r,a,t)**(1-sigmand(r,a)) =e= sum(i$io(r,i,a,t), io(r,i,a,t)
      *  (pa(r,i,a,t)/lambdaio(r,i,a,t))**(1-sigmand(r,a))) ;

*  Value added decomposition

*  Demand for value added -- (E_qfe (1103), ENDWDEMAND (1423))

xfeq(r,fp,a,t)$(rs(r) and ts(t) and xfFlag(r,fp,a))..
   xf(r,fp,a,t) =e= af(r,fp,a,t)*va(r,a,t)*(pva(r,a,t)/M_PFA(r,fp,a,t))**sigmav(r,a)
                *  (lambdaf(r,fp,a,t))**(sigmav(r,a)-1) ;

*  Price of value added bundle -- (E_pva (1110), VAPRICE (1403))

pvaeq(r,a,t)$(rs(r) and ts(t) and vaFlag(r,a))..
   pva(r,a,t)**(1-sigmav(r,a)) =e=
      sum(fp, af(r,fp,a,t)*(M_PFA(r,fp,a,t)/lambdaf(r,fp,a,t))**(1-sigmav(r,a))) ;

*  Sourcing of commodities by firm (see below xdeq, xmeq, paeq -- one set of Armington equations)
*  Consolidates E_qfd, E_qfm and E_pfa

*  -------------------------------------------------------------------------------------------------
*
*     Section 2 -- Commodity supply
*
*  -------------------------------------------------------------------------------------------------

*  Convert output into commodities -- (E_qca (1253), no equivalent)
*  N.B. The GAMS version allows for perfect transformation

xeq(r,a,i,t)$(rs(r) and ts(t) and xFlag(r,a,i))..
   0 =e= (x(r,a,i,t) - gx(r,a,i,t)*(xp(r,a,t)/xscale(r,a))
      *  (p(r,a,i,t)/px(r,a,t))**omegas(r,a))$(omegas(r,a) ne inf)
      +  (p(r,a,i,t) - px(r,a,t))$(omegas(r,a) eq inf)
      ;

*  Zero profit for make 'CET' -- (E_po (1259), no equivalent)

xpeq(r,a,t)$(rs(r) and ts(t) and xpFlag(r,a))..
   0 =e= (xp(r,a,t)/xscale(r,a) - sum(i$xFlag(r,a,i), x(r,a,i,t)))
      $  (omegas(r,a) eq inf)
      +  (px(r,a,t)**(1+omegas(r,a))
      -   sum(i$xFlag(r,a,i), gx(r,a,i,t)*p(r,a,i,t)**(1+omegas(r,a))))
      $  (omegas(r,a) ne inf)
      ;

*  Output tax on commodity i produced by activity a -- (E_ps (1264), ~OUTPUTPRICES (1436))

ppeq(r,a,i,t)$(rs(r) and ts(t) and xFlag(r,a,i) and not ifSUB)..
   pp(r,a,i,t) =e= ((1 + prdtx(r,a,i,t))*p(r,a,i,t)) ;

*  N.B. We are not calculating pb in the GAMS version (E_pb (1286))

*  Aggregate commodities (E_pca (1327) -- no equivalent)

peq(r,a,i,t)$(rs(r) and ts(t) and xFlag(r,a,i))..
   0 =e= (x(r,a,i,t) - ax(r,a,i,t)*xs(r,i,t)
      *  (ps(r,i,t)/M_PP(r,a,i,t))**sigmas(r,i))$(sigmas(r,i) ne inf)
      +  (M_PP(r,a,i,t) - ps(r,i,t))$(sigmas(r,i) eq inf)
      ;

*  Price of domestic supply (E_qc (1333) -- no equivalent)

pseq(r,i,t)$(rs(r) and ts(t) and xsFlag(r,i))..
   0 =e= (xs(r,i,t) - sum(a$xFlag(r,a,i), x(r,a,i,t)))
      $  (sigmas(r,i) eq inf)
      +  (ps(r,i,t)**(1-sigmas(r,i))
      -     sum(a$xFlag(r,a,i), ax(r,a,i,t)*M_PP(r,a,i,t)**(1-sigmas(r,i))))
      $  (sigmas(r,i) ne inf)
      ;

*  -------------------------------------------------------------------------------------------------
*
*     Section 3 -- Income distribution
*
*  -------------------------------------------------------------------------------------------------

*  N.B. In the GEMPACK code, the tax revenue equations are listed towards the
*       end of the code, section 9.
*  Income distribution

ytaxeq(r,gy,t)$(rs(r) and ts(t))..

   ytax(r,gy,t) =e=

*  Tax revenues from production tax -- (E_del_taxrout (2582), TOUTRATIO (1446))
      + (sum((a,i)$xFlag(r,a,i), prdtx(r,a,i,t)*p(r,a,i,t)*x(r,a,i,t)))$sameas(gy,"pt")

*  Tax revenues from factor taxes -- (E_del_taxrfu (2589), TFURATIO (1408))
      +  (sum((fp,a)$xfFlag(r,fp,a), fcttx(r,fp,a,t)*pf(r,fp,a,t)*(xf(r,fp,a,t)/xScale(r,a))))
      $sameas(gy,"ft")

*  Revenue costs from factor subsidies -- N/A
      +  (sum((fp,a)$xfFlag(r,fp,a), fctts(r,fp,a,t)*pf(r,fp,a,t)*(xf(r,fp,a,t)/xScale(r,a))))
      $sameas(gy,"fs")

*  Indirect tax revenues from firm consumption -- (E_del_taxriu (2596), TIURATIO (1327))
      + (sum((i,a), dintx(r,i,a,t)*pd(r,i,t)*(xd(r,i,a,t)/xScale(r,a))
      +             mintx(r,i,a,t)*pmt(r,i,t)*(xm(r,i,a,t)/xScale(r,a))))
      $sameas(gy,"fc")

*  Indirect tax revenues from private consumption -- (E_del_taxrpc (2610), TPCRATIO (1133))
      +  (sum((i,h), dintx(r,i,h,t)*pd(r,i,t)*xd(r,i,h,t)
      +              mintx(r,i,h,t)*pmt(r,i,t)*xm(r,i,h,t)))
      $sameas(gy,"pc")

*  Indirect tax revenues from government consumption -- (E_del_taxrgc (2619), TGCRATIO (965))
      +  (sum((i,gov), dintx(r,i,gov,t)*pd(r,i,t)*xd(r,i,gov,t)
      +                mintx(r,i,gov,t)*pmt(r,i,t)*xm(r,i,gov,t)))
      $sameas(gy,"gc")

*  Indirect tax revenues from investment consumption -- (E_del_taxric (2628), TIURATIO (1327))
      +  (sum((i,inv), dintx(r,i,inv,t)*pd(r,i,t)*xd(r,i,inv,t)
      +                mintx(r,i,inv,t)*pmt(r,i,t)*xm(r,i,inv,t)))
      $sameas(gy,"ic")

*  Export tax revenues -- (E_del_taxrexp (2642), TEXPRATIO (1768))
      +  (sum((i,rp), (exptx(r,i,rp,t) + etax(r,i,t))*pe(r,i,rp,t)*xw(r,i,rp,t)))
      $sameas(gy,"et")

*  Import tax revenues -- (E_del_taxrimp (2649), TIMPRATIO (1842))
      +  (sum((i,rp), (imptx(rp,i,r,t) + mtax(r,i,t))*M_PMCIF(rp,i,r,t)*xw(rp,i,r,t)))
      $sameas(gy,"mt")

*  Direct tax revenues -- (E_del_taxrinc (2667), TINCRATIO (2141))
      +  (sum((fp,a)$xfFlag(r,fp,a), kappaf(r,fp,a,t)*pf(r,fp,a,t)*xf(r,fp,a,t)/xScale(r,a)))
      $sameas(gy,"dt")
      ;

*  Total tax revenue -- (E_del_ttaxr (2684), DTAXRATIO (2201))

ytaxToteq(r,t)$(rs(r) and ts(t))..
   ytaxTot(r,t) =e= sum(gy, ytax(r,gy,t)) ;

*  Total indirect tax revenues -- (E_del_indtaxr (2674), DINDTAXRATIO (2192))

ytaxIndeq(r,t)$(rs(r) and ts(t))..
   yTaxInd(r,t) =e= ytaxTot(r,t) - ytax(r,"dt",t) ;

*  Factor income, net of depreciation -- (E_fincome (1351), FACTORINCOME (2183))

factYeq(r,t)$(rs(r) and ts(t))..
   factY(r,t) =e= sum((fp,a)$xfFlag(r,fp,a), pf(r,fp,a,t)*xf(r,fp,a,t)/xScale(r,a))
               -     fdepr(r,t)*pi(r,t)*kstock(r,t) ;

*  Regional income -- (E_y (1370), REGIONALINCOME (2224))

regYeq(r,t)$(rs(r) and ts(t))..
   regY(r,t) =e= factY(r,t) + yTaxInd(r,t) ;

*  -------------------------------------------------------------------------------------------------
*
*     Section 4 -- Allocation of regional income across expenditure categories
*
*  -------------------------------------------------------------------------------------------------

*  N.B. This section needs to be reviewed


*  Total nominal saving -- (E_qsave (1394), SAVING (2270))

rsaveq(r,t)$(rs(r) and ts(t))..
   rsav(r,t) =e= betaS(r,t)*phi(r,t)*regY(r,t) ;


*  Total government consumption expenditure -- (E_yg (1399), GOVCONSEXP (2265))

ygeq(r,t)$(rs(r) and ts(t))..
   yg(r,t) =e= betaG(r,t)*phi(r,t)*regY(r,t) ;

*  Total private consumption expenditure -- (E_yp (1404), PRIVCONSEXP (2260))

yceq(r,t)$(rs(r) and ts(t))..
   yc(r,t) =e= betaP(r,t)*(phi(r,t)/phiP(r,t))*regY(r,t) ;

*  Elasticity of total expenditure wrt to utility -- UTILITELASTIC (2255)

phieq(r,t)$(rs(r) and ts(t))..
   phi(r,t)*(betaP(r,t)/phiP(r,t) + betaG(r,t) + betaS(r,t)) =e= 1 ;

*  Top level utility function -- (E_u (1496), UTILITY (2347))

ueq(r,t)$(rs(r) and ts(t))..
   log(u(r,t)) =e= log(au(r,t))
                +  betaP(r,t)*sum(h, log(uh(r,h,t)))
                +  betaG(r,t)*log(ug(r,t))
                +  betaS(r,t)*log(us(r,t)) ;

*  Utility from national savings consumption -- XXX (XXX)

useq(r,t)$(rs(r) and ts(t))..
   us(r,t) =e= aus(r,t)*(rsav(r,t)/psave(r,t))/pop(r,t) ;

*  -------------------------------------------------------------------------------------------------
*
*     Section 5 -- Domestic final demand
*
*  -------------------------------------------------------------------------------------------------

*  Factor needed for household consumption function

zconseq(r,i,h,t)$(rs(r) and ts(t) and xaFlag(r,i,h) ne 0)..
   zcons(r,i,h,t) =e= (alphaa(r,i,h,t)*bh(r,i,t)
                   * (pa(r,i,h,t)**bh(r,i,t))
                   * (uh(r,h,t)**(eh(r,i,t)*bh(r,i,t)))
                   * ((yc(r,t)/pop(r,t))**(-bh(r,i,t))))$(%utility% eq CDE)
                   + (alphaa(r,i,h,t))$(%utility% eq CD) ;

*  Budget shares

xcshreq(r,i,h,t)$(rs(r) and ts(t) and xaFlag(r,i,h) ne 0)..
   xcshr(r,i,h,t) =e= zcons(r,i,h,t)/sum(j$xaFlag(r,j,h), zcons(r,j,h,t)) ;

*  Household demands for goods and services -- (E_qpa (1570), PRIVDMNDS (1080))

xaceq(r,i,h,t)$(rs(r) and ts(t) and xaFlag(r,i,h) ne 0)..
  pa(r,i,h,t)*xa(r,i,h,t) =e= xcshr(r,i,h,t)*yc(r,t) ;

*  Elasticity of expenditure wrt utility from private consumption
*     -- (E_uepriv (1587), UTILELASPRIV (1026))

phiPeq(r,t)$(rs(r) and ts(t))..
   phiP(r,t) =e= sum((i,h), xcshr(r,i,h,t)*eh(r,i,t))$(%utility% eq CDE)
              +  sum((i,h), xcshr(r,i,h,t))$(%utility% eq CD)
              ;

*  Consumer expenditure deflator (approx.) -- (E_ppriv (1592), PHHLDINDEX (1005))

pconseq(r,t)$(rs(r) and ts(t))..
   pcons(r,t) =e= sum((i,h), xcshr(r,i,h,t)*pa(r,i,h,t)) ;

* Household utility (per capita) -- (E_up (1597), PRIVATEU (1010))

uheq(r,h,t)$(rs(r) and ts(t))..
   0 =e= (1 - sum(i$xaFlag(r,i,h), zcons(r,i,h,t)/bh(r,i,t)))$(%utility% eq CDE)
      +  (uh(r,h,t) - auh(r,h,t)*prod(i$xaFlag(r,i,h), xa(r,i,h,t)**alphaa(r,i,h,t)))
      $(%utility% eq CD) ;

*  Sourcing of commodities by households
*     (see below xdeq, xmeq, paeq -- one set of Armington equations)
*     Consolidates E_qpd, E_qpm and E_ppa

*  Decomposition of public demand

*  CES expenditure function -- (E_qga (1644), GOVDMNDS (913))

xageq(r,i,gov,t)$(rs(r) and ts(t) and xaFlag(r,i,gov) ne 0)..
   xa(r,i,gov,t) =e= alphaa(r,i,gov,t)*xg(r,t)*(pg(r,t)/pa(r,i,gov,t))**sigmag(r) ;

*  Government expenditure price deflator -- (E_pgov (1649), GPRICEINDEX (908))

pgeq(r,t)$(rs(r) and ts(t))..
   0 =e= (axg(r,t)*pg(r,t) - sum(gov,prod(i$xaFlag(r,i,gov),
                  (pa(r,i,gov,t)/alphaa(r,i,gov,t))**alphaa(r,i,gov,t))))$(sigmag(r) eq 1)
      +  ((axg(r,t)*pg(r,t))**(1-sigmag(r)) - sum(gov,sum(i$xaFlag(r,i,gov),
                     alphaa(r,i,gov,t)*pa(r,i,gov,t)**(1-sigmag(r)))))$(sigmag(r) ne 1)
      +  (pg(r,t)*xg(r,t) - (sum(gov, sum(i$xaFlag(r,i,gov), pa(r,i,gov,t)*xa(r,i,gov,t)))))$(0)
      ;

*  Nominal government expenditures

xgeq(r,t)$(rs(r) and ts(t))..
   pg(r,t)*xg(r,t) =e= yg(r,t) ;

*  Utility from government consumption -- (E_ug (1654), GOVU (918))

ugeq(r,t)$(rs(r) and ts(t))..
   ug(r,t) =e= aug(r,t)*xg(r,t)/pop(r,t) ;

*  Sourcing of commodities by government
*     (see below xdeq, xmeq, paeq -- one set of Armington equations)
*     Consolidates E_qgd, E_qgm and E_pga

*  Decomposition of investment demand
*  N.B. Uses CES expenditure function

*  CES expenditure function -- (E_qia (1692), INTDEMAND (1286))

xaieq(r,i,inv,t)$(rs(r) and ts(t) and xaFlag(r,i,inv) ne 0)..
   xa(r,i,inv,t)*lambdai(r,i,t) =e=
      alphaa(r,i,inv,t)*xi(r,t)*(lambdai(r,i,t)*pi(r,t)/pa(r,i,inv,t))**sigmai(r) ;

*  Investment expenditure price deflator -- (E_pinv (1697), ZEROPROFITS (1464))

pieq(r,t)$(rs(r) and ts(t))..
   0 =e= (axi(r,t)*pi(r,t) - sum(inv,prod(i$alphaa(r,i,inv,t),
                  (pa(r,i,inv,t)/(lambdai(r,i,t)*alphaa(r,i,inv,t)))**alphaa(r,i,inv,t))))
      $(sigmai(r) eq 1)
      +  ((axi(r,t)*pi(r,t))**(1-sigmai(r)) - sum(inv,sum(i$alphaa(r,i,inv,t),
                     alphaa(r,i,inv,t)*(pa(r,i,inv,t)/lambdai(r,i,t))**(1-sigmai(r)))))
      $(sigmai(r) ne 1)
      +  (pi(r,t)*xi(r,t) - (sum(inv, sum(i$xaFlag(r,i,inv), pa(r,i,inv,t)*xa(r,i,inv,t)))))
      $(0)
      ;

*  Sourcing of commodities by investment
*     (see below xdeq, xmeq, paeq -- one set of Armington equations)
*     Consolidates E_qid, E_qim and E_pia

*  Volume of investment expenditures

xieq(r,t)$(rs(r) and ts(t))..
   pi(r,t)*xi(r,t) =e= yi(r,t) ;

*  -------------------------------------------------------------------------------------------------
*
*     Section 6 -- Trade, goods market equilibrium and prices
*
*  -------------------------------------------------------------------------------------------------

*
*  Trade section
*
*  1. Armington decomposition across agents
*
*  Firm purchasers' prices

*  Domestic goods --
*           firms -- E_pfd (2141), DMNDDPRICE (1305)
*         private -- E_ppd (2168), PHHDPRICE  (1114)
*          public -- E_pgd (2183), GHHDPRICE  (935)
*      investment -- E_pid (2198), DMNDDPRICE (1305)
*              TT -- N/A

pdpeq(r,i,aa,t)$(rs(r) and ts(t) and alphad(r,i,aa,t) ne 0 and not ifSUB)..
   pdp(r,i,aa,t) =e= pd(r,i,t)*(1 + dintx(r,i,aa,t)) ;

*  Imported goods --
*           firms -- E_pfm (2150), DMNDIPRICES (1317)
*         private -- E_ppm (2173), PHHIPRICES  (1128)
*          public -- E_pgm (2188), GHHIPRICES  (940)
*      investment -- E_pim (2203), DMNDIPRICES (1317)
*              TT -- N/A

pmpeq(r,i,aa,t)$(rs(r) and ts(t) and alpham(r,i,aa,t) ne 0 and not ifSUB)..
   pmp(r,i,aa,t) =e= pmt(r,i,t)*(1 + mintx(r,i,aa,t)) ;

*  Armington price --
*            firms -- E_pfa (1137), ICOMPRICE (1340)
*          private -- E_ppa (1623), PCOMPRICE (1147)
*           public -- E_pga (1675), GCOMPRICE (950)
*       investment -- E_pia (1717), ICOMPRICE (1340)
*               TT -- N/A

paeq(r,i,aa,t)$(rs(r) and ts(t) and xaFlag(r,i,aa) ne 0)..
   pa(r,i,aa,t)**(1-sigmam(r,i,aa)) =e=
          alphad(r,i,aa,t)*M_PDP(r,i,aa,t)**(1-sigmam(r,i,aa))
      +   alpham(r,i,aa,t)*M_PMP(r,i,aa,t)**(1-sigmam(r,i,aa)) ;

*  Armington decomposition of purchases -- domestic
*            firms -- E_qfd (1120), INDDOM   (1350)
*          private -- E_qpd (1607), PHHLDDOM (1152)
*           public -- E_qgd (1665), GHHLDDOM (960)
*       investment -- E_qid (1702), INDDOM   (1350)
*               TT -- N/A

xdeq(r,i,aa,t)$(rs(r) and ts(t) and alphad(r,i,aa,t) ne 0)..
   xd(r,i,aa,t) =e= alphad(r,i,aa,t)*xa(r,i,aa,t)
                 *  (pa(r,i,aa,t)/M_PDP(r,i,aa,t))**sigmam(r,i,aa) ;

*  Armington decomposition of purchases -- imports
*            firms -- E_qfm (1125), INDIMP     (1345)
*          private -- E_qpm (1618), PHHLAGRIMP (1157)
*           public -- E_qgm (1670), GHHLAGRIMP (955)
*       investment -- E_qim (1707), INDIMP     (1345)
*               TT -- N/A

xmeq(r,i,aa,t)$(rs(r) and ts(t) and alpham(r,i,aa,t) ne 0)..
   xm(r,i,aa,t) =e= alpham(r,i,aa,t)*xa(r,i,aa,t)
                 *  (pa(r,i,aa,t)/M_PMP(r,i,aa,t))**sigmam(r,i,aa) ;

*  Allocation of aggregate import by region of origin --

*  Calculate total import demand -- (E_qms (1759), MKTCLIMP (2465))

xmteq(r,i,t)$(rs(r) and ts(t) and xmtFlag(r,i))..
   xmt(r,i,t) =e= sum(aa$alpham(r,i,aa,t), xm(r,i,aa,t)/xScale(r,aa)) ;

*  Calculate bilateral import demand -- (E_qxs (1777), IMPORTDEMAND (1835))

xweq(rp,i,r,t)$(rs(r) and ts(t) and xwFlag(rp,i,r))..
   xw(rp,i,r,t) =e= amw(rp,i,r,t)*xmt(r,i,t)*((pmt(r,i,t)/(M_PM(rp,i,r,t)))**sigmaw(r,i))
                 *  lambdam(rp,i,r,t)**(sigmaw(r,i) - 1) ;

*  Calculate aggregate import price -- (E_pms (1793), DPRICEIMP (1806))

pmteq(r,i,t)$(rs(r) and ts(t) and xmtFlag(r,i))..
   pmt(r,i,t)**(1-sigmaw(r,i)) =e=
      sum(rp$xwFlag(rp,i,r), amw(rp,i,r,t)*(M_PM(rp,i,r,t)/lambdam(rp,i,r,t))**(1-sigmaw(r,i))) ;

*  Allocation of domestic supply (no equivalent as there is no CET)

*  Top nest

xdseq(r,i,t)$(rs(r) and ts(t) and xdFlag(r,i))..
   0 =e= (xds(r,i,t) - gd(r,i,t)*xs(r,i,t)*(pd(r,i,t)/ps(r,i,t))**omegax(r,i))
      $(omegax(r,i) ne inf)
      +  (pd(r,i,t) - ps(r,i,t))
      $(omegax(r,i) eq inf) ;

xeteq(r,i,t)$(rs(r) and ts(t) and xetFlag(r,i))..
   0 =e= (xet(r,i,t) - ge(r,i,t)*xs(r,i,t)*(pet(r,i,t)/ps(r,i,t))**omegax(r,i))
      $(omegax(r,i) ne inf)
      +  (pet(r,i,t) - ps(r,i,t))
      $(omegax(r,i) eq inf) ;

*  Corresponds to equilibrium condition for domestic output --
*     MKTCLTRD_MARG (2429) and MKTCLTRD_MARG (2437)

xseq(r,i,t)$(rs(r) and ts(t) and xsFlag(r,i))..
   0 =e= (ps(r,i,t)**(1+omegax(r,i)) - (gd(r,i,t)*pd(r,i,t)**(1+omegax(r,i))
      +      ge(r,i,t)*pet(r,i,t)**(1+omegax(r,i))))$(omegax(r,i) ne inf)
      +  (xs(r,i,t) - (xds(r,i,t) + xet(r,i,t)))$(omegax(r,i) eq inf)
      ;

*  This function substitutes out the equilibrium condition for bilateral trade

peeq(r,i,rp,t)$(rs(r) and ts(t) and xwFlag(r,i,rp))..
   0 =e= (xw(r,i,rp,t) - gw(r,i,rp,t)*xet(r,i,t)*(pe(r,i,rp,t)/pet(r,i,t))**omegaw(r,i))
      $(omegaw(r,i) ne inf)
      +  (pe(r,i,rp,t) - pet(r,i,t))
      $(omegaw(r,i) eq inf) ;

peteq(r,i,t)$(rs(r) and ts(t) and xetFlag(r,i))..
   0 =e= (pet(r,i,t)**(1+omegaw(r,i))
      -     sum(rp$xwFlag(r,i,rp), gw(r,i,rp,t)*pe(r,i,rp,t)**(1+omegaw(r,i))))
      $(omegaw(r,i) ne inf)
      +  (xet(r,i,t) - sum(rp$xwFlag(r,i,rp), xw(r,i,rp,t)))$
      (omegaw(r,i) eq inf)
      ;

*  Trade margins

*  Total demand for TT services from r to rp for good i -- Additional

xwmgeq(r,i,rp,t)$(ts(t) and tmgFlag(r,i,rp) and not ifSUB)..
   xwmg(r,i,rp,t) =e= tmarg(r,i,rp,t)*xw(r,i,rp,t) ;

*  Demand for TT services using m from r to rp for good i -- (E_qtmfsd (1829), QTRANS_MFSD (1951))

xmgmeq(m,r,i,rp,t)$(rs(r) and ts(t) and amgm(m,r,i,rp) ne 0 and not ifSUB)..
   xmgm(m,r,i,rp,t) =e= amgm(m,r,i,rp)*M_XWMG(r,i,rp,t)/lambdamg(m,r,i,rp,t) ;

*  The aggregate price of transporting i between r and rp --
*        (E_ptrans (1883), TRANSCOSTINDEX (2029))
*  Note--the price per transport mode is uniform globally

pwmgeq(r,i,rp,t)$(ts(t) and tmgFlag(r,i,rp) and not ifSUB)..
   pwmg(r,i,rp,t) =e= sum(m, amgm(m,r,i,rp)*ptmg(m,t)/lambdamg(m,r,i,rp,t)) ;

*  Global demand for TT services of type m -- (E_qtm (1916), TRANSDEMAND (1978))

xtmgeq(m,t)$(ts(t) and mFlag(m))..
   xtmg(m,t) =e= sum((r,i,rp), M_XMGM(m,r,i,rp,t)) ;

*  Allocation across regions -- (E_qst (1930), TRANSVCES (2040))

xatmgeq(r,m,tmg,t)$(rs(r) and ts(t) and xaFlag(r,m,tmg) ne 0)..
   xa(r,m,tmg,t) =e= alphaa(r,m,tmg,t)*xtmg(m,t)*(ptmg(m,t)/pa(r,m,tmg,t))**sigmamg(m) ;

*  The average global price of mode m -- (E_pt (1947), PTRANSPORT (1998))

ptmgeq(m,t)$(ts(t) and mFlag(m))..
   0 =e= (ptmg(m,t)**(1-sigmamg(m))
      - sum(tmg, sum(r, alphaa(r,m,tmg,t)*pa(r,m,tmg,t)**(1-sigmamg(m)))))
      $(sigmamg(m) ne 1)
      +  (axmg(m,t)*ptmg(m,t)
      - sum(tmg, prod(r$alphaa(r,m,tmg,t), (pa(r,m,tmg,t)/alphaa(r,m,tmg,t))**(alphaa(r,m,tmg,t)))))
      $(sigmamg(m) eq 1)
      + (ptmg(m,t)*xtmg(m,t) - (sum((r,tmg), pa(r,m,tmg,t)*xa(r,m,tmg,t))))$(0)
      ;

*  Bilateral FOB export prices -- (E_pfob (2005), EXPRICES (1763))

pefobeq(r,i,rp,t)$(rs(r) and ts(t) and xwFlag(r,i,rp) and not ifSUB)..
   pefob(r,i,rp,t) =e= (1 + exptx(r,i,rp,t) + etax(r,i,t))*pe(r,i,rp,t) ;

*  Border price of bilateral imports -- (E_pcif (2027), FOBCIF (2066))

pmcifeq(r,i,rp,t)$(rs(r) and ts(t) and xwFlag(r,i,rp) and not ifSUB)..
   pmcif(r,i,rp,t) =e= pefob(r,i,rp,t) + pwmg(r,i,rp,t)*tmarg(r,i,rp,t) ;

*  Calculate bilateral import prices tariff inclusive -- (E_pmds (2050), MKTPRICES (1797))

pmeq(r,i,rp,t)$(rs(r) and ts(t) and xwFlag(r,i,rp) and not ifSUB)..
   pm(r,i,rp,t) =e= (1 + imptx(r,i,rp,t) + mtax(rp,i,t))*pmcif(r,i,rp,t)/chipm(r,i,rp) ;

*  Goods market equilibrium
*  The bilateral trade equilibrium is directly substituted out.
*  N.B. Without the CET, the two market equilibrium conditions collapse to the
*        GTAP market condition, i.e. XDS(r,i)+sum(rp, XWD(r,i,rp)) = XS(r,i)

*  Domestic goods market equilibrium --
*     (E_qds (2096) & E_pds (2124), MKTCLDOM (2398))

pdeq(r,i,t)$(rs(r) and ts(t) and xdFlag(r,i))..
   xds(r,i,t) =e= sum(aa$alphad(r,i,aa,t), xd(r,i,aa,t)/xScale(r,aa)) ;

$ontext

*  Bilateral trade equilibrium condition is substituted out

peeq(r,i,t)$(rs(r) and ts(t) and xwFlag(r,i,rp))..
   xwd(r,i,rp,t) =e= xws(r,i,rp,t) ;
$offtext

*  -------------------------------------------------------------------------------------------------
*
*     Section 7 -- Factor supply and market equilibrium
*
*  -------------------------------------------------------------------------------------------------

*  Aggregate supply of mobile factors -- N/A

xfteq(r,fm,t)$(rs(r) and ts(t) and xftFlag(r,fm))..
   xft(r,fm,t) =e= aft(r,fm,t)*(pft(r,fm,t)/pabs(r,t))**etaf(r,fm) ;

*  Sectoral supply of factors -- ENDW_SUPPLY (2166)  and MKTCLENDWS (2487)
*                                for sluggish commodities, eq. condition not explicit

pfeq(r,fp,a,t)$(rs(r) and ts(t) and xfFlag(r,fp,a))..

*  CET expression for partially mobile factors -- (E_qes2 (2283))
*  N.B. We substitute out the supply = demand equation for each activity,
*       wrt to GTAP this is represented by equation E_peb

   0 =e= (xf(r,fp,a,t) - (xscale(r,a)*gf(r,fp,a,t)*xft(r,fp,t)
      *  (M_PFY(r,fp,a,t)/pft(r,fp,t))**omegaf(r,fp)))
*     *  (pf(r,fp,a,t)/pft(r,fp,t))**omegaf(r,fp)))
      $(fm(fp) and omegaf(r,fp) ne inf)

*  Law of one price for perfectly mobile factors -- (E_qes1 (2266))
      +  (M_PFY(r,fp,a,t) - pft(r,fp,t))
*     +  (pf(r,fp,a,t) - pft(r,fp,t))
      $(fm(fp) and omegaf(r,fp) eq inf)

*  Supply function (and equilibrium condition) for sector-specific factors (E_qes3 (2307))
      +  (xf(r,fp,a,t) - xscale(r,a)*gf(r,fp,a,t)*(M_PFY(r,fp,a,t)/pabs(r,t))**etaff(r,fp,a))
*     +  (xf(r,fp,a,t) - xscale(r,a)*gf(r,fp,a,t)*(pf(r,fp,a,t)/pabs(r,t))**etaff(r,fp,a))
      $(fnm(fp)) ;

*  Aggregate factor price -- (E_pe2 (2294), ENDW_PRICE (2152)) for sluggish commodities
*                         -- (E_pe1 (2261), MKTCLENDWM (2478)) for perfectly mobile commodities

pfteq(r,fm,t)$(rs(r) and ts(t) and xftFlag(r,fm))..

*  Aggregate factor price for partially mobile CET
   0 =e= (pft(r,fm,t)**(1+omegaf(r,fm))
      -   sum(a, gf(r,fm,a,t)*M_PFY(r,fm,a,t)**(1+omegaf(r,fm))))$(omegaf(r,fm) ne inf)
*     -   sum(a, gf(r,fm,a,t)*pf(r,fm,a,t)**(1+omegaf(r,fm))))$(omegaf(r,fm) ne inf)

*  Aggregation condition for fully mobile CET --
*     (E_pe1 (2261), MKTCLENDWM (2478)) for perfectly mobile commodities
      +  (xft(r,fm,t) - sum(a, xf(r,fm,a,t)/xScale(r,a)))$(omegaf(r,fm) eq inf)
      ;

*  Agents' price of factors -- (E_pfe (2323), MPFACTPRICE (1364) and SPFACTPRICE (1369))

pfaeq(r,fp,a,t)$(rs(r) and ts(t) and xfFlag(r,fp,a) and not ifSUB)..
   pfa(r,fp,a,t) =e= pf(r,fp,a,t)*(1 + fctts(r,fp,a,t) + fcttx(r,fp,a,t)) ;

pfyeq(r,fp,a,t)$(rs(r) and ts(t) and xfFlag(r,fp,a) and not ifSUB)..
   pfy(r,fp,a,t) =e= pf(r,fp,a,t)*(1 - kappaf(r,fp,a,t)) ;

*  -------------------------------------------------------------------------------------------------
*
*     Section 8 -- Allocation of global saving
*
*  -------------------------------------------------------------------------------------------------

*  Aggregate capital stock -- (E_kb (2402), KAPSVCES (1521)) (To be verified)
*  Non-normalized level of the capital stock

kstockeq(r,t)$(rs(r) and ts(t))..
   krat(r,t)*kstock(r,t) =e= sum(cap, xft(r,cap,t)) ;

*  End of period stock of capital -- (E_ke (2367), KEND (1581))

kapEndeq(r,t)$(rs(r) and ts(t))..
   kapEnd(r,t) =e= (1 - depr(r,t))*kstock(r,t) + xi(r,t) ;

*  Aggregate capital rate of return after tax --
*     (E_rental (2392), KAPRENTAL (1533) (to be verified))

arenteq(r,t)$(rs(r) and ts(t))..
   arent(r,t) =e= sum((a,cap),
                     (1-kappaf(r,cap,a,t))*pf(r,cap,a,t)*xf(r,cap,a,t)/xScale(r,a))
               /  kstock(r,t) ;

*  Net rate of return to capital -- (E_rorc (2397), RORCURRENT (1601))

rorceq(r,t)$(rs(r) and ts(t))..
   rorc(r,t) =e= arent(r,t)/pi(r,t) - fdepr(r,t) ;

*  Expected rate of return -- (E_rore (2430), ROREXPECTED (1620))

roreeq(r,t)$(rs(r) and ts(t))..
   rore(r,t) =e= rorc(r,t)*(kstock(r,t)/kapEnd(r,t))**RoRFlex(r,t) ;

*  Determination of capital flow -- (E_qinv (2459), RORGLOBAL (1669)) (to be verified)
*  Determines savf for all regions in the case of capFlex, for r-1 regions in all other cases

savfeq(r,t)$(rs(r) and ts(t))..
   0 =e= (risk(r,t)*rore(r,t) - rorg(t))
      $(RoRFlag eq capFlex)
      +  (xi(r,t) - depr(r,t)*kstock(r,t) - chiInv(r,t)*xigbl(t))
      $(RoRFlag eq capShrFix and not rres(r))
      +  (savf(r,t) - piGbl(t)*savfBar(r,t))
      $(RoRFlag eq capFix and not rres(r))
      +  (savf(r,t) - chif(r,t)*regY(r,t))
      $(RoRFlag eq capSFix and not rres(r))
      ;

*  Determination of RoRg -- (E_globalcgds (2488), GLOBALINV (1682)) (to be verified)
*  Determines RoRg as an average for all cases save capFlex

rorgeq(t)$(ts(t) and RoRFlag ne capFlex)..
   0 =e= (rorg(t) - sum(r, rore(r,t)*pi(r,t)*(xi(r,t) - depr(r,t)*kstock(r,t)))
                   / sum(rp, pi(rp,t)*(xi(rp,t) - depr(rp,t)*kstock(rp,t)))) ;

*  Determines nominal foreign savings as a share of regional income for
*  all cases except capSFix when chif is exogenous

chifeq(r,t)$(ts(t) and RoRFlag ne capSFix)..
   savf(r,t) =e= chif(r,t)*regY(r,t) ;

*  Determines Sf residually for all cases except capFlex
*  Determines RoRg for capFlex

capAccteq(t)$(ts(t))..
   0 =e= (sum(r, savf(r,t))) ;

* Nominal gross investment

yieq(r,t)$(rs(r) and ts(t) and not rres(r))..
   yi(r,t) =e= pi(r,t)*depr(r,t)*kstock(r,t) + rsav(r,t) + savf(r,t) ;

*  Global net investment -- GLOBINV (1682) (to be verified)

xigbleq(t)$ts(t)..
   xigbl(t) =e= sum(r, xi(r,t) - depr(r,t)*kstock(r,t)) ;

*  Price of global investment -- PRICGDS (1701) (to be verified)

pigbleq(t)$ts(t)..
   pigbl(t)*xigbl(t) =e= sum(r, pi(r,t)*(xi(r,t) - depr(r,t)*kstock(r,t))) ;

*  Price of savings -- SAVEPRICE (1721) (to be verified)

*  Calculate adjustment factor to make global savings and investment price line up

chiSaveeq(t)$ts(t)..
   chiSave(t) =e= sum((rp,t0), invwgt(rp,t)*pi(rp,t)/pi(rp,t0))
                 /  sum((rp,t0), savwgt(rp,t)*psave(rp,t)/psave(rp,t0)) ;

*  Price of savings equals investment price multiplied by adjustment factor

psaveeq(r,t)$(rs(r) and ts(t))..
   psave(r,t) =e= sum(t0, psave(r,t0)*chiSave(t)*pi(r,t)/pi(r,t0)) ;

*  Model prices

$macro mqabs(r,tp,tq) (sum((i,fd), pa(r,i,fd,tp)*xa(r,i,fd,tq)))

pabseq(r,t)$(rs(r) and ts(t))..
$iftheni "%simType%" == "compStat"
*  Always use baseyear
   pabs(r,t) =e= sum(t0, pabs(r,t0)
              *   sqrt((mqabs(r,t,t0)/mqabs(r,t0,t0))
              *        (mqabs(r,t,t)/mqabs(r,t0,t)))) ;
$else
   pabs(r,t) =e= pabs(r,t-1)
              *   sqrt((mqabs(r,t,t-1)/mqabs(r,t-1,t-1))
              *        (mqabs(r,t,t)/mqabs(r,t-1,t))) ;
$endif

$macro mqmuv(tp,tq) (sum((s,j,d)$(rmuv(s) and imuv(j)), M_PEFOB(s,j,d,tp)*xw(s,j,d,tq)))

pmuveq(t)$ts(t)..
$iftheni "%simType%" == "compStat"
*  Always use baseyear
   pmuv(t) =e= sum(t0, pmuv(t0)
            *     sqrt((mqmuv(t,t0)/mqmuv(t0,t0))
            *          (mqmuv(t,t)/mqmuv(t0,t)))) ;
$else
   pmuv(t) =e= pmuv(t-1)
            *     sqrt((mqmuv(t,t-1)/mqmuv(t-1,t-1))
            *          (mqmuv(t,t)/mqmuv(t-1,t))) ;
$endif

*  Regional index of factor prices

$macro mqfactr(r,tp,tq) (sum((fp,a), pf(r,fp,a,tp)*xf(r,fp,a,tq)/xscale(r,a)))

pfacteq(r,t)$ts(t)..
$iftheni "%simType%" == "compStat"
   pfact(r,t) =e= sum(t0, pfact(r,t0)
               *   sqrt((mqfactr(r,t,t0)/mqfactr(r,t0,t0))
               *        (mqfactr(r,t,t)/mqfactr(r,t0,t)))) ;
$else
   pfact(r,t) =e= pfact(r,t-1)
               *   sqrt((mqfactr(r,t,t-1)/mqfactr(r,t-1,t-1))
               *        (mqfactr(r,t,t)/mqfactr(r,t-1,t))) ;
$endif

$macro mqfactw(tp,tq) (sum((r,fp,a), \
    pf(r,fp,a,tp)*xf(r,fp,a,tq)/xscale(r,a)))

pwfacteq(t)$ts(t)..
$iftheni "%simType%" == "compStat"
   pwfact(t) =e= sum(t0, pwfact(t0)
              *   sqrt((mqfactw(t,t0)/mqfactw(t0,t0))
              *        (mqfactw(t,t)/mqfactw(t0,t)))) ;
$else
   pwfact(t) =e= pwfact(t-1)
              *   sqrt((mqfactw(t,t-1)/mqfactw(t-1,t-1))
              *        (mqfactw(t,t)/mqfactw(t-1,t))) ;
$endif

pnumeq(t)$ts(t)..
   pnum(t) =e= pwfact(t) ;

walraseq..
   walras =e= sum((r,t)$(rres(r) and ts(t)),
      yi(r,t) - (pi(r,t)*depr(r,t)*kstock(r,t) + rsav(r,t) + savf(r,t))) ;

* ------------------------------------------------------------------------------
*
*     Closure equations
*
* ------------------------------------------------------------------------------

dintxeq(r,i,aa,t)$(rs(r) and ts(t) and intxFlag(r,i,aa))..
   dintx(r,i,aa,t) =e= dintx0(r,i,aa) + dtxshft(r,i,aa,t) + rtxshft(r,aa,t) ;

mintxeq(r,i,aa,t)$(rs(r) and ts(t) and intxFlag(r,i,aa))..
   mintx(r,i,aa,t) =e= mintx0(r,i,aa) + mtxshft(r,i,aa,t) + rtxshft(r,aa,t) ;

ytaxshreq(r,gy,t)$(rs(r) and ts(t))..
   ytaxshr(r,gy,t) =e= ytax(r,gy,t)/regY(r,t) ;

gdpmpeq(r,t)$(rs(r) and ts(t))..
   gdpmp(r,t) =e= (sum((i,fd), pa(r,i,fd,t)*xa(r,i,fd,t))
               +   sum((i,rp), M_PEFOB(r,i,rp,t)*xw(r,i,rp,t) - M_PMCIF(rp,i,r,t)*xw(rp,i,r,t))) ;

*  Real GDP at market price -- using Fisher indexing

$macro mqgdp(tp,tq)  (sum((i,fd), pa(r,i,fd,tp)*xa(r,i,fd,tq)) + sum((i,rp), M_PEFOB(r,i,rp,tp)*xw(r,i,rp,tq) - M_PMCIF(rp,i,r,tp)*xw(rp,i,r,tq)))

rgdpmpeq(r,t)$(rs(r) and ts(t))..
$iftheni "%simType%" == "compStat"
   rgdpmp(r,t) =e= sum(t0, rgdpmp(r,t0)
                *    sqrt((gdpmp(r,t)/gdpmp(r,t0))*(mqgdp(t0,t)/mqgdp(t,t0)))) ;
$else
   rgdpmp(r,t) =e= rgdpmp(r,t-1)
                *    sqrt((gdpmp(r,t)/gdpmp(r,t-1))*(mqgdp(t-1,t)/mqgdp(t,t-1))) ;
$endif

pgdpmpeq(r,t)$(rs(r) and ts(t))..
   pgdpmp(r,t) =e= gdpmp(r,t)/rgdpmp(r,t) ;

*  Equivalent variation (E(p0,u))

eveq(r,h,t)$(rs(r) and ts(t) and (%utility% eq CDE))..
   sum(i$xaFlag(r,i,h), alphaa(r,i,h,t)*(uh(r,h,t)**(bh(r,i,t)*eh(r,i,t)))
          * (sum(t0, pa(r,i,h,t0))*Pop(r,t)/ev(r,h,t))**bh(r,i,t)) =e= 1 ;

*  Compensating variation (E(p,u0))

cveq(r,h,t)$(rs(r) and ts(t) and (%utility% eq CDE))..
   sum(i$xaFlag(r,i,h), alphaa(r,i,h,t)*(sum(t0, uh(r,h,t0))**(bh(r,i,t)*eh(r,i,t)))
          * (pa(r,i,h,t)*Pop(r,t)/cv(r,h,t))**bh(r,i,t)) =e= 1 ;

* ------------------------------------------------------------------------------
*
*     Dynamic equations
*
* ------------------------------------------------------------------------------

*  Top level uniform productivity shifter -- AOWORLD (1240)

axpeq(r,a,t)$(rs(r) and ts(t) and xpFlag(r,a))..
   axp(r,a,t) =e= axp(r,a,t-1)*(1 + axpsec(a,t) + axpreg(r,t) + axpall(r,a,t)) ;

*  ND bundle shifter -- no equivalent (TBV)

lambdandeq(r,a,t)$(rs(r) and ts(t) and ndFlag(r,a))..
   lambdand(r,a,t) =e= lambdand(r,a,t-1)*(1 + andsec(a,t) + andreg(r,t) + andall(r,a,t)) ;

*  VA bundle shifter -- AVAWORLD (1251)

lambdavaeq(r,a,t)$(rs(r) and ts(t) and vaFlag(r,a))..
   lambdava(r,a,t) =e= lambdava(r,a,t-1)*(1 + avasec(a,t) + avareg(r,t) + avaall(r,a,t)) ;

*  Factor demand technical change -- AFEWORLD (1382)

lambdafeq(r,fp,a,t)$(rs(r) and ts(t) and xfFlag(r,fp,a))..
   lambdaf(r,fp,a,t) =e= lambdaf(r,fp,a,t-1)
      * (1 + afecom(fp,t) + afesec(a,t) + afefac(r,fp,t)
      + afereg(r,t)$afeFlag(r,fp) + afeall(r,fp,a,t)) ;

*  Intermediate augmenting technical change -- AFWORLD (1281) with adjustments

lambdaioeq(r,i,a,t)$(rs(r) and ts(t) and xaFlag(r,i,a))..
   lambdaio(r,i,a,t) =e= lambdaio(r,i,a,t-1)
      *(1 + aiocom(i,t) + aiosec(a,t) + aioreg(r,t) + aioall(r,i,a,t)) ;

*  Real GDP growth

glcaleq(r,t)$(rs(r) and ts(t) and years(t) gt FirstYear)..
   rgdpmp(r,t) =e= (1 + ggdppc(r,t))*rgdpmp(r,t-1)*(pop(r,t)/pop(r,t-1)) ;

*  Labor productivity

afealleq(r,l,a,t)$(ifCal eq 1 and rs(r) and ts(t) and years(t) gt FirstYear and xfFlag(r,l,a))..
   afeall(r,l,a,t) =e= piadd(r,l,a,t) + pimlt(r,l,a,t)*gl(r,t) ;

*  Other equations

uedeq(r,i,j,h,t)$(rs(r) and ts(t) and xaFlag(r,i,h) ne 0 and xaFlag(r,j,h) and (%utility% eq CDE))..
   ued(r,i,j,t) =e= xcshr(r,i,h,t)*(-bh(r,i,t)
                 - (eh(r,i,t)*bh(r,i,t) - sum(jp, xcshr(r,jp,h,t)*eh(r,jp,t)*bh(r,jp,t)))
                 /  sum(jp, xcshr(r,jp,h,t)*eh(r,jp,t))) + kron(i,j)*(bh(r,i,t) - 1) ;

incelaseq(r,i,h,t)$(rs(r) and ts(t) and xaFlag(r,i,h) and (%utility% eq CDE))..
   incelas(r,i,t) =e= (eh(r,i,t)*bh(r,i,t) - sum(jp, xcshr(r,jp,h,t)*eh(r,jp,t)*bh(r,jp,t)))
                   /   sum(jp, xcshr(r,jp,h,t)*eh(r,jp,t))
                   -  (bh(r,i,t) - 1) + sum(jp, xcshr(r,jp,h,t)*bh(r,jp,t)) ;

cedeq(r,i,j,h,t)$(rs(r) and ts(t) and xaFlag(r,i,h) ne 0 and xaFlag(r,j,h) and (%utility% eq CDE))..
   ced(r,i,j,t) =e= ued(r,i,j,t) + xcshr(r,j,h,t) * incelas(r,i,t) ;

apeeq(r,i,j,h,t)$(rs(r) and ts(t) and xaFlag(r,i,h) ne 0 and xaFlag(r,j,h) and (%utility% eq CDE))..
   ape(r,i,j,t) =e= 1 - bh(r,j,t) - bh(r,i,t) + sum(jp, xcshr(r,jp,h,t)*bh(r,jp,t))
                 -  kron(i,j)*(1-bh(r,i,t))/xcshr(r,j,h,t) ;

model gtap /
   axpeq.axp, lambdandeq.lambdand, lambdavaeq.lambdava,
   ndeq.nd, vaeq.va, pxeq.px,
   lambdaioeq.lambdaio, pndeq.pnd, xapeq.xa,
   xeq.x, xpeq.xp, ppeq.pp, peq.p, pseq.ps,
   lambdafeq.lambdaf, xfeq.xf, pvaeq.pva,
   yTaxeq.ytax, yTaxToteq.ytaxtot, yTaxIndeq.ytaxind,
   factYeq.factY, regYeq.regY,
   phiPeq.phiP, phieq.phi, yceq.yc, ygeq.yg, rsaveq.rsav, uheq.uh, ugeq.ug, useq.us, ueq.u,
   zconseq.zcons, xcshreq.xcshr, xaceq.xa, pconseq.pcons,
   xageq.xa, pgeq.pg, xgeq.xg, xaieq.xa, pieq.pi, yieq.yi,
   pdpeq.pdp, pmpeq.pmp,  paeq.pa, xdeq.xd, xmeq.xm,
   xmteq.xmt, xweq.xw, pmteq.pmt, pmeq.pm,
*  xdseq.xds, xeteq.xet, xseq.xs, peeq.pe, peteq.pet, pefobeq.pefob,
   xdseq.xds, xeteq.xet, xseq, peeq.pe, peteq.pet, pefobeq.pefob,
   xwmgeq.xwmg, xmgmeq.xmgm, pwmgeq.pwmg, xtmgeq.xtmg, ptmgeq.ptmg, xatmgeq.xa, pmcifeq.pmcif,
   pdeq.pd,
*  xfteq.xft, pfeq.pf, pfteq.pft, pfaeq.pfa, pfyeq.pfy, kstockeq.kstock,
   xfteq.xft, pfeq.pf, pfteq, pfaeq.pfa, pfyeq.pfy, kstockeq.kstock,
   arenteq.arent, kapEndeq.kapEnd, rorceq.rorc, roreeq.rore, xieq.xi,
   savfeq, rorgeq.rorg, chifeq.chif, capAccteq,
   chiSaveeq.chiSave, psaveeq.psave, xigbleq.xigbl, pigbleq.pigbl,
   dintxeq.dintx, mintxeq.mintx, ytaxshreq.ytaxshr,
   gdpmpeq.gdpmp, rgdpmpeq.rgdpmp, pgdpmpeq.pgdpmp,
   pabseq.pabs, pmuveq.pmuv, pfacteq.pfact, pwfacteq.pwfact, pnumeq,
   eveq, cveq,
   walraseq
/ ;

gtap.holdfixed = 1 ;
gtap.scaleopt      = 1 ;
gtap.tolinfrep     = 1e-5 ;

*  Define dynamic models

model dynCal /
   gtap + glcaleq + afealleq.afeall
/ ;
dyncal.holdfixed = 1 ;
dyncal.scaleopt    = 1 ;
dyncal.tolinfrep   = 1e-5 ;

model dynGTAP /
   gtap + glcaleq.ggdppc
/ ;
dynGTAP.holdfixed = 1 ;
dynGTAP.scaleopt   = 1 ;
dynGTAP.tolinfrep  = 1e-5 ;

*  Define sub model used to calibrate the top level utility function

model betaCal / phieq.phi, yceq.betap, ygeq.betag, rsaveq.betas / ;
betaCal.holdfixed = 1 ;

* ------------------------------------------------------------------------------
*
*  Declare post-simulation parameters
*
* ------------------------------------------------------------------------------

parameters
   sam(r,is,js,t)
;

*  Initialize the model
* --------------------------------------------------------------------------------------------------
*
*  Initialize model variables
*
* --------------------------------------------------------------------------------------------------

* --------------------------------------------------------------------------------------------------
*
*  Initialize prices
*
* --------------------------------------------------------------------------------------------------

loop(t$t0(t),
   px.l(r,a,t)      = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   ps.l(r,i,t)      = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   pft.l(r,fm,t)    = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   if(1,
      pfy.l(r,fnm,a,t) = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   else
      pf.l(r,fnm,a,t)  = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   ) ;
   pa.l(r,i,aa,t)   = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   pg.l(r,t)        = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   pi.l(r,t)        = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   pmt.l(r,i,t)     = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   pnd.l(r,a,t)     = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   pva.l(r,a,t)     = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
   ptmg.l(m,t)      = ifdebug*uniform(0.5,1.5) + (1 - ifDebug)*1 ;
) ;

loop(t$(not t0(t)),
   loop(t0,
      px.l(r,a,t)      = px.l(r,a,t0) ;
      ps.l(r,i,t)      = ps.l(r,i,t0) ;
      pft.l(r,fm,t)    = pft.l(r,fm,t0) ;
      if(1,
         pfy.l(r,fnm,a,t) = pfy.l(r,fnm,a,t0) ;
      else
         pf.l(r,fnm,a,t)  = pf.l(r,fnm,a,t0) ;
      ) ;
      pa.l(r,i,aa,t)   = pa.l(r,i,aa,t0) ;
      pg.l(r,t)        = pg.l(r,t0) ;
      pi.l(r,t)        = pi.l(r,t0) ;
      pmt.l(r,i,t)     = pmt.l(r,i,t0) ;
      pnd.l(r,a,t)     = pnd.l(r,a,t0) ;
      pva.l(r,a,t)     = pva.l(r,a,t0) ;
      ptmg.l(m,t)      = ptmg.l(m,t0) ;
   ) ;
) ;

pd.l(r,i,t)     = ps.l(r,i,t) ;
pabs.l(r,t)     = 1 ;
pmuv.l(t)       = 1 ;
pnum.l(t)       = 1 ;
pfact.l(r,t)    = 1 ;
pwfact.l(t)     = 1 ;

* --------------------------------------------------------------------------------------------------
*
*     Initialize Armington matrices
*
* --------------------------------------------------------------------------------------------------

*  Firm demand

xd.l(r,i,a,t) = sum((i0,a0)$(mapi0(i0,i) and mapa0(a0,a)), inScale*vdfb(i0,a0,r))/pd.l(r,i,t) ;
xm.l(r,i,a,t) = sum((i0,a0)$(mapi0(i0,i) and mapa0(a0,a)), inScale*vmfb(i0,a0,r))/pmt.l(r,i,t) ;

dintx.fx(r,i,a,t)$xd.l(r,i,a,t)
   = sum((i0,a0)$(mapi0(i0,i) and mapa0(a0,a)), inScale*(vdfp(i0,a0,r) - vdfb(i0,a0,r)))
   / (pd.l(r,i,t)*xd.l(r,i,a,t)) ;
mintx.fx(r,i,a,t)$xm.l(r,i,a,t)
   = sum((i0,a0)$(mapi0(i0,i) and mapa0(a0,a)), inScale*(vmfp(i0,a0,r) - vmfb(i0,a0,r)))
   / (pmt.l(r,i,t)*xm.l(r,i,a,t)) ;

*  Private demand

xd.l(r,i,h,t) = sum(i0$mapi0(i0,i), inScale*vdpb(i0,r))/pd.l(r,i,t) ;
xm.l(r,i,h,t) = sum(i0$mapi0(i0,i), inScale*vmpb(i0,r))/pmt.l(r,i,t) ;

dintx.fx(r,i,h,t)$xd.l(r,i,h,t)
   = sum(i0$mapi0(i0,i), inScale*(vdpp(i0,r) - vdpb(i0,r)))
   / (pd.l(r,i,t)*xd.l(r,i,h,t)) ;
mintx.fx(r,i,h,t)$xm.l(r,i,h,t)
   = sum(i0$mapi0(i0,i), inScale*(vmpp(i0,r) - vmpb(i0,r)))
   / (pmt.l(r,i,t)*xm.l(r,i,h,t)) ;

*  Government demand

xd.l(r,i,gov,t) = sum(i0$mapi0(i0,i), inScale*vdgb(i0,r))/pd.l(r,i,t) ;
xm.l(r,i,gov,t) = sum(i0$mapi0(i0,i), inScale*vmgb(i0,r))/pmt.l(r,i,t) ;

dintx.fx(r,i,gov,t)$xd.l(r,i,gov,t)
   = sum(i0$mapi0(i0,i), inScale*(vdgp(i0,r) - vdgb(i0,r)))
   / (pd.l(r,i,t)*xd.l(r,i,gov,t)) ;
mintx.fx(r,i,gov,t)$xm.l(r,i,gov,t)
   = sum(i0$mapi0(i0,i), inScale*(vmgp(i0,r) - vmgb(i0,r)))
   / (pmt.l(r,i,t)*xm.l(r,i,gov,t)) ;

*  Investment demand

xd.l(r,i,inv,t) = sum(i0$mapi0(i0,i), inScale*vdib(i0,r))/pd.l(r,i,t) ;
xm.l(r,i,inv,t) = sum(i0$mapi0(i0,i), inScale*vmib(i0,r))/pmt.l(r,i,t) ;

dintx.fx(r,i,inv,t)$xd.l(r,i,inv,t)
   = sum(i0$mapi0(i0,i), inScale*(vdip(i0,r) - vdib(i0,r)))
   / (pd.l(r,i,t)*xd.l(r,i,inv,t)) ;
mintx.fx(r,i,inv,t)$xm.l(r,i,inv,t)
   = sum(i0$mapi0(i0,i), inScale*(vmip(i0,r) - vmib(i0,r)))
   / (pmt.l(r,i,t)*xm.l(r,i,inv,t)) ;

*  Domestic supply of margin services

xd.l(r,i,tmg,t) = sum(i0$mapi0(i0,i), inScale*vst(i0,r))/pd.l(r,i,t) ;
xm.l(r,i,tmg,t) = 0 ;

dintx.fx(r,i,tmg,t)$xd.l(r,i,tmg,t) = 0 ;
mintx.fx(r,i,tmg,t)$xm.l(r,i,tmg,t) = 0 ;

*  End user prices of goods

pdp.l(r,i,aa,t) = pd.l(r,i,t)*(1 + dintx.l(r,i,aa,t)) ;
pmp.l(r,i,aa,t) = pmt.l(r,i,t)*(1 + mintx.l(r,i,aa,t)) ;

*  Armington demand

xa.l(r,i,aa,t)  = (pdp.l(r,i,aa,t)*xd.l(r,i,aa,t)
                +  pmp.l(r,i,aa,t)*xm.l(r,i,aa,t))/pa.l(r,i,aa,t) ;

* --------------------------------------------------------------------------------------------------
*
* Production module initialization
*
* --------------------------------------------------------------------------------------------------

*  Initialize factor prices and volumes

*  CET decision is based on after tax remuneration --> set pfy = pft ;
*  PFY for sector-specific factors initialized above

if(1,
   pfy.l(r,fm,a,t)    = pft.l(r,fm,t) ;
   kappaf.l(r,fp,a,t) = inscale*(sum(a0$mapa0(a0,a), EVFB(fp,a0,r))) ;
   kappaf.fx(r,fp,a,t)$kappaf.l(r,fp,a,t)
                      = inscale*(sum(a0$mapa0(a0,a), EVFB(fp,a0,r) - EVOS(fp,a0,r)))
                      /    kappaf.l(r,fp,a,t) ;
   pf.l(r,fp,a,t)     = pfy.l(r,fp,a,t)/(1 - kappaf.l(r,fp,a,t)) ;
   xf.l(r,fp,a,t)     = sum(a0$mapa0(a0,a), inScale*evfb(fp,a0,r)) / pf.l(r,fp,a,t) ;
   xft.l(r,fm,t)      = sum(a, pfy.l(r,fm,a,t)*xf.l(r,fm,a,t)) / pft.l(r,fm,t) ;
else
   pf.l(r,fm,a,t)     = pft.l(r,fm,t) ;
   kappaf.l(r,fp,a,t) = inscale*(sum(a0$mapa0(a0,a), EVFB(fp,a0,r))) ;
   kappaf.fx(r,fp,a,t)$kappaf.l(r,fp,a,t)
                      = inscale*(sum(a0$mapa0(a0,a), EVFB(fp,a0,r) - EVOS(fp,a0,r)))
                      /    kappaf.l(r,fp,a,t) ;
   pfy.l(r,fp,a,t)    = pf.l(r,fp,a,t)*(1 - kappaf.l(r,fp,a,t)) ;
   xf.l(r,fp,a,t)     = sum(a0$mapa0(a0,a), inScale*evfb(fp,a0,r)) / pf.l(r,fp,a,t) ;
   xft.l(r,fm,t)      = sum(a, pf.l(r,fm,a,t)*xf.l(r,fm,a,t)) / pft.l(r,fm,t) ;
) ;

fctts.fx(r,fp,a,t)$xf.l(r,fp,a,t)
   = -sum(a0$mapa0(a0,a), inScale*fbep(fp,a0,r))/(pf.l(r,fp,a,t)*xf.l(r,fp,a,t)) ;
fcttx.fx(r,fp,a,t)$xf.l(r,fp,a,t)
   =  sum(a0$mapa0(a0,a), inScale*ftrv(fp,a0,r))/(pf.l(r,fp,a,t)*xf.l(r,fp,a,t)) ;
pfa.l(r,fp,a,t) = pf.l(r,fp,a,t)*(1 + fctts.l(r,fp,a,t) + fcttx.l(r,fp,a,t)) ;

xp.l(r,a,t) = (sum(i, pdp.l(r,i,a,t)*xd.l(r,i,a,t) + pmp.l(r,i,a,t)*xm.l(r,i,a,t))
            +  sum(fp, pfa.l(r,fp,a,t)*xf.l(r,fp,a,t)))/px.l(r,a,t) ;

xScale(r,aa) = 1 ;

loop(t0,
   xpFlag(r,a)$(xp.l(r,a,t0) ne 0) = 1 ;
   xpFlag(r,a)$xpFlag(r,a) = xpScale*10**(-round(log10(xp.l(r,a,t0)))) ;
   xScale(r,a)$xpFlag(r,a) = xpFlag(r,a) ;
) ;

if(0, xScale(r,aa) = 1 ; ) ;

loop(t0,
   xaFlag(r,i,aa)$xa.l(r,i,aa,t0) = 1 ;
) ;

nd.l(r,a,t) = sum(i, pa.l(r,i,a,t)*xa.l(r,i,a,t))/pnd.l(r,a,t) ;
loop(t0,
   ndFlag(r,a)$nd.l(r,a,t0) = 1 ;
) ;

va.l(r,a,t)    = sum(fp, pfa.l(r,fp,a,t)*xf.l(r,fp,a,t))/pva.l(r,a,t) ;
loop(t0,
   vaFlag(r,a)$va.l(r,a,t0) = 1 ;
   xfFlag(r,fp,a)$xf.l(r,fp,a,t0) = 1 ;
) ;

*  Tech parameters

axp.l(r,a,t)        = 1 ;
axpsec.fx(a,t)      = 0 ;
axpreg.fx(r,t)      = 0 ;
axpall.fx(r,a,t)    = 0 ;

lambdand.l(r,a,t)   = 1 ;
andsec.fx(a,t)      = 0 ;
andreg.fx(r,t)      = 0 ;
andall.fx(r,a,t)    = 0 ;

lambdava.l(r,a,t)   = 1 ;
avasec.fx(a,t)      = 0 ;
avareg.fx(r,t)      = 0 ;
avaall.fx(r,a,t)    = 0 ;

lambdaio.l(r,i,a,t) = 1 ;
aiocom.fx(i,t)      = 0 ;
aiosec.fx(a,t)      = 0 ;
aioreg.fx(r,t)      = 0 ;
aioall.fx(r,i,a,t)  = 0 ;

lambdaf.l(r,fp,a,t) = 1 ;
afecom.fx(fp,t)     = 0 ;
afesec.fx(a,t)      = 0 ;
afereg.fx(r,t)      = 0 ;
afefac.fx(r,fp,t)   = 0 ;
afeall.fx(r,fp,a,t) = 0 ;
*  Make labor augmenting by default ;
afeFlag(r,l)        = yes ;

* --------------------------------------------------------------------------------------------------
*
* Private demand module initialization
*
* --------------------------------------------------------------------------------------------------

pop.fx(r,t) = pop0(r) ;

loop(h,

   yc.l(r,t) = sum(i, pa.l(r,i,h,t)*xa.l(r,i,h,t)) ;

   xcshr.l(r,i,h,t) = pa.l(r,i,h,t)*xa.l(r,i,h,t)/yc.l(r,t) ;

   pcons.l(r,t) = sum(i, xcshr.l(r,i,h,t)*pa.l(r,i,h,t)) ;

   uh.l(r,h,t) = 1 ;

   ev.l(r,h,t) = yc.l(r,t) ;
   cv.l(r,h,t) = yc.l(r,t) ;

) ;

* --------------------------------------------------------------------------------------------------
*
* Public demand module initialization
*
* --------------------------------------------------------------------------------------------------

loop(gov,
   yg.l(r,t) = sum(i, pa.l(r,i,gov,t)*xa.l(r,i,gov,t)) ;
   xg.l(r,t) = yg.l(r,t)/pg.l(r,t) ;
   ug.l(r,t) = 1 ;
) ;

* --------------------------------------------------------------------------------------------------
*
* Investment demand module initialization
*
* --------------------------------------------------------------------------------------------------

loop(inv,
   yi.l(r,t)    = sum(i, pa.l(r,i,inv,t)*xa.l(r,i,inv,t)) ;
   xi.l(r,t)    = yi.l(r,t)/pi.l(r,t) ;
   us.l(r,t)    = 1 ;
) ;

* --------------------------------------------------------------------------------------------------
*
*  Make module
*
* --------------------------------------------------------------------------------------------------

*  Set 'pp' to ps as it is more likely that demand has perfect substitutes than
*  supply

*  Calculate x tax-inclusive

x.l(r,a,i,t) = inScale*sum((a0,i0)$(mapa0(a0,a) and mapi0(i0,i)), makb(i0,a0,r)) ;
loop(t0,
   xFlag(r,a,i)$x.l(r,a,i,t0) = 1 ;
) ;

prdtx.l(r,a,i,t) = inScale*sum((a0,i0)$(mapa0(a0,a) and mapi0(i0,i)), maks(i0,a0,r)) ;
prdtx.l(r,a,i,t)$prdtx.l(r,a,i,t) = x.l(r,a,i,t)/prdtx.l(r,a,i,t) - 1 ;
x.l(r,a,i,t)$xFlag(r,a,i) = x.l(r,a,i,t) / ps.l(r,i,t) ;
p.l(r,a,i,t)  = (ps.l(r,i,t)/(1+prdtx.l(r,a,i,t)))$xFlag(r,a,i)
              + 1$(not xFlag(r,a,i)) ;
pp.l(r,a,i,t) = (1+prdtx.l(r,a,i,t))*p.l(r,a,i,t) ;

xs.l(r,i,t)  = sum(a, pp.l(r,a,i,t)*x.l(r,a,i,t))/ps.l(r,i,t) ;

* --------------------------------------------------------------------------------------------------
*
*  Trade module
*
* --------------------------------------------------------------------------------------------------

*  Allow for the possibility of perfect transformation

pet.l(r,i,t)   = ps.l(r,i,t) ;
pd.l(r,i,t)    = ps.l(r,i,t) ;
pe.l(r,i,rp,t) = pet.l(r,i,t) ;

xw.l(r,i,rp,t) = sum(i0$mapi0(i0,i), inScale*VXSB(i0, r, rp))/pe.l(r,i,rp,t) ;
loop(t0,
   xwFlag(r,i,rp)$xw.l(r,i,rp,t0) = 1 ;
) ;
etax.fx(r,i,t) = 0 ;
exptx.fx(r,i,rp,t)$xwFlag(r,i,rp)
   = sum(i0$mapi0(i0,i), inScale*(VFOB(i0, r, rp)-VXSB(i0, r, rp)))
   / (pe.l(r,i,rp,t)*xw.l(r,i,rp,t)) ;
pefob.l(r,i,rp,t)$xwFlag(r,i,rp)  = (1 + exptx.l(r,i,rp,t))*pe.l(r,i,rp,t) ;

pwmg.l(r,i,rp,t) = 1 ;
tmarg.fx(r,i,rp,t)$xwFlag(r,i,rp)
   = sum(i0$mapi0(i0,i), inScale*(VCIF(i0, r, rp)-VFOB(i0, r, rp)))
   / (xw.l(r,i,rp,t)*pwmg.l(r,i,rp,t)) ;

loop(t0,
   tmgFlag(r,i,rp)$tmarg.l(r,i,rp,t0) = 1 ;
) ;

pmCIF.l(r,i,rp,t)$xwFlag(r,i,rp)
   = (pefob.l(r,i,rp,t) + pwmg.l(r,i,rp,t)*tmarg.l(r,i,rp,t)) ;

imptx.fx(rp,i,r,t)$xwFlag(rp,i,r)
   = sum(i0$mapi0(i0,i), inScale*(VMSB(i0, rp, r)-VCIF(i0, rp, r)))
   / (pmCIF.l(rp,i,r,t)*xw.l(rp,i,r,t)) ;
loop(t0,
   chipm(rp,i,r)$xwFlag(rp,i,r)    = (1 + imptx.l(rp,i,r,t0))*pmCIF.l(rp,i,r,t0) ;
) ;
if(1,
   pm.l(r,i,rp,t) = chipm(r,i,rp) ;
   chipm(r,i,rp)   = 1 ;
else
   pm.l(r,i,rp,t)  = 1 ;
) ;

mtax.fx(rp,i,t) = 0 ;
lambdam.fx(rp,i,r,t) = 1 ;

xmt.l(r,i,t) = sum(aa, xm.l(r,i,aa,t)) ;
xds.l(r,i,t) = sum(aa, xd.l(r,i,aa,t)) ;

xet.l(r,i,t) = sum(rp, pe.l(r,i,rp,t)*xw.l(r,i,rp,t)) / pet.l(r,i,t) ;
xet.l(r,i,t) = (ps.l(r,i,t)*xs.l(r,i,t) - pd.l(r,i,t)*xds.l(r,i,t))/pet.l(r,i,t) ;

loop(t0,
   xsFlag(r,i)$xs.l(r,i,t0)   = 1 ;
   xdFlag(r,i)$xds.l(r,i,t0)  = 1 ;
   xmtFlag(r,i)$xmt.l(r,i,t0) = 1 ;
   xetFlag(r,i)$xet.l(r,i,t0) = 1 ;
) ;

* --------------------------------------------------------------------------------------------------
*
*  Margins module
*
* --------------------------------------------------------------------------------------------------

ptmg.l(m,t) = 1 ;
loop(tmg,
   xtmg.l(m,t) = sum(r, pa.l(r,m,tmg,t)*xa.l(r,m,tmg,t)) / ptmg.l(m,t) ;
   xwmg.l(r,i,rp,t) = tmarg.l(r,i,rp,t)*xw.l(r,i,rp,t) ;
   xmgm.l(m,r,i,rp,t)
      = sum((m0,i0)$(mapi0(m0,m) and mapi0(i0,i)), inScale*VTWR(m0,i0,r,rp)) / ptmg.l(m,t) ;
   pwmg.l(r,i,rp,t)$xwmg.l(r,i,rp,t)
      = sum(m, ptmg.l(m,t)*xmgm.l(m,r,i,rp,t)) / xwmg.l(r,i,rp,t) ;
) ;

loop(t0,
   mFlag(m)$xtmg.l(m,t0) = 1 ;
) ;

lambdamg.fx(m,r,i,rp,t) = 1 ;

* --------------------------------------------------------------------------------------------------
*
*  Income distribution
*
* --------------------------------------------------------------------------------------------------

YTAX.l(r,"pt",t) = sum((a,i), prdtx.l(r,a,i,t)*p.l(r,a,i,t)*x.l(r,a,i,t)) ;
YTAX.l(r,"fc",t) = sum((i,a), dintx.l(r,i,a,t)*pd.l(r,i,t)*xd.l(r,i,a,t)
                 +            mintx.l(r,i,a,t)*pmt.l(r,i,t)*xm.l(r,i,a,t)) ;
YTAX.l(r,"pc",t) = sum((i,h), dintx.l(r,i,h,t)*pd.l(r,i,t)*xd.l(r,i,h,t)
                 +            mintx.l(r,i,h,t)*pmt.l(r,i,t)*xm.l(r,i,h,t)) ;
YTAX.l(r,"gc",t) = sum((i,gov), dintx.l(r,i,gov,t)*pd.l(r,i,t)*xd.l(r,i,gov,t)
                 +              mintx.l(r,i,gov,t)*pmt.l(r,i,t)*xm.l(r,i,gov,t)) ;
YTAX.l(r,"ic",t) = sum((i,inv), dintx.l(r,i,inv,t)*pd.l(r,i,t)*xd.l(r,i,inv,t)
                 +              mintx.l(r,i,inv,t)*pmt.l(r,i,t)*xm.l(r,i,inv,t)) ;
YTAX.l(r,"et",t) = sum((i,rp), (exptx.l(r,i,rp,t) + etax.l(r,i,t))*pe.l(r,i,rp,t)*xw.l(r,i,rp,t)) ;
YTAX.l(r,"mt",t) = sum((i,rp), (imptx.l(rp,i,r,t)
                 +   mtax.l(r,i,t))*pmcif.l(rp,i,r,t)*xw.l(rp,i,r,t)) ;
YTAX.l(r,"ft",t) = sum((fp,a), fcttx.l(r,fp,a,t)*pf.l(r,fp,a,t)*xf.l(r,fp,a,t)) ;
YTAX.l(r,"fs",t) = sum((fp,a), fctts.l(r,fp,a,t)*pf.l(r,fp,a,t)*xf.l(r,fp,a,t)) ;

YTAX.l(r,"dt",t) = sum((a,fp), kappaf.l(r,fp,a,t)*pf.l(r,fp,a,t)*xf.l(r,fp,a,t)) ;

yTaxTot.l(r,t)   = sum(gy, ytax.l(r,gy,t)) ;
yTaxInd.l(r,t)   = yTaxTot.l(r,t) - ytax.l(r,"dt",t) ;
kstock.l(r,t)    = inScale*VKB(r) ;
fdepr(r,t)       = inScale*VDEP(r)/(pi.l(r,t)*kstock.l(r,t)) ;
depr(r,t)        = fdepr(r,t) ;
loop(cap,
   krat(r,t)     = xft.l(r,cap,t)/kstock.l(r,t) ;
) ;

loop((h,inv,gov),
   rsav.l(r,t) = inScale*SAVE(r) ;
   savf.l(r,t) = sum((i,rp), pmCIF.l(rp,i,r,t)*xw.l(rp,i,r,t) - peFOB.l(r,i,rp,t)*xw.l(r,i,rp,t))
               - sum(i0, inScale*vst(i0,r)) ;
) ;

work = sum((r,t0)$(not rres(r)), savf.l(r,t0)) ;
savf.l(rres,t) = -work ;

xigbl.l(t)     = sum(r, xi.l(r,t) - depr(r,t)*kstock.l(r,t)) ;
chiInv.l(r,t)  = (xi.l(r,t) - depr(r,t)*kstock.l(r,t))/xigbl.l(t) ;
pigbl.l(t)     = sum(r, pi.l(r,t)*(xi.l(r,t) - depr(r,t)*kstock.l(r,t)))/xigbl.l(t) ;
invwgt(r,t)    = pi.l(r,t)*(xi.l(r,t) - depr(r,t)*kstock.l(r,t))
               / sum(rp, pi.l(rp,t)*(xi.l(rp,t) - depr(rp,t)*kstock.l(rp,t))) ;
savwgt(r,t)    = rsav.l(r,t) / sum(rp, rsav.l(rp,t)) ;
chiSave.l(t)   = 1 ;
psave.l(r,t)   = 1 ;
factY.l(r,t)   = sum((fp,a), pf.l(r,fp,a,t)*xf.l(r,fp,a,t)) - fdepr(r,t)*pi.l(r,t)*kstock.l(r,t) ;
regY.l(r,t)    = factY.l(r,t) + yTaxInd.l(r,t) ;
chif.l(r,t)    = savf.l(r,t) / regY.l(r,t) ;

savfBar(r,t)   = savf.l(r,t)/pigbl.l(t) ;

* --------------------------------------------------------------------------------------------------
*
*  Emissions module
*
* --------------------------------------------------------------------------------------------------

*  Production emissions

emid.l(r,i,a,t) = sum((i0,a0)$(mapi0(i0,i) and mapa0(a0,a)), mdf(i0, a0, r)) ;
emim.l(r,i,a,t) = sum((i0,a0)$(mapi0(i0,i) and mapa0(a0,a)), mmf(i0, a0, r)) ;

*  Household emissions

emid.l(r,i,h,t) = sum(i0$mapi0(i0,i), mdp(i0, r)) ;
emim.l(r,i,h,t) = sum(i0$mapi0(i0,i), mmp(i0, r)) ;

*  Government emissions

emid.l(r,i,gov,t) = sum(i0$mapi0(i0,i), mdg(i0, r)) ;
emim.l(r,i,gov,t) = sum(i0$mapi0(i0,i), mmg(i0, r)) ;

*  Investment emissions

emid.l(r,i,inv,t) = sum(i0$mapi0(i0,i), mdi(i0, r)) ;
emim.l(r,i,inv,t) = sum(i0$mapi0(i0,i), mmi(i0, r)) ;

* --------------------------------------------------------------------------------------------------
*
*  Initialize model parameters
*
* --------------------------------------------------------------------------------------------------

loop(t0,

*  sigmap is top level subsitution elasticity, by default equal to esubt
*  use production weights

   tvom(a0,r) = sum(i0, vdfp(i0,a0,r) + vmfp(i0,a0,r)) + sum(fp, evfp(fp,a0,r)) ;
   sigmap(r,a)$(sigmap(r,a) eq na and xp.l(r,a,t0))
      = sum(a0$mapa0(a0,a), tvom(a0,r)*esubt(a0,r))/sum(a0$mapa0(a0,a), tvom(a0,r)) ;

*  sigmand by default is equal to esubc

   tvom(a0,r) = sum(i0, vdfp(i0,a0,r) + vmfp(i0,a0,r)) ;
   sigmand(r,a)$(sigmand(r,a) eq na and nd.l(r,a,t0))
      = sum(a0$mapa0(a0,a), tvom(a0,r)*esubc(a0,r))/sum(a0$mapa0(a0,a), tvom(a0,r)) ;

*  sigmav is subsitution across factors, by default equal to esubva
*  use value added weights
   tvom(a0,r) = sum(fp, evfp(fp,a0,r)) ;
   sigmav(r,a)$(sigmav(r,a) eq na and va.l(r,a,t0))
      = sum(a0$mapa0(a0,a), tvom(a0,r)*esubva(a0,r))/sum(a0$mapa0(a0,a), tvom(a0,r)) ;

*  sigmam is region and agent specific, by default set to esubd

   sigmam(r,i,a)$(sigmam(r,i,a) eq na and xa.l(r,i,a,t0))
      = sum((i0,a0)$(mapi0(i0,i) and mapa0(a0,a)), (vdfp(i0,a0,r)+vmfp(i0,a0,r))*esubd(i0,r))
      / sum((i0,a0)$(mapi0(i0,i) and mapa0(a0,a)), (vdfp(i0,a0,r)+vmfp(i0,a0,r))) ;
   sigmam(r,i,h)$(sigmam(r,i,h) eq na and xa.l(r,i,h,t0))
      = sum(i0$mapi0(i0,i), (vdpp(i0,r)+vmpp(i0,r))*esubd(i0,r))
      / sum(i0$mapi0(i0,i), (vdpp(i0,r)+vmpp(i0,r))) ;
   sigmam(r,i,gov)$(sigmam(r,i,gov) eq na and xa.l(r,i,gov,t0))
      = sum(i0$mapi0(i0,i), (vdgp(i0,r)+vmgp(i0,r))*esubd(i0,r))
      / sum(i0$mapi0(i0,i), (vdgp(i0,r)+vmgp(i0,r))) ;
   sigmam(r,i,inv)$(sigmam(r,i,inv) eq na and xa.l(r,i,inv,t0))
      = sum(i0$mapi0(i0,i), (vdip(i0,r)+vmip(i0,r))*esubd(i0,r))
      / sum(i0$mapi0(i0,i), (vdip(i0,r)+vmip(i0,r))) ;

*  sigmam for trade margins does not exist in the standard GTAP model and is mostly
*  irrelevant since there are no imports for this activity. Set it to the investment elasticity

   sigmam(r,i,tmg)$(sigmam(r,i,tmg)) = sum(inv, sigmam(r,i,inv)) ;

*  sigmaw is second level Armington subsitution elasticity, by default equal to esubm
*  use import weights

   sigmaw(r,i)$(sigmaw(r,i) eq na and xmt.l(r,i,t0))
      = sum((i0,rp)$mapi0(i0,i), vcif(i0, rp, r)*esubm(i0,r))
      / sum((i0,rp)$mapi0(i0,i), vcif(i0, rp, r)) ;

*  eh0 and bh0 are the CDE expansion and substitution parameters, by default equal
*  to incpar and subpar respectively. Use consumption weights

   loop(h,
      eh0(r,i)$(eh0(r,i) eq na and xa.l(r,i,h,t0))
         = sum(i0$mapi0(i0,i), incpar(i0,r)*(vdpp(i0,r) + vmpp(i0,r)))
         / sum(i0$mapi0(i0,i), (vdpp(i0,r) + vmpp(i0,r))) ;
      bh0(r,i)$(bh0(r,i) eq na and xa.l(r,i,h,t0))
         = sum(i0$mapi0(i0,i), subpar(i0,r)*(vdpp(i0,r) + vmpp(i0,r)))
         / sum(i0$mapi0(i0,i), (vdpp(i0,r) + vmpp(i0,r))) ;
   ) ;

*  The GEMPACK version assumes CET elasticities are negative

   omegaf(r,fp)$(omegaf(r,fp) eq na) = -etrae(fp,r) ;
) ;

$ontext
*  DvdM--12-Dec-2016
*  Aggregation is now done in aggregation facility

loop(t0,

*  sigmap is top level subsitution elasticity, by default equal to esubt
*  use production weights

   sigmap(r,a)$(sigmap(r,a) eq na and xp.l(r,a,t0)) = esubt(a,r) ;

*  sigmand by default is equat to sigmap

   sigmand(r,a)$(sigmand(r,a) eq na) = sigmap(r,a) ;

*  sigmav is subsitution across factors, by default equal to esubva
*  use value added weights

   sigmav(r,a)$(sigmav(r,a) eq na and va.l(r,a,t0)) = esubva(a,r) ;

*  sigmam is region and agent specific, by default set to esubd

   sigmam(r,i,a)$(sigmam(r,i,a) eq na and xa.l(r,i,a,t0)) = esubd(i,r) ;
   sigmam(r,i,h)$(sigmam(r,i,h) eq na and xa.l(r,i,h,t0)) = esubd(i,r) ;
   sigmam(r,i,gov)$(sigmam(r,i,gov) eq na and xa.l(r,i,gov,t0)) = esubd(i,r) ;
   sigmam(r,i,inv)$(sigmam(r,i,inv) eq na and xa.l(r,i,inv,t0)) = esubd(i,r) ;

*  sigmam for trade margins does not exist in the standard GTAP model and is mostly
*  irrelevant since there are no imports for this activity. Set it to the investment elasticity

   sigmam(r,i,tmg)$(sigmam(r,i,tmg)) = sum(inv, sigmam(r,i,inv)) ;

*  sigmaw is second level Armington subsitution elasticity, by default equal to esubm
*  use import weights

   sigmaw(r,i)$(sigmaw(r,i) eq na and xmt.l(r,i,t0)) = esubm(i,r) ;

*  eh0 and bh0 are the CDE expansion and substitution parameters, by default equal
*  to incpar and subpar respectively. Use consumption weights

   loop(h,
*     If we don't have overrides, then aggregate
      eh0(r,i)$(eh0(r,i) eq na) = incpar(i,r) ;
      bh0(r,i)$(bh0(r,i) eq na) = subpar(i,r) ;
   ) ;
) ;
$offtext

* --------------------------------------------------------------------------------------------------
*
*  Income allocation module
*
* --------------------------------------------------------------------------------------------------

eh.fx(r,i,t) = eh0(r,i) ;
bh.fx(r,i,t) = bh0(r,i) ;
phiP.fx(r,t) = sum((i,h), xcshr.l(r,i,h,t)*eh.l(r,i,t))$(%utility% eq CDE)
             + sum((i,h), xcshr.l(r,i,h,t))$(%utility% eq CD) ;

display eh.l, bh.l, eh0, bh0, phiP.l, xcshr.l ;

kron(is,is) = 1 ; display kron ;

loop(h,
   ued.l(r,i,j,t)   = xcshr.l(r,i,h,t)*(-bh.l(r,i,t)
                    - (eh.l(r,i,t)*bh.l(r,i,t)
                    - sum(jp, xcshr.l(r,jp,h,t)*eh.l(r,jp,t)*bh.l(r,jp,t)))
                    /  sum(jp, xcshr.l(r,jp,h,t)*eh.l(r,jp,t))) + kron(i,j)*(bh.l(r,i,t) - 1) ;

   incelas.l(r,i,t) = (eh.l(r,i,t)*bh.l(r,i,t)
                    - sum(jp, xcshr.l(r,jp,h,t)*eh.l(r,jp,t)*bh.l(r,jp,t)))
                    /   sum(jp, xcshr.l(r,jp,h,t)*eh.l(r,jp,t))
                    -  (bh.l(r,i,t) - 1) + sum(jp, xcshr.l(r,jp,h,t)*bh.l(r,jp,t)) ;

   ced.l(r,i,j,t)   = ued.l(r,i,j,t) + xcshr.l(r,j,h,t) * incelas.l(r,i,t) ;

   ape.l(r,i,j,t)$xcshr.l(r,j,h,t)
                    = 1 - bh.l(r,j,t) - bh.l(r,i,t) + sum(jp, xcshr.l(r,jp,h,t)*bh.l(r,jp,t))
                    -  kron(i,j)*(1-bh.l(r,i,t))/xcshr.l(r,j,h,t) ;
) ;

*  Initialize parameters

betaP.l(r,t) = yc.l(r,t)/regY.l(r,t) ;
betaG.l(r,t) = yg.l(r,t)/regY.l(r,t) ;
betaS.l(r,t) = rsav.l(r,t)/regY.l(r,t) ;
phi.l(r,t)   = 1/(phiP.l(r,t)*betaP.l(r,t) + betaG.l(r,t) + betaS.l(r,t)) ;

*  Fix nominal levels

yc.fx(r,t)   = yc.l(r,t) ;
yg.fx(r,t)   = yg.l(r,t) ;
rsav.fx(r,t) = rsav.l(r,t) ;
regY.fx(r,t) = (yc.l(r,t) + yg.l(r,t) + rsav.l(r,t)) ;

rs(r) = yes ;
loop(tsim$t0(tsim),
   ts(tsim) = yes ;
   options limrow=0, limcol=0 ;
   solve betaCal using mcp ;
   ts(tsim) = no ;
) ;
rs(r) = no ;

*  Fix parameters
betaP.fx(r,t) = betaP.l(r,t) ;
betaG.fx(r,t) = betaG.l(r,t) ;
betaS.fx(r,t) = betaS.l(r,t) ;

*  Free nominal variables
phiP.lo(r,t) = -inf ; phiP.up(r,t) = + inf ;
yc.lo(r,t)   = -inf ; yc.up(r,t) = + inf ;
yg.lo(r,t)   = -inf ; yg.up(r,t) = + inf ;
rsav.lo(r,t) = -inf ; rsav.up(r,t) = + inf ;
regY.lo(r,t) = -inf ; regY.up(r,t) = + inf ;

yi.l(r,t) = pi.l(r,t)*depr(r,t)*kstock.l(r,t) + rsav.l(r,t) + savf.l(r,t) ;

* --------------------------------------------------------------------------------------------------
*
*  Factors
*
* --------------------------------------------------------------------------------------------------

loop(t0,
   xftFlag(r,fm)$xft.l(r,fm,t0) = 1 ;
) ;

rorFlag       = %savfFlag% ;

loop(cap,
   arent.l(r,t) = krat(r,t)*sum(a, (1-kappaf.l(r,cap,a,t))*pf.l(r,cap,a,t)*xf.l(r,cap,a,t))
                /    sum((a), xf.l(r,cap,a,t)) ;
) ;

kapEnd.l(r,t) = (1-depr(r,t))*kstock.l(r,t) + xi.l(r,t) ;
rorc.l(r,t)   = arent.l(r,t)/pi.l(r,t) - fdepr(r,t) ;
rore.l(r,t)   = rorc.l(r,t)*(kstock.l(r,t)/kapEnd.l(r,t))**RoRFlex(r,t) ;
rorg.l(t)     = sum(r, rore.l(r,t)*pi.l(r,t)*(xi.l(r,t) - depr(r,t)*kstock.l(r,t)))
              / sum(rp, pi.l(rp,t)*(xi.l(rp,t) - depr(rp,t)*kstock.l(rp,t))) ;
risk(r,t)     = rorg.l(t) / rore.l(r,t) ;

* display arent.l, kapEnd.l, rorc.l, rore.l, rorg.l, risk ;

* --------------------------------------------------------------------------------------------------
*
*  Closure
*
* --------------------------------------------------------------------------------------------------

loop(t0,
   dintx0(r,i,aa) = dintx.l(r,i,aa,t0) ;
   mintx0(r,i,aa) = mintx.l(r,i,aa,t0) ;
) ;
dtxshft.fx(r,i,aa,t) = 0 ;
mtxshft.fx(r,i,aa,t) = 0 ;
rtxshft.fx(r,aa,t)   = 0 ;
intxFlag(r,i,aa)     = 0 ;

ytaxshr.l(r,gy,t)    = ytax.l(r,gy,t) / regY.l(r,t) ;

gdpmp.l(r,t) = (sum((i,fd), pa.l(r,i,fd,t)*xa.l(r,i,fd,t))
             +  sum((i,rp), pefob.l(r,i,rp,t)*xw.l(r,i,rp,t) - pmcif.l(rp,i,r,t)*xw.l(rp,i,r,t))) ;
rgdpmp.l(r,t) = gdpmp.l(r,t) ;
pgdpmp.l(r,t) = gdpmp.l(r,t)/rgdpmp.l(r,t) ;

ggdppc.l(r,t) = 0 ;
gl.l(r,t)     = 0 ;

* --------------------------------------------------------------------------------------------------
*
*  Emissions module
*
* --------------------------------------------------------------------------------------------------

$ontext
emid.l(r,i,aa) = emid0(r,i,aa) ;
emii.l(r,i,aa) = emii0(r,i,aa) ;
$offtext

* --------------------------------------------------------------------------------------------------
*
*  Calibration of parameters
*
* --------------------------------------------------------------------------------------------------

*  Domestic production

and(r,a,t)$ndFlag(r,a) = (nd.l(r,a,t)/xp.l(r,a,t))*(pnd.l(r,a,t)/px.l(r,a,t))**sigmap(r,a) ;
ava(r,a,t)$vaFlag(r,a) = (va.l(r,a,t)/xp.l(r,a,t))*(pva.l(r,a,t)/px.l(r,a,t))**sigmap(r,a) ;

io(r,i,a,t)$xaFlag(r,i,a)   = (xa.l(r,i,a,t)/nd.l(r,a,t))
                            * (pa.l(r,i,a,t)/pnd.l(r,a,t))**sigmand(r,a) ;
af(r,fp,a,t)$xfFlag(r,fp,a) = (xf.l(r,fp,a,t)/va.l(r,a,t))
                            * (pfa.l(r,fp,a,t)/pva.l(r,a,t))**sigmav(r,a) ;

*  Make module

gx(r,a,i,t)$xFlag(r,a,i)
   = ((x.l(r,a,i,t)/xp.l(r,a,t))*(px.l(r,a,t)/p.l(r,a,i,t))**omegas(r,a))$(omegas(r,a) ne inf)
   + ((p.l(r,a,i,t)*x.l(r,a,i,t))/(px.l(r,a,t)*xp.l(r,a,t)))$(omegas(r,a) eq inf) ;

ax(r,a,i,t)$xFlag(r,a,i)
   = ((x.l(r,a,i,t)/xs.l(r,i,t))*((1+prdtx.l(r,a,i,t))*p.l(r,a,i,t)/ps.l(r,i,t))**sigmas(r,i))
   $(sigmas(r,i) ne inf)
   + (((1+prdtx.l(r,a,i,t))*p.l(r,a,i,t)*x.l(r,a,i,t))/(ps.l(r,i,t)*xs.l(r,i,t)))
   $(sigmas(r,i) eq inf) ;

if(%utility% eq CDE,

*  CDE utility function

   alphaa(r,i,h,t)$xaFlag(r,i,h) = ((xcshr.l(r,i,h,t)/bh.l(r,i,t))
                                 *  (((yc.l(r,t)/pop.l(r,t))/pa.l(r,i,h,t))**bh.l(r,i,t))
                                 *  (uh.l(r,h,t)**(-eh.l(r,i,t)*bh.l(r,i,t))))
                                 /  (sum(j$bh.l(r,j,t), xcshr.l(r,j,h,t)/bh.l(r,j,t))) ;

   zcons.l(r,i,h,t)$xaFlag(r,i,h) = alphaa(r,i,h,t)*bh.l(r,i,t)
                 * (uh.l(r,h,t)**(eh.l(r,i,t)*bh.l(r,i,t)))
                 * (pa.l(r,i,h,t)**(bh.l(r,i,t)))
                 * ((yc.l(r,t)/pop.l(r,t))**(-bh.l(r,i,t))) ;

elseif(%utility% eq CD),

*  CD utility function

   alphaa(r,i,h,t)$xaFlag(r,i,h) = xcshr.l(r,i,h,t) ;

*  Get them to add to one

   alphaa(r,i,h,t)$xaFlag(r,i,h)  = alphaa(r,i,h,t)/sum(j, alphaa(r,j,h,t)) ;
   zcons.l(r,i,h,t)$xaFlag(r,i,h) = alphaa(r,i,h,t) ;
   auh(r,h,t) = uh.l(r,h,t)/prod(i$xaFlag(r,i,h), xa.l(r,i,h,t)**alphaa(r,i,h,t)) ;

) ;

*  Government demand

*  sigmag is top level subsitution elasticity, by default equal to esubg

sigmag(r)$(sigmag(r) eq na) = esubg(r) ;
sigmag(r)$(sigmag(r) eq 1)  = 1.01 ;

alphaa(r,i,gov,t)$xaFlag(r,i,gov)
   = (xa.l(r,i,gov,t)/xg.l(r,t))*(pa.l(r,i,gov,t)/pg.l(r,t))**sigmag(r) ;

aug.fx(r,t) = ug.l(r,t)*pop.l(r,t)/xg.l(r,t) ;

loop(gov,
   axg(r,t)$(sigmag(r) ne 1) = 1 ;
   axg(r,t)$(sigmag(r) eq 1) = (prod(i$(alphaa(r,i,gov,t) ne 0),
               (pa.l(r,i,gov,t)/alphaa(r,i,gov,t))**alphaa(r,i,gov,t)))/pg.l(r,t) ;
) ;

*  Investment/savings

*  sigmai is top level subsitution elasticity, by default equal to esubi

sigmai(r)$(sigmai(r) eq na) = esubi(r) ;
sigmai(r)$(sigmai(r) eq 1)  = 1.01 ;

alphaa(r,i,inv,t)$xaFlag(r,i,inv)
   = (xa.l(r,i,inv,t)/xi.l(r,t))*(pa.l(r,i,inv,t)/pi.l(r,t))**sigmai(r) ;

aus.fx(r,t) = us.l(r,t)*pop.l(r,t)/(rsav.l(r,t)/psave.l(r,t)) ;

lambdai.fx(r,i,t) = 1 ;
loop(inv,
   axi(r,t)$(sigmai(r) ne 1) = 1 ;
   axi(r,t)$(sigmai(r) eq 1) = (prod(i$(alphaa(r,i,inv,t) ne 0),
               (pa.l(r,i,inv,t)/alphaa(r,i,inv,t))**alphaa(r,i,inv,t)))/pi.l(r,t) ;
) ;

u.l(r,t)   = 1 ;
au.fx(r,t) = u.l(r,t)*(sum(h,uh.l(r,h,t))**(-betaP.l(r,t)))
           *   (ug.l(r,t)**(-betaG.l(r,t)))
           *   (us.l(r,t)**(-betaS.l(r,t))) ;

*  Top level Armington demand

alphad(r,i,aa,t)$xaFlag(r,i,aa) =
      (xd.l(r,i,aa,t)/xa.l(r,i,aa,t))*(pdp.l(r,i,aa,t)/pa.l(r,i,aa,t))**sigmam(r,i,aa) ;
alpham(r,i,aa,t)$xaFlag(r,i,aa) =
      (xm.l(r,i,aa,t)/xa.l(r,i,aa,t))*(pmp.l(r,i,aa,t)/pa.l(r,i,aa,t))**sigmam(r,i,aa) ;

*  Second level Armington

amw(rp,i,r,t)$xwFlag(rp,i,r) = (xw.l(rp,i,r,t)/xmt.l(r,i,t))
                             *   (pm.l(rp,i,r,t)/pmt.l(r,i,t))**sigmaw(r,i) ;

*  Top level CET

display xdFlag, xetFlag, xsFlag, xs.l ;
gd(r,i,t)$(xdFlag(r,i) and omegax(r,i) ne inf)  = (xds.l(r,i,t)/xs.l(r,i,t))
                                                * (ps.l(r,i,t)/pd.l(r,i,t))**omegax(r,i) ;
ge(r,i,t)$(xetFlag(r,i) and omegax(r,i) ne inf) = (xet.l(r,i,t)/xs.l(r,i,t))
                                                * (ps.l(r,i,t)/pet.l(r,i,t))**omegax(r,i) ;

gd(r,i,t)$(xdFlag(r,i) and omegax(r,i) eq inf)  = (pd.l(r,i,t)*xds.l(r,i,t)
                                                / (ps.l(r,i,t)*xs.l(r,i,t))) ;
ge(r,i,t)$(xetFlag(r,i) and omegax(r,i) eq inf) = (pet.l(r,i,t)*xet.l(r,i,t)
                                                / (ps.l(r,i,t)*xs.l(r,i,t))) ;

*  Second level CET

gw(r,i,rp,t)$(xwFlag(r,i,rp) and omegaw(r,i) ne inf) = (xw.l(r,i,rp,t)/xet.l(r,i,t))
                                                     * (pet.l(r,i,t)/pe.l(r,i,rp,t))**omegaw(r,i) ;
gw(r,i,rp,t)$(xwFlag(r,i,rp) and omegaw(r,i) eq inf) = (pe.l(r,i,rp,t)*xw.l(r,i,rp,t)
                                                     / (pet.l(r,i,t)*xet.l(r,i,t))) ;

*  TT services

loop(t$t0(t),
   amgm(m,r,i,rp)$xwmg.l(r,i,rp,t) = xmgm.l(m,r,i,rp,t)/xwmg.l(r,i,rp,t) ;
) ;

*  sigmamg is top level subsitution elasticity, by default equal to esubs

Loop(t0,
   work = sum((r,rp,m0,i0), VTWR(m0,i0,r,rp)) ;
   sigmamg(m)$(sigmamg(m) eq na and work)
      = sum((r,rp,m0,i0)$mapi0(m0,m), VTWR(m0,i0,r,rp)*esubs(m0)) / work ;
) ;

sigmamg(m)$(sigmamg(m) eq 1) = 1.01 ;

alphaa(r,m,tmg,t)$mflag(m) = (xa.l(r,m,tmg,t)/xtmg.l(m,t))
                           * (pa.l(r,m,tmg,t)/ptmg.l(m,t))**sigmamg(m) ;
loop(tmg,
   axmg(m,t)$(sigmamg(m) eq 1) = prod(r$alphaa(r,m,tmg,t),
      (pa.l(r,m,tmg,t)/alphaa(r,m,tmg,t))**(alphaa(r,m,tmg,t)))/ptmg.l(m,t) ;
) ;
axmg(m,t)$(sigmamg(m) ne 1) = 1 ;

*  Factor markets

aft(r,fm,t) = xft.l(r,fm,t)*(pabs.l(r,t)/pft.l(r,fm,t))**etaf(r,fm) ;

if(1,
   gf(r,fm,a,t)$(xfFlag(r,fm,a) and omegaf(r,fm) ne inf) =
      (xf.l(r,fm,a,t)/xft.l(r,fm,t))*(pft.l(r,fm,t)/pfy.l(r,fm,a,t))**omegaf(r,fm) ;
   gf(r,fm,a,t)$(xfFlag(r,fm,a) and omegaf(r,fm) eq inf) =
      ((pfy.l(r,fm,a,t)*xf.l(r,fm,a,t))/(pft.l(r,fm,t)*xft.l(r,fm,t))) ;
   gf(r,fnm,a,t)$(xfFlag(r,fnm,a)) =
      xf.l(r,fnm,a,t)*(pabs.l(r,t)/pfy.l(r,fnm,a,t))**etaff(r,fnm,a) ;
else
   gf(r,fm,a,t)$(xfFlag(r,fm,a) and omegaf(r,fm) ne inf) =
      (xf.l(r,fm,a,t)/xft.l(r,fm,t))*(pft.l(r,fm,t)/pf.l(r,fm,a,t))**omegaf(r,fm) ;
   gf(r,fm,a,t)$(xfFlag(r,fm,a) and omegaf(r,fm) eq inf) =
      ((pf.l(r,fm,a,t)*xf.l(r,fm,a,t))/(pft.l(r,fm,t)*xft.l(r,fm,t))) ;
   gf(r,fnm,a,t)$(xfFlag(r,fnm,a)) =
      xf.l(r,fnm,a,t)*(pabs.l(r,t)/pf.l(r,fnm,a,t))**etaff(r,fnm,a) ;
) ;

ggdppcT(r,t) = 0 ;

$ontext
walras.l = sum((r,t)$(rres(r) and t0(t)), yi.l(r,t) -
   (pi.l(r,t)*depr(r,t)*kstock.l(r,t) + rsav.l(r,t) + pigbl.l(t)*savf.l(r,t))) ;
display walras.l ;
abort$(1) "Temp" ;
$offtext

* --------------------------------------------------------------------------------------------------
*
*  Rescale production side variables
*
* --------------------------------------------------------------------------------------------------

xd.l(r,i,a,t)  = xScale(r,a)*xd.l(r,i,a,t) ;
xm.l(r,i,a,t)  = xScale(r,a)*xm.l(r,i,a,t) ;
xa.l(r,i,a,t)  = xScale(r,a)*xa.l(r,i,a,t) ;
xf.l(r,fp,a,t) = xScale(r,a)*xf.l(r,fp,a,t) ;
xp.l(r,a,t)    = xScale(r,a)*xp.l(r,a,t) ;
va.l(r,a,t)    = xScale(r,a)*va.l(r,a,t) ;
nd.l(r,a,t)    = xScale(r,a)*nd.l(r,a,t) ;

* -------------------------------------------------------------------------
*
*  Run the simulations for each time period
*
* -------------------------------------------------------------------------

rs(r) = yes ;
ts(t) = no ;

loop(tsim,

   ts(tsim) = yes ;
* --------------------------------------------------------------------------------------------------
*
*  Code implemented between solution periods
*
* --------------------------------------------------------------------------------------------------

if(years(tsim) gt FirstYear and ifDyn,

*  Update variables

*  Calculate the growth of total GDP by region

   rwork(r) = power(1 + ggdppcT(r,tsim), gap(tsim))*pop.l(r,tsim)/pop.l(r,tsim-1) ;


$macro initVar0(x, pFlag)              x.l(r, tsim) = x.l(r, tsim-1)*(1$pFlag + rwork(r)$(not pFlag))
$macro initVar1(x,i__1, pFlag)         x.l(r, i__1, tsim) = x.l(r, i__1, tsim-1)*(1$pFlag + rwork(r)$(not pFlag))
$macro initVar2(x,i__1, i__2, pFlag)   x.l(r, i__1, i__2, tsim) = x.l(r, i__1, i__2, tsim-1)*(1$pFlag + rwork(r)$(not pFlag))

$macro initVar0g(x, pFlag)             x.l(tsim) = x.l(tsim-1)*(1$pFlag + 1$(not pFlag))
$macro initVar1g(x,i__1, pFlag)        x.l(i__1, tsim) = x.l(i__1, tsim-1)*(1$pFlag + 1$(not pFlag))
$macro initVar4g(x,i__1, i__2, i__3, i__4, pFlag)   x.l(i__1, i__2, i__3, i__4, tsim) = x.l(i__1, i__2, i__3, i__4, tsim-1)*(1$pFlag + 1$(not pFlag))

initvar1(nd,a,0) ;
initvar1(va,a,0) ;
initvar1(px,a,1) ;

initvar2(xa,i,aa,0) ;
initvar1(pnd,a,1) ;
initvar2(xf,fp,a,0) ;
initvar1(pva,a,1) ;

initvar2(x,a,i,0) ;
initvar1(xp,a,0) ;
initvar2(pp,a,i,1) ;
initvar2(p,a,i,1) ;
initvar1(ps,i,1) ;

initvar1(ytax,gy,0) ;
initvar0(ytaxTot,0) ;
initvar0(yTaxInd,0) ;
initvar0(factY,0) ;
initvar0(regY,0) ;

initvar0(rsav,0) ;
initvar0(yg,0) ;
initvar0(yc,0) ;
initvar0(phi,1) ;
initvar0(u,0) ;
initvar0(us,0) ;

initvar2(zcons,i,h,0) ;
initvar2(xcshr,i,h,1) ;
*  initvar2(xa,i,h,0) ;
initvar0(phiP,1) ;
initvar0(pcons,1) ;
initvar1(uh,h,1) ;

*  initvar2(xa,i,gov,0) ;
initvar0(pg,1) ;
initvar0(xg,0) ;
initvar0(ug,0) ;

*  initvar2(xa,i,inv,0) ;
initvar0(pi,1) ;
initvar0(xi,0) ;

initvar2(pdp,i,aa,1) ;
initvar2(pmp,i,aa,1) ;
initvar2(pa,i,aa,1) ;
initvar2(xd,i,aa,0) ;
initvar2(xm,i,aa,0) ;

initvar1(xmt,i,0) ;
initvar2(xw,i,rp,0) ;
initvar1(pmt,i,1) ;
initvar1(xds,i,1) ;
initvar1(xet,i,0) ;
initvar1(xs,i,0) ;
initvar2(pe,i,rp,1) ;
initvar1(pet,i,1) ;

initvar2(xwmg,i,rp,0) ;

initvar4g(xmgm,m,r,i,rp,1) ;
initvar2(pwmg,i,rp,1) ;
initvar1g(xtmg,m,1) ;
*  initvar2(xa,i,tmg,0) ;
initvar1g(ptmg,m,1) ;

initvar2(pefob,i,rp,1) ;
initvar2(pmcif,i,rp,1) ;
initvar2(pm,i,rp,1) ;

initvar1(pd,i,1) ;

initvar1(xft,fm,0) ;
initvar2(pf,fp,a,1) ;
initvar1(pft,fm,1) ;
initvar2(pfa,fp,a,1) ;

initvar0(kstock,0) ;
initvar0(kapEnd,0) ;
initvar0(arent,1) ;
initvar0(rorc,1) ;
initvar0(rore,1) ;
initvar0(savf,1) ;
initvar0g(rorg,1) ;
initvar0(yi,0) ;
initvar0g(xigbl,0) ;
initvar0g(pigbl,1) ;
initvar0g(chiSave,1) ;
initvar0(psave,1) ;

initvar0(pabs,1) ;
initvar0g(pmuv,1) ;
initvar0(pfact,1) ;
initvar0g(pwfact,1) ;
initvar0g(pnum,1) ;

initvar2(dintx,i,aa,1) ;
initvar2(mintx,i,aa,1) ;
initvar1(ytaxshr,gy,1) ;

initvar0(gdpmp,0) ;
initvar0(rgdpmp,0) ;
initvar0(pgdpmp,1) ;

initvar1(axp,a,1) ;
initvar1(lambdand,a,1) ;
initvar1(lambdava,a,1) ;
initvar2(lambdaf,fp,a,1) ;
initvar2(lambdaio,i,a,1) ;

initvar0(ggdppc,1) ;
initvar0(gl,1) ;
initvar2(afeall,fp,a,1) ;

initvar2(ued,i,j,1) ;
initvar1(incElas,i,1) ;
initvar2(ced,i,j,1) ;
initvar2(ape,i,j,1) ;

initvar1(ev,h,1) ;
initvar1(cv,h,1) ;


) ;

*  Closure

*  Policy variables

fctts.fx(r,fp,a,tsim) = fctts.l(r,fp,a,tsim) ;
fcttx.fx(r,fp,a,tsim) = fcttx.l(r,fp,a,tsim) ;
prdtx.fx(r,a,i,tsim)  = prdtx.l(r,a,i,tsim) ;
exptx.fx(r,i,rp,tsim) = exptx.l(r,i,rp,tsim) ;
imptx.fx(r,i,rp,tsim) = imptx.l(r,i,rp,tsim) ;

dtxshft.fx(r,i,aa,tsim) = dtxshft.l(r,i,aa,tsim) ;
mtxshft.fx(r,i,aa,tsim) = mtxshft.l(r,i,aa,tsim) ;
rtxshft.fx(r,aa,tsim)   = rtxshft.l(r,aa,tsim) ;
dintx.fx(r,i,aa,tsim)$(not intxFlag(r,i,aa)) =
   dintx0(r,i,aa) + dtxshft.l(r,i,aa,tsim) + rtxshft.l(r,aa,tsim) ;
mintx.fx(r,i,aa,tsim)$(not intxFlag(r,i,aa)) =
   mintx0(r,i,aa) + mtxshft.l(r,i,aa,tsim) + rtxshft.l(r,aa,tsim) ;

ytaxshr.l(r,gy,t)    = ytax.l(r,gy,t) / regY.l(r,t) ;

*  Fix the numeraire

pnum.fx(t)   = pnum.l(t) ;

*  Capital account closure

if(RoRFlag eq capSFix,
   chif.fx(r,t)$(not rres(r)) = chif.l(r,t) ;
) ;

chiInv.lo(r,t) = -inf ;
chiInv.up(r,t) = +inf ;
if(RoRFlag eq capShrFix,
   chiInv.fx(r,t) = chiInv.l(r,t) ;
) ;

*  Technology variables

tmarg.fx(r,i,rp,tsim) = tmarg.l(r,i,rp,tsim) ;

*  Put a lower bound on prices

px.lo(r,a,tsim)$xpFlag(r,a)             = 0.001*px.l(r,a,tsim-1) ;
pva.lo(r,a,tsim)$vaFlag(r,a)            = 0.001*pva.l(r,a,tsim-1) ;
pnd.lo(r,a,tsim)$ndFlag(r,a)            = 0.001*pnd.l(r,a,tsim-1) ;
p.lo(r,a,i,tsim)$xFlag(r,a,i)           = 0.001*p.l(r,a,i,tsim-1) ;
ps.lo(r,i,tsim)$xsFlag(r,i)             = 0.001*ps.l(r,i,tsim-1) ;
pdp.lo(r,i,aa,tsim)$alphad(r,i,aa,tsim) = 0.001*pdp.l(r,i,aa,tsim-1) ;
pmp.lo(r,i,aa,tsim)$alpham(r,i,aa,tsim) = 0.001*pmp.l(r,i,aa,tsim-1) ;
pa.lo(r,i,aa,tsim)$xaFlag(r,i,aa)       = 0.001*pa.l(r,i,aa,tsim-1) ;
pmt.lo(r,i,tsim)$xmtFlag(r,i)           = 0.001*pmt.l(r,i,tsim-1) ;
pe.lo(r,i,rp,tsim)$xwFlag(r,i,rp)       = 0.001*pe.l(r,i,rp,tsim-1) ;
pefob.lo(r,i,rp,tsim)$xwFlag(r,i,rp)    = 0.001*pefob.l(r,i,rp,tsim-1) ;
pmcif.lo(r,i,rp,tsim)$xwFlag(r,i,rp)    = 0.001*pmcif.l(r,i,rp,tsim-1) ;
pm.lo(r,i,rp,tsim)$xwFlag(r,i,rp)       = 0.001*pm.l(r,i,rp,tsim-1) ;
pet.lo(r,i,tsim)$xetFlag(r,i)           = 0.001*pet.l(r,i,tsim-1) ;
pd.lo(r,i,tsim)$xdFlag(r,i)             = 0.001*pd.l(r,i,tsim-1) ;
pwmg.lo(r,i,rp,tsim)$tmgFlag(r,i,rp)    = 0.001*pwmg.l(r,i,rp,tsim-1) ;
pf.lo(r,fp,a,tsim)$xfFlag(r,fp,a)       = 0.001*pf.l(r,fp,a,tsim-1) ;
pfa.lo(r,fp,a,tsim)$xfFlag(r,fp,a)      = 0.001*pfa.l(r,fp,a,tsim-1) ;
*pft.lo(r,fm,tsim)$xftFlag(r,fm)         = 0.001*pft.l(r,fm,tsim-1) ;

uh.lo(r,h,tsim)  = 0.001*uh.l(r,h,tsim-1) ;
ug.lo(r,tsim)    = 0.001*ug.l(r,tsim-1) ;
us.lo(r,tsim)    = 0.001*us.l(r,tsim-1) ;
u.lo(r,tsim)     = 0.001*u.l(r,tsim-1) ;
pcons.lo(r,tsim) = 0.001*pcons.l(r,tsim-1) ;
pg.lo(r,tsim)    = 0.001*pg.l(r,tsim-1) ;
pi.lo(r,tsim)    = 0.001*pi.l(r,tsim-1) ;
ptmg.lo(m,tsim)  = 0.001*ptmg.l(m,tsim-1) ;

*  Fix zero variables

loop(t0,

   lambdand.fx(r,a,tsim)$(not ndFlag(r,a)) = 1 ;
   lambdava.fx(r,a,tsim)$(not vaFlag(r,a)) = 1 ;
   nd.fx(r,a,tsim)$(not ndFlag(r,a))       = 0 ;
   va.fx(r,a,tsim)$(not vaFlag(r,a))       = 0 ;
   px.fx(r,a,tsim)$(not xpFlag(r,a))       = px.l(r,a,t0) ;

   lambdaio.fx(r,i,a,t)$(not xaFlag(r,i,a)) = 1 ;
   xa.fx(r,i,aa,tsim)$(not xaFlag(r,i,aa))  = 0 ;
   pa.fx(r,i,aa,tsim)$(not xaFlag(r,i,aa))  = pa.l(r,i,aa,tsim) ;
   pnd.fx(r,a,tsim)$(not ndFlag(r,a))       = pnd.l(r,a,t0) ;

   lambdaf.fx(r,fp,a,t)$(not xfFlag(r,fp,a)) = 1 ;
   xf.fx(r,fp,a,tsim)$(not xfFlag(r,fp,a))   = 0 ;
   pf.fx(r,fp,a,tsim)$(not xfFlag(r,fp,a))   = pf.l(r,fp,a,t0) ;
   pva.fx(r,a,tsim)$(not vaFlag(r,a))        = pva.l(r,a,t0) ;

   x.fx(r,a,i,tsim)$(not xFlag(r,a,i)) = 0.0 ;
   p.fx(r,a,i,tsim)$(not xFlag(r,a,i)) = p.l(r,a,i,t0) ;
   xs.fx(r,i,tsim)$(not xsFlag(r,i))   = 0.0 ;
   ps.fx(r,i,tsim)$(not xsFlag(r,i))   = ps.l(r,i,t0) ;

   zcons.fx(r,i,h,t)$(not xaFlag(r,i,h)) = 0 ;
   xcshr.fx(r,i,h,t)$(not xaFlag(r,i,h)) = 0 ;

   xd.fx(r,i,aa,t)$(not alphad(r,i,aa,t))  = 0 ;
   xm.fx(r,i,aa,t)$(not alpham(r,i,aa,t))  = 0 ;
   pdp.fx(r,i,aa,t)$(not alphad(r,i,aa,t)) = pdp.l(r,i,aa,t) ;
   pmp.fx(r,i,aa,t)$(not alpham(r,i,aa,t)) = pmp.l(r,i,aa,t) ;

   xmt.fx(r,i,tsim)$(not xmtFlag(r,i)) = 0 ;
   pmt.fx(r,i,tsim)$(not xmtFlag(r,i)) = pmt.l(r,i,t0) ;

   xw.fx(r,i,rp,tsim)$(not xwFlag(r,i,rp))    = 0 ;
   pe.fx(r,i,rp,tsim)$(not xwFlag(r,i,rp))    = pe.l(r,i,rp,t0) ;
   pefob.fx(r,i,rp,tsim)$(not xwFlag(r,i,rp)) = pefob.l(r,i,rp,t0) ;
   pmcif.fx(r,i,rp,tsim)$(not xwFlag(r,i,rp)) = pmcif.l(r,i,rp,t0) ;
   pm.fx(r,i,rp,tsim)$(not xwFlag(r,i,rp))    = pm.l(r,i,rp,t0) ;

   xwmg.fx(r,i,rp,tsim)$(not tmgFlag(r,i,rp))  = 0 ;
*  Force pwmg=0 on no-flow routes (mirrors 9x10 convention; default cal.gms
*  init leaves pwmg=1 here, causing spurious divergence vs Python).
   pwmg.l(r,i,rp,tsim)$(not tmgFlag(r,i,rp))   = 0 ;
   pwmg.fx(r,i,rp,tsim)$(not tmgFlag(r,i,rp))  = pwmg.l(r,i,rp,tsim) ;
   xmgm.fx(m,r,i,rp,tsim)$(not amgm(m,r,i,rp)) = 0 ;

   xds.fx(r,i,tsim)$(not xdFlag(r,i))  = 0 ;
   pd.fx(r,i,tsim)$(not xdFlag(r,i))   = pd.l(r,i,t0) ;
   xet.fx(r,i,tsim)$(not xetFlag(r,i)) = 0 ;
   pet.fx(r,i,tsim)$(not xetFlag(r,i)) = pet.l(r,i,t0) ;

   pfa.fx(r,fp,a,tsim)$(not xfFlag(r,fp,a)) = pfa.l(r,fp,a,t0) ;
   xft.fx(r,fm,tsim)$(not xftFlag(r,fm))    = 0 ;
   pft.fx(r,fm,tsim)$(not xftFlag(r,fm))    = pft.l(r,fm,t0) ;
) ;

*  Fix lags

if(years(tsim) ne firstYear,
   axp.fx(r,a,tsim-1)        = axp.l(r,a,tsim-1) ;
   lambdand.fx(r,a,tsim-1)   = lambdand.l(r,a,tsim-1) ;
   lambdava.fx(r,a,tsim-1)   = lambdava.l(r,a,tsim-1) ;
   lambdaio.fx(r,i,a,tsim-1) = lambdaio.l(r,i,a,tsim-1) ;
   lambdaf.fx(r,fp,a,tsim-1) = lambdaf.l(r,fp,a,tsim-1) ;

   pf.fx(r,fp,a,tsim-1)      = pf.l(r,fp,a,tsim-1) ;
   xf.fx(r,fp,a,tsim-1)      = xf.l(r,fp,a,tsim-1) ;

   pa.fx(r,i,aa,tsim-1)      = pa.l(r,i,aa,tsim-1) ;
   xa.fx(r,i,aa,tsim-1)      = xa.l(r,i,aa,tsim-1) ;
   pe.fx(r,i,rp,tsim-1)      = pe.l(r,i,rp,tsim-1) ;
   pefob.fx(r,i,rp,tsim-1)   = pefob.l(r,i,rp,tsim-1) ;
   pmcif.fx(r,i,rp,tsim-1)   = pmcif.l(r,i,rp,tsim-1) ;
   pm.fx(r,i,rp,tsim-1)      = pm.l(r,i,rp,tsim-1) ;
   xw.fx(r,i,rp,tsim-1)      = xw.l(r,i,rp,tsim-1) ;
   ptmg.fx(m,tsim-1)         = ptmg.l(m,tsim-1) ;

   psave.fx(r,tsim-1)        = psave.l(r,tsim-1) ;
   pi.fx(r,tsim-1)           = pi.l(r,tsim-1) ;

   uh.fx(r,h,tsim-1)         = uh.l(r,h,tsim-1) ;

   pabs.fx(r,tsim-1)         = pabs.l(r,tsim-1) ;
   pmuv.fx(tsim-1)           = pmuv.l(tsim-1) ;
   pfact.fx(r,tsim-1)        = pfact.l(r,tsim-1) ;
   pwfact.fx(tsim-1)         = pwfact.l(tsim-1) ;
   gdpmp.fx(r,tsim-1)        = gdpmp.l(r,tsim-1) ;
   rgdpmp.fx(r,tsim-1)       = rgdpmp.l(r,tsim-1) ;
   pgdpmp.fx(r,tsim-1)       = pgdpmp.l(r,tsim-1) ;
) ;

*  Define the simulation specific shock

* === PURE gtap tm_pct shock: +10% on the POWER of all bilateral
* === import tariffs.  imptx_new = (1+imptx)*1.1 - 1, gated by xwFlag
* === (so the diagonal r==r / no-flow routes, where xwFlag=0, are skipped).
* === Matches the Python gate's _apply_imptx_shock(factor=0.1).
   if(sameas(tsim,'shock'),
      imptx.fx(r,i,rp,tsim)$xwFlag(r,i,rp) =
         (1 + imptx.l(r,i,rp,tsim)) * 1.1 - 1 ;
   ) ;

   options limrow = 3, limcol = 3, solprint = off, iterlim = 50000 ;

* === PATH stabilisation for the ifSUB=1 shock restart (see
* === _inject_path_opt_for_shock): default PATH diverges from the check
* === warm-start on the large datasets; a proximal perturbation + higher
* === iteration limits let it converge.  Faithful (solver tuning, no eq change).
$onecho > path.opt
convergence_tolerance 1e-9
major_iteration_limit 2000
minor_iteration_limit 100000
cumulative_iteration_limit 1000000
proximal_perturbation 1e-2
crash_method pnewton
crash_perturb yes
nms_initial_reference_factor 2
gradient_step_limit 20
restart_limit 5
lemke_start automatic
time_limit 3600
$offecho
   gtap.optfile = 1 ;


   if(years(tsim) gt firstYear,

      $$iftheni.solve "%simType%" == "CompStat"
if(ifMCP,
   solve gtap using mcp ;
else
   solve gtap using nlp maximizing walras ;
) ;

*  Update the substituted variables

if(ifSUB,
   pp.l(r,a,i,t)      = p.l(r,a,i,t)*(1 + prdtx.l(r,a,i,t)) ;
   pdp.l(r,i,aa,t)    = pd.l(r,i,t)*(1 + dintx.l(r,i,aa,t)) ;
   pmp.l(r,i,aa,t)    = pmt.l(r,i,t)*(1 + mintx.l(r,i,aa,t)) ;
   xwmg.l(r,i,rp,t)   = tmarg.l(r,i,rp,t)*xw.l(r,i,rp,t) ;
   xmgm.l(m,r,i,rp,t) = amgm(m,r,i,rp)*xwmg.l(r,i,rp,t)/lambdamg.l(m,r,i,rp,t) ;
   pwmg.l(r,i,rp,t)   = sum(m, amgm(m,r,i,rp)*ptmg.l(m,t)/lambdamg.l(m,r,i,rp,t)) ;
   pefob.l(r,i,rp,t)  = (1 + exptx.l(r,i,rp,t) + etax.l(r,i,t))*pe.l(r,i,rp,t) ;
   pmcif.l(r,i,rp,t)  = pefob.l(r,i,rp,t) + pwmg.l(r,i,rp,t)*tmarg.l(r,i,rp,t) ;
   pm.l(r,i,rp,t)     = (1 + imptx.l(r,i,rp,t) + mtax.l(rp,i,t))*pmcif.l(r,i,rp,t)/chipm(r,i,rp) ;
   pfa.l(r,fp,a,t)    = pf.l(r,fp,a,t)*(1 + fctts.l(r,fp,a,t) + fcttx.l(r,fp,a,t)) ;
   pfy.l(r,fp,a,t)    = pf.l(r,fp,a,t)*(1 - kappaf.l(r,fp,a,t)) ;
) ;


      $$else.solve

         $$ifthen.calStatus %ifCal% == 1
if(ifMCP,
   solve dynCal using mcp ;
else
   solve dynCal using nlp maximizing walras ;
) ;

*  Update the substituted variables

if(ifSUB,
   pp.l(r,a,i,t)      = p.l(r,a,i,t)*(1 + prdtx.l(r,a,i,t)) ;
   pdp.l(r,i,aa,t)    = pd.l(r,i,t)*(1 + dintx.l(r,i,aa,t)) ;
   pmp.l(r,i,aa,t)    = pmt.l(r,i,t)*(1 + mintx.l(r,i,aa,t)) ;
   xwmg.l(r,i,rp,t)   = tmarg.l(r,i,rp,t)*xw.l(r,i,rp,t) ;
   xmgm.l(m,r,i,rp,t) = amgm(m,r,i,rp)*xwmg.l(r,i,rp,t)/lambdamg.l(m,r,i,rp,t) ;
   pwmg.l(r,i,rp,t)   = sum(m, amgm(m,r,i,rp)*ptmg.l(m,t)/lambdamg.l(m,r,i,rp,t)) ;
   pefob.l(r,i,rp,t)  = (1 + exptx.l(r,i,rp,t) + etax.l(r,i,t))*pe.l(r,i,rp,t) ;
   pmcif.l(r,i,rp,t)  = pefob.l(r,i,rp,t) + pwmg.l(r,i,rp,t)*tmarg.l(r,i,rp,t) ;
   pm.l(r,i,rp,t)     = (1 + imptx.l(r,i,rp,t) + mtax.l(rp,i,t))*pmcif.l(r,i,rp,t)/chipm(r,i,rp) ;
   pfa.l(r,fp,a,t)    = pf.l(r,fp,a,t)*(1 + fctts.l(r,fp,a,t) + fcttx.l(r,fp,a,t)) ;
   pfy.l(r,fp,a,t)    = pf.l(r,fp,a,t)*(1 - kappaf.l(r,fp,a,t)) ;
) ;


         $$else.calStatus
if(ifMCP,
   solve dynGTAP using mcp ;
else
   solve dynGTAP using nlp maximizing walras ;
) ;

*  Update the substituted variables

if(ifSUB,
   pp.l(r,a,i,t)      = p.l(r,a,i,t)*(1 + prdtx.l(r,a,i,t)) ;
   pdp.l(r,i,aa,t)    = pd.l(r,i,t)*(1 + dintx.l(r,i,aa,t)) ;
   pmp.l(r,i,aa,t)    = pmt.l(r,i,t)*(1 + mintx.l(r,i,aa,t)) ;
   xwmg.l(r,i,rp,t)   = tmarg.l(r,i,rp,t)*xw.l(r,i,rp,t) ;
   xmgm.l(m,r,i,rp,t) = amgm(m,r,i,rp)*xwmg.l(r,i,rp,t)/lambdamg.l(m,r,i,rp,t) ;
   pwmg.l(r,i,rp,t)   = sum(m, amgm(m,r,i,rp)*ptmg.l(m,t)/lambdamg.l(m,r,i,rp,t)) ;
   pefob.l(r,i,rp,t)  = (1 + exptx.l(r,i,rp,t) + etax.l(r,i,t))*pe.l(r,i,rp,t) ;
   pmcif.l(r,i,rp,t)  = pefob.l(r,i,rp,t) + pwmg.l(r,i,rp,t)*tmarg.l(r,i,rp,t) ;
   pm.l(r,i,rp,t)     = (1 + imptx.l(r,i,rp,t) + mtax.l(rp,i,t))*pmcif.l(r,i,rp,t)/chipm(r,i,rp) ;
   pfa.l(r,fp,a,t)    = pf.l(r,fp,a,t)*(1 + fctts.l(r,fp,a,t) + fcttx.l(r,fp,a,t)) ;
   pfy.l(r,fp,a,t)    = pf.l(r,fp,a,t)*(1 - kappaf.l(r,fp,a,t)) ;
) ;


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
* --------------------------------------------------------------------------------------------------
*
*  Calculate post-simulation results
*
* --------------------------------------------------------------------------------------------------

*  Recalc pdp/pmp post-solve for ALL cells (including alpha=0 ones where the
*  pdpeq/pmpeq are gated off and `iterloop.gms` would otherwise leave them at
*  the cal.gms init of 1.0). This matches the 9x10 reference GDX convention
*  and the NEOS regen -- both have pdp = pd*(1+dintx), pmp = pmt*(1+mintx) for
*  every cell, not just the ones where alpha != 0.
pdp.l(r,i,aa,t) = pd.l(r,i,t)*(1 + dintx.l(r,i,aa,t)) ;
pmp.l(r,i,aa,t) = pmt.l(r,i,t)*(1 + mintx.l(r,i,aa,t)) ;

*  SAM calculation

*  Domestic production

sam(r,i,aa,t)      = pd.l(r,i,t)*xd.l(r,i,aa,t)/xScale(r,aa)
                   + pmt.l(r,i,t)*xm.l(r,i,aa,t)/xScale(r,aa) ;
sam(r,"itax",aa,t) = sum(i, dintx.l(r,i,aa,t)*pd.l(r,i,t)*xd.l(r,i,aa,t)/xScale(r,aa)
                   +   mintx.l(r,i,aa,t)*pmt.l(r,i,t)*xm.l(r,i,aa,t)/xScale(r,aa)) ;
sam(r,fp,a,t)      = pf.l(r,fp,a,t)*xf.l(r,fp,a,t)/xScale(r,a) ;
sam(r,"vtax",a,t)  = sum(fp, fcttx.l(r,fp,a,t)*pf.l(r,fp,a,t)*xf.l(r,fp,a,t)/xScale(r,a)) ;
sam(r,"vsub",a,t)  = sum(fp, fctts.l(r,fp,a,t)*pf.l(r,fp,a,t)*xf.l(r,fp,a,t)/xScale(r,a)) ;

sam(r,a,i,t)$xFlag(r,a,i) = p.l(r,a,i,t)*x.l(r,a,i,t) ;
sam(r,"ptax",i,t)  = sum(a, prdtx.l(r,a,i,t)*p.l(r,a,i,t)*x.l(r,a,i,t)) ;

*  Income distribution
sam(r,h,fp,t)        = sum(a, (1-kappaf.l(r,fp,a,t))*pf.l(r,fp,a,t)*xf.l(r,fp,a,t)/xScale(r,a)) ;
sam(r,"dtax",fp,t)   = sum(a, kappaf.l(r,fp,a,t)*pf.l(r,fp,a,t)*xf.l(r,fp,a,t)/xScale(r,a)) ;
sam(r,gov,"vsub",t)  = sum(a, sam(r,"vsub",a,t)) ;
sam(r,gov,"vtax",t)  = sum(a, sam(r,"vtax",a,t)) ;
sam(r,"etax",i,t)    = sum(rp, exptx.l(r,i,rp,t)*pe.l(r,i,rp,t)*xw.l(r,i,rp,t)) ;
sam(r,"mtax",i,t)    = sum(rp, imptx.l(rp,i,r,t)*pmcif.l(rp,i,r,t)*xw.l(rp,i,r,t)) ;
sam(r,gov,"etax",t)  = sum(j, sam(r,"etax",j,t)) ;
sam(r,gov,"mtax",t)  = sum(j, sam(r,"mtax",j,t)) ;
sam(r,gov,"ptax",t)  = sum(i, sam(r,"ptax",i,t)) ;
sam(r,gov,"itax",t)  = sum(aa, sam(r,"itax",aa,t)) ;
sam(r,gov,"dtax",t)  = sum(js, sam(r,"dtax",js,t)) ;
sam(r,inv,h,t)       = rsav.l(r,t) ;
sam(r,"depry",h,t)   = pi.l(r,t)*kstock.l(r,t)*depr(r,t) ;
sam(r,inv,"depry",t) = pi.l(r,t)*kstock.l(r,t)*depr(r,t) ;
sam(r,inv,"bop",t)   = savf.l(r,t) ;

sam(r,gov,h,t)       = yg.l(r,t) - ytaxTot.l(r,t) ;

*  Trade

sam(r,rp,i,t)     = pmcif.l(rp,i,r,t)*xw.l(rp,i,r,t) ;
sam(r,"bop",rp,t) = sum(i, sam(r,rp,i,t)) ;

sam(r,i,rp,t)     = pefob.l(r,i,rp,t)*xw.l(r,i,rp,t) ;
sam(r,rp,"bop",t) = sum(i, sam(r,i,rp,t)) ;

loop(tmg,
   sam(r,tmg,"bop",t) = sum(i, sam(r,i,tmg,t)) ;
) ;

*  Convert back to scale
sam(r,is,js,t) = sam(r,is,js,t) / inScale ;

*  Calculate other post-simulation statistics

*  Output the results
if(ifCSV,
   put csv ;

   loop((r,is,js,t)$(sam(r,is,js,t) ne 0),
      put "sam", r.tl, is.tl, js.tl, years(t):4:0, sam(r,is,js,t) / ;
   ) ;

*  Production variables

   loop((r,a,t)$xpFlag(r,a),
      put "xp", r.tl, a.tl, "", years(t):4:0, ((xp.l(r,a,t)/xScale(r,a))/inScale) / ;
      put "nd", r.tl, a.tl, "", years(t):4:0, ((nd.l(r,a,t)/xScale(r,a))/inScale) / ;
      put "va", r.tl, a.tl, "", years(t):4:0, ((va.l(r,a,t)/xScale(r,a))/inScale) / ;
      put "px", r.tl, a.tl, "", years(t):4:0, px.l(r,a,t) / ;
   ) ;

*  Supply variables

   loop((r,i,t)$xsFlag(r,i),
      put "xs",  r.tl, i.tl, "", years(t):4:0, (xs.l(r,i,t)/inScale) / ;
      put "xds", r.tl, i.tl, "", years(t):4:0, (xds.l(r,i,t)/inScale) / ;
      put "xet", r.tl, i.tl, "", years(t):4:0, (xet.l(r,i,t)/inScale) / ;
      put "ps",  r.tl, i.tl, "", years(t):4:0, ps.l(r,i,t) / ;
      put "pd",  r.tl, i.tl, "", years(t):4:0, pd.l(r,i,t) / ;
      put "pet", r.tl, i.tl, "", years(t):4:0, pet.l(r,i,t) / ;
      loop(aa$alphad(r,i,aa,t),
         put "xd",  r.tl, i.tl, aa.tl, years(t):4:0, ((xd.l(r,i,aa,t)/xScale(r,aa))/inScale) / ;
         put "pdp", r.tl, i.tl, aa.tl, years(t):4:0, (pdp.l(r,i,aa,t)) / ;
         put "dintx", r.tl, i.tl, aa.tl, years(t):4:0, (dintx.l(r,i,aa,t)) / ;
      ) ;
   ) ;

   loop((r,a,i,t)$xFlag(r,a,i),
      put "prdtx", r.tl,a.tl, i.tl, years(t):4:0, (100*prdtx.l(r,a,i,t)) / ;
      put "p", r.tl,a.tl, i.tl, years(t):4:0, (p.l(r,a,i,t)) / ;
      put "x", r.tl,a.tl, i.tl, years(t):4:0, (x.l(r,a,i,t)/inScale) / ;
   ) ;

*  Factor variables

   loop((r,fp,a,t)$xfFlag(r,fp,a),
      put "xf",    r.tl, a.tl, fp.tl, years(t):4:0, ((xf.l(r,fp,a,t)/xScale(r,a))/inScale) / ;
      put "pf",    r.tl, a.tl, fp.tl, years(t):4:0, (pf.l(r,fp,a,t)) / ;
      put "pfy",   r.tl, a.tl, fp.tl, years(t):4:0, (pfy.l(r,fp,a,t)) / ;
      put "fcttx", r.tl, a.tl, fp.tl, years(t):4:0, (100*fcttx.l(r,fp,a,t)) / ;
      put "fctts", r.tl, a.tl, fp.tl, years(t):4:0, (100*fctts.l(r,fp,a,t)) / ;
   ) ;

*  Income

   loop((r,t),
      loop(gy,
         put "ytax", r.tl, "", gy.tl, years(t):4:0, (ytax.l(r,gy,t)/inScale) / ;
      ) ;
      put "ytax",    r.tl, "", "IND", years(t):4:0, (ytaxInd.l(r,t)/inScale) / ;
      put "ytax",    r.tl, "", "TOT", years(t):4:0, (ytaxTot.l(r,t)/inScale) / ;
      put "factY",   r.tl, "", "", years(t):4:0, (factY.l(r,t)/inScale) / ;
      put "deprY",   r.tl, "", "", years(t):4:0, ((fdepr(r,t)*pi.l(r,t)*kstock.l(r,t))/inScale) / ;
      put "regY",    r.tl, "", "", years(t):4:0, (regY.l(r,t)/inScale) / ;
      put "yg",      r.tl, "", "", years(t):4:0, (yg.l(r,t)/inScale) / ;
      put "yc",      r.tl, "", "", years(t):4:0, (yc.l(r,t)/inScale) / ;
      put "rsav",    r.tl, "", "", years(t):4:0, (rsav.l(r,t)/inScale) / ;
      put "yi",      r.tl, "", "", years(t):4:0, (yi.l(r,t)/inScale) / ;
      put "savf",    r.tl, "", "", years(t):4:0, (savf.l(r,t)/inScale) / ;
      put "uh",      r.tl, "", "", years(t):4:0, (sum(h, uh.l(r,h,t))) / ;
      put "ug",      r.tl, "", "", years(t):4:0, (ug.l(r,t)) / ;
      put "us",      r.tl, "", "", years(t):4:0, (us.l(r,t)) / ;
      put "u",       r.tl, "", "", years(t):4:0, (u.l(r,t)) / ;
      put "pop",     r.tl, "", "", years(t):4:0, (pop.l(r,t)) / ;
      put "pabs",    r.tl, "", "", years(t):4:0, (pabs.l(r,t)) / ;
      put "rgdpmp",  r.tl, "", "", years(t):4:0, (rgdpmp.l(r,t)/inscale) / ;
      put "gdpmp",   r.tl, "", "", years(t):4:0, (gdpmp.l(r,t)/inscale) / ;
      put "gl",      r.tl, "", "", years(t):4:0, (100*gl.l(r,t)) / ;
   ) ;

*  Trade
   loop((r,i,rp,t),
      put "xw",   r.tl, i.tl, rp.tl, years(t):4:0, (xw.l(r,i,rp,t)/inScale) / ;
      put "pe",   r.tl, i.tl, rp.tl, years(t):4:0, (pe.l(r,i,rp,t)) / ;
      put "xwmg", r.tl, i.tl, rp.tl, years(t):4:0, (xwmg.l(r,i,rp,t)/inScale) / ;
      put "pwmg", r.tl, i.tl, rp.tl, years(t):4:0, (pwmg.l(r,i,rp,t)) / ;
      loop(m, put "xmgm", r.tl, i.tl, rp.tl, years(t):4:0, (xmgm.l(m,r,i,rp,t)) / ; ) ;
*     put "amw",  r.tl, i.tl, rp.tl, years(t):4:0, (amw(r,i,rp,t)) / ;
*     put "gw",   r.tl, i.tl, rp.tl, years(t):4:0, (gw(r,i,rp,t)) / ;
   ) ;

   loop((r,i,aa,t),
      put "xa",     r.tl, i.tl, aa.tl, years(t):4:0, ((xa.l(r,i,aa,t)/xScale(r,aa))/inScale) / ;
      put "pa",     r.tl, i.tl, aa.tl, years(t):4:0, (pa.l(r,i,aa,t)) / ;
      put "alphaa", r.tl, i.tl, aa.tl, years(t):4:0, (alphaa(r,i,aa,t)) / ;
*     put "alphad", r.tl, i.tl, aa.tl, years(t):4:0, (alphad(r,i,aa,t)) / ;
*     put "alpham", r.tl, i.tl, aa.tl, years(t):4:0, (alpham(r,i,aa,t)) / ;
   ) ;

   loop((r,i,h,t),
      put "eh",     r.tl, i.tl, h.tl, years(t):4:0, (eh.l(r,i,t)) / ;
      put "bh",     r.tl, i.tl, h.tl, years(t):4:0, (bh.l(r,i,t)) / ;
   ) ;

   loop((m,t),
      put "xtmg", "GBL", m.tl, "", years(t):4:0, (xtmg.l(m,t)/1e-6) / ;
      put "ptmg", "GBL", m.tl, "", years(t):4:0, (ptmg.l(m,t)) / ;
   ) ;

*  Investment variables

   loop((r,t),
      put "arent",  r.tl, "", "", years(t):4:0, (100*arent.l(r,t)) / ;
      put "rorc",   r.tl, "", "", years(t):4:0, (100*rorc.l(r,t)) / ;
      put "rore",   r.tl, "", "", years(t):4:0, (100*rore.l(r,t)) / ;
      put "risk",   r.tl, "", "", years(t):4:0, (risk(r,t)) / ;
      put "kapend", r.tl, "", "", years(t):4:0, (kapend.l(r,t)/inScale) / ;
      put "kstock", r.tl, "", "", years(t):4:0, (kstock.l(r,t)/inScale) / ;
      put "xi",     r.tl, "", "", years(t):4:0, (xi.l(r,t)/inScale) / ;
      loop(cap, put "pcap",   r.tl, "", "", years(t):4:0, (pft.l(r,cap,t)) / ; ) ;
   ) ;

   loop((r,fp,t)$xft.l(r,fp,t),
      put "xft",   r.tl, fp.tl, "", years(t):4:0, (xft.l(r,fp,t)/inscale) / ;
      put "pft",   r.tl, fp.tl, "", years(t):4:0, (pft.l(r,fp,t)) / ;
   ) ;

   loop(t,
      put "rorg",   "GBL", "", "", years(t):4:0, (100*rorg.l(t)) / ;
   ) ;

   $$iftheni "%simType%" == "RcvDyn"
      if(1,
         loop((r,tranche,t),
            put "PopScen", r.tl, tranche.tl, "", years(t):4:0,
               (popScen("%POPSCEN%", r, tranche, t)) / ;
         ) ;
         loop((r,t),
            put "GDPScen", r.tl, "", "", years(t):4:0,
               gdpScen("%SSPMOD%","%SSPSCEN%","GDP",r,t) / ;
         ) ;
      ) ;
   $$endif

   if(1,
      loop(r,
         put "save",  r.tl, "", "", (2011):4:0, (save(r)) / ;
         put "vdep",  r.tl, "", "", (2011):4:0, (vdep(r)) / ;
         loop(fp,
            loop(a0,
               put "evfp", r.tl, fp.tl, a0.tl, (2011):4:0, (evfp(fp,a0,r)) / ;
               put "evfb",  r.tl, fp.tl, a0.tl, (2011):4:0, (evfb(fp,a0,r)) / ;
               put "evos", r.tl, fp.tl, a0.tl, (2011):4:0, (evos(fp,a0,r)) / ;
               put "fbep", r.tl, fp.tl, a0.tl, (2011):4:0, (fbep(fp,a0,r)) / ;
               put "ftrv", r.tl, fp.tl, a0.tl, (2011):4:0, (ftrv(fp,a0,r)) / ;
            ) ;
         ) ;
         loop((i0,a0),
            put "vdfp", r.tl, i0.tl, a0.tl, (2011):4:0, (vdfp(i0,a0,r)) / ;
            put "vdfb", r.tl, i0.tl, a0.tl, (2011):4:0, (vdfb(i0,a0,r)) / ;
            put "vmfp", r.tl, i0.tl, a0.tl, (2011):4:0, (vmfp(i0,a0,r)) / ;
            put "vmfb", r.tl, i0.tl, a0.tl, (2011):4:0, (vmfb(i0,a0,r)) / ;
         ) ;
         loop(i0,
            put "vdpp", r.tl, i0.tl, "", (2011):4:0, (vdpp(i0,r)) / ;
            put "vmpp", r.tl, i0.tl, "", (2011):4:0, (vmpp(i0,r)) / ;
            put "vdpb", r.tl, i0.tl, "", (2011):4:0, (vdpb(i0,r)) / ;
            put "vmpb", r.tl, i0.tl, "", (2011):4:0, (vmpb(i0,r)) / ;
            put "vdgp", r.tl, i0.tl, "", (2011):4:0, (vdgp(i0,r)) / ;
            put "vmgp", r.tl, i0.tl, "", (2011):4:0, (vmgp(i0,r)) / ;
            put "vdgb", r.tl, i0.tl, "", (2011):4:0, (vdgb(i0,r)) / ;
            put "vmgb", r.tl, i0.tl, "", (2011):4:0, (vmgb(i0,r)) / ;
            put "vdip", r.tl, i0.tl, "", (2011):4:0, (vdip(i0,r)) / ;
            put "vmip", r.tl, i0.tl, "", (2011):4:0, (vmip(i0,r)) / ;
            put "vdib", r.tl, i0.tl, "", (2011):4:0, (vdib(i0,r)) / ;
            put "vmib", r.tl, i0.tl, "", (2011):4:0, (vmib(i0,r)) / ;
            put "vst",  r.tl, i0.tl, "", (2011):4:0, (vst(i0,r)) / ;
         ) ;
         loop((i0,rp),
            put "vxsb",  r.tl, i0.tl, rp.tl, (2011):4:0, (vxsb(i0,r,rp)) / ;
            put "vfob",  r.tl, i0.tl, rp.tl, (2011):4:0, (vfob(i0,r,rp)) / ;
            put "vmsb",  r.tl, i0.tl, rp.tl, (2011):4:0, (vmsb(i0,r,rp)) / ;
            put "vcif",  r.tl, i0.tl, rp.tl, (2011):4:0, (vcif(i0,r,rp)) / ;
         ) ;
      ) ;
   ) ;
) ;

execute_unload "out.gdx" ;
