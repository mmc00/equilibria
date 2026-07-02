* -------------------------------------------------------------------------
*
*  Standard model diagnostics
*
*  Model preamble -- user options
*
* -------------------------------------------------------------------------

$setGlobal simType   CompStat
$setGlobal simName   COMP
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

Set acts 'Set ACTS  Activities' /
   a_Rice,
   a_Crops,
   a_Livestock,
   a_FoodProc,
   a_Energy,
   a_Textiles,
   a_Chem,
   a_Manuf,
   a_ForestFish,
   a_Svces
/;

Set comm 'Set COMM  Commodities' /
   c_Rice,
   c_Crops,
   c_Livestock,
   c_FoodProc,
   c_Energy,
   c_Textiles,
   c_Chem,
   c_Manuf,
   c_ForestFish,
   c_Svces
/;

Set reg 'Set REG  Regions' /
   USA,
   EU_28,
   CHN,
   JPN,
   IND,
   SSA,
   ROW
/;

Set endw 'Set ENDW  Endowments' /
   Land,
   UnSkLab,
   SkLab,
   Capital,
   NatRes
/;

Set marg 'Set MARG  Margin commodities' /
   c_Svces
/;

Set endwm 'Set ENDWM  Mobile endowments' /
   UnSkLab,
   SkLab,
   Capital
/;

Set endwf 'Set ENDWF  Sector-specific endowment(s)' /
   NatRes
/;

Set endws 'Set ENDWS  Sluggish endowment(s)' /
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
* pop0 data (7 cells)
pop0('USA') = 325.1471252 ;
pop0('EU_28') = 513.8722534 ;
pop0('CHN') = 1386.39502 ;
pop0('JPN') = 126.7857971 ;
pop0('IND') = 1338.658813 ;
pop0('SSA') = 1051.372681 ;
pop0('ROW') = 2771.6521 ;


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

* vdfb data (700 cells)
vdfb('c_Rice','a_Rice','USA') = 1903.839111 ;
vdfb('c_Rice','a_Rice','EU_28') = 1028.697266 ;
vdfb('c_Rice','a_Rice','CHN') = 53764.62891 ;
vdfb('c_Rice','a_Rice','JPN') = 17034.9375 ;
vdfb('c_Rice','a_Rice','IND') = 12666.06348 ;
vdfb('c_Rice','a_Rice','SSA') = 3749.080811 ;
vdfb('c_Rice','a_Rice','ROW') = 133730.1406 ;
vdfb('c_Rice','a_Crops','USA') = 4.327552319 ;
vdfb('c_Rice','a_Crops','EU_28') = 178.5584564 ;
vdfb('c_Rice','a_Crops','CHN') = 309.7374268 ;
vdfb('c_Rice','a_Crops','JPN') = 204.5247192 ;
vdfb('c_Rice','a_Crops','IND') = 531.4165649 ;
vdfb('c_Rice','a_Crops','SSA') = 39.30194855 ;
vdfb('c_Rice','a_Crops','ROW') = 729.6167603 ;
vdfb('c_Rice','a_Livestock','USA') = 13.98706532 ;
vdfb('c_Rice','a_Livestock','EU_28') = 62.8060112 ;
vdfb('c_Rice','a_Livestock','CHN') = 8482.760742 ;
vdfb('c_Rice','a_Livestock','JPN') = 519.8522949 ;
vdfb('c_Rice','a_Livestock','IND') = 1416.740601 ;
vdfb('c_Rice','a_Livestock','SSA') = 169.396698 ;
vdfb('c_Rice','a_Livestock','ROW') = 5366.98291 ;
vdfb('c_Rice','a_FoodProc','USA') = 1526.539795 ;
vdfb('c_Rice','a_FoodProc','EU_28') = 554.2144775 ;
vdfb('c_Rice','a_FoodProc','CHN') = 32843.01953 ;
vdfb('c_Rice','a_FoodProc','JPN') = 3923.408447 ;
vdfb('c_Rice','a_FoodProc','IND') = 39301.375 ;
vdfb('c_Rice','a_FoodProc','SSA') = 1240.621582 ;
vdfb('c_Rice','a_FoodProc','ROW') = 25191.3457 ;
vdfb('c_Rice','a_Energy','USA') = 0.6881924272 ;
vdfb('c_Rice','a_Energy','EU_28') = 0.580580771 ;
vdfb('c_Rice','a_Energy','CHN') = 17.97001839 ;
vdfb('c_Rice','a_Energy','JPN') = 0.1590573937 ;
vdfb('c_Rice','a_Energy','IND') = 35.91304398 ;
vdfb('c_Rice','a_Energy','SSA') = 4.027539253 ;
vdfb('c_Rice','a_Energy','ROW') = 1258.911133 ;
vdfb('c_Rice','a_Textiles','USA') = 0.4197565019 ;
vdfb('c_Rice','a_Textiles','EU_28') = 2.213948727 ;
vdfb('c_Rice','a_Textiles','CHN') = 1781.418091 ;
vdfb('c_Rice','a_Textiles','JPN') = 0.3588923216 ;
vdfb('c_Rice','a_Textiles','IND') = 0.6348187327 ;
vdfb('c_Rice','a_Textiles','SSA') = 7.869688034 ;
vdfb('c_Rice','a_Textiles','ROW') = 306.496521 ;
vdfb('c_Rice','a_Chem','USA') = 115.4887619 ;
vdfb('c_Rice','a_Chem','EU_28') = 10.17549419 ;
vdfb('c_Rice','a_Chem','CHN') = 7639.360352 ;
vdfb('c_Rice','a_Chem','JPN') = 29.3732338 ;
vdfb('c_Rice','a_Chem','IND') = 183.6500244 ;
vdfb('c_Rice','a_Chem','SSA') = 259.4682617 ;
vdfb('c_Rice','a_Chem','ROW') = 793.322937 ;
vdfb('c_Rice','a_Manuf','USA') = 27.64919281 ;
vdfb('c_Rice','a_Manuf','EU_28') = 15.77128315 ;
vdfb('c_Rice','a_Manuf','CHN') = 4143.34668 ;
vdfb('c_Rice','a_Manuf','JPN') = 99.03513336 ;
vdfb('c_Rice','a_Manuf','IND') = 9.44229126 ;
vdfb('c_Rice','a_Manuf','SSA') = 102.6694107 ;
vdfb('c_Rice','a_Manuf','ROW') = 1045.0271 ;
vdfb('c_Rice','a_ForestFish','USA') = 6.835223675 ;
vdfb('c_Rice','a_ForestFish','EU_28') = 2.82822299 ;
vdfb('c_Rice','a_ForestFish','CHN') = 844.637207 ;
vdfb('c_Rice','a_ForestFish','JPN') = 35.73182678 ;
vdfb('c_Rice','a_ForestFish','IND') = 1.989108562 ;
vdfb('c_Rice','a_ForestFish','SSA') = 1.816149831 ;
vdfb('c_Rice','a_ForestFish','ROW') = 1014.824646 ;
vdfb('c_Rice','a_Svces','USA') = 696.9804688 ;
vdfb('c_Rice','a_Svces','EU_28') = 1027.826538 ;
vdfb('c_Rice','a_Svces','CHN') = 14904.76074 ;
vdfb('c_Rice','a_Svces','JPN') = 4049.369873 ;
vdfb('c_Rice','a_Svces','IND') = 8577.880859 ;
vdfb('c_Rice','a_Svces','SSA') = 1067.511108 ;
vdfb('c_Rice','a_Svces','ROW') = 17055.4668 ;
vdfb('c_Crops','a_Rice','USA') = 8.324521065 ;
vdfb('c_Crops','a_Rice','EU_28') = 11.48566151 ;
vdfb('c_Crops','a_Rice','CHN') = 1216.587524 ;
vdfb('c_Crops','a_Rice','JPN') = 7.131267548 ;
vdfb('c_Crops','a_Rice','IND') = 346.5015564 ;
vdfb('c_Crops','a_Rice','SSA') = 829.3806763 ;
vdfb('c_Crops','a_Rice','ROW') = 1427.107422 ;
vdfb('c_Crops','a_Crops','USA') = 7409.296875 ;
vdfb('c_Crops','a_Crops','EU_28') = 17355.67969 ;
vdfb('c_Crops','a_Crops','CHN') = 64051.45312 ;
vdfb('c_Crops','a_Crops','JPN') = 470.2381897 ;
vdfb('c_Crops','a_Crops','IND') = 3540.449951 ;
vdfb('c_Crops','a_Crops','SSA') = 30531.22852 ;
vdfb('c_Crops','a_Crops','ROW') = 99566.71094 ;
vdfb('c_Crops','a_Livestock','USA') = 7660.948242 ;
vdfb('c_Crops','a_Livestock','EU_28') = 11650.77441 ;
vdfb('c_Crops','a_Livestock','CHN') = 28124.53711 ;
vdfb('c_Crops','a_Livestock','JPN') = 1035.724609 ;
vdfb('c_Crops','a_Livestock','IND') = 8526.450195 ;
vdfb('c_Crops','a_Livestock','SSA') = 3009.150635 ;
vdfb('c_Crops','a_Livestock','ROW') = 36960.62891 ;
vdfb('c_Crops','a_FoodProc','USA') = 75026.24219 ;
vdfb('c_Crops','a_FoodProc','EU_28') = 47017.41016 ;
vdfb('c_Crops','a_FoodProc','CHN') = 268547.125 ;
vdfb('c_Crops','a_FoodProc','JPN') = 5480.990234 ;
vdfb('c_Crops','a_FoodProc','IND') = 36724.09375 ;
vdfb('c_Crops','a_FoodProc','SSA') = 22174.86328 ;
vdfb('c_Crops','a_FoodProc','ROW') = 220311.5938 ;
vdfb('c_Crops','a_Energy','USA') = 33.79624176 ;
vdfb('c_Crops','a_Energy','EU_28') = 79.61803436 ;
vdfb('c_Crops','a_Energy','CHN') = 110.8199768 ;
vdfb('c_Crops','a_Energy','JPN') = 5.274326324 ;
vdfb('c_Crops','a_Energy','IND') = 1515.285522 ;
vdfb('c_Crops','a_Energy','SSA') = 140.1399231 ;
vdfb('c_Crops','a_Energy','ROW') = 1847.400879 ;
vdfb('c_Crops','a_Textiles','USA') = 542.9299927 ;
vdfb('c_Crops','a_Textiles','EU_28') = 741.0822144 ;
vdfb('c_Crops','a_Textiles','CHN') = 48385.60547 ;
vdfb('c_Crops','a_Textiles','JPN') = 154.4386444 ;
vdfb('c_Crops','a_Textiles','IND') = 22126.38477 ;
vdfb('c_Crops','a_Textiles','SSA') = 1298.884521 ;
vdfb('c_Crops','a_Textiles','ROW') = 18938.70703 ;
vdfb('c_Crops','a_Chem','USA') = 13713.02344 ;
vdfb('c_Crops','a_Chem','EU_28') = 1118.502563 ;
vdfb('c_Crops','a_Chem','CHN') = 32907.79688 ;
vdfb('c_Crops','a_Chem','JPN') = 1345.27356 ;
vdfb('c_Crops','a_Chem','IND') = 3471.766846 ;
vdfb('c_Crops','a_Chem','SSA') = 491.4737244 ;
vdfb('c_Crops','a_Chem','ROW') = 13921.34473 ;
vdfb('c_Crops','a_Manuf','USA') = 381.7834778 ;
vdfb('c_Crops','a_Manuf','EU_28') = 227.5586243 ;
vdfb('c_Crops','a_Manuf','CHN') = 17724.4668 ;
vdfb('c_Crops','a_Manuf','JPN') = 51.29619598 ;
vdfb('c_Crops','a_Manuf','IND') = 103.2072601 ;
vdfb('c_Crops','a_Manuf','SSA') = 386.5133972 ;
vdfb('c_Crops','a_Manuf','ROW') = 1532.299438 ;
vdfb('c_Crops','a_ForestFish','USA') = 271.8774719 ;
vdfb('c_Crops','a_ForestFish','EU_28') = 150.2907104 ;
vdfb('c_Crops','a_ForestFish','CHN') = 1234.75708 ;
vdfb('c_Crops','a_ForestFish','JPN') = 1.849598169 ;
vdfb('c_Crops','a_ForestFish','IND') = 20.52301216 ;
vdfb('c_Crops','a_ForestFish','SSA') = 170.0428772 ;
vdfb('c_Crops','a_ForestFish','ROW') = 1247.687134 ;
vdfb('c_Crops','a_Svces','USA') = 6445.199707 ;
vdfb('c_Crops','a_Svces','EU_28') = 12885.9082 ;
vdfb('c_Crops','a_Svces','CHN') = 53384.20312 ;
vdfb('c_Crops','a_Svces','JPN') = 5101.680664 ;
vdfb('c_Crops','a_Svces','IND') = 22357.20117 ;
vdfb('c_Crops','a_Svces','SSA') = 4983.991211 ;
vdfb('c_Crops','a_Svces','ROW') = 41458.46484 ;
vdfb('c_Livestock','a_Rice','USA') = 136.3382416 ;
vdfb('c_Livestock','a_Rice','EU_28') = 20.56066895 ;
vdfb('c_Livestock','a_Rice','CHN') = 31.70809364 ;
vdfb('c_Livestock','a_Rice','JPN') = 122.8559036 ;
vdfb('c_Livestock','a_Rice','IND') = 1090.668701 ;
vdfb('c_Livestock','a_Rice','SSA') = 228.4325867 ;
vdfb('c_Livestock','a_Rice','ROW') = 1085.83606 ;
vdfb('c_Livestock','a_Crops','USA') = 275.1600342 ;
vdfb('c_Livestock','a_Crops','EU_28') = 1626.447266 ;
vdfb('c_Livestock','a_Crops','CHN') = 276.1022949 ;
vdfb('c_Livestock','a_Crops','JPN') = 203.4688416 ;
vdfb('c_Livestock','a_Crops','IND') = 4186.101562 ;
vdfb('c_Livestock','a_Crops','SSA') = 1107.548218 ;
vdfb('c_Livestock','a_Crops','ROW') = 13521.98242 ;
vdfb('c_Livestock','a_Livestock','USA') = 18427.77344 ;
vdfb('c_Livestock','a_Livestock','EU_28') = 5519.396973 ;
vdfb('c_Livestock','a_Livestock','CHN') = 25001.0293 ;
vdfb('c_Livestock','a_Livestock','JPN') = 2029.142822 ;
vdfb('c_Livestock','a_Livestock','IND') = 951.6547852 ;
vdfb('c_Livestock','a_Livestock','SSA') = 9425.314453 ;
vdfb('c_Livestock','a_Livestock','ROW') = 80862.70312 ;
vdfb('c_Livestock','a_FoodProc','USA') = 130922.8906 ;
vdfb('c_Livestock','a_FoodProc','EU_28') = 126761.1875 ;
vdfb('c_Livestock','a_FoodProc','CHN') = 144252.375 ;
vdfb('c_Livestock','a_FoodProc','JPN') = 20948.66797 ;
vdfb('c_Livestock','a_FoodProc','IND') = 17225.99023 ;
vdfb('c_Livestock','a_FoodProc','SSA') = 18031.08203 ;
vdfb('c_Livestock','a_FoodProc','ROW') = 269292.25 ;
vdfb('c_Livestock','a_Energy','USA') = 0.7532737851 ;
vdfb('c_Livestock','a_Energy','EU_28') = 65.95469666 ;
vdfb('c_Livestock','a_Energy','CHN') = 1.611268878 ;
vdfb('c_Livestock','a_Energy','JPN') = 1.0437783 ;
vdfb('c_Livestock','a_Energy','IND') = 138.8007812 ;
vdfb('c_Livestock','a_Energy','SSA') = 11.13362122 ;
vdfb('c_Livestock','a_Energy','ROW') = 214.349472 ;
vdfb('c_Livestock','a_Textiles','USA') = 166.914505 ;
vdfb('c_Livestock','a_Textiles','EU_28') = 434.2250671 ;
vdfb('c_Livestock','a_Textiles','CHN') = 22377.80859 ;
vdfb('c_Livestock','a_Textiles','JPN') = 16.10928917 ;
vdfb('c_Livestock','a_Textiles','IND') = 1262.135864 ;
vdfb('c_Livestock','a_Textiles','SSA') = 922.9078369 ;
vdfb('c_Livestock','a_Textiles','ROW') = 5029.391113 ;
vdfb('c_Livestock','a_Chem','USA') = 177.595993 ;
vdfb('c_Livestock','a_Chem','EU_28') = 1117.703125 ;
vdfb('c_Livestock','a_Chem','CHN') = 20778.58203 ;
vdfb('c_Livestock','a_Chem','JPN') = 83.4835434 ;
vdfb('c_Livestock','a_Chem','IND') = 4190.614258 ;
vdfb('c_Livestock','a_Chem','SSA') = 86.8514328 ;
vdfb('c_Livestock','a_Chem','ROW') = 1755.262207 ;
vdfb('c_Livestock','a_Manuf','USA') = 16.95561981 ;
vdfb('c_Livestock','a_Manuf','EU_28') = 150.1412506 ;
vdfb('c_Livestock','a_Manuf','CHN') = 2863.556641 ;
vdfb('c_Livestock','a_Manuf','JPN') = 6.011878014 ;
vdfb('c_Livestock','a_Manuf','IND') = 65.09170532 ;
vdfb('c_Livestock','a_Manuf','SSA') = 58.34075928 ;
vdfb('c_Livestock','a_Manuf','ROW') = 891.1433105 ;
vdfb('c_Livestock','a_ForestFish','USA') = 390.1132202 ;
vdfb('c_Livestock','a_ForestFish','EU_28') = 116.5031357 ;
vdfb('c_Livestock','a_ForestFish','CHN') = 19.96658516 ;
vdfb('c_Livestock','a_ForestFish','JPN') = 3.30553031 ;
vdfb('c_Livestock','a_ForestFish','IND') = 540.022644 ;
vdfb('c_Livestock','a_ForestFish','SSA') = 953.8569946 ;
vdfb('c_Livestock','a_ForestFish','ROW') = 548.3045044 ;
vdfb('c_Livestock','a_Svces','USA') = 3673.939209 ;
vdfb('c_Livestock','a_Svces','EU_28') = 8497.371094 ;
vdfb('c_Livestock','a_Svces','CHN') = 10362.19922 ;
vdfb('c_Livestock','a_Svces','JPN') = 2726.756836 ;
vdfb('c_Livestock','a_Svces','IND') = 38347.48828 ;
vdfb('c_Livestock','a_Svces','SSA') = 4536.791016 ;
vdfb('c_Livestock','a_Svces','ROW') = 22470.84961 ;
vdfb('c_FoodProc','a_Rice','USA') = 63.38324738 ;
vdfb('c_FoodProc','a_Rice','EU_28') = 41.95152664 ;
vdfb('c_FoodProc','a_Rice','CHN') = 252.4662933 ;
vdfb('c_FoodProc','a_Rice','JPN') = 2.000651121 ;
vdfb('c_FoodProc','a_Rice','IND') = 0.5534915924 ;
vdfb('c_FoodProc','a_Rice','SSA') = 177.3412628 ;
vdfb('c_FoodProc','a_Rice','ROW') = 1354.67395 ;
vdfb('c_FoodProc','a_Crops','USA') = 1.088562369 ;
vdfb('c_FoodProc','a_Crops','EU_28') = 1828.326416 ;
vdfb('c_FoodProc','a_Crops','CHN') = 1379.841187 ;
vdfb('c_FoodProc','a_Crops','JPN') = 63.24759674 ;
vdfb('c_FoodProc','a_Crops','IND') = 1.050860763 ;
vdfb('c_FoodProc','a_Crops','SSA') = 1062.901733 ;
vdfb('c_FoodProc','a_Crops','ROW') = 4347.068848 ;
vdfb('c_FoodProc','a_Livestock','USA') = 34643.47266 ;
vdfb('c_FoodProc','a_Livestock','EU_28') = 38756.54297 ;
vdfb('c_FoodProc','a_Livestock','CHN') = 79583.59375 ;
vdfb('c_FoodProc','a_Livestock','JPN') = 10860.6875 ;
vdfb('c_FoodProc','a_Livestock','IND') = 2576.854248 ;
vdfb('c_FoodProc','a_Livestock','SSA') = 4847.678711 ;
vdfb('c_FoodProc','a_Livestock','ROW') = 73739.17188 ;
vdfb('c_FoodProc','a_FoodProc','USA') = 230133.625 ;
vdfb('c_FoodProc','a_FoodProc','EU_28') = 232287.8125 ;
vdfb('c_FoodProc','a_FoodProc','CHN') = 355625.7812 ;
vdfb('c_FoodProc','a_FoodProc','JPN') = 47243.80469 ;
vdfb('c_FoodProc','a_FoodProc','IND') = 9867.782227 ;
vdfb('c_FoodProc','a_FoodProc','SSA') = 35844.56641 ;
vdfb('c_FoodProc','a_FoodProc','ROW') = 277792.25 ;
vdfb('c_FoodProc','a_Energy','USA') = 127.457634 ;
vdfb('c_FoodProc','a_Energy','EU_28') = 561.6920166 ;
vdfb('c_FoodProc','a_Energy','CHN') = 5781.65625 ;
vdfb('c_FoodProc','a_Energy','JPN') = 3.661335468 ;
vdfb('c_FoodProc','a_Energy','IND') = 79.35069275 ;
vdfb('c_FoodProc','a_Energy','SSA') = 56.10435486 ;
vdfb('c_FoodProc','a_Energy','ROW') = 1553.337524 ;
vdfb('c_FoodProc','a_Textiles','USA') = 1367.703735 ;
vdfb('c_FoodProc','a_Textiles','EU_28') = 2864.071533 ;
vdfb('c_FoodProc','a_Textiles','CHN') = 28619.16406 ;
vdfb('c_FoodProc','a_Textiles','JPN') = 134.1049042 ;
vdfb('c_FoodProc','a_Textiles','IND') = 10.4698 ;
vdfb('c_FoodProc','a_Textiles','SSA') = 877.9793701 ;
vdfb('c_FoodProc','a_Textiles','ROW') = 6750.106934 ;
vdfb('c_FoodProc','a_Chem','USA') = 5254.161621 ;
vdfb('c_FoodProc','a_Chem','EU_28') = 13340.44043 ;
vdfb('c_FoodProc','a_Chem','CHN') = 43174.29297 ;
vdfb('c_FoodProc','a_Chem','JPN') = 1428.333252 ;
vdfb('c_FoodProc','a_Chem','IND') = 2572.860107 ;
vdfb('c_FoodProc','a_Chem','SSA') = 1242.134155 ;
vdfb('c_FoodProc','a_Chem','ROW') = 10969.84277 ;
vdfb('c_FoodProc','a_Manuf','USA') = 3033.822266 ;
vdfb('c_FoodProc','a_Manuf','EU_28') = 4059.606689 ;
vdfb('c_FoodProc','a_Manuf','CHN') = 37275.8125 ;
vdfb('c_FoodProc','a_Manuf','JPN') = 297.4060059 ;
vdfb('c_FoodProc','a_Manuf','IND') = 26.8659153 ;
vdfb('c_FoodProc','a_Manuf','SSA') = 900.8452148 ;
vdfb('c_FoodProc','a_Manuf','ROW') = 5481.606445 ;
vdfb('c_FoodProc','a_ForestFish','USA') = 799.2169189 ;
vdfb('c_FoodProc','a_ForestFish','EU_28') = 741.1043701 ;
vdfb('c_FoodProc','a_ForestFish','CHN') = 25245.83398 ;
vdfb('c_FoodProc','a_ForestFish','JPN') = 1060.003296 ;
vdfb('c_FoodProc','a_ForestFish','IND') = 31.12132835 ;
vdfb('c_FoodProc','a_ForestFish','SSA') = 206.1407471 ;
vdfb('c_FoodProc','a_ForestFish','ROW') = 15869.08105 ;
vdfb('c_FoodProc','a_Svces','USA') = 161293.375 ;
vdfb('c_FoodProc','a_Svces','EU_28') = 211565.2656 ;
vdfb('c_FoodProc','a_Svces','CHN') = 249720.8125 ;
vdfb('c_FoodProc','a_Svces','JPN') = 51805.91797 ;
vdfb('c_FoodProc','a_Svces','IND') = 8796.571289 ;
vdfb('c_FoodProc','a_Svces','SSA') = 23107.15039 ;
vdfb('c_FoodProc','a_Svces','ROW') = 297968.6875 ;
vdfb('c_Energy','a_Rice','USA') = 119.4647751 ;
vdfb('c_Energy','a_Rice','EU_28') = 144.9143829 ;
vdfb('c_Energy','a_Rice','CHN') = 2363.260254 ;
vdfb('c_Energy','a_Rice','JPN') = 326.808136 ;
vdfb('c_Energy','a_Rice','IND') = 22698.52344 ;
vdfb('c_Energy','a_Rice','SSA') = 73.03604889 ;
vdfb('c_Energy','a_Rice','ROW') = 4651.951172 ;
vdfb('c_Energy','a_Crops','USA') = 4812.601562 ;
vdfb('c_Energy','a_Crops','EU_28') = 7341.273438 ;
vdfb('c_Energy','a_Crops','CHN') = 11427.86816 ;
vdfb('c_Energy','a_Crops','JPN') = 846.0897827 ;
vdfb('c_Energy','a_Crops','IND') = 16776.35938 ;
vdfb('c_Energy','a_Crops','SSA') = 690.1558838 ;
vdfb('c_Energy','a_Crops','ROW') = 25312.42773 ;
vdfb('c_Energy','a_Livestock','USA') = 3620.705078 ;
vdfb('c_Energy','a_Livestock','EU_28') = 2993.308838 ;
vdfb('c_Energy','a_Livestock','CHN') = 1842.137939 ;
vdfb('c_Energy','a_Livestock','JPN') = 364.2643127 ;
vdfb('c_Energy','a_Livestock','IND') = 50.60794067 ;
vdfb('c_Energy','a_Livestock','SSA') = 939.7015991 ;
vdfb('c_Energy','a_Livestock','ROW') = 12988.87598 ;
vdfb('c_Energy','a_FoodProc','USA') = 12402.12598 ;
vdfb('c_Energy','a_FoodProc','EU_28') = 18078.80469 ;
vdfb('c_Energy','a_FoodProc','CHN') = 15816.91211 ;
vdfb('c_Energy','a_FoodProc','JPN') = 5346.212402 ;
vdfb('c_Energy','a_FoodProc','IND') = 4149.07959 ;
vdfb('c_Energy','a_FoodProc','SSA') = 1118.123291 ;
vdfb('c_Energy','a_FoodProc','ROW') = 34954.76172 ;
vdfb('c_Energy','a_Energy','USA') = 304586.375 ;
vdfb('c_Energy','a_Energy','EU_28') = 133683.5 ;
vdfb('c_Energy','a_Energy','CHN') = 381460.25 ;
vdfb('c_Energy','a_Energy','JPN') = 41107.19922 ;
vdfb('c_Energy','a_Energy','IND') = 101309.9219 ;
vdfb('c_Energy','a_Energy','SSA') = 27679.23438 ;
vdfb('c_Energy','a_Energy','ROW') = 811974.9375 ;
vdfb('c_Energy','a_Textiles','USA') = 2135.583252 ;
vdfb('c_Energy','a_Textiles','EU_28') = 3701.640381 ;
vdfb('c_Energy','a_Textiles','CHN') = 24261.56055 ;
vdfb('c_Energy','a_Textiles','JPN') = 1057.903687 ;
vdfb('c_Energy','a_Textiles','IND') = 2766.599854 ;
vdfb('c_Energy','a_Textiles','SSA') = 417.0197754 ;
vdfb('c_Energy','a_Textiles','ROW') = 15446.34375 ;
vdfb('c_Energy','a_Chem','USA') = 59964.21484 ;
vdfb('c_Energy','a_Chem','EU_28') = 57616.57812 ;
vdfb('c_Energy','a_Chem','CHN') = 132094.8281 ;
vdfb('c_Energy','a_Chem','JPN') = 33480.88672 ;
vdfb('c_Energy','a_Chem','IND') = 30890.44336 ;
vdfb('c_Energy','a_Chem','SSA') = 4033.392334 ;
vdfb('c_Energy','a_Chem','ROW') = 199770.5156 ;
vdfb('c_Energy','a_Manuf','USA') = 60836.53906 ;
vdfb('c_Energy','a_Manuf','EU_28') = 92154.01562 ;
vdfb('c_Energy','a_Manuf','CHN') = 296860.5625 ;
vdfb('c_Energy','a_Manuf','JPN') = 53051.19922 ;
vdfb('c_Energy','a_Manuf','IND') = 88381.04688 ;
vdfb('c_Energy','a_Manuf','SSA') = 21068.81445 ;
vdfb('c_Energy','a_Manuf','ROW') = 305322.6875 ;
vdfb('c_Energy','a_ForestFish','USA') = 924.746521 ;
vdfb('c_Energy','a_ForestFish','EU_28') = 2402.239014 ;
vdfb('c_Energy','a_ForestFish','CHN') = 3547.619141 ;
vdfb('c_Energy','a_ForestFish','JPN') = 1367.047241 ;
vdfb('c_Energy','a_ForestFish','IND') = 2568.160645 ;
vdfb('c_Energy','a_ForestFish','SSA') = 276.3552246 ;
vdfb('c_Energy','a_ForestFish','ROW') = 9053.212891 ;
vdfb('c_Energy','a_Svces','USA') = 330760.9062 ;
vdfb('c_Energy','a_Svces','EU_28') = 232326.625 ;
vdfb('c_Energy','a_Svces','CHN') = 211807.6094 ;
vdfb('c_Energy','a_Svces','JPN') = 101658.3359 ;
vdfb('c_Energy','a_Svces','IND') = 101846.8672 ;
vdfb('c_Energy','a_Svces','SSA') = 23511.98633 ;
vdfb('c_Energy','a_Svces','ROW') = 528292 ;
vdfb('c_Textiles','a_Rice','USA') = 26.71277809 ;
vdfb('c_Textiles','a_Rice','EU_28') = 16.43350792 ;
vdfb('c_Textiles','a_Rice','CHN') = 365.3483887 ;
vdfb('c_Textiles','a_Rice','JPN') = 32.80321884 ;
vdfb('c_Textiles','a_Rice','IND') = 3.920345783 ;
vdfb('c_Textiles','a_Rice','SSA') = 16.89995193 ;
vdfb('c_Textiles','a_Rice','ROW') = 392.274231 ;
vdfb('c_Textiles','a_Crops','USA') = 115.9589691 ;
vdfb('c_Textiles','a_Crops','EU_28') = 321.631134 ;
vdfb('c_Textiles','a_Crops','CHN') = 28.88716698 ;
vdfb('c_Textiles','a_Crops','JPN') = 47.27773666 ;
vdfb('c_Textiles','a_Crops','IND') = 10.99382973 ;
vdfb('c_Textiles','a_Crops','SSA') = 445.0658875 ;
vdfb('c_Textiles','a_Crops','ROW') = 1019.409912 ;
vdfb('c_Textiles','a_Livestock','USA') = 13.69990635 ;
vdfb('c_Textiles','a_Livestock','EU_28') = 157.3687897 ;
vdfb('c_Textiles','a_Livestock','CHN') = 5.785516262 ;
vdfb('c_Textiles','a_Livestock','JPN') = 10.06723595 ;
vdfb('c_Textiles','a_Livestock','IND') = 3.608666182 ;
vdfb('c_Textiles','a_Livestock','SSA') = 121.4479904 ;
vdfb('c_Textiles','a_Livestock','ROW') = 628.2756348 ;
vdfb('c_Textiles','a_FoodProc','USA') = 139.3927155 ;
vdfb('c_Textiles','a_FoodProc','EU_28') = 1670.043945 ;
vdfb('c_Textiles','a_FoodProc','CHN') = 6941.526855 ;
vdfb('c_Textiles','a_FoodProc','JPN') = 143.5123749 ;
vdfb('c_Textiles','a_FoodProc','IND') = 41.14092255 ;
vdfb('c_Textiles','a_FoodProc','SSA') = 212.9935913 ;
vdfb('c_Textiles','a_FoodProc','ROW') = 3357.301758 ;
vdfb('c_Textiles','a_Energy','USA') = 30.03678513 ;
vdfb('c_Textiles','a_Energy','EU_28') = 199.6544189 ;
vdfb('c_Textiles','a_Energy','CHN') = 2953.399414 ;
vdfb('c_Textiles','a_Energy','JPN') = 21.64062309 ;
vdfb('c_Textiles','a_Energy','IND') = 969.2052002 ;
vdfb('c_Textiles','a_Energy','SSA') = 86.9019928 ;
vdfb('c_Textiles','a_Energy','ROW') = 3015.260498 ;
vdfb('c_Textiles','a_Textiles','USA') = 11332.21875 ;
vdfb('c_Textiles','a_Textiles','EU_28') = 58988.94141 ;
vdfb('c_Textiles','a_Textiles','CHN') = 575629.3125 ;
vdfb('c_Textiles','a_Textiles','JPN') = 6408.469238 ;
vdfb('c_Textiles','a_Textiles','IND') = 23795.71289 ;
vdfb('c_Textiles','a_Textiles','SSA') = 4585.46582 ;
vdfb('c_Textiles','a_Textiles','ROW') = 154254.2656 ;
vdfb('c_Textiles','a_Chem','USA') = 2063.115967 ;
vdfb('c_Textiles','a_Chem','EU_28') = 5339.509277 ;
vdfb('c_Textiles','a_Chem','CHN') = 21557.24023 ;
vdfb('c_Textiles','a_Chem','JPN') = 578.4924927 ;
vdfb('c_Textiles','a_Chem','IND') = 2908.421875 ;
vdfb('c_Textiles','a_Chem','SSA') = 151.4994049 ;
vdfb('c_Textiles','a_Chem','ROW') = 5587.969238 ;
vdfb('c_Textiles','a_Manuf','USA') = 12354.08594 ;
vdfb('c_Textiles','a_Manuf','EU_28') = 16155.08008 ;
vdfb('c_Textiles','a_Manuf','CHN') = 67783.45312 ;
vdfb('c_Textiles','a_Manuf','JPN') = 2394.094727 ;
vdfb('c_Textiles','a_Manuf','IND') = 901.5078125 ;
vdfb('c_Textiles','a_Manuf','SSA') = 710.5241089 ;
vdfb('c_Textiles','a_Manuf','ROW') = 20941.35352 ;
vdfb('c_Textiles','a_ForestFish','USA') = 62.13933945 ;
vdfb('c_Textiles','a_ForestFish','EU_28') = 311.782959 ;
vdfb('c_Textiles','a_ForestFish','CHN') = 201.4907837 ;
vdfb('c_Textiles','a_ForestFish','JPN') = 152.957962 ;
vdfb('c_Textiles','a_ForestFish','IND') = 103.9823837 ;
vdfb('c_Textiles','a_ForestFish','SSA') = 211.0969696 ;
vdfb('c_Textiles','a_ForestFish','ROW') = 953.7183228 ;
vdfb('c_Textiles','a_Svces','USA') = 11241.68066 ;
vdfb('c_Textiles','a_Svces','EU_28') = 21642.76758 ;
vdfb('c_Textiles','a_Svces','CHN') = 93901.41406 ;
vdfb('c_Textiles','a_Svces','JPN') = 6631.835938 ;
vdfb('c_Textiles','a_Svces','IND') = 5068.603027 ;
vdfb('c_Textiles','a_Svces','SSA') = 2854.739746 ;
vdfb('c_Textiles','a_Svces','ROW') = 50776.69531 ;
vdfb('c_Chem','a_Rice','USA') = 369.0286255 ;
vdfb('c_Chem','a_Rice','EU_28') = 171.1269684 ;
vdfb('c_Chem','a_Rice','CHN') = 11441.20312 ;
vdfb('c_Chem','a_Rice','JPN') = 1436.678223 ;
vdfb('c_Chem','a_Rice','IND') = 545.0986938 ;
vdfb('c_Chem','a_Rice','SSA') = 87.22988892 ;
vdfb('c_Chem','a_Rice','ROW') = 5824.39209 ;
vdfb('c_Chem','a_Crops','USA') = 18991.59961 ;
vdfb('c_Chem','a_Crops','EU_28') = 8058.02002 ;
vdfb('c_Chem','a_Crops','CHN') = 73639.89062 ;
vdfb('c_Chem','a_Crops','JPN') = 3258.666748 ;
vdfb('c_Chem','a_Crops','IND') = 630.1951294 ;
vdfb('c_Chem','a_Crops','SSA') = 3270.406006 ;
vdfb('c_Chem','a_Crops','ROW') = 45151.42578 ;
vdfb('c_Chem','a_Livestock','USA') = 1836.103149 ;
vdfb('c_Chem','a_Livestock','EU_28') = 1060.088013 ;
vdfb('c_Chem','a_Livestock','CHN') = 3499.542725 ;
vdfb('c_Chem','a_Livestock','JPN') = 483.2485352 ;
vdfb('c_Chem','a_Livestock','IND') = 22.33856201 ;
vdfb('c_Chem','a_Livestock','SSA') = 253.0210266 ;
vdfb('c_Chem','a_Livestock','ROW') = 8361.758789 ;
vdfb('c_Chem','a_FoodProc','USA') = 32114.36719 ;
vdfb('c_Chem','a_FoodProc','EU_28') = 19245.23242 ;
vdfb('c_Chem','a_FoodProc','CHN') = 28550.94922 ;
vdfb('c_Chem','a_FoodProc','JPN') = 8466.770508 ;
vdfb('c_Chem','a_FoodProc','IND') = 3502.638916 ;
vdfb('c_Chem','a_FoodProc','SSA') = 1070.129639 ;
vdfb('c_Chem','a_FoodProc','ROW') = 35377.09375 ;
vdfb('c_Chem','a_Energy','USA') = 8546.691406 ;
vdfb('c_Chem','a_Energy','EU_28') = 6520.459473 ;
vdfb('c_Chem','a_Energy','CHN') = 13171.35059 ;
vdfb('c_Chem','a_Energy','JPN') = 468.0834961 ;
vdfb('c_Chem','a_Energy','IND') = 9159.387695 ;
vdfb('c_Chem','a_Energy','SSA') = 493.8534241 ;
vdfb('c_Chem','a_Energy','ROW') = 19515.54102 ;
vdfb('c_Chem','a_Textiles','USA') = 11336.25293 ;
vdfb('c_Chem','a_Textiles','EU_28') = 14537.31934 ;
vdfb('c_Chem','a_Textiles','CHN') = 69729.15625 ;
vdfb('c_Chem','a_Textiles','JPN') = 4130.211914 ;
vdfb('c_Chem','a_Textiles','IND') = 17455.2832 ;
vdfb('c_Chem','a_Textiles','SSA') = 663.8555298 ;
vdfb('c_Chem','a_Textiles','ROW') = 29824.11328 ;
vdfb('c_Chem','a_Chem','USA') = 230851.6406 ;
vdfb('c_Chem','a_Chem','EU_28') = 162658.4375 ;
vdfb('c_Chem','a_Chem','CHN') = 659854.875 ;
vdfb('c_Chem','a_Chem','JPN') = 139954.4219 ;
vdfb('c_Chem','a_Chem','IND') = 79511.57812 ;
vdfb('c_Chem','a_Chem','SSA') = 5958.670898 ;
vdfb('c_Chem','a_Chem','ROW') = 325458.0625 ;
vdfb('c_Chem','a_Manuf','USA') = 121307.0469 ;
vdfb('c_Chem','a_Manuf','EU_28') = 131324.4062 ;
vdfb('c_Chem','a_Manuf','CHN') = 357087.625 ;
vdfb('c_Chem','a_Manuf','JPN') = 70090.69531 ;
vdfb('c_Chem','a_Manuf','IND') = 41162.97656 ;
vdfb('c_Chem','a_Manuf','SSA') = 5481.613281 ;
vdfb('c_Chem','a_Manuf','ROW') = 191983.4844 ;
vdfb('c_Chem','a_ForestFish','USA') = 2338.490967 ;
vdfb('c_Chem','a_ForestFish','EU_28') = 672.1784058 ;
vdfb('c_Chem','a_ForestFish','CHN') = 3506.401367 ;
vdfb('c_Chem','a_ForestFish','JPN') = 450.7416992 ;
vdfb('c_Chem','a_ForestFish','IND') = 128.2340851 ;
vdfb('c_Chem','a_ForestFish','SSA') = 307.3945312 ;
vdfb('c_Chem','a_ForestFish','ROW') = 3559.883789 ;
vdfb('c_Chem','a_Svces','USA') = 214567.1406 ;
vdfb('c_Chem','a_Svces','EU_28') = 134750.9062 ;
vdfb('c_Chem','a_Svces','CHN') = 412766.1875 ;
vdfb('c_Chem','a_Svces','JPN') = 97407.09375 ;
vdfb('c_Chem','a_Svces','IND') = 52378.34766 ;
vdfb('c_Chem','a_Svces','SSA') = 7831.037109 ;
vdfb('c_Chem','a_Svces','ROW') = 249660.3906 ;
vdfb('c_Manuf','a_Rice','USA') = 368.3267212 ;
vdfb('c_Manuf','a_Rice','EU_28') = 234.3226013 ;
vdfb('c_Manuf','a_Rice','CHN') = 1612.204468 ;
vdfb('c_Manuf','a_Rice','JPN') = 214.4651184 ;
vdfb('c_Manuf','a_Rice','IND') = 220.6308746 ;
vdfb('c_Manuf','a_Rice','SSA') = 138.7349701 ;
vdfb('c_Manuf','a_Rice','ROW') = 3360.431885 ;
vdfb('c_Manuf','a_Crops','USA') = 4227.12793 ;
vdfb('c_Manuf','a_Crops','EU_28') = 8836.630859 ;
vdfb('c_Manuf','a_Crops','CHN') = 7665.103516 ;
vdfb('c_Manuf','a_Crops','JPN') = 1718.393555 ;
vdfb('c_Manuf','a_Crops','IND') = 651.3054199 ;
vdfb('c_Manuf','a_Crops','SSA') = 1841.930542 ;
vdfb('c_Manuf','a_Crops','ROW') = 12921.52246 ;
vdfb('c_Manuf','a_Livestock','USA') = 2368.661865 ;
vdfb('c_Manuf','a_Livestock','EU_28') = 2596.462158 ;
vdfb('c_Manuf','a_Livestock','CHN') = 1868.959839 ;
vdfb('c_Manuf','a_Livestock','JPN') = 253.2546692 ;
vdfb('c_Manuf','a_Livestock','IND') = 531.7896729 ;
vdfb('c_Manuf','a_Livestock','SSA') = 334.4142151 ;
vdfb('c_Manuf','a_Livestock','ROW') = 5916.144531 ;
vdfb('c_Manuf','a_FoodProc','USA') = 67874.75 ;
vdfb('c_Manuf','a_FoodProc','EU_28') = 52366.59766 ;
vdfb('c_Manuf','a_FoodProc','CHN') = 44743.79688 ;
vdfb('c_Manuf','a_FoodProc','JPN') = 12059.30469 ;
vdfb('c_Manuf','a_FoodProc','IND') = 816.949707 ;
vdfb('c_Manuf','a_FoodProc','SSA') = 3261.459961 ;
vdfb('c_Manuf','a_FoodProc','ROW') = 61989.78516 ;
vdfb('c_Manuf','a_Energy','USA') = 30330.7832 ;
vdfb('c_Manuf','a_Energy','EU_28') = 30757.35547 ;
vdfb('c_Manuf','a_Energy','CHN') = 92465.17969 ;
vdfb('c_Manuf','a_Energy','JPN') = 2099.009521 ;
vdfb('c_Manuf','a_Energy','IND') = 27642.45508 ;
vdfb('c_Manuf','a_Energy','SSA') = 2962.340576 ;
vdfb('c_Manuf','a_Energy','ROW') = 139121.2969 ;
vdfb('c_Manuf','a_Textiles','USA') = 3419.964844 ;
vdfb('c_Manuf','a_Textiles','EU_28') = 12561.59668 ;
vdfb('c_Manuf','a_Textiles','CHN') = 24331.81641 ;
vdfb('c_Manuf','a_Textiles','JPN') = 742.4973145 ;
vdfb('c_Manuf','a_Textiles','IND') = 2895.271729 ;
vdfb('c_Manuf','a_Textiles','SSA') = 1320.803589 ;
vdfb('c_Manuf','a_Textiles','ROW') = 21280.96484 ;
vdfb('c_Manuf','a_Chem','USA') = 62408.12109 ;
vdfb('c_Manuf','a_Chem','EU_28') = 57681.92969 ;
vdfb('c_Manuf','a_Chem','CHN') = 95930.375 ;
vdfb('c_Manuf','a_Chem','JPN') = 13395.47949 ;
vdfb('c_Manuf','a_Chem','IND') = 8394.795898 ;
vdfb('c_Manuf','a_Chem','SSA') = 2621.947266 ;
vdfb('c_Manuf','a_Chem','ROW') = 71583.47656 ;
vdfb('c_Manuf','a_Manuf','USA') = 878579.3125 ;
vdfb('c_Manuf','a_Manuf','EU_28') = 1261312.25 ;
vdfb('c_Manuf','a_Manuf','CHN') = 3800643.25 ;
vdfb('c_Manuf','a_Manuf','JPN') = 899043.375 ;
vdfb('c_Manuf','a_Manuf','IND') = 163220.3594 ;
vdfb('c_Manuf','a_Manuf','SSA') = 80503.07031 ;
vdfb('c_Manuf','a_Manuf','ROW') = 1681277.875 ;
vdfb('c_Manuf','a_ForestFish','USA') = 1143.984375 ;
vdfb('c_Manuf','a_ForestFish','EU_28') = 3450.604736 ;
vdfb('c_Manuf','a_ForestFish','CHN') = 6445.691406 ;
vdfb('c_Manuf','a_ForestFish','JPN') = 781.208313 ;
vdfb('c_Manuf','a_ForestFish','IND') = 2496.894531 ;
vdfb('c_Manuf','a_ForestFish','SSA') = 850.9290771 ;
vdfb('c_Manuf','a_ForestFish','ROW') = 6469.312012 ;
vdfb('c_Manuf','a_Svces','USA') = 838881.125 ;
vdfb('c_Manuf','a_Svces','EU_28') = 792069.6875 ;
vdfb('c_Manuf','a_Svces','CHN') = 1904681.125 ;
vdfb('c_Manuf','a_Svces','JPN') = 300955.4375 ;
vdfb('c_Manuf','a_Svces','IND') = 161820.1094 ;
vdfb('c_Manuf','a_Svces','SSA') = 65826.10938 ;
vdfb('c_Manuf','a_Svces','ROW') = 1203481.625 ;
vdfb('c_ForestFish','a_Rice','USA') = 176.1541443 ;
vdfb('c_ForestFish','a_Rice','EU_28') = 14.23470974 ;
vdfb('c_ForestFish','a_Rice','CHN') = 3.36878252 ;
vdfb('c_ForestFish','a_Rice','JPN') = 1.854398012 ;
vdfb('c_ForestFish','a_Rice','IND') = 1.446414828 ;
vdfb('c_ForestFish','a_Rice','SSA') = 23.52223778 ;
vdfb('c_ForestFish','a_Rice','ROW') = 444.1826477 ;
vdfb('c_ForestFish','a_Crops','USA') = 20124.94727 ;
vdfb('c_ForestFish','a_Crops','EU_28') = 122.622757 ;
vdfb('c_ForestFish','a_Crops','CHN') = 4.623495579 ;
vdfb('c_ForestFish','a_Crops','JPN') = 10.14191246 ;
vdfb('c_ForestFish','a_Crops','IND') = 1.877640605 ;
vdfb('c_ForestFish','a_Crops','SSA') = 146.0706329 ;
vdfb('c_ForestFish','a_Crops','ROW') = 578.4863281 ;
vdfb('c_ForestFish','a_Livestock','USA') = 3117.894043 ;
vdfb('c_ForestFish','a_Livestock','EU_28') = 110.6371307 ;
vdfb('c_ForestFish','a_Livestock','CHN') = 2.710426331 ;
vdfb('c_ForestFish','a_Livestock','JPN') = 3.89397788 ;
vdfb('c_ForestFish','a_Livestock','IND') = 76.78747559 ;
vdfb('c_ForestFish','a_Livestock','SSA') = 295.7230225 ;
vdfb('c_ForestFish','a_Livestock','ROW') = 795.5411987 ;
vdfb('c_ForestFish','a_FoodProc','USA') = 2866.485352 ;
vdfb('c_ForestFish','a_FoodProc','EU_28') = 4825.367676 ;
vdfb('c_ForestFish','a_FoodProc','CHN') = 61950.47266 ;
vdfb('c_ForestFish','a_FoodProc','JPN') = 7459.042969 ;
vdfb('c_ForestFish','a_FoodProc','IND') = 5288.510254 ;
vdfb('c_ForestFish','a_FoodProc','SSA') = 3854.015869 ;
vdfb('c_ForestFish','a_FoodProc','ROW') = 41977.71484 ;
vdfb('c_ForestFish','a_Energy','USA') = 77.4238739 ;
vdfb('c_ForestFish','a_Energy','EU_28') = 502.565155 ;
vdfb('c_ForestFish','a_Energy','CHN') = 208.2406616 ;
vdfb('c_ForestFish','a_Energy','JPN') = 1.232420921 ;
vdfb('c_ForestFish','a_Energy','IND') = 41.83049011 ;
vdfb('c_ForestFish','a_Energy','SSA') = 37.97538757 ;
vdfb('c_ForestFish','a_Energy','ROW') = 960.9115601 ;
vdfb('c_ForestFish','a_Textiles','USA') = 194.2364349 ;
vdfb('c_ForestFish','a_Textiles','EU_28') = 458.8880005 ;
vdfb('c_ForestFish','a_Textiles','CHN') = 405.1068115 ;
vdfb('c_ForestFish','a_Textiles','JPN') = 9.162454605 ;
vdfb('c_ForestFish','a_Textiles','IND') = 52.10761642 ;
vdfb('c_ForestFish','a_Textiles','SSA') = 50.96036148 ;
vdfb('c_ForestFish','a_Textiles','ROW') = 362.9891663 ;
vdfb('c_ForestFish','a_Chem','USA') = 3432.710205 ;
vdfb('c_ForestFish','a_Chem','EU_28') = 747.0515137 ;
vdfb('c_ForestFish','a_Chem','CHN') = 13151.35156 ;
vdfb('c_ForestFish','a_Chem','JPN') = 95.9938736 ;
vdfb('c_ForestFish','a_Chem','IND') = 5045.749512 ;
vdfb('c_ForestFish','a_Chem','SSA') = 245.0513458 ;
vdfb('c_ForestFish','a_Chem','ROW') = 2970.039551 ;
vdfb('c_ForestFish','a_Manuf','USA') = 12698.16309 ;
vdfb('c_ForestFish','a_Manuf','EU_28') = 22781.41406 ;
vdfb('c_ForestFish','a_Manuf','CHN') = 38282.20703 ;
vdfb('c_ForestFish','a_Manuf','JPN') = 3980.454102 ;
vdfb('c_ForestFish','a_Manuf','IND') = 3049.456787 ;
vdfb('c_ForestFish','a_Manuf','SSA') = 6586.484375 ;
vdfb('c_ForestFish','a_Manuf','ROW') = 46612.45312 ;
vdfb('c_ForestFish','a_ForestFish','USA') = 6444.206543 ;
vdfb('c_ForestFish','a_ForestFish','EU_28') = 9852.634766 ;
vdfb('c_ForestFish','a_ForestFish','CHN') = 8718.243164 ;
vdfb('c_ForestFish','a_ForestFish','JPN') = 1168.157104 ;
vdfb('c_ForestFish','a_ForestFish','IND') = 777.6617432 ;
vdfb('c_ForestFish','a_ForestFish','SSA') = 2038.686279 ;
vdfb('c_ForestFish','a_ForestFish','ROW') = 26501.44531 ;
vdfb('c_ForestFish','a_Svces','USA') = 4988.456055 ;
vdfb('c_ForestFish','a_Svces','EU_28') = 8318.379883 ;
vdfb('c_ForestFish','a_Svces','CHN') = 50035.76953 ;
vdfb('c_ForestFish','a_Svces','JPN') = 3444.163086 ;
vdfb('c_ForestFish','a_Svces','IND') = 21696.37891 ;
vdfb('c_ForestFish','a_Svces','SSA') = 7852.92334 ;
vdfb('c_ForestFish','a_Svces','ROW') = 35695.9375 ;
vdfb('c_Svces','a_Rice','USA') = 1828.345093 ;
vdfb('c_Svces','a_Rice','EU_28') = 1757.1875 ;
vdfb('c_Svces','a_Rice','CHN') = 14247.37012 ;
vdfb('c_Svces','a_Rice','JPN') = 5785.041016 ;
vdfb('c_Svces','a_Rice','IND') = 10688.12109 ;
vdfb('c_Svces','a_Rice','SSA') = 1689.8573 ;
vdfb('c_Svces','a_Rice','ROW') = 44820.63281 ;
vdfb('c_Svces','a_Crops','USA') = 51805.90625 ;
vdfb('c_Svces','a_Crops','EU_28') = 38871.29297 ;
vdfb('c_Svces','a_Crops','CHN') = 53447.73828 ;
vdfb('c_Svces','a_Crops','JPN') = 7377.774902 ;
vdfb('c_Svces','a_Crops','IND') = 13449.30078 ;
vdfb('c_Svces','a_Crops','SSA') = 20840.97656 ;
vdfb('c_Svces','a_Crops','ROW') = 106311.8984 ;
vdfb('c_Svces','a_Livestock','USA') = 41028.01953 ;
vdfb('c_Svces','a_Livestock','EU_28') = 21650.61914 ;
vdfb('c_Svces','a_Livestock','CHN') = 28587.50195 ;
vdfb('c_Svces','a_Livestock','JPN') = 7426.266113 ;
vdfb('c_Svces','a_Livestock','IND') = 15161.55762 ;
vdfb('c_Svces','a_Livestock','SSA') = 6509.177246 ;
vdfb('c_Svces','a_Livestock','ROW') = 59357.5 ;
vdfb('c_Svces','a_FoodProc','USA') = 212471.7969 ;
vdfb('c_Svces','a_FoodProc','EU_28') = 306342.6875 ;
vdfb('c_Svces','a_FoodProc','CHN') = 236671.0156 ;
vdfb('c_Svces','a_FoodProc','JPN') = 56670.49219 ;
vdfb('c_Svces','a_FoodProc','IND') = 38525.45703 ;
vdfb('c_Svces','a_FoodProc','SSA') = 51466.51172 ;
vdfb('c_Svces','a_FoodProc','ROW') = 366390.0938 ;
vdfb('c_Svces','a_Energy','USA') = 171896.0938 ;
vdfb('c_Svces','a_Energy','EU_28') = 109273.0703 ;
vdfb('c_Svces','a_Energy','CHN') = 159079.5156 ;
vdfb('c_Svces','a_Energy','JPN') = 67553.83594 ;
vdfb('c_Svces','a_Energy','IND') = 79410.92188 ;
vdfb('c_Svces','a_Energy','SSA') = 35934.60547 ;
vdfb('c_Svces','a_Energy','ROW') = 423966.5 ;
vdfb('c_Svces','a_Textiles','USA') = 17853.83984 ;
vdfb('c_Svces','a_Textiles','EU_28') = 82522.41406 ;
vdfb('c_Svces','a_Textiles','CHN') = 192315.4219 ;
vdfb('c_Svces','a_Textiles','JPN') = 7925.800781 ;
vdfb('c_Svces','a_Textiles','IND') = 21632.63477 ;
vdfb('c_Svces','a_Textiles','SSA') = 11793.25684 ;
vdfb('c_Svces','a_Textiles','ROW') = 122699.7266 ;
vdfb('c_Svces','a_Chem','USA') = 174007.8594 ;
vdfb('c_Svces','a_Chem','EU_28') = 275820.25 ;
vdfb('c_Svces','a_Chem','CHN') = 278943.875 ;
vdfb('c_Svces','a_Chem','JPN') = 93379.66406 ;
vdfb('c_Svces','a_Chem','IND') = 30589.69727 ;
vdfb('c_Svces','a_Chem','SSA') = 11408.39062 ;
vdfb('c_Svces','a_Chem','ROW') = 258459 ;
vdfb('c_Svces','a_Manuf','USA') = 704665.6875 ;
vdfb('c_Svces','a_Manuf','EU_28') = 1090196 ;
vdfb('c_Svces','a_Manuf','CHN') = 1120412 ;
vdfb('c_Svces','a_Manuf','JPN') = 386164.1875 ;
vdfb('c_Svces','a_Manuf','IND') = 110234.8672 ;
vdfb('c_Svces','a_Manuf','SSA') = 96787.14062 ;
vdfb('c_Svces','a_Manuf','ROW') = 1095532.875 ;
vdfb('c_Svces','a_ForestFish','USA') = 8075.587402 ;
vdfb('c_Svces','a_ForestFish','EU_28') = 11587.9209 ;
vdfb('c_Svces','a_ForestFish','CHN') = 34206.41797 ;
vdfb('c_Svces','a_ForestFish','JPN') = 2930.377686 ;
vdfb('c_Svces','a_ForestFish','IND') = 4331.643555 ;
vdfb('c_Svces','a_ForestFish','SSA') = 8900.345703 ;
vdfb('c_Svces','a_ForestFish','ROW') = 36281.53906 ;
vdfb('c_Svces','a_Svces','USA') = 7294576 ;
vdfb('c_Svces','a_Svces','EU_28') = 6652786 ;
vdfb('c_Svces','a_Svces','CHN') = 3999088.75 ;
vdfb('c_Svces','a_Svces','JPN') = 1611724.625 ;
vdfb('c_Svces','a_Svces','IND') = 495561 ;
vdfb('c_Svces','a_Svces','SSA') = 380480.9688 ;
vdfb('c_Svces','a_Svces','ROW') = 6287437.5 ;

* vdfp data (700 cells)
vdfp('c_Rice','a_Rice','USA') = 1909.799438 ;
vdfp('c_Rice','a_Rice','EU_28') = 1012.663513 ;
vdfp('c_Rice','a_Rice','CHN') = 51821.13672 ;
vdfp('c_Rice','a_Rice','JPN') = 17032.80859 ;
vdfp('c_Rice','a_Rice','IND') = 11403.71582 ;
vdfp('c_Rice','a_Rice','SSA') = 3767.589355 ;
vdfp('c_Rice','a_Rice','ROW') = 131585.25 ;
vdfp('c_Rice','a_Crops','USA') = 4.327552319 ;
vdfp('c_Rice','a_Crops','EU_28') = 170.4723969 ;
vdfp('c_Rice','a_Crops','CHN') = 304.3006287 ;
vdfp('c_Rice','a_Crops','JPN') = 203.2329407 ;
vdfp('c_Rice','a_Crops','IND') = 85.54787445 ;
vdfp('c_Rice','a_Crops','SSA') = 39.74756622 ;
vdfp('c_Rice','a_Crops','ROW') = 722.1536255 ;
vdfp('c_Rice','a_Livestock','USA') = 13.67825794 ;
vdfp('c_Rice','a_Livestock','EU_28') = 57.38222504 ;
vdfp('c_Rice','a_Livestock','CHN') = 8353.755859 ;
vdfp('c_Rice','a_Livestock','JPN') = 516.7631836 ;
vdfp('c_Rice','a_Livestock','IND') = 1178.015747 ;
vdfp('c_Rice','a_Livestock','SSA') = 170.5748749 ;
vdfp('c_Rice','a_Livestock','ROW') = 5352.859375 ;
vdfp('c_Rice','a_FoodProc','USA') = 1542.814331 ;
vdfp('c_Rice','a_FoodProc','EU_28') = 552.1846313 ;
vdfp('c_Rice','a_FoodProc','CHN') = 32805.98047 ;
vdfp('c_Rice','a_FoodProc','JPN') = 3923.408447 ;
vdfp('c_Rice','a_FoodProc','IND') = 39032.10156 ;
vdfp('c_Rice','a_FoodProc','SSA') = 1267.614868 ;
vdfp('c_Rice','a_FoodProc','ROW') = 25182.52539 ;
vdfp('c_Rice','a_Energy','USA') = 0.6881924272 ;
vdfp('c_Rice','a_Energy','EU_28') = 0.5837008953 ;
vdfp('c_Rice','a_Energy','CHN') = 17.97001839 ;
vdfp('c_Rice','a_Energy','JPN') = 0.1590573937 ;
vdfp('c_Rice','a_Energy','IND') = 35.93437576 ;
vdfp('c_Rice','a_Energy','SSA') = 4.042829037 ;
vdfp('c_Rice','a_Energy','ROW') = 1259.348389 ;
vdfp('c_Rice','a_Textiles','USA') = 0.4197565019 ;
vdfp('c_Rice','a_Textiles','EU_28') = 2.205401421 ;
vdfp('c_Rice','a_Textiles','CHN') = 1697.82605 ;
vdfp('c_Rice','a_Textiles','JPN') = 0.3588923216 ;
vdfp('c_Rice','a_Textiles','IND') = 0.6348187327 ;
vdfp('c_Rice','a_Textiles','SSA') = 8.173316002 ;
vdfp('c_Rice','a_Textiles','ROW') = 306.6502686 ;
vdfp('c_Rice','a_Chem','USA') = 115.8311157 ;
vdfp('c_Rice','a_Chem','EU_28') = 10.06532383 ;
vdfp('c_Rice','a_Chem','CHN') = 7288.470703 ;
vdfp('c_Rice','a_Chem','JPN') = 29.3732338 ;
vdfp('c_Rice','a_Chem','IND') = 183.3885193 ;
vdfp('c_Rice','a_Chem','SSA') = 269.9428711 ;
vdfp('c_Rice','a_Chem','ROW') = 793.1850586 ;
vdfp('c_Rice','a_Manuf','USA') = 27.9291954 ;
vdfp('c_Rice','a_Manuf','EU_28') = 15.7706337 ;
vdfp('c_Rice','a_Manuf','CHN') = 3953.181396 ;
vdfp('c_Rice','a_Manuf','JPN') = 99.03513336 ;
vdfp('c_Rice','a_Manuf','IND') = 9.445525169 ;
vdfp('c_Rice','a_Manuf','SSA') = 103.515358 ;
vdfp('c_Rice','a_Manuf','ROW') = 1045.189941 ;
vdfp('c_Rice','a_ForestFish','USA') = 6.562182426 ;
vdfp('c_Rice','a_ForestFish','EU_28') = 2.842156172 ;
vdfp('c_Rice','a_ForestFish','CHN') = 837.9677124 ;
vdfp('c_Rice','a_ForestFish','JPN') = 35.73182678 ;
vdfp('c_Rice','a_ForestFish','IND') = 2.000013351 ;
vdfp('c_Rice','a_ForestFish','SSA') = 1.817522883 ;
vdfp('c_Rice','a_ForestFish','ROW') = 1019.39679 ;
vdfp('c_Rice','a_Svces','USA') = 691.25 ;
vdfp('c_Rice','a_Svces','EU_28') = 1035.530151 ;
vdfp('c_Rice','a_Svces','CHN') = 14506.33789 ;
vdfp('c_Rice','a_Svces','JPN') = 4049.369873 ;
vdfp('c_Rice','a_Svces','IND') = 8524.706055 ;
vdfp('c_Rice','a_Svces','SSA') = 1125.824829 ;
vdfp('c_Rice','a_Svces','ROW') = 17197.5957 ;
vdfp('c_Crops','a_Rice','USA') = 8.257699013 ;
vdfp('c_Crops','a_Rice','EU_28') = 11.09047604 ;
vdfp('c_Crops','a_Rice','CHN') = 1160.595581 ;
vdfp('c_Crops','a_Rice','JPN') = 7.117021084 ;
vdfp('c_Crops','a_Rice','IND') = 259.6284485 ;
vdfp('c_Crops','a_Rice','SSA') = 839.6772461 ;
vdfp('c_Crops','a_Rice','ROW') = 1418.781616 ;
vdfp('c_Crops','a_Crops','USA') = 7168.862793 ;
vdfp('c_Crops','a_Crops','EU_28') = 16466.06445 ;
vdfp('c_Crops','a_Crops','CHN') = 62806.34375 ;
vdfp('c_Crops','a_Crops','JPN') = 467.3180847 ;
vdfp('c_Crops','a_Crops','IND') = 597.6104126 ;
vdfp('c_Crops','a_Crops','SSA') = 30728.10742 ;
vdfp('c_Crops','a_Crops','ROW') = 98757.71875 ;
vdfp('c_Crops','a_Livestock','USA') = 7469.704102 ;
vdfp('c_Crops','a_Livestock','EU_28') = 11149.9082 ;
vdfp('c_Crops','a_Livestock','CHN') = 27692.58789 ;
vdfp('c_Crops','a_Livestock','JPN') = 1029.57666 ;
vdfp('c_Crops','a_Livestock','IND') = 7170.675781 ;
vdfp('c_Crops','a_Livestock','SSA') = 3016.583252 ;
vdfp('c_Crops','a_Livestock','ROW') = 36673.05469 ;
vdfp('c_Crops','a_FoodProc','USA') = 75459.33594 ;
vdfp('c_Crops','a_FoodProc','EU_28') = 46219.61719 ;
vdfp('c_Crops','a_FoodProc','CHN') = 255931.3594 ;
vdfp('c_Crops','a_FoodProc','JPN') = 5480.990234 ;
vdfp('c_Crops','a_FoodProc','IND') = 36930.82031 ;
vdfp('c_Crops','a_FoodProc','SSA') = 22531.93359 ;
vdfp('c_Crops','a_FoodProc','ROW') = 221567.1875 ;
vdfp('c_Crops','a_Energy','USA') = 33.80613708 ;
vdfp('c_Crops','a_Energy','EU_28') = 79.8177948 ;
vdfp('c_Crops','a_Energy','CHN') = 109.357399 ;
vdfp('c_Crops','a_Energy','JPN') = 5.274326324 ;
vdfp('c_Crops','a_Energy','IND') = 1533.258423 ;
vdfp('c_Crops','a_Energy','SSA') = 140.3040924 ;
vdfp('c_Crops','a_Energy','ROW') = 1888.477539 ;
vdfp('c_Crops','a_Textiles','USA') = 543.1290894 ;
vdfp('c_Crops','a_Textiles','EU_28') = 734.3357544 ;
vdfp('c_Crops','a_Textiles','CHN') = 46116.18359 ;
vdfp('c_Crops','a_Textiles','JPN') = 154.4386444 ;
vdfp('c_Crops','a_Textiles','IND') = 21927.28711 ;
vdfp('c_Crops','a_Textiles','SSA') = 1315.686768 ;
vdfp('c_Crops','a_Textiles','ROW') = 18865.69727 ;
vdfp('c_Crops','a_Chem','USA') = 13755.21289 ;
vdfp('c_Crops','a_Chem','EU_28') = 1068.151611 ;
vdfp('c_Crops','a_Chem','CHN') = 31362.84766 ;
vdfp('c_Crops','a_Chem','JPN') = 1345.27356 ;
vdfp('c_Crops','a_Chem','IND') = 3472.008789 ;
vdfp('c_Crops','a_Chem','SSA') = 492.4141235 ;
vdfp('c_Crops','a_Chem','ROW') = 14132.98047 ;
vdfp('c_Crops','a_Manuf','USA') = 378.2329102 ;
vdfp('c_Crops','a_Manuf','EU_28') = 223.4051208 ;
vdfp('c_Crops','a_Manuf','CHN') = 16894.75195 ;
vdfp('c_Crops','a_Manuf','JPN') = 51.29619598 ;
vdfp('c_Crops','a_Manuf','IND') = 103.2218857 ;
vdfp('c_Crops','a_Manuf','SSA') = 388.8641052 ;
vdfp('c_Crops','a_Manuf','ROW') = 1539.023804 ;
vdfp('c_Crops','a_ForestFish','USA') = 270.3457031 ;
vdfp('c_Crops','a_ForestFish','EU_28') = 146.5740814 ;
vdfp('c_Crops','a_ForestFish','CHN') = 1177.500977 ;
vdfp('c_Crops','a_ForestFish','JPN') = 1.849598169 ;
vdfp('c_Crops','a_ForestFish','IND') = 20.43853378 ;
vdfp('c_Crops','a_ForestFish','SSA') = 171.861084 ;
vdfp('c_Crops','a_ForestFish','ROW') = 1242.156494 ;
vdfp('c_Crops','a_Svces','USA') = 6418.902832 ;
vdfp('c_Crops','a_Svces','EU_28') = 12935.50098 ;
vdfp('c_Crops','a_Svces','CHN') = 50883.16016 ;
vdfp('c_Crops','a_Svces','JPN') = 5101.680664 ;
vdfp('c_Crops','a_Svces','IND') = 22361.00977 ;
vdfp('c_Crops','a_Svces','SSA') = 5079.707031 ;
vdfp('c_Crops','a_Svces','ROW') = 41735.55078 ;
vdfp('c_Livestock','a_Rice','USA') = 134.5333862 ;
vdfp('c_Livestock','a_Rice','EU_28') = 20.0268364 ;
vdfp('c_Livestock','a_Rice','CHN') = 31.31263351 ;
vdfp('c_Livestock','a_Rice','JPN') = 122.1657028 ;
vdfp('c_Livestock','a_Rice','IND') = 816.9597168 ;
vdfp('c_Livestock','a_Rice','SSA') = 228.6997986 ;
vdfp('c_Livestock','a_Rice','ROW') = 1085.705078 ;
vdfp('c_Livestock','a_Crops','USA') = 266.3331299 ;
vdfp('c_Livestock','a_Crops','EU_28') = 1561.458252 ;
vdfp('c_Livestock','a_Crops','CHN') = 272.6716309 ;
vdfp('c_Livestock','a_Crops','JPN') = 202.2323303 ;
vdfp('c_Livestock','a_Crops','IND') = 778.5941162 ;
vdfp('c_Livestock','a_Crops','SSA') = 1110.8573 ;
vdfp('c_Livestock','a_Crops','ROW') = 13504.58301 ;
vdfp('c_Livestock','a_Livestock','USA') = 17956.32227 ;
vdfp('c_Livestock','a_Livestock','EU_28') = 5280.59668 ;
vdfp('c_Livestock','a_Livestock','CHN') = 24607.0332 ;
vdfp('c_Livestock','a_Livestock','JPN') = 2017.170044 ;
vdfp('c_Livestock','a_Livestock','IND') = 811.1096802 ;
vdfp('c_Livestock','a_Livestock','SSA') = 9532.228516 ;
vdfp('c_Livestock','a_Livestock','ROW') = 80768.21875 ;
vdfp('c_Livestock','a_FoodProc','USA') = 131153.2031 ;
vdfp('c_Livestock','a_FoodProc','EU_28') = 124519.3359 ;
vdfp('c_Livestock','a_FoodProc','CHN') = 144311.75 ;
vdfp('c_Livestock','a_FoodProc','JPN') = 20948.66797 ;
vdfp('c_Livestock','a_FoodProc','IND') = 17157.39844 ;
vdfp('c_Livestock','a_FoodProc','SSA') = 18272.02539 ;
vdfp('c_Livestock','a_FoodProc','ROW') = 273447.5312 ;
vdfp('c_Livestock','a_Energy','USA') = 0.7532737851 ;
vdfp('c_Livestock','a_Energy','EU_28') = 67.74472046 ;
vdfp('c_Livestock','a_Energy','CHN') = 1.611268878 ;
vdfp('c_Livestock','a_Energy','JPN') = 1.0437783 ;
vdfp('c_Livestock','a_Energy','IND') = 138.289032 ;
vdfp('c_Livestock','a_Energy','SSA') = 11.1359129 ;
vdfp('c_Livestock','a_Energy','ROW') = 215.4863434 ;
vdfp('c_Livestock','a_Textiles','USA') = 166.0616608 ;
vdfp('c_Livestock','a_Textiles','EU_28') = 422.7468872 ;
vdfp('c_Livestock','a_Textiles','CHN') = 22390.57227 ;
vdfp('c_Livestock','a_Textiles','JPN') = 16.10928917 ;
vdfp('c_Livestock','a_Textiles','IND') = 1265.112793 ;
vdfp('c_Livestock','a_Textiles','SSA') = 928.4279175 ;
vdfp('c_Livestock','a_Textiles','ROW') = 5048.929688 ;
vdfp('c_Livestock','a_Chem','USA') = 177.482193 ;
vdfp('c_Livestock','a_Chem','EU_28') = 1091.109009 ;
vdfp('c_Livestock','a_Chem','CHN') = 20790.50781 ;
vdfp('c_Livestock','a_Chem','JPN') = 83.4835434 ;
vdfp('c_Livestock','a_Chem','IND') = 4192.739258 ;
vdfp('c_Livestock','a_Chem','SSA') = 89.17465973 ;
vdfp('c_Livestock','a_Chem','ROW') = 1772.678955 ;
vdfp('c_Livestock','a_Manuf','USA') = 16.95561981 ;
vdfp('c_Livestock','a_Manuf','EU_28') = 148.8776398 ;
vdfp('c_Livestock','a_Manuf','CHN') = 2865.186279 ;
vdfp('c_Livestock','a_Manuf','JPN') = 6.011878014 ;
vdfp('c_Livestock','a_Manuf','IND') = 65.05361938 ;
vdfp('c_Livestock','a_Manuf','SSA') = 58.45211029 ;
vdfp('c_Livestock','a_Manuf','ROW') = 899.53125 ;
vdfp('c_Livestock','a_ForestFish','USA') = 389.2718201 ;
vdfp('c_Livestock','a_ForestFish','EU_28') = 114.764267 ;
vdfp('c_Livestock','a_ForestFish','CHN') = 19.96658516 ;
vdfp('c_Livestock','a_ForestFish','JPN') = 3.30553031 ;
vdfp('c_Livestock','a_ForestFish','IND') = 537.296936 ;
vdfp('c_Livestock','a_ForestFish','SSA') = 747.8695068 ;
vdfp('c_Livestock','a_ForestFish','ROW') = 556.0790405 ;
vdfp('c_Livestock','a_Svces','USA') = 3670.827393 ;
vdfp('c_Livestock','a_Svces','EU_28') = 8496.775391 ;
vdfp('c_Livestock','a_Svces','CHN') = 10367.14648 ;
vdfp('c_Livestock','a_Svces','JPN') = 2726.756836 ;
vdfp('c_Livestock','a_Svces','IND') = 38211.96094 ;
vdfp('c_Livestock','a_Svces','SSA') = 4627.213867 ;
vdfp('c_Livestock','a_Svces','ROW') = 22695.03125 ;
vdfp('c_FoodProc','a_Rice','USA') = 64.35842896 ;
vdfp('c_FoodProc','a_Rice','EU_28') = 42.06490707 ;
vdfp('c_FoodProc','a_Rice','CHN') = 255.5483704 ;
vdfp('c_FoodProc','a_Rice','JPN') = 2.000651121 ;
vdfp('c_FoodProc','a_Rice','IND') = 0.5534915924 ;
vdfp('c_FoodProc','a_Rice','SSA') = 177.9290619 ;
vdfp('c_FoodProc','a_Rice','ROW') = 1373.602783 ;
vdfp('c_FoodProc','a_Crops','USA') = 1.088562369 ;
vdfp('c_FoodProc','a_Crops','EU_28') = 1779.967407 ;
vdfp('c_FoodProc','a_Crops','CHN') = 1354.710083 ;
vdfp('c_FoodProc','a_Crops','JPN') = 62.86376953 ;
vdfp('c_FoodProc','a_Crops','IND') = 1.050860763 ;
vdfp('c_FoodProc','a_Crops','SSA') = 1108.424072 ;
vdfp('c_FoodProc','a_Crops','ROW') = 4307.266602 ;
vdfp('c_FoodProc','a_Livestock','USA') = 33813.22656 ;
vdfp('c_FoodProc','a_Livestock','EU_28') = 37154.06641 ;
vdfp('c_FoodProc','a_Livestock','CHN') = 78352.33594 ;
vdfp('c_FoodProc','a_Livestock','JPN') = 10793.87109 ;
vdfp('c_FoodProc','a_Livestock','IND') = 2185.032227 ;
vdfp('c_FoodProc','a_Livestock','SSA') = 4992.541992 ;
vdfp('c_FoodProc','a_Livestock','ROW') = 73233.41406 ;
vdfp('c_FoodProc','a_FoodProc','USA') = 231872.2031 ;
vdfp('c_FoodProc','a_FoodProc','EU_28') = 235071.4688 ;
vdfp('c_FoodProc','a_FoodProc','CHN') = 367548.6875 ;
vdfp('c_FoodProc','a_FoodProc','JPN') = 47243.80469 ;
vdfp('c_FoodProc','a_FoodProc','IND') = 10359.24902 ;
vdfp('c_FoodProc','a_FoodProc','SSA') = 37440.95312 ;
vdfp('c_FoodProc','a_FoodProc','ROW') = 283940.8125 ;
vdfp('c_FoodProc','a_Energy','USA') = 127.8751755 ;
vdfp('c_FoodProc','a_Energy','EU_28') = 571.1998291 ;
vdfp('c_FoodProc','a_Energy','CHN') = 6835.749512 ;
vdfp('c_FoodProc','a_Energy','JPN') = 3.661335468 ;
vdfp('c_FoodProc','a_Energy','IND') = 80.84739685 ;
vdfp('c_FoodProc','a_Energy','SSA') = 58.46503448 ;
vdfp('c_FoodProc','a_Energy','ROW') = 1573.740234 ;
vdfp('c_FoodProc','a_Textiles','USA') = 1368.782837 ;
vdfp('c_FoodProc','a_Textiles','EU_28') = 2902.290771 ;
vdfp('c_FoodProc','a_Textiles','CHN') = 29373.78516 ;
vdfp('c_FoodProc','a_Textiles','JPN') = 134.1049042 ;
vdfp('c_FoodProc','a_Textiles','IND') = 10.55160713 ;
vdfp('c_FoodProc','a_Textiles','SSA') = 922.2225952 ;
vdfp('c_FoodProc','a_Textiles','ROW') = 6928.520508 ;
vdfp('c_FoodProc','a_Chem','USA') = 5290.742188 ;
vdfp('c_FoodProc','a_Chem','EU_28') = 13740.19531 ;
vdfp('c_FoodProc','a_Chem','CHN') = 46648.1875 ;
vdfp('c_FoodProc','a_Chem','JPN') = 1428.333252 ;
vdfp('c_FoodProc','a_Chem','IND') = 2637.283936 ;
vdfp('c_FoodProc','a_Chem','SSA') = 1293.550293 ;
vdfp('c_FoodProc','a_Chem','ROW') = 11241.01855 ;
vdfp('c_FoodProc','a_Manuf','USA') = 3073.59082 ;
vdfp('c_FoodProc','a_Manuf','EU_28') = 4204.724609 ;
vdfp('c_FoodProc','a_Manuf','CHN') = 43998.21094 ;
vdfp('c_FoodProc','a_Manuf','JPN') = 297.4060059 ;
vdfp('c_FoodProc','a_Manuf','IND') = 27.01740646 ;
vdfp('c_FoodProc','a_Manuf','SSA') = 934.6029663 ;
vdfp('c_FoodProc','a_Manuf','ROW') = 5588.401855 ;
vdfp('c_FoodProc','a_ForestFish','USA') = 762.2024536 ;
vdfp('c_FoodProc','a_ForestFish','EU_28') = 758.0110474 ;
vdfp('c_FoodProc','a_ForestFish','CHN') = 25610.87891 ;
vdfp('c_FoodProc','a_ForestFish','JPN') = 1060.003296 ;
vdfp('c_FoodProc','a_ForestFish','IND') = 31.46822929 ;
vdfp('c_FoodProc','a_ForestFish','SSA') = 272.2540588 ;
vdfp('c_FoodProc','a_ForestFish','ROW') = 15996.20801 ;
vdfp('c_FoodProc','a_Svces','USA') = 161877.9531 ;
vdfp('c_FoodProc','a_Svces','EU_28') = 228347.1719 ;
vdfp('c_FoodProc','a_Svces','CHN') = 269682.75 ;
vdfp('c_FoodProc','a_Svces','JPN') = 51805.91797 ;
vdfp('c_FoodProc','a_Svces','IND') = 9120.539062 ;
vdfp('c_FoodProc','a_Svces','SSA') = 24490.81055 ;
vdfp('c_FoodProc','a_Svces','ROW') = 310558.7812 ;
vdfp('c_Energy','a_Rice','USA') = 132.4571991 ;
vdfp('c_Energy','a_Rice','EU_28') = 214.1130371 ;
vdfp('c_Energy','a_Rice','CHN') = 2451.210938 ;
vdfp('c_Energy','a_Rice','JPN') = 384.4426575 ;
vdfp('c_Energy','a_Rice','IND') = 22234.2207 ;
vdfp('c_Energy','a_Rice','SSA') = 73.09047699 ;
vdfp('c_Energy','a_Rice','ROW') = 4484.092285 ;
vdfp('c_Energy','a_Crops','USA') = 5740.961426 ;
vdfp('c_Energy','a_Crops','EU_28') = 11070.54102 ;
vdfp('c_Energy','a_Crops','CHN') = 11969.84766 ;
vdfp('c_Energy','a_Crops','JPN') = 1081.162598 ;
vdfp('c_Energy','a_Crops','IND') = 16627.92383 ;
vdfp('c_Energy','a_Crops','SSA') = 672.4677734 ;
vdfp('c_Energy','a_Crops','ROW') = 24476.77734 ;
vdfp('c_Energy','a_Livestock','USA') = 4208.319824 ;
vdfp('c_Energy','a_Livestock','EU_28') = 4564.235352 ;
vdfp('c_Energy','a_Livestock','CHN') = 2017.034912 ;
vdfp('c_Energy','a_Livestock','JPN') = 400.3699646 ;
vdfp('c_Energy','a_Livestock','IND') = 49.427845 ;
vdfp('c_Energy','a_Livestock','SSA') = 859.4928589 ;
vdfp('c_Energy','a_Livestock','ROW') = 12913.21582 ;
vdfp('c_Energy','a_FoodProc','USA') = 12845.43164 ;
vdfp('c_Energy','a_FoodProc','EU_28') = 23443.10742 ;
vdfp('c_Energy','a_FoodProc','CHN') = 16119.63672 ;
vdfp('c_Energy','a_FoodProc','JPN') = 5652.378906 ;
vdfp('c_Energy','a_FoodProc','IND') = 4034.277588 ;
vdfp('c_Energy','a_FoodProc','SSA') = 1083.991699 ;
vdfp('c_Energy','a_FoodProc','ROW') = 34730.50781 ;
vdfp('c_Energy','a_Energy','USA') = 316150.8125 ;
vdfp('c_Energy','a_Energy','EU_28') = 138577.8906 ;
vdfp('c_Energy','a_Energy','CHN') = 398523.9375 ;
vdfp('c_Energy','a_Energy','JPN') = 41155.54688 ;
vdfp('c_Energy','a_Energy','IND') = 100749.8594 ;
vdfp('c_Energy','a_Energy','SSA') = 27688.33203 ;
vdfp('c_Energy','a_Energy','ROW') = 777126.625 ;
vdfp('c_Energy','a_Textiles','USA') = 2190.811279 ;
vdfp('c_Energy','a_Textiles','EU_28') = 4829.561523 ;
vdfp('c_Energy','a_Textiles','CHN') = 24281.68945 ;
vdfp('c_Energy','a_Textiles','JPN') = 1115.63562 ;
vdfp('c_Energy','a_Textiles','IND') = 2726.638184 ;
vdfp('c_Energy','a_Textiles','SSA') = 399.3370667 ;
vdfp('c_Energy','a_Textiles','ROW') = 15254.83008 ;
vdfp('c_Energy','a_Chem','USA') = 71418.96094 ;
vdfp('c_Energy','a_Chem','EU_28') = 92862.35156 ;
vdfp('c_Energy','a_Chem','CHN') = 138704.0938 ;
vdfp('c_Energy','a_Chem','JPN') = 41240.92969 ;
vdfp('c_Energy','a_Chem','IND') = 31725.81055 ;
vdfp('c_Energy','a_Chem','SSA') = 3816.657227 ;
vdfp('c_Energy','a_Chem','ROW') = 184057.2031 ;
vdfp('c_Energy','a_Manuf','USA') = 64496.73438 ;
vdfp('c_Energy','a_Manuf','EU_28') = 124671.5391 ;
vdfp('c_Energy','a_Manuf','CHN') = 308253.8125 ;
vdfp('c_Energy','a_Manuf','JPN') = 56905.17578 ;
vdfp('c_Energy','a_Manuf','IND') = 88518.73438 ;
vdfp('c_Energy','a_Manuf','SSA') = 19736.7793 ;
vdfp('c_Energy','a_Manuf','ROW') = 295773.75 ;
vdfp('c_Energy','a_ForestFish','USA') = 1148.176025 ;
vdfp('c_Energy','a_ForestFish','EU_28') = 3966.456543 ;
vdfp('c_Energy','a_ForestFish','CHN') = 3880.497559 ;
vdfp('c_Energy','a_ForestFish','JPN') = 1799.971436 ;
vdfp('c_Energy','a_ForestFish','IND') = 2564.391357 ;
vdfp('c_Energy','a_ForestFish','SSA') = 282.8400269 ;
vdfp('c_Energy','a_ForestFish','ROW') = 8866.180664 ;
vdfp('c_Energy','a_Svces','USA') = 387289.5938 ;
vdfp('c_Energy','a_Svces','EU_28') = 370971.4375 ;
vdfp('c_Energy','a_Svces','CHN') = 225198.6875 ;
vdfp('c_Energy','a_Svces','JPN') = 115048.1797 ;
vdfp('c_Energy','a_Svces','IND') = 104783.8203 ;
vdfp('c_Energy','a_Svces','SSA') = 23535.75781 ;
vdfp('c_Energy','a_Svces','ROW') = 516207.7188 ;
vdfp('c_Textiles','a_Rice','USA') = 26.0480957 ;
vdfp('c_Textiles','a_Rice','EU_28') = 16.15740013 ;
vdfp('c_Textiles','a_Rice','CHN') = 362.7811584 ;
vdfp('c_Textiles','a_Rice','JPN') = 32.62643051 ;
vdfp('c_Textiles','a_Rice','IND') = 3.268854618 ;
vdfp('c_Textiles','a_Rice','SSA') = 17.2143898 ;
vdfp('c_Textiles','a_Rice','ROW') = 392.3832092 ;
vdfp('c_Textiles','a_Crops','USA') = 111.4641953 ;
vdfp('c_Textiles','a_Crops','EU_28') = 284.9355164 ;
vdfp('c_Textiles','a_Crops','CHN') = 28.88716698 ;
vdfp('c_Textiles','a_Crops','JPN') = 47.00327301 ;
vdfp('c_Textiles','a_Crops','IND') = 3.631137133 ;
vdfp('c_Textiles','a_Crops','SSA') = 471.9185181 ;
vdfp('c_Textiles','a_Crops','ROW') = 1009.217834 ;
vdfp('c_Textiles','a_Livestock','USA') = 13.42014217 ;
vdfp('c_Textiles','a_Livestock','EU_28') = 146.0309753 ;
vdfp('c_Textiles','a_Livestock','CHN') = 5.785516262 ;
vdfp('c_Textiles','a_Livestock','JPN') = 10.01307011 ;
vdfp('c_Textiles','a_Livestock','IND') = 3.608666182 ;
vdfp('c_Textiles','a_Livestock','SSA') = 125.9747849 ;
vdfp('c_Textiles','a_Livestock','ROW') = 631.2771606 ;
vdfp('c_Textiles','a_FoodProc','USA') = 141.8894653 ;
vdfp('c_Textiles','a_FoodProc','EU_28') = 1675.584473 ;
vdfp('c_Textiles','a_FoodProc','CHN') = 6923.991699 ;
vdfp('c_Textiles','a_FoodProc','JPN') = 143.5123749 ;
vdfp('c_Textiles','a_FoodProc','IND') = 42.03449249 ;
vdfp('c_Textiles','a_FoodProc','SSA') = 222.7799683 ;
vdfp('c_Textiles','a_FoodProc','ROW') = 3429.859131 ;
vdfp('c_Textiles','a_Energy','USA') = 30.30651093 ;
vdfp('c_Textiles','a_Energy','EU_28') = 201.7912598 ;
vdfp('c_Textiles','a_Energy','CHN') = 2934.400146 ;
vdfp('c_Textiles','a_Energy','JPN') = 21.64062309 ;
vdfp('c_Textiles','a_Energy','IND') = 1006.284424 ;
vdfp('c_Textiles','a_Energy','SSA') = 88.36302948 ;
vdfp('c_Textiles','a_Energy','ROW') = 3083.104736 ;
vdfp('c_Textiles','a_Textiles','USA') = 11479.32812 ;
vdfp('c_Textiles','a_Textiles','EU_28') = 59164.30078 ;
vdfp('c_Textiles','a_Textiles','CHN') = 582961.75 ;
vdfp('c_Textiles','a_Textiles','JPN') = 6408.469238 ;
vdfp('c_Textiles','a_Textiles','IND') = 24637.63867 ;
vdfp('c_Textiles','a_Textiles','SSA') = 4769.25293 ;
vdfp('c_Textiles','a_Textiles','ROW') = 156765.0156 ;
vdfp('c_Textiles','a_Chem','USA') = 2097.553955 ;
vdfp('c_Textiles','a_Chem','EU_28') = 5373.61084 ;
vdfp('c_Textiles','a_Chem','CHN') = 21743.3457 ;
vdfp('c_Textiles','a_Chem','JPN') = 578.4924927 ;
vdfp('c_Textiles','a_Chem','IND') = 3016.434082 ;
vdfp('c_Textiles','a_Chem','SSA') = 156.1761475 ;
vdfp('c_Textiles','a_Chem','ROW') = 5670.603027 ;
vdfp('c_Textiles','a_Manuf','USA') = 12528.54297 ;
vdfp('c_Textiles','a_Manuf','EU_28') = 16235.54395 ;
vdfp('c_Textiles','a_Manuf','CHN') = 68223.21094 ;
vdfp('c_Textiles','a_Manuf','JPN') = 2394.094727 ;
vdfp('c_Textiles','a_Manuf','IND') = 934.6286621 ;
vdfp('c_Textiles','a_Manuf','SSA') = 732.1613159 ;
vdfp('c_Textiles','a_Manuf','ROW') = 21219.14844 ;
vdfp('c_Textiles','a_ForestFish','USA') = 63.25150681 ;
vdfp('c_Textiles','a_ForestFish','EU_28') = 317.6309509 ;
vdfp('c_Textiles','a_ForestFish','CHN') = 200.7083282 ;
vdfp('c_Textiles','a_ForestFish','JPN') = 152.957962 ;
vdfp('c_Textiles','a_ForestFish','IND') = 108.0538025 ;
vdfp('c_Textiles','a_ForestFish','SSA') = 221.7301025 ;
vdfp('c_Textiles','a_ForestFish','ROW') = 982.4723511 ;
vdfp('c_Textiles','a_Svces','USA') = 11399.44238 ;
vdfp('c_Textiles','a_Svces','EU_28') = 22745.61328 ;
vdfp('c_Textiles','a_Svces','CHN') = 93709.96875 ;
vdfp('c_Textiles','a_Svces','JPN') = 6631.835938 ;
vdfp('c_Textiles','a_Svces','IND') = 5262.219727 ;
vdfp('c_Textiles','a_Svces','SSA') = 2976.396973 ;
vdfp('c_Textiles','a_Svces','ROW') = 52164.29297 ;
vdfp('c_Chem','a_Rice','USA') = 359.8747253 ;
vdfp('c_Chem','a_Rice','EU_28') = 155.9082184 ;
vdfp('c_Chem','a_Rice','CHN') = 11273.50586 ;
vdfp('c_Chem','a_Rice','JPN') = 1428.14917 ;
vdfp('c_Chem','a_Rice','IND') = 407.6333923 ;
vdfp('c_Chem','a_Rice','SSA') = 90.90305328 ;
vdfp('c_Chem','a_Rice','ROW') = 5741.130371 ;
vdfp('c_Chem','a_Crops','USA') = 18393.48438 ;
vdfp('c_Chem','a_Crops','EU_28') = 7434.619629 ;
vdfp('c_Chem','a_Crops','CHN') = 72285.96875 ;
vdfp('c_Chem','a_Crops','JPN') = 3238.707764 ;
vdfp('c_Chem','a_Crops','IND') = 119.7914047 ;
vdfp('c_Chem','a_Crops','SSA') = 3375.827393 ;
vdfp('c_Chem','a_Crops','ROW') = 44767.875 ;
vdfp('c_Chem','a_Livestock','USA') = 1792.351807 ;
vdfp('c_Chem','a_Livestock','EU_28') = 1010.535706 ;
vdfp('c_Chem','a_Livestock','CHN') = 3445.442871 ;
vdfp('c_Chem','a_Livestock','JPN') = 480.2748718 ;
vdfp('c_Chem','a_Livestock','IND') = 19.23077202 ;
vdfp('c_Chem','a_Livestock','SSA') = 257.7259521 ;
vdfp('c_Chem','a_Livestock','ROW') = 8342.115234 ;
vdfp('c_Chem','a_FoodProc','USA') = 32834.41016 ;
vdfp('c_Chem','a_FoodProc','EU_28') = 19296.18555 ;
vdfp('c_Chem','a_FoodProc','CHN') = 29726.41992 ;
vdfp('c_Chem','a_FoodProc','JPN') = 8466.770508 ;
vdfp('c_Chem','a_FoodProc','IND') = 3649.256348 ;
vdfp('c_Chem','a_FoodProc','SSA') = 1123.721558 ;
vdfp('c_Chem','a_FoodProc','ROW') = 36269.94141 ;
vdfp('c_Chem','a_Energy','USA') = 8745.610352 ;
vdfp('c_Chem','a_Energy','EU_28') = 6535.405273 ;
vdfp('c_Chem','a_Energy','CHN') = 13901.89941 ;
vdfp('c_Chem','a_Energy','JPN') = 468.0834961 ;
vdfp('c_Chem','a_Energy','IND') = 9520.735352 ;
vdfp('c_Chem','a_Energy','SSA') = 511.1245117 ;
vdfp('c_Chem','a_Energy','ROW') = 19889.79102 ;
vdfp('c_Chem','a_Textiles','USA') = 11629.38184 ;
vdfp('c_Chem','a_Textiles','EU_28') = 14559.74609 ;
vdfp('c_Chem','a_Textiles','CHN') = 73375.25781 ;
vdfp('c_Chem','a_Textiles','JPN') = 4130.211914 ;
vdfp('c_Chem','a_Textiles','IND') = 17931.5918 ;
vdfp('c_Chem','a_Textiles','SSA') = 694.8487549 ;
vdfp('c_Chem','a_Textiles','ROW') = 30276.51367 ;
vdfp('c_Chem','a_Chem','USA') = 237819.9688 ;
vdfp('c_Chem','a_Chem','EU_28') = 163006.3438 ;
vdfp('c_Chem','a_Chem','CHN') = 696174.1875 ;
vdfp('c_Chem','a_Chem','JPN') = 139954.4219 ;
vdfp('c_Chem','a_Chem','IND') = 81579.22656 ;
vdfp('c_Chem','a_Chem','SSA') = 6166.356445 ;
vdfp('c_Chem','a_Chem','ROW') = 329420.9062 ;
vdfp('c_Chem','a_Manuf','USA') = 124403.9219 ;
vdfp('c_Chem','a_Manuf','EU_28') = 131617.2188 ;
vdfp('c_Chem','a_Manuf','CHN') = 374392.9375 ;
vdfp('c_Chem','a_Manuf','JPN') = 70090.69531 ;
vdfp('c_Chem','a_Manuf','IND') = 43093.21875 ;
vdfp('c_Chem','a_Manuf','SSA') = 5608.359863 ;
vdfp('c_Chem','a_Manuf','ROW') = 194708.7188 ;
vdfp('c_Chem','a_ForestFish','USA') = 2373.187256 ;
vdfp('c_Chem','a_ForestFish','EU_28') = 679.0629272 ;
vdfp('c_Chem','a_ForestFish','CHN') = 3710.756348 ;
vdfp('c_Chem','a_ForestFish','JPN') = 450.7416992 ;
vdfp('c_Chem','a_ForestFish','IND') = 135.0803833 ;
vdfp('c_Chem','a_ForestFish','SSA') = 309.6803284 ;
vdfp('c_Chem','a_ForestFish','ROW') = 3679.441406 ;
vdfp('c_Chem','a_Svces','USA') = 224625.8125 ;
vdfp('c_Chem','a_Svces','EU_28') = 139860.3438 ;
vdfp('c_Chem','a_Svces','CHN') = 435300.8125 ;
vdfp('c_Chem','a_Svces','JPN') = 97407.09375 ;
vdfp('c_Chem','a_Svces','IND') = 55437.8125 ;
vdfp('c_Chem','a_Svces','SSA') = 8212.264648 ;
vdfp('c_Chem','a_Svces','ROW') = 258170.2031 ;
vdfp('c_Manuf','a_Rice','USA') = 347.2118835 ;
vdfp('c_Manuf','a_Rice','EU_28') = 228.6915741 ;
vdfp('c_Manuf','a_Rice','CHN') = 1607.695068 ;
vdfp('c_Manuf','a_Rice','JPN') = 213.8429871 ;
vdfp('c_Manuf','a_Rice','IND') = 176.2775269 ;
vdfp('c_Manuf','a_Rice','SSA') = 141.9244843 ;
vdfp('c_Manuf','a_Rice','ROW') = 3422.280029 ;
vdfp('c_Manuf','a_Crops','USA') = 4090.846191 ;
vdfp('c_Manuf','a_Crops','EU_28') = 8347.683594 ;
vdfp('c_Manuf','a_Crops','CHN') = 7528.412109 ;
vdfp('c_Manuf','a_Crops','JPN') = 1707.920654 ;
vdfp('c_Manuf','a_Crops','IND') = 167.8027344 ;
vdfp('c_Manuf','a_Crops','SSA') = 1929.927612 ;
vdfp('c_Manuf','a_Crops','ROW') = 12797.6416 ;
vdfp('c_Manuf','a_Livestock','USA') = 2313.397949 ;
vdfp('c_Manuf','a_Livestock','EU_28') = 2467.239014 ;
vdfp('c_Manuf','a_Livestock','CHN') = 1845.739502 ;
vdfp('c_Manuf','a_Livestock','JPN') = 251.8429565 ;
vdfp('c_Manuf','a_Livestock','IND') = 456.7506104 ;
vdfp('c_Manuf','a_Livestock','SSA') = 342.4998779 ;
vdfp('c_Manuf','a_Livestock','ROW') = 5916.121582 ;
vdfp('c_Manuf','a_FoodProc','USA') = 68248.21094 ;
vdfp('c_Manuf','a_FoodProc','EU_28') = 52546.16797 ;
vdfp('c_Manuf','a_FoodProc','CHN') = 45870.22656 ;
vdfp('c_Manuf','a_FoodProc','JPN') = 12059.30469 ;
vdfp('c_Manuf','a_FoodProc','IND') = 864.428772 ;
vdfp('c_Manuf','a_FoodProc','SSA') = 3456.560791 ;
vdfp('c_Manuf','a_FoodProc','ROW') = 63431.01562 ;
vdfp('c_Manuf','a_Energy','USA') = 29254.625 ;
vdfp('c_Manuf','a_Energy','EU_28') = 30770.36719 ;
vdfp('c_Manuf','a_Energy','CHN') = 94750.76562 ;
vdfp('c_Manuf','a_Energy','JPN') = 2099.009521 ;
vdfp('c_Manuf','a_Energy','IND') = 31527.76562 ;
vdfp('c_Manuf','a_Energy','SSA') = 3058.158203 ;
vdfp('c_Manuf','a_Energy','ROW') = 140644.8906 ;
vdfp('c_Manuf','a_Textiles','USA') = 3482.026611 ;
vdfp('c_Manuf','a_Textiles','EU_28') = 12593.70996 ;
vdfp('c_Manuf','a_Textiles','CHN') = 25037.39844 ;
vdfp('c_Manuf','a_Textiles','JPN') = 742.4973145 ;
vdfp('c_Manuf','a_Textiles','IND') = 3363.860352 ;
vdfp('c_Manuf','a_Textiles','SSA') = 1383.230957 ;
vdfp('c_Manuf','a_Textiles','ROW') = 21658.9043 ;
vdfp('c_Manuf','a_Chem','USA') = 63295.12891 ;
vdfp('c_Manuf','a_Chem','EU_28') = 57053.44141 ;
vdfp('c_Manuf','a_Chem','CHN') = 101003.4297 ;
vdfp('c_Manuf','a_Chem','JPN') = 13395.47949 ;
vdfp('c_Manuf','a_Chem','IND') = 8916.662109 ;
vdfp('c_Manuf','a_Chem','SSA') = 2718.562988 ;
vdfp('c_Manuf','a_Chem','ROW') = 73176.65625 ;
vdfp('c_Manuf','a_Manuf','USA') = 912027.75 ;
vdfp('c_Manuf','a_Manuf','EU_28') = 1263117.125 ;
vdfp('c_Manuf','a_Manuf','CHN') = 3948662.75 ;
vdfp('c_Manuf','a_Manuf','JPN') = 899043.375 ;
vdfp('c_Manuf','a_Manuf','IND') = 175409.0938 ;
vdfp('c_Manuf','a_Manuf','SSA') = 81766.14844 ;
vdfp('c_Manuf','a_Manuf','ROW') = 1702661.125 ;
vdfp('c_Manuf','a_ForestFish','USA') = 1090.136841 ;
vdfp('c_Manuf','a_ForestFish','EU_28') = 3485.44751 ;
vdfp('c_Manuf','a_ForestFish','CHN') = 6665.776855 ;
vdfp('c_Manuf','a_ForestFish','JPN') = 781.208313 ;
vdfp('c_Manuf','a_ForestFish','IND') = 2630.681641 ;
vdfp('c_Manuf','a_ForestFish','SSA') = 896.5093994 ;
vdfp('c_Manuf','a_ForestFish','ROW') = 6601.775879 ;
vdfp('c_Manuf','a_Svces','USA') = 829835.625 ;
vdfp('c_Manuf','a_Svces','EU_28') = 816566.9375 ;
vdfp('c_Manuf','a_Svces','CHN') = 1965628.125 ;
vdfp('c_Manuf','a_Svces','JPN') = 300955.4375 ;
vdfp('c_Manuf','a_Svces','IND') = 182007.5938 ;
vdfp('c_Manuf','a_Svces','SSA') = 68917.35938 ;
vdfp('c_Manuf','a_Svces','ROW') = 1229896.75 ;
vdfp('c_ForestFish','a_Rice','USA') = 171.5830383 ;
vdfp('c_ForestFish','a_Rice','EU_28') = 14.25375748 ;
vdfp('c_ForestFish','a_Rice','CHN') = 3.36878252 ;
vdfp('c_ForestFish','a_Rice','JPN') = 1.849402308 ;
vdfp('c_ForestFish','a_Rice','IND') = 1.446414828 ;
vdfp('c_ForestFish','a_Rice','SSA') = 23.59587479 ;
vdfp('c_ForestFish','a_Rice','ROW') = 443.4173584 ;
vdfp('c_ForestFish','a_Crops','USA') = 19463.33398 ;
vdfp('c_ForestFish','a_Crops','EU_28') = 119.89328 ;
vdfp('c_ForestFish','a_Crops','CHN') = 4.623495579 ;
vdfp('c_ForestFish','a_Crops','JPN') = 10.08015919 ;
vdfp('c_ForestFish','a_Crops','IND') = 1.877640605 ;
vdfp('c_ForestFish','a_Crops','SSA') = 151.7750397 ;
vdfp('c_ForestFish','a_Crops','ROW') = 573.7330322 ;
vdfp('c_ForestFish','a_Livestock','USA') = 3041.90625 ;
vdfp('c_ForestFish','a_Livestock','EU_28') = 106.7314911 ;
vdfp('c_ForestFish','a_Livestock','CHN') = 2.710426331 ;
vdfp('c_ForestFish','a_Livestock','JPN') = 3.874721766 ;
vdfp('c_ForestFish','a_Livestock','IND') = 65.63840485 ;
vdfp('c_ForestFish','a_Livestock','SSA') = 298.6358032 ;
vdfp('c_ForestFish','a_Livestock','ROW') = 794.1749878 ;
vdfp('c_ForestFish','a_FoodProc','USA') = 2878.827393 ;
vdfp('c_ForestFish','a_FoodProc','EU_28') = 4831.72168 ;
vdfp('c_ForestFish','a_FoodProc','CHN') = 62096.30859 ;
vdfp('c_ForestFish','a_FoodProc','JPN') = 7459.042969 ;
vdfp('c_ForestFish','a_FoodProc','IND') = 5306.334961 ;
vdfp('c_ForestFish','a_FoodProc','SSA') = 3895.064453 ;
vdfp('c_ForestFish','a_FoodProc','ROW') = 42134.30078 ;
vdfp('c_ForestFish','a_Energy','USA') = 76.10570526 ;
vdfp('c_ForestFish','a_Energy','EU_28') = 500.6363831 ;
vdfp('c_ForestFish','a_Energy','CHN') = 205.4492645 ;
vdfp('c_ForestFish','a_Energy','JPN') = 1.232420921 ;
vdfp('c_ForestFish','a_Energy','IND') = 41.71484756 ;
vdfp('c_ForestFish','a_Energy','SSA') = 38.16710663 ;
vdfp('c_ForestFish','a_Energy','ROW') = 963.0726929 ;
vdfp('c_ForestFish','a_Textiles','USA') = 190.7759094 ;
vdfp('c_ForestFish','a_Textiles','EU_28') = 458.0014954 ;
vdfp('c_ForestFish','a_Textiles','CHN') = 399.4096069 ;
vdfp('c_ForestFish','a_Textiles','JPN') = 9.162454605 ;
vdfp('c_ForestFish','a_Textiles','IND') = 52.94993973 ;
vdfp('c_ForestFish','a_Textiles','SSA') = 51.04862976 ;
vdfp('c_ForestFish','a_Textiles','ROW') = 367.6912537 ;
vdfp('c_ForestFish','a_Chem','USA') = 3372.175781 ;
vdfp('c_ForestFish','a_Chem','EU_28') = 749.6140137 ;
vdfp('c_ForestFish','a_Chem','CHN') = 12960.81738 ;
vdfp('c_ForestFish','a_Chem','JPN') = 95.9938736 ;
vdfp('c_ForestFish','a_Chem','IND') = 5101.293457 ;
vdfp('c_ForestFish','a_Chem','SSA') = 249.6134644 ;
vdfp('c_ForestFish','a_Chem','ROW') = 3004.545654 ;
vdfp('c_ForestFish','a_Manuf','USA') = 12471.78613 ;
vdfp('c_ForestFish','a_Manuf','EU_28') = 22699.88672 ;
vdfp('c_ForestFish','a_Manuf','CHN') = 37732.01172 ;
vdfp('c_ForestFish','a_Manuf','JPN') = 3980.454102 ;
vdfp('c_ForestFish','a_Manuf','IND') = 3100.67334 ;
vdfp('c_ForestFish','a_Manuf','SSA') = 5674.094727 ;
vdfp('c_ForestFish','a_Manuf','ROW') = 47183.70703 ;
vdfp('c_ForestFish','a_ForestFish','USA') = 6347.384766 ;
vdfp('c_ForestFish','a_ForestFish','EU_28') = 9855.419922 ;
vdfp('c_ForestFish','a_ForestFish','CHN') = 8677.009766 ;
vdfp('c_ForestFish','a_ForestFish','JPN') = 1168.157104 ;
vdfp('c_ForestFish','a_ForestFish','IND') = 777.7148438 ;
vdfp('c_ForestFish','a_ForestFish','SSA') = 2060.05835 ;
vdfp('c_ForestFish','a_ForestFish','ROW') = 26657.5957 ;
vdfp('c_ForestFish','a_Svces','USA') = 4956.678223 ;
vdfp('c_ForestFish','a_Svces','EU_28') = 8409.610352 ;
vdfp('c_ForestFish','a_Svces','CHN') = 49844.13281 ;
vdfp('c_ForestFish','a_Svces','JPN') = 3444.163086 ;
vdfp('c_ForestFish','a_Svces','IND') = 22045.61719 ;
vdfp('c_ForestFish','a_Svces','SSA') = 7912.13623 ;
vdfp('c_ForestFish','a_Svces','ROW') = 35863.60156 ;
vdfp('c_Svces','a_Rice','USA') = 1828.86377 ;
vdfp('c_Svces','a_Rice','EU_28') = 1736.003662 ;
vdfp('c_Svces','a_Rice','CHN') = 14377.84082 ;
vdfp('c_Svces','a_Rice','JPN') = 5760.716309 ;
vdfp('c_Svces','a_Rice','IND') = 7992.012695 ;
vdfp('c_Svces','a_Rice','SSA') = 1705.562378 ;
vdfp('c_Svces','a_Rice','ROW') = 44779.85938 ;
vdfp('c_Svces','a_Crops','USA') = 50171.10547 ;
vdfp('c_Svces','a_Crops','EU_28') = 36714.21484 ;
vdfp('c_Svces','a_Crops','CHN') = 52457.76562 ;
vdfp('c_Svces','a_Crops','JPN') = 7332.393066 ;
vdfp('c_Svces','a_Crops','IND') = 2526.674316 ;
vdfp('c_Svces','a_Crops','SSA') = 21226.49219 ;
vdfp('c_Svces','a_Crops','ROW') = 104949.3906 ;
vdfp('c_Svces','a_Livestock','USA') = 40031.39062 ;
vdfp('c_Svces','a_Livestock','EU_28') = 20763.15625 ;
vdfp('c_Svces','a_Livestock','CHN') = 28143.85938 ;
vdfp('c_Svces','a_Livestock','JPN') = 7380.766602 ;
vdfp('c_Svces','a_Livestock','IND') = 12780.0293 ;
vdfp('c_Svces','a_Livestock','SSA') = 6524.640625 ;
vdfp('c_Svces','a_Livestock','ROW') = 59071.92969 ;
vdfp('c_Svces','a_FoodProc','USA') = 214156.4531 ;
vdfp('c_Svces','a_FoodProc','EU_28') = 306946.9688 ;
vdfp('c_Svces','a_FoodProc','CHN') = 246537.9531 ;
vdfp('c_Svces','a_FoodProc','JPN') = 56670.49219 ;
vdfp('c_Svces','a_FoodProc','IND') = 38536.49219 ;
vdfp('c_Svces','a_FoodProc','SSA') = 52074.56641 ;
vdfp('c_Svces','a_FoodProc','ROW') = 370938 ;
vdfp('c_Svces','a_Energy','USA') = 172307.6875 ;
vdfp('c_Svces','a_Energy','EU_28') = 109715.9375 ;
vdfp('c_Svces','a_Energy','CHN') = 164887.4375 ;
vdfp('c_Svces','a_Energy','JPN') = 67553.83594 ;
vdfp('c_Svces','a_Energy','IND') = 80364.21875 ;
vdfp('c_Svces','a_Energy','SSA') = 36849.99219 ;
vdfp('c_Svces','a_Energy','ROW') = 427874.8125 ;
vdfp('c_Svces','a_Textiles','USA') = 18009.38281 ;
vdfp('c_Svces','a_Textiles','EU_28') = 82959.25 ;
vdfp('c_Svces','a_Textiles','CHN') = 199742.8594 ;
vdfp('c_Svces','a_Textiles','JPN') = 7925.800781 ;
vdfp('c_Svces','a_Textiles','IND') = 21731.22266 ;
vdfp('c_Svces','a_Textiles','SSA') = 11939.06348 ;
vdfp('c_Svces','a_Textiles','ROW') = 123826.8828 ;
vdfp('c_Svces','a_Chem','USA') = 175415.4062 ;
vdfp('c_Svces','a_Chem','EU_28') = 276942.2812 ;
vdfp('c_Svces','a_Chem','CHN') = 289229.1875 ;
vdfp('c_Svces','a_Chem','JPN') = 93379.66406 ;
vdfp('c_Svces','a_Chem','IND') = 30726.63672 ;
vdfp('c_Svces','a_Chem','SSA') = 11690.76562 ;
vdfp('c_Svces','a_Chem','ROW') = 261471.4688 ;
vdfp('c_Svces','a_Manuf','USA') = 718223.625 ;
vdfp('c_Svces','a_Manuf','EU_28') = 1094091 ;
vdfp('c_Svces','a_Manuf','CHN') = 1165707.75 ;
vdfp('c_Svces','a_Manuf','JPN') = 386164.1875 ;
vdfp('c_Svces','a_Manuf','IND') = 111359.9688 ;
vdfp('c_Svces','a_Manuf','SSA') = 98044.75781 ;
vdfp('c_Svces','a_Manuf','ROW') = 1104553 ;
vdfp('c_Svces','a_ForestFish','USA') = 8054.905762 ;
vdfp('c_Svces','a_ForestFish','EU_28') = 11737.78223 ;
vdfp('c_Svces','a_ForestFish','CHN') = 35166.79688 ;
vdfp('c_Svces','a_ForestFish','JPN') = 2930.377686 ;
vdfp('c_Svces','a_ForestFish','IND') = 4363.240723 ;
vdfp('c_Svces','a_ForestFish','SSA') = 9054.05957 ;
vdfp('c_Svces','a_ForestFish','ROW') = 36582.79297 ;
vdfp('c_Svces','a_Svces','USA') = 7352975 ;
vdfp('c_Svces','a_Svces','EU_28') = 6846185 ;
vdfp('c_Svces','a_Svces','CHN') = 4138268.75 ;
vdfp('c_Svces','a_Svces','JPN') = 1611724.625 ;
vdfp('c_Svces','a_Svces','IND') = 504249.375 ;
vdfp('c_Svces','a_Svces','SSA') = 390781.2812 ;
vdfp('c_Svces','a_Svces','ROW') = 6400212 ;

* vmfb data (700 cells)
vmfb('c_Rice','a_Rice','USA') = 60.48835373 ;
vmfb('c_Rice','a_Rice','EU_28') = 181.233017 ;
vmfb('c_Rice','a_Rice','CHN') = 125.7138824 ;
vmfb('c_Rice','a_Rice','JPN') = 3.850282907 ;
vmfb('c_Rice','a_Rice','IND') = 0.3480866849 ;
vmfb('c_Rice','a_Rice','SSA') = 260.9320984 ;
vmfb('c_Rice','a_Rice','ROW') = 1422.571533 ;
vmfb('c_Rice','a_Crops','USA') = 0.1339964867 ;
vmfb('c_Rice','a_Crops','EU_28') = 91.15914154 ;
vmfb('c_Rice','a_Crops','CHN') = 9.444348335 ;
vmfb('c_Rice','a_Crops','JPN') = 0.05587971583 ;
vmfb('c_Rice','a_Crops','IND') = 0.03994027525 ;
vmfb('c_Rice','a_Crops','SSA') = 22.98767853 ;
vmfb('c_Rice','a_Crops','ROW') = 124.7267609 ;
vmfb('c_Rice','a_Livestock','USA') = 0.06122069806 ;
vmfb('c_Rice','a_Livestock','EU_28') = 3.3391366 ;
vmfb('c_Rice','a_Livestock','CHN') = 99.14782715 ;
vmfb('c_Rice','a_Livestock','JPN') = 0.4902597666 ;
vmfb('c_Rice','a_Livestock','IND') = 0.03309823945 ;
vmfb('c_Rice','a_Livestock','SSA') = 122.4364853 ;
vmfb('c_Rice','a_Livestock','ROW') = 155.7140503 ;
vmfb('c_Rice','a_FoodProc','USA') = 708.4730835 ;
vmfb('c_Rice','a_FoodProc','EU_28') = 1066.560791 ;
vmfb('c_Rice','a_FoodProc','CHN') = 922.9833374 ;
vmfb('c_Rice','a_FoodProc','JPN') = 266.4388733 ;
vmfb('c_Rice','a_FoodProc','IND') = 3.444492579 ;
vmfb('c_Rice','a_FoodProc','SSA') = 1969.231812 ;
vmfb('c_Rice','a_FoodProc','ROW') = 5990.733398 ;
vmfb('c_Rice','a_Energy','USA') = 0.00194822019 ;
vmfb('c_Rice','a_Energy','EU_28') = 0.2969454527 ;
vmfb('c_Rice','a_Energy','CHN') = 0.01762035489 ;
vmfb('c_Rice','a_Energy','JPN') = 0.032638859 ;
vmfb('c_Rice','a_Energy','IND') = 0.001125297742 ;
vmfb('c_Rice','a_Energy','SSA') = 2.876867056 ;
vmfb('c_Rice','a_Energy','ROW') = 6.686428547 ;
vmfb('c_Rice','a_Textiles','USA') = 0.00430318946 ;
vmfb('c_Rice','a_Textiles','EU_28') = 1.955079556 ;
vmfb('c_Rice','a_Textiles','CHN') = 1.522222638 ;
vmfb('c_Rice','a_Textiles','JPN') = 0.04886856675 ;
vmfb('c_Rice','a_Textiles','IND') = 0.005367415491 ;
vmfb('c_Rice','a_Textiles','SSA') = 8.372336388 ;
vmfb('c_Rice','a_Textiles','ROW') = 13.71656895 ;
vmfb('c_Rice','a_Chem','USA') = 0.02629317157 ;
vmfb('c_Rice','a_Chem','EU_28') = 9.450286865 ;
vmfb('c_Rice','a_Chem','CHN') = 4.90109396 ;
vmfb('c_Rice','a_Chem','JPN') = 0.6622909904 ;
vmfb('c_Rice','a_Chem','IND') = 0.02761853114 ;
vmfb('c_Rice','a_Chem','SSA') = 63.81906891 ;
vmfb('c_Rice','a_Chem','ROW') = 25.89195824 ;
vmfb('c_Rice','a_Manuf','USA') = 0.1030574366 ;
vmfb('c_Rice','a_Manuf','EU_28') = 4.718075752 ;
vmfb('c_Rice','a_Manuf','CHN') = 3.037428856 ;
vmfb('c_Rice','a_Manuf','JPN') = 0.7033855319 ;
vmfb('c_Rice','a_Manuf','IND') = 0.02184144221 ;
vmfb('c_Rice','a_Manuf','SSA') = 23.078619 ;
vmfb('c_Rice','a_Manuf','ROW') = 31.79608536 ;
vmfb('c_Rice','a_ForestFish','USA') = 0.01401897799 ;
vmfb('c_Rice','a_ForestFish','EU_28') = 1.594964385 ;
vmfb('c_Rice','a_ForestFish','CHN') = 18.4213295 ;
vmfb('c_Rice','a_ForestFish','JPN') = 0.8786650896 ;
vmfb('c_Rice','a_ForestFish','IND') = 0.01977348328 ;
vmfb('c_Rice','a_ForestFish','SSA') = 2.553561687 ;
vmfb('c_Rice','a_ForestFish','ROW') = 31.46564865 ;
vmfb('c_Rice','a_Svces','USA') = 3.395895481 ;
vmfb('c_Rice','a_Svces','EU_28') = 301.6132812 ;
vmfb('c_Rice','a_Svces','CHN') = 168.0889587 ;
vmfb('c_Rice','a_Svces','JPN') = 249.9504395 ;
vmfb('c_Rice','a_Svces','IND') = 0.2241821736 ;
vmfb('c_Rice','a_Svces','SSA') = 459.1246643 ;
vmfb('c_Rice','a_Svces','ROW') = 1363.944458 ;
vmfb('c_Crops','a_Rice','USA') = 14.86710167 ;
vmfb('c_Crops','a_Rice','EU_28') = 39.64731216 ;
vmfb('c_Crops','a_Rice','CHN') = 3128.848877 ;
vmfb('c_Crops','a_Rice','JPN') = 144.3750305 ;
vmfb('c_Crops','a_Rice','IND') = 7.235378742 ;
vmfb('c_Crops','a_Rice','SSA') = 154.5962677 ;
vmfb('c_Crops','a_Rice','ROW') = 3218.12085 ;
vmfb('c_Crops','a_Crops','USA') = 1907.352539 ;
vmfb('c_Crops','a_Crops','EU_28') = 4965.337891 ;
vmfb('c_Crops','a_Crops','CHN') = 3718.899414 ;
vmfb('c_Crops','a_Crops','JPN') = 297.2564087 ;
vmfb('c_Crops','a_Crops','IND') = 505.223114 ;
vmfb('c_Crops','a_Crops','SSA') = 493.078186 ;
vmfb('c_Crops','a_Crops','ROW') = 9576.288086 ;
vmfb('c_Crops','a_Livestock','USA') = 369.7680359 ;
vmfb('c_Crops','a_Livestock','EU_28') = 4828.63916 ;
vmfb('c_Crops','a_Livestock','CHN') = 2128.427734 ;
vmfb('c_Crops','a_Livestock','JPN') = 921.9118652 ;
vmfb('c_Crops','a_Livestock','IND') = 979.4416504 ;
vmfb('c_Crops','a_Livestock','SSA') = 820.3032227 ;
vmfb('c_Crops','a_Livestock','ROW') = 15587.36426 ;
vmfb('c_Crops','a_FoodProc','USA') = 14799.43848 ;
vmfb('c_Crops','a_FoodProc','EU_28') = 58078.31641 ;
vmfb('c_Crops','a_FoodProc','CHN') = 29914.03906 ;
vmfb('c_Crops','a_FoodProc','JPN') = 11023.15527 ;
vmfb('c_Crops','a_FoodProc','IND') = 1720.338501 ;
vmfb('c_Crops','a_FoodProc','SSA') = 2489.545654 ;
vmfb('c_Crops','a_FoodProc','ROW') = 68503.85938 ;
vmfb('c_Crops','a_Energy','USA') = 3.463158369 ;
vmfb('c_Crops','a_Energy','EU_28') = 51.98121643 ;
vmfb('c_Crops','a_Energy','CHN') = 4.497493267 ;
vmfb('c_Crops','a_Energy','JPN') = 0.2920863926 ;
vmfb('c_Crops','a_Energy','IND') = 27.30981255 ;
vmfb('c_Crops','a_Energy','SSA') = 8.352303505 ;
vmfb('c_Crops','a_Energy','ROW') = 98.39885712 ;
vmfb('c_Crops','a_Textiles','USA') = 29.09440804 ;
vmfb('c_Crops','a_Textiles','EU_28') = 391.110199 ;
vmfb('c_Crops','a_Textiles','CHN') = 3253.835938 ;
vmfb('c_Crops','a_Textiles','JPN') = 107.0198059 ;
vmfb('c_Crops','a_Textiles','IND') = 303.7763367 ;
vmfb('c_Crops','a_Textiles','SSA') = 40.22593689 ;
vmfb('c_Crops','a_Textiles','ROW') = 6456.247559 ;
vmfb('c_Crops','a_Chem','USA') = 826.7022095 ;
vmfb('c_Crops','a_Chem','EU_28') = 2395.740967 ;
vmfb('c_Crops','a_Chem','CHN') = 1852.713501 ;
vmfb('c_Crops','a_Chem','JPN') = 953.2808228 ;
vmfb('c_Crops','a_Chem','IND') = 176.8400726 ;
vmfb('c_Crops','a_Chem','SSA') = 30.13173485 ;
vmfb('c_Crops','a_Chem','ROW') = 3829.71167 ;
vmfb('c_Crops','a_Manuf','USA') = 195.7828369 ;
vmfb('c_Crops','a_Manuf','EU_28') = 486.6393433 ;
vmfb('c_Crops','a_Manuf','CHN') = 989.574707 ;
vmfb('c_Crops','a_Manuf','JPN') = 30.47153282 ;
vmfb('c_Crops','a_Manuf','IND') = 13.59422016 ;
vmfb('c_Crops','a_Manuf','SSA') = 50.4525032 ;
vmfb('c_Crops','a_Manuf','ROW') = 879.0714111 ;
vmfb('c_Crops','a_ForestFish','USA') = 116.4501419 ;
vmfb('c_Crops','a_ForestFish','EU_28') = 283.8670349 ;
vmfb('c_Crops','a_ForestFish','CHN') = 71.04924774 ;
vmfb('c_Crops','a_ForestFish','JPN') = 0.7752049565 ;
vmfb('c_Crops','a_ForestFish','IND') = 13.13344765 ;
vmfb('c_Crops','a_ForestFish','SSA') = 41.46600342 ;
vmfb('c_Crops','a_ForestFish','ROW') = 221.4492645 ;
vmfb('c_Crops','a_Svces','USA') = 2607.005615 ;
vmfb('c_Crops','a_Svces','EU_28') = 10316.32031 ;
vmfb('c_Crops','a_Svces','CHN') = 3732.898926 ;
vmfb('c_Crops','a_Svces','JPN') = 1893.074097 ;
vmfb('c_Crops','a_Svces','IND') = 894.7000732 ;
vmfb('c_Crops','a_Svces','SSA') = 427.5800781 ;
vmfb('c_Crops','a_Svces','ROW') = 19270.51758 ;
vmfb('c_Livestock','a_Rice','USA') = 1.115633488 ;
vmfb('c_Livestock','a_Rice','EU_28') = 12.99374676 ;
vmfb('c_Livestock','a_Rice','CHN') = 1.132836223 ;
vmfb('c_Livestock','a_Rice','JPN') = 2.240638494 ;
vmfb('c_Livestock','a_Rice','IND') = 5.509529591 ;
vmfb('c_Livestock','a_Rice','SSA') = 1.794489264 ;
vmfb('c_Livestock','a_Rice','ROW') = 109.155365 ;
vmfb('c_Livestock','a_Crops','USA') = 9.311227798 ;
vmfb('c_Livestock','a_Crops','EU_28') = 191.0492706 ;
vmfb('c_Livestock','a_Crops','CHN') = 1.631801128 ;
vmfb('c_Livestock','a_Crops','JPN') = 2.653899908 ;
vmfb('c_Livestock','a_Crops','IND') = 11.50057411 ;
vmfb('c_Livestock','a_Crops','SSA') = 2.05946517 ;
vmfb('c_Livestock','a_Crops','ROW') = 112.5493774 ;
vmfb('c_Livestock','a_Livestock','USA') = 651.3471069 ;
vmfb('c_Livestock','a_Livestock','EU_28') = 176.3220215 ;
vmfb('c_Livestock','a_Livestock','CHN') = 615.4328613 ;
vmfb('c_Livestock','a_Livestock','JPN') = 72.50975037 ;
vmfb('c_Livestock','a_Livestock','IND') = 3.356042862 ;
vmfb('c_Livestock','a_Livestock','SSA') = 77.87031555 ;
vmfb('c_Livestock','a_Livestock','ROW') = 1651.463257 ;
vmfb('c_Livestock','a_FoodProc','USA') = 2975.097168 ;
vmfb('c_Livestock','a_FoodProc','EU_28') = 15087.33203 ;
vmfb('c_Livestock','a_FoodProc','CHN') = 3462.964355 ;
vmfb('c_Livestock','a_FoodProc','JPN') = 284.7634583 ;
vmfb('c_Livestock','a_FoodProc','IND') = 33.46453857 ;
vmfb('c_Livestock','a_FoodProc','SSA') = 403.8100281 ;
vmfb('c_Livestock','a_FoodProc','ROW') = 7470.824219 ;
vmfb('c_Livestock','a_Energy','USA') = 0.0185928531 ;
vmfb('c_Livestock','a_Energy','EU_28') = 19.3517971 ;
vmfb('c_Livestock','a_Energy','CHN') = 0.1226000413 ;
vmfb('c_Livestock','a_Energy','JPN') = 0.106916301 ;
vmfb('c_Livestock','a_Energy','IND') = 0.4456175268 ;
vmfb('c_Livestock','a_Energy','SSA') = 0.6471878886 ;
vmfb('c_Livestock','a_Energy','ROW') = 3.17826128 ;
vmfb('c_Livestock','a_Textiles','USA') = 1.21642077 ;
vmfb('c_Livestock','a_Textiles','EU_28') = 156.1855621 ;
vmfb('c_Livestock','a_Textiles','CHN') = 1106.171265 ;
vmfb('c_Livestock','a_Textiles','JPN') = 5.094622135 ;
vmfb('c_Livestock','a_Textiles','IND') = 24.01761627 ;
vmfb('c_Livestock','a_Textiles','SSA') = 24.76261711 ;
vmfb('c_Livestock','a_Textiles','ROW') = 991.8764648 ;
vmfb('c_Livestock','a_Chem','USA') = 2.870850325 ;
vmfb('c_Livestock','a_Chem','EU_28') = 771.8670044 ;
vmfb('c_Livestock','a_Chem','CHN') = 633.0626831 ;
vmfb('c_Livestock','a_Chem','JPN') = 3.603037596 ;
vmfb('c_Livestock','a_Chem','IND') = 51.07644272 ;
vmfb('c_Livestock','a_Chem','SSA') = 0.506852746 ;
vmfb('c_Livestock','a_Chem','ROW') = 96.55136871 ;
vmfb('c_Livestock','a_Manuf','USA') = 1.134715319 ;
vmfb('c_Livestock','a_Manuf','EU_28') = 42.63737106 ;
vmfb('c_Livestock','a_Manuf','CHN') = 81.18517303 ;
vmfb('c_Livestock','a_Manuf','JPN') = 2.025376558 ;
vmfb('c_Livestock','a_Manuf','IND') = 0.703625977 ;
vmfb('c_Livestock','a_Manuf','SSA') = 4.035212994 ;
vmfb('c_Livestock','a_Manuf','ROW') = 27.44955444 ;
vmfb('c_Livestock','a_ForestFish','USA') = 17.49068069 ;
vmfb('c_Livestock','a_ForestFish','EU_28') = 19.70446968 ;
vmfb('c_Livestock','a_ForestFish','CHN') = 0.1461976171 ;
vmfb('c_Livestock','a_ForestFish','JPN') = 0.9594054818 ;
vmfb('c_Livestock','a_ForestFish','IND') = 0.06490787119 ;
vmfb('c_Livestock','a_ForestFish','SSA') = 1.226854205 ;
vmfb('c_Livestock','a_ForestFish','ROW') = 13.17309761 ;
vmfb('c_Livestock','a_Svces','USA') = 135.3172455 ;
vmfb('c_Livestock','a_Svces','EU_28') = 1563.469238 ;
vmfb('c_Livestock','a_Svces','CHN') = 374.2361755 ;
vmfb('c_Livestock','a_Svces','JPN') = 123.6765823 ;
vmfb('c_Livestock','a_Svces','IND') = 102.704895 ;
vmfb('c_Livestock','a_Svces','SSA') = 35.64498138 ;
vmfb('c_Livestock','a_Svces','ROW') = 2102.758301 ;
vmfb('c_FoodProc','a_Rice','USA') = 3577.644287 ;
vmfb('c_FoodProc','a_Rice','EU_28') = 87.24240112 ;
vmfb('c_FoodProc','a_Rice','CHN') = 15.05121803 ;
vmfb('c_FoodProc','a_Rice','JPN') = 2.200978041 ;
vmfb('c_FoodProc','a_Rice','IND') = 2.554528952 ;
vmfb('c_FoodProc','a_Rice','SSA') = 32.93291855 ;
vmfb('c_FoodProc','a_Rice','ROW') = 735.1972046 ;
vmfb('c_FoodProc','a_Crops','USA') = 22.75304222 ;
vmfb('c_FoodProc','a_Crops','EU_28') = 1011.857727 ;
vmfb('c_FoodProc','a_Crops','CHN') = 110.4807434 ;
vmfb('c_FoodProc','a_Crops','JPN') = 25.39503479 ;
vmfb('c_FoodProc','a_Crops','IND') = 13.83600807 ;
vmfb('c_FoodProc','a_Crops','SSA') = 1033.228882 ;
vmfb('c_FoodProc','a_Crops','ROW') = 3840.855713 ;
vmfb('c_FoodProc','a_Livestock','USA') = 894.2599487 ;
vmfb('c_FoodProc','a_Livestock','EU_28') = 7116.182617 ;
vmfb('c_FoodProc','a_Livestock','CHN') = 2408.766357 ;
vmfb('c_FoodProc','a_Livestock','JPN') = 1839.644043 ;
vmfb('c_FoodProc','a_Livestock','IND') = 1085.629883 ;
vmfb('c_FoodProc','a_Livestock','SSA') = 1038.535522 ;
vmfb('c_FoodProc','a_Livestock','ROW') = 12510.74219 ;
vmfb('c_FoodProc','a_FoodProc','USA') = 17245.51758 ;
vmfb('c_FoodProc','a_FoodProc','EU_28') = 89129.30469 ;
vmfb('c_FoodProc','a_FoodProc','CHN') = 16577.80664 ;
vmfb('c_FoodProc','a_FoodProc','JPN') = 10843.81152 ;
vmfb('c_FoodProc','a_FoodProc','IND') = 1591.76001 ;
vmfb('c_FoodProc','a_FoodProc','SSA') = 6482.43457 ;
vmfb('c_FoodProc','a_FoodProc','ROW') = 82721.08594 ;
vmfb('c_FoodProc','a_Energy','USA') = 13.47109318 ;
vmfb('c_FoodProc','a_Energy','EU_28') = 184.1845703 ;
vmfb('c_FoodProc','a_Energy','CHN') = 131.5312653 ;
vmfb('c_FoodProc','a_Energy','JPN') = 0.7323576808 ;
vmfb('c_FoodProc','a_Energy','IND') = 9.878072739 ;
vmfb('c_FoodProc','a_Energy','SSA') = 30.53613281 ;
vmfb('c_FoodProc','a_Energy','ROW') = 300.1292725 ;
vmfb('c_FoodProc','a_Textiles','USA') = 179.5069733 ;
vmfb('c_FoodProc','a_Textiles','EU_28') = 2206.553223 ;
vmfb('c_FoodProc','a_Textiles','CHN') = 1328.849365 ;
vmfb('c_FoodProc','a_Textiles','JPN') = 121.8572159 ;
vmfb('c_FoodProc','a_Textiles','IND') = 2.430003405 ;
vmfb('c_FoodProc','a_Textiles','SSA') = 84.53905487 ;
vmfb('c_FoodProc','a_Textiles','ROW') = 2063.214844 ;
vmfb('c_FoodProc','a_Chem','USA') = 350.9767151 ;
vmfb('c_FoodProc','a_Chem','EU_28') = 14150.2041 ;
vmfb('c_FoodProc','a_Chem','CHN') = 1767.004639 ;
vmfb('c_FoodProc','a_Chem','JPN') = 372.3395386 ;
vmfb('c_FoodProc','a_Chem','IND') = 184.9611664 ;
vmfb('c_FoodProc','a_Chem','SSA') = 99.24049377 ;
vmfb('c_FoodProc','a_Chem','ROW') = 2874.470459 ;
vmfb('c_FoodProc','a_Manuf','USA') = 264.9154053 ;
vmfb('c_FoodProc','a_Manuf','EU_28') = 1362.174194 ;
vmfb('c_FoodProc','a_Manuf','CHN') = 857.6647949 ;
vmfb('c_FoodProc','a_Manuf','JPN') = 78.27157593 ;
vmfb('c_FoodProc','a_Manuf','IND') = 2.067677021 ;
vmfb('c_FoodProc','a_Manuf','SSA') = 131.0739441 ;
vmfb('c_FoodProc','a_Manuf','ROW') = 907.5940552 ;
vmfb('c_FoodProc','a_ForestFish','USA') = 24.91542244 ;
vmfb('c_FoodProc','a_ForestFish','EU_28') = 283.853302 ;
vmfb('c_FoodProc','a_ForestFish','CHN') = 795.6211548 ;
vmfb('c_FoodProc','a_ForestFish','JPN') = 182.644516 ;
vmfb('c_FoodProc','a_ForestFish','IND') = 2.3567276 ;
vmfb('c_FoodProc','a_ForestFish','SSA') = 42.94499969 ;
vmfb('c_FoodProc','a_ForestFish','ROW') = 4709.263184 ;
vmfb('c_FoodProc','a_Svces','USA') = 19399.23633 ;
vmfb('c_FoodProc','a_Svces','EU_28') = 58128.11719 ;
vmfb('c_FoodProc','a_Svces','CHN') = 8523.886719 ;
vmfb('c_FoodProc','a_Svces','JPN') = 13927.92676 ;
vmfb('c_FoodProc','a_Svces','IND') = 488.9767151 ;
vmfb('c_FoodProc','a_Svces','SSA') = 4645.455566 ;
vmfb('c_FoodProc','a_Svces','ROW') = 65661.44531 ;
vmfb('c_Energy','a_Rice','USA') = 4.174340248 ;
vmfb('c_Energy','a_Rice','EU_28') = 47.77633286 ;
vmfb('c_Energy','a_Rice','CHN') = 101.6303635 ;
vmfb('c_Energy','a_Rice','JPN') = 29.1375103 ;
vmfb('c_Energy','a_Rice','IND') = 175.0426178 ;
vmfb('c_Energy','a_Rice','SSA') = 10.2583046 ;
vmfb('c_Energy','a_Rice','ROW') = 644.453064 ;
vmfb('c_Energy','a_Crops','USA') = 517.0202637 ;
vmfb('c_Energy','a_Crops','EU_28') = 1631.734619 ;
vmfb('c_Energy','a_Crops','CHN') = 578.333374 ;
vmfb('c_Energy','a_Crops','JPN') = 110.9344025 ;
vmfb('c_Energy','a_Crops','IND') = 278.784668 ;
vmfb('c_Energy','a_Crops','SSA') = 308.6402588 ;
vmfb('c_Energy','a_Crops','ROW') = 3801.13623 ;
vmfb('c_Energy','a_Livestock','USA') = 323.7112122 ;
vmfb('c_Energy','a_Livestock','EU_28') = 426.7485046 ;
vmfb('c_Energy','a_Livestock','CHN') = 138.7085114 ;
vmfb('c_Energy','a_Livestock','JPN') = 14.10831833 ;
vmfb('c_Energy','a_Livestock','IND') = 16.87977791 ;
vmfb('c_Energy','a_Livestock','SSA') = 113.8411331 ;
vmfb('c_Energy','a_Livestock','ROW') = 800.8017578 ;
vmfb('c_Energy','a_FoodProc','USA') = 186.2766571 ;
vmfb('c_Energy','a_FoodProc','EU_28') = 4631.557617 ;
vmfb('c_Energy','a_FoodProc','CHN') = 759.3608398 ;
vmfb('c_Energy','a_FoodProc','JPN') = 896.7839355 ;
vmfb('c_Energy','a_FoodProc','IND') = 278.2876892 ;
vmfb('c_Energy','a_FoodProc','SSA') = 299.1864929 ;
vmfb('c_Energy','a_FoodProc','ROW') = 3821.383301 ;
vmfb('c_Energy','a_Energy','USA') = 156432.4375 ;
vmfb('c_Energy','a_Energy','EU_28') = 313315 ;
vmfb('c_Energy','a_Energy','CHN') = 189651.9062 ;
vmfb('c_Energy','a_Energy','JPN') = 120324.7422 ;
vmfb('c_Energy','a_Energy','IND') = 106756.0078 ;
vmfb('c_Energy','a_Energy','SSA') = 14069.43457 ;
vmfb('c_Energy','a_Energy','ROW') = 375067.7188 ;
vmfb('c_Energy','a_Textiles','USA') = 27.77539253 ;
vmfb('c_Energy','a_Textiles','EU_28') = 824.4799194 ;
vmfb('c_Energy','a_Textiles','CHN') = 584.0578003 ;
vmfb('c_Energy','a_Textiles','JPN') = 99.0807724 ;
vmfb('c_Energy','a_Textiles','IND') = 111.1727829 ;
vmfb('c_Energy','a_Textiles','SSA') = 73.87995911 ;
vmfb('c_Energy','a_Textiles','ROW') = 1479.456055 ;
vmfb('c_Energy','a_Chem','USA') = 7179.572266 ;
vmfb('c_Energy','a_Chem','EU_28') = 24345.5293 ;
vmfb('c_Energy','a_Chem','CHN') = 8757.930664 ;
vmfb('c_Energy','a_Chem','JPN') = 5373.071289 ;
vmfb('c_Energy','a_Chem','IND') = 4862.74707 ;
vmfb('c_Energy','a_Chem','SSA') = 961.5206299 ;
vmfb('c_Energy','a_Chem','ROW') = 32642.84961 ;
vmfb('c_Energy','a_Manuf','USA') = 1322.344238 ;
vmfb('c_Energy','a_Manuf','EU_28') = 21668.0957 ;
vmfb('c_Energy','a_Manuf','CHN') = 14264.3457 ;
vmfb('c_Energy','a_Manuf','JPN') = 6099.987793 ;
vmfb('c_Energy','a_Manuf','IND') = 5850.583984 ;
vmfb('c_Energy','a_Manuf','SSA') = 3308.200439 ;
vmfb('c_Energy','a_Manuf','ROW') = 34183.46094 ;
vmfb('c_Energy','a_ForestFish','USA') = 123.2623596 ;
vmfb('c_Energy','a_ForestFish','EU_28') = 548.3814087 ;
vmfb('c_Energy','a_ForestFish','CHN') = 491.7514038 ;
vmfb('c_Energy','a_ForestFish','JPN') = 204.7935486 ;
vmfb('c_Energy','a_ForestFish','IND') = 82.38459778 ;
vmfb('c_Energy','a_ForestFish','SSA') = 358.7876892 ;
vmfb('c_Energy','a_ForestFish','ROW') = 2479.132324 ;
vmfb('c_Energy','a_Svces','USA') = 32753.53711 ;
vmfb('c_Energy','a_Svces','EU_28') = 90007.88281 ;
vmfb('c_Energy','a_Svces','CHN') = 11842.75684 ;
vmfb('c_Energy','a_Svces','JPN') = 8412.65332 ;
vmfb('c_Energy','a_Svces','IND') = 5389.903809 ;
vmfb('c_Energy','a_Svces','SSA') = 22397.46289 ;
vmfb('c_Energy','a_Svces','ROW') = 141750.0469 ;
vmfb('c_Textiles','a_Rice','USA') = 7.472150326 ;
vmfb('c_Textiles','a_Rice','EU_28') = 14.5686388 ;
vmfb('c_Textiles','a_Rice','CHN') = 18.53999329 ;
vmfb('c_Textiles','a_Rice','JPN') = 50.11898041 ;
vmfb('c_Textiles','a_Rice','IND') = 3.101886749 ;
vmfb('c_Textiles','a_Rice','SSA') = 17.31275368 ;
vmfb('c_Textiles','a_Rice','ROW') = 253.3098297 ;
vmfb('c_Textiles','a_Crops','USA') = 95.78682709 ;
vmfb('c_Textiles','a_Crops','EU_28') = 388.1950684 ;
vmfb('c_Textiles','a_Crops','CHN') = 7.618337631 ;
vmfb('c_Textiles','a_Crops','JPN') = 80.48036957 ;
vmfb('c_Textiles','a_Crops','IND') = 3.966736794 ;
vmfb('c_Textiles','a_Crops','SSA') = 510.8585815 ;
vmfb('c_Textiles','a_Crops','ROW') = 1217.375122 ;
vmfb('c_Textiles','a_Livestock','USA') = 30.3583889 ;
vmfb('c_Textiles','a_Livestock','EU_28') = 179.2806702 ;
vmfb('c_Textiles','a_Livestock','CHN') = 1.677989721 ;
vmfb('c_Textiles','a_Livestock','JPN') = 14.8998518 ;
vmfb('c_Textiles','a_Livestock','IND') = 2.079076529 ;
vmfb('c_Textiles','a_Livestock','SSA') = 104.1929703 ;
vmfb('c_Textiles','a_Livestock','ROW') = 264.3848572 ;
vmfb('c_Textiles','a_FoodProc','USA') = 166.6505737 ;
vmfb('c_Textiles','a_FoodProc','EU_28') = 1452.411743 ;
vmfb('c_Textiles','a_FoodProc','CHN') = 269.7313232 ;
vmfb('c_Textiles','a_FoodProc','JPN') = 232.4787903 ;
vmfb('c_Textiles','a_FoodProc','IND') = 9.285899162 ;
vmfb('c_Textiles','a_FoodProc','SSA') = 252.5281372 ;
vmfb('c_Textiles','a_FoodProc','ROW') = 2017.83667 ;
vmfb('c_Textiles','a_Energy','USA') = 66.49993134 ;
vmfb('c_Textiles','a_Energy','EU_28') = 317.51651 ;
vmfb('c_Textiles','a_Energy','CHN') = 120.1287689 ;
vmfb('c_Textiles','a_Energy','JPN') = 42.11201096 ;
vmfb('c_Textiles','a_Energy','IND') = 118.2641449 ;
vmfb('c_Textiles','a_Energy','SSA') = 378.2735596 ;
vmfb('c_Textiles','a_Energy','ROW') = 1492.099609 ;
vmfb('c_Textiles','a_Textiles','USA') = 6547.404297 ;
vmfb('c_Textiles','a_Textiles','EU_28') = 74044.85938 ;
vmfb('c_Textiles','a_Textiles','CHN') = 28392.79688 ;
vmfb('c_Textiles','a_Textiles','JPN') = 6348.95752 ;
vmfb('c_Textiles','a_Textiles','IND') = 2445.539307 ;
vmfb('c_Textiles','a_Textiles','SSA') = 4066.482666 ;
vmfb('c_Textiles','a_Textiles','ROW') = 109542.8359 ;
vmfb('c_Textiles','a_Chem','USA') = 1469.489502 ;
vmfb('c_Textiles','a_Chem','EU_28') = 10471.18555 ;
vmfb('c_Textiles','a_Chem','CHN') = 769.7293701 ;
vmfb('c_Textiles','a_Chem','JPN') = 618.9006958 ;
vmfb('c_Textiles','a_Chem','IND') = 373.9119263 ;
vmfb('c_Textiles','a_Chem','SSA') = 163.8496704 ;
vmfb('c_Textiles','a_Chem','ROW') = 4682.442871 ;
vmfb('c_Textiles','a_Manuf','USA') = 7608.016602 ;
vmfb('c_Textiles','a_Manuf','EU_28') = 20032.59375 ;
vmfb('c_Textiles','a_Manuf','CHN') = 3342.009521 ;
vmfb('c_Textiles','a_Manuf','JPN') = 3352.326904 ;
vmfb('c_Textiles','a_Manuf','IND') = 121.8386383 ;
vmfb('c_Textiles','a_Manuf','SSA') = 789.5548706 ;
vmfb('c_Textiles','a_Manuf','ROW') = 17834.72656 ;
vmfb('c_Textiles','a_ForestFish','USA') = 58.41469193 ;
vmfb('c_Textiles','a_ForestFish','EU_28') = 444.6881714 ;
vmfb('c_Textiles','a_ForestFish','CHN') = 60.60900116 ;
vmfb('c_Textiles','a_ForestFish','JPN') = 174.032074 ;
vmfb('c_Textiles','a_ForestFish','IND') = 17.76542664 ;
vmfb('c_Textiles','a_ForestFish','SSA') = 378.9949951 ;
vmfb('c_Textiles','a_ForestFish','ROW') = 927.8796997 ;
vmfb('c_Textiles','a_Svces','USA') = 18679.19141 ;
vmfb('c_Textiles','a_Svces','EU_28') = 31754.72461 ;
vmfb('c_Textiles','a_Svces','CHN') = 3446.982666 ;
vmfb('c_Textiles','a_Svces','JPN') = 10723.43262 ;
vmfb('c_Textiles','a_Svces','IND') = 545.9885254 ;
vmfb('c_Textiles','a_Svces','SSA') = 3467.42749 ;
vmfb('c_Textiles','a_Svces','ROW') = 35142.94922 ;
vmfb('c_Chem','a_Rice','USA') = 75.55315399 ;
vmfb('c_Chem','a_Rice','EU_28') = 162.0815887 ;
vmfb('c_Chem','a_Rice','CHN') = 2326.698486 ;
vmfb('c_Chem','a_Rice','JPN') = 337.4963684 ;
vmfb('c_Chem','a_Rice','IND') = 1231.19458 ;
vmfb('c_Chem','a_Rice','SSA') = 222.4669952 ;
vmfb('c_Chem','a_Rice','ROW') = 6374.95752 ;
vmfb('c_Chem','a_Crops','USA') = 5557.206543 ;
vmfb('c_Chem','a_Crops','EU_28') = 8050.106445 ;
vmfb('c_Chem','a_Crops','CHN') = 15111.54785 ;
vmfb('c_Chem','a_Crops','JPN') = 710.7751465 ;
vmfb('c_Chem','a_Crops','IND') = 1753.304199 ;
vmfb('c_Chem','a_Crops','SSA') = 5410.023926 ;
vmfb('c_Chem','a_Crops','ROW') = 29204.30273 ;
vmfb('c_Chem','a_Livestock','USA') = 525.4929199 ;
vmfb('c_Chem','a_Livestock','EU_28') = 3789.00293 ;
vmfb('c_Chem','a_Livestock','CHN') = 178.132019 ;
vmfb('c_Chem','a_Livestock','JPN') = 151.0594482 ;
vmfb('c_Chem','a_Livestock','IND') = 21.12916756 ;
vmfb('c_Chem','a_Livestock','SSA') = 519.5646362 ;
vmfb('c_Chem','a_Livestock','ROW') = 4911.522461 ;
vmfb('c_Chem','a_FoodProc','USA') = 6570.876465 ;
vmfb('c_Chem','a_FoodProc','EU_28') = 25125.95117 ;
vmfb('c_Chem','a_FoodProc','CHN') = 6829.356445 ;
vmfb('c_Chem','a_FoodProc','JPN') = 1652.791016 ;
vmfb('c_Chem','a_FoodProc','IND') = 1098.258789 ;
vmfb('c_Chem','a_FoodProc','SSA') = 1759.373413 ;
vmfb('c_Chem','a_FoodProc','ROW') = 28097.82812 ;
vmfb('c_Chem','a_Energy','USA') = 2708.979004 ;
vmfb('c_Chem','a_Energy','EU_28') = 7126.275391 ;
vmfb('c_Chem','a_Energy','CHN') = 2711.926025 ;
vmfb('c_Chem','a_Energy','JPN') = 230.984787 ;
vmfb('c_Chem','a_Energy','IND') = 4543.683105 ;
vmfb('c_Chem','a_Energy','SSA') = 1246.63501 ;
vmfb('c_Chem','a_Energy','ROW') = 15225.44238 ;
vmfb('c_Chem','a_Textiles','USA') = 2724.866211 ;
vmfb('c_Chem','a_Textiles','EU_28') = 17566.26758 ;
vmfb('c_Chem','a_Textiles','CHN') = 12554.38965 ;
vmfb('c_Chem','a_Textiles','JPN') = 864.3790894 ;
vmfb('c_Chem','a_Textiles','IND') = 4454.385254 ;
vmfb('c_Chem','a_Textiles','SSA') = 659.1970215 ;
vmfb('c_Chem','a_Textiles','ROW') = 19549.48242 ;
vmfb('c_Chem','a_Chem','USA') = 57402.83984 ;
vmfb('c_Chem','a_Chem','EU_28') = 286369.875 ;
vmfb('c_Chem','a_Chem','CHN') = 95285.35938 ;
vmfb('c_Chem','a_Chem','JPN') = 29298.76367 ;
vmfb('c_Chem','a_Chem','IND') = 25307.76758 ;
vmfb('c_Chem','a_Chem','SSA') = 5330.46875 ;
vmfb('c_Chem','a_Chem','ROW') = 248747.5 ;
vmfb('c_Chem','a_Manuf','USA') = 29388.03125 ;
vmfb('c_Chem','a_Manuf','EU_28') = 109701.7031 ;
vmfb('c_Chem','a_Manuf','CHN') = 54351.22266 ;
vmfb('c_Chem','a_Manuf','JPN') = 11415.18555 ;
vmfb('c_Chem','a_Manuf','IND') = 6066.159668 ;
vmfb('c_Chem','a_Manuf','SSA') = 4750.274902 ;
vmfb('c_Chem','a_Manuf','ROW') = 131387.8125 ;
vmfb('c_Chem','a_ForestFish','USA') = 943.1162109 ;
vmfb('c_Chem','a_ForestFish','EU_28') = 878.8380737 ;
vmfb('c_Chem','a_ForestFish','CHN') = 561.534729 ;
vmfb('c_Chem','a_ForestFish','JPN') = 71.30358887 ;
vmfb('c_Chem','a_ForestFish','IND') = 16.86178589 ;
vmfb('c_Chem','a_ForestFish','SSA') = 215.9802856 ;
vmfb('c_Chem','a_ForestFish','ROW') = 1678.352905 ;
vmfb('c_Chem','a_Svces','USA') = 62922.75 ;
vmfb('c_Chem','a_Svces','EU_28') = 202469.0625 ;
vmfb('c_Chem','a_Svces','CHN') = 54709.50391 ;
vmfb('c_Chem','a_Svces','JPN') = 28923.49023 ;
vmfb('c_Chem','a_Svces','IND') = 9972.44043 ;
vmfb('c_Chem','a_Svces','SSA') = 10250.66016 ;
vmfb('c_Chem','a_Svces','ROW') = 163278.875 ;
vmfb('c_Manuf','a_Rice','USA') = 39.04264832 ;
vmfb('c_Manuf','a_Rice','EU_28') = 136.9023895 ;
vmfb('c_Manuf','a_Rice','CHN') = 190.8116913 ;
vmfb('c_Manuf','a_Rice','JPN') = 20.19631958 ;
vmfb('c_Manuf','a_Rice','IND') = 48.9920311 ;
vmfb('c_Manuf','a_Rice','SSA') = 116.9915466 ;
vmfb('c_Manuf','a_Rice','ROW') = 1502.46936 ;
vmfb('c_Manuf','a_Crops','USA') = 1757.918701 ;
vmfb('c_Manuf','a_Crops','EU_28') = 3018.962158 ;
vmfb('c_Manuf','a_Crops','CHN') = 1085.23584 ;
vmfb('c_Manuf','a_Crops','JPN') = 84.81323242 ;
vmfb('c_Manuf','a_Crops','IND') = 82.0761795 ;
vmfb('c_Manuf','a_Crops','SSA') = 2783.230225 ;
vmfb('c_Manuf','a_Crops','ROW') = 7689.478516 ;
vmfb('c_Manuf','a_Livestock','USA') = 1126.944946 ;
vmfb('c_Manuf','a_Livestock','EU_28') = 1634.748413 ;
vmfb('c_Manuf','a_Livestock','CHN') = 307.4096069 ;
vmfb('c_Manuf','a_Livestock','JPN') = 44.46448898 ;
vmfb('c_Manuf','a_Livestock','IND') = 127.0065002 ;
vmfb('c_Manuf','a_Livestock','SSA') = 444.3458252 ;
vmfb('c_Manuf','a_Livestock','ROW') = 4539.58252 ;
vmfb('c_Manuf','a_FoodProc','USA') = 12076.95996 ;
vmfb('c_Manuf','a_FoodProc','EU_28') = 23451.66797 ;
vmfb('c_Manuf','a_FoodProc','CHN') = 3490.25708 ;
vmfb('c_Manuf','a_FoodProc','JPN') = 1047.988403 ;
vmfb('c_Manuf','a_FoodProc','IND') = 377.2027588 ;
vmfb('c_Manuf','a_FoodProc','SSA') = 3485.798096 ;
vmfb('c_Manuf','a_FoodProc','ROW') = 25494.86328 ;
vmfb('c_Manuf','a_Energy','USA') = 6834.032227 ;
vmfb('c_Manuf','a_Energy','EU_28') = 18172.77734 ;
vmfb('c_Manuf','a_Energy','CHN') = 7716.378906 ;
vmfb('c_Manuf','a_Energy','JPN') = 696.739624 ;
vmfb('c_Manuf','a_Energy','IND') = 4208.384277 ;
vmfb('c_Manuf','a_Energy','SSA') = 5099.111816 ;
vmfb('c_Manuf','a_Energy','ROW') = 72732.59375 ;
vmfb('c_Manuf','a_Textiles','USA') = 1029.290161 ;
vmfb('c_Manuf','a_Textiles','EU_28') = 9963.199219 ;
vmfb('c_Manuf','a_Textiles','CHN') = 3331.518555 ;
vmfb('c_Manuf','a_Textiles','JPN') = 167.90448 ;
vmfb('c_Manuf','a_Textiles','IND') = 689.7332764 ;
vmfb('c_Manuf','a_Textiles','SSA') = 717.0878906 ;
vmfb('c_Manuf','a_Textiles','ROW') = 12599.89551 ;
vmfb('c_Manuf','a_Chem','USA') = 15046.06543 ;
vmfb('c_Manuf','a_Chem','EU_28') = 44097.88672 ;
vmfb('c_Manuf','a_Chem','CHN') = 17998.91406 ;
vmfb('c_Manuf','a_Chem','JPN') = 1746.077271 ;
vmfb('c_Manuf','a_Chem','IND') = 4297.597168 ;
vmfb('c_Manuf','a_Chem','SSA') = 1190.07666 ;
vmfb('c_Manuf','a_Chem','ROW') = 34369.41406 ;
vmfb('c_Manuf','a_Manuf','USA') = 385805.8438 ;
vmfb('c_Manuf','a_Manuf','EU_28') = 1311801.625 ;
vmfb('c_Manuf','a_Manuf','CHN') = 665336.875 ;
vmfb('c_Manuf','a_Manuf','JPN') = 154641.4688 ;
vmfb('c_Manuf','a_Manuf','IND') = 99641.57812 ;
vmfb('c_Manuf','a_Manuf','SSA') = 33978.20312 ;
vmfb('c_Manuf','a_Manuf','ROW') = 1326157.375 ;
vmfb('c_Manuf','a_ForestFish','USA') = 361.683136 ;
vmfb('c_Manuf','a_ForestFish','EU_28') = 2037.83374 ;
vmfb('c_Manuf','a_ForestFish','CHN') = 757.7175293 ;
vmfb('c_Manuf','a_ForestFish','JPN') = 163.1708221 ;
vmfb('c_Manuf','a_ForestFish','IND') = 619.2880249 ;
vmfb('c_Manuf','a_ForestFish','SSA') = 1126.487305 ;
vmfb('c_Manuf','a_ForestFish','ROW') = 3921.138428 ;
vmfb('c_Manuf','a_Svces','USA') = 242852.2344 ;
vmfb('c_Manuf','a_Svces','EU_28') = 489558.0938 ;
vmfb('c_Manuf','a_Svces','CHN') = 184028.2344 ;
vmfb('c_Manuf','a_Svces','JPN') = 53263.78906 ;
vmfb('c_Manuf','a_Svces','IND') = 38581.125 ;
vmfb('c_Manuf','a_Svces','SSA') = 41579.28125 ;
vmfb('c_Manuf','a_Svces','ROW') = 618706.375 ;
vmfb('c_ForestFish','a_Rice','USA') = 0.05433415994 ;
vmfb('c_ForestFish','a_Rice','EU_28') = 5.745155334 ;
vmfb('c_ForestFish','a_Rice','CHN') = 0.2450711429 ;
vmfb('c_ForestFish','a_Rice','JPN') = 0.04287629575 ;
vmfb('c_ForestFish','a_Rice','IND') = 0.01898382232 ;
vmfb('c_ForestFish','a_Rice','SSA') = 0.07930976897 ;
vmfb('c_ForestFish','a_Rice','ROW') = 11.51736355 ;
vmfb('c_ForestFish','a_Crops','USA') = 4.38259697 ;
vmfb('c_ForestFish','a_Crops','EU_28') = 18.08460045 ;
vmfb('c_ForestFish','a_Crops','CHN') = 1.303160906 ;
vmfb('c_ForestFish','a_Crops','JPN') = 1.890565395 ;
vmfb('c_ForestFish','a_Crops','IND') = 0.2417928874 ;
vmfb('c_ForestFish','a_Crops','SSA') = 0.5244846344 ;
vmfb('c_ForestFish','a_Crops','ROW') = 20.50427628 ;
vmfb('c_ForestFish','a_Livestock','USA') = 0.7333287597 ;
vmfb('c_ForestFish','a_Livestock','EU_28') = 10.13285828 ;
vmfb('c_ForestFish','a_Livestock','CHN') = 0.1809564829 ;
vmfb('c_ForestFish','a_Livestock','JPN') = 0.03739479557 ;
vmfb('c_ForestFish','a_Livestock','IND') = 2.521846771 ;
vmfb('c_ForestFish','a_Livestock','SSA') = 1.777273297 ;
vmfb('c_ForestFish','a_Livestock','ROW') = 14.09546947 ;
vmfb('c_ForestFish','a_FoodProc','USA') = 1192.689087 ;
vmfb('c_ForestFish','a_FoodProc','EU_28') = 5087.297363 ;
vmfb('c_ForestFish','a_FoodProc','CHN') = 852.6604614 ;
vmfb('c_ForestFish','a_FoodProc','JPN') = 1066.005005 ;
vmfb('c_ForestFish','a_FoodProc','IND') = 66.17717743 ;
vmfb('c_ForestFish','a_FoodProc','SSA') = 37.78542709 ;
vmfb('c_ForestFish','a_FoodProc','ROW') = 1344.688965 ;
vmfb('c_ForestFish','a_Energy','USA') = 1.304291606 ;
vmfb('c_ForestFish','a_Energy','EU_28') = 38.01895523 ;
vmfb('c_ForestFish','a_Energy','CHN') = 28.44144058 ;
vmfb('c_ForestFish','a_Energy','JPN') = 0.06488320976 ;
vmfb('c_ForestFish','a_Energy','IND') = 0.4973107278 ;
vmfb('c_ForestFish','a_Energy','SSA') = 0.4909920096 ;
vmfb('c_ForestFish','a_Energy','ROW') = 5.634269714 ;
vmfb('c_ForestFish','a_Textiles','USA') = 3.34718132 ;
vmfb('c_ForestFish','a_Textiles','EU_28') = 114.5797119 ;
vmfb('c_ForestFish','a_Textiles','CHN') = 53.09630585 ;
vmfb('c_ForestFish','a_Textiles','JPN') = 1.634410381 ;
vmfb('c_ForestFish','a_Textiles','IND') = 1.635375857 ;
vmfb('c_ForestFish','a_Textiles','SSA') = 0.1412920803 ;
vmfb('c_ForestFish','a_Textiles','ROW') = 5.137924194 ;
vmfb('c_ForestFish','a_Chem','USA') = 67.18754578 ;
vmfb('c_ForestFish','a_Chem','EU_28') = 379.2522583 ;
vmfb('c_ForestFish','a_Chem','CHN') = 1780.792603 ;
vmfb('c_ForestFish','a_Chem','JPN') = 21.13298225 ;
vmfb('c_ForestFish','a_Chem','IND') = 120.3005676 ;
vmfb('c_ForestFish','a_Chem','SSA') = 5.902513981 ;
vmfb('c_ForestFish','a_Chem','ROW') = 186.2387543 ;
vmfb('c_ForestFish','a_Manuf','USA') = 213.2220001 ;
vmfb('c_ForestFish','a_Manuf','EU_28') = 3803.773926 ;
vmfb('c_ForestFish','a_Manuf','CHN') = 5207.436523 ;
vmfb('c_ForestFish','a_Manuf','JPN') = 647.3535156 ;
vmfb('c_ForestFish','a_Manuf','IND') = 100.7577591 ;
vmfb('c_ForestFish','a_Manuf','SSA') = 80.54338837 ;
vmfb('c_ForestFish','a_Manuf','ROW') = 1558.600708 ;
vmfb('c_ForestFish','a_ForestFish','USA') = 36.76443481 ;
vmfb('c_ForestFish','a_ForestFish','EU_28') = 912.7949829 ;
vmfb('c_ForestFish','a_ForestFish','CHN') = 550.5001831 ;
vmfb('c_ForestFish','a_ForestFish','JPN') = 183.0137329 ;
vmfb('c_ForestFish','a_ForestFish','IND') = 5.473010063 ;
vmfb('c_ForestFish','a_ForestFish','SSA') = 24.17269135 ;
vmfb('c_ForestFish','a_ForestFish','ROW') = 1228.006104 ;
vmfb('c_ForestFish','a_Svces','USA') = 892.9715576 ;
vmfb('c_ForestFish','a_Svces','EU_28') = 2813.12085 ;
vmfb('c_ForestFish','a_Svces','CHN') = 2921.278809 ;
vmfb('c_ForestFish','a_Svces','JPN') = 463.3709412 ;
vmfb('c_ForestFish','a_Svces','IND') = 680.519043 ;
vmfb('c_ForestFish','a_Svces','SSA') = 20.08573151 ;
vmfb('c_ForestFish','a_Svces','ROW') = 1538.568726 ;
vmfb('c_Svces','a_Rice','USA') = 26.44984245 ;
vmfb('c_Svces','a_Rice','EU_28') = 152.9582977 ;
vmfb('c_Svces','a_Rice','CHN') = 252.2639008 ;
vmfb('c_Svces','a_Rice','JPN') = 98.60840607 ;
vmfb('c_Svces','a_Rice','IND') = 184.1719208 ;
vmfb('c_Svces','a_Rice','SSA') = 56.94506836 ;
vmfb('c_Svces','a_Rice','ROW') = 1096.015503 ;
vmfb('c_Svces','a_Crops','USA') = 818.1546021 ;
vmfb('c_Svces','a_Crops','EU_28') = 4533.246582 ;
vmfb('c_Svces','a_Crops','CHN') = 1108.221069 ;
vmfb('c_Svces','a_Crops','JPN') = 115.1134109 ;
vmfb('c_Svces','a_Crops','IND') = 587.8417969 ;
vmfb('c_Svces','a_Crops','SSA') = 1447.896851 ;
vmfb('c_Svces','a_Crops','ROW') = 4892.256836 ;
vmfb('c_Svces','a_Livestock','USA') = 573.6816406 ;
vmfb('c_Svces','a_Livestock','EU_28') = 2154.779053 ;
vmfb('c_Svces','a_Livestock','CHN') = 436.3206177 ;
vmfb('c_Svces','a_Livestock','JPN') = 120.0923691 ;
vmfb('c_Svces','a_Livestock','IND') = 143.6562195 ;
vmfb('c_Svces','a_Livestock','SSA') = 355.0368958 ;
vmfb('c_Svces','a_Livestock','ROW') = 2076.54834 ;
vmfb('c_Svces','a_FoodProc','USA') = 2708.391357 ;
vmfb('c_Svces','a_FoodProc','EU_28') = 37972.28906 ;
vmfb('c_Svces','a_FoodProc','CHN') = 4898.236816 ;
vmfb('c_Svces','a_FoodProc','JPN') = 1330.948364 ;
vmfb('c_Svces','a_FoodProc','IND') = 550.701355 ;
vmfb('c_Svces','a_FoodProc','SSA') = 1995.325806 ;
vmfb('c_Svces','a_FoodProc','ROW') = 16050.10938 ;
vmfb('c_Svces','a_Energy','USA') = 2508.917725 ;
vmfb('c_Svces','a_Energy','EU_28') = 9512.299805 ;
vmfb('c_Svces','a_Energy','CHN') = 3307.572754 ;
vmfb('c_Svces','a_Energy','JPN') = 2507.064941 ;
vmfb('c_Svces','a_Energy','IND') = 3762.458496 ;
vmfb('c_Svces','a_Energy','SSA') = 5756.741211 ;
vmfb('c_Svces','a_Energy','ROW') = 34849.67969 ;
vmfb('c_Svces','a_Textiles','USA') = 343.1921082 ;
vmfb('c_Svces','a_Textiles','EU_28') = 10165.22852 ;
vmfb('c_Svces','a_Textiles','CHN') = 4920.981445 ;
vmfb('c_Svces','a_Textiles','JPN') = 196.1030731 ;
vmfb('c_Svces','a_Textiles','IND') = 592.670105 ;
vmfb('c_Svces','a_Textiles','SSA') = 612.4405518 ;
vmfb('c_Svces','a_Textiles','ROW') = 6904.710449 ;
vmfb('c_Svces','a_Chem','USA') = 3547.513916 ;
vmfb('c_Svces','a_Chem','EU_28') = 62748.15234 ;
vmfb('c_Svces','a_Chem','CHN') = 7627.950684 ;
vmfb('c_Svces','a_Chem','JPN') = 2177.458252 ;
vmfb('c_Svces','a_Chem','IND') = 1262.153809 ;
vmfb('c_Svces','a_Chem','SSA') = 793.734436 ;
vmfb('c_Svces','a_Chem','ROW') = 26319.74219 ;
vmfb('c_Svces','a_Manuf','USA') = 17600.53906 ;
vmfb('c_Svces','a_Manuf','EU_28') = 166835.0625 ;
vmfb('c_Svces','a_Manuf','CHN') = 34537.70312 ;
vmfb('c_Svces','a_Manuf','JPN') = 10428.6748 ;
vmfb('c_Svces','a_Manuf','IND') = 5357.34082 ;
vmfb('c_Svces','a_Manuf','SSA') = 5170.827637 ;
vmfb('c_Svces','a_Manuf','ROW') = 79130.86719 ;
vmfb('c_Svces','a_ForestFish','USA') = 232.3023224 ;
vmfb('c_Svces','a_ForestFish','EU_28') = 1596.275635 ;
vmfb('c_Svces','a_ForestFish','CHN') = 956.6572266 ;
vmfb('c_Svces','a_ForestFish','JPN') = 82.32695007 ;
vmfb('c_Svces','a_ForestFish','IND') = 352.3557129 ;
vmfb('c_Svces','a_ForestFish','SSA') = 908.649292 ;
vmfb('c_Svces','a_ForestFish','ROW') = 2659.874756 ;
vmfb('c_Svces','a_Svces','USA') = 221647.8281 ;
vmfb('c_Svces','a_Svces','EU_28') = 1174965.75 ;
vmfb('c_Svces','a_Svces','CHN') = 125330.6562 ;
vmfb('c_Svces','a_Svces','JPN') = 64262.52734 ;
vmfb('c_Svces','a_Svces','IND') = 23730.22461 ;
vmfb('c_Svces','a_Svces','SSA') = 30924.81445 ;
vmfb('c_Svces','a_Svces','ROW') = 647501.625 ;

* vmfp data (700 cells)
vmfp('c_Rice','a_Rice','USA') = 61.17081833 ;
vmfp('c_Rice','a_Rice','EU_28') = 181.9016113 ;
vmfp('c_Rice','a_Rice','CHN') = 136.9499359 ;
vmfp('c_Rice','a_Rice','JPN') = 3.871606827 ;
vmfp('c_Rice','a_Rice','IND') = 0.3258773386 ;
vmfp('c_Rice','a_Rice','SSA') = 269.5458984 ;
vmfp('c_Rice','a_Rice','ROW') = 1435.835205 ;
vmfp('c_Rice','a_Crops','USA') = 0.1339964867 ;
vmfp('c_Rice','a_Crops','EU_28') = 88.39334106 ;
vmfp('c_Rice','a_Crops','CHN') = 9.278704643 ;
vmfp('c_Rice','a_Crops','JPN') = 0.05587971583 ;
vmfp('c_Rice','a_Crops','IND') = 0.03994027525 ;
vmfp('c_Rice','a_Crops','SSA') = 23.50056267 ;
vmfp('c_Rice','a_Crops','ROW') = 125.1598511 ;
vmfp('c_Rice','a_Livestock','USA') = 0.06122069806 ;
vmfp('c_Rice','a_Livestock','EU_28') = 3.168824911 ;
vmfp('c_Rice','a_Livestock','CHN') = 97.62368011 ;
vmfp('c_Rice','a_Livestock','JPN') = 0.4876268804 ;
vmfp('c_Rice','a_Livestock','IND') = 0.03309823945 ;
vmfp('c_Rice','a_Livestock','SSA') = 123.8912888 ;
vmfp('c_Rice','a_Livestock','ROW') = 156.3365784 ;
vmfp('c_Rice','a_FoodProc','USA') = 716.4729004 ;
vmfp('c_Rice','a_FoodProc','EU_28') = 1076.13269 ;
vmfp('c_Rice','a_FoodProc','CHN') = 1006.115601 ;
vmfp('c_Rice','a_FoodProc','JPN') = 274.3141174 ;
vmfp('c_Rice','a_FoodProc','IND') = 3.444492579 ;
vmfp('c_Rice','a_FoodProc','SSA') = 2116.043457 ;
vmfp('c_Rice','a_FoodProc','ROW') = 6224.90918 ;
vmfp('c_Rice','a_Energy','USA') = 0.00194822019 ;
vmfp('c_Rice','a_Energy','EU_28') = 0.298384726 ;
vmfp('c_Rice','a_Energy','CHN') = 0.01762035489 ;
vmfp('c_Rice','a_Energy','JPN') = 0.032638859 ;
vmfp('c_Rice','a_Energy','IND') = 0.001125297742 ;
vmfp('c_Rice','a_Energy','SSA') = 2.899610043 ;
vmfp('c_Rice','a_Energy','ROW') = 6.702142715 ;
vmfp('c_Rice','a_Textiles','USA') = 0.00430318946 ;
vmfp('c_Rice','a_Textiles','EU_28') = 1.955079556 ;
vmfp('c_Rice','a_Textiles','CHN') = 1.585413337 ;
vmfp('c_Rice','a_Textiles','JPN') = 0.04886856675 ;
vmfp('c_Rice','a_Textiles','IND') = 0.005367415491 ;
vmfp('c_Rice','a_Textiles','SSA') = 8.715288162 ;
vmfp('c_Rice','a_Textiles','ROW') = 13.74420071 ;
vmfp('c_Rice','a_Chem','USA') = 0.02629317157 ;
vmfp('c_Rice','a_Chem','EU_28') = 9.762376785 ;
vmfp('c_Rice','a_Chem','CHN') = 5.313475609 ;
vmfp('c_Rice','a_Chem','JPN') = 0.6622909904 ;
vmfp('c_Rice','a_Chem','IND') = 0.02761853114 ;
vmfp('c_Rice','a_Chem','SSA') = 66.3620224 ;
vmfp('c_Rice','a_Chem','ROW') = 26.19374847 ;
vmfp('c_Rice','a_Manuf','USA') = 0.1030574366 ;
vmfp('c_Rice','a_Manuf','EU_28') = 4.777945995 ;
vmfp('c_Rice','a_Manuf','CHN') = 3.21680975 ;
vmfp('c_Rice','a_Manuf','JPN') = 0.7033855319 ;
vmfp('c_Rice','a_Manuf','IND') = 0.02184144221 ;
vmfp('c_Rice','a_Manuf','SSA') = 23.62065887 ;
vmfp('c_Rice','a_Manuf','ROW') = 32.1419754 ;
vmfp('c_Rice','a_ForestFish','USA') = 0.01401897799 ;
vmfp('c_Rice','a_ForestFish','EU_28') = 1.640322208 ;
vmfp('c_Rice','a_ForestFish','CHN') = 20.05663681 ;
vmfp('c_Rice','a_ForestFish','JPN') = 0.9013211131 ;
vmfp('c_Rice','a_ForestFish','IND') = 0.01977348328 ;
vmfp('c_Rice','a_ForestFish','SSA') = 2.577943087 ;
vmfp('c_Rice','a_ForestFish','ROW') = 31.61849022 ;
vmfp('c_Rice','a_Svces','USA') = 3.393145561 ;
vmfp('c_Rice','a_Svces','EU_28') = 314.6420898 ;
vmfp('c_Rice','a_Svces','CHN') = 182.9845886 ;
vmfp('c_Rice','a_Svces','JPN') = 257.3041077 ;
vmfp('c_Rice','a_Svces','IND') = 0.2241821736 ;
vmfp('c_Rice','a_Svces','SSA') = 486.5134277 ;
vmfp('c_Rice','a_Svces','ROW') = 1432.016846 ;
vmfp('c_Crops','a_Rice','USA') = 14.84565353 ;
vmfp('c_Crops','a_Rice','EU_28') = 39.72296524 ;
vmfp('c_Crops','a_Rice','CHN') = 3404.870605 ;
vmfp('c_Crops','a_Rice','JPN') = 151.8690796 ;
vmfp('c_Crops','a_Rice','IND') = 5.604398251 ;
vmfp('c_Crops','a_Rice','SSA') = 157.0717316 ;
vmfp('c_Crops','a_Rice','ROW') = 3252.93335 ;
vmfp('c_Crops','a_Crops','USA') = 1834.071777 ;
vmfp('c_Crops','a_Crops','EU_28') = 4741.956055 ;
vmfp('c_Crops','a_Crops','CHN') = 3650.700928 ;
vmfp('c_Crops','a_Crops','JPN') = 295.4860535 ;
vmfp('c_Crops','a_Crops','IND') = 59.34206772 ;
vmfp('c_Crops','a_Crops','SSA') = 504.5728149 ;
vmfp('c_Crops','a_Crops','ROW') = 9499.618164 ;
vmfp('c_Crops','a_Livestock','USA') = 360.6622314 ;
vmfp('c_Crops','a_Livestock','EU_28') = 4607.293945 ;
vmfp('c_Crops','a_Livestock','CHN') = 2095.3479 ;
vmfp('c_Crops','a_Livestock','JPN') = 916.4198608 ;
vmfp('c_Crops','a_Livestock','IND') = 832.7177734 ;
vmfp('c_Crops','a_Livestock','SSA') = 830.2697754 ;
vmfp('c_Crops','a_Livestock','ROW') = 15573.75293 ;
vmfp('c_Crops','a_FoodProc','USA') = 14812.5625 ;
vmfp('c_Crops','a_FoodProc','EU_28') = 58646.66797 ;
vmfp('c_Crops','a_FoodProc','CHN') = 33152.67578 ;
vmfp('c_Crops','a_FoodProc','JPN') = 11576.49219 ;
vmfp('c_Crops','a_FoodProc','IND') = 1720.338501 ;
vmfp('c_Crops','a_FoodProc','SSA') = 2550.623779 ;
vmfp('c_Crops','a_FoodProc','ROW') = 69284.14062 ;
vmfp('c_Crops','a_Energy','USA') = 3.463528156 ;
vmfp('c_Crops','a_Energy','EU_28') = 52.17811966 ;
vmfp('c_Crops','a_Energy','CHN') = 4.497493267 ;
vmfp('c_Crops','a_Energy','JPN') = 0.2920863926 ;
vmfp('c_Crops','a_Energy','IND') = 27.30981255 ;
vmfp('c_Crops','a_Energy','SSA') = 8.404819489 ;
vmfp('c_Crops','a_Energy','ROW') = 104.8999481 ;
vmfp('c_Crops','a_Textiles','USA') = 29.0988636 ;
vmfp('c_Crops','a_Textiles','EU_28') = 396.1600037 ;
vmfp('c_Crops','a_Textiles','CHN') = 3605.685059 ;
vmfp('c_Crops','a_Textiles','JPN') = 111.2606049 ;
vmfp('c_Crops','a_Textiles','IND') = 303.7763367 ;
vmfp('c_Crops','a_Textiles','SSA') = 41.00656509 ;
vmfp('c_Crops','a_Textiles','ROW') = 6455.765137 ;
vmfp('c_Crops','a_Chem','USA') = 825.6069946 ;
vmfp('c_Crops','a_Chem','EU_28') = 2411.231689 ;
vmfp('c_Crops','a_Chem','CHN') = 2051.366455 ;
vmfp('c_Crops','a_Chem','JPN') = 992.8882446 ;
vmfp('c_Crops','a_Chem','IND') = 176.8400726 ;
vmfp('c_Crops','a_Chem','SSA') = 33.55638885 ;
vmfp('c_Crops','a_Chem','ROW') = 3853.008301 ;
vmfp('c_Crops','a_Manuf','USA') = 193.725174 ;
vmfp('c_Crops','a_Manuf','EU_28') = 493.3613281 ;
vmfp('c_Crops','a_Manuf','CHN') = 1096.272827 ;
vmfp('c_Crops','a_Manuf','JPN') = 31.56473923 ;
vmfp('c_Crops','a_Manuf','IND') = 13.59422016 ;
vmfp('c_Crops','a_Manuf','SSA') = 61.74080658 ;
vmfp('c_Crops','a_Manuf','ROW') = 880.4550781 ;
vmfp('c_Crops','a_ForestFish','USA') = 115.3361969 ;
vmfp('c_Crops','a_ForestFish','EU_28') = 285.5862732 ;
vmfp('c_Crops','a_ForestFish','CHN') = 78.49636841 ;
vmfp('c_Crops','a_ForestFish','JPN') = 0.7752049565 ;
vmfp('c_Crops','a_ForestFish','IND') = 13.13344765 ;
vmfp('c_Crops','a_ForestFish','SSA') = 41.67062759 ;
vmfp('c_Crops','a_ForestFish','ROW') = 225.0957642 ;
vmfp('c_Crops','a_Svces','USA') = 2588.444824 ;
vmfp('c_Crops','a_Svces','EU_28') = 10561.96484 ;
vmfp('c_Crops','a_Svces','CHN') = 4135.079102 ;
vmfp('c_Crops','a_Svces','JPN') = 1944.253784 ;
vmfp('c_Crops','a_Svces','IND') = 894.7000732 ;
vmfp('c_Crops','a_Svces','SSA') = 446.4135742 ;
vmfp('c_Crops','a_Svces','ROW') = 19466.88086 ;
vmfp('c_Livestock','a_Rice','USA') = 1.111370683 ;
vmfp('c_Livestock','a_Rice','EU_28') = 12.99332047 ;
vmfp('c_Livestock','a_Rice','CHN') = 1.120004535 ;
vmfp('c_Livestock','a_Rice','JPN') = 2.228828192 ;
vmfp('c_Livestock','a_Rice','IND') = 4.125477314 ;
vmfp('c_Livestock','a_Rice','SSA') = 1.819263816 ;
vmfp('c_Livestock','a_Rice','ROW') = 109.2739716 ;
vmfp('c_Livestock','a_Crops','USA') = 9.039996147 ;
vmfp('c_Livestock','a_Crops','EU_28') = 184.744278 ;
vmfp('c_Livestock','a_Crops','CHN') = 1.621680737 ;
vmfp('c_Livestock','a_Crops','JPN') = 2.638736963 ;
vmfp('c_Livestock','a_Crops','IND') = 2.334111214 ;
vmfp('c_Livestock','a_Crops','SSA') = 2.064633608 ;
vmfp('c_Livestock','a_Crops','ROW') = 113.7304306 ;
vmfp('c_Livestock','a_Livestock','USA') = 634.6759033 ;
vmfp('c_Livestock','a_Livestock','EU_28') = 168.975296 ;
vmfp('c_Livestock','a_Livestock','CHN') = 605.625 ;
vmfp('c_Livestock','a_Livestock','JPN') = 72.08383942 ;
vmfp('c_Livestock','a_Livestock','IND') = 2.864137173 ;
vmfp('c_Livestock','a_Livestock','SSA') = 80.91104889 ;
vmfp('c_Livestock','a_Livestock','ROW') = 1677.749512 ;
vmfp('c_Livestock','a_FoodProc','USA') = 2973.019043 ;
vmfp('c_Livestock','a_FoodProc','EU_28') = 15228.54199 ;
vmfp('c_Livestock','a_FoodProc','CHN') = 3774.891846 ;
vmfp('c_Livestock','a_FoodProc','JPN') = 302.4428406 ;
vmfp('c_Livestock','a_FoodProc','IND') = 33.46454239 ;
vmfp('c_Livestock','a_FoodProc','SSA') = 409.2224426 ;
vmfp('c_Livestock','a_FoodProc','ROW') = 7487.866211 ;
vmfp('c_Livestock','a_Energy','USA') = 0.0185928531 ;
vmfp('c_Livestock','a_Energy','EU_28') = 19.38903999 ;
vmfp('c_Livestock','a_Energy','CHN') = 0.1226000413 ;
vmfp('c_Livestock','a_Energy','JPN') = 0.106916301 ;
vmfp('c_Livestock','a_Energy','IND') = 0.4456175268 ;
vmfp('c_Livestock','a_Energy','SSA') = 0.6495677829 ;
vmfp('c_Livestock','a_Energy','ROW') = 3.185010672 ;
vmfp('c_Livestock','a_Textiles','USA') = 1.213134527 ;
vmfp('c_Livestock','a_Textiles','EU_28') = 158.5678253 ;
vmfp('c_Livestock','a_Textiles','CHN') = 1203.922241 ;
vmfp('c_Livestock','a_Textiles','JPN') = 5.132497787 ;
vmfp('c_Livestock','a_Textiles','IND') = 24.01761627 ;
vmfp('c_Livestock','a_Textiles','SSA') = 25.00941658 ;
vmfp('c_Livestock','a_Textiles','ROW') = 1013.879395 ;
vmfp('c_Livestock','a_Chem','USA') = 2.858268738 ;
vmfp('c_Livestock','a_Chem','EU_28') = 780.8200684 ;
vmfp('c_Livestock','a_Chem','CHN') = 689.7246704 ;
vmfp('c_Livestock','a_Chem','JPN') = 3.670356035 ;
vmfp('c_Livestock','a_Chem','IND') = 51.07644272 ;
vmfp('c_Livestock','a_Chem','SSA') = 0.5134125948 ;
vmfp('c_Livestock','a_Chem','ROW') = 98.99484253 ;
vmfp('c_Livestock','a_Manuf','USA') = 1.134715319 ;
vmfp('c_Livestock','a_Manuf','EU_28') = 43.09702301 ;
vmfp('c_Livestock','a_Manuf','CHN') = 88.44561005 ;
vmfp('c_Livestock','a_Manuf','JPN') = 2.025376558 ;
vmfp('c_Livestock','a_Manuf','IND') = 0.7036259174 ;
vmfp('c_Livestock','a_Manuf','SSA') = 5.449497223 ;
vmfp('c_Livestock','a_Manuf','ROW') = 27.51062584 ;
vmfp('c_Livestock','a_ForestFish','USA') = 17.43488121 ;
vmfp('c_Livestock','a_ForestFish','EU_28') = 19.81300735 ;
vmfp('c_Livestock','a_ForestFish','CHN') = 0.1461976171 ;
vmfp('c_Livestock','a_ForestFish','JPN') = 0.9594054818 ;
vmfp('c_Livestock','a_ForestFish','IND') = 0.06490787119 ;
vmfp('c_Livestock','a_ForestFish','SSA') = 0.8455529213 ;
vmfp('c_Livestock','a_ForestFish','ROW') = 13.26441288 ;
vmfp('c_Livestock','a_Svces','USA') = 134.7550354 ;
vmfp('c_Livestock','a_Svces','EU_28') = 1601.501709 ;
vmfp('c_Livestock','a_Svces','CHN') = 407.2669067 ;
vmfp('c_Livestock','a_Svces','JPN') = 127.6872635 ;
vmfp('c_Livestock','a_Svces','IND') = 102.704895 ;
vmfp('c_Livestock','a_Svces','SSA') = 37.53453064 ;
vmfp('c_Livestock','a_Svces','ROW') = 2120.63623 ;
vmfp('c_FoodProc','a_Rice','USA') = 3618.05835 ;
vmfp('c_FoodProc','a_Rice','EU_28') = 87.29421997 ;
vmfp('c_FoodProc','a_Rice','CHN') = 14.92481804 ;
vmfp('c_FoodProc','a_Rice','JPN') = 2.191716671 ;
vmfp('c_FoodProc','a_Rice','IND') = 1.921524525 ;
vmfp('c_FoodProc','a_Rice','SSA') = 33.11755371 ;
vmfp('c_FoodProc','a_Rice','ROW') = 742.4446411 ;
vmfp('c_FoodProc','a_Crops','USA') = 22.02719307 ;
vmfp('c_FoodProc','a_Crops','EU_28') = 987.1462402 ;
vmfp('c_FoodProc','a_Crops','CHN') = 108.4203033 ;
vmfp('c_FoodProc','a_Crops','JPN') = 25.24214554 ;
vmfp('c_FoodProc','a_Crops','IND') = 2.805652857 ;
vmfp('c_FoodProc','a_Crops','SSA') = 1038.668579 ;
vmfp('c_FoodProc','a_Crops','ROW') = 3904.38208 ;
vmfp('c_FoodProc','a_Livestock','USA') = 873.9575195 ;
vmfp('c_FoodProc','a_Livestock','EU_28') = 6851.772461 ;
vmfp('c_FoodProc','a_Livestock','CHN') = 2371.253418 ;
vmfp('c_FoodProc','a_Livestock','JPN') = 1828.329468 ;
vmfp('c_FoodProc','a_Livestock','IND') = 914.4620972 ;
vmfp('c_FoodProc','a_Livestock','SSA') = 1106.516235 ;
vmfp('c_FoodProc','a_Livestock','ROW') = 12458.15723 ;
vmfp('c_FoodProc','a_FoodProc','USA') = 17389.06445 ;
vmfp('c_FoodProc','a_FoodProc','EU_28') = 90969.96875 ;
vmfp('c_FoodProc','a_FoodProc','CHN') = 18340.30859 ;
vmfp('c_FoodProc','a_FoodProc','JPN') = 11485.72949 ;
vmfp('c_FoodProc','a_FoodProc','IND') = 1591.76001 ;
vmfp('c_FoodProc','a_FoodProc','SSA') = 6793.228516 ;
vmfp('c_FoodProc','a_FoodProc','ROW') = 85042.36719 ;
vmfp('c_FoodProc','a_Energy','USA') = 13.56861305 ;
vmfp('c_FoodProc','a_Energy','EU_28') = 188.5522766 ;
vmfp('c_FoodProc','a_Energy','CHN') = 145.7058868 ;
vmfp('c_FoodProc','a_Energy','JPN') = 0.7323576808 ;
vmfp('c_FoodProc','a_Energy','IND') = 9.878072739 ;
vmfp('c_FoodProc','a_Energy','SSA') = 35.88212204 ;
vmfp('c_FoodProc','a_Energy','ROW') = 303.9833374 ;
vmfp('c_FoodProc','a_Textiles','USA') = 179.6339569 ;
vmfp('c_FoodProc','a_Textiles','EU_28') = 2227.112793 ;
vmfp('c_FoodProc','a_Textiles','CHN') = 1430.950317 ;
vmfp('c_FoodProc','a_Textiles','JPN') = 126.0708313 ;
vmfp('c_FoodProc','a_Textiles','IND') = 2.430003405 ;
vmfp('c_FoodProc','a_Textiles','SSA') = 88.31256866 ;
vmfp('c_FoodProc','a_Textiles','ROW') = 2124.773193 ;
vmfp('c_FoodProc','a_Chem','USA') = 352.9654846 ;
vmfp('c_FoodProc','a_Chem','EU_28') = 14722.47168 ;
vmfp('c_FoodProc','a_Chem','CHN') = 1973.306274 ;
vmfp('c_FoodProc','a_Chem','JPN') = 387.9205627 ;
vmfp('c_FoodProc','a_Chem','IND') = 184.9611664 ;
vmfp('c_FoodProc','a_Chem','SSA') = 103.6517715 ;
vmfp('c_FoodProc','a_Chem','ROW') = 2959.665527 ;
vmfp('c_FoodProc','a_Manuf','USA') = 268.2631836 ;
vmfp('c_FoodProc','a_Manuf','EU_28') = 1437.584839 ;
vmfp('c_FoodProc','a_Manuf','CHN') = 949.0641479 ;
vmfp('c_FoodProc','a_Manuf','JPN') = 80.71518707 ;
vmfp('c_FoodProc','a_Manuf','IND') = 2.067677021 ;
vmfp('c_FoodProc','a_Manuf','SSA') = 159.9487762 ;
vmfp('c_FoodProc','a_Manuf','ROW') = 943.9523315 ;
vmfp('c_FoodProc','a_ForestFish','USA') = 24.15935516 ;
vmfp('c_FoodProc','a_ForestFish','EU_28') = 290.3783875 ;
vmfp('c_FoodProc','a_ForestFish','CHN') = 874.24646 ;
vmfp('c_FoodProc','a_ForestFish','JPN') = 196.4720154 ;
vmfp('c_FoodProc','a_ForestFish','IND') = 2.3567276 ;
vmfp('c_FoodProc','a_ForestFish','SSA') = 44.6833992 ;
vmfp('c_FoodProc','a_ForestFish','ROW') = 4727.725098 ;
vmfp('c_FoodProc','a_Svces','USA') = 19488.64648 ;
vmfp('c_FoodProc','a_Svces','EU_28') = 64813.52344 ;
vmfp('c_FoodProc','a_Svces','CHN') = 9385.30957 ;
vmfp('c_FoodProc','a_Svces','JPN') = 15629 ;
vmfp('c_FoodProc','a_Svces','IND') = 488.9767761 ;
vmfp('c_FoodProc','a_Svces','SSA') = 5253.371582 ;
vmfp('c_FoodProc','a_Svces','ROW') = 70252.33594 ;
vmfp('c_Energy','a_Rice','USA') = 5.442742348 ;
vmfp('c_Energy','a_Rice','EU_28') = 73.6005249 ;
vmfp('c_Energy','a_Rice','CHN') = 114.0918884 ;
vmfp('c_Energy','a_Rice','JPN') = 38.48757172 ;
vmfp('c_Energy','a_Rice','IND') = 188.7977142 ;
vmfp('c_Energy','a_Rice','SSA') = 10.81722164 ;
vmfp('c_Energy','a_Rice','ROW') = 652.3049316 ;
vmfp('c_Energy','a_Crops','USA') = 693.3273926 ;
vmfp('c_Energy','a_Crops','EU_28') = 2921.927734 ;
vmfp('c_Energy','a_Crops','CHN') = 650.1383057 ;
vmfp('c_Energy','a_Crops','JPN') = 148.5122528 ;
vmfp('c_Energy','a_Crops','IND') = 299.7307434 ;
vmfp('c_Energy','a_Crops','SSA') = 332.2598267 ;
vmfp('c_Energy','a_Crops','ROW') = 4573.530273 ;
vmfp('c_Energy','a_Livestock','USA') = 432.1714478 ;
vmfp('c_Energy','a_Livestock','EU_28') = 774.6335449 ;
vmfp('c_Energy','a_Livestock','CHN') = 155.9545135 ;
vmfp('c_Energy','a_Livestock','JPN') = 18.87781525 ;
vmfp('c_Energy','a_Livestock','IND') = 17.43144989 ;
vmfp('c_Energy','a_Livestock','SSA') = 113.2649536 ;
vmfp('c_Energy','a_Livestock','ROW') = 943.1743774 ;
vmfp('c_Energy','a_FoodProc','USA') = 196.3681946 ;
vmfp('c_Energy','a_FoodProc','EU_28') = 5528.403809 ;
vmfp('c_Energy','a_FoodProc','CHN') = 810.3313599 ;
vmfp('c_Energy','a_FoodProc','JPN') = 931.6809692 ;
vmfp('c_Energy','a_FoodProc','IND') = 282.4660034 ;
vmfp('c_Energy','a_FoodProc','SSA') = 323.7913208 ;
vmfp('c_Energy','a_FoodProc','ROW') = 3900.230469 ;
vmfp('c_Energy','a_Energy','USA') = 157361.5938 ;
vmfp('c_Energy','a_Energy','EU_28') = 315042.75 ;
vmfp('c_Energy','a_Energy','CHN') = 192800.5469 ;
vmfp('c_Energy','a_Energy','JPN') = 120376.0625 ;
vmfp('c_Energy','a_Energy','IND') = 107046.9922 ;
vmfp('c_Energy','a_Energy','SSA') = 14201.7666 ;
vmfp('c_Energy','a_Energy','ROW') = 370898.4688 ;
vmfp('c_Energy','a_Textiles','USA') = 31.26780891 ;
vmfp('c_Energy','a_Textiles','EU_28') = 946.1473389 ;
vmfp('c_Energy','a_Textiles','CHN') = 672.1812134 ;
vmfp('c_Energy','a_Textiles','JPN') = 104.4559555 ;
vmfp('c_Energy','a_Textiles','IND') = 114.4711151 ;
vmfp('c_Energy','a_Textiles','SSA') = 78.41272736 ;
vmfp('c_Energy','a_Textiles','ROW') = 1468.815186 ;
vmfp('c_Energy','a_Chem','USA') = 9088.836914 ;
vmfp('c_Energy','a_Chem','EU_28') = 40256.4375 ;
vmfp('c_Energy','a_Chem','CHN') = 10009.14258 ;
vmfp('c_Energy','a_Chem','JPN') = 6613.341309 ;
vmfp('c_Energy','a_Chem','IND') = 5133.466309 ;
vmfp('c_Energy','a_Chem','SSA') = 1015.010071 ;
vmfp('c_Energy','a_Chem','ROW') = 33515.28125 ;
vmfp('c_Energy','a_Manuf','USA') = 1595.953735 ;
vmfp('c_Energy','a_Manuf','EU_28') = 27789.4375 ;
vmfp('c_Energy','a_Manuf','CHN') = 16231.74316 ;
vmfp('c_Energy','a_Manuf','JPN') = 6732.148438 ;
vmfp('c_Energy','a_Manuf','IND') = 6010.062012 ;
vmfp('c_Energy','a_Manuf','SSA') = 3401.147705 ;
vmfp('c_Energy','a_Manuf','ROW') = 36008.1875 ;
vmfp('c_Energy','a_ForestFish','USA') = 166.0809174 ;
vmfp('c_Energy','a_ForestFish','EU_28') = 987.3132935 ;
vmfp('c_Energy','a_ForestFish','CHN') = 552.8024292 ;
vmfp('c_Energy','a_ForestFish','JPN') = 274.1964417 ;
vmfp('c_Energy','a_ForestFish','IND') = 88.44503784 ;
vmfp('c_Energy','a_ForestFish','SSA') = 391.4177246 ;
vmfp('c_Energy','a_ForestFish','ROW') = 2497.782959 ;
vmfp('c_Energy','a_Svces','USA') = 43398.08984 ;
vmfp('c_Energy','a_Svces','EU_28') = 164243.9375 ;
vmfp('c_Energy','a_Svces','CHN') = 13308.48926 ;
vmfp('c_Energy','a_Svces','JPN') = 10382.25391 ;
vmfp('c_Energy','a_Svces','IND') = 5738.475098 ;
vmfp('c_Energy','a_Svces','SSA') = 22903.19336 ;
vmfp('c_Energy','a_Svces','ROW') = 157956.875 ;
vmfp('c_Textiles','a_Rice','USA') = 7.299894333 ;
vmfp('c_Textiles','a_Rice','EU_28') = 14.56050205 ;
vmfp('c_Textiles','a_Rice','CHN') = 19.96034431 ;
vmfp('c_Textiles','a_Rice','JPN') = 49.87926102 ;
vmfp('c_Textiles','a_Rice','IND') = 2.353225708 ;
vmfp('c_Textiles','a_Rice','SSA') = 18.2704792 ;
vmfp('c_Textiles','a_Rice','ROW') = 252.3002625 ;
vmfp('c_Textiles','a_Crops','USA') = 91.95333099 ;
vmfp('c_Textiles','a_Crops','EU_28') = 375.1776428 ;
vmfp('c_Textiles','a_Crops','CHN') = 7.519579411 ;
vmfp('c_Textiles','a_Crops','JPN') = 79.99256897 ;
vmfp('c_Textiles','a_Crops','IND') = 0.6632803082 ;
vmfp('c_Textiles','a_Crops','SSA') = 546.7007446 ;
vmfp('c_Textiles','a_Crops','ROW') = 1213.842529 ;
vmfp('c_Textiles','a_Livestock','USA') = 29.60349083 ;
vmfp('c_Textiles','a_Livestock','EU_28') = 173.519928 ;
vmfp('c_Textiles','a_Livestock','CHN') = 1.662313938 ;
vmfp('c_Textiles','a_Livestock','JPN') = 14.81058121 ;
vmfp('c_Textiles','a_Livestock','IND') = 1.79659009 ;
vmfp('c_Textiles','a_Livestock','SSA') = 106.3433304 ;
vmfp('c_Textiles','a_Livestock','ROW') = 266.5496826 ;
vmfp('c_Textiles','a_FoodProc','USA') = 169.2012329 ;
vmfp('c_Textiles','a_FoodProc','EU_28') = 1487.532837 ;
vmfp('c_Textiles','a_FoodProc','CHN') = 303.4482422 ;
vmfp('c_Textiles','a_FoodProc','JPN') = 239.3979645 ;
vmfp('c_Textiles','a_FoodProc','IND') = 9.285899162 ;
vmfp('c_Textiles','a_FoodProc','SSA') = 265.2901306 ;
vmfp('c_Textiles','a_FoodProc','ROW') = 2045.940674 ;
vmfp('c_Textiles','a_Energy','USA') = 66.76277161 ;
vmfp('c_Textiles','a_Energy','EU_28') = 320.7935486 ;
vmfp('c_Textiles','a_Energy','CHN') = 133.9307709 ;
vmfp('c_Textiles','a_Energy','JPN') = 42.54476547 ;
vmfp('c_Textiles','a_Energy','IND') = 118.2506027 ;
vmfp('c_Textiles','a_Energy','SSA') = 455.3795471 ;
vmfp('c_Textiles','a_Energy','ROW') = 1502.978516 ;
vmfp('c_Textiles','a_Textiles','USA') = 6603.450195 ;
vmfp('c_Textiles','a_Textiles','EU_28') = 75856.32031 ;
vmfp('c_Textiles','a_Textiles','CHN') = 32617.99609 ;
vmfp('c_Textiles','a_Textiles','JPN') = 6619.635742 ;
vmfp('c_Textiles','a_Textiles','IND') = 2445.539307 ;
vmfp('c_Textiles','a_Textiles','SSA') = 4270.282715 ;
vmfp('c_Textiles','a_Textiles','ROW') = 110716.5547 ;
vmfp('c_Textiles','a_Chem','USA') = 1495.103882 ;
vmfp('c_Textiles','a_Chem','EU_28') = 10623.31543 ;
vmfp('c_Textiles','a_Chem','CHN') = 885.6781006 ;
vmfp('c_Textiles','a_Chem','JPN') = 643.4073486 ;
vmfp('c_Textiles','a_Chem','IND') = 373.9119263 ;
vmfp('c_Textiles','a_Chem','SSA') = 168.3428497 ;
vmfp('c_Textiles','a_Chem','ROW') = 4759.670898 ;
vmfp('c_Textiles','a_Manuf','USA') = 7684.530762 ;
vmfp('c_Textiles','a_Manuf','EU_28') = 20371.73438 ;
vmfp('c_Textiles','a_Manuf','CHN') = 3782.06543 ;
vmfp('c_Textiles','a_Manuf','JPN') = 3463.822021 ;
vmfp('c_Textiles','a_Manuf','IND') = 121.8386383 ;
vmfp('c_Textiles','a_Manuf','SSA') = 838.5788574 ;
vmfp('c_Textiles','a_Manuf','ROW') = 18031.79688 ;
vmfp('c_Textiles','a_ForestFish','USA') = 59.38301468 ;
vmfp('c_Textiles','a_ForestFish','EU_28') = 457.5525513 ;
vmfp('c_Textiles','a_ForestFish','CHN') = 70.15359497 ;
vmfp('c_Textiles','a_ForestFish','JPN') = 180.8363953 ;
vmfp('c_Textiles','a_ForestFish','IND') = 17.76542664 ;
vmfp('c_Textiles','a_ForestFish','SSA') = 395.6745605 ;
vmfp('c_Textiles','a_ForestFish','ROW') = 951.4020386 ;
vmfp('c_Textiles','a_Svces','USA') = 18820.2168 ;
vmfp('c_Textiles','a_Svces','EU_28') = 34567.32422 ;
vmfp('c_Textiles','a_Svces','CHN') = 3876.053467 ;
vmfp('c_Textiles','a_Svces','JPN') = 11041.27637 ;
vmfp('c_Textiles','a_Svces','IND') = 545.9870605 ;
vmfp('c_Textiles','a_Svces','SSA') = 3737.508545 ;
vmfp('c_Textiles','a_Svces','ROW') = 36531.53125 ;
vmfp('c_Chem','a_Rice','USA') = 73.62728882 ;
vmfp('c_Chem','a_Rice','EU_28') = 160.3995667 ;
vmfp('c_Chem','a_Rice','CHN') = 2294.986328 ;
vmfp('c_Chem','a_Rice','JPN') = 335.8095703 ;
vmfp('c_Chem','a_Rice','IND') = 919.4692383 ;
vmfp('c_Chem','a_Rice','SSA') = 225.4079437 ;
vmfp('c_Chem','a_Rice','ROW') = 6274.639648 ;
vmfp('c_Chem','a_Crops','USA') = 5384.292969 ;
vmfp('c_Chem','a_Crops','EU_28') = 7721.614746 ;
vmfp('c_Chem','a_Crops','CHN') = 14832.28027 ;
vmfp('c_Chem','a_Crops','JPN') = 706.4315186 ;
vmfp('c_Chem','a_Crops','IND') = 313.5353699 ;
vmfp('c_Chem','a_Crops','SSA') = 5562.153809 ;
vmfp('c_Chem','a_Crops','ROW') = 29030.00781 ;
vmfp('c_Chem','a_Livestock','USA') = 513.0168457 ;
vmfp('c_Chem','a_Livestock','EU_28') = 3647.864014 ;
vmfp('c_Chem','a_Livestock','CHN') = 175.5305481 ;
vmfp('c_Chem','a_Livestock','JPN') = 150.1275177 ;
vmfp('c_Chem','a_Livestock','IND') = 18.05657196 ;
vmfp('c_Chem','a_Livestock','SSA') = 532.333313 ;
vmfp('c_Chem','a_Livestock','ROW') = 4901.002441 ;
vmfp('c_Chem','a_FoodProc','USA') = 6702.990234 ;
vmfp('c_Chem','a_FoodProc','EU_28') = 25377.31055 ;
vmfp('c_Chem','a_FoodProc','CHN') = 7951.209473 ;
vmfp('c_Chem','a_FoodProc','JPN') = 1728.370239 ;
vmfp('c_Chem','a_FoodProc','IND') = 1098.258667 ;
vmfp('c_Chem','a_FoodProc','SSA') = 1834.676392 ;
vmfp('c_Chem','a_FoodProc','ROW') = 28619.50195 ;
vmfp('c_Chem','a_Energy','USA') = 2768.193848 ;
vmfp('c_Chem','a_Energy','EU_28') = 7185.25 ;
vmfp('c_Chem','a_Energy','CHN') = 3142.799805 ;
vmfp('c_Chem','a_Energy','JPN') = 236.8647156 ;
vmfp('c_Chem','a_Energy','IND') = 4539.90918 ;
vmfp('c_Chem','a_Energy','SSA') = 1411.25647 ;
vmfp('c_Chem','a_Energy','ROW') = 15436.46777 ;
vmfp('c_Chem','a_Textiles','USA') = 2795.101807 ;
vmfp('c_Chem','a_Textiles','EU_28') = 17754.23828 ;
vmfp('c_Chem','a_Textiles','CHN') = 14602.68457 ;
vmfp('c_Chem','a_Textiles','JPN') = 903.1824951 ;
vmfp('c_Chem','a_Textiles','IND') = 4454.385742 ;
vmfp('c_Chem','a_Textiles','SSA') = 710.6029663 ;
vmfp('c_Chem','a_Textiles','ROW') = 19947.16797 ;
vmfp('c_Chem','a_Chem','USA') = 59066.78125 ;
vmfp('c_Chem','a_Chem','EU_28') = 288365.6875 ;
vmfp('c_Chem','a_Chem','CHN') = 110698.0781 ;
vmfp('c_Chem','a_Chem','JPN') = 30619.70117 ;
vmfp('c_Chem','a_Chem','IND') = 25307.76758 ;
vmfp('c_Chem','a_Chem','SSA') = 5589.47168 ;
vmfp('c_Chem','a_Chem','ROW') = 252176.0938 ;
vmfp('c_Chem','a_Manuf','USA') = 30087.16211 ;
vmfp('c_Chem','a_Manuf','EU_28') = 111288.3047 ;
vmfp('c_Chem','a_Manuf','CHN') = 63231.49609 ;
vmfp('c_Chem','a_Manuf','JPN') = 11940.41406 ;
vmfp('c_Chem','a_Manuf','IND') = 6066.159668 ;
vmfp('c_Chem','a_Manuf','SSA') = 4941.613281 ;
vmfp('c_Chem','a_Manuf','ROW') = 132865.0156 ;
vmfp('c_Chem','a_ForestFish','USA') = 956.3396606 ;
vmfp('c_Chem','a_ForestFish','EU_28') = 893.0582275 ;
vmfp('c_Chem','a_ForestFish','CHN') = 652.2442627 ;
vmfp('c_Chem','a_ForestFish','JPN') = 74.63299561 ;
vmfp('c_Chem','a_ForestFish','IND') = 16.86178589 ;
vmfp('c_Chem','a_ForestFish','SSA') = 223.1220245 ;
vmfp('c_Chem','a_ForestFish','ROW') = 1716.37439 ;
vmfp('c_Chem','a_Svces','USA') = 65880.4375 ;
vmfp('c_Chem','a_Svces','EU_28') = 221414.6562 ;
vmfp('c_Chem','a_Svces','CHN') = 63539.07422 ;
vmfp('c_Chem','a_Svces','JPN') = 30261.49023 ;
vmfp('c_Chem','a_Svces','IND') = 9972.219727 ;
vmfp('c_Chem','a_Svces','SSA') = 10762.49902 ;
vmfp('c_Chem','a_Svces','ROW') = 167546.6719 ;
vmfp('c_Manuf','a_Rice','USA') = 38.16287231 ;
vmfp('c_Manuf','a_Rice','EU_28') = 136.9540863 ;
vmfp('c_Manuf','a_Rice','CHN') = 193.4227142 ;
vmfp('c_Manuf','a_Rice','JPN') = 20.31934738 ;
vmfp('c_Manuf','a_Rice','IND') = 37.72236252 ;
vmfp('c_Manuf','a_Rice','SSA') = 120.7360764 ;
vmfp('c_Manuf','a_Rice','ROW') = 1501.477173 ;
vmfp('c_Manuf','a_Crops','USA') = 1701.692139 ;
vmfp('c_Manuf','a_Crops','EU_28') = 2906.25415 ;
vmfp('c_Manuf','a_Crops','CHN') = 1065.587036 ;
vmfp('c_Manuf','a_Crops','JPN') = 84.30924988 ;
vmfp('c_Manuf','a_Crops','IND') = 19.86198425 ;
vmfp('c_Manuf','a_Crops','SSA') = 2868.952881 ;
vmfp('c_Manuf','a_Crops','ROW') = 7684.980957 ;
vmfp('c_Manuf','a_Livestock','USA') = 1100.210938 ;
vmfp('c_Manuf','a_Livestock','EU_28') = 1574.076416 ;
vmfp('c_Manuf','a_Livestock','CHN') = 302.9829102 ;
vmfp('c_Manuf','a_Livestock','JPN') = 44.2393837 ;
vmfp('c_Manuf','a_Livestock','IND') = 107.7020493 ;
vmfp('c_Manuf','a_Livestock','SSA') = 461.3960876 ;
vmfp('c_Manuf','a_Livestock','ROW') = 4511.624023 ;
vmfp('c_Manuf','a_FoodProc','USA') = 12506.62012 ;
vmfp('c_Manuf','a_FoodProc','EU_28') = 23642.98828 ;
vmfp('c_Manuf','a_FoodProc','CHN') = 3964.980469 ;
vmfp('c_Manuf','a_FoodProc','JPN') = 1094.233887 ;
vmfp('c_Manuf','a_FoodProc','IND') = 377.2027588 ;
vmfp('c_Manuf','a_FoodProc','SSA') = 3632.647217 ;
vmfp('c_Manuf','a_FoodProc','ROW') = 25849.19531 ;
vmfp('c_Manuf','a_Energy','USA') = 7060.480469 ;
vmfp('c_Manuf','a_Energy','EU_28') = 18282.75781 ;
vmfp('c_Manuf','a_Energy','CHN') = 8729.325195 ;
vmfp('c_Manuf','a_Energy','JPN') = 714.4686279 ;
vmfp('c_Manuf','a_Energy','IND') = 4199.499512 ;
vmfp('c_Manuf','a_Energy','SSA') = 5501.789551 ;
vmfp('c_Manuf','a_Energy','ROW') = 73436.22656 ;
vmfp('c_Manuf','a_Textiles','USA') = 1089.393677 ;
vmfp('c_Manuf','a_Textiles','EU_28') = 10012.94141 ;
vmfp('c_Manuf','a_Textiles','CHN') = 3747.616943 ;
vmfp('c_Manuf','a_Textiles','JPN') = 172.5587311 ;
vmfp('c_Manuf','a_Textiles','IND') = 689.7332764 ;
vmfp('c_Manuf','a_Textiles','SSA') = 742.6376343 ;
vmfp('c_Manuf','a_Textiles','ROW') = 12747.65137 ;
vmfp('c_Manuf','a_Chem','USA') = 15766.80664 ;
vmfp('c_Manuf','a_Chem','EU_28') = 44470.5 ;
vmfp('c_Manuf','a_Chem','CHN') = 20505.14844 ;
vmfp('c_Manuf','a_Chem','JPN') = 1823.536377 ;
vmfp('c_Manuf','a_Chem','IND') = 4297.597656 ;
vmfp('c_Manuf','a_Chem','SSA') = 1230.11377 ;
vmfp('c_Manuf','a_Chem','ROW') = 34742.4375 ;
vmfp('c_Manuf','a_Manuf','USA') = 403418.9375 ;
vmfp('c_Manuf','a_Manuf','EU_28') = 1319997.5 ;
vmfp('c_Manuf','a_Manuf','CHN') = 754238.375 ;
vmfp('c_Manuf','a_Manuf','JPN') = 160904.75 ;
vmfp('c_Manuf','a_Manuf','IND') = 99641.57812 ;
vmfp('c_Manuf','a_Manuf','SSA') = 35234.69531 ;
vmfp('c_Manuf','a_Manuf','ROW') = 1341635.125 ;
vmfp('c_Manuf','a_ForestFish','USA') = 368.1846619 ;
vmfp('c_Manuf','a_ForestFish','EU_28') = 2073.312988 ;
vmfp('c_Manuf','a_ForestFish','CHN') = 844.0291138 ;
vmfp('c_Manuf','a_ForestFish','JPN') = 168.2225037 ;
vmfp('c_Manuf','a_ForestFish','IND') = 619.2880249 ;
vmfp('c_Manuf','a_ForestFish','SSA') = 1166.206299 ;
vmfp('c_Manuf','a_ForestFish','ROW') = 3960.928711 ;
vmfp('c_Manuf','a_Svces','USA') = 253281.2812 ;
vmfp('c_Manuf','a_Svces','EU_28') = 508811.625 ;
vmfp('c_Manuf','a_Svces','CHN') = 207404.9688 ;
vmfp('c_Manuf','a_Svces','JPN') = 55344.16797 ;
vmfp('c_Manuf','a_Svces','IND') = 38578.56641 ;
vmfp('c_Manuf','a_Svces','SSA') = 44693.42578 ;
vmfp('c_Manuf','a_Svces','ROW') = 631463.3125 ;
vmfp('c_ForestFish','a_Rice','USA') = 0.05433415994 ;
vmfp('c_ForestFish','a_Rice','EU_28') = 5.744525909 ;
vmfp('c_ForestFish','a_Rice','CHN') = 0.2450711429 ;
vmfp('c_ForestFish','a_Rice','JPN') = 0.04287629575 ;
vmfp('c_ForestFish','a_Rice','IND') = 0.01898382232 ;
vmfp('c_ForestFish','a_Rice','SSA') = 0.08248731494 ;
vmfp('c_ForestFish','a_Rice','ROW') = 11.41436958 ;
vmfp('c_ForestFish','a_Crops','USA') = 4.237986088 ;
vmfp('c_ForestFish','a_Crops','EU_28') = 17.82388878 ;
vmfp('c_ForestFish','a_Crops','CHN') = 1.293670297 ;
vmfp('c_ForestFish','a_Crops','JPN') = 1.878968477 ;
vmfp('c_ForestFish','a_Crops','IND') = 0.06491453201 ;
vmfp('c_ForestFish','a_Crops','SSA') = 0.527112186 ;
vmfp('c_ForestFish','a_Crops','ROW') = 20.40672493 ;
vmfp('c_ForestFish','a_Livestock','USA') = 0.7206557393 ;
vmfp('c_ForestFish','a_Livestock','EU_28') = 9.928297997 ;
vmfp('c_ForestFish','a_Livestock','CHN') = 0.1809564829 ;
vmfp('c_ForestFish','a_Livestock','JPN') = 0.03739479557 ;
vmfp('c_ForestFish','a_Livestock','IND') = 2.156460524 ;
vmfp('c_ForestFish','a_Livestock','SSA') = 1.784724593 ;
vmfp('c_ForestFish','a_Livestock','ROW') = 14.05985451 ;
vmfp('c_ForestFish','a_FoodProc','USA') = 1202.344849 ;
vmfp('c_ForestFish','a_FoodProc','EU_28') = 5109.38623 ;
vmfp('c_ForestFish','a_FoodProc','CHN') = 930.1137695 ;
vmfp('c_ForestFish','a_FoodProc','JPN') = 1110.934814 ;
vmfp('c_ForestFish','a_FoodProc','IND') = 66.17717743 ;
vmfp('c_ForestFish','a_FoodProc','SSA') = 39.44313049 ;
vmfp('c_ForestFish','a_FoodProc','ROW') = 1356.977417 ;
vmfp('c_ForestFish','a_Energy','USA') = 1.282049417 ;
vmfp('c_ForestFish','a_Energy','EU_28') = 38.06052399 ;
vmfp('c_ForestFish','a_Energy','CHN') = 31.81116104 ;
vmfp('c_ForestFish','a_Energy','JPN') = 0.06488320976 ;
vmfp('c_ForestFish','a_Energy','IND') = 0.4973107278 ;
vmfp('c_ForestFish','a_Energy','SSA') = 0.4934261143 ;
vmfp('c_ForestFish','a_Energy','ROW') = 5.659125805 ;
vmfp('c_ForestFish','a_Textiles','USA') = 3.292218208 ;
vmfp('c_ForestFish','a_Textiles','EU_28') = 114.7744904 ;
vmfp('c_ForestFish','a_Textiles','CHN') = 59.88498688 ;
vmfp('c_ForestFish','a_Textiles','JPN') = 1.72721839 ;
vmfp('c_ForestFish','a_Textiles','IND') = 1.635375857 ;
vmfp('c_ForestFish','a_Textiles','SSA') = 0.1435197741 ;
vmfp('c_ForestFish','a_Textiles','ROW') = 5.168981075 ;
vmfp('c_ForestFish','a_Chem','USA') = 66.25779724 ;
vmfp('c_ForestFish','a_Chem','EU_28') = 381.9627686 ;
vmfp('c_ForestFish','a_Chem','CHN') = 2009.212402 ;
vmfp('c_ForestFish','a_Chem','JPN') = 22.01392174 ;
vmfp('c_ForestFish','a_Chem','IND') = 120.3005524 ;
vmfp('c_ForestFish','a_Chem','SSA') = 6.039269924 ;
vmfp('c_ForestFish','a_Chem','ROW') = 188.0586548 ;
vmfp('c_ForestFish','a_Manuf','USA') = 209.4268341 ;
vmfp('c_ForestFish','a_Manuf','EU_28') = 3809.336426 ;
vmfp('c_ForestFish','a_Manuf','CHN') = 5872.794922 ;
vmfp('c_ForestFish','a_Manuf','JPN') = 683.5963135 ;
vmfp('c_ForestFish','a_Manuf','IND') = 100.7577591 ;
vmfp('c_ForestFish','a_Manuf','SSA') = 80.93427277 ;
vmfp('c_ForestFish','a_Manuf','ROW') = 1565.540039 ;
vmfp('c_ForestFish','a_ForestFish','USA') = 36.12144089 ;
vmfp('c_ForestFish','a_ForestFish','EU_28') = 916.9125977 ;
vmfp('c_ForestFish','a_ForestFish','CHN') = 618.0199585 ;
vmfp('c_ForestFish','a_ForestFish','JPN') = 192.5113373 ;
vmfp('c_ForestFish','a_ForestFish','IND') = 5.473010063 ;
vmfp('c_ForestFish','a_ForestFish','SSA') = 24.36708641 ;
vmfp('c_ForestFish','a_ForestFish','ROW') = 1237.951782 ;
vmfp('c_ForestFish','a_Svces','USA') = 900.0993042 ;
vmfp('c_ForestFish','a_Svces','EU_28') = 2868.251953 ;
vmfp('c_ForestFish','a_Svces','CHN') = 3278.414551 ;
vmfp('c_ForestFish','a_Svces','JPN') = 484.3358765 ;
vmfp('c_ForestFish','a_Svces','IND') = 680.519043 ;
vmfp('c_ForestFish','a_Svces','SSA') = 20.85749626 ;
vmfp('c_ForestFish','a_Svces','ROW') = 1577.875244 ;
vmfp('c_Svces','a_Rice','USA') = 26.41955566 ;
vmfp('c_Svces','a_Rice','EU_28') = 152.304657 ;
vmfp('c_Svces','a_Rice','CHN') = 256.3632202 ;
vmfp('c_Svces','a_Rice','JPN') = 98.19528198 ;
vmfp('c_Svces','a_Rice','IND') = 138.2388458 ;
vmfp('c_Svces','a_Rice','SSA') = 57.31511307 ;
vmfp('c_Svces','a_Rice','ROW') = 1089.643799 ;
vmfp('c_Svces','a_Crops','USA') = 791.3149414 ;
vmfp('c_Svces','a_Crops','EU_28') = 4315.253418 ;
vmfp('c_Svces','a_Crops','CHN') = 1087.785156 ;
vmfp('c_Svces','a_Crops','JPN') = 114.4099274 ;
vmfp('c_Svces','a_Crops','IND') = 85.5951767 ;
vmfp('c_Svces','a_Crops','SSA') = 1454.045166 ;
vmfp('c_Svces','a_Crops','ROW') = 4855.841797 ;
vmfp('c_Svces','a_Livestock','USA') = 559.6428223 ;
vmfp('c_Svces','a_Livestock','EU_28') = 2068.210938 ;
vmfp('c_Svces','a_Livestock','CHN') = 429.4632263 ;
vmfp('c_Svces','a_Livestock','JPN') = 119.3636627 ;
vmfp('c_Svces','a_Livestock','IND') = 121.458107 ;
vmfp('c_Svces','a_Livestock','SSA') = 372.6543884 ;
vmfp('c_Svces','a_Livestock','ROW') = 2082.968506 ;
vmfp('c_Svces','a_FoodProc','USA') = 2739.516602 ;
vmfp('c_Svces','a_FoodProc','EU_28') = 38270.23828 ;
vmfp('c_Svces','a_FoodProc','CHN') = 5270.861816 ;
vmfp('c_Svces','a_FoodProc','JPN') = 1331.228882 ;
vmfp('c_Svces','a_FoodProc','IND') = 550.6959229 ;
vmfp('c_Svces','a_FoodProc','SSA') = 2012.40625 ;
vmfp('c_Svces','a_FoodProc','ROW') = 16258.39355 ;
vmfp('c_Svces','a_Energy','USA') = 2504.241699 ;
vmfp('c_Svces','a_Energy','EU_28') = 9605.416016 ;
vmfp('c_Svces','a_Energy','CHN') = 3562.682129 ;
vmfp('c_Svces','a_Energy','JPN') = 2507.097412 ;
vmfp('c_Svces','a_Energy','IND') = 3762.350586 ;
vmfp('c_Svces','a_Energy','SSA') = 5930.787109 ;
vmfp('c_Svces','a_Energy','ROW') = 35100.35547 ;
vmfp('c_Svces','a_Textiles','USA') = 346.1118469 ;
vmfp('c_Svces','a_Textiles','EU_28') = 10271.54199 ;
vmfp('c_Svces','a_Textiles','CHN') = 5236.101074 ;
vmfp('c_Svces','a_Textiles','JPN') = 196.1745911 ;
vmfp('c_Svces','a_Textiles','IND') = 592.6686401 ;
vmfp('c_Svces','a_Textiles','SSA') = 614.8399658 ;
vmfp('c_Svces','a_Textiles','ROW') = 6970.617188 ;
vmfp('c_Svces','a_Chem','USA') = 3573.27832 ;
vmfp('c_Svces','a_Chem','EU_28') = 63116.82422 ;
vmfp('c_Svces','a_Chem','CHN') = 8171.244629 ;
vmfp('c_Svces','a_Chem','JPN') = 2177.811279 ;
vmfp('c_Svces','a_Chem','IND') = 1262.138916 ;
vmfp('c_Svces','a_Chem','SSA') = 800.7359009 ;
vmfp('c_Svces','a_Chem','ROW') = 26512.58203 ;
vmfp('c_Svces','a_Manuf','USA') = 17782.99023 ;
vmfp('c_Svces','a_Manuf','EU_28') = 168193.9062 ;
vmfp('c_Svces','a_Manuf','CHN') = 36543.1875 ;
vmfp('c_Svces','a_Manuf','JPN') = 10430.24219 ;
vmfp('c_Svces','a_Manuf','IND') = 5357.269043 ;
vmfp('c_Svces','a_Manuf','SSA') = 5348.523926 ;
vmfp('c_Svces','a_Manuf','ROW') = 79730.10938 ;
vmfp('c_Svces','a_ForestFish','USA') = 233.1974487 ;
vmfp('c_Svces','a_ForestFish','EU_28') = 1637.891724 ;
vmfp('c_Svces','a_ForestFish','CHN') = 1036.87439 ;
vmfp('c_Svces','a_ForestFish','JPN') = 82.33956146 ;
vmfp('c_Svces','a_ForestFish','IND') = 352.3551941 ;
vmfp('c_Svces','a_ForestFish','SSA') = 930.7883301 ;
vmfp('c_Svces','a_ForestFish','ROW') = 2685.142578 ;
vmfp('c_Svces','a_Svces','USA') = 224986.5938 ;
vmfp('c_Svces','a_Svces','EU_28') = 1214330.25 ;
vmfp('c_Svces','a_Svces','CHN') = 136314.9219 ;
vmfp('c_Svces','a_Svces','JPN') = 64284.63281 ;
vmfp('c_Svces','a_Svces','IND') = 23725.51562 ;
vmfp('c_Svces','a_Svces','SSA') = 31571.71484 ;
vmfp('c_Svces','a_Svces','ROW') = 656804.1875 ;

* vdpb data (70 cells)
vdpb('c_Rice','USA') = 4570.596191 ;
vdpb('c_Rice','EU_28') = 2580.30835 ;
vdpb('c_Rice','CHN') = 53829.125 ;
vdpb('c_Rice','JPN') = 12246.43262 ;
vdpb('c_Rice','IND') = 14031.07715 ;
vdpb('c_Rice','SSA') = 10944.77246 ;
vdpb('c_Rice','ROW') = 164793.5469 ;
vdpb('c_Crops','USA') = 45801.00391 ;
vdpb('c_Crops','EU_28') = 63108.25 ;
vdpb('c_Crops','CHN') = 191434.9062 ;
vdpb('c_Crops','JPN') = 19088.62109 ;
vdpb('c_Crops','IND') = 132760.6562 ;
vdpb('c_Crops','SSA') = 134715.8594 ;
vdpb('c_Crops','ROW') = 251105.0781 ;
vdpb('c_Livestock','USA') = 14565.50586 ;
vdpb('c_Livestock','EU_28') = 24576.38672 ;
vdpb('c_Livestock','CHN') = 114475.6484 ;
vdpb('c_Livestock','JPN') = 6550.102051 ;
vdpb('c_Livestock','IND') = 57831.71875 ;
vdpb('c_Livestock','SSA') = 28199.44531 ;
vdpb('c_Livestock','ROW') = 128444.2422 ;
vdpb('c_FoodProc','USA') = 633797.9375 ;
vdpb('c_FoodProc','EU_28') = 533944.8125 ;
vdpb('c_FoodProc','CHN') = 752912.0625 ;
vdpb('c_FoodProc','JPN') = 200819.4688 ;
vdpb('c_FoodProc','IND') = 160763.2812 ;
vdpb('c_FoodProc','SSA') = 208114.6406 ;
vdpb('c_FoodProc','ROW') = 1200112.25 ;
vdpb('c_Energy','USA') = 239341.6406 ;
vdpb('c_Energy','EU_28') = 184299.9531 ;
vdpb('c_Energy','CHN') = 153850.4062 ;
vdpb('c_Energy','JPN') = 71551.89062 ;
vdpb('c_Energy','IND') = 90954.1875 ;
vdpb('c_Energy','SSA') = 24006.89258 ;
vdpb('c_Energy','ROW') = 475345.9375 ;
vdpb('c_Textiles','USA') = 30215.48047 ;
vdpb('c_Textiles','EU_28') = 60681.63672 ;
vdpb('c_Textiles','CHN') = 189531.9531 ;
vdpb('c_Textiles','JPN') = 15577.27539 ;
vdpb('c_Textiles','IND') = 86027.14062 ;
vdpb('c_Textiles','SSA') = 28712.21289 ;
vdpb('c_Textiles','ROW') = 279773.8125 ;
vdpb('c_Chem','USA') = 162893.6562 ;
vdpb('c_Chem','EU_28') = 63160.28906 ;
vdpb('c_Chem','CHN') = 95128.63281 ;
vdpb('c_Chem','JPN') = 26964.37891 ;
vdpb('c_Chem','IND') = 26593.05078 ;
vdpb('c_Chem','SSA') = 14418.82031 ;
vdpb('c_Chem','ROW') = 200212.8438 ;
vdpb('c_Manuf','USA') = 518719.5 ;
vdpb('c_Manuf','EU_28') = 256059.7031 ;
vdpb('c_Manuf','CHN') = 333492.4375 ;
vdpb('c_Manuf','JPN') = 109757.6562 ;
vdpb('c_Manuf','IND') = 78289.99219 ;
vdpb('c_Manuf','SSA') = 38941.86328 ;
vdpb('c_Manuf','ROW') = 519647.75 ;
vdpb('c_ForestFish','USA') = 3927.19165 ;
vdpb('c_ForestFish','EU_28') = 14389.79688 ;
vdpb('c_ForestFish','CHN') = 71678.17188 ;
vdpb('c_ForestFish','JPN') = 4060.15918 ;
vdpb('c_ForestFish','IND') = 38044.10938 ;
vdpb('c_ForestFish','SSA') = 43292.98047 ;
vdpb('c_ForestFish','ROW') = 88406.25781 ;
vdpb('c_Svces','USA') = 10502550 ;
vdpb('c_Svces','EU_28') = 6331369.5 ;
vdpb('c_Svces','CHN') = 2267669 ;
vdpb('c_Svces','JPN') = 2019539 ;
vdpb('c_Svces','IND') = 848619.25 ;
vdpb('c_Svces','SSA') = 422321.1875 ;
vdpb('c_Svces','ROW') = 7755806 ;

* vdpp data (70 cells)
vdpp('c_Rice','USA') = 4486.549316 ;
vdpp('c_Rice','EU_28') = 2863.28125 ;
vdpp('c_Rice','CHN') = 51861.85547 ;
vdpp('c_Rice','JPN') = 12246.43262 ;
vdpp('c_Rice','IND') = 13851.42871 ;
vdpp('c_Rice','SSA') = 11067.93359 ;
vdpp('c_Rice','ROW') = 168019.2969 ;
vdpp('c_Crops','USA') = 45741.63281 ;
vdpp('c_Crops','EU_28') = 68111.78125 ;
vdpp('c_Crops','CHN') = 174254.9375 ;
vdpp('c_Crops','JPN') = 19088.62109 ;
vdpp('c_Crops','IND') = 132292.2031 ;
vdpp('c_Crops','SSA') = 135294.4531 ;
vdpp('c_Crops','ROW') = 253048.7344 ;
vdpp('c_Livestock','USA') = 14554.04785 ;
vdpp('c_Livestock','EU_28') = 26979.87891 ;
vdpp('c_Livestock','CHN') = 114412.1406 ;
vdpp('c_Livestock','JPN') = 6550.102051 ;
vdpp('c_Livestock','IND') = 57313.55469 ;
vdpp('c_Livestock','SSA') = 28623.65039 ;
vdpp('c_Livestock','ROW') = 130148.9219 ;
vdpp('c_FoodProc','USA') = 632445.75 ;
vdpp('c_FoodProc','EU_28') = 709820.9375 ;
vdpp('c_FoodProc','CHN') = 823222.8125 ;
vdpp('c_FoodProc','JPN') = 200819.4688 ;
vdpp('c_FoodProc','IND') = 178233.0469 ;
vdpp('c_FoodProc','SSA') = 215237.3438 ;
vdpp('c_FoodProc','ROW') = 1311064.75 ;
vdpp('c_Energy','USA') = 276856.5312 ;
vdpp('c_Energy','EU_28') = 382142.3438 ;
vdpp('c_Energy','CHN') = 148529.5625 ;
vdpp('c_Energy','JPN') = 102323.4141 ;
vdpp('c_Energy','IND') = 85858.33594 ;
vdpp('c_Energy','SSA') = 23259.62305 ;
vdpp('c_Energy','ROW') = 494685.0625 ;
vdpp('c_Textiles','USA') = 30339.33594 ;
vdpp('c_Textiles','EU_28') = 82716.20312 ;
vdpp('c_Textiles','CHN') = 188896.7656 ;
vdpp('c_Textiles','JPN') = 15577.27539 ;
vdpp('c_Textiles','IND') = 91643.39062 ;
vdpp('c_Textiles','SSA') = 30218.66797 ;
vdpp('c_Textiles','ROW') = 294390.2188 ;
vdpp('c_Chem','USA') = 177229.0156 ;
vdpp('c_Chem','EU_28') = 81409.90625 ;
vdpp('c_Chem','CHN') = 106626.2969 ;
vdpp('c_Chem','JPN') = 26964.37891 ;
vdpp('c_Chem','IND') = 30574.9668 ;
vdpp('c_Chem','SSA') = 15520.50586 ;
vdpp('c_Chem','ROW') = 220263.7031 ;
vdpp('c_Manuf','USA') = 489792.25 ;
vdpp('c_Manuf','EU_28') = 316494.1562 ;
vdpp('c_Manuf','CHN') = 357099.2812 ;
vdpp('c_Manuf','JPN') = 109757.6562 ;
vdpp('c_Manuf','IND') = 84974.51562 ;
vdpp('c_Manuf','SSA') = 41034.23438 ;
vdpp('c_Manuf','ROW') = 574081.0625 ;
vdpp('c_ForestFish','USA') = 3947.145996 ;
vdpp('c_ForestFish','EU_28') = 15891.10156 ;
vdpp('c_ForestFish','CHN') = 71978.75 ;
vdpp('c_ForestFish','JPN') = 4060.15918 ;
vdpp('c_ForestFish','IND') = 38263.03125 ;
vdpp('c_ForestFish','SSA') = 44629.41797 ;
vdpp('c_ForestFish','ROW') = 89160.42969 ;
vdpp('c_Svces','USA') = 10726112 ;
vdpp('c_Svces','EU_28') = 6579813.5 ;
vdpp('c_Svces','CHN') = 2443109.5 ;
vdpp('c_Svces','JPN') = 2019539 ;
vdpp('c_Svces','IND') = 861426.3125 ;
vdpp('c_Svces','SSA') = 435631.2812 ;
vdpp('c_Svces','ROW') = 7984025.5 ;

* vmpb data (70 cells)
vmpb('c_Rice','USA') = 10.17045021 ;
vmpb('c_Rice','EU_28') = 1203.569824 ;
vmpb('c_Rice','CHN') = 722.9509888 ;
vmpb('c_Rice','JPN') = 696.8635254 ;
vmpb('c_Rice','IND') = 11.02929115 ;
vmpb('c_Rice','SSA') = 4989.609375 ;
vmpb('c_Rice','ROW') = 5316.573242 ;
vmpb('c_Crops','USA') = 22603.90039 ;
vmpb('c_Crops','EU_28') = 57276.07812 ;
vmpb('c_Crops','CHN') = 17458.0918 ;
vmpb('c_Crops','JPN') = 4383.429199 ;
vmpb('c_Crops','IND') = 7656.352539 ;
vmpb('c_Crops','SSA') = 6701.901855 ;
vmpb('c_Crops','ROW') = 62435.80859 ;
vmpb('c_Livestock','USA') = 421.9237366 ;
vmpb('c_Livestock','EU_28') = 2014.988037 ;
vmpb('c_Livestock','CHN') = 2097.986572 ;
vmpb('c_Livestock','JPN') = 640.0258179 ;
vmpb('c_Livestock','IND') = 56.1111412 ;
vmpb('c_Livestock','SSA') = 378.1662903 ;
vmpb('c_Livestock','ROW') = 7063.58252 ;
vmpb('c_FoodProc','USA') = 67913.38281 ;
vmpb('c_FoodProc','EU_28') = 215977.9844 ;
vmpb('c_FoodProc','CHN') = 29696.27539 ;
vmpb('c_FoodProc','JPN') = 34152.73828 ;
vmpb('c_FoodProc','IND') = 16134.79688 ;
vmpb('c_FoodProc','SSA') = 29159.22656 ;
vmpb('c_FoodProc','ROW') = 222347.875 ;
vmpb('c_Energy','USA') = 17448.66016 ;
vmpb('c_Energy','EU_28') = 38157.03516 ;
vmpb('c_Energy','CHN') = 7418.686035 ;
vmpb('c_Energy','JPN') = 6532.266602 ;
vmpb('c_Energy','IND') = 4386.317383 ;
vmpb('c_Energy','SSA') = 10311.26953 ;
vmpb('c_Energy','ROW') = 53988.64062 ;
vmpb('c_Textiles','USA') = 158664.1719 ;
vmpb('c_Textiles','EU_28') = 241355.2812 ;
vmpb('c_Textiles','CHN') = 9272.580078 ;
vmpb('c_Textiles','JPN') = 33862.14062 ;
vmpb('c_Textiles','IND') = 7303.916504 ;
vmpb('c_Textiles','SSA') = 26169.60938 ;
vmpb('c_Textiles','ROW') = 197697.75 ;
vmpb('c_Chem','USA') = 105177.5547 ;
vmpb('c_Chem','EU_28') = 121656.2031 ;
vmpb('c_Chem','CHN') = 10632.05273 ;
vmpb('c_Chem','JPN') = 8461.664062 ;
vmpb('c_Chem','IND') = 4542.619141 ;
vmpb('c_Chem','SSA') = 23238.44141 ;
vmpb('c_Chem','ROW') = 152459.9844 ;
vmpb('c_Manuf','USA') = 352315.2812 ;
vmpb('c_Manuf','EU_28') = 439152.6562 ;
vmpb('c_Manuf','CHN') = 121758.5859 ;
vmpb('c_Manuf','JPN') = 38983.92188 ;
vmpb('c_Manuf','IND') = 14936.86328 ;
vmpb('c_Manuf','SSA') = 36299.82812 ;
vmpb('c_Manuf','ROW') = 551682 ;
vmpb('c_ForestFish','USA') = 1067.34314 ;
vmpb('c_ForestFish','EU_28') = 5993.331543 ;
vmpb('c_ForestFish','CHN') = 1023.120972 ;
vmpb('c_ForestFish','JPN') = 559.8696899 ;
vmpb('c_ForestFish','IND') = 455.5964355 ;
vmpb('c_ForestFish','SSA') = 144.3791199 ;
vmpb('c_ForestFish','ROW') = 4121.694824 ;
vmpb('c_Svces','USA') = 178576.3125 ;
vmpb('c_Svces','EU_28') = 159395.5 ;
vmpb('c_Svces','CHN') = 78959.98438 ;
vmpb('c_Svces','JPN') = 46168.71094 ;
vmpb('c_Svces','IND') = 32459.48047 ;
vmpb('c_Svces','SSA') = 29291.50391 ;
vmpb('c_Svces','ROW') = 459474.8438 ;

* vmpp data (70 cells)
vmpp('c_Rice','USA') = 10.0776186 ;
vmpp('c_Rice','EU_28') = 1439.269165 ;
vmpp('c_Rice','CHN') = 865.9726562 ;
vmpp('c_Rice','JPN') = 739.3372803 ;
vmpp('c_Rice','IND') = 11.02929115 ;
vmpp('c_Rice','SSA') = 5530.704102 ;
vmpp('c_Rice','ROW') = 5489.155273 ;
vmpp('c_Crops','USA') = 22560.36523 ;
vmpp('c_Crops','EU_28') = 63152.49609 ;
vmpp('c_Crops','CHN') = 21641.52734 ;
vmpp('c_Crops','JPN') = 4625.543457 ;
vmpp('c_Crops','IND') = 7656.352539 ;
vmpp('c_Crops','SSA') = 6884.333496 ;
vmpp('c_Crops','ROW') = 63811.71875 ;
vmpp('c_Livestock','USA') = 419.9342957 ;
vmpp('c_Livestock','EU_28') = 2253.805664 ;
vmpp('c_Livestock','CHN') = 2516.67334 ;
vmpp('c_Livestock','JPN') = 683.8051147 ;
vmpp('c_Livestock','IND') = 56.1111412 ;
vmpp('c_Livestock','SSA') = 393.1453857 ;
vmpp('c_Livestock','ROW') = 7303.42627 ;
vmpp('c_FoodProc','USA') = 67819.90625 ;
vmpp('c_FoodProc','EU_28') = 287502.5 ;
vmpp('c_FoodProc','CHN') = 36144.09375 ;
vmpp('c_FoodProc','JPN') = 46980.3125 ;
vmpp('c_FoodProc','IND') = 16134.79785 ;
vmpp('c_FoodProc','SSA') = 31724.44922 ;
vmpp('c_FoodProc','ROW') = 250939.3281 ;
vmpp('c_Energy','USA') = 24221.68164 ;
vmpp('c_Energy','EU_28') = 87695.08594 ;
vmpp('c_Energy','CHN') = 7338.152832 ;
vmpp('c_Energy','JPN') = 11121.89062 ;
vmpp('c_Energy','IND') = 3985.434326 ;
vmpp('c_Energy','SSA') = 10520.39453 ;
vmpp('c_Energy','ROW') = 73738.24219 ;
vmpp('c_Textiles','USA') = 158534.875 ;
vmpp('c_Textiles','EU_28') = 333366.9375 ;
vmpp('c_Textiles','CHN') = 11724.48242 ;
vmpp('c_Textiles','JPN') = 35752.62891 ;
vmpp('c_Textiles','IND') = 7303.916016 ;
vmpp('c_Textiles','SSA') = 29422.84375 ;
vmpp('c_Textiles','ROW') = 215958.5938 ;
vmpp('c_Chem','USA') = 116431.8281 ;
vmpp('c_Chem','EU_28') = 156458.4844 ;
vmpp('c_Chem','CHN') = 14683.48047 ;
vmpp('c_Chem','JPN') = 9271.783203 ;
vmpp('c_Chem','IND') = 4542.619141 ;
vmpp('c_Chem','SSA') = 24912.68945 ;
vmpp('c_Chem','ROW') = 164922.4531 ;
vmpp('c_Manuf','USA') = 356344.125 ;
vmpp('c_Manuf','EU_28') = 552675.75 ;
vmpp('c_Manuf','CHN') = 151404.2812 ;
vmpp('c_Manuf','JPN') = 41594.48438 ;
vmpp('c_Manuf','IND') = 14936.86328 ;
vmpp('c_Manuf','SSA') = 38986.21875 ;
vmpp('c_Manuf','ROW') = 602083.875 ;
vmpp('c_ForestFish','USA') = 1074.780762 ;
vmpp('c_ForestFish','EU_28') = 6857.801758 ;
vmpp('c_ForestFish','CHN') = 1240.415039 ;
vmpp('c_ForestFish','JPN') = 616.3120117 ;
vmpp('c_ForestFish','IND') = 455.5964661 ;
vmpp('c_ForestFish','SSA') = 150.9103241 ;
vmpp('c_ForestFish','ROW') = 4251.088867 ;
vmpp('c_Svces','USA') = 183644.7344 ;
vmpp('c_Svces','EU_28') = 169805.3281 ;
vmpp('c_Svces','CHN') = 93180.66406 ;
vmpp('c_Svces','JPN') = 46185.94531 ;
vmpp('c_Svces','IND') = 32455.99219 ;
vmpp('c_Svces','SSA') = 30076.90039 ;
vmpp('c_Svces','ROW') = 474379.5312 ;

* vdgb data (63 cells)
vdgb('c_Rice','USA') = 0.1222780794 ;
vdgb('c_Rice','EU_28') = 4.330604076 ;
vdgb('c_Rice','CHN') = 0.1949149668 ;
vdgb('c_Rice','JPN') = 1.386882782 ;
vdgb('c_Rice','IND') = 0.04409774765 ;
vdgb('c_Rice','SSA') = 0.09242534637 ;
vdgb('c_Rice','ROW') = 1498.05127 ;
vdgb('c_Crops','USA') = 440.4668274 ;
vdgb('c_Crops','EU_28') = 1956.259155 ;
vdgb('c_Crops','CHN') = 938.7022705 ;
vdgb('c_Crops','JPN') = 322.8476562 ;
vdgb('c_Crops','IND') = 72.59395599 ;
vdgb('c_Crops','SSA') = 83.89903259 ;
vdgb('c_Crops','ROW') = 1803.254395 ;
vdgb('c_Livestock','USA') = 160.2139587 ;
vdgb('c_Livestock','EU_28') = 1005.742371 ;
vdgb('c_Livestock','CHN') = 171.8484497 ;
vdgb('c_Livestock','JPN') = 203.7824097 ;
vdgb('c_Livestock','IND') = 17.73552895 ;
vdgb('c_Livestock','SSA') = 98.3658905 ;
vdgb('c_Livestock','ROW') = 1377.15564 ;
vdgb('c_FoodProc','USA') = 17.73976326 ;
vdgb('c_FoodProc','EU_28') = 911.9969482 ;
vdgb('c_FoodProc','CHN') = 13.09091091 ;
vdgb('c_FoodProc','JPN') = 1928.005371 ;
vdgb('c_FoodProc','IND') = 1.845435262 ;
vdgb('c_FoodProc','SSA') = 47.01810455 ;
vdgb('c_FoodProc','ROW') = 2220.086914 ;
vdgb('c_Textiles','USA') = 13.01213169 ;
vdgb('c_Textiles','EU_28') = 135.2053375 ;
vdgb('c_Textiles','CHN') = 9.503994942 ;
vdgb('c_Textiles','JPN') = 22.36952209 ;
vdgb('c_Textiles','IND') = 1.375081658 ;
vdgb('c_Textiles','SSA') = 1.324502707 ;
vdgb('c_Textiles','ROW') = 2343.052246 ;
vdgb('c_Chem','USA') = 33.94022369 ;
vdgb('c_Chem','EU_28') = 28507.70117 ;
vdgb('c_Chem','CHN') = 24.16793633 ;
vdgb('c_Chem','JPN') = 106.6478653 ;
vdgb('c_Chem','IND') = 3.513437986 ;
vdgb('c_Chem','SSA') = 128.8380432 ;
vdgb('c_Chem','ROW') = 12031.82129 ;
vdgb('c_Manuf','USA') = 358.9374695 ;
vdgb('c_Manuf','EU_28') = 7211.553711 ;
vdgb('c_Manuf','CHN') = 263.3473206 ;
vdgb('c_Manuf','JPN') = 188.8950958 ;
vdgb('c_Manuf','IND') = 38.21921539 ;
vdgb('c_Manuf','SSA') = 42.77360535 ;
vdgb('c_Manuf','ROW') = 44480.00781 ;
vdgb('c_ForestFish','USA') = 405.2270203 ;
vdgb('c_ForestFish','EU_28') = 1396.64502 ;
vdgb('c_ForestFish','CHN') = 304.2010803 ;
vdgb('c_ForestFish','JPN') = 144.0723877 ;
vdgb('c_ForestFish','IND') = 44.19706345 ;
vdgb('c_ForestFish','SSA') = 28.12443733 ;
vdgb('c_ForestFish','ROW') = 1052.857178 ;
vdgb('c_Svces','USA') = 2543765 ;
vdgb('c_Svces','EU_28') = 3504409.5 ;
vdgb('c_Svces','CHN') = 1978844.125 ;
vdgb('c_Svces','JPN') = 957869.0625 ;
vdgb('c_Svces','IND') = 299009.6562 ;
vdgb('c_Svces','SSA') = 209029.6406 ;
vdgb('c_Svces','ROW') = 3529746 ;

* vdgp data (63 cells)
vdgp('c_Rice','USA') = 0.1222780794 ;
vdgp('c_Rice','EU_28') = 4.344447613 ;
vdgp('c_Rice','CHN') = 0.1949149668 ;
vdgp('c_Rice','JPN') = 1.386882782 ;
vdgp('c_Rice','IND') = 0.04409774765 ;
vdgp('c_Rice','SSA') = 0.09313297272 ;
vdgp('c_Rice','ROW') = 1503.77002 ;
vdgp('c_Crops','USA') = 440.4668274 ;
vdgp('c_Crops','EU_28') = 1976.633179 ;
vdgp('c_Crops','CHN') = 938.7022705 ;
vdgp('c_Crops','JPN') = 322.8476562 ;
vdgp('c_Crops','IND') = 72.59395599 ;
vdgp('c_Crops','SSA') = 83.90490723 ;
vdgp('c_Crops','ROW') = 1820.108032 ;
vdgp('c_Livestock','USA') = 160.2139587 ;
vdgp('c_Livestock','EU_28') = 1030.600464 ;
vdgp('c_Livestock','CHN') = 171.8484497 ;
vdgp('c_Livestock','JPN') = 203.7824097 ;
vdgp('c_Livestock','IND') = 17.73552895 ;
vdgp('c_Livestock','SSA') = 98.48234558 ;
vdgp('c_Livestock','ROW') = 1392.9646 ;
vdgp('c_FoodProc','USA') = 17.73976326 ;
vdgp('c_FoodProc','EU_28') = 1006.219543 ;
vdgp('c_FoodProc','CHN') = 13.09091091 ;
vdgp('c_FoodProc','JPN') = 1928.005371 ;
vdgp('c_FoodProc','IND') = 1.845435262 ;
vdgp('c_FoodProc','SSA') = 49.54713821 ;
vdgp('c_FoodProc','ROW') = 2273.849609 ;
vdgp('c_Textiles','USA') = 13.01213169 ;
vdgp('c_Textiles','EU_28') = 156.1360474 ;
vdgp('c_Textiles','CHN') = 9.503994942 ;
vdgp('c_Textiles','JPN') = 22.36952209 ;
vdgp('c_Textiles','IND') = 1.375081658 ;
vdgp('c_Textiles','SSA') = 1.35074544 ;
vdgp('c_Textiles','ROW') = 2410.145264 ;
vdgp('c_Chem','USA') = 33.94022369 ;
vdgp('c_Chem','EU_28') = 33012.35156 ;
vdgp('c_Chem','CHN') = 24.16793633 ;
vdgp('c_Chem','JPN') = 106.6478653 ;
vdgp('c_Chem','IND') = 3.513437986 ;
vdgp('c_Chem','SSA') = 128.8692322 ;
vdgp('c_Chem','ROW') = 12267.27051 ;
vdgp('c_Manuf','USA') = 358.9374695 ;
vdgp('c_Manuf','EU_28') = 7789.701172 ;
vdgp('c_Manuf','CHN') = 263.3473206 ;
vdgp('c_Manuf','JPN') = 188.8950958 ;
vdgp('c_Manuf','IND') = 38.21921539 ;
vdgp('c_Manuf','SSA') = 42.89476776 ;
vdgp('c_Manuf','ROW') = 44649.10938 ;
vdgp('c_ForestFish','USA') = 405.2270203 ;
vdgp('c_ForestFish','EU_28') = 1436.700806 ;
vdgp('c_ForestFish','CHN') = 304.2010803 ;
vdgp('c_ForestFish','JPN') = 144.0723877 ;
vdgp('c_ForestFish','IND') = 44.19706345 ;
vdgp('c_ForestFish','SSA') = 28.1310215 ;
vdgp('c_ForestFish','ROW') = 1060.737915 ;
vdgp('c_Svces','USA') = 2729009.5 ;
vdgp('c_Svces','EU_28') = 3507268.75 ;
vdgp('c_Svces','CHN') = 2000814.25 ;
vdgp('c_Svces','JPN') = 957869.0625 ;
vdgp('c_Svces','IND') = 299901.2188 ;
vdgp('c_Svces','SSA') = 209661.0469 ;
vdgp('c_Svces','ROW') = 3549092.75 ;

* vmgb data (63 cells)
vmgb('c_Rice','USA') = 0.01328294259 ;
vmgb('c_Rice','EU_28') = 0.8605020642 ;
vmgb('c_Rice','CHN') = 0.1432879865 ;
vmgb('c_Rice','JPN') = 0.06303656101 ;
vmgb('c_Rice','IND') = 0.00106750126 ;
vmgb('c_Rice','SSA') = 0.1064615324 ;
vmgb('c_Rice','ROW') = 32.56974411 ;
vmgb('c_Crops','USA') = 1.506719112 ;
vmgb('c_Crops','EU_28') = 3.272561789 ;
vmgb('c_Crops','CHN') = 0.6565289497 ;
vmgb('c_Crops','JPN') = 1.068147063 ;
vmgb('c_Crops','IND') = 0.6776886582 ;
vmgb('c_Crops','SSA') = 0.4136890173 ;
vmgb('c_Crops','ROW') = 295.9707642 ;
vmgb('c_Livestock','USA') = 0.1068898961 ;
vmgb('c_Livestock','EU_28') = 0.3678362668 ;
vmgb('c_Livestock','CHN') = 0.197441563 ;
vmgb('c_Livestock','JPN') = 0.4142270982 ;
vmgb('c_Livestock','IND') = 0.01103526726 ;
vmgb('c_Livestock','SSA') = 0.09363599867 ;
vmgb('c_Livestock','ROW') = 30.24571037 ;
vmgb('c_FoodProc','USA') = 6.314660072 ;
vmgb('c_FoodProc','EU_28') = 193.3376312 ;
vmgb('c_FoodProc','CHN') = 3.389645576 ;
vmgb('c_FoodProc','JPN') = 321.5488281 ;
vmgb('c_FoodProc','IND') = 0.2446876466 ;
vmgb('c_FoodProc','SSA') = 21.96788216 ;
vmgb('c_FoodProc','ROW') = 1233.861694 ;
vmgb('c_Textiles','USA') = 5.819881439 ;
vmgb('c_Textiles','EU_28') = 269.8993835 ;
vmgb('c_Textiles','CHN') = 3.716394663 ;
vmgb('c_Textiles','JPN') = 20.69091606 ;
vmgb('c_Textiles','IND') = 0.707882762 ;
vmgb('c_Textiles','SSA') = 3.2305758 ;
vmgb('c_Textiles','ROW') = 2198.718506 ;
vmgb('c_Chem','USA') = 24.07859039 ;
vmgb('c_Chem','EU_28') = 62426.15625 ;
vmgb('c_Chem','CHN') = 26.91961098 ;
vmgb('c_Chem','JPN') = 64.30712891 ;
vmgb('c_Chem','IND') = 2.398311377 ;
vmgb('c_Chem','SSA') = 11.11270809 ;
vmgb('c_Chem','ROW') = 10496.65723 ;
vmgb('c_Manuf','USA') = 302.0697937 ;
vmgb('c_Manuf','EU_28') = 5421.098633 ;
vmgb('c_Manuf','CHN') = 444.3608093 ;
vmgb('c_Manuf','JPN') = 122.9887848 ;
vmgb('c_Manuf','IND') = 44.75777435 ;
vmgb('c_Manuf','SSA') = 57.54653931 ;
vmgb('c_Manuf','ROW') = 53581.25 ;
vmgb('c_ForestFish','USA') = 0.000837104395 ;
vmgb('c_ForestFish','EU_28') = 3.08358264 ;
vmgb('c_ForestFish','CHN') = 0.007510072086 ;
vmgb('c_ForestFish','JPN') = 0.008915246464 ;
vmgb('c_ForestFish','IND') = 0.0003279024968 ;
vmgb('c_ForestFish','SSA') = 0.001907113125 ;
vmgb('c_ForestFish','ROW') = 11.03927612 ;
vmgb('c_Svces','USA') = 19186.18555 ;
vmgb('c_Svces','EU_28') = 7234.123535 ;
vmgb('c_Svces','CHN') = 33935.25391 ;
vmgb('c_Svces','JPN') = 3493.63208 ;
vmgb('c_Svces','IND') = 4590.567383 ;
vmgb('c_Svces','SSA') = 5370.649414 ;
vmgb('c_Svces','ROW') = 54590.00781 ;

* vmgp data (63 cells)
vmgp('c_Rice','USA') = 0.01328294259 ;
vmgp('c_Rice','EU_28') = 1.028742075 ;
vmgp('c_Rice','CHN') = 0.1432879865 ;
vmgp('c_Rice','JPN') = 0.06303656101 ;
vmgp('c_Rice','IND') = 0.00106750126 ;
vmgp('c_Rice','SSA') = 0.1064815819 ;
vmgp('c_Rice','ROW') = 32.61642838 ;
vmgp('c_Crops','USA') = 1.506719112 ;
vmgp('c_Crops','EU_28') = 3.463422537 ;
vmgp('c_Crops','CHN') = 0.6565289497 ;
vmgp('c_Crops','JPN') = 1.068147063 ;
vmgp('c_Crops','IND') = 0.6776886582 ;
vmgp('c_Crops','SSA') = 0.4138699472 ;
vmgp('c_Crops','ROW') = 296.3012695 ;
vmgp('c_Livestock','USA') = 0.1068898961 ;
vmgp('c_Livestock','EU_28') = 0.3814324141 ;
vmgp('c_Livestock','CHN') = 0.197441563 ;
vmgp('c_Livestock','JPN') = 0.4142270982 ;
vmgp('c_Livestock','IND') = 0.01103526726 ;
vmgp('c_Livestock','SSA') = 0.09373633564 ;
vmgp('c_Livestock','ROW') = 30.39574623 ;
vmgp('c_FoodProc','USA') = 6.314660072 ;
vmgp('c_FoodProc','EU_28') = 215.5359039 ;
vmgp('c_FoodProc','CHN') = 3.389645576 ;
vmgp('c_FoodProc','JPN') = 340.7662354 ;
vmgp('c_FoodProc','IND') = 0.2446876466 ;
vmgp('c_FoodProc','SSA') = 23.08028221 ;
vmgp('c_FoodProc','ROW') = 1235.767822 ;
vmgp('c_Textiles','USA') = 5.819881439 ;
vmgp('c_Textiles','EU_28') = 343.8320312 ;
vmgp('c_Textiles','CHN') = 3.716394663 ;
vmgp('c_Textiles','JPN') = 20.69091606 ;
vmgp('c_Textiles','IND') = 0.707882762 ;
vmgp('c_Textiles','SSA') = 3.255966663 ;
vmgp('c_Textiles','ROW') = 2213.432373 ;
vmgp('c_Chem','USA') = 24.07859039 ;
vmgp('c_Chem','EU_28') = 70365.78125 ;
vmgp('c_Chem','CHN') = 26.91961098 ;
vmgp('c_Chem','JPN') = 66.33345795 ;
vmgp('c_Chem','IND') = 2.398311377 ;
vmgp('c_Chem','SSA') = 11.46278381 ;
vmgp('c_Chem','ROW') = 10646.64258 ;
vmgp('c_Manuf','USA') = 302.0697937 ;
vmgp('c_Manuf','EU_28') = 6083.074219 ;
vmgp('c_Manuf','CHN') = 444.3608093 ;
vmgp('c_Manuf','JPN') = 125.2974854 ;
vmgp('c_Manuf','IND') = 44.75777435 ;
vmgp('c_Manuf','SSA') = 57.63170624 ;
vmgp('c_Manuf','ROW') = 53328.98438 ;
vmgp('c_ForestFish','USA') = 0.000837104395 ;
vmgp('c_ForestFish','EU_28') = 3.096780539 ;
vmgp('c_ForestFish','CHN') = 0.007510072086 ;
vmgp('c_ForestFish','JPN') = 0.008915246464 ;
vmgp('c_ForestFish','IND') = 0.0003279024968 ;
vmgp('c_ForestFish','SSA') = 0.001916707493 ;
vmgp('c_ForestFish','ROW') = 11.06474113 ;
vmgp('c_Svces','USA') = 19186.18555 ;
vmgp('c_Svces','EU_28') = 9761.845703 ;
vmgp('c_Svces','CHN') = 38809.62109 ;
vmgp('c_Svces','JPN') = 3494.71875 ;
vmgp('c_Svces','IND') = 4590.567383 ;
vmgp('c_Svces','SSA') = 5586.718262 ;
vmgp('c_Svces','ROW') = 55232.77734 ;

* vdib data (63 cells)
vdib('c_Rice','USA') = 0.4742407799 ;
vdib('c_Rice','EU_28') = 5.992819309 ;
vdib('c_Rice','CHN') = 0.7569878101 ;
vdib('c_Rice','JPN') = 0.2699179649 ;
vdib('c_Rice','IND') = 0.1496295631 ;
vdib('c_Rice','SSA') = 207.6387329 ;
vdib('c_Rice','ROW') = 200.2437897 ;
vdib('c_Crops','USA') = 357.835083 ;
vdib('c_Crops','EU_28') = 2424.16626 ;
vdib('c_Crops','CHN') = 1269.725342 ;
vdib('c_Crops','JPN') = 484.5107422 ;
vdib('c_Crops','IND') = 113.9300995 ;
vdib('c_Crops','SSA') = 235.4969788 ;
vdib('c_Crops','ROW') = 18263.05664 ;
vdib('c_Livestock','USA') = 135.9919281 ;
vdib('c_Livestock','EU_28') = 2208.307129 ;
vdib('c_Livestock','CHN') = 23035.39453 ;
vdib('c_Livestock','JPN') = 756.3590088 ;
vdib('c_Livestock','IND') = 408.2211304 ;
vdib('c_Livestock','SSA') = 3729.053467 ;
vdib('c_Livestock','ROW') = 20591.68555 ;
vdib('c_FoodProc','USA') = 20.57414055 ;
vdib('c_FoodProc','EU_28') = 183.7268372 ;
vdib('c_FoodProc','CHN') = 25.67851067 ;
vdib('c_FoodProc','JPN') = 15.03824234 ;
vdib('c_FoodProc','IND') = 4.13186264 ;
vdib('c_FoodProc','SSA') = 4536.757812 ;
vdib('c_FoodProc','ROW') = 7713.134277 ;
vdib('c_Textiles','USA') = 1211.788452 ;
vdib('c_Textiles','EU_28') = 4050.397461 ;
vdib('c_Textiles','CHN') = 69.88786316 ;
vdib('c_Textiles','JPN') = 1326.21875 ;
vdib('c_Textiles','IND') = 655.006897 ;
vdib('c_Textiles','SSA') = 239.9647217 ;
vdib('c_Textiles','ROW') = 3655.04126 ;
vdib('c_Chem','USA') = 3026.328613 ;
vdib('c_Chem','EU_28') = 6477.914551 ;
vdib('c_Chem','CHN') = 140.9984894 ;
vdib('c_Chem','JPN') = 73.8341217 ;
vdib('c_Chem','IND') = 2145.465576 ;
vdib('c_Chem','SSA') = 778.4736328 ;
vdib('c_Chem','ROW') = 25460.35156 ;
vdib('c_Manuf','USA') = 761694.3125 ;
vdib('c_Manuf','EU_28') = 462723.1562 ;
vdib('c_Manuf','CHN') = 947049.75 ;
vdib('c_Manuf','JPN') = 313798.8438 ;
vdib('c_Manuf','IND') = 172546.9375 ;
vdib('c_Manuf','SSA') = 36705.62891 ;
vdib('c_Manuf','ROW') = 597128.5625 ;
vdib('c_ForestFish','USA') = 7.774711609 ;
vdib('c_ForestFish','EU_28') = 873.9394531 ;
vdib('c_ForestFish','CHN') = 28.079916 ;
vdib('c_ForestFish','JPN') = 2.971425772 ;
vdib('c_ForestFish','IND') = 1.629967928 ;
vdib('c_ForestFish','SSA') = 1179.614624 ;
vdib('c_ForestFish','ROW') = 3338.261963 ;
vdib('c_Svces','USA') = 2873172.5 ;
vdib('c_Svces','EU_28') = 2258155.25 ;
vdib('c_Svces','CHN') = 3693320.5 ;
vdib('c_Svces','JPN') = 801181.8125 ;
vdib('c_Svces','IND') = 532308.6875 ;
vdib('c_Svces','SSA') = 247959.0156 ;
vdib('c_Svces','ROW') = 3615884.5 ;

* vdip data (63 cells)
vdip('c_Rice','USA') = 0.4742407799 ;
vdip('c_Rice','EU_28') = 5.94858408 ;
vdip('c_Rice','CHN') = 0.7569878101 ;
vdip('c_Rice','JPN') = 0.2699179649 ;
vdip('c_Rice','IND') = 0.1496295631 ;
vdip('c_Rice','SSA') = 223.7341461 ;
vdip('c_Rice','ROW') = 207.4293976 ;
vdip('c_Crops','USA') = 357.835083 ;
vdip('c_Crops','EU_28') = 2290.322266 ;
vdip('c_Crops','CHN') = 1269.725342 ;
vdip('c_Crops','JPN') = 484.5107422 ;
vdip('c_Crops','IND') = 113.9300995 ;
vdip('c_Crops','SSA') = 237.4392548 ;
vdip('c_Crops','ROW') = 18294.58789 ;
vdip('c_Livestock','USA') = 135.9919281 ;
vdip('c_Livestock','EU_28') = 2150.950684 ;
vdip('c_Livestock','CHN') = 23061.66211 ;
vdip('c_Livestock','JPN') = 756.3590088 ;
vdip('c_Livestock','IND') = 410.133728 ;
vdip('c_Livestock','SSA') = 3750.735107 ;
vdip('c_Livestock','ROW') = 21019.91602 ;
vdip('c_FoodProc','USA') = 20.57414055 ;
vdip('c_FoodProc','EU_28') = 183.5378876 ;
vdip('c_FoodProc','CHN') = 25.67851067 ;
vdip('c_FoodProc','JPN') = 15.03824234 ;
vdip('c_FoodProc','IND') = 4.13186264 ;
vdip('c_FoodProc','SSA') = 4703.117676 ;
vdip('c_FoodProc','ROW') = 7781.074219 ;
vdip('c_Textiles','USA') = 1224.154541 ;
vdip('c_Textiles','EU_28') = 4121.087891 ;
vdip('c_Textiles','CHN') = 69.88786316 ;
vdip('c_Textiles','JPN') = 1326.21875 ;
vdip('c_Textiles','IND') = 643.1776123 ;
vdip('c_Textiles','SSA') = 257.2770691 ;
vdip('c_Textiles','ROW') = 3780.29541 ;
vdip('c_Chem','USA') = 3103.839111 ;
vdip('c_Chem','EU_28') = 6804.167969 ;
vdip('c_Chem','CHN') = 140.9984894 ;
vdip('c_Chem','JPN') = 73.8341217 ;
vdip('c_Chem','IND') = 2464.930908 ;
vdip('c_Chem','SSA') = 853.324646 ;
vdip('c_Chem','ROW') = 25839.11719 ;
vdip('c_Manuf','USA') = 771481.125 ;
vdip('c_Manuf','EU_28') = 479862.4375 ;
vdip('c_Manuf','CHN') = 1020122.562 ;
vdip('c_Manuf','JPN') = 313798.8438 ;
vdip('c_Manuf','IND') = 186970.75 ;
vdip('c_Manuf','SSA') = 38319.71484 ;
vdip('c_Manuf','ROW') = 627629.125 ;
vdip('c_ForestFish','USA') = 7.774711609 ;
vdip('c_ForestFish','EU_28') = 860.7876587 ;
vdip('c_ForestFish','CHN') = 28.079916 ;
vdip('c_ForestFish','JPN') = 2.971425772 ;
vdip('c_ForestFish','IND') = 1.629967928 ;
vdip('c_ForestFish','SSA') = 1179.976318 ;
vdip('c_ForestFish','ROW') = 3374.136475 ;
vdip('c_Svces','USA') = 2692840.75 ;
vdip('c_Svces','EU_28') = 2438634.25 ;
vdip('c_Svces','CHN') = 3877627.75 ;
vdip('c_Svces','JPN') = 801181.8125 ;
vdip('c_Svces','IND') = 530251.5625 ;
vdip('c_Svces','SSA') = 249520.9375 ;
vdip('c_Svces','ROW') = 3722471.75 ;

* vmib data (63 cells)
vmib('c_Rice','USA') = 0.02249273472 ;
vmib('c_Rice','EU_28') = 1.902012944 ;
vmib('c_Rice','CHN') = 0.4101616144 ;
vmib('c_Rice','JPN') = 0.12182118 ;
vmib('c_Rice','IND') = 0.003489058698 ;
vmib('c_Rice','SSA') = 46.03876114 ;
vmib('c_Rice','ROW') = 98.14502716 ;
vmib('c_Crops','USA') = 9.661768913 ;
vmib('c_Crops','EU_28') = 1293.218018 ;
vmib('c_Crops','CHN') = 8.193821907 ;
vmib('c_Crops','JPN') = 95.60572052 ;
vmib('c_Crops','IND') = 7.018688202 ;
vmib('c_Crops','SSA') = 13.2028389 ;
vmib('c_Crops','ROW') = 2828.552979 ;
vmib('c_Livestock','USA') = 2.02631259 ;
vmib('c_Livestock','EU_28') = 319.3482361 ;
vmib('c_Livestock','CHN') = 386.4115906 ;
vmib('c_Livestock','JPN') = 53.15500641 ;
vmib('c_Livestock','IND') = 11.77356911 ;
vmib('c_Livestock','SSA') = 83.65659332 ;
vmib('c_Livestock','ROW') = 1477.59436 ;
vmib('c_FoodProc','USA') = 10.52007389 ;
vmib('c_FoodProc','EU_28') = 348.8583679 ;
vmib('c_FoodProc','CHN') = 9.87424469 ;
vmib('c_FoodProc','JPN') = 3.843831301 ;
vmib('c_FoodProc','IND') = 0.9115660787 ;
vmib('c_FoodProc','SSA') = 273.7076721 ;
vmib('c_FoodProc','ROW') = 3851.695312 ;
vmib('c_Textiles','USA') = 434.6702881 ;
vmib('c_Textiles','EU_28') = 3092.936035 ;
vmib('c_Textiles','CHN') = 121.4763718 ;
vmib('c_Textiles','JPN') = 1916.976685 ;
vmib('c_Textiles','IND') = 77.93212128 ;
vmib('c_Textiles','SSA') = 290.842804 ;
vmib('c_Textiles','ROW') = 2955.680908 ;
vmib('c_Chem','USA') = 1192.005859 ;
vmib('c_Chem','EU_28') = 6256.783203 ;
vmib('c_Chem','CHN') = 150.3626862 ;
vmib('c_Chem','JPN') = 34.19699478 ;
vmib('c_Chem','IND') = 333.3038025 ;
vmib('c_Chem','SSA') = 764.430603 ;
vmib('c_Chem','ROW') = 19848.55078 ;
vmib('c_Manuf','USA') = 472151.7188 ;
vmib('c_Manuf','EU_28') = 573132.9375 ;
vmib('c_Manuf','CHN') = 235050.9375 ;
vmib('c_Manuf','JPN') = 102032.25 ;
vmib('c_Manuf','IND') = 73451.25 ;
vmib('c_Manuf','SSA') = 59308.14844 ;
vmib('c_Manuf','ROW') = 798337.4375 ;
vmib('c_ForestFish','USA') = 0.1773753464 ;
vmib('c_ForestFish','EU_28') = 84.72657013 ;
vmib('c_ForestFish','CHN') = 0.788733542 ;
vmib('c_ForestFish','JPN') = 0.3916803002 ;
vmib('c_ForestFish','IND') = 0.09139864147 ;
vmib('c_ForestFish','SSA') = 0.3230300844 ;
vmib('c_ForestFish','ROW') = 264.2147827 ;
vmib('c_Svces','USA') = 99352.00781 ;
vmib('c_Svces','EU_28') = 92506.16406 ;
vmib('c_Svces','CHN') = 16403.72266 ;
vmib('c_Svces','JPN') = 14452.92871 ;
vmib('c_Svces','IND') = 2711.4375 ;
vmib('c_Svces','SSA') = 5716.712891 ;
vmib('c_Svces','ROW') = 100085.1953 ;

* vmip data (63 cells)
vmip('c_Rice','USA') = 0.02249273472 ;
vmip('c_Rice','EU_28') = 1.902012944 ;
vmip('c_Rice','CHN') = 0.4101616144 ;
vmip('c_Rice','JPN') = 0.12182118 ;
vmip('c_Rice','IND') = 0.003489058698 ;
vmip('c_Rice','SSA') = 52.29346848 ;
vmip('c_Rice','ROW') = 98.13011169 ;
vmip('c_Crops','USA') = 9.661768913 ;
vmip('c_Crops','EU_28') = 1315.939575 ;
vmip('c_Crops','CHN') = 8.193821907 ;
vmip('c_Crops','JPN') = 101.9810562 ;
vmip('c_Crops','IND') = 7.018688202 ;
vmip('c_Crops','SSA') = 13.70001984 ;
vmip('c_Crops','ROW') = 2851.798096 ;
vmip('c_Livestock','USA') = 2.02631259 ;
vmip('c_Livestock','EU_28') = 324.7572327 ;
vmip('c_Livestock','CHN') = 462.7410278 ;
vmip('c_Livestock','JPN') = 53.60737228 ;
vmip('c_Livestock','IND') = 11.77356911 ;
vmip('c_Livestock','SSA') = 84.62858582 ;
vmip('c_Livestock','ROW') = 1505.765503 ;
vmip('c_FoodProc','USA') = 10.52007389 ;
vmip('c_FoodProc','EU_28') = 349.0545959 ;
vmip('c_FoodProc','CHN') = 9.87424469 ;
vmip('c_FoodProc','JPN') = 3.843831301 ;
vmip('c_FoodProc','IND') = 0.9115660787 ;
vmip('c_FoodProc','SSA') = 294.5663147 ;
vmip('c_FoodProc','ROW') = 3852.035645 ;
vmip('c_Textiles','USA') = 439.4242554 ;
vmip('c_Textiles','EU_28') = 3150.375977 ;
vmip('c_Textiles','CHN') = 121.4763718 ;
vmip('c_Textiles','JPN') = 2040.870728 ;
vmip('c_Textiles','IND') = 77.93212128 ;
vmip('c_Textiles','SSA') = 339.8530579 ;
vmip('c_Textiles','ROW') = 3026.304688 ;
vmip('c_Chem','USA') = 1223.213135 ;
vmip('c_Chem','EU_28') = 6302.812012 ;
vmip('c_Chem','CHN') = 150.3626862 ;
vmip('c_Chem','JPN') = 34.19699478 ;
vmip('c_Chem','IND') = 333.3038025 ;
vmip('c_Chem','SSA') = 870.4917603 ;
vmip('c_Chem','ROW') = 19953.69727 ;
vmip('c_Manuf','USA') = 489776.875 ;
vmip('c_Manuf','EU_28') = 605249.9375 ;
vmip('c_Manuf','CHN') = 293198.875 ;
vmip('c_Manuf','JPN') = 108628.0703 ;
vmip('c_Manuf','IND') = 73451.25 ;
vmip('c_Manuf','SSA') = 62798.85547 ;
vmip('c_Manuf','ROW') = 830832.1875 ;
vmip('c_ForestFish','USA') = 0.1773753464 ;
vmip('c_ForestFish','EU_28') = 85.69670105 ;
vmip('c_ForestFish','CHN') = 0.788733542 ;
vmip('c_ForestFish','JPN') = 0.3916803002 ;
vmip('c_ForestFish','IND') = 0.09139864147 ;
vmip('c_ForestFish','SSA') = 0.3242782056 ;
vmip('c_ForestFish','ROW') = 264.4901123 ;
vmip('c_Svces','USA') = 87960.89844 ;
vmip('c_Svces','EU_28') = 99512.54688 ;
vmip('c_Svces','CHN') = 19477.99023 ;
vmip('c_Svces','JPN') = 14452.93652 ;
vmip('c_Svces','IND') = 2711.441406 ;
vmip('c_Svces','SSA') = 5731.401367 ;
vmip('c_Svces','ROW') = 100903.375 ;

* evfb data (252 cells)
evfb('Land','a_Rice','USA') = 237.9737701 ;
evfb('Land','a_Rice','EU_28') = 295.8180542 ;
evfb('Land','a_Rice','CHN') = 31457.03906 ;
evfb('Land','a_Rice','JPN') = 3342.075928 ;
evfb('Land','a_Rice','IND') = 11868.10547 ;
evfb('Land','a_Rice','SSA') = 2171.266357 ;
evfb('Land','a_Rice','ROW') = 37054.76562 ;
evfb('Land','a_Crops','USA') = 29688.15234 ;
evfb('Land','a_Crops','EU_28') = 40074.32031 ;
evfb('Land','a_Crops','CHN') = 161184.9375 ;
evfb('Land','a_Crops','JPN') = 4196.393066 ;
evfb('Land','a_Crops','IND') = 65975.74219 ;
evfb('Land','a_Crops','SSA') = 47590.85156 ;
evfb('Land','a_Crops','ROW') = 145005.25 ;
evfb('Land','a_Livestock','USA') = 11425.30664 ;
evfb('Land','a_Livestock','EU_28') = 27087.86328 ;
evfb('Land','a_Livestock','CHN') = 58363.21094 ;
evfb('Land','a_Livestock','JPN') = 1902.996948 ;
evfb('Land','a_Livestock','IND') = 29963.11328 ;
evfb('Land','a_Livestock','SSA') = 12083.93066 ;
evfb('Land','a_Livestock','ROW') = 57094.54297 ;
evfb('UnSkLab','a_Rice','USA') = 328.598938 ;
evfb('UnSkLab','a_Rice','EU_28') = 570.7598267 ;
evfb('UnSkLab','a_Rice','CHN') = 32998.42188 ;
evfb('UnSkLab','a_Rice','JPN') = 4247.269531 ;
evfb('UnSkLab','a_Rice','IND') = 22501.77148 ;
evfb('UnSkLab','a_Rice','SSA') = 2644.286377 ;
evfb('UnSkLab','a_Rice','ROW') = 56916.61328 ;
evfb('UnSkLab','a_Crops','USA') = 15732.84277 ;
evfb('UnSkLab','a_Crops','EU_28') = 48871.24219 ;
evfb('UnSkLab','a_Crops','CHN') = 185500.0625 ;
evfb('UnSkLab','a_Crops','JPN') = 6687.17627 ;
evfb('UnSkLab','a_Crops','IND') = 125051.6328 ;
evfb('UnSkLab','a_Crops','SSA') = 49562.19141 ;
evfb('UnSkLab','a_Crops','ROW') = 231106.6094 ;
evfb('UnSkLab','a_Livestock','USA') = 8012.672363 ;
evfb('UnSkLab','a_Livestock','EU_28') = 27739.08594 ;
evfb('UnSkLab','a_Livestock','CHN') = 68819.07812 ;
evfb('UnSkLab','a_Livestock','JPN') = 2961.500977 ;
evfb('UnSkLab','a_Livestock','IND') = 56788.375 ;
evfb('UnSkLab','a_Livestock','SSA') = 12008.6875 ;
evfb('UnSkLab','a_Livestock','ROW') = 94299.64062 ;
evfb('UnSkLab','a_FoodProc','USA') = 86589.10938 ;
evfb('UnSkLab','a_FoodProc','EU_28') = 59243.22656 ;
evfb('UnSkLab','a_FoodProc','CHN') = 134536.6562 ;
evfb('UnSkLab','a_FoodProc','JPN') = 16822.48242 ;
evfb('UnSkLab','a_FoodProc','IND') = 21342.25 ;
evfb('UnSkLab','a_FoodProc','SSA') = 55074.11719 ;
evfb('UnSkLab','a_FoodProc','ROW') = 156589.5469 ;
evfb('UnSkLab','a_Energy','USA') = 52987.96094 ;
evfb('UnSkLab','a_Energy','EU_28') = 19774.36133 ;
evfb('UnSkLab','a_Energy','CHN') = 107518.1094 ;
evfb('UnSkLab','a_Energy','JPN') = 6556.210938 ;
evfb('UnSkLab','a_Energy','IND') = 43081.73438 ;
evfb('UnSkLab','a_Energy','SSA') = 10074.56738 ;
evfb('UnSkLab','a_Energy','ROW') = 138968.9688 ;
evfb('UnSkLab','a_Textiles','USA') = 16518.31836 ;
evfb('UnSkLab','a_Textiles','EU_28') = 24240.91211 ;
evfb('UnSkLab','a_Textiles','CHN') = 126870.1641 ;
evfb('UnSkLab','a_Textiles','JPN') = 4399.085938 ;
evfb('UnSkLab','a_Textiles','IND') = 30787.01953 ;
evfb('UnSkLab','a_Textiles','SSA') = 6673.010254 ;
evfb('UnSkLab','a_Textiles','ROW') = 87037.47656 ;
evfb('UnSkLab','a_Chem','USA') = 76132.85156 ;
evfb('UnSkLab','a_Chem','EU_28') = 66021.32812 ;
evfb('UnSkLab','a_Chem','CHN') = 137848.6406 ;
evfb('UnSkLab','a_Chem','JPN') = 20538.01367 ;
evfb('UnSkLab','a_Chem','IND') = 21634.98438 ;
evfb('UnSkLab','a_Chem','SSA') = 4694.52832 ;
evfb('UnSkLab','a_Chem','ROW') = 113756.5 ;
evfb('UnSkLab','a_Manuf','USA') = 598912.5625 ;
evfb('UnSkLab','a_Manuf','EU_28') = 346252.4375 ;
evfb('UnSkLab','a_Manuf','CHN') = 761312.625 ;
evfb('UnSkLab','a_Manuf','JPN') = 140832.5469 ;
evfb('UnSkLab','a_Manuf','IND') = 81098.85938 ;
evfb('UnSkLab','a_Manuf','SSA') = 33909.06641 ;
evfb('UnSkLab','a_Manuf','ROW') = 548440.125 ;
evfb('UnSkLab','a_ForestFish','USA') = 8981.837891 ;
evfb('UnSkLab','a_ForestFish','EU_28') = 5734.775879 ;
evfb('UnSkLab','a_ForestFish','CHN') = 94875.16406 ;
evfb('UnSkLab','a_ForestFish','JPN') = 1564.188599 ;
evfb('UnSkLab','a_ForestFish','IND') = 28750.31836 ;
evfb('UnSkLab','a_ForestFish','SSA') = 20499.79883 ;
evfb('UnSkLab','a_ForestFish','ROW') = 37118.88281 ;
evfb('UnSkLab','a_Svces','USA') = 3282724.5 ;
evfb('UnSkLab','a_Svces','EU_28') = 1784304.5 ;
evfb('UnSkLab','a_Svces','CHN') = 2457188 ;
evfb('UnSkLab','a_Svces','JPN') = 615202.3125 ;
evfb('UnSkLab','a_Svces','IND') = 505118.3125 ;
evfb('UnSkLab','a_Svces','SSA') = 286792.7188 ;
evfb('UnSkLab','a_Svces','ROW') = 3247012.25 ;
evfb('SkLab','a_Rice','USA') = 192.1274872 ;
evfb('SkLab','a_Rice','EU_28') = 317.4306946 ;
evfb('SkLab','a_Rice','CHN') = 2629.330811 ;
evfb('SkLab','a_Rice','JPN') = 322.0011292 ;
evfb('SkLab','a_Rice','IND') = 40.8694458 ;
evfb('SkLab','a_Rice','SSA') = 220.9703064 ;
evfb('SkLab','a_Rice','ROW') = 5714.638672 ;
evfb('SkLab','a_Crops','USA') = 15368.17188 ;
evfb('SkLab','a_Crops','EU_28') = 12945.85254 ;
evfb('SkLab','a_Crops','CHN') = 11655.85645 ;
evfb('SkLab','a_Crops','JPN') = 412.239502 ;
evfb('SkLab','a_Crops','IND') = 221.1025848 ;
evfb('SkLab','a_Crops','SSA') = 2984.689941 ;
evfb('SkLab','a_Crops','ROW') = 17769.0293 ;
evfb('SkLab','a_Livestock','USA') = 7827.333984 ;
evfb('SkLab','a_Livestock','EU_28') = 9063.463867 ;
evfb('SkLab','a_Livestock','CHN') = 4324.113281 ;
evfb('SkLab','a_Livestock','JPN') = 182.4585571 ;
evfb('SkLab','a_Livestock','IND') = 100.4069824 ;
evfb('SkLab','a_Livestock','SSA') = 965.2688599 ;
evfb('SkLab','a_Livestock','ROW') = 10732.31641 ;
evfb('SkLab','a_FoodProc','USA') = 29087.99805 ;
evfb('SkLab','a_FoodProc','EU_28') = 66393.5625 ;
evfb('SkLab','a_FoodProc','CHN') = 26544.71484 ;
evfb('SkLab','a_FoodProc','JPN') = 8929.611328 ;
evfb('SkLab','a_FoodProc','IND') = 1791.658081 ;
evfb('SkLab','a_FoodProc','SSA') = 10008.34277 ;
evfb('SkLab','a_FoodProc','ROW') = 88573.14062 ;
evfb('SkLab','a_Energy','USA') = 21668.21484 ;
evfb('SkLab','a_Energy','EU_28') = 23952.67188 ;
evfb('SkLab','a_Energy','CHN') = 21067.27148 ;
evfb('SkLab','a_Energy','JPN') = 7134.15918 ;
evfb('SkLab','a_Energy','IND') = 35670.98828 ;
evfb('SkLab','a_Energy','SSA') = 5106.476074 ;
evfb('SkLab','a_Energy','ROW') = 119910.2344 ;
evfb('SkLab','a_Textiles','USA') = 5549.020508 ;
evfb('SkLab','a_Textiles','EU_28') = 24802.07227 ;
evfb('SkLab','a_Textiles','CHN') = 25032.07812 ;
evfb('SkLab','a_Textiles','JPN') = 2335.096924 ;
evfb('SkLab','a_Textiles','IND') = 2584.536133 ;
evfb('SkLab','a_Textiles','SSA') = 1399.537964 ;
evfb('SkLab','a_Textiles','ROW') = 45058.73438 ;
evfb('SkLab','a_Chem','USA') = 25575.41602 ;
evfb('SkLab','a_Chem','EU_28') = 72736.3125 ;
evfb('SkLab','a_Chem','CHN') = 27198.18359 ;
evfb('SkLab','a_Chem','JPN') = 10901.86719 ;
evfb('SkLab','a_Chem','IND') = 1816.23291 ;
evfb('SkLab','a_Chem','SSA') = 2355.787598 ;
evfb('SkLab','a_Chem','ROW') = 66846.82031 ;
evfb('SkLab','a_Manuf','USA') = 198553.4844 ;
evfb('SkLab','a_Manuf','EU_28') = 376365.4688 ;
evfb('SkLab','a_Manuf','CHN') = 146303.9688 ;
evfb('SkLab','a_Manuf','JPN') = 74695.21875 ;
evfb('SkLab','a_Manuf','IND') = 6948.978027 ;
evfb('SkLab','a_Manuf','SSA') = 12846.45312 ;
evfb('SkLab','a_Manuf','ROW') = 323507.1875 ;
evfb('SkLab','a_ForestFish','USA') = 8779.629883 ;
evfb('SkLab','a_ForestFish','EU_28') = 1702.996094 ;
evfb('SkLab','a_ForestFish','CHN') = 5960.460938 ;
evfb('SkLab','a_ForestFish','JPN') = 96.33818054 ;
evfb('SkLab','a_ForestFish','IND') = 50.83315659 ;
evfb('SkLab','a_ForestFish','SSA') = 750.3212891 ;
evfb('SkLab','a_ForestFish','ROW') = 4289.424316 ;
evfb('SkLab','a_Svces','USA') = 3780121.5 ;
evfb('SkLab','a_Svces','EU_28') = 2512409.5 ;
evfb('SkLab','a_Svces','CHN') = 1334502.625 ;
evfb('SkLab','a_Svces','JPN') = 550407.875 ;
evfb('SkLab','a_Svces','IND') = 347012.6562 ;
evfb('SkLab','a_Svces','SSA') = 211630.6875 ;
evfb('SkLab','a_Svces','ROW') = 3201343.5 ;
evfb('Capital','a_Rice','USA') = 937.3956299 ;
evfb('Capital','a_Rice','EU_28') = 1297.121826 ;
evfb('Capital','a_Rice','CHN') = 33949.44531 ;
evfb('Capital','a_Rice','JPN') = 4626.61084 ;
evfb('Capital','a_Rice','IND') = 5285.979004 ;
evfb('Capital','a_Rice','SSA') = 4857.964355 ;
evfb('Capital','a_Rice','ROW') = 54054.97266 ;
evfb('Capital','a_Crops','USA') = 54448.17969 ;
evfb('Capital','a_Crops','EU_28') = 49784.51172 ;
evfb('Capital','a_Crops','CHN') = 163105.125 ;
evfb('Capital','a_Crops','JPN') = 6409.053223 ;
evfb('Capital','a_Crops','IND') = 28824.5918 ;
evfb('Capital','a_Crops','SSA') = 54032.89844 ;
evfb('Capital','a_Crops','ROW') = 135998.1562 ;
evfb('Capital','a_Livestock','USA') = 27912.09961 ;
evfb('Capital','a_Livestock','EU_28') = 37928.51562 ;
evfb('Capital','a_Livestock','CHN') = 58506.13281 ;
evfb('Capital','a_Livestock','JPN') = 2839.307129 ;
evfb('Capital','a_Livestock','IND') = 13084.06445 ;
evfb('Capital','a_Livestock','SSA') = 14406.81055 ;
evfb('Capital','a_Livestock','ROW') = 63904.36328 ;
evfb('Capital','a_FoodProc','USA') = 132815.875 ;
evfb('Capital','a_FoodProc','EU_28') = 186700.9688 ;
evfb('Capital','a_FoodProc','CHN') = 178375.4531 ;
evfb('Capital','a_FoodProc','JPN') = 56064.10938 ;
evfb('Capital','a_FoodProc','IND') = 18155.46484 ;
evfb('Capital','a_FoodProc','SSA') = 73441.82031 ;
evfb('Capital','a_FoodProc','ROW') = 406259.6875 ;
evfb('Capital','a_Energy','USA') = 220980.7656 ;
evfb('Capital','a_Energy','EU_28') = 165926.8906 ;
evfb('Capital','a_Energy','CHN') = 187239.3438 ;
evfb('Capital','a_Energy','JPN') = 28119.59961 ;
evfb('Capital','a_Energy','IND') = 49939.65234 ;
evfb('Capital','a_Energy','SSA') = 67794.67969 ;
evfb('Capital','a_Energy','ROW') = 1159170.25 ;
evfb('Capital','a_Textiles','USA') = 6214.395996 ;
evfb('Capital','a_Textiles','EU_28') = 50620.69531 ;
evfb('Capital','a_Textiles','CHN') = 100960.4219 ;
evfb('Capital','a_Textiles','JPN') = 1821.347168 ;
evfb('Capital','a_Textiles','IND') = 27095.65625 ;
evfb('Capital','a_Textiles','SSA') = 6895.679688 ;
evfb('Capital','a_Textiles','ROW') = 134060.6719 ;
evfb('Capital','a_Chem','USA') = 232738.4688 ;
evfb('Capital','a_Chem','EU_28') = 283986.4688 ;
evfb('Capital','a_Chem','CHN') = 230319.9375 ;
evfb('Capital','a_Chem','JPN') = 61442.45312 ;
evfb('Capital','a_Chem','IND') = 48152.12891 ;
evfb('Capital','a_Chem','SSA') = 11409.67969 ;
evfb('Capital','a_Chem','ROW') = 353462.25 ;
evfb('Capital','a_Manuf','USA') = 638997.5625 ;
evfb('Capital','a_Manuf','EU_28') = 723452.25 ;
evfb('Capital','a_Manuf','CHN') = 937258.875 ;
evfb('Capital','a_Manuf','JPN') = 296133.3125 ;
evfb('Capital','a_Manuf','IND') = 110814.0469 ;
evfb('Capital','a_Manuf','SSA') = 74401.75 ;
evfb('Capital','a_Manuf','ROW') = 1300070.125 ;
evfb('Capital','a_ForestFish','USA') = 9328.414062 ;
evfb('Capital','a_ForestFish','EU_28') = 19083.57227 ;
evfb('Capital','a_ForestFish','CHN') = 7448.650391 ;
evfb('Capital','a_ForestFish','JPN') = 5042.11377 ;
evfb('Capital','a_ForestFish','IND') = 21811.125 ;
evfb('Capital','a_ForestFish','SSA') = 19815.53125 ;
evfb('Capital','a_ForestFish','ROW') = 66969.60156 ;
evfb('Capital','a_Svces','USA') = 5918652.5 ;
evfb('Capital','a_Svces','EU_28') = 5561561.5 ;
evfb('Capital','a_Svces','CHN') = 2660009.25 ;
evfb('Capital','a_Svces','JPN') = 1860368.5 ;
evfb('Capital','a_Svces','IND') = 736965.9375 ;
evfb('Capital','a_Svces','SSA') = 401853.6875 ;
evfb('Capital','a_Svces','ROW') = 6945574.5 ;
evfb('NatRes','a_Energy','USA') = 61101.10156 ;
evfb('NatRes','a_Energy','EU_28') = 13122.53516 ;
evfb('NatRes','a_Energy','CHN') = 53481.14453 ;
evfb('NatRes','a_Energy','JPN') = 87.80825043 ;
evfb('NatRes','a_Energy','IND') = 12273.74316 ;
evfb('NatRes','a_Energy','SSA') = 29107.96094 ;
evfb('NatRes','a_Energy','ROW') = 411645.375 ;
evfb('NatRes','a_Manuf','USA') = 14772.66016 ;
evfb('NatRes','a_Manuf','EU_28') = 6984.152344 ;
evfb('NatRes','a_Manuf','CHN') = 15029.12793 ;
evfb('NatRes','a_Manuf','JPN') = 927.2113037 ;
evfb('NatRes','a_Manuf','IND') = 2638.803711 ;
evfb('NatRes','a_Manuf','SSA') = 5342.300293 ;
evfb('NatRes','a_Manuf','ROW') = 41519.42578 ;
evfb('NatRes','a_ForestFish','USA') = 5286.424805 ;
evfb('NatRes','a_ForestFish','EU_28') = 8407.166016 ;
evfb('NatRes','a_ForestFish','CHN') = 44576.29688 ;
evfb('NatRes','a_ForestFish','JPN') = 3203.031738 ;
evfb('NatRes','a_ForestFish','IND') = 11570.00391 ;
evfb('NatRes','a_ForestFish','SSA') = 9842.236328 ;
evfb('NatRes','a_ForestFish','ROW') = 43246.70312 ;

* evfp data (252 cells)
evfp('Land','a_Rice','USA') = 175.0491028 ;
evfp('Land','a_Rice','EU_28') = 121.2894058 ;
evfp('Land','a_Rice','CHN') = 23235.52148 ;
evfp('Land','a_Rice','JPN') = 2060.299561 ;
evfp('Land','a_Rice','IND') = 11854.92578 ;
evfp('Land','a_Rice','SSA') = 2181.598877 ;
evfp('Land','a_Rice','ROW') = 36021.98047 ;
evfp('Land','a_Crops','USA') = 22212.12305 ;
evfp('Land','a_Crops','EU_28') = 26790.91992 ;
evfp('Land','a_Crops','CHN') = 149587.2031 ;
evfp('Land','a_Crops','JPN') = 3400.30249 ;
evfp('Land','a_Crops','IND') = 65937.10938 ;
evfp('Land','a_Crops','SSA') = 47756.49609 ;
evfp('Land','a_Crops','ROW') = 142926.6562 ;
evfp('Land','a_Livestock','USA') = 11615.86426 ;
evfp('Land','a_Livestock','EU_28') = 18522.625 ;
evfp('Land','a_Livestock','CHN') = 55378.47266 ;
evfp('Land','a_Livestock','JPN') = 1505.716919 ;
evfp('Land','a_Livestock','IND') = 29941.45312 ;
evfp('Land','a_Livestock','SSA') = 12140.49707 ;
evfp('Land','a_Livestock','ROW') = 56362.92969 ;
evfp('UnSkLab','a_Rice','USA') = 379.2288513 ;
evfp('UnSkLab','a_Rice','EU_28') = 734.2156982 ;
evfp('UnSkLab','a_Rice','CHN') = 32623.83594 ;
evfp('UnSkLab','a_Rice','JPN') = 5684.537598 ;
evfp('UnSkLab','a_Rice','IND') = 22497.80078 ;
evfp('UnSkLab','a_Rice','SSA') = 2693.616211 ;
evfp('UnSkLab','a_Rice','ROW') = 59315.44141 ;
evfp('UnSkLab','a_Crops','USA') = 17415.61523 ;
evfp('UnSkLab','a_Crops','EU_28') = 59511.92578 ;
evfp('UnSkLab','a_Crops','CHN') = 182968.4844 ;
evfp('UnSkLab','a_Crops','JPN') = 9075.378906 ;
evfp('UnSkLab','a_Crops','IND') = 125059.3828 ;
evfp('UnSkLab','a_Crops','SSA') = 50473.73047 ;
evfp('UnSkLab','a_Crops','ROW') = 255805.0156 ;
evfp('UnSkLab','a_Livestock','USA') = 9107.523438 ;
evfp('UnSkLab','a_Livestock','EU_28') = 33523.55469 ;
evfp('UnSkLab','a_Livestock','CHN') = 67736.52344 ;
evfp('UnSkLab','a_Livestock','JPN') = 4018.745361 ;
evfp('UnSkLab','a_Livestock','IND') = 56788.35156 ;
evfp('UnSkLab','a_Livestock','SSA') = 12213.47461 ;
evfp('UnSkLab','a_Livestock','ROW') = 106645.3281 ;
evfp('UnSkLab','a_FoodProc','USA') = 104216 ;
evfp('UnSkLab','a_FoodProc','EU_28') = 86580.94531 ;
evfp('UnSkLab','a_FoodProc','CHN') = 136798.25 ;
evfp('UnSkLab','a_FoodProc','JPN') = 24436.80078 ;
evfp('UnSkLab','a_FoodProc','IND') = 21349.25781 ;
evfp('UnSkLab','a_FoodProc','SSA') = 55827.43359 ;
evfp('UnSkLab','a_FoodProc','ROW') = 174787.8906 ;
evfp('UnSkLab','a_Energy','USA') = 63774.6875 ;
evfp('UnSkLab','a_Energy','EU_28') = 28768.41016 ;
evfp('UnSkLab','a_Energy','CHN') = 109325.5156 ;
evfp('UnSkLab','a_Energy','JPN') = 9523.733398 ;
evfp('UnSkLab','a_Energy','IND') = 43095.87891 ;
evfp('UnSkLab','a_Energy','SSA') = 10339.13477 ;
evfp('UnSkLab','a_Energy','ROW') = 156388.625 ;
evfp('UnSkLab','a_Textiles','USA') = 19880.94141 ;
evfp('UnSkLab','a_Textiles','EU_28') = 35891.08203 ;
evfp('UnSkLab','a_Textiles','CHN') = 129002.8906 ;
evfp('UnSkLab','a_Textiles','JPN') = 6390.233398 ;
evfp('UnSkLab','a_Textiles','IND') = 30797.12891 ;
evfp('UnSkLab','a_Textiles','SSA') = 6773.041504 ;
evfp('UnSkLab','a_Textiles','ROW') = 96273.45312 ;
evfp('UnSkLab','a_Chem','USA') = 91631.17188 ;
evfp('UnSkLab','a_Chem','EU_28') = 96711.01562 ;
evfp('UnSkLab','a_Chem','CHN') = 140165.9062 ;
evfp('UnSkLab','a_Chem','JPN') = 29834.08398 ;
evfp('UnSkLab','a_Chem','IND') = 21642.08789 ;
evfp('UnSkLab','a_Chem','SSA') = 4823.51416 ;
evfp('UnSkLab','a_Chem','ROW') = 130197.0625 ;
evfp('UnSkLab','a_Manuf','USA') = 720832.875 ;
evfp('UnSkLab','a_Manuf','EU_28') = 507185.6562 ;
evfp('UnSkLab','a_Manuf','CHN') = 774110.5 ;
evfp('UnSkLab','a_Manuf','JPN') = 204577.2344 ;
evfp('UnSkLab','a_Manuf','IND') = 81125.49219 ;
evfp('UnSkLab','a_Manuf','SSA') = 34646.4375 ;
evfp('UnSkLab','a_Manuf','ROW') = 625222.75 ;
evfp('UnSkLab','a_ForestFish','USA') = 10810.26562 ;
evfp('UnSkLab','a_ForestFish','EU_28') = 8495.539062 ;
evfp('UnSkLab','a_ForestFish','CHN') = 96470.03906 ;
evfp('UnSkLab','a_ForestFish','JPN') = 2272.18335 ;
evfp('UnSkLab','a_ForestFish','IND') = 28759.75977 ;
evfp('UnSkLab','a_ForestFish','SSA') = 20719.72266 ;
evfp('UnSkLab','a_ForestFish','ROW') = 40038.26562 ;
evfp('UnSkLab','a_Svces','USA') = 3950986.75 ;
evfp('UnSkLab','a_Svces','EU_28') = 2543290 ;
evfp('UnSkLab','a_Svces','CHN') = 2498494 ;
evfp('UnSkLab','a_Svces','JPN') = 893659.8125 ;
evfp('UnSkLab','a_Svces','IND') = 505284.1562 ;
evfp('UnSkLab','a_Svces','SSA') = 291320.125 ;
evfp('UnSkLab','a_Svces','ROW') = 3691171.5 ;
evfp('SkLab','a_Rice','USA') = 215.3659515 ;
evfp('SkLab','a_Rice','EU_28') = 436.3407898 ;
evfp('SkLab','a_Rice','CHN') = 2614.811768 ;
evfp('SkLab','a_Rice','JPN') = 437.1203308 ;
evfp('SkLab','a_Rice','IND') = 40.8627739 ;
evfp('SkLab','a_Rice','SSA') = 226.5501099 ;
evfp('SkLab','a_Rice','ROW') = 6250.526367 ;
evfp('SkLab','a_Crops','USA') = 17013.17383 ;
evfp('SkLab','a_Crops','EU_28') = 15547.78809 ;
evfp('SkLab','a_Crops','CHN') = 11494.85742 ;
evfp('SkLab','a_Crops','JPN') = 558.8115234 ;
evfp('SkLab','a_Crops','IND') = 221.1163025 ;
evfp('SkLab','a_Crops','SSA') = 3051.185791 ;
evfp('SkLab','a_Crops','ROW') = 19222.01367 ;
evfp('SkLab','a_Livestock','USA') = 8897.066406 ;
evfp('SkLab','a_Livestock','EU_28') = 10038.95215 ;
evfp('SkLab','a_Livestock','CHN') = 4255.495605 ;
evfp('SkLab','a_Livestock','JPN') = 247.4521027 ;
evfp('SkLab','a_Livestock','IND') = 100.4069366 ;
evfp('SkLab','a_Livestock','SSA') = 987.2731323 ;
evfp('SkLab','a_Livestock','ROW') = 11622.05273 ;
evfp('SkLab','a_FoodProc','USA') = 35009.42578 ;
evfp('SkLab','a_FoodProc','EU_28') = 95137.3125 ;
evfp('SkLab','a_FoodProc','CHN') = 26990.9375 ;
evfp('SkLab','a_FoodProc','JPN') = 12971.40039 ;
evfp('SkLab','a_FoodProc','IND') = 1792.246338 ;
evfp('SkLab','a_FoodProc','SSA') = 10219.10254 ;
evfp('SkLab','a_FoodProc','ROW') = 98851.39062 ;
evfp('SkLab','a_Energy','USA') = 26079.20117 ;
evfp('SkLab','a_Energy','EU_28') = 34500.87109 ;
evfp('SkLab','a_Energy','CHN') = 21421.41797 ;
evfp('SkLab','a_Energy','JPN') = 10363.27637 ;
evfp('SkLab','a_Energy','IND') = 35682.69922 ;
evfp('SkLab','a_Energy','SSA') = 5245 ;
evfp('SkLab','a_Energy','ROW') = 133876.7812 ;
evfp('SkLab','a_Textiles','USA') = 6678.631348 ;
evfp('SkLab','a_Textiles','EU_28') = 36208.80469 ;
evfp('SkLab','a_Textiles','CHN') = 25452.875 ;
evfp('SkLab','a_Textiles','JPN') = 3392.026123 ;
evfp('SkLab','a_Textiles','IND') = 2585.384766 ;
evfp('SkLab','a_Textiles','SSA') = 1432.476196 ;
evfp('SkLab','a_Textiles','ROW') = 49419.73828 ;
evfp('SkLab','a_Chem','USA') = 30781.78906 ;
evfp('SkLab','a_Chem','EU_28') = 104813.0625 ;
evfp('SkLab','a_Chem','CHN') = 27655.39453 ;
evfp('SkLab','a_Chem','JPN') = 15836.35254 ;
evfp('SkLab','a_Chem','IND') = 1816.829346 ;
evfp('SkLab','a_Chem','SSA') = 2442.817383 ;
evfp('SkLab','a_Chem','ROW') = 76250.57812 ;
evfp('SkLab','a_Manuf','USA') = 238972.9062 ;
evfp('SkLab','a_Manuf','EU_28') = 542165.75 ;
evfp('SkLab','a_Manuf','CHN') = 148763.375 ;
evfp('SkLab','a_Manuf','JPN') = 108504.3359 ;
evfp('SkLab','a_Manuf','IND') = 6951.259766 ;
evfp('SkLab','a_Manuf','SSA') = 13204.73438 ;
evfp('SkLab','a_Manuf','ROW') = 368192.4062 ;
evfp('SkLab','a_ForestFish','USA') = 10566.89453 ;
evfp('SkLab','a_ForestFish','EU_28') = 2388.358398 ;
evfp('SkLab','a_ForestFish','CHN') = 6060.657715 ;
evfp('SkLab','a_ForestFish','JPN') = 139.9434967 ;
evfp('SkLab','a_ForestFish','IND') = 50.84984589 ;
evfp('SkLab','a_ForestFish','SSA') = 759.8832397 ;
evfp('SkLab','a_ForestFish','ROW') = 4695.709473 ;
evfp('SkLab','a_Svces','USA') = 4549638.5 ;
evfp('SkLab','a_Svces','EU_28') = 3595405 ;
evfp('SkLab','a_Svces','CHN') = 1356936 ;
evfp('SkLab','a_Svces','JPN') = 799537.6875 ;
evfp('SkLab','a_Svces','IND') = 347126.5938 ;
evfp('SkLab','a_Svces','SSA') = 216050.7812 ;
evfp('SkLab','a_Svces','ROW') = 3624202.75 ;
evfp('Capital','a_Rice','USA') = 954.8636475 ;
evfp('Capital','a_Rice','EU_28') = 1271.075684 ;
evfp('Capital','a_Rice','CHN') = 30948.73828 ;
evfp('Capital','a_Rice','JPN') = 4298.065918 ;
evfp('Capital','a_Rice','IND') = 5200.060059 ;
evfp('Capital','a_Rice','SSA') = 4877.038086 ;
evfp('Capital','a_Rice','ROW') = 53897.51953 ;
evfp('Capital','a_Crops','USA') = 54419.69922 ;
evfp('Capital','a_Crops','EU_28') = 42393.32812 ;
evfp('Capital','a_Crops','CHN') = 154573.4375 ;
evfp('Capital','a_Crops','JPN') = 5856.07666 ;
evfp('Capital','a_Crops','IND') = 28572.74609 ;
evfp('Capital','a_Crops','SSA') = 54204.74219 ;
evfp('Capital','a_Crops','ROW') = 133750.4062 ;
evfp('Capital','a_Livestock','USA') = 28458.86719 ;
evfp('Capital','a_Livestock','EU_28') = 29858.27539 ;
evfp('Capital','a_Livestock','CHN') = 57224.42578 ;
evfp('Capital','a_Livestock','JPN') = 2593.178955 ;
evfp('Capital','a_Livestock','IND') = 12974.62891 ;
evfp('Capital','a_Livestock','SSA') = 14465.99121 ;
evfp('Capital','a_Livestock','ROW') = 61127.49609 ;
evfp('Capital','a_FoodProc','USA') = 139467.5156 ;
evfp('Capital','a_FoodProc','EU_28') = 193159.4688 ;
evfp('Capital','a_FoodProc','CHN') = 181374 ;
evfp('Capital','a_FoodProc','JPN') = 57647.87109 ;
evfp('Capital','a_FoodProc','IND') = 18155.46484 ;
evfp('Capital','a_FoodProc','SSA') = 73671.85938 ;
evfp('Capital','a_FoodProc','ROW') = 411427.2812 ;
evfp('Capital','a_Energy','USA') = 232047.8594 ;
evfp('Capital','a_Energy','EU_28') = 171349.125 ;
evfp('Capital','a_Energy','CHN') = 190386.875 ;
evfp('Capital','a_Energy','JPN') = 28913.95508 ;
evfp('Capital','a_Energy','IND') = 49939.65234 ;
evfp('Capital','a_Energy','SSA') = 68128.32031 ;
evfp('Capital','a_Energy','ROW') = 1174419.375 ;
evfp('Capital','a_Textiles','USA') = 6525.624023 ;
evfp('Capital','a_Textiles','EU_28') = 52016.11719 ;
evfp('Capital','a_Textiles','CHN') = 102657.5938 ;
evfp('Capital','a_Textiles','JPN') = 1872.798584 ;
evfp('Capital','a_Textiles','IND') = 27095.65625 ;
evfp('Capital','a_Textiles','SSA') = 6928.507812 ;
evfp('Capital','a_Textiles','ROW') = 135184.6562 ;
evfp('Capital','a_Chem','USA') = 244394.4062 ;
evfp('Capital','a_Chem','EU_28') = 292243.5 ;
evfp('Capital','a_Chem','CHN') = 234191.6719 ;
evfp('Capital','a_Chem','JPN') = 63178.14844 ;
evfp('Capital','a_Chem','IND') = 48152.12891 ;
evfp('Capital','a_Chem','SSA') = 11500.25781 ;
evfp('Capital','a_Chem','ROW') = 359438 ;
evfp('Capital','a_Manuf','USA') = 670999.625 ;
evfp('Capital','a_Manuf','EU_28') = 743195.3125 ;
evfp('Capital','a_Manuf','CHN') = 953014.4375 ;
evfp('Capital','a_Manuf','JPN') = 304498.8438 ;
evfp('Capital','a_Manuf','IND') = 110814.0469 ;
evfp('Capital','a_Manuf','SSA') = 74912.88281 ;
evfp('Capital','a_Manuf','ROW') = 1322278.125 ;
evfp('Capital','a_ForestFish','USA') = 9795.595703 ;
evfp('Capital','a_ForestFish','EU_28') = 19744.96094 ;
evfp('Capital','a_ForestFish','CHN') = 7573.864258 ;
evfp('Capital','a_ForestFish','JPN') = 5184.548828 ;
evfp('Capital','a_ForestFish','IND') = 21811.125 ;
evfp('Capital','a_ForestFish','SSA') = 19851.13086 ;
evfp('Capital','a_ForestFish','ROW') = 67593.66406 ;
evfp('Capital','a_Svces','USA') = 6215069 ;
evfp('Capital','a_Svces','EU_28') = 5746506.5 ;
evfp('Capital','a_Svces','CHN') = 2704724.75 ;
evfp('Capital','a_Svces','JPN') = 1912922.25 ;
evfp('Capital','a_Svces','IND') = 736965.9375 ;
evfp('Capital','a_Svces','SSA') = 404798.7812 ;
evfp('Capital','a_Svces','ROW') = 7071225 ;
evfp('NatRes','a_Energy','USA') = 64161.14844 ;
evfp('NatRes','a_Energy','EU_28') = 13490.23828 ;
evfp('NatRes','a_Energy','CHN') = 54380.17578 ;
evfp('NatRes','a_Energy','JPN') = 90.28875732 ;
evfp('NatRes','a_Energy','IND') = 12273.74316 ;
evfp('NatRes','a_Energy','SSA') = 29231.15039 ;
evfp('NatRes','a_Energy','ROW') = 417124.4062 ;
evfp('NatRes','a_Manuf','USA') = 15512.5 ;
evfp('NatRes','a_Manuf','EU_28') = 7221.031738 ;
evfp('NatRes','a_Manuf','CHN') = 15281.77148 ;
evfp('NatRes','a_Manuf','JPN') = 953.4042358 ;
evfp('NatRes','a_Manuf','IND') = 2638.803711 ;
evfp('NatRes','a_Manuf','SSA') = 5417.024414 ;
evfp('NatRes','a_Manuf','ROW') = 42361.21094 ;
evfp('NatRes','a_ForestFish','USA') = 5551.177734 ;
evfp('NatRes','a_ForestFish','EU_28') = 8703.983398 ;
evfp('NatRes','a_ForestFish','CHN') = 45325.63672 ;
evfp('NatRes','a_ForestFish','JPN') = 3293.514648 ;
evfp('NatRes','a_ForestFish','IND') = 11570.00391 ;
evfp('NatRes','a_ForestFish','SSA') = 9863.556641 ;
evfp('NatRes','a_ForestFish','ROW') = 43704.02344 ;

* evos data (252 cells)
evos('Land','a_Rice','USA') = 228.2940826 ;
evos('Land','a_Rice','EU_28') = 280.8450317 ;
evos('Land','a_Rice','CHN') = 27955.06445 ;
evos('Land','a_Rice','JPN') = 3084.003662 ;
evos('Land','a_Rice','IND') = 10999.45508 ;
evos('Land','a_Rice','SSA') = 2071.828857 ;
evos('Land','a_Rice','ROW') = 34734.95312 ;
evos('Land','a_Crops','USA') = 28480.57422 ;
evos('Land','a_Crops','EU_28') = 37928.20312 ;
evos('Land','a_Crops','CHN') = 143240.9219 ;
evos('Land','a_Crops','JPN') = 3872.351318 ;
evos('Land','a_Crops','IND') = 61146.84766 ;
evos('Land','a_Crops','SSA') = 45418.17578 ;
evos('Land','a_Crops','ROW') = 135155.5312 ;
evos('Land','a_Livestock','USA') = 10960.57715 ;
evos('Land','a_Livestock','EU_28') = 25557.00781 ;
evos('Land','a_Livestock','CHN') = 51865.89062 ;
evos('Land','a_Livestock','JPN') = 1756.049072 ;
evos('Land','a_Livestock','IND') = 27770.05078 ;
evos('Land','a_Livestock','SSA') = 11579.07617 ;
evos('Land','a_Livestock','ROW') = 53169.59766 ;
evos('UnSkLab','a_Rice','USA') = 247.4241028 ;
evos('UnSkLab','a_Rice','EU_28') = 396.8377991 ;
evos('UnSkLab','a_Rice','CHN') = 31975.72852 ;
evos('UnSkLab','a_Rice','JPN') = 3421.477051 ;
evos('UnSkLab','a_Rice','IND') = 21371.73438 ;
evos('UnSkLab','a_Rice','SSA') = 2558.562988 ;
evos('UnSkLab','a_Rice','ROW') = 54053.53906 ;
evos('UnSkLab','a_Crops','USA') = 11846.30859 ;
evos('UnSkLab','a_Crops','EU_28') = 34355.87891 ;
evos('UnSkLab','a_Crops','CHN') = 179751.0156 ;
evos('UnSkLab','a_Crops','JPN') = 5386.995117 ;
evos('UnSkLab','a_Crops','IND') = 118771.5547 ;
evos('UnSkLab','a_Crops','SSA') = 47181.67969 ;
evos('UnSkLab','a_Crops','ROW') = 212018.7656 ;
evos('UnSkLab','a_Livestock','USA') = 6033.276855 ;
evos('UnSkLab','a_Livestock','EU_28') = 19063.16016 ;
evos('UnSkLab','a_Livestock','CHN') = 66686.22656 ;
evos('UnSkLab','a_Livestock','JPN') = 2385.699463 ;
evos('UnSkLab','a_Livestock','IND') = 53936.47266 ;
evos('UnSkLab','a_Livestock','SSA') = 11255.74609 ;
evos('UnSkLab','a_Livestock','ROW') = 85511.92969 ;
evos('UnSkLab','a_FoodProc','USA') = 65198.72656 ;
evos('UnSkLab','a_FoodProc','EU_28') = 41358.76562 ;
evos('UnSkLab','a_FoodProc','CHN') = 130367.0703 ;
evos('UnSkLab','a_FoodProc','JPN') = 13551.7041 ;
evos('UnSkLab','a_FoodProc','IND') = 20270.44531 ;
evos('UnSkLab','a_FoodProc','SSA') = 53044.93359 ;
evos('UnSkLab','a_FoodProc','ROW') = 142366.2188 ;
evos('UnSkLab','a_Energy','USA') = 39898.17969 ;
evos('UnSkLab','a_Energy','EU_28') = 14185.1875 ;
evos('UnSkLab','a_Energy','CHN') = 104185.8906 ;
evos('UnSkLab','a_Energy','JPN') = 5281.493652 ;
evos('UnSkLab','a_Energy','IND') = 40918.17969 ;
evos('UnSkLab','a_Energy','SSA') = 8947.392578 ;
evos('UnSkLab','a_Energy','ROW') = 126218.1953 ;
evos('UnSkLab','a_Textiles','USA') = 12437.74512 ;
evos('UnSkLab','a_Textiles','EU_28') = 16771.96289 ;
evos('UnSkLab','a_Textiles','CHN') = 122938.1875 ;
evos('UnSkLab','a_Textiles','JPN') = 3543.776123 ;
evos('UnSkLab','a_Textiles','IND') = 29240.90039 ;
evos('UnSkLab','a_Textiles','SSA') = 6359.553223 ;
evos('UnSkLab','a_Textiles','ROW') = 80802.78906 ;
evos('UnSkLab','a_Chem','USA') = 57325.51172 ;
evos('UnSkLab','a_Chem','EU_28') = 45676.14062 ;
evos('UnSkLab','a_Chem','CHN') = 133576.4062 ;
evos('UnSkLab','a_Chem','JPN') = 16544.82812 ;
evos('UnSkLab','a_Chem','IND') = 20548.47852 ;
evos('UnSkLab','a_Chem','SSA') = 4220.524414 ;
evos('UnSkLab','a_Chem','ROW') = 101883.25 ;
evos('UnSkLab','a_Manuf','USA') = 450961.3438 ;
evos('UnSkLab','a_Manuf','EU_28') = 238032.4219 ;
evos('UnSkLab','a_Manuf','CHN') = 737717.9375 ;
evos('UnSkLab','a_Manuf','JPN') = 113450.6172 ;
evos('UnSkLab','a_Manuf','IND') = 77026.09375 ;
evos('UnSkLab','a_Manuf','SSA') = 30445.4707 ;
evos('UnSkLab','a_Manuf','ROW') = 490048.875 ;
evos('UnSkLab','a_ForestFish','USA') = 6763.026367 ;
evos('UnSkLab','a_ForestFish','EU_28') = 3957.259033 ;
evos('UnSkLab','a_ForestFish','CHN') = 91934.78125 ;
evos('UnSkLab','a_ForestFish','JPN') = 1260.064941 ;
evos('UnSkLab','a_ForestFish','IND') = 27306.48242 ;
evos('UnSkLab','a_ForestFish','SSA') = 19879.26367 ;
evos('UnSkLab','a_ForestFish','ROW') = 34334.34766 ;
evos('UnSkLab','a_Svces','USA') = 2471782.75 ;
evos('UnSkLab','a_Svces','EU_28') = 1264046.5 ;
evos('UnSkLab','a_Svces','CHN') = 2381034.75 ;
evos('UnSkLab','a_Svces','JPN') = 495589.1562 ;
evos('UnSkLab','a_Svces','IND') = 479751.375 ;
evos('UnSkLab','a_Svces','SSA') = 267121.7188 ;
evos('UnSkLab','a_Svces','ROW') = 2886359.25 ;
evos('SkLab','a_Rice','USA') = 144.6656342 ;
evos('SkLab','a_Rice','EU_28') = 208.9764557 ;
evos('SkLab','a_Rice','CHN') = 2547.842285 ;
evos('SkLab','a_Rice','JPN') = 259.3948364 ;
evos('SkLab','a_Rice','IND') = 38.81698608 ;
evos('SkLab','a_Rice','SSA') = 208.3143311 ;
evos('SkLab','a_Rice','ROW') = 5279.5 ;
evos('SkLab','a_Crops','USA') = 11571.72461 ;
evos('SkLab','a_Crops','EU_28') = 8405.756836 ;
evos('SkLab','a_Crops','CHN') = 11294.61621 ;
evos('SkLab','a_Crops','JPN') = 332.0882568 ;
evos('SkLab','a_Crops','IND') = 209.9988403 ;
evos('SkLab','a_Crops','SSA') = 2771.30127 ;
evos('SkLab','a_Crops','ROW') = 15779.29004 ;
evos('SkLab','a_Livestock','USA') = 5893.723633 ;
evos('SkLab','a_Livestock','EU_28') = 6126.311523 ;
evos('SkLab','a_Livestock','CHN') = 4190.099609 ;
evos('SkLab','a_Livestock','JPN') = 146.9833374 ;
evos('SkLab','a_Livestock','IND') = 95.36456299 ;
evos('SkLab','a_Livestock','SSA') = 856.692627 ;
evos('SkLab','a_Livestock','ROW') = 9128.285156 ;
evos('SkLab','a_FoodProc','USA') = 21902.30078 ;
evos('SkLab','a_FoodProc','EU_28') = 46755.06641 ;
evos('SkLab','a_FoodProc','CHN') = 25722.03711 ;
evos('SkLab','a_FoodProc','JPN') = 7193.4375 ;
evos('SkLab','a_FoodProc','IND') = 1701.681274 ;
evos('SkLab','a_FoodProc','SSA') = 9419.998047 ;
evos('SkLab','a_FoodProc','ROW') = 78854.15625 ;
evos('SkLab','a_Energy','USA') = 16315.44922 ;
evos('SkLab','a_Energy','EU_28') = 17062.15625 ;
evos('SkLab','a_Energy','CHN') = 20414.35156 ;
evos('SkLab','a_Energy','JPN') = 5747.072754 ;
evos('SkLab','a_Energy','IND') = 33879.59766 ;
evos('SkLab','a_Energy','SSA') = 4532.788574 ;
evos('SkLab','a_Energy','ROW') = 108299.4844 ;
evos('SkLab','a_Textiles','USA') = 4178.229004 ;
evos('SkLab','a_Textiles','EU_28') = 17152.96875 ;
evos('SkLab','a_Textiles','CHN') = 24256.28125 ;
evos('SkLab','a_Textiles','JPN') = 1881.086548 ;
evos('SkLab','a_Textiles','IND') = 2454.741211 ;
evos('SkLab','a_Textiles','SSA') = 1279.411743 ;
evos('SkLab','a_Textiles','ROW') = 41863.94531 ;
evos('SkLab','a_Chem','USA') = 19257.44336 ;
evos('SkLab','a_Chem','EU_28') = 50808.05859 ;
evos('SkLab','a_Chem','CHN') = 26355.25391 ;
evos('SkLab','a_Chem','JPN') = 8782.229492 ;
evos('SkLab','a_Chem','IND') = 1725.021973 ;
evos('SkLab','a_Chem','SSA') = 2060.499756 ;
evos('SkLab','a_Chem','ROW') = 58624.01562 ;
evos('SkLab','a_Manuf','USA') = 149504.2031 ;
evos('SkLab','a_Manuf','EU_28') = 260536.6406 ;
evos('SkLab','a_Manuf','CHN') = 141769.6875 ;
evos('SkLab','a_Manuf','JPN') = 60172.3125 ;
evos('SkLab','a_Manuf','IND') = 6600.001465 ;
evos('SkLab','a_Manuf','SSA') = 11167.17676 ;
evos('SkLab','a_Manuf','ROW') = 282783.4375 ;
evos('SkLab','a_ForestFish','USA') = 6610.771484 ;
evos('SkLab','a_ForestFish','EU_28') = 1193.836426 ;
evos('SkLab','a_ForestFish','CHN') = 5775.733398 ;
evos('SkLab','a_ForestFish','JPN') = 77.60725403 ;
evos('SkLab','a_ForestFish','IND') = 48.28032303 ;
evos('SkLab','a_ForestFish','SSA') = 720.098877 ;
evos('SkLab','a_ForestFish','ROW') = 3788.522217 ;
evos('SkLab','a_Svces','USA') = 2846306.5 ;
evos('SkLab','a_Svces','EU_28') = 1767065.125 ;
evos('SkLab','a_Svces','CHN') = 1293143.625 ;
evos('SkLab','a_Svces','JPN') = 443392.7188 ;
evos('SkLab','a_Svces','IND') = 329585.7188 ;
evos('SkLab','a_Svces','SSA') = 191948.2188 ;
evos('SkLab','a_Svces','ROW') = 2775193.25 ;
evos('Capital','a_Rice','USA') = 899.2664795 ;
evos('Capital','a_Rice','EU_28') = 1220.541626 ;
evos('Capital','a_Rice','CHN') = 30170.00195 ;
evos('Capital','a_Rice','JPN') = 4269.348145 ;
evos('Capital','a_Rice','IND') = 4899.087402 ;
evos('Capital','a_Rice','SSA') = 4677.773438 ;
evos('Capital','a_Rice','ROW') = 50956.37109 ;
evos('Capital','a_Crops','USA') = 52233.46875 ;
evos('Capital','a_Crops','EU_28') = 47020.67578 ;
evos('Capital','a_Crops','CHN') = 144947.3594 ;
evos('Capital','a_Crops','JPN') = 5914.152344 ;
evos('Capital','a_Crops','IND') = 26714.86328 ;
evos('Capital','a_Crops','SSA') = 51421.68359 ;
evos('Capital','a_Crops','ROW') = 126290.7344 ;
evos('Capital','a_Livestock','USA') = 26776.75977 ;
evos('Capital','a_Livestock','EU_28') = 35731.85547 ;
evos('Capital','a_Livestock','CHN') = 51992.90234 ;
evos('Capital','a_Livestock','JPN') = 2620.058594 ;
evos('Capital','a_Livestock','IND') = 12126.41602 ;
evos('Capital','a_Livestock','SSA') = 13726.53027 ;
evos('Capital','a_Livestock','ROW') = 59332.20312 ;
evos('Capital','a_FoodProc','USA') = 127413.5156 ;
evos('Capital','a_FoodProc','EU_28') = 175741.9062 ;
evos('Capital','a_FoodProc','CHN') = 158517.7031 ;
evos('Capital','a_FoodProc','JPN') = 51734.89062 ;
evos('Capital','a_FoodProc','IND') = 16826.63086 ;
evos('Capital','a_FoodProc','SSA') = 70336.82812 ;
evos('Capital','a_FoodProc','ROW') = 379891.25 ;
evos('Capital','a_Energy','USA') = 211992.2344 ;
evos('Capital','a_Energy','EU_28') = 155744.4219 ;
evos('Capital','a_Energy','CHN') = 166394.8125 ;
evos('Capital','a_Energy','JPN') = 25948.23047 ;
evos('Capital','a_Energy','IND') = 46284.46875 ;
evos('Capital','a_Energy','SSA') = 63379.25391 ;
evos('Capital','a_Energy','ROW') = 1079842.5 ;
evos('Capital','a_Textiles','USA') = 5961.62207 ;
evos('Capital','a_Textiles','EU_28') = 47836.09766 ;
evos('Capital','a_Textiles','CHN') = 89720.94531 ;
evos('Capital','a_Textiles','JPN') = 1680.704346 ;
evos('Capital','a_Textiles','IND') = 25112.47266 ;
evos('Capital','a_Textiles','SSA') = 6512.75 ;
evos('Capital','a_Textiles','ROW') = 126658.5312 ;
evos('Capital','a_Chem','USA') = 223271.7031 ;
evos('Capital','a_Chem','EU_28') = 267023.7188 ;
evos('Capital','a_Chem','CHN') = 204679.4375 ;
evos('Capital','a_Chem','JPN') = 56697.92188 ;
evos('Capital','a_Chem','IND') = 44627.78125 ;
evos('Capital','a_Chem','SSA') = 10749.14746 ;
evos('Capital','a_Chem','ROW') = 330735.2812 ;
evos('Capital','a_Manuf','USA') = 613005.9375 ;
evos('Capital','a_Manuf','EU_28') = 682320.375 ;
evos('Capital','a_Manuf','CHN') = 832917.9375 ;
evos('Capital','a_Manuf','JPN') = 273266.1875 ;
evos('Capital','a_Manuf','IND') = 102703.3516 ;
evos('Capital','a_Manuf','SSA') = 69989.24219 ;
evos('Capital','a_Manuf','ROW') = 1210678.5 ;
evos('Capital','a_ForestFish','USA') = 8948.974609 ;
evos('Capital','a_ForestFish','EU_28') = 17928.60938 ;
evos('Capital','a_ForestFish','CHN') = 6619.425293 ;
evos('Capital','a_ForestFish','JPN') = 4652.766602 ;
evos('Capital','a_ForestFish','IND') = 20214.72656 ;
evos('Capital','a_ForestFish','SSA') = 19041.54492 ;
evos('Capital','a_ForestFish','ROW') = 62571.43359 ;
evos('Capital','a_Svces','USA') = 5677908 ;
evos('Capital','a_Svces','EU_28') = 5227559 ;
evos('Capital','a_Svces','CHN') = 2363882.25 ;
evos('Capital','a_Svces','JPN') = 1716712.5 ;
evos('Capital','a_Svces','IND') = 683025.9375 ;
evos('Capital','a_Svces','SSA') = 379166.7812 ;
evos('Capital','a_Svces','ROW') = 6445030.5 ;
evos('NatRes','a_Energy','USA') = 58615.78906 ;
evos('NatRes','a_Energy','EU_28') = 12277.47656 ;
evos('NatRes','a_Energy','CHN') = 47527.31641 ;
evos('NatRes','a_Energy','JPN') = 81.02777863 ;
evos('NatRes','a_Energy','IND') = 11375.4043 ;
evos('NatRes','a_Energy','SSA') = 27126.47266 ;
evos('NatRes','a_Energy','ROW') = 384213.375 ;
evos('NatRes','a_Manuf','USA') = 14171.77539 ;
evos('NatRes','a_Manuf','EU_28') = 6557.898438 ;
evos('NatRes','a_Manuf','CHN') = 13355.99902 ;
evos('NatRes','a_Manuf','JPN') = 855.612915 ;
evos('NatRes','a_Manuf','IND') = 2445.664551 ;
evos('NatRes','a_Manuf','SSA') = 4909.294922 ;
evos('NatRes','a_Manuf','ROW') = 38005.75 ;
evos('NatRes','a_ForestFish','USA') = 5071.396973 ;
evos('NatRes','a_ForestFish','EU_28') = 7891.441406 ;
evos('NatRes','a_ForestFish','CHN') = 39613.80469 ;
evos('NatRes','a_ForestFish','JPN') = 2955.696777 ;
evos('NatRes','a_ForestFish','IND') = 10723.17383 ;
evos('NatRes','a_ForestFish','SSA') = 9361.168945 ;
evos('NatRes','a_ForestFish','ROW') = 40157.16797 ;

* vxsb data (490 cells)
vxsb('c_Rice','USA','USA') = 1.999999995e-06 ;
vxsb('c_Rice','USA','EU_28') = 44.97357559 ;
vxsb('c_Rice','USA','CHN') = 1.263966799 ;
vxsb('c_Rice','USA','JPN') = 214.5385895 ;
vxsb('c_Rice','USA','IND') = 1.709350109 ;
vxsb('c_Rice','USA','SSA') = 67.43743134 ;
vxsb('c_Rice','USA','ROW') = 1443.024658 ;
vxsb('c_Rice','EU_28','USA') = 25.27398682 ;
vxsb('c_Rice','EU_28','EU_28') = 1302.710693 ;
vxsb('c_Rice','EU_28','CHN') = 0.2879837453 ;
vxsb('c_Rice','EU_28','JPN') = 0.2021557838 ;
vxsb('c_Rice','EU_28','IND') = 0.9261800647 ;
vxsb('c_Rice','EU_28','SSA') = 9.387452126 ;
vxsb('c_Rice','EU_28','ROW') = 236.8567505 ;
vxsb('c_Rice','CHN','USA') = 3.625701189 ;
vxsb('c_Rice','CHN','EU_28') = 2.09355998 ;
vxsb('c_Rice','CHN','CHN') = 1.999999995e-06 ;
vxsb('c_Rice','CHN','JPN') = 3.507432699 ;
vxsb('c_Rice','CHN','IND') = 0.7717570066 ;
vxsb('c_Rice','CHN','SSA') = 330.4960938 ;
vxsb('c_Rice','CHN','ROW') = 308.3541565 ;
vxsb('c_Rice','JPN','USA') = 2.930562019 ;
vxsb('c_Rice','JPN','EU_28') = 3.004138708 ;
vxsb('c_Rice','JPN','CHN') = 2.286667347 ;
vxsb('c_Rice','JPN','JPN') = 1.999999995e-06 ;
vxsb('c_Rice','JPN','IND') = 0.03325640783 ;
vxsb('c_Rice','JPN','SSA') = 8.922866821 ;
vxsb('c_Rice','JPN','ROW') = 23.54179001 ;
vxsb('c_Rice','IND','USA') = 185.7251282 ;
vxsb('c_Rice','IND','EU_28') = 418.2321167 ;
vxsb('c_Rice','IND','CHN') = 0.5097866654 ;
vxsb('c_Rice','IND','JPN') = 0.6821960807 ;
vxsb('c_Rice','IND','IND') = 1.999999995e-06 ;
vxsb('c_Rice','IND','SSA') = 1857.39917 ;
vxsb('c_Rice','IND','ROW') = 4946.1875 ;
vxsb('c_Rice','SSA','USA') = 1.417518735 ;
vxsb('c_Rice','SSA','EU_28') = 4.179264069 ;
vxsb('c_Rice','SSA','CHN') = 2.087674379 ;
vxsb('c_Rice','SSA','JPN') = 0.229634881 ;
vxsb('c_Rice','SSA','IND') = 0.9178020954 ;
vxsb('c_Rice','SSA','SSA') = 159.8452301 ;
vxsb('c_Rice','SSA','ROW') = 3.688732147 ;
vxsb('c_Rice','ROW','USA') = 491.8948975 ;
vxsb('c_Rice','ROW','EU_28') = 807.8592529 ;
vxsb('c_Rice','ROW','CHN') = 1877.578979 ;
vxsb('c_Rice','ROW','JPN') = 150.1592102 ;
vxsb('c_Rice','ROW','IND') = 6.683067799 ;
vxsb('c_Rice','ROW','SSA') = 4006.174316 ;
vxsb('c_Rice','ROW','ROW') = 5180.779785 ;
vxsb('c_Crops','USA','USA') = 7.000000096e-06 ;
vxsb('c_Crops','USA','EU_28') = 6003.389648 ;
vxsb('c_Crops','USA','CHN') = 15919.92969 ;
vxsb('c_Crops','USA','JPN') = 6028.73877 ;
vxsb('c_Crops','USA','IND') = 1399.672363 ;
vxsb('c_Crops','USA','SSA') = 1081.456909 ;
vxsb('c_Crops','USA','ROW') = 38439.23828 ;
vxsb('c_Crops','EU_28','USA') = 1171.49353 ;
vxsb('c_Crops','EU_28','EU_28') = 74566.375 ;
vxsb('c_Crops','EU_28','CHN') = 483.0260315 ;
vxsb('c_Crops','EU_28','JPN') = 267.2954712 ;
vxsb('c_Crops','EU_28','IND') = 297.4518433 ;
vxsb('c_Crops','EU_28','SSA') = 1939.92749 ;
vxsb('c_Crops','EU_28','ROW') = 14962.14258 ;
vxsb('c_Crops','CHN','USA') = 672.7272339 ;
vxsb('c_Crops','CHN','EU_28') = 1255.498779 ;
vxsb('c_Crops','CHN','CHN') = 7.000000096e-06 ;
vxsb('c_Crops','CHN','JPN') = 1333.206055 ;
vxsb('c_Crops','CHN','IND') = 245.3829956 ;
vxsb('c_Crops','CHN','SSA') = 159.2085114 ;
vxsb('c_Crops','CHN','ROW') = 12313.98828 ;
vxsb('c_Crops','JPN','USA') = 45.87311935 ;
vxsb('c_Crops','JPN','EU_28') = 47.68291473 ;
vxsb('c_Crops','JPN','CHN') = 141.1171875 ;
vxsb('c_Crops','JPN','JPN') = 7.000000096e-06 ;
vxsb('c_Crops','JPN','IND') = 1.291500807 ;
vxsb('c_Crops','JPN','SSA') = 0.3904442489 ;
vxsb('c_Crops','JPN','ROW') = 311.1679688 ;
vxsb('c_Crops','IND','USA') = 824.0525513 ;
vxsb('c_Crops','IND','EU_28') = 1461.367676 ;
vxsb('c_Crops','IND','CHN') = 417.6261597 ;
vxsb('c_Crops','IND','JPN') = 164.7831421 ;
vxsb('c_Crops','IND','IND') = 7.000000096e-06 ;
vxsb('c_Crops','IND','SSA') = 151.2463837 ;
vxsb('c_Crops','IND','ROW') = 6685.890625 ;
vxsb('c_Crops','SSA','USA') = 2183.51709 ;
vxsb('c_Crops','SSA','EU_28') = 9538.619141 ;
vxsb('c_Crops','SSA','CHN') = 1492.188477 ;
vxsb('c_Crops','SSA','JPN') = 660.302124 ;
vxsb('c_Crops','SSA','IND') = 2236.642578 ;
vxsb('c_Crops','SSA','SSA') = 2501.776123 ;
vxsb('c_Crops','SSA','ROW') = 10894.4375 ;
vxsb('c_Crops','ROW','USA') = 34864.45703 ;
vxsb('c_Crops','ROW','EU_28') = 35480.33984 ;
vxsb('c_Crops','ROW','CHN') = 39501.73828 ;
vxsb('c_Crops','ROW','JPN') = 7739.811523 ;
vxsb('c_Crops','ROW','IND') = 5245.16748 ;
vxsb('c_Crops','ROW','SSA') = 3715.770752 ;
vxsb('c_Crops','ROW','ROW') = 83940.67969 ;
vxsb('c_Livestock','USA','USA') = 3.99999999e-06 ;
vxsb('c_Livestock','USA','EU_28') = 489.8058167 ;
vxsb('c_Livestock','USA','CHN') = 1243.558228 ;
vxsb('c_Livestock','USA','JPN') = 117.396553 ;
vxsb('c_Livestock','USA','IND') = 8.432930946 ;
vxsb('c_Livestock','USA','SSA') = 16.16414261 ;
vxsb('c_Livestock','USA','ROW') = 2310.008057 ;
vxsb('c_Livestock','EU_28','USA') = 609.1591797 ;
vxsb('c_Livestock','EU_28','EU_28') = 14647.82617 ;
vxsb('c_Livestock','EU_28','CHN') = 1215.565063 ;
vxsb('c_Livestock','EU_28','JPN') = 178.1795044 ;
vxsb('c_Livestock','EU_28','IND') = 22.76642799 ;
vxsb('c_Livestock','EU_28','SSA') = 122.1440277 ;
vxsb('c_Livestock','EU_28','ROW') = 4194.852051 ;
vxsb('c_Livestock','CHN','USA') = 214.4929199 ;
vxsb('c_Livestock','CHN','EU_28') = 382.752533 ;
vxsb('c_Livestock','CHN','CHN') = 3.99999999e-06 ;
vxsb('c_Livestock','CHN','JPN') = 216.7125549 ;
vxsb('c_Livestock','CHN','IND') = 5.354556084 ;
vxsb('c_Livestock','CHN','SSA') = 29.04610443 ;
vxsb('c_Livestock','CHN','ROW') = 789.6785278 ;
vxsb('c_Livestock','JPN','USA') = 0.761225462 ;
vxsb('c_Livestock','JPN','EU_28') = 4.285318851 ;
vxsb('c_Livestock','JPN','CHN') = 15.75510025 ;
vxsb('c_Livestock','JPN','JPN') = 3.99999999e-06 ;
vxsb('c_Livestock','JPN','IND') = 0.1227647662 ;
vxsb('c_Livestock','JPN','SSA') = 0.2118977755 ;
vxsb('c_Livestock','JPN','ROW') = 143.2527008 ;
vxsb('c_Livestock','IND','USA') = 86.2745285 ;
vxsb('c_Livestock','IND','EU_28') = 15.39801598 ;
vxsb('c_Livestock','IND','CHN') = 3.220872641 ;
vxsb('c_Livestock','IND','JPN') = 13.32358074 ;
vxsb('c_Livestock','IND','IND') = 3.99999999e-06 ;
vxsb('c_Livestock','IND','SSA') = 3.116108179 ;
vxsb('c_Livestock','IND','ROW') = 155.0703888 ;
vxsb('c_Livestock','SSA','USA') = 39.07450867 ;
vxsb('c_Livestock','SSA','EU_28') = 187.0770264 ;
vxsb('c_Livestock','SSA','CHN') = 302.7624512 ;
vxsb('c_Livestock','SSA','JPN') = 10.45896816 ;
vxsb('c_Livestock','SSA','IND') = 18.77957535 ;
vxsb('c_Livestock','SSA','SSA') = 417.440155 ;
vxsb('c_Livestock','SSA','ROW') = 1194.701416 ;
vxsb('c_Livestock','ROW','USA') = 3171.66626 ;
vxsb('c_Livestock','ROW','EU_28') = 1687.04248 ;
vxsb('c_Livestock','ROW','CHN') = 4188.151367 ;
vxsb('c_Livestock','ROW','JPN') = 367.6002808 ;
vxsb('c_Livestock','ROW','IND') = 215.7967987 ;
vxsb('c_Livestock','ROW','SSA') = 77.15182495 ;
vxsb('c_Livestock','ROW','ROW') = 5838.499023 ;
vxsb('c_FoodProc','USA','USA') = 7.000000096e-06 ;
vxsb('c_FoodProc','USA','EU_28') = 6040.095215 ;
vxsb('c_FoodProc','USA','CHN') = 4124.189941 ;
vxsb('c_FoodProc','USA','JPN') = 8298.050781 ;
vxsb('c_FoodProc','USA','IND') = 504.3951416 ;
vxsb('c_FoodProc','USA','SSA') = 1052.822021 ;
vxsb('c_FoodProc','USA','ROW') = 58111.72656 ;
vxsb('c_FoodProc','EU_28','USA') = 23398.62891 ;
vxsb('c_FoodProc','EU_28','EU_28') = 289786.625 ;
vxsb('c_FoodProc','EU_28','CHN') = 13131.25 ;
vxsb('c_FoodProc','EU_28','JPN') = 9396.450195 ;
vxsb('c_FoodProc','EU_28','IND') = 581.539917 ;
vxsb('c_FoodProc','EU_28','SSA') = 8628.206055 ;
vxsb('c_FoodProc','EU_28','ROW') = 71062.69531 ;
vxsb('c_FoodProc','CHN','USA') = 7007.264648 ;
vxsb('c_FoodProc','CHN','EU_28') = 6217.501953 ;
vxsb('c_FoodProc','CHN','CHN') = 7.000000096e-06 ;
vxsb('c_FoodProc','CHN','JPN') = 7412.940918 ;
vxsb('c_FoodProc','CHN','IND') = 225.4461823 ;
vxsb('c_FoodProc','CHN','SSA') = 1941.929077 ;
vxsb('c_FoodProc','CHN','ROW') = 26301.56641 ;
vxsb('c_FoodProc','JPN','USA') = 858.5811157 ;
vxsb('c_FoodProc','JPN','EU_28') = 295.9664612 ;
vxsb('c_FoodProc','JPN','CHN') = 664.1581421 ;
vxsb('c_FoodProc','JPN','JPN') = 7.000000096e-06 ;
vxsb('c_FoodProc','JPN','IND') = 6.02203989 ;
vxsb('c_FoodProc','JPN','SSA') = 89.887146 ;
vxsb('c_FoodProc','JPN','ROW') = 3225.572266 ;
vxsb('c_FoodProc','IND','USA') = 3681.194824 ;
vxsb('c_FoodProc','IND','EU_28') = 2649.042969 ;
vxsb('c_FoodProc','IND','CHN') = 628.9592285 ;
vxsb('c_FoodProc','IND','JPN') = 695.612793 ;
vxsb('c_FoodProc','IND','IND') = 7.000000096e-06 ;
vxsb('c_FoodProc','IND','SSA') = 1075.758057 ;
vxsb('c_FoodProc','IND','ROW') = 9508.731445 ;
vxsb('c_FoodProc','SSA','USA') = 689.4959106 ;
vxsb('c_FoodProc','SSA','EU_28') = 6891.169434 ;
vxsb('c_FoodProc','SSA','CHN') = 1188.367432 ;
vxsb('c_FoodProc','SSA','JPN') = 453.3733521 ;
vxsb('c_FoodProc','SSA','IND') = 73.99943542 ;
vxsb('c_FoodProc','SSA','SSA') = 8452.591797 ;
vxsb('c_FoodProc','SSA','ROW') = 2975.86499 ;
vxsb('c_FoodProc','ROW','USA') = 67380.92188 ;
vxsb('c_FoodProc','ROW','EU_28') = 58754.59375 ;
vxsb('c_FoodProc','ROW','CHN') = 34278.93359 ;
vxsb('c_FoodProc','ROW','JPN') = 26121.16406 ;
vxsb('c_FoodProc','ROW','IND') = 13684.03809 ;
vxsb('c_FoodProc','ROW','SSA') = 15051.66797 ;
vxsb('c_FoodProc','ROW','ROW') = 181513.5938 ;
vxsb('c_Energy','USA','USA') = 6.000000212e-06 ;
vxsb('c_Energy','USA','EU_28') = 17274.26758 ;
vxsb('c_Energy','USA','CHN') = 8890.428711 ;
vxsb('c_Energy','USA','JPN') = 6424.569824 ;
vxsb('c_Energy','USA','IND') = 2949.366699 ;
vxsb('c_Energy','USA','SSA') = 2227.437012 ;
vxsb('c_Energy','USA','ROW') = 106683.1406 ;
vxsb('c_Energy','EU_28','USA') = 9159.887695 ;
vxsb('c_Energy','EU_28','EU_28') = 93616.85156 ;
vxsb('c_Energy','EU_28','CHN') = 2122.032471 ;
vxsb('c_Energy','EU_28','JPN') = 124.0232773 ;
vxsb('c_Energy','EU_28','IND') = 477.4463196 ;
vxsb('c_Energy','EU_28','SSA') = 9902.011719 ;
vxsb('c_Energy','EU_28','ROW') = 39961.36719 ;
vxsb('c_Energy','CHN','USA') = 2249.4729 ;
vxsb('c_Energy','CHN','EU_28') = 1136.972412 ;
vxsb('c_Energy','CHN','CHN') = 6.000000212e-06 ;
vxsb('c_Energy','CHN','JPN') = 1177.370972 ;
vxsb('c_Energy','CHN','IND') = 1034.238281 ;
vxsb('c_Energy','CHN','SSA') = 3167.375 ;
vxsb('c_Energy','CHN','ROW') = 22290.38672 ;
vxsb('c_Energy','JPN','USA') = 808.0050659 ;
vxsb('c_Energy','JPN','EU_28') = 237.9857178 ;
vxsb('c_Energy','JPN','CHN') = 1136.789795 ;
vxsb('c_Energy','JPN','JPN') = 6.000000212e-06 ;
vxsb('c_Energy','JPN','IND') = 213.1560364 ;
vxsb('c_Energy','JPN','SSA') = 14.86950397 ;
vxsb('c_Energy','JPN','ROW') = 6658.386719 ;
vxsb('c_Energy','IND','USA') = 3604.374268 ;
vxsb('c_Energy','IND','EU_28') = 1867.522095 ;
vxsb('c_Energy','IND','CHN') = 504.6731873 ;
vxsb('c_Energy','IND','JPN') = 1240.539185 ;
vxsb('c_Energy','IND','IND') = 6.000000212e-06 ;
vxsb('c_Energy','IND','SSA') = 2457.116455 ;
vxsb('c_Energy','IND','ROW') = 13651.46387 ;
vxsb('c_Energy','SSA','USA') = 11351.80371 ;
vxsb('c_Energy','SSA','EU_28') = 20940.73828 ;
vxsb('c_Energy','SSA','CHN') = 29591.00195 ;
vxsb('c_Energy','SSA','JPN') = 1081.455688 ;
vxsb('c_Energy','SSA','IND') = 15719.85254 ;
vxsb('c_Energy','SSA','SSA') = 10998.92871 ;
vxsb('c_Energy','SSA','ROW') = 16123.48926 ;
vxsb('c_Energy','ROW','USA') = 179528.625 ;
vxsb('c_Energy','ROW','EU_28') = 336092.4062 ;
vxsb('c_Energy','ROW','CHN') = 179737.7344 ;
vxsb('c_Energy','ROW','JPN') = 127864.9609 ;
vxsb('c_Energy','ROW','IND') = 100292.2188 ;
vxsb('c_Energy','ROW','SSA') = 17784.66406 ;
vxsb('c_Energy','ROW','ROW') = 396160.5 ;
vxsb('c_Textiles','USA','USA') = 3.000000106e-06 ;
vxsb('c_Textiles','USA','EU_28') = 2945.998779 ;
vxsb('c_Textiles','USA','CHN') = 1628.699097 ;
vxsb('c_Textiles','USA','JPN') = 473.0762634 ;
vxsb('c_Textiles','USA','IND') = 190.5631256 ;
vxsb('c_Textiles','USA','SSA') = 238.4391174 ;
vxsb('c_Textiles','USA','ROW') = 20115.61133 ;
vxsb('c_Textiles','EU_28','USA') = 10732.72168 ;
vxsb('c_Textiles','EU_28','EU_28') = 179170.6562 ;
vxsb('c_Textiles','EU_28','CHN') = 9241.160156 ;
vxsb('c_Textiles','EU_28','JPN') = 4686.79248 ;
vxsb('c_Textiles','EU_28','IND') = 715.6351929 ;
vxsb('c_Textiles','EU_28','SSA') = 1948.80481 ;
vxsb('c_Textiles','EU_28','ROW') = 45067.29297 ;
vxsb('c_Textiles','CHN','USA') = 72444.53125 ;
vxsb('c_Textiles','CHN','EU_28') = 69866.30469 ;
vxsb('c_Textiles','CHN','CHN') = 3.000000106e-06 ;
vxsb('c_Textiles','CHN','JPN') = 31034.14648 ;
vxsb('c_Textiles','CHN','IND') = 6096.498047 ;
vxsb('c_Textiles','CHN','SSA') = 19423.98242 ;
vxsb('c_Textiles','CHN','ROW') = 167993.8281 ;
vxsb('c_Textiles','JPN','USA') = 688.2854614 ;
vxsb('c_Textiles','JPN','EU_28') = 802.5441284 ;
vxsb('c_Textiles','JPN','CHN') = 3390.571289 ;
vxsb('c_Textiles','JPN','JPN') = 3.000000106e-06 ;
vxsb('c_Textiles','JPN','IND') = 116.3020172 ;
vxsb('c_Textiles','JPN','SSA') = 166.7913208 ;
vxsb('c_Textiles','JPN','ROW') = 3706.44751 ;
vxsb('c_Textiles','IND','USA') = 9294.326172 ;
vxsb('c_Textiles','IND','EU_28') = 12595.98438 ;
vxsb('c_Textiles','IND','CHN') = 2197.755859 ;
vxsb('c_Textiles','IND','JPN') = 567.5150146 ;
vxsb('c_Textiles','IND','IND') = 3.000000106e-06 ;
vxsb('c_Textiles','IND','SSA') = 2163.53418 ;
vxsb('c_Textiles','IND','ROW') = 15447.76562 ;
vxsb('c_Textiles','SSA','USA') = 1232.49353 ;
vxsb('c_Textiles','SSA','EU_28') = 1427.046143 ;
vxsb('c_Textiles','SSA','CHN') = 304.7578735 ;
vxsb('c_Textiles','SSA','JPN') = 24.65365219 ;
vxsb('c_Textiles','SSA','IND') = 59.2138176 ;
vxsb('c_Textiles','SSA','SSA') = 2209.59375 ;
vxsb('c_Textiles','SSA','ROW') = 447.1387024 ;
vxsb('c_Textiles','ROW','USA') = 75102.10938 ;
vxsb('c_Textiles','ROW','EU_28') = 93887.14062 ;
vxsb('c_Textiles','ROW','CHN') = 24385.22852 ;
vxsb('c_Textiles','ROW','JPN') = 15001.34082 ;
vxsb('c_Textiles','ROW','IND') = 2305.706055 ;
vxsb('c_Textiles','ROW','SSA') = 2777.587158 ;
vxsb('c_Textiles','ROW','ROW') = 84717.85938 ;
vxsb('c_Chem','USA','USA') = 3.000000106e-06 ;
vxsb('c_Chem','USA','EU_28') = 57563.60938 ;
vxsb('c_Chem','USA','CHN') = 18629.79688 ;
vxsb('c_Chem','USA','JPN') = 13690.64453 ;
vxsb('c_Chem','USA','IND') = 4051.425293 ;
vxsb('c_Chem','USA','SSA') = 1950.41687 ;
vxsb('c_Chem','USA','ROW') = 142658.2344 ;
vxsb('c_Chem','EU_28','USA') = 98441.82031 ;
vxsb('c_Chem','EU_28','EU_28') = 589822.75 ;
vxsb('c_Chem','EU_28','CHN') = 47310.01172 ;
vxsb('c_Chem','EU_28','JPN') = 23066.14844 ;
vxsb('c_Chem','EU_28','IND') = 7210.421875 ;
vxsb('c_Chem','EU_28','SSA') = 13802.81738 ;
vxsb('c_Chem','EU_28','ROW') = 218808.1562 ;
vxsb('c_Chem','CHN','USA') = 31703.88867 ;
vxsb('c_Chem','CHN','EU_28') = 28238.19336 ;
vxsb('c_Chem','CHN','CHN') = 3.000000106e-06 ;
vxsb('c_Chem','CHN','JPN') = 13877.27832 ;
vxsb('c_Chem','CHN','IND') = 12627.6543 ;
vxsb('c_Chem','CHN','SSA') = 9186.109375 ;
vxsb('c_Chem','CHN','ROW') = 100932.1562 ;
vxsb('c_Chem','JPN','USA') = 12140.72852 ;
vxsb('c_Chem','JPN','EU_28') = 9335.737305 ;
vxsb('c_Chem','JPN','CHN') = 31360.16602 ;
vxsb('c_Chem','JPN','JPN') = 3.000000106e-06 ;
vxsb('c_Chem','JPN','IND') = 2003.283813 ;
vxsb('c_Chem','JPN','SSA') = 490.5314941 ;
vxsb('c_Chem','JPN','ROW') = 45184.35156 ;
vxsb('c_Chem','IND','USA') = 10060.4209 ;
vxsb('c_Chem','IND','EU_28') = 8072.149414 ;
vxsb('c_Chem','IND','CHN') = 2842.82666 ;
vxsb('c_Chem','IND','JPN') = 899.7098999 ;
vxsb('c_Chem','IND','IND') = 3.000000106e-06 ;
vxsb('c_Chem','IND','SSA') = 4734.601074 ;
vxsb('c_Chem','IND','ROW') = 20227.42578 ;
vxsb('c_Chem','SSA','USA') = 1054.388794 ;
vxsb('c_Chem','SSA','EU_28') = 2589.791992 ;
vxsb('c_Chem','SSA','CHN') = 607.9290161 ;
vxsb('c_Chem','SSA','JPN') = 86.35708618 ;
vxsb('c_Chem','SSA','IND') = 750.8780518 ;
vxsb('c_Chem','SSA','SSA') = 6785.23291 ;
vxsb('c_Chem','SSA','ROW') = 2160.778809 ;
vxsb('c_Chem','ROW','USA') = 108219.5078 ;
vxsb('c_Chem','ROW','EU_28') = 124058.0625 ;
vxsb('c_Chem','ROW','CHN') = 129713.2969 ;
vxsb('c_Chem','ROW','JPN') = 25513.56836 ;
vxsb('c_Chem','ROW','IND') = 25410.16992 ;
vxsb('c_Chem','ROW','SSA') = 11398.6875 ;
vxsb('c_Chem','ROW','ROW') = 241794.875 ;
vxsb('c_Manuf','USA','USA') = 1.299999985e-05 ;
vxsb('c_Manuf','USA','EU_28') = 147518.3906 ;
vxsb('c_Manuf','USA','CHN') = 73166.96875 ;
vxsb('c_Manuf','USA','JPN') = 37466.13672 ;
vxsb('c_Manuf','USA','IND') = 16732.99609 ;
vxsb('c_Manuf','USA','SSA') = 7846.399414 ;
vxsb('c_Manuf','USA','ROW') = 579015.875 ;
vxsb('c_Manuf','EU_28','USA') = 261452.5938 ;
vxsb('c_Manuf','EU_28','EU_28') = 1889556.875 ;
vxsb('c_Manuf','EU_28','CHN') = 194633.8281 ;
vxsb('c_Manuf','EU_28','JPN') = 41259.33594 ;
vxsb('c_Manuf','EU_28','IND') = 34675.55469 ;
vxsb('c_Manuf','EU_28','SSA') = 41977.57812 ;
vxsb('c_Manuf','EU_28','ROW') = 717691.625 ;
vxsb('c_Manuf','CHN','USA') = 344823.2188 ;
vxsb('c_Manuf','CHN','EU_28') = 237306.7188 ;
vxsb('c_Manuf','CHN','CHN') = 1.299999985e-05 ;
vxsb('c_Manuf','CHN','JPN') = 120771.7969 ;
vxsb('c_Manuf','CHN','IND') = 55197.28125 ;
vxsb('c_Manuf','CHN','SSA') = 47418.73047 ;
vxsb('c_Manuf','CHN','ROW') = 677704.0625 ;
vxsb('c_Manuf','JPN','USA') = 117311.25 ;
vxsb('c_Manuf','JPN','EU_28') = 64193.13672 ;
vxsb('c_Manuf','JPN','CHN') = 160044.9844 ;
vxsb('c_Manuf','JPN','JPN') = 1.299999985e-05 ;
vxsb('c_Manuf','JPN','IND') = 7284.441895 ;
vxsb('c_Manuf','JPN','SSA') = 6220.371094 ;
vxsb('c_Manuf','JPN','ROW') = 246635.7344 ;
vxsb('c_Manuf','IND','USA') = 19836.42773 ;
vxsb('c_Manuf','IND','EU_28') = 20170.9082 ;
vxsb('c_Manuf','IND','CHN') = 11570.07324 ;
vxsb('c_Manuf','IND','JPN') = 1730.199585 ;
vxsb('c_Manuf','IND','IND') = 1.299999985e-05 ;
vxsb('c_Manuf','IND','SSA') = 6423.375977 ;
vxsb('c_Manuf','IND','ROW') = 62179.42969 ;
vxsb('c_Manuf','SSA','USA') = 7867.123047 ;
vxsb('c_Manuf','SSA','EU_28') = 28398.60742 ;
vxsb('c_Manuf','SSA','CHN') = 40102.85156 ;
vxsb('c_Manuf','SSA','JPN') = 5023.112305 ;
vxsb('c_Manuf','SSA','IND') = 11887.16602 ;
vxsb('c_Manuf','SSA','SSA') = 21093.43164 ;
vxsb('c_Manuf','SSA','ROW') = 41559.29688 ;
vxsb('c_Manuf','ROW','USA') = 684972 ;
vxsb('c_Manuf','ROW','EU_28') = 435463.0312 ;
vxsb('c_Manuf','ROW','CHN') = 675140.9375 ;
vxsb('c_Manuf','ROW','JPN') = 128928.5859 ;
vxsb('c_Manuf','ROW','IND') = 89220.94531 ;
vxsb('c_Manuf','ROW','SSA') = 30972.66211 ;
vxsb('c_Manuf','ROW','ROW') = 962144.375 ;
vxsb('c_ForestFish','USA','USA') = 1.999999995e-06 ;
vxsb('c_ForestFish','USA','EU_28') = 309.0090027 ;
vxsb('c_ForestFish','USA','CHN') = 759.1322021 ;
vxsb('c_ForestFish','USA','JPN') = 492.7588501 ;
vxsb('c_ForestFish','USA','IND') = 13.36308289 ;
vxsb('c_ForestFish','USA','SSA') = 2.570113182 ;
vxsb('c_ForestFish','USA','ROW') = 1021.2453 ;
vxsb('c_ForestFish','EU_28','USA') = 445.2124634 ;
vxsb('c_ForestFish','EU_28','EU_28') = 10978.31836 ;
vxsb('c_ForestFish','EU_28','CHN') = 650.5681763 ;
vxsb('c_ForestFish','EU_28','JPN') = 31.03465652 ;
vxsb('c_ForestFish','EU_28','IND') = 35.24705505 ;
vxsb('c_ForestFish','EU_28','SSA') = 20.65450859 ;
vxsb('c_ForestFish','EU_28','ROW') = 930.6312256 ;
vxsb('c_ForestFish','CHN','USA') = 30.90802002 ;
vxsb('c_ForestFish','CHN','EU_28') = 77.82155609 ;
vxsb('c_ForestFish','CHN','CHN') = 1.999999995e-06 ;
vxsb('c_ForestFish','CHN','JPN') = 441.2687683 ;
vxsb('c_ForestFish','CHN','IND') = 1.51383853 ;
vxsb('c_ForestFish','CHN','SSA') = 1.57301724 ;
vxsb('c_ForestFish','CHN','ROW') = 813.5484009 ;
vxsb('c_ForestFish','JPN','USA') = 47.61443329 ;
vxsb('c_ForestFish','JPN','EU_28') = 19.87048531 ;
vxsb('c_ForestFish','JPN','CHN') = 130.7776642 ;
vxsb('c_ForestFish','JPN','JPN') = 1.999999995e-06 ;
vxsb('c_ForestFish','JPN','IND') = 1.717635393 ;
vxsb('c_ForestFish','JPN','SSA') = 0.5552031398 ;
vxsb('c_ForestFish','JPN','ROW') = 334.0418701 ;
vxsb('c_ForestFish','IND','USA') = 40.87734604 ;
vxsb('c_ForestFish','IND','EU_28') = 63.67672729 ;
vxsb('c_ForestFish','IND','CHN') = 75.42963409 ;
vxsb('c_ForestFish','IND','JPN') = 5.160473347 ;
vxsb('c_ForestFish','IND','IND') = 1.999999995e-06 ;
vxsb('c_ForestFish','IND','SSA') = 4.449133396 ;
vxsb('c_ForestFish','IND','ROW') = 166.7712097 ;
vxsb('c_ForestFish','SSA','USA') = 59.67477036 ;
vxsb('c_ForestFish','SSA','EU_28') = 379.5172424 ;
vxsb('c_ForestFish','SSA','CHN') = 1857.954712 ;
vxsb('c_ForestFish','SSA','JPN') = 22.08677292 ;
vxsb('c_ForestFish','SSA','IND') = 176.8222961 ;
vxsb('c_ForestFish','SSA','SSA') = 150.3866272 ;
vxsb('c_ForestFish','SSA','ROW') = 529.0928345 ;
vxsb('c_ForestFish','ROW','USA') = 2665.548584 ;
vxsb('c_ForestFish','ROW','EU_28') = 6151.437988 ;
vxsb('c_ForestFish','ROW','CHN') = 7654.5625 ;
vxsb('c_ForestFish','ROW','JPN') = 1646.94104 ;
vxsb('c_ForestFish','ROW','IND') = 1024.547363 ;
vxsb('c_ForestFish','ROW','SSA') = 90.41984558 ;
vxsb('c_ForestFish','ROW','ROW') = 5442.82373 ;
vxsb('c_Svces','USA','USA') = 1.800000064e-05 ;
vxsb('c_Svces','USA','EU_28') = 223204.1875 ;
vxsb('c_Svces','USA','CHN') = 57874.48828 ;
vxsb('c_Svces','USA','JPN') = 42788.98047 ;
vxsb('c_Svces','USA','IND') = 19128.24609 ;
vxsb('c_Svces','USA','SSA') = 18268.01758 ;
vxsb('c_Svces','USA','ROW') = 342195.6562 ;
vxsb('c_Svces','EU_28','USA') = 199294.9219 ;
vxsb('c_Svces','EU_28','EU_28') = 1038910.938 ;
vxsb('c_Svces','EU_28','CHN') = 52260.73047 ;
vxsb('c_Svces','EU_28','JPN') = 32276.91992 ;
vxsb('c_Svces','EU_28','IND') = 17354.9375 ;
vxsb('c_Svces','EU_28','SSA') = 29422.0332 ;
vxsb('c_Svces','EU_28','ROW') = 431834.0312 ;
vxsb('c_Svces','CHN','USA') = 19002.94922 ;
vxsb('c_Svces','CHN','EU_28') = 35278.92969 ;
vxsb('c_Svces','CHN','CHN') = 1.800000064e-05 ;
vxsb('c_Svces','CHN','JPN') = 7994.644531 ;
vxsb('c_Svces','CHN','IND') = 2810.67749 ;
vxsb('c_Svces','CHN','SSA') = 4210.168945 ;
vxsb('c_Svces','CHN','ROW') = 105634.6719 ;
vxsb('c_Svces','JPN','USA') = 20040.07617 ;
vxsb('c_Svces','JPN','EU_28') = 18437.45898 ;
vxsb('c_Svces','JPN','CHN') = 15669.74121 ;
vxsb('c_Svces','JPN','JPN') = 1.800000064e-05 ;
vxsb('c_Svces','JPN','IND') = 984.914978 ;
vxsb('c_Svces','JPN','SSA') = 1856.799316 ;
vxsb('c_Svces','JPN','ROW') = 50100.16797 ;
vxsb('c_Svces','IND','USA') = 27956.55273 ;
vxsb('c_Svces','IND','EU_28') = 28240.13477 ;
vxsb('c_Svces','IND','CHN') = 2620.395752 ;
vxsb('c_Svces','IND','JPN') = 2652.44751 ;
vxsb('c_Svces','IND','IND') = 1.800000064e-05 ;
vxsb('c_Svces','IND','SSA') = 4550.882812 ;
vxsb('c_Svces','IND','ROW') = 43900.6875 ;
vxsb('c_Svces','SSA','USA') = 9115.646484 ;
vxsb('c_Svces','SSA','EU_28') = 23896.33984 ;
vxsb('c_Svces','SSA','CHN') = 5176.616699 ;
vxsb('c_Svces','SSA','JPN') = 1891.782471 ;
vxsb('c_Svces','SSA','IND') = 2612.720215 ;
vxsb('c_Svces','SSA','SSA') = 3836.831055 ;
vxsb('c_Svces','SSA','ROW') = 19791.97656 ;
vxsb('c_Svces','ROW','USA') = 271711.4062 ;
vxsb('c_Svces','ROW','EU_28') = 361803.8438 ;
vxsb('c_Svces','ROW','CHN') = 179073.5156 ;
vxsb('c_Svces','ROW','JPN') = 57829.41406 ;
vxsb('c_Svces','ROW','IND') = 33393.57812 ;
vxsb('c_Svces','ROW','SSA') = 26256.54492 ;
vxsb('c_Svces','ROW','ROW') = 442174.2812 ;

* vfob data (490 cells)
vfob('c_Rice','USA','USA') = 1.999999995e-06 ;
vfob('c_Rice','USA','EU_28') = 44.97689056 ;
vfob('c_Rice','USA','CHN') = 1.263971448 ;
vfob('c_Rice','USA','JPN') = 214.5385895 ;
vfob('c_Rice','USA','IND') = 1.709352136 ;
vfob('c_Rice','USA','SSA') = 67.43784332 ;
vfob('c_Rice','USA','ROW') = 1444.219849 ;
vfob('c_Rice','EU_28','USA') = 25.27382851 ;
vfob('c_Rice','EU_28','EU_28') = 1302.710693 ;
vfob('c_Rice','EU_28','CHN') = 0.2879816592 ;
vfob('c_Rice','EU_28','JPN') = 0.202155605 ;
vfob('c_Rice','EU_28','IND') = 0.9261784554 ;
vfob('c_Rice','EU_28','SSA') = 9.387120247 ;
vfob('c_Rice','EU_28','ROW') = 237.0394135 ;
vfob('c_Rice','CHN','USA') = 3.604462624 ;
vfob('c_Rice','CHN','EU_28') = 2.059485435 ;
vfob('c_Rice','CHN','CHN') = 1.999999995e-06 ;
vfob('c_Rice','CHN','JPN') = 3.498678446 ;
vfob('c_Rice','CHN','IND') = 0.7517479658 ;
vfob('c_Rice','CHN','SSA') = 330.4536743 ;
vfob('c_Rice','CHN','ROW') = 305.3183594 ;
vfob('c_Rice','JPN','USA') = 2.930562019 ;
vfob('c_Rice','JPN','EU_28') = 3.004138708 ;
vfob('c_Rice','JPN','CHN') = 2.286667347 ;
vfob('c_Rice','JPN','JPN') = 1.999999995e-06 ;
vfob('c_Rice','JPN','IND') = 0.0332564041 ;
vfob('c_Rice','JPN','SSA') = 8.922866821 ;
vfob('c_Rice','JPN','ROW') = 23.54179001 ;
vfob('c_Rice','IND','USA') = 185.7220612 ;
vfob('c_Rice','IND','EU_28') = 418.2257996 ;
vfob('c_Rice','IND','CHN') = 0.5064390898 ;
vfob('c_Rice','IND','JPN') = 0.6818506122 ;
vfob('c_Rice','IND','IND') = 1.999999995e-06 ;
vfob('c_Rice','IND','SSA') = 1857.385986 ;
vfob('c_Rice','IND','ROW') = 4945.712891 ;
vfob('c_Rice','SSA','USA') = 1.417819619 ;
vfob('c_Rice','SSA','EU_28') = 4.180170536 ;
vfob('c_Rice','SSA','CHN') = 2.087954283 ;
vfob('c_Rice','SSA','JPN') = 0.2296916842 ;
vfob('c_Rice','SSA','IND') = 0.9179609418 ;
vfob('c_Rice','SSA','SSA') = 159.8666229 ;
vfob('c_Rice','SSA','ROW') = 3.689422131 ;
vfob('c_Rice','ROW','USA') = 491.8952637 ;
vfob('c_Rice','ROW','EU_28') = 807.8812256 ;
vfob('c_Rice','ROW','CHN') = 1877.578613 ;
vfob('c_Rice','ROW','JPN') = 150.1592255 ;
vfob('c_Rice','ROW','IND') = 6.683083534 ;
vfob('c_Rice','ROW','SSA') = 4006.181396 ;
vfob('c_Rice','ROW','ROW') = 5181.436523 ;
vfob('c_Crops','USA','USA') = 7.000000096e-06 ;
vfob('c_Crops','USA','EU_28') = 6033.270508 ;
vfob('c_Crops','USA','CHN') = 16105.73145 ;
vfob('c_Crops','USA','JPN') = 6046.056152 ;
vfob('c_Crops','USA','IND') = 1399.682495 ;
vfob('c_Crops','USA','SSA') = 1081.620483 ;
vfob('c_Crops','USA','ROW') = 38550.00391 ;
vfob('c_Crops','EU_28','USA') = 1171.440674 ;
vfob('c_Crops','EU_28','EU_28') = 74566.375 ;
vfob('c_Crops','EU_28','CHN') = 483.4272461 ;
vfob('c_Crops','EU_28','JPN') = 267.2990723 ;
vfob('c_Crops','EU_28','IND') = 297.4772034 ;
vfob('c_Crops','EU_28','SSA') = 1939.93396 ;
vfob('c_Crops','EU_28','ROW') = 14962.94727 ;
vfob('c_Crops','CHN','USA') = 671.867981 ;
vfob('c_Crops','CHN','EU_28') = 1246.777954 ;
vfob('c_Crops','CHN','CHN') = 7.000000096e-06 ;
vfob('c_Crops','CHN','JPN') = 1329.477661 ;
vfob('c_Crops','CHN','IND') = 244.7827454 ;
vfob('c_Crops','CHN','SSA') = 159.1275635 ;
vfob('c_Crops','CHN','ROW') = 12273.76367 ;
vfob('c_Crops','JPN','USA') = 45.87311935 ;
vfob('c_Crops','JPN','EU_28') = 47.68291473 ;
vfob('c_Crops','JPN','CHN') = 141.1171875 ;
vfob('c_Crops','JPN','JPN') = 7.000000096e-06 ;
vfob('c_Crops','JPN','IND') = 1.291500807 ;
vfob('c_Crops','JPN','SSA') = 0.3904442489 ;
vfob('c_Crops','JPN','ROW') = 311.1679688 ;
vfob('c_Crops','IND','USA') = 827.1725464 ;
vfob('c_Crops','IND','EU_28') = 1464.447266 ;
vfob('c_Crops','IND','CHN') = 418.1096191 ;
vfob('c_Crops','IND','JPN') = 164.803009 ;
vfob('c_Crops','IND','IND') = 7.000000096e-06 ;
vfob('c_Crops','IND','SSA') = 151.582077 ;
vfob('c_Crops','IND','ROW') = 6702.408203 ;
vfob('c_Crops','SSA','USA') = 2183.541504 ;
vfob('c_Crops','SSA','EU_28') = 9538.519531 ;
vfob('c_Crops','SSA','CHN') = 1500.44043 ;
vfob('c_Crops','SSA','JPN') = 660.3609009 ;
vfob('c_Crops','SSA','IND') = 2236.673584 ;
vfob('c_Crops','SSA','SSA') = 2502.084473 ;
vfob('c_Crops','SSA','ROW') = 10895.07129 ;
vfob('c_Crops','ROW','USA') = 34859.29688 ;
vfob('c_Crops','ROW','EU_28') = 35535.13281 ;
vfob('c_Crops','ROW','CHN') = 40107.48438 ;
vfob('c_Crops','ROW','JPN') = 7735.448242 ;
vfob('c_Crops','ROW','IND') = 5244.922852 ;
vfob('c_Crops','ROW','SSA') = 3715.732422 ;
vfob('c_Crops','ROW','ROW') = 84070.27344 ;
vfob('c_Livestock','USA','USA') = 3.99999999e-06 ;
vfob('c_Livestock','USA','EU_28') = 489.7987061 ;
vfob('c_Livestock','USA','CHN') = 1243.518799 ;
vfob('c_Livestock','USA','JPN') = 117.3965454 ;
vfob('c_Livestock','USA','IND') = 8.426260948 ;
vfob('c_Livestock','USA','SSA') = 16.15662193 ;
vfob('c_Livestock','USA','ROW') = 2310.005371 ;
vfob('c_Livestock','EU_28','USA') = 609.1422119 ;
vfob('c_Livestock','EU_28','EU_28') = 14647.82617 ;
vfob('c_Livestock','EU_28','CHN') = 1214.94104 ;
vfob('c_Livestock','EU_28','JPN') = 178.1788483 ;
vfob('c_Livestock','EU_28','IND') = 22.77078819 ;
vfob('c_Livestock','EU_28','SSA') = 122.1336594 ;
vfob('c_Livestock','EU_28','ROW') = 4194.916016 ;
vfob('c_Livestock','CHN','USA') = 214.495224 ;
vfob('c_Livestock','CHN','EU_28') = 382.7563477 ;
vfob('c_Livestock','CHN','CHN') = 3.99999999e-06 ;
vfob('c_Livestock','CHN','JPN') = 216.7192383 ;
vfob('c_Livestock','CHN','IND') = 5.356408596 ;
vfob('c_Livestock','CHN','SSA') = 29.04741478 ;
vfob('c_Livestock','CHN','ROW') = 789.7025146 ;
vfob('c_Livestock','JPN','USA') = 0.761225462 ;
vfob('c_Livestock','JPN','EU_28') = 4.285318851 ;
vfob('c_Livestock','JPN','CHN') = 15.75510025 ;
vfob('c_Livestock','JPN','JPN') = 3.99999999e-06 ;
vfob('c_Livestock','JPN','IND') = 0.1227647588 ;
vfob('c_Livestock','JPN','SSA') = 0.2118977755 ;
vfob('c_Livestock','JPN','ROW') = 143.2527008 ;
vfob('c_Livestock','IND','USA') = 86.27463531 ;
vfob('c_Livestock','IND','EU_28') = 15.39832592 ;
vfob('c_Livestock','IND','CHN') = 3.220989227 ;
vfob('c_Livestock','IND','JPN') = 13.32359314 ;
vfob('c_Livestock','IND','IND') = 3.99999999e-06 ;
vfob('c_Livestock','IND','SSA') = 3.11619401 ;
vfob('c_Livestock','IND','ROW') = 155.0713806 ;
vfob('c_Livestock','SSA','USA') = 39.07648468 ;
vfob('c_Livestock','SSA','EU_28') = 187.076767 ;
vfob('c_Livestock','SSA','CHN') = 302.7650452 ;
vfob('c_Livestock','SSA','JPN') = 10.45958614 ;
vfob('c_Livestock','SSA','IND') = 18.77995491 ;
vfob('c_Livestock','SSA','SSA') = 417.4406738 ;
vfob('c_Livestock','SSA','ROW') = 1194.704224 ;
vfob('c_Livestock','ROW','USA') = 3171.648682 ;
vfob('c_Livestock','ROW','EU_28') = 1687.725464 ;
vfob('c_Livestock','ROW','CHN') = 4191.059082 ;
vfob('c_Livestock','ROW','JPN') = 367.5088806 ;
vfob('c_Livestock','ROW','IND') = 215.9279938 ;
vfob('c_Livestock','ROW','SSA') = 77.15149689 ;
vfob('c_Livestock','ROW','ROW') = 5839.054688 ;
vfob('c_FoodProc','USA','USA') = 7.000000096e-06 ;
vfob('c_FoodProc','USA','EU_28') = 6040.095215 ;
vfob('c_FoodProc','USA','CHN') = 4124.189941 ;
vfob('c_FoodProc','USA','JPN') = 8298.050781 ;
vfob('c_FoodProc','USA','IND') = 504.3951416 ;
vfob('c_FoodProc','USA','SSA') = 1052.822021 ;
vfob('c_FoodProc','USA','ROW') = 58111.72656 ;
vfob('c_FoodProc','EU_28','USA') = 23398.62891 ;
vfob('c_FoodProc','EU_28','EU_28') = 289786.625 ;
vfob('c_FoodProc','EU_28','CHN') = 13131.25 ;
vfob('c_FoodProc','EU_28','JPN') = 9396.450195 ;
vfob('c_FoodProc','EU_28','IND') = 581.539917 ;
vfob('c_FoodProc','EU_28','SSA') = 8628.206055 ;
vfob('c_FoodProc','EU_28','ROW') = 71062.69531 ;
vfob('c_FoodProc','CHN','USA') = 7007.264648 ;
vfob('c_FoodProc','CHN','EU_28') = 6217.501953 ;
vfob('c_FoodProc','CHN','CHN') = 7.000000096e-06 ;
vfob('c_FoodProc','CHN','JPN') = 7412.940918 ;
vfob('c_FoodProc','CHN','IND') = 225.4461823 ;
vfob('c_FoodProc','CHN','SSA') = 1941.929077 ;
vfob('c_FoodProc','CHN','ROW') = 26301.56641 ;
vfob('c_FoodProc','JPN','USA') = 858.5811157 ;
vfob('c_FoodProc','JPN','EU_28') = 295.9664612 ;
vfob('c_FoodProc','JPN','CHN') = 664.1581421 ;
vfob('c_FoodProc','JPN','JPN') = 7.000000096e-06 ;
vfob('c_FoodProc','JPN','IND') = 6.02203989 ;
vfob('c_FoodProc','JPN','SSA') = 89.887146 ;
vfob('c_FoodProc','JPN','ROW') = 3225.572266 ;
vfob('c_FoodProc','IND','USA') = 3681.194824 ;
vfob('c_FoodProc','IND','EU_28') = 2649.042969 ;
vfob('c_FoodProc','IND','CHN') = 628.9592285 ;
vfob('c_FoodProc','IND','JPN') = 695.612793 ;
vfob('c_FoodProc','IND','IND') = 7.000000096e-06 ;
vfob('c_FoodProc','IND','SSA') = 1075.758057 ;
vfob('c_FoodProc','IND','ROW') = 9508.731445 ;
vfob('c_FoodProc','SSA','USA') = 689.4959106 ;
vfob('c_FoodProc','SSA','EU_28') = 6891.169434 ;
vfob('c_FoodProc','SSA','CHN') = 1188.367432 ;
vfob('c_FoodProc','SSA','JPN') = 453.3733521 ;
vfob('c_FoodProc','SSA','IND') = 73.99943542 ;
vfob('c_FoodProc','SSA','SSA') = 8452.591797 ;
vfob('c_FoodProc','SSA','ROW') = 2975.86499 ;
vfob('c_FoodProc','ROW','USA') = 67373.6875 ;
vfob('c_FoodProc','ROW','EU_28') = 58743.66406 ;
vfob('c_FoodProc','ROW','CHN') = 34278.375 ;
vfob('c_FoodProc','ROW','JPN') = 26119.74023 ;
vfob('c_FoodProc','ROW','IND') = 13683.93457 ;
vfob('c_FoodProc','ROW','SSA') = 15050.92285 ;
vfob('c_FoodProc','ROW','ROW') = 181503.7344 ;
vfob('c_Energy','USA','USA') = 6.000000212e-06 ;
vfob('c_Energy','USA','EU_28') = 18447.85938 ;
vfob('c_Energy','USA','CHN') = 9279.05957 ;
vfob('c_Energy','USA','JPN') = 6799.443359 ;
vfob('c_Energy','USA','IND') = 3250.067383 ;
vfob('c_Energy','USA','SSA') = 2258.705566 ;
vfob('c_Energy','USA','ROW') = 108434.2812 ;
vfob('c_Energy','EU_28','USA') = 9830.814453 ;
vfob('c_Energy','EU_28','EU_28') = 93616.85156 ;
vfob('c_Energy','EU_28','CHN') = 2175.595947 ;
vfob('c_Energy','EU_28','JPN') = 127.4416199 ;
vfob('c_Energy','EU_28','IND') = 491.5715027 ;
vfob('c_Energy','EU_28','SSA') = 10286.3916 ;
vfob('c_Energy','EU_28','ROW') = 41725.30469 ;
vfob('c_Energy','CHN','USA') = 2643.189697 ;
vfob('c_Energy','CHN','EU_28') = 1324.661499 ;
vfob('c_Energy','CHN','CHN') = 6.000000212e-06 ;
vfob('c_Energy','CHN','JPN') = 1415.666138 ;
vfob('c_Energy','CHN','IND') = 1211.646484 ;
vfob('c_Energy','CHN','SSA') = 3708.278564 ;
vfob('c_Energy','CHN','ROW') = 25902.10547 ;
vfob('c_Energy','JPN','USA') = 808.0050659 ;
vfob('c_Energy','JPN','EU_28') = 237.9857178 ;
vfob('c_Energy','JPN','CHN') = 1136.789795 ;
vfob('c_Energy','JPN','JPN') = 6.000000212e-06 ;
vfob('c_Energy','JPN','IND') = 213.1560364 ;
vfob('c_Energy','JPN','SSA') = 14.86950397 ;
vfob('c_Energy','JPN','ROW') = 6658.386719 ;
vfob('c_Energy','IND','USA') = 4415.659668 ;
vfob('c_Energy','IND','EU_28') = 2283.45874 ;
vfob('c_Energy','IND','CHN') = 617.3285522 ;
vfob('c_Energy','IND','JPN') = 1520.084839 ;
vfob('c_Energy','IND','IND') = 6.000000212e-06 ;
vfob('c_Energy','IND','SSA') = 3006.854004 ;
vfob('c_Energy','IND','ROW') = 16522.48828 ;
vfob('c_Energy','SSA','USA') = 11352.24609 ;
vfob('c_Energy','SSA','EU_28') = 20954.48242 ;
vfob('c_Energy','SSA','CHN') = 29576.71484 ;
vfob('c_Energy','SSA','JPN') = 1081.477905 ;
vfob('c_Energy','SSA','IND') = 15726.13574 ;
vfob('c_Energy','SSA','SSA') = 11051.45117 ;
vfob('c_Energy','SSA','ROW') = 16112.59473 ;
vfob('c_Energy','ROW','USA') = 179825.2188 ;
vfob('c_Energy','ROW','EU_28') = 339356.9062 ;
vfob('c_Energy','ROW','CHN') = 180209.1875 ;
vfob('c_Energy','ROW','JPN') = 128177.8672 ;
vfob('c_Energy','ROW','IND') = 100509.4922 ;
vfob('c_Energy','ROW','SSA') = 17829.84961 ;
vfob('c_Energy','ROW','ROW') = 399147.5 ;
vfob('c_Textiles','USA','USA') = 3.000000106e-06 ;
vfob('c_Textiles','USA','EU_28') = 2964.635498 ;
vfob('c_Textiles','USA','CHN') = 1642.61145 ;
vfob('c_Textiles','USA','JPN') = 475.9830322 ;
vfob('c_Textiles','USA','IND') = 192.0359344 ;
vfob('c_Textiles','USA','SSA') = 239.2085724 ;
vfob('c_Textiles','USA','ROW') = 20179.44922 ;
vfob('c_Textiles','EU_28','USA') = 10754.29492 ;
vfob('c_Textiles','EU_28','EU_28') = 179170.6562 ;
vfob('c_Textiles','EU_28','CHN') = 9252.995117 ;
vfob('c_Textiles','EU_28','JPN') = 4694.84082 ;
vfob('c_Textiles','EU_28','IND') = 717.4725952 ;
vfob('c_Textiles','EU_28','SSA') = 1955.141602 ;
vfob('c_Textiles','EU_28','ROW') = 45178.74219 ;
vfob('c_Textiles','CHN','USA') = 72738.88281 ;
vfob('c_Textiles','CHN','EU_28') = 70089.96094 ;
vfob('c_Textiles','CHN','CHN') = 3.000000106e-06 ;
vfob('c_Textiles','CHN','JPN') = 31072.95312 ;
vfob('c_Textiles','CHN','IND') = 6160.269531 ;
vfob('c_Textiles','CHN','SSA') = 19553.92969 ;
vfob('c_Textiles','CHN','ROW') = 169048.6875 ;
vfob('c_Textiles','JPN','USA') = 688.2854614 ;
vfob('c_Textiles','JPN','EU_28') = 802.5441284 ;
vfob('c_Textiles','JPN','CHN') = 3390.571289 ;
vfob('c_Textiles','JPN','JPN') = 3.000000106e-06 ;
vfob('c_Textiles','JPN','IND') = 116.3020172 ;
vfob('c_Textiles','JPN','SSA') = 166.7913208 ;
vfob('c_Textiles','JPN','ROW') = 3706.44751 ;
vfob('c_Textiles','IND','USA') = 9613.743164 ;
vfob('c_Textiles','IND','EU_28') = 13055.37598 ;
vfob('c_Textiles','IND','CHN') = 2268.990967 ;
vfob('c_Textiles','IND','JPN') = 587.9018555 ;
vfob('c_Textiles','IND','IND') = 3.000000106e-06 ;
vfob('c_Textiles','IND','SSA') = 2236.47583 ;
vfob('c_Textiles','IND','ROW') = 15970.61133 ;
vfob('c_Textiles','SSA','USA') = 1272.611206 ;
vfob('c_Textiles','SSA','EU_28') = 1436.235718 ;
vfob('c_Textiles','SSA','CHN') = 311.7030029 ;
vfob('c_Textiles','SSA','JPN') = 25.40793991 ;
vfob('c_Textiles','SSA','IND') = 60.83529663 ;
vfob('c_Textiles','SSA','SSA') = 2261.151855 ;
vfob('c_Textiles','SSA','ROW') = 457.3328247 ;
vfob('c_Textiles','ROW','USA') = 75519.3125 ;
vfob('c_Textiles','ROW','EU_28') = 94375.78906 ;
vfob('c_Textiles','ROW','CHN') = 24532.20117 ;
vfob('c_Textiles','ROW','JPN') = 15072.65625 ;
vfob('c_Textiles','ROW','IND') = 2322.39502 ;
vfob('c_Textiles','ROW','SSA') = 2790.878174 ;
vfob('c_Textiles','ROW','ROW') = 85227.28125 ;
vfob('c_Chem','USA','USA') = 3.000000106e-06 ;
vfob('c_Chem','USA','EU_28') = 61509.09766 ;
vfob('c_Chem','USA','CHN') = 19329.29688 ;
vfob('c_Chem','USA','JPN') = 14474.16016 ;
vfob('c_Chem','USA','IND') = 4193.458984 ;
vfob('c_Chem','USA','SSA') = 2018.639038 ;
vfob('c_Chem','USA','ROW') = 145321.375 ;
vfob('c_Chem','EU_28','USA') = 98506.63281 ;
vfob('c_Chem','EU_28','EU_28') = 589822.75 ;
vfob('c_Chem','EU_28','CHN') = 47340.98828 ;
vfob('c_Chem','EU_28','JPN') = 23079.99414 ;
vfob('c_Chem','EU_28','IND') = 7215.681641 ;
vfob('c_Chem','EU_28','SSA') = 13816.08789 ;
vfob('c_Chem','EU_28','ROW') = 218975.7656 ;
vfob('c_Chem','CHN','USA') = 33171.83203 ;
vfob('c_Chem','CHN','EU_28') = 29630.6582 ;
vfob('c_Chem','CHN','CHN') = 3.000000106e-06 ;
vfob('c_Chem','CHN','JPN') = 14592.25195 ;
vfob('c_Chem','CHN','IND') = 13335.68066 ;
vfob('c_Chem','CHN','SSA') = 9634.904297 ;
vfob('c_Chem','CHN','ROW') = 106148.3359 ;
vfob('c_Chem','JPN','USA') = 12140.72852 ;
vfob('c_Chem','JPN','EU_28') = 9335.737305 ;
vfob('c_Chem','JPN','CHN') = 31360.16602 ;
vfob('c_Chem','JPN','JPN') = 3.000000106e-06 ;
vfob('c_Chem','JPN','IND') = 2003.283813 ;
vfob('c_Chem','JPN','SSA') = 490.5314941 ;
vfob('c_Chem','JPN','ROW') = 45184.35156 ;
vfob('c_Chem','IND','USA') = 10587.91699 ;
vfob('c_Chem','IND','EU_28') = 8452.287109 ;
vfob('c_Chem','IND','CHN') = 2947.882568 ;
vfob('c_Chem','IND','JPN') = 937.3740234 ;
vfob('c_Chem','IND','IND') = 3.000000106e-06 ;
vfob('c_Chem','IND','SSA') = 4987.414062 ;
vfob('c_Chem','IND','ROW') = 21129.52148 ;
vfob('c_Chem','SSA','USA') = 1060.599121 ;
vfob('c_Chem','SSA','EU_28') = 2630.519043 ;
vfob('c_Chem','SSA','CHN') = 617.5953979 ;
vfob('c_Chem','SSA','JPN') = 86.48426819 ;
vfob('c_Chem','SSA','IND') = 756.3842163 ;
vfob('c_Chem','SSA','SSA') = 6901.043457 ;
vfob('c_Chem','SSA','ROW') = 2189.395264 ;
vfob('c_Chem','ROW','USA') = 108570.1172 ;
vfob('c_Chem','ROW','EU_28') = 124727.1172 ;
vfob('c_Chem','ROW','CHN') = 130171.3594 ;
vfob('c_Chem','ROW','JPN') = 25664.9082 ;
vfob('c_Chem','ROW','IND') = 25570.86523 ;
vfob('c_Chem','ROW','SSA') = 11458.2832 ;
vfob('c_Chem','ROW','ROW') = 243235.8281 ;
vfob('c_Manuf','USA','USA') = 1.299999985e-05 ;
vfob('c_Manuf','USA','EU_28') = 154211.8438 ;
vfob('c_Manuf','USA','CHN') = 76487.80469 ;
vfob('c_Manuf','USA','JPN') = 39510.22656 ;
vfob('c_Manuf','USA','IND') = 17447.58984 ;
vfob('c_Manuf','USA','SSA') = 8182.056152 ;
vfob('c_Manuf','USA','ROW') = 591842.0625 ;
vfob('c_Manuf','EU_28','USA') = 261621.4375 ;
vfob('c_Manuf','EU_28','EU_28') = 1889556.875 ;
vfob('c_Manuf','EU_28','CHN') = 194711.0938 ;
vfob('c_Manuf','EU_28','JPN') = 41281.79688 ;
vfob('c_Manuf','EU_28','IND') = 34643.51953 ;
vfob('c_Manuf','EU_28','SSA') = 42005.53125 ;
vfob('c_Manuf','EU_28','ROW') = 718052.5 ;
vfob('c_Manuf','CHN','USA') = 350293.375 ;
vfob('c_Manuf','CHN','EU_28') = 241437.1406 ;
vfob('c_Manuf','CHN','CHN') = 1.299999985e-05 ;
vfob('c_Manuf','CHN','JPN') = 122221.8594 ;
vfob('c_Manuf','CHN','IND') = 55729.98047 ;
vfob('c_Manuf','CHN','SSA') = 48639.96875 ;
vfob('c_Manuf','CHN','ROW') = 688218.875 ;
vfob('c_Manuf','JPN','USA') = 117311.25 ;
vfob('c_Manuf','JPN','EU_28') = 64193.13672 ;
vfob('c_Manuf','JPN','CHN') = 160044.9844 ;
vfob('c_Manuf','JPN','JPN') = 1.299999985e-05 ;
vfob('c_Manuf','JPN','IND') = 7284.441895 ;
vfob('c_Manuf','JPN','SSA') = 6220.371094 ;
vfob('c_Manuf','JPN','ROW') = 246635.7344 ;
vfob('c_Manuf','IND','USA') = 20825.64648 ;
vfob('c_Manuf','IND','EU_28') = 21132.08594 ;
vfob('c_Manuf','IND','CHN') = 11978.26172 ;
vfob('c_Manuf','IND','JPN') = 1790.728394 ;
vfob('c_Manuf','IND','IND') = 1.299999985e-05 ;
vfob('c_Manuf','IND','SSA') = 6793.99707 ;
vfob('c_Manuf','IND','ROW') = 64948.75391 ;
vfob('c_Manuf','SSA','USA') = 7886.19873 ;
vfob('c_Manuf','SSA','EU_28') = 28611.78516 ;
vfob('c_Manuf','SSA','CHN') = 40339.07031 ;
vfob('c_Manuf','SSA','JPN') = 5026.380859 ;
vfob('c_Manuf','SSA','IND') = 11998.79004 ;
vfob('c_Manuf','SSA','SSA') = 21277.2793 ;
vfob('c_Manuf','SSA','ROW') = 42153.46875 ;
vfob('c_Manuf','ROW','USA') = 686990.0625 ;
vfob('c_Manuf','ROW','EU_28') = 438340.5625 ;
vfob('c_Manuf','ROW','CHN') = 677040 ;
vfob('c_Manuf','ROW','JPN') = 129822.2031 ;
vfob('c_Manuf','ROW','IND') = 89544 ;
vfob('c_Manuf','ROW','SSA') = 31307.16211 ;
vfob('c_Manuf','ROW','ROW') = 970280.875 ;
vfob('c_ForestFish','USA','USA') = 1.999999995e-06 ;
vfob('c_ForestFish','USA','EU_28') = 307.537262 ;
vfob('c_ForestFish','USA','CHN') = 750.0759277 ;
vfob('c_ForestFish','USA','JPN') = 485.0227051 ;
vfob('c_ForestFish','USA','IND') = 13.35292053 ;
vfob('c_ForestFish','USA','SSA') = 2.542307138 ;
vfob('c_ForestFish','USA','ROW') = 1021.110779 ;
vfob('c_ForestFish','EU_28','USA') = 446.888916 ;
vfob('c_ForestFish','EU_28','EU_28') = 10978.31836 ;
vfob('c_ForestFish','EU_28','CHN') = 649.3841553 ;
vfob('c_ForestFish','EU_28','JPN') = 31.08044052 ;
vfob('c_ForestFish','EU_28','IND') = 34.78451157 ;
vfob('c_ForestFish','EU_28','SSA') = 20.55836487 ;
vfob('c_ForestFish','EU_28','ROW') = 930.4381104 ;
vfob('c_ForestFish','CHN','USA') = 30.56794739 ;
vfob('c_ForestFish','CHN','EU_28') = 76.83499146 ;
vfob('c_ForestFish','CHN','CHN') = 1.999999995e-06 ;
vfob('c_ForestFish','CHN','JPN') = 441.0579834 ;
vfob('c_ForestFish','CHN','IND') = 1.508146882 ;
vfob('c_ForestFish','CHN','SSA') = 1.563876271 ;
vfob('c_ForestFish','CHN','ROW') = 814.1583252 ;
vfob('c_ForestFish','JPN','USA') = 47.61442947 ;
vfob('c_ForestFish','JPN','EU_28') = 19.8704834 ;
vfob('c_ForestFish','JPN','CHN') = 130.7776642 ;
vfob('c_ForestFish','JPN','JPN') = 1.999999995e-06 ;
vfob('c_ForestFish','JPN','IND') = 1.717635393 ;
vfob('c_ForestFish','JPN','SSA') = 0.5552030802 ;
vfob('c_ForestFish','JPN','ROW') = 334.0418396 ;
vfob('c_ForestFish','IND','USA') = 41.38765717 ;
vfob('c_ForestFish','IND','EU_28') = 64.56699371 ;
vfob('c_ForestFish','IND','CHN') = 76.60847473 ;
vfob('c_ForestFish','IND','JPN') = 5.239107132 ;
vfob('c_ForestFish','IND','IND') = 1.999999995e-06 ;
vfob('c_ForestFish','IND','SSA') = 4.50546217 ;
vfob('c_ForestFish','IND','ROW') = 167.5648499 ;
vfob('c_ForestFish','SSA','USA') = 59.93788147 ;
vfob('c_ForestFish','SSA','EU_28') = 383.5697021 ;
vfob('c_ForestFish','SSA','CHN') = 1904.595947 ;
vfob('c_ForestFish','SSA','JPN') = 22.51743126 ;
vfob('c_ForestFish','SSA','IND') = 177.6902466 ;
vfob('c_ForestFish','SSA','SSA') = 154.1670837 ;
vfob('c_ForestFish','SSA','ROW') = 550.7576904 ;
vfob('c_ForestFish','ROW','USA') = 2666.06665 ;
vfob('c_ForestFish','ROW','EU_28') = 6143.647949 ;
vfob('c_ForestFish','ROW','CHN') = 7677.151367 ;
vfob('c_ForestFish','ROW','JPN') = 1648.36084 ;
vfob('c_ForestFish','ROW','IND') = 1027.411865 ;
vfob('c_ForestFish','ROW','SSA') = 90.46655273 ;
vfob('c_ForestFish','ROW','ROW') = 5441.612793 ;
vfob('c_Svces','USA','USA') = 1.800000064e-05 ;
vfob('c_Svces','USA','EU_28') = 223204.1875 ;
vfob('c_Svces','USA','CHN') = 57874.48828 ;
vfob('c_Svces','USA','JPN') = 42788.98047 ;
vfob('c_Svces','USA','IND') = 19128.24609 ;
vfob('c_Svces','USA','SSA') = 18268.01758 ;
vfob('c_Svces','USA','ROW') = 342195.6562 ;
vfob('c_Svces','EU_28','USA') = 199294.9219 ;
vfob('c_Svces','EU_28','EU_28') = 1038910.938 ;
vfob('c_Svces','EU_28','CHN') = 52260.73047 ;
vfob('c_Svces','EU_28','JPN') = 32276.91992 ;
vfob('c_Svces','EU_28','IND') = 17354.9375 ;
vfob('c_Svces','EU_28','SSA') = 29422.0332 ;
vfob('c_Svces','EU_28','ROW') = 431834.0312 ;
vfob('c_Svces','CHN','USA') = 19002.94922 ;
vfob('c_Svces','CHN','EU_28') = 35278.92969 ;
vfob('c_Svces','CHN','CHN') = 1.800000064e-05 ;
vfob('c_Svces','CHN','JPN') = 7994.644531 ;
vfob('c_Svces','CHN','IND') = 2810.67749 ;
vfob('c_Svces','CHN','SSA') = 4210.168945 ;
vfob('c_Svces','CHN','ROW') = 105634.6719 ;
vfob('c_Svces','JPN','USA') = 20040.07617 ;
vfob('c_Svces','JPN','EU_28') = 18437.45898 ;
vfob('c_Svces','JPN','CHN') = 15669.74121 ;
vfob('c_Svces','JPN','JPN') = 1.800000064e-05 ;
vfob('c_Svces','JPN','IND') = 984.914978 ;
vfob('c_Svces','JPN','SSA') = 1856.799316 ;
vfob('c_Svces','JPN','ROW') = 50100.16797 ;
vfob('c_Svces','IND','USA') = 27956.55273 ;
vfob('c_Svces','IND','EU_28') = 28240.13477 ;
vfob('c_Svces','IND','CHN') = 2620.395752 ;
vfob('c_Svces','IND','JPN') = 2652.44751 ;
vfob('c_Svces','IND','IND') = 1.800000064e-05 ;
vfob('c_Svces','IND','SSA') = 4550.882812 ;
vfob('c_Svces','IND','ROW') = 43900.6875 ;
vfob('c_Svces','SSA','USA') = 9115.646484 ;
vfob('c_Svces','SSA','EU_28') = 23896.33984 ;
vfob('c_Svces','SSA','CHN') = 5176.616699 ;
vfob('c_Svces','SSA','JPN') = 1891.782471 ;
vfob('c_Svces','SSA','IND') = 2612.720215 ;
vfob('c_Svces','SSA','SSA') = 3836.831055 ;
vfob('c_Svces','SSA','ROW') = 19791.97656 ;
vfob('c_Svces','ROW','USA') = 271711.4062 ;
vfob('c_Svces','ROW','EU_28') = 361803.8438 ;
vfob('c_Svces','ROW','CHN') = 179073.5156 ;
vfob('c_Svces','ROW','JPN') = 57829.41406 ;
vfob('c_Svces','ROW','IND') = 33393.57812 ;
vfob('c_Svces','ROW','SSA') = 26256.54492 ;
vfob('c_Svces','ROW','ROW') = 442174.2812 ;

* vcif data (490 cells)
vcif('c_Rice','USA','USA') = 1.999999995e-06 ;
vcif('c_Rice','USA','EU_28') = 49.06710434 ;
vcif('c_Rice','USA','CHN') = 1.336138248 ;
vcif('c_Rice','USA','JPN') = 234.4432373 ;
vcif('c_Rice','USA','IND') = 1.849077225 ;
vcif('c_Rice','USA','SSA') = 73.68439484 ;
vcif('c_Rice','USA','ROW') = 1534.652344 ;
vcif('c_Rice','EU_28','USA') = 27.57825279 ;
vcif('c_Rice','EU_28','EU_28') = 1360.712036 ;
vcif('c_Rice','EU_28','CHN') = 0.2881236374 ;
vcif('c_Rice','EU_28','JPN') = 0.2179896086 ;
vcif('c_Rice','EU_28','IND') = 1.006360531 ;
vcif('c_Rice','EU_28','SSA') = 10.2446909 ;
vcif('c_Rice','EU_28','ROW') = 255.6999512 ;
vcif('c_Rice','CHN','USA') = 3.858866692 ;
vcif('c_Rice','CHN','EU_28') = 2.122078419 ;
vcif('c_Rice','CHN','CHN') = 1.999999995e-06 ;
vcif('c_Rice','CHN','JPN') = 3.790362358 ;
vcif('c_Rice','CHN','IND') = 0.7555149794 ;
vcif('c_Rice','CHN','SSA') = 361.0581055 ;
vcif('c_Rice','CHN','ROW') = 331.0325317 ;
vcif('c_Rice','JPN','USA') = 3.193453789 ;
vcif('c_Rice','JPN','EU_28') = 3.269026756 ;
vcif('c_Rice','JPN','CHN') = 2.391013145 ;
vcif('c_Rice','JPN','JPN') = 1.999999995e-06 ;
vcif('c_Rice','JPN','IND') = 0.03363240138 ;
vcif('c_Rice','JPN','SSA') = 9.74629879 ;
vcif('c_Rice','JPN','ROW') = 25.63937759 ;
vcif('c_Rice','IND','USA') = 202.9166412 ;
vcif('c_Rice','IND','EU_28') = 456.9573059 ;
vcif('c_Rice','IND','CHN') = 0.5073561072 ;
vcif('c_Rice','IND','JPN') = 0.7403765917 ;
vcif('c_Rice','IND','IND') = 1.999999995e-06 ;
vcif('c_Rice','IND','SSA') = 2029.685425 ;
vcif('c_Rice','IND','ROW') = 5402.431641 ;
vcif('c_Rice','SSA','USA') = 1.453320026 ;
vcif('c_Rice','SSA','EU_28') = 4.20027256 ;
vcif('c_Rice','SSA','CHN') = 2.087954283 ;
vcif('c_Rice','SSA','JPN') = 0.2296916842 ;
vcif('c_Rice','SSA','IND') = 0.9179609418 ;
vcif('c_Rice','SSA','SSA') = 174.2300415 ;
vcif('c_Rice','SSA','ROW') = 3.71612978 ;
vcif('c_Rice','ROW','USA') = 535.8032837 ;
vcif('c_Rice','ROW','EU_28') = 880.3384399 ;
vcif('c_Rice','ROW','CHN') = 2049.745117 ;
vcif('c_Rice','ROW','JPN') = 163.802948 ;
vcif('c_Rice','ROW','IND') = 6.755671501 ;
vcif('c_Rice','ROW','SSA') = 4377.462402 ;
vcif('c_Rice','ROW','ROW') = 5652.948242 ;
vcif('c_Crops','USA','USA') = 7.000000096e-06 ;
vcif('c_Crops','USA','EU_28') = 6342.673828 ;
vcif('c_Crops','USA','CHN') = 17350.08008 ;
vcif('c_Crops','USA','JPN') = 6682.10791 ;
vcif('c_Crops','USA','IND') = 1458.352661 ;
vcif('c_Crops','USA','SSA') = 1148.655273 ;
vcif('c_Crops','USA','ROW') = 41029.03516 ;
vcif('c_Crops','EU_28','USA') = 1265.922119 ;
vcif('c_Crops','EU_28','EU_28') = 80757.09375 ;
vcif('c_Crops','EU_28','CHN') = 522.0060425 ;
vcif('c_Crops','EU_28','JPN') = 282.8962097 ;
vcif('c_Crops','EU_28','IND') = 320.0098877 ;
vcif('c_Crops','EU_28','SSA') = 2081.263916 ;
vcif('c_Crops','EU_28','ROW') = 16243.62793 ;
vcif('c_Crops','CHN','USA') = 722.880188 ;
vcif('c_Crops','CHN','EU_28') = 1331.489258 ;
vcif('c_Crops','CHN','CHN') = 7.000000096e-06 ;
vcif('c_Crops','CHN','JPN') = 1478.715332 ;
vcif('c_Crops','CHN','IND') = 274.3277893 ;
vcif('c_Crops','CHN','SSA') = 172.9049072 ;
vcif('c_Crops','CHN','ROW') = 13906.69629 ;
vcif('c_Crops','JPN','USA') = 52.17575073 ;
vcif('c_Crops','JPN','EU_28') = 51.06281662 ;
vcif('c_Crops','JPN','CHN') = 148.7971954 ;
vcif('c_Crops','JPN','JPN') = 7.000000096e-06 ;
vcif('c_Crops','JPN','IND') = 1.329788804 ;
vcif('c_Crops','JPN','SSA') = 0.3999382257 ;
vcif('c_Crops','JPN','ROW') = 351.2736511 ;
vcif('c_Crops','IND','USA') = 876.1329956 ;
vcif('c_Crops','IND','EU_28') = 1547.924194 ;
vcif('c_Crops','IND','CHN') = 472.7420044 ;
vcif('c_Crops','IND','JPN') = 171.5207214 ;
vcif('c_Crops','IND','IND') = 7.000000096e-06 ;
vcif('c_Crops','IND','SSA') = 161.2894897 ;
vcif('c_Crops','IND','ROW') = 7135.337402 ;
vcif('c_Crops','SSA','USA') = 2250.392578 ;
vcif('c_Crops','SSA','EU_28') = 10375.26562 ;
vcif('c_Crops','SSA','CHN') = 1617.436401 ;
vcif('c_Crops','SSA','JPN') = 702.8994141 ;
vcif('c_Crops','SSA','IND') = 2317.199463 ;
vcif('c_Crops','SSA','SSA') = 2727.173096 ;
vcif('c_Crops','SSA','ROW') = 11552.11816 ;
vcif('c_Crops','ROW','USA') = 37456.33594 ;
vcif('c_Crops','ROW','EU_28') = 38917.1875 ;
vcif('c_Crops','ROW','CHN') = 43771.25 ;
vcif('c_Crops','ROW','JPN') = 8518.894531 ;
vcif('c_Crops','ROW','IND') = 5570.843262 ;
vcif('c_Crops','ROW','SSA') = 3972.125244 ;
vcif('c_Crops','ROW','ROW') = 91896.09375 ;
vcif('c_Livestock','USA','USA') = 3.99999999e-06 ;
vcif('c_Livestock','USA','EU_28') = 516.3571777 ;
vcif('c_Livestock','USA','CHN') = 1307.636108 ;
vcif('c_Livestock','USA','JPN') = 124.3979492 ;
vcif('c_Livestock','USA','IND') = 8.833126068 ;
vcif('c_Livestock','USA','SSA') = 23.35129738 ;
vcif('c_Livestock','USA','ROW') = 2469.760498 ;
vcif('c_Livestock','EU_28','USA') = 635.7582397 ;
vcif('c_Livestock','EU_28','EU_28') = 17415.18164 ;
vcif('c_Livestock','EU_28','CHN') = 1262.370972 ;
vcif('c_Livestock','EU_28','JPN') = 187.6554565 ;
vcif('c_Livestock','EU_28','IND') = 25.61420822 ;
vcif('c_Livestock','EU_28','SSA') = 153.0424194 ;
vcif('c_Livestock','EU_28','ROW') = 5813.776367 ;
vcif('c_Livestock','CHN','USA') = 225.3478851 ;
vcif('c_Livestock','CHN','EU_28') = 396.6674805 ;
vcif('c_Livestock','CHN','CHN') = 3.99999999e-06 ;
vcif('c_Livestock','CHN','JPN') = 225.9030609 ;
vcif('c_Livestock','CHN','IND') = 5.498160362 ;
vcif('c_Livestock','CHN','SSA') = 30.08186913 ;
vcif('c_Livestock','CHN','ROW') = 902.3123779 ;
vcif('c_Livestock','JPN','USA') = 0.7928496003 ;
vcif('c_Livestock','JPN','EU_28') = 4.523818016 ;
vcif('c_Livestock','JPN','CHN') = 16.85446548 ;
vcif('c_Livestock','JPN','JPN') = 3.99999999e-06 ;
vcif('c_Livestock','JPN','IND') = 0.1275717616 ;
vcif('c_Livestock','JPN','SSA') = 0.2170037627 ;
vcif('c_Livestock','JPN','ROW') = 151.4929504 ;
vcif('c_Livestock','IND','USA') = 89.97945404 ;
vcif('c_Livestock','IND','EU_28') = 16.20622444 ;
vcif('c_Livestock','IND','CHN') = 3.254530191 ;
vcif('c_Livestock','IND','JPN') = 14.5610342 ;
vcif('c_Livestock','IND','IND') = 3.99999999e-06 ;
vcif('c_Livestock','IND','SSA') = 3.261775017 ;
vcif('c_Livestock','IND','ROW') = 162.8080139 ;
vcif('c_Livestock','SSA','USA') = 40.79273605 ;
vcif('c_Livestock','SSA','EU_28') = 193.9913788 ;
vcif('c_Livestock','SSA','CHN') = 313.5618896 ;
vcif('c_Livestock','SSA','JPN') = 10.78339863 ;
vcif('c_Livestock','SSA','IND') = 19.18458939 ;
vcif('c_Livestock','SSA','SSA') = 688.9608154 ;
vcif('c_Livestock','SSA','ROW') = 1327.005249 ;
vcif('c_Livestock','ROW','USA') = 3219.150635 ;
vcif('c_Livestock','ROW','EU_28') = 1749.866333 ;
vcif('c_Livestock','ROW','CHN') = 4543.434082 ;
vcif('c_Livestock','ROW','JPN') = 426.214325 ;
vcif('c_Livestock','ROW','IND') = 224.4713898 ;
vcif('c_Livestock','ROW','SSA') = 85.39053345 ;
vcif('c_Livestock','ROW','ROW') = 8377.875977 ;
vcif('c_FoodProc','USA','USA') = 7.000000096e-06 ;
vcif('c_FoodProc','USA','EU_28') = 6365.067383 ;
vcif('c_FoodProc','USA','CHN') = 4328.779785 ;
vcif('c_FoodProc','USA','JPN') = 8714.887695 ;
vcif('c_FoodProc','USA','IND') = 523.387146 ;
vcif('c_FoodProc','USA','SSA') = 1103.50647 ;
vcif('c_FoodProc','USA','ROW') = 60325.29688 ;
vcif('c_FoodProc','EU_28','USA') = 24777.64062 ;
vcif('c_FoodProc','EU_28','EU_28') = 299926.3438 ;
vcif('c_FoodProc','EU_28','CHN') = 13801.71387 ;
vcif('c_FoodProc','EU_28','JPN') = 9854.014648 ;
vcif('c_FoodProc','EU_28','IND') = 609.6130371 ;
vcif('c_FoodProc','EU_28','SSA') = 9138.116211 ;
vcif('c_FoodProc','EU_28','ROW') = 74718.85156 ;
vcif('c_FoodProc','CHN','USA') = 7440.003418 ;
vcif('c_FoodProc','CHN','EU_28') = 6586.01123 ;
vcif('c_FoodProc','CHN','CHN') = 7.000000096e-06 ;
vcif('c_FoodProc','CHN','JPN') = 7847.18457 ;
vcif('c_FoodProc','CHN','IND') = 238.0043335 ;
vcif('c_FoodProc','CHN','SSA') = 2054.1521 ;
vcif('c_FoodProc','CHN','ROW') = 27869.15234 ;
vcif('c_FoodProc','JPN','USA') = 913.4293213 ;
vcif('c_FoodProc','JPN','EU_28') = 312.629364 ;
vcif('c_FoodProc','JPN','CHN') = 697.2129517 ;
vcif('c_FoodProc','JPN','JPN') = 7.000000096e-06 ;
vcif('c_FoodProc','JPN','IND') = 6.318625927 ;
vcif('c_FoodProc','JPN','SSA') = 93.91686249 ;
vcif('c_FoodProc','JPN','ROW') = 3413.888428 ;
vcif('c_FoodProc','IND','USA') = 3820.317139 ;
vcif('c_FoodProc','IND','EU_28') = 2776.680176 ;
vcif('c_FoodProc','IND','CHN') = 663.9684448 ;
vcif('c_FoodProc','IND','JPN') = 728.3532715 ;
vcif('c_FoodProc','IND','IND') = 7.000000096e-06 ;
vcif('c_FoodProc','IND','SSA') = 1134.853516 ;
vcif('c_FoodProc','IND','ROW') = 9958.327148 ;
vcif('c_FoodProc','SSA','USA') = 723.5046387 ;
vcif('c_FoodProc','SSA','EU_28') = 7195.169922 ;
vcif('c_FoodProc','SSA','CHN') = 1232.711304 ;
vcif('c_FoodProc','SSA','JPN') = 474.1105042 ;
vcif('c_FoodProc','SSA','IND') = 76.72075653 ;
vcif('c_FoodProc','SSA','SSA') = 8952.25 ;
vcif('c_FoodProc','SSA','ROW') = 3117.292236 ;
vcif('c_FoodProc','ROW','USA') = 69857.92188 ;
vcif('c_FoodProc','ROW','EU_28') = 61811.03516 ;
vcif('c_FoodProc','ROW','CHN') = 36129.58203 ;
vcif('c_FoodProc','ROW','JPN') = 27321.69727 ;
vcif('c_FoodProc','ROW','IND') = 14444.08984 ;
vcif('c_FoodProc','ROW','SSA') = 15923.77344 ;
vcif('c_FoodProc','ROW','ROW') = 191629.0156 ;
vcif('c_Energy','USA','USA') = 6.000000212e-06 ;
vcif('c_Energy','USA','EU_28') = 19536.76758 ;
vcif('c_Energy','USA','CHN') = 9697.918945 ;
vcif('c_Energy','USA','JPN') = 7227.493164 ;
vcif('c_Energy','USA','IND') = 3509.164551 ;
vcif('c_Energy','USA','SSA') = 2350.631104 ;
vcif('c_Energy','USA','ROW') = 113294.4609 ;
vcif('c_Energy','EU_28','USA') = 10228.87695 ;
vcif('c_Energy','EU_28','EU_28') = 96163.35156 ;
vcif('c_Energy','EU_28','CHN') = 2238.274902 ;
vcif('c_Energy','EU_28','JPN') = 131.8227692 ;
vcif('c_Energy','EU_28','IND') = 521.5407715 ;
vcif('c_Energy','EU_28','SSA') = 10704.74316 ;
vcif('c_Energy','EU_28','ROW') = 43134.14453 ;
vcif('c_Energy','CHN','USA') = 2754.911621 ;
vcif('c_Energy','CHN','EU_28') = 1383.091431 ;
vcif('c_Energy','CHN','CHN') = 6.000000212e-06 ;
vcif('c_Energy','CHN','JPN') = 1513.97583 ;
vcif('c_Energy','CHN','IND') = 1314.818481 ;
vcif('c_Energy','CHN','SSA') = 3881.236328 ;
vcif('c_Energy','CHN','ROW') = 27023.6582 ;
vcif('c_Energy','JPN','USA') = 841.6571655 ;
vcif('c_Energy','JPN','EU_28') = 257.5595703 ;
vcif('c_Energy','JPN','CHN') = 1184.084473 ;
vcif('c_Energy','JPN','JPN') = 6.000000212e-06 ;
vcif('c_Energy','JPN','IND') = 231.7566528 ;
vcif('c_Energy','JPN','SSA') = 15.0863018 ;
vcif('c_Energy','JPN','ROW') = 6933.508789 ;
vcif('c_Energy','IND','USA') = 4595.575684 ;
vcif('c_Energy','IND','EU_28') = 2374.604004 ;
vcif('c_Energy','IND','CHN') = 642.8779297 ;
vcif('c_Energy','IND','JPN') = 1582.191528 ;
vcif('c_Energy','IND','IND') = 6.000000212e-06 ;
vcif('c_Energy','IND','SSA') = 3130.306641 ;
vcif('c_Energy','IND','ROW') = 17176.10938 ;
vcif('c_Energy','SSA','USA') = 11630.84863 ;
vcif('c_Energy','SSA','EU_28') = 21796.08398 ;
vcif('c_Energy','SSA','CHN') = 30280.47266 ;
vcif('c_Energy','SSA','JPN') = 1177.484619 ;
vcif('c_Energy','SSA','IND') = 16460.84375 ;
vcif('c_Energy','SSA','SSA') = 11387.58496 ;
vcif('c_Energy','SSA','ROW') = 17027.14844 ;
vcif('c_Energy','ROW','USA') = 185510 ;
vcif('c_Energy','ROW','EU_28') = 353042.875 ;
vcif('c_Energy','ROW','CHN') = 188491.6719 ;
vcif('c_Energy','ROW','JPN') = 136337.8594 ;
vcif('c_Energy','ROW','IND') = 104964.8125 ;
vcif('c_Energy','ROW','SSA') = 18503.55664 ;
vcif('c_Energy','ROW','ROW') = 418311.0938 ;
vcif('c_Textiles','USA','USA') = 3.000000106e-06 ;
vcif('c_Textiles','USA','EU_28') = 3104.064209 ;
vcif('c_Textiles','USA','CHN') = 1717.289062 ;
vcif('c_Textiles','USA','JPN') = 499.5263977 ;
vcif('c_Textiles','USA','IND') = 202.9371643 ;
vcif('c_Textiles','USA','SSA') = 250.5522461 ;
vcif('c_Textiles','USA','ROW') = 20717.30664 ;
vcif('c_Textiles','EU_28','USA') = 11208.06055 ;
vcif('c_Textiles','EU_28','EU_28') = 183440.4062 ;
vcif('c_Textiles','EU_28','CHN') = 9651.767578 ;
vcif('c_Textiles','EU_28','JPN') = 4889.530762 ;
vcif('c_Textiles','EU_28','IND') = 753.4883423 ;
vcif('c_Textiles','EU_28','SSA') = 2043.424805 ;
vcif('c_Textiles','EU_28','ROW') = 46940.44531 ;
vcif('c_Textiles','CHN','USA') = 75835.49219 ;
vcif('c_Textiles','CHN','EU_28') = 73016.57812 ;
vcif('c_Textiles','CHN','CHN') = 3.000000106e-06 ;
vcif('c_Textiles','CHN','JPN') = 32363.47266 ;
vcif('c_Textiles','CHN','IND') = 6464.804688 ;
vcif('c_Textiles','CHN','SSA') = 20428.59766 ;
vcif('c_Textiles','CHN','ROW') = 176835.0938 ;
vcif('c_Textiles','JPN','USA') = 726.6304321 ;
vcif('c_Textiles','JPN','EU_28') = 848.1339111 ;
vcif('c_Textiles','JPN','CHN') = 3582.748291 ;
vcif('c_Textiles','JPN','JPN') = 3.000000106e-06 ;
vcif('c_Textiles','JPN','IND') = 121.7703705 ;
vcif('c_Textiles','JPN','SSA') = 175.8091125 ;
vcif('c_Textiles','JPN','ROW') = 3906.927246 ;
vcif('c_Textiles','IND','USA') = 10032.43945 ;
vcif('c_Textiles','IND','EU_28') = 13589.57812 ;
vcif('c_Textiles','IND','CHN') = 2360.280762 ;
vcif('c_Textiles','IND','JPN') = 612.0010986 ;
vcif('c_Textiles','IND','IND') = 3.000000106e-06 ;
vcif('c_Textiles','IND','SSA') = 2337.181396 ;
vcif('c_Textiles','IND','ROW') = 16677.12109 ;
vcif('c_Textiles','SSA','USA') = 1315.021118 ;
vcif('c_Textiles','SSA','EU_28') = 1485.18689 ;
vcif('c_Textiles','SSA','CHN') = 320.8284607 ;
vcif('c_Textiles','SSA','JPN') = 26.1140461 ;
vcif('c_Textiles','SSA','IND') = 62.30408478 ;
vcif('c_Textiles','SSA','SSA') = 2363.090088 ;
vcif('c_Textiles','SSA','ROW') = 472.4171143 ;
vcif('c_Textiles','ROW','USA') = 78062.55469 ;
vcif('c_Textiles','ROW','EU_28') = 97929.09375 ;
vcif('c_Textiles','ROW','CHN') = 25575.12305 ;
vcif('c_Textiles','ROW','JPN') = 15685.9082 ;
vcif('c_Textiles','ROW','IND') = 2443.56665 ;
vcif('c_Textiles','ROW','SSA') = 2928.615234 ;
vcif('c_Textiles','ROW','ROW') = 89132.89062 ;
vcif('c_Chem','USA','USA') = 3.000000106e-06 ;
vcif('c_Chem','USA','EU_28') = 63090.48438 ;
vcif('c_Chem','USA','CHN') = 20154.12891 ;
vcif('c_Chem','USA','JPN') = 14882.60547 ;
vcif('c_Chem','USA','IND') = 4384.878906 ;
vcif('c_Chem','USA','SSA') = 2114.39624 ;
vcif('c_Chem','USA','ROW') = 149166.125 ;
vcif('c_Chem','EU_28','USA') = 100272.1562 ;
vcif('c_Chem','EU_28','EU_28') = 602671.3125 ;
vcif('c_Chem','EU_28','CHN') = 48770.44531 ;
vcif('c_Chem','EU_28','JPN') = 23481.03906 ;
vcif('c_Chem','EU_28','IND') = 7525.338379 ;
vcif('c_Chem','EU_28','SSA') = 14269.19824 ;
vcif('c_Chem','EU_28','ROW') = 225236.625 ;
vcif('c_Chem','CHN','USA') = 35006.34766 ;
vcif('c_Chem','CHN','EU_28') = 31036.47852 ;
vcif('c_Chem','CHN','CHN') = 3.000000106e-06 ;
vcif('c_Chem','CHN','JPN') = 15322.9707 ;
vcif('c_Chem','CHN','IND') = 13841.67188 ;
vcif('c_Chem','CHN','SSA') = 10184.88672 ;
vcif('c_Chem','CHN','ROW') = 111545.9609 ;
vcif('c_Chem','JPN','USA') = 12597.42578 ;
vcif('c_Chem','JPN','EU_28') = 9674.351562 ;
vcif('c_Chem','JPN','CHN') = 32975.24609 ;
vcif('c_Chem','JPN','JPN') = 3.000000106e-06 ;
vcif('c_Chem','JPN','IND') = 2112.722168 ;
vcif('c_Chem','JPN','SSA') = 514.5753174 ;
vcif('c_Chem','JPN','ROW') = 47352.24609 ;
vcif('c_Chem','IND','USA') = 10794.43945 ;
vcif('c_Chem','IND','EU_28') = 8717.201172 ;
vcif('c_Chem','IND','CHN') = 3079.621582 ;
vcif('c_Chem','IND','JPN') = 961.9577026 ;
vcif('c_Chem','IND','IND') = 3.000000106e-06 ;
vcif('c_Chem','IND','SSA') = 5115.547852 ;
vcif('c_Chem','IND','ROW') = 21890.62891 ;
vcif('c_Chem','SSA','USA') = 1125.325073 ;
vcif('c_Chem','SSA','EU_28') = 2784.996094 ;
vcif('c_Chem','SSA','CHN') = 646.1533203 ;
vcif('c_Chem','SSA','JPN') = 90.41468048 ;
vcif('c_Chem','SSA','IND') = 819.862854 ;
vcif('c_Chem','SSA','SSA') = 7281.556152 ;
vcif('c_Chem','SSA','ROW') = 2327.688232 ;
vcif('c_Chem','ROW','USA') = 111499.1016 ;
vcif('c_Chem','ROW','EU_28') = 128613.5938 ;
vcif('c_Chem','ROW','CHN') = 137783.2031 ;
vcif('c_Chem','ROW','JPN') = 26847.73047 ;
vcif('c_Chem','ROW','IND') = 27080.93945 ;
vcif('c_Chem','ROW','SSA') = 12086.55273 ;
vcif('c_Chem','ROW','ROW') = 255539 ;
vcif('c_Manuf','USA','USA') = 1.299999985e-05 ;
vcif('c_Manuf','USA','EU_28') = 158118.125 ;
vcif('c_Manuf','USA','CHN') = 78784.1875 ;
vcif('c_Manuf','USA','JPN') = 40498.60156 ;
vcif('c_Manuf','USA','IND') = 17900.18164 ;
vcif('c_Manuf','USA','SSA') = 8420.379883 ;
vcif('c_Manuf','USA','ROW') = 601537.5 ;
vcif('c_Manuf','EU_28','USA') = 268906.9688 ;
vcif('c_Manuf','EU_28','EU_28') = 1930929.125 ;
vcif('c_Manuf','EU_28','CHN') = 200496.0312 ;
vcif('c_Manuf','EU_28','JPN') = 42397.36719 ;
vcif('c_Manuf','EU_28','IND') = 35684.64062 ;
vcif('c_Manuf','EU_28','SSA') = 43592.24219 ;
vcif('c_Manuf','EU_28','ROW') = 739287.3125 ;
vcif('c_Manuf','CHN','USA') = 363258.0625 ;
vcif('c_Manuf','CHN','EU_28') = 250631.3438 ;
vcif('c_Manuf','CHN','CHN') = 1.299999985e-05 ;
vcif('c_Manuf','CHN','JPN') = 126547.9922 ;
vcif('c_Manuf','CHN','IND') = 57500.17188 ;
vcif('c_Manuf','CHN','SSA') = 50948.45312 ;
vcif('c_Manuf','CHN','ROW') = 713920.5 ;
vcif('c_Manuf','JPN','USA') = 120329.7734 ;
vcif('c_Manuf','JPN','EU_28') = 65845.28125 ;
vcif('c_Manuf','JPN','CHN') = 164494.2656 ;
vcif('c_Manuf','JPN','JPN') = 1.299999985e-05 ;
vcif('c_Manuf','JPN','IND') = 7550.478027 ;
vcif('c_Manuf','JPN','SSA') = 6399.851562 ;
vcif('c_Manuf','JPN','ROW') = 254108.0156 ;
vcif('c_Manuf','IND','USA') = 21550.04297 ;
vcif('c_Manuf','IND','EU_28') = 22050.70508 ;
vcif('c_Manuf','IND','CHN') = 12644.25391 ;
vcif('c_Manuf','IND','JPN') = 1877.191895 ;
vcif('c_Manuf','IND','IND') = 1.299999985e-05 ;
vcif('c_Manuf','IND','SSA') = 7087.601562 ;
vcif('c_Manuf','IND','ROW') = 67343.24219 ;
vcif('c_Manuf','SSA','USA') = 8114.736816 ;
vcif('c_Manuf','SSA','EU_28') = 29450.13281 ;
vcif('c_Manuf','SSA','CHN') = 42051.86328 ;
vcif('c_Manuf','SSA','JPN') = 5219.314453 ;
vcif('c_Manuf','SSA','IND') = 12276.72754 ;
vcif('c_Manuf','SSA','SSA') = 22530.74805 ;
vcif('c_Manuf','SSA','ROW') = 43040.26562 ;
vcif('c_Manuf','ROW','USA') = 698777.625 ;
vcif('c_Manuf','ROW','EU_28') = 452190.7188 ;
vcif('c_Manuf','ROW','CHN') = 700951.625 ;
vcif('c_Manuf','ROW','JPN') = 135909 ;
vcif('c_Manuf','ROW','IND') = 92377.07812 ;
vcif('c_Manuf','ROW','SSA') = 32695.12695 ;
vcif('c_Manuf','ROW','ROW') = 1004028.5 ;
vcif('c_ForestFish','USA','USA') = 1.999999995e-06 ;
vcif('c_ForestFish','USA','EU_28') = 336.3588257 ;
vcif('c_ForestFish','USA','CHN') = 823.5733643 ;
vcif('c_ForestFish','USA','JPN') = 537.987915 ;
vcif('c_ForestFish','USA','IND') = 14.04646301 ;
vcif('c_ForestFish','USA','SSA') = 2.82115221 ;
vcif('c_ForestFish','USA','ROW') = 1060.832642 ;
vcif('c_ForestFish','EU_28','USA') = 473.8957825 ;
vcif('c_ForestFish','EU_28','EU_28') = 11709.3252 ;
vcif('c_ForestFish','EU_28','CHN') = 708.8208618 ;
vcif('c_ForestFish','EU_28','JPN') = 33.37355423 ;
vcif('c_ForestFish','EU_28','IND') = 38.46363831 ;
vcif('c_ForestFish','EU_28','SSA') = 22.26017189 ;
vcif('c_ForestFish','EU_28','ROW') = 1012.633301 ;
vcif('c_ForestFish','CHN','USA') = 34.18108749 ;
vcif('c_ForestFish','CHN','EU_28') = 85.4548645 ;
vcif('c_ForestFish','CHN','CHN') = 1.999999995e-06 ;
vcif('c_ForestFish','CHN','JPN') = 496.0134888 ;
vcif('c_ForestFish','CHN','IND') = 1.552072883 ;
vcif('c_ForestFish','CHN','SSA') = 1.658076167 ;
vcif('c_ForestFish','CHN','ROW') = 895.4845581 ;
vcif('c_ForestFish','JPN','USA') = 53.54919052 ;
vcif('c_ForestFish','JPN','EU_28') = 23.22491455 ;
vcif('c_ForestFish','JPN','CHN') = 144.8859558 ;
vcif('c_ForestFish','JPN','JPN') = 1.999999995e-06 ;
vcif('c_ForestFish','JPN','IND') = 1.80237639 ;
vcif('c_ForestFish','JPN','SSA') = 0.7413770556 ;
vcif('c_ForestFish','JPN','ROW') = 372.4014282 ;
vcif('c_ForestFish','IND','USA') = 45.00341034 ;
vcif('c_ForestFish','IND','EU_28') = 72.91044617 ;
vcif('c_ForestFish','IND','CHN') = 86.44630432 ;
vcif('c_ForestFish','IND','JPN') = 5.746335983 ;
vcif('c_ForestFish','IND','IND') = 1.999999995e-06 ;
vcif('c_ForestFish','IND','SSA') = 4.773351192 ;
vcif('c_ForestFish','IND','ROW') = 186.6848907 ;
vcif('c_ForestFish','SSA','USA') = 64.19580078 ;
vcif('c_ForestFish','SSA','EU_28') = 427.0680542 ;
vcif('c_ForestFish','SSA','CHN') = 2111.120117 ;
vcif('c_ForestFish','SSA','JPN') = 23.75351715 ;
vcif('c_ForestFish','SSA','IND') = 194.0139465 ;
vcif('c_ForestFish','SSA','SSA') = 178.102356 ;
vcif('c_ForestFish','SSA','ROW') = 605.8207397 ;
vcif('c_ForestFish','ROW','USA') = 2806.745361 ;
vcif('c_ForestFish','ROW','EU_28') = 6474.93457 ;
vcif('c_ForestFish','ROW','CHN') = 8445.070312 ;
vcif('c_ForestFish','ROW','JPN') = 1777.664673 ;
vcif('c_ForestFish','ROW','IND') = 1133.889893 ;
vcif('c_ForestFish','ROW','SSA') = 95.99095154 ;
vcif('c_ForestFish','ROW','ROW') = 5912.951172 ;
vcif('c_Svces','USA','USA') = 1.800000064e-05 ;
vcif('c_Svces','USA','EU_28') = 223204.1875 ;
vcif('c_Svces','USA','CHN') = 57874.48828 ;
vcif('c_Svces','USA','JPN') = 42788.98047 ;
vcif('c_Svces','USA','IND') = 19128.24609 ;
vcif('c_Svces','USA','SSA') = 18268.01758 ;
vcif('c_Svces','USA','ROW') = 342195.6562 ;
vcif('c_Svces','EU_28','USA') = 199294.9219 ;
vcif('c_Svces','EU_28','EU_28') = 1038910.938 ;
vcif('c_Svces','EU_28','CHN') = 52260.73047 ;
vcif('c_Svces','EU_28','JPN') = 32276.91992 ;
vcif('c_Svces','EU_28','IND') = 17354.9375 ;
vcif('c_Svces','EU_28','SSA') = 29422.0332 ;
vcif('c_Svces','EU_28','ROW') = 431834.0312 ;
vcif('c_Svces','CHN','USA') = 19002.94922 ;
vcif('c_Svces','CHN','EU_28') = 35278.92969 ;
vcif('c_Svces','CHN','CHN') = 1.800000064e-05 ;
vcif('c_Svces','CHN','JPN') = 7994.644531 ;
vcif('c_Svces','CHN','IND') = 2810.67749 ;
vcif('c_Svces','CHN','SSA') = 4210.168945 ;
vcif('c_Svces','CHN','ROW') = 105634.6719 ;
vcif('c_Svces','JPN','USA') = 20040.07617 ;
vcif('c_Svces','JPN','EU_28') = 18437.45898 ;
vcif('c_Svces','JPN','CHN') = 15669.74121 ;
vcif('c_Svces','JPN','JPN') = 1.800000064e-05 ;
vcif('c_Svces','JPN','IND') = 984.914978 ;
vcif('c_Svces','JPN','SSA') = 1856.799316 ;
vcif('c_Svces','JPN','ROW') = 50100.16797 ;
vcif('c_Svces','IND','USA') = 27956.55273 ;
vcif('c_Svces','IND','EU_28') = 28240.13477 ;
vcif('c_Svces','IND','CHN') = 2620.395752 ;
vcif('c_Svces','IND','JPN') = 2652.44751 ;
vcif('c_Svces','IND','IND') = 1.800000064e-05 ;
vcif('c_Svces','IND','SSA') = 4550.882812 ;
vcif('c_Svces','IND','ROW') = 43900.6875 ;
vcif('c_Svces','SSA','USA') = 9115.646484 ;
vcif('c_Svces','SSA','EU_28') = 23896.33984 ;
vcif('c_Svces','SSA','CHN') = 5176.616699 ;
vcif('c_Svces','SSA','JPN') = 1891.782471 ;
vcif('c_Svces','SSA','IND') = 2612.720215 ;
vcif('c_Svces','SSA','SSA') = 3836.831055 ;
vcif('c_Svces','SSA','ROW') = 19791.97656 ;
vcif('c_Svces','ROW','USA') = 271711.4062 ;
vcif('c_Svces','ROW','EU_28') = 361803.8438 ;
vcif('c_Svces','ROW','CHN') = 179073.5156 ;
vcif('c_Svces','ROW','JPN') = 57829.41406 ;
vcif('c_Svces','ROW','IND') = 33393.57812 ;
vcif('c_Svces','ROW','SSA') = 26256.54492 ;
vcif('c_Svces','ROW','ROW') = 442174.2812 ;

* vmsb data (490 cells)
vmsb('c_Rice','USA','USA') = 1.999999995e-06 ;
vmsb('c_Rice','USA','EU_28') = 58.82020569 ;
vmsb('c_Rice','USA','CHN') = 1.349482656 ;
vmsb('c_Rice','USA','JPN') = 722.5848999 ;
vmsb('c_Rice','USA','IND') = 3.142932653 ;
vmsb('c_Rice','USA','SSA') = 81.8949585 ;
vmsb('c_Rice','USA','ROW') = 1706.036621 ;
vmsb('c_Rice','EU_28','USA') = 29.14957237 ;
vmsb('c_Rice','EU_28','EU_28') = 1360.712036 ;
vmsb('c_Rice','EU_28','CHN') = 0.2884743512 ;
vmsb('c_Rice','EU_28','JPN') = 0.9003689289 ;
vmsb('c_Rice','EU_28','IND') = 1.698609352 ;
vmsb('c_Rice','EU_28','SSA') = 11.71206856 ;
vmsb('c_Rice','EU_28','ROW') = 297.5647888 ;
vmsb('c_Rice','CHN','USA') = 4.058561802 ;
vmsb('c_Rice','CHN','EU_28') = 2.302124977 ;
vmsb('c_Rice','CHN','CHN') = 1.999999995e-06 ;
vmsb('c_Rice','CHN','JPN') = 10.98286057 ;
vmsb('c_Rice','CHN','IND') = 1.083248258 ;
vmsb('c_Rice','CHN','SSA') = 409.4416504 ;
vmsb('c_Rice','CHN','ROW') = 365.8224792 ;
vmsb('c_Rice','JPN','USA') = 3.345901728 ;
vmsb('c_Rice','JPN','EU_28') = 3.92182827 ;
vmsb('c_Rice','JPN','CHN') = 2.414923191 ;
vmsb('c_Rice','JPN','JPN') = 1.999999995e-06 ;
vmsb('c_Rice','JPN','IND') = 0.05769827589 ;
vmsb('c_Rice','JPN','SSA') = 10.74849987 ;
vmsb('c_Rice','JPN','ROW') = 35.25725937 ;
vmsb('c_Rice','IND','USA') = 204.3927917 ;
vmsb('c_Rice','IND','EU_28') = 494.5645752 ;
vmsb('c_Rice','IND','CHN') = 0.5075881481 ;
vmsb('c_Rice','IND','JPN') = 2.044070244 ;
vmsb('c_Rice','IND','IND') = 1.999999995e-06 ;
vmsb('c_Rice','IND','SSA') = 2199.593994 ;
vmsb('c_Rice','IND','ROW') = 5891.867188 ;
vmsb('c_Rice','SSA','USA') = 1.469449997 ;
vmsb('c_Rice','SSA','EU_28') = 4.222956181 ;
vmsb('c_Rice','SSA','CHN') = 2.087954283 ;
vmsb('c_Rice','SSA','JPN') = 0.2296916842 ;
vmsb('c_Rice','SSA','IND') = 0.9179609418 ;
vmsb('c_Rice','SSA','SSA') = 175.0057373 ;
vmsb('c_Rice','SSA','ROW') = 3.753634691 ;
vmsb('c_Rice','ROW','USA') = 540.4920654 ;
vmsb('c_Rice','ROW','EU_28') = 943.7096558 ;
vmsb('c_Rice','ROW','CHN') = 2070.134766 ;
vmsb('c_Rice','ROW','JPN') = 483.4180603 ;
vmsb('c_Rice','ROW','IND') = 8.2989254 ;
vmsb('c_Rice','ROW','SSA') = 5082.773926 ;
vmsb('c_Rice','ROW','ROW') = 6314.235352 ;
vmsb('c_Crops','USA','USA') = 7.000000096e-06 ;
vmsb('c_Crops','USA','EU_28') = 6441.488281 ;
vmsb('c_Crops','USA','CHN') = 18126.13477 ;
vmsb('c_Crops','USA','JPN') = 7903.375977 ;
vmsb('c_Crops','USA','IND') = 1673.001953 ;
vmsb('c_Crops','USA','SSA') = 1245.551636 ;
vmsb('c_Crops','USA','ROW') = 42260.39453 ;
vmsb('c_Crops','EU_28','USA') = 1950.35791 ;
vmsb('c_Crops','EU_28','EU_28') = 80757.09375 ;
vmsb('c_Crops','EU_28','CHN') = 558.9281616 ;
vmsb('c_Crops','EU_28','JPN') = 294.0189209 ;
vmsb('c_Crops','EU_28','IND') = 456.5235291 ;
vmsb('c_Crops','EU_28','SSA') = 2293.30542 ;
vmsb('c_Crops','EU_28','ROW') = 17896.09961 ;
vmsb('c_Crops','CHN','USA') = 732.1986694 ;
vmsb('c_Crops','CHN','EU_28') = 1395.851074 ;
vmsb('c_Crops','CHN','CHN') = 7.000000096e-06 ;
vmsb('c_Crops','CHN','JPN') = 1518.314331 ;
vmsb('c_Crops','CHN','IND') = 388.2774963 ;
vmsb('c_Crops','CHN','SSA') = 202.787674 ;
vmsb('c_Crops','CHN','ROW') = 14542.78223 ;
vmsb('c_Crops','JPN','USA') = 54.05696487 ;
vmsb('c_Crops','JPN','EU_28') = 52.40800476 ;
vmsb('c_Crops','JPN','CHN') = 154.8975525 ;
vmsb('c_Crops','JPN','JPN') = 7.000000096e-06 ;
vmsb('c_Crops','JPN','IND') = 1.449666262 ;
vmsb('c_Crops','JPN','SSA') = 0.4071832895 ;
vmsb('c_Crops','JPN','ROW') = 386.6468506 ;
vmsb('c_Crops','IND','USA') = 878.4006348 ;
vmsb('c_Crops','IND','EU_28') = 1566.211792 ;
vmsb('c_Crops','IND','CHN') = 535.2227783 ;
vmsb('c_Crops','IND','JPN') = 173.0288544 ;
vmsb('c_Crops','IND','IND') = 7.000000096e-06 ;
vmsb('c_Crops','IND','SSA') = 174.6694641 ;
vmsb('c_Crops','IND','ROW') = 7564.964844 ;
vmsb('c_Crops','SSA','USA') = 2250.863525 ;
vmsb('c_Crops','SSA','EU_28') = 10436.24316 ;
vmsb('c_Crops','SSA','CHN') = 1707.042358 ;
vmsb('c_Crops','SSA','JPN') = 731.0632324 ;
vmsb('c_Crops','SSA','IND') = 2625.027832 ;
vmsb('c_Crops','SSA','SSA') = 2833.76709 ;
vmsb('c_Crops','SSA','ROW') = 12090.21191 ;
vmsb('c_Crops','ROW','USA') = 37619.11328 ;
vmsb('c_Crops','ROW','EU_28') = 39760.88281 ;
vmsb('c_Crops','ROW','CHN') = 45179.49609 ;
vmsb('c_Crops','ROW','JPN') = 9231.917969 ;
vmsb('c_Crops','ROW','IND') = 7161.362305 ;
vmsb('c_Crops','ROW','SSA') = 4520.763672 ;
vmsb('c_Crops','ROW','ROW') = 98460.28906 ;
vmsb('c_Livestock','USA','USA') = 3.99999999e-06 ;
vmsb('c_Livestock','USA','EU_28') = 517.5747681 ;
vmsb('c_Livestock','USA','CHN') = 1399.228638 ;
vmsb('c_Livestock','USA','JPN') = 187.8671722 ;
vmsb('c_Livestock','USA','IND') = 10.59145164 ;
vmsb('c_Livestock','USA','SSA') = 26.20913887 ;
vmsb('c_Livestock','USA','ROW') = 2707.378418 ;
vmsb('c_Livestock','EU_28','USA') = 637.1538696 ;
vmsb('c_Livestock','EU_28','EU_28') = 17415.18164 ;
vmsb('c_Livestock','EU_28','CHN') = 1429.594604 ;
vmsb('c_Livestock','EU_28','JPN') = 250.2157745 ;
vmsb('c_Livestock','EU_28','IND') = 29.20930672 ;
vmsb('c_Livestock','EU_28','SSA') = 167.6500702 ;
vmsb('c_Livestock','EU_28','ROW') = 6544.82959 ;
vmsb('c_Livestock','CHN','USA') = 226.4472809 ;
vmsb('c_Livestock','CHN','EU_28') = 426.2423706 ;
vmsb('c_Livestock','CHN','CHN') = 3.99999999e-06 ;
vmsb('c_Livestock','CHN','JPN') = 247.3534698 ;
vmsb('c_Livestock','CHN','IND') = 6.319579124 ;
vmsb('c_Livestock','CHN','SSA') = 33.59001923 ;
vmsb('c_Livestock','CHN','ROW') = 915.0906372 ;
vmsb('c_Livestock','JPN','USA') = 0.7954630256 ;
vmsb('c_Livestock','JPN','EU_28') = 4.52866745 ;
vmsb('c_Livestock','JPN','CHN') = 18.29362679 ;
vmsb('c_Livestock','JPN','JPN') = 3.99999999e-06 ;
vmsb('c_Livestock','JPN','IND') = 0.1290695667 ;
vmsb('c_Livestock','JPN','SSA') = 0.2170837969 ;
vmsb('c_Livestock','JPN','ROW') = 152.1505127 ;
vmsb('c_Livestock','IND','USA') = 90.42378998 ;
vmsb('c_Livestock','IND','EU_28') = 16.32385063 ;
vmsb('c_Livestock','IND','CHN') = 3.345079899 ;
vmsb('c_Livestock','IND','JPN') = 14.67167473 ;
vmsb('c_Livestock','IND','IND') = 3.99999999e-06 ;
vmsb('c_Livestock','IND','SSA') = 3.467049122 ;
vmsb('c_Livestock','IND','ROW') = 167.0259552 ;
vmsb('c_Livestock','SSA','USA') = 40.7976532 ;
vmsb('c_Livestock','SSA','EU_28') = 193.9924927 ;
vmsb('c_Livestock','SSA','CHN') = 405.5348206 ;
vmsb('c_Livestock','SSA','JPN') = 10.79146385 ;
vmsb('c_Livestock','SSA','IND') = 19.73477173 ;
vmsb('c_Livestock','SSA','SSA') = 691.8432617 ;
vmsb('c_Livestock','SSA','ROW') = 1330.858398 ;
vmsb('c_Livestock','ROW','USA') = 3223.358643 ;
vmsb('c_Livestock','ROW','EU_28') = 1801.776245 ;
vmsb('c_Livestock','ROW','CHN') = 5504.683594 ;
vmsb('c_Livestock','ROW','JPN') = 480.3292236 ;
vmsb('c_Livestock','ROW','IND') = 234.7552948 ;
vmsb('c_Livestock','ROW','SSA') = 91.29800415 ;
vmsb('c_Livestock','ROW','ROW') = 9333.068359 ;
vmsb('c_FoodProc','USA','USA') = 7.000000096e-06 ;
vmsb('c_FoodProc','USA','EU_28') = 6902.637695 ;
vmsb('c_FoodProc','USA','CHN') = 4821.256836 ;
vmsb('c_FoodProc','USA','JPN') = 10390.51855 ;
vmsb('c_FoodProc','USA','IND') = 1003.448669 ;
vmsb('c_FoodProc','USA','SSA') = 1283.802734 ;
vmsb('c_FoodProc','USA','ROW') = 63870.42969 ;
vmsb('c_FoodProc','EU_28','USA') = 25468.66016 ;
vmsb('c_FoodProc','EU_28','EU_28') = 299926.3438 ;
vmsb('c_FoodProc','EU_28','CHN') = 15558.80078 ;
vmsb('c_FoodProc','EU_28','JPN') = 11536.89844 ;
vmsb('c_FoodProc','EU_28','IND') = 1017.973816 ;
vmsb('c_FoodProc','EU_28','SSA') = 10306.5752 ;
vmsb('c_FoodProc','EU_28','ROW') = 84800.89062 ;
vmsb('c_FoodProc','CHN','USA') = 7658.34668 ;
vmsb('c_FoodProc','CHN','EU_28') = 7188.12793 ;
vmsb('c_FoodProc','CHN','CHN') = 7.000000096e-06 ;
vmsb('c_FoodProc','CHN','JPN') = 8512.37207 ;
vmsb('c_FoodProc','CHN','IND') = 316.9013977 ;
vmsb('c_FoodProc','CHN','SSA') = 2371.891357 ;
vmsb('c_FoodProc','CHN','ROW') = 30016.79492 ;
vmsb('c_FoodProc','JPN','USA') = 938.7324219 ;
vmsb('c_FoodProc','JPN','EU_28') = 339.0199585 ;
vmsb('c_FoodProc','JPN','CHN') = 816.9777832 ;
vmsb('c_FoodProc','JPN','JPN') = 7.000000096e-06 ;
vmsb('c_FoodProc','JPN','IND') = 8.838775635 ;
vmsb('c_FoodProc','JPN','SSA') = 102.2670364 ;
vmsb('c_FoodProc','JPN','ROW') = 3652.76123 ;
vmsb('c_FoodProc','IND','USA') = 3870.892578 ;
vmsb('c_FoodProc','IND','EU_28') = 2890.205322 ;
vmsb('c_FoodProc','IND','CHN') = 721.487793 ;
vmsb('c_FoodProc','IND','JPN') = 738.1878662 ;
vmsb('c_FoodProc','IND','IND') = 7.000000096e-06 ;
vmsb('c_FoodProc','IND','SSA') = 1335.550537 ;
vmsb('c_FoodProc','IND','ROW') = 11076.12305 ;
vmsb('c_FoodProc','SSA','USA') = 747.6011963 ;
vmsb('c_FoodProc','SSA','EU_28') = 7258.17334 ;
vmsb('c_FoodProc','SSA','CHN') = 1355.624268 ;
vmsb('c_FoodProc','SSA','JPN') = 494.4460449 ;
vmsb('c_FoodProc','SSA','IND') = 93.91573334 ;
vmsb('c_FoodProc','SSA','SSA') = 9328.091797 ;
vmsb('c_FoodProc','SSA','ROW') = 3687.662598 ;
vmsb('c_FoodProc','ROW','USA') = 71219.19531 ;
vmsb('c_FoodProc','ROW','EU_28') = 65675.35938 ;
vmsb('c_FoodProc','ROW','CHN') = 38952.05469 ;
vmsb('c_FoodProc','ROW','JPN') = 30200.5293 ;
vmsb('c_FoodProc','ROW','IND') = 17079.32617 ;
vmsb('c_FoodProc','ROW','SSA') = 18347.64844 ;
vmsb('c_FoodProc','ROW','ROW') = 206652.8125 ;
vmsb('c_Energy','USA','USA') = 6.000000212e-06 ;
vmsb('c_Energy','USA','EU_28') = 19696.41406 ;
vmsb('c_Energy','USA','CHN') = 10015.68555 ;
vmsb('c_Energy','USA','JPN') = 7233.90918 ;
vmsb('c_Energy','USA','IND') = 3611.272217 ;
vmsb('c_Energy','USA','SSA') = 2470.025635 ;
vmsb('c_Energy','USA','ROW') = 114597.6172 ;
vmsb('c_Energy','EU_28','USA') = 10408.63867 ;
vmsb('c_Energy','EU_28','EU_28') = 96163.35156 ;
vmsb('c_Energy','EU_28','CHN') = 2284.990723 ;
vmsb('c_Energy','EU_28','JPN') = 133.2752228 ;
vmsb('c_Energy','EU_28','IND') = 551.1957397 ;
vmsb('c_Energy','EU_28','SSA') = 11514.25879 ;
vmsb('c_Energy','EU_28','ROW') = 43576.61328 ;
vmsb('c_Energy','CHN','USA') = 2793.549805 ;
vmsb('c_Energy','CHN','EU_28') = 1400.750732 ;
vmsb('c_Energy','CHN','CHN') = 6.000000212e-06 ;
vmsb('c_Energy','CHN','JPN') = 1518.262695 ;
vmsb('c_Energy','CHN','IND') = 1374.967163 ;
vmsb('c_Energy','CHN','SSA') = 4194.141602 ;
vmsb('c_Energy','CHN','ROW') = 27408.16211 ;
vmsb('c_Energy','JPN','USA') = 856.6589966 ;
vmsb('c_Energy','JPN','EU_28') = 258.2913818 ;
vmsb('c_Energy','JPN','CHN') = 1259.680664 ;
vmsb('c_Energy','JPN','JPN') = 6.000000212e-06 ;
vmsb('c_Energy','JPN','IND') = 239.4447784 ;
vmsb('c_Energy','JPN','SSA') = 15.38948441 ;
vmsb('c_Energy','JPN','ROW') = 7024.764648 ;
vmsb('c_Energy','IND','USA') = 4664.246582 ;
vmsb('c_Energy','IND','EU_28') = 2416.965332 ;
vmsb('c_Energy','IND','CHN') = 686.3769531 ;
vmsb('c_Energy','IND','JPN') = 1588.476562 ;
vmsb('c_Energy','IND','IND') = 6.000000212e-06 ;
vmsb('c_Energy','IND','SSA') = 3292.510498 ;
vmsb('c_Energy','IND','ROW') = 17685.62305 ;
vmsb('c_Energy','SSA','USA') = 11631.18164 ;
vmsb('c_Energy','SSA','EU_28') = 21797.16992 ;
vmsb('c_Energy','SSA','CHN') = 30284.84766 ;
vmsb('c_Energy','SSA','JPN') = 1177.529907 ;
vmsb('c_Energy','SSA','IND') = 16574.11523 ;
vmsb('c_Energy','SSA','SSA') = 11469.31445 ;
vmsb('c_Energy','SSA','ROW') = 17339.17773 ;
vmsb('c_Energy','ROW','USA') = 185964.4844 ;
vmsb('c_Energy','ROW','EU_28') = 353871.2812 ;
vmsb('c_Energy','ROW','CHN') = 190057.9688 ;
vmsb('c_Energy','ROW','JPN') = 136446.1094 ;
vmsb('c_Energy','ROW','IND') = 105837.1016 ;
vmsb('c_Energy','ROW','SSA') = 19256.83984 ;
vmsb('c_Energy','ROW','ROW') = 423027.1562 ;
vmsb('c_Textiles','USA','USA') = 3.000000106e-06 ;
vmsb('c_Textiles','USA','EU_28') = 3308.365967 ;
vmsb('c_Textiles','USA','CHN') = 1874.556519 ;
vmsb('c_Textiles','USA','JPN') = 535.6190796 ;
vmsb('c_Textiles','USA','IND') = 223.3753967 ;
vmsb('c_Textiles','USA','SSA') = 300.0736084 ;
vmsb('c_Textiles','USA','ROW') = 21084.40039 ;
vmsb('c_Textiles','EU_28','USA') = 12082.6123 ;
vmsb('c_Textiles','EU_28','EU_28') = 183440.4062 ;
vmsb('c_Textiles','EU_28','CHN') = 10708.16406 ;
vmsb('c_Textiles','EU_28','JPN') = 5437.901367 ;
vmsb('c_Textiles','EU_28','IND') = 833.4849243 ;
vmsb('c_Textiles','EU_28','SSA') = 2421.034912 ;
vmsb('c_Textiles','EU_28','ROW') = 48257.82812 ;
vmsb('c_Textiles','CHN','USA') = 84281.95312 ;
vmsb('c_Textiles','CHN','EU_28') = 80461.59375 ;
vmsb('c_Textiles','CHN','CHN') = 3.000000106e-06 ;
vmsb('c_Textiles','CHN','JPN') = 34921.37891 ;
vmsb('c_Textiles','CHN','IND') = 7204.472656 ;
vmsb('c_Textiles','CHN','SSA') = 24851.63867 ;
vmsb('c_Textiles','CHN','ROW') = 190804.0469 ;
vmsb('c_Textiles','JPN','USA') = 767.8302002 ;
vmsb('c_Textiles','JPN','EU_28') = 900.7167358 ;
vmsb('c_Textiles','JPN','CHN') = 3907.300293 ;
vmsb('c_Textiles','JPN','JPN') = 3.000000106e-06 ;
vmsb('c_Textiles','JPN','IND') = 122.7007446 ;
vmsb('c_Textiles','JPN','SSA') = 190.4062653 ;
vmsb('c_Textiles','JPN','ROW') = 4067.690186 ;
vmsb('c_Textiles','IND','USA') = 10915.57715 ;
vmsb('c_Textiles','IND','EU_28') = 14517.65137 ;
vmsb('c_Textiles','IND','CHN') = 2498.402832 ;
vmsb('c_Textiles','IND','JPN') = 622.5472412 ;
vmsb('c_Textiles','IND','IND') = 3.000000106e-06 ;
vmsb('c_Textiles','IND','SSA') = 2874.364014 ;
vmsb('c_Textiles','IND','ROW') = 17821.9082 ;
vmsb('c_Textiles','SSA','USA') = 1318.015747 ;
vmsb('c_Textiles','SSA','EU_28') = 1487.043213 ;
vmsb('c_Textiles','SSA','CHN') = 333.5270691 ;
vmsb('c_Textiles','SSA','JPN') = 26.69602013 ;
vmsb('c_Textiles','SSA','IND') = 66.44233704 ;
vmsb('c_Textiles','SSA','SSA') = 2415.344971 ;
vmsb('c_Textiles','SSA','ROW') = 496.5903015 ;
vmsb('c_Textiles','ROW','USA') = 84467.92969 ;
vmsb('c_Textiles','ROW','EU_28') = 99702.39844 ;
vmsb('c_Textiles','ROW','CHN') = 26505.64062 ;
vmsb('c_Textiles','ROW','JPN') = 15893.41992 ;
vmsb('c_Textiles','ROW','IND') = 2573.818359 ;
vmsb('c_Textiles','ROW','SSA') = 3540.302979 ;
vmsb('c_Textiles','ROW','ROW') = 93695.57812 ;
vmsb('c_Chem','USA','USA') = 3.000000106e-06 ;
vmsb('c_Chem','USA','EU_28') = 64580.32812 ;
vmsb('c_Chem','USA','CHN') = 21506.51562 ;
vmsb('c_Chem','USA','JPN') = 15075.2793 ;
vmsb('c_Chem','USA','IND') = 4749.623535 ;
vmsb('c_Chem','USA','SSA') = 2237.837158 ;
vmsb('c_Chem','USA','ROW') = 151245.3906 ;
vmsb('c_Chem','EU_28','USA') = 101871.9609 ;
vmsb('c_Chem','EU_28','EU_28') = 602671.3125 ;
vmsb('c_Chem','EU_28','CHN') = 51668.49219 ;
vmsb('c_Chem','EU_28','JPN') = 23691.18359 ;
vmsb('c_Chem','EU_28','IND') = 8157.538086 ;
vmsb('c_Chem','EU_28','SSA') = 14835.33398 ;
vmsb('c_Chem','EU_28','ROW') = 230035.6562 ;
vmsb('c_Chem','CHN','USA') = 36183.05469 ;
vmsb('c_Chem','CHN','EU_28') = 32495.58789 ;
vmsb('c_Chem','CHN','CHN') = 3.000000106e-06 ;
vmsb('c_Chem','CHN','JPN') = 15344.49414 ;
vmsb('c_Chem','CHN','IND') = 14907.05664 ;
vmsb('c_Chem','CHN','SSA') = 11197.16992 ;
vmsb('c_Chem','CHN','ROW') = 115319.0156 ;
vmsb('c_Chem','JPN','USA') = 12895.57812 ;
vmsb('c_Chem','JPN','EU_28') = 10039.40234 ;
vmsb('c_Chem','JPN','CHN') = 35081.60547 ;
vmsb('c_Chem','JPN','JPN') = 3.000000106e-06 ;
vmsb('c_Chem','JPN','IND') = 2179.457764 ;
vmsb('c_Chem','JPN','SSA') = 559.0515747 ;
vmsb('c_Chem','JPN','ROW') = 48826.20312 ;
vmsb('c_Chem','IND','USA') = 10854.97266 ;
vmsb('c_Chem','IND','EU_28') = 8933.425781 ;
vmsb('c_Chem','IND','CHN') = 3232.743164 ;
vmsb('c_Chem','IND','JPN') = 962.4578857 ;
vmsb('c_Chem','IND','IND') = 3.000000106e-06 ;
vmsb('c_Chem','IND','SSA') = 5320.286621 ;
vmsb('c_Chem','IND','ROW') = 22692.89062 ;
vmsb('c_Chem','SSA','USA') = 1130.203857 ;
vmsb('c_Chem','SSA','EU_28') = 2785.345459 ;
vmsb('c_Chem','SSA','CHN') = 691.0810547 ;
vmsb('c_Chem','SSA','JPN') = 90.42695618 ;
vmsb('c_Chem','SSA','IND') = 845.4738159 ;
vmsb('c_Chem','SSA','SSA') = 7444.62207 ;
vmsb('c_Chem','SSA','ROW') = 2389.552979 ;
vmsb('c_Chem','ROW','USA') = 112277.625 ;
vmsb('c_Chem','ROW','EU_28') = 130072.9609 ;
vmsb('c_Chem','ROW','CHN') = 143248.5781 ;
vmsb('c_Chem','ROW','JPN') = 27052.57422 ;
vmsb('c_Chem','ROW','IND') = 28504.3418 ;
vmsb('c_Chem','ROW','SSA') = 12784.32812 ;
vmsb('c_Chem','ROW','ROW') = 260752.6562 ;
vmsb('c_Manuf','USA','USA') = 1.299999985e-05 ;
vmsb('c_Manuf','USA','EU_28') = 160861.1562 ;
vmsb('c_Manuf','USA','CHN') = 84301.32031 ;
vmsb('c_Manuf','USA','JPN') = 40587.38281 ;
vmsb('c_Manuf','USA','IND') = 19349.9043 ;
vmsb('c_Manuf','USA','SSA') = 9111.621094 ;
vmsb('c_Manuf','USA','ROW') = 609816.6875 ;
vmsb('c_Manuf','EU_28','USA') = 271934.375 ;
vmsb('c_Manuf','EU_28','EU_28') = 1930929.125 ;
vmsb('c_Manuf','EU_28','CHN') = 216974.2344 ;
vmsb('c_Manuf','EU_28','JPN') = 42531.67578 ;
vmsb('c_Manuf','EU_28','IND') = 38623.19531 ;
vmsb('c_Manuf','EU_28','SSA') = 46530.37109 ;
vmsb('c_Manuf','EU_28','ROW') = 759553.5625 ;
vmsb('c_Manuf','CHN','USA') = 367159.75 ;
vmsb('c_Manuf','CHN','EU_28') = 255178.2188 ;
vmsb('c_Manuf','CHN','CHN') = 1.299999985e-05 ;
vmsb('c_Manuf','CHN','JPN') = 126682.4375 ;
vmsb('c_Manuf','CHN','IND') = 60547.65234 ;
vmsb('c_Manuf','CHN','SSA') = 56947.82422 ;
vmsb('c_Manuf','CHN','ROW') = 737275 ;
vmsb('c_Manuf','JPN','USA') = 122030.5312 ;
vmsb('c_Manuf','JPN','EU_28') = 68421.63281 ;
vmsb('c_Manuf','JPN','CHN') = 174287.5312 ;
vmsb('c_Manuf','JPN','JPN') = 1.299999985e-05 ;
vmsb('c_Manuf','JPN','IND') = 7827.66748 ;
vmsb('c_Manuf','JPN','SSA') = 7257.074707 ;
vmsb('c_Manuf','JPN','ROW') = 264859.4688 ;
vmsb('c_Manuf','IND','USA') = 21642.6875 ;
vmsb('c_Manuf','IND','EU_28') = 22342.35742 ;
vmsb('c_Manuf','IND','CHN') = 12910.16211 ;
vmsb('c_Manuf','IND','JPN') = 1878.468872 ;
vmsb('c_Manuf','IND','IND') = 1.299999985e-05 ;
vmsb('c_Manuf','IND','SSA') = 7855.792969 ;
vmsb('c_Manuf','IND','ROW') = 70890.22656 ;
vmsb('c_Manuf','SSA','USA') = 8116.612793 ;
vmsb('c_Manuf','SSA','EU_28') = 29461.83398 ;
vmsb('c_Manuf','SSA','CHN') = 42625.625 ;
vmsb('c_Manuf','SSA','JPN') = 5222.461914 ;
vmsb('c_Manuf','SSA','IND') = 13130.0459 ;
vmsb('c_Manuf','SSA','SSA') = 23079.4668 ;
vmsb('c_Manuf','SSA','ROW') = 43740.32031 ;
vmsb('c_Manuf','ROW','USA') = 700815.125 ;
vmsb('c_Manuf','ROW','EU_28') = 454386.2188 ;
vmsb('c_Manuf','ROW','CHN') = 710398.5 ;
vmsb('c_Manuf','ROW','JPN') = 136113.3438 ;
vmsb('c_Manuf','ROW','IND') = 97627.39844 ;
vmsb('c_Manuf','ROW','SSA') = 35404.01172 ;
vmsb('c_Manuf','ROW','ROW') = 1025179 ;
vmsb('c_ForestFish','USA','USA') = 1.999999995e-06 ;
vmsb('c_ForestFish','USA','EU_28') = 351.308136 ;
vmsb('c_ForestFish','USA','CHN') = 839.461853 ;
vmsb('c_ForestFish','USA','JPN') = 539.3880005 ;
vmsb('c_ForestFish','USA','IND') = 16.06798172 ;
vmsb('c_ForestFish','USA','SSA') = 2.855175734 ;
vmsb('c_ForestFish','USA','ROW') = 1070.827759 ;
vmsb('c_ForestFish','EU_28','USA') = 474.6137085 ;
vmsb('c_ForestFish','EU_28','EU_28') = 11709.3252 ;
vmsb('c_ForestFish','EU_28','CHN') = 726.4550781 ;
vmsb('c_ForestFish','EU_28','JPN') = 34.08146286 ;
vmsb('c_ForestFish','EU_28','IND') = 42.15673828 ;
vmsb('c_ForestFish','EU_28','SSA') = 23.6146946 ;
vmsb('c_ForestFish','EU_28','ROW') = 1044.476929 ;
vmsb('c_ForestFish','CHN','USA') = 34.48443985 ;
vmsb('c_ForestFish','CHN','EU_28') = 85.83966827 ;
vmsb('c_ForestFish','CHN','CHN') = 1.999999995e-06 ;
vmsb('c_ForestFish','CHN','JPN') = 518.4633179 ;
vmsb('c_ForestFish','CHN','IND') = 1.821168423 ;
vmsb('c_ForestFish','CHN','SSA') = 1.741437912 ;
vmsb('c_ForestFish','CHN','ROW') = 943.4078979 ;
vmsb('c_ForestFish','JPN','USA') = 53.83761215 ;
vmsb('c_ForestFish','JPN','EU_28') = 23.64214134 ;
vmsb('c_ForestFish','JPN','CHN') = 147.4194946 ;
vmsb('c_ForestFish','JPN','JPN') = 1.999999995e-06 ;
vmsb('c_ForestFish','JPN','IND') = 1.940386772 ;
vmsb('c_ForestFish','JPN','SSA') = 0.8162996769 ;
vmsb('c_ForestFish','JPN','ROW') = 392.771759 ;
vmsb('c_ForestFish','IND','USA') = 45.02563477 ;
vmsb('c_ForestFish','IND','EU_28') = 73.72306824 ;
vmsb('c_ForestFish','IND','CHN') = 89.06599426 ;
vmsb('c_ForestFish','IND','JPN') = 5.751568794 ;
vmsb('c_ForestFish','IND','IND') = 1.999999995e-06 ;
vmsb('c_ForestFish','IND','SSA') = 5.02468586 ;
vmsb('c_ForestFish','IND','ROW') = 196.1800537 ;
vmsb('c_ForestFish','SSA','USA') = 64.28972626 ;
vmsb('c_ForestFish','SSA','EU_28') = 427.4476929 ;
vmsb('c_ForestFish','SSA','CHN') = 2113.478271 ;
vmsb('c_ForestFish','SSA','JPN') = 23.92433357 ;
vmsb('c_ForestFish','SSA','IND') = 200.2926636 ;
vmsb('c_ForestFish','SSA','SSA') = 183.2236481 ;
vmsb('c_ForestFish','SSA','ROW') = 613.3599243 ;
vmsb('c_ForestFish','ROW','USA') = 2807.926514 ;
vmsb('c_ForestFish','ROW','EU_28') = 6592.65625 ;
vmsb('c_ForestFish','ROW','CHN') = 8503.974609 ;
vmsb('c_ForestFish','ROW','JPN') = 1823.208618 ;
vmsb('c_ForestFish','ROW','IND') = 1171.552612 ;
vmsb('c_ForestFish','ROW','SSA') = 98.93125153 ;
vmsb('c_ForestFish','ROW','ROW') = 6048.91748 ;
vmsb('c_Svces','USA','USA') = 1.800000064e-05 ;
vmsb('c_Svces','USA','EU_28') = 223204.1875 ;
vmsb('c_Svces','USA','CHN') = 57874.48828 ;
vmsb('c_Svces','USA','JPN') = 42788.98047 ;
vmsb('c_Svces','USA','IND') = 19128.24609 ;
vmsb('c_Svces','USA','SSA') = 18268.01758 ;
vmsb('c_Svces','USA','ROW') = 342195.6562 ;
vmsb('c_Svces','EU_28','USA') = 199294.9219 ;
vmsb('c_Svces','EU_28','EU_28') = 1038910.938 ;
vmsb('c_Svces','EU_28','CHN') = 52260.73047 ;
vmsb('c_Svces','EU_28','JPN') = 32276.91992 ;
vmsb('c_Svces','EU_28','IND') = 17354.9375 ;
vmsb('c_Svces','EU_28','SSA') = 29422.0332 ;
vmsb('c_Svces','EU_28','ROW') = 431834.0312 ;
vmsb('c_Svces','CHN','USA') = 19002.94922 ;
vmsb('c_Svces','CHN','EU_28') = 35278.92969 ;
vmsb('c_Svces','CHN','CHN') = 1.800000064e-05 ;
vmsb('c_Svces','CHN','JPN') = 7994.644531 ;
vmsb('c_Svces','CHN','IND') = 2810.67749 ;
vmsb('c_Svces','CHN','SSA') = 4210.168945 ;
vmsb('c_Svces','CHN','ROW') = 105634.6719 ;
vmsb('c_Svces','JPN','USA') = 20040.07617 ;
vmsb('c_Svces','JPN','EU_28') = 18437.45898 ;
vmsb('c_Svces','JPN','CHN') = 15669.74121 ;
vmsb('c_Svces','JPN','JPN') = 1.800000064e-05 ;
vmsb('c_Svces','JPN','IND') = 984.914978 ;
vmsb('c_Svces','JPN','SSA') = 1856.799316 ;
vmsb('c_Svces','JPN','ROW') = 50100.16797 ;
vmsb('c_Svces','IND','USA') = 27956.55273 ;
vmsb('c_Svces','IND','EU_28') = 28240.13477 ;
vmsb('c_Svces','IND','CHN') = 2620.395752 ;
vmsb('c_Svces','IND','JPN') = 2652.44751 ;
vmsb('c_Svces','IND','IND') = 1.800000064e-05 ;
vmsb('c_Svces','IND','SSA') = 4550.882812 ;
vmsb('c_Svces','IND','ROW') = 43900.6875 ;
vmsb('c_Svces','SSA','USA') = 9115.646484 ;
vmsb('c_Svces','SSA','EU_28') = 23896.33984 ;
vmsb('c_Svces','SSA','CHN') = 5176.616699 ;
vmsb('c_Svces','SSA','JPN') = 1891.782471 ;
vmsb('c_Svces','SSA','IND') = 2612.720215 ;
vmsb('c_Svces','SSA','SSA') = 3836.831055 ;
vmsb('c_Svces','SSA','ROW') = 19791.97656 ;
vmsb('c_Svces','ROW','USA') = 271711.4062 ;
vmsb('c_Svces','ROW','EU_28') = 361803.8438 ;
vmsb('c_Svces','ROW','CHN') = 179073.5156 ;
vmsb('c_Svces','ROW','JPN') = 57829.41406 ;
vmsb('c_Svces','ROW','IND') = 33393.57812 ;
vmsb('c_Svces','ROW','SSA') = 26256.54492 ;
vmsb('c_Svces','ROW','ROW') = 442174.2812 ;

* vst data (7 cells)
vst('c_Svces','USA') = 31137.61328 ;
vst('c_Svces','EU_28') = 275190.5 ;
vst('c_Svces','CHN') = 24354.56445 ;
vst('c_Svces','JPN') = 33450.96094 ;
vst('c_Svces','IND') = 12699.37695 ;
vst('c_Svces','SSA') = 8560.361328 ;
vst('c_Svces','ROW') = 180795.8906 ;

* vtwr data (402 cells)
vtwr('c_Svces','c_Rice','USA','EU_28') = 4.090220928 ;
vtwr('c_Svces','c_Rice','USA','CHN') = 0.0721667856 ;
vtwr('c_Svces','c_Rice','USA','JPN') = 19.90464401 ;
vtwr('c_Svces','c_Rice','USA','IND') = 0.1397251189 ;
vtwr('c_Svces','c_Rice','USA','SSA') = 6.246553421 ;
vtwr('c_Svces','c_Rice','USA','ROW') = 90.43333435 ;
vtwr('c_Svces','c_Rice','EU_28','USA') = 2.304424763 ;
vtwr('c_Svces','c_Rice','EU_28','EU_28') = 58.00122833 ;
vtwr('c_Svces','c_Rice','EU_28','CHN') = 0.0001419999317 ;
vtwr('c_Svces','c_Rice','EU_28','JPN') = 0.01583400741 ;
vtwr('c_Svces','c_Rice','EU_28','IND') = 0.0801820308 ;
vtwr('c_Svces','c_Rice','EU_28','SSA') = 0.8575708866 ;
vtwr('c_Svces','c_Rice','EU_28','ROW') = 18.66053772 ;
vtwr('c_Svces','c_Rice','CHN','USA') = 0.2544041574 ;
vtwr('c_Svces','c_Rice','CHN','EU_28') = 0.0625930056 ;
vtwr('c_Svces','c_Rice','CHN','JPN') = 0.2916838229 ;
vtwr('c_Svces','c_Rice','CHN','IND') = 0.003766994225 ;
vtwr('c_Svces','c_Rice','CHN','SSA') = 30.60440063 ;
vtwr('c_Svces','c_Rice','CHN','ROW') = 25.71404076 ;
vtwr('c_Svces','c_Rice','JPN','USA') = 0.2628916502 ;
vtwr('c_Svces','c_Rice','JPN','EU_28') = 0.2648882568 ;
vtwr('c_Svces','c_Rice','JPN','CHN') = 0.1043460444 ;
vtwr('c_Svces','c_Rice','JPN','IND') = 0.0003760001273 ;
vtwr('c_Svces','c_Rice','JPN','SSA') = 0.823431015 ;
vtwr('c_Svces','c_Rice','JPN','ROW') = 2.09758544 ;
vtwr('c_Svces','c_Rice','IND','USA') = 17.19456863 ;
vtwr('c_Svces','c_Rice','IND','EU_28') = 38.73153305 ;
vtwr('c_Svces','c_Rice','IND','CHN') = 0.0009169995319 ;
vtwr('c_Svces','c_Rice','IND','JPN') = 0.05852596834 ;
vtwr('c_Svces','c_Rice','IND','SSA') = 172.2996216 ;
vtwr('c_Svces','c_Rice','IND','ROW') = 456.7183228 ;
vtwr('c_Svces','c_Rice','SSA','USA') = 0.03550044075 ;
vtwr('c_Svces','c_Rice','SSA','EU_28') = 0.02010197006 ;
vtwr('c_Svces','c_Rice','SSA','SSA') = 14.36340618 ;
vtwr('c_Svces','c_Rice','SSA','ROW') = 0.02670765668 ;
vtwr('c_Svces','c_Rice','ROW','USA') = 43.9080162 ;
vtwr('c_Svces','c_Rice','ROW','EU_28') = 72.45721436 ;
vtwr('c_Svces','c_Rice','ROW','CHN') = 172.1663055 ;
vtwr('c_Svces','c_Rice','ROW','JPN') = 13.64373016 ;
vtwr('c_Svces','c_Rice','ROW','IND') = 0.07258802652 ;
vtwr('c_Svces','c_Rice','ROW','SSA') = 371.2810974 ;
vtwr('c_Svces','c_Rice','ROW','ROW') = 471.5134277 ;
vtwr('c_Svces','c_Crops','USA','EU_28') = 309.4026184 ;
vtwr('c_Svces','c_Crops','USA','CHN') = 1244.348389 ;
vtwr('c_Svces','c_Crops','USA','JPN') = 636.0515747 ;
vtwr('c_Svces','c_Crops','USA','IND') = 58.67026138 ;
vtwr('c_Svces','c_Crops','USA','SSA') = 67.03475189 ;
vtwr('c_Svces','c_Crops','USA','ROW') = 2479.030029 ;
vtwr('c_Svces','c_Crops','EU_28','USA') = 94.48140717 ;
vtwr('c_Svces','c_Crops','EU_28','EU_28') = 6190.710449 ;
vtwr('c_Svces','c_Crops','EU_28','CHN') = 38.57877731 ;
vtwr('c_Svces','c_Crops','EU_28','JPN') = 15.5971489 ;
vtwr('c_Svces','c_Crops','EU_28','IND') = 22.53268433 ;
vtwr('c_Svces','c_Crops','EU_28','SSA') = 141.3300781 ;
vtwr('c_Svces','c_Crops','EU_28','ROW') = 1280.681274 ;
vtwr('c_Svces','c_Crops','CHN','USA') = 51.01221085 ;
vtwr('c_Svces','c_Crops','CHN','EU_28') = 84.71118927 ;
vtwr('c_Svces','c_Crops','CHN','JPN') = 149.2377625 ;
vtwr('c_Svces','c_Crops','CHN','IND') = 29.54511833 ;
vtwr('c_Svces','c_Crops','CHN','SSA') = 13.77734184 ;
vtwr('c_Svces','c_Crops','CHN','ROW') = 1632.933228 ;
vtwr('c_Svces','c_Crops','JPN','USA') = 6.302629471 ;
vtwr('c_Svces','c_Crops','JPN','EU_28') = 3.379983425 ;
vtwr('c_Svces','c_Crops','JPN','CHN') = 7.680016994 ;
vtwr('c_Svces','c_Crops','JPN','IND') = 0.03828800097 ;
vtwr('c_Svces','c_Crops','JPN','SSA') = 0.009493978694 ;
vtwr('c_Svces','c_Crops','JPN','ROW') = 40.10568237 ;
vtwr('c_Svces','c_Crops','IND','USA') = 48.96046829 ;
vtwr('c_Svces','c_Crops','IND','EU_28') = 83.47668457 ;
vtwr('c_Svces','c_Crops','IND','CHN') = 54.63238525 ;
vtwr('c_Svces','c_Crops','IND','JPN') = 6.717710495 ;
vtwr('c_Svces','c_Crops','IND','SSA') = 9.707403183 ;
vtwr('c_Svces','c_Crops','IND','ROW') = 432.9289246 ;
vtwr('c_Svces','c_Crops','SSA','USA') = 66.85109711 ;
vtwr('c_Svces','c_Crops','SSA','EU_28') = 836.7476807 ;
vtwr('c_Svces','c_Crops','SSA','CHN') = 116.9958344 ;
vtwr('c_Svces','c_Crops','SSA','JPN') = 42.538517 ;
vtwr('c_Svces','c_Crops','SSA','IND') = 80.52589417 ;
vtwr('c_Svces','c_Crops','SSA','SSA') = 225.0883636 ;
vtwr('c_Svces','c_Crops','SSA','ROW') = 657.0478516 ;
vtwr('c_Svces','c_Crops','ROW','USA') = 2597.040039 ;
vtwr('c_Svces','c_Crops','ROW','EU_28') = 3382.054199 ;
vtwr('c_Svces','c_Crops','ROW','CHN') = 3663.766846 ;
vtwr('c_Svces','c_Crops','ROW','JPN') = 783.4464722 ;
vtwr('c_Svces','c_Crops','ROW','IND') = 325.9208069 ;
vtwr('c_Svces','c_Crops','ROW','SSA') = 256.3928528 ;
vtwr('c_Svces','c_Crops','ROW','ROW') = 7825.811523 ;
vtwr('c_Svces','c_Livestock','USA','EU_28') = 26.55854607 ;
vtwr('c_Svces','c_Livestock','USA','CHN') = 64.11737061 ;
vtwr('c_Svces','c_Livestock','USA','JPN') = 7.001403809 ;
vtwr('c_Svces','c_Livestock','USA','IND') = 0.4068657756 ;
vtwr('c_Svces','c_Livestock','USA','SSA') = 7.194734573 ;
vtwr('c_Svces','c_Livestock','USA','ROW') = 159.7551117 ;
vtwr('c_Svces','c_Livestock','EU_28','USA') = 26.6160183 ;
vtwr('c_Svces','c_Livestock','EU_28','EU_28') = 2767.376709 ;
vtwr('c_Svces','c_Livestock','EU_28','CHN') = 47.42992401 ;
vtwr('c_Svces','c_Livestock','EU_28','JPN') = 9.476604462 ;
vtwr('c_Svces','c_Livestock','EU_28','IND') = 2.843416214 ;
vtwr('c_Svces','c_Livestock','EU_28','SSA') = 30.90870285 ;
vtwr('c_Svces','c_Livestock','EU_28','ROW') = 1618.863037 ;
vtwr('c_Svces','c_Livestock','CHN','USA') = 10.85265827 ;
vtwr('c_Svces','c_Livestock','CHN','EU_28') = 13.91102695 ;
vtwr('c_Svces','c_Livestock','CHN','JPN') = 9.183815956 ;
vtwr('c_Svces','c_Livestock','CHN','IND') = 0.1417531967 ;
vtwr('c_Svces','c_Livestock','CHN','SSA') = 1.03445375 ;
vtwr('c_Svces','c_Livestock','CHN','ROW') = 112.6097794 ;
vtwr('c_Svces','c_Livestock','JPN','USA') = 0.03162408993 ;
vtwr('c_Svces','c_Livestock','JPN','EU_28') = 0.2384856641 ;
vtwr('c_Svces','c_Livestock','JPN','CHN') = 1.099365354 ;
vtwr('c_Svces','c_Livestock','JPN','IND') = 0.004807002842 ;
vtwr('c_Svces','c_Livestock','JPN','SSA') = 0.005106118042 ;
vtwr('c_Svces','c_Livestock','JPN','ROW') = 8.240243912 ;
vtwr('c_Svces','c_Livestock','IND','USA') = 3.704815865 ;
vtwr('c_Svces','c_Livestock','IND','EU_28') = 0.8079074025 ;
vtwr('c_Svces','c_Livestock','IND','CHN') = 0.03354059905 ;
vtwr('c_Svces','c_Livestock','IND','JPN') = 1.237440944 ;
vtwr('c_Svces','c_Livestock','IND','SSA') = 0.1455811113 ;
vtwr('c_Svces','c_Livestock','IND','ROW') = 7.736511707 ;
vtwr('c_Svces','c_Livestock','SSA','USA') = 1.716252089 ;
vtwr('c_Svces','c_Livestock','SSA','EU_28') = 6.914689541 ;
vtwr('c_Svces','c_Livestock','SSA','CHN') = 10.79684448 ;
vtwr('c_Svces','c_Livestock','SSA','JPN') = 0.3238119185 ;
vtwr('c_Svces','c_Livestock','SSA','IND') = 0.4046350121 ;
vtwr('c_Svces','c_Livestock','SSA','SSA') = 271.5197144 ;
vtwr('c_Svces','c_Livestock','SSA','ROW') = 132.3019409 ;
vtwr('c_Svces','c_Livestock','ROW','USA') = 47.50193787 ;
vtwr('c_Svces','c_Livestock','ROW','EU_28') = 62.14130783 ;
vtwr('c_Svces','c_Livestock','ROW','CHN') = 352.3748474 ;
vtwr('c_Svces','c_Livestock','ROW','JPN') = 58.70546341 ;
vtwr('c_Svces','c_Livestock','ROW','IND') = 8.543392181 ;
vtwr('c_Svces','c_Livestock','ROW','SSA') = 8.238998413 ;
vtwr('c_Svces','c_Livestock','ROW','ROW') = 2538.81958 ;
vtwr('c_Svces','c_FoodProc','USA','EU_28') = 324.969635 ;
vtwr('c_Svces','c_FoodProc','USA','CHN') = 204.5898132 ;
vtwr('c_Svces','c_FoodProc','USA','JPN') = 416.8370972 ;
vtwr('c_Svces','c_FoodProc','USA','IND') = 18.99201775 ;
vtwr('c_Svces','c_FoodProc','USA','SSA') = 50.68429947 ;
vtwr('c_Svces','c_FoodProc','USA','ROW') = 2213.575195 ;
vtwr('c_Svces','c_FoodProc','EU_28','USA') = 1379.010864 ;
vtwr('c_Svces','c_FoodProc','EU_28','EU_28') = 10139.79688 ;
vtwr('c_Svces','c_FoodProc','EU_28','CHN') = 670.4636841 ;
vtwr('c_Svces','c_FoodProc','EU_28','JPN') = 457.5644531 ;
vtwr('c_Svces','c_FoodProc','EU_28','IND') = 28.07312202 ;
vtwr('c_Svces','c_FoodProc','EU_28','SSA') = 509.9098206 ;
vtwr('c_Svces','c_FoodProc','EU_28','ROW') = 3656.149658 ;
vtwr('c_Svces','c_FoodProc','CHN','USA') = 432.7387695 ;
vtwr('c_Svces','c_FoodProc','CHN','EU_28') = 368.5120544 ;
vtwr('c_Svces','c_FoodProc','CHN','JPN') = 434.2437744 ;
vtwr('c_Svces','c_FoodProc','CHN','IND') = 12.5582428 ;
vtwr('c_Svces','c_FoodProc','CHN','SSA') = 112.2228394 ;
vtwr('c_Svces','c_FoodProc','CHN','ROW') = 1567.585571 ;
vtwr('c_Svces','c_FoodProc','JPN','USA') = 54.8482132 ;
vtwr('c_Svces','c_FoodProc','JPN','EU_28') = 16.66296005 ;
vtwr('c_Svces','c_FoodProc','JPN','CHN') = 33.05475616 ;
vtwr('c_Svces','c_FoodProc','JPN','IND') = 0.2965859175 ;
vtwr('c_Svces','c_FoodProc','JPN','SSA') = 4.02971077 ;
vtwr('c_Svces','c_FoodProc','JPN','ROW') = 188.315918 ;
vtwr('c_Svces','c_FoodProc','IND','USA') = 139.1222687 ;
vtwr('c_Svces','c_FoodProc','IND','EU_28') = 127.6380844 ;
vtwr('c_Svces','c_FoodProc','IND','CHN') = 35.00914764 ;
vtwr('c_Svces','c_FoodProc','IND','JPN') = 32.74049759 ;
vtwr('c_Svces','c_FoodProc','IND','SSA') = 59.0953064 ;
vtwr('c_Svces','c_FoodProc','IND','ROW') = 449.5955811 ;
vtwr('c_Svces','c_FoodProc','SSA','USA') = 34.00876999 ;
vtwr('c_Svces','c_FoodProc','SSA','EU_28') = 303.9998169 ;
vtwr('c_Svces','c_FoodProc','SSA','CHN') = 44.34393692 ;
vtwr('c_Svces','c_FoodProc','SSA','JPN') = 20.73714256 ;
vtwr('c_Svces','c_FoodProc','SSA','IND') = 2.721326828 ;
vtwr('c_Svces','c_FoodProc','SSA','SSA') = 499.657135 ;
vtwr('c_Svces','c_FoodProc','SSA','ROW') = 141.4272308 ;
vtwr('c_Svces','c_FoodProc','ROW','USA') = 2484.229492 ;
vtwr('c_Svces','c_FoodProc','ROW','EU_28') = 3067.386475 ;
vtwr('c_Svces','c_FoodProc','ROW','CHN') = 1851.206421 ;
vtwr('c_Svces','c_FoodProc','ROW','JPN') = 1201.955566 ;
vtwr('c_Svces','c_FoodProc','ROW','IND') = 760.15448 ;
vtwr('c_Svces','c_FoodProc','ROW','SSA') = 872.8492432 ;
vtwr('c_Svces','c_FoodProc','ROW','ROW') = 10125.2627 ;
vtwr('c_Svces','c_Energy','USA','EU_28') = 1088.909058 ;
vtwr('c_Svces','c_Energy','USA','CHN') = 418.859436 ;
vtwr('c_Svces','c_Energy','USA','JPN') = 428.0500183 ;
vtwr('c_Svces','c_Energy','USA','IND') = 259.097229 ;
vtwr('c_Svces','c_Energy','USA','SSA') = 91.92569733 ;
vtwr('c_Svces','c_Energy','USA','ROW') = 4860.174316 ;
vtwr('c_Svces','c_Energy','EU_28','USA') = 398.0622864 ;
vtwr('c_Svces','c_Energy','EU_28','EU_28') = 2546.504639 ;
vtwr('c_Svces','c_Energy','EU_28','CHN') = 62.67894745 ;
vtwr('c_Svces','c_Energy','EU_28','JPN') = 4.381140232 ;
vtwr('c_Svces','c_Energy','EU_28','IND') = 29.96928978 ;
vtwr('c_Svces','c_Energy','EU_28','SSA') = 418.3511658 ;
vtwr('c_Svces','c_Energy','EU_28','ROW') = 1408.838379 ;
vtwr('c_Svces','c_Energy','CHN','USA') = 111.7216949 ;
vtwr('c_Svces','c_Energy','CHN','EU_28') = 58.42987823 ;
vtwr('c_Svces','c_Energy','CHN','JPN') = 98.30980682 ;
vtwr('c_Svces','c_Energy','CHN','IND') = 103.1720276 ;
vtwr('c_Svces','c_Energy','CHN','SSA') = 172.9576111 ;
vtwr('c_Svces','c_Energy','CHN','ROW') = 1121.552979 ;
vtwr('c_Svces','c_Energy','JPN','USA') = 33.65209198 ;
vtwr('c_Svces','c_Energy','JPN','EU_28') = 19.57383728 ;
vtwr('c_Svces','c_Energy','JPN','CHN') = 47.29463959 ;
vtwr('c_Svces','c_Energy','JPN','IND') = 18.60061073 ;
vtwr('c_Svces','c_Energy','JPN','SSA') = 0.2167983502 ;
vtwr('c_Svces','c_Energy','JPN','ROW') = 275.1222534 ;
vtwr('c_Svces','c_Energy','IND','USA') = 179.915863 ;
vtwr('c_Svces','c_Energy','IND','EU_28') = 91.14524841 ;
vtwr('c_Svces','c_Energy','IND','CHN') = 25.54937363 ;
vtwr('c_Svces','c_Energy','IND','JPN') = 62.10665894 ;
vtwr('c_Svces','c_Energy','IND','SSA') = 123.4527435 ;
vtwr('c_Svces','c_Energy','IND','ROW') = 653.6221313 ;
vtwr('c_Svces','c_Energy','SSA','USA') = 278.6029968 ;
vtwr('c_Svces','c_Energy','SSA','EU_28') = 841.6030884 ;
vtwr('c_Svces','c_Energy','SSA','CHN') = 703.7589111 ;
vtwr('c_Svces','c_Energy','SSA','JPN') = 96.00667572 ;
vtwr('c_Svces','c_Energy','SSA','IND') = 734.7073975 ;
vtwr('c_Svces','c_Energy','SSA','SSA') = 336.1331177 ;
vtwr('c_Svces','c_Energy','SSA','ROW') = 914.5533447 ;
vtwr('c_Svces','c_Energy','ROW','USA') = 5684.778809 ;
vtwr('c_Svces','c_Energy','ROW','EU_28') = 13685.99512 ;
vtwr('c_Svces','c_Energy','ROW','CHN') = 8282.486328 ;
vtwr('c_Svces','c_Energy','ROW','JPN') = 8160 ;
vtwr('c_Svces','c_Energy','ROW','IND') = 4455.317871 ;
vtwr('c_Svces','c_Energy','ROW','SSA') = 673.7069702 ;
vtwr('c_Svces','c_Energy','ROW','ROW') = 19163.59375 ;
vtwr('c_Svces','c_Textiles','USA','EU_28') = 139.4281616 ;
vtwr('c_Svces','c_Textiles','USA','CHN') = 74.67747498 ;
vtwr('c_Svces','c_Textiles','USA','JPN') = 23.54336929 ;
vtwr('c_Svces','c_Textiles','USA','IND') = 10.90122414 ;
vtwr('c_Svces','c_Textiles','USA','SSA') = 11.34365845 ;
vtwr('c_Svces','c_Textiles','USA','ROW') = 537.862915 ;
vtwr('c_Svces','c_Textiles','EU_28','USA') = 453.765564 ;
vtwr('c_Svces','c_Textiles','EU_28','EU_28') = 4269.784668 ;
vtwr('c_Svces','c_Textiles','EU_28','CHN') = 398.7732849 ;
vtwr('c_Svces','c_Textiles','EU_28','JPN') = 194.6898651 ;
vtwr('c_Svces','c_Textiles','EU_28','IND') = 36.01577759 ;
vtwr('c_Svces','c_Textiles','EU_28','SSA') = 88.28320312 ;
vtwr('c_Svces','c_Textiles','EU_28','ROW') = 1761.697754 ;
vtwr('c_Svces','c_Textiles','CHN','USA') = 3096.604736 ;
vtwr('c_Svces','c_Textiles','CHN','EU_28') = 2926.6604 ;
vtwr('c_Svces','c_Textiles','CHN','JPN') = 1290.518677 ;
vtwr('c_Svces','c_Textiles','CHN','IND') = 304.5348511 ;
vtwr('c_Svces','c_Textiles','CHN','SSA') = 874.668396 ;
vtwr('c_Svces','c_Textiles','CHN','ROW') = 7786.410645 ;
vtwr('c_Svces','c_Textiles','JPN','USA') = 38.34497833 ;
vtwr('c_Svces','c_Textiles','JPN','EU_28') = 45.59008026 ;
vtwr('c_Svces','c_Textiles','JPN','CHN') = 192.1771088 ;
vtwr('c_Svces','c_Textiles','JPN','IND') = 5.468349934 ;
vtwr('c_Svces','c_Textiles','JPN','SSA') = 9.017764091 ;
vtwr('c_Svces','c_Textiles','JPN','ROW') = 200.47966 ;
vtwr('c_Svces','c_Textiles','IND','USA') = 418.6965942 ;
vtwr('c_Svces','c_Textiles','IND','EU_28') = 534.2144775 ;
vtwr('c_Svces','c_Textiles','IND','CHN') = 91.28924561 ;
vtwr('c_Svces','c_Textiles','IND','JPN') = 24.0992012 ;
vtwr('c_Svces','c_Textiles','IND','SSA') = 100.7053223 ;
vtwr('c_Svces','c_Textiles','IND','ROW') = 706.5093994 ;
vtwr('c_Svces','c_Textiles','SSA','USA') = 42.41001511 ;
vtwr('c_Svces','c_Textiles','SSA','EU_28') = 48.95233917 ;
vtwr('c_Svces','c_Textiles','SSA','CHN') = 9.125442505 ;
vtwr('c_Svces','c_Textiles','SSA','JPN') = 0.7061049342 ;
vtwr('c_Svces','c_Textiles','SSA','IND') = 1.468785763 ;
vtwr('c_Svces','c_Textiles','SSA','SSA') = 101.9375229 ;
vtwr('c_Svces','c_Textiles','SSA','ROW') = 15.08427143 ;
vtwr('c_Svces','c_Textiles','ROW','USA') = 2543.243408 ;
vtwr('c_Svces','c_Textiles','ROW','EU_28') = 3553.33252 ;
vtwr('c_Svces','c_Textiles','ROW','CHN') = 1042.920898 ;
vtwr('c_Svces','c_Textiles','ROW','JPN') = 613.2518311 ;
vtwr('c_Svces','c_Textiles','ROW','IND') = 121.1716919 ;
vtwr('c_Svces','c_Textiles','ROW','SSA') = 137.7368011 ;
vtwr('c_Svces','c_Textiles','ROW','ROW') = 3905.606445 ;
vtwr('c_Svces','c_Chem','USA','EU_28') = 1581.39563 ;
vtwr('c_Svces','c_Chem','USA','CHN') = 824.8319702 ;
vtwr('c_Svces','c_Chem','USA','JPN') = 408.446106 ;
vtwr('c_Svces','c_Chem','USA','IND') = 191.4198456 ;
vtwr('c_Svces','c_Chem','USA','SSA') = 95.75714874 ;
vtwr('c_Svces','c_Chem','USA','ROW') = 3844.711914 ;
vtwr('c_Svces','c_Chem','EU_28','USA') = 1765.527344 ;
vtwr('c_Svces','c_Chem','EU_28','EU_28') = 12848.81348 ;
vtwr('c_Svces','c_Chem','EU_28','CHN') = 1429.459229 ;
vtwr('c_Svces','c_Chem','EU_28','JPN') = 401.0444031 ;
vtwr('c_Svces','c_Chem','EU_28','IND') = 309.657074 ;
vtwr('c_Svces','c_Chem','EU_28','SSA') = 453.1101074 ;
vtwr('c_Svces','c_Chem','EU_28','ROW') = 6260.814941 ;
vtwr('c_Svces','c_Chem','CHN','USA') = 1834.514893 ;
vtwr('c_Svces','c_Chem','CHN','EU_28') = 1405.859131 ;
vtwr('c_Svces','c_Chem','CHN','JPN') = 730.718811 ;
vtwr('c_Svces','c_Chem','CHN','IND') = 505.9957275 ;
vtwr('c_Svces','c_Chem','CHN','SSA') = 549.9828491 ;
vtwr('c_Svces','c_Chem','CHN','ROW') = 5397.624023 ;
vtwr('c_Svces','c_Chem','JPN','USA') = 456.6981201 ;
vtwr('c_Svces','c_Chem','JPN','EU_28') = 338.6200256 ;
vtwr('c_Svces','c_Chem','JPN','CHN') = 1615.079834 ;
vtwr('c_Svces','c_Chem','JPN','IND') = 109.4384079 ;
vtwr('c_Svces','c_Chem','JPN','SSA') = 24.04384804 ;
vtwr('c_Svces','c_Chem','JPN','ROW') = 2167.885986 ;
vtwr('c_Svces','c_Chem','IND','USA') = 206.5233459 ;
vtwr('c_Svces','c_Chem','IND','EU_28') = 264.918457 ;
vtwr('c_Svces','c_Chem','IND','CHN') = 131.7388611 ;
vtwr('c_Svces','c_Chem','IND','JPN') = 24.58366966 ;
vtwr('c_Svces','c_Chem','IND','SSA') = 128.1336517 ;
vtwr('c_Svces','c_Chem','IND','ROW') = 761.1030273 ;
vtwr('c_Svces','c_Chem','SSA','USA') = 64.72596741 ;
vtwr('c_Svces','c_Chem','SSA','EU_28') = 154.4781494 ;
vtwr('c_Svces','c_Chem','SSA','CHN') = 28.55791473 ;
vtwr('c_Svces','c_Chem','SSA','JPN') = 3.930413723 ;
vtwr('c_Svces','c_Chem','SSA','IND') = 63.47864532 ;
vtwr('c_Svces','c_Chem','SSA','SSA') = 380.5132751 ;
vtwr('c_Svces','c_Chem','SSA','ROW') = 138.292923 ;
vtwr('c_Svces','c_Chem','ROW','USA') = 2928.98999 ;
vtwr('c_Svces','c_Chem','ROW','EU_28') = 3886.536865 ;
vtwr('c_Svces','c_Chem','ROW','CHN') = 7611.83252 ;
vtwr('c_Svces','c_Chem','ROW','JPN') = 1182.822266 ;
vtwr('c_Svces','c_Chem','ROW','IND') = 1510.074829 ;
vtwr('c_Svces','c_Chem','ROW','SSA') = 628.2702637 ;
vtwr('c_Svces','c_Chem','ROW','ROW') = 12303.1543 ;
vtwr('c_Svces','c_Manuf','USA','EU_28') = 3906.334717 ;
vtwr('c_Svces','c_Manuf','USA','CHN') = 2296.384277 ;
vtwr('c_Svces','c_Manuf','USA','JPN') = 988.3754883 ;
vtwr('c_Svces','c_Manuf','USA','IND') = 452.5919495 ;
vtwr('c_Svces','c_Manuf','USA','SSA') = 238.3239899 ;
vtwr('c_Svces','c_Manuf','USA','ROW') = 9695.478516 ;
vtwr('c_Svces','c_Manuf','EU_28','USA') = 7285.537109 ;
vtwr('c_Svces','c_Manuf','EU_28','EU_28') = 41372.75781 ;
vtwr('c_Svces','c_Manuf','EU_28','CHN') = 5784.931641 ;
vtwr('c_Svces','c_Manuf','EU_28','JPN') = 1115.569702 ;
vtwr('c_Svces','c_Manuf','EU_28','IND') = 1041.12207 ;
vtwr('c_Svces','c_Manuf','EU_28','SSA') = 1586.708618 ;
vtwr('c_Svces','c_Manuf','EU_28','ROW') = 21234.78125 ;
vtwr('c_Svces','c_Manuf','CHN','USA') = 12964.67676 ;
vtwr('c_Svces','c_Manuf','CHN','EU_28') = 9194.335938 ;
vtwr('c_Svces','c_Manuf','CHN','JPN') = 4326.134766 ;
vtwr('c_Svces','c_Manuf','CHN','IND') = 1770.191528 ;
vtwr('c_Svces','c_Manuf','CHN','SSA') = 2308.483643 ;
vtwr('c_Svces','c_Manuf','CHN','ROW') = 25701.52148 ;
vtwr('c_Svces','c_Manuf','JPN','USA') = 3018.51709 ;
vtwr('c_Svces','c_Manuf','JPN','EU_28') = 1652.171997 ;
vtwr('c_Svces','c_Manuf','JPN','CHN') = 4449.266113 ;
vtwr('c_Svces','c_Manuf','JPN','IND') = 266.0362854 ;
vtwr('c_Svces','c_Manuf','JPN','SSA') = 179.4806976 ;
vtwr('c_Svces','c_Manuf','JPN','ROW') = 7472.270996 ;
vtwr('c_Svces','c_Manuf','IND','USA') = 724.3952637 ;
vtwr('c_Svces','c_Manuf','IND','EU_28') = 918.6466675 ;
vtwr('c_Svces','c_Manuf','IND','CHN') = 665.9892578 ;
vtwr('c_Svces','c_Manuf','IND','JPN') = 86.46348572 ;
vtwr('c_Svces','c_Manuf','IND','SSA') = 293.6043701 ;
vtwr('c_Svces','c_Manuf','IND','ROW') = 2394.486572 ;
vtwr('c_Svces','c_Manuf','SSA','USA') = 228.5377808 ;
vtwr('c_Svces','c_Manuf','SSA','EU_28') = 838.3413086 ;
vtwr('c_Svces','c_Manuf','SSA','CHN') = 1712.791016 ;
vtwr('c_Svces','c_Manuf','SSA','JPN') = 192.9337769 ;
vtwr('c_Svces','c_Manuf','SSA','IND') = 277.9377136 ;
vtwr('c_Svces','c_Manuf','SSA','SSA') = 1253.473022 ;
vtwr('c_Svces','c_Manuf','SSA','ROW') = 886.7958984 ;
vtwr('c_Svces','c_Manuf','ROW','USA') = 11787.56152 ;
vtwr('c_Svces','c_Manuf','ROW','EU_28') = 13850.30957 ;
vtwr('c_Svces','c_Manuf','ROW','CHN') = 23911.66602 ;
vtwr('c_Svces','c_Manuf','ROW','JPN') = 6086.797852 ;
vtwr('c_Svces','c_Manuf','ROW','IND') = 2833.081787 ;
vtwr('c_Svces','c_Manuf','ROW','SSA') = 1387.964111 ;
vtwr('c_Svces','c_Manuf','ROW','ROW') = 33747.55469 ;
vtwr('c_Svces','c_ForestFish','USA','EU_28') = 28.82236671 ;
vtwr('c_Svces','c_ForestFish','USA','CHN') = 73.49706268 ;
vtwr('c_Svces','c_ForestFish','USA','JPN') = 52.96520996 ;
vtwr('c_Svces','c_ForestFish','USA','IND') = 0.6935417652 ;
vtwr('c_Svces','c_ForestFish','USA','SSA') = 0.2788452506 ;
vtwr('c_Svces','c_ForestFish','USA','ROW') = 39.7217865 ;
vtwr('c_Svces','c_ForestFish','EU_28','USA') = 27.00686073 ;
vtwr('c_Svces','c_ForestFish','EU_28','EU_28') = 731.0102539 ;
vtwr('c_Svces','c_ForestFish','EU_28','CHN') = 59.43674088 ;
vtwr('c_Svces','c_ForestFish','EU_28','JPN') = 2.293114901 ;
vtwr('c_Svces','c_ForestFish','EU_28','IND') = 3.679128408 ;
vtwr('c_Svces','c_ForestFish','EU_28','SSA') = 1.701805949 ;
vtwr('c_Svces','c_ForestFish','EU_28','ROW') = 82.19512939 ;
vtwr('c_Svces','c_ForestFish','CHN','USA') = 3.61314106 ;
vtwr('c_Svces','c_ForestFish','CHN','EU_28') = 8.619956017 ;
vtwr('c_Svces','c_ForestFish','CHN','JPN') = 54.95546722 ;
vtwr('c_Svces','c_ForestFish','CHN','IND') = 0.04392611608 ;
vtwr('c_Svces','c_ForestFish','CHN','SSA') = 0.09420002997 ;
vtwr('c_Svces','c_ForestFish','CHN','ROW') = 81.32624054 ;
vtwr('c_Svces','c_ForestFish','JPN','USA') = 5.934758186 ;
vtwr('c_Svces','c_ForestFish','JPN','EU_28') = 3.354498625 ;
vtwr('c_Svces','c_ForestFish','JPN','CHN') = 14.10829639 ;
vtwr('c_Svces','c_ForestFish','JPN','IND') = 0.08474142104 ;
vtwr('c_Svces','c_ForestFish','JPN','SSA') = 0.1861739606 ;
vtwr('c_Svces','c_ForestFish','JPN','ROW') = 38.35958862 ;
vtwr('c_Svces','c_ForestFish','IND','USA') = 3.615754366 ;
vtwr('c_Svces','c_ForestFish','IND','EU_28') = 8.343405724 ;
vtwr('c_Svces','c_ForestFish','IND','CHN') = 9.837830544 ;
vtwr('c_Svces','c_ForestFish','IND','JPN') = 0.5072293282 ;
vtwr('c_Svces','c_ForestFish','IND','SSA') = 0.2678886652 ;
vtwr('c_Svces','c_ForestFish','IND','ROW') = 19.12003517 ;
vtwr('c_Svces','c_ForestFish','SSA','USA') = 4.257921219 ;
vtwr('c_Svces','c_ForestFish','SSA','EU_28') = 43.49824524 ;
vtwr('c_Svces','c_ForestFish','SSA','CHN') = 206.5236664 ;
vtwr('c_Svces','c_ForestFish','SSA','JPN') = 1.236085296 ;
vtwr('c_Svces','c_ForestFish','SSA','IND') = 16.32370567 ;
vtwr('c_Svces','c_ForestFish','SSA','SSA') = 23.93517876 ;
vtwr('c_Svces','c_ForestFish','SSA','ROW') = 55.06304169 ;
vtwr('c_Svces','c_ForestFish','ROW','USA') = 140.6786194 ;
vtwr('c_Svces','c_ForestFish','ROW','EU_28') = 331.2901917 ;
vtwr('c_Svces','c_ForestFish','ROW','CHN') = 767.9172363 ;
vtwr('c_Svces','c_ForestFish','ROW','JPN') = 129.3038025 ;
vtwr('c_Svces','c_ForestFish','ROW','IND') = 106.4781418 ;
vtwr('c_Svces','c_ForestFish','ROW','SSA') = 5.524363041 ;
vtwr('c_Svces','c_ForestFish','ROW','ROW') = 471.3364563 ;

* save data (7 cells)
save('USA') = 1580785.625 ;
save('EU_28') = 2133911.25 ;
save('CHN') = 3840168.5 ;
save('JPN') = 543592.4375 ;
save('IND') = 467486.4375 ;
save('SSA') = 209834.7656 ;
save('ROW') = 3415068 ;

* vdep data (7 cells)
vdep('USA') = 1816268 ;
vdep('EU_28') = 1809763.25 ;
vdep('CHN') = 1607693.375 ;
vdep('JPN') = 707913.5 ;
vdep('IND') = 217300.7188 ;
vdep('SSA') = 124198.7969 ;
vdep('ROW') = 2264923 ;

* vkb data (7 cells)
vkb('USA') = 45406700 ;
vkb('EU_28') = 45244080 ;
vkb('CHN') = 40192336 ;
vkb('JPN') = 17697838 ;
vkb('IND') = 5432518 ;
vkb('SSA') = 3104970 ;
vkb('ROW') = 56623080 ;

* maks data (70 cells)
maks('c_Rice','a_Rice','USA') = 10553.6875 ;
maks('c_Rice','a_Rice','EU_28') = 6879.371094 ;
maks('c_Rice','a_Rice','CHN') = 179204.8438 ;
maks('c_Rice','a_Rice','JPN') = 38168.63672 ;
maks('c_Rice','a_Rice','IND') = 84187.94531 ;
maks('c_Rice','a_Rice','SSA') = 17939.17383 ;
maks('c_Rice','a_Rice','ROW') = 365534.25 ;
maks('c_Crops','a_Crops','USA') = 227004.5156 ;
maks('c_Crops','a_Crops','EU_28') = 252454.1094 ;
maks('c_Crops','a_Crops','CHN') = 729052.125 ;
maks('c_Crops','a_Crops','JPN') = 34702.44141 ;
maks('c_Crops','a_Crops','IND') = 241484.8438 ;
maks('c_Crops','a_Crops','SSA') = 228635.1406 ;
maks('c_Crops','a_Crops','ROW') = 918591.9375 ;
maks('c_Livestock','a_Livestock','USA') = 173237.7656 ;
maks('c_Livestock','a_Livestock','EU_28') = 194522.7344 ;
maks('c_Livestock','a_Livestock','CHN') = 365296.8125 ;
maks('c_Livestock','a_Livestock','JPN') = 34414.39453 ;
maks('c_Livestock','a_Livestock','IND') = 126543.0391 ;
maks('c_Livestock','a_Livestock','SSA') = 69657.5 ;
maks('c_Livestock','a_Livestock','ROW') = 562039.5625 ;
maks('c_FoodProc','a_FoodProc','USA') = 1109233.875 ;
maks('c_FoodProc','a_FoodProc','EU_28') = 1455317.25 ;
maks('c_FoodProc','a_FoodProc','CHN') = 1628540.5 ;
maks('c_FoodProc','a_FoodProc','JPN') = 293179.2812 ;
maks('c_FoodProc','a_FoodProc','IND') = 202942.4531 ;
maks('c_FoodProc','a_FoodProc','SSA') = 301065 ;
maks('c_FoodProc','a_FoodProc','ROW') = 2286207.75 ;
maks('c_Energy','a_Energy','USA') = 1082570.75 ;
maks('c_Energy','a_Energy','EU_28') = 885865.4375 ;
maks('c_Energy','a_Energy','CHN') = 1266234 ;
maks('c_Energy','a_Energy','JPN') = 284079 ;
maks('c_Energy','a_Energy','IND') = 485696.0312 ;
maks('c_Energy','a_Energy','SSA') = 208941 ;
maks('c_Energy','a_Energy','ROW') = 3753127.5 ;
maks('c_Textiles','a_Textiles','USA') = 93223.86719 ;
maks('c_Textiles','a_Textiles','EU_28') = 420481.9062 ;
maks('c_Textiles','a_Textiles','CHN') = 1325668.625 ;
maks('c_Textiles','a_Textiles','JPN') = 40532.09375 ;
maks('c_Textiles','a_Textiles','IND') = 162754.3125 ;
maks('c_Textiles','a_Textiles','SSA') = 44125.28125 ;
maks('c_Textiles','a_Textiles','ROW') = 821641.625 ;
maks('c_Chem','a_Chem','USA') = 1029804.375 ;
maks('c_Chem','a_Chem','EU_28') = 1570803.75 ;
maks('c_Chem','a_Chem','CHN') = 1924916.25 ;
maks('c_Chem','a_Chem','JPN') = 443665 ;
maks('c_Chem','a_Chem','IND') = 280070.625 ;
maks('c_Chem','a_Chem','SSA') = 54723.60156 ;
maks('c_Chem','a_Chem','ROW') = 1809458.875 ;
maks('c_Manuf','a_Manuf','USA') = 3955209.25 ;
maks('c_Manuf','a_Manuf','EU_28') = 6110222 ;
maks('c_Manuf','a_Manuf','CHN') = 8743891 ;
maks('c_Manuf','a_Manuf','JPN') = 2231835.5 ;
maks('c_Manuf','a_Manuf','IND') = 741464.6875 ;
maks('c_Manuf','a_Manuf','SSA') = 391325.0625 ;
maks('c_Manuf','a_Manuf','ROW') = 7344946 ;
maks('c_ForestFish','a_ForestFish','USA') = 59205.60938 ;
maks('c_ForestFish','a_ForestFish','EU_28') = 77960.28906 ;
maks('c_ForestFish','a_ForestFish','CHN') = 246125.125 ;
maks('c_ForestFish','a_ForestFish','JPN') = 20446.3418 ;
maks('c_ForestFish','a_ForestFish','IND') = 74477.86719 ;
maks('c_ForestFish','a_ForestFish','SSA') = 68434.32812 ;
maks('c_ForestFish','a_ForestFish','ROW') = 276263.0625 ;
maks('c_Svces','a_Svces','USA') = 24328918 ;
maks('c_Svces','a_Svces','EU_28') = 22564282 ;
maks('c_Svces','a_Svces','CHN') = 14255377 ;
maks('c_Svces','a_Svces','JPN') = 5994771.5 ;
maks('c_Svces','a_Svces','IND') = 2622109.25 ;
maks('c_Svces','a_Svces','SSA') = 1569741.5 ;
maks('c_Svces','a_Svces','ROW') = 25016454 ;

* makb data (70 cells)
makb('c_Rice','a_Rice','USA') = 10640.89551 ;
makb('c_Rice','a_Rice','EU_28') = 7049.949707 ;
makb('c_Rice','a_Rice','CHN') = 179210.5625 ;
makb('c_Rice','a_Rice','JPN') = 38185.5625 ;
makb('c_Rice','a_Rice','IND') = 84165.10938 ;
makb('c_Rice','a_Rice','SSA') = 17966.63281 ;
makb('c_Rice','a_Rice','ROW') = 365505.0938 ;
makb('c_Crops','a_Crops','USA') = 226965.1406 ;
makb('c_Crops','a_Crops','EU_28') = 252414.6875 ;
makb('c_Crops','a_Crops','CHN') = 725310.6875 ;
makb('c_Crops','a_Crops','JPN') = 34097.39844 ;
makb('c_Crops','a_Crops','IND') = 241384 ;
makb('c_Crops','a_Crops','SSA') = 228558.4062 ;
makb('c_Crops','a_Crops','ROW') = 918871.25 ;
makb('c_Livestock','a_Livestock','USA') = 173235.5312 ;
makb('c_Livestock','a_Livestock','EU_28') = 193090.4062 ;
makb('c_Livestock','a_Livestock','CHN') = 365285.9062 ;
makb('c_Livestock','a_Livestock','JPN') = 33815.48047 ;
makb('c_Livestock','a_Livestock','IND') = 126532.6484 ;
makb('c_Livestock','a_Livestock','SSA') = 69559.41406 ;
makb('c_Livestock','a_Livestock','ROW') = 561631.0625 ;
makb('c_FoodProc','a_FoodProc','USA') = 1148684.875 ;
makb('c_FoodProc','a_FoodProc','EU_28') = 1457072.75 ;
makb('c_FoodProc','a_FoodProc','CHN') = 1628716.625 ;
makb('c_FoodProc','a_FoodProc','JPN') = 320801.8438 ;
makb('c_FoodProc','a_FoodProc','IND') = 202972.0469 ;
makb('c_FoodProc','a_FoodProc','SSA') = 301746.125 ;
makb('c_FoodProc','a_FoodProc','ROW') = 2302656.25 ;
makb('c_Energy','a_Energy','USA') = 1163954.125 ;
makb('c_Energy','a_Energy','EU_28') = 890106.5 ;
makb('c_Energy','a_Energy','CHN') = 1266388.875 ;
makb('c_Energy','a_Energy','JPN') = 319227 ;
makb('c_Energy','a_Energy','IND') = 485717.4062 ;
makb('c_Energy','a_Energy','SSA') = 209621.9688 ;
makb('c_Energy','a_Energy','ROW') = 3760574.5 ;
makb('c_Textiles','a_Textiles','USA') = 94411.70312 ;
makb('c_Textiles','a_Textiles','EU_28') = 421233.5312 ;
makb('c_Textiles','a_Textiles','CHN') = 1325838.625 ;
makb('c_Textiles','a_Textiles','JPN') = 42217.95703 ;
makb('c_Textiles','a_Textiles','IND') = 162757.5 ;
makb('c_Textiles','a_Textiles','SSA') = 44055.03516 ;
makb('c_Textiles','a_Textiles','ROW') = 824875.375 ;
makb('c_Chem','a_Chem','USA') = 1046756.25 ;
makb('c_Chem','a_Chem','EU_28') = 1575606.25 ;
makb('c_Chem','a_Chem','CHN') = 1925106.375 ;
makb('c_Chem','a_Chem','JPN') = 453806.25 ;
makb('c_Chem','a_Chem','IND') = 280075.2188 ;
makb('c_Chem','a_Chem','SSA') = 54778.69531 ;
makb('c_Chem','a_Chem','ROW') = 1818529.375 ;
makb('c_Manuf','a_Manuf','USA') = 4032121.75 ;
makb('c_Manuf','a_Manuf','EU_28') = 6129109 ;
makb('c_Manuf','a_Manuf','CHN') = 8744415 ;
makb('c_Manuf','a_Manuf','JPN') = 2256697.5 ;
makb('c_Manuf','a_Manuf','IND') = 741476.125 ;
makb('c_Manuf','a_Manuf','SSA') = 391283.5938 ;
makb('c_Manuf','a_Manuf','ROW') = 7375501.5 ;
makb('c_ForestFish','a_ForestFish','USA') = 61058.9375 ;
makb('c_ForestFish','a_ForestFish','EU_28') = 77485.84375 ;
makb('c_ForestFish','a_ForestFish','CHN') = 246139.1562 ;
makb('c_ForestFish','a_ForestFish','JPN') = 20915.87891 ;
makb('c_ForestFish','a_ForestFish','IND') = 74478.10938 ;
makb('c_ForestFish','a_ForestFish','SSA') = 68807.66406 ;
makb('c_ForestFish','a_ForestFish','ROW') = 274373.3438 ;
makb('c_Svces','a_Svces','USA') = 25332292 ;
makb('c_Svces','a_Svces','EU_28') = 22761286 ;
makb('c_Svces','a_Svces','CHN') = 14256120 ;
makb('c_Svces','a_Svces','JPN') = 6166068 ;
makb('c_Svces','a_Svces','IND') = 2622143.25 ;
makb('c_Svces','a_Svces','SSA') = 1580003.375 ;
makb('c_Svces','a_Svces','ROW') = 25255732 ;

* esubt (empty in GDX — relying on $onImplicitAssign suppression)

* esubc (empty in GDX — relying on $onImplicitAssign suppression)

* esubva data (70 cells)
esubva('a_Rice','USA') = 0.4664094746 ;
esubva('a_Rice','EU_28') = 0.4664094746 ;
esubva('a_Rice','CHN') = 0.4664094746 ;
esubva('a_Rice','JPN') = 0.4664094746 ;
esubva('a_Rice','IND') = 0.4664094746 ;
esubva('a_Rice','SSA') = 0.4664094746 ;
esubva('a_Rice','ROW') = 0.4664094746 ;
esubva('a_Crops','USA') = 0.2592099011 ;
esubva('a_Crops','EU_28') = 0.2592099011 ;
esubva('a_Crops','CHN') = 0.2592099011 ;
esubva('a_Crops','JPN') = 0.2592099011 ;
esubva('a_Crops','IND') = 0.2592099011 ;
esubva('a_Crops','SSA') = 0.2592099011 ;
esubva('a_Crops','ROW') = 0.2592099011 ;
esubva('a_Livestock','USA') = 0.2592099011 ;
esubva('a_Livestock','EU_28') = 0.2592099011 ;
esubva('a_Livestock','CHN') = 0.2592099011 ;
esubva('a_Livestock','JPN') = 0.2592099011 ;
esubva('a_Livestock','IND') = 0.2592099011 ;
esubva('a_Livestock','SSA') = 0.2592099011 ;
esubva('a_Livestock','ROW') = 0.2592099011 ;
esubva('a_FoodProc','USA') = 1.120000005 ;
esubva('a_FoodProc','EU_28') = 1.120000005 ;
esubva('a_FoodProc','CHN') = 1.120000005 ;
esubva('a_FoodProc','JPN') = 1.120000005 ;
esubva('a_FoodProc','IND') = 1.120000005 ;
esubva('a_FoodProc','SSA') = 1.120000005 ;
esubva('a_FoodProc','ROW') = 1.120000005 ;
esubva('a_Energy','USA') = 0.7166335583 ;
esubva('a_Energy','EU_28') = 0.7166335583 ;
esubva('a_Energy','CHN') = 0.7166335583 ;
esubva('a_Energy','JPN') = 0.7166335583 ;
esubva('a_Energy','IND') = 0.7166335583 ;
esubva('a_Energy','SSA') = 0.7166335583 ;
esubva('a_Energy','ROW') = 0.7166335583 ;
esubva('a_Textiles','USA') = 1.25999999 ;
esubva('a_Textiles','EU_28') = 1.25999999 ;
esubva('a_Textiles','CHN') = 1.25999999 ;
esubva('a_Textiles','JPN') = 1.25999999 ;
esubva('a_Textiles','IND') = 1.25999999 ;
esubva('a_Textiles','SSA') = 1.25999999 ;
esubva('a_Textiles','ROW') = 1.25999999 ;
esubva('a_Chem','USA') = 1.25999999 ;
esubva('a_Chem','EU_28') = 1.25999999 ;
esubva('a_Chem','CHN') = 1.25999999 ;
esubva('a_Chem','JPN') = 1.25999999 ;
esubva('a_Chem','IND') = 1.25999999 ;
esubva('a_Chem','SSA') = 1.25999999 ;
esubva('a_Chem','ROW') = 1.25999999 ;
esubva('a_Manuf','USA') = 1.178595543 ;
esubva('a_Manuf','EU_28') = 1.178595543 ;
esubva('a_Manuf','CHN') = 1.178595543 ;
esubva('a_Manuf','JPN') = 1.178595543 ;
esubva('a_Manuf','IND') = 1.178595543 ;
esubva('a_Manuf','SSA') = 1.178595543 ;
esubva('a_Manuf','ROW') = 1.178595543 ;
esubva('a_ForestFish','USA') = 0.200000003 ;
esubva('a_ForestFish','EU_28') = 0.200000003 ;
esubva('a_ForestFish','CHN') = 0.200000003 ;
esubva('a_ForestFish','JPN') = 0.200000003 ;
esubva('a_ForestFish','IND') = 0.200000003 ;
esubva('a_ForestFish','SSA') = 0.200000003 ;
esubva('a_ForestFish','ROW') = 0.200000003 ;
esubva('a_Svces','USA') = 1.37246573 ;
esubva('a_Svces','EU_28') = 1.37246573 ;
esubva('a_Svces','CHN') = 1.37246573 ;
esubva('a_Svces','JPN') = 1.37246573 ;
esubva('a_Svces','IND') = 1.37246573 ;
esubva('a_Svces','SSA') = 1.37246573 ;
esubva('a_Svces','ROW') = 1.37246573 ;

* etraq data (70 cells)
etraq('a_Rice','USA') = -5 ;
etraq('a_Rice','EU_28') = -5 ;
etraq('a_Rice','CHN') = -4.999999523 ;
etraq('a_Rice','JPN') = -5 ;
etraq('a_Rice','IND') = -5 ;
etraq('a_Rice','SSA') = -5 ;
etraq('a_Rice','ROW') = -5 ;
etraq('a_Crops','USA') = -5 ;
etraq('a_Crops','EU_28') = -5 ;
etraq('a_Crops','CHN') = -5 ;
etraq('a_Crops','JPN') = -5 ;
etraq('a_Crops','IND') = -5 ;
etraq('a_Crops','SSA') = -5 ;
etraq('a_Crops','ROW') = -5 ;
etraq('a_Livestock','USA') = -5 ;
etraq('a_Livestock','EU_28') = -5 ;
etraq('a_Livestock','CHN') = -5 ;
etraq('a_Livestock','JPN') = -5 ;
etraq('a_Livestock','IND') = -5 ;
etraq('a_Livestock','SSA') = -5 ;
etraq('a_Livestock','ROW') = -5 ;
etraq('a_FoodProc','USA') = -5 ;
etraq('a_FoodProc','EU_28') = -5 ;
etraq('a_FoodProc','CHN') = -5 ;
etraq('a_FoodProc','JPN') = -5 ;
etraq('a_FoodProc','IND') = -5 ;
etraq('a_FoodProc','SSA') = -5 ;
etraq('a_FoodProc','ROW') = -5 ;
etraq('a_Energy','USA') = -5 ;
etraq('a_Energy','EU_28') = -5 ;
etraq('a_Energy','CHN') = -5 ;
etraq('a_Energy','JPN') = -5 ;
etraq('a_Energy','IND') = -5 ;
etraq('a_Energy','SSA') = -5 ;
etraq('a_Energy','ROW') = -5 ;
etraq('a_Textiles','USA') = -5 ;
etraq('a_Textiles','EU_28') = -5 ;
etraq('a_Textiles','CHN') = -5 ;
etraq('a_Textiles','JPN') = -5 ;
etraq('a_Textiles','IND') = -5 ;
etraq('a_Textiles','SSA') = -5 ;
etraq('a_Textiles','ROW') = -5 ;
etraq('a_Chem','USA') = -5 ;
etraq('a_Chem','EU_28') = -5 ;
etraq('a_Chem','CHN') = -5 ;
etraq('a_Chem','JPN') = -5 ;
etraq('a_Chem','IND') = -5 ;
etraq('a_Chem','SSA') = -5 ;
etraq('a_Chem','ROW') = -5 ;
etraq('a_Manuf','USA') = -5 ;
etraq('a_Manuf','EU_28') = -5 ;
etraq('a_Manuf','CHN') = -5 ;
etraq('a_Manuf','JPN') = -5 ;
etraq('a_Manuf','IND') = -5 ;
etraq('a_Manuf','SSA') = -5 ;
etraq('a_Manuf','ROW') = -5 ;
etraq('a_ForestFish','USA') = -5 ;
etraq('a_ForestFish','EU_28') = -5 ;
etraq('a_ForestFish','CHN') = -5 ;
etraq('a_ForestFish','JPN') = -5 ;
etraq('a_ForestFish','IND') = -5 ;
etraq('a_ForestFish','SSA') = -5 ;
etraq('a_ForestFish','ROW') = -5 ;
etraq('a_Svces','USA') = -5 ;
etraq('a_Svces','EU_28') = -5 ;
etraq('a_Svces','CHN') = -5 ;
etraq('a_Svces','JPN') = -5 ;
etraq('a_Svces','IND') = -5 ;
etraq('a_Svces','SSA') = -5 ;
etraq('a_Svces','ROW') = -5 ;

* esubq (empty in GDX — relying on $onImplicitAssign suppression)

* incpar data (70 cells)
incpar('c_Rice','USA') = 9.999999975e-07 ;
incpar('c_Rice','EU_28') = 0.002127082553 ;
incpar('c_Rice','CHN') = 0.1202474609 ;
incpar('c_Rice','JPN') = 9.999999975e-07 ;
incpar('c_Rice','IND') = 0.4121294618 ;
incpar('c_Rice','SSA') = 0.5564925671 ;
incpar('c_Rice','ROW') = 0.3055584431 ;
incpar('c_Crops','USA') = 0.0426319465 ;
incpar('c_Crops','EU_28') = 0.006428936031 ;
incpar('c_Crops','CHN') = 0.1209676266 ;
incpar('c_Crops','JPN') = 0.003530829446 ;
incpar('c_Crops','IND') = 0.4131002128 ;
incpar('c_Crops','SSA') = 0.5895308852 ;
incpar('c_Crops','ROW') = 0.2211325616 ;
incpar('c_Livestock','USA') = 0.5035835505 ;
incpar('c_Livestock','EU_28') = 0.5541610122 ;
incpar('c_Livestock','CHN') = 0.4628626108 ;
incpar('c_Livestock','JPN') = 0.5488638282 ;
incpar('c_Livestock','IND') = 0.6080049276 ;
incpar('c_Livestock','SSA') = 0.8615323305 ;
incpar('c_Livestock','ROW') = 0.5765627623 ;
incpar('c_FoodProc','USA') = 0.4122424424 ;
incpar('c_FoodProc','EU_28') = 0.4656523168 ;
incpar('c_FoodProc','CHN') = 0.3850696087 ;
incpar('c_FoodProc','JPN') = 0.4367224872 ;
incpar('c_FoodProc','IND') = 0.574326694 ;
incpar('c_FoodProc','SSA') = 0.6866018176 ;
incpar('c_FoodProc','ROW') = 0.4553508162 ;
incpar('c_Energy','USA') = 0.9133253694 ;
incpar('c_Energy','EU_28') = 0.9933156967 ;
incpar('c_Energy','CHN') = 1.022505641 ;
incpar('c_Energy','JPN') = 0.9625540376 ;
incpar('c_Energy','IND') = 0.9281172752 ;
incpar('c_Energy','SSA') = 1.040355086 ;
incpar('c_Energy','ROW') = 1.001412392 ;
incpar('c_Textiles','USA') = 0.490339607 ;
incpar('c_Textiles','EU_28') = 0.5438744426 ;
incpar('c_Textiles','CHN') = 0.4493468404 ;
incpar('c_Textiles','JPN') = 0.5349581242 ;
incpar('c_Textiles','IND') = 0.6007944942 ;
incpar('c_Textiles','SSA') = 0.7493164539 ;
incpar('c_Textiles','ROW') = 0.5290850997 ;
incpar('c_Chem','USA') = 0.9594091773 ;
incpar('c_Chem','EU_28') = 1.040493965 ;
incpar('c_Chem','CHN') = 1.055357337 ;
incpar('c_Chem','JPN') = 1.009908676 ;
incpar('c_Chem','IND') = 0.8242521286 ;
incpar('c_Chem','SSA') = 0.929810226 ;
incpar('c_Chem','ROW') = 1.031634808 ;
incpar('c_Manuf','USA') = 0.9517101645 ;
incpar('c_Manuf','EU_28') = 1.033332825 ;
incpar('c_Manuf','CHN') = 1.038700104 ;
incpar('c_Manuf','JPN') = 1.004759431 ;
incpar('c_Manuf','IND') = 0.8267973661 ;
incpar('c_Manuf','SSA') = 0.9444041252 ;
incpar('c_Manuf','ROW') = 1.01845932 ;
incpar('c_ForestFish','USA') = 0.5593986511 ;
incpar('c_ForestFish','EU_28') = 0.7426393032 ;
incpar('c_ForestFish','CHN') = 0.478631407 ;
incpar('c_ForestFish','JPN') = 0.7029562593 ;
incpar('c_ForestFish','IND') = 0.6821090579 ;
incpar('c_ForestFish','SSA') = 0.7866491675 ;
incpar('c_ForestFish','ROW') = 0.6687270999 ;
incpar('c_Svces','USA') = 1.061069846 ;
incpar('c_Svces','EU_28') = 1.124576092 ;
incpar('c_Svces','CHN') = 1.366120815 ;
incpar('c_Svces','JPN') = 1.100636721 ;
incpar('c_Svces','IND') = 1.310168743 ;
incpar('c_Svces','SSA') = 1.381296515 ;
incpar('c_Svces','ROW') = 1.178853035 ;

* subpar data (70 cells)
subpar('c_Rice','USA') = 0.9999990463 ;
subpar('c_Rice','EU_28') = 0.9949424863 ;
subpar('c_Rice','CHN') = 0.8740237951 ;
subpar('c_Rice','JPN') = 0.9999990463 ;
subpar('c_Rice','IND') = 0.8446068168 ;
subpar('c_Rice','SSA') = 0.8567221761 ;
subpar('c_Rice','ROW') = 0.8840133548 ;
subpar('c_Crops','USA') = 0.9272732735 ;
subpar('c_Crops','EU_28') = 0.9904057384 ;
subpar('c_Crops','CHN') = 0.8735871315 ;
subpar('c_Crops','JPN') = 0.9955404997 ;
subpar('c_Crops','IND') = 0.8442919254 ;
subpar('c_Crops','SSA') = 0.8542761803 ;
subpar('c_Crops','ROW') = 0.9007891417 ;
subpar('c_Livestock','USA') = 0.159861207 ;
subpar('c_Livestock','EU_28') = 0.3749217093 ;
subpar('c_Livestock','CHN') = 0.671416223 ;
subpar('c_Livestock','JPN') = 0.3188154995 ;
subpar('c_Livestock','IND') = 0.7840914726 ;
subpar('c_Livestock','SSA') = 0.7964909673 ;
subpar('c_Livestock','ROW') = 0.6884165406 ;
subpar('c_FoodProc','USA') = 0.1924026757 ;
subpar('c_FoodProc','EU_28') = 0.3960759938 ;
subpar('c_FoodProc','CHN') = 0.7169975042 ;
subpar('c_FoodProc','JPN') = 0.3747712076 ;
subpar('c_FoodProc','IND') = 0.7923504114 ;
subpar('c_FoodProc','SSA') = 0.7970803976 ;
subpar('c_FoodProc','ROW') = 0.6416326761 ;
subpar('c_Energy','USA') = 0.09387533367 ;
subpar('c_Energy','EU_28') = 0.2317585796 ;
subpar('c_Energy','CHN') = 0.4935285449 ;
subpar('c_Energy','JPN') = 0.2136103213 ;
subpar('c_Energy','IND') = 0.7051666379 ;
subpar('c_Energy','SSA') = 0.6626578569 ;
subpar('c_Energy','ROW') = 0.44322294 ;
subpar('c_Textiles','USA') = 0.1635115743 ;
subpar('c_Textiles','EU_28') = 0.3407291174 ;
subpar('c_Textiles','CHN') = 0.6744819283 ;
subpar('c_Textiles','JPN') = 0.3242999315 ;
subpar('c_Textiles','IND') = 0.7834095359 ;
subpar('c_Textiles','SSA') = 0.7827576399 ;
subpar('c_Textiles','ROW') = 0.5901503563 ;
subpar('c_Chem','USA') = 0.08916409314 ;
subpar('c_Chem','EU_28') = 0.2280205786 ;
subpar('c_Chem','CHN') = 0.4869003892 ;
subpar('c_Chem','JPN') = 0.2049494237 ;
subpar('c_Chem','IND') = 0.7276021838 ;
subpar('c_Chem','SSA') = 0.7167853117 ;
subpar('c_Chem','ROW') = 0.4331519902 ;
subpar('c_Manuf','USA') = 0.08993383497 ;
subpar('c_Manuf','EU_28') = 0.2172584087 ;
subpar('c_Manuf','CHN') = 0.4907082617 ;
subpar('c_Manuf','JPN') = 0.205864653 ;
subpar('c_Manuf','IND') = 0.7270501852 ;
subpar('c_Manuf','SSA') = 0.6940402389 ;
subpar('c_Manuf','ROW') = 0.3916885853 ;
subpar('c_ForestFish','USA') = 0.1512012929 ;
subpar('c_ForestFish','EU_28') = 0.3289707899 ;
subpar('c_ForestFish','CHN') = 0.6665375829 ;
subpar('c_ForestFish','JPN') = 0.280757606 ;
subpar('c_ForestFish','IND') = 0.7647792697 ;
subpar('c_ForestFish','SSA') = 0.7846815586 ;
subpar('c_ForestFish','ROW') = 0.6729961634 ;
subpar('c_Svces','USA') = 0.08792939782 ;
subpar('c_Svces','EU_28') = 0.2063865215 ;
subpar('c_Svces','CHN') = 0.4421200156 ;
subpar('c_Svces','JPN') = 0.1965861768 ;
subpar('c_Svces','IND') = 0.6477729678 ;
subpar('c_Svces','SSA') = 0.6090341806 ;
subpar('c_Svces','ROW') = 0.3598118424 ;

* esubg data (7 cells)
esubg('USA') = 1 ;
esubg('EU_28') = 1 ;
esubg('CHN') = 1 ;
esubg('JPN') = 1 ;
esubg('IND') = 1 ;
esubg('SSA') = 1 ;
esubg('ROW') = 1 ;

* esubi (empty in GDX — relying on $onImplicitAssign suppression)

* esubd data (70 cells)
esubd('c_Rice','USA') = 3.869595766 ;
esubd('c_Rice','EU_28') = 3.869595766 ;
esubd('c_Rice','CHN') = 3.869595766 ;
esubd('c_Rice','JPN') = 3.869595766 ;
esubd('c_Rice','IND') = 3.869595766 ;
esubd('c_Rice','SSA') = 3.869595766 ;
esubd('c_Rice','ROW') = 3.869595766 ;
esubd('c_Crops','USA') = 2.244109392 ;
esubd('c_Crops','EU_28') = 2.244109392 ;
esubd('c_Crops','CHN') = 2.244109392 ;
esubd('c_Crops','JPN') = 2.244109392 ;
esubd('c_Crops','IND') = 2.244109392 ;
esubd('c_Crops','SSA') = 2.244109392 ;
esubd('c_Crops','ROW') = 2.244109392 ;
esubd('c_Livestock','USA') = 2.15929389 ;
esubd('c_Livestock','EU_28') = 2.15929389 ;
esubd('c_Livestock','CHN') = 2.15929389 ;
esubd('c_Livestock','JPN') = 2.15929389 ;
esubd('c_Livestock','IND') = 2.15929389 ;
esubd('c_Livestock','SSA') = 2.15929389 ;
esubd('c_Livestock','ROW') = 2.15929389 ;
esubd('c_FoodProc','USA') = 2.478586674 ;
esubd('c_FoodProc','EU_28') = 2.478586674 ;
esubd('c_FoodProc','CHN') = 2.478586674 ;
esubd('c_FoodProc','JPN') = 2.478586674 ;
esubd('c_FoodProc','IND') = 2.478586674 ;
esubd('c_FoodProc','SSA') = 2.478586674 ;
esubd('c_FoodProc','ROW') = 2.478586674 ;
esubd('c_Energy','USA') = 3.818507195 ;
esubd('c_Energy','EU_28') = 3.818507195 ;
esubd('c_Energy','CHN') = 3.818507195 ;
esubd('c_Energy','JPN') = 3.818507195 ;
esubd('c_Energy','IND') = 3.818507195 ;
esubd('c_Energy','SSA') = 3.818507195 ;
esubd('c_Energy','ROW') = 3.818507195 ;
esubd('c_Textiles','USA') = 3.779940128 ;
esubd('c_Textiles','EU_28') = 3.779940128 ;
esubd('c_Textiles','CHN') = 3.779940128 ;
esubd('c_Textiles','JPN') = 3.779940128 ;
esubd('c_Textiles','IND') = 3.779940128 ;
esubd('c_Textiles','SSA') = 3.779940128 ;
esubd('c_Textiles','ROW') = 3.779940128 ;
esubd('c_Chem','USA') = 3.299999952 ;
esubd('c_Chem','EU_28') = 3.299999952 ;
esubd('c_Chem','CHN') = 3.299999952 ;
esubd('c_Chem','JPN') = 3.299999952 ;
esubd('c_Chem','IND') = 3.299999952 ;
esubd('c_Chem','SSA') = 3.299999952 ;
esubd('c_Chem','ROW') = 3.299999952 ;
esubd('c_Manuf','USA') = 3.547324181 ;
esubd('c_Manuf','EU_28') = 3.547324181 ;
esubd('c_Manuf','CHN') = 3.547324181 ;
esubd('c_Manuf','JPN') = 3.547324181 ;
esubd('c_Manuf','IND') = 3.547324181 ;
esubd('c_Manuf','SSA') = 3.547324181 ;
esubd('c_Manuf','ROW') = 3.547324181 ;
esubd('c_ForestFish','USA') = 1.829421639 ;
esubd('c_ForestFish','EU_28') = 1.829421639 ;
esubd('c_ForestFish','CHN') = 1.829421639 ;
esubd('c_ForestFish','JPN') = 1.829421639 ;
esubd('c_ForestFish','IND') = 1.829421639 ;
esubd('c_ForestFish','SSA') = 1.829421639 ;
esubd('c_ForestFish','ROW') = 1.829421639 ;
esubd('c_Svces','USA') = 1.909135103 ;
esubd('c_Svces','EU_28') = 1.909135103 ;
esubd('c_Svces','CHN') = 1.909135103 ;
esubd('c_Svces','JPN') = 1.909135103 ;
esubd('c_Svces','IND') = 1.909135103 ;
esubd('c_Svces','SSA') = 1.909135103 ;
esubd('c_Svces','ROW') = 1.909135103 ;

* esubm data (70 cells)
esubm('c_Rice','USA') = 5.354156017 ;
esubm('c_Rice','EU_28') = 5.354156017 ;
esubm('c_Rice','CHN') = 5.354156017 ;
esubm('c_Rice','JPN') = 5.354156017 ;
esubm('c_Rice','IND') = 5.354156017 ;
esubm('c_Rice','SSA') = 5.354156017 ;
esubm('c_Rice','ROW') = 5.354156017 ;
esubm('c_Crops','USA') = 4.815529346 ;
esubm('c_Crops','EU_28') = 4.815529346 ;
esubm('c_Crops','CHN') = 4.815529346 ;
esubm('c_Crops','JPN') = 4.815529346 ;
esubm('c_Crops','IND') = 4.815529346 ;
esubm('c_Crops','SSA') = 4.815529346 ;
esubm('c_Crops','ROW') = 4.815529346 ;
esubm('c_Livestock','USA') = 4.020486355 ;
esubm('c_Livestock','EU_28') = 4.020486355 ;
esubm('c_Livestock','CHN') = 4.020486355 ;
esubm('c_Livestock','JPN') = 4.020486355 ;
esubm('c_Livestock','IND') = 4.020486355 ;
esubm('c_Livestock','SSA') = 4.020486355 ;
esubm('c_Livestock','ROW') = 4.020486355 ;
esubm('c_FoodProc','USA') = 4.813882351 ;
esubm('c_FoodProc','EU_28') = 4.813882351 ;
esubm('c_FoodProc','CHN') = 4.813882351 ;
esubm('c_FoodProc','JPN') = 4.813882351 ;
esubm('c_FoodProc','IND') = 4.813882351 ;
esubm('c_FoodProc','SSA') = 4.813882351 ;
esubm('c_FoodProc','ROW') = 4.813882351 ;
esubm('c_Energy','USA') = 11.49682426 ;
esubm('c_Energy','EU_28') = 11.49682426 ;
esubm('c_Energy','CHN') = 11.49682426 ;
esubm('c_Energy','JPN') = 11.49682426 ;
esubm('c_Energy','IND') = 11.49682426 ;
esubm('c_Energy','SSA') = 11.49682426 ;
esubm('c_Energy','ROW') = 11.49682426 ;
esubm('c_Textiles','USA') = 7.589058399 ;
esubm('c_Textiles','EU_28') = 7.589058399 ;
esubm('c_Textiles','CHN') = 7.589058399 ;
esubm('c_Textiles','JPN') = 7.589058399 ;
esubm('c_Textiles','IND') = 7.589058399 ;
esubm('c_Textiles','SSA') = 7.589058399 ;
esubm('c_Textiles','ROW') = 7.589058399 ;
esubm('c_Chem','USA') = 6.599999905 ;
esubm('c_Chem','EU_28') = 6.599999905 ;
esubm('c_Chem','CHN') = 6.599999905 ;
esubm('c_Chem','JPN') = 6.599999905 ;
esubm('c_Chem','IND') = 6.599999905 ;
esubm('c_Chem','SSA') = 6.599999905 ;
esubm('c_Chem','ROW') = 6.599999905 ;
esubm('c_Manuf','USA') = 7.461285591 ;
esubm('c_Manuf','EU_28') = 7.461285591 ;
esubm('c_Manuf','CHN') = 7.461285591 ;
esubm('c_Manuf','JPN') = 7.461285591 ;
esubm('c_Manuf','IND') = 7.461285591 ;
esubm('c_Manuf','SSA') = 7.461285591 ;
esubm('c_Manuf','ROW') = 7.461285591 ;
esubm('c_ForestFish','USA') = 3.603965282 ;
esubm('c_ForestFish','EU_28') = 3.603965282 ;
esubm('c_ForestFish','CHN') = 3.603965282 ;
esubm('c_ForestFish','JPN') = 3.603965282 ;
esubm('c_ForestFish','IND') = 3.603965282 ;
esubm('c_ForestFish','SSA') = 3.603965282 ;
esubm('c_ForestFish','ROW') = 3.603965282 ;
esubm('c_Svces','USA') = 3.800415277 ;
esubm('c_Svces','EU_28') = 3.800415277 ;
esubm('c_Svces','CHN') = 3.800415277 ;
esubm('c_Svces','JPN') = 3.800415277 ;
esubm('c_Svces','IND') = 3.800415277 ;
esubm('c_Svces','SSA') = 3.800415277 ;
esubm('c_Svces','ROW') = 3.800415277 ;

* esubs data (1 cells)
esubs('c_Svces') = 1 ;

* etrae data (14 cells)
etrae('Land','USA') = -1 ;
etrae('Land','EU_28') = -1 ;
etrae('Land','CHN') = -1 ;
etrae('Land','JPN') = -1 ;
etrae('Land','IND') = -1 ;
etrae('Land','SSA') = -1 ;
etrae('Land','ROW') = -1 ;
etrae('NatRes','USA') = -0.001000000047 ;
etrae('NatRes','EU_28') = -0.001000000047 ;
etrae('NatRes','CHN') = -0.001000000047 ;
etrae('NatRes','JPN') = -0.001000000047 ;
etrae('NatRes','IND') = -0.001000000047 ;
etrae('NatRes','SSA') = -0.001000000047 ;
etrae('NatRes','ROW') = -0.001000000047 ;

* rorFlex0 data (7 cells)
rorFlex0('USA') = 10 ;
rorFlex0('EU_28') = 10 ;
rorFlex0('CHN') = 10 ;
rorFlex0('JPN') = 10 ;
rorFlex0('IND') = 10 ;
rorFlex0('SSA') = 10 ;
rorFlex0('ROW') = 10 ;

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
   ROW
/ ;

set rmuv(r) "RMUV regions" /
   USA, EU_28, CHN, JPN, IND, SSA, ROW
/ ;

set imuv(i) "IMUV commodities" /
   c_Rice, c_Crops, c_Livestock, c_FoodProc, c_Energy, c_Textiles, c_Chem, c_Manuf, c_ForestFish, c_Svces
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

   options limrow = 3, limcol = 3, solprint = off, iterlim = 1000 ;

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
