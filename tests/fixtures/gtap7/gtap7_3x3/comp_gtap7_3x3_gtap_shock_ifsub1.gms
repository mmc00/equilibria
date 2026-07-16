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
   a_Food,
   a_Mnfcs,
   a_Svces
/;

Set comm 'Set COMM  Commodities' /
   c_Food,
   c_Mnfcs,
   c_Svces
/;

Set reg 'Set REG  Regions' /
   USA,
   EU_28,
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
* vdfb data (27 cells)
vdfb('c_Food','a_Food','USA') = 542355 ;
vdfb('c_Food','a_Food','EU_28') = 500638.0625 ;
vdfb('c_Food','a_Food','ROW') = 2901770.75 ;
vdfb('c_Food','a_Mnfcs','USA') = 41297.44922 ;
vdfb('c_Food','a_Mnfcs','EU_28') = 48286.37891 ;
vdfb('c_Food','a_Mnfcs','ROW') = 505785.125 ;
vdfb('c_Food','a_Svces','USA') = 177164.1875 ;
vdfb('c_Food','a_Svces','EU_28') = 243287.625 ;
vdfb('c_Food','a_Svces','ROW') = 1010040.562 ;
vdfb('c_Mnfcs','a_Food','USA') = 137371.9844 ;
vdfb('c_Mnfcs','a_Food','EU_28') = 106197.9219 ;
vdfb('c_Mnfcs','a_Food','ROW') = 483896.0938 ;
vdfb('c_Mnfcs','a_Mnfcs','USA') = 1611896.25 ;
vdfb('c_Mnfcs','a_Mnfcs','EU_28') = 1839815.75 ;
vdfb('c_Mnfcs','a_Mnfcs','ROW') = 11147014 ;
vdfb('c_Mnfcs','a_Svces','USA') = 1283868.25 ;
vdfb('c_Mnfcs','a_Svces','EU_28') = 1113946.25 ;
vdfb('c_Mnfcs','a_Svces','ROW') = 5528694 ;
vdfb('c_Svces','a_Food','USA') = 331707.6562 ;
vdfb('c_Svces','a_Food','EU_28') = 404140.875 ;
vdfb('c_Svces','a_Food','ROW') = 1365169.875 ;
vdfb('c_Svces','a_Mnfcs','USA') = 1069950.375 ;
vdfb('c_Svces','a_Mnfcs','EU_28') = 1605810.625 ;
vdfb('c_Svces','a_Mnfcs','ROW') = 5065759.5 ;
vdfb('c_Svces','a_Svces','USA') = 7592817.5 ;
vdfb('c_Svces','a_Svces','EU_28') = 6877007.5 ;
vdfb('c_Svces','a_Svces','ROW') = 13860507 ;

* vdfp data (27 cells)
vdfp('c_Food','a_Food','USA') = 542169.5 ;
vdfp('c_Food','a_Food','EU_28') = 497018.5625 ;
vdfp('c_Food','a_Food','ROW') = 2896536.75 ;
vdfp('c_Food','a_Mnfcs','USA') = 41122.10938 ;
vdfp('c_Food','a_Mnfcs','EU_28') = 48691.21875 ;
vdfp('c_Food','a_Mnfcs','ROW') = 511651.5938 ;
vdfp('c_Food','a_Svces','USA') = 177681.8281 ;
vdfp('c_Food','a_Svces','EU_28') = 260224.7344 ;
vdfp('c_Food','a_Svces','ROW') = 1043322.688 ;
vdfp('c_Mnfcs','a_Food','USA') = 139415.3906 ;
vdfp('c_Mnfcs','a_Food','EU_28') = 111128.0781 ;
vdfp('c_Mnfcs','a_Food','ROW') = 491771.1562 ;
vdfp('c_Mnfcs','a_Mnfcs','USA') = 1669532.125 ;
vdfp('c_Mnfcs','a_Mnfcs','EU_28') = 1874450.375 ;
vdfp('c_Mnfcs','a_Mnfcs','ROW') = 11450795 ;
vdfp('c_Mnfcs','a_Svces','USA') = 1348394.875 ;
vdfp('c_Mnfcs','a_Svces','EU_28') = 1253490 ;
vdfp('c_Mnfcs','a_Svces','ROW') = 5716079 ;
vdfp('c_Svces','a_Food','USA') = 331101.875 ;
vdfp('c_Svces','a_Food','EU_28') = 408120.375 ;
vdfp('c_Svces','a_Food','ROW') = 1357937 ;
vdfp('c_Svces','a_Mnfcs','USA') = 1086692.25 ;
vdfp('c_Svces','a_Mnfcs','EU_28') = 1651084.25 ;
vdfp('c_Svces','a_Mnfcs','ROW') = 5127382 ;
vdfp('c_Svces','a_Svces','USA') = 7656514 ;
vdfp('c_Svces','a_Svces','EU_28') = 7102183 ;
vdfp('c_Svces','a_Svces','ROW') = 14101464 ;

* vmfb data (27 cells)
vmfb('c_Food','a_Food','USA') = 44631.125 ;
vmfb('c_Food','a_Food','EU_28') = 188689.5938 ;
vmfb('c_Food','a_Food','ROW') = 335635.3438 ;
vmfb('c_Food','a_Mnfcs','USA') = 2144.677979 ;
vmfb('c_Food','a_Mnfcs','EU_28') = 26389.56641 ;
vmfb('c_Food','a_Mnfcs','ROW') = 43227.74609 ;
vmfb('c_Food','a_Svces','USA') = 23047.59961 ;
vmfb('c_Food','a_Svces','EU_28') = 73303.75 ;
vmfb('c_Food','a_Svces','ROW') = 130346.7578 ;
vmfb('c_Mnfcs','a_Food','USA') = 30397.6543 ;
vmfb('c_Mnfcs','a_Food','EU_28') = 76841.01562 ;
vmfb('c_Mnfcs','a_Food','ROW') = 192434.2031 ;
vmfb('c_Mnfcs','a_Mnfcs','USA') = 664758.25 ;
vmfb('c_Mnfcs','a_Mnfcs','EU_28') = 2185452.5 ;
vmfb('c_Mnfcs','a_Mnfcs','ROW') = 3972290.5 ;
vmfb('c_Mnfcs','a_Svces','USA') = 370647.5 ;
vmfb('c_Mnfcs','a_Svces','EU_28') = 876487 ;
vmfb('c_Mnfcs','a_Svces','ROW') = 1688362.25 ;
vmfb('c_Svces','a_Food','USA') = 4509.246094 ;
vmfb('c_Svces','a_Food','EU_28') = 47619.97266 ;
vmfb('c_Svces','a_Food','ROW') = 44004.40625 ;
vmfb('c_Svces','a_Mnfcs','USA') = 23632.78125 ;
vmfb('c_Svces','a_Mnfcs','EU_28') = 250204.2031 ;
vmfb('c_Svces','a_Mnfcs','ROW') = 222973.7812 ;
vmfb('c_Svces','a_Svces','USA') = 225410.625 ;
vmfb('c_Svces','a_Svces','EU_28') = 1195691.375 ;
vmfb('c_Svces','a_Svces','ROW') = 921570.5 ;

* vmfp data (27 cells)
vmfp('c_Food','a_Food','USA') = 44721.35938 ;
vmfp('c_Food','a_Food','EU_28') = 190533.8906 ;
vmfp('c_Food','a_Food','ROW') = 345928.1875 ;
vmfp('c_Food','a_Mnfcs','USA') = 2142.175537 ;
vmfp('c_Food','a_Mnfcs','EU_28') = 27106.54688 ;
vmfp('c_Food','a_Mnfcs','ROW') = 45766.32422 ;
vmfp('c_Food','a_Svces','USA') = 23125.10938 ;
vmfp('c_Food','a_Svces','EU_28') = 80344.78906 ;
vmfp('c_Food','a_Svces','ROW') = 139387.9688 ;
vmfp('c_Mnfcs','a_Food','USA') = 31042.53711 ;
vmfp('c_Mnfcs','a_Food','EU_28') = 79483.52344 ;
vmfp('c_Mnfcs','a_Food','ROW') = 195033.1562 ;
vmfp('c_Mnfcs','a_Mnfcs','USA') = 688171 ;
vmfp('c_Mnfcs','a_Mnfcs','EU_28') = 2220816.5 ;
vmfp('c_Mnfcs','a_Mnfcs','ROW') = 4138977.25 ;
vmfp('c_Mnfcs','a_Svces','USA') = 395723.5 ;
vmfp('c_Mnfcs','a_Svces','EU_28') = 990513.75 ;
vmfp('c_Mnfcs','a_Svces','ROW') = 1768015.875 ;
vmfp('c_Svces','a_Food','USA') = 4503.521973 ;
vmfp('c_Svces','a_Food','EU_28') = 47888.42969 ;
vmfp('c_Svces','a_Food','ROW') = 44102.89453 ;
vmfp('c_Svces','a_Mnfcs','USA') = 23847.4082 ;
vmfp('c_Svces','a_Mnfcs','EU_28') = 253778.3906 ;
vmfp('c_Svces','a_Mnfcs','ROW') = 227237.5469 ;
vmfp('c_Svces','a_Svces','USA') = 228811.7344 ;
vmfp('c_Svces','a_Svces','EU_28') = 1238011 ;
vmfp('c_Svces','a_Svces','ROW') = 943045.625 ;

* vdpb data (9 cells)
vdpb('c_Food','USA') = 702662.25 ;
vdpb('c_Food','EU_28') = 638599.5625 ;
vdpb('c_Food','ROW') = 4088654.5 ;
vdpb('c_Mnfcs','USA') = 792360.1875 ;
vdpb('c_Mnfcs','EU_28') = 433667 ;
vdpb('c_Mnfcs','ROW') = 2320420.75 ;
vdpb('c_Svces','USA') = 10661360 ;
vdpb('c_Svces','EU_28') = 6461904 ;
vdpb('c_Svces','ROW') = 13852313 ;

* vdpp data (9 cells)
vdpp('c_Food','USA') = 701175.125 ;
vdpp('c_Food','EU_28') = 823667 ;
vdpp('c_Food','ROW') = 4284743.5 ;
vdpp('c_Mnfcs','USA') = 811949.3125 ;
vdpp('c_Mnfcs','EU_28') = 637483.5625 ;
vdpp('c_Mnfcs','ROW') = 2518896.5 ;
vdpp('c_Svces','USA') = 10888380 ;
vdpp('c_Svces','EU_28') = 6805092.5 ;
vdpp('c_Svces','ROW') = 14267114 ;

* vmpb data (9 cells)
vmpb('c_Food','USA') = 92016.71875 ;
vmpb('c_Food','EU_28') = 282465.9375 ;
vmpb('c_Food','ROW') = 458404.0625 ;
vmpb('c_Mnfcs','USA') = 632188.8125 ;
vmpb('c_Mnfcs','EU_28') = 834982.125 ;
vmpb('c_Mnfcs','ROW') = 1313897 ;
vmpb('c_Svces','USA') = 179993.1562 ;
vmpb('c_Svces','EU_28') = 164734.5156 ;
vmpb('c_Svces','ROW') = 652396.625 ;

* vmpp data (9 cells)
vmpp('c_Food','USA') = 91885.0625 ;
vmpp('c_Food','EU_28') = 361205.875 ;
vmpp('c_Food','ROW') = 516846.125 ;
vmpp('c_Mnfcs','USA') = 654085.1875 ;
vmpp('c_Mnfcs','EU_28') = 1120764.375 ;
vmpp('c_Mnfcs','ROW') = 1467644.125 ;
vmpp('c_Svces','USA') = 185092.0469 ;
vmpp('c_Svces','EU_28') = 179237.2031 ;
vmpp('c_Svces','ROW') = 682840.25 ;

* vdgb data (9 cells)
vdgb('c_Food','USA') = 1023.769836 ;
vdgb('c_Food','EU_28') = 5274.974121 ;
vdgb('c_Food','ROW') = 12373.4541 ;
vdgb('c_Mnfcs','USA') = 405.8898315 ;
vdgb('c_Mnfcs','EU_28') = 35854.46094 ;
vdgb('c_Mnfcs','ROW') = 59685.85547 ;
vdgb('c_Svces','USA') = 2543765 ;
vdgb('c_Svces','EU_28') = 3504409.5 ;
vdgb('c_Svces','ROW') = 6974498.5 ;

* vdgp data (9 cells)
vdgp('c_Food','USA') = 1023.769836 ;
vdgp('c_Food','EU_28') = 5454.498535 ;
vdgp('c_Food','ROW') = 12476.13672 ;
vdgp('c_Mnfcs','USA') = 405.8898315 ;
vdgp('c_Mnfcs','EU_28') = 40958.1875 ;
vdgp('c_Mnfcs','ROW') = 60157.67969 ;
vdgp('c_Svces','USA') = 2729009.5 ;
vdgp('c_Svces','EU_28') = 3507268.75 ;
vdgp('c_Svces','ROW') = 7017338 ;

* vmgb data (9 cells)
vmgb('c_Food','USA') = 7.942389011 ;
vmgb('c_Food','EU_28') = 200.9221039 ;
vmgb('c_Food','ROW') = 1954.703125 ;
vmgb('c_Mnfcs','USA') = 331.9682617 ;
vmgb('c_Mnfcs','EU_28') = 68117.15625 ;
vmgb('c_Mnfcs','ROW') = 67079.35938 ;
vmgb('c_Svces','USA') = 19186.18555 ;
vmgb('c_Svces','EU_28') = 7234.123535 ;
vmgb('c_Svces','ROW') = 101980.1094 ;

* vmgp data (9 cells)
vmgp('c_Food','USA') = 7.942389011 ;
vmgp('c_Food','EU_28') = 223.5062866 ;
vmgp('c_Food','ROW') = 1977.492065 ;
vmgp('c_Mnfcs','USA') = 331.9682617 ;
vmgp('c_Mnfcs','EU_28') = 76792.6875 ;
vmgp('c_Mnfcs','ROW') = 66996.59375 ;
vmgp('c_Svces','USA') = 19186.18555 ;
vmgp('c_Svces','EU_28') = 9761.845703 ;
vmgp('c_Svces','ROW') = 107714.3984 ;

* vdib data (9 cells)
vdib('c_Food','USA') = 522.6500854 ;
vdib('c_Food','EU_28') = 5696.132324 ;
vdib('c_Food','ROW') = 86141.78906 ;
vdib('c_Mnfcs','USA') = 765932.4375 ;
vdib('c_Mnfcs','EU_28') = 473251.4688 ;
vdib('c_Mnfcs','ROW') = 2101775 ;
vdib('c_Svces','USA') = 2873172.5 ;
vdib('c_Svces','EU_28') = 2258155.25 ;
vdib('c_Svces','ROW') = 8890655 ;

* vdip data (9 cells)
vdip('c_Food','USA') = 522.6500854 ;
vdip('c_Food','EU_28') = 5491.546875 ;
vdip('c_Food','ROW') = 86947.17188 ;
vdip('c_Mnfcs','USA') = 775809.125 ;
vdip('c_Mnfcs','EU_28') = 490787.6875 ;
vdip('c_Mnfcs','ROW') = 2222290 ;
vdip('c_Svces','USA') = 2692840.75 ;
vdip('c_Svces','EU_28') = 2438634.25 ;
vdip('c_Svces','ROW') = 9181054 ;

* vmib data (9 cells)
vmib('c_Food','USA') = 22.40802383 ;
vmib('c_Food','EU_28') = 2048.053223 ;
vmib('c_Food','ROW') = 9515.726562 ;
vmib('c_Mnfcs','USA') = 473778.375 ;
vmib('c_Mnfcs','EU_28') = 582482.6875 ;
vmib('c_Mnfcs','ROW') = 1294673.75 ;
vmib('c_Svces','USA') = 99352.00781 ;
vmib('c_Svces','EU_28') = 92506.16406 ;
vmib('c_Svces','ROW') = 139370 ;

* vmip data (9 cells)
vmip('c_Food','USA') = 22.40802383 ;
vmip('c_Food','EU_28') = 2077.350098 ;
vmip('c_Food','ROW') = 9679.484375 ;
vmip('c_Mnfcs','USA') = 491439.5 ;
vmip('c_Mnfcs','EU_28') = 614703.125 ;
vmip('c_Mnfcs','ROW') = 1395857.75 ;
vmip('c_Svces','USA') = 87960.89844 ;
vmip('c_Svces','EU_28') = 99512.54688 ;
vmip('c_Svces','ROW') = 143277.1406 ;

* evfb data (36 cells)
evfb('Land','a_Food','USA') = 41351.43359 ;
evfb('Land','a_Food','EU_28') = 67458 ;
evfb('Land','a_Food','ROW') = 669254.25 ;
evfb('UnSkLab','a_Food','USA') = 119645.0625 ;
evfb('UnSkLab','a_Food','EU_28') = 142159.0938 ;
evfb('UnSkLab','a_Food','ROW') = 1519266.75 ;
evfb('UnSkLab','a_Mnfcs','USA') = 705015.5625 ;
evfb('UnSkLab','a_Mnfcs','EU_28') = 439850.9062 ;
evfb('UnSkLab','a_Mnfcs','ROW') = 2233035.75 ;
evfb('UnSkLab','a_Svces','USA') = 3322260.5 ;
evfb('UnSkLab','a_Svces','EU_28') = 1800742.625 ;
evfb('UnSkLab','a_Svces','ROW') = 7304310 ;
evfb('SkLab','a_Food','USA') = 61255.26172 ;
evfb('SkLab','a_Food','EU_28') = 90423.3125 ;
evfb('SkLab','a_Food','ROW') = 205270.1406 ;
evfb('SkLab','a_Mnfcs','USA') = 233255.8906 ;
evfb('SkLab','a_Mnfcs','EU_28') = 477022.5938 ;
evfb('SkLab','a_Mnfcs','ROW') = 811459.3125 ;
evfb('SkLab','a_Svces','USA') = 3798211.75 ;
evfb('SkLab','a_Svces','EU_28') = 2533243.25 ;
evfb('SkLab','a_Svces','ROW') = 5772158 ;
evfb('Capital','a_Food','USA') = 225441.9688 ;
evfb('Capital','a_Food','EU_28') = 294794.6875 ;
evfb('Capital','a_Food','ROW') = 1497269 ;
evfb('Capital','a_Mnfcs','USA') = 968529.625 ;
evfb('Capital','a_Mnfcs','EU_28') = 1078327.5 ;
evfb('Capital','a_Mnfcs','ROW') = 4621116.5 ;
evfb('Capital','a_Svces','USA') = 6049054 ;
evfb('Capital','a_Svces','EU_28') = 5707220.5 ;
evfb('Capital','a_Svces','ROW') = 13170218 ;
evfb('NatRes','a_Food','USA') = 5286.424805 ;
evfb('NatRes','a_Food','EU_28') = 8407.166016 ;
evfb('NatRes','a_Food','ROW') = 112438.2734 ;
evfb('NatRes','a_Mnfcs','USA') = 75873.76562 ;
evfb('NatRes','a_Mnfcs','EU_28') = 20106.6875 ;
evfb('NatRes','a_Mnfcs','ROW') = 572052.9375 ;

* evfp data (36 cells)
evfp('Land','a_Food','USA') = 34003.03516 ;
evfp('Land','a_Food','EU_28') = 45434.83203 ;
evfp('Land','a_Food','ROW') = 640291.1875 ;
evfp('UnSkLab','a_Food','USA') = 141928.6406 ;
evfp('UnSkLab','a_Food','EU_28') = 188846.1875 ;
evfp('UnSkLab','a_Food','ROW') = 1595059.25 ;
evfp('UnSkLab','a_Mnfcs','USA') = 848535.1875 ;
evfp('UnSkLab','a_Mnfcs','EU_28') = 644717 ;
evfp('UnSkLab','a_Mnfcs','ROW') = 2434527.25 ;
evfp('UnSkLab','a_Svces','USA') = 3998571.25 ;
evfp('UnSkLab','a_Svces','EU_28') = 2567129 ;
evfp('UnSkLab','a_Svces','ROW') = 8089657 ;
evfp('SkLab','a_Food','USA') = 71701.92969 ;
evfp('SkLab','a_Food','EU_28') = 123548.7578 ;
evfp('SkLab','a_Food','ROW') = 223862.6562 ;
evfp('SkLab','a_Mnfcs','USA') = 280739.6562 ;
evfp('SkLab','a_Mnfcs','EU_28') = 687521.6875 ;
evfp('SkLab','a_Mnfcs','ROW') = 917795.4375 ;
evfp('SkLab','a_Svces','USA') = 4571411.5 ;
evfp('SkLab','a_Svces','EU_28') = 3625571.75 ;
evfp('SkLab','a_Svces','ROW') = 6484548 ;
evfp('Capital','a_Food','USA') = 233096.5469 ;
evfp('Capital','a_Food','EU_28') = 286427.125 ;
evfp('Capital','a_Food','ROW') = 1488855.375 ;
evfp('Capital','a_Mnfcs','USA') = 1017035.188 ;
evfp('Capital','a_Mnfcs','EU_28') = 1108303.5 ;
evfp('Capital','a_Mnfcs','ROW') = 4693207 ;
evfp('Capital','a_Svces','USA') = 6352001 ;
evfp('Capital','a_Svces','EU_28') = 5897007 ;
evfp('Capital','a_Svces','ROW') = 13404936 ;
evfp('NatRes','a_Food','USA') = 5551.177734 ;
evfp('NatRes','a_Food','EU_28') = 8703.983398 ;
evfp('NatRes','a_Food','ROW') = 113756.7344 ;
evfp('NatRes','a_Mnfcs','USA') = 79673.64844 ;
evfp('NatRes','a_Mnfcs','EU_28') = 20711.26953 ;
evfp('NatRes','a_Mnfcs','ROW') = 579752 ;

* evos data (36 cells)
evos('Land','a_Food','USA') = 39669.44531 ;
evos('Land','a_Food','EU_28') = 63766.05859 ;
evos('Land','a_Food','ROW') = 613819.8125 ;
evos('UnSkLab','a_Food','USA') = 90088.76562 ;
evos('UnSkLab','a_Food','EU_28') = 99131.89844 ;
evos('UnSkLab','a_Food','ROW') = 1430582.5 ;
evos('UnSkLab','a_Mnfcs','USA') = 530853.375 ;
evos('UnSkLab','a_Mnfcs','EU_28') = 302901.2812 ;
evos('UnSkLab','a_Mnfcs','ROW') = 2075360.25 ;
evos('UnSkLab','a_Svces','USA') = 2501552 ;
evos('UnSkLab','a_Svces','EU_28') = 1275810.875 ;
evos('UnSkLab','a_Svces','ROW') = 6688394.5 ;
evos('SkLab','a_Food','USA') = 46123.1875 ;
evos('SkLab','a_Food','EU_28') = 62689.94922 ;
evos('SkLab','a_Food','ROW') = 186440.1406 ;
evos('SkLab','a_Mnfcs','USA') = 175633.9688 ;
evos('SkLab','a_Mnfcs','EU_28') = 330767.4688 ;
evos('SkLab','a_Mnfcs','ROW') = 729076.625 ;
evos('SkLab','a_Svces','USA') = 2859927.75 ;
evos('SkLab','a_Svces','EU_28') = 1781857.5 ;
evos('SkLab','a_Svces','ROW') = 5148835.5 ;
evos('Capital','a_Food','USA') = 216271.9844 ;
evos('Capital','a_Food','EU_28') = 277643.5938 ;
evos('Capital','a_Food','ROW') = 1380466.75 ;
evos('Capital','a_Mnfcs','USA') = 929134.125 ;
evos('Capital','a_Mnfcs','EU_28') = 1016128.125 ;
evos('Capital','a_Mnfcs','ROW') = 4249840.5 ;
evos('Capital','a_Svces','USA') = 5803005.5 ;
evos('Capital','a_Svces','EU_28') = 5364355.5 ;
evos('Capital','a_Svces','ROW') = 12106557 ;
evos('NatRes','a_Food','USA') = 5071.396973 ;
evos('NatRes','a_Food','EU_28') = 7891.441406 ;
evos('NatRes','a_Food','ROW') = 102811.0156 ;
evos('NatRes','a_Mnfcs','USA') = 72787.5625 ;
evos('NatRes','a_Mnfcs','EU_28') = 18835.375 ;
evos('NatRes','a_Mnfcs','ROW') = 529895.9375 ;

* vxsb data (27 cells)
vxsb('c_Food','USA','USA') = 2.200000017e-05 ;
vxsb('c_Food','USA','EU_28') = 12887.27344 ;
vxsb('c_Food','USA','ROW') = 142672.8281 ;
vxsb('c_Food','EU_28','USA') = 25649.76758 ;
vxsb('c_Food','EU_28','EU_28') = 391281.8438 ;
vxsb('c_Food','EU_28','ROW') = 128399.2891 ;
vxsb('c_Food','ROW','USA') = 125250.5703 ;
vxsb('c_Food','ROW','EU_28') = 132796.0312 ;
vxsb('c_Food','ROW','ROW') = 568873.5 ;
vxsb('c_Mnfcs','USA','USA') = 2.300000051e-05 ;
vxsb('c_Mnfcs','USA','EU_28') = 225169.7969 ;
vxsb('c_Mnfcs','USA','ROW') = 1043731.562 ;
vxsb('c_Mnfcs','EU_28','USA') = 379717.875 ;
vxsb('c_Mnfcs','EU_28','EU_28') = 2729899.75 ;
vxsb('c_Mnfcs','EU_28','ROW') = 1449099.875 ;
vxsb('c_Mnfcs','ROW','USA') = 1689399.625 ;
vxsb('c_Mnfcs','ROW','EU_28') = 1490862.25 ;
vxsb('c_Mnfcs','ROW','ROW') = 5414330.5 ;
vxsb('c_Svces','USA','USA') = 1.999999949e-05 ;
vxsb('c_Svces','USA','EU_28') = 223336.6562 ;
vxsb('c_Svces','USA','ROW') = 481554.0312 ;
vxsb('c_Svces','EU_28','USA') = 199364.0625 ;
vxsb('c_Svces','EU_28','EU_28') = 1061178.25 ;
vxsb('c_Svces','EU_28','ROW') = 568730.8125 ;
vxsb('c_Svces','ROW','USA') = 352720.0312 ;
vxsb('c_Svces','ROW','EU_28') = 473475.4688 ;
vxsb('c_Svces','ROW','ROW') = 1031974 ;

* vfob data (27 cells)
vfob('c_Food','USA','USA') = 2.200000017e-05 ;
vfob('c_Food','USA','EU_28') = 12915.67871 ;
vfob('c_Food','USA','ROW') = 142971.0625 ;
vfob('c_Food','EU_28','USA') = 25651.37305 ;
vfob('c_Food','EU_28','EU_28') = 391281.8438 ;
vfob('c_Food','EU_28','ROW') = 128398.2578 ;
vfob('c_Food','ROW','USA') = 125241.3828 ;
vfob('c_Food','ROW','EU_28') = 132830.9844 ;
vfob('c_Food','ROW','ROW') = 569675.6875 ;
vfob('c_Mnfcs','USA','USA') = 2.300000051e-05 ;
vfob('c_Mnfcs','USA','EU_28') = 237000.9688 ;
vfob('c_Mnfcs','USA','ROW') = 1070258.875 ;
vfob('c_Mnfcs','EU_28','USA') = 380644.0312 ;
vfob('c_Mnfcs','EU_28','EU_28') = 2729899.75 ;
vfob('c_Mnfcs','EU_28','ROW') = 1452146.25 ;
vfob('c_Mnfcs','ROW','USA') = 1702821.5 ;
vfob('c_Mnfcs','ROW','EU_28') = 1506589.625 ;
vfob('c_Mnfcs','ROW','ROW') = 5470442 ;
vfob('c_Svces','USA','USA') = 1.999999949e-05 ;
vfob('c_Svces','USA','EU_28') = 223336.6562 ;
vfob('c_Svces','USA','ROW') = 481554.0312 ;
vfob('c_Svces','EU_28','USA') = 199364.0625 ;
vfob('c_Svces','EU_28','EU_28') = 1061178.25 ;
vfob('c_Svces','EU_28','ROW') = 568730.8125 ;
vfob('c_Svces','ROW','USA') = 352720.0312 ;
vfob('c_Svces','ROW','EU_28') = 473475.4688 ;
vfob('c_Svces','ROW','ROW') = 1031974 ;

* vcif data (27 cells)
vcif('c_Food','USA','USA') = 2.200000017e-05 ;
vcif('c_Food','USA','EU_28') = 13609.52441 ;
vcif('c_Food','USA','ROW') = 150883.2969 ;
vcif('c_Food','EU_28','USA') = 27180.79492 ;
vcif('c_Food','EU_28','EU_28') = 411168.6562 ;
vcif('c_Food','EU_28','ROW') = 137097.5781 ;
vcif('c_Food','ROW','USA') = 131440.0625 ;
vcif('c_Food','ROW','EU_28') = 141696.1875 ;
vcif('c_Food','ROW','ROW') = 611737.4375 ;
vcif('c_Mnfcs','USA','USA') = 2.300000051e-05 ;
vcif('c_Mnfcs','USA','EU_28') = 243716.9688 ;
vcif('c_Mnfcs','USA','ROW') = 1096011.625 ;
vcif('c_Mnfcs','EU_28','USA') = 390546.9375 ;
vcif('c_Mnfcs','EU_28','EU_28') = 2790937 ;
vcif('c_Mnfcs','EU_28','ROW') = 1496167.25 ;
vcif('c_Mnfcs','ROW','USA') = 1749464.625 ;
vcif('c_Mnfcs','ROW','EU_28') = 1560898.875 ;
vcif('c_Mnfcs','ROW','ROW') = 5692401.5 ;
vcif('c_Svces','USA','USA') = 1.999999949e-05 ;
vcif('c_Svces','USA','EU_28') = 223336.6562 ;
vcif('c_Svces','USA','ROW') = 481554.0312 ;
vcif('c_Svces','EU_28','USA') = 199364.0625 ;
vcif('c_Svces','EU_28','EU_28') = 1061178.25 ;
vcif('c_Svces','EU_28','ROW') = 568730.8125 ;
vcif('c_Svces','ROW','USA') = 352720.0312 ;
vcif('c_Svces','ROW','EU_28') = 473475.4688 ;
vcif('c_Svces','ROW','ROW') = 1031974 ;

* vmsb data (27 cells)
vmsb('c_Food','USA','USA') = 2.200000017e-05 ;
vmsb('c_Food','USA','EU_28') = 14271.8291 ;
vmsb('c_Food','USA','ROW') = 161892.7969 ;
vmsb('c_Food','EU_28','USA') = 28559.93555 ;
vmsb('c_Food','EU_28','EU_28') = 411168.6562 ;
vmsb('c_Food','EU_28','ROW') = 155324.4688 ;
vmsb('c_Food','ROW','USA') = 133310.5469 ;
vmsb('c_Food','ROW','EU_28') = 147657.375 ;
vmsb('c_Food','ROW','ROW') = 661867.125 ;
vmsb('c_Mnfcs','USA','USA') = 2.300000051e-05 ;
vmsb('c_Mnfcs','USA','EU_28') = 248313.7812 ;
vmsb('c_Mnfcs','USA','ROW') = 1118628.375 ;
vmsb('c_Mnfcs','EU_28','USA') = 396228.4688 ;
vmsb('c_Mnfcs','EU_28','EU_28') = 2790937 ;
vmsb('c_Mnfcs','EU_28','ROW') = 1552737.75 ;
vmsb('c_Mnfcs','ROW','USA') = 1775874.125 ;
vmsb('c_Mnfcs','ROW','EU_28') = 1585112.125 ;
vmsb('c_Mnfcs','ROW','ROW') = 5857372 ;
vmsb('c_Svces','USA','USA') = 1.999999949e-05 ;
vmsb('c_Svces','USA','EU_28') = 223336.6562 ;
vmsb('c_Svces','USA','ROW') = 481555.0938 ;
vmsb('c_Svces','EU_28','USA') = 199364.0625 ;
vmsb('c_Svces','EU_28','EU_28') = 1061178.25 ;
vmsb('c_Svces','EU_28','ROW') = 568730.875 ;
vmsb('c_Svces','ROW','USA') = 352720.0312 ;
vmsb('c_Svces','ROW','EU_28') = 473475.4688 ;
vmsb('c_Svces','ROW','ROW') = 1032009.438 ;

* vst data (3 cells)
vst('c_Svces','USA') = 31137.61328 ;
vst('c_Svces','EU_28') = 275190.5 ;
vst('c_Svces','ROW') = 259861.1562 ;

* vtwr data (16 cells)
vtwr('c_Svces','c_Food','USA','EU_28') = 693.8433838 ;
vtwr('c_Svces','c_Food','USA','ROW') = 7912.241699 ;
vtwr('c_Svces','c_Food','EU_28','USA') = 1529.419678 ;
vtwr('c_Svces','c_Food','EU_28','EU_28') = 19886.89453 ;
vtwr('c_Svces','c_Food','EU_28','ROW') = 8699.322266 ;
vtwr('c_Svces','c_Food','ROW','USA') = 6198.676758 ;
vtwr('c_Svces','c_Food','ROW','EU_28') = 8865.225586 ;
vtwr('c_Svces','c_Food','ROW','ROW') = 42061.73047 ;
vtwr('c_Svces','c_Mnfcs','USA','EU_28') = 6716.067383 ;
vtwr('c_Svces','c_Mnfcs','USA','ROW') = 25752.75586 ;
vtwr('c_Svces','c_Mnfcs','EU_28','USA') = 9902.892578 ;
vtwr('c_Svces','c_Mnfcs','EU_28','EU_28') = 61037.85938 ;
vtwr('c_Svces','c_Mnfcs','EU_28','ROW') = 44020.87891 ;
vtwr('c_Svces','c_Mnfcs','ROW','USA') = 46643.11328 ;
vtwr('c_Svces','c_Mnfcs','ROW','EU_28') = 54309.71484 ;
vtwr('c_Svces','c_Mnfcs','ROW','ROW') = 221959.2969 ;

* save data (3 cells)
save('USA') = 1580785.625 ;
save('EU_28') = 2133911.25 ;
save('ROW') = 8476150 ;

* vdep data (3 cells)
vdep('USA') = 1816268 ;
vdep('EU_28') = 1809763.25 ;
vdep('ROW') = 4922029.5 ;

* vkb data (3 cells)
vkb('USA') = 45406700 ;
vkb('EU_28') = 45244080 ;
vkb('ROW') = 123050744 ;

* pop0 data (3 cells)
pop0('USA') = 325.1471252 ;
pop0('EU_28') = 513.8722534 ;
pop0('ROW') = 6674.864258 ;

* maks data (9 cells)
maks('c_Food','a_Food','USA') = 1579235.5 ;
maks('c_Food','a_Food','EU_28') = 1987133.75 ;
maks('c_Food','a_Food','ROW') = 9393134 ;
maks('c_Mnfcs','a_Mnfcs','USA') = 5737490.5 ;
maks('c_Mnfcs','a_Mnfcs','EU_28') = 8537181 ;
maks('c_Mnfcs','a_Mnfcs','ROW') = 30127092 ;
maks('c_Svces','a_Svces','USA') = 24752234 ;
maks('c_Svces','a_Svces','EU_28') = 23014476 ;
maks('c_Svces','a_Svces','ROW') = 51690456 ;

* makb data (9 cells)
makb('c_Food','a_Food','USA') = 1620585.375 ;
makb('c_Food','a_Food','EU_28') = 1987113.625 ;
makb('c_Food','a_Food','ROW') = 9431686 ;
makb('c_Mnfcs','a_Mnfcs','USA') = 5860736 ;
makb('c_Mnfcs','a_Mnfcs','EU_28') = 8561450 ;
makb('c_Mnfcs','a_Mnfcs','ROW') = 30236078 ;
makb('c_Svces','a_Svces','USA') = 25808800 ;
makb('c_Svces','a_Svces','EU_28') = 23215892 ;
makb('c_Svces','a_Svces','ROW') = 52126932 ;

* esubt (empty in GDX — relying on $onImplicitAssign suppression)

* esubc (empty in GDX — relying on $onImplicitAssign suppression)


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

* esubva data (9 cells)
esubva('a_Food','USA') = 0.5901468396 ;
esubva('a_Food','EU_28') = 0.5901468396 ;
esubva('a_Food','ROW') = 0.5901468396 ;
esubva('a_Mnfcs','USA') = 1.076765656 ;
esubva('a_Mnfcs','EU_28') = 1.076765656 ;
esubva('a_Mnfcs','ROW') = 1.076765656 ;
esubva('a_Svces','USA') = 1.369734406 ;
esubva('a_Svces','EU_28') = 1.369734406 ;
esubva('a_Svces','ROW') = 1.369734406 ;

* etraq data (9 cells)
etraq('a_Food','USA') = -5 ;
etraq('a_Food','EU_28') = -5 ;
etraq('a_Food','ROW') = -5 ;
etraq('a_Mnfcs','USA') = -5 ;
etraq('a_Mnfcs','EU_28') = -5 ;
etraq('a_Mnfcs','ROW') = -5 ;
etraq('a_Svces','USA') = -5 ;
etraq('a_Svces','EU_28') = -5 ;
etraq('a_Svces','ROW') = -5 ;

* esubq (empty in GDX — relying on $onImplicitAssign suppression)

* incpar data (9 cells)
incpar('c_Food','USA') = 0.3807289898 ;
incpar('c_Food','EU_28') = 0.42059654 ;
incpar('c_Food','ROW') = 0.4375711083 ;
incpar('c_Mnfcs','USA') = 0.8847161531 ;
incpar('c_Mnfcs','EU_28') = 0.906521678 ;
incpar('c_Mnfcs','ROW') = 0.8920969367 ;
incpar('c_Svces','USA') = 1.059613585 ;
incpar('c_Svces','EU_28') = 1.12182653 ;
incpar('c_Svces','ROW') = 1.209413648 ;

* subpar data (9 cells)
subpar('c_Food','USA') = 0.2593968511 ;
subpar('c_Food','EU_28') = 0.4622822106 ;
subpar('c_Food','ROW') = 0.7167526484 ;
subpar('c_Mnfcs','USA') = 0.1001683772 ;
subpar('c_Mnfcs','EU_28') = 0.2513343692 ;
subpar('c_Mnfcs','ROW') = 0.4843812287 ;
subpar('c_Svces','USA') = 0.0879457891 ;
subpar('c_Svces','EU_28') = 0.2068711072 ;
subpar('c_Svces','ROW') = 0.3796758056 ;

* esubg data (3 cells)
esubg('USA') = 1 ;
esubg('EU_28') = 1 ;
esubg('ROW') = 1 ;

* esubi (empty in GDX — relying on $onImplicitAssign suppression)

* esubd data (9 cells)
esubd('c_Food','USA') = 2.429511309 ;
esubd('c_Food','EU_28') = 2.429511309 ;
esubd('c_Food','ROW') = 2.429511309 ;
esubd('c_Mnfcs','USA') = 3.625655174 ;
esubd('c_Mnfcs','EU_28') = 3.625655174 ;
esubd('c_Mnfcs','ROW') = 3.625655174 ;
esubd('c_Svces','USA') = 1.937635064 ;
esubd('c_Svces','EU_28') = 1.937635064 ;
esubd('c_Svces','ROW') = 1.937635064 ;

* esubm data (9 cells)
esubm('c_Food','USA') = 4.764378548 ;
esubm('c_Food','EU_28') = 4.764378548 ;
esubm('c_Food','ROW') = 4.764378548 ;
esubm('c_Mnfcs','USA') = 7.873477459 ;
esubm('c_Mnfcs','EU_28') = 7.873477459 ;
esubm('c_Mnfcs','ROW') = 7.873477459 ;
esubm('c_Svces','USA') = 3.826968431 ;
esubm('c_Svces','EU_28') = 3.826968431 ;
esubm('c_Svces','ROW') = 3.826968431 ;

* esubs data (1 cells)
esubs('c_Svces') = 1 ;

* etrae data (6 cells)
etrae('Land','USA') = -1 ;
etrae('Land','EU_28') = -1 ;
etrae('Land','ROW') = -1 ;
etrae('NatRes','USA') = -0.001000000047 ;
etrae('NatRes','EU_28') = -0.001000000047 ;
etrae('NatRes','ROW') = -0.001000000047 ;

* rorFlex0 data (3 cells)
rorFlex0('USA') = 10 ;
rorFlex0('EU_28') = 10 ;
rorFlex0('ROW') = 10 ;
* FBEP data (injected from basedata.har)
FBEP('Land','a_Food','USA') = -9419.345703 ;
FBEP('Land','a_Food','EU_28') = -24050.85156 ;
FBEP('Land','a_Food','ROW') = -36261.22656 ;
FBEP('UnSkLab','a_Food','USA') = -2072.498291 ;
FBEP('UnSkLab','a_Food','EU_28') = -23071.52148 ;
FBEP('UnSkLab','a_Food','ROW') = -12537.58203 ;
FBEP('SkLab','a_Food','USA') = -2023.032715 ;
FBEP('SkLab','a_Food','EU_28') = -7168.943359 ;
FBEP('SkLab','a_Food','ROW') = -1042.674805 ;
FBEP('Capital','a_Food','USA') = -3635.937988 ;
FBEP('Capital','a_Food','EU_28') = -18409.45508 ;
FBEP('Capital','a_Food','ROW') = -27742.77734 ;
* FTRV data (injected from basedata.har)
FTRV('Land','a_Food','USA') = 2070.950684 ;
FTRV('Land','a_Food','EU_28') = 2027.685181 ;
FTRV('Land','a_Food','ROW') = 7298.186035 ;
FTRV('UnSkLab','a_Food','USA') = 24356.07227 ;
FTRV('UnSkLab','a_Food','EU_28') = 69758.61719 ;
FTRV('UnSkLab','a_Food','ROW') = 88330.11719 ;
FTRV('UnSkLab','a_Mnfcs','USA') = 143519.5938 ;
FTRV('UnSkLab','a_Mnfcs','EU_28') = 204866.1094 ;
FTRV('UnSkLab','a_Mnfcs','ROW') = 201491.4844 ;
FTRV('UnSkLab','a_Svces','USA') = 676310.5625 ;
FTRV('UnSkLab','a_Svces','EU_28') = 766386.5 ;
FTRV('UnSkLab','a_Svces','ROW') = 785347.0625 ;
FTRV('SkLab','a_Food','USA') = 12469.69629 ;
FTRV('SkLab','a_Food','EU_28') = 40294.38672 ;
FTRV('SkLab','a_Food','ROW') = 19635.19336 ;
FTRV('SkLab','a_Mnfcs','USA') = 47483.75781 ;
FTRV('SkLab','a_Mnfcs','EU_28') = 210499.0781 ;
FTRV('SkLab','a_Mnfcs','ROW') = 106336.125 ;
FTRV('SkLab','a_Svces','USA') = 773199.6875 ;
FTRV('SkLab','a_Svces','EU_28') = 1092328.375 ;
FTRV('SkLab','a_Svces','ROW') = 712390.3125 ;
FTRV('Capital','a_Food','USA') = 11290.52051 ;
FTRV('Capital','a_Food','EU_28') = 10041.88184 ;
FTRV('Capital','a_Food','ROW') = 19329.10938 ;
FTRV('Capital','a_Mnfcs','USA') = 48505.62109 ;
FTRV('Capital','a_Mnfcs','EU_28') = 29975.99414 ;
FTRV('Capital','a_Mnfcs','ROW') = 72090.55469 ;
FTRV('Capital','a_Svces','USA') = 302947 ;
FTRV('Capital','a_Svces','EU_28') = 189786.7812 ;
FTRV('Capital','a_Svces','ROW') = 234718.4219 ;
FTRV('NatRes','a_Food','USA') = 264.7532043 ;
FTRV('NatRes','a_Food','EU_28') = 296.8175659 ;
FTRV('NatRes','a_Food','ROW') = 1318.46167 ;
FTRV('NatRes','a_Mnfcs','USA') = 3799.888184 ;
FTRV('NatRes','a_Mnfcs','EU_28') = 604.5817871 ;
FTRV('NatRes','a_Mnfcs','ROW') = 7699.074219 ;

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
   USA, EU_28, ROW
/ ;

set imuv(i) "IMUV commodities" /
   c_Food, c_Mnfcs, c_Svces
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
