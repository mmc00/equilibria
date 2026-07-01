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
   a_Agri,
   a_Food,
   a_Energy,
   a_Manuf,
   a_Svces
/;

Set comm 'Set COMM  Commodities' /
   c_Agri,
   c_Food,
   c_Energy,
   c_Manuf,
   c_Svces
/;

Set reg 'Set REG  Regions' /
   USA,
   EU_28,
   CHN,
   LatinAmer,
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
* pop0 data (5 cells)
pop0('USA') = 325.1471252 ;
pop0('EU_28') = 513.8722534 ;
pop0('CHN') = 1386.39502 ;
pop0('LatinAmer') = 635.7049561 ;
pop0('ROW') = 4652.764648 ;


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

* vdfb data (125 cells)
vdfb('c_Agri','a_Agri','USA') = 64414.13672 ;
vdfb('c_Agri','a_Agri','EU_28') = 46679.66016 ;
vdfb('c_Agri','a_Agri','CHN') = 145540 ;
vdfb('c_Agri','a_Agri','LatinAmer') = 56397.67578 ;
vdfb('c_Agri','a_Agri','ROW') = 319295.75 ;
vdfb('c_Agri','a_Food','USA') = 210682.4219 ;
vdfb('c_Agri','a_Food','EU_28') = 179794.1562 ;
vdfb('c_Agri','a_Food','CHN') = 520748.6875 ;
vdfb('c_Agri','a_Food','LatinAmer') = 214039.4688 ;
vdfb('c_Agri','a_Food','ROW') = 639240.6875 ;
vdfb('c_Agri','a_Energy','USA') = 112.1591492 ;
vdfb('c_Agri','a_Energy','EU_28') = 648.3098145 ;
vdfb('c_Agri','a_Energy','CHN') = 338.4370117 ;
vdfb('c_Agri','a_Energy','LatinAmer') = 1008.599487 ;
vdfb('c_Agri','a_Energy','ROW') = 4964.636719 ;
vdfb('c_Agri','a_Manuf','USA') = 31437.75586 ;
vdfb('c_Agri','a_Manuf','EU_28') = 27788.65234 ;
vdfb('c_Agri','a_Manuf','CHN') = 210230.6875 ;
vdfb('c_Agri','a_Manuf','LatinAmer') = 23706.66211 ;
vdfb('c_Agri','a_Manuf','ROW') = 124788.1328 ;
vdfb('c_Agri','a_Svces','USA') = 15139.64551 ;
vdfb('c_Agri','a_Svces','EU_28') = 29826.33984 ;
vdfb('c_Agri','a_Svces','CHN') = 123518.0078 ;
vdfb('c_Agri','a_Svces','LatinAmer') = 17050.29102 ;
vdfb('c_Agri','a_Svces','ROW') = 204120.9844 ;
vdfb('c_Food','a_Agri','USA') = 35451.42188 ;
vdfb('c_Food','a_Agri','EU_28') = 41441.86719 ;
vdfb('c_Food','a_Agri','CHN') = 110312.2734 ;
vdfb('c_Food','a_Agri','LatinAmer') = 29905.2832 ;
vdfb('c_Food','a_Agri','ROW') = 87544.49219 ;
vdfb('c_Food','a_Food','USA') = 231806.9844 ;
vdfb('c_Food','a_Food','EU_28') = 232722.3906 ;
vdfb('c_Food','a_Food','CHN') = 385166.375 ;
vdfb('c_Food','a_Food','LatinAmer') = 98557.96875 ;
vdfb('c_Food','a_Food','ROW') = 295022.125 ;
vdfb('c_Food','a_Energy','USA') = 127.9600601 ;
vdfb('c_Food','a_Energy','EU_28') = 562.100647 ;
vdfb('c_Food','a_Energy','CHN') = 5781.861328 ;
vdfb('c_Food','a_Energy','LatinAmer') = 269.5948181 ;
vdfb('c_Food','a_Energy','ROW') = 1664.011963 ;
vdfb('c_Food','a_Manuf','USA') = 9685.801758 ;
vdfb('c_Food','a_Manuf','EU_28') = 20280.19531 ;
vdfb('c_Food','a_Manuf','CHN') = 109279.1953 ;
vdfb('c_Food','a_Manuf','LatinAmer') = 8174.798828 ;
vdfb('c_Food','a_Manuf','ROW') = 24110.14258 ;
vdfb('c_Food','a_Svces','USA') = 161958.3125 ;
vdfb('c_Food','a_Svces','EU_28') = 212468.4062 ;
vdfb('c_Food','a_Svces','CHN') = 254889.75 ;
vdfb('c_Food','a_Svces','LatinAmer') = 83192.34375 ;
vdfb('c_Food','a_Svces','ROW') = 318737.5625 ;
vdfb('c_Energy','a_Agri','USA') = 9416.429688 ;
vdfb('c_Energy','a_Agri','EU_28') = 12788.13086 ;
vdfb('c_Energy','a_Agri','CHN') = 18587.23047 ;
vdfb('c_Energy','a_Agri','LatinAmer') = 12756.36035 ;
vdfb('c_Energy','a_Agri','ROW') = 83432.78125 ;
vdfb('c_Energy','a_Food','USA') = 12463.21484 ;
vdfb('c_Energy','a_Food','EU_28') = 18172.4082 ;
vdfb('c_Energy','a_Food','CHN') = 16410.56836 ;
vdfb('c_Energy','a_Food','LatinAmer') = 9817.558594 ;
vdfb('c_Energy','a_Food','ROW') = 38545.05469 ;
vdfb('c_Energy','a_Energy','USA') = 304586.375 ;
vdfb('c_Energy','a_Energy','EU_28') = 133683.5 ;
vdfb('c_Energy','a_Energy','CHN') = 381460.25 ;
vdfb('c_Energy','a_Energy','LatinAmer') = 129872.4219 ;
vdfb('c_Energy','a_Energy','ROW') = 852198.875 ;
vdfb('c_Energy','a_Manuf','USA') = 122936.3359 ;
vdfb('c_Energy','a_Manuf','EU_28') = 153472.2344 ;
vdfb('c_Energy','a_Manuf','CHN') = 453216.9375 ;
vdfb('c_Energy','a_Manuf','LatinAmer') = 85415.47656 ;
vdfb('c_Energy','a_Manuf','ROW') = 670271.375 ;
vdfb('c_Energy','a_Svces','USA') = 330760.9062 ;
vdfb('c_Energy','a_Svces','EU_28') = 232326.625 ;
vdfb('c_Energy','a_Svces','CHN') = 211807.6094 ;
vdfb('c_Energy','a_Svces','LatinAmer') = 98911.39844 ;
vdfb('c_Energy','a_Svces','ROW') = 656397.75 ;
vdfb('c_Manuf','a_Agri','USA') = 31645.64844 ;
vdfb('c_Manuf','a_Agri','EU_28') = 25614.10938 ;
vdfb('c_Manuf','a_Agri','CHN') = 108807.8438 ;
vdfb('c_Manuf','a_Agri','LatinAmer') = 42702.26953 ;
vdfb('c_Manuf','a_Agri','ROW') = 69920.64062 ;
vdfb('c_Manuf','a_Food','USA') = 100344.6953 ;
vdfb('c_Manuf','a_Food','EU_28') = 73554.41406 ;
vdfb('c_Manuf','a_Food','CHN') = 81708.9375 ;
vdfb('c_Manuf','a_Food','LatinAmer') = 44638.4375 ;
vdfb('c_Manuf','a_Food','ROW') = 89663.60938 ;
vdfb('c_Manuf','a_Energy','USA') = 38907.51172 ;
vdfb('c_Manuf','a_Energy','EU_28') = 37477.46875 ;
vdfb('c_Manuf','a_Energy','CHN') = 108589.9297 ;
vdfb('c_Manuf','a_Energy','LatinAmer') = 21948.73828 ;
vdfb('c_Manuf','a_Energy','ROW') = 183606.2344 ;
vdfb('c_Manuf','a_Manuf','USA') = 1333651.75 ;
vdfb('c_Manuf','a_Manuf','EU_28') = 1720559.5 ;
vdfb('c_Manuf','a_Manuf','CHN') = 5672547.5 ;
vdfb('c_Manuf','a_Manuf','LatinAmer') = 551436.3125 ;
vdfb('c_Manuf','a_Manuf','ROW') = 3529736.5 ;
vdfb('c_Manuf','a_Svces','USA') = 1064690 ;
vdfb('c_Manuf','a_Svces','EU_28') = 948463.375 ;
vdfb('c_Manuf','a_Svces','CHN') = 2411348.75 ;
vdfb('c_Manuf','a_Svces','LatinAmer') = 321995.4688 ;
vdfb('c_Manuf','a_Svces','ROW') = 1882696.625 ;
vdfb('c_Svces','a_Agri','USA') = 101483.0781 ;
vdfb('c_Svces','a_Agri','EU_28') = 72343.59375 ;
vdfb('c_Svces','a_Agri','CHN') = 124291.2266 ;
vdfb('c_Svces','a_Agri','LatinAmer') = 55893.90625 ;
vdfb('c_Svces','a_Agri','ROW') = 260708.8125 ;
vdfb('c_Svces','a_Food','USA') = 213726.5781 ;
vdfb('c_Svces','a_Food','EU_28') = 307866.125 ;
vdfb('c_Svces','a_Food','CHN') = 242868.8125 ;
vdfb('c_Svces','a_Food','LatinAmer') = 125553.6094 ;
vdfb('c_Svces','a_Food','ROW') = 422758.25 ;
vdfb('c_Svces','a_Energy','USA') = 171896.0938 ;
vdfb('c_Svces','a_Energy','EU_28') = 109273.0703 ;
vdfb('c_Svces','a_Energy','CHN') = 159079.5156 ;
vdfb('c_Svces','a_Energy','LatinAmer') = 64796.23047 ;
vdfb('c_Svces','a_Energy','ROW') = 542069.625 ;
vdfb('c_Svces','a_Manuf','USA') = 896527.375 ;
vdfb('c_Svces','a_Manuf','EU_28') = 1448538.625 ;
vdfb('c_Svces','a_Manuf','CHN') = 1591671.25 ;
vdfb('c_Svces','a_Manuf','LatinAmer') = 340705.875 ;
vdfb('c_Svces','a_Manuf','ROW') = 1905901.375 ;
vdfb('c_Svces','a_Svces','USA') = 7294576 ;
vdfb('c_Svces','a_Svces','EU_28') = 6652786 ;
vdfb('c_Svces','a_Svces','CHN') = 3999088.75 ;
vdfb('c_Svces','a_Svces','LatinAmer') = 1169808 ;
vdfb('c_Svces','a_Svces','ROW') = 7605396 ;

* vdfp data (125 cells)
vdfp('c_Agri','a_Agri','USA') = 62658.00391 ;
vdfp('c_Agri','a_Agri','EU_28') = 44961.21875 ;
vdfp('c_Agri','a_Agri','CHN') = 143057.8594 ;
vdfp('c_Agri','a_Agri','LatinAmer') = 56157.1875 ;
vdfp('c_Agri','a_Agri','ROW') = 306617.8125 ;
vdfp('c_Agri','a_Food','USA') = 211363.25 ;
vdfp('c_Agri','a_Food','EU_28') = 176745.5 ;
vdfp('c_Agri','a_Food','CHN') = 506177.2812 ;
vdfp('c_Agri','a_Food','LatinAmer') = 219292.9375 ;
vdfp('c_Agri','a_Food','ROW') = 639683.6875 ;
vdfp('c_Agri','a_Energy','USA') = 110.8508759 ;
vdfp('c_Agri','a_Energy','EU_28') = 648.3708496 ;
vdfp('c_Agri','a_Energy','CHN') = 334.1830444 ;
vdfp('c_Agri','a_Energy','LatinAmer') = 1011.17749 ;
vdfp('c_Agri','a_Energy','ROW') = 5024.141113 ;
vdfp('c_Agri','a_Manuf','USA') = 31185.56641 ;
vdfp('c_Agri','a_Manuf','EU_28') = 27608.08789 ;
vdfp('c_Agri','a_Manuf','CHN') = 204239.5938 ;
vdfp('c_Agri','a_Manuf','LatinAmer') = 24274.39844 ;
vdfp('c_Agri','a_Manuf','ROW') = 124052.3047 ;
vdfp('c_Agri','a_Svces','USA') = 15078.53125 ;
vdfp('c_Agri','a_Svces','EU_28') = 29967.88281 ;
vdfp('c_Agri','a_Svces','CHN') = 120373.0391 ;
vdfp('c_Agri','a_Svces','LatinAmer') = 17282.9375 ;
vdfp('c_Agri','a_Svces','ROW') = 204965.5 ;
vdfp('c_Food','a_Agri','USA') = 34583.89062 ;
vdfp('c_Food','a_Agri','EU_28') = 39805.29297 ;
vdfp('c_Food','a_Agri','CHN') = 109373.0312 ;
vdfp('c_Food','a_Agri','LatinAmer') = 29657.50195 ;
vdfp('c_Food','a_Agri','ROW') = 87180.125 ;
vdfp('c_Food','a_Food','USA') = 233564.3281 ;
vdfp('c_Food','a_Food','EU_28') = 235506.5469 ;
vdfp('c_Food','a_Food','CHN') = 397432.25 ;
vdfp('c_Food','a_Food','LatinAmer') = 102167.5547 ;
vdfp('c_Food','a_Food','ROW') = 299739.5312 ;
vdfp('c_Food','a_Energy','USA') = 128.377594 ;
vdfp('c_Food','a_Energy','EU_28') = 571.6115723 ;
vdfp('c_Food','a_Energy','CHN') = 6835.95459 ;
vdfp('c_Food','a_Energy','LatinAmer') = 272.769989 ;
vdfp('c_Food','a_Energy','ROW') = 1685.56665 ;
vdfp('c_Food','a_Manuf','USA') = 9763.541016 ;
vdfp('c_Food','a_Manuf','EU_28') = 20863.29297 ;
vdfp('c_Food','a_Manuf','CHN') = 120232.3516 ;
vdfp('c_Food','a_Manuf','LatinAmer') = 8576.262695 ;
vdfp('c_Food','a_Manuf','ROW') = 24474.07031 ;
vdfp('c_Food','a_Svces','USA') = 162537.0781 ;
vdfp('c_Food','a_Svces','EU_28') = 229256.7031 ;
vdfp('c_Food','a_Svces','CHN') = 274910.5 ;
vdfp('c_Food','a_Svces','LatinAmer') = 90974.60156 ;
vdfp('c_Food','a_Svces','ROW') = 325454.9062 ;
vdfp('c_Energy','a_Agri','USA') = 11167.01758 ;
vdfp('c_Energy','a_Agri','EU_28') = 19694.88672 ;
vdfp('c_Energy','a_Agri','CHN') = 19718.7168 ;
vdfp('c_Energy','a_Agri','LatinAmer') = 15367.60254 ;
vdfp('c_Energy','a_Agri','ROW') = 79605.67188 ;
vdfp('c_Energy','a_Food','USA') = 12908.32812 ;
vdfp('c_Energy','a_Food','EU_28') = 23563.56445 ;
vdfp('c_Energy','a_Food','CHN') = 16719.51172 ;
vdfp('c_Energy','a_Food','LatinAmer') = 11164.07812 ;
vdfp('c_Energy','a_Food','ROW') = 37133.875 ;
vdfp('c_Energy','a_Energy','USA') = 316150.8125 ;
vdfp('c_Energy','a_Energy','EU_28') = 138577.8906 ;
vdfp('c_Energy','a_Energy','CHN') = 398523.9375 ;
vdfp('c_Energy','a_Energy','LatinAmer') = 128992.3047 ;
vdfp('c_Energy','a_Energy','ROW') = 817728.0625 ;
vdfp('c_Energy','a_Manuf','USA') = 138106.5156 ;
vdfp('c_Energy','a_Manuf','EU_28') = 222363.4531 ;
vdfp('c_Energy','a_Manuf','CHN') = 471239.625 ;
vdfp('c_Energy','a_Manuf','LatinAmer') = 90297.375 ;
vdfp('c_Energy','a_Manuf','ROW') = 650974.125 ;
vdfp('c_Energy','a_Svces','USA') = 387289.5938 ;
vdfp('c_Energy','a_Svces','EU_28') = 370971.4375 ;
vdfp('c_Energy','a_Svces','CHN') = 225198.6875 ;
vdfp('c_Energy','a_Svces','LatinAmer') = 108994.4688 ;
vdfp('c_Energy','a_Svces','ROW') = 650581 ;
vdfp('c_Manuf','a_Agri','USA') = 30775.24609 ;
vdfp('c_Manuf','a_Agri','EU_28') = 24301.41992 ;
vdfp('c_Manuf','a_Agri','CHN') = 107458.6094 ;
vdfp('c_Manuf','a_Agri','LatinAmer') = 42437.44141 ;
vdfp('c_Manuf','a_Agri','ROW') = 68936.96875 ;
vdfp('c_Manuf','a_Food','USA') = 101423.9375 ;
vdfp('c_Manuf','a_Food','EU_28') = 73790.46094 ;
vdfp('c_Manuf','a_Food','CHN') = 84023.48438 ;
vdfp('c_Manuf','a_Food','LatinAmer') = 46500.34766 ;
vdfp('c_Manuf','a_Food','ROW') = 90777.09375 ;
vdfp('c_Manuf','a_Energy','USA') = 38030.54297 ;
vdfp('c_Manuf','a_Energy','EU_28') = 37507.5625 ;
vdfp('c_Manuf','a_Energy','CHN') = 111587.0625 ;
vdfp('c_Manuf','a_Energy','LatinAmer') = 23200.85547 ;
vdfp('c_Manuf','a_Energy','ROW') = 188718.0938 ;
vdfp('c_Manuf','a_Manuf','USA') = 1378763.625 ;
vdfp('c_Manuf','a_Manuf','EU_28') = 1722721 ;
vdfp('c_Manuf','a_Manuf','CHN') = 5891574.5 ;
vdfp('c_Manuf','a_Manuf','LatinAmer') = 572022.3125 ;
vdfp('c_Manuf','a_Manuf','ROW') = 3563150.5 ;
vdfp('c_Manuf','a_Svces','USA') = 1065860.875 ;
vdfp('c_Manuf','a_Svces','EU_28') = 979172.875 ;
vdfp('c_Manuf','a_Svces','CHN') = 2494639 ;
vdfp('c_Manuf','a_Svces','LatinAmer') = 339410.5625 ;
vdfp('c_Manuf','a_Svces','ROW') = 1928628.75 ;
vdfp('c_Svces','a_Agri','USA') = 98816.14844 ;
vdfp('c_Svces','a_Agri','EU_28') = 69426.82031 ;
vdfp('c_Svces','a_Agri','CHN') = 123680.6484 ;
vdfp('c_Svces','a_Agri','LatinAmer') = 55693.24609 ;
vdfp('c_Svces','a_Agri','ROW') = 243768.6875 ;
vdfp('c_Svces','a_Food','USA') = 215426.5625 ;
vdfp('c_Svces','a_Food','EU_28') = 308471.3125 ;
vdfp('c_Svces','a_Food','CHN') = 253003.5781 ;
vdfp('c_Svces','a_Food','LatinAmer') = 128941.9844 ;
vdfp('c_Svces','a_Food','ROW') = 424776.5938 ;
vdfp('c_Svces','a_Energy','USA') = 172307.6875 ;
vdfp('c_Svces','a_Energy','EU_28') = 109715.9375 ;
vdfp('c_Svces','a_Energy','CHN') = 164887.4375 ;
vdfp('c_Svces','a_Energy','LatinAmer') = 67638.95312 ;
vdfp('c_Svces','a_Energy','ROW') = 545003.9375 ;
vdfp('c_Svces','a_Manuf','USA') = 911648.4375 ;
vdfp('c_Svces','a_Manuf','EU_28') = 1453992.625 ;
vdfp('c_Svces','a_Manuf','CHN') = 1654679.75 ;
vdfp('c_Svces','a_Manuf','LatinAmer') = 349988.125 ;
vdfp('c_Svces','a_Manuf','ROW') = 1912825.375 ;
vdfp('c_Svces','a_Svces','USA') = 7352975 ;
vdfp('c_Svces','a_Svces','EU_28') = 6846185 ;
vdfp('c_Svces','a_Svces','CHN') = 4138268.75 ;
vdfp('c_Svces','a_Svces','LatinAmer') = 1221028.75 ;
vdfp('c_Svces','a_Svces','ROW') = 7685938.5 ;

* vmfb data (125 cells)
vmfb('c_Agri','a_Agri','USA') = 3115.156006 ;
vmfb('c_Agri','a_Agri','EU_28') = 11424.28809 ;
vmfb('c_Agri','a_Agri','CHN') = 7588.853027 ;
vmfb('c_Agri','a_Agri','LatinAmer') = 2717.823975 ;
vmfb('c_Agri','a_Agri','ROW') = 30297.79883 ;
vmfb('c_Agri','a_Food','USA') = 18982.03711 ;
vmfb('c_Agri','a_Food','EU_28') = 78343.48438 ;
vmfb('c_Agri','a_Food','CHN') = 36865.05078 ;
vmfb('c_Agri','a_Food','LatinAmer') = 15496.51465 ;
vmfb('c_Agri','a_Food','ROW') = 83114.40625 ;
vmfb('c_Agri','a_Energy','USA') = 4.786043644 ;
vmfb('c_Agri','a_Energy','EU_28') = 109.3571854 ;
vmfb('c_Agri','a_Energy','CHN') = 33.0625267 ;
vmfb('c_Agri','a_Energy','LatinAmer') = 21.17004013 ;
vmfb('c_Agri','a_Energy','ROW') = 125.1231613 ;
vmfb('c_Agri','a_Manuf','USA') = 1340.610962 ;
vmfb('c_Agri','a_Manuf','EU_28') = 8541.954102 ;
vmfb('c_Agri','a_Manuf','CHN') = 14959.10156 ;
vmfb('c_Agri','a_Manuf','LatinAmer') = 813.6791382 ;
vmfb('c_Agri','a_Manuf','ROW') = 16031.00879 ;
vmfb('c_Agri','a_Svces','USA') = 3635.301758 ;
vmfb('c_Agri','a_Svces','EU_28') = 14696.97168 ;
vmfb('c_Agri','a_Svces','CHN') = 7028.907715 ;
vmfb('c_Agri','a_Svces','LatinAmer') = 1765.857666 ;
vmfb('c_Agri','a_Svces','ROW') = 25804.92383 ;
vmfb('c_Food','a_Agri','USA') = 942.239624 ;
vmfb('c_Food','a_Agri','EU_28') = 8502.418945 ;
vmfb('c_Food','a_Agri','CHN') = 3454.560059 ;
vmfb('c_Food','a_Agri','LatinAmer') = 3389.242188 ;
vmfb('c_Food','a_Agri','ROW') = 23497.91797 ;
vmfb('c_Food','a_Food','USA') = 21591.69141 ;
vmfb('c_Food','a_Food','EU_28') = 90419.39844 ;
vmfb('c_Food','a_Food','CHN') = 17622.43359 ;
vmfb('c_Food','a_Food','LatinAmer') = 11593.16699 ;
vmfb('c_Food','a_Food','ROW') = 99997.57031 ;
vmfb('c_Food','a_Energy','USA') = 13.47304058 ;
vmfb('c_Food','a_Energy','EU_28') = 184.4763031 ;
vmfb('c_Food','a_Energy','CHN') = 131.5478973 ;
vmfb('c_Food','a_Energy','LatinAmer') = 12.88973713 ;
vmfb('c_Food','a_Energy','ROW') = 337.1084595 ;
vmfb('c_Food','a_Manuf','USA') = 795.4799805 ;
vmfb('c_Food','a_Manuf','EU_28') = 17734.88672 ;
vmfb('c_Food','a_Manuf','CHN') = 3961.746338 ;
vmfb('c_Food','a_Manuf','LatinAmer') = 1118.454712 ;
vmfb('c_Food','a_Manuf','ROW') = 5958.966797 ;
vmfb('c_Food','a_Svces','USA') = 19402.625 ;
vmfb('c_Food','a_Svces','EU_28') = 58425.66797 ;
vmfb('c_Food','a_Svces','CHN') = 8691.481445 ;
vmfb('c_Food','a_Svces','LatinAmer') = 7822.338867 ;
vmfb('c_Food','a_Svces','ROW') = 78957.13281 ;
vmfb('c_Energy','a_Agri','USA') = 967.6511841 ;
vmfb('c_Energy','a_Agri','EU_28') = 2627.845703 ;
vmfb('c_Energy','a_Agri','CHN') = 1296.378174 ;
vmfb('c_Energy','a_Agri','LatinAmer') = 1533.268555 ;
vmfb('c_Energy','a_Agri','ROW') = 7737.938965 ;
vmfb('c_Energy','a_Food','USA') = 186.7936401 ;
vmfb('c_Energy','a_Food','EU_28') = 4658.352539 ;
vmfb('c_Energy','a_Food','CHN') = 773.4063721 ;
vmfb('c_Energy','a_Food','LatinAmer') = 820.1263428 ;
vmfb('c_Energy','a_Food','ROW') = 4633.423828 ;
vmfb('c_Energy','a_Energy','USA') = 156432.4375 ;
vmfb('c_Energy','a_Energy','EU_28') = 313315 ;
vmfb('c_Energy','a_Energy','CHN') = 189651.9062 ;
vmfb('c_Energy','a_Energy','LatinAmer') = 55403.19531 ;
vmfb('c_Energy','a_Energy','ROW') = 560814.75 ;
vmfb('c_Energy','a_Manuf','USA') = 8529.691406 ;
vmfb('c_Energy','a_Manuf','EU_28') = 46838.10547 ;
vmfb('c_Energy','a_Manuf','CHN') = 23606.33398 ;
vmfb('c_Energy','a_Manuf','LatinAmer') = 15212.59375 ;
vmfb('c_Energy','a_Manuf','ROW') = 79833.42188 ;
vmfb('c_Energy','a_Svces','USA') = 32753.53711 ;
vmfb('c_Energy','a_Svces','EU_28') = 90007.88281 ;
vmfb('c_Energy','a_Svces','CHN') = 11842.75684 ;
vmfb('c_Energy','a_Svces','LatinAmer') = 39155.07812 ;
vmfb('c_Energy','a_Svces','ROW') = 138794.9844 ;
vmfb('c_Manuf','a_Agri','USA') = 10572.72852 ;
vmfb('c_Manuf','a_Agri','EU_28') = 20506.76562 ;
vmfb('c_Manuf','a_Agri','CHN') = 20505.3457 ;
vmfb('c_Manuf','a_Agri','LatinAmer') = 12329.48438 ;
vmfb('c_Manuf','a_Agri','ROW') = 66526.53125 ;
vmfb('c_Manuf','a_Food','USA') = 18820.74805 ;
vmfb('c_Manuf','a_Food','EU_28') = 50258.47266 ;
vmfb('c_Manuf','a_Food','CHN') = 10691.53125 ;
vmfb('c_Manuf','a_Food','LatinAmer') = 14893.00684 ;
vmfb('c_Manuf','a_Food','ROW') = 51941.98828 ;
vmfb('c_Manuf','a_Energy','USA') = 9609.510742 ;
vmfb('c_Manuf','a_Energy','EU_28') = 25616.57031 ;
vmfb('c_Manuf','a_Energy','CHN') = 10548.43359 ;
vmfb('c_Manuf','a_Energy','LatinAmer') = 9220.357422 ;
vmfb('c_Manuf','a_Energy','ROW') = 96793.96875 ;
vmfb('c_Manuf','a_Manuf','USA') = 507021.8438 ;
vmfb('c_Manuf','a_Manuf','EU_28') = 1884049.125 ;
vmfb('c_Manuf','a_Manuf','CHN') = 881362.8125 ;
vmfb('c_Manuf','a_Manuf','LatinAmer') = 348833.5 ;
vmfb('c_Manuf','a_Manuf','ROW') = 1959535.625 ;
vmfb('c_Manuf','a_Svces','USA') = 324454.1875 ;
vmfb('c_Manuf','a_Svces','EU_28') = 723781.875 ;
vmfb('c_Manuf','a_Svces','CHN') = 242184.7188 ;
vmfb('c_Manuf','a_Svces','LatinAmer') = 122099.5156 ;
vmfb('c_Manuf','a_Svces','ROW') = 892336.3125 ;
vmfb('c_Svces','a_Agri','USA') = 1633.474609 ;
vmfb('c_Svces','a_Agri','EU_28') = 8317.634766 ;
vmfb('c_Svces','a_Agri','CHN') = 2670.855713 ;
vmfb('c_Svces','a_Agri','LatinAmer') = 2007.528564 ;
vmfb('c_Svces','a_Agri','ROW') = 12495.15137 ;
vmfb('c_Svces','a_Food','USA') = 2725.504883 ;
vmfb('c_Svces','a_Food','EU_28') = 38091.91406 ;
vmfb('c_Svces','a_Food','CHN') = 4980.84375 ;
vmfb('c_Svces','a_Food','LatinAmer') = 4802.037598 ;
vmfb('c_Svces','a_Food','ROW') = 15799.75781 ;
vmfb('c_Svces','a_Energy','USA') = 2508.917725 ;
vmfb('c_Svces','a_Energy','EU_28') = 9512.299805 ;
vmfb('c_Svces','a_Energy','CHN') = 3307.572754 ;
vmfb('c_Svces','a_Energy','LatinAmer') = 5734.916992 ;
vmfb('c_Svces','a_Energy','ROW') = 41141.02734 ;
vmfb('c_Svces','a_Manuf','USA') = 21491.24609 ;
vmfb('c_Svces','a_Manuf','EU_28') = 239748.4531 ;
vmfb('c_Svces','a_Manuf','CHN') = 47086.63672 ;
vmfb('c_Svces','a_Manuf','LatinAmer') = 14798.47852 ;
vmfb('c_Svces','a_Manuf','ROW') = 124148.2422 ;
vmfb('c_Svces','a_Svces','USA') = 221647.8281 ;
vmfb('c_Svces','a_Svces','EU_28') = 1174965.75 ;
vmfb('c_Svces','a_Svces','CHN') = 125330.6562 ;
vmfb('c_Svces','a_Svces','LatinAmer') = 87761.875 ;
vmfb('c_Svces','a_Svces','ROW') = 678657.3125 ;

* vmfp data (125 cells)
vmfp('c_Agri','a_Agri','USA') = 3013.829346 ;
vmfp('c_Agri','a_Agri','EU_28') = 10970.46191 ;
vmfp('c_Agri','a_Agri','CHN') = 7544.106934 ;
vmfp('c_Agri','a_Agri','LatinAmer') = 2687.537598 ;
vmfp('c_Agri','a_Agri','ROW') = 29698.36914 ;
vmfp('c_Agri','a_Food','USA') = 19002.74023 ;
vmfp('c_Agri','a_Food','EU_28') = 79075.57031 ;
vmfp('c_Agri','a_Food','CHN') = 40778.19531 ;
vmfp('c_Agri','a_Food','LatinAmer') = 15921.90332 ;
vmfp('c_Agri','a_Food','ROW') = 84243.6875 ;
vmfp('c_Agri','a_Energy','USA') = 4.764171124 ;
vmfp('c_Agri','a_Energy','EU_28') = 109.6329041 ;
vmfp('c_Agri','a_Energy','CHN') = 36.43224716 ;
vmfp('c_Agri','a_Energy','LatinAmer') = 21.38573837 ;
vmfp('c_Agri','a_Energy','ROW') = 131.5113068 ;
vmfp('c_Agri','a_Manuf','USA') = 1332.666748 ;
vmfp('c_Agri','a_Manuf','EU_28') = 8589.479492 ;
vmfp('c_Agri','a_Manuf','CHN') = 16678.64453 ;
vmfp('c_Agri','a_Manuf','LatinAmer') = 820.725647 ;
vmfp('c_Agri','a_Manuf','ROW') = 16181.41992 ;
vmfp('c_Agri','a_Svces','USA') = 3623.306641 ;
vmfp('c_Agri','a_Svces','EU_28') = 15035.85742 ;
vmfp('c_Agri','a_Svces','CHN') = 7821.254395 ;
vmfp('c_Agri','a_Svces','LatinAmer') = 1813.459473 ;
vmfp('c_Agri','a_Svces','ROW') = 26109.30664 ;
vmfp('c_Food','a_Agri','USA') = 920.4552612 ;
vmfp('c_Food','a_Agri','EU_28') = 8217.826172 ;
vmfp('c_Food','a_Agri','CHN') = 3493.33667 ;
vmfp('c_Food','a_Agri','LatinAmer') = 3348.464844 ;
vmfp('c_Food','a_Agri','ROW') = 23464.08008 ;
vmfp('c_Food','a_Food','USA') = 21784.33398 ;
vmfp('c_Food','a_Food','EU_28') = 92270.02344 ;
vmfp('c_Food','a_Food','CHN') = 19478.89453 ;
vmfp('c_Food','a_Food','LatinAmer') = 12022.80469 ;
vmfp('c_Food','a_Food','ROW') = 103246.8125 ;
vmfp('c_Food','a_Energy','USA') = 13.57056046 ;
vmfp('c_Food','a_Energy','EU_28') = 188.8454437 ;
vmfp('c_Food','a_Energy','CHN') = 145.7225189 ;
vmfp('c_Food','a_Energy','LatinAmer') = 13.16069698 ;
vmfp('c_Food','a_Energy','ROW') = 346.0621643 ;
vmfp('c_Food','a_Manuf','USA') = 800.9435425 ;
vmfp('c_Food','a_Manuf','EU_28') = 18403.49609 ;
vmfp('c_Food','a_Manuf','CHN') = 4362.101074 ;
vmfp('c_Food','a_Manuf','LatinAmer') = 1176.743164 ;
vmfp('c_Food','a_Manuf','ROW') = 6147.183594 ;
vmfp('c_Food','a_Svces','USA') = 19492.0332 ;
vmfp('c_Food','a_Svces','EU_28') = 65124.02734 ;
vmfp('c_Food','a_Svces','CHN') = 9567.799805 ;
vmfp('c_Food','a_Svces','LatinAmer') = 8168.782715 ;
vmfp('c_Food','a_Svces','ROW') = 85612.59375 ;
vmfp('c_Energy','a_Agri','USA') = 1296.496826 ;
vmfp('c_Energy','a_Agri','EU_28') = 4727.34668 ;
vmfp('c_Energy','a_Agri','CHN') = 1457.386963 ;
vmfp('c_Energy','a_Agri','LatinAmer') = 1816.354736 ;
vmfp('c_Energy','a_Agri','ROW') = 8614.723633 ;
vmfp('c_Energy','a_Food','USA') = 196.8938293 ;
vmfp('c_Energy','a_Food','EU_28') = 5558.531738 ;
vmfp('c_Energy','a_Food','CHN') = 825.9315186 ;
vmfp('c_Energy','a_Food','LatinAmer') = 900.4014282 ;
vmfp('c_Energy','a_Food','ROW') = 4695.720703 ;
vmfp('c_Energy','a_Energy','USA') = 157361.5938 ;
vmfp('c_Energy','a_Energy','EU_28') = 315042.75 ;
vmfp('c_Energy','a_Energy','CHN') = 192800.5469 ;
vmfp('c_Energy','a_Energy','LatinAmer') = 55610.71094 ;
vmfp('c_Energy','a_Energy','ROW') = 556912.5625 ;
vmfp('c_Energy','a_Manuf','USA') = 10716.05859 ;
vmfp('c_Energy','a_Manuf','EU_28') = 68992.02344 ;
vmfp('c_Energy','a_Manuf','CHN') = 26913.06836 ;
vmfp('c_Energy','a_Manuf','LatinAmer') = 16776.88086 ;
vmfp('c_Energy','a_Manuf','ROW') = 83417.92188 ;
vmfp('c_Energy','a_Svces','USA') = 43398.08984 ;
vmfp('c_Energy','a_Svces','EU_28') = 164243.9375 ;
vmfp('c_Energy','a_Svces','CHN') = 13308.48926 ;
vmfp('c_Energy','a_Svces','LatinAmer') = 44117.08203 ;
vmfp('c_Energy','a_Svces','ROW') = 152863.7188 ;
vmfp('c_Manuf','a_Agri','USA') = 10317.47949 ;
vmfp('c_Manuf','a_Agri','EU_28') = 19905.38867 ;
vmfp('c_Manuf','a_Agri','CHN') = 20344.0957 ;
vmfp('c_Manuf','a_Agri','LatinAmer') = 12338.04102 ;
vmfp('c_Manuf','a_Agri','ROW') = 64810.32422 ;
vmfp('c_Manuf','a_Food','USA') = 19385.09766 ;
vmfp('c_Manuf','a_Food','EU_28') = 50736.78906 ;
vmfp('c_Manuf','a_Food','CHN') = 12335.90039 ;
vmfp('c_Manuf','a_Food','LatinAmer') = 15301.16406 ;
vmfp('c_Manuf','a_Food','ROW') = 52821.58594 ;
vmfp('c_Manuf','a_Energy','USA') = 9895.436523 ;
vmfp('c_Manuf','a_Energy','EU_28') = 25788.80273 ;
vmfp('c_Manuf','a_Energy','CHN') = 12006.05566 ;
vmfp('c_Manuf','a_Energy','LatinAmer') = 9505.841797 ;
vmfp('c_Manuf','a_Energy','ROW') = 98089.78906 ;
vmfp('c_Manuf','a_Manuf','USA') = 528007.25 ;
vmfp('c_Manuf','a_Manuf','EU_28') = 1898740.625 ;
vmfp('c_Manuf','a_Manuf','CHN') = 1004309.125 ;
vmfp('c_Manuf','a_Manuf','LatinAmer') = 355503.1562 ;
vmfp('c_Manuf','a_Manuf','ROW') = 1986334.125 ;
vmfp('c_Manuf','a_Svces','USA') = 337981.9375 ;
vmfp('c_Manuf','a_Svces','EU_28') = 764793.625 ;
vmfp('c_Manuf','a_Svces','CHN') = 274820.0938 ;
vmfp('c_Manuf','a_Svces','LatinAmer') = 127611.5 ;
vmfp('c_Manuf','a_Svces','ROW') = 912867.1875 ;
vmfp('c_Svces','a_Agri','USA') = 1593.259766 ;
vmfp('c_Svces','a_Agri','EU_28') = 8053.70752 ;
vmfp('c_Svces','a_Agri','CHN') = 2720.863525 ;
vmfp('c_Svces','a_Agri','LatinAmer') = 2026.738159 ;
vmfp('c_Svces','a_Agri','ROW') = 11934.39941 ;
vmfp('c_Svces','a_Food','USA') = 2756.831543 ;
vmfp('c_Svces','a_Food','EU_28') = 38390.19141 ;
vmfp('c_Svces','a_Food','CHN') = 5360.484375 ;
vmfp('c_Svces','a_Food','LatinAmer') = 4947.199707 ;
vmfp('c_Svces','a_Food','ROW') = 15884.74219 ;
vmfp('c_Svces','a_Energy','USA') = 2504.241699 ;
vmfp('c_Svces','a_Energy','EU_28') = 9605.416016 ;
vmfp('c_Svces','a_Energy','CHN') = 3562.682129 ;
vmfp('c_Svces','a_Energy','LatinAmer') = 5864.360352 ;
vmfp('c_Svces','a_Energy','ROW') = 41436.23047 ;
vmfp('c_Svces','a_Manuf','USA') = 21702.38086 ;
vmfp('c_Svces','a_Manuf','EU_28') = 241582.2656 ;
vmfp('c_Svces','a_Manuf','CHN') = 49950.53516 ;
vmfp('c_Svces','a_Manuf','LatinAmer') = 15137.97461 ;
vmfp('c_Svces','a_Manuf','ROW') = 124855.7422 ;
vmfp('c_Svces','a_Svces','USA') = 224986.5938 ;
vmfp('c_Svces','a_Svces','EU_28') = 1214330.25 ;
vmfp('c_Svces','a_Svces','CHN') = 136314.9219 ;
vmfp('c_Svces','a_Svces','LatinAmer') = 90671.48438 ;
vmfp('c_Svces','a_Svces','ROW') = 685714.5625 ;

* vdpb data (25 cells)
vdpb('c_Agri','USA') = 64309.02344 ;
vdpb('c_Agri','EU_28') = 102098.8594 ;
vdpb('c_Agri','CHN') = 406032.9375 ;
vdpb('c_Agri','LatinAmer') = 85459.96094 ;
vdpb('c_Agri','ROW') = 868073.875 ;
vdpb('c_Food','USA') = 638353.25 ;
vdpb('c_Food','EU_28') = 536500.6875 ;
vdpb('c_Food','CHN') = 778296.9375 ;
vdpb('c_Food','LatinAmer') = 462912.6875 ;
vdpb('c_Food','ROW') = 1487878.125 ;
vdpb('c_Energy','USA') = 239341.6406 ;
vdpb('c_Energy','EU_28') = 184299.9531 ;
vdpb('c_Energy','CHN') = 153850.4062 ;
vdpb('c_Energy','LatinAmer') = 90066.77344 ;
vdpb('c_Energy','ROW') = 571792.125 ;
vdpb('c_Manuf','USA') = 711828.625 ;
vdpb('c_Manuf','EU_28') = 379901.625 ;
vdpb('c_Manuf','CHN') = 618153 ;
vdpb('c_Manuf','LatinAmer') = 373707.75 ;
vdpb('c_Manuf','ROW') = 1051209 ;
vdpb('c_Svces','USA') = 10502550 ;
vdpb('c_Svces','EU_28') = 6331369.5 ;
vdpb('c_Svces','CHN') = 2267669 ;
vdpb('c_Svces','LatinAmer') = 2257558 ;
vdpb('c_Svces','ROW') = 8788728 ;

* vdpp data (25 cells)
vdpp('c_Agri','USA') = 64258.19141 ;
vdpp('c_Agri','EU_28') = 111013.6719 ;
vdpp('c_Agri','CHN') = 386537.3438 ;
vdpp('c_Agri','LatinAmer') = 87161.66406 ;
vdpp('c_Agri','ROW') = 872162.3125 ;
vdpp('c_Food','USA') = 636916.9375 ;
vdpp('c_Food','EU_28') = 712653.3125 ;
vdpp('c_Food','CHN') = 849193.1875 ;
vdpp('c_Food','LatinAmer') = 503925.8438 ;
vdpp('c_Food','ROW') = 1585763.125 ;
vdpp('c_Energy','USA') = 276856.5312 ;
vdpp('c_Energy','EU_28') = 382142.3438 ;
vdpp('c_Energy','CHN') = 148529.5625 ;
vdpp('c_Energy','LatinAmer') = 126182.7031 ;
vdpp('c_Energy','ROW') = 579943.75 ;
vdpp('c_Manuf','USA') = 697360.625 ;
vdpp('c_Manuf','EU_28') = 480620.25 ;
vdpp('c_Manuf','CHN') = 652622.375 ;
vdpp('c_Manuf','LatinAmer') = 424137.75 ;
vdpp('c_Manuf','ROW') = 1110862.875 ;
vdpp('c_Svces','USA') = 10726112 ;
vdpp('c_Svces','EU_28') = 6579813.5 ;
vdpp('c_Svces','CHN') = 2443109.5 ;
vdpp('c_Svces','LatinAmer') = 2345675.5 ;
vdpp('c_Svces','ROW') = 8954946 ;

* vmpb data (25 cells)
vmpb('c_Agri','USA') = 24094.9082 ;
vmpb('c_Agri','EU_28') = 65286.51172 ;
vmpb('c_Agri','CHN') = 20579.50586 ;
vmpb('c_Agri','LatinAmer') = 7996.402832 ;
vmpb('c_Agri','ROW') = 86824.83594 ;
vmpb('c_Food','USA') = 67921.8125 ;
vmpb('c_Food','EU_28') = 217179.4375 ;
vmpb('c_Food','CHN') = 30418.91992 ;
vmpb('c_Food','LatinAmer') = 41635.84375 ;
vmpb('c_Food','ROW') = 270948.5312 ;
vmpb('c_Energy','USA') = 17448.66016 ;
vmpb('c_Energy','EU_28') = 38157.03516 ;
vmpb('c_Energy','CHN') = 7418.686035 ;
vmpb('c_Energy','LatinAmer') = 9293.576172 ;
vmpb('c_Energy','ROW') = 65924.91406 ;
vmpb('c_Manuf','USA') = 616157 ;
vmpb('c_Manuf','EU_28') = 802164.125 ;
vmpb('c_Manuf','CHN') = 141663.2188 ;
vmpb('c_Manuf','LatinAmer') = 155067.5781 ;
vmpb('c_Manuf','ROW') = 940571.1875 ;
vmpb('c_Svces','USA') = 178576.3125 ;
vmpb('c_Svces','EU_28') = 159395.5 ;
vmpb('c_Svces','CHN') = 78959.98438 ;
vmpb('c_Svces','LatinAmer') = 94672.34375 ;
vmpb('c_Svces','ROW') = 472722.1875 ;

* vmpp data (25 cells)
vmpp('c_Agri','USA') = 24056.82031 ;
vmpp('c_Agri','EU_28') = 72266.21875 ;
vmpp('c_Agri','CHN') = 25398.99414 ;
vmpp('c_Agri','LatinAmer') = 8257.866211 ;
vmpp('c_Agri','ROW') = 88863.25 ;
vmpp('c_Food','USA') = 67828.24219 ;
vmpp('c_Food','EU_28') = 288939.6562 ;
vmpp('c_Food','CHN') = 37009.6875 ;
vmpp('c_Food','LatinAmer') = 45970.14062 ;
vmpp('c_Food','ROW') = 311346.1875 ;
vmpp('c_Energy','USA') = 24221.68164 ;
vmpp('c_Energy','EU_28') = 87695.08594 ;
vmpp('c_Energy','CHN') = 7338.152832 ;
vmpp('c_Energy','LatinAmer') = 15193.38574 ;
vmpp('c_Energy','ROW') = 84172.57812 ;
vmpp('c_Manuf','USA') = 631310.8125 ;
vmpp('c_Manuf','EU_28') = 1042501.188 ;
vmpp('c_Manuf','CHN') = 177812.2344 ;
vmpp('c_Manuf','LatinAmer') = 172019.4375 ;
vmpp('c_Manuf','ROW') = 1017669.5 ;
vmpp('c_Svces','USA') = 183644.7344 ;
vmpp('c_Svces','EU_28') = 169805.3281 ;
vmpp('c_Svces','CHN') = 93180.66406 ;
vmpp('c_Svces','LatinAmer') = 100198.2891 ;
vmpp('c_Svces','ROW') = 482900.0938 ;

* vdgb data (20 cells)
vdgb('c_Agri','USA') = 1005.939697 ;
vdgb('c_Agri','EU_28') = 4358.82959 ;
vdgb('c_Agri','CHN') = 1414.880737 ;
vdgb('c_Agri','LatinAmer') = 533.4562988 ;
vdgb('c_Agri','ROW') = 4732.152344 ;
vdgb('c_Food','USA') = 17.83016968 ;
vdgb('c_Food','EU_28') = 916.1443481 ;
vdgb('c_Food','CHN') = 13.15686607 ;
vdgb('c_Food','LatinAmer') = 372.4311829 ;
vdgb('c_Food','ROW') = 5307.375977 ;
vdgb('c_Manuf','USA') = 405.8898315 ;
vdgb('c_Manuf','EU_28') = 35854.46094 ;
vdgb('c_Manuf','CHN') = 297.0192566 ;
vdgb('c_Manuf','LatinAmer') = 1593.909058 ;
vdgb('c_Manuf','ROW') = 57794.92969 ;
vdgb('c_Svces','USA') = 2543765 ;
vdgb('c_Svces','EU_28') = 3504409.5 ;
vdgb('c_Svces','CHN') = 1978844.125 ;
vdgb('c_Svces','LatinAmer') = 924374.6875 ;
vdgb('c_Svces','ROW') = 4071279.5 ;

* vdgp data (20 cells)
vdgp('c_Agri','USA') = 1005.939697 ;
vdgp('c_Agri','EU_28') = 4444.121582 ;
vdgp('c_Agri','CHN') = 1414.880737 ;
vdgp('c_Agri','LatinAmer') = 543.3477783 ;
vdgp('c_Agri','ROW') = 4763.270508 ;
vdgp('c_Food','USA') = 17.83016968 ;
vdgp('c_Food','EU_28') = 1010.377075 ;
vdgp('c_Food','CHN') = 13.15686607 ;
vdgp('c_Food','LatinAmer') = 392.0939636 ;
vdgp('c_Food','ROW') = 5349.387207 ;
vdgp('c_Manuf','USA') = 405.8898315 ;
vdgp('c_Manuf','EU_28') = 40958.1875 ;
vdgp('c_Manuf','CHN') = 297.0192566 ;
vdgp('c_Manuf','LatinAmer') = 1611.773193 ;
vdgp('c_Manuf','ROW') = 58248.88672 ;
vdgp('c_Svces','USA') = 2729009.5 ;
vdgp('c_Svces','EU_28') = 3507268.75 ;
vdgp('c_Svces','CHN') = 2000814.25 ;
vdgp('c_Svces','LatinAmer') = 925519.6875 ;
vdgp('c_Svces','ROW') = 4091004.25 ;

* vmgb data (20 cells)
vmgb('c_Agri','USA') = 1.614451051 ;
vmgb('c_Agri','EU_28') = 6.724220753 ;
vmgb('c_Agri','CHN') = 0.8614827991 ;
vmgb('c_Agri','LatinAmer') = 2.849271059 ;
vmgb('c_Agri','ROW') = 337.1213379 ;
vmgb('c_Food','USA') = 6.32793808 ;
vmgb('c_Food','EU_28') = 194.1978912 ;
vmgb('c_Food','CHN') = 3.532931328 ;
vmgb('c_Food','LatinAmer') = 53.76917267 ;
vmgb('c_Food','ROW') = 1556.568848 ;
vmgb('c_Manuf','USA') = 331.9682617 ;
vmgb('c_Manuf','EU_28') = 68117.15625 ;
vmgb('c_Manuf','CHN') = 474.9967957 ;
vmgb('c_Manuf','LatinAmer') = 813.696167 ;
vmgb('c_Manuf','ROW') = 65790.67188 ;
vmgb('c_Svces','USA') = 19186.18555 ;
vmgb('c_Svces','EU_28') = 7234.123535 ;
vmgb('c_Svces','CHN') = 33935.25391 ;
vmgb('c_Svces','LatinAmer') = 6868.725098 ;
vmgb('c_Svces','ROW') = 61176.13281 ;

* vmgp data (20 cells)
vmgp('c_Agri','USA') = 1.614451051 ;
vmgp('c_Agri','EU_28') = 6.941887856 ;
vmgp('c_Agri','CHN') = 0.8614827991 ;
vmgp('c_Agri','LatinAmer') = 2.868597031 ;
vmgp('c_Agri','ROW') = 337.6085205 ;
vmgp('c_Food','USA') = 6.32793808 ;
vmgp('c_Food','EU_28') = 216.5643921 ;
vmgp('c_Food','CHN') = 3.532931328 ;
vmgp('c_Food','LatinAmer') = 54.48241806 ;
vmgp('c_Food','ROW') = 1578.138062 ;
vmgp('c_Manuf','USA') = 331.9682617 ;
vmgp('c_Manuf','EU_28') = 76792.6875 ;
vmgp('c_Manuf','CHN') = 474.9967957 ;
vmgp('c_Manuf','LatinAmer') = 834.3256836 ;
vmgp('c_Manuf','ROW') = 65687.26562 ;
vmgp('c_Svces','USA') = 19186.18555 ;
vmgp('c_Svces','EU_28') = 9761.845703 ;
vmgp('c_Svces','CHN') = 38809.62109 ;
vmgp('c_Svces','LatinAmer') = 7140.59668 ;
vmgp('c_Svces','ROW') = 61764.18359 ;

* vdib data (20 cells)
vdib('c_Agri','USA') = 501.6325073 ;
vdib('c_Agri','EU_28') = 5512.021973 ;
vdib('c_Agri','CHN') = 24333.41016 ;
vdib('c_Agri','LatinAmer') = 12658.22461 ;
vdib('c_Agri','ROW') = 36491.71875 ;
vdib('c_Food','USA') = 21.01760101 ;
vdib('c_Food','EU_28') = 184.1103363 ;
vdib('c_Food','CHN') = 26.22566414 ;
vdib('c_Food','LatinAmer') = 2437.61499 ;
vdib('c_Food','ROW') = 10194.59668 ;
vdib('c_Manuf','USA') = 765932.4375 ;
vdib('c_Manuf','EU_28') = 473251.4688 ;
vdib('c_Manuf','CHN') = 947260.6875 ;
vdib('c_Manuf','LatinAmer') = 166852.6875 ;
vdib('c_Manuf','ROW') = 987661.625 ;
vdib('c_Svces','USA') = 2873172.5 ;
vdib('c_Svces','EU_28') = 2258155.25 ;
vdib('c_Svces','CHN') = 3693320.5 ;
vdib('c_Svces','LatinAmer') = 658275.25 ;
vdib('c_Svces','ROW') = 4539059 ;

* vdip data (20 cells)
vdip('c_Agri','USA') = 501.6325073 ;
vdip('c_Agri','EU_28') = 5307.625488 ;
vdip('c_Agri','CHN') = 24359.67773 ;
vdip('c_Agri','LatinAmer') = 13212.60254 ;
vdip('c_Agri','ROW') = 36458.83594 ;
vdip('c_Food','USA') = 21.01760101 ;
vdip('c_Food','EU_28') = 183.9213867 ;
vdip('c_Food','CHN') = 26.22566414 ;
vdip('c_Food','LatinAmer') = 2473.797119 ;
vdip('c_Food','ROW') = 10416.03613 ;
vdip('c_Manuf','USA') = 775809.125 ;
vdip('c_Manuf','EU_28') = 490787.6875 ;
vdip('c_Manuf','CHN') = 1020333.438 ;
vdip('c_Manuf','LatinAmer') = 181197.9062 ;
vdip('c_Manuf','ROW') = 1020758.688 ;
vdip('c_Svces','USA') = 2692840.75 ;
vdip('c_Svces','EU_28') = 2438634.25 ;
vdip('c_Svces','CHN') = 3877627.75 ;
vdip('c_Svces','LatinAmer') = 668482.75 ;
vdip('c_Svces','ROW') = 4634943.5 ;

* vmib data (20 cells)
vmib('c_Agri','USA') = 11.86547756 ;
vmib('c_Agri','EU_28') = 1697.599854 ;
vmib('c_Agri','CHN') = 395.394165 ;
vmib('c_Agri','LatinAmer') = 596.8382568 ;
vmib('c_Agri','ROW') = 4238.918457 ;
vmib('c_Food','USA') = 10.54254627 ;
vmib('c_Food','EU_28') = 350.4533691 ;
vmib('c_Food','CHN') = 10.2843914 ;
vmib('c_Food','LatinAmer') = 44.69393539 ;
vmib('c_Food','ROW') = 4229.597656 ;
vmib('c_Manuf','USA') = 473778.375 ;
vmib('c_Manuf','EU_28') = 582482.6875 ;
vmib('c_Manuf','CHN') = 235322.7812 ;
vmib('c_Manuf','LatinAmer') = 153429.5 ;
vmib('c_Manuf','ROW') = 905921.5625 ;
vmib('c_Svces','USA') = 99352.00781 ;
vmib('c_Svces','EU_28') = 92506.16406 ;
vmib('c_Svces','CHN') = 16403.72266 ;
vmib('c_Svces','LatinAmer') = 5939.583496 ;
vmib('c_Svces','ROW') = 117026.6875 ;

* vmip data (20 cells)
vmip('c_Agri','USA') = 11.86547756 ;
vmip('c_Agri','EU_28') = 1726.700562 ;
vmip('c_Agri','CHN') = 471.7236023 ;
vmip('c_Agri','LatinAmer') = 610.0951538 ;
vmip('c_Agri','ROW') = 4285.651367 ;
vmip('c_Food','USA') = 10.54254627 ;
vmip('c_Food','EU_28') = 350.6495972 ;
vmip('c_Food','CHN') = 10.2843914 ;
vmip('c_Food','LatinAmer') = 45.01720428 ;
vmip('c_Food','ROW') = 4256.712891 ;
vmip('c_Manuf','USA') = 491439.5 ;
vmip('c_Manuf','EU_28') = 614703.125 ;
vmip('c_Manuf','CHN') = 293470.6875 ;
vmip('c_Manuf','LatinAmer') = 161907.0625 ;
vmip('c_Manuf','ROW') = 940479.9375 ;
vmip('c_Svces','USA') = 87960.89844 ;
vmip('c_Svces','EU_28') = 99512.54688 ;
vmip('c_Svces','CHN') = 19477.99023 ;
vmip('c_Svces','LatinAmer') = 6202.465332 ;
vmip('c_Svces','ROW') = 117596.6875 ;

* evfb data (95 cells)
evfb('Land','a_Agri','USA') = 41351.43359 ;
evfb('Land','a_Agri','EU_28') = 67458 ;
evfb('Land','a_Agri','CHN') = 251005.1875 ;
evfb('Land','a_Agri','LatinAmer') = 59448.42578 ;
evfb('Land','a_Agri','ROW') = 358800.625 ;
evfb('UnSkLab','a_Agri','USA') = 32854.89844 ;
evfb('UnSkLab','a_Agri','EU_28') = 82710.47656 ;
evfb('UnSkLab','a_Agri','CHN') = 378059.0625 ;
evfb('UnSkLab','a_Agri','LatinAmer') = 114264.7031 ;
evfb('UnSkLab','a_Agri','ROW') = 624863.625 ;
evfb('UnSkLab','a_Food','USA') = 86790.16406 ;
evfb('UnSkLab','a_Food','EU_28') = 59448.61719 ;
evfb('UnSkLab','a_Food','CHN') = 138670.3281 ;
evfb('UnSkLab','a_Food','LatinAmer') = 56313.24219 ;
evfb('UnSkLab','a_Food','ROW') = 207095.7656 ;
evfb('UnSkLab','a_Energy','USA') = 52987.96094 ;
evfb('UnSkLab','a_Energy','EU_28') = 19774.36133 ;
evfb('UnSkLab','a_Energy','CHN') = 107518.1094 ;
evfb('UnSkLab','a_Energy','LatinAmer') = 17726.69336 ;
evfb('UnSkLab','a_Energy','ROW') = 180954.7812 ;
evfb('UnSkLab','a_Manuf','USA') = 691563.75 ;
evfb('UnSkLab','a_Manuf','EU_28') = 436514.6875 ;
evfb('UnSkLab','a_Manuf','CHN') = 1026031.438 ;
evfb('UnSkLab','a_Manuf','LatinAmer') = 193146.5312 ;
evfb('UnSkLab','a_Manuf','ROW') = 900654.6875 ;
evfb('UnSkLab','a_Svces','USA') = 3282724.5 ;
evfb('UnSkLab','a_Svces','EU_28') = 1784304.5 ;
evfb('UnSkLab','a_Svces','CHN') = 2457188 ;
evfb('UnSkLab','a_Svces','LatinAmer') = 937292.4375 ;
evfb('UnSkLab','a_Svces','ROW') = 3716833 ;
evfb('SkLab','a_Agri','USA') = 32099.72266 ;
evfb('SkLab','a_Agri','EU_28') = 23833.0293 ;
evfb('SkLab','a_Agri','CHN') = 23754.16602 ;
evfb('SkLab','a_Agri','LatinAmer') = 12459.03223 ;
evfb('SkLab','a_Agri','ROW') = 27777.48047 ;
evfb('SkLab','a_Food','USA') = 29155.53906 ;
evfb('SkLab','a_Food','EU_28') = 66590.28125 ;
evfb('SkLab','a_Food','CHN') = 27360.30859 ;
evfb('SkLab','a_Food','LatinAmer') = 27975.19336 ;
evfb('SkLab','a_Food','ROW') = 85943.95312 ;
evfb('SkLab','a_Energy','USA') = 21668.21484 ;
evfb('SkLab','a_Energy','EU_28') = 23952.67188 ;
evfb('SkLab','a_Energy','CHN') = 21067.27148 ;
evfb('SkLab','a_Energy','LatinAmer') = 13941.58984 ;
evfb('SkLab','a_Energy','ROW') = 153880.2656 ;
evfb('SkLab','a_Manuf','USA') = 229677.9219 ;
evfb('SkLab','a_Manuf','EU_28') = 473903.8438 ;
evfb('SkLab','a_Manuf','CHN') = 198534.2344 ;
evfb('SkLab','a_Manuf','LatinAmer') = 79832.89062 ;
evfb('SkLab','a_Manuf','ROW') = 471463.5625 ;
evfb('SkLab','a_Svces','USA') = 3780121.5 ;
evfb('SkLab','a_Svces','EU_28') = 2512409.5 ;
evfb('SkLab','a_Svces','CHN') = 1334502.625 ;
evfb('SkLab','a_Svces','LatinAmer') = 649102.875 ;
evfb('SkLab','a_Svces','ROW') = 3661292 ;
evfb('Capital','a_Agri','USA') = 92125.17969 ;
evfb('Capital','a_Agri','EU_28') = 107050.1406 ;
evfb('Capital','a_Agri','CHN') = 256185.375 ;
evfb('Capital','a_Agri','LatinAmer') = 85173.6875 ;
evfb('Capital','a_Agri','ROW') = 376261.1875 ;
evfb('Capital','a_Food','USA') = 133316.7812 ;
evfb('Capital','a_Food','EU_28') = 187744.5312 ;
evfb('Capital','a_Food','CHN') = 185199.4531 ;
evfb('Capital','a_Food','LatinAmer') = 147757.25 ;
evfb('Capital','a_Food','ROW') = 446692.0938 ;
evfb('Capital','a_Energy','USA') = 220980.7656 ;
evfb('Capital','a_Energy','EU_28') = 165926.8906 ;
evfb('Capital','a_Energy','CHN') = 187239.3438 ;
evfb('Capital','a_Energy','LatinAmer') = 161737.0781 ;
evfb('Capital','a_Energy','ROW') = 1143287.125 ;
evfb('Capital','a_Manuf','USA') = 877950.375 ;
evfb('Capital','a_Manuf','EU_28') = 1058059.375 ;
evfb('Capital','a_Manuf','CHN') = 1268539.25 ;
evfb('Capital','a_Manuf','LatinAmer') = 396093.2812 ;
evfb('Capital','a_Manuf','ROW') = 2029665.875 ;
evfb('Capital','a_Svces','USA') = 5918652.5 ;
evfb('Capital','a_Svces','EU_28') = 5561561.5 ;
evfb('Capital','a_Svces','CHN') = 2660009.25 ;
evfb('Capital','a_Svces','LatinAmer') = 1825079.125 ;
evfb('Capital','a_Svces','ROW') = 8119683.5 ;
evfb('NatRes','a_Agri','USA') = 5286.424805 ;
evfb('NatRes','a_Agri','EU_28') = 8407.166016 ;
evfb('NatRes','a_Agri','CHN') = 44576.29688 ;
evfb('NatRes','a_Agri','LatinAmer') = 7114.182129 ;
evfb('NatRes','a_Agri','ROW') = 60747.79297 ;
evfb('NatRes','a_Energy','USA') = 61101.10156 ;
evfb('NatRes','a_Energy','EU_28') = 13122.53516 ;
evfb('NatRes','a_Energy','CHN') = 53481.14453 ;
evfb('NatRes','a_Energy','LatinAmer') = 43739.21875 ;
evfb('NatRes','a_Energy','ROW') = 409375.6875 ;
evfb('NatRes','a_Manuf','USA') = 14772.66016 ;
evfb('NatRes','a_Manuf','EU_28') = 6984.152344 ;
evfb('NatRes','a_Manuf','CHN') = 15029.12793 ;
evfb('NatRes','a_Manuf','LatinAmer') = 9267.223633 ;
evfb('NatRes','a_Manuf','ROW') = 41160.51562 ;

* evfp data (95 cells)
evfp('Land','a_Agri','USA') = 34003.03516 ;
evfp('Land','a_Agri','EU_28') = 45434.83203 ;
evfp('Land','a_Agri','CHN') = 228201.2031 ;
evfp('Land','a_Agri','LatinAmer') = 59821.10547 ;
evfp('Land','a_Agri','ROW') = 352268.875 ;
evfp('UnSkLab','a_Agri','USA') = 37470.65234 ;
evfp('UnSkLab','a_Agri','EU_28') = 101959.7734 ;
evfp('UnSkLab','a_Agri','CHN') = 375595.7188 ;
evfp('UnSkLab','a_Agri','LatinAmer') = 133473.3281 ;
evfp('UnSkLab','a_Agri','ROW') = 654104.4375 ;
evfp('UnSkLab','a_Food','USA') = 104457.9766 ;
evfp('UnSkLab','a_Food','EU_28') = 86886.41406 ;
evfp('UnSkLab','a_Food','CHN') = 141001.4219 ;
evfp('UnSkLab','a_Food','LatinAmer') = 65632.14844 ;
evfp('UnSkLab','a_Food','ROW') = 225252.2031 ;
evfp('UnSkLab','a_Energy','USA') = 63774.6875 ;
evfp('UnSkLab','a_Energy','EU_28') = 28768.41016 ;
evfp('UnSkLab','a_Energy','CHN') = 109325.5156 ;
evfp('UnSkLab','a_Energy','LatinAmer') = 20720.83789 ;
evfp('UnSkLab','a_Energy','ROW') = 198626.5469 ;
evfp('UnSkLab','a_Manuf','USA') = 832344.9375 ;
evfp('UnSkLab','a_Manuf','EU_28') = 639787.75 ;
evfp('UnSkLab','a_Manuf','CHN') = 1043279.312 ;
evfp('UnSkLab','a_Manuf','LatinAmer') = 224882.9688 ;
evfp('UnSkLab','a_Manuf','ROW') = 1047419.5 ;
evfp('UnSkLab','a_Svces','USA') = 3950986.75 ;
evfp('UnSkLab','a_Svces','EU_28') = 2543290 ;
evfp('UnSkLab','a_Svces','CHN') = 2498494 ;
evfp('UnSkLab','a_Svces','LatinAmer') = 1103080.25 ;
evfp('UnSkLab','a_Svces','ROW') = 4278355.5 ;
evfp('SkLab','a_Agri','USA') = 36611.21094 ;
evfp('SkLab','a_Agri','EU_28') = 28117.21289 ;
evfp('SkLab','a_Agri','CHN') = 23596.51758 ;
evfp('SkLab','a_Agri','LatinAmer') = 13845.66797 ;
evfp('SkLab','a_Agri','ROW') = 29669.06836 ;
evfp('SkLab','a_Food','USA') = 35090.71484 ;
evfp('SkLab','a_Food','EU_28') = 95431.53906 ;
evfp('SkLab','a_Food','CHN') = 27820.24219 ;
evfp('SkLab','a_Food','LatinAmer') = 31990.30273 ;
evfp('SkLab','a_Food','ROW') = 96940.85938 ;
evfp('SkLab','a_Energy','USA') = 26079.20117 ;
evfp('SkLab','a_Energy','EU_28') = 34500.87109 ;
evfp('SkLab','a_Energy','CHN') = 21421.41797 ;
evfp('SkLab','a_Energy','LatinAmer') = 15900.03711 ;
evfp('SkLab','a_Energy','ROW') = 169267.7188 ;
evfp('SkLab','a_Manuf','USA') = 276433.3125 ;
evfp('SkLab','a_Manuf','EU_28') = 683187.625 ;
evfp('SkLab','a_Manuf','CHN') = 201871.6562 ;
evfp('SkLab','a_Manuf','LatinAmer') = 91532.28125 ;
evfp('SkLab','a_Manuf','ROW') = 558496.6875 ;
evfp('SkLab','a_Svces','USA') = 4549638.5 ;
evfp('SkLab','a_Svces','EU_28') = 3595405 ;
evfp('SkLab','a_Svces','CHN') = 1356936 ;
evfp('SkLab','a_Svces','LatinAmer') = 748339.75 ;
evfp('SkLab','a_Svces','ROW') = 4238578 ;
evfp('Capital','a_Agri','USA') = 93103.03125 ;
evfp('Capital','a_Agri','EU_28') = 92191.05469 ;
evfp('Capital','a_Agri','CHN') = 243381.7656 ;
evfp('Capital','a_Agri','LatinAmer') = 83529.85156 ;
evfp('Capital','a_Agri','ROW') = 371885.1562 ;
evfp('Capital','a_Food','USA') = 139993.5156 ;
evfp('Capital','a_Food','EU_28') = 194236.0469 ;
evfp('Capital','a_Food','CHN') = 188312.7031 ;
evfp('Capital','a_Food','LatinAmer') = 150112.4688 ;
evfp('Capital','a_Food','ROW') = 451633.4375 ;
evfp('Capital','a_Energy','USA') = 232047.8594 ;
evfp('Capital','a_Energy','EU_28') = 171349.125 ;
evfp('Capital','a_Energy','CHN') = 190386.875 ;
evfp('Capital','a_Energy','LatinAmer') = 164931.7344 ;
evfp('Capital','a_Energy','ROW') = 1156469.625 ;
evfp('Capital','a_Manuf','USA') = 921919.625 ;
evfp('Capital','a_Manuf','EU_28') = 1087454.875 ;
evfp('Capital','a_Manuf','CHN') = 1289863.75 ;
evfp('Capital','a_Manuf','LatinAmer') = 402780.8438 ;
evfp('Capital','a_Manuf','ROW') = 2063073.25 ;
evfp('Capital','a_Svces','USA') = 6215069 ;
evfp('Capital','a_Svces','EU_28') = 5746506.5 ;
evfp('Capital','a_Svces','CHN') = 2704724.75 ;
evfp('Capital','a_Svces','LatinAmer') = 1858644 ;
evfp('Capital','a_Svces','ROW') = 8267268 ;
evfp('NatRes','a_Agri','USA') = 5551.177734 ;
evfp('NatRes','a_Agri','EU_28') = 8703.983398 ;
evfp('NatRes','a_Agri','CHN') = 45325.63672 ;
evfp('NatRes','a_Agri','LatinAmer') = 7230.231445 ;
evfp('NatRes','a_Agri','ROW') = 61200.86719 ;
evfp('NatRes','a_Energy','USA') = 64161.14844 ;
evfp('NatRes','a_Energy','EU_28') = 13490.23828 ;
evfp('NatRes','a_Energy','CHN') = 54380.17578 ;
evfp('NatRes','a_Energy','LatinAmer') = 44641.88281 ;
evfp('NatRes','a_Energy','ROW') = 414077.7188 ;
evfp('NatRes','a_Manuf','USA') = 15512.5 ;
evfp('NatRes','a_Manuf','EU_28') = 7221.031738 ;
evfp('NatRes','a_Manuf','CHN') = 15281.77148 ;
evfp('NatRes','a_Manuf','LatinAmer') = 9408.245117 ;
evfp('NatRes','a_Manuf','ROW') = 41962.19531 ;

* evos data (95 cells)
evos('Land','a_Agri','USA') = 39669.44531 ;
evos('Land','a_Agri','EU_28') = 63766.05859 ;
evos('Land','a_Agri','CHN') = 223061.875 ;
evos('Land','a_Agri','LatinAmer') = 55195.61328 ;
evos('Land','a_Agri','ROW') = 335562.3125 ;
evos('UnSkLab','a_Agri','USA') = 24738.65039 ;
evos('UnSkLab','a_Agri','EU_28') = 57630.91016 ;
evos('UnSkLab','a_Agri','CHN') = 366342.1875 ;
evos('UnSkLab','a_Agri','LatinAmer') = 105564.7109 ;
evos('UnSkLab','a_Agri','ROW') = 582229.8125 ;
evos('UnSkLab','a_Food','USA') = 65350.11328 ;
evos('UnSkLab','a_Food','EU_28') = 41500.99219 ;
evos('UnSkLab','a_Food','CHN') = 134372.6406 ;
evos('UnSkLab','a_Food','LatinAmer') = 51998.39844 ;
evos('UnSkLab','a_Food','ROW') = 190074.6875 ;
evos('UnSkLab','a_Energy','USA') = 39898.17969 ;
evos('UnSkLab','a_Energy','EU_28') = 14185.1875 ;
evos('UnSkLab','a_Energy','CHN') = 104185.8906 ;
evos('UnSkLab','a_Energy','LatinAmer') = 16198.49023 ;
evos('UnSkLab','a_Energy','ROW') = 165166.7656 ;
evos('UnSkLab','a_Manuf','USA') = 520724.5938 ;
evos('UnSkLab','a_Manuf','EU_28') = 300480.5312 ;
evos('UnSkLab','a_Manuf','CHN') = 994232.5 ;
evos('UnSkLab','a_Manuf','LatinAmer') = 177115.9688 ;
evos('UnSkLab','a_Manuf','ROW') = 796999.1875 ;
evos('UnSkLab','a_Svces','USA') = 2471782.75 ;
evos('UnSkLab','a_Svces','EU_28') = 1264046.5 ;
evos('UnSkLab','a_Svces','CHN') = 2381034.75 ;
evos('UnSkLab','a_Svces','LatinAmer') = 861850.875 ;
evos('UnSkLab','a_Svces','ROW') = 3266970.5 ;
evos('SkLab','a_Agri','USA') = 24170.0293 ;
evos('SkLab','a_Agri','EU_28') = 15801.24414 ;
evos('SkLab','a_Agri','CHN') = 23017.97461 ;
evos('SkLab','a_Agri','LatinAmer') = 11788.13574 ;
evos('SkLab','a_Agri','ROW') = 23710.57227 ;
evos('SkLab','a_Food','USA') = 21953.15625 ;
evos('SkLab','a_Food','EU_28') = 46888.70312 ;
evos('SkLab','a_Food','CHN') = 26512.35352 ;
evos('SkLab','a_Food','LatinAmer') = 26133.65625 ;
evos('SkLab','a_Food','ROW') = 75277.44531 ;
evos('SkLab','a_Energy','USA') = 16315.44922 ;
evos('SkLab','a_Energy','EU_28') = 17062.15625 ;
evos('SkLab','a_Energy','CHN') = 20414.35156 ;
evos('SkLab','a_Energy','LatinAmer') = 12683.43164 ;
evos('SkLab','a_Energy','ROW') = 139775.5156 ;
evos('SkLab','a_Manuf','USA') = 172939.875 ;
evos('SkLab','a_Manuf','EU_28') = 328497.6562 ;
evos('SkLab','a_Manuf','CHN') = 192381.2344 ;
evos('SkLab','a_Manuf','LatinAmer') = 74211.76562 ;
evos('SkLab','a_Manuf','ROW') = 405182.0938 ;
evos('SkLab','a_Svces','USA') = 2846306.5 ;
evos('SkLab','a_Svces','EU_28') = 1767065.125 ;
evos('SkLab','a_Svces','CHN') = 1293143.625 ;
evos('SkLab','a_Svces','LatinAmer') = 596355.25 ;
evos('SkLab','a_Svces','ROW') = 3143764.75 ;
evos('Capital','a_Agri','USA') = 88377.9375 ;
evos('Capital','a_Agri','EU_28') = 100920.5312 ;
evos('Capital','a_Agri','CHN') = 227665.375 ;
evos('Capital','a_Agri','LatinAmer') = 79138.69531 ;
evos('Capital','a_Agri','ROW') = 351941.0312 ;
evos('Capital','a_Food','USA') = 127894.0469 ;
evos('Capital','a_Food','EU_28') = 176723.0469 ;
evos('Capital','a_Food','CHN') = 164582.0156 ;
evos('Capital','a_Food','LatinAmer') = 137954.0625 ;
evos('Capital','a_Food','ROW') = 419185.5312 ;
evos('Capital','a_Energy','USA') = 211992.2344 ;
evos('Capital','a_Energy','EU_28') = 155744.4219 ;
evos('Capital','a_Energy','CHN') = 166394.8125 ;
evos('Capital','a_Energy','LatinAmer') = 149677.9531 ;
evos('Capital','a_Energy','ROW') = 1065776.5 ;
evos('Capital','a_Manuf','USA') = 842239.25 ;
evos('Capital','a_Manuf','EU_28') = 997180.1875 ;
evos('Capital','a_Manuf','CHN') = 1127318.375 ;
evos('Capital','a_Manuf','LatinAmer') = 369155.375 ;
evos('Capital','a_Manuf','ROW') = 1890256.5 ;
evos('Capital','a_Svces','USA') = 5677908 ;
evos('Capital','a_Svces','EU_28') = 5227559 ;
evos('Capital','a_Svces','CHN') = 2363882.25 ;
evos('Capital','a_Svces','LatinAmer') = 1702158.5 ;
evos('Capital','a_Svces','ROW') = 7521777.5 ;
evos('NatRes','a_Agri','USA') = 5071.396973 ;
evos('NatRes','a_Agri','EU_28') = 7891.441406 ;
evos('NatRes','a_Agri','CHN') = 39613.80469 ;
evos('NatRes','a_Agri','LatinAmer') = 6598.524414 ;
evos('NatRes','a_Agri','ROW') = 56598.68359 ;
evos('NatRes','a_Energy','USA') = 58615.78906 ;
evos('NatRes','a_Energy','EU_28') = 12277.47656 ;
evos('NatRes','a_Energy','CHN') = 47527.31641 ;
evos('NatRes','a_Energy','LatinAmer') = 40473.07422 ;
evos('NatRes','a_Energy','ROW') = 382323.1875 ;
evos('NatRes','a_Manuf','USA') = 14171.77539 ;
evos('NatRes','a_Manuf','EU_28') = 6557.898438 ;
evos('NatRes','a_Manuf','CHN') = 13355.99902 ;
evos('NatRes','a_Manuf','LatinAmer') = 8581.604492 ;
evos('NatRes','a_Manuf','ROW') = 37634.71484 ;

* vxsb data (125 cells)
vxsb('c_Agri','USA','USA') = 1.400000019e-05 ;
vxsb('c_Agri','USA','EU_28') = 6803.400879 ;
vxsb('c_Agri','USA','CHN') = 17922.62109 ;
vxsb('c_Agri','USA','LatinAmer') = 13557.67578 ;
vxsb('c_Agri','USA','ROW') = 37804.58203 ;
vxsb('c_Agri','EU_28','USA') = 2226.004395 ;
vxsb('c_Agri','EU_28','EU_28') = 100227.4297 ;
vxsb('c_Agri','EU_28','CHN') = 2349.161865 ;
vxsb('c_Agri','EU_28','LatinAmer') = 1016.438293 ;
vxsb('c_Agri','EU_28','ROW') = 21999.10352 ;
vxsb('c_Agri','CHN','USA') = 918.5802002 ;
vxsb('c_Agri','CHN','EU_28') = 1716.798096 ;
vxsb('c_Agri','CHN','CHN') = 1.400000019e-05 ;
vxsb('c_Agri','CHN','LatinAmer') = 379.8391418 ;
vxsb('c_Agri','CHN','ROW') = 16036.77734 ;
vxsb('c_Agri','LatinAmer','USA') = 28035.06055 ;
vxsb('c_Agri','LatinAmer','EU_28') = 17740.53516 ;
vxsb('c_Agri','LatinAmer','CHN') = 26747.97656 ;
vxsb('c_Agri','LatinAmer','LatinAmer') = 8855.363281 ;
vxsb('c_Agri','LatinAmer','ROW') = 26931.62695 ;
vxsb('c_Agri','ROW','USA') = 15995.86621 ;
vxsb('c_Agri','ROW','EU_28') = 37312.98438 ;
vxsb('c_Agri','ROW','CHN') = 29041.34375 ;
vxsb('c_Agri','ROW','LatinAmer') = 2985.500977 ;
vxsb('c_Agri','ROW','ROW') = 103801.0156 ;
vxsb('c_Food','USA','USA') = 7.99999998e-06 ;
vxsb('c_Food','USA','EU_28') = 6083.873047 ;
vxsb('c_Food','USA','CHN') = 4125.452148 ;
vxsb('c_Food','USA','LatinAmer') = 20857.46289 ;
vxsb('c_Food','USA','ROW') = 48405.03125 ;
vxsb('c_Food','EU_28','USA') = 23423.76367 ;
vxsb('c_Food','EU_28','EU_28') = 291054.4375 ;
vxsb('c_Food','EU_28','CHN') = 13131.53516 ;
vxsb('c_Food','EU_28','LatinAmer') = 6786.769531 ;
vxsb('c_Food','EU_28','ROW') = 83116.28125 ;
vxsb('c_Food','CHN','USA') = 7010.438477 ;
vxsb('c_Food','CHN','EU_28') = 6218.870117 ;
vxsb('c_Food','CHN','CHN') = 7.99999998e-06 ;
vxsb('c_Food','CHN','LatinAmer') = 2068.731689 ;
vxsb('c_Food','CHN','ROW') = 34390.14453 ;
vxsb('c_Food','LatinAmer','USA') = 25579.46094 ;
vxsb('c_Food','LatinAmer','EU_28') = 20379.01758 ;
vxsb('c_Food','LatinAmer','CHN') = 8027.956543 ;
vxsb('c_Food','LatinAmer','LatinAmer') = 24203.82812 ;
vxsb('c_Food','LatinAmer','ROW') = 49317.23438 ;
vxsb('c_Food','ROW','USA') = 47711.16797 ;
vxsb('c_Food','ROW','EU_28') = 49427.82422 ;
vxsb('c_Food','ROW','CHN') = 30606.88672 ;
vxsb('c_Food','ROW','LatinAmer') = 5192.499512 ;
vxsb('c_Food','ROW','ROW') = 200286.75 ;
vxsb('c_Energy','USA','USA') = 6.000000212e-06 ;
vxsb('c_Energy','USA','EU_28') = 17274.26758 ;
vxsb('c_Energy','USA','CHN') = 8890.428711 ;
vxsb('c_Energy','USA','LatinAmer') = 74734.97656 ;
vxsb('c_Energy','USA','ROW') = 43549.54297 ;
vxsb('c_Energy','EU_28','USA') = 9159.887695 ;
vxsb('c_Energy','EU_28','EU_28') = 93616.85156 ;
vxsb('c_Energy','EU_28','CHN') = 2122.032471 ;
vxsb('c_Energy','EU_28','LatinAmer') = 3219.932373 ;
vxsb('c_Energy','EU_28','ROW') = 47244.91797 ;
vxsb('c_Energy','CHN','USA') = 2249.4729 ;
vxsb('c_Energy','CHN','EU_28') = 1136.972412 ;
vxsb('c_Energy','CHN','CHN') = 6.000000212e-06 ;
vxsb('c_Energy','CHN','LatinAmer') = 943.8937988 ;
vxsb('c_Energy','CHN','ROW') = 26725.47656 ;
vxsb('c_Energy','LatinAmer','USA') = 41936.05859 ;
vxsb('c_Energy','LatinAmer','EU_28') = 11052.50391 ;
vxsb('c_Energy','LatinAmer','CHN') = 16630.09375 ;
vxsb('c_Energy','LatinAmer','LatinAmer') = 22378.64062 ;
vxsb('c_Energy','LatinAmer','ROW') = 18672.64258 ;
vxsb('c_Energy','ROW','USA') = 153356.75 ;
vxsb('c_Energy','ROW','EU_28') = 348086.1562 ;
vxsb('c_Energy','ROW','CHN') = 194340.1094 ;
vxsb('c_Energy','ROW','LatinAmer') = 11928.12793 ;
vxsb('c_Energy','ROW','ROW') = 657282.1875 ;
vxsb('c_Manuf','USA','USA') = 1.899999916e-05 ;
vxsb('c_Manuf','USA','EU_28') = 208028 ;
vxsb('c_Manuf','USA','CHN') = 93425.46875 ;
vxsb('c_Manuf','USA','LatinAmer') = 274506.5312 ;
vxsb('c_Manuf','USA','ROW') = 549923.3125 ;
vxsb('c_Manuf','EU_28','USA') = 370627.1562 ;
vxsb('c_Manuf','EU_28','EU_28') = 2658550.25 ;
vxsb('c_Manuf','EU_28','CHN') = 251185 ;
vxsb('c_Manuf','EU_28','LatinAmer') = 115615.5859 ;
vxsb('c_Manuf','EU_28','ROW') = 1035294.562 ;
vxsb('c_Manuf','CHN','USA') = 448971.625 ;
vxsb('c_Manuf','CHN','EU_28') = 335411.2188 ;
vxsb('c_Manuf','CHN','CHN') = 1.899999916e-05 ;
vxsb('c_Manuf','CHN','LatinAmer') = 131926.7344 ;
vxsb('c_Manuf','CHN','ROW') = 1130336.875 ;
vxsb('c_Manuf','LatinAmer','USA') = 338947.4375 ;
vxsb('c_Manuf','LatinAmer','EU_28') = 49987.93359 ;
vxsb('c_Manuf','LatinAmer','CHN') = 77448.10156 ;
vxsb('c_Manuf','LatinAmer','LatinAmer') = 95349.42188 ;
vxsb('c_Manuf','LatinAmer','ROW') = 95603.08594 ;
vxsb('c_Manuf','ROW','USA') = 708831.625 ;
vxsb('c_Manuf','ROW','EU_28') = 751006.1875 ;
vxsb('c_Manuf','ROW','CHN') = 1004213.25 ;
vxsb('c_Manuf','ROW','LatinAmer') = 131345.8906 ;
vxsb('c_Manuf','ROW','ROW') = 1816156.625 ;
vxsb('c_Svces','USA','USA') = 1.800000064e-05 ;
vxsb('c_Svces','USA','EU_28') = 223204.1875 ;
vxsb('c_Svces','USA','CHN') = 57874.48828 ;
vxsb('c_Svces','USA','LatinAmer') = 93008.17188 ;
vxsb('c_Svces','USA','ROW') = 329372.75 ;
vxsb('c_Svces','EU_28','USA') = 199294.9219 ;
vxsb('c_Svces','EU_28','EU_28') = 1038910.938 ;
vxsb('c_Svces','EU_28','CHN') = 52260.73047 ;
vxsb('c_Svces','EU_28','LatinAmer') = 54577.70703 ;
vxsb('c_Svces','EU_28','ROW') = 456310.2188 ;
vxsb('c_Svces','CHN','USA') = 19002.94922 ;
vxsb('c_Svces','CHN','EU_28') = 35278.92969 ;
vxsb('c_Svces','CHN','CHN') = 1.800000064e-05 ;
vxsb('c_Svces','CHN','LatinAmer') = 8103.09082 ;
vxsb('c_Svces','CHN','ROW') = 112547.0703 ;
vxsb('c_Svces','LatinAmer','USA') = 67931.73438 ;
vxsb('c_Svces','LatinAmer','EU_28') = 45743.12891 ;
vxsb('c_Svces','LatinAmer','CHN') = 11061.87012 ;
vxsb('c_Svces','LatinAmer','LatinAmer') = 20302.31445 ;
vxsb('c_Svces','LatinAmer','ROW') = 52412.22266 ;
vxsb('c_Svces','ROW','USA') = 260891.9531 ;
vxsb('c_Svces','ROW','EU_28') = 386634.6562 ;
vxsb('c_Svces','ROW','CHN') = 191478.3906 ;
vxsb('c_Svces','ROW','LatinAmer') = 46594.23047 ;
vxsb('c_Svces','ROW','ROW') = 572524.25 ;

* vfob data (125 cells)
vfob('c_Agri','USA','USA') = 1.400000019e-05 ;
vfob('c_Agri','USA','EU_28') = 6831.806152 ;
vfob('c_Agri','USA','CHN') = 18099.32812 ;
vfob('c_Agri','USA','LatinAmer') = 13593.30957 ;
vfob('c_Agri','USA','ROW') = 37890.47266 ;
vfob('c_Agri','EU_28','USA') = 2227.61084 ;
vfob('c_Agri','EU_28','EU_28') = 100227.4297 ;
vfob('c_Agri','EU_28','CHN') = 2347.754883 ;
vfob('c_Agri','EU_28','LatinAmer') = 1016.292908 ;
vfob('c_Agri','EU_28','ROW') = 21999.62109 ;
vfob('c_Agri','CHN','USA') = 917.3619995 ;
vfob('c_Agri','CHN','EU_28') = 1707.060547 ;
vfob('c_Agri','CHN','CHN') = 1.400000019e-05 ;
vfob('c_Agri','CHN','LatinAmer') = 379.7016602 ;
vfob('c_Agri','CHN','ROW') = 15989.5918 ;
vfob('c_Agri','LatinAmer','USA') = 28035.67383 ;
vfob('c_Agri','LatinAmer','EU_28') = 17802.70703 ;
vfob('c_Agri','LatinAmer','CHN') = 27357.92578 ;
vfob('c_Agri','LatinAmer','LatinAmer') = 8879.870117 ;
vfob('c_Agri','LatinAmer','ROW') = 27051.10352 ;
vfob('c_Agri','ROW','USA') = 15994.51074 ;
vfob('c_Agri','ROW','EU_28') = 37306.4375 ;
vfob('c_Agri','ROW','CHN') = 29119.19336 ;
vfob('c_Agri','ROW','LatinAmer') = 2985.793945 ;
vfob('c_Agri','ROW','ROW') = 103831.1875 ;
vfob('c_Food','USA','USA') = 7.99999998e-06 ;
vfob('c_Food','USA','EU_28') = 6083.873047 ;
vfob('c_Food','USA','CHN') = 4125.452148 ;
vfob('c_Food','USA','LatinAmer') = 20857.46289 ;
vfob('c_Food','USA','ROW') = 48405.03125 ;
vfob('c_Food','EU_28','USA') = 23423.76367 ;
vfob('c_Food','EU_28','EU_28') = 291054.4375 ;
vfob('c_Food','EU_28','CHN') = 13131.53516 ;
vfob('c_Food','EU_28','LatinAmer') = 6786.769531 ;
vfob('c_Food','EU_28','ROW') = 83116.28125 ;
vfob('c_Food','CHN','USA') = 7010.438477 ;
vfob('c_Food','CHN','EU_28') = 6218.870117 ;
vfob('c_Food','CHN','CHN') = 7.99999998e-06 ;
vfob('c_Food','CHN','LatinAmer') = 2068.731689 ;
vfob('c_Food','CHN','ROW') = 34390.14453 ;
vfob('c_Food','LatinAmer','USA') = 25579.46094 ;
vfob('c_Food','LatinAmer','EU_28') = 20379.01758 ;
vfob('c_Food','LatinAmer','CHN') = 8027.956543 ;
vfob('c_Food','LatinAmer','LatinAmer') = 24203.82812 ;
vfob('c_Food','LatinAmer','ROW') = 49317.23438 ;
vfob('c_Food','ROW','USA') = 47703.93359 ;
vfob('c_Food','ROW','EU_28') = 49416.89453 ;
vfob('c_Food','ROW','CHN') = 30606.32812 ;
vfob('c_Food','ROW','LatinAmer') = 5191.382324 ;
vfob('c_Food','ROW','ROW') = 200275.75 ;
vfob('c_Energy','USA','USA') = 6.000000212e-06 ;
vfob('c_Energy','USA','EU_28') = 18447.85938 ;
vfob('c_Energy','USA','CHN') = 9279.05957 ;
vfob('c_Energy','USA','LatinAmer') = 75504.69531 ;
vfob('c_Energy','USA','ROW') = 45237.80469 ;
vfob('c_Energy','EU_28','USA') = 9830.814453 ;
vfob('c_Energy','EU_28','EU_28') = 93616.85156 ;
vfob('c_Energy','EU_28','CHN') = 2175.595947 ;
vfob('c_Energy','EU_28','LatinAmer') = 3310.338135 ;
vfob('c_Energy','EU_28','ROW') = 49320.37109 ;
vfob('c_Energy','CHN','USA') = 2643.189697 ;
vfob('c_Energy','CHN','EU_28') = 1324.661499 ;
vfob('c_Energy','CHN','CHN') = 6.000000212e-06 ;
vfob('c_Energy','CHN','LatinAmer') = 1103.771362 ;
vfob('c_Energy','CHN','ROW') = 31133.92578 ;
vfob('c_Energy','LatinAmer','USA') = 42105.82812 ;
vfob('c_Energy','LatinAmer','EU_28') = 11081.37012 ;
vfob('c_Energy','LatinAmer','CHN') = 16655.21094 ;
vfob('c_Energy','LatinAmer','LatinAmer') = 23261.99219 ;
vfob('c_Energy','LatinAmer','ROW') = 18741.12109 ;
vfob('c_Energy','ROW','USA') = 154295.2969 ;
vfob('c_Energy','ROW','EU_28') = 351751.4375 ;
vfob('c_Energy','ROW','CHN') = 194884.8125 ;
vfob('c_Energy','ROW','LatinAmer') = 12005.6123 ;
vfob('c_Energy','ROW','ROW') = 663563.5 ;
vfob('c_Manuf','USA','USA') = 1.899999916e-05 ;
vfob('c_Manuf','USA','EU_28') = 218685.5781 ;
vfob('c_Manuf','USA','CHN') = 97459.71875 ;
vfob('c_Manuf','USA','LatinAmer') = 277893.125 ;
vfob('c_Manuf','USA','ROW') = 566183.125 ;
vfob('c_Manuf','EU_28','USA') = 370882.3438 ;
vfob('c_Manuf','EU_28','EU_28') = 2658550.25 ;
vfob('c_Manuf','EU_28','CHN') = 251305.0781 ;
vfob('c_Manuf','EU_28','LatinAmer') = 115692.1094 ;
vfob('c_Manuf','EU_28','ROW') = 1035924.938 ;
vfob('c_Manuf','CHN','USA') = 456204.0938 ;
vfob('c_Manuf','CHN','EU_28') = 341157.75 ;
vfob('c_Manuf','CHN','CHN') = 1.899999916e-05 ;
vfob('c_Manuf','CHN','LatinAmer') = 134580.8438 ;
vfob('c_Manuf','CHN','ROW') = 1149776.875 ;
vfob('c_Manuf','LatinAmer','USA') = 339853 ;
vfob('c_Manuf','LatinAmer','EU_28') = 50724.76562 ;
vfob('c_Manuf','LatinAmer','CHN') = 78322.91406 ;
vfob('c_Manuf','LatinAmer','LatinAmer') = 97245.02344 ;
vfob('c_Manuf','LatinAmer','ROW') = 96708.29688 ;
vfob('c_Manuf','ROW','USA') = 712613.5 ;
vfob('c_Manuf','ROW','EU_28') = 756368.4375 ;
vfob('c_Manuf','ROW','CHN') = 1006679.875 ;
vfob('c_Manuf','ROW','LatinAmer') = 132544.9062 ;
vfob('c_Manuf','ROW','ROW') = 1830184.125 ;
vfob('c_Svces','USA','USA') = 1.800000064e-05 ;
vfob('c_Svces','USA','EU_28') = 223204.1875 ;
vfob('c_Svces','USA','CHN') = 57874.48828 ;
vfob('c_Svces','USA','LatinAmer') = 93008.17188 ;
vfob('c_Svces','USA','ROW') = 329372.75 ;
vfob('c_Svces','EU_28','USA') = 199294.9219 ;
vfob('c_Svces','EU_28','EU_28') = 1038910.938 ;
vfob('c_Svces','EU_28','CHN') = 52260.73047 ;
vfob('c_Svces','EU_28','LatinAmer') = 54577.70703 ;
vfob('c_Svces','EU_28','ROW') = 456310.2188 ;
vfob('c_Svces','CHN','USA') = 19002.94922 ;
vfob('c_Svces','CHN','EU_28') = 35278.92969 ;
vfob('c_Svces','CHN','CHN') = 1.800000064e-05 ;
vfob('c_Svces','CHN','LatinAmer') = 8103.09082 ;
vfob('c_Svces','CHN','ROW') = 112547.0703 ;
vfob('c_Svces','LatinAmer','USA') = 67931.73438 ;
vfob('c_Svces','LatinAmer','EU_28') = 45743.12891 ;
vfob('c_Svces','LatinAmer','CHN') = 11061.87012 ;
vfob('c_Svces','LatinAmer','LatinAmer') = 20302.31445 ;
vfob('c_Svces','LatinAmer','ROW') = 52412.22266 ;
vfob('c_Svces','ROW','USA') = 260891.9531 ;
vfob('c_Svces','ROW','EU_28') = 386634.6562 ;
vfob('c_Svces','ROW','CHN') = 191478.3906 ;
vfob('c_Svces','ROW','LatinAmer') = 46594.23047 ;
vfob('c_Svces','ROW','ROW') = 572524.25 ;

* vcif data (125 cells)
vcif('c_Agri','USA','USA') = 1.400000019e-05 ;
vcif('c_Agri','USA','EU_28') = 7196.661621 ;
vcif('c_Agri','USA','CHN') = 19481.29102 ;
vcif('c_Agri','USA','LatinAmer') = 14300.01758 ;
vcif('c_Agri','USA','ROW') = 40709.66797 ;
vcif('c_Agri','EU_28','USA') = 2375.721924 ;
vcif('c_Agri','EU_28','EU_28') = 109918.0625 ;
vcif('c_Agri','EU_28','CHN') = 2493.200439 ;
vcif('c_Agri','EU_28','LatinAmer') = 1107.172119 ;
vcif('c_Agri','EU_28','ROW') = 25121.66016 ;
vcif('c_Agri','CHN','USA') = 982.8400269 ;
vcif('c_Agri','CHN','EU_28') = 1814.302734 ;
vcif('c_Agri','CHN','CHN') = 1.400000019e-05 ;
vcif('c_Agri','CHN','LatinAmer') = 415.4750671 ;
vcif('c_Agri','CHN','ROW') = 18042.25 ;
vcif('c_Agri','LatinAmer','USA') = 30376.96484 ;
vcif('c_Agri','LatinAmer','EU_28') = 19589.37305 ;
vcif('c_Agri','LatinAmer','CHN') = 29670.63672 ;
vcif('c_Agri','LatinAmer','LatinAmer') = 9745.605469 ;
vcif('c_Agri','LatinAmer','ROW') = 30306.62695 ;
vcif('c_Agri','ROW','USA') = 16579.83203 ;
vcif('c_Agri','ROW','EU_28') = 40282.87891 ;
vcif('c_Agri','ROW','CHN') = 32012.62109 ;
vcif('c_Agri','ROW','LatinAmer') = 3138.751953 ;
vcif('c_Agri','ROW','ROW') = 114163.5391 ;
vcif('c_Food','USA','USA') = 7.99999998e-06 ;
vcif('c_Food','USA','EU_28') = 6412.863281 ;
vcif('c_Food','USA','CHN') = 4330.11377 ;
vcif('c_Food','USA','LatinAmer') = 21618.25 ;
vcif('c_Food','USA','ROW') = 50443.95312 ;
vcif('c_Food','EU_28','USA') = 24805.07227 ;
vcif('c_Food','EU_28','EU_28') = 301250.5938 ;
vcif('c_Food','EU_28','CHN') = 13801.99902 ;
vcif('c_Food','EU_28','LatinAmer') = 7166.636719 ;
vcif('c_Food','EU_28','ROW') = 87406.90625 ;
vcif('c_Food','CHN','USA') = 7443.431641 ;
vcif('c_Food','CHN','EU_28') = 6587.441895 ;
vcif('c_Food','CHN','CHN') = 7.99999998e-06 ;
vcif('c_Food','CHN','LatinAmer') = 2177.897949 ;
vcif('c_Food','CHN','ROW') = 36460.65625 ;
vcif('c_Food','LatinAmer','USA') = 26547.10938 ;
vcif('c_Food','LatinAmer','EU_28') = 21518.36133 ;
vcif('c_Food','LatinAmer','CHN') = 8429.90918 ;
vcif('c_Food','LatinAmer','LatinAmer') = 25672.22266 ;
vcif('c_Food','LatinAmer','ROW') = 51841.26562 ;
vcif('c_Food','ROW','USA') = 49509.87891 ;
vcif('c_Food','ROW','EU_28') = 51903.83594 ;
vcif('c_Food','ROW','CHN') = 32339.88867 ;
vcif('c_Food','ROW','LatinAmer') = 5438.756348 ;
vcif('c_Food','ROW','ROW') = 211881.3594 ;
vcif('c_Energy','USA','USA') = 6.000000212e-06 ;
vcif('c_Energy','USA','EU_28') = 19536.76758 ;
vcif('c_Energy','USA','CHN') = 9697.918945 ;
vcif('c_Energy','USA','LatinAmer') = 78759.28906 ;
vcif('c_Energy','USA','ROW') = 47622.45703 ;
vcif('c_Energy','EU_28','USA') = 10228.87695 ;
vcif('c_Energy','EU_28','EU_28') = 96163.35156 ;
vcif('c_Energy','EU_28','CHN') = 2238.274902 ;
vcif('c_Energy','EU_28','LatinAmer') = 3441.856201 ;
vcif('c_Energy','EU_28','ROW') = 51050.39453 ;
vcif('c_Energy','CHN','USA') = 2754.911621 ;
vcif('c_Energy','CHN','EU_28') = 1383.091431 ;
vcif('c_Energy','CHN','CHN') = 6.000000212e-06 ;
vcif('c_Energy','CHN','LatinAmer') = 1172.695068 ;
vcif('c_Energy','CHN','ROW') = 32560.99414 ;
vcif('c_Energy','LatinAmer','USA') = 43424.90625 ;
vcif('c_Energy','LatinAmer','EU_28') = 11617.90723 ;
vcif('c_Energy','LatinAmer','CHN') = 17059.80664 ;
vcif('c_Energy','LatinAmer','LatinAmer') = 24170 ;
vcif('c_Energy','LatinAmer','ROW') = 19474.99023 ;
vcif('c_Energy','ROW','USA') = 159153.1719 ;
vcif('c_Energy','ROW','EU_28') = 365853.2188 ;
vcif('c_Energy','ROW','CHN') = 203539.2969 ;
vcif('c_Energy','ROW','LatinAmer') = 12625.31152 ;
vcif('c_Energy','ROW','ROW') = 696969.0625 ;
vcif('c_Manuf','USA','USA') = 1.899999916e-05 ;
vcif('c_Manuf','USA','EU_28') = 224312.6719 ;
vcif('c_Manuf','USA','CHN') = 100655.6094 ;
vcif('c_Manuf','USA','LatinAmer') = 283045.375 ;
vcif('c_Manuf','USA','ROW') = 577529.5625 ;
vcif('c_Manuf','EU_28','USA') = 380387.1875 ;
vcif('c_Manuf','EU_28','EU_28') = 2717041 ;
vcif('c_Manuf','EU_28','CHN') = 258918.2344 ;
vcif('c_Manuf','EU_28','LatinAmer') = 119666.1094 ;
vcif('c_Manuf','EU_28','ROW') = 1066434.5 ;
vcif('c_Manuf','CHN','USA') = 474099.875 ;
vcif('c_Manuf','CHN','EU_28') = 354684.4062 ;
vcif('c_Manuf','CHN','CHN') = 1.899999916e-05 ;
vcif('c_Manuf','CHN','LatinAmer') = 139964.4375 ;
vcif('c_Manuf','CHN','ROW') = 1195940.125 ;
vcif('c_Manuf','LatinAmer','USA') = 343970.7812 ;
vcif('c_Manuf','LatinAmer','EU_28') = 53201.09375 ;
vcif('c_Manuf','LatinAmer','CHN') = 84094.41406 ;
vcif('c_Manuf','LatinAmer','LatinAmer') = 101860.5234 ;
vcif('c_Manuf','LatinAmer','ROW') = 101383.0781 ;
vcif('c_Manuf','ROW','USA') = 730954.375 ;
vcif('c_Manuf','ROW','EU_28') = 779977.875 ;
vcif('c_Manuf','ROW','CHN') = 1042370.812 ;
vcif('c_Manuf','ROW','LatinAmer') = 137257.3438 ;
vcif('c_Manuf','ROW','ROW') = 1898909.375 ;
vcif('c_Svces','USA','USA') = 1.800000064e-05 ;
vcif('c_Svces','USA','EU_28') = 223204.1875 ;
vcif('c_Svces','USA','CHN') = 57874.48828 ;
vcif('c_Svces','USA','LatinAmer') = 93008.17188 ;
vcif('c_Svces','USA','ROW') = 329372.75 ;
vcif('c_Svces','EU_28','USA') = 199294.9219 ;
vcif('c_Svces','EU_28','EU_28') = 1038910.938 ;
vcif('c_Svces','EU_28','CHN') = 52260.73047 ;
vcif('c_Svces','EU_28','LatinAmer') = 54577.70703 ;
vcif('c_Svces','EU_28','ROW') = 456310.2188 ;
vcif('c_Svces','CHN','USA') = 19002.94922 ;
vcif('c_Svces','CHN','EU_28') = 35278.92969 ;
vcif('c_Svces','CHN','CHN') = 1.800000064e-05 ;
vcif('c_Svces','CHN','LatinAmer') = 8103.09082 ;
vcif('c_Svces','CHN','ROW') = 112547.0703 ;
vcif('c_Svces','LatinAmer','USA') = 67931.73438 ;
vcif('c_Svces','LatinAmer','EU_28') = 45743.12891 ;
vcif('c_Svces','LatinAmer','CHN') = 11061.87012 ;
vcif('c_Svces','LatinAmer','LatinAmer') = 20302.31445 ;
vcif('c_Svces','LatinAmer','ROW') = 52412.22266 ;
vcif('c_Svces','ROW','USA') = 260891.9531 ;
vcif('c_Svces','ROW','EU_28') = 386634.6562 ;
vcif('c_Svces','ROW','CHN') = 191478.3906 ;
vcif('c_Svces','ROW','LatinAmer') = 46594.23047 ;
vcif('c_Svces','ROW','ROW') = 572524.25 ;

* vmsb data (125 cells)
vmsb('c_Agri','USA','USA') = 1.400000019e-05 ;
vmsb('c_Agri','USA','EU_28') = 7311.947266 ;
vmsb('c_Agri','USA','CHN') = 20364.82617 ;
vmsb('c_Agri','USA','LatinAmer') = 14583.94238 ;
vmsb('c_Agri','USA','ROW') = 43543.80469 ;
vmsb('c_Agri','EU_28','USA') = 3062.273438 ;
vmsb('c_Agri','EU_28','EU_28') = 109918.0625 ;
vmsb('c_Agri','EU_28','CHN') = 2714.980469 ;
vmsb('c_Agri','EU_28','LatinAmer') = 1161.42334 ;
vmsb('c_Agri','EU_28','ROW') = 27932.82227 ;
vmsb('c_Agri','CHN','USA') = 993.5612183 ;
vmsb('c_Agri','CHN','EU_28') = 1908.62439 ;
vmsb('c_Agri','CHN','CHN') = 1.400000019e-05 ;
vmsb('c_Agri','CHN','LatinAmer') = 458.8178711 ;
vmsb('c_Agri','CHN','ROW') = 18933.44531 ;
vmsb('c_Agri','LatinAmer','USA') = 30421.97266 ;
vmsb('c_Agri','LatinAmer','EU_28') = 19965.36914 ;
vmsb('c_Agri','LatinAmer','CHN') = 30584.38086 ;
vmsb('c_Agri','LatinAmer','LatinAmer') = 9926.263672 ;
vmsb('c_Agri','LatinAmer','ROW') = 33311.59766 ;
vmsb('c_Agri','ROW','USA') = 16708.4707 ;
vmsb('c_Agri','ROW','EU_28') = 41002.89453 ;
vmsb('c_Agri','ROW','CHN') = 33786.54297 ;
vmsb('c_Agri','ROW','LatinAmer') = 3280.695068 ;
vmsb('c_Agri','ROW','ROW') = 123052.5 ;
vmsb('c_Food','USA','USA') = 7.99999998e-06 ;
vmsb('c_Food','USA','EU_28') = 6959.881836 ;
vmsb('c_Food','USA','CHN') = 4822.604492 ;
vmsb('c_Food','USA','LatinAmer') = 22427.0332 ;
vmsb('c_Food','USA','ROW') = 56150.58984 ;
vmsb('c_Food','EU_28','USA') = 25497.66211 ;
vmsb('c_Food','EU_28','EU_28') = 301250.5938 ;
vmsb('c_Food','EU_28','CHN') = 15559.08691 ;
vmsb('c_Food','EU_28','LatinAmer') = 7930.137207 ;
vmsb('c_Food','EU_28','ROW') = 100026.0156 ;
vmsb('c_Food','CHN','USA') = 7661.974121 ;
vmsb('c_Food','CHN','EU_28') = 7189.73877 ;
vmsb('c_Food','CHN','CHN') = 7.99999998e-06 ;
vmsb('c_Food','CHN','LatinAmer') = 2523.89209 ;
vmsb('c_Food','CHN','ROW') = 39409.08594 ;
vmsb('c_Food','LatinAmer','USA') = 27097.24219 ;
vmsb('c_Food','LatinAmer','EU_28') = 23245.85352 ;
vmsb('c_Food','LatinAmer','CHN') = 9309.023438 ;
vmsb('c_Food','LatinAmer','LatinAmer') = 26729.27539 ;
vmsb('c_Food','LatinAmer','ROW') = 58129.625 ;
vmsb('c_Food','ROW','USA') = 50427.32812 ;
vmsb('c_Food','ROW','EU_28') = 54344.89844 ;
vmsb('c_Food','ROW','CHN') = 34603.79297 ;
vmsb('c_Food','ROW','LatinAmer') = 6060.070801 ;
vmsb('c_Food','ROW','ROW') = 231768.1406 ;
vmsb('c_Energy','USA','USA') = 6.000000212e-06 ;
vmsb('c_Energy','USA','EU_28') = 19696.41406 ;
vmsb('c_Energy','USA','CHN') = 10015.68555 ;
vmsb('c_Energy','USA','LatinAmer') = 79587.35938 ;
vmsb('c_Energy','USA','ROW') = 48325.46484 ;
vmsb('c_Energy','EU_28','USA') = 10408.63867 ;
vmsb('c_Energy','EU_28','EU_28') = 96163.35156 ;
vmsb('c_Energy','EU_28','CHN') = 2284.990723 ;
vmsb('c_Energy','EU_28','LatinAmer') = 3474.208008 ;
vmsb('c_Energy','EU_28','ROW') = 52301.13281 ;
vmsb('c_Energy','CHN','USA') = 2793.549805 ;
vmsb('c_Energy','CHN','EU_28') = 1400.750732 ;
vmsb('c_Energy','CHN','CHN') = 6.000000212e-06 ;
vmsb('c_Energy','CHN','LatinAmer') = 1190.630737 ;
vmsb('c_Energy','CHN','ROW') = 33304.90234 ;
vmsb('c_Energy','LatinAmer','USA') = 43528.59375 ;
vmsb('c_Energy','LatinAmer','EU_28') = 11635.41992 ;
vmsb('c_Energy','LatinAmer','CHN') = 17088.125 ;
vmsb('c_Energy','LatinAmer','LatinAmer') = 24420.04492 ;
vmsb('c_Energy','LatinAmer','ROW') = 19617.41406 ;
vmsb('c_Energy','ROW','USA') = 159587.9688 ;
vmsb('c_Energy','ROW','EU_28') = 366708.2812 ;
vmsb('c_Energy','ROW','CHN') = 205200.75 ;
vmsb('c_Energy','ROW','LatinAmer') = 12745.60059 ;
vmsb('c_Energy','ROW','ROW') = 704190.5 ;
vmsb('c_Manuf','USA','USA') = 1.899999916e-05 ;
vmsb('c_Manuf','USA','EU_28') = 228749.8438 ;
vmsb('c_Manuf','USA','CHN') = 107682.3984 ;
vmsb('c_Manuf','USA','LatinAmer') = 289036.75 ;
vmsb('c_Manuf','USA','ROW') = 585280.4375 ;
vmsb('c_Manuf','EU_28','USA') = 385888.9688 ;
vmsb('c_Manuf','EU_28','EU_28') = 2717041 ;
vmsb('c_Manuf','EU_28','CHN') = 279350.875 ;
vmsb('c_Manuf','EU_28','LatinAmer') = 126009.4688 ;
vmsb('c_Manuf','EU_28','ROW') = 1094899.25 ;
vmsb('c_Manuf','CHN','USA') = 487624.75 ;
vmsb('c_Manuf','CHN','EU_28') = 368135.4062 ;
vmsb('c_Manuf','CHN','CHN') = 1.899999916e-05 ;
vmsb('c_Manuf','CHN','LatinAmer') = 151566 ;
vmsb('c_Manuf','CHN','ROW') = 1244436.25 ;
vmsb('c_Manuf','LatinAmer','USA') = 345057.4688 ;
vmsb('c_Manuf','LatinAmer','EU_28') = 53544.38672 ;
vmsb('c_Manuf','LatinAmer','CHN') = 85091.22656 ;
vmsb('c_Manuf','LatinAmer','LatinAmer') = 103770.7969 ;
vmsb('c_Manuf','LatinAmer','ROW') = 103119.9531 ;
vmsb('c_Manuf','ROW','USA') = 742175.25 ;
vmsb('c_Manuf','ROW','EU_28') = 789506.5625 ;
vmsb('c_Manuf','ROW','CHN') = 1070629.5 ;
vmsb('c_Manuf','ROW','LatinAmer') = 146303.7969 ;
vmsb('c_Manuf','ROW','ROW') = 1951682.375 ;
vmsb('c_Svces','USA','USA') = 1.800000064e-05 ;
vmsb('c_Svces','USA','EU_28') = 223204.1875 ;
vmsb('c_Svces','USA','CHN') = 57874.48828 ;
vmsb('c_Svces','USA','LatinAmer') = 93008.17188 ;
vmsb('c_Svces','USA','ROW') = 329372.75 ;
vmsb('c_Svces','EU_28','USA') = 199294.9219 ;
vmsb('c_Svces','EU_28','EU_28') = 1038910.938 ;
vmsb('c_Svces','EU_28','CHN') = 52260.73047 ;
vmsb('c_Svces','EU_28','LatinAmer') = 54577.70703 ;
vmsb('c_Svces','EU_28','ROW') = 456310.2188 ;
vmsb('c_Svces','CHN','USA') = 19002.94922 ;
vmsb('c_Svces','CHN','EU_28') = 35278.92969 ;
vmsb('c_Svces','CHN','CHN') = 1.800000064e-05 ;
vmsb('c_Svces','CHN','LatinAmer') = 8103.09082 ;
vmsb('c_Svces','CHN','ROW') = 112547.0703 ;
vmsb('c_Svces','LatinAmer','USA') = 67931.73438 ;
vmsb('c_Svces','LatinAmer','EU_28') = 45743.12891 ;
vmsb('c_Svces','LatinAmer','CHN') = 11061.87012 ;
vmsb('c_Svces','LatinAmer','LatinAmer') = 20302.31445 ;
vmsb('c_Svces','LatinAmer','ROW') = 52412.22266 ;
vmsb('c_Svces','ROW','USA') = 260891.9531 ;
vmsb('c_Svces','ROW','EU_28') = 386634.6562 ;
vmsb('c_Svces','ROW','CHN') = 191478.3906 ;
vmsb('c_Svces','ROW','LatinAmer') = 46594.23047 ;
vmsb('c_Svces','ROW','ROW') = 572524.25 ;

* vst data (5 cells)
vst('c_Svces','USA') = 31137.61328 ;
vst('c_Svces','EU_28') = 275190.5 ;
vst('c_Svces','CHN') = 24354.56445 ;
vst('c_Svces','LatinAmer') = 15379.7832 ;
vst('c_Svces','ROW') = 220126.7969 ;

* vtwr data (92 cells)
vtwr('c_Svces','c_Agri','USA','EU_28') = 364.8555908 ;
vtwr('c_Svces','c_Agri','USA','CHN') = 1381.962769 ;
vtwr('c_Svces','c_Agri','USA','LatinAmer') = 706.7055664 ;
vtwr('c_Svces','c_Agri','USA','ROW') = 2819.19165 ;
vtwr('c_Svces','c_Agri','EU_28','USA') = 148.1111145 ;
vtwr('c_Svces','c_Agri','EU_28','EU_28') = 9690.652344 ;
vtwr('c_Svces','c_Agri','EU_28','CHN') = 145.4454346 ;
vtwr('c_Svces','c_Agri','EU_28','LatinAmer') = 90.87866974 ;
vtwr('c_Svces','c_Agri','EU_28','ROW') = 3122.042969 ;
vtwr('c_Svces','c_Agri','CHN','USA') = 65.47801208 ;
vtwr('c_Svces','c_Agri','CHN','EU_28') = 107.2421722 ;
vtwr('c_Svces','c_Agri','CHN','LatinAmer') = 35.77340317 ;
vtwr('c_Svces','c_Agri','CHN','ROW') = 2052.659668 ;
vtwr('c_Svces','c_Agri','LatinAmer','USA') = 2341.29126 ;
vtwr('c_Svces','c_Agri','LatinAmer','EU_28') = 1786.670166 ;
vtwr('c_Svces','c_Agri','LatinAmer','CHN') = 2312.712891 ;
vtwr('c_Svces','c_Agri','LatinAmer','LatinAmer') = 865.7352905 ;
vtwr('c_Svces','c_Agri','LatinAmer','ROW') = 3255.517578 ;
vtwr('c_Svces','c_Agri','ROW','USA') = 585.3214111 ;
vtwr('c_Svces','c_Agri','ROW','EU_28') = 2976.439209 ;
vtwr('c_Svces','c_Agri','ROW','CHN') = 2893.425781 ;
vtwr('c_Svces','c_Agri','ROW','LatinAmer') = 152.9580841 ;
vtwr('c_Svces','c_Agri','ROW','ROW') = 10332.35254 ;
vtwr('c_Svces','c_Food','USA','EU_28') = 328.987793 ;
vtwr('c_Svces','c_Food','USA','CHN') = 204.661972 ;
vtwr('c_Svces','c_Food','USA','LatinAmer') = 760.7947388 ;
vtwr('c_Svces','c_Food','USA','ROW') = 2038.925049 ;
vtwr('c_Svces','c_Food','EU_28','USA') = 1381.308472 ;
vtwr('c_Svces','c_Food','EU_28','EU_28') = 10196.24316 ;
vtwr('c_Svces','c_Food','EU_28','CHN') = 670.4638672 ;
vtwr('c_Svces','c_Food','EU_28','LatinAmer') = 379.8668823 ;
vtwr('c_Svces','c_Food','EU_28','ROW') = 4290.624512 ;
vtwr('c_Svces','c_Food','CHN','USA') = 432.9931641 ;
vtwr('c_Svces','c_Food','CHN','EU_28') = 368.574646 ;
vtwr('c_Svces','c_Food','CHN','LatinAmer') = 109.1659851 ;
vtwr('c_Svces','c_Food','CHN','ROW') = 2070.508301 ;
vtwr('c_Svces','c_Food','LatinAmer','USA') = 967.6480103 ;
vtwr('c_Svces','c_Food','LatinAmer','EU_28') = 1139.344116 ;
vtwr('c_Svces','c_Food','LatinAmer','CHN') = 401.953125 ;
vtwr('c_Svces','c_Food','LatinAmer','LatinAmer') = 1468.392944 ;
vtwr('c_Svces','c_Food','LatinAmer','ROW') = 2524.035645 ;
vtwr('c_Svces','c_Food','ROW','USA') = 1805.944824 ;
vtwr('c_Svces','c_Food','ROW','EU_28') = 2486.954834 ;
vtwr('c_Svces','c_Food','ROW','CHN') = 1733.560791 ;
vtwr('c_Svces','c_Food','ROW','LatinAmer') = 247.3735199 ;
vtwr('c_Svces','c_Food','ROW','ROW') = 11605.60547 ;
vtwr('c_Svces','c_Energy','USA','EU_28') = 1088.909058 ;
vtwr('c_Svces','c_Energy','USA','CHN') = 418.859436 ;
vtwr('c_Svces','c_Energy','USA','LatinAmer') = 3254.592285 ;
vtwr('c_Svces','c_Energy','USA','ROW') = 2384.655029 ;
vtwr('c_Svces','c_Energy','EU_28','USA') = 398.0622864 ;
vtwr('c_Svces','c_Energy','EU_28','EU_28') = 2546.504639 ;
vtwr('c_Svces','c_Energy','EU_28','CHN') = 62.67894745 ;
vtwr('c_Svces','c_Energy','EU_28','LatinAmer') = 131.5179138 ;
vtwr('c_Svces','c_Energy','EU_28','ROW') = 1730.022095 ;
vtwr('c_Svces','c_Energy','CHN','USA') = 111.7216949 ;
vtwr('c_Svces','c_Energy','CHN','EU_28') = 58.42987823 ;
vtwr('c_Svces','c_Energy','CHN','LatinAmer') = 68.92368317 ;
vtwr('c_Svces','c_Energy','CHN','ROW') = 1427.068726 ;
vtwr('c_Svces','c_Energy','LatinAmer','USA') = 1319.079468 ;
vtwr('c_Svces','c_Energy','LatinAmer','EU_28') = 536.536499 ;
vtwr('c_Svces','c_Energy','LatinAmer','CHN') = 404.5961914 ;
vtwr('c_Svces','c_Energy','LatinAmer','LatinAmer') = 908.008667 ;
vtwr('c_Svces','c_Energy','LatinAmer','ROW') = 733.8682861 ;
vtwr('c_Svces','c_Energy','ROW','USA') = 4857.870117 ;
vtwr('c_Svces','c_Energy','ROW','EU_28') = 14101.78027 ;
vtwr('c_Svces','c_Energy','ROW','CHN') = 8654.493164 ;
vtwr('c_Svces','c_Energy','ROW','LatinAmer') = 619.699707 ;
vtwr('c_Svces','c_Energy','ROW','ROW') = 33405.5625 ;
vtwr('c_Svces','c_Manuf','USA','EU_28') = 5627.158691 ;
vtwr('c_Svces','c_Manuf','USA','CHN') = 3195.893555 ;
vtwr('c_Svces','c_Manuf','USA','LatinAmer') = 5152.268066 ;
vtwr('c_Svces','c_Manuf','USA','ROW') = 11346.4873 ;
vtwr('c_Svces','c_Manuf','EU_28','USA') = 9504.830078 ;
vtwr('c_Svces','c_Manuf','EU_28','EU_28') = 58491.35547 ;
vtwr('c_Svces','c_Manuf','EU_28','CHN') = 7613.164062 ;
vtwr('c_Svces','c_Manuf','EU_28','LatinAmer') = 3973.923096 ;
vtwr('c_Svces','c_Manuf','EU_28','ROW') = 30509.57227 ;
vtwr('c_Svces','c_Manuf','CHN','USA') = 17895.79688 ;
vtwr('c_Svces','c_Manuf','CHN','EU_28') = 13526.85449 ;
vtwr('c_Svces','c_Manuf','CHN','LatinAmer') = 5383.530273 ;
vtwr('c_Svces','c_Manuf','CHN','ROW') = 46163.25391 ;
vtwr('c_Svces','c_Manuf','LatinAmer','USA') = 4117.768555 ;
vtwr('c_Svces','c_Manuf','LatinAmer','EU_28') = 2476.327393 ;
vtwr('c_Svces','c_Manuf','LatinAmer','CHN') = 5771.499023 ;
vtwr('c_Svces','c_Manuf','LatinAmer','LatinAmer') = 4615.491699 ;
vtwr('c_Svces','c_Manuf','LatinAmer','ROW') = 4674.784668 ;
vtwr('c_Svces','c_Manuf','ROW','USA') = 18340.875 ;
vtwr('c_Svces','c_Manuf','ROW','EU_28') = 23609.78516 ;
vtwr('c_Svces','c_Manuf','ROW','CHN') = 35690.93359 ;
vtwr('c_Svces','c_Manuf','ROW','LatinAmer') = 4712.389648 ;
vtwr('c_Svces','c_Manuf','ROW','ROW') = 68725.1875 ;

* save data (5 cells)
save('USA') = 1580785.625 ;
save('EU_28') = 2133911.25 ;
save('CHN') = 3840168.5 ;
save('LatinAmer') = 474877.5 ;
save('ROW') = 4161104.25 ;

* vdep data (5 cells)
vdep('USA') = 1816268 ;
vdep('EU_28') = 1809763.25 ;
vdep('CHN') = 1607693.375 ;
vdep('LatinAmer') = 568090.375 ;
vdep('ROW') = 2746245.75 ;

* vkb data (5 cells)
vkb('USA') = 45406700 ;
vkb('EU_28') = 45244080 ;
vkb('CHN') = 40192336 ;
vkb('LatinAmer') = 14202259 ;
vkb('ROW') = 68656144 ;

* maks data (25 cells)
maks('c_Agri','a_Agri','USA') = 461880.9375 ;
maks('c_Agri','a_Agri','EU_28') = 526471.25 ;
maks('c_Agri','a_Agri','CHN') = 1454949.5 ;
maks('c_Agri','a_Agri','LatinAmer') = 519430.3125 ;
maks('c_Agri','a_Agri','ROW') = 2393759.5 ;
maks('c_Food','a_Food','USA') = 1117354.5 ;
maks('c_Food','a_Food','EU_28') = 1460662.5 ;
maks('c_Food','a_Food','CHN') = 1693269.875 ;
maks('c_Food','a_Food','LatinAmer') = 804895.3125 ;
maks('c_Food','a_Food','ROW') = 2526829.75 ;
maks('c_Energy','a_Energy','USA') = 1082570.75 ;
maks('c_Energy','a_Energy','EU_28') = 885865.4375 ;
maks('c_Energy','a_Energy','CHN') = 1266234 ;
maks('c_Energy','a_Energy','LatinAmer') = 538326 ;
maks('c_Energy','a_Energy','ROW') = 4193517.5 ;
maks('c_Manuf','a_Manuf','USA') = 5078237.5 ;
maks('c_Manuf','a_Manuf','EU_28') = 8101507.5 ;
maks('c_Manuf','a_Manuf','CHN') = 11994476 ;
maks('c_Manuf','a_Manuf','LatinAmer') = 2163178.25 ;
maks('c_Manuf','a_Manuf','ROW') = 12203364 ;
maks('c_Svces','a_Svces','USA') = 24328918 ;
maks('c_Svces','a_Svces','EU_28') = 22564282 ;
maks('c_Svces','a_Svces','CHN') = 14255377 ;
maks('c_Svces','a_Svces','LatinAmer') = 5760137.5 ;
maks('c_Svces','a_Svces','ROW') = 29442938 ;

* makb data (25 cells)
makb('c_Agri','a_Agri','USA') = 463691 ;
makb('c_Agri','a_Agri','EU_28') = 524524.9375 ;
makb('c_Agri','a_Agri','CHN') = 1451209.125 ;
makb('c_Agri','a_Agri','LatinAmer') = 519164.8438 ;
makb('c_Agri','a_Agri','ROW') = 2390844.5 ;
makb('c_Food','a_Food','USA') = 1156894.375 ;
makb('c_Food','a_Food','EU_28') = 1462588.625 ;
makb('c_Food','a_Food','CHN') = 1693453.875 ;
makb('c_Food','a_Food','LatinAmer') = 813330.1875 ;
makb('c_Food','a_Food','ROW') = 2563683.5 ;
makb('c_Energy','a_Energy','USA') = 1163954.125 ;
makb('c_Energy','a_Energy','EU_28') = 890106.5 ;
makb('c_Energy','a_Energy','CHN') = 1266388.875 ;
makb('c_Energy','a_Energy','LatinAmer') = 537509.875 ;
makb('c_Energy','a_Energy','ROW') = 4237631 ;
makb('c_Manuf','a_Manuf','USA') = 5173289.5 ;
makb('c_Manuf','a_Manuf','EU_28') = 8125949 ;
makb('c_Manuf','a_Manuf','CHN') = 11995360 ;
makb('c_Manuf','a_Manuf','LatinAmer') = 2182211.5 ;
makb('c_Manuf','a_Manuf','ROW') = 12263843 ;
makb('c_Svces','a_Svces','USA') = 25332292 ;
makb('c_Svces','a_Svces','EU_28') = 22761286 ;
makb('c_Svces','a_Svces','CHN') = 14256120 ;
makb('c_Svces','a_Svces','LatinAmer') = 5809796.5 ;
makb('c_Svces','a_Svces','ROW') = 29814150 ;

* esubt (empty in GDX — relying on $onImplicitAssign suppression)

* esubc (empty in GDX — relying on $onImplicitAssign suppression)

* esubva data (25 cells)
esubva('a_Agri','USA') = 0.2496392429 ;
esubva('a_Agri','EU_28') = 0.2496392429 ;
esubva('a_Agri','CHN') = 0.2496392429 ;
esubva('a_Agri','LatinAmer') = 0.2496392429 ;
esubva('a_Agri','ROW') = 0.2496392429 ;
esubva('a_Food','USA') = 1.120000005 ;
esubva('a_Food','EU_28') = 1.120000005 ;
esubva('a_Food','CHN') = 1.120000005 ;
esubva('a_Food','LatinAmer') = 1.120000005 ;
esubva('a_Food','ROW') = 1.120000005 ;
esubva('a_Energy','USA') = 0.7166335583 ;
esubva('a_Energy','EU_28') = 0.7166335583 ;
esubva('a_Energy','CHN') = 0.7166335583 ;
esubva('a_Energy','LatinAmer') = 0.7166335583 ;
esubva('a_Energy','ROW') = 0.7166335583 ;
esubva('a_Manuf','USA') = 1.198568106 ;
esubva('a_Manuf','EU_28') = 1.198568106 ;
esubva('a_Manuf','CHN') = 1.198568106 ;
esubva('a_Manuf','LatinAmer') = 1.198568106 ;
esubva('a_Manuf','ROW') = 1.198568106 ;
esubva('a_Svces','USA') = 1.37246573 ;
esubva('a_Svces','EU_28') = 1.37246573 ;
esubva('a_Svces','CHN') = 1.37246573 ;
esubva('a_Svces','LatinAmer') = 1.37246573 ;
esubva('a_Svces','ROW') = 1.37246573 ;

* etraq data (25 cells)
etraq('a_Agri','USA') = -5 ;
etraq('a_Agri','EU_28') = -5 ;
etraq('a_Agri','CHN') = -5 ;
etraq('a_Agri','LatinAmer') = -5 ;
etraq('a_Agri','ROW') = -5 ;
etraq('a_Food','USA') = -5 ;
etraq('a_Food','EU_28') = -5 ;
etraq('a_Food','CHN') = -5 ;
etraq('a_Food','LatinAmer') = -5 ;
etraq('a_Food','ROW') = -5 ;
etraq('a_Energy','USA') = -5 ;
etraq('a_Energy','EU_28') = -5 ;
etraq('a_Energy','CHN') = -5 ;
etraq('a_Energy','LatinAmer') = -5 ;
etraq('a_Energy','ROW') = -5 ;
etraq('a_Manuf','USA') = -5 ;
etraq('a_Manuf','EU_28') = -5 ;
etraq('a_Manuf','CHN') = -5 ;
etraq('a_Manuf','LatinAmer') = -5 ;
etraq('a_Manuf','ROW') = -5 ;
etraq('a_Svces','USA') = -5 ;
etraq('a_Svces','EU_28') = -5 ;
etraq('a_Svces','CHN') = -5 ;
etraq('a_Svces','LatinAmer') = -5 ;
etraq('a_Svces','ROW') = -5 ;

* esubq (empty in GDX — relying on $onImplicitAssign suppression)

* incpar data (25 cells)
incpar('c_Agri','USA') = 0.1501642466 ;
incpar('c_Agri','EU_28') = 0.1851723045 ;
incpar('c_Agri','CHN') = 0.2815423608 ;
incpar('c_Agri','LatinAmer') = 0.3134820461 ;
incpar('c_Agri','ROW') = 0.4802162945 ;
incpar('c_Food','USA') = 0.4096221626 ;
incpar('c_Food','EU_28') = 0.4636764228 ;
incpar('c_Food','CHN') = 0.3770502806 ;
incpar('c_Food','LatinAmer') = 0.4360114634 ;
incpar('c_Food','ROW') = 0.4848128557 ;
incpar('c_Energy','USA') = 0.9133253694 ;
incpar('c_Energy','EU_28') = 0.9933156967 ;
incpar('c_Energy','CHN') = 1.022505641 ;
incpar('c_Energy','LatinAmer') = 0.9881127477 ;
incpar('c_Energy','ROW') = 0.9896709919 ;
incpar('c_Manuf','USA') = 0.8878267407 ;
incpar('c_Manuf','EU_28') = 0.9007419348 ;
incpar('c_Manuf','CHN') = 0.8987540007 ;
incpar('c_Manuf','LatinAmer') = 0.9234779477 ;
incpar('c_Manuf','ROW') = 0.8737057447 ;
incpar('c_Svces','USA') = 1.061069846 ;
incpar('c_Svces','EU_28') = 1.124576092 ;
incpar('c_Svces','CHN') = 1.366120815 ;
incpar('c_Svces','LatinAmer') = 1.172920465 ;
incpar('c_Svces','ROW') = 1.185697436 ;

* subpar data (25 cells)
subpar('c_Agri','USA') = 0.753040731 ;
subpar('c_Agri','EU_28') = 0.8101371527 ;
subpar('c_Agri','CHN') = 0.7794262767 ;
subpar('c_Agri','LatinAmer') = 0.8008343577 ;
subpar('c_Agri','ROW') = 0.8159159422 ;
subpar('c_Food','USA') = 0.197535947 ;
subpar('c_Food','EU_28') = 0.3986287713 ;
subpar('c_Food','CHN') = 0.721752584 ;
subpar('c_Food','LatinAmer') = 0.7026839256 ;
subpar('c_Food','ROW') = 0.6504235268 ;
subpar('c_Energy','USA') = 0.09387533367 ;
subpar('c_Energy','EU_28') = 0.2317585796 ;
subpar('c_Energy','CHN') = 0.4935285449 ;
subpar('c_Energy','LatinAmer') = 0.5036433339 ;
subpar('c_Energy','ROW') = 0.4377360046 ;
subpar('c_Manuf','USA') = 0.1002229825 ;
subpar('c_Manuf','EU_28') = 0.2526686192 ;
subpar('c_Manuf','CHN') = 0.5345491171 ;
subpar('c_Manuf','LatinAmer') = 0.5261639953 ;
subpar('c_Manuf','ROW') = 0.4590882659 ;
subpar('c_Svces','USA') = 0.08792939782 ;
subpar('c_Svces','EU_28') = 0.2063865215 ;
subpar('c_Svces','CHN') = 0.4421200156 ;
subpar('c_Svces','LatinAmer') = 0.4599919021 ;
subpar('c_Svces','ROW') = 0.3376946151 ;

* esubg data (5 cells)
esubg('USA') = 1 ;
esubg('EU_28') = 1 ;
esubg('CHN') = 1 ;
esubg('LatinAmer') = 1 ;
esubg('ROW') = 1 ;

* esubi (empty in GDX — relying on $onImplicitAssign suppression)

* esubd data (25 cells)
esubd('c_Agri','USA') = 2.345425129 ;
esubd('c_Agri','EU_28') = 2.345425129 ;
esubd('c_Agri','CHN') = 2.345425129 ;
esubd('c_Agri','LatinAmer') = 2.345425129 ;
esubd('c_Agri','ROW') = 2.345425129 ;
esubd('c_Food','USA') = 2.483478308 ;
esubd('c_Food','EU_28') = 2.483478308 ;
esubd('c_Food','CHN') = 2.483478308 ;
esubd('c_Food','LatinAmer') = 2.483478308 ;
esubd('c_Food','ROW') = 2.483478308 ;
esubd('c_Energy','USA') = 3.818507195 ;
esubd('c_Energy','EU_28') = 3.818507195 ;
esubd('c_Energy','CHN') = 3.818507195 ;
esubd('c_Energy','LatinAmer') = 3.818507195 ;
esubd('c_Energy','ROW') = 3.818507195 ;
esubd('c_Manuf','USA') = 3.520190239 ;
esubd('c_Manuf','EU_28') = 3.520190239 ;
esubd('c_Manuf','CHN') = 3.520190239 ;
esubd('c_Manuf','LatinAmer') = 3.520190239 ;
esubd('c_Manuf','ROW') = 3.520190239 ;
esubd('c_Svces','USA') = 1.909135103 ;
esubd('c_Svces','EU_28') = 1.909135103 ;
esubd('c_Svces','CHN') = 1.909135103 ;
esubd('c_Svces','LatinAmer') = 1.909135103 ;
esubd('c_Svces','ROW') = 1.909135103 ;

* esubm data (25 cells)
esubm('c_Agri','USA') = 4.643953323 ;
esubm('c_Agri','EU_28') = 4.643953323 ;
esubm('c_Agri','CHN') = 4.643953323 ;
esubm('c_Agri','LatinAmer') = 4.643953323 ;
esubm('c_Agri','ROW') = 4.643953323 ;
esubm('c_Food','USA') = 4.823147297 ;
esubm('c_Food','EU_28') = 4.823147297 ;
esubm('c_Food','CHN') = 4.823147297 ;
esubm('c_Food','LatinAmer') = 4.823147297 ;
esubm('c_Food','ROW') = 4.823147297 ;
esubm('c_Energy','USA') = 11.49682426 ;
esubm('c_Energy','EU_28') = 11.49682426 ;
esubm('c_Energy','CHN') = 11.49682426 ;
esubm('c_Energy','LatinAmer') = 11.49682426 ;
esubm('c_Energy','ROW') = 11.49682426 ;
esubm('c_Manuf','USA') = 7.318256378 ;
esubm('c_Manuf','EU_28') = 7.318256378 ;
esubm('c_Manuf','CHN') = 7.318256378 ;
esubm('c_Manuf','LatinAmer') = 7.318256378 ;
esubm('c_Manuf','ROW') = 7.318256378 ;
esubm('c_Svces','USA') = 3.800415277 ;
esubm('c_Svces','EU_28') = 3.800415277 ;
esubm('c_Svces','CHN') = 3.800415277 ;
esubm('c_Svces','LatinAmer') = 3.800415277 ;
esubm('c_Svces','ROW') = 3.800415277 ;

* esubs data (1 cells)
esubs('c_Svces') = 1 ;

* etrae data (10 cells)
etrae('Land','USA') = -1 ;
etrae('Land','EU_28') = -1 ;
etrae('Land','CHN') = -1 ;
etrae('Land','LatinAmer') = -1 ;
etrae('Land','ROW') = -1 ;
etrae('NatRes','USA') = -0.001000000047 ;
etrae('NatRes','EU_28') = -0.001000000047 ;
etrae('NatRes','CHN') = -0.001000000047 ;
etrae('NatRes','LatinAmer') = -0.001000000047 ;
etrae('NatRes','ROW') = -0.001000000047 ;

* rorFlex0 data (5 cells)
rorFlex0('USA') = 10 ;
rorFlex0('EU_28') = 10 ;
rorFlex0('CHN') = 10 ;
rorFlex0('LatinAmer') = 10 ;
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
   USA, EU_28, CHN, LatinAmer, ROW
/ ;

set imuv(i) "IMUV commodities" /
   c_Agri, c_Food, c_Energy, c_Manuf, c_Svces
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
