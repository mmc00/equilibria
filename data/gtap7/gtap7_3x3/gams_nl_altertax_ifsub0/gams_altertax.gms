$setGlobal simType   CompStat
$setGlobal simName   COMP
$setGlobal baseName  9x10
$setGlobal inDir     .
$setGlobal outDir    .
$setGlobal utility   cde
$setGlobal savfFlag  capFix
$setGlobal ifCal     0
$setGlobal ifSUB     0
$setGlobal ifCSV     0
$setGlobal ifMCP     0
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
$setGlobal ifSUB     0
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


* getData.gms REPLACEMENT (aggregated subset) -- from v7_consolidated.gdx
sets
   acts           "Activities"      / a_Food, a_Mnfcs, a_Svces /
   comm           "Commodities"     / c_Food, c_Mnfcs, c_Svces /
   marg(comm)     "Margin comm"     / c_Svces /
   reg            "Regions"         / USA, EU_28, ROW /
   endw           "Endowments"      / Land, UnSkLab, SkLab, Capital, NatRes /
   endwf(endw)    "Fixed factors"   / NatRes /
   endwm(endw)    "Mobile factors"  / UnSkLab, SkLab, Capital /
   endws(endw)    "Sluggish factors"/ Land /
;

set stdlab "Standard SAM labels" /
   TRD, hhd, gov, inv, deprY, tmg, itax, ptax, mtax, etax, vtax, vsub, dtax, ctax, bop, tot
/ ;

set findem(stdlab) "Final demand accounts" / hhd, gov, inv, tmg / ;

set is "SAM accounts" /
   a_Food, a_Mnfcs, a_Svces, c_Food, c_Mnfcs, c_Svces, Land, UnSkLab, SkLab, Capital, NatRes, TRD, hhd, gov, inv, deprY, tmg, itax, ptax, mtax, etax,
   vtax, vsub, dtax, ctax, bop, tot, USA, EU_28, ROW
/ ;
alias(is, js) ;

set aa(is) "Armington agents" / a_Food, a_Mnfcs, a_Svces, hhd, gov, inv, tmg / ;
set a(aa)  "Activities"       / a_Food, a_Mnfcs, a_Svces / ;
set i(is)  "Commodities"      / c_Food, c_Mnfcs, c_Svces / ;
alias(i, j) ;

set r(is)  "Regions" / USA, EU_28, ROW / ;
alias(r,s) ; alias(r,d) ; alias(r,rp) ;

set fp(is) "Factors"          / Land, UnSkLab, SkLab, Capital, NatRes / ;
set fnm(fp) "Non-mobile"      / NatRes / ;
set fm(fp) "Mobile factors" ;
fm(fp)$(not fnm(fp)) = yes ;

set fd(aa) "Domestic final demand agents" / hhd, gov, inv, tmg / ;
set h(fd)  "Households"   / hhd / ;
set gov(fd) "Government"  / gov / ;
set inv(fd) "Investment"  / inv / ;
set fdc(fd) "CES final demand" / gov, inv / ;
set tmg(fd) "Trade margins" / tmg / ;

alias(i0,i) ; alias(a0,a) ; alias(j0,i0) ; alias(m0,i0) ;
set mapa0(a,a0) ; set mapi0(i,i0) ;
mapa0(a,a0)$(sameas(a,a0)) = yes ;
mapi0(i,i0)$(sameas(i,i0)) = yes ;

$onImplicitAssign
parameter vdfb(*,*,*) ;
vdfb("c_Food","a_Food","USA") = 542355 ;
vdfb("c_Food","a_Food","EU_28") = 500638.0625 ;
vdfb("c_Food","a_Food","ROW") = 2901770.75 ;
vdfb("c_Food","a_Mnfcs","USA") = 41297.44921875 ;
vdfb("c_Food","a_Mnfcs","EU_28") = 48286.37890625 ;
vdfb("c_Food","a_Mnfcs","ROW") = 505785.125 ;
vdfb("c_Food","a_Svces","USA") = 177164.1875 ;
vdfb("c_Food","a_Svces","EU_28") = 243287.625 ;
vdfb("c_Food","a_Svces","ROW") = 1010040.5625 ;
vdfb("c_Mnfcs","a_Food","USA") = 137371.984375 ;
vdfb("c_Mnfcs","a_Food","EU_28") = 106197.921875 ;
vdfb("c_Mnfcs","a_Food","ROW") = 483896.09375 ;
vdfb("c_Mnfcs","a_Mnfcs","USA") = 1611896.25 ;
vdfb("c_Mnfcs","a_Mnfcs","EU_28") = 1839815.75 ;
vdfb("c_Mnfcs","a_Mnfcs","ROW") = 11147014 ;
vdfb("c_Mnfcs","a_Svces","USA") = 1283868.25 ;
vdfb("c_Mnfcs","a_Svces","EU_28") = 1113946.25 ;
vdfb("c_Mnfcs","a_Svces","ROW") = 5528694 ;
vdfb("c_Svces","a_Food","USA") = 331707.65625 ;
vdfb("c_Svces","a_Food","EU_28") = 404140.875 ;
vdfb("c_Svces","a_Food","ROW") = 1365169.875 ;
vdfb("c_Svces","a_Mnfcs","USA") = 1069950.375 ;
vdfb("c_Svces","a_Mnfcs","EU_28") = 1605810.625 ;
vdfb("c_Svces","a_Mnfcs","ROW") = 5065759.5 ;
vdfb("c_Svces","a_Svces","USA") = 7592817.5 ;
vdfb("c_Svces","a_Svces","EU_28") = 6877007.5 ;
vdfb("c_Svces","a_Svces","ROW") = 13860507 ;
parameter vdfp(*,*,*) ;
vdfp("c_Food","a_Food","USA") = 542169.5 ;
vdfp("c_Food","a_Food","EU_28") = 497018.5625 ;
vdfp("c_Food","a_Food","ROW") = 2896536.75 ;
vdfp("c_Food","a_Mnfcs","USA") = 41122.109375 ;
vdfp("c_Food","a_Mnfcs","EU_28") = 48691.21875 ;
vdfp("c_Food","a_Mnfcs","ROW") = 511651.59375 ;
vdfp("c_Food","a_Svces","USA") = 177681.828125 ;
vdfp("c_Food","a_Svces","EU_28") = 260224.734375 ;
vdfp("c_Food","a_Svces","ROW") = 1043322.6875 ;
vdfp("c_Mnfcs","a_Food","USA") = 139415.390625 ;
vdfp("c_Mnfcs","a_Food","EU_28") = 111128.078125 ;
vdfp("c_Mnfcs","a_Food","ROW") = 491771.15625 ;
vdfp("c_Mnfcs","a_Mnfcs","USA") = 1669532.125 ;
vdfp("c_Mnfcs","a_Mnfcs","EU_28") = 1874450.375 ;
vdfp("c_Mnfcs","a_Mnfcs","ROW") = 11450795 ;
vdfp("c_Mnfcs","a_Svces","USA") = 1348394.875 ;
vdfp("c_Mnfcs","a_Svces","EU_28") = 1253490 ;
vdfp("c_Mnfcs","a_Svces","ROW") = 5716079 ;
vdfp("c_Svces","a_Food","USA") = 331101.875 ;
vdfp("c_Svces","a_Food","EU_28") = 408120.375 ;
vdfp("c_Svces","a_Food","ROW") = 1357937 ;
vdfp("c_Svces","a_Mnfcs","USA") = 1086692.25 ;
vdfp("c_Svces","a_Mnfcs","EU_28") = 1651084.25 ;
vdfp("c_Svces","a_Mnfcs","ROW") = 5127382 ;
vdfp("c_Svces","a_Svces","USA") = 7656514 ;
vdfp("c_Svces","a_Svces","EU_28") = 7102183 ;
vdfp("c_Svces","a_Svces","ROW") = 14101464 ;
parameter vmfb(*,*,*) ;
vmfb("c_Food","a_Food","USA") = 44631.125 ;
vmfb("c_Food","a_Food","EU_28") = 188689.59375 ;
vmfb("c_Food","a_Food","ROW") = 335635.34375 ;
vmfb("c_Food","a_Mnfcs","USA") = 2144.67797851562 ;
vmfb("c_Food","a_Mnfcs","EU_28") = 26389.56640625 ;
vmfb("c_Food","a_Mnfcs","ROW") = 43227.74609375 ;
vmfb("c_Food","a_Svces","USA") = 23047.599609375 ;
vmfb("c_Food","a_Svces","EU_28") = 73303.75 ;
vmfb("c_Food","a_Svces","ROW") = 130346.7578125 ;
vmfb("c_Mnfcs","a_Food","USA") = 30397.654296875 ;
vmfb("c_Mnfcs","a_Food","EU_28") = 76841.015625 ;
vmfb("c_Mnfcs","a_Food","ROW") = 192434.203125 ;
vmfb("c_Mnfcs","a_Mnfcs","USA") = 664758.25 ;
vmfb("c_Mnfcs","a_Mnfcs","EU_28") = 2185452.5 ;
vmfb("c_Mnfcs","a_Mnfcs","ROW") = 3972290.5 ;
vmfb("c_Mnfcs","a_Svces","USA") = 370647.5 ;
vmfb("c_Mnfcs","a_Svces","EU_28") = 876487 ;
vmfb("c_Mnfcs","a_Svces","ROW") = 1688362.25 ;
vmfb("c_Svces","a_Food","USA") = 4509.24609375 ;
vmfb("c_Svces","a_Food","EU_28") = 47619.97265625 ;
vmfb("c_Svces","a_Food","ROW") = 44004.40625 ;
vmfb("c_Svces","a_Mnfcs","USA") = 23632.78125 ;
vmfb("c_Svces","a_Mnfcs","EU_28") = 250204.203125 ;
vmfb("c_Svces","a_Mnfcs","ROW") = 222973.78125 ;
vmfb("c_Svces","a_Svces","USA") = 225410.625 ;
vmfb("c_Svces","a_Svces","EU_28") = 1195691.375 ;
vmfb("c_Svces","a_Svces","ROW") = 921570.5 ;
parameter vmfp(*,*,*) ;
vmfp("c_Food","a_Food","USA") = 44721.359375 ;
vmfp("c_Food","a_Food","EU_28") = 190533.890625 ;
vmfp("c_Food","a_Food","ROW") = 345928.1875 ;
vmfp("c_Food","a_Mnfcs","USA") = 2142.17553710938 ;
vmfp("c_Food","a_Mnfcs","EU_28") = 27106.546875 ;
vmfp("c_Food","a_Mnfcs","ROW") = 45766.32421875 ;
vmfp("c_Food","a_Svces","USA") = 23125.109375 ;
vmfp("c_Food","a_Svces","EU_28") = 80344.7890625 ;
vmfp("c_Food","a_Svces","ROW") = 139387.96875 ;
vmfp("c_Mnfcs","a_Food","USA") = 31042.537109375 ;
vmfp("c_Mnfcs","a_Food","EU_28") = 79483.5234375 ;
vmfp("c_Mnfcs","a_Food","ROW") = 195033.15625 ;
vmfp("c_Mnfcs","a_Mnfcs","USA") = 688171 ;
vmfp("c_Mnfcs","a_Mnfcs","EU_28") = 2220816.5 ;
vmfp("c_Mnfcs","a_Mnfcs","ROW") = 4138977.25 ;
vmfp("c_Mnfcs","a_Svces","USA") = 395723.5 ;
vmfp("c_Mnfcs","a_Svces","EU_28") = 990513.75 ;
vmfp("c_Mnfcs","a_Svces","ROW") = 1768015.875 ;
vmfp("c_Svces","a_Food","USA") = 4503.52197265625 ;
vmfp("c_Svces","a_Food","EU_28") = 47888.4296875 ;
vmfp("c_Svces","a_Food","ROW") = 44102.89453125 ;
vmfp("c_Svces","a_Mnfcs","USA") = 23847.408203125 ;
vmfp("c_Svces","a_Mnfcs","EU_28") = 253778.390625 ;
vmfp("c_Svces","a_Mnfcs","ROW") = 227237.546875 ;
vmfp("c_Svces","a_Svces","USA") = 228811.734375 ;
vmfp("c_Svces","a_Svces","EU_28") = 1238011 ;
vmfp("c_Svces","a_Svces","ROW") = 943045.625 ;
parameter vdpb(*,*) ;
vdpb("c_Food","USA") = 702662.25 ;
vdpb("c_Food","EU_28") = 638599.5625 ;
vdpb("c_Food","ROW") = 4088654.5 ;
vdpb("c_Mnfcs","USA") = 792360.1875 ;
vdpb("c_Mnfcs","EU_28") = 433667 ;
vdpb("c_Mnfcs","ROW") = 2320420.75 ;
vdpb("c_Svces","USA") = 10661360 ;
vdpb("c_Svces","EU_28") = 6461904 ;
vdpb("c_Svces","ROW") = 13852313 ;
parameter vdpp(*,*) ;
vdpp("c_Food","USA") = 701175.125 ;
vdpp("c_Food","EU_28") = 823667 ;
vdpp("c_Food","ROW") = 4284743.5 ;
vdpp("c_Mnfcs","USA") = 811949.3125 ;
vdpp("c_Mnfcs","EU_28") = 637483.5625 ;
vdpp("c_Mnfcs","ROW") = 2518896.5 ;
vdpp("c_Svces","USA") = 10888380 ;
vdpp("c_Svces","EU_28") = 6805092.5 ;
vdpp("c_Svces","ROW") = 14267114 ;
parameter vmpb(*,*) ;
vmpb("c_Food","USA") = 92016.71875 ;
vmpb("c_Food","EU_28") = 282465.9375 ;
vmpb("c_Food","ROW") = 458404.0625 ;
vmpb("c_Mnfcs","USA") = 632188.8125 ;
vmpb("c_Mnfcs","EU_28") = 834982.125 ;
vmpb("c_Mnfcs","ROW") = 1313897 ;
vmpb("c_Svces","USA") = 179993.15625 ;
vmpb("c_Svces","EU_28") = 164734.515625 ;
vmpb("c_Svces","ROW") = 652396.625 ;
parameter vmpp(*,*) ;
vmpp("c_Food","USA") = 91885.0625 ;
vmpp("c_Food","EU_28") = 361205.875 ;
vmpp("c_Food","ROW") = 516846.125 ;
vmpp("c_Mnfcs","USA") = 654085.1875 ;
vmpp("c_Mnfcs","EU_28") = 1120764.375 ;
vmpp("c_Mnfcs","ROW") = 1467644.125 ;
vmpp("c_Svces","USA") = 185092.046875 ;
vmpp("c_Svces","EU_28") = 179237.203125 ;
vmpp("c_Svces","ROW") = 682840.25 ;
parameter vdgb(*,*) ;
vdgb("c_Food","USA") = 1023.76983642578 ;
vdgb("c_Food","EU_28") = 5274.97412109375 ;
vdgb("c_Food","ROW") = 12373.4541015625 ;
vdgb("c_Mnfcs","USA") = 405.889831542969 ;
vdgb("c_Mnfcs","EU_28") = 35854.4609375 ;
vdgb("c_Mnfcs","ROW") = 59685.85546875 ;
vdgb("c_Svces","USA") = 2543765 ;
vdgb("c_Svces","EU_28") = 3504409.5 ;
vdgb("c_Svces","ROW") = 6974498.5 ;
parameter vdgp(*,*) ;
vdgp("c_Food","USA") = 1023.76983642578 ;
vdgp("c_Food","EU_28") = 5454.49853515625 ;
vdgp("c_Food","ROW") = 12476.13671875 ;
vdgp("c_Mnfcs","USA") = 405.889831542969 ;
vdgp("c_Mnfcs","EU_28") = 40958.1875 ;
vdgp("c_Mnfcs","ROW") = 60157.6796875 ;
vdgp("c_Svces","USA") = 2729009.5 ;
vdgp("c_Svces","EU_28") = 3507268.75 ;
vdgp("c_Svces","ROW") = 7017338 ;
parameter vmgb(*,*) ;
vmgb("c_Food","USA") = 7.94238901138306 ;
vmgb("c_Food","EU_28") = 200.922103881836 ;
vmgb("c_Food","ROW") = 1954.703125 ;
vmgb("c_Mnfcs","USA") = 331.96826171875 ;
vmgb("c_Mnfcs","EU_28") = 68117.15625 ;
vmgb("c_Mnfcs","ROW") = 67079.359375 ;
vmgb("c_Svces","USA") = 19186.185546875 ;
vmgb("c_Svces","EU_28") = 7234.12353515625 ;
vmgb("c_Svces","ROW") = 101980.109375 ;
parameter vmgp(*,*) ;
vmgp("c_Food","USA") = 7.94238901138306 ;
vmgp("c_Food","EU_28") = 223.506286621094 ;
vmgp("c_Food","ROW") = 1977.49206542969 ;
vmgp("c_Mnfcs","USA") = 331.96826171875 ;
vmgp("c_Mnfcs","EU_28") = 76792.6875 ;
vmgp("c_Mnfcs","ROW") = 66996.59375 ;
vmgp("c_Svces","USA") = 19186.185546875 ;
vmgp("c_Svces","EU_28") = 9761.845703125 ;
vmgp("c_Svces","ROW") = 107714.3984375 ;
parameter vdib(*,*) ;
vdib("c_Food","USA") = 522.650085449219 ;
vdib("c_Food","EU_28") = 5696.13232421875 ;
vdib("c_Food","ROW") = 86141.7890625 ;
vdib("c_Mnfcs","USA") = 765932.4375 ;
vdib("c_Mnfcs","EU_28") = 473251.46875 ;
vdib("c_Mnfcs","ROW") = 2101775 ;
vdib("c_Svces","USA") = 2873172.5 ;
vdib("c_Svces","EU_28") = 2258155.25 ;
vdib("c_Svces","ROW") = 8890655 ;
parameter vdip(*,*) ;
vdip("c_Food","USA") = 522.650085449219 ;
vdip("c_Food","EU_28") = 5491.546875 ;
vdip("c_Food","ROW") = 86947.171875 ;
vdip("c_Mnfcs","USA") = 775809.125 ;
vdip("c_Mnfcs","EU_28") = 490787.6875 ;
vdip("c_Mnfcs","ROW") = 2222290 ;
vdip("c_Svces","USA") = 2692840.75 ;
vdip("c_Svces","EU_28") = 2438634.25 ;
vdip("c_Svces","ROW") = 9181054 ;
parameter vmib(*,*) ;
vmib("c_Food","USA") = 22.4080238342285 ;
vmib("c_Food","EU_28") = 2048.05322265625 ;
vmib("c_Food","ROW") = 9515.7265625 ;
vmib("c_Mnfcs","USA") = 473778.375 ;
vmib("c_Mnfcs","EU_28") = 582482.6875 ;
vmib("c_Mnfcs","ROW") = 1294673.75 ;
vmib("c_Svces","USA") = 99352.0078125 ;
vmib("c_Svces","EU_28") = 92506.1640625 ;
vmib("c_Svces","ROW") = 139370 ;
parameter vmip(*,*) ;
vmip("c_Food","USA") = 22.4080238342285 ;
vmip("c_Food","EU_28") = 2077.35009765625 ;
vmip("c_Food","ROW") = 9679.484375 ;
vmip("c_Mnfcs","USA") = 491439.5 ;
vmip("c_Mnfcs","EU_28") = 614703.125 ;
vmip("c_Mnfcs","ROW") = 1395857.75 ;
vmip("c_Svces","USA") = 87960.8984375 ;
vmip("c_Svces","EU_28") = 99512.546875 ;
vmip("c_Svces","ROW") = 143277.140625 ;
parameter evfb(*,*,*) ;
evfb("Land","a_Food","USA") = 41351.43359375 ;
evfb("Land","a_Food","EU_28") = 67458 ;
evfb("Land","a_Food","ROW") = 669254.25 ;
evfb("UnSkLab","a_Food","USA") = 119645.0625 ;
evfb("UnSkLab","a_Food","EU_28") = 142159.09375 ;
evfb("UnSkLab","a_Food","ROW") = 1519266.75 ;
evfb("UnSkLab","a_Mnfcs","USA") = 705015.5625 ;
evfb("UnSkLab","a_Mnfcs","EU_28") = 439850.90625 ;
evfb("UnSkLab","a_Mnfcs","ROW") = 2233035.75 ;
evfb("UnSkLab","a_Svces","USA") = 3322260.5 ;
evfb("UnSkLab","a_Svces","EU_28") = 1800742.625 ;
evfb("UnSkLab","a_Svces","ROW") = 7304310 ;
evfb("SkLab","a_Food","USA") = 61255.26171875 ;
evfb("SkLab","a_Food","EU_28") = 90423.3125 ;
evfb("SkLab","a_Food","ROW") = 205270.140625 ;
evfb("SkLab","a_Mnfcs","USA") = 233255.890625 ;
evfb("SkLab","a_Mnfcs","EU_28") = 477022.59375 ;
evfb("SkLab","a_Mnfcs","ROW") = 811459.3125 ;
evfb("SkLab","a_Svces","USA") = 3798211.75 ;
evfb("SkLab","a_Svces","EU_28") = 2533243.25 ;
evfb("SkLab","a_Svces","ROW") = 5772158 ;
evfb("Capital","a_Food","USA") = 225441.96875 ;
evfb("Capital","a_Food","EU_28") = 294794.6875 ;
evfb("Capital","a_Food","ROW") = 1497269 ;
evfb("Capital","a_Mnfcs","USA") = 968529.625 ;
evfb("Capital","a_Mnfcs","EU_28") = 1078327.5 ;
evfb("Capital","a_Mnfcs","ROW") = 4621116.5 ;
evfb("Capital","a_Svces","USA") = 6049054 ;
evfb("Capital","a_Svces","EU_28") = 5707220.5 ;
evfb("Capital","a_Svces","ROW") = 13170218 ;
evfb("NatRes","a_Food","USA") = 5286.4248046875 ;
evfb("NatRes","a_Food","EU_28") = 8407.166015625 ;
evfb("NatRes","a_Food","ROW") = 112438.2734375 ;
evfb("NatRes","a_Mnfcs","USA") = 75873.765625 ;
evfb("NatRes","a_Mnfcs","EU_28") = 20106.6875 ;
evfb("NatRes","a_Mnfcs","ROW") = 572052.9375 ;
parameter evfp(*,*,*) ;
evfp("Land","a_Food","USA") = 34003.03515625 ;
evfp("Land","a_Food","EU_28") = 45434.83203125 ;
evfp("Land","a_Food","ROW") = 640291.1875 ;
evfp("UnSkLab","a_Food","USA") = 141928.640625 ;
evfp("UnSkLab","a_Food","EU_28") = 188846.1875 ;
evfp("UnSkLab","a_Food","ROW") = 1595059.25 ;
evfp("UnSkLab","a_Mnfcs","USA") = 848535.1875 ;
evfp("UnSkLab","a_Mnfcs","EU_28") = 644717 ;
evfp("UnSkLab","a_Mnfcs","ROW") = 2434527.25 ;
evfp("UnSkLab","a_Svces","USA") = 3998571.25 ;
evfp("UnSkLab","a_Svces","EU_28") = 2567129 ;
evfp("UnSkLab","a_Svces","ROW") = 8089657 ;
evfp("SkLab","a_Food","USA") = 71701.9296875 ;
evfp("SkLab","a_Food","EU_28") = 123548.7578125 ;
evfp("SkLab","a_Food","ROW") = 223862.65625 ;
evfp("SkLab","a_Mnfcs","USA") = 280739.65625 ;
evfp("SkLab","a_Mnfcs","EU_28") = 687521.6875 ;
evfp("SkLab","a_Mnfcs","ROW") = 917795.4375 ;
evfp("SkLab","a_Svces","USA") = 4571411.5 ;
evfp("SkLab","a_Svces","EU_28") = 3625571.75 ;
evfp("SkLab","a_Svces","ROW") = 6484548 ;
evfp("Capital","a_Food","USA") = 233096.546875 ;
evfp("Capital","a_Food","EU_28") = 286427.125 ;
evfp("Capital","a_Food","ROW") = 1488855.375 ;
evfp("Capital","a_Mnfcs","USA") = 1017035.1875 ;
evfp("Capital","a_Mnfcs","EU_28") = 1108303.5 ;
evfp("Capital","a_Mnfcs","ROW") = 4693207 ;
evfp("Capital","a_Svces","USA") = 6352001 ;
evfp("Capital","a_Svces","EU_28") = 5897007 ;
evfp("Capital","a_Svces","ROW") = 13404936 ;
evfp("NatRes","a_Food","USA") = 5551.177734375 ;
evfp("NatRes","a_Food","EU_28") = 8703.9833984375 ;
evfp("NatRes","a_Food","ROW") = 113756.734375 ;
evfp("NatRes","a_Mnfcs","USA") = 79673.6484375 ;
evfp("NatRes","a_Mnfcs","EU_28") = 20711.26953125 ;
evfp("NatRes","a_Mnfcs","ROW") = 579752 ;
parameter evos(*,*,*) ;
evos("Land","a_Food","USA") = 39669.4453125 ;
evos("Land","a_Food","EU_28") = 63766.05859375 ;
evos("Land","a_Food","ROW") = 613819.8125 ;
evos("UnSkLab","a_Food","USA") = 90088.765625 ;
evos("UnSkLab","a_Food","EU_28") = 99131.8984375 ;
evos("UnSkLab","a_Food","ROW") = 1430582.5 ;
evos("UnSkLab","a_Mnfcs","USA") = 530853.375 ;
evos("UnSkLab","a_Mnfcs","EU_28") = 302901.28125 ;
evos("UnSkLab","a_Mnfcs","ROW") = 2075360.25 ;
evos("UnSkLab","a_Svces","USA") = 2501552 ;
evos("UnSkLab","a_Svces","EU_28") = 1275810.875 ;
evos("UnSkLab","a_Svces","ROW") = 6688394.5 ;
evos("SkLab","a_Food","USA") = 46123.1875 ;
evos("SkLab","a_Food","EU_28") = 62689.94921875 ;
evos("SkLab","a_Food","ROW") = 186440.140625 ;
evos("SkLab","a_Mnfcs","USA") = 175633.96875 ;
evos("SkLab","a_Mnfcs","EU_28") = 330767.46875 ;
evos("SkLab","a_Mnfcs","ROW") = 729076.625 ;
evos("SkLab","a_Svces","USA") = 2859927.75 ;
evos("SkLab","a_Svces","EU_28") = 1781857.5 ;
evos("SkLab","a_Svces","ROW") = 5148835.5 ;
evos("Capital","a_Food","USA") = 216271.984375 ;
evos("Capital","a_Food","EU_28") = 277643.59375 ;
evos("Capital","a_Food","ROW") = 1380466.75 ;
evos("Capital","a_Mnfcs","USA") = 929134.125 ;
evos("Capital","a_Mnfcs","EU_28") = 1016128.125 ;
evos("Capital","a_Mnfcs","ROW") = 4249840.5 ;
evos("Capital","a_Svces","USA") = 5803005.5 ;
evos("Capital","a_Svces","EU_28") = 5364355.5 ;
evos("Capital","a_Svces","ROW") = 12106557 ;
evos("NatRes","a_Food","USA") = 5071.39697265625 ;
evos("NatRes","a_Food","EU_28") = 7891.44140625 ;
evos("NatRes","a_Food","ROW") = 102811.015625 ;
evos("NatRes","a_Mnfcs","USA") = 72787.5625 ;
evos("NatRes","a_Mnfcs","EU_28") = 18835.375 ;
evos("NatRes","a_Mnfcs","ROW") = 529895.9375 ;
parameter vxsb(*,*,*) ;
vxsb("c_Food","USA","USA") = 2.2000000171829E-5 ;
vxsb("c_Food","USA","EU_28") = 12887.2734375 ;
vxsb("c_Food","USA","ROW") = 142672.828125 ;
vxsb("c_Food","EU_28","USA") = 25649.767578125 ;
vxsb("c_Food","EU_28","EU_28") = 391281.84375 ;
vxsb("c_Food","EU_28","ROW") = 128399.2890625 ;
vxsb("c_Food","ROW","USA") = 125250.5703125 ;
vxsb("c_Food","ROW","EU_28") = 132796.03125 ;
vxsb("c_Food","ROW","ROW") = 568873.5 ;
vxsb("c_Mnfcs","USA","USA") = 2.30000005103648E-5 ;
vxsb("c_Mnfcs","USA","EU_28") = 225169.796875 ;
vxsb("c_Mnfcs","USA","ROW") = 1043731.5625 ;
vxsb("c_Mnfcs","EU_28","USA") = 379717.875 ;
vxsb("c_Mnfcs","EU_28","EU_28") = 2729899.75 ;
vxsb("c_Mnfcs","EU_28","ROW") = 1449099.875 ;
vxsb("c_Mnfcs","ROW","USA") = 1689399.625 ;
vxsb("c_Mnfcs","ROW","EU_28") = 1490862.25 ;
vxsb("c_Mnfcs","ROW","ROW") = 5414330.5 ;
vxsb("c_Svces","USA","USA") = 1.99999994947575E-5 ;
vxsb("c_Svces","USA","EU_28") = 223336.65625 ;
vxsb("c_Svces","USA","ROW") = 481554.03125 ;
vxsb("c_Svces","EU_28","USA") = 199364.0625 ;
vxsb("c_Svces","EU_28","EU_28") = 1061178.25 ;
vxsb("c_Svces","EU_28","ROW") = 568730.8125 ;
vxsb("c_Svces","ROW","USA") = 352720.03125 ;
vxsb("c_Svces","ROW","EU_28") = 473475.46875 ;
vxsb("c_Svces","ROW","ROW") = 1031974 ;
parameter vfob(*,*,*) ;
vfob("c_Food","USA","USA") = 2.2000000171829E-5 ;
vfob("c_Food","USA","EU_28") = 12915.6787109375 ;
vfob("c_Food","USA","ROW") = 142971.0625 ;
vfob("c_Food","EU_28","USA") = 25651.373046875 ;
vfob("c_Food","EU_28","EU_28") = 391281.84375 ;
vfob("c_Food","EU_28","ROW") = 128398.2578125 ;
vfob("c_Food","ROW","USA") = 125241.3828125 ;
vfob("c_Food","ROW","EU_28") = 132830.984375 ;
vfob("c_Food","ROW","ROW") = 569675.6875 ;
vfob("c_Mnfcs","USA","USA") = 2.30000005103648E-5 ;
vfob("c_Mnfcs","USA","EU_28") = 237000.96875 ;
vfob("c_Mnfcs","USA","ROW") = 1070258.875 ;
vfob("c_Mnfcs","EU_28","USA") = 380644.03125 ;
vfob("c_Mnfcs","EU_28","EU_28") = 2729899.75 ;
vfob("c_Mnfcs","EU_28","ROW") = 1452146.25 ;
vfob("c_Mnfcs","ROW","USA") = 1702821.5 ;
vfob("c_Mnfcs","ROW","EU_28") = 1506589.625 ;
vfob("c_Mnfcs","ROW","ROW") = 5470442 ;
vfob("c_Svces","USA","USA") = 1.99999994947575E-5 ;
vfob("c_Svces","USA","EU_28") = 223336.65625 ;
vfob("c_Svces","USA","ROW") = 481554.03125 ;
vfob("c_Svces","EU_28","USA") = 199364.0625 ;
vfob("c_Svces","EU_28","EU_28") = 1061178.25 ;
vfob("c_Svces","EU_28","ROW") = 568730.8125 ;
vfob("c_Svces","ROW","USA") = 352720.03125 ;
vfob("c_Svces","ROW","EU_28") = 473475.46875 ;
vfob("c_Svces","ROW","ROW") = 1031974 ;
parameter vcif(*,*,*) ;
vcif("c_Food","USA","USA") = 2.2000000171829E-5 ;
vcif("c_Food","USA","EU_28") = 13609.5244140625 ;
vcif("c_Food","USA","ROW") = 150883.296875 ;
vcif("c_Food","EU_28","USA") = 27180.794921875 ;
vcif("c_Food","EU_28","EU_28") = 411168.65625 ;
vcif("c_Food","EU_28","ROW") = 137097.578125 ;
vcif("c_Food","ROW","USA") = 131440.0625 ;
vcif("c_Food","ROW","EU_28") = 141696.1875 ;
vcif("c_Food","ROW","ROW") = 611737.4375 ;
vcif("c_Mnfcs","USA","USA") = 2.30000005103648E-5 ;
vcif("c_Mnfcs","USA","EU_28") = 243716.96875 ;
vcif("c_Mnfcs","USA","ROW") = 1096011.625 ;
vcif("c_Mnfcs","EU_28","USA") = 390546.9375 ;
vcif("c_Mnfcs","EU_28","EU_28") = 2790937 ;
vcif("c_Mnfcs","EU_28","ROW") = 1496167.25 ;
vcif("c_Mnfcs","ROW","USA") = 1749464.625 ;
vcif("c_Mnfcs","ROW","EU_28") = 1560898.875 ;
vcif("c_Mnfcs","ROW","ROW") = 5692401.5 ;
vcif("c_Svces","USA","USA") = 1.99999994947575E-5 ;
vcif("c_Svces","USA","EU_28") = 223336.65625 ;
vcif("c_Svces","USA","ROW") = 481554.03125 ;
vcif("c_Svces","EU_28","USA") = 199364.0625 ;
vcif("c_Svces","EU_28","EU_28") = 1061178.25 ;
vcif("c_Svces","EU_28","ROW") = 568730.8125 ;
vcif("c_Svces","ROW","USA") = 352720.03125 ;
vcif("c_Svces","ROW","EU_28") = 473475.46875 ;
vcif("c_Svces","ROW","ROW") = 1031974 ;
parameter vmsb(*,*,*) ;
vmsb("c_Food","USA","USA") = 2.2000000171829E-5 ;
vmsb("c_Food","USA","EU_28") = 14271.8291015625 ;
vmsb("c_Food","USA","ROW") = 161892.796875 ;
vmsb("c_Food","EU_28","USA") = 28559.935546875 ;
vmsb("c_Food","EU_28","EU_28") = 411168.65625 ;
vmsb("c_Food","EU_28","ROW") = 155324.46875 ;
vmsb("c_Food","ROW","USA") = 133310.546875 ;
vmsb("c_Food","ROW","EU_28") = 147657.375 ;
vmsb("c_Food","ROW","ROW") = 661867.125 ;
vmsb("c_Mnfcs","USA","USA") = 2.30000005103648E-5 ;
vmsb("c_Mnfcs","USA","EU_28") = 248313.78125 ;
vmsb("c_Mnfcs","USA","ROW") = 1118628.375 ;
vmsb("c_Mnfcs","EU_28","USA") = 396228.46875 ;
vmsb("c_Mnfcs","EU_28","EU_28") = 2790937 ;
vmsb("c_Mnfcs","EU_28","ROW") = 1552737.75 ;
vmsb("c_Mnfcs","ROW","USA") = 1775874.125 ;
vmsb("c_Mnfcs","ROW","EU_28") = 1585112.125 ;
vmsb("c_Mnfcs","ROW","ROW") = 5857372 ;
vmsb("c_Svces","USA","USA") = 1.99999994947575E-5 ;
vmsb("c_Svces","USA","EU_28") = 223336.65625 ;
vmsb("c_Svces","USA","ROW") = 481555.09375 ;
vmsb("c_Svces","EU_28","USA") = 199364.0625 ;
vmsb("c_Svces","EU_28","EU_28") = 1061178.25 ;
vmsb("c_Svces","EU_28","ROW") = 568730.875 ;
vmsb("c_Svces","ROW","USA") = 352720.03125 ;
vmsb("c_Svces","ROW","EU_28") = 473475.46875 ;
vmsb("c_Svces","ROW","ROW") = 1032009.4375 ;
parameter vst(*,*) ;
vst("c_Svces","USA") = 31137.61328125 ;
vst("c_Svces","EU_28") = 275190.5 ;
vst("c_Svces","ROW") = 259861.15625 ;
parameter vtwr(*,*,*,*) ;
vtwr("c_Svces","c_Food","USA","EU_28") = 693.843383789062 ;
vtwr("c_Svces","c_Food","USA","ROW") = 7912.24169921875 ;
vtwr("c_Svces","c_Food","EU_28","USA") = 1529.41967773438 ;
vtwr("c_Svces","c_Food","EU_28","EU_28") = 19886.89453125 ;
vtwr("c_Svces","c_Food","EU_28","ROW") = 8699.322265625 ;
vtwr("c_Svces","c_Food","ROW","USA") = 6198.6767578125 ;
vtwr("c_Svces","c_Food","ROW","EU_28") = 8865.2255859375 ;
vtwr("c_Svces","c_Food","ROW","ROW") = 42061.73046875 ;
vtwr("c_Svces","c_Mnfcs","USA","EU_28") = 6716.0673828125 ;
vtwr("c_Svces","c_Mnfcs","USA","ROW") = 25752.755859375 ;
vtwr("c_Svces","c_Mnfcs","EU_28","USA") = 9902.892578125 ;
vtwr("c_Svces","c_Mnfcs","EU_28","EU_28") = 61037.859375 ;
vtwr("c_Svces","c_Mnfcs","EU_28","ROW") = 44020.87890625 ;
vtwr("c_Svces","c_Mnfcs","ROW","USA") = 46643.11328125 ;
vtwr("c_Svces","c_Mnfcs","ROW","EU_28") = 54309.71484375 ;
vtwr("c_Svces","c_Mnfcs","ROW","ROW") = 221959.296875 ;
parameter save(*) ;
save("USA") = 1580785.625 ;
save("EU_28") = 2133911.25 ;
save("ROW") = 8476150 ;
parameter vdep(*) ;
vdep("USA") = 1816268 ;
vdep("EU_28") = 1809763.25 ;
vdep("ROW") = 4922029.5 ;
parameter vkb(*) ;
vkb("USA") = 45406700 ;
vkb("EU_28") = 45244080 ;
vkb("ROW") = 123050744 ;
parameter pop0(*) ;
pop0("USA") = 325.147125244141 ;
pop0("EU_28") = 513.872253417969 ;
pop0("ROW") = 6674.8642578125 ;
parameter maks(*,*,*) ;
maks("c_Food","a_Food","USA") = 1579235.5 ;
maks("c_Food","a_Food","EU_28") = 1987133.75 ;
maks("c_Food","a_Food","ROW") = 9393134 ;
maks("c_Mnfcs","a_Mnfcs","USA") = 5737490.5 ;
maks("c_Mnfcs","a_Mnfcs","EU_28") = 8537181 ;
maks("c_Mnfcs","a_Mnfcs","ROW") = 30127092 ;
maks("c_Svces","a_Svces","USA") = 24752234 ;
maks("c_Svces","a_Svces","EU_28") = 23014476 ;
maks("c_Svces","a_Svces","ROW") = 51690456 ;
parameter makb(*,*,*) ;
makb("c_Food","a_Food","USA") = 1620585.375 ;
makb("c_Food","a_Food","EU_28") = 1987113.625 ;
makb("c_Food","a_Food","ROW") = 9431686 ;
makb("c_Mnfcs","a_Mnfcs","USA") = 5860736 ;
makb("c_Mnfcs","a_Mnfcs","EU_28") = 8561450 ;
makb("c_Mnfcs","a_Mnfcs","ROW") = 30236078 ;
makb("c_Svces","a_Svces","USA") = 25808800 ;
makb("c_Svces","a_Svces","EU_28") = 23215892 ;
makb("c_Svces","a_Svces","ROW") = 52126932 ;
parameter esubt(*,*) ;
parameter esubc(*,*) ;
parameter esubva(*,*) ;
esubva("a_Food","USA") = 0.590146839618683 ;
esubva("a_Food","EU_28") = 0.590146839618683 ;
esubva("a_Food","ROW") = 0.590146839618683 ;
esubva("a_Mnfcs","USA") = 1.07676565647125 ;
esubva("a_Mnfcs","EU_28") = 1.07676565647125 ;
esubva("a_Mnfcs","ROW") = 1.07676565647125 ;
esubva("a_Svces","USA") = 1.36973440647125 ;
esubva("a_Svces","EU_28") = 1.36973440647125 ;
esubva("a_Svces","ROW") = 1.36973440647125 ;
parameter etraq(*,*) ;
etraq("a_Food","USA") = -5 ;
etraq("a_Food","EU_28") = -5 ;
etraq("a_Food","ROW") = -5 ;
etraq("a_Mnfcs","USA") = -5 ;
etraq("a_Mnfcs","EU_28") = -5 ;
etraq("a_Mnfcs","ROW") = -5 ;
etraq("a_Svces","USA") = -5 ;
etraq("a_Svces","EU_28") = -5 ;
etraq("a_Svces","ROW") = -5 ;
parameter esubq(*,*) ;
parameter incpar(*,*) ;
incpar("c_Food","USA") = 0.380728989839554 ;
incpar("c_Food","EU_28") = 0.420596539974213 ;
incpar("c_Food","ROW") = 0.437571108341217 ;
incpar("c_Mnfcs","USA") = 0.884716153144836 ;
incpar("c_Mnfcs","EU_28") = 0.906521677970886 ;
incpar("c_Mnfcs","ROW") = 0.892096936702728 ;
incpar("c_Svces","USA") = 1.05961358547211 ;
incpar("c_Svces","EU_28") = 1.12182652950287 ;
incpar("c_Svces","ROW") = 1.20941364765167 ;
parameter subpar(*,*) ;
subpar("c_Food","USA") = 0.259396851062775 ;
subpar("c_Food","EU_28") = 0.462282210588455 ;
subpar("c_Food","ROW") = 0.716752648353577 ;
subpar("c_Mnfcs","USA") = 0.100168377161026 ;
subpar("c_Mnfcs","EU_28") = 0.251334369182587 ;
subpar("c_Mnfcs","ROW") = 0.484381228685379 ;
subpar("c_Svces","USA") = 0.0879457890987396 ;
subpar("c_Svces","EU_28") = 0.20687110722065 ;
subpar("c_Svces","ROW") = 0.379675805568695 ;
parameter esubg(*) ;
esubg("USA") = 1 ;
esubg("EU_28") = 1 ;
esubg("ROW") = 1 ;
parameter esubi(*) ;
parameter esubd(*,*) ;
esubd("c_Food","USA") = 2.42951130867004 ;
esubd("c_Food","EU_28") = 2.42951130867004 ;
esubd("c_Food","ROW") = 2.42951130867004 ;
esubd("c_Mnfcs","USA") = 3.62565517425537 ;
esubd("c_Mnfcs","EU_28") = 3.62565517425537 ;
esubd("c_Mnfcs","ROW") = 3.62565517425537 ;
esubd("c_Svces","USA") = 1.93763506412506 ;
esubd("c_Svces","EU_28") = 1.93763506412506 ;
esubd("c_Svces","ROW") = 1.93763506412506 ;
parameter esubm(*,*) ;
esubm("c_Food","USA") = 4.76437854766846 ;
esubm("c_Food","EU_28") = 4.76437854766846 ;
esubm("c_Food","ROW") = 4.76437854766846 ;
esubm("c_Mnfcs","USA") = 7.87347745895386 ;
esubm("c_Mnfcs","EU_28") = 7.87347745895386 ;
esubm("c_Mnfcs","ROW") = 7.87347745895386 ;
esubm("c_Svces","USA") = 3.82696843147278 ;
esubm("c_Svces","EU_28") = 3.82696843147278 ;
esubm("c_Svces","ROW") = 3.82696843147278 ;
parameter esubs(*) ;
esubs("c_Svces") = 1 ;
parameter etrae(*,*) ;
etrae("Land","USA") = -1 ;
etrae("Land","EU_28") = -1 ;
etrae("Land","ROW") = -1 ;
etrae("NatRes","USA") = -0.00100000004749745 ;
etrae("NatRes","EU_28") = -0.00100000004749745 ;
etrae("NatRes","ROW") = -0.00100000004749745 ;
parameter rorFlex0(*) ;
rorFlex0("USA") = 10 ;
rorFlex0("EU_28") = 10 ;
rorFlex0("ROW") = 10 ;

parameter esubd(*,*) ;
esubd("c_Food","USA") = 2.429511308670044 ;
esubd("c_Mnfcs","USA") = 3.625655174255371 ;
esubd("c_Svces","USA") = 1.937635064125061 ;
esubd("c_Food","EU_28") = 2.429511308670044 ;
esubd("c_Mnfcs","EU_28") = 3.625655174255371 ;
esubd("c_Svces","EU_28") = 1.937635064125061 ;
esubd("c_Food","ROW") = 2.429511308670044 ;
esubd("c_Mnfcs","ROW") = 3.625655174255371 ;
esubd("c_Svces","ROW") = 1.937635064125061 ;
parameter esubm(*,*) ;
esubm("c_Food","USA") = 4.764378547668457 ;
esubm("c_Mnfcs","USA") = 7.873477458953857 ;
esubm("c_Svces","USA") = 3.8269684314727783 ;
esubm("c_Food","EU_28") = 4.764378547668457 ;
esubm("c_Mnfcs","EU_28") = 7.873477458953857 ;
esubm("c_Svces","EU_28") = 3.8269684314727783 ;
esubm("c_Food","ROW") = 4.764378547668457 ;
esubm("c_Mnfcs","ROW") = 7.873477458953857 ;
esubm("c_Svces","ROW") = 3.8269684314727783 ;
parameter esubt(*,*) ;
parameter esubc(*,*) ;
parameter esubva(*,*) ;
esubva("a_Food","USA") = 0.5901468396186829 ;
esubva("a_Mnfcs","USA") = 1.0767656564712524 ;
esubva("a_Svces","USA") = 1.3697344064712524 ;
esubva("a_Food","EU_28") = 0.5901468396186829 ;
esubva("a_Mnfcs","EU_28") = 1.0767656564712524 ;
esubva("a_Svces","EU_28") = 1.3697344064712524 ;
esubva("a_Food","ROW") = 0.5901468396186829 ;
esubva("a_Mnfcs","ROW") = 1.0767656564712524 ;
esubva("a_Svces","ROW") = 1.3697344064712524 ;
parameter esubq(*,*) ;
parameter esubg(*) ;
esubg("USA") = 1.0 ;
esubg("EU_28") = 1.0 ;
esubg("ROW") = 1.0 ;
parameter esubi(*) ;
parameter esubs(*) ;
esubs("c_Svces") = 1.0 ;
parameter etraq(*,*) ;
etraq("a_Food","USA") = -5.0 ;
etraq("a_Mnfcs","USA") = -5.0 ;
etraq("a_Svces","USA") = -5.0 ;
etraq("a_Food","EU_28") = -5.0 ;
etraq("a_Mnfcs","EU_28") = -5.0 ;
etraq("a_Svces","EU_28") = -5.0 ;
etraq("a_Food","ROW") = -5.0 ;
etraq("a_Mnfcs","ROW") = -5.0 ;
etraq("a_Svces","ROW") = -5.0 ;
parameter etrae(*,*) ;
etrae("Land","USA") = -1.0 ;
etrae("NatRes","USA") = -0.0010000000474974513 ;
etrae("Land","EU_28") = -1.0 ;
etrae("NatRes","EU_28") = -0.0010000000474974513 ;
etrae("Land","ROW") = -1.0 ;
etrae("NatRes","ROW") = -0.0010000000474974513 ;
parameter incpar(*,*) ;
incpar("c_Food","USA") = 0.38072898983955383 ;
incpar("c_Mnfcs","USA") = 0.8847161531448364 ;
incpar("c_Svces","USA") = 1.059613585472107 ;
incpar("c_Food","EU_28") = 0.42059653997421265 ;
incpar("c_Mnfcs","EU_28") = 0.9065216779708862 ;
incpar("c_Svces","EU_28") = 1.1218265295028687 ;
incpar("c_Food","ROW") = 0.43757110834121704 ;
incpar("c_Mnfcs","ROW") = 0.8920969367027283 ;
incpar("c_Svces","ROW") = 1.2094136476516724 ;
parameter subpar(*,*) ;
subpar("c_Food","USA") = 0.25939685106277466 ;
subpar("c_Mnfcs","USA") = 0.100168377161026 ;
subpar("c_Svces","USA") = 0.08794578909873962 ;
subpar("c_Food","EU_28") = 0.4622822105884552 ;
subpar("c_Mnfcs","EU_28") = 0.25133436918258667 ;
subpar("c_Svces","EU_28") = 0.20687110722064972 ;
subpar("c_Food","ROW") = 0.7167526483535767 ;
subpar("c_Mnfcs","ROW") = 0.48438122868537903 ;
subpar("c_Svces","ROW") = 0.37967580556869507 ;
parameter rorFlex0(*) ;
rorFlex0("USA") = 10.0 ;
rorFlex0("EU_28") = 10.0 ;
rorFlex0("ROW") = 10.0 ;
parameter fbep(*,*,*) ; fbep(fp,a0,r) = 0 ;
parameter ftrv(*,*,*) ; ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;
parameter ptax(*,*,*) ; ptax(i0,a0,r) = makb(i0,a0,r) - maks(i0,a0,r) ;
parameter tvom(a0,r)  ; tvom(a0,r) = sum(i0, vdfp(i0,a0,r) + vmfp(i0,a0,r)) + sum(fp, evfp(fp,a0,r)) ;
parameter check(a0,r) ; check(a0,r) = 0 ;

parameters mdf(i0,a0,r), mmf(i0,a0,r), mdp(i0,r), mmp(i0,r),
   mdg(i0,r), mmg(i0,r), mdi(i0,r), mmi(i0,r) ;
mdf(i0,a0,r)=0; mmf(i0,a0,r)=0; mdp(i0,r)=0; mmp(i0,r)=0;
mdg(i0,r)=0; mmg(i0,r)=0; mdi(i0,r)=0; mmi(i0,r)=0;

parameters sigmap(r,a), sigmand(r,a), sigmav(r,a), omegas(r,a), sigmas(r,i),
   eh0(r,i), bh0(r,i), sigmag(r), sigmai(r), sigmam(r,i,aa), sigmaw(r,i),
   sigmamg(i), omegaf(r,fp), omegax(r,i), omegaw(r,i), etaf(r,fp),
   etaff(r,fp,a), mdtx0(r), RoRFlag ;
sigmap(r,a)=na; sigmand(r,a)=na; sigmav(r,a)=na;
omegas(r,a) = -etraq(a,r) ;
sigmas(r,i) = inf$(esubq(i,r) eq 0) + (1/esubq(i,r))$(esubq(i,r) ne 0) ;
eh0(r,i)=na; bh0(r,i)=na; sigmag(r)=na; sigmai(r)=na;
sigmam(r,i,aa)=na; sigmaw(r,i)=na; sigmamg(i)=na;
loop(fp$(not fnm(fp)), loop(endwm, omegaf(r,fp)$sameas(endwm,fp) = inf ; ) ;
                        loop(endws, omegaf(r,fp)$sameas(endws,fp) = na  ; ) ; ) ;
parameter rorFlex0(*) ;
parameter rorFlex(r,t) ; rorFlex(r,t) = rorFlex0(r) ;
omegax(r,i)=inf; omegaw(r,i)=inf; etaf(r,fp)=0; etaff(r,fp,a)=0; mdtx0(r)=na;
* END getData.gms replacement (aggregated subset)


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

set rres(r) "rres" /
   ROW
/ ;

set rmuv(r) "rmuv" /
   EU_28
/ ;

set imuv(i) "imuv" /
   c_Svces
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

* --- inlined: model.gms ---
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

* --- end: model.gms ---

*  Initialize the model

* --- inlined: cal.gms ---
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

* --- end: cal.gms ---


* ============================================================
* CONVERT-mode solve (ALTERTAX period): write gtap.nl — Layer 6 diagnostic
* Mirrors Malcolm (1998) altertax: CD elasticities (all subst. elast. = 1),
* mobile factors (omegaf=inf or large), Armington 0.95.
* Signatures from standard_gtap_7/cal.gms:
*   esubva(a,r), esubd(i,r), esubm(i,r), etrae(fp,r), omegaf(r,fp).
* Use NLP mode (ifMCP=0) — GAMS CONVERT cannot write .nl for MCP.
* ============================================================
rs(r) = yes ;
ts('base') = yes ;

* Apply altertax elasticity overrides (parameter_altertax.gms equivalent)
esubva(a,r)  = 1 ;
esubd(i,r)   = 0.95 ;
esubm(i,r)   = 0.95 ;
etrae(fp,r)  = 1 ;
omegaf(r,fp) = 1 ;

* Recalibrate derived elasticities from the overrides (mirrors cal.gms flow)
sigmav(r,a)$(va.l(r,a,'base')) = 1 ;
sigmam(r,i,aa)$(xa.l(r,i,aa,'base')) = 0.95 ;
sigmaw(r,i)$(xmt.l(r,i,'base')) = 0.95 ;

option limrow = 0, limcol = 0, solprint = off ;
option NLP = CONVERT;
gtap.optfile = 1 ;
solve gtap using nlp maximizing walras ;
