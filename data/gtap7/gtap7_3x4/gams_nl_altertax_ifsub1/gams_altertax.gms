$setGlobal simType   CompStat
$setGlobal simName   COMP
$setGlobal baseName  9x10
$setGlobal inDir     .
$setGlobal outDir    .
$setGlobal utility   cde
$setGlobal savfFlag  capFix
$setGlobal ifCal     0
$setGlobal ifSUB     1
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
$setGlobal ifSUB     1
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
   acts           "Activities"      / Food, Mnfcs, Svces /
   comm           "Commodities"     / Food, Mnfcs, Svces /
   marg(comm)     "Margin comm"     / Svces /
   reg            "Regions"         / USA, EU_28, EGY, ROW /
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
   Food, Mnfcs, Svces, Land, UnSkLab, SkLab, Capital, NatRes, TRD, hhd, gov, inv, deprY, tmg, itax, ptax, mtax, etax, vtax, vsub, dtax, ctax, bop, tot,
   USA, EU_28, EGY, ROW
/ ;
alias(is, js) ;

set aa(is) "Armington agents" / Food, Mnfcs, Svces, hhd, gov, inv, tmg / ;
set a(aa)  "Activities"       / Food, Mnfcs, Svces / ;
set i(is)  "Commodities"      / Food, Mnfcs, Svces / ;
alias(i, j) ;

set r(is)  "Regions" / USA, EU_28, EGY, ROW / ;
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
parameter DPSM(*) ;
DPSM("USA") = 1 ;
DPSM("EU_28") = 1 ;
DPSM("EGY") = 1 ;
DPSM("ROW") = 1 ;
parameter VDFB(*,*,*) ;
VDFB("Food","Food","USA") = 542355 ;
VDFB("Food","Food","EU_28") = 500638.0625 ;
VDFB("Food","Food","EGY") = 18253.5546875 ;
VDFB("Food","Food","ROW") = 2883517.25 ;
VDFB("Food","Mnfcs","USA") = 41297.44921875 ;
VDFB("Food","Mnfcs","EU_28") = 48286.37890625 ;
VDFB("Food","Mnfcs","EGY") = 754.209228515625 ;
VDFB("Food","Mnfcs","ROW") = 505030.9375 ;
VDFB("Food","Svces","USA") = 177164.1875 ;
VDFB("Food","Svces","EU_28") = 243287.625 ;
VDFB("Food","Svces","EGY") = 5867.33642578125 ;
VDFB("Food","Svces","ROW") = 1004173.25 ;
VDFB("Mnfcs","Food","USA") = 137371.984375 ;
VDFB("Mnfcs","Food","EU_28") = 106197.921875 ;
VDFB("Mnfcs","Food","EGY") = 2704.27172851562 ;
VDFB("Mnfcs","Food","ROW") = 481191.8125 ;
VDFB("Mnfcs","Mnfcs","USA") = 1611896.25 ;
VDFB("Mnfcs","Mnfcs","EU_28") = 1839815.75 ;
VDFB("Mnfcs","Mnfcs","EGY") = 26598.685546875 ;
VDFB("Mnfcs","Mnfcs","ROW") = 11120416 ;
VDFB("Mnfcs","Svces","USA") = 1283868.25 ;
VDFB("Mnfcs","Svces","EU_28") = 1113946.25 ;
VDFB("Mnfcs","Svces","EGY") = 33387.8203125 ;
VDFB("Mnfcs","Svces","ROW") = 5495306.5 ;
VDFB("Svces","Food","USA") = 331707.65625 ;
VDFB("Svces","Food","EU_28") = 404140.875 ;
VDFB("Svces","Food","EGY") = 4651.265625 ;
VDFB("Svces","Food","ROW") = 1360518.5 ;
VDFB("Svces","Mnfcs","USA") = 1069950.375 ;
VDFB("Svces","Mnfcs","EU_28") = 1605810.625 ;
VDFB("Svces","Mnfcs","EGY") = 19633.658203125 ;
VDFB("Svces","Mnfcs","ROW") = 5046125.5 ;
VDFB("Svces","Svces","USA") = 7592817.5 ;
VDFB("Svces","Svces","EU_28") = 6877007.5 ;
VDFB("Svces","Svces","EGY") = 25979.509765625 ;
VDFB("Svces","Svces","ROW") = 13834527 ;
parameter VDFP(*,*,*) ;
VDFP("Food","Food","USA") = 542169.5 ;
VDFP("Food","Food","EU_28") = 497018.5625 ;
VDFP("Food","Food","EGY") = 18253.41796875 ;
VDFP("Food","Food","ROW") = 2878283.25 ;
VDFP("Food","Mnfcs","USA") = 41122.109375 ;
VDFP("Food","Mnfcs","EU_28") = 48691.21875 ;
VDFP("Food","Mnfcs","EGY") = 755.476135253906 ;
VDFP("Food","Mnfcs","ROW") = 510896.125 ;
VDFP("Food","Svces","USA") = 177681.828125 ;
VDFP("Food","Svces","EU_28") = 260224.734375 ;
VDFP("Food","Svces","EGY") = 5861.7294921875 ;
VDFP("Food","Svces","ROW") = 1037460.9375 ;
VDFP("Mnfcs","Food","USA") = 139415.390625 ;
VDFP("Mnfcs","Food","EU_28") = 111128.078125 ;
VDFP("Mnfcs","Food","EGY") = 2423.99975585938 ;
VDFP("Mnfcs","Food","ROW") = 489347.15625 ;
VDFP("Mnfcs","Mnfcs","USA") = 1669532.125 ;
VDFP("Mnfcs","Mnfcs","EU_28") = 1874450.375 ;
VDFP("Mnfcs","Mnfcs","EGY") = 25783.0703125 ;
VDFP("Mnfcs","Mnfcs","ROW") = 11425012 ;
VDFP("Mnfcs","Svces","USA") = 1348394.875 ;
VDFP("Mnfcs","Svces","EU_28") = 1253490 ;
VDFP("Mnfcs","Svces","EGY") = 28016.091796875 ;
VDFP("Mnfcs","Svces","ROW") = 5688063 ;
VDFP("Svces","Food","USA") = 331101.875 ;
VDFP("Svces","Food","EU_28") = 408120.375 ;
VDFP("Svces","Food","EGY") = 4052.90405273438 ;
VDFP("Svces","Food","ROW") = 1353884.125 ;
VDFP("Svces","Mnfcs","USA") = 1086692.25 ;
VDFP("Svces","Mnfcs","EU_28") = 1651084.25 ;
VDFP("Svces","Mnfcs","EGY") = 17362.498046875 ;
VDFP("Svces","Mnfcs","ROW") = 5110019.5 ;
VDFP("Svces","Svces","USA") = 7656514 ;
VDFP("Svces","Svces","EU_28") = 7102183 ;
VDFP("Svces","Svces","EGY") = 22831.5625 ;
VDFP("Svces","Svces","ROW") = 14078633 ;
parameter VMFB(*,*,*) ;
VMFB("Food","Food","USA") = 44631.125 ;
VMFB("Food","Food","EU_28") = 188689.59375 ;
VMFB("Food","Food","EGY") = 10800.80078125 ;
VMFB("Food","Food","ROW") = 324834.53125 ;
VMFB("Food","Mnfcs","USA") = 2144.67797851562 ;
VMFB("Food","Mnfcs","EU_28") = 26389.56640625 ;
VMFB("Food","Mnfcs","EGY") = 731.0107421875 ;
VMFB("Food","Mnfcs","ROW") = 42496.734375 ;
VMFB("Food","Svces","USA") = 23047.599609375 ;
VMFB("Food","Svces","EU_28") = 73303.75 ;
VMFB("Food","Svces","EGY") = 1884.67724609375 ;
VMFB("Food","Svces","ROW") = 128462.078125 ;
VMFB("Mnfcs","Food","USA") = 30397.654296875 ;
VMFB("Mnfcs","Food","EU_28") = 76841.015625 ;
VMFB("Mnfcs","Food","EGY") = 3507.85913085938 ;
VMFB("Mnfcs","Food","ROW") = 188926.34375 ;
VMFB("Mnfcs","Mnfcs","USA") = 664758.25 ;
VMFB("Mnfcs","Mnfcs","EU_28") = 2185452.5 ;
VMFB("Mnfcs","Mnfcs","EGY") = 26431.388671875 ;
VMFB("Mnfcs","Mnfcs","ROW") = 3945859 ;
VMFB("Mnfcs","Svces","USA") = 370647.5 ;
VMFB("Mnfcs","Svces","EU_28") = 876487 ;
VMFB("Mnfcs","Svces","EGY") = 18297.591796875 ;
VMFB("Mnfcs","Svces","ROW") = 1670064.625 ;
VMFB("Svces","Food","USA") = 4509.24609375 ;
VMFB("Svces","Food","EU_28") = 47619.97265625 ;
VMFB("Svces","Food","EGY") = 631.299987792969 ;
VMFB("Svces","Food","ROW") = 43373.10546875 ;
VMFB("Svces","Mnfcs","USA") = 23632.78125 ;
VMFB("Svces","Mnfcs","EU_28") = 250204.203125 ;
VMFB("Svces","Mnfcs","EGY") = 4013.50463867188 ;
VMFB("Svces","Mnfcs","ROW") = 218960.28125 ;
VMFB("Svces","Svces","USA") = 225410.625 ;
VMFB("Svces","Svces","EU_28") = 1195691.375 ;
VMFB("Svces","Svces","EGY") = 6327.55859375 ;
VMFB("Svces","Svces","ROW") = 915242.9375 ;
parameter VMFP(*,*,*) ;
VMFP("Food","Food","USA") = 44721.359375 ;
VMFP("Food","Food","EU_28") = 190533.890625 ;
VMFP("Food","Food","EGY") = 10802.009765625 ;
VMFP("Food","Food","ROW") = 335126.1875 ;
VMFP("Food","Mnfcs","USA") = 2142.17553710938 ;
VMFP("Food","Mnfcs","EU_28") = 27106.546875 ;
VMFP("Food","Mnfcs","EGY") = 733.151916503906 ;
VMFP("Food","Mnfcs","ROW") = 45033.171875 ;
VMFP("Food","Svces","USA") = 23125.109375 ;
VMFP("Food","Svces","EU_28") = 80344.7890625 ;
VMFP("Food","Svces","EGY") = 1882.412109375 ;
VMFP("Food","Svces","ROW") = 137505.546875 ;
VMFP("Mnfcs","Food","USA") = 31042.537109375 ;
VMFP("Mnfcs","Food","EU_28") = 79483.5234375 ;
VMFP("Mnfcs","Food","EGY") = 3375.9140625 ;
VMFP("Mnfcs","Food","ROW") = 191657.25 ;
VMFP("Mnfcs","Mnfcs","USA") = 688171 ;
VMFP("Mnfcs","Mnfcs","EU_28") = 2220816.5 ;
VMFP("Mnfcs","Mnfcs","EGY") = 25746.8828125 ;
VMFP("Mnfcs","Mnfcs","ROW") = 4113230.5 ;
VMFP("Mnfcs","Svces","USA") = 395723.5 ;
VMFP("Mnfcs","Svces","EU_28") = 990513.75 ;
VMFP("Mnfcs","Svces","EGY") = 15854.8681640625 ;
VMFP("Mnfcs","Svces","ROW") = 1752161 ;
VMFP("Svces","Food","USA") = 4503.52197265625 ;
VMFP("Svces","Food","EU_28") = 47888.4296875 ;
VMFP("Svces","Food","EGY") = 631.936157226562 ;
VMFP("Svces","Food","ROW") = 43470.95703125 ;
VMFP("Svces","Mnfcs","USA") = 23847.408203125 ;
VMFP("Svces","Mnfcs","EU_28") = 253778.390625 ;
VMFP("Svces","Mnfcs","EGY") = 4013.95166015625 ;
VMFP("Svces","Mnfcs","ROW") = 223223.59375 ;
VMFP("Svces","Svces","USA") = 228811.734375 ;
VMFP("Svces","Svces","EU_28") = 1238011 ;
VMFP("Svces","Svces","EGY") = 6340.2880859375 ;
VMFP("Svces","Svces","ROW") = 936705.375 ;
parameter VDPB(*,*) ;
VDPB("Food","USA") = 702662.25 ;
VDPB("Food","EU_28") = 638599.5625 ;
VDPB("Food","EGY") = 51113.64453125 ;
VDPB("Food","ROW") = 4037541 ;
VDPB("Mnfcs","USA") = 792360.1875 ;
VDPB("Mnfcs","EU_28") = 433667 ;
VDPB("Mnfcs","EGY") = 48288.2109375 ;
VDPB("Mnfcs","ROW") = 2272132.5 ;
VDPB("Svces","USA") = 10661360 ;
VDPB("Svces","EU_28") = 6461904 ;
VDPB("Svces","EGY") = 110519.3359375 ;
VDPB("Svces","ROW") = 13741794 ;
parameter VDPP(*,*) ;
VDPP("Food","USA") = 701175.125 ;
VDPP("Food","EU_28") = 823667 ;
VDPP("Food","EGY") = 51113.765625 ;
VDPP("Food","ROW") = 4233629.5 ;
VDPP("Mnfcs","USA") = 811949.3125 ;
VDPP("Mnfcs","EU_28") = 637483.5625 ;
VDPP("Mnfcs","EGY") = 44298.50390625 ;
VDPP("Mnfcs","ROW") = 2474598 ;
VDPP("Svces","USA") = 10888380 ;
VDPP("Svces","EU_28") = 6805092.5 ;
VDPP("Svces","EGY") = 106365.4375 ;
VDPP("Svces","ROW") = 14160749 ;
parameter VMPB(*,*) ;
VMPB("Food","USA") = 92016.71875 ;
VMPB("Food","EU_28") = 282465.9375 ;
VMPB("Food","EGY") = 2255.00634765625 ;
VMPB("Food","ROW") = 456149.03125 ;
VMPB("Mnfcs","USA") = 632188.8125 ;
VMPB("Mnfcs","EU_28") = 834982.125 ;
VMPB("Mnfcs","EGY") = 5817.544921875 ;
VMPB("Mnfcs","ROW") = 1308079.5 ;
VMPB("Svces","USA") = 179993.15625 ;
VMPB("Svces","EU_28") = 164734.515625 ;
VMPB("Svces","EGY") = 67.8350296020508 ;
VMPB("Svces","ROW") = 652328.75 ;
parameter VMPP(*,*) ;
VMPP("Food","USA") = 91885.0625 ;
VMPP("Food","EU_28") = 361205.875 ;
VMPP("Food","EGY") = 2255.36547851562 ;
VMPP("Food","ROW") = 514590.78125 ;
VMPP("Mnfcs","USA") = 654085.1875 ;
VMPP("Mnfcs","EU_28") = 1120764.375 ;
VMPP("Mnfcs","EGY") = 5785.94482421875 ;
VMPP("Mnfcs","ROW") = 1461858.125 ;
VMPP("Svces","USA") = 185092.046875 ;
VMPP("Svces","EU_28") = 179237.203125 ;
VMPP("Svces","EGY") = 68.974479675293 ;
VMPP("Svces","ROW") = 682771.25 ;
parameter VDGB(*,*) ;
VDGB("Food","USA") = 1023.76983642578 ;
VDGB("Food","EU_28") = 5274.97412109375 ;
VDGB("Food","EGY") = 95.81298828125 ;
VDGB("Food","ROW") = 12277.640625 ;
VDGB("Mnfcs","USA") = 405.889831542969 ;
VDGB("Mnfcs","EU_28") = 35854.4609375 ;
VDGB("Mnfcs","EGY") = 138.426818847656 ;
VDGB("Mnfcs","ROW") = 59547.4296875 ;
VDGB("Svces","USA") = 2543765 ;
VDGB("Svces","EU_28") = 3504409.5 ;
VDGB("Svces","EGY") = 23423.26953125 ;
VDGB("Svces","ROW") = 6951075 ;
parameter VDGP(*,*) ;
VDGP("Food","USA") = 1023.76983642578 ;
VDGP("Food","EU_28") = 5454.49853515625 ;
VDGP("Food","EGY") = 95.8700180053711 ;
VDGP("Food","ROW") = 12380.267578125 ;
VDGP("Mnfcs","USA") = 405.889831542969 ;
VDGP("Mnfcs","EU_28") = 40958.1875 ;
VDGP("Mnfcs","EGY") = 138.409225463867 ;
VDGP("Mnfcs","ROW") = 60019.26953125 ;
VDGP("Svces","USA") = 2729009.5 ;
VDGP("Svces","EU_28") = 3507268.75 ;
VDGP("Svces","EGY") = 23423.384765625 ;
VDGP("Svces","ROW") = 6993915 ;
parameter VMGB(*,*) ;
VMGB("Food","USA") = 7.94238901138306 ;
VMGB("Food","EU_28") = 200.922103881836 ;
VMGB("Food","EGY") = 1.67209494113922 ;
VMGB("Food","ROW") = 1953.03100585938 ;
VMGB("Mnfcs","USA") = 331.96826171875 ;
VMGB("Mnfcs","EU_28") = 68117.15625 ;
VMGB("Mnfcs","EGY") = 86.3442077636719 ;
VMGB("Mnfcs","ROW") = 66993.015625 ;
VMGB("Svces","USA") = 19186.185546875 ;
VMGB("Svces","EU_28") = 7234.12353515625 ;
VMGB("Svces","EGY") = 249.014266967773 ;
VMGB("Svces","ROW") = 101731.09375 ;
parameter VMGP(*,*) ;
VMGP("Food","USA") = 7.94238901138306 ;
VMGP("Food","EU_28") = 223.506286621094 ;
VMGP("Food","EGY") = 1.84822034835815 ;
VMGP("Food","ROW") = 1975.64379882812 ;
VMGP("Mnfcs","USA") = 331.96826171875 ;
VMGP("Mnfcs","EU_28") = 76792.6875 ;
VMGP("Mnfcs","EGY") = 92.0447692871094 ;
VMGP("Mnfcs","ROW") = 66904.546875 ;
VMGP("Svces","USA") = 19186.185546875 ;
VMGP("Svces","EU_28") = 9761.845703125 ;
VMGP("Svces","EGY") = 285.450561523438 ;
VMGP("Svces","ROW") = 107428.953125 ;
parameter VDIB(*,*) ;
VDIB("Food","USA") = 522.650085449219 ;
VDIB("Food","EU_28") = 5696.13232421875 ;
VDIB("Food","EGY") = 219.726898193359 ;
VDIB("Food","ROW") = 85922.0625 ;
VDIB("Mnfcs","USA") = 765932.4375 ;
VDIB("Mnfcs","EU_28") = 473251.46875 ;
VDIB("Mnfcs","EGY") = 4181.40283203125 ;
VDIB("Mnfcs","ROW") = 2097593.5 ;
VDIB("Svces","USA") = 2873172.5 ;
VDIB("Svces","EU_28") = 2258155.25 ;
VDIB("Svces","EGY") = 27085.1015625 ;
VDIB("Svces","ROW") = 8863570 ;
parameter VDIP(*,*) ;
VDIP("Food","USA") = 522.650085449219 ;
VDIP("Food","EU_28") = 5491.546875 ;
VDIP("Food","EGY") = 219.726898193359 ;
VDIP("Food","ROW") = 86727.4453125 ;
VDIP("Mnfcs","USA") = 775809.125 ;
VDIP("Mnfcs","EU_28") = 490787.6875 ;
VDIP("Mnfcs","EGY") = 4181.41748046875 ;
VDIP("Mnfcs","ROW") = 2218108.5 ;
VDIP("Svces","USA") = 2692840.75 ;
VDIP("Svces","EU_28") = 2438634.25 ;
VDIP("Svces","EGY") = 27086.42578125 ;
VDIP("Svces","ROW") = 9153967 ;
parameter VMIB(*,*) ;
VMIB("Food","USA") = 22.4080238342285 ;
VMIB("Food","EU_28") = 2048.05322265625 ;
VMIB("Food","EGY") = 0.738950550556183 ;
VMIB("Food","ROW") = 9514.98828125 ;
VMIB("Mnfcs","USA") = 473778.375 ;
VMIB("Mnfcs","EU_28") = 582482.6875 ;
VMIB("Mnfcs","EGY") = 3696.0966796875 ;
VMIB("Mnfcs","ROW") = 1290977.75 ;
VMIB("Svces","USA") = 99352.0078125 ;
VMIB("Svces","EU_28") = 92506.1640625 ;
VMIB("Svces","EGY") = 29.5622272491455 ;
VMIB("Svces","ROW") = 139340.4375 ;
parameter VMIP(*,*) ;
VMIP("Food","USA") = 22.4080238342285 ;
VMIP("Food","EU_28") = 2077.35009765625 ;
VMIP("Food","EGY") = 0.74113941192627 ;
VMIP("Food","ROW") = 9678.7431640625 ;
VMIP("Mnfcs","USA") = 491439.5 ;
VMIP("Mnfcs","EU_28") = 614703.125 ;
VMIP("Mnfcs","EGY") = 3782.42797851562 ;
VMIP("Mnfcs","ROW") = 1392075.25 ;
VMIP("Svces","USA") = 87960.8984375 ;
VMIP("Svces","EU_28") = 99512.546875 ;
VMIP("Svces","EGY") = 29.7311820983887 ;
VMIP("Svces","ROW") = 143247.421875 ;
parameter EVFB(*,*,*) ;
EVFB("Land","Food","USA") = 41351.43359375 ;
EVFB("Land","Food","EU_28") = 67458 ;
EVFB("Land","Food","EGY") = 11286.869140625 ;
EVFB("Land","Food","ROW") = 657967.375 ;
EVFB("UnSkLab","Food","USA") = 119645.0625 ;
EVFB("UnSkLab","Food","EU_28") = 142159.09375 ;
EVFB("UnSkLab","Food","EGY") = 5294.88818359375 ;
EVFB("UnSkLab","Food","ROW") = 1513971.875 ;
EVFB("UnSkLab","Mnfcs","USA") = 705015.5625 ;
EVFB("UnSkLab","Mnfcs","EU_28") = 439850.90625 ;
EVFB("UnSkLab","Mnfcs","EGY") = 5368.556640625 ;
EVFB("UnSkLab","Mnfcs","ROW") = 2227667.25 ;
EVFB("UnSkLab","Svces","USA") = 3322260.5 ;
EVFB("UnSkLab","Svces","EU_28") = 1800742.625 ;
EVFB("UnSkLab","Svces","EGY") = 31012.890625 ;
EVFB("UnSkLab","Svces","ROW") = 7273297 ;
EVFB("SkLab","Food","USA") = 61255.26171875 ;
EVFB("SkLab","Food","EU_28") = 90423.3125 ;
EVFB("SkLab","Food","EGY") = 7801.65869140625 ;
EVFB("SkLab","Food","ROW") = 197468.484375 ;
EVFB("SkLab","Mnfcs","USA") = 233255.890625 ;
EVFB("SkLab","Mnfcs","EU_28") = 477022.59375 ;
EVFB("SkLab","Mnfcs","EGY") = 11898.587890625 ;
EVFB("SkLab","Mnfcs","ROW") = 799560.75 ;
EVFB("SkLab","Svces","USA") = 3798211.75 ;
EVFB("SkLab","Svces","EU_28") = 2533243.25 ;
EVFB("SkLab","Svces","EGY") = 45485.515625 ;
EVFB("SkLab","Svces","ROW") = 5726672.5 ;
EVFB("Capital","Food","USA") = 225441.96875 ;
EVFB("Capital","Food","EU_28") = 294794.6875 ;
EVFB("Capital","Food","EGY") = 16000.41796875 ;
EVFB("Capital","Food","ROW") = 1481268.625 ;
EVFB("Capital","Mnfcs","USA") = 968529.625 ;
EVFB("Capital","Mnfcs","EU_28") = 1078327.5 ;
EVFB("Capital","Mnfcs","EGY") = 40278.09375 ;
EVFB("Capital","Mnfcs","ROW") = 4580838 ;
EVFB("Capital","Svces","USA") = 6049054 ;
EVFB("Capital","Svces","EU_28") = 5707220.5 ;
EVFB("Capital","Svces","EGY") = 69308.8125 ;
EVFB("Capital","Svces","ROW") = 13100909 ;
EVFB("NatRes","Food","USA") = 5286.4248046875 ;
EVFB("NatRes","Food","EU_28") = 8407.166015625 ;
EVFB("NatRes","Food","EGY") = 1102.265625 ;
EVFB("NatRes","Food","ROW") = 111336.0078125 ;
EVFB("NatRes","Mnfcs","USA") = 75873.765625 ;
EVFB("NatRes","Mnfcs","EU_28") = 20106.6875 ;
EVFB("NatRes","Mnfcs","EGY") = 4422.7607421875 ;
EVFB("NatRes","Mnfcs","ROW") = 567630.125 ;
parameter EVFP(*,*,*) ;
EVFP("Land","Food","USA") = 34003.03515625 ;
EVFP("Land","Food","EU_28") = 45434.83203125 ;
EVFP("Land","Food","EGY") = 11296.6181640625 ;
EVFP("Land","Food","ROW") = 628994.5625 ;
EVFP("UnSkLab","Food","USA") = 141928.640625 ;
EVFP("UnSkLab","Food","EU_28") = 188846.1875 ;
EVFP("UnSkLab","Food","EGY") = 5567.13720703125 ;
EVFP("UnSkLab","Food","ROW") = 1589492.125 ;
EVFP("UnSkLab","Mnfcs","USA") = 848535.1875 ;
EVFP("UnSkLab","Mnfcs","EU_28") = 644717 ;
EVFP("UnSkLab","Mnfcs","EGY") = 5644.5927734375 ;
EVFP("UnSkLab","Mnfcs","ROW") = 2428882.75 ;
EVFP("UnSkLab","Svces","USA") = 3998571.25 ;
EVFP("UnSkLab","Svces","EU_28") = 2567129 ;
EVFP("UnSkLab","Svces","EGY") = 32607.48828125 ;
EVFP("UnSkLab","Svces","ROW") = 8057049.5 ;
EVFP("SkLab","Food","USA") = 71701.9296875 ;
EVFP("SkLab","Food","EU_28") = 123548.7578125 ;
EVFP("SkLab","Food","EGY") = 8202.798828125 ;
EVFP("SkLab","Food","ROW") = 215659.859375 ;
EVFP("SkLab","Mnfcs","USA") = 280739.65625 ;
EVFP("SkLab","Mnfcs","EU_28") = 687521.6875 ;
EVFP("SkLab","Mnfcs","EGY") = 12510.380859375 ;
EVFP("SkLab","Mnfcs","ROW") = 905285.0625 ;
EVFP("SkLab","Svces","USA") = 4571411.5 ;
EVFP("SkLab","Svces","EU_28") = 3625571.75 ;
EVFP("SkLab","Svces","EGY") = 47824.2578125 ;
EVFP("SkLab","Svces","ROW") = 6436724 ;
EVFP("Capital","Food","USA") = 233096.546875 ;
EVFP("Capital","Food","EU_28") = 286427.125 ;
EVFP("Capital","Food","EGY") = 16014.2373046875 ;
EVFP("Capital","Food","ROW") = 1472841.125 ;
EVFP("Capital","Mnfcs","USA") = 1017035.1875 ;
EVFP("Capital","Mnfcs","EU_28") = 1108303.5 ;
EVFP("Capital","Mnfcs","EGY") = 40312.8828125 ;
EVFP("Capital","Mnfcs","ROW") = 4652894 ;
EVFP("Capital","Svces","USA") = 6352001 ;
EVFP("Capital","Svces","EU_28") = 5897007 ;
EVFP("Capital","Svces","EGY") = 69368.671875 ;
EVFP("Capital","Svces","ROW") = 13335567 ;
EVFP("NatRes","Food","USA") = 5551.177734375 ;
EVFP("NatRes","Food","EU_28") = 8703.9833984375 ;
EVFP("NatRes","Food","EGY") = 1103.21765136719 ;
EVFP("NatRes","Food","ROW") = 112653.515625 ;
EVFP("NatRes","Mnfcs","USA") = 79673.6484375 ;
EVFP("NatRes","Mnfcs","EU_28") = 20711.26953125 ;
EVFP("NatRes","Mnfcs","EGY") = 4426.58056640625 ;
EVFP("NatRes","Mnfcs","ROW") = 575325.4375 ;
parameter EVOS(*,*,*) ;
EVOS("Land","Food","USA") = 39669.4453125 ;
EVOS("Land","Food","EU_28") = 63766.05859375 ;
EVOS("Land","Food","EGY") = 10496.69140625 ;
EVOS("Land","Food","ROW") = 603323.125 ;
EVOS("UnSkLab","Food","USA") = 90088.765625 ;
EVOS("UnSkLab","Food","EU_28") = 99131.8984375 ;
EVOS("UnSkLab","Food","EGY") = 5123.17578125 ;
EVOS("UnSkLab","Food","ROW") = 1425459.25 ;
EVOS("UnSkLab","Mnfcs","USA") = 530853.375 ;
EVOS("UnSkLab","Mnfcs","EU_28") = 302901.28125 ;
EVOS("UnSkLab","Mnfcs","EGY") = 5194.455078125 ;
EVOS("UnSkLab","Mnfcs","ROW") = 2070165.875 ;
EVOS("UnSkLab","Svces","USA") = 2501552 ;
EVOS("UnSkLab","Svces","EU_28") = 1275810.875 ;
EVOS("UnSkLab","Svces","EGY") = 30007.1484375 ;
EVOS("UnSkLab","Svces","ROW") = 6658387.5 ;
EVOS("SkLab","Food","USA") = 46123.1875 ;
EVOS("SkLab","Food","EU_28") = 62689.94921875 ;
EVOS("SkLab","Food","EGY") = 7548.65234375 ;
EVOS("SkLab","Food","ROW") = 178891.484375 ;
EVOS("SkLab","Mnfcs","USA") = 175633.96875 ;
EVOS("SkLab","Mnfcs","EU_28") = 330767.46875 ;
EVOS("SkLab","Mnfcs","EGY") = 11512.71875 ;
EVOS("SkLab","Mnfcs","ROW") = 717563.9375 ;
EVOS("SkLab","Svces","USA") = 2859927.75 ;
EVOS("SkLab","Svces","EU_28") = 1781857.5 ;
EVOS("SkLab","Svces","EGY") = 44010.4296875 ;
EVOS("SkLab","Svces","ROW") = 5104825 ;
EVOS("Capital","Food","USA") = 216271.984375 ;
EVOS("Capital","Food","EU_28") = 277643.59375 ;
EVOS("Capital","Food","EGY") = 14880.248046875 ;
EVOS("Capital","Food","ROW") = 1365586.5 ;
EVOS("Capital","Mnfcs","USA") = 929134.125 ;
EVOS("Capital","Mnfcs","EU_28") = 1016128.125 ;
EVOS("Capital","Mnfcs","EGY") = 37458.2734375 ;
EVOS("Capital","Mnfcs","ROW") = 4212382.5 ;
EVOS("Capital","Svces","USA") = 5803005.5 ;
EVOS("Capital","Svces","EU_28") = 5364355.5 ;
EVOS("Capital","Svces","EGY") = 64456.58984375 ;
EVOS("Capital","Svces","ROW") = 12042100 ;
EVOS("NatRes","Food","USA") = 5071.39697265625 ;
EVOS("NatRes","Food","EU_28") = 7891.44140625 ;
EVOS("NatRes","Food","EGY") = 1025.09753417969 ;
EVOS("NatRes","Food","ROW") = 101785.9140625 ;
EVOS("NatRes","Mnfcs","USA") = 72787.5625 ;
EVOS("NatRes","Mnfcs","EU_28") = 18835.375 ;
EVOS("NatRes","Mnfcs","EGY") = 4113.12890625 ;
EVOS("NatRes","Mnfcs","ROW") = 525782.75 ;
parameter VXSB(*,*,*) ;
VXSB("Food","USA","USA") = 2.2000000171829E-5 ;
VXSB("Food","USA","EU_28") = 12887.2734375 ;
VXSB("Food","USA","EGY") = 784.747131347656 ;
VXSB("Food","USA","ROW") = 141888.078125 ;
VXSB("Food","EU_28","USA") = 25649.767578125 ;
VXSB("Food","EU_28","EU_28") = 391281.84375 ;
VXSB("Food","EU_28","EGY") = 1714.29028320312 ;
VXSB("Food","EU_28","ROW") = 126685 ;
VXSB("Food","EGY","USA") = 138.262451171875 ;
VXSB("Food","EGY","EU_28") = 1050.630859375 ;
VXSB("Food","EGY","EGY") = 2.2000000171829E-5 ;
VXSB("Food","EGY","ROW") = 4228.529296875 ;
VXSB("Food","ROW","USA") = 125112.3125 ;
VXSB("Food","ROW","EU_28") = 131745.40625 ;
VXSB("Food","ROW","EGY") = 10609.853515625 ;
VXSB("Food","ROW","ROW") = 554035.125 ;
VXSB("Mnfcs","USA","USA") = 2.30000005103648E-5 ;
VXSB("Mnfcs","USA","EU_28") = 225169.796875 ;
VXSB("Mnfcs","USA","EGY") = 2550.80419921875 ;
VXSB("Mnfcs","USA","ROW") = 1041180.75 ;
VXSB("Mnfcs","EU_28","USA") = 379717.875 ;
VXSB("Mnfcs","EU_28","EU_28") = 2729899.75 ;
VXSB("Mnfcs","EU_28","EGY") = 16143.79296875 ;
VXSB("Mnfcs","EU_28","ROW") = 1432956.125 ;
VXSB("Mnfcs","EGY","USA") = 1660.669921875 ;
VXSB("Mnfcs","EGY","EU_28") = 6934.099609375 ;
VXSB("Mnfcs","EGY","EGY") = 2.30000005103648E-5 ;
VXSB("Mnfcs","EGY","ROW") = 13405.0888671875 ;
VXSB("Mnfcs","ROW","USA") = 1687738.875 ;
VXSB("Mnfcs","ROW","EU_28") = 1483928.125 ;
VXSB("Mnfcs","ROW","EGY") = 32927.69921875 ;
VXSB("Mnfcs","ROW","ROW") = 5367998 ;
VXSB("Svces","USA","USA") = 1.99999994947575E-5 ;
VXSB("Svces","USA","EU_28") = 223336.65625 ;
VXSB("Svces","USA","EGY") = 2707.74658203125 ;
VXSB("Svces","USA","ROW") = 478846.28125 ;
VXSB("Svces","EU_28","USA") = 199364.0625 ;
VXSB("Svces","EU_28","EU_28") = 1061178.25 ;
VXSB("Svces","EU_28","EGY") = 4062.7734375 ;
VXSB("Svces","EU_28","ROW") = 564668.0625 ;
VXSB("Svces","EGY","USA") = 2516.6015625 ;
VXSB("Svces","EGY","EU_28") = 6072.31982421875 ;
VXSB("Svces","EGY","EGY") = 1.99999994947575E-5 ;
VXSB("Svces","EGY","ROW") = 7648.29248046875 ;
VXSB("Svces","ROW","USA") = 350203.4375 ;
VXSB("Svces","ROW","EU_28") = 467403.15625 ;
VXSB("Svces","ROW","EGY") = 4548.25537109375 ;
VXSB("Svces","ROW","ROW") = 1019777.4375 ;
parameter VFOB(*,*,*) ;
VFOB("Food","USA","USA") = 2.2000000171829E-5 ;
VFOB("Food","USA","EU_28") = 12915.6787109375 ;
VFOB("Food","USA","EGY") = 790.277954101562 ;
VFOB("Food","USA","ROW") = 142180.78125 ;
VFOB("Food","EU_28","USA") = 25651.373046875 ;
VFOB("Food","EU_28","EU_28") = 391281.84375 ;
VFOB("Food","EU_28","EGY") = 1714.65283203125 ;
VFOB("Food","EU_28","ROW") = 126683.6015625 ;
VFOB("Food","EGY","USA") = 138.26286315918 ;
VFOB("Food","EGY","EU_28") = 1050.64233398438 ;
VFOB("Food","EGY","EGY") = 2.2000000171829E-5 ;
VFOB("Food","EGY","ROW") = 4228.619140625 ;
VFOB("Food","ROW","USA") = 125103.1171875 ;
VFOB("Food","ROW","EU_28") = 131780.34375 ;
VFOB("Food","ROW","EGY") = 10608.09375 ;
VFOB("Food","ROW","ROW") = 554839 ;
VFOB("Mnfcs","USA","USA") = 2.30000005103648E-5 ;
VFOB("Mnfcs","USA","EU_28") = 237000.96875 ;
VFOB("Mnfcs","USA","EGY") = 2691.998046875 ;
VFOB("Mnfcs","USA","ROW") = 1067566.875 ;
VFOB("Mnfcs","EU_28","USA") = 380644.03125 ;
VFOB("Mnfcs","EU_28","EU_28") = 2729899.75 ;
VFOB("Mnfcs","EU_28","EGY") = 16235.666015625 ;
VFOB("Mnfcs","EU_28","ROW") = 1435910.625 ;
VFOB("Mnfcs","EGY","USA") = 1660.58532714844 ;
VFOB("Mnfcs","EGY","EU_28") = 6933.74072265625 ;
VFOB("Mnfcs","EGY","EGY") = 2.30000005103648E-5 ;
VFOB("Mnfcs","EGY","ROW") = 13404.8740234375 ;
VFOB("Mnfcs","ROW","USA") = 1701160.875 ;
VFOB("Mnfcs","ROW","EU_28") = 1499655.875 ;
VFOB("Mnfcs","ROW","EGY") = 33351.453125 ;
VFOB("Mnfcs","ROW","ROW") = 5423686 ;
VFOB("Svces","USA","USA") = 1.99999994947575E-5 ;
VFOB("Svces","USA","EU_28") = 223336.65625 ;
VFOB("Svces","USA","EGY") = 2707.74658203125 ;
VFOB("Svces","USA","ROW") = 478846.28125 ;
VFOB("Svces","EU_28","USA") = 199364.0625 ;
VFOB("Svces","EU_28","EU_28") = 1061178.25 ;
VFOB("Svces","EU_28","EGY") = 4062.7734375 ;
VFOB("Svces","EU_28","ROW") = 564668.0625 ;
VFOB("Svces","EGY","USA") = 2516.6015625 ;
VFOB("Svces","EGY","EU_28") = 6072.31982421875 ;
VFOB("Svces","EGY","EGY") = 1.99999994947575E-5 ;
VFOB("Svces","EGY","ROW") = 7648.29248046875 ;
VFOB("Svces","ROW","USA") = 350203.4375 ;
VFOB("Svces","ROW","EU_28") = 467403.15625 ;
VFOB("Svces","ROW","EGY") = 4548.25537109375 ;
VFOB("Svces","ROW","ROW") = 1019777.4375 ;
parameter VCIF(*,*,*) ;
VCIF("Food","USA","USA") = 2.2000000171829E-5 ;
VCIF("Food","USA","EU_28") = 13609.5244140625 ;
VCIF("Food","USA","EGY") = 846.967346191406 ;
VCIF("Food","USA","ROW") = 150036.328125 ;
VCIF("Food","EU_28","USA") = 27180.794921875 ;
VCIF("Food","EU_28","EU_28") = 411168.65625 ;
VCIF("Food","EU_28","EGY") = 1855.27258300781 ;
VCIF("Food","EU_28","ROW") = 135242.3125 ;
VCIF("Food","EGY","USA") = 146.688278198242 ;
VCIF("Food","EGY","EU_28") = 1165.31433105469 ;
VCIF("Food","EGY","EGY") = 2.2000000171829E-5 ;
VCIF("Food","EGY","ROW") = 4654.14892578125 ;
VCIF("Food","ROW","USA") = 131293.375 ;
VCIF("Food","ROW","EU_28") = 140530.875 ;
VCIF("Food","ROW","EGY") = 11363.1328125 ;
VCIF("Food","ROW","ROW") = 595720.1875 ;
VCIF("Mnfcs","USA","USA") = 2.30000005103648E-5 ;
VCIF("Mnfcs","USA","EU_28") = 243716.96875 ;
VCIF("Mnfcs","USA","EGY") = 2826.32836914062 ;
VCIF("Mnfcs","USA","ROW") = 1093185.25 ;
VCIF("Mnfcs","EU_28","USA") = 390546.9375 ;
VCIF("Mnfcs","EU_28","EU_28") = 2790937 ;
VCIF("Mnfcs","EU_28","EGY") = 16896.837890625 ;
VCIF("Mnfcs","EU_28","ROW") = 1479270.375 ;
VCIF("Mnfcs","EGY","USA") = 1743.38391113281 ;
VCIF("Mnfcs","EGY","EU_28") = 7279.71826171875 ;
VCIF("Mnfcs","EGY","EGY") = 2.30000005103648E-5 ;
VCIF("Mnfcs","EGY","ROW") = 14096.75390625 ;
VCIF("Mnfcs","ROW","USA") = 1747721.25 ;
VCIF("Mnfcs","ROW","EU_28") = 1553619.125 ;
VCIF("Mnfcs","ROW","EGY") = 35055.79296875 ;
VCIF("Mnfcs","ROW","ROW") = 5643249 ;
VCIF("Svces","USA","USA") = 1.99999994947575E-5 ;
VCIF("Svces","USA","EU_28") = 223336.65625 ;
VCIF("Svces","USA","EGY") = 2707.74658203125 ;
VCIF("Svces","USA","ROW") = 478846.28125 ;
VCIF("Svces","EU_28","USA") = 199364.0625 ;
VCIF("Svces","EU_28","EU_28") = 1061178.25 ;
VCIF("Svces","EU_28","EGY") = 4062.7734375 ;
VCIF("Svces","EU_28","ROW") = 564668.0625 ;
VCIF("Svces","EGY","USA") = 2516.6015625 ;
VCIF("Svces","EGY","EU_28") = 6072.31982421875 ;
VCIF("Svces","EGY","EGY") = 1.99999994947575E-5 ;
VCIF("Svces","EGY","ROW") = 7648.29248046875 ;
VCIF("Svces","ROW","USA") = 350203.4375 ;
VCIF("Svces","ROW","EU_28") = 467403.15625 ;
VCIF("Svces","ROW","EGY") = 4548.25537109375 ;
VCIF("Svces","ROW","ROW") = 1019777.4375 ;
parameter VMSB(*,*,*) ;
VMSB("Food","USA","USA") = 2.2000000171829E-5 ;
VMSB("Food","USA","EU_28") = 14271.8291015625 ;
VMSB("Food","USA","EGY") = 1282.5849609375 ;
VMSB("Food","USA","ROW") = 160610.21875 ;
VMSB("Food","EU_28","USA") = 28559.935546875 ;
VMSB("Food","EU_28","EU_28") = 411168.65625 ;
VMSB("Food","EU_28","EGY") = 2463.50854492188 ;
VMSB("Food","EU_28","ROW") = 152860.953125 ;
VMSB("Food","EGY","USA") = 149.104827880859 ;
VMSB("Food","EGY","EU_28") = 1182.98413085938 ;
VMSB("Food","EGY","EGY") = 2.2000000171829E-5 ;
VMSB("Food","EGY","ROW") = 4866.69384765625 ;
VMSB("Food","ROW","USA") = 133161.4375 ;
VMSB("Food","ROW","EU_28") = 146474.390625 ;
VMSB("Food","ROW","EGY") = 11927.81640625 ;
VMSB("Food","ROW","ROW") = 645072.625 ;
VMSB("Mnfcs","USA","USA") = 2.30000005103648E-5 ;
VMSB("Mnfcs","USA","EU_28") = 248313.78125 ;
VMSB("Mnfcs","USA","EGY") = 3025.5986328125 ;
VMSB("Mnfcs","USA","ROW") = 1115602.875 ;
VMSB("Mnfcs","EU_28","USA") = 396228.46875 ;
VMSB("Mnfcs","EU_28","EU_28") = 2790937 ;
VMSB("Mnfcs","EU_28","EGY") = 17289.03125 ;
VMSB("Mnfcs","EU_28","ROW") = 1535448.75 ;
VMSB("Mnfcs","EGY","USA") = 1874.96423339844 ;
VMSB("Mnfcs","EGY","EU_28") = 7279.71826171875 ;
VMSB("Mnfcs","EGY","EGY") = 2.30000005103648E-5 ;
VMSB("Mnfcs","EGY","ROW") = 14331.0810546875 ;
VMSB("Mnfcs","ROW","USA") = 1773999.25 ;
VMSB("Mnfcs","ROW","EU_28") = 1577832.375 ;
VMSB("Mnfcs","ROW","EGY") = 37522.203125 ;
VMSB("Mnfcs","ROW","ROW") = 5805518.5 ;
VMSB("Svces","USA","USA") = 1.99999994947575E-5 ;
VMSB("Svces","USA","EU_28") = 223336.65625 ;
VMSB("Svces","USA","EGY") = 2707.74658203125 ;
VMSB("Svces","USA","ROW") = 478847.34375 ;
VMSB("Svces","EU_28","USA") = 199364.0625 ;
VMSB("Svces","EU_28","EU_28") = 1061178.25 ;
VMSB("Svces","EU_28","EGY") = 4062.7734375 ;
VMSB("Svces","EU_28","ROW") = 564668.0625 ;
VMSB("Svces","EGY","USA") = 2516.6015625 ;
VMSB("Svces","EGY","EU_28") = 6072.31982421875 ;
VMSB("Svces","EGY","EGY") = 1.99999994947575E-5 ;
VMSB("Svces","EGY","ROW") = 7648.29248046875 ;
VMSB("Svces","ROW","USA") = 350203.4375 ;
VMSB("Svces","ROW","EU_28") = 467403.15625 ;
VMSB("Svces","ROW","EGY") = 4548.25537109375 ;
VMSB("Svces","ROW","ROW") = 1019812.875 ;
parameter VST(*,*) ;
VST("Svces","USA") = 31137.61328125 ;
VST("Svces","EU_28") = 275190.5 ;
VST("Svces","EGY") = 3017.31274414062 ;
VST("Svces","ROW") = 256843.84375 ;
parameter VTWR(*,*,*,*) ;
VTWR("Svces","Food","USA","EU_28") = 693.843383789062 ;
VTWR("Svces","Food","USA","EGY") = 56.6893882751465 ;
VTWR("Svces","Food","USA","ROW") = 7855.55224609375 ;
VTWR("Svces","Food","EU_28","USA") = 1529.41967773438 ;
VTWR("Svces","Food","EU_28","EU_28") = 19886.89453125 ;
VTWR("Svces","Food","EU_28","EGY") = 140.619689941406 ;
VTWR("Svces","Food","EU_28","ROW") = 8558.703125 ;
VTWR("Svces","Food","EGY","USA") = 8.42541408538818 ;
VTWR("Svces","Food","EGY","EU_28") = 114.666870117188 ;
VTWR("Svces","Food","EGY","ROW") = 425.528442382812 ;
VTWR("Svces","Food","ROW","USA") = 6190.25146484375 ;
VTWR("Svces","Food","ROW","EU_28") = 8750.55859375 ;
VTWR("Svces","Food","ROW","EGY") = 755.040100097656 ;
VTWR("Svces","Food","ROW","ROW") = 40881.1640625 ;
VTWR("Svces","Mnfcs","USA","EU_28") = 6716.0673828125 ;
VTWR("Svces","Mnfcs","USA","EGY") = 134.330215454102 ;
VTWR("Svces","Mnfcs","USA","ROW") = 25618.42578125 ;
VTWR("Svces","Mnfcs","EU_28","USA") = 9902.892578125 ;
VTWR("Svces","Mnfcs","EU_28","EU_28") = 61037.859375 ;
VTWR("Svces","Mnfcs","EU_28","EGY") = 661.171508789062 ;
VTWR("Svces","Mnfcs","EU_28","ROW") = 43359.70703125 ;
VTWR("Svces","Mnfcs","EGY","USA") = 82.7985610961914 ;
VTWR("Svces","Mnfcs","EGY","EU_28") = 345.983673095703 ;
VTWR("Svces","Mnfcs","EGY","ROW") = 691.876525878906 ;
VTWR("Svces","Mnfcs","ROW","USA") = 46560.3125 ;
VTWR("Svces","Mnfcs","ROW","EU_28") = 53963.73046875 ;
VTWR("Svces","Mnfcs","ROW","EGY") = 1704.33911132812 ;
VTWR("Svces","Mnfcs","ROW","ROW") = 219563.078125 ;
parameter SAVE(*) ;
SAVE("USA") = 1580785.625 ;
SAVE("EU_28") = 2133911.25 ;
SAVE("EGY") = -12335.3984375 ;
SAVE("ROW") = 8488486 ;
parameter VDEP(*) ;
VDEP("USA") = 1816268 ;
VDEP("EU_28") = 1809763.25 ;
VDEP("EGY") = 14144.015625 ;
VDEP("ROW") = 4907885.5 ;
parameter VKB(*) ;
VKB("USA") = 45406700 ;
VKB("EU_28") = 45244080 ;
VKB("EGY") = 353600.40625 ;
VKB("ROW") = 122697136 ;
parameter pop0(*) ;
pop0("USA") = 325.147125244141 ;
pop0("EU_28") = 513.872253417969 ;
pop0("EGY") = 96.4425964355469 ;
pop0("ROW") = 6578.421875 ;
parameter MAKS(*,*,*) ;
MAKS("Food","Food","USA") = 1579235.5 ;
MAKS("Food","Food","EU_28") = 1987133.75 ;
MAKS("Food","Food","EGY") = 81724.1875 ;
MAKS("Food","Food","ROW") = 9311410 ;
MAKS("Mnfcs","Mnfcs","USA") = 5737490.5 ;
MAKS("Mnfcs","Mnfcs","EU_28") = 8537181 ;
MAKS("Mnfcs","Mnfcs","EGY") = 137289.46875 ;
MAKS("Mnfcs","Mnfcs","ROW") = 29989802 ;
MAKS("Svces","Svces","USA") = 24752234 ;
MAKS("Svces","Svces","EU_28") = 23014476 ;
MAKS("Svces","Svces","EGY") = 230587.375 ;
MAKS("Svces","Svces","ROW") = 51459868 ;
parameter MAKB(*,*,*) ;
MAKB("Food","Food","USA") = 1620585.375 ;
MAKB("Food","Food","EU_28") = 1987113.625 ;
MAKB("Food","Food","EGY") = 81721.7109375 ;
MAKB("Food","Food","ROW") = 9349964 ;
MAKB("Mnfcs","Mnfcs","USA") = 5860736 ;
MAKB("Mnfcs","Mnfcs","EU_28") = 8561450 ;
MAKB("Mnfcs","Mnfcs","EGY") = 137298.671875 ;
MAKB("Mnfcs","Mnfcs","ROW") = 30098780 ;
MAKB("Svces","Svces","USA") = 25808800 ;
MAKB("Svces","Svces","EU_28") = 23215892 ;
MAKB("Svces","Svces","EGY") = 230546.65625 ;
MAKB("Svces","Svces","ROW") = 51896388 ;
parameter PTAX(*,*,*) ;
PTAX("Food","Food","USA") = 41349.8828125 ;
PTAX("Food","Food","EU_28") = -20.1218852996826 ;
PTAX("Food","Food","EGY") = -2.48143577575684 ;
PTAX("Food","Food","ROW") = 38554.33203125 ;
PTAX("Mnfcs","Mnfcs","USA") = 123245.5078125 ;
PTAX("Mnfcs","Mnfcs","EU_28") = 24269.640625 ;
PTAX("Mnfcs","Mnfcs","EGY") = 9.2112340927124 ;
PTAX("Mnfcs","Mnfcs","ROW") = 108976.9375 ;
PTAX("Svces","Svces","USA") = 1056565.5 ;
PTAX("Svces","Svces","EU_28") = 201416.421875 ;
PTAX("Svces","Svces","EGY") = -40.7100219726562 ;
PTAX("Svces","Svces","ROW") = 436516.65625 ;
parameter FBEP(*,*,*) ;
FBEP("Land","Food","USA") = -9419.345703125 ;
FBEP("Land","Food","EU_28") = -24050.8515625 ;
FBEP("Land","Food","ROW") = -36261.2265625 ;
FBEP("UnSkLab","Food","USA") = -2072.49829101562 ;
FBEP("UnSkLab","Food","EU_28") = -23071.521484375 ;
FBEP("UnSkLab","Food","ROW") = -12537.58203125 ;
FBEP("SkLab","Food","USA") = -2023.03271484375 ;
FBEP("SkLab","Food","EU_28") = -7168.943359375 ;
FBEP("SkLab","Food","ROW") = -1042.6748046875 ;
FBEP("Capital","Food","USA") = -3635.93798828125 ;
FBEP("Capital","Food","EU_28") = -18409.455078125 ;
FBEP("Capital","Food","ROW") = -27742.77734375 ;
parameter FTRV(*,*,*) ;
FTRV("Land","Food","USA") = 2070.95068359375 ;
FTRV("Land","Food","EU_28") = 2027.68518066406 ;
FTRV("Land","Food","EGY") = 9.74806976318359 ;
FTRV("Land","Food","ROW") = 7288.43798828125 ;
FTRV("UnSkLab","Food","USA") = 24356.072265625 ;
FTRV("UnSkLab","Food","EU_28") = 69758.6171875 ;
FTRV("UnSkLab","Food","EGY") = 272.248657226562 ;
FTRV("UnSkLab","Food","ROW") = 88057.8671875 ;
FTRV("UnSkLab","Mnfcs","USA") = 143519.59375 ;
FTRV("UnSkLab","Mnfcs","EU_28") = 204866.109375 ;
FTRV("UnSkLab","Mnfcs","EGY") = 276.036468505859 ;
FTRV("UnSkLab","Mnfcs","ROW") = 201215.453125 ;
FTRV("UnSkLab","Svces","USA") = 676310.5625 ;
FTRV("UnSkLab","Svces","EU_28") = 766386.5 ;
FTRV("UnSkLab","Svces","EGY") = 1594.59790039062 ;
FTRV("UnSkLab","Svces","ROW") = 783752.4375 ;
FTRV("SkLab","Food","USA") = 12469.6962890625 ;
FTRV("SkLab","Food","EU_28") = 40294.38671875 ;
FTRV("SkLab","Food","EGY") = 401.139953613281 ;
FTRV("SkLab","Food","ROW") = 19234.052734375 ;
FTRV("SkLab","Mnfcs","USA") = 47483.7578125 ;
FTRV("SkLab","Mnfcs","EU_28") = 210499.078125 ;
FTRV("SkLab","Mnfcs","EGY") = 611.792785644531 ;
FTRV("SkLab","Mnfcs","ROW") = 105724.3359375 ;
FTRV("SkLab","Svces","USA") = 773199.6875 ;
FTRV("SkLab","Svces","EU_28") = 1092328.375 ;
FTRV("SkLab","Svces","EGY") = 2338.74072265625 ;
FTRV("SkLab","Svces","ROW") = 710051.5625 ;
FTRV("Capital","Food","USA") = 11290.5205078125 ;
FTRV("Capital","Food","EU_28") = 10041.8818359375 ;
FTRV("Capital","Food","EGY") = 13.8189945220947 ;
FTRV("Capital","Food","ROW") = 19315.291015625 ;
FTRV("Capital","Mnfcs","USA") = 48505.62109375 ;
FTRV("Capital","Mnfcs","EU_28") = 29975.994140625 ;
FTRV("Capital","Mnfcs","EGY") = 34.7867660522461 ;
FTRV("Capital","Mnfcs","ROW") = 72055.765625 ;
FTRV("Capital","Svces","USA") = 302947 ;
FTRV("Capital","Svces","EU_28") = 189786.78125 ;
FTRV("Capital","Svces","EGY") = 59.8595695495605 ;
FTRV("Capital","Svces","ROW") = 234658.5625 ;
FTRV("NatRes","Food","USA") = 264.753204345703 ;
FTRV("NatRes","Food","EU_28") = 296.817565917969 ;
FTRV("NatRes","Food","EGY") = 0.951987862586975 ;
FTRV("NatRes","Food","ROW") = 1317.50964355469 ;
FTRV("NatRes","Mnfcs","USA") = 3799.88818359375 ;
FTRV("NatRes","Mnfcs","EU_28") = 604.581787109375 ;
FTRV("NatRes","Mnfcs","EGY") = 3.81978178024292 ;
FTRV("NatRes","Mnfcs","ROW") = 7695.25439453125 ;
parameter VMTS(*,*,*,*) ;
VMTS("ADV","Food","USA","EU_28") = 411.712921142578 ;
VMTS("ADV","Food","USA","EGY") = 434.450653076172 ;
VMTS("ADV","Food","USA","ROW") = 9475.6220703125 ;
VMTS("ADV","Food","EU_28","USA") = 329.060546875 ;
VMTS("ADV","Food","EU_28","EGY") = 604.819458007812 ;
VMTS("ADV","Food","EU_28","ROW") = 12795.5478515625 ;
VMTS("ADV","Food","EGY","USA") = 1.56542658805847 ;
VMTS("ADV","Food","EGY","EU_28") = 8.60965538024902 ;
VMTS("ADV","Food","EGY","ROW") = 177.982025146484 ;
VMTS("ADV","Food","ROW","USA") = 1182.42053222656 ;
VMTS("ADV","Food","ROW","EU_28") = 3722.5419921875 ;
VMTS("ADV","Food","ROW","EGY") = 523.773559570312 ;
VMTS("ADV","Food","ROW","ROW") = 41442.00390625 ;
VMTS("ADV","Mnfcs","USA","EU_28") = 4593.53515625 ;
VMTS("ADV","Mnfcs","USA","EGY") = 199.270248413086 ;
VMTS("ADV","Mnfcs","USA","ROW") = 21965.71875 ;
VMTS("ADV","Mnfcs","EU_28","USA") = 5621.17333984375 ;
VMTS("ADV","Mnfcs","EU_28","EGY") = 392.193908691406 ;
VMTS("ADV","Mnfcs","EU_28","ROW") = 53889.62109375 ;
VMTS("ADV","Mnfcs","EGY","USA") = 130.536682128906 ;
VMTS("ADV","Mnfcs","EGY","ROW") = 223.935989379883 ;
VMTS("ADV","Mnfcs","ROW","USA") = 25775.126953125 ;
VMTS("ADV","Mnfcs","ROW","EU_28") = 24207.943359375 ;
VMTS("ADV","Mnfcs","ROW","EGY") = 2466.4111328125 ;
VMTS("ADV","Mnfcs","ROW","ROW") = 157517.625 ;
VMTS("ADV","Svces","USA","ROW") = 1.05791878700256 ;
VMTS("ADV","Svces","EU_28","ROW") = 0.058397714048624 ;
VMTS("ADV","Svces","ROW","ROW") = 35.4154167175293 ;
VMTS("SPE","Food","USA","EU_28") = 250.591720581055 ;
VMTS("SPE","Food","USA","EGY") = 1.16699194908142 ;
VMTS("SPE","Food","USA","ROW") = 1098.26879882812 ;
VMTS("SPE","Food","EU_28","USA") = 1050.08081054688 ;
VMTS("SPE","Food","EU_28","EGY") = 3.41647744178772 ;
VMTS("SPE","Food","EU_28","ROW") = 4823.1005859375 ;
VMTS("SPE","Food","EGY","USA") = 0.851125717163086 ;
VMTS("SPE","Food","EGY","EU_28") = 9.06014060974121 ;
VMTS("SPE","Food","EGY","ROW") = 34.5627174377441 ;
VMTS("SPE","Food","ROW","USA") = 685.653686523438 ;
VMTS("SPE","Food","ROW","EU_28") = 2220.97338867188 ;
VMTS("SPE","Food","ROW","EGY") = 40.9097518920898 ;
VMTS("SPE","Food","ROW","ROW") = 7910.46044921875 ;
VMTS("SPE","Mnfcs","USA","EU_28") = 3.28220987319946 ;
VMTS("SPE","Mnfcs","USA","ROW") = 451.809448242188 ;
VMTS("SPE","Mnfcs","EU_28","USA") = 60.3619689941406 ;
VMTS("SPE","Mnfcs","EU_28","ROW") = 2288.72485351562 ;
VMTS("SPE","Mnfcs","EGY","USA") = 1.04360902309418 ;
VMTS("SPE","Mnfcs","EGY","ROW") = 10.3913917541504 ;
VMTS("SPE","Mnfcs","ROW","USA") = 502.868316650391 ;
VMTS("SPE","Mnfcs","ROW","EU_28") = 5.29240608215332 ;
VMTS("SPE","Mnfcs","ROW","ROW") = 4751.77978515625 ;
parameter TFRV(*,*,*) ;
TFRV("Food","USA","EU_28") = 662.304626464844 ;
TFRV("Food","USA","EGY") = 435.617645263672 ;
TFRV("Food","USA","ROW") = 10573.890625 ;
TFRV("Food","EU_28","USA") = 1379.14135742188 ;
TFRV("Food","EU_28","EGY") = 608.235961914062 ;
TFRV("Food","EU_28","ROW") = 17618.6484375 ;
TFRV("Food","EGY","USA") = 2.41655230522156 ;
TFRV("Food","EGY","EU_28") = 17.6697959899902 ;
TFRV("Food","EGY","ROW") = 212.544738769531 ;
TFRV("Food","ROW","USA") = 1868.07421875 ;
TFRV("Food","ROW","EU_28") = 5943.51513671875 ;
TFRV("Food","ROW","EGY") = 564.683288574219 ;
TFRV("Food","ROW","ROW") = 49352.46484375 ;
TFRV("Mnfcs","USA","EU_28") = 4596.8173828125 ;
TFRV("Mnfcs","USA","EGY") = 199.270248413086 ;
TFRV("Mnfcs","USA","ROW") = 22417.529296875 ;
TFRV("Mnfcs","EU_28","USA") = 5681.53515625 ;
TFRV("Mnfcs","EU_28","EGY") = 392.193908691406 ;
TFRV("Mnfcs","EU_28","ROW") = 56178.34765625 ;
TFRV("Mnfcs","EGY","USA") = 131.580291748047 ;
TFRV("Mnfcs","EGY","ROW") = 234.327392578125 ;
TFRV("Mnfcs","ROW","USA") = 26277.99609375 ;
TFRV("Mnfcs","ROW","EU_28") = 24213.236328125 ;
TFRV("Mnfcs","ROW","EGY") = 2466.4111328125 ;
TFRV("Mnfcs","ROW","ROW") = 162269.40625 ;
TFRV("Svces","USA","ROW") = 1.05791878700256 ;
TFRV("Svces","EU_28","ROW") = 0.058397714048624 ;
TFRV("Svces","ROW","ROW") = 35.4154167175293 ;
parameter ISEP(*,*,*) ;
ISEP("Food","EU_28","domestic") = -204.585494995117 ;
ISEP("Food","EU_28","imported") = 29.2969245910645 ;
ISEP("Food","EGY","domestic") = -1.9073486328125E-6 ;
ISEP("Food","EGY","imported") = 0.00218886509537697 ;
ISEP("Food","ROW","domestic") = 805.383972167969 ;
ISEP("Food","ROW","imported") = 163.755493164062 ;
ISEP("Mnfcs","USA","domestic") = 9876.6533203125 ;
ISEP("Mnfcs","USA","imported") = 17661.12890625 ;
ISEP("Mnfcs","EU_28","domestic") = 17536.20703125 ;
ISEP("Mnfcs","EU_28","imported") = 32220.43359375 ;
ISEP("Mnfcs","EGY","domestic") = 0.0147075653076172 ;
ISEP("Mnfcs","EGY","imported") = 86.3313598632812 ;
ISEP("Mnfcs","ROW","domestic") = 120515.0546875 ;
ISEP("Mnfcs","ROW","imported") = 101097.5546875 ;
ISEP("Svces","USA","domestic") = -180331.6875 ;
ISEP("Svces","USA","imported") = -11391.1064453125 ;
ISEP("Svces","EU_28","domestic") = 180479 ;
ISEP("Svces","EU_28","imported") = 7006.3828125 ;
ISEP("Svces","EGY","domestic") = 1.32464909553528 ;
ISEP("Svces","EGY","imported") = 0.168954372406006 ;
ISEP("Svces","ROW","domestic") = 290397.875 ;
ISEP("Svces","ROW","imported") = 3906.98291015625 ;
parameter CSEP(*,*,*,*) ;
CSEP("Food","Food","USA","domestic") = -185.501815795898 ;
CSEP("Food","Food","USA","imported") = 90.2366485595703 ;
CSEP("Food","Food","EU_28","domestic") = -3619.49877929688 ;
CSEP("Food","Food","EU_28","imported") = 1844.29028320312 ;
CSEP("Food","Food","EGY","domestic") = -0.138179212808609 ;
CSEP("Food","Food","EGY","imported") = 1.20898973941803 ;
CSEP("Food","Food","ROW","domestic") = -5233.88427734375 ;
CSEP("Food","Food","ROW","imported") = 10291.6376953125 ;
CSEP("Food","Mnfcs","USA","domestic") = -175.340866088867 ;
CSEP("Food","Mnfcs","USA","imported") = -2.5024893283844 ;
CSEP("Food","Mnfcs","EU_28","domestic") = 404.840637207031 ;
CSEP("Food","Mnfcs","EU_28","imported") = 716.980590820312 ;
CSEP("Food","Mnfcs","EGY","domestic") = 1.26693606376648 ;
CSEP("Food","Mnfcs","EGY","imported") = 2.14114117622375 ;
CSEP("Food","Mnfcs","ROW","domestic") = 5865.19677734375 ;
CSEP("Food","Mnfcs","ROW","imported") = 2536.43603515625 ;
CSEP("Food","Svces","USA","domestic") = 517.655456542969 ;
CSEP("Food","Svces","USA","imported") = 77.5098190307617 ;
CSEP("Food","Svces","EU_28","domestic") = 16937.10546875 ;
CSEP("Food","Svces","EU_28","imported") = 7041.04345703125 ;
CSEP("Food","Svces","EGY","domestic") = -5.6068058013916 ;
CSEP("Food","Svces","EGY","imported") = -2.26516079902649 ;
CSEP("Food","Svces","ROW","domestic") = 33287.6953125 ;
CSEP("Food","Svces","ROW","imported") = 9043.474609375 ;
CSEP("Mnfcs","Food","USA","domestic") = 2043.39721679688 ;
CSEP("Mnfcs","Food","USA","imported") = 644.882873535156 ;
CSEP("Mnfcs","Food","EU_28","domestic") = 4930.15625 ;
CSEP("Mnfcs","Food","EU_28","imported") = 2642.51049804688 ;
CSEP("Mnfcs","Food","EGY","domestic") = -280.271911621094 ;
CSEP("Mnfcs","Food","EGY","imported") = -131.945251464844 ;
CSEP("Mnfcs","Food","ROW","domestic") = 8155.32470703125 ;
CSEP("Mnfcs","Food","ROW","imported") = 2730.90625 ;
CSEP("Mnfcs","Mnfcs","USA","domestic") = 57635.8515625 ;
CSEP("Mnfcs","Mnfcs","USA","imported") = 23412.701171875 ;
CSEP("Mnfcs","Mnfcs","EU_28","domestic") = 34634.7265625 ;
CSEP("Mnfcs","Mnfcs","EU_28","imported") = 35364.0625 ;
CSEP("Mnfcs","Mnfcs","EGY","domestic") = -815.6162109375 ;
CSEP("Mnfcs","Mnfcs","EGY","imported") = -684.504943847656 ;
CSEP("Mnfcs","Mnfcs","ROW","domestic") = 304596.34375 ;
CSEP("Mnfcs","Mnfcs","ROW","imported") = 167371.40625 ;
CSEP("Mnfcs","Svces","USA","domestic") = 64526.64453125 ;
CSEP("Mnfcs","Svces","USA","imported") = 25075.982421875 ;
CSEP("Mnfcs","Svces","EU_28","domestic") = 139543.65625 ;
CSEP("Mnfcs","Svces","EU_28","imported") = 114026.75 ;
CSEP("Mnfcs","Svces","EGY","domestic") = -5371.72705078125 ;
CSEP("Mnfcs","Svces","EGY","imported") = -2442.7236328125 ;
CSEP("Mnfcs","Svces","ROW","domestic") = 192756.546875 ;
CSEP("Mnfcs","Svces","ROW","imported") = 82096.34375 ;
CSEP("Svces","Food","USA","domestic") = -605.782470703125 ;
CSEP("Svces","Food","USA","imported") = -5.72422218322754 ;
CSEP("Svces","Food","EU_28","domestic") = 3979.51416015625 ;
CSEP("Svces","Food","EU_28","imported") = 268.456512451172 ;
CSEP("Svces","Food","EGY","domestic") = -598.36181640625 ;
CSEP("Svces","Food","EGY","imported") = 0.63614410161972 ;
CSEP("Svces","Food","ROW","domestic") = -6634.46630859375 ;
CSEP("Svces","Food","ROW","imported") = 97.8528442382812 ;
CSEP("Svces","Mnfcs","USA","domestic") = 16741.787109375 ;
CSEP("Svces","Mnfcs","USA","imported") = 214.627288818359 ;
CSEP("Svces","Mnfcs","EU_28","domestic") = 45273.55859375 ;
CSEP("Svces","Mnfcs","EU_28","imported") = 3574.1845703125 ;
CSEP("Svces","Mnfcs","EGY","domestic") = -2271.16040039062 ;
CSEP("Svces","Mnfcs","EGY","imported") = 0.446901887655258 ;
CSEP("Svces","Mnfcs","ROW","domestic") = 63893.90625 ;
CSEP("Svces","Mnfcs","ROW","imported") = 4263.318359375 ;
CSEP("Svces","Svces","USA","domestic") = 63696.56640625 ;
CSEP("Svces","Svces","USA","imported") = 3401.10791015625 ;
CSEP("Svces","Svces","EU_28","domestic") = 225175.375 ;
CSEP("Svces","Svces","EU_28","imported") = 42319.68359375 ;
CSEP("Svces","Svces","EGY","domestic") = -3147.94799804688 ;
CSEP("Svces","Svces","EGY","imported") = 12.7291507720947 ;
CSEP("Svces","Svces","ROW","domestic") = 244105.484375 ;
CSEP("Svces","Svces","ROW","imported") = 21462.42578125 ;
parameter XTRV(*,*,*) ;
XTRV("Food","USA","EU_28") = 28.4051189422607 ;
XTRV("Food","USA","EGY") = 5.53082323074341 ;
XTRV("Food","USA","ROW") = 292.702178955078 ;
XTRV("Food","EU_28","USA") = 1.60645747184753 ;
XTRV("Food","EU_28","EGY") = 0.362512499094009 ;
XTRV("Food","EU_28","ROW") = -1.39634692668915 ;
XTRV("Food","EGY","USA") = 0.000417490780819207 ;
XTRV("Food","EGY","EU_28") = 0.011446095071733 ;
XTRV("Food","EGY","ROW") = 0.0895593389868736 ;
XTRV("Food","ROW","USA") = -9.19455051422119 ;
XTRV("Food","ROW","EU_28") = 34.9464569091797 ;
XTRV("Food","ROW","EGY") = -1.75934886932373 ;
XTRV("Food","ROW","ROW") = 803.904541015625 ;
XTRV("Mnfcs","USA","EU_28") = 11831.1728515625 ;
XTRV("Mnfcs","USA","EGY") = 141.193756103516 ;
XTRV("Mnfcs","USA","ROW") = 26386.087890625 ;
XTRV("Mnfcs","EU_28","USA") = 926.1416015625 ;
XTRV("Mnfcs","EU_28","EGY") = 91.8727493286133 ;
XTRV("Mnfcs","EU_28","ROW") = 2954.51953125 ;
XTRV("Mnfcs","EGY","USA") = -0.084558442234993 ;
XTRV("Mnfcs","EGY","EU_28") = -0.359088629484177 ;
XTRV("Mnfcs","EGY","ROW") = -0.2153400182724 ;
XTRV("Mnfcs","ROW","USA") = 13421.9697265625 ;
XTRV("Mnfcs","ROW","EU_28") = 15727.794921875 ;
XTRV("Mnfcs","ROW","EGY") = 423.752471923828 ;
XTRV("Mnfcs","ROW","ROW") = 55688.03515625 ;
parameter ADRV(*,*,*) ;
parameter VRRV(*,*,*) ;
parameter MFRV(*,*,*) ;
parameter PURV(*,*,*) ;
parameter rTO(*,*,*) ;
rTO("Food","Food","USA") = 2.61834764480591 ;
rTO("Food","Food","EU_28") = -0.00101276522036642 ;
rTO("Food","Food","EGY") = -0.00303039117716253 ;
rTO("Food","Food","ROW") = 0.41405114531517 ;
rTO("Mnfcs","Mnfcs","USA") = 2.14807319641113 ;
rTO("Mnfcs","Mnfcs","EU_28") = 0.284274160861969 ;
rTO("Mnfcs","Mnfcs","EGY") = 0.00670344568789005 ;
rTO("Mnfcs","Mnfcs","ROW") = 0.363383531570435 ;
rTO("Svces","Svces","USA") = 4.26856803894043 ;
rTO("Svces","Svces","EU_28") = 0.875170886516571 ;
rTO("Svces","Svces","EGY") = -0.017658706754446 ;
rTO("Svces","Svces","ROW") = 0.848272681236267 ;
parameter rTIN(*,*,*) ;
rTIN("Land","Food","USA") = 4.24000978469849 ;
rTIN("Land","Food","EU_28") = 5.78982210159302 ;
rTIN("Land","Food","EGY") = 7.52787446975708 ;
rTIN("Land","Food","ROW") = 9.05721092224121 ;
rTIN("UnSkLab","Food","USA") = 32.8079719543457 ;
rTIN("UnSkLab","Food","EU_28") = 43.4039878845215 ;
rTIN("UnSkLab","Food","EGY") = 3.3516788482666 ;
rTIN("UnSkLab","Food","ROW") = 6.20941114425659 ;
rTIN("UnSkLab","Mnfcs","USA") = 32.8079643249512 ;
rTIN("UnSkLab","Mnfcs","EU_28") = 45.2126274108887 ;
rTIN("UnSkLab","Mnfcs","EGY") = 3.35168099403381 ;
rTIN("UnSkLab","Mnfcs","ROW") = 7.60815238952637 ;
rTIN("UnSkLab","Svces","USA") = 32.8079719543457 ;
rTIN("UnSkLab","Svces","EU_28") = 41.1449508666992 ;
rTIN("UnSkLab","Svces","EGY") = 3.35167527198792 ;
rTIN("UnSkLab","Svces","ROW") = 9.23511123657227 ;
rTIN("SkLab","Food","USA") = 32.8079528808594 ;
rTIN("SkLab","Food","EU_28") = 44.2389297485352 ;
rTIN("SkLab","Food","EGY") = 3.35167574882507 ;
rTIN("SkLab","Food","ROW") = 10.3845081329346 ;
rTIN("SkLab","Mnfcs","USA") = 32.8079605102539 ;
rTIN("SkLab","Mnfcs","EU_28") = 44.2168998718262 ;
rTIN("SkLab","Mnfcs","EGY") = 3.35167694091797 ;
rTIN("SkLab","Mnfcs","ROW") = 11.4271087646484 ;
rTIN("SkLab","Svces","USA") = 32.8079605102539 ;
rTIN("SkLab","Svces","EU_28") = 42.1686782836914 ;
rTIN("SkLab","Svces","EGY") = 3.35167360305786 ;
rTIN("SkLab","Svces","ROW") = 12.1815633773804 ;
rTIN("Capital","Food","USA") = 4.24002408981323 ;
rTIN("Capital","Food","EU_28") = 6.17737770080566 ;
rTIN("Capital","Food","EGY") = 7.52789831161499 ;
rTIN("Capital","Food","ROW") = 8.47124099731445 ;
rTIN("Capital","Mnfcs","USA") = 4.24002313613892 ;
rTIN("Capital","Mnfcs","EU_28") = 6.12121391296387 ;
rTIN("Capital","Mnfcs","EGY") = 7.52789735794067 ;
rTIN("Capital","Mnfcs","ROW") = 8.74696254730225 ;
rTIN("Capital","Svces","USA") = 4.24001836776733 ;
rTIN("Capital","Svces","EU_28") = 6.39154148101807 ;
rTIN("Capital","Svces","EGY") = 7.52789211273193 ;
rTIN("Capital","Svces","ROW") = 8.79256153106689 ;
rTIN("NatRes","Food","USA") = 4.24001169204712 ;
rTIN("NatRes","Food","EU_28") = 6.53523969650269 ;
rTIN("NatRes","Food","EGY") = 7.52787780761719 ;
rTIN("NatRes","Food","ROW") = 9.38253021240234 ;
rTIN("NatRes","Mnfcs","USA") = 4.24001455307007 ;
rTIN("NatRes","Mnfcs","EU_28") = 6.74959993362427 ;
rTIN("NatRes","Mnfcs","EGY") = 7.5278902053833 ;
rTIN("NatRes","Mnfcs","ROW") = 7.95906209945679 ;
parameter rTFE(*,*,*) ;
rTFE("Land","Food","USA") = -17.770601272583 ;
rTFE("Land","Food","EU_28") = -32.6472282409668 ;
rTFE("Land","Food","EGY") = 0.0863749161362648 ;
rTFE("Land","Food","ROW") = -4.40338134765625 ;
rTFE("UnSkLab","Food","USA") = 18.6247367858887 ;
rTFE("UnSkLab","Food","EU_28") = 32.841438293457 ;
rTFE("UnSkLab","Food","EGY") = 5.14173316955566 ;
rTFE("UnSkLab","Food","ROW") = 4.98822021484375 ;
rTFE("UnSkLab","Mnfcs","USA") = 20.3569450378418 ;
rTFE("UnSkLab","Mnfcs","EU_28") = 46.5762596130371 ;
rTFE("UnSkLab","Mnfcs","EGY") = 5.14171981811523 ;
rTFE("UnSkLab","Mnfcs","ROW") = 9.03256511688232 ;
rTFE("UnSkLab","Svces","USA") = 20.3569450378418 ;
rTFE("UnSkLab","Svces","EU_28") = 42.5594635009766 ;
rTFE("UnSkLab","Svces","EGY") = 5.14172554016113 ;
rTFE("UnSkLab","Svces","ROW") = 10.7757530212402 ;
rTFE("SkLab","Food","USA") = 17.0543193817139 ;
rTFE("SkLab","Food","EU_28") = 36.6337432861328 ;
rTFE("SkLab","Food","EGY") = 5.14172887802124 ;
rTFE("SkLab","Food","ROW") = 9.21229267120361 ;
rTFE("SkLab","Mnfcs","USA") = 20.3569412231445 ;
rTFE("SkLab","Mnfcs","EU_28") = 44.1276969909668 ;
rTFE("SkLab","Mnfcs","EGY") = 5.14172744750977 ;
rTFE("SkLab","Mnfcs","ROW") = 13.2227993011475 ;
rTFE("SkLab","Svces","USA") = 20.3569412231445 ;
rTFE("SkLab","Svces","EU_28") = 43.1197624206543 ;
rTFE("SkLab","Svces","EGY") = 5.1417293548584 ;
rTFE("SkLab","Svces","ROW") = 12.3990240097046 ;
rTFE("Capital","Food","USA") = 3.3953652381897 ;
rTFE("Capital","Food","EU_28") = -2.83843731880188 ;
rTFE("Capital","Food","EGY") = 0.086368590593338 ;
rTFE("Capital","Food","ROW") = -0.568938016891479 ;
rTFE("Capital","Mnfcs","USA") = 5.00816488265991 ;
rTFE("Capital","Mnfcs","EU_28") = 2.779860496521 ;
rTFE("Capital","Mnfcs","EGY") = 0.0863721668720245 ;
rTFE("Capital","Mnfcs","ROW") = 1.57298731803894 ;
rTFE("Capital","Svces","USA") = 5.00817155838013 ;
rTFE("Capital","Svces","EU_28") = 3.32537531852722 ;
rTFE("Capital","Svces","EGY") = 0.0863661840558052 ;
rTFE("Capital","Svces","ROW") = 1.7911581993103 ;
rTFE("NatRes","Food","USA") = 5.00816583633423 ;
rTFE("NatRes","Food","EU_28") = 3.53052830696106 ;
rTFE("NatRes","Food","EGY") = 0.0863699615001678 ;
rTFE("NatRes","Food","ROW") = 1.18336176872253 ;
rTFE("NatRes","Mnfcs","USA") = 5.00816440582275 ;
rTFE("NatRes","Mnfcs","EU_28") = 3.00687026977539 ;
rTFE("NatRes","Mnfcs","EGY") = 0.0863674208521843 ;
rTFE("NatRes","Mnfcs","ROW") = 1.35569131374359 ;
parameter rTPD(*,*) ;
rTPD("Food","USA") = -0.211641505360603 ;
rTPD("Food","EU_28") = 28.9802017211914 ;
rTPD("Food","EGY") = 0.000236910811509006 ;
rTPD("Food","ROW") = 4.85663175582886 ;
rTPD("Mnfcs","USA") = 2.47224998474121 ;
rTPD("Mnfcs","EU_28") = 46.9984016418457 ;
rTPD("Mnfcs","EGY") = -8.26227951049805 ;
rTPD("Mnfcs","ROW") = 8.91081428527832 ;
rTPD("Svces","USA") = 2.12937188148499 ;
rTPD("Svces","EU_28") = 5.31095027923584 ;
rTPD("Svces","EGY") = -3.75852632522583 ;
rTPD("Svces","ROW") = 3.04876494407654 ;
parameter rTPM(*,*) ;
rTPM("Food","USA") = -0.143078625202179 ;
rTPM("Food","EU_28") = 27.8759059906006 ;
rTPM("Food","EGY") = 0.0159259364008904 ;
rTPM("Food","ROW") = 12.8119859695435 ;
rTPM("Mnfcs","USA") = 3.46358156204224 ;
rTPM("Mnfcs","EU_28") = 34.2261505126953 ;
rTPM("Mnfcs","EGY") = -0.543186128139496 ;
rTPM("Mnfcs","ROW") = 11.7560615539551 ;
rTPM("Svces","USA") = 2.83282470703125 ;
rTPM("Svces","EU_28") = 8.80367279052734 ;
rTPM("Svces","EGY") = 1.67973697185516 ;
rTPM("Svces","ROW") = 4.66674184799194 ;
parameter rTGD(*,*) ;
rTGD("Food","EU_28") = 3.40332317352295 ;
rTGD("Food","EGY") = 0.0595219135284424 ;
rTGD("Food","ROW") = 0.835884988307953 ;
rTGD("Mnfcs","EU_28") = 14.234564781189 ;
rTGD("Mnfcs","EGY") = -0.0127095198258758 ;
rTGD("Mnfcs","ROW") = 0.792376518249512 ;
rTGD("Svces","USA") = 7.2822961807251 ;
rTGD("Svces","EU_28") = 0.0815900638699532 ;
rTGD("Svces","EGY") = 0.000491965387482196 ;
rTGD("Svces","ROW") = 0.616307556629181 ;
parameter rTGM(*,*) ;
rTGM("Food","EU_28") = 11.2402677536011 ;
rTGM("Food","EGY") = 10.5332183837891 ;
rTGM("Food","ROW") = 1.15783071517944 ;
rTGM("Mnfcs","EU_28") = 12.7361907958984 ;
rTGM("Mnfcs","EGY") = 6.602135181427 ;
rTGM("Mnfcs","ROW") = -0.132056683301926 ;
rTGM("Svces","EU_28") = 34.941650390625 ;
rTGM("Svces","EGY") = 14.6322116851807 ;
rTGM("Svces","ROW") = 5.60090255737305 ;
parameter rTID(*,*) ;
rTID("Food","EU_28") = -3.59165549278259 ;
rTID("Food","ROW") = 0.937341094017029 ;
rTID("Mnfcs","USA") = 1.28949856758118 ;
rTID("Mnfcs","EU_28") = 3.70547580718994 ;
rTID("Mnfcs","EGY") = 0.000350323505699635 ;
rTID("Mnfcs","ROW") = 5.7453932762146 ;
rTID("Svces","USA") = -6.27639818191528 ;
rTID("Svces","EU_28") = 7.99232006072998 ;
rTID("Svces","EGY") = 0.00488910404965281 ;
rTID("Svces","ROW") = 3.27629828453064 ;
parameter rTIM(*,*) ;
rTIM("Food","EU_28") = 1.43047428131104 ;
rTIM("Food","EGY") = 0.296212166547775 ;
rTIM("Food","ROW") = 1.72102034091949 ;
rTIM("Mnfcs","USA") = 3.72771859169006 ;
rTIM("Mnfcs","EU_28") = 5.53156995773315 ;
rTIM("Mnfcs","EGY") = 2.33574247360229 ;
rTIM("Mnfcs","ROW") = 7.83107995986938 ;
rTIM("Svces","USA") = -11.465404510498 ;
rTIM("Svces","EU_28") = 7.5739631652832 ;
rTIM("Svces","EGY") = 0.57152271270752 ;
rTIM("Svces","ROW") = 2.8039128780365 ;
parameter rTFD(*,*,*) ;
rTFD("Food","Food","USA") = -0.034202691167593 ;
rTFD("Food","Food","EU_28") = -0.72297739982605 ;
rTFD("Food","Food","EGY") = -0.000748997961636633 ;
rTFD("Food","Food","ROW") = -0.181514427065849 ;
rTFD("Food","Mnfcs","USA") = -0.42457789182663 ;
rTFD("Food","Mnfcs","EU_28") = 0.838414192199707 ;
rTFD("Food","Mnfcs","EGY") = 0.167978152632713 ;
rTFD("Food","Mnfcs","ROW") = 1.16135215759277 ;
rTFD("Food","Svces","USA") = 0.292181313037872 ;
rTFD("Food","Svces","EU_28") = 6.96176385879517 ;
rTFD("Food","Svces","EGY") = -0.0955618247389793 ;
rTFD("Food","Svces","ROW") = 3.31493473052979 ;
rTFD("Mnfcs","Food","USA") = 1.48749852180481 ;
rTFD("Mnfcs","Food","EU_28") = 4.64242267608643 ;
rTFD("Mnfcs","Food","EGY") = -10.3640460968018 ;
rTFD("Mnfcs","Food","ROW") = 1.69482183456421 ;
rTFD("Mnfcs","Mnfcs","USA") = 3.57565665245056 ;
rTFD("Mnfcs","Mnfcs","EU_28") = 1.88250505924225 ;
rTFD("Mnfcs","Mnfcs","EGY") = -3.0663743019104 ;
rTFD("Mnfcs","Mnfcs","ROW") = 2.73907017707825 ;
rTFD("Mnfcs","Svces","USA") = 5.02595376968384 ;
rTFD("Mnfcs","Svces","EU_28") = 12.5269737243652 ;
rTFD("Mnfcs","Svces","EGY") = -16.0888862609863 ;
rTFD("Mnfcs","Svces","ROW") = 3.50765705108643 ;
rTFD("Svces","Food","USA") = -0.182625040411949 ;
rTFD("Svces","Food","EU_28") = 0.984681367874146 ;
rTFD("Svces","Food","EGY") = -12.8644895553589 ;
rTFD("Svces","Food","ROW") = -0.487635791301727 ;
rTFD("Svces","Mnfcs","USA") = 1.5647337436676 ;
rTFD("Svces","Mnfcs","EU_28") = 2.81936264038086 ;
rTFD("Svces","Mnfcs","EGY") = -11.5676870346069 ;
rTFD("Svces","Mnfcs","ROW") = 1.26619923114777 ;
rTFD("Svces","Svces","USA") = 0.838904678821564 ;
rTFD("Svces","Svces","EU_28") = 3.2743239402771 ;
rTFD("Svces","Svces","EGY") = -12.1170387268066 ;
rTFD("Svces","Svces","ROW") = 1.76446938514709 ;
parameter rTFM(*,*,*) ;
rTFM("Food","Food","USA") = 0.202178135514259 ;
rTFM("Food","Food","EU_28") = 0.97742372751236 ;
rTFM("Food","Food","EGY") = 0.0111934700980783 ;
rTFM("Food","Food","ROW") = 3.16827654838562 ;
rTFM("Food","Mnfcs","USA") = -0.116681449115276 ;
rTFM("Food","Mnfcs","EU_28") = 2.71690893173218 ;
rTFM("Food","Mnfcs","EGY") = 0.292905986309052 ;
rTFM("Food","Mnfcs","ROW") = 5.96854686737061 ;
rTFM("Food","Svces","USA") = 0.336302995681763 ;
rTFM("Food","Svces","EU_28") = 9.60529136657715 ;
rTFM("Food","Svces","EGY") = -0.120186984539032 ;
rTFM("Food","Svces","ROW") = 7.03979635238647 ;
rTFM("Mnfcs","Food","USA") = 2.12148880958557 ;
rTFM("Mnfcs","Food","EU_28") = 3.43892884254456 ;
rTFM("Mnfcs","Food","EGY") = -3.76141285896301 ;
rTFM("Mnfcs","Food","ROW") = 1.44548726081848 ;
rTFM("Mnfcs","Mnfcs","USA") = 3.52199459075928 ;
rTFM("Mnfcs","Mnfcs","EU_28") = 1.61815464496613 ;
rTFM("Mnfcs","Mnfcs","EGY") = -2.58974599838257 ;
rTFM("Mnfcs","Mnfcs","ROW") = 4.24170017242432 ;
rTFM("Mnfcs","Svces","USA") = 6.76545763015747 ;
rTFM("Mnfcs","Svces","EU_28") = 13.0095195770264 ;
rTFM("Mnfcs","Svces","EGY") = -13.3499736785889 ;
rTFM("Mnfcs","Svces","ROW") = 4.91576051712036 ;
rTFM("Svces","Food","USA") = -0.126941859722137 ;
rTFM("Svces","Food","EU_28") = 0.563748836517334 ;
rTFM("Svces","Food","EGY") = 0.100771337747574 ;
rTFM("Svces","Food","ROW") = 0.225604236125946 ;
rTFM("Svces","Mnfcs","USA") = 0.908174753189087 ;
rTFM("Svces","Mnfcs","EU_28") = 1.42850816249847 ;
rTFM("Svces","Mnfcs","EGY") = 0.0111379334703088 ;
rTFM("Svces","Mnfcs","ROW") = 1.94707119464874 ;
rTFM("Svces","Svces","USA") = 1.50885045528412 ;
rTFM("Svces","Svces","EU_28") = 3.53934359550476 ;
rTFM("Svces","Svces","EGY") = 0.201175406575203 ;
rTFM("Svces","Svces","ROW") = 2.34499907493591 ;
parameter rTXS(*,*,*) ;
rTXS("Food","USA","EU_28") = 0.220413371920586 ;
rTXS("Food","USA","EGY") = 0.704790413379669 ;
rTXS("Food","USA","ROW") = 0.206291556358337 ;
rTXS("Food","EU_28","USA") = 0.00625919410958886 ;
rTXS("Food","EU_28","EGY") = 0.0211486257612705 ;
rTXS("Food","EU_28","ROW") = -0.00110386987216771 ;
rTXS("Food","EGY","USA") = 0.000297974824206904 ;
rTXS("Food","EGY","EU_28") = 0.0010921637294814 ;
rTXS("Food","EGY","ROW") = 0.00212470442056656 ;
rTXS("Food","ROW","USA") = -0.00734964618459344 ;
rTXS("Food","ROW","EU_28") = 0.026518952101469 ;
rTXS("Food","ROW","EGY") = -0.0165861435234547 ;
rTXS("Food","ROW","ROW") = 0.145094588398933 ;
rTXS("Mnfcs","USA","EU_28") = 5.25433349609375 ;
rTXS("Mnfcs","USA","EGY") = 5.53526782989502 ;
rTXS("Mnfcs","USA","ROW") = 2.53425025939941 ;
rTXS("Mnfcs","EU_28","USA") = 0.243906408548355 ;
rTXS("Mnfcs","EU_28","EGY") = 0.569092094898224 ;
rTXS("Mnfcs","EU_28","ROW") = 0.206182166934013 ;
rTXS("Mnfcs","EGY","USA") = -0.00509401224553585 ;
rTXS("Mnfcs","EGY","EU_28") = -0.00517567852512002 ;
rTXS("Mnfcs","EGY","ROW") = -0.00160270289052278 ;
rTXS("Mnfcs","ROW","USA") = 0.795265197753906 ;
rTXS("Mnfcs","ROW","EU_28") = 1.05987274646759 ;
rTXS("Mnfcs","ROW","EGY") = 1.28692233562469 ;
rTXS("Mnfcs","ROW","ROW") = 1.03740727901459 ;
parameter rTMS(*,*,*) ;
rTMS("Food","USA","EU_28") = 4.86647939682007 ;
rTMS("Food","USA","EGY") = 51.4326324462891 ;
rTMS("Food","USA","ROW") = 7.04755353927612 ;
rTMS("Food","EU_28","USA") = 5.07395267486572 ;
rTMS("Food","EU_28","EGY") = 32.7841835021973 ;
rTMS("Food","EU_28","ROW") = 13.0274620056152 ;
rTMS("Food","EGY","USA") = 1.64740478992462 ;
rTMS("Food","EGY","EU_28") = 1.51631188392639 ;
rTMS("Food","EGY","ROW") = 4.5667839050293 ;
rTMS("Food","ROW","USA") = 1.42281556129456 ;
rTMS("Food","ROW","EU_28") = 4.22933101654053 ;
rTMS("Food","ROW","EGY") = 4.9694356918335 ;
rTMS("Food","ROW","ROW") = 8.28450012207031 ;
rTMS("Mnfcs","USA","EU_28") = 1.88612735271454 ;
rTMS("Mnfcs","USA","EGY") = 7.0504994392395 ;
rTMS("Mnfcs","USA","ROW") = 2.05067038536072 ;
rTMS("Mnfcs","EU_28","USA") = 1.45476269721985 ;
rTMS("Mnfcs","EU_28","EGY") = 2.32110500335693 ;
rTMS("Mnfcs","EU_28","ROW") = 3.79770827293396 ;
rTMS("Mnfcs","EGY","USA") = 7.54740953445435 ;
rTMS("Mnfcs","EGY","ROW") = 1.66227734088898 ;
rTMS("Mnfcs","ROW","USA") = 1.50355780124664 ;
rTMS("Mnfcs","ROW","EU_28") = 1.55850613117218 ;
rTMS("Mnfcs","ROW","EGY") = 7.0356707572937 ;
rTMS("Mnfcs","ROW","ROW") = 2.87546229362488 ;
rTMS("Svces","USA","ROW") = 0.000221887486986816 ;
rTMS("Svces","ROW","ROW") = 0.00347502296790481 ;

parameter esubd(*,*) ;
esubd("Food","USA") = 2.429511308670044 ;
esubd("Mnfcs","USA") = 3.625655174255371 ;
esubd("Svces","USA") = 1.937635064125061 ;
esubd("Food","EU_28") = 2.429511308670044 ;
esubd("Mnfcs","EU_28") = 3.625655174255371 ;
esubd("Svces","EU_28") = 1.937635064125061 ;
esubd("Food","EGY") = 2.429511308670044 ;
esubd("Mnfcs","EGY") = 3.625655174255371 ;
esubd("Svces","EGY") = 1.937635064125061 ;
esubd("Food","ROW") = 2.429511308670044 ;
esubd("Mnfcs","ROW") = 3.625655174255371 ;
esubd("Svces","ROW") = 1.937635064125061 ;
parameter esubm(*,*) ;
esubm("Food","USA") = 4.764378547668457 ;
esubm("Mnfcs","USA") = 7.873477458953857 ;
esubm("Svces","USA") = 3.8269684314727783 ;
esubm("Food","EU_28") = 4.764378547668457 ;
esubm("Mnfcs","EU_28") = 7.873477458953857 ;
esubm("Svces","EU_28") = 3.8269684314727783 ;
esubm("Food","EGY") = 4.764378547668457 ;
esubm("Mnfcs","EGY") = 7.873477458953857 ;
esubm("Svces","EGY") = 3.8269684314727783 ;
esubm("Food","ROW") = 4.764378547668457 ;
esubm("Mnfcs","ROW") = 7.873477458953857 ;
esubm("Svces","ROW") = 3.8269684314727783 ;
parameter esubt(*,*) ;
parameter esubc(*,*) ;
parameter esubva(*,*) ;
esubva("Food","USA") = 0.5901468396186829 ;
esubva("Mnfcs","USA") = 1.0767656564712524 ;
esubva("Svces","USA") = 1.3697344064712524 ;
esubva("Food","EU_28") = 0.5901468396186829 ;
esubva("Mnfcs","EU_28") = 1.0767656564712524 ;
esubva("Svces","EU_28") = 1.3697344064712524 ;
esubva("Food","EGY") = 0.5901468396186829 ;
esubva("Mnfcs","EGY") = 1.0767656564712524 ;
esubva("Svces","EGY") = 1.3697344064712524 ;
esubva("Food","ROW") = 0.5901468396186829 ;
esubva("Mnfcs","ROW") = 1.0767656564712524 ;
esubva("Svces","ROW") = 1.3697344064712524 ;
parameter esubq(*,*) ;
parameter esubg(*) ;
esubg("USA") = 1.0 ;
esubg("EU_28") = 1.0 ;
esubg("EGY") = 1.0 ;
esubg("ROW") = 1.0 ;
parameter esubi(*) ;
parameter esubs(*) ;
esubs("Svces") = 1.0 ;
parameter etraq(*,*) ;
etraq("Food","USA") = -5.0 ;
etraq("Mnfcs","USA") = -5.0 ;
etraq("Svces","USA") = -5.0 ;
etraq("Food","EU_28") = -5.0 ;
etraq("Mnfcs","EU_28") = -5.0 ;
etraq("Svces","EU_28") = -5.0 ;
etraq("Food","EGY") = -5.0 ;
etraq("Mnfcs","EGY") = -5.0 ;
etraq("Svces","EGY") = -5.0 ;
etraq("Food","ROW") = -5.0 ;
etraq("Mnfcs","ROW") = -5.0 ;
etraq("Svces","ROW") = -5.0 ;
parameter etrae(*,*) ;
etrae("Land","USA") = -1.0 ;
etrae("NatRes","USA") = -0.0010000000474974513 ;
etrae("Land","EU_28") = -1.0 ;
etrae("NatRes","EU_28") = -0.0010000000474974513 ;
etrae("Land","EGY") = -1.0 ;
etrae("NatRes","EGY") = -0.0010000000474974513 ;
etrae("Land","ROW") = -1.0 ;
etrae("NatRes","ROW") = -0.0010000000474974513 ;
parameter incpar(*,*) ;
incpar("Food","USA") = 0.38072898983955383 ;
incpar("Mnfcs","USA") = 0.8847161531448364 ;
incpar("Svces","USA") = 1.059613585472107 ;
incpar("Food","EU_28") = 0.42059653997421265 ;
incpar("Mnfcs","EU_28") = 0.9065216779708862 ;
incpar("Svces","EU_28") = 1.1218265295028687 ;
incpar("Food","EGY") = 0.5228255987167358 ;
incpar("Mnfcs","EGY") = 0.8387210369110107 ;
incpar("Svces","EGY") = 1.3151606321334839 ;
incpar("Food","ROW") = 0.4366128742694855 ;
incpar("Mnfcs","ROW") = 0.8927760720252991 ;
incpar("Svces","ROW") = 1.2086553573608398 ;
parameter subpar(*,*) ;
subpar("Food","USA") = 0.25939685106277466 ;
subpar("Mnfcs","USA") = 0.100168377161026 ;
subpar("Svces","USA") = 0.08794578909873962 ;
subpar("Food","EU_28") = 0.4622822105884552 ;
subpar("Mnfcs","EU_28") = 0.25133436918258667 ;
subpar("Svces","EU_28") = 0.20687110722064972 ;
subpar("Food","EGY") = 0.8122340440750122 ;
subpar("Mnfcs","EGY") = 0.7324209213256836 ;
subpar("Svces","EGY") = 0.6404356956481934 ;
subpar("Food","ROW") = 0.7156794667243958 ;
subpar("Mnfcs","ROW") = 0.4812253713607788 ;
subpar("Svces","ROW") = 0.37780606746673584 ;
parameter rorFlex0(*) ;
rorFlex0("USA") = 10.0 ;
rorFlex0("EU_28") = 10.0 ;
rorFlex0("EGY") = 10.0 ;
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
   Svces
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
