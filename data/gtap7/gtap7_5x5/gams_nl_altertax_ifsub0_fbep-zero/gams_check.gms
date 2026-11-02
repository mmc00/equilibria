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
   acts           "Activities"      / a_Agri, a_Food, a_Energy, a_Manuf, a_Svces /
   comm           "Commodities"     / c_Agri, c_Food, c_Energy, c_Manuf, c_Svces /
   marg(comm)     "Margin comm"     / c_Svces /
   reg            "Regions"         / USA, EU_28, CHN, LatinAmer, ROW /
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
   a_Agri, a_Food, a_Energy, a_Manuf, a_Svces, c_Agri, c_Food, c_Energy, c_Manuf, c_Svces, Land, UnSkLab, SkLab, Capital, NatRes, TRD, hhd, gov, inv,
   deprY, tmg, itax, ptax, mtax, etax, vtax, vsub, dtax, ctax, bop, tot, USA, EU_28, CHN, LatinAmer, ROW
/ ;
alias(is, js) ;

set aa(is) "Armington agents" / a_Agri, a_Food, a_Energy, a_Manuf, a_Svces, hhd, gov, inv, tmg / ;
set a(aa)  "Activities"       / a_Agri, a_Food, a_Energy, a_Manuf, a_Svces / ;
set i(is)  "Commodities"      / c_Agri, c_Food, c_Energy, c_Manuf, c_Svces / ;
alias(i, j) ;

set r(is)  "Regions" / USA, EU_28, CHN, LatinAmer, ROW / ;
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
vdfb("c_Agri","a_Agri","USA") = 64414.13671875 ;
vdfb("c_Agri","a_Agri","EU_28") = 46679.66015625 ;
vdfb("c_Agri","a_Agri","CHN") = 145540 ;
vdfb("c_Agri","a_Agri","LatinAmer") = 56397.67578125 ;
vdfb("c_Agri","a_Agri","ROW") = 319295.75 ;
vdfb("c_Agri","a_Food","USA") = 210682.421875 ;
vdfb("c_Agri","a_Food","EU_28") = 179794.15625 ;
vdfb("c_Agri","a_Food","CHN") = 520748.6875 ;
vdfb("c_Agri","a_Food","LatinAmer") = 214039.46875 ;
vdfb("c_Agri","a_Food","ROW") = 639240.6875 ;
vdfb("c_Agri","a_Energy","USA") = 112.159149169922 ;
vdfb("c_Agri","a_Energy","EU_28") = 648.309814453125 ;
vdfb("c_Agri","a_Energy","CHN") = 338.43701171875 ;
vdfb("c_Agri","a_Energy","LatinAmer") = 1008.59948730469 ;
vdfb("c_Agri","a_Energy","ROW") = 4964.63671875 ;
vdfb("c_Agri","a_Manuf","USA") = 31437.755859375 ;
vdfb("c_Agri","a_Manuf","EU_28") = 27788.65234375 ;
vdfb("c_Agri","a_Manuf","CHN") = 210230.6875 ;
vdfb("c_Agri","a_Manuf","LatinAmer") = 23706.662109375 ;
vdfb("c_Agri","a_Manuf","ROW") = 124788.1328125 ;
vdfb("c_Agri","a_Svces","USA") = 15139.6455078125 ;
vdfb("c_Agri","a_Svces","EU_28") = 29826.33984375 ;
vdfb("c_Agri","a_Svces","CHN") = 123518.0078125 ;
vdfb("c_Agri","a_Svces","LatinAmer") = 17050.291015625 ;
vdfb("c_Agri","a_Svces","ROW") = 204120.984375 ;
vdfb("c_Food","a_Agri","USA") = 35451.421875 ;
vdfb("c_Food","a_Agri","EU_28") = 41441.8671875 ;
vdfb("c_Food","a_Agri","CHN") = 110312.2734375 ;
vdfb("c_Food","a_Agri","LatinAmer") = 29905.283203125 ;
vdfb("c_Food","a_Agri","ROW") = 87544.4921875 ;
vdfb("c_Food","a_Food","USA") = 231806.984375 ;
vdfb("c_Food","a_Food","EU_28") = 232722.390625 ;
vdfb("c_Food","a_Food","CHN") = 385166.375 ;
vdfb("c_Food","a_Food","LatinAmer") = 98557.96875 ;
vdfb("c_Food","a_Food","ROW") = 295022.125 ;
vdfb("c_Food","a_Energy","USA") = 127.960060119629 ;
vdfb("c_Food","a_Energy","EU_28") = 562.100646972656 ;
vdfb("c_Food","a_Energy","CHN") = 5781.861328125 ;
vdfb("c_Food","a_Energy","LatinAmer") = 269.594818115234 ;
vdfb("c_Food","a_Energy","ROW") = 1664.01196289062 ;
vdfb("c_Food","a_Manuf","USA") = 9685.8017578125 ;
vdfb("c_Food","a_Manuf","EU_28") = 20280.1953125 ;
vdfb("c_Food","a_Manuf","CHN") = 109279.1953125 ;
vdfb("c_Food","a_Manuf","LatinAmer") = 8174.798828125 ;
vdfb("c_Food","a_Manuf","ROW") = 24110.142578125 ;
vdfb("c_Food","a_Svces","USA") = 161958.3125 ;
vdfb("c_Food","a_Svces","EU_28") = 212468.40625 ;
vdfb("c_Food","a_Svces","CHN") = 254889.75 ;
vdfb("c_Food","a_Svces","LatinAmer") = 83192.34375 ;
vdfb("c_Food","a_Svces","ROW") = 318737.5625 ;
vdfb("c_Energy","a_Agri","USA") = 9416.4296875 ;
vdfb("c_Energy","a_Agri","EU_28") = 12788.130859375 ;
vdfb("c_Energy","a_Agri","CHN") = 18587.23046875 ;
vdfb("c_Energy","a_Agri","LatinAmer") = 12756.3603515625 ;
vdfb("c_Energy","a_Agri","ROW") = 83432.78125 ;
vdfb("c_Energy","a_Food","USA") = 12463.21484375 ;
vdfb("c_Energy","a_Food","EU_28") = 18172.408203125 ;
vdfb("c_Energy","a_Food","CHN") = 16410.568359375 ;
vdfb("c_Energy","a_Food","LatinAmer") = 9817.55859375 ;
vdfb("c_Energy","a_Food","ROW") = 38545.0546875 ;
vdfb("c_Energy","a_Energy","USA") = 304586.375 ;
vdfb("c_Energy","a_Energy","EU_28") = 133683.5 ;
vdfb("c_Energy","a_Energy","CHN") = 381460.25 ;
vdfb("c_Energy","a_Energy","LatinAmer") = 129872.421875 ;
vdfb("c_Energy","a_Energy","ROW") = 852198.875 ;
vdfb("c_Energy","a_Manuf","USA") = 122936.3359375 ;
vdfb("c_Energy","a_Manuf","EU_28") = 153472.234375 ;
vdfb("c_Energy","a_Manuf","CHN") = 453216.9375 ;
vdfb("c_Energy","a_Manuf","LatinAmer") = 85415.4765625 ;
vdfb("c_Energy","a_Manuf","ROW") = 670271.375 ;
vdfb("c_Energy","a_Svces","USA") = 330760.90625 ;
vdfb("c_Energy","a_Svces","EU_28") = 232326.625 ;
vdfb("c_Energy","a_Svces","CHN") = 211807.609375 ;
vdfb("c_Energy","a_Svces","LatinAmer") = 98911.3984375 ;
vdfb("c_Energy","a_Svces","ROW") = 656397.75 ;
vdfb("c_Manuf","a_Agri","USA") = 31645.6484375 ;
vdfb("c_Manuf","a_Agri","EU_28") = 25614.109375 ;
vdfb("c_Manuf","a_Agri","CHN") = 108807.84375 ;
vdfb("c_Manuf","a_Agri","LatinAmer") = 42702.26953125 ;
vdfb("c_Manuf","a_Agri","ROW") = 69920.640625 ;
vdfb("c_Manuf","a_Food","USA") = 100344.6953125 ;
vdfb("c_Manuf","a_Food","EU_28") = 73554.4140625 ;
vdfb("c_Manuf","a_Food","CHN") = 81708.9375 ;
vdfb("c_Manuf","a_Food","LatinAmer") = 44638.4375 ;
vdfb("c_Manuf","a_Food","ROW") = 89663.609375 ;
vdfb("c_Manuf","a_Energy","USA") = 38907.51171875 ;
vdfb("c_Manuf","a_Energy","EU_28") = 37477.46875 ;
vdfb("c_Manuf","a_Energy","CHN") = 108589.9296875 ;
vdfb("c_Manuf","a_Energy","LatinAmer") = 21948.73828125 ;
vdfb("c_Manuf","a_Energy","ROW") = 183606.234375 ;
vdfb("c_Manuf","a_Manuf","USA") = 1333651.75 ;
vdfb("c_Manuf","a_Manuf","EU_28") = 1720559.5 ;
vdfb("c_Manuf","a_Manuf","CHN") = 5672547.5 ;
vdfb("c_Manuf","a_Manuf","LatinAmer") = 551436.3125 ;
vdfb("c_Manuf","a_Manuf","ROW") = 3529736.5 ;
vdfb("c_Manuf","a_Svces","USA") = 1064690 ;
vdfb("c_Manuf","a_Svces","EU_28") = 948463.375 ;
vdfb("c_Manuf","a_Svces","CHN") = 2411348.75 ;
vdfb("c_Manuf","a_Svces","LatinAmer") = 321995.46875 ;
vdfb("c_Manuf","a_Svces","ROW") = 1882696.625 ;
vdfb("c_Svces","a_Agri","USA") = 101483.078125 ;
vdfb("c_Svces","a_Agri","EU_28") = 72343.59375 ;
vdfb("c_Svces","a_Agri","CHN") = 124291.2265625 ;
vdfb("c_Svces","a_Agri","LatinAmer") = 55893.90625 ;
vdfb("c_Svces","a_Agri","ROW") = 260708.8125 ;
vdfb("c_Svces","a_Food","USA") = 213726.578125 ;
vdfb("c_Svces","a_Food","EU_28") = 307866.125 ;
vdfb("c_Svces","a_Food","CHN") = 242868.8125 ;
vdfb("c_Svces","a_Food","LatinAmer") = 125553.609375 ;
vdfb("c_Svces","a_Food","ROW") = 422758.25 ;
vdfb("c_Svces","a_Energy","USA") = 171896.09375 ;
vdfb("c_Svces","a_Energy","EU_28") = 109273.0703125 ;
vdfb("c_Svces","a_Energy","CHN") = 159079.515625 ;
vdfb("c_Svces","a_Energy","LatinAmer") = 64796.23046875 ;
vdfb("c_Svces","a_Energy","ROW") = 542069.625 ;
vdfb("c_Svces","a_Manuf","USA") = 896527.375 ;
vdfb("c_Svces","a_Manuf","EU_28") = 1448538.625 ;
vdfb("c_Svces","a_Manuf","CHN") = 1591671.25 ;
vdfb("c_Svces","a_Manuf","LatinAmer") = 340705.875 ;
vdfb("c_Svces","a_Manuf","ROW") = 1905901.375 ;
vdfb("c_Svces","a_Svces","USA") = 7294576 ;
vdfb("c_Svces","a_Svces","EU_28") = 6652786 ;
vdfb("c_Svces","a_Svces","CHN") = 3999088.75 ;
vdfb("c_Svces","a_Svces","LatinAmer") = 1169808 ;
vdfb("c_Svces","a_Svces","ROW") = 7605396 ;
parameter vdfp(*,*,*) ;
vdfp("c_Agri","a_Agri","USA") = 62658.00390625 ;
vdfp("c_Agri","a_Agri","EU_28") = 44961.21875 ;
vdfp("c_Agri","a_Agri","CHN") = 143057.859375 ;
vdfp("c_Agri","a_Agri","LatinAmer") = 56157.1875 ;
vdfp("c_Agri","a_Agri","ROW") = 306617.8125 ;
vdfp("c_Agri","a_Food","USA") = 211363.25 ;
vdfp("c_Agri","a_Food","EU_28") = 176745.5 ;
vdfp("c_Agri","a_Food","CHN") = 506177.28125 ;
vdfp("c_Agri","a_Food","LatinAmer") = 219292.9375 ;
vdfp("c_Agri","a_Food","ROW") = 639683.6875 ;
vdfp("c_Agri","a_Energy","USA") = 110.850875854492 ;
vdfp("c_Agri","a_Energy","EU_28") = 648.370849609375 ;
vdfp("c_Agri","a_Energy","CHN") = 334.183044433594 ;
vdfp("c_Agri","a_Energy","LatinAmer") = 1011.17749023438 ;
vdfp("c_Agri","a_Energy","ROW") = 5024.14111328125 ;
vdfp("c_Agri","a_Manuf","USA") = 31185.56640625 ;
vdfp("c_Agri","a_Manuf","EU_28") = 27608.087890625 ;
vdfp("c_Agri","a_Manuf","CHN") = 204239.59375 ;
vdfp("c_Agri","a_Manuf","LatinAmer") = 24274.3984375 ;
vdfp("c_Agri","a_Manuf","ROW") = 124052.3046875 ;
vdfp("c_Agri","a_Svces","USA") = 15078.53125 ;
vdfp("c_Agri","a_Svces","EU_28") = 29967.8828125 ;
vdfp("c_Agri","a_Svces","CHN") = 120373.0390625 ;
vdfp("c_Agri","a_Svces","LatinAmer") = 17282.9375 ;
vdfp("c_Agri","a_Svces","ROW") = 204965.5 ;
vdfp("c_Food","a_Agri","USA") = 34583.890625 ;
vdfp("c_Food","a_Agri","EU_28") = 39805.29296875 ;
vdfp("c_Food","a_Agri","CHN") = 109373.03125 ;
vdfp("c_Food","a_Agri","LatinAmer") = 29657.501953125 ;
vdfp("c_Food","a_Agri","ROW") = 87180.125 ;
vdfp("c_Food","a_Food","USA") = 233564.328125 ;
vdfp("c_Food","a_Food","EU_28") = 235506.546875 ;
vdfp("c_Food","a_Food","CHN") = 397432.25 ;
vdfp("c_Food","a_Food","LatinAmer") = 102167.5546875 ;
vdfp("c_Food","a_Food","ROW") = 299739.53125 ;
vdfp("c_Food","a_Energy","USA") = 128.377593994141 ;
vdfp("c_Food","a_Energy","EU_28") = 571.611572265625 ;
vdfp("c_Food","a_Energy","CHN") = 6835.95458984375 ;
vdfp("c_Food","a_Energy","LatinAmer") = 272.769989013672 ;
vdfp("c_Food","a_Energy","ROW") = 1685.56665039062 ;
vdfp("c_Food","a_Manuf","USA") = 9763.541015625 ;
vdfp("c_Food","a_Manuf","EU_28") = 20863.29296875 ;
vdfp("c_Food","a_Manuf","CHN") = 120232.3515625 ;
vdfp("c_Food","a_Manuf","LatinAmer") = 8576.2626953125 ;
vdfp("c_Food","a_Manuf","ROW") = 24474.0703125 ;
vdfp("c_Food","a_Svces","USA") = 162537.078125 ;
vdfp("c_Food","a_Svces","EU_28") = 229256.703125 ;
vdfp("c_Food","a_Svces","CHN") = 274910.5 ;
vdfp("c_Food","a_Svces","LatinAmer") = 90974.6015625 ;
vdfp("c_Food","a_Svces","ROW") = 325454.90625 ;
vdfp("c_Energy","a_Agri","USA") = 11167.017578125 ;
vdfp("c_Energy","a_Agri","EU_28") = 19694.88671875 ;
vdfp("c_Energy","a_Agri","CHN") = 19718.716796875 ;
vdfp("c_Energy","a_Agri","LatinAmer") = 15367.6025390625 ;
vdfp("c_Energy","a_Agri","ROW") = 79605.671875 ;
vdfp("c_Energy","a_Food","USA") = 12908.328125 ;
vdfp("c_Energy","a_Food","EU_28") = 23563.564453125 ;
vdfp("c_Energy","a_Food","CHN") = 16719.51171875 ;
vdfp("c_Energy","a_Food","LatinAmer") = 11164.078125 ;
vdfp("c_Energy","a_Food","ROW") = 37133.875 ;
vdfp("c_Energy","a_Energy","USA") = 316150.8125 ;
vdfp("c_Energy","a_Energy","EU_28") = 138577.890625 ;
vdfp("c_Energy","a_Energy","CHN") = 398523.9375 ;
vdfp("c_Energy","a_Energy","LatinAmer") = 128992.3046875 ;
vdfp("c_Energy","a_Energy","ROW") = 817728.0625 ;
vdfp("c_Energy","a_Manuf","USA") = 138106.515625 ;
vdfp("c_Energy","a_Manuf","EU_28") = 222363.453125 ;
vdfp("c_Energy","a_Manuf","CHN") = 471239.625 ;
vdfp("c_Energy","a_Manuf","LatinAmer") = 90297.375 ;
vdfp("c_Energy","a_Manuf","ROW") = 650974.125 ;
vdfp("c_Energy","a_Svces","USA") = 387289.59375 ;
vdfp("c_Energy","a_Svces","EU_28") = 370971.4375 ;
vdfp("c_Energy","a_Svces","CHN") = 225198.6875 ;
vdfp("c_Energy","a_Svces","LatinAmer") = 108994.46875 ;
vdfp("c_Energy","a_Svces","ROW") = 650581 ;
vdfp("c_Manuf","a_Agri","USA") = 30775.24609375 ;
vdfp("c_Manuf","a_Agri","EU_28") = 24301.419921875 ;
vdfp("c_Manuf","a_Agri","CHN") = 107458.609375 ;
vdfp("c_Manuf","a_Agri","LatinAmer") = 42437.44140625 ;
vdfp("c_Manuf","a_Agri","ROW") = 68936.96875 ;
vdfp("c_Manuf","a_Food","USA") = 101423.9375 ;
vdfp("c_Manuf","a_Food","EU_28") = 73790.4609375 ;
vdfp("c_Manuf","a_Food","CHN") = 84023.484375 ;
vdfp("c_Manuf","a_Food","LatinAmer") = 46500.34765625 ;
vdfp("c_Manuf","a_Food","ROW") = 90777.09375 ;
vdfp("c_Manuf","a_Energy","USA") = 38030.54296875 ;
vdfp("c_Manuf","a_Energy","EU_28") = 37507.5625 ;
vdfp("c_Manuf","a_Energy","CHN") = 111587.0625 ;
vdfp("c_Manuf","a_Energy","LatinAmer") = 23200.85546875 ;
vdfp("c_Manuf","a_Energy","ROW") = 188718.09375 ;
vdfp("c_Manuf","a_Manuf","USA") = 1378763.625 ;
vdfp("c_Manuf","a_Manuf","EU_28") = 1722721 ;
vdfp("c_Manuf","a_Manuf","CHN") = 5891574.5 ;
vdfp("c_Manuf","a_Manuf","LatinAmer") = 572022.3125 ;
vdfp("c_Manuf","a_Manuf","ROW") = 3563150.5 ;
vdfp("c_Manuf","a_Svces","USA") = 1065860.875 ;
vdfp("c_Manuf","a_Svces","EU_28") = 979172.875 ;
vdfp("c_Manuf","a_Svces","CHN") = 2494639 ;
vdfp("c_Manuf","a_Svces","LatinAmer") = 339410.5625 ;
vdfp("c_Manuf","a_Svces","ROW") = 1928628.75 ;
vdfp("c_Svces","a_Agri","USA") = 98816.1484375 ;
vdfp("c_Svces","a_Agri","EU_28") = 69426.8203125 ;
vdfp("c_Svces","a_Agri","CHN") = 123680.6484375 ;
vdfp("c_Svces","a_Agri","LatinAmer") = 55693.24609375 ;
vdfp("c_Svces","a_Agri","ROW") = 243768.6875 ;
vdfp("c_Svces","a_Food","USA") = 215426.5625 ;
vdfp("c_Svces","a_Food","EU_28") = 308471.3125 ;
vdfp("c_Svces","a_Food","CHN") = 253003.578125 ;
vdfp("c_Svces","a_Food","LatinAmer") = 128941.984375 ;
vdfp("c_Svces","a_Food","ROW") = 424776.59375 ;
vdfp("c_Svces","a_Energy","USA") = 172307.6875 ;
vdfp("c_Svces","a_Energy","EU_28") = 109715.9375 ;
vdfp("c_Svces","a_Energy","CHN") = 164887.4375 ;
vdfp("c_Svces","a_Energy","LatinAmer") = 67638.953125 ;
vdfp("c_Svces","a_Energy","ROW") = 545003.9375 ;
vdfp("c_Svces","a_Manuf","USA") = 911648.4375 ;
vdfp("c_Svces","a_Manuf","EU_28") = 1453992.625 ;
vdfp("c_Svces","a_Manuf","CHN") = 1654679.75 ;
vdfp("c_Svces","a_Manuf","LatinAmer") = 349988.125 ;
vdfp("c_Svces","a_Manuf","ROW") = 1912825.375 ;
vdfp("c_Svces","a_Svces","USA") = 7352975 ;
vdfp("c_Svces","a_Svces","EU_28") = 6846185 ;
vdfp("c_Svces","a_Svces","CHN") = 4138268.75 ;
vdfp("c_Svces","a_Svces","LatinAmer") = 1221028.75 ;
vdfp("c_Svces","a_Svces","ROW") = 7685938.5 ;
parameter vmfb(*,*,*) ;
vmfb("c_Agri","a_Agri","USA") = 3115.15600585938 ;
vmfb("c_Agri","a_Agri","EU_28") = 11424.2880859375 ;
vmfb("c_Agri","a_Agri","CHN") = 7588.85302734375 ;
vmfb("c_Agri","a_Agri","LatinAmer") = 2717.82397460938 ;
vmfb("c_Agri","a_Agri","ROW") = 30297.798828125 ;
vmfb("c_Agri","a_Food","USA") = 18982.037109375 ;
vmfb("c_Agri","a_Food","EU_28") = 78343.484375 ;
vmfb("c_Agri","a_Food","CHN") = 36865.05078125 ;
vmfb("c_Agri","a_Food","LatinAmer") = 15496.5146484375 ;
vmfb("c_Agri","a_Food","ROW") = 83114.40625 ;
vmfb("c_Agri","a_Energy","USA") = 4.78604364395142 ;
vmfb("c_Agri","a_Energy","EU_28") = 109.35718536377 ;
vmfb("c_Agri","a_Energy","CHN") = 33.0625267028809 ;
vmfb("c_Agri","a_Energy","LatinAmer") = 21.1700401306152 ;
vmfb("c_Agri","a_Energy","ROW") = 125.123161315918 ;
vmfb("c_Agri","a_Manuf","USA") = 1340.61096191406 ;
vmfb("c_Agri","a_Manuf","EU_28") = 8541.9541015625 ;
vmfb("c_Agri","a_Manuf","CHN") = 14959.1015625 ;
vmfb("c_Agri","a_Manuf","LatinAmer") = 813.679138183594 ;
vmfb("c_Agri","a_Manuf","ROW") = 16031.0087890625 ;
vmfb("c_Agri","a_Svces","USA") = 3635.3017578125 ;
vmfb("c_Agri","a_Svces","EU_28") = 14696.9716796875 ;
vmfb("c_Agri","a_Svces","CHN") = 7028.90771484375 ;
vmfb("c_Agri","a_Svces","LatinAmer") = 1765.85766601562 ;
vmfb("c_Agri","a_Svces","ROW") = 25804.923828125 ;
vmfb("c_Food","a_Agri","USA") = 942.239624023438 ;
vmfb("c_Food","a_Agri","EU_28") = 8502.4189453125 ;
vmfb("c_Food","a_Agri","CHN") = 3454.56005859375 ;
vmfb("c_Food","a_Agri","LatinAmer") = 3389.2421875 ;
vmfb("c_Food","a_Agri","ROW") = 23497.91796875 ;
vmfb("c_Food","a_Food","USA") = 21591.69140625 ;
vmfb("c_Food","a_Food","EU_28") = 90419.3984375 ;
vmfb("c_Food","a_Food","CHN") = 17622.43359375 ;
vmfb("c_Food","a_Food","LatinAmer") = 11593.1669921875 ;
vmfb("c_Food","a_Food","ROW") = 99997.5703125 ;
vmfb("c_Food","a_Energy","USA") = 13.4730405807495 ;
vmfb("c_Food","a_Energy","EU_28") = 184.476303100586 ;
vmfb("c_Food","a_Energy","CHN") = 131.547897338867 ;
vmfb("c_Food","a_Energy","LatinAmer") = 12.8897371292114 ;
vmfb("c_Food","a_Energy","ROW") = 337.108459472656 ;
vmfb("c_Food","a_Manuf","USA") = 795.47998046875 ;
vmfb("c_Food","a_Manuf","EU_28") = 17734.88671875 ;
vmfb("c_Food","a_Manuf","CHN") = 3961.74633789062 ;
vmfb("c_Food","a_Manuf","LatinAmer") = 1118.45471191406 ;
vmfb("c_Food","a_Manuf","ROW") = 5958.966796875 ;
vmfb("c_Food","a_Svces","USA") = 19402.625 ;
vmfb("c_Food","a_Svces","EU_28") = 58425.66796875 ;
vmfb("c_Food","a_Svces","CHN") = 8691.4814453125 ;
vmfb("c_Food","a_Svces","LatinAmer") = 7822.3388671875 ;
vmfb("c_Food","a_Svces","ROW") = 78957.1328125 ;
vmfb("c_Energy","a_Agri","USA") = 967.651184082031 ;
vmfb("c_Energy","a_Agri","EU_28") = 2627.845703125 ;
vmfb("c_Energy","a_Agri","CHN") = 1296.37817382812 ;
vmfb("c_Energy","a_Agri","LatinAmer") = 1533.2685546875 ;
vmfb("c_Energy","a_Agri","ROW") = 7737.93896484375 ;
vmfb("c_Energy","a_Food","USA") = 186.793640136719 ;
vmfb("c_Energy","a_Food","EU_28") = 4658.3525390625 ;
vmfb("c_Energy","a_Food","CHN") = 773.406372070312 ;
vmfb("c_Energy","a_Food","LatinAmer") = 820.126342773438 ;
vmfb("c_Energy","a_Food","ROW") = 4633.423828125 ;
vmfb("c_Energy","a_Energy","USA") = 156432.4375 ;
vmfb("c_Energy","a_Energy","EU_28") = 313315 ;
vmfb("c_Energy","a_Energy","CHN") = 189651.90625 ;
vmfb("c_Energy","a_Energy","LatinAmer") = 55403.1953125 ;
vmfb("c_Energy","a_Energy","ROW") = 560814.75 ;
vmfb("c_Energy","a_Manuf","USA") = 8529.69140625 ;
vmfb("c_Energy","a_Manuf","EU_28") = 46838.10546875 ;
vmfb("c_Energy","a_Manuf","CHN") = 23606.333984375 ;
vmfb("c_Energy","a_Manuf","LatinAmer") = 15212.59375 ;
vmfb("c_Energy","a_Manuf","ROW") = 79833.421875 ;
vmfb("c_Energy","a_Svces","USA") = 32753.537109375 ;
vmfb("c_Energy","a_Svces","EU_28") = 90007.8828125 ;
vmfb("c_Energy","a_Svces","CHN") = 11842.7568359375 ;
vmfb("c_Energy","a_Svces","LatinAmer") = 39155.078125 ;
vmfb("c_Energy","a_Svces","ROW") = 138794.984375 ;
vmfb("c_Manuf","a_Agri","USA") = 10572.728515625 ;
vmfb("c_Manuf","a_Agri","EU_28") = 20506.765625 ;
vmfb("c_Manuf","a_Agri","CHN") = 20505.345703125 ;
vmfb("c_Manuf","a_Agri","LatinAmer") = 12329.484375 ;
vmfb("c_Manuf","a_Agri","ROW") = 66526.53125 ;
vmfb("c_Manuf","a_Food","USA") = 18820.748046875 ;
vmfb("c_Manuf","a_Food","EU_28") = 50258.47265625 ;
vmfb("c_Manuf","a_Food","CHN") = 10691.53125 ;
vmfb("c_Manuf","a_Food","LatinAmer") = 14893.0068359375 ;
vmfb("c_Manuf","a_Food","ROW") = 51941.98828125 ;
vmfb("c_Manuf","a_Energy","USA") = 9609.5107421875 ;
vmfb("c_Manuf","a_Energy","EU_28") = 25616.5703125 ;
vmfb("c_Manuf","a_Energy","CHN") = 10548.43359375 ;
vmfb("c_Manuf","a_Energy","LatinAmer") = 9220.357421875 ;
vmfb("c_Manuf","a_Energy","ROW") = 96793.96875 ;
vmfb("c_Manuf","a_Manuf","USA") = 507021.84375 ;
vmfb("c_Manuf","a_Manuf","EU_28") = 1884049.125 ;
vmfb("c_Manuf","a_Manuf","CHN") = 881362.8125 ;
vmfb("c_Manuf","a_Manuf","LatinAmer") = 348833.5 ;
vmfb("c_Manuf","a_Manuf","ROW") = 1959535.625 ;
vmfb("c_Manuf","a_Svces","USA") = 324454.1875 ;
vmfb("c_Manuf","a_Svces","EU_28") = 723781.875 ;
vmfb("c_Manuf","a_Svces","CHN") = 242184.71875 ;
vmfb("c_Manuf","a_Svces","LatinAmer") = 122099.515625 ;
vmfb("c_Manuf","a_Svces","ROW") = 892336.3125 ;
vmfb("c_Svces","a_Agri","USA") = 1633.474609375 ;
vmfb("c_Svces","a_Agri","EU_28") = 8317.634765625 ;
vmfb("c_Svces","a_Agri","CHN") = 2670.85571289062 ;
vmfb("c_Svces","a_Agri","LatinAmer") = 2007.52856445312 ;
vmfb("c_Svces","a_Agri","ROW") = 12495.1513671875 ;
vmfb("c_Svces","a_Food","USA") = 2725.5048828125 ;
vmfb("c_Svces","a_Food","EU_28") = 38091.9140625 ;
vmfb("c_Svces","a_Food","CHN") = 4980.84375 ;
vmfb("c_Svces","a_Food","LatinAmer") = 4802.03759765625 ;
vmfb("c_Svces","a_Food","ROW") = 15799.7578125 ;
vmfb("c_Svces","a_Energy","USA") = 2508.91772460938 ;
vmfb("c_Svces","a_Energy","EU_28") = 9512.2998046875 ;
vmfb("c_Svces","a_Energy","CHN") = 3307.57275390625 ;
vmfb("c_Svces","a_Energy","LatinAmer") = 5734.9169921875 ;
vmfb("c_Svces","a_Energy","ROW") = 41141.02734375 ;
vmfb("c_Svces","a_Manuf","USA") = 21491.24609375 ;
vmfb("c_Svces","a_Manuf","EU_28") = 239748.453125 ;
vmfb("c_Svces","a_Manuf","CHN") = 47086.63671875 ;
vmfb("c_Svces","a_Manuf","LatinAmer") = 14798.478515625 ;
vmfb("c_Svces","a_Manuf","ROW") = 124148.2421875 ;
vmfb("c_Svces","a_Svces","USA") = 221647.828125 ;
vmfb("c_Svces","a_Svces","EU_28") = 1174965.75 ;
vmfb("c_Svces","a_Svces","CHN") = 125330.65625 ;
vmfb("c_Svces","a_Svces","LatinAmer") = 87761.875 ;
vmfb("c_Svces","a_Svces","ROW") = 678657.3125 ;
parameter vmfp(*,*,*) ;
vmfp("c_Agri","a_Agri","USA") = 3013.82934570312 ;
vmfp("c_Agri","a_Agri","EU_28") = 10970.4619140625 ;
vmfp("c_Agri","a_Agri","CHN") = 7544.10693359375 ;
vmfp("c_Agri","a_Agri","LatinAmer") = 2687.53759765625 ;
vmfp("c_Agri","a_Agri","ROW") = 29698.369140625 ;
vmfp("c_Agri","a_Food","USA") = 19002.740234375 ;
vmfp("c_Agri","a_Food","EU_28") = 79075.5703125 ;
vmfp("c_Agri","a_Food","CHN") = 40778.1953125 ;
vmfp("c_Agri","a_Food","LatinAmer") = 15921.9033203125 ;
vmfp("c_Agri","a_Food","ROW") = 84243.6875 ;
vmfp("c_Agri","a_Energy","USA") = 4.76417112350464 ;
vmfp("c_Agri","a_Energy","EU_28") = 109.632904052734 ;
vmfp("c_Agri","a_Energy","CHN") = 36.4322471618652 ;
vmfp("c_Agri","a_Energy","LatinAmer") = 21.3857383728027 ;
vmfp("c_Agri","a_Energy","ROW") = 131.511306762695 ;
vmfp("c_Agri","a_Manuf","USA") = 1332.66674804688 ;
vmfp("c_Agri","a_Manuf","EU_28") = 8589.4794921875 ;
vmfp("c_Agri","a_Manuf","CHN") = 16678.64453125 ;
vmfp("c_Agri","a_Manuf","LatinAmer") = 820.725646972656 ;
vmfp("c_Agri","a_Manuf","ROW") = 16181.419921875 ;
vmfp("c_Agri","a_Svces","USA") = 3623.306640625 ;
vmfp("c_Agri","a_Svces","EU_28") = 15035.857421875 ;
vmfp("c_Agri","a_Svces","CHN") = 7821.25439453125 ;
vmfp("c_Agri","a_Svces","LatinAmer") = 1813.45947265625 ;
vmfp("c_Agri","a_Svces","ROW") = 26109.306640625 ;
vmfp("c_Food","a_Agri","USA") = 920.455261230469 ;
vmfp("c_Food","a_Agri","EU_28") = 8217.826171875 ;
vmfp("c_Food","a_Agri","CHN") = 3493.33666992188 ;
vmfp("c_Food","a_Agri","LatinAmer") = 3348.46484375 ;
vmfp("c_Food","a_Agri","ROW") = 23464.080078125 ;
vmfp("c_Food","a_Food","USA") = 21784.333984375 ;
vmfp("c_Food","a_Food","EU_28") = 92270.0234375 ;
vmfp("c_Food","a_Food","CHN") = 19478.89453125 ;
vmfp("c_Food","a_Food","LatinAmer") = 12022.8046875 ;
vmfp("c_Food","a_Food","ROW") = 103246.8125 ;
vmfp("c_Food","a_Energy","USA") = 13.5705604553223 ;
vmfp("c_Food","a_Energy","EU_28") = 188.845443725586 ;
vmfp("c_Food","a_Energy","CHN") = 145.722518920898 ;
vmfp("c_Food","a_Energy","LatinAmer") = 13.1606969833374 ;
vmfp("c_Food","a_Energy","ROW") = 346.062164306641 ;
vmfp("c_Food","a_Manuf","USA") = 800.943542480469 ;
vmfp("c_Food","a_Manuf","EU_28") = 18403.49609375 ;
vmfp("c_Food","a_Manuf","CHN") = 4362.10107421875 ;
vmfp("c_Food","a_Manuf","LatinAmer") = 1176.7431640625 ;
vmfp("c_Food","a_Manuf","ROW") = 6147.18359375 ;
vmfp("c_Food","a_Svces","USA") = 19492.033203125 ;
vmfp("c_Food","a_Svces","EU_28") = 65124.02734375 ;
vmfp("c_Food","a_Svces","CHN") = 9567.7998046875 ;
vmfp("c_Food","a_Svces","LatinAmer") = 8168.78271484375 ;
vmfp("c_Food","a_Svces","ROW") = 85612.59375 ;
vmfp("c_Energy","a_Agri","USA") = 1296.49682617188 ;
vmfp("c_Energy","a_Agri","EU_28") = 4727.3466796875 ;
vmfp("c_Energy","a_Agri","CHN") = 1457.38696289062 ;
vmfp("c_Energy","a_Agri","LatinAmer") = 1816.35473632812 ;
vmfp("c_Energy","a_Agri","ROW") = 8614.7236328125 ;
vmfp("c_Energy","a_Food","USA") = 196.893829345703 ;
vmfp("c_Energy","a_Food","EU_28") = 5558.53173828125 ;
vmfp("c_Energy","a_Food","CHN") = 825.931518554688 ;
vmfp("c_Energy","a_Food","LatinAmer") = 900.401428222656 ;
vmfp("c_Energy","a_Food","ROW") = 4695.720703125 ;
vmfp("c_Energy","a_Energy","USA") = 157361.59375 ;
vmfp("c_Energy","a_Energy","EU_28") = 315042.75 ;
vmfp("c_Energy","a_Energy","CHN") = 192800.546875 ;
vmfp("c_Energy","a_Energy","LatinAmer") = 55610.7109375 ;
vmfp("c_Energy","a_Energy","ROW") = 556912.5625 ;
vmfp("c_Energy","a_Manuf","USA") = 10716.05859375 ;
vmfp("c_Energy","a_Manuf","EU_28") = 68992.0234375 ;
vmfp("c_Energy","a_Manuf","CHN") = 26913.068359375 ;
vmfp("c_Energy","a_Manuf","LatinAmer") = 16776.880859375 ;
vmfp("c_Energy","a_Manuf","ROW") = 83417.921875 ;
vmfp("c_Energy","a_Svces","USA") = 43398.08984375 ;
vmfp("c_Energy","a_Svces","EU_28") = 164243.9375 ;
vmfp("c_Energy","a_Svces","CHN") = 13308.4892578125 ;
vmfp("c_Energy","a_Svces","LatinAmer") = 44117.08203125 ;
vmfp("c_Energy","a_Svces","ROW") = 152863.71875 ;
vmfp("c_Manuf","a_Agri","USA") = 10317.4794921875 ;
vmfp("c_Manuf","a_Agri","EU_28") = 19905.388671875 ;
vmfp("c_Manuf","a_Agri","CHN") = 20344.095703125 ;
vmfp("c_Manuf","a_Agri","LatinAmer") = 12338.041015625 ;
vmfp("c_Manuf","a_Agri","ROW") = 64810.32421875 ;
vmfp("c_Manuf","a_Food","USA") = 19385.09765625 ;
vmfp("c_Manuf","a_Food","EU_28") = 50736.7890625 ;
vmfp("c_Manuf","a_Food","CHN") = 12335.900390625 ;
vmfp("c_Manuf","a_Food","LatinAmer") = 15301.1640625 ;
vmfp("c_Manuf","a_Food","ROW") = 52821.5859375 ;
vmfp("c_Manuf","a_Energy","USA") = 9895.4365234375 ;
vmfp("c_Manuf","a_Energy","EU_28") = 25788.802734375 ;
vmfp("c_Manuf","a_Energy","CHN") = 12006.0556640625 ;
vmfp("c_Manuf","a_Energy","LatinAmer") = 9505.841796875 ;
vmfp("c_Manuf","a_Energy","ROW") = 98089.7890625 ;
vmfp("c_Manuf","a_Manuf","USA") = 528007.25 ;
vmfp("c_Manuf","a_Manuf","EU_28") = 1898740.625 ;
vmfp("c_Manuf","a_Manuf","CHN") = 1004309.125 ;
vmfp("c_Manuf","a_Manuf","LatinAmer") = 355503.15625 ;
vmfp("c_Manuf","a_Manuf","ROW") = 1986334.125 ;
vmfp("c_Manuf","a_Svces","USA") = 337981.9375 ;
vmfp("c_Manuf","a_Svces","EU_28") = 764793.625 ;
vmfp("c_Manuf","a_Svces","CHN") = 274820.09375 ;
vmfp("c_Manuf","a_Svces","LatinAmer") = 127611.5 ;
vmfp("c_Manuf","a_Svces","ROW") = 912867.1875 ;
vmfp("c_Svces","a_Agri","USA") = 1593.259765625 ;
vmfp("c_Svces","a_Agri","EU_28") = 8053.70751953125 ;
vmfp("c_Svces","a_Agri","CHN") = 2720.86352539062 ;
vmfp("c_Svces","a_Agri","LatinAmer") = 2026.73815917969 ;
vmfp("c_Svces","a_Agri","ROW") = 11934.3994140625 ;
vmfp("c_Svces","a_Food","USA") = 2756.83154296875 ;
vmfp("c_Svces","a_Food","EU_28") = 38390.19140625 ;
vmfp("c_Svces","a_Food","CHN") = 5360.484375 ;
vmfp("c_Svces","a_Food","LatinAmer") = 4947.19970703125 ;
vmfp("c_Svces","a_Food","ROW") = 15884.7421875 ;
vmfp("c_Svces","a_Energy","USA") = 2504.24169921875 ;
vmfp("c_Svces","a_Energy","EU_28") = 9605.416015625 ;
vmfp("c_Svces","a_Energy","CHN") = 3562.68212890625 ;
vmfp("c_Svces","a_Energy","LatinAmer") = 5864.3603515625 ;
vmfp("c_Svces","a_Energy","ROW") = 41436.23046875 ;
vmfp("c_Svces","a_Manuf","USA") = 21702.380859375 ;
vmfp("c_Svces","a_Manuf","EU_28") = 241582.265625 ;
vmfp("c_Svces","a_Manuf","CHN") = 49950.53515625 ;
vmfp("c_Svces","a_Manuf","LatinAmer") = 15137.974609375 ;
vmfp("c_Svces","a_Manuf","ROW") = 124855.7421875 ;
vmfp("c_Svces","a_Svces","USA") = 224986.59375 ;
vmfp("c_Svces","a_Svces","EU_28") = 1214330.25 ;
vmfp("c_Svces","a_Svces","CHN") = 136314.921875 ;
vmfp("c_Svces","a_Svces","LatinAmer") = 90671.484375 ;
vmfp("c_Svces","a_Svces","ROW") = 685714.5625 ;
parameter vdpb(*,*) ;
vdpb("c_Agri","USA") = 64309.0234375 ;
vdpb("c_Agri","EU_28") = 102098.859375 ;
vdpb("c_Agri","CHN") = 406032.9375 ;
vdpb("c_Agri","LatinAmer") = 85459.9609375 ;
vdpb("c_Agri","ROW") = 868073.875 ;
vdpb("c_Food","USA") = 638353.25 ;
vdpb("c_Food","EU_28") = 536500.6875 ;
vdpb("c_Food","CHN") = 778296.9375 ;
vdpb("c_Food","LatinAmer") = 462912.6875 ;
vdpb("c_Food","ROW") = 1487878.125 ;
vdpb("c_Energy","USA") = 239341.640625 ;
vdpb("c_Energy","EU_28") = 184299.953125 ;
vdpb("c_Energy","CHN") = 153850.40625 ;
vdpb("c_Energy","LatinAmer") = 90066.7734375 ;
vdpb("c_Energy","ROW") = 571792.125 ;
vdpb("c_Manuf","USA") = 711828.625 ;
vdpb("c_Manuf","EU_28") = 379901.625 ;
vdpb("c_Manuf","CHN") = 618153 ;
vdpb("c_Manuf","LatinAmer") = 373707.75 ;
vdpb("c_Manuf","ROW") = 1051209 ;
vdpb("c_Svces","USA") = 10502550 ;
vdpb("c_Svces","EU_28") = 6331369.5 ;
vdpb("c_Svces","CHN") = 2267669 ;
vdpb("c_Svces","LatinAmer") = 2257558 ;
vdpb("c_Svces","ROW") = 8788728 ;
parameter vdpp(*,*) ;
vdpp("c_Agri","USA") = 64258.19140625 ;
vdpp("c_Agri","EU_28") = 111013.671875 ;
vdpp("c_Agri","CHN") = 386537.34375 ;
vdpp("c_Agri","LatinAmer") = 87161.6640625 ;
vdpp("c_Agri","ROW") = 872162.3125 ;
vdpp("c_Food","USA") = 636916.9375 ;
vdpp("c_Food","EU_28") = 712653.3125 ;
vdpp("c_Food","CHN") = 849193.1875 ;
vdpp("c_Food","LatinAmer") = 503925.84375 ;
vdpp("c_Food","ROW") = 1585763.125 ;
vdpp("c_Energy","USA") = 276856.53125 ;
vdpp("c_Energy","EU_28") = 382142.34375 ;
vdpp("c_Energy","CHN") = 148529.5625 ;
vdpp("c_Energy","LatinAmer") = 126182.703125 ;
vdpp("c_Energy","ROW") = 579943.75 ;
vdpp("c_Manuf","USA") = 697360.625 ;
vdpp("c_Manuf","EU_28") = 480620.25 ;
vdpp("c_Manuf","CHN") = 652622.375 ;
vdpp("c_Manuf","LatinAmer") = 424137.75 ;
vdpp("c_Manuf","ROW") = 1110862.875 ;
vdpp("c_Svces","USA") = 10726112 ;
vdpp("c_Svces","EU_28") = 6579813.5 ;
vdpp("c_Svces","CHN") = 2443109.5 ;
vdpp("c_Svces","LatinAmer") = 2345675.5 ;
vdpp("c_Svces","ROW") = 8954946 ;
parameter vmpb(*,*) ;
vmpb("c_Agri","USA") = 24094.908203125 ;
vmpb("c_Agri","EU_28") = 65286.51171875 ;
vmpb("c_Agri","CHN") = 20579.505859375 ;
vmpb("c_Agri","LatinAmer") = 7996.40283203125 ;
vmpb("c_Agri","ROW") = 86824.8359375 ;
vmpb("c_Food","USA") = 67921.8125 ;
vmpb("c_Food","EU_28") = 217179.4375 ;
vmpb("c_Food","CHN") = 30418.919921875 ;
vmpb("c_Food","LatinAmer") = 41635.84375 ;
vmpb("c_Food","ROW") = 270948.53125 ;
vmpb("c_Energy","USA") = 17448.66015625 ;
vmpb("c_Energy","EU_28") = 38157.03515625 ;
vmpb("c_Energy","CHN") = 7418.68603515625 ;
vmpb("c_Energy","LatinAmer") = 9293.576171875 ;
vmpb("c_Energy","ROW") = 65924.9140625 ;
vmpb("c_Manuf","USA") = 616157 ;
vmpb("c_Manuf","EU_28") = 802164.125 ;
vmpb("c_Manuf","CHN") = 141663.21875 ;
vmpb("c_Manuf","LatinAmer") = 155067.578125 ;
vmpb("c_Manuf","ROW") = 940571.1875 ;
vmpb("c_Svces","USA") = 178576.3125 ;
vmpb("c_Svces","EU_28") = 159395.5 ;
vmpb("c_Svces","CHN") = 78959.984375 ;
vmpb("c_Svces","LatinAmer") = 94672.34375 ;
vmpb("c_Svces","ROW") = 472722.1875 ;
parameter vmpp(*,*) ;
vmpp("c_Agri","USA") = 24056.8203125 ;
vmpp("c_Agri","EU_28") = 72266.21875 ;
vmpp("c_Agri","CHN") = 25398.994140625 ;
vmpp("c_Agri","LatinAmer") = 8257.8662109375 ;
vmpp("c_Agri","ROW") = 88863.25 ;
vmpp("c_Food","USA") = 67828.2421875 ;
vmpp("c_Food","EU_28") = 288939.65625 ;
vmpp("c_Food","CHN") = 37009.6875 ;
vmpp("c_Food","LatinAmer") = 45970.140625 ;
vmpp("c_Food","ROW") = 311346.1875 ;
vmpp("c_Energy","USA") = 24221.681640625 ;
vmpp("c_Energy","EU_28") = 87695.0859375 ;
vmpp("c_Energy","CHN") = 7338.15283203125 ;
vmpp("c_Energy","LatinAmer") = 15193.3857421875 ;
vmpp("c_Energy","ROW") = 84172.578125 ;
vmpp("c_Manuf","USA") = 631310.8125 ;
vmpp("c_Manuf","EU_28") = 1042501.1875 ;
vmpp("c_Manuf","CHN") = 177812.234375 ;
vmpp("c_Manuf","LatinAmer") = 172019.4375 ;
vmpp("c_Manuf","ROW") = 1017669.5 ;
vmpp("c_Svces","USA") = 183644.734375 ;
vmpp("c_Svces","EU_28") = 169805.328125 ;
vmpp("c_Svces","CHN") = 93180.6640625 ;
vmpp("c_Svces","LatinAmer") = 100198.2890625 ;
vmpp("c_Svces","ROW") = 482900.09375 ;
parameter vdgb(*,*) ;
vdgb("c_Agri","USA") = 1005.93969726562 ;
vdgb("c_Agri","EU_28") = 4358.82958984375 ;
vdgb("c_Agri","CHN") = 1414.88073730469 ;
vdgb("c_Agri","LatinAmer") = 533.456298828125 ;
vdgb("c_Agri","ROW") = 4732.15234375 ;
vdgb("c_Food","USA") = 17.8301696777344 ;
vdgb("c_Food","EU_28") = 916.144348144531 ;
vdgb("c_Food","CHN") = 13.1568660736084 ;
vdgb("c_Food","LatinAmer") = 372.431182861328 ;
vdgb("c_Food","ROW") = 5307.3759765625 ;
vdgb("c_Manuf","USA") = 405.889831542969 ;
vdgb("c_Manuf","EU_28") = 35854.4609375 ;
vdgb("c_Manuf","CHN") = 297.019256591797 ;
vdgb("c_Manuf","LatinAmer") = 1593.90905761719 ;
vdgb("c_Manuf","ROW") = 57794.9296875 ;
vdgb("c_Svces","USA") = 2543765 ;
vdgb("c_Svces","EU_28") = 3504409.5 ;
vdgb("c_Svces","CHN") = 1978844.125 ;
vdgb("c_Svces","LatinAmer") = 924374.6875 ;
vdgb("c_Svces","ROW") = 4071279.5 ;
parameter vdgp(*,*) ;
vdgp("c_Agri","USA") = 1005.93969726562 ;
vdgp("c_Agri","EU_28") = 4444.12158203125 ;
vdgp("c_Agri","CHN") = 1414.88073730469 ;
vdgp("c_Agri","LatinAmer") = 543.347778320312 ;
vdgp("c_Agri","ROW") = 4763.2705078125 ;
vdgp("c_Food","USA") = 17.8301696777344 ;
vdgp("c_Food","EU_28") = 1010.37707519531 ;
vdgp("c_Food","CHN") = 13.1568660736084 ;
vdgp("c_Food","LatinAmer") = 392.093963623047 ;
vdgp("c_Food","ROW") = 5349.38720703125 ;
vdgp("c_Manuf","USA") = 405.889831542969 ;
vdgp("c_Manuf","EU_28") = 40958.1875 ;
vdgp("c_Manuf","CHN") = 297.019256591797 ;
vdgp("c_Manuf","LatinAmer") = 1611.77319335938 ;
vdgp("c_Manuf","ROW") = 58248.88671875 ;
vdgp("c_Svces","USA") = 2729009.5 ;
vdgp("c_Svces","EU_28") = 3507268.75 ;
vdgp("c_Svces","CHN") = 2000814.25 ;
vdgp("c_Svces","LatinAmer") = 925519.6875 ;
vdgp("c_Svces","ROW") = 4091004.25 ;
parameter vmgb(*,*) ;
vmgb("c_Agri","USA") = 1.61445105075836 ;
vmgb("c_Agri","EU_28") = 6.72422075271606 ;
vmgb("c_Agri","CHN") = 0.861482799053192 ;
vmgb("c_Agri","LatinAmer") = 2.84927105903625 ;
vmgb("c_Agri","ROW") = 337.121337890625 ;
vmgb("c_Food","USA") = 6.32793807983398 ;
vmgb("c_Food","EU_28") = 194.197891235352 ;
vmgb("c_Food","CHN") = 3.53293132781982 ;
vmgb("c_Food","LatinAmer") = 53.769172668457 ;
vmgb("c_Food","ROW") = 1556.56884765625 ;
vmgb("c_Manuf","USA") = 331.96826171875 ;
vmgb("c_Manuf","EU_28") = 68117.15625 ;
vmgb("c_Manuf","CHN") = 474.996795654297 ;
vmgb("c_Manuf","LatinAmer") = 813.696166992188 ;
vmgb("c_Manuf","ROW") = 65790.671875 ;
vmgb("c_Svces","USA") = 19186.185546875 ;
vmgb("c_Svces","EU_28") = 7234.12353515625 ;
vmgb("c_Svces","CHN") = 33935.25390625 ;
vmgb("c_Svces","LatinAmer") = 6868.72509765625 ;
vmgb("c_Svces","ROW") = 61176.1328125 ;
parameter vmgp(*,*) ;
vmgp("c_Agri","USA") = 1.61445105075836 ;
vmgp("c_Agri","EU_28") = 6.94188785552979 ;
vmgp("c_Agri","CHN") = 0.861482799053192 ;
vmgp("c_Agri","LatinAmer") = 2.86859703063965 ;
vmgp("c_Agri","ROW") = 337.608520507812 ;
vmgp("c_Food","USA") = 6.32793807983398 ;
vmgp("c_Food","EU_28") = 216.564392089844 ;
vmgp("c_Food","CHN") = 3.53293132781982 ;
vmgp("c_Food","LatinAmer") = 54.4824180603027 ;
vmgp("c_Food","ROW") = 1578.13806152344 ;
vmgp("c_Manuf","USA") = 331.96826171875 ;
vmgp("c_Manuf","EU_28") = 76792.6875 ;
vmgp("c_Manuf","CHN") = 474.996795654297 ;
vmgp("c_Manuf","LatinAmer") = 834.32568359375 ;
vmgp("c_Manuf","ROW") = 65687.265625 ;
vmgp("c_Svces","USA") = 19186.185546875 ;
vmgp("c_Svces","EU_28") = 9761.845703125 ;
vmgp("c_Svces","CHN") = 38809.62109375 ;
vmgp("c_Svces","LatinAmer") = 7140.5966796875 ;
vmgp("c_Svces","ROW") = 61764.18359375 ;
parameter vdib(*,*) ;
vdib("c_Agri","USA") = 501.632507324219 ;
vdib("c_Agri","EU_28") = 5512.02197265625 ;
vdib("c_Agri","CHN") = 24333.41015625 ;
vdib("c_Agri","LatinAmer") = 12658.224609375 ;
vdib("c_Agri","ROW") = 36491.71875 ;
vdib("c_Food","USA") = 21.0176010131836 ;
vdib("c_Food","EU_28") = 184.110336303711 ;
vdib("c_Food","CHN") = 26.2256641387939 ;
vdib("c_Food","LatinAmer") = 2437.61499023438 ;
vdib("c_Food","ROW") = 10194.5966796875 ;
vdib("c_Manuf","USA") = 765932.4375 ;
vdib("c_Manuf","EU_28") = 473251.46875 ;
vdib("c_Manuf","CHN") = 947260.6875 ;
vdib("c_Manuf","LatinAmer") = 166852.6875 ;
vdib("c_Manuf","ROW") = 987661.625 ;
vdib("c_Svces","USA") = 2873172.5 ;
vdib("c_Svces","EU_28") = 2258155.25 ;
vdib("c_Svces","CHN") = 3693320.5 ;
vdib("c_Svces","LatinAmer") = 658275.25 ;
vdib("c_Svces","ROW") = 4539059 ;
parameter vdip(*,*) ;
vdip("c_Agri","USA") = 501.632507324219 ;
vdip("c_Agri","EU_28") = 5307.62548828125 ;
vdip("c_Agri","CHN") = 24359.677734375 ;
vdip("c_Agri","LatinAmer") = 13212.6025390625 ;
vdip("c_Agri","ROW") = 36458.8359375 ;
vdip("c_Food","USA") = 21.0176010131836 ;
vdip("c_Food","EU_28") = 183.92138671875 ;
vdip("c_Food","CHN") = 26.2256641387939 ;
vdip("c_Food","LatinAmer") = 2473.79711914062 ;
vdip("c_Food","ROW") = 10416.0361328125 ;
vdip("c_Manuf","USA") = 775809.125 ;
vdip("c_Manuf","EU_28") = 490787.6875 ;
vdip("c_Manuf","CHN") = 1020333.4375 ;
vdip("c_Manuf","LatinAmer") = 181197.90625 ;
vdip("c_Manuf","ROW") = 1020758.6875 ;
vdip("c_Svces","USA") = 2692840.75 ;
vdip("c_Svces","EU_28") = 2438634.25 ;
vdip("c_Svces","CHN") = 3877627.75 ;
vdip("c_Svces","LatinAmer") = 668482.75 ;
vdip("c_Svces","ROW") = 4634943.5 ;
parameter vmib(*,*) ;
vmib("c_Agri","USA") = 11.8654775619507 ;
vmib("c_Agri","EU_28") = 1697.59985351562 ;
vmib("c_Agri","CHN") = 395.394165039062 ;
vmib("c_Agri","LatinAmer") = 596.838256835938 ;
vmib("c_Agri","ROW") = 4238.91845703125 ;
vmib("c_Food","USA") = 10.5425462722778 ;
vmib("c_Food","EU_28") = 350.453369140625 ;
vmib("c_Food","CHN") = 10.2843914031982 ;
vmib("c_Food","LatinAmer") = 44.6939353942871 ;
vmib("c_Food","ROW") = 4229.59765625 ;
vmib("c_Manuf","USA") = 473778.375 ;
vmib("c_Manuf","EU_28") = 582482.6875 ;
vmib("c_Manuf","CHN") = 235322.78125 ;
vmib("c_Manuf","LatinAmer") = 153429.5 ;
vmib("c_Manuf","ROW") = 905921.5625 ;
vmib("c_Svces","USA") = 99352.0078125 ;
vmib("c_Svces","EU_28") = 92506.1640625 ;
vmib("c_Svces","CHN") = 16403.72265625 ;
vmib("c_Svces","LatinAmer") = 5939.58349609375 ;
vmib("c_Svces","ROW") = 117026.6875 ;
parameter vmip(*,*) ;
vmip("c_Agri","USA") = 11.8654775619507 ;
vmip("c_Agri","EU_28") = 1726.70056152344 ;
vmip("c_Agri","CHN") = 471.723602294922 ;
vmip("c_Agri","LatinAmer") = 610.095153808594 ;
vmip("c_Agri","ROW") = 4285.6513671875 ;
vmip("c_Food","USA") = 10.5425462722778 ;
vmip("c_Food","EU_28") = 350.649597167969 ;
vmip("c_Food","CHN") = 10.2843914031982 ;
vmip("c_Food","LatinAmer") = 45.017204284668 ;
vmip("c_Food","ROW") = 4256.712890625 ;
vmip("c_Manuf","USA") = 491439.5 ;
vmip("c_Manuf","EU_28") = 614703.125 ;
vmip("c_Manuf","CHN") = 293470.6875 ;
vmip("c_Manuf","LatinAmer") = 161907.0625 ;
vmip("c_Manuf","ROW") = 940479.9375 ;
vmip("c_Svces","USA") = 87960.8984375 ;
vmip("c_Svces","EU_28") = 99512.546875 ;
vmip("c_Svces","CHN") = 19477.990234375 ;
vmip("c_Svces","LatinAmer") = 6202.46533203125 ;
vmip("c_Svces","ROW") = 117596.6875 ;
parameter evfb(*,*,*) ;
evfb("Land","a_Agri","USA") = 41351.43359375 ;
evfb("Land","a_Agri","EU_28") = 67458 ;
evfb("Land","a_Agri","CHN") = 251005.1875 ;
evfb("Land","a_Agri","LatinAmer") = 59448.42578125 ;
evfb("Land","a_Agri","ROW") = 358800.625 ;
evfb("UnSkLab","a_Agri","USA") = 32854.8984375 ;
evfb("UnSkLab","a_Agri","EU_28") = 82710.4765625 ;
evfb("UnSkLab","a_Agri","CHN") = 378059.0625 ;
evfb("UnSkLab","a_Agri","LatinAmer") = 114264.703125 ;
evfb("UnSkLab","a_Agri","ROW") = 624863.625 ;
evfb("UnSkLab","a_Food","USA") = 86790.1640625 ;
evfb("UnSkLab","a_Food","EU_28") = 59448.6171875 ;
evfb("UnSkLab","a_Food","CHN") = 138670.328125 ;
evfb("UnSkLab","a_Food","LatinAmer") = 56313.2421875 ;
evfb("UnSkLab","a_Food","ROW") = 207095.765625 ;
evfb("UnSkLab","a_Energy","USA") = 52987.9609375 ;
evfb("UnSkLab","a_Energy","EU_28") = 19774.361328125 ;
evfb("UnSkLab","a_Energy","CHN") = 107518.109375 ;
evfb("UnSkLab","a_Energy","LatinAmer") = 17726.693359375 ;
evfb("UnSkLab","a_Energy","ROW") = 180954.78125 ;
evfb("UnSkLab","a_Manuf","USA") = 691563.75 ;
evfb("UnSkLab","a_Manuf","EU_28") = 436514.6875 ;
evfb("UnSkLab","a_Manuf","CHN") = 1026031.4375 ;
evfb("UnSkLab","a_Manuf","LatinAmer") = 193146.53125 ;
evfb("UnSkLab","a_Manuf","ROW") = 900654.6875 ;
evfb("UnSkLab","a_Svces","USA") = 3282724.5 ;
evfb("UnSkLab","a_Svces","EU_28") = 1784304.5 ;
evfb("UnSkLab","a_Svces","CHN") = 2457188 ;
evfb("UnSkLab","a_Svces","LatinAmer") = 937292.4375 ;
evfb("UnSkLab","a_Svces","ROW") = 3716833 ;
evfb("SkLab","a_Agri","USA") = 32099.72265625 ;
evfb("SkLab","a_Agri","EU_28") = 23833.029296875 ;
evfb("SkLab","a_Agri","CHN") = 23754.166015625 ;
evfb("SkLab","a_Agri","LatinAmer") = 12459.0322265625 ;
evfb("SkLab","a_Agri","ROW") = 27777.48046875 ;
evfb("SkLab","a_Food","USA") = 29155.5390625 ;
evfb("SkLab","a_Food","EU_28") = 66590.28125 ;
evfb("SkLab","a_Food","CHN") = 27360.30859375 ;
evfb("SkLab","a_Food","LatinAmer") = 27975.193359375 ;
evfb("SkLab","a_Food","ROW") = 85943.953125 ;
evfb("SkLab","a_Energy","USA") = 21668.21484375 ;
evfb("SkLab","a_Energy","EU_28") = 23952.671875 ;
evfb("SkLab","a_Energy","CHN") = 21067.271484375 ;
evfb("SkLab","a_Energy","LatinAmer") = 13941.58984375 ;
evfb("SkLab","a_Energy","ROW") = 153880.265625 ;
evfb("SkLab","a_Manuf","USA") = 229677.921875 ;
evfb("SkLab","a_Manuf","EU_28") = 473903.84375 ;
evfb("SkLab","a_Manuf","CHN") = 198534.234375 ;
evfb("SkLab","a_Manuf","LatinAmer") = 79832.890625 ;
evfb("SkLab","a_Manuf","ROW") = 471463.5625 ;
evfb("SkLab","a_Svces","USA") = 3780121.5 ;
evfb("SkLab","a_Svces","EU_28") = 2512409.5 ;
evfb("SkLab","a_Svces","CHN") = 1334502.625 ;
evfb("SkLab","a_Svces","LatinAmer") = 649102.875 ;
evfb("SkLab","a_Svces","ROW") = 3661292 ;
evfb("Capital","a_Agri","USA") = 92125.1796875 ;
evfb("Capital","a_Agri","EU_28") = 107050.140625 ;
evfb("Capital","a_Agri","CHN") = 256185.375 ;
evfb("Capital","a_Agri","LatinAmer") = 85173.6875 ;
evfb("Capital","a_Agri","ROW") = 376261.1875 ;
evfb("Capital","a_Food","USA") = 133316.78125 ;
evfb("Capital","a_Food","EU_28") = 187744.53125 ;
evfb("Capital","a_Food","CHN") = 185199.453125 ;
evfb("Capital","a_Food","LatinAmer") = 147757.25 ;
evfb("Capital","a_Food","ROW") = 446692.09375 ;
evfb("Capital","a_Energy","USA") = 220980.765625 ;
evfb("Capital","a_Energy","EU_28") = 165926.890625 ;
evfb("Capital","a_Energy","CHN") = 187239.34375 ;
evfb("Capital","a_Energy","LatinAmer") = 161737.078125 ;
evfb("Capital","a_Energy","ROW") = 1143287.125 ;
evfb("Capital","a_Manuf","USA") = 877950.375 ;
evfb("Capital","a_Manuf","EU_28") = 1058059.375 ;
evfb("Capital","a_Manuf","CHN") = 1268539.25 ;
evfb("Capital","a_Manuf","LatinAmer") = 396093.28125 ;
evfb("Capital","a_Manuf","ROW") = 2029665.875 ;
evfb("Capital","a_Svces","USA") = 5918652.5 ;
evfb("Capital","a_Svces","EU_28") = 5561561.5 ;
evfb("Capital","a_Svces","CHN") = 2660009.25 ;
evfb("Capital","a_Svces","LatinAmer") = 1825079.125 ;
evfb("Capital","a_Svces","ROW") = 8119683.5 ;
evfb("NatRes","a_Agri","USA") = 5286.4248046875 ;
evfb("NatRes","a_Agri","EU_28") = 8407.166015625 ;
evfb("NatRes","a_Agri","CHN") = 44576.296875 ;
evfb("NatRes","a_Agri","LatinAmer") = 7114.18212890625 ;
evfb("NatRes","a_Agri","ROW") = 60747.79296875 ;
evfb("NatRes","a_Energy","USA") = 61101.1015625 ;
evfb("NatRes","a_Energy","EU_28") = 13122.53515625 ;
evfb("NatRes","a_Energy","CHN") = 53481.14453125 ;
evfb("NatRes","a_Energy","LatinAmer") = 43739.21875 ;
evfb("NatRes","a_Energy","ROW") = 409375.6875 ;
evfb("NatRes","a_Manuf","USA") = 14772.66015625 ;
evfb("NatRes","a_Manuf","EU_28") = 6984.15234375 ;
evfb("NatRes","a_Manuf","CHN") = 15029.1279296875 ;
evfb("NatRes","a_Manuf","LatinAmer") = 9267.2236328125 ;
evfb("NatRes","a_Manuf","ROW") = 41160.515625 ;
parameter evfp(*,*,*) ;
evfp("Land","a_Agri","USA") = 34003.03515625 ;
evfp("Land","a_Agri","EU_28") = 45434.83203125 ;
evfp("Land","a_Agri","CHN") = 228201.203125 ;
evfp("Land","a_Agri","LatinAmer") = 59821.10546875 ;
evfp("Land","a_Agri","ROW") = 352268.875 ;
evfp("UnSkLab","a_Agri","USA") = 37470.65234375 ;
evfp("UnSkLab","a_Agri","EU_28") = 101959.7734375 ;
evfp("UnSkLab","a_Agri","CHN") = 375595.71875 ;
evfp("UnSkLab","a_Agri","LatinAmer") = 133473.328125 ;
evfp("UnSkLab","a_Agri","ROW") = 654104.4375 ;
evfp("UnSkLab","a_Food","USA") = 104457.9765625 ;
evfp("UnSkLab","a_Food","EU_28") = 86886.4140625 ;
evfp("UnSkLab","a_Food","CHN") = 141001.421875 ;
evfp("UnSkLab","a_Food","LatinAmer") = 65632.1484375 ;
evfp("UnSkLab","a_Food","ROW") = 225252.203125 ;
evfp("UnSkLab","a_Energy","USA") = 63774.6875 ;
evfp("UnSkLab","a_Energy","EU_28") = 28768.41015625 ;
evfp("UnSkLab","a_Energy","CHN") = 109325.515625 ;
evfp("UnSkLab","a_Energy","LatinAmer") = 20720.837890625 ;
evfp("UnSkLab","a_Energy","ROW") = 198626.546875 ;
evfp("UnSkLab","a_Manuf","USA") = 832344.9375 ;
evfp("UnSkLab","a_Manuf","EU_28") = 639787.75 ;
evfp("UnSkLab","a_Manuf","CHN") = 1043279.3125 ;
evfp("UnSkLab","a_Manuf","LatinAmer") = 224882.96875 ;
evfp("UnSkLab","a_Manuf","ROW") = 1047419.5 ;
evfp("UnSkLab","a_Svces","USA") = 3950986.75 ;
evfp("UnSkLab","a_Svces","EU_28") = 2543290 ;
evfp("UnSkLab","a_Svces","CHN") = 2498494 ;
evfp("UnSkLab","a_Svces","LatinAmer") = 1103080.25 ;
evfp("UnSkLab","a_Svces","ROW") = 4278355.5 ;
evfp("SkLab","a_Agri","USA") = 36611.2109375 ;
evfp("SkLab","a_Agri","EU_28") = 28117.212890625 ;
evfp("SkLab","a_Agri","CHN") = 23596.517578125 ;
evfp("SkLab","a_Agri","LatinAmer") = 13845.66796875 ;
evfp("SkLab","a_Agri","ROW") = 29669.068359375 ;
evfp("SkLab","a_Food","USA") = 35090.71484375 ;
evfp("SkLab","a_Food","EU_28") = 95431.5390625 ;
evfp("SkLab","a_Food","CHN") = 27820.2421875 ;
evfp("SkLab","a_Food","LatinAmer") = 31990.302734375 ;
evfp("SkLab","a_Food","ROW") = 96940.859375 ;
evfp("SkLab","a_Energy","USA") = 26079.201171875 ;
evfp("SkLab","a_Energy","EU_28") = 34500.87109375 ;
evfp("SkLab","a_Energy","CHN") = 21421.41796875 ;
evfp("SkLab","a_Energy","LatinAmer") = 15900.037109375 ;
evfp("SkLab","a_Energy","ROW") = 169267.71875 ;
evfp("SkLab","a_Manuf","USA") = 276433.3125 ;
evfp("SkLab","a_Manuf","EU_28") = 683187.625 ;
evfp("SkLab","a_Manuf","CHN") = 201871.65625 ;
evfp("SkLab","a_Manuf","LatinAmer") = 91532.28125 ;
evfp("SkLab","a_Manuf","ROW") = 558496.6875 ;
evfp("SkLab","a_Svces","USA") = 4549638.5 ;
evfp("SkLab","a_Svces","EU_28") = 3595405 ;
evfp("SkLab","a_Svces","CHN") = 1356936 ;
evfp("SkLab","a_Svces","LatinAmer") = 748339.75 ;
evfp("SkLab","a_Svces","ROW") = 4238578 ;
evfp("Capital","a_Agri","USA") = 93103.03125 ;
evfp("Capital","a_Agri","EU_28") = 92191.0546875 ;
evfp("Capital","a_Agri","CHN") = 243381.765625 ;
evfp("Capital","a_Agri","LatinAmer") = 83529.8515625 ;
evfp("Capital","a_Agri","ROW") = 371885.15625 ;
evfp("Capital","a_Food","USA") = 139993.515625 ;
evfp("Capital","a_Food","EU_28") = 194236.046875 ;
evfp("Capital","a_Food","CHN") = 188312.703125 ;
evfp("Capital","a_Food","LatinAmer") = 150112.46875 ;
evfp("Capital","a_Food","ROW") = 451633.4375 ;
evfp("Capital","a_Energy","USA") = 232047.859375 ;
evfp("Capital","a_Energy","EU_28") = 171349.125 ;
evfp("Capital","a_Energy","CHN") = 190386.875 ;
evfp("Capital","a_Energy","LatinAmer") = 164931.734375 ;
evfp("Capital","a_Energy","ROW") = 1156469.625 ;
evfp("Capital","a_Manuf","USA") = 921919.625 ;
evfp("Capital","a_Manuf","EU_28") = 1087454.875 ;
evfp("Capital","a_Manuf","CHN") = 1289863.75 ;
evfp("Capital","a_Manuf","LatinAmer") = 402780.84375 ;
evfp("Capital","a_Manuf","ROW") = 2063073.25 ;
evfp("Capital","a_Svces","USA") = 6215069 ;
evfp("Capital","a_Svces","EU_28") = 5746506.5 ;
evfp("Capital","a_Svces","CHN") = 2704724.75 ;
evfp("Capital","a_Svces","LatinAmer") = 1858644 ;
evfp("Capital","a_Svces","ROW") = 8267268 ;
evfp("NatRes","a_Agri","USA") = 5551.177734375 ;
evfp("NatRes","a_Agri","EU_28") = 8703.9833984375 ;
evfp("NatRes","a_Agri","CHN") = 45325.63671875 ;
evfp("NatRes","a_Agri","LatinAmer") = 7230.2314453125 ;
evfp("NatRes","a_Agri","ROW") = 61200.8671875 ;
evfp("NatRes","a_Energy","USA") = 64161.1484375 ;
evfp("NatRes","a_Energy","EU_28") = 13490.23828125 ;
evfp("NatRes","a_Energy","CHN") = 54380.17578125 ;
evfp("NatRes","a_Energy","LatinAmer") = 44641.8828125 ;
evfp("NatRes","a_Energy","ROW") = 414077.71875 ;
evfp("NatRes","a_Manuf","USA") = 15512.5 ;
evfp("NatRes","a_Manuf","EU_28") = 7221.03173828125 ;
evfp("NatRes","a_Manuf","CHN") = 15281.771484375 ;
evfp("NatRes","a_Manuf","LatinAmer") = 9408.2451171875 ;
evfp("NatRes","a_Manuf","ROW") = 41962.1953125 ;
parameter evos(*,*,*) ;
evos("Land","a_Agri","USA") = 39669.4453125 ;
evos("Land","a_Agri","EU_28") = 63766.05859375 ;
evos("Land","a_Agri","CHN") = 223061.875 ;
evos("Land","a_Agri","LatinAmer") = 55195.61328125 ;
evos("Land","a_Agri","ROW") = 335562.3125 ;
evos("UnSkLab","a_Agri","USA") = 24738.650390625 ;
evos("UnSkLab","a_Agri","EU_28") = 57630.91015625 ;
evos("UnSkLab","a_Agri","CHN") = 366342.1875 ;
evos("UnSkLab","a_Agri","LatinAmer") = 105564.7109375 ;
evos("UnSkLab","a_Agri","ROW") = 582229.8125 ;
evos("UnSkLab","a_Food","USA") = 65350.11328125 ;
evos("UnSkLab","a_Food","EU_28") = 41500.9921875 ;
evos("UnSkLab","a_Food","CHN") = 134372.640625 ;
evos("UnSkLab","a_Food","LatinAmer") = 51998.3984375 ;
evos("UnSkLab","a_Food","ROW") = 190074.6875 ;
evos("UnSkLab","a_Energy","USA") = 39898.1796875 ;
evos("UnSkLab","a_Energy","EU_28") = 14185.1875 ;
evos("UnSkLab","a_Energy","CHN") = 104185.890625 ;
evos("UnSkLab","a_Energy","LatinAmer") = 16198.490234375 ;
evos("UnSkLab","a_Energy","ROW") = 165166.765625 ;
evos("UnSkLab","a_Manuf","USA") = 520724.59375 ;
evos("UnSkLab","a_Manuf","EU_28") = 300480.53125 ;
evos("UnSkLab","a_Manuf","CHN") = 994232.5 ;
evos("UnSkLab","a_Manuf","LatinAmer") = 177115.96875 ;
evos("UnSkLab","a_Manuf","ROW") = 796999.1875 ;
evos("UnSkLab","a_Svces","USA") = 2471782.75 ;
evos("UnSkLab","a_Svces","EU_28") = 1264046.5 ;
evos("UnSkLab","a_Svces","CHN") = 2381034.75 ;
evos("UnSkLab","a_Svces","LatinAmer") = 861850.875 ;
evos("UnSkLab","a_Svces","ROW") = 3266970.5 ;
evos("SkLab","a_Agri","USA") = 24170.029296875 ;
evos("SkLab","a_Agri","EU_28") = 15801.244140625 ;
evos("SkLab","a_Agri","CHN") = 23017.974609375 ;
evos("SkLab","a_Agri","LatinAmer") = 11788.1357421875 ;
evos("SkLab","a_Agri","ROW") = 23710.572265625 ;
evos("SkLab","a_Food","USA") = 21953.15625 ;
evos("SkLab","a_Food","EU_28") = 46888.703125 ;
evos("SkLab","a_Food","CHN") = 26512.353515625 ;
evos("SkLab","a_Food","LatinAmer") = 26133.65625 ;
evos("SkLab","a_Food","ROW") = 75277.4453125 ;
evos("SkLab","a_Energy","USA") = 16315.44921875 ;
evos("SkLab","a_Energy","EU_28") = 17062.15625 ;
evos("SkLab","a_Energy","CHN") = 20414.3515625 ;
evos("SkLab","a_Energy","LatinAmer") = 12683.431640625 ;
evos("SkLab","a_Energy","ROW") = 139775.515625 ;
evos("SkLab","a_Manuf","USA") = 172939.875 ;
evos("SkLab","a_Manuf","EU_28") = 328497.65625 ;
evos("SkLab","a_Manuf","CHN") = 192381.234375 ;
evos("SkLab","a_Manuf","LatinAmer") = 74211.765625 ;
evos("SkLab","a_Manuf","ROW") = 405182.09375 ;
evos("SkLab","a_Svces","USA") = 2846306.5 ;
evos("SkLab","a_Svces","EU_28") = 1767065.125 ;
evos("SkLab","a_Svces","CHN") = 1293143.625 ;
evos("SkLab","a_Svces","LatinAmer") = 596355.25 ;
evos("SkLab","a_Svces","ROW") = 3143764.75 ;
evos("Capital","a_Agri","USA") = 88377.9375 ;
evos("Capital","a_Agri","EU_28") = 100920.53125 ;
evos("Capital","a_Agri","CHN") = 227665.375 ;
evos("Capital","a_Agri","LatinAmer") = 79138.6953125 ;
evos("Capital","a_Agri","ROW") = 351941.03125 ;
evos("Capital","a_Food","USA") = 127894.046875 ;
evos("Capital","a_Food","EU_28") = 176723.046875 ;
evos("Capital","a_Food","CHN") = 164582.015625 ;
evos("Capital","a_Food","LatinAmer") = 137954.0625 ;
evos("Capital","a_Food","ROW") = 419185.53125 ;
evos("Capital","a_Energy","USA") = 211992.234375 ;
evos("Capital","a_Energy","EU_28") = 155744.421875 ;
evos("Capital","a_Energy","CHN") = 166394.8125 ;
evos("Capital","a_Energy","LatinAmer") = 149677.953125 ;
evos("Capital","a_Energy","ROW") = 1065776.5 ;
evos("Capital","a_Manuf","USA") = 842239.25 ;
evos("Capital","a_Manuf","EU_28") = 997180.1875 ;
evos("Capital","a_Manuf","CHN") = 1127318.375 ;
evos("Capital","a_Manuf","LatinAmer") = 369155.375 ;
evos("Capital","a_Manuf","ROW") = 1890256.5 ;
evos("Capital","a_Svces","USA") = 5677908 ;
evos("Capital","a_Svces","EU_28") = 5227559 ;
evos("Capital","a_Svces","CHN") = 2363882.25 ;
evos("Capital","a_Svces","LatinAmer") = 1702158.5 ;
evos("Capital","a_Svces","ROW") = 7521777.5 ;
evos("NatRes","a_Agri","USA") = 5071.39697265625 ;
evos("NatRes","a_Agri","EU_28") = 7891.44140625 ;
evos("NatRes","a_Agri","CHN") = 39613.8046875 ;
evos("NatRes","a_Agri","LatinAmer") = 6598.5244140625 ;
evos("NatRes","a_Agri","ROW") = 56598.68359375 ;
evos("NatRes","a_Energy","USA") = 58615.7890625 ;
evos("NatRes","a_Energy","EU_28") = 12277.4765625 ;
evos("NatRes","a_Energy","CHN") = 47527.31640625 ;
evos("NatRes","a_Energy","LatinAmer") = 40473.07421875 ;
evos("NatRes","a_Energy","ROW") = 382323.1875 ;
evos("NatRes","a_Manuf","USA") = 14171.775390625 ;
evos("NatRes","a_Manuf","EU_28") = 6557.8984375 ;
evos("NatRes","a_Manuf","CHN") = 13355.9990234375 ;
evos("NatRes","a_Manuf","LatinAmer") = 8581.6044921875 ;
evos("NatRes","a_Manuf","ROW") = 37634.71484375 ;
parameter vxsb(*,*,*) ;
vxsb("c_Agri","USA","USA") = 1.40000001920271E-5 ;
vxsb("c_Agri","USA","EU_28") = 6803.40087890625 ;
vxsb("c_Agri","USA","CHN") = 17922.62109375 ;
vxsb("c_Agri","USA","LatinAmer") = 13557.67578125 ;
vxsb("c_Agri","USA","ROW") = 37804.58203125 ;
vxsb("c_Agri","EU_28","USA") = 2226.00439453125 ;
vxsb("c_Agri","EU_28","EU_28") = 100227.4296875 ;
vxsb("c_Agri","EU_28","CHN") = 2349.16186523438 ;
vxsb("c_Agri","EU_28","LatinAmer") = 1016.43829345703 ;
vxsb("c_Agri","EU_28","ROW") = 21999.103515625 ;
vxsb("c_Agri","CHN","USA") = 918.580200195312 ;
vxsb("c_Agri","CHN","EU_28") = 1716.79809570312 ;
vxsb("c_Agri","CHN","CHN") = 1.40000001920271E-5 ;
vxsb("c_Agri","CHN","LatinAmer") = 379.839141845703 ;
vxsb("c_Agri","CHN","ROW") = 16036.77734375 ;
vxsb("c_Agri","LatinAmer","USA") = 28035.060546875 ;
vxsb("c_Agri","LatinAmer","EU_28") = 17740.53515625 ;
vxsb("c_Agri","LatinAmer","CHN") = 26747.9765625 ;
vxsb("c_Agri","LatinAmer","LatinAmer") = 8855.36328125 ;
vxsb("c_Agri","LatinAmer","ROW") = 26931.626953125 ;
vxsb("c_Agri","ROW","USA") = 15995.8662109375 ;
vxsb("c_Agri","ROW","EU_28") = 37312.984375 ;
vxsb("c_Agri","ROW","CHN") = 29041.34375 ;
vxsb("c_Agri","ROW","LatinAmer") = 2985.5009765625 ;
vxsb("c_Agri","ROW","ROW") = 103801.015625 ;
vxsb("c_Food","USA","USA") = 7.99999997980194E-6 ;
vxsb("c_Food","USA","EU_28") = 6083.873046875 ;
vxsb("c_Food","USA","CHN") = 4125.4521484375 ;
vxsb("c_Food","USA","LatinAmer") = 20857.462890625 ;
vxsb("c_Food","USA","ROW") = 48405.03125 ;
vxsb("c_Food","EU_28","USA") = 23423.763671875 ;
vxsb("c_Food","EU_28","EU_28") = 291054.4375 ;
vxsb("c_Food","EU_28","CHN") = 13131.53515625 ;
vxsb("c_Food","EU_28","LatinAmer") = 6786.76953125 ;
vxsb("c_Food","EU_28","ROW") = 83116.28125 ;
vxsb("c_Food","CHN","USA") = 7010.4384765625 ;
vxsb("c_Food","CHN","EU_28") = 6218.8701171875 ;
vxsb("c_Food","CHN","CHN") = 7.99999997980194E-6 ;
vxsb("c_Food","CHN","LatinAmer") = 2068.73168945312 ;
vxsb("c_Food","CHN","ROW") = 34390.14453125 ;
vxsb("c_Food","LatinAmer","USA") = 25579.4609375 ;
vxsb("c_Food","LatinAmer","EU_28") = 20379.017578125 ;
vxsb("c_Food","LatinAmer","CHN") = 8027.95654296875 ;
vxsb("c_Food","LatinAmer","LatinAmer") = 24203.828125 ;
vxsb("c_Food","LatinAmer","ROW") = 49317.234375 ;
vxsb("c_Food","ROW","USA") = 47711.16796875 ;
vxsb("c_Food","ROW","EU_28") = 49427.82421875 ;
vxsb("c_Food","ROW","CHN") = 30606.88671875 ;
vxsb("c_Food","ROW","LatinAmer") = 5192.49951171875 ;
vxsb("c_Food","ROW","ROW") = 200286.75 ;
vxsb("c_Energy","USA","USA") = 6.00000021222513E-6 ;
vxsb("c_Energy","USA","EU_28") = 17274.267578125 ;
vxsb("c_Energy","USA","CHN") = 8890.4287109375 ;
vxsb("c_Energy","USA","LatinAmer") = 74734.9765625 ;
vxsb("c_Energy","USA","ROW") = 43549.54296875 ;
vxsb("c_Energy","EU_28","USA") = 9159.8876953125 ;
vxsb("c_Energy","EU_28","EU_28") = 93616.8515625 ;
vxsb("c_Energy","EU_28","CHN") = 2122.03247070312 ;
vxsb("c_Energy","EU_28","LatinAmer") = 3219.93237304688 ;
vxsb("c_Energy","EU_28","ROW") = 47244.91796875 ;
vxsb("c_Energy","CHN","USA") = 2249.47290039062 ;
vxsb("c_Energy","CHN","EU_28") = 1136.97241210938 ;
vxsb("c_Energy","CHN","CHN") = 6.00000021222513E-6 ;
vxsb("c_Energy","CHN","LatinAmer") = 943.893798828125 ;
vxsb("c_Energy","CHN","ROW") = 26725.4765625 ;
vxsb("c_Energy","LatinAmer","USA") = 41936.05859375 ;
vxsb("c_Energy","LatinAmer","EU_28") = 11052.50390625 ;
vxsb("c_Energy","LatinAmer","CHN") = 16630.09375 ;
vxsb("c_Energy","LatinAmer","LatinAmer") = 22378.640625 ;
vxsb("c_Energy","LatinAmer","ROW") = 18672.642578125 ;
vxsb("c_Energy","ROW","USA") = 153356.75 ;
vxsb("c_Energy","ROW","EU_28") = 348086.15625 ;
vxsb("c_Energy","ROW","CHN") = 194340.109375 ;
vxsb("c_Energy","ROW","LatinAmer") = 11928.1279296875 ;
vxsb("c_Energy","ROW","ROW") = 657282.1875 ;
vxsb("c_Manuf","USA","USA") = 1.89999991562217E-5 ;
vxsb("c_Manuf","USA","EU_28") = 208028 ;
vxsb("c_Manuf","USA","CHN") = 93425.46875 ;
vxsb("c_Manuf","USA","LatinAmer") = 274506.53125 ;
vxsb("c_Manuf","USA","ROW") = 549923.3125 ;
vxsb("c_Manuf","EU_28","USA") = 370627.15625 ;
vxsb("c_Manuf","EU_28","EU_28") = 2658550.25 ;
vxsb("c_Manuf","EU_28","CHN") = 251185 ;
vxsb("c_Manuf","EU_28","LatinAmer") = 115615.5859375 ;
vxsb("c_Manuf","EU_28","ROW") = 1035294.5625 ;
vxsb("c_Manuf","CHN","USA") = 448971.625 ;
vxsb("c_Manuf","CHN","EU_28") = 335411.21875 ;
vxsb("c_Manuf","CHN","CHN") = 1.89999991562217E-5 ;
vxsb("c_Manuf","CHN","LatinAmer") = 131926.734375 ;
vxsb("c_Manuf","CHN","ROW") = 1130336.875 ;
vxsb("c_Manuf","LatinAmer","USA") = 338947.4375 ;
vxsb("c_Manuf","LatinAmer","EU_28") = 49987.93359375 ;
vxsb("c_Manuf","LatinAmer","CHN") = 77448.1015625 ;
vxsb("c_Manuf","LatinAmer","LatinAmer") = 95349.421875 ;
vxsb("c_Manuf","LatinAmer","ROW") = 95603.0859375 ;
vxsb("c_Manuf","ROW","USA") = 708831.625 ;
vxsb("c_Manuf","ROW","EU_28") = 751006.1875 ;
vxsb("c_Manuf","ROW","CHN") = 1004213.25 ;
vxsb("c_Manuf","ROW","LatinAmer") = 131345.890625 ;
vxsb("c_Manuf","ROW","ROW") = 1816156.625 ;
vxsb("c_Svces","USA","USA") = 1.80000006366754E-5 ;
vxsb("c_Svces","USA","EU_28") = 223204.1875 ;
vxsb("c_Svces","USA","CHN") = 57874.48828125 ;
vxsb("c_Svces","USA","LatinAmer") = 93008.171875 ;
vxsb("c_Svces","USA","ROW") = 329372.75 ;
vxsb("c_Svces","EU_28","USA") = 199294.921875 ;
vxsb("c_Svces","EU_28","EU_28") = 1038910.9375 ;
vxsb("c_Svces","EU_28","CHN") = 52260.73046875 ;
vxsb("c_Svces","EU_28","LatinAmer") = 54577.70703125 ;
vxsb("c_Svces","EU_28","ROW") = 456310.21875 ;
vxsb("c_Svces","CHN","USA") = 19002.94921875 ;
vxsb("c_Svces","CHN","EU_28") = 35278.9296875 ;
vxsb("c_Svces","CHN","CHN") = 1.80000006366754E-5 ;
vxsb("c_Svces","CHN","LatinAmer") = 8103.0908203125 ;
vxsb("c_Svces","CHN","ROW") = 112547.0703125 ;
vxsb("c_Svces","LatinAmer","USA") = 67931.734375 ;
vxsb("c_Svces","LatinAmer","EU_28") = 45743.12890625 ;
vxsb("c_Svces","LatinAmer","CHN") = 11061.8701171875 ;
vxsb("c_Svces","LatinAmer","LatinAmer") = 20302.314453125 ;
vxsb("c_Svces","LatinAmer","ROW") = 52412.22265625 ;
vxsb("c_Svces","ROW","USA") = 260891.953125 ;
vxsb("c_Svces","ROW","EU_28") = 386634.65625 ;
vxsb("c_Svces","ROW","CHN") = 191478.390625 ;
vxsb("c_Svces","ROW","LatinAmer") = 46594.23046875 ;
vxsb("c_Svces","ROW","ROW") = 572524.25 ;
parameter vfob(*,*,*) ;
vfob("c_Agri","USA","USA") = 1.40000001920271E-5 ;
vfob("c_Agri","USA","EU_28") = 6831.80615234375 ;
vfob("c_Agri","USA","CHN") = 18099.328125 ;
vfob("c_Agri","USA","LatinAmer") = 13593.3095703125 ;
vfob("c_Agri","USA","ROW") = 37890.47265625 ;
vfob("c_Agri","EU_28","USA") = 2227.61083984375 ;
vfob("c_Agri","EU_28","EU_28") = 100227.4296875 ;
vfob("c_Agri","EU_28","CHN") = 2347.7548828125 ;
vfob("c_Agri","EU_28","LatinAmer") = 1016.29290771484 ;
vfob("c_Agri","EU_28","ROW") = 21999.62109375 ;
vfob("c_Agri","CHN","USA") = 917.361999511719 ;
vfob("c_Agri","CHN","EU_28") = 1707.060546875 ;
vfob("c_Agri","CHN","CHN") = 1.40000001920271E-5 ;
vfob("c_Agri","CHN","LatinAmer") = 379.70166015625 ;
vfob("c_Agri","CHN","ROW") = 15989.591796875 ;
vfob("c_Agri","LatinAmer","USA") = 28035.673828125 ;
vfob("c_Agri","LatinAmer","EU_28") = 17802.70703125 ;
vfob("c_Agri","LatinAmer","CHN") = 27357.92578125 ;
vfob("c_Agri","LatinAmer","LatinAmer") = 8879.8701171875 ;
vfob("c_Agri","LatinAmer","ROW") = 27051.103515625 ;
vfob("c_Agri","ROW","USA") = 15994.5107421875 ;
vfob("c_Agri","ROW","EU_28") = 37306.4375 ;
vfob("c_Agri","ROW","CHN") = 29119.193359375 ;
vfob("c_Agri","ROW","LatinAmer") = 2985.7939453125 ;
vfob("c_Agri","ROW","ROW") = 103831.1875 ;
vfob("c_Food","USA","USA") = 7.99999997980194E-6 ;
vfob("c_Food","USA","EU_28") = 6083.873046875 ;
vfob("c_Food","USA","CHN") = 4125.4521484375 ;
vfob("c_Food","USA","LatinAmer") = 20857.462890625 ;
vfob("c_Food","USA","ROW") = 48405.03125 ;
vfob("c_Food","EU_28","USA") = 23423.763671875 ;
vfob("c_Food","EU_28","EU_28") = 291054.4375 ;
vfob("c_Food","EU_28","CHN") = 13131.53515625 ;
vfob("c_Food","EU_28","LatinAmer") = 6786.76953125 ;
vfob("c_Food","EU_28","ROW") = 83116.28125 ;
vfob("c_Food","CHN","USA") = 7010.4384765625 ;
vfob("c_Food","CHN","EU_28") = 6218.8701171875 ;
vfob("c_Food","CHN","CHN") = 7.99999997980194E-6 ;
vfob("c_Food","CHN","LatinAmer") = 2068.73168945312 ;
vfob("c_Food","CHN","ROW") = 34390.14453125 ;
vfob("c_Food","LatinAmer","USA") = 25579.4609375 ;
vfob("c_Food","LatinAmer","EU_28") = 20379.017578125 ;
vfob("c_Food","LatinAmer","CHN") = 8027.95654296875 ;
vfob("c_Food","LatinAmer","LatinAmer") = 24203.828125 ;
vfob("c_Food","LatinAmer","ROW") = 49317.234375 ;
vfob("c_Food","ROW","USA") = 47703.93359375 ;
vfob("c_Food","ROW","EU_28") = 49416.89453125 ;
vfob("c_Food","ROW","CHN") = 30606.328125 ;
vfob("c_Food","ROW","LatinAmer") = 5191.38232421875 ;
vfob("c_Food","ROW","ROW") = 200275.75 ;
vfob("c_Energy","USA","USA") = 6.00000021222513E-6 ;
vfob("c_Energy","USA","EU_28") = 18447.859375 ;
vfob("c_Energy","USA","CHN") = 9279.0595703125 ;
vfob("c_Energy","USA","LatinAmer") = 75504.6953125 ;
vfob("c_Energy","USA","ROW") = 45237.8046875 ;
vfob("c_Energy","EU_28","USA") = 9830.814453125 ;
vfob("c_Energy","EU_28","EU_28") = 93616.8515625 ;
vfob("c_Energy","EU_28","CHN") = 2175.59594726562 ;
vfob("c_Energy","EU_28","LatinAmer") = 3310.33813476562 ;
vfob("c_Energy","EU_28","ROW") = 49320.37109375 ;
vfob("c_Energy","CHN","USA") = 2643.18969726562 ;
vfob("c_Energy","CHN","EU_28") = 1324.66149902344 ;
vfob("c_Energy","CHN","CHN") = 6.00000021222513E-6 ;
vfob("c_Energy","CHN","LatinAmer") = 1103.77136230469 ;
vfob("c_Energy","CHN","ROW") = 31133.92578125 ;
vfob("c_Energy","LatinAmer","USA") = 42105.828125 ;
vfob("c_Energy","LatinAmer","EU_28") = 11081.3701171875 ;
vfob("c_Energy","LatinAmer","CHN") = 16655.2109375 ;
vfob("c_Energy","LatinAmer","LatinAmer") = 23261.9921875 ;
vfob("c_Energy","LatinAmer","ROW") = 18741.12109375 ;
vfob("c_Energy","ROW","USA") = 154295.296875 ;
vfob("c_Energy","ROW","EU_28") = 351751.4375 ;
vfob("c_Energy","ROW","CHN") = 194884.8125 ;
vfob("c_Energy","ROW","LatinAmer") = 12005.6123046875 ;
vfob("c_Energy","ROW","ROW") = 663563.5 ;
vfob("c_Manuf","USA","USA") = 1.89999991562217E-5 ;
vfob("c_Manuf","USA","EU_28") = 218685.578125 ;
vfob("c_Manuf","USA","CHN") = 97459.71875 ;
vfob("c_Manuf","USA","LatinAmer") = 277893.125 ;
vfob("c_Manuf","USA","ROW") = 566183.125 ;
vfob("c_Manuf","EU_28","USA") = 370882.34375 ;
vfob("c_Manuf","EU_28","EU_28") = 2658550.25 ;
vfob("c_Manuf","EU_28","CHN") = 251305.078125 ;
vfob("c_Manuf","EU_28","LatinAmer") = 115692.109375 ;
vfob("c_Manuf","EU_28","ROW") = 1035924.9375 ;
vfob("c_Manuf","CHN","USA") = 456204.09375 ;
vfob("c_Manuf","CHN","EU_28") = 341157.75 ;
vfob("c_Manuf","CHN","CHN") = 1.89999991562217E-5 ;
vfob("c_Manuf","CHN","LatinAmer") = 134580.84375 ;
vfob("c_Manuf","CHN","ROW") = 1149776.875 ;
vfob("c_Manuf","LatinAmer","USA") = 339853 ;
vfob("c_Manuf","LatinAmer","EU_28") = 50724.765625 ;
vfob("c_Manuf","LatinAmer","CHN") = 78322.9140625 ;
vfob("c_Manuf","LatinAmer","LatinAmer") = 97245.0234375 ;
vfob("c_Manuf","LatinAmer","ROW") = 96708.296875 ;
vfob("c_Manuf","ROW","USA") = 712613.5 ;
vfob("c_Manuf","ROW","EU_28") = 756368.4375 ;
vfob("c_Manuf","ROW","CHN") = 1006679.875 ;
vfob("c_Manuf","ROW","LatinAmer") = 132544.90625 ;
vfob("c_Manuf","ROW","ROW") = 1830184.125 ;
vfob("c_Svces","USA","USA") = 1.80000006366754E-5 ;
vfob("c_Svces","USA","EU_28") = 223204.1875 ;
vfob("c_Svces","USA","CHN") = 57874.48828125 ;
vfob("c_Svces","USA","LatinAmer") = 93008.171875 ;
vfob("c_Svces","USA","ROW") = 329372.75 ;
vfob("c_Svces","EU_28","USA") = 199294.921875 ;
vfob("c_Svces","EU_28","EU_28") = 1038910.9375 ;
vfob("c_Svces","EU_28","CHN") = 52260.73046875 ;
vfob("c_Svces","EU_28","LatinAmer") = 54577.70703125 ;
vfob("c_Svces","EU_28","ROW") = 456310.21875 ;
vfob("c_Svces","CHN","USA") = 19002.94921875 ;
vfob("c_Svces","CHN","EU_28") = 35278.9296875 ;
vfob("c_Svces","CHN","CHN") = 1.80000006366754E-5 ;
vfob("c_Svces","CHN","LatinAmer") = 8103.0908203125 ;
vfob("c_Svces","CHN","ROW") = 112547.0703125 ;
vfob("c_Svces","LatinAmer","USA") = 67931.734375 ;
vfob("c_Svces","LatinAmer","EU_28") = 45743.12890625 ;
vfob("c_Svces","LatinAmer","CHN") = 11061.8701171875 ;
vfob("c_Svces","LatinAmer","LatinAmer") = 20302.314453125 ;
vfob("c_Svces","LatinAmer","ROW") = 52412.22265625 ;
vfob("c_Svces","ROW","USA") = 260891.953125 ;
vfob("c_Svces","ROW","EU_28") = 386634.65625 ;
vfob("c_Svces","ROW","CHN") = 191478.390625 ;
vfob("c_Svces","ROW","LatinAmer") = 46594.23046875 ;
vfob("c_Svces","ROW","ROW") = 572524.25 ;
parameter vcif(*,*,*) ;
vcif("c_Agri","USA","USA") = 1.40000001920271E-5 ;
vcif("c_Agri","USA","EU_28") = 7196.66162109375 ;
vcif("c_Agri","USA","CHN") = 19481.291015625 ;
vcif("c_Agri","USA","LatinAmer") = 14300.017578125 ;
vcif("c_Agri","USA","ROW") = 40709.66796875 ;
vcif("c_Agri","EU_28","USA") = 2375.72192382812 ;
vcif("c_Agri","EU_28","EU_28") = 109918.0625 ;
vcif("c_Agri","EU_28","CHN") = 2493.20043945312 ;
vcif("c_Agri","EU_28","LatinAmer") = 1107.17211914062 ;
vcif("c_Agri","EU_28","ROW") = 25121.66015625 ;
vcif("c_Agri","CHN","USA") = 982.840026855469 ;
vcif("c_Agri","CHN","EU_28") = 1814.302734375 ;
vcif("c_Agri","CHN","CHN") = 1.40000001920271E-5 ;
vcif("c_Agri","CHN","LatinAmer") = 415.475067138672 ;
vcif("c_Agri","CHN","ROW") = 18042.25 ;
vcif("c_Agri","LatinAmer","USA") = 30376.96484375 ;
vcif("c_Agri","LatinAmer","EU_28") = 19589.373046875 ;
vcif("c_Agri","LatinAmer","CHN") = 29670.63671875 ;
vcif("c_Agri","LatinAmer","LatinAmer") = 9745.60546875 ;
vcif("c_Agri","LatinAmer","ROW") = 30306.626953125 ;
vcif("c_Agri","ROW","USA") = 16579.83203125 ;
vcif("c_Agri","ROW","EU_28") = 40282.87890625 ;
vcif("c_Agri","ROW","CHN") = 32012.62109375 ;
vcif("c_Agri","ROW","LatinAmer") = 3138.751953125 ;
vcif("c_Agri","ROW","ROW") = 114163.5390625 ;
vcif("c_Food","USA","USA") = 7.99999997980194E-6 ;
vcif("c_Food","USA","EU_28") = 6412.86328125 ;
vcif("c_Food","USA","CHN") = 4330.11376953125 ;
vcif("c_Food","USA","LatinAmer") = 21618.25 ;
vcif("c_Food","USA","ROW") = 50443.953125 ;
vcif("c_Food","EU_28","USA") = 24805.072265625 ;
vcif("c_Food","EU_28","EU_28") = 301250.59375 ;
vcif("c_Food","EU_28","CHN") = 13801.9990234375 ;
vcif("c_Food","EU_28","LatinAmer") = 7166.63671875 ;
vcif("c_Food","EU_28","ROW") = 87406.90625 ;
vcif("c_Food","CHN","USA") = 7443.431640625 ;
vcif("c_Food","CHN","EU_28") = 6587.44189453125 ;
vcif("c_Food","CHN","CHN") = 7.99999997980194E-6 ;
vcif("c_Food","CHN","LatinAmer") = 2177.89794921875 ;
vcif("c_Food","CHN","ROW") = 36460.65625 ;
vcif("c_Food","LatinAmer","USA") = 26547.109375 ;
vcif("c_Food","LatinAmer","EU_28") = 21518.361328125 ;
vcif("c_Food","LatinAmer","CHN") = 8429.9091796875 ;
vcif("c_Food","LatinAmer","LatinAmer") = 25672.22265625 ;
vcif("c_Food","LatinAmer","ROW") = 51841.265625 ;
vcif("c_Food","ROW","USA") = 49509.87890625 ;
vcif("c_Food","ROW","EU_28") = 51903.8359375 ;
vcif("c_Food","ROW","CHN") = 32339.888671875 ;
vcif("c_Food","ROW","LatinAmer") = 5438.75634765625 ;
vcif("c_Food","ROW","ROW") = 211881.359375 ;
vcif("c_Energy","USA","USA") = 6.00000021222513E-6 ;
vcif("c_Energy","USA","EU_28") = 19536.767578125 ;
vcif("c_Energy","USA","CHN") = 9697.9189453125 ;
vcif("c_Energy","USA","LatinAmer") = 78759.2890625 ;
vcif("c_Energy","USA","ROW") = 47622.45703125 ;
vcif("c_Energy","EU_28","USA") = 10228.876953125 ;
vcif("c_Energy","EU_28","EU_28") = 96163.3515625 ;
vcif("c_Energy","EU_28","CHN") = 2238.27490234375 ;
vcif("c_Energy","EU_28","LatinAmer") = 3441.85620117188 ;
vcif("c_Energy","EU_28","ROW") = 51050.39453125 ;
vcif("c_Energy","CHN","USA") = 2754.91162109375 ;
vcif("c_Energy","CHN","EU_28") = 1383.09143066406 ;
vcif("c_Energy","CHN","CHN") = 6.00000021222513E-6 ;
vcif("c_Energy","CHN","LatinAmer") = 1172.69506835938 ;
vcif("c_Energy","CHN","ROW") = 32560.994140625 ;
vcif("c_Energy","LatinAmer","USA") = 43424.90625 ;
vcif("c_Energy","LatinAmer","EU_28") = 11617.9072265625 ;
vcif("c_Energy","LatinAmer","CHN") = 17059.806640625 ;
vcif("c_Energy","LatinAmer","LatinAmer") = 24170 ;
vcif("c_Energy","LatinAmer","ROW") = 19474.990234375 ;
vcif("c_Energy","ROW","USA") = 159153.171875 ;
vcif("c_Energy","ROW","EU_28") = 365853.21875 ;
vcif("c_Energy","ROW","CHN") = 203539.296875 ;
vcif("c_Energy","ROW","LatinAmer") = 12625.3115234375 ;
vcif("c_Energy","ROW","ROW") = 696969.0625 ;
vcif("c_Manuf","USA","USA") = 1.89999991562217E-5 ;
vcif("c_Manuf","USA","EU_28") = 224312.671875 ;
vcif("c_Manuf","USA","CHN") = 100655.609375 ;
vcif("c_Manuf","USA","LatinAmer") = 283045.375 ;
vcif("c_Manuf","USA","ROW") = 577529.5625 ;
vcif("c_Manuf","EU_28","USA") = 380387.1875 ;
vcif("c_Manuf","EU_28","EU_28") = 2717041 ;
vcif("c_Manuf","EU_28","CHN") = 258918.234375 ;
vcif("c_Manuf","EU_28","LatinAmer") = 119666.109375 ;
vcif("c_Manuf","EU_28","ROW") = 1066434.5 ;
vcif("c_Manuf","CHN","USA") = 474099.875 ;
vcif("c_Manuf","CHN","EU_28") = 354684.40625 ;
vcif("c_Manuf","CHN","CHN") = 1.89999991562217E-5 ;
vcif("c_Manuf","CHN","LatinAmer") = 139964.4375 ;
vcif("c_Manuf","CHN","ROW") = 1195940.125 ;
vcif("c_Manuf","LatinAmer","USA") = 343970.78125 ;
vcif("c_Manuf","LatinAmer","EU_28") = 53201.09375 ;
vcif("c_Manuf","LatinAmer","CHN") = 84094.4140625 ;
vcif("c_Manuf","LatinAmer","LatinAmer") = 101860.5234375 ;
vcif("c_Manuf","LatinAmer","ROW") = 101383.078125 ;
vcif("c_Manuf","ROW","USA") = 730954.375 ;
vcif("c_Manuf","ROW","EU_28") = 779977.875 ;
vcif("c_Manuf","ROW","CHN") = 1042370.8125 ;
vcif("c_Manuf","ROW","LatinAmer") = 137257.34375 ;
vcif("c_Manuf","ROW","ROW") = 1898909.375 ;
vcif("c_Svces","USA","USA") = 1.80000006366754E-5 ;
vcif("c_Svces","USA","EU_28") = 223204.1875 ;
vcif("c_Svces","USA","CHN") = 57874.48828125 ;
vcif("c_Svces","USA","LatinAmer") = 93008.171875 ;
vcif("c_Svces","USA","ROW") = 329372.75 ;
vcif("c_Svces","EU_28","USA") = 199294.921875 ;
vcif("c_Svces","EU_28","EU_28") = 1038910.9375 ;
vcif("c_Svces","EU_28","CHN") = 52260.73046875 ;
vcif("c_Svces","EU_28","LatinAmer") = 54577.70703125 ;
vcif("c_Svces","EU_28","ROW") = 456310.21875 ;
vcif("c_Svces","CHN","USA") = 19002.94921875 ;
vcif("c_Svces","CHN","EU_28") = 35278.9296875 ;
vcif("c_Svces","CHN","CHN") = 1.80000006366754E-5 ;
vcif("c_Svces","CHN","LatinAmer") = 8103.0908203125 ;
vcif("c_Svces","CHN","ROW") = 112547.0703125 ;
vcif("c_Svces","LatinAmer","USA") = 67931.734375 ;
vcif("c_Svces","LatinAmer","EU_28") = 45743.12890625 ;
vcif("c_Svces","LatinAmer","CHN") = 11061.8701171875 ;
vcif("c_Svces","LatinAmer","LatinAmer") = 20302.314453125 ;
vcif("c_Svces","LatinAmer","ROW") = 52412.22265625 ;
vcif("c_Svces","ROW","USA") = 260891.953125 ;
vcif("c_Svces","ROW","EU_28") = 386634.65625 ;
vcif("c_Svces","ROW","CHN") = 191478.390625 ;
vcif("c_Svces","ROW","LatinAmer") = 46594.23046875 ;
vcif("c_Svces","ROW","ROW") = 572524.25 ;
parameter vmsb(*,*,*) ;
vmsb("c_Agri","USA","USA") = 1.40000001920271E-5 ;
vmsb("c_Agri","USA","EU_28") = 7311.947265625 ;
vmsb("c_Agri","USA","CHN") = 20364.826171875 ;
vmsb("c_Agri","USA","LatinAmer") = 14583.9423828125 ;
vmsb("c_Agri","USA","ROW") = 43543.8046875 ;
vmsb("c_Agri","EU_28","USA") = 3062.2734375 ;
vmsb("c_Agri","EU_28","EU_28") = 109918.0625 ;
vmsb("c_Agri","EU_28","CHN") = 2714.98046875 ;
vmsb("c_Agri","EU_28","LatinAmer") = 1161.42333984375 ;
vmsb("c_Agri","EU_28","ROW") = 27932.822265625 ;
vmsb("c_Agri","CHN","USA") = 993.561218261719 ;
vmsb("c_Agri","CHN","EU_28") = 1908.62438964844 ;
vmsb("c_Agri","CHN","CHN") = 1.40000001920271E-5 ;
vmsb("c_Agri","CHN","LatinAmer") = 458.81787109375 ;
vmsb("c_Agri","CHN","ROW") = 18933.4453125 ;
vmsb("c_Agri","LatinAmer","USA") = 30421.97265625 ;
vmsb("c_Agri","LatinAmer","EU_28") = 19965.369140625 ;
vmsb("c_Agri","LatinAmer","CHN") = 30584.380859375 ;
vmsb("c_Agri","LatinAmer","LatinAmer") = 9926.263671875 ;
vmsb("c_Agri","LatinAmer","ROW") = 33311.59765625 ;
vmsb("c_Agri","ROW","USA") = 16708.470703125 ;
vmsb("c_Agri","ROW","EU_28") = 41002.89453125 ;
vmsb("c_Agri","ROW","CHN") = 33786.54296875 ;
vmsb("c_Agri","ROW","LatinAmer") = 3280.69506835938 ;
vmsb("c_Agri","ROW","ROW") = 123052.5 ;
vmsb("c_Food","USA","USA") = 7.99999997980194E-6 ;
vmsb("c_Food","USA","EU_28") = 6959.8818359375 ;
vmsb("c_Food","USA","CHN") = 4822.6044921875 ;
vmsb("c_Food","USA","LatinAmer") = 22427.033203125 ;
vmsb("c_Food","USA","ROW") = 56150.58984375 ;
vmsb("c_Food","EU_28","USA") = 25497.662109375 ;
vmsb("c_Food","EU_28","EU_28") = 301250.59375 ;
vmsb("c_Food","EU_28","CHN") = 15559.0869140625 ;
vmsb("c_Food","EU_28","LatinAmer") = 7930.13720703125 ;
vmsb("c_Food","EU_28","ROW") = 100026.015625 ;
vmsb("c_Food","CHN","USA") = 7661.97412109375 ;
vmsb("c_Food","CHN","EU_28") = 7189.73876953125 ;
vmsb("c_Food","CHN","CHN") = 7.99999997980194E-6 ;
vmsb("c_Food","CHN","LatinAmer") = 2523.89208984375 ;
vmsb("c_Food","CHN","ROW") = 39409.0859375 ;
vmsb("c_Food","LatinAmer","USA") = 27097.2421875 ;
vmsb("c_Food","LatinAmer","EU_28") = 23245.853515625 ;
vmsb("c_Food","LatinAmer","CHN") = 9309.0234375 ;
vmsb("c_Food","LatinAmer","LatinAmer") = 26729.275390625 ;
vmsb("c_Food","LatinAmer","ROW") = 58129.625 ;
vmsb("c_Food","ROW","USA") = 50427.328125 ;
vmsb("c_Food","ROW","EU_28") = 54344.8984375 ;
vmsb("c_Food","ROW","CHN") = 34603.79296875 ;
vmsb("c_Food","ROW","LatinAmer") = 6060.07080078125 ;
vmsb("c_Food","ROW","ROW") = 231768.140625 ;
vmsb("c_Energy","USA","USA") = 6.00000021222513E-6 ;
vmsb("c_Energy","USA","EU_28") = 19696.4140625 ;
vmsb("c_Energy","USA","CHN") = 10015.685546875 ;
vmsb("c_Energy","USA","LatinAmer") = 79587.359375 ;
vmsb("c_Energy","USA","ROW") = 48325.46484375 ;
vmsb("c_Energy","EU_28","USA") = 10408.638671875 ;
vmsb("c_Energy","EU_28","EU_28") = 96163.3515625 ;
vmsb("c_Energy","EU_28","CHN") = 2284.99072265625 ;
vmsb("c_Energy","EU_28","LatinAmer") = 3474.2080078125 ;
vmsb("c_Energy","EU_28","ROW") = 52301.1328125 ;
vmsb("c_Energy","CHN","USA") = 2793.5498046875 ;
vmsb("c_Energy","CHN","EU_28") = 1400.75073242188 ;
vmsb("c_Energy","CHN","CHN") = 6.00000021222513E-6 ;
vmsb("c_Energy","CHN","LatinAmer") = 1190.63073730469 ;
vmsb("c_Energy","CHN","ROW") = 33304.90234375 ;
vmsb("c_Energy","LatinAmer","USA") = 43528.59375 ;
vmsb("c_Energy","LatinAmer","EU_28") = 11635.419921875 ;
vmsb("c_Energy","LatinAmer","CHN") = 17088.125 ;
vmsb("c_Energy","LatinAmer","LatinAmer") = 24420.044921875 ;
vmsb("c_Energy","LatinAmer","ROW") = 19617.4140625 ;
vmsb("c_Energy","ROW","USA") = 159587.96875 ;
vmsb("c_Energy","ROW","EU_28") = 366708.28125 ;
vmsb("c_Energy","ROW","CHN") = 205200.75 ;
vmsb("c_Energy","ROW","LatinAmer") = 12745.6005859375 ;
vmsb("c_Energy","ROW","ROW") = 704190.5 ;
vmsb("c_Manuf","USA","USA") = 1.89999991562217E-5 ;
vmsb("c_Manuf","USA","EU_28") = 228749.84375 ;
vmsb("c_Manuf","USA","CHN") = 107682.3984375 ;
vmsb("c_Manuf","USA","LatinAmer") = 289036.75 ;
vmsb("c_Manuf","USA","ROW") = 585280.4375 ;
vmsb("c_Manuf","EU_28","USA") = 385888.96875 ;
vmsb("c_Manuf","EU_28","EU_28") = 2717041 ;
vmsb("c_Manuf","EU_28","CHN") = 279350.875 ;
vmsb("c_Manuf","EU_28","LatinAmer") = 126009.46875 ;
vmsb("c_Manuf","EU_28","ROW") = 1094899.25 ;
vmsb("c_Manuf","CHN","USA") = 487624.75 ;
vmsb("c_Manuf","CHN","EU_28") = 368135.40625 ;
vmsb("c_Manuf","CHN","CHN") = 1.89999991562217E-5 ;
vmsb("c_Manuf","CHN","LatinAmer") = 151566 ;
vmsb("c_Manuf","CHN","ROW") = 1244436.25 ;
vmsb("c_Manuf","LatinAmer","USA") = 345057.46875 ;
vmsb("c_Manuf","LatinAmer","EU_28") = 53544.38671875 ;
vmsb("c_Manuf","LatinAmer","CHN") = 85091.2265625 ;
vmsb("c_Manuf","LatinAmer","LatinAmer") = 103770.796875 ;
vmsb("c_Manuf","LatinAmer","ROW") = 103119.953125 ;
vmsb("c_Manuf","ROW","USA") = 742175.25 ;
vmsb("c_Manuf","ROW","EU_28") = 789506.5625 ;
vmsb("c_Manuf","ROW","CHN") = 1070629.5 ;
vmsb("c_Manuf","ROW","LatinAmer") = 146303.796875 ;
vmsb("c_Manuf","ROW","ROW") = 1951682.375 ;
vmsb("c_Svces","USA","USA") = 1.80000006366754E-5 ;
vmsb("c_Svces","USA","EU_28") = 223204.1875 ;
vmsb("c_Svces","USA","CHN") = 57874.48828125 ;
vmsb("c_Svces","USA","LatinAmer") = 93008.171875 ;
vmsb("c_Svces","USA","ROW") = 329372.75 ;
vmsb("c_Svces","EU_28","USA") = 199294.921875 ;
vmsb("c_Svces","EU_28","EU_28") = 1038910.9375 ;
vmsb("c_Svces","EU_28","CHN") = 52260.73046875 ;
vmsb("c_Svces","EU_28","LatinAmer") = 54577.70703125 ;
vmsb("c_Svces","EU_28","ROW") = 456310.21875 ;
vmsb("c_Svces","CHN","USA") = 19002.94921875 ;
vmsb("c_Svces","CHN","EU_28") = 35278.9296875 ;
vmsb("c_Svces","CHN","CHN") = 1.80000006366754E-5 ;
vmsb("c_Svces","CHN","LatinAmer") = 8103.0908203125 ;
vmsb("c_Svces","CHN","ROW") = 112547.0703125 ;
vmsb("c_Svces","LatinAmer","USA") = 67931.734375 ;
vmsb("c_Svces","LatinAmer","EU_28") = 45743.12890625 ;
vmsb("c_Svces","LatinAmer","CHN") = 11061.8701171875 ;
vmsb("c_Svces","LatinAmer","LatinAmer") = 20302.314453125 ;
vmsb("c_Svces","LatinAmer","ROW") = 52412.22265625 ;
vmsb("c_Svces","ROW","USA") = 260891.953125 ;
vmsb("c_Svces","ROW","EU_28") = 386634.65625 ;
vmsb("c_Svces","ROW","CHN") = 191478.390625 ;
vmsb("c_Svces","ROW","LatinAmer") = 46594.23046875 ;
vmsb("c_Svces","ROW","ROW") = 572524.25 ;
parameter vst(*,*) ;
vst("c_Svces","USA") = 31137.61328125 ;
vst("c_Svces","EU_28") = 275190.5 ;
vst("c_Svces","CHN") = 24354.564453125 ;
vst("c_Svces","LatinAmer") = 15379.783203125 ;
vst("c_Svces","ROW") = 220126.796875 ;
parameter vtwr(*,*,*,*) ;
vtwr("c_Svces","c_Agri","USA","EU_28") = 364.855590820312 ;
vtwr("c_Svces","c_Agri","USA","CHN") = 1381.96276855469 ;
vtwr("c_Svces","c_Agri","USA","LatinAmer") = 706.70556640625 ;
vtwr("c_Svces","c_Agri","USA","ROW") = 2819.19165039062 ;
vtwr("c_Svces","c_Agri","EU_28","USA") = 148.111114501953 ;
vtwr("c_Svces","c_Agri","EU_28","EU_28") = 9690.65234375 ;
vtwr("c_Svces","c_Agri","EU_28","CHN") = 145.445434570312 ;
vtwr("c_Svces","c_Agri","EU_28","LatinAmer") = 90.8786697387695 ;
vtwr("c_Svces","c_Agri","EU_28","ROW") = 3122.04296875 ;
vtwr("c_Svces","c_Agri","CHN","USA") = 65.4780120849609 ;
vtwr("c_Svces","c_Agri","CHN","EU_28") = 107.242172241211 ;
vtwr("c_Svces","c_Agri","CHN","LatinAmer") = 35.7734031677246 ;
vtwr("c_Svces","c_Agri","CHN","ROW") = 2052.65966796875 ;
vtwr("c_Svces","c_Agri","LatinAmer","USA") = 2341.29125976562 ;
vtwr("c_Svces","c_Agri","LatinAmer","EU_28") = 1786.67016601562 ;
vtwr("c_Svces","c_Agri","LatinAmer","CHN") = 2312.712890625 ;
vtwr("c_Svces","c_Agri","LatinAmer","LatinAmer") = 865.735290527344 ;
vtwr("c_Svces","c_Agri","LatinAmer","ROW") = 3255.517578125 ;
vtwr("c_Svces","c_Agri","ROW","USA") = 585.321411132812 ;
vtwr("c_Svces","c_Agri","ROW","EU_28") = 2976.43920898438 ;
vtwr("c_Svces","c_Agri","ROW","CHN") = 2893.42578125 ;
vtwr("c_Svces","c_Agri","ROW","LatinAmer") = 152.958084106445 ;
vtwr("c_Svces","c_Agri","ROW","ROW") = 10332.3525390625 ;
vtwr("c_Svces","c_Food","USA","EU_28") = 328.98779296875 ;
vtwr("c_Svces","c_Food","USA","CHN") = 204.661972045898 ;
vtwr("c_Svces","c_Food","USA","LatinAmer") = 760.794738769531 ;
vtwr("c_Svces","c_Food","USA","ROW") = 2038.92504882812 ;
vtwr("c_Svces","c_Food","EU_28","USA") = 1381.30847167969 ;
vtwr("c_Svces","c_Food","EU_28","EU_28") = 10196.2431640625 ;
vtwr("c_Svces","c_Food","EU_28","CHN") = 670.4638671875 ;
vtwr("c_Svces","c_Food","EU_28","LatinAmer") = 379.866882324219 ;
vtwr("c_Svces","c_Food","EU_28","ROW") = 4290.62451171875 ;
vtwr("c_Svces","c_Food","CHN","USA") = 432.9931640625 ;
vtwr("c_Svces","c_Food","CHN","EU_28") = 368.574645996094 ;
vtwr("c_Svces","c_Food","CHN","LatinAmer") = 109.165985107422 ;
vtwr("c_Svces","c_Food","CHN","ROW") = 2070.50830078125 ;
vtwr("c_Svces","c_Food","LatinAmer","USA") = 967.648010253906 ;
vtwr("c_Svces","c_Food","LatinAmer","EU_28") = 1139.34411621094 ;
vtwr("c_Svces","c_Food","LatinAmer","CHN") = 401.953125 ;
vtwr("c_Svces","c_Food","LatinAmer","LatinAmer") = 1468.39294433594 ;
vtwr("c_Svces","c_Food","LatinAmer","ROW") = 2524.03564453125 ;
vtwr("c_Svces","c_Food","ROW","USA") = 1805.94482421875 ;
vtwr("c_Svces","c_Food","ROW","EU_28") = 2486.95483398438 ;
vtwr("c_Svces","c_Food","ROW","CHN") = 1733.56079101562 ;
vtwr("c_Svces","c_Food","ROW","LatinAmer") = 247.373519897461 ;
vtwr("c_Svces","c_Food","ROW","ROW") = 11605.60546875 ;
vtwr("c_Svces","c_Energy","USA","EU_28") = 1088.90905761719 ;
vtwr("c_Svces","c_Energy","USA","CHN") = 418.859436035156 ;
vtwr("c_Svces","c_Energy","USA","LatinAmer") = 3254.59228515625 ;
vtwr("c_Svces","c_Energy","USA","ROW") = 2384.65502929688 ;
vtwr("c_Svces","c_Energy","EU_28","USA") = 398.062286376953 ;
vtwr("c_Svces","c_Energy","EU_28","EU_28") = 2546.50463867188 ;
vtwr("c_Svces","c_Energy","EU_28","CHN") = 62.6789474487305 ;
vtwr("c_Svces","c_Energy","EU_28","LatinAmer") = 131.517913818359 ;
vtwr("c_Svces","c_Energy","EU_28","ROW") = 1730.02209472656 ;
vtwr("c_Svces","c_Energy","CHN","USA") = 111.721694946289 ;
vtwr("c_Svces","c_Energy","CHN","EU_28") = 58.4298782348633 ;
vtwr("c_Svces","c_Energy","CHN","LatinAmer") = 68.9236831665039 ;
vtwr("c_Svces","c_Energy","CHN","ROW") = 1427.06872558594 ;
vtwr("c_Svces","c_Energy","LatinAmer","USA") = 1319.07946777344 ;
vtwr("c_Svces","c_Energy","LatinAmer","EU_28") = 536.536499023438 ;
vtwr("c_Svces","c_Energy","LatinAmer","CHN") = 404.59619140625 ;
vtwr("c_Svces","c_Energy","LatinAmer","LatinAmer") = 908.008666992188 ;
vtwr("c_Svces","c_Energy","LatinAmer","ROW") = 733.868286132812 ;
vtwr("c_Svces","c_Energy","ROW","USA") = 4857.8701171875 ;
vtwr("c_Svces","c_Energy","ROW","EU_28") = 14101.7802734375 ;
vtwr("c_Svces","c_Energy","ROW","CHN") = 8654.4931640625 ;
vtwr("c_Svces","c_Energy","ROW","LatinAmer") = 619.69970703125 ;
vtwr("c_Svces","c_Energy","ROW","ROW") = 33405.5625 ;
vtwr("c_Svces","c_Manuf","USA","EU_28") = 5627.15869140625 ;
vtwr("c_Svces","c_Manuf","USA","CHN") = 3195.8935546875 ;
vtwr("c_Svces","c_Manuf","USA","LatinAmer") = 5152.26806640625 ;
vtwr("c_Svces","c_Manuf","USA","ROW") = 11346.4873046875 ;
vtwr("c_Svces","c_Manuf","EU_28","USA") = 9504.830078125 ;
vtwr("c_Svces","c_Manuf","EU_28","EU_28") = 58491.35546875 ;
vtwr("c_Svces","c_Manuf","EU_28","CHN") = 7613.1640625 ;
vtwr("c_Svces","c_Manuf","EU_28","LatinAmer") = 3973.92309570312 ;
vtwr("c_Svces","c_Manuf","EU_28","ROW") = 30509.572265625 ;
vtwr("c_Svces","c_Manuf","CHN","USA") = 17895.796875 ;
vtwr("c_Svces","c_Manuf","CHN","EU_28") = 13526.8544921875 ;
vtwr("c_Svces","c_Manuf","CHN","LatinAmer") = 5383.5302734375 ;
vtwr("c_Svces","c_Manuf","CHN","ROW") = 46163.25390625 ;
vtwr("c_Svces","c_Manuf","LatinAmer","USA") = 4117.7685546875 ;
vtwr("c_Svces","c_Manuf","LatinAmer","EU_28") = 2476.32739257812 ;
vtwr("c_Svces","c_Manuf","LatinAmer","CHN") = 5771.4990234375 ;
vtwr("c_Svces","c_Manuf","LatinAmer","LatinAmer") = 4615.49169921875 ;
vtwr("c_Svces","c_Manuf","LatinAmer","ROW") = 4674.78466796875 ;
vtwr("c_Svces","c_Manuf","ROW","USA") = 18340.875 ;
vtwr("c_Svces","c_Manuf","ROW","EU_28") = 23609.78515625 ;
vtwr("c_Svces","c_Manuf","ROW","CHN") = 35690.93359375 ;
vtwr("c_Svces","c_Manuf","ROW","LatinAmer") = 4712.3896484375 ;
vtwr("c_Svces","c_Manuf","ROW","ROW") = 68725.1875 ;
parameter save(*) ;
save("USA") = 1580785.625 ;
save("EU_28") = 2133911.25 ;
save("CHN") = 3840168.5 ;
save("LatinAmer") = 474877.5 ;
save("ROW") = 4161104.25 ;
parameter vdep(*) ;
vdep("USA") = 1816268 ;
vdep("EU_28") = 1809763.25 ;
vdep("CHN") = 1607693.375 ;
vdep("LatinAmer") = 568090.375 ;
vdep("ROW") = 2746245.75 ;
parameter vkb(*) ;
vkb("USA") = 45406700 ;
vkb("EU_28") = 45244080 ;
vkb("CHN") = 40192336 ;
vkb("LatinAmer") = 14202259 ;
vkb("ROW") = 68656144 ;
parameter pop0(*) ;
pop0("USA") = 325.147125244141 ;
pop0("EU_28") = 513.872253417969 ;
pop0("CHN") = 1386.39501953125 ;
pop0("LatinAmer") = 635.704956054688 ;
pop0("ROW") = 4652.7646484375 ;
parameter maks(*,*,*) ;
maks("c_Agri","a_Agri","USA") = 461880.9375 ;
maks("c_Agri","a_Agri","EU_28") = 526471.25 ;
maks("c_Agri","a_Agri","CHN") = 1454949.5 ;
maks("c_Agri","a_Agri","LatinAmer") = 519430.3125 ;
maks("c_Agri","a_Agri","ROW") = 2393759.5 ;
maks("c_Food","a_Food","USA") = 1117354.5 ;
maks("c_Food","a_Food","EU_28") = 1460662.5 ;
maks("c_Food","a_Food","CHN") = 1693269.875 ;
maks("c_Food","a_Food","LatinAmer") = 804895.3125 ;
maks("c_Food","a_Food","ROW") = 2526829.75 ;
maks("c_Energy","a_Energy","USA") = 1082570.75 ;
maks("c_Energy","a_Energy","EU_28") = 885865.4375 ;
maks("c_Energy","a_Energy","CHN") = 1266234 ;
maks("c_Energy","a_Energy","LatinAmer") = 538326 ;
maks("c_Energy","a_Energy","ROW") = 4193517.5 ;
maks("c_Manuf","a_Manuf","USA") = 5078237.5 ;
maks("c_Manuf","a_Manuf","EU_28") = 8101507.5 ;
maks("c_Manuf","a_Manuf","CHN") = 11994476 ;
maks("c_Manuf","a_Manuf","LatinAmer") = 2163178.25 ;
maks("c_Manuf","a_Manuf","ROW") = 12203364 ;
maks("c_Svces","a_Svces","USA") = 24328918 ;
maks("c_Svces","a_Svces","EU_28") = 22564282 ;
maks("c_Svces","a_Svces","CHN") = 14255377 ;
maks("c_Svces","a_Svces","LatinAmer") = 5760137.5 ;
maks("c_Svces","a_Svces","ROW") = 29442938 ;
parameter makb(*,*,*) ;
makb("c_Agri","a_Agri","USA") = 463691 ;
makb("c_Agri","a_Agri","EU_28") = 524524.9375 ;
makb("c_Agri","a_Agri","CHN") = 1451209.125 ;
makb("c_Agri","a_Agri","LatinAmer") = 519164.84375 ;
makb("c_Agri","a_Agri","ROW") = 2390844.5 ;
makb("c_Food","a_Food","USA") = 1156894.375 ;
makb("c_Food","a_Food","EU_28") = 1462588.625 ;
makb("c_Food","a_Food","CHN") = 1693453.875 ;
makb("c_Food","a_Food","LatinAmer") = 813330.1875 ;
makb("c_Food","a_Food","ROW") = 2563683.5 ;
makb("c_Energy","a_Energy","USA") = 1163954.125 ;
makb("c_Energy","a_Energy","EU_28") = 890106.5 ;
makb("c_Energy","a_Energy","CHN") = 1266388.875 ;
makb("c_Energy","a_Energy","LatinAmer") = 537509.875 ;
makb("c_Energy","a_Energy","ROW") = 4237631 ;
makb("c_Manuf","a_Manuf","USA") = 5173289.5 ;
makb("c_Manuf","a_Manuf","EU_28") = 8125949 ;
makb("c_Manuf","a_Manuf","CHN") = 11995360 ;
makb("c_Manuf","a_Manuf","LatinAmer") = 2182211.5 ;
makb("c_Manuf","a_Manuf","ROW") = 12263843 ;
makb("c_Svces","a_Svces","USA") = 25332292 ;
makb("c_Svces","a_Svces","EU_28") = 22761286 ;
makb("c_Svces","a_Svces","CHN") = 14256120 ;
makb("c_Svces","a_Svces","LatinAmer") = 5809796.5 ;
makb("c_Svces","a_Svces","ROW") = 29814150 ;
parameter esubt(*,*) ;
parameter esubc(*,*) ;
parameter esubva(*,*) ;
esubva("a_Agri","USA") = 0.249639242887497 ;
esubva("a_Agri","EU_28") = 0.249639242887497 ;
esubva("a_Agri","CHN") = 0.249639242887497 ;
esubva("a_Agri","LatinAmer") = 0.249639242887497 ;
esubva("a_Agri","ROW") = 0.249639242887497 ;
esubva("a_Food","USA") = 1.12000000476837 ;
esubva("a_Food","EU_28") = 1.12000000476837 ;
esubva("a_Food","CHN") = 1.12000000476837 ;
esubva("a_Food","LatinAmer") = 1.12000000476837 ;
esubva("a_Food","ROW") = 1.12000000476837 ;
esubva("a_Energy","USA") = 0.716633558273315 ;
esubva("a_Energy","EU_28") = 0.716633558273315 ;
esubva("a_Energy","CHN") = 0.716633558273315 ;
esubva("a_Energy","LatinAmer") = 0.716633558273315 ;
esubva("a_Energy","ROW") = 0.716633558273315 ;
esubva("a_Manuf","USA") = 1.19856810569763 ;
esubva("a_Manuf","EU_28") = 1.19856810569763 ;
esubva("a_Manuf","CHN") = 1.19856810569763 ;
esubva("a_Manuf","LatinAmer") = 1.19856810569763 ;
esubva("a_Manuf","ROW") = 1.19856810569763 ;
esubva("a_Svces","USA") = 1.37246572971344 ;
esubva("a_Svces","EU_28") = 1.37246572971344 ;
esubva("a_Svces","CHN") = 1.37246572971344 ;
esubva("a_Svces","LatinAmer") = 1.37246572971344 ;
esubva("a_Svces","ROW") = 1.37246572971344 ;
parameter etraq(*,*) ;
etraq("a_Agri","USA") = -5 ;
etraq("a_Agri","EU_28") = -5 ;
etraq("a_Agri","CHN") = -5 ;
etraq("a_Agri","LatinAmer") = -5 ;
etraq("a_Agri","ROW") = -5 ;
etraq("a_Food","USA") = -5 ;
etraq("a_Food","EU_28") = -5 ;
etraq("a_Food","CHN") = -5 ;
etraq("a_Food","LatinAmer") = -5 ;
etraq("a_Food","ROW") = -5 ;
etraq("a_Energy","USA") = -5 ;
etraq("a_Energy","EU_28") = -5 ;
etraq("a_Energy","CHN") = -5 ;
etraq("a_Energy","LatinAmer") = -5 ;
etraq("a_Energy","ROW") = -5 ;
etraq("a_Manuf","USA") = -5 ;
etraq("a_Manuf","EU_28") = -5 ;
etraq("a_Manuf","CHN") = -5 ;
etraq("a_Manuf","LatinAmer") = -5 ;
etraq("a_Manuf","ROW") = -5 ;
etraq("a_Svces","USA") = -5 ;
etraq("a_Svces","EU_28") = -5 ;
etraq("a_Svces","CHN") = -5 ;
etraq("a_Svces","LatinAmer") = -5 ;
etraq("a_Svces","ROW") = -5 ;
parameter esubq(*,*) ;
parameter incpar(*,*) ;
incpar("c_Agri","USA") = 0.150164246559143 ;
incpar("c_Agri","EU_28") = 0.18517230451107 ;
incpar("c_Agri","CHN") = 0.281542360782623 ;
incpar("c_Agri","LatinAmer") = 0.313482046127319 ;
incpar("c_Agri","ROW") = 0.480216294527054 ;
incpar("c_Food","USA") = 0.40962216258049 ;
incpar("c_Food","EU_28") = 0.463676422834396 ;
incpar("c_Food","CHN") = 0.377050280570984 ;
incpar("c_Food","LatinAmer") = 0.436011463403702 ;
incpar("c_Food","ROW") = 0.48481285572052 ;
incpar("c_Energy","USA") = 0.913325369358063 ;
incpar("c_Energy","EU_28") = 0.993315696716309 ;
incpar("c_Energy","CHN") = 1.02250564098358 ;
incpar("c_Energy","LatinAmer") = 0.98811274766922 ;
incpar("c_Energy","ROW") = 0.989670991897583 ;
incpar("c_Manuf","USA") = 0.88782674074173 ;
incpar("c_Manuf","EU_28") = 0.900741934776306 ;
incpar("c_Manuf","CHN") = 0.898754000663757 ;
incpar("c_Manuf","LatinAmer") = 0.923477947711945 ;
incpar("c_Manuf","ROW") = 0.873705744743347 ;
incpar("c_Svces","USA") = 1.06106984615326 ;
incpar("c_Svces","EU_28") = 1.12457609176636 ;
incpar("c_Svces","CHN") = 1.3661208152771 ;
incpar("c_Svces","LatinAmer") = 1.17292046546936 ;
incpar("c_Svces","ROW") = 1.1856974363327 ;
parameter subpar(*,*) ;
subpar("c_Agri","USA") = 0.753040730953217 ;
subpar("c_Agri","EU_28") = 0.810137152671814 ;
subpar("c_Agri","CHN") = 0.779426276683807 ;
subpar("c_Agri","LatinAmer") = 0.800834357738495 ;
subpar("c_Agri","ROW") = 0.815915942192078 ;
subpar("c_Food","USA") = 0.197535946965218 ;
subpar("c_Food","EU_28") = 0.398628771305084 ;
subpar("c_Food","CHN") = 0.72175258398056 ;
subpar("c_Food","LatinAmer") = 0.702683925628662 ;
subpar("c_Food","ROW") = 0.650423526763916 ;
subpar("c_Energy","USA") = 0.0938753336668015 ;
subpar("c_Energy","EU_28") = 0.231758579611778 ;
subpar("c_Energy","CHN") = 0.493528544902802 ;
subpar("c_Energy","LatinAmer") = 0.503643333911896 ;
subpar("c_Energy","ROW") = 0.437736004590988 ;
subpar("c_Manuf","USA") = 0.100222982466221 ;
subpar("c_Manuf","EU_28") = 0.252668619155884 ;
subpar("c_Manuf","CHN") = 0.534549117088318 ;
subpar("c_Manuf","LatinAmer") = 0.526163995265961 ;
subpar("c_Manuf","ROW") = 0.459088265895844 ;
subpar("c_Svces","USA") = 0.0879293978214264 ;
subpar("c_Svces","EU_28") = 0.206386521458626 ;
subpar("c_Svces","CHN") = 0.442120015621185 ;
subpar("c_Svces","LatinAmer") = 0.459991902112961 ;
subpar("c_Svces","ROW") = 0.337694615125656 ;
parameter esubg(*) ;
esubg("USA") = 1 ;
esubg("EU_28") = 1 ;
esubg("CHN") = 1 ;
esubg("LatinAmer") = 1 ;
esubg("ROW") = 1 ;
parameter esubi(*) ;
parameter esubd(*,*) ;
esubd("c_Agri","USA") = 2.34542512893677 ;
esubd("c_Agri","EU_28") = 2.34542512893677 ;
esubd("c_Agri","CHN") = 2.34542512893677 ;
esubd("c_Agri","LatinAmer") = 2.34542512893677 ;
esubd("c_Agri","ROW") = 2.34542512893677 ;
esubd("c_Food","USA") = 2.483478307724 ;
esubd("c_Food","EU_28") = 2.483478307724 ;
esubd("c_Food","CHN") = 2.483478307724 ;
esubd("c_Food","LatinAmer") = 2.483478307724 ;
esubd("c_Food","ROW") = 2.483478307724 ;
esubd("c_Energy","USA") = 3.81850719451904 ;
esubd("c_Energy","EU_28") = 3.81850719451904 ;
esubd("c_Energy","CHN") = 3.81850719451904 ;
esubd("c_Energy","LatinAmer") = 3.81850719451904 ;
esubd("c_Energy","ROW") = 3.81850719451904 ;
esubd("c_Manuf","USA") = 3.52019023895264 ;
esubd("c_Manuf","EU_28") = 3.52019023895264 ;
esubd("c_Manuf","CHN") = 3.52019023895264 ;
esubd("c_Manuf","LatinAmer") = 3.52019023895264 ;
esubd("c_Manuf","ROW") = 3.52019023895264 ;
esubd("c_Svces","USA") = 1.90913510322571 ;
esubd("c_Svces","EU_28") = 1.90913510322571 ;
esubd("c_Svces","CHN") = 1.90913510322571 ;
esubd("c_Svces","LatinAmer") = 1.90913510322571 ;
esubd("c_Svces","ROW") = 1.90913510322571 ;
parameter esubm(*,*) ;
esubm("c_Agri","USA") = 4.64395332336426 ;
esubm("c_Agri","EU_28") = 4.64395332336426 ;
esubm("c_Agri","CHN") = 4.64395332336426 ;
esubm("c_Agri","LatinAmer") = 4.64395332336426 ;
esubm("c_Agri","ROW") = 4.64395332336426 ;
esubm("c_Food","USA") = 4.82314729690552 ;
esubm("c_Food","EU_28") = 4.82314729690552 ;
esubm("c_Food","CHN") = 4.82314729690552 ;
esubm("c_Food","LatinAmer") = 4.82314729690552 ;
esubm("c_Food","ROW") = 4.82314729690552 ;
esubm("c_Energy","USA") = 11.4968242645264 ;
esubm("c_Energy","EU_28") = 11.4968242645264 ;
esubm("c_Energy","CHN") = 11.4968242645264 ;
esubm("c_Energy","LatinAmer") = 11.4968242645264 ;
esubm("c_Energy","ROW") = 11.4968242645264 ;
esubm("c_Manuf","USA") = 7.31825637817383 ;
esubm("c_Manuf","EU_28") = 7.31825637817383 ;
esubm("c_Manuf","CHN") = 7.31825637817383 ;
esubm("c_Manuf","LatinAmer") = 7.31825637817383 ;
esubm("c_Manuf","ROW") = 7.31825637817383 ;
esubm("c_Svces","USA") = 3.80041527748108 ;
esubm("c_Svces","EU_28") = 3.80041527748108 ;
esubm("c_Svces","CHN") = 3.80041527748108 ;
esubm("c_Svces","LatinAmer") = 3.80041527748108 ;
esubm("c_Svces","ROW") = 3.80041527748108 ;
parameter esubs(*) ;
esubs("c_Svces") = 1 ;
parameter etrae(*,*) ;
etrae("Land","USA") = -1 ;
etrae("Land","EU_28") = -1 ;
etrae("Land","CHN") = -1 ;
etrae("Land","LatinAmer") = -1 ;
etrae("Land","ROW") = -1 ;
etrae("NatRes","USA") = -0.00100000004749745 ;
etrae("NatRes","EU_28") = -0.00100000004749745 ;
etrae("NatRes","CHN") = -0.00100000004749745 ;
etrae("NatRes","LatinAmer") = -0.00100000004749745 ;
etrae("NatRes","ROW") = -0.00100000004749745 ;
parameter rorFlex0(*) ;
rorFlex0("USA") = 10 ;
rorFlex0("EU_28") = 10 ;
rorFlex0("CHN") = 10 ;
rorFlex0("LatinAmer") = 10 ;
rorFlex0("ROW") = 10 ;

parameter esubd(*,*) ;
esubd("c_Agri","USA") = 2.3454251289367676 ;
esubd("c_Food","USA") = 2.483478307723999 ;
esubd("c_Energy","USA") = 3.818507194519043 ;
esubd("c_Manuf","USA") = 3.5201902389526367 ;
esubd("c_Svces","USA") = 1.909135103225708 ;
esubd("c_Agri","EU_28") = 2.3454251289367676 ;
esubd("c_Food","EU_28") = 2.483478307723999 ;
esubd("c_Energy","EU_28") = 3.818507194519043 ;
esubd("c_Manuf","EU_28") = 3.5201902389526367 ;
esubd("c_Svces","EU_28") = 1.909135103225708 ;
esubd("c_Agri","CHN") = 2.3454251289367676 ;
esubd("c_Food","CHN") = 2.483478307723999 ;
esubd("c_Energy","CHN") = 3.818507194519043 ;
esubd("c_Manuf","CHN") = 3.5201902389526367 ;
esubd("c_Svces","CHN") = 1.909135103225708 ;
esubd("c_Agri","LatinAmer") = 2.3454251289367676 ;
esubd("c_Food","LatinAmer") = 2.483478307723999 ;
esubd("c_Energy","LatinAmer") = 3.818507194519043 ;
esubd("c_Manuf","LatinAmer") = 3.5201902389526367 ;
esubd("c_Svces","LatinAmer") = 1.909135103225708 ;
esubd("c_Agri","ROW") = 2.3454251289367676 ;
esubd("c_Food","ROW") = 2.483478307723999 ;
esubd("c_Energy","ROW") = 3.818507194519043 ;
esubd("c_Manuf","ROW") = 3.5201902389526367 ;
esubd("c_Svces","ROW") = 1.909135103225708 ;
parameter esubm(*,*) ;
esubm("c_Agri","USA") = 4.643953323364258 ;
esubm("c_Food","USA") = 4.823147296905518 ;
esubm("c_Energy","USA") = 11.496824264526367 ;
esubm("c_Manuf","USA") = 7.318256378173828 ;
esubm("c_Svces","USA") = 3.800415277481079 ;
esubm("c_Agri","EU_28") = 4.643953323364258 ;
esubm("c_Food","EU_28") = 4.823147296905518 ;
esubm("c_Energy","EU_28") = 11.496824264526367 ;
esubm("c_Manuf","EU_28") = 7.318256378173828 ;
esubm("c_Svces","EU_28") = 3.800415277481079 ;
esubm("c_Agri","CHN") = 4.643953323364258 ;
esubm("c_Food","CHN") = 4.823147296905518 ;
esubm("c_Energy","CHN") = 11.496824264526367 ;
esubm("c_Manuf","CHN") = 7.318256378173828 ;
esubm("c_Svces","CHN") = 3.800415277481079 ;
esubm("c_Agri","LatinAmer") = 4.643953323364258 ;
esubm("c_Food","LatinAmer") = 4.823147296905518 ;
esubm("c_Energy","LatinAmer") = 11.496824264526367 ;
esubm("c_Manuf","LatinAmer") = 7.318256378173828 ;
esubm("c_Svces","LatinAmer") = 3.800415277481079 ;
esubm("c_Agri","ROW") = 4.643953323364258 ;
esubm("c_Food","ROW") = 4.823147296905518 ;
esubm("c_Energy","ROW") = 11.496824264526367 ;
esubm("c_Manuf","ROW") = 7.318256378173828 ;
esubm("c_Svces","ROW") = 3.800415277481079 ;
parameter esubt(*,*) ;
parameter esubc(*,*) ;
parameter esubva(*,*) ;
esubva("a_Agri","USA") = 0.24963924288749695 ;
esubva("a_Food","USA") = 1.1200000047683716 ;
esubva("a_Energy","USA") = 0.7166335582733154 ;
esubva("a_Manuf","USA") = 1.1985681056976318 ;
esubva("a_Svces","USA") = 1.37246572971344 ;
esubva("a_Agri","EU_28") = 0.24963924288749695 ;
esubva("a_Food","EU_28") = 1.1200000047683716 ;
esubva("a_Energy","EU_28") = 0.7166335582733154 ;
esubva("a_Manuf","EU_28") = 1.1985681056976318 ;
esubva("a_Svces","EU_28") = 1.37246572971344 ;
esubva("a_Agri","CHN") = 0.24963924288749695 ;
esubva("a_Food","CHN") = 1.1200000047683716 ;
esubva("a_Energy","CHN") = 0.7166335582733154 ;
esubva("a_Manuf","CHN") = 1.1985681056976318 ;
esubva("a_Svces","CHN") = 1.37246572971344 ;
esubva("a_Agri","LatinAmer") = 0.24963924288749695 ;
esubva("a_Food","LatinAmer") = 1.1200000047683716 ;
esubva("a_Energy","LatinAmer") = 0.7166335582733154 ;
esubva("a_Manuf","LatinAmer") = 1.1985681056976318 ;
esubva("a_Svces","LatinAmer") = 1.37246572971344 ;
esubva("a_Agri","ROW") = 0.24963924288749695 ;
esubva("a_Food","ROW") = 1.1200000047683716 ;
esubva("a_Energy","ROW") = 0.7166335582733154 ;
esubva("a_Manuf","ROW") = 1.1985681056976318 ;
esubva("a_Svces","ROW") = 1.37246572971344 ;
parameter esubq(*,*) ;
parameter esubg(*) ;
esubg("USA") = 1.0 ;
esubg("EU_28") = 1.0 ;
esubg("CHN") = 1.0 ;
esubg("LatinAmer") = 1.0 ;
esubg("ROW") = 1.0 ;
parameter esubi(*) ;
parameter esubs(*) ;
esubs("c_Svces") = 1.0 ;
parameter etraq(*,*) ;
etraq("a_Agri","USA") = -5.0 ;
etraq("a_Food","USA") = -5.0 ;
etraq("a_Energy","USA") = -5.0 ;
etraq("a_Manuf","USA") = -5.0 ;
etraq("a_Svces","USA") = -5.0 ;
etraq("a_Agri","EU_28") = -5.0 ;
etraq("a_Food","EU_28") = -5.0 ;
etraq("a_Energy","EU_28") = -5.0 ;
etraq("a_Manuf","EU_28") = -5.0 ;
etraq("a_Svces","EU_28") = -5.0 ;
etraq("a_Agri","CHN") = -5.0 ;
etraq("a_Food","CHN") = -5.0 ;
etraq("a_Energy","CHN") = -5.0 ;
etraq("a_Manuf","CHN") = -5.0 ;
etraq("a_Svces","CHN") = -5.0 ;
etraq("a_Agri","LatinAmer") = -5.0 ;
etraq("a_Food","LatinAmer") = -5.0 ;
etraq("a_Energy","LatinAmer") = -5.0 ;
etraq("a_Manuf","LatinAmer") = -5.0 ;
etraq("a_Svces","LatinAmer") = -5.0 ;
etraq("a_Agri","ROW") = -5.0 ;
etraq("a_Food","ROW") = -5.0 ;
etraq("a_Energy","ROW") = -5.0 ;
etraq("a_Manuf","ROW") = -5.0 ;
etraq("a_Svces","ROW") = -5.0 ;
parameter etrae(*,*) ;
etrae("Land","USA") = -1.0 ;
etrae("NatRes","USA") = -0.0010000000474974513 ;
etrae("Land","EU_28") = -1.0 ;
etrae("NatRes","EU_28") = -0.0010000000474974513 ;
etrae("Land","CHN") = -1.0 ;
etrae("NatRes","CHN") = -0.0010000000474974513 ;
etrae("Land","LatinAmer") = -1.0 ;
etrae("NatRes","LatinAmer") = -0.0010000000474974513 ;
etrae("Land","ROW") = -1.0 ;
etrae("NatRes","ROW") = -0.0010000000474974513 ;
parameter incpar(*,*) ;
incpar("c_Agri","USA") = 0.15016424655914307 ;
incpar("c_Food","USA") = 0.4096221625804901 ;
incpar("c_Energy","USA") = 0.9133253693580627 ;
incpar("c_Manuf","USA") = 0.8878267407417297 ;
incpar("c_Svces","USA") = 1.0610698461532593 ;
incpar("c_Agri","EU_28") = 0.18517230451107025 ;
incpar("c_Food","EU_28") = 0.46367642283439636 ;
incpar("c_Energy","EU_28") = 0.9933156967163086 ;
incpar("c_Manuf","EU_28") = 0.9007419347763062 ;
incpar("c_Svces","EU_28") = 1.1245760917663574 ;
incpar("c_Agri","CHN") = 0.2815423607826233 ;
incpar("c_Food","CHN") = 0.3770502805709839 ;
incpar("c_Energy","CHN") = 1.0225056409835815 ;
incpar("c_Manuf","CHN") = 0.8987540006637573 ;
incpar("c_Svces","CHN") = 1.3661208152770996 ;
incpar("c_Agri","LatinAmer") = 0.31348204612731934 ;
incpar("c_Food","LatinAmer") = 0.4360114634037018 ;
incpar("c_Energy","LatinAmer") = 0.98811274766922 ;
incpar("c_Manuf","LatinAmer") = 0.9234779477119446 ;
incpar("c_Svces","LatinAmer") = 1.1729204654693604 ;
incpar("c_Agri","ROW") = 0.48021629452705383 ;
incpar("c_Food","ROW") = 0.48481285572052 ;
incpar("c_Energy","ROW") = 0.989670991897583 ;
incpar("c_Manuf","ROW") = 0.8737057447433472 ;
incpar("c_Svces","ROW") = 1.1856974363327026 ;
parameter subpar(*,*) ;
subpar("c_Agri","USA") = 0.7530407309532166 ;
subpar("c_Food","USA") = 0.1975359469652176 ;
subpar("c_Energy","USA") = 0.09387533366680145 ;
subpar("c_Manuf","USA") = 0.10022298246622086 ;
subpar("c_Svces","USA") = 0.08792939782142639 ;
subpar("c_Agri","EU_28") = 0.810137152671814 ;
subpar("c_Food","EU_28") = 0.39862877130508423 ;
subpar("c_Energy","EU_28") = 0.23175857961177826 ;
subpar("c_Manuf","EU_28") = 0.2526686191558838 ;
subpar("c_Svces","EU_28") = 0.2063865214586258 ;
subpar("c_Agri","CHN") = 0.7794262766838074 ;
subpar("c_Food","CHN") = 0.7217525839805603 ;
subpar("c_Energy","CHN") = 0.4935285449028015 ;
subpar("c_Manuf","CHN") = 0.5345491170883179 ;
subpar("c_Svces","CHN") = 0.4421200156211853 ;
subpar("c_Agri","LatinAmer") = 0.8008343577384949 ;
subpar("c_Food","LatinAmer") = 0.7026839256286621 ;
subpar("c_Energy","LatinAmer") = 0.5036433339118958 ;
subpar("c_Manuf","LatinAmer") = 0.5261639952659607 ;
subpar("c_Svces","LatinAmer") = 0.4599919021129608 ;
subpar("c_Agri","ROW") = 0.8159159421920776 ;
subpar("c_Food","ROW") = 0.650423526763916 ;
subpar("c_Energy","ROW") = 0.43773600459098816 ;
subpar("c_Manuf","ROW") = 0.4590882658958435 ;
subpar("c_Svces","ROW") = 0.33769461512565613 ;
parameter rorFlex0(*) ;
rorFlex0("USA") = 10.0 ;
rorFlex0("EU_28") = 10.0 ;
rorFlex0("CHN") = 10.0 ;
rorFlex0("LatinAmer") = 10.0 ;
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
* CONVERT-mode solve (CHECK period): write gtap.nl — Layer 6 diagnostic
* Altertax check period: CD elasticities + mobile factors, NO tm shock.
* This is the compStat betaCal→check transition: same closure as altertax
* but tariffs unchanged (base period values). Mirrors Python altertax_check.
* Use NLP mode (ifMCP=0) — GAMS CONVERT cannot write .nl for MCP.
* ============================================================
rs(r) = yes ;
ts('base') = yes ;
ts('check') = yes ;

* Apply altertax elasticity overrides (parameter_altertax.gms equivalent)
esubva(a,r)  = 1 ;
esubd(i,r)   = 0.95 ;
esubm(i,r)   = 0.95 ;
etrae(fp,r)  = 1 ;
omegaf(r,fp) = 1 ;

* Recalibrate derived elasticities from the overrides
sigmav(r,a)$(va.l(r,a,'base')) = 1 ;
sigmam(r,i,aa)$(xa.l(r,i,aa,'base')) = 0.95 ;
sigmaw(r,i)$(xmt.l(r,i,'base')) = 0.95 ;

* Solve base first, then write check .nl (no tariff shock applied)
option limrow = 0, limcol = 0, solprint = off ;
option NLP = CONVERT;
gtap.optfile = 1 ;
solve gtap using nlp maximizing walras ;
