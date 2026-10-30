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
   acts           "Activities"      / a_Rice, a_Crops, a_Livestock, a_FoodProc, a_Energy, a_Textiles, a_Chem, a_Manuf, a_ForestFish, a_Svces /
   comm           "Commodities"     / c_Rice, c_Crops, c_Livestock, c_FoodProc, c_Energy, c_Textiles, c_Chem, c_Manuf, c_ForestFish, c_Svces /
   marg(comm)     "Margin comm"     / c_Svces /
   reg            "Regions"         / USA, EU_28, CHN, JPN, IND, SSA, ROW /
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
   a_Rice, a_Crops, a_Livestock, a_FoodProc, a_Energy, a_Textiles, a_Chem, a_Manuf, a_ForestFish, a_Svces, c_Rice, c_Crops, c_Livestock, c_FoodProc,
   c_Energy, c_Textiles, c_Chem, c_Manuf, c_ForestFish, c_Svces, Land, UnSkLab, SkLab, Capital, NatRes, TRD, hhd, gov, inv, deprY, tmg, itax, ptax,
   mtax, etax, vtax, vsub, dtax, ctax, bop, tot, USA, EU_28, CHN, JPN, IND, SSA, ROW
/ ;
alias(is, js) ;

set aa(is) "Armington agents" / a_Rice, a_Crops, a_Livestock, a_FoodProc, a_Energy, a_Textiles, a_Chem, a_Manuf, a_ForestFish, a_Svces, hhd, gov, inv, tmg / ;
set a(aa)  "Activities"       / a_Rice, a_Crops, a_Livestock, a_FoodProc, a_Energy, a_Textiles, a_Chem, a_Manuf, a_ForestFish, a_Svces / ;
set i(is)  "Commodities"      / c_Rice, c_Crops, c_Livestock, c_FoodProc, c_Energy, c_Textiles, c_Chem, c_Manuf, c_ForestFish, c_Svces / ;
alias(i, j) ;

set r(is)  "Regions" / USA, EU_28, CHN, JPN, IND, SSA, ROW / ;
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
vdfb("c_Rice","a_Rice","USA") = 1903.83911132812 ;
vdfb("c_Rice","a_Rice","EU_28") = 1028.697265625 ;
vdfb("c_Rice","a_Rice","CHN") = 53764.62890625 ;
vdfb("c_Rice","a_Rice","JPN") = 17034.9375 ;
vdfb("c_Rice","a_Rice","IND") = 12666.0634765625 ;
vdfb("c_Rice","a_Rice","SSA") = 3749.08081054688 ;
vdfb("c_Rice","a_Rice","ROW") = 133730.140625 ;
vdfb("c_Rice","a_Crops","USA") = 4.327552318573 ;
vdfb("c_Rice","a_Crops","EU_28") = 178.558456420898 ;
vdfb("c_Rice","a_Crops","CHN") = 309.737426757812 ;
vdfb("c_Rice","a_Crops","JPN") = 204.524719238281 ;
vdfb("c_Rice","a_Crops","IND") = 531.416564941406 ;
vdfb("c_Rice","a_Crops","SSA") = 39.3019485473633 ;
vdfb("c_Rice","a_Crops","ROW") = 729.616760253906 ;
vdfb("c_Rice","a_Livestock","USA") = 13.9870653152466 ;
vdfb("c_Rice","a_Livestock","EU_28") = 62.8060111999512 ;
vdfb("c_Rice","a_Livestock","CHN") = 8482.7607421875 ;
vdfb("c_Rice","a_Livestock","JPN") = 519.852294921875 ;
vdfb("c_Rice","a_Livestock","IND") = 1416.74060058594 ;
vdfb("c_Rice","a_Livestock","SSA") = 169.396697998047 ;
vdfb("c_Rice","a_Livestock","ROW") = 5366.98291015625 ;
vdfb("c_Rice","a_FoodProc","USA") = 1526.53979492188 ;
vdfb("c_Rice","a_FoodProc","EU_28") = 554.214477539062 ;
vdfb("c_Rice","a_FoodProc","CHN") = 32843.01953125 ;
vdfb("c_Rice","a_FoodProc","JPN") = 3923.40844726562 ;
vdfb("c_Rice","a_FoodProc","IND") = 39301.375 ;
vdfb("c_Rice","a_FoodProc","SSA") = 1240.62158203125 ;
vdfb("c_Rice","a_FoodProc","ROW") = 25191.345703125 ;
vdfb("c_Rice","a_Energy","USA") = 0.688192427158356 ;
vdfb("c_Rice","a_Energy","EU_28") = 0.580580770969391 ;
vdfb("c_Rice","a_Energy","CHN") = 17.9700183868408 ;
vdfb("c_Rice","a_Energy","JPN") = 0.159057393670082 ;
vdfb("c_Rice","a_Energy","IND") = 35.9130439758301 ;
vdfb("c_Rice","a_Energy","SSA") = 4.02753925323486 ;
vdfb("c_Rice","a_Energy","ROW") = 1258.9111328125 ;
vdfb("c_Rice","a_Textiles","USA") = 0.419756501913071 ;
vdfb("c_Rice","a_Textiles","EU_28") = 2.21394872665405 ;
vdfb("c_Rice","a_Textiles","CHN") = 1781.41809082031 ;
vdfb("c_Rice","a_Textiles","JPN") = 0.358892321586609 ;
vdfb("c_Rice","a_Textiles","IND") = 0.634818732738495 ;
vdfb("c_Rice","a_Textiles","SSA") = 7.86968803405762 ;
vdfb("c_Rice","a_Textiles","ROW") = 306.496520996094 ;
vdfb("c_Rice","a_Chem","USA") = 115.488761901855 ;
vdfb("c_Rice","a_Chem","EU_28") = 10.1754941940308 ;
vdfb("c_Rice","a_Chem","CHN") = 7639.3603515625 ;
vdfb("c_Rice","a_Chem","JPN") = 29.373233795166 ;
vdfb("c_Rice","a_Chem","IND") = 183.650024414062 ;
vdfb("c_Rice","a_Chem","SSA") = 259.46826171875 ;
vdfb("c_Rice","a_Chem","ROW") = 793.322937011719 ;
vdfb("c_Rice","a_Manuf","USA") = 27.6491928100586 ;
vdfb("c_Rice","a_Manuf","EU_28") = 15.7712831497192 ;
vdfb("c_Rice","a_Manuf","CHN") = 4143.3466796875 ;
vdfb("c_Rice","a_Manuf","JPN") = 99.0351333618164 ;
vdfb("c_Rice","a_Manuf","IND") = 9.44229125976562 ;
vdfb("c_Rice","a_Manuf","SSA") = 102.669410705566 ;
vdfb("c_Rice","a_Manuf","ROW") = 1045.02709960938 ;
vdfb("c_Rice","a_ForestFish","USA") = 6.83522367477417 ;
vdfb("c_Rice","a_ForestFish","EU_28") = 2.82822299003601 ;
vdfb("c_Rice","a_ForestFish","CHN") = 844.63720703125 ;
vdfb("c_Rice","a_ForestFish","JPN") = 35.7318267822266 ;
vdfb("c_Rice","a_ForestFish","IND") = 1.98910856246948 ;
vdfb("c_Rice","a_ForestFish","SSA") = 1.81614983081818 ;
vdfb("c_Rice","a_ForestFish","ROW") = 1014.82464599609 ;
vdfb("c_Rice","a_Svces","USA") = 696.98046875 ;
vdfb("c_Rice","a_Svces","EU_28") = 1027.82653808594 ;
vdfb("c_Rice","a_Svces","CHN") = 14904.7607421875 ;
vdfb("c_Rice","a_Svces","JPN") = 4049.36987304688 ;
vdfb("c_Rice","a_Svces","IND") = 8577.880859375 ;
vdfb("c_Rice","a_Svces","SSA") = 1067.51110839844 ;
vdfb("c_Rice","a_Svces","ROW") = 17055.466796875 ;
vdfb("c_Crops","a_Rice","USA") = 8.3245210647583 ;
vdfb("c_Crops","a_Rice","EU_28") = 11.4856615066528 ;
vdfb("c_Crops","a_Rice","CHN") = 1216.58752441406 ;
vdfb("c_Crops","a_Rice","JPN") = 7.13126754760742 ;
vdfb("c_Crops","a_Rice","IND") = 346.501556396484 ;
vdfb("c_Crops","a_Rice","SSA") = 829.380676269531 ;
vdfb("c_Crops","a_Rice","ROW") = 1427.107421875 ;
vdfb("c_Crops","a_Crops","USA") = 7409.296875 ;
vdfb("c_Crops","a_Crops","EU_28") = 17355.6796875 ;
vdfb("c_Crops","a_Crops","CHN") = 64051.453125 ;
vdfb("c_Crops","a_Crops","JPN") = 470.238189697266 ;
vdfb("c_Crops","a_Crops","IND") = 3540.44995117188 ;
vdfb("c_Crops","a_Crops","SSA") = 30531.228515625 ;
vdfb("c_Crops","a_Crops","ROW") = 99566.7109375 ;
vdfb("c_Crops","a_Livestock","USA") = 7660.9482421875 ;
vdfb("c_Crops","a_Livestock","EU_28") = 11650.7744140625 ;
vdfb("c_Crops","a_Livestock","CHN") = 28124.537109375 ;
vdfb("c_Crops","a_Livestock","JPN") = 1035.724609375 ;
vdfb("c_Crops","a_Livestock","IND") = 8526.4501953125 ;
vdfb("c_Crops","a_Livestock","SSA") = 3009.15063476562 ;
vdfb("c_Crops","a_Livestock","ROW") = 36960.62890625 ;
vdfb("c_Crops","a_FoodProc","USA") = 75026.2421875 ;
vdfb("c_Crops","a_FoodProc","EU_28") = 47017.41015625 ;
vdfb("c_Crops","a_FoodProc","CHN") = 268547.125 ;
vdfb("c_Crops","a_FoodProc","JPN") = 5480.990234375 ;
vdfb("c_Crops","a_FoodProc","IND") = 36724.09375 ;
vdfb("c_Crops","a_FoodProc","SSA") = 22174.86328125 ;
vdfb("c_Crops","a_FoodProc","ROW") = 220311.59375 ;
vdfb("c_Crops","a_Energy","USA") = 33.7962417602539 ;
vdfb("c_Crops","a_Energy","EU_28") = 79.618034362793 ;
vdfb("c_Crops","a_Energy","CHN") = 110.819976806641 ;
vdfb("c_Crops","a_Energy","JPN") = 5.27432632446289 ;
vdfb("c_Crops","a_Energy","IND") = 1515.28552246094 ;
vdfb("c_Crops","a_Energy","SSA") = 140.139923095703 ;
vdfb("c_Crops","a_Energy","ROW") = 1847.40087890625 ;
vdfb("c_Crops","a_Textiles","USA") = 542.929992675781 ;
vdfb("c_Crops","a_Textiles","EU_28") = 741.082214355469 ;
vdfb("c_Crops","a_Textiles","CHN") = 48385.60546875 ;
vdfb("c_Crops","a_Textiles","JPN") = 154.43864440918 ;
vdfb("c_Crops","a_Textiles","IND") = 22126.384765625 ;
vdfb("c_Crops","a_Textiles","SSA") = 1298.88452148438 ;
vdfb("c_Crops","a_Textiles","ROW") = 18938.70703125 ;
vdfb("c_Crops","a_Chem","USA") = 13713.0234375 ;
vdfb("c_Crops","a_Chem","EU_28") = 1118.50256347656 ;
vdfb("c_Crops","a_Chem","CHN") = 32907.796875 ;
vdfb("c_Crops","a_Chem","JPN") = 1345.27355957031 ;
vdfb("c_Crops","a_Chem","IND") = 3471.76684570312 ;
vdfb("c_Crops","a_Chem","SSA") = 491.473724365234 ;
vdfb("c_Crops","a_Chem","ROW") = 13921.3447265625 ;
vdfb("c_Crops","a_Manuf","USA") = 381.783477783203 ;
vdfb("c_Crops","a_Manuf","EU_28") = 227.558624267578 ;
vdfb("c_Crops","a_Manuf","CHN") = 17724.466796875 ;
vdfb("c_Crops","a_Manuf","JPN") = 51.2961959838867 ;
vdfb("c_Crops","a_Manuf","IND") = 103.207260131836 ;
vdfb("c_Crops","a_Manuf","SSA") = 386.513397216797 ;
vdfb("c_Crops","a_Manuf","ROW") = 1532.29943847656 ;
vdfb("c_Crops","a_ForestFish","USA") = 271.877471923828 ;
vdfb("c_Crops","a_ForestFish","EU_28") = 150.290710449219 ;
vdfb("c_Crops","a_ForestFish","CHN") = 1234.75708007812 ;
vdfb("c_Crops","a_ForestFish","JPN") = 1.84959816932678 ;
vdfb("c_Crops","a_ForestFish","IND") = 20.5230121612549 ;
vdfb("c_Crops","a_ForestFish","SSA") = 170.042877197266 ;
vdfb("c_Crops","a_ForestFish","ROW") = 1247.68713378906 ;
vdfb("c_Crops","a_Svces","USA") = 6445.19970703125 ;
vdfb("c_Crops","a_Svces","EU_28") = 12885.908203125 ;
vdfb("c_Crops","a_Svces","CHN") = 53384.203125 ;
vdfb("c_Crops","a_Svces","JPN") = 5101.6806640625 ;
vdfb("c_Crops","a_Svces","IND") = 22357.201171875 ;
vdfb("c_Crops","a_Svces","SSA") = 4983.9912109375 ;
vdfb("c_Crops","a_Svces","ROW") = 41458.46484375 ;
vdfb("c_Livestock","a_Rice","USA") = 136.338241577148 ;
vdfb("c_Livestock","a_Rice","EU_28") = 20.5606689453125 ;
vdfb("c_Livestock","a_Rice","CHN") = 31.7080936431885 ;
vdfb("c_Livestock","a_Rice","JPN") = 122.855903625488 ;
vdfb("c_Livestock","a_Rice","IND") = 1090.66870117188 ;
vdfb("c_Livestock","a_Rice","SSA") = 228.432586669922 ;
vdfb("c_Livestock","a_Rice","ROW") = 1085.83605957031 ;
vdfb("c_Livestock","a_Crops","USA") = 275.160034179688 ;
vdfb("c_Livestock","a_Crops","EU_28") = 1626.447265625 ;
vdfb("c_Livestock","a_Crops","CHN") = 276.102294921875 ;
vdfb("c_Livestock","a_Crops","JPN") = 203.468841552734 ;
vdfb("c_Livestock","a_Crops","IND") = 4186.1015625 ;
vdfb("c_Livestock","a_Crops","SSA") = 1107.54821777344 ;
vdfb("c_Livestock","a_Crops","ROW") = 13521.982421875 ;
vdfb("c_Livestock","a_Livestock","USA") = 18427.7734375 ;
vdfb("c_Livestock","a_Livestock","EU_28") = 5519.39697265625 ;
vdfb("c_Livestock","a_Livestock","CHN") = 25001.029296875 ;
vdfb("c_Livestock","a_Livestock","JPN") = 2029.14282226562 ;
vdfb("c_Livestock","a_Livestock","IND") = 951.65478515625 ;
vdfb("c_Livestock","a_Livestock","SSA") = 9425.314453125 ;
vdfb("c_Livestock","a_Livestock","ROW") = 80862.703125 ;
vdfb("c_Livestock","a_FoodProc","USA") = 130922.890625 ;
vdfb("c_Livestock","a_FoodProc","EU_28") = 126761.1875 ;
vdfb("c_Livestock","a_FoodProc","CHN") = 144252.375 ;
vdfb("c_Livestock","a_FoodProc","JPN") = 20948.66796875 ;
vdfb("c_Livestock","a_FoodProc","IND") = 17225.990234375 ;
vdfb("c_Livestock","a_FoodProc","SSA") = 18031.08203125 ;
vdfb("c_Livestock","a_FoodProc","ROW") = 269292.25 ;
vdfb("c_Livestock","a_Energy","USA") = 0.753273785114288 ;
vdfb("c_Livestock","a_Energy","EU_28") = 65.9546966552734 ;
vdfb("c_Livestock","a_Energy","CHN") = 1.61126887798309 ;
vdfb("c_Livestock","a_Energy","JPN") = 1.04377830028534 ;
vdfb("c_Livestock","a_Energy","IND") = 138.80078125 ;
vdfb("c_Livestock","a_Energy","SSA") = 11.1336212158203 ;
vdfb("c_Livestock","a_Energy","ROW") = 214.349472045898 ;
vdfb("c_Livestock","a_Textiles","USA") = 166.914505004883 ;
vdfb("c_Livestock","a_Textiles","EU_28") = 434.225067138672 ;
vdfb("c_Livestock","a_Textiles","CHN") = 22377.80859375 ;
vdfb("c_Livestock","a_Textiles","JPN") = 16.1092891693115 ;
vdfb("c_Livestock","a_Textiles","IND") = 1262.13586425781 ;
vdfb("c_Livestock","a_Textiles","SSA") = 922.907836914062 ;
vdfb("c_Livestock","a_Textiles","ROW") = 5029.39111328125 ;
vdfb("c_Livestock","a_Chem","USA") = 177.595993041992 ;
vdfb("c_Livestock","a_Chem","EU_28") = 1117.703125 ;
vdfb("c_Livestock","a_Chem","CHN") = 20778.58203125 ;
vdfb("c_Livestock","a_Chem","JPN") = 83.4835433959961 ;
vdfb("c_Livestock","a_Chem","IND") = 4190.6142578125 ;
vdfb("c_Livestock","a_Chem","SSA") = 86.851432800293 ;
vdfb("c_Livestock","a_Chem","ROW") = 1755.26220703125 ;
vdfb("c_Livestock","a_Manuf","USA") = 16.9556198120117 ;
vdfb("c_Livestock","a_Manuf","EU_28") = 150.141250610352 ;
vdfb("c_Livestock","a_Manuf","CHN") = 2863.556640625 ;
vdfb("c_Livestock","a_Manuf","JPN") = 6.01187801361084 ;
vdfb("c_Livestock","a_Manuf","IND") = 65.0917053222656 ;
vdfb("c_Livestock","a_Manuf","SSA") = 58.3407592773438 ;
vdfb("c_Livestock","a_Manuf","ROW") = 891.143310546875 ;
vdfb("c_Livestock","a_ForestFish","USA") = 390.113220214844 ;
vdfb("c_Livestock","a_ForestFish","EU_28") = 116.503135681152 ;
vdfb("c_Livestock","a_ForestFish","CHN") = 19.9665851593018 ;
vdfb("c_Livestock","a_ForestFish","JPN") = 3.30553030967712 ;
vdfb("c_Livestock","a_ForestFish","IND") = 540.022644042969 ;
vdfb("c_Livestock","a_ForestFish","SSA") = 953.856994628906 ;
vdfb("c_Livestock","a_ForestFish","ROW") = 548.304504394531 ;
vdfb("c_Livestock","a_Svces","USA") = 3673.93920898438 ;
vdfb("c_Livestock","a_Svces","EU_28") = 8497.37109375 ;
vdfb("c_Livestock","a_Svces","CHN") = 10362.19921875 ;
vdfb("c_Livestock","a_Svces","JPN") = 2726.7568359375 ;
vdfb("c_Livestock","a_Svces","IND") = 38347.48828125 ;
vdfb("c_Livestock","a_Svces","SSA") = 4536.791015625 ;
vdfb("c_Livestock","a_Svces","ROW") = 22470.849609375 ;
vdfb("c_FoodProc","a_Rice","USA") = 63.3832473754883 ;
vdfb("c_FoodProc","a_Rice","EU_28") = 41.9515266418457 ;
vdfb("c_FoodProc","a_Rice","CHN") = 252.466293334961 ;
vdfb("c_FoodProc","a_Rice","JPN") = 2.00065112113953 ;
vdfb("c_FoodProc","a_Rice","IND") = 0.553491592407227 ;
vdfb("c_FoodProc","a_Rice","SSA") = 177.341262817383 ;
vdfb("c_FoodProc","a_Rice","ROW") = 1354.67395019531 ;
vdfb("c_FoodProc","a_Crops","USA") = 1.08856236934662 ;
vdfb("c_FoodProc","a_Crops","EU_28") = 1828.32641601562 ;
vdfb("c_FoodProc","a_Crops","CHN") = 1379.84118652344 ;
vdfb("c_FoodProc","a_Crops","JPN") = 63.2475967407227 ;
vdfb("c_FoodProc","a_Crops","IND") = 1.05086076259613 ;
vdfb("c_FoodProc","a_Crops","SSA") = 1062.90173339844 ;
vdfb("c_FoodProc","a_Crops","ROW") = 4347.06884765625 ;
vdfb("c_FoodProc","a_Livestock","USA") = 34643.47265625 ;
vdfb("c_FoodProc","a_Livestock","EU_28") = 38756.54296875 ;
vdfb("c_FoodProc","a_Livestock","CHN") = 79583.59375 ;
vdfb("c_FoodProc","a_Livestock","JPN") = 10860.6875 ;
vdfb("c_FoodProc","a_Livestock","IND") = 2576.85424804688 ;
vdfb("c_FoodProc","a_Livestock","SSA") = 4847.6787109375 ;
vdfb("c_FoodProc","a_Livestock","ROW") = 73739.171875 ;
vdfb("c_FoodProc","a_FoodProc","USA") = 230133.625 ;
vdfb("c_FoodProc","a_FoodProc","EU_28") = 232287.8125 ;
vdfb("c_FoodProc","a_FoodProc","CHN") = 355625.78125 ;
vdfb("c_FoodProc","a_FoodProc","JPN") = 47243.8046875 ;
vdfb("c_FoodProc","a_FoodProc","IND") = 9867.7822265625 ;
vdfb("c_FoodProc","a_FoodProc","SSA") = 35844.56640625 ;
vdfb("c_FoodProc","a_FoodProc","ROW") = 277792.25 ;
vdfb("c_FoodProc","a_Energy","USA") = 127.457633972168 ;
vdfb("c_FoodProc","a_Energy","EU_28") = 561.692016601562 ;
vdfb("c_FoodProc","a_Energy","CHN") = 5781.65625 ;
vdfb("c_FoodProc","a_Energy","JPN") = 3.66133546829224 ;
vdfb("c_FoodProc","a_Energy","IND") = 79.3506927490234 ;
vdfb("c_FoodProc","a_Energy","SSA") = 56.1043548583984 ;
vdfb("c_FoodProc","a_Energy","ROW") = 1553.33752441406 ;
vdfb("c_FoodProc","a_Textiles","USA") = 1367.70373535156 ;
vdfb("c_FoodProc","a_Textiles","EU_28") = 2864.07153320312 ;
vdfb("c_FoodProc","a_Textiles","CHN") = 28619.1640625 ;
vdfb("c_FoodProc","a_Textiles","JPN") = 134.104904174805 ;
vdfb("c_FoodProc","a_Textiles","IND") = 10.4697999954224 ;
vdfb("c_FoodProc","a_Textiles","SSA") = 877.979370117188 ;
vdfb("c_FoodProc","a_Textiles","ROW") = 6750.10693359375 ;
vdfb("c_FoodProc","a_Chem","USA") = 5254.16162109375 ;
vdfb("c_FoodProc","a_Chem","EU_28") = 13340.4404296875 ;
vdfb("c_FoodProc","a_Chem","CHN") = 43174.29296875 ;
vdfb("c_FoodProc","a_Chem","JPN") = 1428.33325195312 ;
vdfb("c_FoodProc","a_Chem","IND") = 2572.86010742188 ;
vdfb("c_FoodProc","a_Chem","SSA") = 1242.13415527344 ;
vdfb("c_FoodProc","a_Chem","ROW") = 10969.8427734375 ;
vdfb("c_FoodProc","a_Manuf","USA") = 3033.822265625 ;
vdfb("c_FoodProc","a_Manuf","EU_28") = 4059.60668945312 ;
vdfb("c_FoodProc","a_Manuf","CHN") = 37275.8125 ;
vdfb("c_FoodProc","a_Manuf","JPN") = 297.406005859375 ;
vdfb("c_FoodProc","a_Manuf","IND") = 26.8659152984619 ;
vdfb("c_FoodProc","a_Manuf","SSA") = 900.84521484375 ;
vdfb("c_FoodProc","a_Manuf","ROW") = 5481.6064453125 ;
vdfb("c_FoodProc","a_ForestFish","USA") = 799.216918945312 ;
vdfb("c_FoodProc","a_ForestFish","EU_28") = 741.104370117188 ;
vdfb("c_FoodProc","a_ForestFish","CHN") = 25245.833984375 ;
vdfb("c_FoodProc","a_ForestFish","JPN") = 1060.00329589844 ;
vdfb("c_FoodProc","a_ForestFish","IND") = 31.1213283538818 ;
vdfb("c_FoodProc","a_ForestFish","SSA") = 206.140747070312 ;
vdfb("c_FoodProc","a_ForestFish","ROW") = 15869.0810546875 ;
vdfb("c_FoodProc","a_Svces","USA") = 161293.375 ;
vdfb("c_FoodProc","a_Svces","EU_28") = 211565.265625 ;
vdfb("c_FoodProc","a_Svces","CHN") = 249720.8125 ;
vdfb("c_FoodProc","a_Svces","JPN") = 51805.91796875 ;
vdfb("c_FoodProc","a_Svces","IND") = 8796.5712890625 ;
vdfb("c_FoodProc","a_Svces","SSA") = 23107.150390625 ;
vdfb("c_FoodProc","a_Svces","ROW") = 297968.6875 ;
vdfb("c_Energy","a_Rice","USA") = 119.464775085449 ;
vdfb("c_Energy","a_Rice","EU_28") = 144.91438293457 ;
vdfb("c_Energy","a_Rice","CHN") = 2363.26025390625 ;
vdfb("c_Energy","a_Rice","JPN") = 326.808135986328 ;
vdfb("c_Energy","a_Rice","IND") = 22698.5234375 ;
vdfb("c_Energy","a_Rice","SSA") = 73.0360488891602 ;
vdfb("c_Energy","a_Rice","ROW") = 4651.951171875 ;
vdfb("c_Energy","a_Crops","USA") = 4812.6015625 ;
vdfb("c_Energy","a_Crops","EU_28") = 7341.2734375 ;
vdfb("c_Energy","a_Crops","CHN") = 11427.8681640625 ;
vdfb("c_Energy","a_Crops","JPN") = 846.089782714844 ;
vdfb("c_Energy","a_Crops","IND") = 16776.359375 ;
vdfb("c_Energy","a_Crops","SSA") = 690.155883789062 ;
vdfb("c_Energy","a_Crops","ROW") = 25312.427734375 ;
vdfb("c_Energy","a_Livestock","USA") = 3620.705078125 ;
vdfb("c_Energy","a_Livestock","EU_28") = 2993.30883789062 ;
vdfb("c_Energy","a_Livestock","CHN") = 1842.13793945312 ;
vdfb("c_Energy","a_Livestock","JPN") = 364.264312744141 ;
vdfb("c_Energy","a_Livestock","IND") = 50.6079406738281 ;
vdfb("c_Energy","a_Livestock","SSA") = 939.701599121094 ;
vdfb("c_Energy","a_Livestock","ROW") = 12988.8759765625 ;
vdfb("c_Energy","a_FoodProc","USA") = 12402.1259765625 ;
vdfb("c_Energy","a_FoodProc","EU_28") = 18078.8046875 ;
vdfb("c_Energy","a_FoodProc","CHN") = 15816.912109375 ;
vdfb("c_Energy","a_FoodProc","JPN") = 5346.21240234375 ;
vdfb("c_Energy","a_FoodProc","IND") = 4149.07958984375 ;
vdfb("c_Energy","a_FoodProc","SSA") = 1118.12329101562 ;
vdfb("c_Energy","a_FoodProc","ROW") = 34954.76171875 ;
vdfb("c_Energy","a_Energy","USA") = 304586.375 ;
vdfb("c_Energy","a_Energy","EU_28") = 133683.5 ;
vdfb("c_Energy","a_Energy","CHN") = 381460.25 ;
vdfb("c_Energy","a_Energy","JPN") = 41107.19921875 ;
vdfb("c_Energy","a_Energy","IND") = 101309.921875 ;
vdfb("c_Energy","a_Energy","SSA") = 27679.234375 ;
vdfb("c_Energy","a_Energy","ROW") = 811974.9375 ;
vdfb("c_Energy","a_Textiles","USA") = 2135.58325195312 ;
vdfb("c_Energy","a_Textiles","EU_28") = 3701.64038085938 ;
vdfb("c_Energy","a_Textiles","CHN") = 24261.560546875 ;
vdfb("c_Energy","a_Textiles","JPN") = 1057.90368652344 ;
vdfb("c_Energy","a_Textiles","IND") = 2766.59985351562 ;
vdfb("c_Energy","a_Textiles","SSA") = 417.019775390625 ;
vdfb("c_Energy","a_Textiles","ROW") = 15446.34375 ;
vdfb("c_Energy","a_Chem","USA") = 59964.21484375 ;
vdfb("c_Energy","a_Chem","EU_28") = 57616.578125 ;
vdfb("c_Energy","a_Chem","CHN") = 132094.828125 ;
vdfb("c_Energy","a_Chem","JPN") = 33480.88671875 ;
vdfb("c_Energy","a_Chem","IND") = 30890.443359375 ;
vdfb("c_Energy","a_Chem","SSA") = 4033.39233398438 ;
vdfb("c_Energy","a_Chem","ROW") = 199770.515625 ;
vdfb("c_Energy","a_Manuf","USA") = 60836.5390625 ;
vdfb("c_Energy","a_Manuf","EU_28") = 92154.015625 ;
vdfb("c_Energy","a_Manuf","CHN") = 296860.5625 ;
vdfb("c_Energy","a_Manuf","JPN") = 53051.19921875 ;
vdfb("c_Energy","a_Manuf","IND") = 88381.046875 ;
vdfb("c_Energy","a_Manuf","SSA") = 21068.814453125 ;
vdfb("c_Energy","a_Manuf","ROW") = 305322.6875 ;
vdfb("c_Energy","a_ForestFish","USA") = 924.746520996094 ;
vdfb("c_Energy","a_ForestFish","EU_28") = 2402.23901367188 ;
vdfb("c_Energy","a_ForestFish","CHN") = 3547.619140625 ;
vdfb("c_Energy","a_ForestFish","JPN") = 1367.04724121094 ;
vdfb("c_Energy","a_ForestFish","IND") = 2568.16064453125 ;
vdfb("c_Energy","a_ForestFish","SSA") = 276.355224609375 ;
vdfb("c_Energy","a_ForestFish","ROW") = 9053.212890625 ;
vdfb("c_Energy","a_Svces","USA") = 330760.90625 ;
vdfb("c_Energy","a_Svces","EU_28") = 232326.625 ;
vdfb("c_Energy","a_Svces","CHN") = 211807.609375 ;
vdfb("c_Energy","a_Svces","JPN") = 101658.3359375 ;
vdfb("c_Energy","a_Svces","IND") = 101846.8671875 ;
vdfb("c_Energy","a_Svces","SSA") = 23511.986328125 ;
vdfb("c_Energy","a_Svces","ROW") = 528292 ;
vdfb("c_Textiles","a_Rice","USA") = 26.7127780914307 ;
vdfb("c_Textiles","a_Rice","EU_28") = 16.4335079193115 ;
vdfb("c_Textiles","a_Rice","CHN") = 365.348388671875 ;
vdfb("c_Textiles","a_Rice","JPN") = 32.8032188415527 ;
vdfb("c_Textiles","a_Rice","IND") = 3.92034578323364 ;
vdfb("c_Textiles","a_Rice","SSA") = 16.8999519348145 ;
vdfb("c_Textiles","a_Rice","ROW") = 392.274230957031 ;
vdfb("c_Textiles","a_Crops","USA") = 115.958969116211 ;
vdfb("c_Textiles","a_Crops","EU_28") = 321.631134033203 ;
vdfb("c_Textiles","a_Crops","CHN") = 28.8871669769287 ;
vdfb("c_Textiles","a_Crops","JPN") = 47.2777366638184 ;
vdfb("c_Textiles","a_Crops","IND") = 10.9938297271729 ;
vdfb("c_Textiles","a_Crops","SSA") = 445.065887451172 ;
vdfb("c_Textiles","a_Crops","ROW") = 1019.40991210938 ;
vdfb("c_Textiles","a_Livestock","USA") = 13.6999063491821 ;
vdfb("c_Textiles","a_Livestock","EU_28") = 157.368789672852 ;
vdfb("c_Textiles","a_Livestock","CHN") = 5.78551626205444 ;
vdfb("c_Textiles","a_Livestock","JPN") = 10.0672359466553 ;
vdfb("c_Textiles","a_Livestock","IND") = 3.60866618156433 ;
vdfb("c_Textiles","a_Livestock","SSA") = 121.44799041748 ;
vdfb("c_Textiles","a_Livestock","ROW") = 628.275634765625 ;
vdfb("c_Textiles","a_FoodProc","USA") = 139.392715454102 ;
vdfb("c_Textiles","a_FoodProc","EU_28") = 1670.0439453125 ;
vdfb("c_Textiles","a_FoodProc","CHN") = 6941.52685546875 ;
vdfb("c_Textiles","a_FoodProc","JPN") = 143.51237487793 ;
vdfb("c_Textiles","a_FoodProc","IND") = 41.1409225463867 ;
vdfb("c_Textiles","a_FoodProc","SSA") = 212.993591308594 ;
vdfb("c_Textiles","a_FoodProc","ROW") = 3357.3017578125 ;
vdfb("c_Textiles","a_Energy","USA") = 30.0367851257324 ;
vdfb("c_Textiles","a_Energy","EU_28") = 199.654418945312 ;
vdfb("c_Textiles","a_Energy","CHN") = 2953.3994140625 ;
vdfb("c_Textiles","a_Energy","JPN") = 21.6406230926514 ;
vdfb("c_Textiles","a_Energy","IND") = 969.205200195312 ;
vdfb("c_Textiles","a_Energy","SSA") = 86.9019927978516 ;
vdfb("c_Textiles","a_Energy","ROW") = 3015.26049804688 ;
vdfb("c_Textiles","a_Textiles","USA") = 11332.21875 ;
vdfb("c_Textiles","a_Textiles","EU_28") = 58988.94140625 ;
vdfb("c_Textiles","a_Textiles","CHN") = 575629.3125 ;
vdfb("c_Textiles","a_Textiles","JPN") = 6408.46923828125 ;
vdfb("c_Textiles","a_Textiles","IND") = 23795.712890625 ;
vdfb("c_Textiles","a_Textiles","SSA") = 4585.4658203125 ;
vdfb("c_Textiles","a_Textiles","ROW") = 154254.265625 ;
vdfb("c_Textiles","a_Chem","USA") = 2063.11596679688 ;
vdfb("c_Textiles","a_Chem","EU_28") = 5339.50927734375 ;
vdfb("c_Textiles","a_Chem","CHN") = 21557.240234375 ;
vdfb("c_Textiles","a_Chem","JPN") = 578.492492675781 ;
vdfb("c_Textiles","a_Chem","IND") = 2908.421875 ;
vdfb("c_Textiles","a_Chem","SSA") = 151.499404907227 ;
vdfb("c_Textiles","a_Chem","ROW") = 5587.96923828125 ;
vdfb("c_Textiles","a_Manuf","USA") = 12354.0859375 ;
vdfb("c_Textiles","a_Manuf","EU_28") = 16155.080078125 ;
vdfb("c_Textiles","a_Manuf","CHN") = 67783.453125 ;
vdfb("c_Textiles","a_Manuf","JPN") = 2394.0947265625 ;
vdfb("c_Textiles","a_Manuf","IND") = 901.5078125 ;
vdfb("c_Textiles","a_Manuf","SSA") = 710.524108886719 ;
vdfb("c_Textiles","a_Manuf","ROW") = 20941.353515625 ;
vdfb("c_Textiles","a_ForestFish","USA") = 62.1393394470215 ;
vdfb("c_Textiles","a_ForestFish","EU_28") = 311.782958984375 ;
vdfb("c_Textiles","a_ForestFish","CHN") = 201.490783691406 ;
vdfb("c_Textiles","a_ForestFish","JPN") = 152.957962036133 ;
vdfb("c_Textiles","a_ForestFish","IND") = 103.982383728027 ;
vdfb("c_Textiles","a_ForestFish","SSA") = 211.096969604492 ;
vdfb("c_Textiles","a_ForestFish","ROW") = 953.718322753906 ;
vdfb("c_Textiles","a_Svces","USA") = 11241.6806640625 ;
vdfb("c_Textiles","a_Svces","EU_28") = 21642.767578125 ;
vdfb("c_Textiles","a_Svces","CHN") = 93901.4140625 ;
vdfb("c_Textiles","a_Svces","JPN") = 6631.8359375 ;
vdfb("c_Textiles","a_Svces","IND") = 5068.60302734375 ;
vdfb("c_Textiles","a_Svces","SSA") = 2854.73974609375 ;
vdfb("c_Textiles","a_Svces","ROW") = 50776.6953125 ;
vdfb("c_Chem","a_Rice","USA") = 369.028625488281 ;
vdfb("c_Chem","a_Rice","EU_28") = 171.126968383789 ;
vdfb("c_Chem","a_Rice","CHN") = 11441.203125 ;
vdfb("c_Chem","a_Rice","JPN") = 1436.67822265625 ;
vdfb("c_Chem","a_Rice","IND") = 545.098693847656 ;
vdfb("c_Chem","a_Rice","SSA") = 87.2298889160156 ;
vdfb("c_Chem","a_Rice","ROW") = 5824.39208984375 ;
vdfb("c_Chem","a_Crops","USA") = 18991.599609375 ;
vdfb("c_Chem","a_Crops","EU_28") = 8058.02001953125 ;
vdfb("c_Chem","a_Crops","CHN") = 73639.890625 ;
vdfb("c_Chem","a_Crops","JPN") = 3258.66674804688 ;
vdfb("c_Chem","a_Crops","IND") = 630.195129394531 ;
vdfb("c_Chem","a_Crops","SSA") = 3270.40600585938 ;
vdfb("c_Chem","a_Crops","ROW") = 45151.42578125 ;
vdfb("c_Chem","a_Livestock","USA") = 1836.10314941406 ;
vdfb("c_Chem","a_Livestock","EU_28") = 1060.08801269531 ;
vdfb("c_Chem","a_Livestock","CHN") = 3499.54272460938 ;
vdfb("c_Chem","a_Livestock","JPN") = 483.24853515625 ;
vdfb("c_Chem","a_Livestock","IND") = 22.3385620117188 ;
vdfb("c_Chem","a_Livestock","SSA") = 253.021026611328 ;
vdfb("c_Chem","a_Livestock","ROW") = 8361.7587890625 ;
vdfb("c_Chem","a_FoodProc","USA") = 32114.3671875 ;
vdfb("c_Chem","a_FoodProc","EU_28") = 19245.232421875 ;
vdfb("c_Chem","a_FoodProc","CHN") = 28550.94921875 ;
vdfb("c_Chem","a_FoodProc","JPN") = 8466.7705078125 ;
vdfb("c_Chem","a_FoodProc","IND") = 3502.63891601562 ;
vdfb("c_Chem","a_FoodProc","SSA") = 1070.12963867188 ;
vdfb("c_Chem","a_FoodProc","ROW") = 35377.09375 ;
vdfb("c_Chem","a_Energy","USA") = 8546.69140625 ;
vdfb("c_Chem","a_Energy","EU_28") = 6520.45947265625 ;
vdfb("c_Chem","a_Energy","CHN") = 13171.3505859375 ;
vdfb("c_Chem","a_Energy","JPN") = 468.08349609375 ;
vdfb("c_Chem","a_Energy","IND") = 9159.3876953125 ;
vdfb("c_Chem","a_Energy","SSA") = 493.853424072266 ;
vdfb("c_Chem","a_Energy","ROW") = 19515.541015625 ;
vdfb("c_Chem","a_Textiles","USA") = 11336.2529296875 ;
vdfb("c_Chem","a_Textiles","EU_28") = 14537.3193359375 ;
vdfb("c_Chem","a_Textiles","CHN") = 69729.15625 ;
vdfb("c_Chem","a_Textiles","JPN") = 4130.2119140625 ;
vdfb("c_Chem","a_Textiles","IND") = 17455.283203125 ;
vdfb("c_Chem","a_Textiles","SSA") = 663.855529785156 ;
vdfb("c_Chem","a_Textiles","ROW") = 29824.11328125 ;
vdfb("c_Chem","a_Chem","USA") = 230851.640625 ;
vdfb("c_Chem","a_Chem","EU_28") = 162658.4375 ;
vdfb("c_Chem","a_Chem","CHN") = 659854.875 ;
vdfb("c_Chem","a_Chem","JPN") = 139954.421875 ;
vdfb("c_Chem","a_Chem","IND") = 79511.578125 ;
vdfb("c_Chem","a_Chem","SSA") = 5958.6708984375 ;
vdfb("c_Chem","a_Chem","ROW") = 325458.0625 ;
vdfb("c_Chem","a_Manuf","USA") = 121307.046875 ;
vdfb("c_Chem","a_Manuf","EU_28") = 131324.40625 ;
vdfb("c_Chem","a_Manuf","CHN") = 357087.625 ;
vdfb("c_Chem","a_Manuf","JPN") = 70090.6953125 ;
vdfb("c_Chem","a_Manuf","IND") = 41162.9765625 ;
vdfb("c_Chem","a_Manuf","SSA") = 5481.61328125 ;
vdfb("c_Chem","a_Manuf","ROW") = 191983.484375 ;
vdfb("c_Chem","a_ForestFish","USA") = 2338.49096679688 ;
vdfb("c_Chem","a_ForestFish","EU_28") = 672.178405761719 ;
vdfb("c_Chem","a_ForestFish","CHN") = 3506.4013671875 ;
vdfb("c_Chem","a_ForestFish","JPN") = 450.74169921875 ;
vdfb("c_Chem","a_ForestFish","IND") = 128.234085083008 ;
vdfb("c_Chem","a_ForestFish","SSA") = 307.39453125 ;
vdfb("c_Chem","a_ForestFish","ROW") = 3559.8837890625 ;
vdfb("c_Chem","a_Svces","USA") = 214567.140625 ;
vdfb("c_Chem","a_Svces","EU_28") = 134750.90625 ;
vdfb("c_Chem","a_Svces","CHN") = 412766.1875 ;
vdfb("c_Chem","a_Svces","JPN") = 97407.09375 ;
vdfb("c_Chem","a_Svces","IND") = 52378.34765625 ;
vdfb("c_Chem","a_Svces","SSA") = 7831.037109375 ;
vdfb("c_Chem","a_Svces","ROW") = 249660.390625 ;
vdfb("c_Manuf","a_Rice","USA") = 368.326721191406 ;
vdfb("c_Manuf","a_Rice","EU_28") = 234.322601318359 ;
vdfb("c_Manuf","a_Rice","CHN") = 1612.20446777344 ;
vdfb("c_Manuf","a_Rice","JPN") = 214.465118408203 ;
vdfb("c_Manuf","a_Rice","IND") = 220.630874633789 ;
vdfb("c_Manuf","a_Rice","SSA") = 138.734970092773 ;
vdfb("c_Manuf","a_Rice","ROW") = 3360.43188476562 ;
vdfb("c_Manuf","a_Crops","USA") = 4227.1279296875 ;
vdfb("c_Manuf","a_Crops","EU_28") = 8836.630859375 ;
vdfb("c_Manuf","a_Crops","CHN") = 7665.103515625 ;
vdfb("c_Manuf","a_Crops","JPN") = 1718.3935546875 ;
vdfb("c_Manuf","a_Crops","IND") = 651.305419921875 ;
vdfb("c_Manuf","a_Crops","SSA") = 1841.93054199219 ;
vdfb("c_Manuf","a_Crops","ROW") = 12921.5224609375 ;
vdfb("c_Manuf","a_Livestock","USA") = 2368.66186523438 ;
vdfb("c_Manuf","a_Livestock","EU_28") = 2596.46215820312 ;
vdfb("c_Manuf","a_Livestock","CHN") = 1868.95983886719 ;
vdfb("c_Manuf","a_Livestock","JPN") = 253.254669189453 ;
vdfb("c_Manuf","a_Livestock","IND") = 531.789672851562 ;
vdfb("c_Manuf","a_Livestock","SSA") = 334.414215087891 ;
vdfb("c_Manuf","a_Livestock","ROW") = 5916.14453125 ;
vdfb("c_Manuf","a_FoodProc","USA") = 67874.75 ;
vdfb("c_Manuf","a_FoodProc","EU_28") = 52366.59765625 ;
vdfb("c_Manuf","a_FoodProc","CHN") = 44743.796875 ;
vdfb("c_Manuf","a_FoodProc","JPN") = 12059.3046875 ;
vdfb("c_Manuf","a_FoodProc","IND") = 816.94970703125 ;
vdfb("c_Manuf","a_FoodProc","SSA") = 3261.4599609375 ;
vdfb("c_Manuf","a_FoodProc","ROW") = 61989.78515625 ;
vdfb("c_Manuf","a_Energy","USA") = 30330.783203125 ;
vdfb("c_Manuf","a_Energy","EU_28") = 30757.35546875 ;
vdfb("c_Manuf","a_Energy","CHN") = 92465.1796875 ;
vdfb("c_Manuf","a_Energy","JPN") = 2099.00952148438 ;
vdfb("c_Manuf","a_Energy","IND") = 27642.455078125 ;
vdfb("c_Manuf","a_Energy","SSA") = 2962.34057617188 ;
vdfb("c_Manuf","a_Energy","ROW") = 139121.296875 ;
vdfb("c_Manuf","a_Textiles","USA") = 3419.96484375 ;
vdfb("c_Manuf","a_Textiles","EU_28") = 12561.5966796875 ;
vdfb("c_Manuf","a_Textiles","CHN") = 24331.81640625 ;
vdfb("c_Manuf","a_Textiles","JPN") = 742.497314453125 ;
vdfb("c_Manuf","a_Textiles","IND") = 2895.27172851562 ;
vdfb("c_Manuf","a_Textiles","SSA") = 1320.80358886719 ;
vdfb("c_Manuf","a_Textiles","ROW") = 21280.96484375 ;
vdfb("c_Manuf","a_Chem","USA") = 62408.12109375 ;
vdfb("c_Manuf","a_Chem","EU_28") = 57681.9296875 ;
vdfb("c_Manuf","a_Chem","CHN") = 95930.375 ;
vdfb("c_Manuf","a_Chem","JPN") = 13395.4794921875 ;
vdfb("c_Manuf","a_Chem","IND") = 8394.7958984375 ;
vdfb("c_Manuf","a_Chem","SSA") = 2621.947265625 ;
vdfb("c_Manuf","a_Chem","ROW") = 71583.4765625 ;
vdfb("c_Manuf","a_Manuf","USA") = 878579.3125 ;
vdfb("c_Manuf","a_Manuf","EU_28") = 1261312.25 ;
vdfb("c_Manuf","a_Manuf","CHN") = 3800643.25 ;
vdfb("c_Manuf","a_Manuf","JPN") = 899043.375 ;
vdfb("c_Manuf","a_Manuf","IND") = 163220.359375 ;
vdfb("c_Manuf","a_Manuf","SSA") = 80503.0703125 ;
vdfb("c_Manuf","a_Manuf","ROW") = 1681277.875 ;
vdfb("c_Manuf","a_ForestFish","USA") = 1143.984375 ;
vdfb("c_Manuf","a_ForestFish","EU_28") = 3450.60473632812 ;
vdfb("c_Manuf","a_ForestFish","CHN") = 6445.69140625 ;
vdfb("c_Manuf","a_ForestFish","JPN") = 781.208312988281 ;
vdfb("c_Manuf","a_ForestFish","IND") = 2496.89453125 ;
vdfb("c_Manuf","a_ForestFish","SSA") = 850.929077148438 ;
vdfb("c_Manuf","a_ForestFish","ROW") = 6469.31201171875 ;
vdfb("c_Manuf","a_Svces","USA") = 838881.125 ;
vdfb("c_Manuf","a_Svces","EU_28") = 792069.6875 ;
vdfb("c_Manuf","a_Svces","CHN") = 1904681.125 ;
vdfb("c_Manuf","a_Svces","JPN") = 300955.4375 ;
vdfb("c_Manuf","a_Svces","IND") = 161820.109375 ;
vdfb("c_Manuf","a_Svces","SSA") = 65826.109375 ;
vdfb("c_Manuf","a_Svces","ROW") = 1203481.625 ;
vdfb("c_ForestFish","a_Rice","USA") = 176.154144287109 ;
vdfb("c_ForestFish","a_Rice","EU_28") = 14.2347097396851 ;
vdfb("c_ForestFish","a_Rice","CHN") = 3.36878252029419 ;
vdfb("c_ForestFish","a_Rice","JPN") = 1.85439801216125 ;
vdfb("c_ForestFish","a_Rice","IND") = 1.44641482830048 ;
vdfb("c_ForestFish","a_Rice","SSA") = 23.52223777771 ;
vdfb("c_ForestFish","a_Rice","ROW") = 444.182647705078 ;
vdfb("c_ForestFish","a_Crops","USA") = 20124.947265625 ;
vdfb("c_ForestFish","a_Crops","EU_28") = 122.622756958008 ;
vdfb("c_ForestFish","a_Crops","CHN") = 4.62349557876587 ;
vdfb("c_ForestFish","a_Crops","JPN") = 10.1419124603271 ;
vdfb("c_ForestFish","a_Crops","IND") = 1.87764060497284 ;
vdfb("c_ForestFish","a_Crops","SSA") = 146.07063293457 ;
vdfb("c_ForestFish","a_Crops","ROW") = 578.486328125 ;
vdfb("c_ForestFish","a_Livestock","USA") = 3117.89404296875 ;
vdfb("c_ForestFish","a_Livestock","EU_28") = 110.637130737305 ;
vdfb("c_ForestFish","a_Livestock","CHN") = 2.71042633056641 ;
vdfb("c_ForestFish","a_Livestock","JPN") = 3.89397788047791 ;
vdfb("c_ForestFish","a_Livestock","IND") = 76.7874755859375 ;
vdfb("c_ForestFish","a_Livestock","SSA") = 295.723022460938 ;
vdfb("c_ForestFish","a_Livestock","ROW") = 795.541198730469 ;
vdfb("c_ForestFish","a_FoodProc","USA") = 2866.4853515625 ;
vdfb("c_ForestFish","a_FoodProc","EU_28") = 4825.36767578125 ;
vdfb("c_ForestFish","a_FoodProc","CHN") = 61950.47265625 ;
vdfb("c_ForestFish","a_FoodProc","JPN") = 7459.04296875 ;
vdfb("c_ForestFish","a_FoodProc","IND") = 5288.51025390625 ;
vdfb("c_ForestFish","a_FoodProc","SSA") = 3854.01586914062 ;
vdfb("c_ForestFish","a_FoodProc","ROW") = 41977.71484375 ;
vdfb("c_ForestFish","a_Energy","USA") = 77.4238739013672 ;
vdfb("c_ForestFish","a_Energy","EU_28") = 502.565155029297 ;
vdfb("c_ForestFish","a_Energy","CHN") = 208.240661621094 ;
vdfb("c_ForestFish","a_Energy","JPN") = 1.23242092132568 ;
vdfb("c_ForestFish","a_Energy","IND") = 41.8304901123047 ;
vdfb("c_ForestFish","a_Energy","SSA") = 37.9753875732422 ;
vdfb("c_ForestFish","a_Energy","ROW") = 960.911560058594 ;
vdfb("c_ForestFish","a_Textiles","USA") = 194.236434936523 ;
vdfb("c_ForestFish","a_Textiles","EU_28") = 458.888000488281 ;
vdfb("c_ForestFish","a_Textiles","CHN") = 405.106811523438 ;
vdfb("c_ForestFish","a_Textiles","JPN") = 9.16245460510254 ;
vdfb("c_ForestFish","a_Textiles","IND") = 52.1076164245605 ;
vdfb("c_ForestFish","a_Textiles","SSA") = 50.9603614807129 ;
vdfb("c_ForestFish","a_Textiles","ROW") = 362.989166259766 ;
vdfb("c_ForestFish","a_Chem","USA") = 3432.71020507812 ;
vdfb("c_ForestFish","a_Chem","EU_28") = 747.051513671875 ;
vdfb("c_ForestFish","a_Chem","CHN") = 13151.3515625 ;
vdfb("c_ForestFish","a_Chem","JPN") = 95.9938735961914 ;
vdfb("c_ForestFish","a_Chem","IND") = 5045.74951171875 ;
vdfb("c_ForestFish","a_Chem","SSA") = 245.051345825195 ;
vdfb("c_ForestFish","a_Chem","ROW") = 2970.03955078125 ;
vdfb("c_ForestFish","a_Manuf","USA") = 12698.1630859375 ;
vdfb("c_ForestFish","a_Manuf","EU_28") = 22781.4140625 ;
vdfb("c_ForestFish","a_Manuf","CHN") = 38282.20703125 ;
vdfb("c_ForestFish","a_Manuf","JPN") = 3980.4541015625 ;
vdfb("c_ForestFish","a_Manuf","IND") = 3049.45678710938 ;
vdfb("c_ForestFish","a_Manuf","SSA") = 6586.484375 ;
vdfb("c_ForestFish","a_Manuf","ROW") = 46612.453125 ;
vdfb("c_ForestFish","a_ForestFish","USA") = 6444.20654296875 ;
vdfb("c_ForestFish","a_ForestFish","EU_28") = 9852.634765625 ;
vdfb("c_ForestFish","a_ForestFish","CHN") = 8718.2431640625 ;
vdfb("c_ForestFish","a_ForestFish","JPN") = 1168.15710449219 ;
vdfb("c_ForestFish","a_ForestFish","IND") = 777.661743164062 ;
vdfb("c_ForestFish","a_ForestFish","SSA") = 2038.68627929688 ;
vdfb("c_ForestFish","a_ForestFish","ROW") = 26501.4453125 ;
vdfb("c_ForestFish","a_Svces","USA") = 4988.4560546875 ;
vdfb("c_ForestFish","a_Svces","EU_28") = 8318.3798828125 ;
vdfb("c_ForestFish","a_Svces","CHN") = 50035.76953125 ;
vdfb("c_ForestFish","a_Svces","JPN") = 3444.1630859375 ;
vdfb("c_ForestFish","a_Svces","IND") = 21696.37890625 ;
vdfb("c_ForestFish","a_Svces","SSA") = 7852.92333984375 ;
vdfb("c_ForestFish","a_Svces","ROW") = 35695.9375 ;
vdfb("c_Svces","a_Rice","USA") = 1828.34509277344 ;
vdfb("c_Svces","a_Rice","EU_28") = 1757.1875 ;
vdfb("c_Svces","a_Rice","CHN") = 14247.3701171875 ;
vdfb("c_Svces","a_Rice","JPN") = 5785.041015625 ;
vdfb("c_Svces","a_Rice","IND") = 10688.12109375 ;
vdfb("c_Svces","a_Rice","SSA") = 1689.85729980469 ;
vdfb("c_Svces","a_Rice","ROW") = 44820.6328125 ;
vdfb("c_Svces","a_Crops","USA") = 51805.90625 ;
vdfb("c_Svces","a_Crops","EU_28") = 38871.29296875 ;
vdfb("c_Svces","a_Crops","CHN") = 53447.73828125 ;
vdfb("c_Svces","a_Crops","JPN") = 7377.77490234375 ;
vdfb("c_Svces","a_Crops","IND") = 13449.30078125 ;
vdfb("c_Svces","a_Crops","SSA") = 20840.9765625 ;
vdfb("c_Svces","a_Crops","ROW") = 106311.8984375 ;
vdfb("c_Svces","a_Livestock","USA") = 41028.01953125 ;
vdfb("c_Svces","a_Livestock","EU_28") = 21650.619140625 ;
vdfb("c_Svces","a_Livestock","CHN") = 28587.501953125 ;
vdfb("c_Svces","a_Livestock","JPN") = 7426.26611328125 ;
vdfb("c_Svces","a_Livestock","IND") = 15161.5576171875 ;
vdfb("c_Svces","a_Livestock","SSA") = 6509.17724609375 ;
vdfb("c_Svces","a_Livestock","ROW") = 59357.5 ;
vdfb("c_Svces","a_FoodProc","USA") = 212471.796875 ;
vdfb("c_Svces","a_FoodProc","EU_28") = 306342.6875 ;
vdfb("c_Svces","a_FoodProc","CHN") = 236671.015625 ;
vdfb("c_Svces","a_FoodProc","JPN") = 56670.4921875 ;
vdfb("c_Svces","a_FoodProc","IND") = 38525.45703125 ;
vdfb("c_Svces","a_FoodProc","SSA") = 51466.51171875 ;
vdfb("c_Svces","a_FoodProc","ROW") = 366390.09375 ;
vdfb("c_Svces","a_Energy","USA") = 171896.09375 ;
vdfb("c_Svces","a_Energy","EU_28") = 109273.0703125 ;
vdfb("c_Svces","a_Energy","CHN") = 159079.515625 ;
vdfb("c_Svces","a_Energy","JPN") = 67553.8359375 ;
vdfb("c_Svces","a_Energy","IND") = 79410.921875 ;
vdfb("c_Svces","a_Energy","SSA") = 35934.60546875 ;
vdfb("c_Svces","a_Energy","ROW") = 423966.5 ;
vdfb("c_Svces","a_Textiles","USA") = 17853.83984375 ;
vdfb("c_Svces","a_Textiles","EU_28") = 82522.4140625 ;
vdfb("c_Svces","a_Textiles","CHN") = 192315.421875 ;
vdfb("c_Svces","a_Textiles","JPN") = 7925.80078125 ;
vdfb("c_Svces","a_Textiles","IND") = 21632.634765625 ;
vdfb("c_Svces","a_Textiles","SSA") = 11793.2568359375 ;
vdfb("c_Svces","a_Textiles","ROW") = 122699.7265625 ;
vdfb("c_Svces","a_Chem","USA") = 174007.859375 ;
vdfb("c_Svces","a_Chem","EU_28") = 275820.25 ;
vdfb("c_Svces","a_Chem","CHN") = 278943.875 ;
vdfb("c_Svces","a_Chem","JPN") = 93379.6640625 ;
vdfb("c_Svces","a_Chem","IND") = 30589.697265625 ;
vdfb("c_Svces","a_Chem","SSA") = 11408.390625 ;
vdfb("c_Svces","a_Chem","ROW") = 258459 ;
vdfb("c_Svces","a_Manuf","USA") = 704665.6875 ;
vdfb("c_Svces","a_Manuf","EU_28") = 1090196 ;
vdfb("c_Svces","a_Manuf","CHN") = 1120412 ;
vdfb("c_Svces","a_Manuf","JPN") = 386164.1875 ;
vdfb("c_Svces","a_Manuf","IND") = 110234.8671875 ;
vdfb("c_Svces","a_Manuf","SSA") = 96787.140625 ;
vdfb("c_Svces","a_Manuf","ROW") = 1095532.875 ;
vdfb("c_Svces","a_ForestFish","USA") = 8075.58740234375 ;
vdfb("c_Svces","a_ForestFish","EU_28") = 11587.9208984375 ;
vdfb("c_Svces","a_ForestFish","CHN") = 34206.41796875 ;
vdfb("c_Svces","a_ForestFish","JPN") = 2930.37768554688 ;
vdfb("c_Svces","a_ForestFish","IND") = 4331.6435546875 ;
vdfb("c_Svces","a_ForestFish","SSA") = 8900.345703125 ;
vdfb("c_Svces","a_ForestFish","ROW") = 36281.5390625 ;
vdfb("c_Svces","a_Svces","USA") = 7294576 ;
vdfb("c_Svces","a_Svces","EU_28") = 6652786 ;
vdfb("c_Svces","a_Svces","CHN") = 3999088.75 ;
vdfb("c_Svces","a_Svces","JPN") = 1611724.625 ;
vdfb("c_Svces","a_Svces","IND") = 495561 ;
vdfb("c_Svces","a_Svces","SSA") = 380480.96875 ;
vdfb("c_Svces","a_Svces","ROW") = 6287437.5 ;
parameter vdfp(*,*,*) ;
vdfp("c_Rice","a_Rice","USA") = 1909.79943847656 ;
vdfp("c_Rice","a_Rice","EU_28") = 1012.66351318359 ;
vdfp("c_Rice","a_Rice","CHN") = 51821.13671875 ;
vdfp("c_Rice","a_Rice","JPN") = 17032.80859375 ;
vdfp("c_Rice","a_Rice","IND") = 11403.7158203125 ;
vdfp("c_Rice","a_Rice","SSA") = 3767.58935546875 ;
vdfp("c_Rice","a_Rice","ROW") = 131585.25 ;
vdfp("c_Rice","a_Crops","USA") = 4.327552318573 ;
vdfp("c_Rice","a_Crops","EU_28") = 170.472396850586 ;
vdfp("c_Rice","a_Crops","CHN") = 304.300628662109 ;
vdfp("c_Rice","a_Crops","JPN") = 203.232940673828 ;
vdfp("c_Rice","a_Crops","IND") = 85.5478744506836 ;
vdfp("c_Rice","a_Crops","SSA") = 39.7475662231445 ;
vdfp("c_Rice","a_Crops","ROW") = 722.153625488281 ;
vdfp("c_Rice","a_Livestock","USA") = 13.6782579421997 ;
vdfp("c_Rice","a_Livestock","EU_28") = 57.3822250366211 ;
vdfp("c_Rice","a_Livestock","CHN") = 8353.755859375 ;
vdfp("c_Rice","a_Livestock","JPN") = 516.76318359375 ;
vdfp("c_Rice","a_Livestock","IND") = 1178.01574707031 ;
vdfp("c_Rice","a_Livestock","SSA") = 170.57487487793 ;
vdfp("c_Rice","a_Livestock","ROW") = 5352.859375 ;
vdfp("c_Rice","a_FoodProc","USA") = 1542.81433105469 ;
vdfp("c_Rice","a_FoodProc","EU_28") = 552.184631347656 ;
vdfp("c_Rice","a_FoodProc","CHN") = 32805.98046875 ;
vdfp("c_Rice","a_FoodProc","JPN") = 3923.40844726562 ;
vdfp("c_Rice","a_FoodProc","IND") = 39032.1015625 ;
vdfp("c_Rice","a_FoodProc","SSA") = 1267.61486816406 ;
vdfp("c_Rice","a_FoodProc","ROW") = 25182.525390625 ;
vdfp("c_Rice","a_Energy","USA") = 0.688192427158356 ;
vdfp("c_Rice","a_Energy","EU_28") = 0.583700895309448 ;
vdfp("c_Rice","a_Energy","CHN") = 17.9700183868408 ;
vdfp("c_Rice","a_Energy","JPN") = 0.159057393670082 ;
vdfp("c_Rice","a_Energy","IND") = 35.9343757629395 ;
vdfp("c_Rice","a_Energy","SSA") = 4.04282903671265 ;
vdfp("c_Rice","a_Energy","ROW") = 1259.34838867188 ;
vdfp("c_Rice","a_Textiles","USA") = 0.419756501913071 ;
vdfp("c_Rice","a_Textiles","EU_28") = 2.20540142059326 ;
vdfp("c_Rice","a_Textiles","CHN") = 1697.82604980469 ;
vdfp("c_Rice","a_Textiles","JPN") = 0.358892321586609 ;
vdfp("c_Rice","a_Textiles","IND") = 0.634818732738495 ;
vdfp("c_Rice","a_Textiles","SSA") = 8.17331600189209 ;
vdfp("c_Rice","a_Textiles","ROW") = 306.650268554688 ;
vdfp("c_Rice","a_Chem","USA") = 115.831115722656 ;
vdfp("c_Rice","a_Chem","EU_28") = 10.0653238296509 ;
vdfp("c_Rice","a_Chem","CHN") = 7288.470703125 ;
vdfp("c_Rice","a_Chem","JPN") = 29.373233795166 ;
vdfp("c_Rice","a_Chem","IND") = 183.388519287109 ;
vdfp("c_Rice","a_Chem","SSA") = 269.94287109375 ;
vdfp("c_Rice","a_Chem","ROW") = 793.18505859375 ;
vdfp("c_Rice","a_Manuf","USA") = 27.9291954040527 ;
vdfp("c_Rice","a_Manuf","EU_28") = 15.7706336975098 ;
vdfp("c_Rice","a_Manuf","CHN") = 3953.18139648438 ;
vdfp("c_Rice","a_Manuf","JPN") = 99.0351333618164 ;
vdfp("c_Rice","a_Manuf","IND") = 9.44552516937256 ;
vdfp("c_Rice","a_Manuf","SSA") = 103.515357971191 ;
vdfp("c_Rice","a_Manuf","ROW") = 1045.18994140625 ;
vdfp("c_Rice","a_ForestFish","USA") = 6.56218242645264 ;
vdfp("c_Rice","a_ForestFish","EU_28") = 2.84215617179871 ;
vdfp("c_Rice","a_ForestFish","CHN") = 837.967712402344 ;
vdfp("c_Rice","a_ForestFish","JPN") = 35.7318267822266 ;
vdfp("c_Rice","a_ForestFish","IND") = 2.00001335144043 ;
vdfp("c_Rice","a_ForestFish","SSA") = 1.81752288341522 ;
vdfp("c_Rice","a_ForestFish","ROW") = 1019.39678955078 ;
vdfp("c_Rice","a_Svces","USA") = 691.25 ;
vdfp("c_Rice","a_Svces","EU_28") = 1035.53015136719 ;
vdfp("c_Rice","a_Svces","CHN") = 14506.337890625 ;
vdfp("c_Rice","a_Svces","JPN") = 4049.36987304688 ;
vdfp("c_Rice","a_Svces","IND") = 8524.7060546875 ;
vdfp("c_Rice","a_Svces","SSA") = 1125.82482910156 ;
vdfp("c_Rice","a_Svces","ROW") = 17197.595703125 ;
vdfp("c_Crops","a_Rice","USA") = 8.25769901275635 ;
vdfp("c_Crops","a_Rice","EU_28") = 11.0904760360718 ;
vdfp("c_Crops","a_Rice","CHN") = 1160.59558105469 ;
vdfp("c_Crops","a_Rice","JPN") = 7.11702108383179 ;
vdfp("c_Crops","a_Rice","IND") = 259.628448486328 ;
vdfp("c_Crops","a_Rice","SSA") = 839.67724609375 ;
vdfp("c_Crops","a_Rice","ROW") = 1418.78161621094 ;
vdfp("c_Crops","a_Crops","USA") = 7168.86279296875 ;
vdfp("c_Crops","a_Crops","EU_28") = 16466.064453125 ;
vdfp("c_Crops","a_Crops","CHN") = 62806.34375 ;
vdfp("c_Crops","a_Crops","JPN") = 467.318084716797 ;
vdfp("c_Crops","a_Crops","IND") = 597.610412597656 ;
vdfp("c_Crops","a_Crops","SSA") = 30728.107421875 ;
vdfp("c_Crops","a_Crops","ROW") = 98757.71875 ;
vdfp("c_Crops","a_Livestock","USA") = 7469.7041015625 ;
vdfp("c_Crops","a_Livestock","EU_28") = 11149.908203125 ;
vdfp("c_Crops","a_Livestock","CHN") = 27692.587890625 ;
vdfp("c_Crops","a_Livestock","JPN") = 1029.57666015625 ;
vdfp("c_Crops","a_Livestock","IND") = 7170.67578125 ;
vdfp("c_Crops","a_Livestock","SSA") = 3016.58325195312 ;
vdfp("c_Crops","a_Livestock","ROW") = 36673.0546875 ;
vdfp("c_Crops","a_FoodProc","USA") = 75459.3359375 ;
vdfp("c_Crops","a_FoodProc","EU_28") = 46219.6171875 ;
vdfp("c_Crops","a_FoodProc","CHN") = 255931.359375 ;
vdfp("c_Crops","a_FoodProc","JPN") = 5480.990234375 ;
vdfp("c_Crops","a_FoodProc","IND") = 36930.8203125 ;
vdfp("c_Crops","a_FoodProc","SSA") = 22531.93359375 ;
vdfp("c_Crops","a_FoodProc","ROW") = 221567.1875 ;
vdfp("c_Crops","a_Energy","USA") = 33.8061370849609 ;
vdfp("c_Crops","a_Energy","EU_28") = 79.8177947998047 ;
vdfp("c_Crops","a_Energy","CHN") = 109.357398986816 ;
vdfp("c_Crops","a_Energy","JPN") = 5.27432632446289 ;
vdfp("c_Crops","a_Energy","IND") = 1533.25842285156 ;
vdfp("c_Crops","a_Energy","SSA") = 140.304092407227 ;
vdfp("c_Crops","a_Energy","ROW") = 1888.4775390625 ;
vdfp("c_Crops","a_Textiles","USA") = 543.129089355469 ;
vdfp("c_Crops","a_Textiles","EU_28") = 734.335754394531 ;
vdfp("c_Crops","a_Textiles","CHN") = 46116.18359375 ;
vdfp("c_Crops","a_Textiles","JPN") = 154.43864440918 ;
vdfp("c_Crops","a_Textiles","IND") = 21927.287109375 ;
vdfp("c_Crops","a_Textiles","SSA") = 1315.68676757812 ;
vdfp("c_Crops","a_Textiles","ROW") = 18865.697265625 ;
vdfp("c_Crops","a_Chem","USA") = 13755.212890625 ;
vdfp("c_Crops","a_Chem","EU_28") = 1068.15161132812 ;
vdfp("c_Crops","a_Chem","CHN") = 31362.84765625 ;
vdfp("c_Crops","a_Chem","JPN") = 1345.27355957031 ;
vdfp("c_Crops","a_Chem","IND") = 3472.0087890625 ;
vdfp("c_Crops","a_Chem","SSA") = 492.414123535156 ;
vdfp("c_Crops","a_Chem","ROW") = 14132.98046875 ;
vdfp("c_Crops","a_Manuf","USA") = 378.23291015625 ;
vdfp("c_Crops","a_Manuf","EU_28") = 223.405120849609 ;
vdfp("c_Crops","a_Manuf","CHN") = 16894.751953125 ;
vdfp("c_Crops","a_Manuf","JPN") = 51.2961959838867 ;
vdfp("c_Crops","a_Manuf","IND") = 103.221885681152 ;
vdfp("c_Crops","a_Manuf","SSA") = 388.864105224609 ;
vdfp("c_Crops","a_Manuf","ROW") = 1539.02380371094 ;
vdfp("c_Crops","a_ForestFish","USA") = 270.345703125 ;
vdfp("c_Crops","a_ForestFish","EU_28") = 146.574081420898 ;
vdfp("c_Crops","a_ForestFish","CHN") = 1177.5009765625 ;
vdfp("c_Crops","a_ForestFish","JPN") = 1.84959816932678 ;
vdfp("c_Crops","a_ForestFish","IND") = 20.438533782959 ;
vdfp("c_Crops","a_ForestFish","SSA") = 171.861083984375 ;
vdfp("c_Crops","a_ForestFish","ROW") = 1242.15649414062 ;
vdfp("c_Crops","a_Svces","USA") = 6418.90283203125 ;
vdfp("c_Crops","a_Svces","EU_28") = 12935.5009765625 ;
vdfp("c_Crops","a_Svces","CHN") = 50883.16015625 ;
vdfp("c_Crops","a_Svces","JPN") = 5101.6806640625 ;
vdfp("c_Crops","a_Svces","IND") = 22361.009765625 ;
vdfp("c_Crops","a_Svces","SSA") = 5079.70703125 ;
vdfp("c_Crops","a_Svces","ROW") = 41735.55078125 ;
vdfp("c_Livestock","a_Rice","USA") = 134.533386230469 ;
vdfp("c_Livestock","a_Rice","EU_28") = 20.0268363952637 ;
vdfp("c_Livestock","a_Rice","CHN") = 31.3126335144043 ;
vdfp("c_Livestock","a_Rice","JPN") = 122.165702819824 ;
vdfp("c_Livestock","a_Rice","IND") = 816.959716796875 ;
vdfp("c_Livestock","a_Rice","SSA") = 228.699798583984 ;
vdfp("c_Livestock","a_Rice","ROW") = 1085.705078125 ;
vdfp("c_Livestock","a_Crops","USA") = 266.333129882812 ;
vdfp("c_Livestock","a_Crops","EU_28") = 1561.45825195312 ;
vdfp("c_Livestock","a_Crops","CHN") = 272.671630859375 ;
vdfp("c_Livestock","a_Crops","JPN") = 202.232330322266 ;
vdfp("c_Livestock","a_Crops","IND") = 778.594116210938 ;
vdfp("c_Livestock","a_Crops","SSA") = 1110.85729980469 ;
vdfp("c_Livestock","a_Crops","ROW") = 13504.5830078125 ;
vdfp("c_Livestock","a_Livestock","USA") = 17956.322265625 ;
vdfp("c_Livestock","a_Livestock","EU_28") = 5280.5966796875 ;
vdfp("c_Livestock","a_Livestock","CHN") = 24607.033203125 ;
vdfp("c_Livestock","a_Livestock","JPN") = 2017.17004394531 ;
vdfp("c_Livestock","a_Livestock","IND") = 811.109680175781 ;
vdfp("c_Livestock","a_Livestock","SSA") = 9532.228515625 ;
vdfp("c_Livestock","a_Livestock","ROW") = 80768.21875 ;
vdfp("c_Livestock","a_FoodProc","USA") = 131153.203125 ;
vdfp("c_Livestock","a_FoodProc","EU_28") = 124519.3359375 ;
vdfp("c_Livestock","a_FoodProc","CHN") = 144311.75 ;
vdfp("c_Livestock","a_FoodProc","JPN") = 20948.66796875 ;
vdfp("c_Livestock","a_FoodProc","IND") = 17157.3984375 ;
vdfp("c_Livestock","a_FoodProc","SSA") = 18272.025390625 ;
vdfp("c_Livestock","a_FoodProc","ROW") = 273447.53125 ;
vdfp("c_Livestock","a_Energy","USA") = 0.753273785114288 ;
vdfp("c_Livestock","a_Energy","EU_28") = 67.7447204589844 ;
vdfp("c_Livestock","a_Energy","CHN") = 1.61126887798309 ;
vdfp("c_Livestock","a_Energy","JPN") = 1.04377830028534 ;
vdfp("c_Livestock","a_Energy","IND") = 138.289031982422 ;
vdfp("c_Livestock","a_Energy","SSA") = 11.1359128952026 ;
vdfp("c_Livestock","a_Energy","ROW") = 215.486343383789 ;
vdfp("c_Livestock","a_Textiles","USA") = 166.061660766602 ;
vdfp("c_Livestock","a_Textiles","EU_28") = 422.746887207031 ;
vdfp("c_Livestock","a_Textiles","CHN") = 22390.572265625 ;
vdfp("c_Livestock","a_Textiles","JPN") = 16.1092891693115 ;
vdfp("c_Livestock","a_Textiles","IND") = 1265.11279296875 ;
vdfp("c_Livestock","a_Textiles","SSA") = 928.427917480469 ;
vdfp("c_Livestock","a_Textiles","ROW") = 5048.9296875 ;
vdfp("c_Livestock","a_Chem","USA") = 177.482192993164 ;
vdfp("c_Livestock","a_Chem","EU_28") = 1091.10900878906 ;
vdfp("c_Livestock","a_Chem","CHN") = 20790.5078125 ;
vdfp("c_Livestock","a_Chem","JPN") = 83.4835433959961 ;
vdfp("c_Livestock","a_Chem","IND") = 4192.7392578125 ;
vdfp("c_Livestock","a_Chem","SSA") = 89.1746597290039 ;
vdfp("c_Livestock","a_Chem","ROW") = 1772.67895507812 ;
vdfp("c_Livestock","a_Manuf","USA") = 16.9556198120117 ;
vdfp("c_Livestock","a_Manuf","EU_28") = 148.877639770508 ;
vdfp("c_Livestock","a_Manuf","CHN") = 2865.18627929688 ;
vdfp("c_Livestock","a_Manuf","JPN") = 6.01187801361084 ;
vdfp("c_Livestock","a_Manuf","IND") = 65.0536193847656 ;
vdfp("c_Livestock","a_Manuf","SSA") = 58.4521102905273 ;
vdfp("c_Livestock","a_Manuf","ROW") = 899.53125 ;
vdfp("c_Livestock","a_ForestFish","USA") = 389.271820068359 ;
vdfp("c_Livestock","a_ForestFish","EU_28") = 114.764266967773 ;
vdfp("c_Livestock","a_ForestFish","CHN") = 19.9665851593018 ;
vdfp("c_Livestock","a_ForestFish","JPN") = 3.30553030967712 ;
vdfp("c_Livestock","a_ForestFish","IND") = 537.296936035156 ;
vdfp("c_Livestock","a_ForestFish","SSA") = 747.869506835938 ;
vdfp("c_Livestock","a_ForestFish","ROW") = 556.079040527344 ;
vdfp("c_Livestock","a_Svces","USA") = 3670.82739257812 ;
vdfp("c_Livestock","a_Svces","EU_28") = 8496.775390625 ;
vdfp("c_Livestock","a_Svces","CHN") = 10367.146484375 ;
vdfp("c_Livestock","a_Svces","JPN") = 2726.7568359375 ;
vdfp("c_Livestock","a_Svces","IND") = 38211.9609375 ;
vdfp("c_Livestock","a_Svces","SSA") = 4627.2138671875 ;
vdfp("c_Livestock","a_Svces","ROW") = 22695.03125 ;
vdfp("c_FoodProc","a_Rice","USA") = 64.3584289550781 ;
vdfp("c_FoodProc","a_Rice","EU_28") = 42.0649070739746 ;
vdfp("c_FoodProc","a_Rice","CHN") = 255.548370361328 ;
vdfp("c_FoodProc","a_Rice","JPN") = 2.00065112113953 ;
vdfp("c_FoodProc","a_Rice","IND") = 0.553491592407227 ;
vdfp("c_FoodProc","a_Rice","SSA") = 177.929061889648 ;
vdfp("c_FoodProc","a_Rice","ROW") = 1373.60278320312 ;
vdfp("c_FoodProc","a_Crops","USA") = 1.08856236934662 ;
vdfp("c_FoodProc","a_Crops","EU_28") = 1779.96740722656 ;
vdfp("c_FoodProc","a_Crops","CHN") = 1354.71008300781 ;
vdfp("c_FoodProc","a_Crops","JPN") = 62.86376953125 ;
vdfp("c_FoodProc","a_Crops","IND") = 1.05086076259613 ;
vdfp("c_FoodProc","a_Crops","SSA") = 1108.42407226562 ;
vdfp("c_FoodProc","a_Crops","ROW") = 4307.2666015625 ;
vdfp("c_FoodProc","a_Livestock","USA") = 33813.2265625 ;
vdfp("c_FoodProc","a_Livestock","EU_28") = 37154.06640625 ;
vdfp("c_FoodProc","a_Livestock","CHN") = 78352.3359375 ;
vdfp("c_FoodProc","a_Livestock","JPN") = 10793.87109375 ;
vdfp("c_FoodProc","a_Livestock","IND") = 2185.0322265625 ;
vdfp("c_FoodProc","a_Livestock","SSA") = 4992.5419921875 ;
vdfp("c_FoodProc","a_Livestock","ROW") = 73233.4140625 ;
vdfp("c_FoodProc","a_FoodProc","USA") = 231872.203125 ;
vdfp("c_FoodProc","a_FoodProc","EU_28") = 235071.46875 ;
vdfp("c_FoodProc","a_FoodProc","CHN") = 367548.6875 ;
vdfp("c_FoodProc","a_FoodProc","JPN") = 47243.8046875 ;
vdfp("c_FoodProc","a_FoodProc","IND") = 10359.2490234375 ;
vdfp("c_FoodProc","a_FoodProc","SSA") = 37440.953125 ;
vdfp("c_FoodProc","a_FoodProc","ROW") = 283940.8125 ;
vdfp("c_FoodProc","a_Energy","USA") = 127.875175476074 ;
vdfp("c_FoodProc","a_Energy","EU_28") = 571.199829101562 ;
vdfp("c_FoodProc","a_Energy","CHN") = 6835.74951171875 ;
vdfp("c_FoodProc","a_Energy","JPN") = 3.66133546829224 ;
vdfp("c_FoodProc","a_Energy","IND") = 80.8473968505859 ;
vdfp("c_FoodProc","a_Energy","SSA") = 58.4650344848633 ;
vdfp("c_FoodProc","a_Energy","ROW") = 1573.740234375 ;
vdfp("c_FoodProc","a_Textiles","USA") = 1368.78283691406 ;
vdfp("c_FoodProc","a_Textiles","EU_28") = 2902.29077148438 ;
vdfp("c_FoodProc","a_Textiles","CHN") = 29373.78515625 ;
vdfp("c_FoodProc","a_Textiles","JPN") = 134.104904174805 ;
vdfp("c_FoodProc","a_Textiles","IND") = 10.551607131958 ;
vdfp("c_FoodProc","a_Textiles","SSA") = 922.222595214844 ;
vdfp("c_FoodProc","a_Textiles","ROW") = 6928.5205078125 ;
vdfp("c_FoodProc","a_Chem","USA") = 5290.7421875 ;
vdfp("c_FoodProc","a_Chem","EU_28") = 13740.1953125 ;
vdfp("c_FoodProc","a_Chem","CHN") = 46648.1875 ;
vdfp("c_FoodProc","a_Chem","JPN") = 1428.33325195312 ;
vdfp("c_FoodProc","a_Chem","IND") = 2637.28393554688 ;
vdfp("c_FoodProc","a_Chem","SSA") = 1293.55029296875 ;
vdfp("c_FoodProc","a_Chem","ROW") = 11241.0185546875 ;
vdfp("c_FoodProc","a_Manuf","USA") = 3073.5908203125 ;
vdfp("c_FoodProc","a_Manuf","EU_28") = 4204.724609375 ;
vdfp("c_FoodProc","a_Manuf","CHN") = 43998.2109375 ;
vdfp("c_FoodProc","a_Manuf","JPN") = 297.406005859375 ;
vdfp("c_FoodProc","a_Manuf","IND") = 27.017406463623 ;
vdfp("c_FoodProc","a_Manuf","SSA") = 934.602966308594 ;
vdfp("c_FoodProc","a_Manuf","ROW") = 5588.40185546875 ;
vdfp("c_FoodProc","a_ForestFish","USA") = 762.202453613281 ;
vdfp("c_FoodProc","a_ForestFish","EU_28") = 758.011047363281 ;
vdfp("c_FoodProc","a_ForestFish","CHN") = 25610.87890625 ;
vdfp("c_FoodProc","a_ForestFish","JPN") = 1060.00329589844 ;
vdfp("c_FoodProc","a_ForestFish","IND") = 31.4682292938232 ;
vdfp("c_FoodProc","a_ForestFish","SSA") = 272.254058837891 ;
vdfp("c_FoodProc","a_ForestFish","ROW") = 15996.2080078125 ;
vdfp("c_FoodProc","a_Svces","USA") = 161877.953125 ;
vdfp("c_FoodProc","a_Svces","EU_28") = 228347.171875 ;
vdfp("c_FoodProc","a_Svces","CHN") = 269682.75 ;
vdfp("c_FoodProc","a_Svces","JPN") = 51805.91796875 ;
vdfp("c_FoodProc","a_Svces","IND") = 9120.5390625 ;
vdfp("c_FoodProc","a_Svces","SSA") = 24490.810546875 ;
vdfp("c_FoodProc","a_Svces","ROW") = 310558.78125 ;
vdfp("c_Energy","a_Rice","USA") = 132.45719909668 ;
vdfp("c_Energy","a_Rice","EU_28") = 214.113037109375 ;
vdfp("c_Energy","a_Rice","CHN") = 2451.2109375 ;
vdfp("c_Energy","a_Rice","JPN") = 384.442657470703 ;
vdfp("c_Energy","a_Rice","IND") = 22234.220703125 ;
vdfp("c_Energy","a_Rice","SSA") = 73.0904769897461 ;
vdfp("c_Energy","a_Rice","ROW") = 4484.09228515625 ;
vdfp("c_Energy","a_Crops","USA") = 5740.96142578125 ;
vdfp("c_Energy","a_Crops","EU_28") = 11070.541015625 ;
vdfp("c_Energy","a_Crops","CHN") = 11969.84765625 ;
vdfp("c_Energy","a_Crops","JPN") = 1081.16259765625 ;
vdfp("c_Energy","a_Crops","IND") = 16627.923828125 ;
vdfp("c_Energy","a_Crops","SSA") = 672.4677734375 ;
vdfp("c_Energy","a_Crops","ROW") = 24476.77734375 ;
vdfp("c_Energy","a_Livestock","USA") = 4208.31982421875 ;
vdfp("c_Energy","a_Livestock","EU_28") = 4564.2353515625 ;
vdfp("c_Energy","a_Livestock","CHN") = 2017.03491210938 ;
vdfp("c_Energy","a_Livestock","JPN") = 400.369964599609 ;
vdfp("c_Energy","a_Livestock","IND") = 49.4278450012207 ;
vdfp("c_Energy","a_Livestock","SSA") = 859.492858886719 ;
vdfp("c_Energy","a_Livestock","ROW") = 12913.2158203125 ;
vdfp("c_Energy","a_FoodProc","USA") = 12845.431640625 ;
vdfp("c_Energy","a_FoodProc","EU_28") = 23443.107421875 ;
vdfp("c_Energy","a_FoodProc","CHN") = 16119.63671875 ;
vdfp("c_Energy","a_FoodProc","JPN") = 5652.37890625 ;
vdfp("c_Energy","a_FoodProc","IND") = 4034.27758789062 ;
vdfp("c_Energy","a_FoodProc","SSA") = 1083.99169921875 ;
vdfp("c_Energy","a_FoodProc","ROW") = 34730.5078125 ;
vdfp("c_Energy","a_Energy","USA") = 316150.8125 ;
vdfp("c_Energy","a_Energy","EU_28") = 138577.890625 ;
vdfp("c_Energy","a_Energy","CHN") = 398523.9375 ;
vdfp("c_Energy","a_Energy","JPN") = 41155.546875 ;
vdfp("c_Energy","a_Energy","IND") = 100749.859375 ;
vdfp("c_Energy","a_Energy","SSA") = 27688.33203125 ;
vdfp("c_Energy","a_Energy","ROW") = 777126.625 ;
vdfp("c_Energy","a_Textiles","USA") = 2190.81127929688 ;
vdfp("c_Energy","a_Textiles","EU_28") = 4829.5615234375 ;
vdfp("c_Energy","a_Textiles","CHN") = 24281.689453125 ;
vdfp("c_Energy","a_Textiles","JPN") = 1115.63562011719 ;
vdfp("c_Energy","a_Textiles","IND") = 2726.63818359375 ;
vdfp("c_Energy","a_Textiles","SSA") = 399.337066650391 ;
vdfp("c_Energy","a_Textiles","ROW") = 15254.830078125 ;
vdfp("c_Energy","a_Chem","USA") = 71418.9609375 ;
vdfp("c_Energy","a_Chem","EU_28") = 92862.3515625 ;
vdfp("c_Energy","a_Chem","CHN") = 138704.09375 ;
vdfp("c_Energy","a_Chem","JPN") = 41240.9296875 ;
vdfp("c_Energy","a_Chem","IND") = 31725.810546875 ;
vdfp("c_Energy","a_Chem","SSA") = 3816.6572265625 ;
vdfp("c_Energy","a_Chem","ROW") = 184057.203125 ;
vdfp("c_Energy","a_Manuf","USA") = 64496.734375 ;
vdfp("c_Energy","a_Manuf","EU_28") = 124671.5390625 ;
vdfp("c_Energy","a_Manuf","CHN") = 308253.8125 ;
vdfp("c_Energy","a_Manuf","JPN") = 56905.17578125 ;
vdfp("c_Energy","a_Manuf","IND") = 88518.734375 ;
vdfp("c_Energy","a_Manuf","SSA") = 19736.779296875 ;
vdfp("c_Energy","a_Manuf","ROW") = 295773.75 ;
vdfp("c_Energy","a_ForestFish","USA") = 1148.17602539062 ;
vdfp("c_Energy","a_ForestFish","EU_28") = 3966.45654296875 ;
vdfp("c_Energy","a_ForestFish","CHN") = 3880.49755859375 ;
vdfp("c_Energy","a_ForestFish","JPN") = 1799.97143554688 ;
vdfp("c_Energy","a_ForestFish","IND") = 2564.39135742188 ;
vdfp("c_Energy","a_ForestFish","SSA") = 282.840026855469 ;
vdfp("c_Energy","a_ForestFish","ROW") = 8866.1806640625 ;
vdfp("c_Energy","a_Svces","USA") = 387289.59375 ;
vdfp("c_Energy","a_Svces","EU_28") = 370971.4375 ;
vdfp("c_Energy","a_Svces","CHN") = 225198.6875 ;
vdfp("c_Energy","a_Svces","JPN") = 115048.1796875 ;
vdfp("c_Energy","a_Svces","IND") = 104783.8203125 ;
vdfp("c_Energy","a_Svces","SSA") = 23535.7578125 ;
vdfp("c_Energy","a_Svces","ROW") = 516207.71875 ;
vdfp("c_Textiles","a_Rice","USA") = 26.048095703125 ;
vdfp("c_Textiles","a_Rice","EU_28") = 16.1574001312256 ;
vdfp("c_Textiles","a_Rice","CHN") = 362.781158447266 ;
vdfp("c_Textiles","a_Rice","JPN") = 32.6264305114746 ;
vdfp("c_Textiles","a_Rice","IND") = 3.26885461807251 ;
vdfp("c_Textiles","a_Rice","SSA") = 17.2143898010254 ;
vdfp("c_Textiles","a_Rice","ROW") = 392.383209228516 ;
vdfp("c_Textiles","a_Crops","USA") = 111.464195251465 ;
vdfp("c_Textiles","a_Crops","EU_28") = 284.935516357422 ;
vdfp("c_Textiles","a_Crops","CHN") = 28.8871669769287 ;
vdfp("c_Textiles","a_Crops","JPN") = 47.0032730102539 ;
vdfp("c_Textiles","a_Crops","IND") = 3.63113713264465 ;
vdfp("c_Textiles","a_Crops","SSA") = 471.918518066406 ;
vdfp("c_Textiles","a_Crops","ROW") = 1009.21783447266 ;
vdfp("c_Textiles","a_Livestock","USA") = 13.4201421737671 ;
vdfp("c_Textiles","a_Livestock","EU_28") = 146.030975341797 ;
vdfp("c_Textiles","a_Livestock","CHN") = 5.78551626205444 ;
vdfp("c_Textiles","a_Livestock","JPN") = 10.0130701065063 ;
vdfp("c_Textiles","a_Livestock","IND") = 3.60866618156433 ;
vdfp("c_Textiles","a_Livestock","SSA") = 125.974784851074 ;
vdfp("c_Textiles","a_Livestock","ROW") = 631.277160644531 ;
vdfp("c_Textiles","a_FoodProc","USA") = 141.889465332031 ;
vdfp("c_Textiles","a_FoodProc","EU_28") = 1675.58447265625 ;
vdfp("c_Textiles","a_FoodProc","CHN") = 6923.99169921875 ;
vdfp("c_Textiles","a_FoodProc","JPN") = 143.51237487793 ;
vdfp("c_Textiles","a_FoodProc","IND") = 42.0344924926758 ;
vdfp("c_Textiles","a_FoodProc","SSA") = 222.779968261719 ;
vdfp("c_Textiles","a_FoodProc","ROW") = 3429.85913085938 ;
vdfp("c_Textiles","a_Energy","USA") = 30.306510925293 ;
vdfp("c_Textiles","a_Energy","EU_28") = 201.791259765625 ;
vdfp("c_Textiles","a_Energy","CHN") = 2934.40014648438 ;
vdfp("c_Textiles","a_Energy","JPN") = 21.6406230926514 ;
vdfp("c_Textiles","a_Energy","IND") = 1006.28442382812 ;
vdfp("c_Textiles","a_Energy","SSA") = 88.3630294799805 ;
vdfp("c_Textiles","a_Energy","ROW") = 3083.10473632812 ;
vdfp("c_Textiles","a_Textiles","USA") = 11479.328125 ;
vdfp("c_Textiles","a_Textiles","EU_28") = 59164.30078125 ;
vdfp("c_Textiles","a_Textiles","CHN") = 582961.75 ;
vdfp("c_Textiles","a_Textiles","JPN") = 6408.46923828125 ;
vdfp("c_Textiles","a_Textiles","IND") = 24637.638671875 ;
vdfp("c_Textiles","a_Textiles","SSA") = 4769.2529296875 ;
vdfp("c_Textiles","a_Textiles","ROW") = 156765.015625 ;
vdfp("c_Textiles","a_Chem","USA") = 2097.55395507812 ;
vdfp("c_Textiles","a_Chem","EU_28") = 5373.61083984375 ;
vdfp("c_Textiles","a_Chem","CHN") = 21743.345703125 ;
vdfp("c_Textiles","a_Chem","JPN") = 578.492492675781 ;
vdfp("c_Textiles","a_Chem","IND") = 3016.43408203125 ;
vdfp("c_Textiles","a_Chem","SSA") = 156.176147460938 ;
vdfp("c_Textiles","a_Chem","ROW") = 5670.60302734375 ;
vdfp("c_Textiles","a_Manuf","USA") = 12528.54296875 ;
vdfp("c_Textiles","a_Manuf","EU_28") = 16235.5439453125 ;
vdfp("c_Textiles","a_Manuf","CHN") = 68223.2109375 ;
vdfp("c_Textiles","a_Manuf","JPN") = 2394.0947265625 ;
vdfp("c_Textiles","a_Manuf","IND") = 934.628662109375 ;
vdfp("c_Textiles","a_Manuf","SSA") = 732.161315917969 ;
vdfp("c_Textiles","a_Manuf","ROW") = 21219.1484375 ;
vdfp("c_Textiles","a_ForestFish","USA") = 63.2515068054199 ;
vdfp("c_Textiles","a_ForestFish","EU_28") = 317.630950927734 ;
vdfp("c_Textiles","a_ForestFish","CHN") = 200.70832824707 ;
vdfp("c_Textiles","a_ForestFish","JPN") = 152.957962036133 ;
vdfp("c_Textiles","a_ForestFish","IND") = 108.053802490234 ;
vdfp("c_Textiles","a_ForestFish","SSA") = 221.730102539062 ;
vdfp("c_Textiles","a_ForestFish","ROW") = 982.472351074219 ;
vdfp("c_Textiles","a_Svces","USA") = 11399.4423828125 ;
vdfp("c_Textiles","a_Svces","EU_28") = 22745.61328125 ;
vdfp("c_Textiles","a_Svces","CHN") = 93709.96875 ;
vdfp("c_Textiles","a_Svces","JPN") = 6631.8359375 ;
vdfp("c_Textiles","a_Svces","IND") = 5262.2197265625 ;
vdfp("c_Textiles","a_Svces","SSA") = 2976.39697265625 ;
vdfp("c_Textiles","a_Svces","ROW") = 52164.29296875 ;
vdfp("c_Chem","a_Rice","USA") = 359.874725341797 ;
vdfp("c_Chem","a_Rice","EU_28") = 155.908218383789 ;
vdfp("c_Chem","a_Rice","CHN") = 11273.505859375 ;
vdfp("c_Chem","a_Rice","JPN") = 1428.14916992188 ;
vdfp("c_Chem","a_Rice","IND") = 407.633392333984 ;
vdfp("c_Chem","a_Rice","SSA") = 90.9030532836914 ;
vdfp("c_Chem","a_Rice","ROW") = 5741.13037109375 ;
vdfp("c_Chem","a_Crops","USA") = 18393.484375 ;
vdfp("c_Chem","a_Crops","EU_28") = 7434.61962890625 ;
vdfp("c_Chem","a_Crops","CHN") = 72285.96875 ;
vdfp("c_Chem","a_Crops","JPN") = 3238.70776367188 ;
vdfp("c_Chem","a_Crops","IND") = 119.791404724121 ;
vdfp("c_Chem","a_Crops","SSA") = 3375.82739257812 ;
vdfp("c_Chem","a_Crops","ROW") = 44767.875 ;
vdfp("c_Chem","a_Livestock","USA") = 1792.35180664062 ;
vdfp("c_Chem","a_Livestock","EU_28") = 1010.53570556641 ;
vdfp("c_Chem","a_Livestock","CHN") = 3445.44287109375 ;
vdfp("c_Chem","a_Livestock","JPN") = 480.274871826172 ;
vdfp("c_Chem","a_Livestock","IND") = 19.2307720184326 ;
vdfp("c_Chem","a_Livestock","SSA") = 257.725952148438 ;
vdfp("c_Chem","a_Livestock","ROW") = 8342.115234375 ;
vdfp("c_Chem","a_FoodProc","USA") = 32834.41015625 ;
vdfp("c_Chem","a_FoodProc","EU_28") = 19296.185546875 ;
vdfp("c_Chem","a_FoodProc","CHN") = 29726.419921875 ;
vdfp("c_Chem","a_FoodProc","JPN") = 8466.7705078125 ;
vdfp("c_Chem","a_FoodProc","IND") = 3649.25634765625 ;
vdfp("c_Chem","a_FoodProc","SSA") = 1123.72155761719 ;
vdfp("c_Chem","a_FoodProc","ROW") = 36269.94140625 ;
vdfp("c_Chem","a_Energy","USA") = 8745.6103515625 ;
vdfp("c_Chem","a_Energy","EU_28") = 6535.4052734375 ;
vdfp("c_Chem","a_Energy","CHN") = 13901.8994140625 ;
vdfp("c_Chem","a_Energy","JPN") = 468.08349609375 ;
vdfp("c_Chem","a_Energy","IND") = 9520.7353515625 ;
vdfp("c_Chem","a_Energy","SSA") = 511.12451171875 ;
vdfp("c_Chem","a_Energy","ROW") = 19889.791015625 ;
vdfp("c_Chem","a_Textiles","USA") = 11629.3818359375 ;
vdfp("c_Chem","a_Textiles","EU_28") = 14559.74609375 ;
vdfp("c_Chem","a_Textiles","CHN") = 73375.2578125 ;
vdfp("c_Chem","a_Textiles","JPN") = 4130.2119140625 ;
vdfp("c_Chem","a_Textiles","IND") = 17931.591796875 ;
vdfp("c_Chem","a_Textiles","SSA") = 694.848754882812 ;
vdfp("c_Chem","a_Textiles","ROW") = 30276.513671875 ;
vdfp("c_Chem","a_Chem","USA") = 237819.96875 ;
vdfp("c_Chem","a_Chem","EU_28") = 163006.34375 ;
vdfp("c_Chem","a_Chem","CHN") = 696174.1875 ;
vdfp("c_Chem","a_Chem","JPN") = 139954.421875 ;
vdfp("c_Chem","a_Chem","IND") = 81579.2265625 ;
vdfp("c_Chem","a_Chem","SSA") = 6166.3564453125 ;
vdfp("c_Chem","a_Chem","ROW") = 329420.90625 ;
vdfp("c_Chem","a_Manuf","USA") = 124403.921875 ;
vdfp("c_Chem","a_Manuf","EU_28") = 131617.21875 ;
vdfp("c_Chem","a_Manuf","CHN") = 374392.9375 ;
vdfp("c_Chem","a_Manuf","JPN") = 70090.6953125 ;
vdfp("c_Chem","a_Manuf","IND") = 43093.21875 ;
vdfp("c_Chem","a_Manuf","SSA") = 5608.35986328125 ;
vdfp("c_Chem","a_Manuf","ROW") = 194708.71875 ;
vdfp("c_Chem","a_ForestFish","USA") = 2373.18725585938 ;
vdfp("c_Chem","a_ForestFish","EU_28") = 679.062927246094 ;
vdfp("c_Chem","a_ForestFish","CHN") = 3710.75634765625 ;
vdfp("c_Chem","a_ForestFish","JPN") = 450.74169921875 ;
vdfp("c_Chem","a_ForestFish","IND") = 135.080383300781 ;
vdfp("c_Chem","a_ForestFish","SSA") = 309.680328369141 ;
vdfp("c_Chem","a_ForestFish","ROW") = 3679.44140625 ;
vdfp("c_Chem","a_Svces","USA") = 224625.8125 ;
vdfp("c_Chem","a_Svces","EU_28") = 139860.34375 ;
vdfp("c_Chem","a_Svces","CHN") = 435300.8125 ;
vdfp("c_Chem","a_Svces","JPN") = 97407.09375 ;
vdfp("c_Chem","a_Svces","IND") = 55437.8125 ;
vdfp("c_Chem","a_Svces","SSA") = 8212.2646484375 ;
vdfp("c_Chem","a_Svces","ROW") = 258170.203125 ;
vdfp("c_Manuf","a_Rice","USA") = 347.211883544922 ;
vdfp("c_Manuf","a_Rice","EU_28") = 228.69157409668 ;
vdfp("c_Manuf","a_Rice","CHN") = 1607.69506835938 ;
vdfp("c_Manuf","a_Rice","JPN") = 213.842987060547 ;
vdfp("c_Manuf","a_Rice","IND") = 176.277526855469 ;
vdfp("c_Manuf","a_Rice","SSA") = 141.92448425293 ;
vdfp("c_Manuf","a_Rice","ROW") = 3422.28002929688 ;
vdfp("c_Manuf","a_Crops","USA") = 4090.84619140625 ;
vdfp("c_Manuf","a_Crops","EU_28") = 8347.68359375 ;
vdfp("c_Manuf","a_Crops","CHN") = 7528.412109375 ;
vdfp("c_Manuf","a_Crops","JPN") = 1707.92065429688 ;
vdfp("c_Manuf","a_Crops","IND") = 167.802734375 ;
vdfp("c_Manuf","a_Crops","SSA") = 1929.92761230469 ;
vdfp("c_Manuf","a_Crops","ROW") = 12797.6416015625 ;
vdfp("c_Manuf","a_Livestock","USA") = 2313.39794921875 ;
vdfp("c_Manuf","a_Livestock","EU_28") = 2467.23901367188 ;
vdfp("c_Manuf","a_Livestock","CHN") = 1845.73950195312 ;
vdfp("c_Manuf","a_Livestock","JPN") = 251.842956542969 ;
vdfp("c_Manuf","a_Livestock","IND") = 456.750610351562 ;
vdfp("c_Manuf","a_Livestock","SSA") = 342.499877929688 ;
vdfp("c_Manuf","a_Livestock","ROW") = 5916.12158203125 ;
vdfp("c_Manuf","a_FoodProc","USA") = 68248.2109375 ;
vdfp("c_Manuf","a_FoodProc","EU_28") = 52546.16796875 ;
vdfp("c_Manuf","a_FoodProc","CHN") = 45870.2265625 ;
vdfp("c_Manuf","a_FoodProc","JPN") = 12059.3046875 ;
vdfp("c_Manuf","a_FoodProc","IND") = 864.428771972656 ;
vdfp("c_Manuf","a_FoodProc","SSA") = 3456.56079101562 ;
vdfp("c_Manuf","a_FoodProc","ROW") = 63431.015625 ;
vdfp("c_Manuf","a_Energy","USA") = 29254.625 ;
vdfp("c_Manuf","a_Energy","EU_28") = 30770.3671875 ;
vdfp("c_Manuf","a_Energy","CHN") = 94750.765625 ;
vdfp("c_Manuf","a_Energy","JPN") = 2099.00952148438 ;
vdfp("c_Manuf","a_Energy","IND") = 31527.765625 ;
vdfp("c_Manuf","a_Energy","SSA") = 3058.158203125 ;
vdfp("c_Manuf","a_Energy","ROW") = 140644.890625 ;
vdfp("c_Manuf","a_Textiles","USA") = 3482.02661132812 ;
vdfp("c_Manuf","a_Textiles","EU_28") = 12593.7099609375 ;
vdfp("c_Manuf","a_Textiles","CHN") = 25037.3984375 ;
vdfp("c_Manuf","a_Textiles","JPN") = 742.497314453125 ;
vdfp("c_Manuf","a_Textiles","IND") = 3363.8603515625 ;
vdfp("c_Manuf","a_Textiles","SSA") = 1383.23095703125 ;
vdfp("c_Manuf","a_Textiles","ROW") = 21658.904296875 ;
vdfp("c_Manuf","a_Chem","USA") = 63295.12890625 ;
vdfp("c_Manuf","a_Chem","EU_28") = 57053.44140625 ;
vdfp("c_Manuf","a_Chem","CHN") = 101003.4296875 ;
vdfp("c_Manuf","a_Chem","JPN") = 13395.4794921875 ;
vdfp("c_Manuf","a_Chem","IND") = 8916.662109375 ;
vdfp("c_Manuf","a_Chem","SSA") = 2718.56298828125 ;
vdfp("c_Manuf","a_Chem","ROW") = 73176.65625 ;
vdfp("c_Manuf","a_Manuf","USA") = 912027.75 ;
vdfp("c_Manuf","a_Manuf","EU_28") = 1263117.125 ;
vdfp("c_Manuf","a_Manuf","CHN") = 3948662.75 ;
vdfp("c_Manuf","a_Manuf","JPN") = 899043.375 ;
vdfp("c_Manuf","a_Manuf","IND") = 175409.09375 ;
vdfp("c_Manuf","a_Manuf","SSA") = 81766.1484375 ;
vdfp("c_Manuf","a_Manuf","ROW") = 1702661.125 ;
vdfp("c_Manuf","a_ForestFish","USA") = 1090.13684082031 ;
vdfp("c_Manuf","a_ForestFish","EU_28") = 3485.44750976562 ;
vdfp("c_Manuf","a_ForestFish","CHN") = 6665.77685546875 ;
vdfp("c_Manuf","a_ForestFish","JPN") = 781.208312988281 ;
vdfp("c_Manuf","a_ForestFish","IND") = 2630.681640625 ;
vdfp("c_Manuf","a_ForestFish","SSA") = 896.509399414062 ;
vdfp("c_Manuf","a_ForestFish","ROW") = 6601.77587890625 ;
vdfp("c_Manuf","a_Svces","USA") = 829835.625 ;
vdfp("c_Manuf","a_Svces","EU_28") = 816566.9375 ;
vdfp("c_Manuf","a_Svces","CHN") = 1965628.125 ;
vdfp("c_Manuf","a_Svces","JPN") = 300955.4375 ;
vdfp("c_Manuf","a_Svces","IND") = 182007.59375 ;
vdfp("c_Manuf","a_Svces","SSA") = 68917.359375 ;
vdfp("c_Manuf","a_Svces","ROW") = 1229896.75 ;
vdfp("c_ForestFish","a_Rice","USA") = 171.583038330078 ;
vdfp("c_ForestFish","a_Rice","EU_28") = 14.2537574768066 ;
vdfp("c_ForestFish","a_Rice","CHN") = 3.36878252029419 ;
vdfp("c_ForestFish","a_Rice","JPN") = 1.84940230846405 ;
vdfp("c_ForestFish","a_Rice","IND") = 1.44641482830048 ;
vdfp("c_ForestFish","a_Rice","SSA") = 23.595874786377 ;
vdfp("c_ForestFish","a_Rice","ROW") = 443.417358398438 ;
vdfp("c_ForestFish","a_Crops","USA") = 19463.333984375 ;
vdfp("c_ForestFish","a_Crops","EU_28") = 119.893280029297 ;
vdfp("c_ForestFish","a_Crops","CHN") = 4.62349557876587 ;
vdfp("c_ForestFish","a_Crops","JPN") = 10.0801591873169 ;
vdfp("c_ForestFish","a_Crops","IND") = 1.87764060497284 ;
vdfp("c_ForestFish","a_Crops","SSA") = 151.775039672852 ;
vdfp("c_ForestFish","a_Crops","ROW") = 573.733032226562 ;
vdfp("c_ForestFish","a_Livestock","USA") = 3041.90625 ;
vdfp("c_ForestFish","a_Livestock","EU_28") = 106.731491088867 ;
vdfp("c_ForestFish","a_Livestock","CHN") = 2.71042633056641 ;
vdfp("c_ForestFish","a_Livestock","JPN") = 3.87472176551819 ;
vdfp("c_ForestFish","a_Livestock","IND") = 65.6384048461914 ;
vdfp("c_ForestFish","a_Livestock","SSA") = 298.635803222656 ;
vdfp("c_ForestFish","a_Livestock","ROW") = 794.174987792969 ;
vdfp("c_ForestFish","a_FoodProc","USA") = 2878.82739257812 ;
vdfp("c_ForestFish","a_FoodProc","EU_28") = 4831.7216796875 ;
vdfp("c_ForestFish","a_FoodProc","CHN") = 62096.30859375 ;
vdfp("c_ForestFish","a_FoodProc","JPN") = 7459.04296875 ;
vdfp("c_ForestFish","a_FoodProc","IND") = 5306.3349609375 ;
vdfp("c_ForestFish","a_FoodProc","SSA") = 3895.064453125 ;
vdfp("c_ForestFish","a_FoodProc","ROW") = 42134.30078125 ;
vdfp("c_ForestFish","a_Energy","USA") = 76.1057052612305 ;
vdfp("c_ForestFish","a_Energy","EU_28") = 500.636383056641 ;
vdfp("c_ForestFish","a_Energy","CHN") = 205.449264526367 ;
vdfp("c_ForestFish","a_Energy","JPN") = 1.23242092132568 ;
vdfp("c_ForestFish","a_Energy","IND") = 41.7148475646973 ;
vdfp("c_ForestFish","a_Energy","SSA") = 38.167106628418 ;
vdfp("c_ForestFish","a_Energy","ROW") = 963.072692871094 ;
vdfp("c_ForestFish","a_Textiles","USA") = 190.775909423828 ;
vdfp("c_ForestFish","a_Textiles","EU_28") = 458.001495361328 ;
vdfp("c_ForestFish","a_Textiles","CHN") = 399.409606933594 ;
vdfp("c_ForestFish","a_Textiles","JPN") = 9.16245460510254 ;
vdfp("c_ForestFish","a_Textiles","IND") = 52.9499397277832 ;
vdfp("c_ForestFish","a_Textiles","SSA") = 51.0486297607422 ;
vdfp("c_ForestFish","a_Textiles","ROW") = 367.691253662109 ;
vdfp("c_ForestFish","a_Chem","USA") = 3372.17578125 ;
vdfp("c_ForestFish","a_Chem","EU_28") = 749.614013671875 ;
vdfp("c_ForestFish","a_Chem","CHN") = 12960.8173828125 ;
vdfp("c_ForestFish","a_Chem","JPN") = 95.9938735961914 ;
vdfp("c_ForestFish","a_Chem","IND") = 5101.29345703125 ;
vdfp("c_ForestFish","a_Chem","SSA") = 249.613464355469 ;
vdfp("c_ForestFish","a_Chem","ROW") = 3004.54565429688 ;
vdfp("c_ForestFish","a_Manuf","USA") = 12471.7861328125 ;
vdfp("c_ForestFish","a_Manuf","EU_28") = 22699.88671875 ;
vdfp("c_ForestFish","a_Manuf","CHN") = 37732.01171875 ;
vdfp("c_ForestFish","a_Manuf","JPN") = 3980.4541015625 ;
vdfp("c_ForestFish","a_Manuf","IND") = 3100.67333984375 ;
vdfp("c_ForestFish","a_Manuf","SSA") = 5674.0947265625 ;
vdfp("c_ForestFish","a_Manuf","ROW") = 47183.70703125 ;
vdfp("c_ForestFish","a_ForestFish","USA") = 6347.384765625 ;
vdfp("c_ForestFish","a_ForestFish","EU_28") = 9855.419921875 ;
vdfp("c_ForestFish","a_ForestFish","CHN") = 8677.009765625 ;
vdfp("c_ForestFish","a_ForestFish","JPN") = 1168.15710449219 ;
vdfp("c_ForestFish","a_ForestFish","IND") = 777.71484375 ;
vdfp("c_ForestFish","a_ForestFish","SSA") = 2060.05834960938 ;
vdfp("c_ForestFish","a_ForestFish","ROW") = 26657.595703125 ;
vdfp("c_ForestFish","a_Svces","USA") = 4956.67822265625 ;
vdfp("c_ForestFish","a_Svces","EU_28") = 8409.6103515625 ;
vdfp("c_ForestFish","a_Svces","CHN") = 49844.1328125 ;
vdfp("c_ForestFish","a_Svces","JPN") = 3444.1630859375 ;
vdfp("c_ForestFish","a_Svces","IND") = 22045.6171875 ;
vdfp("c_ForestFish","a_Svces","SSA") = 7912.13623046875 ;
vdfp("c_ForestFish","a_Svces","ROW") = 35863.6015625 ;
vdfp("c_Svces","a_Rice","USA") = 1828.86376953125 ;
vdfp("c_Svces","a_Rice","EU_28") = 1736.00366210938 ;
vdfp("c_Svces","a_Rice","CHN") = 14377.8408203125 ;
vdfp("c_Svces","a_Rice","JPN") = 5760.71630859375 ;
vdfp("c_Svces","a_Rice","IND") = 7992.0126953125 ;
vdfp("c_Svces","a_Rice","SSA") = 1705.56237792969 ;
vdfp("c_Svces","a_Rice","ROW") = 44779.859375 ;
vdfp("c_Svces","a_Crops","USA") = 50171.10546875 ;
vdfp("c_Svces","a_Crops","EU_28") = 36714.21484375 ;
vdfp("c_Svces","a_Crops","CHN") = 52457.765625 ;
vdfp("c_Svces","a_Crops","JPN") = 7332.39306640625 ;
vdfp("c_Svces","a_Crops","IND") = 2526.67431640625 ;
vdfp("c_Svces","a_Crops","SSA") = 21226.4921875 ;
vdfp("c_Svces","a_Crops","ROW") = 104949.390625 ;
vdfp("c_Svces","a_Livestock","USA") = 40031.390625 ;
vdfp("c_Svces","a_Livestock","EU_28") = 20763.15625 ;
vdfp("c_Svces","a_Livestock","CHN") = 28143.859375 ;
vdfp("c_Svces","a_Livestock","JPN") = 7380.7666015625 ;
vdfp("c_Svces","a_Livestock","IND") = 12780.029296875 ;
vdfp("c_Svces","a_Livestock","SSA") = 6524.640625 ;
vdfp("c_Svces","a_Livestock","ROW") = 59071.9296875 ;
vdfp("c_Svces","a_FoodProc","USA") = 214156.453125 ;
vdfp("c_Svces","a_FoodProc","EU_28") = 306946.96875 ;
vdfp("c_Svces","a_FoodProc","CHN") = 246537.953125 ;
vdfp("c_Svces","a_FoodProc","JPN") = 56670.4921875 ;
vdfp("c_Svces","a_FoodProc","IND") = 38536.4921875 ;
vdfp("c_Svces","a_FoodProc","SSA") = 52074.56640625 ;
vdfp("c_Svces","a_FoodProc","ROW") = 370938 ;
vdfp("c_Svces","a_Energy","USA") = 172307.6875 ;
vdfp("c_Svces","a_Energy","EU_28") = 109715.9375 ;
vdfp("c_Svces","a_Energy","CHN") = 164887.4375 ;
vdfp("c_Svces","a_Energy","JPN") = 67553.8359375 ;
vdfp("c_Svces","a_Energy","IND") = 80364.21875 ;
vdfp("c_Svces","a_Energy","SSA") = 36849.9921875 ;
vdfp("c_Svces","a_Energy","ROW") = 427874.8125 ;
vdfp("c_Svces","a_Textiles","USA") = 18009.3828125 ;
vdfp("c_Svces","a_Textiles","EU_28") = 82959.25 ;
vdfp("c_Svces","a_Textiles","CHN") = 199742.859375 ;
vdfp("c_Svces","a_Textiles","JPN") = 7925.80078125 ;
vdfp("c_Svces","a_Textiles","IND") = 21731.22265625 ;
vdfp("c_Svces","a_Textiles","SSA") = 11939.0634765625 ;
vdfp("c_Svces","a_Textiles","ROW") = 123826.8828125 ;
vdfp("c_Svces","a_Chem","USA") = 175415.40625 ;
vdfp("c_Svces","a_Chem","EU_28") = 276942.28125 ;
vdfp("c_Svces","a_Chem","CHN") = 289229.1875 ;
vdfp("c_Svces","a_Chem","JPN") = 93379.6640625 ;
vdfp("c_Svces","a_Chem","IND") = 30726.63671875 ;
vdfp("c_Svces","a_Chem","SSA") = 11690.765625 ;
vdfp("c_Svces","a_Chem","ROW") = 261471.46875 ;
vdfp("c_Svces","a_Manuf","USA") = 718223.625 ;
vdfp("c_Svces","a_Manuf","EU_28") = 1094091 ;
vdfp("c_Svces","a_Manuf","CHN") = 1165707.75 ;
vdfp("c_Svces","a_Manuf","JPN") = 386164.1875 ;
vdfp("c_Svces","a_Manuf","IND") = 111359.96875 ;
vdfp("c_Svces","a_Manuf","SSA") = 98044.7578125 ;
vdfp("c_Svces","a_Manuf","ROW") = 1104553 ;
vdfp("c_Svces","a_ForestFish","USA") = 8054.90576171875 ;
vdfp("c_Svces","a_ForestFish","EU_28") = 11737.7822265625 ;
vdfp("c_Svces","a_ForestFish","CHN") = 35166.796875 ;
vdfp("c_Svces","a_ForestFish","JPN") = 2930.37768554688 ;
vdfp("c_Svces","a_ForestFish","IND") = 4363.24072265625 ;
vdfp("c_Svces","a_ForestFish","SSA") = 9054.0595703125 ;
vdfp("c_Svces","a_ForestFish","ROW") = 36582.79296875 ;
vdfp("c_Svces","a_Svces","USA") = 7352975 ;
vdfp("c_Svces","a_Svces","EU_28") = 6846185 ;
vdfp("c_Svces","a_Svces","CHN") = 4138268.75 ;
vdfp("c_Svces","a_Svces","JPN") = 1611724.625 ;
vdfp("c_Svces","a_Svces","IND") = 504249.375 ;
vdfp("c_Svces","a_Svces","SSA") = 390781.28125 ;
vdfp("c_Svces","a_Svces","ROW") = 6400212 ;
parameter vmfb(*,*,*) ;
vmfb("c_Rice","a_Rice","USA") = 60.488353729248 ;
vmfb("c_Rice","a_Rice","EU_28") = 181.233016967773 ;
vmfb("c_Rice","a_Rice","CHN") = 125.713882446289 ;
vmfb("c_Rice","a_Rice","JPN") = 3.85028290748596 ;
vmfb("c_Rice","a_Rice","IND") = 0.348086684942245 ;
vmfb("c_Rice","a_Rice","SSA") = 260.932098388672 ;
vmfb("c_Rice","a_Rice","ROW") = 1422.57153320312 ;
vmfb("c_Rice","a_Crops","USA") = 0.133996486663818 ;
vmfb("c_Rice","a_Crops","EU_28") = 91.1591415405273 ;
vmfb("c_Rice","a_Crops","CHN") = 9.44434833526611 ;
vmfb("c_Rice","a_Crops","JPN") = 0.0558797158300877 ;
vmfb("c_Rice","a_Crops","IND") = 0.0399402752518654 ;
vmfb("c_Rice","a_Crops","SSA") = 22.987678527832 ;
vmfb("c_Rice","a_Crops","ROW") = 124.726760864258 ;
vmfb("c_Rice","a_Livestock","USA") = 0.0612206980586052 ;
vmfb("c_Rice","a_Livestock","EU_28") = 3.33913660049438 ;
vmfb("c_Rice","a_Livestock","CHN") = 99.1478271484375 ;
vmfb("c_Rice","a_Livestock","JPN") = 0.490259766578674 ;
vmfb("c_Rice","a_Livestock","IND") = 0.0330982394516468 ;
vmfb("c_Rice","a_Livestock","SSA") = 122.436485290527 ;
vmfb("c_Rice","a_Livestock","ROW") = 155.714050292969 ;
vmfb("c_Rice","a_FoodProc","USA") = 708.473083496094 ;
vmfb("c_Rice","a_FoodProc","EU_28") = 1066.56079101562 ;
vmfb("c_Rice","a_FoodProc","CHN") = 922.983337402344 ;
vmfb("c_Rice","a_FoodProc","JPN") = 266.438873291016 ;
vmfb("c_Rice","a_FoodProc","IND") = 3.44449257850647 ;
vmfb("c_Rice","a_FoodProc","SSA") = 1969.23181152344 ;
vmfb("c_Rice","a_FoodProc","ROW") = 5990.7333984375 ;
vmfb("c_Rice","a_Energy","USA") = 0.00194822018966079 ;
vmfb("c_Rice","a_Energy","EU_28") = 0.296945452690125 ;
vmfb("c_Rice","a_Energy","CHN") = 0.0176203548908234 ;
vmfb("c_Rice","a_Energy","JPN") = 0.0326388590037823 ;
vmfb("c_Rice","a_Energy","IND") = 0.00112529774196446 ;
vmfb("c_Rice","a_Energy","SSA") = 2.87686705589294 ;
vmfb("c_Rice","a_Energy","ROW") = 6.68642854690552 ;
vmfb("c_Rice","a_Textiles","USA") = 0.00430318946018815 ;
vmfb("c_Rice","a_Textiles","EU_28") = 1.95507955551147 ;
vmfb("c_Rice","a_Textiles","CHN") = 1.52222263813019 ;
vmfb("c_Rice","a_Textiles","JPN") = 0.0488685667514801 ;
vmfb("c_Rice","a_Textiles","IND") = 0.00536741549149156 ;
vmfb("c_Rice","a_Textiles","SSA") = 8.37233638763428 ;
vmfb("c_Rice","a_Textiles","ROW") = 13.7165689468384 ;
vmfb("c_Rice","a_Chem","USA") = 0.026293171569705 ;
vmfb("c_Rice","a_Chem","EU_28") = 9.45028686523438 ;
vmfb("c_Rice","a_Chem","CHN") = 4.90109395980835 ;
vmfb("c_Rice","a_Chem","JPN") = 0.662290990352631 ;
vmfb("c_Rice","a_Chem","IND") = 0.0276185311377048 ;
vmfb("c_Rice","a_Chem","SSA") = 63.8190689086914 ;
vmfb("c_Rice","a_Chem","ROW") = 25.8919582366943 ;
vmfb("c_Rice","a_Manuf","USA") = 0.103057436645031 ;
vmfb("c_Rice","a_Manuf","EU_28") = 4.7180757522583 ;
vmfb("c_Rice","a_Manuf","CHN") = 3.037428855896 ;
vmfb("c_Rice","a_Manuf","JPN") = 0.703385531902313 ;
vmfb("c_Rice","a_Manuf","IND") = 0.0218414422124624 ;
vmfb("c_Rice","a_Manuf","SSA") = 23.0786190032959 ;
vmfb("c_Rice","a_Manuf","ROW") = 31.796085357666 ;
vmfb("c_Rice","a_ForestFish","USA") = 0.0140189779922366 ;
vmfb("c_Rice","a_ForestFish","EU_28") = 1.59496438503265 ;
vmfb("c_Rice","a_ForestFish","CHN") = 18.421329498291 ;
vmfb("c_Rice","a_ForestFish","JPN") = 0.878665089607239 ;
vmfb("c_Rice","a_ForestFish","IND") = 0.0197734832763672 ;
vmfb("c_Rice","a_ForestFish","SSA") = 2.55356168746948 ;
vmfb("c_Rice","a_ForestFish","ROW") = 31.465648651123 ;
vmfb("c_Rice","a_Svces","USA") = 3.39589548110962 ;
vmfb("c_Rice","a_Svces","EU_28") = 301.61328125 ;
vmfb("c_Rice","a_Svces","CHN") = 168.088958740234 ;
vmfb("c_Rice","a_Svces","JPN") = 249.950439453125 ;
vmfb("c_Rice","a_Svces","IND") = 0.224182173609734 ;
vmfb("c_Rice","a_Svces","SSA") = 459.124664306641 ;
vmfb("c_Rice","a_Svces","ROW") = 1363.94445800781 ;
vmfb("c_Crops","a_Rice","USA") = 14.8671016693115 ;
vmfb("c_Crops","a_Rice","EU_28") = 39.6473121643066 ;
vmfb("c_Crops","a_Rice","CHN") = 3128.84887695312 ;
vmfb("c_Crops","a_Rice","JPN") = 144.375030517578 ;
vmfb("c_Crops","a_Rice","IND") = 7.23537874221802 ;
vmfb("c_Crops","a_Rice","SSA") = 154.596267700195 ;
vmfb("c_Crops","a_Rice","ROW") = 3218.12084960938 ;
vmfb("c_Crops","a_Crops","USA") = 1907.3525390625 ;
vmfb("c_Crops","a_Crops","EU_28") = 4965.337890625 ;
vmfb("c_Crops","a_Crops","CHN") = 3718.8994140625 ;
vmfb("c_Crops","a_Crops","JPN") = 297.256408691406 ;
vmfb("c_Crops","a_Crops","IND") = 505.223114013672 ;
vmfb("c_Crops","a_Crops","SSA") = 493.078186035156 ;
vmfb("c_Crops","a_Crops","ROW") = 9576.2880859375 ;
vmfb("c_Crops","a_Livestock","USA") = 369.768035888672 ;
vmfb("c_Crops","a_Livestock","EU_28") = 4828.63916015625 ;
vmfb("c_Crops","a_Livestock","CHN") = 2128.427734375 ;
vmfb("c_Crops","a_Livestock","JPN") = 921.911865234375 ;
vmfb("c_Crops","a_Livestock","IND") = 979.441650390625 ;
vmfb("c_Crops","a_Livestock","SSA") = 820.30322265625 ;
vmfb("c_Crops","a_Livestock","ROW") = 15587.3642578125 ;
vmfb("c_Crops","a_FoodProc","USA") = 14799.4384765625 ;
vmfb("c_Crops","a_FoodProc","EU_28") = 58078.31640625 ;
vmfb("c_Crops","a_FoodProc","CHN") = 29914.0390625 ;
vmfb("c_Crops","a_FoodProc","JPN") = 11023.1552734375 ;
vmfb("c_Crops","a_FoodProc","IND") = 1720.33850097656 ;
vmfb("c_Crops","a_FoodProc","SSA") = 2489.54565429688 ;
vmfb("c_Crops","a_FoodProc","ROW") = 68503.859375 ;
vmfb("c_Crops","a_Energy","USA") = 3.46315836906433 ;
vmfb("c_Crops","a_Energy","EU_28") = 51.9812164306641 ;
vmfb("c_Crops","a_Energy","CHN") = 4.49749326705933 ;
vmfb("c_Crops","a_Energy","JPN") = 0.292086392641068 ;
vmfb("c_Crops","a_Energy","IND") = 27.3098125457764 ;
vmfb("c_Crops","a_Energy","SSA") = 8.35230350494385 ;
vmfb("c_Crops","a_Energy","ROW") = 98.3988571166992 ;
vmfb("c_Crops","a_Textiles","USA") = 29.0944080352783 ;
vmfb("c_Crops","a_Textiles","EU_28") = 391.110198974609 ;
vmfb("c_Crops","a_Textiles","CHN") = 3253.8359375 ;
vmfb("c_Crops","a_Textiles","JPN") = 107.019805908203 ;
vmfb("c_Crops","a_Textiles","IND") = 303.776336669922 ;
vmfb("c_Crops","a_Textiles","SSA") = 40.2259368896484 ;
vmfb("c_Crops","a_Textiles","ROW") = 6456.24755859375 ;
vmfb("c_Crops","a_Chem","USA") = 826.702209472656 ;
vmfb("c_Crops","a_Chem","EU_28") = 2395.74096679688 ;
vmfb("c_Crops","a_Chem","CHN") = 1852.71350097656 ;
vmfb("c_Crops","a_Chem","JPN") = 953.280822753906 ;
vmfb("c_Crops","a_Chem","IND") = 176.840072631836 ;
vmfb("c_Crops","a_Chem","SSA") = 30.1317348480225 ;
vmfb("c_Crops","a_Chem","ROW") = 3829.71166992188 ;
vmfb("c_Crops","a_Manuf","USA") = 195.782836914062 ;
vmfb("c_Crops","a_Manuf","EU_28") = 486.639343261719 ;
vmfb("c_Crops","a_Manuf","CHN") = 989.57470703125 ;
vmfb("c_Crops","a_Manuf","JPN") = 30.4715328216553 ;
vmfb("c_Crops","a_Manuf","IND") = 13.594220161438 ;
vmfb("c_Crops","a_Manuf","SSA") = 50.4525032043457 ;
vmfb("c_Crops","a_Manuf","ROW") = 879.071411132812 ;
vmfb("c_Crops","a_ForestFish","USA") = 116.450141906738 ;
vmfb("c_Crops","a_ForestFish","EU_28") = 283.867034912109 ;
vmfb("c_Crops","a_ForestFish","CHN") = 71.0492477416992 ;
vmfb("c_Crops","a_ForestFish","JPN") = 0.775204956531525 ;
vmfb("c_Crops","a_ForestFish","IND") = 13.1334476470947 ;
vmfb("c_Crops","a_ForestFish","SSA") = 41.4660034179688 ;
vmfb("c_Crops","a_ForestFish","ROW") = 221.449264526367 ;
vmfb("c_Crops","a_Svces","USA") = 2607.00561523438 ;
vmfb("c_Crops","a_Svces","EU_28") = 10316.3203125 ;
vmfb("c_Crops","a_Svces","CHN") = 3732.89892578125 ;
vmfb("c_Crops","a_Svces","JPN") = 1893.07409667969 ;
vmfb("c_Crops","a_Svces","IND") = 894.700073242188 ;
vmfb("c_Crops","a_Svces","SSA") = 427.580078125 ;
vmfb("c_Crops","a_Svces","ROW") = 19270.517578125 ;
vmfb("c_Livestock","a_Rice","USA") = 1.11563348770142 ;
vmfb("c_Livestock","a_Rice","EU_28") = 12.9937467575073 ;
vmfb("c_Livestock","a_Rice","CHN") = 1.13283622264862 ;
vmfb("c_Livestock","a_Rice","JPN") = 2.24063849449158 ;
vmfb("c_Livestock","a_Rice","IND") = 5.50952959060669 ;
vmfb("c_Livestock","a_Rice","SSA") = 1.79448926448822 ;
vmfb("c_Livestock","a_Rice","ROW") = 109.155364990234 ;
vmfb("c_Livestock","a_Crops","USA") = 9.31122779846191 ;
vmfb("c_Livestock","a_Crops","EU_28") = 191.049270629883 ;
vmfb("c_Livestock","a_Crops","CHN") = 1.63180112838745 ;
vmfb("c_Livestock","a_Crops","JPN") = 2.6538999080658 ;
vmfb("c_Livestock","a_Crops","IND") = 11.5005741119385 ;
vmfb("c_Livestock","a_Crops","SSA") = 2.05946516990662 ;
vmfb("c_Livestock","a_Crops","ROW") = 112.549377441406 ;
vmfb("c_Livestock","a_Livestock","USA") = 651.347106933594 ;
vmfb("c_Livestock","a_Livestock","EU_28") = 176.322021484375 ;
vmfb("c_Livestock","a_Livestock","CHN") = 615.432861328125 ;
vmfb("c_Livestock","a_Livestock","JPN") = 72.5097503662109 ;
vmfb("c_Livestock","a_Livestock","IND") = 3.35604286193848 ;
vmfb("c_Livestock","a_Livestock","SSA") = 77.8703155517578 ;
vmfb("c_Livestock","a_Livestock","ROW") = 1651.46325683594 ;
vmfb("c_Livestock","a_FoodProc","USA") = 2975.09716796875 ;
vmfb("c_Livestock","a_FoodProc","EU_28") = 15087.33203125 ;
vmfb("c_Livestock","a_FoodProc","CHN") = 3462.96435546875 ;
vmfb("c_Livestock","a_FoodProc","JPN") = 284.763458251953 ;
vmfb("c_Livestock","a_FoodProc","IND") = 33.4645385742188 ;
vmfb("c_Livestock","a_FoodProc","SSA") = 403.810028076172 ;
vmfb("c_Livestock","a_FoodProc","ROW") = 7470.82421875 ;
vmfb("c_Livestock","a_Energy","USA") = 0.0185928530991077 ;
vmfb("c_Livestock","a_Energy","EU_28") = 19.3517971038818 ;
vmfb("c_Livestock","a_Energy","CHN") = 0.122600041329861 ;
vmfb("c_Livestock","a_Energy","JPN") = 0.106916300952435 ;
vmfb("c_Livestock","a_Energy","IND") = 0.445617526769638 ;
vmfb("c_Livestock","a_Energy","SSA") = 0.647187888622284 ;
vmfb("c_Livestock","a_Energy","ROW") = 3.17826128005981 ;
vmfb("c_Livestock","a_Textiles","USA") = 1.21642076969147 ;
vmfb("c_Livestock","a_Textiles","EU_28") = 156.185562133789 ;
vmfb("c_Livestock","a_Textiles","CHN") = 1106.17126464844 ;
vmfb("c_Livestock","a_Textiles","JPN") = 5.09462213516235 ;
vmfb("c_Livestock","a_Textiles","IND") = 24.0176162719727 ;
vmfb("c_Livestock","a_Textiles","SSA") = 24.7626171112061 ;
vmfb("c_Livestock","a_Textiles","ROW") = 991.87646484375 ;
vmfb("c_Livestock","a_Chem","USA") = 2.87085032463074 ;
vmfb("c_Livestock","a_Chem","EU_28") = 771.867004394531 ;
vmfb("c_Livestock","a_Chem","CHN") = 633.062683105469 ;
vmfb("c_Livestock","a_Chem","JPN") = 3.6030375957489 ;
vmfb("c_Livestock","a_Chem","IND") = 51.0764427185059 ;
vmfb("c_Livestock","a_Chem","SSA") = 0.506852746009827 ;
vmfb("c_Livestock","a_Chem","ROW") = 96.5513687133789 ;
vmfb("c_Livestock","a_Manuf","USA") = 1.13471531867981 ;
vmfb("c_Livestock","a_Manuf","EU_28") = 42.6373710632324 ;
vmfb("c_Livestock","a_Manuf","CHN") = 81.185173034668 ;
vmfb("c_Livestock","a_Manuf","JPN") = 2.02537655830383 ;
vmfb("c_Livestock","a_Manuf","IND") = 0.703625977039337 ;
vmfb("c_Livestock","a_Manuf","SSA") = 4.03521299362183 ;
vmfb("c_Livestock","a_Manuf","ROW") = 27.4495544433594 ;
vmfb("c_Livestock","a_ForestFish","USA") = 17.4906806945801 ;
vmfb("c_Livestock","a_ForestFish","EU_28") = 19.7044696807861 ;
vmfb("c_Livestock","a_ForestFish","CHN") = 0.146197617053986 ;
vmfb("c_Livestock","a_ForestFish","JPN") = 0.959405481815338 ;
vmfb("c_Livestock","a_ForestFish","IND") = 0.0649078711867332 ;
vmfb("c_Livestock","a_ForestFish","SSA") = 1.22685420513153 ;
vmfb("c_Livestock","a_ForestFish","ROW") = 13.1730976104736 ;
vmfb("c_Livestock","a_Svces","USA") = 135.317245483398 ;
vmfb("c_Livestock","a_Svces","EU_28") = 1563.46923828125 ;
vmfb("c_Livestock","a_Svces","CHN") = 374.236175537109 ;
vmfb("c_Livestock","a_Svces","JPN") = 123.676582336426 ;
vmfb("c_Livestock","a_Svces","IND") = 102.704895019531 ;
vmfb("c_Livestock","a_Svces","SSA") = 35.6449813842773 ;
vmfb("c_Livestock","a_Svces","ROW") = 2102.75830078125 ;
vmfb("c_FoodProc","a_Rice","USA") = 3577.64428710938 ;
vmfb("c_FoodProc","a_Rice","EU_28") = 87.2424011230469 ;
vmfb("c_FoodProc","a_Rice","CHN") = 15.0512180328369 ;
vmfb("c_FoodProc","a_Rice","JPN") = 2.20097804069519 ;
vmfb("c_FoodProc","a_Rice","IND") = 2.5545289516449 ;
vmfb("c_FoodProc","a_Rice","SSA") = 32.932918548584 ;
vmfb("c_FoodProc","a_Rice","ROW") = 735.197204589844 ;
vmfb("c_FoodProc","a_Crops","USA") = 22.7530422210693 ;
vmfb("c_FoodProc","a_Crops","EU_28") = 1011.85772705078 ;
vmfb("c_FoodProc","a_Crops","CHN") = 110.480743408203 ;
vmfb("c_FoodProc","a_Crops","JPN") = 25.3950347900391 ;
vmfb("c_FoodProc","a_Crops","IND") = 13.8360080718994 ;
vmfb("c_FoodProc","a_Crops","SSA") = 1033.22888183594 ;
vmfb("c_FoodProc","a_Crops","ROW") = 3840.85571289062 ;
vmfb("c_FoodProc","a_Livestock","USA") = 894.259948730469 ;
vmfb("c_FoodProc","a_Livestock","EU_28") = 7116.1826171875 ;
vmfb("c_FoodProc","a_Livestock","CHN") = 2408.76635742188 ;
vmfb("c_FoodProc","a_Livestock","JPN") = 1839.64404296875 ;
vmfb("c_FoodProc","a_Livestock","IND") = 1085.6298828125 ;
vmfb("c_FoodProc","a_Livestock","SSA") = 1038.53552246094 ;
vmfb("c_FoodProc","a_Livestock","ROW") = 12510.7421875 ;
vmfb("c_FoodProc","a_FoodProc","USA") = 17245.517578125 ;
vmfb("c_FoodProc","a_FoodProc","EU_28") = 89129.3046875 ;
vmfb("c_FoodProc","a_FoodProc","CHN") = 16577.806640625 ;
vmfb("c_FoodProc","a_FoodProc","JPN") = 10843.8115234375 ;
vmfb("c_FoodProc","a_FoodProc","IND") = 1591.76000976562 ;
vmfb("c_FoodProc","a_FoodProc","SSA") = 6482.4345703125 ;
vmfb("c_FoodProc","a_FoodProc","ROW") = 82721.0859375 ;
vmfb("c_FoodProc","a_Energy","USA") = 13.4710931777954 ;
vmfb("c_FoodProc","a_Energy","EU_28") = 184.1845703125 ;
vmfb("c_FoodProc","a_Energy","CHN") = 131.531265258789 ;
vmfb("c_FoodProc","a_Energy","JPN") = 0.732357680797577 ;
vmfb("c_FoodProc","a_Energy","IND") = 9.87807273864746 ;
vmfb("c_FoodProc","a_Energy","SSA") = 30.5361328125 ;
vmfb("c_FoodProc","a_Energy","ROW") = 300.129272460938 ;
vmfb("c_FoodProc","a_Textiles","USA") = 179.506973266602 ;
vmfb("c_FoodProc","a_Textiles","EU_28") = 2206.55322265625 ;
vmfb("c_FoodProc","a_Textiles","CHN") = 1328.84936523438 ;
vmfb("c_FoodProc","a_Textiles","JPN") = 121.857215881348 ;
vmfb("c_FoodProc","a_Textiles","IND") = 2.43000340461731 ;
vmfb("c_FoodProc","a_Textiles","SSA") = 84.5390548706055 ;
vmfb("c_FoodProc","a_Textiles","ROW") = 2063.21484375 ;
vmfb("c_FoodProc","a_Chem","USA") = 350.976715087891 ;
vmfb("c_FoodProc","a_Chem","EU_28") = 14150.2041015625 ;
vmfb("c_FoodProc","a_Chem","CHN") = 1767.00463867188 ;
vmfb("c_FoodProc","a_Chem","JPN") = 372.339538574219 ;
vmfb("c_FoodProc","a_Chem","IND") = 184.961166381836 ;
vmfb("c_FoodProc","a_Chem","SSA") = 99.2404937744141 ;
vmfb("c_FoodProc","a_Chem","ROW") = 2874.47045898438 ;
vmfb("c_FoodProc","a_Manuf","USA") = 264.915405273438 ;
vmfb("c_FoodProc","a_Manuf","EU_28") = 1362.17419433594 ;
vmfb("c_FoodProc","a_Manuf","CHN") = 857.664794921875 ;
vmfb("c_FoodProc","a_Manuf","JPN") = 78.2715759277344 ;
vmfb("c_FoodProc","a_Manuf","IND") = 2.06767702102661 ;
vmfb("c_FoodProc","a_Manuf","SSA") = 131.073944091797 ;
vmfb("c_FoodProc","a_Manuf","ROW") = 907.594055175781 ;
vmfb("c_FoodProc","a_ForestFish","USA") = 24.9154224395752 ;
vmfb("c_FoodProc","a_ForestFish","EU_28") = 283.853302001953 ;
vmfb("c_FoodProc","a_ForestFish","CHN") = 795.621154785156 ;
vmfb("c_FoodProc","a_ForestFish","JPN") = 182.644515991211 ;
vmfb("c_FoodProc","a_ForestFish","IND") = 2.35672760009766 ;
vmfb("c_FoodProc","a_ForestFish","SSA") = 42.9449996948242 ;
vmfb("c_FoodProc","a_ForestFish","ROW") = 4709.26318359375 ;
vmfb("c_FoodProc","a_Svces","USA") = 19399.236328125 ;
vmfb("c_FoodProc","a_Svces","EU_28") = 58128.1171875 ;
vmfb("c_FoodProc","a_Svces","CHN") = 8523.88671875 ;
vmfb("c_FoodProc","a_Svces","JPN") = 13927.9267578125 ;
vmfb("c_FoodProc","a_Svces","IND") = 488.976715087891 ;
vmfb("c_FoodProc","a_Svces","SSA") = 4645.45556640625 ;
vmfb("c_FoodProc","a_Svces","ROW") = 65661.4453125 ;
vmfb("c_Energy","a_Rice","USA") = 4.17434024810791 ;
vmfb("c_Energy","a_Rice","EU_28") = 47.7763328552246 ;
vmfb("c_Energy","a_Rice","CHN") = 101.630363464355 ;
vmfb("c_Energy","a_Rice","JPN") = 29.1375102996826 ;
vmfb("c_Energy","a_Rice","IND") = 175.042617797852 ;
vmfb("c_Energy","a_Rice","SSA") = 10.2583045959473 ;
vmfb("c_Energy","a_Rice","ROW") = 644.453063964844 ;
vmfb("c_Energy","a_Crops","USA") = 517.020263671875 ;
vmfb("c_Energy","a_Crops","EU_28") = 1631.73461914062 ;
vmfb("c_Energy","a_Crops","CHN") = 578.333374023438 ;
vmfb("c_Energy","a_Crops","JPN") = 110.93440246582 ;
vmfb("c_Energy","a_Crops","IND") = 278.78466796875 ;
vmfb("c_Energy","a_Crops","SSA") = 308.640258789062 ;
vmfb("c_Energy","a_Crops","ROW") = 3801.13623046875 ;
vmfb("c_Energy","a_Livestock","USA") = 323.711212158203 ;
vmfb("c_Energy","a_Livestock","EU_28") = 426.748504638672 ;
vmfb("c_Energy","a_Livestock","CHN") = 138.708511352539 ;
vmfb("c_Energy","a_Livestock","JPN") = 14.1083183288574 ;
vmfb("c_Energy","a_Livestock","IND") = 16.8797779083252 ;
vmfb("c_Energy","a_Livestock","SSA") = 113.841133117676 ;
vmfb("c_Energy","a_Livestock","ROW") = 800.8017578125 ;
vmfb("c_Energy","a_FoodProc","USA") = 186.276657104492 ;
vmfb("c_Energy","a_FoodProc","EU_28") = 4631.5576171875 ;
vmfb("c_Energy","a_FoodProc","CHN") = 759.36083984375 ;
vmfb("c_Energy","a_FoodProc","JPN") = 896.783935546875 ;
vmfb("c_Energy","a_FoodProc","IND") = 278.287689208984 ;
vmfb("c_Energy","a_FoodProc","SSA") = 299.186492919922 ;
vmfb("c_Energy","a_FoodProc","ROW") = 3821.38330078125 ;
vmfb("c_Energy","a_Energy","USA") = 156432.4375 ;
vmfb("c_Energy","a_Energy","EU_28") = 313315 ;
vmfb("c_Energy","a_Energy","CHN") = 189651.90625 ;
vmfb("c_Energy","a_Energy","JPN") = 120324.7421875 ;
vmfb("c_Energy","a_Energy","IND") = 106756.0078125 ;
vmfb("c_Energy","a_Energy","SSA") = 14069.4345703125 ;
vmfb("c_Energy","a_Energy","ROW") = 375067.71875 ;
vmfb("c_Energy","a_Textiles","USA") = 27.7753925323486 ;
vmfb("c_Energy","a_Textiles","EU_28") = 824.479919433594 ;
vmfb("c_Energy","a_Textiles","CHN") = 584.057800292969 ;
vmfb("c_Energy","a_Textiles","JPN") = 99.0807723999023 ;
vmfb("c_Energy","a_Textiles","IND") = 111.172782897949 ;
vmfb("c_Energy","a_Textiles","SSA") = 73.8799591064453 ;
vmfb("c_Energy","a_Textiles","ROW") = 1479.4560546875 ;
vmfb("c_Energy","a_Chem","USA") = 7179.572265625 ;
vmfb("c_Energy","a_Chem","EU_28") = 24345.529296875 ;
vmfb("c_Energy","a_Chem","CHN") = 8757.9306640625 ;
vmfb("c_Energy","a_Chem","JPN") = 5373.0712890625 ;
vmfb("c_Energy","a_Chem","IND") = 4862.7470703125 ;
vmfb("c_Energy","a_Chem","SSA") = 961.520629882812 ;
vmfb("c_Energy","a_Chem","ROW") = 32642.849609375 ;
vmfb("c_Energy","a_Manuf","USA") = 1322.34423828125 ;
vmfb("c_Energy","a_Manuf","EU_28") = 21668.095703125 ;
vmfb("c_Energy","a_Manuf","CHN") = 14264.345703125 ;
vmfb("c_Energy","a_Manuf","JPN") = 6099.98779296875 ;
vmfb("c_Energy","a_Manuf","IND") = 5850.583984375 ;
vmfb("c_Energy","a_Manuf","SSA") = 3308.20043945312 ;
vmfb("c_Energy","a_Manuf","ROW") = 34183.4609375 ;
vmfb("c_Energy","a_ForestFish","USA") = 123.262359619141 ;
vmfb("c_Energy","a_ForestFish","EU_28") = 548.381408691406 ;
vmfb("c_Energy","a_ForestFish","CHN") = 491.751403808594 ;
vmfb("c_Energy","a_ForestFish","JPN") = 204.793548583984 ;
vmfb("c_Energy","a_ForestFish","IND") = 82.3845977783203 ;
vmfb("c_Energy","a_ForestFish","SSA") = 358.787689208984 ;
vmfb("c_Energy","a_ForestFish","ROW") = 2479.13232421875 ;
vmfb("c_Energy","a_Svces","USA") = 32753.537109375 ;
vmfb("c_Energy","a_Svces","EU_28") = 90007.8828125 ;
vmfb("c_Energy","a_Svces","CHN") = 11842.7568359375 ;
vmfb("c_Energy","a_Svces","JPN") = 8412.6533203125 ;
vmfb("c_Energy","a_Svces","IND") = 5389.90380859375 ;
vmfb("c_Energy","a_Svces","SSA") = 22397.462890625 ;
vmfb("c_Energy","a_Svces","ROW") = 141750.046875 ;
vmfb("c_Textiles","a_Rice","USA") = 7.47215032577515 ;
vmfb("c_Textiles","a_Rice","EU_28") = 14.5686388015747 ;
vmfb("c_Textiles","a_Rice","CHN") = 18.5399932861328 ;
vmfb("c_Textiles","a_Rice","JPN") = 50.1189804077148 ;
vmfb("c_Textiles","a_Rice","IND") = 3.10188674926758 ;
vmfb("c_Textiles","a_Rice","SSA") = 17.3127536773682 ;
vmfb("c_Textiles","a_Rice","ROW") = 253.309829711914 ;
vmfb("c_Textiles","a_Crops","USA") = 95.7868270874023 ;
vmfb("c_Textiles","a_Crops","EU_28") = 388.195068359375 ;
vmfb("c_Textiles","a_Crops","CHN") = 7.61833763122559 ;
vmfb("c_Textiles","a_Crops","JPN") = 80.4803695678711 ;
vmfb("c_Textiles","a_Crops","IND") = 3.96673679351807 ;
vmfb("c_Textiles","a_Crops","SSA") = 510.858581542969 ;
vmfb("c_Textiles","a_Crops","ROW") = 1217.37512207031 ;
vmfb("c_Textiles","a_Livestock","USA") = 30.3583889007568 ;
vmfb("c_Textiles","a_Livestock","EU_28") = 179.280670166016 ;
vmfb("c_Textiles","a_Livestock","CHN") = 1.67798972129822 ;
vmfb("c_Textiles","a_Livestock","JPN") = 14.8998517990112 ;
vmfb("c_Textiles","a_Livestock","IND") = 2.07907652854919 ;
vmfb("c_Textiles","a_Livestock","SSA") = 104.192970275879 ;
vmfb("c_Textiles","a_Livestock","ROW") = 264.384857177734 ;
vmfb("c_Textiles","a_FoodProc","USA") = 166.650573730469 ;
vmfb("c_Textiles","a_FoodProc","EU_28") = 1452.41174316406 ;
vmfb("c_Textiles","a_FoodProc","CHN") = 269.731323242188 ;
vmfb("c_Textiles","a_FoodProc","JPN") = 232.478790283203 ;
vmfb("c_Textiles","a_FoodProc","IND") = 9.28589916229248 ;
vmfb("c_Textiles","a_FoodProc","SSA") = 252.528137207031 ;
vmfb("c_Textiles","a_FoodProc","ROW") = 2017.83666992188 ;
vmfb("c_Textiles","a_Energy","USA") = 66.4999313354492 ;
vmfb("c_Textiles","a_Energy","EU_28") = 317.516510009766 ;
vmfb("c_Textiles","a_Energy","CHN") = 120.128768920898 ;
vmfb("c_Textiles","a_Energy","JPN") = 42.1120109558105 ;
vmfb("c_Textiles","a_Energy","IND") = 118.264144897461 ;
vmfb("c_Textiles","a_Energy","SSA") = 378.273559570312 ;
vmfb("c_Textiles","a_Energy","ROW") = 1492.099609375 ;
vmfb("c_Textiles","a_Textiles","USA") = 6547.404296875 ;
vmfb("c_Textiles","a_Textiles","EU_28") = 74044.859375 ;
vmfb("c_Textiles","a_Textiles","CHN") = 28392.796875 ;
vmfb("c_Textiles","a_Textiles","JPN") = 6348.95751953125 ;
vmfb("c_Textiles","a_Textiles","IND") = 2445.53930664062 ;
vmfb("c_Textiles","a_Textiles","SSA") = 4066.48266601562 ;
vmfb("c_Textiles","a_Textiles","ROW") = 109542.8359375 ;
vmfb("c_Textiles","a_Chem","USA") = 1469.48950195312 ;
vmfb("c_Textiles","a_Chem","EU_28") = 10471.185546875 ;
vmfb("c_Textiles","a_Chem","CHN") = 769.729370117188 ;
vmfb("c_Textiles","a_Chem","JPN") = 618.900695800781 ;
vmfb("c_Textiles","a_Chem","IND") = 373.911926269531 ;
vmfb("c_Textiles","a_Chem","SSA") = 163.849670410156 ;
vmfb("c_Textiles","a_Chem","ROW") = 4682.44287109375 ;
vmfb("c_Textiles","a_Manuf","USA") = 7608.0166015625 ;
vmfb("c_Textiles","a_Manuf","EU_28") = 20032.59375 ;
vmfb("c_Textiles","a_Manuf","CHN") = 3342.00952148438 ;
vmfb("c_Textiles","a_Manuf","JPN") = 3352.32690429688 ;
vmfb("c_Textiles","a_Manuf","IND") = 121.838638305664 ;
vmfb("c_Textiles","a_Manuf","SSA") = 789.554870605469 ;
vmfb("c_Textiles","a_Manuf","ROW") = 17834.7265625 ;
vmfb("c_Textiles","a_ForestFish","USA") = 58.4146919250488 ;
vmfb("c_Textiles","a_ForestFish","EU_28") = 444.688171386719 ;
vmfb("c_Textiles","a_ForestFish","CHN") = 60.609001159668 ;
vmfb("c_Textiles","a_ForestFish","JPN") = 174.032073974609 ;
vmfb("c_Textiles","a_ForestFish","IND") = 17.7654266357422 ;
vmfb("c_Textiles","a_ForestFish","SSA") = 378.994995117188 ;
vmfb("c_Textiles","a_ForestFish","ROW") = 927.879699707031 ;
vmfb("c_Textiles","a_Svces","USA") = 18679.19140625 ;
vmfb("c_Textiles","a_Svces","EU_28") = 31754.724609375 ;
vmfb("c_Textiles","a_Svces","CHN") = 3446.98266601562 ;
vmfb("c_Textiles","a_Svces","JPN") = 10723.4326171875 ;
vmfb("c_Textiles","a_Svces","IND") = 545.988525390625 ;
vmfb("c_Textiles","a_Svces","SSA") = 3467.42749023438 ;
vmfb("c_Textiles","a_Svces","ROW") = 35142.94921875 ;
vmfb("c_Chem","a_Rice","USA") = 75.5531539916992 ;
vmfb("c_Chem","a_Rice","EU_28") = 162.081588745117 ;
vmfb("c_Chem","a_Rice","CHN") = 2326.69848632812 ;
vmfb("c_Chem","a_Rice","JPN") = 337.496368408203 ;
vmfb("c_Chem","a_Rice","IND") = 1231.19458007812 ;
vmfb("c_Chem","a_Rice","SSA") = 222.466995239258 ;
vmfb("c_Chem","a_Rice","ROW") = 6374.95751953125 ;
vmfb("c_Chem","a_Crops","USA") = 5557.20654296875 ;
vmfb("c_Chem","a_Crops","EU_28") = 8050.1064453125 ;
vmfb("c_Chem","a_Crops","CHN") = 15111.5478515625 ;
vmfb("c_Chem","a_Crops","JPN") = 710.775146484375 ;
vmfb("c_Chem","a_Crops","IND") = 1753.30419921875 ;
vmfb("c_Chem","a_Crops","SSA") = 5410.02392578125 ;
vmfb("c_Chem","a_Crops","ROW") = 29204.302734375 ;
vmfb("c_Chem","a_Livestock","USA") = 525.492919921875 ;
vmfb("c_Chem","a_Livestock","EU_28") = 3789.0029296875 ;
vmfb("c_Chem","a_Livestock","CHN") = 178.132019042969 ;
vmfb("c_Chem","a_Livestock","JPN") = 151.059448242188 ;
vmfb("c_Chem","a_Livestock","IND") = 21.1291675567627 ;
vmfb("c_Chem","a_Livestock","SSA") = 519.564636230469 ;
vmfb("c_Chem","a_Livestock","ROW") = 4911.5224609375 ;
vmfb("c_Chem","a_FoodProc","USA") = 6570.87646484375 ;
vmfb("c_Chem","a_FoodProc","EU_28") = 25125.951171875 ;
vmfb("c_Chem","a_FoodProc","CHN") = 6829.3564453125 ;
vmfb("c_Chem","a_FoodProc","JPN") = 1652.791015625 ;
vmfb("c_Chem","a_FoodProc","IND") = 1098.2587890625 ;
vmfb("c_Chem","a_FoodProc","SSA") = 1759.37341308594 ;
vmfb("c_Chem","a_FoodProc","ROW") = 28097.828125 ;
vmfb("c_Chem","a_Energy","USA") = 2708.97900390625 ;
vmfb("c_Chem","a_Energy","EU_28") = 7126.275390625 ;
vmfb("c_Chem","a_Energy","CHN") = 2711.92602539062 ;
vmfb("c_Chem","a_Energy","JPN") = 230.984786987305 ;
vmfb("c_Chem","a_Energy","IND") = 4543.68310546875 ;
vmfb("c_Chem","a_Energy","SSA") = 1246.63500976562 ;
vmfb("c_Chem","a_Energy","ROW") = 15225.4423828125 ;
vmfb("c_Chem","a_Textiles","USA") = 2724.8662109375 ;
vmfb("c_Chem","a_Textiles","EU_28") = 17566.267578125 ;
vmfb("c_Chem","a_Textiles","CHN") = 12554.3896484375 ;
vmfb("c_Chem","a_Textiles","JPN") = 864.379089355469 ;
vmfb("c_Chem","a_Textiles","IND") = 4454.38525390625 ;
vmfb("c_Chem","a_Textiles","SSA") = 659.197021484375 ;
vmfb("c_Chem","a_Textiles","ROW") = 19549.482421875 ;
vmfb("c_Chem","a_Chem","USA") = 57402.83984375 ;
vmfb("c_Chem","a_Chem","EU_28") = 286369.875 ;
vmfb("c_Chem","a_Chem","CHN") = 95285.359375 ;
vmfb("c_Chem","a_Chem","JPN") = 29298.763671875 ;
vmfb("c_Chem","a_Chem","IND") = 25307.767578125 ;
vmfb("c_Chem","a_Chem","SSA") = 5330.46875 ;
vmfb("c_Chem","a_Chem","ROW") = 248747.5 ;
vmfb("c_Chem","a_Manuf","USA") = 29388.03125 ;
vmfb("c_Chem","a_Manuf","EU_28") = 109701.703125 ;
vmfb("c_Chem","a_Manuf","CHN") = 54351.22265625 ;
vmfb("c_Chem","a_Manuf","JPN") = 11415.185546875 ;
vmfb("c_Chem","a_Manuf","IND") = 6066.15966796875 ;
vmfb("c_Chem","a_Manuf","SSA") = 4750.27490234375 ;
vmfb("c_Chem","a_Manuf","ROW") = 131387.8125 ;
vmfb("c_Chem","a_ForestFish","USA") = 943.1162109375 ;
vmfb("c_Chem","a_ForestFish","EU_28") = 878.838073730469 ;
vmfb("c_Chem","a_ForestFish","CHN") = 561.534729003906 ;
vmfb("c_Chem","a_ForestFish","JPN") = 71.3035888671875 ;
vmfb("c_Chem","a_ForestFish","IND") = 16.8617858886719 ;
vmfb("c_Chem","a_ForestFish","SSA") = 215.980285644531 ;
vmfb("c_Chem","a_ForestFish","ROW") = 1678.35290527344 ;
vmfb("c_Chem","a_Svces","USA") = 62922.75 ;
vmfb("c_Chem","a_Svces","EU_28") = 202469.0625 ;
vmfb("c_Chem","a_Svces","CHN") = 54709.50390625 ;
vmfb("c_Chem","a_Svces","JPN") = 28923.490234375 ;
vmfb("c_Chem","a_Svces","IND") = 9972.4404296875 ;
vmfb("c_Chem","a_Svces","SSA") = 10250.66015625 ;
vmfb("c_Chem","a_Svces","ROW") = 163278.875 ;
vmfb("c_Manuf","a_Rice","USA") = 39.0426483154297 ;
vmfb("c_Manuf","a_Rice","EU_28") = 136.902389526367 ;
vmfb("c_Manuf","a_Rice","CHN") = 190.81169128418 ;
vmfb("c_Manuf","a_Rice","JPN") = 20.1963195800781 ;
vmfb("c_Manuf","a_Rice","IND") = 48.9920310974121 ;
vmfb("c_Manuf","a_Rice","SSA") = 116.991546630859 ;
vmfb("c_Manuf","a_Rice","ROW") = 1502.46936035156 ;
vmfb("c_Manuf","a_Crops","USA") = 1757.91870117188 ;
vmfb("c_Manuf","a_Crops","EU_28") = 3018.96215820312 ;
vmfb("c_Manuf","a_Crops","CHN") = 1085.23583984375 ;
vmfb("c_Manuf","a_Crops","JPN") = 84.813232421875 ;
vmfb("c_Manuf","a_Crops","IND") = 82.0761795043945 ;
vmfb("c_Manuf","a_Crops","SSA") = 2783.23022460938 ;
vmfb("c_Manuf","a_Crops","ROW") = 7689.478515625 ;
vmfb("c_Manuf","a_Livestock","USA") = 1126.94494628906 ;
vmfb("c_Manuf","a_Livestock","EU_28") = 1634.74841308594 ;
vmfb("c_Manuf","a_Livestock","CHN") = 307.409606933594 ;
vmfb("c_Manuf","a_Livestock","JPN") = 44.4644889831543 ;
vmfb("c_Manuf","a_Livestock","IND") = 127.006500244141 ;
vmfb("c_Manuf","a_Livestock","SSA") = 444.345825195312 ;
vmfb("c_Manuf","a_Livestock","ROW") = 4539.58251953125 ;
vmfb("c_Manuf","a_FoodProc","USA") = 12076.9599609375 ;
vmfb("c_Manuf","a_FoodProc","EU_28") = 23451.66796875 ;
vmfb("c_Manuf","a_FoodProc","CHN") = 3490.25708007812 ;
vmfb("c_Manuf","a_FoodProc","JPN") = 1047.98840332031 ;
vmfb("c_Manuf","a_FoodProc","IND") = 377.202758789062 ;
vmfb("c_Manuf","a_FoodProc","SSA") = 3485.79809570312 ;
vmfb("c_Manuf","a_FoodProc","ROW") = 25494.86328125 ;
vmfb("c_Manuf","a_Energy","USA") = 6834.0322265625 ;
vmfb("c_Manuf","a_Energy","EU_28") = 18172.77734375 ;
vmfb("c_Manuf","a_Energy","CHN") = 7716.37890625 ;
vmfb("c_Manuf","a_Energy","JPN") = 696.739624023438 ;
vmfb("c_Manuf","a_Energy","IND") = 4208.38427734375 ;
vmfb("c_Manuf","a_Energy","SSA") = 5099.11181640625 ;
vmfb("c_Manuf","a_Energy","ROW") = 72732.59375 ;
vmfb("c_Manuf","a_Textiles","USA") = 1029.29016113281 ;
vmfb("c_Manuf","a_Textiles","EU_28") = 9963.19921875 ;
vmfb("c_Manuf","a_Textiles","CHN") = 3331.5185546875 ;
vmfb("c_Manuf","a_Textiles","JPN") = 167.904479980469 ;
vmfb("c_Manuf","a_Textiles","IND") = 689.733276367188 ;
vmfb("c_Manuf","a_Textiles","SSA") = 717.087890625 ;
vmfb("c_Manuf","a_Textiles","ROW") = 12599.8955078125 ;
vmfb("c_Manuf","a_Chem","USA") = 15046.0654296875 ;
vmfb("c_Manuf","a_Chem","EU_28") = 44097.88671875 ;
vmfb("c_Manuf","a_Chem","CHN") = 17998.9140625 ;
vmfb("c_Manuf","a_Chem","JPN") = 1746.07727050781 ;
vmfb("c_Manuf","a_Chem","IND") = 4297.59716796875 ;
vmfb("c_Manuf","a_Chem","SSA") = 1190.07666015625 ;
vmfb("c_Manuf","a_Chem","ROW") = 34369.4140625 ;
vmfb("c_Manuf","a_Manuf","USA") = 385805.84375 ;
vmfb("c_Manuf","a_Manuf","EU_28") = 1311801.625 ;
vmfb("c_Manuf","a_Manuf","CHN") = 665336.875 ;
vmfb("c_Manuf","a_Manuf","JPN") = 154641.46875 ;
vmfb("c_Manuf","a_Manuf","IND") = 99641.578125 ;
vmfb("c_Manuf","a_Manuf","SSA") = 33978.203125 ;
vmfb("c_Manuf","a_Manuf","ROW") = 1326157.375 ;
vmfb("c_Manuf","a_ForestFish","USA") = 361.683135986328 ;
vmfb("c_Manuf","a_ForestFish","EU_28") = 2037.83374023438 ;
vmfb("c_Manuf","a_ForestFish","CHN") = 757.717529296875 ;
vmfb("c_Manuf","a_ForestFish","JPN") = 163.170822143555 ;
vmfb("c_Manuf","a_ForestFish","IND") = 619.288024902344 ;
vmfb("c_Manuf","a_ForestFish","SSA") = 1126.4873046875 ;
vmfb("c_Manuf","a_ForestFish","ROW") = 3921.13842773438 ;
vmfb("c_Manuf","a_Svces","USA") = 242852.234375 ;
vmfb("c_Manuf","a_Svces","EU_28") = 489558.09375 ;
vmfb("c_Manuf","a_Svces","CHN") = 184028.234375 ;
vmfb("c_Manuf","a_Svces","JPN") = 53263.7890625 ;
vmfb("c_Manuf","a_Svces","IND") = 38581.125 ;
vmfb("c_Manuf","a_Svces","SSA") = 41579.28125 ;
vmfb("c_Manuf","a_Svces","ROW") = 618706.375 ;
vmfb("c_ForestFish","a_Rice","USA") = 0.0543341599404812 ;
vmfb("c_ForestFish","a_Rice","EU_28") = 5.74515533447266 ;
vmfb("c_ForestFish","a_Rice","CHN") = 0.245071142911911 ;
vmfb("c_ForestFish","a_Rice","JPN") = 0.0428762957453728 ;
vmfb("c_ForestFish","a_Rice","IND") = 0.0189838223159313 ;
vmfb("c_ForestFish","a_Rice","SSA") = 0.079309768974781 ;
vmfb("c_ForestFish","a_Rice","ROW") = 11.5173635482788 ;
vmfb("c_ForestFish","a_Crops","USA") = 4.38259696960449 ;
vmfb("c_ForestFish","a_Crops","EU_28") = 18.0846004486084 ;
vmfb("c_ForestFish","a_Crops","CHN") = 1.30316090583801 ;
vmfb("c_ForestFish","a_Crops","JPN") = 1.89056539535522 ;
vmfb("c_ForestFish","a_Crops","IND") = 0.241792887449265 ;
vmfb("c_ForestFish","a_Crops","SSA") = 0.524484634399414 ;
vmfb("c_ForestFish","a_Crops","ROW") = 20.5042762756348 ;
vmfb("c_ForestFish","a_Livestock","USA") = 0.733328759670258 ;
vmfb("c_ForestFish","a_Livestock","EU_28") = 10.1328582763672 ;
vmfb("c_ForestFish","a_Livestock","CHN") = 0.180956482887268 ;
vmfb("c_ForestFish","a_Livestock","JPN") = 0.0373947955667973 ;
vmfb("c_ForestFish","a_Livestock","IND") = 2.52184677124023 ;
vmfb("c_ForestFish","a_Livestock","SSA") = 1.77727329730988 ;
vmfb("c_ForestFish","a_Livestock","ROW") = 14.0954694747925 ;
vmfb("c_ForestFish","a_FoodProc","USA") = 1192.68908691406 ;
vmfb("c_ForestFish","a_FoodProc","EU_28") = 5087.29736328125 ;
vmfb("c_ForestFish","a_FoodProc","CHN") = 852.660461425781 ;
vmfb("c_ForestFish","a_FoodProc","JPN") = 1066.00500488281 ;
vmfb("c_ForestFish","a_FoodProc","IND") = 66.1771774291992 ;
vmfb("c_ForestFish","a_FoodProc","SSA") = 37.7854270935059 ;
vmfb("c_ForestFish","a_FoodProc","ROW") = 1344.68896484375 ;
vmfb("c_ForestFish","a_Energy","USA") = 1.3042916059494 ;
vmfb("c_ForestFish","a_Energy","EU_28") = 38.0189552307129 ;
vmfb("c_ForestFish","a_Energy","CHN") = 28.4414405822754 ;
vmfb("c_ForestFish","a_Energy","JPN") = 0.0648832097649574 ;
vmfb("c_ForestFish","a_Energy","IND") = 0.497310727834702 ;
vmfb("c_ForestFish","a_Energy","SSA") = 0.49099200963974 ;
vmfb("c_ForestFish","a_Energy","ROW") = 5.63426971435547 ;
vmfb("c_ForestFish","a_Textiles","USA") = 3.34718132019043 ;
vmfb("c_ForestFish","a_Textiles","EU_28") = 114.579711914062 ;
vmfb("c_ForestFish","a_Textiles","CHN") = 53.096305847168 ;
vmfb("c_ForestFish","a_Textiles","JPN") = 1.63441038131714 ;
vmfb("c_ForestFish","a_Textiles","IND") = 1.63537585735321 ;
vmfb("c_ForestFish","a_Textiles","SSA") = 0.141292080283165 ;
vmfb("c_ForestFish","a_Textiles","ROW") = 5.13792419433594 ;
vmfb("c_ForestFish","a_Chem","USA") = 67.1875457763672 ;
vmfb("c_ForestFish","a_Chem","EU_28") = 379.252258300781 ;
vmfb("c_ForestFish","a_Chem","CHN") = 1780.79260253906 ;
vmfb("c_ForestFish","a_Chem","JPN") = 21.1329822540283 ;
vmfb("c_ForestFish","a_Chem","IND") = 120.300567626953 ;
vmfb("c_ForestFish","a_Chem","SSA") = 5.90251398086548 ;
vmfb("c_ForestFish","a_Chem","ROW") = 186.238754272461 ;
vmfb("c_ForestFish","a_Manuf","USA") = 213.22200012207 ;
vmfb("c_ForestFish","a_Manuf","EU_28") = 3803.77392578125 ;
vmfb("c_ForestFish","a_Manuf","CHN") = 5207.4365234375 ;
vmfb("c_ForestFish","a_Manuf","JPN") = 647.353515625 ;
vmfb("c_ForestFish","a_Manuf","IND") = 100.757759094238 ;
vmfb("c_ForestFish","a_Manuf","SSA") = 80.5433883666992 ;
vmfb("c_ForestFish","a_Manuf","ROW") = 1558.60070800781 ;
vmfb("c_ForestFish","a_ForestFish","USA") = 36.7644348144531 ;
vmfb("c_ForestFish","a_ForestFish","EU_28") = 912.794982910156 ;
vmfb("c_ForestFish","a_ForestFish","CHN") = 550.500183105469 ;
vmfb("c_ForestFish","a_ForestFish","JPN") = 183.013732910156 ;
vmfb("c_ForestFish","a_ForestFish","IND") = 5.47301006317139 ;
vmfb("c_ForestFish","a_ForestFish","SSA") = 24.1726913452148 ;
vmfb("c_ForestFish","a_ForestFish","ROW") = 1228.00610351562 ;
vmfb("c_ForestFish","a_Svces","USA") = 892.971557617188 ;
vmfb("c_ForestFish","a_Svces","EU_28") = 2813.12084960938 ;
vmfb("c_ForestFish","a_Svces","CHN") = 2921.27880859375 ;
vmfb("c_ForestFish","a_Svces","JPN") = 463.370941162109 ;
vmfb("c_ForestFish","a_Svces","IND") = 680.51904296875 ;
vmfb("c_ForestFish","a_Svces","SSA") = 20.0857315063477 ;
vmfb("c_ForestFish","a_Svces","ROW") = 1538.56872558594 ;
vmfb("c_Svces","a_Rice","USA") = 26.4498424530029 ;
vmfb("c_Svces","a_Rice","EU_28") = 152.958297729492 ;
vmfb("c_Svces","a_Rice","CHN") = 252.263900756836 ;
vmfb("c_Svces","a_Rice","JPN") = 98.6084060668945 ;
vmfb("c_Svces","a_Rice","IND") = 184.171920776367 ;
vmfb("c_Svces","a_Rice","SSA") = 56.945068359375 ;
vmfb("c_Svces","a_Rice","ROW") = 1096.01550292969 ;
vmfb("c_Svces","a_Crops","USA") = 818.154602050781 ;
vmfb("c_Svces","a_Crops","EU_28") = 4533.24658203125 ;
vmfb("c_Svces","a_Crops","CHN") = 1108.22106933594 ;
vmfb("c_Svces","a_Crops","JPN") = 115.113410949707 ;
vmfb("c_Svces","a_Crops","IND") = 587.841796875 ;
vmfb("c_Svces","a_Crops","SSA") = 1447.89685058594 ;
vmfb("c_Svces","a_Crops","ROW") = 4892.2568359375 ;
vmfb("c_Svces","a_Livestock","USA") = 573.681640625 ;
vmfb("c_Svces","a_Livestock","EU_28") = 2154.77905273438 ;
vmfb("c_Svces","a_Livestock","CHN") = 436.320617675781 ;
vmfb("c_Svces","a_Livestock","JPN") = 120.09236907959 ;
vmfb("c_Svces","a_Livestock","IND") = 143.656219482422 ;
vmfb("c_Svces","a_Livestock","SSA") = 355.036895751953 ;
vmfb("c_Svces","a_Livestock","ROW") = 2076.54833984375 ;
vmfb("c_Svces","a_FoodProc","USA") = 2708.39135742188 ;
vmfb("c_Svces","a_FoodProc","EU_28") = 37972.2890625 ;
vmfb("c_Svces","a_FoodProc","CHN") = 4898.23681640625 ;
vmfb("c_Svces","a_FoodProc","JPN") = 1330.94836425781 ;
vmfb("c_Svces","a_FoodProc","IND") = 550.701354980469 ;
vmfb("c_Svces","a_FoodProc","SSA") = 1995.32580566406 ;
vmfb("c_Svces","a_FoodProc","ROW") = 16050.109375 ;
vmfb("c_Svces","a_Energy","USA") = 2508.91772460938 ;
vmfb("c_Svces","a_Energy","EU_28") = 9512.2998046875 ;
vmfb("c_Svces","a_Energy","CHN") = 3307.57275390625 ;
vmfb("c_Svces","a_Energy","JPN") = 2507.06494140625 ;
vmfb("c_Svces","a_Energy","IND") = 3762.45849609375 ;
vmfb("c_Svces","a_Energy","SSA") = 5756.7412109375 ;
vmfb("c_Svces","a_Energy","ROW") = 34849.6796875 ;
vmfb("c_Svces","a_Textiles","USA") = 343.192108154297 ;
vmfb("c_Svces","a_Textiles","EU_28") = 10165.228515625 ;
vmfb("c_Svces","a_Textiles","CHN") = 4920.9814453125 ;
vmfb("c_Svces","a_Textiles","JPN") = 196.103073120117 ;
vmfb("c_Svces","a_Textiles","IND") = 592.670104980469 ;
vmfb("c_Svces","a_Textiles","SSA") = 612.440551757812 ;
vmfb("c_Svces","a_Textiles","ROW") = 6904.71044921875 ;
vmfb("c_Svces","a_Chem","USA") = 3547.51391601562 ;
vmfb("c_Svces","a_Chem","EU_28") = 62748.15234375 ;
vmfb("c_Svces","a_Chem","CHN") = 7627.95068359375 ;
vmfb("c_Svces","a_Chem","JPN") = 2177.45825195312 ;
vmfb("c_Svces","a_Chem","IND") = 1262.15380859375 ;
vmfb("c_Svces","a_Chem","SSA") = 793.734436035156 ;
vmfb("c_Svces","a_Chem","ROW") = 26319.7421875 ;
vmfb("c_Svces","a_Manuf","USA") = 17600.5390625 ;
vmfb("c_Svces","a_Manuf","EU_28") = 166835.0625 ;
vmfb("c_Svces","a_Manuf","CHN") = 34537.703125 ;
vmfb("c_Svces","a_Manuf","JPN") = 10428.6748046875 ;
vmfb("c_Svces","a_Manuf","IND") = 5357.3408203125 ;
vmfb("c_Svces","a_Manuf","SSA") = 5170.82763671875 ;
vmfb("c_Svces","a_Manuf","ROW") = 79130.8671875 ;
vmfb("c_Svces","a_ForestFish","USA") = 232.302322387695 ;
vmfb("c_Svces","a_ForestFish","EU_28") = 1596.27563476562 ;
vmfb("c_Svces","a_ForestFish","CHN") = 956.6572265625 ;
vmfb("c_Svces","a_ForestFish","JPN") = 82.3269500732422 ;
vmfb("c_Svces","a_ForestFish","IND") = 352.355712890625 ;
vmfb("c_Svces","a_ForestFish","SSA") = 908.649291992188 ;
vmfb("c_Svces","a_ForestFish","ROW") = 2659.87475585938 ;
vmfb("c_Svces","a_Svces","USA") = 221647.828125 ;
vmfb("c_Svces","a_Svces","EU_28") = 1174965.75 ;
vmfb("c_Svces","a_Svces","CHN") = 125330.65625 ;
vmfb("c_Svces","a_Svces","JPN") = 64262.52734375 ;
vmfb("c_Svces","a_Svces","IND") = 23730.224609375 ;
vmfb("c_Svces","a_Svces","SSA") = 30924.814453125 ;
vmfb("c_Svces","a_Svces","ROW") = 647501.625 ;
parameter vmfp(*,*,*) ;
vmfp("c_Rice","a_Rice","USA") = 61.1708183288574 ;
vmfp("c_Rice","a_Rice","EU_28") = 181.901611328125 ;
vmfp("c_Rice","a_Rice","CHN") = 136.949935913086 ;
vmfp("c_Rice","a_Rice","JPN") = 3.87160682678223 ;
vmfp("c_Rice","a_Rice","IND") = 0.325877338647842 ;
vmfp("c_Rice","a_Rice","SSA") = 269.5458984375 ;
vmfp("c_Rice","a_Rice","ROW") = 1435.83520507812 ;
vmfp("c_Rice","a_Crops","USA") = 0.133996486663818 ;
vmfp("c_Rice","a_Crops","EU_28") = 88.3933410644531 ;
vmfp("c_Rice","a_Crops","CHN") = 9.27870464324951 ;
vmfp("c_Rice","a_Crops","JPN") = 0.0558797158300877 ;
vmfp("c_Rice","a_Crops","IND") = 0.0399402752518654 ;
vmfp("c_Rice","a_Crops","SSA") = 23.5005626678467 ;
vmfp("c_Rice","a_Crops","ROW") = 125.159851074219 ;
vmfp("c_Rice","a_Livestock","USA") = 0.0612206980586052 ;
vmfp("c_Rice","a_Livestock","EU_28") = 3.16882491111755 ;
vmfp("c_Rice","a_Livestock","CHN") = 97.6236801147461 ;
vmfp("c_Rice","a_Livestock","JPN") = 0.487626880407333 ;
vmfp("c_Rice","a_Livestock","IND") = 0.0330982394516468 ;
vmfp("c_Rice","a_Livestock","SSA") = 123.891288757324 ;
vmfp("c_Rice","a_Livestock","ROW") = 156.336578369141 ;
vmfp("c_Rice","a_FoodProc","USA") = 716.472900390625 ;
vmfp("c_Rice","a_FoodProc","EU_28") = 1076.13269042969 ;
vmfp("c_Rice","a_FoodProc","CHN") = 1006.11560058594 ;
vmfp("c_Rice","a_FoodProc","JPN") = 274.314117431641 ;
vmfp("c_Rice","a_FoodProc","IND") = 3.44449257850647 ;
vmfp("c_Rice","a_FoodProc","SSA") = 2116.04345703125 ;
vmfp("c_Rice","a_FoodProc","ROW") = 6224.9091796875 ;
vmfp("c_Rice","a_Energy","USA") = 0.00194822018966079 ;
vmfp("c_Rice","a_Energy","EU_28") = 0.298384726047516 ;
vmfp("c_Rice","a_Energy","CHN") = 0.0176203548908234 ;
vmfp("c_Rice","a_Energy","JPN") = 0.0326388590037823 ;
vmfp("c_Rice","a_Energy","IND") = 0.00112529774196446 ;
vmfp("c_Rice","a_Energy","SSA") = 2.89961004257202 ;
vmfp("c_Rice","a_Energy","ROW") = 6.7021427154541 ;
vmfp("c_Rice","a_Textiles","USA") = 0.00430318946018815 ;
vmfp("c_Rice","a_Textiles","EU_28") = 1.95507955551147 ;
vmfp("c_Rice","a_Textiles","CHN") = 1.58541333675385 ;
vmfp("c_Rice","a_Textiles","JPN") = 0.0488685667514801 ;
vmfp("c_Rice","a_Textiles","IND") = 0.00536741549149156 ;
vmfp("c_Rice","a_Textiles","SSA") = 8.71528816223145 ;
vmfp("c_Rice","a_Textiles","ROW") = 13.7442007064819 ;
vmfp("c_Rice","a_Chem","USA") = 0.026293171569705 ;
vmfp("c_Rice","a_Chem","EU_28") = 9.76237678527832 ;
vmfp("c_Rice","a_Chem","CHN") = 5.31347560882568 ;
vmfp("c_Rice","a_Chem","JPN") = 0.662290990352631 ;
vmfp("c_Rice","a_Chem","IND") = 0.0276185311377048 ;
vmfp("c_Rice","a_Chem","SSA") = 66.3620223999023 ;
vmfp("c_Rice","a_Chem","ROW") = 26.1937484741211 ;
vmfp("c_Rice","a_Manuf","USA") = 0.103057436645031 ;
vmfp("c_Rice","a_Manuf","EU_28") = 4.77794599533081 ;
vmfp("c_Rice","a_Manuf","CHN") = 3.21680974960327 ;
vmfp("c_Rice","a_Manuf","JPN") = 0.703385531902313 ;
vmfp("c_Rice","a_Manuf","IND") = 0.0218414422124624 ;
vmfp("c_Rice","a_Manuf","SSA") = 23.6206588745117 ;
vmfp("c_Rice","a_Manuf","ROW") = 32.141975402832 ;
vmfp("c_Rice","a_ForestFish","USA") = 0.0140189779922366 ;
vmfp("c_Rice","a_ForestFish","EU_28") = 1.64032220840454 ;
vmfp("c_Rice","a_ForestFish","CHN") = 20.0566368103027 ;
vmfp("c_Rice","a_ForestFish","JPN") = 0.901321113109589 ;
vmfp("c_Rice","a_ForestFish","IND") = 0.0197734832763672 ;
vmfp("c_Rice","a_ForestFish","SSA") = 2.57794308662415 ;
vmfp("c_Rice","a_ForestFish","ROW") = 31.6184902191162 ;
vmfp("c_Rice","a_Svces","USA") = 3.39314556121826 ;
vmfp("c_Rice","a_Svces","EU_28") = 314.64208984375 ;
vmfp("c_Rice","a_Svces","CHN") = 182.984588623047 ;
vmfp("c_Rice","a_Svces","JPN") = 257.304107666016 ;
vmfp("c_Rice","a_Svces","IND") = 0.224182173609734 ;
vmfp("c_Rice","a_Svces","SSA") = 486.513427734375 ;
vmfp("c_Rice","a_Svces","ROW") = 1432.01684570312 ;
vmfp("c_Crops","a_Rice","USA") = 14.8456535339355 ;
vmfp("c_Crops","a_Rice","EU_28") = 39.7229652404785 ;
vmfp("c_Crops","a_Rice","CHN") = 3404.87060546875 ;
vmfp("c_Crops","a_Rice","JPN") = 151.869079589844 ;
vmfp("c_Crops","a_Rice","IND") = 5.60439825057983 ;
vmfp("c_Crops","a_Rice","SSA") = 157.071731567383 ;
vmfp("c_Crops","a_Rice","ROW") = 3252.93334960938 ;
vmfp("c_Crops","a_Crops","USA") = 1834.07177734375 ;
vmfp("c_Crops","a_Crops","EU_28") = 4741.9560546875 ;
vmfp("c_Crops","a_Crops","CHN") = 3650.70092773438 ;
vmfp("c_Crops","a_Crops","JPN") = 295.486053466797 ;
vmfp("c_Crops","a_Crops","IND") = 59.3420677185059 ;
vmfp("c_Crops","a_Crops","SSA") = 504.572814941406 ;
vmfp("c_Crops","a_Crops","ROW") = 9499.6181640625 ;
vmfp("c_Crops","a_Livestock","USA") = 360.662231445312 ;
vmfp("c_Crops","a_Livestock","EU_28") = 4607.2939453125 ;
vmfp("c_Crops","a_Livestock","CHN") = 2095.34790039062 ;
vmfp("c_Crops","a_Livestock","JPN") = 916.419860839844 ;
vmfp("c_Crops","a_Livestock","IND") = 832.7177734375 ;
vmfp("c_Crops","a_Livestock","SSA") = 830.269775390625 ;
vmfp("c_Crops","a_Livestock","ROW") = 15573.7529296875 ;
vmfp("c_Crops","a_FoodProc","USA") = 14812.5625 ;
vmfp("c_Crops","a_FoodProc","EU_28") = 58646.66796875 ;
vmfp("c_Crops","a_FoodProc","CHN") = 33152.67578125 ;
vmfp("c_Crops","a_FoodProc","JPN") = 11576.4921875 ;
vmfp("c_Crops","a_FoodProc","IND") = 1720.33850097656 ;
vmfp("c_Crops","a_FoodProc","SSA") = 2550.62377929688 ;
vmfp("c_Crops","a_FoodProc","ROW") = 69284.140625 ;
vmfp("c_Crops","a_Energy","USA") = 3.46352815628052 ;
vmfp("c_Crops","a_Energy","EU_28") = 52.1781196594238 ;
vmfp("c_Crops","a_Energy","CHN") = 4.49749326705933 ;
vmfp("c_Crops","a_Energy","JPN") = 0.292086392641068 ;
vmfp("c_Crops","a_Energy","IND") = 27.3098125457764 ;
vmfp("c_Crops","a_Energy","SSA") = 8.40481948852539 ;
vmfp("c_Crops","a_Energy","ROW") = 104.899948120117 ;
vmfp("c_Crops","a_Textiles","USA") = 29.0988636016846 ;
vmfp("c_Crops","a_Textiles","EU_28") = 396.160003662109 ;
vmfp("c_Crops","a_Textiles","CHN") = 3605.68505859375 ;
vmfp("c_Crops","a_Textiles","JPN") = 111.260604858398 ;
vmfp("c_Crops","a_Textiles","IND") = 303.776336669922 ;
vmfp("c_Crops","a_Textiles","SSA") = 41.0065650939941 ;
vmfp("c_Crops","a_Textiles","ROW") = 6455.76513671875 ;
vmfp("c_Crops","a_Chem","USA") = 825.606994628906 ;
vmfp("c_Crops","a_Chem","EU_28") = 2411.23168945312 ;
vmfp("c_Crops","a_Chem","CHN") = 2051.36645507812 ;
vmfp("c_Crops","a_Chem","JPN") = 992.888244628906 ;
vmfp("c_Crops","a_Chem","IND") = 176.840072631836 ;
vmfp("c_Crops","a_Chem","SSA") = 33.5563888549805 ;
vmfp("c_Crops","a_Chem","ROW") = 3853.00830078125 ;
vmfp("c_Crops","a_Manuf","USA") = 193.725173950195 ;
vmfp("c_Crops","a_Manuf","EU_28") = 493.361328125 ;
vmfp("c_Crops","a_Manuf","CHN") = 1096.27282714844 ;
vmfp("c_Crops","a_Manuf","JPN") = 31.5647392272949 ;
vmfp("c_Crops","a_Manuf","IND") = 13.594220161438 ;
vmfp("c_Crops","a_Manuf","SSA") = 61.7408065795898 ;
vmfp("c_Crops","a_Manuf","ROW") = 880.455078125 ;
vmfp("c_Crops","a_ForestFish","USA") = 115.336196899414 ;
vmfp("c_Crops","a_ForestFish","EU_28") = 285.586273193359 ;
vmfp("c_Crops","a_ForestFish","CHN") = 78.4963684082031 ;
vmfp("c_Crops","a_ForestFish","JPN") = 0.775204956531525 ;
vmfp("c_Crops","a_ForestFish","IND") = 13.1334476470947 ;
vmfp("c_Crops","a_ForestFish","SSA") = 41.6706275939941 ;
vmfp("c_Crops","a_ForestFish","ROW") = 225.095764160156 ;
vmfp("c_Crops","a_Svces","USA") = 2588.44482421875 ;
vmfp("c_Crops","a_Svces","EU_28") = 10561.96484375 ;
vmfp("c_Crops","a_Svces","CHN") = 4135.0791015625 ;
vmfp("c_Crops","a_Svces","JPN") = 1944.25378417969 ;
vmfp("c_Crops","a_Svces","IND") = 894.700073242188 ;
vmfp("c_Crops","a_Svces","SSA") = 446.41357421875 ;
vmfp("c_Crops","a_Svces","ROW") = 19466.880859375 ;
vmfp("c_Livestock","a_Rice","USA") = 1.11137068271637 ;
vmfp("c_Livestock","a_Rice","EU_28") = 12.9933204650879 ;
vmfp("c_Livestock","a_Rice","CHN") = 1.12000453472137 ;
vmfp("c_Livestock","a_Rice","JPN") = 2.2288281917572 ;
vmfp("c_Livestock","a_Rice","IND") = 4.12547731399536 ;
vmfp("c_Livestock","a_Rice","SSA") = 1.81926381587982 ;
vmfp("c_Livestock","a_Rice","ROW") = 109.273971557617 ;
vmfp("c_Livestock","a_Crops","USA") = 9.03999614715576 ;
vmfp("c_Livestock","a_Crops","EU_28") = 184.744277954102 ;
vmfp("c_Livestock","a_Crops","CHN") = 1.62168073654175 ;
vmfp("c_Livestock","a_Crops","JPN") = 2.63873696327209 ;
vmfp("c_Livestock","a_Crops","IND") = 2.33411121368408 ;
vmfp("c_Livestock","a_Crops","SSA") = 2.06463360786438 ;
vmfp("c_Livestock","a_Crops","ROW") = 113.730430603027 ;
vmfp("c_Livestock","a_Livestock","USA") = 634.675903320312 ;
vmfp("c_Livestock","a_Livestock","EU_28") = 168.975296020508 ;
vmfp("c_Livestock","a_Livestock","CHN") = 605.625 ;
vmfp("c_Livestock","a_Livestock","JPN") = 72.0838394165039 ;
vmfp("c_Livestock","a_Livestock","IND") = 2.86413717269897 ;
vmfp("c_Livestock","a_Livestock","SSA") = 80.9110488891602 ;
vmfp("c_Livestock","a_Livestock","ROW") = 1677.74951171875 ;
vmfp("c_Livestock","a_FoodProc","USA") = 2973.01904296875 ;
vmfp("c_Livestock","a_FoodProc","EU_28") = 15228.5419921875 ;
vmfp("c_Livestock","a_FoodProc","CHN") = 3774.89184570312 ;
vmfp("c_Livestock","a_FoodProc","JPN") = 302.442840576172 ;
vmfp("c_Livestock","a_FoodProc","IND") = 33.464542388916 ;
vmfp("c_Livestock","a_FoodProc","SSA") = 409.222442626953 ;
vmfp("c_Livestock","a_FoodProc","ROW") = 7487.8662109375 ;
vmfp("c_Livestock","a_Energy","USA") = 0.0185928530991077 ;
vmfp("c_Livestock","a_Energy","EU_28") = 19.3890399932861 ;
vmfp("c_Livestock","a_Energy","CHN") = 0.122600041329861 ;
vmfp("c_Livestock","a_Energy","JPN") = 0.106916300952435 ;
vmfp("c_Livestock","a_Energy","IND") = 0.445617526769638 ;
vmfp("c_Livestock","a_Energy","SSA") = 0.649567782878876 ;
vmfp("c_Livestock","a_Energy","ROW") = 3.1850106716156 ;
vmfp("c_Livestock","a_Textiles","USA") = 1.21313452720642 ;
vmfp("c_Livestock","a_Textiles","EU_28") = 158.567825317383 ;
vmfp("c_Livestock","a_Textiles","CHN") = 1203.92224121094 ;
vmfp("c_Livestock","a_Textiles","JPN") = 5.13249778747559 ;
vmfp("c_Livestock","a_Textiles","IND") = 24.0176162719727 ;
vmfp("c_Livestock","a_Textiles","SSA") = 25.0094165802002 ;
vmfp("c_Livestock","a_Textiles","ROW") = 1013.87939453125 ;
vmfp("c_Livestock","a_Chem","USA") = 2.85826873779297 ;
vmfp("c_Livestock","a_Chem","EU_28") = 780.820068359375 ;
vmfp("c_Livestock","a_Chem","CHN") = 689.724670410156 ;
vmfp("c_Livestock","a_Chem","JPN") = 3.67035603523254 ;
vmfp("c_Livestock","a_Chem","IND") = 51.0764427185059 ;
vmfp("c_Livestock","a_Chem","SSA") = 0.513412594795227 ;
vmfp("c_Livestock","a_Chem","ROW") = 98.9948425292969 ;
vmfp("c_Livestock","a_Manuf","USA") = 1.13471531867981 ;
vmfp("c_Livestock","a_Manuf","EU_28") = 43.0970230102539 ;
vmfp("c_Livestock","a_Manuf","CHN") = 88.4456100463867 ;
vmfp("c_Livestock","a_Manuf","JPN") = 2.02537655830383 ;
vmfp("c_Livestock","a_Manuf","IND") = 0.703625917434692 ;
vmfp("c_Livestock","a_Manuf","SSA") = 5.44949722290039 ;
vmfp("c_Livestock","a_Manuf","ROW") = 27.5106258392334 ;
vmfp("c_Livestock","a_ForestFish","USA") = 17.4348812103271 ;
vmfp("c_Livestock","a_ForestFish","EU_28") = 19.8130073547363 ;
vmfp("c_Livestock","a_ForestFish","CHN") = 0.146197617053986 ;
vmfp("c_Livestock","a_ForestFish","JPN") = 0.959405481815338 ;
vmfp("c_Livestock","a_ForestFish","IND") = 0.0649078711867332 ;
vmfp("c_Livestock","a_ForestFish","SSA") = 0.845552921295166 ;
vmfp("c_Livestock","a_ForestFish","ROW") = 13.2644128799438 ;
vmfp("c_Livestock","a_Svces","USA") = 134.755035400391 ;
vmfp("c_Livestock","a_Svces","EU_28") = 1601.50170898438 ;
vmfp("c_Livestock","a_Svces","CHN") = 407.266906738281 ;
vmfp("c_Livestock","a_Svces","JPN") = 127.68726348877 ;
vmfp("c_Livestock","a_Svces","IND") = 102.704895019531 ;
vmfp("c_Livestock","a_Svces","SSA") = 37.5345306396484 ;
vmfp("c_Livestock","a_Svces","ROW") = 2120.63623046875 ;
vmfp("c_FoodProc","a_Rice","USA") = 3618.05834960938 ;
vmfp("c_FoodProc","a_Rice","EU_28") = 87.2942199707031 ;
vmfp("c_FoodProc","a_Rice","CHN") = 14.9248180389404 ;
vmfp("c_FoodProc","a_Rice","JPN") = 2.19171667098999 ;
vmfp("c_FoodProc","a_Rice","IND") = 1.92152452468872 ;
vmfp("c_FoodProc","a_Rice","SSA") = 33.1175537109375 ;
vmfp("c_FoodProc","a_Rice","ROW") = 742.444641113281 ;
vmfp("c_FoodProc","a_Crops","USA") = 22.027193069458 ;
vmfp("c_FoodProc","a_Crops","EU_28") = 987.146240234375 ;
vmfp("c_FoodProc","a_Crops","CHN") = 108.420303344727 ;
vmfp("c_FoodProc","a_Crops","JPN") = 25.2421455383301 ;
vmfp("c_FoodProc","a_Crops","IND") = 2.80565285682678 ;
vmfp("c_FoodProc","a_Crops","SSA") = 1038.66857910156 ;
vmfp("c_FoodProc","a_Crops","ROW") = 3904.38208007812 ;
vmfp("c_FoodProc","a_Livestock","USA") = 873.95751953125 ;
vmfp("c_FoodProc","a_Livestock","EU_28") = 6851.7724609375 ;
vmfp("c_FoodProc","a_Livestock","CHN") = 2371.25341796875 ;
vmfp("c_FoodProc","a_Livestock","JPN") = 1828.32946777344 ;
vmfp("c_FoodProc","a_Livestock","IND") = 914.462097167969 ;
vmfp("c_FoodProc","a_Livestock","SSA") = 1106.51623535156 ;
vmfp("c_FoodProc","a_Livestock","ROW") = 12458.1572265625 ;
vmfp("c_FoodProc","a_FoodProc","USA") = 17389.064453125 ;
vmfp("c_FoodProc","a_FoodProc","EU_28") = 90969.96875 ;
vmfp("c_FoodProc","a_FoodProc","CHN") = 18340.30859375 ;
vmfp("c_FoodProc","a_FoodProc","JPN") = 11485.7294921875 ;
vmfp("c_FoodProc","a_FoodProc","IND") = 1591.76000976562 ;
vmfp("c_FoodProc","a_FoodProc","SSA") = 6793.228515625 ;
vmfp("c_FoodProc","a_FoodProc","ROW") = 85042.3671875 ;
vmfp("c_FoodProc","a_Energy","USA") = 13.5686130523682 ;
vmfp("c_FoodProc","a_Energy","EU_28") = 188.552276611328 ;
vmfp("c_FoodProc","a_Energy","CHN") = 145.70588684082 ;
vmfp("c_FoodProc","a_Energy","JPN") = 0.732357680797577 ;
vmfp("c_FoodProc","a_Energy","IND") = 9.87807273864746 ;
vmfp("c_FoodProc","a_Energy","SSA") = 35.8821220397949 ;
vmfp("c_FoodProc","a_Energy","ROW") = 303.983337402344 ;
vmfp("c_FoodProc","a_Textiles","USA") = 179.63395690918 ;
vmfp("c_FoodProc","a_Textiles","EU_28") = 2227.11279296875 ;
vmfp("c_FoodProc","a_Textiles","CHN") = 1430.95031738281 ;
vmfp("c_FoodProc","a_Textiles","JPN") = 126.070831298828 ;
vmfp("c_FoodProc","a_Textiles","IND") = 2.43000340461731 ;
vmfp("c_FoodProc","a_Textiles","SSA") = 88.3125686645508 ;
vmfp("c_FoodProc","a_Textiles","ROW") = 2124.77319335938 ;
vmfp("c_FoodProc","a_Chem","USA") = 352.965484619141 ;
vmfp("c_FoodProc","a_Chem","EU_28") = 14722.4716796875 ;
vmfp("c_FoodProc","a_Chem","CHN") = 1973.30627441406 ;
vmfp("c_FoodProc","a_Chem","JPN") = 387.920562744141 ;
vmfp("c_FoodProc","a_Chem","IND") = 184.961166381836 ;
vmfp("c_FoodProc","a_Chem","SSA") = 103.65177154541 ;
vmfp("c_FoodProc","a_Chem","ROW") = 2959.66552734375 ;
vmfp("c_FoodProc","a_Manuf","USA") = 268.26318359375 ;
vmfp("c_FoodProc","a_Manuf","EU_28") = 1437.58483886719 ;
vmfp("c_FoodProc","a_Manuf","CHN") = 949.064147949219 ;
vmfp("c_FoodProc","a_Manuf","JPN") = 80.7151870727539 ;
vmfp("c_FoodProc","a_Manuf","IND") = 2.06767702102661 ;
vmfp("c_FoodProc","a_Manuf","SSA") = 159.948776245117 ;
vmfp("c_FoodProc","a_Manuf","ROW") = 943.952331542969 ;
vmfp("c_FoodProc","a_ForestFish","USA") = 24.1593551635742 ;
vmfp("c_FoodProc","a_ForestFish","EU_28") = 290.378387451172 ;
vmfp("c_FoodProc","a_ForestFish","CHN") = 874.246459960938 ;
vmfp("c_FoodProc","a_ForestFish","JPN") = 196.472015380859 ;
vmfp("c_FoodProc","a_ForestFish","IND") = 2.35672760009766 ;
vmfp("c_FoodProc","a_ForestFish","SSA") = 44.6833992004395 ;
vmfp("c_FoodProc","a_ForestFish","ROW") = 4727.72509765625 ;
vmfp("c_FoodProc","a_Svces","USA") = 19488.646484375 ;
vmfp("c_FoodProc","a_Svces","EU_28") = 64813.5234375 ;
vmfp("c_FoodProc","a_Svces","CHN") = 9385.3095703125 ;
vmfp("c_FoodProc","a_Svces","JPN") = 15629 ;
vmfp("c_FoodProc","a_Svces","IND") = 488.976776123047 ;
vmfp("c_FoodProc","a_Svces","SSA") = 5253.37158203125 ;
vmfp("c_FoodProc","a_Svces","ROW") = 70252.3359375 ;
vmfp("c_Energy","a_Rice","USA") = 5.44274234771729 ;
vmfp("c_Energy","a_Rice","EU_28") = 73.6005249023438 ;
vmfp("c_Energy","a_Rice","CHN") = 114.091888427734 ;
vmfp("c_Energy","a_Rice","JPN") = 38.4875717163086 ;
vmfp("c_Energy","a_Rice","IND") = 188.797714233398 ;
vmfp("c_Energy","a_Rice","SSA") = 10.8172216415405 ;
vmfp("c_Energy","a_Rice","ROW") = 652.304931640625 ;
vmfp("c_Energy","a_Crops","USA") = 693.327392578125 ;
vmfp("c_Energy","a_Crops","EU_28") = 2921.927734375 ;
vmfp("c_Energy","a_Crops","CHN") = 650.138305664062 ;
vmfp("c_Energy","a_Crops","JPN") = 148.512252807617 ;
vmfp("c_Energy","a_Crops","IND") = 299.730743408203 ;
vmfp("c_Energy","a_Crops","SSA") = 332.259826660156 ;
vmfp("c_Energy","a_Crops","ROW") = 4573.5302734375 ;
vmfp("c_Energy","a_Livestock","USA") = 432.171447753906 ;
vmfp("c_Energy","a_Livestock","EU_28") = 774.633544921875 ;
vmfp("c_Energy","a_Livestock","CHN") = 155.954513549805 ;
vmfp("c_Energy","a_Livestock","JPN") = 18.877815246582 ;
vmfp("c_Energy","a_Livestock","IND") = 17.4314498901367 ;
vmfp("c_Energy","a_Livestock","SSA") = 113.264953613281 ;
vmfp("c_Energy","a_Livestock","ROW") = 943.174377441406 ;
vmfp("c_Energy","a_FoodProc","USA") = 196.368194580078 ;
vmfp("c_Energy","a_FoodProc","EU_28") = 5528.40380859375 ;
vmfp("c_Energy","a_FoodProc","CHN") = 810.331359863281 ;
vmfp("c_Energy","a_FoodProc","JPN") = 931.680969238281 ;
vmfp("c_Energy","a_FoodProc","IND") = 282.466003417969 ;
vmfp("c_Energy","a_FoodProc","SSA") = 323.791320800781 ;
vmfp("c_Energy","a_FoodProc","ROW") = 3900.23046875 ;
vmfp("c_Energy","a_Energy","USA") = 157361.59375 ;
vmfp("c_Energy","a_Energy","EU_28") = 315042.75 ;
vmfp("c_Energy","a_Energy","CHN") = 192800.546875 ;
vmfp("c_Energy","a_Energy","JPN") = 120376.0625 ;
vmfp("c_Energy","a_Energy","IND") = 107046.9921875 ;
vmfp("c_Energy","a_Energy","SSA") = 14201.7666015625 ;
vmfp("c_Energy","a_Energy","ROW") = 370898.46875 ;
vmfp("c_Energy","a_Textiles","USA") = 31.2678089141846 ;
vmfp("c_Energy","a_Textiles","EU_28") = 946.147338867188 ;
vmfp("c_Energy","a_Textiles","CHN") = 672.181213378906 ;
vmfp("c_Energy","a_Textiles","JPN") = 104.455955505371 ;
vmfp("c_Energy","a_Textiles","IND") = 114.471115112305 ;
vmfp("c_Energy","a_Textiles","SSA") = 78.412727355957 ;
vmfp("c_Energy","a_Textiles","ROW") = 1468.81518554688 ;
vmfp("c_Energy","a_Chem","USA") = 9088.8369140625 ;
vmfp("c_Energy","a_Chem","EU_28") = 40256.4375 ;
vmfp("c_Energy","a_Chem","CHN") = 10009.142578125 ;
vmfp("c_Energy","a_Chem","JPN") = 6613.34130859375 ;
vmfp("c_Energy","a_Chem","IND") = 5133.46630859375 ;
vmfp("c_Energy","a_Chem","SSA") = 1015.01007080078 ;
vmfp("c_Energy","a_Chem","ROW") = 33515.28125 ;
vmfp("c_Energy","a_Manuf","USA") = 1595.95373535156 ;
vmfp("c_Energy","a_Manuf","EU_28") = 27789.4375 ;
vmfp("c_Energy","a_Manuf","CHN") = 16231.7431640625 ;
vmfp("c_Energy","a_Manuf","JPN") = 6732.1484375 ;
vmfp("c_Energy","a_Manuf","IND") = 6010.06201171875 ;
vmfp("c_Energy","a_Manuf","SSA") = 3401.14770507812 ;
vmfp("c_Energy","a_Manuf","ROW") = 36008.1875 ;
vmfp("c_Energy","a_ForestFish","USA") = 166.080917358398 ;
vmfp("c_Energy","a_ForestFish","EU_28") = 987.313293457031 ;
vmfp("c_Energy","a_ForestFish","CHN") = 552.802429199219 ;
vmfp("c_Energy","a_ForestFish","JPN") = 274.196441650391 ;
vmfp("c_Energy","a_ForestFish","IND") = 88.4450378417969 ;
vmfp("c_Energy","a_ForestFish","SSA") = 391.417724609375 ;
vmfp("c_Energy","a_ForestFish","ROW") = 2497.78295898438 ;
vmfp("c_Energy","a_Svces","USA") = 43398.08984375 ;
vmfp("c_Energy","a_Svces","EU_28") = 164243.9375 ;
vmfp("c_Energy","a_Svces","CHN") = 13308.4892578125 ;
vmfp("c_Energy","a_Svces","JPN") = 10382.25390625 ;
vmfp("c_Energy","a_Svces","IND") = 5738.47509765625 ;
vmfp("c_Energy","a_Svces","SSA") = 22903.193359375 ;
vmfp("c_Energy","a_Svces","ROW") = 157956.875 ;
vmfp("c_Textiles","a_Rice","USA") = 7.29989433288574 ;
vmfp("c_Textiles","a_Rice","EU_28") = 14.5605020523071 ;
vmfp("c_Textiles","a_Rice","CHN") = 19.9603443145752 ;
vmfp("c_Textiles","a_Rice","JPN") = 49.8792610168457 ;
vmfp("c_Textiles","a_Rice","IND") = 2.35322570800781 ;
vmfp("c_Textiles","a_Rice","SSA") = 18.2704792022705 ;
vmfp("c_Textiles","a_Rice","ROW") = 252.300262451172 ;
vmfp("c_Textiles","a_Crops","USA") = 91.9533309936523 ;
vmfp("c_Textiles","a_Crops","EU_28") = 375.177642822266 ;
vmfp("c_Textiles","a_Crops","CHN") = 7.51957941055298 ;
vmfp("c_Textiles","a_Crops","JPN") = 79.9925689697266 ;
vmfp("c_Textiles","a_Crops","IND") = 0.663280308246613 ;
vmfp("c_Textiles","a_Crops","SSA") = 546.700744628906 ;
vmfp("c_Textiles","a_Crops","ROW") = 1213.84252929688 ;
vmfp("c_Textiles","a_Livestock","USA") = 29.6034908294678 ;
vmfp("c_Textiles","a_Livestock","EU_28") = 173.519927978516 ;
vmfp("c_Textiles","a_Livestock","CHN") = 1.66231393814087 ;
vmfp("c_Textiles","a_Livestock","JPN") = 14.8105812072754 ;
vmfp("c_Textiles","a_Livestock","IND") = 1.79659008979797 ;
vmfp("c_Textiles","a_Livestock","SSA") = 106.343330383301 ;
vmfp("c_Textiles","a_Livestock","ROW") = 266.549682617188 ;
vmfp("c_Textiles","a_FoodProc","USA") = 169.201232910156 ;
vmfp("c_Textiles","a_FoodProc","EU_28") = 1487.53283691406 ;
vmfp("c_Textiles","a_FoodProc","CHN") = 303.4482421875 ;
vmfp("c_Textiles","a_FoodProc","JPN") = 239.397964477539 ;
vmfp("c_Textiles","a_FoodProc","IND") = 9.28589916229248 ;
vmfp("c_Textiles","a_FoodProc","SSA") = 265.290130615234 ;
vmfp("c_Textiles","a_FoodProc","ROW") = 2045.94067382812 ;
vmfp("c_Textiles","a_Energy","USA") = 66.7627716064453 ;
vmfp("c_Textiles","a_Energy","EU_28") = 320.793548583984 ;
vmfp("c_Textiles","a_Energy","CHN") = 133.930770874023 ;
vmfp("c_Textiles","a_Energy","JPN") = 42.5447654724121 ;
vmfp("c_Textiles","a_Energy","IND") = 118.250602722168 ;
vmfp("c_Textiles","a_Energy","SSA") = 455.379547119141 ;
vmfp("c_Textiles","a_Energy","ROW") = 1502.978515625 ;
vmfp("c_Textiles","a_Textiles","USA") = 6603.4501953125 ;
vmfp("c_Textiles","a_Textiles","EU_28") = 75856.3203125 ;
vmfp("c_Textiles","a_Textiles","CHN") = 32617.99609375 ;
vmfp("c_Textiles","a_Textiles","JPN") = 6619.6357421875 ;
vmfp("c_Textiles","a_Textiles","IND") = 2445.53930664062 ;
vmfp("c_Textiles","a_Textiles","SSA") = 4270.28271484375 ;
vmfp("c_Textiles","a_Textiles","ROW") = 110716.5546875 ;
vmfp("c_Textiles","a_Chem","USA") = 1495.10388183594 ;
vmfp("c_Textiles","a_Chem","EU_28") = 10623.3154296875 ;
vmfp("c_Textiles","a_Chem","CHN") = 885.678100585938 ;
vmfp("c_Textiles","a_Chem","JPN") = 643.407348632812 ;
vmfp("c_Textiles","a_Chem","IND") = 373.911926269531 ;
vmfp("c_Textiles","a_Chem","SSA") = 168.342849731445 ;
vmfp("c_Textiles","a_Chem","ROW") = 4759.6708984375 ;
vmfp("c_Textiles","a_Manuf","USA") = 7684.53076171875 ;
vmfp("c_Textiles","a_Manuf","EU_28") = 20371.734375 ;
vmfp("c_Textiles","a_Manuf","CHN") = 3782.0654296875 ;
vmfp("c_Textiles","a_Manuf","JPN") = 3463.82202148438 ;
vmfp("c_Textiles","a_Manuf","IND") = 121.838638305664 ;
vmfp("c_Textiles","a_Manuf","SSA") = 838.578857421875 ;
vmfp("c_Textiles","a_Manuf","ROW") = 18031.796875 ;
vmfp("c_Textiles","a_ForestFish","USA") = 59.3830146789551 ;
vmfp("c_Textiles","a_ForestFish","EU_28") = 457.552551269531 ;
vmfp("c_Textiles","a_ForestFish","CHN") = 70.1535949707031 ;
vmfp("c_Textiles","a_ForestFish","JPN") = 180.836395263672 ;
vmfp("c_Textiles","a_ForestFish","IND") = 17.7654266357422 ;
vmfp("c_Textiles","a_ForestFish","SSA") = 395.674560546875 ;
vmfp("c_Textiles","a_ForestFish","ROW") = 951.402038574219 ;
vmfp("c_Textiles","a_Svces","USA") = 18820.216796875 ;
vmfp("c_Textiles","a_Svces","EU_28") = 34567.32421875 ;
vmfp("c_Textiles","a_Svces","CHN") = 3876.05346679688 ;
vmfp("c_Textiles","a_Svces","JPN") = 11041.2763671875 ;
vmfp("c_Textiles","a_Svces","IND") = 545.987060546875 ;
vmfp("c_Textiles","a_Svces","SSA") = 3737.50854492188 ;
vmfp("c_Textiles","a_Svces","ROW") = 36531.53125 ;
vmfp("c_Chem","a_Rice","USA") = 73.6272888183594 ;
vmfp("c_Chem","a_Rice","EU_28") = 160.399566650391 ;
vmfp("c_Chem","a_Rice","CHN") = 2294.986328125 ;
vmfp("c_Chem","a_Rice","JPN") = 335.8095703125 ;
vmfp("c_Chem","a_Rice","IND") = 919.46923828125 ;
vmfp("c_Chem","a_Rice","SSA") = 225.407943725586 ;
vmfp("c_Chem","a_Rice","ROW") = 6274.6396484375 ;
vmfp("c_Chem","a_Crops","USA") = 5384.29296875 ;
vmfp("c_Chem","a_Crops","EU_28") = 7721.61474609375 ;
vmfp("c_Chem","a_Crops","CHN") = 14832.2802734375 ;
vmfp("c_Chem","a_Crops","JPN") = 706.431518554688 ;
vmfp("c_Chem","a_Crops","IND") = 313.535369873047 ;
vmfp("c_Chem","a_Crops","SSA") = 5562.15380859375 ;
vmfp("c_Chem","a_Crops","ROW") = 29030.0078125 ;
vmfp("c_Chem","a_Livestock","USA") = 513.016845703125 ;
vmfp("c_Chem","a_Livestock","EU_28") = 3647.86401367188 ;
vmfp("c_Chem","a_Livestock","CHN") = 175.530548095703 ;
vmfp("c_Chem","a_Livestock","JPN") = 150.127517700195 ;
vmfp("c_Chem","a_Livestock","IND") = 18.0565719604492 ;
vmfp("c_Chem","a_Livestock","SSA") = 532.333312988281 ;
vmfp("c_Chem","a_Livestock","ROW") = 4901.00244140625 ;
vmfp("c_Chem","a_FoodProc","USA") = 6702.990234375 ;
vmfp("c_Chem","a_FoodProc","EU_28") = 25377.310546875 ;
vmfp("c_Chem","a_FoodProc","CHN") = 7951.20947265625 ;
vmfp("c_Chem","a_FoodProc","JPN") = 1728.37023925781 ;
vmfp("c_Chem","a_FoodProc","IND") = 1098.25866699219 ;
vmfp("c_Chem","a_FoodProc","SSA") = 1834.67639160156 ;
vmfp("c_Chem","a_FoodProc","ROW") = 28619.501953125 ;
vmfp("c_Chem","a_Energy","USA") = 2768.19384765625 ;
vmfp("c_Chem","a_Energy","EU_28") = 7185.25 ;
vmfp("c_Chem","a_Energy","CHN") = 3142.7998046875 ;
vmfp("c_Chem","a_Energy","JPN") = 236.864715576172 ;
vmfp("c_Chem","a_Energy","IND") = 4539.9091796875 ;
vmfp("c_Chem","a_Energy","SSA") = 1411.25646972656 ;
vmfp("c_Chem","a_Energy","ROW") = 15436.4677734375 ;
vmfp("c_Chem","a_Textiles","USA") = 2795.10180664062 ;
vmfp("c_Chem","a_Textiles","EU_28") = 17754.23828125 ;
vmfp("c_Chem","a_Textiles","CHN") = 14602.6845703125 ;
vmfp("c_Chem","a_Textiles","JPN") = 903.182495117188 ;
vmfp("c_Chem","a_Textiles","IND") = 4454.3857421875 ;
vmfp("c_Chem","a_Textiles","SSA") = 710.602966308594 ;
vmfp("c_Chem","a_Textiles","ROW") = 19947.16796875 ;
vmfp("c_Chem","a_Chem","USA") = 59066.78125 ;
vmfp("c_Chem","a_Chem","EU_28") = 288365.6875 ;
vmfp("c_Chem","a_Chem","CHN") = 110698.078125 ;
vmfp("c_Chem","a_Chem","JPN") = 30619.701171875 ;
vmfp("c_Chem","a_Chem","IND") = 25307.767578125 ;
vmfp("c_Chem","a_Chem","SSA") = 5589.4716796875 ;
vmfp("c_Chem","a_Chem","ROW") = 252176.09375 ;
vmfp("c_Chem","a_Manuf","USA") = 30087.162109375 ;
vmfp("c_Chem","a_Manuf","EU_28") = 111288.3046875 ;
vmfp("c_Chem","a_Manuf","CHN") = 63231.49609375 ;
vmfp("c_Chem","a_Manuf","JPN") = 11940.4140625 ;
vmfp("c_Chem","a_Manuf","IND") = 6066.15966796875 ;
vmfp("c_Chem","a_Manuf","SSA") = 4941.61328125 ;
vmfp("c_Chem","a_Manuf","ROW") = 132865.015625 ;
vmfp("c_Chem","a_ForestFish","USA") = 956.339660644531 ;
vmfp("c_Chem","a_ForestFish","EU_28") = 893.058227539062 ;
vmfp("c_Chem","a_ForestFish","CHN") = 652.244262695312 ;
vmfp("c_Chem","a_ForestFish","JPN") = 74.6329956054688 ;
vmfp("c_Chem","a_ForestFish","IND") = 16.8617858886719 ;
vmfp("c_Chem","a_ForestFish","SSA") = 223.122024536133 ;
vmfp("c_Chem","a_ForestFish","ROW") = 1716.37438964844 ;
vmfp("c_Chem","a_Svces","USA") = 65880.4375 ;
vmfp("c_Chem","a_Svces","EU_28") = 221414.65625 ;
vmfp("c_Chem","a_Svces","CHN") = 63539.07421875 ;
vmfp("c_Chem","a_Svces","JPN") = 30261.490234375 ;
vmfp("c_Chem","a_Svces","IND") = 9972.2197265625 ;
vmfp("c_Chem","a_Svces","SSA") = 10762.4990234375 ;
vmfp("c_Chem","a_Svces","ROW") = 167546.671875 ;
vmfp("c_Manuf","a_Rice","USA") = 38.1628723144531 ;
vmfp("c_Manuf","a_Rice","EU_28") = 136.954086303711 ;
vmfp("c_Manuf","a_Rice","CHN") = 193.422714233398 ;
vmfp("c_Manuf","a_Rice","JPN") = 20.3193473815918 ;
vmfp("c_Manuf","a_Rice","IND") = 37.7223625183105 ;
vmfp("c_Manuf","a_Rice","SSA") = 120.73607635498 ;
vmfp("c_Manuf","a_Rice","ROW") = 1501.47717285156 ;
vmfp("c_Manuf","a_Crops","USA") = 1701.69213867188 ;
vmfp("c_Manuf","a_Crops","EU_28") = 2906.25415039062 ;
vmfp("c_Manuf","a_Crops","CHN") = 1065.58703613281 ;
vmfp("c_Manuf","a_Crops","JPN") = 84.3092498779297 ;
vmfp("c_Manuf","a_Crops","IND") = 19.8619842529297 ;
vmfp("c_Manuf","a_Crops","SSA") = 2868.95288085938 ;
vmfp("c_Manuf","a_Crops","ROW") = 7684.98095703125 ;
vmfp("c_Manuf","a_Livestock","USA") = 1100.2109375 ;
vmfp("c_Manuf","a_Livestock","EU_28") = 1574.07641601562 ;
vmfp("c_Manuf","a_Livestock","CHN") = 302.98291015625 ;
vmfp("c_Manuf","a_Livestock","JPN") = 44.2393836975098 ;
vmfp("c_Manuf","a_Livestock","IND") = 107.702049255371 ;
vmfp("c_Manuf","a_Livestock","SSA") = 461.396087646484 ;
vmfp("c_Manuf","a_Livestock","ROW") = 4511.6240234375 ;
vmfp("c_Manuf","a_FoodProc","USA") = 12506.6201171875 ;
vmfp("c_Manuf","a_FoodProc","EU_28") = 23642.98828125 ;
vmfp("c_Manuf","a_FoodProc","CHN") = 3964.98046875 ;
vmfp("c_Manuf","a_FoodProc","JPN") = 1094.23388671875 ;
vmfp("c_Manuf","a_FoodProc","IND") = 377.202758789062 ;
vmfp("c_Manuf","a_FoodProc","SSA") = 3632.64721679688 ;
vmfp("c_Manuf","a_FoodProc","ROW") = 25849.1953125 ;
vmfp("c_Manuf","a_Energy","USA") = 7060.48046875 ;
vmfp("c_Manuf","a_Energy","EU_28") = 18282.7578125 ;
vmfp("c_Manuf","a_Energy","CHN") = 8729.3251953125 ;
vmfp("c_Manuf","a_Energy","JPN") = 714.468627929688 ;
vmfp("c_Manuf","a_Energy","IND") = 4199.49951171875 ;
vmfp("c_Manuf","a_Energy","SSA") = 5501.78955078125 ;
vmfp("c_Manuf","a_Energy","ROW") = 73436.2265625 ;
vmfp("c_Manuf","a_Textiles","USA") = 1089.39367675781 ;
vmfp("c_Manuf","a_Textiles","EU_28") = 10012.94140625 ;
vmfp("c_Manuf","a_Textiles","CHN") = 3747.61694335938 ;
vmfp("c_Manuf","a_Textiles","JPN") = 172.558731079102 ;
vmfp("c_Manuf","a_Textiles","IND") = 689.733276367188 ;
vmfp("c_Manuf","a_Textiles","SSA") = 742.637634277344 ;
vmfp("c_Manuf","a_Textiles","ROW") = 12747.6513671875 ;
vmfp("c_Manuf","a_Chem","USA") = 15766.806640625 ;
vmfp("c_Manuf","a_Chem","EU_28") = 44470.5 ;
vmfp("c_Manuf","a_Chem","CHN") = 20505.1484375 ;
vmfp("c_Manuf","a_Chem","JPN") = 1823.53637695312 ;
vmfp("c_Manuf","a_Chem","IND") = 4297.59765625 ;
vmfp("c_Manuf","a_Chem","SSA") = 1230.11376953125 ;
vmfp("c_Manuf","a_Chem","ROW") = 34742.4375 ;
vmfp("c_Manuf","a_Manuf","USA") = 403418.9375 ;
vmfp("c_Manuf","a_Manuf","EU_28") = 1319997.5 ;
vmfp("c_Manuf","a_Manuf","CHN") = 754238.375 ;
vmfp("c_Manuf","a_Manuf","JPN") = 160904.75 ;
vmfp("c_Manuf","a_Manuf","IND") = 99641.578125 ;
vmfp("c_Manuf","a_Manuf","SSA") = 35234.6953125 ;
vmfp("c_Manuf","a_Manuf","ROW") = 1341635.125 ;
vmfp("c_Manuf","a_ForestFish","USA") = 368.184661865234 ;
vmfp("c_Manuf","a_ForestFish","EU_28") = 2073.31298828125 ;
vmfp("c_Manuf","a_ForestFish","CHN") = 844.029113769531 ;
vmfp("c_Manuf","a_ForestFish","JPN") = 168.222503662109 ;
vmfp("c_Manuf","a_ForestFish","IND") = 619.288024902344 ;
vmfp("c_Manuf","a_ForestFish","SSA") = 1166.20629882812 ;
vmfp("c_Manuf","a_ForestFish","ROW") = 3960.9287109375 ;
vmfp("c_Manuf","a_Svces","USA") = 253281.28125 ;
vmfp("c_Manuf","a_Svces","EU_28") = 508811.625 ;
vmfp("c_Manuf","a_Svces","CHN") = 207404.96875 ;
vmfp("c_Manuf","a_Svces","JPN") = 55344.16796875 ;
vmfp("c_Manuf","a_Svces","IND") = 38578.56640625 ;
vmfp("c_Manuf","a_Svces","SSA") = 44693.42578125 ;
vmfp("c_Manuf","a_Svces","ROW") = 631463.3125 ;
vmfp("c_ForestFish","a_Rice","USA") = 0.0543341599404812 ;
vmfp("c_ForestFish","a_Rice","EU_28") = 5.74452590942383 ;
vmfp("c_ForestFish","a_Rice","CHN") = 0.245071142911911 ;
vmfp("c_ForestFish","a_Rice","JPN") = 0.0428762957453728 ;
vmfp("c_ForestFish","a_Rice","IND") = 0.0189838223159313 ;
vmfp("c_ForestFish","a_Rice","SSA") = 0.0824873149394989 ;
vmfp("c_ForestFish","a_Rice","ROW") = 11.4143695831299 ;
vmfp("c_ForestFish","a_Crops","USA") = 4.23798608779907 ;
vmfp("c_ForestFish","a_Crops","EU_28") = 17.8238887786865 ;
vmfp("c_ForestFish","a_Crops","CHN") = 1.29367029666901 ;
vmfp("c_ForestFish","a_Crops","JPN") = 1.87896847724915 ;
vmfp("c_ForestFish","a_Crops","IND") = 0.0649145320057869 ;
vmfp("c_ForestFish","a_Crops","SSA") = 0.527112185955048 ;
vmfp("c_ForestFish","a_Crops","ROW") = 20.4067249298096 ;
vmfp("c_ForestFish","a_Livestock","USA") = 0.720655739307404 ;
vmfp("c_ForestFish","a_Livestock","EU_28") = 9.928297996521 ;
vmfp("c_ForestFish","a_Livestock","CHN") = 0.180956482887268 ;
vmfp("c_ForestFish","a_Livestock","JPN") = 0.0373947955667973 ;
vmfp("c_ForestFish","a_Livestock","IND") = 2.15646052360535 ;
vmfp("c_ForestFish","a_Livestock","SSA") = 1.78472459316254 ;
vmfp("c_ForestFish","a_Livestock","ROW") = 14.0598545074463 ;
vmfp("c_ForestFish","a_FoodProc","USA") = 1202.34484863281 ;
vmfp("c_ForestFish","a_FoodProc","EU_28") = 5109.38623046875 ;
vmfp("c_ForestFish","a_FoodProc","CHN") = 930.11376953125 ;
vmfp("c_ForestFish","a_FoodProc","JPN") = 1110.93481445312 ;
vmfp("c_ForestFish","a_FoodProc","IND") = 66.1771774291992 ;
vmfp("c_ForestFish","a_FoodProc","SSA") = 39.4431304931641 ;
vmfp("c_ForestFish","a_FoodProc","ROW") = 1356.97741699219 ;
vmfp("c_ForestFish","a_Energy","USA") = 1.28204941749573 ;
vmfp("c_ForestFish","a_Energy","EU_28") = 38.0605239868164 ;
vmfp("c_ForestFish","a_Energy","CHN") = 31.8111610412598 ;
vmfp("c_ForestFish","a_Energy","JPN") = 0.0648832097649574 ;
vmfp("c_ForestFish","a_Energy","IND") = 0.497310727834702 ;
vmfp("c_ForestFish","a_Energy","SSA") = 0.493426114320755 ;
vmfp("c_ForestFish","a_Energy","ROW") = 5.65912580490112 ;
vmfp("c_ForestFish","a_Textiles","USA") = 3.29221820831299 ;
vmfp("c_ForestFish","a_Textiles","EU_28") = 114.774490356445 ;
vmfp("c_ForestFish","a_Textiles","CHN") = 59.8849868774414 ;
vmfp("c_ForestFish","a_Textiles","JPN") = 1.72721838951111 ;
vmfp("c_ForestFish","a_Textiles","IND") = 1.63537585735321 ;
vmfp("c_ForestFish","a_Textiles","SSA") = 0.143519774079323 ;
vmfp("c_ForestFish","a_Textiles","ROW") = 5.16898107528687 ;
vmfp("c_ForestFish","a_Chem","USA") = 66.2577972412109 ;
vmfp("c_ForestFish","a_Chem","EU_28") = 381.962768554688 ;
vmfp("c_ForestFish","a_Chem","CHN") = 2009.21240234375 ;
vmfp("c_ForestFish","a_Chem","JPN") = 22.0139217376709 ;
vmfp("c_ForestFish","a_Chem","IND") = 120.300552368164 ;
vmfp("c_ForestFish","a_Chem","SSA") = 6.03926992416382 ;
vmfp("c_ForestFish","a_Chem","ROW") = 188.058654785156 ;
vmfp("c_ForestFish","a_Manuf","USA") = 209.426834106445 ;
vmfp("c_ForestFish","a_Manuf","EU_28") = 3809.33642578125 ;
vmfp("c_ForestFish","a_Manuf","CHN") = 5872.794921875 ;
vmfp("c_ForestFish","a_Manuf","JPN") = 683.596313476562 ;
vmfp("c_ForestFish","a_Manuf","IND") = 100.757759094238 ;
vmfp("c_ForestFish","a_Manuf","SSA") = 80.9342727661133 ;
vmfp("c_ForestFish","a_Manuf","ROW") = 1565.5400390625 ;
vmfp("c_ForestFish","a_ForestFish","USA") = 36.1214408874512 ;
vmfp("c_ForestFish","a_ForestFish","EU_28") = 916.91259765625 ;
vmfp("c_ForestFish","a_ForestFish","CHN") = 618.019958496094 ;
vmfp("c_ForestFish","a_ForestFish","JPN") = 192.511337280273 ;
vmfp("c_ForestFish","a_ForestFish","IND") = 5.47301006317139 ;
vmfp("c_ForestFish","a_ForestFish","SSA") = 24.3670864105225 ;
vmfp("c_ForestFish","a_ForestFish","ROW") = 1237.95178222656 ;
vmfp("c_ForestFish","a_Svces","USA") = 900.099304199219 ;
vmfp("c_ForestFish","a_Svces","EU_28") = 2868.251953125 ;
vmfp("c_ForestFish","a_Svces","CHN") = 3278.41455078125 ;
vmfp("c_ForestFish","a_Svces","JPN") = 484.335876464844 ;
vmfp("c_ForestFish","a_Svces","IND") = 680.51904296875 ;
vmfp("c_ForestFish","a_Svces","SSA") = 20.8574962615967 ;
vmfp("c_ForestFish","a_Svces","ROW") = 1577.87524414062 ;
vmfp("c_Svces","a_Rice","USA") = 26.4195556640625 ;
vmfp("c_Svces","a_Rice","EU_28") = 152.304656982422 ;
vmfp("c_Svces","a_Rice","CHN") = 256.363220214844 ;
vmfp("c_Svces","a_Rice","JPN") = 98.1952819824219 ;
vmfp("c_Svces","a_Rice","IND") = 138.238845825195 ;
vmfp("c_Svces","a_Rice","SSA") = 57.315113067627 ;
vmfp("c_Svces","a_Rice","ROW") = 1089.64379882812 ;
vmfp("c_Svces","a_Crops","USA") = 791.31494140625 ;
vmfp("c_Svces","a_Crops","EU_28") = 4315.25341796875 ;
vmfp("c_Svces","a_Crops","CHN") = 1087.78515625 ;
vmfp("c_Svces","a_Crops","JPN") = 114.409927368164 ;
vmfp("c_Svces","a_Crops","IND") = 85.5951766967773 ;
vmfp("c_Svces","a_Crops","SSA") = 1454.04516601562 ;
vmfp("c_Svces","a_Crops","ROW") = 4855.841796875 ;
vmfp("c_Svces","a_Livestock","USA") = 559.642822265625 ;
vmfp("c_Svces","a_Livestock","EU_28") = 2068.2109375 ;
vmfp("c_Svces","a_Livestock","CHN") = 429.463226318359 ;
vmfp("c_Svces","a_Livestock","JPN") = 119.363662719727 ;
vmfp("c_Svces","a_Livestock","IND") = 121.458106994629 ;
vmfp("c_Svces","a_Livestock","SSA") = 372.654388427734 ;
vmfp("c_Svces","a_Livestock","ROW") = 2082.96850585938 ;
vmfp("c_Svces","a_FoodProc","USA") = 2739.5166015625 ;
vmfp("c_Svces","a_FoodProc","EU_28") = 38270.23828125 ;
vmfp("c_Svces","a_FoodProc","CHN") = 5270.86181640625 ;
vmfp("c_Svces","a_FoodProc","JPN") = 1331.22888183594 ;
vmfp("c_Svces","a_FoodProc","IND") = 550.695922851562 ;
vmfp("c_Svces","a_FoodProc","SSA") = 2012.40625 ;
vmfp("c_Svces","a_FoodProc","ROW") = 16258.3935546875 ;
vmfp("c_Svces","a_Energy","USA") = 2504.24169921875 ;
vmfp("c_Svces","a_Energy","EU_28") = 9605.416015625 ;
vmfp("c_Svces","a_Energy","CHN") = 3562.68212890625 ;
vmfp("c_Svces","a_Energy","JPN") = 2507.09741210938 ;
vmfp("c_Svces","a_Energy","IND") = 3762.3505859375 ;
vmfp("c_Svces","a_Energy","SSA") = 5930.787109375 ;
vmfp("c_Svces","a_Energy","ROW") = 35100.35546875 ;
vmfp("c_Svces","a_Textiles","USA") = 346.111846923828 ;
vmfp("c_Svces","a_Textiles","EU_28") = 10271.5419921875 ;
vmfp("c_Svces","a_Textiles","CHN") = 5236.10107421875 ;
vmfp("c_Svces","a_Textiles","JPN") = 196.174591064453 ;
vmfp("c_Svces","a_Textiles","IND") = 592.668640136719 ;
vmfp("c_Svces","a_Textiles","SSA") = 614.839965820312 ;
vmfp("c_Svces","a_Textiles","ROW") = 6970.6171875 ;
vmfp("c_Svces","a_Chem","USA") = 3573.2783203125 ;
vmfp("c_Svces","a_Chem","EU_28") = 63116.82421875 ;
vmfp("c_Svces","a_Chem","CHN") = 8171.24462890625 ;
vmfp("c_Svces","a_Chem","JPN") = 2177.81127929688 ;
vmfp("c_Svces","a_Chem","IND") = 1262.13891601562 ;
vmfp("c_Svces","a_Chem","SSA") = 800.735900878906 ;
vmfp("c_Svces","a_Chem","ROW") = 26512.58203125 ;
vmfp("c_Svces","a_Manuf","USA") = 17782.990234375 ;
vmfp("c_Svces","a_Manuf","EU_28") = 168193.90625 ;
vmfp("c_Svces","a_Manuf","CHN") = 36543.1875 ;
vmfp("c_Svces","a_Manuf","JPN") = 10430.2421875 ;
vmfp("c_Svces","a_Manuf","IND") = 5357.26904296875 ;
vmfp("c_Svces","a_Manuf","SSA") = 5348.52392578125 ;
vmfp("c_Svces","a_Manuf","ROW") = 79730.109375 ;
vmfp("c_Svces","a_ForestFish","USA") = 233.197448730469 ;
vmfp("c_Svces","a_ForestFish","EU_28") = 1637.89172363281 ;
vmfp("c_Svces","a_ForestFish","CHN") = 1036.87438964844 ;
vmfp("c_Svces","a_ForestFish","JPN") = 82.3395614624023 ;
vmfp("c_Svces","a_ForestFish","IND") = 352.355194091797 ;
vmfp("c_Svces","a_ForestFish","SSA") = 930.788330078125 ;
vmfp("c_Svces","a_ForestFish","ROW") = 2685.142578125 ;
vmfp("c_Svces","a_Svces","USA") = 224986.59375 ;
vmfp("c_Svces","a_Svces","EU_28") = 1214330.25 ;
vmfp("c_Svces","a_Svces","CHN") = 136314.921875 ;
vmfp("c_Svces","a_Svces","JPN") = 64284.6328125 ;
vmfp("c_Svces","a_Svces","IND") = 23725.515625 ;
vmfp("c_Svces","a_Svces","SSA") = 31571.71484375 ;
vmfp("c_Svces","a_Svces","ROW") = 656804.1875 ;
parameter vdpb(*,*) ;
vdpb("c_Rice","USA") = 4570.59619140625 ;
vdpb("c_Rice","EU_28") = 2580.30834960938 ;
vdpb("c_Rice","CHN") = 53829.125 ;
vdpb("c_Rice","JPN") = 12246.4326171875 ;
vdpb("c_Rice","IND") = 14031.0771484375 ;
vdpb("c_Rice","SSA") = 10944.7724609375 ;
vdpb("c_Rice","ROW") = 164793.546875 ;
vdpb("c_Crops","USA") = 45801.00390625 ;
vdpb("c_Crops","EU_28") = 63108.25 ;
vdpb("c_Crops","CHN") = 191434.90625 ;
vdpb("c_Crops","JPN") = 19088.62109375 ;
vdpb("c_Crops","IND") = 132760.65625 ;
vdpb("c_Crops","SSA") = 134715.859375 ;
vdpb("c_Crops","ROW") = 251105.078125 ;
vdpb("c_Livestock","USA") = 14565.505859375 ;
vdpb("c_Livestock","EU_28") = 24576.38671875 ;
vdpb("c_Livestock","CHN") = 114475.6484375 ;
vdpb("c_Livestock","JPN") = 6550.10205078125 ;
vdpb("c_Livestock","IND") = 57831.71875 ;
vdpb("c_Livestock","SSA") = 28199.4453125 ;
vdpb("c_Livestock","ROW") = 128444.2421875 ;
vdpb("c_FoodProc","USA") = 633797.9375 ;
vdpb("c_FoodProc","EU_28") = 533944.8125 ;
vdpb("c_FoodProc","CHN") = 752912.0625 ;
vdpb("c_FoodProc","JPN") = 200819.46875 ;
vdpb("c_FoodProc","IND") = 160763.28125 ;
vdpb("c_FoodProc","SSA") = 208114.640625 ;
vdpb("c_FoodProc","ROW") = 1200112.25 ;
vdpb("c_Energy","USA") = 239341.640625 ;
vdpb("c_Energy","EU_28") = 184299.953125 ;
vdpb("c_Energy","CHN") = 153850.40625 ;
vdpb("c_Energy","JPN") = 71551.890625 ;
vdpb("c_Energy","IND") = 90954.1875 ;
vdpb("c_Energy","SSA") = 24006.892578125 ;
vdpb("c_Energy","ROW") = 475345.9375 ;
vdpb("c_Textiles","USA") = 30215.48046875 ;
vdpb("c_Textiles","EU_28") = 60681.63671875 ;
vdpb("c_Textiles","CHN") = 189531.953125 ;
vdpb("c_Textiles","JPN") = 15577.275390625 ;
vdpb("c_Textiles","IND") = 86027.140625 ;
vdpb("c_Textiles","SSA") = 28712.212890625 ;
vdpb("c_Textiles","ROW") = 279773.8125 ;
vdpb("c_Chem","USA") = 162893.65625 ;
vdpb("c_Chem","EU_28") = 63160.2890625 ;
vdpb("c_Chem","CHN") = 95128.6328125 ;
vdpb("c_Chem","JPN") = 26964.37890625 ;
vdpb("c_Chem","IND") = 26593.05078125 ;
vdpb("c_Chem","SSA") = 14418.8203125 ;
vdpb("c_Chem","ROW") = 200212.84375 ;
vdpb("c_Manuf","USA") = 518719.5 ;
vdpb("c_Manuf","EU_28") = 256059.703125 ;
vdpb("c_Manuf","CHN") = 333492.4375 ;
vdpb("c_Manuf","JPN") = 109757.65625 ;
vdpb("c_Manuf","IND") = 78289.9921875 ;
vdpb("c_Manuf","SSA") = 38941.86328125 ;
vdpb("c_Manuf","ROW") = 519647.75 ;
vdpb("c_ForestFish","USA") = 3927.19165039062 ;
vdpb("c_ForestFish","EU_28") = 14389.796875 ;
vdpb("c_ForestFish","CHN") = 71678.171875 ;
vdpb("c_ForestFish","JPN") = 4060.1591796875 ;
vdpb("c_ForestFish","IND") = 38044.109375 ;
vdpb("c_ForestFish","SSA") = 43292.98046875 ;
vdpb("c_ForestFish","ROW") = 88406.2578125 ;
vdpb("c_Svces","USA") = 10502550 ;
vdpb("c_Svces","EU_28") = 6331369.5 ;
vdpb("c_Svces","CHN") = 2267669 ;
vdpb("c_Svces","JPN") = 2019539 ;
vdpb("c_Svces","IND") = 848619.25 ;
vdpb("c_Svces","SSA") = 422321.1875 ;
vdpb("c_Svces","ROW") = 7755806 ;
parameter vdpp(*,*) ;
vdpp("c_Rice","USA") = 4486.54931640625 ;
vdpp("c_Rice","EU_28") = 2863.28125 ;
vdpp("c_Rice","CHN") = 51861.85546875 ;
vdpp("c_Rice","JPN") = 12246.4326171875 ;
vdpp("c_Rice","IND") = 13851.4287109375 ;
vdpp("c_Rice","SSA") = 11067.93359375 ;
vdpp("c_Rice","ROW") = 168019.296875 ;
vdpp("c_Crops","USA") = 45741.6328125 ;
vdpp("c_Crops","EU_28") = 68111.78125 ;
vdpp("c_Crops","CHN") = 174254.9375 ;
vdpp("c_Crops","JPN") = 19088.62109375 ;
vdpp("c_Crops","IND") = 132292.203125 ;
vdpp("c_Crops","SSA") = 135294.453125 ;
vdpp("c_Crops","ROW") = 253048.734375 ;
vdpp("c_Livestock","USA") = 14554.0478515625 ;
vdpp("c_Livestock","EU_28") = 26979.87890625 ;
vdpp("c_Livestock","CHN") = 114412.140625 ;
vdpp("c_Livestock","JPN") = 6550.10205078125 ;
vdpp("c_Livestock","IND") = 57313.5546875 ;
vdpp("c_Livestock","SSA") = 28623.650390625 ;
vdpp("c_Livestock","ROW") = 130148.921875 ;
vdpp("c_FoodProc","USA") = 632445.75 ;
vdpp("c_FoodProc","EU_28") = 709820.9375 ;
vdpp("c_FoodProc","CHN") = 823222.8125 ;
vdpp("c_FoodProc","JPN") = 200819.46875 ;
vdpp("c_FoodProc","IND") = 178233.046875 ;
vdpp("c_FoodProc","SSA") = 215237.34375 ;
vdpp("c_FoodProc","ROW") = 1311064.75 ;
vdpp("c_Energy","USA") = 276856.53125 ;
vdpp("c_Energy","EU_28") = 382142.34375 ;
vdpp("c_Energy","CHN") = 148529.5625 ;
vdpp("c_Energy","JPN") = 102323.4140625 ;
vdpp("c_Energy","IND") = 85858.3359375 ;
vdpp("c_Energy","SSA") = 23259.623046875 ;
vdpp("c_Energy","ROW") = 494685.0625 ;
vdpp("c_Textiles","USA") = 30339.3359375 ;
vdpp("c_Textiles","EU_28") = 82716.203125 ;
vdpp("c_Textiles","CHN") = 188896.765625 ;
vdpp("c_Textiles","JPN") = 15577.275390625 ;
vdpp("c_Textiles","IND") = 91643.390625 ;
vdpp("c_Textiles","SSA") = 30218.66796875 ;
vdpp("c_Textiles","ROW") = 294390.21875 ;
vdpp("c_Chem","USA") = 177229.015625 ;
vdpp("c_Chem","EU_28") = 81409.90625 ;
vdpp("c_Chem","CHN") = 106626.296875 ;
vdpp("c_Chem","JPN") = 26964.37890625 ;
vdpp("c_Chem","IND") = 30574.966796875 ;
vdpp("c_Chem","SSA") = 15520.505859375 ;
vdpp("c_Chem","ROW") = 220263.703125 ;
vdpp("c_Manuf","USA") = 489792.25 ;
vdpp("c_Manuf","EU_28") = 316494.15625 ;
vdpp("c_Manuf","CHN") = 357099.28125 ;
vdpp("c_Manuf","JPN") = 109757.65625 ;
vdpp("c_Manuf","IND") = 84974.515625 ;
vdpp("c_Manuf","SSA") = 41034.234375 ;
vdpp("c_Manuf","ROW") = 574081.0625 ;
vdpp("c_ForestFish","USA") = 3947.14599609375 ;
vdpp("c_ForestFish","EU_28") = 15891.1015625 ;
vdpp("c_ForestFish","CHN") = 71978.75 ;
vdpp("c_ForestFish","JPN") = 4060.1591796875 ;
vdpp("c_ForestFish","IND") = 38263.03125 ;
vdpp("c_ForestFish","SSA") = 44629.41796875 ;
vdpp("c_ForestFish","ROW") = 89160.4296875 ;
vdpp("c_Svces","USA") = 10726112 ;
vdpp("c_Svces","EU_28") = 6579813.5 ;
vdpp("c_Svces","CHN") = 2443109.5 ;
vdpp("c_Svces","JPN") = 2019539 ;
vdpp("c_Svces","IND") = 861426.3125 ;
vdpp("c_Svces","SSA") = 435631.28125 ;
vdpp("c_Svces","ROW") = 7984025.5 ;
parameter vmpb(*,*) ;
vmpb("c_Rice","USA") = 10.1704502105713 ;
vmpb("c_Rice","EU_28") = 1203.56982421875 ;
vmpb("c_Rice","CHN") = 722.950988769531 ;
vmpb("c_Rice","JPN") = 696.863525390625 ;
vmpb("c_Rice","IND") = 11.0292911529541 ;
vmpb("c_Rice","SSA") = 4989.609375 ;
vmpb("c_Rice","ROW") = 5316.5732421875 ;
vmpb("c_Crops","USA") = 22603.900390625 ;
vmpb("c_Crops","EU_28") = 57276.078125 ;
vmpb("c_Crops","CHN") = 17458.091796875 ;
vmpb("c_Crops","JPN") = 4383.42919921875 ;
vmpb("c_Crops","IND") = 7656.3525390625 ;
vmpb("c_Crops","SSA") = 6701.90185546875 ;
vmpb("c_Crops","ROW") = 62435.80859375 ;
vmpb("c_Livestock","USA") = 421.923736572266 ;
vmpb("c_Livestock","EU_28") = 2014.98803710938 ;
vmpb("c_Livestock","CHN") = 2097.98657226562 ;
vmpb("c_Livestock","JPN") = 640.025817871094 ;
vmpb("c_Livestock","IND") = 56.111141204834 ;
vmpb("c_Livestock","SSA") = 378.166290283203 ;
vmpb("c_Livestock","ROW") = 7063.58251953125 ;
vmpb("c_FoodProc","USA") = 67913.3828125 ;
vmpb("c_FoodProc","EU_28") = 215977.984375 ;
vmpb("c_FoodProc","CHN") = 29696.275390625 ;
vmpb("c_FoodProc","JPN") = 34152.73828125 ;
vmpb("c_FoodProc","IND") = 16134.796875 ;
vmpb("c_FoodProc","SSA") = 29159.2265625 ;
vmpb("c_FoodProc","ROW") = 222347.875 ;
vmpb("c_Energy","USA") = 17448.66015625 ;
vmpb("c_Energy","EU_28") = 38157.03515625 ;
vmpb("c_Energy","CHN") = 7418.68603515625 ;
vmpb("c_Energy","JPN") = 6532.2666015625 ;
vmpb("c_Energy","IND") = 4386.3173828125 ;
vmpb("c_Energy","SSA") = 10311.26953125 ;
vmpb("c_Energy","ROW") = 53988.640625 ;
vmpb("c_Textiles","USA") = 158664.171875 ;
vmpb("c_Textiles","EU_28") = 241355.28125 ;
vmpb("c_Textiles","CHN") = 9272.580078125 ;
vmpb("c_Textiles","JPN") = 33862.140625 ;
vmpb("c_Textiles","IND") = 7303.91650390625 ;
vmpb("c_Textiles","SSA") = 26169.609375 ;
vmpb("c_Textiles","ROW") = 197697.75 ;
vmpb("c_Chem","USA") = 105177.5546875 ;
vmpb("c_Chem","EU_28") = 121656.203125 ;
vmpb("c_Chem","CHN") = 10632.052734375 ;
vmpb("c_Chem","JPN") = 8461.6640625 ;
vmpb("c_Chem","IND") = 4542.619140625 ;
vmpb("c_Chem","SSA") = 23238.44140625 ;
vmpb("c_Chem","ROW") = 152459.984375 ;
vmpb("c_Manuf","USA") = 352315.28125 ;
vmpb("c_Manuf","EU_28") = 439152.65625 ;
vmpb("c_Manuf","CHN") = 121758.5859375 ;
vmpb("c_Manuf","JPN") = 38983.921875 ;
vmpb("c_Manuf","IND") = 14936.86328125 ;
vmpb("c_Manuf","SSA") = 36299.828125 ;
vmpb("c_Manuf","ROW") = 551682 ;
vmpb("c_ForestFish","USA") = 1067.34313964844 ;
vmpb("c_ForestFish","EU_28") = 5993.33154296875 ;
vmpb("c_ForestFish","CHN") = 1023.12097167969 ;
vmpb("c_ForestFish","JPN") = 559.869689941406 ;
vmpb("c_ForestFish","IND") = 455.596435546875 ;
vmpb("c_ForestFish","SSA") = 144.379119873047 ;
vmpb("c_ForestFish","ROW") = 4121.69482421875 ;
vmpb("c_Svces","USA") = 178576.3125 ;
vmpb("c_Svces","EU_28") = 159395.5 ;
vmpb("c_Svces","CHN") = 78959.984375 ;
vmpb("c_Svces","JPN") = 46168.7109375 ;
vmpb("c_Svces","IND") = 32459.48046875 ;
vmpb("c_Svces","SSA") = 29291.50390625 ;
vmpb("c_Svces","ROW") = 459474.84375 ;
parameter vmpp(*,*) ;
vmpp("c_Rice","USA") = 10.077618598938 ;
vmpp("c_Rice","EU_28") = 1439.26916503906 ;
vmpp("c_Rice","CHN") = 865.97265625 ;
vmpp("c_Rice","JPN") = 739.337280273438 ;
vmpp("c_Rice","IND") = 11.0292911529541 ;
vmpp("c_Rice","SSA") = 5530.7041015625 ;
vmpp("c_Rice","ROW") = 5489.1552734375 ;
vmpp("c_Crops","USA") = 22560.365234375 ;
vmpp("c_Crops","EU_28") = 63152.49609375 ;
vmpp("c_Crops","CHN") = 21641.52734375 ;
vmpp("c_Crops","JPN") = 4625.54345703125 ;
vmpp("c_Crops","IND") = 7656.3525390625 ;
vmpp("c_Crops","SSA") = 6884.33349609375 ;
vmpp("c_Crops","ROW") = 63811.71875 ;
vmpp("c_Livestock","USA") = 419.934295654297 ;
vmpp("c_Livestock","EU_28") = 2253.8056640625 ;
vmpp("c_Livestock","CHN") = 2516.67333984375 ;
vmpp("c_Livestock","JPN") = 683.805114746094 ;
vmpp("c_Livestock","IND") = 56.111141204834 ;
vmpp("c_Livestock","SSA") = 393.145385742188 ;
vmpp("c_Livestock","ROW") = 7303.42626953125 ;
vmpp("c_FoodProc","USA") = 67819.90625 ;
vmpp("c_FoodProc","EU_28") = 287502.5 ;
vmpp("c_FoodProc","CHN") = 36144.09375 ;
vmpp("c_FoodProc","JPN") = 46980.3125 ;
vmpp("c_FoodProc","IND") = 16134.7978515625 ;
vmpp("c_FoodProc","SSA") = 31724.44921875 ;
vmpp("c_FoodProc","ROW") = 250939.328125 ;
vmpp("c_Energy","USA") = 24221.681640625 ;
vmpp("c_Energy","EU_28") = 87695.0859375 ;
vmpp("c_Energy","CHN") = 7338.15283203125 ;
vmpp("c_Energy","JPN") = 11121.890625 ;
vmpp("c_Energy","IND") = 3985.43432617188 ;
vmpp("c_Energy","SSA") = 10520.39453125 ;
vmpp("c_Energy","ROW") = 73738.2421875 ;
vmpp("c_Textiles","USA") = 158534.875 ;
vmpp("c_Textiles","EU_28") = 333366.9375 ;
vmpp("c_Textiles","CHN") = 11724.482421875 ;
vmpp("c_Textiles","JPN") = 35752.62890625 ;
vmpp("c_Textiles","IND") = 7303.916015625 ;
vmpp("c_Textiles","SSA") = 29422.84375 ;
vmpp("c_Textiles","ROW") = 215958.59375 ;
vmpp("c_Chem","USA") = 116431.828125 ;
vmpp("c_Chem","EU_28") = 156458.484375 ;
vmpp("c_Chem","CHN") = 14683.48046875 ;
vmpp("c_Chem","JPN") = 9271.783203125 ;
vmpp("c_Chem","IND") = 4542.619140625 ;
vmpp("c_Chem","SSA") = 24912.689453125 ;
vmpp("c_Chem","ROW") = 164922.453125 ;
vmpp("c_Manuf","USA") = 356344.125 ;
vmpp("c_Manuf","EU_28") = 552675.75 ;
vmpp("c_Manuf","CHN") = 151404.28125 ;
vmpp("c_Manuf","JPN") = 41594.484375 ;
vmpp("c_Manuf","IND") = 14936.86328125 ;
vmpp("c_Manuf","SSA") = 38986.21875 ;
vmpp("c_Manuf","ROW") = 602083.875 ;
vmpp("c_ForestFish","USA") = 1074.78076171875 ;
vmpp("c_ForestFish","EU_28") = 6857.8017578125 ;
vmpp("c_ForestFish","CHN") = 1240.4150390625 ;
vmpp("c_ForestFish","JPN") = 616.31201171875 ;
vmpp("c_ForestFish","IND") = 455.596466064453 ;
vmpp("c_ForestFish","SSA") = 150.91032409668 ;
vmpp("c_ForestFish","ROW") = 4251.0888671875 ;
vmpp("c_Svces","USA") = 183644.734375 ;
vmpp("c_Svces","EU_28") = 169805.328125 ;
vmpp("c_Svces","CHN") = 93180.6640625 ;
vmpp("c_Svces","JPN") = 46185.9453125 ;
vmpp("c_Svces","IND") = 32455.9921875 ;
vmpp("c_Svces","SSA") = 30076.900390625 ;
vmpp("c_Svces","ROW") = 474379.53125 ;
parameter vdgb(*,*) ;
vdgb("c_Rice","USA") = 0.122278079390526 ;
vdgb("c_Rice","EU_28") = 4.3306040763855 ;
vdgb("c_Rice","CHN") = 0.194914966821671 ;
vdgb("c_Rice","JPN") = 1.38688278198242 ;
vdgb("c_Rice","IND") = 0.0440977476537228 ;
vdgb("c_Rice","SSA") = 0.0924253463745117 ;
vdgb("c_Rice","ROW") = 1498.05126953125 ;
vdgb("c_Crops","USA") = 440.466827392578 ;
vdgb("c_Crops","EU_28") = 1956.25915527344 ;
vdgb("c_Crops","CHN") = 938.702270507812 ;
vdgb("c_Crops","JPN") = 322.84765625 ;
vdgb("c_Crops","IND") = 72.5939559936523 ;
vdgb("c_Crops","SSA") = 83.8990325927734 ;
vdgb("c_Crops","ROW") = 1803.25439453125 ;
vdgb("c_Livestock","USA") = 160.213958740234 ;
vdgb("c_Livestock","EU_28") = 1005.74237060547 ;
vdgb("c_Livestock","CHN") = 171.848449707031 ;
vdgb("c_Livestock","JPN") = 203.782409667969 ;
vdgb("c_Livestock","IND") = 17.7355289459229 ;
vdgb("c_Livestock","SSA") = 98.3658905029297 ;
vdgb("c_Livestock","ROW") = 1377.15563964844 ;
vdgb("c_FoodProc","USA") = 17.7397632598877 ;
vdgb("c_FoodProc","EU_28") = 911.996948242188 ;
vdgb("c_FoodProc","CHN") = 13.0909109115601 ;
vdgb("c_FoodProc","JPN") = 1928.00537109375 ;
vdgb("c_FoodProc","IND") = 1.84543526172638 ;
vdgb("c_FoodProc","SSA") = 47.0181045532227 ;
vdgb("c_FoodProc","ROW") = 2220.0869140625 ;
vdgb("c_Textiles","USA") = 13.012131690979 ;
vdgb("c_Textiles","EU_28") = 135.205337524414 ;
vdgb("c_Textiles","CHN") = 9.50399494171143 ;
vdgb("c_Textiles","JPN") = 22.3695220947266 ;
vdgb("c_Textiles","IND") = 1.37508165836334 ;
vdgb("c_Textiles","SSA") = 1.32450270652771 ;
vdgb("c_Textiles","ROW") = 2343.05224609375 ;
vdgb("c_Chem","USA") = 33.9402236938477 ;
vdgb("c_Chem","EU_28") = 28507.701171875 ;
vdgb("c_Chem","CHN") = 24.1679363250732 ;
vdgb("c_Chem","JPN") = 106.64786529541 ;
vdgb("c_Chem","IND") = 3.5134379863739 ;
vdgb("c_Chem","SSA") = 128.838043212891 ;
vdgb("c_Chem","ROW") = 12031.8212890625 ;
vdgb("c_Manuf","USA") = 358.937469482422 ;
vdgb("c_Manuf","EU_28") = 7211.5537109375 ;
vdgb("c_Manuf","CHN") = 263.347320556641 ;
vdgb("c_Manuf","JPN") = 188.895095825195 ;
vdgb("c_Manuf","IND") = 38.2192153930664 ;
vdgb("c_Manuf","SSA") = 42.7736053466797 ;
vdgb("c_Manuf","ROW") = 44480.0078125 ;
vdgb("c_ForestFish","USA") = 405.227020263672 ;
vdgb("c_ForestFish","EU_28") = 1396.64501953125 ;
vdgb("c_ForestFish","CHN") = 304.201080322266 ;
vdgb("c_ForestFish","JPN") = 144.072387695312 ;
vdgb("c_ForestFish","IND") = 44.1970634460449 ;
vdgb("c_ForestFish","SSA") = 28.1244373321533 ;
vdgb("c_ForestFish","ROW") = 1052.85717773438 ;
vdgb("c_Svces","USA") = 2543765 ;
vdgb("c_Svces","EU_28") = 3504409.5 ;
vdgb("c_Svces","CHN") = 1978844.125 ;
vdgb("c_Svces","JPN") = 957869.0625 ;
vdgb("c_Svces","IND") = 299009.65625 ;
vdgb("c_Svces","SSA") = 209029.640625 ;
vdgb("c_Svces","ROW") = 3529746 ;
parameter vdgp(*,*) ;
vdgp("c_Rice","USA") = 0.122278079390526 ;
vdgp("c_Rice","EU_28") = 4.34444761276245 ;
vdgp("c_Rice","CHN") = 0.194914966821671 ;
vdgp("c_Rice","JPN") = 1.38688278198242 ;
vdgp("c_Rice","IND") = 0.0440977476537228 ;
vdgp("c_Rice","SSA") = 0.0931329727172852 ;
vdgp("c_Rice","ROW") = 1503.77001953125 ;
vdgp("c_Crops","USA") = 440.466827392578 ;
vdgp("c_Crops","EU_28") = 1976.63317871094 ;
vdgp("c_Crops","CHN") = 938.702270507812 ;
vdgp("c_Crops","JPN") = 322.84765625 ;
vdgp("c_Crops","IND") = 72.5939559936523 ;
vdgp("c_Crops","SSA") = 83.9049072265625 ;
vdgp("c_Crops","ROW") = 1820.10803222656 ;
vdgp("c_Livestock","USA") = 160.213958740234 ;
vdgp("c_Livestock","EU_28") = 1030.60046386719 ;
vdgp("c_Livestock","CHN") = 171.848449707031 ;
vdgp("c_Livestock","JPN") = 203.782409667969 ;
vdgp("c_Livestock","IND") = 17.7355289459229 ;
vdgp("c_Livestock","SSA") = 98.4823455810547 ;
vdgp("c_Livestock","ROW") = 1392.96459960938 ;
vdgp("c_FoodProc","USA") = 17.7397632598877 ;
vdgp("c_FoodProc","EU_28") = 1006.21954345703 ;
vdgp("c_FoodProc","CHN") = 13.0909109115601 ;
vdgp("c_FoodProc","JPN") = 1928.00537109375 ;
vdgp("c_FoodProc","IND") = 1.84543526172638 ;
vdgp("c_FoodProc","SSA") = 49.5471382141113 ;
vdgp("c_FoodProc","ROW") = 2273.849609375 ;
vdgp("c_Textiles","USA") = 13.012131690979 ;
vdgp("c_Textiles","EU_28") = 156.136047363281 ;
vdgp("c_Textiles","CHN") = 9.50399494171143 ;
vdgp("c_Textiles","JPN") = 22.3695220947266 ;
vdgp("c_Textiles","IND") = 1.37508165836334 ;
vdgp("c_Textiles","SSA") = 1.35074543952942 ;
vdgp("c_Textiles","ROW") = 2410.14526367188 ;
vdgp("c_Chem","USA") = 33.9402236938477 ;
vdgp("c_Chem","EU_28") = 33012.3515625 ;
vdgp("c_Chem","CHN") = 24.1679363250732 ;
vdgp("c_Chem","JPN") = 106.64786529541 ;
vdgp("c_Chem","IND") = 3.5134379863739 ;
vdgp("c_Chem","SSA") = 128.869232177734 ;
vdgp("c_Chem","ROW") = 12267.2705078125 ;
vdgp("c_Manuf","USA") = 358.937469482422 ;
vdgp("c_Manuf","EU_28") = 7789.701171875 ;
vdgp("c_Manuf","CHN") = 263.347320556641 ;
vdgp("c_Manuf","JPN") = 188.895095825195 ;
vdgp("c_Manuf","IND") = 38.2192153930664 ;
vdgp("c_Manuf","SSA") = 42.8947677612305 ;
vdgp("c_Manuf","ROW") = 44649.109375 ;
vdgp("c_ForestFish","USA") = 405.227020263672 ;
vdgp("c_ForestFish","EU_28") = 1436.70080566406 ;
vdgp("c_ForestFish","CHN") = 304.201080322266 ;
vdgp("c_ForestFish","JPN") = 144.072387695312 ;
vdgp("c_ForestFish","IND") = 44.1970634460449 ;
vdgp("c_ForestFish","SSA") = 28.1310214996338 ;
vdgp("c_ForestFish","ROW") = 1060.73791503906 ;
vdgp("c_Svces","USA") = 2729009.5 ;
vdgp("c_Svces","EU_28") = 3507268.75 ;
vdgp("c_Svces","CHN") = 2000814.25 ;
vdgp("c_Svces","JPN") = 957869.0625 ;
vdgp("c_Svces","IND") = 299901.21875 ;
vdgp("c_Svces","SSA") = 209661.046875 ;
vdgp("c_Svces","ROW") = 3549092.75 ;
parameter vmgb(*,*) ;
vmgb("c_Rice","USA") = 0.0132829425856471 ;
vmgb("c_Rice","EU_28") = 0.860502064228058 ;
vmgb("c_Rice","CHN") = 0.143287986516953 ;
vmgb("c_Rice","JPN") = 0.0630365610122681 ;
vmgb("c_Rice","IND") = 0.00106750125996768 ;
vmgb("c_Rice","SSA") = 0.10646153241396 ;
vmgb("c_Rice","ROW") = 32.5697441101074 ;
vmgb("c_Crops","USA") = 1.50671911239624 ;
vmgb("c_Crops","EU_28") = 3.27256178855896 ;
vmgb("c_Crops","CHN") = 0.656528949737549 ;
vmgb("c_Crops","JPN") = 1.06814706325531 ;
vmgb("c_Crops","IND") = 0.677688658237457 ;
vmgb("c_Crops","SSA") = 0.413689017295837 ;
vmgb("c_Crops","ROW") = 295.970764160156 ;
vmgb("c_Livestock","USA") = 0.106889896094799 ;
vmgb("c_Livestock","EU_28") = 0.367836266756058 ;
vmgb("c_Livestock","CHN") = 0.197441563010216 ;
vmgb("c_Livestock","JPN") = 0.414227098226547 ;
vmgb("c_Livestock","IND") = 0.0110352672636509 ;
vmgb("c_Livestock","SSA") = 0.0936359986662865 ;
vmgb("c_Livestock","ROW") = 30.2457103729248 ;
vmgb("c_FoodProc","USA") = 6.31466007232666 ;
vmgb("c_FoodProc","EU_28") = 193.337631225586 ;
vmgb("c_FoodProc","CHN") = 3.38964557647705 ;
vmgb("c_FoodProc","JPN") = 321.548828125 ;
vmgb("c_FoodProc","IND") = 0.244687646627426 ;
vmgb("c_FoodProc","SSA") = 21.9678821563721 ;
vmgb("c_FoodProc","ROW") = 1233.86169433594 ;
vmgb("c_Textiles","USA") = 5.81988143920898 ;
vmgb("c_Textiles","EU_28") = 269.899383544922 ;
vmgb("c_Textiles","CHN") = 3.71639466285706 ;
vmgb("c_Textiles","JPN") = 20.6909160614014 ;
vmgb("c_Textiles","IND") = 0.707882761955261 ;
vmgb("c_Textiles","SSA") = 3.23057579994202 ;
vmgb("c_Textiles","ROW") = 2198.71850585938 ;
vmgb("c_Chem","USA") = 24.0785903930664 ;
vmgb("c_Chem","EU_28") = 62426.15625 ;
vmgb("c_Chem","CHN") = 26.9196109771729 ;
vmgb("c_Chem","JPN") = 64.30712890625 ;
vmgb("c_Chem","IND") = 2.39831137657166 ;
vmgb("c_Chem","SSA") = 11.1127080917358 ;
vmgb("c_Chem","ROW") = 10496.6572265625 ;
vmgb("c_Manuf","USA") = 302.069793701172 ;
vmgb("c_Manuf","EU_28") = 5421.0986328125 ;
vmgb("c_Manuf","CHN") = 444.360809326172 ;
vmgb("c_Manuf","JPN") = 122.988784790039 ;
vmgb("c_Manuf","IND") = 44.7577743530273 ;
vmgb("c_Manuf","SSA") = 57.5465393066406 ;
vmgb("c_Manuf","ROW") = 53581.25 ;
vmgb("c_ForestFish","USA") = 0.000837104395031929 ;
vmgb("c_ForestFish","EU_28") = 3.08358263969421 ;
vmgb("c_ForestFish","CHN") = 0.0075100720860064 ;
vmgb("c_ForestFish","JPN") = 0.00891524646431208 ;
vmgb("c_ForestFish","IND") = 0.000327902496792376 ;
vmgb("c_ForestFish","SSA") = 0.00190711312461644 ;
vmgb("c_ForestFish","ROW") = 11.0392761230469 ;
vmgb("c_Svces","USA") = 19186.185546875 ;
vmgb("c_Svces","EU_28") = 7234.12353515625 ;
vmgb("c_Svces","CHN") = 33935.25390625 ;
vmgb("c_Svces","JPN") = 3493.63208007812 ;
vmgb("c_Svces","IND") = 4590.5673828125 ;
vmgb("c_Svces","SSA") = 5370.6494140625 ;
vmgb("c_Svces","ROW") = 54590.0078125 ;
parameter vmgp(*,*) ;
vmgp("c_Rice","USA") = 0.0132829425856471 ;
vmgp("c_Rice","EU_28") = 1.02874207496643 ;
vmgp("c_Rice","CHN") = 0.143287986516953 ;
vmgp("c_Rice","JPN") = 0.0630365610122681 ;
vmgp("c_Rice","IND") = 0.00106750125996768 ;
vmgp("c_Rice","SSA") = 0.106481581926346 ;
vmgp("c_Rice","ROW") = 32.6164283752441 ;
vmgp("c_Crops","USA") = 1.50671911239624 ;
vmgp("c_Crops","EU_28") = 3.46342253684998 ;
vmgp("c_Crops","CHN") = 0.656528949737549 ;
vmgp("c_Crops","JPN") = 1.06814706325531 ;
vmgp("c_Crops","IND") = 0.677688658237457 ;
vmgp("c_Crops","SSA") = 0.413869947195053 ;
vmgp("c_Crops","ROW") = 296.30126953125 ;
vmgp("c_Livestock","USA") = 0.106889896094799 ;
vmgp("c_Livestock","EU_28") = 0.381432414054871 ;
vmgp("c_Livestock","CHN") = 0.197441563010216 ;
vmgp("c_Livestock","JPN") = 0.414227098226547 ;
vmgp("c_Livestock","IND") = 0.0110352672636509 ;
vmgp("c_Livestock","SSA") = 0.0937363356351852 ;
vmgp("c_Livestock","ROW") = 30.3957462310791 ;
vmgp("c_FoodProc","USA") = 6.31466007232666 ;
vmgp("c_FoodProc","EU_28") = 215.535903930664 ;
vmgp("c_FoodProc","CHN") = 3.38964557647705 ;
vmgp("c_FoodProc","JPN") = 340.766235351562 ;
vmgp("c_FoodProc","IND") = 0.244687646627426 ;
vmgp("c_FoodProc","SSA") = 23.0802822113037 ;
vmgp("c_FoodProc","ROW") = 1235.76782226562 ;
vmgp("c_Textiles","USA") = 5.81988143920898 ;
vmgp("c_Textiles","EU_28") = 343.83203125 ;
vmgp("c_Textiles","CHN") = 3.71639466285706 ;
vmgp("c_Textiles","JPN") = 20.6909160614014 ;
vmgp("c_Textiles","IND") = 0.707882761955261 ;
vmgp("c_Textiles","SSA") = 3.2559666633606 ;
vmgp("c_Textiles","ROW") = 2213.43237304688 ;
vmgp("c_Chem","USA") = 24.0785903930664 ;
vmgp("c_Chem","EU_28") = 70365.78125 ;
vmgp("c_Chem","CHN") = 26.9196109771729 ;
vmgp("c_Chem","JPN") = 66.3334579467773 ;
vmgp("c_Chem","IND") = 2.39831137657166 ;
vmgp("c_Chem","SSA") = 11.4627838134766 ;
vmgp("c_Chem","ROW") = 10646.642578125 ;
vmgp("c_Manuf","USA") = 302.069793701172 ;
vmgp("c_Manuf","EU_28") = 6083.07421875 ;
vmgp("c_Manuf","CHN") = 444.360809326172 ;
vmgp("c_Manuf","JPN") = 125.297485351562 ;
vmgp("c_Manuf","IND") = 44.7577743530273 ;
vmgp("c_Manuf","SSA") = 57.631706237793 ;
vmgp("c_Manuf","ROW") = 53328.984375 ;
vmgp("c_ForestFish","USA") = 0.000837104395031929 ;
vmgp("c_ForestFish","EU_28") = 3.09678053855896 ;
vmgp("c_ForestFish","CHN") = 0.0075100720860064 ;
vmgp("c_ForestFish","JPN") = 0.00891524646431208 ;
vmgp("c_ForestFish","IND") = 0.000327902496792376 ;
vmgp("c_ForestFish","SSA") = 0.00191670749336481 ;
vmgp("c_ForestFish","ROW") = 11.0647411346436 ;
vmgp("c_Svces","USA") = 19186.185546875 ;
vmgp("c_Svces","EU_28") = 9761.845703125 ;
vmgp("c_Svces","CHN") = 38809.62109375 ;
vmgp("c_Svces","JPN") = 3494.71875 ;
vmgp("c_Svces","IND") = 4590.5673828125 ;
vmgp("c_Svces","SSA") = 5586.71826171875 ;
vmgp("c_Svces","ROW") = 55232.77734375 ;
parameter vdib(*,*) ;
vdib("c_Rice","USA") = 0.474240779876709 ;
vdib("c_Rice","EU_28") = 5.99281930923462 ;
vdib("c_Rice","CHN") = 0.756987810134888 ;
vdib("c_Rice","JPN") = 0.269917964935303 ;
vdib("c_Rice","IND") = 0.149629563093185 ;
vdib("c_Rice","SSA") = 207.638732910156 ;
vdib("c_Rice","ROW") = 200.243789672852 ;
vdib("c_Crops","USA") = 357.835083007812 ;
vdib("c_Crops","EU_28") = 2424.16625976562 ;
vdib("c_Crops","CHN") = 1269.72534179688 ;
vdib("c_Crops","JPN") = 484.5107421875 ;
vdib("c_Crops","IND") = 113.930099487305 ;
vdib("c_Crops","SSA") = 235.496978759766 ;
vdib("c_Crops","ROW") = 18263.056640625 ;
vdib("c_Livestock","USA") = 135.991928100586 ;
vdib("c_Livestock","EU_28") = 2208.30712890625 ;
vdib("c_Livestock","CHN") = 23035.39453125 ;
vdib("c_Livestock","JPN") = 756.359008789062 ;
vdib("c_Livestock","IND") = 408.221130371094 ;
vdib("c_Livestock","SSA") = 3729.05346679688 ;
vdib("c_Livestock","ROW") = 20591.685546875 ;
vdib("c_FoodProc","USA") = 20.5741405487061 ;
vdib("c_FoodProc","EU_28") = 183.726837158203 ;
vdib("c_FoodProc","CHN") = 25.6785106658936 ;
vdib("c_FoodProc","JPN") = 15.0382423400879 ;
vdib("c_FoodProc","IND") = 4.13186264038086 ;
vdib("c_FoodProc","SSA") = 4536.7578125 ;
vdib("c_FoodProc","ROW") = 7713.13427734375 ;
vdib("c_Textiles","USA") = 1211.78845214844 ;
vdib("c_Textiles","EU_28") = 4050.3974609375 ;
vdib("c_Textiles","CHN") = 69.8878631591797 ;
vdib("c_Textiles","JPN") = 1326.21875 ;
vdib("c_Textiles","IND") = 655.006896972656 ;
vdib("c_Textiles","SSA") = 239.964721679688 ;
vdib("c_Textiles","ROW") = 3655.04125976562 ;
vdib("c_Chem","USA") = 3026.32861328125 ;
vdib("c_Chem","EU_28") = 6477.91455078125 ;
vdib("c_Chem","CHN") = 140.998489379883 ;
vdib("c_Chem","JPN") = 73.8341217041016 ;
vdib("c_Chem","IND") = 2145.46557617188 ;
vdib("c_Chem","SSA") = 778.4736328125 ;
vdib("c_Chem","ROW") = 25460.3515625 ;
vdib("c_Manuf","USA") = 761694.3125 ;
vdib("c_Manuf","EU_28") = 462723.15625 ;
vdib("c_Manuf","CHN") = 947049.75 ;
vdib("c_Manuf","JPN") = 313798.84375 ;
vdib("c_Manuf","IND") = 172546.9375 ;
vdib("c_Manuf","SSA") = 36705.62890625 ;
vdib("c_Manuf","ROW") = 597128.5625 ;
vdib("c_ForestFish","USA") = 7.77471160888672 ;
vdib("c_ForestFish","EU_28") = 873.939453125 ;
vdib("c_ForestFish","CHN") = 28.0799160003662 ;
vdib("c_ForestFish","JPN") = 2.97142577171326 ;
vdib("c_ForestFish","IND") = 1.62996792793274 ;
vdib("c_ForestFish","SSA") = 1179.61462402344 ;
vdib("c_ForestFish","ROW") = 3338.26196289062 ;
vdib("c_Svces","USA") = 2873172.5 ;
vdib("c_Svces","EU_28") = 2258155.25 ;
vdib("c_Svces","CHN") = 3693320.5 ;
vdib("c_Svces","JPN") = 801181.8125 ;
vdib("c_Svces","IND") = 532308.6875 ;
vdib("c_Svces","SSA") = 247959.015625 ;
vdib("c_Svces","ROW") = 3615884.5 ;
parameter vdip(*,*) ;
vdip("c_Rice","USA") = 0.474240779876709 ;
vdip("c_Rice","EU_28") = 5.94858407974243 ;
vdip("c_Rice","CHN") = 0.756987810134888 ;
vdip("c_Rice","JPN") = 0.269917964935303 ;
vdip("c_Rice","IND") = 0.149629563093185 ;
vdip("c_Rice","SSA") = 223.734146118164 ;
vdip("c_Rice","ROW") = 207.429397583008 ;
vdip("c_Crops","USA") = 357.835083007812 ;
vdip("c_Crops","EU_28") = 2290.322265625 ;
vdip("c_Crops","CHN") = 1269.72534179688 ;
vdip("c_Crops","JPN") = 484.5107421875 ;
vdip("c_Crops","IND") = 113.930099487305 ;
vdip("c_Crops","SSA") = 237.439254760742 ;
vdip("c_Crops","ROW") = 18294.587890625 ;
vdip("c_Livestock","USA") = 135.991928100586 ;
vdip("c_Livestock","EU_28") = 2150.95068359375 ;
vdip("c_Livestock","CHN") = 23061.662109375 ;
vdip("c_Livestock","JPN") = 756.359008789062 ;
vdip("c_Livestock","IND") = 410.133728027344 ;
vdip("c_Livestock","SSA") = 3750.73510742188 ;
vdip("c_Livestock","ROW") = 21019.916015625 ;
vdip("c_FoodProc","USA") = 20.5741405487061 ;
vdip("c_FoodProc","EU_28") = 183.537887573242 ;
vdip("c_FoodProc","CHN") = 25.6785106658936 ;
vdip("c_FoodProc","JPN") = 15.0382423400879 ;
vdip("c_FoodProc","IND") = 4.13186264038086 ;
vdip("c_FoodProc","SSA") = 4703.11767578125 ;
vdip("c_FoodProc","ROW") = 7781.07421875 ;
vdip("c_Textiles","USA") = 1224.15454101562 ;
vdip("c_Textiles","EU_28") = 4121.087890625 ;
vdip("c_Textiles","CHN") = 69.8878631591797 ;
vdip("c_Textiles","JPN") = 1326.21875 ;
vdip("c_Textiles","IND") = 643.177612304688 ;
vdip("c_Textiles","SSA") = 257.277069091797 ;
vdip("c_Textiles","ROW") = 3780.29541015625 ;
vdip("c_Chem","USA") = 3103.83911132812 ;
vdip("c_Chem","EU_28") = 6804.16796875 ;
vdip("c_Chem","CHN") = 140.998489379883 ;
vdip("c_Chem","JPN") = 73.8341217041016 ;
vdip("c_Chem","IND") = 2464.93090820312 ;
vdip("c_Chem","SSA") = 853.324645996094 ;
vdip("c_Chem","ROW") = 25839.1171875 ;
vdip("c_Manuf","USA") = 771481.125 ;
vdip("c_Manuf","EU_28") = 479862.4375 ;
vdip("c_Manuf","CHN") = 1020122.5625 ;
vdip("c_Manuf","JPN") = 313798.84375 ;
vdip("c_Manuf","IND") = 186970.75 ;
vdip("c_Manuf","SSA") = 38319.71484375 ;
vdip("c_Manuf","ROW") = 627629.125 ;
vdip("c_ForestFish","USA") = 7.77471160888672 ;
vdip("c_ForestFish","EU_28") = 860.787658691406 ;
vdip("c_ForestFish","CHN") = 28.0799160003662 ;
vdip("c_ForestFish","JPN") = 2.97142577171326 ;
vdip("c_ForestFish","IND") = 1.62996792793274 ;
vdip("c_ForestFish","SSA") = 1179.97631835938 ;
vdip("c_ForestFish","ROW") = 3374.13647460938 ;
vdip("c_Svces","USA") = 2692840.75 ;
vdip("c_Svces","EU_28") = 2438634.25 ;
vdip("c_Svces","CHN") = 3877627.75 ;
vdip("c_Svces","JPN") = 801181.8125 ;
vdip("c_Svces","IND") = 530251.5625 ;
vdip("c_Svces","SSA") = 249520.9375 ;
vdip("c_Svces","ROW") = 3722471.75 ;
parameter vmib(*,*) ;
vmib("c_Rice","USA") = 0.0224927347153425 ;
vmib("c_Rice","EU_28") = 1.9020129442215 ;
vmib("c_Rice","CHN") = 0.41016161441803 ;
vmib("c_Rice","JPN") = 0.121821179986 ;
vmib("c_Rice","IND") = 0.00348905869759619 ;
vmib("c_Rice","SSA") = 46.038761138916 ;
vmib("c_Rice","ROW") = 98.1450271606445 ;
vmib("c_Crops","USA") = 9.66176891326904 ;
vmib("c_Crops","EU_28") = 1293.21801757812 ;
vmib("c_Crops","CHN") = 8.19382190704346 ;
vmib("c_Crops","JPN") = 95.6057205200195 ;
vmib("c_Crops","IND") = 7.0186882019043 ;
vmib("c_Crops","SSA") = 13.2028388977051 ;
vmib("c_Crops","ROW") = 2828.55297851562 ;
vmib("c_Livestock","USA") = 2.02631258964539 ;
vmib("c_Livestock","EU_28") = 319.348236083984 ;
vmib("c_Livestock","CHN") = 386.411590576172 ;
vmib("c_Livestock","JPN") = 53.1550064086914 ;
vmib("c_Livestock","IND") = 11.7735691070557 ;
vmib("c_Livestock","SSA") = 83.6565933227539 ;
vmib("c_Livestock","ROW") = 1477.59436035156 ;
vmib("c_FoodProc","USA") = 10.520073890686 ;
vmib("c_FoodProc","EU_28") = 348.858367919922 ;
vmib("c_FoodProc","CHN") = 9.87424468994141 ;
vmib("c_FoodProc","JPN") = 3.84383130073547 ;
vmib("c_FoodProc","IND") = 0.911566078662872 ;
vmib("c_FoodProc","SSA") = 273.707672119141 ;
vmib("c_FoodProc","ROW") = 3851.6953125 ;
vmib("c_Textiles","USA") = 434.670288085938 ;
vmib("c_Textiles","EU_28") = 3092.93603515625 ;
vmib("c_Textiles","CHN") = 121.476371765137 ;
vmib("c_Textiles","JPN") = 1916.97668457031 ;
vmib("c_Textiles","IND") = 77.9321212768555 ;
vmib("c_Textiles","SSA") = 290.842803955078 ;
vmib("c_Textiles","ROW") = 2955.68090820312 ;
vmib("c_Chem","USA") = 1192.005859375 ;
vmib("c_Chem","EU_28") = 6256.783203125 ;
vmib("c_Chem","CHN") = 150.362686157227 ;
vmib("c_Chem","JPN") = 34.1969947814941 ;
vmib("c_Chem","IND") = 333.303802490234 ;
vmib("c_Chem","SSA") = 764.430603027344 ;
vmib("c_Chem","ROW") = 19848.55078125 ;
vmib("c_Manuf","USA") = 472151.71875 ;
vmib("c_Manuf","EU_28") = 573132.9375 ;
vmib("c_Manuf","CHN") = 235050.9375 ;
vmib("c_Manuf","JPN") = 102032.25 ;
vmib("c_Manuf","IND") = 73451.25 ;
vmib("c_Manuf","SSA") = 59308.1484375 ;
vmib("c_Manuf","ROW") = 798337.4375 ;
vmib("c_ForestFish","USA") = 0.177375346422195 ;
vmib("c_ForestFish","EU_28") = 84.7265701293945 ;
vmib("c_ForestFish","CHN") = 0.788733541965485 ;
vmib("c_ForestFish","JPN") = 0.391680300235748 ;
vmib("c_ForestFish","IND") = 0.0913986414670944 ;
vmib("c_ForestFish","SSA") = 0.323030084371567 ;
vmib("c_ForestFish","ROW") = 264.214782714844 ;
vmib("c_Svces","USA") = 99352.0078125 ;
vmib("c_Svces","EU_28") = 92506.1640625 ;
vmib("c_Svces","CHN") = 16403.72265625 ;
vmib("c_Svces","JPN") = 14452.9287109375 ;
vmib("c_Svces","IND") = 2711.4375 ;
vmib("c_Svces","SSA") = 5716.712890625 ;
vmib("c_Svces","ROW") = 100085.1953125 ;
parameter vmip(*,*) ;
vmip("c_Rice","USA") = 0.0224927347153425 ;
vmip("c_Rice","EU_28") = 1.9020129442215 ;
vmip("c_Rice","CHN") = 0.41016161441803 ;
vmip("c_Rice","JPN") = 0.121821179986 ;
vmip("c_Rice","IND") = 0.00348905869759619 ;
vmip("c_Rice","SSA") = 52.2934684753418 ;
vmip("c_Rice","ROW") = 98.1301116943359 ;
vmip("c_Crops","USA") = 9.66176891326904 ;
vmip("c_Crops","EU_28") = 1315.93957519531 ;
vmip("c_Crops","CHN") = 8.19382190704346 ;
vmip("c_Crops","JPN") = 101.981056213379 ;
vmip("c_Crops","IND") = 7.0186882019043 ;
vmip("c_Crops","SSA") = 13.7000198364258 ;
vmip("c_Crops","ROW") = 2851.79809570312 ;
vmip("c_Livestock","USA") = 2.02631258964539 ;
vmip("c_Livestock","EU_28") = 324.757232666016 ;
vmip("c_Livestock","CHN") = 462.741027832031 ;
vmip("c_Livestock","JPN") = 53.6073722839355 ;
vmip("c_Livestock","IND") = 11.7735691070557 ;
vmip("c_Livestock","SSA") = 84.6285858154297 ;
vmip("c_Livestock","ROW") = 1505.76550292969 ;
vmip("c_FoodProc","USA") = 10.520073890686 ;
vmip("c_FoodProc","EU_28") = 349.054595947266 ;
vmip("c_FoodProc","CHN") = 9.87424468994141 ;
vmip("c_FoodProc","JPN") = 3.84383130073547 ;
vmip("c_FoodProc","IND") = 0.911566078662872 ;
vmip("c_FoodProc","SSA") = 294.566314697266 ;
vmip("c_FoodProc","ROW") = 3852.03564453125 ;
vmip("c_Textiles","USA") = 439.424255371094 ;
vmip("c_Textiles","EU_28") = 3150.3759765625 ;
vmip("c_Textiles","CHN") = 121.476371765137 ;
vmip("c_Textiles","JPN") = 2040.87072753906 ;
vmip("c_Textiles","IND") = 77.9321212768555 ;
vmip("c_Textiles","SSA") = 339.853057861328 ;
vmip("c_Textiles","ROW") = 3026.3046875 ;
vmip("c_Chem","USA") = 1223.21313476562 ;
vmip("c_Chem","EU_28") = 6302.81201171875 ;
vmip("c_Chem","CHN") = 150.362686157227 ;
vmip("c_Chem","JPN") = 34.1969947814941 ;
vmip("c_Chem","IND") = 333.303802490234 ;
vmip("c_Chem","SSA") = 870.491760253906 ;
vmip("c_Chem","ROW") = 19953.697265625 ;
vmip("c_Manuf","USA") = 489776.875 ;
vmip("c_Manuf","EU_28") = 605249.9375 ;
vmip("c_Manuf","CHN") = 293198.875 ;
vmip("c_Manuf","JPN") = 108628.0703125 ;
vmip("c_Manuf","IND") = 73451.25 ;
vmip("c_Manuf","SSA") = 62798.85546875 ;
vmip("c_Manuf","ROW") = 830832.1875 ;
vmip("c_ForestFish","USA") = 0.177375346422195 ;
vmip("c_ForestFish","EU_28") = 85.6967010498047 ;
vmip("c_ForestFish","CHN") = 0.788733541965485 ;
vmip("c_ForestFish","JPN") = 0.391680300235748 ;
vmip("c_ForestFish","IND") = 0.0913986414670944 ;
vmip("c_ForestFish","SSA") = 0.324278205633163 ;
vmip("c_ForestFish","ROW") = 264.490112304688 ;
vmip("c_Svces","USA") = 87960.8984375 ;
vmip("c_Svces","EU_28") = 99512.546875 ;
vmip("c_Svces","CHN") = 19477.990234375 ;
vmip("c_Svces","JPN") = 14452.9365234375 ;
vmip("c_Svces","IND") = 2711.44140625 ;
vmip("c_Svces","SSA") = 5731.4013671875 ;
vmip("c_Svces","ROW") = 100903.375 ;
parameter evfb(*,*,*) ;
evfb("Land","a_Rice","USA") = 237.973770141602 ;
evfb("Land","a_Rice","EU_28") = 295.818054199219 ;
evfb("Land","a_Rice","CHN") = 31457.0390625 ;
evfb("Land","a_Rice","JPN") = 3342.07592773438 ;
evfb("Land","a_Rice","IND") = 11868.10546875 ;
evfb("Land","a_Rice","SSA") = 2171.26635742188 ;
evfb("Land","a_Rice","ROW") = 37054.765625 ;
evfb("Land","a_Crops","USA") = 29688.15234375 ;
evfb("Land","a_Crops","EU_28") = 40074.3203125 ;
evfb("Land","a_Crops","CHN") = 161184.9375 ;
evfb("Land","a_Crops","JPN") = 4196.39306640625 ;
evfb("Land","a_Crops","IND") = 65975.7421875 ;
evfb("Land","a_Crops","SSA") = 47590.8515625 ;
evfb("Land","a_Crops","ROW") = 145005.25 ;
evfb("Land","a_Livestock","USA") = 11425.306640625 ;
evfb("Land","a_Livestock","EU_28") = 27087.86328125 ;
evfb("Land","a_Livestock","CHN") = 58363.2109375 ;
evfb("Land","a_Livestock","JPN") = 1902.99694824219 ;
evfb("Land","a_Livestock","IND") = 29963.11328125 ;
evfb("Land","a_Livestock","SSA") = 12083.9306640625 ;
evfb("Land","a_Livestock","ROW") = 57094.54296875 ;
evfb("UnSkLab","a_Rice","USA") = 328.598937988281 ;
evfb("UnSkLab","a_Rice","EU_28") = 570.759826660156 ;
evfb("UnSkLab","a_Rice","CHN") = 32998.421875 ;
evfb("UnSkLab","a_Rice","JPN") = 4247.26953125 ;
evfb("UnSkLab","a_Rice","IND") = 22501.771484375 ;
evfb("UnSkLab","a_Rice","SSA") = 2644.28637695312 ;
evfb("UnSkLab","a_Rice","ROW") = 56916.61328125 ;
evfb("UnSkLab","a_Crops","USA") = 15732.8427734375 ;
evfb("UnSkLab","a_Crops","EU_28") = 48871.2421875 ;
evfb("UnSkLab","a_Crops","CHN") = 185500.0625 ;
evfb("UnSkLab","a_Crops","JPN") = 6687.17626953125 ;
evfb("UnSkLab","a_Crops","IND") = 125051.6328125 ;
evfb("UnSkLab","a_Crops","SSA") = 49562.19140625 ;
evfb("UnSkLab","a_Crops","ROW") = 231106.609375 ;
evfb("UnSkLab","a_Livestock","USA") = 8012.67236328125 ;
evfb("UnSkLab","a_Livestock","EU_28") = 27739.0859375 ;
evfb("UnSkLab","a_Livestock","CHN") = 68819.078125 ;
evfb("UnSkLab","a_Livestock","JPN") = 2961.5009765625 ;
evfb("UnSkLab","a_Livestock","IND") = 56788.375 ;
evfb("UnSkLab","a_Livestock","SSA") = 12008.6875 ;
evfb("UnSkLab","a_Livestock","ROW") = 94299.640625 ;
evfb("UnSkLab","a_FoodProc","USA") = 86589.109375 ;
evfb("UnSkLab","a_FoodProc","EU_28") = 59243.2265625 ;
evfb("UnSkLab","a_FoodProc","CHN") = 134536.65625 ;
evfb("UnSkLab","a_FoodProc","JPN") = 16822.482421875 ;
evfb("UnSkLab","a_FoodProc","IND") = 21342.25 ;
evfb("UnSkLab","a_FoodProc","SSA") = 55074.1171875 ;
evfb("UnSkLab","a_FoodProc","ROW") = 156589.546875 ;
evfb("UnSkLab","a_Energy","USA") = 52987.9609375 ;
evfb("UnSkLab","a_Energy","EU_28") = 19774.361328125 ;
evfb("UnSkLab","a_Energy","CHN") = 107518.109375 ;
evfb("UnSkLab","a_Energy","JPN") = 6556.2109375 ;
evfb("UnSkLab","a_Energy","IND") = 43081.734375 ;
evfb("UnSkLab","a_Energy","SSA") = 10074.5673828125 ;
evfb("UnSkLab","a_Energy","ROW") = 138968.96875 ;
evfb("UnSkLab","a_Textiles","USA") = 16518.318359375 ;
evfb("UnSkLab","a_Textiles","EU_28") = 24240.912109375 ;
evfb("UnSkLab","a_Textiles","CHN") = 126870.1640625 ;
evfb("UnSkLab","a_Textiles","JPN") = 4399.0859375 ;
evfb("UnSkLab","a_Textiles","IND") = 30787.01953125 ;
evfb("UnSkLab","a_Textiles","SSA") = 6673.01025390625 ;
evfb("UnSkLab","a_Textiles","ROW") = 87037.4765625 ;
evfb("UnSkLab","a_Chem","USA") = 76132.8515625 ;
evfb("UnSkLab","a_Chem","EU_28") = 66021.328125 ;
evfb("UnSkLab","a_Chem","CHN") = 137848.640625 ;
evfb("UnSkLab","a_Chem","JPN") = 20538.013671875 ;
evfb("UnSkLab","a_Chem","IND") = 21634.984375 ;
evfb("UnSkLab","a_Chem","SSA") = 4694.5283203125 ;
evfb("UnSkLab","a_Chem","ROW") = 113756.5 ;
evfb("UnSkLab","a_Manuf","USA") = 598912.5625 ;
evfb("UnSkLab","a_Manuf","EU_28") = 346252.4375 ;
evfb("UnSkLab","a_Manuf","CHN") = 761312.625 ;
evfb("UnSkLab","a_Manuf","JPN") = 140832.546875 ;
evfb("UnSkLab","a_Manuf","IND") = 81098.859375 ;
evfb("UnSkLab","a_Manuf","SSA") = 33909.06640625 ;
evfb("UnSkLab","a_Manuf","ROW") = 548440.125 ;
evfb("UnSkLab","a_ForestFish","USA") = 8981.837890625 ;
evfb("UnSkLab","a_ForestFish","EU_28") = 5734.77587890625 ;
evfb("UnSkLab","a_ForestFish","CHN") = 94875.1640625 ;
evfb("UnSkLab","a_ForestFish","JPN") = 1564.18859863281 ;
evfb("UnSkLab","a_ForestFish","IND") = 28750.318359375 ;
evfb("UnSkLab","a_ForestFish","SSA") = 20499.798828125 ;
evfb("UnSkLab","a_ForestFish","ROW") = 37118.8828125 ;
evfb("UnSkLab","a_Svces","USA") = 3282724.5 ;
evfb("UnSkLab","a_Svces","EU_28") = 1784304.5 ;
evfb("UnSkLab","a_Svces","CHN") = 2457188 ;
evfb("UnSkLab","a_Svces","JPN") = 615202.3125 ;
evfb("UnSkLab","a_Svces","IND") = 505118.3125 ;
evfb("UnSkLab","a_Svces","SSA") = 286792.71875 ;
evfb("UnSkLab","a_Svces","ROW") = 3247012.25 ;
evfb("SkLab","a_Rice","USA") = 192.127487182617 ;
evfb("SkLab","a_Rice","EU_28") = 317.430694580078 ;
evfb("SkLab","a_Rice","CHN") = 2629.33081054688 ;
evfb("SkLab","a_Rice","JPN") = 322.001129150391 ;
evfb("SkLab","a_Rice","IND") = 40.8694458007812 ;
evfb("SkLab","a_Rice","SSA") = 220.970306396484 ;
evfb("SkLab","a_Rice","ROW") = 5714.638671875 ;
evfb("SkLab","a_Crops","USA") = 15368.171875 ;
evfb("SkLab","a_Crops","EU_28") = 12945.8525390625 ;
evfb("SkLab","a_Crops","CHN") = 11655.8564453125 ;
evfb("SkLab","a_Crops","JPN") = 412.239501953125 ;
evfb("SkLab","a_Crops","IND") = 221.102584838867 ;
evfb("SkLab","a_Crops","SSA") = 2984.68994140625 ;
evfb("SkLab","a_Crops","ROW") = 17769.029296875 ;
evfb("SkLab","a_Livestock","USA") = 7827.333984375 ;
evfb("SkLab","a_Livestock","EU_28") = 9063.4638671875 ;
evfb("SkLab","a_Livestock","CHN") = 4324.11328125 ;
evfb("SkLab","a_Livestock","JPN") = 182.458557128906 ;
evfb("SkLab","a_Livestock","IND") = 100.406982421875 ;
evfb("SkLab","a_Livestock","SSA") = 965.268859863281 ;
evfb("SkLab","a_Livestock","ROW") = 10732.31640625 ;
evfb("SkLab","a_FoodProc","USA") = 29087.998046875 ;
evfb("SkLab","a_FoodProc","EU_28") = 66393.5625 ;
evfb("SkLab","a_FoodProc","CHN") = 26544.71484375 ;
evfb("SkLab","a_FoodProc","JPN") = 8929.611328125 ;
evfb("SkLab","a_FoodProc","IND") = 1791.65808105469 ;
evfb("SkLab","a_FoodProc","SSA") = 10008.3427734375 ;
evfb("SkLab","a_FoodProc","ROW") = 88573.140625 ;
evfb("SkLab","a_Energy","USA") = 21668.21484375 ;
evfb("SkLab","a_Energy","EU_28") = 23952.671875 ;
evfb("SkLab","a_Energy","CHN") = 21067.271484375 ;
evfb("SkLab","a_Energy","JPN") = 7134.1591796875 ;
evfb("SkLab","a_Energy","IND") = 35670.98828125 ;
evfb("SkLab","a_Energy","SSA") = 5106.47607421875 ;
evfb("SkLab","a_Energy","ROW") = 119910.234375 ;
evfb("SkLab","a_Textiles","USA") = 5549.0205078125 ;
evfb("SkLab","a_Textiles","EU_28") = 24802.072265625 ;
evfb("SkLab","a_Textiles","CHN") = 25032.078125 ;
evfb("SkLab","a_Textiles","JPN") = 2335.09692382812 ;
evfb("SkLab","a_Textiles","IND") = 2584.5361328125 ;
evfb("SkLab","a_Textiles","SSA") = 1399.53796386719 ;
evfb("SkLab","a_Textiles","ROW") = 45058.734375 ;
evfb("SkLab","a_Chem","USA") = 25575.416015625 ;
evfb("SkLab","a_Chem","EU_28") = 72736.3125 ;
evfb("SkLab","a_Chem","CHN") = 27198.18359375 ;
evfb("SkLab","a_Chem","JPN") = 10901.8671875 ;
evfb("SkLab","a_Chem","IND") = 1816.23291015625 ;
evfb("SkLab","a_Chem","SSA") = 2355.78759765625 ;
evfb("SkLab","a_Chem","ROW") = 66846.8203125 ;
evfb("SkLab","a_Manuf","USA") = 198553.484375 ;
evfb("SkLab","a_Manuf","EU_28") = 376365.46875 ;
evfb("SkLab","a_Manuf","CHN") = 146303.96875 ;
evfb("SkLab","a_Manuf","JPN") = 74695.21875 ;
evfb("SkLab","a_Manuf","IND") = 6948.97802734375 ;
evfb("SkLab","a_Manuf","SSA") = 12846.453125 ;
evfb("SkLab","a_Manuf","ROW") = 323507.1875 ;
evfb("SkLab","a_ForestFish","USA") = 8779.6298828125 ;
evfb("SkLab","a_ForestFish","EU_28") = 1702.99609375 ;
evfb("SkLab","a_ForestFish","CHN") = 5960.4609375 ;
evfb("SkLab","a_ForestFish","JPN") = 96.3381805419922 ;
evfb("SkLab","a_ForestFish","IND") = 50.8331565856934 ;
evfb("SkLab","a_ForestFish","SSA") = 750.3212890625 ;
evfb("SkLab","a_ForestFish","ROW") = 4289.42431640625 ;
evfb("SkLab","a_Svces","USA") = 3780121.5 ;
evfb("SkLab","a_Svces","EU_28") = 2512409.5 ;
evfb("SkLab","a_Svces","CHN") = 1334502.625 ;
evfb("SkLab","a_Svces","JPN") = 550407.875 ;
evfb("SkLab","a_Svces","IND") = 347012.65625 ;
evfb("SkLab","a_Svces","SSA") = 211630.6875 ;
evfb("SkLab","a_Svces","ROW") = 3201343.5 ;
evfb("Capital","a_Rice","USA") = 937.395629882812 ;
evfb("Capital","a_Rice","EU_28") = 1297.12182617188 ;
evfb("Capital","a_Rice","CHN") = 33949.4453125 ;
evfb("Capital","a_Rice","JPN") = 4626.61083984375 ;
evfb("Capital","a_Rice","IND") = 5285.97900390625 ;
evfb("Capital","a_Rice","SSA") = 4857.96435546875 ;
evfb("Capital","a_Rice","ROW") = 54054.97265625 ;
evfb("Capital","a_Crops","USA") = 54448.1796875 ;
evfb("Capital","a_Crops","EU_28") = 49784.51171875 ;
evfb("Capital","a_Crops","CHN") = 163105.125 ;
evfb("Capital","a_Crops","JPN") = 6409.05322265625 ;
evfb("Capital","a_Crops","IND") = 28824.591796875 ;
evfb("Capital","a_Crops","SSA") = 54032.8984375 ;
evfb("Capital","a_Crops","ROW") = 135998.15625 ;
evfb("Capital","a_Livestock","USA") = 27912.099609375 ;
evfb("Capital","a_Livestock","EU_28") = 37928.515625 ;
evfb("Capital","a_Livestock","CHN") = 58506.1328125 ;
evfb("Capital","a_Livestock","JPN") = 2839.30712890625 ;
evfb("Capital","a_Livestock","IND") = 13084.064453125 ;
evfb("Capital","a_Livestock","SSA") = 14406.810546875 ;
evfb("Capital","a_Livestock","ROW") = 63904.36328125 ;
evfb("Capital","a_FoodProc","USA") = 132815.875 ;
evfb("Capital","a_FoodProc","EU_28") = 186700.96875 ;
evfb("Capital","a_FoodProc","CHN") = 178375.453125 ;
evfb("Capital","a_FoodProc","JPN") = 56064.109375 ;
evfb("Capital","a_FoodProc","IND") = 18155.46484375 ;
evfb("Capital","a_FoodProc","SSA") = 73441.8203125 ;
evfb("Capital","a_FoodProc","ROW") = 406259.6875 ;
evfb("Capital","a_Energy","USA") = 220980.765625 ;
evfb("Capital","a_Energy","EU_28") = 165926.890625 ;
evfb("Capital","a_Energy","CHN") = 187239.34375 ;
evfb("Capital","a_Energy","JPN") = 28119.599609375 ;
evfb("Capital","a_Energy","IND") = 49939.65234375 ;
evfb("Capital","a_Energy","SSA") = 67794.6796875 ;
evfb("Capital","a_Energy","ROW") = 1159170.25 ;
evfb("Capital","a_Textiles","USA") = 6214.39599609375 ;
evfb("Capital","a_Textiles","EU_28") = 50620.6953125 ;
evfb("Capital","a_Textiles","CHN") = 100960.421875 ;
evfb("Capital","a_Textiles","JPN") = 1821.34716796875 ;
evfb("Capital","a_Textiles","IND") = 27095.65625 ;
evfb("Capital","a_Textiles","SSA") = 6895.6796875 ;
evfb("Capital","a_Textiles","ROW") = 134060.671875 ;
evfb("Capital","a_Chem","USA") = 232738.46875 ;
evfb("Capital","a_Chem","EU_28") = 283986.46875 ;
evfb("Capital","a_Chem","CHN") = 230319.9375 ;
evfb("Capital","a_Chem","JPN") = 61442.453125 ;
evfb("Capital","a_Chem","IND") = 48152.12890625 ;
evfb("Capital","a_Chem","SSA") = 11409.6796875 ;
evfb("Capital","a_Chem","ROW") = 353462.25 ;
evfb("Capital","a_Manuf","USA") = 638997.5625 ;
evfb("Capital","a_Manuf","EU_28") = 723452.25 ;
evfb("Capital","a_Manuf","CHN") = 937258.875 ;
evfb("Capital","a_Manuf","JPN") = 296133.3125 ;
evfb("Capital","a_Manuf","IND") = 110814.046875 ;
evfb("Capital","a_Manuf","SSA") = 74401.75 ;
evfb("Capital","a_Manuf","ROW") = 1300070.125 ;
evfb("Capital","a_ForestFish","USA") = 9328.4140625 ;
evfb("Capital","a_ForestFish","EU_28") = 19083.572265625 ;
evfb("Capital","a_ForestFish","CHN") = 7448.650390625 ;
evfb("Capital","a_ForestFish","JPN") = 5042.11376953125 ;
evfb("Capital","a_ForestFish","IND") = 21811.125 ;
evfb("Capital","a_ForestFish","SSA") = 19815.53125 ;
evfb("Capital","a_ForestFish","ROW") = 66969.6015625 ;
evfb("Capital","a_Svces","USA") = 5918652.5 ;
evfb("Capital","a_Svces","EU_28") = 5561561.5 ;
evfb("Capital","a_Svces","CHN") = 2660009.25 ;
evfb("Capital","a_Svces","JPN") = 1860368.5 ;
evfb("Capital","a_Svces","IND") = 736965.9375 ;
evfb("Capital","a_Svces","SSA") = 401853.6875 ;
evfb("Capital","a_Svces","ROW") = 6945574.5 ;
evfb("NatRes","a_Energy","USA") = 61101.1015625 ;
evfb("NatRes","a_Energy","EU_28") = 13122.53515625 ;
evfb("NatRes","a_Energy","CHN") = 53481.14453125 ;
evfb("NatRes","a_Energy","JPN") = 87.8082504272461 ;
evfb("NatRes","a_Energy","IND") = 12273.7431640625 ;
evfb("NatRes","a_Energy","SSA") = 29107.9609375 ;
evfb("NatRes","a_Energy","ROW") = 411645.375 ;
evfb("NatRes","a_Manuf","USA") = 14772.66015625 ;
evfb("NatRes","a_Manuf","EU_28") = 6984.15234375 ;
evfb("NatRes","a_Manuf","CHN") = 15029.1279296875 ;
evfb("NatRes","a_Manuf","JPN") = 927.211303710938 ;
evfb("NatRes","a_Manuf","IND") = 2638.8037109375 ;
evfb("NatRes","a_Manuf","SSA") = 5342.30029296875 ;
evfb("NatRes","a_Manuf","ROW") = 41519.42578125 ;
evfb("NatRes","a_ForestFish","USA") = 5286.4248046875 ;
evfb("NatRes","a_ForestFish","EU_28") = 8407.166015625 ;
evfb("NatRes","a_ForestFish","CHN") = 44576.296875 ;
evfb("NatRes","a_ForestFish","JPN") = 3203.03173828125 ;
evfb("NatRes","a_ForestFish","IND") = 11570.00390625 ;
evfb("NatRes","a_ForestFish","SSA") = 9842.236328125 ;
evfb("NatRes","a_ForestFish","ROW") = 43246.703125 ;
parameter evfp(*,*,*) ;
evfp("Land","a_Rice","USA") = 175.049102783203 ;
evfp("Land","a_Rice","EU_28") = 121.289405822754 ;
evfp("Land","a_Rice","CHN") = 23235.521484375 ;
evfp("Land","a_Rice","JPN") = 2060.29956054688 ;
evfp("Land","a_Rice","IND") = 11854.92578125 ;
evfp("Land","a_Rice","SSA") = 2181.59887695312 ;
evfp("Land","a_Rice","ROW") = 36021.98046875 ;
evfp("Land","a_Crops","USA") = 22212.123046875 ;
evfp("Land","a_Crops","EU_28") = 26790.919921875 ;
evfp("Land","a_Crops","CHN") = 149587.203125 ;
evfp("Land","a_Crops","JPN") = 3400.30249023438 ;
evfp("Land","a_Crops","IND") = 65937.109375 ;
evfp("Land","a_Crops","SSA") = 47756.49609375 ;
evfp("Land","a_Crops","ROW") = 142926.65625 ;
evfp("Land","a_Livestock","USA") = 11615.8642578125 ;
evfp("Land","a_Livestock","EU_28") = 18522.625 ;
evfp("Land","a_Livestock","CHN") = 55378.47265625 ;
evfp("Land","a_Livestock","JPN") = 1505.71691894531 ;
evfp("Land","a_Livestock","IND") = 29941.453125 ;
evfp("Land","a_Livestock","SSA") = 12140.4970703125 ;
evfp("Land","a_Livestock","ROW") = 56362.9296875 ;
evfp("UnSkLab","a_Rice","USA") = 379.228851318359 ;
evfp("UnSkLab","a_Rice","EU_28") = 734.215698242188 ;
evfp("UnSkLab","a_Rice","CHN") = 32623.8359375 ;
evfp("UnSkLab","a_Rice","JPN") = 5684.53759765625 ;
evfp("UnSkLab","a_Rice","IND") = 22497.80078125 ;
evfp("UnSkLab","a_Rice","SSA") = 2693.6162109375 ;
evfp("UnSkLab","a_Rice","ROW") = 59315.44140625 ;
evfp("UnSkLab","a_Crops","USA") = 17415.615234375 ;
evfp("UnSkLab","a_Crops","EU_28") = 59511.92578125 ;
evfp("UnSkLab","a_Crops","CHN") = 182968.484375 ;
evfp("UnSkLab","a_Crops","JPN") = 9075.37890625 ;
evfp("UnSkLab","a_Crops","IND") = 125059.3828125 ;
evfp("UnSkLab","a_Crops","SSA") = 50473.73046875 ;
evfp("UnSkLab","a_Crops","ROW") = 255805.015625 ;
evfp("UnSkLab","a_Livestock","USA") = 9107.5234375 ;
evfp("UnSkLab","a_Livestock","EU_28") = 33523.5546875 ;
evfp("UnSkLab","a_Livestock","CHN") = 67736.5234375 ;
evfp("UnSkLab","a_Livestock","JPN") = 4018.74536132812 ;
evfp("UnSkLab","a_Livestock","IND") = 56788.3515625 ;
evfp("UnSkLab","a_Livestock","SSA") = 12213.474609375 ;
evfp("UnSkLab","a_Livestock","ROW") = 106645.328125 ;
evfp("UnSkLab","a_FoodProc","USA") = 104216 ;
evfp("UnSkLab","a_FoodProc","EU_28") = 86580.9453125 ;
evfp("UnSkLab","a_FoodProc","CHN") = 136798.25 ;
evfp("UnSkLab","a_FoodProc","JPN") = 24436.80078125 ;
evfp("UnSkLab","a_FoodProc","IND") = 21349.2578125 ;
evfp("UnSkLab","a_FoodProc","SSA") = 55827.43359375 ;
evfp("UnSkLab","a_FoodProc","ROW") = 174787.890625 ;
evfp("UnSkLab","a_Energy","USA") = 63774.6875 ;
evfp("UnSkLab","a_Energy","EU_28") = 28768.41015625 ;
evfp("UnSkLab","a_Energy","CHN") = 109325.515625 ;
evfp("UnSkLab","a_Energy","JPN") = 9523.7333984375 ;
evfp("UnSkLab","a_Energy","IND") = 43095.87890625 ;
evfp("UnSkLab","a_Energy","SSA") = 10339.134765625 ;
evfp("UnSkLab","a_Energy","ROW") = 156388.625 ;
evfp("UnSkLab","a_Textiles","USA") = 19880.94140625 ;
evfp("UnSkLab","a_Textiles","EU_28") = 35891.08203125 ;
evfp("UnSkLab","a_Textiles","CHN") = 129002.890625 ;
evfp("UnSkLab","a_Textiles","JPN") = 6390.2333984375 ;
evfp("UnSkLab","a_Textiles","IND") = 30797.12890625 ;
evfp("UnSkLab","a_Textiles","SSA") = 6773.04150390625 ;
evfp("UnSkLab","a_Textiles","ROW") = 96273.453125 ;
evfp("UnSkLab","a_Chem","USA") = 91631.171875 ;
evfp("UnSkLab","a_Chem","EU_28") = 96711.015625 ;
evfp("UnSkLab","a_Chem","CHN") = 140165.90625 ;
evfp("UnSkLab","a_Chem","JPN") = 29834.083984375 ;
evfp("UnSkLab","a_Chem","IND") = 21642.087890625 ;
evfp("UnSkLab","a_Chem","SSA") = 4823.51416015625 ;
evfp("UnSkLab","a_Chem","ROW") = 130197.0625 ;
evfp("UnSkLab","a_Manuf","USA") = 720832.875 ;
evfp("UnSkLab","a_Manuf","EU_28") = 507185.65625 ;
evfp("UnSkLab","a_Manuf","CHN") = 774110.5 ;
evfp("UnSkLab","a_Manuf","JPN") = 204577.234375 ;
evfp("UnSkLab","a_Manuf","IND") = 81125.4921875 ;
evfp("UnSkLab","a_Manuf","SSA") = 34646.4375 ;
evfp("UnSkLab","a_Manuf","ROW") = 625222.75 ;
evfp("UnSkLab","a_ForestFish","USA") = 10810.265625 ;
evfp("UnSkLab","a_ForestFish","EU_28") = 8495.5390625 ;
evfp("UnSkLab","a_ForestFish","CHN") = 96470.0390625 ;
evfp("UnSkLab","a_ForestFish","JPN") = 2272.18334960938 ;
evfp("UnSkLab","a_ForestFish","IND") = 28759.759765625 ;
evfp("UnSkLab","a_ForestFish","SSA") = 20719.72265625 ;
evfp("UnSkLab","a_ForestFish","ROW") = 40038.265625 ;
evfp("UnSkLab","a_Svces","USA") = 3950986.75 ;
evfp("UnSkLab","a_Svces","EU_28") = 2543290 ;
evfp("UnSkLab","a_Svces","CHN") = 2498494 ;
evfp("UnSkLab","a_Svces","JPN") = 893659.8125 ;
evfp("UnSkLab","a_Svces","IND") = 505284.15625 ;
evfp("UnSkLab","a_Svces","SSA") = 291320.125 ;
evfp("UnSkLab","a_Svces","ROW") = 3691171.5 ;
evfp("SkLab","a_Rice","USA") = 215.365951538086 ;
evfp("SkLab","a_Rice","EU_28") = 436.340789794922 ;
evfp("SkLab","a_Rice","CHN") = 2614.81176757812 ;
evfp("SkLab","a_Rice","JPN") = 437.120330810547 ;
evfp("SkLab","a_Rice","IND") = 40.8627738952637 ;
evfp("SkLab","a_Rice","SSA") = 226.550109863281 ;
evfp("SkLab","a_Rice","ROW") = 6250.5263671875 ;
evfp("SkLab","a_Crops","USA") = 17013.173828125 ;
evfp("SkLab","a_Crops","EU_28") = 15547.7880859375 ;
evfp("SkLab","a_Crops","CHN") = 11494.857421875 ;
evfp("SkLab","a_Crops","JPN") = 558.8115234375 ;
evfp("SkLab","a_Crops","IND") = 221.116302490234 ;
evfp("SkLab","a_Crops","SSA") = 3051.18579101562 ;
evfp("SkLab","a_Crops","ROW") = 19222.013671875 ;
evfp("SkLab","a_Livestock","USA") = 8897.06640625 ;
evfp("SkLab","a_Livestock","EU_28") = 10038.9521484375 ;
evfp("SkLab","a_Livestock","CHN") = 4255.49560546875 ;
evfp("SkLab","a_Livestock","JPN") = 247.452102661133 ;
evfp("SkLab","a_Livestock","IND") = 100.406936645508 ;
evfp("SkLab","a_Livestock","SSA") = 987.273132324219 ;
evfp("SkLab","a_Livestock","ROW") = 11622.052734375 ;
evfp("SkLab","a_FoodProc","USA") = 35009.42578125 ;
evfp("SkLab","a_FoodProc","EU_28") = 95137.3125 ;
evfp("SkLab","a_FoodProc","CHN") = 26990.9375 ;
evfp("SkLab","a_FoodProc","JPN") = 12971.400390625 ;
evfp("SkLab","a_FoodProc","IND") = 1792.24633789062 ;
evfp("SkLab","a_FoodProc","SSA") = 10219.1025390625 ;
evfp("SkLab","a_FoodProc","ROW") = 98851.390625 ;
evfp("SkLab","a_Energy","USA") = 26079.201171875 ;
evfp("SkLab","a_Energy","EU_28") = 34500.87109375 ;
evfp("SkLab","a_Energy","CHN") = 21421.41796875 ;
evfp("SkLab","a_Energy","JPN") = 10363.2763671875 ;
evfp("SkLab","a_Energy","IND") = 35682.69921875 ;
evfp("SkLab","a_Energy","SSA") = 5245 ;
evfp("SkLab","a_Energy","ROW") = 133876.78125 ;
evfp("SkLab","a_Textiles","USA") = 6678.63134765625 ;
evfp("SkLab","a_Textiles","EU_28") = 36208.8046875 ;
evfp("SkLab","a_Textiles","CHN") = 25452.875 ;
evfp("SkLab","a_Textiles","JPN") = 3392.02612304688 ;
evfp("SkLab","a_Textiles","IND") = 2585.384765625 ;
evfp("SkLab","a_Textiles","SSA") = 1432.47619628906 ;
evfp("SkLab","a_Textiles","ROW") = 49419.73828125 ;
evfp("SkLab","a_Chem","USA") = 30781.7890625 ;
evfp("SkLab","a_Chem","EU_28") = 104813.0625 ;
evfp("SkLab","a_Chem","CHN") = 27655.39453125 ;
evfp("SkLab","a_Chem","JPN") = 15836.3525390625 ;
evfp("SkLab","a_Chem","IND") = 1816.82934570312 ;
evfp("SkLab","a_Chem","SSA") = 2442.8173828125 ;
evfp("SkLab","a_Chem","ROW") = 76250.578125 ;
evfp("SkLab","a_Manuf","USA") = 238972.90625 ;
evfp("SkLab","a_Manuf","EU_28") = 542165.75 ;
evfp("SkLab","a_Manuf","CHN") = 148763.375 ;
evfp("SkLab","a_Manuf","JPN") = 108504.3359375 ;
evfp("SkLab","a_Manuf","IND") = 6951.259765625 ;
evfp("SkLab","a_Manuf","SSA") = 13204.734375 ;
evfp("SkLab","a_Manuf","ROW") = 368192.40625 ;
evfp("SkLab","a_ForestFish","USA") = 10566.89453125 ;
evfp("SkLab","a_ForestFish","EU_28") = 2388.3583984375 ;
evfp("SkLab","a_ForestFish","CHN") = 6060.65771484375 ;
evfp("SkLab","a_ForestFish","JPN") = 139.943496704102 ;
evfp("SkLab","a_ForestFish","IND") = 50.8498458862305 ;
evfp("SkLab","a_ForestFish","SSA") = 759.883239746094 ;
evfp("SkLab","a_ForestFish","ROW") = 4695.70947265625 ;
evfp("SkLab","a_Svces","USA") = 4549638.5 ;
evfp("SkLab","a_Svces","EU_28") = 3595405 ;
evfp("SkLab","a_Svces","CHN") = 1356936 ;
evfp("SkLab","a_Svces","JPN") = 799537.6875 ;
evfp("SkLab","a_Svces","IND") = 347126.59375 ;
evfp("SkLab","a_Svces","SSA") = 216050.78125 ;
evfp("SkLab","a_Svces","ROW") = 3624202.75 ;
evfp("Capital","a_Rice","USA") = 954.863647460938 ;
evfp("Capital","a_Rice","EU_28") = 1271.07568359375 ;
evfp("Capital","a_Rice","CHN") = 30948.73828125 ;
evfp("Capital","a_Rice","JPN") = 4298.06591796875 ;
evfp("Capital","a_Rice","IND") = 5200.06005859375 ;
evfp("Capital","a_Rice","SSA") = 4877.0380859375 ;
evfp("Capital","a_Rice","ROW") = 53897.51953125 ;
evfp("Capital","a_Crops","USA") = 54419.69921875 ;
evfp("Capital","a_Crops","EU_28") = 42393.328125 ;
evfp("Capital","a_Crops","CHN") = 154573.4375 ;
evfp("Capital","a_Crops","JPN") = 5856.07666015625 ;
evfp("Capital","a_Crops","IND") = 28572.74609375 ;
evfp("Capital","a_Crops","SSA") = 54204.7421875 ;
evfp("Capital","a_Crops","ROW") = 133750.40625 ;
evfp("Capital","a_Livestock","USA") = 28458.8671875 ;
evfp("Capital","a_Livestock","EU_28") = 29858.275390625 ;
evfp("Capital","a_Livestock","CHN") = 57224.42578125 ;
evfp("Capital","a_Livestock","JPN") = 2593.17895507812 ;
evfp("Capital","a_Livestock","IND") = 12974.62890625 ;
evfp("Capital","a_Livestock","SSA") = 14465.9912109375 ;
evfp("Capital","a_Livestock","ROW") = 61127.49609375 ;
evfp("Capital","a_FoodProc","USA") = 139467.515625 ;
evfp("Capital","a_FoodProc","EU_28") = 193159.46875 ;
evfp("Capital","a_FoodProc","CHN") = 181374 ;
evfp("Capital","a_FoodProc","JPN") = 57647.87109375 ;
evfp("Capital","a_FoodProc","IND") = 18155.46484375 ;
evfp("Capital","a_FoodProc","SSA") = 73671.859375 ;
evfp("Capital","a_FoodProc","ROW") = 411427.28125 ;
evfp("Capital","a_Energy","USA") = 232047.859375 ;
evfp("Capital","a_Energy","EU_28") = 171349.125 ;
evfp("Capital","a_Energy","CHN") = 190386.875 ;
evfp("Capital","a_Energy","JPN") = 28913.955078125 ;
evfp("Capital","a_Energy","IND") = 49939.65234375 ;
evfp("Capital","a_Energy","SSA") = 68128.3203125 ;
evfp("Capital","a_Energy","ROW") = 1174419.375 ;
evfp("Capital","a_Textiles","USA") = 6525.6240234375 ;
evfp("Capital","a_Textiles","EU_28") = 52016.1171875 ;
evfp("Capital","a_Textiles","CHN") = 102657.59375 ;
evfp("Capital","a_Textiles","JPN") = 1872.79858398438 ;
evfp("Capital","a_Textiles","IND") = 27095.65625 ;
evfp("Capital","a_Textiles","SSA") = 6928.5078125 ;
evfp("Capital","a_Textiles","ROW") = 135184.65625 ;
evfp("Capital","a_Chem","USA") = 244394.40625 ;
evfp("Capital","a_Chem","EU_28") = 292243.5 ;
evfp("Capital","a_Chem","CHN") = 234191.671875 ;
evfp("Capital","a_Chem","JPN") = 63178.1484375 ;
evfp("Capital","a_Chem","IND") = 48152.12890625 ;
evfp("Capital","a_Chem","SSA") = 11500.2578125 ;
evfp("Capital","a_Chem","ROW") = 359438 ;
evfp("Capital","a_Manuf","USA") = 670999.625 ;
evfp("Capital","a_Manuf","EU_28") = 743195.3125 ;
evfp("Capital","a_Manuf","CHN") = 953014.4375 ;
evfp("Capital","a_Manuf","JPN") = 304498.84375 ;
evfp("Capital","a_Manuf","IND") = 110814.046875 ;
evfp("Capital","a_Manuf","SSA") = 74912.8828125 ;
evfp("Capital","a_Manuf","ROW") = 1322278.125 ;
evfp("Capital","a_ForestFish","USA") = 9795.595703125 ;
evfp("Capital","a_ForestFish","EU_28") = 19744.9609375 ;
evfp("Capital","a_ForestFish","CHN") = 7573.8642578125 ;
evfp("Capital","a_ForestFish","JPN") = 5184.548828125 ;
evfp("Capital","a_ForestFish","IND") = 21811.125 ;
evfp("Capital","a_ForestFish","SSA") = 19851.130859375 ;
evfp("Capital","a_ForestFish","ROW") = 67593.6640625 ;
evfp("Capital","a_Svces","USA") = 6215069 ;
evfp("Capital","a_Svces","EU_28") = 5746506.5 ;
evfp("Capital","a_Svces","CHN") = 2704724.75 ;
evfp("Capital","a_Svces","JPN") = 1912922.25 ;
evfp("Capital","a_Svces","IND") = 736965.9375 ;
evfp("Capital","a_Svces","SSA") = 404798.78125 ;
evfp("Capital","a_Svces","ROW") = 7071225 ;
evfp("NatRes","a_Energy","USA") = 64161.1484375 ;
evfp("NatRes","a_Energy","EU_28") = 13490.23828125 ;
evfp("NatRes","a_Energy","CHN") = 54380.17578125 ;
evfp("NatRes","a_Energy","JPN") = 90.2887573242188 ;
evfp("NatRes","a_Energy","IND") = 12273.7431640625 ;
evfp("NatRes","a_Energy","SSA") = 29231.150390625 ;
evfp("NatRes","a_Energy","ROW") = 417124.40625 ;
evfp("NatRes","a_Manuf","USA") = 15512.5 ;
evfp("NatRes","a_Manuf","EU_28") = 7221.03173828125 ;
evfp("NatRes","a_Manuf","CHN") = 15281.771484375 ;
evfp("NatRes","a_Manuf","JPN") = 953.404235839844 ;
evfp("NatRes","a_Manuf","IND") = 2638.8037109375 ;
evfp("NatRes","a_Manuf","SSA") = 5417.0244140625 ;
evfp("NatRes","a_Manuf","ROW") = 42361.2109375 ;
evfp("NatRes","a_ForestFish","USA") = 5551.177734375 ;
evfp("NatRes","a_ForestFish","EU_28") = 8703.9833984375 ;
evfp("NatRes","a_ForestFish","CHN") = 45325.63671875 ;
evfp("NatRes","a_ForestFish","JPN") = 3293.5146484375 ;
evfp("NatRes","a_ForestFish","IND") = 11570.00390625 ;
evfp("NatRes","a_ForestFish","SSA") = 9863.556640625 ;
evfp("NatRes","a_ForestFish","ROW") = 43704.0234375 ;
parameter evos(*,*,*) ;
evos("Land","a_Rice","USA") = 228.294082641602 ;
evos("Land","a_Rice","EU_28") = 280.845031738281 ;
evos("Land","a_Rice","CHN") = 27955.064453125 ;
evos("Land","a_Rice","JPN") = 3084.00366210938 ;
evos("Land","a_Rice","IND") = 10999.455078125 ;
evos("Land","a_Rice","SSA") = 2071.82885742188 ;
evos("Land","a_Rice","ROW") = 34734.953125 ;
evos("Land","a_Crops","USA") = 28480.57421875 ;
evos("Land","a_Crops","EU_28") = 37928.203125 ;
evos("Land","a_Crops","CHN") = 143240.921875 ;
evos("Land","a_Crops","JPN") = 3872.35131835938 ;
evos("Land","a_Crops","IND") = 61146.84765625 ;
evos("Land","a_Crops","SSA") = 45418.17578125 ;
evos("Land","a_Crops","ROW") = 135155.53125 ;
evos("Land","a_Livestock","USA") = 10960.5771484375 ;
evos("Land","a_Livestock","EU_28") = 25557.0078125 ;
evos("Land","a_Livestock","CHN") = 51865.890625 ;
evos("Land","a_Livestock","JPN") = 1756.04907226562 ;
evos("Land","a_Livestock","IND") = 27770.05078125 ;
evos("Land","a_Livestock","SSA") = 11579.076171875 ;
evos("Land","a_Livestock","ROW") = 53169.59765625 ;
evos("UnSkLab","a_Rice","USA") = 247.424102783203 ;
evos("UnSkLab","a_Rice","EU_28") = 396.837799072266 ;
evos("UnSkLab","a_Rice","CHN") = 31975.728515625 ;
evos("UnSkLab","a_Rice","JPN") = 3421.47705078125 ;
evos("UnSkLab","a_Rice","IND") = 21371.734375 ;
evos("UnSkLab","a_Rice","SSA") = 2558.56298828125 ;
evos("UnSkLab","a_Rice","ROW") = 54053.5390625 ;
evos("UnSkLab","a_Crops","USA") = 11846.30859375 ;
evos("UnSkLab","a_Crops","EU_28") = 34355.87890625 ;
evos("UnSkLab","a_Crops","CHN") = 179751.015625 ;
evos("UnSkLab","a_Crops","JPN") = 5386.9951171875 ;
evos("UnSkLab","a_Crops","IND") = 118771.5546875 ;
evos("UnSkLab","a_Crops","SSA") = 47181.6796875 ;
evos("UnSkLab","a_Crops","ROW") = 212018.765625 ;
evos("UnSkLab","a_Livestock","USA") = 6033.27685546875 ;
evos("UnSkLab","a_Livestock","EU_28") = 19063.16015625 ;
evos("UnSkLab","a_Livestock","CHN") = 66686.2265625 ;
evos("UnSkLab","a_Livestock","JPN") = 2385.69946289062 ;
evos("UnSkLab","a_Livestock","IND") = 53936.47265625 ;
evos("UnSkLab","a_Livestock","SSA") = 11255.74609375 ;
evos("UnSkLab","a_Livestock","ROW") = 85511.9296875 ;
evos("UnSkLab","a_FoodProc","USA") = 65198.7265625 ;
evos("UnSkLab","a_FoodProc","EU_28") = 41358.765625 ;
evos("UnSkLab","a_FoodProc","CHN") = 130367.0703125 ;
evos("UnSkLab","a_FoodProc","JPN") = 13551.7041015625 ;
evos("UnSkLab","a_FoodProc","IND") = 20270.4453125 ;
evos("UnSkLab","a_FoodProc","SSA") = 53044.93359375 ;
evos("UnSkLab","a_FoodProc","ROW") = 142366.21875 ;
evos("UnSkLab","a_Energy","USA") = 39898.1796875 ;
evos("UnSkLab","a_Energy","EU_28") = 14185.1875 ;
evos("UnSkLab","a_Energy","CHN") = 104185.890625 ;
evos("UnSkLab","a_Energy","JPN") = 5281.49365234375 ;
evos("UnSkLab","a_Energy","IND") = 40918.1796875 ;
evos("UnSkLab","a_Energy","SSA") = 8947.392578125 ;
evos("UnSkLab","a_Energy","ROW") = 126218.1953125 ;
evos("UnSkLab","a_Textiles","USA") = 12437.7451171875 ;
evos("UnSkLab","a_Textiles","EU_28") = 16771.962890625 ;
evos("UnSkLab","a_Textiles","CHN") = 122938.1875 ;
evos("UnSkLab","a_Textiles","JPN") = 3543.77612304688 ;
evos("UnSkLab","a_Textiles","IND") = 29240.900390625 ;
evos("UnSkLab","a_Textiles","SSA") = 6359.55322265625 ;
evos("UnSkLab","a_Textiles","ROW") = 80802.7890625 ;
evos("UnSkLab","a_Chem","USA") = 57325.51171875 ;
evos("UnSkLab","a_Chem","EU_28") = 45676.140625 ;
evos("UnSkLab","a_Chem","CHN") = 133576.40625 ;
evos("UnSkLab","a_Chem","JPN") = 16544.828125 ;
evos("UnSkLab","a_Chem","IND") = 20548.478515625 ;
evos("UnSkLab","a_Chem","SSA") = 4220.5244140625 ;
evos("UnSkLab","a_Chem","ROW") = 101883.25 ;
evos("UnSkLab","a_Manuf","USA") = 450961.34375 ;
evos("UnSkLab","a_Manuf","EU_28") = 238032.421875 ;
evos("UnSkLab","a_Manuf","CHN") = 737717.9375 ;
evos("UnSkLab","a_Manuf","JPN") = 113450.6171875 ;
evos("UnSkLab","a_Manuf","IND") = 77026.09375 ;
evos("UnSkLab","a_Manuf","SSA") = 30445.470703125 ;
evos("UnSkLab","a_Manuf","ROW") = 490048.875 ;
evos("UnSkLab","a_ForestFish","USA") = 6763.0263671875 ;
evos("UnSkLab","a_ForestFish","EU_28") = 3957.25903320312 ;
evos("UnSkLab","a_ForestFish","CHN") = 91934.78125 ;
evos("UnSkLab","a_ForestFish","JPN") = 1260.06494140625 ;
evos("UnSkLab","a_ForestFish","IND") = 27306.482421875 ;
evos("UnSkLab","a_ForestFish","SSA") = 19879.263671875 ;
evos("UnSkLab","a_ForestFish","ROW") = 34334.34765625 ;
evos("UnSkLab","a_Svces","USA") = 2471782.75 ;
evos("UnSkLab","a_Svces","EU_28") = 1264046.5 ;
evos("UnSkLab","a_Svces","CHN") = 2381034.75 ;
evos("UnSkLab","a_Svces","JPN") = 495589.15625 ;
evos("UnSkLab","a_Svces","IND") = 479751.375 ;
evos("UnSkLab","a_Svces","SSA") = 267121.71875 ;
evos("UnSkLab","a_Svces","ROW") = 2886359.25 ;
evos("SkLab","a_Rice","USA") = 144.665634155273 ;
evos("SkLab","a_Rice","EU_28") = 208.976455688477 ;
evos("SkLab","a_Rice","CHN") = 2547.84228515625 ;
evos("SkLab","a_Rice","JPN") = 259.394836425781 ;
evos("SkLab","a_Rice","IND") = 38.8169860839844 ;
evos("SkLab","a_Rice","SSA") = 208.314331054688 ;
evos("SkLab","a_Rice","ROW") = 5279.5 ;
evos("SkLab","a_Crops","USA") = 11571.724609375 ;
evos("SkLab","a_Crops","EU_28") = 8405.7568359375 ;
evos("SkLab","a_Crops","CHN") = 11294.6162109375 ;
evos("SkLab","a_Crops","JPN") = 332.088256835938 ;
evos("SkLab","a_Crops","IND") = 209.998840332031 ;
evos("SkLab","a_Crops","SSA") = 2771.30126953125 ;
evos("SkLab","a_Crops","ROW") = 15779.2900390625 ;
evos("SkLab","a_Livestock","USA") = 5893.7236328125 ;
evos("SkLab","a_Livestock","EU_28") = 6126.3115234375 ;
evos("SkLab","a_Livestock","CHN") = 4190.099609375 ;
evos("SkLab","a_Livestock","JPN") = 146.983337402344 ;
evos("SkLab","a_Livestock","IND") = 95.3645629882812 ;
evos("SkLab","a_Livestock","SSA") = 856.692626953125 ;
evos("SkLab","a_Livestock","ROW") = 9128.28515625 ;
evos("SkLab","a_FoodProc","USA") = 21902.30078125 ;
evos("SkLab","a_FoodProc","EU_28") = 46755.06640625 ;
evos("SkLab","a_FoodProc","CHN") = 25722.037109375 ;
evos("SkLab","a_FoodProc","JPN") = 7193.4375 ;
evos("SkLab","a_FoodProc","IND") = 1701.68127441406 ;
evos("SkLab","a_FoodProc","SSA") = 9419.998046875 ;
evos("SkLab","a_FoodProc","ROW") = 78854.15625 ;
evos("SkLab","a_Energy","USA") = 16315.44921875 ;
evos("SkLab","a_Energy","EU_28") = 17062.15625 ;
evos("SkLab","a_Energy","CHN") = 20414.3515625 ;
evos("SkLab","a_Energy","JPN") = 5747.07275390625 ;
evos("SkLab","a_Energy","IND") = 33879.59765625 ;
evos("SkLab","a_Energy","SSA") = 4532.78857421875 ;
evos("SkLab","a_Energy","ROW") = 108299.484375 ;
evos("SkLab","a_Textiles","USA") = 4178.22900390625 ;
evos("SkLab","a_Textiles","EU_28") = 17152.96875 ;
evos("SkLab","a_Textiles","CHN") = 24256.28125 ;
evos("SkLab","a_Textiles","JPN") = 1881.08654785156 ;
evos("SkLab","a_Textiles","IND") = 2454.7412109375 ;
evos("SkLab","a_Textiles","SSA") = 1279.41174316406 ;
evos("SkLab","a_Textiles","ROW") = 41863.9453125 ;
evos("SkLab","a_Chem","USA") = 19257.443359375 ;
evos("SkLab","a_Chem","EU_28") = 50808.05859375 ;
evos("SkLab","a_Chem","CHN") = 26355.25390625 ;
evos("SkLab","a_Chem","JPN") = 8782.2294921875 ;
evos("SkLab","a_Chem","IND") = 1725.02197265625 ;
evos("SkLab","a_Chem","SSA") = 2060.49975585938 ;
evos("SkLab","a_Chem","ROW") = 58624.015625 ;
evos("SkLab","a_Manuf","USA") = 149504.203125 ;
evos("SkLab","a_Manuf","EU_28") = 260536.640625 ;
evos("SkLab","a_Manuf","CHN") = 141769.6875 ;
evos("SkLab","a_Manuf","JPN") = 60172.3125 ;
evos("SkLab","a_Manuf","IND") = 6600.00146484375 ;
evos("SkLab","a_Manuf","SSA") = 11167.1767578125 ;
evos("SkLab","a_Manuf","ROW") = 282783.4375 ;
evos("SkLab","a_ForestFish","USA") = 6610.771484375 ;
evos("SkLab","a_ForestFish","EU_28") = 1193.83642578125 ;
evos("SkLab","a_ForestFish","CHN") = 5775.7333984375 ;
evos("SkLab","a_ForestFish","JPN") = 77.6072540283203 ;
evos("SkLab","a_ForestFish","IND") = 48.2803230285645 ;
evos("SkLab","a_ForestFish","SSA") = 720.098876953125 ;
evos("SkLab","a_ForestFish","ROW") = 3788.52221679688 ;
evos("SkLab","a_Svces","USA") = 2846306.5 ;
evos("SkLab","a_Svces","EU_28") = 1767065.125 ;
evos("SkLab","a_Svces","CHN") = 1293143.625 ;
evos("SkLab","a_Svces","JPN") = 443392.71875 ;
evos("SkLab","a_Svces","IND") = 329585.71875 ;
evos("SkLab","a_Svces","SSA") = 191948.21875 ;
evos("SkLab","a_Svces","ROW") = 2775193.25 ;
evos("Capital","a_Rice","USA") = 899.266479492188 ;
evos("Capital","a_Rice","EU_28") = 1220.54162597656 ;
evos("Capital","a_Rice","CHN") = 30170.001953125 ;
evos("Capital","a_Rice","JPN") = 4269.34814453125 ;
evos("Capital","a_Rice","IND") = 4899.08740234375 ;
evos("Capital","a_Rice","SSA") = 4677.7734375 ;
evos("Capital","a_Rice","ROW") = 50956.37109375 ;
evos("Capital","a_Crops","USA") = 52233.46875 ;
evos("Capital","a_Crops","EU_28") = 47020.67578125 ;
evos("Capital","a_Crops","CHN") = 144947.359375 ;
evos("Capital","a_Crops","JPN") = 5914.15234375 ;
evos("Capital","a_Crops","IND") = 26714.86328125 ;
evos("Capital","a_Crops","SSA") = 51421.68359375 ;
evos("Capital","a_Crops","ROW") = 126290.734375 ;
evos("Capital","a_Livestock","USA") = 26776.759765625 ;
evos("Capital","a_Livestock","EU_28") = 35731.85546875 ;
evos("Capital","a_Livestock","CHN") = 51992.90234375 ;
evos("Capital","a_Livestock","JPN") = 2620.05859375 ;
evos("Capital","a_Livestock","IND") = 12126.416015625 ;
evos("Capital","a_Livestock","SSA") = 13726.5302734375 ;
evos("Capital","a_Livestock","ROW") = 59332.203125 ;
evos("Capital","a_FoodProc","USA") = 127413.515625 ;
evos("Capital","a_FoodProc","EU_28") = 175741.90625 ;
evos("Capital","a_FoodProc","CHN") = 158517.703125 ;
evos("Capital","a_FoodProc","JPN") = 51734.890625 ;
evos("Capital","a_FoodProc","IND") = 16826.630859375 ;
evos("Capital","a_FoodProc","SSA") = 70336.828125 ;
evos("Capital","a_FoodProc","ROW") = 379891.25 ;
evos("Capital","a_Energy","USA") = 211992.234375 ;
evos("Capital","a_Energy","EU_28") = 155744.421875 ;
evos("Capital","a_Energy","CHN") = 166394.8125 ;
evos("Capital","a_Energy","JPN") = 25948.23046875 ;
evos("Capital","a_Energy","IND") = 46284.46875 ;
evos("Capital","a_Energy","SSA") = 63379.25390625 ;
evos("Capital","a_Energy","ROW") = 1079842.5 ;
evos("Capital","a_Textiles","USA") = 5961.6220703125 ;
evos("Capital","a_Textiles","EU_28") = 47836.09765625 ;
evos("Capital","a_Textiles","CHN") = 89720.9453125 ;
evos("Capital","a_Textiles","JPN") = 1680.70434570312 ;
evos("Capital","a_Textiles","IND") = 25112.47265625 ;
evos("Capital","a_Textiles","SSA") = 6512.75 ;
evos("Capital","a_Textiles","ROW") = 126658.53125 ;
evos("Capital","a_Chem","USA") = 223271.703125 ;
evos("Capital","a_Chem","EU_28") = 267023.71875 ;
evos("Capital","a_Chem","CHN") = 204679.4375 ;
evos("Capital","a_Chem","JPN") = 56697.921875 ;
evos("Capital","a_Chem","IND") = 44627.78125 ;
evos("Capital","a_Chem","SSA") = 10749.1474609375 ;
evos("Capital","a_Chem","ROW") = 330735.28125 ;
evos("Capital","a_Manuf","USA") = 613005.9375 ;
evos("Capital","a_Manuf","EU_28") = 682320.375 ;
evos("Capital","a_Manuf","CHN") = 832917.9375 ;
evos("Capital","a_Manuf","JPN") = 273266.1875 ;
evos("Capital","a_Manuf","IND") = 102703.3515625 ;
evos("Capital","a_Manuf","SSA") = 69989.2421875 ;
evos("Capital","a_Manuf","ROW") = 1210678.5 ;
evos("Capital","a_ForestFish","USA") = 8948.974609375 ;
evos("Capital","a_ForestFish","EU_28") = 17928.609375 ;
evos("Capital","a_ForestFish","CHN") = 6619.42529296875 ;
evos("Capital","a_ForestFish","JPN") = 4652.7666015625 ;
evos("Capital","a_ForestFish","IND") = 20214.7265625 ;
evos("Capital","a_ForestFish","SSA") = 19041.544921875 ;
evos("Capital","a_ForestFish","ROW") = 62571.43359375 ;
evos("Capital","a_Svces","USA") = 5677908 ;
evos("Capital","a_Svces","EU_28") = 5227559 ;
evos("Capital","a_Svces","CHN") = 2363882.25 ;
evos("Capital","a_Svces","JPN") = 1716712.5 ;
evos("Capital","a_Svces","IND") = 683025.9375 ;
evos("Capital","a_Svces","SSA") = 379166.78125 ;
evos("Capital","a_Svces","ROW") = 6445030.5 ;
evos("NatRes","a_Energy","USA") = 58615.7890625 ;
evos("NatRes","a_Energy","EU_28") = 12277.4765625 ;
evos("NatRes","a_Energy","CHN") = 47527.31640625 ;
evos("NatRes","a_Energy","JPN") = 81.0277786254883 ;
evos("NatRes","a_Energy","IND") = 11375.404296875 ;
evos("NatRes","a_Energy","SSA") = 27126.47265625 ;
evos("NatRes","a_Energy","ROW") = 384213.375 ;
evos("NatRes","a_Manuf","USA") = 14171.775390625 ;
evos("NatRes","a_Manuf","EU_28") = 6557.8984375 ;
evos("NatRes","a_Manuf","CHN") = 13355.9990234375 ;
evos("NatRes","a_Manuf","JPN") = 855.612915039062 ;
evos("NatRes","a_Manuf","IND") = 2445.66455078125 ;
evos("NatRes","a_Manuf","SSA") = 4909.294921875 ;
evos("NatRes","a_Manuf","ROW") = 38005.75 ;
evos("NatRes","a_ForestFish","USA") = 5071.39697265625 ;
evos("NatRes","a_ForestFish","EU_28") = 7891.44140625 ;
evos("NatRes","a_ForestFish","CHN") = 39613.8046875 ;
evos("NatRes","a_ForestFish","JPN") = 2955.69677734375 ;
evos("NatRes","a_ForestFish","IND") = 10723.173828125 ;
evos("NatRes","a_ForestFish","SSA") = 9361.1689453125 ;
evos("NatRes","a_ForestFish","ROW") = 40157.16796875 ;
parameter vxsb(*,*,*) ;
vxsb("c_Rice","USA","USA") = 1.99999999495049E-6 ;
vxsb("c_Rice","USA","EU_28") = 44.973575592041 ;
vxsb("c_Rice","USA","CHN") = 1.26396679878235 ;
vxsb("c_Rice","USA","JPN") = 214.538589477539 ;
vxsb("c_Rice","USA","IND") = 1.70935010910034 ;
vxsb("c_Rice","USA","SSA") = 67.4374313354492 ;
vxsb("c_Rice","USA","ROW") = 1443.02465820312 ;
vxsb("c_Rice","EU_28","USA") = 25.2739868164062 ;
vxsb("c_Rice","EU_28","EU_28") = 1302.71069335938 ;
vxsb("c_Rice","EU_28","CHN") = 0.287983745336533 ;
vxsb("c_Rice","EU_28","JPN") = 0.202155783772469 ;
vxsb("c_Rice","EU_28","IND") = 0.926180064678192 ;
vxsb("c_Rice","EU_28","SSA") = 9.38745212554932 ;
vxsb("c_Rice","EU_28","ROW") = 236.856750488281 ;
vxsb("c_Rice","CHN","USA") = 3.62570118904114 ;
vxsb("c_Rice","CHN","EU_28") = 2.09355998039246 ;
vxsb("c_Rice","CHN","CHN") = 1.99999999495049E-6 ;
vxsb("c_Rice","CHN","JPN") = 3.50743269920349 ;
vxsb("c_Rice","CHN","IND") = 0.771757006645203 ;
vxsb("c_Rice","CHN","SSA") = 330.49609375 ;
vxsb("c_Rice","CHN","ROW") = 308.354156494141 ;
vxsb("c_Rice","JPN","USA") = 2.93056201934814 ;
vxsb("c_Rice","JPN","EU_28") = 3.00413870811462 ;
vxsb("c_Rice","JPN","CHN") = 2.28666734695435 ;
vxsb("c_Rice","JPN","JPN") = 1.99999999495049E-6 ;
vxsb("c_Rice","JPN","IND") = 0.0332564078271389 ;
vxsb("c_Rice","JPN","SSA") = 8.92286682128906 ;
vxsb("c_Rice","JPN","ROW") = 23.5417900085449 ;
vxsb("c_Rice","IND","USA") = 185.725128173828 ;
vxsb("c_Rice","IND","EU_28") = 418.232116699219 ;
vxsb("c_Rice","IND","CHN") = 0.509786665439606 ;
vxsb("c_Rice","IND","JPN") = 0.682196080684662 ;
vxsb("c_Rice","IND","IND") = 1.99999999495049E-6 ;
vxsb("c_Rice","IND","SSA") = 1857.39916992188 ;
vxsb("c_Rice","IND","ROW") = 4946.1875 ;
vxsb("c_Rice","SSA","USA") = 1.41751873493195 ;
vxsb("c_Rice","SSA","EU_28") = 4.17926406860352 ;
vxsb("c_Rice","SSA","CHN") = 2.08767437934875 ;
vxsb("c_Rice","SSA","JPN") = 0.229634881019592 ;
vxsb("c_Rice","SSA","IND") = 0.917802095413208 ;
vxsb("c_Rice","SSA","SSA") = 159.845230102539 ;
vxsb("c_Rice","SSA","ROW") = 3.6887321472168 ;
vxsb("c_Rice","ROW","USA") = 491.894897460938 ;
vxsb("c_Rice","ROW","EU_28") = 807.859252929688 ;
vxsb("c_Rice","ROW","CHN") = 1877.57897949219 ;
vxsb("c_Rice","ROW","JPN") = 150.159210205078 ;
vxsb("c_Rice","ROW","IND") = 6.6830677986145 ;
vxsb("c_Rice","ROW","SSA") = 4006.17431640625 ;
vxsb("c_Rice","ROW","ROW") = 5180.77978515625 ;
vxsb("c_Crops","USA","USA") = 7.00000009601354E-6 ;
vxsb("c_Crops","USA","EU_28") = 6003.3896484375 ;
vxsb("c_Crops","USA","CHN") = 15919.9296875 ;
vxsb("c_Crops","USA","JPN") = 6028.73876953125 ;
vxsb("c_Crops","USA","IND") = 1399.67236328125 ;
vxsb("c_Crops","USA","SSA") = 1081.45690917969 ;
vxsb("c_Crops","USA","ROW") = 38439.23828125 ;
vxsb("c_Crops","EU_28","USA") = 1171.49353027344 ;
vxsb("c_Crops","EU_28","EU_28") = 74566.375 ;
vxsb("c_Crops","EU_28","CHN") = 483.026031494141 ;
vxsb("c_Crops","EU_28","JPN") = 267.295471191406 ;
vxsb("c_Crops","EU_28","IND") = 297.451843261719 ;
vxsb("c_Crops","EU_28","SSA") = 1939.92749023438 ;
vxsb("c_Crops","EU_28","ROW") = 14962.142578125 ;
vxsb("c_Crops","CHN","USA") = 672.727233886719 ;
vxsb("c_Crops","CHN","EU_28") = 1255.49877929688 ;
vxsb("c_Crops","CHN","CHN") = 7.00000009601354E-6 ;
vxsb("c_Crops","CHN","JPN") = 1333.2060546875 ;
vxsb("c_Crops","CHN","IND") = 245.382995605469 ;
vxsb("c_Crops","CHN","SSA") = 159.208511352539 ;
vxsb("c_Crops","CHN","ROW") = 12313.98828125 ;
vxsb("c_Crops","JPN","USA") = 45.873119354248 ;
vxsb("c_Crops","JPN","EU_28") = 47.6829147338867 ;
vxsb("c_Crops","JPN","CHN") = 141.1171875 ;
vxsb("c_Crops","JPN","JPN") = 7.00000009601354E-6 ;
vxsb("c_Crops","JPN","IND") = 1.29150080680847 ;
vxsb("c_Crops","JPN","SSA") = 0.390444248914719 ;
vxsb("c_Crops","JPN","ROW") = 311.16796875 ;
vxsb("c_Crops","IND","USA") = 824.052551269531 ;
vxsb("c_Crops","IND","EU_28") = 1461.36767578125 ;
vxsb("c_Crops","IND","CHN") = 417.626159667969 ;
vxsb("c_Crops","IND","JPN") = 164.783142089844 ;
vxsb("c_Crops","IND","IND") = 7.00000009601354E-6 ;
vxsb("c_Crops","IND","SSA") = 151.246383666992 ;
vxsb("c_Crops","IND","ROW") = 6685.890625 ;
vxsb("c_Crops","SSA","USA") = 2183.51708984375 ;
vxsb("c_Crops","SSA","EU_28") = 9538.619140625 ;
vxsb("c_Crops","SSA","CHN") = 1492.1884765625 ;
vxsb("c_Crops","SSA","JPN") = 660.302124023438 ;
vxsb("c_Crops","SSA","IND") = 2236.642578125 ;
vxsb("c_Crops","SSA","SSA") = 2501.77612304688 ;
vxsb("c_Crops","SSA","ROW") = 10894.4375 ;
vxsb("c_Crops","ROW","USA") = 34864.45703125 ;
vxsb("c_Crops","ROW","EU_28") = 35480.33984375 ;
vxsb("c_Crops","ROW","CHN") = 39501.73828125 ;
vxsb("c_Crops","ROW","JPN") = 7739.8115234375 ;
vxsb("c_Crops","ROW","IND") = 5245.16748046875 ;
vxsb("c_Crops","ROW","SSA") = 3715.77075195312 ;
vxsb("c_Crops","ROW","ROW") = 83940.6796875 ;
vxsb("c_Livestock","USA","USA") = 3.99999998990097E-6 ;
vxsb("c_Livestock","USA","EU_28") = 489.805816650391 ;
vxsb("c_Livestock","USA","CHN") = 1243.55822753906 ;
vxsb("c_Livestock","USA","JPN") = 117.396553039551 ;
vxsb("c_Livestock","USA","IND") = 8.4329309463501 ;
vxsb("c_Livestock","USA","SSA") = 16.1641426086426 ;
vxsb("c_Livestock","USA","ROW") = 2310.00805664062 ;
vxsb("c_Livestock","EU_28","USA") = 609.1591796875 ;
vxsb("c_Livestock","EU_28","EU_28") = 14647.826171875 ;
vxsb("c_Livestock","EU_28","CHN") = 1215.56506347656 ;
vxsb("c_Livestock","EU_28","JPN") = 178.179504394531 ;
vxsb("c_Livestock","EU_28","IND") = 22.7664279937744 ;
vxsb("c_Livestock","EU_28","SSA") = 122.144027709961 ;
vxsb("c_Livestock","EU_28","ROW") = 4194.85205078125 ;
vxsb("c_Livestock","CHN","USA") = 214.492919921875 ;
vxsb("c_Livestock","CHN","EU_28") = 382.752532958984 ;
vxsb("c_Livestock","CHN","CHN") = 3.99999998990097E-6 ;
vxsb("c_Livestock","CHN","JPN") = 216.712554931641 ;
vxsb("c_Livestock","CHN","IND") = 5.3545560836792 ;
vxsb("c_Livestock","CHN","SSA") = 29.0461044311523 ;
vxsb("c_Livestock","CHN","ROW") = 789.678527832031 ;
vxsb("c_Livestock","JPN","USA") = 0.761225461959839 ;
vxsb("c_Livestock","JPN","EU_28") = 4.28531885147095 ;
vxsb("c_Livestock","JPN","CHN") = 15.7551002502441 ;
vxsb("c_Livestock","JPN","JPN") = 3.99999998990097E-6 ;
vxsb("c_Livestock","JPN","IND") = 0.122764766216278 ;
vxsb("c_Livestock","JPN","SSA") = 0.211897775530815 ;
vxsb("c_Livestock","JPN","ROW") = 143.252700805664 ;
vxsb("c_Livestock","IND","USA") = 86.274528503418 ;
vxsb("c_Livestock","IND","EU_28") = 15.3980159759521 ;
vxsb("c_Livestock","IND","CHN") = 3.22087264060974 ;
vxsb("c_Livestock","IND","JPN") = 13.3235807418823 ;
vxsb("c_Livestock","IND","IND") = 3.99999998990097E-6 ;
vxsb("c_Livestock","IND","SSA") = 3.11610817909241 ;
vxsb("c_Livestock","IND","ROW") = 155.070388793945 ;
vxsb("c_Livestock","SSA","USA") = 39.0745086669922 ;
vxsb("c_Livestock","SSA","EU_28") = 187.077026367188 ;
vxsb("c_Livestock","SSA","CHN") = 302.762451171875 ;
vxsb("c_Livestock","SSA","JPN") = 10.4589681625366 ;
vxsb("c_Livestock","SSA","IND") = 18.7795753479004 ;
vxsb("c_Livestock","SSA","SSA") = 417.440155029297 ;
vxsb("c_Livestock","SSA","ROW") = 1194.70141601562 ;
vxsb("c_Livestock","ROW","USA") = 3171.66625976562 ;
vxsb("c_Livestock","ROW","EU_28") = 1687.04248046875 ;
vxsb("c_Livestock","ROW","CHN") = 4188.1513671875 ;
vxsb("c_Livestock","ROW","JPN") = 367.600280761719 ;
vxsb("c_Livestock","ROW","IND") = 215.796798706055 ;
vxsb("c_Livestock","ROW","SSA") = 77.1518249511719 ;
vxsb("c_Livestock","ROW","ROW") = 5838.4990234375 ;
vxsb("c_FoodProc","USA","USA") = 7.00000009601354E-6 ;
vxsb("c_FoodProc","USA","EU_28") = 6040.09521484375 ;
vxsb("c_FoodProc","USA","CHN") = 4124.18994140625 ;
vxsb("c_FoodProc","USA","JPN") = 8298.05078125 ;
vxsb("c_FoodProc","USA","IND") = 504.395141601562 ;
vxsb("c_FoodProc","USA","SSA") = 1052.82202148438 ;
vxsb("c_FoodProc","USA","ROW") = 58111.7265625 ;
vxsb("c_FoodProc","EU_28","USA") = 23398.62890625 ;
vxsb("c_FoodProc","EU_28","EU_28") = 289786.625 ;
vxsb("c_FoodProc","EU_28","CHN") = 13131.25 ;
vxsb("c_FoodProc","EU_28","JPN") = 9396.4501953125 ;
vxsb("c_FoodProc","EU_28","IND") = 581.539916992188 ;
vxsb("c_FoodProc","EU_28","SSA") = 8628.2060546875 ;
vxsb("c_FoodProc","EU_28","ROW") = 71062.6953125 ;
vxsb("c_FoodProc","CHN","USA") = 7007.2646484375 ;
vxsb("c_FoodProc","CHN","EU_28") = 6217.501953125 ;
vxsb("c_FoodProc","CHN","CHN") = 7.00000009601354E-6 ;
vxsb("c_FoodProc","CHN","JPN") = 7412.94091796875 ;
vxsb("c_FoodProc","CHN","IND") = 225.446182250977 ;
vxsb("c_FoodProc","CHN","SSA") = 1941.92907714844 ;
vxsb("c_FoodProc","CHN","ROW") = 26301.56640625 ;
vxsb("c_FoodProc","JPN","USA") = 858.581115722656 ;
vxsb("c_FoodProc","JPN","EU_28") = 295.966461181641 ;
vxsb("c_FoodProc","JPN","CHN") = 664.158142089844 ;
vxsb("c_FoodProc","JPN","JPN") = 7.00000009601354E-6 ;
vxsb("c_FoodProc","JPN","IND") = 6.02203989028931 ;
vxsb("c_FoodProc","JPN","SSA") = 89.8871459960938 ;
vxsb("c_FoodProc","JPN","ROW") = 3225.572265625 ;
vxsb("c_FoodProc","IND","USA") = 3681.19482421875 ;
vxsb("c_FoodProc","IND","EU_28") = 2649.04296875 ;
vxsb("c_FoodProc","IND","CHN") = 628.959228515625 ;
vxsb("c_FoodProc","IND","JPN") = 695.61279296875 ;
vxsb("c_FoodProc","IND","IND") = 7.00000009601354E-6 ;
vxsb("c_FoodProc","IND","SSA") = 1075.75805664062 ;
vxsb("c_FoodProc","IND","ROW") = 9508.7314453125 ;
vxsb("c_FoodProc","SSA","USA") = 689.495910644531 ;
vxsb("c_FoodProc","SSA","EU_28") = 6891.16943359375 ;
vxsb("c_FoodProc","SSA","CHN") = 1188.36743164062 ;
vxsb("c_FoodProc","SSA","JPN") = 453.373352050781 ;
vxsb("c_FoodProc","SSA","IND") = 73.9994354248047 ;
vxsb("c_FoodProc","SSA","SSA") = 8452.591796875 ;
vxsb("c_FoodProc","SSA","ROW") = 2975.86499023438 ;
vxsb("c_FoodProc","ROW","USA") = 67380.921875 ;
vxsb("c_FoodProc","ROW","EU_28") = 58754.59375 ;
vxsb("c_FoodProc","ROW","CHN") = 34278.93359375 ;
vxsb("c_FoodProc","ROW","JPN") = 26121.1640625 ;
vxsb("c_FoodProc","ROW","IND") = 13684.0380859375 ;
vxsb("c_FoodProc","ROW","SSA") = 15051.66796875 ;
vxsb("c_FoodProc","ROW","ROW") = 181513.59375 ;
vxsb("c_Energy","USA","USA") = 6.00000021222513E-6 ;
vxsb("c_Energy","USA","EU_28") = 17274.267578125 ;
vxsb("c_Energy","USA","CHN") = 8890.4287109375 ;
vxsb("c_Energy","USA","JPN") = 6424.56982421875 ;
vxsb("c_Energy","USA","IND") = 2949.36669921875 ;
vxsb("c_Energy","USA","SSA") = 2227.43701171875 ;
vxsb("c_Energy","USA","ROW") = 106683.140625 ;
vxsb("c_Energy","EU_28","USA") = 9159.8876953125 ;
vxsb("c_Energy","EU_28","EU_28") = 93616.8515625 ;
vxsb("c_Energy","EU_28","CHN") = 2122.03247070312 ;
vxsb("c_Energy","EU_28","JPN") = 124.023277282715 ;
vxsb("c_Energy","EU_28","IND") = 477.446319580078 ;
vxsb("c_Energy","EU_28","SSA") = 9902.01171875 ;
vxsb("c_Energy","EU_28","ROW") = 39961.3671875 ;
vxsb("c_Energy","CHN","USA") = 2249.47290039062 ;
vxsb("c_Energy","CHN","EU_28") = 1136.97241210938 ;
vxsb("c_Energy","CHN","CHN") = 6.00000021222513E-6 ;
vxsb("c_Energy","CHN","JPN") = 1177.37097167969 ;
vxsb("c_Energy","CHN","IND") = 1034.23828125 ;
vxsb("c_Energy","CHN","SSA") = 3167.375 ;
vxsb("c_Energy","CHN","ROW") = 22290.38671875 ;
vxsb("c_Energy","JPN","USA") = 808.005065917969 ;
vxsb("c_Energy","JPN","EU_28") = 237.985717773438 ;
vxsb("c_Energy","JPN","CHN") = 1136.78979492188 ;
vxsb("c_Energy","JPN","JPN") = 6.00000021222513E-6 ;
vxsb("c_Energy","JPN","IND") = 213.156036376953 ;
vxsb("c_Energy","JPN","SSA") = 14.8695039749146 ;
vxsb("c_Energy","JPN","ROW") = 6658.38671875 ;
vxsb("c_Energy","IND","USA") = 3604.37426757812 ;
vxsb("c_Energy","IND","EU_28") = 1867.52209472656 ;
vxsb("c_Energy","IND","CHN") = 504.673187255859 ;
vxsb("c_Energy","IND","JPN") = 1240.53918457031 ;
vxsb("c_Energy","IND","IND") = 6.00000021222513E-6 ;
vxsb("c_Energy","IND","SSA") = 2457.11645507812 ;
vxsb("c_Energy","IND","ROW") = 13651.4638671875 ;
vxsb("c_Energy","SSA","USA") = 11351.8037109375 ;
vxsb("c_Energy","SSA","EU_28") = 20940.73828125 ;
vxsb("c_Energy","SSA","CHN") = 29591.001953125 ;
vxsb("c_Energy","SSA","JPN") = 1081.45568847656 ;
vxsb("c_Energy","SSA","IND") = 15719.8525390625 ;
vxsb("c_Energy","SSA","SSA") = 10998.9287109375 ;
vxsb("c_Energy","SSA","ROW") = 16123.4892578125 ;
vxsb("c_Energy","ROW","USA") = 179528.625 ;
vxsb("c_Energy","ROW","EU_28") = 336092.40625 ;
vxsb("c_Energy","ROW","CHN") = 179737.734375 ;
vxsb("c_Energy","ROW","JPN") = 127864.9609375 ;
vxsb("c_Energy","ROW","IND") = 100292.21875 ;
vxsb("c_Energy","ROW","SSA") = 17784.6640625 ;
vxsb("c_Energy","ROW","ROW") = 396160.5 ;
vxsb("c_Textiles","USA","USA") = 3.00000010611257E-6 ;
vxsb("c_Textiles","USA","EU_28") = 2945.99877929688 ;
vxsb("c_Textiles","USA","CHN") = 1628.69909667969 ;
vxsb("c_Textiles","USA","JPN") = 473.076263427734 ;
vxsb("c_Textiles","USA","IND") = 190.563125610352 ;
vxsb("c_Textiles","USA","SSA") = 238.439117431641 ;
vxsb("c_Textiles","USA","ROW") = 20115.611328125 ;
vxsb("c_Textiles","EU_28","USA") = 10732.7216796875 ;
vxsb("c_Textiles","EU_28","EU_28") = 179170.65625 ;
vxsb("c_Textiles","EU_28","CHN") = 9241.16015625 ;
vxsb("c_Textiles","EU_28","JPN") = 4686.79248046875 ;
vxsb("c_Textiles","EU_28","IND") = 715.635192871094 ;
vxsb("c_Textiles","EU_28","SSA") = 1948.80480957031 ;
vxsb("c_Textiles","EU_28","ROW") = 45067.29296875 ;
vxsb("c_Textiles","CHN","USA") = 72444.53125 ;
vxsb("c_Textiles","CHN","EU_28") = 69866.3046875 ;
vxsb("c_Textiles","CHN","CHN") = 3.00000010611257E-6 ;
vxsb("c_Textiles","CHN","JPN") = 31034.146484375 ;
vxsb("c_Textiles","CHN","IND") = 6096.498046875 ;
vxsb("c_Textiles","CHN","SSA") = 19423.982421875 ;
vxsb("c_Textiles","CHN","ROW") = 167993.828125 ;
vxsb("c_Textiles","JPN","USA") = 688.285461425781 ;
vxsb("c_Textiles","JPN","EU_28") = 802.544128417969 ;
vxsb("c_Textiles","JPN","CHN") = 3390.5712890625 ;
vxsb("c_Textiles","JPN","JPN") = 3.00000010611257E-6 ;
vxsb("c_Textiles","JPN","IND") = 116.302017211914 ;
vxsb("c_Textiles","JPN","SSA") = 166.791320800781 ;
vxsb("c_Textiles","JPN","ROW") = 3706.44750976562 ;
vxsb("c_Textiles","IND","USA") = 9294.326171875 ;
vxsb("c_Textiles","IND","EU_28") = 12595.984375 ;
vxsb("c_Textiles","IND","CHN") = 2197.755859375 ;
vxsb("c_Textiles","IND","JPN") = 567.515014648438 ;
vxsb("c_Textiles","IND","IND") = 3.00000010611257E-6 ;
vxsb("c_Textiles","IND","SSA") = 2163.5341796875 ;
vxsb("c_Textiles","IND","ROW") = 15447.765625 ;
vxsb("c_Textiles","SSA","USA") = 1232.49353027344 ;
vxsb("c_Textiles","SSA","EU_28") = 1427.04614257812 ;
vxsb("c_Textiles","SSA","CHN") = 304.757873535156 ;
vxsb("c_Textiles","SSA","JPN") = 24.6536521911621 ;
vxsb("c_Textiles","SSA","IND") = 59.2138175964355 ;
vxsb("c_Textiles","SSA","SSA") = 2209.59375 ;
vxsb("c_Textiles","SSA","ROW") = 447.138702392578 ;
vxsb("c_Textiles","ROW","USA") = 75102.109375 ;
vxsb("c_Textiles","ROW","EU_28") = 93887.140625 ;
vxsb("c_Textiles","ROW","CHN") = 24385.228515625 ;
vxsb("c_Textiles","ROW","JPN") = 15001.3408203125 ;
vxsb("c_Textiles","ROW","IND") = 2305.7060546875 ;
vxsb("c_Textiles","ROW","SSA") = 2777.58715820312 ;
vxsb("c_Textiles","ROW","ROW") = 84717.859375 ;
vxsb("c_Chem","USA","USA") = 3.00000010611257E-6 ;
vxsb("c_Chem","USA","EU_28") = 57563.609375 ;
vxsb("c_Chem","USA","CHN") = 18629.796875 ;
vxsb("c_Chem","USA","JPN") = 13690.64453125 ;
vxsb("c_Chem","USA","IND") = 4051.42529296875 ;
vxsb("c_Chem","USA","SSA") = 1950.41687011719 ;
vxsb("c_Chem","USA","ROW") = 142658.234375 ;
vxsb("c_Chem","EU_28","USA") = 98441.8203125 ;
vxsb("c_Chem","EU_28","EU_28") = 589822.75 ;
vxsb("c_Chem","EU_28","CHN") = 47310.01171875 ;
vxsb("c_Chem","EU_28","JPN") = 23066.1484375 ;
vxsb("c_Chem","EU_28","IND") = 7210.421875 ;
vxsb("c_Chem","EU_28","SSA") = 13802.8173828125 ;
vxsb("c_Chem","EU_28","ROW") = 218808.15625 ;
vxsb("c_Chem","CHN","USA") = 31703.888671875 ;
vxsb("c_Chem","CHN","EU_28") = 28238.193359375 ;
vxsb("c_Chem","CHN","CHN") = 3.00000010611257E-6 ;
vxsb("c_Chem","CHN","JPN") = 13877.2783203125 ;
vxsb("c_Chem","CHN","IND") = 12627.654296875 ;
vxsb("c_Chem","CHN","SSA") = 9186.109375 ;
vxsb("c_Chem","CHN","ROW") = 100932.15625 ;
vxsb("c_Chem","JPN","USA") = 12140.728515625 ;
vxsb("c_Chem","JPN","EU_28") = 9335.7373046875 ;
vxsb("c_Chem","JPN","CHN") = 31360.166015625 ;
vxsb("c_Chem","JPN","JPN") = 3.00000010611257E-6 ;
vxsb("c_Chem","JPN","IND") = 2003.28381347656 ;
vxsb("c_Chem","JPN","SSA") = 490.531494140625 ;
vxsb("c_Chem","JPN","ROW") = 45184.3515625 ;
vxsb("c_Chem","IND","USA") = 10060.4208984375 ;
vxsb("c_Chem","IND","EU_28") = 8072.1494140625 ;
vxsb("c_Chem","IND","CHN") = 2842.82666015625 ;
vxsb("c_Chem","IND","JPN") = 899.709899902344 ;
vxsb("c_Chem","IND","IND") = 3.00000010611257E-6 ;
vxsb("c_Chem","IND","SSA") = 4734.60107421875 ;
vxsb("c_Chem","IND","ROW") = 20227.42578125 ;
vxsb("c_Chem","SSA","USA") = 1054.38879394531 ;
vxsb("c_Chem","SSA","EU_28") = 2589.7919921875 ;
vxsb("c_Chem","SSA","CHN") = 607.929016113281 ;
vxsb("c_Chem","SSA","JPN") = 86.3570861816406 ;
vxsb("c_Chem","SSA","IND") = 750.878051757812 ;
vxsb("c_Chem","SSA","SSA") = 6785.23291015625 ;
vxsb("c_Chem","SSA","ROW") = 2160.77880859375 ;
vxsb("c_Chem","ROW","USA") = 108219.5078125 ;
vxsb("c_Chem","ROW","EU_28") = 124058.0625 ;
vxsb("c_Chem","ROW","CHN") = 129713.296875 ;
vxsb("c_Chem","ROW","JPN") = 25513.568359375 ;
vxsb("c_Chem","ROW","IND") = 25410.169921875 ;
vxsb("c_Chem","ROW","SSA") = 11398.6875 ;
vxsb("c_Chem","ROW","ROW") = 241794.875 ;
vxsb("c_Manuf","USA","USA") = 1.29999998534913E-5 ;
vxsb("c_Manuf","USA","EU_28") = 147518.390625 ;
vxsb("c_Manuf","USA","CHN") = 73166.96875 ;
vxsb("c_Manuf","USA","JPN") = 37466.13671875 ;
vxsb("c_Manuf","USA","IND") = 16732.99609375 ;
vxsb("c_Manuf","USA","SSA") = 7846.3994140625 ;
vxsb("c_Manuf","USA","ROW") = 579015.875 ;
vxsb("c_Manuf","EU_28","USA") = 261452.59375 ;
vxsb("c_Manuf","EU_28","EU_28") = 1889556.875 ;
vxsb("c_Manuf","EU_28","CHN") = 194633.828125 ;
vxsb("c_Manuf","EU_28","JPN") = 41259.3359375 ;
vxsb("c_Manuf","EU_28","IND") = 34675.5546875 ;
vxsb("c_Manuf","EU_28","SSA") = 41977.578125 ;
vxsb("c_Manuf","EU_28","ROW") = 717691.625 ;
vxsb("c_Manuf","CHN","USA") = 344823.21875 ;
vxsb("c_Manuf","CHN","EU_28") = 237306.71875 ;
vxsb("c_Manuf","CHN","CHN") = 1.29999998534913E-5 ;
vxsb("c_Manuf","CHN","JPN") = 120771.796875 ;
vxsb("c_Manuf","CHN","IND") = 55197.28125 ;
vxsb("c_Manuf","CHN","SSA") = 47418.73046875 ;
vxsb("c_Manuf","CHN","ROW") = 677704.0625 ;
vxsb("c_Manuf","JPN","USA") = 117311.25 ;
vxsb("c_Manuf","JPN","EU_28") = 64193.13671875 ;
vxsb("c_Manuf","JPN","CHN") = 160044.984375 ;
vxsb("c_Manuf","JPN","JPN") = 1.29999998534913E-5 ;
vxsb("c_Manuf","JPN","IND") = 7284.44189453125 ;
vxsb("c_Manuf","JPN","SSA") = 6220.37109375 ;
vxsb("c_Manuf","JPN","ROW") = 246635.734375 ;
vxsb("c_Manuf","IND","USA") = 19836.427734375 ;
vxsb("c_Manuf","IND","EU_28") = 20170.908203125 ;
vxsb("c_Manuf","IND","CHN") = 11570.0732421875 ;
vxsb("c_Manuf","IND","JPN") = 1730.19958496094 ;
vxsb("c_Manuf","IND","IND") = 1.29999998534913E-5 ;
vxsb("c_Manuf","IND","SSA") = 6423.3759765625 ;
vxsb("c_Manuf","IND","ROW") = 62179.4296875 ;
vxsb("c_Manuf","SSA","USA") = 7867.123046875 ;
vxsb("c_Manuf","SSA","EU_28") = 28398.607421875 ;
vxsb("c_Manuf","SSA","CHN") = 40102.8515625 ;
vxsb("c_Manuf","SSA","JPN") = 5023.1123046875 ;
vxsb("c_Manuf","SSA","IND") = 11887.166015625 ;
vxsb("c_Manuf","SSA","SSA") = 21093.431640625 ;
vxsb("c_Manuf","SSA","ROW") = 41559.296875 ;
vxsb("c_Manuf","ROW","USA") = 684972 ;
vxsb("c_Manuf","ROW","EU_28") = 435463.03125 ;
vxsb("c_Manuf","ROW","CHN") = 675140.9375 ;
vxsb("c_Manuf","ROW","JPN") = 128928.5859375 ;
vxsb("c_Manuf","ROW","IND") = 89220.9453125 ;
vxsb("c_Manuf","ROW","SSA") = 30972.662109375 ;
vxsb("c_Manuf","ROW","ROW") = 962144.375 ;
vxsb("c_ForestFish","USA","USA") = 1.99999999495049E-6 ;
vxsb("c_ForestFish","USA","EU_28") = 309.009002685547 ;
vxsb("c_ForestFish","USA","CHN") = 759.132202148438 ;
vxsb("c_ForestFish","USA","JPN") = 492.758850097656 ;
vxsb("c_ForestFish","USA","IND") = 13.3630828857422 ;
vxsb("c_ForestFish","USA","SSA") = 2.57011318206787 ;
vxsb("c_ForestFish","USA","ROW") = 1021.24530029297 ;
vxsb("c_ForestFish","EU_28","USA") = 445.212463378906 ;
vxsb("c_ForestFish","EU_28","EU_28") = 10978.318359375 ;
vxsb("c_ForestFish","EU_28","CHN") = 650.568176269531 ;
vxsb("c_ForestFish","EU_28","JPN") = 31.0346565246582 ;
vxsb("c_ForestFish","EU_28","IND") = 35.2470550537109 ;
vxsb("c_ForestFish","EU_28","SSA") = 20.6545085906982 ;
vxsb("c_ForestFish","EU_28","ROW") = 930.631225585938 ;
vxsb("c_ForestFish","CHN","USA") = 30.9080200195312 ;
vxsb("c_ForestFish","CHN","EU_28") = 77.8215560913086 ;
vxsb("c_ForestFish","CHN","CHN") = 1.99999999495049E-6 ;
vxsb("c_ForestFish","CHN","JPN") = 441.268768310547 ;
vxsb("c_ForestFish","CHN","IND") = 1.51383852958679 ;
vxsb("c_ForestFish","CHN","SSA") = 1.57301723957062 ;
vxsb("c_ForestFish","CHN","ROW") = 813.548400878906 ;
vxsb("c_ForestFish","JPN","USA") = 47.6144332885742 ;
vxsb("c_ForestFish","JPN","EU_28") = 19.8704853057861 ;
vxsb("c_ForestFish","JPN","CHN") = 130.77766418457 ;
vxsb("c_ForestFish","JPN","JPN") = 1.99999999495049E-6 ;
vxsb("c_ForestFish","JPN","IND") = 1.7176353931427 ;
vxsb("c_ForestFish","JPN","SSA") = 0.555203139781952 ;
vxsb("c_ForestFish","JPN","ROW") = 334.041870117188 ;
vxsb("c_ForestFish","IND","USA") = 40.8773460388184 ;
vxsb("c_ForestFish","IND","EU_28") = 63.6767272949219 ;
vxsb("c_ForestFish","IND","CHN") = 75.4296340942383 ;
vxsb("c_ForestFish","IND","JPN") = 5.16047334671021 ;
vxsb("c_ForestFish","IND","IND") = 1.99999999495049E-6 ;
vxsb("c_ForestFish","IND","SSA") = 4.44913339614868 ;
vxsb("c_ForestFish","IND","ROW") = 166.771209716797 ;
vxsb("c_ForestFish","SSA","USA") = 59.6747703552246 ;
vxsb("c_ForestFish","SSA","EU_28") = 379.517242431641 ;
vxsb("c_ForestFish","SSA","CHN") = 1857.95471191406 ;
vxsb("c_ForestFish","SSA","JPN") = 22.0867729187012 ;
vxsb("c_ForestFish","SSA","IND") = 176.822296142578 ;
vxsb("c_ForestFish","SSA","SSA") = 150.386627197266 ;
vxsb("c_ForestFish","SSA","ROW") = 529.092834472656 ;
vxsb("c_ForestFish","ROW","USA") = 2665.54858398438 ;
vxsb("c_ForestFish","ROW","EU_28") = 6151.43798828125 ;
vxsb("c_ForestFish","ROW","CHN") = 7654.5625 ;
vxsb("c_ForestFish","ROW","JPN") = 1646.94104003906 ;
vxsb("c_ForestFish","ROW","IND") = 1024.54736328125 ;
vxsb("c_ForestFish","ROW","SSA") = 90.4198455810547 ;
vxsb("c_ForestFish","ROW","ROW") = 5442.82373046875 ;
vxsb("c_Svces","USA","USA") = 1.80000006366754E-5 ;
vxsb("c_Svces","USA","EU_28") = 223204.1875 ;
vxsb("c_Svces","USA","CHN") = 57874.48828125 ;
vxsb("c_Svces","USA","JPN") = 42788.98046875 ;
vxsb("c_Svces","USA","IND") = 19128.24609375 ;
vxsb("c_Svces","USA","SSA") = 18268.017578125 ;
vxsb("c_Svces","USA","ROW") = 342195.65625 ;
vxsb("c_Svces","EU_28","USA") = 199294.921875 ;
vxsb("c_Svces","EU_28","EU_28") = 1038910.9375 ;
vxsb("c_Svces","EU_28","CHN") = 52260.73046875 ;
vxsb("c_Svces","EU_28","JPN") = 32276.919921875 ;
vxsb("c_Svces","EU_28","IND") = 17354.9375 ;
vxsb("c_Svces","EU_28","SSA") = 29422.033203125 ;
vxsb("c_Svces","EU_28","ROW") = 431834.03125 ;
vxsb("c_Svces","CHN","USA") = 19002.94921875 ;
vxsb("c_Svces","CHN","EU_28") = 35278.9296875 ;
vxsb("c_Svces","CHN","CHN") = 1.80000006366754E-5 ;
vxsb("c_Svces","CHN","JPN") = 7994.64453125 ;
vxsb("c_Svces","CHN","IND") = 2810.67749023438 ;
vxsb("c_Svces","CHN","SSA") = 4210.1689453125 ;
vxsb("c_Svces","CHN","ROW") = 105634.671875 ;
vxsb("c_Svces","JPN","USA") = 20040.076171875 ;
vxsb("c_Svces","JPN","EU_28") = 18437.458984375 ;
vxsb("c_Svces","JPN","CHN") = 15669.7412109375 ;
vxsb("c_Svces","JPN","JPN") = 1.80000006366754E-5 ;
vxsb("c_Svces","JPN","IND") = 984.914978027344 ;
vxsb("c_Svces","JPN","SSA") = 1856.79931640625 ;
vxsb("c_Svces","JPN","ROW") = 50100.16796875 ;
vxsb("c_Svces","IND","USA") = 27956.552734375 ;
vxsb("c_Svces","IND","EU_28") = 28240.134765625 ;
vxsb("c_Svces","IND","CHN") = 2620.39575195312 ;
vxsb("c_Svces","IND","JPN") = 2652.44750976562 ;
vxsb("c_Svces","IND","IND") = 1.80000006366754E-5 ;
vxsb("c_Svces","IND","SSA") = 4550.8828125 ;
vxsb("c_Svces","IND","ROW") = 43900.6875 ;
vxsb("c_Svces","SSA","USA") = 9115.646484375 ;
vxsb("c_Svces","SSA","EU_28") = 23896.33984375 ;
vxsb("c_Svces","SSA","CHN") = 5176.61669921875 ;
vxsb("c_Svces","SSA","JPN") = 1891.78247070312 ;
vxsb("c_Svces","SSA","IND") = 2612.72021484375 ;
vxsb("c_Svces","SSA","SSA") = 3836.8310546875 ;
vxsb("c_Svces","SSA","ROW") = 19791.9765625 ;
vxsb("c_Svces","ROW","USA") = 271711.40625 ;
vxsb("c_Svces","ROW","EU_28") = 361803.84375 ;
vxsb("c_Svces","ROW","CHN") = 179073.515625 ;
vxsb("c_Svces","ROW","JPN") = 57829.4140625 ;
vxsb("c_Svces","ROW","IND") = 33393.578125 ;
vxsb("c_Svces","ROW","SSA") = 26256.544921875 ;
vxsb("c_Svces","ROW","ROW") = 442174.28125 ;
parameter vfob(*,*,*) ;
vfob("c_Rice","USA","USA") = 1.99999999495049E-6 ;
vfob("c_Rice","USA","EU_28") = 44.9768905639648 ;
vfob("c_Rice","USA","CHN") = 1.26397144794464 ;
vfob("c_Rice","USA","JPN") = 214.538589477539 ;
vfob("c_Rice","USA","IND") = 1.70935213565826 ;
vfob("c_Rice","USA","SSA") = 67.4378433227539 ;
vfob("c_Rice","USA","ROW") = 1444.21984863281 ;
vfob("c_Rice","EU_28","USA") = 25.2738285064697 ;
vfob("c_Rice","EU_28","EU_28") = 1302.71069335938 ;
vfob("c_Rice","EU_28","CHN") = 0.287981659173965 ;
vfob("c_Rice","EU_28","JPN") = 0.202155604958534 ;
vfob("c_Rice","EU_28","IND") = 0.926178455352783 ;
vfob("c_Rice","EU_28","SSA") = 9.38712024688721 ;
vfob("c_Rice","EU_28","ROW") = 237.039413452148 ;
vfob("c_Rice","CHN","USA") = 3.60446262359619 ;
vfob("c_Rice","CHN","EU_28") = 2.05948543548584 ;
vfob("c_Rice","CHN","CHN") = 1.99999999495049E-6 ;
vfob("c_Rice","CHN","JPN") = 3.49867844581604 ;
vfob("c_Rice","CHN","IND") = 0.751747965812683 ;
vfob("c_Rice","CHN","SSA") = 330.453674316406 ;
vfob("c_Rice","CHN","ROW") = 305.318359375 ;
vfob("c_Rice","JPN","USA") = 2.93056201934814 ;
vfob("c_Rice","JPN","EU_28") = 3.00413870811462 ;
vfob("c_Rice","JPN","CHN") = 2.28666734695435 ;
vfob("c_Rice","JPN","JPN") = 1.99999999495049E-6 ;
vfob("c_Rice","JPN","IND") = 0.0332564041018486 ;
vfob("c_Rice","JPN","SSA") = 8.92286682128906 ;
vfob("c_Rice","JPN","ROW") = 23.5417900085449 ;
vfob("c_Rice","IND","USA") = 185.722061157227 ;
vfob("c_Rice","IND","EU_28") = 418.225799560547 ;
vfob("c_Rice","IND","CHN") = 0.506439089775085 ;
vfob("c_Rice","IND","JPN") = 0.681850612163544 ;
vfob("c_Rice","IND","IND") = 1.99999999495049E-6 ;
vfob("c_Rice","IND","SSA") = 1857.38598632812 ;
vfob("c_Rice","IND","ROW") = 4945.712890625 ;
vfob("c_Rice","SSA","USA") = 1.41781961917877 ;
vfob("c_Rice","SSA","EU_28") = 4.18017053604126 ;
vfob("c_Rice","SSA","CHN") = 2.08795428276062 ;
vfob("c_Rice","SSA","JPN") = 0.229691684246063 ;
vfob("c_Rice","SSA","IND") = 0.917960941791534 ;
vfob("c_Rice","SSA","SSA") = 159.866622924805 ;
vfob("c_Rice","SSA","ROW") = 3.68942213058472 ;
vfob("c_Rice","ROW","USA") = 491.895263671875 ;
vfob("c_Rice","ROW","EU_28") = 807.881225585938 ;
vfob("c_Rice","ROW","CHN") = 1877.57861328125 ;
vfob("c_Rice","ROW","JPN") = 150.159225463867 ;
vfob("c_Rice","ROW","IND") = 6.68308353424072 ;
vfob("c_Rice","ROW","SSA") = 4006.18139648438 ;
vfob("c_Rice","ROW","ROW") = 5181.4365234375 ;
vfob("c_Crops","USA","USA") = 7.00000009601354E-6 ;
vfob("c_Crops","USA","EU_28") = 6033.2705078125 ;
vfob("c_Crops","USA","CHN") = 16105.7314453125 ;
vfob("c_Crops","USA","JPN") = 6046.05615234375 ;
vfob("c_Crops","USA","IND") = 1399.68249511719 ;
vfob("c_Crops","USA","SSA") = 1081.62048339844 ;
vfob("c_Crops","USA","ROW") = 38550.00390625 ;
vfob("c_Crops","EU_28","USA") = 1171.44067382812 ;
vfob("c_Crops","EU_28","EU_28") = 74566.375 ;
vfob("c_Crops","EU_28","CHN") = 483.42724609375 ;
vfob("c_Crops","EU_28","JPN") = 267.299072265625 ;
vfob("c_Crops","EU_28","IND") = 297.477203369141 ;
vfob("c_Crops","EU_28","SSA") = 1939.93395996094 ;
vfob("c_Crops","EU_28","ROW") = 14962.947265625 ;
vfob("c_Crops","CHN","USA") = 671.867980957031 ;
vfob("c_Crops","CHN","EU_28") = 1246.77795410156 ;
vfob("c_Crops","CHN","CHN") = 7.00000009601354E-6 ;
vfob("c_Crops","CHN","JPN") = 1329.47766113281 ;
vfob("c_Crops","CHN","IND") = 244.782745361328 ;
vfob("c_Crops","CHN","SSA") = 159.127563476562 ;
vfob("c_Crops","CHN","ROW") = 12273.763671875 ;
vfob("c_Crops","JPN","USA") = 45.873119354248 ;
vfob("c_Crops","JPN","EU_28") = 47.6829147338867 ;
vfob("c_Crops","JPN","CHN") = 141.1171875 ;
vfob("c_Crops","JPN","JPN") = 7.00000009601354E-6 ;
vfob("c_Crops","JPN","IND") = 1.29150080680847 ;
vfob("c_Crops","JPN","SSA") = 0.390444248914719 ;
vfob("c_Crops","JPN","ROW") = 311.16796875 ;
vfob("c_Crops","IND","USA") = 827.172546386719 ;
vfob("c_Crops","IND","EU_28") = 1464.447265625 ;
vfob("c_Crops","IND","CHN") = 418.109619140625 ;
vfob("c_Crops","IND","JPN") = 164.803009033203 ;
vfob("c_Crops","IND","IND") = 7.00000009601354E-6 ;
vfob("c_Crops","IND","SSA") = 151.582077026367 ;
vfob("c_Crops","IND","ROW") = 6702.408203125 ;
vfob("c_Crops","SSA","USA") = 2183.54150390625 ;
vfob("c_Crops","SSA","EU_28") = 9538.51953125 ;
vfob("c_Crops","SSA","CHN") = 1500.4404296875 ;
vfob("c_Crops","SSA","JPN") = 660.360900878906 ;
vfob("c_Crops","SSA","IND") = 2236.67358398438 ;
vfob("c_Crops","SSA","SSA") = 2502.08447265625 ;
vfob("c_Crops","SSA","ROW") = 10895.0712890625 ;
vfob("c_Crops","ROW","USA") = 34859.296875 ;
vfob("c_Crops","ROW","EU_28") = 35535.1328125 ;
vfob("c_Crops","ROW","CHN") = 40107.484375 ;
vfob("c_Crops","ROW","JPN") = 7735.4482421875 ;
vfob("c_Crops","ROW","IND") = 5244.9228515625 ;
vfob("c_Crops","ROW","SSA") = 3715.732421875 ;
vfob("c_Crops","ROW","ROW") = 84070.2734375 ;
vfob("c_Livestock","USA","USA") = 3.99999998990097E-6 ;
vfob("c_Livestock","USA","EU_28") = 489.798706054688 ;
vfob("c_Livestock","USA","CHN") = 1243.51879882812 ;
vfob("c_Livestock","USA","JPN") = 117.396545410156 ;
vfob("c_Livestock","USA","IND") = 8.42626094818115 ;
vfob("c_Livestock","USA","SSA") = 16.1566219329834 ;
vfob("c_Livestock","USA","ROW") = 2310.00537109375 ;
vfob("c_Livestock","EU_28","USA") = 609.142211914062 ;
vfob("c_Livestock","EU_28","EU_28") = 14647.826171875 ;
vfob("c_Livestock","EU_28","CHN") = 1214.94104003906 ;
vfob("c_Livestock","EU_28","JPN") = 178.178848266602 ;
vfob("c_Livestock","EU_28","IND") = 22.770788192749 ;
vfob("c_Livestock","EU_28","SSA") = 122.133659362793 ;
vfob("c_Livestock","EU_28","ROW") = 4194.916015625 ;
vfob("c_Livestock","CHN","USA") = 214.495223999023 ;
vfob("c_Livestock","CHN","EU_28") = 382.75634765625 ;
vfob("c_Livestock","CHN","CHN") = 3.99999998990097E-6 ;
vfob("c_Livestock","CHN","JPN") = 216.71923828125 ;
vfob("c_Livestock","CHN","IND") = 5.35640859603882 ;
vfob("c_Livestock","CHN","SSA") = 29.0474147796631 ;
vfob("c_Livestock","CHN","ROW") = 789.702514648438 ;
vfob("c_Livestock","JPN","USA") = 0.761225461959839 ;
vfob("c_Livestock","JPN","EU_28") = 4.28531885147095 ;
vfob("c_Livestock","JPN","CHN") = 15.7551002502441 ;
vfob("c_Livestock","JPN","JPN") = 3.99999998990097E-6 ;
vfob("c_Livestock","JPN","IND") = 0.122764758765697 ;
vfob("c_Livestock","JPN","SSA") = 0.211897775530815 ;
vfob("c_Livestock","JPN","ROW") = 143.252700805664 ;
vfob("c_Livestock","IND","USA") = 86.2746353149414 ;
vfob("c_Livestock","IND","EU_28") = 15.398325920105 ;
vfob("c_Livestock","IND","CHN") = 3.22098922729492 ;
vfob("c_Livestock","IND","JPN") = 13.3235931396484 ;
vfob("c_Livestock","IND","IND") = 3.99999998990097E-6 ;
vfob("c_Livestock","IND","SSA") = 3.11619400978088 ;
vfob("c_Livestock","IND","ROW") = 155.071380615234 ;
vfob("c_Livestock","SSA","USA") = 39.0764846801758 ;
vfob("c_Livestock","SSA","EU_28") = 187.076766967773 ;
vfob("c_Livestock","SSA","CHN") = 302.765045166016 ;
vfob("c_Livestock","SSA","JPN") = 10.4595861434937 ;
vfob("c_Livestock","SSA","IND") = 18.7799549102783 ;
vfob("c_Livestock","SSA","SSA") = 417.440673828125 ;
vfob("c_Livestock","SSA","ROW") = 1194.70422363281 ;
vfob("c_Livestock","ROW","USA") = 3171.64868164062 ;
vfob("c_Livestock","ROW","EU_28") = 1687.72546386719 ;
vfob("c_Livestock","ROW","CHN") = 4191.05908203125 ;
vfob("c_Livestock","ROW","JPN") = 367.508880615234 ;
vfob("c_Livestock","ROW","IND") = 215.927993774414 ;
vfob("c_Livestock","ROW","SSA") = 77.151496887207 ;
vfob("c_Livestock","ROW","ROW") = 5839.0546875 ;
vfob("c_FoodProc","USA","USA") = 7.00000009601354E-6 ;
vfob("c_FoodProc","USA","EU_28") = 6040.09521484375 ;
vfob("c_FoodProc","USA","CHN") = 4124.18994140625 ;
vfob("c_FoodProc","USA","JPN") = 8298.05078125 ;
vfob("c_FoodProc","USA","IND") = 504.395141601562 ;
vfob("c_FoodProc","USA","SSA") = 1052.82202148438 ;
vfob("c_FoodProc","USA","ROW") = 58111.7265625 ;
vfob("c_FoodProc","EU_28","USA") = 23398.62890625 ;
vfob("c_FoodProc","EU_28","EU_28") = 289786.625 ;
vfob("c_FoodProc","EU_28","CHN") = 13131.25 ;
vfob("c_FoodProc","EU_28","JPN") = 9396.4501953125 ;
vfob("c_FoodProc","EU_28","IND") = 581.539916992188 ;
vfob("c_FoodProc","EU_28","SSA") = 8628.2060546875 ;
vfob("c_FoodProc","EU_28","ROW") = 71062.6953125 ;
vfob("c_FoodProc","CHN","USA") = 7007.2646484375 ;
vfob("c_FoodProc","CHN","EU_28") = 6217.501953125 ;
vfob("c_FoodProc","CHN","CHN") = 7.00000009601354E-6 ;
vfob("c_FoodProc","CHN","JPN") = 7412.94091796875 ;
vfob("c_FoodProc","CHN","IND") = 225.446182250977 ;
vfob("c_FoodProc","CHN","SSA") = 1941.92907714844 ;
vfob("c_FoodProc","CHN","ROW") = 26301.56640625 ;
vfob("c_FoodProc","JPN","USA") = 858.581115722656 ;
vfob("c_FoodProc","JPN","EU_28") = 295.966461181641 ;
vfob("c_FoodProc","JPN","CHN") = 664.158142089844 ;
vfob("c_FoodProc","JPN","JPN") = 7.00000009601354E-6 ;
vfob("c_FoodProc","JPN","IND") = 6.02203989028931 ;
vfob("c_FoodProc","JPN","SSA") = 89.8871459960938 ;
vfob("c_FoodProc","JPN","ROW") = 3225.572265625 ;
vfob("c_FoodProc","IND","USA") = 3681.19482421875 ;
vfob("c_FoodProc","IND","EU_28") = 2649.04296875 ;
vfob("c_FoodProc","IND","CHN") = 628.959228515625 ;
vfob("c_FoodProc","IND","JPN") = 695.61279296875 ;
vfob("c_FoodProc","IND","IND") = 7.00000009601354E-6 ;
vfob("c_FoodProc","IND","SSA") = 1075.75805664062 ;
vfob("c_FoodProc","IND","ROW") = 9508.7314453125 ;
vfob("c_FoodProc","SSA","USA") = 689.495910644531 ;
vfob("c_FoodProc","SSA","EU_28") = 6891.16943359375 ;
vfob("c_FoodProc","SSA","CHN") = 1188.36743164062 ;
vfob("c_FoodProc","SSA","JPN") = 453.373352050781 ;
vfob("c_FoodProc","SSA","IND") = 73.9994354248047 ;
vfob("c_FoodProc","SSA","SSA") = 8452.591796875 ;
vfob("c_FoodProc","SSA","ROW") = 2975.86499023438 ;
vfob("c_FoodProc","ROW","USA") = 67373.6875 ;
vfob("c_FoodProc","ROW","EU_28") = 58743.6640625 ;
vfob("c_FoodProc","ROW","CHN") = 34278.375 ;
vfob("c_FoodProc","ROW","JPN") = 26119.740234375 ;
vfob("c_FoodProc","ROW","IND") = 13683.9345703125 ;
vfob("c_FoodProc","ROW","SSA") = 15050.9228515625 ;
vfob("c_FoodProc","ROW","ROW") = 181503.734375 ;
vfob("c_Energy","USA","USA") = 6.00000021222513E-6 ;
vfob("c_Energy","USA","EU_28") = 18447.859375 ;
vfob("c_Energy","USA","CHN") = 9279.0595703125 ;
vfob("c_Energy","USA","JPN") = 6799.443359375 ;
vfob("c_Energy","USA","IND") = 3250.0673828125 ;
vfob("c_Energy","USA","SSA") = 2258.70556640625 ;
vfob("c_Energy","USA","ROW") = 108434.28125 ;
vfob("c_Energy","EU_28","USA") = 9830.814453125 ;
vfob("c_Energy","EU_28","EU_28") = 93616.8515625 ;
vfob("c_Energy","EU_28","CHN") = 2175.59594726562 ;
vfob("c_Energy","EU_28","JPN") = 127.441619873047 ;
vfob("c_Energy","EU_28","IND") = 491.571502685547 ;
vfob("c_Energy","EU_28","SSA") = 10286.3916015625 ;
vfob("c_Energy","EU_28","ROW") = 41725.3046875 ;
vfob("c_Energy","CHN","USA") = 2643.18969726562 ;
vfob("c_Energy","CHN","EU_28") = 1324.66149902344 ;
vfob("c_Energy","CHN","CHN") = 6.00000021222513E-6 ;
vfob("c_Energy","CHN","JPN") = 1415.66613769531 ;
vfob("c_Energy","CHN","IND") = 1211.646484375 ;
vfob("c_Energy","CHN","SSA") = 3708.27856445312 ;
vfob("c_Energy","CHN","ROW") = 25902.10546875 ;
vfob("c_Energy","JPN","USA") = 808.005065917969 ;
vfob("c_Energy","JPN","EU_28") = 237.985717773438 ;
vfob("c_Energy","JPN","CHN") = 1136.78979492188 ;
vfob("c_Energy","JPN","JPN") = 6.00000021222513E-6 ;
vfob("c_Energy","JPN","IND") = 213.156036376953 ;
vfob("c_Energy","JPN","SSA") = 14.8695039749146 ;
vfob("c_Energy","JPN","ROW") = 6658.38671875 ;
vfob("c_Energy","IND","USA") = 4415.65966796875 ;
vfob("c_Energy","IND","EU_28") = 2283.45874023438 ;
vfob("c_Energy","IND","CHN") = 617.328552246094 ;
vfob("c_Energy","IND","JPN") = 1520.08483886719 ;
vfob("c_Energy","IND","IND") = 6.00000021222513E-6 ;
vfob("c_Energy","IND","SSA") = 3006.85400390625 ;
vfob("c_Energy","IND","ROW") = 16522.48828125 ;
vfob("c_Energy","SSA","USA") = 11352.24609375 ;
vfob("c_Energy","SSA","EU_28") = 20954.482421875 ;
vfob("c_Energy","SSA","CHN") = 29576.71484375 ;
vfob("c_Energy","SSA","JPN") = 1081.47790527344 ;
vfob("c_Energy","SSA","IND") = 15726.1357421875 ;
vfob("c_Energy","SSA","SSA") = 11051.451171875 ;
vfob("c_Energy","SSA","ROW") = 16112.5947265625 ;
vfob("c_Energy","ROW","USA") = 179825.21875 ;
vfob("c_Energy","ROW","EU_28") = 339356.90625 ;
vfob("c_Energy","ROW","CHN") = 180209.1875 ;
vfob("c_Energy","ROW","JPN") = 128177.8671875 ;
vfob("c_Energy","ROW","IND") = 100509.4921875 ;
vfob("c_Energy","ROW","SSA") = 17829.849609375 ;
vfob("c_Energy","ROW","ROW") = 399147.5 ;
vfob("c_Textiles","USA","USA") = 3.00000010611257E-6 ;
vfob("c_Textiles","USA","EU_28") = 2964.63549804688 ;
vfob("c_Textiles","USA","CHN") = 1642.61145019531 ;
vfob("c_Textiles","USA","JPN") = 475.983032226562 ;
vfob("c_Textiles","USA","IND") = 192.035934448242 ;
vfob("c_Textiles","USA","SSA") = 239.208572387695 ;
vfob("c_Textiles","USA","ROW") = 20179.44921875 ;
vfob("c_Textiles","EU_28","USA") = 10754.294921875 ;
vfob("c_Textiles","EU_28","EU_28") = 179170.65625 ;
vfob("c_Textiles","EU_28","CHN") = 9252.9951171875 ;
vfob("c_Textiles","EU_28","JPN") = 4694.8408203125 ;
vfob("c_Textiles","EU_28","IND") = 717.472595214844 ;
vfob("c_Textiles","EU_28","SSA") = 1955.1416015625 ;
vfob("c_Textiles","EU_28","ROW") = 45178.7421875 ;
vfob("c_Textiles","CHN","USA") = 72738.8828125 ;
vfob("c_Textiles","CHN","EU_28") = 70089.9609375 ;
vfob("c_Textiles","CHN","CHN") = 3.00000010611257E-6 ;
vfob("c_Textiles","CHN","JPN") = 31072.953125 ;
vfob("c_Textiles","CHN","IND") = 6160.26953125 ;
vfob("c_Textiles","CHN","SSA") = 19553.9296875 ;
vfob("c_Textiles","CHN","ROW") = 169048.6875 ;
vfob("c_Textiles","JPN","USA") = 688.285461425781 ;
vfob("c_Textiles","JPN","EU_28") = 802.544128417969 ;
vfob("c_Textiles","JPN","CHN") = 3390.5712890625 ;
vfob("c_Textiles","JPN","JPN") = 3.00000010611257E-6 ;
vfob("c_Textiles","JPN","IND") = 116.302017211914 ;
vfob("c_Textiles","JPN","SSA") = 166.791320800781 ;
vfob("c_Textiles","JPN","ROW") = 3706.44750976562 ;
vfob("c_Textiles","IND","USA") = 9613.7431640625 ;
vfob("c_Textiles","IND","EU_28") = 13055.3759765625 ;
vfob("c_Textiles","IND","CHN") = 2268.99096679688 ;
vfob("c_Textiles","IND","JPN") = 587.90185546875 ;
vfob("c_Textiles","IND","IND") = 3.00000010611257E-6 ;
vfob("c_Textiles","IND","SSA") = 2236.47583007812 ;
vfob("c_Textiles","IND","ROW") = 15970.611328125 ;
vfob("c_Textiles","SSA","USA") = 1272.61120605469 ;
vfob("c_Textiles","SSA","EU_28") = 1436.23571777344 ;
vfob("c_Textiles","SSA","CHN") = 311.703002929688 ;
vfob("c_Textiles","SSA","JPN") = 25.4079399108887 ;
vfob("c_Textiles","SSA","IND") = 60.8352966308594 ;
vfob("c_Textiles","SSA","SSA") = 2261.15185546875 ;
vfob("c_Textiles","SSA","ROW") = 457.332824707031 ;
vfob("c_Textiles","ROW","USA") = 75519.3125 ;
vfob("c_Textiles","ROW","EU_28") = 94375.7890625 ;
vfob("c_Textiles","ROW","CHN") = 24532.201171875 ;
vfob("c_Textiles","ROW","JPN") = 15072.65625 ;
vfob("c_Textiles","ROW","IND") = 2322.39501953125 ;
vfob("c_Textiles","ROW","SSA") = 2790.87817382812 ;
vfob("c_Textiles","ROW","ROW") = 85227.28125 ;
vfob("c_Chem","USA","USA") = 3.00000010611257E-6 ;
vfob("c_Chem","USA","EU_28") = 61509.09765625 ;
vfob("c_Chem","USA","CHN") = 19329.296875 ;
vfob("c_Chem","USA","JPN") = 14474.16015625 ;
vfob("c_Chem","USA","IND") = 4193.458984375 ;
vfob("c_Chem","USA","SSA") = 2018.63903808594 ;
vfob("c_Chem","USA","ROW") = 145321.375 ;
vfob("c_Chem","EU_28","USA") = 98506.6328125 ;
vfob("c_Chem","EU_28","EU_28") = 589822.75 ;
vfob("c_Chem","EU_28","CHN") = 47340.98828125 ;
vfob("c_Chem","EU_28","JPN") = 23079.994140625 ;
vfob("c_Chem","EU_28","IND") = 7215.681640625 ;
vfob("c_Chem","EU_28","SSA") = 13816.087890625 ;
vfob("c_Chem","EU_28","ROW") = 218975.765625 ;
vfob("c_Chem","CHN","USA") = 33171.83203125 ;
vfob("c_Chem","CHN","EU_28") = 29630.658203125 ;
vfob("c_Chem","CHN","CHN") = 3.00000010611257E-6 ;
vfob("c_Chem","CHN","JPN") = 14592.251953125 ;
vfob("c_Chem","CHN","IND") = 13335.6806640625 ;
vfob("c_Chem","CHN","SSA") = 9634.904296875 ;
vfob("c_Chem","CHN","ROW") = 106148.3359375 ;
vfob("c_Chem","JPN","USA") = 12140.728515625 ;
vfob("c_Chem","JPN","EU_28") = 9335.7373046875 ;
vfob("c_Chem","JPN","CHN") = 31360.166015625 ;
vfob("c_Chem","JPN","JPN") = 3.00000010611257E-6 ;
vfob("c_Chem","JPN","IND") = 2003.28381347656 ;
vfob("c_Chem","JPN","SSA") = 490.531494140625 ;
vfob("c_Chem","JPN","ROW") = 45184.3515625 ;
vfob("c_Chem","IND","USA") = 10587.9169921875 ;
vfob("c_Chem","IND","EU_28") = 8452.287109375 ;
vfob("c_Chem","IND","CHN") = 2947.88256835938 ;
vfob("c_Chem","IND","JPN") = 937.3740234375 ;
vfob("c_Chem","IND","IND") = 3.00000010611257E-6 ;
vfob("c_Chem","IND","SSA") = 4987.4140625 ;
vfob("c_Chem","IND","ROW") = 21129.521484375 ;
vfob("c_Chem","SSA","USA") = 1060.59912109375 ;
vfob("c_Chem","SSA","EU_28") = 2630.51904296875 ;
vfob("c_Chem","SSA","CHN") = 617.595397949219 ;
vfob("c_Chem","SSA","JPN") = 86.4842681884766 ;
vfob("c_Chem","SSA","IND") = 756.384216308594 ;
vfob("c_Chem","SSA","SSA") = 6901.04345703125 ;
vfob("c_Chem","SSA","ROW") = 2189.39526367188 ;
vfob("c_Chem","ROW","USA") = 108570.1171875 ;
vfob("c_Chem","ROW","EU_28") = 124727.1171875 ;
vfob("c_Chem","ROW","CHN") = 130171.359375 ;
vfob("c_Chem","ROW","JPN") = 25664.908203125 ;
vfob("c_Chem","ROW","IND") = 25570.865234375 ;
vfob("c_Chem","ROW","SSA") = 11458.283203125 ;
vfob("c_Chem","ROW","ROW") = 243235.828125 ;
vfob("c_Manuf","USA","USA") = 1.29999998534913E-5 ;
vfob("c_Manuf","USA","EU_28") = 154211.84375 ;
vfob("c_Manuf","USA","CHN") = 76487.8046875 ;
vfob("c_Manuf","USA","JPN") = 39510.2265625 ;
vfob("c_Manuf","USA","IND") = 17447.58984375 ;
vfob("c_Manuf","USA","SSA") = 8182.05615234375 ;
vfob("c_Manuf","USA","ROW") = 591842.0625 ;
vfob("c_Manuf","EU_28","USA") = 261621.4375 ;
vfob("c_Manuf","EU_28","EU_28") = 1889556.875 ;
vfob("c_Manuf","EU_28","CHN") = 194711.09375 ;
vfob("c_Manuf","EU_28","JPN") = 41281.796875 ;
vfob("c_Manuf","EU_28","IND") = 34643.51953125 ;
vfob("c_Manuf","EU_28","SSA") = 42005.53125 ;
vfob("c_Manuf","EU_28","ROW") = 718052.5 ;
vfob("c_Manuf","CHN","USA") = 350293.375 ;
vfob("c_Manuf","CHN","EU_28") = 241437.140625 ;
vfob("c_Manuf","CHN","CHN") = 1.29999998534913E-5 ;
vfob("c_Manuf","CHN","JPN") = 122221.859375 ;
vfob("c_Manuf","CHN","IND") = 55729.98046875 ;
vfob("c_Manuf","CHN","SSA") = 48639.96875 ;
vfob("c_Manuf","CHN","ROW") = 688218.875 ;
vfob("c_Manuf","JPN","USA") = 117311.25 ;
vfob("c_Manuf","JPN","EU_28") = 64193.13671875 ;
vfob("c_Manuf","JPN","CHN") = 160044.984375 ;
vfob("c_Manuf","JPN","JPN") = 1.29999998534913E-5 ;
vfob("c_Manuf","JPN","IND") = 7284.44189453125 ;
vfob("c_Manuf","JPN","SSA") = 6220.37109375 ;
vfob("c_Manuf","JPN","ROW") = 246635.734375 ;
vfob("c_Manuf","IND","USA") = 20825.646484375 ;
vfob("c_Manuf","IND","EU_28") = 21132.0859375 ;
vfob("c_Manuf","IND","CHN") = 11978.26171875 ;
vfob("c_Manuf","IND","JPN") = 1790.72839355469 ;
vfob("c_Manuf","IND","IND") = 1.29999998534913E-5 ;
vfob("c_Manuf","IND","SSA") = 6793.9970703125 ;
vfob("c_Manuf","IND","ROW") = 64948.75390625 ;
vfob("c_Manuf","SSA","USA") = 7886.19873046875 ;
vfob("c_Manuf","SSA","EU_28") = 28611.78515625 ;
vfob("c_Manuf","SSA","CHN") = 40339.0703125 ;
vfob("c_Manuf","SSA","JPN") = 5026.380859375 ;
vfob("c_Manuf","SSA","IND") = 11998.7900390625 ;
vfob("c_Manuf","SSA","SSA") = 21277.279296875 ;
vfob("c_Manuf","SSA","ROW") = 42153.46875 ;
vfob("c_Manuf","ROW","USA") = 686990.0625 ;
vfob("c_Manuf","ROW","EU_28") = 438340.5625 ;
vfob("c_Manuf","ROW","CHN") = 677040 ;
vfob("c_Manuf","ROW","JPN") = 129822.203125 ;
vfob("c_Manuf","ROW","IND") = 89544 ;
vfob("c_Manuf","ROW","SSA") = 31307.162109375 ;
vfob("c_Manuf","ROW","ROW") = 970280.875 ;
vfob("c_ForestFish","USA","USA") = 1.99999999495049E-6 ;
vfob("c_ForestFish","USA","EU_28") = 307.537261962891 ;
vfob("c_ForestFish","USA","CHN") = 750.075927734375 ;
vfob("c_ForestFish","USA","JPN") = 485.022705078125 ;
vfob("c_ForestFish","USA","IND") = 13.3529205322266 ;
vfob("c_ForestFish","USA","SSA") = 2.54230713844299 ;
vfob("c_ForestFish","USA","ROW") = 1021.11077880859 ;
vfob("c_ForestFish","EU_28","USA") = 446.888916015625 ;
vfob("c_ForestFish","EU_28","EU_28") = 10978.318359375 ;
vfob("c_ForestFish","EU_28","CHN") = 649.384155273438 ;
vfob("c_ForestFish","EU_28","JPN") = 31.0804405212402 ;
vfob("c_ForestFish","EU_28","IND") = 34.7845115661621 ;
vfob("c_ForestFish","EU_28","SSA") = 20.5583648681641 ;
vfob("c_ForestFish","EU_28","ROW") = 930.438110351562 ;
vfob("c_ForestFish","CHN","USA") = 30.5679473876953 ;
vfob("c_ForestFish","CHN","EU_28") = 76.8349914550781 ;
vfob("c_ForestFish","CHN","CHN") = 1.99999999495049E-6 ;
vfob("c_ForestFish","CHN","JPN") = 441.057983398438 ;
vfob("c_ForestFish","CHN","IND") = 1.50814688205719 ;
vfob("c_ForestFish","CHN","SSA") = 1.56387627124786 ;
vfob("c_ForestFish","CHN","ROW") = 814.158325195312 ;
vfob("c_ForestFish","JPN","USA") = 47.614429473877 ;
vfob("c_ForestFish","JPN","EU_28") = 19.8704833984375 ;
vfob("c_ForestFish","JPN","CHN") = 130.77766418457 ;
vfob("c_ForestFish","JPN","JPN") = 1.99999999495049E-6 ;
vfob("c_ForestFish","JPN","IND") = 1.7176353931427 ;
vfob("c_ForestFish","JPN","SSA") = 0.555203080177307 ;
vfob("c_ForestFish","JPN","ROW") = 334.041839599609 ;
vfob("c_ForestFish","IND","USA") = 41.3876571655273 ;
vfob("c_ForestFish","IND","EU_28") = 64.5669937133789 ;
vfob("c_ForestFish","IND","CHN") = 76.6084747314453 ;
vfob("c_ForestFish","IND","JPN") = 5.23910713195801 ;
vfob("c_ForestFish","IND","IND") = 1.99999999495049E-6 ;
vfob("c_ForestFish","IND","SSA") = 4.50546216964722 ;
vfob("c_ForestFish","IND","ROW") = 167.564849853516 ;
vfob("c_ForestFish","SSA","USA") = 59.9378814697266 ;
vfob("c_ForestFish","SSA","EU_28") = 383.569702148438 ;
vfob("c_ForestFish","SSA","CHN") = 1904.59594726562 ;
vfob("c_ForestFish","SSA","JPN") = 22.5174312591553 ;
vfob("c_ForestFish","SSA","IND") = 177.690246582031 ;
vfob("c_ForestFish","SSA","SSA") = 154.167083740234 ;
vfob("c_ForestFish","SSA","ROW") = 550.757690429688 ;
vfob("c_ForestFish","ROW","USA") = 2666.06665039062 ;
vfob("c_ForestFish","ROW","EU_28") = 6143.64794921875 ;
vfob("c_ForestFish","ROW","CHN") = 7677.1513671875 ;
vfob("c_ForestFish","ROW","JPN") = 1648.36083984375 ;
vfob("c_ForestFish","ROW","IND") = 1027.41186523438 ;
vfob("c_ForestFish","ROW","SSA") = 90.466552734375 ;
vfob("c_ForestFish","ROW","ROW") = 5441.61279296875 ;
vfob("c_Svces","USA","USA") = 1.80000006366754E-5 ;
vfob("c_Svces","USA","EU_28") = 223204.1875 ;
vfob("c_Svces","USA","CHN") = 57874.48828125 ;
vfob("c_Svces","USA","JPN") = 42788.98046875 ;
vfob("c_Svces","USA","IND") = 19128.24609375 ;
vfob("c_Svces","USA","SSA") = 18268.017578125 ;
vfob("c_Svces","USA","ROW") = 342195.65625 ;
vfob("c_Svces","EU_28","USA") = 199294.921875 ;
vfob("c_Svces","EU_28","EU_28") = 1038910.9375 ;
vfob("c_Svces","EU_28","CHN") = 52260.73046875 ;
vfob("c_Svces","EU_28","JPN") = 32276.919921875 ;
vfob("c_Svces","EU_28","IND") = 17354.9375 ;
vfob("c_Svces","EU_28","SSA") = 29422.033203125 ;
vfob("c_Svces","EU_28","ROW") = 431834.03125 ;
vfob("c_Svces","CHN","USA") = 19002.94921875 ;
vfob("c_Svces","CHN","EU_28") = 35278.9296875 ;
vfob("c_Svces","CHN","CHN") = 1.80000006366754E-5 ;
vfob("c_Svces","CHN","JPN") = 7994.64453125 ;
vfob("c_Svces","CHN","IND") = 2810.67749023438 ;
vfob("c_Svces","CHN","SSA") = 4210.1689453125 ;
vfob("c_Svces","CHN","ROW") = 105634.671875 ;
vfob("c_Svces","JPN","USA") = 20040.076171875 ;
vfob("c_Svces","JPN","EU_28") = 18437.458984375 ;
vfob("c_Svces","JPN","CHN") = 15669.7412109375 ;
vfob("c_Svces","JPN","JPN") = 1.80000006366754E-5 ;
vfob("c_Svces","JPN","IND") = 984.914978027344 ;
vfob("c_Svces","JPN","SSA") = 1856.79931640625 ;
vfob("c_Svces","JPN","ROW") = 50100.16796875 ;
vfob("c_Svces","IND","USA") = 27956.552734375 ;
vfob("c_Svces","IND","EU_28") = 28240.134765625 ;
vfob("c_Svces","IND","CHN") = 2620.39575195312 ;
vfob("c_Svces","IND","JPN") = 2652.44750976562 ;
vfob("c_Svces","IND","IND") = 1.80000006366754E-5 ;
vfob("c_Svces","IND","SSA") = 4550.8828125 ;
vfob("c_Svces","IND","ROW") = 43900.6875 ;
vfob("c_Svces","SSA","USA") = 9115.646484375 ;
vfob("c_Svces","SSA","EU_28") = 23896.33984375 ;
vfob("c_Svces","SSA","CHN") = 5176.61669921875 ;
vfob("c_Svces","SSA","JPN") = 1891.78247070312 ;
vfob("c_Svces","SSA","IND") = 2612.72021484375 ;
vfob("c_Svces","SSA","SSA") = 3836.8310546875 ;
vfob("c_Svces","SSA","ROW") = 19791.9765625 ;
vfob("c_Svces","ROW","USA") = 271711.40625 ;
vfob("c_Svces","ROW","EU_28") = 361803.84375 ;
vfob("c_Svces","ROW","CHN") = 179073.515625 ;
vfob("c_Svces","ROW","JPN") = 57829.4140625 ;
vfob("c_Svces","ROW","IND") = 33393.578125 ;
vfob("c_Svces","ROW","SSA") = 26256.544921875 ;
vfob("c_Svces","ROW","ROW") = 442174.28125 ;
parameter vcif(*,*,*) ;
vcif("c_Rice","USA","USA") = 1.99999999495049E-6 ;
vcif("c_Rice","USA","EU_28") = 49.0671043395996 ;
vcif("c_Rice","USA","CHN") = 1.3361382484436 ;
vcif("c_Rice","USA","JPN") = 234.443237304688 ;
vcif("c_Rice","USA","IND") = 1.84907722473145 ;
vcif("c_Rice","USA","SSA") = 73.6843948364258 ;
vcif("c_Rice","USA","ROW") = 1534.65234375 ;
vcif("c_Rice","EU_28","USA") = 27.5782527923584 ;
vcif("c_Rice","EU_28","EU_28") = 1360.71203613281 ;
vcif("c_Rice","EU_28","CHN") = 0.28812363743782 ;
vcif("c_Rice","EU_28","JPN") = 0.217989608645439 ;
vcif("c_Rice","EU_28","IND") = 1.00636053085327 ;
vcif("c_Rice","EU_28","SSA") = 10.2446908950806 ;
vcif("c_Rice","EU_28","ROW") = 255.699951171875 ;
vcif("c_Rice","CHN","USA") = 3.85886669158936 ;
vcif("c_Rice","CHN","EU_28") = 2.12207841873169 ;
vcif("c_Rice","CHN","CHN") = 1.99999999495049E-6 ;
vcif("c_Rice","CHN","JPN") = 3.79036235809326 ;
vcif("c_Rice","CHN","IND") = 0.755514979362488 ;
vcif("c_Rice","CHN","SSA") = 361.05810546875 ;
vcif("c_Rice","CHN","ROW") = 331.032531738281 ;
vcif("c_Rice","JPN","USA") = 3.19345378875732 ;
vcif("c_Rice","JPN","EU_28") = 3.26902675628662 ;
vcif("c_Rice","JPN","CHN") = 2.39101314544678 ;
vcif("c_Rice","JPN","JPN") = 1.99999999495049E-6 ;
vcif("c_Rice","JPN","IND") = 0.0336324013769627 ;
vcif("c_Rice","JPN","SSA") = 9.74629878997803 ;
vcif("c_Rice","JPN","ROW") = 25.6393775939941 ;
vcif("c_Rice","IND","USA") = 202.916641235352 ;
vcif("c_Rice","IND","EU_28") = 456.957305908203 ;
vcif("c_Rice","IND","CHN") = 0.507356107234955 ;
vcif("c_Rice","IND","JPN") = 0.740376591682434 ;
vcif("c_Rice","IND","IND") = 1.99999999495049E-6 ;
vcif("c_Rice","IND","SSA") = 2029.68542480469 ;
vcif("c_Rice","IND","ROW") = 5402.431640625 ;
vcif("c_Rice","SSA","USA") = 1.45332002639771 ;
vcif("c_Rice","SSA","EU_28") = 4.20027256011963 ;
vcif("c_Rice","SSA","CHN") = 2.08795428276062 ;
vcif("c_Rice","SSA","JPN") = 0.229691684246063 ;
vcif("c_Rice","SSA","IND") = 0.917960941791534 ;
vcif("c_Rice","SSA","SSA") = 174.230041503906 ;
vcif("c_Rice","SSA","ROW") = 3.71612977981567 ;
vcif("c_Rice","ROW","USA") = 535.803283691406 ;
vcif("c_Rice","ROW","EU_28") = 880.338439941406 ;
vcif("c_Rice","ROW","CHN") = 2049.7451171875 ;
vcif("c_Rice","ROW","JPN") = 163.802947998047 ;
vcif("c_Rice","ROW","IND") = 6.75567150115967 ;
vcif("c_Rice","ROW","SSA") = 4377.46240234375 ;
vcif("c_Rice","ROW","ROW") = 5652.9482421875 ;
vcif("c_Crops","USA","USA") = 7.00000009601354E-6 ;
vcif("c_Crops","USA","EU_28") = 6342.673828125 ;
vcif("c_Crops","USA","CHN") = 17350.080078125 ;
vcif("c_Crops","USA","JPN") = 6682.10791015625 ;
vcif("c_Crops","USA","IND") = 1458.35266113281 ;
vcif("c_Crops","USA","SSA") = 1148.6552734375 ;
vcif("c_Crops","USA","ROW") = 41029.03515625 ;
vcif("c_Crops","EU_28","USA") = 1265.92211914062 ;
vcif("c_Crops","EU_28","EU_28") = 80757.09375 ;
vcif("c_Crops","EU_28","CHN") = 522.006042480469 ;
vcif("c_Crops","EU_28","JPN") = 282.896209716797 ;
vcif("c_Crops","EU_28","IND") = 320.009887695312 ;
vcif("c_Crops","EU_28","SSA") = 2081.26391601562 ;
vcif("c_Crops","EU_28","ROW") = 16243.6279296875 ;
vcif("c_Crops","CHN","USA") = 722.880187988281 ;
vcif("c_Crops","CHN","EU_28") = 1331.4892578125 ;
vcif("c_Crops","CHN","CHN") = 7.00000009601354E-6 ;
vcif("c_Crops","CHN","JPN") = 1478.71533203125 ;
vcif("c_Crops","CHN","IND") = 274.327789306641 ;
vcif("c_Crops","CHN","SSA") = 172.904907226562 ;
vcif("c_Crops","CHN","ROW") = 13906.6962890625 ;
vcif("c_Crops","JPN","USA") = 52.1757507324219 ;
vcif("c_Crops","JPN","EU_28") = 51.062816619873 ;
vcif("c_Crops","JPN","CHN") = 148.79719543457 ;
vcif("c_Crops","JPN","JPN") = 7.00000009601354E-6 ;
vcif("c_Crops","JPN","IND") = 1.32978880405426 ;
vcif("c_Crops","JPN","SSA") = 0.399938225746155 ;
vcif("c_Crops","JPN","ROW") = 351.273651123047 ;
vcif("c_Crops","IND","USA") = 876.132995605469 ;
vcif("c_Crops","IND","EU_28") = 1547.92419433594 ;
vcif("c_Crops","IND","CHN") = 472.742004394531 ;
vcif("c_Crops","IND","JPN") = 171.520721435547 ;
vcif("c_Crops","IND","IND") = 7.00000009601354E-6 ;
vcif("c_Crops","IND","SSA") = 161.289489746094 ;
vcif("c_Crops","IND","ROW") = 7135.33740234375 ;
vcif("c_Crops","SSA","USA") = 2250.392578125 ;
vcif("c_Crops","SSA","EU_28") = 10375.265625 ;
vcif("c_Crops","SSA","CHN") = 1617.43640136719 ;
vcif("c_Crops","SSA","JPN") = 702.8994140625 ;
vcif("c_Crops","SSA","IND") = 2317.19946289062 ;
vcif("c_Crops","SSA","SSA") = 2727.17309570312 ;
vcif("c_Crops","SSA","ROW") = 11552.1181640625 ;
vcif("c_Crops","ROW","USA") = 37456.3359375 ;
vcif("c_Crops","ROW","EU_28") = 38917.1875 ;
vcif("c_Crops","ROW","CHN") = 43771.25 ;
vcif("c_Crops","ROW","JPN") = 8518.89453125 ;
vcif("c_Crops","ROW","IND") = 5570.84326171875 ;
vcif("c_Crops","ROW","SSA") = 3972.12524414062 ;
vcif("c_Crops","ROW","ROW") = 91896.09375 ;
vcif("c_Livestock","USA","USA") = 3.99999998990097E-6 ;
vcif("c_Livestock","USA","EU_28") = 516.357177734375 ;
vcif("c_Livestock","USA","CHN") = 1307.63610839844 ;
vcif("c_Livestock","USA","JPN") = 124.39794921875 ;
vcif("c_Livestock","USA","IND") = 8.83312606811523 ;
vcif("c_Livestock","USA","SSA") = 23.35129737854 ;
vcif("c_Livestock","USA","ROW") = 2469.76049804688 ;
vcif("c_Livestock","EU_28","USA") = 635.758239746094 ;
vcif("c_Livestock","EU_28","EU_28") = 17415.181640625 ;
vcif("c_Livestock","EU_28","CHN") = 1262.37097167969 ;
vcif("c_Livestock","EU_28","JPN") = 187.655456542969 ;
vcif("c_Livestock","EU_28","IND") = 25.6142082214355 ;
vcif("c_Livestock","EU_28","SSA") = 153.042419433594 ;
vcif("c_Livestock","EU_28","ROW") = 5813.7763671875 ;
vcif("c_Livestock","CHN","USA") = 225.347885131836 ;
vcif("c_Livestock","CHN","EU_28") = 396.66748046875 ;
vcif("c_Livestock","CHN","CHN") = 3.99999998990097E-6 ;
vcif("c_Livestock","CHN","JPN") = 225.903060913086 ;
vcif("c_Livestock","CHN","IND") = 5.49816036224365 ;
vcif("c_Livestock","CHN","SSA") = 30.0818691253662 ;
vcif("c_Livestock","CHN","ROW") = 902.312377929688 ;
vcif("c_Livestock","JPN","USA") = 0.792849600315094 ;
vcif("c_Livestock","JPN","EU_28") = 4.52381801605225 ;
vcif("c_Livestock","JPN","CHN") = 16.8544654846191 ;
vcif("c_Livestock","JPN","JPN") = 3.99999998990097E-6 ;
vcif("c_Livestock","JPN","IND") = 0.127571761608124 ;
vcif("c_Livestock","JPN","SSA") = 0.217003762722015 ;
vcif("c_Livestock","JPN","ROW") = 151.492950439453 ;
vcif("c_Livestock","IND","USA") = 89.9794540405273 ;
vcif("c_Livestock","IND","EU_28") = 16.2062244415283 ;
vcif("c_Livestock","IND","CHN") = 3.25453019142151 ;
vcif("c_Livestock","IND","JPN") = 14.5610342025757 ;
vcif("c_Livestock","IND","IND") = 3.99999998990097E-6 ;
vcif("c_Livestock","IND","SSA") = 3.26177501678467 ;
vcif("c_Livestock","IND","ROW") = 162.808013916016 ;
vcif("c_Livestock","SSA","USA") = 40.7927360534668 ;
vcif("c_Livestock","SSA","EU_28") = 193.99137878418 ;
vcif("c_Livestock","SSA","CHN") = 313.561889648438 ;
vcif("c_Livestock","SSA","JPN") = 10.7833986282349 ;
vcif("c_Livestock","SSA","IND") = 19.1845893859863 ;
vcif("c_Livestock","SSA","SSA") = 688.960815429688 ;
vcif("c_Livestock","SSA","ROW") = 1327.00524902344 ;
vcif("c_Livestock","ROW","USA") = 3219.15063476562 ;
vcif("c_Livestock","ROW","EU_28") = 1749.86633300781 ;
vcif("c_Livestock","ROW","CHN") = 4543.43408203125 ;
vcif("c_Livestock","ROW","JPN") = 426.214324951172 ;
vcif("c_Livestock","ROW","IND") = 224.471389770508 ;
vcif("c_Livestock","ROW","SSA") = 85.3905334472656 ;
vcif("c_Livestock","ROW","ROW") = 8377.8759765625 ;
vcif("c_FoodProc","USA","USA") = 7.00000009601354E-6 ;
vcif("c_FoodProc","USA","EU_28") = 6365.0673828125 ;
vcif("c_FoodProc","USA","CHN") = 4328.77978515625 ;
vcif("c_FoodProc","USA","JPN") = 8714.8876953125 ;
vcif("c_FoodProc","USA","IND") = 523.387145996094 ;
vcif("c_FoodProc","USA","SSA") = 1103.50646972656 ;
vcif("c_FoodProc","USA","ROW") = 60325.296875 ;
vcif("c_FoodProc","EU_28","USA") = 24777.640625 ;
vcif("c_FoodProc","EU_28","EU_28") = 299926.34375 ;
vcif("c_FoodProc","EU_28","CHN") = 13801.7138671875 ;
vcif("c_FoodProc","EU_28","JPN") = 9854.0146484375 ;
vcif("c_FoodProc","EU_28","IND") = 609.613037109375 ;
vcif("c_FoodProc","EU_28","SSA") = 9138.1162109375 ;
vcif("c_FoodProc","EU_28","ROW") = 74718.8515625 ;
vcif("c_FoodProc","CHN","USA") = 7440.00341796875 ;
vcif("c_FoodProc","CHN","EU_28") = 6586.01123046875 ;
vcif("c_FoodProc","CHN","CHN") = 7.00000009601354E-6 ;
vcif("c_FoodProc","CHN","JPN") = 7847.1845703125 ;
vcif("c_FoodProc","CHN","IND") = 238.004333496094 ;
vcif("c_FoodProc","CHN","SSA") = 2054.15209960938 ;
vcif("c_FoodProc","CHN","ROW") = 27869.15234375 ;
vcif("c_FoodProc","JPN","USA") = 913.429321289062 ;
vcif("c_FoodProc","JPN","EU_28") = 312.629364013672 ;
vcif("c_FoodProc","JPN","CHN") = 697.212951660156 ;
vcif("c_FoodProc","JPN","JPN") = 7.00000009601354E-6 ;
vcif("c_FoodProc","JPN","IND") = 6.31862592697144 ;
vcif("c_FoodProc","JPN","SSA") = 93.916862487793 ;
vcif("c_FoodProc","JPN","ROW") = 3413.88842773438 ;
vcif("c_FoodProc","IND","USA") = 3820.31713867188 ;
vcif("c_FoodProc","IND","EU_28") = 2776.68017578125 ;
vcif("c_FoodProc","IND","CHN") = 663.968444824219 ;
vcif("c_FoodProc","IND","JPN") = 728.353271484375 ;
vcif("c_FoodProc","IND","IND") = 7.00000009601354E-6 ;
vcif("c_FoodProc","IND","SSA") = 1134.853515625 ;
vcif("c_FoodProc","IND","ROW") = 9958.3271484375 ;
vcif("c_FoodProc","SSA","USA") = 723.504638671875 ;
vcif("c_FoodProc","SSA","EU_28") = 7195.169921875 ;
vcif("c_FoodProc","SSA","CHN") = 1232.71130371094 ;
vcif("c_FoodProc","SSA","JPN") = 474.110504150391 ;
vcif("c_FoodProc","SSA","IND") = 76.7207565307617 ;
vcif("c_FoodProc","SSA","SSA") = 8952.25 ;
vcif("c_FoodProc","SSA","ROW") = 3117.29223632812 ;
vcif("c_FoodProc","ROW","USA") = 69857.921875 ;
vcif("c_FoodProc","ROW","EU_28") = 61811.03515625 ;
vcif("c_FoodProc","ROW","CHN") = 36129.58203125 ;
vcif("c_FoodProc","ROW","JPN") = 27321.697265625 ;
vcif("c_FoodProc","ROW","IND") = 14444.08984375 ;
vcif("c_FoodProc","ROW","SSA") = 15923.7734375 ;
vcif("c_FoodProc","ROW","ROW") = 191629.015625 ;
vcif("c_Energy","USA","USA") = 6.00000021222513E-6 ;
vcif("c_Energy","USA","EU_28") = 19536.767578125 ;
vcif("c_Energy","USA","CHN") = 9697.9189453125 ;
vcif("c_Energy","USA","JPN") = 7227.4931640625 ;
vcif("c_Energy","USA","IND") = 3509.16455078125 ;
vcif("c_Energy","USA","SSA") = 2350.63110351562 ;
vcif("c_Energy","USA","ROW") = 113294.4609375 ;
vcif("c_Energy","EU_28","USA") = 10228.876953125 ;
vcif("c_Energy","EU_28","EU_28") = 96163.3515625 ;
vcif("c_Energy","EU_28","CHN") = 2238.27490234375 ;
vcif("c_Energy","EU_28","JPN") = 131.822769165039 ;
vcif("c_Energy","EU_28","IND") = 521.540771484375 ;
vcif("c_Energy","EU_28","SSA") = 10704.7431640625 ;
vcif("c_Energy","EU_28","ROW") = 43134.14453125 ;
vcif("c_Energy","CHN","USA") = 2754.91162109375 ;
vcif("c_Energy","CHN","EU_28") = 1383.09143066406 ;
vcif("c_Energy","CHN","CHN") = 6.00000021222513E-6 ;
vcif("c_Energy","CHN","JPN") = 1513.97583007812 ;
vcif("c_Energy","CHN","IND") = 1314.81848144531 ;
vcif("c_Energy","CHN","SSA") = 3881.236328125 ;
vcif("c_Energy","CHN","ROW") = 27023.658203125 ;
vcif("c_Energy","JPN","USA") = 841.657165527344 ;
vcif("c_Energy","JPN","EU_28") = 257.5595703125 ;
vcif("c_Energy","JPN","CHN") = 1184.08447265625 ;
vcif("c_Energy","JPN","JPN") = 6.00000021222513E-6 ;
vcif("c_Energy","JPN","IND") = 231.756652832031 ;
vcif("c_Energy","JPN","SSA") = 15.0863018035889 ;
vcif("c_Energy","JPN","ROW") = 6933.5087890625 ;
vcif("c_Energy","IND","USA") = 4595.57568359375 ;
vcif("c_Energy","IND","EU_28") = 2374.60400390625 ;
vcif("c_Energy","IND","CHN") = 642.8779296875 ;
vcif("c_Energy","IND","JPN") = 1582.19152832031 ;
vcif("c_Energy","IND","IND") = 6.00000021222513E-6 ;
vcif("c_Energy","IND","SSA") = 3130.306640625 ;
vcif("c_Energy","IND","ROW") = 17176.109375 ;
vcif("c_Energy","SSA","USA") = 11630.8486328125 ;
vcif("c_Energy","SSA","EU_28") = 21796.083984375 ;
vcif("c_Energy","SSA","CHN") = 30280.47265625 ;
vcif("c_Energy","SSA","JPN") = 1177.48461914062 ;
vcif("c_Energy","SSA","IND") = 16460.84375 ;
vcif("c_Energy","SSA","SSA") = 11387.5849609375 ;
vcif("c_Energy","SSA","ROW") = 17027.1484375 ;
vcif("c_Energy","ROW","USA") = 185510 ;
vcif("c_Energy","ROW","EU_28") = 353042.875 ;
vcif("c_Energy","ROW","CHN") = 188491.671875 ;
vcif("c_Energy","ROW","JPN") = 136337.859375 ;
vcif("c_Energy","ROW","IND") = 104964.8125 ;
vcif("c_Energy","ROW","SSA") = 18503.556640625 ;
vcif("c_Energy","ROW","ROW") = 418311.09375 ;
vcif("c_Textiles","USA","USA") = 3.00000010611257E-6 ;
vcif("c_Textiles","USA","EU_28") = 3104.06420898438 ;
vcif("c_Textiles","USA","CHN") = 1717.2890625 ;
vcif("c_Textiles","USA","JPN") = 499.526397705078 ;
vcif("c_Textiles","USA","IND") = 202.937164306641 ;
vcif("c_Textiles","USA","SSA") = 250.55224609375 ;
vcif("c_Textiles","USA","ROW") = 20717.306640625 ;
vcif("c_Textiles","EU_28","USA") = 11208.060546875 ;
vcif("c_Textiles","EU_28","EU_28") = 183440.40625 ;
vcif("c_Textiles","EU_28","CHN") = 9651.767578125 ;
vcif("c_Textiles","EU_28","JPN") = 4889.53076171875 ;
vcif("c_Textiles","EU_28","IND") = 753.488342285156 ;
vcif("c_Textiles","EU_28","SSA") = 2043.4248046875 ;
vcif("c_Textiles","EU_28","ROW") = 46940.4453125 ;
vcif("c_Textiles","CHN","USA") = 75835.4921875 ;
vcif("c_Textiles","CHN","EU_28") = 73016.578125 ;
vcif("c_Textiles","CHN","CHN") = 3.00000010611257E-6 ;
vcif("c_Textiles","CHN","JPN") = 32363.47265625 ;
vcif("c_Textiles","CHN","IND") = 6464.8046875 ;
vcif("c_Textiles","CHN","SSA") = 20428.59765625 ;
vcif("c_Textiles","CHN","ROW") = 176835.09375 ;
vcif("c_Textiles","JPN","USA") = 726.630432128906 ;
vcif("c_Textiles","JPN","EU_28") = 848.133911132812 ;
vcif("c_Textiles","JPN","CHN") = 3582.74829101562 ;
vcif("c_Textiles","JPN","JPN") = 3.00000010611257E-6 ;
vcif("c_Textiles","JPN","IND") = 121.770370483398 ;
vcif("c_Textiles","JPN","SSA") = 175.809112548828 ;
vcif("c_Textiles","JPN","ROW") = 3906.92724609375 ;
vcif("c_Textiles","IND","USA") = 10032.439453125 ;
vcif("c_Textiles","IND","EU_28") = 13589.578125 ;
vcif("c_Textiles","IND","CHN") = 2360.28076171875 ;
vcif("c_Textiles","IND","JPN") = 612.001098632812 ;
vcif("c_Textiles","IND","IND") = 3.00000010611257E-6 ;
vcif("c_Textiles","IND","SSA") = 2337.18139648438 ;
vcif("c_Textiles","IND","ROW") = 16677.12109375 ;
vcif("c_Textiles","SSA","USA") = 1315.02111816406 ;
vcif("c_Textiles","SSA","EU_28") = 1485.18688964844 ;
vcif("c_Textiles","SSA","CHN") = 320.828460693359 ;
vcif("c_Textiles","SSA","JPN") = 26.1140460968018 ;
vcif("c_Textiles","SSA","IND") = 62.304084777832 ;
vcif("c_Textiles","SSA","SSA") = 2363.09008789062 ;
vcif("c_Textiles","SSA","ROW") = 472.417114257812 ;
vcif("c_Textiles","ROW","USA") = 78062.5546875 ;
vcif("c_Textiles","ROW","EU_28") = 97929.09375 ;
vcif("c_Textiles","ROW","CHN") = 25575.123046875 ;
vcif("c_Textiles","ROW","JPN") = 15685.908203125 ;
vcif("c_Textiles","ROW","IND") = 2443.56665039062 ;
vcif("c_Textiles","ROW","SSA") = 2928.615234375 ;
vcif("c_Textiles","ROW","ROW") = 89132.890625 ;
vcif("c_Chem","USA","USA") = 3.00000010611257E-6 ;
vcif("c_Chem","USA","EU_28") = 63090.484375 ;
vcif("c_Chem","USA","CHN") = 20154.12890625 ;
vcif("c_Chem","USA","JPN") = 14882.60546875 ;
vcif("c_Chem","USA","IND") = 4384.87890625 ;
vcif("c_Chem","USA","SSA") = 2114.39624023438 ;
vcif("c_Chem","USA","ROW") = 149166.125 ;
vcif("c_Chem","EU_28","USA") = 100272.15625 ;
vcif("c_Chem","EU_28","EU_28") = 602671.3125 ;
vcif("c_Chem","EU_28","CHN") = 48770.4453125 ;
vcif("c_Chem","EU_28","JPN") = 23481.0390625 ;
vcif("c_Chem","EU_28","IND") = 7525.33837890625 ;
vcif("c_Chem","EU_28","SSA") = 14269.1982421875 ;
vcif("c_Chem","EU_28","ROW") = 225236.625 ;
vcif("c_Chem","CHN","USA") = 35006.34765625 ;
vcif("c_Chem","CHN","EU_28") = 31036.478515625 ;
vcif("c_Chem","CHN","CHN") = 3.00000010611257E-6 ;
vcif("c_Chem","CHN","JPN") = 15322.970703125 ;
vcif("c_Chem","CHN","IND") = 13841.671875 ;
vcif("c_Chem","CHN","SSA") = 10184.88671875 ;
vcif("c_Chem","CHN","ROW") = 111545.9609375 ;
vcif("c_Chem","JPN","USA") = 12597.42578125 ;
vcif("c_Chem","JPN","EU_28") = 9674.3515625 ;
vcif("c_Chem","JPN","CHN") = 32975.24609375 ;
vcif("c_Chem","JPN","JPN") = 3.00000010611257E-6 ;
vcif("c_Chem","JPN","IND") = 2112.72216796875 ;
vcif("c_Chem","JPN","SSA") = 514.575317382812 ;
vcif("c_Chem","JPN","ROW") = 47352.24609375 ;
vcif("c_Chem","IND","USA") = 10794.439453125 ;
vcif("c_Chem","IND","EU_28") = 8717.201171875 ;
vcif("c_Chem","IND","CHN") = 3079.62158203125 ;
vcif("c_Chem","IND","JPN") = 961.957702636719 ;
vcif("c_Chem","IND","IND") = 3.00000010611257E-6 ;
vcif("c_Chem","IND","SSA") = 5115.5478515625 ;
vcif("c_Chem","IND","ROW") = 21890.62890625 ;
vcif("c_Chem","SSA","USA") = 1125.32507324219 ;
vcif("c_Chem","SSA","EU_28") = 2784.99609375 ;
vcif("c_Chem","SSA","CHN") = 646.1533203125 ;
vcif("c_Chem","SSA","JPN") = 90.414680480957 ;
vcif("c_Chem","SSA","IND") = 819.862854003906 ;
vcif("c_Chem","SSA","SSA") = 7281.55615234375 ;
vcif("c_Chem","SSA","ROW") = 2327.68823242188 ;
vcif("c_Chem","ROW","USA") = 111499.1015625 ;
vcif("c_Chem","ROW","EU_28") = 128613.59375 ;
vcif("c_Chem","ROW","CHN") = 137783.203125 ;
vcif("c_Chem","ROW","JPN") = 26847.73046875 ;
vcif("c_Chem","ROW","IND") = 27080.939453125 ;
vcif("c_Chem","ROW","SSA") = 12086.552734375 ;
vcif("c_Chem","ROW","ROW") = 255539 ;
vcif("c_Manuf","USA","USA") = 1.29999998534913E-5 ;
vcif("c_Manuf","USA","EU_28") = 158118.125 ;
vcif("c_Manuf","USA","CHN") = 78784.1875 ;
vcif("c_Manuf","USA","JPN") = 40498.6015625 ;
vcif("c_Manuf","USA","IND") = 17900.181640625 ;
vcif("c_Manuf","USA","SSA") = 8420.3798828125 ;
vcif("c_Manuf","USA","ROW") = 601537.5 ;
vcif("c_Manuf","EU_28","USA") = 268906.96875 ;
vcif("c_Manuf","EU_28","EU_28") = 1930929.125 ;
vcif("c_Manuf","EU_28","CHN") = 200496.03125 ;
vcif("c_Manuf","EU_28","JPN") = 42397.3671875 ;
vcif("c_Manuf","EU_28","IND") = 35684.640625 ;
vcif("c_Manuf","EU_28","SSA") = 43592.2421875 ;
vcif("c_Manuf","EU_28","ROW") = 739287.3125 ;
vcif("c_Manuf","CHN","USA") = 363258.0625 ;
vcif("c_Manuf","CHN","EU_28") = 250631.34375 ;
vcif("c_Manuf","CHN","CHN") = 1.29999998534913E-5 ;
vcif("c_Manuf","CHN","JPN") = 126547.9921875 ;
vcif("c_Manuf","CHN","IND") = 57500.171875 ;
vcif("c_Manuf","CHN","SSA") = 50948.453125 ;
vcif("c_Manuf","CHN","ROW") = 713920.5 ;
vcif("c_Manuf","JPN","USA") = 120329.7734375 ;
vcif("c_Manuf","JPN","EU_28") = 65845.28125 ;
vcif("c_Manuf","JPN","CHN") = 164494.265625 ;
vcif("c_Manuf","JPN","JPN") = 1.29999998534913E-5 ;
vcif("c_Manuf","JPN","IND") = 7550.47802734375 ;
vcif("c_Manuf","JPN","SSA") = 6399.8515625 ;
vcif("c_Manuf","JPN","ROW") = 254108.015625 ;
vcif("c_Manuf","IND","USA") = 21550.04296875 ;
vcif("c_Manuf","IND","EU_28") = 22050.705078125 ;
vcif("c_Manuf","IND","CHN") = 12644.25390625 ;
vcif("c_Manuf","IND","JPN") = 1877.19189453125 ;
vcif("c_Manuf","IND","IND") = 1.29999998534913E-5 ;
vcif("c_Manuf","IND","SSA") = 7087.6015625 ;
vcif("c_Manuf","IND","ROW") = 67343.2421875 ;
vcif("c_Manuf","SSA","USA") = 8114.73681640625 ;
vcif("c_Manuf","SSA","EU_28") = 29450.1328125 ;
vcif("c_Manuf","SSA","CHN") = 42051.86328125 ;
vcif("c_Manuf","SSA","JPN") = 5219.314453125 ;
vcif("c_Manuf","SSA","IND") = 12276.7275390625 ;
vcif("c_Manuf","SSA","SSA") = 22530.748046875 ;
vcif("c_Manuf","SSA","ROW") = 43040.265625 ;
vcif("c_Manuf","ROW","USA") = 698777.625 ;
vcif("c_Manuf","ROW","EU_28") = 452190.71875 ;
vcif("c_Manuf","ROW","CHN") = 700951.625 ;
vcif("c_Manuf","ROW","JPN") = 135909 ;
vcif("c_Manuf","ROW","IND") = 92377.078125 ;
vcif("c_Manuf","ROW","SSA") = 32695.126953125 ;
vcif("c_Manuf","ROW","ROW") = 1004028.5 ;
vcif("c_ForestFish","USA","USA") = 1.99999999495049E-6 ;
vcif("c_ForestFish","USA","EU_28") = 336.358825683594 ;
vcif("c_ForestFish","USA","CHN") = 823.573364257812 ;
vcif("c_ForestFish","USA","JPN") = 537.987915039062 ;
vcif("c_ForestFish","USA","IND") = 14.0464630126953 ;
vcif("c_ForestFish","USA","SSA") = 2.8211522102356 ;
vcif("c_ForestFish","USA","ROW") = 1060.83264160156 ;
vcif("c_ForestFish","EU_28","USA") = 473.895782470703 ;
vcif("c_ForestFish","EU_28","EU_28") = 11709.3251953125 ;
vcif("c_ForestFish","EU_28","CHN") = 708.820861816406 ;
vcif("c_ForestFish","EU_28","JPN") = 33.3735542297363 ;
vcif("c_ForestFish","EU_28","IND") = 38.4636383056641 ;
vcif("c_ForestFish","EU_28","SSA") = 22.2601718902588 ;
vcif("c_ForestFish","EU_28","ROW") = 1012.63330078125 ;
vcif("c_ForestFish","CHN","USA") = 34.1810874938965 ;
vcif("c_ForestFish","CHN","EU_28") = 85.4548645019531 ;
vcif("c_ForestFish","CHN","CHN") = 1.99999999495049E-6 ;
vcif("c_ForestFish","CHN","JPN") = 496.013488769531 ;
vcif("c_ForestFish","CHN","IND") = 1.55207288265228 ;
vcif("c_ForestFish","CHN","SSA") = 1.65807616710663 ;
vcif("c_ForestFish","CHN","ROW") = 895.484558105469 ;
vcif("c_ForestFish","JPN","USA") = 53.5491905212402 ;
vcif("c_ForestFish","JPN","EU_28") = 23.2249145507812 ;
vcif("c_ForestFish","JPN","CHN") = 144.885955810547 ;
vcif("c_ForestFish","JPN","JPN") = 1.99999999495049E-6 ;
vcif("c_ForestFish","JPN","IND") = 1.80237638950348 ;
vcif("c_ForestFish","JPN","SSA") = 0.741377055644989 ;
vcif("c_ForestFish","JPN","ROW") = 372.401428222656 ;
vcif("c_ForestFish","IND","USA") = 45.0034103393555 ;
vcif("c_ForestFish","IND","EU_28") = 72.9104461669922 ;
vcif("c_ForestFish","IND","CHN") = 86.4463043212891 ;
vcif("c_ForestFish","IND","JPN") = 5.74633598327637 ;
vcif("c_ForestFish","IND","IND") = 1.99999999495049E-6 ;
vcif("c_ForestFish","IND","SSA") = 4.77335119247437 ;
vcif("c_ForestFish","IND","ROW") = 186.68489074707 ;
vcif("c_ForestFish","SSA","USA") = 64.19580078125 ;
vcif("c_ForestFish","SSA","EU_28") = 427.068054199219 ;
vcif("c_ForestFish","SSA","CHN") = 2111.1201171875 ;
vcif("c_ForestFish","SSA","JPN") = 23.7535171508789 ;
vcif("c_ForestFish","SSA","IND") = 194.013946533203 ;
vcif("c_ForestFish","SSA","SSA") = 178.102355957031 ;
vcif("c_ForestFish","SSA","ROW") = 605.820739746094 ;
vcif("c_ForestFish","ROW","USA") = 2806.74536132812 ;
vcif("c_ForestFish","ROW","EU_28") = 6474.9345703125 ;
vcif("c_ForestFish","ROW","CHN") = 8445.0703125 ;
vcif("c_ForestFish","ROW","JPN") = 1777.66467285156 ;
vcif("c_ForestFish","ROW","IND") = 1133.88989257812 ;
vcif("c_ForestFish","ROW","SSA") = 95.9909515380859 ;
vcif("c_ForestFish","ROW","ROW") = 5912.951171875 ;
vcif("c_Svces","USA","USA") = 1.80000006366754E-5 ;
vcif("c_Svces","USA","EU_28") = 223204.1875 ;
vcif("c_Svces","USA","CHN") = 57874.48828125 ;
vcif("c_Svces","USA","JPN") = 42788.98046875 ;
vcif("c_Svces","USA","IND") = 19128.24609375 ;
vcif("c_Svces","USA","SSA") = 18268.017578125 ;
vcif("c_Svces","USA","ROW") = 342195.65625 ;
vcif("c_Svces","EU_28","USA") = 199294.921875 ;
vcif("c_Svces","EU_28","EU_28") = 1038910.9375 ;
vcif("c_Svces","EU_28","CHN") = 52260.73046875 ;
vcif("c_Svces","EU_28","JPN") = 32276.919921875 ;
vcif("c_Svces","EU_28","IND") = 17354.9375 ;
vcif("c_Svces","EU_28","SSA") = 29422.033203125 ;
vcif("c_Svces","EU_28","ROW") = 431834.03125 ;
vcif("c_Svces","CHN","USA") = 19002.94921875 ;
vcif("c_Svces","CHN","EU_28") = 35278.9296875 ;
vcif("c_Svces","CHN","CHN") = 1.80000006366754E-5 ;
vcif("c_Svces","CHN","JPN") = 7994.64453125 ;
vcif("c_Svces","CHN","IND") = 2810.67749023438 ;
vcif("c_Svces","CHN","SSA") = 4210.1689453125 ;
vcif("c_Svces","CHN","ROW") = 105634.671875 ;
vcif("c_Svces","JPN","USA") = 20040.076171875 ;
vcif("c_Svces","JPN","EU_28") = 18437.458984375 ;
vcif("c_Svces","JPN","CHN") = 15669.7412109375 ;
vcif("c_Svces","JPN","JPN") = 1.80000006366754E-5 ;
vcif("c_Svces","JPN","IND") = 984.914978027344 ;
vcif("c_Svces","JPN","SSA") = 1856.79931640625 ;
vcif("c_Svces","JPN","ROW") = 50100.16796875 ;
vcif("c_Svces","IND","USA") = 27956.552734375 ;
vcif("c_Svces","IND","EU_28") = 28240.134765625 ;
vcif("c_Svces","IND","CHN") = 2620.39575195312 ;
vcif("c_Svces","IND","JPN") = 2652.44750976562 ;
vcif("c_Svces","IND","IND") = 1.80000006366754E-5 ;
vcif("c_Svces","IND","SSA") = 4550.8828125 ;
vcif("c_Svces","IND","ROW") = 43900.6875 ;
vcif("c_Svces","SSA","USA") = 9115.646484375 ;
vcif("c_Svces","SSA","EU_28") = 23896.33984375 ;
vcif("c_Svces","SSA","CHN") = 5176.61669921875 ;
vcif("c_Svces","SSA","JPN") = 1891.78247070312 ;
vcif("c_Svces","SSA","IND") = 2612.72021484375 ;
vcif("c_Svces","SSA","SSA") = 3836.8310546875 ;
vcif("c_Svces","SSA","ROW") = 19791.9765625 ;
vcif("c_Svces","ROW","USA") = 271711.40625 ;
vcif("c_Svces","ROW","EU_28") = 361803.84375 ;
vcif("c_Svces","ROW","CHN") = 179073.515625 ;
vcif("c_Svces","ROW","JPN") = 57829.4140625 ;
vcif("c_Svces","ROW","IND") = 33393.578125 ;
vcif("c_Svces","ROW","SSA") = 26256.544921875 ;
vcif("c_Svces","ROW","ROW") = 442174.28125 ;
parameter vmsb(*,*,*) ;
vmsb("c_Rice","USA","USA") = 1.99999999495049E-6 ;
vmsb("c_Rice","USA","EU_28") = 58.8202056884766 ;
vmsb("c_Rice","USA","CHN") = 1.34948265552521 ;
vmsb("c_Rice","USA","JPN") = 722.584899902344 ;
vmsb("c_Rice","USA","IND") = 3.14293265342712 ;
vmsb("c_Rice","USA","SSA") = 81.8949584960938 ;
vmsb("c_Rice","USA","ROW") = 1706.03662109375 ;
vmsb("c_Rice","EU_28","USA") = 29.1495723724365 ;
vmsb("c_Rice","EU_28","EU_28") = 1360.71203613281 ;
vmsb("c_Rice","EU_28","CHN") = 0.288474351167679 ;
vmsb("c_Rice","EU_28","JPN") = 0.900368928909302 ;
vmsb("c_Rice","EU_28","IND") = 1.69860935211182 ;
vmsb("c_Rice","EU_28","SSA") = 11.7120685577393 ;
vmsb("c_Rice","EU_28","ROW") = 297.564788818359 ;
vmsb("c_Rice","CHN","USA") = 4.0585618019104 ;
vmsb("c_Rice","CHN","EU_28") = 2.30212497711182 ;
vmsb("c_Rice","CHN","CHN") = 1.99999999495049E-6 ;
vmsb("c_Rice","CHN","JPN") = 10.9828605651855 ;
vmsb("c_Rice","CHN","IND") = 1.08324825763702 ;
vmsb("c_Rice","CHN","SSA") = 409.441650390625 ;
vmsb("c_Rice","CHN","ROW") = 365.822479248047 ;
vmsb("c_Rice","JPN","USA") = 3.34590172767639 ;
vmsb("c_Rice","JPN","EU_28") = 3.9218282699585 ;
vmsb("c_Rice","JPN","CHN") = 2.41492319107056 ;
vmsb("c_Rice","JPN","JPN") = 1.99999999495049E-6 ;
vmsb("c_Rice","JPN","IND") = 0.0576982758939266 ;
vmsb("c_Rice","JPN","SSA") = 10.7484998703003 ;
vmsb("c_Rice","JPN","ROW") = 35.2572593688965 ;
vmsb("c_Rice","IND","USA") = 204.392791748047 ;
vmsb("c_Rice","IND","EU_28") = 494.564575195312 ;
vmsb("c_Rice","IND","CHN") = 0.507588148117065 ;
vmsb("c_Rice","IND","JPN") = 2.04407024383545 ;
vmsb("c_Rice","IND","IND") = 1.99999999495049E-6 ;
vmsb("c_Rice","IND","SSA") = 2199.59399414062 ;
vmsb("c_Rice","IND","ROW") = 5891.8671875 ;
vmsb("c_Rice","SSA","USA") = 1.46944999694824 ;
vmsb("c_Rice","SSA","EU_28") = 4.22295618057251 ;
vmsb("c_Rice","SSA","CHN") = 2.08795428276062 ;
vmsb("c_Rice","SSA","JPN") = 0.229691684246063 ;
vmsb("c_Rice","SSA","IND") = 0.917960941791534 ;
vmsb("c_Rice","SSA","SSA") = 175.005737304688 ;
vmsb("c_Rice","SSA","ROW") = 3.7536346912384 ;
vmsb("c_Rice","ROW","USA") = 540.492065429688 ;
vmsb("c_Rice","ROW","EU_28") = 943.709655761719 ;
vmsb("c_Rice","ROW","CHN") = 2070.134765625 ;
vmsb("c_Rice","ROW","JPN") = 483.418060302734 ;
vmsb("c_Rice","ROW","IND") = 8.29892539978027 ;
vmsb("c_Rice","ROW","SSA") = 5082.77392578125 ;
vmsb("c_Rice","ROW","ROW") = 6314.2353515625 ;
vmsb("c_Crops","USA","USA") = 7.00000009601354E-6 ;
vmsb("c_Crops","USA","EU_28") = 6441.48828125 ;
vmsb("c_Crops","USA","CHN") = 18126.134765625 ;
vmsb("c_Crops","USA","JPN") = 7903.3759765625 ;
vmsb("c_Crops","USA","IND") = 1673.001953125 ;
vmsb("c_Crops","USA","SSA") = 1245.55163574219 ;
vmsb("c_Crops","USA","ROW") = 42260.39453125 ;
vmsb("c_Crops","EU_28","USA") = 1950.35791015625 ;
vmsb("c_Crops","EU_28","EU_28") = 80757.09375 ;
vmsb("c_Crops","EU_28","CHN") = 558.928161621094 ;
vmsb("c_Crops","EU_28","JPN") = 294.018920898438 ;
vmsb("c_Crops","EU_28","IND") = 456.523529052734 ;
vmsb("c_Crops","EU_28","SSA") = 2293.30541992188 ;
vmsb("c_Crops","EU_28","ROW") = 17896.099609375 ;
vmsb("c_Crops","CHN","USA") = 732.198669433594 ;
vmsb("c_Crops","CHN","EU_28") = 1395.85107421875 ;
vmsb("c_Crops","CHN","CHN") = 7.00000009601354E-6 ;
vmsb("c_Crops","CHN","JPN") = 1518.31433105469 ;
vmsb("c_Crops","CHN","IND") = 388.277496337891 ;
vmsb("c_Crops","CHN","SSA") = 202.787673950195 ;
vmsb("c_Crops","CHN","ROW") = 14542.7822265625 ;
vmsb("c_Crops","JPN","USA") = 54.0569648742676 ;
vmsb("c_Crops","JPN","EU_28") = 52.4080047607422 ;
vmsb("c_Crops","JPN","CHN") = 154.897552490234 ;
vmsb("c_Crops","JPN","JPN") = 7.00000009601354E-6 ;
vmsb("c_Crops","JPN","IND") = 1.44966626167297 ;
vmsb("c_Crops","JPN","SSA") = 0.407183289527893 ;
vmsb("c_Crops","JPN","ROW") = 386.646850585938 ;
vmsb("c_Crops","IND","USA") = 878.400634765625 ;
vmsb("c_Crops","IND","EU_28") = 1566.21179199219 ;
vmsb("c_Crops","IND","CHN") = 535.222778320312 ;
vmsb("c_Crops","IND","JPN") = 173.028854370117 ;
vmsb("c_Crops","IND","IND") = 7.00000009601354E-6 ;
vmsb("c_Crops","IND","SSA") = 174.669464111328 ;
vmsb("c_Crops","IND","ROW") = 7564.96484375 ;
vmsb("c_Crops","SSA","USA") = 2250.86352539062 ;
vmsb("c_Crops","SSA","EU_28") = 10436.2431640625 ;
vmsb("c_Crops","SSA","CHN") = 1707.04235839844 ;
vmsb("c_Crops","SSA","JPN") = 731.063232421875 ;
vmsb("c_Crops","SSA","IND") = 2625.02783203125 ;
vmsb("c_Crops","SSA","SSA") = 2833.76708984375 ;
vmsb("c_Crops","SSA","ROW") = 12090.2119140625 ;
vmsb("c_Crops","ROW","USA") = 37619.11328125 ;
vmsb("c_Crops","ROW","EU_28") = 39760.8828125 ;
vmsb("c_Crops","ROW","CHN") = 45179.49609375 ;
vmsb("c_Crops","ROW","JPN") = 9231.91796875 ;
vmsb("c_Crops","ROW","IND") = 7161.3623046875 ;
vmsb("c_Crops","ROW","SSA") = 4520.763671875 ;
vmsb("c_Crops","ROW","ROW") = 98460.2890625 ;
vmsb("c_Livestock","USA","USA") = 3.99999998990097E-6 ;
vmsb("c_Livestock","USA","EU_28") = 517.574768066406 ;
vmsb("c_Livestock","USA","CHN") = 1399.22863769531 ;
vmsb("c_Livestock","USA","JPN") = 187.867172241211 ;
vmsb("c_Livestock","USA","IND") = 10.5914516448975 ;
vmsb("c_Livestock","USA","SSA") = 26.2091388702393 ;
vmsb("c_Livestock","USA","ROW") = 2707.37841796875 ;
vmsb("c_Livestock","EU_28","USA") = 637.153869628906 ;
vmsb("c_Livestock","EU_28","EU_28") = 17415.181640625 ;
vmsb("c_Livestock","EU_28","CHN") = 1429.59460449219 ;
vmsb("c_Livestock","EU_28","JPN") = 250.215774536133 ;
vmsb("c_Livestock","EU_28","IND") = 29.2093067169189 ;
vmsb("c_Livestock","EU_28","SSA") = 167.65007019043 ;
vmsb("c_Livestock","EU_28","ROW") = 6544.82958984375 ;
vmsb("c_Livestock","CHN","USA") = 226.447280883789 ;
vmsb("c_Livestock","CHN","EU_28") = 426.242370605469 ;
vmsb("c_Livestock","CHN","CHN") = 3.99999998990097E-6 ;
vmsb("c_Livestock","CHN","JPN") = 247.353469848633 ;
vmsb("c_Livestock","CHN","IND") = 6.31957912445068 ;
vmsb("c_Livestock","CHN","SSA") = 33.5900192260742 ;
vmsb("c_Livestock","CHN","ROW") = 915.090637207031 ;
vmsb("c_Livestock","JPN","USA") = 0.795463025569916 ;
vmsb("c_Livestock","JPN","EU_28") = 4.52866744995117 ;
vmsb("c_Livestock","JPN","CHN") = 18.2936267852783 ;
vmsb("c_Livestock","JPN","JPN") = 3.99999998990097E-6 ;
vmsb("c_Livestock","JPN","IND") = 0.129069566726685 ;
vmsb("c_Livestock","JPN","SSA") = 0.217083796858788 ;
vmsb("c_Livestock","JPN","ROW") = 152.150512695312 ;
vmsb("c_Livestock","IND","USA") = 90.4237899780273 ;
vmsb("c_Livestock","IND","EU_28") = 16.3238506317139 ;
vmsb("c_Livestock","IND","CHN") = 3.34507989883423 ;
vmsb("c_Livestock","IND","JPN") = 14.6716747283936 ;
vmsb("c_Livestock","IND","IND") = 3.99999998990097E-6 ;
vmsb("c_Livestock","IND","SSA") = 3.46704912185669 ;
vmsb("c_Livestock","IND","ROW") = 167.025955200195 ;
vmsb("c_Livestock","SSA","USA") = 40.7976531982422 ;
vmsb("c_Livestock","SSA","EU_28") = 193.992492675781 ;
vmsb("c_Livestock","SSA","CHN") = 405.534820556641 ;
vmsb("c_Livestock","SSA","JPN") = 10.7914638519287 ;
vmsb("c_Livestock","SSA","IND") = 19.7347717285156 ;
vmsb("c_Livestock","SSA","SSA") = 691.84326171875 ;
vmsb("c_Livestock","SSA","ROW") = 1330.8583984375 ;
vmsb("c_Livestock","ROW","USA") = 3223.35864257812 ;
vmsb("c_Livestock","ROW","EU_28") = 1801.77624511719 ;
vmsb("c_Livestock","ROW","CHN") = 5504.68359375 ;
vmsb("c_Livestock","ROW","JPN") = 480.329223632812 ;
vmsb("c_Livestock","ROW","IND") = 234.755294799805 ;
vmsb("c_Livestock","ROW","SSA") = 91.2980041503906 ;
vmsb("c_Livestock","ROW","ROW") = 9333.068359375 ;
vmsb("c_FoodProc","USA","USA") = 7.00000009601354E-6 ;
vmsb("c_FoodProc","USA","EU_28") = 6902.6376953125 ;
vmsb("c_FoodProc","USA","CHN") = 4821.2568359375 ;
vmsb("c_FoodProc","USA","JPN") = 10390.5185546875 ;
vmsb("c_FoodProc","USA","IND") = 1003.44866943359 ;
vmsb("c_FoodProc","USA","SSA") = 1283.802734375 ;
vmsb("c_FoodProc","USA","ROW") = 63870.4296875 ;
vmsb("c_FoodProc","EU_28","USA") = 25468.66015625 ;
vmsb("c_FoodProc","EU_28","EU_28") = 299926.34375 ;
vmsb("c_FoodProc","EU_28","CHN") = 15558.80078125 ;
vmsb("c_FoodProc","EU_28","JPN") = 11536.8984375 ;
vmsb("c_FoodProc","EU_28","IND") = 1017.97381591797 ;
vmsb("c_FoodProc","EU_28","SSA") = 10306.5751953125 ;
vmsb("c_FoodProc","EU_28","ROW") = 84800.890625 ;
vmsb("c_FoodProc","CHN","USA") = 7658.3466796875 ;
vmsb("c_FoodProc","CHN","EU_28") = 7188.1279296875 ;
vmsb("c_FoodProc","CHN","CHN") = 7.00000009601354E-6 ;
vmsb("c_FoodProc","CHN","JPN") = 8512.3720703125 ;
vmsb("c_FoodProc","CHN","IND") = 316.901397705078 ;
vmsb("c_FoodProc","CHN","SSA") = 2371.89135742188 ;
vmsb("c_FoodProc","CHN","ROW") = 30016.794921875 ;
vmsb("c_FoodProc","JPN","USA") = 938.732421875 ;
vmsb("c_FoodProc","JPN","EU_28") = 339.019958496094 ;
vmsb("c_FoodProc","JPN","CHN") = 816.977783203125 ;
vmsb("c_FoodProc","JPN","JPN") = 7.00000009601354E-6 ;
vmsb("c_FoodProc","JPN","IND") = 8.83877563476562 ;
vmsb("c_FoodProc","JPN","SSA") = 102.267036437988 ;
vmsb("c_FoodProc","JPN","ROW") = 3652.76123046875 ;
vmsb("c_FoodProc","IND","USA") = 3870.892578125 ;
vmsb("c_FoodProc","IND","EU_28") = 2890.20532226562 ;
vmsb("c_FoodProc","IND","CHN") = 721.48779296875 ;
vmsb("c_FoodProc","IND","JPN") = 738.187866210938 ;
vmsb("c_FoodProc","IND","IND") = 7.00000009601354E-6 ;
vmsb("c_FoodProc","IND","SSA") = 1335.55053710938 ;
vmsb("c_FoodProc","IND","ROW") = 11076.123046875 ;
vmsb("c_FoodProc","SSA","USA") = 747.601196289062 ;
vmsb("c_FoodProc","SSA","EU_28") = 7258.17333984375 ;
vmsb("c_FoodProc","SSA","CHN") = 1355.62426757812 ;
vmsb("c_FoodProc","SSA","JPN") = 494.446044921875 ;
vmsb("c_FoodProc","SSA","IND") = 93.9157333374023 ;
vmsb("c_FoodProc","SSA","SSA") = 9328.091796875 ;
vmsb("c_FoodProc","SSA","ROW") = 3687.66259765625 ;
vmsb("c_FoodProc","ROW","USA") = 71219.1953125 ;
vmsb("c_FoodProc","ROW","EU_28") = 65675.359375 ;
vmsb("c_FoodProc","ROW","CHN") = 38952.0546875 ;
vmsb("c_FoodProc","ROW","JPN") = 30200.529296875 ;
vmsb("c_FoodProc","ROW","IND") = 17079.326171875 ;
vmsb("c_FoodProc","ROW","SSA") = 18347.6484375 ;
vmsb("c_FoodProc","ROW","ROW") = 206652.8125 ;
vmsb("c_Energy","USA","USA") = 6.00000021222513E-6 ;
vmsb("c_Energy","USA","EU_28") = 19696.4140625 ;
vmsb("c_Energy","USA","CHN") = 10015.685546875 ;
vmsb("c_Energy","USA","JPN") = 7233.9091796875 ;
vmsb("c_Energy","USA","IND") = 3611.27221679688 ;
vmsb("c_Energy","USA","SSA") = 2470.02563476562 ;
vmsb("c_Energy","USA","ROW") = 114597.6171875 ;
vmsb("c_Energy","EU_28","USA") = 10408.638671875 ;
vmsb("c_Energy","EU_28","EU_28") = 96163.3515625 ;
vmsb("c_Energy","EU_28","CHN") = 2284.99072265625 ;
vmsb("c_Energy","EU_28","JPN") = 133.27522277832 ;
vmsb("c_Energy","EU_28","IND") = 551.195739746094 ;
vmsb("c_Energy","EU_28","SSA") = 11514.2587890625 ;
vmsb("c_Energy","EU_28","ROW") = 43576.61328125 ;
vmsb("c_Energy","CHN","USA") = 2793.5498046875 ;
vmsb("c_Energy","CHN","EU_28") = 1400.75073242188 ;
vmsb("c_Energy","CHN","CHN") = 6.00000021222513E-6 ;
vmsb("c_Energy","CHN","JPN") = 1518.2626953125 ;
vmsb("c_Energy","CHN","IND") = 1374.96716308594 ;
vmsb("c_Energy","CHN","SSA") = 4194.1416015625 ;
vmsb("c_Energy","CHN","ROW") = 27408.162109375 ;
vmsb("c_Energy","JPN","USA") = 856.658996582031 ;
vmsb("c_Energy","JPN","EU_28") = 258.291381835938 ;
vmsb("c_Energy","JPN","CHN") = 1259.6806640625 ;
vmsb("c_Energy","JPN","JPN") = 6.00000021222513E-6 ;
vmsb("c_Energy","JPN","IND") = 239.444778442383 ;
vmsb("c_Energy","JPN","SSA") = 15.3894844055176 ;
vmsb("c_Energy","JPN","ROW") = 7024.7646484375 ;
vmsb("c_Energy","IND","USA") = 4664.24658203125 ;
vmsb("c_Energy","IND","EU_28") = 2416.96533203125 ;
vmsb("c_Energy","IND","CHN") = 686.376953125 ;
vmsb("c_Energy","IND","JPN") = 1588.4765625 ;
vmsb("c_Energy","IND","IND") = 6.00000021222513E-6 ;
vmsb("c_Energy","IND","SSA") = 3292.51049804688 ;
vmsb("c_Energy","IND","ROW") = 17685.623046875 ;
vmsb("c_Energy","SSA","USA") = 11631.181640625 ;
vmsb("c_Energy","SSA","EU_28") = 21797.169921875 ;
vmsb("c_Energy","SSA","CHN") = 30284.84765625 ;
vmsb("c_Energy","SSA","JPN") = 1177.52990722656 ;
vmsb("c_Energy","SSA","IND") = 16574.115234375 ;
vmsb("c_Energy","SSA","SSA") = 11469.314453125 ;
vmsb("c_Energy","SSA","ROW") = 17339.177734375 ;
vmsb("c_Energy","ROW","USA") = 185964.484375 ;
vmsb("c_Energy","ROW","EU_28") = 353871.28125 ;
vmsb("c_Energy","ROW","CHN") = 190057.96875 ;
vmsb("c_Energy","ROW","JPN") = 136446.109375 ;
vmsb("c_Energy","ROW","IND") = 105837.1015625 ;
vmsb("c_Energy","ROW","SSA") = 19256.83984375 ;
vmsb("c_Energy","ROW","ROW") = 423027.15625 ;
vmsb("c_Textiles","USA","USA") = 3.00000010611257E-6 ;
vmsb("c_Textiles","USA","EU_28") = 3308.36596679688 ;
vmsb("c_Textiles","USA","CHN") = 1874.55651855469 ;
vmsb("c_Textiles","USA","JPN") = 535.619079589844 ;
vmsb("c_Textiles","USA","IND") = 223.375396728516 ;
vmsb("c_Textiles","USA","SSA") = 300.073608398438 ;
vmsb("c_Textiles","USA","ROW") = 21084.400390625 ;
vmsb("c_Textiles","EU_28","USA") = 12082.6123046875 ;
vmsb("c_Textiles","EU_28","EU_28") = 183440.40625 ;
vmsb("c_Textiles","EU_28","CHN") = 10708.1640625 ;
vmsb("c_Textiles","EU_28","JPN") = 5437.9013671875 ;
vmsb("c_Textiles","EU_28","IND") = 833.484924316406 ;
vmsb("c_Textiles","EU_28","SSA") = 2421.03491210938 ;
vmsb("c_Textiles","EU_28","ROW") = 48257.828125 ;
vmsb("c_Textiles","CHN","USA") = 84281.953125 ;
vmsb("c_Textiles","CHN","EU_28") = 80461.59375 ;
vmsb("c_Textiles","CHN","CHN") = 3.00000010611257E-6 ;
vmsb("c_Textiles","CHN","JPN") = 34921.37890625 ;
vmsb("c_Textiles","CHN","IND") = 7204.47265625 ;
vmsb("c_Textiles","CHN","SSA") = 24851.638671875 ;
vmsb("c_Textiles","CHN","ROW") = 190804.046875 ;
vmsb("c_Textiles","JPN","USA") = 767.830200195312 ;
vmsb("c_Textiles","JPN","EU_28") = 900.716735839844 ;
vmsb("c_Textiles","JPN","CHN") = 3907.30029296875 ;
vmsb("c_Textiles","JPN","JPN") = 3.00000010611257E-6 ;
vmsb("c_Textiles","JPN","IND") = 122.700744628906 ;
vmsb("c_Textiles","JPN","SSA") = 190.406265258789 ;
vmsb("c_Textiles","JPN","ROW") = 4067.69018554688 ;
vmsb("c_Textiles","IND","USA") = 10915.5771484375 ;
vmsb("c_Textiles","IND","EU_28") = 14517.6513671875 ;
vmsb("c_Textiles","IND","CHN") = 2498.40283203125 ;
vmsb("c_Textiles","IND","JPN") = 622.547241210938 ;
vmsb("c_Textiles","IND","IND") = 3.00000010611257E-6 ;
vmsb("c_Textiles","IND","SSA") = 2874.36401367188 ;
vmsb("c_Textiles","IND","ROW") = 17821.908203125 ;
vmsb("c_Textiles","SSA","USA") = 1318.01574707031 ;
vmsb("c_Textiles","SSA","EU_28") = 1487.04321289062 ;
vmsb("c_Textiles","SSA","CHN") = 333.527069091797 ;
vmsb("c_Textiles","SSA","JPN") = 26.6960201263428 ;
vmsb("c_Textiles","SSA","IND") = 66.4423370361328 ;
vmsb("c_Textiles","SSA","SSA") = 2415.34497070312 ;
vmsb("c_Textiles","SSA","ROW") = 496.590301513672 ;
vmsb("c_Textiles","ROW","USA") = 84467.9296875 ;
vmsb("c_Textiles","ROW","EU_28") = 99702.3984375 ;
vmsb("c_Textiles","ROW","CHN") = 26505.640625 ;
vmsb("c_Textiles","ROW","JPN") = 15893.419921875 ;
vmsb("c_Textiles","ROW","IND") = 2573.818359375 ;
vmsb("c_Textiles","ROW","SSA") = 3540.30297851562 ;
vmsb("c_Textiles","ROW","ROW") = 93695.578125 ;
vmsb("c_Chem","USA","USA") = 3.00000010611257E-6 ;
vmsb("c_Chem","USA","EU_28") = 64580.328125 ;
vmsb("c_Chem","USA","CHN") = 21506.515625 ;
vmsb("c_Chem","USA","JPN") = 15075.279296875 ;
vmsb("c_Chem","USA","IND") = 4749.62353515625 ;
vmsb("c_Chem","USA","SSA") = 2237.83715820312 ;
vmsb("c_Chem","USA","ROW") = 151245.390625 ;
vmsb("c_Chem","EU_28","USA") = 101871.9609375 ;
vmsb("c_Chem","EU_28","EU_28") = 602671.3125 ;
vmsb("c_Chem","EU_28","CHN") = 51668.4921875 ;
vmsb("c_Chem","EU_28","JPN") = 23691.18359375 ;
vmsb("c_Chem","EU_28","IND") = 8157.5380859375 ;
vmsb("c_Chem","EU_28","SSA") = 14835.333984375 ;
vmsb("c_Chem","EU_28","ROW") = 230035.65625 ;
vmsb("c_Chem","CHN","USA") = 36183.0546875 ;
vmsb("c_Chem","CHN","EU_28") = 32495.587890625 ;
vmsb("c_Chem","CHN","CHN") = 3.00000010611257E-6 ;
vmsb("c_Chem","CHN","JPN") = 15344.494140625 ;
vmsb("c_Chem","CHN","IND") = 14907.056640625 ;
vmsb("c_Chem","CHN","SSA") = 11197.169921875 ;
vmsb("c_Chem","CHN","ROW") = 115319.015625 ;
vmsb("c_Chem","JPN","USA") = 12895.578125 ;
vmsb("c_Chem","JPN","EU_28") = 10039.40234375 ;
vmsb("c_Chem","JPN","CHN") = 35081.60546875 ;
vmsb("c_Chem","JPN","JPN") = 3.00000010611257E-6 ;
vmsb("c_Chem","JPN","IND") = 2179.45776367188 ;
vmsb("c_Chem","JPN","SSA") = 559.051574707031 ;
vmsb("c_Chem","JPN","ROW") = 48826.203125 ;
vmsb("c_Chem","IND","USA") = 10854.97265625 ;
vmsb("c_Chem","IND","EU_28") = 8933.42578125 ;
vmsb("c_Chem","IND","CHN") = 3232.7431640625 ;
vmsb("c_Chem","IND","JPN") = 962.457885742188 ;
vmsb("c_Chem","IND","IND") = 3.00000010611257E-6 ;
vmsb("c_Chem","IND","SSA") = 5320.28662109375 ;
vmsb("c_Chem","IND","ROW") = 22692.890625 ;
vmsb("c_Chem","SSA","USA") = 1130.20385742188 ;
vmsb("c_Chem","SSA","EU_28") = 2785.34545898438 ;
vmsb("c_Chem","SSA","CHN") = 691.0810546875 ;
vmsb("c_Chem","SSA","JPN") = 90.4269561767578 ;
vmsb("c_Chem","SSA","IND") = 845.473815917969 ;
vmsb("c_Chem","SSA","SSA") = 7444.6220703125 ;
vmsb("c_Chem","SSA","ROW") = 2389.55297851562 ;
vmsb("c_Chem","ROW","USA") = 112277.625 ;
vmsb("c_Chem","ROW","EU_28") = 130072.9609375 ;
vmsb("c_Chem","ROW","CHN") = 143248.578125 ;
vmsb("c_Chem","ROW","JPN") = 27052.57421875 ;
vmsb("c_Chem","ROW","IND") = 28504.341796875 ;
vmsb("c_Chem","ROW","SSA") = 12784.328125 ;
vmsb("c_Chem","ROW","ROW") = 260752.65625 ;
vmsb("c_Manuf","USA","USA") = 1.29999998534913E-5 ;
vmsb("c_Manuf","USA","EU_28") = 160861.15625 ;
vmsb("c_Manuf","USA","CHN") = 84301.3203125 ;
vmsb("c_Manuf","USA","JPN") = 40587.3828125 ;
vmsb("c_Manuf","USA","IND") = 19349.904296875 ;
vmsb("c_Manuf","USA","SSA") = 9111.62109375 ;
vmsb("c_Manuf","USA","ROW") = 609816.6875 ;
vmsb("c_Manuf","EU_28","USA") = 271934.375 ;
vmsb("c_Manuf","EU_28","EU_28") = 1930929.125 ;
vmsb("c_Manuf","EU_28","CHN") = 216974.234375 ;
vmsb("c_Manuf","EU_28","JPN") = 42531.67578125 ;
vmsb("c_Manuf","EU_28","IND") = 38623.1953125 ;
vmsb("c_Manuf","EU_28","SSA") = 46530.37109375 ;
vmsb("c_Manuf","EU_28","ROW") = 759553.5625 ;
vmsb("c_Manuf","CHN","USA") = 367159.75 ;
vmsb("c_Manuf","CHN","EU_28") = 255178.21875 ;
vmsb("c_Manuf","CHN","CHN") = 1.29999998534913E-5 ;
vmsb("c_Manuf","CHN","JPN") = 126682.4375 ;
vmsb("c_Manuf","CHN","IND") = 60547.65234375 ;
vmsb("c_Manuf","CHN","SSA") = 56947.82421875 ;
vmsb("c_Manuf","CHN","ROW") = 737275 ;
vmsb("c_Manuf","JPN","USA") = 122030.53125 ;
vmsb("c_Manuf","JPN","EU_28") = 68421.6328125 ;
vmsb("c_Manuf","JPN","CHN") = 174287.53125 ;
vmsb("c_Manuf","JPN","JPN") = 1.29999998534913E-5 ;
vmsb("c_Manuf","JPN","IND") = 7827.66748046875 ;
vmsb("c_Manuf","JPN","SSA") = 7257.07470703125 ;
vmsb("c_Manuf","JPN","ROW") = 264859.46875 ;
vmsb("c_Manuf","IND","USA") = 21642.6875 ;
vmsb("c_Manuf","IND","EU_28") = 22342.357421875 ;
vmsb("c_Manuf","IND","CHN") = 12910.162109375 ;
vmsb("c_Manuf","IND","JPN") = 1878.46887207031 ;
vmsb("c_Manuf","IND","IND") = 1.29999998534913E-5 ;
vmsb("c_Manuf","IND","SSA") = 7855.79296875 ;
vmsb("c_Manuf","IND","ROW") = 70890.2265625 ;
vmsb("c_Manuf","SSA","USA") = 8116.61279296875 ;
vmsb("c_Manuf","SSA","EU_28") = 29461.833984375 ;
vmsb("c_Manuf","SSA","CHN") = 42625.625 ;
vmsb("c_Manuf","SSA","JPN") = 5222.4619140625 ;
vmsb("c_Manuf","SSA","IND") = 13130.0458984375 ;
vmsb("c_Manuf","SSA","SSA") = 23079.466796875 ;
vmsb("c_Manuf","SSA","ROW") = 43740.3203125 ;
vmsb("c_Manuf","ROW","USA") = 700815.125 ;
vmsb("c_Manuf","ROW","EU_28") = 454386.21875 ;
vmsb("c_Manuf","ROW","CHN") = 710398.5 ;
vmsb("c_Manuf","ROW","JPN") = 136113.34375 ;
vmsb("c_Manuf","ROW","IND") = 97627.3984375 ;
vmsb("c_Manuf","ROW","SSA") = 35404.01171875 ;
vmsb("c_Manuf","ROW","ROW") = 1025179 ;
vmsb("c_ForestFish","USA","USA") = 1.99999999495049E-6 ;
vmsb("c_ForestFish","USA","EU_28") = 351.308135986328 ;
vmsb("c_ForestFish","USA","CHN") = 839.461853027344 ;
vmsb("c_ForestFish","USA","JPN") = 539.388000488281 ;
vmsb("c_ForestFish","USA","IND") = 16.0679817199707 ;
vmsb("c_ForestFish","USA","SSA") = 2.85517573356628 ;
vmsb("c_ForestFish","USA","ROW") = 1070.82775878906 ;
vmsb("c_ForestFish","EU_28","USA") = 474.613708496094 ;
vmsb("c_ForestFish","EU_28","EU_28") = 11709.3251953125 ;
vmsb("c_ForestFish","EU_28","CHN") = 726.455078125 ;
vmsb("c_ForestFish","EU_28","JPN") = 34.0814628601074 ;
vmsb("c_ForestFish","EU_28","IND") = 42.15673828125 ;
vmsb("c_ForestFish","EU_28","SSA") = 23.6146945953369 ;
vmsb("c_ForestFish","EU_28","ROW") = 1044.47692871094 ;
vmsb("c_ForestFish","CHN","USA") = 34.4844398498535 ;
vmsb("c_ForestFish","CHN","EU_28") = 85.8396682739258 ;
vmsb("c_ForestFish","CHN","CHN") = 1.99999999495049E-6 ;
vmsb("c_ForestFish","CHN","JPN") = 518.463317871094 ;
vmsb("c_ForestFish","CHN","IND") = 1.82116842269897 ;
vmsb("c_ForestFish","CHN","SSA") = 1.7414379119873 ;
vmsb("c_ForestFish","CHN","ROW") = 943.407897949219 ;
vmsb("c_ForestFish","JPN","USA") = 53.8376121520996 ;
vmsb("c_ForestFish","JPN","EU_28") = 23.6421413421631 ;
vmsb("c_ForestFish","JPN","CHN") = 147.419494628906 ;
vmsb("c_ForestFish","JPN","JPN") = 1.99999999495049E-6 ;
vmsb("c_ForestFish","JPN","IND") = 1.94038677215576 ;
vmsb("c_ForestFish","JPN","SSA") = 0.816299676895142 ;
vmsb("c_ForestFish","JPN","ROW") = 392.771759033203 ;
vmsb("c_ForestFish","IND","USA") = 45.025634765625 ;
vmsb("c_ForestFish","IND","EU_28") = 73.7230682373047 ;
vmsb("c_ForestFish","IND","CHN") = 89.0659942626953 ;
vmsb("c_ForestFish","IND","JPN") = 5.75156879425049 ;
vmsb("c_ForestFish","IND","IND") = 1.99999999495049E-6 ;
vmsb("c_ForestFish","IND","SSA") = 5.02468585968018 ;
vmsb("c_ForestFish","IND","ROW") = 196.180053710938 ;
vmsb("c_ForestFish","SSA","USA") = 64.2897262573242 ;
vmsb("c_ForestFish","SSA","EU_28") = 427.447692871094 ;
vmsb("c_ForestFish","SSA","CHN") = 2113.47827148438 ;
vmsb("c_ForestFish","SSA","JPN") = 23.9243335723877 ;
vmsb("c_ForestFish","SSA","IND") = 200.292663574219 ;
vmsb("c_ForestFish","SSA","SSA") = 183.223648071289 ;
vmsb("c_ForestFish","SSA","ROW") = 613.359924316406 ;
vmsb("c_ForestFish","ROW","USA") = 2807.92651367188 ;
vmsb("c_ForestFish","ROW","EU_28") = 6592.65625 ;
vmsb("c_ForestFish","ROW","CHN") = 8503.974609375 ;
vmsb("c_ForestFish","ROW","JPN") = 1823.20861816406 ;
vmsb("c_ForestFish","ROW","IND") = 1171.55261230469 ;
vmsb("c_ForestFish","ROW","SSA") = 98.9312515258789 ;
vmsb("c_ForestFish","ROW","ROW") = 6048.91748046875 ;
vmsb("c_Svces","USA","USA") = 1.80000006366754E-5 ;
vmsb("c_Svces","USA","EU_28") = 223204.1875 ;
vmsb("c_Svces","USA","CHN") = 57874.48828125 ;
vmsb("c_Svces","USA","JPN") = 42788.98046875 ;
vmsb("c_Svces","USA","IND") = 19128.24609375 ;
vmsb("c_Svces","USA","SSA") = 18268.017578125 ;
vmsb("c_Svces","USA","ROW") = 342195.65625 ;
vmsb("c_Svces","EU_28","USA") = 199294.921875 ;
vmsb("c_Svces","EU_28","EU_28") = 1038910.9375 ;
vmsb("c_Svces","EU_28","CHN") = 52260.73046875 ;
vmsb("c_Svces","EU_28","JPN") = 32276.919921875 ;
vmsb("c_Svces","EU_28","IND") = 17354.9375 ;
vmsb("c_Svces","EU_28","SSA") = 29422.033203125 ;
vmsb("c_Svces","EU_28","ROW") = 431834.03125 ;
vmsb("c_Svces","CHN","USA") = 19002.94921875 ;
vmsb("c_Svces","CHN","EU_28") = 35278.9296875 ;
vmsb("c_Svces","CHN","CHN") = 1.80000006366754E-5 ;
vmsb("c_Svces","CHN","JPN") = 7994.64453125 ;
vmsb("c_Svces","CHN","IND") = 2810.67749023438 ;
vmsb("c_Svces","CHN","SSA") = 4210.1689453125 ;
vmsb("c_Svces","CHN","ROW") = 105634.671875 ;
vmsb("c_Svces","JPN","USA") = 20040.076171875 ;
vmsb("c_Svces","JPN","EU_28") = 18437.458984375 ;
vmsb("c_Svces","JPN","CHN") = 15669.7412109375 ;
vmsb("c_Svces","JPN","JPN") = 1.80000006366754E-5 ;
vmsb("c_Svces","JPN","IND") = 984.914978027344 ;
vmsb("c_Svces","JPN","SSA") = 1856.79931640625 ;
vmsb("c_Svces","JPN","ROW") = 50100.16796875 ;
vmsb("c_Svces","IND","USA") = 27956.552734375 ;
vmsb("c_Svces","IND","EU_28") = 28240.134765625 ;
vmsb("c_Svces","IND","CHN") = 2620.39575195312 ;
vmsb("c_Svces","IND","JPN") = 2652.44750976562 ;
vmsb("c_Svces","IND","IND") = 1.80000006366754E-5 ;
vmsb("c_Svces","IND","SSA") = 4550.8828125 ;
vmsb("c_Svces","IND","ROW") = 43900.6875 ;
vmsb("c_Svces","SSA","USA") = 9115.646484375 ;
vmsb("c_Svces","SSA","EU_28") = 23896.33984375 ;
vmsb("c_Svces","SSA","CHN") = 5176.61669921875 ;
vmsb("c_Svces","SSA","JPN") = 1891.78247070312 ;
vmsb("c_Svces","SSA","IND") = 2612.72021484375 ;
vmsb("c_Svces","SSA","SSA") = 3836.8310546875 ;
vmsb("c_Svces","SSA","ROW") = 19791.9765625 ;
vmsb("c_Svces","ROW","USA") = 271711.40625 ;
vmsb("c_Svces","ROW","EU_28") = 361803.84375 ;
vmsb("c_Svces","ROW","CHN") = 179073.515625 ;
vmsb("c_Svces","ROW","JPN") = 57829.4140625 ;
vmsb("c_Svces","ROW","IND") = 33393.578125 ;
vmsb("c_Svces","ROW","SSA") = 26256.544921875 ;
vmsb("c_Svces","ROW","ROW") = 442174.28125 ;
parameter vst(*,*) ;
vst("c_Svces","USA") = 31137.61328125 ;
vst("c_Svces","EU_28") = 275190.5 ;
vst("c_Svces","CHN") = 24354.564453125 ;
vst("c_Svces","JPN") = 33450.9609375 ;
vst("c_Svces","IND") = 12699.376953125 ;
vst("c_Svces","SSA") = 8560.361328125 ;
vst("c_Svces","ROW") = 180795.890625 ;
parameter vtwr(*,*,*,*) ;
vtwr("c_Svces","c_Rice","USA","EU_28") = 4.09022092819214 ;
vtwr("c_Svces","c_Rice","USA","CHN") = 0.0721667855978012 ;
vtwr("c_Svces","c_Rice","USA","JPN") = 19.9046440124512 ;
vtwr("c_Svces","c_Rice","USA","IND") = 0.139725118875504 ;
vtwr("c_Svces","c_Rice","USA","SSA") = 6.24655342102051 ;
vtwr("c_Svces","c_Rice","USA","ROW") = 90.4333343505859 ;
vtwr("c_Svces","c_Rice","EU_28","USA") = 2.30442476272583 ;
vtwr("c_Svces","c_Rice","EU_28","EU_28") = 58.0012283325195 ;
vtwr("c_Svces","c_Rice","EU_28","CHN") = 0.000141999931656756 ;
vtwr("c_Svces","c_Rice","EU_28","JPN") = 0.0158340074121952 ;
vtwr("c_Svces","c_Rice","EU_28","IND") = 0.0801820307970047 ;
vtwr("c_Svces","c_Rice","EU_28","SSA") = 0.857570886611938 ;
vtwr("c_Svces","c_Rice","EU_28","ROW") = 18.6605377197266 ;
vtwr("c_Svces","c_Rice","CHN","USA") = 0.254404157400131 ;
vtwr("c_Svces","c_Rice","CHN","EU_28") = 0.0625930055975914 ;
vtwr("c_Svces","c_Rice","CHN","JPN") = 0.291683822870255 ;
vtwr("c_Svces","c_Rice","CHN","IND") = 0.00376699422486126 ;
vtwr("c_Svces","c_Rice","CHN","SSA") = 30.6044006347656 ;
vtwr("c_Svces","c_Rice","CHN","ROW") = 25.7140407562256 ;
vtwr("c_Svces","c_Rice","JPN","USA") = 0.26289165019989 ;
vtwr("c_Svces","c_Rice","JPN","EU_28") = 0.264888256788254 ;
vtwr("c_Svces","c_Rice","JPN","CHN") = 0.104346044361591 ;
vtwr("c_Svces","c_Rice","JPN","IND") = 0.000376000127289444 ;
vtwr("c_Svces","c_Rice","JPN","SSA") = 0.823431015014648 ;
vtwr("c_Svces","c_Rice","JPN","ROW") = 2.09758543968201 ;
vtwr("c_Svces","c_Rice","IND","USA") = 17.1945686340332 ;
vtwr("c_Svces","c_Rice","IND","EU_28") = 38.7315330505371 ;
vtwr("c_Svces","c_Rice","IND","CHN") = 0.000916999531909823 ;
vtwr("c_Svces","c_Rice","IND","JPN") = 0.0585259683430195 ;
vtwr("c_Svces","c_Rice","IND","SSA") = 172.299621582031 ;
vtwr("c_Svces","c_Rice","IND","ROW") = 456.718322753906 ;
vtwr("c_Svces","c_Rice","SSA","USA") = 0.0355004407465458 ;
vtwr("c_Svces","c_Rice","SSA","EU_28") = 0.0201019700616598 ;
vtwr("c_Svces","c_Rice","SSA","SSA") = 14.3634061813354 ;
vtwr("c_Svces","c_Rice","SSA","ROW") = 0.0267076566815376 ;
vtwr("c_Svces","c_Rice","ROW","USA") = 43.908016204834 ;
vtwr("c_Svces","c_Rice","ROW","EU_28") = 72.4572143554688 ;
vtwr("c_Svces","c_Rice","ROW","CHN") = 172.166305541992 ;
vtwr("c_Svces","c_Rice","ROW","JPN") = 13.6437301635742 ;
vtwr("c_Svces","c_Rice","ROW","IND") = 0.0725880265235901 ;
vtwr("c_Svces","c_Rice","ROW","SSA") = 371.281097412109 ;
vtwr("c_Svces","c_Rice","ROW","ROW") = 471.513427734375 ;
vtwr("c_Svces","c_Crops","USA","EU_28") = 309.402618408203 ;
vtwr("c_Svces","c_Crops","USA","CHN") = 1244.34838867188 ;
vtwr("c_Svces","c_Crops","USA","JPN") = 636.051574707031 ;
vtwr("c_Svces","c_Crops","USA","IND") = 58.6702613830566 ;
vtwr("c_Svces","c_Crops","USA","SSA") = 67.0347518920898 ;
vtwr("c_Svces","c_Crops","USA","ROW") = 2479.03002929688 ;
vtwr("c_Svces","c_Crops","EU_28","USA") = 94.4814071655273 ;
vtwr("c_Svces","c_Crops","EU_28","EU_28") = 6190.71044921875 ;
vtwr("c_Svces","c_Crops","EU_28","CHN") = 38.5787773132324 ;
vtwr("c_Svces","c_Crops","EU_28","JPN") = 15.5971488952637 ;
vtwr("c_Svces","c_Crops","EU_28","IND") = 22.5326843261719 ;
vtwr("c_Svces","c_Crops","EU_28","SSA") = 141.330078125 ;
vtwr("c_Svces","c_Crops","EU_28","ROW") = 1280.68127441406 ;
vtwr("c_Svces","c_Crops","CHN","USA") = 51.0122108459473 ;
vtwr("c_Svces","c_Crops","CHN","EU_28") = 84.7111892700195 ;
vtwr("c_Svces","c_Crops","CHN","JPN") = 149.237762451172 ;
vtwr("c_Svces","c_Crops","CHN","IND") = 29.5451183319092 ;
vtwr("c_Svces","c_Crops","CHN","SSA") = 13.7773418426514 ;
vtwr("c_Svces","c_Crops","CHN","ROW") = 1632.93322753906 ;
vtwr("c_Svces","c_Crops","JPN","USA") = 6.3026294708252 ;
vtwr("c_Svces","c_Crops","JPN","EU_28") = 3.37998342514038 ;
vtwr("c_Svces","c_Crops","JPN","CHN") = 7.68001699447632 ;
vtwr("c_Svces","c_Crops","JPN","IND") = 0.0382880009710789 ;
vtwr("c_Svces","c_Crops","JPN","SSA") = 0.00949397869408131 ;
vtwr("c_Svces","c_Crops","JPN","ROW") = 40.1056823730469 ;
vtwr("c_Svces","c_Crops","IND","USA") = 48.9604682922363 ;
vtwr("c_Svces","c_Crops","IND","EU_28") = 83.4766845703125 ;
vtwr("c_Svces","c_Crops","IND","CHN") = 54.6323852539062 ;
vtwr("c_Svces","c_Crops","IND","JPN") = 6.71771049499512 ;
vtwr("c_Svces","c_Crops","IND","SSA") = 9.7074031829834 ;
vtwr("c_Svces","c_Crops","IND","ROW") = 432.928924560547 ;
vtwr("c_Svces","c_Crops","SSA","USA") = 66.8510971069336 ;
vtwr("c_Svces","c_Crops","SSA","EU_28") = 836.747680664062 ;
vtwr("c_Svces","c_Crops","SSA","CHN") = 116.995834350586 ;
vtwr("c_Svces","c_Crops","SSA","JPN") = 42.538516998291 ;
vtwr("c_Svces","c_Crops","SSA","IND") = 80.5258941650391 ;
vtwr("c_Svces","c_Crops","SSA","SSA") = 225.088363647461 ;
vtwr("c_Svces","c_Crops","SSA","ROW") = 657.0478515625 ;
vtwr("c_Svces","c_Crops","ROW","USA") = 2597.0400390625 ;
vtwr("c_Svces","c_Crops","ROW","EU_28") = 3382.05419921875 ;
vtwr("c_Svces","c_Crops","ROW","CHN") = 3663.76684570312 ;
vtwr("c_Svces","c_Crops","ROW","JPN") = 783.446472167969 ;
vtwr("c_Svces","c_Crops","ROW","IND") = 325.920806884766 ;
vtwr("c_Svces","c_Crops","ROW","SSA") = 256.392852783203 ;
vtwr("c_Svces","c_Crops","ROW","ROW") = 7825.8115234375 ;
vtwr("c_Svces","c_Livestock","USA","EU_28") = 26.5585460662842 ;
vtwr("c_Svces","c_Livestock","USA","CHN") = 64.1173706054688 ;
vtwr("c_Svces","c_Livestock","USA","JPN") = 7.00140380859375 ;
vtwr("c_Svces","c_Livestock","USA","IND") = 0.406865775585175 ;
vtwr("c_Svces","c_Livestock","USA","SSA") = 7.19473457336426 ;
vtwr("c_Svces","c_Livestock","USA","ROW") = 159.755111694336 ;
vtwr("c_Svces","c_Livestock","EU_28","USA") = 26.6160182952881 ;
vtwr("c_Svces","c_Livestock","EU_28","EU_28") = 2767.37670898438 ;
vtwr("c_Svces","c_Livestock","EU_28","CHN") = 47.4299240112305 ;
vtwr("c_Svces","c_Livestock","EU_28","JPN") = 9.47660446166992 ;
vtwr("c_Svces","c_Livestock","EU_28","IND") = 2.84341621398926 ;
vtwr("c_Svces","c_Livestock","EU_28","SSA") = 30.9087028503418 ;
vtwr("c_Svces","c_Livestock","EU_28","ROW") = 1618.86303710938 ;
vtwr("c_Svces","c_Livestock","CHN","USA") = 10.8526582717896 ;
vtwr("c_Svces","c_Livestock","CHN","EU_28") = 13.9110269546509 ;
vtwr("c_Svces","c_Livestock","CHN","JPN") = 9.18381595611572 ;
vtwr("c_Svces","c_Livestock","CHN","IND") = 0.141753196716309 ;
vtwr("c_Svces","c_Livestock","CHN","SSA") = 1.03445374965668 ;
vtwr("c_Svces","c_Livestock","CHN","ROW") = 112.60977935791 ;
vtwr("c_Svces","c_Livestock","JPN","USA") = 0.0316240899264812 ;
vtwr("c_Svces","c_Livestock","JPN","EU_28") = 0.238485664129257 ;
vtwr("c_Svces","c_Livestock","JPN","CHN") = 1.09936535358429 ;
vtwr("c_Svces","c_Livestock","JPN","IND") = 0.0048070028424263 ;
vtwr("c_Svces","c_Livestock","JPN","SSA") = 0.00510611804202199 ;
vtwr("c_Svces","c_Livestock","JPN","ROW") = 8.24024391174316 ;
vtwr("c_Svces","c_Livestock","IND","USA") = 3.70481586456299 ;
vtwr("c_Svces","c_Livestock","IND","EU_28") = 0.807907402515411 ;
vtwr("c_Svces","c_Livestock","IND","CHN") = 0.0335405990481377 ;
vtwr("c_Svces","c_Livestock","IND","JPN") = 1.23744094371796 ;
vtwr("c_Svces","c_Livestock","IND","SSA") = 0.145581111311913 ;
vtwr("c_Svces","c_Livestock","IND","ROW") = 7.73651170730591 ;
vtwr("c_Svces","c_Livestock","SSA","USA") = 1.71625208854675 ;
vtwr("c_Svces","c_Livestock","SSA","EU_28") = 6.91468954086304 ;
vtwr("c_Svces","c_Livestock","SSA","CHN") = 10.7968444824219 ;
vtwr("c_Svces","c_Livestock","SSA","JPN") = 0.323811918497086 ;
vtwr("c_Svces","c_Livestock","SSA","IND") = 0.404635012149811 ;
vtwr("c_Svces","c_Livestock","SSA","SSA") = 271.519714355469 ;
vtwr("c_Svces","c_Livestock","SSA","ROW") = 132.301940917969 ;
vtwr("c_Svces","c_Livestock","ROW","USA") = 47.5019378662109 ;
vtwr("c_Svces","c_Livestock","ROW","EU_28") = 62.1413078308105 ;
vtwr("c_Svces","c_Livestock","ROW","CHN") = 352.374847412109 ;
vtwr("c_Svces","c_Livestock","ROW","JPN") = 58.7054634094238 ;
vtwr("c_Svces","c_Livestock","ROW","IND") = 8.54339218139648 ;
vtwr("c_Svces","c_Livestock","ROW","SSA") = 8.23899841308594 ;
vtwr("c_Svces","c_Livestock","ROW","ROW") = 2538.81958007812 ;
vtwr("c_Svces","c_FoodProc","USA","EU_28") = 324.969635009766 ;
vtwr("c_Svces","c_FoodProc","USA","CHN") = 204.589813232422 ;
vtwr("c_Svces","c_FoodProc","USA","JPN") = 416.837097167969 ;
vtwr("c_Svces","c_FoodProc","USA","IND") = 18.9920177459717 ;
vtwr("c_Svces","c_FoodProc","USA","SSA") = 50.6842994689941 ;
vtwr("c_Svces","c_FoodProc","USA","ROW") = 2213.5751953125 ;
vtwr("c_Svces","c_FoodProc","EU_28","USA") = 1379.01086425781 ;
vtwr("c_Svces","c_FoodProc","EU_28","EU_28") = 10139.796875 ;
vtwr("c_Svces","c_FoodProc","EU_28","CHN") = 670.463684082031 ;
vtwr("c_Svces","c_FoodProc","EU_28","JPN") = 457.564453125 ;
vtwr("c_Svces","c_FoodProc","EU_28","IND") = 28.0731220245361 ;
vtwr("c_Svces","c_FoodProc","EU_28","SSA") = 509.909820556641 ;
vtwr("c_Svces","c_FoodProc","EU_28","ROW") = 3656.14965820312 ;
vtwr("c_Svces","c_FoodProc","CHN","USA") = 432.73876953125 ;
vtwr("c_Svces","c_FoodProc","CHN","EU_28") = 368.512054443359 ;
vtwr("c_Svces","c_FoodProc","CHN","JPN") = 434.243774414062 ;
vtwr("c_Svces","c_FoodProc","CHN","IND") = 12.5582427978516 ;
vtwr("c_Svces","c_FoodProc","CHN","SSA") = 112.222839355469 ;
vtwr("c_Svces","c_FoodProc","CHN","ROW") = 1567.58557128906 ;
vtwr("c_Svces","c_FoodProc","JPN","USA") = 54.8482131958008 ;
vtwr("c_Svces","c_FoodProc","JPN","EU_28") = 16.6629600524902 ;
vtwr("c_Svces","c_FoodProc","JPN","CHN") = 33.0547561645508 ;
vtwr("c_Svces","c_FoodProc","JPN","IND") = 0.296585917472839 ;
vtwr("c_Svces","c_FoodProc","JPN","SSA") = 4.02971076965332 ;
vtwr("c_Svces","c_FoodProc","JPN","ROW") = 188.31591796875 ;
vtwr("c_Svces","c_FoodProc","IND","USA") = 139.122268676758 ;
vtwr("c_Svces","c_FoodProc","IND","EU_28") = 127.638084411621 ;
vtwr("c_Svces","c_FoodProc","IND","CHN") = 35.009147644043 ;
vtwr("c_Svces","c_FoodProc","IND","JPN") = 32.7404975891113 ;
vtwr("c_Svces","c_FoodProc","IND","SSA") = 59.0953063964844 ;
vtwr("c_Svces","c_FoodProc","IND","ROW") = 449.595581054688 ;
vtwr("c_Svces","c_FoodProc","SSA","USA") = 34.0087699890137 ;
vtwr("c_Svces","c_FoodProc","SSA","EU_28") = 303.999816894531 ;
vtwr("c_Svces","c_FoodProc","SSA","CHN") = 44.343936920166 ;
vtwr("c_Svces","c_FoodProc","SSA","JPN") = 20.7371425628662 ;
vtwr("c_Svces","c_FoodProc","SSA","IND") = 2.72132682800293 ;
vtwr("c_Svces","c_FoodProc","SSA","SSA") = 499.657135009766 ;
vtwr("c_Svces","c_FoodProc","SSA","ROW") = 141.427230834961 ;
vtwr("c_Svces","c_FoodProc","ROW","USA") = 2484.2294921875 ;
vtwr("c_Svces","c_FoodProc","ROW","EU_28") = 3067.38647460938 ;
vtwr("c_Svces","c_FoodProc","ROW","CHN") = 1851.20642089844 ;
vtwr("c_Svces","c_FoodProc","ROW","JPN") = 1201.95556640625 ;
vtwr("c_Svces","c_FoodProc","ROW","IND") = 760.154479980469 ;
vtwr("c_Svces","c_FoodProc","ROW","SSA") = 872.849243164062 ;
vtwr("c_Svces","c_FoodProc","ROW","ROW") = 10125.2626953125 ;
vtwr("c_Svces","c_Energy","USA","EU_28") = 1088.90905761719 ;
vtwr("c_Svces","c_Energy","USA","CHN") = 418.859436035156 ;
vtwr("c_Svces","c_Energy","USA","JPN") = 428.050018310547 ;
vtwr("c_Svces","c_Energy","USA","IND") = 259.097229003906 ;
vtwr("c_Svces","c_Energy","USA","SSA") = 91.9256973266602 ;
vtwr("c_Svces","c_Energy","USA","ROW") = 4860.17431640625 ;
vtwr("c_Svces","c_Energy","EU_28","USA") = 398.062286376953 ;
vtwr("c_Svces","c_Energy","EU_28","EU_28") = 2546.50463867188 ;
vtwr("c_Svces","c_Energy","EU_28","CHN") = 62.6789474487305 ;
vtwr("c_Svces","c_Energy","EU_28","JPN") = 4.38114023208618 ;
vtwr("c_Svces","c_Energy","EU_28","IND") = 29.9692897796631 ;
vtwr("c_Svces","c_Energy","EU_28","SSA") = 418.351165771484 ;
vtwr("c_Svces","c_Energy","EU_28","ROW") = 1408.83837890625 ;
vtwr("c_Svces","c_Energy","CHN","USA") = 111.721694946289 ;
vtwr("c_Svces","c_Energy","CHN","EU_28") = 58.4298782348633 ;
vtwr("c_Svces","c_Energy","CHN","JPN") = 98.3098068237305 ;
vtwr("c_Svces","c_Energy","CHN","IND") = 103.172027587891 ;
vtwr("c_Svces","c_Energy","CHN","SSA") = 172.957611083984 ;
vtwr("c_Svces","c_Energy","CHN","ROW") = 1121.55297851562 ;
vtwr("c_Svces","c_Energy","JPN","USA") = 33.6520919799805 ;
vtwr("c_Svces","c_Energy","JPN","EU_28") = 19.5738372802734 ;
vtwr("c_Svces","c_Energy","JPN","CHN") = 47.2946395874023 ;
vtwr("c_Svces","c_Energy","JPN","IND") = 18.6006107330322 ;
vtwr("c_Svces","c_Energy","JPN","SSA") = 0.216798350214958 ;
vtwr("c_Svces","c_Energy","JPN","ROW") = 275.122253417969 ;
vtwr("c_Svces","c_Energy","IND","USA") = 179.915863037109 ;
vtwr("c_Svces","c_Energy","IND","EU_28") = 91.1452484130859 ;
vtwr("c_Svces","c_Energy","IND","CHN") = 25.549373626709 ;
vtwr("c_Svces","c_Energy","IND","JPN") = 62.1066589355469 ;
vtwr("c_Svces","c_Energy","IND","SSA") = 123.452743530273 ;
vtwr("c_Svces","c_Energy","IND","ROW") = 653.622131347656 ;
vtwr("c_Svces","c_Energy","SSA","USA") = 278.602996826172 ;
vtwr("c_Svces","c_Energy","SSA","EU_28") = 841.603088378906 ;
vtwr("c_Svces","c_Energy","SSA","CHN") = 703.758911132812 ;
vtwr("c_Svces","c_Energy","SSA","JPN") = 96.0066757202148 ;
vtwr("c_Svces","c_Energy","SSA","IND") = 734.707397460938 ;
vtwr("c_Svces","c_Energy","SSA","SSA") = 336.133117675781 ;
vtwr("c_Svces","c_Energy","SSA","ROW") = 914.553344726562 ;
vtwr("c_Svces","c_Energy","ROW","USA") = 5684.77880859375 ;
vtwr("c_Svces","c_Energy","ROW","EU_28") = 13685.9951171875 ;
vtwr("c_Svces","c_Energy","ROW","CHN") = 8282.486328125 ;
vtwr("c_Svces","c_Energy","ROW","JPN") = 8160 ;
vtwr("c_Svces","c_Energy","ROW","IND") = 4455.31787109375 ;
vtwr("c_Svces","c_Energy","ROW","SSA") = 673.706970214844 ;
vtwr("c_Svces","c_Energy","ROW","ROW") = 19163.59375 ;
vtwr("c_Svces","c_Textiles","USA","EU_28") = 139.428161621094 ;
vtwr("c_Svces","c_Textiles","USA","CHN") = 74.6774749755859 ;
vtwr("c_Svces","c_Textiles","USA","JPN") = 23.5433692932129 ;
vtwr("c_Svces","c_Textiles","USA","IND") = 10.9012241363525 ;
vtwr("c_Svces","c_Textiles","USA","SSA") = 11.3436584472656 ;
vtwr("c_Svces","c_Textiles","USA","ROW") = 537.862915039062 ;
vtwr("c_Svces","c_Textiles","EU_28","USA") = 453.765563964844 ;
vtwr("c_Svces","c_Textiles","EU_28","EU_28") = 4269.78466796875 ;
vtwr("c_Svces","c_Textiles","EU_28","CHN") = 398.773284912109 ;
vtwr("c_Svces","c_Textiles","EU_28","JPN") = 194.689865112305 ;
vtwr("c_Svces","c_Textiles","EU_28","IND") = 36.0157775878906 ;
vtwr("c_Svces","c_Textiles","EU_28","SSA") = 88.283203125 ;
vtwr("c_Svces","c_Textiles","EU_28","ROW") = 1761.69775390625 ;
vtwr("c_Svces","c_Textiles","CHN","USA") = 3096.60473632812 ;
vtwr("c_Svces","c_Textiles","CHN","EU_28") = 2926.66040039062 ;
vtwr("c_Svces","c_Textiles","CHN","JPN") = 1290.51867675781 ;
vtwr("c_Svces","c_Textiles","CHN","IND") = 304.534851074219 ;
vtwr("c_Svces","c_Textiles","CHN","SSA") = 874.668395996094 ;
vtwr("c_Svces","c_Textiles","CHN","ROW") = 7786.41064453125 ;
vtwr("c_Svces","c_Textiles","JPN","USA") = 38.3449783325195 ;
vtwr("c_Svces","c_Textiles","JPN","EU_28") = 45.5900802612305 ;
vtwr("c_Svces","c_Textiles","JPN","CHN") = 192.177108764648 ;
vtwr("c_Svces","c_Textiles","JPN","IND") = 5.46834993362427 ;
vtwr("c_Svces","c_Textiles","JPN","SSA") = 9.0177640914917 ;
vtwr("c_Svces","c_Textiles","JPN","ROW") = 200.47966003418 ;
vtwr("c_Svces","c_Textiles","IND","USA") = 418.696594238281 ;
vtwr("c_Svces","c_Textiles","IND","EU_28") = 534.214477539062 ;
vtwr("c_Svces","c_Textiles","IND","CHN") = 91.2892456054688 ;
vtwr("c_Svces","c_Textiles","IND","JPN") = 24.0992012023926 ;
vtwr("c_Svces","c_Textiles","IND","SSA") = 100.705322265625 ;
vtwr("c_Svces","c_Textiles","IND","ROW") = 706.509399414062 ;
vtwr("c_Svces","c_Textiles","SSA","USA") = 42.4100151062012 ;
vtwr("c_Svces","c_Textiles","SSA","EU_28") = 48.9523391723633 ;
vtwr("c_Svces","c_Textiles","SSA","CHN") = 9.12544250488281 ;
vtwr("c_Svces","c_Textiles","SSA","JPN") = 0.706104934215546 ;
vtwr("c_Svces","c_Textiles","SSA","IND") = 1.46878576278687 ;
vtwr("c_Svces","c_Textiles","SSA","SSA") = 101.937522888184 ;
vtwr("c_Svces","c_Textiles","SSA","ROW") = 15.0842714309692 ;
vtwr("c_Svces","c_Textiles","ROW","USA") = 2543.24340820312 ;
vtwr("c_Svces","c_Textiles","ROW","EU_28") = 3553.33251953125 ;
vtwr("c_Svces","c_Textiles","ROW","CHN") = 1042.9208984375 ;
vtwr("c_Svces","c_Textiles","ROW","JPN") = 613.251831054688 ;
vtwr("c_Svces","c_Textiles","ROW","IND") = 121.171691894531 ;
vtwr("c_Svces","c_Textiles","ROW","SSA") = 137.736801147461 ;
vtwr("c_Svces","c_Textiles","ROW","ROW") = 3905.6064453125 ;
vtwr("c_Svces","c_Chem","USA","EU_28") = 1581.39562988281 ;
vtwr("c_Svces","c_Chem","USA","CHN") = 824.831970214844 ;
vtwr("c_Svces","c_Chem","USA","JPN") = 408.446105957031 ;
vtwr("c_Svces","c_Chem","USA","IND") = 191.419845581055 ;
vtwr("c_Svces","c_Chem","USA","SSA") = 95.7571487426758 ;
vtwr("c_Svces","c_Chem","USA","ROW") = 3844.7119140625 ;
vtwr("c_Svces","c_Chem","EU_28","USA") = 1765.52734375 ;
vtwr("c_Svces","c_Chem","EU_28","EU_28") = 12848.8134765625 ;
vtwr("c_Svces","c_Chem","EU_28","CHN") = 1429.45922851562 ;
vtwr("c_Svces","c_Chem","EU_28","JPN") = 401.044403076172 ;
vtwr("c_Svces","c_Chem","EU_28","IND") = 309.657073974609 ;
vtwr("c_Svces","c_Chem","EU_28","SSA") = 453.110107421875 ;
vtwr("c_Svces","c_Chem","EU_28","ROW") = 6260.81494140625 ;
vtwr("c_Svces","c_Chem","CHN","USA") = 1834.51489257812 ;
vtwr("c_Svces","c_Chem","CHN","EU_28") = 1405.85913085938 ;
vtwr("c_Svces","c_Chem","CHN","JPN") = 730.718811035156 ;
vtwr("c_Svces","c_Chem","CHN","IND") = 505.995727539062 ;
vtwr("c_Svces","c_Chem","CHN","SSA") = 549.982849121094 ;
vtwr("c_Svces","c_Chem","CHN","ROW") = 5397.6240234375 ;
vtwr("c_Svces","c_Chem","JPN","USA") = 456.698120117188 ;
vtwr("c_Svces","c_Chem","JPN","EU_28") = 338.620025634766 ;
vtwr("c_Svces","c_Chem","JPN","CHN") = 1615.07983398438 ;
vtwr("c_Svces","c_Chem","JPN","IND") = 109.438407897949 ;
vtwr("c_Svces","c_Chem","JPN","SSA") = 24.0438480377197 ;
vtwr("c_Svces","c_Chem","JPN","ROW") = 2167.88598632812 ;
vtwr("c_Svces","c_Chem","IND","USA") = 206.523345947266 ;
vtwr("c_Svces","c_Chem","IND","EU_28") = 264.91845703125 ;
vtwr("c_Svces","c_Chem","IND","CHN") = 131.738861083984 ;
vtwr("c_Svces","c_Chem","IND","JPN") = 24.5836696624756 ;
vtwr("c_Svces","c_Chem","IND","SSA") = 128.133651733398 ;
vtwr("c_Svces","c_Chem","IND","ROW") = 761.10302734375 ;
vtwr("c_Svces","c_Chem","SSA","USA") = 64.7259674072266 ;
vtwr("c_Svces","c_Chem","SSA","EU_28") = 154.478149414062 ;
vtwr("c_Svces","c_Chem","SSA","CHN") = 28.5579147338867 ;
vtwr("c_Svces","c_Chem","SSA","JPN") = 3.93041372299194 ;
vtwr("c_Svces","c_Chem","SSA","IND") = 63.478645324707 ;
vtwr("c_Svces","c_Chem","SSA","SSA") = 380.513275146484 ;
vtwr("c_Svces","c_Chem","SSA","ROW") = 138.292922973633 ;
vtwr("c_Svces","c_Chem","ROW","USA") = 2928.98999023438 ;
vtwr("c_Svces","c_Chem","ROW","EU_28") = 3886.53686523438 ;
vtwr("c_Svces","c_Chem","ROW","CHN") = 7611.83251953125 ;
vtwr("c_Svces","c_Chem","ROW","JPN") = 1182.822265625 ;
vtwr("c_Svces","c_Chem","ROW","IND") = 1510.07482910156 ;
vtwr("c_Svces","c_Chem","ROW","SSA") = 628.270263671875 ;
vtwr("c_Svces","c_Chem","ROW","ROW") = 12303.154296875 ;
vtwr("c_Svces","c_Manuf","USA","EU_28") = 3906.33471679688 ;
vtwr("c_Svces","c_Manuf","USA","CHN") = 2296.38427734375 ;
vtwr("c_Svces","c_Manuf","USA","JPN") = 988.37548828125 ;
vtwr("c_Svces","c_Manuf","USA","IND") = 452.591949462891 ;
vtwr("c_Svces","c_Manuf","USA","SSA") = 238.323989868164 ;
vtwr("c_Svces","c_Manuf","USA","ROW") = 9695.478515625 ;
vtwr("c_Svces","c_Manuf","EU_28","USA") = 7285.537109375 ;
vtwr("c_Svces","c_Manuf","EU_28","EU_28") = 41372.7578125 ;
vtwr("c_Svces","c_Manuf","EU_28","CHN") = 5784.931640625 ;
vtwr("c_Svces","c_Manuf","EU_28","JPN") = 1115.56970214844 ;
vtwr("c_Svces","c_Manuf","EU_28","IND") = 1041.1220703125 ;
vtwr("c_Svces","c_Manuf","EU_28","SSA") = 1586.70861816406 ;
vtwr("c_Svces","c_Manuf","EU_28","ROW") = 21234.78125 ;
vtwr("c_Svces","c_Manuf","CHN","USA") = 12964.6767578125 ;
vtwr("c_Svces","c_Manuf","CHN","EU_28") = 9194.3359375 ;
vtwr("c_Svces","c_Manuf","CHN","JPN") = 4326.134765625 ;
vtwr("c_Svces","c_Manuf","CHN","IND") = 1770.19152832031 ;
vtwr("c_Svces","c_Manuf","CHN","SSA") = 2308.48364257812 ;
vtwr("c_Svces","c_Manuf","CHN","ROW") = 25701.521484375 ;
vtwr("c_Svces","c_Manuf","JPN","USA") = 3018.51708984375 ;
vtwr("c_Svces","c_Manuf","JPN","EU_28") = 1652.17199707031 ;
vtwr("c_Svces","c_Manuf","JPN","CHN") = 4449.26611328125 ;
vtwr("c_Svces","c_Manuf","JPN","IND") = 266.036285400391 ;
vtwr("c_Svces","c_Manuf","JPN","SSA") = 179.480697631836 ;
vtwr("c_Svces","c_Manuf","JPN","ROW") = 7472.27099609375 ;
vtwr("c_Svces","c_Manuf","IND","USA") = 724.395263671875 ;
vtwr("c_Svces","c_Manuf","IND","EU_28") = 918.646667480469 ;
vtwr("c_Svces","c_Manuf","IND","CHN") = 665.9892578125 ;
vtwr("c_Svces","c_Manuf","IND","JPN") = 86.4634857177734 ;
vtwr("c_Svces","c_Manuf","IND","SSA") = 293.604370117188 ;
vtwr("c_Svces","c_Manuf","IND","ROW") = 2394.48657226562 ;
vtwr("c_Svces","c_Manuf","SSA","USA") = 228.537780761719 ;
vtwr("c_Svces","c_Manuf","SSA","EU_28") = 838.34130859375 ;
vtwr("c_Svces","c_Manuf","SSA","CHN") = 1712.791015625 ;
vtwr("c_Svces","c_Manuf","SSA","JPN") = 192.933776855469 ;
vtwr("c_Svces","c_Manuf","SSA","IND") = 277.937713623047 ;
vtwr("c_Svces","c_Manuf","SSA","SSA") = 1253.47302246094 ;
vtwr("c_Svces","c_Manuf","SSA","ROW") = 886.7958984375 ;
vtwr("c_Svces","c_Manuf","ROW","USA") = 11787.5615234375 ;
vtwr("c_Svces","c_Manuf","ROW","EU_28") = 13850.3095703125 ;
vtwr("c_Svces","c_Manuf","ROW","CHN") = 23911.666015625 ;
vtwr("c_Svces","c_Manuf","ROW","JPN") = 6086.7978515625 ;
vtwr("c_Svces","c_Manuf","ROW","IND") = 2833.08178710938 ;
vtwr("c_Svces","c_Manuf","ROW","SSA") = 1387.96411132812 ;
vtwr("c_Svces","c_Manuf","ROW","ROW") = 33747.5546875 ;
vtwr("c_Svces","c_ForestFish","USA","EU_28") = 28.8223667144775 ;
vtwr("c_Svces","c_ForestFish","USA","CHN") = 73.4970626831055 ;
vtwr("c_Svces","c_ForestFish","USA","JPN") = 52.9652099609375 ;
vtwr("c_Svces","c_ForestFish","USA","IND") = 0.693541765213013 ;
vtwr("c_Svces","c_ForestFish","USA","SSA") = 0.278845250606537 ;
vtwr("c_Svces","c_ForestFish","USA","ROW") = 39.7217864990234 ;
vtwr("c_Svces","c_ForestFish","EU_28","USA") = 27.0068607330322 ;
vtwr("c_Svces","c_ForestFish","EU_28","EU_28") = 731.01025390625 ;
vtwr("c_Svces","c_ForestFish","EU_28","CHN") = 59.4367408752441 ;
vtwr("c_Svces","c_ForestFish","EU_28","JPN") = 2.29311490058899 ;
vtwr("c_Svces","c_ForestFish","EU_28","IND") = 3.67912840843201 ;
vtwr("c_Svces","c_ForestFish","EU_28","SSA") = 1.70180594921112 ;
vtwr("c_Svces","c_ForestFish","EU_28","ROW") = 82.1951293945312 ;
vtwr("c_Svces","c_ForestFish","CHN","USA") = 3.61314105987549 ;
vtwr("c_Svces","c_ForestFish","CHN","EU_28") = 8.61995601654053 ;
vtwr("c_Svces","c_ForestFish","CHN","JPN") = 54.9554672241211 ;
vtwr("c_Svces","c_ForestFish","CHN","IND") = 0.043926116079092 ;
vtwr("c_Svces","c_ForestFish","CHN","SSA") = 0.0942000299692154 ;
vtwr("c_Svces","c_ForestFish","CHN","ROW") = 81.3262405395508 ;
vtwr("c_Svces","c_ForestFish","JPN","USA") = 5.93475818634033 ;
vtwr("c_Svces","c_ForestFish","JPN","EU_28") = 3.35449862480164 ;
vtwr("c_Svces","c_ForestFish","JPN","CHN") = 14.1082963943481 ;
vtwr("c_Svces","c_ForestFish","JPN","IND") = 0.0847414210438728 ;
vtwr("c_Svces","c_ForestFish","JPN","SSA") = 0.186173960566521 ;
vtwr("c_Svces","c_ForestFish","JPN","ROW") = 38.3595886230469 ;
vtwr("c_Svces","c_ForestFish","IND","USA") = 3.61575436592102 ;
vtwr("c_Svces","c_ForestFish","IND","EU_28") = 8.34340572357178 ;
vtwr("c_Svces","c_ForestFish","IND","CHN") = 9.83783054351807 ;
vtwr("c_Svces","c_ForestFish","IND","JPN") = 0.507229328155518 ;
vtwr("c_Svces","c_ForestFish","IND","SSA") = 0.26788866519928 ;
vtwr("c_Svces","c_ForestFish","IND","ROW") = 19.1200351715088 ;
vtwr("c_Svces","c_ForestFish","SSA","USA") = 4.25792121887207 ;
vtwr("c_Svces","c_ForestFish","SSA","EU_28") = 43.4982452392578 ;
vtwr("c_Svces","c_ForestFish","SSA","CHN") = 206.523666381836 ;
vtwr("c_Svces","c_ForestFish","SSA","JPN") = 1.23608529567719 ;
vtwr("c_Svces","c_ForestFish","SSA","IND") = 16.3237056732178 ;
vtwr("c_Svces","c_ForestFish","SSA","SSA") = 23.9351787567139 ;
vtwr("c_Svces","c_ForestFish","SSA","ROW") = 55.0630416870117 ;
vtwr("c_Svces","c_ForestFish","ROW","USA") = 140.678619384766 ;
vtwr("c_Svces","c_ForestFish","ROW","EU_28") = 331.290191650391 ;
vtwr("c_Svces","c_ForestFish","ROW","CHN") = 767.917236328125 ;
vtwr("c_Svces","c_ForestFish","ROW","JPN") = 129.303802490234 ;
vtwr("c_Svces","c_ForestFish","ROW","IND") = 106.478141784668 ;
vtwr("c_Svces","c_ForestFish","ROW","SSA") = 5.52436304092407 ;
vtwr("c_Svces","c_ForestFish","ROW","ROW") = 471.336456298828 ;
parameter save(*) ;
save("USA") = 1580785.625 ;
save("EU_28") = 2133911.25 ;
save("CHN") = 3840168.5 ;
save("JPN") = 543592.4375 ;
save("IND") = 467486.4375 ;
save("SSA") = 209834.765625 ;
save("ROW") = 3415068 ;
parameter vdep(*) ;
vdep("USA") = 1816268 ;
vdep("EU_28") = 1809763.25 ;
vdep("CHN") = 1607693.375 ;
vdep("JPN") = 707913.5 ;
vdep("IND") = 217300.71875 ;
vdep("SSA") = 124198.796875 ;
vdep("ROW") = 2264923 ;
parameter vkb(*) ;
vkb("USA") = 45406700 ;
vkb("EU_28") = 45244080 ;
vkb("CHN") = 40192336 ;
vkb("JPN") = 17697838 ;
vkb("IND") = 5432518 ;
vkb("SSA") = 3104970 ;
vkb("ROW") = 56623080 ;
parameter pop0(*) ;
pop0("USA") = 325.147125244141 ;
pop0("EU_28") = 513.872253417969 ;
pop0("CHN") = 1386.39501953125 ;
pop0("JPN") = 126.785797119141 ;
pop0("IND") = 1338.65881347656 ;
pop0("SSA") = 1051.37268066406 ;
pop0("ROW") = 2771.65209960938 ;
parameter maks(*,*,*) ;
maks("c_Rice","a_Rice","USA") = 10553.6875 ;
maks("c_Rice","a_Rice","EU_28") = 6879.37109375 ;
maks("c_Rice","a_Rice","CHN") = 179204.84375 ;
maks("c_Rice","a_Rice","JPN") = 38168.63671875 ;
maks("c_Rice","a_Rice","IND") = 84187.9453125 ;
maks("c_Rice","a_Rice","SSA") = 17939.173828125 ;
maks("c_Rice","a_Rice","ROW") = 365534.25 ;
maks("c_Crops","a_Crops","USA") = 227004.515625 ;
maks("c_Crops","a_Crops","EU_28") = 252454.109375 ;
maks("c_Crops","a_Crops","CHN") = 729052.125 ;
maks("c_Crops","a_Crops","JPN") = 34702.44140625 ;
maks("c_Crops","a_Crops","IND") = 241484.84375 ;
maks("c_Crops","a_Crops","SSA") = 228635.140625 ;
maks("c_Crops","a_Crops","ROW") = 918591.9375 ;
maks("c_Livestock","a_Livestock","USA") = 173237.765625 ;
maks("c_Livestock","a_Livestock","EU_28") = 194522.734375 ;
maks("c_Livestock","a_Livestock","CHN") = 365296.8125 ;
maks("c_Livestock","a_Livestock","JPN") = 34414.39453125 ;
maks("c_Livestock","a_Livestock","IND") = 126543.0390625 ;
maks("c_Livestock","a_Livestock","SSA") = 69657.5 ;
maks("c_Livestock","a_Livestock","ROW") = 562039.5625 ;
maks("c_FoodProc","a_FoodProc","USA") = 1109233.875 ;
maks("c_FoodProc","a_FoodProc","EU_28") = 1455317.25 ;
maks("c_FoodProc","a_FoodProc","CHN") = 1628540.5 ;
maks("c_FoodProc","a_FoodProc","JPN") = 293179.28125 ;
maks("c_FoodProc","a_FoodProc","IND") = 202942.453125 ;
maks("c_FoodProc","a_FoodProc","SSA") = 301065 ;
maks("c_FoodProc","a_FoodProc","ROW") = 2286207.75 ;
maks("c_Energy","a_Energy","USA") = 1082570.75 ;
maks("c_Energy","a_Energy","EU_28") = 885865.4375 ;
maks("c_Energy","a_Energy","CHN") = 1266234 ;
maks("c_Energy","a_Energy","JPN") = 284079 ;
maks("c_Energy","a_Energy","IND") = 485696.03125 ;
maks("c_Energy","a_Energy","SSA") = 208941 ;
maks("c_Energy","a_Energy","ROW") = 3753127.5 ;
maks("c_Textiles","a_Textiles","USA") = 93223.8671875 ;
maks("c_Textiles","a_Textiles","EU_28") = 420481.90625 ;
maks("c_Textiles","a_Textiles","CHN") = 1325668.625 ;
maks("c_Textiles","a_Textiles","JPN") = 40532.09375 ;
maks("c_Textiles","a_Textiles","IND") = 162754.3125 ;
maks("c_Textiles","a_Textiles","SSA") = 44125.28125 ;
maks("c_Textiles","a_Textiles","ROW") = 821641.625 ;
maks("c_Chem","a_Chem","USA") = 1029804.375 ;
maks("c_Chem","a_Chem","EU_28") = 1570803.75 ;
maks("c_Chem","a_Chem","CHN") = 1924916.25 ;
maks("c_Chem","a_Chem","JPN") = 443665 ;
maks("c_Chem","a_Chem","IND") = 280070.625 ;
maks("c_Chem","a_Chem","SSA") = 54723.6015625 ;
maks("c_Chem","a_Chem","ROW") = 1809458.875 ;
maks("c_Manuf","a_Manuf","USA") = 3955209.25 ;
maks("c_Manuf","a_Manuf","EU_28") = 6110222 ;
maks("c_Manuf","a_Manuf","CHN") = 8743891 ;
maks("c_Manuf","a_Manuf","JPN") = 2231835.5 ;
maks("c_Manuf","a_Manuf","IND") = 741464.6875 ;
maks("c_Manuf","a_Manuf","SSA") = 391325.0625 ;
maks("c_Manuf","a_Manuf","ROW") = 7344946 ;
maks("c_ForestFish","a_ForestFish","USA") = 59205.609375 ;
maks("c_ForestFish","a_ForestFish","EU_28") = 77960.2890625 ;
maks("c_ForestFish","a_ForestFish","CHN") = 246125.125 ;
maks("c_ForestFish","a_ForestFish","JPN") = 20446.341796875 ;
maks("c_ForestFish","a_ForestFish","IND") = 74477.8671875 ;
maks("c_ForestFish","a_ForestFish","SSA") = 68434.328125 ;
maks("c_ForestFish","a_ForestFish","ROW") = 276263.0625 ;
maks("c_Svces","a_Svces","USA") = 24328918 ;
maks("c_Svces","a_Svces","EU_28") = 22564282 ;
maks("c_Svces","a_Svces","CHN") = 14255377 ;
maks("c_Svces","a_Svces","JPN") = 5994771.5 ;
maks("c_Svces","a_Svces","IND") = 2622109.25 ;
maks("c_Svces","a_Svces","SSA") = 1569741.5 ;
maks("c_Svces","a_Svces","ROW") = 25016454 ;
parameter makb(*,*,*) ;
makb("c_Rice","a_Rice","USA") = 10640.8955078125 ;
makb("c_Rice","a_Rice","EU_28") = 7049.94970703125 ;
makb("c_Rice","a_Rice","CHN") = 179210.5625 ;
makb("c_Rice","a_Rice","JPN") = 38185.5625 ;
makb("c_Rice","a_Rice","IND") = 84165.109375 ;
makb("c_Rice","a_Rice","SSA") = 17966.6328125 ;
makb("c_Rice","a_Rice","ROW") = 365505.09375 ;
makb("c_Crops","a_Crops","USA") = 226965.140625 ;
makb("c_Crops","a_Crops","EU_28") = 252414.6875 ;
makb("c_Crops","a_Crops","CHN") = 725310.6875 ;
makb("c_Crops","a_Crops","JPN") = 34097.3984375 ;
makb("c_Crops","a_Crops","IND") = 241384 ;
makb("c_Crops","a_Crops","SSA") = 228558.40625 ;
makb("c_Crops","a_Crops","ROW") = 918871.25 ;
makb("c_Livestock","a_Livestock","USA") = 173235.53125 ;
makb("c_Livestock","a_Livestock","EU_28") = 193090.40625 ;
makb("c_Livestock","a_Livestock","CHN") = 365285.90625 ;
makb("c_Livestock","a_Livestock","JPN") = 33815.48046875 ;
makb("c_Livestock","a_Livestock","IND") = 126532.6484375 ;
makb("c_Livestock","a_Livestock","SSA") = 69559.4140625 ;
makb("c_Livestock","a_Livestock","ROW") = 561631.0625 ;
makb("c_FoodProc","a_FoodProc","USA") = 1148684.875 ;
makb("c_FoodProc","a_FoodProc","EU_28") = 1457072.75 ;
makb("c_FoodProc","a_FoodProc","CHN") = 1628716.625 ;
makb("c_FoodProc","a_FoodProc","JPN") = 320801.84375 ;
makb("c_FoodProc","a_FoodProc","IND") = 202972.046875 ;
makb("c_FoodProc","a_FoodProc","SSA") = 301746.125 ;
makb("c_FoodProc","a_FoodProc","ROW") = 2302656.25 ;
makb("c_Energy","a_Energy","USA") = 1163954.125 ;
makb("c_Energy","a_Energy","EU_28") = 890106.5 ;
makb("c_Energy","a_Energy","CHN") = 1266388.875 ;
makb("c_Energy","a_Energy","JPN") = 319227 ;
makb("c_Energy","a_Energy","IND") = 485717.40625 ;
makb("c_Energy","a_Energy","SSA") = 209621.96875 ;
makb("c_Energy","a_Energy","ROW") = 3760574.5 ;
makb("c_Textiles","a_Textiles","USA") = 94411.703125 ;
makb("c_Textiles","a_Textiles","EU_28") = 421233.53125 ;
makb("c_Textiles","a_Textiles","CHN") = 1325838.625 ;
makb("c_Textiles","a_Textiles","JPN") = 42217.95703125 ;
makb("c_Textiles","a_Textiles","IND") = 162757.5 ;
makb("c_Textiles","a_Textiles","SSA") = 44055.03515625 ;
makb("c_Textiles","a_Textiles","ROW") = 824875.375 ;
makb("c_Chem","a_Chem","USA") = 1046756.25 ;
makb("c_Chem","a_Chem","EU_28") = 1575606.25 ;
makb("c_Chem","a_Chem","CHN") = 1925106.375 ;
makb("c_Chem","a_Chem","JPN") = 453806.25 ;
makb("c_Chem","a_Chem","IND") = 280075.21875 ;
makb("c_Chem","a_Chem","SSA") = 54778.6953125 ;
makb("c_Chem","a_Chem","ROW") = 1818529.375 ;
makb("c_Manuf","a_Manuf","USA") = 4032121.75 ;
makb("c_Manuf","a_Manuf","EU_28") = 6129109 ;
makb("c_Manuf","a_Manuf","CHN") = 8744415 ;
makb("c_Manuf","a_Manuf","JPN") = 2256697.5 ;
makb("c_Manuf","a_Manuf","IND") = 741476.125 ;
makb("c_Manuf","a_Manuf","SSA") = 391283.59375 ;
makb("c_Manuf","a_Manuf","ROW") = 7375501.5 ;
makb("c_ForestFish","a_ForestFish","USA") = 61058.9375 ;
makb("c_ForestFish","a_ForestFish","EU_28") = 77485.84375 ;
makb("c_ForestFish","a_ForestFish","CHN") = 246139.15625 ;
makb("c_ForestFish","a_ForestFish","JPN") = 20915.87890625 ;
makb("c_ForestFish","a_ForestFish","IND") = 74478.109375 ;
makb("c_ForestFish","a_ForestFish","SSA") = 68807.6640625 ;
makb("c_ForestFish","a_ForestFish","ROW") = 274373.34375 ;
makb("c_Svces","a_Svces","USA") = 25332292 ;
makb("c_Svces","a_Svces","EU_28") = 22761286 ;
makb("c_Svces","a_Svces","CHN") = 14256120 ;
makb("c_Svces","a_Svces","JPN") = 6166068 ;
makb("c_Svces","a_Svces","IND") = 2622143.25 ;
makb("c_Svces","a_Svces","SSA") = 1580003.375 ;
makb("c_Svces","a_Svces","ROW") = 25255732 ;
parameter esubt(*,*) ;
parameter esubc(*,*) ;
parameter esubva(*,*) ;
esubva("a_Rice","USA") = 0.466409474611282 ;
esubva("a_Rice","EU_28") = 0.466409474611282 ;
esubva("a_Rice","CHN") = 0.466409474611282 ;
esubva("a_Rice","JPN") = 0.466409474611282 ;
esubva("a_Rice","IND") = 0.466409474611282 ;
esubva("a_Rice","SSA") = 0.466409474611282 ;
esubva("a_Rice","ROW") = 0.466409474611282 ;
esubva("a_Crops","USA") = 0.259209901094437 ;
esubva("a_Crops","EU_28") = 0.259209901094437 ;
esubva("a_Crops","CHN") = 0.259209901094437 ;
esubva("a_Crops","JPN") = 0.259209901094437 ;
esubva("a_Crops","IND") = 0.259209901094437 ;
esubva("a_Crops","SSA") = 0.259209901094437 ;
esubva("a_Crops","ROW") = 0.259209901094437 ;
esubva("a_Livestock","USA") = 0.259209901094437 ;
esubva("a_Livestock","EU_28") = 0.259209901094437 ;
esubva("a_Livestock","CHN") = 0.259209901094437 ;
esubva("a_Livestock","JPN") = 0.259209901094437 ;
esubva("a_Livestock","IND") = 0.259209901094437 ;
esubva("a_Livestock","SSA") = 0.259209901094437 ;
esubva("a_Livestock","ROW") = 0.259209901094437 ;
esubva("a_FoodProc","USA") = 1.12000000476837 ;
esubva("a_FoodProc","EU_28") = 1.12000000476837 ;
esubva("a_FoodProc","CHN") = 1.12000000476837 ;
esubva("a_FoodProc","JPN") = 1.12000000476837 ;
esubva("a_FoodProc","IND") = 1.12000000476837 ;
esubva("a_FoodProc","SSA") = 1.12000000476837 ;
esubva("a_FoodProc","ROW") = 1.12000000476837 ;
esubva("a_Energy","USA") = 0.716633558273315 ;
esubva("a_Energy","EU_28") = 0.716633558273315 ;
esubva("a_Energy","CHN") = 0.716633558273315 ;
esubva("a_Energy","JPN") = 0.716633558273315 ;
esubva("a_Energy","IND") = 0.716633558273315 ;
esubva("a_Energy","SSA") = 0.716633558273315 ;
esubva("a_Energy","ROW") = 0.716633558273315 ;
esubva("a_Textiles","USA") = 1.25999999046326 ;
esubva("a_Textiles","EU_28") = 1.25999999046326 ;
esubva("a_Textiles","CHN") = 1.25999999046326 ;
esubva("a_Textiles","JPN") = 1.25999999046326 ;
esubva("a_Textiles","IND") = 1.25999999046326 ;
esubva("a_Textiles","SSA") = 1.25999999046326 ;
esubva("a_Textiles","ROW") = 1.25999999046326 ;
esubva("a_Chem","USA") = 1.25999999046326 ;
esubva("a_Chem","EU_28") = 1.25999999046326 ;
esubva("a_Chem","CHN") = 1.25999999046326 ;
esubva("a_Chem","JPN") = 1.25999999046326 ;
esubva("a_Chem","IND") = 1.25999999046326 ;
esubva("a_Chem","SSA") = 1.25999999046326 ;
esubva("a_Chem","ROW") = 1.25999999046326 ;
esubva("a_Manuf","USA") = 1.17859554290771 ;
esubva("a_Manuf","EU_28") = 1.17859554290771 ;
esubva("a_Manuf","CHN") = 1.17859554290771 ;
esubva("a_Manuf","JPN") = 1.17859554290771 ;
esubva("a_Manuf","IND") = 1.17859554290771 ;
esubva("a_Manuf","SSA") = 1.17859554290771 ;
esubva("a_Manuf","ROW") = 1.17859554290771 ;
esubva("a_ForestFish","USA") = 0.200000002980232 ;
esubva("a_ForestFish","EU_28") = 0.200000002980232 ;
esubva("a_ForestFish","CHN") = 0.200000002980232 ;
esubva("a_ForestFish","JPN") = 0.200000002980232 ;
esubva("a_ForestFish","IND") = 0.200000002980232 ;
esubva("a_ForestFish","SSA") = 0.200000002980232 ;
esubva("a_ForestFish","ROW") = 0.200000002980232 ;
esubva("a_Svces","USA") = 1.37246572971344 ;
esubva("a_Svces","EU_28") = 1.37246572971344 ;
esubva("a_Svces","CHN") = 1.37246572971344 ;
esubva("a_Svces","JPN") = 1.37246572971344 ;
esubva("a_Svces","IND") = 1.37246572971344 ;
esubva("a_Svces","SSA") = 1.37246572971344 ;
esubva("a_Svces","ROW") = 1.37246572971344 ;
parameter etraq(*,*) ;
etraq("a_Rice","USA") = -5 ;
etraq("a_Rice","EU_28") = -5 ;
etraq("a_Rice","CHN") = -4.99999952316284 ;
etraq("a_Rice","JPN") = -5 ;
etraq("a_Rice","IND") = -5 ;
etraq("a_Rice","SSA") = -5 ;
etraq("a_Rice","ROW") = -5 ;
etraq("a_Crops","USA") = -5 ;
etraq("a_Crops","EU_28") = -5 ;
etraq("a_Crops","CHN") = -5 ;
etraq("a_Crops","JPN") = -5 ;
etraq("a_Crops","IND") = -5 ;
etraq("a_Crops","SSA") = -5 ;
etraq("a_Crops","ROW") = -5 ;
etraq("a_Livestock","USA") = -5 ;
etraq("a_Livestock","EU_28") = -5 ;
etraq("a_Livestock","CHN") = -5 ;
etraq("a_Livestock","JPN") = -5 ;
etraq("a_Livestock","IND") = -5 ;
etraq("a_Livestock","SSA") = -5 ;
etraq("a_Livestock","ROW") = -5 ;
etraq("a_FoodProc","USA") = -5 ;
etraq("a_FoodProc","EU_28") = -5 ;
etraq("a_FoodProc","CHN") = -5 ;
etraq("a_FoodProc","JPN") = -5 ;
etraq("a_FoodProc","IND") = -5 ;
etraq("a_FoodProc","SSA") = -5 ;
etraq("a_FoodProc","ROW") = -5 ;
etraq("a_Energy","USA") = -5 ;
etraq("a_Energy","EU_28") = -5 ;
etraq("a_Energy","CHN") = -5 ;
etraq("a_Energy","JPN") = -5 ;
etraq("a_Energy","IND") = -5 ;
etraq("a_Energy","SSA") = -5 ;
etraq("a_Energy","ROW") = -5 ;
etraq("a_Textiles","USA") = -5 ;
etraq("a_Textiles","EU_28") = -5 ;
etraq("a_Textiles","CHN") = -5 ;
etraq("a_Textiles","JPN") = -5 ;
etraq("a_Textiles","IND") = -5 ;
etraq("a_Textiles","SSA") = -5 ;
etraq("a_Textiles","ROW") = -5 ;
etraq("a_Chem","USA") = -5 ;
etraq("a_Chem","EU_28") = -5 ;
etraq("a_Chem","CHN") = -5 ;
etraq("a_Chem","JPN") = -5 ;
etraq("a_Chem","IND") = -5 ;
etraq("a_Chem","SSA") = -5 ;
etraq("a_Chem","ROW") = -5 ;
etraq("a_Manuf","USA") = -5 ;
etraq("a_Manuf","EU_28") = -5 ;
etraq("a_Manuf","CHN") = -5 ;
etraq("a_Manuf","JPN") = -5 ;
etraq("a_Manuf","IND") = -5 ;
etraq("a_Manuf","SSA") = -5 ;
etraq("a_Manuf","ROW") = -5 ;
etraq("a_ForestFish","USA") = -5 ;
etraq("a_ForestFish","EU_28") = -5 ;
etraq("a_ForestFish","CHN") = -5 ;
etraq("a_ForestFish","JPN") = -5 ;
etraq("a_ForestFish","IND") = -5 ;
etraq("a_ForestFish","SSA") = -5 ;
etraq("a_ForestFish","ROW") = -5 ;
etraq("a_Svces","USA") = -5 ;
etraq("a_Svces","EU_28") = -5 ;
etraq("a_Svces","CHN") = -5 ;
etraq("a_Svces","JPN") = -5 ;
etraq("a_Svces","IND") = -5 ;
etraq("a_Svces","SSA") = -5 ;
etraq("a_Svces","ROW") = -5 ;
parameter esubq(*,*) ;
parameter incpar(*,*) ;
incpar("c_Rice","USA") = 9.99999997475243E-7 ;
incpar("c_Rice","EU_28") = 0.00212708255276084 ;
incpar("c_Rice","CHN") = 0.120247460901737 ;
incpar("c_Rice","JPN") = 9.99999997475243E-7 ;
incpar("c_Rice","IND") = 0.412129461765289 ;
incpar("c_Rice","SSA") = 0.556492567062378 ;
incpar("c_Rice","ROW") = 0.305558443069458 ;
incpar("c_Crops","USA") = 0.0426319465041161 ;
incpar("c_Crops","EU_28") = 0.0064289360307157 ;
incpar("c_Crops","CHN") = 0.120967626571655 ;
incpar("c_Crops","JPN") = 0.00353082944639027 ;
incpar("c_Crops","IND") = 0.413100212812424 ;
incpar("c_Crops","SSA") = 0.589530885219574 ;
incpar("c_Crops","ROW") = 0.221132561564445 ;
incpar("c_Livestock","USA") = 0.503583550453186 ;
incpar("c_Livestock","EU_28") = 0.554161012172699 ;
incpar("c_Livestock","CHN") = 0.462862610816956 ;
incpar("c_Livestock","JPN") = 0.54886382818222 ;
incpar("c_Livestock","IND") = 0.608004927635193 ;
incpar("c_Livestock","SSA") = 0.861532330513 ;
incpar("c_Livestock","ROW") = 0.576562762260437 ;
incpar("c_FoodProc","USA") = 0.412242442369461 ;
incpar("c_FoodProc","EU_28") = 0.465652316808701 ;
incpar("c_FoodProc","CHN") = 0.385069608688354 ;
incpar("c_FoodProc","JPN") = 0.436722487211227 ;
incpar("c_FoodProc","IND") = 0.574326694011688 ;
incpar("c_FoodProc","SSA") = 0.68660181760788 ;
incpar("c_FoodProc","ROW") = 0.455350816249847 ;
incpar("c_Energy","USA") = 0.913325369358063 ;
incpar("c_Energy","EU_28") = 0.993315696716309 ;
incpar("c_Energy","CHN") = 1.02250564098358 ;
incpar("c_Energy","JPN") = 0.962554037570953 ;
incpar("c_Energy","IND") = 0.928117275238037 ;
incpar("c_Energy","SSA") = 1.0403550863266 ;
incpar("c_Energy","ROW") = 1.0014123916626 ;
incpar("c_Textiles","USA") = 0.490339607000351 ;
incpar("c_Textiles","EU_28") = 0.543874442577362 ;
incpar("c_Textiles","CHN") = 0.449346840381622 ;
incpar("c_Textiles","JPN") = 0.534958124160767 ;
incpar("c_Textiles","IND") = 0.600794494152069 ;
incpar("c_Textiles","SSA") = 0.749316453933716 ;
incpar("c_Textiles","ROW") = 0.529085099697113 ;
incpar("c_Chem","USA") = 0.959409177303314 ;
incpar("c_Chem","EU_28") = 1.04049396514893 ;
incpar("c_Chem","CHN") = 1.05535733699799 ;
incpar("c_Chem","JPN") = 1.00990867614746 ;
incpar("c_Chem","IND") = 0.824252128601074 ;
incpar("c_Chem","SSA") = 0.929810225963593 ;
incpar("c_Chem","ROW") = 1.03163480758667 ;
incpar("c_Manuf","USA") = 0.951710164546967 ;
incpar("c_Manuf","EU_28") = 1.03333282470703 ;
incpar("c_Manuf","CHN") = 1.03870010375977 ;
incpar("c_Manuf","JPN") = 1.00475943088531 ;
incpar("c_Manuf","IND") = 0.826797366142273 ;
incpar("c_Manuf","SSA") = 0.944404125213623 ;
incpar("c_Manuf","ROW") = 1.01845932006836 ;
incpar("c_ForestFish","USA") = 0.559398651123047 ;
incpar("c_ForestFish","EU_28") = 0.742639303207397 ;
incpar("c_ForestFish","CHN") = 0.478631407022476 ;
incpar("c_ForestFish","JPN") = 0.702956259250641 ;
incpar("c_ForestFish","IND") = 0.68210905790329 ;
incpar("c_ForestFish","SSA") = 0.786649167537689 ;
incpar("c_ForestFish","ROW") = 0.668727099895477 ;
incpar("c_Svces","USA") = 1.06106984615326 ;
incpar("c_Svces","EU_28") = 1.12457609176636 ;
incpar("c_Svces","CHN") = 1.3661208152771 ;
incpar("c_Svces","JPN") = 1.10063672065735 ;
incpar("c_Svces","IND") = 1.31016874313354 ;
incpar("c_Svces","SSA") = 1.38129651546478 ;
incpar("c_Svces","ROW") = 1.17885303497314 ;
parameter subpar(*,*) ;
subpar("c_Rice","USA") = 0.999999046325684 ;
subpar("c_Rice","EU_28") = 0.994942486286163 ;
subpar("c_Rice","CHN") = 0.874023795127869 ;
subpar("c_Rice","JPN") = 0.999999046325684 ;
subpar("c_Rice","IND") = 0.844606816768646 ;
subpar("c_Rice","SSA") = 0.856722176074982 ;
subpar("c_Rice","ROW") = 0.88401335477829 ;
subpar("c_Crops","USA") = 0.927273273468018 ;
subpar("c_Crops","EU_28") = 0.990405738353729 ;
subpar("c_Crops","CHN") = 0.873587131500244 ;
subpar("c_Crops","JPN") = 0.995540499687195 ;
subpar("c_Crops","IND") = 0.844291925430298 ;
subpar("c_Crops","SSA") = 0.854276180267334 ;
subpar("c_Crops","ROW") = 0.900789141654968 ;
subpar("c_Livestock","USA") = 0.159861207008362 ;
subpar("c_Livestock","EU_28") = 0.374921709299088 ;
subpar("c_Livestock","CHN") = 0.671416223049164 ;
subpar("c_Livestock","JPN") = 0.318815499544144 ;
subpar("c_Livestock","IND") = 0.784091472625732 ;
subpar("c_Livestock","SSA") = 0.796490967273712 ;
subpar("c_Livestock","ROW") = 0.688416540622711 ;
subpar("c_FoodProc","USA") = 0.192402675747871 ;
subpar("c_FoodProc","EU_28") = 0.396075993776321 ;
subpar("c_FoodProc","CHN") = 0.716997504234314 ;
subpar("c_FoodProc","JPN") = 0.37477120757103 ;
subpar("c_FoodProc","IND") = 0.7923504114151 ;
subpar("c_FoodProc","SSA") = 0.797080397605896 ;
subpar("c_FoodProc","ROW") = 0.641632676124573 ;
subpar("c_Energy","USA") = 0.0938753336668015 ;
subpar("c_Energy","EU_28") = 0.231758579611778 ;
subpar("c_Energy","CHN") = 0.493528544902802 ;
subpar("c_Energy","JPN") = 0.21361032128334 ;
subpar("c_Energy","IND") = 0.705166637897491 ;
subpar("c_Energy","SSA") = 0.662657856941223 ;
subpar("c_Energy","ROW") = 0.443222939968109 ;
subpar("c_Textiles","USA") = 0.163511574268341 ;
subpar("c_Textiles","EU_28") = 0.340729117393494 ;
subpar("c_Textiles","CHN") = 0.674481928348541 ;
subpar("c_Textiles","JPN") = 0.324299931526184 ;
subpar("c_Textiles","IND") = 0.783409535884857 ;
subpar("c_Textiles","SSA") = 0.782757639884949 ;
subpar("c_Textiles","ROW") = 0.590150356292725 ;
subpar("c_Chem","USA") = 0.0891640931367874 ;
subpar("c_Chem","EU_28") = 0.228020578622818 ;
subpar("c_Chem","CHN") = 0.486900389194489 ;
subpar("c_Chem","JPN") = 0.204949423670769 ;
subpar("c_Chem","IND") = 0.727602183818817 ;
subpar("c_Chem","SSA") = 0.716785311698914 ;
subpar("c_Chem","ROW") = 0.433151990175247 ;
subpar("c_Manuf","USA") = 0.0899338349699974 ;
subpar("c_Manuf","EU_28") = 0.217258408665657 ;
subpar("c_Manuf","CHN") = 0.490708261728287 ;
subpar("c_Manuf","JPN") = 0.205864652991295 ;
subpar("c_Manuf","IND") = 0.727050185203552 ;
subpar("c_Manuf","SSA") = 0.694040238857269 ;
subpar("c_Manuf","ROW") = 0.391688585281372 ;
subpar("c_ForestFish","USA") = 0.151201292872429 ;
subpar("c_ForestFish","EU_28") = 0.328970789909363 ;
subpar("c_ForestFish","CHN") = 0.666537582874298 ;
subpar("c_ForestFish","JPN") = 0.28075760602951 ;
subpar("c_ForestFish","IND") = 0.764779269695282 ;
subpar("c_ForestFish","SSA") = 0.784681558609009 ;
subpar("c_ForestFish","ROW") = 0.672996163368225 ;
subpar("c_Svces","USA") = 0.0879293978214264 ;
subpar("c_Svces","EU_28") = 0.206386521458626 ;
subpar("c_Svces","CHN") = 0.442120015621185 ;
subpar("c_Svces","JPN") = 0.196586176753044 ;
subpar("c_Svces","IND") = 0.647772967815399 ;
subpar("c_Svces","SSA") = 0.609034180641174 ;
subpar("c_Svces","ROW") = 0.359811842441559 ;
parameter esubg(*) ;
esubg("USA") = 1 ;
esubg("EU_28") = 1 ;
esubg("CHN") = 1 ;
esubg("JPN") = 1 ;
esubg("IND") = 1 ;
esubg("SSA") = 1 ;
esubg("ROW") = 1 ;
parameter esubi(*) ;
parameter esubd(*,*) ;
esubd("c_Rice","USA") = 3.8695957660675 ;
esubd("c_Rice","EU_28") = 3.8695957660675 ;
esubd("c_Rice","CHN") = 3.8695957660675 ;
esubd("c_Rice","JPN") = 3.8695957660675 ;
esubd("c_Rice","IND") = 3.8695957660675 ;
esubd("c_Rice","SSA") = 3.8695957660675 ;
esubd("c_Rice","ROW") = 3.8695957660675 ;
esubd("c_Crops","USA") = 2.24410939216614 ;
esubd("c_Crops","EU_28") = 2.24410939216614 ;
esubd("c_Crops","CHN") = 2.24410939216614 ;
esubd("c_Crops","JPN") = 2.24410939216614 ;
esubd("c_Crops","IND") = 2.24410939216614 ;
esubd("c_Crops","SSA") = 2.24410939216614 ;
esubd("c_Crops","ROW") = 2.24410939216614 ;
esubd("c_Livestock","USA") = 2.15929388999939 ;
esubd("c_Livestock","EU_28") = 2.15929388999939 ;
esubd("c_Livestock","CHN") = 2.15929388999939 ;
esubd("c_Livestock","JPN") = 2.15929388999939 ;
esubd("c_Livestock","IND") = 2.15929388999939 ;
esubd("c_Livestock","SSA") = 2.15929388999939 ;
esubd("c_Livestock","ROW") = 2.15929388999939 ;
esubd("c_FoodProc","USA") = 2.47858667373657 ;
esubd("c_FoodProc","EU_28") = 2.47858667373657 ;
esubd("c_FoodProc","CHN") = 2.47858667373657 ;
esubd("c_FoodProc","JPN") = 2.47858667373657 ;
esubd("c_FoodProc","IND") = 2.47858667373657 ;
esubd("c_FoodProc","SSA") = 2.47858667373657 ;
esubd("c_FoodProc","ROW") = 2.47858667373657 ;
esubd("c_Energy","USA") = 3.81850719451904 ;
esubd("c_Energy","EU_28") = 3.81850719451904 ;
esubd("c_Energy","CHN") = 3.81850719451904 ;
esubd("c_Energy","JPN") = 3.81850719451904 ;
esubd("c_Energy","IND") = 3.81850719451904 ;
esubd("c_Energy","SSA") = 3.81850719451904 ;
esubd("c_Energy","ROW") = 3.81850719451904 ;
esubd("c_Textiles","USA") = 3.77994012832642 ;
esubd("c_Textiles","EU_28") = 3.77994012832642 ;
esubd("c_Textiles","CHN") = 3.77994012832642 ;
esubd("c_Textiles","JPN") = 3.77994012832642 ;
esubd("c_Textiles","IND") = 3.77994012832642 ;
esubd("c_Textiles","SSA") = 3.77994012832642 ;
esubd("c_Textiles","ROW") = 3.77994012832642 ;
esubd("c_Chem","USA") = 3.29999995231628 ;
esubd("c_Chem","EU_28") = 3.29999995231628 ;
esubd("c_Chem","CHN") = 3.29999995231628 ;
esubd("c_Chem","JPN") = 3.29999995231628 ;
esubd("c_Chem","IND") = 3.29999995231628 ;
esubd("c_Chem","SSA") = 3.29999995231628 ;
esubd("c_Chem","ROW") = 3.29999995231628 ;
esubd("c_Manuf","USA") = 3.54732418060303 ;
esubd("c_Manuf","EU_28") = 3.54732418060303 ;
esubd("c_Manuf","CHN") = 3.54732418060303 ;
esubd("c_Manuf","JPN") = 3.54732418060303 ;
esubd("c_Manuf","IND") = 3.54732418060303 ;
esubd("c_Manuf","SSA") = 3.54732418060303 ;
esubd("c_Manuf","ROW") = 3.54732418060303 ;
esubd("c_ForestFish","USA") = 1.82942163944244 ;
esubd("c_ForestFish","EU_28") = 1.82942163944244 ;
esubd("c_ForestFish","CHN") = 1.82942163944244 ;
esubd("c_ForestFish","JPN") = 1.82942163944244 ;
esubd("c_ForestFish","IND") = 1.82942163944244 ;
esubd("c_ForestFish","SSA") = 1.82942163944244 ;
esubd("c_ForestFish","ROW") = 1.82942163944244 ;
esubd("c_Svces","USA") = 1.90913510322571 ;
esubd("c_Svces","EU_28") = 1.90913510322571 ;
esubd("c_Svces","CHN") = 1.90913510322571 ;
esubd("c_Svces","JPN") = 1.90913510322571 ;
esubd("c_Svces","IND") = 1.90913510322571 ;
esubd("c_Svces","SSA") = 1.90913510322571 ;
esubd("c_Svces","ROW") = 1.90913510322571 ;
parameter esubm(*,*) ;
esubm("c_Rice","USA") = 5.35415601730347 ;
esubm("c_Rice","EU_28") = 5.35415601730347 ;
esubm("c_Rice","CHN") = 5.35415601730347 ;
esubm("c_Rice","JPN") = 5.35415601730347 ;
esubm("c_Rice","IND") = 5.35415601730347 ;
esubm("c_Rice","SSA") = 5.35415601730347 ;
esubm("c_Rice","ROW") = 5.35415601730347 ;
esubm("c_Crops","USA") = 4.81552934646606 ;
esubm("c_Crops","EU_28") = 4.81552934646606 ;
esubm("c_Crops","CHN") = 4.81552934646606 ;
esubm("c_Crops","JPN") = 4.81552934646606 ;
esubm("c_Crops","IND") = 4.81552934646606 ;
esubm("c_Crops","SSA") = 4.81552934646606 ;
esubm("c_Crops","ROW") = 4.81552934646606 ;
esubm("c_Livestock","USA") = 4.02048635482788 ;
esubm("c_Livestock","EU_28") = 4.02048635482788 ;
esubm("c_Livestock","CHN") = 4.02048635482788 ;
esubm("c_Livestock","JPN") = 4.02048635482788 ;
esubm("c_Livestock","IND") = 4.02048635482788 ;
esubm("c_Livestock","SSA") = 4.02048635482788 ;
esubm("c_Livestock","ROW") = 4.02048635482788 ;
esubm("c_FoodProc","USA") = 4.81388235092163 ;
esubm("c_FoodProc","EU_28") = 4.81388235092163 ;
esubm("c_FoodProc","CHN") = 4.81388235092163 ;
esubm("c_FoodProc","JPN") = 4.81388235092163 ;
esubm("c_FoodProc","IND") = 4.81388235092163 ;
esubm("c_FoodProc","SSA") = 4.81388235092163 ;
esubm("c_FoodProc","ROW") = 4.81388235092163 ;
esubm("c_Energy","USA") = 11.4968242645264 ;
esubm("c_Energy","EU_28") = 11.4968242645264 ;
esubm("c_Energy","CHN") = 11.4968242645264 ;
esubm("c_Energy","JPN") = 11.4968242645264 ;
esubm("c_Energy","IND") = 11.4968242645264 ;
esubm("c_Energy","SSA") = 11.4968242645264 ;
esubm("c_Energy","ROW") = 11.4968242645264 ;
esubm("c_Textiles","USA") = 7.58905839920044 ;
esubm("c_Textiles","EU_28") = 7.58905839920044 ;
esubm("c_Textiles","CHN") = 7.58905839920044 ;
esubm("c_Textiles","JPN") = 7.58905839920044 ;
esubm("c_Textiles","IND") = 7.58905839920044 ;
esubm("c_Textiles","SSA") = 7.58905839920044 ;
esubm("c_Textiles","ROW") = 7.58905839920044 ;
esubm("c_Chem","USA") = 6.59999990463257 ;
esubm("c_Chem","EU_28") = 6.59999990463257 ;
esubm("c_Chem","CHN") = 6.59999990463257 ;
esubm("c_Chem","JPN") = 6.59999990463257 ;
esubm("c_Chem","IND") = 6.59999990463257 ;
esubm("c_Chem","SSA") = 6.59999990463257 ;
esubm("c_Chem","ROW") = 6.59999990463257 ;
esubm("c_Manuf","USA") = 7.46128559112549 ;
esubm("c_Manuf","EU_28") = 7.46128559112549 ;
esubm("c_Manuf","CHN") = 7.46128559112549 ;
esubm("c_Manuf","JPN") = 7.46128559112549 ;
esubm("c_Manuf","IND") = 7.46128559112549 ;
esubm("c_Manuf","SSA") = 7.46128559112549 ;
esubm("c_Manuf","ROW") = 7.46128559112549 ;
esubm("c_ForestFish","USA") = 3.60396528244019 ;
esubm("c_ForestFish","EU_28") = 3.60396528244019 ;
esubm("c_ForestFish","CHN") = 3.60396528244019 ;
esubm("c_ForestFish","JPN") = 3.60396528244019 ;
esubm("c_ForestFish","IND") = 3.60396528244019 ;
esubm("c_ForestFish","SSA") = 3.60396528244019 ;
esubm("c_ForestFish","ROW") = 3.60396528244019 ;
esubm("c_Svces","USA") = 3.80041527748108 ;
esubm("c_Svces","EU_28") = 3.80041527748108 ;
esubm("c_Svces","CHN") = 3.80041527748108 ;
esubm("c_Svces","JPN") = 3.80041527748108 ;
esubm("c_Svces","IND") = 3.80041527748108 ;
esubm("c_Svces","SSA") = 3.80041527748108 ;
esubm("c_Svces","ROW") = 3.80041527748108 ;
parameter esubs(*) ;
esubs("c_Svces") = 1 ;
parameter etrae(*,*) ;
etrae("Land","USA") = -1 ;
etrae("Land","EU_28") = -1 ;
etrae("Land","CHN") = -1 ;
etrae("Land","JPN") = -1 ;
etrae("Land","IND") = -1 ;
etrae("Land","SSA") = -1 ;
etrae("Land","ROW") = -1 ;
etrae("NatRes","USA") = -0.00100000004749745 ;
etrae("NatRes","EU_28") = -0.00100000004749745 ;
etrae("NatRes","CHN") = -0.00100000004749745 ;
etrae("NatRes","JPN") = -0.00100000004749745 ;
etrae("NatRes","IND") = -0.00100000004749745 ;
etrae("NatRes","SSA") = -0.00100000004749745 ;
etrae("NatRes","ROW") = -0.00100000004749745 ;
parameter rorFlex0(*) ;
rorFlex0("USA") = 10 ;
rorFlex0("EU_28") = 10 ;
rorFlex0("CHN") = 10 ;
rorFlex0("JPN") = 10 ;
rorFlex0("IND") = 10 ;
rorFlex0("SSA") = 10 ;
rorFlex0("ROW") = 10 ;

parameter esubd(*,*) ;
esubd("c_Rice","USA") = 3.869595766067505 ;
esubd("c_Crops","USA") = 2.2441093921661377 ;
esubd("c_Livestock","USA") = 2.1592938899993896 ;
esubd("c_FoodProc","USA") = 2.4785866737365723 ;
esubd("c_Energy","USA") = 3.818507194519043 ;
esubd("c_Textiles","USA") = 3.779940128326416 ;
esubd("c_Chem","USA") = 3.299999952316284 ;
esubd("c_Manuf","USA") = 3.5473241806030273 ;
esubd("c_ForestFish","USA") = 1.8294216394424438 ;
esubd("c_Svces","USA") = 1.909135103225708 ;
esubd("c_Rice","EU_28") = 3.869595766067505 ;
esubd("c_Crops","EU_28") = 2.2441093921661377 ;
esubd("c_Livestock","EU_28") = 2.1592938899993896 ;
esubd("c_FoodProc","EU_28") = 2.4785866737365723 ;
esubd("c_Energy","EU_28") = 3.818507194519043 ;
esubd("c_Textiles","EU_28") = 3.779940128326416 ;
esubd("c_Chem","EU_28") = 3.299999952316284 ;
esubd("c_Manuf","EU_28") = 3.5473241806030273 ;
esubd("c_ForestFish","EU_28") = 1.8294216394424438 ;
esubd("c_Svces","EU_28") = 1.909135103225708 ;
esubd("c_Rice","CHN") = 3.869595766067505 ;
esubd("c_Crops","CHN") = 2.2441093921661377 ;
esubd("c_Livestock","CHN") = 2.1592938899993896 ;
esubd("c_FoodProc","CHN") = 2.4785866737365723 ;
esubd("c_Energy","CHN") = 3.818507194519043 ;
esubd("c_Textiles","CHN") = 3.779940128326416 ;
esubd("c_Chem","CHN") = 3.299999952316284 ;
esubd("c_Manuf","CHN") = 3.5473241806030273 ;
esubd("c_ForestFish","CHN") = 1.8294216394424438 ;
esubd("c_Svces","CHN") = 1.909135103225708 ;
esubd("c_Rice","JPN") = 3.869595766067505 ;
esubd("c_Crops","JPN") = 2.2441093921661377 ;
esubd("c_Livestock","JPN") = 2.1592938899993896 ;
esubd("c_FoodProc","JPN") = 2.4785866737365723 ;
esubd("c_Energy","JPN") = 3.818507194519043 ;
esubd("c_Textiles","JPN") = 3.779940128326416 ;
esubd("c_Chem","JPN") = 3.299999952316284 ;
esubd("c_Manuf","JPN") = 3.5473241806030273 ;
esubd("c_ForestFish","JPN") = 1.8294216394424438 ;
esubd("c_Svces","JPN") = 1.909135103225708 ;
esubd("c_Rice","IND") = 3.869595766067505 ;
esubd("c_Crops","IND") = 2.2441093921661377 ;
esubd("c_Livestock","IND") = 2.1592938899993896 ;
esubd("c_FoodProc","IND") = 2.4785866737365723 ;
esubd("c_Energy","IND") = 3.818507194519043 ;
esubd("c_Textiles","IND") = 3.779940128326416 ;
esubd("c_Chem","IND") = 3.299999952316284 ;
esubd("c_Manuf","IND") = 3.5473241806030273 ;
esubd("c_ForestFish","IND") = 1.8294216394424438 ;
esubd("c_Svces","IND") = 1.909135103225708 ;
esubd("c_Rice","SSA") = 3.869595766067505 ;
esubd("c_Crops","SSA") = 2.2441093921661377 ;
esubd("c_Livestock","SSA") = 2.1592938899993896 ;
esubd("c_FoodProc","SSA") = 2.4785866737365723 ;
esubd("c_Energy","SSA") = 3.818507194519043 ;
esubd("c_Textiles","SSA") = 3.779940128326416 ;
esubd("c_Chem","SSA") = 3.299999952316284 ;
esubd("c_Manuf","SSA") = 3.5473241806030273 ;
esubd("c_ForestFish","SSA") = 1.8294216394424438 ;
esubd("c_Svces","SSA") = 1.909135103225708 ;
esubd("c_Rice","ROW") = 3.869595766067505 ;
esubd("c_Crops","ROW") = 2.2441093921661377 ;
esubd("c_Livestock","ROW") = 2.1592938899993896 ;
esubd("c_FoodProc","ROW") = 2.4785866737365723 ;
esubd("c_Energy","ROW") = 3.818507194519043 ;
esubd("c_Textiles","ROW") = 3.779940128326416 ;
esubd("c_Chem","ROW") = 3.299999952316284 ;
esubd("c_Manuf","ROW") = 3.5473241806030273 ;
esubd("c_ForestFish","ROW") = 1.8294216394424438 ;
esubd("c_Svces","ROW") = 1.909135103225708 ;
parameter esubm(*,*) ;
esubm("c_Rice","USA") = 5.354156017303467 ;
esubm("c_Crops","USA") = 4.8155293464660645 ;
esubm("c_Livestock","USA") = 4.020486354827881 ;
esubm("c_FoodProc","USA") = 4.813882350921631 ;
esubm("c_Energy","USA") = 11.496824264526367 ;
esubm("c_Textiles","USA") = 7.5890583992004395 ;
esubm("c_Chem","USA") = 6.599999904632568 ;
esubm("c_Manuf","USA") = 7.461285591125488 ;
esubm("c_ForestFish","USA") = 3.6039652824401855 ;
esubm("c_Svces","USA") = 3.800415277481079 ;
esubm("c_Rice","EU_28") = 5.354156017303467 ;
esubm("c_Crops","EU_28") = 4.8155293464660645 ;
esubm("c_Livestock","EU_28") = 4.020486354827881 ;
esubm("c_FoodProc","EU_28") = 4.813882350921631 ;
esubm("c_Energy","EU_28") = 11.496824264526367 ;
esubm("c_Textiles","EU_28") = 7.5890583992004395 ;
esubm("c_Chem","EU_28") = 6.599999904632568 ;
esubm("c_Manuf","EU_28") = 7.461285591125488 ;
esubm("c_ForestFish","EU_28") = 3.6039652824401855 ;
esubm("c_Svces","EU_28") = 3.800415277481079 ;
esubm("c_Rice","CHN") = 5.354156017303467 ;
esubm("c_Crops","CHN") = 4.8155293464660645 ;
esubm("c_Livestock","CHN") = 4.020486354827881 ;
esubm("c_FoodProc","CHN") = 4.813882350921631 ;
esubm("c_Energy","CHN") = 11.496824264526367 ;
esubm("c_Textiles","CHN") = 7.5890583992004395 ;
esubm("c_Chem","CHN") = 6.599999904632568 ;
esubm("c_Manuf","CHN") = 7.461285591125488 ;
esubm("c_ForestFish","CHN") = 3.6039652824401855 ;
esubm("c_Svces","CHN") = 3.800415277481079 ;
esubm("c_Rice","JPN") = 5.354156017303467 ;
esubm("c_Crops","JPN") = 4.8155293464660645 ;
esubm("c_Livestock","JPN") = 4.020486354827881 ;
esubm("c_FoodProc","JPN") = 4.813882350921631 ;
esubm("c_Energy","JPN") = 11.496824264526367 ;
esubm("c_Textiles","JPN") = 7.5890583992004395 ;
esubm("c_Chem","JPN") = 6.599999904632568 ;
esubm("c_Manuf","JPN") = 7.461285591125488 ;
esubm("c_ForestFish","JPN") = 3.6039652824401855 ;
esubm("c_Svces","JPN") = 3.800415277481079 ;
esubm("c_Rice","IND") = 5.354156017303467 ;
esubm("c_Crops","IND") = 4.8155293464660645 ;
esubm("c_Livestock","IND") = 4.020486354827881 ;
esubm("c_FoodProc","IND") = 4.813882350921631 ;
esubm("c_Energy","IND") = 11.496824264526367 ;
esubm("c_Textiles","IND") = 7.5890583992004395 ;
esubm("c_Chem","IND") = 6.599999904632568 ;
esubm("c_Manuf","IND") = 7.461285591125488 ;
esubm("c_ForestFish","IND") = 3.6039652824401855 ;
esubm("c_Svces","IND") = 3.800415277481079 ;
esubm("c_Rice","SSA") = 5.354156017303467 ;
esubm("c_Crops","SSA") = 4.8155293464660645 ;
esubm("c_Livestock","SSA") = 4.020486354827881 ;
esubm("c_FoodProc","SSA") = 4.813882350921631 ;
esubm("c_Energy","SSA") = 11.496824264526367 ;
esubm("c_Textiles","SSA") = 7.5890583992004395 ;
esubm("c_Chem","SSA") = 6.599999904632568 ;
esubm("c_Manuf","SSA") = 7.461285591125488 ;
esubm("c_ForestFish","SSA") = 3.6039652824401855 ;
esubm("c_Svces","SSA") = 3.800415277481079 ;
esubm("c_Rice","ROW") = 5.354156017303467 ;
esubm("c_Crops","ROW") = 4.8155293464660645 ;
esubm("c_Livestock","ROW") = 4.020486354827881 ;
esubm("c_FoodProc","ROW") = 4.813882350921631 ;
esubm("c_Energy","ROW") = 11.496824264526367 ;
esubm("c_Textiles","ROW") = 7.5890583992004395 ;
esubm("c_Chem","ROW") = 6.599999904632568 ;
esubm("c_Manuf","ROW") = 7.461285591125488 ;
esubm("c_ForestFish","ROW") = 3.6039652824401855 ;
esubm("c_Svces","ROW") = 3.800415277481079 ;
parameter esubt(*,*) ;
parameter esubc(*,*) ;
parameter esubva(*,*) ;
esubva("a_Rice","USA") = 0.46640947461128235 ;
esubva("a_Crops","USA") = 0.25920990109443665 ;
esubva("a_Livestock","USA") = 0.25920990109443665 ;
esubva("a_FoodProc","USA") = 1.1200000047683716 ;
esubva("a_Energy","USA") = 0.7166335582733154 ;
esubva("a_Textiles","USA") = 1.2599999904632568 ;
esubva("a_Chem","USA") = 1.2599999904632568 ;
esubva("a_Manuf","USA") = 1.1785955429077148 ;
esubva("a_ForestFish","USA") = 0.20000000298023224 ;
esubva("a_Svces","USA") = 1.37246572971344 ;
esubva("a_Rice","EU_28") = 0.46640947461128235 ;
esubva("a_Crops","EU_28") = 0.25920990109443665 ;
esubva("a_Livestock","EU_28") = 0.25920990109443665 ;
esubva("a_FoodProc","EU_28") = 1.1200000047683716 ;
esubva("a_Energy","EU_28") = 0.7166335582733154 ;
esubva("a_Textiles","EU_28") = 1.2599999904632568 ;
esubva("a_Chem","EU_28") = 1.2599999904632568 ;
esubva("a_Manuf","EU_28") = 1.1785955429077148 ;
esubva("a_ForestFish","EU_28") = 0.20000000298023224 ;
esubva("a_Svces","EU_28") = 1.37246572971344 ;
esubva("a_Rice","CHN") = 0.46640947461128235 ;
esubva("a_Crops","CHN") = 0.25920990109443665 ;
esubva("a_Livestock","CHN") = 0.25920990109443665 ;
esubva("a_FoodProc","CHN") = 1.1200000047683716 ;
esubva("a_Energy","CHN") = 0.7166335582733154 ;
esubva("a_Textiles","CHN") = 1.2599999904632568 ;
esubva("a_Chem","CHN") = 1.2599999904632568 ;
esubva("a_Manuf","CHN") = 1.1785955429077148 ;
esubva("a_ForestFish","CHN") = 0.20000000298023224 ;
esubva("a_Svces","CHN") = 1.37246572971344 ;
esubva("a_Rice","JPN") = 0.46640947461128235 ;
esubva("a_Crops","JPN") = 0.25920990109443665 ;
esubva("a_Livestock","JPN") = 0.25920990109443665 ;
esubva("a_FoodProc","JPN") = 1.1200000047683716 ;
esubva("a_Energy","JPN") = 0.7166335582733154 ;
esubva("a_Textiles","JPN") = 1.2599999904632568 ;
esubva("a_Chem","JPN") = 1.2599999904632568 ;
esubva("a_Manuf","JPN") = 1.1785955429077148 ;
esubva("a_ForestFish","JPN") = 0.20000000298023224 ;
esubva("a_Svces","JPN") = 1.37246572971344 ;
esubva("a_Rice","IND") = 0.46640947461128235 ;
esubva("a_Crops","IND") = 0.25920990109443665 ;
esubva("a_Livestock","IND") = 0.25920990109443665 ;
esubva("a_FoodProc","IND") = 1.1200000047683716 ;
esubva("a_Energy","IND") = 0.7166335582733154 ;
esubva("a_Textiles","IND") = 1.2599999904632568 ;
esubva("a_Chem","IND") = 1.2599999904632568 ;
esubva("a_Manuf","IND") = 1.1785955429077148 ;
esubva("a_ForestFish","IND") = 0.20000000298023224 ;
esubva("a_Svces","IND") = 1.37246572971344 ;
esubva("a_Rice","SSA") = 0.46640947461128235 ;
esubva("a_Crops","SSA") = 0.25920990109443665 ;
esubva("a_Livestock","SSA") = 0.25920990109443665 ;
esubva("a_FoodProc","SSA") = 1.1200000047683716 ;
esubva("a_Energy","SSA") = 0.7166335582733154 ;
esubva("a_Textiles","SSA") = 1.2599999904632568 ;
esubva("a_Chem","SSA") = 1.2599999904632568 ;
esubva("a_Manuf","SSA") = 1.1785955429077148 ;
esubva("a_ForestFish","SSA") = 0.20000000298023224 ;
esubva("a_Svces","SSA") = 1.37246572971344 ;
esubva("a_Rice","ROW") = 0.46640947461128235 ;
esubva("a_Crops","ROW") = 0.25920990109443665 ;
esubva("a_Livestock","ROW") = 0.25920990109443665 ;
esubva("a_FoodProc","ROW") = 1.1200000047683716 ;
esubva("a_Energy","ROW") = 0.7166335582733154 ;
esubva("a_Textiles","ROW") = 1.2599999904632568 ;
esubva("a_Chem","ROW") = 1.2599999904632568 ;
esubva("a_Manuf","ROW") = 1.1785955429077148 ;
esubva("a_ForestFish","ROW") = 0.20000000298023224 ;
esubva("a_Svces","ROW") = 1.37246572971344 ;
parameter esubq(*,*) ;
parameter esubg(*) ;
esubg("USA") = 1.0 ;
esubg("EU_28") = 1.0 ;
esubg("CHN") = 1.0 ;
esubg("JPN") = 1.0 ;
esubg("IND") = 1.0 ;
esubg("SSA") = 1.0 ;
esubg("ROW") = 1.0 ;
parameter esubi(*) ;
parameter esubs(*) ;
esubs("c_Svces") = 1.0 ;
parameter etraq(*,*) ;
etraq("a_Rice","USA") = -5.0 ;
etraq("a_Crops","USA") = -5.0 ;
etraq("a_Livestock","USA") = -5.0 ;
etraq("a_FoodProc","USA") = -5.0 ;
etraq("a_Energy","USA") = -5.0 ;
etraq("a_Textiles","USA") = -5.0 ;
etraq("a_Chem","USA") = -5.0 ;
etraq("a_Manuf","USA") = -5.0 ;
etraq("a_ForestFish","USA") = -5.0 ;
etraq("a_Svces","USA") = -5.0 ;
etraq("a_Rice","EU_28") = -5.0 ;
etraq("a_Crops","EU_28") = -5.0 ;
etraq("a_Livestock","EU_28") = -5.0 ;
etraq("a_FoodProc","EU_28") = -5.0 ;
etraq("a_Energy","EU_28") = -5.0 ;
etraq("a_Textiles","EU_28") = -5.0 ;
etraq("a_Chem","EU_28") = -5.0 ;
etraq("a_Manuf","EU_28") = -5.0 ;
etraq("a_ForestFish","EU_28") = -5.0 ;
etraq("a_Svces","EU_28") = -5.0 ;
etraq("a_Rice","CHN") = -4.999999523162842 ;
etraq("a_Crops","CHN") = -5.0 ;
etraq("a_Livestock","CHN") = -5.0 ;
etraq("a_FoodProc","CHN") = -5.0 ;
etraq("a_Energy","CHN") = -5.0 ;
etraq("a_Textiles","CHN") = -5.0 ;
etraq("a_Chem","CHN") = -5.0 ;
etraq("a_Manuf","CHN") = -5.0 ;
etraq("a_ForestFish","CHN") = -5.0 ;
etraq("a_Svces","CHN") = -5.0 ;
etraq("a_Rice","JPN") = -5.0 ;
etraq("a_Crops","JPN") = -5.0 ;
etraq("a_Livestock","JPN") = -5.0 ;
etraq("a_FoodProc","JPN") = -5.0 ;
etraq("a_Energy","JPN") = -5.0 ;
etraq("a_Textiles","JPN") = -5.0 ;
etraq("a_Chem","JPN") = -5.0 ;
etraq("a_Manuf","JPN") = -5.0 ;
etraq("a_ForestFish","JPN") = -5.0 ;
etraq("a_Svces","JPN") = -5.0 ;
etraq("a_Rice","IND") = -5.0 ;
etraq("a_Crops","IND") = -5.0 ;
etraq("a_Livestock","IND") = -5.0 ;
etraq("a_FoodProc","IND") = -5.0 ;
etraq("a_Energy","IND") = -5.0 ;
etraq("a_Textiles","IND") = -5.0 ;
etraq("a_Chem","IND") = -5.0 ;
etraq("a_Manuf","IND") = -5.0 ;
etraq("a_ForestFish","IND") = -5.0 ;
etraq("a_Svces","IND") = -5.0 ;
etraq("a_Rice","SSA") = -5.0 ;
etraq("a_Crops","SSA") = -5.0 ;
etraq("a_Livestock","SSA") = -5.0 ;
etraq("a_FoodProc","SSA") = -5.0 ;
etraq("a_Energy","SSA") = -5.0 ;
etraq("a_Textiles","SSA") = -5.0 ;
etraq("a_Chem","SSA") = -5.0 ;
etraq("a_Manuf","SSA") = -5.0 ;
etraq("a_ForestFish","SSA") = -5.0 ;
etraq("a_Svces","SSA") = -5.0 ;
etraq("a_Rice","ROW") = -5.0 ;
etraq("a_Crops","ROW") = -5.0 ;
etraq("a_Livestock","ROW") = -5.0 ;
etraq("a_FoodProc","ROW") = -5.0 ;
etraq("a_Energy","ROW") = -5.0 ;
etraq("a_Textiles","ROW") = -5.0 ;
etraq("a_Chem","ROW") = -5.0 ;
etraq("a_Manuf","ROW") = -5.0 ;
etraq("a_ForestFish","ROW") = -5.0 ;
etraq("a_Svces","ROW") = -5.0 ;
parameter etrae(*,*) ;
etrae("Land","USA") = -1.0 ;
etrae("NatRes","USA") = -0.0010000000474974513 ;
etrae("Land","EU_28") = -1.0 ;
etrae("NatRes","EU_28") = -0.0010000000474974513 ;
etrae("Land","CHN") = -1.0 ;
etrae("NatRes","CHN") = -0.0010000000474974513 ;
etrae("Land","JPN") = -1.0 ;
etrae("NatRes","JPN") = -0.0010000000474974513 ;
etrae("Land","IND") = -1.0 ;
etrae("NatRes","IND") = -0.0010000000474974513 ;
etrae("Land","SSA") = -1.0 ;
etrae("NatRes","SSA") = -0.0010000000474974513 ;
etrae("Land","ROW") = -1.0 ;
etrae("NatRes","ROW") = -0.0010000000474974513 ;
parameter incpar(*,*) ;
incpar("c_Rice","USA") = 9.999999974752427e-07 ;
incpar("c_Crops","USA") = 0.04263194650411606 ;
incpar("c_Livestock","USA") = 0.503583550453186 ;
incpar("c_FoodProc","USA") = 0.41224244236946106 ;
incpar("c_Energy","USA") = 0.9133253693580627 ;
incpar("c_Textiles","USA") = 0.49033960700035095 ;
incpar("c_Chem","USA") = 0.9594091773033142 ;
incpar("c_Manuf","USA") = 0.9517101645469666 ;
incpar("c_ForestFish","USA") = 0.5593986511230469 ;
incpar("c_Svces","USA") = 1.0610698461532593 ;
incpar("c_Rice","EU_28") = 0.0021270825527608395 ;
incpar("c_Crops","EU_28") = 0.006428936030715704 ;
incpar("c_Livestock","EU_28") = 0.554161012172699 ;
incpar("c_FoodProc","EU_28") = 0.46565231680870056 ;
incpar("c_Energy","EU_28") = 0.9933156967163086 ;
incpar("c_Textiles","EU_28") = 0.5438744425773621 ;
incpar("c_Chem","EU_28") = 1.0404939651489258 ;
incpar("c_Manuf","EU_28") = 1.0333328247070312 ;
incpar("c_ForestFish","EU_28") = 0.7426393032073975 ;
incpar("c_Svces","EU_28") = 1.1245760917663574 ;
incpar("c_Rice","CHN") = 0.12024746090173721 ;
incpar("c_Crops","CHN") = 0.12096762657165527 ;
incpar("c_Livestock","CHN") = 0.46286261081695557 ;
incpar("c_FoodProc","CHN") = 0.3850696086883545 ;
incpar("c_Energy","CHN") = 1.0225056409835815 ;
incpar("c_Textiles","CHN") = 0.4493468403816223 ;
incpar("c_Chem","CHN") = 1.0553573369979858 ;
incpar("c_Manuf","CHN") = 1.0387001037597656 ;
incpar("c_ForestFish","CHN") = 0.4786314070224762 ;
incpar("c_Svces","CHN") = 1.3661208152770996 ;
incpar("c_Rice","JPN") = 9.999999974752427e-07 ;
incpar("c_Crops","JPN") = 0.003530829446390271 ;
incpar("c_Livestock","JPN") = 0.5488638281822205 ;
incpar("c_FoodProc","JPN") = 0.4367224872112274 ;
incpar("c_Energy","JPN") = 0.9625540375709534 ;
incpar("c_Textiles","JPN") = 0.5349581241607666 ;
incpar("c_Chem","JPN") = 1.009908676147461 ;
incpar("c_Manuf","JPN") = 1.004759430885315 ;
incpar("c_ForestFish","JPN") = 0.7029562592506409 ;
incpar("c_Svces","JPN") = 1.1006367206573486 ;
incpar("c_Rice","IND") = 0.4121294617652893 ;
incpar("c_Crops","IND") = 0.4131002128124237 ;
incpar("c_Livestock","IND") = 0.6080049276351929 ;
incpar("c_FoodProc","IND") = 0.5743266940116882 ;
incpar("c_Energy","IND") = 0.9281172752380371 ;
incpar("c_Textiles","IND") = 0.6007944941520691 ;
incpar("c_Chem","IND") = 0.8242521286010742 ;
incpar("c_Manuf","IND") = 0.826797366142273 ;
incpar("c_ForestFish","IND") = 0.6821090579032898 ;
incpar("c_Svces","IND") = 1.310168743133545 ;
incpar("c_Rice","SSA") = 0.5564925670623779 ;
incpar("c_Crops","SSA") = 0.589530885219574 ;
incpar("c_Livestock","SSA") = 0.8615323305130005 ;
incpar("c_FoodProc","SSA") = 0.6866018176078796 ;
incpar("c_Energy","SSA") = 1.0403550863265991 ;
incpar("c_Textiles","SSA") = 0.7493164539337158 ;
incpar("c_Chem","SSA") = 0.9298102259635925 ;
incpar("c_Manuf","SSA") = 0.944404125213623 ;
incpar("c_ForestFish","SSA") = 0.7866491675376892 ;
incpar("c_Svces","SSA") = 1.3812965154647827 ;
incpar("c_Rice","ROW") = 0.305558443069458 ;
incpar("c_Crops","ROW") = 0.2211325615644455 ;
incpar("c_Livestock","ROW") = 0.576562762260437 ;
incpar("c_FoodProc","ROW") = 0.4553508162498474 ;
incpar("c_Energy","ROW") = 1.0014123916625977 ;
incpar("c_Textiles","ROW") = 0.529085099697113 ;
incpar("c_Chem","ROW") = 1.03163480758667 ;
incpar("c_Manuf","ROW") = 1.0184593200683594 ;
incpar("c_ForestFish","ROW") = 0.6687270998954773 ;
incpar("c_Svces","ROW") = 1.1788530349731445 ;
parameter subpar(*,*) ;
subpar("c_Rice","USA") = 0.9999990463256836 ;
subpar("c_Crops","USA") = 0.9272732734680176 ;
subpar("c_Livestock","USA") = 0.15986120700836182 ;
subpar("c_FoodProc","USA") = 0.1924026757478714 ;
subpar("c_Energy","USA") = 0.09387533366680145 ;
subpar("c_Textiles","USA") = 0.16351157426834106 ;
subpar("c_Chem","USA") = 0.08916409313678741 ;
subpar("c_Manuf","USA") = 0.0899338349699974 ;
subpar("c_ForestFish","USA") = 0.1512012928724289 ;
subpar("c_Svces","USA") = 0.08792939782142639 ;
subpar("c_Rice","EU_28") = 0.9949424862861633 ;
subpar("c_Crops","EU_28") = 0.9904057383537292 ;
subpar("c_Livestock","EU_28") = 0.3749217092990875 ;
subpar("c_FoodProc","EU_28") = 0.3960759937763214 ;
subpar("c_Energy","EU_28") = 0.23175857961177826 ;
subpar("c_Textiles","EU_28") = 0.34072911739349365 ;
subpar("c_Chem","EU_28") = 0.228020578622818 ;
subpar("c_Manuf","EU_28") = 0.21725840866565704 ;
subpar("c_ForestFish","EU_28") = 0.3289707899093628 ;
subpar("c_Svces","EU_28") = 0.2063865214586258 ;
subpar("c_Rice","CHN") = 0.8740237951278687 ;
subpar("c_Crops","CHN") = 0.8735871315002441 ;
subpar("c_Livestock","CHN") = 0.6714162230491638 ;
subpar("c_FoodProc","CHN") = 0.716997504234314 ;
subpar("c_Energy","CHN") = 0.4935285449028015 ;
subpar("c_Textiles","CHN") = 0.6744819283485413 ;
subpar("c_Chem","CHN") = 0.4869003891944885 ;
subpar("c_Manuf","CHN") = 0.49070826172828674 ;
subpar("c_ForestFish","CHN") = 0.6665375828742981 ;
subpar("c_Svces","CHN") = 0.4421200156211853 ;
subpar("c_Rice","JPN") = 0.9999990463256836 ;
subpar("c_Crops","JPN") = 0.9955404996871948 ;
subpar("c_Livestock","JPN") = 0.3188154995441437 ;
subpar("c_FoodProc","JPN") = 0.37477120757102966 ;
subpar("c_Energy","JPN") = 0.21361032128334045 ;
subpar("c_Textiles","JPN") = 0.3242999315261841 ;
subpar("c_Chem","JPN") = 0.20494942367076874 ;
subpar("c_Manuf","JPN") = 0.20586465299129486 ;
subpar("c_ForestFish","JPN") = 0.2807576060295105 ;
subpar("c_Svces","JPN") = 0.19658617675304413 ;
subpar("c_Rice","IND") = 0.8446068167686462 ;
subpar("c_Crops","IND") = 0.8442919254302979 ;
subpar("c_Livestock","IND") = 0.7840914726257324 ;
subpar("c_FoodProc","IND") = 0.7923504114151001 ;
subpar("c_Energy","IND") = 0.7051666378974915 ;
subpar("c_Textiles","IND") = 0.7834095358848572 ;
subpar("c_Chem","IND") = 0.7276021838188171 ;
subpar("c_Manuf","IND") = 0.7270501852035522 ;
subpar("c_ForestFish","IND") = 0.764779269695282 ;
subpar("c_Svces","IND") = 0.6477729678153992 ;
subpar("c_Rice","SSA") = 0.8567221760749817 ;
subpar("c_Crops","SSA") = 0.854276180267334 ;
subpar("c_Livestock","SSA") = 0.7964909672737122 ;
subpar("c_FoodProc","SSA") = 0.797080397605896 ;
subpar("c_Energy","SSA") = 0.6626578569412231 ;
subpar("c_Textiles","SSA") = 0.7827576398849487 ;
subpar("c_Chem","SSA") = 0.7167853116989136 ;
subpar("c_Manuf","SSA") = 0.6940402388572693 ;
subpar("c_ForestFish","SSA") = 0.7846815586090088 ;
subpar("c_Svces","SSA") = 0.6090341806411743 ;
subpar("c_Rice","ROW") = 0.8840133547782898 ;
subpar("c_Crops","ROW") = 0.9007891416549683 ;
subpar("c_Livestock","ROW") = 0.6884165406227112 ;
subpar("c_FoodProc","ROW") = 0.6416326761245728 ;
subpar("c_Energy","ROW") = 0.44322293996810913 ;
subpar("c_Textiles","ROW") = 0.5901503562927246 ;
subpar("c_Chem","ROW") = 0.4331519901752472 ;
subpar("c_Manuf","ROW") = 0.39168858528137207 ;
subpar("c_ForestFish","ROW") = 0.6729961633682251 ;
subpar("c_Svces","ROW") = 0.35981184244155884 ;
parameter rorFlex0(*) ;
rorFlex0("USA") = 10.0 ;
rorFlex0("EU_28") = 10.0 ;
rorFlex0("CHN") = 10.0 ;
rorFlex0("JPN") = 10.0 ;
rorFlex0("IND") = 10.0 ;
rorFlex0("SSA") = 10.0 ;
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
