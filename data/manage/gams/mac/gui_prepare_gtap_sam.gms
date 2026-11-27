* --- What the GUI writes to fromGuiToRun.gms for the task "Prepare GTAP SAM"
*     (gtapSamV6.gms), with the GUI defaults from gui/manage.xml, for Ghana.
$setglobal task Prepare GTAP SAM
$setglobal JAVA on
* the GUI passes the model directory as an absolute path
$setglobal curDir %gams.workdir%
$setglobal datDir ./dat
$setglobal scrdir %gams.scrdir%
$setglobal Version gtapData9
$setglobal calcGblCostShares off
$setglobal regName GHA
$setglobal minProdValue 3
$setglobal minCostShare 4
$setglobal useSamCorrFile off
$setglobal samCorrFile empty
$setglobal useAez on
$setglobal buildGender off
$setglobal ctlRais off
$setglobal otpEle off
$setglobal careEcon off
$setglobal waterSplit off
$setglobal debtStckD 0.5
$setglobal debtIntD 5
$setglobal debtPrinciple 2
$setglobal newDebtToGovConsumption 0.05
$setglobal shareForeignLab 0.1
$setglobal shareForeignCap 0.1
* CONOPT4 (GUI default) exceeds the license on this Mac: IPOPT
$setglobal QCPSolver IPOPT
