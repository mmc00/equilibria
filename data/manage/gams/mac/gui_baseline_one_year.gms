* --- What the GUI writes to fromGuiToRun.gms for the task "Baseline"
*     (run.gms), with the GUI defaults from gui/manage.xml, for Ghana:
*     base year calibration plus one solved year, no shock (none.inc),
*     dynamics, damages and links off.
*     %lastYear% is filled in by run_gha_one_year.sh (base year + 1).
$setglobal task Baseline
$setglobal JAVA on
* the GUI passes the model directory as an absolute path
$setglobal curDir %gams.workdir%
$setglobal scrdir %gams.scrdir%
* GGIG names the set documentation GDX (inc/<name>.gdx, ignored by git as gams*.gdx)
$setglobal gamsDocName gamsDocSets
$setglobal RegGtapName GHA
$setglobal bridgeFile BaseBridge_GHA_GTAP_9
$setglobal lastYear @LASTYEAR@
$setglobal lastSimYear @LASTYEAR@
$setglobal SimName
$setglobal SimNamePostFix
$setglobal ShockFileType GAMS
$setglobal shk_file none
$setglobal shk_file_xslx
$setglobal add_shk_file empty
$setglobal usedAsBaU off
* --- options
$setglobal CCDamages off
$setglobal InvestLink off
$setglobal ifAirPollDmg off
$setglobal FiscalResp off
$setglobal DebtGDPTgtUp 100
$setglobal DebtGDPTgtLo 0
$setglobal relExpInc 100
$setglobal shortRunShock off
$setglobal GDPCompScen bas
$setglobal GDPCompGovCons no change real
$setglobal GDPCompGovInv no change real
$setglobal GDPCompNetExports closure rule
$setglobal GDPCompSavf closure rule
$setglobal sectorShareTargets off
* --- baseline tab
$setglobal BGCheck off
$setglobal benchmarkCheck on
$setglobal skipGTAPEmis off
$setglobal ForwardLooking off
$setglobal futureYears 4
$setglobal expWeight 0.1
$setglobal labVintages off
$setglobal stickyWage 0
$setglobal vintages off
$setglobal updateOldVintage off
$setglobal endoTech off
$setglobal CapAccClosure Fixed as nominal GDP Share
$setglobal rowTransferDriver Driven by population growth
$setglobal sectSpecInv off
$setglobal oldCapInvShare 0.025
$setglobal IncomeDepCDE off
$setglobal chicFacCap off
$setglobal DiffLabProd off
$setglobal LandProdAsLabor 0
$setglobal NatresProdAsLabor 0
$setglobal capProdAsLabor 0
* --- climate damages (off), energy, GHG
$setglobal tempIncreaseBaseYear 0.5
$setglobal tempIncreaseSimYear 2.0
$setglobal useRosonSartoriLabProd off
$setglobal useRosonSartoriLandStock off
$setglobal useRosonSartoriEnergyDemand off
$setglobal useRosonSartoriTourism off
$setglobal useRosonSartoriCropYields off
$setglobal useRosonSartoriHealth off
$setglobal GHGMitigation off
$setglobal fixPowerMix off
$setglobal CCDamagesIEC off
$setglobal CCEnergy off
$setglobal PowerScen bas
$setglobal fixElecPrice off
$setglobal EnergyScen BAU
$setglobal enerProjStartYear 2018
$setglobal EnergyDemandProjections on
$setglobal EnergyActProjections on
$setglobal EnergyEffProjections on
$setglobal WorldMarketPriceProjections off
$setglobal GHGMitigationScenario BAU
$setglobal considerCarbon off
$setglobal MACCS off
$setglobal dirTaxRecl 0.1
$setglobal factTaxRecl 0.1
$setglobal salesTaxRecl 0.1
$setglobal allTaxRecl 0.1
$setglobal transfersRecl 0.1
$setglobal IECScen CG
$setglobal IECClim Wet/Warm mean
$setglobal IECClimType Wet/Warm
$setglobal IECChnls
$setglobal investModules
$setglobal investFeedbacks
* --- output
$setglobal outputGUI on
$setglobal reducedOutput off
$setglobal repFreq 1
$setglobal outputSAM off
$setglobal outputGDX off
$setglobal outputfrEnd off
$setglobal userOutput off
$setglobal add_output_file empty
* --- GAMS tab
$setglobal iniAddVars off
$setglobal iniXfac off
$setglobal solveType @SOLVETYPE@
$setglobal MCPSolver @MCPSOLVER@
$setglobal scaleModel on
$setglobal scaleSAM on
$setglobal autoScaleBase 5
$setglobal optfile 2
$setglobal resLim 300
$setglobal reslimFirstSolve 5
$setglobal maxIter 6
