********************************************************************************
$ontext

   CGEBOX project

   GAMS file : Altertax.gms

   @purpose  : Define model configuration for altertax
   @author   : W.Britz
   @date     : 20.10.16
   @since    :
   @refDoc   :
   @seeAlso  :
   @calledBy : com_.gms

$offtext
********************************************************************************

  SET s_META /
   'Simulation'.'Dynamics' 'Comparative static [default]'
   'Simulation'.'Regional coverage' 'Global model [default]'
   'Simulation'.'hide_Modules'
   'Simulation'.'Non default parameters' 'altertax'
   'Simulation'.'hide_DemandSystem' 'CD'
  /;
*
* --- Comparativ static, global model
*
  $$setglobal dynMode Comparative static
  $$setglobal regMode Global model
*
* --- switch OFF additional modules
*
  $$setglobal modulesGTAP_AEZ OFF
  $$setglobal modulesGTAP_AGR OFF
  $$setglobal modulesGTAP_E   OFF
  $$SETGLOBAL modulesLabor_nest OFF
  $$SETGLOBAL modulesCapSkLab_nest OFF
  $$SETGLOBAL modulesCO2_Emissions OFF
  $$SETGLOBAL modulesGTAP_MELITZ OFF
  $$SETGLOBAL modulesRegional_household OFF
  $$SETGLOBAL modulesAggregate_firm_demand OFF
  $$SETGLOBAL modulesAggregate_Armington OFF
  $$SETGLOBAL modulesCapVintages OFF
  $$SETGLOBAL modulesmyGTAP OFF
  $$SETGLOBAL DemSystem CD
*
* ---closures
*
  $$SETGLOBAL invSav Global equal returns to capital
  $$SETGLOBAL govClosure Tax income
  $$SETGLOBAL FinalConsumptionClosure Spending
  $$SETGLOBAL RegionalNumeraire Exchange Rate

  $$SETGLOBAL subsPrices on

*
* -- no fixed factor prices
*
  option kill=fpf_sel;
*
* -- all factors are fully mobile
*
  set fmm(*);
  execute_load "%resDir%/build/%dataset%.gdx" fmm=f;
  fm_sel(fmm) = YES;
*
* --- output in altertax format
*
  $$SETGLOBAL outputtypesAlterTax ON
  $$SETGLOBAL SAMtoGDX ON
*
* --- use parameters where all substiution elasticities are unity (CD)
*
  $$setglobal NonDefaultParameters on
  $$SETGLOBAL parameters Parameter_altertax
  $$SETGLOBAL parameters_underScores Parameter_altertax
  $$SETGLOBAL parameters_withoutPath Parameter_altertax

  $$if not errorfree $abort Compilation error after file: %system.fn%
  if ( execerror, abort "Run-Time error in file: %system.fn%, line: %system.incline%");
