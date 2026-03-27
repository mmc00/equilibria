$TITLE Compare default PEP11 shocks under CNS and MCP

$if not set SHOCK $setglobal SHOCK export_tax
$if not set SOLVE_TYPE $setglobal SOLVE_TYPE MCP
$setglobal SKIP_RESULTS 1
$setglobal SKIP_INITIAL_SOLVE 1

$include PEP-1-1_v2_1.gms

* Reset the hardcoded example shock from the reference script before applying a chosen one.
PWM.fx(i)  = PWMO(i);
ttix.fx(i) = ttixO(i);
G.fx       = GO;

$ifthenI "%SHOCK%" == "export_tax"
  ttix.fx(i) = ttixO(i)*0.75;
$elseifI "%SHOCK%" == "gov_spending"
  G.fx = GO*1.2;
$elseifI "%SHOCK%" == "import_price_agr"
  PWM.fx('agr') = PWMO('agr')*1.25;
$else
$abort Unknown SHOCK value. Use export_tax, gov_spending, or import_price_agr.
$endif

$ifthenI "%SOLVE_TYPE%" == "CNS"
  OPTION CNS = CONOPT4;
  PEP11.HOLDFIXED = 1;
  SOLVE PEP11 USING CNS;
$elseifI "%SOLVE_TYPE%" == "MCP"
  OPTION MCP = PATH;
  PEP11.HOLDFIXED = 1;
  SOLVE PEP11 USING MCP;
$else
$abort Unknown SOLVE_TYPE value. Use CNS or MCP.
$endif

PARAMETER SUMMARY(*);
SUMMARY('GDP_MP')    = GDP_MP.l;
SUMMARY('GDP_BP')    = GDP_BP.l;
SUMMARY('PIXCON')    = PIXCON.l;
SUMMARY('SG')        = SG.l;
SUMMARY('G')         = G.l;
SUMMARY('YG')        = YG.l;
SUMMARY('SROW')      = SROW.l;
SUMMARY('CAB')       = CAB.l;
SUMMARY('LEON')      = LEON.l;
SUMMARY('EXD_TOTAL') = SUM(i, EXD.l(i));
SUMMARY('IM_TOTAL')  = SUM(i, IM.l(i));

DISPLAY SUMMARY, PEP11.modelstat, PEP11.solvestat, PEP11.numinfes;
