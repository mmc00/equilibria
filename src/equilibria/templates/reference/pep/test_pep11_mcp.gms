$TITLE Test PEP11 MCP on equilibria reference template

$setglobal SKIP_RESULTS 1
$setglobal SKIP_INITIAL_SOLVE 1
$include PEP-1-1_v2_1_modular.gms

OPTION MCP = PATH;
PEP11.HOLDFIXED = 1;
SOLVE PEP11 USING MCP;

display PEP11.modelstat, PEP11.solvestat, PEP11.numinfes;
