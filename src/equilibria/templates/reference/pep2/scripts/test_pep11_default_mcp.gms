$TITLE Test default PEP11 benchmark as MCP

$setglobal SKIP_RESULTS 1
$setglobal SKIP_INITIAL_SOLVE 1
$include PEP-1-1_v2_1.gms

OPTION MCP = PATH;
PEP11.HOLDFIXED = 1;
SOLVE PEP11 USING MCP;

display PEP11.modelstat, PEP11.solvestat, PEP11.numinfes;
