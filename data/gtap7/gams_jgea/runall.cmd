call gams gft       -pw=140 -ps=999999
call gams gftCapFix -pw=140 -ps=999999
call gams gftCET    -pw=140 -ps=999999
call gams gftFlxFF  -pw=140 -ps=999999
call gams gftLnd    -pw=140 -ps=999999
merge
