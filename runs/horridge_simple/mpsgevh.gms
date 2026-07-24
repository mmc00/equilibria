$title  A Simple General Equilibrium Model

* Next line enables end-of-line comments following #
$eolcom #

set
        i       Goods and sectors    # COM, or IND
        fd      Components final demand /hou,gov,inv/,   # LOCFIN
        f       Primary factors /labor,capital/,  # This capital is mobile   # FAC
        src     Sources /dom,imp/;   # SRC

$gdxin input.gdx # simdata
$load i
parameter
        use(i,src,*)    Value of inputs,
        factor(f,i)     Factor demands,
        sigma(i)        CES domestic-import elasticity;
$loaddc use factor sigma

alias (i,j);

parameter
        output(i)       Aggregate output          COSTS(c)
        expend(fd)      Final demand expenditure    COSTS(L)
        export(i)       Benchmark exports            USE(c:dom:exp)
        epsilonx(i)     Export demand elasticity        5.0
        bopdef          Base year current account deficit
        sigmaf(i)       Elasticity of substitution among factors    one half
        prd(f,j)        Productivity index   AFAC(f:i)
        ssk(j)          Sector-specific capital
        endow(f)        Primary factor endowment vector    SUM(i:FACTOR(f:i))
        gdp0            Benchmark gross domestic product    VGDP   ;

prd(f,j) = 1;
output(j) = sum((i,src), use(i,src,j)) + sum(f,factor(f,j));
export(i) = use(i,"dom","exp");
expend(fd) = sum((i,src),use(i,src,fd));
epsilonx(i) = 5;
sigmaf(i) = 0.5;
ssk(j) = 0;   # start with no sector-specific capital
bopdef = sum((i,j), use(i,"imp",j)) + sum((i,fd),use(i,"imp",fd)) - sum(i,export(i));
endow(f) = (sum(j,factor(f,j)));
gdp0 = sum(f,endow(f));

$ontext
$model:simple

* Comment within MPSGE code

$sectors:
        Y(j)            ! 1-based index of production   ZINDEX

$commodities:
        P(j)            ! Price of domestic output    p(c:dom)
        PFX             ! Exchange rate (Dollar domestic per foreign dollar) PHI
* There are 2 kinds of capital -
*  Mobile has price W("capital") and fixed capital in sector j has price RK(j)
* In SIMPLE.TAB factor price is pfac(f:i) which might depend on using industry i.
        W(f)$endow(f)   ! Factor price      PFAC(f:i)
        PGNP            ! Price index for factor income
        RK(j)$ssk(j)    ! Rental rate on sector-specific capital  PFAC(capital:i)

$consumers:
        GNP             ! Aggregate national income
        RA(fd)          ! Expenditure by representative agent   WTOT(L)

$auxiliary:
        X(i)            ! Export quantity     XEXP(c)
        VX              ! Value of exports (Foreign currency)   COSTS(exp) in domestic dollars
        GDP             ! 1-based index of the value of GDP in foreign dollars

$Prod:Y(j)      s:0 va:sigmaf(j) i.tl:sigma(i)
        o:P(j)          q:output(j)
        i:W(f)          q:(factor(f,j)/prd(f,j)) p:prd(f,j)     va:
        i:RK(j)         q:ssk(j)                va:
        i:PFX#(i)       q:use(i,"imp",j)        i.tl:
        i:P(i)          q:use(i,"dom",j)        i.tl:

$report:
        v:wd(f,j)       i:W(f)  prod:Y(j)
        v:VINTIMP(j)     i:PFX#(i)  prod:Y(j)

$demand:GNP
* Next two lines add up to primary factor income
        e:W(f)          q:endow(f)
        e:RK(j)         q:ssk(j)
* Next 2 lines seem to cancel each other out
        e:P(i)          q:(-1)                  r:X(i)
        e:PFX           q:1                     r:VX
* Next says there is a foreign gift of PFX*bopdef*GDP local dollars
        e:PFX           q:bopdef                r:GDP
        d:PGNP

$demand:RA(fd)  s:1  i.tl:sigma(i)
* Next says income(fd)=PGNP*expend(fd) for each of the 3 fd's
        e:PGNP          q:expend(fd)
* Next 2 lines say all fd have armington demands for import v domestic with elasticity sigma
* and that elasticity is 1 between different commodities
        d:P(i)          q:use(i,"dom",fd)       i.tl:
        d:PFX#(i)       q:use(i,"imp",fd)       i.tl:

$report:
        v:VFINIMP(fd)   d:PFX#(i)  demand:RA(fd)

$constraint:X(i)
        X(i) =e= export(i) * (PFX/P(i))**epsilonx(i);

$constraint:VX
        PFX * VX =e= sum(i, P(i)*X(i));

$constraint:GDP
* GDP is a 1-based index of the value of GDP in foreign dollars
* Actual GDP is LHS of next equation.
        GDP*gdp0*PFX =e= sum(f, W(f)*endow(f)) + sum(j, RK(j)*ssk(j));

$offtext
$sysinclude mpsgeset simple

* NOTE. Foreign gift divided by actual GDP is equal to
*    PFX*bopdef*GDP/GDP*gpd0*PFX = bopdef/gdp0 which is constant

GDP.L = 1;
X.L(i) = export(i);
X.FX(i)$(export(i)=0) = 0;
VX.L = sum(i, X.L(i));

*       Assign a numeraire to simplify comparison with the GEMPACK solution:
PFX.FX = 1;

simple.iterlim = 0;
simple.workspace = 50;
$include simple.gen
solve simple using mcp;
abort$(simple.objval>1e-2) "Benchmark without SSK inconsistent.";

*       Swap the closure and verify

ssk(j) = factor("capital",j);
factor("capital",j) = 0;
endow("capital") = 0;

simple.iterlim = 0;
simple.workspace = 50;
$include simple.gen
solve simple using mcp;
abort$(simple.objval>1e-2) "Benchmark with SSK inconsistent.";

display VINTIMP.L;

parameter summary  Summary of model results ;
summary("ActualGDP", "initial") = sum(f, W.L(f)*endow(f)) + sum(j, RK.L(j)*ssk(j)) ;
summary("VIMP","initial") = sum(j, VINTIMP.L(j)) + sum(fd, VFINIMP.L(fd)) ;
summary("VEXP","initial") = sum(i, P.L(i)*X.L(i)) ;
summary("BOP", "initial") = summary("VEXP","initial") - summary("VIMP", "initial") ;
summary("BOPRatio","initial") = summary("BOP","initial")/summary("ActualGDP","initial") ;

*       Simulate a productivity shock for labor inputs to the
*       service sector:

prd("labor","srv") = 1/0.90;

*       First simulation is based on mobile capital:

factor("capital",j) = ssk(j);
endow("capital") = sum(j,ssk(j));
ssk(j) = 0;

simple.iterlim = 20000;
$include simple.gen
solve simple using mcp;

parameter       impact  Economic impact;
impact("Y%",j,"mobileK") =  100 * (Y.L(j)-1);
impact("X%",j,"mobileK")$export(j) =  100 * (X.L(j)/export(j)-1);
impact("P%",j,"mobileK") =  100 * (P.L(j)/PFX.L-1);
impact("P%",f,"mobileK")$endow(f) =  100 * (W.L(f)/PFX.L-1);
impact("RK%",j,"mobileK")$ssk(j) =  100 * (RK.L(j)/PFX.L-1);

summary("ActualGDP", "mobileK") = sum(f, W.L(f)*endow(f)) + sum(j, RK.L(j)*ssk(j)) ;
summary("VIMP","mobileK") = sum(j, VINTIMP.L(j)) + sum(fd, VFINIMP.L(fd)) ;
summary("VEXP","mobileK") = sum(i, P.L(i)*X.L(i)) ;
summary("BOP", "mobileK") = summary("VEXP","mobileK") - summary("VIMP", "mobileK") ;
summary("BOPRatio","mobileK") = summary("BOP","mobileK")/summary("ActualGDP","mobileK") ;

*       Second simulation is based on sector-specific capital:

ssk(j) = factor("capital",j);
factor("capital",j) = 0;
endow("capital") = 0;

simple.iterlim = 20000;
$include simple.gen
solve simple using mcp;

impact("Y%",j,"SSK") =  100 * (Y.L(j)-1);
impact("X%",j,"SSK")$export(j) =  100 * (X.L(j)/export(j)-1);
impact("P%",j,"SSK") =  100 * (P.L(j)/PFX.L-1);
impact("P%",f,"SSK")$endow(f) =  100 * (W.L(f)/PFX.L-1);
impact("RK%",j,"SSK")$ssk(j) =  100 * (RK.L(j)/PFX.L-1);

summary("ActualGDP", "SSK") = sum(f, W.L(f)*endow(f)) + sum(j, RK.L(j)*ssk(j)) ;
summary("VIMP","SSK") = sum(j, VINTIMP.L(j)) + sum(fd, VFINIMP.L(fd)) ;
summary("VEXP","SSK") = sum(i, P.L(i)*X.L(i)) ;
summary("BOP", "SSK") = summary("VEXP","SSK") - summary("VIMP", "SSK") ;
summary("BOPRatio","SSK") = summary("BOP","SSK")/summary("ActualGDP","SSK") ;

option impact:3:1:1;
display impact;

option summary:5:1:1;
display summary;
EXECUTE_UNLOAD 'RESULTS', Impact;

