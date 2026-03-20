$title SimpleOpen v1 benchmark reference (stdcge-reduced)
$offsymlist
$offsymxref
$eolcom !

$if not set CLOSURE $setglobal CLOSURE simple_open_default
$if not set OUT_GDX  $setglobal OUT_GDX  simple_open_v1_benchmark_results.gdx

set
    var   benchmark variables / VA, INT, X, D, E, ER, PFX, CAB, FSAV /
    eqn   benchmark equations / EQ_VA, EQ_INT, EQ_CET /
    cl    closure labels / simple_open_default, flexible_external_balance /;

set active_closure(cl);
active_closure("%CLOSURE%") = yes;
abort$(card(active_closure) <> 1) "Unknown CLOSURE value", "%CLOSURE%";

scalar
    closure_code
    alpha_va
    rho_va
    a_int
    b_ext
    theta_cet
    phi_cet
    ER0
    PFX0
    D0
    E0
    CAB0
    FSAV0;

$ifthen "%CLOSURE%" == "simple_open_default"
    closure_code = 101;
    alpha_va  = 0.40;
    rho_va    = 0.75;
    a_int     = 0.50;
    b_ext     = 0.10;
    theta_cet = 0.60;
    phi_cet   = 1.20;
    ER0       = 1.00;
    PFX0      = 1.00;
    D0        = 1.00;
    E0        = 1.00;
    CAB0      = 1.00;
    FSAV0     = 1.00;
$elseif "%CLOSURE%" == "flexible_external_balance"
    closure_code = 202;
    alpha_va  = 0.45;
    rho_va    = 0.70;
    a_int     = 0.55;
    b_ext     = 0.08;
    theta_cet = 0.58;
    phi_cet   = 1.25;
    ER0       = 1.08;
    PFX0      = 1.00;
    D0        = 1.04;
    E0        = 0.93;
    CAB0      = 0.82;
    FSAV0     = 0.82;
$else
$abort "Unsupported closure for simple_open_v1 benchmark"
$endif

positive variables
    VA
    INT
    X
    D
    E
    ER
    PFX
    CAB
    FSAV;

variable OBJ;

equations
    eq_va
    eq_int
    eq_cet
    objdef;

eq_va..
    VA =e=
        rPower(
            alpha_va * rPower(ER, rho_va)
          + (1 - alpha_va) * rPower(PFX, rho_va),
            1 / rho_va
        );

eq_int..
    INT =e= a_int * X + b_ext * (CAB - FSAV);

eq_cet..
    X =e=
        rPower(
            theta_cet * rPower(D, phi_cet)
          + (1 - theta_cet) * rPower(E * ER / PFX, phi_cet),
            1 / phi_cet
        );

objdef.. OBJ =e= 0;

VA.lo   = 1e-8;
INT.lo  = 1e-8;
X.lo    = 1e-8;
D.lo    = 1e-8;
E.lo    = 1e-8;
ER.lo   = 1e-8;
PFX.lo  = 1e-8;
CAB.lo  = 1e-8;
FSAV.lo = 1e-8;

D.fx    = D0;
E.fx    = E0;
ER.fx   = ER0;
PFX.fx  = PFX0;
CAB.fx  = CAB0;
FSAV.fx = FSAV0;

VA.l   = rPower(alpha_va * rPower(ER0, rho_va) + (1 - alpha_va) * rPower(PFX0, rho_va), 1 / rho_va);
X.l    = rPower(theta_cet * rPower(D0, phi_cet) + (1 - theta_cet) * rPower(E0 * ER0 / PFX0, phi_cet), 1 / phi_cet);
INT.l  = a_int * X.l + b_ext * (CAB0 - FSAV0);
D.l    = D0;
E.l    = E0;
ER.l   = ER0;
PFX.l  = PFX0;
CAB.l  = CAB0;
FSAV.l = FSAV0;
OBJ.l  = 0;

model simple_open_v1_benchmark / eq_va, eq_int, eq_cet, objdef /;

solve simple_open_v1_benchmark minimizing OBJ using nlp;

parameter
    benchmark(var)   canonical benchmark levels
    level(var)       solved levels
    residual(eqn)    post-solve residuals
    calib(*)         benchmark parameters and metadata;

benchmark("VA")   = rPower(alpha_va * rPower(ER0, rho_va) + (1 - alpha_va) * rPower(PFX0, rho_va), 1 / rho_va);
benchmark("X")    = rPower(theta_cet * rPower(D0, phi_cet) + (1 - theta_cet) * rPower(E0 * ER0 / PFX0, phi_cet), 1 / phi_cet);
benchmark("INT")  = a_int * benchmark("X") + b_ext * (CAB0 - FSAV0);
benchmark("D")    = D0;
benchmark("E")    = E0;
benchmark("ER")   = ER0;
benchmark("PFX")  = PFX0;
benchmark("CAB")  = CAB0;
benchmark("FSAV") = FSAV0;

level("VA")   = VA.l;
level("INT")  = INT.l;
level("X")    = X.l;
level("D")    = D.l;
level("E")    = E.l;
level("ER")   = ER.l;
level("PFX")  = PFX.l;
level("CAB")  = CAB.l;
level("FSAV") = FSAV.l;

residual("EQ_VA") =
    VA.l
  - rPower(
        alpha_va * rPower(ER.l, rho_va)
      + (1 - alpha_va) * rPower(PFX.l, rho_va),
        1 / rho_va
    );

residual("EQ_INT") = INT.l - (a_int * X.l + b_ext * (CAB.l - FSAV.l));

residual("EQ_CET") =
    X.l
  - rPower(
        theta_cet * rPower(D.l, phi_cet)
      + (1 - theta_cet) * rPower(E.l * ER.l / PFX.l, phi_cet),
        1 / phi_cet
    );

calib("alpha_va")  = alpha_va;
calib("rho_va")    = rho_va;
calib("a_int")     = a_int;
calib("b_ext")     = b_ext;
calib("theta_cet") = theta_cet;
calib("phi_cet")   = phi_cet;
calib("closure_code") = closure_code;
calib("modelstat") = simple_open_v1_benchmark.modelstat;
calib("solvestat") = simple_open_v1_benchmark.solvestat;
calib("obj")       = OBJ.l;

execute_unload "%OUT_GDX%", active_closure, benchmark, level, residual, calib;
