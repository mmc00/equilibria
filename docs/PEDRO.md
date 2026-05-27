PEDRO
Parallel Engine for Dynamic and Reduced Optimization

A next-generation MCP/CGE solver focused on:

large-scale complementarity problems,
GPU acceleration,
matrix-free Newton-Krylov methods,
structural reduction,
and high-throughput scenario solving.

PEDRO is designed to be compatible with:

Pyomo,
PyOptInterface,
and future GTAP-style workflows.
Vision

Traditional MCP solvers such as:

PATH solver
MILES

were designed in a CPU-centric era.

PEDRO explores a different architecture:

matrix-free operators,
GPU-native kernels,
hybrid sparse strategies,
structural reduction,
and parallel scenario execution.

The goal is not merely to reproduce PATH, but to build a modern HPC-oriented platform for large-scale CGE/MCP models.

Core Design Philosophy

PEDRO is based on five principles:

1. Operator-based architecture

PEDRO avoids hard coupling between:

model,
Jacobian,
solver,
and backend.

Everything is expressed as operators.

2. Matrix-free first

Instead of explicitly building:

J(x)

PEDRO initially focuses on:

J(x)v

Jacobian-vector products.

This:

reduces memory pressure,
improves scalability,
enables iterative Krylov methods,
and maps naturally to accelerators.
3. Hybrid sparse support

PEDRO is not purely matrix-free.

The architecture supports:

explicit sparse Jacobians,
block preconditioners,
hybrid operators,
and future symbolic reduction.
4. GPU-oriented computation

PEDRO treats GPU acceleration as a core architectural concern.

Target GPU workloads:

residual evaluation,
complementarity reformulations,
Jacobian-vector products,
sparse linear algebra,
batch scenario solving.
5. Structural reduction

Long-term, PEDRO aims to include:

equation elimination,
variable substitution,
block decomposition,
and compiler-style model reduction.

This is essential for competing with highly optimized CGE systems.

Mathematical Foundation

PEDRO targets Mixed Complementarity Problems (MCP):

0≤x⊥F(x)≥0

Equivalent conditions:

x
i
	​

≥0
F
i
	​

(x)≥0
x
i
	​

F
i
	​

(x)=0

PEDRO reformulates complementarity into a continuous residual system using functions such as Fischer-Burmeister:

ϕ(a,b)=
a
2
+b
2
	​

−(a+b)

The nonlinear solve becomes:

R(x)=0

where:

R
i
	​

(x)=ϕ(x
i
	​

,F
i
	​

(x))
Numerical Strategy
Phase 1
Nonlinear method
Inexact Newton
Linear method
GMRES
Jacobian handling
Matrix-free Jv
Globalization
Damping / line search
Complementarity reformulation
Fischer-Burmeister
Long-Term Solver Strategy

Future PEDRO versions may include:

semismooth Newton,
explicit sparse Jacobians,
adaptive hybrid modes,
block preconditioners,
multilevel methods,
multi-GPU execution,
mixed precision,
distributed solving.
High-Level Architecture
┌───────────────────────────┐
│        Frontends          │
│ Pyomo / PyOptInterface    │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│      Model Compiler       │
│ residual graph / reducer  │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│     Residual Operator     │
│         R(x)              │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│      Jv Operator          │
│       J(x)v               │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│      Linear Solver        │
│          GMRES            │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│     Nonlinear Solver      │
│     Newton / damping      │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│     CPU / GPU Backend     │
└───────────────────────────┘
Repository Structure
pedro/
├── cpp/
│   ├── include/pedro/
│   ├── src/
│   ├── tests/
│   └── examples/
├── python/
│   ├── experiments/
│   └── bindings/
├── docs/
├── benchmarks/
└── README.md
Phase 1 Architecture

Phase 1 focuses on correctness and extensibility.

Core components:

Component	Responsibility
Model	Defines F(x)
ResidualOperator	Builds R(x)
JvOperator	Computes J(x)v
LinearSolver	Solves linear systems
GMRES	Krylov iterative method
NonlinearSolver	Newton iteration
Damping	Stabilization
Solution	Result metadata
Initial Technology Stack
Core language
C++17 or C++20
Linear algebra
Eigen
Build system
CMake
Testing
Catch2 or GoogleTest
Python integration
pybind11 (future phase)
GPU stack (future phase)
CUDA
cuSPARSE
cuSOLVER
Why C++

PEDRO is designed for:

HPC,
sparse computation,
GPU acceleration,
custom operators,
and future CUDA integration.

C++ provides:

memory control,
backend flexibility,
high-performance interoperability,
and low-overhead abstractions.

Python remains important for:

modeling,
experimentation,
orchestration,
and ecosystem compatibility.
Example MCP

Simple scalar MCP:

0≤x⊥(x−2)≥0

Solution:

x=2
Phase Roadmap
Phase 1
CPU solver
Jv
GMRES
toy MCPs
Phase 2
preconditioners
scaling
robustness
Phase 3
GPU kernels
residual/Jv acceleration
Phase 4
hybrid explicit/matrix-free modes
Phase 5
structural reduction
symbolic elimination
Phase 6
batch solving
multi-scenario execution
distributed systems
Long-Term Research Directions

Potential future research topics:

semismooth Newton on GPU,
complementarity-specific preconditioners,
automatic differentiation backends,
graph-based model decomposition,
asynchronous nonlinear solves,
dynamic precision strategies,
scenario batching,
distributed CGE pipelines.
Performance Expectations vs Established Solvers

The following table compares projected PEDRO runtimes against measured
runtimes of GEMPACK (the de-facto reference for CGE/GTAP) and GAMS/MCP
(PATH via the AMPL/NL interface) for a typical GTAP v6.2 BOOK3X3-style
10% tariff-cut experiment across the gtap6_* aggregations.

Units are seconds unless noted. PEDRO entries are extrapolations from
the architectural choices (Newton-Krylov + Fischer-Burmeister + matrix-free Jv);
GEMPACK and GAMS entries are from published benchmarks and our own runs.
Python prototype = scipy.optimize.root(method="krylov") + Fischer-Burmeister
wrapper over an existing Pyomo callback_f(x), with block-Jacobi
preconditioner. Empirical profiling on gtap6_15x10 shows Pyomo
expression evaluation is only 7% of PATH wall time, so a Python
prototype is roughly 3-5× slower than the equivalent C++ implementation
— not 50× as commonly assumed.

Dataset       Vars    GEMPACK   GAMS MCP   Py prototype (NK+FB)   PEDRO P1 (CPU)   PEDRO P2 (+precond)   PEDRO P3 (GPU)
gtap6_3x3      663      <1       5–15           5–30                  1–5                1–3                  1–3
gtap6_5x5     2239      1–3      30–60          15–60                 5–15               3–8                  1–3
gtap6_10x7    8965      5–15     2–5 min        1–3 min               30–90              15–45                3–10
gtap6_15x10  25720     15–30    5–15 min       3–10 min               1–3 min            30–90                5–20
gtap6_20x41  ~290K    1–3 min  30–90 min      30–90 min              10–30 min           3–8 min              30–90

How the Python prototype timing is derived

Empirical per-Newton-iteration cost on gtap6_15x10 with the existing
Pyomo callbacks (measured during the Phase 3.36 investigation):

  callback_f (single residual evaluation):     ~0.2 s
  callback_jac (sparse reverse-numeric Jac):   ~1.3 s

For matrix-free Newton-Krylov each inner GMRES iteration needs one Jv,
which we approximate using forward-difference (Brown-Saad style):

  Jv ≈ (F(x + h*v) - F(x)) / h      → one callback_f per Jv

So per Newton outer iteration:

  cost ≈ 1 callback_f (residual)
       + k callback_f (Jv with k inner GMRES iterations)
       = (k+1) * 0.2 s

With block-Jacobi preconditioner, k typically reaches 10–30 on CGE.
Total per Newton step: 2–6 s.

Newton outer iterations to convergence: ~30–100 for a tariff shock.

Total prototype wall time at 25K vars: 60–600 s — i.e. 1–10 minutes.

For gtap6_20x41 (290K vars), callback_f scales roughly linearly with
the number of variables (Pyomo expression evaluation is sparse), so
each callback_f is ~2–3 s. Per Newton step: 20–90 s. Total: 30–90
minutes.

What the prototype validates that C++ does not change

The Python prototype answers convergence questions, which dominate the
PEDRO risk:

  * Does Fischer-Burmeister + Newton-Krylov converge where PATH stalls?
  * Is block-Jacobi enough as a preconditioner, or do we need ILU?
  * Does the continuation framework recover from the merit-function
    traps we observe on gtap6_15x10 shocked?
  * What fraction of gtap6 shocks (across 100s of parameter shocks)
    converge with this algorithm vs PATH?

A C++ implementation that converges 3-5× faster on a problem where
NEITHER converges is worth nothing. The Python prototype answers the
convergence question for the cost of one engineer-month, before any
C++ investment.

Reference: Hertel–Pearson type benchmark (500 sectors) — solver time
elasticity with respect to sector count:

GEMPACK exe:    2.9   (~26 s on 500 sectors)
GEMSIM:         3.3
MPSGE:          6.6
GAMS NLP:       8.8
GAMS MCP:      16.4   (~7550 s on 500 sectors)

GEMPACK's low elasticity (2.9) reflects Gragg-multi 2-4-6 deferred
correction over a fixed-size linearized system plus highly tuned
sparse-LU pivoting (MA28-class) and decades of Fortran optimization.

GAMS MCP scales poorly (16.4) because PATH solves the full nonlinear
MCP via Newton + LUSOL on the un-condensed system; sparse-LU on the
nonlinear Jacobian becomes the dominant cost as the problem grows.

How PEDRO's estimates are derived

Per Newton iteration the dominant cost is:

  cost = residual_eval + k * Jv_product

where k is the number of inner GMRES iterations. Empirically:

  Phase 1 (no preconditioner):           k = 100–500
  Phase 2 (ILU / block-Jacobi):           k = 10–50
  Phase 3 (GPU, same algorithmic k):     each Jv ~50× faster

For a CGE Jacobian with ~12·n non-zeros, one Jv on CPU C++ runs in
about 0.005 s for n ≈ 25,000 and 0.05 s for n ≈ 290,000. Newton
iterations to convergence on a well-shocked CGE typically range
30–100. Putting these together yields the table above.

Strategic implications

GEMPACK remains the runtime gold standard on CPU. Its combination of
Gragg-multi, condensed linearization, and tuned LU cannot be matched
without either (i) GPU-accelerated matrix-free Newton-Krylov, or
(ii) a comparably tuned Fortran-style direct sparse LU on the same
condensed form. PEDRO bets on (i).

GAMS MCP is reliable but slow on the larger aggregations. For
gtap6_20x41 the expected GAMS MCP runtime exceeds 30 minutes, which
limits its usefulness for scenario sweeps and dynamic models.

PEDRO Phase 1 (CPU, no preconditioner) is roughly competitive with
GAMS MCP and reaches the larger datasets where the unprepared Newton
solver currently does not converge. This is the first interesting
threshold — making 20x41 tractable at all.

PEDRO Phase 2 (with preconditioner) brings the CPU runtimes into the
1-min range for 15x10 and the few-minute range for 20x41. This is
where PEDRO becomes attractive for routine scenario work.

PEDRO Phase 3 (GPU) is where PEDRO matches or beats GEMPACK on the
largest datasets. For gtap6_20x41 the expected runtime is in the
30–90 s range, comparable to GEMPACK's CPU performance, but within
an open Python-compatible stack.

A Python-only PEDRO prototype (e.g. scipy.optimize.root with
method="krylov" plus a Fischer-Burmeister wrapper around an existing
Pyomo F(x) callback) is approximately 3–5× slower than the C++
target. That still places gtap6_15x10 in the 5–15 min range and
gtap6_20x41 around an hour — useful for validating the approach
before investing in a C++/CUDA implementation.

Caveats

PEDRO numbers are projections, not measurements. The actual
performance will depend on:

  * conditioning of the Jacobian under Fischer-Burmeister,
  * effectiveness of the preconditioner family chosen,
  * residual-evaluation cost from the modelling frontend,
  * GPU memory bandwidth and kernel-launch overhead,
  * structural reduction quality (eliminating trivial cells).

The convergence question is at least as important as the runtime
question. Phase 1 must first demonstrate that Fischer-Burmeister +
inexact Newton + GMRES converges reliably on real CGE shocks
(particularly the 10% tariff family on GTAP v6.2) before the runtime
comparison becomes meaningful.

Open Architectural Questions and Known Risks

Engineering review based on practical experience solving gtap6_*
datasets with PATH+LUSOL+Pyomo. These items are ordered by impact —
the first four are critical, the next four important, and the last
four are incremental polish.

CRITICAL — these change whether PEDRO works at all on CGE

1. Preconditioning must be Phase 1, not Phase 2.
   GMRES without preconditioning fails on CGE. GTAP prices and
   quantities span 12+ orders of magnitude, so the Jacobian condition
   number routinely reaches 10^8–10^12. Unpreconditioned GMRES would
   need thousands of inner iterations, killing the matrix-free
   advantage. Minimum viable Phase 1: block-Jacobi diagonal plus
   variable/equation scaling. Without this, even gtap6_5x5 will not
   converge.

2. Variable and equation scaling as a first-class citizen.
   In our PATH+LUSOL work we added variable_scaling and
   equation_scaling as ad hoc options on top of the wrapper. In
   PEDRO these must be part of the R(x) operator itself, not optional
   flags. Without them eq_market residuals at ~10^4 coexist with
   eq_pms residuals at ~10^0, and Newton-Krylov cannot find a
   meaningful step direction.

3. Explicit automatic-differentiation strategy.
   PEDRO says "matrix-free Jv" but does not specify how Jv is
   computed. Finite differences are cheap but suffer catastrophic
   cancellation on CGE Jacobians. Forward-mode AD is more robust but
   needs infrastructure. Reverse-mode AD is expensive for large
   Jacobians. This is a foundational decision, not a detail —
   empirical work on nested-CES expressions (GTAP) shows reverse_numeric
   beats symbolic by ~15× per Jv. The AD choice determines whether
   PEDRO scales.

4. Homotopy / continuation as a first-class framework.
   PEDRO currently treats MCP as single-solve. In CGE practice the
   problem is ALWAYS continuation in a shock parameter λ ∈ [0,1].
   PATH with substep-ladder homotopy works up to ~10K variables and
   fails at ~25K. PEDRO needs predictor-corrector continuation
   built in: Euler predictor + Newton corrector, adaptive step size
   (empirical rule: ~10% per substep at 10K vars, ~5% at 25K vars),
   and arc-length continuation for fold points. Without this Phase 1
   may converge on baseline but fail every shocked experiment.

IMPORTANT — high return on investment

5. Python prototype before C++.
   PEDRO commits to C++ for Phase 1 with a 5-8 month timeline. The
   algorithm should be validated in Python first: scipy.optimize.root
   with method="krylov" plus a Fischer-Burmeister wrapper around an
   existing Pyomo F(x) callback is ~500 lines of code and 1-2 weeks
   of work. If the approach fails on gtap6_15x10 in Python, it will
   also fail in C++. Empirical profiling shows Python expression
   evaluation is only 7% of wall time on gtap6_15x10, so Python is
   not the bottleneck.

6. First-class observability and structured logging.
   Debugging nonlinear solvers without iteration-by-iteration trace
   wastes weeks. The reference PATH solver in our project writes to
   an internal output channel that the C-API wrapper does not
   forward, leaving us unable to confirm whether PATH actually
   loaded UMFPACK or silently fell back to LUSOL. PEDRO must have,
   from day one:

     * structured per-iteration logging (JSON-lines or similar),
     * documented callbacks for intermediate-state inspection,
     * metrics: GMRES inner iters, residual reduction, step length,
       basin attraction estimates,
     * an opt-in trace mode that writes (x_k, F(x_k), J*v samples)
       to disk for offline analysis.

7. Fallback / multi-method strategy.
   Production PATH dispatches between Lemke restart, perturbation,
   and multiple crash methods when Newton fails. PEDRO's current
   plan ("Newton + damping") is fragile. A method dispatcher should:

     * switch to semismooth Newton if FB stalls,
     * apply randomised perturbation if iterates plateau,
     * roll back to the previous step with a more conservative
       predictor,
     * fall back to a penalty / barrier formulation as last resort.

8. Calibration / SAM pre-processing layer.
   Real CGE data (GTAP, GAMS models) has imperfect calibration —
   roughly 150 SAM cells in gtap6_15x10 have baseline residuals in
   the 1e-5 to 1e-3 range that are NOT exact zero. Without a
   pre-processing layer that absorbs these as constants ("bake"),
   detects structurally-zero variables (pwmg=0, qfe=0, etc.), and
   runs data-driven conditional fixing, PEDRO will hit residual
   floors that look like solver bugs but are actually data issues.

INCREMENTAL — quality improvements

9. cuSPARSE is not enough for Phase 3.
   cuSPARSE's sparse LU is relatively weak compared to alternatives.
   For Phase 3 evaluate:

     * Ginkgo: open source, multi-GPU, strong for sparse iterative,
     * HYPRE: multigrid, scales distributed,
     * rocSPARSE: AMD equivalent for hardware portability.

10. Mixed-precision strategy must be explicit early.
    A reasonable default: double precision for residual evaluation
    (numerical stability), single precision for preconditioner
    application (memory, speed), iterative refinement to recover
    accuracy. Hardcoding double everywhere in Phase 1 makes a
    refactor expensive later.

11. Frontend independence: trade-offs should be documented.
    PEDRO mentions Pyomo and PyOptInterface as frontends.
    PyOptInterface is roughly 10× faster than Pyomo in expression
    evaluation, but profiling shows that on our reference problem
    expression evaluation is only 7% of total time — so the choice
    is about usability and ecosystem more than performance. A
    direct NL/AMPL backend is the maximum-performance option at
    the cost of flexibility. These trade-offs should be in the doc.

12. Explicit benchmark suite from Phase 1.
    Without a test suite Phase 1 "correctness" is unverifiable.
    Recommended targets:

      * MCPLIB (the 1005-problem standard benchmark),
      * GAMS GTAP shocks on BOOK3X3 and the gtap6_* aggregations,
      * synthetic CGE generator scaling to 10K, 100K, 1M variables.

The deeper question: convergence precedes performance

PEDRO's documentation emphasises performance (matrix-free, GPU, batch
scenarios). The actual problem with PATH on large CGE is not speed
but RELIABILITY OF CONVERGENCE. We observed gtap6_15x10 failing to
advance even from a perfect baseline (residual 5e-8 at start, growing
monotonically to 8.2e-2 after 16 substeps with VIWS unchanged).

PEDRO will only displace PATH for CGE work if it converges where
PATH does not. The success criterion that should drive Phase 1 is:

  "On gtap6_15x10 with a 10% tariff shock, PEDRO reaches a residual
   below 1e-6 with VIWS within 1% of the GEMPACK reference, in any
   wall-clock time."

If Phase 1 cannot meet that bar, Phase 2 and 3 are irrelevant.

Python Prototype Design Specification

The prototype is a thin Python stack that swaps PATH's Newton+LUSOL
core for matrix-free Newton-Krylov on a Fischer-Burmeister residual,
reusing every other layer of the existing equilibria pipeline.

Module structure

  pedro_proto/
  ├── __init__.py
  ├── operators.py          R(x) operator: variable/equation scaling + FB
  ├── jv.py                 Matrix-free Jacobian-vector product (Brown-Saad FD)
  ├── preconditioners.py    Block-Jacobi from sparsity pattern; ILU0 fallback
  ├── newton_krylov.py      Outer Newton; inner GMRES via scipy
  ├── globalization.py      Line search (Armijo); trust region (future)
  ├── continuation.py       Predictor-corrector for shock parameter λ
  ├── multi_method.py       Fallback dispatcher (FB → semismooth → perturb)
  ├── observability.py      Structured logging, per-iter metrics
  └── pyomo_bridge.py       Integration with existing equilibria pipeline

Core algorithms

Fischer-Burmeister reformulation. For each complementarity pair
(x_i, F_i) with x_i ∈ [lb_i, ub_i]:

  φ(a, b) = sqrt(a² + b²) - (a + b)

  R(x)_i = φ(x_i - lb_i, F_i(x))     if x_i is lower-bounded
         = -φ(ub_i - x_i, -F_i(x))   if x_i is upper-bounded
         = F_i(x)                    if x_i is free

Solving R(x) = 0 is mathematically equivalent to the original MCP,
but R is C^1 (with subgradient at (0,0)) and admits standard
Newton-type methods.

Jacobian-vector product (matrix-free). Brown-Saad style:

  Jv ≈ (R(x + h·v) - R(x)) / h
  h = sqrt(eps_mach) · max(||x||_∞, 1) / ||v||_∞

One callback_f per Jv evaluation. Robust on sparsity, no symbolic
AD required, fits naturally into scipy.sparse.linalg.LinearOperator.

Newton-Krylov outer loop.

  k = 0
  while ||R(x_k)||_∞ > tol_outer:
      Δx = GMRES_solve(J(x_k), -R(x_k), M=preconditioner)
      α  = armijo_line_search(x_k, Δx, R)
      x_{k+1} = x_k + α · Δx
      k += 1

GMRES is provided by scipy.optimize.root(method="krylov") or
scipy.sparse.linalg.gmres directly for finer control.

Block-Jacobi preconditioner. The PEDRO design assumes block
structure in the CGE Jacobian. The prototype implements a 3-block
diagonal:

  Block 1 — prices       (variables prefixed p)
  Block 2 — quantities   (variables prefixed q)
  Block 3 — macro        (y, walras, savf, etc.)

M = blockdiag(J_11, J_22, J_33) is factorized once per Newton outer
step via scipy.sparse.linalg.splu on each block. M⁻¹·v is the
preconditioner action. The 3-block decomposition is detected
automatically from variable names; an explicit override hook lets
the user specify alternative partitions.

Continuation framework (predictor-corrector). For a shock indexed by
α ∈ [0, 1]:

  α = 0; x = baseline_equilibrium
  while α < 1:
      Δα = adaptive_step(prev_residual_reduction, prev_inner_iters)
      x_pred = x + Δα · (∂x/∂α)              # tangent extrapolation
      set_shock_parameter(α + Δα)
      x_new, info = newton_krylov(x_pred)
      if info.converged:
          x = x_new; α += Δα
      else:
          Δα = Δα / 2                         # shrink step, retry

Tangent ∂x/∂α is approximated by one extra finite-difference call
or solved as a side linear system at each successful substep.
Adaptive sizing reduces Δα on slow convergence and grows it when
GMRES inner iterations are small.

Reuse from the equilibria codebase

The prototype is additive — no existing v6.2 component is replaced
or rewritten. It plugs into:

  equilibria.templates.gtap_v62.GTAPv62ModelEquations
    → model definition; unchanged

  scripts/gtap_v62/_make_square.apply_v62_pipeline
    → closure, bake, conditional fix, dead-row drop; unchanged

  path_capi_python.PyomoMCPAdapter.build_nonlinear_from_equality_constraints
    → already produces callback_f(x); we reuse this callback as the
      F(x) input to the Fischer-Burmeister wrapper

What the prototype replaces

  PATH's Newton (active-set + merit function)  →  Newton-Krylov on R(x)
  LUSOL linear solve                           →  scipy.sparse + block-Jacobi
  PATH's substep loop                          →  predictor-corrector continuation
  No observability                             →  structured per-iter logging

Validation datasets

  Toy:        1-variable MCP (sanity check)
  Real CGE:   gtap6_3x3 (663),    gtap6_5x5 (2.2K),  gtap6_10x7 (9K),
              gtap6_15x10 (25.7K), gtap6_20x41 (290K)
  Benchmark:  selected MCPLIB problems (optional)

Questions the Python prototype must answer

The prototype is an experiment, and an experiment without a clear
question is wasted work. The following questions, organised by
priority, drive the prototype.

CONVERGENCE — must succeed for PEDRO to be worth building

  C1. Does FB + Newton-Krylov converge on gtap6_15x10 baseline?
      Success criterion: ||R||_∞ < 1e-6 in any wall-clock time.
      Failure here invalidates the entire PEDRO algorithmic premise.

  C2. Does the same algorithm converge on gtap6_15x10 SHOCKED
      (10% tariff cut on tms[OtherFood, USA, EU_28])?
      Success criterion: VIWS within 1% of GEMPACK reference
      (+66.36% per Phase 3.36 measurements).
      This is the experiment PATH currently fails (residual grows
      monotonically, VIWS stays at 0%).

  C3. Does the continuation framework recover from intermediate
      failures?
      Test: deliberately make one substep fail (e.g. shrink the
      Jacobian by 1e-4). Verify the framework backtracks, reduces
      step size, and continues to convergence.

CONDITIONING — defines whether PEDRO needs Phase 2

  P1. Is the 3-block (price/quantity/macro) block-Jacobi sufficient?
      Measure: average GMRES inner iterations per Newton outer step.
      <100 inner iters is a good sign; 100-500 means upgrade needed.

  P2. If not, does ILU(0) bring inner iterations under control?
      Same measurement as P1.

  P3. How do inner iterations scale with problem size?
      Run on 3x3 → 20x41 and plot. Log-linear scaling is OK.
      Worse than log-linear means the preconditioner choice is
      fundamentally inadequate at scale.

PERFORMANCE — defines realistic timelines

  T1. Wall-clock for gtap6_15x10 baseline.
      Target: 1-3 min. Acceptable: <10 min. Failing: >30 min.

  T2. Wall-clock for gtap6_15x10 SHOCKED with continuation.
      Target: 3-10 min. Acceptable: <30 min. Failing: >2 hr.

  T3. Does the prototype scale to gtap6_20x41 without crashing?
      Concerns: scipy.sparse memory ceiling, callback_f NaNs on
      degenerate cells, Pyomo overhead at 290K variables.

ROBUSTNESS — defines production-readiness

  R1. Does the prototype produce the same answer regardless of
      Python hash seed?
      Run 10 times with random PYTHONHASHSEED on the same dataset.
      All 10 should converge to the same equilibrium within 1e-6.

  R2. Does it handle the bipartite-matching variations that trap
      PATH?
      Force PYTHONHASHSEED=0 (the matching that lands PATH in
      the lottery-loss basin) and verify FB+NK still converges.

  R3. How is the prototype affected by SAM imperfections?
      Compare runs with and without bake_tolerance=1e-6.
      The prototype should handle imperfect SAM gracefully.

Decision matrix from prototype results

After 3-5 weeks of prototype work, one of four outcomes is expected:

OUTCOME A — Full success.
  C1, C2 pass; P1 ≤100 inner iters; T1 ≤3 min; T2 ≤10 min.
  Action: commit to PEDRO C++ Phase 1 with high confidence.
  The algorithmic premise is validated; C++ is a performance
  pipeline optimisation, not an architectural risk.

OUTCOME B — Convergence good, preconditioning weak.
  C1 passes; C2 passes after several continuation backtracks;
  P1 yields 200-1000 GMRES inner iters.
  Action: Phase 1 scope expands. ILU(0) or block-ILU is necessary
  before C++ investment is justified. Timeline +3-6 months.

OUTCOME C — Baseline OK, shocked fails.
  C1 passes, C2 fails even with continuation.
  Action: FB + NK is not sufficient for CGE shocked. Investigate
  semismooth Newton (Chen-Harker-Kanzow), Fischer-Burmeister with
  smoothing, or alternative reformulations (Mangasarian-Solodov).
  Re-evaluate before any C++ work — there may be no PEDRO Phase 1.

OUTCOME D — Convergence good, performance unusable.
  C1, C2 pass; T1 > 30 min on 15x10.
  Action: the Pyomo expression evaluation is the bottleneck. Two
  paths:
    a) Switch frontend to PyOptInterface (10× faster expression eval).
    b) Generate .nl file via Pyomo writer and use ASL callbacks
       (50-100× faster but requires C interop).
  C++ is justified but the bottleneck is the frontend, not the solver.

The prototype is the highest-leverage step in the PEDRO roadmap. One
engineer-month of Python work answers a question that would otherwise
require 5-8 engineer-months of C++ work to answer — and may invalidate
the entire architecture if the answer is C.

Python Prototype Validation Plan

The cheapest way to test the four critical questions above is a
Python prototype using scipy and the existing Pyomo callbacks from
the equilibria repo. Estimated timeline:

Week 1 — skeleton (5 working days)

  * day 1: Fischer-Burmeister wrapper around an existing Pyomo
    callback_f(x). Sanity-check the FB residual gradient by
    finite-difference comparison against analytical formula.
  * day 2: variable/equation scaling layer ported from
    scripts/gtap_v62/_path_capi_solver.py. Both scaling and FB
    composed into a single R(x) operator.
  * day 3: scipy.optimize.root(method="krylov") integration. The
    LinearOperator interface for Jv (finite differences first,
    upgrade to forward-AD later if conditioning issues appear).
  * day 4: 1-variable toy MCP (0 <= x perp (x-2) >= 0). Verify
    convergence to x=2.
  * day 5: small Cottle-Pang test (3-variable LCP from MCPLIB).
    Verify FB+NK produces the documented solution.

Week 2 — first real CGE (5 days)

  * day 1-2: gtap6_3x3 baseline. Reproduce PATH's tc=1 r~1e-7
    result via FB+NK. This is the smallest non-trivial CGE — if
    FB+NK fails here, abort.
  * day 3: gtap6_3x3 shocked with 10% tariff. Compare VIWS to the
    GEMPACK +62.36% reference. Must be within 0.5pp.
  * day 4-5: block-Jacobi preconditioner via
    scipy.sparse.linalg.LinearOperator. Measure GMRES inner-iter
    reduction vs unpreconditioned.

Week 3 — scaling up (5 days)

  * day 1-2: gtap6_5x5 baseline + shocked. Parity check.
  * day 3-4: gtap6_10x7 baseline + shocked.
  * day 5: gtap6_15x10 baseline. Main convergence test: does
    Newton-Krylov on Fischer-Burmeister converge where PATH gets
    stuck at r=1.97e-5?

Week 4 — continuation framework + 20x41 (5 days)

  * day 1-2: predictor-corrector continuation for the shock
    parameter. Replace fixed substep ladder with adaptive step
    sizing based on residual reduction.
  * day 3-4: gtap6_15x10 shocked end-to-end with continuation.
    Success criterion: VIWS within 1% of GEMPACK reference.
  * day 5: gtap6_20x41 baseline attempt. This is the
    feasibility test for the larger dataset class — does the
    prototype even fit in memory and complete?

Estimated wall time for the prototype experiment

  Best case (no major surprises):    3 weeks (15 days)
  Expected:                          4-5 weeks (20-25 days)
  With one major issue:              6-8 weeks (30-40 days)

  Major issues that could extend timeline:

    * Fischer-Burmeister has bad conditioning around active
      constraints — would need to swap in semismooth Newton or
      Chen-Mangasarian smoothing (+1-2 weeks).
    * Block-Jacobi preconditioning insufficient — would need
      ILU(0) or higher fill-level (+1 week).
    * gtap6_20x41 does not fit in scipy memory budget — would
      need explicit out-of-core or distributed handling
      (+several weeks, possibly defer to C++ phase).

Decision points after the prototype

  * If the prototype solves gtap6_15x10 shocked: PEDRO's
    algorithmic premise is validated. Proceed to C++/CUDA work
    with confidence.
  * If the prototype solves baseline but not shocked: continuation
    framework or globalisation is the bottleneck. Phase 1 must be
    re-scoped to include those before C++ investment.
  * If the prototype fails even on baseline: Fischer-Burmeister +
    Newton-Krylov is not the right approach for CGE at this scale.
    Re-evaluate before committing to C++.

The prototype cost is roughly one engineer-month for a high
confidence answer to a question that otherwise needs 5-8 engineer-
months of C++ work to answer. It is the highest-leverage step in
the PEDRO roadmap.

Current Status

PEDRO is currently in early architectural design.

The immediate objective is:

correctness,
modularity,
and numerical stability.

Performance optimization comes after establishing a reliable nonlinear foundation.

Design Goals

PEDRO aims to become:

modular,
extensible,
HPC-oriented,
GPU-aware,
and suitable for very large-scale economic equilibrium models.

The project intentionally separates:

model definition,
nonlinear strategy,
linear algebra backend,
and hardware execution layer.

This separation enables future experimentation without redesigning the entire system.

License

TBD

Author

PEDRO Project
Research-oriented MCP/CGE solver architecture for modern hardware.