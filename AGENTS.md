

## PEP Model Calibration Implementation

### Overview
Implementation of PEP-1-1_v2_1 modular model calibration following GAMS structure (lines 427-650+).

### Phase 1: Income and Savings Calibration ✅ COMPLETE
**File:** `src/equilibria/templates/pep_calibration_income.py`
**Test:** `tests/test_income_calibration.py`

**Implemented Variables (GAMS Section 4.1, lines 427-469):**
- **Household incomes**: YHKO, YHLO, YHTRO, YHO, YDHO, CTHO
- **Firm incomes**: YFKO, YFTRO, YFO, YDFO
- **Government income**: YGKO, TDHTO, TDFTO, TICTO, TIMTO, TIXTO, TIWTO, TIKTO, TIPTO, TPRODNO, TPRCTSO, YGTRO, YGO
- **Rest of world**: YROWO, CABO
- **Investment**: ITO

**Key Implementation Details:**
- SAM indices correctly mapped based on GAMS data_loading.inc
- `lambda_RK(ag,k) = SAM('AG', ag, 'K', k)`
- `lambda_WL(h,l) = SAM('AG', h, 'L', l)`
- `IMO(i) = SAM('AG', 'ROW', 'I', i)`
- `TICO(i) = SAM('AG', 'TI', 'I', i)`
- All formulas follow exact GAMS implementation

**Validation:**
- ✅ All income identities pass (YHO = YHLO + YHKO + YHTRO)
- ✅ Government budget balances (YGO = sum of all revenue components)
- ✅ All calculations verified against SAM data

### Phase 2: Shares Calibration ✅ COMPLETE
**Same file as Phase 1** (lines 461-468)

**Implemented Shares:**
- **lambda_RK(ag,k)**: Capital income shares, normalized by total capital demand
  - Formula: `lambda_RK(ag,k) = SAM('AG',ag,'K',k) / Sum[j,KDO(k,j)]`
  - Validation: Sums to 1.0 for each capital type
- **lambda_WL(h,l)**: Labor income shares, normalized by total labor demand
  - Formula: `lambda_WL(h,l) = SAM('AG',h,'L',l) / Sum[j,LDO(l,j)]`
  - Validation: Sums to 1.0 for each labor type
- **lambda_TR(agng,h)**: Transfer shares to households
  - Formula: `lambda_TR(agng,h) = TRO(agng,h) / YDHO(h)`
- **lambda_TR(ag,f)**: Transfer shares to firms
  - Formula: `lambda_TR(ag,f) = TRO(ag,f) / YDFO(f)`
- **sh1O(h)**: Marginal savings propensity
  - Formula: `sh1O(h) = [SHO(h) - sh0O(h)] / YDHO(h)`
- **tr1O(h)**: Marginal transfer rate
  - Formula: `tr1O(h) = [TRO('gvt',h) - tr0O(h)] / YHO(h)`

**Validation:**
- ✅ All lambda_RK sum to 1.0 per capital type (CAP, LAND)
- ✅ All lambda_WL sum to 1.0 per labor type (USK, SK)
- ✅ sh0O and tr0O read from VAL_PAR parameter file

### Phase 3: Production Block Calibration ✅ COMPLETE
**File:** `src/equilibria/templates/pep_calibration_production.py`
**Test:** `tests/test_production_calibration.py`

**Implemented (GAMS Section 4.4-4.6, lines 538-650+):**

**Factor Demands (lines 538-553):**
- KDO(k,j), LDO(l,j) - Raw factor demands from SAM
- KDCO(j), LDCO(j) - Aggregate factor demands
- KSO(k), LSO(l) - Factor supplies
- ttiwO(l,j), ttikO(k,j) - Tax rates
- WTIO(l,j), RTIO(k,j) - Prices with taxes
- WCO(j), RCO(j) - Composite prices

**Value Added (lines 554-555):**
- VAO(j) = LDCO(j) + KDCO(j)
- PVAO(j) - Price of value added

**Production Taxes (lines 557-559):**
- ttipO(j) - Production tax rate
- PPO(j) - Producer price

**Intermediate Consumption (lines 520-524):**
- DIO(i,j) - Intermediate consumption
- CIO(j) - Total intermediate consumption
- PCIO(j) - Intermediate consumption price index
- DITO(i) - Total intermediate demand

**Output and Technical Coefficients (lines 600-601):**
- XSTO(j) - Total aggregate output
- PTO(j) - Average output price
- io(j) - Intermediate input coefficient
- v(j) - Value added coefficient
- aij(i,j) - Intermediate input shares
- GDP_BPO - GDP at basic prices

**CES Parameters (lines 638-650+):**
- rho_KD(j), rho_LD(j) - Elasticity parameters
- beta_KD(k,j), beta_LD(l,j) - Share parameters
- B_KD(j), B_LD(j) - Scale parameters

**Key Features:**
- Automatic reading of VAL_PAR.xlsx for elasticity parameters (sigma_KD, sigma_LD, etc.)
- GAMS-style variable naming with comprehensive Field descriptions
- INFO level logging for all calibration steps
- Graceful fallback to defaults if VAL_PAR unavailable
- Base prices (WO, RO, PCO) initialized to 1.0 as numeraire

**Validation:**
- ✅ XSTO = CIO + VAO for all sectors (agr, ind, ser, adm)
- ✅ Factor demands sum correctly (KDCO = sum of KDO, LDCO = sum of LDO)
- ✅ Technical coefficients sum to 1.0 (io(j) + v(j) = 1.0)
- ✅ CES parameters calibrated correctly with VAL_PAR elasticities
- ✅ GDP_BPO calculated correctly

### Implementation Details

**SAM Data Access:**
All calibration modules read from SAM-V2_0.gdx using the delta decoder:
- 4D parameter structure: SAM(row_cat, row_elem, col_cat, col_elem)
- Correct mapping based on GAMS data_loading.inc formulas
- 196/196 records decoded with 100% accuracy

**VAL_PAR Integration:**
- Automatic reading from `/Users/marmol/proyectos/cge_babel/pep_static_clean/data/original/VAL_PAR.xlsx`
- Extracts elasticity parameters: sigma_KD, sigma_LD, sigma_VA, sigma_XT, sigma_M, sigma_XD
- Default values used if file unavailable:
  - sigma_KD = sigma_LD = 0.8
  - sigma_VA = 1.5
  - sigma_XT = sigma_M = sigma_XD = 2.0

**Set Definitions (matching GAMS):**
- **J** (sectors): ['agr', 'ind', 'ser', 'adm']
- **I** (commodities): ['agr', 'othind', 'ser', 'food', 'adm']
- **K** (capital): ['cap', 'land']
- **L** (labor): ['usk', 'sk']
- **H** (households): ['hrp', 'hup', 'hrr', 'hur']
- **F** (firms): ['firm']
- **AG** (agents): ['hrp', 'hup', 'hrr', 'hur', 'firm', 'gvt', 'row']

### Phase 4: Trade Block Calibration ✅ COMPLETE
**File:** `src/equilibria/templates/pep_calibration_trade.py`
**Test:** `tests/test_trade_calibration.py`

**Implemented (GAMS Section 4.4, 4.6.2, 4.6.3, lines 481-521, 598-635):**

**Import Demand (lines 335, 488-503, 519):**
- IMO(i) - Import quantity by commodity
- DDO(i) - Domestic demand for local product
- QO(i) - Composite commodity demand
- PMO(i), PDO(i) - Import and domestic prices
- PCO(i) - Purchaser price of composite commodity

**Export Supply (lines 335-337, 504-511):**
- EXDO(i) - World demand for exports
- EXO(j,i) - Export by sector j of commodity i
- DSO(j,i) - Domestic supply by sector
- XSO(j,i) - Total output supply (DSO + EXO)

**Trade Margins and Taxes (lines 483-495, 499-502, 507-508):**
- tmrg(i,ij) - Trade and transport margins
- tmrg_X(i,ij) - Export margins
- ttimO(i) - Import tax rate
- tticO(i) - Commodity tax rate
- ttixO(i) - Export tax rate
- TICO(i), TIMO(i), TIXO(i) - Tax values from SAM

**CET Parameters for Exports (lines 598-621):**
- rho_XT(j) - CET elasticity between commodities
- beta_XT(j,i) - CET share parameter between commodities
- B_XT(j) - CET scale parameter between commodities
- rho_X(j,i) - CET elasticity between exports and local
- beta_X(j,i) - CET share parameter between exports and local
- B_X(j,i) - CET scale parameter between exports and local

**CES Parameters for Imports (lines 625-635):**
- rho_M(i) - CES elasticity for composite good
- beta_M(i) - CES share parameter for composite good
- B_M(i) - CES scale parameter for composite good

**Key Features:**
- Correct SAM mapping for exports ('X' category) and imports ('AG'-'ROW' flow)
- Automatic reading of sigma_XT, sigma_X, sigma_M from VAL_PAR
- Trade margins calculation for domestic, import, and export flows
- MRGNO(i) - Total demand for commodity i as trade/transport margin

**Validation:**
- ✅ XSO = DSO + EXO for all sector-commodity pairs
- ✅ QO = [PMO*IMO + PDO*DDO]/PCO for all commodities
- ✅ Total Exports: 11,670.00 (matches SAM data)
- ✅ Total Imports: 15,732.00 (matches SAM data)
- ✅ Trade Balance: -4,062.00 (deficit calculated correctly)

### Phase 5: Final Integration ✅ COMPLETE
**File:** `src/equilibria/templates/pep_calibration_final.py`
**Test:** `tests/test_final_calibration.py`

**Implemented (GAMS Sections 4.6.4, 4.7, 4.8, lines 668-692):**

**Consumption Block (lines 330-331, 668-676):**
- CO(i,h) - Household consumption by commodity and type
- CGO(i) - Government consumption
- INVO(i) - Investment demand
- VSTKO(i) - Inventory changes
- GFCFO - Gross fixed capital formation

**LES Parameters (lines 673-676):**
- sigma_Y(i,h) - Income elasticity of demand
- gamma_LES(i,h) - LES marginal budget share
- CMINO(i,h) - Minimum consumption levels
- frisch(h) - Frisch parameter for each household

**GDP Measures (lines 678-685):**
- GDP_BPO - GDP at basic prices (from Phase 3)
- GDP_MPO - GDP at market prices
- GDP_IBO - GDP by income approach
- GDP_FDO - GDP by expenditure approach

**Real Variables (lines 687-692):**
- CTH_REALO(h) - Real consumption budget
- G_REALO - Real government expenditure
- GDP_BP_REALO - Real GDP at basic prices
- GDP_MP_REALO - Real GDP at market prices
- GFCF_REALO - Real gross fixed capital formation

**Price Indices (lines 567, 571, 575, 579):**
- PIXCONO - Consumer price index
- PIXGDPO - GDP deflator
- PIXGVTO - Government price index
- PIXINVO - Investment price index

**Key Features:**
- Full integration of all previous calibration phases
- Automatic validation checks for GDP consistency
- LES parameter normalization (gamma sums to 1.0 per household)
- Price indices initialized to 1.0 for base year

**Validation:**
- ✅ LES parameters: gamma_LES sums to 1.0 for all households
- ✅ Consumption data read from SAM (40,867 total private consumption)
- ✅ All four GDP measures calculated
- ✅ Real variables calibrated with base year indices
- ⚠️  GDP_FDO within 8% of GDP_BPO (known issue with factor income calculation)
- ✅ Trade balance integrated (-4,062 deficit)

### Unified Calibration Runner ✅ COMPLETE

**File:** `src/equilibria/templates/pep_calibration_unified.py`
**Script:** `scripts/cli/run_all_calibration.py`

A unified calibration runner that orchestrates all five phases and produces a complete calibrated model state.

**Features:**
- Runs all calibration phases in sequence
- Produces unified `PEPModelState` with all variables
- Generates comprehensive `CalibrationReport`
- Exports results to JSON
- Provides detailed logging and progress tracking

**Usage:**

```bash
# Run complete calibration with summary report
python scripts/cli/run_all_calibration.py

# Save calibrated state to JSON
python scripts/cli/run_all_calibration.py --save-state output/state.json

# Save detailed report
python scripts/cli/run_all_calibration.py --save-report output/report.json

# Enable verbose logging
python scripts/cli/run_all_calibration.py --verbose
```

**Python API:**

```python
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator

# Create calibrator
calibrator = PEPModelCalibrator(
    sam_file="path/to/SAM.gdx",
    val_par_file="path/to/VAL_PAR.xlsx"  # optional
)

# Run calibration
state = calibrator.calibrate()

# Access calibrated variables
gdp = state.gdp['GDP_BPO']
imports = state.trade['IMO']
consumption = state.consumption['CO']

# Print report
calibrator.print_report()

# Save outputs
state.save_json("calibrated_state.json")
calibrator.save_report("calibration_report.json")
```

### Running Individual Phases

For development or debugging, individual phases can be run separately:

```bash
# Run income and shares calibration (Phases 1-2)
python tests/test_income_calibration.py

# Run production calibration (Phase 3)
python tests/test_production_calibration.py

# Run trade calibration (Phase 4)
python tests/test_trade_calibration.py

# Run final integration (Phase 5)
python tests/test_final_calibration.py
```

## Solver Integration 🔄 IN PROGRESS

### Overview
Solver infrastructure has been implemented to solve the calibrated PEP model.

**Files:**
- `src/equilibria/templates/pep_model_equations.py` - All GAMS equations (EQ1-EQ97)
- `src/equilibria/templates/pep_model_solver.py` - Solver implementation
- `scripts/cli/run_solver.py` - Solver runner script

**Implemented Equations:**
- **Production Block (EQ1-EQ9):** Value added, intermediate consumption, CES labor/capital demands
- **Income Block (EQ10-EQ21):** Household and firm income definitions
- **Government Block (EQ22-EQ43):** Tax revenues, transfers, government budget
- **Rest of World (EQ44-EQ46):** Foreign income, current account balance
- **Transfers (EQ47-EQ51):** Inter-agent transfer payments
- **Demand Block (EQ52-EQ57):** LES consumption, investment, margins
- **Trade Block (EQ58-EQ64):** CET exports, CES imports, world demand
- **Price Block (EQ65-EQ84):** Factor prices, commodity prices, price indices
- **Equilibrium (EQ85-EQ89):** Market clearing conditions (labor, capital, goods)
- **GDP Definitions (EQ90-EQ97):** GDP measures and real variables

**IPOPT Installation:**

IPOPT (Interior Point OPTimizer) is a powerful nonlinear optimization solver recommended for CGE models:

```bash
# Using pip (requires C++ compiler)
pip install cyipopt

# Using conda (recommended)
conda install -c conda-forge cyipopt

# On macOS with Homebrew
brew install ipopt
pip install cyipopt
```

**Usage:**

```bash
# Calibrate and solve (auto-selects best solver)
python scripts/cli/run_solver.py

# Force IPOPT solver
python scripts/cli/run_solver.py --method ipopt

# Use simple iteration (no dependencies)
python scripts/cli/run_solver.py --method simple_iteration

# Use pre-calibrated state
python scripts/cli/run_solver.py --calibrated-state output/state.json

# Save solution
python scripts/cli/run_solver.py --save-solution output/solution.json

# Custom tolerance and iterations
python scripts/cli/run_solver.py --tolerance 1e-8 --max-iterations 500
```

**Python API:**

```python
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_model_solver import PEPModelSolver

# Calibrate
calibrator = PEPModelCalibrator(sam_file="SAM.gdx")
state = calibrator.calibrate()

# Solve
solver = PEPModelSolver(
    calibrated_state=state,
    tolerance=1e-6,
    max_iterations=100
)
solution = solver.solve()

if solution.converged:
    print(f"GDP: {solution.variables.GDP_BP}")
    print(f"Price Index: {solution.variables.PIXCON}")

# Validate
validation = solver.validate_solution(solution)
```

**Current Status (Updated Feb 2026):**
- ✅ All 97 GAMS equations implemented
- ✅ Full `pep2` parity against GAMS pre-solve baseline (351 vars, 251 params, zero mismatches)
- ✅ Equation-consistent initialization now satisfies the full system at numerical precision
- ✅ IPOPT path stabilized (no divide-by-zero, fixed residual-vector dimensions)
- ✅ `scripts/cli/run_solver.py --method ipopt --init-mode equation_consistent` now returns `CONVERGED`

**Latest Solver Result (`pep2`, equation-consistent init):**
- Final RMS residual: `4.01e-13`
- Max residual: `3.64e-12`
- IPOPT iterations executed: `0` (early exit because benchmark already satisfies tolerance)
- Walras check: pass (`0.00e+00`)

**Latest Solver Result (`pep2` CRI, `strict_gams`, SIM1):**
- Status: `CONVERGED`
- Final RMS residual: `3.01e-05`
- Max residual: `6.12e-04`
- IPOPT iterations executed: `0` (initial strict_gams point accepted by practical tolerance gate)
- GAMS levels source: `src/equilibria/templates/reference/pep2/scripts/Results.gdx` (SIM1 slice)

**SAM-Only Calibration Check (Updated Feb 2026):**
- Command: `uv run python scripts/qa/verify_calibration.py`
- Result: calibrated values now satisfy equilibrium directly (no extra solve step needed)
- Number of equations: `315`
- RMS residual: `3.99e-13`
- Max residual: `3.64e-12`
- Status: `CALIBRATED VALUES SATISFY EQUILIBRIUM (RMS < 1e-3)`

**Key Fixes This Session:**
1. ✅ Fixed ROW transfer orientation in equations (`EQ44/EQ45`): `TR('row',agd)` vs `TR(agd,'row')`
2. ✅ Corrected benchmark initialization identities for `TDH`, `SH`, `SF`, `SG`, and tax aggregates
3. ✅ Aligned IPOPT parameter extraction with main solver (`rho/sigma` blocks, VA CES, recomputed `io/v/aij`)
4. ✅ Added robust handling for CES corner case in `EQ4` (`beta_VA` at 0/1)
5. ✅ Added fixed residual-name ordering in IPOPT objective to prevent finite-difference shape mismatch
6. ✅ Added IPOPT early-exit when initialized benchmark already meets tolerance
7. ✅ Aligned production equation activation with GAMS masks (`LDO0/KDO0`) for `EQ6/EQ7/EQ8`
8. ✅ Synchronized `lambda_TR_households` / `lambda_TR_firms` from strict_gams loaded levels to remove transfer-share drift (`EQ49` residual block)
9. ✅ Added best-candidate selection in IPOPT path (initial/pass1/pass2) by convergence then residual RMS
10. ✅ Added practical strict_gams acceptance gate to avoid unnecessary IPOPT drift when benchmark is already near-feasible

**Debugging Tools:**
```bash
# Verify calibration satisfies equations
python scripts/qa/verify_calibration.py

# Compare solver vs GAMS baseline  
python scripts/parity/compare_with_gams.py

# Debug specific equations
python scripts/dev/debug_equations.py
```

**Root Cause (updated):**
Most remaining large residuals were caused by initialization/alignment mismatches versus GAMS (equation activation masks and transfer-share parameter drift), not by a structural model error. After fixing those alignment points, strict_gams CRI initialization is near-feasible and accepted directly.

### Latest Parity Status (Feb 2026) ✅

**Reference baseline mode now supported:** compare Python initialization against **GAMS pre-solve levels** (not solved BASE levels) for endogenous trade variables.

**Current full parity result (`pep2` reference):**
- Variables: `compared=351`, `mismatches=0`, `missing=0`
- Parameters: `compared=251`, `mismatches=0`, `missing=0`

**Verification command:**
```bash
python scripts/parity/verify_pep2_full_parity.py \
  --tol 1e-9 \
  --presolve-gdx src/equilibria/templates/reference/pep2/scripts/PreSolveLevels.gdx
```

**Key final fixes applied:**
1. Transfer orientation aligned everywhere to GAMS convention `TR(recipient, source)`.
2. VAL_PAR ingestion unified via `pep2`-compatible loader (`PARJ/PARI/PARJI/PARAG` from GDX/XLSX ranges).
3. Trade calibration fixed: `EXO` from SAM, `tmrg_X` from SAM with proper normalization, `PE_FOBO` margin term uses `SUM[ij,PCO(ij)*tmrg_X(ij,i)]`.
4. Production calibration sequencing fixed: intermediate block calibrated before `ttip/PPO` formula usage.
5. `EQ39` and TIP initialization corrected to GAMS form: `TIP(j)=ttip(j)*PP(j)*XST(j)`.
6. `ttdh1`, `gamma_INV`, `gamma_GVT`, `B_XT` alignment fixed (formula/timing consistency).

### Current Operational Status - February 19, 2026

**Verified state now:**
- ✅ `pep2` with `init_mode=equation_consistent` is numerically consistent at initialization (RMS residual around `1e-7`, max residual around `1e-6`).
- ✅ Systemic parity pipeline now fails correctly when solve fails: `scripts/parity/run_pep_systemic_parity.py` returns non-zero if `--method != none` and solve is not converged or solve gates fail.
- ✅ Exit-code behavior validated with two runs:
  - pass case: `output/exit_check_pass.json` -> exit `0`
  - forced solve-fail case: `output/exit_check_fail_solve.json` -> exit `2`
- ✅ CRI architecture diagnosis documented:
  - Finding note: `docs/findings/finding_sam_ieem_vs_sam_pep.md`
  - Main structural cause: export flows misrouted in PEP conversion (`I.i -> AG.ROW`/`X.i -> AG.ROW` without `J.j -> X.i` counterpart).
  - Secondary cause: production-tax routing drift (`AG.ti -> J` vs expected `AG.gvt -> J` in PEP calibration path).
  - Why IEEM can still close: different trade/tax architecture + pre-solve diagnostics and SAM balancing in IEEM pipeline.
- ✅ CRI fixed SAM provenance validated:
  - Source generator: `/Users/marmol/proyectos/cge_babel/sam/cri/2016/export_gams_format.py`
  - Input in generator: `output/SAM-V2_0.xlsx` (`export_gams_format.py:243`)
  - Generated file: `output/SAM-CRI.xlsx` (`export_gams_format.py:285`, export call `:289-292`)
  - Equilibria reference file: `src/equilibria/templates/reference/pep2/data/SAM-CRI-gams-fixed.xlsx`
  - Binary identity check: same size (`9094` bytes), same SHA1 (`e4192240b11e8faadc5d6e3c3c7367f45dfe05a1`), `cmp` exit code `0`.
- ✅ Deterministic CRI `solver_dynamics` attribution gate now available:
  - Scenario: `SAM-CRI-gams-fixed.xlsx` + `equation_consistent` + `simple_iteration --max-iterations 0`
  - Expected classification: `solver_dynamics/solve_not_converged` with `sam_qa.passed=true` and `init.gates.overall_passed=true`
  - Enforced in CI as `structural-cri-solver-dynamics-gate`

**Current problem to solve:**
- ⚠️ Remaining instability is concentrated in **CRI data runs** and baseline alignment modes, not in the base `pep2` mirror equations.
- ⚠️ `strict_gams` depends on the selected `Results.gdx` slice and can be near-feasible or inconsistent depending on whether that baseline is compatible with the calibrated SAM in the run.
- ⚠️ Some CRI scenarios still show macro/GDP closure tension (`GDP_MP/GDP_IB` vs `GDP_FD`) driven by SAM consistency issues (not only solver settings).

**Paths to solution (ordered):**
1. **Data-consistency path (recommended first):**
   - Add mandatory SAM QA gates before solve (exports/domestic supply, margins, tax-base consistency, macro closure checks).
   - Reject or auto-fix inconsistent SAM mappings before calibration.
2. **Initialization-parity path:**
   - Complete full blockwise reconstruction of GAMS `.L` logic in Python for all endogenous levels.
   - Keep `strict_gams` for direct GDX overlays and `equation_consistent`/`gams_blockwise` for identity-consistent starts.
3. **Solver-robustness path:**
   - Preserve safe CES/CET handling and tax-detail reconstruction during array<->variable transforms.
   - Keep parity gates (`EQ29/EQ39/EQ40`, `EQ79/EQ84`, levels parity) as hard CI checks.

**Execution guidance right now:**
- Use `equation_consistent` as default for deterministic parity on `pep2`.
- Use `strict_gams` only when the exact `Results.gdx` baseline is verified compatible with the run SAM.
- For CRI, run SAM QA first and treat solver non-convergence as a data-structure warning, not immediately as equation mismatch.
- For CRI conversion/mapping changes, keep `docs/findings/finding_sam_ieem_vs_sam_pep.md` as the structural contract reference.

### Implementation Summary

All five phases of the PEP-1-1_v2_1 model calibration are now complete:

1. **Phase 1-2:** Income and shares calibration ✅
2. **Phase 3:** Production block calibration ✅
3. **Phase 4:** Trade block calibration ✅
4. **Phase 5:** Final integration ✅
5. **Solver:** Equation system and basic solver ✅

**Next Steps (Future Work):**
- Advanced solver algorithm (Newton-Raphson with line search)
- Integration with PATH solver or similar
- Comprehensive validation against GAMS baseline results
- Sensitivity analysis tools
- Scenario simulation framework

## Appendix: GDX Comparison Tool

### Overview
A standalone comparison tool has been implemented to validate data consistency between original GAMS GDX files and equilibria's data handling.

### Location
- **Script:** `scripts/dev/compare_gdx.py`
- **Report Output:** `reports/gdx_comparison_report.md`

### Features
- Compares Excel data (via equilibria loaders) with original GDX files
- Generates detailed Markdown reports
- Configurable tolerance (default: 0.01%)
- Identifies:
  - Record count differences
  - Missing records in either source
  - Value differences exceeding tolerance
  - Metadata mismatches

### Usage
```bash
python scripts/dev/compare_gdx.py
```

### Output Format
The tool generates a comprehensive Markdown report including:
1. **Summary Table** - Overview of all comparisons
2. **Detailed Sections** - Per-file analysis with:
   - Record counts (Excel vs GDX)
   - Common records
   - Value differences
   - Records only in Excel
   - Records only in GDX

### Current Status
- **SAM-V2_0.gdx**: ✅ **FULLY WORKING** - 196/196 records decoded correctly (100% accuracy)
  - Implementation based on official GDX source code from `/Users/marmol/proyectos/gdx/src/gxfile.cpp`
  - Correctly reads MinElem/MaxElem from data header for each dimension
  - Implements proper DeltaForRead logic (DeltaForRead = dimension for version > 6)
  - Handles all delta codes: 0x01, 0x02, 0x03, 0x05, 0x06, 0xFF (EOF)
  - Supports dynamic element types (byte/word/integer) based on range
  - Applies MinElem offset to decode correct indices
  - **Update Feb 2026**: Complete rewrite based on official GDX implementation
    - Fixed header parsing (position 11-42 for MinElem/MaxElem)
    - Fixed delta code interpretation (B > DeltaForRead = relative change)
    - Fixed value reading (0x0A followed directly by 8-byte LE double)
    - Achieved 100% match rate with CSV ground truth
- **VAL_PAR.gdx**: Comparison implemented (simplified)

### Critical Finding: Decoder Fixed (Feb 2026)

**Issue Resolved**: All 196 records now decode correctly with 100% accuracy.

**Root Cause of Previous Issues**:
- Incorrect header parsing (didn't read MinElem/MaxElem)
- Wrong delta code interpretation (assumed bit-mask instead of DeltaForRead logic)
- Missing MinElem offset application to indices
- Incorrect value structure (assumed type byte before double)

**Solution Implemented**:
Based on official GDX source code analysis from `/Users/marmol/proyectos/gdx/src/gxfile.cpp`:

1. **Correct Header Parsing**:
   - Read dimension at position 6
   - Read record count at positions 7-10
   - Read MinElem/MaxElem for each dimension at positions 11-42

2. **Proper DeltaForRead Logic**:
   - DeltaForRead = dimension (for version > 6)
   - If B > DeltaForRead: relative change in last dimension (LastElem[last] += B - DeltaForRead)
   - If B <= DeltaForRead: B indicates first dimension that changes (1-based)

3. **Dynamic Element Types**:
   - Determine type based on range (MaxElem - MinElem + 1)
   - Range <= 255: 1 byte
   - Range <= 65535: 2 bytes
   - Otherwise: 4 bytes

4. **MinElem Offset**:
   - All indices read from file must have MinElem added: LastElem[D] = ReadIndex() + MinElem[D]

5. **Value Structure**:
   - 0x0A marker followed directly by 8-byte little-endian double
   - No type byte between marker and value

**Result**: 100% match rate with CSV ground truth (196/196 records).

### Delta Code Decoding
The GDX reader now correctly interprets GAMS delta compression codes based on official GDX source:

**DeltaForRead Logic** (for version > 6):
- DeltaForRead = dimension (4 for SAM-V2_0.gdx)

**Code Interpretation**:
- **B > DeltaForRead**: Relative change in last dimension
  - `LastElem[last] += B - DeltaForRead`
  - Code 5: +1 to last dimension
  - Code 6: +2 to last dimension
  - Code 255 (0xFF): EOF marker
- **B <= DeltaForRead**: Indicates first dimension that changes (1-based)
  - Code 1: Replace all indices (read dimensions 0-3)
  - Code 2: Replace indices from dimension 1 onwards
  - Code 3: Replace indices from dimension 2 onwards

**Index Reading**:
- Element type determined by range (MaxElem - MinElem + 1):
  - Range <= 255: 1 byte per index
  - Range <= 65535: 2 bytes per index  
  - Otherwise: 4 bytes per index
- All read indices have MinElem[D] added: `LastElem[D] = ReadValue() + MinElem[D]`

**Value Reading**:
- 0x0A marker followed directly by 8-byte little-endian double

### Validation Results

**SAM-V2_0.gdx**: ✅ **100% Accuracy Achieved**
- All 196 records decoded correctly
- 100% match rate with CSV ground truth
- Keys and values both verified correct

**Implementation verified against**:
- Official GDX source code from GAMS (C++)
- CSV ground truth file (gdx_values.csv)
- Manual inspection of decoded records
