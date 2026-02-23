# Equilibria-GAMS Exact Replication Plan

## Objective
Achieve exact replication of the GAMS benchmark behavior in Equilibria templates, including:
- Variable-level parity at initialization (`*.l` in GAMS vs Python init state)
- Equation residual parity at benchmark (near machine precision)
- Consistent solver behavior after parity is established

## Scope and Constraints
- Introduce a strict **parity mode** distinct from equation-consistent fallback mode.
- In parity mode, avoid inferred defaults unless explicitly defined in GAMS.
- Require complete calibrated inputs; fail fast on missing required arrays.

## Work Plan

### 1. Freeze Parity Mode Scope
- Define parity mode semantics and flags.
- Disable fallback reconstruction logic in parity mode (e.g. inferred `sh0/tr0/TRO`, reconstructed `TIW/TIK`).
- Enforce strict required-input checks.

### 2. Build the GAMS-Python Mapping Matrix
- Create a complete mapping table:
  - `GAMS symbol -> Python variable -> source state key -> equation block`
- Include transfer direction conventions (`TR(a,b)` orientation).
- Document mapping in `docs/architecture/gams_parity_matrix.md`.

### 3. Lock Equations to Exact GAMS Forms
- Compare each implemented equation with the GAMS reference line-by-line.
- Replace any modified forms in parity mode with exact GAMS expressions and conditions (`$` guards).
- Prioritize blocks with highest historical drift:
  - Government and tax equations
  - Transfer equations
  - ROW/current-account equations
  - Price equations
  - GDP identities

### 4. Lock Initialization to GAMS `*.l`
- In parity mode, initialize from calibrated `*O` arrays exactly as in `model_solution.inc`.
- Avoid post-init recomputation unless the same operation appears in GAMS initialization.
- Maintain separate mode for equation-consistent initialization where needed.

### 5. Complete Data Availability in Calibrated State
- Ensure all required benchmark arrays are exported and accessible:
  - `TIWO`, `TIKO`, `TRO`, `SROWO`, and any other missing GAMS-level data
- Patch calibration modules to persist missing values with GAMS-consistent naming.

### 6. Add a Residual Parity Harness
- Add `scripts/parity/check_gams_parity.py` to:
  - Run parity-mode initialization
  - Compute and report full residual vector
  - Compare key initialized variables to `*O` levels with strict tolerance
- Output top residuals and parity mismatches in a deterministic report.

### 7. Define Acceptance Gates
- **Gate A: Variable parity**
  - All mapped benchmark values match within tight tolerance (e.g. `1e-9`)
- **Gate B: Equation parity**
  - Residual RMS near machine precision
  - No block-level outliers
- **Gate C: Solver parity readiness**
  - Only after A+B pass

### 8. Solver Parity Stage
- Align solve setup with GAMS conventions after equation/init parity is achieved:
  - Closure and fixed variables
  - Scaling/normalization
  - Objective/residual handling
- Compare baseline and one or more scenario runs against GAMS outputs.

### 9. CI and Regression Controls
- Add parity tests to CI:
  - Strict init parity test
  - Residual parity test
  - Baseline scenario parity smoke test
- Prevent future convention drift with mandatory parity checks.

## Execution Order (Milestones)
1. Parity mode scaffold + mapping matrix
2. Exact initialization path (`*.l` mirror)
3. Equation parity corrections + parity harness
4. Acceptance gates and CI
5. Solver parity and scenario validation

## Deliverables
- `docs/architecture/gams_parity_matrix.md`
- `scripts/parity/check_gams_parity.py`
- Parity mode implementation in solver/equation stack
- CI parity tests and regression reports
