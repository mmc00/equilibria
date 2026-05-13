# GTAP Welfare Decomposition (cgebox extras) — Implementation Plan

**Date:** 2026-05-13
**Status:** Draft — pending green-light to implement
**Reference impl:** `/Users/marmol/proyectos2/cge_babel/gtap/cgebox/gams/postModel/` (W. Britz, 2016)
**Already shipped (PR #6, commit `88685e6`):** Huff (1996)/McDougall (2003) first-order
welfare decomposition + N-segment homotopy + `WELVIEW.har` writer.

## Why this plan

`equilibria` already implements the *standard* RunGTAP-style welfare
decomposition: 11 allocative buckets (ALLOC), terms-of-trade (T),
investment-savings (IS), endowment (ENDW), technical change (TECH),
written to `WELVIEW.har` for RunGTAP compatibility.

The **cgebox** project adds three layers on top that are not (yet) in
equilibria:

1. **Money-metric (Exact CV)** decomposition using the calibrated
   utility function — not just the first-order Huff approximation.
2. **Per-product price-change decomposition** — splits welfare effects
   by individual commodity and by source (domestic vs import vs
   bilateral Armington top).
3. **Per-factor income decomposition** — splits the "endowment" effect
   into contributions of each factor (skilled labour, unskilled, land,
   natural resources, capital).

These are useful for *policy-attribution* analyses where the user
wants to say "the tariff cut raised welfare by $X M, of which $Y came
from cheaper imports of c_TextWapp from EastAsia and $Z from the
reallocation of unskilled labour."

The cgebox reference also has a simpler "GTAP-style" tax-stream
decomposition (`EVDecomp.gms`) that complements the Huff one — useful
when the user just wants per-tax revenue impact attribution.

## Reference implementation (cgebox)

Three GAMS files in `cge_babel/gtap/cgebox/gams/postModel/`:

### 1. `EVDecomp.gms` — simple GTAP-style decomposition (57 lines)

Decomposes EV by **source of nominal income change**:

```gams
p_deltaEV(rsNat,"xft",%arg2%) = sum((f,a),  pf*xf - pf("t0")*xf("t0"));
p_deltaEV(rsNat,"depr",%arg2%) = valDep("t0") - valDep(%arg2%);
p_deltaEV(rsNat,gy,%arg2%)     = yTax(gy,%arg2%) - yTax(gy,"t0");
p_deltaEV(rsNat,"t.o.t",%arg2%) = valFobCif(%arg2%) - valFobCif("t0");
p_deltaEV(rsNat,"tot",%arg2%)  = sum(evEff, p_deltaEV(evEff,%arg2%));
```

Components (`evEff`):
- `xft` — factor income change (sum over f, a of `pf·xf`).
- `depr` — capital depreciation (sign: −ΔDep).
- `gy` (10 streams) — per-tax-stream revenue change.
- `t.o.t` — terms-of-trade gain (ΔFOB − ΔCIF).
- `tot` — sum of the above (additive, not multiplicative).

This is **not** the Huff decomposition. It is a balance-sheet
accounting of nominal income change. Equilibria can produce this
trivially from already-collected `ytax(r, gy)` + factor incomes + tariff
revenue series.

### 2. `welfDecomp.gms` — money-metric decomposition (583 lines)

Three blocks. Uses a **CD utility function** so the expenditure function
inverts in closed form. Calls helper `calcExp.gms` to compute the money
metric `u.l` at each intermediate price vector.

#### Block I — Price indices and income components (lines ~1-200)

Computes utility-consistent price indices for households (`uh`),
government (`ug`), savings (`us`) via CD price-elasticity-of-1
inversion. Decomposes regional income by:

1. **VA channel** — `pf·xf` for each factor (loop `f0`).
2. **Depreciation** — `Δ(p_fdepr · pi · kstock)`.
3. **Indirect taxes** — sum over `gy` tax streams (`xt`).
4. **Per-tax-stream gy** — same 10 streams as `EVDecomp.gms`.
5. **Per-factor-f0** — splits VA channel by factor (the per-factor
   ENDW decomposition).

After each component, calls `calcExp.gms` to update `u.l` and record
the money-metric change attributable to that income source.

#### Block II — Per-product price changes (lines ~200-500)

For each product `j` and each price orientation:

- `"tot"` — Armington composite price `pa(r, i, aa)`.
- `"dms"` — domestic component `pd(r, i)`.
- `"imp"` — import composite `pmt(r, i)` or bilateral CIF `pmcif`.

Sequentially perturbs each price from its base level to its shock
level, re-evaluates the CES Armington nest:

```
pa(r,i,aa) = ((1-shr) · pd^(1-σ) + shr · pmt^(1-σ))^(1/(1-σ))
```

Re-aggregates through `pdNest`, `pcons`, `pg`, `psave` according to
the active demand system (LES / CDE / AIDADS). Each step calls
`calcExp.gms` to record the money-metric delta.

This is the **path-dependent** part: results depend on the order of
perturbation. The cgebox convention is alphabetical by product within
each orientation; total residual is reported separately.

#### Block III — Aggregation and residual (lines ~500-583)

Aggregates Block II contributions over products. Computes residual:

```
residual = ΔEV_total − Σ_block_I − Σ_block_II
```

The residual captures **interaction terms** (e.g. simultaneous changes
in `pa` and factor income that the sequential decomposition cannot
disentangle). Typically <5% of |ΔEV|.

### 3. `welfare_reports.gms` — orchestration (285 lines)

Wraps both decomps. Key outputs:

- **Tax positions** (`incPos`): `totf`, `vb`, `o`, `fd`, `fi`, `f`,
  `pd`, `pi`, `gd`, `gi`, `xs`, `ms`, `emis`.
- **GDP variants**: `vaGDP` (production-side via factor incomes) vs
  `r_GDP` (consumption-side `Σ pa·xa`).
- **Beta share recalculation**: `betap = yc/regy`, `betas = rsav/regy`,
  `betag = yg/regy` (post-solve).
- **Multipliers**: `u`, `p_results("au", ...)` (welfare per unit
  consumption).
- **Dispatch**: calls `welfDecomp.gms` when `welfDecomp=on`, else
  `EVDecomp.gms`.

## What equilibria already has

Shipped in PR #6 (`88685e6`):

- `src/equilibria/templates/gtap/welfare_decomp.py`
  - `compute_welfare_decomposition(base, shock, *, homotopy=N)`.
  - 11 ALLOC buckets + T + IS + ENDW + TECH per region.
  - Returns `WelfareComponents` dataclass (EV, A, T, IS, ENDW, TECH, residual).
  - N-segment homotopy for exact RunGTAP equivalence.

- `src/equilibria/templates/gtap/welfare_decomp_har.py`
  - GEMPACK HAR writer (EVAL, ALET, ALEF, TOTE, ISE, ENDW, TECH, TOT).
  - Round-tripped against `babel.har.reader` for byte-correctness.

- 17 unit + integration tests (`tests/templates/gtap/test_welfare_decomp.py`).

**Not shipped:**
- Per-product price-change decomposition (cgebox Block II).
- Per-factor income decomposition (cgebox Block I, factor split).
- CD money-metric expenditure function (cgebox `calcExp.gms`).
- GTAP-style EV decomposition by tax stream + factor income +
  depreciation + T.o.T. (cgebox `EVDecomp.gms`).

## Equilibria port plan

Three pieces. Conceptually orthogonal to the existing Huff decomp — no
changes to the equation system, only new post-solve modules.

### Piece A — `ev_decomp_simple.py` (~150 LoC)

Translate `EVDecomp.gms`. Cheapest piece, useful in its own right.

```python
from equilibria.templates.gtap.welfare_decomp import compute_ev_simple

decomp = compute_ev_simple(base_model, shock_model)
# Returns dict[region, EVSimpleComponents] with:
#   xft (factor income), depr (depreciation),
#   gy_<stream> (per-tax-stream, 10 streams),
#   tot (terms-of-trade), total
```

Inputs from solved Pyomo:
- `pf[r,f,a].value`, `xf[r,f,a].value` (factor incomes).
- `pi[r].value`, `kstock[r]` parameter, `p_fdepr[r]` parameter.
- `ytax(r, gy)` per-stream (already collected for `ytax` parity).
- FOB/CIF totals: `valFob = Σ pe·xw`, `valCif = Σ pmcif·xw`.

No CD inversion needed; this is purely accounting. Validate against
cgebox by writing a tiny shock through both pipelines.

### Piece B — `ev_decomp_per_factor.py` (~200 LoC)

Adds the per-factor split of the ENDW bucket already in the Huff
decomp. Output:

```python
@dataclass
class FactorEndowmentSplit:
    by_factor: dict[str, float]   # SkLab, UnSkLab, Land, NatRes, Capital
    total: float                   # = sum(by_factor.values())
```

Used to attribute ENDW to specific factor markets. No CD inversion;
trivial:

```
contrib(f) = Σ_a (pf_shock·xf_shock − pf_base·xf_base) for that f
```

This is information equilibria already has internally (the Huff ENDW
bucket is computed by summing over `f`), just not surfaced. Small
refactor of `welfare_decomp.py` to expose per-`f` contributions in
addition to the scalar `ENDW`. Add columns to `WELVIEW.har` (`ENDF`,
2D over FACTOR × REG).

### Piece C — `ev_decomp_money_metric.py` (~500 LoC)

The hard piece. Translates `welfDecomp.gms` Blocks I + II + III.

#### C.1 Calibrated utility function (`calc_exp.py`, ~120 LoC)

Direct translation of `calcExp.gms`. CD case:

```python
def cd_money_metric(p_base, p_shock, x_base, alpha):
    """Money-metric utility under CD."""
    eb = np.prod(p_base ** alpha)
    es = np.prod(p_shock ** alpha)
    return (es / eb) * np.dot(p_base, x_base)
```

LES / CDE / AIDADS variants follow the cgebox structure. For the
**initial** port, support only CD (matches altertax preset; sufficient
for the headline use case).

#### C.2 Block I — income components (`block1_income.py`, ~150 LoC)

Sequentially perturb income channels (VA → depr → tax_stream_1 → …),
recomputing money metric at each step. Each step's delta is that
component's contribution.

#### C.3 Block II — per-product price decomposition (`block2_prices.py`, ~200 LoC)

For each product j:
- Compute money-metric delta when only `pa(*, j, *)` moves to shock.
- Re-evaluate Armington nest with intermediate prices.
- Three orientations (tot/dms/imp).
- Aggregate over j.

This is `O(|R| · |I| · 3)` calls to `cd_money_metric`. For 9x10:
9 × 10 × 3 = 270 calls per region. Cheap.

#### C.4 Block III — aggregation + residual (`block3_aggregate.py`, ~50 LoC)

Sum, compute residual, write to a new HAR file `WELDETAIL.har` (parallel
to `WELVIEW.har` but with per-product/per-factor 2D tables).

### Piece D — CLI + tests (~150 LoC)

Extend `scripts/gtap/run_gtap.py welfare-decomp`:

```bash
# Existing: Huff decomp
uv run python scripts/gtap/run_gtap.py welfare-decomp \
    --base ... --shock ... --output reports/welview.har

# New: + simple EV decomp + per-factor + money-metric
uv run python scripts/gtap/run_gtap.py welfare-decomp \
    --base ... --shock ... \
    --output reports/welview.har \
    --include-ev-simple \
    --include-per-factor \
    --include-money-metric \
    --detail-output reports/weldetail.har
```

Tests (`tests/templates/gtap/test_welfare_decomp_extended.py`):

1. **EV-simple additivity** — `sum(components) ≈ EV_huff` within 5%
   for a 10% tariff shock (the two decomps are not identical but
   should be close in magnitude).

2. **Per-factor sum** — `sum_f ENDF(f, r) == ENDW(r)` exactly.

3. **CD money metric vs Huff** — for a small shock (1%), the CD
   money-metric EV should agree with the Huff first-order EV within
   <1% (first-order Taylor is exact in the limit).

4. **Block II + residual** — `sum_j Block2(j) + residual ≈ ΔEV_total`
   exactly (by construction).

5. **HAR round-trip** — write `WELDETAIL.har`, read back, byte-equal.

## Open questions for the implementer

1. **Demand system coverage.** Cgebox supports CD/LES/CDE/AIDADS in
   `calcExp.gms`. Equilibria's standard GTAP datasets use CDE for
   households. Decision: initial port CD-only; document the gap; add
   CDE in a follow-up if needed for production policy work.

2. **Path dependence of Block II.** The cgebox convention is
   alphabetical product order. Equilibria should document this and
   expose the order as a parameter so users can verify residual size
   under alternative orderings.

3. **Naming conflict with existing `WELVIEW.har`.** Existing file has
   headers `EVAL, ALET, ALEF, TOTE, ISE, ENDW, TECH, TOT`. Per-factor
   ENDF would extend this. Per-product / money-metric would go to a
   separate `WELDETAIL.har` to keep RunGTAP compatibility intact.

4. **Whether to deprecate `EV_simple` once money-metric exists.** No —
   they answer different questions. Simple = balance-sheet
   attribution; money-metric = utility-consistent valuation. Keep
   both.

5. **GAMS reference run for validation.** Cgebox altertax run produces
   the decomposition output. We need an equivalent reference run via
   NEOS (or local GAMS for NUS333 size) to cell-by-cell validate. Use
   the same `9x10` baseline + 10% tariff shock as PR #6.

## Implementation order (suggested)

1. **Piece A** alone — simple EV decomp. Cheap (~150 LoC), useful
   immediately, doesn't touch any existing code. (~1 day)
2. **Piece B** — per-factor split. Trivial refactor + HAR addition.
   (~half day)
3. **Piece C.1 + C.2** — CD money metric + Block I. Validates the
   utility-function infrastructure on the income side first. (~2 days)
4. **Piece C.3 + C.4** — Block II + III. Hardest piece, builds on C.1.
   (~2 days)
5. **Piece D** — CLI + tests. End-to-end. (~1 day)

Total: ~1 week for full cgebox welfare parity (CD only).

CDE / LES / AIDADS demand-system support would be a separate PR
(~1 week each).

## Validation strategy

1. Reuse the cgebox 9x10 (or NUS333) altertax run from the altertax
   plan, with `welfDecomp=on`. Produces reference per-product /
   per-factor / money-metric decomposition.
2. Run equilibria extended welfare-decomp on the same shock.
3. Cell-by-cell diff:
   - `ENDF(f, r)`: tolerance `1e-3` USD M.
   - `EV_simple(component, r)`: tolerance `1e-3` USD M.
   - `Block2(j, r)`: tolerance `1e-2` USD M (path-dependence noise).
   - `residual(r)`: must be < 5% of |ΔEV|.
4. Total `EV` should match Huff within `1e-4` and money-metric within
   `1e-2` (for moderate shocks).

If `cgebox` GAMS license isn't sufficient, fall back to NEOS.

## Files to be added

```
src/equilibria/templates/gtap/
├── welfare_decomp_extended/
│   ├── __init__.py
│   ├── ev_simple.py           # Piece A
│   ├── per_factor.py          # Piece B
│   ├── calc_exp.py            # Piece C.1 — utility function
│   ├── block1_income.py       # Piece C.2
│   ├── block2_prices.py       # Piece C.3
│   ├── block3_aggregate.py    # Piece C.4
│   ├── weldetail_har.py       # WELDETAIL.har writer
│   └── README.md
└── welfare_decomp.py          # Refactor: expose per-factor in ENDW

scripts/gtap/run_gtap.py       # Piece D — extended CLI flags

tests/templates/gtap/
└── test_welfare_decomp_extended.py

docs/site/guide/
└── welfare_decomposition.md   # Add "Extended decomposition" section
```

## Acceptance criteria

- [ ] `compute_ev_simple(base, shock)` returns 13-component dict
      per region (xft, depr, 10 gy streams, t.o.t, total).
- [ ] `compute_welfare_decomposition(..., include_per_factor=True)`
      returns per-factor ENDW breakdown.
- [ ] `compute_money_metric_decomposition(...)` returns Block I +
      Block II + residual under CD utility.
- [ ] `WELDETAIL.har` writes/reads cleanly (round-trip parity).
- [ ] EV-simple components sum to within 5% of Huff EV.
- [ ] Per-factor ENDF sums exactly to scalar ENDW.
- [ ] CD money-metric matches Huff EV within 1% for small shocks.
- [ ] Block II + residual = ΔEV_total exactly.
- [ ] CLI flags `--include-ev-simple`, `--include-per-factor`,
      `--include-money-metric` work end-to-end on 9x10 and NUS333.
- [ ] Docs: extended-decomp section in `welfare_decomposition.md`.

## Relationship to altertax plan

These two plans share infrastructure:

- **CD utility** (this plan, Piece C.1) — same elasticity preset as
  altertax (`esubva=1`, `esubinv=1`, factor-mobility-ω=1).
  Implementation could share a `cd_utility.py` module.
- **`p_scaleRGdpMp`** — altertax writes it; welfare-decomp reads it
  for real-GDP-scaled welfare reporting.
- Both build on solved Pyomo state from a single shock run, so a CLI
  could call both back-to-back:

  ```bash
  uv run python scripts/gtap/run_gtap.py shock-and-rebalance \
      --shock ... \
      --altertax-output reports/9x10_alttax.har \
      --welfare-output reports/welview.har \
      --welfare-detail-output reports/weldetail.har
  ```

## References

- Huff, K. and McDougall, R. (1996/2003), "Decomposing Welfare Changes
  in the GTAP Model," GTAP Technical Paper 5.
- Britz, W. (2016+), CGEBOX project — `cge_babel/gtap/cgebox/gams/postModel/`.
  - `EVDecomp.gms` — 57 lines.
  - `welfDecomp.gms` — 583 lines.
  - `welfare_reports.gms` — 285 lines.
  - `calcExp.gms` — utility-function helper.
- Hertel, T.W. (1997), *Global Trade Analysis: Modeling and Applications*,
  Cambridge University Press (Ch. 2: Welfare measurement).
- McDougall, R. (2003), "A New Regional Household Demand System for
  GTAP," GTAP Technical Paper 20 (CDE).
- This repo: PR #6 (`88685e6`) — Huff first-order baseline.
- Companion plan: `gtap_altertax_implementation_plan_2026-05-13.md`.
