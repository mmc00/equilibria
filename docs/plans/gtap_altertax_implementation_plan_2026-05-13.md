# GTAP Altertax — Implementation Plan

**Date:** 2026-05-13
**Status:** Draft — pending green-light to implement
**Reference impl:** `/Users/marmol/proyectos2/cge_babel/gtap/cgebox/` (W. Britz, 2016)

## Why altertax

Altertax (Malcolm 1998) is the canonical GTAP procedure for producing a
**re-balanced baseline GDX** that absorbs an exogenous change in
taxes/parameters into the data, instead of into behavioural quantities.
Used to:

- **Update tax rates** without invalidating the rest of the SAM. The
  new dataset is internally consistent (Walras=0) at the new tax
  structure, so subsequent counterfactuals can shock from a clean base.
- **Pre-process aggregations** that change taxes (e.g. removing trade
  margins from a subset of bilaterals).
- **Recalibrate after a parameter change** (e.g. swapping factor
  mobility regimes) without re-running the whole calibration upstream.

The trick: run the model with **all elasticities set to unity (CD)** so
that any tax change leaves equilibrium *quantities* unchanged (CD
share-stability). What changes is the *value-flow decomposition* — and
that becomes the new SAM.

## Reference implementation (cgebox)

Three pieces in `cge_babel/gtap/cgebox/`:

### 1. `gams/configs/altertax.gms` — run config (81 lines)
- Comparative static, global model.
- All optional modules OFF (AEZ, AGR, E, MELITZ, MyGTAP, Aggregate
  Armington/firm demand, CapVintages, Regional household, Labor nest,
  CapSkLab nest, CO2 emissions).
- DemSystem = CD.
- All factors fully mobile (`fm_sel(fmm) = YES`).
- Closure: Global equal returns to capital / Tax income / Spending /
  Exchange rate numeraire.
- Output flags: `outputtypesAlterTax=ON`, `SAMtoGDX=ON`.
- Loads `Parameter_altertax` as elasticity override.

### 2. `gams/scen/Parameters/parameter_altertax.gms` — elasticity overrides (42 lines)
| Param | Original GTAP | Altertax |
|-------|---------------|----------|
| `p_sigmaNest(r,"VA",a)` | calibrated | **1** (CD) |
| `p_sigmai(r)` | calibrated | **1** (CD) |
| `p_sigmam(r,i)` | calibrated | **0.95** |
| `p_sigmaw(r,i)` | calibrated | **0.95** |
| `p_omegaf(r,f)` | calibrated | **1** (CD) |
| `RorFlex(r,t)` | default | unchanged |

### 3. `gams/postModel/altertax.gms` — re-balancing (492 lines)
The actual altertax step. Runs *after* solve. Writes
`<dataset>_alttax.gdx` containing a new internally-consistent baseline.

**Eleven steps:**

1. **Extract SAM** from `t=shock` levels: `sam0(rSamReg,is,js) = sam(rSamReg,is,js,"shock")`.
2. **Armington at agent prices** (`xda0`, `xma0`):
   ```
   xda0(r,i,aa) = max(0, xd.l(r,i,aa,"shock")) * pdp.l(r,i,aa,"shock") / gblScale
   xma0(r,i,aa) = max(0, xm.l(r,i,aa,"shock")) * pmp.l(r,i,aa,"shock") / gblScale
   ```
3. **Armington at market prices** (`xdm0`, `xmm0`):
   ```
   xdm0(r,i,aa) = max(0, xd.l(r,i,aa,"shock")) * pd.l(r,i,"shock")  / gblScale
   xmm0(r,i,aa) = max(0, xm.l(r,i,aa,"shock")) * pmt.l(r,i,"shock") / gblScale
   ```
4. **Transport margins** (`xmarg0`):
   ```
   xmarg0(m,r,i,rp) = tmarg(r,i,rp,"shock") * xw.l(r,i,rp,"shock")
                      * p_amgm(m,r,i,rp) / m_lambdamg(m,r,i,rp,"shock")
                      * ptmg.l(m,"shock") / gblScale
   ```
5. **Capital stocks and depreciation:**
   ```
   kstock0(r) = kstock.l(r,"shock") / gblScale
   depry0(r)  = p_fdepr(r,"shock") * pi.l(r,"shock") * kstock.l(r,"shock") / gblScale
   ```
6. **Tax revenue series** (critical for altertax):
   ```
   imptxY0(r,rp,i) = imptx.l(r,i,rp,"shock") * pmcif.l(r,i,rp,"shock") * xw.l(r,i,rp,"shock") / gblScale
   exptxY0(r,rp,i) = exptx.l(r,i,rp,"shock") * pe.l(r,i,rp,"shock")    * xw.l(r,i,rp,"shock") / gblScale
   fcttsY0(r,f,a)  = fctts.l(r,f,a,"shock") * sam(r,f,a,"shock")
   fcttxY0(r,f,a)  = fcttx.l(r,f,a,"shock") * sam(r,f,a,"shock")  [+ CO2 component if enabled]
   ```
7. **Collapse multi-household to single `hhsld`:**
   ```
   xda0(r,i,"hhsld") = sum(h, xda0(r,i,h));  xda0(r,i,h≠hhsld) = 0
   xma0(r,i,"hhsld") = sum(h, xma0(r,i,h));  xma0(r,i,h≠hhsld) = 0
   xdm0(r,i,"hhsld") = sum(h, xdm0(r,i,h));  ...
   xmm0(r,i,"hhsld") = sum(h, xmm0(r,i,h));  ...
   ```
8. **Optional modules** (skip in initial port):
   - AEZ: land-use re-derivation with proportionality assumptions.
   - CO2/non-CO2: rebase emission factors.
   - MRIO: split factors.
   - GMIG: migration.
   - FABIO: nutrition.
9. **Real-GDP scale:** `p_scaleRGdpMp(r) = 1 / pgdpmp.l(r,"shock")`.
10. **Calibration param preservation** (`cal.gms:395`): keep `eh0`, `bh0`
    (CDE/AIDADS demand params) when altertax mode is on — normally
    killed after calibration, but altertax needs them for the rewrite.
11. **Unload to GDX**: `<dataset>_alttax.gdx` with sets/parameters
    listed in the `execute_unload` block (lines 439–484 of original).

The flag `scalar altertax / 1 /` written into the output GDX marks it
as an altertax-derived dataset.

## Equilibria port plan

Conceptually a clean fit. No changes to the Pyomo model equations. Four
new modules + CLI integration. Estimated 600–900 LoC + tests.

### Piece A — `gtap_altertax_closure.py` (~50 LoC)
New closure preset, parallel to `build_gtap_contract("standard")`:

```python
from equilibria.templates.gtap import build_gtap_contract
contract = build_gtap_contract("altertax")
# returns a GTAPContract with the cgebox altertax conventions baked in
```

Concretely:
- `fix_endowments = False` (mobile)
- `numeraire = "exchange_rate"` (analog to "Exchange Rate")
- Returns regime: global equal returns to capital
- Gov closure: tax income
- Final consumption closure: spending
- `RorFlex = 10.0` (existing default — unchanged)
- Module flags all OFF (no Melitz, no AEZ, etc.)

### Piece B — `parameter_altertax.py` (~30 LoC)
Elasticity overrides applied to `GTAPParameters` before building the
model:

```python
from equilibria.templates.gtap.altertax import apply_altertax_elasticities

altertax_params = apply_altertax_elasticities(base_params)
# In-place is opt-in; default returns a deep copy.
```

Sets `esubva → 1`, `esubinv → 1`, `esubd → 0.95`, `esubm → 0.95`,
`etrae → 1` across all (r, sector) cells. Same pattern as
`apply_tariff_shock` from the existing shock infrastructure.

### Piece C — `gtap_altertax_postmodel.py` (~400 LoC)
The actual altertax step. Reads solved Pyomo `var.value` and writes a
new GTAP-format dataset.

```python
def rebalance_to_altertax_dataset(
    base_params: GTAPParameters,
    shock_params: GTAPParameters,
    shock_model: pyomo.ConcreteModel,
    *,
    output_path: Path,                  # .har (preferred) or .gdx
    keep_demand_params: bool = True,    # eh0 / bh0 (CDE/AIDADS coefs)
    output_format: Literal["har", "gdx"] = "har",
) -> Path:
    """Implement cgebox/gtap/gams/postModel/altertax.gms in Python.

    Reads:  solved shock_model + base/shock parameters
    Writes: balanced GTAP-format file consumable by subsequent
            equilibria runs as a new baseline.
    """
```

Translation table (cgebox → equilibria):

| cgebox symbol | equilibria source |
|---|---|
| `xd.l(r,i,aa,"shock")` | `shock_model.xd[r,i,aa].value` |
| `xm.l(r,i,aa,"shock")` | `shock_model.xm[r,i,aa].value` |
| `pdp.l`, `pmp.l` | `pdp`, `pmp` Pyomo vars (postsim values) |
| `pd.l(r,i,"shock")`, `pmt.l(r,i,"shock")` | `pd[r,i].value`, `pmt[r,i].value` |
| `xw.l`, `imptx.l`, `exptx.l`, `pe.l`, `pmcif.l` | corresponding Pyomo vars |
| `ptmg.l`, `tmarg`, `p_amgm`, `m_lambdamg` | margin vars/params from `shock_params` |
| `kstock.l`, `p_fdepr`, `pi.l` | shock_model + base_params |
| `sam(r,f,a,"shock")`, `fctts.l`, `fcttx.l` | `shock_params.benchmarks.evfm` + factor-tax containers |
| `gblScale` | constant 1.0 (equilibria doesn't use global scaling) |
| `pgdpmp.l` | `shock_model.pgdpmp[r].value` |

Output format:
- **HAR (preferred)**: use `equilibria.babel.har.write_har` (native;
  already used for WELVIEW.har). Match GTAP basedata header layout
  (`VDFB`, `VMFB`, `VDPB`, `VMPB`, `VDGB`, `VMGB`, `VDIB`, `VMIB`,
  `EVFB`, `EVOS`, `VXSB`, `VMSB`, `VTWR`, `VST`, `SAVE`, `VDEP`,
  `MAKB`, `PTAX`, `RTMS`, `RTXS`, …).
- **GDX (optional)**: use `equilibria.babel.gdx.writer` if upstream
  consumer requires GDX. Sets/parameters mirror the cgebox
  `execute_unload` block (lines 439–484 of the GAMS file).

Modules NOT ported in this iteration (defer to follow-ups):
- AEZ land use
- CO2 / non-CO2 emissions rebase
- MRIO split factors
- GMIG migration
- FABIO nutrition
- Recursive dynamic mode (only comparative static)

These add ~200 LoC each and only kick in when the corresponding module
is active — out of scope for the standard 9x10/NUS333 datasets.

### Piece D — CLI + tests (~150 LoC)

New `altertax` subcommand on `scripts/gtap/run_gtap.py`:

```bash
uv run python scripts/gtap/run_gtap.py altertax \
    --gdx-file data/9x10/9x10Dat.gdx \
    --shock-variable rtms \
    --shock-index "(Oceania,c_Crops,EastAsia)" \
    --shock-value 0.10 \
    --shock-mode tm_pct \
    --output reports/altertax/9x10_alttax.har \
    --output-format har
```

Wiring: applies altertax closure + elasticity overrides, applies the
user's shock, solves with PATH C API nonlinear full + equation_scaling
(per CLAUDE.md), then calls `rebalance_to_altertax_dataset`.

Tests (`tests/templates/gtap/test_altertax.py`):

1. **Closure & elasticity preset round-trip** — verify
   `build_gtap_contract("altertax")` and `apply_altertax_elasticities`
   produce the expected values.

2. **CD invariance check** — with altertax elasticities and a small
   tariff shock, verify that *quantities* stay essentially unchanged
   relative to the no-shock baseline (the defining property of CD).
   Tolerance: `rel_diff < 1%` on `xp`, `xc`, `xw`.

3. **Internal consistency of output** — load the produced
   `_alttax.har` and verify:
   - All `VDxx`/`VMxx`/`EVFB` totals satisfy SAM row/col balance.
   - Tax revenue series (`imptxY0`, `exptxY0`, `fcttxY0`) reconcile
     against quantity × rate × price.
   - `p_scaleRGdpMp` × shocked `pgdpmp` = 1.

4. **Round-trip baseline** — load the altertax `.har` as a new
   baseline via `load_from_har`, build & solve at the new base, verify
   `walras < 1e-9` and that the equilibrium reproduces the shocked
   levels of the previous run (within MCP tolerance).

5. **Multi-format parity** — write both `.har` and `.gdx`, load both,
   verify identical cell values.

## Open questions for the implementer

1. **Whether to support `gblScale ≠ 1`.** Cgebox uses a global scaling
   factor that equilibria doesn't have. For a clean port, hardcode
   `gblScale = 1.0` and document the divergence.

2. **`p_scaleRGdpMp` consumer** — cgebox uses this for real-GDP
   reporting in derived simulations. Equilibria doesn't yet have a
   "derived simulation" concept on top of altertax-rebalanced
   datasets. Decision: write the field but document that no current
   equilibria pipeline reads it.

3. **Multi-household handling.** Cgebox collapses multiple households
   to single `hhsld`. Equilibria's standard 9x10/NUS333 datasets only
   have one household — confirm before implementing the collapse
   logic. If not needed, skip step 7.

4. **HAR vs GDX as canonical output.** Recommend HAR as default
   because (a) babel HAR writer is mature, (b) GEMPACK tools read it
   natively, (c) the cgebox output is the only "official" GDX form
   but equilibria's load path is HAR-first. Keep GDX as opt-in.

5. **Numeraire after rebalance.** The new dataset has its own
   numeraire convention (Exchange Rate). When re-loaded as a baseline,
   confirm the closure presets in equilibria don't re-pin the
   numeraire in a conflicting way.

## Implementation order (suggested)

1. **Piece A + B** in one PR — closures and elasticity overrides.
   These are cheap and surface the new public API. (~1 day)
2. **Piece C without optional modules** — core post-model
   re-balancing. Hardest piece. (~2-3 days)
3. **Piece D** — CLI + test suite. End-to-end validation. (~1 day)
4. **Optional modules** (CO2, AEZ, etc.) — separate follow-up PRs as
   needed.

Total: ~1 week for a usable altertax against 9x10 and NUS333.

## Validation strategy

Compare against the cgebox GAMS implementation:

1. Run cgebox altertax on `9x10` (if license allows) or a smaller
   dataset → produces `9x10_alttax.gdx`.
2. Run equilibria altertax with the same shock → produces
   `9x10_alttax.har`.
3. Convert one to the other's format (babel can do HAR↔GDX).
4. Cell-by-cell diff using `scripts/gtap/diff_9x10_full.py` adapted
   for the dataset comparison (rather than the variable comparison it
   does today). Target: 100% match within `tol_rel=1e-3`.

If GAMS local license isn't sufficient for 9x10, fall back to NEOS or
use the smaller NUS333 (which fits in the community license).

## Files to be added

```
src/equilibria/templates/gtap/
├── altertax/
│   ├── __init__.py
│   ├── closure.py             # Piece A
│   ├── parameter_overrides.py # Piece B
│   ├── postmodel.py           # Piece C — the heart
│   └── README.md              # Module documentation
├── README.md                   # Add "Altertax" section
└── ...

scripts/gtap/run_gtap.py        # Piece D — altertax subcommand

tests/templates/gtap/
└── test_altertax.py            # Piece D — full test suite

docs/site/guide/
└── altertax.md                 # User-facing guide

docs/site/guide/index.md        # Register altertax in toctree
```

## Acceptance criteria

- [ ] `build_gtap_contract("altertax")` returns a working contract.
- [ ] `apply_altertax_elasticities(params)` overrides 5 elasticity
      containers as specified.
- [ ] `rebalance_to_altertax_dataset(...)` produces a `.har` that
      reloads via `load_from_har` and solves with `walras < 1e-9`.
- [ ] CD invariance: with altertax preset and a 10% tariff shock,
      `|Δxp| / xp_base < 1%` (the defining property of altertax).
- [ ] CLI subcommand `altertax` works end-to-end on 9x10 and NUS333.
- [ ] All 5 test groups pass.
- [ ] Docs: dedicated guide + Step 6 in `gtap_quickstart.md` + README
      section. Sphinx build clean.

## References

- Malcolm, G. (1998), "Adjusting Tax Rates in the GTAP Data Base,"
  GTAP Technical Paper 12.
- Britz, W. (2016+), CGEBOX project — `cge_babel/gtap/cgebox/gams/`.
- Hertel, T.W. (1997), *Global Trade Analysis: Modeling and Applications*,
  Cambridge University Press.
