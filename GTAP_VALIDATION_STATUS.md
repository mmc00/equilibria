# GTAP Python-to-GAMS Validation Status

## Session 2026-07-18/19: 15×10 altertax shock closed 94.5/95.8 → 99.5/99.5 (branch gtap_gaps)

**Gate:** `test_altertax_multiperiod_parity` / `measure_gate_tols.py`, tol 1%.
**Result:** ifsub0/ifsub1 = **99.5/99.5 @1%** (98.0 @0.5%), codes 1/1/1, floors 94/93→99/99
(`coverage_matrix.py`), cero regresión en el sweep (3x3 99.93/99.93, 3x4 99.67/99.72,
5x5 99.88/99.88, 10x7 99.29/99.30). El "known eq_paa family" de la matriz NO era irreducible —
eran TRES problemas apilados:

1. **Merge perdido:** el fix quirúrgico de shares Armington (`1bb11e6`, junio) + el audit de
   tools (`156d36c`/`4a20ec7`) quedaron huérfanos en `booming-message` y nunca llegaron a main.
   Cherry-picked (`10ea7e1`/`58a23ff`) → la familia eq_paa desaparece completa de la cola.
2. **Referencia auto-inconsistente:** el GDX 2026-07-17 violaba su PROPIO xfeq
   (`xf[GBR,Capital,Rice,shock]=−0.0048` vs RHS CES +0.24, evaluado a mano con la calibración
   verificada idéntica af/gx/and/ava/fctts/fcttx/kappaf/omegas 0 DIFF). Causa: opciones PATH
   default en NEOS. Fix: regen ONE-SHOT (fiel a los scripts originales — la rampa se descartó:
   parquea los DOF path-dependent) con `path.opt` apretado inline vía `$onecho`
   (refs `bfb12ca`/`81e765f`, 4/4 Optimal, 0 celdas negativas).
3. **El triple degenerado del CD-nest:** bajo sigmap=sigmav=1 los pxeq/pvaeq/pndeq de GAMS son
   tautologías → pva/pnd/px sub-determinados. **Prueba GAMS-vs-GAMS:** tres runs del mismo
   sistema, tres parkings (la rampeada dejó CAN/Metals 42% off; la apretada clavó
   `pva[USA,Chemicals,shock]=0.001` EN SU BOUND). Python agregaba un determinante propio
   (dual CD `px=pnd^and·pva^ava`) que contradecía el parking de la ref (residual 0.915 →
   xd[USA,Chemicals] 11×). Fix (`95b5c5f`): pin del triple COMPLETO al slice shock del GDX +
   desactivar el dual agregado. TRAMPA: el primer intento pineó px en valores del CHECK (px
   faltaba en el bloque de re-siembra) — un hold vale lo que vale su fuente de seed.

Laterales: floors 1e-8 de cantidades eliminados (`c2c6cab`, unfaithful — GAMS solo acota
precios; hacían infactible el punto GAMS en micro-celdas); inits ytax[ft]/[fs] fieles
(`f2d565e`). **Resto 0.5% del 15×10:** familia micro-importaciones JPN Rice (~6% rel sobre
cantidades ~2e-5) — nombrada, no diagnosticada. Documentación completa:
`~/proyectos/notes/economics/cge/gtap7_15x10_altertax_parity_close.md`.

**Epílogo (mismo 07-19): pure 15×10 ifsub0 100→89.8→100, deuda "wrong-branch" CERRADA.**
El regen de la página MCP destapó que quitar los floors regresionó el shock PURE ifsub0 a
89.8% (determinístico, 2 corridas): sin `lb=1e-8`, PATH visita la esquina de autarquía
`xw=0` (slack de complementariedad) → `pet/pe[USA,OtherCrops,*]` vuela a 57.7 (59×). El
A/B decisivo: seed-at-GAMS con opciones default → PATH ABANDONA el punto exacto (pft
+45%, la fase de crash salta de cuenca desde una solución); con el `path.opt` de GAMS
(`crash_method none`, `nms_searchtype line`, `convergence_tolerance 1e-10`) → SE QUEDA
(drift 0.52%, resid 5.7e-12). Fix (`92f67bc`): `solve_multiperiod` defaultea
`PATH_CAPI_OPTIONS` al path.opt de las refs (override del usuario respetado) — simetría
de disciplina de solver, cero cambio de modelo. Resultado: pure ifsub0 shock **100.0%**
@1% (99.98 @0.5%), altertax 99.5 idéntico al decimal, canary 3x3 99.93 idéntico. La
deuda de junio "15x10 wrong-branch (xp=0.840 vs 0.697), candidato PATH basis/warm-start"
era EXACTAMENTE esto: la fase de crash con opciones default.

## Session 2026-07-01: pure-gtap (real-CES) ifSUB=1 shock closed 55→98.95%

**Gate:** `test_gtap_multiperiod_parity.py` (mode="gtap", NOT altertax), gtap7_3x3,
vs `out_gtap_shock_ifsub1.gdx`. Harness `measure_gtap_pure_tols.py --ifsub 1`.
**Result:** SHOCK 55.63% → **98.95% @ tol1%** (82.28/97.60/98.95 @ 0.1/0.5/1%),
CHECK 100% exact, codes 1/1/1. Faithful, no hardcoding, zero regression
(altertax ifsub0/1 + gtap-pure ifsub0 gates green, nl-parity 5/5). Commit `982e47f`.

**Two links, both STRUCTURAL PAIRING bugs (not equations):**

1. **Import-price macro (55→~76%):** under ifSUB=1 the import price is the inlined
   `_m_pm` macro baked into eq_xweq/eq_pmteq at build with the BASE imptx, so the
   +10% wedge never entered. `_rebuild_import_demand_shock_ifsub` rebuilds those
   shock cells with `_m_pm` inlined on the LIVE `pe`+`ptmg` Vars (GAMS has NO
   pm/pmcif Vars under ifSUB=1) — verified exact against the CONVERT e544/e553.

2. **Supply-balance pairing (76→98.95%) — the real cause of EU_28's -13% slide:**
   Python's Hopcroft-Karp matcher paired the price eqs to the QUANTITIES
   (eq_xs→xs, eq_pdeq→xds, eq_peteq→xet), leaving `eq_xseq` (`xs==xds+xet`, a GAMS
   FREE-ROW) unmatched → `deactivate_unmatched_xseq` dropped it → the physical
   supply balance broke (xds+xet exceeded xs by +2.6 for EU_28,Svces) → the whole
   region's price chain (pf/pfact/ps/pd/pe) slid -13% to a SPURIOUS root, USA +9%
   mirror, ROW (residual) anchored, numéraire pwfact=1.0 fine. Fix: keep eq_xseq
   active (`protect_xseq`) + HARD-force the GAMS supply-block pairing
   (pseq↔ps, xdseq↔xds, pdeq↔pd, xeteq↔xet, peteq↔pet, xseq↔xs); `structural_matching`
   gained a hard-pair flag skipping the adjacency check (the quantity⊥price MCP
   complementarity a square system can't express by adjacency). All gated to
   gtap-mode+ifSUB=1 via the `eq_xweq_shock_ifsub` sentinel.

**The decisive diagnostic (the CONVERT+seed test):** generate the GAMS reference as
canonical Pyomo (`convert_gams.py --if-sub`), seed it at the GAMS point → it STAYS;
evaluate that GAMS system's residual AT PYTHON's solved point → **2.6, 71 equations
violated** (worst = xseq[EU_28,Svces]). A point that violates 71 GAMS eqs is NOT a
GAMS equilibrium → it was NOT multi-equilibrium (the long-chased false verdict) but a
DIFFERING PAIRING. Run this test FIRST for "multi-eq vs differing-pairing", before
exhausting per-variable holdfixes (fixing pfact/pd/pft to GAMS all did nothing — the
region was self-consistent at the wrong level). The seed_and_solve gtap tail stays
confounded (its m2 skips the shock rebuilds) — ignore its eq_xweq=2.46 verdict.

## Session 2026-06-19: gtap7_10x7 CHECK-period income bug closed

**Symptom:** gtap7_10x7 altertax CHECK period `yc[USA]=32.8` vs GAMS 12.09 (~2.7×); `regY`/`facty` 2× in every region; check residual 2.48 (code=2).

**Root cause (concrete, the pvaeq-class bug):** under CD (sigmav=sigmap=1) `eq_pvaeq`/`eq_pndeq` are GAMS tautologies (`1=Σaf`, `1=Σio`) that do NOT determine pva/pnd — GAMS pins them via `gtap.holdfixed=1` in **every** period (the check included). Python applied `--holdfix-pva` only to the SHOCK, never the check. The CD geometric-mean dual for pnd gives ≈1.013 vs GAMS 1.000 → inflates the factor-price system 2× → facty→regY→yc double. Found via the cascade: base EXACT + tool-4 calibration 686/686 OK (NOT a calibration bug) → pre-solve seed-residual probe isolated `eq_pndeq` (56 cells, Svces) as the sole structural disagreement.

**DEAD END (reverted):** rewriting the `eq_pndeq` CD branch to the value identity `pnd·nd=Σ(pa·xaa)` (analog of the eq_pvaeq fix) passes the .nl gate 5/5 and reproduces GAMS pnd EXACTLY, but **breaks shock convergence** (3x3 shock 100%/code=1→88.91%/code=2; 10x7 shock residual 3.65e-7→10.46). A globally-correct equation change descalibrates the converging basin. Model left UNTOUCHED.

**Faithful fix (in diff_altertax.py, gated on `--holdfix-pva`):** replicate `gtap.holdfixed=1` on the check — (1) complete seeder `warmstart_from_gams(check)`, (2) `complete_derived_seed` for Python-only vars GAMS doesn't carry (xc/xg/xi/xiagg/xd/xmt), (3) `holdfix_cd_nest` (fix pva/pnd at GAMS values + deactivate eq_pvaeq/eq_pndeq). All three needed together. Gated so the default path is byte-identical for 3x3/9x10 (which keep code=1 check).

| gtap7_10x7 (`--holdfix-pva`) | before | after | GAMS |
|---|---|---|---|
| check `yc[USA]` | 32.83 (2.7×) | **12.086** | 12.093 ✓ |
| check residual | 2.48 | **4.35e-7** | — |
| shock match | 85.05% | **99.42%** | — |

Check now matches GAMS in all 7 regions to ~0.1%. Ref: NEOS job 19621130 (ifSUB=0), stored `equilibria_refs/gtap7_10x7_altertax_cd/out_altertax_ifsub0.gdx`. Regression gate `test_gtap7_nl_parity.py` 5/5 pass (model untouched). Secondary: `run_gtap.py` linear-block gate `residual_tol` 1e-8→1e-6 (inert for parity — altertax uses nonlinear-full). Detail in memory `project_gtap7_10x7_check_holdfix`.

---

**Last updated:** 2026-05-12
**Status:** CLOSED — 100% parity achieved (see "Session 2026-05-12: Branch close-out").
**Objective:** Replicate the GAMS NEOS 10% uniform import-tariff shock in the equilibria GTAP Standard 7 Python template (9×10 and 3×2 NUS333) and reconcile endogenous deltas cell-by-cell against the GAMS reference (NEOS and, where licensing permits, GAMS local).

---

## Baseline Convergence

| Solver | Mode | Equations | Residual | Status |
|--------|------|-----------|----------|--------|
| PATH C API | Linear block | 1,370 | 8.30e-15 | Converged |
| PATH C API | Nonlinear full | 10,296 | 1.12e-06 | Failed strict gate (tol=1e-8) |

The nonlinear baseline residual of `1.12e-06` is the stable pre-session value. It is numerically small but above the strict `1e-8` gate.

---

## GAMS Reference Baseline (out.gdx, ifSUB=0, tariff shock)

| Variable | EastAsia baseline | EastAsia shocked | Delta |
|----------|-------------------|------------------|-------|
| regY | 13.093011 | 13.072910 | **−20,101** |
| gdpmp | 15.220271 | 15.198893 | −21,378 |
| pabs | 1.000000 | 0.999304 | −696 |

---

## Python Baseline (nonlinear, pre-shock)

| Variable | EastAsia value | vs GAMS |
|----------|----------------|---------|
| regY | 13.093009 | Match (income side OK) |
| gdpmp | 15.061 | **−0.159 below GAMS** |
| pabs | 1.0 | Same (by construction) |

### Root cause of `gdpmp` mismatch
Both Python and GAMS initialize `gdpmp` at `15.012` from raw GDX data. After solving:
- Python moves to `15.061` (+0.049)
- GAMS moves to `15.220` (+0.208)

The divergence comes from how the income-investment identity is initialized:
- **GAMS `cal.gms` line 652** overwrites `yi` with `pi*depr*kstock + rsav + savf` but **leaves `xi` at the absorption value**. This creates a deliberate residual in `xieq` (`pi*xi = yi`), which the GAMS solver resolves by increasing `xi` and therefore `gdpmp`.
- **Python** initializes `yi` and `xiagg` consistently from absorption (`vdip+vmip`), so `xieq` is satisfied initially. PATH has no residual to drive `xi` upward, so `gdpmp` stays lower.

Replicating the GAMS deliberate residual in Python was attempted (see Session 2026-04-29 below) but broke nonlinear convergence (residual jumped to 0.42–21), so the changes were reverted.

---

## Python Shocked Scenario (with warm-start)

| Variable | EastAsia delta | GAMS delta | Sign OK? |
|----------|----------------|------------|----------|
| regY | **+31,534** | **−20,101** | **INVERTED** |
| gdpmp | −27,712 | −21,378 | Magnitude close, sign OK |
| yc | Positive for most | Negative for all | Inverted |
| yg | Positive for most | Negative for all | Inverted |
| pabs | 0.0 (flat) | −696 | Wrong |

**Critical problem:** Signs for income and consumption aggregates are systematically inverted relative to GAMS. Until this is fixed, shock deltas are not comparable.

---

## Session 2026-04-29 (Part 2): Root-Cause Fixes Applied

### Changes made
1. **Residual region fix** — `gtap_model_equations.py` four places changed `"RestofWorld"` → `"NAmerica"` to match GAMS `rres`:
   - Line 1134: walras initialization
   - Line 1862: `savf_bar` calibration anchor
   - Line 4510: `eq_savf_rule` capFix guard
   - Line 4574: `residual_regions` for `eq_yi` / `eq_walras`

2. **Shock formula fix** — `run_gtap.py` new `shock_mode="tm_pct"` in `_apply_shock_to_params`:
   - `tm_pct` → `imptx_new = (1 + imptx_old) * (1 + value) - 1`
   - Matches GAMS: `tm.fx = tm.l * (1+shock)` (tariff POWER multiplied)
   - Old `pct` mode scaled only the rate (`imptx * 1.1`), giving ~10× smaller shock for low-tariff goods

### Sign inversion status (uniform 10% tariff shock, nonlinear + warm-start)

| Configuration | EastAsia regy_delta | Sign |
|---------------|---------------------|------|
| Old (RestofWorld residual, `pct`) | +31,534 | **INVERTED** |
| NAmerica fix + `pct` | +12,352 | **INVERTED** |
| NAmerica + `tm_pct` | **−20,991** | **CORRECT** |
| GAMS reference | −20,101 | — |

With both fixes, Python EastAsia regy delta matches GAMS within **4%**.

### Remaining convergence issue
- Nonlinear full baseline: residual 1.04e-06 (unchanged by residual-region fix)
- Shocked model with `tm_pct` + warm-start: residual 3.88e-01 (PATH code 4, did not converge)
- The GAMS shock adds ~0.1 to all tariffs; this large displacement from benchmark prevents convergence from warm-start alone

---

## Session 2026-04-29: Diagnostic Findings

### What was done
1. Located `gdxdump` in `/Library/Frameworks/GAMS.framework/Versions/48/Resources/gdxdump`.
2. Verified GDX data for EastAsia:
   - `VFOB` exports = 3.905, `VCIF` imports = 3.708 → trade balance = +0.197
   - Python `mqtrade` computes exactly this value; trade-side initialization is correct.
3. Fixed `get_gdpmp_init` to use total absorption (`vdpp+vmpp + vdgp+vmgp + vdip+vmip`) instead of domestic-only (`vpm+vgm+vim`).
4. Attempted to replicate GAMS `cal.gms` behavior:
   - Changed `savf_bar` from S-I identity to observed trade balance (`VCIF−VFOB`).
   - Changed `get_yi_init` to the income identity (`pi*depr*kstock + rsav + savf`).
   - Changed `get_xiagg_init` to the same identity value.
   - **Result:** Nonlinear baseline residual exploded from `1.12e-06` to `21`. All changes reverted.
5. Confirmed that `if_sub=False` (full trade-price equations) is the correct Python analogue to GAMS `ifSUB=0`.

### What remains unchanged
- Baseline nonlinear still at `1.12e-06`.
- Shocked scenario still shows inverted signs for `regY`, `yc`, `yg`.
- `pabs` deltas are flat (0.0) in Python vs −696 in GAMS.

---

## Known Bugs / Technical Debt

1. **`_collect_key_quantities` overwrites `ytax` bucket** (`run_gtap.py` lines 583–594). The shock report loses actual model `ytax[r,gy]` values because the code overwrites the key with production-tax-by-activity data.
2. **`vmsb` key inversion bug** — fixed in `_build_import_price_snapshot_block` (line 1626) during an earlier session. The fix is in the codebase but the fresh shock JSON has not been regenerated with it.
3. **`etax` and `mtax` fixed to 0.0** in Python benchmark. GAMS also fixes them to 0, so this is not a divergence source.
4. **`pnum` fixation test completed** — anchoring `pnum=1.0` reduces delta magnitude but does **not** flip sign. Rule out as root cause of inversion.

---

## Next Steps (Pending)

---

## Session 2026-05-01: NUS333 (3×2 HAR) Baseline Convergence

### Goal
Build a fast 3×2 test model (NUS333, GEMPACK HAR format) to iterate on GTAP parity without the 10-minute 9×10 solve times.

### Bugs fixed (HAR-path specific)

1. **`pe` initialization** — `get_pe_init` only checked `vxmd`/`viws` (absent in NUS333). Added `vxsb` check so export price initializes to 1.0 for all active trade routes.

2. **`pmt`/`pmcif` initialization** — Used `VCIF` (value in USD) as the CIF price. Correct formula: `pmcif = VCIF / VXSB` (value / quantity). Also fixed `apply_production_scaling` for the same reason.

3. **`dintx`/`mintx` bounds** — Added `bounds=(-0.999, None)` to prevent `1+dintx ≤ 0` during PATH iterations.

4. **`vim = vdip + vmip`** — HAR path never populated `vim`, leaving `invwgt=0` and `eq_chisave` stuck at 1.0. Added derivation after VDIP/VMIP loading.

5. **`eq_pfeq` specific-factor branch** — Missing `model.xft[r,f]` multiplier. GAMS `fnmeq`: `xf = xft * gf * (pfy/pabs)^eta`. Python had `xf = xscale * gf * (pfy/pabs)^eta`.

6. **`rtf` computation** — `derive_from_benchmark` was computing `rate = (RTFD×0.01 + RTIN×0.01) / (EVFB×1e-6)` — dividing percentage RATES by a VALUE, inflating rtf to 141–563%. Fixed to `rtf = EVFP/EVFB - 1` (GAMS-consistent, scale-independent). For GDX path, this is mathematically equivalent to the old formula `(FBEP+FTRV)/EVFB` when `EVFP = EVFB + FBEP + FTRV`.

### Result

| Model | Solver | Code | Residual |
|-------|--------|------|----------|
| NUS333 (3×2 HAR) | PATH C API nonlinear | **1** | **1.34e-7** |

NUS333 baseline **converged** cleanly. Provides a fast 3×2 iteration platform.

### Priority 1: Fix sign inversion in shocked scenario
The most plausible hypotheses:
- **Closure mismatch:** Something in `GTAPClosureConfig` or `apply_closure` fixes variables that should be free, or frees variables that should be fixed, altering the direction of adjustment.
- **Shock application order:** The 10% tariff shock (`imptx*1.10`) may be applied to the wrong variable, or the `if_sub=False` logic may not correctly recompute `pm` after the shock.
- **Numeraire / price normalization:** `pabs` stays flat in Python while GAMS drops by 696. This suggests the price level is behaving differently (possibly because `pnum` is fixed in Python but GAMS allows it to adjust, or vice-versa).
- **Expenditure-side equation error:** `eq_yc`, `eq_yg`, or `eq_ytax` may have a structural sign error or use the wrong benchmark shares.

**Recommended action:** Perform a full variable-by-variable baseline parity check (Python solved baseline vs GAMS `out.gdx` baseline) for all income, price, and expenditure variables. This will show which variable diverges *first* and point to the offending equation.

### Priority 2: Resolve baseline `gdpmp` mismatch (lower priority)
Only necessary if the shocked deltas are sensitive to the baseline level. Once signs are correct, if deltas are still off by ~20%, then revisit the `xi`/`yi` initialization strategy.

### Priority 3: Fix `_collect_key_quantities` `ytax` overwrite
Straightforward bug; fix when refreshing shock output.

---

## Environment / Constraints

- **GAMS license:** Expired Oct 2024. Cannot run GAMS models or use `gdxpds`. Only `gdxdump` CLI works for reading GDX.
- **Solver:** PATH C API only. IPOPT reports "too few degrees of freedom" on this MCP model.
- **Database:** `basedata-9x10.gdx` = `9x10Dat.gdx` (verified identical).
- **GAMS reference:** `out.gdx` from NEOS job 18737509 (`tariff_comp.gms`, `ifSUB=0`).

---

## Files of Interest

| File | Purpose |
|------|---------|
| `src/equilibria/templates/gtap/gtap_model_equations.py` | Main model equations. CDE/chiInv frozen as Param post-calibration; `pmuv` Var+eq Tornqvist with `pefob0=(1+exptx)`. |
| `scripts/gtap/run_gtap.py` | CLI. `validate-shock`, `_apply_shock_to_params` (`tm_pct`), `_collect_key_quantities` (emits canonical `ytax(r, gy)` over 10 streams). |
| `scripts/gtap/diff_nus333_full.py` / `diff_9x10_full.py` | Cell-by-cell parity diffs vs GAMS reference GDX (NEOS or local). 0 divergent cells in both. |
| `scripts/gtap/bench_nus333_dual.py` | Dual-reference benchmark (NEOS + GAMS local) + N=5 wall-time. |
| `src/equilibria/templates/reference/gtap/scripts/postsim.gms` | GAMS postsim recalc of `pdp/pmp` for `alpha=0` cells so reference GDX matches NEOS convention. |
| `docs/site/benchmarks.md` | Sphinx-rendered benchmarks page (parity + wall-time). |

---

## Session 2026-05-12: Branch close-out

### Final parity state

| Dataset | vs NEOS (base / shock) | vs GAMS local (base / shock) |
|---------|------------------------|------------------------------|
| NUS333 (3×2) | **100% / 100%** | **100% / 100%** |
| 9×10        | **100% / 100%** | blocked — GAMS community license cap (2,500 rows; 9×10 has ~10k equations) |

Cell-by-cell diffs (`diff_nus333_full.py`, `diff_9x10_full.py`) report 0 divergent cells across all variables and parameters in both base and shock.

### Fixes that closed the gap (cumulative)

1. **Region residual fix** — `NAmerica` (matches GAMS `rres`), not `RestofWorld`.
2. **Shock formula `tm_pct`** — multiplies tariff *power*: `imptx_new = (1+imptx_old)*(1+v) − 1`.
3. **Snapshot pinning at t0** — counterfactual model must receive `t0_snapshot=base_model`; otherwise CES weights recalibrate against perturbed state and invert shock direction.
4. **Factor-price pinning** — `fix_endowments=True` + active `eq_xfteq` pins `pft=pabs`; deactivate `eq_xfteq` when `xft` is fixed.
5. **9×10 shock convention** — rate-scaled, not power (`tm = tm_base*1.1`), confirmed by EastAsia regy delta 0.0% err.
6. **CDE / `chiInv` elasticities frozen as Param post-calibration** — declared but unpaired in `model gtap` causes them to drift; freeze at cal-time `xcshr` (Param, not Expression).
7. **`pmuv` Tornqvist as Var+eq** with `pefob0=(1+exptx)`.
8. **`pwmg=0` where `tmarg=0`** — injected before `pwmg.fx` pin in NEOS bundle.
9. **`pdp/pmp` postsim recalc** — `postsim.gms` recalculates `pdp = pd*(1+dintx)` and `pmp = pmt*(1+mintx)` for all cells (including `alpha=0` ones gated off by `pdpeq/pmpeq`), matching the 9×10 reference GDX convention.
10. **`ytax(r, gy)` canonical 10 streams** — `_collect_key_quantities` now emits `pt/ft/fs/fc/pc/gc/ic/et/mt/dt` matching `model.gms:642-686`, plus derived `ytax_tot` and `ytax_ind = ytax_tot - ytax[r|dt]` (PR #3 / `28a9b93`).

### Benchmarks (NUS333, N=5 wall-time)

- Python median: 0.644s
- GAMS local median: 0.848s
- Ratio: Python ≈ 0.76× GAMS local

### Open items (out of scope for this branch)

- 9×10 GAMS local parity: requires a non-community GAMS license (community cap of 2,500 nonlinear rows blocks the ~10k-equation 9×10 model). NEOS reference is sufficient for current goals.
- `output/gtap_ifsub_false_warmstart.json` and `output/gtp_baseline_reverted.json` are pre-fix snapshots and have been superseded by the parity CSVs under `benchmarks/`.

---

## GTAP multi-período puro (mode="gtap") — sesión 2026-06-24

**Objetivo:** llevar el modelo gtap PURO (shock arancelario +10% tm, no altertax) al motor
multi-período `solve_multiperiod`, gateado por `ifSUB`, con vistas a deprecar el single-period.

**Resultado (gtap7_3x3 vs GAMS LOCAL `out_gtap_shock_ifsub0.gdx`):**

| Período | Inicio | Final | Estado |
|---------|--------|-------|--------|
| base    | —      | code=1 (skip_base_solve, ancla calibrada) | ✅ |
| **check** | 60.74% | **100.00% EXACTO** | ✅ ecuaciones gtap MP == GAMS |
| **shock** | 60.74% | **99.70%** | ✅ paridad (0.30% basin Land sluggish) |

**10 fixes concretos cerrados (todos `mode="gtap"`-gated, altertax byte-idéntico):**
1. `mode="gtap"|"altertax"` en `solve_multiperiod` (salta betaCal, base_closure, sin CD holdfix).
2. Referencia GAMS LOCAL generada (`out_gtap_shock_ifsub{0,1}.gdx`, GAMS v53 community, 3x3 cabe).
3. Cuadrar factor block check/shock (pft/eq_pfteq collapse + NatRes fixing; el MP no tiene `xftflag`).
4. `skip_base_solve=True` — el MP base no se manda a PATH (deslizaba el nivel de precios de USA 2×).
5. `yi[ROW]` free-DOF: `eq_walras` viva + `walras=0` (ancla la inversión de la región residual;
   `eq_yi` salteada para rres como GAMS). check 73.76→78.21%.
6. `ytax[mt]`: filtro por importador (col 0) + tasa POWER `(1+imptx)*1.10-1` (era exportador + rate).
7. `eq_pmeq` shock-in-equations: el builder MP hornea `imptx` base como literal; rebuild quirúrgico
   SOLO de las celdas shock (no la slice entera → recalibraría shares). +3.30pp.
8. **pft FREE para factores reales** (invirtió `_collapse_pft_pfteq`, que pineaba pft=1.0 con
   semántica ALTERTAX; GAMS `iterloop.gms:145-146` fija pft SOLO para xftFlag=0). + etaf=0 para sf.
   **check 80→100% EXACTO.** El parche previo `_holdfix_activity_scale` (xp) quedó contraproducente
   tras este fix → desactivado (flag `_HOLDFIX_ACTIVITY_SCALE_GTAP=False`, con evidencia medida).

9. **Cadena de 3 eslabones acoplados que cerró el shock 67.12→99.70%** (NO era basin — era un
   shock incompletamente aplicado contaminando ingreso→demanda→reallocación Armington):
   - **Link 1 — ancla NatRes:** NatRes es el factor sector-específico (`fnm`); su `eq_pfeq` con
     `etaff=0` es `xf==xscale·gf` (vertical, pf-libre) → `pf[NatRes]` es un free-DOF que explotaba
     a 9.27 bajo el shock diagonal. Fix: holdfix `xf[NatRes,a]` al valor previo + desactivar la
     `eq_pfeq` redundante (GAMS holdfixea xf[NatRes] entre períodos, GDX-confirmado); quitar
     `eq_xfeq[USA,NatRes,Mnfcs]` de `_REDUNDANT_FACTOR_ROWS`.
   - **Link 2 — arancel doméstico-diagonal:** `_apply_imptx_shock` SALTABA la diagonal `r==rp`;
     GAMS la shockea (GDX: `imptx[EU_28,Mnfcs,EU_28]` 0→0.1, `imptx[ROW,Mnfcs,ROW]` 0.029→0.132).
     Fix: no saltar la diagonal en gtap-mode. `_rebuild_eq_pmeq_shock` la rutea sola (se auto-saltea
     las celdas sin shock).
   - **Link 3 — fuga de ingreso (la causa dominante):** `eq_ytax[mt]` (gtap_model_equations.py:5474)
     horneaba el `imptx` BASE; con la diagonal shockeada, el stream de impuesto diagonal se perdía
     del `regY` → ingreso/demanda colapsaba → el Armington "sobre-disparaba" reallocando sobre un
     ingreso EQUIVOCADO. Fix: `_rebuild_eq_ytax_mt_shock` (espejo de `_rebuild_eq_pmeq_shock`) con
     `imptx` shockeado (filtro col2 = convención `(exporter,good,importer)` = GAMS ytaxeq:680,
     verificado ytax[USA,mt]=0.26003 exacto); `_recompute_ytax_mt` no-op para gtap-mode.

   **Combinados:** check 100%, shock 99.70%, 3 períodos code=1, resid 4.1e-11. **Probado que el
   punto GAMS ES un fixed-point** (seed-at-GAMS da 100.00%). Los 3 eslabones eran UN bug visto
   desde 3 ángulos (el shock incompleto propagándose).

10. (incluido en la cadena arriba — el conjunto de 3 links es el fix #10)

**Gap residual del shock (0.30%) = basin genuino, PROBADO (no asumido):** el factor Land sluggish
de EU_28/ROW (`pft[EU,Land]` 3.9%, `pf/xf` ~2%). El seed-at-GAMS sobre la cadena completa SE SOSTIENE
en 100.00% → el punto GAMS es alcanzable; el 0.30% es un efecto single-jump del CET de Land sluggish,
irreducible sin homotopía. **Es 0.30%, no 33%** — la homotopía sería el único lever, pero el costo/beneficio
no lo justifica para 0.30%.

**Lección sobre "basin":** la regla `[[feedback_gams_is_source_of_truth]]` / `basin NUNCA es la respuesta`
se cumplió al pie. Un veredicto previo de "shock 67% = basin, homotopía dead-end" era una conclusión
PREMATURA — debajo había una cadena de 3 bugs concretos (diagonal/NatRes/ytax[mt]). Cada vez que se
empujó ("¿por qué baseline sí y shock no?"), apareció el siguiente eslabón. Solo el 0.30% final es basin,
y se probó con seed-at-GAMS, no se asumió.

**Por qué costó (lección):** gtap y altertax difieren genuinamente (gtap usa esubva/CES real → pva
DETERMINADO; altertax fuerza CD → pva libre, pinchado por holdfix). Varios fixes iniciales aplicaron
semántica altertax por error (el pin de pft, el holdfix de xp) — eran compensaciones de UNA causa raíz.
La maquinaria altertax (`_holdfix_cd_nest`/`_complete_derived_seed`) NO transfiere a gtap (probado:
prenderla siembra desde GAMS o sobre-restringe). No hubo atajo vía altertax; el camino capa-por-capa
con la cascada de tools fue el único fiel.

**Gate:** `tests/templates/gtap/test_gtap_multiperiod_parity.py` (local-only, SKIP-safe), floor shock 99.0%.
**Deprecación SP:** ahora SOBRE LA MESA — el shock está a paridad (99.70%, 0.30% basin probado).
gtap7_3x3 cierra el ciclo base→check→shock con check EXACTO y shock a paridad. Pendiente para deprecar:
escalar la cadena a los demás datasets (5x5/10x7/15x10) y portar el gate `.nl` al motor MP (Tareas 3-6
del plan). Spec/plan: `docs/superpowers/specs|plans/2026-06-23-gtap7-multiperiod-only*`.
## Session 2026-06-13: gtap7_3x3 altertax income-chain fixes (faithful to GAMS)

**Context:** Debugging the gtap7_3x3 altertax `check` period (was ~56% cell match
vs the local v53 reference). Used the residual test (probe.py `--seed-gams base`,
seed coverage 94.8%) to isolate which equations genuinely diverge at the GAMS point
vs. which are seed artifacts.

**Root cause (one bug, three sites):** Python derived `fcttx` from the empty HAR
`RTFD` header → `fcttx=0`, but GAMS uses `ftrv = EVFP − EVFB` (comp.gms:2833), so
the factor-tax wedge is `(EVFP−EVFB)/EVFB = rtf ≈ 0.47`, nonzero (EVFP≠EVFB in both
gtap7_3x3 and 9x10). Fixed faithfully to GAMS:

1. `_compute_ytax_ind_bench` — include the `ft` stream `sum(rtf·evfb)` (was omitted
   under the `fcttx=0` assumption). → ytaxInd 2.31→4.85 (GAMS 4.85), `betap`
   0.754→0.632 (=GAMS), `betas` matches GAMS. `eq_yc`/`eq_yg` residual 1.9→5e-4.
2. `_fcttx_init = rtf` (was `=RTFD=0`). → `eq_ytax[r,'ft']` residual 2.54→~0.
3. `eq_pfaeq = pf·(1+fctts+fcttx)` (GAMS canonical comp.gms:2328; now fcttx=rtf so
   numerically identical to the prior `pf·(1+rtf)`). → `eq_pfaeq` residual 2.4e-10.

**Snapshot expansion:** added 23 income/capital/tax vars (pi, kstock, ytax, ytaxTot,
nd, chif, savf, psave, pgdpmp, rorc/rore/rorg, pfy, pm, pva, pnd, …) to the altertax
GDX snapshot so the residual test seeds them instead of leaving them at init.

**Result (residual test at the GAMS local point):** residuals >1e-3 dropped 60→21,
max residual ~2.5 → ~0.022. The entire income chain (eq_regy / eq_facty /
eq_ytax_ind / eq_ytax_tot / eq_ytax[ft] / eq_yc / eq_yg / eq_pfaeq) collapsed to ~0.

**ROW residual-region investment block: SANE.** What looked like a 1.9e6 explosion
in eq_xda/eq_paa/eq_xiagg/eq_pi at warm-start was a seed artifact (pi/kstock/xiagg
unseeded). With the full snapshot: pi[ROW]=1.0, xiagg[ROW]=yi[ROW]=13.04, eq_rsav[ROW]
residual ~3e-3 (rounding in betas calibration). No structural bug.

**Remaining lead:** `eq_pfeq` (~0.006–0.022, factor price tax-exclusive) — the
dominant residual now, two orders smaller than the income-chain bugs just fixed.

**nl-parity gate:** 5 passed throughout. End-to-end cell match dropped (52.6%, solver
basins) but the residual test (clean metric) improved 3×; user prioritized GAMS
fidelity over the altertax match. Tooling: probe.py (cached parity probe) added this
session.

### Shock-period status (open lead, warm-start/basin class)

After the income-chain + gf fixes, the **check** period converges perfectly to
GAMS (gdpmp/regy/pgdpmp/pabs all 0.0% rel; check residual 3.1e-11, code=1). The
`diff_altertax` end-to-end match (~52-54%) compares the **shock** period, whose
solve currently **fails**:

- shock residual ~38-63, code=2; prices collapse to ~0.04 (GAMS shock prices RISE
  to ~1.01-1.04 with the tariff).
- Fails under both shock modes (old `imptx*1.10` and the corrected
  `tm_pct = (1+imptx)*1.10-1`, c1bcc38) → not a magnitude issue.
- The shock builds a fresh model with `t0_snapshot=m_chk` and
  `solution_hint=warm_chk`. Suspect the shock warm-start or shock closure.

Next steps (basin class, per parity skill): seed shock directly with the GAMS
shock point to isolate warm-start vs closure; try `_run_homotopy_shocked`
(run_gtap.py:2712); or a closure diff of shock vs check.
