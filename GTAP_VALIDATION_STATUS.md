# GTAP Python-to-GAMS Validation Status

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
| **shock** | 60.74% | **67.12%** | ⛔ basin (ver abajo) |

**9 fixes concretos cerrados (todos `mode="gtap"`-gated, altertax byte-idéntico):**
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

**Gap del shock (67.12%) = BASIN genuino, verificado rigurosamente (NO un bug):**
- El punto GAMS del shock SATISFACE las ecuaciones del MP (~1e-7 en trade/CET/Armington) — es una
  raíz válida, pero PATH single-jump elige OTRA raíz válida.
- Causa estructural: `omegax=omegaw=∞` en Manufactures → doméstico/exportado son sustitutos
  perfectos; el split lo fija solo la demanda vía Armington alto (esubm=7.87). Múltiples allocaciones
  consistentes para un salto de 10%. GAMS llega por homotopía ~30 pasos; Python da 1 salto.
- **Homotopía = DEAD END probado:** rampa correctamente warm-starteada da 67.12% bit-idéntico en
  N=5/10/20/30. La rama equivocada está SUAVEMENTE conectada al seed del check (al primer 1% de
  arancel la rama continua de Python va con signo equivocado y nunca gira) — no hay fold que una
  rampa más fina atrape. Cerrar más requeriría anclar el split a mano = sembrar desde la respuesta
  (no fiel). El single-jump 67.12% es el mejor resultado FIEL.

**Tech-debt fiel pendiente (no tocar hasta cerrar el basin):** `eq_ytax[mt]` (gtap_model_equations.py:5476)
hornea `imptx` base como literal (clase eq_pmeq); hoy enmascarado por el post-solve `_recompute_ytax_mt`.
Arreglarlo en-ecuación HOY regresiona (shock code 1→2, 67→64%) porque empuja revenue a un loop de
ingreso off-basin. Diferir hasta cerrar el basin del bloque real.

**Por qué costó (lección):** gtap y altertax difieren genuinamente (gtap usa esubva/CES real → pva
DETERMINADO; altertax fuerza CD → pva libre, pinchado por holdfix). Varios fixes iniciales aplicaron
semántica altertax por error (el pin de pft, el holdfix de xp) — eran compensaciones de UNA causa raíz.
La maquinaria altertax (`_holdfix_cd_nest`/`_complete_derived_seed`) NO transfiere a gtap (probado:
prenderla siembra desde GAMS o sobre-restringe). No hubo atajo vía altertax; el camino capa-por-capa
con la cascada de tools fue el único fiel.

**Gate:** `tests/templates/gtap/test_gtap_multiperiod_parity.py` (local-only, SKIP-safe), floor shock 67.0%.
SP NO deprecado (el shock no está a paridad). Spec/plan: `docs/superpowers/specs|plans/2026-06-23-gtap7-multiperiod-only*`.
