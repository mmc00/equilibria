# GTAP Python-to-GAMS Validation Status

**Last updated:** 2026-05-01
**Objective:** Replicate the GAMS NEOS 10% uniform import-tariff shock in the equilibria GTAP Standard 7 (9×10) Python template and reconcile endogenous deltas against a correct GAMS reference.

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
| `src/equilibria/templates/gtap/gtap_model_equations.py` | Main model equations. Key areas: `get_gdpmp_init`, `get_yi_init`, `get_xiagg_init`, `eq_ytax`, `eq_yc`, `eq_yg`, `eq_pabs`, `eq_gdpmp` |
| `scripts/gtap/run_gtap.py` | CLI. Contains `validate-shock`, `_collect_key_quantities` (ytax bug), `_build_import_price_snapshot_block` |
| `src/equilibria/templates/gtap/gtap_solver.py` | Solver wrapper. `apply_closure`, `apply_aggressive_fixing_for_mcp`, numeraire fixing |
| `src/equilibria/templates/gtap/gtap_parameters.py` | Parameter loading, `savf_bar` calculation, final-demand splits |
| `output/gtap_ifsub_false_warmstart.json` | Last good shocked run (signs inverted) |
| `output/gtp_baseline_reverted.json` | Last good baseline run (res 1.12e-06) |
