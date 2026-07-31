# F3.5 — Base-calibrado sin check: adopt the settled point as base (2026-07-30)

**Status:** Done. A selectable `base_calibrated` mode on the GTAP block model pre-settles
the specific-factor prices (Land/NatRes) at calibration time, so the shock runs
`base→shock` (no `check`) and the land-price response matches GEMPACK (−3.0% vs −2.68%)
instead of the check-contaminated GAMS path (−18%). The faithful-to-GAMS default
(`base→check→shock`) is untouched (parity gates green, 0-diff vs GAMS).

Branch `f3.5-base-calib` (from `main` `e681d26`, after F3 merged).

## The problem

GTAP normalizes base prices to `1.0`, but for the **specific factors** with a sloped
supply nest (Land/NatRes, CET `eq_pfteq`, ω=`etrae`) that `1.0` is a normalization
artifact — not the model-consistent price. GAMS (and equilibria, faithful to it) settles
it in a `check` period: a no-shock solve that lets the land price fall to its equilibrium.
On `gtap7_3x3`, `pft[EU_28,Land]` slides `1.0 → 0.845` (−15.5%) in the check, then the
shock moves it a further −3%. GEMPACK/Julia/CGEBox instead pre-calibrate the base
consistent, so their land price only responds to the shock (GEMPACK: −2.68%).

Against GEMPACK, the −15.5% check re-settlement reads as a ~15pp discrepancy that is
really a *methodology* difference between engines, not a bug (confirmed in F3 by reading
5 implementations — see `memory: project_gempack_shock_sweep_check_gap`).

## The spike that redefined the mechanism

The design's first mechanism (re-derive `gf_share` to freeze the settled price) was
**refuted by a spike before implementing it**:

- In gtap-mode `etaf=0` for all factors incl. Land (`_aft_etaf`: "GAMS uses etaf=0 for ALL
  fm incl sluggish Land") → the specific-factor supply curve is **vertical**.
- Land in EU_28 feeds a **single** sector (Food) → `gf_share≈1`; re-deriving it is a no-op.
- The settled `pft` (0.845) is a **whole-system result**, not a re-derivable parameter.

Verified numerically on the reference GDX:

| `pft[EU_28,Land]` response | value |
|---|---|
| shock vs RAW base (1.0) — the check-contaminated GAMS path | **−18.09%** |
| shock vs SETTLED base (=check 0.845) | **−3.03%** |
| GEMPACK `pfe[Land,Food,EU_28]` (sl4dump) | **−2.681%** |

−3.03% ≈ −2.68%. **Correct mechanism: adopt the settled (check) point as the base.**

## Design

`FactorBlock.calibrate_base()` runs the settle solve ONCE (the `base→check→shock` stack
reused as a calibration tool) and returns the settled check-period point. The composer's
`build_block_model(base_calibrated=True)` runs it and stamps `m._settled_seed` +
`m._base_calibrated`. The driver, when `_base_calibrated`, seeds the base period from the
settled point, **skips the check phase**, and seeds the shock from base. The check
*calculation* is used once at calibration time, outside the simulation — it is not
re-introduced into the sim path.

The settled point is a given base (already an equilibrium from the settle solve), so the
calibrated mode uses `skip_base_solve=True` — like the faithful-to-GAMS base.

## Results (gtap7_3x3)

- **Non-regression (gate #1):** `run_parity_gates.py` GREEN — the default
  `base→check→shock` is 0-diff vs GAMS, untouched. Form gate 14/14.
- **Against GEMPACK (gate #2):** base-calibrated `pft[EU_28,Land]` base=0.845 →
  shock=0.819 = **−3.03%**, all periods `code=1`, no check phase. GEMPACK: −2.68%.
  `measure_gempack_blocks.py` reports `gap_vs_gempack_pp = −0.352`.
- **Against GEMPACK — the WHOLE quantity set (not just the land price):** across all
  15 mapped `Q_TO_VAR` variables (190 cells), the overall match within 1pp rises
  **75.8% (default) → 96.3% (base-calibrated)**, median error 0.38pp → 0.17pp — the
  same figure F3 measured "excluding check-movers". The Food/Land-chain quantities
  (`xp`/`xda`/`xaa`/`xma`/`xd`) go to 100%.

### Don't re-base the derived-demand vars

An early version seeded ALL 83 settled var families into the base; that regressed
`xg`/`xc` (government/household demand) by ~1.3pp on the EU_28 cells (78% match).
Those are DERIVED-demand vars (computed from prices+incomes), so seeding them at the
check value is a small inconsistency. Skipping them from the base seed
(`_F35_DERIVED_DEMAND` = xg/xc/xg_agg/xi/xiagg/yg/yc/zcons/u*/ev/cv) lets them
re-derive consistently → `xg`/`xc` 78%→100%, overall 94%→96.3%. The equilibrium core
(prices, factor/production/trade quantities) IS seeded. This is exactly the "seed
only the equilibrium vars" narrowing flagged as a risk in the plan.

### Cross-check against Julia (the same-methodology engine)

Julia (mivanic/GlobalTradeAnalysisProjectModelV7.jl) uses the SAME base-calibrated
methodology as F3.5 (an explicit `calibrate()` → `calibrated_data` → single shock
solve, no check period) — so it is the natural oracle for the mechanism (unlike
GEMPACK, it has no linearized-formulation difference). Running Julia's own sample
data end-to-end (`calibrate → +10% tariff → run_model!`, `scratchpad/
julia_mechanism_probe.jl`, Julia instantiated + Ipopt solve `Optimal Solution
Found`): the **largest specific-factor price move calibrated-base→shock is 5.36%**
(single-digit) — Julia's calibrated base does NOT re-settle the specific factor the
−15% way. This confirms F3.5's mechanism with Julia's engine executing (not just
reading its code): calibrate the base → the specific-factor price responds only to
the shock. F3.5's −3.03% on gtap7_3x3 is the same behavior. A full cell-by-cell
port of gtap7_3x3 into Julia (via `HeaderArrayFile.jl`, same author) is a follow-up
(its own mini-phase) for exact-number parity.

### Why not exactly −2.68%?

The residual 0.35pp is **irreducible formulation difference**, not a defect. GAMS/equilibria
solve an exact power-CET (`pft^(1+ω)=Σ gf·pfy^(1+ω)`); GEMPACK solves its **linearized**
form (`qes=qe−ETRAE·(pes−pe)`). Different equations give −3.03% vs −2.68% even at the same
base and shock. Closing it would require adopting GEMPACK's linearized equation — which
would break the default's `0-diff` vs GAMS. equilibria is faithful to GAMS; GAMS is faithful
to its own CET; the residual is an engine-level tie, consistent with F3's finding.

## What shipped

- `src/equilibria/blocks/gtap/factor.py` — `FactorBlock.calibrate_base()`
- `src/equilibria/templates/gtap/gtap_block_model.py` — `build_block_model(base_calibrated=)`
- `src/equilibria/templates/gtap/gtap_multiperiod_driver.py` — seed base from settled + skip check
- `scripts/gtap/measure_gempack_blocks.py` — against-GEMPACK land-response measurement
- `tests/templates/gtap/test_f3_5_base_calib.py` — 5 tests (spike oracle + calibrate + composer + end-to-end + against-GEMPACK)

## Reproduction

```bash
# the against-GEMPACK measurement (needs PATH + the committed sl4dump)
uv run python scripts/gtap/measure_gempack_blocks.py --dataset gtap7_3x3 \
    --base-calibrated --gempack-har tests/fixtures/gtap7_gempack/sl4dump_gtap7_3x3_tm10.har
# → land_resp_pct -3.03, gempack_land_pct -2.681, gap_vs_gempack_pp -0.352

# the F3.5 tests
uv run python -m pytest tests/templates/gtap/test_f3_5_base_calib.py -v

# non-regression (default 0-diff vs GAMS)
uv run python scripts/gtap/run_parity_gates.py
```

## Scope / YAGNI

- Only the specific factors are calibrated (the 97 base→check movers all cascade from one
  root: the land price). No Julia-style 6-phase calibration of Armington/production/demand.
- `gtap7_3x3` only for now; extend to other datasets in follow-up.
- The default mode is never changed — F3.5 is a selectable variant.
