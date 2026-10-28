# Welfare: regional EV and its decomposition

GTAP reports welfare as the equivalent variation (EV) of each regional
household. equilibria gives you two things:

- **The regional EV** (`regional_ev`), computed on the solved block model.
  This is the number to read after a shock.
- **The Huff (1996) / McDougall (2003) decomposition** of that EV into
  allocative efficiency, terms of trade, investment-saving, endowment and
  technical contributions, in the RunGTAP `WELVIEW.har` convention. The
  formulas are in `templates/gtap/welfare_decomp.py`; read
  [its status](#status-of-the-decomposition) before using it.

## Regional EV

GTAP defines EV in `gtapv7.tab` §11: the income that, at the prices before
the shock, reaches the utility after it, minus the income before. In levels
that is the model's own household block (`eq_yc` … `eq_u`) solved with the
prices of the `check` period, utility fixed at its `shock` value and income
free. `regional_ev` does exactly that and leaves the model as it was:

```python
from equilibria import load_bundled
from equilibria.templates.gtap import apply_shock
from equilibria.templates.gtap.gtap_block_model import build_block_model
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod
from equilibria.templates.gtap.welfare_regional import regional_ev

# The model of the GTAP quickstart: +10 % on the tariff power of one route.
params = load_bundled("gtap", "nus333")
closure = GTAPClosureConfig(
    name="base", closure_type="MCP", capital_mobility="sluggish",
    fix_endowments=False, fix_taxes=False, fix_technology=False,
    if_sub=False, numeraire="pnum",
)
m, _ = build_block_model(params, params.sets, closure, "ROW")
apply_shock(m, {"imptx": {("USA", "MFG", "ROW"): 10.0}})
solve_multiperiod(m, params, closure, mode="gtap", skip_base_solve=True, solve_check=True)

print(regional_ev(m))  # in units of regy
# {'USA': -0.1079..., 'ROW': 0.0885...}
```

By default it reads `period="shock"` at the prices of `ref="check"`. The
model's `regy` is in the scaled units of the benchmark, so divide by
`regy[r, "check"]` for EV as a share of income.

`tests/templates/gtap/test_nus333_ev_regional.py` checks it against
GEMPACK's EV (Gragg) on five exercises of the Burfisher book (TBL62A, TBL45A,
TBL45C, TBL63A, ME7A), within 0.1 % of income, and checks that EV is zero
when nothing is shocked.

## The decomposition

```
EV_r ≈ A_r + T_r + IS_r + ENDW_r + TECH_r + residual
```

### Components

| Component | Formula (per region r) | Captures |
|---|---|---|
| **A** — Allocative efficiency | `Σ_b τ_b · Δq_b` over 11 distortion sources (see below) | Welfare change from activity moving across tax wedges |
| **T** — Terms of trade | `(Σ pfob_0·Δxw − Σ pcif_0·Δxw) / pnum` | Gain/loss from world-price changes (deflated by numeraire) |
| **IS** — Investment-Saving | `(Δyi − Δxi) / pnum` | Imbalance in global savings pool, pnum-deflated |
| **ENDW** — Endowment | `Σ_f pf_0 · Δxft` | Only non-zero if endowments are exogenously shocked |
| **TECH** — Technical change | `Σ vom_0 · Δao` (currently stub) | Only non-zero if aoreg/afe/aocgds shocked |
| **EV** — Total | `vpm_base · (uh_shock − uh_base)` | Hertel (1997) $-equivalent utility change |

#### 11 allocative sub-buckets

Matching RunGTAP `ALEF` header (source × region):

| Bucket | Distortion | Tax rate | Quantity change |
|---|---|---|---|
| `ptax` | Output tax | `rto` | `Δvom` |
| `imptx` | Import tariff | `rtms` | `Δvmsb` (credited to importer) |
| `exptx` | Export tax/subsidy | `rtxs` | `Δvxsb` (credited to exporter) |
| `dftax` | Factor tax (domestic) | `(evfb-evos)/evfb` | `Δevfb` |
| `mftax` | Factor tax (imported) | merged into dftax | — |
| `dctax` | Private consumption tax (dom) | `(vdpp-vdpb)/vdpb` | `Δvdpb` |
| `mctax` | Private consumption tax (imp) | `(vmpp-vmpb)/vmpb` | `Δvmpb` |
| `dgtax` | Gov consumption tax (dom) | `(vdgp-vdgb)/vdgb` | `Δvdgb` |
| `mgtax` | Gov consumption tax (imp) | `(vmgp-vmgb)/vmgb` | `Δvmgb` |
| `ditx` | Investment tax (dom) | `(vdip-vdib)/vdib` | `Δvdib` |
| `mitx` | Investment tax (imp) | `(vmip-vmib)/vmib` | `Δvmib` |

### Status of the decomposition

Measured on `main` on 2026-10-28, the documented way to run it does not work:
`scripts/gtap/run_gtap.py validate-shock --welfare-decomp` fails with its
default `--path-capi-mode linear` (`Expression 3 is nonlinear; linear adapter
cannot handle it`), and with `--path-capi-mode nonlinear` both the baseline
and the shocked solve fail (`res=inf`) and the CSV it writes is all zeros.
That CLI still builds the old single-file model and applies the shock to the
parameters, the path ADR-0003 removed.

`compute_welfare_decomposition(base_params, shock_params, base_levels,
shock_levels)` and its homotopy variant take snapshots of levels, which today
only that script builds. Moving the decomposition onto the block model is
tracked with the monolith's retirement (`R-MONOLITH` in `ROADMAP.md`). Until
then, use `regional_ev` for the EV itself.

### Why a residual?

RunGTAP uses GEMPACK's **linearized** model: every variable is already a percent change, and the Huff decomposition is an algebraic identity that closes to zero by construction (Euler/Gragg integration internally).

Equilibria uses a **levels** MCP model with absolute quantities and prices. The Huff formula `τ_b · Δq_b` is a first-order approximation to the exact path integral `∫ τ_b(t) · dq_b(t)`. The second-order error term `½ Σ Δτ_b · Δq_b` is typically 1-3% of EV for 10% shocks.

The homotopy variant collapses this error by computing the Huff formula on N small segments instead of one big segment — error is O(1/N²). N=4 brings the residual below the parity tolerance Equilibria already enforces against GAMS (1e-6).

## Shadow demand system (gtapv7.tab §11 port)

For shocks under **non-default closures** — most notably `capFix` with the
`swap dpsave(r) = del_tbalry(r)` recipe used to fix the trade-balance /
world-income ratio — the simple 5-term Huff decomposition does not match
RunGTAP exactly because RunGTAP routes the welfare calculation through an
auxiliary *shadow demand system* (gtapv7.tab section 11) that picks up
preference-shift effects from `dpsave`.

Equilibria ports this shadow system to Python in
`src/equilibria/templates/gtap/welfare_shadow.py`. It exposes a single
function `integrate(...)` that solves the chain

```
E_qpev → E_ueprivev → E_dpavev → E_uelasev → E_ypev/E_ygev/E_ysaveev → E_yev → E_EV
```

step-by-step under one of four integrators:

| Method | Order | Use case |
|---|---|---|
| `euler` (default, N=25) | O(h) | **Recommended for RunGTAP parity** — calibrated to GEMPACK's Gragg-8-16-32 discretisation |
| `midpoint` | O(h²) | RK2 alternative; diverges from RunGTAP by ~2% at high N |
| `gragg` | O(h²) leapfrog | Gragg modified midpoint, single ladder |
| `bulirsch_stoer` | O(h^2k) | Gragg + Richardson on a ladder (academic reference; converges to the asymptotic ODE fixed point, **not** to RunGTAP's discrete answer) |

### Inputs

`integrate(base, u_pct=..., dpsave_pct=...)` takes a `ShadowBaseline` for one
region (its private, government and saving expenditure, income, and the
per-commodity `VPP`, `INCPAR` and `ALPHA` of the benchmark) plus the
cumulative percentage changes of utility `u` and of `dpsave`, and returns the
EV in USD millions. The worked example,
`runs/nus333_compare/verify_shadow_integrator.py`, reads those benchmark
values from a RunGTAP export (`runs/nus333_compare/rungtap/basedata.xls`) that
is not shipped with the repository, so it runs only where that export exists.

## RunGTAP parity validation of the shadow integrator (NUS333 + 9x10)

Measured on 2026-05-20 (PR #10) and not re-measured since; the regional EV
above has its own, newer test against GEMPACK. Both datasets run under capFix closure; RunGTAP gets the swap
`dpsave(r)=del_tbalry(r)` for all non-residual regions, equilibria uses
`GTAPClosureConfig(savf_flag='capFix')`. Shock: 10% uniform `tm_pct`.

### NUS333 (2 regions × 3 sectors, residual=ROW)

| Variable | equilibria | RunGTAP | gap |
|---|---:|---:|---:|
| u USA %Δ | +0.1649% | +0.1725% | 0.008 pp |
| u ROW %Δ | -0.8199% | -0.8308% | 0.011 pp |
| **EV USA ($M)** | +14,888 | +14,933 | **-0.30%** |
| **EV ROW ($M)** | -306,987 | -308,210 | **-0.40%** |

### 9x10 (10 regions × 10 commodities, residual=NAmerica)

Per-region EV via `welfare_shadow.integrate()`:

| Region | equilibria | RunGTAP | gap | dpsave |
|---|---:|---:|---:|---:|
| Oceania | -8,529 | -8,556 | -0.32% | -1.27 |
| EastAsia | -86,648 | -86,942 | -0.34% | -1.24 |
| SEAsia | -24,141 | -24,593 | **-1.84%** | -14.23 |
| SouthAsia | -12,189 | -12,221 | -0.26% | +0.84 |
| NAmerica (residual) | -4,555 | -4,555 | -0.01% | 0.00 |
| LatinAmer | -22,464 | -22,521 | -0.25% | +2.99 |
| EU_28 | -81,358 | -85,873 | **-5.26%** | -25.99 |
| MENA | -47,377 | -47,667 | -0.61% | +1.32 |
| SSA | -13,572 | -13,634 | -0.45% | +1.51 |
| RestofWorld | -49,663 | -50,042 | -0.76% | -2.95 |
| **WORLD** | **-350,497** | **-356,605** | **-1.71%** | |

### Summary

| Metric | NUS333 | 9x10 |
|---|---:|---:|
| Avg per-region `u` gap | 0.010 pp | 0.114 pp |
| EV WORLD gap | 0.35% | 1.71% |
| Regions with EV gap <1% | 2/2 | 8/10 |
| cgebox tolerance | 5% | 5% |

The two outlier regions in 9x10 (SEAsia, EU_28) have extreme `|dpsave|`
(>14% and >26% respectively) — the linear-trajectory approximation in the
shadow integrator degrades when the closure absorbs very large
trade-balance imbalances. The remaining 8 regions reproduce RunGTAP
welfare to within 1% — well inside the cgebox-tolerated 5% residual band.

Full reproducible workspace lives under `runs/nus333_compare/` and
`runs/9x10_compare/`. See `verify_shadow_*.py`, `compare_*_report.py`
and the corresponding `comparison.md` files.

## References

- Huff, K.M. and T.W. Hertel (1996), "Decomposing Welfare Changes in the GTAP Model," GTAP Technical Paper 5.
- Hertel, T.W. (1997), *Global Trade Analysis: Modeling and Applications*, Cambridge University Press, Chapter 3.
- gtapv7.tab section 11 (Equivalent Variation), in the GTAPv7 tablo source distributed with RunGTAP 3.75.
- McDougall, R.A. (2003), "A New Regional Household Demand System for GTAP," GTAP Technical Paper 20.
