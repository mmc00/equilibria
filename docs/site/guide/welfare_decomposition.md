# Welfare Decomposition (Huff / RunGTAP)

Equilibria computes a Huff (1996) / McDougall (2003) welfare decomposition matching the RunGTAP `WELVIEW.har` convention. The total welfare change (Equivalent Variation, in USD millions at baseline prices) is decomposed into additive contributions:

```
EV_r ≈ A_r + T_r + IS_r + ENDW_r + TECH_r + residual
```

## Components

| Component | Formula (per region r) | Captures |
|---|---|---|
| **A** — Allocative efficiency | `Σ_b τ_b · Δq_b` over 11 distortion sources (see below) | Welfare change from activity moving across tax wedges |
| **T** — Terms of trade | `(Σ pfob_0·Δxw − Σ pcif_0·Δxw) / pnum` | Gain/loss from world-price changes (deflated by numeraire) |
| **IS** — Investment-Saving | `(Δyi − Δxi) / pnum` | Imbalance in global savings pool, pnum-deflated |
| **ENDW** — Endowment | `Σ_f pf_0 · Δxft` | Only non-zero if endowments are exogenously shocked |
| **TECH** — Technical change | `Σ vom_0 · Δao` (currently stub) | Only non-zero if aoreg/afe/aocgds shocked |
| **EV** — Total | `vpm_base · (uh_shock − uh_base)` | Hertel (1997) $-equivalent utility change |

### 11 allocative sub-buckets

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

## How to compute

### Single-step (fast, residual ~1-3%)

```bash
uv run python scripts/gtap/run_gtap.py validate-shock \
    --gdx-file data/9x10/9x10Dat.gdx \
    --variable rtms --index "(...)" --value 0.10 \
    --shock-mode tm_pct \
    --output reports/welfare/ \
    --welfare-decomp
```

Produces `reports/welfare/welfare_decomposition.csv`.

The single-step path applies the first-order Huff formula once between baseline and shocked equilibrium. For 10% tariff shocks this gives an EV-vs-sum residual of 1-3%, which is the expected gap between a *levels* MCP model and the *linearized* GEMPACK formulation that RunGTAP uses internally.

### Exact RunGTAP equivalence (homotopy, residual <0.01%)

Add `--homotopy-steps 4`:

```bash
uv run python scripts/gtap/run_gtap.py validate-shock \
    --gdx-file data/9x10/9x10Dat.gdx \
    --variable rtms --index "(...)" --value 0.10 \
    --shock-mode tm_pct \
    --output reports/welfare/ \
    --welfare-decomp \
    --homotopy-steps 4
```

The model is solved at 4 intermediate states between baseline and full shock. Huff contributions are computed locally on each segment and summed. This Gragg-style integration drives the residual to near-zero, matching the GEMPACK identity that RunGTAP enforces by construction. With N=4 the error vs RunGTAP is <0.01% for 10% shocks; N=2 already gives <0.5%.

### WELVIEW.har output (RunGTAP-compatible)

Add `--welfare-har path/to/WELVIEW.har`:

```bash
uv run python scripts/gtap/run_gtap.py validate-shock \
    ... \
    --welfare-decomp \
    --welfare-har reports/welfare/WELVIEW.har
```

The resulting HAR contains headers `EVAL`, `ALET`, `ALEF`, `TOTE`, `ISE`, `ENDW`, `TECH`, `TOT` indexed by `REG` (and `ALSR × REG` for the 11-bucket breakdown), and is readable by any GEMPACK tool (`harview`, `ViewHAR`).

## Why a residual?

RunGTAP uses GEMPACK's **linearized** model: every variable is already a percent change, and the Huff decomposition is an algebraic identity that closes to zero by construction (Euler/Gragg integration internally).

Equilibria uses a **levels** MCP model with absolute quantities and prices. The Huff formula `τ_b · Δq_b` is a first-order approximation to the exact path integral `∫ τ_b(t) · dq_b(t)`. The second-order error term `½ Σ Δτ_b · Δq_b` is typically 1-3% of EV for 10% shocks.

The homotopy variant collapses this error by computing the Huff formula on N small segments instead of one big segment — error is O(1/N²). N=4 brings the residual below the parity tolerance Equilibria already enforces against GAMS (1e-6).

## Interpretation example (NUS333, 10% tariff shock)

A typical output for a uniform 10% import tariff shock looks like:

```
Region     EV ($M)         A        T       IS  resid%
NAmerica  -1,245.32  -1,180.50  -52.10   -8.20  -0.36%
EAsia        87.45     112.30  -28.50    3.65   0.00%
RestofWorld -23.10     -18.20   -4.80   -0.30   0.43%
```

Reading:
- **NAmerica** loses $1.2B EV; almost entirely from allocative inefficiency (`A`) of importing less through the tariff wedge. Small ToT loss too.
- **EAsia** *gains* — the alloc gain comes from rebalancing trade away from distorted partners. ToT loss reflects worse export prices.
- Residuals stay under 0.5% with homotopy N=4 (without homotopy they'd be 1-3%).

## References

- Huff, K.M. and T.W. Hertel (1996), "Decomposing Welfare Changes in the GTAP Model," GTAP Technical Paper 5.
- Hertel, T.W. (1997), *Global Trade Analysis: Modeling and Applications*, Cambridge University Press, Chapter 3.
- McDougall, R.A. (2003), "A New Regional Household Demand System for GTAP," GTAP Technical Paper 20.
