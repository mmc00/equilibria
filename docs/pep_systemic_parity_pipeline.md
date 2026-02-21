# PEP Systemic Parity Pipeline

This workflow enforces parity with block-level fail-fast gates instead of ad-hoc residual inspection.

## What It Adds

- Equation contracts by block:
  - `production_tax_consistency`: `EQ29`, `EQ39`, `EQ40`
  - `trade_price_index_consistency`: `EQ79`, `EQ84`
  - `trade_market_clearing`: `EQ64`, `EQ88`
  - `macro_closure`: `EQ44`, `EQ45`, `EQ46`, `EQ87`, `EQ93`, `WALRAS`
- Block gates (`max_abs`, `rms`) evaluated in sequence
- Optional fail-fast behavior
- JSON trace report with top residuals and first failing block

## Command

```bash
uv run python scripts/run_pep_systemic_parity.py \
  --sam-file src/equilibria/templates/reference/pep2/data/SAM-CRI-gams-fixed.xlsx \
  --val-par-file src/equilibria/templates/reference/pep2/data/VAL_PAR-CRI-gams.xlsx \
  --dynamic-sam \
  --init-mode gams_blockwise \
  --method none \
  --fail-fast \
  --eq29-eq39-parity \
  --blockwise-commodity-alpha 0.75 \
  --blockwise-trade-market-alpha 0.5 \
  --blockwise-macro-alpha 1.0 \
  --save-report output/pep_systemic_parity_report.json
```

Use `--method ipopt` to run solve-stage gates after initialization.

## Interpretation

- If init gates fail, fix that block first before solving.
- If init gates pass but solve gates fail, issue is solver dynamics or cross-block coupling.
- Primary debug key is `first_failed_block` in the report.
- Report field `classification.kind` gives final attribution:
  - `data_contract`: SAM QA/init/parity contract failure
  - `solver_dynamics`: solve-stage convergence/gate failure after init passed
  - `pass`: all requested gates passed
- With `--eq29-eq39-parity`, the report adds a GAMS-anchored check:
  - `EQ29`: `TIPT = sum_j TIP(j)`
  - `EQ39_j`: `TIP(j) = ttip(j) * PP(j) * XST(j)`
  - `EQ40_i`: `TIC(i) = [ttic(i)/(1+ttic(i))] * [PD(i)*DD(i) + PM(i)*IM(i)]`
