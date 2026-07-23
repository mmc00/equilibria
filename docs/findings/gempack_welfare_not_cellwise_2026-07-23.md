# Raw welfare `u` is NOT cell-by-cell comparable across engines (2026-07-23)

**Status:** Finding — welfare is deliberately EXCLUDED from the against-GEMPACK
quantity page. Recorded here for reference.

## Context

The against-GEMPACK coverage page (`gtap7_coverage_matrix_gempack.md`) compares
post-shock **quantities** cell-by-cell in percentage points, where Python ≡ GAMS
to the bit and both differ from GEMPACK only by the small structural
linearized↔levels gap (median ~0.4pp). We probed whether **welfare** (`u`) could
join that page. It cannot.

## Measurement (gtap7_3x3, uniform 10% `tm` shock)

Per-region utility `u` %-change, three engines:

| Region | GEMPACK `u` | GAMS `u` (.gdx) | Python `u` |
|--------|------------:|----------------:|-----------:|
| USA    | −0.118%     | **+0.088%**     | +0.171%    |
| EU_28  | −0.784%     | **−0.052%**     | +0.811%    |
| ROW    | −0.651%     | −0.292%         | +0.348%    |

## Why welfare is different from quantities

1. **All three engines disagree** — unlike quantities (Python ≡ GAMS exact), here
   even GAMS differs from GEMPACK (opposite signs on USA/EU_28) AND from Python
   (same sign, ~2× magnitude). So there is no clean Python-vs-GEMPACK pp-residual
   to floor-gate.
2. **Welfare is the most sensitive CGE output.** It is a second-order quantity
   (a money-metric of a utility change), so the Gragg-linearized↔levels gap that
   is ~0.4pp on quantities is amplified into sign flips on `u`.
3. **Raw `u` from the multi-period solve is not the comparable object.** The
   established RunGTAP welfare parity (`rungtap_welfare_parity_2026-05-15.md`,
   nus333 + 9x10) reached agreement only via a purpose-built EV aggregator — the
   3-branch Huff/McDougall decomposition `EV_r = yc·Δuh + yg·Δug + rsav·Δus` and a
   shadow integrator — NOT by comparing the raw `u` variable.

## Decision

- Welfare stays **OUT** of the quantity page (mixing a sign-flipping, second-order,
  requires-special-aggregator variable with the clean 0.4pp quantity cells would be
  misleading).
- Welfare parity vs GEMPACK/RunGTAP lives in its **own track**
  (`compare_nus333_rungtap.py` / `compare_9x10_rungtap.py` + the EV aggregator),
  where it was validated to ~0.01–0.3pp on `u` and ~0.3–1.7% on EV.
- The Python-vs-GAMS ~2× welfare-`u` gap (USA +0.088 vs +0.171) is noted for future
  investigation but is NOT part of the against-GEMPACK quantity work — it is the
  same known welfare-sensitivity territory the EV track already handles.

## Repro
`scratchpad/check_welfare.py` (u/y/up vs GEMPACK) and `scratchpad/diag_welfare.py`
(u three-way GEMPACK/GAMS/Python), gtap7_3x3.
