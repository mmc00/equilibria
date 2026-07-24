# GTAP7 vs GEMPACK — linearization study

This page quantifies **why** the against-GEMPACK quantity match is only ~52% at a
+10% global bilateral tariff, across all five matrix datasets. It measures the
match% as (1) the shock shrinks and (2) GEMPACK's Gragg solution method is refined
— two independent angles on the claim that the residual is the Gragg-linearized↔
levels method gap, **not** a model defect.

**Reference number (PR #40):** on Horridge's small SIMPLE shock, GAMS(levels) ≡
GEMPACK(linearized) at 100% within 1pp. This study extends that to the real GTAP7
matrix.

## Scope

Datasets: `gtap7_3x3`, `gtap7_3x4`, `gtap7_5x5`, `gtap7_10x7`, `gtap7_15x10`.
(nus333 / 9x10 remain out of the matrix scope.) Metric: fraction of mapped quantity
cells within **1 percentage point** (absolute pp on %-changes — GEMPACK output is
%-change, so the comparison is |Δ(%change)| ≤ 1pp, not a relative tol).

Cells read **—** where the Windows-produced GEMPACK fixture is not yet present.

## Evidence 1 — shock-size sweep (default Gragg steps)

A uniform `tm` shock at decreasing magnitude. As the shock → 0 the match → 100%,
confirming the residual is shock-size (linearization). The **non-linearity** column
is |match(10%) − match(0.1%)|.

| dataset | 10% | 3% | 1% | 0.3% | 0.1% | non-linearity |
|---|---|---|---|---|---|---|
| `gtap7_3x3` | 76% | — | — | — | — | — |
| `gtap7_3x4` | — (no seed GDX) | — | — | — | — | — |
| `gtap7_5x5` | 73% | — | — | — | — | — |
| `gtap7_10x7` | 64% | — | — | — | — | — |
| `gtap7_15x10` | 52% | — | — | — | — | — |

## Evidence 2 — Gragg refinement (fixed 10% shock)

Solving the SAME 10% shock with a finer Gragg subinterval count. A match% that
rises with Steps isolates GEMPACK's numerical method as the source, not the model.

| dataset | Steps 4 | Steps 8 | Steps 16 | Steps 32 | Steps 64 |
|---|---|---|---|---|---|
| `gtap7_3x3` | — | — | — | — | — |
| `gtap7_3x4` | — | — | — | — | — |
| `gtap7_5x5` | — | — | — | — | — |
| `gtap7_10x7` | — | — | — | — | — |
| `gtap7_15x10` | — | — | — | — | — |

## Variables compared

The quantity map (`Q_TO_VAR` in `gempack_reference.py`) — GEMPACK quantity → Python
Var, verified empirically and by economic meaning:

| GEMPACK | Python Var |
|---|---|
| `qfd` | `xda` |
| `qfm` | `xma` |
| `qfa` | `xaa` |
| `qxs` | `xw` |
| `qxw` | `xet` |
| `qms` | `xmt` |
| `qds` | `xd` |
| `qpa` | `xc` |
| `qga` | `xg` |
| `qc` | `xs` |
| `qe` | `xft` |
| `qtm` | `xtmg` |
| `qinv` | `xiagg` |
| `qva` | `xp` |
| `qgdp` | `rgdpmp` |

## Welfare (diagnostic only)

Welfare EV is read from GEMPACK's `decomp.har` (WELVIEW) via `gempack_welfare_ev`
and compared to Python's EV by the 3-branch decomposition (allocative /
terms-of-trade / investment-savings). It is **not** floor-gated: raw welfare `u` is
a sign-flipping second-order quantity where even GAMS and GEMPACK disagree — see
`docs/findings/gempack_welfare_not_cellwise_2026-07-23.md`. This section reports the
EV$ decomposition side by side where a `decomp.har` fixture is present, purely as a
diagnostic.

## ifSUB is faithful

Python's ifSUB condensation reproduces GAMS to the bit (mac gate #1,
`verify_ifsub_equivalence.py`; primary quantity block 99.3–99.9% consistent across
modes on 16,424 cells). See
`docs/findings/gtap_ifsub_is_faithful_not_inverted_2026-07-24.md`.

## Provenance

- van der Mensbrugghe, *The Standard GTAP Model in GAMS, Version 7* (Table D.1:
  `ifSUB` = model condensation, not economics; GEMPACK is a Johansen percent-change
  solver, condensed by nature).
- Horridge, *tpmh0103* SIMPLE model (the GAMS-vs-GEMPACK reference, PR #40).
