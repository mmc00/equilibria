# ifSUB is faithful — the "inversion" was a report-var read artifact (2026-07-24)

**Status:** Finding — the Python GTAP model's ifSUB condensation is faithful to
GAMS. No flag inversion, no margin-condensation bug. Recorded because the
investigation passed through two wrong verdicts before landing here.

## Context

Mac gate #1 of the against-GEMPACK linearization study
(`scripts/gtap/verify_ifsub_equivalence.py`) originally asserted "Python ifSUB=1 ≡
ifSUB=0 in post-shock levels." It failed at 96.5% on gtap7_3x3, diverging only in
`xwmg`/`xmgm` (the international trade-and-transport margin block). Chasing that led
to a false "the ifSUB flag is inverted" verdict, which running GAMS locally in both
modes then dissolved.

## What is actually true

Running GAMS 53 locally in both modes (same `.gms`, only `$setGlobal ifSUB 1/0`
differs), comparing the SUBSTITUTED expression the model actually solves:

| Under `if_sub=True` (condensed), read… | vs GAMS ifSUB=1 |
|---|---|
| raw `V(m.xwmg)` Var | 41% within 0.1% |
| `tmarg·xw` (the substituted expression) | **100%** within 0.1% (max rel 1.3e-6) |

The model is **faithful**. `if_sub=True` deactivates the margin equations
(`eq_xwmg`/`eq_xmgm` absent when True, 48/48 active when False) and the real
equations use the algebraic macro `_m_xwmg = tmarg·xw` directly
(`gtap_model_equations.py:5504`) — exactly GAMS's ifSUB=1 condensation.

## Root cause of the false signal

`xwmg`/`xmgm` become substituted-out **report vars** under condensation. Their Var
objects keep their seed value; the solution lives in `tmarg·xw`. The driver's
`_recompute_ifsub_report_vars` (`gtap_multiperiod_driver.py:2295`) exists precisely
to refill such vars post-solve — its docstring warns *"a direct read mis-reports
them… GAMS recomputes them post-solve in postsim"* — and it recomputes
`pfa/pfy/pp/pwmg/pefob/pmcif/pm` **but omits `xwmg`/`xmgm`**. So `V(m.xwmg)` returns
the stale seed. The gate read the seed, not the solution.

Why the coverage matrix is unaffected: the REAL equations use the macro, and the
study's `Q_TO_VAR` map compares `xw` (bilateral trade), not `xwmg`. Only a direct
read of the two report vars is incomplete. The matrix passes 99-100%.

## The corrected gate

`verify_ifsub_equivalence.py` now compares the PRIMARY quantity block
(`xw`/`xet`/`xp`/`xd`/… — solved explicitly in both modes) across ifSUB modes,
skipping the substituted-out report vars. Measured agreement:

| dataset | cells | agree% |
|---|---|---|
| gtap7_3x3 | 297 | 99.33% |
| gtap7_5x5 | 1005 | 99.60% |
| gtap7_10x7 | 3962 | 99.82% |
| gtap7_15x10 | 11160 | 99.91% |

The residual is the margin-driven bilateral-trade cells (which legitimately move
between modes, as GAMS's own `xwmg` does — 16/27 cells) plus solver-tol noise on
`rgdpmp` (~1e-3). ifSUB modes are NOT expected to be bit-identical; the gate checks
the primary block is not corrupted by the switch, and per-mode GAMS fidelity is the
coverage matrix's job.

## Two minor items (cosmetic, not blockers)

1. `xwmg`/`xmgm` are missing from `_recompute_ifsub_report_vars` — a report-var gap.
   A faithful fix adds them (compute `tmarg·xw`; note `tmarg` is a param indexed
   `(r,i,rp)` WITHOUT the `t` axis), with the coverage matrix as the gate (it won't
   move — the real equations already use the macro).
2. Fixture `out_gtap_shock_ifsub0.gdx` is separately corrupt in the margin block
   (its `xwmg` carries ifSUB=1 values). Worth regenerating.

## Lesson

Before calling a Python-vs-GAMS gap a bug: (a) run GAMS in BOTH arms of the switch,
(b) pair against the symbol the pipeline actually reads, and (c) check whether the
var is a substituted-out report var under the active mode — a raw `V(var)` read of a
condensed var is the seed, not the solution.

## Repro
`scratchpad/final_proof.py` (raw Var vs `tmarg·xw` under condensation, gtap7_3x3),
GAMS runs under `scratchpad/gams_ifsub{0,1}/`.
