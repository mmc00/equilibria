# Against-GEMPACK linearization study — status (2026-07-24)

**Status:** Tooling complete (mac side); the Windows GEMPACK grid is the remaining
step. This finding consolidates what the study established and what it measured so
far. Spec: `gempack_linearization_study_spec_2026-07-24.md`; plan:
`gempack_linearization_study_plan_2026-07-24.md`.

## The question

Is the against-GEMPACK quantity residual (~52% within 1pp at a +10% global bilateral
tariff on gtap7_15x10) the Gragg-linearized↔levels method gap, or a model defect?

## What the study built (mac side, all committed)

Four evidences, of which the two fidelity gates run entirely on mac and are done:

1. **ifSUB fidelity (evidence 3) — DONE.** `verify_ifsub_equivalence.py`: the
   PRIMARY quantity block (xw/xet/xp/… — solved explicitly in both modes) is
   consistent across ifSUB modes at **99.33–99.91%** over 4 datasets (16,424
   cells). The Python model is faithful to GAMS under condensation. See
   `gtap_ifsub_is_faithful_not_inverted_2026-07-24.md` — a first "flag inverted"
   verdict was disproved by running GAMS locally in both modes; the symptom was
   reading `V(m.xwmg)` (a substituted-out report var = seed) instead of the
   substituted expression `tmarg·xw` (= 100% vs GAMS, rel 1e-6).

2. **Shock-size sweep (evidence 1) + Gragg refinement (evidence 2) — tooling DONE,
   data pending Windows.** `run_gempack_matrix.py --grid` emits, per dataset, the
   shock sweep (10/3/1/0.3/0.1% at Steps 8 16 32) + Gragg sweep (Steps 4/8/16/32/64
   at 10%) = 10 config-tagged `.cmf` files; `--emit-bat` writes the Windows driver.
   `gen_linearization_study.py` builds the page from whatever fixtures exist.

3. **Welfare (evidence 4) — reader DONE.** `gempack_welfare_ev()` reads the EV
   decomposition from `decomp.har` (WELVIEW) by header name; diagnostic only (no
   floor — welfare `u` sign-flips across engines, see
   `gempack_welfare_not_cellwise_2026-07-23.md`).

## Measured so far (10% baseline, Python↔GEMPACK within 1pp)

From the existing `sl4dump_<ds>_tm10.har` fixtures:

| dataset | 10% match |
|---|---|
| gtap7_3x3 | 76% |
| gtap7_3x4 | — (no seed GDX) |
| gtap7_5x5 | 73% |
| gtap7_10x7 | 64% |
| gtap7_15x10 | 52% |

The 15x10 52% reproduces the motivating number. The gradient (larger dataset →
lower 10% match) is the expected non-linearity signature: more regions/commodities
→ larger cross-effects → a shock further into the non-linear regime. The
shock-sweep and Gragg columns are "—" pending the Windows grid.

## What remains

Run the grid on Windows (RunGTAP + GEMPACK), guide §10:

```bat
uv run python scripts\gtap\run_gempack_matrix.py --grid --no-solve --emit-bat
cd runs\gempack_matrix && run_study_windows.bat
uv run python scripts\gtap\gen_linearization_study.py
```

~50 solves. Expected: each dataset's shock-sweep row climbs toward ~100% as the
shock → 0.1%, and its Gragg row rises with Steps — the quantitative confirmation
that the residual is the linearization/method gap, consistent with PR #40's SIMPLE
GAMS≡GEMPACK = 100% on a small shock.

## Two minor items surfaced (cosmetic, not blockers)

1. `xwmg`/`xmgm` are missing from `_recompute_ifsub_report_vars` — a report-var
   read gap (the real equations use the macro; the coverage matrix is unaffected).
2. Fixture `out_gtap_shock_ifsub0.gdx` (gtap7_3x3) is corrupt in the margin block
   (its `xwmg` carries ifSUB=1 values) — worth regenerating.
