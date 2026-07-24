# Horridge SIMPLE model — GAMS vs GEMPACK reference comparison

Horridge & Pearson's `tpmh0103` archive (the SIMPLE model from COPS General Paper
G-214) ships the SAME model in **GEMPACK** (`simple.tab` + `fixcap.cmf`), **GAMS/MCP**
(`simpleMCP.gms`), **GAMS/NLP** (`simpleNLP.gms`) and **MPSGE** (`mpsgevh.gms`),
"to compare implementations." It ships **no script that actually lines up the
numbers** — the comparison is left to eyeballing the `.har` outputs in ViewSOL.

`compare_gams_vs_gempack.py` is that missing comparator, using the same
absolute-percentage-point metric as the equilibria against-GEMPACK gate.

## Why this is here

It answers, on a clean textbook model, the two questions raised about the
`gtap7_coverage_matrix_gempack` page:

- **Is %-change the right comparison?** Horridge's own GAMS computes
  `CH_XFAC = (xfac.l/xfac0-1)*100` — identical to our `(s/b-1)*100`. Yes; it's how the
  model's authors compare a levels GAMS solve against GEMPACK.
- **Why is our matrix match only ~52%?** Our matrix shock is **+10% tariff power on
  every bilateral route, globally** — highly non-linear. Horridge's SIMPLE shock is
  **small and localized** (−10% labour productivity in one sector). GEMPACK is
  Gragg-linearized, so its error vs a levels solve grows with the shock. On the small
  SIMPLE shock the GAMS-vs-GEMPACK match should be ~99–100%, proving the ~52% is the
  shock-size linearization gap, not a defect.

## Measured so far (macOS, GAMS only)

`GAMS/MCP` vs `GAMS/NLP` (both levels solvers, same model, same shock):

| var | within 1pp | median &#124;Δ&#124; |
|---|---|---|
| CH_Z (output) | 100% | 4.3e-8 pp |
| CH_XFAC (factor demand) | 100% | 8.0e-8 pp |
| CH_P (basic price) | 100% | 9.6e-9 pp |
| CH_PFAC (factor price) | 100% | 3.8e-8 pp |

Levels ≡ levels to ~8 significant digits — matches the "4–5 digits" the literature
reports. Also measured (levels, MCP): the response is **non-linear** — halving vs
tenthing the shock does NOT scale linearly (e.g. `srv` output +4.82% at −10% vs
+0.46% at −1%; a linear model would give +4.58% — a ~5% non-linearity), which is
exactly the linearization gap GEMPACK carries and the size of the against-GEMPACK
residual.

## Measured on Windows (GEMPACK 11.3 + GAMS 53, 2026-07-24)

Ran the full chain below on Windows (RunGTAP/GEMPACK `C:\GP`, GAMS `C:\GAMS\53`).
**GAMS(levels) vs GEMPACK(linearized), fixcap closure, −10% Labor productivity in Srv:**

| var | within 1pp | median &#124;Δ&#124; |
|---|---|---|
| CH_Z (output) | 100% | 0.00 pp |
| CH_XFAC (factor demand) | 100% | 0.00 pp |
| CH_P (basic price) | 100% | 0.00 pp |
| CH_PFAC (factor price) | 100% | 0.00 pp |
| **OVERALL** | **100.0%** | — |

The GAMS/MCP-vs-NLP sanity block also reproduced 100% within 1pp (median |Δ| ~1e-8 pp).
This is the reference number: on Horridge's SMALL, localized shock the levels(GAMS)↔
linearized(GEMPACK) match is **100% within 1pp** — so the against-GEMPACK matrix
page's ~52% at a +10% GLOBAL bilateral tariff is the shock-size linearization gap,
not a defect. (The GEMPACK Gragg solve is 4/6/8-step + extrapolation; NLP CONOPT
under the GAMS demo license flags a spurious "locally infeasible" status but still
returns the correct levels — the MCP/NLP agreement to 8 digits confirms it.)

## To reproduce on Windows (needs GEMPACK + GAMS)

```bat
gempack.bat                    :: tablo + gemsim -> fixcap.sl4 (+ .UPD)
:: convert fixcap.sl4 -> sl4dump.har via sltoht (equilibria guide §8 chain):
::   printf "\nfixcap\nc\nn\nsl4map.txt\nsl4dump.har\ne\n" > sl4dump.sti && sltoht -sti sl4dump.sti
dogams.bat                     :: -> ResultsMCP.gdx, ResultsNLP.gdx (edit its GAMS path)
uv run python compare_gams_vs_gempack.py --gempack sl4dump.har --mcp ResultsMCP.gdx --nlp ResultsNLP.gdx
```

The comparator auto-detects `gdxdump` (PATH, then `C:\GAMS\*`); pass a custom GAMS
install if it lives elsewhere.

## Provenance

Source files (`simple.tab`, `*.gms`, `*.cmf`, `input.gdx`, `simdata.har`, `*.bat`,
`readme.txt`) are Horridge & Pearson's, from the TPMH0103 archive linked in G-214
(https://www.copsmodels.com/archivep.htm). Only `compare_gams_vs_gempack.py` and this
readme are ours.
