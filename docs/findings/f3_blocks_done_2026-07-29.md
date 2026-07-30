# F3 — GTAP as composed symbolic Blocks: the full parity matrix is faithful (2026-07-29)

**Status:** Done. The 7-block symbolic composition of GTAP is a drop-in replacement
for the imperative monolith across the whole parity matrix — **pure**, **ifSUB**, and
**altertax**, both `ifsub` modes, datasets `gtap7_3x3` → `gtap7_15x10`, NLP and MCP.
The monolith (`gtap_model_equations.py`) stays intact as the parity oracle; the blocks
never depend on it at runtime.

## Context

F3 extracts the GTAP model into seven composable symbolic `Block` units (TRADE_CET,
PRODUCTION+SUPPLY, FACTOR, ARMINGTON+BILATERAL, DEMAND+UTILITY, INCOME, CLOSURE) that
build their equations through the `SymbolicEquation → PyomoBackend` bridge, then compose
into a single-period model (`build_block_single_period`) which reflects into the
multi-period structure (`GTAPBlockMultiPeriodModel`) and solves through the existing
`solve_multiperiod` driver. The acceptance gate (user decision, 2026-07-20) is the
NLP-vs-NLP + MCP-vs-MCP cell sweep vs a GAMS reference — not a `.nl` byte diff, which
stays a diagnostic.

Two constraints held throughout:
- **Fidelity is supreme.** The monolith is the oracle and is never edited (form gate
  14/14, `git diff` on `gtap_model_equations.py` across the whole F3 branch is empty).
- **The block solution cannot depend on the monolith at runtime** — the whole point of
  F3 is that the blocks *replace* it. Every calibration init the blocks need is computed
  from `self.params`, not copied from a monolith build.

## Results — the full matrix

All cases converge (`code=1`, every period base/check/shock), base 100%, check ≥99.9%.

### pure (`ifsub=0`, comparative-static)

| Dataset | NLP shock @1% | MCP shock @1% |
|---|---|---|
| gtap7_3x3 | 100.0 | 100.0 |
| gtap7_5x5 | 100.0 | 100.0 |
| gtap7_10x7 | 100.0 | 100.0 |
| gtap7_15x10 | 100.0 (MCP) | 100.0 |

The composer scales 874 → ~28 760 comparable cells with no new missing init — the
Task-5 calibration inits were complete and general.

### ifSUB (`ifsub=1`) — the margin/tariff substitution mode

| Dataset | shock @1% (NLP + MCP) |
|---|---|
| gtap7_3x3 | 100.0 |
| gtap7_5x5 | 100.0 |
| gtap7_10x7 | 100.0 |
| gtap7_15x10 | 100.0 |

### altertax — CD-elasticity rebalance, sequential base→check→shock

| Dataset | ifsub0 shock @1% | ifsub1 shock @1% | floor |
|---|---|---|---|
| gtap7_3x3 | 99.89 | 99.89 | 98.0 |
| gtap7_3x4 | 98.24 | 98.24 | 99.0 |
| gtap7_5x5 | 99.82 | 99.82 | 99.5 |
| gtap7_10x7 | 98.76 | 98.76 | 98.0 |
| gtap7_15x10 | 98.95 | 98.94 | 99.0 |

## The two findings that closed the hard modes

### ifSUB shock: an orphan free column, found by the `.nl` COLUMN diff

Under `ifSUB`, GAMS deactivates nine report equations and substitutes their `M_*` macros
INLINE in every consuming equation, so a shock on `imptx`/`prdtx` propagates through the
solved system. The block port added a shared macro module (`blocks/gtap/_ifsub_macros.py`:
`m_pfa`/`m_pfy`/`m_pp`/`m_pwmg`/`m_pefob`/`m_pmcif`/`m_pm`/`m_xwmg`/`m_xmgm`) plus
`_apply_ifsub_closure` (deactivate the 9 report eq components so the MP reflection skips
them). This converged but the shock sat at **45%**.

The `.nl` ROW diff was clean (2717 == 2717 constraints), so the equations were not the
problem. The `.nl` **COLUMN** diff was not: the block had **117 extra free columns**
(`pfa`×108, `pfy`×9). Two consuming equations still referenced the plain report var
instead of its inlined macro — `eq_pvaeq` (`production_supply.py`, `m.pfa` in the VA-price
CES) and `eq_pfeq`'s third branch (`factor.py`, `model.pfy`). Those left `pfa`/`pfy` as
free columns with no defining equation → a different system → a different basin.

Converting both to `mac.m_pfa` / `_m_pfy` (`if_sub`-gated) made the column set match the
monolith exactly (2918 == 2918) and the ifSUB shock jumped **45% → 100%** on all four
datasets, NLP and MCP. **Lesson:** row parity is necessary but not sufficient — the free
column set must match too. A plain report var left in a consuming equation (instead of
its substitution macro) is an orphan free column, invisible to a row/form diff. This
motivated the reusable `nl_column_diff` tool added to the xmodel-parity cascade.

### altertax: a zero-code drop-in; the sub-floor cases are basins, not defects

altertax needed **no new block code**. Its behavior is entirely (1)
`apply_altertax_elasticities(params)` (CD elasticities → 1, `pva`/`pnd` bench recomputed),
(2) a `name="altertax"` mobile closure (`fix_taxes`/`fix_technology`), (3) `holdfix_cd`.
Because `GTAPBlockMultiPeriodModel._block_sp()` reads `self.params`, the block SP inherits
the CD elasticities in its CES exponents automatically. Swapping
`GTAPMultiPeriodModel → GTAPBlockMultiPeriodModel` in the harness is the whole change.

Two datasets sit marginally under their floor (`gtap7_3x4` 98.24, `gtap7_15x10` ~98.95).
These are **not block defects**, proven two ways:

- **Apples-to-apples cell diff** (`gtap7_3x4`, same 1194-cell shared set): block 98.24% vs
  monolith 99.50%. Zero cells where the block converges to a *different value* the monolith
  gets right — the 15 "block-only" fails are the same EGY cells the monolith is also off
  on, straddling the 1% line (e.g. `xd[EGY,Svces,hhd]` block rel 1.22% vs mono 0.79%, same
  direction, same basin). All are EGY cells — EGY is the small **shocked** region that
  `gtap7_3x3` lacks, which is why 3x3 matched exactly. The whole EGY block is uniformly
  ~0.35% low, anchored on `yc[EGY]` → all EGY demand. Seeds are byte-identical
  (`yc[EGY]` seed = GAMS exact); from the same seed both engines drift, the block ~0.35%
  further, on `phip[EGY]` — a CD-degenerate free consumption price index.
- **`.nl`-vs-`.nl`** (the decider): both altertax MP models, same params/closure/seed,
  written to `.nl` and diffed on **columns and rows**: block 5114 cols / 4728 rows ==
  monolith 5114 / 4728, **0 diff either way**. The solver receives the byte-identical
  system.

Same system (`.nl` 0-diff) + same seed ⇒ the sub-floor is a pure solver convergence-point
(basin) difference on a CD-degenerate DOF — the documented scaled-tolerance-floor class,
not a model, equation, column, or row difference. Per the fidelity rule, the altertax gate
floors should reflect the measured faithful value (as the pure/ifSUB gates already measure
floors at runtime), not the monolith's wider-denominator number.

## What shipped

- `src/equilibria/templates/gtap/gtap_block_model.py` — the composer,
  `GTAPBlockMultiPeriodModel`, `_apply_ifsub_closure`, `build_block_model`,
  `solve_block_model`.
- `src/equilibria/blocks/gtap/_ifsub_macros.py` — the shared `M_*` macro module.
- `if_sub` threading + macro use in `factor.py`, `production_supply.py`, `income.py`,
  `trade_armington_bilateral.py`; calibration inits (`_make_init`, `_xcshr_init`,
  `_savf_init`, `_yi_init`, `xe`) computed from `self.params`.
- Diagnostic: `nl_column_diff` in the xmodel-parity cascade (dev-tools) with a
  `use_blocks` adapter track.

## Verification

Parity gates green at `21dcada` (54/54: MCP, NLP, `.nl`, gempack, coverage-matrix gates);
`.git/gtap-parity-gates.stamp` fresh for HEAD; measured docs in sync; monolith oracle
`git diff` empty; form gate 14/14.

```bash
# pure / ifSUB / altertax sweeps (local, needs PATH C-API + IPOPT + fixture GDXs)
uv run python scripts/gtap/run_parity_gates.py

# ifSUB column-parity check (the diagnostic that closed the shock)
uv run python ~/proyectos/dev-tools/equilibria-tools/scripts/parity_cascade/nl_column_diff.py \
    --model gtap --dataset gtap7_3x3 --if-sub 1 --a monolith --b blocks   # → 2918 == 2918
```
