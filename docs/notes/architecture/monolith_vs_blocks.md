# GTAP: monolith and blocks

Newcomers to the repo tend to assume that `templates/gtap/gtap_model_equations.py` (the
"monolith") and `blocks/gtap/` (the "blocks") are two alternative implementations of the same
model, and that the first is legacy waiting to be deleted.

**They are not.** This document exists to prevent that mistake, which has already cost time once.

## Who depends on whom

The modules of `blocks/gtap/*.py` import only `gtap_parameters`. The mentions of
`gtap_model_equations` in their docstrings are provenance comments
(`VERBATIM from ... líneas X-Y`): they say where each equation was copied from, they do not create a dependency.

That led to the conclusion —wrong— that the monolith is no longer used. Since 2026-09-16 the
composer **no longer imports it**: benchmark scaling is a separate module and the multiperiod
reflection asks for its SP model per method. What remains is of another kind:

| Place | What it does |
|---|---|
| `templates/gtap/__init__.py:58` | Any `import equilibria.templates.gtap` loads the monolith |
| 1 of the 5 gates | The monolith is the oracle against GAMS (only `nl`) |

Deleting `gtap_model_equations.py` causes a *collection error* in ~34 test files — **including
the 9 that exercise only blocks**, because the import fails before reaching them.

## Equation coverage

101 `Constraint(...)` families in the monolith against 100 `eq_*` in blocks.

Of the 7 that existed only in the monolith, 6 were declared and deactivated on the next line
—they never entered the `.nl`— and have been removed. `eq_pmuv` is left as the only real gap, and it only
bites when `closure.rmuv` and `imuv` are both non-empty, which happens in no dataset of the
gate. It is documented in `blocks/gtap/__init__.py:47-53`.

The 6 exclusive to blocks (`eq_mfr_{bs,sb,ss}`, `eq_mfw_{bs,sb,ss}` in `closure.py:218-341`) are not
new functionality: they are the decomposition into constraints of what the monolith inlines as
Expressions inside `eq_pfact`/`eq_pwfact`.

## The blocks are a functional superset

The asymmetry runs the opposite way to what intuition suggests:

| Mode | Monolith | Blocks |
|---|---|---|
| `capFix`, `capSFix` | yes | yes |
| `capFlex` | **no** (falls into a generic `else` at `:7276`) | yes (`demand_utility.py:584`) |
| `capFixDp` | **no** | yes (`demand_utility.py:116,166`) |
| `base_calibrated=True` | **does not exist** | yes (`gtap_block_model.py:372,440-446`) |
| basis `gempack` | partial | yes (`gtap_contract.py:309-321`) |

`gtap_contract.py:357` declares 5 `savf_flag`; the monolith implements 2.

## Why the monolith is kept

It is the fidelity oracle against **GAMS** in 1 of the 5 gates of
`scripts/gtap/run_parity_gates.py` — only `nl`. (There were 3: `nlp` and `mcp` moved to blocks on
2026-09-17.)

| Gate | Path | Reference |
|---|---|---|
| `test_gtap7_mcp_parity.py` | **blocks** (since 2026-09-17) | GAMS/PATH |
| `test_gtap7_nlp_parity.py` | **blocks** (since 2026-09-17) | GAMS/IPOPT |
| `test_gtap7_nl_parity.py` | monolith (via `nl_compare`) | GAMS `.nl` |
| `test_gtap7_gempack_parity.py` | **blocks** | GEMPACK SL4 |
| `tests/templates/gtap_logvalue/` | blocks | Julia/port |

The blocks path is the oracle against **GEMPACK**, and only it can be: that gate needs
`capFlex` + `base_calibrated=True`, which the monolith does not implement.

## Debt: how it would be retired

**First cut — done (2026-09-16).** Benchmark scaling now lives in
`templates/gtap/gtap_benchmark_scaling.py`: four free functions that take a
`ScalingContext` instead of `self`. The monolith keeps them as one-line delegators, so
its behavior does not change and it is still the oracle. The composer no longer builds a
`GTAPModelEquations` to borrow private methods from it.

Check of that cut: the post-scaling levels of both models on `gtap7_3x3` are
bit-for-bit identical —3142 cells, 0 differences— against the previous commit.

**Second cut — done (2026-09-16).** The multiperiod reflection asks for its single-period
model from `GTAPMultiPeriodModel._build_sp()`, which `GTAPBlockMultiPeriodModel` overrides. Before,
it monkey-patched `GTAPModelEquations.build_model` inside a `try/finally`. With that,
`gtap_block_model.py` **stopped importing the monolith**.

Check: the blocks multiperiod model —7392 entries across Var levels and
Constraint bodies— is identical against the previous commit.

(`_block_sp()` is kept as the method the subclasses override: `GTAPLogLevelsMultiPeriodModel`
does. Renaming it would have made the log-levels model fall
silently back to the normal blocks one.)

**Third cut — done (2026-09-17).** The `nlp` (14/14) and `mcp` (18/18) gates build
blocks. The monolith is marked as a **manual reference** in its docstring; the only gate
that measures it is `nl`.

The `nl` gate still reads it, deliberately: it compares the **emitted** `.nl` against GAMS's
—structure, not solve— and has no equivalent in blocks.

### Why the migration was possible: the gap was in the denominator

The obstacle recorded here was that blocks scored below the monolith on `altertax`
(−0.17pp on 3x4, −0.35pp on 10x7). **That gap did not measure model quality.**

Measured by comparing concrete cells instead of percentages: on `gtap7_10x7` **the same 110
cells fail in both models, 0 exclusive to either**. The failure count matches in all five
cases with a gap (110, 107, 6, 5, 1). The only thing that differs is the denominator:

```
110/15314 = 0.72% → 99.28%   (monolito)
110/10349 = 1.06% → 98.94%   (bloques)
```

The 4965 cells of difference are 28 exogenous shock variables (technical `a*`, efficiency `lambda*`,
`*txshft`/`etax`/`mtax`/`kappaf`). They are not solved: they hold whatever the closure fixes them to, and
they match GAMS in the 4610 measurable ones **without a single exception**. They are automatic hits that
inflated the monolith's denominator.

Blocks passes the current floors without recalibrating them **in the 14 rows of the NLP gate**: minimum margin
+0.497pp (3x4 altertax ifsub0) against +0.669pp for the monolith. That margin came from the
`rsav` fix (`7eca586`); before it, that row stood at 98.24, below its floor.

**That measurement did NOT cover the MCP gate**, which includes `gtap7_15x10` — a dataset absent from the
NLP matrix. There blocks failed, for the reason in the next section.

## MCP with blocks: RESOLVED (2026-09-17)

For a while `gtap7_15x10-pure-ifsub1` gave **87.00%** with blocks against a floor of
99.0, with `pft[USA,Land]` stuck at its floor `1e-3`, `pf[USA,Land,*]` at 0.00104 and
`xf[USA,Land,*]` between 4x and 12x above GAMS. It converged (`code=1`): it landed on ANOTHER point.

**The cause was 33 DEAD rows and 33 DEAD columns.**

`build_equations_fisher` (in the base class, shared by both models) replaces
`eq_pfact`/`eq_pwfact` with **cross-period** versions over `mq_factr_*`/`mq_factw_*`, and
deletes the intra-period ones. But blocks also declares its own auxiliary aggregates
`eq_mfr_*`/`eq_mfw_*` (`closure.py:216-341`), with which it splits the wide sum of **its**
intra-period `eq_pfact`. Once that `eq_pfact` was deleted, those rows were left with no
consumer — and nobody deleted them.

Measured: **0 consumers** of `mfr_*`/`mfw_*`, against `eq_pfact` (60) + `eq_pwfact` (6)
for `mq_fact*`.

And they are not a new aggregate: they are **exact duplicates** of the `mq_factr_*`/`mq_factw_*` that
the cross-period version declares. Same sum, same width (151 vars), same period. Measured on
`gtap7_15x10` shock:

```
mq_factr_ss[USA,shock] = 16.014072   residual 0        ← el agregado vivo, bien sembrado
mfr_ss[USA,shock]      = 15.605181   residual 0.409    ← el duplicado, con el seed viejo
```

Blocks emitted **66** aggregate rows where the monolith emits **33**.

### Why it breaks: the sacrificed equation

A redundant definition **cannot move the equilibrium by itself** — it defines a variable
that nobody consumes. What moves the equilibrium is what *squaring* does to compensate for it.

The 33 extra rows leave the system **overdetermined**, and
`deactivate_zero_unique_var_eqs` squares it **by deactivating a real equation**. Measured:

```
[nonlinear-full] DEACTIVATED-ROW: eq_xseq[USA,VegFruit,check]
[nonlinear-full] DEACTIVATED-ROW: eq_xseq[USA,VegFruit,shock]
```

`eq_xseq` is the physical supply balance `xs == xds + xet`. `_closure_patches.py:437` itself
warns that dropping it *"breaks the physical balance and lands a spurious root"* — which is
exactly what happened: `pf`/`pft[USA,*]` at the floor `1e-3`.

A check that rules out the alternative explanation: seed the aggregates
**consistently** (residual 0 in the 33 rows) and solve with the rows in place → `pft` stays at
`0.001`. It was not the seed value; it was the lost equation.

**The fix** (`gtap_model_multiperiod.build_equations_fisher`): also delete the six
auxiliary families **and their Vars**, right where the code already deleted
`eq_pabs`/`eq_pfact`/`eq_pwfact`. The Vars are **deleted**, not fixed: they are defined sums, and
freezing them at a stale value stalls the shock at `code=0` (that is the reason
`refresh_fisher_aggregates` exists).

Result: structure byte-identical to the monolith —74754 rows, 85857 columns, `solo-A=0` and
`solo-B=0` in rows and columns— and **MCP gate 18/18 with blocks**, NLP 14/14.

### Why it took so long to find

Five comparisons said "identical" and none of them was the real system: the `.nl` ROW+COL diff
compares **names** (and the 99 extra rows/columns seemed to balance out), the pairing
paired with itself, and the seed and the bounds matched. The only signal was counting **nonzeros of the
Jacobian**, not rows or columns. Earlier dead ends, all measured: forcing the pairing
of the Fisher rows (87.00% bit-identical) and making the `forced_pairs` immutable
(87.00% → 53.76%, actively harmful — that Hopcroft-Karp can re-pair is *load-bearing*).

**Probe trap.** A single-period probe gives "33/33 Fisher rows mis-paired, all
stealing `xf[*,Land,*]`", which fits the symptom perfectly (`pf[USA,Land]`) — and it is an
**artifact**: truncating `fv[:n]` to square it cuts exactly the Fisher columns at the end, and the
gate does not use single-period. Enumerating the raw MP does not work either: 74853 rows × 85956 vars are
the three periods uncrossed (74853 = 3×24951), not the system the solver factorizes
(24851). The sign that you are looking at the wrong system is that the row count is an
exact multiple of the number of periods.

### Ten hypotheses ruled out (so as not to repeat them)

Different seeds/floors, missing rows, a different `mqfactr_bb`, `pfa` wrongly fixed, different
`eq_xweq` expressions, nested `sqrt`, basin/start (seeding blocks at the monolith's solution:
it drifts away the same), the aggregates' bounds (`Reals`→`NonNegativeReals`: bit-identical), fixing the aggregates
(bit-identical), and the 33 rows as the direct cause (deactivating them: 3738 the same).

**The method trap that cost the session:** measuring by counting `component_data_objects(Constraint,
active=True)` gives the THREE periods unsquared, not the system the solver factorizes. The sign that
gives it away is in `project_btf_probe_y_condensacion_cuadratura_2026_09_02` (dev-tools): *"count
rows and check whether it is a multiple of the number of periods"*. Mine were 74754 and 74853 — exact
multiples of 3. The real square system is read from the `structural_matching` log: 24818 vs 24851.

**A method trap, for the second time in this repo:** a gap between two percentages can be in the
numerator or in the denominator, and the percentage alone does not tell them apart. It had already happened with `qga`
(see memory `project_gtap_qga_floor_levels_vs_linearization`). Comparing the failing cells
must be the first step, not the fourth.

### What is lost, and why it was accepted

With the gates on blocks, the monolith is no longer measured against anything. That redundancy —two
implementations against two oracles— is what made the Fisher snapshot bug visible. The
alternative evaluated was running both models in the gates, at twice the time. Blocks
alone was chosen: the monolith stays available as a reference, but it must not generate noise or be used as if
it were live code.

See also: `docs/roadmap.md` (technical debt record) and
`docs/archive/findings/repo_cleanup_spec_2026-09-16.md` (the measurement that gave rise to this document).

## Correction (2026-09-21): the Fisher fix did NOT lower the altertax match

The message of `2602db2` and PR #67 claim that `faa911b` lowers `gtap7_10x7 altertax
shock` from **99.3% to 98.97%**, and argue why to keep it anyway (fidelity over
match%, GEMPACK byte-identical). **The drop does not exist: there was nothing to justify.**

Measured with the same probe on two trees that differ only in `faa911b` — the
pre-Fisher `cbf060f` got the GDX reader fix (`8a501df`) backported, so that
Fisher was the only variable:

```
post-Fisher (main 2c53711)   10349 celdas, 107 fallos, 98.966%
pre-Fisher  (cbf060f + gdx)  10349 celdas, 107 fallos, 98.966%

celdas idénticas 107/107 · solo-post 0 · solo-pre 0
peor dif_rel del valor Python entre árboles: 6.1e-10
```

The `99.3%` it was compared against **was not a live measurement**: it is the
`gap_note` field of `scripts/gtap/coverage_matrix.py`, text written in `8dc561b`
(2026-06-30) and measured against the **monolith**. And `faa911b` is exactly the commit where the
MCP gate starts building `GTAPBlockMultiPeriodModel` (visible in its test diff).

In other words: **a June note about the monolith was subtracted from a September measurement of
blocks**. It is the same denominator mistake that the section "Why the migration was possible" had already
documented for the NLP gate (110 cells, 15314 vs 10349) — except that there it was
measured and here it was not. The real floor of that row is **98.0**, and 98.966 clears it with margin.

### The "92.5% directional bias" is a counting artifact

Of the 107 cells, 99 (92.5%) sit above GAMS, which was read as *a
missing term* (the documented `qxs` residual is symmetric, ~49%). **They are not 107
errors: they are 10**, one per `(importing region, sector)` pair, repeated once per
buying agent and per bilateral origin.

The standard deviation within each pair is **0.03–0.23** against means of +1.1 to
+6.4: it is one number repeated, not N independent errors.

```
JPN,c_Rice   agentes(xm) +5.99 +6.22 +6.51 +6.55 +6.57
             total (xmt)  +6.22
             bilateral por DESTINO +6.21
```

`xw` grouped by **destination** is tight; by **origin**, scattered → the error lives
on the **importer** side. And `pa` (price) moves opposite to the quantity, which is
what a demand shift produces, not noise between engines.

The 10 pairs: `JPN{Rice,Livestock,FoodProc,Crops}`, `IND{Crops,FoodProc}`,
`CHN{Livestock,FoodProc}`, `SSA{Textiles}`, `USA{Textiles}`.

**What is really still open** is a residual of agricultural imports ~1–6%
above GAMS in those 10 pairs, **predating** all this work, of the same family as the
`qxs` residual of 2026-08-21 — where the term-by-term culprit turned out to be
`pim`, an import price index. That is the thread, and it is old.

### Lesson

A number in a note **is not a measurement**. Before calling a drop a regression,
check that the old number and the new one measure **the same object**: here the
engine (monolith → blocks) and the denominator changed, three months apart.

And before explaining a directional bias, group by the **structural** unit (the pair
`r,i`) instead of counting cells: the multiplicity of agents fabricates biases that do not
exist.
