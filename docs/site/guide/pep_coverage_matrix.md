# PEP-1-1 Parity Coverage Matrix

<!-- GENERATED FROM scripts/pep/pep_coverage_matrix.py — do not edit by hand.
     Regenerate: uv run python scripts/pep/gen_pep_coverage_doc.py -->

Coverage for the `pep2` dataset — the PEP-1-1 v2.1 CGE model ported to Pyomo (`equilibria.templates.pep_pyomo`). Every row compares the Pyomo solve against a GAMS reference solved by the **same engine** (IPOPT vs IPOPT, or PATH vs PATH), so the solver's equality tolerance cancels and the cell-by-cell match reflects **model fidelity**, not solver noise. `match` is the measured snapshot; `gate` is the pytest that re-measures and enforces it. `ci_status` is `local` for every row (each needs PATH+GAMS) — PEP has no solver-free CI gate because its reference is a solved GDX, not a `.nl` coefficient dump.

## NLP form vs GAMS CNS

The Pyomo model solved as an NLP (IPOPT on the raw model, `nlp_scaling_method=none`, faithful to GAMS's raw solve) against the GAMS CNS reference `Results.gdx`. The benchmark BASE reproduces the SAM, so the seeded point is the calibration answer (the solver early-exits there, mirroring the original cyipopt solver and GAMS CNS). Run the gate with `phase1_nlp.py --model pep --dataset pep2 --period BASE`.

| scenario | form | cells | match | gate | ci_status | ref |
|---|---|---|---|---|---|---|
| base | nlp | 317 | 100% | `phase1_nlp.py --form nlp / test_faithful_at_benchmark` | local | Results.gdx (GAMS CNS, val*) |

## MCP form vs GAMS-native MCP

The Pyomo model solved as a complementarity problem via PATH against the **GAMS-native** MCP reference (`PEP-1-1_v2_1_mcp_solve.gms`: `MODEL /ALL/` + `SOLVE USING MCP`, so GAMS infers the equation↔variable pairing). This is the first PEP-MCP. The `sim1` row is the reference counterfactual — a 25% export-tax cut (`ttix.fx=ttixO*0.75`) — applied faithfully in Python by scaling the `ttixO` benchmark before build; both engines move GDP_BP 46707→46748.2. Run the base gate with `phase1_nlp.py --model pep --dataset pep2 --form mcp`.

| scenario | form | cells | match | gate | ci_status | ref |
|---|---|---|---|---|---|---|
| base | mcp | 367 | 100% | `test_mcp_matches_gams_native_mcp` | local | Results_mcp.gdx (GAMS /ALL/ MCP) |
| sim1 | mcp | 314 | 100% (GDP_BP 46707→46748.2) | `test_mcp_sim1_shock_matches_gams` | local | Results_mcp_sim1.gdx (GAMS MCP, ttix·0.75) |

## NLP↔MCP mirror

The two Pyomo forms, solved from the same feasible benchmark seed, land on the identical point (LEON, the form-defining Walras slack, is excluded). Closing the one historical gap — `PD['othind']`, the price of a zero-domestic-demand good, now filled from its `*O` benchmark (1.132) in both forms rather than a blind 1.0 — lifted the mirror to a clean 100%.

| scenario | form | cells | match | gate | ci_status | ref |
|---|---|---|---|---|---|---|
| base | both | 466 | 100% | `test_nlp_mcp_mirror` | local | self (NLP vs MCP, LEON excl) |

## objdef variant

The `objdef` variant adds a dummy objective (`OBJDEF: OBJ==0`, minimize OBJ) — the `SOLVE NLP MINIMIZING OBJ` lineage. A constant objective cannot move the equilibrium, so objdef-NLP lands on the exact base-NLP point. In the MCP form the OBJ variable is NOT declared (its OBJDEF equation is NLP-only; declaring it would leave an unpaired free variable and break squareness), so objdef+MCP stays square and solves.

| scenario | form | cells | match | gate | ci_status | ref |
|---|---|---|---|---|---|---|
| base | nlp | 467 | 100% (== base-NLP) | `test_objdef_variant_equals_base_nlp` | local | self (objdef vs base) |
| base | mcp | 358 | square, code=1 | `test_objdef_mcp_is_square_and_solves` | local | self (objdef+MCP squareness) |
