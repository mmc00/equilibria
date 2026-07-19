# PEP-1-1 Parity Coverage Matrix

<!-- GENERATED FROM scripts/pep/pep_coverage_matrix.py — do not edit by hand.
     Regenerate: uv run python scripts/pep/gen_pep_coverage_doc.py -->

Coverage for the `pep2` dataset — the PEP-1-1 v2.1 CGE model ported to Pyomo (`equilibria.templates.pep_pyomo`). Every row compares the Pyomo solve against a GAMS reference solved by the **same engine** (IPOPT vs IPOPT, or PATH vs PATH), so the solver's equality tolerance cancels and the cell-by-cell match reflects **model fidelity**, not solver noise.

```{raw} html
<div class="mx-legend"><span class="mx-li"><b>match</b> is the measured snapshot; the named pytest gate re-measures and enforces it</span><span class="mx-li"><span class="mx-chip mx-neutral">local</span> every row needs PATH/IPOPT + GAMS refs — no solver-free CI gate (the PEP reference is a solved GDX, not a .nl dump)</span></div>
```

## NLP vs NLP

The Pyomo model solved as an NLP (IPOPT on the raw model, `nlp_scaling_method=none`, faithful to GAMS's raw solve) against the GAMS CNS reference `Results.gdx`. The benchmark BASE reproduces the SAM, so the seeded point is the calibration answer. Run: `phase1_nlp.py --model pep --dataset pep2 --period BASE`.

```{raw} html
<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Scenario · form</th><th>Cells</th><th>Match</th><th>Gate</th><th>Reference</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">base</span><span class="mx-sub">nlp</span></td><td>317</td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span><span class="mx-chip mx-neutral">local</span></div></td><td><span class="mx-ref">phase1_nlp.py --form nlp / test_faithful_at_benchmark</span></td><td><span class="mx-ref">Results.gdx (GAMS CNS, val*)</span></td></tr></tbody></table></div></div>
```

## MCP vs MCP

The Pyomo model solved as a complementarity problem via PATH against the **GAMS-native** MCP reference (`PEP-1-1_v2_1_mcp_solve.gms`: `MODEL /ALL/` + `SOLVE USING MCP`, so GAMS infers the equation↔variable pairing). The `sim1` row is the reference counterfactual — a 25% export-tax cut (`ttix.fx=ttixO*0.75`) — applied faithfully in Python by scaling the `ttixO` benchmark before build. Run: `phase1_nlp.py --model pep --dataset pep2 --form mcp`.

```{raw} html
<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Scenario · form</th><th>Cells</th><th>Match</th><th>Gate</th><th>Reference</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">base</span><span class="mx-sub">mcp</span></td><td>367</td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span><span class="mx-chip mx-neutral">local</span></div></td><td><span class="mx-ref">test_mcp_matches_gams_native_mcp</span></td><td><span class="mx-ref">Results_mcp.gdx (GAMS /ALL/ MCP)</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">sim1</span><span class="mx-sub">mcp</span></td><td>314</td><td><div class="mx-cell"><span class="mx-num mx-good">100% (GDP_BP 46707→46748.2)</span><span class="mx-chip mx-neutral">local</span></div></td><td><span class="mx-ref">test_mcp_sim1_shock_matches_gams</span></td><td><span class="mx-ref">Results_mcp_sim1.gdx (GAMS MCP, ttix·0.75)</span></td></tr></tbody></table></div></div>
```

## NLP↔MCP mirror

The two Pyomo forms, solved from the same feasible benchmark seed, land on the identical point (LEON, the form-defining Walras slack, is excluded). The one historical gap — `PD['othind']`, filled from its `*O` benchmark (1.132) rather than a blind 1.0 — closed the mirror to a clean 100%.

```{raw} html
<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Scenario · form</th><th>Cells</th><th>Match</th><th>Gate</th><th>Reference</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">base</span><span class="mx-sub">both</span></td><td>466</td><td><div class="mx-cell"><span class="mx-num mx-good">100%</span><span class="mx-chip mx-neutral">local</span></div></td><td><span class="mx-ref">test_nlp_mcp_mirror</span></td><td><span class="mx-ref">self (NLP vs MCP, LEON excl)</span></td></tr></tbody></table></div></div>
```

## objdef variant

The `objdef` variant adds a dummy objective (`OBJDEF: OBJ==0`, minimize OBJ) — the `SOLVE NLP MINIMIZING OBJ` lineage. A constant objective cannot move the equilibrium, so objdef-NLP lands on the exact base-NLP point; in the MCP form OBJ is not declared, keeping the system square.

```{raw} html
<div class="mx-card"><div class="mx-scroll"><table class="mx-table"><thead><tr><th class="mx-lbl">Scenario · form</th><th>Cells</th><th>Match</th><th>Gate</th><th>Reference</th></tr></thead><tbody><tr><td class="mx-lbl"><span class="mx-ds">base</span><span class="mx-sub">nlp</span></td><td>467</td><td><div class="mx-cell"><span class="mx-num mx-good">100% (== base-NLP)</span><span class="mx-chip mx-neutral">local</span></div></td><td><span class="mx-ref">test_objdef_variant_equals_base_nlp</span></td><td><span class="mx-ref">self (objdef vs base)</span></td></tr><tr><td class="mx-lbl"><span class="mx-ds">base</span><span class="mx-sub">mcp</span></td><td>358</td><td><div class="mx-cell"><span class="mx-num mx-warn">square, code=1</span><span class="mx-chip mx-neutral">local</span></div></td><td><span class="mx-ref">test_objdef_mcp_is_square_and_solves</span></td><td><span class="mx-ref">self (objdef+MCP squareness)</span></td></tr></tbody></table></div></div>
```
