# Architecture overview

`equilibria` is a Python framework for Computable General Equilibrium (CGE)
modeling whose defining constraint is **parity with reference GAMS models**:
every template is validated cell-by-cell against the original GAMS
implementation before it is considered done.

## The pipeline

```{raw} html
<svg viewBox="0 0 920 150" role="img" aria-label="equilibria pipeline"
     style="max-width:100%;height:auto;font-family:sans-serif;font-size:13px">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b96a3"/>
    </marker>
  </defs>
  <g fill="none" stroke="#8b96a3" stroke-width="1.5" marker-end="url(#arr)">
    <line x1="168" y1="75" x2="192" y2="75"/>
    <line x1="352" y1="75" x2="376" y2="75"/>
    <line x1="536" y1="75" x2="560" y2="75"/>
    <line x1="720" y1="75" x2="744" y2="75"/>
  </g>
  <g text-anchor="middle">
    <rect x="8" y="45" width="160" height="60" rx="10" fill="#e8f0f8" stroke="#2f6fb0"/>
    <text x="88" y="70" fill="#1a2129" font-weight="600">Data I/O</text>
    <text x="88" y="90" fill="#5a6673" font-size="11">SAM · HAR · GDX · MIP</text>
    <rect x="192" y="45" width="160" height="60" rx="10" fill="#e8f0f8" stroke="#2f6fb0"/>
    <text x="272" y="70" fill="#1a2129" font-weight="600">Calibration</text>
    <text x="272" y="90" fill="#5a6673" font-size="11">benchmark → parameters</text>
    <rect x="376" y="45" width="160" height="60" rx="10" fill="#e8f0f8" stroke="#2f6fb0"/>
    <text x="456" y="70" fill="#1a2129" font-weight="600">Templates</text>
    <text x="456" y="90" fill="#5a6673" font-size="11">gtap · pep_pyomo · simple_open</text>
    <rect x="560" y="45" width="160" height="60" rx="10" fill="#e8f0f8" stroke="#2f6fb0"/>
    <text x="640" y="70" fill="#1a2129" font-weight="600">Solvers</text>
    <text x="640" y="90" fill="#5a6673" font-size="11">PATH (MCP) · IPOPT (NLP)</text>
    <rect x="744" y="45" width="168" height="60" rx="10" fill="#e6f3ec" stroke="#2f8f5b"/>
    <text x="828" y="70" fill="#1a2129" font-weight="600">Validation</text>
    <text x="828" y="90" fill="#5a6673" font-size="11">parity gates vs GAMS</text>
  </g>
</svg>
```

1. **Data I/O** (`equilibria.babel`, `equilibria.sam_tools`) reads and
   balances the input data: SAM matrices, GEMPACK HAR files (native
   pure-Python reader and writer), GAMS GDX containers, and raw MIP/IO
   tables ({doc}`MIP → SAM <mip_to_sam>`, {doc}`HAR I/O <har_io>`).
2. **Calibration** (`equilibria.calibration`, `equilibria.core`, plus
   per-template calibration modules) fits the model's parameters so the
   benchmark equilibrium reproduces the SAM exactly.
3. **Templates** (`equilibria.templates`) are complete model
   implementations — sets, parameters, equations, closure — assembled on
   the core (`equilibria.core`, `equilibria.backends` for Pyomo) and the
   generic building blocks in `equilibria.blocks` (production, trade,
   demand, institutions, equilibrium).
4. **Solvers**: the same model solves as an **MCP** (complementarity,
   PATH via the C API — {doc}`path_capi`) or as an **NLP** (IPOPT),
   mirroring GAMS's `ifMCP` switch. Solver support utilities live in
   `equilibria.solver`; the PATH/IPOPT drivers ship with each template.
5. **Validation** compares every solved cell against a reference GAMS
   solution ({doc}`benchmarks`, {doc}`gtap7_coverage_matrix`,
   {doc}`pep_coverage_matrix`).

## The parity philosophy

- **GAMS is the source of truth.** A divergent cell is a bug (in the model,
  the calibration, or the reference itself) until proven otherwise — cells
  are never excluded to inflate a match number.
- **Same-engine comparisons.** Parity is measured NLP-vs-NLP (IPOPT both
  sides) and MCP-vs-MCP (PATH both sides), so the solver's equality
  tolerance cancels and the match% reflects *model fidelity*, not solver
  noise.
- **Gates, not snapshots.** Coverage lives in one declarative matrix per
  model (`scripts/gtap/coverage_matrix.py`,
  `scripts/pep/pep_coverage_matrix.py`). Pytest gates re-measure the match
  and assert conservative floors; CI keeps the rendered docs in sync with
  the matrix source.
- **A solver-free structural canary.** The `.nl` coefficient gate diffs
  Python-vs-GAMS Jacobian coefficients on every push — no solver needed, so
  it runs in CI and catches equation/parameter regressions in seconds.

## Package map

| Package | Responsibility |
|---|---|
| `equilibria.sam_tools` | SAM audit, balancing, MIP → SAM transforms |
| `equilibria.babel` | File formats: HAR reader/writer, GDX reader, SAM loaders |
| `equilibria.core` | Model assembly core: sets, parameters, variables, equations, calibration phases |
| `equilibria.model`, `equilibria.datasets` | Public model API and bundled datasets (`load_bundled`) |
| `equilibria.blocks` | Generic equation blocks: production, trade, demand, institutions, equilibrium |
| `equilibria.calibration` | Generic calibration utilities (CES, Leontief) |
| `equilibria.backends` | Pyomo backend (model construction) |
| `equilibria.solver` | Solver support: guards, Jacobian tooling, transforms |
| `equilibria.templates.gtap` | GTAP Standard 7 (multi-period, MCP + NLP, 6 datasets) |
| `equilibria.templates.pep_pyomo` | PEP-1-1 v2.1 ported to Pyomo (NLP + MCP) |
| `equilibria.templates` (simple_open) | Didactic open-economy model |
| `equilibria.simulations` | Scenario runtime, adapters, multi-model wrappers |
| `equilibria.contracts`, `equilibria.qa` | Validation contracts and SAM/QA checks |

See the templates overview (next page) for per-template status and
the {doc}`API reference <../api/index>` for the generated reference.
