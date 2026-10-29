# scripts

Lo que no viaja en el paquete: gates de paridad, generadores de documentación,
mediciones y diagnósticos. Se corren desde la raíz del repo con
`uv run python scripts/<carpeta>/<script>.py`.

## Por dónde empezar

| Para | Script |
|---|---|
| Correr los gates de paridad de GTAP (MCP, NLP, `.nl` contra GAMS; niveles contra GEMPACK) | `gtap/run_parity_gates.py` (`--quick`: sólo los tests) |
| Depurar una brecha de paridad contra GAMS | `parity/triage.py` (cascada: locate → isolate → trace) |
| Reproducir las tablas del paper GTAP (JGEA) | `paper/run_tables.py` |
| Comparar niveles de equilibria contra GEMPACK | `gtap/compare_gempack_levels.py` |
| Cronometrar las fases de un solve de GTAP | `gtap/phase_times.py` |
| Regenerar las matrices de cobertura del sitio | `gtap/gen_coverage_doc.py`, `gtap/gen_gempack_doc.py`, `pep/gen_pep_coverage_doc.py` |
| Specs y ADRs | `dev/spec_tools.py` |
| Este mapa | `dev/gen_scripts_map.py` |

## Las carpetas

- **`gtap/`** — gates, generadores de docs, mediciones y diagnósticos de GTAP.
  La columna *Modelo* del mapa dice si un script arma el modelo con bloques o
  con el monolito: el driver ya no resuelve el monolito, así que un script
  marcado *monolito* que resuelve no corre hasta migrarlo (`R-MONOLITH`).
- **`gtap_julia/`** — el oráculo Julia (GTAPv7.jl) y su comparación celda a celda.
- **`parity/`** — la cascada genérica de depuración de paridad (`triage`, `probe`)
  y comparaciones de PEP y simple_open contra GAMS.
- **`pep/`** — la matriz de cobertura de PEP y los diagnósticos del emparejamiento MCP.
- **`cli/`** — comandos del motor PEP viejo (el que corre `PepSimulator`), que se
  borra en la etapa E1 del roadmap de PEP.
- **`dev/`** — herramientas de desarrollo (specs, el candado de `ty`, este mapa)
  y diagnósticos viejos de GDX.
- **`qa/`** — chequeos de SAM y de calibración.
- **`sam_tools/`** — el pipeline IEEM → PEP paso a paso y los ajustes de la SAM de Costa Rica.
- **`har/`** — el oráculo del escritor HAR.
- **`paper/`** — las tablas y el tamaño del modelo del paper.
- **`windows/`** — las corridas del paper en Windows.

## Política

- Layout (raíz cerrada, nombres en minúsculas y en inglés): `python scripts/dev/check_layout.py`.
- Un script nuevo lleva un docstring cuya primera línea diga qué hace: es lo que
  muestra el mapa. Después de agregar o borrar scripts, regenera el mapa con
  `uv run python scripts/dev/gen_scripts_map.py`.

## Mapa

Generado por `scripts/dev/gen_scripts_map.py`: la primera línea del docstring de
cada script y quién lo nombra (CI, tests, docs u otro código). Un script con
"—" no lo nombra nadie: es una medición puntual o está abandonado.

<!-- MAPA:INICIO -->
### raíz (4)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `cal_dump.gms` | GTAP calibration dump (Python parity helper) | docs, código |
| `cal_dump_neos.gms` | GTAP calibration dump for NEOS (self-contained GTAP script conventions) | — |
| `dev/check_layout.py` | Closed root, lowercase English names, no loose scripts (ADR-0010). | CI, pre-commit |
| `setup_ipopt.sh` | Setup script to install cyipopt with system IPOPT | — |

### `cli/` (7)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `run_all_calibration.py` | Run complete PEP model calibration. | docs |
| `run_dynamic_sam_python.py` | Run Python PEP calibration equivalent to dynamic SAM GAMS template. | — |
| `run_pep_base_export_tax_parity.py` | Run PEP BASE and EXPORT_TAX scenarios and compare each one to GAMS. | tests, docs |
| `run_pep_core_scenarios_gate.py` | Run the core PEP scenario pack and optionally enforce GAMS parity by scenario. | tests, docs |
| `run_solver.py` | Run PEP model solver. | docs |
| `run_solver_cri_dynamic_sam.py` | Run PEP dynamic-SAM solver for CRI dataset (GDX or Excel). | — |
| `run_solver_excel.py` | Run PEP model solver using SAM loaded directly from Excel. | — |

### `dev/` (9)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `analyze_gams_baseline.py` | Execute PEP model with calibration from GAMS baseline. | — |
| `analyze_sam_storage.py` | Analyze SAM GDX storage differences between original and equilibria formats. | — |
| `compare_excel_values.py` | Direct Excel Comparison for SAM and VAL_PAR | — |
| `compare_gdx.py` | GDX Comparison Tool | docs, código |
| `compare_gdx_detailed.py` | Comprehensive GDX Value Comparison Tool | — |
| `debug_equations.py` | Debug script to find equation discrepancies. | — |
| `gen_ty_exclude.py` | Genera el bloque ``exclude:`` del hook ``ty`` en ``.pre-commit-config.yaml``. | docs |
| `spec_tools.py` | Parse specs and map repo code to architecture areas, plus the spec-flow CLI. | CI, tests, docs |
| `validate_gdx.py` | Validate equilibria GDX files against original cge_babel data. | — |

### `gtap/` (106)

| Script | Qué hace | Lo nombra | Modelo |
|---|---|---|---|
| `_closure_patches.py` | Compatibility shim — the real module now lives in the installed package. | tests, docs, código |  |
| `_cudss_linsolve.py` | cuDSS (NVIDIA GPU multifrontal direct solver) backend for the TR linear solve. | tests, código |  |
| `_diff_core.py` | Shared GAMS↔Python variable-by-variable diff helpers. | tests, código |  |
| `_nl_parser.py` | Minimal AMPL .nl file parser for GTAP model comparison. | tests, código |  |
| `_parity_datasets.py` | Shared, sets-agnostic dataset registry for the three parity tools. | tests, código | monolito |
| `_parity_json.py` | Shared JSON-output contract for the parity-debug cascade tools. | tests, docs, código |  |
| `bench.py` | FAST ITERATION BENCH — edit a convention → run → see the per-cell delta vs the | tests, código |  |
| `bench_build_once.py` | A/B wall-clock of the two multi-period equations-build paths on a dataset. | — | monolito |
| `bench_nlp_timing.py` | NLP wall-time benchmark: Python (IPOPT, EQUILIBRIA_GTAP_SOLVE_NLP=1) vs GAMS | CI | monolito |
| `bench_nus333_dual.py` | NUS333 dual-reference benchmark: parity vs GAMS local + GAMS NEOS, with wall-time. | CI, código | monolito |
| `bench_settle_cut.py` | A/B wall-clock of calibrate_base's settle: settle_only (cut at check) vs the | — |  |
| `bench_symbolic_reuse.py` | A/B: MUMPS symbolic-reuse ON vs OFF on 15x10 GMIN+MUMPS. Prints solve wall-clock | código |  |
| `blocks_diag.py` | Diagnostic tooling for the first B-block solve (F3). | tests, docs | monolito |
| `build_9x10_altertax_neos_bundle.py` | Build a self-contained NEOS bundle for 9x10 altertax (fully inlined data). | código |  |
| `build_altertax_neos_bundle.py` | Build a self-contained NEOS bundle for 9x10 altertax (fully inlined data). | código |  |
| `build_gtap7_altertax_neos_bundle.py` | Build a self-contained NEOS altertax bundle from a v7_consolidated.gdx. | código |  |
| `build_gtap7_pure_local_bundle.py` | Build a self-contained, LOCALLY-runnable PURE-gtap tariff-shock bundle. | docs, código |  |
| `build_v7_processed_from_raw.py` | Build a PROCESSED v7_consolidated.gdx from a RAW GEMPACK GtapView GDX. | código |  |
| `calibrate_and_align.py` | GTAP SAM Calibration and Parity Alignment | — |  |
| `gtap_calibrated.gms` | GTAP Model - Calibrated from SAM | código |  |
| `cascade_classify.py` | Four-way classification of a cascade-tool result. GOVERNING PRINCIPLE: no error | tests, código |  |
| `cascade_config.py` | Pure config resolution for the cascade orchestrator: period->scenario and the | tests, código |  |
| `cascade_kkt.py` | KKT/marginals cascade layer: read equation duals (.m) from the local reference | tests, código |  |
| `cascade_layers.py` | LAYER_SPECS: the single source of truth for cascade-tool invocation. Each | tests, código |  |
| `cascade_orchestrator.py` | Phase-2 cascade orchestrator. Runs the six JSON cascade tools across periods, | tests |  |
| `cascade_run.py` | Subprocess runner + per-period sweep for the cascade orchestrator. | tests, código |  |
| `check_parity_gates_stamp.py` | Fast staleness check for the mandatory local GTAP parity gates (stdlib-only). | tests, docs, código |  |
| `block_push_without_gates.py` | Claude Code PreToolUse hook: block `git push` / `gh pr create` without fresh gates. | tests, docs, código |  |
| `compare_9x10_rungtap.py` | 9x10 head-to-head: equilibria (Python MCP) vs RunGTAP under capFix closure. | docs, código | monolito |
| `compare_burfisher_gempack.py` | Burfisher 3e / nus333: equilibria contra GEMPACK en NIVELES, ejercicio por ejercicio. | tests, código |  |
| `compare_calibration.py` | Compare GTAP Python calibration objects against a GAMS calibration dump. | docs, código |  |
| `compare_gempack_levels.py` | equilibria contra GEMPACK en NIVELES con el mapeo GTAP (``maps/gtap7.yaml``). | tests, docs, código | bloques |
| `compare_nus333_baseline_vars.py` | Compare Python baseline (no shock) values vs GAMS NEOS baseline (t='base'). | — | monolito |
| `compare_nus333_rungtap.py` | NUS333 head-to-head: equilibria (Python) vs RunGTAP/GEMPACK (this machine). | docs, código | monolito |
| `compare_nus333_vs_neos.py` | NUS333 baseline + 10% import-tariff power shock — compare vs NEOS GAMS. | tests, docs, código | monolito |
| `test_model.gms` | Simple GTAP test model - Square CNS system | código |  |
| `convert_gams.py` | CONVERT — emit the GAMS reference model as canonical Pyomo (.py + dict), level-scaled, | — |  |
| `coverage_matrix.py` | GTAP7 parity coverage matrix — the single source of truth. | tests, docs, código |  |
| `demo_parity_both_models.py` | Demo: Run Both Python GTAP and Simulated GAMS - Compare Solutions | — | monolito |
| `demo_parity_simple.py` | Demo: Parity Comparison Between Python and GAMS | código |  |
| `diag_mp_3x3.py` | Task-6 decision gate: 3x3 multi-period model → does shock reach code=1? | código | monolito |
| `diag_mp_check_drift.py` | PURE DIAGNOSTIC (no fix): instrument the multi-period CHECK solve. | — | monolito |
| `diff_9x10_full.py` | Full variable-by-variable diff: Python GTAP 9x10 vs GAMS COMP.gdx. | CI, docs, código | monolito |
| `diff_9x10_vs_gams.py` | Compare Python 9x10 baseline + shocked solutions vs GAMS COMP.gdx. | — | monolito |
| `diff_altertax.py` | Cell-by-cell diff: Python altertax vs GAMS NEOS altertax out.gdx. | código | monolito |
| `diff_calibration.py` | Diff Python's CALIBRATION INPUTS against GAMS, per region and per tax stream. | tests, código |  |
| `diff_causal_propagation.py` | Tool #10 of the parity-debug cascade: CAUSAL-PROPAGATION (symptom-vs-cause detector). | código | monolito |
| `diff_equation_form.py` | Show a Python equation's expanded symbolic form next to its GAMS source line. | docs, código |  |
| `diff_holdfixed.py` | Tool #8 of the parity-debug cascade: HOLDFIXED / SEQUENCE diff. | tests, código |  |
| `diff_mcp_pairing.py` | Diff the MCP equation<->variable pairing: GAMS `model gtap /.../` vs Python. | tests, código |  |
| `diff_nus333_full.py` | Full variable-by-variable diff: Python GTAP NUS333 vs GAMS NEOS out.gdx. | CI, docs, código | monolito |
| `diff_tautology.py` | Tool #9 of the parity-debug cascade: TAUTOLOGY / UNANCHORED-VARIABLE detector. | tests, código | monolito |
| `drift_test.py` | Tool #7 of the parity-debug cascade: the DRIFT TEST (free-DOF / basin detector). | tests, código | monolito |
| `dump_py_solution_gdx.py` | Solve the Python multi-period model as NLP (in-process), then write ALL its | — |  |
| `example_basic.py` | Example: Basic GTAP Model Usage | — | monolito |
| `example_parity.py` | Example: GTAP Parity Testing | — | monolito |
| `export_sl4_levels.py` | Export pre- and post-simulation LEVELS from any GEMPACK SL4 solution to a plain HAR. | tests, código |  |
| `gtap_sync.gms` | GTAP Synchronized Model - Matches Python exactly | código |  |
| `final_parity_test.py` | Final GTAP Parity: Synchronized Calibration | — | monolito |
| `gtap_model.gms` | Full GTAP Model - Multi-region CGE | docs, código |  |
| `comp_shock.gms` | comp_shock.gms - comp_nus333.gms (base/check/shock, NUS333) with the | tests, código |  |
| `gams_welfare_fixture.py` | Bienestar de GAMS en los ejercicios de Burfisher: el fixture del test de bienestar. | tests |  |
| `gdx_full_diff.py` | Diff ALL symbols (variables AND params) between a Python-solution GDX and a | — |  |
| `gdx_to_gams_inline.py` | Convert a GDX file to inline GAMS set/parameter declarations. | código |  |
| `gempack_reference.py` | Read a RunGTAP updated.har (post-shock levels) and expose Var cells shaped | tests, docs, código |  |
| `gen_burfisher_gams.py` | Burfisher 3e / nus333: genera el oraculo GAMS de cada ejercicio. | tests, código |  |
| `gen_coverage_doc.py` | Generate docs/site/guide/gtap7_coverage_matrix.md from coverage_matrix.ROWS. | tests, docs, código |  |
| `gen_gempack_doc.py` | Generate docs/site/guide/gtap7_coverage_matrix_gempack.md from the gempack rows. | tests, docs, código |  |
| `gen_linearization_study.py` | Generate the against-GEMPACK linearization-study page from whatever grid | tests, docs, código | monolito |
| `gen_nlp_matrix_page.py` | Generate docs/site/guide/gtap7_nlp_matrix.html — the rich per-stage NLP-vs-NLP | docs, código |  |
| `homotopy_shock.py` | GAMS-faithful homotopy driver for the altertax +10% import-tariff shock. | código | monolito |
| `inheritance_table.py` | EXHAUSTIVE inter-period INHERITANCE sweep: every variable GAMS seeds from the prior | — |  |
| `inspect_updated_har.py` | Inspect a RunGTAP updated.har: print every header's name, rank, shape, and | — |  |
| `make_levels_tab.py` | Write gtapv7_levels.tab: gtapv7.tab with ``orig_level`` on six more variables. | tests, código |  |
| `measure_gate_tols.py` | Fresh status sweep: run the EXACT altertax multi-period gate solve path | — | monolito |
| `measure_gempack_blocks.py` | F3.5 against-GEMPACK measurement: the specific-factor (Land) price response. | tests, docs | bloques |
| `measure_gtap_pure_tols.py` | Fresh status sweep for the PURE-GTAP (non-altertax) multi-period SOLVE gate. | código | monolito |
| `measure_nlp_vs_nlp.py` | NLP-vs-NLP parity: solve the Python multi-period model as an NLP (maximize | tests, código | monolito |
| `measure_nlp_vs_nlp_altertax.py` | NLP-vs-NLP parity: solve the Python multi-period model as an NLP (maximize | — | monolito |
| `mk_exp_directos.py` | Traduce la sintaxis de shock de la GUI de RunGTAP a valores directos. | tests |  |
| `nl_compare.py` | GTAP .nl comparison tool — Layer 6 diagnostic. | tests, docs, código | monolito |
| `pairing_table.py` | EXHAUSTIVE pairing/free-row table: every equation in GAMS `model gtap /.../` vs | — |  |
| `gtap_model.gms` | GTAP Model for Parity Testing | docs, código |  |
| `perf_budget_20x41.py` | Benchmark del 20x41 contra la linea base: correr antes de mergear un PR de rendimiento. | tests, código |  |
| `phase_times.py` | Reparto del tiempo de una corrida GTAP por fase, con cronometros y no con cProfile. | CI, docs, código |  |
| `profile_20x41_full.py` | Instrument the FULL 20x41 end-to-end (build + calibrate + solve) to find where | — |  |
| `reconstruct_9x10_eflg.py` | Reconstruct the 9x10 reference default.prm's missing EFLG (ENDOWFLAG) header. | código |  |
| `regen_altertax_reference.py` | Regenerate a subsidy-aware altertax reference GDX by building the NEOS bundle, | código |  |
| `run_burfisher.py` | Burfisher 3e / nus333: corre cada ejercicio con equilibria y lo compara contra GAMS. | tests, docs, código | bloques |
| `run_full_gtap_comparison.py` | Full GTAP Model Execution and Parity Check | — |  |
| `run_gempack_matrix.py` | Produce the RunGTAP (GEMPACK/GTAPv7) post-shock `updated.har` for every | tests, docs, código |  |
| `run_gtap.py` | GTAP CLI - Command-line interface for GTAP CGE model | tests, docs, código | monolito |
| `run_gtap_parity.py` | GTAP Parity Test Runner | código |  |
| `run_parity_both_models.py` | Execute Both GTAP Models and Compare Results | — | monolito |
| `run_parity_gates.py` | Mandatory LOCAL integration gates: NLP-vs-NLP + MCP-vs-MCP full sweeps. | tests, docs, código |  |
| `run_real_cgebox_comparison.py` | Execute Real CGEBox GTAP Model and Compare with Python | — | monolito |
| `seed_and_solve.py` | Cascade tool 11 — SEED-AND-SOLVE: the equilibrium-selection vs differing-equation | tests, código |  |
| `show_parity_variables.py` | GTAP Parity Variables Summary | — |  |
| `submit_9x10_altertax_neos.py` | Submit the 9x10 altertax self-contained bundle to NEOS. | — |  |
| `submit_altertax_neos.py` | Submit an altertax self-contained bundle to NEOS. | — |  |
| `test_full_model.py` | Test Complete GTAP Python Model | — | monolito |
| `validate_gams_parity.py` | GAMS parity validation for GTAP Standard 7 (9x10). | docs, código | monolito |
| `validate_nus333_parity.py` | Smoke test: load NUS333 via HAR and validate parameters are populated. | — |  |
| `validate_reference.py` | Validate that a GAMS reference GDX satisfies its OWN equations. | tests, código | monolito |
| `verify_ifsub_equivalence.py` | Verify each Python ifSUB mode is faithful to its own GAMS oracle, per dataset. | tests, docs, código |  |
| `welfare_gempack_fixture.py` | Bienestar de GEMPACK en los ejercicios de Burfisher: el oraculo del EV regional. | tests |  |

### `gtap_julia/` (14)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `__init__.py` |  | tests, docs, código |
| `compare.py` | Julia-vs-equilibria cell-by-cell parity tool for the gtap_julia port. | tests, docs, código |
| `dump_calibrated.jl` | Dump Julia's CALIBRATED point (all mc.data keys: α/γ/σ/ϵ params + seeded | tests, código |
| `dump_shock_solution.jl` | Dump Julia's SHOCKED full solution (all mc.data vars after calibrate -> base -> | código |
| `dump_solution.jl` | Dump Julia's full BASE SOLUTION (all mc.data vars after a base solve) plus all | tests, código |
| `eval_residuals.jl` | Evaluate Julia's constraint residuals at an EXTERNAL point (equilibria's shocked | — |
| `export_har_for_julia.py` | Export one of our aggregated GTAP datasets (basedata.har / default.prm / | código |
| `gempack_adapter.py` | Adapt a Julia GTAPv7 base+shock solution (the CSVs dumped by run_from_csv.jl) | — |
| `jparity.py` | jparity — Julia-vs-equilibria NLP-vs-NLP parity for the gtap_julia port. | tests, código |
| `oracle.py` | Run the Julia GTAPv7 model as the reference oracle for the gtap_julia port. | tests, código |
| `run_from_csv.jl` | Run the Julia GTAPv7 model on one of OUR aggregated datasets, fed via the flat | tests, código |
| `run_julia_oracle.jl` | Parameterized Julia GTAPv7 oracle: calibrate -> base -> tariff shock -> dump CSV. | código |
| `run_our_dataset.jl` | Run the Julia GTAPv7 model on ONE OF OUR aggregated datasets (basedata.har / | — |
| `verify_root.jl` | Symmetric-root check for jparity: seed the Julia model at an EXTERNAL point | código |

### `har/` (1)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `oracle_check.py` | Manual sandbox harness: harpy3 as black-box oracle for the HAR writer. | tests, docs, código |

### `paper/` (2)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `model_size.py` | Tamano del modelo GTAP (Seccion 2 del paper): familias de ecuaciones y de variables | tests, código |
| `run_tables.py` | Corre los resultados del paper GTAP (JGEA), un paso por tabla, y deja la salida en --out. | tests, docs |

### `parity/` (21)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `__init__.py` | Generic equilibria parity-debug CLI and adapter registry. | tests, docs, código |
| `_adapter_protocol.py` | ParityAdapter Protocol + per-template registry. | tests, código |
| `_probe_cache.py` | Cache key + store/load for the probe's solve point. Pure (no Pyomo). | tests, código |
| `_probe_params.py` | Param/calibration diff: compare a built model's Pyomo Params vs the GAMS GDX. | tests, código |
| `_probe_queries.py` | Query implementations operating on a built+valued Pyomo GTAP model. | tests, código |
| `_triage_steps.py` | Three triage steps for closing parity gaps: locate → isolate → trace. | código |
| `check_gams_parity.py` | Check parity between Python initialized variables and GAMS benchmark levels. | docs |
| `compare_path_solver_parity.py` | Compare a small MCP solved with PATHAMPL (Pyomo PATH) vs path-capi-python. | tests |
| `compare_pep2_gams_python.py` | Compare pep2 GAMS Results.gdx BASE values against Python PEP initialization. | — |
| `compare_pep_results.py` | Compare Python and GAMS PEP model results. | — |
| `compare_with_gams.py` | Compare solver solution with GAMS baseline. | tests, código |
| `generate_baseline_manifest.py` | Generate strict-gams baseline manifest for a calibrated run. | — |
| `generate_pep_gams_nlp_reference.py` | Generate an official GAMS + IPOPT + NLP reference manifest for PEP. | tests, docs |
| `generate_simple_open_gams_reference.py` | Generate the official GAMS benchmark reference manifest for SimpleOpen. | tests, docs |
| `measure_pep_jacobian_modes.py` | Benchmark analytic vs numeric Jacobian modes on the public PEP scenario pack. | tests, docs |
| `measure_simple_open_jacobian_modes.py` | Benchmark analytic vs numeric Jacobian modes for the simple-open template. | tests, docs |
| `probe.py` | Cached, flag-driven parity probe — Tool #1 of the cascade (seed-at-GAMS scan). | tests, código |
| `run_pep_systemic_parity.py` | Run systemic parity gates for PEP model. | CI, tests, docs |
| `run_simple_open_gams_parity.py` | Compare canonical SimpleOpen closures against GAMS benchmark GDX artifacts. | tests, docs |
| `triage.py` | Generic equilibria parity-debug CLI: locate → isolate → trace → check-warmstart → check-solution. | docs, código |
| `verify_pep2_full_parity.py` | Full parity audit between pep2 GAMS outputs and Python equilibria state. | código |

### `pep/` (9)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `derive_mcp_pairing.py` | Bipartite matching of PEP equations ↔ variables for the MCP pairing. | — |
| `gen_pep_coverage_doc.py` | Generate docs/site/guide/pep_coverage_matrix.md from pep_coverage_matrix.ROWS. | tests, docs, código |
| `mcp_diff_eq_families.py` |  | docs |
| `mcp_economic_pairing.py` |  | — |
| `mcp_find_orphans.py` |  | — |
| `mcp_hopcroft_karp.py` |  | — |
| `mcp_pairing_econ_restricted.py` |  | — |
| `mcp_square_diag.py` |  | — |
| `pep_coverage_matrix.py` | PEP-1-1 parity coverage matrix — the single source of truth. | tests, docs, código |

### `qa/` (3)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `print_structural_report_summary.py` | Print compact summaries for systemic parity JSON reports. | CI, docs |
| `run_sam_qa.py` | Run structural SAM QA gates before PEP calibration/solve. | docs |
| `verify_calibration.py` | Verify that calibrated values satisfy equilibrium conditions. | — |

### `sam_tools/` (4)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `audit_cri_raw_mapping.py` | Audit full raw-CRI SAM account mapping into PEP accounts. | — |
| `create_cri_labor_tax_mapping.py` | Create a CRI mapping variant where social contributions map to taxes. | — |
| `fix_cri_sam_for_pep.py` | Transform and rebalance a CRI SAM into a PEP-compatible structure. | — |
| `run_manual_sam_pipeline.py` | Utility script that executes the manual IEEM->PEP pipeline step by step. | docs |

### `windows/` (1)

| Script | Qué hace | Lo nombra |
|---|---|---|
| `corridas_paper.ps1` | Corridas del paper que solo se pueden hacer en Windows (GEMPACK y sltoht). | tests, código |
<!-- MAPA:FIN -->
