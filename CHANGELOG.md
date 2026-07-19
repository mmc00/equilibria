# Changelog

All notable changes to `equilibria` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are curated milestones (user-visible capability), not an exhaustive
commit log.

## [Unreleased]

### Added
- F-docs: restructured documentation site (six sections), architecture
  overview, templates overview, artifact-style parity matrices
  (NLP vs NLP / MCP vs MCP for GTAP and PEP), expanded API reference,
  this changelog, and a `docs-build` CI job.

### Fixed
- GTAP7 15x10 altertax shock parity closed: 94.5/95.8% → **99.5/99.5%** @1%
  (gate floors 94/93 → 99/99), no regression on any other dataset. Three
  stacked causes: (1) the June Armington-shares fix was orphaned on an
  unmerged branch — recovered by cherry-pick; (2) the reference GDX violated
  its own `xfeq` (negative factor demands) — regenerated one-shot on NEOS
  with tight PATH options (`convergence_tolerance 1e-10`, inlined
  `path.opt`); (3) under altertax-CD the GAMS `pxeq/pvaeq/pndeq` rows are
  tautologies (proved under-determined GAMS-vs-GAMS: three runs, three
  parkings), while Python added its own CD-dual `pxeq` determinant — the
  full degenerate triple (`pva/pnd/px`) is now pinned at the reference's
  shock slice, replicating `gtap.holdfixed=1`.
- Removed the unfaithful `1e-8` lower-bound floor on quantity variables
  (GAMS bounds prices only): microscopic cells below the floor made the
  GAMS point infeasible in Python and blew up their paired CES prices.
- `ytax[ft]/[fs]` variable inits aligned with the live `eq_ytax`
  (`Σftrv` / `−Σfbep`; `fs` was hardcoded 0).

## [0.5.1] — 2026-07-18

### Added
- PEP-1-1 v2.1 ported to Pyomo (`equilibria.templates.pep_pyomo`): NLP
  (IPOPT) and MCP (PATH) forms at 100% cell parity vs the GAMS CNS and
  GAMS-native MCP references, base + SIM1 export-tax shock (PR #25).
- PEP parity coverage matrix (`scripts/pep/pep_coverage_matrix.py`) with
  its generated docs page and CI sync gate.
- GTAP7 NLP-vs-NLP and MCP-vs-MCP per-stage fidelity gates
  (`test_gtap7_nlp_parity.py`, `test_gtap7_mcp_parity.py`) with
  NEOS-regenerated references (PR #24).
- GTAP altertax multi-period pipeline (base → check → shock) with the CI
  `.nl` parity gate extended to the check phase (PR #20).
- GTAP7 parity coverage matrix as single source of truth — declarative
  ROWS driving the pytest gates and the generated docs page (PR #21).
- GTAPAgg datasets registered (`gtap7_3x3` … `gtap7_15x10`, incl. the
  10r×15c consolidated GDX).
- equilibria-1.0 roadmap reconciled with real state (PR #27).

### Fixed
- Closed the GTAP7 shock-parity gap (PR #19) and reached 9x10 full NEOS
  parity (sluggish factors + NEOS compile fixes).
- Reverted the unfaithful WCO/RCO→1.0 normalization in the PEP
  calibration; closed xmodel phase-2/3 (PR #26).

## [0.5.0] — 2026-05-20

### Added
- RunGTAP welfare-parity engineering: shadow demand integrator + babel
  HAR writer wired into the GTAP welfare pipeline (PR #10).

## [0.4.0] — 2026-05-20

### Added
- Clean-room HAR writer (`babel.har`): `HarWriter` builder with L3/L5/L7
  record validation (PR #11).
- GTAP welfare decomposition + per-OS benchmarks page (PR #6).
- CGEBox altertax + welfare-decomposition port plans (PR #7).

## [0.3.0] — 2026-05-12

### Added
- GTAP Standard 7 template at **100% parity** (base + shock) vs the GAMS
  NEOS references for the 9x10 and NUS333 datasets.
- Native pure-Python HAR reader in `babel.har` (drops the `harpy3`
  dependency); bundled 9x10/NUS333 HAR datasets behind `load_bundled`.
- Public shock API: `apply_shock` parent + `apply_tariff_shock`.
- Sphinx + MyST + sphinx-gallery documentation site for Read the Docs,
  with MIP→SAM, PEP and GTAP quickstarts and a benchmarks page rendered
  from committed parity CSVs (dual NEOS/local reference + wall-time).
- `ytax(r,gy)` emitted with the 10 canonical GAMS tax streams (PR #3);
  postsim `pdp`/`pmp` recalc for alpha=0 cells.
- MIP→SAM pipeline closure in `sam_tools` (balanced `simple_mip` without
  xfail).

## [0.2.0] — 2026-03-08

### Added
- `simulations` runtime contract: mapping adapters, ieem/gtap/icio model
  adapters, multi-model wrappers, and CLI parity-runner coverage.
