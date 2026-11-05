# One data folder — Phase 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the root and order `docs/`: historical documents go to `docs/archive/` as they are, living documents go to `docs/notes/` in English, `GLOSSARY.md` and `ROADMAP.md` move into `docs/`, and the Spanish file names left in `SPANISH_LEGACY` (outside phase 3b) are renamed.

**Architecture:** `scripts/dev/check_layout.py` gains the allowed top level of `docs/` and loses the root entries this phase empties. Each task moves one family with `git mv`, rewrites every pointer to it, and shrinks an allowance, so the layout check and a link test fail until the move is complete.

**Tech Stack:** Python ≥3.10, pytest, `uv`, Sphinx (`-W`), `scripts/dev/check_layout.py`.

**Spec:** `specs/data/changes/one-data-folder/design.md` (phase 6 in "Phases"), `docs/adr/0010-one-data-folder.md`. Maintainer decisions of 2026-11-03: docs contents in English except the `pep_*` guides (Ricardo, R-PEP); what goes to `docs/archive/` is **not** translated; `GLOSSARY.md` and `ROADMAP.md` go to `docs/` (as in fastapi, pydantic, uv, pandas, numpy, scikit-learn, none keeps them at the root).

## Global Constraints

- No number and no test assertion changes (ADR-0008). Moves are `git mv`, so history follows.
- Archived documents keep their bytes, except relative links that the move breaks.
- Translations keep every number, command, path and code block exactly; only prose changes.
- Names in English and lowercase (`check_layout.py` rules).
- Suite before and after identical, test by test and skip by skip, except renamed tests (mapped old → new in the PR) and the tests this phase adds. Sphinx `-W` builds. `tests/docs/test_doc_examples.py` passes.
- Commits in English, `type(scope): description`, with the session's `Co-Authored-By` line.

### Out of scope

- `runs/`, `scripts/windows/`, `tests/fixtures/gtap7_gempack` (phase 3b), `results/` and `src/equilibria/templates/reference/pep*` (phase 5).
- The theme change (`docs/pydata-theme`, a separate PR).

### Target layout

```
README.md  LICENSE  CITATION.cff  CHANGELOG.md  CONTRIBUTING.md  NOTICE   (+ config files)
docs/
  readme.md        English index of docs/
  glossary.md      was GLOSSARY.md
  roadmap.md       was ROADMAP.md (translated)
  adr/  agents/  assets/  site/      unchanged
  notes/           living notes, English
    architecture/  gdx/  sam/  guides/  pep/ (pep_* and ieem_to_pep_*, untranslated, R-PEP)
  archive/         historical, as written; readme.md lists each file in one line
```

### Destinations (exact)

| From | To | Translate |
|---|---|---|
| `docs/analysis/**`, `docs/findings/*`, `docs/plans/*`, `docs/gtap_gams_mcp_parity_plan.md` | `docs/archive/` (same sub-paths: `analysis/`, `findings/`, `plans/`) | no |
| `docs/architecture/gams_parity_matrix.md` | `docs/archive/architecture/` | no |
| `docs/reference/gdx/{delta_gdx,delta_gdx2,gdx_params,6d_implementation,gdx_geometric_interpolation,multidim_implementation}.md` | `docs/archive/gdx/` | no |
| `reports/*` (3) | `docs/archive/reports/` | no |
| `docs/architecture/monolito_vs_bloques.md`, `registro_de_bloques.md` | `docs/notes/architecture/monolith_vs_blocks.md`, `no_block_registry.md` | yes |
| `docs/reference/gdx/{gdx_reader_guide,multidim_parameters}.md` | `docs/notes/gdx/{reader_guide,multidim_parameters}.md` | yes |
| `docs/technical/*` (4) | `docs/notes/sam/` (`matrix_balancing_methods.md`, `mip_balance_for_cge_models.md`, `mip_balancing_methods.md`, `ras_variants_comparison.md`) | yes (3 in Spanish) |
| `docs/guides/{bolivia_mip_to_sam_complete_guide,mip_to_sam_conversion,mip_to_sam_guide_en,model_jacobian_harness,simple_open_*,simulations_api}.md` | `docs/notes/guides/` (`bolivia_mip_to_sam_complete_guide.md`, `mip_to_sam_conversion.md`, …; `mip_to_sam_guide_en.md` → `mip_to_sam_guide.md`) | yes (the Spanish ones) |
| `docs/guides/pep_*.md` (7), `docs/ieem_to_pep_transformations{,_en,_es}.md` | `docs/notes/pep/` | no (R-PEP) |
| `docs/archive/*` (2, already there) | stay | no |
| `docs/readme.md` | stays, rewritten in English as the index of the layout above | yes |
| `GLOSSARY.md` | `docs/glossary.md` | already English |
| `ROADMAP.md` | `docs/roadmap.md` | yes |
| `examples/{readme,sam/readme,cge/readme,pep/readme}.md` | stay | yes (the Spanish ones) |

Spanish test names in `SPANISH_LEGACY` renamed here (the 3b ones stay): `test_gdxdump_sin_rutas_fijas.py` → `test_gdxdump_no_fixed_paths.py`, `test_sin_registro_de_bloques.py` → `test_no_block_registry.py`, `_desempleo.py` → `_unemployment.py`, `test_driver_default_es_bloques.py` → `test_driver_default_is_blocks.py`, `test_nus333_bienestar.py` → `test_nus333_welfare.py`, `test_nus333_me9_pasos.py` → `test_nus333_me9_steps.py`, `test_nus333_tbl65b_desempleo.py` → `test_nus333_tbl65b_unemployment.py`.

## Review Focus

1. A link to a moved document left broken (Markdown links in `docs/`, `README.md`, ADRs, docstrings, `scripts/dev/gen_scripts_map.py`, `docs/site/_scripts/render_benchmarks.py`): Task 0's link test fails on any relative link whose target does not exist, and on any text mention of a moved path.
2. A translation that changes a number, a command or a code block: each translated file's fenced code blocks and numbers must be identical before and after (Task 4 test, run on the translated files).
3. A test renamed that the suite diff then reports as "missing": the PR lists the old → new map, and the diff is checked with that map applied.
4. A tool that reads `GLOSSARY.md` or `ROADMAP.md` at the root (`docs/agents/domain.md` tells agents where the glossary is; `check_layout.ALLOWED_EXACT`; the Sphinx changelog/benchmarks pages): Task 2 rewrites every one, and the link test sees them.
5. The personal `CLAUDE.md` (in `~/proyectos/dev-tools/equilibria-tools/agent-docs/`, not in this repo) points at `GLOSSARY.md` at the root: Task 6 updates it in that repo, in a separate commit, after asking.

---

### Task 0: baseline, layout rules and a link test

**Files:** Modify `scripts/dev/check_layout.py`; Create `tests/docs/test_doc_links.py`.

- [ ] **Step 1:** Baseline: full suite on main with `--junitxml` (with the `modelcache`, `ipopt`, `docs` extras), saved outside the repo.
- [ ] **Step 2: Write the tests.** `test_doc_links.py::test_relative_links_resolve` parses every tracked `*.md` outside `specs/` and `docs/archive/` for `](relative/path)` links and asserts each target exists. `test_no_mention_of_moved_paths` greps code, `docs/` (except `docs/archive/`) and the root `*.md` for each "From" path of this plan; it starts with an allowance holding today's hits, which each task shrinks.
- [ ] **Step 3:** In `check_layout.py`, add `DOCS_TOP = {"readme.md", "glossary.md", "roadmap.md", "adr", "agents", "assets", "site", "notes", "archive"}` plus a `DOCS_TOP_TEMPORARY` holding today's other entries; an entry under `docs/` outside both fails. Run it: PASS. Remove one temporary entry by hand: FAIL. Put it back.
- [ ] **Step 4:** Commit `test(docs): pin the docs layout and the relative links (phase 6)`.

### Task 1: the archive

- [ ] **Step 1:** Shrink `DOCS_TOP_TEMPORARY` (drop `analysis`, `findings`, `plans`, `gtap_gams_mcp_parity_plan.md`) and `ROOT_DIRS_TEMPORARY` (drop `reports`, `output`); `check_layout` FAILS.
- [ ] **Step 2:** `git mv` per the table. Write `docs/archive/readme.md`: what the archive is (written then, not translated, not maintained) and one line per file (path, date, one-sentence English summary).
- [ ] **Step 3:** Fix the relative links inside archived files that the move broke (only those), and every pointer to them outside the archive.
- [ ] **Step 4:** `check_layout.py` clean; `pytest tests/docs tests/scripts` PASS; Sphinx `-W`.
- [ ] **Step 5:** Commit `refactor(docs): historical documents to docs/archive (phase 6)`.

### Task 2: glossary and roadmap into docs/

- [ ] **Step 1:** Drop `GLOSSARY.md` and `ROADMAP.md` from `check_layout.ALLOWED_EXACT`/`ROOT_FILES`; FAIL.
- [ ] **Step 2:** `git mv GLOSSARY.md docs/glossary.md`; `git mv ROADMAP.md docs/roadmap.md`, then translate `docs/roadmap.md` to English.
- [ ] **Step 3:** Rewrite every pointer (README, `docs/agents/domain.md`, ADRs, `docs/site`, `render_benchmarks.py`, `gen_scripts_map.py`).
- [ ] **Step 4:** `pytest tests/docs tests/scripts` PASS; Sphinx `-W`; commit `refactor(docs): glossary and roadmap move into docs/ (phase 6)`.

### Task 3: living notes in English

- [ ] **Step 1:** Shrink `DOCS_TOP_TEMPORARY` (drop `architecture`, `reference`, `technical`, `guides`, `ieem_to_pep_*`); FAIL.
- [ ] **Step 2:** `git mv` per the table (one commit for the moves, so the translations show as diffs).
- [ ] **Step 3:** Translate the Spanish notes (architecture 2, gdx 2, sam 3, guides 2). Not the `pep/` ones.
- [ ] **Step 4:** Rewrite pointers; drop the two `docs/architecture/*` entries from `SPANISH_LEGACY`.
- [ ] **Step 5:** `pytest tests/docs tests/scripts tests/blocks` PASS; Sphinx `-W`; commit `docs(notes): living notes in docs/notes, in English (phase 6)`.

### Task 4: translation fidelity, docs/readme and examples

- [ ] **Step 1: Write the test.** `tests/docs/test_translations_keep_code.py`: for each file translated in this phase (a list in the test, old path at `origin/main` → new path), the fenced code blocks and the multiset of numbers are identical between the `main` version and the translation. Plant a changed number in one translation: FAIL.
- [ ] **Step 2:** Rewrite `docs/readme.md` in English as the index of the target layout; translate the Spanish `examples/**/readme.md`.
- [ ] **Step 3:** `pytest tests/docs` PASS; commit `docs: docs index and example readmes in English (phase 6)`.

### Task 5: Spanish test names

- [ ] **Step 1:** Drop the 7 test entries from `SPANISH_LEGACY`; `check_layout` FAILS.
- [ ] **Step 2:** `git mv` per the list above; fix imports (`_desempleo` → `_unemployment`) and every text mention.
- [ ] **Step 3:** Run those tests; full suite with `--junitxml`; diff against the baseline with the rename map applied: identical by name and skip reason, plus the new tests.
- [ ] **Step 4:** Commit `refactor(tests): the last Spanish test names outside phase 3b (phase 6)`.

### Task 6: verification and pointers outside the repo

- [ ] **Step 1:** `check_layout.py` clean; `DOCS_TOP_TEMPORARY` is empty and removed; the link test's allowance is empty and removed.
- [ ] **Step 2:** Sphinx `-W`; `tests/docs/test_doc_examples.py`; full suite already diffed in Task 5.
- [ ] **Step 3:** Ask the maintainer, then update `~/proyectos/dev-tools/equilibria-tools/agent-docs/CLAUDE.md` (`GLOSSARY.md` → `docs/glossary.md`) in the dev-tools repo.
- [ ] **Step 4:** Ask before push and PR.
