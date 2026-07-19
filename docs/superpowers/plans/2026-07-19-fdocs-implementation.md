# F-docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute F-docs (roadmap §3.5): restructure the Sphinx site into 6 sections, render the GTAP/PEP coverage matrices and benchmarks in the artifact visual format (NLP vs NLP / MCP vs MCP headers), expand the API autodoc, create CHANGELOG.md, review quickstarts, and add a `docs-build -W` CI job.

**Architecture:** All comparison pages get their markup from one shared helper (`docs/site/_scripts/matrix_html.py`) + one CSS file (`docs/site/_static/matrix.css`); the two coverage-doc generators emit MyST pages with embedded `{raw} html` blocks (the CI sync tests keep their string-equality mechanism); `render_benchmarks.py` reuses the same helper at build time. The site toctree moves from one flat "User guide" to 6 captioned toctrees in `index.md`.

**Tech Stack:** Sphinx 7 + furo + MyST + sphinx-gallery, pytest, uv, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-07-18-fdocs-design.md`

## Global Constraints

- **Docs-only:** nothing under `src/equilibria/` changes. A broken docstring found during autodoc is *recorded as debt* in the PR body, not fixed.
- Existing guide pages do **not move files** (zero URL churn); only toctree membership changes.
- The sync gates keep their mechanism: `test_coverage_doc_in_sync` (gtap) and `test_pep_coverage_doc_in_sync` (pep) compare committed `.md` == `render()`. After every generator change: regenerate, run the test, commit both.
- The `.nl` canary must stay green after every task: `uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q`.
- The site must build clean under `-W` after every task: `uv run sphinx-build -W -b html docs/site docs/site/_build/html`.
- New site pages are written in **English** (site language). Coverage rows/floors come from `scripts/gtap/coverage_matrix.py` / `scripts/pep/pep_coverage_matrix.py` — **never hardcode row data in the generators**.
- CSS classes are namespaced `mx-` to avoid colliding with furo.
- Commit at the end of every task (repo discipline: every validation run has a clean commit).

---

### Task 1: Baseline `-W` build green

Make `sphinx-build -W` pass on the site as it is today, so every later task has a hard gate.

**Files:**
- Modify: `docs/site/conf.py`
- Possibly modify: `docs/site/index.md` (only if a warning demands it)

**Interfaces:**
- Produces: the command `uv run sphinx-build -W -b html docs/site docs/site/_build/html` exiting 0 — the gate every later task runs.

- [ ] **Step 1: Install docs deps and run the baseline build**

```bash
uv sync --extra docs
uv run sphinx-build -W -b html docs/site docs/site/_build/html 2>&1 | tail -30
```

Expected: FAIL (warnings treated as errors). Record every warning. Known/likely offenders and their exact fixes:

| Warning | Fix |
|---|---|
| `docs/site/sg_execution_times.rst: document isn't included in any toctree` | Add `"sg_execution_times.rst"` to `exclude_patterns` in `conf.py` |
| `gallery/... not included in any toctree` | Already covered by `gallery/index` toctree — if it still warns, add `"gallery/*/index.rst"`-style entries to `exclude_patterns` only for the offending generated files |
| myst header / cross-ref warnings in legacy pages | Fix the offending markdown in place (docs-only edits) |

- [ ] **Step 2: Apply fixes, rebuild until clean**

In `docs/site/conf.py`:

```python
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "sg_execution_times.rst"]
```

Re-run the build command until it exits 0. If a warning originates in a *docstring inside `src/equilibria/`*: do NOT fix the source — note it in a `DEUDAS.md` scratch note (later pasted into the PR body) and, if it blocks the build, silence at the docs layer (e.g., drop that automodule — not applicable yet in this task).

- [ ] **Step 3: Verify the canary still passes**

```bash
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q
```

Expected: PASS (5 datasets).

- [ ] **Step 4: Commit**

```bash
git add docs/site/conf.py docs/site/index.md
git commit -m "docs: make sphinx-build -W pass on the current site (baseline for F-docs)"
```

---

### Task 2: Shared artifact-format infra — `matrix.css` + `matrix_html.py`

**Files:**
- Create: `docs/site/_static/matrix.css`
- Create: `docs/site/_scripts/matrix_html.py`
- Modify: `docs/site/conf.py` (add `html_css_files`)

**Interfaces:**
- Produces (used by Tasks 3, 4, 5):
  - `matrix_html.raw(html: str) -> str` — wraps HTML in a MyST `{raw} html` fence
  - `matrix_html.chip(label: str, tone: str) -> str` — tone ∈ {"good","warn","bad","neutral"}
  - `matrix_html.num(value: str, tone: str) -> str`
  - `matrix_html.cell(*parts: str) -> str`
  - `matrix_html.label(ds: str, sub: str = "") -> str`
  - `matrix_html.ref(text: str) -> str`
  - `matrix_html.tablecard(headers: list[str], rows: list[list[str]]) -> str`
  - `matrix_html.note(html: str) -> str`
  - `matrix_html.legend(items_html: str) -> str`
  - `matrix_html.floor_tone(f: float) -> str` — ≥99 → "good", ≥95 → "warn", else "bad"

- [ ] **Step 1: Write `docs/site/_static/matrix.css`**

```css
/* matrix.css — artifact-style parity components (mx-*), single source for the
   coverage matrices and benchmarks. Light on :root; dark via furo body[data-theme]
   plus prefers-color-scheme for the "auto" theme. */
:root {
  --mx-panel: #ffffff;
  --mx-ink: #1a2129;
  --mx-ink-soft: #5a6673;
  --mx-ink-faint: #8b96a3;
  --mx-line: #e2e6eb;
  --mx-line-soft: #eef1f4;
  --mx-accent: #2f6fb0;
  --mx-accent-soft: #e8f0f8;
  --mx-good: #2f8f5b;  --mx-good-bg: #e6f3ec;
  --mx-warn: #b5851f;  --mx-warn-bg: #f7efd8;
  --mx-bad: #c05038;   --mx-bad-bg: #f7e5df;
  --mx-mono: "SFMono-Regular", "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
}
body[data-theme="dark"] {
  --mx-panel: #161d26; --mx-ink: #e6ebf1; --mx-ink-soft: #9aa7b4;
  --mx-ink-faint: #6a7885; --mx-line: #26313d; --mx-line-soft: #1d2732;
  --mx-accent: #5b9bd8; --mx-accent-soft: #1a2836;
  --mx-good: #4bb37c; --mx-good-bg: #14291e;
  --mx-warn: #d0a542; --mx-warn-bg: #2c2411;
  --mx-bad: #db6a4f;  --mx-bad-bg: #2e1a14;
}
@media (prefers-color-scheme: dark) {
  body[data-theme="auto"] {
    --mx-panel: #161d26; --mx-ink: #e6ebf1; --mx-ink-soft: #9aa7b4;
    --mx-ink-faint: #6a7885; --mx-line: #26313d; --mx-line-soft: #1d2732;
    --mx-accent: #5b9bd8; --mx-accent-soft: #1a2836;
    --mx-good: #4bb37c; --mx-good-bg: #14291e;
    --mx-warn: #d0a542; --mx-warn-bg: #2c2411;
    --mx-bad: #db6a4f;  --mx-bad-bg: #2e1a14;
  }
}
.mx-card { background: var(--mx-panel); border: 1px solid var(--mx-line);
  border-radius: 12px; overflow: hidden; margin: 12px 0 20px; }
.mx-scroll { overflow-x: auto; }
.mx-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; margin: 0; }
.mx-table thead th { font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--mx-ink-faint); font-weight: 600; text-align: center;
  padding: 12px 10px 10px; border-bottom: 1px solid var(--mx-line); white-space: nowrap; }
.mx-table thead th.mx-lbl { text-align: left; padding-left: 18px; }
.mx-table tbody td { padding: 11px 10px; border-bottom: 1px solid var(--mx-line-soft);
  text-align: center; font-family: var(--mx-mono); font-size: 13px; }
.mx-table tbody tr:last-child td { border-bottom: none; }
.mx-table td.mx-lbl { text-align: left; padding-left: 18px; font-family: inherit; white-space: nowrap; }
.mx-ds { font-weight: 600; font-size: 13.5px; }
.mx-sub { font-family: var(--mx-mono); font-size: 11px; color: var(--mx-ink-faint); margin-left: 7px; }
.mx-cell { display: inline-flex; flex-direction: column; align-items: center; gap: 3px; }
.mx-num { font-weight: 560; font-size: 13.5px; }
.mx-num.mx-good { color: var(--mx-good); }
.mx-num.mx-warn { color: var(--mx-warn); }
.mx-num.mx-bad { color: var(--mx-bad); }
.mx-chip { font-family: var(--mx-mono); font-size: 9.5px; letter-spacing: .04em;
  padding: 1px 6px; border-radius: 20px; text-transform: uppercase; font-weight: 600; }
.mx-chip.mx-good { background: var(--mx-good-bg); color: var(--mx-good); }
.mx-chip.mx-warn { background: var(--mx-warn-bg); color: var(--mx-warn); }
.mx-chip.mx-bad { background: var(--mx-bad-bg); color: var(--mx-bad); }
.mx-chip.mx-neutral { background: var(--mx-accent-soft); color: var(--mx-accent); }
.mx-ref { font-family: var(--mx-mono); font-size: 11px; color: var(--mx-ink-faint); }
.mx-legend { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 16px 0;
  padding: 14px 16px; background: var(--mx-panel); border: 1px solid var(--mx-line);
  border-radius: 10px; font-size: 12.5px; color: var(--mx-ink-soft); }
.mx-legend .mx-li { display: inline-flex; align-items: center; gap: 7px; }
.mx-swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; display: inline-block; }
.mx-note { display: flex; gap: 9px; margin: 14px 0 20px; padding: 13px 16px;
  background: var(--mx-panel); border: 1px solid var(--mx-line);
  border-left: 3px solid var(--mx-accent); border-radius: 8px; font-size: 13px;
  color: var(--mx-ink-soft); }
.mx-note b { color: var(--mx-ink); }
```

- [ ] **Step 2: Write `docs/site/_scripts/matrix_html.py`**

```python
"""Shared HTML building blocks for the artifact-style parity matrices.

Consumed by scripts/gtap/gen_coverage_doc.py, scripts/pep/gen_pep_coverage_doc.py
(committed golden .md files, sync-tested in CI) and by render_benchmarks.py at
build time. Styling lives in _static/matrix.css (loaded site-wide via conf.py's
html_css_files); this module only emits markup with the mx-* classes.
"""
from __future__ import annotations

_TONES = {"good", "warn", "bad", "neutral"}


def raw(html: str) -> str:
    """Wrap html in a MyST raw fence so it passes through Sphinx untouched."""
    return "```{raw} html\n" + html + "\n```"


def chip(label: str, tone: str) -> str:
    assert tone in _TONES, tone
    return f'<span class="mx-chip mx-{tone}">{label}</span>'


def num(value: str, tone: str) -> str:
    assert tone in _TONES, tone
    return f'<span class="mx-num mx-{tone}">{value}</span>'


def cell(*parts: str) -> str:
    return '<div class="mx-cell">' + "".join(parts) + "</div>"


def label(ds: str, sub: str = "") -> str:
    s = f'<span class="mx-ds">{ds}</span>'
    if sub:
        s += f'<span class="mx-sub">{sub}</span>'
    return s


def ref(text: str) -> str:
    return f'<span class="mx-ref">{text}</span>'


def tablecard(headers: list[str], rows: list[list[str]]) -> str:
    """First column is the left-aligned label column; the rest are centered."""
    thead = "".join(
        f'<th class="mx-lbl">{h}</th>' if i == 0 else f"<th>{h}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for r in rows:
        tds = "".join(
            f'<td class="mx-lbl">{c}</td>' if i == 0 else f"<td>{c}</td>"
            for i, c in enumerate(r)
        )
        body.append(f"<tr>{tds}</tr>")
    return (
        '<div class="mx-card"><div class="mx-scroll"><table class="mx-table">'
        f"<thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody>"
        "</table></div></div>"
    )


def note(html: str) -> str:
    return f'<div class="mx-note"><span>⤷</span><span>{html}</span></div>'


def legend(items_html: str) -> str:
    return f'<div class="mx-legend">{items_html}</div>'


def floor_tone(f: float) -> str:
    return "good" if f >= 99 else "warn" if f >= 95 else "bad"
```

- [ ] **Step 3: Load the CSS site-wide in `conf.py`**

After `html_static_path = ["_static"]` add:

```python
html_css_files = ["matrix.css"]
```

- [ ] **Step 4: Verify build + canary**

```bash
uv run sphinx-build -W -b html docs/site docs/site/_build/html
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q
```

Expected: both PASS (the helper is not used yet — this proves the CSS/conf change is inert).

- [ ] **Step 5: Commit**

```bash
git add docs/site/_static/matrix.css docs/site/_scripts/matrix_html.py docs/site/conf.py
git commit -m "docs: shared artifact-format infra (matrix.css + matrix_html helper)"
```

---

### Task 3: GTAP coverage matrix → artifact format (NLP vs NLP / MCP vs MCP)

**Files:**
- Modify: `scripts/gtap/gen_coverage_doc.py` (rewrite `render()`; drop `.nl`/SOLVE tables)
- Regenerate: `docs/site/guide/gtap7_coverage_matrix.md`
- Test: `tests/templates/gtap/test_coverage_matrix.py::test_coverage_doc_in_sync`

**Interfaces:**
- Consumes: `matrix_html` (Task 2 API), `coverage_matrix.nlp_rows()/mcp_rows()` — each Row has `.dataset .ifsub .ref .mode ("pure"|"altertax") .stage_floors (tuple of (stage, floor)) .code2_ok (tuple of stage names)`.
- Produces: the committed artifact-format `gtap7_coverage_matrix.md`. `coverage_matrix.py` itself is **untouched** (its ROWS still drive all the tests).

- [ ] **Step 1: Rewrite `scripts/gtap/gen_coverage_doc.py`**

Full new content:

```python
"""Generate docs/site/guide/gtap7_coverage_matrix.md from coverage_matrix.ROWS.

Run:  uv run python scripts/gtap/gen_coverage_doc.py
The output is a committed golden file; test_coverage_doc_in_sync enforces that the
committed file equals render() (CI fails on drift).

The page renders ONLY the two same-engine fidelity gates (NLP vs NLP, MCP vs MCP)
in the artifact card format. The .nl coefficient gate and the legacy PATH SOLVE
gates (kind altertax/gtap_solve) remain in coverage_matrix.ROWS and keep driving
their pytest gates — they are just no longer rendered as tables here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "docs/site/_scripts"))

from coverage_matrix import nlp_rows, mcp_rows  # noqa: E402
import matrix_html as mx  # noqa: E402

DOC_PATH = ROOT / "docs/site/guide/gtap7_coverage_matrix.md"

_BANNER = (
    "<!-- GENERATED FROM scripts/gtap/coverage_matrix.py — do not edit by hand.\n"
    "     Regenerate: uv run python scripts/gtap/gen_coverage_doc.py -->"
)

_LEGEND = mx.legend(
    '<span class="mx-li">Each cell is the per-stage <b>floor the pytest gate '
    "asserts</b> (measured match% @ tol 1% must be ≥ floor):"
    '<span class="mx-swatch" style="background:var(--mx-good);margin-left:4px"></span>≥99'
    '<span class="mx-swatch" style="background:var(--mx-warn)"></span>95–99'
    '<span class="mx-swatch" style="background:var(--mx-bad)"></span>&lt;95</span>'
    '<span class="mx-li"><b>convergence</b> chip: '
    + mx.chip("✓ code 1", "good") + " required "
    + mx.chip("code ≤ 2", "warn") + " audited near-miss</span>"
    '<span class="mx-li">' + mx.chip("local", "neutral")
    + " gates need PATH/IPOPT + GAMS refs — run by hand, not in CI</span>"
)


def _stage_cell(r, stage: str) -> str:
    floors = dict(r.stage_floors)
    f = floors[stage]
    conv = mx.chip("code ≤ 2", "warn") if stage in r.code2_ok else mx.chip("✓ code 1", "good")
    return mx.cell(mx.num(f"≥ {f:g}", mx.floor_tone(f)), conv)


def _gate_table(rows) -> str:
    headers = ["Dataset · ifSUB", "Base", "Check", "Shock", "Reference"]
    body = [
        [
            mx.label(r.dataset, f"ifSUB={r.ifsub}"),
            _stage_cell(r, "base"),
            _stage_cell(r, "check"),
            _stage_cell(r, "shock"),
            mx.ref(r.ref),
        ]
        for r in rows
    ]
    return mx.tablecard(headers, body)


def render() -> str:
    nlp = list(nlp_rows())
    mcp = list(mcp_rows())
    parts = [
        "# GTAP 7 Parity Coverage Matrix",
        "",
        _BANNER,
        "",
        "Same-engine parity between the Python `equilibria` GTAP Standard 7 "
        "implementation and GAMS: **NLP vs NLP** (IPOPT both sides) and "
        "**MCP vs MCP** (PATH both sides), so the solver's equality tolerance "
        "cancels and the cell-by-cell match reflects **model fidelity**, not "
        "solver noise. Cells show the conservative per-stage **floor** each "
        "pytest gate asserts (the gate runs the real solve, measures match% "
        "@ tol 1% and the return code, and requires `match ≥ floor` and "
        "`code == 1`); the measured snapshot is not stored here — the live "
        "views re-run the measurement. A separate solver-free `.nl` "
        "coefficient gate (0 Jacobian diffs vs GAMS, "
        "`test_gtap7_nl_parity.py`) runs in CI as the structural canary.",
        "",
        mx.raw(_LEGEND),
        "",
        "## NLP vs NLP",
        "",
        "Python solved as an NLP (`EQUILIBRIA_GTAP_SOLVE_NLP=1`, maximize "
        "walras) against the GAMS `ifMCP=0` NLP reference — same IPOPT both "
        "sides. Gate: `test_gtap7_nlp_parity.py` (local; needs IPOPT + the "
        "committed GAMS refs). Measured view: "
        "[live NLP matrix](../_static/gtap7_nlp_matrix.html).",
        "",
        "### Pure-gtap (real-CES)",
        "",
        mx.raw(_gate_table([r for r in nlp if r.mode == "pure"])),
        "",
        mx.raw(mx.note(
            "<b>100% across every stage, both ifSUB</b>, after the Jacobian "
            "pre-scale skip (commit e4c40d7) — GAMS solves the raw model; the "
            "Python-only pre-scale steered IPOPT to a wrong basin on the 5×5 "
            "shock (59.56% → 100%)."
        )),
        "",
        "### Altertax (CD)",
        "",
        mx.raw(_gate_table([r for r in nlp if r.mode == "altertax"])),
        "",
        mx.raw(mx.note(
            "<b>The check/shock ceiling here is the reference, not the "
            "model.</b> The altertax NLP references are mis-converged (IPOPT "
            "\"Locally Optimal\", the ref violates its own eq_pxeq in the ag "
            "sector). Where a cleanly-converged MCP reference exists the same "
            "Python solve reaches 99.9% — see the MCP gate below."
        )),
        "",
        "## MCP vs MCP",
        "",
        "Python solved via PATH (nonlinear-full MCP) against the "
        "cleanly-converged **NEOS** MCP references — same PATH both sides. "
        "Gate: `test_gtap7_mcp_parity.py` (local; needs PATH + the committed "
        "refs). This is the daily fidelity gate. Measured view: "
        "[live MCP matrix](../_static/gtap7_mcp_matrix.html).",
        "",
        "### Pure-gtap (real-CES)",
        "",
        mx.raw(_gate_table([r for r in mcp if r.mode == "pure"])),
        "",
        "### Altertax (CD)",
        "",
        mx.raw(_gate_table([r for r in mcp if r.mode == "altertax"])),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    DOC_PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
```

**Nota:** si al correr esto los floors/notes de `coverage_matrix.py` difieren de los citados en las notas de texto (p. ej. tras un rebase donde 15×10 cerró a 99.5), la nota de texto se ajusta a lo que digan los ROWS — los ROWS mandan.

- [ ] **Step 2: Regenerate and run the sync test (expect it to pass against the fresh file)**

```bash
uv run python scripts/gtap/gen_coverage_doc.py
uv run pytest tests/templates/gtap/test_coverage_matrix.py -q
```

Expected: `wrote …gtap7_coverage_matrix.md`, then PASS (all tests in the file — the schema tests read ROWS, untouched).

- [ ] **Step 3: Build the site and eyeball the page**

```bash
uv run sphinx-build -W -b html docs/site docs/site/_build/html
open docs/site/_build/html/guide/gtap7_coverage_matrix.html
```

Expected: build exits 0; the page shows the legend card + 4 tablecards with `≥ 99` tinted numbers and `✓ code 1` chips; dark-mode toggle recolors them.

- [ ] **Step 4: Commit**

```bash
git add scripts/gtap/gen_coverage_doc.py docs/site/guide/gtap7_coverage_matrix.md
git commit -m "docs(gtap): coverage matrix page → artifact format, NLP-vs-NLP + MCP-vs-MCP only"
```

---

### Task 4: PEP coverage matrix → artifact format

**Files:**
- Modify: `scripts/pep/gen_pep_coverage_doc.py` (rewrite `render()`)
- Regenerate: `docs/site/guide/pep_coverage_matrix.md`
- Test: `tests/templates/pep_pyomo/test_pep_coverage_matrix.py::test_pep_coverage_doc_in_sync`

**Interfaces:**
- Consumes: `matrix_html` (Task 2 API), `pep_coverage_matrix.nlp_rows()/mcp_rows()/mirror_rows()/variant_rows()` — each Row has `.scenario .form .cells .match_note .gate .ci_status .ref`.
- Produces: the committed artifact-format `pep_coverage_matrix.md`.

- [ ] **Step 1: Rewrite `scripts/pep/gen_pep_coverage_doc.py`**

Keep the module docstring, imports, `DOC_PATH`, `_BANNER`, and `main()` as they are; add the `matrix_html` import next to the existing sys.path block and replace `_HEADER`/`_row_cells`/`_table`/`render()`:

```python
sys.path.insert(0, str(ROOT / "docs/site/_scripts"))
import matrix_html as mx  # noqa: E402


def _match_tone(match_note: str) -> str:
    return "good" if match_note.startswith("100") else "warn"


def _pep_table(rows) -> str:
    headers = ["Scenario · form", "Cells", "Match", "Gate", "Reference"]
    body = [
        [
            mx.label(r.scenario, r.form),
            str(r.cells),
            mx.cell(
                mx.num(r.match_note, _match_tone(r.match_note)),
                mx.chip("local", "neutral"),
            ),
            mx.ref(r.gate),
            mx.ref(r.ref),
        ]
        for r in rows
    ]
    return mx.tablecard(headers, body)


_LEGEND = mx.legend(
    '<span class="mx-li"><b>match</b> is the measured snapshot; the named '
    "pytest gate re-measures and enforces it</span>"
    '<span class="mx-li">' + mx.chip("local", "neutral")
    + " every row needs PATH/IPOPT + GAMS refs — no solver-free CI gate "
    "(the PEP reference is a solved GDX, not a .nl dump)</span>"
)


def render() -> str:
    parts = [
        "# PEP-1-1 Parity Coverage Matrix",
        "",
        _BANNER,
        "",
        f"Coverage for the `{DATASET}` dataset — the PEP-1-1 v2.1 CGE model "
        "ported to Pyomo (`equilibria.templates.pep_pyomo`). Every row "
        "compares the Pyomo solve against a GAMS reference solved by the "
        "**same engine** (IPOPT vs IPOPT, or PATH vs PATH), so the solver's "
        "equality tolerance cancels and the cell-by-cell match reflects "
        "**model fidelity**, not solver noise.",
        "",
        mx.raw(_LEGEND),
        "",
        "## NLP vs NLP",
        "",
        "The Pyomo model solved as an NLP (IPOPT on the raw model, "
        "`nlp_scaling_method=none`, faithful to GAMS's raw solve) against the "
        "GAMS CNS reference `Results.gdx`. The benchmark BASE reproduces the "
        "SAM, so the seeded point is the calibration answer. Run: "
        "`phase1_nlp.py --model pep --dataset pep2 --period BASE`.",
        "",
        mx.raw(_pep_table(nlp_rows())),
        "",
        "## MCP vs MCP",
        "",
        "The Pyomo model solved as a complementarity problem via PATH against "
        "the **GAMS-native** MCP reference (`PEP-1-1_v2_1_mcp_solve.gms`: "
        "`MODEL /ALL/` + `SOLVE USING MCP`, so GAMS infers the "
        "equation↔variable pairing). The `sim1` row is the reference "
        "counterfactual — a 25% export-tax cut (`ttix.fx=ttixO*0.75`) — "
        "applied faithfully in Python by scaling the `ttixO` benchmark before "
        "build. Run: `phase1_nlp.py --model pep --dataset pep2 --form mcp`.",
        "",
        mx.raw(_pep_table(mcp_rows())),
        "",
        "## NLP↔MCP mirror",
        "",
        "The two Pyomo forms, solved from the same feasible benchmark seed, "
        "land on the identical point (LEON, the form-defining Walras slack, "
        "is excluded). The one historical gap — `PD['othind']`, filled from "
        "its `*O` benchmark (1.132) rather than a blind 1.0 — closed the "
        "mirror to a clean 100%.",
        "",
        mx.raw(_pep_table(mirror_rows())),
        "",
        "## objdef variant",
        "",
        "The `objdef` variant adds a dummy objective (`OBJDEF: OBJ==0`, "
        "minimize OBJ) — the `SOLVE NLP MINIMIZING OBJ` lineage. A constant "
        "objective cannot move the equilibrium, so objdef-NLP lands on the "
        "exact base-NLP point; in the MCP form OBJ is not declared, keeping "
        "the system square.",
        "",
        mx.raw(_pep_table(variant_rows())),
        "",
    ]
    return "\n".join(parts)
```

**Cuidado:** `variant_rows()` incluye una fila cuyo `match_note` es `"square, code=1"` — `_match_tone` la pinta "warn"; eso es correcto (no es un match del 100%, es un contrato de squareness).

- [ ] **Step 2: Regenerate + sync test**

```bash
uv run python scripts/pep/gen_pep_coverage_doc.py
uv run pytest tests/templates/pep_pyomo/test_pep_coverage_matrix.py -q
```

Expected: PASS.

- [ ] **Step 3: Build + eyeball**

```bash
uv run sphinx-build -W -b html docs/site docs/site/_build/html
open docs/site/_build/html/guide/pep_coverage_matrix.html
```

- [ ] **Step 4: Commit**

```bash
git add scripts/pep/gen_pep_coverage_doc.py docs/site/guide/pep_coverage_matrix.md
git commit -m "docs(pep): coverage matrix page → artifact format, NLP-vs-NLP + MCP-vs-MCP headers"
```

---

### Task 5: Benchmarks page → same visual language

**Files:**
- Modify: `docs/site/_scripts/render_benchmarks.py`
- Note: `docs/site/guide/benchmarks.md` is overwritten at every build by `conf.py` — commit whatever the build regenerates so the repo copy matches.

**Interfaces:**
- Consumes: `matrix_html` (same directory — plain `import matrix_html as mx` works both from `conf.py` (which already puts `_scripts` on sys.path) and standalone).

- [ ] **Step 1: Restyle the summary/timing blocks**

At the top of `render_benchmarks.py` add:

```python
import matrix_html as mx
```

Replace `_parity_block` with:

```python
def _parity_block(rows: list[dict], heading: str) -> list[str]:
    """Render a parity summary card + top-diverging cards for `rows`."""
    parts = [f"### {heading}\n"]
    headers = ["Phase", "Vars matched", "Cells", "Match", "Diverge",
               "Missing", "Match rate", "Residual", "Solve time"]
    body = []
    for phase in ("base", "shock"):
        s = _summary_row(rows, phase)
        if not s:
            continue
        cells = int(s["cells"]); match = int(s["match"])
        diverge = int(s["diverge"]); missing = int(s["missing"])
        n_vars = sum(1 for r in rows
                     if r["phase"] == phase and r["var"] != "__SUMMARY__"
                     and int(r["diverge"]) == 0 and int(r["missing"]) == 0
                     and r["py_var"])
        n_total_vars = sum(1 for r in rows
                           if r["phase"] == phase and r["var"] != "__SUMMARY__")
        try:
            secs = float(s.get("solve_seconds") or 0.0)
            secs_str = f"{secs:.2f}s"
        except (ValueError, TypeError):
            secs_str = "—"
        rate = (match / cells * 100) if cells else 0.0
        rate_cell = mx.cell(
            mx.num(f"{rate:.2f}%", mx.floor_tone(rate)),
            mx.chip("✓ match" if diverge == 0 else f"{diverge} diverge",
                    "good" if diverge == 0 else "warn"),
        )
        body.append([
            mx.label(phase), str(n_vars) + "/" + str(n_total_vars), str(cells),
            str(match), str(diverge), str(missing), rate_cell,
            mx.ref(f"{float(s['residual']):.2e}"), mx.ref(secs_str),
        ])
    parts.append(mx.raw(mx.tablecard(headers, body)))
    parts.append("")

    for phase in ("base", "shock"):
        worst = _top_diverging(rows, phase, n=10)
        if not worst:
            continue
        parts.append(f"#### Top diverging variables — `{phase}`\n")
        tbody = [
            [mx.label(r["var"], r["py_var"]), r["cells"], r["diverge"],
             mx.ref(r["max_abs_err"]), mx.ref(r["max_rel_err"])]
            for r in worst
        ]
        parts.append(mx.raw(mx.tablecard(
            ["Var · py var", "Cells", "Diverge", "Max abs err", "Max rel err"],
            tbody)))
        parts.append("")
    return parts
```

In `_timing_block`, replace everything from the two `_stat(...)` destructuring lines (`n_py, med_py, mn_py, mx_py, mean_py = _stat(py)` and the `gams` one) through the final ratio `parts.append(...)`, up to but not including the closing `parts.append("")` / `return parts`, with:

```python
    n_py, med_py, mn_py, mx_py_, mean_py = _stat(py)
    n_g, med_g, mn_g, mx_g, mean_g = _stat(gams)
    body = [
        [mx.label("Python equilibria", "PATH C API, nonlinear full"),
         n_py, med_py, mn_py, mx_py_, mean_py],
        [mx.label("GAMS local", "comp_nus333.gms, PATH via GAMS 53"),
         n_g, med_g, mn_g, mx_g, mean_g],
    ]
    parts.append(mx.raw(mx.tablecard(
        ["Solver", "N", "Median", "Min", "Max", "Mean"], body)))
    if py and gams:
        ratio = statistics.median(py) / statistics.median(gams)
        parts.append("")
        parts.append(mx.raw(mx.note(
            f"Median ratio Python / GAMS-local: <b>{ratio:.3f}×</b>")))
    parts.append("")
    return parts
```

(`mx_py_` con guión bajo para no sombrear el import `mx`.)

- [ ] **Step 2: Build and commit the regenerated page**

```bash
uv run sphinx-build -W -b html docs/site docs/site/_build/html
git status --short docs/site/guide/benchmarks.md
```

Expected: build exits 0 and `benchmarks.md` shows as modified (regenerated with cards).

- [ ] **Step 3: Commit**

```bash
git add docs/site/_scripts/render_benchmarks.py docs/site/guide/benchmarks.md
git commit -m "docs: benchmarks page → artifact visual language (cards/chips via matrix_html)"
```

---

### Task 6: Site restructure — 6 captioned toctrees

**Files:**
- Modify: `docs/site/index.md`
- Modify: `docs/site/guide/index.md`

**Interfaces:**
- Produces: the 6-section sidebar. Tasks 7–10 each add one toctree line to `index.md` (exact lines given in those tasks).

- [ ] **Step 1: Rewrite the toctrees in `docs/site/index.md`**

Keep the title/rubric/cards block (lines 1–44) as is, but change the User-guide card link from `guide/index` → stays `guide/index` (it remains the guide map). Replace the three `{toctree}` blocks at the bottom with:

````markdown
```{toctree}
:hidden:
:caption: Start here

guide/installation
```

```{toctree}
:hidden:
:caption: Templates

guide/gtap_quickstart
guide/pep_quickstart
guide/welfare_decomposition
```

```{toctree}
:hidden:
:caption: Data & solvers

guide/mip_to_sam
guide/har_io
guide/path_capi
```

```{toctree}
:hidden:
:caption: Validation & parity

guide/benchmarks
guide/gtap7_coverage_matrix
guide/pep_coverage_matrix
```

```{toctree}
:hidden:
:caption: Examples

gallery/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
```
````

- [ ] **Step 2: Turn `docs/site/guide/index.md` into an orphan map page**

Full new content:

```markdown
---
orphan: true
---

# User guide map

The documentation is organized in five areas — this page is the map.

**Start here** — [Installation](installation.md), then the architecture
overview for how the framework fits together.

**Templates** — the templates overview lists every model template and its
parity status; then per-template guides:
[GTAP quickstart](gtap_quickstart.md), [PEP quickstart](pep_quickstart.md),
[welfare decomposition](welfare_decomposition.md).

**Data & solvers** — [MIP → SAM](mip_to_sam.md) balancing,
[HAR I/O](har_io.md), and the [PATH C API](path_capi.md) solver wrapper.

**Validation & parity** — [benchmarks](benchmarks.md) and the coverage
matrices: [GTAP 7](gtap7_coverage_matrix.md) ·
[PEP-1-1](pep_coverage_matrix.md).

**Reference** — the [API reference](../api/index.md) and the changelog.

The {doc}`example gallery <../gallery/index>` complements these chapters
with short, runnable scripts re-executed on every documentation build.
```

**Nota:** "the architecture overview", "the templates overview" y "the changelog" van deliberadamente SIN link en este task — esas páginas se crean en Tasks 7–9 y un link ahora sería un broken-link warning bajo `-W`. El task que crea cada página convierte su mención en link: Task 7 → `[architecture overview](architecture.md)`, Task 8 → `[templates overview](templates.md)`, Task 9 → `[changelog](../changelog.md)`.

- [ ] **Step 3: Build + canary**

```bash
uv run sphinx-build -W -b html docs/site docs/site/_build/html
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q
```

Expected: both pass; the sidebar shows the captions.

- [ ] **Step 4: Commit**

```bash
git add docs/site/index.md docs/site/guide/index.md
git commit -m "docs: restructure site sidebar into 6 captioned sections"
```

---

### Task 7: Architecture overview page

**Files:**
- Create: `docs/site/guide/architecture.md`
- Modify: `docs/site/index.md` (add toctree line), `docs/site/guide/index.md` (turn the plain-text mention into a link)

**Interfaces:**
- Consumes: source material in `docs/architecture/gams_parity_matrix.md`, `src/equilibria/` package layout, `CLAUDE.md` parity philosophy.

- [ ] **Step 1: Write `docs/site/guide/architecture.md`**

Full content (the SVG is inline so the page is self-contained):

````markdown
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

1. **Data I/O** (`equilibria.sam_tools`, `equilibria.backends`) reads and
   balances the input data: SAM matrices, GEMPACK HAR files, GAMS GDX
   containers, and raw MIP/IO tables ({doc}`MIP → SAM <mip_to_sam>`,
   {doc}`HAR I/O <har_io>`).
2. **Calibration** (`equilibria.calibration`, per-template calibration
   modules) fits the model's parameters so the benchmark equilibrium
   reproduces the SAM exactly.
3. **Templates** (`equilibria.templates`) are complete model
   implementations — equations, sets, parameters, closure — built on the
   generic building blocks in `equilibria.blocks` (production, trade,
   demand, institutions, equilibrium).
4. **Solvers** (`equilibria.solver`): the same model solves as an **MCP**
   (complementarity, PATH via the C API — {doc}`path_capi`) or as an
   **NLP** (IPOPT), mirroring GAMS's `ifMCP` switch.
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
| `equilibria.sam_tools` | SAM audit, balancing, aggregation |
| `equilibria.backends` | File formats: HAR, GDX readers/writers |
| `equilibria.calibration` | Generic calibration utilities |
| `equilibria.blocks` | Generic equation blocks: production, trade, demand, institutions, equilibrium |
| `equilibria.templates.gtap` | GTAP Standard 7 (multi-period, MCP + NLP, 6 datasets) |
| `equilibria.templates.pep_pyomo` | PEP-1-1 v2.1 ported to Pyomo (NLP + MCP) |
| `equilibria.templates` (simple_open) | Didactic open-economy model |
| `equilibria.solver` | Solver drivers (PATH C API, IPOPT) |
| `equilibria.model`, `equilibria.core` | Model assembly core |
| `equilibria.contracts`, `equilibria.qa` | Validation contracts and QA checks |

See the {doc}`templates overview <templates>` for per-template status and
the {doc}`API reference <../api/index>` for the generated reference.
````

**Nota:** el link `{doc}`templates overview <templates>`` apunta a la página del Task 8 — dejalo como texto plano en este task ("See the templates overview (next page)") y convertilo en `{doc}` link en el Task 8, para que `-W` pase. Verificá también que cada entrada del "Package map" exista (`ls src/equilibria/`) y ajustá la tabla a lo que realmente hay.

- [ ] **Step 2: Wire it into the toctree and the map page**

In `docs/site/index.md`, "Start here" toctree becomes:

```markdown
guide/installation
guide/architecture
```

In `docs/site/guide/index.md`, replace the plain-text architecture mention with the link `[architecture overview](architecture.md)`.

- [ ] **Step 3: Build + canary, then commit**

```bash
uv run sphinx-build -W -b html docs/site docs/site/_build/html
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q
git add docs/site/guide/architecture.md docs/site/index.md docs/site/guide/index.md
git commit -m "docs: architecture overview page (pipeline, parity philosophy, package map)"
```

---

### Task 8: Templates overview page

**Files:**
- Create: `docs/site/guide/templates.md`
- Modify: `docs/site/index.md` (toctree), `docs/site/guide/index.md` (link), `docs/site/guide/architecture.md` (turn the plain-text mention into `{doc}` link)

- [ ] **Step 1: Write `docs/site/guide/templates.md`**

Full content:

````markdown
# Templates overview

A *template* is a complete, validated CGE model implementation: sets,
parameters, equations, calibration, and closure, ready to load a dataset
and solve. Status below is the parity status against the reference GAMS
implementation.

| Template | Model | Solve forms | Parity status |
|---|---|---|---|
| `equilibria.templates.gtap` | GTAP Standard 7 | MCP (PATH) · NLP (IPOPT), single- and multi-period | See the [GTAP 7 coverage matrix](gtap7_coverage_matrix.md) |
| `equilibria.templates.pep_pyomo` | PEP-1-1 v2.1 | NLP (IPOPT) · MCP (PATH) | 100% vs GAMS — see the [PEP coverage matrix](pep_coverage_matrix.md) |
| `equilibria.templates.simple_open` | Didactic open economy | NLP | GAMS-parity contract (`simple_open_contract.py`) |
| `equilibria.templates` (legacy PEP) | PEP-1-1 (cyipopt) | NLP | Superseded by `pep_pyomo` |

## GTAP Standard 7 (`gtap`)

The flagship template: GTAP Standard 7 with the altertax variant,
single-period and multi-period (base → check → shock), both `ifSUB` modes,
solved as MCP (PATH) or NLP (IPOPT) across 6 datasets (3×3 … 20×41).
Start with the {doc}`GTAP quickstart <gtap_quickstart>`; welfare analysis
in {doc}`welfare_decomposition`.

## PEP-1-1 v2.1 (`pep_pyomo`)

The PEP-1-1 v2.1 single-country CGE ported from GAMS to Pyomo — six
modules (`pep_pyomo_sets`, `pep_pyomo_parameters`, `pep_pyomo_equations`,
`pep_pyomo_blocks`, `pep_pyomo_scenarios`, `pep_pyomo_solver`). Both the
NLP form (vs GAMS CNS) and the MCP form (vs GAMS-native MCP) reproduce the
GAMS reference at 100% cell parity, including the SIM1 export-tax
counterfactual. Start with the {doc}`PEP quickstart <pep_quickstart>`.

The original cyipopt-based PEP template (`pep_*` modules under
`equilibria.templates`) remains for reference but is superseded by
`pep_pyomo`.

## simple_open (`simple_open`)

A small open-economy model used to exercise the framework end-to-end and
as a didactic entry point — one SAM, a handful of sectors, an explicit
GAMS-parity contract (`simple_open_contract.py`,
`simple_open_parity_pipeline.py`). It has no dedicated quickstart; its
parity pipeline doubles as the usage example.
````

Verificá los claims contra el código antes de commitear: `ls src/equilibria/templates/pep_pyomo/` (los 6 módulos), `ls src/equilibria/templates/ | grep simple_open`. Ajustá nombres si difieren.

- [ ] **Step 2: Wire links**

- `docs/site/index.md` — "Templates" toctree becomes:

```markdown
guide/templates
guide/gtap_quickstart
guide/pep_quickstart
guide/welfare_decomposition
```

- `docs/site/guide/index.md`: replace the plain-text templates mention with `[templates overview](templates.md)`.
- `docs/site/guide/architecture.md`: replace "See the templates overview (next page)" with ``See the {doc}`templates overview <templates>` ``.

- [ ] **Step 3: Build + canary, then commit**

```bash
uv run sphinx-build -W -b html docs/site docs/site/_build/html
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q
git add docs/site/guide/templates.md docs/site/index.md docs/site/guide/index.md docs/site/guide/architecture.md
git commit -m "docs: templates overview page (gtap, pep_pyomo, simple_open, legacy pep)"
```

---

### Task 9: CHANGELOG.md + site changelog page

**Files:**
- Create: `CHANGELOG.md` (repo root)
- Create: `docs/site/changelog.md`
- Modify: `docs/site/index.md` (Reference toctree), `docs/site/guide/index.md` (link)

- [ ] **Step 1: Date the 0.5.1 section**

```bash
git log -1 --format="%ad" --date=short dd68304   # merge date of PR #27 → date for [0.5.1]
```

Use that date where the seed below says `2026-07-XX`. Tag dates already
verified: v0.2.0 = 2026-03-08, v0.3.0 = 2026-05-12, v0.4.0 = 2026-05-20,
v0.5.0 = 2026-05-20 (same-day release pair).

- [ ] **Step 2: Write `CHANGELOG.md`**

Full content (bullets already verified against the git ranges on 2026-07-19;
only fill the `2026-07-XX` date from Step 1):

```markdown
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

## [0.5.1] — 2026-07-XX

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
```

- [ ] **Step 3: Write `docs/site/changelog.md`**

````markdown
# Changelog

```{include} ../../CHANGELOG.md
:start-line: 1
```
````

(`start-line: 1` salta el H1 duplicado del archivo raíz para no tener dos H1 en la página; verificá con el build que el título renderiza una sola vez.)

- [ ] **Step 4: Wire toctree + map**

- `docs/site/index.md` — "Reference" toctree becomes:

```markdown
api/index
changelog
```

- `docs/site/guide/index.md`: replace the plain-text changelog mention with `[changelog](../changelog.md)`.

- [ ] **Step 5: Build + canary, then commit**

```bash
uv run sphinx-build -W -b html docs/site docs/site/_build/html
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q
git add CHANGELOG.md docs/site/changelog.md docs/site/index.md docs/site/guide/index.md
git commit -m "docs: CHANGELOG.md (Keep-a-Changelog, curated 0.x milestones) + site page"
```

---

### Task 10: Expanded API reference

**Files:**
- Create: `docs/site/api/core.md`, `docs/site/api/blocks.md`, `docs/site/api/templates_gtap.md`, `docs/site/api/templates_pep_pyomo.md`
- Modify: `docs/site/api/index.md`, `docs/site/conf.py` (mock imports if needed), `pyproject.toml` (docs extra += pyomo, only if needed)

- [ ] **Step 1: Discover the documentable modules**

```bash
uv run python - <<'EOF'
import pkgutil
import equilibria, equilibria.blocks, equilibria.solver, equilibria.calibration
for pkg in (equilibria, equilibria.blocks, equilibria.solver, equilibria.calibration):
    print(pkg.__name__, [m.name for m in pkgutil.iter_modules(pkg.__path__)])
EOF
```

Use the output to adjust the module lists below (drop what doesn't exist, add close siblings that obviously belong).

- [ ] **Step 2: Write the four pages**

`docs/site/api/core.md`:

````markdown
# Core

Model assembly, datasets, calibration and solver drivers.

```{eval-rst}
.. automodule:: equilibria.model
.. automodule:: equilibria.datasets
.. automodule:: equilibria.calibration
.. automodule:: equilibria.solver
.. automodule:: equilibria.contracts
```
````

`docs/site/api/blocks.md`:

````markdown
# Equation blocks

Generic CGE building blocks composed by the templates.

```{eval-rst}
.. automodule:: equilibria.blocks.base
.. automodule:: equilibria.blocks.production
.. automodule:: equilibria.blocks.trade
.. automodule:: equilibria.blocks.demand
.. automodule:: equilibria.blocks.institutions
.. automodule:: equilibria.blocks.equilibrium
```
````

`docs/site/api/templates_gtap.md`:

````markdown
# GTAP template

Public surface of `equilibria.templates.gtap`. The equation monolith
(`gtap_model_equations`, ~6k lines) is deliberately not auto-documented —
its extraction into `equilibria.blocks` is roadmap phase F3.

```{eval-rst}
.. automodule:: equilibria.templates.gtap.gtap_contract
.. automodule:: equilibria.templates.gtap.gtap_parameters
.. automodule:: equilibria.templates.gtap.gtap_sets
.. automodule:: equilibria.templates.gtap.gtap_solver
.. automodule:: equilibria.templates.gtap.gtap_multiperiod_driver
.. automodule:: equilibria.templates.gtap.shocks
.. automodule:: equilibria.templates.gtap.welfare_decomp
```
````

`docs/site/api/templates_pep_pyomo.md`:

````markdown
# PEP template (Pyomo)

The PEP-1-1 v2.1 port — see the
[templates overview](../guide/templates.md) for context.

```{eval-rst}
.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_sets
.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_parameters
.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_equations
.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_blocks
.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_scenarios
.. automodule:: equilibria.templates.pep_pyomo.pep_pyomo_solver
```
````

`docs/site/api/index.md`:

````markdown
# API reference

```{toctree}
:maxdepth: 2

core
blocks
sam_tools
templates_gtap
templates_pep_pyomo
```
````

- [ ] **Step 3: Build; resolve import errors WITHOUT touching src/**

```bash
uv run sphinx-build -W -b html docs/site docs/site/_build/html 2>&1 | grep -i "warning\|error" | head -20
```

Resolution ladder, in order:
1. Module needs `pyomo` → add it to the docs extra in `pyproject.toml` (`"pyomo>=6.7",` inside `docs = [...]`) and `uv sync --extra docs`. RTD picks it up automatically (installs `.[docs]`).
2. Module needs a compiled/optional dep (`cyipopt`, `pyoptinterface`, PATH libs) → add to `conf.py`: `autodoc_mock_imports = ["cyipopt", "pyoptinterface"]` (extend the list with exactly what fails).
3. Module raises at import for another reason → remove that single `automodule` line and record it in the debt note (goes to the PR body).
4. A *docstring* produces a Sphinx warning → do NOT edit src; add the module to the debt note and drop its automodule line only if the warning is unavoidable.

Repeat build → fix → build until `-W` exits 0.

- [ ] **Step 4: Canary + commit**

```bash
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q
git add docs/site/api/ docs/site/conf.py pyproject.toml uv.lock
git commit -m "docs: expand API reference (core, blocks, gtap public surface, pep_pyomo)"
```

(If `pyproject.toml`/`uv.lock` didn't change, drop them from the add.)

---

### Task 11: Quickstart review (gtap + pep)

**Files:**
- Modify (text only, as needed): `docs/site/guide/gtap_quickstart.md`, `docs/site/guide/pep_quickstart.md`

- [ ] **Step 1: Extract and verify every command/flag/path in both pages**

```bash
grep -n '^\s*\(uv \|python \|make \|\$ \)' docs/site/guide/gtap_quickstart.md docs/site/guide/pep_quickstart.md
grep -n 'scripts/\|EQUILIBRIA_\|--' docs/site/guide/gtap_quickstart.md docs/site/guide/pep_quickstart.md
```

For each command found, verify it against reality:

```bash
uv run python scripts/gtap/run_gtap.py --help            # flags/subcommands cited exist?
ls scripts/gtap/ scripts/pep/                            # cited script paths exist?
grep -rn "EQUILIBRIA_GTAP_SOLVE_NLP" src/ scripts/ | head -3   # cited env vars exist?
uv run python -c "import equilibria; print(equilibria.__version__)"  # cited version current?
```

Also verify dataset names cited in the pages exist in `equilibria.datasets` (`uv run python -c "import equilibria.datasets as d; print([x for x in dir(d) if not x.startswith('_')])"`).

- [ ] **Step 2: Fix drift (text-only) and record anything deeper as debt**

Rules: renamed flag → update the page; removed subcommand → replace with the current equivalent; a claim that is no longer true (e.g., a parity number) → update from the coverage matrix; anything that would need a code change to make the docs true → leave the docs describing reality, add to debt note.

- [ ] **Step 3: Build + canary, then commit**

```bash
uv run sphinx-build -W -b html docs/site docs/site/_build/html
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q
git add docs/site/guide/gtap_quickstart.md docs/site/guide/pep_quickstart.md
git commit -m "docs: verify quickstarts against current CLIs, fix drift"
```

(If nothing needed fixing, skip the commit and note "quickstarts verified, no drift" for the PR body.)

---

### Task 12: `docs-build` CI job

**Files:**
- Modify: `.github/workflows/tests.yml`

- [ ] **Step 1: Append the job**

Add at the end of `tests.yml`, aligned with the existing jobs (same indentation level as `tests:`):

```yaml
  docs-build:
    name: Docs build (sphinx -W)
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup uv
        uses: astral-sh/setup-uv@v3

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install docs dependencies
        run: uv sync --extra docs

      - name: Build site (warnings are errors)
        run: uv run sphinx-build -W -b html docs/site docs/site/_build/html
```

Copiá el patrón exacto de setup (uv/python versions) del job `tests:` existente — si ese job pinnea otra versión de Python o pasa flags extra a `uv sync`, replicalos.

- [ ] **Step 2: Validate the YAML + run the same commands locally**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))"
uv sync --extra docs && uv run sphinx-build -W -b html docs/site docs/site/_build/html
```

Expected: YAML parses; build exits 0.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "ci: docs-build job — sphinx-build -W on docs/site"
```

---

### Task 13: Final gates, rebase, PR

- [ ] **Step 1: Full local gate sweep**

```bash
uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -q
uv run pytest tests/templates/gtap/test_coverage_matrix.py tests/templates/pep_pyomo/test_pep_coverage_matrix.py -q
uv run sphinx-build -W -b html docs/site docs/site/_build/html
```

Expected: all PASS / exit 0.

- [ ] **Step 2: Rebase onto latest origin/main**

`main` may have advanced (e.g., the 15×10 altertax close changes floors in `coverage_matrix.py`). After rebasing, the generated matrix pages MUST be regenerated from the new ROWS:

```bash
git fetch origin
git rebase origin/main
uv run python scripts/gtap/gen_coverage_doc.py
uv run python scripts/pep/gen_pep_coverage_doc.py
git status --short docs/site/guide/
```

If the regeneration changed the pages: re-check that any prose notes in `gen_coverage_doc.py` still match the ROWS (e.g., a floor that used to be 94 and is now 99 → fix the note text in the generator, regenerate again), then:

```bash
uv run pytest tests/templates/gtap/test_coverage_matrix.py tests/templates/pep_pyomo/test_pep_coverage_matrix.py -q
git add scripts/gtap/gen_coverage_doc.py docs/site/guide/gtap7_coverage_matrix.md docs/site/guide/pep_coverage_matrix.md
git commit -m "docs: regenerate coverage matrices against rebased ROWS"
```

Re-run the full Step 1 sweep after the rebase.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin fdocs
gh pr create --base main --title "F-docs: site restructure, artifact-format parity matrices, API reference, CHANGELOG" --body "$(cat <<'EOF'
## F-docs (roadmap §3.5) — docs-only

Spec: `docs/superpowers/specs/2026-07-18-fdocs-design.md`
Plan: `docs/superpowers/plans/2026-07-19-fdocs-implementation.md`

- Site restructured into 6 captioned sections (Start here / Templates / Data & solvers / Validation & parity / Examples / Reference)
- New pages: architecture overview, templates overview, changelog
- GTAP + PEP coverage matrices rendered in the artifact card format under **NLP vs NLP** / **MCP vs MCP** headers; `.nl` table and legacy SOLVE tables removed from the doc (their pytest gates are untouched)
- Benchmarks page restyled with the same shared `matrix.css` / `matrix_html.py`
- API reference expanded: core, blocks, gtap public surface, pep_pyomo
- `CHANGELOG.md` created (Keep-a-Changelog, curated 0.x milestones) — `pyproject` already linked to it
- CI: new `docs-build` job (`sphinx-build -W`)

### Gates
- `sphinx-build -W`: clean
- `test_coverage_doc_in_sync` + `test_pep_coverage_doc_in_sync`: green
- `.nl` canary (`test_gtap7_nl_parity.py`): green
- `src/equilibria/`: untouched

### Deudas anotadas (no tocadas, fuera de F-docs)
<!-- pegar aquí la debt note acumulada; "ninguna" si vacía -->

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace the debt-note comment with the accumulated debt items (or "ninguna") before submitting.

---

## Self-review notes

- **Spec coverage:** §1 estructura → Task 6; §2 architecture → Task 7; §3 templates → Task 8; §4 matrices+benchmarks → Tasks 2–5; §5 API → Task 10; §6 changelog → Task 9; §7 quickstarts → Task 11; §8 gate/CI → Tasks 1+12; criterio de terminado → Task 13.
- **Broken-link ordering:** Tasks 6–9 create pages that reference each other; each task keeps `-W` green by linking only to pages that already exist (plain text until the target page's task converts it).
- **`benchmarks.md` is build-regenerated:** conf.py overwrites it on every build — Task 5 commits the regenerated copy so the repo matches.
- **The ROWS stay authoritative:** generators never hardcode floors; after the Task 13 rebase the pages are regenerated from the (possibly advanced) ROWS.
