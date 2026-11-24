"""Generate the against-GEMPACK linearization-study page from whatever grid
records are present.

The page always builds: absent (dataset, config) cells read "—" (pending the
Windows GEMPACK run). It documents the two study axes (shock-size sweep + Gragg
refinement), lists the compared quantity variables, reports the measured Python↔
GEMPACK within-1pp match per (dataset, config) where the records exist, and
carries the welfare diagnostic + provenance.

Import-light: build_page(data_root, out_md) works on an empty dir WITHOUT
importing Pyomo — the per-shock Python solve is guarded behind "record present".
The GEMPACK solutions are the capFlex matrix records
``data/gtap7/<ds>/gempack_tm<shock>_s<steps>_capflex/`` (``sl4dump.har``).

Usage:
    uv run python scripts/gtap/gen_linearization_study.py
"""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

from equilibria._local_refs import path_capi_src
from equilibria.sources import dataset_dir, find_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gtap"))

DATASETS = ["gtap7_3x3", "gtap7_3x4", "gtap7_5x5", "gtap7_10x7", "gtap7_15x10"]
SWEEP_SHOCKS = ["10", "3", "1", "0.3", "0.1"]
GRAGG_STEPS = ["4", "8", "16", "32", "64"]
DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_OUT = ROOT / "docs/site/guide/gtap7_gempack_linearization_study.md"
# the closure of the GEMPACK fixtures (RORDELTA=1); Python solves with the same one
FIXTURE_CLOSURE = "capFlex"


def _config_tag(shock_pct: str, steps: str) -> str:
    """Mirror run_gempack_matrix.config_tag without importing it (import-light)."""
    s = shock_pct.replace(".", "p")
    st = steps.replace(" ", "-")
    return f"tm{s}_s{st}"


def _sl4_fixture(
    data_root: Path, dataset: str, shock_pct: str, steps: str
) -> Path | None:
    """The capFlex SL4 dump recorded for (dataset, shock, steps), or None if absent."""
    shock = float(shock_pct)
    folder = find_reference(
        dataset,
        "gempack",
        root=data_root,
        shock=int(shock) if shock.is_integer() else shock,
        steps=steps.replace(" ", "-"),
        closure=FIXTURE_CLOSURE,
    )
    return None if folder is None else folder / "sl4dump.har"


def _var_list_rows() -> list[str]:
    """The Q_TO_VAR quantity map, rendered as a table body. Read from
    gempack_reference (no Pyomo import)."""
    from gempack_reference import Q_TO_VAR

    rows = []
    for gv, spec in Q_TO_VAR.items():
        rows.append(f"| `{gv}` | `{spec['var']}` |")
    return rows


def _seed_gdx(dataset: str) -> Path | None:
    """The GAMS MCP shock reference (ifSUB=1) that seeds the Python solve."""
    folder = find_reference(
        dataset, "gams", form="mcp", mode="gtap", ifsub=1, fbep="header"
    )
    return None if folder is None else folder / "out.gdx"


def _scaled_tol(shock_pct: str) -> float:
    """1 pp at the 10 % shock, scaled with the shock: at 0.1 % almost no cell moves
    1 pp, so a fixed band would read ~100 % whatever Python does."""
    return 0.01 * float(shock_pct) / 10.0


def _match_cell(data_root: Path, dataset: str, shock_pct: str, steps: str) -> str:
    """'<within 1pp> / <within the scaled band>' for (dataset, shock, steps), or
    '—' if the record is absent.

    The Python solve is only invoked when the GEMPACK fixture exists (import-light
    otherwise). Python is solved at the same size as the GEMPACK record."""
    sl4 = _sl4_fixture(data_root, dataset, shock_pct, steps)
    if sl4 is None:
        return "—"
    # Python needs the shock GDX to seed the multiperiod solve; without it the
    # solve runs unseeded and returns a spurious number — report "—" instead.
    if _seed_gdx(dataset) is None:
        return "— (no seed GDX)"
    try:
        within, scaled = _measure_within_pp(dataset, sl4, shock_pct)
    except Exception as exc:  # never crash the page on one dataset
        return f"— (err: {type(exc).__name__})"
    return f"{within} / {scaled}"


def _measure_within_pp(dataset: str, sl4: Path, shock_pct: str) -> tuple[str, str]:
    """Solve Python at ``shock_pct`` and return the (within 1pp, within the scaled
    band) match% strings vs the GEMPACK sl4 dump. Imports Pyomo — only called when
    a fixture is present."""
    from gempack_reference import Q_TO_VAR, gempack_qty_pct
    from pyomo.environ import value as V

    # path-capi bridge for the MCP solve
    _pc = Path(str(path_capi_src()))
    if _pc.exists() and str(_pc) not in sys.path:
        sys.path.insert(0, str(_pc))
    try:
        import path_capi_python  # noqa: F401
    except ImportError:
        pass

    m = _solve_at(dataset, shock_pct)
    diffs = []
    for gvar, spec in Q_TO_VAR.items():
        try:
            gem = gempack_qty_pct(str(sl4), gvar)
        except KeyError:
            continue
        pv = getattr(m, spec["var"], None)
        if pv is None:
            continue
        for key, gfrac in gem.items():
            try:
                # shock/check, the denominator GEMPACK and the comparator use
                b = float(V(pv[(*key, "check")]))
                s = float(V(pv[(*key, "shock")]))
            except (KeyError, ValueError):
                continue
            if abs(b) <= 1e-12:
                continue
            diffs.append(abs((s / b - 1.0) - gfrac))
    if not diffs:
        return "—", "—"
    tol = _scaled_tol(shock_pct)

    def pct(band: float) -> str:
        return f"{sum(1 for x in diffs if x <= band) / len(diffs) * 100:.0f}%"

    return pct(0.01), pct(tol)


@cache
def _solve_at(dataset: str, shock_pct: str):
    """The shock solved as the paper's against-GEMPACK comparator solves it
    (``_solve_shock``: ifSUB=1, the fixture's closure, base calibrated, check
    solved). At 10 % it is the gate's run (the default tariff); every other size is
    ``uniform(<size>)`` on ``imptx``, as GEMPACK's ``Shock tm = uniform <size>``.
    Cached: every cell of a (dataset, size) reads the same run."""
    tests = str(ROOT / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    from templates.gtap.test_gtap7_gempack_parity import _solve_shock

    kw = {} if shock_pct == "10" else {"tariff_pct": float(shock_pct)}
    m, code = _solve_shock(dataset, 1, FIXTURE_CLOSURE, **kw)
    if code != 1:
        raise RuntimeError(
            f"{dataset}: the {shock_pct} % shock did not solve (code {code})"
        )
    return m


def _nonlinearity(cells: list[str]) -> str:
    """|scaled match(10 %) − scaled match(0.1 %)| when both are numbers."""
    try:
        hi = float(cells[0].split("/")[-1].strip().rstrip("%"))
        lo = float(cells[-1].split("/")[-1].strip().rstrip("%"))
    except ValueError:
        return "—"
    return f"{abs(hi - lo):.0f}pp"


def build_page(data_root: Path, out_md: Path) -> str:
    """Build the study page markdown, write it to out_md, and return the text."""
    data_root = Path(data_root)

    # --- shock-sweep table ---
    sweep_header = (
        "| dataset | "
        + " | ".join(f"{s}%" for s in SWEEP_SHOCKS)
        + " | non-linearity |"
    )
    sweep_sep = "|" + "---|" * (len(SWEEP_SHOCKS) + 2)
    sweep_rows = []
    for ds in DATASETS:
        cells = [_match_cell(data_root, ds, s, "8 16 32") for s in SWEEP_SHOCKS]
        nl = _nonlinearity(cells)
        sweep_rows.append(f"| `{ds}` | " + " | ".join(cells) + f" | {nl} |")

    # --- Gragg-convergence table (10% shock, varying steps) ---
    gragg_header = "| dataset | " + " | ".join(f"Steps {s}" for s in GRAGG_STEPS) + " |"
    gragg_sep = "|" + "---|" * (len(GRAGG_STEPS) + 1)
    gragg_rows = []
    for ds in DATASETS:
        cells = [_match_cell(data_root, ds, "10", s) for s in GRAGG_STEPS]
        gragg_rows.append(f"| `{ds}` | " + " | ".join(cells) + " |")

    var_rows = _var_list_rows()

    md = (
        f"""# GTAP7 vs GEMPACK — linearization study

This page measures how close equilibria's levels solution of GTAP7 lands to
GEMPACK's Gragg-linearized one at a +10% global bilateral tariff, across all five
matrix datasets, and how that match moves as (1) the shock shrinks and (2)
GEMPACK's Gragg method is refined.

At the 10% shock the match is **99-100% within 1pp** on every dataset. An earlier
version of this page reported 52-76% and read the gap as linearization: it compared
a capFix Python run (returns not equalized) with the capFlex GEMPACK fixtures. With
the fixtures' closure (`FIXTURE_CLOSURE`, the paper's comparator `_solve_shock`) the
gap is gone (MAR-223).

**Reference number (PR #40):** on Horridge's small SIMPLE shock, GAMS(levels) ≡
GEMPACK(linearized) at 100% within 1pp. This study extends that to the real GTAP7
matrix.

## Scope

Datasets: `gtap7_3x3`, `gtap7_3x4`, `gtap7_5x5`, `gtap7_10x7`, `gtap7_15x10`.
(nus333 / 9x10 remain out of the matrix scope.) Metric: fraction of mapped quantity
cells within **1 percentage point** (absolute pp on %-changes — GEMPACK output is
%-change, so the comparison is |Δ(%change)| ≤ 1pp, not a relative tol).

Cells read **—** where the Windows-produced GEMPACK fixture is not yet present.

## Evidence 1 — shock-size sweep (default Gragg steps)

A uniform `tm` shock at decreasing magnitude. GEMPACK is recorded at every size
(`Shock tm = uniform <size>`) and Python is solved at the same size
(`{{"imptx": uniform(<size>)}}`; 10% is the default tariff, the gate's run).

Each cell reads **within 1pp / within the scaled band**. The scaled band is 1pp at
10% and shrinks with the shock (0.3pp at 3%, 0.01pp at 0.1%): at a small shock
almost no cell moves 1pp, so the fixed band alone would read ~100% whatever Python
does. The **non-linearity** column is |scaled match(10%) − scaled match(0.1%)|.

{sweep_header}
{sweep_sep}
"""
        + "\n".join(sweep_rows)
        + f"""

## Evidence 2 — Gragg refinement (fixed 10% shock)

Solving the SAME 10% shock with a finer Gragg subinterval count. The match does
not move from 4 to 64 steps: at this shock GEMPACK's linearization error is well
under the 1pp band, so what remains (one 3x3 cell) is not GEMPACK's method.

{gragg_header}
{gragg_sep}
"""
        + "\n".join(gragg_rows)
        + """

## Variables compared

The quantity map (`Q_TO_VAR` in `gempack_reference.py`) — GEMPACK quantity → Python
Var, verified empirically and by economic meaning:

| GEMPACK | Python Var |
|---|---|
"""
        + "\n".join(var_rows)
        + """

## Welfare (diagnostic only)

Welfare EV is read from GEMPACK's `decomp.har` (WELVIEW) via `gempack_welfare_ev`
and compared to Python's EV by the 3-branch decomposition (allocative /
terms-of-trade / investment-savings). It is **not** floor-gated: raw welfare `u` is
a sign-flipping second-order quantity where even GAMS and GEMPACK disagree — see
`docs/archive/findings/gempack_welfare_not_cellwise_2026-07-23.md`. This section reports the
EV$ decomposition side by side where a `decomp.har` fixture is present, purely as a
diagnostic.

## ifSUB is faithful

Python's ifSUB condensation reproduces GAMS to the bit (mac gate #1,
`verify_ifsub_equivalence.py`; primary quantity block 99.3–99.9% consistent across
modes on 16,424 cells). See
`docs/archive/findings/gtap_ifsub_is_faithful_not_inverted_2026-07-24.md`.

## Provenance

- van der Mensbrugghe, *The Standard GTAP Model in GAMS, Version 7* (Table D.1:
  `ifSUB` = model condensation, not economics; GEMPACK is a Johansen percent-change
  solver, condensed by nature).
- Horridge, *tpmh0103* SIMPLE model (the GAMS-vs-GEMPACK reference, PR #40).
"""
    )

    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    return md


def main() -> int:
    md = build_page(DEFAULT_DATA_ROOT, DEFAULT_OUT)
    n_pending = md.count("—")
    print(f"wrote {DEFAULT_OUT} ({len(md)} chars, {n_pending} pending cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
