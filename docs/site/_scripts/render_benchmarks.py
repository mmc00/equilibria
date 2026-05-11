"""Render guide/benchmarks.md from CSVs in _data/benchmarks/.

Invoked from conf.py at build time so Read the Docs only needs to read the
committed CSVs — it does not have GAMS/PATH and cannot run the diff scripts
itself. Regenerate the CSVs locally with `make benchmarks` and commit them.

Per dataset we render up to three blocks:

1. *Parity vs NEOS reference* — always (`<slug>.csv`).
2. *Parity vs GAMS local* — when `<slug>_local.csv` exists (currently only
   NUS333; 9x10 doesn't fit the GAMS community license).
3. *Wall-time benchmark* — when `<slug>_timing.csv` exists.
"""
from __future__ import annotations
import csv
import statistics
from pathlib import Path

DATASETS = [
    ("9x10", "GTAP Standard 7 — 9 sectors × 10 regions",
     "Reference: `src/equilibria/templates/reference/gtap/output/COMP.gdx` "
     "(rate-scaled 10% imptx shock, `if_sub=False`, `rorflex=10`)."),
    ("nus333", "GTAP Standard 7 — NUS333 (3 sectors × 3 regions × 3 factors)",
     "Reference: `output/nus333_neos/out.gdx` (NEOS job 18744693, "
     "power-scaled 10% imptx shock, residual region `ROW`)."),
]

LOCAL_BLOCKER_NOTE = {
    "9x10": (
        "> ℹ️ **GAMS-local parity not available for 9x10.** The model has "
        "~10k equations and exceeds the GAMS community-license limit of "
        "2500 rows/cols for nonlinear models. Only the NEOS reference run "
        "is used for 9x10."
    ),
}


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _summary_row(rows: list[dict], phase: str) -> dict | None:
    for r in rows:
        if r["phase"] == phase and r["var"] == "__SUMMARY__":
            return r
    return None


def _top_diverging(rows: list[dict], phase: str, n: int = 10) -> list[dict]:
    candidates = [r for r in rows
                  if r["phase"] == phase and r["var"] != "__SUMMARY__"
                  and int(r["diverge"]) > 0]
    candidates.sort(key=lambda r: float(r["max_abs_err"] or 0.0), reverse=True)
    return candidates[:n]


def _fmt_pct(num: int, den: int) -> str:
    if not den:
        return "—"
    return f"{(num / den) * 100:.2f}%"


def _parity_block(rows: list[dict], heading: str) -> list[str]:
    """Render a parity Summary table + Top-diverging sections for `rows`."""
    parts = [f"### {heading}\n"]
    parts.append("| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual | Solve time |")
    parts.append("|-------|--------------|-------|-------|---------|---------|------------|----------|------------|")
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
        parts.append(
            f"| `{phase}` | {n_vars}/{n_total_vars} | {cells} | {match} | "
            f"{diverge} | {missing} | {_fmt_pct(match, cells)} | "
            f"{float(s['residual']):.2e} | {secs_str} |"
        )
    parts.append("")

    for phase in ("base", "shock"):
        worst = _top_diverging(rows, phase, n=10)
        if not worst:
            continue
        parts.append(f"#### Top diverging variables — `{phase}`\n")
        parts.append("| Var | Py var | Cells | Diverge | Max abs err | Max rel err |")
        parts.append("|-----|--------|-------|---------|-------------|-------------|")
        for r in worst:
            parts.append(
                f"| `{r['var']}` | `{r['py_var']}` | {r['cells']} | "
                f"{r['diverge']} | {r['max_abs_err']} | {r['max_rel_err']} |"
            )
        parts.append("")
    return parts


def _timing_block(timing_csv: Path) -> list[str]:
    rows = _read_csv(timing_csv)
    if not rows:
        return []

    def _samples(col: str) -> list[float]:
        out = []
        for r in rows:
            v = r.get(col, "").strip()
            if not v:
                continue
            try:
                out.append(float(v))
            except ValueError:
                pass
        return out

    py = _samples("python_seconds")
    gams = _samples("gams_local_seconds")
    if not py and not gams:
        return []

    def _stat(s: list[float]) -> tuple[str, str, str, str, str]:
        if not s:
            return ("—",) * 5
        n = len(s)
        med = statistics.median(s)
        return (
            str(n),
            f"{med:.3f}s",
            f"{min(s):.3f}s",
            f"{max(s):.3f}s",
            f"{statistics.fmean(s):.3f}s",
        )

    parts = ["### Wall-time benchmark\n"]
    parts.append(
        "Median / min / max / mean across the runs in "
        f"`{timing_csv.name}`. The warm-up run is discarded — both sides solve "
        "from cold state then are re-run N times. Lower is better.\n"
    )
    parts.append("| Solver | N | Median | Min | Max | Mean |")
    parts.append("|--------|---|--------|-----|-----|------|")
    n_py, med_py, mn_py, mx_py, mean_py = _stat(py)
    n_g, med_g, mn_g, mx_g, mean_g = _stat(gams)
    parts.append(f"| Python `equilibria` (PATH C API, nonlinear full) | {n_py} | {med_py} | {mn_py} | {mx_py} | {mean_py} |")
    parts.append(f"| GAMS local (`comp_nus333.gms`, PATH via GAMS 53) | {n_g} | {med_g} | {mn_g} | {mx_g} | {mean_g} |")
    if py and gams:
        ratio = statistics.median(py) / statistics.median(gams)
        parts.append("")
        parts.append(f"*Median ratio Python / GAMS-local: **{ratio:.3f}×***")
    parts.append("")
    return parts


def _render_dataset(slug: str, title: str, blurb: str, data_dir: Path) -> str:
    parts = [f"## {title}\n", blurb, ""]
    neos_csv = data_dir / f"{slug}.csv"
    local_csv = data_dir / f"{slug}_local.csv"
    timing_csv = data_dir / f"{slug}_timing.csv"

    if not neos_csv.exists():
        try:
            shown = neos_csv.relative_to(neos_csv.parents[4])
        except ValueError:
            shown = neos_csv
        parts.append(
            f"⚠️ `{shown}` not found. "
            "Run `make benchmarks` and commit the result.\n"
        )
        return "\n".join(parts)

    rows = _read_csv(neos_csv)
    if not rows:
        parts.append("⚠️ Empty CSV.\n")
        return "\n".join(parts)
    meta = rows[0]
    parts.append(
        f"*Generated `{meta['generated_at']}` from commit `{meta['git_sha']}`.*\n"
    )

    parts.extend(_parity_block(rows, "Parity vs GAMS NEOS reference"))

    if local_csv.exists():
        local_rows = _read_csv(local_csv)
        if local_rows:
            parts.extend(_parity_block(local_rows, "Parity vs GAMS local"))
    elif slug in LOCAL_BLOCKER_NOTE:
        parts.append(LOCAL_BLOCKER_NOTE[slug])
        parts.append("")

    if timing_csv.exists():
        parts.extend(_timing_block(timing_csv))

    return "\n".join(parts)


HEADER = """# Benchmarks

Variable-by-variable parity between the Python `equilibria` GTAP
Standard 7 implementation and reference GAMS runs, plus wall-time benchmarks
when GAMS can run locally. Numbers come from CSVs committed under
`docs/site/_data/benchmarks/` — Read the Docs renders this page from those
files (it has no GAMS/PATH installed). Regenerate locally with:

```bash
make benchmarks           # all datasets
make benchmarks-nus333    # NUS333 only (also produces local parity + timing)
```

The default number of timing runs is `BENCH_RUNS=5` (override on the
make command line).

Each parity row reports, for one (dataset, phase, variable) triple, how
many Pyomo Var cells match GAMS within `tol_rel=1e-3 / tol_abs=1e-6`
and the worst absolute / relative error observed. The `__SUMMARY__` rows
in the underlying CSV hold per-phase totals.

"""


def render(out_path: Path, data_dir: Path) -> None:
    body = [HEADER]
    for slug, title, blurb in DATASETS:
        body.append(_render_dataset(slug, title, blurb, data_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    site = here.parent
    render(site / "guide" / "benchmarks.md", site / "_data" / "benchmarks")
    print(f"Wrote {site / 'guide' / 'benchmarks.md'}")
