"""Render guide/benchmarks.md from CSVs in _data/benchmarks/.

Invoked from conf.py at build time so Read the Docs only needs to read the
committed CSVs — it does not have GAMS/PATH and cannot run the diff scripts
itself. Regenerate the CSVs locally with `make benchmarks` and commit them.
"""
from __future__ import annotations
import csv
from pathlib import Path

DATASETS = [
    ("9x10", "GTAP Standard 7 — 9 sectors × 10 regions",
     "Reference: `src/equilibria/templates/reference/gtap/output/COMP.gdx` "
     "(rate-scaled 10% imptx shock, `if_sub=False`, `rorflex=10`)."),
    ("nus333", "GTAP Standard 7 — NUS333 (3 sectors × 3 regions × 3 factors)",
     "Reference: `output/nus333_neos/out.gdx` (NEOS job 18744693, "
     "power-scaled 10% imptx shock, residual region `ROW`)."),
]


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


def _render_dataset(slug: str, title: str, blurb: str, csv_path: Path) -> str:
    parts = [f"## {title}\n", blurb, ""]
    if not csv_path.exists():
        try:
            shown = csv_path.relative_to(csv_path.parents[4])
        except ValueError:
            shown = csv_path
        parts.append(
            f"⚠️ `{shown}` not found. "
            "Run `make benchmarks` and commit the result.\n"
        )
        return "\n".join(parts)
    rows = _read_csv(csv_path)
    if not rows:
        parts.append("⚠️ Empty CSV.\n")
        return "\n".join(parts)
    meta = rows[0]
    parts.append(
        f"*Generated `{meta['generated_at']}` from commit `{meta['git_sha']}`.*\n"
    )

    parts.append("### Summary\n")
    parts.append("| Phase | Vars matched | Cells | Match | Diverge | Missing | Match rate | Residual |")
    parts.append("|-------|--------------|-------|-------|---------|---------|------------|----------|")
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
        parts.append(
            f"| `{phase}` | {n_vars}/{n_total_vars} | {cells} | {match} | "
            f"{diverge} | {missing} | {_fmt_pct(match, cells)} | "
            f"{float(s['residual']):.2e} |"
        )
    parts.append("")

    for phase in ("base", "shock"):
        worst = _top_diverging(rows, phase, n=10)
        if not worst:
            continue
        parts.append(f"### Top diverging variables — `{phase}`\n")
        parts.append("| Var | Py var | Cells | Diverge | Max abs err | Max rel err |")
        parts.append("|-----|--------|-------|---------|-------------|-------------|")
        for r in worst:
            parts.append(
                f"| `{r['var']}` | `{r['py_var']}` | {r['cells']} | "
                f"{r['diverge']} | {r['max_abs_err']} | {r['max_rel_err']} |"
            )
        parts.append("")
    return "\n".join(parts)


HEADER = """# Benchmarks

Variable-by-variable parity between the Python `equilibria` GTAP
Standard 7 implementation and the reference GAMS runs (NEOS), for both the
9x10 and NUS333 datasets. Numbers come from CSVs committed under
`docs/site/_data/benchmarks/` — Read the Docs renders this page from those
files (it has no GAMS/PATH installed). Regenerate locally with:

```bash
make benchmarks
git add docs/site/_data/benchmarks/*.csv
git commit -m "benchmarks: refresh"
```

Each row reports, for one (dataset, phase, variable) triple, how many
Pyomo Var cells match GAMS within `tol_rel=1e-3 / tol_abs=1e-6` and the
worst absolute / relative error observed. The `__SUMMARY__` rows in the
underlying CSV hold per-phase totals.

"""


def render(out_path: Path, data_dir: Path) -> None:
    body = [HEADER]
    for slug, title, blurb in DATASETS:
        body.append(_render_dataset(slug, title, blurb, data_dir / f"{slug}.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    site = here.parent
    render(site / "guide" / "benchmarks.md", site / "_data" / "benchmarks")
    print(f"Wrote {site / 'guide' / 'benchmarks.md'}")
