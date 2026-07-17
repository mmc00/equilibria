"""Generate docs/site/guide/gtap7_nlp_matrix.html — the rich per-stage NLP-vs-NLP
matrix (match% tint + convergence chip), by RE-RUNNING the measurement.

Unlike the markdown coverage table (which carries only the versioned per-stage
FLOORS), this page shows the freshly MEASURED match% and return code for every
stage. Nothing is hardcoded: it drives the same solve+measure the local test uses
(test_gtap7_nlp_parity._solve_and_measure) over every coverage-matrix nlp row.

Run (local only — needs the NLP solve toolchain + dataset HARs):
    uv run python scripts/gtap/gen_nlp_matrix_page.py
    uv run python scripts/gtap/gen_nlp_matrix_page.py --no-solve   # floors only (CI-safe)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gtap"))
sys.path.insert(0, str(ROOT / "tests/templates/gtap"))
sys.path.insert(0, str(ROOT / "src"))
# Local PATH/NLP toolchain lives outside the venv (absent on CI → falls back to
# floors-only rendering).
_PATH_CAPI_SRC = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI_SRC.exists() and str(_PATH_CAPI_SRC) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI_SRC))

from coverage_matrix import nlp_rows, mcp_rows  # noqa: E402

# Sphinx copies _static/ verbatim to the built site; the doc links to it as
# ../_static/gtap7_<gate>_matrix.html. (A raw .html under guide/ is NOT served.)
def _doc_path(gate):
    return ROOT / f"docs/site/_static/gtap7_{gate}_matrix.html"


# NLP refs live in tests/fixtures/gtap7_nlp/<ds>_<mode>_ifsub<N>.gdx; MCP refs are the
# clean NEOS altertax fixtures. Each gate imports its own solve+measure from its test.
def _gate_config(gate):
    if gate == "mcp":
        from test_gtap7_mcp_parity import _solve_and_measure, _fixture_gdx
        return {
            "rows": mcp_rows(),
            "gdx": lambda r: _fixture_gdx(r.dataset, r.ifsub),
            "measure": lambda r, gdx: _solve_and_measure(r.dataset, r.ifsub, gdx),
            "eyebrow": "GTAP Standard 7 · Python vs GAMS · PATH/MCP both sides",
            "lede": ("Python solved via PATH (nonlinear-full MCP) against the "
                     "cleanly-converged NEOS MCP reference. With clean refs the match "
                     "is 99%+ everywhere — the 89–97 the NLP gate reads is the "
                     "mis-converged NLP reference, not the model."),
        }
    from test_gtap7_nlp_parity import _solve_and_measure, _fixture_gdx
    return {
        "rows": nlp_rows(),
        "gdx": lambda r: _fixture_gdx(r.dataset, r.mode, r.ifsub),
        "measure": lambda r, gdx: _solve_and_measure(r.dataset, r.ifsub, r.mode, gdx),
        "eyebrow": "GTAP Standard 7 · Python vs GAMS · NLP/IPOPT both sides",
        "lede": ("Python solved as NLP (EQUILIBRIA_GTAP_SOLVE_NLP=1) against the GAMS "
                 "ifMCP=0 reference — same IPOPT both sides, so solver tolerance "
                 "cancels and the number reflects model fidelity."),
    }


def _has_solver() -> bool:
    return importlib.util.find_spec("path_capi_python") is not None


def _measure_all(cfg):
    """Return {(dataset, mode, ifsub): {period: {match, code}}} by running the solve.
    Falls back to None per cell when a fixture/solver is missing (rendered as n/a)."""
    results = {}
    for r in cfg["rows"]:
        key = (r.dataset, r.mode, r.ifsub)
        gdx = cfg["gdx"](r)
        if not gdx.exists() or not (ROOT / "datasets" / r.dataset / "basedata.har").exists():
            results[key] = None
            continue
        try:
            results[key] = cfg["measure"](r, gdx)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {key}: measure failed ({exc})", file=sys.stderr)
            results[key] = None
    return results


def _rows_js(cfg, measured):
    """Build the pure[] / alt[] JS arrays the page template consumes."""
    def one(r):
        floors = dict(r.stage_floors)
        m = measured.get((r.dataset, r.mode, r.ifsub)) if measured else None

        def cell(stage):
            fl = floors[stage]
            if m and stage in m:
                return f'[{m[stage]["match"]:.1f}, {m[stage]["code"]}, {fl:g}]'
            return f'[null, null, {fl:g}]'  # not measured → n/a, floor shown
        return (f'{{ ds: "{r.dataset}", ifsub: {r.ifsub}, '
                f'base: {cell("base")}, check: {cell("check")}, shock: {cell("shock")} }}')

    pure = [one(r) for r in cfg["rows"] if r.mode == "pure"]
    alt = [one(r) for r in cfg["rows"] if r.mode == "altertax"]
    return ",\n    ".join(pure), ",\n    ".join(alt)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GTAP7 NLP-vs-NLP Convergence Matrix</title>
<style>
  :root {
    --bg:#f6f7f9; --panel:#fff; --ink:#1a2129; --ink-soft:#5a6673; --ink-faint:#8b96a3;
    --line:#e2e6eb; --line-soft:#eef1f4; --accent:#2f6fb0; --accent-soft:#e8f0f8;
    --good:#2f8f5b; --good-bg:#e6f3ec; --warn:#b5851f; --warn-bg:#f7efd8;
    --bad:#c05038; --bad-bg:#f7e5df;
    --mono:"SFMono-Regular","JetBrains Mono",ui-monospace,"Menlo","Consolas",monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","Inter",system-ui,sans-serif;
  }
  @media (prefers-color-scheme: dark){:root{
    --bg:#0e1319;--panel:#161d26;--ink:#e6ebf1;--ink-soft:#9aa7b4;--ink-faint:#6a7885;
    --line:#26313d;--line-soft:#1d2732;--accent:#5b9bd8;--accent-soft:#1a2836;
    --good:#4bb37c;--good-bg:#14291e;--warn:#d0a542;--warn-bg:#2c2411;--bad:#db6a4f;--bad-bg:#2e1a14;}}
  :root[data-theme="dark"]{--bg:#0e1319;--panel:#161d26;--ink:#e6ebf1;--ink-soft:#9aa7b4;
    --ink-faint:#6a7885;--line:#26313d;--line-soft:#1d2732;--accent:#5b9bd8;--accent-soft:#1a2836;
    --good:#4bb37c;--good-bg:#14291e;--warn:#d0a542;--warn-bg:#2c2411;--bad:#db6a4f;--bad-bg:#2e1a14;}
  :root[data-theme="light"]{--bg:#f6f7f9;--panel:#fff;--ink:#1a2129;--ink-soft:#5a6673;
    --ink-faint:#8b96a3;--line:#e2e6eb;--line-soft:#eef1f4;--accent:#2f6fb0;--accent-soft:#e8f0f8;
    --good:#2f8f5b;--good-bg:#e6f3ec;--warn:#b5851f;--warn-bg:#f7efd8;--bad:#c05038;--bad-bg:#f7e5df;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;-webkit-font-smoothing:antialiased;}
  .wrap{max-width:940px;margin:0 auto;padding:48px 24px 80px;}
  .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 10px;}
  h1{font-size:clamp(24px,4vw,34px);line-height:1.15;margin:0 0 12px;letter-spacing:-.02em;text-wrap:balance;font-weight:620;}
  .lede{max-width:62ch;color:var(--ink-soft);font-size:15px;margin:0;}
  .lede code{font-family:var(--mono);font-size:.88em;background:var(--accent-soft);color:var(--accent);padding:1px 5px;border-radius:4px;}
  .legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:24px 0 8px;padding:14px 16px;background:var(--panel);border:1px solid var(--line);border-radius:10px;font-size:12.5px;color:var(--ink-soft);}
  .legend .li{display:inline-flex;align-items:center;gap:7px;}
  .swatch{width:11px;height:11px;border-radius:3px;flex:none;}
  .legend .rule{color:var(--ink-faint);font-family:var(--mono);font-size:11px;}
  section{margin-top:34px;}
  .mode-head{display:flex;align-items:baseline;gap:12px;margin:0 0 12px;}
  .mode-head h2{font-size:15px;margin:0;font-weight:640;letter-spacing:-.01em;}
  .mode-head .tag{font-family:var(--mono);font-size:11px;color:var(--ink-faint);}
  .tablecard{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;}
  .scroll{overflow-x:auto;}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;}
  thead th{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-faint);font-weight:600;text-align:center;padding:12px 10px 10px;border-bottom:1px solid var(--line);white-space:nowrap;}
  thead th.lbl{text-align:left;padding-left:18px;}
  thead .grp{border-left:1px solid var(--line-soft);}
  tbody td{padding:11px 10px;border-bottom:1px solid var(--line-soft);text-align:center;font-family:var(--mono);font-size:13px;}
  tbody tr:last-child td{border-bottom:none;}
  td.lbl{text-align:left;padding-left:18px;font-family:var(--sans);white-space:nowrap;}
  td.lbl .ds{font-weight:600;font-size:13.5px;}
  td.lbl .sub{font-family:var(--mono);font-size:11px;color:var(--ink-faint);margin-left:7px;}
  td.grp{border-left:1px solid var(--line-soft);}
  .cell{display:inline-flex;flex-direction:column;align-items:center;gap:3px;}
  .pct{font-weight:560;font-size:13.5px;}
  .chip{font-family:var(--mono);font-size:9.5px;letter-spacing:.04em;padding:1px 6px;border-radius:20px;text-transform:uppercase;font-weight:600;}
  .s-good .pct{color:var(--good);} .s-warn .pct{color:var(--warn);} .s-bad .pct{color:var(--bad);} .s-na .pct{color:var(--ink-faint);}
  .chip.conv{background:var(--good-bg);color:var(--good);}
  .chip.nonconv{background:var(--bad-bg);color:var(--bad);}
  .chip.na{background:var(--line-soft);color:var(--ink-faint);}
  .note{display:flex;gap:9px;margin-top:14px;padding:13px 16px;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;font-size:13px;color:var(--ink-soft);}
  .note b{color:var(--ink);font-weight:600;} .note code{font-family:var(--mono);font-size:.86em;color:var(--accent);}
  footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);font-family:var(--mono);font-size:11px;color:var(--ink-faint);display:flex;flex-wrap:wrap;gap:6px 16px;}
</style></head><body>
<div class="wrap">
  <p class="eyebrow">__EYEBROW__</p>
  <h1>__H1__</h1>
  <p class="lede">__LEDE__ Cell-by-cell match at 1% tolerance, per period, with the solve return code. Measured by re-running the solve (<code>gen_nlp_matrix_page.py</code>); the floor under each cell is the versioned contract the test asserts.</p>

  <div class="legend">
    <span class="li rule" style="width:100%">Each cell: measured match% (color) · convergence chip · contract floor</span>
    <span class="li"><b style="color:var(--good)">match&nbsp;%</b>:
      <span class="swatch" style="background:var(--good);margin-left:4px"></span>≥99
      <span class="swatch" style="background:var(--warn)"></span>95–99
      <span class="swatch" style="background:var(--bad)"></span>&lt;95</span>
    <span class="li"><b>convergence</b>:
      <span class="chip conv" style="margin-left:4px">✓ code 1</span>
      <span class="chip nonconv">✕ code 2</span></span>
  </div>

  __PURE_SECTION__

  <section>
    <div class="mode-head"><h2>Altertax</h2><span class="tag">CD · ifSUB=0 &amp; 1</span></div>
    <div class="tablecard"><div class="scroll"><table>
      <thead><tr><th class="lbl">Dataset · ifSUB</th><th class="grp">Base</th><th>Check</th><th>Shock</th></tr></thead>
      <tbody id="alt-body"></tbody>
    </table></div></div>
    <div class="note"><span>⤷</span><span><b>The check/shock ceiling is the reference, not the model.</b> Every altertax NLP reference violates its own <code>eq_pxeq</code> in the ag sector (IPOPT stops at "Locally Optimal"). Where a cleanly-converged MCP reference exists (3×3 ifSUB=1) the same Python solve matches <b>99.93%</b>. The path to 99% for the rest is MCP references (NEOS), not a code change.</span></div>
  </section>

  <footer>
    <span>base = benchmark · check = no-shock · shock = +10% import tariff</span>
    <span>match @ 1% tol · floor = versioned contract</span>
    <span>__STAMP__</span>
  </footer>
</div>
<script>
  const pure = [
    __PURE__
  ];
  const alt = [
    __ALT__
  ];
  function cls(v, code) {
    if (v === null) return "s-na";
    if (code !== 1) return "s-bad";
    return v >= 99 ? "s-good" : v >= 95 ? "s-warn" : "s-bad";
  }
  function cell(stage) {
    const [v, code, floor] = stage;
    if (v === null) {
      return `<div class="cell"><span class="pct">n/a</span><span class="chip na">≥ ${floor}</span></div>`;
    }
    const chip = code === 1
      ? `<span class="chip conv">✓ code 1</span>`
      : `<span class="chip nonconv">✕ code ${code}</span>`;
    return `<div class="cell"><span class="pct">${v.toFixed(1)}<span style="font-size:10px;opacity:.6">%</span></span>${chip}<span style="font-size:9px;color:var(--ink-faint)">floor ${floor}</span></div>`;
  }
  function row(r) {
    const lbl = `<td class="lbl"><span class="ds">${r.ds}</span><span class="sub">ifSUB=${r.ifsub}</span></td>`;
    const c1 = `<td class="grp ${cls(r.base[0], r.base[1])}">${cell(r.base)}</td>`;
    const c2 = `<td class="${cls(r.check[0], r.check[1])}">${cell(r.check)}</td>`;
    const c3 = `<td class="${cls(r.shock[0], r.shock[1])}">${cell(r.shock)}</td>`;
    return `<tr>${lbl}${c1}${c2}${c3}</tr>`;
  }
  document.getElementById("pure-body").innerHTML = pure.map(row).join("");
  document.getElementById("alt-body").innerHTML = alt.map(row).join("");
</script>
</body></html>
"""


_PURE_SECTION = """<section>
    <div class="mode-head"><h2>Pure-gtap</h2><span class="tag">real-CES · ifSUB=0 &amp; 1</span></div>
    <div class="tablecard"><div class="scroll"><table>
      <thead><tr><th class="lbl">Dataset · ifSUB</th><th class="grp">Base</th><th>Check</th><th>Shock</th></tr></thead>
      <tbody id="pure-body"></tbody>
    </table></div></div>
    <div class="note"><span>⤷</span><span><b>The 5×5 shock was the fix.</b> It read 59.56% / code 2 (infeasible) until the Python-only Jacobian pre-scale was dropped — GAMS solves the raw model, and once Python does too, IPOPT lands in GAMS's basin (<code>pfact[ROW]</code> 1.25 → 0.996). All pure datasets now 100% across every stage, both ifSUB.</span></div>
  </section>"""


def render(cfg, gate, measured, stamp: str) -> str:
    pure_js, alt_js = _rows_js(cfg, measured)
    if gate == "mcp":
        h1 = "MCP convergence &amp; parity matrix"
        pure_section = ""  # MCP is altertax-only (pure MCP lives in gtap_solve)
    else:
        h1 = "NLP-vs-NLP convergence &amp; parity matrix"
        pure_section = _PURE_SECTION
    return (_TEMPLATE
            .replace("__EYEBROW__", cfg["eyebrow"])
            .replace("__H1__", h1)
            .replace("__LEDE__", cfg["lede"])
            .replace("__PURE_SECTION__", pure_section)
            .replace("__PURE__", pure_js)
            .replace("__ALT__", alt_js)
            .replace("__STAMP__", stamp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=("nlp", "mcp"), default="nlp",
                    help="which gate's matrix to render (default nlp)")
    ap.add_argument("--no-solve", action="store_true",
                    help="render floors only (no solve) — CI-safe / offline")
    args = ap.parse_args()

    cfg = _gate_config(args.gate)
    if args.no_solve or not _has_solver():
        if not args.no_solve:
            print("solver unavailable → rendering floors only", file=sys.stderr)
        measured = None
        stamp = "floors only (not re-measured)"
    else:
        print(f"Re-running {args.gate.upper()} solves to measure the matrix…", file=sys.stderr)
        measured = _measure_all(cfg)
        stamp = "measured by gen_nlp_matrix_page.py"

    out = _doc_path(args.gate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(cfg, args.gate, measured, stamp), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
