"""Compare GAMS (levels) vs GEMPACK (linearized) on Horridge's SIMPLE model.

Horridge's tpmh0103 archive ships the SAME SIMPLE model in GEMPACK, GAMS/MCP,
GAMS/NLP and MPSGE "to compare implementations" — but ships NO script that actually
lines up the numbers (comparison is left to eyeballing the .har outputs in ViewSOL).
This is that missing script: it reads each engine's %-change results and reports the
GAMS-vs-GEMPACK match cell-by-cell, in ABSOLUTE PERCENTAGE POINTS — the same metric
as the equilibria against-GEMPACK gate.

This settles two questions about the against-GEMPACK page:
  * Q3 (is %-change the right comparison?): Horridge's own GAMS computes
    CH_XFAC=(xfac.l/xfac0-1)*100 — identical to our (s/b-1)*100. Yes.
  * Q2 (why is our matrix match only ~52%?): our matrix shock is +10% tariff power
    on every bilateral route globally — highly non-linear. Horridge's SIMPLE shock is
    small/localized (−10% labour productivity in one sector). This script measures
    GAMS-vs-GEMPACK on THAT small shock; it should be ~99–100% within 1pp, proving
    the ~52% is the shock-size linearization gap, not a defect.

WORKFLOW (Windows, needs GEMPACK + GAMS):
  1. GEMPACK:  gempack.bat        -> writes fixcap.sl4 (+ .har)
               sltoht the .sl4 -> sl4dump.har   (see equilibria guide §8 chain)
  2. GAMS:     dogams.bat         -> ResultsMCP.gdx, ResultsNLP.gdx (levels %-changes)
  3. Compare:  python compare_gams_vs_gempack.py \
                   --gempack sl4dump.har --mcp ResultsMCP.gdx [--nlp ResultsNLP.gdx]

On macOS (no GEMPACK) you can still run the GAMS side and the MCP-vs-NLP levels
cross-check (both should agree to ~8 digits), which this script also reports.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

# ── GEMPACK-variable → GAMS-CH-parameter map for the SIMPLE model ──
# The GAMS .gms exports CH_Z/CH_XFAC/CH_P/CH_PFAC; GEMPACK's SL4 (via sltoht) keeps
# each variable name in its header long_name. SIMPLE's names: z (output), xfac
# (factor demand), p (basic price), pfac (factor price).
G2C = {
    "z": "CH_Z",
    "xfac": "CH_XFAC",
    "p": "CH_P",
    "pfac": "CH_PFAC",
}

GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/Current/Resources/gdxdump"


def read_gams_ch(gdx: Path, sym: str) -> dict[tuple[str, ...], float]:
    """Read a CH_* parameter from a GAMS results gdx as {key: pct}."""
    out: dict[tuple[str, ...], float] = {}
    proc = subprocess.run([GDXDUMP, str(gdx), f"symb={sym}", "format=csv"],
                          capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    for ln in lines[1:]:  # skip header row
        parts = [p.strip('"') for p in ln.split(",")]
        try:
            val = float(parts[-1])
        except ValueError:
            continue
        out[tuple(parts[:-1])] = val
    return out


def read_gempack_ch(har: Path, gvar: str) -> dict[tuple[str, ...], float]:
    """Read a variable's %-change cells from an sltoht SL4 HAR, by name."""
    sys.path.insert(0, str(Path("/Users/marmol/proyectos2/equilibria/src")))
    from equilibria.babel.har.reader import read_har

    headers = read_har(str(har))
    name_to_id = {
        (ha.long_name or "").split("#", 1)[0].strip(): hid
        for hid, ha in headers.items()
        if (ha.long_name or "").strip()
    }
    hid = name_to_id.get(gvar)
    if hid is None:
        raise KeyError(f"{gvar!r} not in SL4 dump {har}")
    ha = headers[hid]
    import numpy as np
    dims = [list(e) for e in ha.set_elements]
    out = {}
    for idx in np.ndindex(*ha.array.shape):
        key = tuple(dims[d][i].lower() for d, i in enumerate(idx))
        out[key] = float(ha.array[idx])
    return out


def match_pp(a: dict, b: dict, tol_pp: float = 1.0) -> tuple[float, float, int]:
    """within-tol fraction, median |Δpp|, n — over shared keys (values in percent)."""
    diffs = []
    for k in a:
        kk = tuple(x.lower() for x in k)
        for cand in (k, kk):
            if cand in b:
                diffs.append(abs(a[k] - b[cand]))
                break
    if not diffs:
        return 0.0, 0.0, 0
    within = sum(1 for d in diffs if d <= tol_pp) / len(diffs)
    return within, statistics.median(diffs), len(diffs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gempack", type=Path, help="sl4dump.har from GEMPACK (sltoht)")
    ap.add_argument("--mcp", type=Path, required=True, help="ResultsMCP.gdx (GAMS levels)")
    ap.add_argument("--nlp", type=Path, help="ResultsNLP.gdx (GAMS levels, optional)")
    ap.add_argument("--tol-pp", type=float, default=1.0)
    args = ap.parse_args()

    print(f"SIMPLE model — GAMS(levels) vs GEMPACK(linearized), tol {args.tol_pp}pp\n")

    if args.nlp and args.nlp.exists():
        print("=== sanity: GAMS/MCP vs GAMS/NLP (both levels — should be ~8 digits) ===")
        for gv, ch in G2C.items():
            m = read_gams_ch(args.mcp, ch)
            n = read_gams_ch(args.nlp, ch)
            w, med, cnt = match_pp(m, n, args.tol_pp)
            print(f"  {ch:10s} within {args.tol_pp}pp={w * 100:5.1f}%  median|Δ|={med:.2e}pp  (n={cnt})")
        print()

    if not args.gempack or not args.gempack.exists():
        print("No --gempack SL4 dump given (or missing) — run on Windows with GEMPACK.\n"
              "The GAMS-vs-GAMS cross-check above already shows levels≡levels agreement.")
        return 0

    print("=== GAMS(levels) vs GEMPACK(linearized) — the key comparison ===")
    all_within = []
    for gv, ch in G2C.items():
        try:
            gem = read_gempack_ch(args.gempack, gv)
        except KeyError as e:
            print(f"  {ch:10s} (skip: {e})")
            continue
        gams = read_gams_ch(args.mcp, ch)
        w, med, cnt = match_pp(gams, gem, args.tol_pp)
        all_within.append((w, cnt))
        print(f"  {ch:10s} within {args.tol_pp}pp={w * 100:5.1f}%  median|Δ|={med:.2f}pp  (n={cnt})")
    if all_within:
        tot = sum(c for _, c in all_within)
        wavg = sum(w * c for w, c in all_within) / tot
        print(f"\n  OVERALL within {args.tol_pp}pp = {wavg * 100:.1f}%")
        print("  (Horridge's small localized shock should give ~99–100% — vs our matrix's")
        print("   ~52% at a +10% global bilateral tariff = the shock-size linearization gap.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
