"""Regenerate a subsidy-aware altertax reference GDX by building the NEOS bundle,
solving it locally via MCP/PATH (GTAPAgg datasets small enough to fit under PATH's
1000-row demo cap — 3x3 altertax = 804 single equations), and installing the
result as tests/fixtures/gtap7_altertax/<dataset>/out_altertax_ifsub{0,1}.gdx.

WHY (2026-07-16): the committed altertax fixtures were built BEFORE the ftrv
case-sensitivity fix (commit d3dfb73) AND the builder lacked FBEP injection, so
the reference was subsidy-BLIND — ytax(ft)=0, ytax(fs)≈0, fcttx=0, fctts=0 — while
Python loads real FBEP/FTRV from basedata.har. The parity gate saw ytax[USA,ft]
py=3.099 vs GAMS 0 (a phantom, the reference was wrong). The fixed builder
(case-insensitive dat routing + FBEP/FTRV HAR injection) yields a subsidy-aware
reference. Per feedback_gams_is_source_of_truth: repair the reference, don't
exclude cells.

Usage:
    uv run python scripts/gtap/regen_altertax_reference.py --dataset gtap7_3x3
    uv run python scripts/gtap/regen_altertax_reference.py --dataset gtap7_3x3 --ifsub 1
"""
from __future__ import annotations
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GAMS = "/Library/Frameworks/GAMS.framework/Versions/Current/Resources/gams"
GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/Current/Resources/gdxdump"
FIXTURES_DIR = ROOT / "tests" / "fixtures" / "gtap7_altertax"


def _build_bundle(dataset: str, ifsub: int) -> Path:
    out_dir = ROOT / "output" / f"{dataset}_altertax_neos_bundle"
    subprocess.run(
        ["uv", "run", "python",
         str(ROOT / "scripts/gtap/build_gtap7_altertax_neos_bundle.py"),
         "--dataset", dataset, "--ifsub", str(ifsub)],
        check=True, cwd=str(ROOT),
    )
    gms = out_dir / f"comp_{dataset}_altertax_neos_ifsub{ifsub}.gms"
    if not gms.exists():
        raise SystemExit(f"builder did not produce {gms}")
    return gms


def _solve(gms: Path, ifsub: int, solver: str) -> Path:
    """Solve the bundle and return the produced out.gdx path.

    solver="mcp": as-is (the bundle is already ifMCP=1/PATH). Only works for
      models under PATH's 1000-row demo cap (3x3 ifsub1 = 804 eqs; ifsub0 does
      NOT reduce size and exceeds the cap → status 7 Licensing Problems).
    solver="nlp": force ifMCP=0 + `option nlp=ipopt` (GAMS's `solve ... using nlp
      maximizing walras` branch). No row cap, so it works for ifsub0 too. Yields
      the SAME economic equilibrium (base=check=shock are equilibrium points; MCP
      and NLP-maximizing-walras converge to the same optimum — the whole NLP-vs-NLP
      saga rests on this). Model status 1 (Optimal) or 2 (Locally Optimal) both OK.
    """
    out_dir = gms.parent
    if solver == "nlp":
        text = gms.read_text()
        text = re.sub(r'ifMCP\s+"[^"]*"\s+/ 1 /',
                      'ifMCP       "Set to 1 to solve using MCP"           / 0 /', text)
        for _m in ("gtap", "dynCal", "dynGTAP"):
            text = text.replace(
                f"   solve {_m} using nlp maximizing walras ;",
                f"   option nlp=ipopt;\n   solve {_m} using nlp maximizing walras ;", 1)
        run_gms = gms.with_name(gms.stem + "_nlp.gms")
        run_gms.write_text(text)
    else:
        run_gms = gms
    gdx_stem = f"out_altertax_{solver}_ifsub{ifsub}"
    r = subprocess.run(
        [GAMS, run_gms.name, "lo=2", f"gdx={gdx_stem}"],
        cwd=str(out_dir), capture_output=True, text=True,
    )
    lst = out_dir / run_gms.with_suffix(".lst").name
    statuses = re.findall(r"MODEL STATUS\s+(\d+)", lst.read_text()) if lst.exists() else []
    bad = [s for s in statuses if s not in ("1", "2")]
    gdx = out_dir / f"{gdx_stem}.gdx"
    if not gdx.exists():
        raise SystemExit(f"{solver} solve produced no GDX (rc={r.returncode})\n{r.stdout[-2000:]}")
    if bad or not statuses:
        raise SystemExit(f"{solver} solve non-optimal model status(es): {statuses}")
    print(f"  {solver} solve model statuses: {statuses} (all optimal)")
    return gdx


def _verify_subsidy_aware(gdx: Path) -> None:
    """Sanity: the regenerated reference MUST carry nonzero ytax(ft) and ytax(fs)
    (the subsidy/tax streams the old contaminated reference zeroed)."""
    def _dump(sym: str) -> str:
        return subprocess.run([GDXDUMP, str(gdx), f"Symb={sym}", "Format=csv"],
                              capture_output=True, text=True).stdout
    ft = [ln for ln in _dump("ytax").splitlines() if '"ft"' in ln and "check" in ln]
    fs = [ln for ln in _dump("ytax").splitlines() if '"fs"' in ln and "check" in ln]

    def _nonzero(lines):
        vals = []
        for ln in lines:
            try:
                vals.append(abs(float(ln.rsplit(",", 1)[-1])))
            except ValueError:
                pass
        return any(v > 1e-6 for v in vals)

    if not _nonzero(ft):
        raise SystemExit("REGRESSION: regenerated reference has ytax(ft)≈0 — FTRV missing")
    if not _nonzero(fs):
        raise SystemExit("REGRESSION: regenerated reference has ytax(fs)≈0 — FBEP missing")
    print("  verified: ytax(ft) and ytax(fs) are nonzero (subsidy-aware)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ifsub", type=int, choices=(0, 1, 2), default=2,
                    help="0, 1, or 2 (=both, default)")
    ap.add_argument("--solver", choices=("nlp", "mcp"), default="nlp",
                    help="nlp (default, no row cap, consistent across ifsub0/ifsub1) "
                         "or mcp (PATH, only for models <1000 rows)")
    ap.add_argument("--no-install", action="store_true",
                    help="build+solve+verify but do not overwrite the fixture")
    args = ap.parse_args()

    subs = (0, 1) if args.ifsub == 2 else (args.ifsub,)
    for ifsub in subs:
        print(f"\n=== altertax reference: {args.dataset} ifsub{ifsub} ({args.solver}) ===")
        gms = _build_bundle(args.dataset, ifsub)
        gdx = _solve(gms, ifsub, args.solver)
        _verify_subsidy_aware(gdx)
        if not args.no_install:
            dest_dir = FIXTURES_DIR / args.dataset
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"out_altertax_ifsub{ifsub}.gdx"
            shutil.copy2(gdx, dest)
            print(f"  installed → {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
