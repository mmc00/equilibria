"""CONVERT — emit the GAMS reference model as canonical Pyomo (.py + dict), level-scaled,
ready to diff equation-by-equation against equilibria. Formalizes the manual /tmp recipe
that settled "does the model differ, or did I mistranslate?" on neutral Pyomo-vs-Pyomo terrain.

WHAT IT DOES: takes the committed comp .gms for a dataset, rewrites it to (a) turn OFF scaling
(gtap.scaleopt=0 → levels-vs-levels, so a residual at a point is the REAL residual not a scaled
one), (b) route the MCP solve through GAMS's CONVERT solver, which writes the model as a Pyomo
ConcreteModel (conv.py) plus a dict mapping x###→var(idx)/e###→eq(idx). For the CHECK period it
inserts an abort right after the check solve so CONVERT emits the check model, not the shock.

WHY (and where it fits — it CONFIRMS, it does not discover): CONVERT removes the
order/names/standard-form ambiguity by putting GAMS's equation and equilibria's in the SAME
language. But it only shows that two equations DIFFER IN WRITING — not whether that difference
MATTERS (two forms can be written differently and give the same result; e.g. complementarity-vs-
solved CD diverge in form but are equivalent). So CONVERT is step (c) in the sequence, AFTER the
discriminator and the residual tail have NAMED the candidate equation:
    a. seed_and_solve  → STAYS (selection, stop) or GOES (an eq differs, + the residual TAIL)
    b. residual tail   → which equation leaves residual at the GAMS point = the candidate
    c. CONVERT (this)  → diff THAT equation, written, vs GAMS in Pyomo → confirm real vs equivalent
    d. bench           → confirm the fix lifts the cell-by-cell gate without breaking it

PRECEDENT: gtap7_3x3. CONVERT proved the translation FAITHFUL (~70/87 families match) and showed
the CD-nest A2 diffs were writing-only (inert). It does NOT find the culprit — seed_and_solve does
— it confirms whether the named candidate genuinely differs.

REQUIRES a local GAMS (CONVERT solver ships with it). GAMS at
/Library/Frameworks/GAMS.framework/Versions/<v>/Resources/gams. No license needed for CONVERT.

Usage:
    uv run python scripts/gtap/convert_gams.py --dataset gtap7_3x3 --period shock \\
        --out-dir /tmp/gtap_convert
    # → <out-dir>/conv_shock.py  (canonical Pyomo)  +  <out-dir>/dict_shock.txt  (x###/e### map)

Then diff a named equation: find its e### in dict_shock.txt, read m.e### in conv_shock.py,
compare to equilibria's eq_<name> rule.
"""
from __future__ import annotations
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _find_gams() -> str:
    """Locate the GAMS binary (CONVERT ships with it; no license needed)."""
    found = shutil.which("gams")
    if found:
        return found
    base = Path("/Library/Frameworks/GAMS.framework/Versions")
    if base.exists():
        cands = sorted(base.glob("*/Resources/gams"), reverse=True)
        if cands:
            return str(cands[0])
    raise RuntimeError(
        "GAMS binary not found. CONVERT requires a local GAMS install "
        "(checked PATH and /Library/Frameworks/GAMS.framework/Versions/*/Resources/gams)."
    )


def _source_gms(dataset: str, if_sub: bool) -> Path:
    suffix = "ifsub1" if if_sub else "ifsub0"
    # comp .gms naming: comp_<dataset>_gtap_shock_<suffix>.gms (committed fixtures)
    candidates = list((ROOT / "tests" / "fixtures" / "gtap7" / dataset).glob(
        f"comp_{dataset}_*_{suffix}.gms"))
    if not candidates:
        raise RuntimeError(
            f"No comp .gms found for {dataset}/{suffix} under tests/fixtures/gtap7/{dataset}/")
    # prefer the shock comp (the canonical altertax/gtap shock reference)
    shock = [c for c in candidates if "shock" in c.name]
    return (shock or candidates)[0]


def _rewrite_for_convert(src_text: str, period: str) -> str:
    """Rewrite the comp .gms to (1) turn scaling OFF (levels-vs-levels), (2) route the
    gtap MCP solve through CONVERT, (3) for check, abort right after the check solve."""
    out = src_text

    # (1) scaleopt = 0 on gtap (NOT on dyncal/dynGTAP — leave those as-is).
    out = re.sub(r"gtap\.scaleopt\s*=\s*1\s*;", "gtap.scaleopt = 0 ;", out, count=1)

    # (2) route the gtap solve through CONVERT. The optfile mechanism is the clean way:
    #     set gtap.optfile and write convert.opt alongside. We force the solver to CONVERT
    #     by injecting `option mcp=convert;` just before the FIRST `solve gtap using mcp`.
    #     (betaCal/dyncal solves are left on the default solver.)
    m = re.search(r"^(\s*)solve\s+gtap\s+using\s+mcp\s*;", out, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find `solve gtap using mcp ;` in the .gms to route to CONVERT.")
    indent = m.group(1)
    inject = f"{indent}option mcp=convert;\n{indent}gtap.optfile = 1;\n"
    out = out[:m.start()] + inject + out[m.start():]

    # (3) period selection. For shock: let it run through (CONVERT writes the LAST gtap solve,
    #     which is the shock). For check: abort right AFTER the check solve so CONVERT emits the
    #     check model. The check solve is the gtap solve INSIDE the homotopy/check loop; we abort
    #     after the first standalone `solve gtap using mcp ;` that is NOT inside the ramp.
    if period == "check":
        # Insert an abort after the check-period solve. The committed comp has a standalone
        # `solve gtap using mcp ;` for the check before the shock block; abort right after it.
        solves = list(re.finditer(r"^(\s*)solve\s+gtap\s+using\s+mcp\s*;", out, flags=re.MULTILINE))
        if len(solves) >= 2:
            # second gtap-solve = the check standalone (first is now CONVERT-routed); abort after it
            after = solves[1].end()
            out = out[:after] + f'\n{solves[1].group(1)}abort$(1) "CONVERT: emitted check model" ;\n' + out[after:]
    return out


def run_convert(dataset: str, period: str, if_sub: bool, out_dir: Path) -> tuple[Path, Path]:
    gams = _find_gams()
    src = _source_gms(dataset, if_sub)
    out_dir.mkdir(parents=True, exist_ok=True)

    py_out = out_dir / f"conv_{period}.py"
    dict_out = out_dir / f"dict_{period}.txt"
    model_gms = out_dir / f"model_{period}.gms"
    opt = out_dir / "convert.opt"

    # convert.opt: emit Pyomo + dict mapping. Paths are relative to the GAMS run cwd (out_dir).
    opt.write_text(f"pyomo {py_out.name}\ndict {dict_out.name}\n")

    model_gms.write_text(_rewrite_for_convert(src.read_text(), period))

    print(f"[convert] dataset={dataset} period={period} ifsub={int(if_sub)}", file=sys.stderr)
    print(f"[convert] source .gms: {src}", file=sys.stderr)
    print(f"[convert] running GAMS ({gams}) with mcp=convert, scaleopt=0 ...", file=sys.stderr)
    # GAMS writes scratch in cwd; run inside out_dir so conv/dict land there.
    proc = subprocess.run(
        [gams, model_gms.name, "lo=2", "optdir=" + str(out_dir)],
        cwd=str(out_dir), capture_output=True, text=True, timeout=600)
    if not py_out.exists():
        tail = (proc.stdout or "")[-1500:] + "\n--- stderr ---\n" + (proc.stderr or "")[-800:]
        raise RuntimeError(f"CONVERT did not produce {py_out.name}. GAMS log tail:\n{tail}")
    return py_out, dict_out


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit the GAMS reference as canonical Pyomo (CONVERT).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--period", default="shock", choices=["check", "shock"])
    ap.add_argument("--if-sub", action="store_true", help="use the ifSUB=1 comp .gms (default ifSUB=0)")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/gtap_convert"))
    args = ap.parse_args()
    py_out, dict_out = run_convert(args.dataset, args.period, args.if_sub, args.out_dir)
    print(f"canonical Pyomo: {py_out}")
    print(f"var/eq map:      {dict_out}")
    print(f"\nNext: find the candidate eq's e### in {dict_out.name}, read m.e### in {py_out.name}, "
          f"compare to equilibria's eq_<name> rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
