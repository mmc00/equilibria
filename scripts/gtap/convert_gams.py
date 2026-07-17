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
    gtap MCP solve through CONVERT, (3) for check, abort right after the check solve.

    Two distinct comp.gms shapes exist across datasets:
    (a) separate standalone `solve gtap using mcp ;` per period (2+ matches) — CONVERT-route
        the first, abort right after the second (the check standalone).
    (b) ONE `solve gtap using mcp ;` textually, executed multiple times inside a single
        `loop(tsim, ...)` over t={base,check,shock} (gtap7_15x10 and any other multi-period
        bundle sharing this structure). `option mcp=convert` is a SOLVER SETTING, not
        control flow — setting it once before the loop routes CONVERT to the FIRST
        execution of that solve statement (tsim="base", the first set element), regardless
        of any abort placed after it. An abort inserted unconditionally right after the
        solve (case a's approach) would therefore silently capture BASE, not check/shock,
        even though it looks like it "worked" (GAMS does halt, dict/pyomo files DO get
        written) — confirmed empirically: two prior attempts both captured base while
        appearing to target check. Fix: gate `option mcp=convert;` itself on
        `sameas(tsim,"check")` (or "shock"), placed INSIDE the loop each iteration, so
        CONVERT only activates when the loop reaches the requested period — then abort
        unconditionally right after that solve.
    """
    out = src_text

    # (1) scaleopt = 0 on gtap (NOT on dyncal/dynGTAP — leave those as-is).
    out = re.sub(r"gtap\.scaleopt\s*=\s*1\s*;", "gtap.scaleopt = 0 ;", out, count=1)

    solves = list(re.finditer(r"^(\s*)solve\s+gtap\s+using\s+mcp\s*;", out, flags=re.MULTILINE))
    if not solves:
        raise RuntimeError("Could not find `solve gtap using mcp ;` in the .gms to route to CONVERT.")

    if len(solves) >= 2:
        # Case (a): separate standalone solves per period. CONVERT-route the first
        # unconditionally (base/check share no loop — the first IS the one we want for
        # base, and for check we route+abort at the 2nd standalone solve instead).
        first = solves[0]
        indent = first.group(1)
        inject = f"{indent}option mcp=convert;\n{indent}gtap.optfile = 1;\n"
        out = out[:first.start()] + inject + out[first.start():]
        if period == "check":
            # Re-locate solves in the now-shifted text and route+abort at the 2nd match
            # instead (the check standalone) — undo the unconditional routing above by
            # re-running from scratch with the routing targeted at solves[1].
            out = src_text
            out = re.sub(r"gtap\.scaleopt\s*=\s*1\s*;", "gtap.scaleopt = 0 ;", out, count=1)
            solves2 = list(re.finditer(r"^(\s*)solve\s+gtap\s+using\s+mcp\s*;", out, flags=re.MULTILINE))
            target = solves2[1]
            indent = target.group(1)
            inject = f"{indent}option mcp=convert;\n{indent}gtap.optfile = 1;\n"
            out = out[:target.start()] + inject + out[target.start():]
            after = target.end() + len(inject)
            out = out[:after] + f'\n{indent}abort$(1) "CONVERT: emitted check model" ;\n' + out[after:]
        return out

    # Case (b): one solve statement, executed repeatedly inside loop(tsim, ...). `option`
    # is a global solver setting, not a conditional assignment — `option x$cond` is not
    # valid GAMS. Gate CONVERT's activation with an `if(sameas(tsim,"want"), ...)` block
    # around the option/optfile lines instead, so the solver only routes to CONVERT on
    # the loop iteration we actually want, not the first one textually reached.
    target = solves[0]
    indent = target.group(1)
    want = "check" if period == "check" else "shock"
    inject = (
        f'{indent}if(sameas(tsim,"{want}"),\n'
        f'{indent}   option mcp=convert;\n'
        f'{indent}   gtap.optfile = 1;\n'
        f'{indent}) ;\n'
    )
    out = out[:target.start()] + inject + out[target.start():]
    after = target.end() + len(inject)
    out = (
        out[:after]
        + f'\n{indent}abort$(sameas(tsim,"{want}")) "CONVERT: emitted {period} model" ;\n'
        + out[after:]
    )
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
