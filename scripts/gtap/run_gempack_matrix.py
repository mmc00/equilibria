"""Produce the RunGTAP (GEMPACK/GTAPv7) post-shock `updated.har` for every
against-GEMPACK matrix dataset, on a Windows machine with RunGTAP installed.

For each gtap7_* dataset under `datasets/<ds>/` this:
  1. creates an isolated run folder `runs/gempack_matrix/<ds>/`,
  2. copies the dataset inputs (sets.har / basedata.har / default.prm),
  3. writes a `tm10.cmf` mirroring the vetted nus333/9x10 experiment exactly —
     `Shock tm = uniform 10`, GTAPv7 condensed closure, capFix
     `swap dpsave(r)=del_tbalry(r)` for every NON-residual region, residual =
     LAST region (== the Python gate's `rr = list(sets.r)[-1]`),
  4. solves it with `gtapv7.exe -cmf tm10.cmf`,
  5. on a clean solve, copies `updated.har` into
     `tests/fixtures/gtap7_gempack/updated_<ds>_tm10.har` where the SKIP-if-
     missing parity gate looks.

The two hand-tuned datasets nus333 and 9x10 keep their committed `.cmf` under
`runs/<ds>_compare/rungtap/` (9x10 uses residual=NAmerica and needs the EFLG
reconstruction in reconstruct_9x10_eflg.py) — they are NOT regenerated here.

gtap7_20x41 does not solve in GEMPACK (loge-of-negative in E_u for Caribbean,
consistent with its 'blocked' status in the coverage matrix); the runner reports
it as FAILED and moves on without emitting a fixture.

Usage (Windows PowerShell, from repo root):
    uv run python scripts/gtap/run_gempack_matrix.py                # generate + solve + collect
    uv run python scripts/gtap/run_gempack_matrix.py --no-solve     # (re)generate .cmf only
    uv run python scripts/gtap/run_gempack_matrix.py --gtapv7 C:/path/gtapv7.exe
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from equilibria.babel.har.reader import read_har  # noqa: E402

# gtap7_* matrix datasets that ship RunGTAP-ready inputs under datasets/.
DATASETS = [
    "gtap7_3x3",
    "gtap7_3x4",
    "gtap7_5x5",
    "gtap7_10x7",
    "gtap7_15x10",
    "gtap7_20x41",  # known-blocked; attempted so the failure is on the record
]
DATA_DIR = ROOT / "datasets"
RUN_ROOT = ROOT / "runs/gempack_matrix"
FIXTURES = ROOT / "tests/fixtures/gtap7_gempack"
DEFAULT_GTAPV7 = Path(r"C:\runGTAP375\gtapv7.exe")
INPUT_FILES = ("sets.har", "basedata.har", "default.prm")

EXOG_BLOCK = """Exogenous
          pop
          psaveslack pfactwld
          profitslack incomeslack endwslack
          cgdslack
          tradslack
          ams atm atf ats atd
          aosec aoreg avasec avareg
          aintsec aintreg aintall
          afcom afsec afreg afecom afesec afereg
          aoall afall afeall
          au dppriv dpgov dpsave
          to tinc
          tpreg tm tms tx txs
          qe
          qesf ;
Rest Endogenous ;"""


def regions(ds_dir: Path) -> list[str]:
    h = read_har(str(ds_dir / "sets.har"))
    return [str(x).strip() for x in h["REG"].array.tolist()]


def make_cmf(name: str, regs: list[str]) -> str:
    residual = regs[-1]
    swaps = "\n".join(
        f'swap dpsave("{r}") = del_tbalry("{r}") ;' for r in regs[:-1]
    )
    return f"""! {name} uniform 10% shock to import tariff power (tm) - GEMPACK / GTAPv7
! capFix closure mirroring equilibria's Python gate (residual = last region = "{residual}",
! rr=list(sets.r)[-1]); swap dpsave(r)=del_tbalry(r) for every NON-residual region.
Auxiliary files = C:\\runGTAP375\\gtapv7 ;

file GTAPSETS = sets.har ;
file GTAPDATA = basedata.har ;
file GTAPPARM = default.prm ;
file GTAPSUM  = summary.har ;
file WELVIEW  = decomp.har ;
file GTAPVOL  = volume.har ;
Updated file GTAPDATA = updated.har ;

Method = Gragg ;
Steps  = 8 16 32 ;
Automatic accuracy = no ;
Subintervals = 1 ;

Verbal Description =
{name} uniform 10pct shock to import tariff power (tm), capFix closure ;

{EXOG_BLOCK}

! capFix closure: pin del_tbalry for all NON-RESIDUAL regions; "{residual}"
! absorbs the capital-account identity. Releases dpsave(r) for each non-residual region.
{swaps}

Shock tm = uniform 10 ;

CPU = yes ;
NDS = yes ;
log file = yes ;
Extrapolation accuracy file = NO ;
"""


def prepare(name: str) -> Path:
    ds_dir = DATA_DIR / name
    regs = regions(ds_dir)
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in INPUT_FILES:
        shutil.copy2(ds_dir / f, run_dir / f)
    (run_dir / "tm10.cmf").write_text(make_cmf(name, regs), encoding="ascii")
    print(f"  {name:14s} residual={regs[-1]:<12s} regions={len(regs):2d}  -> {run_dir}")
    return run_dir


def solve(run_dir: Path, gtapv7: Path) -> tuple[bool, str]:
    console = run_dir / "solve_console.txt"
    with console.open("w", encoding="utf-8", errors="replace") as fh:
        subprocess.run(
            [str(gtapv7), "-cmf", "tm10.cmf"],
            cwd=run_dir, stdout=fh, stderr=subprocess.STDOUT, check=False,
        )
    text = console.read_text(encoding="utf-8", errors="replace")
    ok = "completed without error" in text.lower()
    res = ""
    for line in text.splitlines():
        if "maximum residual ratio" in line.lower():
            res = line.strip()
            break
    return ok, res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtapv7", type=Path, default=DEFAULT_GTAPV7,
                    help=f"path to gtapv7.exe (default {DEFAULT_GTAPV7})")
    ap.add_argument("--no-solve", action="store_true",
                    help="only (re)generate the .cmf + input copies, do not solve")
    ap.add_argument("--datasets", nargs="*", default=DATASETS,
                    help="subset of datasets to run")
    args = ap.parse_args()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] generating .cmf + copying inputs into {RUN_ROOT}")
    run_dirs = {name: prepare(name) for name in args.datasets}

    if args.no_solve:
        print("--no-solve: stopping after generation.")
        return 0

    if not args.gtapv7.exists():
        print(f"ERROR: gtapv7 not found at {args.gtapv7} — pass --gtapv7 PATH", file=sys.stderr)
        return 2

    print(f"\n[2/3] solving with {args.gtapv7}")
    results: dict[str, tuple[bool, str]] = {}
    for name, run_dir in run_dirs.items():
        for stale in ("updated.har", "summary.har", "decomp.har", "volume.har"):
            (run_dir / stale).unlink(missing_ok=True)
        ok, res = solve(run_dir, args.gtapv7)
        results[name] = (ok, res)
        upd = run_dir / "updated.har"
        sz = upd.stat().st_size if upd.exists() else 0
        print(f"  {name:14s} {'OK ' if ok else 'FAIL'}  {res or '(no residual line)'}  updated.har={sz}B")
        if ok and sz > 0:
            shutil.copy2(upd, FIXTURES / f"updated_{name}_tm10.har")

    print("\n[3/3] summary")
    good = [n for n, (ok, _) in results.items() if ok]
    bad = [n for n, (ok, _) in results.items() if not ok]
    print(f"  solved  ({len(good)}): {', '.join(good) or '-'}")
    print(f"  blocked ({len(bad)}): {', '.join(bad) or '-'}")
    print(f"  fixtures written to {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
