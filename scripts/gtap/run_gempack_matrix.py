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
DEFAULT_SLTOHT = Path(r"C:\GP\sltoht.exe")  # GEMPACK SL4→HAR converter
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


def config_tag(shock_pct: float, steps: str, rordelta: int | None = None) -> str:
    """Stable, filesystem-safe tag for a (shock, steps[, rordelta]) config.

    e.g. (10, "8 16 32") -> "tm10_s8-16-32"; (0.1, ...) -> "tm0p1_s..." — no dots
    or spaces, so it is safe as a .cmf / .har basename in the study grid.

    ``rordelta`` (the capital-account closure) appends the closure variant:
      rordelta=0 -> "_capfix"  (returns DIFFER across regions),
      rordelta=1 -> "_capflex" (returns EQUALIZE — rore uniform).
    Omit it (None) for the legacy single-variant tag (backward compatible).
    """
    s = f"{shock_pct:g}".replace(".", "p")
    st = steps.replace(" ", "-")
    base = f"tm{s}_s{st}"
    if rordelta is None:
        return base
    return f"{base}_{'capflex' if int(rordelta) == 1 else 'capfix'}"


def make_cmf(name: str, regs: list[str], shock_pct: float = 10.0,
             steps: str = "8 16 32", updated_name: str = "updated.har",
             rordelta: int | None = None, prm_name: str = "default.prm") -> str:
    """Write the .cmf for the STANDARD GTAP closure. ``rordelta`` selects the
    capital-account variant (GTAPv7.tab RORDELTA, from GTAPPARM header RDLT):

      * rordelta=1 -> capFlex: investment reallocated so expected returns EQUALIZE
        (rore uniform). Matches equilibria savf_flag=capFlex.
      * rordelta=0 -> capFix: investment by fixed capital shares, returns DIFFER by
        region. Matches equilibria savf_flag=capFix.

    The value is forced in the .cmf with ``xSet`` so we don't depend on the dataset's
    RDLT header. ``rordelta=None`` (legacy) leaves RORDELTA at the dataset's RDLT value.

    Both variants keep the STANDARD saving closure (dpsave EXOGENOUS — the Julia GTAPv7 /
    GEMPACK-canonical closure); the earlier `swap dpsave=del_tbalry` (non-standard, savings
    redistribute) is GONE. See the dev-tools attempts log
    (2026-07-31-gams-gempack-gap-attempts-log.md).
    """
    pct = f"{shock_pct:g}"
    if rordelta is None:
        variant = "dataset RDLT"
    else:
        variant = "capflex (RORDELTA=1)" if int(rordelta) == 1 else "capfix (RORDELTA=0)"
    # RORDELTA is a GEMPACK Coefficient read from the GTAPPARM header RDLT — it is NOT
    # settable in the .cmf (that is `xSet`, for sets only). `prepare` therefore writes the
    # RDLT header of the run_dir's default.prm to the requested value (see _force_rdlt).
    rline = ""
    return f"""! {name} uniform {pct}% shock to import tariff power (tm) - GEMPACK / GTAPv7
! STANDARD GTAP closure (GEMPACK gtapv7.cmf): dpsave EXOGENOUS, no swaps. Capital-account
! variant: {variant}. capFix (RORDELTA=0) returns differ; capFlex (RORDELTA=1) equalize.
Auxiliary files = C:\\runGTAP375\\gtapv7 ;

file GTAPSETS = sets.har ;
file GTAPDATA = basedata.har ;
file GTAPPARM = {prm_name} ;
file GTAPSUM  = summary.har ;
file WELVIEW  = decomp.har ;
file GTAPVOL  = volume.har ;
Updated file GTAPDATA = {updated_name} ;

Method = Gragg ;
Steps  = {steps} ;
Automatic accuracy = no ;
Subintervals = 1 ;

Verbal Description =
{name} uniform {pct}pct shock to import tariff power (tm), standard GTAP closure, {variant} ;

{rline}{EXOG_BLOCK}

Shock tm = uniform {pct} ;

CPU = yes ;
NDS = yes ;
log file = yes ;
Extrapolation accuracy file = NO ;
"""


def _force_rdlt(prm_path: Path, rordelta: int) -> None:
    """Rewrite the GTAPPARM header RDLT (the RORDELTA coefficient) in-place to ``rordelta``.

    RORDELTA selects the capital-account closure (GTAPv7.tab 2528: read from header RDLT):
    0 = capFix (returns differ), 1 = capFlex (returns equalize). Editing the .prm header is
    the robust way to force it — RORDELTA is a Coefficient, not settable from the .cmf.
    """
    import numpy as np

    from equilibria.babel.har import write_har
    from equilibria.babel.har.reader import read_har as _rd

    hars = _rd(str(prm_path))
    key = next((k for k in hars if str(k).upper() == "RDLT"), None)
    if key is None:
        return  # no RDLT header — leave the .prm as-is (dataset default)
    ha = hars[key]
    arr = np.asarray(ha.array, dtype=float)
    arr[...] = float(rordelta)
    ha.array = arr
    write_har(str(prm_path), hars)


def prepare(name: str, shock_pct: float = 10.0, steps: str = "8 16 32",
            rordelta: int | None = None) -> tuple[Path, str]:
    """Write a config-tagged .cmf for (name, shock_pct, steps[, rordelta]); return
    (run_dir, tag).

    The .cmf is named <tag>.cmf (not the fixed tm10.cmf) so the grid — same dataset, many
    (shock, steps, closure) configs — does not overwrite itself. When ``rordelta`` is given
    the tag/fixtures carry the ``_capfix`` / ``_capflex`` variant so BOTH capital-account
    closures coexist in the fixtures dir.
    """
    ds_dir = DATA_DIR / name
    regs = regions(ds_dir)
    run_dir = RUN_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in INPUT_FILES:
        shutil.copy2(ds_dir / f, run_dir / f)
    tag = config_tag(shock_pct, steps, rordelta)
    # Force the capital-account closure by writing RDLT into a VARIANT-SPECIFIC .prm
    # (RORDELTA is a Coefficient, not settable from the .cmf). A per-variant .prm avoids the
    # two closures overwriting a shared default.prm in the same run_dir. rordelta=None keeps
    # the plain default.prm (dataset default).
    if rordelta is not None:
        prm_name = f"default_{tag}.prm"
        shutil.copy2(ds_dir / "default.prm", run_dir / prm_name)
        _force_rdlt(run_dir / prm_name, int(rordelta))
    else:
        prm_name = "default.prm"
    cmf = make_cmf(name, regs, shock_pct, steps, updated_name=f"updated_{tag}.har",
                   rordelta=rordelta, prm_name=prm_name)
    (run_dir / f"{tag}.cmf").write_text(cmf, encoding="ascii")
    print(f"  {name:14s} {tag:26s} residual={regs[-1]:<10s} regions={len(regs):2d}")
    return run_dir, tag


def solve(run_dir: Path, gtapv7: Path, tag: str) -> tuple[bool, str]:
    console = run_dir / f"solve_console_{tag}.txt"
    with console.open("w", encoding="utf-8", errors="replace") as fh:
        subprocess.run(
            [str(gtapv7), "-cmf", f"{tag}.cmf"],
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


def convert_sl4(run_dir: Path, sltoht: Path, tag: str) -> Path | None:
    """Convert the auto-written GEMPACK solution `<tag>.sl4` (per-variable %-changes,
    the QUANTITY vars qfd/qxs/qo/... the values-only updated.har does not carry) into
    a plain HAR readable by `read_har`, mirroring trajectory_runner.py's sltoht chain.

    The .sti drives sltoht in batch:
        <cmf stem> / c / n / har / <out.har> / e
    Returns the produced HAR path, or None if the .sl4 is absent (older solve).
    """
    sl4 = run_dir / f"{tag}.sl4"
    if not sl4.exists():
        return None
    out = run_dir / f"sl4dump_{tag}.har"
    sti = run_dir / f"sl4dump_{tag}.sti"
    sti.write_text(f"\n{tag}\nc\nn\nhar\nsl4dump_{tag}.har\ne\n", encoding="ascii")
    console = run_dir / f"sltoht_console_{tag}.txt"
    with console.open("w", encoding="utf-8", errors="replace") as fh:
        subprocess.run(
            [str(sltoht), "-sti", f"sl4dump_{tag}.sti"],
            cwd=run_dir, stdout=fh, stderr=subprocess.STDOUT, check=False,
        )
    return out if out.exists() else None


# Linearization-study grid axes (see docs/findings/gempack_linearization_study_spec).
SWEEP_SHOCKS = [10.0, 3.0, 1.0, 0.3, 0.1]   # shock-size sweep at default steps
GRAGG_STEPS = ["4", "8", "16", "32", "64"]  # Gragg refinement at 10% shock


def _grid_configs() -> list[tuple[float, str]]:
    """The study grid: shock sweep (default steps) + Gragg sweep (10% shock),
    de-duplicated on the (10.0, '8 16 32') overlap."""
    configs = [(s, "8 16 32") for s in SWEEP_SHOCKS]
    configs += [(10.0, st) for st in GRAGG_STEPS]
    return list(dict.fromkeys(configs))


def emit_bat(jobs: list[tuple[str, Path, str, float, str]], out_bat: Path) -> None:
    """Write a Windows .bat that solves every (dataset, tag) job: gtapv7 -cmf,
    sltoht SL4→HAR, and copy updated/decomp/sl4dump into the fixtures dir.

    Uses %GTAPV7% / %SLTOHT% (with C:\\ defaults) so the operator only edits the
    two paths at the top. Paths are REPO-RELATIVE (the .bat lives in
    runs/gempack_matrix/ and is run from there on the Windows clone), so it does
    NOT hard-code the mac worktree it was generated on. Idempotent per job; a
    failed solve leaves "—" downstream.
    """
    # The .bat sits in runs/gempack_matrix/; fixtures are ../../tests/fixtures/...
    fixtures_rel = "..\\..\\tests\\fixtures\\gtap7_gempack"
    lines = [
        "@echo off",
        "REM Auto-generated by run_gempack_matrix.py --grid --emit-bat.",
        "REM Runs the whole linearization-study grid on Windows (RunGTAP + GEMPACK).",
        "REM Run this .bat from its own folder (runs\\gempack_matrix). Edit the two",
        "REM tool paths below if your install differs.",
        'if "%GTAPV7%"=="" set GTAPV7=C:\\runGTAP375\\gtapv7.exe',
        'if "%SLTOHT%"=="" set SLTOHT=C:\\GP\\sltoht.exe',
        'set HERE=%~dp0',
        f'set FIXTURES=%HERE%{fixtures_rel}',
        "",
    ]
    for name, _run_dir, tag, _shock, _steps in jobs:
        lines += [
            f"echo === {name} {tag} ===",
            f"cd /d %HERE%{name}",
            f'"%GTAPV7%" -cmf {tag}.cmf',
            # SL4 -> HAR quantity dump via sltoht (.sti batch script)
            f'( echo.& echo {tag}& echo c& echo n& echo har& '
            f'echo sl4dump_{tag}.har& echo e ) > sl4dump_{tag}.sti',
            f'"%SLTOHT%" -sti sl4dump_{tag}.sti',
            # collect fixtures (named as gen_linearization_study expects)
            f'if exist updated_{tag}.har copy /y updated_{tag}.har '
            f'"%FIXTURES%\\updated_{name}_{tag}.har"',
            f'if exist sl4dump_{tag}.har copy /y sl4dump_{tag}.har '
            f'"%FIXTURES%\\sl4dump_{name}_{tag}.har"',
            f'if exist decomp.har copy /y decomp.har '
            f'"%FIXTURES%\\decomp_{name}_{tag}.har"',
            "",
        ]
    lines += [
        "echo === grid complete ===",
        "echo Fixtures copied to %FIXTURES%",
        "echo Now run:  uv run python scripts/gtap/gen_linearization_study.py",
    ]
    out_bat.write_text("\r\n".join(lines), encoding="ascii")
    print(f"  emitted {out_bat}  ({len(jobs)} jobs)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtapv7", type=Path, default=DEFAULT_GTAPV7,
                    help=f"path to gtapv7.exe (default {DEFAULT_GTAPV7})")
    ap.add_argument("--sltoht", type=Path, default=DEFAULT_SLTOHT,
                    help=f"path to sltoht.exe for SL4→HAR quantity export "
                         f"(default {DEFAULT_SLTOHT}); skipped if not found")
    ap.add_argument("--shock-pct", type=float, default=10.0,
                    help="uniform tm shock magnitude in %% (default 10; use 1 for the "
                         "linearization check, guide §9)")
    ap.add_argument("--steps", type=str, default="8 16 32",
                    help="Gragg Steps line for a single-config run (default '8 16 32')")
    ap.add_argument("--grid", action="store_true",
                    help="emit the full linearization-study grid: shock sweep "
                         "(10/3/1/0.3/0.1%% at default steps) + Gragg sweep (steps "
                         "4/8/16/32/64 at 10%%). Overrides --shock-pct/--steps.")
    ap.add_argument("--no-solve", action="store_true",
                    help="only (re)generate the .cmf + input copies, do not solve")
    ap.add_argument("--emit-bat", action="store_true",
                    help="with --grid, also write run_study_windows.bat driving "
                         "every config (solve + sltoht + copy to fixtures)")
    ap.add_argument("--datasets", nargs="*", default=DATASETS,
                    help="subset of datasets to run")
    ap.add_argument("--rordelta", type=int, choices=[0, 1], action="append",
                    help="capital-account closure variant(s) to generate: 0=capFix "
                         "(returns differ), 1=capFlex (returns equalize). Repeat for both "
                         "(--rordelta 0 --rordelta 1). Default: both, so the fixtures dir "
                         "carries _capfix and _capflex side by side. Omit-and-empty keeps "
                         "the legacy single variant (dataset RDLT).")
    args = ap.parse_args()

    configs = _grid_configs() if args.grid else [(args.shock_pct, args.steps)]
    # Default: generate BOTH closures so capFix (fast gate) and capFlex (GEMPACK-faithful)
    # fixtures coexist. `--rordelta` can restrict to one.
    rordeltas = args.rordelta if args.rordelta else [0, 1]

    FIXTURES.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] generating {len(configs)} config(s) × {len(args.datasets)} dataset(s) "
          f"× {len(rordeltas)} closure(s) {rordeltas} into {RUN_ROOT}")
    # jobs: (name, run_dir, tag, shock_pct, steps)
    jobs: list[tuple[str, Path, str, float, str]] = []
    for name in args.datasets:
        for shock_pct, steps in configs:
            for rd in rordeltas:
                run_dir, tag = prepare(name, shock_pct, steps, rordelta=rd)
                jobs.append((name, run_dir, tag, shock_pct, steps))

    if args.emit_bat:
        emit_bat(jobs, RUN_ROOT / "run_study_windows.bat")

    if args.no_solve:
        print(f"--no-solve: stopping after generating {len(jobs)} .cmf file(s).")
        return 0

    if not args.gtapv7.exists():
        print(f"ERROR: gtapv7 not found at {args.gtapv7} — pass --gtapv7 PATH", file=sys.stderr)
        return 2

    print(f"\n[2/3] solving {len(jobs)} job(s) with {args.gtapv7}")
    results: dict[str, tuple[bool, str]] = {}
    have_sltoht = args.sltoht.exists()
    if not have_sltoht:
        print(f"  note: sltoht not at {args.sltoht} — SL4 quantity export skipped "
              f"(pass --sltoht PATH to enable)")
    for name, run_dir, tag, _shock, _steps in jobs:
        for stale in (f"updated_{tag}.har", "summary.har", "decomp.har",
                      "volume.har", f"{tag}.sl4", f"sl4dump_{tag}.har"):
            (run_dir / stale).unlink(missing_ok=True)
        ok, res = solve(run_dir, args.gtapv7, tag)
        results[f"{name}/{tag}"] = (ok, res)
        upd = run_dir / f"updated_{tag}.har"
        sz = upd.stat().st_size if upd.exists() else 0
        note = ""
        if ok and sz > 0:
            shutil.copy2(upd, FIXTURES / f"updated_{name}_{tag}.har")
            # decomp.har (WELVIEW) carries the welfare EV decomposition — collect it
            # per config so the study's welfare section can read it.
            decomp = run_dir / "decomp.har"
            if decomp.exists() and decomp.stat().st_size > 0:
                shutil.copy2(decomp, FIXTURES / f"decomp_{name}_{tag}.har")
            # Second pass: SL4 → HAR quantity %-changes (qfd/qxs/qo/...) that the
            # values-only updated.har lacks. Committed as a separate fixture.
            if have_sltoht:
                dump = convert_sl4(run_dir, args.sltoht, tag)
                if dump and dump.stat().st_size > 0:
                    shutil.copy2(dump, FIXTURES / f"sl4dump_{name}_{tag}.har")
                    note = f"  sl4dump={dump.stat().st_size}B"
                else:
                    note = "  sl4dump=FAILED"
        print(f"  {name:14s} {tag:22s} {'OK ' if ok else 'FAIL'}  "
              f"{res or '(no residual line)'}  updated={sz}B{note}")

    print("\n[3/3] summary")
    good = [k for k, (ok, _) in results.items() if ok]
    bad = [k for k, (ok, _) in results.items() if not ok]
    print(f"  solved  ({len(good)}): {', '.join(good) or '-'}")
    print(f"  blocked ({len(bad)}): {', '.join(bad) or '-'}")
    print(f"  fixtures written to {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
