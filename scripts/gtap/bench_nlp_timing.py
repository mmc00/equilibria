"""NLP wall-time benchmark: Python (IPOPT, EQUILIBRIA_GTAP_SOLVE_NLP=1) vs GAMS
ifMCP=0 (NLP/IPOPT) local, same machine, N runs per side.

Motivation: unlike the MCP/PATH path (which the community license caps at 1000
rows, forcing NEOS for anything bigger than ~NUS333), the NLP path uses IPOPT —
an open-source solver GAMS does NOT license-limit — so BOTH sides run locally on
the same host, up to gtap7_15x10. The existing `nus333_timing.csv` covers only
the MCP case; this fills the NLP gap with a like-for-like ratio.

For each (dataset, mode, ifsub) with a local GAMS NLP fixture, both sides are
timed. Datasets without a GAMS fixture (e.g. gtap7_15x10) are timed Python-only
(the `gams_local_seconds` column is left blank).

The wall-time measured is the SOLVE of the full multi-period sequence
(base -> check -> shock), warm-up run discarded, mirroring bench_nus333_dual.py.
Parity is NOT re-measured here (that is the job of the NLP parity gate,
test_gtap7_nlp_parity.py); this script only records timing.

Run:
    uv run python scripts/gtap/bench_nlp_timing.py --runs 5 \
        --timing-csv docs/site/_data/benchmarks/nlp_timing.csv

    # subset while iterating:
    uv run python scripts/gtap/bench_nlp_timing.py --runs 2 \
        --only gtap7_3x3 gtap7_5x5 --timing-csv /tmp/nlp_timing.csv
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_PATH_CAPI = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI.exists() and str(_PATH_CAPI) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI))

from _diff_core import git_short_sha  # noqa: E402

DATASETS_DIR = ROOT / "datasets"
NLP_FIXTURES = ROOT / "tests" / "fixtures" / "gtap7_nlp"
GAMS = "/Library/Frameworks/GAMS.framework/Versions/Current/Resources/gams"

# The benchmark plan: (dataset, mode, ifsub, run_gams).
# run_gams=False -> Python-only timing (no local GAMS NLP fixture / too big to
# rebuild cheaply). Order is smallest-first so a partial run still yields the
# fast rows.
PLAN: list[tuple[str, str, int, bool]] = [
    ("gtap7_3x3", "pure", 0, True),
    ("gtap7_3x3", "altertax", 0, True),
    ("gtap7_3x4", "altertax", 0, True),
    ("gtap7_5x5", "pure", 0, True),
    ("gtap7_5x5", "altertax", 0, True),
    ("gtap7_10x7", "pure", 0, True),
    ("gtap7_10x7", "altertax", 0, True),
    ("gtap7_15x10", "altertax", 0, False),  # Python-only (IPOPT solves it; no local GAMS fixture)
]


def _section(msg: str) -> None:
    print(f"\n{'=' * 70}\n {msg}\n{'=' * 70}", flush=True)


def _stats(samples: list[float]) -> dict[str, float]:
    return {
        "n": len(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "min": min(samples),
        "max": max(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _fmt_stats(label: str, s: dict[str, float]) -> str:
    if not s:
        return f"  {label:<18} (not run)"
    return (f"  {label:<18} n={s['n']}  median={s['median']:.3f}s  "
            f"min={s['min']:.3f}s  max={s['max']:.3f}s  "
            f"mean={s['mean']:.3f}s  stdev={s['stdev']:.3f}s")


def _fixture_gdx(dataset: str, mode: str, ifsub: int) -> Path:
    return NLP_FIXTURES / f"{dataset}_{mode}_ifsub{ifsub}.gdx"


# ---------------------------------------------------------------------------
# Python side (IPOPT NLP solve of the full base->check->shock sequence)
# ---------------------------------------------------------------------------
def _time_python_once(dataset: str, mode: str, ifsub: int, gdx: Path) -> float:
    """Build + seed + solve as NLP; return SOLVE wall-time (seconds).

    Build/seed happen before the clock starts so the timing isolates the solve
    (matching how a user re-solves a warm model), consistent across datasets.
    """
    os.environ["EQUILIBRIA_GTAP_SOLVE_NLP"] = "1"
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_model_multiperiod import (
        GTAPMultiPeriodModel, PERIODS,
    )
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    d = DATASETS_DIR / dataset
    p = GTAPParameters()
    p.load_from_har(basedata_path=d / "basedata.har", sets_path=d / "sets.har",
                    default_path=d / "default.prm", baserate_path=d / "baserate.har")
    rr = list(p.sets.r)[-1]

    if mode == "altertax":
        from equilibria.templates.gtap.altertax import apply_altertax_elasticities
        pa = apply_altertax_elasticities(p, in_place=False)
        ac = GTAPClosureConfig(name="altertax", closure_type="MCP",
                               capital_mobility="mobile", fix_endowments=False,
                               fix_taxes=True, fix_technology=True,
                               if_sub=bool(ifsub), numeraire="pnum")
        solve_mode, holdfix_cd = "altertax", True
    else:  # pure real-CES
        pa = p
        ac = GTAPClosureConfig(name="base", closure_type="MCP",
                               capital_mobility="sluggish", fix_endowments=False,
                               fix_taxes=False, fix_technology=False,
                               if_sub=bool(ifsub), numeraire="pnum")
        solve_mode, holdfix_cd = "gtap", False

    mp = GTAPMultiPeriodModel(pa.sets, pa, ac, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for per in PERIODS:
        mp.build_equations_intra(m, per)
    mp.build_equations_fisher(m)
    m._residual_region = rr
    mp.seed_all_periods(m, gdx)

    t0 = time.perf_counter()
    solve_multiperiod(m, p, ac, ref_gdx=gdx, skip_base_solve=True,
                      mute_welfare=True, seed_from_prior=False,
                      holdfix_cd=holdfix_cd, mode=solve_mode)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# GAMS side (ifMCP=0 + option nlp=ipopt, run locally)
# ---------------------------------------------------------------------------
def _build_nlp_gms(dataset: str, mode: str, ifsub: int) -> Path:
    """Build the bundle .gms and rewrite it to ifMCP=0 + IPOPT (as
    regen_altertax_reference._solve does). Returns the runnable *_nlp.gms path.
    Done ONCE per plan row; the timed runs re-execute this file."""
    if mode == "altertax":
        builder = "build_gtap7_altertax_neos_bundle.py"
        out_dir = ROOT / "output" / f"{dataset}_altertax_neos_bundle"
        gms_stem = f"comp_{dataset}_altertax_neos_ifsub{ifsub}"
    else:
        builder = "build_gtap7_pure_local_bundle.py"
        out_dir = ROOT / "output" / f"{dataset}_pure_local_bundle"
        gms_stem = f"comp_{dataset}_gtap_shock_ifsub{ifsub}"

    subprocess.run(["uv", "run", "python", str(ROOT / "scripts/gtap" / builder),
                    "--dataset", dataset, "--ifsub", str(ifsub)],
                   check=True, cwd=str(ROOT), capture_output=True, text=True)
    gms = out_dir / f"{gms_stem}.gms"
    if not gms.exists():
        raise RuntimeError(f"builder did not produce {gms}")

    text = gms.read_text()
    text = re.sub(r'ifMCP\s+"[^"]*"\s+/ 1 /',
                  'ifMCP       "Set to 1 to solve using MCP"           / 0 /', text)
    for _m in ("gtap", "dynCal", "dynGTAP"):
        text = text.replace(
            f"   solve {_m} using nlp maximizing walras ;",
            f"   option nlp=ipopt;\n   solve {_m} using nlp maximizing walras ;", 1)
    run_gms = gms.with_name(gms.stem + "_nlp.gms")
    run_gms.write_text(text)
    return run_gms


def _time_gams_once(run_gms: Path, tag: str) -> float:
    """Run the ifMCP=0 NLP .gms once; return wall-time (seconds).
    Asserts optimal model status so a silently-broken solve can't pollute the
    timing."""
    out_dir = run_gms.parent
    t0 = time.perf_counter()
    r = subprocess.run([GAMS, run_gms.name, "lo=2", f"gdx=bench_{run_gms.stem}"],
                       cwd=str(out_dir), capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    lst = out_dir / run_gms.with_suffix(".lst").name
    statuses = re.findall(r"MODEL STATUS\s+(\d+)", lst.read_text()) if lst.exists() else []
    bad = [s for s in statuses if s not in ("1", "2")]
    if r.returncode != 0 or bad or not statuses:
        print(f"  GAMS stdout tail: {r.stdout[-1200:]}", file=sys.stderr)
        raise RuntimeError(f"[{tag}] GAMS NLP solve failed "
                           f"(rc={r.returncode}, statuses={statuses})")
    return elapsed


# ---------------------------------------------------------------------------
def _write_timing_csv(path: Path, rows: list[dict], git_sha: str,
                      generated_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "mode", "ifsub", "run", "python_seconds",
              "gams_local_seconds", "git_sha", "generated_at"]
    with path.open("w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({**row, "git_sha": git_sha, "generated_at": generated_at})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=int(os.environ.get("BENCH_RUNS", "5")),
                    help="N timing runs per side (default 5, env BENCH_RUNS)")
    ap.add_argument("--timing-csv", type=Path, required=True,
                    help="Output CSV for per-run NLP wall-time samples")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these dataset names (e.g. gtap7_3x3 gtap7_5x5)")
    ap.add_argument("--no-gams", action="store_true",
                    help="time the Python side only (skip all GAMS runs)")
    args = ap.parse_args()

    n = args.runs
    git_sha = git_short_sha(ROOT)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not Path(GAMS).exists() and not args.no_gams:
        print(f"WARNING: GAMS not found at {GAMS} — running Python-only.",
              file=sys.stderr)
        args.no_gams = True

    plan = [row for row in PLAN
            if args.only is None or row[0] in args.only]

    csv_rows: list[dict] = []
    summary: list[tuple[str, dict, dict]] = []

    for dataset, mode, ifsub, want_gams in plan:
        tag = f"{dataset}/{mode}/ifSUB={ifsub}"
        gdx = _fixture_gdx(dataset, mode, ifsub)
        run_gams = want_gams and not args.no_gams and gdx.exists()
        _section(f"{tag}  (Python{'+GAMS' if run_gams else ' only'}, N={n})")

        # --- GAMS side (build once, warm-up, then N timed runs) ---
        gams_times: list[float] = []
        if run_gams:
            try:
                gms = _build_nlp_gms(dataset, mode, ifsub)
                t_warm = _time_gams_once(gms, tag)
                print(f"  GAMS warm-up: {t_warm:.3f}s (discarded)", flush=True)
                for i in range(n):
                    t = _time_gams_once(gms, tag)
                    gams_times.append(t)
                    print(f"  GAMS run {i+1}/{n}: {t:.3f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  GAMS side FAILED for {tag}: {e}", file=sys.stderr)
                gams_times = []

        # --- Python side (warm-up + N timed runs) ---
        if not gdx.exists() and mode != "altertax":
            # Python needs a seed GDX; without a fixture we can only time
            # datasets whose seed we can still supply. 15x10 uses its committed
            # MCP fixture as the seed even though the GAMS-NLP pair is absent.
            pass
        py_seed = gdx if gdx.exists() else _find_seed(dataset, mode, ifsub)
        py_times: list[float] = []
        if py_seed is not None:
            try:
                t_warm = _time_python_once(dataset, mode, ifsub, py_seed)
                print(f"  Python warm-up: {t_warm:.3f}s (discarded)", flush=True)
                for i in range(n):
                    t = _time_python_once(dataset, mode, ifsub, py_seed)
                    py_times.append(t)
                    print(f"  Python run {i+1}/{n}: {t:.3f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  Python side FAILED for {tag}: {e}", file=sys.stderr)
                py_times = []
        else:
            print(f"  no seed GDX for {tag} — Python side skipped", file=sys.stderr)

        for i in range(max(len(py_times), len(gams_times))):
            csv_rows.append({
                "dataset": dataset, "mode": mode, "ifsub": ifsub, "run": i + 1,
                "python_seconds": f"{py_times[i]:.6f}" if i < len(py_times) else "",
                "gams_local_seconds": f"{gams_times[i]:.6f}" if i < len(gams_times) else "",
            })
        summary.append((tag,
                        _stats(py_times) if py_times else {},
                        _stats(gams_times) if gams_times else {}))
        # incremental write so a long run is never lost
        _write_timing_csv(args.timing_csv, csv_rows, git_sha, generated_at)

    _section("NLP Benchmark Summary")
    for tag, py_s, gm_s in summary:
        print(f"\n  {tag}")
        print(_fmt_stats("Python (IPOPT)", py_s))
        print(_fmt_stats("GAMS local NLP", gm_s))
        if py_s and gm_s and gm_s["median"] > 0:
            print(f"    Median ratio Python / GAMS-local: "
                  f"{py_s['median'] / gm_s['median']:.3f}x")
    print(f"\n  Timing CSV: {args.timing_csv}")
    return 0


def _find_seed(dataset: str, mode: str, ifsub: int) -> Path | None:
    """Locate a seed GDX for a dataset that has no NLP fixture (e.g. 15x10).
    Falls back to the committed MCP/altertax fixture used by the parity gate."""
    for cand in (
        ROOT / "tests" / "fixtures" / "gtap7_altertax" / dataset / f"out_altertax_ifsub{ifsub}.gdx",
        ROOT / "tests" / "fixtures" / "gtap7" / dataset / f"out_gtap_shock_ifsub{ifsub}.gdx",
    ):
        if cand.exists():
            return cand
    return None


if __name__ == "__main__":
    raise SystemExit(main())
