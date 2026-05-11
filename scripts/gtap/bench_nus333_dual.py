"""NUS333 dual-reference benchmark: parity vs GAMS local + GAMS NEOS, with wall-time.

Pipeline:
  1. Build the 3-GDX bundle from NUS333 HARs via the native pure-Python writer.
  2. Run comp_nus333.gms locally N times (timing); keep last COMP.gdx.
  3. Run equilibria NUS333 (base + shock) N times in Python (timing); keep last models.
  4. Diff Python last-run vs (a) NEOS reference GDX, (b) local COMP.gdx.
  5. Emit:
       - nus333.csv         (Python vs NEOS — overwrites existing)
       - nus333_local.csv   (Python vs GAMS local)
       - nus333_timing.csv  (per-run wall-times)

Run:
    .venv/bin/python scripts/gtap/bench_nus333_dual.py --runs 5 \
        --neos-csv docs/site/_data/benchmarks/nus333.csv \
        --local-csv docs/site/_data/benchmarks/nus333_local.csv \
        --timing-csv docs/site/_data/benchmarks/nus333_timing.csv
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/Users/marmol/proyectos/path-capi-python/src")
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from _diff_core import (
    list_populated_vars, diff_phase_rows, write_csv, git_short_sha, build_derived,
)
from diff_nus333_full import _nus333_key_remap

NUS333_HAR = Path(os.environ.get("EQUILIBRIA_NUS333_DIR", "/Users/marmol/Downloads/10284"))
NEOS_GDX = ROOT / "output/nus333_neos/out.gdx"
GAMS = "/Library/Frameworks/GAMS.framework/Versions/53/Resources/gams"
SCRIPTS_DIR = ROOT / "src/equilibria/templates/reference/gtap/scripts"
LOCAL_BUNDLE = Path("/tmp/nus333_bench_local")


def _section(msg: str) -> None:
    print(f"\n{'═' * 70}\n {msg}\n{'═' * 70}", flush=True)


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
    return (f"  {label:<14} n={s['n']}  median={s['median']:.3f}s  "
            f"min={s['min']:.3f}s  max={s['max']:.3f}s  "
            f"mean={s['mean']:.3f}s  stdev={s['stdev']:.3f}s")


def build_bundle_once() -> None:
    from equilibria.babel.har_to_gdx import write_nus333_gdx_bundle
    LOCAL_BUNDLE.mkdir(parents=True, exist_ok=True)
    for p in LOCAL_BUNDLE.glob("*"):
        p.unlink()
    write_nus333_gdx_bundle(NUS333_HAR, LOCAL_BUNDLE)


def run_gams_local() -> tuple[Path, float]:
    """Run comp_nus333.gms once; return (COMP.gdx path, wall-time seconds)."""
    t0 = time.perf_counter()
    res = subprocess.run(
        [GAMS, "comp_nus333.gms",
         f"--inDir={LOCAL_BUNDLE}",
         f"--BaseName=nus333",
         f"--outDir={LOCAL_BUNDLE}",
         "lo=2"],
        cwd=SCRIPTS_DIR,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print("GAMS stdout (tail):", res.stdout[-1500:])
        print("GAMS stderr (tail):", res.stderr[-500:])
        raise RuntimeError(f"comp_nus333.gms failed (rc={res.returncode})")
    comp = LOCAL_BUNDLE / "COMP.gdx"
    if not comp.exists():
        raise RuntimeError(f"COMP.gdx not produced at {comp}")
    return comp, elapsed


def run_python_once() -> tuple[dict, float, float, float]:
    """Run Python NUS333 base + shock. Return (models, total_s, res_base, res_shock)."""
    from compare_nus333_vs_neos import _solve, _apply_tariff_shock, _copy_var_levels
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_model_equations import GTAPModelEquations
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    params = GTAPParameters()
    params.load_from_har(
        basedata_path=NUS333_HAR / "basedata.har",
        sets_path=NUS333_HAR / "sets.har",
        default_path=NUS333_HAR / "default.prm",
        baserate_path=NUS333_HAR / "baserate.har",
    )
    closure = GTAPClosureConfig(if_sub=False, rmuv=("ROW",), imuv=("MFG",))

    t0 = time.perf_counter()
    builder_b = GTAPModelEquations(params.sets, params, residual_region="ROW", closure=closure)
    m_b = builder_b.build_model()
    r_b = _solve(m_b, params, label="base")
    res_b = float(getattr(r_b, "residual", 0.0) or 0.0)

    _apply_tariff_shock(params, factor=1.10)
    builder_s = GTAPModelEquations(
        params.sets, params, residual_region="ROW", closure=closure, t0_snapshot=m_b,
    )
    m_s = builder_s.build_model()
    _copy_var_levels(m_b, m_s)
    r_s = _solve(m_s, params, label="shock")
    res_s = float(getattr(r_s, "residual", 0.0) or 0.0)
    elapsed = time.perf_counter() - t0
    return {"base": m_b, "shock": m_s}, elapsed, res_b, res_s


def _diff_against(
    *, gdx: Path, dataset: str, models: dict, residuals: dict[str, float],
    solve_seconds: dict[str, float], tol_rel: float, tol_abs: float,
    git_sha: str, generated_at: str,
) -> list[dict]:
    var_names = list_populated_vars(gdx)
    print(f"  Populated vars in {gdx.name}: {len(var_names)}")
    rows: list[dict] = []
    for phase in ("base", "shock"):
        m_py = models[phase]
        residual = residuals[phase]
        secs = solve_seconds[phase]
        phase_rows, agg = diff_phase_rows(
            dataset=dataset, phase=phase, var_names=var_names,
            gdx_path=gdx, model_py=m_py,
            tol_rel=tol_rel, tol_abs=tol_abs,
            residual=residual, git_sha=git_sha, generated_at=generated_at,
            derived=build_derived(m_py), key_remap=_nus333_key_remap,
            solve_seconds=secs,
        )
        rows.extend(phase_rows)
        cov = (agg["cells_match"] / agg["cells_total"] * 100.0) if agg["cells_total"] else 0.0
        print(f"  phase={phase:<5s} cells={agg['cells_total']:>4d}  "
              f"match={agg['cells_match']:>4d}  diverge={agg['cells_diverge']:>3d}  "
              f"missing={agg['cells_missing']:>3d}  rate={cov:6.2f}%")
    return rows


def _write_timing_csv(
    path: Path, *, runs_python: list[float], runs_gams: list[float],
    git_sha: str, generated_at: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(len(runs_python), len(runs_gams))
    fields = ["dataset", "run", "python_seconds", "gams_local_seconds",
              "git_sha", "generated_at"]
    with path.open("w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(n):
            w.writerow({
                "dataset": "nus333",
                "run": i + 1,
                "python_seconds": f"{runs_python[i]:.6f}" if i < len(runs_python) else "",
                "gams_local_seconds": f"{runs_gams[i]:.6f}" if i < len(runs_gams) else "",
                "git_sha": git_sha,
                "generated_at": generated_at,
            })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=int(os.environ.get("BENCH_RUNS", "5")),
                    help="N timing runs per side (default 5, env BENCH_RUNS)")
    ap.add_argument("--tol-rel", type=float, default=1e-3)
    ap.add_argument("--tol-abs", type=float, default=1e-6)
    ap.add_argument("--neos-csv", type=Path, required=True,
                    help="Output CSV for Python-vs-NEOS parity")
    ap.add_argument("--local-csv", type=Path, required=True,
                    help="Output CSV for Python-vs-GAMS-local parity")
    ap.add_argument("--timing-csv", type=Path, required=True,
                    help="Output CSV for per-run wall-time samples")
    args = ap.parse_args()

    n = args.runs
    git_sha = git_short_sha(ROOT)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not NEOS_GDX.exists():
        print(f"ERROR: NEOS reference not found: {NEOS_GDX}", file=sys.stderr)
        return 2

    _section(f"1/5  Build HAR→GDX bundle + warm-up GAMS")
    build_bundle_once()
    _, t_warm_gams = run_gams_local()
    print(f"  Warm-up GAMS wall-time: {t_warm_gams:.3f}s (discarded)")

    _section(f"2/5  Timing GAMS local  (N={n} runs)")
    gams_times: list[float] = []
    comp_gdx: Path | None = None
    for i in range(n):
        comp_gdx, t = run_gams_local()
        gams_times.append(t)
        print(f"  run {i+1}/{n}:  {t:.3f}s")
    gams_s = _stats(gams_times)
    assert comp_gdx is not None

    _section(f"3/5  Timing Python  (warm-up + N={n} runs)")
    _, t_warm_py, _, _ = run_python_once()
    print(f"  Warm-up Python: {t_warm_py:.3f}s (discarded)")
    py_times: list[float] = []
    last_models = None
    last_res_b = last_res_s = 0.0
    for i in range(n):
        models, t, res_b, res_s = run_python_once()
        py_times.append(t)
        last_models = models
        last_res_b, last_res_s = res_b, res_s
        print(f"  run {i+1}/{n}:  {t:.3f}s  (res_b={res_b:.2e}  res_s={res_s:.2e})")
    py_s = _stats(py_times)
    assert last_models is not None

    # Approximate per-phase solve_seconds: split the median run roughly in half.
    # Both phases are similar in size; the timing CSV is the authoritative source.
    half = py_s["median"] / 2.0
    solve_seconds = {"base": half, "shock": half}
    residuals = {"base": last_res_b, "shock": last_res_s}

    _section("4/5  Diff Python vs NEOS reference")
    neos_rows = _diff_against(
        gdx=NEOS_GDX, dataset="nus333", models=last_models,
        residuals=residuals, solve_seconds=solve_seconds,
        tol_rel=args.tol_rel, tol_abs=args.tol_abs,
        git_sha=git_sha, generated_at=generated_at,
    )

    _section("5/5  Diff Python vs GAMS local COMP.gdx")
    local_rows = _diff_against(
        gdx=comp_gdx, dataset="nus333", models=last_models,
        residuals=residuals, solve_seconds=solve_seconds,
        tol_rel=args.tol_rel, tol_abs=args.tol_abs,
        git_sha=git_sha, generated_at=generated_at,
    )

    write_csv(args.neos_csv, neos_rows)
    write_csv(args.local_csv, local_rows)
    _write_timing_csv(args.timing_csv,
                      runs_python=py_times, runs_gams=gams_times,
                      git_sha=git_sha, generated_at=generated_at)

    _section("Benchmark Summary")
    print(_fmt_stats("Python",     py_s))
    print(_fmt_stats("GAMS local", gams_s))
    ratio = py_s["median"] / gams_s["median"] if gams_s["median"] > 0 else 0.0
    print(f"\n  Median ratio Python / GAMS-local: {ratio:.3f}x")
    print(f"\n  NEOS parity CSV:  {args.neos_csv}")
    print(f"  Local parity CSV: {args.local_csv}")
    print(f"  Timing CSV:       {args.timing_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
