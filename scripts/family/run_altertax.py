"""GEMPACK-faithful ALTERTAX on the JAX "family" route: one command, five output files.

Usage (the worktree's python, MUMPS or not; the solver settings default to the ones the
tests use and any ``EQUILIBRIA_GTAP_*`` variable already set wins)::

    python scripts/family/run_altertax.py <dataset-dir> [--prm altertax.prm] [--out DIR]
                                          [--compare GEMPACK_DIR]

Solves the check and shock periods of the book shock (``ALTAX_SHOCK``: ``to(AGR,AGR,USA)``
of Burfisher's ``altax.exp``) on ``<dataset-dir>`` (a GTAP 7 package: ``basedata.har``,
``sets.har``, the ``--prm`` file, ``baserate.har``) and writes into ``--out`` (default
``./altertax_out``):

- ``updated.har``, ``summary.har``, ``decomp.har``, ``volume.har``, ``pct.har``;
- ``run.json``: ``codes`` per period, ``nit``, ``residual``, ``minutes`` and ``solves``
  (every Newton solve: code, iterations, residual, null pivots, GMIN retries; always
  written).

If any period does not solve (code != 1), or any Newton solve of the run failed, only
``run.json`` is written, nothing is compared and the exit status is 3, the message naming
the period or the solve. ``compare.json`` lists the solves too.

``--compare GEMPACK_DIR`` checks every output against a GEMPACK run folder and writes
``compare.json`` (per output: max gap, worst 10 cells, the gate, pass/fail, the known
misses applied). The folder holds GEMPACK's own file names (``updated.har``,
``summary.har``, ``decomp.har``, ``volume.har``, ``solution.sl4`` and ``sl4levels.har``;
``sl4dump.har`` is optional and not read yet) or the burfisher fixture names
(``altax_*.har``, ``altax.upd``, ``altax.sl4.gz``, ``sl4levels_altax.har``). A missing file
is reported, naming it, BEFORE any solve (exit 2). Exit 1 on a failed gate.

Exit status: 0 ok, 1 failed gate, 2 usage or missing file, 3 not solved.

Extension point (Phase C): :func:`load_shock` is the only place that picks the shock; a
``--shock-file`` option would be read there. It is not implemented.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def _solver_env() -> None:
    """The Newton settings of the tests (``NEWTON_MUMPS`` with the direct linear solver);
    a variable already in the environment is kept."""
    path = ROOT / "scripts" / "gtap" / "perf_budget_20x41.py"
    spec = importlib.util.spec_from_file_location("perf_budget_20x41", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in {**mod.NEWTON_MUMPS, "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}.items():
        os.environ.setdefault(k, v)


OUTPUTS = ("updated.har", "summary.har", "decomp.har", "volume.har", "pct.har")


def _clean(out: Path) -> None:
    """Remove the files an earlier run left in ``out`` (outputs and ``compare.json``)."""
    for name in (*OUTPUTS, "compare.json"):
        (out / name).unlink(missing_ok=True)


def _write_json(path: Path, obj: dict) -> None:
    """Valid JSON: a non-finite number (an infinite gap) is written as ``null``."""
    from equilibria.family.altertax.gates import json_safe

    path.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False) + "\n")


def _print_solves(solves: list[dict]) -> None:
    """One line per Newton solve; a failed one is marked FAILED."""
    for r in solves:
        print(
            f"  solve {r['solve']}: code {r['code']} nit {r['nit']} residual "
            f"{r['residual']} null pivots {r['null_pivots']} gmin retries "
            f"{r['gmin_retries']}" + ("" if int(r["code"]) == 1 else "  FAILED")
        )


def load_shock(args: argparse.Namespace) -> dict:
    """The shock of the run: Burfisher's ``altax.exp`` (nus333). Phase C: a
    ``--shock-file`` would be parsed here and returned instead."""
    from equilibria.family.altertax.run import ALTAX_SHOCK

    return ALTAX_SHOCK


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_altertax.py", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("dataset_dir", type=Path, help="GTAP 7 package folder (nus333)")
    ap.add_argument(
        "--prm", default="altertax.prm", help="parameter file (default %(default)s)"
    )
    ap.add_argument(
        "--out", type=Path, default=Path("altertax_out"), help="output folder"
    )
    ap.add_argument(
        "--compare",
        type=Path,
        metavar="GEMPACK_DIR",
        help="GEMPACK run folder to check against",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ds = args.dataset_dir
    if not (ds / "basedata.har").exists() or not (ds / args.prm).exists():
        print(f"error: {ds} lacks basedata.har or {args.prm}", file=sys.stderr)
        return 2
    _solver_env()
    from equilibria.family.altertax import gates
    from equilibria.family.altertax.decomp import write_decomp_har
    from equilibria.family.altertax.pct import write_pct_har
    from equilibria.family.altertax.run import solve_altertax
    from equilibria.family.altertax.summary import write_summary_har
    from equilibria.family.altertax.updated import write_updated_har
    from equilibria.family.altertax.volume import write_volume_har

    if args.compare is not None:
        try:  # fail before the solve, not after it
            gates.gempack_files(args.compare)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    shock = load_shock(args)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    info: dict = {
        "dataset": str(ds),
        "prm": args.prm,
        "shock": {
            k: {",".join(c): v for c, v in cells.items()} for k, cells in shock.items()
        },
    }
    try:
        run = solve_altertax(ds, prm=args.prm, shock=shock)
    except Exception as e:  # noqa: BLE001 - any failure of the solve is "not solved"
        _clean(out)
        info |= {"codes": {}, "solved": False, "error": f"{type(e).__name__}: {e}"}
        _write_json(out / "run.json", info)
        print(f"error: ALTERTAX did not solve: {info['error']}", file=sys.stderr)
        return 3
    results = run.result.results
    failed = run.failed_solves()
    info |= {
        "codes": run.codes,
        "nit": {k: results[k].get("nit") for k in run.codes},
        "residual": {k: results[k].get("residual") for k in run.codes},
        "minutes": run.minutes,
        "solved": all(c == 1 for c in run.codes.values()) and not failed,
        # every Newton solve: code, iterations, null pivots, GMIN retries
        "solves": run.solve_records(),
    }
    _clean(out)  # nothing stale from an earlier run survives this one
    _write_json(out / "run.json", info)
    _print_solves(info["solves"])
    bad = [k for k, c in run.codes.items() if c != 1]
    if bad or failed:
        what = (
            f"period {bad[0]!r} has code {run.codes[bad[0]]}"
            if bad
            else f"solve {failed[0]['solve']!r} has code {failed[0]['code']}"
        )
        print(
            f"error: ALTERTAX did not solve, {what} (codes {run.codes}); no outputs "
            f"written, see {out / 'run.json'}",
            file=sys.stderr,
        )
        return 3
    write_updated_har(run, out / "updated.har")
    write_summary_har(run, out / "summary.har")
    write_decomp_har(run, out / "decomp.har")
    write_volume_har(run, out / "volume.har")
    write_pct_har(run, out / "pct.har")
    print(f"solved in {run.minutes:.2f} min, codes {run.codes}; outputs in {out}")
    if args.compare is None:
        return 0
    cmp = gates.compare_all(run, args.compare)
    cmp["gempack_dir"] = str(args.compare)
    _write_json(out / "compare.json", cmp)
    for o in cmp["outputs"]:
        print(
            f"  {o['output']:8s} max {o['max_kind'].split()[0]} gap {o['max']:.3e}  "
            f"{'PASS' if o['pass'] else 'FAIL'}"
        )
    print(f"  solves   {'PASS' if cmp['solves_pass'] else 'FAIL'}")
    if not cmp["pass"]:
        print("error: a gate failed, see compare.json", file=sys.stderr)
        return 1
    return 0


def _exit_now(code: int) -> None:
    """End the command without the interpreter's teardown: every output is written
    and closed by now; freeing the run's arrays (~6 GB on 20x41) took 28-51 s. Only
    here, at the entry point; ``main`` returns normally to any other caller."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    _exit_now(main())
