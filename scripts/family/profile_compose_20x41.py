"""Profile compose_block_state on GTAP datasets.

Times compose_block_state calibration and writes the top 25 functions by
cumulative time to a file. Uses cProfile and disables GC during measurement.

    python scripts/family/profile_compose_20x41.py [--dataset gtap7_20x41] [--out FILE]
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _closure():
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    # The closure tests/templates/gtap/test_gtap7_gempack_parity.py::_solve_shock uses.
    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=True,
        savf_flag="capFix",
        numeraire="pnum",
        va_subsidy_basis="gempack",
    )


def profile_compose(dataset: str) -> tuple[str, float]:
    """Profile compose_block_state and return formatted output and wall time."""
    import equilibria.templates.gtap.gtap_block_model as gbm
    from equilibria.sources import dataset_dir
    from equilibria.templates.gtap import GTAPParameters

    # Load params
    d = dataset_dir(dataset)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )

    # Get residual region
    rr = list(p.sets.r)[-1]
    closure = _closure()

    # Profile compose_block_state
    profiler = cProfile.Profile()
    t_start = time.perf_counter()
    profiler.enable()
    gbm.compose_block_state(p, p.sets, closure, rr)
    profiler.disable()
    t_end = time.perf_counter()
    wall_time = t_end - t_start

    # Format stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats("cumulative")
    ps.print_stats(25)

    return s.getvalue(), wall_time


if __name__ == "__main__":
    gc.disable()
    ap = argparse.ArgumentParser(
        description="Profile compose_block_state on GTAP datasets"
    )
    ap.add_argument("--dataset", default="gtap7_20x41", help="Dataset name")
    ap.add_argument("--out", type=Path, help="Output file for profile results")
    a = ap.parse_args()

    stats_output, wall_time = profile_compose(a.dataset)

    # Build final output: first line with dataset and wall time, then stats
    first_line = f"Dataset: {a.dataset}, Wall time: {wall_time:.2f}s\n"
    final_output = first_line + stats_output

    if a.out:
        a.out.write_text(final_output)
    else:
        print(final_output)
