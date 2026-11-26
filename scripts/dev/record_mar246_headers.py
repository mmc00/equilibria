"""Record the AlterTax outputs MAR-246 speeds up, on nus333 (oracle written on main).

MAR-246 rewrites, without changing a number, ``post_solve.mp_margin_params``, the
trade block of ``post_solve.recompute_ifsub_report_vars``, the trade prices of
``altertax/updated.py`` and ``decomp._require_unchanged``. Run on ``main`` before the
change, this writes per case: the margin params, the ifSUB report variables at
``shock``, the value headers of ``check`` and ``shock`` (``headers_at``), the percent
changes (``pct_headers``), the volume headers and the decomposition headers (the path's
when the case has ``decomp_steps``). ``tests/family/test_mar246_headers_golden.py``
compares the current checkout against it. Each case runs in its own child process with
the environment of the ALTERTAX tests (Newton-TR + MUMPS, direct linear solve).

usage: python scripts/dev/record_mar246_headers.py --out FILE [--only CASE ...]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CASES: dict[str, dict] = {
    "nominal": {},
    "ratio_k2": {"trade_balance": "ratio", "decomp_steps": 2},
}

REPORT_VARS = ("pfa", "pfy", "pp_rai", "pwmg", "pefob", "pmcif", "pm")

_CHILD = r"""
import json, sys
sys.path[:0] = [{src!r}, {root!r}]
from equilibria._local_refs import nus333_dir
from equilibria.family.altertax.decomp import decomp_headers
from equilibria.family.altertax.pct import pct_headers
from equilibria.family.altertax.run import solve_altertax
from equilibria.family.altertax.updated import headers_at
from equilibria.family.altertax.volume import volume_headers
from equilibria.family.post_solve import mp_margin_params
from equilibria.family.sp_arrays import period_table

run = solve_altertax(nus333_dir(), cache_root={cache!r}, **{kw!r})
mp = run.result.mp
arrays = lambda hs: {{n: h.array.tolist() for n, h in hs.items()}}
margins = mp_margin_params(run.params, run.params.sets)
report = {{}}
for v in {report_vars!r}:
    if v in mp.vars:
        var = mp.vars[v]
        report[v] = {{"|".join(k): float(var.value[i])
                     for k, i in period_table(var, "shock").items()}}
heads = run.decomp_path.headers if run.decomp_path is not None else decomp_headers(run)
out = {{
    "codes": {{k: int(v) for k, v in run.codes.items()}},
    "margins": {{n: {{"|".join(k): float(x) for k, x in d.items()}}
                for n, (_, d) in margins.items()}},
    "report_vars": report,
    "headers": {{p: arrays(headers_at(run, p)) for p in ("check", "shock")}},
    "pct": arrays(pct_headers(run)),
    "volume": arrays(volume_headers(run)),
    "decomp": arrays(heads),
}}
print("RESULT " + json.dumps(out))
"""


def _newton_env() -> dict[str, str]:
    sys.path.insert(0, str(ROOT))
    from tests.family._prep import _newton_mumps

    return {**_newton_mumps(), "EQUILIBRIA_GTAP_TR_LINSOLVE": "direct"}


def run_case(name: str, root: Path) -> dict:
    with tempfile.TemporaryDirectory() as cache:
        code = _CHILD.format(
            src=str(root / "src"),
            root=str(root),
            kw=dict(CASES[name]),
            cache=cache,
            report_vars=REPORT_VARS,
        )
        env = {**os.environ, **_newton_env(), "EQUILIBRIA_FAMILY_CACHE": cache}
        env["PYTHONPATH"] = f"{root / 'src'}:{root}"
        p = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
    line = next((s for s in p.stdout.splitlines() if s.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(f"{name}: exit {p.returncode}\n{p.stderr[-3000:]}")
    return json.loads(line[len("RESULT ") :])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    names = a.only or list(CASES)
    out = {}
    for name in names:
        print(f"case {name}", flush=True)
        out[name] = run_case(name, a.root.resolve())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
