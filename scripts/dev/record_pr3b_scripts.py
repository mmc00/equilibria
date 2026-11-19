"""Record what the scripts moved by PR 3b of gtap-on-core solve, on the old driver.

Writes ``tests/scripts/golden/pr3b_scripts_main.json.gz`` (gzip: ~1.8 MB of JSON): per case, the codes of the
old driver (``{period: int}``) and every Var level of the solved model
(``tests._golden.model_levels``). The scripts' own outputs are rounded summaries;
the model levels are the oracle. Run it on ``main`` before PR 3b: each case runs in
a fresh process (some scripts set solver env vars at import) with
``solve_multiperiod`` wrapped to capture the model it solves.

    PYTHONPATH=$PWD/src:$PWD python scripts/dev/record_pr3b_scripts.py
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/scripts/golden/pr3b_scripts_main.json.gz"
REF = "data/gtap7/gtap7_3x3/gams_mcp_gtap_ifsub{}_fbep-header/out.gdx"

# case id -> python run inside the child, after the capture is installed
CASES = {
    "gempack_blocks[gtap7_3x3-default]": (
        "import measure_gempack_blocks as s\n"
        f"s._land_response('gtap7_3x3', False, ROOT / '{REF.format(1)}')"
    ),
    "gempack_blocks[gtap7_3x3-calibrated]": (
        "import measure_gempack_blocks as s\n"
        f"s._land_response('gtap7_3x3', True, ROOT / '{REF.format(1)}')"
    ),
    "ifsub_equivalence[gtap7_3x3-ifsub0]": (
        "import verify_ifsub_equivalence as s\ns._solve('gtap7_3x3', 0)"
    ),
    "ifsub_equivalence[gtap7_3x3-ifsub1]": (
        "import verify_ifsub_equivalence as s\ns._solve('gtap7_3x3', 1)"
    ),
    "seed_and_solve_gtap[gtap7_3x3-ifsub0-shock]": (
        "import seed_and_solve as s\n"
        f"s.run_seed_and_solve_gtap('gtap7_3x3', ROOT / '{REF.format(0)}', 'shock', 5, False)"
    ),
    "dump_py[gtap7_3x3]": (
        "import runpy\n"
        "(ROOT / 'output').mkdir(exist_ok=True)  # the script writes its .gms there\n"
        "sys.argv = ['dump_py_solution_gdx.py', 'gtap7_3x3']\n"
        "runpy.run_path(str(ROOT / 'scripts/gtap/dump_py_solution_gdx.py'), run_name='record')"
    ),
    "demo_gtap[nus333]": (
        "import runpy\n"
        "runpy.run_path(str(ROOT / 'examples/demo_gtap_pep.py'), run_name='record')"
    ),
}

CHILD = """
import json, os, sys
from pathlib import Path
ROOT = Path({root!r})
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts/gtap')]
os.environ['EQUILIBRIA_SEED_CACHE_DISABLE'] = '1'
from equilibria._local_refs import path_capi_src
sys.path.insert(0, str(path_capi_src()))
import equilibria.templates.gtap.gtap_multiperiod_driver as drv
from tests._golden import model_levels
_solve = drv.solve_multiperiod
seen = []
def capture(m, *a, **k):
    res = _solve(m, *a, **k)
    seen.append((m, res))
    return res
drv.solve_multiperiod = capture
{body}
m, res = seen[-1]
json.dump({{'codes': {{k: int(r['code']) for k, r in res.items()}},
           'levels': model_levels(m), 'solves': len(seen)}}, open({out!r}, 'w'))
"""


def record_case(case: str, body: str, tmp: Path) -> dict:
    out = tmp / "case.json"
    child = CHILD.format(root=str(ROOT), body=body, out=str(out))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", child], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{case}: rc={r.returncode}\n{r.stderr[-3000:]}")
    return json.loads(out.read_text())


def main() -> None:
    import tempfile

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    out: dict = {"commit": head}
    with tempfile.TemporaryDirectory() as d:
        for case, body in CASES.items():
            out[case] = record_case(case, body, Path(d))
            n = sum(len(c) for c in out[case]["levels"].values())
            print(f"{case}: codes {out[case]['codes']}, {n} cells, "
                  f"{out[case]['solves']} solve(s)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n"
    OUT.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KiB), commit {head}")


if __name__ == "__main__":
    main()
