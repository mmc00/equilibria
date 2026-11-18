"""Record the settle outputs the switch to the core must keep (to ``REL_TOL``).

Writes ``tests/blocks/gtap/golden/calibrate_base_main.json``: for each case, the
``FactorBlock.calibrate_base`` settled seed (``{var: {body: value}}``) and the capFlex
``risk`` of the capFix twin-solve (``_calibrate_capflex_risk`` with no GDX), every
value as ``repr(float)``. Run it on the commit whose numbers are the reference
(``main`` before PR 3a of gtap-on-core); the seed cache is disabled.

    PYTHONPATH=$PWD/src:$PWD python scripts/dev/record_calibrate_base.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/blocks/gtap/golden/calibrate_base_main.json"

# (case id, dataset, ifSUB)
SETTLE_CASES = (
    ("gtap7_3x3-ifsub0", "gtap7_3x3", 0),
    ("gtap7_3x3-ifsub1", "gtap7_3x3", 1),
    ("nus333-ifsub0", "nus333", 0),
)
CAPFLEX_CASES = (("gtap7_3x3-ifsub0", "gtap7_3x3", 0),)


def _params(dataset: str):
    from equilibria.sources import dataset_dir
    from equilibria.templates.gtap import GTAPParameters

    d = dataset_dir(dataset)
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    return p


def closure(ifsub: int, savf_flag: str = "capFix"):
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=bool(ifsub),
        savf_flag=savf_flag,
        numeraire="pnum",
    )


def _settle(dataset: str, ifsub: int) -> dict:
    from equilibria.blocks.gtap.factor import FactorBlock
    from equilibria.templates.gtap.gtap_block_model import build_block_model

    p = _params(dataset)
    rr = list(p.sets.r)[-1]
    cl = closure(ifsub)
    m, _ = build_block_model(p, p.sets, cl, rr)
    seed = FactorBlock(sets=p.sets, params=p).calibrate_base(
        p, p.sets, cl, rr, ref_gdx=None, model=m
    )
    return {
        var: {repr(body): repr(float(v)) for body, v in cells.items()}
        for var, cells in seed.items()
    }


def _risk(dataset: str, ifsub: int) -> dict:
    from equilibria.templates.gtap.gtap_block_model import _calibrate_capflex_risk

    p = _params(dataset)
    rr = list(p.sets.r)[-1]
    risk = _calibrate_capflex_risk(p, p.sets, closure(ifsub, "capFlex"), rr, ref_gdx=None)
    if not risk:
        raise SystemExit(f"{dataset}: the capFix twin-solve returned no risk")
    return {r: repr(float(v)) for r, v in risk.items()}


REL_TOL = 1e-12


def mismatches(got, want, path: tuple = ()) -> list[str]:
    """Where a recomputed record leaves the golden: a key on one side only, or a value
    off by more than ``REL_TOL`` relative. The golden was recorded on one machine;
    another CPU or BLAS moves the last bits (~1e-15), a real change moves far more."""
    if isinstance(want, dict) and isinstance(got, dict):
        out = [f"{(*path, k)}: only in golden" for k in want.keys() - got.keys()]
        out += [f"{(*path, k)}: not in golden" for k in got.keys() - want.keys()]
        for k in want.keys() & got.keys():
            out += mismatches(got[k], want[k], (*path, k))
        return out
    g, w = float(got), float(want)
    if abs(g - w) > REL_TOL * max(abs(g), abs(w)):
        return [f"{path}: {g!r} != {w!r}"]
    return []


def main() -> None:
    os.environ["EQUILIBRIA_SEED_CACHE_DISABLE"] = "1"
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    out = {
        "commit": head,
        "settle": {cid: _settle(ds, i) for cid, ds, i in SETTLE_CASES},
        "capflex_risk": {cid: _risk(ds, i) for cid, ds, i in CAPFLEX_CASES},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    n = sum(len(c) for s in out["settle"].values() for c in s.values())
    print(f"wrote {OUT.relative_to(ROOT)}: {n} settled cells, commit {head}")


if __name__ == "__main__":
    main()
