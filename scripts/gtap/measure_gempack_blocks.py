"""F3.5 against-GEMPACK measurement: the specific-factor (Land) price response.

Builds the block model in default OR base-calibrated mode, solves base→(check)→shock,
reads the Land price response (shock vs base), and compares it to GEMPACK's own
%-change from the sl4dump fixture (pfe[Land,Food,EU_28]).

The point of F3.5: in default mode the response is contaminated by the check
re-settlement (~-18%); in base-calibrated mode it is the clean shock response
(~-3%), close to GEMPACK's -2.68%.

Usage:
    uv run python scripts/gtap/measure_gempack_blocks.py --dataset gtap7_3x3
    uv run python scripts/gtap/measure_gempack_blocks.py --dataset gtap7_3x3 --base-calibrated
    # add --gempack-har <sl4dump.har> to compare vs GEMPACK (else SKIPs that part)

Emits one JSON line to stdout; all logs to stderr.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))
_PATH_CAPI = Path("/Users/marmol/proyectos/path-capi-python/src")
if _PATH_CAPI.exists() and str(_PATH_CAPI) not in sys.path:
    sys.path.insert(0, str(_PATH_CAPI))

DATASETS = ROOT / "datasets"
# specific factor + the activity that actually uses it (Land→Food); the residual
# region rres is ROW, EU_28 is the sharpest re-settlement cell.
LAND_CELL = ("EU_28", "Land")


def _land_response(dataset: str, base_calibrated: bool, ref_gdx: Path) -> dict:
    from pyomo.environ import value as V

    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_block_model import build_block_model
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    d = DATASETS / dataset
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=d / "basedata.har",
        sets_path=d / "sets.har",
        default_path=d / "default.prm",
        baserate_path=d / "baserate.har",
    )
    rr = list(p.sets.r)[-1]
    gc = GTAPClosureConfig(
        name="base", closure_type="MCP", capital_mobility="sluggish",
        fix_endowments=False, fix_taxes=False, fix_technology=False,
        if_sub=False, numeraire="pnum",
    )
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        m, mp = build_block_model(p, p.sets, gc, rr, base_calibrated=base_calibrated)
        if not base_calibrated:
            mp.seed_all_periods(m, ref_gdx)
        res = solve_multiperiod(
            m, p, gc, ref_gdx=ref_gdx, skip_base_solve=True, mode="gtap"
        )
    r, f = LAND_CELL
    base = float(V(m.pft[r, f, "base"]))
    shock = float(V(m.pft[r, f, "shock"]))
    resp = 100.0 * (shock - base) / base if abs(base) > 1e-12 else None
    return {
        "codes": {k: res[k]["code"] for k in res},
        "pft_base": round(base, 5),
        "pft_shock": round(shock, 5),
        "land_resp_pct": round(resp, 3) if resp is not None else None,
    }


def _gempack_land_pct(gempack_har: Path) -> float | None:
    """GEMPACK pfe[Land,Food,EU_28] %-change from the sl4dump."""
    from gempack_reference import sl4_levels

    for var in ("pfe", "pes"):
        try:
            cells = sl4_levels(str(gempack_har), var)
        except Exception:
            continue
        for k, v in cells.items():
            if "Land" in k and "Food" in k and "EU_28" in k:
                return round(v, 3)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gtap7_3x3")
    ap.add_argument("--base-calibrated", action="store_true")
    ap.add_argument("--ref-gdx", default=None)
    ap.add_argument("--gempack-har", default=None)
    a = ap.parse_args()

    ref = Path(a.ref_gdx) if a.ref_gdx else (
        ROOT / "tests/fixtures/gtap7" / a.dataset / "out_gtap_shock_ifsub1.gdx"
    )
    out = {
        "tool": "measure_gempack_blocks",
        "dataset": a.dataset,
        "mode": "base_calibrated" if a.base_calibrated else "default",
    }
    out.update(_land_response(a.dataset, a.base_calibrated, ref))

    if a.gempack_har:
        gem = _gempack_land_pct(Path(a.gempack_har))
        if gem is None:
            out["gempack"] = {"status": "skipped", "reason": "land pfe/pes not in fixture"}
        else:
            out["gempack_land_pct"] = gem
            if out.get("land_resp_pct") is not None:
                out["gap_vs_gempack_pp"] = round(out["land_resp_pct"] - gem, 3)
    else:
        out["gempack"] = {"status": "skipped", "reason": "no --gempack-har given"}

    print(json.dumps(out))


if __name__ == "__main__":
    main()
