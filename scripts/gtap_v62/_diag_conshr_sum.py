"""Verify sum_i CONSHR_i_0 per region for gtap6 datasets.

Hypothesis: eq_pcons baseline residual = 1.0 means sum_i CONSHR_i = 2.0
in gtap6_15x10. Either the calibration is double-counting or sets.i
has extra ghost indices.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from equilibria.templates.gtap_v62 import GTAPv62Sets, GTAPv62Parameters
from equilibria.templates.gtap_v62.gtap_v62_calibration import derive_calibration

DATASETS = ["gtap6_3x3", "gtap6_5x5", "gtap6_10x7", "gtap6_15x10"]

for name in DATASETS:
    DATA = Path(f"datasets/{name}")
    if not DATA.exists():
        continue
    sets = GTAPv62Sets()
    sets.load_from_har(DATA / "sets.har", default_path=DATA / "default.prm")
    params = GTAPv62Parameters()
    params.load_from_har(basedata_path=DATA / "basedata.har",
                         default_prm_path=DATA / "default.prm", sets=sets)
    derived = derive_calibration(sets, params)
    print(f"\n=== {name} ===  |i|={len(sets.i)} |r|={len(sets.r)}")
    print(f"  i = {list(sets.i)}")
    print(f"  r = {list(sets.r)}")
    for r in sets.r:
        s = sum(float(derived.share_hhd_cd.get((i, r), 0.0)) for i in sets.i)
        n_nonzero = sum(1 for i in sets.i if derived.share_hhd_cd.get((i, r), 0.0) > 0)
        print(f"  {r:15s}  sum_i CONSHR = {s:.6f}  (nonzero={n_nonzero})")
