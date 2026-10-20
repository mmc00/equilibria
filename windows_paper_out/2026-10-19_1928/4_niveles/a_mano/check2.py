import sys, importlib.util
from pathlib import Path
import numpy as np
from equilibria.babel.har import read_har
from equilibria.compare import from_har
root = Path.cwd(); N = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("e", root / "scripts/gtap/export_sl4_levels.py"); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
lv = e.parse_levels(N / sys.argv[2])
src = read_har(str(root / sys.argv[3]))
n2h = {(h.long_name or "").split("#", 1)[0].strip(): k for k, h in src.items() if (h.long_name or "").strip()}
for var, (pct, pre, post, chg) in lv.items():
    hid = n2h.get(var)
    if hid is None: continue
    t = src[hid]
    if t.array.dtype.kind not in "fiu" or t.array.size != post.size: continue
    ref = np.asarray(t.array, dtype=np.float64).ravel(order="F")
    if not np.allclose(ref, pct, atol=1e-3, rtol=1e-3):
        d = np.abs(ref - pct); k = int(d.argmax())
        print(f"MISMATCH {var}: n={ref.size} celdas_malas={int((~np.isclose(ref,pct,atol=1e-3,rtol=1e-3)).sum())} max|d|={d.max():.4g} en k={k} ref={ref[k]:.6g} sl4={pct[k]:.6g}")
SIX = ("y", "qinv", "rore", "pinv", "pva", "pe")
sol = from_har(N / sys.argv[4])
print("seis en NOLV:", sorted(set(SIX) & set(sol.no_level)), "| seis con nivel:", [v for v in SIX if v in sol.levels])
spec = importlib.util.spec_from_file_location("o", root / "tests/compare/oracles/levb_six_vars.py"); o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
exp = o.levb(sys.argv[5]); bad = 0
for var in ("y", "qinv", "rore"):
    got = {k[1][0]: x for k, x in sol.levels[var].items() if k[0] == "pre"}
    assert got.keys() == exp[var].keys(), var
    worst = max(abs(got[r] / x - 1) if x else abs(got[r]) for r, x in exp[var].items())
    bad += worst > 1e-6; print(f"  {var}: {len(got)} regiones, max rel err vs oraculo {worst:.2e}")
for var in ("pinv", "pva", "pe"):
    pre = [x for k, x in sol.levels[var].items() if k[0] == "pre"]
    worst = max(abs(x - 1) for x in pre); bad += worst > 1e-6
    print(f"  {var}: {len(pre)} celdas, max |pre-1| {worst:.2e}")
print("ACEPTACION:", "PASA" if not bad else "FALLA")
