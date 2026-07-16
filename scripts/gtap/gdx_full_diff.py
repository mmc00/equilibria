"""Diff ALL symbols (variables AND params) between a Python-solution GDX and a
GAMS reference GDX. Both written by GAMS; read via gdxdump. Python symbols are
prefixed v_ (vars) / p_ (params); GAMS symbols are bare. Ranks by divergence."""
import subprocess, csv, sys
from pathlib import Path
ROOT = Path("/Users/marmol/.superset/worktrees/b14cb643-ee65-449d-b3f0-be8003b60783/scratched-stag")
GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/Current/Resources/gdxdump"
PY_GDX = ROOT / "output/pyshock_sol.gdx"
REF_GDX = ROOT / "output/gtap7_3x3_pure_local_bundle/out_3x3_nlp.gdx"
PERIOD = sys.argv[1] if len(sys.argv) > 1 else "shock"
TIMES = ("base", "check", "shock")

def dump(gdx, sym):
    r = subprocess.run([GDXDUMP, str(gdx), "Format=csv", f"Symb={sym}"], capture_output=True, text=True)
    out = {}
    if r.returncode != 0 or not r.stdout.strip():
        return out
    rd = csv.reader(r.stdout.splitlines()); next(rd, None)
    for row in rd:
        if len(row) < 2: continue
        *keys, val = row
        try: out[tuple(k.strip('"') for k in keys)] = float(val)
        except ValueError: pass
    return out

def filt(d):
    """keep PERIOD cells (drop the t index) plus time-less cells."""
    out = {}
    for k, v in d.items():
        if k and k[-1] in TIMES:
            if k[-1] == PERIOD: out[k[:-1]] = v
        else:
            out[k] = v
    return out

def symbols(gdx):
    r = subprocess.run([GDXDUMP, str(gdx), "Symbols"], capture_output=True, text=True)
    names = []
    for L in r.stdout.splitlines():
        for p in L.split():
            if p.isidentifier():
                names.append(p); break
    return names

diffs = []
for rs in symbols(REF_GDX):
    ref = filt(dump(REF_GDX, rs))
    if not ref: continue
    pref = None; pyd = {}
    for pr in ("v_", "p_"):
        cand = filt(dump(PY_GDX, pr + rs))
        if cand:
            pref, pyd = pr, cand; break
    if not pyd: continue
    for key, rv in ref.items():
        pv = pyd.get(key)
        if pv is None: continue
        rel = abs(pv - rv) / max(abs(rv), 1e-9)
        if rel > 0.02 and abs(pv - rv) > 1e-5:
            diffs.append((rel, pref.rstrip("_") + ":" + rs, key, pv, rv))
diffs.sort(reverse=True)
from collections import Counter
fam = Counter(s for _, s, _, _, _ in diffs)
print(f"period={PERIOD}  divergent cells (>2%): {len(diffs)}")
print("families:", dict(fam.most_common(20)))
print()
for rel, s, key, pv, rv in diffs[:40]:
    print(f"  {rel*100:8.1f}%  {s}{key}  py={pv:.6g} gams={rv:.6g}")
