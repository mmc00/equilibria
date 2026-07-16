"""Diff ALL variables: Python-solution GDX (v_* symbols) vs GAMS-ref GDX.
Both GDXs were written by GAMS; read via gdxdump (Current). Ranks by divergence."""
import subprocess, csv, sys
from pathlib import Path
ROOT = Path("/Users/marmol/.superset/worktrees/b14cb643-ee65-449d-b3f0-be8003b60783/scratched-stag")
GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/Current/Resources/gdxdump"
PY_GDX = ROOT / "output/pyshock_sol.gdx"
REF_GDX = ROOT / "output/gtap7_3x3_pure_local_bundle/out_3x3_nlp.gdx"
PERIOD = sys.argv[1] if len(sys.argv) > 1 else "shock"

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

def symbols(gdx):
    r = subprocess.run([GDXDUMP, str(gdx), "Symbols"], capture_output=True, text=True)
    names = []
    for L in r.stdout.splitlines():
        parts = L.split()
        # format: "  1 Symbol name ..." — grab token after the index
        for p in parts:
            if p.isidentifier():
                names.append(p); break
    return names

ref_syms = [s for s in symbols(REF_GDX)]
diffs = []
for rs in ref_syms:
    ref = {k[:-1]: v for k, v in dump(REF_GDX, rs).items() if k and k[-1] == PERIOD}
    if not ref: continue
    py = {k[:-1]: v for k, v in dump(PY_GDX, "v_" + rs).items() if k and k[-1] == PERIOD}
    if not py: continue
    for key, rv in ref.items():
        pv = py.get(key)
        if pv is None: continue
        rel = abs(pv - rv) / max(abs(rv), 1e-9)
        if rel > 0.02 and abs(pv - rv) > 1e-5:
            diffs.append((rel, rs, key, pv, rv))
diffs.sort(reverse=True)
from collections import Counter
fam = Counter(s for _, s, _, _, _ in diffs)
print(f"period={PERIOD}  divergent cells (>2%): {len(diffs)}")
print("families:", dict(fam.most_common(15)))
print()
for rel, s, key, pv, rv in diffs[:35]:
    print(f"  {rel*100:8.1f}%  {s}{key}  py={pv:.5f} gams={rv:.5f}")
