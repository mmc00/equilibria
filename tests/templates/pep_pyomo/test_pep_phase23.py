"""PEP phase-2/3 gate: export the Pyomo model to .gms, solve it IN GAMS (phase-2), then
diff that pyomo-origin gdx cell-by-cell against the GAMS-native MCP gdx (phase-3). Both are
solved by GAMS, so the solver tolerance cancels and any difference is a Pyomo→.gms
TRANSLATION error. 100% here proves the export is faithful. Needs GAMS; skips cleanly."""
from __future__ import annotations
from pathlib import Path
import re
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[3]
SAM = ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
MCP_REF = ROOT / "src/equilibria/templates/reference/pep2/scripts/Results_mcp.gdx"
GAMS = "/Library/Frameworks/GAMS.framework/Versions/53/Resources/gams"
GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/53/Resources/gdxdump"


def _read_native(gdx):
    txt = subprocess.run([GDXDUMP, str(gdx)], capture_output=True, text=True).stdout
    out, cur = {}, None
    for line in txt.splitlines():
        sm = re.match(r"\s*(?:free|positive)?\s*Variable\s+([A-Za-z_]+)\s+.*?/L\s+([-\d.E+]+)", line)
        if sm and "(" not in line.split("/")[0]:
            out[(sm.group(1), None)] = float(sm.group(2)); continue
        hm = re.match(r"\s*(?:free|positive)?\s*Variable\s+([A-Za-z_]+)\(", line)
        if hm:
            cur = hm.group(1); continue
        if cur:
            rm = re.match(r"\s*(.+?)\.L\s+([-\d.E+]+)", line)
            if rm:
                kp = rm.group(1).replace("'", "").strip()
                idx = tuple(kp.split(".")) if "." in kp else kp
                out[(cur, idx)] = float(rm.group(2))
            if line.strip().endswith("/;"):
                cur = None
    return out


def _descalarize(raw, names_longest_first):
    core = raw[:-1] if raw.endswith("_") else raw
    for nm in names_longest_first:
        if core == nm:
            return (nm, None)
        if core.startswith(nm + "_"):
            rest = core[len(nm) + 1:]
            parts = rest.split("_")
            return (nm, parts[0] if len(parts) == 1 else tuple(parts))
    return None


def _read_pyomo_origin(gdx, names):
    txt = subprocess.run([GDXDUMP, str(gdx)], capture_output=True, text=True).stdout
    out = {}
    for line in txt.splitlines():
        sm = re.match(r"\s*(?:free|positive)?\s*Variable\s+([A-Za-z0-9_]+)\s*/L\s+([-\d.E+]+)", line)
        if sm:
            de = _descalarize(sm.group(1), names)
            if de:
                out[de] = float(sm.group(2))
    return out


@pytest.mark.skipif(not SAM.exists() or not MCP_REF.exists(),
                    reason="pep2 SAM or MCP reference not present")
@pytest.mark.skipif(not Path(GAMS).exists() or not Path(GDXDUMP).exists(),
                    reason="GAMS/gdxdump unavailable")
def test_pep_phase2_3_translation_faithful(tmp_path):
    """Pyomo→.gms→GAMS (phase-2) then diff vs the GAMS-native MCP (phase-3) = 100%."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from pyomo.environ import Var
    from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model

    root = ROOT / "src/equilibria/templates/reference/pep2"
    state = PEPModelCalibrator(sam_file=root / "data/SAM-V2_0.gdx",
                               val_par_file=root / "data/VAL_PAR.xlsx").calibrate()
    m = build_pep_model(state, variant="base", form="nlp")
    names = sorted({v.name for v in m.component_objects(Var, active=True)},
                   key=len, reverse=True)

    # phase-2: export + solve in GAMS
    gms = tmp_path / "pep_pyomo.gms"
    origin = tmp_path / "pep_origin.gdx"
    m.write(str(gms), format="gams", io_options={"symbolic_solver_labels": True})
    r = subprocess.run([GAMS, str(gms), f"gdx={origin}", "lo=2", "reslim=120"],
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert origin.exists() and origin.stat().st_size > 0, (
        f"GAMS did not produce a results gdx (rc={r.returncode})")

    # phase-3: de-scalarize + diff vs gams-native MCP
    native = _read_native(MCP_REF)
    pyomo = _read_pyomo_origin(origin, names)
    keys = [k for k in (set(native) & set(pyomo)) if k[0] != "LEON"]
    assert len(keys) > 200, f"too few comparable cells ({len(keys)}) — de-scalarize broke"

    def match(a, b):
        return abs(a - b) <= 1e-4 + 1e-4 * max(abs(a), abs(b))
    bad = [(k, pyomo[k], native[k]) for k in keys if not match(pyomo[k], native[k])]
    assert not bad, f"phase-3 translation mismatches ({len(keys) - len(bad)}/{len(keys)}): {bad[:5]}"
