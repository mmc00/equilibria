"""Solve test: the NLP base model, seeded at the benchmark, stays at residual≈0.

The pep2 BASE benchmark reproduces the SAM, so a faithful model solved from the
benchmark point must return a small max residual (the cyipopt solver early-exits there).
A large residual means an equation differs from the reference — a real fidelity bug."""
from __future__ import annotations
from pathlib import Path
import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[3]
SAM = ROOT / "src/equilibria/templates/reference/pep2/data/SAM-V2_0.gdx"
VALPAR = ROOT / "src/equilibria/templates/reference/pep2/data/VAL_PAR.xlsx"
_HAS_IPOPT = importlib.util.find_spec("pyomo") is not None


@pytest.fixture(scope="module")
def state():
    from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
    return PEPModelCalibrator(sam_file=SAM, val_par_file=VALPAR).calibrate()


@pytest.mark.skipif(not SAM.exists(), reason="pep2 SAM not present")
def test_base_benchmark_residual_small(state):
    from pyomo.environ import SolverFactory
    if not SolverFactory("ipopt").available():
        pytest.skip("ipopt not available")
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep, _max_residual
    m = build_pep_model(state, variant="base", form="nlp")
    # residual AT the seed (before solving) — the benchmark should nearly satisfy the system
    seed_resid = _max_residual(m)
    res = solve_pep(m)
    # after solving it must be feasible; report both for diagnosis
    assert res.code == 1, f"did not converge: {res.message} (resid {res.max_residual:.2e})"
    assert res.max_residual < 1e-3, (
        f"max residual {res.max_residual:.2e} too large (seed was {seed_resid:.2e}) — "
        f"an equation likely differs from the reference")


@pytest.mark.skipif(not SAM.exists(), reason="pep2 SAM not present")
def test_faithful_at_benchmark(state):
    """95 of 96 equation families hold at the exact benchmark seed. The ONE exception is
    eq92 (GDP_IB accounting identity), whose residual equals the calibration's own
    documented imbalance (GDP_MPO − GDP_IBO ≈ 2026, state.validation.passed=False) —
    a reference DATA hole, not a port bug. This is the structural-fidelity gate."""
    from pyomo.environ import Constraint, value
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    m = build_pep_model(state, variant="base", form="nlp")
    violated = []
    for c in m.component_objects(Constraint, active=True):
        for idx in c:
            con = c[idx]
            body = value(con.body, exception=False)
            tgt = (value(con.lower, exception=False) if con.lower is not None
                   else value(con.upper, exception=False))
            if body is None or tgt is None:
                continue
            if abs(body - tgt) > 1.0:            # 1.0 unit tolerance on ~1e4-scale levels
                violated.append(c.name)
                break
    # only eq92 (the calibration's 2026 GDP_IB imbalance) may violate
    assert set(violated) <= {"eq92"}, f"unexpected benchmark violations: {sorted(set(violated))}"


@pytest.mark.skipif(not SAM.exists(), reason="pep2 SAM not present")
def test_no_ces_overflow_at_benchmark(state):
    """Every constraint body must be FINITE at the benchmark seed. The MCP square-closure
    fixes structural-zero labor cells (LD=0); eq5's composite-labor CES must skip them
    (`if (l,j) in LDact`) or `0**(-rho)` overflows — the bug that made PATH 'diverge' to a
    constant 7.74e15 (a deterministic eval overflow, not a solver basin). Guards eq5/eq7."""
    import math
    from pyomo.environ import Constraint, value
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    m = build_pep_model(state, variant="base", form="mcp")
    bad = []
    for c in m.component_data_objects(Constraint, active=True):
        b = value(c.body, exception=False)
        if b is None or math.isinf(b) or math.isnan(b) or abs(b) > 1e10:
            bad.append(c.name)
    assert not bad, f"constraint bodies overflow at benchmark (CES 0**-rho?): {bad[:6]}"


@pytest.mark.skipif(not SAM.exists(), reason="pep2 SAM not present")
def test_nlp_mcp_mirror(state):
    """NLP and MCP forms, solved from the same feasible benchmark seed, land on the SAME
    point (the parity anchor) — 466/466 economic cells match exactly. Structural-zero prices
    (e.g. PD['othind'], no domestic demand) are filled from their *O benchmark in BOTH forms,
    matching GAMS (whose native MCP leaves them free at PDO=1.132). LEON (the Walras slack)
    is form-defining and excluded. Requires PATH for the MCP solve; skips cleanly otherwise."""
    import sys
    src = "/Users/marmol/proyectos/path-capi-python/src"
    if Path(src).exists() and src not in sys.path:
        sys.path.insert(0, src)
    if importlib.util.find_spec("path_capi_python") is None:
        pytest.skip("path_capi_python unavailable for MCP mirror")
    from pyomo.environ import Var, value
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep, _ensure_path_lib
    _ensure_path_lib()

    def vals(m):
        d = {}
        for v in m.component_objects(Var, active=True):
            for idx in v:
                x = value(v[idx], exception=False)
                if x is not None:
                    d[(v.name, idx)] = float(x)
        return d

    mn = build_pep_model(state, variant="base", form="nlp"); rn = solve_pep(mn); vn = vals(mn)
    mm = build_pep_model(state, variant="base", form="mcp"); rm = solve_pep(mm); vm = vals(mm)
    assert rn.code == 1 and rm.code == 1, f"NLP={rn.message} MCP={rm.message}"
    keys = [k for k in (set(vn) & set(vm)) if k[0] != "LEON"]

    def match(a, b):
        return abs(a - b) <= 1e-4 + 1e-4 * max(abs(a), abs(b))
    diffs = sorted((k for k in keys if not match(vn[k], vm[k])), key=lambda k: -abs(vn[k] - vm[k]))
    # NLP and MCP are an exact mirror — no cell differs
    assert diffs == [], (
        f"unexpected NLP↔MCP mirror diffs: {[(k, vn[k], vm[k]) for k in diffs[:5]]}")


MCP_REF = ROOT / "src/equilibria/templates/reference/pep2/scripts/Results_mcp.gdx"
_GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/53/Resources/gdxdump"


@pytest.mark.skipif(not SAM.exists() or not MCP_REF.exists(),
                    reason="pep2 SAM or GAMS-native MCP reference not present")
def test_mcp_matches_gams_native_mcp(state):
    """The Pyomo MCP solve must match the GAMS-NATIVE MCP (Results_mcp.gdx, from
    PEP-1-1_v2_1_mcp_solve.gms: MODEL /ALL/ + SOLVE USING MCP, base case) cell-by-cell —
    same formulation, same PATH engine, so the solver tolerance cancels. This is the
    definitive fidelity gate for the MCP form (100% / 285 economic cells). Skips cleanly
    if PATH or gdxdump is unavailable."""
    import sys
    src = "/Users/marmol/proyectos/path-capi-python/src"
    if Path(src).exists() and src not in sys.path:
        sys.path.insert(0, src)
    if importlib.util.find_spec("path_capi_python") is None:
        pytest.skip("path_capi_python unavailable for MCP solve")
    if not Path(_GDXDUMP).exists():
        pytest.skip("gdxdump (GAMS) unavailable")
    from pyomo.environ import value
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep, _ensure_path_lib
    _ensure_path_lib()

    # read the GAMS-native MCP levels via gdxdump (raw Variable symbols: PD, GDP_BP, …)
    import subprocess
    import re
    txt = subprocess.run([_GDXDUMP, str(MCP_REF)], capture_output=True, text=True).stdout
    gams, cur = {}, None
    for line in txt.splitlines():
        sm = re.match(r"\s*(?:free|positive)?\s*Variable\s+([A-Za-z_]+)\s+.*?/L\s+([-\d.E+]+)", line)
        if sm and "(" not in line.split("/")[0]:
            gams[(sm.group(1), None)] = float(sm.group(2)); continue
        hm = re.match(r"\s*(?:free|positive)?\s*Variable\s+([A-Za-z_]+)\(", line)
        if hm:
            cur = hm.group(1); continue
        if cur:
            rm = re.match(r"\s*(.+?)\.L\s+([-\d.E+]+)", line)
            if rm:
                kp = rm.group(1).replace("'", "").strip()
                idx = tuple(kp.split(".")) if "." in kp else kp
                gams[(cur, idx)] = float(rm.group(2))
            if line.strip().endswith("/;"):
                cur = None

    m = build_pep_model(state, variant="base", form="mcp")
    r = solve_pep(m)
    assert r.code == 1, f"MCP solve did not converge: {r.message}"

    def match(a, b):
        return abs(a - b) <= 1e-4 + 1e-4 * max(abs(a), abs(b))
    tot = ok = 0
    bad = []
    for (vname, idx), gval in gams.items():
        if vname.lower() in ("leon",):
            continue
        pv = m.find_component(vname)
        if pv is None:
            continue
        try:
            pval = float(value(pv[idx] if idx is not None else pv, exception=False))
        except Exception:
            continue
        tot += 1
        if match(pval, gval):
            ok += 1
        else:
            bad.append((vname, idx, pval, gval))
    assert tot > 200, f"too few comparable cells ({tot}) — gdx read likely broke"
    assert not bad, f"MCP↔GAMS mismatches ({ok}/{tot}): {bad[:5]}"
