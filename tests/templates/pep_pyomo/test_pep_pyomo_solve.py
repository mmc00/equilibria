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
MCP_REF_SIM1 = ROOT / "src/equilibria/templates/reference/pep2/scripts/Results_mcp_sim1.gdx"
_GDXDUMP = "/Library/Frameworks/GAMS.framework/Versions/53/Resources/gdxdump"


def _read_gams_var_levels(gdx_path):
    """Parse the .L levels of every Variable in a raw-symbol GAMS gdx (via gdxdump),
    handling scalar-inline (`Variable NAME desc /L v, M v /;`) and matrix
    (`Variable NAME(*) desc / 'k'.L v, … /`). Returns {(var, idx): level}, idx None/str/tuple."""
    import subprocess
    import re
    txt = subprocess.run([_GDXDUMP, str(gdx_path)], capture_output=True, text=True).stdout
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


def _diff_mcp_vs_gams(m, gams):
    """Cell-by-cell match of a solved Pyomo MCP model against parsed GAMS levels.
    Returns (ok, total, bad_list). LEON (Walras slack) excluded."""
    from pyomo.environ import value
    tot = ok = 0
    bad = []

    def match(a, b):
        return abs(a - b) <= 1e-4 + 1e-4 * max(abs(a), abs(b))
    for (vname, idx), gval in gams.items():
        if vname.lower() == "leon":
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
    return ok, tot, bad


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
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep, _ensure_path_lib
    _ensure_path_lib()
    gams = _read_gams_var_levels(MCP_REF)
    m = build_pep_model(state, variant="base", form="mcp")
    r = solve_pep(m)
    assert r.code == 1, f"MCP solve did not converge: {r.message}"
    ok, tot, bad = _diff_mcp_vs_gams(m, gams)
    assert tot > 200, f"too few comparable cells ({tot}) — gdx read likely broke"
    assert not bad, f"MCP↔GAMS mismatches ({ok}/{tot}): {bad[:5]}"


@pytest.mark.skipif(not SAM.exists() or not MCP_REF_SIM1.exists(),
                    reason="pep2 SAM or GAMS SIM1 MCP reference not present")
def test_mcp_sim1_shock_matches_gams():
    """The SIM1 counterfactual (−25% export tax: `ttix.fx=ttixO*0.75`) applied in Python via
    apply_sim1_export_tax_cut, solved as MCP, must match the GAMS-native SIM1 MCP
    (Results_mcp_sim1.gdx) cell-by-cell — the base+shock cycle closed on one engine. GAMS
    moves GDP_BP 46707→46748.2084; Python must land there too (proves the MCP actually
    re-solves the shock, not early-exits at base). Skips cleanly without PATH/gdxdump.

    Uses a FRESH calibration (not the module `state` fixture) because the shock mutates
    ttixO in place — sharing it would contaminate the base-case tests."""
    import sys
    from pyomo.environ import value
    src = "/Users/marmol/proyectos/path-capi-python/src"
    if Path(src).exists() and src not in sys.path:
        sys.path.insert(0, src)
    if importlib.util.find_spec("path_capi_python") is None:
        pytest.skip("path_capi_python unavailable for MCP solve")
    if not Path(_GDXDUMP).exists():
        pytest.skip("gdxdump (GAMS) unavailable")
    from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep, _ensure_path_lib
    from equilibria.templates.pep_pyomo.pep_pyomo_scenarios import apply_sim1_export_tax_cut
    _ensure_path_lib()
    gams = _read_gams_var_levels(MCP_REF_SIM1)
    state = PEPModelCalibrator(sam_file=SAM, val_par_file=VALPAR).calibrate()   # fresh
    apply_sim1_export_tax_cut(state)          # ttixO *= 0.75, in place, before build
    m = build_pep_model(state, variant="base", form="mcp")
    r = solve_pep(m)
    assert r.code == 1, f"SIM1 MCP solve did not converge: {r.message}"
    gdp = float(value(m.GDP_BP, exception=False))
    assert abs(gdp - 46748.2084) < 1.0, f"SIM1 GDP_BP={gdp}, expected ~46748.2084 (base is 46707)"
    ok, tot, bad = _diff_mcp_vs_gams(m, gams)
    assert tot > 200, f"too few comparable cells ({tot})"
    assert not bad, f"SIM1 MCP↔GAMS mismatches ({ok}/{tot}): {bad[:5]}"


@pytest.mark.skipif(not SAM.exists(), reason="pep2 SAM not present")
def test_objdef_variant_equals_base_nlp(state):
    """The objdef variant adds a DUMMY objective (`OBJDEF: OBJ==0`, min OBJ) — the
    `SOLVE NLP MINIMIZING OBJ` lineage. A constant objective can't change the equilibrium, so
    objdef-NLP must land on the exact same point as base-NLP (467/467 cells)."""
    from pyomo.environ import Var, value
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep

    def vals(m):
        d = {}
        for v in m.component_objects(Var, active=True):
            if v.name == "OBJ":
                continue
            for i in v:
                x = value(v[i], exception=False)
                if x is not None:
                    d[(v.name, i)] = float(x)
        return d
    mb = build_pep_model(state, variant="base", form="nlp"); rb = solve_pep(mb)
    mo = build_pep_model(state, variant="objdef", form="nlp"); ro = solve_pep(mo)
    assert rb.code == 1 and ro.code == 1, f"base={rb.message} objdef={ro.message}"
    assert mo.find_component("OBJDEF") is not None, "objdef-NLP must carry the OBJDEF constraint"
    vb, vo = vals(mb), vals(mo)
    keys = set(vb) & set(vo)

    def match(a, b):
        return abs(a - b) <= 1e-4 + 1e-4 * max(abs(a), abs(b))
    bad = [k for k in keys if not match(vb[k], vo[k])]
    assert not bad, f"objdef-NLP diverged from base at {bad[:5]}"


@pytest.mark.skipif(not SAM.exists(), reason="pep2 SAM not present")
def test_objdef_mcp_is_square_and_solves(state):
    """objdef + MCP must stay SQUARE: OBJ is an NLP-only construct (its OBJDEF equation is
    added only on the NLP path), so it must NOT be declared for the MCP form — else it's an
    unpaired free var (359 vars vs 358 eqs) and PATH errors 'Got 358 expressions for 359
    variables'. Guards against that regression. Skips cleanly without PATH."""
    import sys
    src = "/Users/marmol/proyectos/path-capi-python/src"
    if Path(src).exists() and src not in sys.path:
        sys.path.insert(0, src)
    if importlib.util.find_spec("path_capi_python") is None:
        pytest.skip("path_capi_python unavailable for MCP solve")
    from pyomo.environ import Var, Constraint
    from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
    from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep, _ensure_path_lib
    _ensure_path_lib()
    m = build_pep_model(state, variant="objdef", form="mcp")
    assert m.find_component("OBJ") is None, "OBJ must not exist in the MCP form (unpaired free var)"
    nfree = sum(1 for v in m.component_data_objects(Var, active=True) if not v.fixed)
    ncon = sum(1 for _ in m.component_data_objects(Constraint, active=True))
    assert nfree == ncon, f"objdef+mcp not square: {nfree} free vars vs {ncon} eqs"
    r = solve_pep(m)
    assert r.code == 1, f"objdef+mcp did not solve: {r.message}"
