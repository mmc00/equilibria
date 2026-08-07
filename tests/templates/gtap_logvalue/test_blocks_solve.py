"""Transitive fidelity gate: the composed log-value block model reproduces the
calibrated point (base) and matches the port monolith (shock), both seeded from the
same Julia calibrated point. Since port ≡ Julia (jparity 100%), blocks ≡ Julia.
"""

from pyomo.environ import value as V
from tests.templates.gtap_logvalue._harness import load_sol


def test_blocks_base_solves_and_reproduces_calibration():
    from equilibria.templates.gtap_logvalue.composer import solve

    sol = load_sol("gtap7_3x3")
    res = solve(sol, rordelta=1)
    assert res["ok"], f"base solve not optimal: {res['status']}"
    m = res["model"]
    # the base solve must reproduce the calibrated seed (fixed-point) for key vars
    import statistics

    for var in ("qxs", "pva", "qva", "y", "pfe"):
        comp = m.component(var)
        seed = sol.get(var)
        if comp is None or not isinstance(seed, dict):
            continue
        moves = []
        for idx in comp:
            key = idx if isinstance(idx, tuple) else (idx,)
            s = seed.get(key)
            if s is None or abs(s) < 1e-6 or s != s:
                continue
            moves.append(abs(V(comp[idx]) / s - 1.0))
        if moves:
            assert statistics.median(moves) < 1e-4, (
                f"{var} base drifted from calibration: median {statistics.median(moves)}"
            )


def test_blocks_shock_matches_port_monolith():
    """Transitive gate: blocks-logvalue shock ≡ port monolith shock, both seeded from
    the same calibrated point. Difference is only equation FORM → must be ~0."""
    import statistics

    from equilibria.templates.gtap_julia.model import solve_shock as port_shock
    from equilibria.templates.gtap_julia.solution import load_solution
    from equilibria.templates.gtap_logvalue.composer import solve_shock as blk_shock

    sol = load_sol("gtap7_3x3")
    mb = blk_shock(sol, 1.10, rordelta=1)
    assert mb["ok"], f"blocks shock not optimal: {mb['status']}"
    psol = load_solution(
        "/private/tmp/claude-501/-Users-marmol--superset-worktrees-b14cb643-ee65-449d-b3f0-be8003b60783-gray-carver/45c8a8b5-8bb9-485a-8e5c-e498c5bb605d/scratchpad/wp3x3_b.csv"
    )
    mp = port_shock(psol, 1.10, rordelta=1)["model"]
    b = mb["model"]
    for var in ("qxs", "pva", "qva", "y", "pfe", "qfe", "pmds", "qms"):
        cb = b.component(var)
        cp = getattr(mp, var, None)
        if cb is None or cp is None:
            continue
        diffs = []
        for idx in cb:
            try:
                vb = V(cb[idx])
                vp = V(cp[idx])
            except Exception:
                continue
            if abs(vp) < 1e-6:
                continue
            diffs.append(abs(vb / vp - 1.0))
        if diffs:
            assert statistics.median(diffs) < 1e-5, (
                f"{var} blocks vs port median {statistics.median(diffs)}"
            )
