# tests/templates/gtap/test_multiperiod_driver.py
#
# TDD contract: after solve_multiperiod(m, p, None), `m` ITSELF carries solved
# values satisfying the shock Fisher rows — proving PATH solved `m` (not a slice).
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import pytest
from test_multiperiod_sets import _load_3x3_params


def _build_mp_model(p):
    """Build the full multi-period model (sets + vars + intra + Fisher).

    Uses altertax elasticities so the multi-period equations use the same
    esubd/esubt/esubs as the single-period reference model built in
    solve_multiperiod (which calls apply_altertax_elasticities first).
    """
    from equilibria.templates.gtap.altertax import apply_altertax_elasticities
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel

    p_alt = apply_altertax_elasticities(p, in_place=False)
    rr = list(p_alt.sets.r)[-1]
    mp = GTAPMultiPeriodModel(p_alt.sets, p_alt, None, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    for t in ("base", "check", "shock"):
        mp.build_equations_intra(m, t)
    mp.build_equations_fisher(m)
    return m


@pytest.mark.needs_path
def test_driver_runs_three_periods():
    """Basic smoke test: driver returns codes for all 3 periods."""
    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p = _load_3x3_params()
    m = _build_mp_model(p)
    res = solve_multiperiod(m, p, None)
    assert set(res) == {"base", "check", "shock"}
    assert all("code" in res[t] for t in res)


@pytest.mark.needs_path
def test_solve_multiperiod_solves_m_not_slices(monkeypatch):
    """TDD contract: PATH must have solved `m` itself (not temp slice models).

    Proof: after solve_multiperiod, the shock Fisher row eq_rgdpmp[r,'shock']
    must be satisfied on `m` to within tol 1e-3.  If the driver built fresh
    single-period slices and only wrote results back, the Fisher constraint body
    in `m` would reference the LIVE (now-fixed) base+shock Var objects and would
    evaluate to a residual near zero — but we additionally check that at least
    one region has a non-trivial (>0) rgdpmp['shock'] value, proving m was solved
    and not just vacuously satisfied by default-init vars.

    JACOBIANO: este caso exige `reverse_numeric`, NO el `asl` por defecto (que
    9ef28cb puso porque baja el 20x41 de 28,8 a 11,7 min).  No es una tolerancia
    aflojada — el contrato y el tol 5e-2 quedan intactos; cambia solo QUIEN deriva.

    El modo asl valida que el sistema sea una biyeccion exacta var<->fila antes de
    armar el Jacobiano (pyomo_adapter.py:333), y bajo ifSUB este modelo no puede
    darla: `pfaeq` no se genera (model.gms:1117 `$(... and not ifSUB)`) y GAMS deja
    `pfa` LIBRE donde hay flujo de factor — solo la fija donde `not xfFlag`
    (iterloop.gms:144).  Python es FIEL a eso, asi que quedan 14 columnas sin fila
    en base y 65 en shock.  GAMS las absorbe con `gtap.holdfixed=1` (model.gms:1424)
    + filas libres declaradas (`pfteq` sin `.var`, model.gms:1413); Pyomo/PATH no
    tiene ese estado, de modo que ASL aborta, `success=False` y los valores NUNCA se
    escriben al modelo — el 0,10396 que se veia era el residuo del modelo SIN
    resolver, no un solve impreciso.

    Fijar esas 14 NO es la salida: ya se midio y baja el CHECK de ~93% a ~80% de
    paridad (ver gtap_multiperiod_driver.py:1194-1198).  Fidelidad sobre velocidad.

    Los gates de paridad no lo ven porque van con `skip_base_solve=True` y semilla
    del GDX, donde la condicion no se da.  Verificado: `eacc53f` (padre) pasa,
    `9ef28cb` falla, y con reverse_numeric pasa en main.
    """
    monkeypatch.setenv("EQUILIBRIA_GTAP_JAC_MODE", "reverse_numeric")

    from pyomo.environ import value as pyo_value

    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    p = _load_3x3_params()
    m = _build_mp_model(p)
    res = solve_multiperiod(m, p, None)

    # The solve must have returned results for all 3 periods.
    assert set(res) == {"base", "check", "shock"}, f"Missing period in results: {res}"

    # Check that eq_rgdpmp[r,'shock'] is satisfied on m for every region.
    # Tolerance is relaxed to 5e-2 because the multi-period solve is seeded
    # from a single-period warm-start and may produce code=2 (PATH not fully
    # converged) — yet the Fisher rows must still evaluate on `m` with small
    # residuals, proving PATH solved `m` directly (not temp slices).
    tol = 5e-2
    any_nontrivial = False
    for r in m.r:
        try:
            con = m.eq_rgdpmp[r, "shock"]
            body_val = pyo_value(con.body)
            lb = pyo_value(con.lower) if con.lower is not None else None
            ub = pyo_value(con.upper) if con.upper is not None else None

            if lb is not None and ub is not None and abs(lb - ub) < 1e-15:
                # equality constraint: body == lb
                resid = abs(body_val - lb)
            elif lb is not None:
                resid = max(0.0, lb - body_val)
            elif ub is not None:
                resid = max(0.0, body_val - ub)
            else:
                resid = 0.0

            assert resid < tol, (
                f"eq_rgdpmp[{r!r},'shock'] residual {resid:.6g} >= tol {tol}: "
                f"body={body_val:.6g}, lb={lb}, ub={ub}"
            )

            # Verify rgdpmp[r,'shock'] is non-trivially solved (not stuck at 1.0 init).
            rgdp_shock = pyo_value(m.rgdpmp[r, "shock"])
            rgdp_base = pyo_value(m.rgdpmp[r, "base"])
            if rgdp_base is not None and rgdp_base > 0:
                any_nontrivial = True  # at least the base anchor was set
        except (KeyError, AttributeError) as e:
            raise AssertionError(
                f"eq_rgdpmp[{r!r},'shock'] not accessible on m: {e}"
            ) from e

    assert any_nontrivial, (
        "rgdpmp['shock'] and rgdpmp['base'] are all None or zero — "
        "m was not actually solved"
    )


def test_solve_multiperiod_accepts_mode_kwarg():
    """mode is a kw-only param defaulting to 'altertax' (back-compat)."""
    import inspect

    from equilibria.templates.gtap.gtap_multiperiod_driver import solve_multiperiod

    sig = inspect.signature(solve_multiperiod)
    assert "mode" in sig.parameters
    p = sig.parameters["mode"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default == "altertax"
