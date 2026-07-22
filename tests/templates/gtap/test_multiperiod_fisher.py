"""Task 4: inter-temporal Fisher links as real Jacobian rows.

Contract: with all 3 periods seeded from the GAMS GDX, eq_rgdpmp[r,'shock']
residual must be < 1e-3 for all r.

Task 4 extension: eq_pabs, eq_pfact, eq_pwfact must ALSO be cross-period rows.
The sensitivity test detects the bug: if the constraint still uses construction-time
float constants (not live base-period Vars), perturbing base-period Vars has no
effect on the shock residual — making it insensitive when it should be sensitive.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gtap"))

import contextlib

from pyomo.environ import value as pv
from test_multiperiod_sets import _load_3x3_params

REF = pathlib.Path(
    "/Users/marmol/proyectos2/equilibria_refs/gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx"
)


def _build_and_seed(p):
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel

    rr = list(p.sets.r)[-1]
    mp = GTAPMultiPeriodModel(p.sets, p, None, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    mp.build_equations_intra(m, "base")
    mp.build_equations_intra(m, "check")
    mp.build_equations_intra(m, "shock")
    mp.build_equations_fisher(m)
    mp.seed_all_periods(m, REF)
    return m, p.sets.r


def test_fisher_rgdpmp_zero_at_gams_point():
    p = _load_3x3_params()
    m, regions = _build_and_seed(p)
    # eq_rgdpmp[*,shock] must give residual ~0 (base is now in the system)
    worst = max(
        abs(pv(m.eq_rgdpmp[r, "shock"].body) - float(m.eq_rgdpmp[r, "shock"].lower))
        for r in regions
    )
    assert worst < 1e-3, f"eq_rgdpmp shock residual {worst} > 1e-3"


def test_fisher_pabs_pfact_pwfact_are_cross_period_rows():
    """eq_pabs/eq_pfact/eq_pwfact at shock must reference live base-period Vars.

    Detection: seed from GAMS (residuals ~0), then scale all base-period pa/xaa/pf/xf
    by a large factor (5x).  A cross-period row reacts → shock residual becomes >> 0.
    A constant-frozen row does NOT react → shock residual stays ~0.
    We assert the residual REACTS (> 0.1) to prove the constraint is live.
    """
    p = _load_3x3_params()
    m, regions = _build_and_seed(p)

    fd_agents = ("hhd", "gov", "inv", "tmg")
    SCALE = 5.0

    # Save residuals before perturbation (should be ~0 at GAMS seed point)
    def _resid_pabs(r):
        c = m.eq_pabs[r, "shock"]
        return abs(pv(c.body) - float(c.lower))

    def _resid_pfact(r):
        c = m.eq_pfact[r, "shock"]
        return abs(pv(c.body) - float(c.lower))

    def _resid_pwfact():
        c = m.eq_pwfact["shock"]
        return abs(pv(c.body) - float(c.lower))

    # At GAMS seed, all residuals should be ~0
    for r in regions:
        assert _resid_pabs(r) < 1e-3, (
            f"eq_pabs[{r},shock] pre-perturb resid too large: {_resid_pabs(r)}"
        )
        assert _resid_pfact(r) < 1e-3, (
            f"eq_pfact[{r},shock] pre-perturb resid too large: {_resid_pfact(r)}"
        )
    assert _resid_pwfact() < 1e-3, (
        f"eq_pwfact[shock] pre-perturb resid too large: {_resid_pwfact()}"
    )

    # Perturb ALL base-period pa/xaa/pf/xf by SCALE (far from equilibrium).
    # A cross-period Fisher row uses m.pa[r,i,fd,'base'] → residual grows.
    # A frozen-constant row is unaffected → residual stays ~0 (BUG).
    for r in regions:
        for i in m.i:
            for a in fd_agents:
                with contextlib.suppress(KeyError, AttributeError):
                    m.pa[r, i, a, "base"].set_value(pv(m.pa[r, i, a, "base"]) * SCALE)
                with contextlib.suppress(KeyError, AttributeError):
                    m.xaa[r, i, a, "base"].set_value(pv(m.xaa[r, i, a, "base"]) * SCALE)
        for f in m.f:
            for a in m.a:
                with contextlib.suppress(KeyError, AttributeError):
                    m.pf[r, f, a, "base"].set_value(pv(m.pf[r, f, a, "base"]) * SCALE)
                with contextlib.suppress(KeyError, AttributeError):
                    m.xf[r, f, a, "base"].set_value(pv(m.xf[r, f, a, "base"]) * SCALE)

    # After perturbation: residuals must REACT (prove the constraint is live/cross-period)
    pabs_worst = max(_resid_pabs(r) for r in regions)
    pfact_worst = max(_resid_pfact(r) for r in regions)
    pwfact_resid = _resid_pwfact()

    assert pabs_worst > 0.1, (
        f"eq_pabs shock residual did NOT react to base-period perturbation "
        f"(worst={pabs_worst:.4e}): constraint still uses frozen constants instead of "
        f"live m.pa/m.xaa['base'] Vars — fix build_equations_fisher to add cross-period rows"
    )
    assert pfact_worst > 0.1, (
        f"eq_pfact shock residual did NOT react to base-period perturbation "
        f"(worst={pfact_worst:.4e}): constraint still uses frozen pf0/xf0/mqfactr_bb constants — "
        f"fix build_equations_fisher to add cross-period rows"
    )
    assert pwfact_resid > 0.1, (
        f"eq_pwfact shock residual did NOT react to base-period perturbation "
        f"(resid={pwfact_resid:.4e}): constraint still uses frozen pf0/xf0/mqfactw_bb constants — "
        f"fix build_equations_fisher to add cross-period rows"
    )
