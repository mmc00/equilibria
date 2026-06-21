"""Task 4: inter-temporal Fisher links as real Jacobian rows.

Contract: with all 3 periods seeded from the GAMS GDX, eq_rgdpmp[r,'shock']
residual must be < 1e-3 for all r.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/gtap"))

from test_multiperiod_sets import _load_3x3_params
from pyomo.environ import value as pv

REF = pathlib.Path(
    "/Users/marmol/proyectos2/equilibria_refs/gtap7_3x3_altertax_cd/out_altertax_ifsub0.gdx"
)


def test_fisher_rgdpmp_zero_at_gams_point():
    from equilibria.templates.gtap.gtap_model_multiperiod import GTAPMultiPeriodModel

    p = _load_3x3_params()
    rr = list(p.sets.r)[-1]
    mp = GTAPMultiPeriodModel(p.sets, p, None, residual_region=rr)
    m = mp.build_sets()
    mp.build_vars(m)
    mp.build_equations_intra(m, "base")
    mp.build_equations_intra(m, "check")
    mp.build_equations_intra(m, "shock")
    mp.build_equations_fisher(m)
    # Seed all periods from the GAMS GDX (base/check/shock)
    mp.seed_all_periods(m, REF)
    # eq_rgdpmp[*,shock] must give residual ~0 (base is now in the system)
    worst = max(
        abs(pv(m.eq_rgdpmp[r, "shock"].body) - float(m.eq_rgdpmp[r, "shock"].lower))
        for r in p.sets.r
    )
    assert worst < 1e-3, f"eq_rgdpmp shock residual {worst} > 1e-3"
