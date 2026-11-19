"""Two-minute demo: a GTAP tariff shock and the PEP reference shock checked against GAMS."""
import contextlib
import copy
import io
import time
from pathlib import Path

from pyomo.environ import value

import equilibria.templates
from equilibria import load_bundled
from equilibria.core.driver import NOT_SOLVED
from equilibria.core.exercise import Exercise
from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
from equilibria.templates.gtap.recipe import gtap_static
from equilibria.templates.gtap.template import GTAP
from equilibria.templates.gtap.welfare_regional import regional_ev
from equilibria.templates.pep_calibration_unified import PEPModelCalibrator
from equilibria.templates.pep_pyomo.pep_pyomo_equations import build_pep_model
from equilibria.templates.pep_pyomo.pep_pyomo_scenarios import apply_sim1_export_tax_cut
from equilibria.templates.pep_pyomo.pep_pyomo_solver import solve_pep


def titulo(texto):
    print(f"\n{'=' * 64}\n  {texto}\n{'=' * 64}")


def gtap_tariff_demo():
    """+10% tariff on US manufacturing exports to ROW, on nus333: base (the data) ->
    check -> shock on the core driver. Returns (solved Pyomo model, codes per period:
    1 = solved, as the old driver counted them; the base is the data, not solved)."""
    params = load_bundled("gtap", "nus333")
    closure = GTAPClosureConfig(name="base", closure_type="MCP", capital_mobility="sluggish",
                                fix_endowments=False, fix_taxes=False, fix_technology=False,
                                if_sub=False, numeraire="pnum")
    ex = Exercise(GTAP(params, closure, residual_region="ROW"),
                  gtap_static("gtap", solve_check=True),
                  shock={"imptx": {("USA", "MFG", "ROW"): 10.0}})
    res = ex.run()
    codes = {
        t: 1 if r["code"] in ("optimal", NOT_SOLVED) else int(r["code"].removeprefix("code "))
        for t, r in res.items()
    }
    return ex.model.pyomo, codes


if __name__ == "__main__":
    titulo("1. GTAP Standard 7: +10% tariff on US manufacturing exports to ROW")
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):  # hide the solver's internal messages
        m, codes = gtap_tariff_demo()
    print(f"  Solved: {'yes' if codes['shock'] == 1 else 'no'}   ({time.time() - t0:.1f} s, no GAMS or GEMPACK)")
    print("  Welfare change, equivalent variation:")
    for reg, ev in regional_ev(m).items():
        print(f"    {reg:<5}{ev:>10.3f}")

    titulo("2. PEP-1-1 single-country CGE: export taxes cut by 25%")
    data = Path(equilibria.templates.__file__).parent / "reference" / "pep2" / "data"
    state = PEPModelCalibrator(sam_file=data / "SAM-V2_0.gdx", val_par_file=data / "VAL_PAR.xlsx").calibrate()
    shocked = apply_sim1_export_tax_cut(copy.deepcopy(state), factor=0.75)
    with contextlib.redirect_stdout(io.StringIO()):  # hide the solver's internal messages
        mp = build_pep_model(shocked, variant="base", form="mcp")
        codigo = solve_pep(mp).code
    print(f"  Solved: {'yes' if codigo == 1 else 'no'}")
    print(f"  GDP at basic prices   equilibria: {value(mp.GDP_BP):,.1f}   GAMS reference: 46,748.2")
