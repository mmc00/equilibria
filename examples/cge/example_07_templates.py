"""Example 7: a model template, Hosoe's standard CGE model (simple_open)

This example shows how to:
1. Build SimpleOpenEconomy (gamslib stdcge) from its SAM
2. Run base, check and shock (every import tariff set to zero) as an
   Exercise: the template under its recipe, flown by the core driver
3. Read the change in imports and the equivalent variation
"""

import pyomo.environ as pyo

from equilibria.core import Exercise
from equilibria.templates import SimpleOpenEconomy
from equilibria.templates.simple_open.template import equivalent_variation


def main():
    """Run the tariff abolition of Hosoe's chapter 6."""
    print("=" * 70)
    print("Example 7: Hosoe's standard CGE model (simple_open)")
    print("=" * 70)

    template = SimpleOpenEconomy()
    info = template.get_info()
    print(f"\n{template!r}")
    print(f"  SAM: {info['sam']}")
    print(f"  Goods: {info['goods']}   Factors: {info['factors']}")
    print(f"  Armington sigma: {info['sigma']}   CET psi: {info['psi']}")
    print(f"  Blocks: {', '.join(info['blocks'])}")

    ex = Exercise(template, shock={"tau_m": 0.0})
    results = ex.run()
    m = ex.model.pyomo  # the Pyomo model: every variable indexed by period
    for period, r in results.items():
        print(f"  {period:<6} {r['code']:<11} residual {r['residual']:.1e}")

    print("\nAbolition of import tariffs: change from the base (%)")
    print(f"  {'good':<6} {'imports':>9} {'exports':>9} {'consumption':>12}")
    for good in info["goods"]:
        dm = 100 * (pyo.value(m.M[good, "shock"]) / pyo.value(m.M[good, "base"]) - 1)
        de = 100 * (pyo.value(m.E[good, "shock"]) / pyo.value(m.E[good, "base"]) - 1)
        dxp = 100 * (pyo.value(m.XP[good, "shock"]) / pyo.value(m.XP[good, "base"]) - 1)
        print(f"  {good:<6} {dm:9.3f} {de:9.3f} {dxp:12.3f}")

    ev = equivalent_variation(template.calibration(), ex.model)
    print(f"\nEquivalent variation: {ev:.6f}")
    print(f"Walras slack at the shock: {pyo.value(m.WALRAS['shock']):.1e}")


if __name__ == "__main__":
    main()
