"""``equilibria.compare.gtap7_side``: the equilibria half of ``maps/gtap7.yaml``.

The GEMPACK translation (``equilibria_side``, ``EQ_VARS``, ``model_levels``) lived in
``scripts/gtap/compare_gempack_levels.py``; MAR-249 moved it into the package so code
installed with pip does not load a repo script. The script keeps the names (re-exported)
for the repo scripts that still load it. The numbers are pinned by
``test_gtap7_side_golden.py`` (recorded on main before the move).
"""

from __future__ import annotations

from equilibria.compare import gtap7_side
from equilibria.family.altertax.compare import load_compare_script


def test_script_reexports_the_package_names() -> None:
    cg = load_compare_script()
    for name in (
        "equilibria_side",
        "EQ_VARS",
        "model_levels",
        "MARGIN_AGENT",
        "FINAL_AGENTS",
        "BASE",
    ):
        assert getattr(cg, name) is getattr(gtap7_side, name), name
