"""capFlex ``risk[r]`` at the benchmark: the GAMS values and today's capFix twin (MAR-251).

``RISK_GDX``: GAMS's own ``risk(r,'base')`` parameter (``cal.gms:684``), read by hand
with GAMS 53's ``gdxdump`` (15 significant digits), not with equilibria code::

    gdxdump data/gtap7/<dataset>/gams_mcp_gtap_ifsub<i>_fbep-<f>/out.gdx symb=risk

The values are the same for ifSUB=0 and ifSUB=1 in every dataset that has both, so one
entry per dataset. ``nus333`` is Burfisher's dataset (``run_burfisher.py``).

``RISK_TWIN``: what Pyomo uses today when it has no GDX (Burfisher passes
``ref_gdx=None``): ``_calibrate_capflex_risk`` solving a capFix twin, on origin/main
7f5b8ad9 with ``EQUILIBRIA_SEED_CACHE_DISABLE=1``. ``gtap7_3x3`` is copied from
``tests/blocks/gtap/golden/calibrate_base_main.json``; ``nus333`` was measured with
``default.prm`` and with ``sectorspecifick.prm`` (same digits) on ifSUB=0. The twin
differs from GAMS by about 1e-7 relative (its Newton tolerance).
"""

from __future__ import annotations

# dataset -> (gdx path relative to the repo root, {region: risk})
RISK_GDX: dict[str, tuple[str, dict[str, float]]] = {
    "nus333": (
        "data/gtap7/nus333/gams_mcp_gtap_ifsub0_fbep-zero/out.gdx",
        {"USA": 1.37836000356937, "ROW": 0.933317215521012},
    ),
    "gtap7_3x3": (
        "data/gtap7/gtap7_3x3/gams_mcp_gtap_ifsub0_fbep-header/out.gdx",
        {"USA": 0.861785013838015, "EU_28": 0.838245599394536, "ROW": 1.09635233032442},
    ),
    "gtap7_3x4": (
        "data/gtap7/gtap7_3x4/gams_mcp_gtap_ifsub0_fbep-header/out.gdx",
        {
            "USA": 0.861674133918,
            "EU_28": 0.838137748128244,
            "EGY": 0.371201737151377,
            "ROW": 1.10207033329357,
        },
    ),
    "gtap7_5x5": (
        "data/gtap7/gtap7_5x5/gams_mcp_gtap_ifsub0_fbep-header/out.gdx",
        {
            "USA": 0.830082675740001,
            "EU_28": 0.807409184814034,
            "CHN": 2.26764290874778,
            "LatinAmer": 0.608991955792814,
            "ROW": 0.828504216861169,
        },
    ),
    "gtap7_10x7": (
        "data/gtap7/gtap7_10x7/gams_mcp_gtap_ifsub0_fbep-header/out.gdx",
        {
            "USA": 0.832234855040761,
            "EU_28": 0.809502577730261,
            "CHN": 2.27352236343425,
            "JPN": 0.966579980119289,
            "IND": 1.13989432727318,
            "SSA": 0.683966022242548,
            "ROW": 0.741907566738615,
        },
    ),
}

# dataset -> {region: risk} of today's capFix twin (default.prm, ifSUB=0)
RISK_TWIN: dict[str, dict[str, float]] = {
    "gtap7_3x3": {
        "USA": 0.8617849990844603,
        "EU_28": 0.8382456789409285,
        "ROW": 1.0963523187070126,
    },
    "nus333": {"USA": 1.3783599368259598, "ROW": 0.9333172306224439},
}

# gdxdump prints 15 significant digits. Measured formula vs GAMS: nus333 < 1e-14,
# gtap7_* up to 9.7e-11 (gtap7_10x7); the tolerance sits an order above that.
GDX_REL = 1e-9
# the twin is a Newton solve; measured gap to GAMS
TWIN_REL = 1e-6
