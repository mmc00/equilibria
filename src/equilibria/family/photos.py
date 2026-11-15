"""The saved Pyomo photos the family (JAX) route is checked against.

A photo is a Pyomo run of one dataset and closure, captured by
``scripts/family/regen_photos.py`` on a machine with MUMPS and committed under
``data/gtap7/<dataset>/equilibria_pyomo-photo_<closure tag>/`` with its
``source.yaml``: the family tests read it instead of solving Pyomo.  Its
``model_print.json`` (``fingerprint.ModelPrint``) says what it was made from, so a
change in the Pyomo-only code makes it stale (see ``freshness``).

This module is the read side; writing needs Pyomo and lives in the script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from equilibria.sources import DATA_ROOT

PHOTO_FOLDER = "equilibria_pyomo-photo"
# the dataset files a photo's data print covers (the inputs _solve_shock loads)
DATA_FILES = ("basedata.har", "sets.har", "default.prm", "baserate.har")
LINSOLVE = "mumps"  # every photo is made with MUMPS (20x41 needs it)
REGEN = "scripts/family/regen_photos.py"

# (dataset, ifsub, savf, mode, keep): "all" keeps the per-step photos the prep tests
# read, "gate" only final.pkl, settled.pkl and the solve records.
VARIANTS: tuple[tuple[str, int, str, str, str], ...] = (
    ("gtap7_3x3", 1, "capFix", "gtap", "all"),
    ("gtap7_3x3", 0, "capFix", "gtap", "all"),
    ("gtap7_3x3", 1, "capFlex", "gtap", "all"),
    ("gtap7_10x7", 1, "capFix", "gtap", "all"),
    ("gtap7_15x10", 1, "capFix", "gtap", "gate"),
    ("gtap7_20x41", 1, "capFix", "gtap", "gate"),
)


def closure_tag(ifsub: int, savf: str, mode: str) -> str:
    """Folder suffix of a closure; today's default keeps the reference's old name."""
    if (ifsub, savf, mode) == (1, "capFix", "gtap"):
        return "tol_today"
    return f"{mode}_ifsub{ifsub}_{savf}"


def photo_dir(
    dataset: str,
    ifsub: int = 1,
    savf: str = "capFix",
    mode: str = "gtap",
    *,
    root: Path = DATA_ROOT,
) -> Path:
    family = dataset.split("_", 1)[0]
    return root / family / dataset / f"{PHOTO_FOLDER}_{closure_tag(ifsub, savf, mode)}"


def photo_closure(
    ifsub: int,
    savf: Literal["capFix", "capSFix", "capShrFix", "capFlex", "capFixDp"],
    unit_sigma: Literal["gams", "gempack"] = "gams",
):
    """The closure ``_solve_shock`` (the photos' Pyomo run) solves with."""
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=bool(ifsub),
        savf_flag=savf,
        numeraire="pnum",
        # GEMPACK is the reference: use the data's factor wedge (EVFP = EVFB +
        # FTRV + FBEP), the gtap7_gempack basis. The GAMS sign drops 15x10 to 93.9%.
        va_subsidy_basis="gempack",
        unit_sigma=unit_sigma,
    )
