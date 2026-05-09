"""Bundled reference datasets shipped with `equilibria`.

`load_bundled(category, name)` returns a fully calibrated parameter
container for one of the small datasets that travel with the package
(useful for tutorials, tests, and example notebooks). Callers that
need a custom aggregation should still go through
`GTAPParameters.load_from_har` / `load_from_gdx` directly.

For the ``"gtap"`` category, datasets are always loaded from the native
GEMPACK HAR/PRM files — that is the canonical GTAP distribution format
and what the official ``convert.cmd`` flow builds GDX from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

Category = Literal["gtap", "pep"]

_DATA_ROOT = Path(__file__).parent / "templates" / "reference"

_GTAP_DATASETS = {
    "9x10": _DATA_ROOT / "gtap" / "data" / "9x10",
    "nus333": _DATA_ROOT / "gtap" / "data" / "nus333",
}

_PEP_DATASETS = {
    "default": Path(__file__).parent / "templates" / "data" / "pep",
}


def list_bundled(category: Category) -> list[str]:
    """List dataset names available for a category."""
    if category == "gtap":
        return sorted(_GTAP_DATASETS)
    if category == "pep":
        return sorted(_PEP_DATASETS)
    raise ValueError(f"Unknown category: {category!r}. Expected 'gtap' or 'pep'.")


def dataset_path(category: Category, name: str) -> Path:
    """Return the directory holding a bundled dataset's raw files."""
    if category == "gtap":
        if name not in _GTAP_DATASETS:
            raise ValueError(
                f"Unknown gtap dataset: {name!r}. "
                f"Available: {sorted(_GTAP_DATASETS)}"
            )
        return _GTAP_DATASETS[name]
    if category == "pep":
        if name not in _PEP_DATASETS:
            raise ValueError(
                f"Unknown pep dataset: {name!r}. "
                f"Available: {sorted(_PEP_DATASETS)}"
            )
        return _PEP_DATASETS[name]
    raise ValueError(f"Unknown category: {category!r}. Expected 'gtap' or 'pep'.")


def load_bundled(category: Category, name: str = "default"):
    """Load a bundled dataset and return its calibrated parameter container.

    For ``category="gtap"`` this always reads native HAR/PRM files
    (`basedata.har`, `sets.har`, `default.prm`, plus optional
    `baserate.har` for GTAPv7-style aggregations like NUS333) and
    returns a ready-to-use `GTAPParameters`.

    For ``category="pep"`` it returns the bundled PEP SAM/calibration
    GDX paths as a dict — PEP loading is staged through several
    calibrators, so the caller picks which one to drive.
    """
    path = dataset_path(category, name)

    if category == "gtap":
        from equilibria.templates.gtap import GTAPParameters

        basedata = path / "basedata.har"
        sets_har = path / "sets.har"
        default_prm = path / "default.prm"
        baserate = path / "baserate.har"

        for required in (basedata, sets_har, default_prm):
            if not required.exists():
                raise FileNotFoundError(
                    f"Bundled gtap dataset {name!r} is missing {required.name} "
                    f"at {required}"
                )

        params = GTAPParameters()
        params.load_from_har(
            basedata_path=basedata,
            sets_path=sets_har,
            default_path=default_prm,
            baserate_path=baserate if baserate.exists() else None,
        )
        return params

    if category == "pep":
        return {
            "sam": path / "SAM-V2_0_4D.gdx",
            "sets": path / "pep_sets.gdx",
            "calibration": path / "pep_calibration.gdx",
            "val_par": path / "VAL_PAR.gdx",
        }

    raise ValueError(f"Unknown category: {category!r}. Expected 'gtap' or 'pep'.")


__all__ = ["Category", "dataset_path", "list_bundled", "load_bundled"]
