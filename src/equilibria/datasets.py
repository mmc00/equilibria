"""Bundled reference datasets shipped with `equilibria`.

`load_bundled(category, name)` returns a ready-to-use object built
from the canonical source files for one of the small datasets that
travel with the package (useful for tutorials, tests, and example
notebooks). Callers that need a custom aggregation should still go
through the underlying loaders directly.

Source-of-truth conventions per category:
  * ``"gtap"`` — native GEMPACK HAR/PRM files. That is what the official
    ``convert.cmd`` flow builds GDX from upstream.
  * ``"pep"`` — Excel workbooks (`SAM-V2_0.xlsx`, `VAL_PAR.xlsx`). That
    is the form in which the PEP-1-1 reference data is published.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

Category = Literal["gtap", "pep"]

_REFERENCE_ROOT = Path(__file__).parent / "templates" / "reference"

_GTAP_DATASETS = {
    "9x10": _REFERENCE_ROOT / "gtap" / "data" / "9x10",
    "nus333": _REFERENCE_ROOT / "gtap" / "data" / "nus333",
}

_PEP_DATASETS = {
    "default": _REFERENCE_ROOT / "pep",
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
    """Load a bundled dataset and return a ready-to-use object.

    For ``category="gtap"`` this always reads native HAR/PRM files
    (`basedata.har`, `sets.har`, `default.prm`, plus optional
    `baserate.har` for GTAPv7-style aggregations like NUS333) and
    returns a calibrated `GTAPParameters`.

    For ``category="pep"`` this always reads the canonical Excel
    workbooks (`SAM-V2_0.xlsx` and `VAL_PAR.xlsx`), converts the SAM
    on-the-fly to the 4D GDX layout the PEP calibrator consumes, and
    returns a `PEPModelCalibrator` ready to call `.calibrate()`. The
    intermediate GDX is written to a tmp directory under
    `~/.cache/equilibria/pep/` so subsequent calls reuse it.
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
        from equilibria.templates.data.pep.generate_sam_4d import generate_sam_4d_gdx
        from equilibria.templates.pep_calibration_unified import PEPModelCalibrator

        sam_xlsx = path / "SAM-V2_0.xlsx"
        val_par_xlsx = path / "VAL_PAR.xlsx"

        for required in (sam_xlsx, val_par_xlsx):
            if not required.exists():
                raise FileNotFoundError(
                    f"Bundled pep dataset {name!r} is missing {required.name} "
                    f"at {required}"
                )

        cache_dir = Path.home() / ".cache" / "equilibria" / "pep" / name
        cache_dir.mkdir(parents=True, exist_ok=True)
        sam_gdx = cache_dir / "SAM-V2_0_4D.gdx"

        if (
            not sam_gdx.exists()
            or sam_gdx.stat().st_mtime < sam_xlsx.stat().st_mtime
        ):
            generate_sam_4d_gdx(sam_xlsx, sam_gdx)

        return PEPModelCalibrator(sam_file=sam_gdx, val_par_file=val_par_xlsx)

    raise ValueError(f"Unknown category: {category!r}. Expected 'gtap' or 'pep'.")


__all__ = ["Category", "dataset_path", "list_bundled", "load_bundled"]
