"""``load``: GTAP parameters from a bundled base, a ``data/`` dataset or a folder."""

from __future__ import annotations

import os
from pathlib import Path

from equilibria import datasets, sources
from equilibria.templates.gtap import GTAPParameters


def _folder(source: str | os.PathLike) -> Path:
    name = os.fspath(source)
    if name in datasets.list_bundled("gtap"):
        return datasets.dataset_path("gtap", name)
    if sources.has_dataset(name):
        return sources.dataset_dir(name)
    if Path(name).is_dir():
        return Path(name)
    raise FileNotFoundError(
        f"GTAP source {name!r} not found: it is not a bundled base "
        f"({', '.join(datasets.list_bundled('gtap'))}), not a dataset of the "
        f"repository data/ folder, and not an existing folder"
    )


def load(source: str | os.PathLike, prm: str = "default.prm") -> GTAPParameters:
    """The parameters of a GTAP dataset, read from its HAR files.

    ``source`` is looked up in this order: the name of a base shipped with the package
    (``"nus333"``, ``"9x10"``), the name of a dataset of the repository ``data/``
    folder (only in a checkout, e.g. ``"gtap7_10x7"``), or a path to a folder that
    holds ``basedata.har``, ``sets.har``, the ``prm`` file and, if present,
    ``baserate.har``. ``prm`` picks the elasticities file of that folder
    (``"default.prm"``, ``"altertax.prm"``, ``"3x3ces.prm"``...). A missing file is a
    ``FileNotFoundError`` that names the folder and the file.
    """
    folder = _folder(source)
    files = {n: folder / n for n in ("basedata.har", "sets.har", prm)}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{folder} has no {name} (needed to load GTAP)")
    baserate = folder / "baserate.har"
    params = GTAPParameters()
    params.load_from_har(
        basedata_path=files["basedata.har"],
        sets_path=files["sets.har"],
        default_path=files[prm],
        baserate_path=baserate if baserate.exists() else None,
    )
    return params
