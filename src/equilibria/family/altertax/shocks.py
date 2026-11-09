"""Shocks of the ALTERTAX runs on the family route (no Pyomo).

``load_tms_shock`` reads the ``tms_rate10.shk`` of a RunGTAP AlterTax reference folder: the
header ``N002`` of ``shock tms(COMM,REG,REG) = select from file`` (GEMPACK ``tms(c,s,d)``:
commodity, SOURCE region, DESTINATION region), values in percent change of the tms power.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from equilibria.babel.har import read_har

TMS_HEADER_DIMS = ("COMM", "REG", "REG")


def read_spreadsheet_header(path: Path) -> np.ndarray:
    """The values of a one-header GEMPACK text file ``d1 d2 ... Real SpreadSheet Header``
    in file order, shaped as the header dimensions (see :func:`load_tms_shock` for what
    the order means)."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    dims = [int(t) for t in lines[0].split("Real")[0].split()]
    rows = [[float(x) for x in ln.replace(",", " ").split()] for ln in lines[1:]]
    flat = np.array([x for row in rows for x in row])
    if flat.size != int(np.prod(dims)):
        raise ValueError(f"{path}: {flat.size} values for dimensions {dims}")
    return flat.reshape(dims)


def load_tms_shock(
    path: Path, dataset_dir: Path | None = None
) -> dict[str, dict[tuple[str, str, str], float]]:
    """``{"imptx": {(source, commodity, destination): pct}}`` of a ``tms`` ``.shk`` file.

    Element names come from ``sets.har`` of ``dataset_dir`` (default: the folder above the
    run folder, i.e. the dataset). ``imptx[r,i,rp]`` is (exporter, good, importer) and the
    file's ``tms(c,s,d)`` is (good, source, destination), so the cell is ``(s, c, d)``.
    Zero entries are omitted (a 0 % shock leaves the cell at its check value, which the
    shock refuses). ``imptx`` is a GEMPACK ``power`` instrument: the percent is the change of
    ``1 + rate``, i.e. ``rate_new = (1 + rate)(1 + pct/100) - 1``."""
    path = Path(path)
    d = Path(dataset_dir) if dataset_dir is not None else path.resolve().parents[1]
    sets = read_har(d / "sets.har", select_headers=["COMM", "REG"])
    comm = [str(e).strip() for e in sets["COMM"].array]
    reg = [str(e).strip() for e in sets["REG"].array]
    raw = read_spreadsheet_header(path)
    if sorted(raw.shape) != sorted((len(comm), len(reg), len(reg))):
        raise ValueError(
            f"{path}: shape {raw.shape} does not fit (COMM, REG, REG) = "
            f"{(len(comm), len(reg), len(reg))} of {d / 'sets.har'}"
        )
    # The text lines hold the SOURCE index across (fastest), then the commodity, the
    # destination slowest: the file is tms(s, c, d) in Fortran order although the header
    # names the dimensions (COMM, REG, REG). Measured on nus333 and gtap7_10x7: only this
    # order gives 10 * rate / (1 + rate) against baserate.har for every cell.
    flat = raw.ravel()
    arr = flat.reshape(len(reg), len(comm), len(reg)).transpose(1, 2, 0)  # (c, s, d)
    cells = {
        (reg[s], comm[c], reg[t]): float(arr[c, s, t])
        for c, s, t in np.ndindex(arr.shape)
        if arr[c, s, t] != 0.0
    }
    return {"imptx": cells}
