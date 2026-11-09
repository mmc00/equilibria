"""Gates of the ALTERTAX outputs against a GEMPACK run folder: the ONE implementation.

Every ``test_altertax_*`` test and ``scripts/family/run_altertax.py --compare`` call the
functions here. One section per output (updated, summary, volume, decomposition, % changes,
levels); each returns plain data (gaps, failing cells) and ``report_*`` turns it into the
record the command writes to ``compare.json``. Tolerances come from
:mod:`equilibria.family.altertax.tolerances`.

Extension point for a GEMPACK folder of another dataset (3x3 ... 20x41): the known-miss
lists below are keyed by cell and are nus333-specific; a key that does not occur in the
folder is simply not applied.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from equilibria.babel.har.reader import read_har
from equilibria.babel.har.symbols import HeaderArray
from equilibria.compare.measure import rel_diff
from equilibria.family.altertax import compare as _levels
from equilibria.family.altertax.decomp import DECOMP_HEADERS, decomp_headers
from equilibria.family.altertax.pct import gempack_changes, pct_changes, zero_base_cells
from equilibria.family.altertax.run import AltertaxRun
from equilibria.family.altertax.summary import SUMMARY_HEADERS, summary_headers
from equilibria.family.altertax.tolerances import (
    ABS_FLOOR,
    TOL_CHANGE,
    TOL_LEVELS,
    TOL_PCT,
    TOL_SUMMARY,
    TOL_UPDATED,
)
from equilibria.family.altertax.updated import UPDATED_HEADERS, updated_headers
from equilibria.family.altertax.volume import (
    VOLUME_HEADERS,
    volume_bases,
    volume_headers,
)

# --- GEMPACK run-folder names: GEMPACK's own first, then the burfisher fixture names ----
UPDATED_FILES = ("updated.har", "altax_updated.har", "altax.upd")
SUMMARY_FILES = ("summary.har", "altax_summary.har")
DECOMP_FILES = ("decomp.har", "altax_decomp.har")
VOLUME_FILES = ("volume.har", "altax_volume.har")
SL4_FILES = ("solution.sl4", "altax.sl4", "solution.sl4.gz", "altax.sl4.gz")
LEVELS_FILES = _levels.LEVELS_NAMES

THRESHOLD_ULPS = (
    8  # volume: cells above 8 float32 ulp of the level are compared relatively
)
NOISE = 1e-6  # decomposition: relative to max|EV| of the region (user 2026-11-09)
RATE_COLS = ("taxrateb", "taxrateu")  # decomp columns that are tax rates (levels)
TEXT_HEADERS = ("XXHS",)  # updated.har headers that are text
SUMMARY_LEVELS = ("GDPE", "COST", "GDPI", "SALE")
KNOWN_MISS_FACTOR = 1.5  # a known miss may grow to 1.5x its measured error


def ulp32(x: Any) -> np.ndarray:
    return np.spacing(
        np.abs(np.asarray(x, dtype=np.float64)).astype(np.float32)
    ).astype(np.float64)


def find_files(folder: Path, groups: dict[str, tuple[str, ...]]) -> dict[str, Path]:
    """The first existing name of each group in ``folder``; a clear error naming every
    missing one."""
    folder = Path(folder)
    found: dict[str, Path] = {}
    missing = []
    for key, names in groups.items():
        for n in names:
            if (folder / n).exists():
                found[key] = folder / n
                break
        else:
            missing.append(f"{key}: {' | '.join(names)}")
    if missing:
        raise FileNotFoundError(f"GEMPACK folder {folder} lacks: " + "; ".join(missing))
    return found


def read_sl4(path: Path):
    """``read_sl4_changes`` of a ``.sl4`` or ``.sl4.gz``."""
    from equilibria.babel.har.sl4 import read_sl4_changes

    path = Path(path)
    if path.suffix != ".gz":
        return read_sl4_changes(path)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / path.stem
        with gzip.open(path, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return read_sl4_changes(out)


# --- shared helpers -----------------------------------------------------------------


def worst_gap(a: Any, b: Any) -> tuple[float, tuple]:
    """Max of ``rel_diff(a, b, ref=b, ABS_FLOOR)`` over the cells, and its index."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.vectorize(lambda x, y: rel_diff(x, y, y, ABS_FLOOR))(a, b)
    i = np.unravel_index(int(np.nanargmax(d)), d.shape) if d.size else ()
    return float(d[i]) if d.size else 0.0, i


def gaps_by_header(
    ours: dict[str, HeaderArray], gem: dict[str, HeaderArray], names: Iterable[str]
) -> dict[str, tuple[float, tuple]]:
    return {n: worst_gap(ours[n].array, gem[n].array) for n in names}


def worst_cells(
    ours: dict[str, HeaderArray],
    gem: dict[str, HeaderArray],
    names: Iterable[str],
    n: int = 10,
) -> list[dict]:
    """The ``n`` cells with the largest relative gap over ``names``."""
    rows = []
    for name in names:
        a = np.asarray(ours[name].array, dtype=float)
        b = np.asarray(gem[name].array, dtype=float)
        for ix in np.ndindex(a.shape):
            rows.append(
                (rel_diff(a[ix], b[ix], b[ix], ABS_FLOOR), name, ix, a[ix], b[ix])
            )
    rows.sort(key=lambda r: (r[0] != r[0], r[0]), reverse=True)
    return [_cell(h, ix, m, g, d) for d, h, ix, m, g in rows[:n]]


def _cell(header: str, ix: Any, ours: float, gem: float, gap: float) -> dict:
    return {
        "header": header,
        "cell": [int(i) for i in ix],
        "ours": float(ours),
        "gempack": float(gem),
        "gap": float(gap),
    }


def structure_problems(
    ours: dict[str, HeaderArray], gem: dict[str, HeaderArray], names: Iterable[str]
) -> list[str]:
    """Headers whose sets, shape, coefficient or long name differ from GEMPACK's."""
    bad = []
    for name in names:
        a, b = ours[name], gem[name]
        if (
            a.set_names != b.set_names
            or a.set_elements != b.set_elements
            or a.array.shape != b.array.shape
            or (a.coeff_name, a.long_name) != (b.coeff_name, b.long_name)
        ):
            bad.append(name)
    return bad


def miss_is_bounded(err: float, ref: float) -> bool:
    """A known miss keeps its sign and stays within 1.5x of its measured error."""
    return err * ref > 0 and abs(err) <= KNOWN_MISS_FACTOR * abs(ref)


def _record(output, gate, max_, kind, worst, passed, **extra) -> dict:
    return {
        "output": output,
        "gate": gate,
        "max": max_,
        "max_kind": kind,
        "worst": worst,
        "pass": bool(passed),
        **extra,
    }


# --- updated.har ----------------------------------------------------------------------


def updated_failures(ours, gem) -> dict[str, tuple[float, tuple]]:
    """Numeric headers of ``updated.har`` above TOL_UPDATED."""
    names = [n for n in UPDATED_HEADERS if n not in TEXT_HEADERS]
    return {
        n: g
        for n, g in gaps_by_header(ours, gem, names).items()
        if not g[0] <= TOL_UPDATED
    }


def report_updated(ours, gem) -> dict:
    names = [n for n in UPDATED_HEADERS if n not in TEXT_HEADERS]
    gaps = gaps_by_header(ours, gem, names)
    bad_struct = structure_problems(ours, gem, UPDATED_HEADERS)
    mx = max((g[0] for g in gaps.values()), default=0.0)
    ok = not updated_failures(ours, gem) and not bad_struct
    return _record(
        "updated",
        {"TOL_UPDATED": TOL_UPDATED},
        mx,
        "relative",
        worst_cells(ours, gem, names),
        ok,
        known_miss=[],
        structure_problems=bad_struct,
    )


# --- summary.har ----------------------------------------------------------------------

# Accounting checks that are float32 residuals of GEMPACK's data and sums, so they are
# zero or a few ulp of the flow F they check (ulp32(F) is cell dependent) and a relative
# metric against them is meaningless: the headers below are NOT matched at TOL_SUMMARY
# and the test below fails if one starts to be. Measured cause per header:
#  * VMCK/VMCP/VTCK/VTCP (pre): GEMPACK's order of float32 accumulation of VCB, MAKEBCOM
#    and VTMUSE is not reproducible (a sequential float32 sum of the same basedata gives 3
#    of 6 VMCK cells, by chance); in float64 the data residual is another small number.
#  * VMCQ/TRCQ/VTCQ/MPCQ (post): GEMPACK's residuals on the updated data against the
#    exact ~0 of the float64 model. TRCQ and MPCQ are the input dataset's own float32
#    imbalance carried through GEMPACK's update (TRCK cells 3, 6, 7 equal TRCQ; float64 on
#    altax_updated.har reproduces them), not GEMPACK computing noise; VMCQ and VTCQ are
#    accumulation-order noise as VMCK and VTCK.
NOISE_UNMATCHED = frozenset(
    ["VMCK", "VMCP", "VTCK", "VTCP", "VMCQ", "TRCQ", "VTCQ", "MPCQ"]
)


def summary_failures(ours, gem, names) -> dict[str, tuple[float, tuple]]:
    return {
        n: g
        for n, g in gaps_by_header(ours, gem, names).items()
        if not g[0] <= TOL_SUMMARY
    }


def report_summary(ours, gem) -> dict:
    checks = [n for n in SUMMARY_HEADERS if n not in SUMMARY_LEVELS]
    bad_levels = summary_failures(ours, gem, SUMMARY_LEVELS)
    bad_checks = summary_failures(ours, gem, checks)
    unexpected = sorted(set(bad_checks) - NOISE_UNMATCHED)
    bad_struct = structure_problems(ours, gem, SUMMARY_HEADERS)
    gaps = gaps_by_header(ours, gem, SUMMARY_LEVELS)
    mx = max(g[0] for g in gaps.values())
    return _record(
        "summary",
        {"TOL_SUMMARY": TOL_SUMMARY, "gated": list(SUMMARY_LEVELS)},
        mx,
        "relative",
        worst_cells(ours, gem, SUMMARY_LEVELS),
        not bad_levels and not unexpected and not bad_struct,
        known_miss=sorted(set(bad_checks) & NOISE_UNMATCHED),
        failing_unexpected=unexpected,
        structure_problems=bad_struct,
    )


# --- volume.har -----------------------------------------------------------------------


def change_scan(
    name: str,
    ours: Any,
    gem: Any,
    thr: Callable[[tuple], float],
    *,
    level_cols: Callable[[tuple], bool] | None = None,
) -> tuple[list[tuple], list[tuple]]:
    """Cells of a change table above their noise threshold ``thr(ix)`` on either side are
    compared relatively (same sign, ``TOL_CHANGE``); the rest must agree in absolute terms
    within the threshold itself. ``level_cols(ix)`` marks a cell that is a level (compared
    relatively, no threshold). Returns ``(above, abs_bad)``; ``above`` holds
    ``(name, ix, ours, gem, rel, ok)``."""
    ours = np.asarray(ours, dtype=np.float64)
    gem = np.asarray(gem, dtype=np.float64)
    above, abs_bad = [], []
    for ix in np.ndindex(ours.shape):
        m, g = float(ours[ix]), float(gem[ix])
        t = thr(ix)
        d = rel_diff(m, g, g, ABS_FLOOR)
        if level_cols is not None and level_cols(ix):
            above.append((name, ix, m, g, d, d <= TOL_CHANGE))
        elif max(abs(m), abs(g)) > t:
            above.append((name, ix, m, g, d, m * g > 0 and d <= TOL_CHANGE))
        elif abs(m - g) > t:
            abs_bad.append((name, ix, m, g, t))
    return above, abs_bad


def volume_compare(name, ours, gem, level):
    """(cells above threshold, relative failures, absolute failures below it)."""
    thr = THRESHOLD_ULPS * ulp32(level)
    above, abs_bad = change_scan(name, ours, gem, lambda ix: float(thr[ix]))
    rel_bad = [(n, ix, m, g, d) for n, ix, m, g, d, ok in above if not ok]
    return len(above), rel_bad, abs_bad


def report_volume(run: AltertaxRun, ours, gem) -> dict:
    bases = volume_bases(run)
    above, abs_bad = [], []
    for name in VOLUME_HEADERS:
        thr = THRESHOLD_ULPS * ulp32(bases[name])
        a, b = change_scan(
            name, ours[name].array, gem[name].array, lambda ix: float(thr[ix])
        )
        above += a
        abs_bad += b
    bad = [c for c in above if not c[5]]
    worst = sorted(above, key=lambda c: c[4], reverse=True)[:10]
    bad_struct = structure_problems(ours, gem, VOLUME_HEADERS)
    return _record(
        "volume",
        {"TOL_CHANGE": TOL_CHANGE, "threshold_ulps": THRESHOLD_ULPS},
        max((c[4] for c in above), default=0.0),
        "relative",
        [_cell(n, ix, m, g, d) | {"ok": ok} for n, ix, m, g, d, ok in worst],
        not bad and not abs_bad and not bad_struct,
        known_miss=[],
        cells_compared=len(above),
        failing=[_cell(n, ix, m, g, d) for n, ix, m, g, d, _ in bad],
        failing_absolute=len(abs_bad),
        structure_problems=bad_struct,
    )


# --- decomp.har -----------------------------------------------------------------------


def decomp_scale(gem) -> np.ndarray:
    """max|EV| of each region: the largest |component| of its WELFARE row."""
    return np.abs(np.asarray(gem["A"].array, dtype=np.float64)).max(axis=1)


def region_scale(ha, ix, scale):
    """Scale of cell ``ix``: the largest scale of the regions it is indexed by."""
    pos = [i for i, n in enumerate(ha.set_names) if n == "REG"]
    return max(scale[ix[i]] for i in pos) if pos else scale.max()


def decomp_compare(name, ours_ha, gem_ha, scale):
    """(cells above the noise threshold, relative failures, absolute failures)."""
    last = gem_ha.set_elements[-1] if gem_ha.set_names else []

    def is_rate(ix):
        return (
            bool(gem_ha.set_names)
            and gem_ha.set_names[-1] == "COL"
            and last[ix[-1]] in RATE_COLS
        )

    above, abs_bad = change_scan(
        name,
        ours_ha.array,
        gem_ha.array,
        lambda ix: NOISE * region_scale(gem_ha, ix, scale),
        level_cols=is_rate,
    )
    rel_bad = [(n, ix, m, g, d) for n, ix, m, g, d, ok in above if not ok]
    return len(above), rel_bad, abs_bad


def _regional_totals(name, ha):
    """``{(header, part, region): total}``: the sum over every dimension but the region
    of a header's welfare columns. ``COL`` headers give ``welcnt`` and ``dvol`` (the tax
    rates are levels); the trade tax headers attribute a cell to the region the CNT uses
    (``xtax``/export: the source, axis 1; ``mtax``/import: the destination, axis 2)."""
    arr = np.asarray(ha.array, dtype=np.float64)
    regs = ha.set_elements[ha.set_names.index("REG")]
    parts = []  # (label, array, region axis)
    if name == "A23":
        parts = [("xtax", arr[..., 0], 1), ("mtax", arr[..., 1], 2)]
    elif name == "A231":
        for c, lab in enumerate(("welcnt", "dvol")):
            parts += [(f"{lab}/export", arr[:, :, :, c, 1], 1)]
            parts += [(f"{lab}/import", arr[:, :, :, c, 0], 2)]
    elif ha.set_names[-1] == "COL":
        ax = ha.set_names.index("REG")
        parts = [(lab, arr[..., c], ax) for c, lab in enumerate(("welcnt", "dvol"))]
    else:
        parts = [("", arr, ha.set_names.index("REG"))]
    out = {}
    for lab, a, ax in parts:
        tot = a.sum(axis=tuple(i for i in range(a.ndim) if i != ax))
        for r, t in zip(regs, tot, strict=True):
            out[(name, lab, r)] = float(t)
    return out


def total_gaps(headers, gempack, scale):
    """Relative gap of every regional total above the noise floor (``NOISE * max|EV|``)."""
    floor = NOISE * scale.max()
    gaps = {}  # two-sided: a total above the floor on EITHER side is gated
    for name in DECOMP_HEADERS:
        if name.startswith("C"):
            continue  # exogenous at zero: no cell above the floor
        ours = _regional_totals(name, headers[name])
        gem = _regional_totals(name, gempack[name])
        for k, g in gem.items():
            m = ours[k]
            if max(abs(g), abs(m)) > floor:
                gaps[k] = (m, g, rel_diff(m, g, g, ABS_FLOOR) if m * g > 0 else np.inf)
    return gaps


# Regional totals that miss TOL_CHANGE with the nominal trade balance closure, with the
# signed error (ours - GEMPACK) / |GEMPACK| measured at the 1.01 stand-in. Sweeps of the
# stand-in (sigma 1.01 / 1.005 / 1.0025, in %), the gap being linear in sigma - 1 where it
# is the stand-in:
#   A21 USA   -3.82 / -1.79 / -0.78      A24 ROW  -53.3 / -26.7 / -13.5 (6e-4 of EV)
#   A222 USA  +3.91 / +2.31 / +1.51      A23 xtax ROW  -4.08 / -1.95 / -0.83 (-> ~+0.3 at 1)
#   A231 dvol/export ROW  -3.59 / -1.72 / -0.73 (-> ~0 at 1)
#   A23 mtax USA  -30.8 / -14.5 / -6.3 (-> ~+2 at 1: mostly the stand-in, borderline)
# All of these are the stand-in except one. A231 dvol/import USA (-338 / -160 / -70 %,
# -> ~+18 % at sigma = 1) is an OPEN small-cell model residual, not the stand-in: it is the
# near-cancelling sum of three +-1.3e-2 cells (~4e-4 of EV) driven by the MFG ROW -> USA
# dvol (+186 % at 1.01, +39 % at 1.0025, ~-10 % at sigma = 1: -4.0e-3 vs GEMPACK -4.4e-3),
# the same family as Task 6's trade residual (DQXS/DQFM import cells at sigma -> 1).
# ``test_known_misses_are_bounded`` keeps each within 1.5x of its measured error (same
# sign), and still a miss: it fails when one starts to pass or gets worse.
DECOMP_KNOWN_MISS = {
    ("A21", "", "USA"): -0.0382,
    ("A211", "welcnt", "USA"): -0.0382,
    ("A222", "", "USA"): +0.0391,
    ("A22I", "welcnt", "USA"): +0.0391,
    ("A22I", "dvol", "USA"): +0.0286,
    ("A24", "", "ROW"): -0.5326,
    ("A241", "welcnt", "ROW"): -0.5326,
    ("A23", "mtax", "USA"): -0.3082,
    ("A231", "welcnt/import", "USA"): -0.3082,
    ("A23", "xtax", "ROW"): -0.0408,
    ("A231", "welcnt/export", "ROW"): -0.0408,
    ("A231", "dvol/export", "ROW"): -0.0359,
    ("A231", "dvol/import", "USA"): -3.3757,  # open residual, see above
}


def ev_gaps(ours, gem) -> list[float]:
    """Relative gap of the EV (sum of the WELFARE row) of each region."""
    ev = np.asarray(ours["A"].array, dtype=np.float64).sum(axis=1)
    g = np.asarray(gem["A"].array, dtype=np.float64).sum(axis=1)
    return [rel_diff(m, x, x, ABS_FLOOR) for m, x in zip(ev, g, strict=True)]


def report_decomp(ours, gem) -> dict:
    """Gate: EV of each region and each header's regional totals above the noise floor at
    TOL_CHANGE (controller ruling 2026-11-09); cell-level components are reported, not
    gated. Totals in ``DECOMP_KNOWN_MISS`` pass while they keep their sign and stay within
    1.5x of their measured error."""
    scale = decomp_scale(gem)
    ev = ev_gaps(ours, gem)
    gaps = total_gaps(ours, gem, scale)
    failing, known = [], []
    for k, (m, g, rel) in gaps.items():
        if rel <= TOL_CHANGE:
            continue
        err = (m - g) / abs(g)
        rec = {
            "total": list(k),
            "ours": m,
            "gempack": g,
            "gap": float(rel),
            "error": err,
        }
        if k in DECOMP_KNOWN_MISS and miss_is_bounded(err, DECOMP_KNOWN_MISS[k]):
            known.append(rec | {"measured": DECOMP_KNOWN_MISS[k]})
        else:
            failing.append(rec)
    cells = ok_cells = 0
    for name in DECOMP_HEADERS:
        n, r, _ = decomp_compare(name, ours[name], gem[name], scale)
        cells += n
        ok_cells += n - len(r)
    worst = sorted(
        [
            {"total": list(k), "ours": m, "gempack": g, "gap": float(r)}
            for k, (m, g, r) in gaps.items()
        ],
        key=lambda d: d["gap"],
        reverse=True,
    )[:10]
    bad_struct = structure_problems(ours, gem, DECOMP_HEADERS)
    ev_ok = max(ev) <= TOL_CHANGE
    return _record(
        "decomp",
        {"TOL_CHANGE": TOL_CHANGE, "noise": NOISE, "gated": "EV + regional totals"},
        max([*ev, *(r for *_, r in gaps.values() if np.isfinite(r))]),
        "relative",
        worst,
        ev_ok and not failing and not bad_struct,
        known_miss=known,
        failing=failing,
        ev_gap=[float(x) for x in ev],
        totals_compared=len(gaps),
        cells_within_tol=[ok_cells, cells],
        structure_problems=bad_struct,
    )


# --- pct (solution.sl4) ---------------------------------------------------------------


def pct_passes(m: float, g: float) -> bool:
    if abs(m - g) > max(TOL_CHANGE * abs(g), TOL_PCT):
        return False
    return not (abs(g) > 10 * TOL_PCT and m * g <= 0)


def pct_rel(m: float, g: float) -> float:
    return (m - g) / max(abs(g), TOL_PCT)


# Cells whose base flow is 0 (EVFB(LAND, MFG|SER, USA|ROW) = 0 in basedata.har: 8 cells, qes
# and qfe of LAND in MFG and SER of both regions) are excluded by rule. A % change on a zero
# flow is undefined; we report 0, GEMPACK reports the value of its price formula: E_qes2 holds
# for every cell regardless of the weight, and the 0.0013 comes from -ETRAE * (pes - pe) on a
# zero flow (qe itself is 0 in the .sl4). GEMPACK's own volume change DQFE = qfe / 100 * BEVFB
# zeroes it. Four of the 8 would pass anyway through the absolute floor (ROW).
ZERO_BASE = {
    ("qes", ("LAND", "MFG", "USA")),
    ("qes", ("LAND", "SER", "USA")),
    ("qfe", ("LAND", "MFG", "USA")),
    ("qfe", ("LAND", "SER", "USA")),
    ("qes", ("LAND", "MFG", "ROW")),
    ("qes", ("LAND", "SER", "ROW")),
    ("qfe", ("LAND", "MFG", "ROW")),
    ("qfe", ("LAND", "SER", "ROW")),
}

# Measured misses of the ALTERTAX shock at the 1.01 stand-in for the unit elasticities
# (user 2026-11-09: kept). Values are the relative error (ours - gem) / max(|gem|, TOL_PCT).
# - pfactor (USA +4.52, ROW -4.52): ~3e-5 pp. Sigma sweep of altertax.prm's unit elasticities
#   (1.01, 1.001, 1.0001), pfactor USA 2.50e-5, 2.91e-4, 2.95e-3 pp and ROW -8.8e-6, -1.0e-4,
#   -1.0e-3. It fits a + b / (sigma - 1) with b = +2.95e-7 (USA), -1.04e-7 (ROW) and
#   a = -4.2e-6 (USA), +1.5e-6 (ROW), against GEMPACK -7.1e-6 and +2.5e-6: a deterministic
#   term of our near-unit CES formulation, independent of the solver tolerance (TR_FTOL 1e-13
#   and RELTOL 1e-12 leave it unchanged). It is NOT the Fisher/Divisia numeraire difference
#   (about 1e-12 pp here). OPEN LEAD: the origin of the b / (sigma - 1) term is not found.
# - pgdp (USA -0.17, ROW +0.42), qgdp (USA +4.62, ROW -2.43): ~1e-6 pp. Flat in sigma (pgdp
#   USA 1.93e-5, 1.99e-5, 2.00e-5 against GEMPACK 2.31e-5; qgdp USA 4.02e-6, 4.07e-6, 4.07e-6
#   against 7.2e-7: b is 5e-11 and 3e-12), so the stand-in is not their cause: a residual of
#   the model at the 1e-6 pp scale, unexplained.
# - qpm(AGR,USA), qms(AGR,USA), qxs(AGR,ROW,USA): the small-cell trade residual of ROW->USA
#   imports of AGR (same family as Task 6's DQXS and Task 7's A23 residuals): qms and qxs
#   (identical, one flow) -4.50e-5 vs -4.75e-5 at 1.01, -5.32e-5 at 1.001, -5.39e-5 at 1.0001
#   (does not converge); qpm 2.79e-5, 2.03e-5, 1.96e-5 vs 1.91e-5 (converges: the stand-in).
PCT_KNOWN_MISS = {
    ("pfactor", ("USA",)): +4.5248,
    ("pfactor", ("ROW",)): -4.5248,
    ("pgdp", ("USA",)): -0.1665,
    ("pgdp", ("ROW",)): +0.4180,
    ("qgdp", ("USA",)): +4.6243,
    ("qgdp", ("ROW",)): -2.4310,
    ("qpm", ("AGR", "USA")): +0.4643,
    ("qms", ("AGR", "USA")): +0.0530,
    ("qxs", ("AGR", "ROW", "USA")): +0.0530,
}


def pct_compared(ours, gempack, zero=ZERO_BASE):
    """``{(var, cell): (ours, gem)}`` of the mapped variables, zero-base cells excluded."""
    return {
        (n, k): (ours[n][k], g)
        for n, cells in gempack.items()
        if n in ours
        for k, g in cells.items()
        if (n, k) not in zero
    }


def pct_failures(ours, gempack, zero=ZERO_BASE):
    return {
        key: pct_rel(m, g)
        for key, (m, g) in pct_compared(ours, gempack, zero).items()
        if not pct_passes(m, g)
    }


def report_pct(run: AltertaxRun, ours, gem) -> dict:
    """Gate per cell: ``|ours - gem| <= max(TOL_CHANGE * |gem|, TOL_PCT)`` and the same
    sign where ``|gem| > 10 * TOL_PCT``. Zero-base cells of this run are excluded."""
    zero = {(n, k) for n, cells in zero_base_cells(run).items() for k in cells}
    cells = pct_compared(ours, gem, zero)
    fails = pct_failures(ours, gem, zero)
    failing, known = [], []
    for key, err in fails.items():
        m, g = cells[key]
        rec = {
            "variable": key[0],
            "cell": list(key[1]),
            "ours": m,
            "gempack": g,
            "error": err,
        }
        if key in PCT_KNOWN_MISS and miss_is_bounded(err, PCT_KNOWN_MISS[key]):
            known.append(rec | {"measured": PCT_KNOWN_MISS[key]})
        else:
            failing.append(rec)
    errs = sorted(
        ((abs(m - g), key, m, g) for key, (m, g) in cells.items()),
        reverse=True,
    )[:10]
    worst = [
        {"variable": k[0], "cell": list(k[1]), "ours": m, "gempack": g, "abs_gap": d}
        for d, k, m, g in errs
    ]
    return _record(
        "pct",
        {"TOL_CHANGE": TOL_CHANGE, "TOL_PCT": TOL_PCT},
        max((abs(m - g) for m, g in cells.values()), default=0.0),
        "absolute (percentage points)",
        worst,
        not failing and bool(cells),
        known_miss=known,
        failing=failing,
        cells_compared=len(cells),
        zero_base_excluded=sorted([n, list(k)] for n, k in zero),
    )


# --- levels ---------------------------------------------------------------------------


def levels_ok(report) -> bool:
    """Levels gate: cells compared, max gap within TOL_LEVELS, no cell missing on a side."""
    return (
        report.overall["cells"] > 0
        and report.overall["max"] <= TOL_LEVELS
        and report.overall["missing_in_a"] == report.overall["missing_in_b"] == 0
        and not report.unmatched
    )


def report_levels(run: AltertaxRun, gempack_dir: Path, basedata: Path | None) -> dict:
    report = _levels.compare_levels(run, gempack_dir, basedata)
    worst = sorted(report.cells, key=lambda c: (c.d != c.d, c.d), reverse=True)[:10]
    mx = report.overall["max"]
    return _record(
        "levels",
        {"TOL_LEVELS": TOL_LEVELS},
        mx,
        "relative",
        [
            {
                "variable": c.variable,
                "period": c.period,
                "cell": list(c.index)
                if isinstance(c.index, (tuple, list))
                else c.index,
                "ours": c.a,
                "gempack": c.b,
                "gap": c.d,
            }
            for c in worst
        ],
        levels_ok(report),
        known_miss=[{"variable": e.variable, "why": e.why} for e in report.exclusions],
        cells=report.overall["cells"],
    )


# --- everything -----------------------------------------------------------------------


def gempack_files(gempack_dir: Path) -> dict[str, Path]:
    """The GEMPACK files ``compare_all`` reads, found by either naming (GEMPACK's own or
    the burfisher fixture's). ``sl4dump.har`` (Phase C folders) is optional and not read
    yet. Raises ``FileNotFoundError`` naming every missing file."""
    return find_files(
        gempack_dir,
        {
            "updated": UPDATED_FILES,
            "summary": SUMMARY_FILES,
            "decomp": DECOMP_FILES,
            "volume": VOLUME_FILES,
            "pct (solution.sl4)": SL4_FILES,
            "levels": LEVELS_FILES,
        },
    )


def compare_all(run: AltertaxRun, gempack_dir: Path) -> dict:
    """The six outputs of ``run`` against a GEMPACK run folder: ``{"outputs": [...],
    "pass": bool}``. A missing GEMPACK file is a ``FileNotFoundError`` naming it."""
    files = gempack_files(gempack_dir)
    basedata = Path(run.dataset_dir) / "basedata.har"
    outputs = [
        report_levels(run, gempack_dir, basedata),
        report_updated(updated_headers(run), read_har(str(files["updated"]))),
        report_summary(summary_headers(run), read_har(str(files["summary"]))),
        report_volume(run, volume_headers(run), read_har(str(files["volume"]))),
        report_decomp(decomp_headers(run), read_har(str(files["decomp"]))),
        report_pct(
            run,
            pct_changes(run),
            gempack_changes(read_sl4(files["pct (solution.sl4)"])),
        ),
    ]
    return {"outputs": outputs, "pass": all(o["pass"] for o in outputs)}
