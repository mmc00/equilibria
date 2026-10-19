"""Export pre- and post-simulation LEVELS from any GEMPACK SL4 solution to a plain HAR.

The `sltoht` HAR/VAI output carries only the cumulative %-changes (the
`sl4dump_*.har` fixtures). But the SL4 solution also holds the pre-/post-simulation
LEVELS (headers LEVB/LEVA), which sltoht exposes ONLY in its TEXT modes under option
SHL: each variable prints 4 solution columns
    [ %-change , pre-sim level (LEVB) , post-sim level (LEVA) , change ].
This script drives `sltoht -SHL -SIC` (solutions-in-columns text) on the given .sl4,
or reads a saved copy of that text (`--from-text`, the only mode that runs off
Windows), parses BOTH level columns and writes them as a HAR.

Component order in the text equals the variable's HAR flat order (FORTRAN /
column-major), so shapes / set-elements / long-name / numeric id come from the
matching `sl4dump` (`--sl4dump`); each variable is CROSS-CHECKED (the text's
%-change column must equal the sl4dump values) before its levels are kept, and its
LEVELS are then VALIDATED cell by cell against the solution (`check_levels`):
post - pre must equal the change column, and 100*(post/pre - 1) must equal the
%-change (percentage-change variables) or the solution column must equal the
change column (ordinary-change variables, `is_ordinary_change`), to LEVEL_RTOL.
A contradiction aborts the export naming the variable and the cell.

Output HAR layout (read by the compare source, #131):
  * LEVA (post-sim): header = the sl4dump numeric id (`"0052"`), long_name and
    coeff_name copied from the sl4dump (`"qo # output of activity ... #"`). This is
    exactly the layout of the historical LEVA-only `sl4levels_*.har` fixtures, so
    `gempack_reference.sl4_levels(path, "qo")` keeps returning the post-sim level.
  * LEVB (pre-sim): header = `"B"` + the last 3 digits of that id (`"B052"`),
    long_name = `"LEVB "` + the sl4dump long_name (cut to the 70-byte HAR limit).
    The prefix keeps LEVB out of the name->id lookup of `sl4_levels`.
  * A variable with NO level in the SL4 (GEMPACK's sentinel 1e10 in both level
    columns) gets NO LEVA/LEVB header: the sentinel is never stored as a level.
    Its name is listed instead in the 1-D character header `NOLV` (always written,
    empty if every variable has levels). A sentinel on only some cells of a
    variable is an error.
  * A HAR without any `B***` / `NOLV` header is LEVA-only (fixtures made before
    #127) and may still hold 1e10-sentinel arrays as if they were levels.

Usage (from repo root, Windows; default --out = sl4levels_<suffix of the sl4dump>.har
next to the sl4dump):
    uv run python scripts/gtap/export_sl4_levels.py ^
        runs/gempack_matrix/gtap7_20x41/tm10_s8-16-32_capfix.sl4 ^
        --sl4dump tests/fixtures/gtap7_gempack/sl4dump_gtap7_20x41_tm10_s8-16-32_capfix.har ^
        --sltoht C:/GP/sltoht.exe
Any OS, from a saved sltoht text:
    uv run python scripts/gtap/export_sl4_levels.py levels.txt --from-text --sl4dump X.har
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from equilibria.babel.har.reader import read_har  # noqa: E402

# The LEVA/LEVB/NOLV convention lives in ONE place, shared with the compare reader.
from equilibria.babel.har.sl4_levels import (  # noqa: E402,F401
    LEVB_PREFIX,
    LONG_NAME_WIDTH,
    NO_LEVEL,
    NO_LEVEL_HEADER,
    NO_LEVEL_LONG_NAME,
    levb_header,
    levb_long_name,
)
from equilibria.babel.har.symbols import HeaderArray  # noqa: E402
from equilibria.babel.har.writer import write_har  # noqa: E402

DEFAULT_SLTOHT = Path(r"C:\GP\sltoht.exe")
PCT_COL, PRE_COL, POST_COL, CHANGE_COL = 0, 1, 2, 3  # [%-change, PRE, POST, change]
# Level-consistency tolerance. sltoht prints every solution column with ~8
# significant digits (guide 8a: `-1.5210288 1579235.5 1555214.9 -24020.625`; the SL4
# stores levels as REAL*4), i.e. each printed number carries a relative rounding of
# at most ~6e-8. Propagated: post/pre is off by <=~1.2e-7 relative, so the
# %-change implied by the levels, 100*(post/pre-1), is off by <=~1.2e-7*(100+|pct|)
# pp; post-pre is off by <=~6e-8*(|pre|+|post|). LEVEL_RTOL = 1e-6 gives an ~8x
# margin over that rounding and is still ~1e4x tighter than any real mistake
# (a swapped or misaligned column moves these by percents).
LEVEL_RTOL = 1e-6
_VAR_RE = re.compile(r"^\s*! Variable (\S+) #")
_ROW_RE = re.compile(r"!%1=\s*(\d+)\s*$")


def run_sltoht_sic(sl4: Path, sltoht: Path) -> Path:
    """Drive `sltoht -SHL -SIC` on `sl4` -> levels text `<stem>_levels.txt` beside it."""
    run_dir, stem = sl4.parent, sl4.stem
    out = run_dir / f"{stem}_levels.txt"
    sti = run_dir / f"{stem}_levels.sti"
    # options: SHL (show levels) + SIC (solutions-in-columns text); blank = finish
    # options; <stem> = solution; c = levels+cumulative; out name; e = exit.
    sti.write_text(f"SHL\nSIC\n\n{stem}\nc\n{out.name}\ne\n", encoding="ascii")
    console = run_dir / f"{stem}_levels_console.txt"
    with console.open("w", encoding="utf-8", errors="replace") as fh:
        subprocess.run([str(sltoht), "-sti", sti.name], cwd=run_dir,
                       stdout=fh, stderr=subprocess.STDOUT, check=False)
    return out


Columns = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def parse_levels(text_path: Path) -> dict[str, Columns]:
    """Parse the SIC text -> {var_name: (pct_change, pre_level, post_level, change)}
    in the text's component order (= the HAR flat order)."""
    out: dict[str, Columns] = {}
    cur: str | None = None
    cols: list[list[float]] = [[], [], [], []]

    def flush() -> None:
        if cur and cols[0]:
            out[cur] = tuple(np.array(v, dtype=np.float64) for v in cols)  # type: ignore[assignment]

    for line in text_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _VAR_RE.match(line)
        if m:
            flush()
            cur, cols = m.group(1), [[], [], [], []]
            continue
        if cur and _ROW_RE.search(line):
            vals = [float(x) for x in line.split("!", 1)[0].split()]
            if len(vals) >= 4:
                for k, col in enumerate((PCT_COL, PRE_COL, POST_COL, CHANGE_COL)):
                    cols[k].append(vals[col])
    flush()
    return out


class LevelMismatchError(ValueError):
    """The level columns of a variable contradict its %-change / change columns."""


def _cell_label(tmpl: HeaderArray, k: int) -> str:
    """Flat FORTRAN-order component k (0-based) -> '(elem1, elem2, ...)'."""
    if not tmpl.set_elements:
        return f"(component {k + 1})"
    idx = np.unravel_index(k, tmpl.array.shape, order="F")
    return "(" + ", ".join(str(e[i]) for e, i in zip(tmpl.set_elements, idx, strict=True)) + ")"


def is_ordinary_change(pct: np.ndarray, pre: np.ndarray, post: np.ndarray,
                       change: np.ndarray) -> bool:
    """True if the variable is a GEMPACK ordinary-CHANGE variable (TABLO `(change)`,
    e.g. del_tbal) rather than a percentage-change one.

    sltoht's -SHL text carries no explicit kind tag that we can rely on (the sl4dump
    HAR has none either), but its 4th column does: it is the ordinary change in the
    level, post - pre, for BOTH kinds. For a change variable the solution column IS
    that ordinary change, so column 1 == column 4 on every cell; for a %-change
    variable column 1 is 100*(post/pre-1) instead. The kind is decided per VARIABLE
    (all cells with levels), never per cell. A variable whose two columns coincide
    on every cell (e.g. an unshocked one, all zeros) is consistent with both readings
    and is validated as a change variable, which still enforces post - pre == change.
    """
    real = (pre != NO_LEVEL) & (post != NO_LEVEL)
    same = np.abs(pct - change) <= LEVEL_RTOL * (np.abs(pct) + np.abs(change))
    return bool(np.all(same[real]))


def check_levels(var: str, tmpl: HeaderArray, pct: np.ndarray, pre: np.ndarray,
                 post: np.ndarray, change: np.ndarray) -> None:
    """Raise LevelMismatchError naming `var` and the first bad cell unless, on every
    cell with real levels (neither is the 1e10 sentinel):
      * always:                 post - pre       == change (4th column);
      * %-change variables:     100*(post/pre-1) == %-change (1st column; skipped
                                where pre == 0, where the ratio is undefined);
      * ordinary-change ones:   1st column == change (see is_ordinary_change);
    all within LEVEL_RTOL (see its comment)."""
    real = (pre != NO_LEVEL) & (post != NO_LEVEL)
    diff_ok = np.abs((post - pre) - change) <= LEVEL_RTOL * (
        np.abs(pre) + np.abs(post) + np.abs(change))
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = 100.0 * (post / pre - 1.0)
    if is_ordinary_change(pct, pre, post, change):
        pct_ok = np.ones_like(real)  # col1 == col4 already, and diff_ok checks col4
    else:
        pct_ok = (pre == 0.0) | (
            np.abs(implied - pct) <= LEVEL_RTOL * (100.0 + np.abs(pct)))
    bad = np.flatnonzero(real & ~(diff_ok & pct_ok))
    if bad.size:
        k = int(bad[0])
        raise LevelMismatchError(
            f"{var} {_cell_label(tmpl, k)}: levels contradict the solution columns — "
            f"pre={pre[k]!r} post={post[k]!r} %-change={pct[k]!r} change={change[k]!r} "
            f"(100*(post/pre-1)={implied[k]!r}, post-pre={post[k] - pre[k]!r}); "
            f"{bad.size} bad cell(s)")


def build_levels_har(levels: dict, sl4dump: Path, out_har: Path) -> int:
    """Write a HAR of LEVA + LEVB levels (layout in the module docstring), taking
    shape/sets/id/long_name from the %-change sl4dump and cross-checking the
    %-change column per variable. Returns the number of variables kept."""
    src = read_har(str(sl4dump))
    name_to_hid = {
        (ha.long_name or "").split("#", 1)[0].strip(): hid
        for hid, ha in src.items() if (ha.long_name or "").strip()
    }
    headers: dict[str, HeaderArray] = {}
    kept = mismatched = 0
    no_level: list[str] = []
    for var, (pct_vec, pre_vec, post_vec, chg_vec) in levels.items():
        hid = name_to_hid.get(var)
        if hid is None:
            continue
        tmpl = src[hid]
        if tmpl.array.dtype.kind not in "fiu" or tmpl.array.size != post_vec.size:
            continue
        # cross-check: the text %-change column must equal the sl4dump values.
        # GEMPACK writes components in FORTRAN (column-major) order — verified
        # against all 252 gtap7_3x3 variables (C-order agrees only for rank-1).
        ref = np.asarray(tmpl.array, dtype=np.float64).ravel(order="F")
        if not np.allclose(ref, pct_vec, atol=1e-3, rtol=1e-3):
            mismatched += 1
            continue
        sentinel = (pre_vec == NO_LEVEL) | (post_vec == NO_LEVEL)
        if sentinel.all():
            no_level.append(var)
            continue
        if sentinel.any():
            k = int(np.flatnonzero(sentinel)[0])
            raise LevelMismatchError(
                f"{var} {_cell_label(tmpl, k)}: no-level sentinel 1e10 on only some "
                f"cells ({int(sentinel.sum())}/{sentinel.size}); GEMPACK levels are "
                f"per variable, so this text is inconsistent")
        check_levels(var, tmpl, pct_vec, pre_vec, post_vec, chg_vec)
        for name, long_name, vec in (
            (hid, tmpl.long_name, post_vec),
            (levb_header(hid), levb_long_name(tmpl.long_name), pre_vec),
        ):
            headers[name] = HeaderArray(
                name=name, coeff_name=tmpl.coeff_name, long_name=long_name,
                array=vec.reshape(tmpl.array.shape, order="F").astype(np.float32),
                set_names=list(tmpl.set_names),
                set_elements=[list(e) for e in tmpl.set_elements],
            )
        kept += 1
    if not headers:
        print(f"  no levels parsed from {sl4dump.name} — nothing written")
        return 0
    headers[NO_LEVEL_HEADER] = HeaderArray(
        name=NO_LEVEL_HEADER, coeff_name=NO_LEVEL_HEADER, long_name=NO_LEVEL_LONG_NAME,
        array=np.array(no_level, dtype=object), set_names=[], set_elements=[],
    )
    write_har(str(out_har), headers)
    print(f"  levels vars={kept} (LEVB+LEVA)  no-level={len(no_level)} ({NO_LEVEL_HEADER})"
          f"  (xcheck-mismatch={mismatched})  -> {out_har}")
    return kept


def default_out(sl4dump: Path) -> Path:
    """sl4dump_<x>.har -> sl4levels_<x>.har in the same folder."""
    return sl4dump.with_name(sl4dump.name.replace("sl4dump_", "sl4levels_", 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path,
                    help="the .sl4 solution, or (with --from-text) its saved sltoht -SHL -SIC text")
    ap.add_argument("--sl4dump", type=Path, required=True,
                    help="matching %%-change HAR (sltoht HAR export) of the same solution")
    ap.add_argument("--out", type=Path, default=None,
                    help="output HAR (default: sl4levels_<suffix>.har beside --sl4dump)")
    ap.add_argument("--from-text", action="store_true",
                    help="SOURCE is a saved sltoht text; do not run sltoht")
    ap.add_argument("--sltoht", type=Path, default=DEFAULT_SLTOHT)
    args = ap.parse_args(argv)

    if not args.sl4dump.exists():
        print(f"ERROR: sl4dump not found: {args.sl4dump}", file=sys.stderr)
        return 2
    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        return 2
    text = args.source
    if not args.from_text:
        if args.source.suffix.lower() != ".sl4":
            print(f"ERROR: {args.source} is not a .sl4 (pass --from-text for a saved "
                  f"sltoht text)", file=sys.stderr)
            return 2
        if not args.sltoht.exists():
            print(f"ERROR: sltoht not found at {args.sltoht}", file=sys.stderr)
            return 2
        text = run_sltoht_sic(args.source, args.sltoht)
        if not text.exists():
            print(f"ERROR: sltoht produced no levels text ({text}); see the "
                  f"*_levels_console.txt beside the .sl4", file=sys.stderr)
            return 1

    out = args.out or default_out(args.sl4dump)
    try:
        kept = build_levels_har(parse_levels(text), args.sl4dump, out)
    except LevelMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
