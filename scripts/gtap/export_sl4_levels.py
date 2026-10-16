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
%-change column must equal the sl4dump values) before its levels are kept.

Output HAR layout (read by the compare source, #131):
  * LEVA (post-sim): header = the sl4dump numeric id (`"0052"`), long_name and
    coeff_name copied from the sl4dump (`"qo # output of activity ... #"`). This is
    exactly the layout of the historical LEVA-only `sl4levels_*.har` fixtures, so
    `gempack_reference.sl4_levels(path, "qo")` keeps returning the post-sim level.
  * LEVB (pre-sim): header = `"B"` + the last 3 digits of that id (`"B052"`),
    long_name = `"LEVB "` + the sl4dump long_name (cut to the 70-byte HAR limit).
    The prefix keeps LEVB out of the name->id lookup of `sl4_levels`.
  * A variable with no level in the SL4 keeps GEMPACK's sentinel (1e10) in both.
  * A HAR without any `B***` header is LEVA-only (fixtures made before #127).

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
from equilibria.babel.har.symbols import HeaderArray  # noqa: E402
from equilibria.babel.har.writer import write_har  # noqa: E402

DEFAULT_SLTOHT = Path(r"C:\GP\sltoht.exe")
PRE_COL, POST_COL = 1, 2  # 0-based: [%-change, PRE-level, POST-level, change]
LEVB_PREFIX = "LEVB "
LONG_NAME_WIDTH = 70
_VAR_RE = re.compile(r"^\s*! Variable (\S+) #")
_ROW_RE = re.compile(r"!%1=\s*(\d+)\s*$")
_LEVA_ID_RE = re.compile(r"^0\d{3}$")


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


def parse_levels(text_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Parse the SIC text -> {var_name: (pct_change, pre_level, post_level)} in the
    text's component order (= the HAR flat order)."""
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cur: str | None = None
    pct: list[float] = []
    pre: list[float] = []
    post: list[float] = []

    def flush() -> None:
        if cur and pct:
            out[cur] = tuple(np.array(v, dtype=np.float64) for v in (pct, pre, post))  # type: ignore[assignment]

    for line in text_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _VAR_RE.match(line)
        if m:
            flush()
            cur, pct, pre, post = m.group(1), [], [], []
            continue
        if cur and _ROW_RE.search(line):
            vals = [float(x) for x in line.split("!", 1)[0].split()]
            if len(vals) >= 4:
                pct.append(vals[0])
                pre.append(vals[PRE_COL])
                post.append(vals[POST_COL])
    flush()
    return out


def _levb_id(hid: str) -> str:
    if not _LEVA_ID_RE.match(hid):
        raise ValueError(f"sl4dump header {hid!r} is not a numeric id 0000-0999; "
                         f"cannot derive its LEVB header 'B'+id[1:]")
    return "B" + hid[1:]


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
    for var, (pct_vec, pre_vec, post_vec) in levels.items():
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
        for name, long_name, vec in (
            (hid, tmpl.long_name, post_vec),
            (_levb_id(hid), (LEVB_PREFIX + tmpl.long_name)[:LONG_NAME_WIDTH], pre_vec),
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
    write_har(str(out_har), headers)
    print(f"  levels vars={kept} (LEVB+LEVA)  (xcheck-mismatch={mismatched})  -> {out_har}")
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
    kept = build_levels_har(parse_levels(text), args.sl4dump, out)
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
