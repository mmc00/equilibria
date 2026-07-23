"""Export POST-SIMULATION LEVELS from a GEMPACK SL4 solution to a plain HAR.

The `sltoht` HAR/VAI output carries only the cumulative %-changes (the
`sl4dump_<ds>_tm10.har` fixtures). But the SL4 solution also holds the
pre-/post-simulation LEVELS (headers LEVB/LEVA), which sltoht exposes ONLY in
its TEXT modes under option SHL: each variable prints 4 solution columns
    [ %-change , pre-sim level , post-sim level , change ].
This script drives `sltoht -SHL -SIC` (solutions-in-columns text), parses the
POST-SIM LEVEL column, and writes it back as a HAR
`sl4levels_<ds>_tm10.har` whose per-variable headers carry the ABSOLUTE
post-shock levels — the clean quantity-vs-quantity reference the values-only
updated.har cannot give.

Component order in the text output equals the variable's HAR flat order, so the
shapes / set-elements / long-name / numeric id are taken from the matching
`sl4dump_<ds>_tm10.har`; the parse is CROSS-CHECKED per variable (the text's
%-change column must equal the sl4dump values) before the level array is kept.

Usage (from repo root, Windows):
    uv run python scripts/gtap/export_sl4_levels.py --sltoht C:/GP/sltoht.exe
    uv run python scripts/gtap/export_sl4_levels.py --datasets gtap7_3x3
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

FIXTURES = ROOT / "tests/fixtures/gtap7_gempack"
DEFAULT_SLTOHT = Path(r"C:\GP\sltoht.exe")

# dataset -> RunGTAP run folder holding tm10.sl4
RUN_DIRS = {
    "nus333": ROOT / "runs/nus333_compare/rungtap",
    "9x10": ROOT / "runs/9x10_compare/rungtap",
    "gtap7_3x3": ROOT / "runs/gempack_matrix/gtap7_3x3",
    "gtap7_3x4": ROOT / "runs/gempack_matrix/gtap7_3x4",
    "gtap7_5x5": ROOT / "runs/gempack_matrix/gtap7_5x5",
    "gtap7_10x7": ROOT / "runs/gempack_matrix/gtap7_10x7",
    "gtap7_15x10": ROOT / "runs/gempack_matrix/gtap7_15x10",
}
POST_COL = 2  # 0-based: [%-change, pre-level, POST-level, change]
_VAR_RE = re.compile(r"^\s*! Variable (\S+) #")
_ROW_RE = re.compile(r"!%1=\s*(\d+)\s*$")


def run_sltoht_sic(run_dir: Path, sltoht: Path) -> Path:
    """Drive `sltoht -SHL -SIC` on tm10.sl4 → levels text file."""
    out = run_dir / "sl4_levels.txt"
    sti = run_dir / "sl4_levels.sti"
    # options: SHL (show levels) + SIC (solutions-in-columns text); blank = finish
    # options; tm10 = solution; c = levels+cumulative; out name; e = exit.
    sti.write_text("SHL\nSIC\n\ntm10\nc\n" + out.name + "\ne\n", encoding="ascii")
    console = run_dir / "sl4_levels_console.txt"
    with console.open("w", encoding="utf-8", errors="replace") as fh:
        subprocess.run([str(sltoht), "-sti", sti.name], cwd=run_dir,
                       stdout=fh, stderr=subprocess.STDOUT, check=False)
    return out


def parse_levels(text_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse the SIC text → {var_name: (pct_change_vec, post_level_vec)} in the
    text's component order (= the HAR flat order)."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cur: str | None = None
    pct: list[float] = []
    lev: list[float] = []

    def flush() -> None:
        if cur and pct:
            out[cur] = (np.array(pct, dtype=np.float64), np.array(lev, dtype=np.float64))

    for line in text_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _VAR_RE.match(line)
        if m:
            flush()
            cur, pct, lev = m.group(1), [], []
            continue
        if cur and _ROW_RE.search(line):
            nums = line.split("!", 1)[0].split()
            vals = [float(x) for x in nums]
            if len(vals) >= 4:
                pct.append(vals[0])
                lev.append(vals[POST_COL])
    flush()
    return out


def build_levels_har(dataset: str, levels: dict, sl4dump: Path, out_har: Path) -> int:
    """Write a HAR of post-sim LEVELS, taking shape/sets/id/long_name from the
    matching %-change sl4dump and cross-checking the %-change column per var."""
    src = read_har(str(sl4dump))
    name_to_hid = {
        (ha.long_name or "").split("#", 1)[0].strip(): hid
        for hid, ha in src.items() if (ha.long_name or "").strip()
    }
    headers: dict[str, HeaderArray] = {}
    kept = mismatched = 0
    for var, (pct_vec, lev_vec) in levels.items():
        hid = name_to_hid.get(var)
        if hid is None:
            continue
        tmpl = src[hid]
        if tmpl.array.dtype.kind not in "fiu" or tmpl.array.size != lev_vec.size:
            continue
        # cross-check: the text %-change column must equal the sl4dump values.
        # GEMPACK writes components in FORTRAN (column-major) order — verified
        # against all 252 gtap7_3x3 variables (C-order agrees only for rank-1).
        ref = np.asarray(tmpl.array, dtype=np.float64).ravel(order="F")
        if not np.allclose(ref, pct_vec, atol=1e-3, rtol=1e-3):
            mismatched += 1
            continue
        arr = lev_vec.reshape(tmpl.array.shape, order="F").astype(np.float32)
        headers[hid] = HeaderArray(
            name=hid, coeff_name=tmpl.coeff_name, long_name=tmpl.long_name,
            array=arr, set_names=list(tmpl.set_names),
            set_elements=[list(e) for e in tmpl.set_elements],
        )
        kept += 1
    if not headers:
        print(f"  {dataset:14s} no levels parsed — SKIP")
        return 0
    write_har(str(out_har), headers)
    print(f"  {dataset:14s} levels vars={kept}  (xcheck-mismatch={mismatched})  -> {out_har.name}")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sltoht", type=Path, default=DEFAULT_SLTOHT)
    ap.add_argument("--datasets", nargs="*", default=list(RUN_DIRS))
    ap.add_argument("--from-text", action="store_true",
                    help="reuse an existing sl4_levels.txt instead of running sltoht")
    args = ap.parse_args()

    if not args.from_text and not args.sltoht.exists():
        print(f"ERROR: sltoht not found at {args.sltoht}", file=sys.stderr)
        return 2

    print("exporting POST-SIM LEVELS from SL4 solutions")
    total = 0
    for ds in args.datasets:
        run_dir = RUN_DIRS[ds]
        sl4 = run_dir / "tm10.sl4"
        sl4dump = FIXTURES / f"sl4dump_{ds}_tm10.har"
        if not sl4.exists() or not sl4dump.exists():
            print(f"  {ds:14s} missing tm10.sl4 or sl4dump fixture — SKIP")
            continue
        text = run_dir / "sl4_levels.txt"
        if not args.from_text:
            text = run_sltoht_sic(run_dir, args.sltoht)
        if not text.exists():
            print(f"  {ds:14s} no levels text produced — SKIP")
            continue
        levels = parse_levels(text)
        total += build_levels_har(ds, levels, sl4dump, FIXTURES / f"sl4levels_{ds}_tm10.har")
    print(f"done ({total} variable-headers written across datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
