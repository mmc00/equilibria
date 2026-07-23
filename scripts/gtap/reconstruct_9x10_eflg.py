"""Reconstruct the 9x10 reference default.prm's missing EFLG (ENDOWFLAG) header.

RunGTAP/GTAPv7 aborts reading 9x10's default.prm ("no header EFLG") because that
parm file lacked the ENDOWFLAG(ENDW, ENDWT) header every other dataset carries.
EFLG classifies each endowment by mobility type (ENDWT = [mobile, sluggish,
fixed]) — a *standard* GTAP property, independent of the regional aggregation.

9x10's own sets.har already carries that classification in the ENDM/ENDS/ENDF
subsets, so the header is reconstructed deterministically from 9x10's own data
(NOT fabricated). As an independent correctness check we assert the result is
cell-identical to gtap7_5x5's EFLG (same 5 endowments, same order).

The header is APPENDED byte-exact to the original file's records (HAR has no
file-level preamble, so concatenation is valid) rather than rewritten — that
keeps every pre-existing header, including the special RDLT record, unchanged
and GEMPACK-parseable.

Usage (from repo root):
    uv run python scripts/gtap/reconstruct_9x10_eflg.py            # patch the reference prm in place
    uv run python scripts/gtap/reconstruct_9x10_eflg.py --check    # verify only, exit 1 if EFLG missing
    uv run python scripts/gtap/reconstruct_9x10_eflg.py --out PATH  # write to PATH instead (e.g. a run folder copy)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from equilibria.babel.har.reader import read_har  # noqa: E402
from equilibria.babel.har.symbols import HeaderArray  # noqa: E402
from equilibria.babel.har.writer import write_har  # noqa: E402

REF_PRM = ROOT / "src/equilibria/templates/reference/gtap/data/9x10/default.prm"
REF_SETS = ROOT / "src/equilibria/templates/reference/gtap/data/9x10/sets.har"
SIBLING_PRM = ROOT / "datasets/gtap7_5x5/default.prm"  # identical 5-endowment EFLG
ENDWT = ["mobile", "sluggish", "fixed"]


def _labels(ha) -> list[str]:
    return [str(x).strip() for x in ha.array.tolist()]


def build_eflg() -> HeaderArray:
    s = read_har(str(REF_SETS))
    endw = _labels(s["ENDW"])
    mobile, sluggish, fixed = (set(_labels(s[k])) for k in ("ENDM", "ENDS", "ENDF"))
    col = {"mobile": 0, "sluggish": 1, "fixed": 2}
    flag = np.zeros((len(endw), 3), dtype=np.float32)
    for i, e in enumerate(endw):
        if e in mobile:
            flag[i, col["mobile"]] = 1.0
        elif e in sluggish:
            flag[i, col["sluggish"]] = 1.0
        elif e in fixed:
            flag[i, col["fixed"]] = 1.0
        else:
            raise SystemExit(f"endowment {e!r} not classified in ENDM/ENDS/ENDF")

    ref = np.asarray(read_har(str(SIBLING_PRM))["EFLG"].array)
    if not np.array_equal(flag, ref):
        raise SystemExit(f"derived EFLG != gtap7_5x5 standard EFLG\n{flag}\n{ref}")
    return HeaderArray(
        name="EFLG",
        coeff_name="ENDOWFLAG",
        long_name="ENDOWFLAG: endowment mobility type flags",
        array=flag,
        set_names=["ENDW", "ENDWT"],
        set_elements=[endw, ENDWT],
    )


def eflg_header_bytes(eflg: HeaderArray) -> bytes:
    # write a one-header HAR to a temp file; its bytes are appendable as-is.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "eflg.har"
        write_har(str(tmp), {"EFLG": eflg})
        return tmp.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the reference prm already has EFLG; exit 1 if not")
    ap.add_argument("--out", type=Path, default=REF_PRM,
                    help=f"target prm to write (default: the reference file {REF_PRM})")
    args = ap.parse_args()

    if args.check:
        present = "EFLG" in read_har(str(REF_PRM))
        print(f"reference default.prm EFLG present: {present}")
        return 0 if present else 1

    if "EFLG" in read_har(str(REF_PRM)):
        print(f"EFLG already present in {REF_PRM} — nothing to do.")
        return 0

    eflg = build_eflg()
    print("EFLG reconstructed from 9x10 ENDM/ENDS/ENDF, matches gtap7_5x5 standard:")
    print(np.asarray(eflg.array))

    patched = REF_PRM.read_bytes() + eflg_header_bytes(eflg)
    args.out.write_bytes(patched)
    chk = read_har(str(args.out))
    print(f"\nwrote {args.out} ({len(patched)}B); headers={len(chk)}, EFLG={'EFLG' in chk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
