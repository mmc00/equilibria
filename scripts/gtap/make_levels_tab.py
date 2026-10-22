"""Write gtapv7_levels.tab: gtapv7.tab with ``orig_level`` on six more variables.

GEMPACK stores pre-/post-simulation LEVELS in the .sl4 (LEVB/LEVA, read by
``export_sl4_levels.py``) only for variables declared with ``orig_level``. In
gtapv7.tab six variables the paper compares are plain linear variables without it,
so they come out as NOLV in ``sl4levels_*.har``:

    y(r)       orig_level = INCOME(r)        (tab: INCOME = PRIVEXP + GOVEXP + SAVE)
    qinv(r)    orig_level = REGINV(r)        (tab: REGINV = sum_c VIP; pinv = 1)
    rore(r)    orig_level = RORELEV(r)       (new coefficient, below)
    pinv(r), pva(a,r), pe(e,r): orig_level = 1.0  (price indices)

RORELEV is the level form of E_rore (``rore = rorc - RORFLEX*[ke - kb]``):
``RORC * (KB/KE)^RORFLEX`` with ``RORC = [GROSSCAP - VDEP]/VKB`` and
``KE = VKB - VDEP + REGINV``. It is inserted right after RORFLEX is read, where
every coefficient it uses already exists.

``orig_level`` only tells GEMPACK the starting level to report; no equation changes,
so the solution (the %-changes) must not change. ``export_sl4_levels.py`` checks
exactly that when it is given the sl4dump of the unpatched run.

The expected LEVB values were fixed before this script, by hand from basedata.har:
``tests/compare/oracles/levb_six_vars.py`` (matches GAMS's base to 1e-11).

Also writes gtapv7_levels.sti: the same condensation, but GEMSIM output (``pgs``)
instead of a Fortran program (``wfp``), so no Fortran compiler is needed.

``--uncondensed`` writes gtapuv7_levels.tab/.sti instead: the STI without its
condensation block (``c`` ... ``e``, omissions included). That is GTAPUV7, the
model RunGTAP uses for Burfisher's experiments: their ``.EXP`` fix and shock
variables (atall, tfd, tpdall...) that the condensation omits.

Usage (any OS; compiling with tablo is Windows-only, see scripts/windows):
    uv run python scripts/gtap/make_levels_tab.py C:/runGTAP375/gtapv7.tab ^
        --sti C:/runGTAP375/gtapv7.sti --out-dir out/ [--uncondensed]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NAME = "gtapv7_levels"
NAME_UNCONDENSED = "gtapuv7_levels"
ENCODING = "latin-1"  # TAB files are plain ASCII; latin-1 round-trips any byte

# variable -> orig_level (a coefficient of the TAB or a constant)
LEVELS: dict[str, str] = {
    "y": "INCOME",
    "qinv": "REGINV",
    "rore": "RORELEV",
    "pinv": "1.0",
    "pva": "1.0",
    "pe": "1.0",
}

RORELEV_BLOCK = """\
Coefficient (all,r,REG)
    RORELEV(r) # level of rore, for orig_level (equilibria make_levels_tab) #;
Formula (all,r,REG)
    RORELEV(r) = {[GROSSCAP(r) - VDEP(r)] / VKB(r)}
        * [VKB(r) / {VKB(r) - VDEP(r) + REGINV(r)}]^RORFLEX(r);
"""

_RORFLEX_READ = re.compile(
    r'^[ \t]*RORFLEX[ \t]+from[ \t]+file[ \t]+GTAPPARM[ \t]+header[ \t]+"RFLX"[ \t]*;[ \t]*\n',
    re.IGNORECASE | re.MULTILINE,
)


class TabPatchError(ValueError):
    """The TAB/STI does not look like the gtapv7 this patch was written for."""


def _declaration(var: str) -> re.Pattern[str]:
    # "Variable <qualifiers>\n    <var>(" -- qualifiers are (...) groups, e.g.
    # (all,r,REG) or (orig_level=VOS). Group "quals" is where orig_level goes.
    return re.compile(
        r"^(?P<kw>[ \t]*Variable[ \t]*)(?P<quals>(?:\([^)\n]*\)[ \t]*)*)[ \t]*\n"
        rf"[ \t]*{re.escape(var)}[ \t]*\(",
        re.IGNORECASE | re.MULTILINE,
    )


def patch_tab(tab: str) -> str:
    """Return ``tab`` with orig_level on the six variables and RORELEV defined."""
    out = tab
    for var, level in LEVELS.items():
        found = list(_declaration(var).finditer(out))
        if len(found) != 1:
            raise TabPatchError(
                f"expected exactly one declaration 'Variable ...\\n    {var}(' "
                f"in the TAB, found {len(found)}"
            )
        m = found[0]
        if "orig_level" in m.group("quals").lower():
            raise TabPatchError(
                f"variable {var} already has orig_level: {m.group(0)!r}"
            )
        at = m.start("quals")
        out = f"{out[:at]}(orig_level={level}){out[at:]}"

    reads = list(_RORFLEX_READ.finditer(out))
    if len(reads) != 1:
        raise TabPatchError(
            f"expected exactly one 'RORFLEX from file GTAPPARM header \"RFLX\";', "
            f"found {len(reads)}"
        )
    at = reads[0].end()
    return out[:at] + RORELEV_BLOCK + out[at:]


def _replace_line(lines: list[str], marker: str, new: str) -> None:
    hits = [i for i, ln in enumerate(lines) if marker in ln]
    if len(hits) != 1:
        raise TabPatchError(f"expected one STI line with '{marker}', found {len(hits)}")
    lines[hits[0]] = new


def patch_sti(sti: str, name: str = NAME, condense: bool = True) -> str:
    """The same stored input, pointed at ``name`` and generating GEMSIM output.

    ``condense=False`` drops the condensation block, from ``c`` to ``e``."""
    lines = sti.splitlines()
    if not condense:
        lines = _drop_condensation(lines)
    _replace_line(
        lines, "name of TABLO input file", f"{name} ! name of TABLO input file"
    )
    _replace_line(
        lines, "name of information file", f"{name} ! name of information file"
    )
    wfp = [i for i, ln in enumerate(lines) if ln.split("!")[0].strip().lower() == "wfp"]
    if len(wfp) != 1:
        raise TabPatchError(f"expected one 'wfp' line in the STI, found {len(wfp)}")
    lines[wfp[0]] = "pgs ! Prepare output for GEMSIM"
    _replace_line(
        lines,
        "name of Fortran program file",
        f"{name} ! name of GEMSIM Auxiliary files",
    )
    return "\n".join(lines) + ("\n" if sti.endswith("\n") else "")


def _drop_condensation(lines: list[str]) -> list[str]:
    code = [ln.split("!")[0].strip().lower() for ln in lines]
    starts = [i for i, c in enumerate(code) if c == "c"]
    ends = [i for i, c in enumerate(code) if c == "e"]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise TabPatchError(
            "expected one condensation block ('c' ... 'e') in the STI, found "
            f"{len(starts)} 'c' and {len(ends)} 'e'"
        )
    return lines[: starts[0]] + lines[ends[0] + 1 :]


def _read(path: Path) -> tuple[str, str]:
    """Text with \n line ends, plus the line end the file used (TABs ship CRLF)."""
    raw = path.read_bytes().decode(ENCODING)
    eol = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), eol


def _write(path: Path, text: str, eol: str) -> None:
    path.write_bytes(text.replace("\n", eol).encode(ENCODING))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("tab", type=Path, help="gtapv7.tab to patch (left untouched)")
    ap.add_argument("--sti", type=Path, required=True, help="its gtapv7.sti")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--uncondensed", action="store_true", help=f"write {NAME_UNCONDENSED} (GTAPUV7)"
    )
    args = ap.parse_args(argv)
    name = NAME_UNCONDENSED if args.uncondensed else NAME

    tab, tab_eol = _read(args.tab)
    sti, sti_eol = _read(args.sti)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write(args.out_dir / f"{name}.tab", patch_tab(tab), tab_eol)
    _write(
        args.out_dir / f"{name}.sti",
        patch_sti(sti, name, condense=not args.uncondensed),
        sti_eol,
    )
    print(
        f"wrote {args.out_dir / name}.tab and .sti (orig_level on {', '.join(LEVELS)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
