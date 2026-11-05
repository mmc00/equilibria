"""A fake of the GEMPACK programs the phase 3a-2 job modules call.

Replaces ``gemsim_job.run`` (the one place every job module runs a program):
it looks at the command and writes what the real program would leave, with
the banner of the release it is given. Tests set ``fail_start`` to simulate a
program that never starts, or ``console_extra`` / ``mismatch`` to steer a run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

BANNER = " <GEMPACK Release {}   January 2020>\n"


@dataclass
class FakeGempack:
    releases: dict[str, str] = field(default_factory=lambda: {
        "gtapv7.exe": "12.1.003", "gtapuv7.exe": "12.1.003",
        "gemsim.exe": "11.3.004", "tablo.exe": "11.3.004", "sltoht.exe": "11.3.004",
    })  # fmt: skip
    fail_start: set[str] = field(
        default_factory=set
    )  # program names that never start once
    abort: bool = False  # gtapv7 aborts with the 20x41 signature
    mismatch: int = 0  # xcheck-mismatch the export reports
    calls: list[tuple[str, Path]] = field(default_factory=list)

    def __call__(self, cmd: list[str], cwd: Path, console: Path) -> int:
        prog = Path(cmd[0]).name.lower()
        if prog.startswith("python"):
            prog = Path(cmd[1]).name
        self.calls.append((prog, cwd))
        if prog in self.fail_start:
            self.fail_start.discard(prog)
            console.write_text("")
            return -1073741502
        if prog == "export_sl4_levels.py":
            return self._export(cmd, console)
        banner = BANNER.format(self.releases.get(prog, "11.3.004"))
        if prog == "tablo.exe":
            for ext in (".gss", ".gst"):
                (cwd / f"simple{ext}").write_bytes(b"x")
            console.write_text(banner)
            return 0
        if prog == "sltoht.exe":
            sti = (cwd / cmd[cmd.index("-sti") + 1]).read_text().split("\n")
            (cwd / sti[5]).write_bytes(b"dump")
            console.write_text(banner)
            return 0
        cmf = cwd / cmd[cmd.index("-cmf") + 1]
        stem = cmf.stem
        text = cmf.read_text(encoding="latin-1")
        if self.abort:
            (cwd / f"{stem}-assert-arith-fail.har").write_bytes(b"dump")
            # the lines of runs/gempack_20x41_validation/solve_console.txt (2725-2730)
            console.write_text(banner + '     in equation "E_u".\n\n'
                               "    Attempt to take LOGE of zero or of a negative number.\n"
                               "    value of LOGE argument is -0.2668599\n"
                               " (E-loge of zero or of a negative number)\n")  # fmt: skip
            return 1
        m = re.search(r'(?im)^\s*updated file \w+\s*=\s*"?([^";]+?)"?\s*;', text)
        upd = m.group(1).replace("<cmf>", stem) if m else f"{stem}.upd"
        sol = re.search(r'(?im)^\s*solution file\s*=\s*"?([^";]+?)"?\s*;', text)
        sl4 = Path(sol.group(1) + ".sl4") if sol else Path(f"{stem}.sl4")
        for f in (upd, str(sl4), "decomp.har", "summary.har", "volume.har"):
            target = Path(f) if Path(f).is_absolute() else cwd / f
            target.write_bytes(b"x")
        console.write_text(banner + "  completed without error\n")
        return 0

    def _export(self, cmd: list[str], console: Path) -> int:
        sl4 = Path(cmd[2])
        out = Path(cmd[cmd.index("--out") + 1])
        (sl4.parent / f"{sl4.stem}_levels.txt").write_text("levels")
        out.write_bytes(b"levels")
        console.write_text(f"levels vars=50 (xcheck-mismatch={self.mismatch} [])\n")
        return 0
