"""One GEMPACK altertax reference: RunGTAP's AlterTax parameters or those of GAMS altertax.

What RunGTAP's Tools..AlterTax does, without its window (checked on gtap7_10x7, 2026-11-09:
the updated data equals RunGTAP's to 3e-7, single-precision rounding):

- the closure is GTAPV7.CLP plus del_tbalry for every region but the last and
  cgdslack for the last one, so regional trade balances are held;
- ``parameters=rungtap`` is the file altparv7.exe writes (everything Cobb-Douglas,
  all factors mobile, ESBD = ESBM = 1, CDE turned off);
- ``parameters=gams`` is default.prm with the values of comp_altertax.gms: ESBV, ESBT,
  ESBC = 1, ESBD = ESBM = 0.95, ETRE = -1, all factors sluggish. Investment stays
  Leontief: gtapv7.tab has no ESUBI (E_qia: qia = qinv), GAMS has esubi = 1.

The shock is the one of the GAMS altertax model, ``imptx * 1.1`` on every bilateral
flow, which RunGTAP calls ``tms = rate% 10``: the power changes by 100*0.1*t/(1+t) %.
It is not ``tm = uniform 10``, which is what GAMS's pure mode does.

The .prm is spliced record by record, not rewritten with write_har: that changes
formats GEMPACK can reject (run_gempack_matrix._force_rdlt).
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _job_cli import common, releases
from run_gempack_matrix import (
    EXOG_BLOCK,
    WORK_ROOT,
    _redact_work_path,
    collect_job,
    convert_sl4,
    gempack_release,
    regions,
    solve,
)

from equilibria.babel.har.reader import read_har
from equilibria.sources import dataset_dir

TAG = "altertax"
RATE_INCREASE = 0.10
STEPS = "8 16 32"
PRM_NAME = "parameters.prm"
SHOCK_NAME = "tms_rate10.shk"
INPUTS = ("sets.har", "basedata.har", "default.prm")
EXTRA_OUTPUTS = ("summary.har", "volume.har")  # GTAPSUM and GTAPVOL of run.cmf

# Headers of parameters=gams taken from what altparv7 wrote; the rest stay default.prm.
GAMS_FROM_ALTPAR = ("ESBV", "ESBT", "ESBC", "ETRE", "ESBD", "ESBM", "EFLG")
GAMS_VALUES = {"ESBV": 1.0, "ESBT": 1.0, "ESBC": 1.0, "ETRE": -1.0, "ESBD": 0.95, "ESBM": 0.95}
RUNGTAP_VALUES = {
    "ESBV": 1.0, "ESBT": 1.0, "ESBC": 1.0, "ETRE": -1.0, "ESBD": 1.0, "ESBM": 1.0,
    "ESBG": 1.0, "ESBI": 1.0, "ESBS": 1.0, "ETRQ": -1.0, "ESBQ": 1.0, "SUBP": 0.0, "INCP": 1.0,
}  # fmt: skip
ENDW_TYPES = ("mobile", "sluggish", "fixed")


def altertax_closure(regs: list[str]) -> str:
    """EXOG_BLOCK with RunGTAP's altertax specification, as its altertax.cls."""
    line = "          cgdslack\n"
    if line not in EXOG_BLOCK or "qesf ;" not in EXOG_BLOCK:
        raise ValueError("EXOG_BLOCK changed: the altertax closure must be rebuilt")
    held = "".join(f'          del_tbalry("{r}")\n' for r in regs[:-1])
    return EXOG_BLOCK.replace(line, "").replace(
        "qesf ;", f'qesf\n{held}          cgdslack("{regs[-1]}") ;'
    )


def tms_shock(vmsb: np.ndarray, vcif: np.ndarray) -> np.ndarray:
    """% change of the power of the tariff that raises its ad valorem rate 10%."""
    vmsb, vcif = vmsb.astype(float), vcif.astype(float)
    out = np.zeros_like(vcif)
    flows = vcif > 0
    rate = vmsb[flows] / vcif[flows] - 1
    out[flows] = 100 * RATE_INCREASE * rate / (1 + rate)
    return out


def shock_text(shock: np.ndarray) -> str:
    """GEMPACK text array as RunGTAP writes it: one COMM x REG table per destination."""
    ncomm, nreg, ndest = shock.shape
    lines = [f'{ncomm} {nreg} {ndest} Real SpreadSheet Header "N002" LongName "tms rate 10";']
    for d in range(ndest):
        lines += [",".join(f"{v:.11g}" for v in row) for row in shock[:, :, d]]
    return "\n".join(lines) + "\n"


def _blocks(path: Path) -> list[tuple[str, list[bytes]]]:
    """HAR = FORTRAN records; a 4-byte record opens each header."""
    data = path.read_bytes()
    pos, out = 0, []
    while pos < len(data):
        (n,) = struct.unpack_from("<i", data, pos)
        rec = data[pos + 4 : pos + 4 + n]
        pos += 8 + n
        if n == 4:
            out.append((rec.decode("latin-1").strip(), [rec]))
        else:
            out[-1][1].append(rec)
    return out


def _serialize(blocks: list[tuple[str, list[bytes]]]) -> bytes:
    out = bytearray()
    for _, recs in blocks:
        for rec in recs:
            out += struct.pack("<i", len(rec)) + rec + struct.pack("<i", len(rec))
    return bytes(out)


def _ones_to_095(recs: list[bytes], n: int) -> list[bytes]:
    """altparv7 writes ESBD/ESBM as full arrays of 1: the n values end the last record."""
    one, new = struct.pack("<f", 1.0), struct.pack("<f", 0.95)
    if recs[-1][-4 * n :] != one * n:
        raise ValueError("expected an array of ones at the end of the record")
    return [*recs[:-1], recs[-1][: -4 * n] + new * n]


def _all_sluggish(recs: list[bytes], nendw: int) -> list[bytes]:
    """altparv7 writes EFLG sparse, all mobile: 4 blanks, 3 ints, nendw indices, nendw values."""
    data = bytearray(recs[-1])
    first = 16
    if len(data) != first + 8 * nendw:
        raise ValueError(f"EFLG has an unexpected layout: {len(data)} bytes, {nendw} factors")
    column = ENDW_TYPES.index("sluggish")
    for i in range(nendw):
        struct.pack_into("<i", data, first + 4 * i, i + 1 + nendw * column)
    return [*recs[:-1], bytes(data)]


def gams_parameters(default_prm: Path, altpar_prm: Path) -> bytes:
    """default.prm with GAMS altertax's values; every other header keeps its bytes."""
    arrays = read_har(str(default_prm))
    altpar = dict(_blocks(altpar_prm))
    out = []
    for name, recs in _blocks(default_prm):
        if name in ("ESBD", "ESBM"):
            out.append((name, _ones_to_095(altpar[name], arrays[name].array.size)))
        elif name == "EFLG":
            out.append((name, _all_sluggish(altpar[name], arrays[name].array.shape[0])))
        elif name in GAMS_FROM_ALTPAR:
            out.append((name, altpar[name]))
        else:
            out.append((name, recs))
    return _serialize(out)


def check_parameters(prm: Path, default_prm: Path, parameters: str) -> None:
    """Fail unless the file holds exactly what ``parameters`` promises."""
    new, base = read_har(str(prm)), read_har(str(default_prm))
    eflg = new["EFLG"].array
    if parameters == "gams":
        expected, kept = GAMS_VALUES, set(base) - set(GAMS_FROM_ALTPAR)
        slot = ENDW_TYPES.index("sluggish")
    else:
        expected, kept = RUNGTAP_VALUES, {"RFLX", "RDLT"}
        slot = ENDW_TYPES.index("mobile")
    for name, value in expected.items():
        if not np.allclose(new[name].array, value, atol=1e-6):
            raise ValueError(f"{prm.name}: {name} is not {value}")
    if not (np.allclose(eflg[:, slot], 1) and np.allclose(eflg.sum(axis=1), 1)):
        raise ValueError(f"{prm.name}: EFLG is not all {ENDW_TYPES[slot]}")
    for name in sorted(kept):
        if new[name].array.tolist() != base[name].array.tolist():
            raise ValueError(f"{prm.name}: {name} differs from default.prm")


def check_rates(updated: Path, basedata: Path) -> float:
    """Largest |new rate - 1.1 * old rate|; raises when the shock did not land."""
    base, upd = read_har(str(basedata)), read_har(str(updated))
    flows = base["VCIF"].array > 0
    old = base["VMSB"].array[flows].astype(float) / base["VCIF"].array[flows].astype(float) - 1
    new = upd["VMSB"].array[flows].astype(float) / upd["VCIF"].array[flows].astype(float) - 1
    worst = float(np.abs(new - (1 + RATE_INCREASE) * old).max())
    if worst > 1e-4:
        raise ValueError(f"tariff rates moved by {worst} away from 1.1 x base")
    return worst


def cmf_text(regs: list[str], aux: Path) -> str:
    return f"""! altertax closure, tms rate +10% on every flow - GEMPACK / GTAPv7
Auxiliary files = {aux} ;

file GTAPSETS = sets.har ;
file GTAPDATA = basedata.har ;
file GTAPPARM = {PRM_NAME} ;
file GTAPSUM  = summary.har ;
file WELVIEW  = decomp.har ;
file GTAPVOL  = volume.har ;
Updated file GTAPDATA = updated_{TAG}.har ;

Method = Gragg ;
Steps  = {STEPS} ;
Automatic accuracy = no ;
Subintervals = 1 ;

Verbal Description =
altertax closure, tariff rate up 10pct on every flow ;

{altertax_closure(regs)}

shock tms(COMM,REG,REG) = select from file {SHOCK_NAME} ;

CPU = yes ;
NDS = yes ;
log file = yes ;
Extrapolation accuracy file = NO ;
"""


def run_altpar(run_dir: Path, exe: Path, console: Path) -> Path:
    """altparv7.exe writes RunGTAP's altertax.prm from sets.har alone."""
    cmf = run_dir / "altpar.cmf"
    cmf.write_text(
        f"aux files = {exe.with_suffix('')} ;\nfile GTAPSETS = sets.har ;\nfile ALTERPAR = altertax.prm ;\n",
        encoding="ascii",
    )
    with console.open("w", encoding="utf-8", errors="replace") as fh:
        subprocess.run([str(exe), "-cmf", cmf.name], cwd=run_dir, stdout=fh,
                       stderr=subprocess.STDOUT, check=False)  # fmt: skip
    prm = run_dir / "altertax.prm"
    text = console.read_text(encoding="utf-8", errors="replace").lower()
    if "completed without error" not in text or not prm.exists():
        raise RuntimeError(f"altparv7 failed; see the end of\n{console.read_text()[-1500:]}")
    return prm


def _release_ok(console: Path, expected: str | None, name: str) -> bool:
    found = gempack_release(console)
    if expected and found != expected:
        print(f"FAILED {name}: {console.name} reports GEMPACK Release {found}, expected {expected}",
              file=sys.stderr)  # fmt: skip
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--parameters", required=True, choices=["rungtap", "gams"])
    ap.add_argument("--gtapv7", type=Path, required=True)
    ap.add_argument("--altparv7", type=Path, required=True)
    ap.add_argument("--sltoht", type=Path, required=True)
    common(ap)
    args = ap.parse_args(argv)

    name = f"{args.dataset}/{args.parameters}"
    expected = releases(args)
    ds_dir = dataset_dir(args.dataset)
    out: Path = args.out.resolve()
    work_root = args.work or WORK_ROOT
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="altertax_", dir=work_root,
                                     ignore_cleanup_errors=True) as work:  # fmt: skip
        run_dir = Path(work)
        for file in INPUTS:
            shutil.copy2(ds_dir / file, run_dir / file)

        altpar_console = run_dir / "altpar_console.txt"
        altpar = run_altpar(run_dir, args.altparv7, altpar_console)
        if not _release_ok(altpar_console, expected.get("altparv7.exe"), name):
            return 1
        prm = run_dir / PRM_NAME
        if args.parameters == "gams":
            prm.write_bytes(gams_parameters(run_dir / "default.prm", altpar))
        else:
            shutil.copy2(altpar, prm)
        check_parameters(prm, run_dir / "default.prm", args.parameters)

        base = read_har(str(run_dir / "basedata.har"))
        shock = tms_shock(base["VMSB"].array, base["VCIF"].array)
        (run_dir / SHOCK_NAME).write_text(shock_text(shock), encoding="ascii")
        (run_dir / f"{TAG}.cmf").write_text(
            cmf_text(regions(ds_dir), args.gtapv7.with_suffix("")), encoding="ascii"
        )

        ok, res = solve(run_dir, args.gtapv7, TAG)
        console = run_dir / f"solve_console_{TAG}.txt"
        updated = run_dir / f"updated_{TAG}.har"
        if not ok or not updated.exists():
            tail = console.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            print(f"FAILED {name}: {res or 'no residual line'}\n" + "\n".join(tail), file=sys.stderr)
            return 1
        if not _release_ok(console, args.expect_release, name):
            return 1
        worst = check_rates(updated, run_dir / "basedata.har")
        dump = convert_sl4(run_dir, args.sltoht, TAG)
        if dump is None or dump.stat().st_size == 0:
            print(f"FAILED {name}: sltoht produced no sl4dump", file=sys.stderr)
            return 1
        written = collect_job(run_dir, TAG, out)
        # collect_job's list predates these two; a missing one fails the job
        for file in EXTRA_OUTPUTS:
            shutil.copy2(run_dir / file, out / file)
            written.append(file)
        shutil.copy2(prm, out / PRM_NAME)
        shutil.copy2(run_dir / SHOCK_NAME, out / SHOCK_NAME)
        shutil.copy2(altpar_console, out / altpar_console.name)
        _redact_work_path(out / altpar_console.name, run_dir)
    print(f"OK {name} -> {out}: {', '.join(written)}; max rate error {worst:.3g}; {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
