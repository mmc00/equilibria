"""Collect a finished NEOS job, fetched by mac/neos_submit.py, into a result folder.

    python mac/neos_collect.py --job-dir mac/neos/25276438 \
        --sent mac/neos/gha2015_path.gms --out <folder>

Nothing is sent to NEOS. From the job folder it reads solver-output.zip
(solve.lst, solve.log, out.gdx) and job.txt (the job number), and writes to
<folder>, in the layout of the equilibria NEOS results:

    model.gms      the file that was sent (--sent; its sha256 is recorded)
    model.lst      solve.lst
    neos.log       solve.log
    out.gdx        out.gdx
    neos_job.json  job, times, category/solver/input method, options, the GAMS
                   version NEOS ran and sent_sha256; never the password or an
                   e-mail address

The times come from solve.log ("Job ... Start/Stop", the NEOS server's clock,
--tz gives its UTC offset). The submission time is not known to neos_submit.py
and is written as null. The job must have solved: the display of stat in
solve.lst must show solvestat 1 and modelstat 1 or 2, otherwise nothing is
written.
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stamp(line: str, tz: str) -> str:
    m = re.search(r"(\d\d)/(\d\d)/(\d\d) (\d\d):(\d\d):(\d\d)", line)
    mo, d, y, hh, mm, ss = (int(x) for x in m.groups())
    sign = -1 if tz.startswith("-") else 1
    oh, om = (int(x) for x in tz.lstrip("+-").split(":"))
    local = dt.datetime(2000 + y, mo, d, hh, mm, ss,
                        tzinfo=dt.timezone(sign * dt.timedelta(hours=oh, minutes=om)))
    return local.astimezone(dt.timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job-dir", type=Path, required=True)
    ap.add_argument("--sent", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--category", default="cp")
    ap.add_argument("--solver", default="PATH")
    ap.add_argument("--input-method", default="GAMS")
    ap.add_argument("--tz", default="-06:00", help="UTC offset of the NEOS clock in solve.log")
    a = ap.parse_args()

    job = int(re.search(r"^job (\d+)", (a.job_dir / "job.txt").read_text(), re.M).group(1))
    z = zipfile.ZipFile(a.job_dir / "solver-output.zip")
    lst, log, gdx = (z.read(n) for n in ("solve.lst", "solve.log", "out.gdx"))
    text = lst.decode("utf-8", "replace")
    stat = dict(re.findall(r"^(solvestat|modelstat) +([0-9.]+)", text, re.M))
    if float(stat.get("solvestat", 0)) != 1 or float(stat.get("modelstat", 0)) not in (1, 2):
        print(f"job {job} did not solve ({stat}); nothing written")
        return 1
    version = re.search(r"^GAMS (\d+\.\d+\.\d+)", text, re.M).group(1)
    logtext = log.decode("utf-8", "replace").splitlines()
    start = next(l for l in logtext if re.match(r"--- Job \S+ Start", l))
    stop = next(l for l in logtext if re.match(r"--- Job \S+ Stop", l))
    sent = a.sent.read_bytes()

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "model.gms").write_bytes(sent)
    (a.out / "model.lst").write_bytes(lst)
    (a.out / "neos.log").write_bytes(log)
    (a.out / "out.gdx").write_bytes(gdx)
    record = {
        "job": job,
        "submitted_utc": None,
        "started_utc": stamp(start, a.tz),
        "finished_utc": stamp(stop, a.tz),
        "category": a.category,
        "solver": a.solver,
        "input_method": a.input_method,
        "options": "",
        "gams_version": version,
        "sent_sha256": sha256(sent),
    }
    (a.out / "neos_job.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
