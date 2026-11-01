"""Solve one self-contained GAMS MCP bundle on NEOS and keep what it returns.

The NEOS XML-RPC API (category cp, solver PATH, input GAMS): submit, poll,
then fetch the log and solver-output.zip. ``out`` gets the files a local run
leaves (``out.gdx``, ``model.lst``) plus ``neos.log`` and ``neos_job.json``,
the configuration write_source.py --remote neos records. The job password and
the e-mail NEOS requires are never written.

    NEOS_EMAIL=you@example.org uv run python data/gtap7/gams/tools/neos_job.py \
        --gms model.gms --out folder [--options path.opt]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import sys
import time
import xmlrpc.client
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NEOS_URL = "https://neos-server.org:3333"
CATEGORY, SOLVER, INPUT_METHOD = "cp", "PATH", "GAMS"
POLL_S = 15.0
TIMEOUT_S = 8 * 3600.0
_FAILED = ("Failed", "Killed", "Bad job", "Error")
# a dropped connection while waiting must not lose a job of hours: retry the
# read-only calls (status, results); never the submission, which would send it twice
NETWORK_RETRIES = 40
_NETWORK = (OSError, xmlrpc.client.ProtocolError)
_OK_STATUS = {"1", "2"}
_VERSION = re.compile(r"\bGAMS (\d+\.\d+\.\d+)\b")
_MODEL_STATUS = re.compile(r"MODEL STATUS\s+(\d+)")


class NeosError(RuntimeError):
    """NEOS did not return a solved model."""


@dataclass(frozen=True)
class NeosRun:
    job: int
    gams_version: str
    statuses: tuple[str, ...]


def default_proxy() -> Any:
    return xmlrpc.client.ServerProxy(NEOS_URL, allow_none=True)


def _bytes(blob: Any) -> bytes:
    return blob.data if hasattr(blob, "data") else bytes(blob)


def _retry(call: Any, *args: Any, wait: float) -> Any:
    for attempt in range(NETWORK_RETRIES + 1):
        try:
            return call(*args)
        except _NETWORK as err:
            if attempt == NETWORK_RETRIES:
                raise NeosError(f"NEOS unreachable after {attempt + 1} tries: {err}") from err
            print(f"  network error ({err}); retrying", flush=True)
            time.sleep(max(wait, 1.0) if wait else 0)
    raise AssertionError("unreachable")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def job_xml(model: str, options: str, email: str) -> str:
    return f"""<document>
<category>{CATEGORY}</category>
<solver>{SOLVER}</solver>
<inputMethod>{INPUT_METHOD}</inputMethod>
<email>{email}</email>
<model><![CDATA[{model}]]></model>
<options><![CDATA[{options}]]></options>
<parameters><![CDATA[]]></parameters>
<wantgdx><![CDATA[yes]]></wantgdx>
<wantlst><![CDATA[yes]]></wantlst>
<wantlog><![CDATA[yes]]></wantlog>
<comments><![CDATA[equilibria GAMS reference (data/gtap7/gams/tools/neos_job.py)]]></comments>
</document>"""


def run(
    gms: Path,
    out: Path,
    *,
    options: str = "",
    email: str | None = None,
    proxy: Any = None,
    poll_s: float | None = None,
    timeout_s: float = TIMEOUT_S,
) -> NeosRun:
    """Submit ``gms`` to NEOS, wait, and write its results into ``out``.

    Raises NeosError when the job fails, is killed, times out, returns no
    GDX, or returns a model status outside {1, 2}; ``out`` is then untouched.
    """
    email = email or os.environ.get("NEOS_EMAIL", "")
    if not email:
        raise NeosError("NEOS requires an e-mail: pass --email or set NEOS_EMAIL")
    neos = proxy if proxy is not None else default_proxy()
    poll = POLL_S if poll_s is None else poll_s
    model = gms.read_text()
    neos.ping()
    submitted = _now()
    job, password = neos.submitJob(job_xml(model, options, email))
    if not job:
        raise NeosError(f"NEOS refused the job: {password}")
    print(f"NEOS job {job} submitted", flush=True)

    seen: list[str] = []
    t0 = time.monotonic()
    while True:
        status = _retry(neos.getJobStatus, job, password, wait=poll)
        if not seen or seen[-1] != status:
            seen.append(status)
            print(f"  [{time.monotonic() - t0:6.0f}s] {status}", flush=True)
        if status == "Done":
            break
        if status in _FAILED:
            raise NeosError(f"NEOS job {job}: {status}")
        if time.monotonic() - t0 >= timeout_s:
            raise NeosError(f"NEOS job {job} timed out after {timeout_s:.0f}s ({status})")
        time.sleep(poll)
    finished = _now()

    log = _bytes(_retry(neos.getFinalResults, job, password, wait=poll)).decode(
        "utf-8", errors="replace"
    )
    payload = _bytes(_retry(neos.getOutputFile, job, password, "solver-output.zip", wait=poll))
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as err:
        raise NeosError(f"NEOS job {job}: solver-output.zip is not a zip") from err
    names = archive.namelist()
    gdx_names = [n for n in names if n.endswith(".gdx")]
    gdx_name = "out.gdx" if "out.gdx" in names else (gdx_names[0] if len(gdx_names) == 1 else None)
    if gdx_name is None:
        raise NeosError(f"NEOS job {job}: no single out.gdx in solver-output.zip ({names})")
    lst_names = [n for n in names if n.endswith(".lst")]
    lst = archive.read(lst_names[0]).decode("utf-8", errors="replace") if lst_names else log

    statuses = _MODEL_STATUS.findall(lst)
    if not statuses or set(statuses) - _OK_STATUS:
        raise NeosError(f"NEOS job {job}: model statuses {statuses} (not all 1/2)")
    version = _VERSION.search(lst) or _VERSION.search(log)
    if version is None:
        raise NeosError(f"NEOS job {job}: no GAMS version in the listing")

    out.mkdir(parents=True, exist_ok=True)
    (out / "out.gdx").write_bytes(archive.read(gdx_name))
    (out / "model.lst").write_text(lst)
    (out / "neos.log").write_text(log)
    record = {
        "job": job,
        "submitted_utc": submitted,
        "finished_utc": finished,
        "category": CATEGORY,
        "solver": SOLVER,
        "input_method": INPUT_METHOD,
        "options": options,
        "gams_version": version.group(1),
        "sent_sha256": hashlib.sha256(model.encode()).hexdigest(),
    }
    (out / "neos_job.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"NEOS job {job} done: model statuses {statuses}, GAMS {version.group(1)}")
    return NeosRun(job=job, gams_version=version.group(1), statuses=tuple(statuses))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--options", type=Path, help="PATH options file (path.opt)")
    ap.add_argument("--email", help="default: $NEOS_EMAIL; never written to disk")
    args = ap.parse_args(argv)
    options = args.options.read_text() if args.options else ""
    try:
        run(args.gms, args.out, options=options, email=args.email)
    except NeosError as err:
        print(err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
