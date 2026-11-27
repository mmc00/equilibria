"""Solve the 2015 GHA MCP (mac/neos/gha2015_path.gms) with PATH on NEOS.

    export NEOS_EMAIL=you@example.org        # NEOS requires a valid address
    python mac/neos_submit.py                 # show the job XML, submit nothing
    python mac/neos_submit.py --submit        # submit, print job and password, poll, save
    python mac/neos_submit.py --job N --password P   # poll/fetch an earlier job

Uses the NEOS XML-RPC API (https://neos-server.org:3333), category cp,
solver PATH, input method GAMS; the XML follows getSolverTemplate("cp",
"PATH", "GAMS"). wantgdx/wantlst/wantlog are set, so NEOS returns the
listing (with the display of lev and stat), the log and the GDX of all
symbols. Results go to mac/neos/<job>/:
    results.txt          getFinalResults (the listing/log text NEOS returns)
    solver-output.zip    unpacked next to it: solve.lst, solve.log, out.gdx
    job.txt              job number, password, status
Run with the GAMS Python or any Python 3 (standard library only).
"""
import argparse
import base64
import io
import os
import sys
import time
import xmlrpc.client
import zipfile

URL = "https://neos-server.org:3333"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "neos", "gha2015_path.gms")
# NEOS hands out by name only: results.txt, ampl.sol, solver-output.zip, job.in, job.out, job.results.
# For GAMS jobs solver-output.zip holds solve.lst, solve.log and out.gdx (wantgdx).
OUTPUT_FILES = ["solver-output.zip"]


def job_xml(model_text, email):
    return f"""<document>
<category>cp</category>
<solver>PATH</solver>
<inputMethod>GAMS</inputMethod>
<email>{email}</email>
<model><![CDATA[{model_text}]]></model>
<options><![CDATA[]]></options>
<parameters><![CDATA[]]></parameters>
<wantgdx><![CDATA[yes]]></wantgdx>
<wantlst><![CDATA[yes]]></wantlst>
<wantlog><![CDATA[yes]]></wantlog>
<comments><![CDATA[MANAGE-WB Ghana 2015, CONVERT scalar MCP, 9452 x 9452]]></comments>
</document>
"""


def as_bytes(obj):
    if isinstance(obj, xmlrpc.client.Binary):
        return obj.data
    if isinstance(obj, str):
        try:
            return base64.b64decode(obj, validate=True)
        except Exception:
            return obj.encode()
    return bytes(obj)


def fetch(neos, job, password, outdir, poll):
    os.makedirs(outdir, exist_ok=True)
    status = neos.getJobStatus(job, password)
    while status in ("Waiting", "Running") and poll:
        print(f"job {job}: {status}", flush=True)
        time.sleep(poll)
        status = neos.getJobStatus(job, password)
    print(f"job {job}: {status}")
    with open(os.path.join(outdir, "job.txt"), "w") as fh:
        fh.write(f"job {job}\npassword {password}\nstatus {status}\n")
    if status != "Done":
        return 1
    text = as_bytes(neos.getFinalResults(job, password))
    with open(os.path.join(outdir, "results.txt"), "wb") as fh:
        fh.write(text)
    print("saved", os.path.join(outdir, "results.txt"))
    for name in OUTPUT_FILES:
        try:
            data = as_bytes(neos.getOutputFile(job, password, name))
        except Exception as e:  # file not offered for this job
            print(f"  {name}: not available ({e.__class__.__name__})")
            continue
        if not data or data.startswith((b"Output file", b"Illegal file requested")) or b"does not exist" in data[:200]:
            print(f"  {name}: not available")
            continue
        path = os.path.join(outdir, name)
        with open(path, "wb") as fh:
            fh.write(data)
        print(f"  saved {path} ({len(data)} bytes)")
        if name.endswith(".zip"):
            try:
                zipfile.ZipFile(io.BytesIO(data)).extractall(outdir)
                print(f"  unpacked {name}")
            except zipfile.BadZipFile:
                pass
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submit", action="store_true", help="really submit the job to NEOS")
    ap.add_argument("--job", type=int)
    ap.add_argument("--password")
    ap.add_argument("--poll", type=int, default=30, help="seconds between status checks (0: no polling)")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    neos = xmlrpc.client.ServerProxy(URL)
    if a.job:
        return fetch(neos, a.job, a.password, os.path.join(HERE, "neos", str(a.job)), a.poll)

    email = os.environ.get("NEOS_EMAIL", "").strip()
    if not email:
        print("set NEOS_EMAIL to the address NEOS should use")
        return 1
    model_text = open(a.model).read()
    if "]]>" in model_text:
        print("model contains ']]>' and cannot go into CDATA")
        return 1
    xml = job_xml(model_text, email)
    print(f"model {a.model}: {len(model_text) / 1e6:.2f} MB, job XML {len(xml) / 1e6:.2f} MB")
    if not a.submit:
        print(xml[:600] + "\n...\n" + xml[-500:])
        print("not submitted (use --submit)")
        return 0
    print(neos.ping())
    job, password = neos.submitJob(xml)
    if job == 0:
        print("NEOS refused the job:", password)
        return 1
    print(f"job {job} password {password}")
    print(f"status page: https://neos-server.org/neos/cgi-bin/nph-neos-solver.cgi?admin=results&jobnumber={job}&pass={password}")
    return fetch(neos, job, password, os.path.join(HERE, "neos", str(job)), a.poll)


if __name__ == "__main__":
    sys.exit(main())
