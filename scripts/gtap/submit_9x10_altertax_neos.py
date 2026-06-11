"""Submit the 9x10 altertax self-contained bundle to NEOS.

Reads:
  output/9x10_altertax_neos_bundle/comp_9x10_altertax_neos.gms
  output/9x10_altertax_neos_bundle/in.gdx

Submits to NEOS (category cp, solver PATH, inputMethod GAMS) with the GDX
encoded as base64 inside the <gdx> tag. Polls until done; saves the
returned out.gdx + log to output/9x10_altertax_neos_bundle/.
"""
from __future__ import annotations
import argparse
import base64
import time
import xmlrpc.client
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEOS_URL = "https://neos-server.org:3333"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", type=Path,
                    default=ROOT / "output/9x10_altertax_neos_bundle")
    ap.add_argument("--email", default="dracomarmol@gmail.com")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    gms_path = args.bundle_dir / "comp_9x10_altertax_neos.gms"
    gdx_path = args.bundle_dir / "in.gdx"
    if not gms_path.exists() or not gdx_path.exists():
        raise SystemExit(f"Run build_9x10_altertax_neos_bundle.py first; "
                          f"missing {gms_path} or {gdx_path}")

    model = gms_path.read_text()
    gdx_b64 = base64.b64encode(gdx_path.read_bytes()).decode("ascii")

    print(f"Model: {gms_path.name}  ({len(model):,} chars)")
    print(f"GDX:   {gdx_path.name}  ({gdx_path.stat().st_size:,} bytes "
          f"→ {len(gdx_b64):,} base64 chars)")

    job_xml = f"""<document>
<category>cp</category>
<solver>PATH</solver>
<inputMethod>GAMS</inputMethod>
<email>{args.email}</email>
<model><![CDATA[{model}]]></model>
<options><![CDATA[]]></options>
<parameters><![CDATA[]]></parameters>
<gdx>{gdx_b64}</gdx>
<wantgdx><![CDATA[yes]]></wantgdx>
<wantlst><![CDATA[yes]]></wantlst>
<wantlog><![CDATA[yes]]></wantlog>
<comments><![CDATA[9x10 ALTERTAX baseline (CD VA, Armington 0.95, all factors mobile)]]></comments>
</document>"""

    print(f"\nConnecting to NEOS ({NEOS_URL})...")
    neos = xmlrpc.client.ServerProxy(NEOS_URL)
    print(neos.ping())

    print("Submitting job...")
    job_number, password = neos.submitJob(job_xml)
    print(f"  Job #: {job_number}")
    print(f"  Password: {password}")
    print(f"  Status URL: https://neos-server.org/neos/cgi-bin/nph-neos-solver.cgi"
          f"?admin=results&jobnumber={job_number}&pass={password}")

    info_path = args.bundle_dir / "neos_altertax_job_info.txt"
    info_path.write_text(f"job_number={job_number}\npassword={password}\n")
    print(f"  Saved job info to {info_path}")

    if args.no_wait:
        return

    print("\nPolling status (every 15s)...")
    last_status = ""
    t0 = time.time()
    while True:
        time.sleep(15)
        status = neos.getJobStatus(job_number, password)
        elapsed = int(time.time() - t0)
        if status != last_status:
            print(f"  [{elapsed:4d}s] {status}")
            last_status = status
        if status == "Done":
            break
        if status in ("Failed", "Killed", "Bad job"):
            raise SystemExit(f"NEOS job {status}")

    print("\nFetching results...")
    log = neos.getFinalResults(job_number, password)
    log_text = log.data.decode("utf-8", errors="replace") if hasattr(log, "data") else str(log)
    log_path = args.bundle_dir / "neos_altertax_log.txt"
    log_path.write_text(log_text)
    print(f"  Log → {log_path} ({len(log_text):,} chars)")

    gdx_saved = False
    for candidate in ("solver-output.zip", "COMP_ALTERTAX.gdx", "out.gdx", "results.gdx"):
        try:
            blob = neos.getOutputFile(job_number, password, candidate)
            if blob and (hasattr(blob, "data") and len(blob.data) > 200 or
                         isinstance(blob, (bytes, bytearray)) and len(blob) > 200):
                payload = blob.data if hasattr(blob, "data") else blob
                out = args.bundle_dir / f"neos_altertax_{candidate}"
                out.write_bytes(payload)
                print(f"  Saved → {out} ({len(payload):,} bytes)")
                gdx_saved = True
        except Exception as e:
            print(f"  (skip {candidate}: {e})")

    if not gdx_saved:
        print("  No output file retrieved; check log.")


if __name__ == "__main__":
    main()
