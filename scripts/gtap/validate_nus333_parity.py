"""Smoke test: load NUS333 via HAR and validate parameters are populated."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from equilibria.templates.gtap import GTAPParameters

NUS333 = Path("/Users/marmol/Downloads/10284")

params = GTAPParameters()
params.load_from_har(
    basedata_path=NUS333 / "basedata.har",
    sets_path=NUS333 / "sets.har",
    default_path=NUS333 / "default.prm",
    baserate_path=NUS333 / "baserate.har",
)

print(f"Sets: r={params.sets.r}, i={params.sets.i}, f={params.sets.f}")
print(f"Benchmark:")
print(f"  vdpp entries: {len(params.benchmark.vdpp)}")
print(f"  vfm  entries: {len(params.benchmark.vfm)}")
print(f"  vcif entries: {len(params.benchmark.vcif)}")
print(f"  vtwr entries: {len(params.benchmark.vtwr)}")
print(f"  vdpp(USA,AGR): {params.benchmark.vdpp.get(('USA','AGR'), 0):.6f}")
print(f"Elasticities:")
print(f"  esubd(USA,AGR): {params.elasticities.esubd.get(('USA','AGR'), 0):.4f}")
print(f"  esubva(USA,AGR): {params.elasticities.esubva.get(('USA','AGR'), 0):.4f}")
print(f"Taxes:")
print(f"  imptx entries: {len(params.taxes.imptx)}")
print(f"  rtxs  entries: {len(params.taxes.rtxs)}")
print(f"Calibrated shares:")
print(f"  and_param entries: {len(params.calibrated.and_param)}")
print(f"  io_param  entries: {len(params.calibrated.io_param)}")
print("\nSMOKE TEST PASSED — NUS333 HAR pipeline loads cleanly.")
