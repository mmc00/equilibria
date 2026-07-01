"""Build a self-contained, LOCALLY-runnable PURE-gtap tariff-shock bundle.

This is the PURE STD_GTAP model (NOT altertax). It assembles a single .gms
from a dataset's v7_consolidated.gdx that:

  1. Loads the dataset sets/params inline (no external GDX at execution).
  2. Solves the baseline (loop tsim: base -> check), `Solve gtap using mcp`.
  3. Applies a +10% tm_pct import-tariff shock on the 'shock' period:
        imptx.fx = (1 + imptx.l) * 1.10 - 1   (gated by xwFlag, so the
        diagonal r==r / no-flow routes are skipped automatically).
     This matches the Python gate's _apply_imptx_shock(factor=0.10):
        imptx_new = (1 + imptx_old) * (1 + 0.10) - 1.
  4. Solves the shock period and execute_unloads to out.gdx.

It is produced for BOTH ifSUB=0 and ifSUB=1 (two .gms / two out.gdx) and is
meant to be run LOCALLY with GAMS v53 (community license) — gtap7_3x3 fits the
~2500 row/col limit.

Differences from build_gtap7_altertax_neos_bundle.py:
  * starts from comp.gms (PURE) instead of comp_altertax.gms,
  * NO altertax elasticity overrides,
  * shock = tm_pct import tariff (not the numeraire pnum.fx jump),
  * NO homotopy ramp / path.opt (the PURE +10% tariff is a soft basin),
  * runs locally (no NEOS submission),
  * pdp/pmp recalc and pwmg=0 are already in postsim.gms / iterloop.gms.

Usage:
    uv run python scripts/gtap/build_gtap7_pure_local_bundle.py \
        --dataset gtap7_3x3 --ifsub 0
    uv run python scripts/gtap/build_gtap7_pure_local_bundle.py \
        --dataset gtap7_3x3 --ifsub 1
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

# Reuse the proven inlining machinery from the altertax NEOS builder.
import build_gtap7_altertax_neos_bundle as alt

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = alt.UPSTREAM
DATASETS_DIR = ROOT / "datasets"


def _patch_shock_to_tariff(text: str, tariff_increase: float = 0.10) -> str:
    """Replace the numeraire shock block with a +tariff% tm_pct import shock.

    comp.gms shock block:
        if(sameas(tsim,'shock'),
           pnum.fx(tsim) = 1.5 ;
        ) ;

    Replace with the tm_pct power-of-tariff shock, gated by xwFlag so the
    diagonal (r==r) and no-flow routes are skipped (xwFlag=0 there).
    """
    old_block = (
        "   if(sameas(tsim,'shock'),\n"
        "      pnum.fx(tsim) = 1.5 ;\n"
        "   ) ;"
    )
    factor = 1.0 + tariff_increase
    pct = tariff_increase * 100
    new_block = (
        f"* === PURE gtap tm_pct shock: +{pct:.0f}% on the POWER of all bilateral\n"
        f"* === import tariffs.  imptx_new = (1+imptx)*{factor} - 1, gated by xwFlag\n"
        f"* === (so the diagonal r==r / no-flow routes, where xwFlag=0, are skipped).\n"
        f"* === Matches the Python gate's _apply_imptx_shock(factor={tariff_increase}).\n"
        f"   if(sameas(tsim,'shock'),\n"
        f"      imptx.fx(r,i,rp,tsim)$xwFlag(r,i,rp) =\n"
        f"         (1 + imptx.l(r,i,rp,tsim)) * {factor} - 1 ;\n"
        f"   ) ;"
    )
    if old_block not in text:
        raise SystemExit(
            "Could not locate numeraire shock block to replace.\nExpected:\n"
            + old_block
        )
    n = text.count(old_block)
    if n != 1:
        raise SystemExit(f"Expected 1 occurrence of shock block, found {n}.")
    return text.replace(old_block, new_block, 1)


def _force_ifsub(text: str, ifsub: int) -> str:
    """Force the ifSUB compile-time switch to a fixed value.

    comp.gms uses `$setGlobal ifSUB 1`.  Replace with the requested value.
    """
    new, n = re.subn(
        r"^\$setGlobal\s+ifSUB\s+\d+\s*$",
        f"$setGlobal ifSUB       {ifsub}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise SystemExit(f"Could not find `$setGlobal ifSUB` to set to {ifsub}.")
    return new


def _elements_of(text: str, set_name: str) -> set:
    """Extract the element names of an inline `Set <name> / ... /;` block."""
    m = re.search(
        rf'(?im)^\s*[Ss]et\s+{re.escape(set_name)}\b[^\n]*/\s*\n(.*?)\n\s*/\s*;',
        text, re.S,
    )
    if not m:
        return set()
    els = set()
    for ln in m.group(1).splitlines():
        ln = ln.split("*", 1)[0].strip().rstrip(",").strip()
        if not ln or ln.startswith("set."):
            continue
        tok = re.split(r"[\s,.]", ln)[0].strip().strip("'\"")
        if tok:
            els.add(tok)
    return els


def _fix_acts_comm_collision(text: str) -> str:
    """Drop the `set.comm` line from the `set is` composite when acts == comm.

    `set is` (SAM accounts) unions acts ∪ comm ∪ endw ∪ stdlab ∪ reg by listing each
    `set.<name>` marker.  In datasets whose activity and commodity elements share the
    SAME names (e.g. gtap7_15x10: both are Rice/Grains/… with NO a_/c_ prefix, unlike
    3x3/5x5/10x7 which are prefixed), listing set.acts THEN set.comm redeclares those
    elements and GAMS raises "$172 Element is redefined" (verified locally: $onMulti
    does NOT help — it permits set RE-declaration, not duplicate elements in one list).
    When acts == comm the union equals acts alone, so dropping the `set.comm` line from
    the `set is` block is exact and safe.  Only fires on the collision (prefixed
    datasets are untouched).  set.comm stays in the OTHER composites (aa, i) which are
    subsets of the already-complete `is`.
    """
    acts = _elements_of(text, "acts")
    comm = _elements_of(text, "comm")
    if not acts or acts != comm:
        return text  # prefixed (no collision) or sets not found — nothing to fix
    m = re.search(
        r'(set is "SAM accounts for aggregated SAM" /.*?/\s*;)', text, re.S
    )
    if not m:
        return text
    block = m.group(1)
    new_block = re.sub(
        r'(^\s*)set\.comm(\s*$)',
        r'\1*  [dropped: acts==comm, union unchanged] set.comm\2',
        block, count=1, flags=re.MULTILINE,
    )
    if new_block == block:
        return text
    return text.replace(block, new_block, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset name, e.g. gtap7_3x3")
    ap.add_argument("--ifsub", type=int, choices=(0, 1), required=True,
                    help="ifSUB mode: 0 (full) or 1 (reduced).")
    ap.add_argument("--tariff", type=float, default=0.10,
                    help="Uniform tariff power shock fraction (default 0.10).")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    dataset = args.dataset
    ifsub = args.ifsub
    src_gdx = DATASETS_DIR / dataset / "v7_consolidated.gdx"
    if not src_gdx.exists():
        raise SystemExit(f"v7_consolidated.gdx not found: {src_gdx}")

    out_dir = args.out_dir or ROOT / "output" / f"{dataset}_pure_local_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [1/4] Copying {dataset} consolidated GDX as in.gdx ===")
    in_gdx = out_dir / "in.gdx"
    shutil.copy2(src_gdx, in_gdx)
    print(f"  copied -> {in_gdx.name} ({in_gdx.stat().st_size:,} bytes)")

    ds_sets = alt._get_dataset_sets(in_gdx)
    print(f"  dataset sets: reg={ds_sets['reg']}, comm={ds_sets['comm']}")

    print("\n=== [2/4] Inlining GDX -> GAMS Set/Param declarations ===")
    data_block_path = out_dir / "inline_data.gms"
    params_path = out_dir / "inline_params.gms"
    alt.run_inliner(in_gdx, data_block_path, params_path)

    data_block_text = alt._filter_metadata_sets(data_block_path.read_text())
    data_block_text = alt._normalize_set_names(data_block_text)
    data_block_path.write_text(data_block_text)
    params_text = alt._rename_pop_to_pop0(params_path.read_text())
    params_path.write_text(params_text)

    print("\n=== [3/4] Inlining $include directives (PURE comp.gms) ===")
    main_script = alt._read("comp.gms")
    main_script = _force_ifsub(main_script, ifsub)
    inlined = alt._inline_includes(main_script)
    print(f"  inlined size: {len(inlined):,} chars")

    print("\n=== [4/4] Stripping GDX loads, injecting data, applying shock ===")
    inlined = alt._strip_gdx_loads(inlined)
    inlined = alt._strip_getdata_set_block(inlined)
    inlined = alt._patch_dataset_sets(inlined, ds_sets)

    # Inject the inline SETS block at the stripped set-decl anchor.
    data_block = data_block_path.read_text()
    marker = "* [getData set decl stripped]"
    idx = inlined.find(marker)
    if idx == -1:
        raise SystemExit("Could not find getData set-decl marker.")
    inlined = inlined[:idx] + "\n" + data_block + "\n" + inlined[idx:]

    # Fix the acts==comm element collision in the `set is` composite (unprefixed
    # datasets like gtap7_15x10 → "$172 Element is redefined").  See
    # _fix_acts_comm_collision.  No-op for prefixed datasets (3x3/5x5/10x7).
    inlined = _fix_acts_comm_collision(inlined)

    # Split inline_params into Dat-side and Prm-side (same split as altertax).
    params_block = params_path.read_text()
    dat_param_names = {
        "VDFB", "VDFP", "VMFB", "VMFP", "VDPB", "VDPP", "VMPB", "VMPP",
        "VDGB", "VDGP", "VMGB", "VMGP", "VDIB", "VDIP", "VMIB", "VMIP",
        "EVFB", "EVFP", "EVOS", "VXSB", "VFOB", "VCIF", "VMSB",
        "VST", "VTWR", "SAVE", "VDEP", "VKB", "POP", "MAKS", "MAKB", "pop0",
    }
    header = params_block.split("\n")[0]
    dat_lines, prm_lines = [header], [header]
    current_target = None
    for line in params_block.splitlines()[1:]:
        m = re.match(r"^\*\s*([A-Za-z][A-Za-z0-9_]*)\s+(?:data|\(empty\))", line)
        if m:
            current_target = m.group(1)
        elif line.startswith("* skipped "):
            current_target = None
            continue
        elif line.startswith("$on") or line.startswith("$off"):
            dat_lines.append(line)
            prm_lines.append(line)
            continue
        target_dat = current_target in dat_param_names if current_target else False
        (dat_lines if target_dat else prm_lines).append(line)
    dat_block = "\n".join(dat_lines)
    prm_block = "\n".join(prm_lines)

    ftrv_anchor = "ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;"
    f_idx = inlined.find(ftrv_anchor)
    if f_idx < 0:
        raise SystemExit("Could not find ftrv anchor for Dat params block.")
    inlined = inlined[:f_idx] + "\n" + dat_block + "\n\n" + inlined[f_idx:]

    rorflex_anchor = "* [stripped for NEOS inline] $loadDC etrae=etrae rorFlex0=rorFlex\n"
    a_idx = inlined.find(rorflex_anchor)
    if a_idx < 0:
        raise SystemExit("Could not locate rorFlex stripped-load anchor.")
    close_marker = "* [stripped for NEOS inline] $gdxin\n"
    close_idx = inlined.find(close_marker, a_idx)
    if close_idx < 0:
        raise SystemExit("Could not locate Prm.gdx close $gdxin marker.")
    insert_at = close_idx + len(close_marker)
    inlined = inlined[:insert_at] + "\n" + prm_block + "\n" + inlined[insert_at:]

    # Redirect output files to the bundle dir (no %outDir%/%simName% macros).
    inlined = inlined.replace('"%outDir%/%simName%.csv"', '"COMP.csv"')
    inlined = inlined.replace('"%outDir%/%simName%DBG.csv"', '"COMPDBG.csv"')
    inlined = inlined.replace('"%outDir%/%simName%.gdx"', '"out.gdx"')

    # Apply the tm_pct tariff shock (replaces the numeraire pnum.fx jump).
    inlined = _patch_shock_to_tariff(inlined, tariff_increase=args.tariff)

    out_gms = out_dir / f"comp_{dataset}_gtap_shock_ifsub{ifsub}.gms"
    out_gms.write_text(inlined)
    print(f"\n  wrote {out_gms.name} ({len(inlined):,} chars)")

    remaining = re.findall(r"\$\$?(?:bat)?include\s+\"(\S+)\"", inlined)
    if remaining:
        print(f"  WARNING: unresolved includes: {set(remaining)}")
    else:
        print("  All $include directives resolved.")

    active_macros = set()
    for line in inlined.splitlines():
        if line.lstrip().startswith("*"):
            continue
        for macro in ("%inDir%", "%BaseName%", "%outDir%", "%simName%"):
            if macro in line:
                active_macros.add(macro)
    if active_macros:
        print(f"  NOTE: leftover macros in active code: {active_macros}")

    print(f"\nNext: run LOCALLY with GAMS v53:")
    print(f"  gams {out_gms.name} curDir={out_dir}  (-> out.gdx)")


if __name__ == "__main__":
    main()
