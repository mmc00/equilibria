"""Build a self-contained NEOS bundle for 9x10 altertax (fully inlined data).

Mirrors `build_nus333_neos_bundle.py` strategy:
  - NEOS's <gdx> base64 upload doesn't reach $gdxin reliably for large
    multi-symbol files; instead inline ALL data as GAMS Set/Parameter
    declarations + element-by-element assignments.

Pipeline:
  1. Merge `9x10Sets.gdx + 9x10Dat.gdx + 9x10Prm.gdx` → in.gdx (for the
     gdx_to_gams_inline.py converter to read in one pass).
  2. Run gdx_to_gams_inline.py → inline_data.gms (Set decls) +
     inline_params.gms (Param assignments).
  3. Inline all $include directives from comp_altertax.gms
     (getData/model/cal/iterloop/solve/postsim/mvar).
  4. Strip $gdxin / $load / $loadDC / closing $gdxin lines (comment-out).
  5. Strip the redeclaration block in getData.gms (sets acts comm ...;)
     since inline_data already declares them.
  6. Inject inline_data BEFORE the getData set-decl anchor.
  7. Split inline_params into Dat-side (used before ftrv assignment) and
     Prm-side (used after the Prm.gdx block), inject each at the right
     anchor.
  8. Rewrite output file paths to flat NEOS workdir (COMP.csv, out.gdx).
  9. Insert pwmg.l = 0 where tmarg=0 (9x10 reference convention).
 10. Insert pdp/pmp postsim recalc for alpha=0 cells.

Output:
  output/9x10_altertax_neos_bundle/comp_9x10_altertax_neos.gms
  output/9x10_altertax_neos_bundle/in.gdx  (kept for reference; not used by NEOS)
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path("/Users/marmol/proyectos2/cge_babel/standard_gtap_7")
GAMS_SYS = "/Library/Frameworks/GAMS.framework/Versions/48/Resources"


def _read(name: str) -> str:
    return (UPSTREAM / name).read_text()


def _resolve_mvar(arg_macro: str, arg_year: str) -> str:
    body = _read("mvar.gms")
    body = re.sub(r"^\$setargs\s+.*$", "", body, count=1, flags=re.MULTILINE)
    body = body.replace("%theMacro%", arg_macro).replace("%theYear%", arg_year)
    return body


def _resolve_solve(arg_solver: str) -> str:
    body = _read("solve.gms")
    body = body.replace("%1", arg_solver)
    return body


def _inline_includes(text: str) -> str:
    def _sub_batinclude(m: re.Match) -> str:
        target = m.group("file")
        args = (m.group("args") or "").split()
        if target == "solve.gms":
            return _resolve_solve(args[0] if args else "gtap")
        if target == "mvar.gms":
            return _resolve_mvar(*(args + ["", ""])[:2])
        body = _read(target)
        for i, a in enumerate(args, start=1):
            body = body.replace(f"%{i}", a)
        return body

    def _sub_include(m: re.Match) -> str:
        return _read(m.group("file"))

    pat_batinclude = re.compile(
        r"^\s*\$\$?batinclude\s+\"(?P<file>\S+\.(?:gms|inc))\"(?P<args>[^\n]*)$",
        re.MULTILINE,
    )
    pat_include = re.compile(
        r"^\s*\$\$?include\s+\"(?P<file>\S+\.(?:gms|inc))\"\s*$",
        re.MULTILINE,
    )

    for _ in range(8):
        new = pat_batinclude.sub(_sub_batinclude, text)
        new = pat_include.sub(_sub_include, new)
        if new == text:
            break
        text = new
    return text


def _strip_gdx_loads(text: str) -> str:
    """Comment out $gdxin / $load(DC) lines so the inline blocks take over."""
    pat = re.compile(
        r'^(\s*)(\$(?:gdxin|load[a-zA-Z]*)\b[^\n]*)$',
        re.MULTILINE,
    )
    return pat.sub(r'\1* [stripped for NEOS inline] \2', text)


def _strip_getdata_set_block(text: str) -> str:
    """Comment out the upstream `sets acts comm ...; ... endws(endw) ...;` block.

    Our inline_data.gms re-declares these. Without stripping we'd get a
    redeclaration error.
    """
    pat = re.compile(
        r'^sets\s*\n'
        r'\s*acts\s+"Activities"\s*\n'
        r'(?:\s*\w+[^\n]*\n)+?'
        r'\s*endws\(endw\)\s+"Sluggish factors"\s*\n'
        r';\s*$',
        re.MULTILINE,
    )
    def _commentify(m: re.Match) -> str:
        return "\n".join("* [getData set decl stripped] " + ln
                          for ln in m.group(0).splitlines())
    new_text, n = pat.subn(_commentify, text)
    if n == 0:
        raise SystemExit("Could not find getData.gms `sets acts comm ...;` block.")
    return new_text


def _insert_pwmg_fix_before_solve(text: str) -> str:
    fix = (
        "\n* === NEOS bundle fix: enforce 9x10 pwmg convention (0 where tmarg=0) ===\n"
        "pwmg.l(r,i,rp,tsim)$(tmarg.l(r,i,rp,tsim) eq 0) = 0 ;\n"
    )
    pat = re.compile(
        r"(pwmg\.fx\(r,i,rp,tsim\)\$\(not tmgFlag\(r,i,rp\)\)\s*=\s*pwmg\.l\(r,i,rp,tsim\)\s*;)",
        re.IGNORECASE,
    )
    if not pat.search(text):
        raise SystemExit("Could not locate `pwmg.fx ... = pwmg.l` line.")
    return pat.sub(fix + r"\1", text, count=1)


def _insert_pdp_pmp_recalc_before_unload(text: str) -> str:
    """Force pdp/pmp = pd/pm * (1+tax) for ALL cells before execute_unload."""
    fix = (
        "\n* === NEOS bundle fix: recompute pdp/pmp for ALL cells (alpha=0 too) ===\n"
        "pdp.l(r,i,aa,tsim) = pd.l(r,i,tsim) * (1 + dintx.l(r,i,aa,tsim)) ;\n"
        "pmp.l(r,i,aa,tsim) = pmt.l(r,i,tsim) * (1 + mintx.l(r,i,aa,tsim)) ;\n"
    )
    pat = re.compile(r'(execute_unload\s+"out\.gdx"\s*;)', re.IGNORECASE)
    if not pat.search(text):
        print("WARNING: could not locate execute_unload anchor.")
        return text
    return pat.sub(fix + r"\1", text, count=1)


def merge_gdx(out_path: Path) -> None:
    """Merge 9x10Sets.gdx + 9x10Dat.gdx + 9x10Prm.gdx → in.gdx."""
    import gams.transfer as gt

    merged = gt.Container(system_directory=GAMS_SYS)
    for src in ("9x10Sets.gdx", "9x10Dat.gdx", "9x10Prm.gdx"):
        c = gt.Container(str(UPSTREAM / src), system_directory=GAMS_SYS)
        for name, sym in c.data.items():
            if name in merged.data:
                continue
            sym._container = merged
            merged.data[name] = sym
    merged.write(str(out_path))
    print(f"  merged → {out_path.name} ({out_path.stat().st_size:,} bytes, "
          f"{len(merged.data)} symbols)")


def run_inliner(in_gdx: Path, out_data: Path, out_params: Path) -> None:
    cmd = [
        "uv", "run", "--with", "gamsapi", "--with", "pandas",
        "python", str(ROOT / "scripts/gtap/gdx_to_gams_inline.py"),
        "--gdx", str(in_gdx),
        "--out", str(out_data),
        "--out-params", str(out_params),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "output/9x10_altertax_neos_bundle")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== [1/4] Merging 9x10 GDX files into single in.gdx ===")
    in_gdx = args.out_dir / "in.gdx"
    merge_gdx(in_gdx)

    print("\n=== [2/4] Inlining GDX → GAMS Set/Param declarations ===")
    data_block_path = args.out_dir / "inline_data.gms"
    params_path = args.out_dir / "inline_params.gms"
    run_inliner(in_gdx, data_block_path, params_path)
    print(f"  inline_data:   {data_block_path.stat().st_size:,} bytes")
    print(f"  inline_params: {params_path.stat().st_size:,} bytes")

    print("\n=== [3/4] Inlining $include directives ===")
    main_script = _read("comp_altertax.gms")
    inlined = _inline_includes(main_script)
    print(f"  inlined size: {len(inlined):,} chars")

    print("\n=== [4/4] Stripping GDX loads, injecting data, applying fixes ===")
    inlined = _strip_gdx_loads(inlined)
    inlined = _strip_getdata_set_block(inlined)

    # Inject inline SETS block right at the stripped set-decl anchor.
    data_block = data_block_path.read_text()
    marker = "* [getData set decl stripped]"
    idx = inlined.find(marker)
    if idx == -1:
        raise SystemExit("Could not find getData set-decl marker.")
    inlined = inlined[:idx] + "\n" + data_block + "\n" + inlined[idx:]

    # Split inline_params into Dat-side (loaded into structures that
    # ftrv/ptax depend on) and Prm-side (elasticities loaded later).
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

    # Inject Dat-side params just before the ftrv assignment.
    ftrv_anchor = "ftrv(fp,a0,r) = evfp(fp,a0,r) - evfb(fp,a0,r) ;"
    f_idx = inlined.find(ftrv_anchor)
    if f_idx < 0:
        raise SystemExit("Could not find ftrv anchor for Dat params block.")
    inlined = inlined[:f_idx] + "\n" + dat_block + "\n\n" + inlined[f_idx:]

    # Inject Prm-side params after the Prm.gdx close marker.
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

    # Rewrite output paths to flat NEOS workdir.
    inlined = inlined.replace('"%outDir%/%simName%.csv"', '"COMP.csv"')
    inlined = inlined.replace('"%outDir%/%simName%DBG.csv"', '"COMPDBG.csv"')
    inlined = inlined.replace('"%outDir%/%simName%.gdx"', '"out.gdx"')

    # Convention fixes.
    inlined = _insert_pwmg_fix_before_solve(inlined)
    inlined = _insert_pdp_pmp_recalc_before_unload(inlined)

    out_gms = args.out_dir / "comp_9x10_altertax_neos.gms"
    out_gms.write_text(inlined)
    print(f"\n  wrote {out_gms.name} ({len(inlined):,} chars)")

    # Sanity checks.
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
        print(f"  NOTE: leftover macros in active code: {active_macros} "
              f"(check if these are in $ifthen exist guards)")


if __name__ == "__main__":
    main()
