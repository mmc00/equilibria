#!/usr/bin/env python3
"""Lee la corrida de GEMPACK traida de la maquina Windows.

Los .UPD son HAR, asi que los lee el lector propio de equilibria: no hace
falta GEMPACK en la Mac. Los .sl4.txt (texto de sltoht) se leen como respaldo.

Uso:
    python read_oracle.py <carpeta-out>              # inventario
    python read_oracle.py <carpeta-out> --exp TBL45A # una celda
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _har(p: Path):
    from equilibria.babel.har import read_har
    return read_har(p)


def inventory(out: Path) -> int:
    base = out / "_base"
    print(f"carpeta: {out}")
    rep = out / "_report.txt"
    if rep.exists():
        txt = rep.read_text(errors="replace")
        print(f"  fallos reportados: {txt.count('FALLO')}  ok: {txt.count('OK ')}")
    else:
        print("  [aviso] no hay _report.txt: la corrida puede estar incompleta")

    if (base / "base.upd").exists():
        print(f"  niveles BASE: {base/'base.upd'}")
    else:
        print("  [FALTA] _base/base.upd -- sin esto no hay comparacion en niveles")

    exps = sorted(d for d in out.iterdir() if d.is_dir() and not d.name.startswith("_"))
    print(f"  experimentos con carpeta: {len(exps)}")
    missing_upd = [d.name for d in exps if not list(d.glob("*.upd"))]
    if missing_upd:
        print(f"  [aviso] sin .upd ({len(missing_upd)}): {', '.join(missing_upd[:8])}"
              + (" ..." if len(missing_upd) > 8 else ""))
    return 0


def show(out: Path, exp: str) -> int:
    d = out / exp
    if not d.is_dir():
        print(f"no existe {d}", file=sys.stderr)
        return 1
    upds = list(d.glob("*.upd"))
    if not upds:
        print(f"{exp}: sin .upd", file=sys.stderr)
        return 1
    try:
        data = _har(upds[0])
    except Exception as e:
        print(f"no pude leer {upds[0]}: {e}", file=sys.stderr)
        return 1
    print(f"{exp}: {upds[0].name}  headers={len(data)}")
    for k in sorted(data)[:40]:
        arr = getattr(data[k], "array", data[k])
        shape = getattr(arr, "shape", "?")
        print(f"   {k:<10} {shape}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--exp")
    a = ap.parse_args()
    if not a.out.is_dir():
        print(f"no existe {a.out}", file=sys.stderr)
        return 1
    return show(a.out, a.exp) if a.exp else inventory(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
