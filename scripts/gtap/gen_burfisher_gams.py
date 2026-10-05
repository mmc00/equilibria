"""Burfisher 3e / nus333: genera el oraculo GAMS de cada ejercicio.

Lee los shocks de la MISMA tabla que usa equilibria (``run_burfisher.EXERCISES``),
escribe ``<EXP>.inc`` (el shock en niveles, en la celda del periodo shock) y corre
``gams/comp_shock.gms`` (base/check/shock con model.gms + cal.gms del GTAP 7 de
referencia). Salida: ``<out_dir>/<EXP>_capFlex.gdx``, la que lee
``run_burfisher.py --gams-dir <out_dir>``.

Antes de correr, parchea una copia de las fuentes GAMS de referencia (nunca el repo):

- getData.gms: un factor con ETRE finito en el .prm es sluggish, como en GEMPACK y en
  equilibria (factor.py). GAMS lo toma de endwm/endws e ignora el ETRE.
- model.gms: tres cierres que GEMPACK hace con ``swap`` y GAMS no tiene. Con su flag en
  0 (todos los demas ejercicios, y base/check) las filas no existen o son las originales.
  * desempleo (UNEMP): xfteq fija el salario real pft/deflactor en vez del empleo;
  * objetivo de cantidad (QCA): x(r,a,i) fijo con prdtx endogeno;
  * PIB real objetivo (GDP): rgdpmp(r) fijo con axpreg endogeno.

Uso:
    .venv/bin/python scripts/gtap/gen_burfisher_gams.py <out_dir> [--only TBL46A,ME9C]

Requiere GAMS (``gams`` en el PATH o ``--gams``) con PATH, y el dataset nus333 con
los .prm del libro (``EQUILIBRIA_NUS333_DIR``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from run_burfisher import EXERCISES, _me9  # noqa: E402

GAMS_SRC = ROOT / "src" / "equilibria" / "templates" / "reference" / "gtap" / "scripts"
COMP_SHOCK = Path(__file__).resolve().parent / "gams" / "comp_shock.gms"
HAR_FILES = ("basedata.har", "sets.har", "baserate.har")


def q(*labels: str) -> str:
    return ",".join(f"'{x}'" for x in labels)


# Ejercicios con cierre distinto de pleno empleo: (ejercicio con el mismo shock,
# celdas (r, fm) con salario real pft/deflactor fijo y empleo xft endogeno).
# GEMPACK: swap qe(f,r) = pebfactreal(f,r) (ME3C.EXP); pebfactreal = peb/ppriv.
UNEMP = {
    "TBL65B": ("TBL65A", [("USA", "LABOR")]),
    "ME3C": ("TBL65A", [("USA", "LABOR")]),
}


# Deflactor del salario real (GEMPACK ppriv, un Divisia): (expresion en la fila del
# shock, su valor en check). Medido contra TBL65B.sl4 (qe LABOR USA +38,7830):
# tornq +38,7809, cv +38,842, pabs +38,844, pcons +37,014. Default: tornq.
DEFL = {
    "pcons": ("pcons(r,t)", "pcons.l({r},'check')"),
    "cv": ("cv(r,'hhd',t)", "cv.l({r},'hhd','check')"),
    "pabs": ("pabs(r,t)", "pabs.l({r},'check')"),
    # Tornqvist del consumo de hogares contra check (= Divisia de ppriv en un paso):
    # ln P = sum_i 1/2 (s_i,check + s_i) ln(pa_i/pa_i,check), s_i = pa*xa/sum pa*xa.
    "tornq": (
        "exp(sum(i$xaFlag(r,i,'hhd'), 0.5*(sck(r,i) + pa(r,i,'hhd',t)*xa(r,i,'hhd',t)"
        "/sum(j$xaFlag(r,j,'hhd'), pa(r,j,'hhd',t)*xa(r,j,'hhd',t)))"
        "*log(pa(r,i,'hhd',t)/pack(r,i))))",
        "1",
    ),
}


# Cierres con un objetivo de cantidad: (prm, [(r, a, i, %)]). GEMPACK `swap qca(i,a,r)
# = to(i,a,r)` + `shock qca = %`: x(r,a,i) queda en check*(1+%/100) y el impuesto
# prdtx(r,a,i) de esa celda pasa a endogeno.
QCA = {
    "TBL94": ("default.prm", [("USA", "MFG", "MFG", -1.0)]),
}


# Cierres con PIB real objetivo: (prm, {r: %}, shocks). GEMPACK `swap qgdp(reg) =
# aoreg(reg)` + `shock qgdp = %`: rgdpmp(r) queda en check*(1+%/100) y el shifter
# regional axpreg(r) (axp = axp(t-1)*(1+axpsec+axpreg+axpall), model.gms) pasa a
# endogeno. Los demas shocks son los de ME9B sin los de axp (ME9B fija aoreg en
# 31,94/42,31, que es lo que ME9A calcula).
GDP = {
    "ME9A": (
        "climatechange.prm",
        {"USA": 109.6, "ROW": 284.5},
        [s for s in _me9(land=(-0.93, 4.4), ao_agr=0.0, afe=None) if s[0] != "axp"],
    ),
}


# Ejercicios cuyo shock PATH no resuelve desde el check: se arranca desde el shock de
# otro ejercicio GAMS (homotopia, nada de equilibria). ME9D = ME9C + afeall LABOR:
# desde el check PATH termina Locally Infeasible; desde el shock de ME9C, Optimal.
# Las Vars son las endogenas del shock (sin instrumentos .fx: pop va fijo en el .inc).
WARM_VARS = [
    "arent",
    "axp",
    "chiSave",
    "chif",
    "dintx",
    "factY",
    "fcttx",
    "gdpmp",
    "kapEnd",
    "kappaf",
    "kstock",
    "lambdaf",
    "lambdam",
    "lambdava",
    "mintx",
    "nd",
    "p",
    "pa",
    "pabs",
    "pcons",
    "pd",
    "pe",
    "pet",
    "pf",
    "pfact",
    "pft",
    "pg",
    "pgdpmp",
    "phi",
    "phiP",
    "pi",
    "pigbl",
    "pmt",
    "pnd",
    "pnum",
    "ps",
    "psave",
    "ptmg",
    "pva",
    "pwfact",
    "px",
    "regY",
    "rgdpmp",
    "rorc",
    "rore",
    "rorg",
    "rsav",
    "savf",
    "va",
    "x",
    "xa",
    "xd",
    "xds",
    "xet",
    "xf",
    "xft",
    "xi",
    "xigbl",
    "xm",
    "xmt",
    "xp",
    "xs",
    "xtmg",
    "xw",
    "yc",
    "yg",
    "yi",
    "ytax",
    "ytaxInd",
    "ytaxTot",
    "ytaxshr",
]
WARM = {"ME9D": "ME9C"}


def all_exercises() -> list[str]:
    """Los 37 de EXERCISES (ME9C antes que ME9D) y los 4 de cierre propio."""
    return [*EXERCISES, *UNEMP, *QCA, *GDP]


def warm_lines(out: Path, src: str) -> list[str]:
    """Carga los niveles de WARM_VARS del GDX de ``src`` antes del solve del shock."""
    gdx = out / f"{src}_capFlex.gdx"
    if not gdx.exists():
        raise SystemExit(f"falta {gdx}: correr {src} antes (p.ej. --only {src},...)")
    return [f'execute_loadpoint "{gdx}", {", ".join(WARM_VARS)} ;']


def gdp_lines(targets: dict[str, float]) -> list[str]:
    """Libera axpreg de cada region en el shock y fija rgdpmp en su objetivo."""
    out = []
    for r, pct in targets.items():
        k = q(r)
        out.append(f"gdpFlag({k}) = 1 ;")
        out.append(f"gdpTgt({k}) = rgdpmp.l({k},'check')*{1 + pct / 100!r} ;")
        out.append(f"axpreg.lo({k},tsim) = -inf ; axpreg.up({k},tsim) = +inf ;")
    return out


def qca_lines(cells: list[tuple[str, str, str, float]]) -> list[str]:
    """Libera prdtx de cada celda en el shock y fija x en su objetivo."""
    out = []
    for r, a, i, pct in cells:
        k = q(r, "a_" + a, "c_" + i)
        out.append(f"qcaFlag({k}) = 1 ;")
        out.append(f"qcaTgt({k}) = x.l({k},'check')*{1 + pct / 100!r} ;")
        out.append(f"prdtx.lo({k},tsim) = -inf ; prdtx.up({k},tsim) = +inf ;")
    return out


def unemp_lines(cells: list[tuple[str, str]], defl: str) -> list[str]:
    """Activa el salario real fijo en el shock, en su valor de check."""
    out = []
    for r, f in cells:
        k = q(r, f)
        rr = q(r)
        out.append(f"wrFlag({k}) = 1 ;")
        out.append(f"wreal0({k}) = pft.l({k},'check')/{DEFL[defl][1].format(r=rr)} ;")
        out.append(f"pack({rr},i) = pa.l({rr},i,'hhd','check') ;")
        out.append(
            f"sck({rr},i)$xaFlag({rr},i,'hhd') = "
            f"pa.l({rr},i,'hhd','check')*xa.l({rr},i,'hhd','check')"
            f"/sum(j$xaFlag({rr},j,'hhd'), "
            f"pa.l({rr},j,'hhd','check')*xa.l({rr},j,'hhd','check')) ;"
        )
    return out


def gams_line(name: str, idx: tuple, kind: str, pct: float) -> str:
    """Una sentencia GAMS por shock. .l(...,tsim) es el valor de check (iterloop)."""
    f = 1 + pct / 100
    if name == "lambdava":  # lambdava = lambdava(t-1)*(1+avaall)
        r, a = idx
        return f"avaall.fx({q(r, 'a_' + a)},tsim) = {pct / 100!r} ;"
    if name == "aft":  # xft = aft*(...)^etaf, aft Parameter por t
        r, fa = idx
        return f"aft({q(r, fa)},tsim) = aft({q(r, fa)},tsim)*{f!r} ;"
    if name == "imptx":
        e, i, d = idx
        k = q(e, "c_" + i, d)
        return f"imptx.fx({k},tsim) = (1 + imptx.l({k},tsim))*{f!r} - 1 ;"
    if name == "prdtx_rai":
        r, a, i = idx
        k = q(r, "a_" + a, "c_" + i)
        return f"prdtx.fx({k},tsim) = (1 + prdtx.l({k},tsim))*{f!r} - 1 ;"
    if name == "fcttx":
        r, fa, a = idx
        k = q(r, fa, "a_" + a)
        return (
            f"fcttx.fx({k},tsim) = (1 + fctts.l({k},tsim) + fcttx.l({k},tsim))*{f!r}"
            f" - 1 - fctts.l({k},tsim) ;"
        )
    if name in ("dintx_tgt", "mintx_tgt"):
        var = name.removesuffix("_tgt")
        r, i, aa = idx
        k = q(r, "c_" + i, aa if aa in ("hhd", "gov", "inv") else "a_" + aa)
        return f"{var}.fx({k},tsim) = (1 + {var}.l({k},tsim))*{f!r} - 1 ;"
    if name == "kappaf":
        # tinc = potencia EVFB/EVOS = 1/(1-kappaf) (cal.gms) -> 1-kappaf' = (1-kappaf)/f.
        r, fa, a = idx
        k = q(r, fa, "a_" + a)
        return f"kappaf.fx({k},tsim) = 1 - (1 - kappaf.l({k},tsim))/{f!r} ;"
    if name == "exptx":  # txs = potencia 1+exptx (cal.gms)
        e, i, d = idx
        k = q(e, "c_" + i, d)
        return f"exptx.fx({k},tsim) = (1 + exptx.l({k},tsim))*{f!r} - 1 ;"
    if name == "pop":  # pop(r,t) variable fija (cal.gms)
        (r,) = idx
        return f"pop.fx({q(r)},tsim) = pop.l({q(r)},tsim)*{f!r} ;"
    if name == "lambdaf":  # lambdaf = lambdaf(t-1)*(1+afeall)
        r, fa, a = idx
        return f"afeall.fx({q(r, fa, 'a_' + a)},tsim) = {pct / 100!r} ;"
    if name == "axp":  # axp = axp(t-1)*(1+axpall)
        r, a = idx
        return f"axpall.fx({q(r, 'a_' + a)},tsim) = {pct / 100!r} ;"
    if name == "lambdamg":  # xmgm = amgm*xwmg/lambdamg (model.gms)
        mg, e, i, d = idx
        k = q("c_" + mg, e, "c_" + i, d)
        return f"lambdamg.fx({k},tsim) = lambdamg.l({k},tsim)*{f!r} ;"
    if name == "lambdam":
        e, i, d = idx
        k = q(e, "c_" + i, d)
        return f"lambdam.fx({k},tsim) = lambdam.l({k},tsim)*{f!r} ;"
    raise ValueError(name)


def shock_inc(exp: str, out: Path, defl: str) -> tuple[str, str]:
    """(.prm del ejercicio, texto del .inc con su shock y su cierre)."""
    if exp in GDP:
        prm, targets, shocks = GDP[exp]
        extra = gdp_lines(targets)
    elif exp in QCA:
        prm, cells = QCA[exp]
        shocks = []
        extra = qca_lines(cells)
    elif exp in UNEMP:
        src, cells = UNEMP[exp]
        prm, shocks = EXERCISES[src]
        extra = unemp_lines(cells, defl)
    else:
        prm, shocks = EXERCISES[exp]
        extra = []
    if exp in WARM:
        extra = [*extra, *warm_lines(out, WARM[exp])]
    lines = [*(gams_line(*s) for s in shocks), *extra]
    header = f"* {exp} ({prm}) -- generado por gen_burfisher_gams.py\n"
    return prm, header + "\n".join(lines) + "\n"


def patch_gams_sources(work: Path, defl: str) -> None:
    """Aplica los parches de movilidad y de cierres a la copia en ``work``."""
    gd = work / "getData.gms"
    txt = gd.read_text()
    mark = "loop(fm,\n   loop(endwm,"
    patch = (
        "* [F-val] factor con ETRE != 0 en el prm -> sluggish (regla GEMPACK/equilibria)\n"
        "loop((fp,endw)$(sameas(fp,endw) and sum(r, abs(etrae(fp,r)))),\n"
        "   endws(endw) = yes ; endwm(endw) = no ;\n) ;\n"
    )
    if patch not in txt:
        assert mark in txt, "getData.gms cambio: revisar el parche de movilidad"
        gd.write_text(txt.replace(mark, patch + mark, 1))

    md = work / "model.gms"
    txt = md.read_text()
    old = (
        "xfteq(r,fm,t)$(rs(r) and ts(t) and xftFlag(r,fm))..\n"
        "   xft(r,fm,t) =e= aft(r,fm,t)*(pft(r,fm,t)/pabs(r,t))**etaf(r,fm) ;\n"
    )
    new = (
        "* [F-val] cierre de desempleo: salario real fijo en las celdas con wrFlag\n"
        "Parameter wrFlag(r,fp), wreal0(r,fp), sck(r,i), pack(r,i) ;\n"
        "wrFlag(r,fp) = 0 ; wreal0(r,fp) = 1 ; sck(r,i) = 0 ; pack(r,i) = 1 ;\n"
        "xfteq(r,fm,t)$(rs(r) and ts(t) and xftFlag(r,fm))..\n"
        f"   xft(r,fm,t)$(not wrFlag(r,fm)) + (pft(r,fm,t)/{DEFL[defl][0]})$wrFlag(r,fm)\n"
        "   =e= (aft(r,fm,t)*(pft(r,fm,t)/pabs(r,t))**etaf(r,fm))$(not wrFlag(r,fm))\n"
        "     + wreal0(r,fm)$wrFlag(r,fm) ;\n"
    )
    if new not in txt:
        assert old in txt, "model.gms cambio: revisar el parche de xfteq"
        txt = txt.replace(old, new, 1)

    old_m = "model gtap /\n"
    new_m = (
        "* [F-val] objetivo de cantidad: x(r,a,i) = qcaTgt con prdtx endogeno\n"
        "Parameter qcaFlag(r,a,i), qcaTgt(r,a,i) ;\n"
        "qcaFlag(r,a,i) = 0 ; qcaTgt(r,a,i) = 0 ;\n"
        "Equation qcaeq(r,a,i,t) ;\n"
        "qcaeq(r,a,i,t)$(rs(r) and ts(t) and qcaFlag(r,a,i))..\n"
        "   x(r,a,i,t) =e= qcaTgt(r,a,i) ;\n\n"
        "* [F-val] PIB real objetivo: rgdpmp(r) = gdpTgt con axpreg endogeno\n"
        "Parameter gdpFlag(r), gdpTgt(r) ;\n"
        "gdpFlag(r) = 0 ; gdpTgt(r) = 0 ;\n"
        "Equation gdpeq(r,t) ;\n"
        "gdpeq(r,t)$(rs(r) and ts(t) and gdpFlag(r))..\n"
        "   rgdpmp(r,t) =e= gdpTgt(r) ;\n\n"
        "model gtap /\n   qcaeq.prdtx, gdpeq.axpreg,\n"
    )
    if new_m not in txt:
        assert txt.count(old_m) == 1, "model.gms cambio: revisar el parche de qcaeq"
        txt = txt.replace(old_m, new_m, 1)
    md.write_text(txt)


def run_exercise(
    exp: str, out: Path, work: Path, nus333: Path, gams: str, iterlim: int, defl: str
) -> str:
    """Genera el .inc, los GDX de entrada y corre GAMS. Devuelve el MODEL STATUS."""
    from equilibria.babel.har_to_gdx import write_nus333_gdx_bundle

    prm, text = shock_inc(exp, out, defl)
    inc = work / "shocks" / f"{exp}.inc"
    inc.parent.mkdir(exist_ok=True)
    inc.write_text(text)

    har = work / f"har_{exp}"
    har.mkdir(exist_ok=True)
    for name, src in [*((f, f) for f in HAR_FILES), ("default.prm", prm)]:
        (har / name).unlink(missing_ok=True)
        (har / name).symlink_to(nus333 / src)
    gdx_in = work / f"gdx_{exp}"
    write_nus333_gdx_bundle(har, gdx_in)

    util = "CD" if "cobbdouglas" in prm.lower() else "cde"
    subprocess.run(
        [
            gams,
            "comp_shock.gms",
            "--baseName=nus333",
            f"--inDir={gdx_in}",
            f"--outDir={out}",
            "--savfFlag=capFlex",
            f"--utility={util}",
            f"--shockInc={inc}",
            f"--simName={exp}_capFlex",
            f"--iterLim={iterlim}",
            f"o={out}/{exp}.lst",
            "lo=2",
            f"lf={out}/{exp}.log",
        ],
        cwd=work,
        capture_output=True,
    )
    lst = (out / f"{exp}.lst").read_text(errors="ignore")
    status = sorted({ln.strip() for ln in lst.splitlines() if "MODEL STATUS" in ln})
    return f"{prm} {util} {status}"


def main() -> int:
    from equilibria._local_refs import nus333_dir

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path, help="donde quedan <EXP>_capFlex.gdx")
    ap.add_argument("--only", default="", help="EXP separados por coma")
    ap.add_argument("--gams", default=shutil.which("gams"), help="binario de GAMS")
    ap.add_argument(
        "--work-dir", type=Path, help="copia parcheada de las fuentes (def. temporal)"
    )
    ap.add_argument("--iterlim", type=int, default=1000, help="iteraciones de PATH")
    ap.add_argument(
        "--unemp-defl",
        choices=sorted(DEFL),
        default="tornq",
        help="deflactor del salario real en el cierre de desempleo",
    )
    args = ap.parse_args()
    if not args.gams:
        ap.error("no se encontro gams en el PATH: usar --gams")
    nus333 = nus333_dir()
    if not (nus333 / "basedata.har").exists():
        ap.error(f"no esta el dataset nus333 en {nus333} (EQUILIBRIA_NUS333_DIR)")

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="burfisher_gams_"))
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    shutil.copytree(GAMS_SRC, work, dirs_exist_ok=True)
    shutil.copy(COMP_SHOCK, work)
    patch_gams_sources(work, args.unemp_defl)

    todo = [e for e in args.only.split(",") if e] or all_exercises()
    for exp in todo:
        status = run_exercise(
            exp, out, work, nus333, args.gams, args.iterlim, args.unemp_defl
        )
        print(exp, status, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
