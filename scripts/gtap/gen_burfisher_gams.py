"""Burfisher 3e / nus333: genera el oraculo GAMS de cada ejercicio.

Lee los shocks de la MISMA tabla que usa equilibria (``run_burfisher.EXERCISES``),
escribe ``<EXP>.inc`` (el shock en niveles, en la celda del periodo shock) y corre
``gams/comp_shock.gms`` (base/check/shock con model.gms + cal.gms del GTAP 7 de
referencia). Salida: ``<out_dir>/<EXP>_capFlex.gdx``, la que lee
``run_burfisher.py --gams-dir <out_dir>``. El shock en niveles sale de la misma
formula que usa equilibria (``run_burfisher.shocked``), escrita como expresion GAMS.

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

Codigo de salida: 0 si todos terminan en un optimo; 2 si alguno no (deja su GDX:
ME4A/B terminan Intermediate Infeasible, asi que la corrida completa da 2); 1 si GAMS
fallo en alguno (sin GDX).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "gtap"))

from run_burfisher import EXERCISES, me9, shocked  # noqa: E402

GAMS_SRC = ROOT / "src" / "equilibria" / "templates" / "reference" / "gtap" / "scripts"
COMP_SHOCK = Path(__file__).resolve().parent / "gams" / "comp_shock.gms"
HAR_FILES = ("basedata.har", "sets.har", "baserate.har")


def gams_labels(*labels: str) -> str:
    """Las etiquetas de un indice GAMS entre comillas: 'USA','LABOR'."""
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
UNEMP_DEFLATORS = {
    "pcons": ("pcons(r,t)", "pcons.l({r},'check')"),
    "cv": ("cv(r,'hhd',t)", "cv.l({r},'hhd','check')"),
    "pabs": ("pabs(r,t)", "pabs.l({r},'check')"),
    # Tornqvist del consumo de hogares contra check (= Divisia de ppriv en un paso):
    # ln P = sum_i 1/2 (s_i,check + s_i) ln(pa_i/pa_i,check), s_i = pa*xa/sum pa*xa.
    # sck(r,i) y pack(r,i) son s_i y pa_i en check (los fija unemp_lines).
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
        [s for s in me9(land=(-0.93, 4.4), ao_agr=0.0, afe=None) if s[0] != "axp"],
    ),
}


# Ejercicios cuyo shock PATH no resuelve desde el check: se arranca desde el shock de
# otro ejercicio GAMS (homotopia, nada de equilibria). ME9D = ME9C + afeall LABOR:
# desde el check PATH termina Locally Infeasible; desde el shock de ME9C, Optimal.
# Las Vars son las endogenas del shock (sin instrumentos .fx: pop va fijo en el .inc).
WARM_VARS = (  # noqa: SIM905
    "arent axp chiSave chif dintx factY fcttx gdpmp kapEnd kappaf kstock lambdaf "
    "lambdam lambdava mintx nd p pa pabs pcons pd pe pet pf pfact pft pg pgdpmp phi "
    "phiP pi pigbl pmt pnd pnum ps psave ptmg pva pwfact px regY rgdpmp rorc rore "
    "rorg rsav savf va x xa xd xds xet xf xft xi xigbl xm xmt xp xs xtmg xw yc yg yi "
    "ytax ytaxInd ytaxTot ytaxshr"
).split()
WARM = {"ME9D": "ME9C"}


def all_exercises() -> list[str]:
    """Los 37 de EXERCISES (ME9C antes que ME9D) y los 4 de cierre propio."""
    return [*EXERCISES, *UNEMP, *QCA, *GDP]


class GamsExpr:
    """Una expresion GAMS en texto, para evaluar ``run_burfisher.shocked`` sobre ella.

    Pone parentesis solo donde hacen falta: ``(1 + x)*1.1 - 1``, ``a/(b*c)``.
    """

    ATOM, PRODUCT, SUM = 0, 1, 2  # operador al tope de la expresion

    def __init__(self, text: str, top: int = 0) -> None:
        self.text = text
        self.top = top

    @staticmethod
    def _text(x: object, wrap_from: int = 3) -> str:
        """El texto de ``x``, entre parentesis si su operador al tope es >= wrap_from."""
        if isinstance(x, GamsExpr):
            return f"({x.text})" if x.top >= wrap_from else x.text
        return repr(x)

    def _sum(self, a: object, op: str, b: object) -> GamsExpr:
        right = self._text(b, wrap_from=self.SUM if op == "-" else 3)
        return GamsExpr(f"{self._text(a)} {op} {right}", self.SUM)

    def _product(self, a: object, op: str, b: object) -> GamsExpr:
        # a/(b*c) y a/(b/c) necesitan parentesis; a*(b/c) = a*b/c no.
        right = self._text(b, wrap_from=self.PRODUCT if op == "/" else self.SUM)
        return GamsExpr(f"{self._text(a, wrap_from=self.SUM)}{op}{right}", self.PRODUCT)

    def __add__(self, o: object) -> GamsExpr:
        return self._sum(self, "+", o)

    def __radd__(self, o: object) -> GamsExpr:
        return self._sum(o, "+", self)

    def __sub__(self, o: object) -> GamsExpr:
        return self._sum(self, "-", o)

    def __rsub__(self, o: object) -> GamsExpr:
        return self._sum(o, "-", self)

    def __mul__(self, o: object) -> GamsExpr:
        return self._product(self, "*", o)

    def __truediv__(self, o: object) -> GamsExpr:
        return self._product(self, "/", o)

    def __str__(self) -> str:
        return self.text


def _act(a: str) -> str:
    return "a_" + a


def _com(i: str) -> str:
    return "c_" + i


def _agent(aa: str) -> str:
    return aa if aa in ("hhd", "gov", "inv") else _act(aa)


# Instrumento de equilibria -> (variable GAMS, su indice desde el de equilibria).
LEVEL_VARS = {
    "aft": ("aft", lambda r, fa: (r, fa)),  # Parameter por t, sin .fx/.l
    "imptx": ("imptx", lambda e, i, d: (e, _com(i), d)),
    "exptx": ("exptx", lambda e, i, d: (e, _com(i), d)),
    "lambdam": ("lambdam", lambda e, i, d: (e, _com(i), d)),
    "prdtx_rai": ("prdtx", lambda r, a, i: (r, _act(a), _com(i))),
    "fcttx": ("fcttx", lambda r, fa, a: (r, fa, _act(a))),
    "kappaf": ("kappaf", lambda r, fa, a: (r, fa, _act(a))),
    "dintx_tgt": ("dintx", lambda r, i, aa: (r, _com(i), _agent(aa))),
    "mintx_tgt": ("mintx", lambda r, i, aa: (r, _com(i), _agent(aa))),
    "pop": ("pop", lambda r: (r,)),
    "lambdamg": ("lambdamg", lambda mg, e, i, d: (_com(mg), e, _com(i), d)),
}
# Instrumentos que en GAMS crecen por una tasa: x = x(t-1)*(1+tasa). Con kind "pct",
# la tasa es pct/100 (model.gms: lambdava/avaall, lambdaf/afeall, axp/axpall).
RATE_VARS = {
    "lambdava": ("avaall", lambda r, a: (r, _act(a))),
    "lambdaf": ("afeall", lambda r, fa, a: (r, fa, _act(a))),
    "axp": ("axpall", lambda r, a: (r, _act(a))),
}


def gams_line(name: str, idx: tuple, kind: str, pct: float) -> str:
    """Una sentencia GAMS por shock. .l(...,tsim) es el valor de check (iterloop)."""
    if name in RATE_VARS:
        if kind != "pct":
            raise ValueError(f"{name}: kind {kind!r}, se esperaba 'pct'")
        var, index = RATE_VARS[name]
        return f"{var}.fx({gams_labels(*index(*idx))},tsim) = {pct / 100!r} ;"
    if name not in LEVEL_VARS:
        raise ValueError(f"instrumento sin traduccion a GAMS: {name!r}")
    var, index = LEVEL_VARS[name]
    k = gams_labels(*index(*idx))
    if name == "aft":
        lhs = chk = f"aft({k},tsim)"
    else:
        lhs, chk = f"{var}.fx({k},tsim)", f"{var}.l({k},tsim)"
    fs = GamsExpr(f"fctts.l({k},tsim)")
    value = shocked(kind, GamsExpr(chk), 1 + pct / 100, fs)
    return f"{lhs} = {value} ;"


def warm_lines(out: Path, src: str) -> list[str]:
    """Carga los niveles de WARM_VARS del GDX de ``src`` antes del solve del shock."""
    gdx = out / f"{src}_capFlex.gdx"
    if not gdx.exists():
        raise FileNotFoundError(f"falta {gdx}: correr {src} antes (--only {src},...)")
    return [f'execute_loadpoint "{gdx}", {", ".join(WARM_VARS)} ;']


def gdp_lines(targets: dict[str, float]) -> list[str]:
    """Libera axpreg de cada region en el shock y fija rgdpmp en su objetivo."""
    out = []
    for r, pct in targets.items():
        k = gams_labels(r)
        out.append(f"gdpFlag({k}) = 1 ;")
        out.append(f"gdpTgt({k}) = rgdpmp.l({k},'check')*{1 + pct / 100!r} ;")
        out.append(f"axpreg.lo({k},tsim) = -inf ; axpreg.up({k},tsim) = +inf ;")
    return out


def qca_lines(cells: list[tuple[str, str, str, float]]) -> list[str]:
    """Libera prdtx de cada celda en el shock y fija x en su objetivo."""
    out = []
    for r, a, i, pct in cells:
        k = gams_labels(r, _act(a), _com(i))
        out.append(f"qcaFlag({k}) = 1 ;")
        out.append(f"qcaTgt({k}) = x.l({k},'check')*{1 + pct / 100!r} ;")
        out.append(f"prdtx.lo({k},tsim) = -inf ; prdtx.up({k},tsim) = +inf ;")
    return out


def unemp_lines(cells: list[tuple[str, str]], deflator: str) -> list[str]:
    """Activa el salario real fijo en el shock, en su valor de check."""
    check_value = UNEMP_DEFLATORS[deflator][1]
    out = []
    for r, f in cells:
        k, rr = gams_labels(r, f), gams_labels(r)
        out.append(f"wrFlag({k}) = 1 ;")
        out.append(f"wreal0({k}) = pft.l({k},'check')/{check_value.format(r=rr)} ;")
        out.append(f"pack({rr},i) = pa.l({rr},i,'hhd','check') ;")
        out.append(
            f"sck({rr},i)$xaFlag({rr},i,'hhd') = "
            f"pa.l({rr},i,'hhd','check')*xa.l({rr},i,'hhd','check')"
            f"/sum(j$xaFlag({rr},j,'hhd'), "
            f"pa.l({rr},j,'hhd','check')*xa.l({rr},j,'hhd','check')) ;"
        )
    return out


def exercise_prm(exp: str) -> str:
    """El .prm del libro que usa el ejercicio."""
    if exp in GDP:
        return GDP[exp][0]
    if exp in QCA:
        return QCA[exp][0]
    return EXERCISES[UNEMP[exp][0] if exp in UNEMP else exp][0]


def shock_inc(exp: str, out: Path, deflator: str) -> tuple[str, str]:
    """(.prm del ejercicio, texto del .inc con su shock y su cierre)."""
    if exp in GDP:
        _, targets, shocks = GDP[exp]
        extra = gdp_lines(targets)
    elif exp in QCA:
        shocks, extra = [], qca_lines(QCA[exp][1])
    elif exp in UNEMP:
        src, cells = UNEMP[exp]
        shocks, extra = EXERCISES[src][1], unemp_lines(cells, deflator)
    else:
        shocks, extra = EXERCISES[exp][1], []
    if exp in WARM:
        extra = [*extra, *warm_lines(out, WARM[exp])]
    prm = exercise_prm(exp)
    lines = [*(gams_line(*s) for s in shocks), *extra]
    header = f"* {exp} ({prm}) -- generado por gen_burfisher_gams.py\n"
    return prm, header + "\n".join(lines) + "\n"


def patch_gams_sources(work: Path, deflator: str) -> None:
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
        if mark not in txt:
            raise RuntimeError("getData.gms cambio: revisar el parche de movilidad")
        gd.write_text(txt.replace(mark, patch + mark, 1))

    md = work / "model.gms"
    txt = md.read_text()
    old = (
        "xfteq(r,fm,t)$(rs(r) and ts(t) and xftFlag(r,fm))..\n"
        "   xft(r,fm,t) =e= aft(r,fm,t)*(pft(r,fm,t)/pabs(r,t))**etaf(r,fm) ;\n"
    )
    deflator_expr = UNEMP_DEFLATORS[deflator][0]
    new = (
        "* [F-val] cierre de desempleo: salario real fijo en las celdas con wrFlag\n"
        "Parameter wrFlag(r,fp), wreal0(r,fp), sck(r,i), pack(r,i) ;\n"
        "wrFlag(r,fp) = 0 ; wreal0(r,fp) = 1 ; sck(r,i) = 0 ; pack(r,i) = 1 ;\n"
        "xfteq(r,fm,t)$(rs(r) and ts(t) and xftFlag(r,fm))..\n"
        f"   xft(r,fm,t)$(not wrFlag(r,fm)) + (pft(r,fm,t)/{deflator_expr})$wrFlag(r,fm)\n"
        "   =e= (aft(r,fm,t)*(pft(r,fm,t)/pabs(r,t))**etaf(r,fm))$(not wrFlag(r,fm))\n"
        "     + wreal0(r,fm)$wrFlag(r,fm) ;\n"
    )
    if new not in txt:
        if old not in txt:
            raise RuntimeError("model.gms cambio: revisar el parche de xfteq")
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
        if txt.count(old_m) != 1:
            raise RuntimeError("model.gms cambio: revisar el parche de qcaeq")
        txt = txt.replace(old_m, new_m, 1)
    md.write_text(txt)


def missing_inputs(nus333: Path, exercises: list[str]) -> list[Path]:
    """Los archivos del dataset que faltan para correr ``exercises``."""
    names = {*HAR_FILES, *(exercise_prm(e) for e in exercises)}
    return sorted(nus333 / n for n in names if not (nus333 / n).exists())


OPTIMAL = ("1 Optimal", "2 Locally Optimal")


def all_optimal(statuses: list[str]) -> bool:
    """Si todos los solves del .lst (check y shock) terminaron en un optimo."""
    return bool(statuses) and all(s.endswith(OPTIMAL) for s in statuses)


@dataclass(frozen=True)
class GamsRun:
    """Lo comun a todos los ejercicios de una corrida."""

    out: Path  # donde quedan <EXP>_capFlex.gdx/.lst/.log
    work: Path  # copia parcheada de las fuentes GAMS
    nus333: Path
    gams: str  # ruta absoluta (GAMS corre con cwd=work)
    iterlim: int
    deflator: str


SOLVED, NOT_OPTIMAL, FAILED = "resuelto", "sin optimo", "fallo"


def run_exercise(exp: str, run: GamsRun) -> tuple[str, str]:
    """Genera el .inc, los GDX de entrada y corre GAMS.

    (SOLVED | NOT_OPTIMAL | FAILED, linea para imprimir). NOT_OPTIMAL deja el GDX
    (ME4A/B terminan asi y el paper los reporta como N/A); FAILED no deja nada.
    """
    from equilibria.babel.har_to_gdx import write_nus333_gdx_bundle

    for f in ("lst", "log"):
        (run.out / f"{exp}.{f}").unlink(missing_ok=True)
    gdx_out = run.out / f"{exp}_capFlex.gdx"
    gdx_out.unlink(missing_ok=True)  # que un resultado viejo no pase por nuevo
    try:
        prm, text = shock_inc(exp, run.out, run.deflator)
    except FileNotFoundError as e:  # el arranque de ME9D sin el GDX de ME9C
        return FAILED, str(e)
    inc = run.work / "shocks" / f"{exp}.inc"
    inc.parent.mkdir(exist_ok=True)
    inc.write_text(text)

    har = run.work / f"har_{exp}"
    har.mkdir(exist_ok=True)
    for name, src in [*((f, f) for f in HAR_FILES), ("default.prm", prm)]:
        (har / name).unlink(missing_ok=True)
        (har / name).symlink_to(run.nus333 / src)
    gdx_in = run.work / f"gdx_{exp}"
    write_nus333_gdx_bundle(har, gdx_in)

    lst, log = run.out / f"{exp}.lst", run.out / f"{exp}.log"
    util = "CD" if "cobbdouglas" in prm.lower() else "cde"
    proc = subprocess.run(
        [
            run.gams,
            "comp_shock.gms",
            "--baseName=nus333",
            f"--inDir={gdx_in}",
            f"--outDir={run.out}",
            "--savfFlag=capFlex",
            f"--utility={util}",
            f"--shockInc={inc}",
            f"--simName={exp}_capFlex",
            f"--iterLim={run.iterlim}",
            f"o={lst}",
            "lo=2",
            f"lf={log}",
        ],
        cwd=run.work,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not lst.exists():
        detail = (proc.stderr or proc.stdout).strip()[-500:]
        return FAILED, f"GAMS fallo (codigo {proc.returncode}, ver {log}) {detail}"
    status = sorted(
        {
            ln.strip()
            for ln in lst.read_text(errors="ignore").splitlines()
            if "MODEL STATUS" in ln
        }
    )
    outcome = SOLVED if all_optimal(status) and gdx_out.exists() else NOT_OPTIMAL
    return outcome, f"{prm} {util} {status}"


def main() -> int:
    from equilibria._local_refs import nus333_dir

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path, help="donde quedan <EXP>_capFlex.gdx")
    ap.add_argument("--only", default="", help="EXP separados por coma")
    ap.add_argument("--gams", default="gams", help="binario de GAMS (def. del PATH)")
    ap.add_argument(
        "--work-dir", type=Path, help="guardar aca la copia parcheada (def. temporal)"
    )
    ap.add_argument("--iterlim", type=int, default=1000, help="iteraciones de PATH")
    ap.add_argument(
        "--unemp-defl",
        choices=sorted(UNEMP_DEFLATORS),
        default="tornq",
        help="deflactor del salario real en el cierre de desempleo",
    )
    args = ap.parse_args()

    gams = shutil.which(args.gams)
    if gams is None:
        ap.error(f"no se encontro GAMS ({args.gams!r}): ponerlo en el PATH o --gams")
    known = all_exercises()
    todo = [e for e in args.only.split(",") if e] or known
    unknown = [e for e in todo if e not in known]
    if unknown:
        ap.error(f"ejercicios desconocidos: {', '.join(unknown)}")
    nus333 = nus333_dir()
    missing = missing_inputs(nus333, todo)
    if missing:
        ap.error(
            "faltan archivos del dataset nus333 (EQUILIBRIA_NUS333_DIR): "
            + ", ".join(str(m) for m in missing)
        )

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="burfisher_gams_") as tmp:
        work = (args.work_dir or Path(tmp)).resolve()
        work.mkdir(parents=True, exist_ok=True)
        shutil.copytree(GAMS_SRC, work, dirs_exist_ok=True)
        shutil.copy(COMP_SHOCK, work)
        patch_gams_sources(work, args.unemp_defl)

        run = GamsRun(
            out=out,
            work=work,
            nus333=nus333,
            gams=str(Path(gams).resolve()),
            iterlim=args.iterlim,
            deflator=args.unemp_defl,
        )
        by_outcome: dict[str, list[str]] = {SOLVED: [], NOT_OPTIMAL: [], FAILED: []}
        for exp in todo:
            outcome, line = run_exercise(exp, run)
            print(exp, line, flush=True)
            by_outcome[outcome].append(exp)
    for outcome in (NOT_OPTIMAL, FAILED):
        if by_outcome[outcome]:
            print(f"{outcome}: {', '.join(by_outcome[outcome])}", file=sys.stderr)
    return 1 if by_outcome[FAILED] else 2 if by_outcome[NOT_OPTIMAL] else 0


if __name__ == "__main__":
    sys.exit(main())
