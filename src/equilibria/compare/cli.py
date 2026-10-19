"""``equilibria compare A B --map MAPA``: la API de :func:`compare` desde la terminal.

Imprime el resumen por variable y global; ``--out`` escribe el JSON del reporte.
Sale con codigo distinto de 0 solo si se pide ``--fail-below PCT`` y el % de celdas
que pasan no lo alcanza. Ese % se mide sobre todas las celdas que el mapeo pide
comparar: parejas mas celdas sin pareja (una celda que falta nunca pasa). Las
celdas excluidas por el mapeo no se miden y no cuentan.

Fuentes por extension:

- ``.gdx``: niveles de GAMS (:func:`~equilibria.compare.from_gdx`), mas los
  parametros que pida el ``divide_by`` del mapeo.
- ``.har``: niveles de GEMPACK, ``pre`` y ``post`` (:func:`~equilibria.compare.from_har`).
- ``.json`` / ``.json.gz``: niveles como ``{var: [[[indices..., periodo], nivel], ...]}``
  (lo mismo que :func:`~equilibria.compare.from_dict`, en un archivo).

Codigos de salida: 0 bien; 1 no se alcanzo ``--fail-below``; 2 error de uso o de
datos (mapeo invalido, fuente ilegible, variable que falta).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import click

from equilibria.compare.errors import CompareError
from equilibria.compare.mapping import Mapping
from equilibria.compare.report import Report, compare
from equilibria.compare.solution import Solution
from equilibria.compare.sources import from_dict, from_gdx, from_har

FAIL_BELOW_EXIT = 1
ERROR_EXIT = 2


class _CompareFailed(click.ClickException):
    exit_code = ERROR_EXIT


def _read_json(path: Path) -> Any:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def load_source(
    path: Path,
    period_dim: int | None,
    period: str | None,
    *,
    variables: list[str],
    params: list[str] | None = None,
    own_period: bool = False,
) -> Solution:
    """Lee ``path`` segun su extension. ``variables`` y ``params`` son las variables
    y los parametros (del ``divide_by``) que pide el mapeo para ese lado. Un ``.har``
    trae sus propios periodos (``pre``/``post``): las opciones globales de periodo no
    se le aplican, y las de su lado (``own_period``) son error."""
    name = path.name.lower()
    if name.endswith((".sl4", ".har")):
        if params:
            raise CompareError(
                f"{path}: un .har de GEMPACK no trae parametros {params}"
            )
        if own_period:
            raise CompareError(
                f"{path}: un .har trae sus propios periodos (pre/post); no le pases "
                "--a-period*/--b-period*"
            )
        return from_har(path, variables=variables)
    if name.endswith(".gdx"):
        return from_gdx(
            path,
            period_dim=period_dim,
            period=period,
            variables=variables,
            params=params,
        )
    if name.endswith((".json", ".json.gz")):
        raw = _read_json(path)
        if not isinstance(raw, dict):
            raise CompareError(f"{path}: se esperaba {{var: [[indices, nivel], ...]}}")
        data: dict[str, dict[Any, Any]] = {}
        for var in variables:
            if var not in raw:
                raise CompareError(f"{path}: no hay una variable {var!r}")
            data[var] = {tuple(k): v for k, v in raw[var]}
        return from_dict(data, period_dim=period_dim, period=period, origin=str(path))
    raise CompareError(
        f"{path}: extension no reconocida (se aceptan .gdx, .har, .json, .json.gz)"
    )


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _sci(x: float | None) -> str:
    return "-" if x is None else f"{x:.3e}"


def _cell(variable: str | None, cell: Any) -> str:
    """``periodo:a.b`` (o ``variable:periodo:a.b``) de la celda del maximo."""
    if cell is None:
        return "-"
    text = f"{cell[0]}:{'.'.join(cell[1])}"
    return text if variable is None else f"{variable}:{text}"


def format_summary(report: Report) -> str:
    """El resumen como tabla: una fila por variable y la fila TOTAL. ``pasan%``
    cuenta las celdas sin pareja como que no pasan: es el % de ``--fail-below``."""
    th = report.thresholds
    head = (
        ["variable", "celdas", "pasan%"]
        + [f"<={t:g}" for t in th]
        + ["mediana", "maximo", "celda_max", "falta_a", "falta_b", "excl"]
    )

    def row(name: str, s: dict[str, Any], cell: str) -> list[str]:
        return (
            [name, str(s["cells"]), _pct(s["pass_pct_all"])]
            + [_pct(s["pct_within"][t]) for t in th]
            + [
                _sci(s["median"]),
                _sci(s["max"]),
                cell,
                str(s["missing_in_a"]),
                str(s["missing_in_b"]),
                str(s["excluded"]),
            ]
        )

    rows = [head]
    for name, s in report.summary.items():
        rows.append(row(name, s, _cell(None, s["max_cell"])))
    o = report.overall
    rows.append(row("TOTAL", o, _cell(o["max_variable"], o["max_cell"])))
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
        for r in rows
    ]

    lines.append("")
    lines.append(f"referencia: {report.reference}")
    lines.append(f"periodos comparados: {', '.join(report.periods_compared) or '-'}")
    if report.periods_only_in_a:
        lines.append(f"periodos solo en A: {', '.join(report.periods_only_in_a)}")
    if report.periods_only_in_b:
        lines.append(f"periodos solo en B: {', '.join(report.periods_only_in_b)}")
    lines.append(
        f"pasan%: sobre parejas + sin pareja (una celda que falta nunca pasa); "
        f"solo las parejas: {_pct(o['pass_pct'])}%"
    )
    for var, why in report.excluded.items():
        lines.append(f"variable excluida: {var}: {why}")
    for e in report.exclusions:
        where = ", ".join(f"{k}={v}" for k, v in e.where.items())
        lines.append(f"celdas excluidas: {e.variable} [{where}] {e.cells}: {e.why}")
    return "\n".join(lines)


@click.command("compare")
@click.argument("a", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("b", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--map",
    "map_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Mapeo YAML (variables, dims, tolerancias, exclusiones).",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Escribe aqui el JSON del reporte.",
)
@click.option(
    "--fail-below",
    type=click.FloatRange(0, 100),
    default=None,
    help="Sale con 1 si el % de celdas que pasan (contando sin pareja) es menor.",
)
@click.option(
    "--reference",
    type=click.Choice(["a", "b"]),
    default="b",
    show_default=True,
    help="Lado del denominador de d.",
)
@click.option(
    "--period-dim",
    type=int,
    default=None,
    help="Posicion de la dimension de periodo en A y B (p. ej. -1).",
)
@click.option(
    "--period",
    default=None,
    help="Nombre fijo del periodo de A y B (soluciones sin periodos).",
)
@click.option("--a-period-dim", type=int, default=None, help="--period-dim solo de A.")
@click.option("--b-period-dim", type=int, default=None, help="--period-dim solo de B.")
@click.option("--a-period", default=None, help="--period solo de A.")
@click.option("--b-period", default=None, help="--period solo de B.")
def compare_command(
    a: Path,
    b: Path,
    map_path: Path,
    out: Path | None,
    fail_below: float | None,
    reference: str,
    period_dim: int | None,
    period: str | None,
    a_period_dim: int | None,
    b_period_dim: int | None,
    a_period: str | None,
    b_period: str | None,
) -> None:
    """Compara las soluciones A y B, en niveles, celda por celda, segun MAPA.

    El periodo de cada fuente se declara, nunca se adivina: --period-dim o --period
    (para los dos lados) o --a-*/--b-* (por lado; un valor por lado tiene prioridad).
    """

    def side_rule(pd: int | None, p: str | None) -> tuple[int | None, str | None]:
        if pd is not None or p is not None:
            return pd, p
        return period_dim, period

    try:
        mapping = Mapping.from_yaml(map_path)
        sol_a = load_source(
            a,
            *side_rule(a_period_dim, a_period),
            own_period=a_period_dim is not None or a_period is not None,
            variables=sorted({v.a.var for v in mapping.variables}),
            params=mapping.params("a"),
        )
        sol_b = load_source(
            b,
            *side_rule(b_period_dim, b_period),
            own_period=b_period_dim is not None or b_period is not None,
            variables=sorted({v.b.var for v in mapping.variables}),
            params=mapping.params("b"),
        )
        report = compare(sol_a, sol_b, mapping, reference=reference)
    except CompareError as exc:  # MappingError es una CompareError
        raise _CompareFailed(str(exc)) from exc

    click.echo(format_summary(report))
    if out is not None:
        report.to_json(out)
        click.echo(f"reporte: {out}")
    if fail_below is not None:
        got = report.overall["pass_pct_all"]
        if got is None or got < fail_below:
            click.echo(
                f"FALLA: pasan {_pct(got)}% < --fail-below {fail_below:g}%", err=True
            )
            raise SystemExit(FAIL_BELOW_EXIT)
