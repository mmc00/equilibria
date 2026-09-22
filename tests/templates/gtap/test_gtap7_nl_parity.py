"""GTAP7 .nl structural parity gate.

For each dataset in tests/fixtures/gtap7/, regenerates the Python .nl and
compares its coefficients family-by-family against the GAMS reference .nl
fixture (generated once via NEOS and committed to the repo).

This test does NOT solve the model — it only builds the Pyomo NL and diffs
its coefficients. A change in any equation, parameter loading, or set
structure will surface here immediately.

Datasets covered (in git):
  gtap7_3x3, gtap7_3x4, gtap7_5x5, gtap7_10x7, gtap7_15x10

Large datasets (gtap7_20x41) are excluded from CI because the GAMS fixture
is too large for git; run manually with --dataset gtap7_20x41.

Run:
    uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -v
    uv run pytest tests/templates/gtap/test_gtap7_nl_parity.py -v -k gtap7_10x7

ALCANCE (medido, no supuesto).  Un .nl guarda cada fila en dos partes: los
coeficientes LINEALES del Jacobiano y el segmento NO LINEAL (potencias CES,
elasticidades).  ``diff_nl_models`` compara la primera.  Por tanto este gate
detecta:
  - coeficientes lineales distintos, filas/columnas que aparecen o desaparecen,
    bounds distintos, y familias enteras ausentes en un lado.
y NO detecta:
  - un cambio DENTRO de un termino no lineal.  MEDIDO (2026-09-21): alterar
    ``and_val * m.xp[r,a]`` a ``1.0001 * and_val * ...`` en eq_nd deja el gate
    en VERDE, porque ese factor se absorbe en el segmento no lineal (eq_nd emite
    solo 2 columnas lineales: nd y xp).
  - nada de una ecuacion que no exista en GAMS (p.ej. eq_pp_rai: 0 ocurrencias
    en el comp .gms), porque no hay contraparte con la que emparejar.
Para el nucleo no lineal el gate que manda es el de solve+niveles (mcp/nlp).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "tests/fixtures/gtap7"
DATASETS_DIR = ROOT / "datasets"

sys.path.insert(0, str(ROOT / "scripts/gtap"))

from coverage_matrix import nl_rows  # noqa: E402


def _available_datasets() -> list[str]:
    """Return dataset names that have both fixture NL files and HAR data.

    No longer feeds DATASETS — kept to document the on-disk contract and
    allow manual inspection.  The authoritative CI list comes from the
    coverage matrix (nl_rows()).
    """
    result = []
    for d in sorted(FIXTURES_DIR.iterdir()):
        if not d.is_dir():
            continue
        has_fixtures = (d / "gams_base.nl").exists() and (d / "gams_shock.nl").exists()
        has_data = (DATASETS_DIR / d.name / "basedata.har").exists()
        if has_fixtures and has_data:
            result.append(d.name)
    return result


# Datasets whose .nl gate runs in CI, per the coverage matrix.
# Restricted to gtap7_* names: nus333/9x10 are kind="gtap" ci_status="ci"
# but their parity is covered by dedicated tests, not this .nl fixture gate.
DATASETS = [
    r.dataset
    for r in nl_rows()
    if r.ci_status == "ci" and r.dataset.startswith("gtap7_")
]


# Familias donde Python y GAMS escriben la MISMA ecuacion de forma distinta.  El
# .nl compara coeficientes del Jacobiano, que dependen de COMO esta escrita la
# igualdad, asi que estas aparecen como diff sin que haya diferencia de modelo.
# Ambas se midieron celda a celda (2026-09-21) antes de entrar aqui:
#
# eq_pft — la igualdad esta escrita al REVES.  GAMS: `0 = xft - sum(xf/xscale)`
#   (xf con coeficiente negativo); Python: `xft == sum(xf/xscale)` (positivo).
#   MEDIDO en gtap7_10x7/base: 27 de 30 celdas cumplen py == -gams EXACTO; las 3
#   restantes son el mismo caso con ruido de redondeo.  Solo factores moviles.
#
# eq_pf — forma CROSS-MULTIPLICADA deliberada.  GAMS despeja xf
#   (`0 = xf - xscale*gf*xft*(pfy/pft)^omegaf`); Python escribe
#   `pf*(1-kappa)*denom == pft*xf` para NO dividir por denom=xscale*gf*xft, que
#   puede ser casi cero (xft[CAN,Land,shock]~9e-4) y producia una columna del
#   Jacobiano 3100x peor que la de GAMS — ver el comentario en
#   gtap_model_equations.py (caso ug_jacobian_collapse_root_cause).  Firma:
#   py=0.0 frente a gams=+-1.0 en las 21/21 celdas de gtap7_10x7/base.
#   ESCALA CON EL PERIODO, y esto tambien se midio en vez de extrapolarlo: en
#   `base` solo falla Land (omegaf={} -> solo Land cae en CET via etrae=-1);
#   en `check` fallan los 15 factores porque altertax fija omegaf=1.0 para TODOS
#   (apply_altertax_elasticities), asi que todos entran a la misma rama CET.
#   Misma causa, mas celdas: 3->36 en 3x3, 5->95 en 5x5, 21->252 en 10x7.
#
# Excluirlas NO las oculta: el resto de las 53 familias sigue exigiendo 0 diffs,
# asi que un diff NUEVO en cualquier otra familia falla el gate igual.  Si alguna
# de estas dos se reescribe para coincidir con GAMS, quitarla de aqui.
_WRITTEN_FORM_EQUIVALENT = frozenset({"eq_pf", "eq_pft"})


@pytest.mark.parametrize("dataset", DATASETS)
def test_gtap7_nl_parity(dataset: str, tmp_path: Path) -> None:
    """Python .nl coefficients match GAMS fixture for base/check/shock phases.

    The "check" phase (multi-period altertax CD step) is only diffed when a
    gams_check.nl fixture is present for the dataset.
    """
    from _nl_parser import parse_nl
    from _parity_datasets import DATASETS as DS_REGISTRY
    from nl_compare import build_python_nls, diff_bounds, diff_nl_models

    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    ds = DS_REGISTRY.get(dataset)
    if ds is None:
        pytest.skip(f"Dataset {dataset!r} not in registry")

    fixture_dir = FIXTURES_DIR / dataset
    if not (fixture_dir / "gams_base.nl").exists():
        pytest.skip(f"no .nl fixtures for {dataset}")
    har_dir = DATASETS_DIR / dataset
    closure_config = GTAPClosureConfig(if_sub=False)

    # The "check" phase (multi-period altertax CD step) is opt-in per dataset:
    # it is compared only when a gams_check.nl fixture exists, so datasets that
    # only carry base/shock fixtures keep passing unchanged.
    phases = ["base", "shock"]
    if (fixture_dir / "gams_check.nl").exists():
        phases.insert(1, "check")

    build_python_nls(
        phases=phases,
        out_dir=tmp_path,
        closure_config=closure_config,
        har_dir=har_dir,
    )

    tol_rel = 1e-4
    for phase in phases:
        py_nl = parse_nl(tmp_path / f"python_{phase}.nl")
        gams_nl = parse_nl(fixture_dir / f"gams_{phase}.nl")

        result = diff_nl_models(py_nl, gams_nl, tol_rel=tol_rel, py_period=phase)
        b_diffs, _, _ = diff_bounds(py_nl, gams_nl, tol_rel=tol_rel)

        # ANTI-VACUIDAD.  La comparacion empareja filas POR NOMBRE, y los nombres
        # de un .nl viven en los sidecars .col/.row, no dentro del archivo.  Si la
        # fixture GAMS no los trae, ambos mapas salen vacios, `failures` queda
        # vacia y el assert de abajo pasa SIN HABER COMPARADO NADA — verde
        # fantasma.  nl_compare.py:2247 ya trata este caso como error en la
        # herramienta de diagnostico; el gate tiene que hacer lo mismo.
        assert result["n_common_cons"] > 0, (
            f"[{dataset}/{phase}] COMPARACION VACUA: 0 filas con nombre en comun "
            f"entre el .nl de Python y la fixture GAMS. Casi siempre es que a "
            f"{fixture_dir / f'gams_{phase}.nl'} le faltan los sidecars .col/.row, "
            f"asi que sus filas se parsean anonimas. Un '0 diffs' aqui no "
            f"significa 'la algebra cuadra'; significa que no se comparo nada."
        )

        failures = [
            f"{fam}: {st['n_diff']} diffs (max_rel={st['max_rel']:.2e})"
            for fam, st in result["family_stats"].items()
            if st.get("n_diff", 0) > 0
            and not st.get("_structural_fp")
            and fam not in _WRITTEN_FORM_EQUIVALENT
        ]
        assert not failures, (
            f"[{dataset}/{phase}] Coefficient diffs vs GAMS fixture:\n"
            + "\n".join(f"  {f}" for f in failures)
        )
        assert len(b_diffs) == 0, (
            f"[{dataset}/{phase}] {len(b_diffs)} variables with different bounds"
        )
