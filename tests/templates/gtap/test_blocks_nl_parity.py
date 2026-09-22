"""Gate estructural del .nl para el camino de BLOQUES.

El hueco que tapa: los cuatro gates de bloques (mcp, nlp, gempack, logvalue)
RESUELVEN y comparan niveles.  Un modelo estructuralmente distinto que converja
al mismo punto los pasa todos en verde.  Este gate compara la FORMA —los
coeficientes del .nl emitido, sin resolver— contra la misma fixture de GAMS que
usa ``test_gtap7_nl_parity`` para el monolito.

Es el unico gate de bloques que no necesita solver ni GDX, igual que su gemelo
del monolito.

Alcance: fase ``base``.  ``build_block_single_period`` no acepta todavia
``is_counterfactual``/``t0_snapshot``, que es lo que check/shock necesitan para
referenciar los niveles del periodo base; ampliarlo es trabajo aparte.

Run:
    uv run pytest tests/templates/gtap/test_blocks_nl_parity.py -v
    uv run pytest tests/templates/gtap/test_blocks_nl_parity.py -v -k gtap7_3x3

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gtap7_nl_parity import _WRITTEN_FORM_EQUIVALENT  # noqa: E402

# Misma lista que el gate del monolito: la matriz de cobertura es la fuente unica.
DATASETS = [
    r.dataset
    for r in nl_rows()
    if r.ci_status == "ci" and r.dataset.startswith("gtap7_")
]


@pytest.mark.parametrize("dataset", DATASETS)
def test_blocks_nl_parity_base(dataset: str, tmp_path: Path) -> None:
    """Los coeficientes del .nl de BLOQUES cuadran con la fixture GAMS (base)."""
    from _nl_parser import parse_nl
    from nl_compare import diff_bounds, diff_nl_models

    from equilibria.templates.gtap.gtap_block_model import build_block_single_period
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig
    from equilibria.templates.gtap.gtap_parameters import GTAPParameters
    from equilibria.templates.gtap.gtap_solver import GTAPSolver

    fixture_dir = FIXTURES_DIR / dataset
    gams_nl_path = fixture_dir / "gams_base.nl"
    if not gams_nl_path.exists():
        pytest.skip(f"no hay fixture .nl para {dataset}")
    har_dir = DATASETS_DIR / dataset
    if not (har_dir / "basedata.har").exists():
        pytest.skip(f"no hay datos HAR para {dataset}")

    closure_config = GTAPClosureConfig(if_sub=False)

    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har_dir / "basedata.har",
        sets_path=har_dir / "sets.har",
        default_path=har_dir / "default.prm",
        baserate_path=har_dir / "baserate.har",
    )

    residual_region = list(p.sets.r)[-1]
    m = build_block_single_period(
        p, p.sets, closure=closure_config, residual_region=residual_region
    )

    # El mismo pipeline de cierre + parches que aplica el gate del monolito, para
    # que la comparacion mida la FORMA del modelo y no una diferencia de closure.
    solver_helper = GTAPSolver(m, closure=closure_config, solver_name="path", params=p)
    solver_helper.apply_closure(closure_config)
    solver_helper.apply_conditional_fixing()

    from _closure_patches import apply_squareness_patches

    apply_squareness_patches(m, p, label="blocks-nl-write-base")
    solver_helper.apply_aggressive_fixing_for_mcp()

    out_path = tmp_path / "blocks_base.nl"
    m.write(str(out_path), format="nl", io_options={"symbolic_solver_labels": True})

    tol_rel = 1e-4
    py_nl = parse_nl(out_path)
    gams_nl = parse_nl(gams_nl_path)

    result = diff_nl_models(py_nl, gams_nl, tol_rel=tol_rel, py_period="base")
    b_diffs, _, _ = diff_bounds(py_nl, gams_nl, tol_rel=tol_rel)

    # ANTI-VACUIDAD: ver la nota gemela en test_gtap7_nl_parity.  Sin los
    # sidecars .col/.row de la fixture GAMS no hay nombres que emparejar, el
    # diff sale vacio y el assert pasaria sin comparar nada.  MEDIDO: antes de
    # este guard el test salia VERDE comparando 0 filas — asi se descubrio que
    # test_gtap7_nl_parity, el unico gate GTAP en CI, llevaba haciendo lo mismo.
    assert result["n_common_cons"] > 0, (
        f"[{dataset}/base] COMPARACION VACUA: 0 filas con nombre en comun entre "
        f"el .nl de bloques y la fixture GAMS. Casi siempre es que a "
        f"{gams_nl_path} le faltan los sidecars .col/.row. Un '0 diffs' aqui no "
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
        f"[{dataset}/base] diffs de coeficientes vs fixture GAMS:\n"
        + "\n".join(f"  {f}" for f in failures)
    )
    assert len(b_diffs) == 0, (
        f"[{dataset}/base] {len(b_diffs)} variables con bounds distintos"
    )
