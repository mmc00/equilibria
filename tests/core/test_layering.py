# tests/core/test_layering.py
#
# Candado de capas: las capas bajas no importan de las altas.
#
#   core/   -> primitivas, no conoce bloques ni templates
#   babel/  -> I/O, no conoce el modelo
#   blocks/ -> ecuaciones, no conoce al compositor (templates/)
#
# Cuando una constante del dominio hace falta en dos capas, va en la BAJA y la
# alta la re-exporta (asi se hizo con GTAP_*_AGENT: viven en
# blocks/gtap/agents.py y gtap_parameters las re-exporta).
import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "equilibria"

# (capa, prefijos prohibidos, excepciones conocidas con su motivo)
REGLAS = [
    ("core", ("equilibria.blocks", "equilibria.templates"), {}),
    ("babel", ("equilibria.blocks", "equilibria.templates"), {}),
    (
        "blocks",
        ("equilibria.templates",),
        {
            # F3: `FactorBlock.calibrate_base` construye y resuelve un modelo
            # entero (build_block_model + solve_block_model) desde dentro del
            # bloque. No es una constante mal colocada sino una
            # responsabilidad invertida: el arreglo es inyectar el builder, y
            # eso cambia la firma del camino base_calibrated=True (F3.5).
            "gtap/factor.py": "F3: calibrate_base orquesta al compositor",
        },
    ),
]


def _imports_de(path: pathlib.Path) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
        elif isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
    return out


@pytest.mark.parametrize("capa,prohibidos,excepciones", REGLAS)
def test_la_capa_no_importa_hacia_arriba(capa, prohibidos, excepciones):
    base = SRC / capa
    infracciones = []
    for f in sorted(base.rglob("*.py")):
        rel = f.relative_to(base).as_posix()
        malos = sorted(m for m in _imports_de(f) if m.startswith(prohibidos))
        if not malos:
            continue
        if rel in excepciones:
            continue
        infracciones.append(f"{capa}/{rel}: {', '.join(malos)}")

    assert not infracciones, f"la capa {capa}/ importa hacia arriba:\n  " + "\n  ".join(
        infracciones
    )


def test_las_excepciones_siguen_existiendo():
    """Si una excepcion ya no infringe nada, quitala de la lista.

    Evita que la lista se quede con permisos fantasma que tapen una regresion
    futura en ese mismo fichero.
    """
    obsoletas = []
    for capa, prohibidos, excepciones in REGLAS:
        for rel, motivo in excepciones.items():
            f = SRC / capa / rel
            if not f.exists():
                obsoletas.append(f"{capa}/{rel} ya no existe ({motivo})")
            elif not any(m.startswith(prohibidos) for m in _imports_de(f)):
                obsoletas.append(f"{capa}/{rel} ya no infringe nada ({motivo})")
    assert not obsoletas, "excepciones obsoletas:\n  " + "\n  ".join(obsoletas)


def test_las_etiquetas_de_agente_viven_en_la_capa_baja():
    """Y `gtap_parameters` las re-exporta con el mismo valor."""
    from equilibria.blocks.gtap import agents
    from equilibria.templates.gtap import gtap_parameters

    for nombre in (
        "GTAP_HOUSEHOLD_AGENT",
        "GTAP_GOVERNMENT_AGENT",
        "GTAP_INVESTMENT_AGENT",
        "GTAP_MARGIN_AGENT",
    ):
        assert getattr(agents, nombre) == getattr(gtap_parameters, nombre)

    assert agents.GTAP_FINAL_DEMAND_AGENTS == ("hhd", "gov", "inv", "tmg")
