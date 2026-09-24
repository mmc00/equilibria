# tests/blocks/test_sin_registro_de_bloques.py
#
# Candado de la decision documentada en
# docs/architecture/registro_de_bloques.md: no hay registro de bloques.
#
# No enumera evasiones (un `getattr` dinamico se salta cualquier lista de
# nombres); afirma el INVARIANTE POSITIVO: los bloques se componen por import
# directo, asi que ningun nombre del registro vive en la superficie publica.
import equilibria
import equilibria.blocks
import equilibria.blocks.base

NOMBRES = ("BlockRegistry", "get_registry", "register_block")


def test_el_registro_no_esta_en_la_superficie_publica():
    """Si vuelve, que vuelva con un usuario y con el ADR actualizado."""
    for mod in (equilibria, equilibria.blocks, equilibria.blocks.base):
        presentes = [n for n in NOMBRES if hasattr(mod, n)]
        assert not presentes, (
            f"{mod.__name__} reexpone {presentes}; ver "
            "docs/architecture/registro_de_bloques.md"
        )
        exportados = [n for n in NOMBRES if n in getattr(mod, "__all__", ())]
        assert not exportados, f"{mod.__name__}.__all__ exporta {exportados}"


def test_los_bloques_se_componen_por_import_directo():
    """El invariante que sustituye al registro.

    `_block_classes` resuelve las 7 clases GTAP en orden de dependencia sin
    consultar ninguna tabla global: ese es el mecanismo real de composicion.
    """
    from equilibria.templates.gtap.gtap_block_model import _block_classes

    clases = _block_classes()
    assert len(clases) == 7, f"esperadas 7 clases GTAP, hay {len(clases)}"
    assert clases[-1].__name__ == "ClosureBlock", (
        "closure va al final: depende de todo lo demas"
    )
    for c in clases:
        assert isinstance(c, type), f"{c!r} no es una clase"
