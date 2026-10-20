"""make_levels_tab.py: gtapv7.tab con orig_level en las seis variables sin nivel.

GEMPACK solo guarda niveles (LEVB/LEVA en el .sl4) de las variables declaradas con
``orig_level``. En gtapv7.tab no lo tienen y, y, qinv, rore, pinv, pva, pe, que
salen como NOLV en ``sl4levels_*.har``. El parche les agrega el qualifier; no toca
ninguna ecuacion, asi que la solucion (los % de cambio) no cambia.

Los niveles que debe reportar GEMPACK estan fijados aparte, antes de este codigo, en
``tests/compare/oracles/levb_six_vars.py`` (contra la base del GDX de GAMS). Aqui se
prueba el texto del parche sobre un TAB minimo con las mismas declaraciones que
gtapv7.tab (GTAP 11a), copiadas tal cual; con ``GTAPV7_TAB`` apuntando a un
gtapv7.tab real, tambien sobre ese.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gtap" / "make_levels_tab.py"

# Declaraciones copiadas de gtapv7.tab (GTAP 11a): lineas 576, 940, 1063, 1776,
# 2334, 2475, y el Read de RORFLEX (2505-2509). qo ya tiene orig_level: no se toca.
MINI_TAB = """\
Variable (all,r,REG)
    pinv(r) # price index of investment goods #;
Variable (all,r,REG)
    y(r) # regional household income in region r #;
Variable (orig_level=VOS)(all,a,ACTS)(all,r,REG)
    qo(a,r) # output of activity a in region r #;
Variable (all,a,ACTS)(all,r,REG)
    pva(a,r) # firms' price of value added in act. a of region r #;
Variable (all,c,COMM)(all,r,REG)
    qia(c,r) # demand for investment commodity c in region r #;
Variable (all,r,REG)
    qinv(r) # demand for investment/capital goods in region r #;
Variable (all,e,ENDWMS)(all,r,REG)
    pe(e,r) # price of mobile and sluggish endowment e in region r #;
Variable (all,r,REG)
    rore(r) # expected net rate of return on capital stock in r #;
Variable (all,r,REG)
    rorc(r) # current net rate of return on capital stock in r #;
Coefficient (parameter)(all,r,REG)
    RORFLEX(r)
    # flexibility of expected net ROR on capital stock in r wrt investment #;
Read
    RORFLEX from file GTAPPARM header "RFLX";
!< comentario >!
Equation E_rore
# expected rate of return depends on the current return and investment #
(all,r,REG)
    rore(r) = rorc(r) - RORFLEX(r) * [ke(r) - kb(r)];
"""

MINI_STI = """\
bat         ! Run in batch.

gtapv7      ! name of TABLO input file
gtapv7      ! name of information file
c           ! Perform condensation
b
pva
E_pva
e           ! Exit condensation.
a           ! Proceed to automatic code generation.
wfp         ! Write a Fortran program.

gtapv7      ! name of Fortran program file
!
"""


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("make_levels_tab", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mlt() -> Any:
    return _load_module()


def test_las_seis_variables_quedan_con_orig_level(mlt: Any) -> None:
    out = mlt.patch_tab(MINI_TAB)
    assert "Variable (orig_level=1.0)(all,r,REG)\n    pinv(r)" in out
    assert "Variable (orig_level=INCOME)(all,r,REG)\n    y(r)" in out
    assert "Variable (orig_level=1.0)(all,a,ACTS)(all,r,REG)\n    pva(a,r)" in out
    assert "Variable (orig_level=REGINV)(all,r,REG)\n    qinv(r)" in out
    assert "Variable (orig_level=1.0)(all,e,ENDWMS)(all,r,REG)\n    pe(e,r)" in out
    assert "Variable (orig_level=RORELEV)(all,r,REG)\n    rore(r)" in out


def test_no_toca_otras_variables_ni_ecuaciones(mlt: Any) -> None:
    out = mlt.patch_tab(MINI_TAB)
    assert "Variable (orig_level=VOS)(all,a,ACTS)(all,r,REG)\n    qo(a,r)" in out
    assert "Variable (all,c,COMM)(all,r,REG)\n    qia(c,r)" in out
    assert "Variable (all,r,REG)\n    rorc(r)" in out
    assert "    rore(r) = rorc(r) - RORFLEX(r) * [ke(r) - kb(r)];" in out
    # quitando lo agregado, el TAB es el original byte a byte
    undone = out.replace(mlt.RORELEV_BLOCK, "")
    for qual in (
        "(orig_level=1.0)",
        "(orig_level=INCOME)",
        "(orig_level=REGINV)",
        "(orig_level=RORELEV)",
    ):
        undone = undone.replace(qual, "")
    assert undone == MINI_TAB


def test_rorelev_va_despues_del_read_de_rorflex(mlt: Any) -> None:
    out = mlt.patch_tab(MINI_TAB)
    read = 'RORFLEX from file GTAPPARM header "RFLX";\n'
    i = out.index(read) + len(read)
    assert out[i:].startswith(mlt.RORELEV_BLOCK)
    # la definicion del oraculo: RORC * (KB/KE)^RORFLEX, KE = VKB - VDEP + REGINV
    assert (
        "RORELEV(r) = {[GROSSCAP(r) - VDEP(r)] / VKB(r)}\n"
        "        * [VKB(r) / {VKB(r) - VDEP(r) + REGINV(r)}]^RORFLEX(r);"
    ) in mlt.RORELEV_BLOCK


def test_falta_una_declaracion_es_error(mlt: Any) -> None:
    tab = MINI_TAB.replace("    qinv(r) #", "    qinvX(r) #")
    with pytest.raises(mlt.TabPatchError, match="qinv"):
        mlt.patch_tab(tab)


def test_declaracion_que_ya_tiene_orig_level_es_error(mlt: Any) -> None:
    tab = MINI_TAB.replace(
        "Variable (all,r,REG)\n    y(r)",
        "Variable (orig_level=INCOME)(all,r,REG)\n    y(r)",
    )
    with pytest.raises(mlt.TabPatchError, match=r"\by\b"):
        mlt.patch_tab(tab)


def test_parchar_dos_veces_es_error(mlt: Any) -> None:
    with pytest.raises(mlt.TabPatchError):
        mlt.patch_tab(mlt.patch_tab(MINI_TAB))


def test_sin_el_read_de_rorflex_es_error(mlt: Any) -> None:
    tab = MINI_TAB.replace('header "RFLX"', 'header "XXXX"')
    with pytest.raises(mlt.TabPatchError, match="RORFLEX"):
        mlt.patch_tab(tab)


def test_sti_apunta_al_tab_nuevo_y_usa_gemsim(mlt: Any) -> None:
    out = mlt.patch_sti(MINI_STI, "gtapv7_levels")
    lines = out.splitlines()
    assert "gtapv7_levels ! name of TABLO input file" in lines
    assert "gtapv7_levels ! name of information file" in lines
    assert "pgs ! Prepare output for GEMSIM" in lines
    assert "gtapv7_levels ! name of GEMSIM Auxiliary files" in lines
    assert not any(ln.startswith("wfp") for ln in lines)
    # la condensacion no cambia
    assert "b\npva\nE_pva\n" in out


def test_sti_sin_las_lineas_esperadas_es_error(mlt: Any) -> None:
    with pytest.raises(mlt.TabPatchError, match="TABLO input file"):
        mlt.patch_sti(
            MINI_STI.replace("name of TABLO input file", "otra cosa"), "gtapv7_levels"
        )


def test_cli_escribe_tab_y_sti(mlt: Any, tmp_path: Path) -> None:
    (tmp_path / "gtapv7.tab").write_text(MINI_TAB)
    (tmp_path / "gtapv7.sti").write_text(MINI_STI)
    out = tmp_path / "out"
    rc = mlt.main(
        [
            str(tmp_path / "gtapv7.tab"),
            "--sti",
            str(tmp_path / "gtapv7.sti"),
            "--out-dir",
            str(out),
        ]
    )
    assert rc == 0
    assert (out / "gtapv7_levels.tab").read_bytes() == mlt.patch_tab(MINI_TAB).encode()
    assert (out / "gtapv7_levels.sti").read_bytes() == mlt.patch_sti(
        MINI_STI, "gtapv7_levels"
    ).encode()


def test_cli_conserva_crlf(mlt: Any, tmp_path: Path) -> None:
    """gtapv7.tab viene con CRLF: la salida tambien, para que el diff sea solo el parche."""
    crlf = lambda s: s.replace("\n", "\r\n").encode()  # noqa: E731
    (tmp_path / "gtapv7.tab").write_bytes(crlf(MINI_TAB))
    (tmp_path / "gtapv7.sti").write_bytes(crlf(MINI_STI))
    out = tmp_path / "out"
    mlt.main(
        [
            str(tmp_path / "gtapv7.tab"),
            "--sti",
            str(tmp_path / "gtapv7.sti"),
            "--out-dir",
            str(out),
        ]
    )
    assert (out / "gtapv7_levels.tab").read_bytes() == crlf(mlt.patch_tab(MINI_TAB))
    assert (out / "gtapv7_levels.sti").read_bytes() == crlf(
        mlt.patch_sti(MINI_STI, "gtapv7_levels")
    )


@pytest.mark.skipif(
    not os.environ.get("GTAPV7_TAB"), reason="GTAPV7_TAB no apunta a un gtapv7.tab real"
)
def test_tab_real(mlt: Any) -> None:
    tab = (
        Path(os.environ["GTAPV7_TAB"])
        .read_bytes()
        .decode("latin-1")
        .replace("\r\n", "\n")
    )
    out = mlt.patch_tab(tab)
    for var, level in mlt.LEVELS.items():
        assert f"(orig_level={level})" in out, var
    assert out.count(mlt.RORELEV_BLOCK) == 1
