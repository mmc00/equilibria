"""Lo comun de los tests nus333 (Burfisher 3e): datos, cierre, % de cambio y oraculos.

``nus333_params`` hace SKIP si falta el dataset (o el ``.prm``): los tests son
LOCAL-only.
"""

from __future__ import annotations

import gzip
import importlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest

if TYPE_CHECKING:
    from equilibria.templates.gtap import GTAPParameters
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

ROOT = Path(__file__).resolve().parents[3]


def nus333_params(prm: str = "default.prm") -> GTAPParameters:
    """Parametros nus333 con el ``.prm`` del ejercicio; SKIP si falta algo."""
    from equilibria._local_refs import nus333_dir
    from equilibria.templates.gtap import GTAPParameters

    har = nus333_dir()
    if not (har / "basedata.har").exists() or not (har / prm).exists():
        pytest.skip(f"nus333 o {prm} no disponible en {har}")
    p = GTAPParameters()
    p.load_from_har(
        basedata_path=har / "basedata.har",
        sets_path=har / "sets.har",
        default_path=har / prm,
        baserate_path=har / "baserate.har",
    )
    return p


def closure(
    savf_flag: Literal["capFix", "capFlex"] = "capFlex",
) -> GTAPClosureConfig:
    """El cierre MCP de los tests nus333: CAPITAL sluggish, dotaciones, impuestos y
    tecnologia libres, numerario ``pnum``.

    - ``"capFlex"`` (por defecto): el de los ejercicios de Burfisher.
    - ``"capFix"``: el de los tests que solo arman el modelo sin resolver el
      ejercicio (``conftest.nus333_mp_model``, ``test_shock_rejections``); es el
      ``savf_flag`` por defecto de ``GTAPClosureConfig``.

    No es ``scripts/gtap/_parity_datasets.nus333_closure()``: ese es el cierre fiel
    a GAMS de la cascada de paridad (``gams_factor_pairing``, ``rmuv``/``imuv``)."""
    from equilibria.templates.gtap.gtap_contract import GTAPClosureConfig

    return GTAPClosureConfig(
        name="base",
        closure_type="MCP",
        capital_mobility="sluggish",
        fix_endowments=False,
        fix_taxes=False,
        fix_technology=False,
        if_sub=False,
        savf_flag=savf_flag,
        numeraire="pnum",
    )


def pct(m: Any, var: str, key: tuple, num: str = "shock", den: str = "check") -> float:
    """% de cambio de ``var[key]`` entre los periodos ``den`` y ``num``."""
    from pyomo.environ import value

    comp = getattr(m, var)
    return 100.0 * (
        float(value(comp[(*key, num)])) / float(value(comp[(*key, den)])) - 1.0
    )


def gams_levels(fixture: Path, exp: str) -> dict[str, dict[tuple, float]]:
    """Niveles GAMS de ``exp`` guardados en un fixture ``.json.gz``
    (``{exp: {var: [[clave, nivel], ...]}}``, claves de ``_diff_core.gams_levels``)."""
    raw = json.loads(gzip.decompress(fixture.read_bytes()))[exp]
    return {vn: {tuple(k): v for k, v in cells} for vn, cells in raw.items()}


def run_burfisher() -> Any:
    """``scripts/gtap/run_burfisher.py`` (no es un paquete: se carga por ruta)."""
    path = str(ROOT / "scripts" / "gtap")
    if path not in sys.path:
        sys.path.insert(0, path)
    return cast(Any, importlib.import_module("run_burfisher"))
