# Por que no hay registro de bloques

**Fecha:** 2026-09-24 · **Estado:** decidido, antes del primer release a PyPI

Se quitaron `BlockRegistry`, `get_registry` y `register_block` de
`blocks/base.py` y de los dos `__all__` (`equilibria/` y `equilibria/blocks/`).

## Que eran

Una tabla `nombre -> clase de bloque` con un singleton global y un decorador,
para poder pedir un bloque por su nombre en texto:

```python
bloque = get_registry().create("IncomeBlock", sigma=0.8)
```

## Por que se fueron

Su unico usuario era un ejemplo que lo demostraba a si mismo. Medido antes de
borrar:

- **33 subclases de `Block`** en `src/` (GTAP, gtap_logvalue, PEP, trade,
  institutions, demand, production, equilibrium). **Ninguna** llevaba
  `@register_block`.
- La unica excepcion estaba en `examples/`: `example_04_custom_blocks.py`
  decoraba su `SimpleDemand`, imprimia `registry.list_blocks()` y dedicaba un
  paso entero a `registry.create("SimpleDemand")`. Era un usuario REAL —el CI lo
  ejecuta como test— pero de un tipo particular: el ejemplo existia para enseñar
  el registro, no lo usaba para nada que necesitara. El bloque se instanciaba
  directo (`SimpleDemand(name=...)`) y el propio texto decia
  "Use @register_block decorator (optional)". Se reescribio en este mismo commit
  para componer por import directo, que es lo que hace el resto del repo.
- Fuera de ese ejemplo, `get_registry()` no se llamaba desde ningun sitio: las
  demas menciones vivas estaban dentro del propio codigo muerto
  (`register_block` llamandose a si mismo) o en ejemplos `>>>` de docstring.
- El repo necesito un registro **tres veces** en otros sitios y las tres escribio
  un `dict` de modulo sin tocar este: `_ADAPTER_REGISTRY`
  (`simulations/simulator.py`), `_MAPPING_RUNTIME_REGISTRY`
  (`simulations/runtimes.py`), `_REGISTRY` (`templates/gtap/shocks.py`).

Y su firma no modelaba el problema real. Los bloques no se crean con parametros
sueltos: `_mk_unit` (`templates/gtap/gtap_block_model.py`) los instancia con
`sets` + `params` y enhebra `residual_region` / `if_sub` / `savf_flag` segun los
campos que declare cada clase, y `_block_classes` los devuelve en **orden de
dependencia**. `create(name, **kwargs)` no expresa ni el orden ni ese paso de
contexto.

## El caso que si nos interesa: describir un modelo en YAML

La motivacion para conservarlo era que un economista pudiera definir su modelo
sin escribir Python. Se esbozo el fichero que querriamos poder escribir:

```yaml
modelo: gtap7
dataset: gtap7_10x7          # registry de datasets (ya existe)

closure: gtap_standard       # _closure_template_data (ya existe)
  # savf_flag: capFlex       # ...o sobrescribir campos sueltos
  # if_sub: false

region_residual: ROW
base_calibrada: true

shock:
  tipo: arancel              # _REGISTRY de templates/gtap/shocks.py (ya existe)
  destino: tm
  valor: +10%
  sobre: {region: USA, sector: VegFruit}

solver:
  motor: ipopt
  periodos: [base, check, shock]
```

**El boceto no tiene una lista de bloques, y ese es el hallazgo.** Un usuario de
GTAP no elige bloques: los 7 *son* el modelo GTAP, y quitar uno no da otro modelo
sino uno roto. Elige dataset, closure, shock y periodos — y las cuatro cosas ya
tienen su registro. Ninguna es el de bloques.

El camino YAML -> modelo es, casi entero, leer el fichero y llamar a
`build_block_model(params, sets, closure, residual_region, ...)` con piezas que ya
existen. Los bloques los pone el template, como ahora.

## Cuando habria que reabrir esto

Cuando un **tercero** publique bloques en su propio paquete (`equilibria_energia`
con sus `Block` propios) sin tocar este repo. Ese es el unico caso donde el
decorador `@register_block` es imprescindible, porque el bloque vive en codigo que
no controlamos.

Si llega ese dia, el registro se disena a partir de `_block_classes` y `_mk_unit`
en `gtap_block_model.py` — que ya son el registro real, escrito a mano y con el
orden de dependencia que el borrado no modelaba — y no recuperando esta version.
