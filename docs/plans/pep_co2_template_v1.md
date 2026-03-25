# PEP CO2 Template v1

## Objetivo

Agregar un template `pep_co2` basado en `pep` para simular politica climatica con un vector externo por sector, sin duplicar el modelo PEP completo ni ampliar el vector del solver mas de lo necesario.

## Corte v1

- Solo `CO2`
- Intensidad exogena por sector `J`
- Tasa base de impuesto por sector `tco2b[j]`
- Escalador global `tco2scal`
- Misma calibracion y mismo closure base de `pep`
- Sin AFOLU, sin `ghg x fuente x actividad`, sin metas endogenas de emisiones

## Contrato de datos

El runtime acepta:

- `co2_intensity`: mapping `sector -> intensidad`
- `tco2b`: mapping `sector -> tasa base`
- `tco2scal`: escalar global
- `co2_data`: CSV/XLSX con columnas:
  - `sector`
  - `co2_intensity`
  - `tco2b` opcional
  - `tco2scal` opcional, constante en todas las filas

Las etiquetas de sector se validan contra `J`. Los sectores faltantes se rellenan con `0.0`. Los sectores desconocidos fallan duro.

## Formulacion

La idea de v1 es tratar el carbono como un impuesto especifico por unidad de actividad, no como un bloque nuevo de variables endogenas.

Definiciones:

```text
carbon_unit_tax(j) = co2_intensity(j) * tco2b(j) * tco2scal * PIXCON
co2_emissions(j)   = co2_intensity(j) * XST(j)
co2_tax(j)         = carbon_unit_tax(j) * XST(j)
```

Overrides sobre `pep`:

- `EQ66`

```text
PT(j) = (1 + ttip(j)) * PP(j) + carbon_unit_tax(j)
```

- `EQ39`

```text
TIP(j) = [ttip(j) * PP(j) + carbon_unit_tax(j)] * XST(j)
```

Con esto:

- `TIPT` ya incluye la recaudacion de carbono
- `TPRODN` ya incluye la recaudacion de carbono
- `YG` ya la absorbe por el canal normal de impuestos a la produccion

No se agrega una variable endogena `YTAXCO2` al sistema cuadrado en v1. Esa recaudacion se reporta como metrica derivada.

## API publica

Modelo:

```python
from equilibria.simulations import PepCO2Simulator

sim = PepCO2Simulator(
    sam_file="SAM.gdx",
    val_par_file="VAL_PAR.xlsx",
    co2_data="co2_by_sector.xlsx",
).fit()
```

Shocks expuestos:

- `tco2b[j]`
- `tco2scal`
- `co2_intensity[j]`
- todos los shocks ya existentes de `pep`

Ejemplos:

```python
sim.run_carbon_tax_scale(multiplier=2.0)
sim.run_shock(var="tco2b", index="agr", multiplier=1.5, name="agr_carbon_tax")
sim.run_shock(var="co2_intensity", index="ind", multiplier=0.9, name="cleaner_industry")
```

## Criterios de validacion

- Si `co2_intensity == 0` o `tco2b == 0`, `pep_co2` debe colapsar a `pep`
- `EQ39` y `EQ66` deben reflejar el wedge de carbono
- `available_models()` debe incluir `pep_co2`
- El simulador debe aceptar shocks sectoriales y escala global

## No objetivos de v1

- Compatibilidad completa con el bloque ambiental multidimensional de IEEM
- Emisiones por insumo/factor/hogar/gobierno
- Endogeneizar `tco2scal` para pegar una meta de recaudacion
- Calibrar `tco2b` desde una cuenta `taxco2` en la SAM
