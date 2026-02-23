# ieem to pep transformations (es)

Version en espanol. Explica las transformaciones IEEM->PEP de `src/equilibria/sam_tools/ieem_to_pep_transformations.py` con ejemplos minimos antes/despues.

## abstracciones centrales

Todos los pasos trabajan sobre instancias de `Sam` manejadas por `SamTransform`. La clase `Sam` garantiza que la matriz sea cuadrada, filas y columnas compartan las mismas cuentas, y normaliza las claves a tuplas (p.ej. `("AG","gvt")`). `SamTransform` registra metadatos del origen (ruta/formato) y expone la matriz editable usada en las transformaciones YAML. Esta base es la que soporta agregacion, balanceo y conversion de exportaciones.

Nota:
- `aggregate_mapping` queda como operacion generica (agnostica de formato) en `src/equilibria/sam_tools/ieem_raw_excel.py`.
- `scale_all`, `scale_slice`, `rebalance_ipfp` y `enforce_export_balance` siguen en el ejecutor (`src/equilibria/sam_tools/executor.py`).

## contexto

En PEP, la oferta de commodities suele entrar por actividades (`J`) y ciertos flujos fiscales deben caer en cuentas especificas (`AG.ti`).
Una SAM tipo IEEM puede traer flujos validos contablemente, pero en "puertas" que PEP no usa igual.
Estas transformaciones reubican esos flujos sin cambiar el total agregado.

## glosario de simbolos y notacion

- `A -> B`: flujo desde la fila `A` hacia la columna `B` en la SAM.
- `*`: comodin (todos los elementos de una categoria), por ejemplo `K.*`.
- `i`: placeholder de commodity (por ejemplo `agr`, `ser`, `food`).
- `map(i)`: mapeo commodity -> sector (por ejemplo `food -> ind`).

Categorias principales:

- `I`: commodities (bienes/servicios).
- `J`: actividades/sectores productivos.
- `K`: factores de capital.
- `L`: factores de trabajo.
- `AG`: cuentas institucionales y fiscales (hogares, gobierno, resto del mundo, impuestos).
- `MARG`: cuenta de margenes de comercio/transporte.
- `OTH`: otras cuentas macro (inversion, variacion de inventarios).

Elementos frecuentes:

- `K.cap`: capital.
- `K.land`: tierra.
- `L.usk`: trabajo no calificado.
- `L.sk`: trabajo calificado.
- `AG.gvt`: gobierno.
- `AG.row`: resto del mundo.
- `AG.ti`: impuestos indirectos sobre commodities.
- `AG.tx`: impuestos asociados a exportacion/otra ruta fiscal no usada aqui en columnas `I`.
- `AG.td`: impuestos directos.
- `AG.tm`: aranceles/impuestos de importacion.
- `MARG.MARG`: cuenta agregada de margenes.

## 1) move_k_to_ji

Que hace:
- Mueve pagos en columnas de commodities que vienen desde factores de capital (`K.* -> I.i`) hacia actividades (`J.map(i) -> I.i`).

Logica economica simple:
- En PEP, la oferta de un bien la hace un sector productivo (`J`), no el factor capital directamente.

Ejemplo:

Antes:
- `K.cap -> I.agr = 50`
- `J.agr -> I.agr = 120`

Despues:
- `K.cap -> I.agr = 0`
- `J.agr -> I.agr = 170`

## 2) move_l_to_ji

Que hace:
- Mueve pagos en columnas de commodities desde trabajo (`L.* -> I.i`) hacia actividades (`J.map(i) -> I.i`).

Logica economica simple:
- Igual idea que capital: trabajo no "ofrece" directamente commodities en el bloque de mercado PEP; lo hace la actividad.

Ejemplo:

Antes:
- `L.usk -> I.ser = 30`
- `J.ser -> I.ser = 200`

Despues:
- `L.usk -> I.ser = 0`
- `J.ser -> I.ser = 230`

## 3) move_margin_to_i_margin

Que hace:
- Toma `MARG.MARG -> I.i` y lo reubica en una fila de commodity margen, tipicamente `I.ser -> I.i`.

Logica economica simple:
- En vez de dejar margenes en una cuenta que PEP no usa para oferta efectiva, se llevan a un commodity (servicios de comercio/transporte) para que entren por canales de mercado compatibles.

Ejemplo:

Antes:
- `MARG.MARG -> I.food = 12`
- `I.ser -> I.food = 8`

Despues:
- `MARG.MARG -> I.food = 0`
- `I.ser -> I.food = 20`

## 4) move_tx_to_ti_on_i

Que hace:
- Para columnas `I.*`, mueve `AG.tx -> I.i` hacia `AG.ti -> I.i`.

Logica economica simple:
- En PEP, estos impuestos sobre commodities se esperan en la cuenta de impuestos indirectos (`ti`) para que las identidades fiscales cierren como estan escritas.

Ejemplo:

Antes:
- `AG.tx -> I.agr = 9`
- `AG.ti -> I.agr = 4`

Despues:
- `AG.tx -> I.agr = 0`
- `AG.ti -> I.agr = 13`

## 5) apply_pep_structural_moves (legacy/composite)

Que hace:
- Ejecuta en secuencia las cuatro transformaciones anteriores.

Cuando usarlo:
- Solo por compatibilidad con configuraciones antiguas.
- Para trazabilidad fina, conviene usar los cuatro pasos explicitos en YAML.

## 6) balance_ras

Que hace:
- Ejecuta un balanceo RAS sobre una SAM RAW para reducir brechas fila vs columna antes de normalizar cuentas.

Logica economica simple:
- Luego de agregar, pueden quedar inconsistencias pequenas.
- `balance_ras` distribuye factores de ajuste por filas/columnas para recuperar cierre contable.

Parametros clave:
- `ras_type`: `arithmetic` (default), `geometric`, `row`, `column`.
- `tol`, `max_iter`: tolerancia numerica y maximo de iteraciones.
- Clase de implementacion: `src/equilibria/sam_tools/balancing.py` (`RASBalancer`).

Ejemplo:

Antes:
- diferencia maxima fila-columna = `120`

Despues:
- diferencia maxima fila-columna = `0.000001` (aprox, segun tolerancia)

## 7) normalize_pep_accounts

Que hace:
- Convierte etiquetas RAW agregadas (`A-*`, `C-*`, `USK`, `GVT`, etc.) a la estructura canonica PEP (`J/I/L/K/AG/MARG/OTH`).

Logica economica simple:
- Las ecuaciones PEP esperan cuentas/puertas especificas.
- Este paso reubica los mismos flujos en esas puertas esperadas.

Ejemplo:

Antes:
- `RAW.A-AGR -> RAW.C-AGR = 170`

Despues:
- `J.agr -> I.agr = 170`

## 8) create_x_block

Que hace:
- Agrega una cuenta `X.i` (fila y columna) por cada commodity `I.i`.

Logica economica simple:
- PEP separa flujo domestico del commodity (`I`) y canal de exportacion (`X`).

Ejemplo:

Antes:
- no existe cuenta `X.agr`

Despues:
- existe `X.agr` y puede recibir flujos de exportacion

## 9) convert_exports_to_x

Que hace:
- Mueve exportaciones de `I.i -> AG.row` a `X.i -> AG.row`.
- Reasigna oferta de `J -> I` a `J -> X` de forma proporcional para mantener consistencia oferta-uso.

Logica economica simple:
- Restaura la puerta comercial esperada en PEP: exportaciones deben salir por `X`, no directo por `I`.

## 10) align_ti_to_gvt_j

Que hace:
- Mueve entradas fiscales `AG.ti -> J.*` hacia `AG.gvt -> J.*`.

Logica economica simple:
- Mantiene el ruteo impuesto/gobierno consistente con como calibracion e identidades PEP leen esos flujos.

## 11) scale_all y scale_slice (reescalar)

Donde viven:
- `src/equilibria/sam_tools/executor.py`

Que hacen:
- `scale_all`: multiplica toda la SAM por un factor.
- `scale_slice`: multiplica solo un bloque (`row` x `col`) por un factor.

Logica economica simple:
- Sirve para cambiar unidades (por ejemplo, pasar de millones a miles) o aplicar un ajuste parcial controlado antes de reequilibrar.

Ejemplo `scale_all`:

Antes:
- `J.agr -> I.agr = 100`

Despues con `factor=0.001`:
- `J.agr -> I.agr = 0.1`

Ejemplo `scale_slice`:

Antes:
- `AG.tm -> I.agr = 40`

Despues con `row=AG.tm`, `col=I.*`, `factor=1.1`:
- `AG.tm -> I.agr = 44`

## 12) rebalance_ipfp

Donde vive:
- `src/equilibria/sam_tools/executor.py` (llama utilidades en `pep_sam_compat.py`)

Que hace:
- Reequilibra la matriz para reducir diferencias fila vs columna respetando una mascara de soporte (`support`), usando IPFP.

Logica economica simple:
- Despues de mover flujos, puede quedar un pequeno descuadre por fila/columna.
- IPFP reparte ese ajuste sobre celdas permitidas sin romper la estructura objetivo.

Parametros clave:
- `target_mode`: como definir metas de filas/columnas (`geomean`, `average`, `original`).
- `support`: donde se permite ajustar (`pep_compat` o `full`).
- `epsilon`, `tol`, `max_iter`: estabilidad y tolerancia numerica.

Ejemplo:

Antes:
- Diferencia maxima fila-columna: `120`

Despues:
- Diferencia maxima fila-columna: `0.000001` (aprox)

## 13) enforce_export_balance

Donde vive:
- `src/equilibria/sam_tools/executor.py` (llama `enforce_export_value_balance`)

Que hace:
- Fuerza la identidad de valor de exportaciones para que el bloque de comercio quede consistente con la contabilidad esperada por PEP.

Logica economica simple:
- Si despues de transformaciones/rebalanceo queda una brecha pequena en exportaciones, este paso hace un ajuste final puntual para cerrar esa identidad.

Ejemplo:

Antes:
- valor exportado por oferta = `1000`
- valor exportado por demanda/precio = `997`

Despues:
- ambos lados = `1000` (dentro de `tol`)

## nota de conservacion

Las transformaciones estructurales reubican montos entre celdas, no crean ni destruyen valor. El total de la SAM se conserva; luego `rebalance_ipfp` y `enforce_export_balance` se usan para asegurar cierres contables y de comercio.
