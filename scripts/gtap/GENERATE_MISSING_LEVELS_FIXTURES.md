# Generar las fixtures de NIVELES GEMPACK que faltan (capFix de 3x3/3x4/5x5)

**Para correr en la máquina WINDOWS con RunGTAP/GEMPACK instalado.**

## Contexto

La comparación de docs contra GEMPACK se va a migrar de la métrica de **cambio %**
(engañosamente baja, 88–95%, por celdas de valor base ≈0) a la métrica de **niveles
absolutos de valor** (~99.4%, la económicamente correcta). Esa métrica lee los
`updated_<ds>_<closure>.har` de GEMPACK (la base de datos GTAP post-shock en niveles).

Ya existen los `updated_*_capfix.har` de **10x7 y 15x10**, y los capFlex (default, sin
sufijo) de todos. **Faltan los `_capfix` de 3x3, 3x4 y 5x5.** Este documento explica
cómo generarlos.

- **20x41 se OMITE**: GEMPACK no resuelve ese dataset (loge-of-negative en E_u para el
  Caribe) — es un límite del lado GEMPACK, no del nuestro.

## Requisitos en la máquina Windows

- RunGTAP / `gtapv7.exe` (por defecto el script busca `C:\runGTAP375\gtapv7.exe`).
- `sltoht.exe` (opcional, por defecto `C:\GP\sltoht.exe`) — solo para el sl4dump de
  %-cambios; NO es necesario para los niveles. Si no está, el script lo salta.
- El repo clonado y actualizado (`git pull`), en la rama que traiga este archivo.
- `uv` disponible (o ajustar a `python` directo).

## Comando a ejecutar

Desde la raíz del repo, generar SOLO los capFix (rordelta=0) de los tres datasets que
faltan:

```bat
uv run python scripts\gtap\run_gempack_matrix.py ^
    --datasets gtap7_3x3 gtap7_3x4 gtap7_5x5 ^
    --rordelta 0 ^
    --gtapv7 C:\runGTAP375\gtapv7.exe
```

(Ajusta `--gtapv7` si tu ruta difiere. Añade `--sltoht C:\ruta\sltoht.exe` solo si
también quieres regenerar los sl4dump de %-cambios; para niveles no hace falta.)

El script:
1. arma un run-folder aislado por dataset,
2. escribe el `.cmf` con la closure capFix (RORDELTA=0),
3. resuelve con `gtapv7.exe`,
4. copia `updated_<ds>_tm10_s8-16-32_capfix.har` a `tests/fixtures/gtap7_gempack/`.

## Salida esperada

Tres archivos nuevos en `tests/fixtures/gtap7_gempack/`:
- `updated_gtap7_3x3_tm10_s8-16-32_capfix.har`
- `updated_gtap7_3x4_tm10_s8-16-32_capfix.har`
- `updated_gtap7_5x5_tm10_s8-16-32_capfix.har`

El script imprime `OK` / `FAIL` por dataset y el tamaño del `updated=...B`. Un OK con
tamaño > 0 = fixture buena.

## Después (de vuelta en el Mac / repo principal)

Commitear los tres `.har` nuevos y avisar. Con eso la matriz de niveles queda completa
para los 5 datasets resolubles × ambos closures, y se puede migrar el gate/doc de %-chg
a niveles. (La verificación de niveles del lado Python ya está hecha: 3x3/3x4/5x5 capFlex,
10x7 ambos, 15x10 capFix — todos ~99.4–99.6%.)
