# Lectura de Parámetros Multidimensionales en GDX

## Resumen

El módulo `equilibria.babel.gdx.reader` ahora soporta la lectura de parámetros con 3 o más dimensiones desde archivos GDX. Esta funcionalidad es esencial para modelos CGE complejos que usan tablas multidimensionales.

## Capacidades

### Completamente Soportado
- ✅ Parámetros 3D sparse (ej: `PARAMETER p(i,j,k)` con valores dispersos)
- ✅ Parámetros 4D sparse 
- ✅ Parámetros 5D sparse
- ✅ Parámetros 5D dense (completo)
- ✅ Parámetros 6D+ sparse (arquitectura lista, no probado extensivamente)

### Soporte Parcial
- ⚠️  Parámetros densos 3D-4D con compresión GDX (parcial)
- ⚠️  Algunos patrones de compresión avanzada pueden requerir más trabajo

## Uso

```python
from equilibria.babel.gdx.reader import read_gdx, read_parameter_values

# Leer archivo GDX
data = read_gdx("model.gdx")

# Leer parámetro 3D
values_3d = read_parameter_values(data, "my_3d_param")
# Resultado: {('i1', 'j1', 'k1'): 111.0, ('i1', 'j2', 'k1'): 121.0, ...}

# Leer parámetro 4D
values_4d = read_parameter_values(data, "my_4d_param")
# Resultado: {('i1', 'j1', 'k1', 'm1'): 1111.0, ...}
```

## Formato Binario GDX para Parámetros 3D+

El formato GDX usa varios patrones de codificación para parámetros multidimensionales:

### Pattern 1: Tupla Completa
```
01 <dim1> 00 00 00 <dim2_int32> <dim3_int32> ... <dimN_int32> 0a <double>
```
- Marker `01`: indica inicio de nueva tupla con todas las coordenadas
- `<dim1>`: byte con índice UEL (1-based) para la primera dimensión
- `00 00 00`: padding
- `<dimX_int32>`: int32 little-endian con índice UEL (1-based) para cada dimensión restante
- `0a`: marker de double
- `<double>`: valor de 8 bytes (little-endian)

### Pattern 2: Actualización Parcial (dimensión 2+)
```
02 <new_dim2> 00 00 00 <dim3_int32> ... <dimN_int32> 0a <double>
```
- Mantiene `dim1` de la tupla anterior
- Actualiza `dim2` y todas las dimensiones siguientes

### Pattern 3: Actualización de Últimas Dimensiones
```
03 <dim3_int32> <dim4_int32> ... <dimN_int32> 0a <double>
```
- Mantiene `dim1` y `dim2` de la tupla anterior
- Actualiza solo las dimensiones 3 en adelante
- Solo aplica para parámetros 4D+

### Pattern 4: Actualización desde Dimensión 4
```
04 <dim4_int32> <dim5_int32> ... <dimN_int32> 0a <double>
```
- Mantiene `dim1`, `dim2`, y `dim3` de la tupla anterior
- Actualiza solo las dimensiones 4 en adelante
- Solo aplica para parámetros 5D+
- Patrón detectado en parámetros 5D sparse

### Pattern General N
```
0N <dimN+1_int32> <dimN+2_int32> ... <dimM_int32> 0a <double>
```
- El marker `0N` indica mantener las primeras N dimensiones
- Actualizar las dimensiones N+1 en adelante
- Permite compresión eficiente cuando cambian solo las últimas dimensiones

## Implementación Técnica

### Estructura de Datos
- Los índices UEL en los patrones son **1-based** (se convierten a 0-based internamente)
- La primera dimensión se codifica como byte, las demás como int32
- Los valores son doubles de 8 bytes (IEEE 754 little-endian)

### Algoritmo de Decodificación
1. Buscar marker `01` para tupla completa
2. Extraer índices UEL y convertir a 0-based
3. Mapear índices UEL a nombres de elementos
4. Buscar patterns `02` y `03` para actualizaciones parciales
5. Mantener estado de la tupla actual para patterns incrementales

## Tests

Los tests están en `tests/babel/gdx/test_multidim_parameters.py` y `tests/babel/gdx/test_5d_parameters.py`:

```bash
# Generar archivos de prueba (requiere GAMS)
cd tests/fixtures
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams generate_multidim_test.gms
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams generate_5d_test.gms

# Ejecutar tests de 3D y 4D
pytest tests/babel/gdx/test_multidim_parameters.py -v

# Ejecutar tests de 5D
pytest tests/babel/gdx/test_5d_parameters.py -v

# Ejecutar todos los tests multidimensionales
pytest tests/babel/gdx/test_*d_parameters.py -v
```

## Limitaciones Conocidas

1. **Parámetros Densos**: La compresión agresiva usada por GDX para parámetros densos no está completamente implementada. Algunos valores pueden no leerse correctamente.

2. **Patrones Adicionales**: Pueden existir otros patrones de codificación que aún no se han encontrado en los datos de prueba.

3. **Rendimiento**: Para parámetros muy grandes (>100k valores), la lectura puede ser lenta.

## Ejemplos de Uso en CGE

```python
# Leer matriz SAM 3D (actividades x productos x regiones)
sam_3d = read_parameter_values(gdx_data, "SAM")

# Leer tabla de input-output 4D
io_4d = read_parameter_values(gdx_data, "IO_TABLE")

# Leer matriz de comercio 5D (sector × producto × origen × destino × tiempo)
trade_5d = read_parameter_values(gdx_data, "TRADE")

# Acceder valores específicos
trade_value = sam_3d[('agr', 'food', 'region1')]
io_coefficient = io_4d[('sector1', 'product1', 'region1', 'year1')]
bilateral_trade = trade_5d[('mfg', 'goods', 'usa', 'mex', '2020')]

# Slicing multidimensional
# Extraer todos los flujos desde 'usa' en 2020
usa_exports_2020 = {
    (s, p, dest): v 
    for (s, p, orig, dest, year), v in trade_5d.items() 
    if orig == 'usa' and year == '2020'
}

# Agregación por región
total_by_region = {}
for (s, p, orig, dest, year), v in trade_5d.items():
    total_by_region[orig] = total_by_region.get(orig, 0) + v
```

Ver ejemplos completos en:
- `examples/multidim_examples.py` - Ejemplos 3D y 4D
- `examples/example_5d_usage.py` - Ejemplos avanzados 5D

## Desarrollo Futuro

- [ ] Implementar decodificación completa de parámetros densos
- [ ] Optimizar lectura para datasets grandes
- [ ] Agregar soporte para variables y ecuaciones multidimensionales
- [ ] Agregar validación de dominios

## Referencias

- [GDX File Format Documentation](https://www.gams.com/latest/docs/UG_GDX.html)
- [GAMS Data Exchange](https://www.gams.com/latest/docs/T_GDXMRW.html)
