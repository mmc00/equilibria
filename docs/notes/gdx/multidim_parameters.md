# Reading multidimensional parameters from GDX

## Summary

The `equilibria.babel.gdx.reader` module now reads parameters with 3 or more dimensions from GDX files. This is essential for complex CGE models that use multidimensional tables.

## Capabilities

### Fully supported
- ✅ Sparse 3D parameters (e.g. `PARAMETER p(i,j,k)` with scattered values)
- ✅ Sparse 4D parameters 
- ✅ Sparse 5D parameters
- ✅ Dense 5D parameters (complete)
- ✅ Sparse 6D+ parameters (architecture ready, not tested extensively)

### Partial support
- ⚠️  Dense 3D-4D parameters with GDX compression (partial)
- ⚠️  Some advanced compression patterns may need more work

## Usage

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

## GDX binary format for 3D+ parameters

The GDX format uses several encoding patterns for multidimensional parameters:

### Pattern 1: full tuple
```
01 <dim1> 00 00 00 <dim2_int32> <dim3_int32> ... <dimN_int32> 0a <double>
```
- Marker `01`: starts a new tuple with all its coordinates
- `<dim1>`: a byte with the UEL index (1-based) of the first dimension
- `00 00 00`: padding
- `<dimX_int32>`: a little-endian int32 with the UEL index (1-based) of each remaining dimension
- `0a`: double marker
- `<double>`: an 8-byte value (little-endian)

### Pattern 2: partial update (dimension 2+)
```
02 <new_dim2> 00 00 00 <dim3_int32> ... <dimN_int32> 0a <double>
```
- Keeps `dim1` from the previous tuple
- Updates `dim2` and every following dimension

### Pattern 3: update of the last dimensions
```
03 <dim3_int32> <dim4_int32> ... <dimN_int32> 0a <double>
```
- Keeps `dim1` and `dim2` from the previous tuple
- Updates only dimensions 3 onwards
- Applies only to 4D+ parameters

### Pattern 4: update from dimension 4
```
04 <dim4_int32> <dim5_int32> ... <dimN_int32> 0a <double>
```
- Keeps `dim1`, `dim2` and `dim3` from the previous tuple
- Updates only dimensions 4 onwards
- Applies only to 5D+ parameters
- Pattern found in sparse 5D parameters

### General pattern N
```
0N <dimN+1_int32> <dimN+2_int32> ... <dimM_int32> 0a <double>
```
- Marker `0N` means keep the first N dimensions
- Update dimensions N+1 onwards
- Compresses well when only the last dimensions change

## Technical implementation

### Data structures
- The UEL indices in the patterns are **1-based** (converted to 0-based internally)
- The first dimension is encoded as a byte, the others as int32
- Values are 8-byte doubles (IEEE 754 little-endian)

### Decoding algorithm
1. Look for marker `01` for a full tuple
2. Extract the UEL indices and convert them to 0-based
3. Map UEL indices to element names
4. Look for patterns `02` and `03` for partial updates
5. Keep the state of the current tuple for incremental patterns

## Tests

The tests are in `tests/babel/gdx/test_multidim_parameters.py` and `tests/babel/gdx/test_5d_parameters.py`:

```bash
# Generar archivos de prueba (requiere GAMS)
cd data/gdx_tests/specimens/<generator>   # e.g. 6d_test
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams generate_multidim_test.gms
/Library/Frameworks/GAMS.framework/Versions/48/Resources/gams generate_5d_test.gms

# Ejecutar tests de 3D y 4D
pytest tests/babel/gdx/test_multidim_parameters.py -v

# Ejecutar tests de 5D
pytest tests/babel/gdx/test_5d_parameters.py -v

# Ejecutar todos los tests multidimensionales
pytest tests/babel/gdx/test_*d_parameters.py -v
```

## Known limitations

1. **Dense parameters**: the aggressive compression GDX uses for dense parameters is not fully implemented. Some values may not be read correctly.

2. **Further patterns**: there may be other encoding patterns not yet found in the test data.

3. **Performance**: for very large parameters (>100k values), reading can be slow.

## Usage examples in CGE

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

Full examples in:
- `examples/gdx/multidim_examples.py` - 3D and 4D examples
- `examples/gdx/example_5d_usage.py` - advanced 5D examples

## Future work

- [ ] Full decoding of dense parameters
- [ ] Faster reading for large datasets
- [ ] Support for multidimensional variables and equations
- [ ] Domain validation

## References

- [GDX File Format Documentation](https://www.gams.com/latest/docs/UG_GDX.html)
- [GAMS Data Exchange](https://www.gams.com/latest/docs/T_GDXMRW.html)
