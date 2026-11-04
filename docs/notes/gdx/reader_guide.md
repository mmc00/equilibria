# GDX reader - equilibria.babel.gdx.reader

A module that reads GDX (GAMS Data Exchange) files in pure Python, with no GAMS installation needed.

## Features

✅ **Full reading of GDX v7 files**
- Headers (version, platform, producer)
- Symbol table (sets, parameters, variables, equations, aliases)
- UEL (Unique Element List) - set elements
- Domains (domain sets)
- Parameter data (numeric values)
- Variable data (level, marginal, bounds, scale)
- Equation data (level, marginal)

✅ **No GAMS dependencies**
- Pure Python implementation
- Needs only standard libraries (struct, pathlib)

✅ **Support for multiple dimensions**
- 1D, 2D, 3D+ parameters
- Multidimensional sets
- Indexed variables and equations

✅ **GDX compression handling**
- Interpolation of values in arithmetic sequences
- Sparse data (efficient storage)

## Installation

```bash
pip install equilibria
```

Or from source:

```bash
git clone https://github.com/tu-usuario/equilibria.git
cd equilibria
pip install -e .
```

## Basic usage

### Read a whole GDX file

```python
from equilibria.babel.gdx.reader import read_gdx

# Leer archivo GDX
gdx_data = read_gdx("model.gdx")

# Acceder a información del header
print(f"Versión GDX: {gdx_data['header']['version']}")
print(f"Plataforma: {gdx_data['header']['platform']}")

# Listar símbolos
for sym in gdx_data['symbols']:
    print(f"{sym['name']}: {sym['type_name']} (dim={sym['dimension']})")

# Ver elementos únicos
print(f"Elementos: {gdx_data['elements']}")
```

### Filter symbols by type

```python
from equilibria.babel.gdx.reader import (
    read_gdx,
    get_sets,
    get_parameters,
    get_variables,
    get_equations
)

gdx_data = read_gdx("model.gdx")

# Obtener solo sets
sets = get_sets(gdx_data)
for s in sets:
    print(f"Set: {s['name']} - {s['description']}")

# Obtener solo parámetros
parameters = get_parameters(gdx_data)
for p in parameters:
    print(f"Parameter: {p['name']} (dim={p['dimension']}, records={p['records']})")

# Obtener variables
variables = get_variables(gdx_data)

# Obtener ecuaciones
equations = get_equations(gdx_data)
```

### Look up one symbol

```python
from equilibria.babel.gdx.reader import read_gdx, get_symbol

gdx_data = read_gdx("model.gdx")

# Buscar un símbolo por nombre
price = get_symbol(gdx_data, "price")
if price:
    print(f"Tipo: {price['type_name']}")
    print(f"Dimensión: {price['dimension']}")
    print(f"Records: {price['records']}")
    print(f"Descripción: {price['description']}")
```

### Read parameter values

```python
from equilibria.babel.gdx.reader import read_gdx, read_parameter_values

gdx_data = read_gdx("model.gdx")

# Leer valores de un parámetro 1D
prices = read_parameter_values(gdx_data, "price")
for key, value in prices.items():
    print(f"price{key} = {value}")
# Salida: price('agr',) = 1.5
#         price('mfg',) = 2.0
#         price('srv',) = 2.5

# Leer valores de un parámetro 2D (matriz)
sam = read_parameter_values(gdx_data, "sam")
for key, value in sam.items():
    print(f"sam{key} = {value}")
# Salida: sam('agr', 'food') = 100.0
#         sam('agr', 'goods') = 50.0
#         ...
```

### Read set elements

```python
from equilibria.babel.gdx.reader import read_gdx, read_set_elements

gdx_data = read_gdx("model.gdx")

# Leer elementos de un conjunto
industries = read_set_elements(gdx_data, "i")
print(industries)
# Salida: [('agr',), ('mfg',), ('srv',)]

# Para sets 2D
pairs = read_set_elements(gdx_data, "ij")
# Salida: [('agr', 'food'), ('mfg', 'goods'), ...]
```

### Read variables (solver results)

```python
from equilibria.babel.gdx.reader import read_gdx, read_variable_values

gdx_data = read_gdx("results.gdx")

# Leer valores de una variable
x_values = read_variable_values(gdx_data, "X")
for key, attrs in x_values.items():
    print(f"X{key}:")
    print(f"  level:    {attrs['level']}")      # Valor de la solución
    print(f"  marginal: {attrs['marginal']}")   # Costo reducido
    print(f"  lower:    {attrs['lower']}")      # Límite inferior
    print(f"  upper:    {attrs['upper']}")      # Límite superior
    print(f"  scale:    {attrs['scale']}")      # Factor de escala
```

### Read equations (dual values)

```python
from equilibria.babel.gdx.reader import read_gdx, read_equation_values

gdx_data = read_gdx("results.gdx")

# Leer valores de una ecuación
eq_values = read_equation_values(gdx_data, "balance")
for key, attrs in eq_values.items():
    print(f"balance{key}:")
    print(f"  level:    {attrs['level']}")      # Nivel de actividad
    print(f"  marginal: {attrs['marginal']}")   # Precio sombra (dual)
```

## Complete example

```python
from pathlib import Path
from equilibria.babel.gdx.reader import (
    read_gdx,
    get_sets,
    get_parameters,
    get_variables,
    read_parameter_values,
    read_variable_values
)

def analyze_gdx(filepath: str):
    """Analiza un archivo GDX y muestra su contenido."""
    
    # 1. Leer el archivo
    gdx_data = read_gdx(filepath)
    
    print("=" * 70)
    print(f"Archivo GDX: {filepath}")
    print("=" * 70)
    
    # 2. Información del header
    print("\nInformación del archivo:")
    print(f"  Versión: {gdx_data['header']['version']}")
    print(f"  Plataforma: {gdx_data['header']['platform']}")
    print(f"  Símbolos: {len(gdx_data['symbols'])}")
    print(f"  Elementos únicos: {len(gdx_data['elements'])}")
    
    # 3. Analizar sets
    print("\nSets:")
    sets = get_sets(gdx_data)
    for s in sets:
        print(f"  {s['name']:15s} dim={s['dimension']}, records={s['records']}")
    
    # 4. Analizar parámetros
    print("\nParámetros:")
    params = get_parameters(gdx_data)
    for p in params:
        print(f"  {p['name']:15s} dim={p['dimension']}, records={p['records']}")
        
        # Leer valores si es pequeño
        if p['records'] <= 20:
            try:
                values = read_parameter_values(gdx_data, p['name'])
                print(f"    Valores: {len(values)} leídos")
                for key, val in list(values.items())[:3]:
                    print(f"      {key} = {val}")
            except Exception as e:
                print(f"    Error: {e}")
    
    # 5. Analizar variables
    print("\nVariables:")
    variables = get_variables(gdx_data)
    for v in variables:
        print(f"  {v['name']:15s} dim={v['dimension']}, records={v['records']}")
        
        # Intentar leer valores
        try:
            values = read_variable_values(gdx_data, v['name'])
            if values:
                print(f"    Valores: {len(values)} leídos")
                sample = list(values.items())[0]
                key, attrs = sample
                print(f"    Ejemplo: {key} level={attrs['level']}")
        except Exception as e:
            print(f"    (No se pudieron leer valores)")
    
    print("\n" + "=" * 70)

# Uso
if __name__ == "__main__":
    analyze_gdx("model.gdx")
```

## Data structures

### The GDX object returned by `read_gdx()`

```python
{
    "filepath": str,                    # Ruta del archivo
    "header": {
        "version": int,                 # Versión GDX (típicamente 7)
        "endianness": str,              # "little" o "big"
        "platform": str,                # "macOS", "Windows", "Linux"
        "producer": str                 # Información del productor
    },
    "symbols": [                        # Lista de símbolos
        {
            "name": str,                # Nombre del símbolo
            "type": int,                # 0=set, 1=parameter, 2=variable, 3=equation, 4=alias
            "type_name": str,           # Nombre del tipo legible
            "type_flag": int,           # Flag binario del tipo (para debugging)
            "dimension": int,           # Número de dimensiones (0-20)
            "records": int,             # Número de registros
            "description": str          # Descripción del símbolo
        },
        ...
    ],
    "elements": [str, ...],            # UEL: Lista de elementos únicos
    "domains": [str, ...]              # Lista de dominios
}
```

### Parameter values

```python
# Para parámetros 1D
{
    ("agr",): 1.5,
    ("mfg",): 2.0,
    ("srv",): 2.5
}

# Para parámetros 2D (matrices)
{
    ("agr", "food"): 100.0,
    ("agr", "goods"): 50.0,
    ("mfg", "food"): 75.0,
    ("mfg", "goods"): 200.0
}
```

### Variable/equation values

```python
{
    ("agr",): {
        "level": 100.0,      # Valor de la solución
        "marginal": 0.0,     # Costo reducido (var) o precio sombra (eq)
        "lower": 0.0,        # Límite inferior
        "upper": float('inf'), # Límite superior
        "scale": 1.0         # Factor de escala
    },
    ...
}
```

## API Reference

### Main functions

#### `read_gdx(filepath: str | Path) -> dict`
Reads a whole GDX file and returns all its information.

#### `get_symbol(gdx_data: dict, name: str) -> dict | None`
Looks up a symbol by name.

#### `get_sets(gdx_data: dict) -> list[dict]`
Returns every set in the file.

#### `get_parameters(gdx_data: dict) -> list[dict]`
Returns every parameter in the file.

#### `get_variables(gdx_data: dict) -> list[dict]`
Returns every variable in the file.

#### `get_equations(gdx_data: dict) -> list[dict]`
Returns every equation in the file.

#### `read_parameter_values(gdx_data: dict, symbol_name: str) -> dict`
Reads the numeric values of a parameter.

#### `read_variable_values(gdx_data: dict, symbol_name: str) -> dict`
Reads the values (level, marginal, etc.) of a variable.

#### `read_equation_values(gdx_data: dict, symbol_name: str) -> dict`
Reads the values (level, marginal) of an equation.

#### `read_set_elements(gdx_data: dict, set_name: str) -> list[tuple]`
Reads the elements of a set.

### Low-level functions

#### `read_header_from_bytes(data: bytes) -> dict`
Reads the GDX header from bytes.

#### `read_symbol_table_from_bytes(data: bytes) -> list`
Reads the symbol table from bytes.

#### `read_uel_from_bytes(data: bytes) -> list`
Reads the unique element list from bytes.

#### `read_domains_from_bytes(data: bytes) -> list`
Reads the domains from bytes.

## Known limitations

1. **Variables and equations**: reading variable and equation values may be incomplete for GDX files with complex formats or advanced compression. The metadata (name, type, dimension) is read correctly.

2. **Data compression**: GDX files can use several compression schemes. Supported today:
   - Explicit values (no compression)
   - **Arithmetic sequences** (linear interpolation): `a, a+d, a+2d, a+3d, ...` ✅
   - Basic sparse data (zeros omitted) ✅
   - **Geometric sequences**: `a, a×r, a×r², a×r³, ...` ❌ (Not supported - they are wrongly interpolated as arithmetic)

3. **Geometric interpolation**: if a GDX parameter is compressed with a geometric progression, the rebuilt values will be **wrong**, because the current code assumes an arithmetic (linear) progression. This can affect parameters with exponential growth rates.

4. **Multidimensional sets**: reading the elements of 2D+ sets is simplified and may not capture every detail.

5. **GDX version**: optimized for GDX v7. Earlier versions may work but are not fully tested.

## Tests

The module comes with a full test suite:

```bash
# Ejecutar todos los tests
pytest tests/babel/gdx/

# Ejecutar solo tests del reader
pytest tests/babel/gdx/test_reader.py -v

# Ejecutar tests de funcionalidad extendida
pytest tests/babel/gdx/test_reader_extended.py -v
```

## Contributing

Contributions are welcome. Areas to improve:

- [ ] Better decoding of variables and equations
- [ ] Support more GDX compression schemes
- [ ] Full reading of multidimensional sets
- [ ] Support for GDX v6 and earlier
- [ ] Writing GDX files (writer)
- [ ] Data integrity validation

## License

See [`LICENSE`](../../../LICENSE).

## References

- [GAMS GDX Documentation](https://www.gams.com/latest/docs/UG_GDX.html)
- GDX format based on reverse engineering and the official GAMS documentation
