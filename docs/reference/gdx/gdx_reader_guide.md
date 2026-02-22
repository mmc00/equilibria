# Lector GDX - equilibria.babel.gdx.reader

Módulo de lectura de archivos GDX (GAMS Data Exchange) en Python puro, sin necesidad de instalación de GAMS.

## Características

✅ **Lectura completa de archivos GDX v7**
- Headers (versión, plataforma, productor)
- Tabla de símbolos (sets, parameters, variables, equations, aliases)
- UEL (Unique Element List) - elementos de conjuntos
- Dominios (domain sets)
- Datos de parámetros (valores numéricos)
- Datos de variables (level, marginal, bounds, scale)
- Datos de ecuaciones (level, marginal)

✅ **Sin dependencias de GAMS**
- Implementación Python pura
- Solo requiere librerías estándar (struct, pathlib)

✅ **Soporte para múltiples dimensiones**
- Parámetros 1D, 2D, 3D+
- Sets multidimensionales
- Variables y ecuaciones con índices

✅ **Manejo de compresión GDX**
- Interpolación de valores en secuencias aritméticas
- Datos sparse (almacenamiento eficiente)

## Instalación

```bash
pip install equilibria
```

O desde el código fuente:

```bash
git clone https://github.com/tu-usuario/equilibria.git
cd equilibria
pip install -e .
```

## Uso Básico

### Leer un archivo GDX completo

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

### Filtrar símbolos por tipo

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

### Buscar un símbolo específico

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

### Leer valores de parámetros

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

### Leer elementos de sets

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

### Leer variables (resultados del solver)

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

### Leer ecuaciones (valores duales)

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

## Ejemplo Completo

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

## Estructura de Datos

### Objeto GDX retornado por `read_gdx()`

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

### Valores de parámetros

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

### Valores de variables/ecuaciones

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

### Funciones principales

#### `read_gdx(filepath: str | Path) -> dict`
Lee un archivo GDX completo y retorna toda la información.

#### `get_symbol(gdx_data: dict, name: str) -> dict | None`
Busca un símbolo por nombre.

#### `get_sets(gdx_data: dict) -> list[dict]`
Retorna todos los sets del archivo.

#### `get_parameters(gdx_data: dict) -> list[dict]`
Retorna todos los parámetros del archivo.

#### `get_variables(gdx_data: dict) -> list[dict]`
Retorna todas las variables del archivo.

#### `get_equations(gdx_data: dict) -> list[dict]`
Retorna todas las ecuaciones del archivo.

#### `read_parameter_values(gdx_data: dict, symbol_name: str) -> dict`
Lee los valores numéricos de un parámetro.

#### `read_variable_values(gdx_data: dict, symbol_name: str) -> dict`
Lee los valores (level, marginal, etc.) de una variable.

#### `read_equation_values(gdx_data: dict, symbol_name: str) -> dict`
Lee los valores (level, marginal) de una ecuación.

#### `read_set_elements(gdx_data: dict, set_name: str) -> list[tuple]`
Lee los elementos de un conjunto.

### Funciones de bajo nivel

#### `read_header_from_bytes(data: bytes) -> dict`
Lee el header del GDX desde bytes.

#### `read_symbol_table_from_bytes(data: bytes) -> list`
Lee la tabla de símbolos desde bytes.

#### `read_uel_from_bytes(data: bytes) -> list`
Lee la lista de elementos únicos desde bytes.

#### `read_domains_from_bytes(data: bytes) -> list`
Lee los dominios desde bytes.

## Limitaciones conocidas

1. **Variables y ecuaciones**: La lectura de valores de variables y ecuaciones puede estar incompleta para archivos GDX con formatos complejos o compresión avanzada. Los metadatos (nombre, tipo, dimensión) se leen correctamente.

2. **Compresión de datos**: Los archivos GDX pueden usar varios esquemas de compresión. Actualmente se soportan:
   - Valores explícitos (sin compresión)
   - **Secuencias aritméticas** (interpolación lineal): `a, a+d, a+2d, a+3d, ...` ✅
   - Datos sparse básicos (omisión de ceros) ✅
   - **Secuencias geométricas**: `a, a×r, a×r², a×r³, ...` ❌ (No soportado - se interpolan incorrectamente como aritméticas)

3. **Interpolación geométrica**: Si un parámetro GDX está comprimido usando una progresión geométrica, los valores reconstruidos serán **incorrectos** porque el código actual asume progresión aritmética (lineal). Esto puede afectar parámetros con tasas de crecimiento exponenciales.

4. **Sets multidimensionales**: La lectura de elementos de sets 2D+ está simplificada y puede no capturar todos los detalles.

5. **Versión GDX**: Optimizado para GDX v7. Versiones anteriores pueden funcionar pero no están completamente probadas.

## Tests

El módulo incluye una suite completa de tests:

```bash
# Ejecutar todos los tests
pytest tests/babel/gdx/

# Ejecutar solo tests del reader
pytest tests/babel/gdx/test_reader.py -v

# Ejecutar tests de funcionalidad extendida
pytest tests/babel/gdx/test_reader_extended.py -v
```

## Contribuir

Contribuciones son bienvenidas! Áreas de mejora:

- [ ] Mejorar decodificación de variables y ecuaciones
- [ ] Soportar más esquemas de compresión GDX
- [ ] Lectura completa de sets multidimensionales
- [ ] Soporte para GDX v6 y anteriores
- [ ] Escritura de archivos GDX (writer)
- [ ] Validación de integridad de datos

## Licencia

[Especificar licencia]

## Referencias

- [GAMS GDX Documentation](https://www.gams.com/latest/docs/UG_GDX.html)
- Formato GDX basado en ingeniería inversa y documentación oficial de GAMS
