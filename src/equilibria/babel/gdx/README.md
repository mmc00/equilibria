# equilibria.babel.gdx

Módulo para lectura y escritura de archivos GDX (GAMS Data Exchange) en Python puro.

## 📦 Componentes

### `reader.py` - Lector GDX
Lectura completa de archivos GDX sin necesidad de GAMS instalado.

**Características:**
- ✅ Headers (versión, plataforma, productor)
- ✅ Símbolos (sets, parameters, variables, equations, aliases)
- ✅ UEL (Unique Element List)
- ✅ Dominios
- ✅ Valores de parámetros (1D, 2D, 3D+)
- ✅ Valores de variables (level, marginal, bounds, scale)
- ✅ Valores de ecuaciones
- ✅ Soporte para compresión GDX
- ✅ Datos sparse

### `symbols.py` - Modelos de datos
Modelos Pydantic para representar estructuras GAMS:
- `Set` - Conjuntos
- `Parameter` - Parámetros
- `Variable` - Variables
- `Equation` - Ecuaciones

### `utils.py` - Utilidades
Constantes y funciones auxiliares para parsing GDX.

### `writer.py` - Escritor GDX (En desarrollo)
Escritura de archivos GDX desde Python.

## 🚀 Inicio Rápido

```python
from equilibria.babel.gdx.reader import read_gdx, get_parameters, read_parameter_values

# Leer archivo GDX
gdx_data = read_gdx("model.gdx")

# Ver información básica
print(f"Versión: {gdx_data['header']['version']}")
print(f"Símbolos: {len(gdx_data['symbols'])}")

# Obtener parámetros
params = get_parameters(gdx_data)
for p in params:
    print(f"{p['name']}: dim={p['dimension']}, records={p['records']}")

# Leer valores de un parámetro
prices = read_parameter_values(gdx_data, "price")
for key, value in prices.items():
    print(f"price{key} = {value}")
```

## 📖 Documentación

- [Guía completa del Reader](../../../../docs/reference/gdx/gdx_reader_guide.md)
- [Ejemplos de uso](../../../../examples/readme.md)

## 🧪 Tests

```bash
# Ejecutar todos los tests del módulo GDX
pytest tests/babel/gdx/ -v

# Tests específicos
pytest tests/babel/gdx/test_reader.py -v
pytest tests/babel/gdx/test_symbols.py -v
pytest tests/babel/gdx/test_utils.py -v
```

## 📊 Ejemplos

### Lectura básica

```python
from equilibria.babel.gdx.reader import read_gdx

gdx_data = read_gdx("model.gdx")

# Acceder a símbolos
for sym in gdx_data['symbols']:
    print(f"{sym['name']}: {sym['type_name']}")

# Ver elementos únicos
print(gdx_data['elements'])
```

### Filtrado por tipo

```python
from equilibria.babel.gdx.reader import (
    read_gdx,
    get_sets,
    get_parameters,
    get_variables,
    get_equations
)

gdx_data = read_gdx("model.gdx")

sets = get_sets(gdx_data)
params = get_parameters(gdx_data)
vars = get_variables(gdx_data)
eqs = get_equations(gdx_data)
```

### Lectura de valores

```python
from equilibria.babel.gdx.reader import (
    read_gdx,
    read_parameter_values,
    read_variable_values,
    read_equation_values,
    read_set_elements
)

gdx_data = read_gdx("results.gdx")

# Parámetros
prices = read_parameter_values(gdx_data, "price")

# Variables
x_vals = read_variable_values(gdx_data, "X")
for key, attrs in x_vals.items():
    print(f"X{key}: level={attrs['level']}, marginal={attrs['marginal']}")

# Ecuaciones
eq_vals = read_equation_values(gdx_data, "balance")

# Sets
elements = read_set_elements(gdx_data, "i")
```

## 🔧 Detalles Técnicos

### Formato GDX v7

El formato GDX utiliza un layout binario específico:

1. **Header** (bytes 0-25):
   - Checksum/metadata (19 bytes)
   - Magic "GAMSGDX" (7 bytes)
   - Version number (4 bytes, little-endian)
   - Producer info string

2. **Marcadores de sección**:
   - `_SYMB_` - Tabla de símbolos
   - `_UEL_` - Lista de elementos únicos
   - `_DOMS_` - Dominios
   - `_DATA_` - Datos (uno por símbolo)

3. **Tipos de símbolos**:
   - 0x01 - Set
   - 0x3F, 0x64, 0x66, 0x6E - Parameter
   - 0x40, 0x48, 0x63, 0x67, 0xFD - Variable
   - 0x41, 0x68, 0x7E, 0xD9 - Equation
   - 0x20 - Alias

### Compresión

GDX usa varios esquemas de compresión:
- **Valores explícitos** (sin compresión) ✅
- **Secuencias aritméticas** (interpolación lineal) ✅
  - Ejemplo: `10, 20, 30, 40` → almacena solo `10, 40` e interpola
- **Secuencias geométricas** (progresión exponencial) ❌
  - Ejemplo: `1, 2, 4, 8, 16` → **NO se interpola correctamente**
  - Limitación actual: se asume progresión aritmética
- **Datos sparse** (omitir ceros) ✅
- **Delta encoding** (diferencias) ⚠️ Parcial

⚠️ **Advertencia**: Los parámetros comprimidos con progresión geométrica se leerán con valores incorrectos debido a que el código actual usa interpolación lineal (aritmética) en lugar de exponencial (geométrica).

## 🎯 Roadmap

### Completado
- [x] Lectura de headers
- [x] Lectura de tabla de símbolos
- [x] Lectura de UEL
- [x] Lectura de dominios
- [x] Lectura de valores de parámetros 1D
- [x] Lectura de valores de parámetros 2D
- [x] Soporte básico para variables
- [x] Soporte básico para ecuaciones
- [x] Tests comprehensivos

### En progreso
- [ ] Mejorar lectura de variables/ecuaciones
- [ ] Lectura completa de sets multidimensionales
- [ ] Más esquemas de compresión

### Planeado
- [ ] Escritura de archivos GDX (writer)
- [ ] Soporte para GDX v6
- [ ] Validación de integridad
- [ ] Conversión a/desde otros formatos (CSV, JSON, Pandas)

## 🤝 Contribuir

Ver [guía de contribución](../../CONTRIBUTING.md)

## 📝 Licencia y origen

Este módulo es una **reimplementación clean-room** del formato GDX v7
(con una excepción atribuida, ver abajo), desarrollada inspeccionando el
layout binario de archivos GDX producidos por GAMS y leyendo documentación
pública del formato. Fue escrito **sin consultar, copiar, traducir ni derivar**
de:

- GAMS Transfer (Python / R / Matlab),
- las librerías cerradas de GAMS (`gdxcclib`, `gdxapi`), ni
- ninguna otra implementación GDX de terceros.

Las constantes de interoperabilidad usadas aquí (magic `GAMSGDX`, marcadores
`_SYMB_` / `_UEL_` / `_DOMS_` / `_DATA_`, códigos de tipo de símbolo, etc.)
son el formato de cable mismo — no son expresión protegida por copyright.

### Excepción: `decoder.py`

El archivo `decoder.py` **no** es clean-room. Es una re-expresión en Python
del algoritmo de decodificación delta de parámetros en `gdx/src/gxfile.cpp`
del proyecto oficial open-source de GAMS:

> https://github.com/GAMS-dev/gdx — © 2017-2026 GAMS Software GmbH / GAMS
> Development Corp. — Licencia MIT.

La licencia MIT de `GAMS-dev/gdx` es compatible con la de `equilibria`. La
única obligación es preservar el aviso de copyright del upstream, que aparece
en la cabecera de `decoder.py` y en el `NOTICE` raíz del repositorio.

Todo `equilibria.babel.gdx` se distribuye bajo licencia MIT. Ver el archivo
`NOTICE` en la raíz para la declaración completa.

## 🔗 Referencias

- [GAMS GDX Documentation](https://www.gams.com/latest/docs/UG_GDX.html)
- [GAMS Transfer Python](https://www.gams.com/latest/docs/API_PY_GAMSTRANSFER.html)
