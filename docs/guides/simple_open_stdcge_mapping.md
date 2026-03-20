# simpleopen stdcge mapping

Esta guia fija como usamos `stdcge.gms` como ancla oficial para `SimpleOpen`.

Referencia oficial:

- `stdcge.gms : A Standard CGE Model`
- https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_stdcge.html

## decision de alcance

No estamos copiando `stdcge` completo.

Estamos usando una reduccion minima y explicita de `stdcge` para el contrato:

- `simple_open_v1`

Eso significa:

- `stdcge` es el ancla conceptual y algebraica
- `simple_open_v1` es una reduccion pequena y controlada
- el `.gms` nuevo es un `stdcge-reduced benchmark`, no un clon del modelo completo

## que partes de stdcge si usamos

### 1. bloque de transformacion

En `stdcge`:

- `eqpzd`
- `eqE`
- `eqDs`

En `SimpleOpen` usamos la idea de transformacion CET entre:

- bien domestico `D`
- exportacion `E`
- tipo de cambio `ER/PFX`

Eso se reduce a:

- `EQ_CET`

## 2. bloque de precios externos

En `stdcge`:

- `eqpe`
- `eqpm`
- `epsilon`

En `SimpleOpen` no modelamos todo el bloque comercial.

Tomamos solo la idea de:

- `ER`
- `PFX`

como precios exogenos del benchmark.

## 3. bloque domestico / absorcion

En `stdcge`:

- `Q`
- `M`
- `D`
- `eqpqs`
- `eqM`
- `eqD`
- `eqpqd`

En `SimpleOpen` no mantenemos el bloque Armington completo.

Solo retenemos:

- `D`

como input exogeno del benchmark reducido.

## 4. bloque de valor agregado

En `stdcge`:

- `Y`
- `F`
- `py`
- `pf`

En `SimpleOpen` se reduce a un agregado minimo:

- `EQ_VA`

La idea es conservar un agregado CES pequeno entre:

- `ER`
- `PFX`

Eso no es una copia literal del bloque productivo de `stdcge`, pero si sigue su espiritu de bloque agregado con parametros calibrados.

## 5. identidad intermedia reducida

`EQ_INT` no viene de una sola ecuacion literal de `stdcge`.

Es una identidad reducida que captura una pequena capa de consistencia interna entre:

- `INT`
- `X`
- `CAB`
- `FSAV`

Sirve para cerrar el benchmark minimo de `SimpleOpen` sin introducir todo el aparato de absorcion/inversion/gobierno de `stdcge`.

## que partes de stdcge no usamos

Por ahora quedan fuera:

- utilidad y demanda de hogares
- consumo de gobierno
- inversion
- impuestos directos y de produccion
- importaciones `M`
- bien compuesto Armington `Q`
- mercados factoriales completos
- balance macro completo

Eso es intencional. El objetivo actual de la epica no es reproducir `stdcge` entero, sino dar a `SimpleOpen` una referencia GAMS minima y limpia.

## resultado de esta decision

El primer artefacto GAMS que sale de esta ancla es:

- [simple_open_v1_benchmark.gms](/Users/marmol/proyectos/equilibria/src/equilibria/templates/reference/simple_open/scripts/simple_open_v1_benchmark.gms)

Ese script:

- usa `stdcge` como referencia conceptual
- respeta el contrato canonico de `SimpleOpen`
- resuelve solo el benchmark reducido
- exporta niveles y residuales a GDX

## por que no copiamos stdcge literal

Porque hoy `SimpleOpen` todavia no tiene:

- el mismo numero de bloques
- el mismo closure macro
- el mismo conjunto de variables

Si copiaramos `stdcge` completo, dejaria de ser `SimpleOpen` y se volveria otro modelo.

La decision correcta aqui es:

- ancla oficial: `stdcge`
- implementacion propia: `simple_open_v1`
- mapeo explicito entre ambos

## siguiente corte

Con este mapeo ya fijado, el siguiente paso es:

1. correr el `.gms` benchmark
2. leer el GDX desde Python
3. montar el primer comparador Python vs GAMS para `SimpleOpen`
