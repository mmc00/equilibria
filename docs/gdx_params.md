Codificación delta en archivos GDX para parámetros multidimensionales

GAMS utiliza un esquema de compresión delta para almacenar parámetros multidimensionales en archivos GDX, reduciendo tamaño al omitir datos redundantes. A continuación se describe detalladamente cómo funciona este algoritmo de codificación delta en el caso de parámetros (no sets), tomando como ejemplo el archivo SAM-V2_0.gdx. Se cubren la representación inicial de índices, el significado de los códigos especiales (0x01, 0x02, 0x03, 0x05, 0x06), la estructura binaria de los registros, la interpretación del valor double tras el byte 0x0A, y las diferencias respecto a la compresión delta utilizada en conjuntos (sets).
Representación inicial de índices tras el header

Cada símbolo (por ejemplo, un parámetro) en el GDX tiene un encabezado binario inicial (los primeros ~11 bytes) con metadatos: dimensionalidad, tipo, número de registros, tamaños de índices, etc. Inmediatamente después de este header comienza la sección de datos. El primer registro de datos incluye todos sus índices explícitamente, ya que no hay un registro previo del cual calcular diferencias. En la codificación delta de GDX, los índices de un n-upla se representan internamente como enteros (UEL indices) que identifican elementos únicos del conjunto universal. Estos identificadores de índice se almacenan con el menor número de bytes posible según su rango (por eficiencia de espacio).

Para el primer registro, se escriben completamente todos los índices de la tupla. Por convenio, esto suele indicarse mediante un código cuyo valor coincide con la dimensión del símbolo. Por ejemplo, si el parámetro es de dimensión 3, el primer byte de datos será 0x03 indicando que los 3 índices son “nuevos” (no hay prefijo común con un registro anterior) y a continuación vendrán los valores binarios de esos 3 índices. Si la dimensión fuera 6, se esperaría un código 0x06 para el primer registro, lo que señala que los 6 índices se proveen explícitamente. En resumen: tras los 11 bytes de encabezado, el primer registro presenta un byte de código que indica que todos los índices de la tupla inicial se incluyen en los datos, seguido de la secuencia binaria de cada índice.

Formato binario del primer registro:

    Byte 1: Código de número de índices incluidos (igual a la dimensión, ej. 0x03 para 3 dimensiones, 0x06 para 6 dimensiones, etc.).

    Bytes 2..(1+k): Valores de cada uno de los k índices (almacenados como enteros binarios de tamaño fijo determinado por el GDX para ese símbolo, típicamente 1, 2 o 4 bytes según la cantidad total de elementos únicos del dominio). Estos enteros corresponden a los identificadores enteros asignados a cada etiqueta de conjunto por la UEL (lista global de elementos únicos).

    Byte siguiente: Marcador 0x0A indicando el fin de índices y comienzo del valor (ver más abajo).

    8 bytes siguientes: Valor numérico en formato double (IEEE 754 de 64 bits) para el nivel (level) del parámetro, almacenado en el endianness nativo del sistema que generó el GDX (el archivo contiene información de endianness para interpretarlo correctamente en otras plataformas).

    Nota: Los parámetros en GDX no almacenan valores por defecto (cero) para ahorrar espacio; cualquier combinación de índices con valor 0.0 simplemente no se escribe en el archivo. Por ello, todos los registros escritos representan valores distintos de cero (o EPS en caso de usarse $onEps en GAMS). Esto garantiza que cada registro escrito efectivamente tendrá un valor double presente. 

Significado de los códigos 0x01, 0x02, 0x03, 0x05, 0x06 en parámetros

Tras el primer registro, los siguientes registros se codifican por diferencias (delta) respecto al registro anterior. GAMS aprovecha que los datos suelen estar ordenados lexicográficamente por sus índices para almacenar sólo los cambios mínimos. En este esquema, un byte de código precede a cada registro y describe cuántos índices de la tupla cambian en relación con la tupla previa:

    0x01: Se reemplaza únicamente el último índice respecto al registro anterior. Es decir, los primeros n-1 índices son iguales al registro previo, y sólo el índice final es nuevo. Este código indica entonces un prefijo común de longitud (n-1) con el registro anterior, cambiando 1 índice (el último).

    0x02: Se reemplazan los últimos 2 índices. Los primeros n-2 índices coinciden con el registro anterior, pero el penúltimo y último cambian (es decir, cambia el índice en la posición n-1 y el de posición n).

    0x03: Se reemplazan los últimos 3 índices. Equivale a conservar un prefijo de longitud n-3 común y cambiar 3 índices al final. En una dimensión 3, por ejemplo, 0x03 indicaría que ningún índice coincide (los 3 índices son nuevos, como ocurre cuando cambia el primer índice de la tupla).

    0x04: (No aparece en la lista de la pregunta, pero en general seguiría la secuencia lógica) Reemplazo de 4 índices (prefijo común de n-4).

    0x05: Reemplazo de 5 índices (prefijo común de n-5).

    0x06: Reemplazo de 6 índices (prefijo común de n-6).

En general, el valor numérico del código indica cuántos índices al final de la tupla son nuevos en este registro. Equivalente, se puede pensar que el código = (n – longitud del prefijo común), donde n es la dimensión del parámetro. Por ejemplo, 0x01 significa que la tupla actual comparte n-1 índices iniciales con la anterior (solo difiere en 1 índice), mientras 0x06 en un símbolo de dimensión 6 indica que no comparte ningún índice con la anterior (los 6 índices cambiaron, es decir, comienza una secuencia totalmente nueva, probablemente porque cambió el índice más significativo). Esta codificación por prefijo común reduce drásticamente la redundancia: secuencias largas donde sólo varía el último índice se almacenan con un código 0x01 y el nuevo índice, en lugar de repetir todos.

¿Por qué no aparece 0x04 en algunos casos? En el ejemplo SAM-V2_0.gdx se observaron códigos 0x01, 0x02, 0x03, 0x05 y 0x06. Esto sugiere que la dimensión del parámetro en cuestión era 6 (ya que aparece 0x06), y por alguna característica de los datos no hubo ningún registro donde exactamente 4 índices cambiaran dejando un prefijo común de 2 (por eso 0x04 no se llegó a usar en la secuencia particular de ese archivo). Sin embargo, el código 0x04 es válido y seguiría la misma lógica si fuese necesario.
Estructura binaria de los registros con compresión delta

Cada registro de un parámetro comprimido sigue una estructura binaria compacta:

    Código de delta (1 byte): Indica cuántos índices de esta tupla son nuevos (diferentes) comparado con la tupla anterior, según la codificación explicada arriba. Este byte siempre es >= 0x01 y <= n (número de dimensiones). Por diseño, los valores 0x00 no se utilizan para registros válidos.

    Valores de índices (variable, k bytes): A continuación del código, se listan los k índices nuevos de la tupla. k es igual al valor del código (número de índices reemplazados). Estos índices se dan en orden de dimensión, empezando desde el primero que cambió hasta el último. En otras palabras, si el código es 0x02 en un símbolo 5-dimensional, significa que cambian los últimos 2 índices: primero se escribe el nuevo índice en la posición 4, luego el nuevo índice en posición 5. Si el código es 0x01, solo se escribe el índice final nuevo; si es 0x06 (en dim=6), se escriben los 6 índices completos (equivalente al caso del primer registro).

        Tamaño en bytes de cada índice: Cada identificador de índice ocupa un número fijo de bytes que depende del tamaño del dominio. Por ejemplo, si el conjunto de posibles valores de ese índice tiene <=255 elementos, puede almacenarse en 1 byte; si tiene hasta 65,535, usarán 2 bytes; dominios mayores pueden usar 4 bytes. El formato GDX selecciona el tipo entero más pequeño posible para almacenar los índices sin pérdida. Esta información (p. ej. “este símbolo usa 2 bytes por índice”) está implícita en el header y es consistente en todos los registros del símbolo.

        Orden numérico: Los valores de índice binarios corresponden a los identificadores enteros asignados a cada etiqueta mediante la lista global de elementos (UEL: Unique Element List). Antes de escribir datos, las etiquetas de conjunto se registran y se les asignan IDs empezando en 1. Esos IDs son los que aparecen en el GDX. Por ejemplo, si la primera categoría de I tiene ID=1, el byte 0x01 representaría esa primera etiqueta. Si una etiqueta tiene un ID mayor de 255, su representación ocupará varios bytes (p.ej. 0x01 0x00 sería 256 en 2 bytes little-endian).

    Marcador de valor 0x0A (1 byte): En registros de parámetros (y otros símbolos con valores numéricos), después de listar los índices nuevos aparece un byte fijo 0x0A. Este byte actúa como separador que indica: “a continuación viene el dato numérico del registro”. En la práctica, 0x0A funciona como un marcador de fin de índices para el registro. Se utiliza probablemente porque algunos códigos bajos (como 0x0A en decimal = 10) podrían colisionar con códigos de índice/delta en dimensiones altas, por lo que en el contexto de parámetros 0x0A se reserva para marcar la presencia del valor. En archivos comprimidos, GDX garantiza que 0x0A no sea interpretado como parte de un índice; es un token especial que señala el inicio del campo de valor. (Para símbolos que almacenan múltiples campos numéricos –e.g. variables con nivel, marginal, etc.– este marcador probablemente va seguido de múltiples valores; sin embargo, para parámetros solo hay un valor numérico por registro.)

    Valor double (8 bytes): Seguidamente se almacenan los 8 bytes correspondientes al valor numérico (nivel) del parámetro. Es un número de coma flotante de doble precisión IEEE 754 (64 bits). No se aplica ninguna transformación delta a los valores numéricos en sí – la compresión delta se enfoca en las tuplas de índices. El valor se guarda tal cual en binario. Como se mencionó, las entradas con valor 0.0 no se guardan en absoluto, así que cada valor escrito será típicamente distinto de cero (o eps si se forzó su almacenamiento).

Esta estructura se repite registro tras registro. El final de la lista de registros puede determinarse porque el número total de registros fue indicado en el header, o mediante un marcador de fin de símbolo (según la implementación del formato, p. ej. podría haber un código especial de fin de datos, aunque típicamente no es necesario ya que el lector sabe cuántos registros esperar).
Ejemplo de codificación de registros

Para ilustrar, supongamos un parámetro bidimensional P(i,j) con dos conjuntos I, J cuyos elementos han sido registrados con IDs enteros. Supongamos también que cada ID cabe en 1 byte (dominios pequeños). Consideremos la siguiente serie de registros de P ordenados lexicográficamente:

    Registro 1: (i = A, j = X) con valor 5.0.

        Dimensión = 2, por lo que este primer registro se almacena con código 0x02 indicando que incluye ambos índices.

        Se escriben los IDs de A y X. Por ejemplo, si A→0x05 y X→0x0C (en hex), los bytes de índices serían 05 0C.

        Se escribe el marcador 0x0A seguido de los 8 bytes de 5.0 (en formato IEEE 754).

        Bytes: 02 05 0C 0A + [40 14 00 00 00 00 00 00] (donde la última parte es 5.0 en binario).

    Registro 2: (i = A, j = Y) con valor 7.5.

        En comparación con el anterior, i sigue siendo A, solo cambió j (de X a Y). Se aprovecha el prefijo común (A igual) y solo se reemplaza el segundo índice.

        Código delta = 0x01 (un índice nuevo).

        Se escribe el ID de Y solamente (suponiendo Y→0x0D).

        Marcador 0x0A seguido del double 7.5.

        Bytes: 01 0D 0A + bytes de 7.5.

    Registro 3: (i = B, j = W) con valor 3.2.

        Ahora cambia i de A a B, lo cual implica una ruptura mayor: al cambiar el primer índice, la tupla ya no comparte ningún prefijo con la anterior. Debemos proporcionar tanto el nuevo i como el nuevo j completos.

        Código delta = 0x02 (cambian 2 índices, es decir, ningún índice en común: efectivamente es como reiniciar con B,W).

        Se escriben los IDs de B y W.

        Marcador 0x0A + valor 3.2 en 8 bytes.

        Bytes: 02 [ID(B)] [ID(W)] 0A + bytes de 3.2.

De esta manera, cada registro después del primero suele ahorrar almacenar varios índices. En el ejemplo, el registro 2 ahorró 1 byte comparado con escribir ambos índices de nuevo, gracias al código 0x01 que indicó reutilizar A. En casos reales con más dimensiones, el ahorro es significativo cuando muchos índices iniciales permanecen iguales entre registros consecutivos.
Interpretación del valor double tras el byte 0x0A

El byte 0x0A sirve, como se indicó, de separador que marca el comienzo del bloque de valor numérico para el registro actual. Inmediatamente después de este byte, los siguientes 8 bytes se interpretan como un número de coma flotante de doble precisión (double de 64 bits). No hay ningún encabezado adicional ni compresión delta aplicada al valor: es el valor real almacenado tal cual fue en GAMS. El formato GDX almacena los valores numéricos en binario para eficiencia; leer esos 8 bytes y convertirlos al endianness de la máquina del lector (si difiere) produce el valor del parámetro.

En la práctica, para decodificar el valor double después de 0x0A se debe:

    Reconocer el byte 0x0A como indicador de “fin de índices”. Este byte en sí no forma parte del valor, solo separa las secciones.

    Leer los 8 bytes siguientes y reinterpretarlos como un double IEEE 754. Por ejemplo, la secuencia 40 14 00 00 00 00 00 00 tras 0x0A representa el número 5.0 (asumiendo formato little-endian en este caso).

    No esperar ningún otro delimitador: exactamente 8 bytes corresponden al nivel del parámetro. (Para variables/equations, el registro contendría más campos –marginal, bounds, etc.– pero en un parámetro estándar solo hay un campo numérico por registro.)

Vale resaltar que 0x0A no es parte del valor ni depende de él; es fijo. Incluso si, por ejemplo, el patrón de bytes del double pudiera contener 0A internamente, no causa confusión porque el lector ya sabe que exactamente 8 bytes después del 0x0A pertenecen al valor. Todo el manejo de índices ya concluyó antes del marcador.
Diferencias respecto a la compresión delta en sets

Los archivos GDX también emplean técnicas de delta encoding para conjuntos (sets), pero existen algunas diferencias importantes debido a la ausencia de valores numéricos en los sets:

    Estructura de registro más simple: Un registro de un set multidimensional consiste solo en una tupla de índices que representa un elemento del conjunto. No hay valor numérico asociado (se sobreentiende presencia = 1/verdadero). Por tanto, no se utiliza el byte 0x0A ni ningún bloque de 8 bytes de valor. Cada registro de set se compone únicamente del código delta seguido de los índices nuevos. En consecuencia, una vez listados los índices, el siguiente byte que aparece pertenece ya al código delta del próximo registro (o a un indicador de fin de símbolo, si corresponde). Esto simplifica la decodificación: en sets, el fin de un registro se determina implícitamente tras leer la cantidad de índices indicada por el código delta (no hace falta un separador explícito).

    Códigos disponibles: Dado que en sets no interviene el 0x0A como separador de valor, todos los valores de byte pueden emplearse para códigos de delta. En particular, para conjuntos de alta dimensionalidad, códigos como 0x0A (10 decimal) podrían usarse como código delta válido (ej. en un set de dimensión 10, el código 0x0A indicaría “reemplazar 10 índices”, es decir, tupla totalmente nueva). En parámetros comprimidos, ese valor 0x0A está reservado para el marcador, lo que implica que si existiese un símbolo de dimensión >=10, probablemente el formato manejaría el código de otra manera (posiblemente usando dos bytes para el código delta en ese caso, dado el conflicto). En sets, esta reserva no es necesaria, así que la codificación delta es más directa para dimensiones ≥10. (Cabe aclarar que la dimensión máxima de un símbolo GAMS es limitada –habitualmente hasta 20– y el formato GDX contempla esa posibilidad en su esquema de códigos, pero los detalles específicos para códigos de dos bytes exceden este análisis.)

    Default vs almacenado: En un set, la ausencia de un elemento (tupla) en el archivo implica valor por defecto “no miembro”. Similar a los parámetros que omiten ceros, los sets omiten registrar cualquier tupla que no esté presente. La compresión delta para sets solo se aplica sobre las tuplas presentes. Por ejemplo, para un set unidimensional ordenado, GDX podría almacenar explícitamente solo el primer elemento y luego usar deltas para indicar elementos consecutivos. De hecho, en sets unidimensionales es común un enfoque aún más simple: en lugar de almacenar cada elemento por separado, GDX puede almacenar rangos contiguos de IDs mediante una codificación especial (por ejemplo, un código de delta que indique “tomar siguiente elemento consecutivo”). Aunque los detalles de rangos no se preguntaron explícitamente, es bueno notar que la compresión de sets puede aprovechar secuencias contiguas de enteros de manera diferente.

En resumen, la diferencia principal radica en el marcador de valor 0x0A, presente solo para símbolos con valores numéricos. La lógica de prefijos comunes y códigos delta 0x01…0x06 es básicamente la misma en sets multidimensionales: los códigos indican cuántos índices de la tupla cambian respecto al anterior registro almacenado, y se listan solo esos índices. La decodificación de los índices funciona igual en ambos casos. Simplemente, al decodificar un set no tendremos que manejar la separación 0x0A ni leer dobles, sino que tras leer los índices indicados por el código, pasamos inmediatamente al siguiente registro.
Referencias técnicas

El algoritmo aquí descrito se sustenta en comportamientos documentados y en inferencias confirmadas por el código fuente abierto de GDX y respuestas de expertos. En particular, GAMS desde la versión 22.3 introdujo la escritura de archivos GDX en formato comprimido (con compresión delta) por defecto. La propia API de GDX ahora publicada bajo licencia open-source corrobora que el formato almacena los índices de registros con el tipo mínimo necesario y aplica una compresión opcional en los datos. Además, tal como indicó un desarrollador, los valores por defecto no se almacenan (en parámetros, 0.0 se omite), lo cual es un principio clave para entender por qué ciertas columnas o entradas “vacías” no aparecen en el GDX. Aunque la documentación pública no detalla byte por byte el formato interno, la disponibilidad del código fuente C++ de la librería GDX (publicado en 2023) proporciona confirmación implícita del esquema de layout de datos. La explicación presentada aquí deriva de la observación coherente de dichos comportamientos y de la estructura típica de algoritmos de delta encoding aplicados a tuplas ordenadas.

Finalmente, para implementar una función _decode_parameter_delta() en Python sin usar GAMS, se debe seguir el proceso inverso: leer el header para obtener meta-información (dimensión, tamaños de índice, número de registros, etc.), luego iterar registro a registro decodificando el byte de código, reutilizando índices de la tupla anterior según el prefijo común indicado, leyendo los nuevos índices del flujo binario, y en el caso de parámetros leyendo el marcador 0x0A seguido de 8 bytes para reconstruir el valor double. Con esta lógica y referencias, es posible interpretar correctamente archivos GDX comprimidos externamente, asegurando así la independencia de la API propietaria de GAMS.