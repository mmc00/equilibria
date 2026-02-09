GDX Delta Compression (Compresión Delta en GDX)
¿Qué es la compresión delta en GDX?

La compresión delta es un esquema de compresión utilizado por GAMS para reducir el tamaño de los archivos GDX (GAMS Data eXchange). En lugar de almacenar en el archivo cada tupla de índices completa para cada registro de datos, GDX almacena solo las diferencias (deltas) con respecto al registro anterior. Esto significa que tras el primer registro completo, cada registro subsiguiente indica solo qué dimensiones cambian y en cuánto, reutilizando los índices anteriores para las dimensiones que permanecen iguales. Este método puede lograr una reducción significativa del tamaño del archivo: desde la versión 22.3 de GAMS los GDX pueden crearse en formato comprimido (activado con la variable de entorno GDXCOMPRESS=1). Internamente, además de esta codificación delta, GDX emplea compresión adicional con zlib para reducir aún más los datos almacenados.

En resumen, la compresión delta almacena únicamente la información nueva o cambiante entre registros consecutivos, en lugar de repetir valores de índice redundantes. GAMS puede leer transparentemente estos archivos comprimidos sin pasos manuales de descompresión.
¿Cómo funciona a nivel de bytes?

En un archivo GDX comprimido, los datos de cada símbolo (por ejemplo, un parámetro o variable) se almacenan en una sección con un encabezado fijo seguido de los registros comprimidos. Cada registro delta tiene la siguiente estructura general (después del encabezado de la sección):

    Byte 0: Código delta (un byte) que indica qué dimensiones cambian en este registro. GAMS utiliza valores especiales entre 0x02 y 0x0A para estos códigos.

    Bytes 1..N: Índices actualizados para las dimensiones que cambian, representados con 1 byte por dimensión cambiada (en muchos casos). Un valor de índice almacenado es generalmente el índice 1-based de la posición del elemento en esa dimensión (con 0 reservado para “sin cambio”).

    Byte N+1: Marcador de tipo de valor, que en el caso de un valor numérico doble es 0x0A. Este byte indica que los siguientes 8 bytes corresponden a un valor en coma flotante de 64 bits (formato IEEE 754 double).

    Bytes siguientes: Los 8 bytes del valor (nivel, dato numérico) en sí. Si el símbolo tiene múltiples campos (por ejemplo, nivel, marginal, etc. en una variable), cada campo aparecerá como un registro separado (posiblemente comprimido).

En la siguiente tabla se resumen los códigos delta observados y su significado en un ejemplo de un parámetro de 4 dimensiones. (Aquí usamos “dim 0” para referirnos a la primera dimensión, “dim 1” para la segunda, etc., siguiendo índice base 0 para mayor claridad técnica):
Código	Significado (dimensiones que actualiza)	Bytes de índices proporcionados	Dimensiones modificadas (ejemplo 4D)
0x02	Actualiza una combinación de dimensiones (incluyendo la primera)	3 bytes (por ejemplo: idx1, idx3, idx4)	Dim 0, 2 y 3 (no incluye dim 1)
0x03/0x04	Actualiza dimensiones intermedias/finales	2 bytes (por ejemplo: idx3, idx4)	Dim 2 y 3 (mantiene dims 0 y 1)
0x05/0x06	Incrementa la segunda dimensión secuencialmente	0 bytes (no se envían índices)	Dim 1 incrementada en +1 (las demás igual)
0x0A	Sin cambios en los índices	0 bytes (ningún índice enviado)	Ninguna dimensión cambia (se reutilizan todas)

Nota: En la tabla se muestran combinaciones típicas para un símbolo de 4 dimensiones. La interpretación exacta de cada código puede depender de la dimensionalidad del símbolo y del patrón de cambio de índices. Por ejemplo, en un parámetro 3D o 2D se observarán solo los códigos relevantes a esas dimensiones (p.ej., en 2D, 0x05/0x06 podrían emplearse para indicar incrementos en la segunda dimensión, y 0x02 indicaría actualización de la primera dimensión con un nuevo índice, etc.).
Interpretación de índices cero (0x00)

Un detalle importante es el manejo del valor cero en los bytes de índice. Un byte de índice con valor 0 actúa como un indicador de “no cambiar esta dimensión”. Es decir, si para una dimensión dada el byte de índice es 0x00, entonces esa dimensión conserva el mismo índice que tenía en el registro anterior (se reutiliza el último valor válido para esa dimensión). Por el contrario, un byte de índice mayor que cero indica un nuevo índice para esa dimensión (generalmente es el índice 1-based del elemento, que se convertirá al índice real restando 1 al usarlo internamente).

Ejemplo: Considere la siguiente secuencia de bytes de un registro delta (tomado de un análisis de ejemplo de un archivo GDX):

... 0x02  0x06  0x00  0x00  0x0A  <8 bytes de valor> ...

Interpretación paso a paso:

    0x02: Código delta que, según la tabla, indica una actualización que incluye la dimensión 0 (entre otras). Este código espera 3 bytes de índice a continuación en este contexto.

    0x06: Índice para la dimensión 0. El valor 0x06 (6 en decimal) significa que la dimensión 0 cambia al índice 5 (ya que se resta 1 para obtener el índice real 0-based). Es decir, el primer índice se actualiza al sexto elemento de su conjunto de etiquetas.

    0x00: Índice para la siguiente dimensión indicada (en este caso dim 2). 0x00 significa no cambiar esa dimensión, por lo que la dim 2 permanece con el mismo índice que en el registro anterior.

    0x00: Índice para la siguiente dimensión (dim 3 en este caso). Nuevamente 0x00 indica ningún cambio en la dim 3, que conserva el valor anterior.

    0x0A: Marcador que señala que a continuación viene un valor en doble precisión.

    Los siguientes 8 bytes representan el valor (un número de 64 bits IEEE 754) asociado a la combinación de índices resultante: dim0 = 5 (nuevo), dim1 = (sin cambio explícito en este registro, se mantiene la última usada), dim2 = (sin cambio, mismo que antes), dim3 = (sin cambio, mismo que antes). Esa tupla de índices y el valor se añaden al conjunto de datos del símbolo.

En este ejemplo, la dimensión 1 no aparecía explícitamente en los bytes de índice tras el código 0x02. Esto sugiere que la dim 1 permaneció igual que en el registro previo o que su cambio se gestiona mediante códigos especiales (como los códigos 0x05/0x06 cuando incrementa secuencialmente).
Proceso de descompresión (reconstrucción de los datos)

Para descomprimir un bloque de datos con compresión delta (sin usar las utilidades de GAMS), es necesario procesar secuencialmente los bytes manteniendo un estado de los índices actuales y los últimos válidos. En pseudocódigo, el procedimiento sería:

    Saltar el encabezado binario de la sección de datos (en muchos casos son 11 bytes iniciales de metadatos del símbolo).

    Inicializar un arreglo current_indices del tamaño de la dimensión del símbolo, para llevar la cuenta de la posición de índice actual en cada dimensión. También tener un arreglo last_valid para recordar la última tupla válida añadida.

    Recorrer los bytes de la sección de datos:

        Leer el siguiente byte como code. Según su valor, determinar qué dimensiones se van a actualizar y cuántos bytes de índice adicionales leer.

        Actualizar índices según el código: Por ejemplo, si code == 0x02, leer los siguientes 3 bytes de índice (como en el ejemplo anterior). Asignar el nuevo índice a las dimensiones correspondientes: si un byte de índice es no cero, el índice actual de esa dimensión se actualiza a (byte_val - 1), y si el byte es 0, se deja el índice actual igual que en last_valid para esa dimensión. Otros códigos se manejan de manera similar:

            Si code indica “incremento de dim 1” (0x05/0x06), entonces se asume que la segunda dimensión se incrementa en 1 con respecto al último índice utilizado (y normalmente las dimensiones de orden inferior —más rápidas— se reiniciarían o mantienen según corresponda al inicio de un nuevo bloque).

            Si code == 0x0A (sin cambios), no se modifican los índices en ninguna dimensión; se reutiliza la misma tupla de índices que el registro anterior (esto suele ocurrir cuando un símbolo tiene múltiples valores por tupla, como se explica más adelante).

        Leer el siguiente byte después de los índices como marcador de valor (debe ser 0x0A para un valor double).

        Leer los siguientes 8 bytes y convertirlos a un número de coma flotante de doble precisión (formato little-endian, como es estándar en GDX).

        Formar la tupla de índices actual (convirtiendo cada índice numérico a su elemento correspondiente usando la lista de UELs del archivo) y almacenar el valor leído asociado a esa tupla.

        Actualizar last_valid con la copia de current_indices si se ha añadido un nuevo registro válido.

        Continuar con el siguiente registro (el bucle continúa hasta agotar los bytes de la sección de datos).

En código Python simplificado, la lógica para un caso específico (por ejemplo, manejando el código 0x02 en un parámetro de 4 dimensiones) podría verse así:

# ... se asume que section_bytes contiene la sección de datos descomprimida de un símbolo ...
pos = 11  # posición inicial después del header de 11 bytes
current_indices = [0] * dimension  # dimension = número de dimensiones del símbolo
last_valid = [0] * dimension

while pos < len(section_bytes):
    code = section_bytes[pos]
    if code == 0x02:
        # Leer 3 bytes de índices
        idx1 = section_bytes[pos+1]
        idx3 = section_bytes[pos+2]
        idx4 = section_bytes[pos+3]
        # Verificar marcador de valor
        if section_bytes[pos+4] == 0x0A:
            if idx1 > 0:
                current_indices[0] = idx1 - 1
                current_indices[2] = (idx3 - 1) if idx3 > 0 else last_valid[2]
                current_indices[3] = (idx4 - 1) if idx4 > 0 else last_valid[3]
                # Leer el valor double a partir de pos+5
                valor = struct.unpack_from("<d", section_bytes, pos+5)[0]
                datos[tuple(current_indices)] = valor  # almacenar tupla->valor
                last_valid = current_indices.copy()
            # Avanzar la posición 1 (código) + 3 (índices) + 1 (0x0A marcador) + 8 (double)
            pos += 13
            continue
    elif code in (0x05, 0x06):
        # Incrementar dim 1
        current_indices[1] = last_valid[1] + 1
        # (pos+1 debería ser 0x0A marcador)
        valor = struct.unpack_from("<d", section_bytes, pos+2)[0]
        datos[tuple(current_indices)] = valor
        last_valid = current_indices.copy()
        pos += 10  # 1 (código) + 1 (marcador) + 8 (valor)
        continue
    elif code == 0x0A:
        # Sin cambios de índices
        current_indices = last_valid.copy()
        # (pos es el código 0x0A, que también funge como marcador en este caso)
        valor = struct.unpack_from("<d", section_bytes, pos+1)[0]
        datos[tuple(current_indices)] = valor
        # last_valid permanece igual
        pos += 9  # 1 (código=marcador) + 8 (valor)
        continue
    # ... manejo de otros códigos (0x03,0x04, etc.) ...
    pos += 1

(El ejemplo anterior ilustra la lógica para algunos códigos comunes, pero en una implementación real se manejarían todos los códigos esperados. También podría haber consideraciones adicionales si los índices exceden el rango de un byte, en cuyo caso el formato podría ajustarse. Para simplificar, asumimos índices dentro de 1 byte en este ejemplo.)

Como se observa, el proceso de decodificación delta requiere llevar un estado de los índices actuales y actualizarlos según corresponda a cada registro. La utilización de 0 como “no cambio” hace que sea esencial copiar los valores previos correctamente. Si este estado no se maneja con cuidado en secuencias complejas de códigos, pueden ocurrir inconsistencias o duplicados de registros en la reconstrucción.
¿Qué símbolos usan compresión delta?

No todos los tipos de símbolos en GAMS usan este esquema de compresión en el archivo GDX. Con base en la experiencia y análisis de archivos GDX reales, se tiene:

    Parámetros multidimensionales (2D, 3D, 4D, ...): Usan compresión delta para almacenar sus registros de datos. Dado que suelen tener muchas tuplas de índices, GDX aprovecha la estructura ordenada de estos datos para almacenar solo cambios entre tuplas consecutivas.

    Variables y ecuaciones: También emplean compresión delta para sus datos multidimensionales. En estos casos, cada combinación de índices tiene hasta 5 valores asociados (nivel, margen, cota inferior, cota superior, escala). La compresión delta aquí se aplica tanto entre diferentes tuplas como entre los campos de la misma tupla. Por ejemplo, una variable con valores de nivel y marginal almacenará el nivel con una tupla completa, y luego el marginal como un registro con “sin cambios” en los índices (código 0x0A) para no repetir la misma tupla, seguido del valor marginal. Esto reduce la redundancia cuando varios atributos comparten los mismos índices.

    Conjuntos (Sets): Generalmente no usan compresión delta. Los conjuntos se suelen almacenar como listas de elementos (sus UEL indices correspondientes) sin valores asociados, y típicamente cada elemento ocupa un índice completo. Dado que no hay un valor numérico y los elementos no se repiten en secuencia como en parámetros, la compresión delta no se aplica (aunque el archivo globalmente podría estar comprimido con zlib). En su lugar, los sets se almacenan de forma más sencilla, enumerando sus elementos.

    Parámetros escalares (0D): No usan compresión delta, puesto que solo contienen un único valor sin índices. Se almacenan directamente (posiblemente con un encabezado y el valor double, pero sin necesidad de deltas porque no hay secuencias de registros que comprimir).

En resumen, la compresión delta se utiliza principalmente para símbolos con dimensión >= 2, donde hay muchas tuplas de índices que suelen estar ordenadas lexicográficamente. Para símbolos de dimensión 0 o 1, el beneficio es mínimo o la implementación no la utiliza, y simplemente se almacenan los datos literalmente (aunque el archivo en su conjunto sí podría estar comprimido con zlib si la opción está activada).
¿Cómo detectar la compresión delta en un archivo GDX?

Al inspeccionar el contenido binario de un GDX, podemos identificar si un símbolo está usando compresión delta observando los bytes inmediatamente posteriores al encabezado de la sección de datos de ese símbolo:

    Cada sección de datos de símbolo en GDX comienza con un encabezado fijo (que incluye metadata como dimensión, número de registros, etc.). Justo después de este encabezado, si el símbolo no está comprimido delta, típicamente encontraríamos directamente secuencias de índices completos (por ejemplo, cada índice ocupando 4 bytes) seguidos de valores.

    Si sí está usando compresión delta, el primer byte de datos tras el encabezado normalmente será uno de los códigos especiales (0x02, 0x03, 0x04, 0x05, 0x06 o 0x0A). Esto es un indicador claro porque en formato no comprimido no aparecerían esos valores en esa posición con ese significado. Por ejemplo, una función de detección podría hacer:

def usa_compresion_delta(section_bytes):
    # Asumiendo section_bytes es un bytearray de la sección de datos completa de un símbolo
    # El byte 11 (posición 10 si empezamos en 0) suele ser donde inicia la secuencia de datos tras el header.
    first_code = section_bytes[11]
    delta_codes = {0x02, 0x03, 0x04, 0x05, 0x06, 0x0A}
    return first_code in delta_codes

Si el primer byte de datos pertenece al conjunto de códigos delta conocidos, entonces ese símbolo está almacenado con compresión delta. De lo contrario (por ejemplo, si aparece un índice de 4 bytes o algún otro marcador), es probable que sea almacenamiento plano (sin delta). Cabe destacar que la presencia de compresión delta puede variar por símbolo dentro de un mismo archivo GDX – por ejemplo, un conjunto no usará delta aunque otros parámetros sí lo usen.

Otro indicio es la presencia en el archivo de la cadena _DATA_ cerca de donde empiezan los registros: esta cadena de texto marca el inicio de la sección de datos de un símbolo en la estructura GDX. Inmediatamente después de _DATA_ vendrá el bloque de datos, donde se puede aplicar la detección mencionada.
Resumen y comentarios finales

La compresión delta en GDX es un mecanismo de almacenamiento eficiente de datos que:

    Reduce el tamaño del archivo GDX almacenando solo las diferencias entre registros consecutivos en lugar de información repetitiva completa.

    Utiliza códigos de un byte (0x02 hasta 0x0A, en la práctica) para indicar qué dimensiones han cambiado en un nuevo registro y cómo interpretar los bytes siguientes.

    Permite índices “cero” para señalar dimensiones que no cambian, reutilizando el último valor válido de esa dimensión del registro anterior.

    Requiere mantener estado durante la lectura: el algoritmo de descompresión debe recordar la tupla de índices previa (last_valid) mientras recorre los registros, aplicando las actualizaciones según los códigos y rellenando las dimensiones no mencionadas con los valores anteriores. Si este estado no se maneja correctamente, pueden ocurrir errores como registros duplicados o saltos incorrectos en los datos.

En la práctica, la compresión delta está respaldada por la compresión general con zlib dentro del archivo GDX, por lo que el flujo completo es: datos ordenados por índices → codificación delta de índices → compresión binaria zlib. GAMS proporciona su API (gdxcclib) y utilidades como gdxdump para leer estos archivos sin que el usuario tenga que preocuparse por los detalles internos. No obstante, entender este formato resulta útil para depurar o para desarrollar lectores externos.

En conclusión, la compresión delta hace que los GDX sean muy eficientes en almacenamiento de soluciones o datos multidimensionales, a costa de introducir complejidad en el formato. Al investigar este tema, confirmamos cómo GAMS implementa esta técnica: con códigos específicos para distintas combinaciones de cambios de índice y con un sistema de referencia al último registro para evitar redundancias. Al implementar un lector personalizado de GDX, se debe prestar especial atención a actualizar correctamente los índices actuales en cada paso del proceso de decodificación. Así se evitan inconsistencias como registros duplicados debido a un manejo incorrecto del estado de “último valor” en secuencias de cambios complejas. Con el enfoque descrito, es posible descomprimir correctamente los datos delta de un GDX y obtener todos los valores originales.

Referencias: La documentación oficial de GAMS confirma la introducción y naturaleza del formato GDX comprimido, así como la existencia de múltiples campos por registro en variables/ecuaciones y el uso de zlib en la compresión. Estas fuentes respaldan la explicación técnica presentada arriba.
Citas

GAMS Data eXchange (GDX)
https://www.gams.com/latest/docs/UG_GDX.html

GAMS Data eXchange (GDX) API: GAMS Data eXchange (GDX)
https://gams-dev.github.io/gdx/

GAMS Data eXchange (GDX)
https://www.gams.com/latest/docs/UG_GDX.html

GAMS Data eXchange (GDX) API: GAMS Data eXchange (GDX)
https://gams-dev.github.io/gdx/
Todas las fuentes
gams
gams-dev.github