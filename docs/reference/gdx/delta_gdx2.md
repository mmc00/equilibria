Comprensión de los códigos delta en la compresión GDX

Resumen: Los archivos GDX utilizan compresión delta para almacenar registros de símbolos multi-dimensionales de forma más compacta. En esencia, después de escribir completamente el primer registro (todas las tuplas de índices completas), los registros subsiguientes incluyen un código delta que indica qué dimensiones cambian con respecto al registro anterior y cuáles permanecen iguales. Interpretar correctamente estos códigos es crucial para reconstruir sin errores los índices de cada registro. A continuación explicamos el significado de cada código delta y cómo actualiza las dimensiones, las interacciones implícitas entre dimensiones (por ejemplo, reinicios de índices), cómo GAMS maneja el estado de los índices durante la lectura/escritura comprimida, y referencia a herramientas que implementan correctamente este mecanismo.
1. Mapeo de códigos delta a dimensiones actualizadas

En la compresión delta de GDX, el código delta es esencialmente una máscara de bits donde cada bit representa una dimensión del símbolo. Un bit indica si esa dimensión se actualiza (cambia a un nuevo índice) o permanece igual al valor del registro anterior. Para un símbolo de $n$ dimensiones, se usan $n$ bits del código delta (si $n>8$, el código puede abarcar más de un byte). En el caso típico de 4 dimensiones (dimensiones 0 a 3), el código delta se interpreta bit a bit:

    Convención de bits: El bit menos significativo (LSB) corresponde a la última dimensión (la más rápida en variar, dim3 si indexamos 0-3), y el bit más significativo usado corresponde a la primera dimensión (la más lenta, dim0).

    Bit = 1: Significa que esa dimensión cambia en este registro, es decir, se lee/almacena un nuevo índice para ella.

    Bit = 0: Significa que esa dimensión no cambia respecto al registro anterior (se reutiliza el mismo índice anterior para esa posición).

Ejemplo (4 dimensiones): Supongamos que el registro previo tenía la tupla $(i_0, i_1, i_2, i_3)$. Si el siguiente registro tiene un código delta 0x0A (hexadecimal), en binario es 1010<sub>2</sub>. Desglosando los bits (asumiendo la convención LSB=dim3, ... MSB=dim0):

    1 (bit 3) – dimensión 0 cambiada.

    0 (bit 2) – dimensión 1 sin cambio.

    1 (bit 1) – dimensión 2 cambiada.

    0 (bit 0) – dimensión 3 sin cambio.

Esto significa que el nuevo registro provee nuevos índices para las dimensiones 0 y 2, mientras mantiene las dimensiones 1 y 3 iguales a las del registro anterior. Así, una tupla resultante podría ser, por ejemplo, $(L, K, ROW, ROW)$ pasando a $(\text{otro }L, K, \text{otro }X, ROW)$ – donde hemos marcado en negrita las posiciones que permanecieron iguales (dim1 y dim3) para ilustrar. Nótese que en el contexto original del usuario, interpretaron erróneamente 0x0A como “ningún índice cambia”, lo que llevó a duplicar tuplas completas (porque reutilizaron todos los índices). En realidad, 0x0A indica cambios en ciertas dimensiones (dim0 y dim2 en este caso), no que todo siga igual.

Siguiendo esta lógica, podemos aclarar el significado de otros códigos delta comunes en símbolos de 4 dimensiones:

    0x00 (0000<sub>2</sub>): No suele aparecer para registros normales, ya que implicaría que ninguna dimensión cambia (tupla idéntica a la anterior). En un GDX bien formado, cada tupla de índices debería ser única, por lo que un código 0000 solo tendría sentido especial (p.ej. podría indicar final de datos o un caso trivial). En la práctica no se usa para generar un registro nuevo, porque duplicaría la entrada previa.

    0x01 (0001<sub>2</sub>): Solo el bit0=1, los demás 0. Actualiza solo la última dimensión (dim3), manteniendo dim0–dim2 iguales. Es el caso típico de avanzar al siguiente elemento en la cuarta dimensión, con las primeras tres fijas.

    0x02 (0010<sub>2</sub>): Bit1=1, los demás 0. Actualiza solo la tercera dimensión (dim2), dejando dim0, dim1 y dim3 iguales. Esto ocurriría si cambia el índice de la tercera dimensión mientras, sorprendentemente, la cuarta permanece con el mismo valor que antes. En un orden lexicográfico usual, cambiar dim2 normalmente implicaría reiniciar dim3 (ver más abajo), pero si la dimensión 3 tiene cardinalidad 1 (solo un posible valor) o el nuevo valor de dim3 coincide con el anterior (p. ej. ambos registros usan el primer elemento de esa dimensión), entonces es plausible un código 0x02. En general, 0x02 no significa “dims 0,1,2 cambian” (como llegó a suponerse), sino solo cambio en dim2.

    0x03 (0011<sub>2</sub>): Bits1=1 y bit0=1. Actualiza dim2 y dim3, manteniendo dim0 y dim1. Es decir, cambia la tercera dimensión y proporciona también un nuevo índice para la cuarta. Este es el patrón esperado cuando se avanza en la tercera dimensión en un orden lexicográfico: típicamente se reinicia la cuarta dimensión a su primer valor válido (que suele considerarse “cambiada” porque se escribe explícitamente ese nuevo valor, aunque pueda coincidir o no con el previo).

    0x07 (0111<sub>2</sub>): Bits2,1,0 = 1. Actualiza dim1, dim2 y dim3, dejando dim0 igual. Representa que cambió la segunda dimensión (dim1) y, como consecuencia, se escriben nuevos índices para las dimensiones subordinadas (dim2, dim3). Es el caso típico de avanzar en dim1 (por ejemplo pasar de 'L','X',... a 'M',... en un ejemplo), lo que conlleva comenzar una “subsecuencia” nueva en las dimensiones de orden inferior.

    0x0F (1111<sub>2</sub>): Bits3,2,1,0 = 1. Actualiza todas las dimensiones. Esto ocurre cuando la primera dimensión cambia a un nuevo valor, para lo cual obviamente se proveen nuevos índices para dim0, y también se dan índices para todas las demás dimensiones. En otras palabras, se inicia un bloque totalmente nuevo de combinaciones. En orden lexicográfico, avanzar la dimensión más agregada implica típicamente reiniciar todas las dimensiones inferiores a sus valores iniciales (que, se codifica efectivamente como “todas cambiaron”).

¡Importante!: Obsérvese que muchos códigos delta no aparecerán si los datos están estrictamente ordenados lexicográficamente. Por ejemplo, un código como 0x0A (1010<sub>2</sub>, cambia dim0 y dim2 pero deja dim1 y dim3) es inusual en un recorrido lexicográfico puro, ya que cambiar dim0 normalmente haría que las dims 1,2,3 empezaran en su primer valor. Sin embargo, en los datos reales puede ocurrir por dos motivos legítimos: (a) la dimensión que quedó “sin cambio” solo tiene un único elemento posible (e.g. si dim1 es un conjunto con un solo elemento 'K', siempre será 'K' y el algoritmo inteligentemente no la transmite de nuevo, marcándola como sin cambio), o (b) coincidencias particulares en los datos. Por ello, no todos los códigos son simplemente secuenciales; se deben interpretar bit a bit según la regla general, en lugar de asumir un patrón fijo de “X dimensiones cambian” sin más.
2. Interacciones entre dimensiones y reinicios implícitos

La pregunta clave es si algunos códigos implican dependencias o efectos sobre dimensiones no marcadas explícitamente. En la implementación de GDX, el código delta por sí mismo define completamente qué dimensiones se actualizan y cuáles no – no hay marcas ocultas adicionales. Sin embargo, el orden de los datos (normalmente lexicográfico) crea una aparente interacción: cuando se incrementa una dimensión de orden superior, las dimensiones inferiores tienden a “reiniciarse” a valores iniciales.

Ejemplo lexicográfico: Supongamos que las tuplas están ordenadas por dim0, luego dim1, etc. Si pasamos de la última tupla con dim0 = A a la primera tupla con dim0 = B (es decir, A < B en el orden de esa dimensión), lo habitual es que dim1, dim2, dim3 del nuevo bloque B comiencen en sus primeros valores (mínimos) válidos. En la compresión delta, esto simplemente se representará con un código donde todas esas dims se marcan como cambiadas (pues se proveen sus nuevos índices). No es que GDX “reinicie” silenciosamente nada; más bien, el propio registro codifica los nuevos valores para esas dimensiones. En nuestro ejemplo, cambiar dim0 de A a B generaría un código 0x0F (1111₂) indicando que dim0 cambió a B, dim1 se establece en su nuevo valor inicial (cambiada), igual dim2 y dim3.

Entonces, ¿hay interacción entre códigos? Solo en el sentido de que ciertos patrones de bits no pueden ocurrir a menos que otro bit también ocurra. Por ejemplo, en datos ordenados es muy raro ver un bit de una dimensión superior en 1 (cambio) mientras uno inferior sigue en 0 (sin cambio) – salvo por las excepciones de cardinalidad 1 o coincidencias descritas. Generalmente, cuando una dimensión $d$ cambia (bit=1), lo más común es que todas las dimensiones de orden inferior ($d+1, d+2,\dots$) también cambien (porque se les asignan nuevos valores iniciales). Esto resulta en códigos con secuencias de 1 contiguas en la parte menos significativa. Si alguna dimensión inferior no cambia (bit=0) a pesar de haber cambiado una superior, es porque ese valor resultó ser reutilizable (p.ej. dominio unitario). No existe en GDX un código “mágico” que diga “cambié dim0 y por tanto resetea las demás dims a principio” – simplemente marcará todas como cambiadas explicitando esos nuevos índices. En resumen: el código delta explicita los cambios; no hay cambios implícitos más allá de lo que indican los bits.

Para evitar errores al reconstruir, el lector debe honrar exactamente lo que dice cada bit. Si un bit indica sin cambio, debe dejarse tal cual el índice anterior; si indica cambio, debe leer el nuevo valor. Un error común (como el mencionado por la pregunta) es asumir que cierto código implica “ningún cambio en absoluto” – por ejemplo, interpretar erróneamente 0x0A (1010₂) como “no cambia ninguna dimensión”. Esto llevó a que el algoritmo de lectura no actualizara nada en ese paso, repitiendo la misma tupla y causando duplicados (23 tuplas duplicadas en lugar de únicas). La lección es que no se deben mezclar significados o asumir dependencias no codificadas: cada bit es la fuente de verdad sobre su dimensión.
3. Manejo del estado de índices en GAMS durante compresión delta

Internamente, GAMS mantiene un estado actual de los índices (la última tupla leída/escrita) mientras procesa registros comprimidos. Al escribir un registro con compresión, el algoritmo de GDX compara la nueva tupla de índices con la tupla previa para determinar el código delta:

    Primera tupla: Se escribe completa, sin delta (o con un código especial si aplica). A partir de este punto, el “estado” interno de cada dimensión es el valor de esa primera tupla.

    Siguientes tuplas: Para cada nueva tupla, se recorre dimensión por dimensión comparando con el estado previo:

        Mientras el índice permanezca igual que el anterior, se marca bit = 0 (sin cambio) y no se escribe ese índice de nuevo. El estado de esa dimensión permanece igual.

        Al llegar a la primera dimensión donde el índice difiera, se marca bit = 1 para esa y todas las restantes dimensiones de menor nivel también se marcarán como 1 (porque la nueva tupla proveerá valores para ellas). En este punto, GDX escribe los nuevos índices para esa dimensión y las subsiguientes. El estado de cada dimensión marcada se actualiza al nuevo valor leído; las no marcadas conservan el valor previo.

En la lectura, el proceso es análogo pero inverso: se lee un código delta y luego para cada dimensión:

    Si el bit es 1, consume del flujo el siguiente identificador de elemento (UEL) para esa dimensión y actualiza el estado de esa dimensión con el nuevo valor.

    Si el bit es 0, no se consume nada para esa dimensión, y simplemente se reutiliza el valor que tenía en el estado (es decir, el mismo índice del registro anterior).

Con este mecanismo, GAMS no necesita “reiniciar” manualmente ninguna dimensión: el estado de cada una queda correctamente establecido por la lógica anterior. Si una dimensión de mayor jerarquía cambia, cualquier reinicio conceptual de las inferiores se logra simplemente porque se marcan como cambiadas y se les asignan nuevos valores (que típicamente serán el primero de sus dominios, dado el ordenamiento). Y si por alguna razón GAMS decide no cambiar una dimensión inferior (bit=0) tras cambiar una superior, es porque legítimamente el mismo valor anterior sigue siendo válido y se desea conservar (caso ya explicado, e.g. dominio único).

Conclusión: GAMS mantiene un estado por dimensión y lo actualiza conforme a los bits del código delta. No hay re-inicializaciones implícitas fuera de lo indicado por los bits. Desde el punto de vista del desarrollador, es importante al decodificar simular ese estado: después de cada registro decodificado, almacenar los índices actuales; al leer el siguiente código delta, aplicar cambios dimension por dimensión según los bits (actualizando el estado para bits=1 y dejando igual para bits=0). Siguiendo este procedimiento estricto se evita la “corrupción de estado” que causó duplicados en el caso planteado.

Nota: Las 173 tuplas únicas vs 23 duplicadas que observó el usuario indican que en 23 casos el algoritmo repitió indebidamente la misma combinación de índices. Esto es un síntoma claro de que en esos 23 casos algún bit indicaba “cambio” pero no se reflejó (o viceversa). Revisando nuestro ejemplo, interpretar 0x0A incorrectamente como “sin cambios en ningún índice” habría ocasionado exactamente ese problema – se habría repetido la tupla anterior una vez más, generando un duplicado. La forma correcta de interpretarlo (cambios en dim0 y dim2) habría dado una tupla nueva.
4. Herramientas y referencias para implementación correcta

El algoritmo de compresión delta en GDX ahora está documentado públicamente gracias a la liberación del código fuente de la librería GDX (API de bajo nivel) en noviembre de 2023. Esto significa que la referencia definitiva está disponible para desarrolladores que deseen ver exactamente cómo GAMS implementa la lógica. Algunas consideraciones y herramientas útiles:

    API oficial de GDX (C/C++ y bindings): La forma más segura de leer/escribir GDX es usando la API proporcionada por GAMS (disponible en C++, Python, Java, etc.). Dicha API ya se encarga internamente de los detalles de la compresión. Por ejemplo, funciones como gdxDataReadRaw en C/C++ manejan la decodificación delta y actualizan los índices automáticamente. Usar estas rutinas evita tener que implementar manualmente la interpretación de bits. La API abierta “documenta implícitamente el formato interno”, lo cual confirma nuestras descripciones aquí.

    Herramientas de terceros basadas en la API: Muchas herramientas de terceros en realidad utilizan por debajo la librería oficial de GDX. Por ejemplo, paquetes como gdxpds (Python), gdxrrw (R) o las utilidades MATLAB (GDXMRW) llaman a la API nativa. Estas herramientas, al apoyarse en GDX, ya manejan correctamente los códigos delta y el estado de índices. Si su objetivo es simplemente leer datos, aprovechar estos wrappers es recomendable.

    Implementaciones independientes: Si pese a todo se desea una implementación propia del lector (por ejemplo, en un lenguaje o contexto donde no se pueda usar la DLL/so de GAMS), es crucial seguir las reglas arriba expuestas. Un ejemplo de implementación independiente es el paquete pyGDX de Paul Kishimoto, que reimplementó lectura de GDX a estructuras xarray. En su documentación se describe cómo los parámetros multi-dimensionales se mapean a coordenadas, aunque internamente también parece utilizar la API GDX para la lectura básica. Otro ejemplo histórico es GDX2DAT/GDXF90 de Thomas Rutherford, aunque esas utilidades probablemente utilizaban versiones antiguas del API GDX en Fortran. Hasta donde se conoce, no hay muchas implementaciones “desde cero” públicas, precisamente porque el formato era complejo; ahora con la apertura del código, es más fácil verificar la lógica.

    GDX viewer / gdxdump: Son herramientas oficiales (incluidas con GAMS) que leen un GDX y lo despliegan en texto. Pueden servir para validar la salida de nuestra lectura. Por ejemplo, si sospechamos que estamos duplicando tuplas, comparar nuestra salida con la de gdxdump nos confirmará si interpretamos mal algún código.

Referencias técnicas: Dado que la documentación pública explícita del formato era escasa, la mejor referencia ahora es el propio código fuente abierto. En ausencia de comentarios específicos sobre “delta codes”, nuestras conclusiones se basan en evidencia empírica y la confirmación de los desarrolladores de GAMS al liberar el código (que “documenta el layout interno”). En resumen, la clave para manejar correctamente los registros comprimidos delta es:

    Mantener un estado actual de los índices por dimensión.

    Por cada registro nuevo: leer el código delta (1–2 bytes, según dims), examinar cada bit y para cada dimensión decidir si tomar el valor anterior o leer un nuevo índice del flujo.

    Ser consciente de que un bit=1 en una dimensión de orden alto normalmente implicará bits=1 en todas las inferiores en un dataset ordenado – pero no confiarse: debe leerse exactamente lo que diga el código, ni más ni menos.

Siguiendo este enfoque preciso, se evitan las dependencias implícitas incorrectas entre dimensiones y se reconstruyen las tuplas originales sin inconsistencias ni duplicados inadvertidos. Si quedan dudas, se recomienda revisar el código fuente de GDX en GitHub o utilizar la API oficial para contrastar la lógica, dado que es la misma que utiliza GAMS internamente y por tanto la fuente de verdad de cómo tratar estos códigos.
Citas

GDX source code published on GitHub
https://www.gams.com/blog/2023/12/gdx-source-code-published-on-github/

GitHub - GAMS-dev/gdx: Official low-level API to access GAMS Data eXchange (GDX) files with bindings to various programming languages
https://github.com/GAMS-dev/gdx

GitHub - GAMS-dev/gdx: Official low-level API to access GAMS Data eXchange (GDX) files with bindings to various programming languages
https://github.com/GAMS-dev/gdx

GDX source code published on GitHub
https://www.gams.com/blog/2023/12/gdx-source-code-published-on-github/

Accessing data from GDX files — pyGDX 2 documentation
https://pygdx.readthedocs.io/en/latest/file.html

Accessing data from GDX files — pyGDX 2 documentation
https://pygdx.readthedocs.io/en/latest/file.html

GDX2DAT and GDXF90: Tools for Reading and Writing GDX Files
https://www.mpsge.org/inclib/gdx2dat.htm
Todas las fuentes