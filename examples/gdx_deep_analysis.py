"""
Análisis detallado del marcador 0x09 y patrones de compresión.

Basado en el análisis anterior, investigamos:
1. El marcador 0x09 que aparece en secuencias geométricas
2. Los bytes que rodean este marcador
3. Cómo se codifican los valores omitidos
"""

import struct
from pathlib import Path


def detailed_section_analysis(filepath: Path, name: str) -> None:
    """Análisis byte a byte de la sección _DATA_."""
    print("\n" + "=" * 80)
    print(f"ANÁLISIS DETALLADO: {name}")
    print(f"Archivo: {filepath.name}")
    print("=" * 80)

    data = filepath.read_bytes()

    # Encontrar sección _DATA_
    data_pos = data.find(b"_DATA_")
    if data_pos == -1:
        print("No se encontró marcador _DATA_")
        return

    # Empezar justo después de _DATA_
    pos = data_pos + 6
    section_start = pos

    print(f"\nPosición _DATA_: 0x{data_pos:08x}")
    print(f"Inicio de análisis: 0x{pos:08x}")

    # Leer los primeros bytes de metadata
    print("\nMetadata (primeros 20 bytes):")
    for i in range(20):
        if pos + i < len(data):
            print(f"  +{i:2d} (0x{pos+i:08x}): 0x{data[pos+i]:02x}  {data[pos+i]:3d}  ", end="")
            if 32 <= data[pos + i] < 127:
                print(f"'{chr(data[pos+i])}'")
            else:
                print(".")

    # Avanzar al área de datos (típicamente después de ~19 bytes)
    pos += 19

    print("\n\nANÁLISIS BYTE A BYTE (desde +19):")
    print("-" * 80)

    value_count = 0
    i = 0

    while pos + i < len(data) and i < 200:
        current_pos = pos + i
        byte = data[current_pos]

        # Análisis detallado de cada byte significativo
        if byte == 0x01:  # ROW_START
            print(f"\n0x{current_pos:08x}: [ROW_START] 0x01", end="")
            if pos + i + 1 < len(data):
                next_byte = data[current_pos + 1]
                print(f" - siguiente: 0x{next_byte:02x} ({next_byte})")

        elif byte == 0x02:  # RECORD_TYPE / CONTINUE
            print(f"\n0x{current_pos:08x}: [RECORD] 0x02", end="")
            if pos + i + 1 < len(data):
                next_byte = data[current_pos + 1]
                if next_byte == 0x09:
                    print(" + [COMPRESS!] 0x09 *** MARCADOR GEOMÉTRICO ***")
                    i += 2
                    continue
                elif next_byte == 0x0A:
                    print(" + [DOUBLE] 0x0a")
                    if pos + i + 10 <= len(data):
                        value = struct.unpack_from("<d", data, current_pos + 2)[0]
                        value_count += 1
                        print(f"                   └─> Valor #{value_count}: {value:15.6f}")
                        i += 10
                        continue
                else:
                    print(f" - siguiente: 0x{next_byte:02x}")

        elif byte == 0x03:  # CONTINUE / SPARSE
            print(f"\n0x{current_pos:08x}: [CONTINUE/SPARSE] 0x03")

        elif byte == 0x06:  # BLOCK_6
            print(f"\n0x{current_pos:08x}: [BLOCK_6] 0x06", end="")
            if pos + i + 4 <= len(data):
                next_bytes = data[current_pos + 1 : current_pos + 4]
                print(f" [{' '.join(f'{b:02x}' for b in next_bytes)}]")

        elif byte == 0x08:  # BLOCK_8
            print(f"\n0x{current_pos:08x}: [BLOCK_8] 0x08")

        elif byte == 0x09:  # COMPRESS marker
            print(f"\n0x{current_pos:08x}: [COMPRESS!] 0x09 *** MARCADOR SOLO O GEOMÉTRICO ***")

        elif byte == 0x0A:  # DOUBLE marker
            print(f"\n0x{current_pos:08x}: [DOUBLE] 0x0a", end="")
            if pos + i + 9 <= len(data):
                value = struct.unpack_from("<d", data, current_pos + 1)[0]
                value_count += 1
                print(f" - Valor #{value_count}: {value:15.6f}")
                i += 9
                continue

        i += 1

    print(f"\n\nTotal valores encontrados: {value_count}")
    print("=" * 80)


def compare_compression_markers():
    """Compara los marcadores de compresión entre diferentes tipos."""
    fixtures = Path(__file__).parent.parent / "tests" / "fixtures"

    print("\n" + "=" * 80)
    print("INVESTIGACIÓN: Marcador 0x09 y su significado")
    print("=" * 80)

    print("\n¿Qué es 0x09?")
    print("-" * 80)
    print("Hipótesis basadas en observaciones:")
    print("  1. Aparece MÁS en secuencias geométricas")
    print("  2. Frecuentemente viene después de 0x02")
    print("  3. Podría indicar 'valor interpolado' o 'compresión especial'")
    print("  4. La secuencia '02 09' podría ser un código de compresión")

    # Analizar archivos
    files_to_analyze = [
        ("test_small_arithmetic.gdx", "ARITMÉTICA PEQUEÑA (1,2,3,4,5)"),
        ("test_small_geometric.gdx", "GEOMÉTRICA PEQUEÑA (2,4,8,16,32)"),
        ("test_arithmetic.gdx", "ARITMÉTICA GRANDE (10,20,30...100)"),
        ("test_geometric.gdx", "GEOMÉTRICA GRANDE (1,2,4,8...512)"),
    ]

    for filename, description in files_to_analyze:
        filepath = fixtures / filename
        if filepath.exists():
            detailed_section_analysis(filepath, description)


def extract_compression_context():
    """Extrae el contexto alrededor del marcador 0x09."""
    fixtures = Path(__file__).parent.parent / "tests" / "fixtures"

    print("\n" + "=" * 80)
    print("CONTEXTO ALREDEDOR DEL MARCADOR 0x09")
    print("=" * 80)

    test_files = [
        ("test_small_arithmetic.gdx", "Aritmética"),
        ("test_small_geometric.gdx", "Geométrica"),
        ("test_geometric.gdx", "Geométrica Grande"),
    ]

    for filename, desc in test_files:
        filepath = fixtures / filename
        if not filepath.exists():
            continue

        data = filepath.read_bytes()

        # Buscar todas las ocurrencias de 0x09
        positions = []
        for i in range(len(data)):
            if data[i] == 0x09:
                positions.append(i)

        if positions:
            print(f"\n{desc} ({filename}):")
            print(f"  Marcador 0x09 encontrado en {len(positions)} posiciones")

            for pos in positions[:3]:  # Solo primeras 3
                # Contexto: 10 bytes antes y después
                start = max(0, pos - 10)
                end = min(len(data), pos + 11)
                context = data[start:end]

                print(f"\n  Posición 0x{pos:08x}:")
                print(f"    Contexto: {' '.join(f'{b:02x}' for b in context)}")
                print(f"              {' ' * (3 * (pos - start))}^^")

                # Intentar interpretar
                print("    Interpretación:")
                if pos > 0:
                    print(f"      Byte anterior: 0x{data[pos-1]:02x} ", end="")
                    if data[pos - 1] == 0x02:
                        print("(RECORD)")
                    elif data[pos - 1] == 0x06:
                        print("(BLOCK_6)")
                    else:
                        print()

                if pos < len(data) - 1:
                    print(f"      Byte siguiente: 0x{data[pos+1]:02x} ", end="")
                    if data[pos + 1] == 0x02:
                        print("(RECORD)")
                    elif data[pos + 1] == 0x0A:
                        print("(DOUBLE)")
                    else:
                        print()


def main():
    """Análisis principal."""
    print("\n" + "=" * 80)
    print("INVESTIGACIÓN PROFUNDA: Formato GDX y Compresión")
    print("Enfoque: Entender el marcador 0x09 y su relación con progresiones")
    print("=" * 80)

    compare_compression_markers()
    extract_compression_context()

    print("\n" + "=" * 80)
    print("CONCLUSIONES PRELIMINARES")
    print("=" * 80)
    print("""
1. MARCADOR 0x09:
   - Aparece frecuentemente en secuencias geométricas
   - Generalmente viene después de 0x02 o 0x06
   - Podría indicar 'valor omitido/interpolado'
   
2. PATRÓN '02 09':
   - Muy común en archivos con compresión
   - Posiblemente significa: "siguiente valor debe ser interpolado"
   
3. PATRÓN '06 02 09':
   - Aparece al inicio de secciones comprimidas
   - 0x06 podría ser un indicador de "bloque comprimido"
   
4. DIFERENCIAS OBSERVADAS:
   - Aritmética: Más valores explícitos (más 0x0A)
   - Geométrica: Más marcadores 0x09 (más compresión)
   
5. HIPÓTESIS PRINCIPAL:
   GDX NO distingue entre aritmética/geométrica en el formato.
   GAMS decide cómo comprimir basándose en el patrón de datos.
   El marcador 0x09 simplemente indica "valor omitido" sin especificar
   el tipo de interpolación a usar.
   
   Esto explica por qué no podemos distinguir automáticamente:
   ¡El formato no contiene esta información!
    """)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
