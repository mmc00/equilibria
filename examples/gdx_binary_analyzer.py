"""
GDX Binary Format Analyzer - Deep Investigation Tool

Analiza el formato binario de archivos GDX en detalle para identificar
patrones de compresión y marcadores que distingan entre progresiones
aritméticas y geométricas.
"""

import struct
from pathlib import Path
from typing import Any


def hexdump(data: bytes, start: int = 0, length: int = 256, width: int = 16) -> None:
    """Pretty print hexdump of binary data."""
    for i in range(0, min(length, len(data) - start), width):
        pos = start + i
        hex_part = " ".join(f"{data[pos + j]:02x}" if pos + j < len(data) else "  " for j in range(width))
        ascii_part = "".join(
            chr(data[pos + j]) if 32 <= data[pos + j] < 127 else "."
            for j in range(width)
            if pos + j < len(data)
        )
        print(f"{pos:08x}: {hex_part:<{width*3}} | {ascii_part}")


def find_marker(data: bytes, marker: bytes) -> list[int]:
    """Find all occurrences of a marker in data."""
    positions = []
    pos = 0
    while True:
        pos = data.find(marker, pos)
        if pos == -1:
            break
        positions.append(pos)
        pos += 1
    return positions


def analyze_data_section(data: bytes, start: int, length: int = 200) -> dict[str, Any]:
    """
    Analiza una sección _DATA_ en detalle.
    
    Busca patrones que puedan indicar el tipo de compresión.
    """
    section = data[start : start + length]

    analysis = {
        "start_pos": start,
        "length": length,
        "markers_found": [],
        "double_values": [],
        "byte_patterns": [],
    }

    # Buscar marcadores comunes
    for marker_name, marker_byte in [
        ("ROW_START", 0x01),
        ("CONTINUE", 0x03),
        ("BLOCK_4", 0x04),
        ("BLOCK_6", 0x06),
        ("BLOCK_8", 0x08),
        ("COMPRESS_09", 0x09),
        ("DOUBLE", 0x0A),
    ]:
        positions = [i for i in range(len(section)) if section[i] == marker_byte]
        if positions:
            analysis["markers_found"].append({
                "name": marker_name,
                "byte": f"0x{marker_byte:02x}",
                "count": len(positions),
                "positions": positions[:10],  # Solo primeras 10
            })

    # Extraer valores double
    pos = 19  # Skip header típico
    while pos < len(section) - 8:
        if section[pos] == 0x0A:  # Double marker
            try:
                value = struct.unpack_from("<d", section, pos + 1)[0]
                if -1e15 < value < 1e15 and value == value:  # Valid range and not NaN
                    analysis["double_values"].append({
                        "pos": pos,
                        "value": value,
                    })
                    pos += 9
                    continue
            except struct.error:
                pass
        pos += 1

    # Buscar patrones de bytes antes de doubles
    if len(analysis["double_values"]) >= 2:
        patterns = []
        for i in range(1, len(analysis["double_values"])):
            prev_pos = analysis["double_values"][i - 1]["pos"]
            curr_pos = analysis["double_values"][i]["pos"]
            between = section[prev_pos + 9 : curr_pos]
            if len(between) > 0:
                patterns.append({
                    "length": len(between),
                    "bytes": " ".join(f"{b:02x}" for b in between),
                    "between_values": [
                        analysis["double_values"][i - 1]["value"],
                        analysis["double_values"][i]["value"],
                    ],
                })
        analysis["byte_patterns"] = patterns

    return analysis


def compare_files(file1: Path, file2: Path, name1: str, name2: str) -> None:
    """Compara dos archivos GDX en detalle."""
    print("=" * 80)
    print(f"COMPARACIÓN: {name1} vs {name2}")
    print("=" * 80)

    data1 = file1.read_bytes()
    data2 = file2.read_bytes()

    print(f"\n{name1}: {len(data1)} bytes")
    print(f"{name2}: {len(data2)} bytes")
    print(f"Diferencia de tamaño: {abs(len(data1) - len(data2))} bytes")

    # Encontrar secciones _DATA_
    data_marker = b"_DATA_"
    data1_positions = find_marker(data1, data_marker)
    data2_positions = find_marker(data2, data_marker)

    print(f"\n{name1} - _DATA_ sections: {len(data1_positions)}")
    print(f"{name2} - _DATA_ sections: {len(data2_positions)}")

    # Analizar primera sección _DATA_ de cada archivo
    if data1_positions and data2_positions:
        print(f"\n{'=' * 80}")
        print(f"ANÁLISIS DETALLADO - {name1}")
        print(f"{'=' * 80}")

        # Posición después del marcador
        section1_start = data1_positions[0] + len(data_marker)
        analysis1 = analyze_data_section(data1, section1_start)

        print("\nMarcadores encontrados:")
        for m in analysis1["markers_found"]:
            print(f"  {m['name']:15s} ({m['byte']}): {m['count']} veces en posiciones {m['positions']}")

        print(f"\nValores double extraídos: {len(analysis1['double_values'])}")
        for dv in analysis1["double_values"][:10]:
            print(f"  Pos {dv['pos']:4d}: {dv['value']:15.6f}")

        if analysis1["byte_patterns"]:
            print("\nPatrones de bytes entre valores:")
            for i, pattern in enumerate(analysis1["byte_patterns"][:5]):
                print(f"  Patrón {i+1}:")
                print(f"    Entre {pattern['between_values'][0]:.2f} y {pattern['between_values'][1]:.2f}")
                print(f"    Longitud: {pattern['length']} bytes")
                print(f"    Bytes: {pattern['bytes']}")

        print(f"\n{'-' * 80}\n")

        print(f"ANÁLISIS DETALLADO - {name2}")
        print(f"{'=' * 80}")

        section2_start = data2_positions[0] + len(data_marker)
        analysis2 = analyze_data_section(data2, section2_start)

        print("\nMarcadores encontrados:")
        for m in analysis2["markers_found"]:
            print(f"  {m['name']:15s} ({m['byte']}): {m['count']} veces en posiciones {m['positions']}")

        print(f"\nValores double extraídos: {len(analysis2['double_values'])}")
        for dv in analysis2["double_values"][:10]:
            print(f"  Pos {dv['pos']:4d}: {dv['value']:15.6f}")

        if analysis2["byte_patterns"]:
            print("\nPatrones de bytes entre valores:")
            for i, pattern in enumerate(analysis2["byte_patterns"][:5]):
                print(f"  Patrón {i+1}:")
                print(f"    Entre {pattern['between_values'][0]:.2f} y {pattern['between_values'][1]:.2f}")
                print(f"    Longitud: {pattern['length']} bytes")
                print(f"    Bytes: {pattern['bytes']}")

        # Comparar diferencias
        print(f"\n{'-' * 80}")
        print("DIFERENCIAS CLAVE:")
        print(f"{'-' * 80}")

        # Comparar conteo de marcadores
        markers1 = {m["name"]: m["count"] for m in analysis1["markers_found"]}
        markers2 = {m["name"]: m["count"] for m in analysis2["markers_found"]}

        all_markers = set(markers1.keys()) | set(markers2.keys())
        print("\nConteo de marcadores:")
        for marker in sorted(all_markers):
            count1 = markers1.get(marker, 0)
            count2 = markers2.get(marker, 0)
            diff = "✓ IGUAL" if count1 == count2 else f"✗ DIFERENTE ({count1} vs {count2})"
            print(f"  {marker:15s}: {name1}={count1:2d}, {name2}={count2:2d}  {diff}")

        # Comparar hexdumps lado a lado
        print(f"\n{'=' * 80}")
        print("HEXDUMP COMPARATIVO (primeros 128 bytes de _DATA_)")
        print(f"{'=' * 80}\n")

        print(f"{name1}:")
        hexdump(data1, section1_start, 128)

        print(f"\n{name2}:")
        hexdump(data2, section2_start, 128)


def main():
    """Análisis principal."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"

    print("\n" + "=" * 80)
    print("ANÁLISIS PROFUNDO DEL FORMATO BINARIO GDX")
    print("Objetivo: Identificar diferencias entre compresión aritmética y geométrica")
    print("=" * 80 + "\n")

    # Verificar que los archivos existen
    files_to_check = [
        "test_arithmetic.gdx",
        "test_geometric.gdx",
        "test_small_arithmetic.gdx",
        "test_small_geometric.gdx",
    ]

    for filename in files_to_check:
        filepath = fixtures_dir / filename
        if not filepath.exists():
            print(f"⚠️  Archivo no encontrado: {filename}")
            print("   Ejecuta primero generate_compression_tests.gms\n")
            return

    # Comparación 1: Aritmética completa vs Geométrica completa
    compare_files(
        fixtures_dir / "test_arithmetic.gdx",
        fixtures_dir / "test_geometric.gdx",
        "ARITMÉTICA (10,20,30...)",
        "GEOMÉTRICA (1,2,4,8...)",
    )

    print("\n\n")

    # Comparación 2: Aritmética pequeña vs Geométrica pequeña
    compare_files(
        fixtures_dir / "test_small_arithmetic.gdx",
        fixtures_dir / "test_small_geometric.gdx",
        "ARITMÉTICA PEQUEÑA (1,2,3,4,5)",
        "GEOMÉTRICA PEQUEÑA (2,4,8,16,32)",
    )


if __name__ == "__main__":
    main()
