"""
Implementación mejorada: Detección automática de progresión aritmética vs geométrica.

Basado en el análisis del formato GDX, sabemos que:
1. GDX NO marca explícitamente el tipo de progresión
2. El marcador 0x09 solo indica "valor omitido"
3. Necesitamos detectar heurísticamente el patrón

Estrategia:
- Si tenemos 3+ valores explícitos, podemos detectar el patrón
- Verificamos si las diferencias son constantes (aritmética)
- Verificamos si los ratios son constantes (geométrica)
- Para 2 valores, intentamos ambos y elegimos el que tiene menos error
"""

def detect_sequence_type(values: list[tuple[int, float]]) -> tuple[str, float]:
    """
    Detecta si una secuencia es aritmética o geométrica.
    
    Args:
        values: Lista de (índice, valor) pares.
        
    Returns:
        ("arithmetic"|"geometric", parameter) donde parameter es:
        - delta para aritmética
        - ratio para geométrica
    """
    if len(values) < 2:
        return ("arithmetic", 0.0)
    
    if len(values) == 2:
        # Solo 2 valores: caso más difícil de detectar
        idx1, val1 = values[0]
        idx2, val2 = values[1]
        gap = idx2 - idx1
        
        if gap == 0 or val1 == 0:
            return ("arithmetic", 0.0)
        
        # Calcular ambos parámetros
        delta = (val2 - val1) / gap
        ratio = (val2 / val1) ** (1.0 / gap) if abs(val1) > 1e-10 else 1.0
        
        # REGLA 1: Potencias de 2 exactas → geométrica
        # Ejemplo: 10→160 en 4 pasos = ratio 2.0
        if abs(ratio - round(ratio)) < 0.01 and round(ratio) >= 2:
            return ("geometric", ratio)
        
        # REGLA 2: Verificar si val2 = val1 * 2^n (potencia exacta)
        if val1 > 0 and val2 > 0:
            try:
                log_ratio = math.log2(val2 / val1)
                if abs(log_ratio - round(log_ratio)) < 0.01:
                    # Es potencia de 2 exacta
                    return ("geometric", ratio)
            except:
                pass
        
        # REGLA 3: Para valores grandes (>50), si delta es grande, es aritmética
        # Ejemplo: 100→500 en 4 pasos = delta 100 por paso
        if abs(val1) >= 50:
            avg_val = (abs(val1) + abs(val2)) / 2
            if abs(delta) >= avg_val * 0.5:  # Delta > 50% del promedio
                return ("arithmetic", delta)
        
        # REGLA 4: Ratio cercano a 1 (<15% cambio/paso) → aritmética
        if 0.85 < ratio < 1.15:
            return ("arithmetic", delta)
        
        # REGLA 5: Para valores pequeños (<50), ratio significativo → geométrica
        if abs(val1) < 50 and (ratio > 1.3 or ratio < 0.77):
            return ("geometric", ratio)
        
        # DEFAULT: usar cambio relativo como criterio
        rel_change_ratio = abs((val2 - val1) / val1)  # Cambio total relativo
        
        # Si el cambio total es > 100% pero el ratio/paso es moderado, probablemente es aritmética
        if rel_change_ratio > 1.0 and ratio < 1.6:
            return ("arithmetic", delta)
        
        # En duda, preferir geométrica para ratios > 1.2
        if ratio > 1.2:
            return ("geometric", ratio)
        
        return ("arithmetic", delta)
    
    # 3+ valores: podemos verificar directamente
    # Probar aritmética
    diffs = []
    for i in range(1, len(values)):
        idx1, val1 = values[i - 1]
        idx2, val2 = values[i]
        gap = idx2 - idx1
        if gap > 0:
            diff_per_step = (val2 - val1) / gap
            diffs.append(diff_per_step)
    
    # Probar geométrica
    ratios = []
    for i in range(1, len(values)):
        idx1, val1 = values[i - 1]
        idx2, val2 = values[i]
        gap = idx2 - idx1
        if gap > 0 and val1 != 0 and abs(val1) > 1e-10:
            ratio_per_step = (val2 / val1) ** (1.0 / gap)
            ratios.append(ratio_per_step)
    
    # Calcular variación en diffs y ratios
    if diffs:
        avg_diff = sum(diffs) / len(diffs)
        var_diff = sum((d - avg_diff) ** 2 for d in diffs) / len(diffs)
        rel_var_diff = var_diff / (avg_diff ** 2) if avg_diff != 0 else float('inf')
    else:
        rel_var_diff = float('inf')
        avg_diff = 0.0
    
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        var_ratio = sum((r - avg_ratio) ** 2 for r in ratios) / len(ratios)
        rel_var_ratio = var_ratio / (avg_ratio ** 2) if avg_ratio != 0 else float('inf')
    else:
        rel_var_ratio = float('inf')
        avg_ratio = 1.0
    
    # Decidir basado en variación relativa
    if rel_var_diff < rel_var_ratio * 0.5:  # Aritmética es más consistente
        return ("arithmetic", avg_diff)
    elif rel_var_ratio < rel_var_diff * 0.5:  # Geométrica es más consistente
        return ("geometric", avg_ratio)
    else:
        # Ambos son similares, usar heurística adicional
        if abs(avg_ratio - 1.0) < 0.01:
            return ("arithmetic", avg_diff)
        else:
            return ("geometric", avg_ratio)


def interpolate_arithmetic(idx: int, values: list[tuple[int, float]], delta: float) -> float:
    """Interpolación aritmética."""
    # Encontrar valores antes y después
    before = [v for v in values if v[0] < idx]
    after = [v for v in values if v[0] > idx]
    
    if before and after:
        # Interpolar entre dos valores
        idx1, val1 = before[-1]
        idx2, val2 = after[0]
        gap = idx2 - idx1
        step = (idx - idx1)
        return val1 + (val2 - val1) * step / gap
    elif before:
        # Extrapolar hacia adelante
        idx1, val1 = before[-1]
        return val1 + (idx - idx1) * delta
    elif after:
        # Extrapolar hacia atrás
        idx1, val1 = after[0]
        return val1 - (idx1 - idx) * delta
    else:
        return 0.0


def interpolate_geometric(idx: int, values: list[tuple[int, float]], ratio: float) -> float:
    """Interpolación geométrica."""
    # Encontrar valores antes y después
    before = [v for v in values if v[0] < idx]
    after = [v for v in values if v[0] > idx]
    
    if before and after:
        # Interpolar geométricamente entre dos valores
        idx1, val1 = before[-1]
        idx2, val2 = after[0]
        gap = idx2 - idx1
        step = idx - idx1
        if val1 != 0:
            local_ratio = (val2 / val1) ** (1.0 / gap)
            return val1 * (local_ratio ** step)
        else:
            # Fallback a aritmética si val1 es 0
            return val1 + (val2 - val1) * step / gap
    elif before:
        # Extrapolar hacia adelante geométricamente
        idx1, val1 = before[-1]
        steps = idx - idx1
        return val1 * (ratio ** steps)
    elif after:
        # Extrapolar hacia atrás geométricamente
        idx1, val1 = after[0]
        steps = idx1 - idx
        return val1 / (ratio ** steps)
    else:
        return 0.0


def reconstruct_sequence(
    stored_values: list[tuple[int, float]],
    expected_length: int
) -> dict[int, float]:
    """
    Reconstruye una secuencia completa con detección automática de tipo.
    
    Args:
        stored_values: Lista de (índice, valor) para valores almacenados explícitamente.
        expected_length: Longitud esperada de la secuencia.
        
    Returns:
        Diccionario {índice: valor} con todos los valores reconstruidos.
    """
    if not stored_values:
        return {}
    
    # Detectar tipo de secuencia
    seq_type, parameter = detect_sequence_type(stored_values)
    
    print(f"  Tipo detectado: {seq_type}")
    print(f"  Parámetro: {parameter:.6f}")
    
    # Reconstruir valores
    result = {}
    
    # Añadir valores explícitos
    for idx, val in stored_values:
        result[idx] = val
    
    # Interpolar valores faltantes
    for idx in range(expected_length):
        if idx not in result:
            if seq_type == "arithmetic":
                result[idx] = interpolate_arithmetic(idx, stored_values, parameter)
            else:  # geometric
                result[idx] = interpolate_geometric(idx, stored_values, parameter)
    
    return result


# Test del detector
def test_detection():
    """Prueba el detector con casos conocidos."""
    print("=" * 80)
    print("TEST: Detección de tipo de secuencia")
    print("=" * 80)
    
    test_cases = [
        # (nombre, [(idx, val), ...], tipo_esperado)
        ("Aritmética simple", [(0, 10), (1, 20), (2, 30), (3, 40)], "arithmetic"),
        ("Geométrica simple", [(0, 2), (1, 4), (2, 8), (3, 16)], "geometric"),
        ("Aritmética 2 valores", [(0, 100), (4, 500)], "arithmetic"),
        ("Geométrica 2 valores", [(0, 10), (4, 160)], "geometric"),  # 10 * 2^4
        ("Growth 5%", [(0, 1.0), (1, 1.05), (2, 1.1025)], "geometric"),
        ("Linear growth", [(0, 100), (5, 125), (10, 150)], "arithmetic"),
    ]
    
    for name, values, expected in test_cases:
        print(f"\n{name}:")
        print(f"  Valores: {values}")
        seq_type, param = detect_sequence_type(values)
        status = "✓" if seq_type == expected else "✗"
        print(f"  {status} Detectado: {seq_type} (parámetro: {param:.6f})")
        print(f"    Esperado: {expected}")
        
        # Reconstruir secuencia completa
        if len(values) >= 2:
            max_idx = max(v[0] for v in values)
            reconstructed = reconstruct_sequence(values, max_idx + 1)
            print(f"  Secuencia reconstruida:")
            for i in sorted(reconstructed.keys()):
                marker = "*" if any(v[0] == i for v in values) else " "
                print(f"    [{i}] {marker} {reconstructed[i]:10.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_detection()
