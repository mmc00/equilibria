def _decode_simple_parameter(
    section: bytes,
    elements: list[str],
    dimension: int,
    domain_offsets: list[int],
) -> dict[tuple[str, ...], float]:
    """
    Decoder GDX v2 - Implementación con estado global.
    
    Basado en análisis del formato delta GDX:
    - Mantiene estado de índices [dim0, dim1, dim2, dim3]
    - Interpreta códigos delta (0x01, 0x02, 0x03, 0x05, 0x0A)
    - Actualiza estado según el tipo de código
    """
    values: dict[tuple[str, ...], float] = {}
    
    if len(section) < 20:
        return _decode_simple_parameter_fallback(section, elements, domain_offsets)
    
    # Estado global de índices (0-based)
    current_indices = [0, 0, 0, 0]
    pos = 11  # Después del header _DATA_
    RECORD_DOUBLE = 0x0A
    
    while pos < len(section) - 10:
        byte = section[pos]
        
        # Tipo 0x01: Registro base con índices variables
        if byte == 0x01:
            # Buscar marcador 0x0A
            marker_pos = pos + 1
            while marker_pos < len(section) and section[marker_pos] != RECORD_DOUBLE:
                marker_pos += 1
                if marker_pos - pos > 20:
                    break
            
            if marker_pos >= len(section) or section[marker_pos] != RECORD_DOUBLE:
                pos += 1
                continue
            
            # Bytes entre 0x01 y 0x0A son índices
            idx_bytes = list(section[pos+1:marker_pos])
            
            # Actualizar índices según bytes (1-based, 0=mantener)
            for i, b in enumerate(idx_bytes[:4]):
                if i < 4 and b > 0 and b <= len(elements):
                    current_indices[i] = b - 1
            
            # Leer valor
            value_pos = marker_pos + 1
            if value_pos + 8 <= len(section):
                try:
                    value = struct.unpack_from("<d", section, value_pos)[0]
                    key = tuple(elements[i] for i in current_indices)
                    values[key] = value
                    pos = value_pos + 8
                    continue
                except:
                    pass
            pos += 1
        
        # Tipo 0x02: Actualizar dims 0, 2, 3
        elif byte == 0x02 and pos + 12 <= len(section) and section[pos + 4] == RECORD_DOUBLE:
            idx1, idx3, idx4 = section[pos+1], section[pos+2], section[pos+3]
            
            if idx1 > 0 and idx1 <= len(elements):
                current_indices[0] = idx1 - 1
            if idx3 > 0 and idx3 <= len(elements):
                current_indices[2] = idx3 - 1
            if idx4 > 0 and idx4 <= len(elements):
                current_indices[3] = idx4 - 1
            
            try:
                value = struct.unpack_from("<d", section, pos + 5)[0]
                key = tuple(elements[i] for i in current_indices)
                values[key] = value
                pos += 13
                continue
            except:
                pass
            pos += 1
        
        # Tipo 0x03: Actualizar dims 2, 3
        elif byte == 0x03 and pos + 11 <= len(section) and section[pos + 3] == RECORD_DOUBLE:
            idx3, idx4 = section[pos+1], section[pos+2]
            
            if idx3 > 0 and idx3 <= len(elements):
                current_indices[2] = idx3 - 1
            if idx4 > 0 and idx4 <= len(elements):
                current_indices[3] = idx4 - 1
            
            try:
                value = struct.unpack_from("<d", section, pos + 4)[0]
                key = tuple(elements[i] for i in current_indices)
                values[key] = value
                pos += 12
                continue
            except:
                pass
            pos += 1
        
        # Tipo 0x05/0x06: Incrementar dim 1
        elif byte in (0x05, 0x06) and pos + 10 <= len(section) and section[pos + 1] == RECORD_DOUBLE:
            current_indices[1] = (current_indices[1] + 1) % len(elements)
            
            try:
                value = struct.unpack_from("<d", section, pos + 2)[0]
                key = tuple(elements[i] for i in current_indices)
                values[key] = value
                pos += 10
                continue
            except:
                pass
            pos += 1
        
        # Tipo 0x0A: Sin cambios
        elif byte == RECORD_DOUBLE and pos + 9 <= len(section):
            try:
                value = struct.unpack_from("<d", section, pos + 1)[0]
                key = tuple(elements[i] for i in current_indices)
                values[key] = value
                pos += 9
                continue
            except:
                pass
            pos += 1
        
        else:
            pos += 1
    
    return values
