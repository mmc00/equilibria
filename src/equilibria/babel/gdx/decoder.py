"""GDX Parameter Delta Decoder.

This module implements the delta decoding algorithm for GDX parameters.

ATTRIBUTION
-----------
The algorithm encoded here is a Python re-expression of the delta-decode
logic in `gdx/src/gxfile.cpp` from the official GAMS GDX API:

    https://github.com/GAMS-dev/gdx
    Copyright (c) 2017-2026 GAMS Software GmbH <support@gams.com>
    Copyright (c) 2017-2026 GAMS Development Corp. <support@gams.com>
    Licensed under the MIT License.

The upstream MIT license is compatible with `equilibria`'s own MIT license;
the only requirement is preservation of the copyright notice above. This file
therefore is NOT a clean-room implementation — see the top-level NOTICE for
the distinction. All other files in `equilibria.babel.gdx` are clean-room.

Delta Encoding Scheme for Parameters (OFFICIAL IMPLEMENTATION):

1. Data Header (after _DATA_ marker):
   - 1 byte: dimension
   - 4 bytes: record count (int32)
   - For each dimension D:
     - 4 bytes: MinElem[D] (int32)
     - 4 bytes: MaxElem[D] (int32)

2. DeltaForRead:
   - For version > 6: DeltaForRead = dimension
   - For version <= 6: DeltaForRead = MaxDimV148

3. Record Structure:
   - 1 byte: delta code (B)
   - If B > DeltaForRead:
     - Relative change in last dimension: LastElem[last] += B - DeltaForRead
     - B == 255: EOF marker
   - If B <= DeltaForRead:
     - AFDim = B (1-based first dimension that changes)
     - Read indices for dimensions AFDim-1 to dimension-1
     - Each index value is read based on ElemType and added to MinElem
   - After indices: read value

4. Element Types (ElemType):
   Determined by range (MaxElem - MinElem + 1):
   - Range <= 255: sz_byte (1 byte)
   - Range <= 65535: sz_word (2 bytes)
   - Otherwise: sz_integer (4 bytes)

5. Value Reading:
   - 1 byte: special value type
   - If type == 0 (vm_normal): 8 bytes double
   - Otherwise: mapped special value
"""

import logging
import struct

logger = logging.getLogger(__name__)


def get_integer_size(range_size: int) -> int:
    """
    Determine the size of integer needed based on range.

    Args:
        range_size: MaxElem - MinElem + 1

    Returns:
        Number of bytes: 1, 2, or 4
    """
    if range_size <= 0:
        return 4
    if range_size <= 255:
        return 1
    if range_size <= 65535:
        return 2
    return 4


def decode_parameter_delta(
    section: bytes,
    elements: list[str],
    dimension: int,
) -> dict[tuple[str, ...], float]:
    """
    Decode GDX parameter with delta compression.

    This implementation follows the official GDX format specification:
    - Reads MinElem/MaxElem for each dimension from header
    - Determines element type based on range
    - Applies delta encoding rules correctly
    - Adds MinElem offset to read indices

    Args:
        section: Raw bytes of _DATA_ section
        elements: UEL elements list
        dimension: Number of dimensions

    Returns:
        Dictionary mapping index tuples to float values
    """
    values: dict[tuple[str, ...], float] = {}
    pos = 6  # Skip _DATA_ marker (6 bytes)

    # Read header
    if pos + 5 > len(section):
        return values

    dim = section[pos]
    pos += 1

    if dim != dimension:
        # Mismatch in dimension
        pass

    # Read record count
    if pos + 4 > len(section):
        return values
    record_count = struct.unpack_from("<I", section, pos)[0]
    pos += 4

    # Read MinElem and MaxElem for each dimension
    min_elem = []
    max_elem = []
    elem_type = []  # 1=byte, 2=word, 4=integer

    for d in range(dimension):
        if pos + 8 > len(section):
            return values
        min_val = struct.unpack_from("<i", section, pos)[0]
        max_val = struct.unpack_from("<i", section, pos + 4)[0]
        min_elem.append(min_val)
        max_elem.append(max_val)

        # Determine element type based on range
        range_size = max_val - min_val + 1
        elem_size = get_integer_size(range_size)
        elem_type.append(elem_size)

        pos += 8

    # DeltaForRead for version > 6
    delta_for_read = dimension

    # Current state of indices (1-based as stored in GDX)
    last_elem = [0] * dimension

    # Read records
    record_num = 0
    while pos < len(section) and record_num < record_count:
        if pos >= len(section):
            break

        b = section[pos]
        pos += 1

        # Check for EOF
        if b == 255:
            break

        if b > delta_for_read:
            # Relative change in last dimension
            if dimension > 0:
                last_elem[dimension - 1] += b - delta_for_read
            af_dim = dimension
        else:
            # B indicates first dimension that changes (1-based)
            af_dim = b
            if af_dim < 1:
                af_dim = 1

            # Read indices for dimensions af_dim-1 to dimension-1
            for d in range(af_dim - 1, dimension):
                if pos >= len(section):
                    break

                # Read based on element type
                if elem_type[d] == 1:
                    # Byte
                    if pos < len(section):
                        idx = section[pos]
                        pos += 1
                    else:
                        break
                elif elem_type[d] == 2:
                    # Word (2 bytes, little-endian)
                    if pos + 2 <= len(section):
                        idx = struct.unpack_from("<H", section, pos)[0]
                        pos += 2
                    else:
                        break
                else:
                    # Integer (4 bytes, little-endian)
                    if pos + 4 <= len(section):
                        idx = struct.unpack_from("<i", section, pos)[0]
                        pos += 4
                    else:
                        break

                # Add MinElem offset
                last_elem[d] = idx + min_elem[d]

        # Read 0x0A marker
        if pos >= len(section) or section[pos] != 0x0A:
            # Not a valid record, skip
            pos += 1
            continue

        pos += 1  # Skip 0x0A

        # Read value (8-byte double, little-endian)
        if pos + 8 <= len(section):
            value = struct.unpack_from("<d", section, pos)[0]
            pos += 8
        else:
            break

        # Create key with element names (convert 1-based to 0-based)
        try:
            key = tuple(elements[i - 1] for i in last_elem if i > 0)
            if len(key) == dimension:
                values[key] = value
                record_num += 1
        except IndexError:
            # Skip invalid indices
            pass

    return values
