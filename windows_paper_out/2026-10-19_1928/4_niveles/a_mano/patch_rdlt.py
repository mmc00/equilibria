"""Copy a .prm byte for byte, changing only the int32 value of the RDLT header (2IFULL, 1 element)."""
import struct, sys
src, dst, val = sys.argv[1], sys.argv[2], int(sys.argv[3])
b = bytearray(open(src, "rb").read()); p = 0; recs = []
while p < len(b):
    n = struct.unpack_from("<i", b, p)[0]; recs.append((p + 4, n)); p += 8 + n
for i, (off, n) in enumerate(recs):
    if n == 4 and bytes(b[off:off+4]) == b"RDLT":
        moff, mn = recs[i + 1]; assert bytes(b[moff+4:moff+10]) == b"2IFULL", b[moff:moff+12]
        doff, dn = recs[i + 2]
        old = struct.unpack_from("<i", b, doff + dn - 4)[0]
        print("RDLT data record len", dn, "last int", old)
        struct.pack_into("<i", b, doff + dn - 4, val); break
else: sys.exit("RDLT no encontrado")
open(dst, "wb").write(b)
