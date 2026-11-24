"""Hand-written GGUF v3 reader/writer (no dependency on the `gguf` package)."""
import struct
from dataclasses import dataclass, field

import numpy as np

from . import quant

GGUF_MAGIC = 0x46554747  # "GGUF"
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
_SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f",
           BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}


def _pack_str(s) -> bytes:
    b = s.encode("utf-8") if isinstance(s, str) else bytes(s)
    return struct.pack("<Q", len(b)) + b


class GGUFWriter:
    def __init__(self, path, alignment: int = 32):
        self.path = path
        self.alignment = alignment
        self.kv = []          # (key, bytes of type+value)
        self.tensors = []     # (name, ne, type, data bytes)
        self.add_u32("general.alignment", alignment)

    # -- metadata ---------------------------------------------------------
    def _add(self, key, vtype, payload: bytes):
        self.kv.append((key, struct.pack("<I", vtype) + payload))

    def add_u32(self, k, v): self._add(k, U32, struct.pack("<I", v))
    def add_i32(self, k, v): self._add(k, I32, struct.pack("<i", v))
    def add_f32(self, k, v): self._add(k, F32, struct.pack("<f", v))
    def add_bool(self, k, v): self._add(k, BOOL, struct.pack("<?", v))
    def add_string(self, k, v): self._add(k, STRING, _pack_str(v))

    def add_array_str(self, k, items):
        payload = struct.pack("<IQ", STRING, len(items)) + b"".join(_pack_str(s) for s in items)
        self._add(k, ARRAY, payload)

    def add_array_i32(self, k, items):
        payload = struct.pack("<IQ", I32, len(items)) + np.asarray(items, "<i4").tobytes()
        self._add(k, ARRAY, payload)

    # -- tensors ----------------------------------------------------------
    def add_tensor(self, name: str, arr: np.ndarray, ggml_type: int = quant.F32):
        arr = np.asarray(arr, dtype=np.float32)
        ne = list(reversed(arr.shape))  # GGUF: ne[0] is the contiguous dimension
        if ggml_type in (quant.Q8_0, quant.Q4_0) and ne[0] % quant.QK:
            raise ValueError(f"{name}: ne0={ne[0]} not a multiple of {quant.QK}")
        self.tensors.append((name, ne, ggml_type, quant.quantize(arr, ggml_type)))

    def write(self):
        a = self.alignment
        head = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(self.tensors), len(self.kv))
        head += b"".join(_pack_str(k) + v for k, v in self.kv)
        off = 0
        offsets = []
        for name, ne, t, data in self.tensors:
            offsets.append(off)
            head += _pack_str(name) + struct.pack("<I", len(ne))
            head += b"".join(struct.pack("<Q", n) for n in ne)
            head += struct.pack("<IQ", t, off)
            off += (len(data) + a - 1) // a * a
        head += b"\0" * ((-len(head)) % a)
        with open(self.path, "wb") as f:
            f.write(head)
            for (_, _, _, data) in self.tensors:
                f.write(data)
                f.write(b"\0" * ((-len(data)) % a))


@dataclass
class GGUFTensor:
    name: str
    ne: list
    type: int
    data: bytes

    def to_numpy(self) -> np.ndarray:
        n = int(np.prod(self.ne))
        return quant.dequantize(self.data, self.type, n).reshape(list(reversed(self.ne)))


@dataclass
class GGUFFile:
    version: int
    kv: dict = field(default_factory=dict)
    tensors: dict = field(default_factory=dict)


def read_gguf(path) -> GGUFFile:
    buf = open(path, "rb").read()
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, buf, pos)
        pos += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    def rstr():
        nonlocal pos
        n = take("<Q")
        s = buf[pos:pos + n]
        pos += n
        return s.decode("utf-8", errors="surrogateescape")

    def rval(t):
        if t == STRING:
            return rstr()
        if t == ARRAY:
            et, n = take("<I"), take("<Q")
            return [rval(et) for _ in range(n)]
        return take(_SCALAR[t])

    magic, version, n_t, n_kv = take("<I"), take("<I"), take("<Q"), take("<Q")
    if magic != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    g = GGUFFile(version)
    for _ in range(n_kv):
        k = rstr()
        g.kv[k] = rval(take("<I"))
    infos = []
    for _ in range(n_t):
        name = rstr()
        nd = take("<I")
        ne = [take("<Q") for _ in range(nd)]
        t, off = take("<I"), take("<Q")
        infos.append((name, ne, t, off))
    a = g.kv.get("general.alignment", 32)
    data0 = (pos + a - 1) // a * a
    for name, ne, t, off in infos:
        rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        nb = quant.row_size(t, ne[0]) * rows
        start = data0 + off
        g.tensors[name] = GGUFTensor(name, ne, t, buf[start:start + nb])
    return g
