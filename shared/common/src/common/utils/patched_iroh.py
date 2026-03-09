import struct
from iroh.iroh_ffi import _UniffiRustBufferBuilder
import ctypes


def patched_write(self, value):
    length = len(value)
    with self._reserve(length):
        if length > 0:
            ctypes.memmove(ctypes.addressof(self.rbuf.data.contents) + self.rbuf.len, value, length)


def _pack_into(self, size, format, value):
    with self._reserve(size):
        packed = struct.pack(format, value)
        if size > 0:
            ctypes.memmove(ctypes.addressof(self.rbuf.data.contents) + self.rbuf.len, packed, size)


_UniffiRustBufferBuilder.write = patched_write
_UniffiRustBufferBuilder._pack_into = _pack_into
