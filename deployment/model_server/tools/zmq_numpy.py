"""Shared HEX ZMQ codec, independent of the optional msgpack-numpy package.

Write msgpack-numpy-compatible numeric arrays/scalars and also read the older
bundled __ndarray__/__npgeneric__ format. Never deserialize pickle/object data.
The separate websocket codec retains its existing wire format.
"""
import functools

import msgpack
import numpy as np


def _numeric_dtype(descr):
    dtype = np.dtype(descr)
    if dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype


def _encode(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        dtype = _numeric_dtype(obj.dtype)
        result = {b"nd": isinstance(obj, np.ndarray), b"type": dtype.str, b"data": obj.tobytes()}
        if isinstance(obj, np.ndarray):
            result.update({b"kind": b"", b"shape": obj.shape})
        return result
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def _decode(obj):
    if b"nd" in obj:
        if obj.get(b"kind") in (b"O", b"V"):
            raise ValueError("Object and structured NumPy data are not supported")
        dtype = _numeric_dtype(obj[b"type"])
        if obj[b"nd"]:
            return np.ndarray(buffer=obj[b"data"], dtype=dtype, shape=obj[b"shape"])
        return np.frombuffer(obj[b"data"], dtype=dtype, count=1)[0]
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=_numeric_dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return _numeric_dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_encode)
unpackb = functools.partial(msgpack.unpackb, object_hook=_decode)
