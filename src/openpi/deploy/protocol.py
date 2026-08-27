"""Length-prefixed JSON header plus raw ndarray blobs over stdio."""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO

import numpy as np

_HEADER = struct.Struct("!Q")
_ARRAY_MARKER = "__ndarray_blob__"


def send_frame(stream: BinaryIO, payload: dict[str, Any]) -> None:
    blobs: list[bytes] = []
    encoded = _encode(payload, blobs)
    header = json.dumps(
        {"payload": encoded, "blob_sizes": [len(blob) for blob in blobs]},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    stream.write(_HEADER.pack(len(header)))
    stream.write(header)
    stream.writelines(blobs)
    stream.flush()


def recv_frame(stream: BinaryIO) -> dict[str, Any]:
    raw_size = _read_exact(stream, _HEADER.size)
    header_size = _HEADER.unpack(raw_size)[0]
    header = json.loads(_read_exact(stream, header_size).decode("utf-8"))
    blobs = [_read_exact(stream, int(size)) for size in header.get("blob_sizes", [])]
    payload = _decode(header["payload"], blobs)
    if not isinstance(payload, dict):
        raise ValueError("protocol payload must be an object")
    return payload


def _encode(value: Any, blobs: list[bytes]) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        index = len(blobs)
        blobs.append(array.tobytes(order="C"))
        return {
            _ARRAY_MARKER: index,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _encode(item, blobs) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item, blobs) for item in value]
    return value


def _decode(value: Any, blobs: list[bytes]) -> Any:
    if isinstance(value, dict) and _ARRAY_MARKER in value:
        array = np.frombuffer(blobs[int(value[_ARRAY_MARKER])], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(int(dim) for dim in value["shape"]))
    if isinstance(value, dict):
        return {key: _decode(item, blobs) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, blobs) for item in value]
    return value


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise EOFError("protocol stream closed")
    return data
