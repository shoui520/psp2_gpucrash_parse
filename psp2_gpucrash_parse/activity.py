from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .core import ParseError


GPU_ACTIVITY_FORMAT_VERSION = 8
GPU_ACTIVITY_BUFFER_VERSION = 0x65
GPU_ACTIVITY_BUFFER_SIZE = 0x810
GPU_ACTIVITY_SLOT_COUNT = 8
GPU_ACTIVITY_SLOT_SIZE = 0x100


def _c_string(data: bytes) -> str:
    return data.split(b"\0", 1)[0].decode("utf-8", errors="replace")


@dataclass(frozen=True)
class GpuActivitySlot:
    """One firmware activity-ring slot.

    Only fields whose placement is established by the 3.68 producer are
    named.  The remaining bytes stay available for future event decoders.
    """

    index: int
    event_count: int
    tag: int
    name: str
    metadata_words: Tuple[int, ...]
    raw: bytes

    @classmethod
    def parse(cls, index: int, data: bytes) -> "GpuActivitySlot":
        if len(data) != GPU_ACTIVITY_SLOT_SIZE:
            raise ParseError("GPU activity slot is not 0x100 bytes")
        event_count, tag = struct.unpack_from("<2I", data)
        metadata = struct.unpack_from("<6I", data, 0x28)
        return cls(index, event_count, tag, _c_string(data[8:0x28]), metadata, data)

    @property
    def populated(self) -> bool:
        return self.event_count != 0 or self.tag != 0 or bool(self.name)

    def summary(self, include_raw: bool = False) -> dict:
        result = {
            "index": self.index,
            "populated": self.populated,
            "event_count": self.event_count,
            "tag": self.tag,
            "name": self.name,
            "metadata_words": list(self.metadata_words),
            "sha256": hashlib.sha256(self.raw).hexdigest(),
        }
        if include_raw:
            result["raw_hex"] = self.raw.hex()
        return result


@dataclass(frozen=True)
class GpuActivityBuffer:
    format_version: int
    event_count: int
    declared_size: int
    header_word3: int
    slots: Tuple[GpuActivitySlot, ...]
    raw: bytes

    @classmethod
    def parse(cls, data: bytes) -> "GpuActivityBuffer":
        if len(data) != GPU_ACTIVITY_BUFFER_SIZE:
            raise ParseError(
                f"GPU activity buffer must be 0x{GPU_ACTIVITY_BUFFER_SIZE:x} bytes, "
                f"has 0x{len(data):x}"
            )
        version, event_count, declared_size, word3 = struct.unpack_from("<4I", data)
        if version != GPU_ACTIVITY_BUFFER_VERSION:
            raise ParseError(f"unsupported GPU activity buffer version 0x{version:x}")
        if declared_size != GPU_ACTIVITY_BUFFER_SIZE:
            raise ParseError(
                f"GPU activity buffer declares 0x{declared_size:x} bytes, "
                f"expected 0x{GPU_ACTIVITY_BUFFER_SIZE:x}"
            )
        slots = tuple(
            GpuActivitySlot.parse(
                index,
                data[
                    0x10 + index * GPU_ACTIVITY_SLOT_SIZE :
                    0x10 + (index + 1) * GPU_ACTIVITY_SLOT_SIZE
                ],
            )
            for index in range(GPU_ACTIVITY_SLOT_COUNT)
        )
        return cls(version, event_count, declared_size, word3, slots, data)

    def summary(self, include_raw: bool = False) -> dict:
        result = {
            "format_version": self.format_version,
            "event_count": self.event_count,
            "declared_size": self.declared_size,
            "header_word3": self.header_word3,
            "slots": [slot.summary(include_raw=include_raw) for slot in self.slots],
            "sha256": hashlib.sha256(self.raw).hexdigest(),
        }
        if include_raw:
            result["raw_hex"] = self.raw.hex()
        return result


@dataclass(frozen=True)
class GpuActivityInfo:
    """Decoded version-8 ``GPU_ACT_INFO`` note.

    Firmware 3.68 writes a fixed three-word header followed by either no
    activity buffer or one 0x810-byte version-0x65 ring snapshot.
    """

    format_version: int
    sequence: int
    payload_size: int
    activity_buffer: Optional[GpuActivityBuffer]
    trailing: bytes

    @classmethod
    def parse(cls, data: bytes) -> "GpuActivityInfo":
        if len(data) < 12:
            raise ParseError("GPU_ACT_INFO is shorter than its three-word header")
        version, sequence, payload_size = struct.unpack_from("<3I", data)
        if version != GPU_ACTIVITY_FORMAT_VERSION:
            raise ParseError(f"unsupported GPU_ACT_INFO version {version}")
        payload_end = 12 + payload_size
        if payload_end > len(data):
            raise ParseError(
                f"GPU_ACT_INFO declares {payload_size} payload bytes, "
                f"only {len(data) - 12} remain"
            )
        payload = data[12:payload_end]
        activity_buffer = GpuActivityBuffer.parse(payload) if payload else None
        return cls(version, sequence, payload_size, activity_buffer, data[payload_end:])

    def summary(self, include_raw: bool = False) -> dict:
        return {
            "format_version": self.format_version,
            "sequence": self.sequence,
            "payload_size": self.payload_size,
            "trailing_size": len(self.trailing),
            "activity_buffer": (
                self.activity_buffer.summary(include_raw=include_raw)
                if self.activity_buffer is not None
                else None
            ),
        }
