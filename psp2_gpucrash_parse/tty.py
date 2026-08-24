from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

from .core import ParseError


# ECMA-48/ANSI control-sequence introducers emitted by VitaSDK's console
# logger. Keep this deliberately narrow: arbitrary binary bytes remain visible
# through the CLI's --raw mode instead of being guessed into text.
_ANSI_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TIMESTAMP = re.compile(r"^\[\s*(\d+(?:\.\d+)?)\]\s?(.*)$")


@dataclass(frozen=True)
class TtyLine:
    index: int
    text: str
    timestamp: Optional[float]
    message: str

    def summary(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "timestamp": self.timestamp,
            "message": self.message,
        }


@dataclass(frozen=True)
class TtyInfo:
    """Structurally decoded TTY_INFO/TTY_INFO2 note.

    Current Vita GPUCRASH cores use a three-word header followed by the byte
    extent named by header word 2. The first two words remain deliberately
    unnamed because their proprietary meaning has not been established.
    """

    header_words: tuple[int, int, int]
    payload: bytes
    trailing: bytes

    @classmethod
    def parse(cls, data: bytes) -> "TtyInfo":
        if len(data) < 12:
            raise ParseError("TTY_INFO is shorter than its three-word header")
        header = struct.unpack_from("<3I", data)
        payload_size = header[2]
        if payload_size > len(data) - 12:
            raise ParseError(
                f"TTY_INFO declares {payload_size} payload bytes, "
                f"only {len(data) - 12} remain"
            )
        payload_end = 12 + payload_size
        return cls(header, data[12:payload_end], data[payload_end:])

    def text(self, preserve_ansi: bool = False) -> str:
        # Empty TTY buffers are represented as NUL-filled fixed-size payloads.
        # Removing NULs also keeps a partially filled buffer readable without
        # claiming that the remaining bytes have circular-buffer semantics.
        value = self.payload.replace(b"\0", b"").decode("utf-8", errors="replace")
        return value if preserve_ansi else _ANSI_CSI.sub("", value)

    def lines(self, preserve_ansi: bool = False) -> Tuple[TtyLine, ...]:
        result = []
        for index, text in enumerate(self.text(preserve_ansi).splitlines()):
            if not text.strip():
                continue
            match = _TIMESTAMP.match(text)
            result.append(
                TtyLine(
                    index,
                    text,
                    float(match.group(1)) if match else None,
                    match.group(2) if match else text,
                )
            )
        return tuple(result)

    def timeline(self, preserve_ansi: bool = False) -> dict:
        lines = self.lines(preserve_ansi)
        stamped = [line for line in lines if line.timestamp is not None]
        gaps = []
        regressions = []
        for previous, current in zip(stamped, stamped[1:]):
            delta = current.timestamp - previous.timestamp
            item = {
                "from_line": previous.index,
                "to_line": current.index,
                "from_timestamp": previous.timestamp,
                "to_timestamp": current.timestamp,
                "delta": delta,
            }
            if delta < 0:
                regressions.append(item)
            else:
                gaps.append(item)
        largest_gap = max(gaps, key=lambda item: item["delta"]) if gaps else None
        first = stamped[0] if stamped else None
        last = stamped[-1] if stamped else None
        return {
            "line_count": len(lines),
            "timestamped_line_count": len(stamped),
            "untimestamped_line_count": len(lines) - len(stamped),
            "first_line_untimestamped": bool(lines and lines[0].timestamp is None),
            "first_timestamp": first.timestamp if first else None,
            "last_timestamp": last.timestamp if last else None,
            "captured_span": (
                last.timestamp - first.timestamp if first and last else None
            ),
            "timestamp_regressions": regressions,
            "largest_nonnegative_gap": largest_gap,
            "last_timestamped_line": last.summary() if last else None,
            "lines": [line.summary() for line in lines],
        }

    def summary(self, preserve_ansi: bool = False) -> dict:
        return {
            "header_words": list(self.header_words),
            "payload_size": len(self.payload),
            "trailing_size": len(self.trailing),
            "text": self.text(preserve_ansi=preserve_ansi),
            "timeline": self.timeline(preserve_ansi=preserve_ansi),
        }
