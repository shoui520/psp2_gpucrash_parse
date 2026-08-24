from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional

from .core import Note, ParseError


@dataclass(frozen=True)
class SummaryEntry:
    note_type: int
    planned_size: int
    written_size: int

    @property
    def status(self) -> str:
        if self.planned_size == self.written_size:
            return "complete" if self.planned_size else "not-present"
        if self.written_size == 0:
            return "omitted"
        if self.written_size < self.planned_size:
            return "truncated"
        return "larger-than-planned"

    def status_for(self, name: Optional[str]) -> str:
        # SUMMARY_INFO necessarily describes itself before its own record has
        # been emitted.  Firmware therefore records a zero written size for
        # this one entry even though the final note is present.
        if name == "SUMMARY_INFO" and self.written_size == 0:
            return "self-reference"
        return self.status

    def summary(self, names: Dict[int, str]) -> dict:
        name = names.get(self.note_type)
        return {
            "note_type": self.note_type,
            "name": name,
            "planned_size": self.planned_size,
            "written_size": self.written_size,
            "status": self.status_for(name),
        }


@dataclass(frozen=True)
class SummaryInfo:
    format_version: int
    header_word1: int
    entries: tuple[SummaryEntry, ...]
    trailing: bytes

    @classmethod
    def parse(cls, data: bytes) -> "SummaryInfo":
        if len(data) < 12:
            raise ParseError("SUMMARY_INFO is shorter than its three-word header")
        version, word1, count = struct.unpack_from("<3I", data)
        required = 12 + count * 12
        if required > len(data):
            raise ParseError(
                f"SUMMARY_INFO needs {required} bytes for {count} entries, "
                f"has {len(data)}"
            )
        entries: List[SummaryEntry] = []
        for index in range(count):
            entries.append(
                SummaryEntry(*struct.unpack_from("<3I", data, 12 + index * 12))
            )
        return cls(version, word1, tuple(entries), data[required:])

    def summary(self, notes: List[Note]) -> dict:
        names = {note.note_type: note.name for note in notes}
        incomplete = [
            entry
            for entry in self.entries
            if entry.status_for(names.get(entry.note_type))
            not in ("complete", "not-present", "self-reference")
        ]
        return {
            "format_version": self.format_version,
            "header_word1": self.header_word1,
            "entry_count": len(self.entries),
            "incomplete_count": len(incomplete),
            "trailing_size": len(self.trailing),
            "entries": [entry.summary(names) for entry in self.entries],
        }
