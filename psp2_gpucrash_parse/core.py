from __future__ import annotations

import gzip
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


class ParseError(ValueError):
    pass


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ParseError(f"u32 at 0x{offset:x} exceeds {len(data)}-byte record")
    return struct.unpack_from("<I", data, offset)[0]


def _c_string(data: bytes) -> str:
    return data.split(b"\0", 1)[0].decode("utf-8", errors="replace")


@dataclass(frozen=True)
class Note:
    name: str
    note_type: int
    description: bytes
    segment_index: int

    def summary(self) -> dict:
        return {
            "name": self.name,
            "type": self.note_type,
            "size": len(self.description),
            "segment_index": self.segment_index,
            "sha256": hashlib.sha256(self.description).hexdigest(),
        }


@dataclass(frozen=True)
class LoadSegment:
    index: int
    file_offset: int
    virtual_address: int
    physical_address: int
    file_size: int
    memory_size: int
    flags: int
    alignment: int
    data: bytes

    @property
    def end_address(self) -> int:
        return self.virtual_address + self.memory_size

    def contains(self, address: int, size: int = 1) -> bool:
        return (
            size >= 0
            and address >= self.virtual_address
            and address + size <= self.virtual_address + self.file_size
        )

    def summary(self) -> dict:
        return {
            "index": self.index,
            "file_offset": self.file_offset,
            "virtual_address": self.virtual_address,
            "physical_address": self.physical_address,
            "file_size": self.file_size,
            "memory_size": self.memory_size,
            "flags": self.flags,
            "alignment": self.alignment,
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }


@dataclass(frozen=True)
class MemoryBlock:
    index: int
    record_flags: int
    uid: int
    name: str
    attributes: int
    base: int
    size: int
    unknown3: int
    unknown4: int
    committed_size: int
    unknown6: int
    unknown7: int

    @property
    def end(self) -> int:
        return self.base + self.size

    def summary(self) -> dict:
        return {
            "index": self.index,
            "record_flags": self.record_flags,
            "uid": self.uid,
            "name": self.name,
            "attributes": self.attributes,
            "base": self.base,
            "size": self.size,
            "committed_size": self.committed_size,
            "unknown": [self.unknown3, self.unknown4, self.unknown6, self.unknown7],
        }


@dataclass
class CoreDump:
    path: Path
    compressed: bool
    raw_file_sha256: str
    image_sha256: str
    image_size: int
    elf_type: int
    machine: int
    flags: int
    notes: List[Note]
    loads: List[LoadSegment]

    @classmethod
    def read(cls, path: Union[Path, str]) -> "CoreDump":
        path = Path(path)
        raw = path.read_bytes()
        compressed = raw.startswith(b"\x1f\x8b")
        try:
            image = gzip.decompress(raw) if compressed else raw
        except (OSError, EOFError) as exc:
            raise ParseError(f"cannot decompress {path}: {exc}") from exc
        return cls._from_elf(path, raw, image, compressed)

    @classmethod
    def _from_elf(
        cls, path: Path, raw: bytes, image: bytes, compressed: bool
    ) -> "CoreDump":
        if len(image) < 52 or image[:4] != b"\x7fELF":
            raise ParseError(f"{path} is not an ELF core")
        if image[4] != 1:
            raise ParseError(f"unsupported ELF class {image[4]}; Vita cores are ELF32")
        if image[5] != 1:
            raise ParseError(f"unsupported ELF byte order {image[5]}")
        (
            elf_type,
            machine,
            _version,
            _entry,
            program_header_offset,
            _section_header_offset,
            flags,
            elf_header_size,
            program_header_size,
            program_header_count,
            _section_header_size,
            _section_header_count,
            _section_name_index,
        ) = struct.unpack_from("<HHIIIIIHHHHHH", image, 16)
        if elf_header_size < 52 or program_header_size < 32:
            raise ParseError("invalid ELF32 header sizes")
        table_end = program_header_offset + program_header_size * program_header_count
        if table_end > len(image):
            raise ParseError("program-header table exceeds core image")

        notes: List[Note] = []
        loads: List[LoadSegment] = []
        for index in range(program_header_count):
            offset = program_header_offset + index * program_header_size
            (
                segment_type,
                file_offset,
                virtual_address,
                physical_address,
                file_size,
                memory_size,
                segment_flags,
                alignment,
            ) = struct.unpack_from("<IIIIIIII", image, offset)
            if file_offset + file_size > len(image):
                raise ParseError(f"program segment {index} exceeds core image")
            payload = image[file_offset : file_offset + file_size]
            if segment_type == 1:
                loads.append(
                    LoadSegment(
                        index,
                        file_offset,
                        virtual_address,
                        physical_address,
                        file_size,
                        memory_size,
                        segment_flags,
                        alignment,
                        payload,
                    )
                )
            elif segment_type == 4:
                notes.extend(_parse_notes(payload, index))
        return cls(
            path=path,
            compressed=compressed,
            raw_file_sha256=hashlib.sha256(raw).hexdigest(),
            image_sha256=hashlib.sha256(image).hexdigest(),
            image_size=len(image),
            elf_type=elf_type,
            machine=machine,
            flags=flags,
            notes=notes,
            loads=loads,
        )

    def notes_named(self, name: str) -> List[Note]:
        return [note for note in self.notes if note.name == name]

    def note(self, name: str) -> Optional[Note]:
        matches = self.notes_named(name)
        return matches[-1] if matches else None

    def read_memory(self, address: int, size: int) -> bytes:
        if size < 0:
            raise ParseError("memory size cannot be negative")
        if size == 0:
            return b""
        cursor = address
        remaining = size
        result = bytearray()
        while remaining:
            candidates = [
                segment
                for segment in self.loads
                if segment.virtual_address <= cursor
                < segment.virtual_address + segment.file_size
            ]
            if not candidates:
                raise ParseError(f"address 0x{cursor:08x} is not captured in a PT_LOAD")
            segment = min(candidates, key=lambda item: item.index)
            local = cursor - segment.virtual_address
            amount = min(remaining, segment.file_size - local)
            result.extend(segment.data[local : local + amount])
            cursor += amount
            remaining -= amount
        return bytes(result)

    def memory_blocks(self) -> List[MemoryBlock]:
        note = self.note("MEM_BLK_INFO")
        if note is None:
            return []
        return parse_memory_blocks(note.description)

    def memory_block_for(self, address: int) -> Optional[MemoryBlock]:
        matches = [block for block in self.memory_blocks() if block.base <= address < block.end]
        return min(matches, key=lambda block: block.size) if matches else None

    def captured_ranges(self, start: int, end: int) -> List[tuple[int, int]]:
        if start < 0 or end < start:
            raise ParseError("invalid captured-range bounds")
        intervals = sorted(
            (
                max(start, segment.virtual_address),
                min(end, segment.virtual_address + segment.file_size),
            )
            for segment in self.loads
            if max(start, segment.virtual_address)
            < min(end, segment.virtual_address + segment.file_size)
        )
        merged: List[tuple[int, int]] = []
        for first, last in intervals:
            if merged and first <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], last))
            else:
                merged.append((first, last))
        return merged

    def captured_bytes(self, start: int, end: int) -> int:
        return sum(last - first for first, last in self.captured_ranges(start, end))

    def load_alias_groups(self) -> List[List[int]]:
        groups: Dict[tuple[int, int, str], List[int]] = {}
        for segment in self.loads:
            key = (
                segment.virtual_address,
                segment.file_size,
                hashlib.sha256(segment.data).hexdigest(),
            )
            groups.setdefault(key, []).append(segment.index)
        return [indices for indices in groups.values() if len(indices) > 1]

    def summary(self) -> dict:
        memory_blocks = []
        for block in self.memory_blocks():
            item = block.summary()
            item["captured_bytes"] = self.captured_bytes(block.base, block.end)
            item["captured_ranges"] = [
                {"start": first, "end": last}
                for first, last in self.captured_ranges(block.base, block.end)
            ]
            memory_blocks.append(item)
        return {
            "path": str(self.path),
            "compressed": self.compressed,
            "raw_file_sha256": self.raw_file_sha256,
            "image_sha256": self.image_sha256,
            "image_size": self.image_size,
            "elf_type": self.elf_type,
            "machine": self.machine,
            "flags": self.flags,
            "notes": [note.summary() for note in self.notes],
            "load_segments": [segment.summary() for segment in self.loads],
            "load_alias_groups": self.load_alias_groups(),
            "memory_blocks": memory_blocks,
        }


def _parse_notes(data: bytes, segment_index: int) -> Iterable[Note]:
    offset = 0
    while offset < len(data):
        if len(data) - offset < 12:
            if any(data[offset:]):
                raise ParseError("nonzero truncated ELF note header")
            break
        name_size, description_size, note_type = struct.unpack_from("<III", data, offset)
        offset += 12
        name_end = offset + name_size
        description_offset = _align(name_end, 4)
        description_end = description_offset + description_size
        next_offset = _align(description_end, 4)
        if next_offset > len(data):
            raise ParseError("ELF note exceeds PT_NOTE segment")
        name = _c_string(data[offset:name_end])
        yield Note(name, note_type, data[description_offset:description_end], segment_index)
        offset = next_offset


def parse_memory_blocks(data: bytes) -> List[MemoryBlock]:
    if len(data) < 8:
        raise ParseError("MEM_BLK_INFO is truncated")
    version, count = struct.unpack_from("<II", data, 0)
    if version != 1:
        raise ParseError(f"unsupported MEM_BLK_INFO version {version}")
    record_size = 72
    required = 8 + count * record_size
    if required > len(data):
        raise ParseError(
            f"MEM_BLK_INFO needs {required} bytes for {count} records, has {len(data)}"
        )
    blocks: List[MemoryBlock] = []
    for index in range(count):
        offset = 8 + index * record_size
        record_flags, uid = struct.unpack_from("<II", data, offset)
        name = _c_string(data[offset + 8 : offset + 40])
        fields = struct.unpack_from("<8I", data, offset + 40)
        blocks.append(
            MemoryBlock(
                index,
                record_flags,
                uid,
                name,
                fields[0],
                fields[1],
                fields[2],
                fields[3],
                fields[4],
                fields[5],
                fields[6],
                fields[7],
            )
        )
    return blocks
