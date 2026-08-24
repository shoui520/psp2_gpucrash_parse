from __future__ import annotations

import hashlib
import os
import re
import shutil
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .core import ParseError


PT_LOAD = 1
SHT_SYMTAB = 2
SHT_DYNSYM = 11
STT_NOTYPE = 0
STT_FUNC = 2
VITA_ET_SCE_RELEXEC = 0xFE04
SCE_SELF_MAGIC = b"SCE\0"
SCE_SELF_HEADER_SIZE = 0x80
SCE_SELF_SEGMENT_INFO_SIZE = 0x20
SCE_SELF_PLAIN = 2
SCE_SELF_UNCOMPRESSED = 1
SCE_SELF_COMPRESSED = 2
MAX_SELF_ELF_SIZE = 512 * 1024 * 1024
MAX_SELF_SEGMENTS = 1024


def _checked_slice(data: bytes, offset: int, size: int, description: str) -> bytes:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ParseError(
            f"{description} at 0x{offset:x}+0x{size:x} exceeds "
            f"0x{len(data):x}-byte image"
        )
    return data[offset : offset + size]


def _decompress_self_segment(
    path: Path, payload: bytes, expected_size: int, index: int
) -> bytes:
    decoder = zlib.decompressobj()
    output = bytearray()
    pending = payload
    try:
        while not decoder.eof:
            limit = expected_size + 1 - len(output)
            if limit <= 0:
                raise ParseError(
                    f"{path} SELF segment {index} expands beyond its "
                    f"0x{expected_size:x}-byte ELF extent"
                )
            chunk = decoder.decompress(pending, limit)
            output.extend(chunk)
            pending = decoder.unconsumed_tail
            if not pending and not decoder.eof:
                break
    except zlib.error as exc:
        raise ParseError(
            f"{path} SELF segment {index} has invalid zlib data: {exc}"
        ) from exc
    if not decoder.eof:
        raise ParseError(f"{path} SELF segment {index} has truncated zlib data")
    if len(output) != expected_size:
        raise ParseError(
            f"{path} SELF segment {index} expands to 0x{len(output):x} bytes; "
            f"ELF program header requires 0x{expected_size:x}"
        )
    trailing = decoder.unused_data
    if trailing and any(trailing):
        raise ParseError(
            f"{path} SELF segment {index} has non-padding data after its zlib stream"
        )
    return bytes(output)


def _extract_vita_self(path: Path, source: bytes) -> Tuple[bytes, dict]:
    if len(source) < SCE_SELF_HEADER_SIZE:
        raise ParseError(f"{path} has a truncated Vita SELF header")
    version = struct.unpack_from("<I", source, 4)[0]
    sdk_type, header_type = struct.unpack_from("<HH", source, 8)
    header_size = struct.unpack_from("<Q", source, 0x10)[0]
    declared_elf_size = struct.unpack_from("<Q", source, 0x18)[0]
    declared_self_size = struct.unpack_from("<Q", source, 0x20)[0]
    elf_offset = struct.unpack_from("<Q", source, 0x40)[0]
    phdr_offset = struct.unpack_from("<Q", source, 0x48)[0]
    shdr_offset = struct.unpack_from("<Q", source, 0x50)[0]
    segment_info_offset = struct.unpack_from("<Q", source, 0x58)[0]
    if header_type != 1:
        raise ParseError(
            f"{path} is an SCE container type {header_type}, not a Vita SELF"
        )
    if declared_self_size and declared_self_size > len(source):
        raise ParseError(
            f"{path} SELF declares a 0x{declared_self_size:x}-byte file but only "
            f"0x{len(source):x} bytes are available"
        )

    elf_header = _checked_slice(source, elf_offset, 52, "SELF embedded ELF header")
    if elf_header[:4] != b"\x7fELF":
        raise ParseError(f"{path} SELF has no ELF header at its declared offset")
    if elf_header[4] != 1 or elf_header[5] != 1:
        raise ParseError(f"{path} SELF does not contain a little-endian ELF32 image")
    (
        _elf_type,
        _machine,
        _elf_version,
        _entry,
        elf_phoff,
        elf_shoff,
        _flags,
        elf_ehsize,
        elf_phentsize,
        elf_phnum,
        elf_shentsize,
        elf_shnum,
        _shstrndx,
    ) = struct.unpack_from("<HHIIIIIHHHHHH", elf_header, 16)
    if elf_ehsize < 52:
        raise ParseError(f"{path} SELF contains an invalid ELF header size")
    if elf_phnum > MAX_SELF_SEGMENTS:
        raise ParseError(
            f"{path} SELF contains {elf_phnum} program headers; maximum is "
            f"{MAX_SELF_SEGMENTS}"
        )
    if elf_phnum and elf_phentsize < 32:
        raise ParseError(f"{path} SELF contains an invalid program-header size")
    if elf_shnum and elf_shentsize < 40:
        raise ParseError(f"{path} SELF contains an invalid section-header size")

    program_headers = _checked_slice(
        source,
        phdr_offset,
        elf_phnum * elf_phentsize,
        "SELF program-header table",
    )
    segment_infos = _checked_slice(
        source,
        segment_info_offset,
        elf_phnum * SCE_SELF_SEGMENT_INFO_SIZE,
        "SELF segment-info table",
    )
    required_size = max(elf_ehsize, elf_phoff + len(program_headers))
    headers = []
    for index in range(elf_phnum):
        header = struct.unpack_from("<8I", program_headers, index * elf_phentsize)
        headers.append(header)
        required_size = max(required_size, header[1] + header[4])
    if elf_shnum:
        required_size = max(required_size, elf_shoff + elf_shnum * elf_shentsize)
    image_size = max(declared_elf_size, required_size)
    if image_size > MAX_SELF_ELF_SIZE:
        raise ParseError(
            f"{path} SELF requires a 0x{image_size:x}-byte reconstructed ELF; "
            f"maximum is 0x{MAX_SELF_ELF_SIZE:x}"
        )
    if declared_elf_size and required_size > declared_elf_size:
        raise ParseError(
            f"{path} SELF program headers exceed its declared ELF size "
            f"0x{declared_elf_size:x}"
        )

    image = bytearray(image_size)
    image[: len(elf_header)] = elf_header
    image[elf_phoff : elf_phoff + len(program_headers)] = program_headers
    if elf_shnum:
        if not shdr_offset:
            raise ParseError(
                f"{path} SELF declares ELF section headers but provides no table offset"
            )
        section_headers = _checked_slice(
            source,
            shdr_offset,
            elf_shnum * elf_shentsize,
            "SELF section-header table",
        )
        image[elf_shoff : elf_shoff + len(section_headers)] = section_headers

    compressed_count = 0
    for index, header in enumerate(headers):
        file_offset, file_size = header[1], header[4]
        payload_offset, payload_size, compression, encryption = struct.unpack_from(
            "<4Q", segment_infos, index * SCE_SELF_SEGMENT_INFO_SIZE
        )
        if encryption != SCE_SELF_PLAIN:
            raise ParseError(
                f"{path} SELF segment {index} is encrypted; only decrypted or "
                "fake SELF images can be analyzed"
            )
        payload = _checked_slice(
            source,
            payload_offset,
            payload_size,
            f"SELF segment {index} payload",
        )
        if compression == SCE_SELF_UNCOMPRESSED:
            if payload_size < file_size:
                raise ParseError(
                    f"{path} SELF segment {index} contains 0x{payload_size:x} bytes; "
                    f"ELF program header requires 0x{file_size:x}"
                )
            segment = payload[:file_size]
        elif compression == SCE_SELF_COMPRESSED:
            compressed_count += 1
            segment = _decompress_self_segment(path, payload, file_size, index)
        else:
            raise ParseError(
                f"{path} SELF segment {index} has unsupported compression mode "
                f"{compression}"
            )
        image[file_offset : file_offset + file_size] = segment

    return bytes(image), {
        "version": version,
        "sdk_type": sdk_type,
        "header_type": header_type,
        "header_size": header_size,
        "declared_elf_size": declared_elf_size,
        "declared_self_size": declared_self_size,
        "program_header_count": elf_phnum,
        "compressed_segment_count": compressed_count,
    }


def _c_string(data: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    return data[offset:].split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _permissions(flags: int) -> str:
    return "".join(
        (
            "r" if flags & 4 else "-",
            "w" if flags & 2 else "-",
            "x" if flags & 1 else "-",
        )
    )


@dataclass(frozen=True)
class ElfLoadSegment:
    index: int
    file_offset: int
    virtual_address: int
    physical_address: int
    file_size: int
    memory_size: int
    flags: int
    alignment: int

    @property
    def permissions(self) -> str:
        return _permissions(self.flags)

    @property
    def memory_end(self) -> int:
        return self.virtual_address + self.memory_size

    def contains(self, address: int) -> bool:
        address &= ~1
        return self.virtual_address <= address < self.memory_end

    def summary(self) -> dict:
        return {
            "index": self.index,
            "file_offset": self.file_offset,
            "virtual_address": self.virtual_address,
            "physical_address": self.physical_address,
            "file_size": self.file_size,
            "memory_size": self.memory_size,
            "permissions": self.permissions,
            "flags": self.flags,
            "alignment": self.alignment,
        }


@dataclass(frozen=True)
class ElfSymbol:
    name: str
    value: int
    size: int
    binding: int
    symbol_type: int
    section_index: int
    table_type: int

    @property
    def address(self) -> int:
        return self.value & ~1

    def summary(self, address: int) -> dict:
        return {
            "name": self.name,
            "address": self.address,
            "offset": address - self.address,
            "size": self.size,
            "binding": self.binding,
            "type": self.symbol_type,
            "thumb": bool(self.value & 1),
        }


@dataclass(frozen=True)
class ElfImportFunction:
    index: int
    nid: int
    stub_address: int


@dataclass(frozen=True)
class ElfImportLibrary:
    index: int
    record_offset: int
    record_size: int
    version: int
    flags: int
    library_nid: int
    library_name: str
    function_count: int
    variable_count: int
    tls_variable_count: int
    functions: Tuple[ElfImportFunction, ...]


@dataclass(frozen=True)
class ElfImage:
    path: Path
    data: bytes
    sha256: str
    source_format: str
    source_size: int
    self_header: Optional[dict]
    elf_type: int
    machine: int
    entry: int
    flags: int
    load_segments: Tuple[ElfLoadSegment, ...]
    module_info_file_offset: Optional[int]
    module_name: Optional[str]
    module_version: Optional[Tuple[int, int]]
    module_attributes: Optional[int]
    imports: Tuple[ElfImportLibrary, ...]
    symbols: Tuple[ElfSymbol, ...]
    section_count: int

    @classmethod
    def read(cls, path: Path) -> "ElfImage":
        path = Path(path)
        return cls.parse(path, path.read_bytes())

    @classmethod
    def parse(cls, path: Path, data: bytes) -> "ElfImage":
        path = Path(path)
        source = data
        source_format = "elf"
        self_header = None
        if data[:4] == SCE_SELF_MAGIC:
            data, self_header = _extract_vita_self(path, data)
            source_format = "self"
        if len(data) < 52 or data[:4] != b"\x7fELF":
            raise ParseError(f"{path} is not an ELF or Vita SELF image")
        if data[4] != 1:
            raise ParseError(f"{path} is not ELF32")
        if data[5] != 1:
            raise ParseError(f"{path} is not little-endian ELF")
        (
            elf_type,
            machine,
            _version,
            entry,
            phoff,
            shoff,
            flags,
            ehsize,
            phentsize,
            phnum,
            shentsize,
            shnum,
            _shstrndx,
        ) = struct.unpack_from("<HHIIIIIHHHHHH", data, 16)
        if ehsize < 52:
            raise ParseError(f"{path} has an invalid ELF header size")
        if phnum and phentsize < 32:
            raise ParseError(f"{path} has an invalid program-header size")
        if shnum and shentsize < 40:
            raise ParseError(f"{path} has an invalid section-header size")

        loads = []
        for index in range(phnum):
            raw = _checked_slice(
                data, phoff + index * phentsize, phentsize, "program header"
            )
            values = struct.unpack_from("<8I", raw)
            if values[0] != PT_LOAD:
                continue
            segment = ElfLoadSegment(index, *values[1:])
            _checked_slice(
                data,
                segment.file_offset,
                segment.file_size,
                f"PT_LOAD {index} file data",
            )
            if segment.file_size > segment.memory_size:
                raise ParseError(
                    f"{path} PT_LOAD {index} file size exceeds its memory size"
                )
            loads.append(segment)

        module_offset = None
        module_name = None
        module_version = None
        module_attributes = None
        module_raw = None
        if elf_type == VITA_ET_SCE_RELEXEC and loads:
            entry_normalized = entry & ~1
            for segment in loads:
                if segment.contains(entry_normalized):
                    offset = entry_normalized - segment.virtual_address
                    if offset < segment.file_size:
                        module_offset = segment.file_offset + offset
                    break
            if module_offset is None and entry_normalized < loads[0].file_size:
                # Decrypted retail SELF ELFs retain e_entry as an offset from
                # the first load image, rather than its linked 0x81xxxxxx VA.
                module_offset = loads[0].file_offset + entry_normalized
            if module_offset is not None:
                raw = _checked_slice(data, module_offset, 0x34, "Vita module info")
                module_raw = raw
                candidate = raw[4:31].split(b"\0", 1)[0]
                if candidate and all(0x20 <= value < 0x7F for value in candidate):
                    module_name = candidate.decode("ascii")
                    module_version = (raw[2], raw[3])
                    module_attributes = struct.unpack_from("<H", raw)[0]

        sections = []
        if shnum:
            for index in range(shnum):
                raw = _checked_slice(
                    data, shoff + index * shentsize, shentsize, "section header"
                )
                sections.append(struct.unpack_from("<10I", raw))
        symbols: List[ElfSymbol] = []
        for section in sections:
            section_type = section[1]
            if section_type not in (SHT_SYMTAB, SHT_DYNSYM):
                continue
            section_offset, section_size = section[4], section[5]
            linked_index, entry_size = section[6], section[9]
            if entry_size < 16 or section_size % entry_size:
                raise ParseError(f"{path} has an invalid ELF symbol table")
            if linked_index >= len(sections):
                raise ParseError(f"{path} symbol table has an invalid string-table link")
            strings_header = sections[linked_index]
            strings = _checked_slice(
                data, strings_header[4], strings_header[5], "symbol string table"
            )
            table = _checked_slice(data, section_offset, section_size, "symbol table")
            for offset in range(0, len(table), entry_size):
                name_offset, value, size, info, _other, section_index = (
                    struct.unpack_from("<IIIBBH", table, offset)
                )
                name = _c_string(strings, name_offset)
                symbol_type = info & 0xF
                if (
                    not name
                    or section_index == 0
                    or symbol_type not in (STT_NOTYPE, STT_FUNC)
                ):
                    continue
                symbols.append(
                    ElfSymbol(
                        name,
                        value,
                        size,
                        info >> 4,
                        symbol_type,
                        section_index,
                        section_type,
                    )
                )

        unique_symbols: Dict[Tuple[str, int, int], ElfSymbol] = {}
        for symbol in symbols:
            unique_symbols[(symbol.name, symbol.value, symbol.size)] = symbol
        ordered_symbols = tuple(
            sorted(unique_symbols.values(), key=lambda item: (item.address, item.name))
        )
        imports = (
            _parse_vita_imports(data, tuple(loads), module_raw)
            if module_raw is not None
            else ()
        )
        return cls(
            path,
            data,
            hashlib.sha256(source).hexdigest(),
            source_format,
            len(source),
            self_header,
            elf_type,
            machine,
            entry,
            flags,
            tuple(loads),
            module_offset,
            module_name,
            module_version,
            module_attributes,
            imports,
            ordered_symbols,
            shnum,
        )

    def symbol_for(self, address: int) -> Optional[ElfSymbol]:
        address &= ~1
        containing = [
            symbol
            for symbol in self.symbols
            if symbol.size and symbol.address <= address < symbol.address + symbol.size
        ]
        if containing:
            return max(containing, key=lambda item: item.address)
        zero_size = [
            symbol
            for symbol in self.symbols
            if not symbol.size
            and symbol.address <= address
            and address - symbol.address <= 0x10000
        ]
        return max(zero_size, key=lambda item: item.address) if zero_size else None

    def summary(self) -> dict:
        imported_function_count = sum(len(item.functions) for item in self.imports)
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "source_format": self.source_format,
            "source_size": self.source_size,
            "reconstructed_elf_size": len(self.data),
            "self_header": self.self_header,
            "elf_type": self.elf_type,
            "machine": self.machine,
            "entry": self.entry,
            "flags": self.flags,
            "section_count": self.section_count,
            "symbol_count": len(self.symbols),
            "module_info_file_offset": self.module_info_file_offset,
            "module_name": self.module_name,
            "module_version": list(self.module_version) if self.module_version else None,
            "module_attributes": self.module_attributes,
            "import_library_count": len(self.imports),
            "import_function_count": imported_function_count,
            "load_segments": [segment.summary() for segment in self.load_segments],
        }


def _pointer_location(
    data: bytes,
    loads: Sequence[ElfLoadSegment],
    pointer: int,
    size: int = 1,
) -> Optional[Tuple[int, int]]:
    normalized = pointer & ~1
    for load in loads:
        if load.virtual_address <= normalized and normalized + size <= (
            load.virtual_address + load.file_size
        ):
            offset = load.file_offset + normalized - load.virtual_address
            return offset, normalized
    if loads and normalized + size <= loads[0].file_size:
        offset = loads[0].file_offset + normalized
        return offset, loads[0].virtual_address + normalized
    return None


def _read_pointer_string(
    data: bytes, loads: Sequence[ElfLoadSegment], pointer: int
) -> str:
    location = _pointer_location(data, loads, pointer)
    if location is None:
        return ""
    offset, _address = location
    end = min(len(data), offset + 128)
    return data[offset:end].split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _read_pointer_words(
    data: bytes,
    loads: Sequence[ElfLoadSegment],
    pointer: int,
    count: int,
) -> Optional[Tuple[int, ...]]:
    if count < 0 or count > 4096:
        return None
    location = _pointer_location(data, loads, pointer, count * 4)
    if location is None:
        return None
    offset, _address = location
    return struct.unpack_from(f"<{count}I", data, offset) if count else ()


def _parse_vita_imports(
    data: bytes,
    loads: Sequence[ElfLoadSegment],
    module_info: bytes,
) -> Tuple[ElfImportLibrary, ...]:
    import_top = struct.unpack_from("<I", module_info, 0x2C)[0]
    import_end = struct.unpack_from("<I", module_info, 0x30)[0]
    top_location = _pointer_location(data, loads, import_top)
    end_location = _pointer_location(data, loads, import_end, 0)
    if top_location is None or end_location is None:
        return ()
    offset, linked_address = top_location
    end_offset, _end_address = end_location
    if end_offset < offset or end_offset - offset > 0x100000:
        return ()
    records = []
    while offset < end_offset and len(records) < 4096:
        if offset + 2 > end_offset:
            break
        record_size = struct.unpack_from("<H", data, offset)[0]
        if record_size not in (0x24, 0x34) or offset + record_size > end_offset:
            break
        raw = data[offset : offset + record_size]
        version = struct.unpack_from("<H", raw, 2)[0]
        flags = struct.unpack_from("<H", raw, 4)[0]
        function_count = struct.unpack_from("<H", raw, 6)[0]
        variable_count = struct.unpack_from("<H", raw, 8)[0]
        tls_count = struct.unpack_from("<H", raw, 0x0A)[0]
        if record_size == 0x34:
            library_nid = struct.unpack_from("<I", raw, 0x10)[0]
            name_pointer = struct.unpack_from("<I", raw, 0x14)[0]
            nid_pointer = struct.unpack_from("<I", raw, 0x1C)[0]
            entry_pointer = struct.unpack_from("<I", raw, 0x20)[0]
        else:
            library_nid = struct.unpack_from("<I", raw, 0x0C)[0]
            name_pointer = struct.unpack_from("<I", raw, 0x10)[0]
            nid_pointer = struct.unpack_from("<I", raw, 0x14)[0]
            entry_pointer = struct.unpack_from("<I", raw, 0x18)[0]
        nids = _read_pointer_words(data, loads, nid_pointer, function_count)
        entries = _read_pointer_words(data, loads, entry_pointer, function_count)
        functions = ()
        if nids is not None and entries is not None:
            functions = tuple(
                ElfImportFunction(index, nid, entries[index] & ~1)
                for index, nid in enumerate(nids)
            )
        records.append(
            ElfImportLibrary(
                len(records),
                linked_address,
                record_size,
                version,
                flags,
                library_nid,
                _read_pointer_string(data, loads, name_pointer),
                function_count,
                variable_count,
                tls_count,
                functions,
            )
        )
        offset += record_size
        linked_address += record_size
    return tuple(records)


def _match_image(image: ElfImage, modules: Optional[dict]) -> dict:
    if image.module_name is None:
        return {
            "status": "rejected",
            "module_index": None,
            "module_uid": None,
            "reasons": ["ELF has no decodable Vita module identity at e_entry"],
        }
    captured = modules["modules"] if modules is not None else []
    candidates = [item for item in captured if item["name"] == image.module_name]
    if not candidates:
        return {
            "status": "rejected",
            "module_index": None,
            "module_uid": None,
            "reasons": [
                f"module {image.module_name!r} is not present in captured MODULE_INFO"
            ],
        }
    if len(candidates) != 1:
        return {
            "status": "rejected",
            "module_index": None,
            "module_uid": None,
            "reasons": [
                f"module identity {image.module_name!r} matches multiple captured modules"
            ],
        }
    module = candidates[0]
    captured_segments = module["segments"]
    reasons = []
    warnings = []
    if len(image.load_segments) != len(captured_segments):
        reasons.append(
            f"PT_LOAD count differs (ELF={len(image.load_segments)}, "
            f"dump={len(captured_segments)})"
        )
    for ordinal, (load, segment) in enumerate(
        zip(image.load_segments, captured_segments), 1
    ):
        if load.permissions != segment["permissions"]:
            reasons.append(
                f"segment {ordinal} permissions differ "
                f"(ELF={load.permissions}, dump={segment['permissions']})"
            )
        if load.memory_size != segment["size"]:
            tail_difference = load.memory_size - segment["size"]
            # Some reconstructed retail ELFs include one final p_align unit
            # beyond the kernel's MODULE_INFO segment extent.  The linked and
            # runtime starts still correspond; only addresses in the captured
            # extent are eligible for translation.
            compatible_tail = (
                tail_difference > 0
                and tail_difference <= max(load.alignment, 0x10)
            )
            if compatible_tail:
                warnings.append(
                    f"segment {ordinal} ELF extent has 0x{tail_difference:x} "
                    "additional tail bytes"
                )
                continue
            reasons.append(
                f"segment {ordinal} memory size differs "
                f"(ELF=0x{load.memory_size:x}, dump=0x{segment['size']:x})"
            )
    return {
        "status": "rejected" if reasons else "matched",
        "match_quality": "compatible" if warnings and not reasons else (
            "exact" if not reasons else None
        ),
        "module_index": module["index"],
        "module_uid": module["uid"],
        "runtime_module_name": module["name"],
        "reasons": reasons,
        "warnings": warnings,
    }


def _find_sdk_tool(sdk: Optional[Path], name: str) -> Optional[str]:
    roots = []
    if sdk is not None:
        roots.append(Path(sdk))
    environment_root = os.environ.get("VITASDK")
    if environment_root:
        roots.append(Path(environment_root))
    for root in roots:
        candidate = root / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(name)


def _sdk_root(sdk: Optional[Path], objdump: Optional[str]) -> Optional[Path]:
    candidates = []
    if sdk is not None:
        candidates.append(Path(sdk))
    environment_root = os.environ.get("VITASDK")
    if environment_root:
        candidates.append(Path(environment_root))
    if objdump is not None:
        candidates.append(Path(objdump).resolve().parent.parent)
    for candidate in candidates:
        if (candidate / "share" / "vita-headers" / "db").is_dir():
            return candidate
    return None


_YAML_LIBRARY = re.compile(r"^ {6}([^\s:#][^:]*):\s*$")
_YAML_NID = re.compile(r"^ {8}nid:\s*0x([0-9a-fA-F]{8})\s*$")
_YAML_SECTION = re.compile(r"^ {8}(functions|variables):\s*$")
_YAML_ENTRY = re.compile(
    r"^ {10}([^\s:#][^:]*):\s*0x([0-9a-fA-F]{8})\s*$"
)


def _load_nid_database(
    sdk_root: Optional[Path],
) -> Tuple[Dict[Tuple[str, int], str], List[str]]:
    if sdk_root is None:
        return {}, []
    database_root = sdk_root / "share" / "vita-headers" / "db"
    version_directories = []
    preferred = database_root / "360"
    if preferred.is_dir():
        version_directories.append(preferred)
    if not version_directories:
        version_directories.extend(
            sorted(
                (item for item in database_root.iterdir() if item.is_dir()),
                key=lambda item: item.name,
                reverse=True,
            )[:1]
        )
    names: Dict[Tuple[str, int], str] = {}
    sources = []
    for directory in version_directories:
        sources.append(str(directory))
        for path in sorted(directory.glob("*.yml")):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            library = None
            library_nid = None
            section = None
            for line in lines:
                match = _YAML_LIBRARY.match(line)
                if match is not None:
                    library = match.group(1).strip().strip("'\"")
                    library_nid = None
                    section = None
                    continue
                match = _YAML_NID.match(line)
                if match is not None and library is not None:
                    library_nid = int(match.group(1), 16)
                    continue
                match = _YAML_SECTION.match(line)
                if match is not None and library is not None:
                    section = match.group(1)
                    continue
                match = _YAML_ENTRY.match(line)
                if (
                    match is not None
                    and library is not None
                    and library_nid is not None
                    and section == "functions"
                ):
                    function_name = match.group(1).strip().strip("'\"")
                    names[(library, int(match.group(2), 16))] = function_name
    return names, sources


def _import_report(
    image: ElfImage, nid_names: Dict[Tuple[str, int], str]
) -> Tuple[List[dict], Dict[int, dict]]:
    libraries = []
    by_address = {}
    for library in image.imports:
        functions = []
        for function in library.functions:
            name = nid_names.get((library.library_name, function.nid))
            record = {
                "index": function.index,
                "nid": function.nid,
                "stub_address": function.stub_address,
                "name": name,
                "library_name": library.library_name,
                "library_nid": library.library_nid,
            }
            functions.append(record)
            by_address[function.stub_address] = record
        libraries.append(
            {
                "index": library.index,
                "record_offset": library.record_offset,
                "record_size": library.record_size,
                "version": library.version,
                "flags": library.flags,
                "library_nid": library.library_nid,
                "library_name": library.library_name,
                "function_count": library.function_count,
                "variable_count": library.variable_count,
                "tls_variable_count": library.tls_variable_count,
                "functions": functions,
            }
        )
    return libraries, by_address


_DISASSEMBLY_TARGET = re.compile(r"0x([0-9a-fA-F]+)")


def _annotate_import_targets(lines: Sequence[str], imports: Dict[int, dict]) -> List[str]:
    result = []
    for line in lines:
        targets = [
            imports.get(int(match.group(1), 16))
            for match in _DISASSEMBLY_TARGET.finditer(line)
        ]
        target = next((item for item in targets if item is not None), None)
        if target is not None:
            name = target["name"] or f"NID_0x{target['nid']:08x}"
            line += f"  <{target['library_name']}:{name}>"
        result.append(line)
    return result


def _source_location(tool: Optional[str], image: ElfImage, address: int) -> Optional[dict]:
    if tool is None or image.section_count == 0:
        return None
    try:
        result = subprocess.run(
            [tool, "-f", "-C", "-e", str(image.path), f"0x{address:x}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode or len(lines) < 2 or lines[1] in ("??:0", "??:?"):
        return None
    return {"function": lines[0], "file_line": lines[1]}


def _disassemble(
    tool: Optional[str],
    image: ElfImage,
    load: ElfLoadSegment,
    address: int,
    instruction_set: str,
) -> Tuple[List[str], Optional[str]]:
    if tool is None:
        return [], None
    alignment = 2 if instruction_set == "thumb" else 4
    address &= ~(alignment - 1)
    segment_file_end = load.virtual_address + load.file_size
    if not load.virtual_address <= address < segment_file_end:
        return [], "address has no file-backed bytes"
    start = max(load.virtual_address, address - 8) & ~(alignment - 1)
    end = min(segment_file_end, address + 24)
    end &= ~(alignment - 1)
    if end <= start:
        return [], "no aligned instruction bytes are available"
    file_offset = load.file_offset + start - load.virtual_address
    window = image.data[file_offset : file_offset + end - start]
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="psp2-gpucrash-", suffix=".bin", delete=False
        ) as temporary:
            temporary.write(window)
            temporary_path = temporary.name
        command = [
            tool,
            "-D",
            "-b",
            "binary",
            "-m",
            "arm",
            f"--adjust-vma=0x{start:x}",
        ]
        if instruction_set == "thumb":
            command.extend(["-M", "force-thumb"])
        command.append(temporary_path)
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
            text=True,
        )
        if result.returncode:
            error = result.stderr.strip().splitlines()
            return [], error[-1] if error else f"objdump exited {result.returncode}"
        lines = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped and ":" in stripped and "out of bounds" not in stripped:
                prefix = stripped.split(":", 1)[0]
                if prefix and all(value in "0123456789abcdefABCDEF" for value in prefix):
                    lines.append(stripped)
        return lines, None
    except (OSError, subprocess.SubprocessError) as exc:
        return [], str(exc)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _disassembly_score(lines: Sequence[str], error: Optional[str]) -> int:
    if error is not None or not lines:
        return 1000000
    return sum(
        10
        for line in lines
        if "<UNDEFINED>" in line or "undefined" in line.casefold()
    )


def _address_location(
    image: ElfImage,
    association: dict,
    runtime_location: dict,
    runtime_address: int,
    imports: Dict[int, dict],
) -> Optional[dict]:
    if runtime_location.get("module_index") != association["module_index"]:
        return None
    ordinal = runtime_location["segment_number"] - 1
    if ordinal < 0 or ordinal >= len(image.load_segments):
        return None
    load = image.load_segments[ordinal]
    offset = (runtime_address & ~1) - runtime_location["segment_start"]
    if offset < 0 or offset >= load.memory_size:
        return None
    elf_address = load.virtual_address + offset
    file_offset = load.file_offset + offset if offset < load.file_size else None
    symbol = image.symbol_for(elf_address)
    return {
        "image_path": str(image.path),
        "module_name": image.module_name,
        "segment_number": ordinal + 1,
        "segment_offset": offset,
        "elf_virtual_address": elf_address,
        "file_offset": file_offset,
        "file_backed": file_offset is not None,
        "symbol": symbol.summary(elf_address) if symbol is not None else None,
        "import": imports.get(elf_address),
    }


def enrich_diagnosis_with_elfs(
    diagnosis: dict,
    paths: Sequence[Path],
    sdk: Optional[Path] = None,
) -> dict:
    objdump = _find_sdk_tool(sdk, "arm-vita-eabi-objdump")
    addr2line = _find_sdk_tool(sdk, "arm-vita-eabi-addr2line")
    nid_names, nid_sources = _load_nid_database(_sdk_root(sdk, objdump))
    execution = diagnosis["execution_context"]
    images = []
    matched_images: List[Tuple[ElfImage, dict, dict, Dict[int, dict]]] = []
    invalid_count = 0
    for path in paths:
        try:
            image = ElfImage.read(Path(path))
        except (OSError, ParseError) as exc:
            invalid_count += 1
            images.append(
                {
                    "path": str(path),
                    "status": "invalid",
                    "error": str(exc),
                }
            )
            continue
        association = _match_image(image, execution["modules"])
        report = image.summary()
        import_libraries, imports_by_address = _import_report(image, nid_names)
        report["imports"] = import_libraries
        report["resolved_import_function_count"] = sum(
            function["name"] is not None
            for library in import_libraries
            for function in library["functions"]
        )
        report["association"] = association
        report["resolved_address_count"] = 0
        report["address_records"] = []
        images.append(report)
        if association["status"] == "matched":
            matched_images.append((image, association, report, imports_by_address))

    address_records: Dict[Tuple[str, int, str], dict] = {}
    for thread in execution["threads"]:
        registers = thread["registers"]
        if registers is None:
            continue
        for role in ("pc", "lr"):
            runtime_location = thread[f"{role}_location"]
            if runtime_location is None:
                continue
            runtime_address = registers[role]
            for image, association, report, imports_by_address in matched_images:
                location = _address_location(
                    image,
                    association,
                    runtime_location,
                    runtime_address,
                    imports_by_address,
                )
                if location is None:
                    continue
                thread[f"{role}_image"] = location
                if role == "pc":
                    instruction_set = (
                        "thumb" if registers["cpsr"] & 0x20 else "arm"
                    )
                    basis = "CPSR.T"
                else:
                    instruction_set = "thumb" if runtime_address & 1 else "arm"
                    basis = "return-address bit 0"
                key = (str(image.path), runtime_address & ~1, instruction_set)
                record = address_records.get(key)
                if record is None:
                    load = image.load_segments[location["segment_number"] - 1]
                    instructions, disassembly_error = _disassemble(
                        objdump,
                        image,
                        load,
                        location["elf_virtual_address"],
                        instruction_set,
                    )
                    alternate_set = "arm" if instruction_set == "thumb" else "thumb"
                    alternate, alternate_error = _disassemble(
                        objdump,
                        image,
                        load,
                        location["elf_virtual_address"],
                        alternate_set,
                    )
                    if _disassembly_score(
                        alternate, alternate_error
                    ) < _disassembly_score(instructions, disassembly_error):
                        instruction_set = alternate_set
                        basis = "instruction decoding; register-state hint was inconsistent"
                        instructions = alternate
                        disassembly_error = alternate_error
                    instructions = _annotate_import_targets(
                        instructions, imports_by_address
                    )
                    source = _source_location(
                        addr2line, image, location["elf_virtual_address"]
                    )
                    record = {
                        "runtime_address": runtime_address,
                        "instruction_set": instruction_set,
                        "instruction_set_basis": basis,
                        "location": location,
                        "source": source,
                        "disassembly": instructions,
                        "disassembly_error": disassembly_error,
                        "references": [],
                    }
                    address_records[key] = record
                    report["address_records"].append(record)
                record["references"].append(
                    {
                        "thread_index": thread["index"],
                        "thread_uid": thread["uid"],
                        "thread_name": thread["name"],
                        "role": role,
                    }
                )

        for candidate in thread["stack_return_candidates"]:
            runtime_location = candidate["location"]
            for image, association, _report, imports_by_address in matched_images:
                location = _address_location(
                    image,
                    association,
                    runtime_location,
                    candidate["value"],
                    imports_by_address,
                )
                if location is not None:
                    candidate["image"] = location

    for _image, _association, report, _imports in matched_images:
        report["resolved_address_count"] = len(report["address_records"])
    matched_count = len(matched_images)
    rejected_count = sum(
        item.get("association", {}).get("status") == "rejected" for item in images
    )
    return {
        "requested_count": len(paths),
        "matched_count": matched_count,
        "rejected_count": rejected_count,
        "invalid_count": invalid_count,
        "objdump": objdump,
        "addr2line": addr2line,
        "nid_database_sources": nid_sources,
        "images": images,
    }
