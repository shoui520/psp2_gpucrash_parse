from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .core import CoreDump, ParseError


def _u16(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 2 > len(data):
        raise ParseError(f"u16 at 0x{offset:x} exceeds {len(data)}-byte record")
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ParseError(f"u32 at 0x{offset:x} exceeds {len(data)}-byte record")
    return struct.unpack_from("<I", data, offset)[0]


def _s32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ParseError(f"s32 at 0x{offset:x} exceeds {len(data)}-byte record")
    return struct.unpack_from("<i", data, offset)[0]


def _c_string(data: bytes) -> str:
    return data.split(b"\0", 1)[0].decode("utf-8", errors="replace")


_THREAD_STATUS_NAMES = {
    0x01: "running",
    0x02: "ready",
    0x04: "standby",
    0x08: "waiting",
    0x10: "dormant",
    0x20: "dead",
}


@dataclass(frozen=True)
class ProcessInfo:
    format_version: int
    flags: int
    process_id: int
    attributes: int
    process_name: str
    executable_path: str
    raw: bytes

    @classmethod
    def parse(cls, data: bytes) -> "ProcessInfo":
        if len(data) < 0x5C:
            raise ParseError("PROCESS_INFO is shorter than its fixed v1 prefix")
        version = _u32(data, 0)
        if version != 1:
            raise ParseError(f"unsupported PROCESS_INFO version {version}")
        return cls(
            version,
            _u32(data, 4),
            _u32(data, 8),
            _u32(data, 0x0C),
            _c_string(data[0x10:0x30]),
            _c_string(data[0x5C:]),
            data,
        )

    def summary(self) -> dict:
        return {
            "format_version": self.format_version,
            "flags": self.flags,
            "process_id": self.process_id,
            "attributes": self.attributes,
            "process_name": self.process_name,
            "executable_path": self.executable_path,
            "raw_size": len(self.raw),
        }


@dataclass(frozen=True)
class ModuleSegment:
    number: int
    record_word0: int
    attributes: int
    start: int
    size: int
    alignment: int

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def permissions(self) -> str:
        protection = self.attributes & 0x7
        return "".join(
            (
                "r" if protection & 4 else "-",
                "w" if protection & 2 else "-",
                "x" if protection & 1 else "-",
            )
        )

    def contains(self, address: int) -> bool:
        address &= ~1
        return self.start <= address < self.end

    def summary(self) -> dict:
        return {
            "number": self.number,
            "record_word0": self.record_word0,
            "attributes": self.attributes,
            "permissions": self.permissions,
            "start": self.start,
            "size": self.size,
            "end": self.end,
            "alignment": self.alignment,
        }


@dataclass(frozen=True)
class ModuleInfoEntry:
    index: int
    record_word0: int
    uid: int
    name: str
    segments: Tuple[ModuleSegment, ...]
    header: bytes
    footer: bytes

    def summary(self) -> dict:
        return {
            "index": self.index,
            "record_word0": self.record_word0,
            "uid": self.uid,
            "name": self.name,
            "segments": [segment.summary() for segment in self.segments],
            "footer_words": list(struct.unpack("<4I", self.footer)),
        }


@dataclass(frozen=True)
class AddressLocation:
    address: int
    module_index: int
    module_name: str
    module_uid: int
    segment_number: int
    segment_start: int
    segment_size: int
    segment_permissions: str
    offset: int

    @property
    def notation(self) -> str:
        return f"{self.module_name}@{self.segment_number}+0x{self.offset:x}"

    def summary(self) -> dict:
        return {
            "address": self.address,
            "module_index": self.module_index,
            "module_name": self.module_name,
            "module_uid": self.module_uid,
            "segment_number": self.segment_number,
            "segment_start": self.segment_start,
            "segment_size": self.segment_size,
            "segment_permissions": self.segment_permissions,
            "offset": self.offset,
            "notation": self.notation,
        }


@dataclass(frozen=True)
class ModuleInfo:
    format_version: int
    modules: Tuple[ModuleInfoEntry, ...]
    trailing: bytes

    @classmethod
    def parse(cls, data: bytes) -> "ModuleInfo":
        if len(data) < 8:
            raise ParseError("MODULE_INFO is truncated")
        version, count = struct.unpack_from("<2I", data)
        if version != 1:
            raise ParseError(f"unsupported MODULE_INFO version {version}")
        offset = 8
        modules: List[ModuleInfoEntry] = []
        for index in range(count):
            if offset + 0x50 > len(data):
                raise ParseError(f"MODULE_INFO module {index} header is truncated")
            header = data[offset : offset + 0x50]
            segment_count = _u32(header, 0x4C)
            if segment_count > 16:
                raise ParseError(
                    f"MODULE_INFO module {index} has implausible segment count "
                    f"{segment_count}"
                )
            offset += 0x50
            required = offset + segment_count * 0x14 + 0x10
            if required > len(data):
                raise ParseError(f"MODULE_INFO module {index} body is truncated")
            segments = []
            for segment_index in range(segment_count):
                values = struct.unpack_from("<5I", data, offset)
                segments.append(ModuleSegment(segment_index + 1, *values))
                offset += 0x14
            footer = data[offset : offset + 0x10]
            offset += 0x10
            modules.append(
                ModuleInfoEntry(
                    index,
                    _u32(header, 0),
                    _u32(header, 4),
                    _c_string(header[0x24:0x4C]),
                    tuple(segments),
                    header,
                    footer,
                )
            )
        return cls(version, tuple(modules), data[offset:])

    def locate(self, address: int, executable_only: bool = False) -> Optional[AddressLocation]:
        normalized = address & ~1
        matches = []
        for module in self.modules:
            for segment in module.segments:
                if segment.contains(normalized) and (
                    not executable_only or "x" in segment.permissions
                ):
                    matches.append((segment.size, module, segment))
        if not matches:
            return None
        _size, module, segment = min(matches, key=lambda item: item[0])
        return AddressLocation(
            address,
            module.index,
            module.name,
            module.uid,
            segment.number,
            segment.start,
            segment.size,
            segment.permissions,
            normalized - segment.start,
        )

    def summary(self) -> dict:
        return {
            "format_version": self.format_version,
            "module_count": len(self.modules),
            "trailing_size": len(self.trailing),
            "modules": [module.summary() for module in self.modules],
        }


@dataclass(frozen=True)
class ThreadInfoEntry:
    index: int
    record_size: int
    uid: int
    name: str
    attributes: int
    status: int
    status_word: int
    entry: int
    stack_base: int
    stack_size: int
    implementation_words: Tuple[int, int, int]
    initial_priority: int
    current_priority: int
    initial_cpu_affinity: int
    current_cpu_affinity: int
    current_cpu_id: int
    wait_type: int
    wait_id: int
    run_clocks: int
    scheduler_word: int
    stop_reason: int
    recorded_pc: int
    raw: bytes

    @property
    def stack_end(self) -> int:
        return self.stack_base + self.stack_size

    @property
    def status_name(self) -> str:
        exact = _THREAD_STATUS_NAMES.get(self.status)
        if exact is not None:
            return exact
        names = [
            name for mask, name in _THREAD_STATUS_NAMES.items() if self.status & mask
        ]
        return "|".join(names) if names else "unknown"

    @classmethod
    def parse(cls, index: int, data: bytes) -> "ThreadInfoEntry":
        if len(data) < 0xA0:
            raise ParseError(f"THREAD_INFO record {index} is shorter than 0xa0 bytes")
        record_size = _u32(data, 0)
        if record_size != len(data):
            raise ParseError(
                f"THREAD_INFO record {index} declares 0x{record_size:x} bytes, "
                f"received 0x{len(data):x}"
            )
        return cls(
            index,
            record_size,
            _u32(data, 4),
            _c_string(data[8:0x28]),
            _u32(data, 0x28),
            _u16(data, 0x30),
            _u32(data, 0x30),
            _u32(data, 0x34),
            _u32(data, 0x38),
            _u32(data, 0x3C),
            (_u32(data, 0x40), _u32(data, 0x44), _u32(data, 0x48)),
            _s32(data, 0x4C),
            _s32(data, 0x50),
            _u32(data, 0x54),
            _u32(data, 0x58),
            _s32(data, 0x5C),
            _u32(data, 0x60),
            _u32(data, 0x64),
            _u32(data, 0x68) | (_u32(data, 0x6C) << 32),
            _u32(data, 0x70),
            _u32(data, 0x74),
            _u32(data, 0x9C),
            data,
        )


@dataclass(frozen=True)
class ThreadInfo:
    format_version: int
    threads: Tuple[ThreadInfoEntry, ...]
    trailing: bytes

    @classmethod
    def parse(cls, data: bytes) -> "ThreadInfo":
        if len(data) < 8:
            raise ParseError("THREAD_INFO is truncated")
        version, count = struct.unpack_from("<2I", data)
        if version != 0x12:
            raise ParseError(f"unsupported THREAD_INFO version {version:#x}")
        offset = 8
        result = []
        for index in range(count):
            if offset + 4 > len(data):
                raise ParseError(f"THREAD_INFO record {index} size is truncated")
            size = _u32(data, offset)
            if size < 0xA0 or offset + size > len(data):
                raise ParseError(
                    f"THREAD_INFO record {index} has invalid size 0x{size:x}"
                )
            result.append(ThreadInfoEntry.parse(index, data[offset : offset + size]))
            offset += size
        return cls(version, tuple(result), data[offset:])


@dataclass(frozen=True)
class ThreadRegisters:
    index: int
    record_size: int
    thread_uid: int
    gpr: Tuple[int, ...]
    cpsr: int
    raw: bytes

    @property
    def sp(self) -> int:
        return self.gpr[13]

    @property
    def lr(self) -> int:
        return self.gpr[14]

    @property
    def pc(self) -> int:
        return self.gpr[15]

    @classmethod
    def parse(cls, index: int, data: bytes) -> "ThreadRegisters":
        if len(data) < 0x4C:
            raise ParseError(
                f"THREAD_REG_INFO record {index} is shorter than the GPR block"
            )
        record_size = _u32(data, 0)
        if record_size != len(data):
            raise ParseError(
                f"THREAD_REG_INFO record {index} declares 0x{record_size:x} bytes, "
                f"received 0x{len(data):x}"
            )
        return cls(
            index,
            record_size,
            _u32(data, 4),
            struct.unpack_from("<16I", data, 8),
            _u32(data, 0x48),
            data,
        )


@dataclass(frozen=True)
class ThreadRegisterInfo:
    format_version: int
    records: Tuple[ThreadRegisters, ...]
    trailing: bytes

    @classmethod
    def parse(cls, data: bytes) -> "ThreadRegisterInfo":
        if len(data) < 8:
            raise ParseError("THREAD_REG_INFO is truncated")
        version, count = struct.unpack_from("<2I", data)
        if version != 0x11:
            raise ParseError(f"unsupported THREAD_REG_INFO version {version:#x}")
        offset = 8
        records = []
        for index in range(count):
            if offset + 4 > len(data):
                raise ParseError(f"THREAD_REG_INFO record {index} size is truncated")
            size = _u32(data, offset)
            if size < 0x4C or offset + size > len(data):
                raise ParseError(
                    f"THREAD_REG_INFO record {index} has invalid size 0x{size:x}"
                )
            records.append(ThreadRegisters.parse(index, data[offset : offset + size]))
            offset += size
        return cls(version, tuple(records), data[offset:])


@dataclass(frozen=True)
class StackUsage:
    index: int
    flags: int
    thread_uid: int
    peak_bytes: int
    current_bytes: int

    def summary(self) -> dict:
        return {
            "index": self.index,
            "flags": self.flags,
            "thread_uid": self.thread_uid,
            "peak_bytes": self.peak_bytes,
            "current_bytes": self.current_bytes,
        }


@dataclass(frozen=True)
class StackInfo:
    format_version: int
    records: Tuple[StackUsage, ...]
    trailing: bytes

    @classmethod
    def parse(cls, data: bytes) -> "StackInfo":
        if len(data) < 8:
            raise ParseError("STACK_INFO is truncated")
        version, count = struct.unpack_from("<2I", data)
        if version != 4:
            raise ParseError(f"unsupported STACK_INFO version {version}")
        required = 8 + count * 0x10
        if required > len(data):
            raise ParseError(
                f"STACK_INFO needs 0x{required:x} bytes for {count} records, "
                f"has 0x{len(data):x}"
            )
        records = tuple(
            StackUsage(index, *struct.unpack_from("<4I", data, 8 + index * 0x10))
            for index in range(count)
        )
        return cls(version, records, data[required:])


@dataclass(frozen=True)
class ExecutionContext:
    process: Optional[ProcessInfo]
    modules: Optional[ModuleInfo]
    threads: Optional[ThreadInfo]
    registers: Optional[ThreadRegisterInfo]
    stacks: Optional[StackInfo]

    @classmethod
    def parse(cls, core: CoreDump) -> "ExecutionContext":
        def parse_note(name: str, parser):
            note = core.note(name)
            return parser(note.description) if note is not None else None

        return cls(
            parse_note("PROCESS_INFO", ProcessInfo.parse),
            parse_note("MODULE_INFO", ModuleInfo.parse),
            parse_note("THREAD_INFO", ThreadInfo.parse),
            parse_note("THREAD_REG_INFO", ThreadRegisterInfo.parse),
            parse_note("STACK_INFO", StackInfo.parse),
        )

    def _stack_candidates(
        self,
        core: CoreDump,
        thread: ThreadInfoEntry,
        registers: ThreadRegisters,
        limit: int = 12,
    ) -> List[dict]:
        if self.modules is None or registers.sp == 0:
            return []
        if not (thread.stack_base <= registers.sp < thread.stack_end):
            return []
        captured = core.captured_ranges(registers.sp, thread.stack_end)
        if not captured or captured[0][0] != registers.sp:
            return []
        end = min(captured[0][1], registers.sp + 0x10000)
        byte_count = (end - registers.sp) & ~3
        if byte_count <= 0:
            return []
        data = core.read_memory(registers.sp, byte_count)
        result = []
        for offset in range(0, len(data), 4):
            value = _u32(data, offset)
            location = self.modules.locate(value, executable_only=True)
            if location is None:
                continue
            result.append(
                {
                    "stack_address": registers.sp + offset,
                    "value": value,
                    "location": location.summary(),
                }
            )
            if len(result) >= limit:
                break
        return result

    def summary(self, core: CoreDump) -> dict:
        registers_by_uid: Dict[int, ThreadRegisters] = {
            record.thread_uid: record
            for record in self.registers.records
        } if self.registers is not None else {}
        stacks_by_uid: Dict[int, StackUsage] = {
            record.thread_uid: record
            for record in self.stacks.records
        } if self.stacks is not None else {}
        thread_summaries = []
        if self.threads is not None:
            for thread in self.threads.threads:
                registers = registers_by_uid.get(thread.uid)
                stack = stacks_by_uid.get(thread.uid)
                pc_location = (
                    self.modules.locate(registers.pc)
                    if registers is not None and self.modules is not None
                    else None
                )
                lr_location = (
                    self.modules.locate(registers.lr)
                    if registers is not None and self.modules is not None
                    else None
                )
                stack_captured = bool(
                    registers is not None
                    and any(segment.contains(registers.sp, 4) for segment in core.loads)
                )
                item = {
                    "index": thread.index,
                    "uid": thread.uid,
                    "name": thread.name,
                    "attributes": thread.attributes,
                    "status": thread.status,
                    "status_word": thread.status_word,
                    "status_name": thread.status_name,
                    "entry": thread.entry,
                    "stack_base": thread.stack_base,
                    "stack_size": thread.stack_size,
                    "stack_end": thread.stack_end,
                    "implementation_words": list(thread.implementation_words),
                    "initial_priority": thread.initial_priority,
                    "current_priority": thread.current_priority,
                    "initial_cpu_affinity": thread.initial_cpu_affinity,
                    "current_cpu_affinity": thread.current_cpu_affinity,
                    "current_cpu_id": thread.current_cpu_id,
                    "wait_type": thread.wait_type,
                    "wait_id": thread.wait_id,
                    "run_clocks": thread.run_clocks,
                    "scheduler_word": thread.scheduler_word,
                    "stop_reason": thread.stop_reason,
                    "recorded_pc": thread.recorded_pc,
                    "registers": (
                        {
                            "gpr": list(registers.gpr),
                            "sp": registers.sp,
                            "lr": registers.lr,
                            "pc": registers.pc,
                            "cpsr": registers.cpsr,
                        }
                        if registers is not None
                        else None
                    ),
                    "pc_location": pc_location.summary() if pc_location else None,
                    "lr_location": lr_location.summary() if lr_location else None,
                    "stack_captured_at_sp": stack_captured,
                    "stack_usage": stack.summary() if stack is not None else None,
                    "stack_return_candidates": (
                        self._stack_candidates(core, thread, registers)
                        if registers is not None and stack_captured
                        else []
                    ),
                }
                thread_summaries.append(item)
        non_kernel = [
            item
            for item in thread_summaries
            if item["pc_location"] is not None
            and item["pc_location"]["module_name"] != "SceLibKernel"
        ]
        return {
            "process": self.process.summary() if self.process is not None else None,
            "modules": self.modules.summary() if self.modules is not None else None,
            "thread_info_version": (
                self.threads.format_version if self.threads is not None else None
            ),
            "thread_register_info_version": (
                self.registers.format_version if self.registers is not None else None
            ),
            "stack_info_version": (
                self.stacks.format_version if self.stacks is not None else None
            ),
            "thread_count": len(thread_summaries),
            "threads": thread_summaries,
            "non_kernel_execution_thread_indices": [
                item["index"] for item in non_kernel
            ],
        }
