from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .core import CoreDump, ParseError


def _u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ParseError(f"u32 at 0x{offset:x} exceeds {len(data)}-byte record")
    return struct.unpack_from("<I", data, offset)[0]


def _c_string(data: bytes) -> str:
    return data.split(b"\0", 1)[0].decode("utf-8", errors="replace")


@dataclass(frozen=True)
class KernelWaiter:
    process_id: int
    thread_uid: int

    def summary(self) -> dict:
        return {"process_id": self.process_id, "thread_uid": self.thread_uid}


@dataclass(frozen=True)
class KernelObject:
    index: int
    kind: str
    note_name: str
    uid: int
    process_id: Optional[int]
    name: str
    attributes: int
    state: Dict[str, int]
    waiters: Tuple[KernelWaiter, ...]
    record_size: int
    raw_words: Tuple[int, ...]

    @property
    def owner_thread_uid(self) -> Optional[int]:
        owner = self.state.get("current_owner_id")
        return owner if owner not in (None, 0, 0xFFFFFFFF) else None

    @property
    def waiting_thread_count(self) -> int:
        if self.kind == "message-pipe":
            return self.state["send_wait_threads"] + self.state["receive_wait_threads"]
        return self.state.get("wait_threads", len(self.waiters))

    def summary(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "note_name": self.note_name,
            "uid": self.uid,
            "process_id": self.process_id,
            "name": self.name,
            "attributes": self.attributes,
            "state": self.state,
            "owner_thread_uid": self.owner_thread_uid,
            "waiting_thread_count": self.waiting_thread_count,
            "waiters": [waiter.summary() for waiter in self.waiters],
            "record_size": self.record_size,
            "raw_words": list(self.raw_words),
        }


@dataclass(frozen=True)
class KernelObjectTable:
    note_name: str
    format_version: int
    objects: Tuple[KernelObject, ...]
    trailing: bytes

    def summary(self) -> dict:
        return {
            "note_name": self.note_name,
            "format_version": self.format_version,
            "object_count": len(self.objects),
            "trailing_size": len(self.trailing),
            "objects": [item.summary() for item in self.objects],
        }


def _header(data: bytes, note_name: str, expected_version: int) -> Tuple[int, int]:
    if len(data) < 8:
        raise ParseError(f"{note_name} is truncated")
    version, count = struct.unpack_from("<2I", data)
    if version != expected_version:
        raise ParseError(
            f"unsupported {note_name} version {version:#x}; expected {expected_version:#x}"
        )
    return version, count


def _waiters(data: bytes, offset: int, count: int, note_name: str) -> Tuple[KernelWaiter, ...]:
    required = offset + count * 8
    if required > len(data):
        raise ParseError(f"{note_name} waiter array is truncated")
    return tuple(
        KernelWaiter(*struct.unpack_from("<2I", data, offset + index * 8))
        for index in range(count)
    )


def _parse_semaphores(data: bytes) -> KernelObjectTable:
    version, count = _header(data, "SEMAPHORE_INFO", 1)
    offset = 8
    objects: List[KernelObject] = []
    for index in range(count):
        base_size = 0x44
        if offset + base_size > len(data):
            raise ParseError(f"SEMAPHORE_INFO object {index} is truncated")
        wait_count = _u32(data, offset + 0x40)
        waiters = _waiters(data, offset + base_size, wait_count, "SEMAPHORE_INFO")
        raw = data[offset : offset + base_size]
        objects.append(
            KernelObject(
                index,
                "semaphore",
                "SEMAPHORE_INFO",
                _u32(raw, 4),
                _u32(raw, 8),
                _c_string(raw[0x0C:0x2C]),
                _u32(raw, 0x2C),
                {
                    "capture_status": _u32(raw, 0),
                    "implementation_word": _u32(raw, 0x30),
                    "initial_count": _u32(raw, 0x34),
                    "current_count": _u32(raw, 0x38),
                    "maximum_count": _u32(raw, 0x3C),
                    "wait_threads": wait_count,
                },
                waiters,
                base_size + wait_count * 8,
                struct.unpack("<17I", raw),
            )
        )
        offset += base_size + wait_count * 8
    return KernelObjectTable("SEMAPHORE_INFO", version, tuple(objects), data[offset:])


def _parse_event_flags(data: bytes) -> KernelObjectTable:
    version, count = _header(data, "EVENTFLAG_INFO", 1)
    offset = 8
    objects: List[KernelObject] = []
    for index in range(count):
        base_size = 0x40
        if offset + base_size > len(data):
            raise ParseError(f"EVENTFLAG_INFO object {index} is truncated")
        wait_count = _u32(data, offset + 0x3C)
        waiters = _waiters(data, offset + base_size, wait_count, "EVENTFLAG_INFO")
        raw = data[offset : offset + base_size]
        objects.append(
            KernelObject(
                index,
                "event-flag",
                "EVENTFLAG_INFO",
                _u32(raw, 4),
                _u32(raw, 8),
                _c_string(raw[0x0C:0x2C]),
                _u32(raw, 0x2C),
                {
                    "capture_status": _u32(raw, 0),
                    "implementation_word": _u32(raw, 0x30),
                    "initial_pattern": _u32(raw, 0x34),
                    "current_pattern": _u32(raw, 0x38),
                    "wait_threads": wait_count,
                },
                waiters,
                base_size + wait_count * 8,
                struct.unpack("<16I", raw),
            )
        )
        offset += base_size + wait_count * 8
    return KernelObjectTable("EVENTFLAG_INFO", version, tuple(objects), data[offset:])


def _parse_mutexes(data: bytes) -> KernelObjectTable:
    version, count = _header(data, "MUTEX_INFO", 9)
    offset = 8
    objects: List[KernelObject] = []
    for index in range(count):
        base_size = 0x48
        if offset + base_size > len(data):
            raise ParseError(f"MUTEX_INFO object {index} is truncated")
        wait_count = _u32(data, offset + 0x40)
        waiters = _waiters(data, offset + base_size, wait_count, "MUTEX_INFO")
        raw = data[offset : offset + base_size]
        objects.append(
            KernelObject(
                index,
                "mutex",
                "MUTEX_INFO",
                _u32(raw, 4),
                _u32(raw, 8),
                _c_string(raw[0x0C:0x2C]),
                _u32(raw, 0x2C),
                {
                    "capture_status": _u32(raw, 0),
                    "implementation_word0": _u32(raw, 0x30),
                    "initial_count": _u32(raw, 0x34),
                    "current_count": _u32(raw, 0x38),
                    "current_owner_id": _u32(raw, 0x3C),
                    "wait_threads": wait_count,
                    "implementation_word1": _u32(raw, 0x44),
                },
                waiters,
                base_size + wait_count * 8,
                struct.unpack("<18I", raw),
            )
        )
        offset += base_size + wait_count * 8
    return KernelObjectTable("MUTEX_INFO", version, tuple(objects), data[offset:])


def _parse_lwmutexes(data: bytes) -> KernelObjectTable:
    version, count = _header(data, "LWMUTEX_INFO", 1)
    offset = 8
    objects: List[KernelObject] = []
    for index in range(count):
        base_size = 0x40
        if offset + base_size > len(data):
            raise ParseError(f"LWMUTEX_INFO object {index} is truncated")
        wait_count = _u32(data, offset + 0x3C)
        waiters = _waiters(data, offset + base_size, wait_count, "LWMUTEX_INFO")
        raw = data[offset : offset + base_size]
        objects.append(
            KernelObject(
                index,
                "lightweight-mutex",
                "LWMUTEX_INFO",
                _u32(raw, 4),
                None,
                _c_string(raw[8:0x28]),
                _u32(raw, 0x28),
                {
                    "capture_status": _u32(raw, 0),
                    "work_address": _u32(raw, 0x2C),
                    "initial_count": _u32(raw, 0x30),
                    "current_count": _u32(raw, 0x34),
                    "current_owner_id": _u32(raw, 0x38),
                    "wait_threads": wait_count,
                },
                waiters,
                base_size + wait_count * 8,
                struct.unpack("<16I", raw),
            )
        )
        offset += base_size + wait_count * 8
    return KernelObjectTable("LWMUTEX_INFO", version, tuple(objects), data[offset:])


def _parse_condition_variables(data: bytes) -> KernelObjectTable:
    version, count = _header(data, "CONDVAR_INFO", 1)
    offset = 8
    objects: List[KernelObject] = []
    for index in range(count):
        base_size = 0x3C
        if offset + base_size > len(data):
            raise ParseError(f"CONDVAR_INFO object {index} is truncated")
        wait_count = _u32(data, offset + 0x38)
        waiters = _waiters(data, offset + base_size, wait_count, "CONDVAR_INFO")
        raw = data[offset : offset + base_size]
        objects.append(
            KernelObject(
                index,
                "condition-variable",
                "CONDVAR_INFO",
                _u32(raw, 4),
                _u32(raw, 8),
                _c_string(raw[0x0C:0x2C]),
                _u32(raw, 0x2C),
                {
                    "capture_status": _u32(raw, 0),
                    "implementation_word": _u32(raw, 0x30),
                    "associated_mutex_uid": _u32(raw, 0x34),
                    "wait_threads": wait_count,
                },
                waiters,
                base_size + wait_count * 8,
                struct.unpack("<15I", raw),
            )
        )
        offset += base_size + wait_count * 8
    return KernelObjectTable("CONDVAR_INFO", version, tuple(objects), data[offset:])


def _parse_lwcondition_variables(data: bytes) -> KernelObjectTable:
    version, count = _header(data, "LWCONDVAR_INFO", 6)
    offset = 8
    objects: List[KernelObject] = []
    for index in range(count):
        prefix_size = 0x38
        footer_size = 8
        if offset + prefix_size + footer_size > len(data):
            raise ParseError(f"LWCONDVAR_INFO object {index} is truncated")
        wait_count = _u32(data, offset + 0x34)
        waiters = _waiters(
            data, offset + prefix_size, wait_count, "LWCONDVAR_INFO"
        )
        record_size = prefix_size + wait_count * 8 + footer_size
        if offset + record_size > len(data):
            raise ParseError(f"LWCONDVAR_INFO object {index} footer is truncated")
        raw = data[offset : offset + record_size]
        footer_offset = prefix_size + wait_count * 8
        objects.append(
            KernelObject(
                index,
                "lightweight-condition-variable",
                "LWCONDVAR_INFO",
                _u32(raw, 4),
                None,
                _c_string(raw[8:0x28]),
                _u32(raw, 0x28),
                {
                    "capture_status": _u32(raw, 0),
                    "work_address": _u32(raw, 0x2C),
                    "associated_lwmutex_uid": _u32(raw, 0x30),
                    "wait_threads": wait_count,
                    "implementation_word0": _u32(raw, footer_offset),
                    "implementation_word1": _u32(raw, footer_offset + 4),
                },
                waiters,
                record_size,
                tuple(
                    struct.unpack_from(
                        f"<{min(len(raw), 0x50) // 4}I", raw
                    )
                ),
            )
        )
        offset += record_size
    return KernelObjectTable(
        "LWCONDVAR_INFO", version, tuple(objects), data[offset:]
    )


def _parse_message_pipes(data: bytes) -> KernelObjectTable:
    version, count = _header(data, "MESG_PIPE_INFO", 0x11)
    offset = 8
    objects: List[KernelObject] = []
    for index in range(count):
        if offset + 0x50 > len(data):
            raise ParseError(f"MESG_PIPE_INFO object {index} is truncated")
        record_size = _u32(data, offset)
        if record_size < 0x50 or record_size % 4 or offset + record_size > len(data):
            raise ParseError(
                f"MESG_PIPE_INFO object {index} has invalid size 0x{record_size:x}"
            )
        raw = data[offset : offset + record_size]
        send_waiters = _u32(raw, 0x3C)
        receive_waiters = _u32(raw, 0x40)
        wait_count = send_waiters + receive_waiters
        if 0x50 + wait_count * 8 > record_size:
            raise ParseError(f"MESG_PIPE_INFO object {index} waiter array is truncated")
        waiters = _waiters(raw, 0x50, wait_count, "MESG_PIPE_INFO")
        objects.append(
            KernelObject(
                index,
                "message-pipe",
                "MESG_PIPE_INFO",
                _u32(raw, 4),
                _u32(raw, 8),
                _c_string(raw[0x0C:0x2C]),
                _u32(raw, 0x2C),
                {
                    "implementation_word0": _u32(raw, 0x30),
                    "buffer_size": _u32(raw, 0x34),
                    "free_size": _u32(raw, 0x38),
                    "send_wait_threads": send_waiters,
                    "receive_wait_threads": receive_waiters,
                    "implementation_word1": _u32(raw, 0x44),
                    "implementation_word2": _u32(raw, 0x48),
                    "implementation_word3": _u32(raw, 0x4C),
                },
                waiters,
                record_size,
                struct.unpack_from("<20I", raw),
            )
        )
        offset += record_size
    return KernelObjectTable("MESG_PIPE_INFO", version, tuple(objects), data[offset:])


_PARSERS = (
    ("SEMAPHORE_INFO", _parse_semaphores),
    ("EVENTFLAG_INFO", _parse_event_flags),
    ("MUTEX_INFO", _parse_mutexes),
    ("LWMUTEX_INFO", _parse_lwmutexes),
    ("CONDVAR_INFO", _parse_condition_variables),
    ("LWCONDVAR_INFO", _parse_lwcondition_variables),
    ("MESG_PIPE_INFO", _parse_message_pipes),
)


@dataclass(frozen=True)
class KernelObjectRegistry:
    tables: Tuple[KernelObjectTable, ...]

    @classmethod
    def parse(cls, core: CoreDump) -> "KernelObjectRegistry":
        tables = []
        for note_name, parser in _PARSERS:
            note = core.note(note_name)
            if note is not None:
                tables.append(parser(note.description))
        return cls(tuple(tables))

    @property
    def objects(self) -> Tuple[KernelObject, ...]:
        return tuple(item for table in self.tables for item in table.objects)

    def resolve(self, uid: int) -> Optional[KernelObject]:
        matches = [item for item in self.objects if item.uid == uid]
        return matches[0] if len(matches) == 1 else None

    def summary(self) -> dict:
        objects = self.objects
        active = [
            item
            for item in objects
            if item.waiting_thread_count or item.owner_thread_uid is not None
        ]
        counts: Dict[str, int] = {}
        for item in objects:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return {
            "table_count": len(self.tables),
            "object_count": len(objects),
            "object_counts": counts,
            "active_object_count": len(active),
            "active_objects": [item.summary() for item in active],
            "tables": [table.summary() for table in self.tables],
        }
