from __future__ import annotations

import hashlib
import re
import struct
from typing import List, Optional

from .core import CoreDump, ParseError


_LIBRARY_NAME = re.compile(rb"Sce[A-Za-z0-9_]{1,63}\x00")
_PATH_VALUE = re.compile(rb"(?:ux0:|ur0:|vs0:|os0:)[ -~]*?\x00")
_TITLE_ID = re.compile(rb"[A-Z0-9]{9}\x00")


def _u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ParseError(f"u32 at 0x{offset:x} exceeds {len(data)}-byte record")
    return struct.unpack_from("<I", data, offset)[0]


def _c_string(data: bytes) -> str:
    return data.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _words(data: bytes, limit: Optional[int] = None) -> List[int]:
    size = len(data) if limit is None else min(len(data), limit)
    size &= ~3
    return list(struct.unpack_from(f"<{size // 4}I", data)) if size else []


def _application_info(core: CoreDump) -> Optional[dict]:
    note = core.note("APP_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 0xA4:
        raise ParseError("APP_INFO is shorter than its version-6 fixed prefix")
    version = _u32(data, 0)
    if version != 6:
        raise ParseError(f"unsupported APP_INFO version {version}")
    return {
        "format_version": version,
        "capture_status": _u32(data, 4),
        "title_id": _c_string(data[8:0x12]),
        "title_name": _c_string(data[0x12:0x92]),
        "title_version": _c_string(data[0x92:0x9A]),
        "tail_words": _words(data[0x9C:]),
        "raw_size": len(data),
    }


def _build_info(core: CoreDump) -> Optional[dict]:
    note = core.note("BUILD_VER_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 0xD4:
        raise ParseError("BUILD_VER_INFO is shorter than its version-6 string block")
    version = _u32(data, 0)
    if version != 6:
        raise ParseError(f"unsupported BUILD_VER_INFO version {version}")
    branches = [
        value
        for value in (
            _c_string(data[0x14:0x54]),
            _c_string(data[0x54:0x94]),
            _c_string(data[0x94:0xD4]),
        )
        if value
    ]
    return {
        "format_version": version,
        "header_words": _words(data[:0x14]),
        "branch_strings": branches,
        "unique_branch_strings": list(dict.fromkeys(branches)),
        "tail_words": _words(data[0xD4:]),
        "raw_size": len(data),
    }


def _application_list_info(core: CoreDump) -> Optional[dict]:
    note = core.note("APP_LIST_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 12:
        raise ParseError("APP_LIST_INFO is truncated")
    version, count, producer_word = struct.unpack_from("<3I", data)
    if version != 6:
        raise ParseError(f"unsupported APP_LIST_INFO version {version}")
    matches = []
    for match in _TITLE_ID.finditer(data, 12):
        if match.start() % 4 or match.start() < 8:
            continue
        title_id = match.group()[:-1].decode("ascii")
        if not any(value.isdigit() for value in title_id):
            continue
        matches.append(match)
    records = []
    for index, match in enumerate(matches[:count]):
        following = matches[index + 1].start() if index + 1 < len(matches) else len(data)
        path_match = _PATH_VALUE.search(data, match.end(), following)
        records.append(
            {
                "index": index,
                "title_id": match.group()[:-1].decode("ascii"),
                "process_id": _u32(data, match.start() - 8),
                "parent_process_id": _u32(data, match.start() - 4),
                "path": (
                    path_match.group()[:-1].decode("utf-8", errors="replace")
                    if path_match is not None
                    else ""
                ),
                "title_offset": match.start(),
            }
        )
    return {
        "format_version": version,
        "declared_count": count,
        "producer_word": producer_word,
        "decoded_count": len(records),
        "count_matches": len(records) == count,
        "records": records,
        "raw_size": len(data),
    }


def _external_process_info(core: CoreDump) -> Optional[dict]:
    note = core.note("EXTNL_PROC_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 8:
        raise ParseError("EXTNL_PROC_INFO is truncated")
    version, count = struct.unpack_from("<2I", data)
    if version != 6:
        raise ParseError(f"unsupported EXTNL_PROC_INFO version {version}")
    candidates = []
    for offset in range(0x10, len(data) - 3, 4):
        name = _c_string(data[offset : offset + 32])
        if not 4 <= len(name) <= 31:
            continue
        if not all(character.isalnum() or character == "_" for character in name):
            continue
        process_id = _u32(data, offset - 0x0C)
        uid = _u32(data, offset - 8)
        attributes = _u32(data, offset - 4)
        if process_id in (0, 0xFFFFFFFF) or not 0x10000 <= uid < 0x1000000:
            continue
        candidates.append((offset - 0x10, offset, name, process_id, uid, attributes))
    # A valid record name is the first aligned identifier in its record.  This
    # removes printable fragments which happen to occur in later raw arrays.
    starts = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        if any(
            existing[1] <= candidate[1] < existing[1] + 32
            for existing in starts
        ):
            continue
        if starts and candidate[0] < starts[-1][0] + 0x30:
            continue
        starts.append(candidate)
    records = []
    for index, candidate in enumerate(starts[:count]):
        start, name_offset, name, process_id, uid, attributes = candidate
        end = starts[index + 1][0] if index + 1 < len(starts) else len(data)
        raw = data[start:end]
        path_match = _PATH_VALUE.search(raw)
        records.append(
            {
                "index": index,
                "record_offset": start,
                "record_size": len(raw),
                "capture_status": _u32(data, start),
                "process_id": process_id,
                "uid": uid,
                "attributes": attributes,
                "name": name,
                "name_offset": name_offset,
                "path": (
                    path_match.group()[:-1].decode("utf-8", errors="replace")
                    if path_match is not None
                    else ""
                ),
                "raw_words": _words(raw),
            }
        )
    return {
        "format_version": version,
        "declared_count": count,
        "decoded_count": len(records),
        "count_matches": len(records) == count,
        "records": records,
        "raw_size": len(data),
    }


def _system_info(core: CoreDump) -> Optional[dict]:
    note = core.note("SYSTEM_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 0x60:
        raise ParseError("SYSTEM_INFO is shorter than its version-4 fixed record")
    version = _u32(data, 0)
    if version != 4:
        raise ParseError(f"unsupported SYSTEM_INFO version {version}")
    return {
        "format_version": version,
        "capture_status": _u32(data, 4),
        "software_version_words": [_u32(data, 8), _u32(data, 0x10)],
        "raw_words": _words(data),
    }


def _hardware_info(core: CoreDump) -> Optional[dict]:
    note = core.note("HW_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 0x10:
        raise ParseError("HW_INFO is truncated")
    version = _u32(data, 0)
    if version != 4:
        raise ParseError(f"unsupported HW_INFO version {version}")
    return {
        "format_version": version,
        "capture_status": _u32(data, 4),
        "hardware_revision_word": _u32(data, 8),
        "hardware_flags_word": _u32(data, 0x0C),
        "raw_words": _words(data),
    }


def _corefile_info(core: CoreDump) -> Optional[dict]:
    note = core.note("COREFILE_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 0x28:
        raise ParseError("COREFILE_INFO is shorter than its version-1 record")
    version = _u32(data, 0)
    if version != 1:
        raise ParseError(f"unsupported COREFILE_INFO version {version}")
    return {
        "format_version": version,
        "capture_status": _u32(data, 4),
        "producer_kind": _u32(data, 8),
        "producer_subtype": _u32(data, 0x0C),
        "producer_words": _words(data[0x10:]),
        "raw_size": len(data),
    }


def _device_info(core: CoreDump) -> Optional[dict]:
    note = core.note("DEVICE_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 8 or len(data) % 4:
        raise ParseError("DEVICE_INFO is truncated or not word aligned")
    version = _u32(data, 0)
    if version != 6:
        raise ParseError(f"unsupported DEVICE_INFO version {version}")
    return {
        "format_version": version,
        "capture_status": _u32(data, 4),
        "raw_words": _words(data),
        "raw_size": len(data),
        "process_references": [],
    }


def _file_info(core: CoreDump) -> Optional[dict]:
    note = core.note("FILE_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 8:
        raise ParseError("FILE_INFO is truncated")
    version, count = struct.unpack_from("<2I", data)
    if version != 5:
        raise ParseError(f"unsupported FILE_INFO version {version}")
    record_size = 0x58
    required = 8 + count * record_size
    if required > len(data):
        raise ParseError("FILE_INFO fixed record array is truncated")
    records = []
    for index in range(count):
        raw = data[8 + index * record_size : 8 + (index + 1) * record_size]
        records.append(
            {
                "index": index,
                "capture_status": _u32(raw, 0),
                "uid": _u32(raw, 4),
                "process_id": _u32(raw, 0x10),
                "process_name": None,
                "process_path": None,
                "raw_words": _words(raw),
            }
        )
    return {
        "format_version": version,
        "record_count": len(records),
        "record_size": record_size,
        "records": records,
        "trailing_size": len(data) - required,
    }


def _metadata_info(core: CoreDump) -> Optional[dict]:
    note = core.note("META_DATA_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 8:
        raise ParseError("META_DATA_INFO is truncated")
    version, count = struct.unpack_from("<2I", data)
    if version != 2:
        raise ParseError(f"unsupported META_DATA_INFO version {version}")
    required = 8 + count * 8
    if required > len(data):
        raise ParseError("META_DATA_INFO identifier table is truncated")
    entries = [
        {
            "index": index,
            "identifier": _u32(data, 8 + index * 8),
            "producer_word": _u32(data, 12 + index * 8),
        }
        for index in range(count)
    ]
    return {
        "format_version": version,
        "entry_count": len(entries),
        "entries": entries,
        "payload_words": _words(data[required:]),
        "raw_size": len(data),
    }


def _system_device_info(core: CoreDump) -> Optional[dict]:
    note = core.note("SYS_DEVICE_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 4:
        raise ParseError("SYS_DEVICE_INFO is truncated")
    version = _u32(data, 0)
    if version != 0x11:
        raise ParseError(f"unsupported SYS_DEVICE_INFO version {version:#x}")
    offset = 4
    records = []
    while offset < len(data):
        if offset + 8 > len(data):
            raise ParseError("SYS_DEVICE_INFO record header is truncated")
        size = _u32(data, offset)
        description_size = _u32(data, offset + 4)
        if size < 8 or size % 4 or offset + size > len(data):
            raise ParseError(f"SYS_DEVICE_INFO has invalid record size 0x{size:x}")
        if description_size > size - 8:
            raise ParseError("SYS_DEVICE_INFO description exceeds its record")
        payload_offset = (8 + description_size + 3) & ~3
        if payload_offset > size:
            raise ParseError("SYS_DEVICE_INFO aligned description exceeds its record")
        raw = data[offset : offset + size]
        records.append(
            {
                "index": len(records),
                "record_size": size,
                "description_size": description_size,
                "description": _c_string(raw[8 : 8 + description_size]),
                "payload_words": _words(raw[payload_offset:]),
            }
        )
        offset += size
    return {
        "format_version": version,
        "record_count": len(records),
        "records": records,
    }


def _system_info2(core: CoreDump) -> Optional[dict]:
    note = core.note("SYSTEM_INFO2")
    if note is None:
        return None
    data = note.description
    if len(data) < 0x30:
        raise ParseError("SYSTEM_INFO2 is truncated")
    version = _u32(data, 0)
    if version != 0x10:
        raise ParseError(f"unsupported SYSTEM_INFO2 version {version:#x}")
    header_size = _u32(data, 0x24)
    if header_size < 0x30 or header_size > len(data) or header_size % 4:
        raise ParseError(f"SYSTEM_INFO2 has invalid header size 0x{header_size:x}")
    payload = data[header_size:]
    return {
        "format_version": version,
        "producer_address_word": _u32(data, 4),
        "magic": _u32(data, 8),
        "header_size": header_size,
        "header_words": _words(data[:header_size]),
        "payload_size": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_nonzero_bytes": sum(value != 0 for value in payload),
        "raw_size": len(data),
    }


def _library_info(core: CoreDump) -> Optional[dict]:
    note = core.note("LIBRARY_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 8:
        raise ParseError("LIBRARY_INFO is truncated")
    version, count = struct.unpack_from("<2I", data)
    if version != 6:
        raise ParseError(f"unsupported LIBRARY_INFO version {version}")
    names = []
    for match in _LIBRARY_NAME.finditer(data):
        value = match.group()[:-1].decode("ascii")
        if value not in names:
            names.append(value)
    gpu_tokens = ("gxm", "gpu", "shacc", "shader", "usse", "pvr")
    return {
        "format_version": version,
        "declared_count": count,
        "decoded_name_count": len(names),
        "name_count_matches": len(names) == count,
        "names": names,
        "gpu_related_names": [
            name
            for name in names
            if any(token in name.casefold() for token in gpu_tokens)
        ],
        "raw_size": len(data),
    }


def _budget_info(core: CoreDump) -> Optional[dict]:
    note = core.note("BUDGET_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 8:
        raise ParseError("BUDGET_INFO is truncated")
    version, count = struct.unpack_from("<2I", data)
    if version != 6:
        raise ParseError(f"unsupported BUDGET_INFO version {version}")
    offset = 8
    budgets = []
    all_partitions = []
    for budget_index in range(count):
        if offset + 0x2C > len(data):
            raise ParseError(f"BUDGET_INFO budget {budget_index} is truncated")
        header = data[offset : offset + 0x2C]
        partition_count = _u32(header, 0x28)
        offset += 0x2C
        partitions = []
        for partition_index in range(partition_count):
            if offset + 0x210 > len(data):
                raise ParseError(
                    f"BUDGET_INFO partition {budget_index}:{partition_index} is truncated"
                )
            raw = data[offset : offset + 0x210]
            size = _u32(raw, 0x38)
            free = _u32(raw, 0x3C)
            used = max(0, size - free)
            partition = {
                "index": partition_index,
                "capture_status": _u32(raw, 0),
                "uid": _u32(raw, 4),
                "name": _c_string(raw[8:0x28]),
                "attributes": _u32(raw, 0x28),
                "producer_words": _words(raw[0x2C:0x34]),
                "base": _u32(raw, 0x34),
                "size": size,
                "used": used,
                "free": free,
                "largest_free_block": _u32(raw, 0x40),
                "utilization": used / size if size else None,
                "state_words": _words(raw[0x44:]),
            }
            partitions.append(partition)
            all_partitions.append(partition)
            offset += 0x210
        budgets.append(
            {
                "index": budget_index,
                "capture_status": _u32(header, 0),
                "uid": _u32(header, 4),
                "name": _c_string(header[8:0x28]),
                "partition_count": len(partitions),
                "partitions": partitions,
            }
        )
    nonempty = [item for item in all_partitions if item["size"]]
    highest = max(nonempty, key=lambda item: item["utilization"]) if nonempty else None
    return {
        "format_version": version,
        "budget_count": len(budgets),
        "partition_count": len(all_partitions),
        "budgets": budgets,
        "highest_utilization_partition": highest,
        "trailing_size": len(data) - offset,
        "trailing_nonzero_bytes": sum(value != 0 for value in data[offset:]),
    }


def _event_log_info(core: CoreDump) -> Optional[dict]:
    note = core.note("EVENT_LOG_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 12:
        raise ParseError("EVENT_LOG_INFO is truncated")
    version, ring_word, count = struct.unpack_from("<3I", data)
    if version != 9:
        raise ParseError(f"unsupported EVENT_LOG_INFO version {version}")
    offset = 12
    records = []
    for index in range(count):
        if offset + 4 > len(data):
            raise ParseError(f"EVENT_LOG_INFO record {index} size is truncated")
        size = _u32(data, offset)
        if size < 0x44 or size % 4 or offset + size > len(data):
            raise ParseError(
                f"EVENT_LOG_INFO record {index} has invalid size 0x{size:x}"
            )
        raw = data[offset : offset + size]
        records.append(
            {
                "index": index,
                "record_size": size,
                "title_id": _c_string(raw[8:0x14]),
                "event_code": _u32(raw, 0x14),
                "source_id": _u32(raw, 0x18),
                "detail_code": _u32(raw, 0x1C),
                "producer_words": _words(raw[0x20:0x3C]),
                "payload_size_word": _u32(raw, 0x3C),
                "payload_words": _words(raw[0x40:]),
                "related_id": _u32(raw, 0x44) if size >= 0x48 else None,
                "related_flags": _u32(raw, 0x48) if size >= 0x4C else None,
                "related_name": _c_string(raw[0x50:]) if size > 0x50 else "",
            }
        )
        offset += size
    trailing = data[offset:]
    return {
        "format_version": version,
        "ring_word": ring_word,
        "record_count": len(records),
        "records": records,
        "used_size": offset,
        "trailing_size": len(trailing),
        "trailing_nonzero_bytes": sum(value != 0 for value in trailing),
    }


def _callback_info(core: CoreDump) -> Optional[dict]:
    note = core.note("CALLBACK_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 8:
        raise ParseError("CALLBACK_INFO is truncated")
    version, count = struct.unpack_from("<2I", data)
    if version != 1:
        raise ParseError(f"unsupported CALLBACK_INFO version {version}")
    required = 8 + count * 0x48
    if required > len(data):
        raise ParseError("CALLBACK_INFO fixed record array is truncated")
    records = []
    for index in range(count):
        raw = data[8 + index * 0x48 : 8 + (index + 1) * 0x48]
        records.append(
            {
                "index": index,
                "capture_status": _u32(raw, 0),
                "uid": _u32(raw, 4),
                "process_id": _u32(raw, 8),
                "thread_uid": _u32(raw, 0x0C),
                "name": _c_string(raw[0x10:0x30]),
                "implementation_words": _words(raw[0x30:]),
            }
        )
    return {
        "format_version": version,
        "record_count": len(records),
        "records": records,
        "trailing_size": len(data) - required,
    }


def _timer_info(core: CoreDump) -> Optional[dict]:
    note = core.note("TIMER_INFO")
    if note is None:
        return None
    data = note.description
    if len(data) < 8:
        raise ParseError("TIMER_INFO is truncated")
    version, count = struct.unpack_from("<2I", data)
    if version != 0x11:
        raise ParseError(f"unsupported TIMER_INFO version {version:#x}")
    offset = 8
    records = []
    for index in range(count):
        if offset + 4 > len(data):
            raise ParseError(f"TIMER_INFO record {index} size is truncated")
        size = _u32(data, offset)
        if size < 0x30 or size % 4 or offset + size > len(data):
            raise ParseError(f"TIMER_INFO record {index} has invalid size 0x{size:x}")
        raw = data[offset : offset + size]
        records.append(
            {
                "index": index,
                "record_size": size,
                "uid": _u32(raw, 4),
                "process_id": _u32(raw, 8),
                "name": _c_string(raw[0x0C:0x2C]),
                "state_words": _words(raw[0x2C:]),
            }
        )
        offset += size
    return {
        "format_version": version,
        "record_count": len(records),
        "records": records,
        "trailing_size": len(data) - offset,
    }


def supporting_context(core: CoreDump) -> dict:
    result = {
        "application": _application_info(core),
        "application_list": _application_list_info(core),
        "external_processes": _external_process_info(core),
        "build": _build_info(core),
        "system": _system_info(core),
        "system2": _system_info2(core),
        "hardware": _hardware_info(core),
        "corefile": _corefile_info(core),
        "device": _device_info(core),
        "files": _file_info(core),
        "metadata": _metadata_info(core),
        "system_devices": _system_device_info(core),
        "libraries": _library_info(core),
        "budgets": _budget_info(core),
        "event_log": _event_log_info(core),
        "callbacks": _callback_info(core),
        "timers": _timer_info(core),
    }
    external = result["external_processes"]
    device = result["device"]
    files = result["files"]
    if external is not None:
        processes = {item["process_id"]: item for item in external["records"]}
    else:
        processes = {}
    if processes and device is not None:
        device["process_references"] = [
            {
                "word_index": index,
                "process_id": value,
                "process_name": processes[value]["name"],
            }
            for index, value in enumerate(device["raw_words"])
            if value in processes
        ]
    if processes and files is not None:
        for record in files["records"]:
            process = processes.get(record["process_id"])
            if process is not None:
                record["process_name"] = process["name"]
                record["process_path"] = process["path"]
    return result
