"""The tagged gpu_core_dump binary written by SceGpuEs4CoreDump.

This capture is independent of the GPU_INFO note. Keep both snapshots intact
when correlating them; counters and debug-port state can change between reads.
"""

import hashlib
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ._des import decrypt_cbc
from .core import CoreDump, ParseError
from .gpu import GpuInfo, MasterRegister, PerCoreRegister
from .indexed import indexed_selector_metadata
from .pipeline import firmware_pipeline_diagnosis
from .registers import RegisterDatabase, decoded_fields


_FILE_MAGIC = 0x01727A72
_BODY_TAG = 0x01304000
_VERSION = 0x00020002
_MAX_FILE_SIZE = 16 * 1024 * 1024
# The original segment descriptors contain these format constants. DES uses
# the numeric keys in MSB-first order; each segment resets its own CBC chain.
_KEY_A = bytes.fromhex("12e23116202ede58")
_KEY_B = bytes.fromhex("5f385e7f2cf4b7f8")
_IV_A = bytes.fromhex("16b193e7f7b01ae0")
_IV_B = bytes.fromhex("4382f668db8bdc3f")
_SEGMENTS = {
    0x81304001: ("master", _KEY_A, _IV_A, 8),
    0x81304002: ("core_debug", _KEY_A, _IV_A, 20),
    0x81304003: ("master_debug", _KEY_B, _IV_B, 8),
    0x81304004: ("core", _KEY_B, _IV_B, 20),
    0x81304005: ("indexed", _KEY_B, _IV_B, 20),
    0x81304006: ("header", None, None, None),
    0x81304007: ("analysis_text", None, None, None),
    0x81304008: ("processes", _KEY_B, _IV_B, None),
    0x81304009: ("render_state", None, None, None),
}


def _decode_table(data: bytes, stride: int, name: str) -> tuple:
    if len(data) < 8:
        raise ParseError(f"GPU dump {name} table is shorter than its count")
    capacity = (len(data) - 4) // stride
    count = struct.unpack_from("<I", data)[0]
    if count > capacity:
        raise ParseError(
            f"GPU dump {name} count {count} exceeds segment capacity {capacity}"
        )
    end = 4 + capacity * stride
    padding = data[end:]
    # Finish pads only an incomplete DES block, never a full block. Do not
    # strip PKCS#7 blindly: an aligned process record can end in any byte.
    if len(padding) not in (0, 4) or padding != bytes([len(padding)]) * len(padding):
        raise ParseError(f"GPU dump {name} has invalid encrypted table padding")
    rows = [
        struct.unpack_from("<" + "I" * (stride // 4), data, 4 + index * stride)
        for index in range(count)
    ]
    return rows, capacity, len(padding)


def _clean_text(data: bytes) -> str:
    text = data.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    # Sony supplies a carriage return as the printf prefix ("\r: ...").
    # Remove that presentation prefix, keeping an exact byte view separately.
    return "\n".join(
        line[2:] if line.startswith(": ") else line
        for line in text.replace("\r", "").splitlines()
    ) + ("\n" if text else "")


def _native_text_facts(text: str) -> dict:
    scope = "master"
    fault_processes = []
    active_processes = []
    partial_renders = None
    for line in text.splitlines():
        core = re.fullmatch(r"GPU Core (\d+) Analysis:", line)
        if core:
            scope = f"core{core.group(1)}"
        process = re.fullmatch(r"\s*- Process : (.+) \(0x([0-9a-fA-F]{8})\)", line)
        if process:
            fault_processes.append({
                "scope": scope,
                "process_name": process.group(1),
                "process_id": int(process.group(2), 16),
            })
        active = re.fullmatch(
            r"- (Vertex|Fragment): Active Process is : (.+) \(0x([0-9a-fA-F]{8})\)", line
        )
        if active:
            active_processes.append({
                "scope": scope,
                "phase": active.group(1).lower(),
                "process_name": active.group(2),
                "process_id": int(active.group(3), 16),
            })
        partial = re.fullmatch(r"Partial renders since boot: (\d+)", line)
        if partial:
            partial_renders = int(partial.group(1))
    return {
        "source": "embedded firmware analysis text",
        "fault_processes": fault_processes,
        "active_processes": active_processes,
        "partial_renders_since_boot": partial_renders,
    }


@dataclass
class DevkitGpuDump:
    path: Path
    data: bytes
    header: dict
    segments: List[dict]
    tables: dict
    processes: list
    render_state: Optional[dict]
    text_bytes: bytes

    @classmethod
    def read(cls, path: Path) -> "DevkitGpuDump":
        path = Path(path)
        with path.open("rb") as source:
            data = source.read(_MAX_FILE_SIZE + 1)
        return cls.parse(data, path)

    @classmethod
    def parse(cls, data: bytes, path: Path = Path("gpu-dump.bin")) -> "DevkitGpuDump":
        if len(data) > _MAX_FILE_SIZE:
            raise ParseError("GPU dump exceeds the 16 MiB reader limit")
        if len(data) < 16:
            raise ParseError("GPU dump is shorter than its file and body headers")
        magic, file_size, body, body_size = struct.unpack_from("<4I", data)
        if magic != _FILE_MAGIC or body != _BODY_TAG:
            raise ParseError("not a tagged SceGpuEs4CoreDump GPU dump")
        if file_size != len(data) - 8 or body_size != len(data) - 16:
            raise ParseError("GPU dump declared file/body sizes do not match the file extent")
        segments = []
        payloads = {}
        tables = {}
        offset = 16
        while offset < len(data):
            if len(data) - offset < 8:
                raise ParseError("GPU dump has a truncated segment header")
            tag, size = struct.unpack_from("<2I", data, offset)
            end = offset + 8 + size
            if size % 8 or end > len(data):
                raise ParseError(f"GPU dump segment {tag:#x} is truncated or not 8-byte aligned")
            name, key, iv, stride = _SEGMENTS.get(tag, ("unknown", None, None, None))
            if tag in _SEGMENTS and tag in payloads:
                raise ParseError(f"GPU dump contains duplicate {name} segments")
            stored = data[offset + 8 : end]
            segment = {
                "tag": tag, "name": name, "file_offset": offset,
                "payload_offset": offset + 8, "stored_size": size,
                "encoding": "des-cbc" if key else "plain" if tag in _SEGMENTS else "unknown",
                "stored_sha256": hashlib.sha256(stored).hexdigest(),
            }
            if tag in _SEGMENTS:
                decoded = decrypt_cbc(stored, key, iv) if key else stored
                payloads[tag] = decoded
                if stride is not None:
                    rows, capacity, padding = _decode_table(decoded, stride, name)
                    tables[name] = rows
                    segment.update({
                        "record_count": len(rows), "record_capacity": capacity,
                        "padding_size": padding,
                    })
            else:
                segment["opaque_payload_hex"] = stored.hex()
            segments.append(segment)
            offset = end
        header_data = payloads.get(0x81304006)
        if header_data is None or len(header_data) != 16:
            raise ParseError("GPU dump requires a 16-byte version header segment")
        version, sdk, build, soc = struct.unpack("<4I", header_data)
        if version != _VERSION:
            raise ParseError(f"unsupported GPU dump format version {version:#x}")
        header = {
            "format_version": version, "sdk_version": sdk,
            "gpu_driver_build": build, "soc_revision": soc,
        }
        process_data = payloads.get(0x81304008)
        processes = []
        if process_data is not None:
            if len(process_data) != 8 * 32:
                raise ParseError("GPU dump process segment must contain eight 32-byte records")
            for index in range(8):
                pid = struct.unpack_from("<q", process_data, index * 32)[0]
                name = process_data[index * 32 + 8 : (index + 1) * 32].split(b"\0", 1)[0]
                processes.append({
                    "data_master": index, "process_id": pid,
                    "process_name": name.decode("utf-8", errors="replace"),
                    "populated": pid >= 0,
                })
        render_data = payloads.get(0x81304009)
        render_state = None
        if render_data is not None:
            if len(render_data) != 16:
                raise ParseError("GPU dump render-state segment must contain four words")
            render_state = dict(zip(
                ("vertex_frame", "fragment_frame", "vertex_scene", "fragment_scene"),
                struct.unpack("<4I", render_data),
            ))
        return cls(Path(path), data, header, segments, tables, processes,
                   render_state, payloads.get(0x81304007, b""))

    def register_state(self) -> GpuInfo:
        """Adapt register tables for shared decoders; no GPU_INFO is fabricated on disk."""
        master = [
            MasterRegister(*row)
            for name in ("master", "master_debug")
            for row in self.tables.get(name, [])
        ]
        per_core = [
            PerCoreRegister(row[0], tuple(row[1:]))
            for name in ("core_debug", "core")
            for row in self.tables.get(name, [])
        ]
        indexed = [
            PerCoreRegister(row[0], tuple(row[1:]))
            for row in self.tables.get("indexed", [])
        ]
        core_count = 4
        core_id = next((item.value for item in master if item.offset == 0x4000), None)
        if core_id is not None:
            core_count = (core_id & 3) + 1
        flags = (1 if master else 0) | (2 if per_core else 0) | (4 if indexed else 0)
        header = (5, 0, flags, 0, 0, 0, 0, self.header["soc_revision"],
                  0, 0, 0, core_count, len(master), len(per_core), len(indexed))
        return GpuInfo(5, header, core_count, master, per_core, indexed, b"")

    def summary(self, database: RegisterDatabase) -> dict:
        state = self.register_state()
        tables = {}
        for name, rows in self.tables.items():
            records = []
            for row in rows:
                master = name in ("master", "master_debug")
                definition = database.definition(row[0], master) if name != "indexed" else None
                metadata = indexed_selector_metadata(row[0]) if name == "indexed" else None
                record = {
                    "offset" if name != "indexed" else "selector": row[0],
                    "name": definition.display_name if definition else metadata["name"] if metadata else None,
                }
                if name == "indexed":
                    record["consumer_interpreted"] = metadata is not None
                if master:
                    record.update({"value": row[1], "fields": decoded_fields(definition, row[1])})
                else:
                    record.update({
                        "values": list(row[1:]),
                        "fields": [decoded_fields(definition, value) for value in row[1:]],
                    })
                records.append(record)
            tables[name] = records
        text = _clean_text(self.text_bytes)
        pipeline = firmware_pipeline_diagnosis(state)
        pipeline["decoder"] = "SceGpuEs4CoreDump ES4 diagnostic tables"
        return {
            "path": str(self.path), "size": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "header": self.header, "core_count": state.core_count,
            "segments": self.segments, "register_tables": tables,
            "missing_segments": [
                name for tag, (name, _key, _iv, _stride) in _SEGMENTS.items()
                if tag not in {item["tag"] for item in self.segments}
            ],
            "processes": self.processes,
            "process_table_scope": (
                "Data-master process IDs resolved using core 0 directory registers; "
                "indices are not GPU_INFO owner-slot indices."
            ),
            "render_state": self.render_state,
            "bif": state.bif_diagnosis(),
            "firmware_pipeline": pipeline,
            "native_diagnostics": _native_text_facts(text),
            "analysis_text": text,
            "analysis_text_raw_hex": self.text_bytes.hex(),
            "limitations": [
                "Register snapshots are captured independently of GPU_INFO; shared values can differ.",
                "This file contains diagnostic state, not GPU memory contents or page tables.",
                "DES-CBC segments have structural checks but no cryptographic authentication.",
            ],
        }


def _table_comparison(left: list, right: list, scalar: bool = False) -> dict:
    def values(rows):
        result = {}
        for row in rows:
            # Match the shared register decoder's first-observation lookup.
            # Raw tables above retain every duplicate, with its source table.
            result.setdefault(row.offset, row.value if scalar else tuple(row.values))
        return result

    native, note = values(left), values(right)
    common = sorted(set(native) & set(note))
    return {
        "shared_count": len(common),
        "matching_count": sum(native[key] == note[key] for key in common),
        "changed": [
            {"offset": key, "gpu_dump": native[key], "gpu_info": note[key]}
            for key in common if native[key] != note[key]
        ],
        "gpu_dump_only": [
            {"offset": key, "value" if scalar else "values": native[key]}
            for key in sorted(set(native) - set(note))
        ],
        "gpu_info_only": [
            {"offset": key, "value" if scalar else "values": note[key]}
            for key in sorted(set(note) - set(native))
        ],
    }


def _owner_name_match(note_name: str, native_name: str) -> Optional[str]:
    if note_name == native_name:
        return "exact"
    # GPU_INFO has a 20-byte name field; the sidecar process record has 24
    # bytes and its diagnostic printf may contain the complete GUID name.
    if len(note_name.encode("utf-8")) == 20 and native_name.startswith(note_name):
        return "truncated-prefix"
    return None


def correlate_gpu_dump(native: DevkitGpuDump, core: CoreDump, info: GpuInfo) -> dict:
    state = native.register_state()
    native_faults = [item for item in state.bif_diagnosis()["fault_sources"] if item["status_latched"]]
    core_faults = [item for item in info.bif_diagnosis()["fault_sources"] if item["status_latched"]]
    fault_fields = ("scope", "address", "context_id", "requestor", "fault_type", "page_access_type", "sideband")

    def signature(faults):
        return sorted(tuple(item[field] for field in fault_fields) for item in faults)

    faults_match = signature(native_faults) == signature(core_faults) if native_faults or core_faults else None
    trailer = info.decoded_trailer()
    owners = {
        item.process_id: item.process_name
        for item in trailer.context_owners if item.populated
    }
    matching_processes = []
    conflicting_processes = []
    # The process tables use different index spaces. Compare IDs and names,
    # never the positions of their records.
    for process in native.processes:
        pid = process["process_id"]
        if pid not in owners:
            continue
        name_match = _owner_name_match(owners[pid], process["process_name"])
        if name_match:
            matching_processes.append({**process, "gpu_info_name": owners[pid], "name_match": name_match})
        else:
            conflicting_processes.append({**process, "gpu_info_name": owners[pid]})
    fault_process_checks = []
    for process in _native_text_facts(_clean_text(native.text_bytes))["fault_processes"]:
        fault = next((item for item in native_faults if item["scope"] == process["scope"]), None)
        if fault is None:
            continue
        context = fault["context_id"]
        slots = {
            row[context] for row in trailer.context_slot_map
            if context < len(row) and row[context] != 0xFF
        }
        owner = None
        if len(slots) == 1:
            slot = next(iter(slots))
            if slot < len(trailer.context_owners) and trailer.context_owners[slot].populated:
                owner = trailer.context_owners[slot]
        check = "unresolved"
        if owner is not None:
            check = "match" if owner.process_id == process["process_id"] and _owner_name_match(
                owner.process_name, process["process_name"]
            ) else "mismatch"
        fault_process_checks.append({
            "scope": process["scope"], "context_id": context, "status": check,
            "gpu_dump_pid": process["process_id"], "gpu_dump_name": process["process_name"],
            "gpu_info_pid": owner.process_id if owner else None,
            "gpu_info_name": owner.process_name if owner else None,
        })
    soc_matches = native.header["soc_revision"] == info.hardware_clock_state()["soc_revision"]
    comparisons = {
        "master": _table_comparison(state.master_registers, info.master_registers, True),
        "per_core": _table_comparison(state.per_core_registers, info.per_core_registers),
        "indexed": _table_comparison(state.indexed_state, info.indexed_state),
    }
    # A matching fault by itself (especially a repeated null-page fault) is
    # insufficient. Require process identity and substantial register overlap.
    shared = sum(item["shared_count"] for item in comparisons.values())
    matching = sum(item["matching_count"] for item in comparisons.values())
    substantial_overlap = shared >= 16 and matching >= shared * 0.9
    status = "inconclusive"
    owner_conflict = any(item["status"] == "mismatch" for item in fault_process_checks)
    owner_corroborated = not fault_process_checks or all(
        item["status"] == "match" for item in fault_process_checks
    )
    if faults_match is False or not soc_matches or conflicting_processes or owner_conflict or state.core_count != info.core_count:
        status = "mismatch"
    elif faults_match is True and matching_processes and substantial_overlap and owner_corroborated:
        status = "consistent"
    timestamp_hint = None
    match = re.fullmatch(r"gpu_core_dump_(\d{4})_(\d{2})_(\d{2})_(\d{12})\.bin", native.path.name)
    if match:
        try:
            stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S%f")
            core_timestamp = info.capture_timestamp()["iso_local"]
            timestamp_hint = {
                "source": "sidecar filename, not embedded metadata",
                "iso_local": stamp.isoformat(timespec="microseconds"),
                "delta_from_gpu_info_microseconds": (
                    round((stamp - datetime.fromisoformat(core_timestamp)).total_seconds() * 1_000_000)
                    if core_timestamp else None
                ),
            }
        except ValueError:
            pass
    return {
        "core_path": str(core.path), "status": status,
        "fault_signature_match": faults_match,
        "soc_revision_match": soc_matches,
        "core_count_match": state.core_count == info.core_count,
        "matching_processes": matching_processes,
        "conflicting_processes": conflicting_processes,
        "fault_process_checks": fault_process_checks,
        "register_comparisons": comparisons,
        "filename_timestamp_hint": timestamp_hint,
        "qualification": (
            "Correlation checks fault fields, process identities, SoC, core count, and "
            "shared register values. No embedded event ID binds the two files; "
            "consistent evidence does not prove they came from the same event. "
            "Register comparisons count unique offsets using the first observation; "
            "raw tables retain duplicates. "
            "Snapshots remain separate and GPU_INFO values are never overwritten."
        ),
    }
