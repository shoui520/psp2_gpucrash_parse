from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .core import ParseError
from .registers import RegisterDatabase, decoded_fields


GPU_INFO_FORMAT_VERSION = 5
GPU_INFO_KNOWN_CAPTURE_FLAGS = 0x3F

_CAPTURE_SECTIONS = (
    (0x01, "master_registers"),
    (0x02, "per_core_registers"),
    (0x04, "indexed_state"),
    (0x08, "context_owners"),
    (0x10, "driver_state"),
    (0x20, "indexed_state_restored"),
)

# SceGpuCoredump's ES4 fault-requestor table.  BIF_INT_STAT.FAULT_REQ is a
# 14-bit bitmap rather than an opaque numeric enumerant.  Keep the order tied
# to the hardware bit positions used by the firmware table.
_BIF_REQUESTORS = (
    "Reserved",
    "Texture Access",
    "Tile Accelerator",
    "VDM Stream or Index Fetch",
    "Transfer",
    "Fragment Processing: Color Surface Write",
    "Hidden Surface Removal: Context Reload",
    "Fragment Interpolation: Parameter Buffer Read",
    "Hidden Surface Removal: Parameter Buffer Read",
    "Hidden Surface Removal: Depth/Stencil Surface or Visibility Buffer",
    "Data Cache (PDS DMA or Shader)",
    "Partial Render Management",
    "Shader Code Fetch",
    "PDS Program Fetch",
)

# SceGpuCoredump likewise treats BIF_INT_STAT.FAULT_TYPE as three independent
# condition bits.  Combined values (notably 5 in the supplied dumps) therefore
# have more than one description.
_BIF_FAULT_CONDITIONS = (
    "Page not mapped",
    "Read/write access error",
    "System page access",
)


def _c_string(data: bytes) -> str:
    return data.split(b"\0", 1)[0].decode("utf-8", errors="replace")


@dataclass(frozen=True)
class MasterRegister:
    offset: int
    value: int


@dataclass(frozen=True)
class PerCoreRegister:
    offset: int
    values: Tuple[int, ...]


@dataclass(frozen=True)
class GpuContextOwner:
    slot: int
    process_id: int
    process_name: str

    @property
    def populated(self) -> bool:
        return self.process_id != 0xFFFFFFFF or bool(self.process_name)

    def summary(self) -> dict:
        return {
            "slot": self.slot,
            "process_id": self.process_id,
            "process_name": self.process_name,
            "populated": self.populated,
        }


@dataclass(frozen=True)
class GpuTrailer:
    """Structurally known portion of the version-5 register-dump trailer."""

    context_owners: Tuple[GpuContextOwner, ...]
    context_slot_map: Tuple[Tuple[int, ...], ...]
    driver_state_words: Tuple[int, ...]
    remainder_offset: int
    remainder: bytes
    raw: bytes

    @classmethod
    def parse(cls, data: bytes) -> "GpuTrailer":
        owners: Tuple[GpuContextOwner, ...] = ()
        slot_map: Tuple[Tuple[int, ...], ...] = ()
        state: Tuple[int, ...] = ()
        offset = 0
        if len(data) >= 0xC0:
            owners = tuple(
                GpuContextOwner(
                    slot=index,
                    process_id=struct.unpack_from("<I", data, index * 0x18)[0],
                    process_name=_c_string(data[index * 0x18 + 4 : (index + 1) * 0x18]),
                )
                for index in range(8)
            )
            offset = 0xC0
        if len(data) >= 0xE8:
            slot_map = tuple(
                tuple(data[0xC0 + row * 8 : 0xC0 + (row + 1) * 8])
                for row in range(5)
            )
            offset = 0xE8
        if len(data) >= 0xF8:
            state = struct.unpack_from("<4I", data, 0xE8)
            offset = 0xF8
        return cls(owners, slot_map, state, offset, data[offset:], data)

    def summary(self) -> dict:
        return {
            "size": len(self.raw),
            "sha256": hashlib.sha256(self.raw).hexdigest(),
            "context_owners": [owner.summary() for owner in self.context_owners],
            "context_slot_map": [list(row) for row in self.context_slot_map],
            "driver_state_words": list(self.driver_state_words),
            "remainder_offset": self.remainder_offset,
            "remainder_size": len(self.remainder),
            "remainder_sha256": hashlib.sha256(self.remainder).hexdigest(),
        }


@dataclass
class GpuInfo:
    format_version: int
    header_words: Tuple[int, ...]
    core_count: int
    master_registers: List[MasterRegister]
    per_core_registers: List[PerCoreRegister]
    indexed_state: List[PerCoreRegister]
    trailer: bytes

    @classmethod
    def parse(cls, data: bytes) -> "GpuInfo":
        if len(data) < 15 * 4 or len(data) % 4:
            raise ParseError("GPU_INFO is truncated or not word aligned")
        words = struct.unpack(f"<{len(data) // 4}I", data)
        header = tuple(words[:15])
        version = header[0]
        if version != GPU_INFO_FORMAT_VERSION:
            raise ParseError(f"unsupported GPU_INFO version {version}")
        core_count = header[11]
        master_count = header[12]
        per_core_count = header[13]
        indexed_count = header[14]
        if core_count == 0 or core_count > 16:
            raise ParseError(f"implausible GPU core count {core_count}")
        offset = 15
        required_words = (
            offset
            + master_count * 2
            + per_core_count * (1 + core_count)
            + indexed_count * (1 + core_count)
        )
        if required_words > len(words):
            raise ParseError(
                f"GPU_INFO tables require {required_words} words, have {len(words)}"
            )
        master = [
            MasterRegister(words[offset + index * 2], words[offset + index * 2 + 1])
            for index in range(master_count)
        ]
        offset += master_count * 2
        per_core: List[PerCoreRegister] = []
        for _ in range(per_core_count):
            register_offset = words[offset]
            per_core.append(
                PerCoreRegister(register_offset, tuple(words[offset + 1 : offset + 1 + core_count]))
            )
            offset += 1 + core_count
        indexed: List[PerCoreRegister] = []
        for _ in range(indexed_count):
            index = words[offset]
            indexed.append(
                PerCoreRegister(index, tuple(words[offset + 1 : offset + 1 + core_count]))
            )
            offset += 1 + core_count
        trailer = data[offset * 4 :]
        return cls(version, header, core_count, master, per_core, indexed, trailer)

    def master_value(self, offset: int) -> Optional[int]:
        for register in self.master_registers:
            if register.offset == offset:
                return register.value
        return None

    def core_values(self, offset: int) -> Optional[Tuple[int, ...]]:
        for register in self.per_core_registers:
            if register.offset == offset:
                return register.values
        return None

    @property
    def register_dump_version(self) -> int:
        return self.header_words[1]

    @property
    def capture_flags(self) -> int:
        return self.header_words[2]

    def capture_sections(self) -> dict:
        return {
            name: bool(self.capture_flags & mask)
            for mask, name in _CAPTURE_SECTIONS
        }

    def snapshot_status(self) -> dict:
        missing = [name for name, captured in self.capture_sections().items() if not captured]
        unknown_flags = self.capture_flags & ~GPU_INFO_KNOWN_CAPTURE_FLAGS
        count_expectations = {
            "master_registers": 0x10A,
            "per_core_registers": 0x18F,
            "indexed_state": 0x6D,
        }
        observed_counts = {
            "master_registers": len(self.master_registers),
            "per_core_registers": len(self.per_core_registers),
            "indexed_state": len(self.indexed_state),
        }
        unexpected_counts = {
            name: {"expected": count_expectations[name], "observed": observed}
            for name, observed in observed_counts.items()
            if observed != count_expectations[name]
        }
        return {
            "complete": not missing and unknown_flags == 0,
            "canonical_3_68_counts": not unexpected_counts,
            "capture_flags": self.capture_flags,
            "sections": self.capture_sections(),
            "missing_sections": missing,
            "unknown_flags": unknown_flags,
            "unexpected_counts": unexpected_counts,
        }

    def decoded_trailer(self) -> GpuTrailer:
        return GpuTrailer.parse(self.trailer)

    def bif_diagnosis(self) -> dict:
        fault_sources = []
        master_status = self.master_value(0x4C04)
        master_fault = self.master_value(0x4C08)
        if master_status is not None and master_status & 0x00073FFF:
            fault_sources.append(_fault_source("master", master_status, master_fault or 0))
        elif master_fault:
            fault_sources.append(_fault_source("master", master_status or 0, master_fault))
        core_status = self.core_values(0x0C04)
        core_fault = self.core_values(0x0C08)
        for core in range(self.core_count):
            status = core_status[core] if core_status is not None else 0
            fault = core_fault[core] if core_fault is not None else 0
            if status & 0x00073FFF or fault:
                fault_sources.append(_fault_source(f"core{core}", status, fault))
        bif_sections_captured = bool(self.capture_flags & 0x03)
        classification = (
            "bif-memory-fault"
            if fault_sources
            else "no-bif-memory-fault"
            if bif_sections_captured
            else "bif-state-unavailable"
        )
        return {
            "classification": classification,
            "fault_sources": fault_sources,
            "master_status": master_status,
            "master_fault": master_fault,
            "core_status": list(core_status) if core_status is not None else None,
            "core_fault": list(core_fault) if core_fault is not None else None,
            "qualification": (
                "BIF/MMU fault state present in captured registers."
                if fault_sources
                else "Captured BIF/MMU fault fields clear; command-stream, shader, firmware-timeout, and pre-snapshot faults remain possible."
                if bif_sections_captured
                else "Register-dump validity bitmap does not mark BIF register sections as captured."
            ),
        }

    def summary(self, database: RegisterDatabase, include_registers: bool = True) -> dict:
        result = {
            "format_version": self.format_version,
            "header_words": list(self.header_words),
            "register_dump_version": self.register_dump_version,
            "snapshot": self.snapshot_status(),
            "producer_words": list(self.header_words[3:11]),
            "core_count": self.core_count,
            "master_register_count": len(self.master_registers),
            "per_core_register_count": len(self.per_core_registers),
            "indexed_state_count": len(self.indexed_state),
            "trailer_size": len(self.trailer),
            "trailer_sha256": hashlib.sha256(self.trailer).hexdigest(),
            "trailer": self.decoded_trailer().summary(),
            "bif": self.bif_diagnosis(),
        }
        if include_registers:
            result["master_registers"] = [
                _master_summary(item, database) for item in self.master_registers
            ]
            result["per_core_registers"] = [
                _core_summary(item, database) for item in self.per_core_registers
            ]
            result["indexed_state"] = [
                {"index": item.offset, "values": list(item.values)}
                for item in self.indexed_state
            ]
        return result


def _fault_source(scope: str, status: int, fault: int) -> dict:
    requestor = status & 0x3FFF
    fault_type = (status >> 16) & 0x7
    return {
        "scope": scope,
        "status": status,
        "fault": fault,
        "status_latched": bool(status & 0x00073FFF),
        "fault_register_nonzero": bool(fault),
        "requestor": requestor,
        "requestor_names": [
            name
            for bit, name in enumerate(_BIF_REQUESTORS)
            if requestor & (1 << bit)
        ],
        "fault_type": fault_type,
        "fault_conditions": [
            name
            for bit, name in enumerate(_BIF_FAULT_CONDITIONS)
            if fault_type & (1 << bit)
        ],
        "context_id": fault & 0xF,
        "sideband": (fault >> 4) & 0x1F,
        "address": fault & 0xFFFFF000,
    }


def _master_summary(register: MasterRegister, database: RegisterDatabase) -> dict:
    definition = database.definition(register.offset, master=True)
    return {
        "offset": register.offset,
        "name": definition.display_name if definition else None,
        "aliases": definition.names if definition else [],
        "value": register.value,
        "fields": decoded_fields(definition, register.value),
    }


def _core_summary(register: PerCoreRegister, database: RegisterDatabase) -> dict:
    definition = database.definition(register.offset, master=False)
    return {
        "offset": register.offset,
        "name": definition.display_name if definition else None,
        "aliases": definition.names if definition else [],
        "values": list(register.values),
        "fields": [decoded_fields(definition, value) for value in register.values],
    }


def compare_gpu_info(left: GpuInfo, right: GpuInfo, database: RegisterDatabase) -> dict:
    left_master = {item.offset: item.value for item in left.master_registers}
    right_master = {item.offset: item.value for item in right.master_registers}
    master_changes = []
    for offset in sorted(set(left_master) | set(right_master)):
        old = left_master.get(offset)
        new = right_master.get(offset)
        if old == new:
            continue
        definition = database.definition(offset, master=True)
        master_changes.append(
            {
                "offset": offset,
                "name": definition.display_name if definition else None,
                "left": old,
                "right": new,
            }
        )
    left_core = {item.offset: item.values for item in left.per_core_registers}
    right_core = {item.offset: item.values for item in right.per_core_registers}
    core_changes = []
    for offset in sorted(set(left_core) | set(right_core)):
        old = left_core.get(offset)
        new = right_core.get(offset)
        if old == new:
            continue
        definition = database.definition(offset, master=False)
        core_changes.append(
            {
                "offset": offset,
                "name": definition.display_name if definition else None,
                "left": list(old) if old is not None else None,
                "right": list(new) if new is not None else None,
            }
        )
    left_indexed = {item.offset: item.values for item in left.indexed_state}
    right_indexed = {item.offset: item.values for item in right.indexed_state}
    indexed_changes = [
        {
            "index": index,
            "left": list(left_indexed.get(index, ())),
            "right": list(right_indexed.get(index, ())),
        }
        for index in sorted(set(left_indexed) | set(right_indexed))
        if left_indexed.get(index) != right_indexed.get(index)
    ]
    left_trailer = left.decoded_trailer()
    right_trailer = right.decoded_trailer()
    left_owners = {
        owner.slot: (owner.process_id, owner.process_name)
        for owner in left_trailer.context_owners
    }
    right_owners = {
        owner.slot: (owner.process_id, owner.process_name)
        for owner in right_trailer.context_owners
    }
    owner_changes = [
        {
            "slot": slot,
            "left": (
                {"process_id": left_owners[slot][0], "process_name": left_owners[slot][1]}
                if slot in left_owners
                else None
            ),
            "right": (
                {"process_id": right_owners[slot][0], "process_name": right_owners[slot][1]}
                if slot in right_owners
                else None
            ),
        }
        for slot in sorted(set(left_owners) | set(right_owners))
        if left_owners.get(slot) != right_owners.get(slot)
    ]
    driver_state_changes = []
    for index in range(
        max(
            len(left_trailer.driver_state_words),
            len(right_trailer.driver_state_words),
        )
    ):
        old = (
            left_trailer.driver_state_words[index]
            if index < len(left_trailer.driver_state_words)
            else None
        )
        new = (
            right_trailer.driver_state_words[index]
            if index < len(right_trailer.driver_state_words)
            else None
        )
        if old != new:
            driver_state_changes.append({"word": index, "left": old, "right": new})
    return {
        "header_changes": [
            {"word": index, "left": old, "right": new}
            for index, (old, new) in enumerate(zip(left.header_words, right.header_words))
            if old != new
        ],
        "left_bif": left.bif_diagnosis(),
        "right_bif": right.bif_diagnosis(),
        "left_snapshot": left.snapshot_status(),
        "right_snapshot": right.snapshot_status(),
        "master_changes": master_changes,
        "per_core_changes": core_changes,
        "indexed_state_changes": indexed_changes,
        "context_owner_changes": owner_changes,
        "context_slot_map_changed": left_trailer.context_slot_map
        != right_trailer.context_slot_map,
        "driver_state_changes": driver_state_changes,
    }
