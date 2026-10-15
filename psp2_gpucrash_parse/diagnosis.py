from __future__ import annotations

import hashlib
import re
import shlex
from typing import List, Optional, Sequence, Tuple

from .activity import GpuActivityInfo
from .core import CoreDump, ParseError
from .execution import ExecutionContext
from .gpu import GPU_CONTEXT_MAP_SCOPES, GpuInfo, GpuTrailer
from .kernel import KernelObjectRegistry
from .metadata import SummaryInfo
from .pipeline import firmware_pipeline_diagnosis
from .registers import RegisterDatabase
from .support import supporting_context
from .tty import TtyInfo


# These are error or resource-pressure latches, rather than ordinary completion
# events.  Names and masks come directly from the SGX543/SGXMP hardware headers.
_EVENT_SIGNALS = (
    (0x10000000, "ta-dpm-fault", "TA/DPM fault event", "error"),
    (0x02000000, "zls-out-of-memory", "depth/stencil (ZLS) out-of-memory event", "warning"),
    (0x00000004, "global-out-of-memory", "global DPM out-of-memory event", "warning"),
    (0x00000002, "macrotile-out-of-memory", "macrotile DPM out-of-memory event", "warning"),
    (0x00008000, "breakpoint", "GPU breakpoint event", "warning"),
)

_MASTER_ONLY_EVENT_SIGNALS = (
    (0x00020000, "isp-halt", "ISP halt event", "error"),
    (0x00010000, "isp-visibility-failure", "ISP visibility failure", "error"),
)

_DPM_DEBUG_SIGNALS = (
    (0x00000001, "dpm-out-of-memory-deadlock", "DPM out-of-memory deadlock", "error"),
    (0x00000002, "dpm-out-of-memory-sleep", "DPM sleeping after out-of-memory", "warning"),
    (0x00000004, "dpm-out-of-memory-wait", "DPM waiting after out-of-memory", "warning"),
)

_STATUS2_SIGNALS = (
    (0x00000800, "data-breakpoint-untrapped", "untrapped GPU data breakpoint", "error"),
    (0x00000400, "data-breakpoint-trapped", "trapped GPU data breakpoint", "warning"),
)

# These two master EVENT_STATUS2 bits are described by SceGpuCoredump as the
# processing phases in which the lockup occurred.  They are supporting state,
# not an assertion that either phase originated the bad command or address.
_LOCKUP_PHASE_SIGNALS = (
    (0x00000010, "vertex", "TA/vertex processing"),
    (0x00000008, "fragment", "3D/fragment processing"),
)

_STATUS2_MARKERS = (
    (0x00000200, "mte-context-drained", "MTE context drained to memory"),
    (0x00000100, "zls-context-store-finished", "ZLS context-switch z-store completed"),
    (0x00000080, "dcu-invalidation-complete", "DCU invalidation completed"),
    (0x00000040, "mte-state-flushed", "MTE state flushed to memory"),
    (0x00000020, "region-headers-initialized", "TE initialized all region headers"),
    (0x00000002, "dpm-3d-free-load-complete", "DPM 3D free-list load completed"),
    (0x00000001, "dpm-ta-free-load-complete", "DPM TA free-list load completed"),
)

_DIRECTORY_OFFSETS = (0x0C84, 0x0C38, 0x0C3C, 0x0C40, 0x0C44, 0x0C48, 0x0C4C, 0x0C50)

# Retained execution/resource pointers with unambiguous address masks in the
# SGX543 register definitions.  These are correlated numerically with
# MEM_BLK_INFO; the match does not substitute for an SGX MMU page walk.
_RESOURCE_POINTER_REGISTERS = (
    (True, 0x4238, 0xFFFFFFFC, "vdm-control-stream", "VDM control stream", "vertex"),
    (False, 0x0238, 0xFFFFFFFC, "vdm-control-stream", "VDM control stream", "vertex"),
    (
        True,
        0x4334,
        0xFFFFFFFC,
        "vdm-context-store-stream",
        "VDM context-store stream",
        "vertex",
    ),
    (
        False,
        0x0334,
        0xFFFFFFFC,
        "vdm-context-store-stream",
        "VDM context-store stream",
        "vertex",
    ),
    (False, 0x0A5C, 0xFFFFFFF0, "pixel-pds-exec", "Pixel PDS executable", "fragment"),
    (False, 0x0A68, 0xFFFFFFF0, "other-pds-exec", "Other-event PDS executable", "shared"),
)

_LOG_CLUE = re.compile(
    r"\b(?:abort|assert|deadlock|error|fail(?:ed|ure)?|fault|hang|invalid|mismatch|"
    r"out[- ]of[- ]memory|oom|panic|stall|timeout|watchdog)\b",
    re.IGNORECASE,
)

_STRUCTURED_NOTE_NAMES = {
    "APP_INFO",
    "APP_LIST_INFO",
    "BUILD_VER_INFO",
    "BUDGET_INFO",
    "CALLBACK_INFO",
    "COREFILE_INFO",
    "CONDVAR_INFO",
    "EVENTFLAG_INFO",
    "EVENT_LOG_INFO",
    "EXTNL_PROC_INFO",
    "DEVICE_INFO",
    "FILE_INFO",
    "GPU_ACT_INFO",
    "GPU_INFO",
    "HW_INFO",
    "LIBRARY_INFO",
    "LWCONDVAR_INFO",
    "LWMUTEX_INFO",
    "MEM_BLK_INFO",
    "MESG_PIPE_INFO",
    "META_DATA_INFO",
    "MODULE_INFO",
    "MUTEX_INFO",
    "PROCESS_INFO",
    "SEMAPHORE_INFO",
    "STACK_INFO",
    "SUMMARY_INFO",
    "SYSTEM_INFO",
    "SYSTEM_INFO2",
    "SYS_DEVICE_INFO",
    "THREAD_INFO",
    "THREAD_REG_INFO",
    "TIMER_INFO",
    "TTY_INFO",
    "TTY_INFO2",
}


def _owner_for_context(trailer: GpuTrailer, context_id: int) -> dict:
    # Compatibility aggregate view only. Faults resolve a specific scope.
    return trailer.context_attribution(context_id)


def _context_assignments(trailer: GpuTrailer) -> List[dict]:
    assignments = []
    for context_id in range(8):
        attribution = _owner_for_context(trailer, context_id)
        if attribution["owner"] is not None:
            assignments.append({"context_id": context_id, **attribution})
    return assignments


def _directory_base(info: GpuInfo, scope: str, context_id: int) -> Optional[int]:
    if context_id < 0 or context_id >= len(_DIRECTORY_OFFSETS):
        return None
    offset = _DIRECTORY_OFFSETS[context_id]
    if scope == "master":
        value = info.master_value(offset + 0x4000)
    elif scope.startswith("core") and scope[4:].isdigit():
        core_index = int(scope[4:])
        values = info.core_values(offset)
        value = values[core_index] if values is not None and core_index < len(values) else None
    else:
        value = None
    return value & 0xFFFFF000 if value is not None else None


def _register_name(
    database: RegisterDatabase, offset: int, master: bool
) -> Optional[str]:
    definition = database.definition(offset, master=master)
    return definition.display_name if definition is not None else None


def _address_register_references(
    info: GpuInfo, database: RegisterDatabase, address: int
) -> List[dict]:
    if address == 0:
        return []

    def matching_fields(offset: int, master: bool, value: int) -> List[str]:
        definition = database.definition(offset, master=master)
        if definition is None:
            return []
        return [
            field.name
            for field in definition.fields
            if field.is_address and field.address(value) & 0xFFFFF000 == address
        ]

    references = []
    for register in info.master_registers:
        if register.offset == 0x4C08:
            continue
        fields = matching_fields(register.offset, True, register.value)
        if fields:
            references.append(
                {
                    "scope": "master",
                    "cores": [],
                    "offset": register.offset,
                    "value": register.value,
                    "name": _register_name(database, register.offset, master=True),
                    "address_fields": fields,
                }
            )
    for register in info.per_core_registers:
        if register.offset == 0x0C08:
            continue
        matches = [
            (index, matching_fields(register.offset, False, value))
            for index, value in enumerate(register.values)
        ]
        cores = [index for index, fields in matches if fields]
        if cores:
            references.append(
                {
                    "scope": "per-core",
                    "cores": cores,
                    "offset": register.offset,
                    "values": [register.values[index] for index in cores],
                    "name": _register_name(database, register.offset, master=False),
                    "address_fields": list(
                        dict.fromkeys(
                            field
                            for _, fields in matches
                            for field in fields
                        )
                    ),
                }
            )
    return references


def _address_capture_evidence(core: CoreDump, address: Optional[int]) -> dict:
    if address is None or address == 0:
        return {
            "address": address,
            "memory_blocks": [],
            "load_segments": [],
            "captured_bytes_at_address": 0,
        }
    blocks = []
    for block in core.memory_blocks():
        if block.base <= address < block.end:
            blocks.append(
                {
                    "index": block.index,
                    "uid": block.uid,
                    "name": block.name,
                    "base": block.base,
                    "end": block.end,
                    "size": block.size,
                    "offset": address - block.base,
                    "attributes": block.attributes,
                }
            )
    loads = [
        {
            "index": segment.index,
            "virtual_address": segment.virtual_address,
            "file_size": segment.file_size,
            "offset": address - segment.virtual_address,
        }
        for segment in core.loads
        if segment.contains(address)
    ]
    return {
        "address": address,
        "memory_blocks": blocks,
        "load_segments": loads,
        "captured_bytes_at_address": core.captured_bytes(address, address + 16),
    }


def _enrich_pipeline_addresses(core: CoreDump, pipeline: dict) -> None:
    for item in pipeline["cores"]:
        item["region_header_evidence"] = _address_capture_evidence(
            core, item["region_header_base"]
        )
        zls = item["zls"]
        if zls is None:
            continue
        for surface in zls["surfaces"].values():
            surface["address_evidence"] = _address_capture_evidence(
                core, surface["address"]
            )


def _fault_evidence(
    core: CoreDump, info: GpuInfo, database: RegisterDatabase
) -> List[dict]:
    result = []
    for source in info.bif_diagnosis()["fault_sources"]:
        item = dict(source)
        context = info.context_attribution(source["context_id"], source["scope"])
        directory_base = _directory_base(info, source["scope"], source["context_id"])
        item.update(
            {
                "gpu_virtual_address": source["address"],
                "null_address": source["address"] == 0,
                "context_attribution": context,
                "page_directory_base": directory_base,
                "page_directory_bytes_captured": bool(
                    directory_base
                    and any(segment.contains(directory_base, 4) for segment in core.loads)
                ),
                "register_references": _address_register_references(
                    info, database, source["address"]
                ),
                "address_capture_evidence": _address_capture_evidence(
                    core, source["address"]
                ),
            }
        )
        result.append(item)
    return result


def _fault_groups(faults: Sequence[dict]) -> List[dict]:
    grouped: dict[tuple, dict] = {}
    for index, fault in enumerate(faults):
        owner = fault["context_attribution"]["owner"]
        owner_process_id = owner["process_id"] if owner is not None else None
        key = (
            fault["gpu_virtual_address"],
            fault["requestor"],
            fault["fault_type"],
            fault["page_access_type"],
            fault["context_id"],
            fault["page_directory_base"],
            owner_process_id,
            fault["status_latched"],
        )
        group = grouped.get(key)
        if group is None:
            group = {
                "index": len(grouped),
                "gpu_virtual_address": fault["gpu_virtual_address"],
                "null_address": fault["null_address"],
                "requestor": fault["requestor"],
                "requestor_names": list(fault["requestor_names"]),
                "requestor_details": list(fault["requestor_details"]),
                "fault_type": fault["fault_type"],
                "fault_type_raw": fault["fault_type_raw"],
                "page_access_type": fault["page_access_type"],
                "fault_conditions": list(fault["fault_conditions"]),
                "context_id": fault["context_id"],
                "context_attribution": fault["context_attribution"],
                "page_directory_base": fault["page_directory_base"],
                "status_latched": fault["status_latched"],
                "source_indices": [],
                "scopes": [],
                "sidebands": [],
                "fault_register_values": [],
            }
            grouped[key] = group
        group["source_indices"].append(index)
        for detail in fault["requestor_details"]:
            if detail not in group["requestor_details"]:
                group["requestor_details"].append(detail)
        if fault["scope"] not in group["scopes"]:
            group["scopes"].append(fault["scope"])
        if fault["sideband"] not in group["sidebands"]:
            group["sidebands"].append(fault["sideband"])
        if fault["fault"] not in group["fault_register_values"]:
            group["fault_register_values"].append(fault["fault"])
    result = list(grouped.values())
    for group in result:
        group["source_count"] = len(group["source_indices"])
        group["multiple_scope_observation"] = len(group["scopes"]) > 1
        group["sidebands"].sort()
    return result


def _append_register_signals(
    result: List[dict], value: Optional[int], scope: str, definitions: Sequence[Tuple[int, str, str, str]]
) -> None:
    if value is None:
        return
    for mask, code, description, severity in definitions:
        if value & mask:
            result.append(
                {
                    "code": code,
                    "description": description,
                    "severity": severity,
                    "scope": scope,
                    "register_value": value,
                    "mask": mask,
                }
            )


def _event_signals(info: GpuInfo) -> List[dict]:
    result: List[dict] = []
    master_event = info.master_value(0x412C)
    _append_register_signals(result, master_event, "master", _EVENT_SIGNALS)
    _append_register_signals(
        result, master_event, "master", _MASTER_ONLY_EVENT_SIGNALS
    )
    _append_register_signals(
        result, info.master_value(0x4118), "master", _STATUS2_SIGNALS
    )
    _append_register_signals(
        result, info.master_value(0x4758), "master", _DPM_DEBUG_SIGNALS
    )
    core_event = info.core_values(0x012C)
    core_event2 = info.core_values(0x0118)
    for index in range(info.core_count):
        if core_event is not None:
            _append_register_signals(
                result, core_event[index], f"core{index}", _EVENT_SIGNALS
            )
        if core_event2 is not None:
            _append_register_signals(
                result, core_event2[index], f"core{index}", _STATUS2_SIGNALS
            )
    return result


def _lockup_phases(info: GpuInfo) -> List[dict]:
    value = info.master_value(0x4118)
    if value is None:
        return []
    return [
        {
            "code": code,
            "description": description,
            "scope": "master",
            "register_offset": 0x4118,
            "register_value": value,
            "mask": mask,
        }
        for mask, code, description in _LOCKUP_PHASE_SIGNALS
        if value & mask
    ]


def _pipeline_markers(info: GpuInfo) -> List[dict]:
    result: List[dict] = []
    master = info.master_value(0x4118)
    if master is not None:
        for mask, code, description in _STATUS2_MARKERS:
            if master & mask:
                result.append(
                    {
                        "code": code,
                        "description": description,
                        "scope": "master",
                        "register_offset": 0x4118,
                        "register_value": master,
                        "mask": mask,
                    }
                )
    per_core = info.core_values(0x0118)
    if per_core is not None:
        for core_index, value in enumerate(per_core):
            for mask, code, description in _STATUS2_MARKERS:
                if value & mask:
                    result.append(
                        {
                            "code": code,
                            "description": description,
                            "scope": f"core{core_index}",
                            "register_offset": 0x0118,
                            "register_value": value,
                            "mask": mask,
                        }
                    )
    partial_render = info.master_value(0x4658)
    if partial_render is not None and partial_render & 1:
        result.append(
            {
                "code": "partial-render-in-progress",
                "description": "partial render in progress",
                "scope": "master",
                "register_offset": 0x4658,
                "register_value": partial_render,
                "mask": 1,
            }
        )
    core_partial = info.core_values(0x0658)
    if core_partial is not None:
        for core_index, value in enumerate(core_partial):
            if value & 1:
                result.append(
                    {
                        "code": "partial-render-in-progress",
                        "description": "partial render in progress",
                        "scope": f"core{core_index}",
                        "register_offset": 0x0658,
                        "register_value": value,
                        "mask": 1,
                    }
                )
    return result


def _memory_evidence(core: CoreDump) -> dict:
    blocks = core.memory_blocks()
    captured_blocks = [
        block for block in blocks if core.captured_bytes(block.base, block.end)
    ]
    aliased_indices = {
        index
        for group in core.load_alias_groups()
        for index in group
        if index != min(group)
    }
    unique_loads = [load for load in core.loads if load.index not in aliased_indices]
    return {
        "memory_block_count": len(blocks),
        "memory_blocks_with_captured_bytes": len(captured_blocks),
        "memory_block_captured_bytes": sum(
            core.captured_bytes(block.base, block.end) for block in blocks
        ),
        "load_segment_count": len(core.loads),
        "unique_load_segment_count": len(unique_loads),
        "unique_load_bytes": sum(load.file_size for load in unique_loads),
        "alias_group_count": len(core.load_alias_groups()),
    }


def _note_coverage(core: CoreDump) -> dict:
    names = sorted({note.name for note in core.notes})
    structured = [name for name in names if name in _STRUCTURED_NOTE_NAMES]
    inventory_only = [name for name in names if name not in _STRUCTURED_NOTE_NAMES]
    return {
        "note_name_count": len(names),
        "structured_count": len(structured),
        "structured_names": structured,
        "inventory_only_count": len(inventory_only),
        "inventory_only_names": inventory_only,
    }
_GPU_ALLOCATION_TAGS = (
    ("gxm", "gxm"),
    ("gpu", "gpu"),
    ("usse", "usse"),
    ("vdm", "vdm-command"),
    ("pds", "pds"),
    ("shader", "shader"),
    ("render target", "render-target"),
    ("texture", "texture"),
    ("fragment", "fragment"),
    ("vertex", "vertex"),
    ("parameter buffer", "parameter-buffer"),
)


def _allocation_evidence(core: CoreDump) -> dict:
    allocations = []
    category_counts: dict[str, int] = {}
    category_bytes: dict[str, int] = {}
    for block in core.memory_blocks():
        lowered = block.name.casefold()
        categories = sorted(
            {category for token, category in _GPU_ALLOCATION_TAGS if token in lowered}
        )
        if not categories:
            continue
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
            category_bytes[category] = category_bytes.get(category, 0) + block.size
        allocations.append(
            {
                "index": block.index,
                "uid": block.uid,
                "name": block.name,
                "base": block.base,
                "end": block.end,
                "size": block.size,
                "committed_size": block.committed_size,
                "attributes": block.attributes,
                "categories": categories,
                "captured_bytes": core.captured_bytes(block.base, block.end),
            }
        )
    allocations.sort(key=lambda item: (-item["size"], item["index"]))
    return {
        "selection": "case-insensitive component tokens in MEM_BLK_INFO names",
        "allocation_count": len(allocations),
        "total_size": sum(item["size"] for item in allocations),
        "total_committed_size": sum(item["committed_size"] for item in allocations),
        "category_counts": category_counts,
        "category_bytes": category_bytes,
        "allocations": allocations,
    }


def _resource_pointer_evidence(core: CoreDump, info: GpuInfo) -> dict:
    grouped: dict[tuple[str, int], dict] = {}
    observation_count = 0
    for master, offset, mask, code, name, stage in _RESOURCE_POINTER_REGISTERS:
        if master:
            values = [
                ("master", None, register.value)
                for register in info.master_registers
                if register.offset == offset
            ]
        else:
            captured = info.core_values(offset)
            values = (
                [("per-core", index, value) for index, value in enumerate(captured)]
                if captured is not None
                else []
            )
        for scope, core_index, raw_value in values:
            address = raw_value & mask
            if address == 0:
                continue
            observation_count += 1
            key = (code, address)
            pointer = grouped.get(key)
            if pointer is None:
                block = core.memory_block_for(address)
                allocation = None
                if block is not None:
                    allocation = {
                        "index": block.index,
                        "uid": block.uid,
                        "name": block.name,
                        "base": block.base,
                        "end": block.end,
                        "size": block.size,
                        "offset": address - block.base,
                        "attributes": block.attributes,
                        "captured_bytes": core.captured_bytes(block.base, block.end),
                    }
                pointer = {
                    "index": len(grouped),
                    "code": code,
                    "name": name,
                    "stage": stage,
                    "address": address,
                    "allocation": allocation,
                    "captured_bytes_at_address": core.captured_bytes(
                        address, address + 16
                    ),
                    "observations": [],
                    "master_observed": False,
                    "core_indices": [],
                }
                grouped[key] = pointer
            pointer["observations"].append(
                {
                    "scope": scope,
                    "core": core_index,
                    "register_offset": offset,
                    "raw_value": raw_value,
                    "address_mask": mask,
                }
            )
            if master:
                pointer["master_observed"] = True
            elif core_index not in pointer["core_indices"]:
                pointer["core_indices"].append(core_index)
    pointers = list(grouped.values())
    for pointer in pointers:
        pointer["core_indices"].sort()
    return {
        "selection": "retained VDM stream and PDS executable address registers",
        "correlation": "direct numeric overlap with MEM_BLK_INFO virtual ranges",
        "pointer_count": len(pointers),
        "observation_count": observation_count,
        "allocation_match_count": sum(
            pointer["allocation"] is not None for pointer in pointers
        ),
        "pointers": pointers,
    }


def _kernel_evidence(core: CoreDump, execution: dict, support: dict) -> dict:
    registry = KernelObjectRegistry.parse(core)
    threads_by_uid = {thread["uid"]: thread for thread in execution["threads"]}
    external = support["external_processes"]
    processes_by_id = {
        item["process_id"]: item
        for item in (external["records"] if external is not None else [])
    }
    for thread in execution["threads"]:
        wait_id = thread["wait_id"]
        item = registry.resolve(wait_id) if wait_id not in (0, 0xFFFFFFFF) else None
        thread["wait_object"] = item.summary() if item is not None else None
        thread["waiter_record_present"] = bool(
            item is not None
            and any(waiter.thread_uid == thread["uid"] for waiter in item.waiters)
        )
    summary = registry.summary()
    for item in summary["active_objects"]:
        owner = threads_by_uid.get(item["owner_thread_uid"])
        item["owner_thread"] = (
            {"uid": owner["uid"], "name": owner["name"], "index": owner["index"]}
            if owner is not None
            else None
        )
        item["waiter_threads"] = [
            {
                "uid": waiter["thread_uid"],
                "process_id": waiter["process_id"],
                "process_name": processes_by_id.get(
                    waiter["process_id"], {}
                ).get("name"),
                "process_path": processes_by_id.get(
                    waiter["process_id"], {}
                ).get("path"),
                "name": threads_by_uid.get(waiter["thread_uid"], {}).get("name"),
                "index": threads_by_uid.get(waiter["thread_uid"], {}).get("index"),
            }
            for waiter in item["waiters"]
        ]
    summary["resolved_thread_wait_count"] = sum(
        thread["wait_object"] is not None for thread in execution["threads"]
    )
    summary["known_external_process_count"] = len(processes_by_id)
    return summary


def _trim_log_line(line: str, limit: int = 240) -> str:
    line = line.strip()
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _log_evidence(core: CoreDump) -> dict:
    note = core.note("TTY_INFO2") or core.note("TTY_INFO")
    if note is None:
        return {
            "available": False,
            "note": None,
            "line_count": 0,
            "noteworthy_lines": [],
            "recent_lines": [],
            "timeline": None,
        }
    try:
        tty = TtyInfo.parse(note.description)
        timeline = tty.timeline()
    except ParseError as exc:
        return {
            "available": False,
            "note": note.name,
            "error": str(exc),
            "line_count": 0,
            "noteworthy_lines": [],
            "recent_lines": [],
            "timeline": None,
        }
    lines = [line["text"] for line in timeline["lines"]]
    noteworthy = []
    for line in lines:
        if _LOG_CLUE.search(line):
            trimmed = _trim_log_line(line)
            if trimmed not in noteworthy:
                noteworthy.append(trimmed)
    return {
        "available": True,
        "note": note.name,
        "line_count": len(lines),
        "noteworthy_lines": noteworthy[-5:],
        "recent_lines": [_trim_log_line(line) for line in lines[-8:]],
        "timeline": timeline,
    }


def _verdict(
    faults: List[dict],
    fault_groups: List[dict],
    signals: List[dict],
    lockup_phases: List[dict],
    firmware_signatures: List[dict],
) -> Tuple[str, str, str, str]:
    if firmware_signatures:
        scopes = ", ".join(item["scope"] for item in firmware_signatures)
        return (
            "gpu-clipper-halt-signature",
            "medium",
            f"GPU halt matches the firmware's possible clipper-hardware-bug signature: {scopes}",
            "Indexed selector 0x25 bit 4 is set with no BIF fault-type latch. The native "
            "driver selects its possible known clipper-hardware-bug recovery path for this "
            "state. This identifies a firmware signature, not a confirmed hardware defect; "
            "submitted command and shader data remain unverified.",
        )
    if faults:
        active_faults = [fault for fault in faults if fault["status_latched"]]
        scopes = ", ".join(item["scope"] for item in faults)
        noun = "fault" if len(faults) == 1 else "faults"
        if not active_faults:
            return (
                "gpu-bif-fault-register-residue",
                "medium",
                f"BIF fault-register state retained on {scopes}; fault-type status clear",
                "BIF requestor/address fields are retained, but the BIF_INT_STAT fault "
                "condition is zero. Treat the address as residual state, not a confirmed active fault.",
            )
        active_groups = [group for group in fault_groups if group["status_latched"]]
        null_groups = [group for group in active_groups if group["null_address"]]
        if null_groups:
            null_scopes = ", ".join(
                scope for group in null_groups for scope in group["scopes"]
            )
            headline = (
                f"GPU BIF/MMU null-page access fault: {null_scopes}"
                if len(null_groups) == len(active_groups)
                else "GPU BIF/MMU access faults, including null-page access on "
                f"{null_scopes}"
            )
            return (
                "gpu-bif-access-fault",
                "high",
                headline,
                "A valid BIF fault request targets GPU virtual page zero. The retained "
                "requestor and MMU context identify the access path; they do not identify "
                "which command or application object supplied the null address. Other "
                "retained fault groups remain listed separately.",
            )
        if len(active_groups) == 1 and active_groups[0]["multiple_scope_observation"]:
            group = active_groups[0]
            grouped_scopes = ", ".join(group["scopes"])
            return (
                "gpu-bif-access-fault",
                "high",
                f"GPU BIF/MMU access fault observed on {grouped_scopes}",
                "The same GPU virtual page, requestor, MMU context, and page-directory "
                "tuple is retained by multiple cores. This is one grouped translation "
                "fault observation; allocation resolution still requires the SGX page "
                "tables or a device-virtual-address allocation map.",
            )
        return (
            "gpu-bif-access-fault",
            "high",
            f"GPU BIF/MMU access {noun}: {scopes}",
            "BIF_INT_STAT contains a fault request/type latch and BIF_FAULT contains the "
            "GPU virtual page and MMU context ID. Allocation resolution requires the SGX "
            "page tables or a device-virtual-address allocation map.",
        )
    codes = {signal["code"] for signal in signals}
    if "dpm-out-of-memory-deadlock" in codes:
        return (
            "gpu-dpm-memory-deadlock",
            "high",
            "GPU DPM out-of-memory deadlock latched",
            "No BIF address fault is retained. DPM_DEBUG_STATUS.OUTOFMEM_DEADLOCK is set.",
        )
    memory_signals = [
        signal
        for signal in signals
        if "out-of-memory" in signal["code"] or signal["code"] == "ta-dpm-fault"
    ]
    if memory_signals:
        return (
            "gpu-dpm-memory-pressure",
            "medium",
            "GPU DPM fault/memory-pressure event; no BIF address fault",
            "DPM fault or out-of-memory event bits are set. Event status may survive "
            "recovery handling; correlate with the TTY tail and a comparison dump.",
        )
    explicit_errors = [signal for signal in signals if signal["severity"] == "error"]
    if explicit_errors:
        return (
            "gpu-hardware-event",
            "medium",
            "GPU error event; no BIF address fault",
            "Error event present; faulting GPU virtual address unavailable. Resource "
            "attribution unavailable.",
        )
    phase_codes = {phase["code"] for phase in lockup_phases}
    if phase_codes == {"vertex", "fragment"}:
        return (
            "gpu-ta-3d-lockup",
            "medium",
            "GPU lockup retained TA/vertex and 3D/fragment phase latches",
            "MASTER_EVENT_STATUS2 has both TRIG_TA and TRIG_3D set. This identifies "
            "the processing phases present at lockup; it does not identify the "
            "malformed command, shader, or resource responsible.",
        )
    if "vertex" in phase_codes:
        return (
            "gpu-ta-lockup",
            "medium",
            "GPU lockup retained the TA/vertex phase latch",
            "MASTER_EVENT_STATUS2.TRIG_TA is set. The snapshot places the lockup in "
            "vertex processing but does not retain a BIF fault address.",
        )
    if "fragment" in phase_codes:
        return (
            "gpu-3d-lockup",
            "medium",
            "GPU lockup retained the 3D/fragment phase latch",
            "MASTER_EVENT_STATUS2.TRIG_3D is set. The snapshot places the lockup in "
            "fragment processing but does not retain a BIF fault address.",
        )
    return (
        "undetermined-gpu-hang",
        "low",
        "GPU hang with no decoded BIF/DPM error latch",
        "No BIF address fault or decoded DPM error latch is present. Possible classes: "
        "invalid command stream, USSE stall, firmware timeout, or cleared transient fault.",
    )


def _root_cause_confidence(classification: str) -> str:
    if classification == "gpu-dpm-memory-deadlock":
        return "medium"
    return "low"


def _failure_identity(
    classification: str,
    faults: List[dict],
    signals: List[dict],
    lockup_phases: List[dict],
) -> Tuple[str, str]:
    parts = [classification.upper().replace("-", "_")]
    if faults:
        for fault in faults:
            parts.extend(
                (
                    fault["scope"].upper(),
                    f"REQ_{fault['requestor']:04X}",
                    f"TYPE_{fault['fault_type_raw']}",
                    f"CID_{fault['context_id']}",
                )
            )
    elif signals:
        parts.extend(
            sorted({signal["code"].upper().replace("-", "_") for signal in signals})
        )
    elif lockup_phases:
        parts.extend(
            sorted(
                f"PHASE_{phase['code'].upper().replace('-', '_')}"
                for phase in lockup_phases
            )
        )
    else:
        parts.append("NO_ERROR_LATCH")
    bucket = "_".join(parts)
    return bucket, hashlib.sha256(bucket.encode("ascii")).hexdigest()


def build_diagnosis(
    core: CoreDump,
    info: GpuInfo,
    database: RegisterDatabase,
    activity: Optional[GpuActivityInfo] = None,
    summary_info: Optional[SummaryInfo] = None,
) -> dict:
    faults = _fault_evidence(core, info, database)
    fault_groups = _fault_groups(faults)
    signals = _event_signals(info)
    lockup_phases = _lockup_phases(info)
    pipeline_markers = _pipeline_markers(info)
    pipeline = firmware_pipeline_diagnosis(info)
    _enrich_pipeline_addresses(core, pipeline)
    firmware_signatures = pipeline["clipper_halt_check"]["matches"]
    for signature in firmware_signatures:
        cid = signature["context_id"]
        signature["context_attribution"] = info.context_attribution(
            cid if cid is not None else -1, signature["scope"]
        )
    classification, confidence, headline, explanation = _verdict(
        faults, fault_groups, signals, lockup_phases, firmware_signatures
    )
    root_cause_confidence = _root_cause_confidence(classification)
    failure_bucket, failure_hash = _failure_identity(
        classification, faults, signals, lockup_phases
    )
    memory = _memory_evidence(core)
    execution = ExecutionContext.parse(core).summary(core)
    support = supporting_context(core)
    metadata = support["metadata"]
    modules = execution["modules"]
    if metadata is not None and modules is not None:
        modules_by_uid = {item["uid"]: item for item in modules["modules"]}
        for entry in metadata["entries"]:
            module = modules_by_uid.get(entry["identifier"])
            entry["module_index"] = module["index"] if module is not None else None
            entry["module_name"] = module["name"] if module is not None else None
        metadata["resolved_module_count"] = sum(
            entry["module_name"] is not None for entry in metadata["entries"]
        )
        metadata["module_count_matches"] = (
            len(metadata["entries"]) == modules["module_count"]
            and metadata["resolved_module_count"] == modules["module_count"]
        )
    kernel = _kernel_evidence(core, execution, support)
    allocations = _allocation_evidence(core)
    resource_pointers = _resource_pointer_evidence(core, info)
    summary = summary_info.summary(core.notes) if summary_info is not None else None
    activity_state = (
        "unavailable"
        if activity is None
        else "retained"
        if activity.activity_buffer is not None
        else "not-retained"
    )
    limitations = []
    if faults and any(not item["page_directory_bytes_captured"] for item in faults):
        limitations.append(
            "SGX page-directory address is not directly mapped by PT_LOAD. GPU-VA page "
            "walk unavailable without a physical-to-kernel address mapping."
        )
    if memory["memory_block_count"] and not memory["memory_blocks_with_captured_bytes"]:
        limitations.append(
            "MEM_BLK_INFO lists application/GXM allocations, but no listed range overlaps "
            "captured PT_LOAD data. Command and shader bytes unavailable."
        )
    if activity_state == "not-retained":
        limitations.append("GPU_ACT_INFO contains no activity-ring payload.")
    quoted = shlex.quote(str(core.path))
    trailer = info.decoded_trailer()
    return {
        "classification": classification,
        "analysis_code": classification.upper().replace("-", "_"),
        "confidence": confidence,
        "classification_confidence": confidence,
        "root_cause_confidence": root_cause_confidence,
        "headline": headline,
        "explanation": explanation,
        "fault_sources": faults,
        "fault_groups": fault_groups,
        "event_signals": signals,
        "lockup_phases": lockup_phases,
        "pipeline_markers": pipeline_markers,
        "firmware_pipeline": pipeline,
        "firmware_signatures": firmware_signatures,
        "snapshot": info.snapshot_status(),
        "context_owners": [
            owner.summary()
            for owner in trailer.context_owners
            if owner.populated
        ],
        "context_assignments": _context_assignments(trailer) if info.capture_flags & 8 else [],
        "context_scope_assignments": [
            {"context_id": cid, **attribution}
            for scope in (*GPU_CONTEXT_MAP_SCOPES[:min(info.core_count, 4)], "master")
            for cid in range(8)
            if (attribution := info.context_attribution(cid, scope))["owner"] is not None
        ],
        "memory_evidence": memory,
        "note_coverage": _note_coverage(core),
        "allocation_evidence": allocations,
        "resource_pointers": resource_pointers,
        "execution_context": execution,
        "kernel_objects": kernel,
        "supporting_context": support,
        "summary_integrity": summary,
        "activity_state": activity_state,
        "log": _log_evidence(core),
        "limitations": limitations,
        "failure_bucket_id": failure_bucket,
        "failure_hash": failure_hash,
        "recommended_commands": [
            f"psp2-gpucrash-parse gpu faults {quoted}",
            f"psp2-gpucrash-parse gpu pipeline {quoted}",
            f"psp2-gpucrash-parse gpu resources {quoted}",
            f"psp2-gpucrash-parse threads {quoted}",
            f"psp2-gpucrash-parse waits {quoted}",
            f"psp2-gpucrash-parse map {quoted} --captured-only --unique",
            *(
                [
                    f"psp2-gpucrash-parse address {quoted} "
                    f"0x{fault_groups[0]['gpu_virtual_address']:08x}",
                    f"psp2-gpucrash-parse xref {quoted} "
                    f"0x{fault_groups[0]['gpu_virtual_address']:08x}",
                ]
                if fault_groups
                else []
            ),
            f"psp2-gpucrash-parse tty {quoted}",
            f"psp2-gpucrash-parse registers {quoted} --matching 'BIF|EVENT|VDM|DPM|ISP'",
            f"psp2-gpucrash-parse segments {quoted} --unique",
            f"psp2-gpucrash-parse validate {quoted}",
            f"psp2-gpucrash-parse compare {quoted} KNOWN_GOOD.psp2dmp",
        ],
    }
