from __future__ import annotations

import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .activity import GpuActivityInfo
from .core import CoreDump, ParseError
from .diagnosis import build_diagnosis
from .execution import ExecutionContext, ModuleInfo
from .gpu import GpuInfo
from .registers import RegisterDatabase


def load_debug_state(
    core: CoreDump, database: RegisterDatabase
) -> Tuple[GpuInfo, dict]:
    gpu_note = core.note("GPU_INFO")
    if gpu_note is None:
        raise ParseError("GPU_INFO note is unavailable")
    info = GpuInfo.parse(gpu_note.description)
    activity_note = core.note("GPU_ACT_INFO")
    activity = (
        GpuActivityInfo.parse(activity_note.description)
        if activity_note is not None
        else None
    )
    return info, build_diagnosis(core, info, database, activity)


def _modules(core: CoreDump) -> Optional[ModuleInfo]:
    return ExecutionContext.parse(core).modules


def _module_location(modules: Optional[ModuleInfo], address: int) -> Optional[dict]:
    location = modules.locate(address) if modules is not None else None
    return location.summary() if location is not None else None


def _memory_window(core: CoreDump, address: int, requested_size: int) -> dict:
    if requested_size < 0:
        raise ParseError("preview size cannot be negative")
    if requested_size == 0:
        return {
            "address": address,
            "requested_size": 0,
            "captured_size": 0,
            "bytes_hex": "",
            "ascii": "",
            "u32": [],
        }
    ranges = core.captured_ranges(address, address + requested_size)
    if not ranges or ranges[0][0] != address:
        return {
            "address": address,
            "requested_size": requested_size,
            "captured_size": 0,
            "bytes_hex": "",
            "ascii": "",
            "u32": [],
        }
    size = ranges[0][1] - address
    data = core.read_memory(address, size)
    words = [
        {
            "address": address + offset,
            "value": struct.unpack_from("<I", data, offset)[0],
        }
        for offset in range(0, len(data) - 3, 4)
    ]
    return {
        "address": address,
        "requested_size": requested_size,
        "captured_size": len(data),
        "bytes_hex": data.hex(),
        "ascii": "".join(chr(value) if 32 <= value < 127 else "." for value in data),
        "u32": words,
    }


def _gpu_register_references(
    info: GpuInfo, database: RegisterDatabase, address: int
) -> List[dict]:
    result = []

    def matches(offset: int, master: bool, value: int) -> List[dict]:
        definition = database.definition(offset, master=master)
        if definition is None:
            return []
        fields = []
        for field in definition.fields:
            if not field.is_address:
                continue
            field_address = field.address(value)
            if field_address == 0:
                continue
            if field_address == address:
                relation = "exact"
            elif field_address & 0xFFFFF000 == address & 0xFFFFF000:
                relation = "same-4k-page"
            else:
                continue
            fields.append(
                {
                    "name": field.name,
                    "address": field_address,
                    "relation": relation,
                    "mask": field.mask,
                    "shift": field.shift,
                }
            )
        return fields

    for register in info.master_registers:
        fields = matches(register.offset, True, register.value)
        if fields:
            definition = database.definition(register.offset, master=True)
            result.append(
                {
                    "scope": "master",
                    "core": None,
                    "offset": register.offset,
                    "name": definition.display_name if definition is not None else None,
                    "raw_value": register.value,
                    "fields": fields,
                }
            )
    for register in info.per_core_registers:
        definition = database.definition(register.offset, master=False)
        for core_index, value in enumerate(register.values):
            fields = matches(register.offset, False, value)
            if fields:
                result.append(
                    {
                        "scope": "per-core",
                        "core": core_index,
                        "offset": register.offset,
                        "name": (
                            definition.display_name if definition is not None else None
                        ),
                        "raw_value": value,
                        "fields": fields,
                    }
                )
    return result


def address_report(
    core: CoreDump,
    database: RegisterDatabase,
    address: int,
    preview_size: int = 64,
) -> dict:
    if preview_size > 0x100000:
        raise ParseError("address preview is limited to 0x100000 bytes")
    info, diagnosis = load_debug_state(core, database)
    execution = diagnosis["execution_context"]
    modules = _modules(core)
    memory_blocks = [
        {
            **block.summary(),
            "end": block.end,
            "offset": address - block.base,
            "captured_bytes": core.captured_bytes(block.base, block.end),
        }
        for block in core.memory_blocks()
        if block.base <= address < block.end
    ]
    load_segments = []
    for segment in core.loads:
        memory_end = segment.virtual_address + segment.memory_size
        if not segment.virtual_address <= address < memory_end:
            continue
        captured = address < segment.virtual_address + segment.file_size
        load_segments.append(
            {
                **segment.summary(),
                "memory_end": memory_end,
                "offset": address - segment.virtual_address,
                "captured_at_address": captured,
            }
        )

    thread_matches = []
    for thread in execution["threads"]:
        register_matches = []
        registers = thread["registers"]
        if registers is not None:
            register_matches = [
                f"r{index}"
                for index, value in enumerate(registers["gpr"])
                if value == address
            ]
        in_stack = thread["stack_base"] <= address < thread["stack_end"]
        if in_stack or register_matches:
            thread_matches.append(
                {
                    "index": thread["index"],
                    "uid": thread["uid"],
                    "name": thread["name"],
                    "status_name": thread["status_name"],
                    "register_context_live": thread["register_context_live"],
                    "register_context_qualification": thread[
                        "register_context_qualification"
                    ],
                    "in_stack_range": in_stack,
                    "stack_offset": address - thread["stack_base"] if in_stack else None,
                    "registers": register_matches,
                }
            )

    page = address & 0xFFFFF000
    fault_groups = [
        group
        for group in diagnosis["fault_groups"]
        if group["gpu_virtual_address"] == page
    ]
    resource_pointers = []
    for pointer in diagnosis["resource_pointers"]["pointers"]:
        allocation = pointer["allocation"]
        exact = pointer["address"] == address
        in_allocation = bool(
            allocation is not None
            and allocation["base"] <= address < allocation["end"]
        )
        if exact or in_allocation:
            resource_pointers.append(
                {
                    **pointer,
                    "relation": "exact" if exact else "same-allocation",
                }
            )

    preview = _memory_window(core, address, preview_size)
    module = _module_location(modules, address)
    gpu_registers = _gpu_register_references(info, database, address)
    captured_at_address = any(
        segment["captured_at_address"] for segment in load_segments
    )
    associations = {
        "captured_memory": captured_at_address,
        "memory_block": bool(memory_blocks),
        "module": module is not None,
        "thread": bool(thread_matches),
        "gpu_fault": bool(fault_groups),
        "gpu_resource": bool(resource_pointers),
        "gpu_register": bool(gpu_registers),
    }
    gpu_only = (
        any(associations[name] for name in ("gpu_fault", "gpu_resource", "gpu_register"))
        and not associations["captured_memory"]
    )
    if associations["captured_memory"]:
        status = "captured"
        conclusion = "Bytes at this virtual address are present in PT_LOAD data."
    elif any(associations.values()):
        status = "metadata-only"
        conclusion = (
            "The address is retained as metadata, but bytes at the address are not "
            "present in the dump."
        )
    else:
        status = "unreferenced"
        conclusion = "No retained range, register, fault, resource, module, or thread matches this address."
    limitations = []
    if gpu_only:
        limitations.append(
            "GPU virtual-address bytes and page tables are not retained; the address cannot be dereferenced or translated."
        )
    value_notes = []
    if address == 0xDEADBEEF:
        value_notes.append(
            "0xDEADBEEF is a poison/uninitialized sentinel in retained CPU state; "
            "treat matching register and memory words as invalid candidates unless "
            "independently corroborated."
        )
    return {
        "schema_version": 1,
        "command": "address",
        "path": str(core.path),
        "address": address,
        "page_4k": page,
        "status": status,
        "conclusion": conclusion,
        "value_notes": value_notes,
        "associations": associations,
        "module": module,
        "memory_blocks": memory_blocks,
        "load_segments": load_segments,
        "threads": thread_matches,
        "gpu_fault_groups": fault_groups,
        "gpu_resource_pointers": resource_pointers,
        "gpu_register_references": gpu_registers,
        "memory_preview": preview,
        "limitations": limitations,
    }


def thread_list_report(core: CoreDump, database: RegisterDatabase) -> dict:
    _info, diagnosis = load_debug_state(core, database)
    execution = diagnosis["execution_context"]
    threads = []
    for item in execution["threads"]:
        registers = item["registers"]
        threads.append(
            {
                "index": item["index"],
                "uid": item["uid"],
                "name": item["name"],
                "status": item["status"],
                "status_name": item["status_name"],
                "register_context_live": item["register_context_live"],
                "register_context_qualification": item[
                    "register_context_qualification"
                ],
                "poison_register_count": item["poison_register_count"],
                "current_priority": item["current_priority"],
                "current_cpu_id": item["current_cpu_id"],
                "wait_type": item["wait_type"],
                "wait_id": item["wait_id"],
                "wait_object": item.get("wait_object"),
                "pc": registers["pc"] if registers is not None else None,
                "lr": registers["lr"] if registers is not None else None,
                "sp": registers["sp"] if registers is not None else None,
                "pc_location": item["pc_location"],
                "lr_location": item["lr_location"],
                "stack_captured_at_sp": item["stack_captured_at_sp"],
            }
        )
    return {
        "schema_version": 1,
        "command": "threads",
        "path": str(core.path),
        "process": execution["process"],
        "thread_count": len(threads),
        "threads": threads,
    }


def _select_thread(threads: Sequence[dict], selector: str) -> dict:
    numeric = None
    try:
        numeric = int(selector, 0)
    except ValueError:
        pass
    if numeric is not None:
        uid_matches = [item for item in threads if item["uid"] == numeric]
        if uid_matches:
            return uid_matches[0]
        index_matches = [item for item in threads if item["index"] == numeric]
        if index_matches:
            return index_matches[0]
    exact = [item for item in threads if item["name"].casefold() == selector.casefold()]
    if len(exact) == 1:
        return exact[0]
    partial = [item for item in threads if selector.casefold() in item["name"].casefold()]
    if len(partial) == 1:
        return partial[0]
    if len(exact) > 1 or len(partial) > 1:
        raise ParseError(f"thread selector {selector!r} is ambiguous")
    raise ParseError(f"thread selector {selector!r} did not match a UID, index, or name")


def thread_detail_report(
    core: CoreDump,
    database: RegisterDatabase,
    selector: str,
    stack_bytes: int = 256,
) -> Tuple[dict, dict]:
    if stack_bytes < 0 or stack_bytes > 0x10000:
        raise ParseError("stack preview size must be between 0 and 0x10000 bytes")
    _info, diagnosis = load_debug_state(core, database)
    execution = diagnosis["execution_context"]
    selected = _select_thread(execution["threads"], selector)
    detail = dict(selected)
    registers = selected["registers"]
    available_stack = (
        max(0, selected["stack_end"] - registers["sp"])
        if registers is not None
        else 0
    )
    detail["stack_preview"] = (
        _memory_window(core, registers["sp"], min(stack_bytes, available_stack))
        if registers is not None and registers["sp"] != 0
        else _memory_window(core, 0, 0)
    )
    return {
        "schema_version": 1,
        "command": "thread",
        "path": str(core.path),
        "process": execution["process"],
        "selector": selector,
        "thread": detail,
    }, diagnosis


def _canonical_segments(core: CoreDump, include_aliases: bool) -> List:
    if include_aliases:
        return list(core.loads)
    duplicates = {
        index
        for group in core.load_alias_groups()
        for index in group
        if index != min(group)
    }
    return [segment for segment in core.loads if segment.index not in duplicates]


def search_memory(
    core: CoreDump,
    pattern: bytes,
    description: str,
    alignment: int = 1,
    max_results: int = 64,
    include_aliases: bool = False,
) -> dict:
    if not pattern:
        raise ParseError("search pattern cannot be empty")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ParseError("search alignment must be a positive power of two")
    if max_results < 0:
        raise ParseError("maximum result count cannot be negative")
    modules = _modules(core)
    alias_lookup: Dict[int, List[int]] = {}
    for group in core.load_alias_groups():
        for index in group:
            alias_lookup[index] = group
    results = []
    match_count = 0
    segments = _canonical_segments(core, include_aliases)
    for segment in segments:
        cursor = 0
        while True:
            offset = segment.data.find(pattern, cursor)
            if offset < 0:
                break
            address = segment.virtual_address + offset
            cursor = offset + 1
            if address & (alignment - 1):
                continue
            match_count += 1
            if len(results) >= max_results:
                continue
            block = core.memory_block_for(address)
            results.append(
                {
                    "address": address,
                    "segment_index": segment.index,
                    "segment_offset": offset,
                    "alias_segment_indices": alias_lookup.get(segment.index, []),
                    "memory_block": (
                        {
                            "index": block.index,
                            "uid": block.uid,
                            "name": block.name,
                            "base": block.base,
                            "end": block.end,
                            "offset": address - block.base,
                        }
                        if block is not None
                        else None
                    ),
                    "module": _module_location(modules, address),
                }
            )
    return {
        "schema_version": 1,
        "command": "search",
        "path": str(core.path),
        "pattern": description,
        "pattern_hex": pattern.hex(),
        "pattern_size": len(pattern),
        "alignment": alignment,
        "include_alias_segments": include_aliases,
        "searched_segment_count": len(segments),
        "searched_bytes": sum(segment.file_size for segment in segments),
        "match_count": match_count,
        "returned_count": len(results),
        "truncated": match_count > len(results),
        "results": results,
    }


def _keyed_changes(
    left_items: Iterable[dict],
    right_items: Iterable[dict],
    key_fields: Sequence[str],
    ignored_fields: Sequence[str] = (),
) -> dict:
    def key(item: dict) -> tuple:
        return tuple(item.get(field) for field in key_fields)

    def ordered(values: Iterable[tuple]) -> List[tuple]:
        return sorted(
            values,
            key=lambda value: tuple(
                (type(component).__name__, repr(component))
                for component in value
            ),
        )

    left = {key(item): item for item in left_items}
    right = {key(item): item for item in right_items}

    def comparable(item: dict) -> dict:
        return {key: value for key, value in item.items() if key not in ignored_fields}

    return {
        "added": [right[item] for item in ordered(set(right) - set(left))],
        "removed": [left[item] for item in ordered(set(left) - set(right))],
        "changed": [
            {
                "key": {
                    field: item[index] for index, field in enumerate(key_fields)
                },
                "left": left[item],
                "right": right[item],
            }
            for item in ordered(set(left) & set(right))
            if comparable(left[item]) != comparable(right[item])
        ],
    }


def _pipeline_core_summary(item: dict) -> dict:
    zls = item["zls"]
    if zls is not None:
        zls = {
            "control": zls["control"],
            "flags": zls["flags"],
            "external_depth_buffer": zls["external_depth_buffer"],
            "extent_field": zls["extent_field"],
            "width_pixels": zls["width_pixels"],
            "depth_load_format": zls["depth_load_format"],
            "depth_store_format": zls["depth_store_format"],
            "surfaces": {
                name: {
                    key: value
                    for key, value in surface.items()
                    if key != "address_evidence"
                }
                for name, surface in zls["surfaces"].items()
            },
        }
    return {
        "core": item["core"],
        "active_phases": item["active_phases"],
        "active_units": [unit["code"] for unit in item["active_units"]],
        "usse_data_master_usage": {
            pipe["pipe"]: pipe["usage"] for pipe in item["usse_data_master_usage"]
        },
        "draw_counters": {
            counter["code"]: counter["draw_number"]
            for counter in item["draw_counters"]
        },
        "tile_progress": {
            tile["code"]: [tile["tile_x"], tile["tile_y"]]
            for tile in item["tile_progress"]
        },
        "render_bounds": item["render_bounds"],
        "isp_progress": item["isp_progress"],
        "region_header_base": item["region_header_base"],
        "zls": zls,
    }


def compare_debug_state(left: dict, right: dict) -> dict:
    left_faults = left["fault_groups"]
    right_faults = right["fault_groups"]
    faults = _keyed_changes(
        left_faults,
        right_faults,
        (
            "gpu_virtual_address",
            "requestor",
            "fault_type",
            "page_access_type",
            "context_id",
            "page_directory_base",
        ),
    )

    left_assignments = {
        item["context_id"]: item for item in left["context_assignments"]
    }
    right_assignments = {
        item["context_id"]: item for item in right["context_assignments"]
    }
    context_changes = [
        {
            "context_id": context_id,
            "left": left_assignments.get(context_id),
            "right": right_assignments.get(context_id),
        }
        for context_id in sorted(set(left_assignments) | set(right_assignments))
        if left_assignments.get(context_id) != right_assignments.get(context_id)
    ]

    left_pipeline = left["firmware_pipeline"]
    right_pipeline = right["firmware_pipeline"]
    left_cores = {
        item["core"]: _pipeline_core_summary(item) for item in left_pipeline["cores"]
    }
    right_cores = {
        item["core"]: _pipeline_core_summary(item) for item in right_pipeline["cores"]
    }
    pipeline_core_changes = [
        {
            "core": core,
            "left": left_cores.get(core),
            "right": right_cores.get(core),
        }
        for core in sorted(set(left_cores) | set(right_cores))
        if left_cores.get(core) != right_cores.get(core)
    ]

    resources = _keyed_changes(
        left["resource_pointers"]["pointers"],
        right["resource_pointers"]["pointers"],
        ("code", "address"),
    )

    left_threads = {
        item["uid"]: item for item in left["execution_context"]["threads"]
    }
    right_threads = {
        item["uid"]: item for item in right["execution_context"]["threads"]
    }

    def thread_state(item: Optional[dict]) -> Optional[dict]:
        if item is None:
            return None
        registers = item["registers"]
        return {
            "index": item["index"],
            "uid": item["uid"],
            "name": item["name"],
            "status": item["status"],
            "status_name": item["status_name"],
            "register_context_live": item["register_context_live"],
            "register_context_qualification": item[
                "register_context_qualification"
            ],
            "poison_register_count": item["poison_register_count"],
            "current_priority": item["current_priority"],
            "current_cpu_id": item["current_cpu_id"],
            "wait_type": item["wait_type"],
            "wait_id": item["wait_id"],
            "stop_reason": item["stop_reason"],
            "pc": registers["pc"] if registers is not None else None,
            "lr": registers["lr"] if registers is not None else None,
            "sp": registers["sp"] if registers is not None else None,
        }

    thread_changes = [
        {
            "uid": uid,
            "left": thread_state(left_threads.get(uid)),
            "right": thread_state(right_threads.get(uid)),
        }
        for uid in sorted(set(left_threads) | set(right_threads))
        if thread_state(left_threads.get(uid)) != thread_state(right_threads.get(uid))
    ]

    left_allocations = left["allocation_evidence"]
    right_allocations = right["allocation_evidence"]
    allocations = _keyed_changes(
        left_allocations["allocations"],
        right_allocations["allocations"],
        ("name", "base", "size", "committed_size"),
        ignored_fields=("index",),
    )
    allocations["left_total_size"] = left_allocations["total_size"]
    allocations["right_total_size"] = right_allocations["total_size"]
    allocations["left_count"] = left_allocations["allocation_count"]
    allocations["right_count"] = right_allocations["allocation_count"]

    return {
        "verdict": {
            "left_classification": left["classification"],
            "right_classification": right["classification"],
            "left_failure_bucket_id": left["failure_bucket_id"],
            "right_failure_bucket_id": right["failure_bucket_id"],
            "changed": left["failure_bucket_id"] != right["failure_bucket_id"],
        },
        "fault_groups": faults,
        "context_assignment_changes": context_changes,
        "pipeline": {
            "left_master": left_pipeline["master"],
            "right_master": right_pipeline["master"],
            "master_changed": left_pipeline["master"] != right_pipeline["master"],
            "core_changes": pipeline_core_changes,
        },
        "resource_pointers": resources,
        "thread_changes": thread_changes,
        "allocations": allocations,
    }
