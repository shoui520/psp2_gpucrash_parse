from __future__ import annotations

import struct
from typing import Dict, List, Optional, Sequence

from .core import CoreDump, ParseError
from .execution import ExecutionContext, ModuleInfo
from .kernel import KernelObjectRegistry
from .registers import RegisterDatabase


def _captured(core: CoreDump, start: int, end: int) -> dict:
    ranges = core.captured_ranges(start, end)
    return {
        "captured_bytes": sum(last - first for first, last in ranges),
        "captured_ranges": [
            {"start": first, "end": last} for first, last in ranges
        ],
    }


def retained_map(core: CoreDump) -> dict:
    execution = ExecutionContext.parse(core)
    entries = []

    if execution.modules is not None:
        for module in execution.modules.modules:
            for segment in module.segments:
                entries.append(
                    {
                        "kind": "module-segment",
                        "start": segment.start,
                        "end": segment.end,
                        "size": segment.size,
                        "name": module.name,
                        "permissions": segment.permissions,
                        "module_index": module.index,
                        "module_uid": module.uid,
                        "segment_number": segment.number,
                        **_captured(core, segment.start, segment.end),
                    }
                )

    if execution.threads is not None:
        for thread in execution.threads.threads:
            entries.append(
                {
                    "kind": "thread-stack",
                    "start": thread.stack_base,
                    "end": thread.stack_end,
                    "size": thread.stack_size,
                    "name": thread.name,
                    "permissions": None,
                    "thread_index": thread.index,
                    "thread_uid": thread.uid,
                    **_captured(core, thread.stack_base, thread.stack_end),
                }
            )

    for block in core.memory_blocks():
        entries.append(
            {
                "kind": "memory-block",
                "start": block.base,
                "end": block.end,
                "size": block.size,
                "name": block.name,
                "permissions": None,
                "block_index": block.index,
                "block_uid": block.uid,
                "attributes": block.attributes,
                "committed_size": block.committed_size,
                **_captured(core, block.base, block.end),
            }
        )

    alias_lookup = {}
    for group in core.load_alias_groups():
        for index in group:
            alias_lookup[index] = group
    for segment in core.loads:
        entries.append(
            {
                "kind": "load-segment",
                "start": segment.virtual_address,
                "end": segment.end_address,
                "file_end": segment.virtual_address + segment.file_size,
                "size": segment.memory_size,
                "name": f"PT_LOAD ph{segment.index}",
                "permissions": "".join(
                    (
                        "r" if segment.flags & 4 else "-",
                        "w" if segment.flags & 2 else "-",
                        "x" if segment.flags & 1 else "-",
                    )
                ),
                "segment_index": segment.index,
                "file_size": segment.file_size,
                "alias_segment_indices": alias_lookup.get(segment.index, []),
                "captured_bytes": segment.file_size,
                "captured_ranges": [
                    {
                        "start": segment.virtual_address,
                        "end": segment.virtual_address + segment.file_size,
                    }
                ]
                if segment.file_size
                else [],
            }
        )

    kind_order = {
        "module-segment": 0,
        "thread-stack": 1,
        "memory-block": 2,
        "load-segment": 3,
    }
    entries.sort(
        key=lambda item: (
            item["start"],
            item["end"],
            kind_order[item["kind"]],
            item["name"],
        )
    )
    overlaps = []
    active = []
    for index, entry in enumerate(entries):
        active = [other for other in active if entries[other]["end"] > entry["start"]]
        for other in active:
            left = entries[other]
            start = max(left["start"], entry["start"])
            end = min(left["end"], entry["end"])
            if start < end:
                overlaps.append(
                    {
                        "left": other,
                        "right": index,
                        "start": start,
                        "end": end,
                        "size": end - start,
                        "exact_range": left["start"] == entry["start"]
                        and left["end"] == entry["end"],
                    }
                )
        active.append(index)
    return {
        "schema_version": 1,
        "command": "map",
        "path": str(core.path),
        "entry_count": len(entries),
        "captured_entry_count": sum(item["captured_bytes"] > 0 for item in entries),
        "metadata_only_entry_count": sum(
            item["captured_bytes"] == 0 for item in entries
        ),
        "overlap_count": len(overlaps),
        "entries": entries,
        "overlaps": overlaps,
    }


def modules_report(core: CoreDump) -> dict:
    execution = ExecutionContext.parse(core)
    modules = []
    if execution.modules is not None:
        summary = execution.summary(core)
        threads = summary["threads"]
        for module in execution.modules.modules:
            segments = []
            for segment in module.segments:
                segments.append({**segment.summary(), **_captured(core, segment.start, segment.end)})
            retained_pc_threads = [
                {
                    "index": thread["index"],
                    "uid": thread["uid"],
                    "name": thread["name"],
                    "pc": thread["registers"]["pc"] if thread["registers"] else None,
                    "location": thread["pc_location"],
                    "status_name": thread["status_name"],
                    "register_context_live": thread["register_context_live"],
                    "register_context_qualification": thread[
                        "register_context_qualification"
                    ],
                }
                for thread in threads
                if thread["pc_location"] is not None
                and thread["pc_location"]["module_index"] == module.index
            ]
            modules.append(
                {
                    "index": module.index,
                    "uid": module.uid,
                    "name": module.name,
                    "record_word0": module.record_word0,
                    "segments": segments,
                    "total_size": sum(segment.size for segment in module.segments),
                    "captured_bytes": sum(item["captured_bytes"] for item in segments),
                    "threads_with_retained_pc": retained_pc_threads,
                    "execution_threads": [
                        thread
                        for thread in retained_pc_threads
                        if thread["register_context_live"]
                    ],
                }
            )
    return {
        "schema_version": 1,
        "command": "modules",
        "path": str(core.path),
        "module_count": len(modules),
        "modules": modules,
    }


def _select_module(modules: Sequence[dict], selector: str) -> dict:
    numeric = None
    try:
        numeric = int(selector, 0)
    except ValueError:
        pass
    if numeric is not None:
        uid = [item for item in modules if item["uid"] == numeric]
        if uid:
            return uid[0]
        index = [item for item in modules if item["index"] == numeric]
        if index:
            return index[0]
    exact = [item for item in modules if item["name"].casefold() == selector.casefold()]
    if len(exact) == 1:
        return exact[0]
    partial = [item for item in modules if selector.casefold() in item["name"].casefold()]
    if len(partial) == 1:
        return partial[0]
    if len(exact) > 1 or len(partial) > 1:
        raise ParseError(f"module selector {selector!r} is ambiguous")
    raise ParseError(f"module selector {selector!r} did not match a UID, index, or name")


def module_report(core: CoreDump, selector: str) -> dict:
    report = modules_report(core)
    selected = _select_module(report["modules"], selector)
    return {
        "schema_version": 1,
        "command": "module",
        "path": str(core.path),
        "selector": selector,
        "module": selected,
    }


def _classification_indexes(core: CoreDump, diagnosis: dict) -> dict:
    execution = diagnosis["execution_context"]
    modules = ExecutionContext.parse(core).modules
    blocks = core.memory_blocks()
    registry = KernelObjectRegistry.parse(core)
    threads = execution["threads"]
    resources = diagnosis["resource_pointers"]["pointers"]
    faults = diagnosis["fault_groups"]
    return {
        "modules": modules,
        "blocks": blocks,
        "objects": {item.uid: item for item in registry.objects},
        "threads_by_uid": {item["uid"]: item for item in threads},
        "threads": threads,
        "resources": resources,
        "faults": faults,
    }


def classify_value(core: CoreDump, indexes: dict, value: int) -> List[dict]:
    if value in (0, 0xFFFFFFFF):
        return []
    result = []
    modules: Optional[ModuleInfo] = indexes["modules"]
    location = modules.locate(value) if modules is not None else None
    if location is not None:
        result.append({"kind": "module-address", **location.summary()})
    for block in indexes["blocks"]:
        if block.base <= value < block.end:
            result.append(
                {
                    "kind": "memory-block-pointer",
                    "index": block.index,
                    "uid": block.uid,
                    "name": block.name,
                    "base": block.base,
                    "end": block.end,
                    "offset": value - block.base,
                    "captured_at_value": bool(core.captured_bytes(value, value + 1)),
                }
            )
    for thread in indexes["threads"]:
        if thread["stack_base"] <= value < thread["stack_end"]:
            result.append(
                {
                    "kind": "thread-stack-pointer",
                    "thread_index": thread["index"],
                    "thread_uid": thread["uid"],
                    "thread_name": thread["name"],
                    "offset": value - thread["stack_base"],
                }
            )
    if value in indexes["threads_by_uid"]:
        thread = indexes["threads_by_uid"][value]
        result.append(
            {
                "kind": "thread-uid",
                "thread_index": thread["index"],
                "thread_uid": thread["uid"],
                "thread_name": thread["name"],
            }
        )
    if value in indexes["objects"]:
        item = indexes["objects"][value]
        result.append(
            {
                "kind": "kernel-object-uid",
                "uid": item.uid,
                "object_kind": item.kind,
                "name": item.name,
            }
        )
    for pointer in indexes["resources"]:
        if pointer["address"] == value:
            result.append(
                {
                    "kind": "gpu-resource-address",
                    "code": pointer["code"],
                    "name": pointer["name"],
                    "stage": pointer["stage"],
                }
            )
    page = value & 0xFFFFF000
    for fault in indexes["faults"]:
        # A retained null-page GPU fault must not turn every small scalar
        # stack word into a pointer candidate.  Zero itself is intentionally
        # excluded above because it is too common to be useful in this view.
        if page != 0 and fault["gpu_virtual_address"] == page:
            result.append(
                {
                    "kind": "gpu-fault-page",
                    "fault_group": fault["index"],
                    "page": page,
                    "offset": value - page,
                    "requestor_names": fault["requestor_names"],
                }
            )
    if any(segment.contains(value) for segment in core.loads):
        result.append({"kind": "captured-address"})
    return result


def annotated_stack(
    core: CoreDump, diagnosis: dict, thread: dict, byte_limit: int
) -> dict:
    registers = thread["registers"]
    if not thread.get("register_context_live", True):
        return {
            "status": "non-live-thread-context",
            "qualification": thread.get("register_context_qualification"),
            "start": registers["sp"] if registers is not None else None,
            "requested_bytes": max(0, byte_limit),
            "captured_bytes": 0,
            "word_count": 0,
            "classified_word_count": 0,
            "words": [],
        }
    if registers is None or registers["sp"] == 0 or byte_limit <= 0:
        return {
            "status": "unavailable",
            "qualification": None,
            "start": registers["sp"] if registers is not None else None,
            "requested_bytes": max(0, byte_limit),
            "captured_bytes": 0,
            "word_count": 0,
            "classified_word_count": 0,
            "words": [],
        }
    start = registers["sp"]
    end = min(thread["stack_end"], start + byte_limit)
    ranges = core.captured_ranges(start, end)
    if not ranges or ranges[0][0] != start:
        captured_end = start
    else:
        captured_end = ranges[0][1]
    size = (captured_end - start) & ~3
    data = core.read_memory(start, size) if size else b""
    indexes = _classification_indexes(core, diagnosis)
    words = []
    for offset in range(0, len(data), 4):
        value = struct.unpack_from("<I", data, offset)[0]
        classifications = classify_value(core, indexes, value)
        words.append(
            {
                "stack_address": start + offset,
                "stack_offset": start + offset - thread["stack_base"],
                "sp_offset": offset,
                "value": value,
                "classifications": classifications,
            }
        )
    return {
        "status": "available",
        "qualification": "captured words classified by retained address metadata",
        "start": start,
        "requested_bytes": byte_limit,
        "captured_bytes": len(data),
        "word_count": len(words),
        "classified_word_count": sum(bool(item["classifications"]) for item in words),
        "words": words,
    }


def xref_report(
    core: CoreDump,
    database: RegisterDatabase,
    address: int,
    max_results: int = 256,
) -> dict:
    # Local import avoids coupling the core debugger model back to this
    # higher-level inspection module.
    from .debugger import address_report, load_debug_state, search_memory

    _info, diagnosis = load_debug_state(core, database)
    base = address_report(core, database, address, preview_size=0)
    pattern = struct.pack("<I", address)
    occurrences = search_memory(
        core,
        pattern,
        f"little-endian pointer to {address:#x}",
        alignment=4,
        max_results=max_results,
    )
    threads = diagnosis["execution_context"]["threads"]
    for occurrence in occurrences["results"]:
        source = occurrence["address"]
        stack_owners = [
            {
                "index": thread["index"],
                "uid": thread["uid"],
                "name": thread["name"],
                "status_name": thread["status_name"],
                "register_context_live": thread["register_context_live"],
                "register_context_qualification": thread[
                    "register_context_qualification"
                ],
                "stack_offset": source - thread["stack_base"],
                "sp_offset": (
                    source - thread["registers"]["sp"]
                    if thread["registers"] is not None
                    and thread["register_context_live"]
                    else None
                ),
            }
            for thread in threads
            if thread["stack_base"] <= source < thread["stack_end"]
        ]
        occurrence["stack_owners"] = stack_owners
        occurrence["source_classification"] = (
            "thread-stack"
            if stack_owners
            else "module"
            if occurrence["module"] is not None
            else "memory-block"
            if occurrence["memory_block"] is not None
            else "captured-segment"
        )
    direct = []
    for thread in base["threads"]:
        for register in thread["registers"]:
            direct.append(
                {
                    "kind": "cpu-register",
                    "thread_index": thread["index"],
                    "thread_uid": thread["uid"],
                    "thread_name": thread["name"],
                    "status_name": thread["status_name"],
                    "register_context_live": thread["register_context_live"],
                    "register_context_qualification": thread[
                        "register_context_qualification"
                    ],
                    "register": register,
                    "register_value_poison": address == 0xDEADBEEF,
                }
            )
    for group in base["gpu_fault_groups"]:
        direct.append({"kind": "gpu-fault", "fault_group": group})
    for pointer in base["gpu_resource_pointers"]:
        direct.append({"kind": "gpu-resource", "resource": pointer})
    for register in base["gpu_register_references"]:
        direct.append({"kind": "gpu-register", "register": register})
    return {
        "schema_version": 1,
        "command": "xref",
        "path": str(core.path),
        "address": address,
        "address_status": base["status"],
        "target": {
            key: base[key]
            for key in (
                "status",
                "conclusion",
                "value_notes",
                "associations",
                "module",
                "memory_blocks",
                "load_segments",
                "threads",
                "gpu_fault_groups",
                "gpu_resource_pointers",
                "gpu_register_references",
            )
        },
        "direct_reference_count": len(direct),
        "memory_reference_count": occurrences["match_count"],
        "returned_memory_reference_count": occurrences["returned_count"],
        "truncated": occurrences["truncated"],
        "direct_references": direct,
        "memory_references": occurrences["results"],
        "limitations": [
            "Cross-references cover CPU registers, retained GPU state, and captured PT_LOAD bytes only."
        ],
    }
