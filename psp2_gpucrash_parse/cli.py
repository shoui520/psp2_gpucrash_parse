from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import struct
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import __version__
from .activity import GpuActivityInfo
from .core import CoreDump, ParseError
from .debugger import (
    address_report,
    compare_debug_state,
    load_debug_state,
    search_memory,
    thread_detail_report,
    thread_list_report,
)
from .diagnosis import build_diagnosis
from .devkit import DevkitGpuDump, correlate_gpu_dump
from .elfimage import enrich_address_with_elfs, enrich_diagnosis_with_elfs
from .gpu import GPU_CONTEXT_MAP_SCOPES, GPU_RENDER_STATE_FIELDS, GpuInfo, compare_gpu_info
from .indexed import indexed_selector_metadata
from .inspection import (
    annotated_stack,
    module_report,
    modules_report,
    retained_map,
    xref_report,
)
from .metadata import SummaryInfo
from .registers import RegisterDatabase, decoded_fields
from .triage import triage_dumps
from .tty import TtyInfo
from .unwind import unwind_thread
from .validation import validate_core
from .waitgraph import object_report, wait_graph


def _number(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _database(args: argparse.Namespace) -> RegisterDatabase:
    return RegisterDatabase.load(args.register_header or ())


def _gpu_info(core: CoreDump) -> Optional[GpuInfo]:
    note = core.note("GPU_INFO")
    return GpuInfo.parse(note.description) if note is not None else None


def _gpu_activity(core: CoreDump) -> Optional[GpuActivityInfo]:
    note = core.note("GPU_ACT_INFO")
    return GpuActivityInfo.parse(note.description) if note is not None else None


def _hex(value: Optional[int], width: int = 8) -> str:
    return "missing" if value is None else f"0x{value:0{width}x}"


def _trim_display_line(value: str, limit: int = 240) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _hexdump(data: bytes, address: int) -> str:
    lines = []
    for offset in range(0, len(data), 16):
        row = data[offset : offset + 16]
        hex_bytes = " ".join(f"{item:02x}" for item in row)
        text = "".join(chr(item) if 32 <= item < 127 else "." for item in row)
        lines.append(f"{address + offset:08x}  {hex_bytes:<47}  |{text}|")
    return "\n".join(lines)


def _classification_text(items: Sequence[dict]) -> str:
    labels = []
    for item in items:
        kind = item["kind"]
        if kind == "module-address":
            labels.append(item["notation"])
        elif kind == "memory-block-pointer":
            labels.append(
                f"mem:{item['name'] or '<unnamed>'}+0x{item['offset']:x}"
            )
        elif kind == "thread-stack-pointer":
            owner = item["thread_name"] or f"0x{item['thread_uid']:08x}"
            labels.append(
                f"stack:{owner}+0x{item['offset']:x}"
            )
        elif kind == "thread-uid":
            labels.append(f"thread-uid:{item['thread_name'] or '<unnamed>'}")
        elif kind == "kernel-object-uid":
            labels.append(
                f"{item['object_kind']}-uid:{item['name'] or '<unnamed>'}"
            )
        elif kind == "gpu-resource-address":
            labels.append(f"gpu:{item['name']}")
        elif kind == "gpu-fault-page":
            labels.append(f"fault-page+0x{item['offset']:x}")
        elif kind != "captured-address":
            labels.append(kind)
    if not labels and any(item["kind"] == "captured-address" for item in items):
        labels.append("captured")
    return ", ".join(labels)


def _print_analysis_banner() -> None:
    border = "*" * 79
    print(border)
    print("*" + "GPUCRASH Analysis".center(77) + "*")
    print(border)


def _print_analysis_summary(
    core: CoreDump, info: GpuInfo, diagnosis: dict
) -> None:
    memory = diagnosis["memory_evidence"]
    summary = diagnosis["summary_integrity"]
    execution = diagnosis["execution_context"]
    kernel = diagnosis["kernel_objects"]
    pipeline = diagnosis["firmware_pipeline"]
    support = diagnosis["supporting_context"]
    application = support["application"] or {}
    external_processes = support["external_processes"] or {}
    libraries = support["libraries"] or {}
    budgets = support["budgets"] or {}
    event_log = support["event_log"] or {}
    binary_images = diagnosis.get("binary_images") or {}
    validation = diagnosis.get("validation") or {}
    waits = diagnosis.get("wait_graph") or {}
    process = execution["process"] or {}
    timestamp = info.capture_timestamp()
    groups = [
        (
            "Analysis",
            [
                ("Analyzer version", __version__),
                ("Analysis code", diagnosis["analysis_code"]),
                (
                    "Classification confidence",
                    diagnosis["classification_confidence"],
                ),
                ("Root-cause confidence", diagnosis["root_cause_confidence"]),
                ("Fixed-format validation", validation.get("status", "not-run")),
            ],
        ),
        (
            "Dump image",
            [
                ("Image size", f"0x{core.image_size:x} ({core.image_size} bytes)"),
                ("Notes", len(core.notes)),
                ("Load segments", len(core.loads)),
                ("Unique load segments", memory["unique_load_segment_count"]),
            ],
        ),
        (
            "GPU capture",
            [
                (
                    "Capture time",
                    (timestamp["iso_local"] + " (local; timezone not retained)")
                    if timestamp["iso_local"]
                    else "unavailable",
                ),
                ("Capture flags", _hex(info.capture_flags)),
                (
                    "Snapshot complete",
                    "yes" if diagnosis["snapshot"]["complete"] else "no",
                ),
                ("GPU cores", info.core_count),
                ("Master registers", len(info.master_registers)),
                ("Per-core registers", len(info.per_core_registers)),
                ("Indexed-state entries", len(info.indexed_state)),
                ("Fault records", len(diagnosis["fault_sources"])),
                ("Fault groups", len(diagnosis["fault_groups"])),
                ("Error signals", len(diagnosis["event_signals"])),
                ("Lockup phases", len(diagnosis["lockup_phases"])),
                ("Active cores", pipeline["active_core_count"]),
                ("Phase-active units", pipeline["phase_active_unit_count"]),
                ("Low-level active units", pipeline["active_unit_count"]),
                ("Context owners", len(diagnosis["context_owners"])),
                ("Activity state", diagnosis["activity_state"]),
            ],
        ),
        (
            "Application and process",
            [
                ("Title ID", application.get("title_id") or "unavailable"),
                ("Title", application.get("title_name") or "unavailable"),
                ("External processes", external_processes.get("decoded_count", 0)),
                ("Process", process.get("process_name") or "unavailable"),
                ("Process ID", _hex(process.get("process_id"))),
                ("Threads", execution["thread_count"]),
                (
                    "Non-kernel execution threads",
                    len(execution["non_kernel_execution_thread_indices"]),
                ),
                ("Modules", (execution["modules"] or {}).get("module_count", 0)),
            ],
        ),
        (
            "Captured support data",
            [
                ("Kernel objects", kernel["object_count"]),
                ("Active kernel objects", kernel["active_object_count"]),
                ("Resolved thread waits", kernel["resolved_thread_wait_count"]),
                ("Wait dependency edges", waits.get("edge_count", 0)),
                ("Wait dependency cycles", waits.get("cycle_count", 0)),
                ("TTY lines", diagnosis["log"]["line_count"]),
                ("Libraries", libraries.get("decoded_name_count", 0)),
                ("Memory-budget partitions", budgets.get("partition_count", 0)),
                ("System event records", event_log.get("record_count", 0)),
            ],
        ),
        (
            "Original binary images",
            [
                ("Requested", binary_images.get("requested_count", 0)),
                ("Matched", binary_images.get("matched_count", 0)),
                ("Rejected", binary_images.get("rejected_count", 0)),
            ],
        ),
    ]
    if summary is not None:
        groups[1][1].append(("Incomplete summary records", summary["incomplete_count"]))
    print("GPUCRASH analysis summary:")
    for index, (group_name, values) in enumerate(groups):
        if index:
            print(f"\n{group_name}:")
        for name, value in values:
            print(f"{name}: {value}")


def _print_dump_attributes(core: CoreDump) -> None:
    print(f"\nDUMP_FILE:  {core.path}")
    print("\nDUMP_FILE_ATTRIBUTES:")
    print(f"    Compression       : {'gzip' if core.compressed else 'none'}")
    print(f"    ELF class/machine : ELF32 / {core.machine}")
    print(f"    ELF type/flags    : {core.elf_type} / {_hex(core.flags)}")
    print(f"    Image size        : 0x{core.image_size:x} ({core.image_size} bytes)")
    print(f"    Notes/PT_LOAD     : {len(core.notes)} / {len(core.loads)}")
    print(f"    File SHA-256      : {core.raw_file_sha256}")
    print(f"    Image SHA-256     : {core.image_sha256}")


def _print_note_coverage(diagnosis: dict) -> None:
    coverage = diagnosis["note_coverage"]
    print("\nNOTE_DECODER_COVERAGE:")
    print(f"    Distinct note names   : {coverage['note_name_count']}")
    print(f"    Structurally decoded  : {coverage['structured_count']}")
    print("        " + ", ".join(coverage["structured_names"]))
    print(f"    Inventory only        : {coverage['inventory_only_count']}")
    print(
        "        "
        + (", ".join(coverage["inventory_only_names"]) or "none")
    )
    print("    Inventory-only notes remain hashed and sized in JSON; fields are not interpreted.")


def _print_supporting_context(diagnosis: dict) -> None:
    support = diagnosis["supporting_context"]
    application = support["application"]
    application_list = support["application_list"]
    external_processes = support["external_processes"]
    build = support["build"]
    system = support["system"]
    hardware = support["hardware"]
    libraries = support["libraries"]
    budgets = support["budgets"]
    event_log = support["event_log"]
    callbacks = support["callbacks"]
    timers = support["timers"]
    system2 = support["system2"]
    corefile = support["corefile"]
    device = support["device"]
    files = support["files"]
    metadata = support["metadata"]
    system_devices = support["system_devices"]

    print("\nCAPTURE_ENVIRONMENT:")
    if application is None:
        print("    APP_INFO unavailable")
    else:
        print(f"    Title ID              : {application['title_id'] or '<unavailable>'}")
        print(f"    Title                 : {application['title_name'] or '<unavailable>'}")
        print(f"    Title version         : {application['title_version'] or '<unavailable>'}")
    if system is not None:
        print(
            "    Software version words: "
            + " ".join(_hex(value) for value in system["software_version_words"])
        )
    if hardware is not None:
        print(f"    Hardware revision word: {_hex(hardware['hardware_revision_word'])}")
        print(f"    Hardware flags word   : {_hex(hardware['hardware_flags_word'])}")
    if build is not None:
        print(
            "    Producer build branch : "
            + (", ".join(build["unique_branch_strings"]) or "<unavailable>")
        )
    print("    Version and hardware fields are retained producer words unless named above.")

    print("\nCAPTURE_AUXILIARY_RECORDS:")
    if corefile is not None:
        print(
            f"    Corefile kind/subtype  : {_hex(corefile['producer_kind'])} / "
            f"{_hex(corefile['producer_subtype'])}"
        )
    if system2 is not None:
        print(
            f"    SYSTEM_INFO2 envelope  : header=0x{system2['header_size']:x} "
            f"payload=0x{system2['payload_size']:x} "
            f"nonzero=0x{system2['payload_nonzero_bytes']:x}"
        )
        print(f"    SYSTEM_INFO2 payload   : sha256={system2['payload_sha256']}")
    if system_devices is not None:
        print(f"    System-device records  : {system_devices['record_count']}")
        for record in system_devices["records"]:
            print(
                f"        [{record['index']:02d}] size=0x{record['record_size']:x} "
                f"{record['description'] or '<no description>'}"
            )
    if files is not None:
        print(
            f"    Captured file records  : {files['record_count']} "
            f"(fixed size 0x{files['record_size']:x})"
        )
        file_owners = []
        for record in files["records"]:
            identity = (record["process_id"], record["process_name"])
            if identity not in file_owners:
                file_owners.append(identity)
        print(
            "    File-record owners     : "
            + ", ".join(
                f"{name or '<unresolved>'}({_hex(process_id)})"
                for process_id, name in file_owners
            )
        )
    if metadata is not None:
        identifiers = " ".join(
            f"{_hex(record['identifier'])}"
            + (f"={record['module_name']}" if record.get("module_name") else "")
            for record in metadata["entries"][:8]
        )
        suffix = " ..." if metadata["entry_count"] > 8 else ""
        print(f"    Metadata identifiers   : {metadata['entry_count']}")
        if "resolved_module_count" in metadata:
            print(
                f"    Metadata module UIDs   : {metadata['resolved_module_count']} resolved; "
                f"complete={str(metadata['module_count_matches']).lower()}"
            )
        print(f"        {identifiers}{suffix}")
    if device is not None:
        unique_references = []
        for reference in device["process_references"]:
            identity = (reference["process_id"], reference["process_name"])
            if identity not in unique_references:
                unique_references.append(identity)
        print(
            "    DEVICE process refs    : "
            + (
                ", ".join(f"{name}({_hex(pid)})" for pid, name in unique_references)
                if unique_references
                else "none resolved"
            )
        )
    print(
        "    Unnamed producer words and opaque SYSTEM_INFO2 payload bytes are "
        "preserved in JSON."
    )

    print("\nAPPLICATION_AND_PROCESS_SET:")
    if application_list is None:
        print("    APP_LIST_INFO unavailable")
    else:
        print(
            f"    Applications          : {application_list['decoded_count']} / "
            f"{application_list['declared_count']} decoded"
        )
        for item in application_list["records"]:
            path = f" path={item['path']}" if item["path"] else ""
            print(
                f"        {item['title_id']} pid={_hex(item['process_id'])} "
                f"parent={_hex(item['parent_process_id'])}{path}"
            )
    if external_processes is None:
        print("    EXTNL_PROC_INFO unavailable")
    else:
        print(
            f"    External processes    : {external_processes['decoded_count']} / "
            f"{external_processes['declared_count']} decoded"
        )
        for item in external_processes["records"]:
            path = f" path={item['path']}" if item["path"] else ""
            print(
                f"        {item['name']} pid={_hex(item['process_id'])} "
                f"uid={_hex(item['uid'])} attr={_hex(item['attributes'])}{path}"
            )

    print("\nCAPTURED_LIBRARY_SET:")
    if libraries is None:
        print("    LIBRARY_INFO unavailable")
    else:
        print(f"    Declared libraries    : {libraries['declared_count']}")
        print(f"    Decoded names         : {libraries['decoded_name_count']}")
        print(f"    Count matches         : {str(libraries['name_count_matches']).lower()}")
        print(
            "    GPU-related libraries : "
            + (", ".join(libraries["gpu_related_names"]) or "none")
        )
        print("    Complete name set     :")
        for base in range(0, len(libraries["names"]), 4):
            print("        " + ", ".join(libraries["names"][base : base + 4]))

    print("\nSYSTEM_EVENT_RECORDS:")
    if event_log is None:
        print("    EVENT_LOG_INFO unavailable")
    else:
        print(f"    Format/records        : {event_log['format_version']} / {event_log['record_count']}")
        print(f"    Producer ring word    : {_hex(event_log['ring_word'])}")
        for record in event_log["records"]:
            related = (
                f" related={record['related_name']}({_hex(record['related_id'])})"
                if record["related_name"] or record["related_id"] is not None
                else ""
            )
            print(
                f"    [{record['index']:02d}] title={record['title_id'] or '<none>'} "
                f"event={_hex(record['event_code'])} source={_hex(record['source_id'])} "
                f"detail={_hex(record['detail_code'])}{related}"
            )
        print("    Record order is producer order; event-code semantics remain unnamed.")

    print("\nPROCESS_MEMORY_BUDGETS:")
    if budgets is None:
        print("    BUDGET_INFO unavailable")
    else:
        print(
            f"    Budgets/partitions     : {budgets['budget_count']} / "
            f"{budgets['partition_count']}"
        )
        for budget in budgets["budgets"]:
            print(
                f"    {budget['name'] or '<unnamed>'} uid={_hex(budget['uid'])}:"
            )
            for partition in budget["partitions"]:
                utilization = (
                    f"{partition['utilization'] * 100:.1f}%"
                    if partition["utilization"] is not None
                    else "n/a"
                )
                print(
                    f"        {partition['name'] or '<unnamed>'}: "
                    f"base={_hex(partition['base'])} size=0x{partition['size']:x} "
                    f"used=0x{partition['used']:x} free=0x{partition['free']:x} "
                    f"largest-free=0x{partition['largest_free_block']:x} "
                    f"utilization={utilization}"
                )
        print("    Partition usage is process-budget state, not SGX DPM page usage.")

    print("\nASYNCHRONOUS_KERNEL_OBJECTS:")
    print(
        f"    Callbacks              : {callbacks['record_count'] if callbacks else 0}"
    )
    print(f"    Timers                 : {timers['record_count'] if timers else 0}")
    if timers is not None:
        for timer in timers["records"]:
            print(
                f"        {_hex(timer['uid'])} pid={_hex(timer['process_id'])} "
                f"{timer['name'] or '<unnamed>'}"
            )


def _location_text(location: Optional[dict], address: Optional[int]) -> str:
    if address is None:
        return "unavailable"
    if location is None:
        return _hex(address)
    return f"{_hex(address)} ({location['notation']}, {location['segment_permissions']})"


def _address_capture_text(evidence: dict) -> str:
    blocks = evidence["memory_blocks"]
    if blocks:
        return "; ".join(
            f"{item['name'] or '<unnamed>'}+0x{item['offset']:x} "
            f"(MEM_BLK {item['index']})"
            for item in blocks
        )
    loads = evidence["load_segments"]
    if loads:
        return "; ".join(
            f"PT_LOAD {item['index']}+0x{item['offset']:x}" for item in loads
        )
    return "no direct MEM_BLK/PT_LOAD numeric overlap"


def _image_address_text(location: dict) -> str:
    symbol = location["symbol"]
    symbolic = (
        f" {symbol['name']}+0x{symbol['offset']:x}" if symbol is not None else ""
    )
    file_offset = (
        f" file+0x{location['file_offset']:x}"
        if location["file_offset"] is not None
        else " non-file-backed"
    )
    imported = location.get("import")
    import_text = ""
    if imported is not None:
        import_name = imported["name"] or f"NID_0x{imported['nid']:08x}"
        import_text = f" {imported['library_name']}:{import_name}"
    return (
        f"ELF={location['elf_virtual_address']:08x}{file_offset}{symbolic}{import_text}"
    )


def _print_execution_context(diagnosis: dict) -> None:
    execution = diagnosis["execution_context"]
    process = execution["process"]
    modules = execution["modules"]
    threads = execution["threads"]
    print("\nPROCESS_CONTEXT:")
    if process is None:
        print("    PROCESS_INFO unavailable")
    else:
        print(f"    Name            : {process['process_name'] or '<unnamed>'}")
        print(f"    Process ID      : {_hex(process['process_id'])}")
        print(f"    Attributes      : {_hex(process['attributes'])}")
        print(f"    Flags           : {_hex(process['flags'])}")
        print(f"    Executable      : {process['executable_path'] or '<unavailable>'}")

    print("\nLOADED_MODULES:")
    if modules is None:
        print("    MODULE_INFO unavailable")
    else:
        print(f"    Count: {modules['module_count']}")
        for module in modules["modules"]:
            print(
                f"    [{module['index']:02d}] uid={_hex(module['uid'])} "
                f"{module['name'] or '<unnamed>'}"
            )
            for segment in module["segments"]:
                print(
                    f"         @{segment['number']} {segment['permissions']} "
                    f"{segment['start']:08x}-{segment['end']:08x} "
                    f"size=0x{segment['size']:x} attr={_hex(segment['attributes'])}"
                )

    print("\nCPU_THREADS:")
    if not threads:
        print("    THREAD_INFO or THREAD_REG_INFO unavailable")
        return
    print(
        "    Index UID       Status     Pri  Thread name / register context"
    )
    for thread in threads:
        registers = thread["registers"]
        print(
            f"    [{thread['index']:03d}] {thread['uid']:08x} "
            f"{thread['status_name']:<10} {thread['current_priority']:4d} "
            f"{thread['name'] or '<unnamed>'}"
        )
        if registers is None:
            print("          register context unavailable")
            continue
        print(
            f"          context={thread['register_context_qualification']} "
            f"poison-registers={thread['poison_register_count']}/16"
        )
        print(
            f"          PC={_location_text(thread['pc_location'], registers['pc'])}"
        )
        if thread.get("pc_image") is not None:
            print(f"             {_image_address_text(thread['pc_image'])}")
        print(
            f"          LR={_location_text(thread['lr_location'], registers['lr'])} "
            f"SP={_hex(registers['sp'])} CPSR={_hex(registers['cpsr'])} "
            f"stack-captured={str(thread['stack_captured_at_sp']).lower()}"
        )
        if thread.get("lr_image") is not None:
            print(f"             {_image_address_text(thread['lr_image'])}")
        stack = thread["stack_usage"]
        if stack is not None:
            print(
                f"          stack={thread['stack_base']:08x}-{thread['stack_end']:08x} "
                f"size=0x{thread['stack_size']:x} current=0x{stack['current_bytes']:x} "
                f"peak=0x{stack['peak_bytes']:x}"
            )
        if thread["wait_type"] or thread["wait_id"]:
            print(
                f"          wait-type={thread['wait_type']} "
                f"wait-id={_hex(thread['wait_id'])}"
            )
            wait_object = thread["wait_object"]
            if wait_object is not None:
                print(
                    f"          wait-object={wait_object['kind']} "
                    f"{wait_object['name'] or '<unnamed>'} "
                    f"waiter-record={str(thread['waiter_record_present']).lower()}"
                )

    selected_indices = set(execution["non_kernel_execution_thread_indices"])
    selected_indices.update(
        thread["index"]
        for thread in threads
        if thread["name"] == "SceGxmDisplayQueue"
        and thread["register_context_live"]
    )
    selected = [thread for thread in threads if thread["index"] in selected_indices]
    print("\nCPU_REGISTER_CONTEXT_SELECTED:")
    if not selected:
        print(
            "    No live saved context has a PC outside SceLibKernel, and no "
            "live SceGxmDisplayQueue context exists."
        )
    else:
        print(
            "    Selection: live saved context with a non-SceLibKernel PC, "
            "or live SceGxmDisplayQueue context."
        )
    for thread in selected:
        registers = thread["registers"]
        if registers is None:
            continue
        gpr = registers["gpr"]
        print(
            f"    Thread {thread['index']} {thread['name']} uid={_hex(thread['uid'])}:"
        )
        for base in (0, 4, 8, 12):
            fields = []
            for index in range(base, min(base + 4, 16)):
                name = {13: "SP", 14: "LR", 15: "PC"}.get(index, f"R{index}")
                fields.append(f"{name}={gpr[index]:08x}")
            print("        " + " ".join(fields))
        print(f"        CPSR={registers['cpsr']:08x}")

    print("\nHEURISTIC_STACK_RETURN_CANDIDATES:")
    candidate_threads = [
        thread for thread in selected if thread["stack_return_candidates"]
    ]
    if not candidate_threads:
        print("    No executable module addresses found in selected captured stacks.")
    for thread in candidate_threads:
        print(
            f"    Thread {thread['index']} {thread['name']} "
            f"({len(thread['stack_return_candidates'])} shown):"
        )
        for candidate in thread["stack_return_candidates"]:
            image_suffix = (
                f"; {_image_address_text(candidate['image'])}"
                if candidate.get("image") is not None
                else ""
            )
            print(
                f"        [{candidate['stack_address']:08x}] "
                f"{candidate['value']:08x} -> {candidate['location']['notation']}"
                f"{image_suffix}"
            )
    print("    Entries are stack-scan candidates, not unwind-verified frames.")


def _print_binary_images(diagnosis: dict) -> None:
    binaries = diagnosis.get("binary_images") or {
        "requested_count": 0,
        "matched_count": 0,
        "rejected_count": 0,
        "invalid_count": 0,
        "objdump": None,
        "addr2line": None,
        "nid_database_sources": [],
        "images": [],
    }
    print("\nORIGINAL_BINARY_IMAGES:")
    print(f"    Requested             : {binaries['requested_count']}")
    print(f"    Matched/rejected      : {binaries['matched_count']} / {binaries['rejected_count']}")
    print(f"    Invalid               : {binaries['invalid_count']}")
    if not binaries["images"]:
        print("    No ELF or SELF was supplied; runtime module+offset notation is retained.")
        return
    print(f"    Objdump               : {binaries['objdump'] or '<unavailable>'}")
    print(f"    Addr2line             : {binaries['addr2line'] or '<unavailable>'}")
    print(
        "    NID database          : "
        + (", ".join(binaries["nid_database_sources"]) or "<unavailable>")
    )
    for index, image in enumerate(binaries["images"]):
        print(f"\n    IMAGE_{index}: {image['path']}")
        if image.get("status") == "invalid":
            print(f"        Status            : invalid ({image['error']})")
            continue
        print(f"        Source format     : {image['source_format'].upper()}")
        print(f"        SHA-256           : {image['sha256']}")
        print(
            f"        Source/ELF size   : 0x{image['source_size']:x} / "
            f"0x{image['reconstructed_elf_size']:x}"
        )
        if image["self_header"] is not None:
            self_header = image["self_header"]
            print(
                f"        SELF version      : {self_header['version']} "
                f"(SDK type=0x{self_header['sdk_type']:04x}, "
                f"compressed={self_header['compressed_segment_count']}/"
                f"{self_header['program_header_count']})"
            )
        print(f"        ELF type/machine  : {_hex(image['elf_type'], 4)} / {image['machine']}")
        print(f"        Embedded module   : {image['module_name'] or '<unavailable>'}")
        if image["module_version"] is not None:
            print(
                f"        Module version    : {image['module_version'][0]}."
                f"{image['module_version'][1]}"
            )
        print(
            f"        Sections/symbols  : {image['section_count']} / "
            f"{image['symbol_count']}"
        )
        print(
            f"        Import libraries  : {image['import_library_count']} "
            f"({image['import_function_count']} functions, "
            f"{image['resolved_import_function_count']} NIDs named)"
        )
        for ordinal, segment in enumerate(image["load_segments"], 1):
            print(
                f"        PT_LOAD {ordinal}       : {segment['permissions']} "
                f"va={segment['virtual_address']:08x} file=0x{segment['file_size']:x} "
                f"memory=0x{segment['memory_size']:x} align=0x{segment['alignment']:x}"
            )
        association = image["association"]
        print(f"        Association       : {association['status']}")
        if association["status"] == "matched":
            print(
                f"        Runtime module    : [{association['module_index']:02d}] "
                f"uid={_hex(association['module_uid'])} "
                f"{association['runtime_module_name']}"
            )
            print(f"        Match quality     : {association['match_quality']}")
            for warning in association.get("warnings", []):
                print(f"        Compatibility     : {warning}")
            print(
                f"        Resolved PC/LR    : {image['resolved_address_count']} "
                "unique addresses"
            )
        for reason in association["reasons"]:
            print(f"        Rejection reason  : {reason}")

        for record in image["address_records"]:
            location = record["location"]
            references = ", ".join(
                f"thread {item['thread_index']} {item['thread_name'] or '<unnamed>'} "
                f"{item['role'].upper()}"
                for item in record["references"]
            )
            print(
                f"\n        {_hex(record['runtime_address'])} -> "
                f"ELF {_hex(location['elf_virtual_address'])} "
                f"file={_hex(location['file_offset'])} "
                f"({record['instruction_set']}, {record['instruction_set_basis']})"
            )
            print(f"            Referenced by: {references}")
            symbol = location["symbol"]
            if symbol is not None:
                print(
                    f"            Symbol       : {symbol['name']}+0x{symbol['offset']:x} "
                    f"(size=0x{symbol['size']:x})"
                )
            imported = location.get("import")
            if imported is not None:
                import_name = imported["name"] or f"NID_0x{imported['nid']:08x}"
                print(
                    f"            Import stub  : {imported['library_name']}:"
                    f"{import_name}"
                )
            source = record["source"]
            if source is not None:
                print(
                    f"            Source       : {source['function']} at "
                    f"{source['file_line']}"
                )
            if record["disassembly"]:
                print("            Disassembly  :")
                for line in record["disassembly"]:
                    marker = "=>" if line.startswith(
                        f"{location['elf_virtual_address']:x}:"
                    ) else "  "
                    print(f"              {marker} {line}")
            elif binaries["objdump"] is None:
                print("            Disassembly  : unavailable (Vita objdump not found)")
            elif record["disassembly_error"]:
                print(
                    f"            Disassembly  : unavailable "
                    f"({record['disassembly_error']})"
                )
    print(
        "    An image is associated only when its embedded Vita module name, "
        "PT_LOAD permissions, and segment extents agree with MODULE_INFO."
    )


def _kernel_state_text(item: dict) -> str:
    state = item["state"]
    if item["kind"] == "semaphore":
        return (
            f"count={state['current_count']}/{state['maximum_count']} "
            f"initial={state['initial_count']}"
        )
    if item["kind"] == "event-flag":
        return (
            f"pattern={_hex(state['current_pattern'])} "
            f"initial={_hex(state['initial_pattern'])}"
        )
    if item["kind"] in ("mutex", "lightweight-mutex"):
        return (
            f"count={state['current_count']} owner={_hex(item['owner_thread_uid'])}"
        )
    if item["kind"] == "condition-variable":
        return f"mutex={_hex(state['associated_mutex_uid'])}"
    if item["kind"] == "lightweight-condition-variable":
        return f"lwmutex={_hex(state['associated_lwmutex_uid'])}"
    if item["kind"] == "message-pipe":
        return f"free=0x{state['free_size']:x}/0x{state['buffer_size']:x}"
    return ""


def _print_kernel_objects(diagnosis: dict) -> None:
    kernel = diagnosis["kernel_objects"]
    print("\nKERNEL_SYNCHRONIZATION_OBJECTS:")
    print(f"    Tables                : {kernel['table_count']}")
    print(f"    Objects               : {kernel['object_count']}")
    print(f"    Active/contended      : {kernel['active_object_count']}")
    print(f"    Resolved thread waits : {kernel['resolved_thread_wait_count']}")
    if kernel["object_counts"]:
        print("    Counts by type        : " + ", ".join(
            f"{kind}={count}"
            for kind, count in sorted(kernel["object_counts"].items())
        ))

    print("\nTHREAD_WAIT_CHAINS:")
    waiting_threads = [
        thread
        for thread in diagnosis["execution_context"]["threads"]
        if thread["wait_type"] or thread["wait_id"]
    ]
    if not waiting_threads:
        print("    No retained thread wait state.")
    for thread in waiting_threads:
        item = thread["wait_object"]
        if item is None:
            print(
                f"    [{thread['index']:03d}] {thread['name'] or '<unnamed>'} "
                f"type={_hex(thread['wait_type'])} uid={_hex(thread['wait_id'])} "
                "-> unresolved or object class not captured"
            )
            continue
        print(
            f"    [{thread['index']:03d}] {thread['name'] or '<unnamed>'} "
            f"type={_hex(thread['wait_type'])} -> {item['kind']} "
            f"{_hex(item['uid'])} {item['name'] or '<unnamed>'}"
        )
        print(
            f"          {_kernel_state_text(item)}; "
            f"waiter-record={str(thread['waiter_record_present']).lower()}"
        )

    wait_graph_report = diagnosis.get("wait_graph") or {}
    cycles = wait_graph_report.get("cycles", [])
    if cycles:
        print("\nCAPTURED_WAIT_DEPENDENCY_CYCLES:")
        for cycle in cycles:
            labels = [
                node.get("name") or node["id"] for node in cycle["path"]
            ]
            print("    " + " -> ".join(labels))
    elif wait_graph_report:
        print("\nCAPTURED_WAIT_DEPENDENCY_CYCLES:")
        print("    None in retained wait and mutex-owner fields.")

    locks = [
        item
        for item in kernel["active_objects"]
        if item["kind"] in ("mutex", "lightweight-mutex")
        and item["owner_thread_uid"] is not None
    ]
    print("\nKERNEL_LOCK_OWNERSHIP:")
    if not locks:
        print("    No captured mutex has a nonzero owner UID.")
    for item in locks:
        owner = item["owner_thread"]
        owner_text = (
            f"[{owner['index']:03d}] {owner['name']}"
            if owner is not None
            else "thread not present in THREAD_INFO"
        )
        print(
            f"    {item['kind']} {_hex(item['uid'])} "
            f"{item['name'] or '<unnamed>'}: owner={_hex(item['owner_thread_uid'])} "
            f"({owner_text}), waiters={item['waiting_thread_count']}"
        )

    contended = [
        item for item in kernel["active_objects"] if item["waiting_thread_count"]
    ]
    print("\nCONTENDED_KERNEL_OBJECTS:")
    if not contended:
        print("    No captured synchronization object reports waiting threads.")
    for item in contended:
        print(
            f"    {item['kind']} {_hex(item['uid'])} "
            f"{item['name'] or '<unnamed>'}: waiters={item['waiting_thread_count']} "
            f"{_kernel_state_text(item)}"
        )
        for waiter in item["waiter_threads"]:
            identity = waiter["name"] or "thread not present in THREAD_INFO"
            index = (
                f"[{waiter['index']:03d}] " if waiter["index"] is not None else ""
            )
            print(
                f"          {index}{identity} uid={_hex(waiter['uid'])} "
                f"pid={_hex(waiter['process_id'])} "
                f"process={waiter['process_name'] or '<unresolved>'}"
            )


def _print_gpu_info(info: GpuInfo, database: RegisterDatabase) -> None:
    snapshot = info.snapshot_status()
    captured = [name for name, present in snapshot["sections"].items() if present]
    trailer = info.decoded_trailer()
    trailer_summary = trailer.summary()
    interpreted_selectors = sum(
        indexed_selector_metadata(item.offset) is not None
        for item in info.indexed_state
    )
    timestamp = info.capture_timestamp()
    clock_state = info.hardware_clock_state()
    print("\nGPU_INFO:")
    print(f"    Format version        : {info.format_version}")
    print(f"    Register dump version : {_hex(info.register_dump_version)}")
    print(f"    Capture flags         : {_hex(info.capture_flags)}")
    print(f"    Snapshot complete     : {str(snapshot['complete']).lower()}")
    print(
        "    Canonical 3.68 counts: "
        f"{str(snapshot['canonical_3_68_counts']).lower()}"
    )
    print(f"    Cores                 : {info.core_count}")
    print(
        f"    Register rows         : master={len(info.master_registers)} "
        f"per-core={len(info.per_core_registers)} indexed={len(info.indexed_state)}"
    )
    print(
        "    Indexed row semantics : "
        f"{interpreted_selectors} firmware-consumed, "
        f"{len(info.indexed_state) - interpreted_selectors} unassigned"
    )
    print(f"    Captured sections     : {', '.join(captured) or 'none'}")
    if timestamp["valid"]:
        print(
            f"    Capture local time    : {timestamp['iso_local']} "
            "(timezone not retained)"
        )
    else:
        print("    Capture local time    : invalid or unavailable")
    print(f"    SoC revision word     : {_hex(clock_state['soc_revision'])}")
    print(
        "    Clock state words     : "
        f"MP={_hex(clock_state['mp_clock_word'])} "
        f"core={_hex(clock_state['core_clock_word'])} "
        f"crossbar={_hex(clock_state['crossbar_clock_word'])}"
    )
    print("    Header words          : " + " ".join(f"{word:08x}" for word in info.header_words))
    print("    Time/environment words: " + " ".join(f"{word:08x}" for word in info.header_words[3:11]))
    print(f"    Trailer               : 0x{len(info.trailer):x} bytes")
    print(f"    Trailer SHA-256       : {hashlib.sha256(info.trailer).hexdigest()}")
    print(
        "    Producer extent       : "
        f"initialized=0x{trailer_summary['producer_initialized_size']:x} "
        f"exported=0x{trailer_summary['producer_export_size']:x}"
    )
    print(
        "    Producer padding      : "
        f"0x{trailer_summary['producer_padding_size']:x} bytes, "
        f"nonzero={trailer_summary['producer_padding_nonzero_bytes']} "
        "(excluded from GPU-state interpretation)"
    )
    print(
        "    Padding SHA-256       : "
        f"{trailer_summary['producer_padding_sha256']}"
    )
    print("    TA/3D control snapshot:")
    for item in trailer_summary["driver_state"]:
        print(
            f"      +0x{item['source_object_offset']:03x}="
            f"0x{item['value']:08x} ({item['name']})"
        )
    if not info.render_state()["available"]:
        print("      Render-state interpretation unavailable: missing capture flag or words.")
    print("\nREGISTER_DEFINITION_SOURCES:")
    print("    Built-in SGX543 diagnostic register set")
    for source in database.sources:
        print(f"    {source}")


def _print_fault_records(diagnosis: dict) -> None:
    faults = diagnosis["fault_sources"]
    groups = diagnosis["fault_groups"]
    print(f"\nFAULT_RECORD_COUNT: {len(faults)}")
    if not faults:
        print("\nFAULTING_GPU_ADDRESS: none retained")
        print("    BIF requestor/type latch clear; BIF fault-address registers clear.")
        return
    print(f"\nFAULT_GROUP_COUNT: {len(groups)}")
    for group in groups:
        print(f"\nFAULT_GROUP_{group['index']}:")
        print(f"    GPU virtual page       : {_hex(group['gpu_virtual_address'])}")
        print(f"    Observed scopes        : {', '.join(group['scopes'])}")
        print(f"    Source records         : {', '.join(str(value) for value in group['source_indices'])}")
        print(
            "    Requestor              : "
            + (", ".join(group["requestor_names"]) or "none")
        )
        _print_requestor_details(group["requestor_details"])
        print("    Fault condition        : " + (", ".join(group["fault_conditions"]) or "none"))
        print(f"    Page access            : {group['page_access_type']}")
        print(f"    MMU context            : {group['context_id']}")
        print(f"    Page directory         : {_hex(group['page_directory_base'])}")
        print(
            "    Sidebands              : "
            + (", ".join(str(value) for value in group["sidebands"]) or "none")
        )
        print(f"    Null GPU page          : {str(group['null_address']).lower()}")
        if group["multiple_scope_observation"]:
            print(
                "    Synthesis              : same page/requestor/context/directory "
                "fault tuple retained by multiple cores"
            )
        elif group["null_address"]:
            print("    Synthesis              : valid BIF request targets GPU page zero")
        else:
            print("    Synthesis              : one retained hardware fault observation")
    for index, fault in enumerate(faults):
        print(f"\nFAULT_RECORD_{index}:")
        print(f"    Scope                  : {fault['scope']}")
        print("    Arguments:")
        print(
            f"      Arg1: {fault['gpu_virtual_address']:08x}, "
            "faulting GPU virtual page address"
        )
        print(f"      Arg2: {fault['status']:08x}, BIF interrupt status")
        print(f"      Arg3: {fault['fault']:08x}, BIF fault register")
        print(f"      Arg4: {fault['context_id']:08x}, BIF MMU context ID")
        print(f"    Requestor bits         : 0x{fault['requestor']:04x}")
        print(
            "    Requestor              : "
            + (", ".join(fault["requestor_names"]) or "none")
        )
        _print_requestor_details(fault["requestor_details"])
        print(f"    Hardware fault type    : {fault['fault_type']}")
        print(f"    Raw fault type         : {fault['fault_type_raw']} (includes access selector)")
        print(f"    Page access            : {fault['page_access_type']}")
        print(
            "    Fault conditions       : "
            + (", ".join(fault["fault_conditions"]) or "none")
        )
        print(f"    Sideband               : {fault['sideband']}")
        print(f"    Status latch valid     : {str(fault['status_latched']).lower()}")
        print(f"    Null GPU page          : {str(fault['null_address']).lower()}")
        attribution = fault["context_attribution"]
        owner = attribution["owner"]
        if owner is not None:
            print(
                f"    Context owner          : {owner['process_name'] or '<unnamed>'} "
                f"(pid={_hex(owner['process_id'])}, slot={owner['slot']}, "
                f"map={attribution['status']})"
            )
        else:
            print(f"    Context owner          : unresolved ({attribution['status']})")
        directory = fault["page_directory_base"]
        print(f"    Page directory         : {_hex(directory)}")
        print(
            "    Directory in PT_LOAD   : "
            f"{str(fault['page_directory_bytes_captured']).lower()}"
        )
        references = fault["register_references"]
        if references:
            print("    Address register refs  :")
            for reference in references:
                name = reference["name"] or f"REG_{reference['offset']:04X}"
                location = reference["scope"]
                if reference["scope"] == "per-core":
                    location += " cores=" + ",".join(
                        str(core) for core in reference["cores"]
                    )
                    values = ",".join(_hex(value) for value in reference["values"])
                else:
                    values = _hex(reference["value"])
                fields = ",".join(reference["address_fields"])
                print(
                    f"      {name} ({location}, offset=0x{reference['offset']:04x}, "
                    f"field={fields}, value={values})"
                )
        capture = fault["address_capture_evidence"]
        print(f"    Dump address overlap    : {_address_capture_text(capture)}")
        print(
            f"    Bytes captured at VA    : 0x{capture['captured_bytes_at_address']:x}/0x10"
        )
        if fault["null_address"]:
            meaning = "GPU virtual page is zero."
        elif references:
            meaning = (
                "Fault page matches captured GPU address-register state. Correlation only; "
                "allocation ownership is not established."
            )
        else:
            meaning = "No captured address register contains the same 4 KiB page."
        print(f"    Interpretation         : {meaning}")


def _print_requestor_details(details: list[dict]) -> None:
    if not details:
        return
    resource_classes = list(
        dict.fromkeys(item["resource_class"] for item in details)
    )
    print(f"    Resource class         : {', '.join(resource_classes)}")
    for item in details:
        print(
            f"    Request path           : bit {item['bit']} "
            f"({item['code']}): {item['interpretation']}"
        )
        if item.get("subunit_name"):
            print(f"    Requestor subunit      : {item['subunit_id']} ({item['subunit_name']})")


def _print_bif_register_state(info: GpuInfo) -> None:
    bif = info.bif_diagnosis()
    print("\nBIF_REGISTER_STATE:")
    print(
        f"    master: INT_STAT={_hex(bif['master_status'])} "
        f"FAULT={_hex(bif['master_fault'])}"
    )
    statuses = bif["core_status"] or []
    faults = bif["core_fault"] or []
    for index in range(info.core_count):
        status = statuses[index] if index < len(statuses) else None
        fault = faults[index] if index < len(faults) else None
        print(f"    core{index}:  INT_STAT={_hex(status)} FAULT={_hex(fault)}")


def _register_field_text(
    database: RegisterDatabase, offset: int, master: bool, value: int
) -> str:
    fields = decoded_fields(database.definition(offset, master=master), value)
    return ", ".join(_decoded_field_text(field) for field in fields)


def _decoded_field_text(field: dict) -> str:
    # Shifted address bitfields are encoded in alignment units.  Human output
    # should show the address represented by the register, not that encoded
    # integer; JSON retains both values.
    value = field.get("address", field["value"])
    return f"{field['name']}={value:#x}"


def _print_pipeline_register_state(
    info: GpuInfo,
    database: RegisterDatabase,
    cores: Optional[Sequence[int]] = None,
) -> None:
    master_offsets = (0x4118, 0x412C, 0x4348, 0x4400, 0x4758)
    core_offsets = (0x0118, 0x012C, 0x0348, 0x0400, 0x0758)
    print("\nPIPELINE_REGISTER_STATE:")
    for offset in master_offsets:
        value = info.master_value(offset)
        if value is None:
            continue
        definition = database.definition(offset, master=True)
        name = definition.display_name if definition else f"REG_{offset:04X}"
        fields = _register_field_text(database, offset, True, value)
        suffix = f" [{fields}]" if fields else ""
        print(f"    {name:<40} = {_hex(value)}{suffix}")
    for offset in core_offsets:
        values = info.core_values(offset)
        if values is None:
            continue
        definition = database.definition(offset, master=False)
        name = definition.display_name if definition else f"REG_{offset:04X}"
        print(f"    {name}:")
        for index, value in enumerate(values):
            if cores is not None and index not in cores:
                continue
            fields = _register_field_text(database, offset, False, value)
            suffix = f" [{fields}]" if fields else ""
            print(f"      core{index}={_hex(value)}{suffix}")


def _print_render_state(pipeline: dict) -> None:
    state = pipeline["render_state"]
    if not state["available"]:
        print("    Frame/scene state     : unavailable (capture flag or snapshot words missing)")
        return
    for phase in ("vertex", "fragment"):
        label = f"{phase.capitalize()} frame/scene"
        print(f"    {label:22}: frame={state[phase + '_frame']} scene={state[phase + '_scene']}")
    print("    Retained counters; not completion fences. Differences alone do not establish a stall.")


def _print_clipper_halt_check(pipeline: dict, cores: Optional[Sequence[int]] = None) -> None:
    check = pipeline["clipper_halt_check"]
    print(f"    Clipper halt signature: {check['status']}")
    for item in check["matches"]:
        if cores is not None and not _gpu_scope_selected(item["scope"], cores):
            print(f"        Match on {item['scope']} is outside the selected cores.")
            continue
        print(
            f"        {item['scope']}: selector=0x{item['selector']:02x} "
            f"value={_hex(item['raw_value'])} mask={_hex(item['mask'])} "
            f"draw={item['draw_number']} counter={item['draw_counter']} ctx={item['context_id']}"
        )
        owner = item.get("context_attribution", {}).get("owner")
        if owner is not None:
            print(f"        TA process: {owner['process_name'] or '<unnamed>'} pid={_hex(owner['process_id'])}")
        print(f"        {item['qualification']}")


def _print_firmware_pipeline_diagnostics(
    diagnosis: dict, cores: Optional[Sequence[int]] = None
) -> None:
    pipeline = diagnosis["firmware_pipeline"]
    master = pipeline["master"]
    print("\nFIRMWARE_PIPELINE_DIAGNOSTICS:")
    print(f"    Decoder               : {pipeline['decoder']}")
    _print_render_state(pipeline)
    _print_clipper_halt_check(pipeline, cores)
    print(
        f"    Phase-active cores    : {pipeline['active_core_count']} "
        f"({','.join(str(item) for item in pipeline['active_cores']) or 'none'})"
    )
    print(f"    Phase-active units    : {pipeline['phase_active_unit_count']}")
    print(f"    All low-active reports: {pipeline['active_unit_count']}")
    print(
        "    Master state          : "
        f"transfer={str(master['transfer_active']).lower()} "
        f"partial-render={str(master['partial_render_active']).lower()} "
        f"partial-render-deadlock={str(master['partial_render_deadlock']).lower()}"
    )
    if cores is not None:
        print(
            "    Selected cores        : "
            + (",".join(str(item) for item in cores) or "none")
        )
    for core in pipeline["cores"]:
        if cores is not None and core["core"] not in cores:
            continue
        phases = ",".join(core["active_phases"]) or "none"
        units = ", ".join(item["name"] for item in core["active_units"]) or "none"
        print(
            f"\n    CORE_{core['core']}: phase-selector={_hex(core['phase_value'])} "
            f"unit-selector={_hex(core['unit_value'])}"
        )
        print(f"        Active phases     : {phases}")
        print(f"        Active units      : {units}")
        for pipe in core["usse_data_master_usage"]:
            print(
                f"        USSE pipe {pipe['pipe']}       : {pipe['usage']} "
                f"raw={_hex(pipe['raw_value'])}"
            )
        if not core["active_phases"]:
            print("        No vertex/fragment activity bit was retained for this core.")
            if not any(pipe["counts"]["vertex"] or pipe["counts"]["fragment"]
                       for pipe in core["usse_data_master_usage"]):
                continue

        counters = core["draw_counters"]
        if counters:
            print("        Draw counters     :")
            for base in range(0, len(counters), 4):
                print(
                    "            "
                    + "  ".join(
                        f"{item['name']}={item['draw_number']}"
                        + (f"(counter={item['counter_value']})" if item["encoding"] == "zero-or-draw-plus-one" else "")
                        + f" [{item['applicability']}]"
                        for item in counters[base : base + 4]
                    )
                )
            print("        Counter checks    : applicable raw counter fields only")
            for check in core["draw_counter_checks"]:
                values = ", ".join(
                    f"{name}={value}" for name, value in check["values"].items()
                )
                print(
                    f"            {check['name']}: "
                    f"{check['status']} ({values or 'fewer than two active fields'})"
                )
                if check["excluded"]:
                    print("                Excluded: " + ", ".join(
                        f"{name}={state}" for name, state in check["excluded"].items()))

        tiles = core["tile_progress"]
        if tiles:
            print("        Tile progress     :")
            for base in range(0, len(tiles), 3):
                print(
                    "            "
                    + "  ".join(
                        f"{item['name']}=({item['tile_x']},{item['tile_y']})"
                        f"/{item['pixel_x']},{item['pixel_y']}px"
                        f" [{item['applicability']}]"
                        for item in tiles[base : base + 3]
                    )
                )
            for check in core["tile_checks"]:
                values = ", ".join(
                    f"({value[0]},{value[1]})" for value in check["values"]
                )
                print(
                    f"            {check['name']}: "
                    f"{check['status']} ({values or 'fewer than two active fields'})"
                )
                if check["excluded"]:
                    print("                Excluded: " + ", ".join(
                        f"{name}={state}" for name, state in check["excluded"].items()))

        progress = core["isp_progress"]
        if progress is not None:
            print(
                "        ISP progress      : "
                f"primitive={progress['primitive_number']} "
                f"tile=({progress['tile_x']},{progress['tile_y']}) "
                f"pixel=({progress['pixel_x']},{progress['pixel_y']}) "
                f"raw={_hex(progress['register_value'])}"
            )
        bounds = core["render_bounds"]
        if bounds is not None:
            print(
                "        Render bounds     : "
                f"max=({bounds['maximum_pixel_x']},{bounds['maximum_pixel_y']}) "
                f"extent={bounds['width_pixels']}x{bounds['height_pixels']} "
                f"raw={_hex(bounds['register_value'])}"
            )
        print(
            f"        Region headers    : {_hex(core['region_header_base'])} "
            f"raw={_hex(core['region_header_base_raw'])}"
        )
        print(
            "        Region mapping    : "
            + _address_capture_text(core["region_header_evidence"])
        )
        zls = core["zls"]
        if zls is not None:
            print(
                "        Depth/stencil     : "
                f"external={str(zls['external_depth_buffer']).lower()} "
                f"width={zls['width_pixels'] if zls['width_pixels'] is not None else 'n/a'}px "
                f"samples={zls['width_samples'] if zls['width_samples'] is not None else 'n/a'} "
                f"msaa={zls['msaa']['mode']} "
                f"flags={','.join(zls['flags']) or 'none'} "
                f"raw={_hex(zls['control'])}"
            )
            enabled = [
                (name, surface)
                for name, surface in zls["surfaces"].items()
                if surface["enabled"]
            ]
            if not enabled:
                print("        Enabled surfaces  : none")
            else:
                print("        Enabled surfaces  :")
                for name, surface in enabled:
                    print(
                        f"            {name}={_hex(surface['address'])} "
                        f"[{_address_capture_text(surface['address_evidence'])}]"
                    )
    print("\n    Draw numbers are 13-bit retained counters; equality does not imply completion.")
    print("    MTE/VDM display zero or counter-1; ISP/TSP display the counter directly.")
    print("    USSE slots: V=vertex, P=fragment, .=idle, S=service; slot 0 is printed first.")
    print("    Tile positions are progress markers in 16x16-pixel units, not fault locations.")
    print("    Inactive/unknown fields are retained but excluded from comparisons.")
    print("    Divergence is a snapshot difference, not proof of a stall or invalid submission.")
    print("    MEM_BLK matches are direct address overlaps; they do not prove GPU-MMU ownership.")


def _print_firmware_pipeline_summary(
    diagnosis: dict, cores: Sequence[int]
) -> None:
    pipeline = diagnosis["firmware_pipeline"]
    master = pipeline["master"]
    print("\nFIRMWARE_PIPELINE:")
    print(f"    Decoder               : {pipeline['decoder']}")
    _print_render_state(pipeline)
    _print_clipper_halt_check(pipeline, cores)
    print(
        "    Master                : "
        f"transfer={str(master['transfer_active']).lower()} "
        f"partial-render={str(master['partial_render_active']).lower()} "
        f"deadlock={str(master['partial_render_deadlock']).lower()}"
    )
    for core in pipeline["cores"]:
        if core["core"] not in cores:
            continue
        phases = ",".join(core["active_phases"]) or "none"
        units = ",".join(item["name"] for item in core["active_units"]) or "none"
        print(
            f"    core{core['core']}: phases={phases} units={units} "
            f"phase-selector={_hex(core['phase_value'])} "
            f"unit-selector={_hex(core['unit_value'])}"
        )
        diverged = [
            item["name"]
            for item in core["draw_counter_checks"] + core["tile_checks"]
            if item["status"] == "diverged"
        ]
        progress = core["isp_progress"]
        bounds = core["render_bounds"]
        details = []
        if progress is not None:
            details.append(
                f"isp=primitive:{progress['primitive_number']} "
                f"tile:{progress['tile_x']},{progress['tile_y']} "
                f"pixel:{progress['pixel_x']},{progress['pixel_y']}"
            )
        if bounds is not None:
            details.append(
                f"bounds={bounds['width_pixels']}x{bounds['height_pixels']}"
            )
        details.append(f"region-headers={_hex(core['region_header_base'])}")
        details.append(f"diverged={','.join(diverged) or 'none'}")
        print("        " + "  ".join(details))
    print("    Use --details for retained counters, tiles, surfaces, and register fields.")


def _compact_firmware_pipeline(pipeline: dict, cores: Sequence[int]) -> dict:
    return {
        "decoder": pipeline["decoder"],
        "master": pipeline["master"],
        "render_state": pipeline["render_state"],
        "clipper_halt_check": pipeline["clipper_halt_check"],
        "selected_cores": list(cores),
        "cores": [
            {
                key: core[key]
                for key in (
                    "core",
                    "phase_value",
                    "unit_value",
                    "active_phases",
                    "active_units",
                    "usse_data_master_usage",
                    "draw_counter_checks",
                    "tile_checks",
                    "isp_progress",
                    "render_bounds",
                    "region_header_base",
                    "region_header_base_raw",
                    "region_header_evidence",
                    "zls",
                )
            }
            for core in pipeline["cores"]
            if core["core"] in cores
        ],
    }


def _print_hardware_signals(diagnosis: dict) -> None:
    signals = diagnosis["event_signals"]
    print("\nHARDWARE_ERROR_SIGNALS:")
    if not signals:
        print("    No decoded DPM, ISP, or breakpoint error latch is set.")
        return
    for signal in signals:
        print(
            f"    {signal['severity'].upper():7} {signal['scope']:6} "
            f"{signal['code']} value={_hex(signal['register_value'])} "
            f"mask={_hex(signal['mask'])}"
        )
        print(f"            {signal['description']}")


def _gpu_scope_selected(scope: str, cores: Optional[Sequence[int]]) -> bool:
    if cores is None or not scope.startswith("core"):
        return True
    try:
        return int(scope[4:]) in cores
    except ValueError:
        return True


def _print_lockup_state(
    diagnosis: dict, cores: Optional[Sequence[int]] = None
) -> None:
    print("\nLOCKUP_PHASES:")
    phases = [
        item
        for item in diagnosis["lockup_phases"]
        if _gpu_scope_selected(item["scope"], cores)
    ]
    if not phases:
        print("    No TA/vertex or 3D/fragment phase latch is retained.")
    for phase in phases:
        print(
            f"    {phase['code'].upper():8} {phase['description']} "
            f"value={_hex(phase['register_value'])} mask={_hex(phase['mask'])}"
        )
    print("\nPIPELINE_COMPLETION_MARKERS:")
    markers = [
        item
        for item in diagnosis["pipeline_markers"]
        if _gpu_scope_selected(item["scope"], cores)
    ]
    if not markers:
        print("    No decoded completion or partial-render marker is set.")
    for marker in markers:
        print(
            f"    {marker['scope']:6} {marker['code']} "
            f"value={_hex(marker['register_value'])} mask={_hex(marker['mask'])}"
        )
        print(f"            {marker['description']}")


def _print_context_table(info: GpuInfo, diagnosis: dict) -> None:
    trailer = info.decoded_trailer()
    print("\nGPU_CONTEXT_OWNERS:")
    if not diagnosis["context_owners"]:
        print("    No populated owner records.")
    for owner in diagnosis["context_owners"]:
        print(
            f"    Slot {owner['slot']}: pid={_hex(owner['process_id'])} "
            f"name={owner['process_name'] or '<unnamed>'}"
        )
    print("\nGPU_CONTEXT_ASSIGNMENTS:")
    if not diagnosis["context_scope_assignments"]:
        print("    No context ID could be resolved in a captured scope.")
    for assignment in diagnosis["context_scope_assignments"]:
        owner = assignment["owner"]
        print(
            f"    {assignment['scope']:6} CID {assignment['context_id']} -> slot {owner['slot']} "
            f"{owner['process_name'] or '<unnamed>'} (map={assignment['status']})"
        )
    print("\nGPU_CONTEXT_MAP_RAW:")
    for index, row in enumerate(trailer.context_slot_map):
        print(f"    Row {index} ({GPU_CONTEXT_MAP_SCOPES[index]:6}): " + " ".join(f"{value:02x}" for value in row))
    print("    Values are owner-slot indices; ff means unused. Fault attribution uses only its scope's row.")


def _print_capture_evidence(
    core: CoreDump,
    activity: Optional[GpuActivityInfo],
    diagnosis: dict,
) -> None:
    memory = diagnosis["memory_evidence"]
    summary = diagnosis["summary_integrity"]
    aliases = core.load_alias_groups()
    print("\nMEMORY_CAPTURE:")
    print(f"    PT_LOAD segments       : {memory['load_segment_count']}")
    print(f"    Unique segments        : {memory['unique_load_segment_count']}")
    print(f"    Unique captured bytes  : 0x{memory['unique_load_bytes']:x}")
    print(f"    Exact alias groups     : {memory['alias_group_count']}")
    if aliases:
        shown_aliases = aliases[:8]
        print(
            "    Alias program headers  : "
            + "; ".join(
                ",".join(str(index) for index in group) for group in shown_aliases
            )
        )
        if len(aliases) > len(shown_aliases):
            print(
                f"    Alias groups omitted   : {len(aliases) - len(shown_aliases)} "
                "(use the segments command for the complete map)"
            )
    print(f"    Inventoried blocks     : {memory['memory_block_count']}")
    print(
        "    Blocks with dump bytes: "
        f"{memory['memory_blocks_with_captured_bytes']}"
    )
    print(f"    Bytes inside blocks    : 0x{memory['memory_block_captured_bytes']:x}")
    print("\nDUMP_COMPLETENESS:")
    snapshot = diagnosis["snapshot"]
    print(f"    GPU capture complete   : {str(snapshot['complete']).lower()}")
    print(f"    Missing GPU sections   : {', '.join(snapshot['missing_sections']) or 'none'}")
    print(f"    Unknown capture flags  : {_hex(snapshot['unknown_flags'])}")
    if summary is None:
        print("    SUMMARY_INFO           : unavailable")
    else:
        print(f"    SUMMARY_INFO version   : {summary['format_version']}")
        print(f"    Planned note entries   : {summary['entry_count']}")
        print(f"    Incomplete note entries: {summary['incomplete_count']}")
        for entry in summary["entries"]:
            if entry["status"] not in ("complete", "not-present", "self-reference"):
                print(
                    f"      type={entry['note_type']:#x} name={entry['name']} "
                    f"planned={entry['planned_size']} written={entry['written_size']} "
                    f"status={entry['status']}"
                )
    print("\nBLACKBOXGPUACTIVITY:")
    if activity is None:
        print("    GPU_ACT_INFO unavailable")
    else:
        print(f"    Format version : {activity.format_version}")
        print(f"    Sequence       : {activity.sequence}")
        print(f"    Payload bytes  : 0x{activity.payload_size:x}")
        print(f"    Trailing bytes : 0x{len(activity.trailing):x}")
        print(
            "    Activity ring  : "
            f"{'retained' if activity.activity_buffer is not None else 'not retained'}"
        )


def _print_resource_pointers(diagnosis: dict) -> None:
    evidence = diagnosis["resource_pointers"]
    print("\nGPU_RETAINED_RESOURCE_POINTERS:")
    print(f"    Selection             : {evidence['selection']}")
    print(f"    Pointer groups        : {evidence['pointer_count']}")
    print(f"    Register observations : {evidence['observation_count']}")
    print(f"    Allocation matches    : {evidence['allocation_match_count']}")
    if not evidence["pointers"]:
        print("    No selected nonzero resource pointer was retained.")
        return
    for pointer in evidence["pointers"]:
        locations = []
        if pointer["master_observed"]:
            locations.append("master")
        if pointer["core_indices"]:
            locations.append(
                "cores=" + ",".join(str(index) for index in pointer["core_indices"])
            )
        print(
            f"    [{pointer['index']:02d}] {pointer['name']} ({pointer['stage']}) "
            f"address={_hex(pointer['address'])} observed={'; '.join(locations)}"
        )
        allocation = pointer["allocation"]
        if allocation is None:
            print("         Allocation       : no direct MEM_BLK_INFO overlap")
        else:
            print(
                f"         Allocation       : [{allocation['index']}] "
                f"{allocation['name'] or '<unnamed>'}+0x{allocation['offset']:x} "
                f"range={allocation['base']:08x}-{allocation['end']:08x}"
            )
            print(
                f"         Captured bytes   : pointer=0x{pointer['captured_bytes_at_address']:x}/0x10 "
                f"allocation=0x{allocation['captured_bytes']:x}/0x{allocation['size']:x}"
            )
        print("         Registers        :")
        for observation in pointer["observations"]:
            location = (
                "master"
                if observation["scope"] == "master"
                else f"core{observation['core']}"
            )
            print(
                f"             {location} offset=0x{observation['register_offset']:04x} "
                f"raw={_hex(observation['raw_value'])} "
                f"mask={_hex(observation['address_mask'])}"
            )
    print(
        "    Allocation matches are direct numeric overlaps; they do not prove "
        "GPU-MMU ownership or retain the pointed-to bytes."
    )


def _print_allocation_evidence(diagnosis: dict) -> None:
    evidence = diagnosis["allocation_evidence"]
    print("\nGPU_NAMED_ALLOCATIONS:")
    print(f"    Selection             : {evidence['selection']}")
    print(f"    Allocation count      : {evidence['allocation_count']}")
    print(f"    Total virtual size    : 0x{evidence['total_size']:x}")
    print(f"    Total committed size  : 0x{evidence['total_committed_size']:x}")
    if evidence["category_bytes"]:
        print("    Categories            :")
        for category in sorted(evidence["category_bytes"]):
            print(
                f"        {category:<20} count={evidence['category_counts'][category]:3d} "
                f"bytes=0x{evidence['category_bytes'][category]:x}"
            )
    if not evidence["allocations"]:
        print("    No MEM_BLK_INFO name contains a known GPU/GXM component token.")
        return
    print("    Ranges                :")
    for allocation in evidence["allocations"]:
        print(
            f"        {allocation['base']:08x}-{allocation['end']:08x} "
            f"size=0x{allocation['size']:x} committed=0x{allocation['committed_size']:x} "
            f"captured=0x{allocation['captured_bytes']:x} "
            f"[{','.join(allocation['categories'])}] {allocation['name']}"
        )
    print(
        "    Name selection does not establish GPU-VA ownership or fault-address overlap."
    )


def _print_allocation_summary(diagnosis: dict) -> None:
    evidence = diagnosis["allocation_evidence"]
    print("\nGPU_NAMED_ALLOCATION_SUMMARY:")
    print(
        f"    count={evidence['allocation_count']} "
        f"virtual=0x{evidence['total_size']:x} "
        f"committed=0x{evidence['total_committed_size']:x}"
    )
    if evidence["category_bytes"]:
        print(
            "    categories: "
            + ", ".join(
                f"{category}={evidence['category_counts'][category]}/"
                f"0x{evidence['category_bytes'][category]:x}"
                for category in sorted(evidence["category_bytes"])
            )
        )
    print("    Use --allocations to list every selected allocation range.")


def _print_log_evidence(diagnosis: dict, line_limit: int) -> None:
    log = diagnosis["log"]
    print("\nBLACKBOXTTY:")
    print(f"    Available : {str(log['available']).lower()}")
    print(f"    Note      : {log['note'] or 'none'}")
    print(f"    Lines     : {log['line_count']}")
    if not log["available"]:
        return
    timeline = log["timeline"]
    print("\nTTY_TIMELINE:")
    print(f"    Timestamped lines    : {timeline['timestamped_line_count']}")
    print(f"    Untimestamped lines  : {timeline['untimestamped_line_count']}")
    print(
        "    Leading partial line : "
        f"{str(timeline['first_line_untimestamped']).lower()}"
    )
    if timeline["first_timestamp"] is None:
        print("    Timestamp range      : unavailable")
    else:
        print(
            f"    Timestamp range      : {timeline['first_timestamp']:.4f} .. "
            f"{timeline['last_timestamp']:.4f}"
        )
        print(f"    Captured span        : {timeline['captured_span']:.4f} s")
    print(f"    Timestamp regressions: {len(timeline['timestamp_regressions'])}")
    gap = timeline["largest_nonnegative_gap"]
    if gap is not None:
        print(
            f"    Largest observed gap : {gap['delta']:.4f} s "
            f"(line {gap['from_line']} -> {gap['to_line']})"
        )
    last = timeline["last_timestamped_line"]
    if last is not None:
        print(f"    Last stamped line    : {_trim_display_line(last['text'])}")
    print("    Timestamps are TTY producer timestamps, not the GPU fault time.")
    if line_limit <= 0:
        return
    noteworthy = log["noteworthy_lines"][-line_limit:]
    recent = [line for line in log["recent_lines"] if line not in noteworthy][
        -line_limit:
    ]
    if noteworthy:
        print("\nTTY_LOG_CLUES:")
        for line in noteworthy:
            print(f"    {line}")
    if recent:
        print("\nTTY_LOG_TAIL:")
        for line in recent:
            print(f"    {line}")


def _print_failure_identity(diagnosis: dict) -> None:
    fault_owners = []
    for fault in diagnosis["fault_sources"]:
        owner = fault["context_attribution"]["owner"]
        fault_owners.append(owner)
    resolved_owners = []
    for owner in fault_owners:
        if owner is not None and owner not in resolved_owners:
            resolved_owners.append(owner)
    fully_attributed = bool(fault_owners) and all(owner is not None for owner in fault_owners)
    if fully_attributed and len(resolved_owners) == 1:
        owner_name = resolved_owners[0]["process_name"]
        owner_id = _hex(resolved_owners[0]["process_id"])
    else:
        owner_name = "<unresolved-or-multiple>"
        owner_id = "unresolved"
    process = diagnosis["execution_context"]["process"] or {}
    process_name = process.get("process_name") or "<unavailable>"
    process_id = _hex(process.get("process_id"))
    scopes = ",".join(
        scope for group in diagnosis["fault_groups"] for scope in group["scopes"]
    ) or "none"
    addresses = ",".join(
        _hex(group["gpu_virtual_address"]) for group in diagnosis["fault_groups"]
    ) or "none"
    print(f"\nPROCESS_NAME:  {process_name}")
    print(f"PROCESS_ID:    {process_id}")
    print(f"GPU_CONTEXT_OWNER_NAME: {owner_name}")
    print(f"GPU_CONTEXT_OWNER_ID:   {owner_id}")
    print(f"FAULTING_GPU_SCOPE: {scopes}")
    print(f"FAULTING_GPU_VA:    {addresses}")
    print(f"FAILURE_BUCKET_ID:  {diagnosis['failure_bucket_id']}")
    print(f"FAILURE_HASH:       {diagnosis['failure_hash']}")


def _register_value_text(value) -> str:
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_hex(item) for item in value) + "]"
    return _hex(value)


def _print_devkit_gpu_dump(report: dict, include_registers: bool = False) -> None:
    header = report["header"]
    print("\nDEVKIT_GPU_DUMP:")
    print(f"    File                  : {report['path']}")
    print(f"    Size                  : {report['size']} bytes")
    print(f"    SHA256                : {report['sha256']}")
    print(f"    Format version        : {_hex(header['format_version'])}")
    print(f"    SDK version           : {_hex(header['sdk_version'])}")
    print(f"    GPU driver build      : {header['gpu_driver_build']} ({_hex(header['gpu_driver_build'])})")
    print(f"    SoC revision          : {_hex(header['soc_revision'])}")
    print(f"    GPU cores             : {report['core_count']}")
    print("\n    Segments:")
    for segment in report["segments"]:
        count = f" records={segment['record_count']}" if "record_count" in segment else ""
        print(
            f"        {segment['tag']:08x} {segment['name']:14} "
            f"offset=0x{segment['file_offset']:04x} size=0x{segment['stored_size']:x} "
            f"{segment['encoding']}{count}"
        )
    if report["missing_segments"]:
        print("    Missing segments      : " + ", ".join(report["missing_segments"]))
    print("\n    Data-master processes:")
    for process in report["processes"]:
        if process["populated"]:
            print(
                f"        DM {process['data_master']}: pid={_hex(process['process_id'])} "
                f"{process['process_name'] or '<unnamed>'}"
            )
        else:
            print(f"        DM {process['data_master']}: unresolved")
    print(f"    {report['process_table_scope']}")
    if report["render_state"] is not None:
        render = report["render_state"]
        print("\n    Render state:")
        for phase in ("vertex", "fragment"):
            print(f"        {phase}: frame={render[phase + '_frame']} scene={render[phase + '_scene']}")
    print("\n    Sidecar BIF faults:")
    if not report["bif"]["fault_sources"]:
        print("        No retained BIF fault record.")
    for fault in report["bif"]["fault_sources"]:
        print(
            f"        {fault['scope']}: INT_STAT={_hex(fault['status'])} "
            f"FAULT={_hex(fault['fault'])} page={_hex(fault['address'])} ctx={fault['context_id']}"
        )
        print(
            f"            {', '.join(fault['fault_conditions']) or 'no fault-type latch'}; "
            f"{fault['page_access_type']}; latched={str(fault['status_latched']).lower()}"
        )
        for detail in fault["requestor_details"]:
            print(f"            {detail['name']}: subunit={detail['subunit_id']} ({detail['subunit_name'] or 'undecoded'})")
    print("\n    USSE data-master slots (V=vertex, P=fragment, .=idle, S=service):")
    for core in report["firmware_pipeline"]["cores"]:
        for pipe in core["usse_data_master_usage"]:
            print(
                f"        core{core['core']} pipe{pipe['pipe']}: {pipe['usage']} "
                f"raw={_hex(pipe['raw_value'])}"
            )
    correlation = report.get("correlation")
    if correlation is not None:
        print("\n    GPU_INFO correlation:")
        print(f"        Core dump         : {correlation['core_path']}")
        print(f"        Result            : {correlation['status']}")
        print(f"        Fault signature   : {correlation['fault_signature_match']}")
        print(f"        SoC / core count  : {correlation['soc_revision_match']} / {correlation['core_count_match']}")
        print(f"        Matching processes: {len(correlation['matching_processes'])}")
        for conflict in correlation["conflicting_processes"]:
            print(f"        Process conflict  : pid={_hex(conflict['process_id'])} sidecar={conflict['process_name']} GPU_INFO={conflict['gpu_info_name']}")
        for check in correlation["fault_process_checks"]:
            print(
                f"        {check['scope']} fault owner : sidecar={check['gpu_dump_name']} "
                f"GPU_INFO={check['gpu_info_name'] or 'unresolved'} ({check['status']})"
            )
        for name, comparison in correlation["register_comparisons"].items():
            print(
                f"        {name:18}: {comparison['matching_count']}/{comparison['shared_count']} shared offsets equal; "
                f"{len(comparison['changed'])} changed; "
                f"sidecar-only={len(comparison['gpu_dump_only'])} GPU_INFO-only={len(comparison['gpu_info_only'])}"
            )
            for change in comparison["changed"]:
                print(
                    f"            0x{change['offset']:04x}: sidecar={_register_value_text(change['gpu_dump'])} "
                    f"GPU_INFO={_register_value_text(change['gpu_info'])}"
                )
            for key, label in (("gpu_dump_only", "Sidecar only"), ("gpu_info_only", "GPU_INFO only")):
                if comparison[key]:
                    print(f"            {label}: " + ", ".join(f"0x{item['offset']:04x}" for item in comparison[key]))
        hint = correlation["filename_timestamp_hint"]
        if hint is not None:
            print(f"        Filename time     : {hint['iso_local']}; delta={hint['delta_from_gpu_info_microseconds']} us")
            print("        Filename time is a naming hint, not an embedded event identifier.")
        print(f"        {correlation['qualification']}")
        if correlation["status"] == "mismatch":
            print("        WARNING: supplied captures disagree; sidecar state is not evidence for this core dump.")
    if include_registers:
        print("\n    Register tables (sidecar snapshot):")
        for name, records in report["register_tables"].items():
            print(f"        {name}:")
            for record in records:
                offset = record.get("offset", record.get("selector"))
                value = record["value"] if "value" in record else record["values"]
                print(f"            0x{offset:04x} {record['name'] or '-'} {_register_value_text(value)}")
    print("\nFIRMWARE_ANALYSIS_TEXT:")
    if report["analysis_text"]:
        print(report["analysis_text"], end="")
    else:
        print("    Embedded firmware analysis text is absent or empty.")
    print("\nDEVKIT_GPU_DUMP_LIMITATIONS:")
    for limitation in report["limitations"]:
        print(f"    {limitation}")


def command_gpu_dump(args: argparse.Namespace) -> int:
    native = DevkitGpuDump.read(args.file)
    report = native.summary(_database(args))
    if args.core_dump:
        core = CoreDump.read(args.core_dump)
        info = _gpu_info(core)
        if info is None:
            raise ParseError("GPU_INFO is unavailable in the supplied core dump; cannot correlate")
        report["correlation"] = correlate_gpu_dump(native, core, info)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    elif args.text:
        print(report["analysis_text"], end="")
    else:
        _print_devkit_gpu_dump(report, args.registers)
    return 0


def _print_diagnosis(
    core: CoreDump,
    info: GpuInfo,
    database: RegisterDatabase,
    activity: Optional[GpuActivityInfo],
    diagnosis: dict,
    args: argparse.Namespace,
) -> None:
    _print_analysis_banner()
    print(f"\n{diagnosis['analysis_code']}\n")
    print(diagnosis["headline"] + ".")
    print(diagnosis["explanation"])
    print(
        f"\nCLASSIFICATION_CONFIDENCE: "
        f"{diagnosis['classification_confidence'].upper()}"
    )
    print(f"ROOT_CAUSE_CONFIDENCE: {diagnosis['root_cause_confidence'].upper()}\n")
    _print_analysis_summary(core, info, diagnosis)
    _print_dump_attributes(core)
    _print_note_coverage(diagnosis)
    _print_supporting_context(diagnosis)
    _print_execution_context(diagnosis)
    _print_binary_images(diagnosis)
    _print_kernel_objects(diagnosis)
    _print_gpu_info(info, database)
    _print_fault_records(diagnosis)
    _print_bif_register_state(info)
    _print_firmware_pipeline_diagnostics(diagnosis)
    _print_pipeline_register_state(info, database)
    _print_hardware_signals(diagnosis)
    _print_lockup_state(diagnosis)
    _print_context_table(info, diagnosis)
    _print_resource_pointers(diagnosis)
    _print_allocation_evidence(diagnosis)
    _print_capture_evidence(core, activity, diagnosis)
    if diagnosis.get("gpu_dump") is not None:
        _print_devkit_gpu_dump(diagnosis["gpu_dump"])
    if not args.no_log:
        _print_log_evidence(diagnosis, args.log_lines)
    if diagnosis["limitations"]:
        print("\nANALYSIS_LIMITATIONS:")
        for limitation in diagnosis["limitations"]:
            print(f"    {limitation}")
    _print_failure_identity(diagnosis)
    print("\nFOLLOWUP_COMMANDS:")
    for command in diagnosis["recommended_commands"]:
        print(f"    {command}")


def command_analyze(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    database = _database(args)
    info = _gpu_info(core)
    result = core.summary()
    validation = validate_core(core)
    result["register_definition_sources"] = database.sources
    result["gpu_info"] = info.summary(database) if info else None
    activity = _gpu_activity(core)
    result["gpu_activity_info"] = activity.summary() if activity else None
    summary_note = core.note("SUMMARY_INFO")
    summary_info = SummaryInfo.parse(summary_note.description) if summary_note else None
    result["summary_info"] = summary_info.summary(core.notes) if summary_info else None
    diagnosis = (
        build_diagnosis(core, info, database, activity, summary_info)
        if info is not None
        else None
    )
    if args.gpu_dump:
        if info is None:
            raise ParseError("GPU_INFO is unavailable; use gpu-dump to inspect the sidecar on its own")
        native = DevkitGpuDump.read(args.gpu_dump)
        report = native.summary(database)
        report["correlation"] = correlate_gpu_dump(native, core, info)
        diagnosis["gpu_dump"] = report
        if report["correlation"]["status"] != "consistent":
            diagnosis["limitations"].append(
                "Supplied GPU sidecar correlation is " + report["correlation"]["status"]
                + "; its state must not be attributed to this GPUCRASH without independent confirmation."
            )
    if diagnosis is not None:
        diagnosis["validation"] = validation
        diagnosis["wait_graph"] = wait_graph(core, database, diagnosis)
        diagnosis["binary_images"] = (
            enrich_diagnosis_with_elfs(diagnosis, args.images, args.sdk)
            if args.images
            else {
                "requested_count": 0,
                "matched_count": 0,
                "rejected_count": 0,
                "invalid_count": 0,
                "objdump": None,
                "addr2line": None,
                "nid_database_sources": [],
                "images": [],
            }
        )
        binary_images = diagnosis["binary_images"]
        matched_images = [
            image
            for image in binary_images["images"]
            if image.get("association", {}).get("status") == "matched"
        ]
        if matched_images and all(
            image["symbol_count"] == 0 for image in matched_images
        ):
            diagnosis["limitations"].append(
                "Matched binary images are stripped. CPU bytes and SCE import "
                "NIDs are available; static function names and source lines are not."
            )
        if matched_images and binary_images["objdump"] is None:
            diagnosis["limitations"].append(
                "Matched binary images are available, but Vita objdump was not found. "
                "Instruction disassembly was not produced."
            )
        if binary_images["rejected_count"]:
            diagnosis["limitations"].append(
                f"{binary_images['rejected_count']} supplied binary image(s) failed module "
                "identity or segment-layout checks and were not used."
            )
        if not args.images:
            quoted = shlex.quote(str(core.path))
            diagnosis["recommended_commands"].append(
                f"psp2-gpucrash-parse analyze {quoted} --image ORIGINAL_MODULE.elf "
                "--sdk VITASDK"
            )
    result["diagnosis"] = diagnosis
    if args.json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    if info is None:
        _print_analysis_banner()
        print("\nGPU_INFO_UNAVAILABLE\n")
        print("GPU_INFO note missing or unsupported. Hardware-state analysis unavailable.")
        print("ELF container metadata remains valid for structural inspection.")
        _print_dump_attributes(core)
        return 2
    _print_diagnosis(core, info, database, activity, diagnosis, args)
    if args.memory_blocks:
        print("\nMEMORY_BLOCK_INVENTORY:")
        for block in core.memory_blocks():
            captured = core.captured_bytes(block.base, block.end)
            print(
                f"    {block.base:08x}-{block.end:08x} size=0x{block.size:x} "
                f"captured=0x{captured:x} uid={block.uid:08x} {block.name}"
            )
    if args.notes:
        print("\nNOTE_INVENTORY:")
        for note in core.notes:
            print(
                f"    ph={note.segment_index} name={note.name} "
                f"type={note.note_type:#x} bytes={len(note.description)} "
                f"sha256={hashlib.sha256(note.description).hexdigest()}"
            )
    return 0


def command_address(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    database = _database(args)
    report = address_report(core, database, args.address, args.bytes)
    if args.images:
        report["binary_images"] = enrich_address_with_elfs(
            core, report, args.images, args.sdk
        )
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0

    print(f"Address: 0x{report['address']:08x}")
    print(f"Status : {report['status']}")
    print(f"Result : {report['conclusion']}")
    for note in report["value_notes"]:
        print(f"Note   : {note}")
    module = report["module"]
    if module is not None:
        print(
            f"Module : {module['notation']} "
            f"({module['segment_permissions']})"
        )
    if report.get("binary_images"):
        for item in report["binary_images"]["resolutions"]:
            location = item["location"]
            symbol = location["symbol"]
            symbol_text = (
                f" {symbol['name']}+0x{symbol['offset']:x}" if symbol else ""
            )
            print(
                f"Image  : {location['image_path']} "
                f"runtime={_hex(report['address'])} "
                f"ELF={_hex(location['elf_virtual_address'])} "
                f"segment{location['segment_number']}+0x{location['segment_offset']:x}"
                f"{symbol_text}"
            )
            source = item.get("source")
            if source is not None:
                print(f"Source : {source['function']} at {source['file_line']}")
            if item["disassembly"]:
                print(
                    f"Decode : {item['instruction_set']} "
                    f"({item['instruction_set_basis']}); addresses below are ELF VAs"
                )
                for line in item["disassembly"]:
                    marker = "=>" if line.startswith(
                        f"{location['elf_virtual_address']:x}:"
                    ) else "  "
                    print(f"    {marker} {line}")
            elif item["disassembly_error"]:
                print(f"Decode : unavailable ({item['disassembly_error']})")
        if not report["binary_images"]["resolutions"]:
            print("Image  : no supplied image resolved this runtime address")
            for item in report["binary_images"]["images"]:
                association = item.get("association", {})
                reasons = association.get("reasons", [])
                reason = reasons[0] if reasons else item.get(
                    "error", item.get("address_status", item["status"])
                )
                print(f"         {item['path']}: {reason}")

    print("\nCAPTURED_RANGES:")
    if not report["load_segments"]:
        print("    none")
    for segment in report["load_segments"]:
        state = "bytes captured" if segment["captured_at_address"] else "memory extent only"
        print(
            f"    ph{segment['index']}+0x{segment['offset']:x} "
            f"{segment['virtual_address']:08x}-{segment['memory_end']:08x} ({state})"
        )

    print("\nMEMORY_BLOCKS:")
    if not report["memory_blocks"]:
        print("    none")
    for block in report["memory_blocks"]:
        print(
            f"    [{block['index']}] {block['base']:08x}-{block['end']:08x} "
            f"+0x{block['offset']:x} captured=0x{block['captured_bytes']:x} "
            f"{block['name'] or '<unnamed>'}"
        )

    print("\nTHREAD_REFERENCES:")
    if not report["threads"]:
        print("    none")
    for thread in report["threads"]:
        details = []
        if thread["in_stack_range"]:
            details.append(f"stack+0x{thread['stack_offset']:x}")
        if thread["registers"]:
            details.append("registers=" + ",".join(thread["registers"]))
        print(
            f"    [{thread['index']}] uid={_hex(thread['uid'])} "
            f"{thread['name'] or '<unnamed>'}: {'; '.join(details)}; "
            f"state={thread['status_name']}; "
            f"context={'saved' if thread['register_context_live'] else 'non-live'}"
        )

    print("\nGPU_REFERENCES:")
    if not any(
        (
            report["gpu_fault_groups"],
            report["gpu_resource_pointers"],
            report["gpu_register_references"],
        )
    ):
        print("    none")
    for group in report["gpu_fault_groups"]:
        owner = group["context_attribution"]["owner"]
        owner_text = (
            f" owner={owner['process_name'] or '<unnamed>'}"
            if owner is not None
            else " owner=<unresolved>"
        )
        print(
            f"    fault group {group['index']}: page={_hex(group['gpu_virtual_address'])} "
            f"requestor={','.join(group['requestor_names']) or 'none'} "
            f"context={group['context_id']}{owner_text} scopes={','.join(group['scopes'])}"
        )
    for pointer in report["gpu_resource_pointers"]:
        print(
            f"    resource {pointer['code']}: {_hex(pointer['address'])} "
            f"({pointer['relation']})"
        )
    for register in report["gpu_register_references"]:
        location = (
            "master" if register["scope"] == "master" else f"core{register['core']}"
        )
        fields = ", ".join(
            f"{field['name']}={_hex(field['address'])} ({field['relation']})"
            for field in register["fields"]
        )
        name = register["name"] or f"REG_{register['offset']:04X}"
        print(
            f"    {location} {name}: {fields}"
        )

    preview = report["memory_preview"]
    print("\nMEMORY_PREVIEW:")
    if preview["captured_size"]:
        print(_hexdump(bytes.fromhex(preview["bytes_hex"]), preview["address"]))
    elif preview["requested_size"] == 0:
        print("    disabled (--bytes 0)")
    else:
        print("    bytes not captured")
    for limitation in report["limitations"]:
        print(f"\nLIMITATION: {limitation}")
    return 0


def command_map(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    report = retained_map(core)
    if args.captured_only or args.unique:
        retained_indices = [
            index
            for index, item in enumerate(report["entries"])
            if (not args.captured_only or item["captured_bytes"])
            and (
                not args.unique
                or item["kind"] != "load-segment"
                or not item["alias_segment_indices"]
                or item["segment_index"] == min(item["alias_segment_indices"])
            )
        ]
        remap = {old: new for new, old in enumerate(retained_indices)}
        report["entries"] = [report["entries"][index] for index in retained_indices]
        report["overlaps"] = [
            {**item, "left": remap[item["left"]], "right": remap[item["right"]]}
            for item in report["overlaps"]
            if item["left"] in remap and item["right"] in remap
        ]
        report["entry_count"] = len(report["entries"])
        report["overlap_count"] = len(report["overlaps"])
        report["captured_entry_count"] = sum(
            bool(item["captured_bytes"]) for item in report["entries"]
        )
        report["metadata_only_entry_count"] = sum(
            not item["captured_bytes"] for item in report["entries"]
        )
    report["captured_only"] = bool(args.captured_only)
    report["unique_aliases"] = bool(args.unique)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(
        f"Address map: {report['entry_count']} ranges  "
        f"captured={report['captured_entry_count']} "
        f"metadata-only={report['metadata_only_entry_count']}"
    )
    print("  START     END       SIZE      CAPTURED  PERM KIND            NAME")
    for item in report["entries"]:
        permissions = item["permissions"] or "---"
        print(
            f"  {item['start']:08x}  {item['end']:08x}  "
            f"{item['size']:08x}  {item['captured_bytes']:08x}  "
            f"{permissions:4} {item['kind']:<15} {item['name'] or '<unnamed>'}"
        )
    print(
        f"\nOverlaps: {report['overlap_count']} "
        "(module, stack, allocation, and PT_LOAD views may describe the same range)"
    )
    return 0


def command_modules(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    report = modules_report(core)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(f"Modules: {report['module_count']}")
    print("  IDX UID        SIZE      CAPTURED EXEC/PC NAME")
    for item in report["modules"]:
        print(
            f"  {item['index']:3d} {item['uid']:08x} {item['total_size']:08x} "
            f"{item['captured_bytes']:08x} "
            f"{len(item['execution_threads']):2d}/"
            f"{len(item['threads_with_retained_pc']):<2d} "
            f"{item['name'] or '<unnamed>'}"
        )
    print("\nUse 'module DUMP SELECTOR' for segment and execution details.")
    return 0


def command_module(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    report = module_report(core, args.module)
    if args.images:
        _info, diagnosis = load_debug_state(core, _database(args))
        binary_images = enrich_diagnosis_with_elfs(diagnosis, args.images, args.sdk)
        module_index = report["module"]["index"]
        report["binary_images"] = {
            **{key: value for key, value in binary_images.items() if key != "images"},
            "images": [
                image
                for image in binary_images["images"]
                if (image.get("association") or {}).get("module_index")
                in (None, module_index)
            ],
        }
        for image in report["binary_images"]["images"]:
            imports = image.get("imports")
            if imports is None:
                continue
            image["import_details_included"] = bool(args.imports)
            if args.imports:
                continue
            image["imports"] = [
                {
                    **{
                        key: value
                        for key, value in library.items()
                        if key != "functions"
                    },
                    "resolved_function_count": sum(
                        function["name"] is not None
                        for function in library["functions"]
                    ),
                }
                for library in imports
            ]
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    item = report["module"]
    print(
        f"Module [{item['index']}] {item['name'] or '<unnamed>'} "
        f"uid={_hex(item['uid'])}"
    )
    print(
        f"Extent: 0x{item['total_size']:x}  captured=0x{item['captured_bytes']:x} "
        f"execution-contexts={len(item['execution_threads'])} "
        f"retained-pcs={len(item['threads_with_retained_pc'])}"
    )
    print("\nSEGMENTS:")
    for segment in item["segments"]:
        print(
            f"    {segment['number']}: {segment['start']:08x}-{segment['end']:08x} "
            f"{segment['permissions']} size=0x{segment['size']:x} "
            f"captured=0x{segment['captured_bytes']:x}"
        )
    print("\nEXECUTION:")
    if not item["threads_with_retained_pc"]:
        print("    no retained thread PC is in this module")
    for thread in item["threads_with_retained_pc"]:
        qualification = "saved" if thread["register_context_live"] else "non-live"
        print(
            f"    [{thread['index']}] {_hex(thread['uid'])} "
            f"{thread['location']['notation']} {thread['status_name']} {qualification} "
            f"{thread['name'] or '<unnamed>'}"
        )
    images = (report.get("binary_images") or {}).get("images", [])
    if args.images:
        print("\nIMAGES:")
        if not images:
            print("    no supplied image matched this captured module")
        for image in images:
            association = image.get("association")
            if association is None:
                print(f"    {image['path']} (invalid: {image['error']})")
                continue
            print(
                f"    {image['path']} ({image['source_format']}, "
                f"{association.get('match_quality') or association['status']})"
            )
            if association["module_index"] == item["index"]:
                print(
                    f"      symbols={image['symbol_count']} "
                    f"imports={image['import_function_count']} "
                    f"resolved-imports={image['resolved_import_function_count']}"
                )
            for reason in association["reasons"]:
                print(f"      rejected: {reason}")
            for warning in association.get("warnings", []):
                print(f"      warning: {warning}")
    return 0


def command_xref(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    report = xref_report(core, _database(args), args.address, args.max_results)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(
        f"Cross-references to {_hex(report['address'])}: "
        f"direct={report['direct_reference_count']} "
        f"memory={report['memory_reference_count']}"
    )
    target = report["target"]
    print(f"\nTARGET: {target['status']}")
    print(f"    {target['conclusion']}")
    for note in target["value_notes"]:
        print(f"    note: {note}")
    if target["module"] is not None:
        print(f"    module: {target['module']['notation']}")
    for block in target["memory_blocks"]:
        print(
            f"    allocation: [{block['index']}] {block['name'] or '<unnamed>'}"
            f"+0x{block['offset']:x}"
        )
    for thread in target["threads"]:
        relations = []
        if thread["in_stack_range"]:
            relations.append(f"stack+0x{thread['stack_offset']:x}")
        relations.extend(thread["registers"])
        context = "saved" if thread["register_context_live"] else "non-live"
        print(
            f"    thread: [{thread['index']}] {thread['name'] or '<unnamed>'} "
            f"({','.join(relations)}; {thread['status_name']}; {context})"
        )
    if target["load_segments"]:
        captured = sum(
            item["captured_at_address"] for item in target["load_segments"]
        )
        print(
            f"    PT_LOAD views: {len(target['load_segments'])} "
            f"({captured} contain captured bytes at the target)"
        )
    for group in target["gpu_fault_groups"]:
        print(
            f"    GPU fault page: group {group['index']} "
            f"requestor={','.join(group['requestor_names']) or 'none'}"
        )
    for pointer in target["gpu_resource_pointers"]:
        print(
            f"    GPU resource: {pointer['name']} ({pointer['stage']}; "
            f"{pointer['relation']})"
        )

    print("\nREFERENCES:")
    if not report["direct_references"] and not report["memory_references"]:
        print("    none retained")
    for item in report["direct_references"]:
        kind = item["kind"]
        if kind == "cpu-register":
            context = "saved" if item["register_context_live"] else "non-live"
            poison = "; poison-sentinel" if item["register_value_poison"] else ""
            print(
                f"    cpu [{item['thread_index']}] {_hex(item['thread_uid'])} "
                f"{item['register']} {item['thread_name'] or '<unnamed>'} "
                f"({item['status_name']}; {context}{poison})"
            )
        elif kind == "gpu-fault":
            group = item["fault_group"]
            print(
                f"    gpu fault group {group['index']} "
                f"requestor={','.join(group['requestor_names']) or 'none'}"
            )
        elif kind == "gpu-resource":
            pointer = item["resource"]
            print(f"    gpu resource {pointer['name']} ({pointer['stage']})")
        else:
            register = item["register"]
            print(
                f"    gpu register {register.get('name') or 'unnamed'} "
                f"offset={_hex(register.get('offset'), 4)}"
            )
    for item in report["memory_references"]:
        context = []
        if item["stack_owners"]:
            owner = item["stack_owners"][0]
            owner_context = "saved" if owner["register_context_live"] else "non-live"
            context.append(
                f"thread[{owner['index']}]+sp{owner['sp_offset']:+#x} "
                f"({owner['status_name']}; {owner_context})"
                if owner["sp_offset"] is not None
                else f"thread[{owner['index']}] stack "
                f"({owner['status_name']}; {owner_context})"
            )
        if item["module"] is not None:
            context.append(item["module"]["notation"])
        if item["memory_block"] is not None:
            block = item["memory_block"]
            context.append(f"{block['name'] or '<unnamed>'}+0x{block['offset']:x}")
        print(
            f"    memory {item['address']:08x} ({item['source_classification']})"
            + (f"  {'; '.join(context)}" if context else "")
        )
    if report["truncated"]:
        print("    result display truncated; increase --max-results")
    return 0


def command_waits(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    report = wait_graph(core, _database(args))
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(
        f"Wait graph: threads={report['thread_count']} "
        f"objects={report['object_count']} edges={report['edge_count']} "
        f"cycles={report['cycle_count']}"
    )
    nodes = {item["id"]: item for item in report["nodes"]}
    for edge in report["edges"]:
        source = nodes.get(edge["source"], {"id": edge["source"]})
        target = nodes.get(edge["target"], {"id": edge["target"]})
        source_name = source.get("name") or source["id"]
        target_name = target.get("name") or target["id"]
        print(
            f"    {source_name} --{edge['relation']}--> {target_name} "
            f"[{'; '.join(edge['evidence'])}]"
        )
    if report["unresolved_waits"]:
        print("\nUNRESOLVED WAIT IDS:")
        for wait in report["unresolved_waits"]:
            print(
                f"    thread[{wait['thread_index']}] {wait['thread_name']} "
                f"type={_hex(wait['wait_type'])} id={_hex(wait['wait_id'])}"
            )
    if report["cycles"]:
        print("\nCAPTURED DEPENDENCY CYCLES:")
        for cycle in report["cycles"]:
            print("    " + " -> ".join(cycle["nodes"]))
    else:
        print("\nNo dependency cycle is present in the retained wait/owner fields.")
    print(f"\nQualification: {report['qualification']}")
    return 0


def command_object(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    report = object_report(core, _database(args), args.uid)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    item = report["object"]
    print(
        f"Object {item['kind']} {_hex(item['uid'])} "
        f"{item['name'] or '<unnamed>'}"
    )
    print(
        f"Waiters: {item['waiting_thread_count']}  "
        f"owner={_hex(item.get('owner_thread_uid'))}  note={item['note_name']}"
    )
    print("State:")
    for key, value in sorted(item["state"].items()):
        display = _hex(value) if isinstance(value, int) else value
        print(f"    {key}: {display}")
    print("Relationships:")
    if not report["edges"] and not report["relationships"]:
        print("    none retained")
    for edge in report["edges"]:
        print(
            f"    {edge['source']} --{edge['relation']}--> {edge['target']} "
            f"[{'; '.join(edge['evidence'])}]"
        )
    for relation in report["relationships"]:
        print(
            f"    {relation['source']} --{relation['relation']}--> "
            f"{relation['target']}"
        )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    report = validate_core(core)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print(
            f"Validation: {report['status']}  "
            f"checks={report['check_count']} passed={report['passed_count']} "
            f"errors={report['failed_error_count']} "
            f"warnings={report['failed_warning_count']}"
        )
        for item in report["checks"]:
            mark = "ok" if item["passed"] else item["severity"]
            print(f"    {mark:<7} {item['code']}: {item['description']}")
            if not item["passed"] and item["evidence"]:
                print(f"            {json.dumps(item['evidence'], sort_keys=True)}")
        print(f"\nScope: {report['scope']}")
    return 0 if report["valid"] else 1


def _triage_group_synopsis(entry: dict, group_kind: str) -> str:
    if group_kind in ("failure", "failure_pattern"):
        fault_parts = []
        for fault in entry["faults"]:
            prefix = (
                f"{_hex(fault['page'])}:"
                if group_kind == "failure"
                else ""
            )
            fault_parts.append(
                f"{prefix}{'/'.join(fault['requestors']) or 'no-requestor'}"
                f"[{','.join(fault['conditions']) or 'no-condition'}]"
                f"/{fault['page_access_type']}"
                f"/cid={fault['context_id'] if fault['context_id'] is not None else 'unknown'}"
                f"/scope={','.join(fault['scopes']) or 'none'}"
            )
        events = entry["signature_components"]["event_codes"]
        return (
            f"classification={entry['classification']} "
            f"faults={'; '.join(fault_parts) or 'none'} "
            f"phases={','.join(entry['lockup_phases']) or 'none'} "
            f"events={','.join(events) or 'none'}"
        )
    if group_kind == "pipeline":
        return (
            f"phases={','.join(entry['lockup_phases']) or 'none'} "
            f"active-units={','.join(entry['active_units']) or 'none'}"
        )
    return "complete retained GPU register/indexed snapshot; volatile capture metadata excluded"


def command_triage(args: argparse.Namespace) -> int:
    report = triage_dumps(args.dumps, _database(args))
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(
        f"GPUCRASH triage: dumps={report['dump_count']} "
        f"errors={report['error_count']} "
        f"exact-groups={len(report['groups']['failure'])} "
        f"pattern-groups={len(report['groups']['failure_pattern'])}"
    )
    print("\nDUMPS:")
    for index, item in enumerate(report["entries"]):
        application = item["application"].get("title_id") or "<unknown-title>"
        pages = ",".join(_hex(value) for value in item["fault_pages"]) or "none"
        print(
            f"    [{index}] {item['classification']} app={application} "
            f"faults={pages} sig={item['failure_signature'][:12]}"
        )
        print(f"        {item['path']}")
    for name, title in (
        ("failure", "EXACT FAILURE GROUPS"),
        ("failure_pattern", "ADDRESS-INDEPENDENT FAILURE GROUPS"),
        ("pipeline", "PIPELINE GROUPS"),
        ("register", "REGISTER-STATE GROUPS"),
    ):
        print(f"\n{title}:")
        print(f"    Basis: {report['signature_notes'][name]}")
        for group in report["groups"][name]:
            indices = ",".join(str(value) for value in group["entry_indices"])
            print(
                f"    {group['signature'][:16]} count={group['count']} "
                f"dumps=[{indices}]"
            )
            representative = report["entries"][group["entry_indices"][0]]
            print(f"        {_triage_group_synopsis(representative, name)}")
    print("\nFACETS:")
    for name, values in report["facets"].items():
        if values:
            text = ", ".join(
                f"{_hex(int(item['value'])) if name == 'fault_page' else item['value']} "
                f"({item['count']})"
                for item in values[:8]
            )
            print(f"    {name}: {text}")
    if report["errors"]:
        print("\nERRORS:")
        for error in report["errors"]:
            print(f"    {error['path']}: {error['error']}")
    return 0


def command_threads(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    report = thread_list_report(core, _database(args))
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    process = report["process"] or {}
    print(
        f"Threads: {report['thread_count']}  process="
        f"{process.get('process_name') or '<unavailable>'} "
        f"pid={_hex(process.get('process_id'))}"
    )
    print("  IDX UID        STATE      CONTEXT  CPU PRI PC         LOCATION                 NAME")
    for thread in report["threads"]:
        location = thread["pc_location"]
        location_text = location["notation"] if location is not None else "<unresolved>"
        pc_text = _hex(thread["pc"])[2:] if thread["pc"] is not None else "missing"
        context = "saved" if thread["register_context_live"] else "non-live"
        print(
            f"  {thread['index']:3d} {thread['uid']:08x} "
            f"{thread['status_name'][:10]:<10} {context:<8} "
            f"{thread['current_cpu_id']:3d} "
            f"{thread['current_priority']:3d} {pc_text:8} "
            f"{location_text[:24]:<24} {thread['name']}"
        )
    print("\nUse 'thread DUMP UID' for registers, stack state, and return candidates.")
    return 0


def _thread_image_record(report: dict, runtime_address: int) -> Optional[dict]:
    images = report.get("binary_images")
    if images is None:
        return None
    for image in images["images"]:
        for record in image.get("address_records", []):
            if record["runtime_address"] == runtime_address:
                return record
    return None


def command_thread(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    database = _database(args)
    report, diagnosis = thread_detail_report(
        core, database, args.thread, args.stack_bytes
    )
    report["thread"]["annotated_stack"] = annotated_stack(
        core, diagnosis, report["thread"], args.stack_bytes
    )
    if args.images:
        diagnosis["binary_images"] = enrich_diagnosis_with_elfs(
            diagnosis, args.images, args.sdk
        )
        selected = next(
            item
            for item in diagnosis["execution_context"]["threads"]
            if item["uid"] == report["thread"]["uid"]
        )
        for key in ("pc_image", "lr_image"):
            if key in selected:
                report["thread"][key] = selected[key]
        report["binary_images"] = diagnosis["binary_images"]
        report["thread"]["unwind"] = unwind_thread(
            core,
            diagnosis,
            report["thread"],
            args.images,
            args.max_frames,
        )
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0

    thread = report["thread"]
    print(
        f"Thread [{thread['index']}] {thread['name'] or '<unnamed>'} "
        f"uid={_hex(thread['uid'])}"
    )
    print(
        f"State   : {thread['status_name']} status={_hex(thread['status'])} "
        f"stop={_hex(thread['stop_reason'])}"
    )
    print(
        f"Context : {thread['register_context_qualification']} "
        f"(poison-registers={thread['poison_register_count']}/16)"
    )
    print(
        f"Schedule: cpu={thread['current_cpu_id']} priority={thread['current_priority']} "
        f"affinity={_hex(thread['current_cpu_affinity'])} clocks={thread['run_clocks']}"
    )
    print(f"Wait    : type={_hex(thread['wait_type'])} id={_hex(thread['wait_id'])}")
    wait = thread.get("wait_object")
    if wait is not None:
        print(
            f"Object  : {wait['kind']} {_hex(wait['uid'])} "
            f"{wait['name'] or '<unnamed>'}"
        )
    registers = thread["registers"]
    print("\nREGISTERS:")
    if registers is None:
        print("    not retained")
    else:
        for row in range(0, 16, 4):
            print(
                "    "
                + "  ".join(
                    f"r{index:<2}={registers['gpr'][index]:08x}"
                    for index in range(row, row + 4)
                )
            )
        print(f"    cpsr={registers['cpsr']:08x}")
        for role in ("pc", "lr"):
            location = thread[f"{role}_location"]
            text = location["notation"] if location is not None else "<unresolved>"
            print(f"    {role.upper():2} -> {text}")
            image_record = _thread_image_record(report, registers[role])
            if image_record is not None:
                symbol = image_record["location"]["symbol"]
                if symbol is not None:
                    print(
                        f"         {symbol['name']}+0x{symbol['offset']:x}"
                    )
                source = image_record["source"]
                if source is not None:
                    print(f"         {source['function']} at {source['file_line']}")
                if image_record["disassembly"]:
                    print(
                        f"         ELF VA={_hex(image_record['location']['elf_virtual_address'])} "
                        f"{image_record['instruction_set']} "
                        f"({image_record['instruction_set_basis']})"
                    )
                for line in image_record["disassembly"]:
                    print(f"         {line}")

    usage = thread["stack_usage"]
    print("\nSTACK:")
    print(
        f"    range={thread['stack_base']:08x}-{thread['stack_end']:08x} "
        f"size=0x{thread['stack_size']:x} captured-at-sp="
        f"{str(thread['stack_captured_at_sp']).lower()}"
    )
    if usage is not None:
        print(
            f"    current=0x{usage['current_bytes']:x} peak=0x{usage['peak_bytes']:x}"
        )
    candidates = thread["stack_return_candidates"]
    print(f"    executable-address candidates={len(candidates)}")
    for candidate in candidates:
        image_location = candidate.get("image")
        symbol = image_location.get("symbol") if image_location is not None else None
        symbol_text = (
            f" ({symbol['name']}+0x{symbol['offset']:x})" if symbol else ""
        )
        print(
            f"      {candidate['stack_address']:08x}: {candidate['value']:08x} "
            f"{candidate['location']['notation']}{symbol_text}"
        )
    annotations = thread["annotated_stack"]
    if annotations["status"] == "non-live-thread-context":
        print("    annotation skipped: non-live thread context")
    else:
        print(
            f"    classified words={annotations['classified_word_count']}/"
            f"{annotations['word_count']}"
        )
    for word in annotations["words"]:
        if word["classifications"]:
            print(
                f"      sp+0x{word['sp_offset']:04x} {word['stack_address']:08x}: "
                f"{word['value']:08x}  "
                f"{_classification_text(word['classifications'])}"
            )
    preview = thread["stack_preview"]
    if preview["captured_size"]:
        print()
        print(_hexdump(bytes.fromhex(preview["bytes_hex"]), preview["address"]))
    elif args.stack_bytes == 0:
        print("    stack preview disabled")
    elif preview["requested_size"] == 0:
        print("    SP has no bytes inside the retained stack range")
    else:
        print("    bytes at SP are not captured")
    unwind = thread.get("unwind")
    if unwind is not None:
        print("\nARM EHABI BACKTRACE:")
        for frame in unwind["frames"]:
            location = frame["runtime_location"]
            where = location["notation"] if location is not None else "<unmapped>"
            symbol = frame["symbol"]
            if symbol is not None:
                where += f" ({symbol['name']}+0x{symbol['offset']:x})"
            print(
                f"    #{frame['index']:<2} pc={frame['pc']:08x} "
                f"sp={frame['sp']:08x} {where} [{frame['source']}]"
            )
        stop = unwind.get("stop") or {}
        print(
            f"    stopped: {stop.get('code', unwind['status'])}"
            + (f" ({stop['detail']})" if stop.get("detail") else "")
        )
        print(f"    {unwind['qualification']}")
    return 0


def command_search(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    if args.u8 is not None:
        if not 0 <= args.u8 <= 0xFF:
            raise ParseError("--u8 value is outside 0..0xff")
        pattern, description, default_alignment = bytes((args.u8,)), f"u8 {args.u8:#x}", 1
    elif args.u16 is not None:
        if not 0 <= args.u16 <= 0xFFFF:
            raise ParseError("--u16 value is outside 0..0xffff")
        pattern, description, default_alignment = struct.pack("<H", args.u16), f"u16 {args.u16:#x}", 2
    elif args.u32 is not None:
        if not 0 <= args.u32 <= 0xFFFFFFFF:
            raise ParseError("--u32 value is outside 0..0xffffffff")
        pattern, description, default_alignment = struct.pack("<I", args.u32), f"u32 {args.u32:#x}", 4
    elif args.pointers_to is not None:
        if not 0 <= args.pointers_to <= 0xFFFFFFFF:
            raise ParseError("--pointers-to address is outside the Vita address space")
        pattern = struct.pack("<I", args.pointers_to)
        description = f"little-endian pointer to {args.pointers_to:#x}"
        default_alignment = 4
    elif args.bytes_pattern is not None:
        compact = re.sub(r"[\s:_-]", "", args.bytes_pattern)
        try:
            pattern = bytes.fromhex(compact)
        except ValueError as exc:
            raise ParseError(f"invalid hexadecimal byte pattern: {exc}") from exc
        description, default_alignment = f"bytes {pattern.hex()}", 1
    else:
        try:
            pattern = args.string.encode(args.encoding)
        except (LookupError, UnicodeEncodeError) as exc:
            raise ParseError(f"cannot encode --string with {args.encoding}: {exc}") from exc
        description, default_alignment = f"{args.encoding} string {args.string!r}", 1
    alignment = args.alignment if args.alignment is not None else default_alignment
    report = search_memory(
        core,
        pattern,
        description,
        alignment=alignment,
        max_results=args.max_results,
        include_aliases=args.include_aliases,
    )
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(
        f"Search: {report['pattern']}  alignment={report['alignment']} "
        f"segments={report['searched_segment_count']} bytes=0x{report['searched_bytes']:x}"
    )
    print(
        f"Matches: {report['match_count']} "
        f"(showing {report['returned_count']})"
    )
    for item in report["results"]:
        labels = []
        if item["module"] is not None:
            labels.append(item["module"]["notation"])
        if item["memory_block"] is not None:
            block = item["memory_block"]
            labels.append(f"{block['name'] or '<unnamed>'}+0x{block['offset']:x}")
        alias = (
            " aliases=" + ",".join(str(value) for value in item["alias_segment_indices"])
            if item["alias_segment_indices"]
            else ""
        )
        print(
            f"    {item['address']:08x} ph{item['segment_index']}+"
            f"0x{item['segment_offset']:x}{alias}"
            + (f"  {'; '.join(labels)}" if labels else "")
        )
    if report["truncated"]:
        print("    result display truncated; increase --max-results for additional matches")
    return 0


def command_gpu(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    database = _database(args)
    info, diagnosis = load_debug_state(core, database)
    if args.view == "faults":
        result = {
            "path": str(core.path),
            "classification": diagnosis["classification"],
            "fault_sources": diagnosis["fault_sources"],
            "fault_groups": diagnosis["fault_groups"],
            "bif": info.bif_diagnosis(),
        }
    elif args.view == "contexts":
        result = {
            "path": str(core.path),
            "context_owners": diagnosis["context_owners"],
            "context_assignments": diagnosis["context_assignments"],
            "context_scope_assignments": diagnosis["context_scope_assignments"],
            "context_slot_map_scopes": list(GPU_CONTEXT_MAP_SCOPES),
            "context_slot_map": [
                list(row) for row in info.decoded_trailer().context_slot_map
            ],
        }
    elif args.view == "pipeline":
        selected_cores = list(dict.fromkeys(args.cores or range(info.core_count)))
        invalid_cores = [
            core for core in selected_cores if not 0 <= core < info.core_count
        ]
        if invalid_cores:
            raise ParseError(
                "GPU core index out of range: "
                + ", ".join(str(core) for core in invalid_cores)
                + f" (available: 0-{info.core_count - 1})"
            )
        pipeline = diagnosis["firmware_pipeline"]
        firmware_pipeline = (
            {
                **pipeline,
                "selected_cores": selected_cores,
                "cores": [
                    core for core in pipeline["cores"] if core["core"] in selected_cores
                ],
            }
            if args.details
            else _compact_firmware_pipeline(pipeline, selected_cores)
        )
        result = {
            "path": str(core.path),
            "lockup_phases": [
                item
                for item in diagnosis["lockup_phases"]
                if _gpu_scope_selected(item["scope"], selected_cores)
            ],
            "pipeline_markers": [
                item
                for item in diagnosis["pipeline_markers"]
                if _gpu_scope_selected(item["scope"], selected_cores)
            ],
            "details_included": bool(args.details),
            "firmware_pipeline": firmware_pipeline,
        }
    else:
        allocation_evidence = diagnosis["allocation_evidence"]
        result = {
            "path": str(core.path),
            "resource_pointers": diagnosis["resource_pointers"],
            "allocation_details_included": bool(args.allocations),
            "gpu_named_allocations": (
                allocation_evidence
                if args.allocations
                else {
                    key: value
                    for key, value in allocation_evidence.items()
                    if key != "allocations"
                }
            ),
        }
    result["schema_version"] = 1
    result["command"] = f"gpu {args.view}"
    if args.json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    if args.view == "faults":
        print(f"GPU fault state: {diagnosis['classification']}")
        _print_fault_records(diagnosis)
        _print_bif_register_state(info)
    elif args.view == "contexts":
        _print_context_table(info, diagnosis)
    elif args.view == "pipeline":
        if args.details:
            _print_firmware_pipeline_diagnostics(diagnosis, selected_cores)
            _print_pipeline_register_state(info, database, selected_cores)
        else:
            _print_firmware_pipeline_summary(diagnosis, selected_cores)
        _print_lockup_state(diagnosis, selected_cores)
    else:
        _print_resource_pointers(diagnosis)
        if args.allocations:
            _print_allocation_evidence(diagnosis)
        else:
            _print_allocation_summary(diagnosis)
    return 0


def _register_matches(name: str, offset: int, matcher: Optional[re.Pattern]) -> bool:
    return matcher is None or matcher.search(name) is not None or matcher.search(f"0x{offset:04x}") is not None


def command_registers(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    database = _database(args)
    info = _gpu_info(core)
    if info is None:
        raise ParseError("GPU_INFO note is unavailable")
    matcher = re.compile(args.matching, re.IGNORECASE) if args.matching else None
    if args.json:
        summary = info.summary(database)
        master = [
            item
            for item in summary["master_registers"]
            if (args.show_zero or item["value"] != 0)
            and _register_matches(
                item["name"] or f"REG_{item['offset']:04X}", item["offset"], matcher
            )
        ]
        per_core = [
            item
            for item in summary["per_core_registers"]
            if (args.show_zero or any(item["values"]))
            and _register_matches(
                item["name"] or f"REG_{item['offset']:04X}", item["offset"], matcher
            )
        ]
        indexed = [
            item
            for item in summary["indexed_state"]
            if args.indexed
            and (args.show_zero or any(item["values"]))
            and _register_matches(
                item["name"] or f"SELECTOR_{item['index']:02X}",
                item["index"],
                matcher,
            )
        ]
        json.dump(
            {
                "schema_version": 1,
                "command": "registers",
                "path": str(core.path),
                "core_count": info.core_count,
                "master_registers": master,
                "per_core_registers": per_core,
                "indexed_state": indexed,
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        print()
        return 0
    print("Master registers:")
    for register in info.master_registers:
        definition = database.definition(register.offset, master=True)
        name = definition.display_name if definition else f"REG_{register.offset:04X}"
        if not args.show_zero and register.value == 0:
            continue
        if not _register_matches(name, register.offset, matcher):
            continue
        fields = decoded_fields(definition, register.value)
        suffix = " " + ", ".join(_decoded_field_text(item) for item in fields) if fields else ""
        print(f"  {register.offset:04x} {register.value:08x} {name}{suffix}")
    print("Per-core registers:")
    for register in info.per_core_registers:
        definition = database.definition(register.offset, master=False)
        name = definition.display_name if definition else f"REG_{register.offset:04X}"
        if not args.show_zero and not any(register.values):
            continue
        if not _register_matches(name, register.offset, matcher):
            continue
        values = " ".join(f"c{index}={value:08x}" for index, value in enumerate(register.values))
        field_text = [
            _register_field_text(database, register.offset, False, value)
            for value in register.values
        ]
        if field_text and len(set(field_text)) == 1 and field_text[0]:
            suffix = f" {field_text[0]}"
        else:
            decoded = " ".join(
                f"c{index}[{text}]"
                for index, text in enumerate(field_text)
                if text
            )
            suffix = f" {decoded}" if decoded else ""
        print(f"  {register.offset:04x} {values} {name}{suffix}")
    if args.indexed:
        print("Indexed per-core state:")
        for item in info.indexed_state:
            metadata = indexed_selector_metadata(item.offset)
            name = metadata["name"] if metadata is not None else f"SELECTOR_{item.offset:02X}"
            if not args.show_zero and not any(item.values):
                continue
            if not _register_matches(name, item.offset, matcher):
                continue
            values = " ".join(f"c{index}={value:08x}" for index, value in enumerate(item.values))
            print(f"  {item.offset:04x} {values} {name}")
    return 0


def command_compare(args: argparse.Namespace) -> int:
    left_core = CoreDump.read(args.left)
    right_core = CoreDump.read(args.right)
    database = _database(args)
    left = _gpu_info(left_core)
    right = _gpu_info(right_core)
    if left is None or right is None:
        raise ParseError("both dumps must contain GPU_INFO")
    comparison = compare_gpu_info(left, right, database)
    unfiltered_change_counts = {
        "header": len(comparison["header_changes"]),
        "master_register": len(comparison["master_changes"]),
        "per_core_register": len(comparison["per_core_changes"]),
        "indexed_state": len(comparison["indexed_state_changes"]),
        "context_owner": len(comparison["context_owner_changes"]),
        "driver_state": len(comparison["driver_state_changes"]),
    }
    _left_info, left_diagnosis = load_debug_state(left_core, database)
    _right_info, right_diagnosis = load_debug_state(right_core, database)
    debug_comparison = compare_debug_state(left_diagnosis, right_diagnosis)
    allowed_sections = {
        "verdict",
        "faults",
        "contexts",
        "pipeline",
        "resources",
        "threads",
        "allocations",
    }
    requested_sections = []
    for value in args.sections:
        requested_sections.extend(
            item.strip().casefold() for item in value.split(",") if item.strip()
        )
    unknown_sections = sorted(set(requested_sections) - allowed_sections)
    if unknown_sections:
        raise ParseError(
            "unknown compare section(s): " + ", ".join(unknown_sections)
        )
    selected_sections = set(requested_sections) or allowed_sections
    matcher = None
    if args.matching:
        try:
            matcher = re.compile(args.matching, re.IGNORECASE)
        except re.error as exc:
            raise ParseError(f"invalid --matching regular expression: {exc}") from exc
    raw_registers = args.registers or args.show_indexed or matcher is not None
    if matcher is not None:
        comparison["master_changes"] = [
            item
            for item in comparison["master_changes"]
            if _register_matches(
                item["name"] or f"REG_{item['offset']:04X}",
                item["offset"],
                matcher,
            )
        ]
        comparison["per_core_changes"] = [
            item
            for item in comparison["per_core_changes"]
            if _register_matches(
                item["name"] or f"REG_{item['offset']:04X}",
                item["offset"],
                matcher,
            )
        ]
        comparison["indexed_state_changes"] = [
            item
            for item in comparison["indexed_state_changes"]
            if _register_matches(
                item["name"] or f"SELECTOR_{item['index']:02X}",
                item["index"],
                matcher,
            )
        ]
    postmortem_keys = {
        "verdict": "verdict",
        "faults": "fault_groups",
        "contexts": "context_assignment_changes",
        "pipeline": "pipeline",
        "resources": "resource_pointers",
        "threads": "thread_changes",
        "allocations": "allocations",
    }
    postmortem_view = {
        output_key: debug_comparison[output_key]
        for section, output_key in postmortem_keys.items()
        if section in selected_sections
    }
    if "contexts" in selected_sections:
        postmortem_view["context_scope_assignment_changes"] = debug_comparison["context_scope_assignment_changes"]
    if args.summary_only:
        summarized = {}
        if "verdict" in selected_sections:
            summarized["verdict"] = debug_comparison["verdict"]
        if "faults" in selected_sections:
            value = debug_comparison["fault_groups"]
            summarized["fault_groups"] = {
                "added_count": len(value["added"]),
                "removed_count": len(value["removed"]),
                "changed_count": len(value["changed"]),
            }
        if "contexts" in selected_sections:
            summarized["context_assignment_change_count"] = len(
                debug_comparison["context_assignment_changes"]
            )
            changes = debug_comparison["context_scope_assignment_changes"]
            summarized["context_scope_assignment_change_count"] = sum(len(items) for items in changes.values())
        if "pipeline" in selected_sections:
            value = debug_comparison["pipeline"]
            summarized["pipeline"] = {
                "master_changed": value["master_changed"],
                "core_change_count": len(value["core_changes"]),
                "render_state_changed": value["render_state_changed"],
                "clipper_halt_check_changed": value["clipper_halt_check_changed"],
            }
        if "resources" in selected_sections:
            value = debug_comparison["resource_pointers"]
            summarized["resource_pointers"] = {
                "added_count": len(value["added"]),
                "removed_count": len(value["removed"]),
                "changed_count": len(value["changed"]),
            }
        if "threads" in selected_sections:
            summarized["thread_change_count"] = len(
                debug_comparison["thread_changes"]
            )
        if "allocations" in selected_sections:
            value = debug_comparison["allocations"]
            summarized["allocations"] = {
                "left_count": value["left_count"],
                "right_count": value["right_count"],
                "added_count": len(value["added"]),
                "removed_count": len(value["removed"]),
                "changed_count": len(value["changed"]),
            }
        postmortem_view = summarized
    gpu_summary = {
        "left_bif": comparison["left_bif"],
        "right_bif": comparison["right_bif"],
        "change_counts": {
            "header": len(comparison["header_changes"]),
            "master_register": len(comparison["master_changes"]),
            "per_core_register": len(comparison["per_core_changes"]),
            "indexed_state": len(comparison["indexed_state_changes"]),
            "context_owner": len(comparison["context_owner_changes"]),
            "driver_state": len(comparison["driver_state_changes"]),
        },
        "unfiltered_change_counts": unfiltered_change_counts,
        "register_filter_applied": matcher is not None,
        "context_slot_map_changed": comparison["context_slot_map_changed"],
    }
    result = {
        "schema_version": 1,
        "command": "compare",
        "left": {"path": str(left_core.path), "sha256": left_core.raw_file_sha256},
        "right": {"path": str(right_core.path), "sha256": right_core.raw_file_sha256},
        "selected_sections": sorted(selected_sections),
        "summary_only": args.summary_only,
        "register_details": raw_registers and not args.summary_only,
        "matching": args.matching,
        "gpu": comparison if raw_registers and not args.summary_only else gpu_summary,
        "postmortem": postmortem_view,
    }
    if args.json:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(f"GPU crash comparison:\n  left:  {left_core.path}\n  right: {right_core.path}")
    print(
        f"BIF: {comparison['left_bif']['classification']} -> "
        f"{comparison['right_bif']['classification']}"
    )
    if "verdict" in selected_sections:
        verdict = debug_comparison["verdict"]
        print(
            f"Verdict: {verdict['left_classification']} -> "
            f"{verdict['right_classification']} "
            f"(bucket changed={str(verdict['changed']).lower()})"
        )
    if "faults" in selected_sections:
        fault_changes = debug_comparison["fault_groups"]
        print(
            f"Fault groups: +{len(fault_changes['added'])} "
            f"-{len(fault_changes['removed'])} "
            f"~{len(fault_changes['changed'])}"
        )
        if not args.summary_only:
            for direction, items in (
                ("+", fault_changes["added"]),
                ("-", fault_changes["removed"]),
            ):
                for item in items:
                    print(
                        f"  {direction} page={_hex(item['gpu_virtual_address'])} "
                        f"requestor={','.join(item['requestor_names']) or 'none'} "
                        f"context={item['context_id']} scopes={','.join(item['scopes'])}"
                    )
    if "contexts" in selected_sections:
        context_changes = debug_comparison["context_assignment_changes"]
        print(f"Context assignment changes: {len(context_changes)}")
        scoped = debug_comparison["context_scope_assignment_changes"]
        print(f"Scoped context changes: +{len(scoped['added'])} -{len(scoped['removed'])} ~{len(scoped['changed'])}")
        if not args.summary_only:
            def context_text(value: Optional[dict]) -> str:
                if value is None:
                    return "missing"
                owner = value.get("owner")
                if owner is None:
                    return f"{value['status']} slots={value['candidate_slots']}"
                return (
                    f"slot{owner['slot']} {owner['process_name'] or '<unnamed>'} "
                    f"pid={_hex(owner['process_id'])} ({value['status']})"
                )
            for item in context_changes:
                print(
                    f"  CID {item['context_id']}: {context_text(item['left'])} -> "
                    f"{context_text(item['right'])}"
                )
            for sign, items in (("+", scoped["added"]), ("-", scoped["removed"])):
                for item in items:
                    print(f"  {sign} {item['scope']} CID {item['context_id']}: {context_text(item)}")
            for change in scoped["changed"]:
                key = change["key"]
                print(f"  {key['scope']} CID {key['context_id']}: {context_text(change['left'])} -> {context_text(change['right'])}")
    if "pipeline" in selected_sections:
        pipeline_change = debug_comparison["pipeline"]
        print(
            "Pipeline changes: "
            f"master={str(pipeline_change['master_changed']).lower()} "
            f"cores={len(pipeline_change['core_changes'])} "
            f"frame/scene={str(pipeline_change['render_state_changed']).lower()} "
            f"clipper-signature={str(pipeline_change['clipper_halt_check_changed']).lower()}"
        )
        if not args.summary_only:
            if pipeline_change["render_state_changed"]:
                old_state = pipeline_change["left_render_state"]
                new_state = pipeline_change["right_render_state"]
                if old_state["available"] != new_state["available"]:
                    print(f"  frame/scene available: {old_state['available']} -> {new_state['available']}")
                for name in GPU_RENDER_STATE_FIELDS:
                    if old_state[name] != new_state[name]:
                        print(f"  {name}: {old_state[name]} -> {new_state[name]}")
            if pipeline_change["clipper_halt_check_changed"]:
                print(f"  clipper signature: {pipeline_change['left_clipper_halt_check']['status']} -> {pipeline_change['right_clipper_halt_check']['status']}")
            if pipeline_change["master_changed"]:
                print(
                    f"  master: {pipeline_change['left_master']} -> "
                    f"{pipeline_change['right_master']}"
                )
            for change in pipeline_change["core_changes"]:
                left_core = change["left"] or {}
                right_core = change["right"] or {}
                print(f"  core{change['core']}:")
                for key, label in (
                    ("active_phases", "phases"),
                    ("active_units", "units"),
                    ("usse_data_master_usage", "usse-usage"),
                    ("draw_counters", "draw-counters"),
                    ("draw_counter_applicability", "draw-applicability"),
                    ("isp_progress", "isp-progress"),
                    ("render_bounds", "render-bounds"),
                    ("region_header_base", "region-headers"),
                    ("tile_progress", "tile-progress"),
                    ("tile_applicability", "tile-applicability"),
                    ("zls", "depth/stencil"),
                ):
                    if left_core.get(key) != right_core.get(key):
                        print(
                            f"    {label}: {left_core.get(key)} -> "
                            f"{right_core.get(key)}"
                        )
    if "resources" in selected_sections:
        resources = debug_comparison["resource_pointers"]
        print(
            f"Resource pointers: +{len(resources['added'])} "
            f"-{len(resources['removed'])} ~{len(resources['changed'])}"
        )
        if not args.summary_only:
            for sign, items in (("+", resources["added"]), ("-", resources["removed"])):
                for item in items:
                    allocation = item.get("allocation")
                    allocation_text = (
                        f" {allocation['name'] or '<unnamed>'}+0x{allocation['offset']:x}"
                        if allocation is not None
                        else " unresolved-allocation"
                    )
                    print(
                        f"  {sign} {item['code']} {_hex(item['address'])}"
                        f"{allocation_text}"
                    )
            for item in resources["changed"]:
                print(
                    f"  ~ {item['key']}: left={item['left']} right={item['right']}"
                )
    if "threads" in selected_sections:
        thread_changes = debug_comparison["thread_changes"]
        print(f"Thread state changes: {len(thread_changes)}")
        if not args.summary_only:
            def thread_text(value: Optional[dict]) -> str:
                if value is None:
                    return "missing"
                context = "saved" if value["register_context_live"] else "non-live"
                return (
                    f"[{value['index']}] {value['name'] or '<unnamed>'} "
                    f"{value['status_name']} context={context} "
                    f"cpu={value['current_cpu_id']} "
                    f"pc={_hex(value['pc'])} sp={_hex(value['sp'])} "
                    f"wait={_hex(value['wait_id'])}"
                )

            for item in thread_changes:
                print(
                    f"  {_hex(item['uid'])}: {thread_text(item['left'])} -> "
                    f"{thread_text(item['right'])}"
                )
    if "allocations" in selected_sections:
        allocations = debug_comparison["allocations"]
        print(
            f"GPU-named allocations: {allocations['left_count']} -> "
            f"{allocations['right_count']} (+{len(allocations['added'])} "
            f"-{len(allocations['removed'])} ~{len(allocations['changed'])})"
        )
        if not args.summary_only:
            for sign, items in (
                ("+", allocations["added"]),
                ("-", allocations["removed"]),
            ):
                for item in items:
                    print(
                        f"  {sign} {item['base']:08x}-{item['end']:08x} "
                        f"size=0x{item['size']:x} {item['name'] or '<unnamed>'}"
                    )
            for item in allocations["changed"]:
                ignored = {
                    "index", "name", "base", "end", "size", "committed_size"
                }
                changed_fields = [
                    key
                    for key in sorted(set(item["left"]) | set(item["right"]))
                    if key not in ignored
                    and item["left"].get(key) != item["right"].get(key)
                ]
                print(
                    f"  ~ {item['key']['base']:08x} size=0x{item['key']['size']:x} "
                    f"{item['key']['name']}: "
                    + "; ".join(
                        f"{key}={item['left'].get(key)!r}->{item['right'].get(key)!r}"
                        for key in changed_fields
                    )
                )
    if raw_registers:
        print("\nCAPTURED GPU STATE:")
        if matcher is not None:
            print(f"Register filter: {args.matching!r}")
        print(f"Header changes: {len(comparison['header_changes'])}")
        if not args.summary_only and matcher is None:
            for item in comparison["header_changes"]:
                print(
                    f"  word{item['word']}: {item['left']:08x} -> "
                    f"{item['right']:08x}"
                )
        print(f"Master register changes: {len(comparison['master_changes'])}")
        if not args.summary_only:
            for item in comparison["master_changes"]:
                name = item["name"] or f"REG_{item['offset']:04X}"
                print(
                    f"  {item['offset']:04x} {_hex(item['left'])} -> "
                    f"{_hex(item['right'])} {name}"
                )
        print(f"Per-core register changes: {len(comparison['per_core_changes'])}")
        if not args.summary_only:
            for item in comparison["per_core_changes"]:
                name = item["name"] or f"REG_{item['offset']:04X}"
                old = " ".join(f"{value:08x}" for value in (item["left"] or []))
                new = " ".join(f"{value:08x}" for value in (item["right"] or []))
                print(f"  {item['offset']:04x} [{old}] -> [{new}] {name}")
        print(f"Indexed state changes: {len(comparison['indexed_state_changes'])}")
        if not args.summary_only and args.show_indexed:
            for item in comparison["indexed_state_changes"]:
                old = " ".join(f"{value:08x}" for value in item["left"])
                new = " ".join(f"{value:08x}" for value in item["right"])
                name = item["name"] or f"SELECTOR_{item['index']:02X}"
                print(f"  {item['index']:04x} [{old}] -> [{new}] {name}")
        print(f"GPU context-owner changes: {len(comparison['context_owner_changes'])}")
        if not args.summary_only and matcher is None:
            for item in comparison["context_owner_changes"]:
                old = item["left"]
                new = item["right"]
                old_text = (
                    f"{_hex(old['process_id'])} {old['process_name']}"
                    if old
                    else "missing"
                )
                new_text = (
                    f"{_hex(new['process_id'])} {new['process_name']}"
                    if new
                    else "missing"
                )
                print(f"  slot{item['slot']}: {old_text} -> {new_text}")
        print(
            f"Context-slot map changed: {int(comparison['context_slot_map_changed'])}; "
            f"frame/scene words changed: {len(comparison['driver_state_changes'])}"
        )
    else:
        counts = gpu_summary["change_counts"]
        print(
            "Captured GPU state changes: "
            f"master={counts['master_register']} per-core={counts['per_core_register']} "
            f"indexed={counts['indexed_state']} (use --registers for details)"
        )
    return 0


def command_memory(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    data = core.read_memory(args.address, args.size)
    block = core.memory_block_for(args.address)
    if args.output is not None and args.json:
        raise ParseError("--output and --json are mutually exclusive")
    if args.output is not None:
        args.output.write_bytes(data)
        print(f"wrote {len(data)} bytes from 0x{args.address:08x} to {args.output}")
        return 0
    if args.json:
        json.dump(
            {
                "schema_version": 1,
                "command": "memory",
                "path": str(core.path),
                "address": args.address,
                "size": len(data),
                "bytes_hex": data.hex(),
                "ascii": "".join(
                    chr(value) if 32 <= value < 127 else "." for value in data
                ),
                "u32": [
                    {
                        "address": args.address + offset,
                        "value": struct.unpack_from("<I", data, offset)[0],
                    }
                    for offset in range(0, len(data) - 3, 4)
                ],
                "memory_block": (
                    {
                        **block.summary(),
                        "end": block.end,
                        "offset": args.address - block.base,
                    }
                    if block is not None
                    else None
                ),
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        print()
        return 0
    if block:
        print(f"memory block: {block.name} [{block.base:08x}, {block.end:08x})")
    print(_hexdump(data, args.address))
    return 0


def command_activity(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    info = _gpu_activity(core)
    if info is None:
        raise ParseError("GPU_ACT_INFO note is unavailable")
    if args.json:
        json.dump(info.summary(include_raw=args.raw_hex), sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(
        f"GPU_ACT_INFO: v{info.format_version} sequence={info.sequence} "
        f"payload=0x{info.payload_size:x} trailing=0x{len(info.trailing):x}"
    )
    if info.activity_buffer is None:
        print("  no activity ring was retained")
        return 0
    ring = info.activity_buffer
    print(
        f"  ring: v0x{ring.format_version:x} events={ring.event_count} "
        f"size=0x{ring.declared_size:x}"
    )
    for slot in ring.slots:
        if slot.populated or args.show_empty:
            print(
                f"  slot{slot.index}: events={slot.event_count} tag={_hex(slot.tag)} "
                f"{slot.name or '<unnamed>'}"
            )
    return 0


def command_segments(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    alias_groups = core.load_alias_groups()
    alias_by_index = {}
    for group_number, indices in enumerate(alias_groups, 1):
        for index in indices:
            alias_by_index[index] = group_number
    canonical_aliases = {min(indices) for indices in alias_groups}
    segments = [
        segment
        for segment in core.loads
        if not args.unique or segment.index not in alias_by_index
        or segment.index in canonical_aliases
    ]
    if args.json:
        json.dump(
            {
                "path": str(core.path),
                "alias_groups": alias_groups,
                "segments": [segment.summary() for segment in segments],
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        print()
        return 0
    print(f"Captured PT_LOAD segments: {len(segments)} shown, {len(core.loads)} total")
    for segment in segments:
        block = core.memory_block_for(segment.virtual_address)
        alias = f" alias-group={alias_by_index[segment.index]}" if segment.index in alias_by_index else ""
        label = f" {block.name}" if block else ""
        print(
            f"  ph{segment.index}: {segment.virtual_address:08x}-{segment.virtual_address + segment.file_size:08x} "
            f"size=0x{segment.file_size:x} flags={segment.flags:x}{alias}{label}"
        )
    return 0


def command_tty(args: argparse.Namespace) -> int:
    if args.tail is not None and args.tail < 0:
        raise ParseError("--tail cannot be negative")
    if args.raw and (args.tail is not None or args.matching):
        raise ParseError("--tail and --matching cannot be combined with --raw")
    try:
        matcher = re.compile(args.matching, re.IGNORECASE) if args.matching else None
    except re.error as exc:
        raise ParseError(f"invalid --matching regular expression: {exc}") from exc
    core = CoreDump.read(args.dump)
    names = (args.note,) if args.note != "all" else ("TTY_INFO2", "TTY_INFO")
    decoded = []
    for name in names:
        note = core.note(name)
        if note is None:
            continue
        info = TtyInfo.parse(note.description)
        decoded.append((name, info))
    if not decoded:
        raise ParseError(f"requested TTY note {args.note!r} is unavailable")
    if args.raw:
        if len(decoded) != 1:
            raise ParseError("--raw requires one explicit --note")
        sys.stdout.buffer.write(decoded[0][1].payload)
        return 0

    def selected_lines(info: TtyInfo):
        lines = list(info.lines(args.preserve_ansi))
        matched = [line for line in lines if matcher is None or matcher.search(line.text)]
        if args.tail is not None:
            matched = matched[-args.tail :] if args.tail else []
        return lines, matched

    if args.json:
        result = {}
        for name, info in decoded:
            summary = info.summary(preserve_ansi=args.preserve_ansi)
            if args.tail is not None or matcher is not None:
                lines, selected = selected_lines(info)
                summary["text"] = (
                    "\n".join(line.text for line in selected) + ("\n" if selected else "")
                )
                summary["timeline"]["lines"] = [line.summary() for line in selected]
                summary["selection"] = {
                    "matching": args.matching,
                    "tail": args.tail,
                    "original_line_count": len(lines),
                    "selected_line_count": len(selected),
                }
            result[name] = summary
        json.dump(
            result,
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        print()
        return 0
    if args.tail is not None or matcher is not None:
        selected_by_note = [
            (name, selected_lines(info)[1]) for name, info in decoded
        ]
        selected_by_note = [item for item in selected_by_note if item[1]]
        if not selected_by_note:
            print("No TTY lines matched the requested selection.")
            return 0
        for index, (name, lines) in enumerate(selected_by_note):
            if len(selected_by_note) > 1:
                if index:
                    print()
                print(f"== {name} ==")
            print("\n".join(line.text for line in lines))
        return 0
    nonempty = [(name, info) for name, info in decoded if info.text(args.preserve_ansi)]
    selected = nonempty or decoded
    for index, (name, info) in enumerate(selected):
        if len(selected) > 1:
            if index:
                print()
            print(f"== {name} ==")
        text = info.text(preserve_ansi=args.preserve_ansi)
        if text:
            print(text, end="" if text.endswith("\n") else "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psp2-gpucrash-parse",
        description="Parse PS Vita GPUCRASH PSP2 core dumps",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--register-header",
        action="append",
        default=[],
        metavar="PATH",
        help="SGX autogenerated register header (repeatable)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="diagnose one GPU crash and explain the available evidence"
    )
    analyze.add_argument("dump", type=Path)
    analyze.add_argument(
        "--image",
        "--elf",
        "--self",
        dest="images",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help=(
            "original Vita ELF, decrypted SELF, or fake SELF to use for address "
            "analysis (repeatable)"
        ),
    )
    analyze.add_argument(
        "--sdk",
        type=Path,
        metavar="PATH",
        help="VitaSDK root used to locate objdump, addr2line, and the NID database",
    )
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--notes", action="store_true", help="list every ELF note")
    analyze.add_argument(
        "--memory-blocks",
        action="store_true",
        help="list the full memory-block inventory after the diagnosis",
    )
    analyze.add_argument(
        "--log-lines",
        type=int,
        default=3,
        metavar="N",
        help="maximum noteworthy and recent TTY lines to show (default: 3)",
    )
    analyze.add_argument(
        "--no-log", action="store_true", help="omit application TTY clues"
    )
    analyze.add_argument(
        "--gpu-dump", type=Path, metavar="PATH",
        help="optional devkit gpu_core_dump binary; correlate without replacing GPU_INFO",
    )
    analyze.set_defaults(handler=command_analyze)

    gpu_dump = subparsers.add_parser(
        "gpu-dump", help="decode a devkit gpu_core_dump binary, optionally correlate a GPUCRASH"
    )
    gpu_dump.add_argument("file", type=Path)
    gpu_dump.add_argument("--core-dump", type=Path, metavar="PATH", help="GPUCRASH to correlate")
    gpu_dump_output = gpu_dump.add_mutually_exclusive_group()
    gpu_dump_output.add_argument("--json", action="store_true", help="include all decoded tables and raw diagnostic text bytes")
    gpu_dump_output.add_argument("--text", action="store_true", help="print only the embedded firmware report")
    gpu_dump.add_argument("--registers", action="store_true", help="include all sidecar register values in text output")
    gpu_dump.set_defaults(handler=command_gpu_dump)

    address = subparsers.add_parser(
        "address",
        help="resolve a CPU or GPU virtual address against retained dump evidence",
    )
    address.add_argument("dump", type=Path)
    address.add_argument("address", type=_number)
    address.add_argument(
        "--bytes",
        type=_number,
        default=64,
        metavar="N",
        help="maximum captured bytes to display (default: 64)",
    )
    address.add_argument(
        "--image",
        "--elf",
        "--self",
        dest="images",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="identity-checked original ELF or decrypted/fake SELF (repeatable)",
    )
    address.add_argument("--sdk", type=Path, metavar="PATH")
    address.add_argument("--json", action="store_true")
    address.set_defaults(handler=command_address)

    address_map = subparsers.add_parser(
        "map", help="show the unified retained address map"
    )
    address_map.add_argument("dump", type=Path)
    address_map.add_argument(
        "--captured-only",
        action="store_true",
        help="omit ranges represented only by metadata",
    )
    address_map.add_argument(
        "--unique",
        action="store_true",
        help="show one PT_LOAD entry from each exact alias group",
    )
    address_map.add_argument("--json", action="store_true")
    address_map.set_defaults(handler=command_map)

    modules = subparsers.add_parser(
        "modules", help="list captured process modules and segment coverage"
    )
    modules.add_argument("dump", type=Path)
    modules.add_argument("--json", action="store_true")
    modules.set_defaults(handler=command_modules)

    module = subparsers.add_parser(
        "module", help="inspect one captured module by UID, index, or name"
    )
    module.add_argument("dump", type=Path)
    module.add_argument("module", help="module UID, table index, or unique name")
    module.add_argument(
        "--image",
        "--elf",
        "--self",
        dest="images",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="identity-checked original ELF or decrypted/fake SELF (repeatable)",
    )
    module.add_argument("--sdk", type=Path, metavar="PATH")
    module.add_argument(
        "--imports",
        action="store_true",
        help="include every imported-function record in image JSON",
    )
    module.add_argument("--json", action="store_true")
    module.set_defaults(handler=command_module)

    xref = subparsers.add_parser(
        "xref", help="find and classify retained references to an address"
    )
    xref.add_argument("dump", type=Path)
    xref.add_argument("address", type=_number)
    xref.add_argument(
        "--max-results", type=int, default=256, metavar="N",
        help="memory references to return (default: 256)",
    )
    xref.add_argument("--json", action="store_true")
    xref.set_defaults(handler=command_xref)

    threads = subparsers.add_parser(
        "threads", help="list retained CPU thread state"
    )
    threads.add_argument("dump", type=Path)
    threads.add_argument("--json", action="store_true")
    threads.set_defaults(handler=command_threads)

    thread = subparsers.add_parser(
        "thread", help="inspect one retained CPU thread by UID, index, or name"
    )
    thread.add_argument("dump", type=Path)
    thread.add_argument("thread", help="thread UID, table index, or unique name")
    thread.add_argument(
        "--stack-bytes",
        type=_number,
        default=256,
        metavar="N",
        help="maximum captured bytes to show from SP (default: 256)",
    )
    thread.add_argument(
        "--image",
        "--elf",
        "--self",
        dest="images",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="identity-checked original ELF or decrypted/fake SELF (repeatable)",
    )
    thread.add_argument("--sdk", type=Path, metavar="PATH")
    thread.add_argument(
        "--max-frames",
        type=int,
        default=32,
        metavar="N",
        help="maximum verified ARM EHABI frames when images are supplied (default: 32)",
    )
    thread.add_argument("--json", action="store_true")
    thread.set_defaults(handler=command_thread)

    waits = subparsers.add_parser(
        "waits", help="show retained thread waits, owners, chains, and cycles"
    )
    waits.add_argument("dump", type=Path)
    waits.add_argument("--json", action="store_true")
    waits.set_defaults(handler=command_waits)

    object_command = subparsers.add_parser(
        "object", help="inspect one retained kernel synchronization object"
    )
    object_command.add_argument("dump", type=Path)
    object_command.add_argument("uid", type=_number)
    object_command.add_argument("--json", action="store_true")
    object_command.set_defaults(handler=command_object)

    search = subparsers.add_parser(
        "search", help="search bytes retained in PT_LOAD segments"
    )
    search.add_argument("dump", type=Path)
    pattern = search.add_mutually_exclusive_group(required=True)
    pattern.add_argument("--u8", type=_number, metavar="VALUE")
    pattern.add_argument("--u16", type=_number, metavar="VALUE")
    pattern.add_argument("--u32", type=_number, metavar="VALUE")
    pattern.add_argument("--pointers-to", type=_number, metavar="ADDRESS")
    pattern.add_argument("--bytes", dest="bytes_pattern", metavar="HEX")
    pattern.add_argument("--string", metavar="TEXT")
    search.add_argument(
        "--encoding", default="utf-8", help="encoding used by --string (default: utf-8)"
    )
    search.add_argument(
        "--alignment",
        type=_number,
        metavar="N",
        help="address alignment; numeric searches default to their value width",
    )
    search.add_argument(
        "--max-results", type=int, default=64, metavar="N", help="results to return (default: 64)"
    )
    search.add_argument(
        "--include-aliases",
        action="store_true",
        help="search exact duplicate PT_LOAD aliases separately",
    )
    search.add_argument("--json", action="store_true")
    search.set_defaults(handler=command_search)

    gpu = subparsers.add_parser(
        "gpu", help="inspect one focused area of retained GPU state"
    )
    gpu_views = gpu.add_subparsers(dest="view", required=True)
    gpu_view_parsers = {}
    for view, help_text in (
        ("faults", "show BIF faults and fault-address evidence"),
        ("contexts", "show context owners and context-slot assignments"),
        ("pipeline", "show firmware-decoded pipeline progress"),
        ("resources", "show retained GPU resource pointers and named allocations"),
    ):
        view_parser = gpu_views.add_parser(view, help=help_text)
        view_parser.add_argument("dump", type=Path)
        view_parser.add_argument("--json", action="store_true")
        view_parser.set_defaults(handler=command_gpu)
        gpu_view_parsers[view] = view_parser
    gpu_view_parsers["pipeline"].add_argument(
        "--core",
        dest="cores",
        action="append",
        type=int,
        metavar="N",
        help="limit output to one GPU core (repeatable)",
    )
    gpu_view_parsers["pipeline"].add_argument(
        "--details",
        action="store_true",
        help="include retained counters, tiles, surfaces, and register fields",
    )
    gpu_view_parsers["resources"].add_argument(
        "--allocations",
        action="store_true",
        help="list every selected GPU-named allocation range",
    )

    registers = subparsers.add_parser("registers", help="print captured GPU registers")
    registers.add_argument("dump", type=Path)
    registers.add_argument("--matching", help="case-insensitive name/offset regex")
    registers.add_argument("--show-zero", action="store_true")
    registers.add_argument("--indexed", action="store_true", help="include undocumented indexed state")
    registers.add_argument("--json", action="store_true")
    registers.set_defaults(handler=command_registers)

    triage = subparsers.add_parser(
        "triage", help="group and facet multiple fixed-format GPUCRASH dumps"
    )
    triage.add_argument("dumps", nargs="+", type=Path)
    triage.add_argument("--json", action="store_true")
    triage.set_defaults(handler=command_triage)

    compare = subparsers.add_parser("compare", help="compare two GPU crash snapshots")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--json", action="store_true")
    compare.add_argument(
        "--summary-only",
        action="store_true",
        help="show section counts without individual changed records",
    )
    compare.add_argument(
        "--section",
        dest="sections",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "limit semantic output to verdict, faults, contexts, pipeline, "
            "resources, threads, or allocations (repeatable or comma-separated)"
        ),
    )
    compare.add_argument(
        "--registers",
        action="store_true",
        help="include changed GPU header, register, indexed, and trailer state",
    )
    compare.add_argument(
        "--matching",
        help="include register details matching a case-insensitive name/offset regex",
    )
    compare.add_argument(
        "--show-indexed",
        action="store_true",
        help="print individual indexed-state rows (implies --registers)",
    )
    compare.set_defaults(handler=command_compare)

    validate = subparsers.add_parser(
        "validate", help="check the fixed Vita GPUCRASH producer invariants"
    )
    validate.add_argument("dump", type=Path)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(handler=command_validate)

    memory = subparsers.add_parser("memory", help="hex-dump a captured virtual range")
    memory.add_argument("dump", type=Path)
    memory.add_argument("address", type=_number)
    memory.add_argument("size", type=_number)
    memory.add_argument("--output", type=Path, help="write bytes to a file instead of hex-dumping")
    memory.add_argument("--json", action="store_true")
    memory.set_defaults(handler=command_memory)

    activity = subparsers.add_parser(
        "activity", help="decode the optional GPU activity-ring snapshot"
    )
    activity.add_argument("dump", type=Path)
    activity.add_argument("--json", action="store_true")
    activity.add_argument("--raw-hex", action="store_true", help="include raw slot bytes in JSON")
    activity.add_argument("--show-empty", action="store_true")
    activity.set_defaults(handler=command_activity)

    segments = subparsers.add_parser(
        "segments", help="list captured memory segments and duplicate aliases"
    )
    segments.add_argument("dump", type=Path)
    segments.add_argument("--json", action="store_true")
    segments.add_argument("--unique", action="store_true", help="show one copy of exact aliases")
    segments.set_defaults(handler=command_segments)

    tty = subparsers.add_parser(
        "tty", help="extract the last captured Vita TTY/log buffer"
    )
    tty.add_argument("dump", type=Path)
    tty.add_argument(
        "--note",
        choices=("TTY_INFO2", "TTY_INFO", "all"),
        default="TTY_INFO2",
        help="TTY note to extract (default: TTY_INFO2)",
    )
    tty.add_argument("--json", action="store_true")
    tty.add_argument(
        "--raw", action="store_true", help="write the captured payload bytes verbatim"
    )
    tty.add_argument(
        "--preserve-ansi",
        action="store_true",
        help="retain console ANSI escape sequences in decoded text",
    )
    tty.add_argument(
        "--tail",
        type=int,
        metavar="N",
        help="show only the last N selected non-empty lines",
    )
    tty.add_argument(
        "--matching",
        metavar="REGEX",
        help="show only lines matching a case-insensitive regular expression",
    )
    tty.set_defaults(handler=command_tty)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except (ParseError, OSError) as exc:
        print(f"psp2-gpucrash-parse: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
