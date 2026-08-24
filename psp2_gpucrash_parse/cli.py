from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Iterable, Optional

from . import __version__
from .activity import GpuActivityInfo
from .core import CoreDump, ParseError
from .diagnosis import build_diagnosis
from .elfimage import enrich_diagnosis_with_elfs
from .gpu import GpuInfo, compare_gpu_info
from .metadata import SummaryInfo
from .registers import RegisterDatabase, decoded_fields
from .tty import TtyInfo


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


def _print_analysis_banner() -> None:
    border = "*" * 79
    print(border)
    print("*" + "GPUCRASH Analysis".center(77) + "*")
    print(border)


def _print_key_values(
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
    process = execution["process"] or {}
    values = [
        ("Analysis.Version", __version__),
        ("Analysis.Code", diagnosis["analysis_code"]),
        ("Analysis.Confidence", diagnosis["confidence"]),
        ("Dump.ImageBytes", f"0x{core.image_size:x}"),
        ("Dump.NoteCount", len(core.notes)),
        ("Dump.LoadSegments", len(core.loads)),
        ("Dump.UniqueLoadSegments", memory["unique_load_segment_count"]),
        ("GPU.CoreCount", info.core_count),
        ("GPU.Registers.Master", len(info.master_registers)),
        ("GPU.Registers.PerCore", len(info.per_core_registers)),
        ("GPU.Registers.Indexed", len(info.indexed_state)),
        ("GPU.CaptureFlags", _hex(info.capture_flags)),
        ("GPU.Snapshot.Complete", str(diagnosis["snapshot"]["complete"]).lower()),
        ("GPU.FaultCount", len(diagnosis["fault_sources"])),
        ("GPU.ErrorSignalCount", len(diagnosis["event_signals"])),
        ("GPU.LockupPhaseCount", len(diagnosis["lockup_phases"])),
        ("GPU.ActiveCoreCount", pipeline["active_core_count"]),
        ("GPU.PhaseActiveUnitCount", pipeline["phase_active_unit_count"]),
        ("GPU.LowActiveUnitCount", pipeline["active_unit_count"]),
        ("GPU.ContextOwnerCount", len(diagnosis["context_owners"])),
        ("GPU.ActivityState", diagnosis["activity_state"]),
        ("TTY.LineCount", diagnosis["log"]["line_count"]),
        ("Application.TitleId", application.get("title_id", "unavailable")),
        ("Application.Title", application.get("title_name", "unavailable")),
        ("ExternalProcesses.Count", external_processes.get("decoded_count", 0)),
        ("Process.Name", process.get("process_name", "unavailable")),
        ("Process.Id", _hex(process.get("process_id"))),
        ("CPU.ThreadCount", execution["thread_count"]),
        (
            "CPU.NonKernelExecutionThreads",
            len(execution["non_kernel_execution_thread_indices"]),
        ),
        (
            "CPU.ModuleCount",
            (execution["modules"] or {}).get("module_count", 0),
        ),
        ("Kernel.ObjectCount", kernel["object_count"]),
        ("Kernel.ActiveObjectCount", kernel["active_object_count"]),
        ("Kernel.ResolvedThreadWaits", kernel["resolved_thread_wait_count"]),
        ("Libraries.Count", libraries.get("decoded_name_count", 0)),
        ("MemoryBudget.PartitionCount", budgets.get("partition_count", 0)),
        ("SystemEventLog.RecordCount", event_log.get("record_count", 0)),
        ("OriginalElf.Requested", binary_images.get("requested_count", 0)),
        ("OriginalElf.Matched", binary_images.get("matched_count", 0)),
        ("OriginalElf.Rejected", binary_images.get("rejected_count", 0)),
    ]
    if summary is not None:
        values.append(("Dump.SummaryIncomplete", summary["incomplete_count"]))
    print("KEY_VALUES_STRING: 1\n")
    for key, value in values:
        print(f"    Key  : {key}")
        print(f"    Value: {value}")


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
        thread["index"] for thread in threads if thread["name"] == "SceGxmDisplayQueue"
    )
    selected = [thread for thread in threads if thread["index"] in selected_indices]
    print("\nCPU_REGISTER_CONTEXT_SELECTED:")
    if not selected:
        print("    No thread is executing outside SceLibKernel and no GXM display thread exists.")
    else:
        print("    Selection: non-SceLibKernel PC or SceGxmDisplayQueue thread.")
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
    print(f"    Captured sections     : {', '.join(captured) or 'none'}")
    print("    Header words          : " + " ".join(f"{word:08x}" for word in info.header_words))
    print("    Producer words 3-10   : " + " ".join(f"{word:08x}" for word in info.header_words[3:11]))
    print(f"    Trailer               : 0x{len(info.trailer):x} bytes")
    print(f"    Trailer SHA-256       : {hashlib.sha256(info.trailer).hexdigest()}")
    print("    Driver state words    : " + " ".join(f"{word:08x}" for word in trailer.driver_state_words))
    print("\nREGISTER_DEFINITION_SOURCES:")
    print("    Built-in SGX543 diagnostic register set")
    for source in database.sources:
        print(f"    {source}")


def _print_fault_records(diagnosis: dict) -> None:
    faults = diagnosis["fault_sources"]
    print(f"\nFAULT_RECORD_COUNT: {len(faults)}")
    if not faults:
        print("\nFAULTING_GPU_ADDRESS: none retained")
        print("    BIF requestor/type latch clear; BIF fault-address registers clear.")
        return
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
        print(f"    Hardware fault type    : {fault['fault_type']}")
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
                print(
                    f"      {name} ({location}, offset=0x{reference['offset']:04x}, "
                    f"value={values})"
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
    return ", ".join(f"{field['name']}={field['value']:#x}" for field in fields)


def _print_pipeline_register_state(
    info: GpuInfo, database: RegisterDatabase
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
            fields = _register_field_text(database, offset, False, value)
            suffix = f" [{fields}]" if fields else ""
            print(f"      core{index}={_hex(value)}{suffix}")


def _print_firmware_pipeline_diagnostics(diagnosis: dict) -> None:
    pipeline = diagnosis["firmware_pipeline"]
    master = pipeline["master"]
    print("\nFIRMWARE_PIPELINE_DIAGNOSTICS:")
    print(f"    Decoder               : {pipeline['decoder']}")
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
    for core in pipeline["cores"]:
        phases = ",".join(core["active_phases"]) or "none"
        units = ", ".join(item["name"] for item in core["active_units"]) or "none"
        print(
            f"\n    CORE_{core['core']}: phase-selector={_hex(core['phase_value'])} "
            f"unit-selector={_hex(core['unit_value'])}"
        )
        print(f"        Active phases     : {phases}")
        print(f"        Active units      : {units}")
        if not core["active_phases"]:
            print("        No vertex/fragment activity bit was retained for this core.")
            continue

        counters = core["draw_counters"]
        if counters:
            print("        Draw counters     :")
            for base in range(0, len(counters), 4):
                print(
                    "            "
                    + "  ".join(
                        f"{item['name']}={item['draw_number']}"
                        for item in counters[base : base + 4]
                    )
                )
            print("        Counter checks    :")
            for check in core["draw_counter_checks"]:
                values = ", ".join(
                    f"{name}={value}" for name, value in check["values"].items()
                )
                print(
                    f"            {check['name']}: "
                    f"{'aligned' if check['aligned'] else 'diverged'} ({values})"
                )

        tiles = core["tile_progress"]
        if tiles:
            print("        Tile progress     :")
            for base in range(0, len(tiles), 3):
                print(
                    "            "
                    + "  ".join(
                        f"{item['name']}=({item['tile_x']},{item['tile_y']})"
                        f"/{item['pixel_x']},{item['pixel_y']}px"
                        for item in tiles[base : base + 3]
                    )
                )
            for check in core["tile_checks"]:
                values = ", ".join(
                    f"({value[0]},{value[1]})" for value in check["values"]
                )
                print(
                    f"            {check['name']}: "
                    f"{'aligned' if check['aligned'] else 'diverged'} ({values})"
                )

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
                f"width={zls['width_pixels'] if zls['width_pixels'] is not None else 'n/a'} "
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
    print("    Tile positions are progress markers in 16x16-pixel units, not fault locations.")
    print("    MEM_BLK matches are direct address overlaps; they do not prove GPU-MMU ownership.")


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


def _print_lockup_state(diagnosis: dict) -> None:
    print("\nLOCKUP_PHASES:")
    phases = diagnosis["lockup_phases"]
    if not phases:
        print("    No TA/vertex or 3D/fragment phase latch is retained.")
    for phase in phases:
        print(
            f"    {phase['code'].upper():8} {phase['description']} "
            f"value={_hex(phase['register_value'])} mask={_hex(phase['mask'])}"
        )
    print("\nPIPELINE_COMPLETION_MARKERS:")
    markers = diagnosis["pipeline_markers"]
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
    if not diagnosis["context_assignments"]:
        print("    No context ID could be resolved consistently across the producer map.")
    for assignment in diagnosis["context_assignments"]:
        owner = assignment["owner"]
        print(
            f"    CID {assignment['context_id']} -> slot {owner['slot']} "
            f"{owner['process_name'] or '<unnamed>'} (map={assignment['status']})"
        )
    print("\nGPU_CONTEXT_MAP_RAW:")
    for index, row in enumerate(trailer.context_slot_map):
        print(f"    Row {index}: " + " ".join(f"{value:02x}" for value in row))
    print("    Values are owner-slot indices; ff means unused. Producer row roles remain unnamed.")


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
    scopes = ",".join(fault["scope"] for fault in diagnosis["fault_sources"]) or "none"
    addresses = ",".join(
        _hex(fault["gpu_virtual_address"]) for fault in diagnosis["fault_sources"]
    ) or "none"
    print(f"\nPROCESS_NAME:  {process_name}")
    print(f"PROCESS_ID:    {process_id}")
    print(f"GPU_CONTEXT_OWNER_NAME: {owner_name}")
    print(f"GPU_CONTEXT_OWNER_ID:   {owner_id}")
    print(f"FAULTING_GPU_SCOPE: {scopes}")
    print(f"FAULTING_GPU_VA:    {addresses}")
    print(f"FAILURE_BUCKET_ID:  {diagnosis['failure_bucket_id']}")
    print(f"FAILURE_HASH:       {diagnosis['failure_hash']}")


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
    print(f"\nANALYSIS_CONFIDENCE: {diagnosis['confidence'].upper()}\n")
    _print_key_values(core, info, diagnosis)
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
    _print_allocation_evidence(diagnosis)
    _print_capture_evidence(core, activity, diagnosis)
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
    if diagnosis is not None:
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


def _register_matches(name: str, offset: int, matcher: Optional[re.Pattern]) -> bool:
    return matcher is None or matcher.search(name) is not None or matcher.search(f"0x{offset:04x}") is not None


def command_registers(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    database = _database(args)
    info = _gpu_info(core)
    if info is None:
        raise ParseError("GPU_INFO note is unavailable")
    matcher = re.compile(args.matching, re.IGNORECASE) if args.matching else None
    print("Master registers:")
    for register in info.master_registers:
        definition = database.definition(register.offset, master=True)
        name = definition.display_name if definition else f"REG_{register.offset:04X}"
        if not args.show_zero and register.value == 0:
            continue
        if not _register_matches(name, register.offset, matcher):
            continue
        fields = decoded_fields(definition, register.value)
        suffix = " " + ", ".join(f"{item['name']}={item['value']:#x}" for item in fields) if fields else ""
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
        print(f"  {register.offset:04x} {values} {name}")
    if args.indexed:
        print("Indexed per-core state:")
        for item in info.indexed_state:
            if not args.show_zero and not any(item.values):
                continue
            values = " ".join(f"c{index}={value:08x}" for index, value in enumerate(item.values))
            print(f"  {item.offset:04x} {values}")
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
    result = {
        "left": {"path": str(left_core.path), "sha256": left_core.raw_file_sha256},
        "right": {"path": str(right_core.path), "sha256": right_core.raw_file_sha256},
        "gpu": comparison,
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
    print(f"Header changes: {len(comparison['header_changes'])}")
    for item in comparison["header_changes"]:
        print(f"  word{item['word']}: {item['left']:08x} -> {item['right']:08x}")
    print(f"Master register changes: {len(comparison['master_changes'])}")
    for item in comparison["master_changes"]:
        name = item["name"] or f"REG_{item['offset']:04X}"
        print(f"  {item['offset']:04x} {_hex(item['left'])} -> {_hex(item['right'])} {name}")
    print(f"Per-core register changes: {len(comparison['per_core_changes'])}")
    for item in comparison["per_core_changes"]:
        name = item["name"] or f"REG_{item['offset']:04X}"
        old = " ".join(f"{value:08x}" for value in (item["left"] or []))
        new = " ".join(f"{value:08x}" for value in (item["right"] or []))
        print(f"  {item['offset']:04x} [{old}] -> [{new}] {name}")
    print(f"Indexed state changes: {len(comparison['indexed_state_changes'])}")
    if args.show_indexed:
        for item in comparison["indexed_state_changes"]:
            old = " ".join(f"{value:08x}" for value in item["left"])
            new = " ".join(f"{value:08x}" for value in item["right"])
            print(f"  {item['index']:04x} [{old}] -> [{new}]")
    print(f"GPU context-owner changes: {len(comparison['context_owner_changes'])}")
    for item in comparison["context_owner_changes"]:
        old = item["left"]
        new = item["right"]
        old_text = (
            f"{_hex(old['process_id'])} {old['process_name']}" if old else "missing"
        )
        new_text = (
            f"{_hex(new['process_id'])} {new['process_name']}" if new else "missing"
        )
        print(f"  slot{item['slot']}: {old_text} -> {new_text}")
    print(
        f"Context-slot map changed: {int(comparison['context_slot_map_changed'])}; "
        f"driver-state words changed: {len(comparison['driver_state_changes'])}"
    )
    return 0


def command_memory(args: argparse.Namespace) -> int:
    core = CoreDump.read(args.dump)
    data = core.read_memory(args.address, args.size)
    block = core.memory_block_for(args.address)
    if args.output is not None:
        args.output.write_bytes(data)
        print(f"wrote {len(data)} bytes from 0x{args.address:08x} to {args.output}")
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
    if args.json:
        json.dump(
            {
                name: info.summary(preserve_ansi=args.preserve_ansi)
                for name, info in decoded
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        print()
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
    analyze.set_defaults(handler=command_analyze)

    registers = subparsers.add_parser("registers", help="print captured GPU registers")
    registers.add_argument("dump", type=Path)
    registers.add_argument("--matching", help="case-insensitive name/offset regex")
    registers.add_argument("--show-zero", action="store_true")
    registers.add_argument("--indexed", action="store_true", help="include undocumented indexed state")
    registers.set_defaults(handler=command_registers)

    compare = subparsers.add_parser("compare", help="compare two GPU register snapshots")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--json", action="store_true")
    compare.add_argument("--show-indexed", action="store_true")
    compare.set_defaults(handler=command_compare)

    memory = subparsers.add_parser("memory", help="hex-dump a captured virtual range")
    memory.add_argument("dump", type=Path)
    memory.add_argument("address", type=_number)
    memory.add_argument("size", type=_number)
    memory.add_argument("--output", type=Path, help="write bytes to a file instead of hex-dumping")
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
