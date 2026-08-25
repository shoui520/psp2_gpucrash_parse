from __future__ import annotations

from collections import Counter

from .activity import GpuActivityInfo
from .core import CoreDump, ParseError
from .execution import ExecutionContext
from .gpu import GpuInfo
from .metadata import SummaryInfo


_REQUIRED_FIXED_NOTES = {
    "APP_INFO",
    "APP_LIST_INFO",
    "BUDGET_INFO",
    "BUILD_VER_INFO",
    "COREFILE_INFO",
    "DEVICE_INFO",
    "EVENTFLAG_INFO",
    "EVENT_LOG_INFO",
    "EXTNL_PROC_INFO",
    "FILE_INFO",
    "GPU_ACT_INFO",
    "GPU_INFO",
    "HW_INFO",
    "LIBRARY_INFO",
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
    "TTY_INFO",
    "TTY_INFO2",
}


def validate_core(core: CoreDump) -> dict:
    checks = []

    def check(
        code: str,
        passed: bool,
        description: str,
        severity: str = "error",
        **evidence,
    ) -> None:
        checks.append(
            {
                "code": code,
                "passed": bool(passed),
                "severity": severity,
                "description": description,
                "evidence": evidence,
            }
        )

    names = Counter(note.name for note in core.notes)
    missing = sorted(_REQUIRED_FIXED_NOTES - set(names))
    check(
        "fixed-note-set",
        not missing,
        "All notes emitted by the fixed Vita GPUCRASH producer are present.",
        missing=missing,
        required_count=len(_REQUIRED_FIXED_NOTES),
    )
    duplicates = {name: count for name, count in names.items() if count != 1}
    check(
        "unique-note-names",
        not duplicates,
        "Each fixed-format note name occurs once.",
        duplicates=duplicates,
    )

    gpu = None
    gpu_note = core.note("GPU_INFO")
    if gpu_note is not None:
        check(
            "gpu-info-size",
            len(gpu_note.description) == 0x3404,
            "GPU_INFO description is the producer's fixed 0x3404-byte export.",
            observed_size=len(gpu_note.description),
            expected_size=0x3404,
        )
        try:
            gpu = GpuInfo.parse(gpu_note.description)
        except ParseError as exc:
            check("gpu-info-parse", False, "GPU_INFO parses as version 5.", error=str(exc))
        else:
            check("gpu-info-parse", True, "GPU_INFO parses as version 5.")
            snapshot = gpu.snapshot_status()
            check(
                "gpu-capture-flags",
                snapshot["complete"],
                "All fixed GPU capture sections are marked valid.",
                capture_flags=gpu.capture_flags,
                missing_sections=snapshot["missing_sections"],
                unknown_flags=snapshot["unknown_flags"],
            )
            check(
                "gpu-table-counts",
                snapshot["canonical_3_68_counts"],
                "GPU register tables have the canonical 266/399/109 row counts.",
                observed={
                    "master": len(gpu.master_registers),
                    "per_core": len(gpu.per_core_registers),
                    "indexed": len(gpu.indexed_state),
                },
                unexpected=snapshot["unexpected_counts"],
            )
            trailer = gpu.decoded_trailer()
            trailer_summary = trailer.summary()
            check(
                "gpu-producer-padding",
                trailer_summary["producer_padding_size_matches"],
                "The export tail is the expected 0x2d0-byte producer padding extent.",
                observed_size=trailer_summary["producer_padding_size"],
                expected_size=trailer_summary["producer_padding_expected_size"],
                nonzero_bytes=trailer_summary["producer_padding_nonzero_bytes"],
            )
            invalid_slots = sorted(
                {
                    value
                    for row in trailer.context_slot_map
                    for value in row
                    if value != 0xFF and value >= len(trailer.context_owners)
                }
            )
            check(
                "gpu-context-slot-range",
                not invalid_slots,
                "Context-map values are owner slots 0..7 or 0xff.",
                invalid_values=invalid_slots,
            )

    activity_note = core.note("GPU_ACT_INFO")
    if activity_note is not None:
        try:
            activity = GpuActivityInfo.parse(activity_note.description)
        except ParseError as exc:
            check(
                "gpu-activity-fixed-empty",
                False,
                "GPU_ACT_INFO is the fixed empty v8 record emitted in Vita GPUCRASH dumps.",
                error=str(exc),
            )
        else:
            fixed_empty = (
                len(activity_note.description) == 12
                and activity.format_version == 8
                and activity.payload_size == 0
                and activity.activity_buffer is None
            )
            check(
                "gpu-activity-fixed-empty",
                fixed_empty,
                "GPU_ACT_INFO is the fixed empty v8 record emitted in Vita GPUCRASH dumps.",
                observed_size=len(activity_note.description),
                format_version=activity.format_version,
                payload_size=activity.payload_size,
            )

    execution = None
    try:
        execution = ExecutionContext.parse(core)
    except ParseError as exc:
        check(
            "execution-notes-parse",
            False,
            "Process, module, thread, register, and stack notes parse consistently.",
            error=str(exc),
        )
    else:
        check(
            "execution-notes-parse",
            True,
            "Process, module, thread, register, and stack notes parse consistently.",
        )
        threads = (
            {item.uid: item for item in execution.threads.threads}
            if execution.threads is not None
            else {}
        )
        registers = (
            {item.thread_uid: item for item in execution.registers.records}
            if execution.registers is not None
            else {}
        )
        stacks = (
            {item.thread_uid: item for item in execution.stacks.records}
            if execution.stacks is not None
            else {}
        )
        check(
            "thread-register-uids",
            set(threads) == set(registers),
            "THREAD_INFO and THREAD_REG_INFO contain the same thread UIDs.",
            missing_registers=sorted(set(threads) - set(registers)),
            orphan_registers=sorted(set(registers) - set(threads)),
        )
        check(
            "thread-stack-uids",
            set(threads) == set(stacks),
            "THREAD_INFO and STACK_INFO contain the same thread UIDs.",
            missing_stack_records=sorted(set(threads) - set(stacks)),
            orphan_stack_records=sorted(set(stacks) - set(threads)),
        )
        active_outside = []
        uncaptured_active = []
        for uid, thread in threads.items():
            registers_item = registers.get(uid)
            if registers_item is None:
                continue
            in_stack = thread.stack_base <= registers_item.sp < thread.stack_end
            if not in_stack and thread.status_name != "dormant":
                active_outside.append(uid)
            if in_stack and not any(
                segment.contains(registers_item.sp, 4) for segment in core.loads
            ):
                uncaptured_active.append(uid)
        check(
            "active-thread-sp-ranges",
            not active_outside,
            "Every non-dormant retained SP lies inside its THREAD_INFO stack range.",
            thread_uids=active_outside,
        )
        check(
            "active-thread-sp-capture",
            not uncaptured_active,
            "Every in-range retained SP has captured PT_LOAD bytes.",
            thread_uids=uncaptured_active,
        )
        invalid_segments = []
        if execution.modules is not None:
            for module in execution.modules.modules:
                for segment in module.segments:
                    if segment.size == 0 or segment.end < segment.start:
                        invalid_segments.append(
                            {"module": module.name, "segment": segment.number}
                        )
        check(
            "module-segment-ranges",
            not invalid_segments,
            "MODULE_INFO segment ranges are nonempty and do not wrap.",
            invalid_segments=invalid_segments,
        )

    nonexact_overlaps = []
    loads = sorted(core.loads, key=lambda item: (item.virtual_address, item.index))
    for index, left in enumerate(loads):
        left_end = left.virtual_address + left.file_size
        for right in loads[index + 1 :]:
            if right.virtual_address >= left_end:
                break
            exact_alias = (
                left.virtual_address == right.virtual_address
                and left.file_size == right.file_size
                and left.data == right.data
            )
            if not exact_alias:
                nonexact_overlaps.append([left.index, right.index])
    check(
        "load-overlaps",
        not nonexact_overlaps,
        "Overlapping captured PT_LOAD ranges are exact aliases.",
        nonexact_overlaps=nonexact_overlaps,
        exact_alias_group_count=len(core.load_alias_groups()),
    )

    summary_note = core.note("SUMMARY_INFO")
    if summary_note is not None:
        try:
            summary = SummaryInfo.parse(summary_note.description).summary(core.notes)
        except ParseError as exc:
            check(
                "summary-completion",
                False,
                "SUMMARY_INFO parses and reports no incomplete note writes.",
                error=str(exc),
            )
        else:
            check(
                "summary-completion",
                summary["incomplete_count"] == 0,
                "SUMMARY_INFO parses and reports no incomplete note writes.",
                incomplete_count=summary["incomplete_count"],
                entries=[
                    entry
                    for entry in summary["entries"]
                    if entry["status"]
                    not in ("complete", "not-present", "self-reference")
                ],
            )

    failed_errors = [
        item for item in checks if not item["passed"] and item["severity"] == "error"
    ]
    failed_warnings = [
        item for item in checks if not item["passed"] and item["severity"] == "warning"
    ]
    status = (
        "invalid"
        if failed_errors
        else "valid-with-warnings"
        if failed_warnings
        else "valid"
    )
    return {
        "schema_version": 1,
        "command": "validate",
        "path": str(core.path),
        "status": status,
        "valid": not failed_errors,
        "check_count": len(checks),
        "passed_count": sum(item["passed"] for item in checks),
        "failed_error_count": len(failed_errors),
        "failed_warning_count": len(failed_warnings),
        "checks": checks,
        "scope": (
            "Fixed Vita GPUCRASH producer invariants established by the existing dump corpus and firmware layout."
        ),
    }
