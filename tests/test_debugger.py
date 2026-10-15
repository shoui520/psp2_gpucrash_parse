import struct
import unittest
from dataclasses import replace
from pathlib import Path

from psp2_gpucrash_parse.core import CoreDump, LoadSegment, Note
from psp2_gpucrash_parse.debugger import (
    address_report,
    compare_debug_state,
    load_debug_state,
    search_memory,
    thread_detail_report,
    thread_list_report,
)
from psp2_gpucrash_parse.registers import RegisterDatabase
from psp2_gpucrash_parse.diagnosis import build_diagnosis
from psp2_gpucrash_parse.gpu import GpuInfo
from tests.test_diagnosis import core_fixture, gpu_fixture


def debug_core(fault=False):
    core = core_fixture()
    core.notes.append(Note("GPU_INFO", 0x2000, gpu_fixture(fault=fault), 0))
    return core


def execution_core():
    core = debug_core()

    process = bytearray(0x5C)
    struct.pack_into("<IIII", process, 0, 1, 0, 0x123, 0)
    process[0x10:0x1B] = b"TestProcess"

    module_header = bytearray(0x50)
    struct.pack_into("<II", module_header, 0, 0x50, 0x1234)
    module_header[0x24:0x2E] = b"TestModule"
    struct.pack_into("<I", module_header, 0x4C, 1)
    module_segment = struct.pack("<5I", 0, 5, 0x81000000, 0x1000, 0x1000)
    module = (
        struct.pack("<2I", 1, 1)
        + module_header
        + module_segment
        + bytes(0x10)
    )

    thread = bytearray(0xA0)
    struct.pack_into("<II", thread, 0, len(thread), 0x40010001)
    thread[8:18] = b"RenderTask"
    struct.pack_into("<I", thread, 0x30, 8)
    struct.pack_into("<III", thread, 0x34, 0x81000001, 0x83000000, 0x2000)
    struct.pack_into("<ii", thread, 0x4C, 96, 100)
    struct.pack_into("<i", thread, 0x5C, 1)
    thread_info = struct.pack("<2I", 0x12, 1) + thread

    gpr = [0] * 16
    gpr[0] = 0xDEADBEEF
    gpr[13:16] = [0x83001000, 0x81000021, 0x81000041]
    register = struct.pack("<2I16II", 0x4C, 0x40010001, *gpr, 0x20)
    register_info = struct.pack("<2I", 0x11, 1) + register
    stack_info = struct.pack("<2I4I", 4, 1, 1, 0x40010001, 0x1000, 0x20)

    core.notes.extend(
        (
            Note("PROCESS_INFO", 0, bytes(process), 0),
            Note("MODULE_INFO", 0, module, 0),
            Note("THREAD_INFO", 0, thread_info, 0),
            Note("THREAD_REG_INFO", 0, register_info, 0),
            Note("STACK_INFO", 0, stack_info, 0),
        )
    )
    stack = struct.pack("<II", 0x81000081, 0xDEADBEEF) + bytes(0x38)
    core.loads.append(
        LoadSegment(1, 0, 0x83001000, 0, len(stack), len(stack), 6, 4, stack)
    )
    return core


def dormant_execution_core():
    core = execution_core()
    for index, note in enumerate(core.notes):
        if note.name != "THREAD_INFO":
            continue
        description = bytearray(note.description)
        struct.pack_into("<I", description, 8 + 0x30, 0x10)
        core.notes[index] = Note(
            note.name, note.note_type, bytes(description), note.segment_index
        )
        break
    return core


class DebuggerTests(unittest.TestCase):
    def test_compare_exposes_frame_scene_and_scope_changes_when_aggregate_is_unchanged(self):
        info = GpuInfo.parse(gpu_fixture())
        left = build_diagnosis(core_fixture(), info, RegisterDatabase())
        trailer = bytearray(info.trailer)
        struct.pack_into("<4I", trailer, 0xE8, 50, 49, 3, 2)
        trailer[0xC0 + 1] = 0xFF
        right = build_diagnosis(core_fixture(), replace(info, trailer=bytes(trailer)), RegisterDatabase())
        comparison = compare_debug_state(left, right)
        self.assertTrue(comparison["pipeline"]["render_state_changed"])
        self.assertEqual(comparison["pipeline"]["right_render_state"]["fragment_frame"], 49)
        self.assertFalse(comparison["pipeline"]["clipper_halt_check_changed"])
        scoped = comparison["context_scope_assignment_changes"]
        self.assertEqual([(item["scope"], item["context_id"]) for item in scoped["removed"]], [("core0", 1)])
        same = compare_debug_state(right, right)
        self.assertFalse(same["pipeline"]["render_state_changed"])
        self.assertEqual(same["context_scope_assignment_changes"], {"added": [], "removed": [], "changed": []})

    def test_fault_address_reports_metadata_without_claiming_bytes(self):
        report = address_report(
            debug_core(fault=True), RegisterDatabase(), 0x3E060000
        )
        self.assertEqual(report["status"], "metadata-only")
        self.assertEqual(len(report["gpu_fault_groups"]), 1)
        self.assertEqual(report["memory_preview"]["captured_size"], 0)
        self.assertIn("cannot be dereferenced", report["limitations"][0])

    def test_address_resolves_retained_thread_state(self):
        core = execution_core()
        report = address_report(
            core, RegisterDatabase(), 0xDEADBEEF, preview_size=16
        )
        self.assertEqual(report["threads"][0]["registers"], ["r0"])
        self.assertEqual(report["status"], "metadata-only")
        captured = address_report(
            core, RegisterDatabase(), 0x83001000, preview_size=0
        )
        self.assertEqual(captured["status"], "captured")

    def test_thread_list_and_detail_expose_registers_and_stack(self):
        core = execution_core()
        listing = thread_list_report(core, RegisterDatabase())
        self.assertEqual(listing["thread_count"], 1)
        self.assertEqual(listing["threads"][0]["pc_location"]["module_name"], "TestModule")

        report, _diagnosis = thread_detail_report(
            core, RegisterDatabase(), "RenderTask", stack_bytes=16
        )
        thread = report["thread"]
        self.assertEqual(thread["registers"]["pc"], 0x81000041)
        self.assertEqual(thread["stack_preview"]["captured_size"], 16)
        self.assertEqual(
            thread["stack_return_candidates"][0]["location"]["module_name"],
            "TestModule",
        )

    def test_dormant_register_record_is_not_reported_as_live_execution(self):
        core = dormant_execution_core()
        listing = thread_list_report(core, RegisterDatabase())
        thread = listing["threads"][0]
        self.assertEqual(thread["status_name"], "dormant")
        self.assertFalse(thread["register_context_live"])
        report, _diagnosis = thread_detail_report(
            core, RegisterDatabase(), "RenderTask", stack_bytes=16
        )
        self.assertEqual(report["thread"]["stack_return_candidates"], [])

    def test_memory_search_deduplicates_exact_alias_segments(self):
        core = debug_core()
        data = b"ABCD----ABCD"
        core.loads.extend(
            (
                LoadSegment(1, 0, 0x81000000, 0, len(data), len(data), 4, 4, data),
                LoadSegment(2, 0, 0x81000000, 0, len(data), len(data), 4, 4, data),
            )
        )
        report = search_memory(core, b"ABCD", "test")
        self.assertEqual(report["match_count"], 2)
        self.assertEqual(report["searched_segment_count"], 1)
        self.assertEqual(report["results"][0]["alias_segment_indices"], [1, 2])

    def test_postmortem_compare_reports_fault_and_pipeline_changes(self):
        _left_info, left = load_debug_state(
            debug_core(fault=True), RegisterDatabase()
        )
        _right_info, right = load_debug_state(debug_core(), RegisterDatabase())
        comparison = compare_debug_state(left, right)
        self.assertTrue(comparison["verdict"]["changed"])
        self.assertEqual(len(comparison["fault_groups"]["removed"]), 1)
        self.assertIn("core_changes", comparison["pipeline"])


if __name__ == "__main__":
    unittest.main()
