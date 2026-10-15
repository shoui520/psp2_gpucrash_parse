import argparse
import io
import json
import struct
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from psp2_gpucrash_parse.cli import _print_diagnosis, _print_firmware_pipeline_diagnostics, main
from psp2_gpucrash_parse.core import CoreDump, Note
from psp2_gpucrash_parse.diagnosis import _fault_groups, build_diagnosis
from psp2_gpucrash_parse.gpu import GpuInfo, PerCoreRegister, _fault_source
from psp2_gpucrash_parse.registers import RegisterDatabase
from tests.test_pipeline import clipper_fixture


def gpu_fixture(
    fault=False,
    event_value=0,
    residual_fault=False,
    fault_cores=None,
    fault_address=0x3E060000,
    resource_pointer=None,
):
    active_cores = set(fault_cores if fault_cores is not None else ((2,) if fault else ()))
    statuses = [0xD0400 if index in active_cores else 0x80000 for index in range(4)]
    faults = [
        fault_address | (((4 + index) & 0x1F) << 4) | 1
        if index in active_cores
        else 0
        for index in range(4)
    ]
    if residual_fault:
        faults[2] = fault_address | 0x41

    master_rows = [
        (0x4C04, 0x80000),
        (0x4C08, 0),
        (0x4408, fault_address if active_cores or residual_fault else 0),
    ]
    per_core_rows = [
        (0x0C04, statuses),
        (0x0C08, faults),
        (0x0C38, [0x41996000] * 4),
        (0x012C, [event_value] * 4),
    ]
    if resource_pointer is not None:
        master_rows.append((0x4238, resource_pointer))
        per_core_rows.append((0x0238, [resource_pointer] * 4))
        per_core_rows.append((0x0A5C, [resource_pointer + 0x1000] * 4))

    header = [
        5,
        0x10003,
        0x3F,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4,
        len(master_rows),
        len(per_core_rows),
        0,
    ]
    words = header[:]
    for offset, value in master_rows:
        words.extend((offset, value))
    for offset, values in per_core_rows:
        words.extend((offset, *values))

    trailer = bytearray(0xF8)
    for slot in range(8):
        struct.pack_into("<I", trailer, slot * 0x18, 0xFFFFFFFF)
    struct.pack_into("<I", trailer, 0x18, 0x172901)
    trailer[0x1C:0x29] = b"SampleProcess"
    trailer[0xC0:0xE8] = b"\xff" * 0x28
    for row in range(5):
        trailer[0xC0 + row * 8 + 1] = 1
    return struct.pack(f"<{len(words)}I", *words) + trailer


def core_fixture(log=b"", memory_block=None):
    notes = []
    if log:
        tty = struct.pack("<3I", 19, 0, len(log)) + log
        notes.append(Note("TTY_INFO2", 0x1031, tty, 0))
    if memory_block is not None:
        name, base, size = memory_block
        encoded_name = name.encode("ascii")[:31] + b"\0"
        encoded_name = encoded_name.ljust(32, b"\0")
        record = struct.pack("<II", 0, 0x40010001) + encoded_name
        record += struct.pack("<8I", 0, base, size, 0, 0, size, 0, 0)
        notes.append(Note("MEM_BLK_INFO", 0x1021, struct.pack("<II", 1, 1) + record, 0))
    return CoreDump(
        path=Path("test.psp2dmp"),
        compressed=False,
        raw_file_sha256="",
        image_sha256="",
        image_size=0,
        elf_type=4,
        machine=40,
        flags=0,
        notes=notes,
        loads=[],
    )


class DiagnosisTests(unittest.TestCase):
    def test_gpu_views_expose_scoped_contexts_and_frame_state_in_compact_json_and_text(self):
        info = GpuInfo.parse(gpu_fixture())
        trailer = bytearray(info.trailer)
        struct.pack_into("<4I", trailer, 0xE8, 50, 49, 3, 2)
        info = replace(info, trailer=bytes(trailer))
        diagnosis = build_diagnosis(core_fixture(), info, RegisterDatabase())
        for view in ("contexts", "pipeline"):
            for json_output in (False, True):
                output = io.StringIO()
                with patch("psp2_gpucrash_parse.cli.CoreDump.read", return_value=core_fixture()), \
                     patch("psp2_gpucrash_parse.cli.load_debug_state", return_value=(info, diagnosis)), \
                     redirect_stdout(output):
                    self.assertEqual(main(["gpu", view, "core.psp2dmp", *(["--json"] if json_output else [])]), 0)
                if json_output:
                    report = json.loads(output.getvalue())
                    if view == "contexts":
                        self.assertEqual(report["context_slot_map_scopes"], ["core0", "core1", "core2", "core3", "master"])
                        self.assertEqual(len(report["context_scope_assignments"]), 5)
                    else:
                        self.assertEqual(report["firmware_pipeline"]["render_state"]["fragment_frame"], 49)
                        self.assertIn("clipper_halt_check", report["firmware_pipeline"])
                elif view == "contexts":
                    self.assertIn("core2  CID 1 -> slot 1", output.getvalue())
                    self.assertNotIn("remain unnamed", output.getvalue())
                else:
                    self.assertIn("frame=50 scene=3", output.getvalue())
                    self.assertIn("frame=49 scene=2", output.getvalue())

    def test_text_keeps_inactive_counter_values_without_false_divergence_and_labels_msaa_units(self):
        info = GpuInfo.parse(gpu_fixture())
        info = replace(info, per_core_registers=[
            *info.per_core_registers, PerCoreRegister(0xAA4, (0xAAAAAAAA,) * 4),
            PerCoreRegister(0x480, (0x100000 | (19 << 3),) * 4),
            PerCoreRegister(0x42C, (2,) * 4),
        ], indexed_state=[PerCoreRegister(0, (2,) * 4), PerCoreRegister(2, (0xFFFFFFFF,) * 4),
                          PerCoreRegister(0x26, (99,) * 4), PerCoreRegister(0x12, (3 << 16,) * 4)])
        diagnosis = build_diagnosis(core_fixture(), info, RegisterDatabase())
        output = io.StringIO()
        with redirect_stdout(output):
            _print_firmware_pipeline_diagnostics(diagnosis)
        text = output.getvalue()
        self.assertIn("MTE=98(counter=99) [inactive]", text)
        self.assertIn("TA MTE/VDM draw counters: not-comparable", text)
        self.assertNotIn("diverged", text)
        self.assertIn("width=320px samples=640 msaa=4x", text)

    def test_compare_summary_and_details_include_frame_scene_changes(self):
        left, right = core_fixture(), core_fixture()
        left.path, right.path = Path("left.psp2dmp"), Path("right.psp2dmp")
        left.notes.append(Note("GPU_INFO", 0x2000, gpu_fixture(), 0))
        data = bytearray(gpu_fixture())
        struct.pack_into("<4I", data, len(data) - 0xF8 + 0xE8, 50, 49, 3, 2)
        right.notes.append(Note("GPU_INFO", 0x2000, bytes(data), 0))
        cores = {left.path: left, right.path: right}
        for options in (["--json", "--summary-only"], ["--json"], []):
            output = io.StringIO()
            with patch("psp2_gpucrash_parse.cli.CoreDump.read", side_effect=lambda path: cores[Path(path)]), redirect_stdout(output):
                self.assertEqual(main(["compare", str(left.path), str(right.path), *options]), 0)
            if "--json" in options:
                report = json.loads(output.getvalue())
                self.assertTrue(report["postmortem"]["pipeline"]["render_state_changed"])
                if "--summary-only" not in options:
                    self.assertEqual(report["postmortem"]["pipeline"]["right_render_state"]["vertex_frame"], 50)
                    self.assertIn("context_scope_assignment_changes", report["postmortem"])
            else:
                self.assertIn("vertex_frame: 0 -> 50", output.getvalue())
                self.assertIn("frame/scene=true", output.getvalue())

    def test_fault_ownership_uses_source_row_even_when_other_cores_disagree(self):
        info = GpuInfo.parse(gpu_fixture(fault=True, fault_cores=(0, 2)))
        trailer = bytearray(info.trailer)
        struct.pack_into("<I", trailer, 2 * 24, 0x234567)
        trailer[2 * 24 + 4:2 * 24 + 4 + 12] = b"OtherProcess"
        trailer[0xC0 + 1] = 2
        info = replace(info, trailer=bytes(trailer))
        diagnosis = build_diagnosis(core_fixture(), info, RegisterDatabase())
        self.assertEqual([item["context_attribution"]["owner"]["process_name"]
                          for item in diagnosis["fault_sources"]], ["OtherProcess", "SampleProcess"])
        self.assertEqual(len(diagnosis["fault_groups"]), 2)
        self.assertEqual([item["context_attribution"]["map_row"] for item in diagnosis["fault_sources"]], [0, 2])
        self.assertEqual(diagnosis["context_assignments"], [])  # Aggregate is ambiguous.
        self.assertEqual(len(diagnosis["context_scope_assignments"]), 5)

    def test_unassigned_fault_scope_does_not_borrow_another_scope_owner(self):
        info = GpuInfo.parse(gpu_fixture(fault=True))
        trailer = bytearray(info.trailer)
        trailer[0xC0 + 2 * 8 + 1] = 0xFF
        diagnosis = build_diagnosis(core_fixture(), replace(info, trailer=bytes(trailer)), RegisterDatabase())
        self.assertIsNone(diagnosis["fault_sources"][0]["context_attribution"]["owner"])
        self.assertEqual(diagnosis["fault_sources"][0]["context_attribution"]["status"], "unassigned")

    def test_master_fault_ownership_uses_last_map_row(self):
        info = GpuInfo.parse(gpu_fixture())
        trailer = bytearray(info.trailer)
        trailer[0xC0 + 4 * 8 + 1] = 0xFF
        rows = [replace(row, value=0x10002 if row.offset == 0x4C04 else 0x70001001)
                if row.offset in (0x4C04, 0x4C08) else row for row in info.master_registers]
        diagnosis = build_diagnosis(core_fixture(), replace(info, trailer=bytes(trailer), master_registers=rows), RegisterDatabase())
        fault = diagnosis["fault_sources"][0]
        self.assertEqual(fault["scope"], "master")
        self.assertEqual(fault["context_attribution"]["map_row"], 4)
        self.assertIsNone(fault["context_attribution"]["owner"])

    def test_clipper_signature_reports_candidate_and_scoped_ta_owner_not_proven_defect(self):
        info = replace(clipper_fixture(), trailer=GpuInfo.parse(gpu_fixture()).trailer)
        diagnosis = build_diagnosis(core_fixture(), info, RegisterDatabase())
        self.assertEqual(diagnosis["classification"], "gpu-clipper-halt-signature")
        self.assertEqual(diagnosis["classification_confidence"], "medium")
        self.assertEqual(diagnosis["root_cause_confidence"], "low")
        signature = diagnosis["firmware_signatures"][0]
        self.assertEqual(signature["context_attribution"]["owner"]["process_name"], "SampleProcess")
        self.assertEqual(signature["context_attribution"]["map_row"], 2)
        self.assertIn("not a confirmed hardware defect", diagnosis["explanation"])
        self.assertEqual(diagnosis["fault_sources"], [])
        output = io.StringIO()
        with redirect_stdout(output):
            _print_diagnosis(core_fixture(), info, RegisterDatabase(), None, diagnosis,
                             argparse.Namespace(no_log=True, log_lines=0))
        self.assertIn("Clipper halt signature: matched", output.getvalue())
        self.assertIn("draw=100 counter=101 ctx=1", output.getvalue())
        self.assertIn("TA process: SampleProcess", output.getvalue())

    def test_real_bif_fault_takes_precedence_over_clipper_selector_bit(self):
        info = clipper_fixture()
        info.per_core_registers[0] = replace(info.per_core_registers[0], values=(0, 0, 0x10002, 0))
        diagnosis = build_diagnosis(core_fixture(), info, RegisterDatabase())
        self.assertEqual(diagnosis["classification"], "gpu-bif-access-fault")
        self.assertEqual(diagnosis["firmware_signatures"], [])

    def test_fault_groups_keep_entry_directory_distinct_and_preserve_subunits(self):
        def evidence(scope, status, subunit):
            fault = _fault_source(scope, status, 0x70001001 | (subunit << 4))
            return {**fault, "context_attribution": {"owner": None},
                    "gpu_virtual_address": fault["address"], "null_address": False,
                    "page_directory_base": 0x41000000}

        groups = _fault_groups([
            evidence("core0", 0x10002, 0), evidence("core1", 0x50002, 0),
            evidence("core2", 0x10002, 2),
        ])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["scopes"], ["core0", "core2"])
        self.assertEqual(groups[0]["page_access_type"], "Entry access")
        self.assertEqual(groups[1]["page_access_type"], "Directory access")
        self.assertEqual([item["subunit_name"] for item in groups[0]["requestor_details"]],
                         ["Dependent Texture", "Non-Dependent Texture"])

    def test_attributes_bif_fault_to_context_owner(self):
        core = core_fixture(b"GPU watchdog timeout\nlast submission\n")
        info = GpuInfo.parse(gpu_fixture(fault=True))
        diagnosis = build_diagnosis(core, info, RegisterDatabase())
        fault = diagnosis["fault_sources"][0]
        self.assertEqual(diagnosis["classification"], "gpu-bif-access-fault")
        self.assertEqual(diagnosis["confidence"], "high")
        self.assertEqual(diagnosis["classification_confidence"], "high")
        self.assertEqual(diagnosis["root_cause_confidence"], "low")
        self.assertEqual(fault["gpu_virtual_address"], 0x3E060000)
        self.assertEqual(fault["requestor_details"][0]["code"], "data-cache")
        self.assertEqual(
            fault["requestor_details"][0]["resource_class"],
            "PDS DMA or shader data",
        )
        self.assertEqual(fault["page_directory_base"], 0x41996000)
        self.assertEqual(fault["register_references"][0]["offset"], 0x4408)
        self.assertEqual(
            fault["register_references"][0]["name"], "EUR_CR_MASTER_ISP_RGN_BASE"
        )
        self.assertEqual(
            fault["context_attribution"]["owner"]["process_name"], "SampleProcess"
        )
        self.assertIn("watchdog timeout", diagnosis["log"]["noteworthy_lines"][0])

    def test_reports_undetermined_hang_without_error_latches(self):
        diagnosis = build_diagnosis(
            core_fixture(), GpuInfo.parse(gpu_fixture()), RegisterDatabase()
        )
        self.assertEqual(diagnosis["classification"], "undetermined-gpu-hang")
        self.assertEqual(diagnosis["confidence"], "low")

    def test_surfaces_dpm_memory_pressure_event(self):
        diagnosis = build_diagnosis(
            core_fixture(),
            GpuInfo.parse(gpu_fixture(event_value=0x02000000)),
            RegisterDatabase(),
        )
        self.assertEqual(diagnosis["classification"], "gpu-dpm-memory-pressure")
        self.assertEqual(
            {signal["scope"] for signal in diagnosis["event_signals"]},
            {"core0", "core1", "core2", "core3"},
        )

    def test_does_not_call_residual_fault_register_high_confidence(self):
        diagnosis = build_diagnosis(
            core_fixture(),
            GpuInfo.parse(gpu_fixture(residual_fault=True)),
            RegisterDatabase(),
        )
        self.assertEqual(
            diagnosis["classification"], "gpu-bif-fault-register-residue"
        )
        self.assertEqual(diagnosis["confidence"], "medium")

    def test_groups_same_fault_observed_by_multiple_cores(self):
        diagnosis = build_diagnosis(
            core_fixture(),
            GpuInfo.parse(gpu_fixture(fault=True, fault_cores=(0, 2))),
            RegisterDatabase(),
        )
        self.assertEqual(len(diagnosis["fault_sources"]), 2)
        self.assertEqual(len(diagnosis["fault_groups"]), 1)
        group = diagnosis["fault_groups"][0]
        self.assertEqual(group["scopes"], ["core0", "core2"])
        self.assertEqual(group["sidebands"], [4, 6])
        self.assertTrue(group["multiple_scope_observation"])
        self.assertIn("observed on core0, core2", diagnosis["headline"])

    def test_promotes_null_gpu_page_fault_to_headline(self):
        diagnosis = build_diagnosis(
            core_fixture(),
            GpuInfo.parse(gpu_fixture(fault=True, fault_address=0)),
            RegisterDatabase(),
        )
        self.assertTrue(diagnosis["fault_groups"][0]["null_address"])
        self.assertIn("null-page", diagnosis["headline"])

    def test_correlates_retained_resource_pointers_with_memory_blocks(self):
        base = 0x70100000
        diagnosis = build_diagnosis(
            core_fixture(memory_block=("GXM command allocation", base, 0x4000)),
            GpuInfo.parse(gpu_fixture(resource_pointer=base + 0x124)),
            RegisterDatabase(),
        )
        evidence = diagnosis["resource_pointers"]
        self.assertEqual(evidence["pointer_count"], 2)
        self.assertEqual(evidence["allocation_match_count"], 2)
        control = next(
            pointer
            for pointer in evidence["pointers"]
            if pointer["code"] == "vdm-control-stream"
        )
        self.assertTrue(control["master_observed"])
        self.assertEqual(control["core_indices"], [0, 1, 2, 3])
        self.assertEqual(control["allocation"]["offset"], 0x124)

    def test_verbose_report_keeps_debugger_sections_and_raw_arguments(self):
        core = core_fixture()
        info = GpuInfo.parse(gpu_fixture(fault=True))
        database = RegisterDatabase()
        diagnosis = build_diagnosis(core, info, database)
        output = io.StringIO()
        args = argparse.Namespace(no_log=True, log_lines=0)
        with redirect_stdout(output):
            _print_diagnosis(core, info, database, None, diagnosis, args)
        text = output.getvalue()
        self.assertIn("GPUCRASH Analysis", text)
        self.assertIn("GPUCRASH analysis summary:", text)
        self.assertIn("\nGPU capture:\n", text)
        self.assertIn("Fault groups: 1", text)
        self.assertNotIn("KEY_VALUES_STRING", text)
        self.assertIn("CLASSIFICATION_CONFIDENCE: HIGH", text)
        self.assertIn("ROOT_CAUSE_CONFIDENCE: LOW", text)
        self.assertIn("FAULT_GROUP_0:", text)
        self.assertIn("FAULT_RECORD_0:", text)
        self.assertIn("Arg1: 3e060000", text)
        self.assertIn("BIF_REGISTER_STATE:", text)
        self.assertIn("PIPELINE_REGISTER_STATE:", text)
        self.assertIn("FIRMWARE_PIPELINE_DIAGNOSTICS:", text)
        self.assertIn("GPU_CONTEXT_MAP_RAW:", text)
        self.assertIn("FAILURE_BUCKET_ID:", text)


if __name__ == "__main__":
    unittest.main()
