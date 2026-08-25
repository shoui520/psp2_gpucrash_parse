import argparse
import io
import struct
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from psp2_gpucrash_parse.cli import _print_diagnosis
from psp2_gpucrash_parse.core import CoreDump, Note
from psp2_gpucrash_parse.diagnosis import build_diagnosis
from psp2_gpucrash_parse.gpu import GpuInfo
from psp2_gpucrash_parse.registers import RegisterDatabase


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
