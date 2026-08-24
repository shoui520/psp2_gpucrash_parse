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


def gpu_fixture(fault=False, event_value=0, residual_fault=False):
    header = [5, 0x10003, 0x3F, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 4, 0]
    words = header[:]
    address = 0x3E060041 if fault or residual_fault else 0
    words += [0x4C04, 0x80000, 0x4C08, 0, 0x4408, address & 0xFFFFF000]
    statuses = [0x80000, 0x80000, 0xD0400 if fault else 0x80000, 0x80000]
    faults = [0, 0, address, 0]
    words += [0x0C04, *statuses]
    words += [0x0C08, *faults]
    words += [0x0C38, 0x41996000, 0x41996000, 0x41996000, 0x41996000]
    words += [0x012C, event_value, event_value, event_value, event_value]

    trailer = bytearray(0xF8)
    for slot in range(8):
        struct.pack_into("<I", trailer, slot * 0x18, 0xFFFFFFFF)
    struct.pack_into("<I", trailer, 0x18, 0x172901)
    trailer[0x1C:0x29] = b"SampleProcess"
    trailer[0xC0:0xE8] = b"\xff" * 0x28
    for row in range(5):
        trailer[0xC0 + row * 8 + 1] = 1
    return struct.pack(f"<{len(words)}I", *words) + trailer


def core_fixture(log=b""):
    notes = []
    if log:
        tty = struct.pack("<3I", 19, 0, len(log)) + log
        notes.append(Note("TTY_INFO2", 0x1031, tty, 0))
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
        self.assertEqual(fault["gpu_virtual_address"], 0x3E060000)
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
        self.assertIn("KEY_VALUES_STRING: 1", text)
        self.assertIn("FAULT_RECORD_0:", text)
        self.assertIn("Arg1: 3e060000", text)
        self.assertIn("BIF_REGISTER_STATE:", text)
        self.assertIn("PIPELINE_REGISTER_STATE:", text)
        self.assertIn("FIRMWARE_PIPELINE_DIAGNOSTICS:", text)
        self.assertIn("GPU_CONTEXT_MAP_RAW:", text)
        self.assertIn("FAILURE_BUCKET_ID:", text)


if __name__ == "__main__":
    unittest.main()
