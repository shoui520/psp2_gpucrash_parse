import io
import json
import struct
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from psp2_gpucrash_parse._des import decrypt_cbc
from psp2_gpucrash_parse.cli import main
from psp2_gpucrash_parse.core import ParseError
from psp2_gpucrash_parse.devkit import DevkitGpuDump, _owner_name_match, correlate_gpu_dump
from psp2_gpucrash_parse.gpu import PerCoreRegister
from psp2_gpucrash_parse.registers import RegisterDatabase
from tests.test_diagnosis import core_fixture


# Synthetic records encrypted independently using a standard DES provider.
# No captured application data or runtime crypto dependency is used here.
MASTER = bytes.fromhex(
    "d46584dd7a9d547601d186813b7c4556b68ea4dac8a81d22e861a8ef4c095fad0"
    "e37ea8a6469ef5fa985a935106fae6eec0053f4cc06cd0e46f64a372fff4e59e8"
    "e23022057c0b9d88ec0540a84aab1cf9cd40c1ba72d36e820983886b84b07a3f"
    "4146f94c2d6b8e9152dbc1280bb9fc85f85c2b571c327dc24a20cc8cfe594be186457b1d475b68"
)
CORE_DEBUG = bytes.fromhex(
    "d0409c97d4e53b9b8331bbc81515e1731aab86e971ae285d9b11f6db48aceb5e"
    "f18b76715758fdd4395cf3e6ae737898"
)
CORE = bytes.fromhex(
    "66656725ca1e0e60edc057c074c9393dcab8ac03a60cd9635be212b00d785704"
    "06903c412e20aa6bd9fc7968e92bdb06"
)
INDEXED = bytes.fromhex(
    "f1437f35f05f85f6cd85a3fd29207298ad2577082991e6b673ff4108e526802b9"
    "3b7771ba315910bfb83c70f9fe1b5e229f06cc8fb6114e6e2bbac51084ee6555"
    "704ee30cc21838f8293bdddddf0b53c41fbb9cfde6b87fa7acef57800c07ab3f"
    "93048d2d8148c726b0109109f85c31f4bb80a05e9cd600e91a5e99fa103a7cd"
    "26b35138a0f4832481aebc5d6d9b005941a852984d47e6243d938ace48e9ad13"
    "d13d61703fa1eb5695de362c2b006c86f3e52811325999332ef2b874701455952"
    "1a8448146906aef50033fad511d2d98aba5fe0b3f263640fcef30572f29ccf262"
    "0d28d5bae72b8da5fc6f998de437ab90731722c21f37e88415edc9e3063a1d74"
    "c3df5b0db7948f0205aef4809aeeacd24f31ae92564f8b6af42e08b48252443d"
    "da2f95625eba30693ba965cc91a85c3e3ae68ce1d11a6fe9d9a903db543c7d77"
    "f0dd7836f42744"
)
PROCESSES = bytes.fromhex(
    "115f6bb3891a77fd3e6a3f0fa23cc296b7a195e73e9fb492e042ca164a580c28"
    "72b8ddfb09fef2c982cf3ab4f5630bc25b1693650529e05c359cee0b4be15ea4"
    "e83cd465a094e2e0c4b85a00b402b0ae995f24d0b456a4f207dfbbf2ddaf0f2d"
    "f23e5a201c2afd4ecf16368c147392e23e673946f995a44d13f9181cfe039819"
    "a64c5d24d71a9ccffcf4277c021fe60f3315d494bc9d1dd735f672e38464169"
    "d1f5f1cab215816724664de9a03e234b23ac43e3e958f1331e16327f77946c46"
    "d3dffe7f5fb89e56af9b2d575b95377e3c54a76ee19ec93783dc7a171bedffcd8"
    "b329e387cd39725c9249e17676e0dde7089166802dde18ebd82c41838f431ff9"
)
TEXT = (
    b"\r: GPU Master Analysis:\n\r: GPU Core 1 Analysis:\n"
    b"\r:   - Process : SampleProcess (0x00000456)\n"
    b"\r: - Fragment: Active Process is : SampleProcess (0x00000456)\n"
    b"\r: Partial renders since boot: 12\n"
)


def segments():
    return [
        (0x81304007, TEXT.ljust(256, b"\0")),
        (0x81304006, struct.pack("<4I", 0x20002, 0x03680011, 12345, 0x42)),
        (0x81304001, MASTER), (0x81304002, CORE_DEBUG),
        (0x81304003, bytes.fromhex("5ce09d21a4440429")),
        (0x81304004, CORE), (0x81304005, INDEXED),
        (0x81304008, PROCESSES), (0x81304009, struct.pack("<4I", 10, 20, 30, 40)),
    ]


def container(records=None):
    body = b"".join(struct.pack("<2I", tag, len(data)) + data for tag, data in
                    (records if records is not None else segments()))
    return struct.pack("<4I", 0x01727A72, len(body) + 8, 0x01304000, len(body)) + body


def matching_info(native):
    info = native.register_state()
    trailer = bytearray(0xF8)
    for slot in range(8):
        struct.pack_into("<I", trailer, slot * 24, 0xFFFFFFFF)
    # Owner-slot indices intentionally differ from the data-master table.
    for slot, pid, name in ((0, 0x789, b"KernelProcess"), (1, 0x456, b"SampleProcess"),
                            (2, 0x123, b"SystemProcess")):
        struct.pack_into("<I", trailer, slot * 24, pid)
        trailer[slot * 24 + 4 : slot * 24 + 4 + len(name)] = name
    trailer[0xC0:0xE8] = b"\xff" * 40
    for row in range(5):
        trailer[0xC0 + row * 8 + 1] = 1
    header = list(info.header_words)
    header[2] |= 8  # The synthetic note now includes a captured owner table.
    return replace(info, trailer=bytes(trailer), header_words=tuple(header))


class DesTests(unittest.TestCase):
    def test_standard_des_known_answer(self):
        result = decrypt_cbc(bytes.fromhex("85e813540f0ab405"),
                             bytes.fromhex("133457799bbcdff1"), bytes(8))
        self.assertEqual(result.hex(), "0123456789abcdef")

    def test_cbc_multiple_blocks_and_iv(self):
        ciphertext = bytes.fromhex("0b1052b4b12ba3b393ca78065059b421c2adf0928d1aa662")
        self.assertEqual(decrypt_cbc(ciphertext, bytes.fromhex("133457799bbcdff1"),
                                     bytes.fromhex("0123456789abcdef")), bytes(range(24)))

    def test_rejects_unaligned_input_or_bad_key_iv(self):
        for data, key, iv in ((b"x", bytes(8), bytes(8)), (bytes(8), b"x", bytes(8)),
                              (bytes(8), bytes(8), b"x")):
            with self.subTest(data=data, key=key, iv=iv), self.assertRaises(ParseError):
                decrypt_cbc(data, key, iv)


class DevkitTests(unittest.TestCase):
    def test_process_names_respect_gpu_info_fixed_width_without_accepting_short_prefixes(self):
        self.assertEqual(_owner_name_match("LongProcessNameABCDE", "LongProcessNameABCDEF"), "truncated-prefix")
        self.assertIsNone(_owner_name_match("Sample", "SampleOtherProcess"))
        self.assertIsNone(_owner_name_match("DifferentProcessName", "SampleProcess"))

    def setUp(self):
        self.native = DevkitGpuDump.parse(container())

    def test_decrypts_all_table_types_and_both_key_groups(self):
        self.assertEqual(self.native.tables["master"][0], (0x4000, 3))
        self.assertEqual(self.native.tables["core_debug"][1], (0xC08, 0, 0x70001011, 0, 0))
        self.assertEqual(self.native.tables["master_debug"], [])
        self.assertEqual(self.native.tables["core"][0][2], 0x55555555)
        self.assertEqual(len(self.native.tables["indexed"]), 16)
        self.assertEqual(self.native.processes[7]["process_name"], "KernelProcess")
        self.assertEqual(self.native.processes[3]["process_id"], -1)
        self.assertEqual(self.native.render_state, {
            "vertex_frame": 10, "fragment_frame": 20, "vertex_scene": 30, "fragment_scene": 40,
        })
        self.assertEqual(self.native.segments[2]["padding_size"], 4)
        self.assertEqual(self.native.segments[3]["padding_size"], 4)

    def test_summary_decodes_fault_subunit_slots_and_firmware_text(self):
        report = self.native.summary(RegisterDatabase.load())
        fault = report["bif"]["fault_sources"][0]
        self.assertEqual(fault["requestor_details"][0]["subunit_name"], "Dependent Texture")
        self.assertEqual(fault["page_access_type"], "Entry access")
        pipe = report["firmware_pipeline"]["cores"][1]["usse_data_master_usage"][1]
        self.assertEqual(pipe["usage"], "PPPPPPPP........")
        self.assertTrue(report["firmware_pipeline"]["render_state"]["available"])
        self.assertEqual(report["firmware_pipeline"]["render_state"]["vertex_frame"], 10)
        self.assertEqual(report["native_diagnostics"]["partial_renders_since_boot"], 12)
        self.assertEqual(report["native_diagnostics"]["fault_processes"][0]["process_id"], 0x456)
        self.assertEqual(bytes.fromhex(report["analysis_text_raw_hex"]), self.native.text_bytes)
        self.assertNotIn("\r: ", report["analysis_text"])

    def test_correlates_identity_without_equating_process_table_indices(self):
        info = matching_info(self.native)
        original = tuple(info.master_registers)
        report = correlate_gpu_dump(self.native, core_fixture(), info)
        self.assertEqual(report["status"], "consistent")
        self.assertEqual(len(report["matching_processes"]), 3)
        self.assertEqual(report["fault_process_checks"][0]["status"], "match")
        self.assertEqual(report["register_comparisons"]["master"]["matching_count"], 16)
        self.assertEqual(tuple(info.master_registers), original)

    def test_detects_fault_mismatch_and_keeps_both_values(self):
        info = matching_info(self.native)
        rows = [PerCoreRegister(row.offset, (0, 0x71001011, 0, 0))
                if row.offset == 0xC08 else row for row in info.per_core_registers]
        report = correlate_gpu_dump(self.native, core_fixture(), replace(info, per_core_registers=rows))
        self.assertEqual(report["status"], "mismatch")
        change = report["register_comparisons"]["per_core"]["changed"][0]
        self.assertEqual(change["gpu_dump"][1], 0x70001011)
        self.assertEqual(change["gpu_info"][1], 0x71001011)

    def test_shared_system_processes_do_not_hide_fault_owner_mismatch(self):
        info = matching_info(self.native)
        trailer = bytearray(info.trailer)
        struct.pack_into("<I", trailer, 24, 0x999)
        report = correlate_gpu_dump(self.native, core_fixture(), replace(info, trailer=bytes(trailer)))
        self.assertEqual(len(report["matching_processes"]), 2)
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["fault_process_checks"][0]["status"], "mismatch")

    def test_correlation_uses_fault_scope_map_without_cross_core_fallback(self):
        info = matching_info(self.native)
        trailer = bytearray(info.trailer)
        trailer[0xC0 + 1] = 2  # Unrelated core 0 owns CID 1 differently.
        report = correlate_gpu_dump(self.native, core_fixture(), replace(info, trailer=bytes(trailer)))
        self.assertEqual(report["status"], "consistent")
        self.assertEqual(report["fault_process_checks"][0]["context_attribution"]["map_row"], 1)
        trailer[0xC0 + 8 + 1] = 0xFF  # The faulting core has no assignment.
        report = correlate_gpu_dump(self.native, core_fixture(), replace(info, trailer=bytes(trailer)))
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["fault_process_checks"][0]["status"], "unresolved")
        trailer[0xC0 + 8 + 1] = 2
        report = correlate_gpu_dump(self.native, core_fixture(), replace(info, trailer=bytes(trailer)))
        self.assertEqual(report["status"], "mismatch")

    def test_absent_owner_evidence_is_inconclusive(self):
        report = correlate_gpu_dump(self.native, core_fixture(), self.native.register_state())
        self.assertEqual(report["status"], "inconclusive")

    def test_preserves_unknown_segments_and_reports_missing_ones(self):
        records = segments()[:2] + [(0x813040FF, b"opaque00")]
        report = DevkitGpuDump.parse(container(records)).summary(RegisterDatabase())
        self.assertEqual(report["segments"][-1]["opaque_payload_hex"], b"opaque00".hex())
        self.assertIn("core", report["missing_segments"])
        self.assertEqual(report["register_tables"], {})

    def test_rejects_truncation_bad_sizes_duplicates_versions_and_table_corruption(self):
        bad_count = bytes.fromhex("803eead9508922a4")
        bad_padding = bytes.fromhex("c4cbdf976c1a8b27cb8c8ce7822d63d4")
        cases = [b"", container()[:-1], b"bad!" + container()[4:],
                 container(segments() + [segments()[1]]),
                 container([(0x81304006, struct.pack("<4I", 99, 0, 0, 0))]),
                 container(segments()[:2] + [(0x81304001, bad_count)]),
                 container(segments()[:2] + [(0x81304001, bad_padding)]),
                 container([(0x81304006, b"short000")]),
                 container(segments()[:2] + [(0x81304008, PROCESSES[:8])]),
                 container(segments()[:2] + [(0x81304009, bytes(8))]),
                 container(segments()[:2] + [(0x813040FF, b"unaligned")]),
                 container([])]
        for index, data in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ParseError):
                DevkitGpuDump.parse(data)

    def test_filename_time_is_only_a_hint_and_renaming_does_not_change_correlation(self):
        renamed = replace(self.native, path=Path("gpu_core_dump_2020_01_02_030405000006.bin"))
        report = correlate_gpu_dump(renamed, core_fixture(), matching_info(self.native))
        self.assertEqual(report["status"], "consistent")
        self.assertEqual(report["filename_timestamp_hint"]["iso_local"], "2020-01-02T03:04:05.000006")
        self.assertIsNone(report["filename_timestamp_hint"]["delta_from_gpu_info_microseconds"])

    def test_cli_standalone_json_text_and_register_listing(self):
        for options in (["--json"], ["--text"], ["--registers"]):
            output = io.StringIO()
            with patch("psp2_gpucrash_parse.cli.DevkitGpuDump.read", return_value=self.native), redirect_stdout(output):
                self.assertEqual(main(["gpu-dump", "gpu-dump.bin", *options]), 0)
            if options == ["--json"]:
                self.assertEqual(json.loads(output.getvalue())["header"]["format_version"], 0x20002)
            elif options == ["--text"]:
                self.assertTrue(output.getvalue().startswith("GPU Master Analysis:"))
                self.assertNotIn("DEVKIT_GPU_DUMP:", output.getvalue())
            else:
                self.assertIn("Register tables (sidecar snapshot):", output.getvalue())

    def test_analyze_sidecar_is_optional_and_kept_separate(self):
        core = core_fixture()
        info = matching_info(self.native)
        with patch("psp2_gpucrash_parse.cli.CoreDump.read", return_value=core), \
             patch("psp2_gpucrash_parse.cli._gpu_info", return_value=info), \
             patch("psp2_gpucrash_parse.cli.DevkitGpuDump.read", return_value=self.native) as read_native:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["analyze", "core.psp2dmp", "--json"]), 0)
            read_native.assert_not_called()
            self.assertNotIn("gpu_dump", json.loads(output.getvalue())["diagnosis"])
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["analyze", "core.psp2dmp", "--gpu-dump", "gpu-dump.bin", "--json"]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["diagnosis"]["gpu_dump"]["correlation"]["status"], "consistent")
            self.assertNotEqual(result["gpu_info"]["trailer_sha256"], result["diagnosis"]["gpu_dump"]["sha256"])

    def test_cli_correlation_and_missing_gpu_info_error(self):
        for has_info in (True, False):
            core = core_fixture()
            info = matching_info(self.native) if has_info else None
            with patch("psp2_gpucrash_parse.cli.CoreDump.read", return_value=core), \
                 patch("psp2_gpucrash_parse.cli._gpu_info", return_value=info), \
                 patch("psp2_gpucrash_parse.cli.DevkitGpuDump.read", return_value=self.native), \
                 redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                code = main(["gpu-dump", "gpu-dump.bin", "--core-dump", "core.psp2dmp", "--json"])
            if has_info:
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(out.getvalue())["correlation"]["status"], "consistent")
            else:
                self.assertEqual(code, 2)
                self.assertIn("GPU_INFO is unavailable", err.getvalue())

    def test_analyze_mismatch_is_reported_without_changing_primary_fault(self):
        info = matching_info(self.native)
        rows = [PerCoreRegister(row.offset, (0, 0x71001011, 0, 0))
                if row.offset == 0xC08 else row for row in info.per_core_registers]
        info = replace(info, per_core_registers=rows)
        with patch("psp2_gpucrash_parse.cli.CoreDump.read", return_value=core_fixture()), \
             patch("psp2_gpucrash_parse.cli._gpu_info", return_value=info), \
             patch("psp2_gpucrash_parse.cli.DevkitGpuDump.read", return_value=self.native):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["analyze", "core.psp2dmp", "--gpu-dump", "gpu-dump.bin", "--json"]), 0)
            diagnosis = json.loads(output.getvalue())["diagnosis"]
            self.assertEqual(diagnosis["fault_sources"][0]["gpu_virtual_address"], 0x71001000)
            self.assertEqual(diagnosis["gpu_dump"]["correlation"]["status"], "mismatch")
            self.assertIn("correlation is mismatch", diagnosis["limitations"][-1])
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["analyze", "core.psp2dmp", "--gpu-dump", "gpu-dump.bin", "--no-log"]), 0)
            self.assertIn("WARNING: supplied captures disagree", output.getvalue())


if __name__ == "__main__":
    unittest.main()
