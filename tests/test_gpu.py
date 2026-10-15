import struct
import unittest
from dataclasses import replace

from psp2_gpucrash_parse.gpu import GpuInfo, GpuTrailer, _fault_source, compare_gpu_info
from psp2_gpucrash_parse.registers import RegisterDatabase


def fixture(
    master_fault=0,
    core_fault=(0, 0, 0, 0),
    environment_words=(0, 0, 0, 0, 0, 0, 0, 0),
    indexed_selector=7,
):
    header = [5, 0x10003, 0x3F, *environment_words, 4, 2, 2, 1]
    words = header[:]
    words += [0x4C04, 0x80000, 0x4C08, master_fault]
    words += [0x0C04, 0x80000, 0x80000, 0x80000, 0x80000]
    words += [0x0C08, *core_fault]
    words += [indexed_selector, 1, 2, 3, 4]
    words += [0xDEADBEEF]
    return struct.pack(f"<{len(words)}I", *words)


class GpuInfoTests(unittest.TestCase):
    def test_render_state_names_words_and_requires_capture_flag_and_complete_trailer(self):
        trailer = bytearray(0xF8)
        struct.pack_into("<4I", trailer, 0xE8, 700, 699, 3, 3)
        info = GpuInfo.parse(fixture()[:-4] + trailer)
        state = info.render_state()
        self.assertTrue(state["available"])
        self.assertEqual((state["vertex_frame"], state["fragment_frame"],
                          state["vertex_scene"], state["fragment_scene"]), (700, 699, 3, 3))
        self.assertEqual(info.summary(RegisterDatabase())["render_state"], state)
        self.assertEqual(info.decoded_trailer().summary()["driver_state"][0]["name"], "vertex_frame")
        header = list(info.header_words)
        header[2] &= ~0x10
        missing_flag = replace(info, header_words=tuple(header)).render_state()
        self.assertFalse(missing_flag["available"])
        self.assertIsNone(missing_flag["vertex_frame"])
        self.assertEqual(missing_flag["raw_words"], [700, 699, 3, 3])
        self.assertFalse(GpuInfo.parse(fixture()).render_state()["available"])
        self.assertTrue(replace(info, trailer=bytes(0xF8)).render_state()["available"])

    def test_context_map_resolves_exact_scope_without_other_row_fallback(self):
        trailer = bytearray(0xF8)
        for slot in range(8):
            struct.pack_into("<I", trailer, slot * 24, 0xFFFFFFFF)
        struct.pack_into("<I", trailer, 0, 0x123)
        struct.pack_into("<I", trailer, 24, 0x456)
        trailer[0xC0:0xE8] = b"\xff" * 40
        for row, slot in ((0, 0), (1, 1), (4, 1)):
            trailer[0xC0 + row * 8 + 2] = slot
        info = GpuInfo.parse(fixture()[:-4] + trailer)
        self.assertEqual(info.context_attribution(2)["status"], "ambiguous")
        for scope, row, pid in (("core0", 0, 0x123), ("core1", 1, 0x456), ("master", 4, 0x456)):
            with self.subTest(scope=scope):
                owner = info.context_attribution(2, scope)
                self.assertEqual(owner["map_row"], row)
                self.assertEqual(owner["owner"]["process_id"], pid)
        self.assertIsNone(info.context_attribution(2, "core2")["owner"])
        self.assertEqual(info.context_attribution(2, "core2")["status"], "unassigned")
        for cid, scope in ((8, "core0"), (-1, "master"), (2, "core4")):
            self.assertEqual(info.context_attribution(cid, scope)["status"], "out-of-range")
        trailer[0xC0 + 2] = 9
        self.assertEqual(replace(info, trailer=bytes(trailer)).context_attribution(2, "core0")["status"], "invalid-slot")
        header = list(info.header_words)
        header[2] &= ~8
        self.assertIsNone(replace(info, header_words=tuple(header)).context_attribution(2, "core0")["owner"])

    def test_bif_condition_is_an_enum_with_separate_directory_access_bit(self):
        for raw, condition, access in (
            (1, "Page not mapped", "Entry access"),
            (2, "Read/write access error", "Entry access"),
            (3, "System page access", "Entry access"),
            (5, "Page not mapped", "Directory access"),
            (6, "Read/write access error", "Directory access"),
            (7, "System page access", "Directory access"),
        ):
            with self.subTest(raw=raw):
                fault = _fault_source("core1", (raw << 16) | 2, 0x70001011)
                self.assertEqual(fault["fault_type"], raw & 3)
                self.assertEqual(fault["fault_type_raw"], raw)
                self.assertEqual(fault["fault_conditions"], [condition])
                self.assertEqual(fault["page_access_type"], access)
                self.assertTrue(fault["status_latched"])

    def test_directory_access_selector_and_requestor_alone_are_not_fault_latches(self):
        fault = _fault_source("core1", 0x40002, 0x70001011)
        self.assertFalse(fault["status_latched"])
        self.assertEqual(fault["fault_conditions"], [])

    def test_requestor_subunits_follow_firmware_tables_and_scope(self):
        for scope, bit, subunit, name in (
            ("core1", 1, 1, "Dependent Texture"),
            ("core1", 1, 2, "Non-Dependent Texture"),
            ("core1", 1, 11, "Non-Dependent Texture (Palettized)"),
            ("core1", 3, 1, "Index List"),
            ("master", 3, 1, "(Internal)"),
            ("master", 8, 5, "Core 1 Control stream"),
            ("core1", 8, 2, "Primitive blocks"),
            ("core1", 10, 6, "PDS (Vertex Data, Uniform Fetch)"),
            ("core1", 10, 12, "USSE Data Fence"),
            ("core1", 13, 1, "Code Fetch"),
            ("core1", 1, 31, None),
            ("core1", 5, 1, None),
        ):
            with self.subTest(scope=scope, bit=bit, subunit=subunit):
                fault = _fault_source(scope, 0x10000 | (1 << bit), 0x70001000 | (subunit << 4) | 1)
                self.assertEqual(fault["requestor_details"][0]["subunit_id"], subunit)
                self.assertEqual(fault["requestor_details"][0]["subunit_name"], name)

    def test_parses_v5_tables_and_trailer(self):
        info = GpuInfo.parse(fixture())
        self.assertEqual(info.core_count, 4)
        self.assertEqual(len(info.master_registers), 2)
        self.assertEqual(len(info.per_core_registers), 2)
        self.assertEqual(info.indexed_state[0].values, (1, 2, 3, 4))
        self.assertEqual(info.trailer, struct.pack("<I", 0xDEADBEEF))
        self.assertEqual(info.bif_diagnosis()["classification"], "no-bif-memory-fault")

    def test_bif_fault_is_classified(self):
        info = GpuInfo.parse(fixture(core_fault=(0, 0x12345001, 0, 0)))
        diagnosis = info.bif_diagnosis()
        self.assertEqual(diagnosis["classification"], "bif-memory-fault")
        self.assertEqual(diagnosis["fault_sources"][0]["scope"], "core1")
        self.assertEqual(diagnosis["fault_sources"][0]["address"], 0x12345000)

    def test_indexed_rows_expose_only_firmware_consumed_semantics(self):
        database = RegisterDatabase()
        known = GpuInfo.parse(fixture(indexed_selector=0x26)).summary(database)
        row = known["indexed_state"][0]
        self.assertTrue(row["consumer_interpreted"])
        self.assertEqual(row["name"], "MTE draw number")
        self.assertEqual(
            known["indexed_state_interpretation"]["consumer_interpreted_rows"], 1
        )

        unknown = GpuInfo.parse(fixture(indexed_selector=7)).summary(database)
        self.assertFalse(unknown["indexed_state"][0]["consumer_interpreted"])
        self.assertIsNone(unknown["indexed_state"][0]["name"])

    def test_decodes_capture_bitmap(self):
        info = GpuInfo.parse(fixture())
        self.assertTrue(info.snapshot_status()["complete"])
        self.assertTrue(info.capture_sections()["context_owners"])

    def test_decodes_capture_timestamp_and_hardware_clock_words(self):
        environment = (
            (8 << 16) | 2026,
            (23 << 16) | 25,
            (43 << 16) | 4,
            291204,
            0x42,
            0x29,
            0x29,
            0x6F,
        )
        info = GpuInfo.parse(fixture(environment_words=environment))
        timestamp = info.capture_timestamp()
        self.assertTrue(timestamp["valid"])
        self.assertEqual(timestamp["iso_local"], "2026-08-25T23:04:43.291204")
        self.assertFalse(timestamp["timezone_known"])
        self.assertEqual(info.hardware_clock_state()["soc_revision"], 0x42)
        self.assertEqual(info.hardware_clock_state()["crossbar_clock_word"], 0x6F)

    def test_decodes_context_owner_trailer(self):
        data = bytearray(0xF8)
        struct.pack_into("<I", data, 0, 0x10005)
        data[4:20] = b"SceKernelProcess"
        data[0xC0:0xC8] = bytes((0, 1, 0xFF, 3, 4, 5, 6, 7))
        info = GpuInfo.parse(fixture()[:-4] + data)
        trailer = info.decoded_trailer()
        self.assertEqual(trailer.context_owners[0].process_id, 0x10005)
        self.assertEqual(trailer.context_owners[0].process_name, "SceKernelProcess")
        self.assertEqual(trailer.context_slot_map[0][2], 0xFF)

    def test_classifies_export_tail_as_producer_padding(self):
        data = bytearray(0xF8 + 0x2D0)
        struct.pack_into("<4I", data, 0xE8, 1, 2, 3, 4)
        data[0xF8 + 8] = 0xA5
        summary = GpuTrailer.parse(data).summary()
        self.assertEqual(summary["producer_initialized_size"], 0x3130)
        self.assertEqual(summary["producer_export_size"], 0x3400)
        self.assertEqual(summary["producer_padding_size"], 0x2D0)
        self.assertEqual(summary["producer_padding_nonzero_bytes"], 1)
        self.assertTrue(summary["producer_padding_size_matches"])
        self.assertEqual(summary["driver_state"][0]["source_object_offset"], 0x108)

    def test_compare_reports_changed_core_register(self):
        left = GpuInfo.parse(fixture())
        right = GpuInfo.parse(fixture(core_fault=(0, 1, 0, 0)))
        diff = compare_gpu_info(left, right, RegisterDatabase())
        self.assertEqual(len(diff["per_core_changes"]), 1)
        self.assertEqual(diff["per_core_changes"][0]["offset"], 0x0C08)
        self.assertIn("context_owner_changes", diff)

    def test_compare_reports_driver_state_missing_from_one_side(self):
        trailer = bytearray(0xF8)
        struct.pack_into("<4I", trailer, 0xE8, 1, 2, 3, 4)
        left = GpuInfo.parse(fixture())
        right = GpuInfo.parse(fixture()[:-4] + trailer)
        changes = compare_gpu_info(left, right, RegisterDatabase())[
            "driver_state_changes"
        ]
        self.assertEqual(len(changes), 4)
        self.assertIsNone(changes[0]["left"])
        self.assertEqual(changes[0]["right"], 1)


if __name__ == "__main__":
    unittest.main()
