import struct
import unittest

from psp2_gpucrash_parse.gpu import GpuInfo, compare_gpu_info
from psp2_gpucrash_parse.registers import RegisterDatabase


def fixture(master_fault=0, core_fault=(0, 0, 0, 0)):
    header = [5, 0x10003, 0x3F, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 2, 1]
    words = header[:]
    words += [0x4C04, 0x80000, 0x4C08, master_fault]
    words += [0x0C04, 0x80000, 0x80000, 0x80000, 0x80000]
    words += [0x0C08, *core_fault]
    words += [7, 1, 2, 3, 4]
    words += [0xDEADBEEF]
    return struct.pack(f"<{len(words)}I", *words)


class GpuInfoTests(unittest.TestCase):
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

    def test_decodes_capture_bitmap(self):
        info = GpuInfo.parse(fixture())
        self.assertTrue(info.snapshot_status()["complete"])
        self.assertTrue(info.capture_sections()["context_owners"])

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
