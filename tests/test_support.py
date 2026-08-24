import struct
import unittest
from pathlib import Path

from psp2_gpucrash_parse.core import CoreDump, Note
from psp2_gpucrash_parse.support import supporting_context


class SupportingContextTests(unittest.TestCase):
    def test_decodes_application_libraries_and_variable_event_records(self):
        app = bytearray(0x104)
        struct.pack_into("<I", app, 0, 6)
        app[8:0x12] = b"TEST00001\0"
        app[0x12:0x92] = b"Renderer Test\0".ljust(0x80, b"\0")
        app[0x92:0x9A] = b"01.02\0\0\0"

        libraries = (
            struct.pack("<2I", 6, 2)
            + b"binary\0SceGxm\0more-binary\0SceLibKernel\0"
        )

        event = bytearray(0x5C)
        struct.pack_into("<II", event, 0, len(event), 10)
        event[8:0x14] = b"TEST00001\0\0\0"
        struct.pack_into("<III", event, 0x14, 0x12711, 0x10005, 0x10213)
        struct.pack_into("<I", event, 0x3C, 0x1C)
        struct.pack_into("<II", event, 0x44, 0x1234, 0x1000000)
        event[0x50:] = b"RenderTask\0\0"
        event_log = struct.pack("<3I", 9, 3, 1) + event + bytes(0x20)

        core = CoreDump(
            path=Path("support.psp2dmp"),
            compressed=False,
            raw_file_sha256="",
            image_sha256="",
            image_size=0,
            elf_type=4,
            machine=40,
            flags=0,
            notes=[
                Note("APP_INFO", 0, bytes(app), 0),
                Note("LIBRARY_INFO", 0, libraries, 1),
                Note("EVENT_LOG_INFO", 0, event_log, 2),
            ],
            loads=[],
        )
        result = supporting_context(core)
        self.assertEqual(result["application"]["title_id"], "TEST00001")
        self.assertEqual(result["application"]["title_name"], "Renderer Test")
        self.assertEqual(result["libraries"]["names"], ["SceGxm", "SceLibKernel"])
        self.assertTrue(result["libraries"]["name_count_matches"])
        record = result["event_log"]["records"][0]
        self.assertEqual(record["event_code"], 0x12711)
        self.assertEqual(record["related_id"], 0x1234)
        self.assertEqual(record["related_name"], "RenderTask")
        self.assertEqual(result["event_log"]["trailing_nonzero_bytes"], 0)

    def test_decodes_process_sets_budgets_and_auxiliary_records(self):
        application_list = bytearray(0x48)
        struct.pack_into("<3I", application_list, 0, 6, 1, 0x55)
        struct.pack_into("<II", application_list, 12, 0x12345, 0x10005)
        application_list[20:30] = b"TEST00001\0"
        application_list[32:48] = b"ux0:app/test\0\0\0"

        external = bytearray(0x58)
        struct.pack_into("<2I", external, 0, 6, 1)
        struct.pack_into("<IIII", external, 8, 0, 0x12345, 0x10001, 0x20)
        external[24:35] = b"TestProcess"
        external[56:72] = b"ux0:app/test\0\0\0"

        budget_header = bytearray(0x2C)
        budget_header[8:15] = b"Game C\0"
        struct.pack_into("<I", budget_header, 0x28, 1)
        partition = bytearray(0x210)
        partition[8:17] = b"User Main"
        struct.pack_into("<II", partition, 0x34, 0x90000000, 0x1000)
        struct.pack_into("<II", partition, 0x3C, 0x400, 0x200)
        budget = struct.pack("<2I", 6, 1) + budget_header + partition

        system2 = bytearray(0x34)
        struct.pack_into("<III", system2, 0, 0x10, 0x41460000, 0x53434543)
        struct.pack_into("<I", system2, 0x24, 0x30)
        system2[0x30:] = b"data"
        system_device = (
            struct.pack("<I", 0x11)
            + struct.pack("<II", 0x10, 5)
            + b"Card\0\0\0\0"
        )
        file_record = bytearray(0x58)
        struct.pack_into("<I", file_record, 0x10, 0x12345)
        file_info = struct.pack("<2I", 5, 1) + file_record
        metadata = struct.pack("<4I", 2, 1, 0x40010001, 0) + b"payload\0"
        corefile = struct.pack("<10I", 1, 0, 19, 2, 0, 0, 0, 0, 0, 0)
        device = struct.pack("<6I", 6, 0, 0, 0x12345, 0, 0)

        notes = [
            Note("APP_LIST_INFO", 0, bytes(application_list), 0),
            Note("EXTNL_PROC_INFO", 0, bytes(external), 1),
            Note("BUDGET_INFO", 0, budget, 2),
            Note("SYSTEM_INFO2", 0, bytes(system2), 3),
            Note("SYS_DEVICE_INFO", 0, system_device, 4),
            Note("FILE_INFO", 0, file_info, 5),
            Note("META_DATA_INFO", 0, metadata, 6),
            Note("COREFILE_INFO", 0, corefile, 7),
            Note("DEVICE_INFO", 0, device, 8),
        ]
        core = CoreDump(
            path=Path("support-extra.psp2dmp"),
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
        result = supporting_context(core)
        self.assertEqual(result["application_list"]["records"][0]["title_id"], "TEST00001")
        self.assertEqual(result["external_processes"]["records"][0]["name"], "TestProcess")
        self.assertEqual(result["budgets"]["partition_count"], 1)
        self.assertAlmostEqual(
            result["budgets"]["highest_utilization_partition"]["utilization"], 0.75
        )
        self.assertEqual(result["system2"]["payload_sha256"],
                         "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7")
        self.assertEqual(result["system_devices"]["records"][0]["description"], "Card")
        self.assertEqual(result["files"]["record_count"], 1)
        self.assertEqual(result["files"]["records"][0]["process_name"], "TestProcess")
        self.assertEqual(result["metadata"]["entries"][0]["identifier"], 0x40010001)
        self.assertEqual(result["corefile"]["producer_kind"], 19)
        self.assertEqual(result["device"]["process_references"][0]["process_name"], "TestProcess")


if __name__ == "__main__":
    unittest.main()
