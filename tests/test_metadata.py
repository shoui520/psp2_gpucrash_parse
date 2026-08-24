import struct
import unittest

from psp2_gpucrash_parse.core import Note
from psp2_gpucrash_parse.metadata import SummaryInfo


class SummaryInfoTests(unittest.TestCase):
    def test_decodes_note_completion_inventory(self):
        data = struct.pack(
            "<9I",
            0x11,
            0,
            2,
            0x1000,
            40,
            40,
            0x2000,
            100,
            64,
        )
        info = SummaryInfo.parse(data)
        self.assertEqual(info.entries[0].status, "complete")
        self.assertEqual(info.entries[1].status, "truncated")

    def test_summary_self_entry_is_not_incomplete(self):
        data = struct.pack("<6I", 0x11, 0, 1, 0x1035, 64, 0)
        info = SummaryInfo.parse(data)
        notes = [Note("SUMMARY_INFO", 0x1035, data, 0)]
        summary = info.summary(notes)
        self.assertEqual(summary["incomplete_count"], 0)
        self.assertEqual(summary["entries"][0]["status"], "self-reference")


if __name__ == "__main__":
    unittest.main()
