import struct
import unittest

from psp2_gpucrash_parse.core import ParseError
from psp2_gpucrash_parse.tty import TtyInfo


class TtyInfoTests(unittest.TestCase):
    def test_extracts_fixed_payload_and_strips_console_ansi(self):
        payload = b"partial line\x1b[0m\nnext\0\0"
        record = struct.pack("<3I", 19, 0, len(payload)) + payload
        info = TtyInfo.parse(record)
        self.assertEqual(info.header_words, (19, 0, len(payload)))
        self.assertEqual(info.payload, payload)
        self.assertEqual(info.text(), "partial line\nnext")
        self.assertIn("\x1b[0m", info.text(preserve_ansi=True))

    def test_rejects_truncated_declared_payload(self):
        with self.assertRaises(ParseError):
            TtyInfo.parse(struct.pack("<3I", 19, 0, 4096) + b"short")

    def test_extracts_generic_timestamp_chronology(self):
        payload = (
            b"partial ring-buffer line\n"
            b"[   10.2500] submitted work\n"
            b"[   10.7500] completed work\n"
            b"[   12.0000] next batch\n"
        )
        info = TtyInfo.parse(struct.pack("<3I", 1, 0, len(payload)) + payload)
        timeline = info.timeline()
        self.assertEqual(timeline["line_count"], 4)
        self.assertEqual(timeline["timestamped_line_count"], 3)
        self.assertTrue(timeline["first_line_untimestamped"])
        self.assertEqual(timeline["first_timestamp"], 10.25)
        self.assertEqual(timeline["last_timestamp"], 12.0)
        self.assertEqual(timeline["captured_span"], 1.75)
        self.assertEqual(timeline["largest_nonnegative_gap"]["delta"], 1.25)
        self.assertEqual(timeline["timestamp_regressions"], [])


if __name__ == "__main__":
    unittest.main()
