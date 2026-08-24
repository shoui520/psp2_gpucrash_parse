import struct
import unittest

from psp2_gpucrash_parse.activity import (
    GPU_ACTIVITY_BUFFER_SIZE,
    GpuActivityInfo,
)
from psp2_gpucrash_parse.core import ParseError


class GpuActivityTests(unittest.TestCase):
    def test_parses_empty_version_8_header(self):
        info = GpuActivityInfo.parse(struct.pack("<3I", 8, 17, 0))
        self.assertEqual(info.sequence, 17)
        self.assertIsNone(info.activity_buffer)

    def test_parses_firmware_ring_shape(self):
        payload = bytearray(GPU_ACTIVITY_BUFFER_SIZE)
        struct.pack_into("<4I", payload, 0, 0x65, 3, GPU_ACTIVITY_BUFFER_SIZE, 0)
        struct.pack_into("<2I", payload, 0x10, 2, 0x1234)
        payload[0x18:0x1E] = b"render"
        record = struct.pack("<3I", 8, 9, len(payload)) + payload
        info = GpuActivityInfo.parse(record)
        self.assertEqual(info.activity_buffer.event_count, 3)
        self.assertEqual(info.activity_buffer.slots[0].name, "render")
        self.assertTrue(info.activity_buffer.slots[0].populated)

    def test_rejects_truncated_payload(self):
        with self.assertRaises(ParseError):
            GpuActivityInfo.parse(struct.pack("<3I", 8, 0, 0x810))


if __name__ == "__main__":
    unittest.main()
