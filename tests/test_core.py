import gzip
import struct
import tempfile
import unittest
from pathlib import Path

from psp2_gpucrash_parse.core import CoreDump


def note(name, description, note_type=0):
    encoded = name.encode() + b"\0"
    result = struct.pack("<III", len(encoded), len(description), note_type)
    result += encoded + b"\0" * ((-len(encoded)) & 3)
    result += description + b"\0" * ((-len(description)) & 3)
    return result


def elf_fixture():
    note_data = note("GPU_ACT_INFO", struct.pack("<III", 8, 0, 0))
    load_data = bytes(range(32))
    phoff = 52
    phnum = 2
    note_offset = phoff + phnum * 32
    load_offset = note_offset + len(note_data)
    header = bytearray(52)
    header[:16] = b"\x7fELF\x01\x01\x01" + b"\0" * 9
    struct.pack_into("<HHIIIIIHHHHHH", header, 16, 4, 40, 1, 0, phoff, 0, 0, 52, 32, phnum, 0, 0, 0)
    programs = struct.pack("<IIIIIIII", 4, note_offset, 0, 0, len(note_data), len(note_data), 0, 4)
    programs += struct.pack("<IIIIIIII", 1, load_offset, 0x81000000, 0, len(load_data), len(load_data), 6, 0x1000)
    return bytes(header) + programs + note_data + load_data


class CoreTests(unittest.TestCase):
    def test_reads_gzip_elf_notes_and_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "core.psp2dmp"
            path.write_bytes(gzip.compress(elf_fixture()))
            core = CoreDump.read(path)
            self.assertTrue(core.compressed)
            self.assertEqual(core.note("GPU_ACT_INFO").description[:4], struct.pack("<I", 8))
            self.assertEqual(core.read_memory(0x81000004, 4), bytes(range(4, 8)))
            self.assertEqual(core.captured_bytes(0x81000008, 0x81000018), 16)
            self.assertEqual(core.captured_ranges(0x81000008, 0x81000018), [(0x81000008, 0x81000018)])


if __name__ == "__main__":
    unittest.main()
