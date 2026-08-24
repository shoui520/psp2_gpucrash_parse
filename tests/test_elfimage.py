import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from psp2_gpucrash_parse.elfimage import ElfImage, enrich_diagnosis_with_elfs


def _vita_elf(module_name: str = "TestModule") -> bytes:
    data = bytearray(0x400)
    data[:16] = b"\x7fELF\x01\x01\x01" + bytes(9)
    struct.pack_into(
        "<HHIIIIIHHHHHH",
        data,
        16,
        0xFE04,
        40,
        1,
        0x40,
        52,
        0x300,
        0x05000000,
        52,
        32,
        2,
        40,
        4,
        0,
    )
    struct.pack_into(
        "<8I", data, 52, 1, 0x100, 0x81000000, 0, 0x100, 0x100, 5, 0x10
    )
    struct.pack_into(
        "<8I", data, 84, 1, 0x200, 0x81001000, 0, 0, 0x80, 6, 0x10
    )
    struct.pack_into("<HBB", data, 0x140, 0, 1, 2)
    data[0x144 : 0x144 + len(module_name)] = module_name.encode("ascii")
    struct.pack_into("<II", data, 0x16C, 0x80, 0xB4)
    struct.pack_into("<HHHHHH", data, 0x180, 0x34, 1, 0, 1, 0, 0)
    struct.pack_into("<I", data, 0x190, 0x12345678)
    struct.pack_into("<I", data, 0x194, 0x810000C0)
    struct.pack_into("<II", data, 0x19C, 0x810000D0, 0x810000D4)
    data[0x1C0:0x1C8] = b"SceTest\0"
    struct.pack_into("<II", data, 0x1D0, 0xDEADBEEF, 0x81000030)

    strings = b"\0render_main\0"
    data[0x280 : 0x280 + len(strings)] = strings
    struct.pack_into("<IIIBBH", data, 0x2B0, 1, 0x81000011, 0x20, 0x12, 0, 1)
    # section 1: executable image; section 2: strings; section 3: symbols
    struct.pack_into("<10I", data, 0x328, 0, 1, 6, 0x81000000, 0x100, 0x100, 0, 0, 4, 0)
    struct.pack_into("<10I", data, 0x350, 0, 3, 0, 0, 0x280, len(strings), 0, 0, 1, 0)
    struct.pack_into("<10I", data, 0x378, 0, 2, 0, 0, 0x2A0, 0x20, 2, 1, 4, 0x10)
    return bytes(data)


def _diagnosis() -> dict:
    pc = 0x82000018
    return {
        "execution_context": {
            "modules": {
                "modules": [
                    {
                        "index": 3,
                        "uid": 0x40010003,
                        "name": "TestModule",
                        "segments": [
                            {
                                "number": 1,
                                "permissions": "r-x",
                                "start": 0x82000000,
                                "size": 0x100,
                            },
                            {
                                "number": 2,
                                "permissions": "rw-",
                                "start": 0x83000000,
                                "size": 0x80,
                            },
                        ],
                    }
                ]
            },
            "threads": [
                {
                    "index": 7,
                    "uid": 0x40010007,
                    "name": "RenderThread",
                    "registers": {"pc": pc, "lr": 0, "cpsr": 0x20},
                    "pc_location": {
                        "module_index": 3,
                        "segment_number": 1,
                        "segment_start": 0x82000000,
                    },
                    "lr_location": None,
                    "stack_return_candidates": [],
                }
            ],
        }
    }


class ElfImageTests(unittest.TestCase):
    def test_parses_vita_module_identity_loads_and_symbols(self):
        image = ElfImage.parse(Path("test.elf"), _vita_elf())
        self.assertEqual(image.module_name, "TestModule")
        self.assertEqual(image.module_version, (1, 2))
        self.assertEqual([item.permissions for item in image.load_segments], ["r-x", "rw-"])
        self.assertEqual(image.imports[0].library_name, "SceTest")
        self.assertEqual(image.imports[0].functions[0].nid, 0xDEADBEEF)
        symbol = image.symbol_for(0x81000018)
        self.assertIsNotNone(symbol)
        self.assertEqual(symbol.name, "render_main")
        self.assertEqual(symbol.summary(0x81000018)["offset"], 8)

    def test_matches_layout_and_enriches_runtime_pc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.elf"
            path.write_bytes(_vita_elf())
            database = Path(directory) / "share" / "vita-headers" / "db" / "360"
            database.mkdir(parents=True)
            (database / "SceTest.yml").write_text(
                "modules:\n"
                "  SceTest:\n"
                "    libraries:\n"
                "      SceTest:\n"
                "        nid: 0x12345678\n"
                "        functions:\n"
                "          sceTestCall: 0xDEADBEEF\n",
                encoding="utf-8",
            )
            diagnosis = _diagnosis()
            with patch("psp2_gpucrash_parse.elfimage.shutil.which", return_value=None):
                result = enrich_diagnosis_with_elfs(
                    diagnosis, [path], Path(directory)
                )
        self.assertEqual(result["matched_count"], 1)
        association = result["images"][0]["association"]
        self.assertEqual(association["match_quality"], "exact")
        location = diagnosis["execution_context"]["threads"][0]["pc_image"]
        self.assertEqual(location["elf_virtual_address"], 0x81000018)
        self.assertEqual(location["file_offset"], 0x118)
        self.assertEqual(location["symbol"]["name"], "render_main")
        self.assertEqual(result["images"][0]["resolved_import_function_count"], 1)
        self.assertEqual(
            result["images"][0]["imports"][0]["functions"][0]["name"],
            "sceTestCall",
        )

    def test_rejects_embedded_module_name_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.elf"
            path.write_bytes(_vita_elf("OtherModule"))
            with patch("psp2_gpucrash_parse.elfimage.shutil.which", return_value=None):
                result = enrich_diagnosis_with_elfs(_diagnosis(), [path])
        self.assertEqual(result["matched_count"], 0)
        self.assertEqual(result["rejected_count"], 1)
        self.assertIn("not present", result["images"][0]["association"]["reasons"][0])


if __name__ == "__main__":
    unittest.main()
