import unittest

from psp2_gpucrash_parse.gpu import GpuInfo, MasterRegister, PerCoreRegister
from psp2_gpucrash_parse.pipeline import firmware_pipeline_diagnosis


class FirmwarePipelineTests(unittest.TestCase):
    def test_decodes_firmware_phase_units_counters_tiles_and_surfaces(self):
        inactive = 0xFFFFFFFF
        unit_value = inactive & ~(0x01000000 | 0x04000000 | 0x10000000)
        indexed = [
            PerCoreRegister(0x00, (0x03,)),
            PerCoreRegister(0x02, (unit_value,)),
            PerCoreRegister(0x11, (0x00000A14,)),
            PerCoreRegister(0x12, (0x12340001,)),
            PerCoreRegister(0x26, (0x00002ABC,)),
            PerCoreRegister(0x2E, (0x00450206,)),
            PerCoreRegister(0x2F, (0x00000307,)),
            PerCoreRegister(0x31, (0x00670308,)),
            PerCoreRegister(0x33, (0x00000409,)),
            PerCoreRegister(0x51, (0x00000B15,)),
            PerCoreRegister(0x52, (0,)),
            PerCoreRegister(0x71, (0,)),
            PerCoreRegister(0x72, (0,)),
            PerCoreRegister(0x13, (0,)),
            PerCoreRegister(0x14, (0,)),
            PerCoreRegister(0x15, (0,)),
        ]
        zls = 0x00100000 | 0x00004000 | 0x00040000 | (19 << 3)
        registers = [
            PerCoreRegister(0x0248, (0x000DF27F,)),
            PerCoreRegister(0x0408, (0x50042003,)),
            PerCoreRegister(0x0480, (zls,)),
            PerCoreRegister(0x0484, (0x6200004F,)),
            PerCoreRegister(0x0488, (0x6208004F,)),
            PerCoreRegister(0x048C, (0,)),
            PerCoreRegister(0x0490, (0,)),
            PerCoreRegister(0x04E4, (0x00010A24,)),
        ]
        info = GpuInfo(
            5,
            (5, 0x10003, 0x3F, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
            1,
            [MasterRegister(0x4118, 0x1C), MasterRegister(0x4178, 1)],
            registers,
            indexed,
            b"",
        )

        result = firmware_pipeline_diagnosis(info)
        core = result["cores"][0]
        self.assertEqual(core["active_phases"], ["vertex", "fragment"])
        self.assertEqual(
            [item["code"] for item in core["active_units"]],
            ["pds", "mte", "isp"],
        )
        counters = {item["code"]: item["draw_number"] for item in core["draw_counters"]}
        self.assertEqual(counters["mte"], 0xABC)
        self.assertEqual(counters["vdm-pipe-0"], 0x1234)
        self.assertEqual(counters["isp-output"], 0x45)
        tiles = {item["code"]: item for item in core["tile_progress"]}
        self.assertEqual((tiles["pds"]["tile_x"], tiles["pds"]["tile_y"]), (10, 20))
        self.assertEqual((tiles["isp-output"]["tile_x"], tiles["isp-output"]["tile_y"]), (6, 2))
        self.assertEqual(core["render_bounds"]["width_pixels"], 640)
        self.assertEqual(core["render_bounds"]["height_pixels"], 224)
        self.assertEqual(core["region_header_base"], 0x50042000)
        self.assertEqual(core["zls"]["width_pixels"], 640)
        self.assertEqual(core["zls"]["surfaces"]["depth_load"]["address"], 0x62000040)
        self.assertTrue(result["master"]["transfer_active"])
        self.assertTrue(result["master"]["partial_render_deadlock"])


if __name__ == "__main__":
    unittest.main()
