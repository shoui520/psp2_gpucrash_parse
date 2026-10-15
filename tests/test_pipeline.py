import unittest
from dataclasses import replace

from psp2_gpucrash_parse.gpu import GpuInfo, MasterRegister, PerCoreRegister
from psp2_gpucrash_parse.pipeline import firmware_pipeline_diagnosis


def clipper_fixture():
    header = (5, 0x10003, 0x3F, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0)
    return GpuInfo(5, header, 4, [MasterRegister(0x4C04, 0x80000)], [
        PerCoreRegister(0xC04, (0x80000,) * 4),
        PerCoreRegister(0xC74, (0, 0, 0xC0, 0)),
        PerCoreRegister(0xC78, (0x20,) * 4),
        PerCoreRegister(0xC7C, (0x10,) * 4),
    ], [PerCoreRegister(0x25, (0, 0, 0x10, 0)),
        PerCoreRegister(0x26, (0, 0, 101, 0))], b"")


class FirmwarePipelineTests(unittest.TestCase):
    def test_progress_applicability_matches_firmware_phase_pipe_and_unit_conditions(self):
        unit = 0xFFFFFFFF & ~(0x01000000 | 0x40000000 | 0x00100000 | 0x00800000)
        for phase in (0, 1, 2, 3):
            with self.subTest(phase=phase):
                info = GpuInfo(5, (5,) * 15, 1, [], [
                    PerCoreRegister(0xAA4, (0xAAAAAAE4,)),
                    PerCoreRegister(0xAA8, (0xAAAAAAAA,)),
                ], [PerCoreRegister(0, (phase,)), PerCoreRegister(2, (unit,)),
                    *[PerCoreRegister(selector, (1,)) for selector in
                      (0x26, 0x12, 0x13, 0x2E, 0x31, 0x51, 0x52, 0x11, 0x2F, 0x33)]], b"")
                core = firmware_pipeline_diagnosis(info)["cores"][0]
                counters = {item["code"]: item["applicability"] for item in core["draw_counters"]}
                tiles = {item["code"]: item["applicability"] for item in core["tile_progress"]}
                self.assertEqual(counters["mte"], "active" if phase & 1 else "inactive")
                self.assertEqual(counters["vdm-pipe-0"], "active")
                self.assertEqual(counters["vdm-pipe-1"], "inactive")
                self.assertEqual(counters["isp-output"], "active" if phase & 2 else "inactive")
                self.assertEqual(tiles["pds"], "active" if phase & 2 else "inactive")
                for code in ("usse-pipe-0", "isp-input", "isp-output", "tsp-parameter-fetch", "color-output"):
                    self.assertEqual(tiles[code], "active")
                self.assertEqual(tiles["usse-pipe-1"], "inactive")

    def test_clipper_first_match_bank_zero_and_missing_detail_remain_explicit(self):
        info = clipper_fixture()
        info = replace(info, indexed_state=[PerCoreRegister(0x25, (0x10, 0, 0x10, 0))])
        check = firmware_pipeline_diagnosis(info)["clipper_halt_check"]
        self.assertEqual([match["scope"] for match in check["matches"]], ["core0"])
        match = check["matches"][0]
        self.assertEqual((match["bank_offset"], match["context_id"]), (0xC78, 2))
        self.assertIsNone(match["draw_counter"])
        self.assertIsNone(match["draw_number"])
        info = replace(info, per_core_registers=[info.per_core_registers[0]])
        match = firmware_pipeline_diagnosis(info)["clipper_halt_check"]["matches"][0]
        self.assertIsNone(match["context_id"])
        self.assertIsNone(match["bank_offset"])

    def test_clipper_signature_uses_native_raw_bif_guard_and_first_matching_core(self):
        info = clipper_fixture()
        check = firmware_pipeline_diagnosis(info)["clipper_halt_check"]
        self.assertEqual(check["status"], "matched")
        self.assertEqual(len(check["matches"]), 1)
        match = check["matches"][0]
        self.assertEqual((match["scope"], match["selector"], match["mask"]), ("core2", 0x25, 0x10))
        self.assertEqual((match["draw_counter"], match["draw_number"]), (101, 100))
        self.assertEqual((match["bank_offset"], match["context_id"]), (0xC7C, 1))
        for status in (0x10000, 0x40000, 0x70000):
            with self.subTest(status=status):
                faulty = replace(info, master_registers=[MasterRegister(0x4C04, status)])
                self.assertEqual(firmware_pipeline_diagnosis(faulty)["clipper_halt_check"]["status"], "bif-fault-present")
                faulty = replace(info, per_core_registers=[PerCoreRegister(0xC04, (0, status, 0, 0))])
                self.assertEqual(firmware_pipeline_diagnosis(faulty)["clipper_halt_check"]["matches"], [])

    def test_clipper_signature_missing_capture_or_selector_does_not_match(self):
        info = clipper_fixture()
        for flags in (0, 3, 5, 6):
            header = list(info.header_words)
            header[2] = flags
            check = firmware_pipeline_diagnosis(replace(info, header_words=tuple(header)))["clipper_halt_check"]
            self.assertEqual(check, {"status": "unavailable", "matches": []})
        for incomplete in (replace(info, master_registers=[]), replace(info, per_core_registers=[]),
                           replace(info, indexed_state=[])):
            self.assertEqual(firmware_pipeline_diagnosis(incomplete)["clipper_halt_check"]["status"], "unavailable")
        clear = replace(info, indexed_state=[PerCoreRegister(0x25, (0,) * 4)])
        self.assertEqual(firmware_pipeline_diagnosis(clear)["clipper_halt_check"]["status"], "not-matched")

    def test_depth_extent_is_samples_and_four_x_msaa_halves_pixel_width(self):
        for mode, pixels, samples in ((0, 640, 1), (1, 640, 2), (2, 320, 4), (3, None, None), (None, None, None)):
            with self.subTest(mode=mode):
                registers = [PerCoreRegister(0x480, (0x100000 | (19 << 3),))]
                if mode is not None:
                    registers.append(PerCoreRegister(0x42C, (mode,)))
                info = GpuInfo(5, (5,) * 15, 1, [], registers, [], b"")
                zls = firmware_pipeline_diagnosis(info)["cores"][0]["zls"]
                self.assertEqual(zls["width_samples"], 640)
                self.assertEqual(zls["width_pixels"], pixels)
                self.assertEqual(zls["msaa"]["samples"], samples)
        info.per_core_registers[0] = PerCoreRegister(0x480, (19 << 3,))
        zls = firmware_pipeline_diagnosis(info)["cores"][0]["zls"]
        self.assertIsNone(zls["width_samples"])
        self.assertIsNone(zls["width_pixels"])

    def test_inactive_phase_and_pipe_counters_do_not_report_divergence(self):
        info = GpuInfo(5, (5,) * 15, 1, [], [
            PerCoreRegister(0xAA4, (0x55555555,)),
            PerCoreRegister(0xAA8, (0xAAAAAAAA,)),
        ], [PerCoreRegister(0, (2,)), PerCoreRegister(2, (0xFFFFFFFF,)),
            PerCoreRegister(0x26, (99,)), PerCoreRegister(0x12, (3 << 16,)),
            PerCoreRegister(0x13, (4 << 16,)),
            PerCoreRegister(0x2E, (7 << 16,)), PerCoreRegister(0x31, (7 << 16,)),
            PerCoreRegister(0x51, (0x0203,)), PerCoreRegister(0x52, (0x090A,))], b"")
        core = firmware_pipeline_diagnosis(info)["cores"][0]
        ta, fragment = core["draw_counter_checks"]
        self.assertEqual(ta["status"], "not-comparable")
        self.assertIsNone(ta["aligned"])
        self.assertEqual(ta["values"], {})
        self.assertEqual(ta["retained_values"], {"mte": 99, "vdm-pipe-0": 3, "vdm-pipe-1": 4})
        self.assertEqual(fragment["status"], "aligned")
        self.assertEqual(core["tile_checks"][0]["status"], "not-comparable")
        self.assertEqual(core["tile_checks"][0]["members"], ["usse-pipe-0"])
        self.assertEqual(len(core["tile_progress"]), 4)

    def test_pipe_tiles_compare_only_fragment_pipes_and_keep_inactive_values(self):
        info = GpuInfo(5, (5,) * 15, 1, [], [
            PerCoreRegister(0xAA4, (0x55555555,)), PerCoreRegister(0xAA8, (0x55555555,)),
            PerCoreRegister(0xB80, (0xAAAAAAAA,)),
        ], [PerCoreRegister(0x51, (0x0203,)), PerCoreRegister(0x52, (0x0203,)),
            PerCoreRegister(0x71, (0x0708,))], b"")
        check = firmware_pipeline_diagnosis(info)["cores"][0]["tile_checks"][0]
        self.assertEqual(check["status"], "aligned")
        self.assertEqual(check["members"], ["usse-pipe-0", "usse-pipe-1"])
        self.assertEqual(check["retained_values"]["usse-pipe-2"], [8, 7])

    def test_missing_activity_evidence_keeps_progress_unqualified(self):
        info = GpuInfo(5, (5,) * 15, 1, [], [], [
            PerCoreRegister(0x26, (10,)), PerCoreRegister(0x12, (1 << 16,)),
            PerCoreRegister(0x51, (0x0203,)), PerCoreRegister(0x52, (0x0708,)),
        ], b"")
        core = firmware_pipeline_diagnosis(info)["cores"][0]
        for field in core["draw_counters"] + core["tile_progress"]:
            self.assertEqual(field["applicability"], "unknown")
        for check in core["draw_counter_checks"] + core["tile_checks"]:
            self.assertEqual(check["status"], "not-comparable")

    def test_usse_data_master_slots_are_lsb_first_and_not_instruction_state(self):
        info = GpuInfo(5, (5,) * 15, 1, [], [
            PerCoreRegister(0xAA4, (0xAAAAAAE4,)),
            PerCoreRegister(0xAA8, (0x55555555,)),
            PerCoreRegister(0xB80, (0xAAAAAAAA,)),
            PerCoreRegister(0xB84, (0xFFFFFFFF,)),
        ], [], b"")
        pipes = firmware_pipeline_diagnosis(info)["cores"][0]["usse_data_master_usage"]
        self.assertEqual(pipes[0]["usage"], "VP.S............")
        self.assertEqual(pipes[0]["counts"], {"vertex": 1, "fragment": 1, "idle": 13, "service": 1})
        self.assertEqual(pipes[1]["usage"], "P" * 16)
        self.assertEqual(pipes[2]["usage"], "." * 16)
        self.assertEqual(pipes[3]["usage"], "S" * 16)

    def test_draw_number_zero_and_one_both_display_zero_only_for_ta(self):
        info = GpuInfo(5, (5,) * 15, 1, [], [PerCoreRegister(0xAA4, (0,))], [
            PerCoreRegister(0, (1,)),
            PerCoreRegister(0x26, (0,)), PerCoreRegister(0x12, (0x10000,)),
            PerCoreRegister(0x2E, (0x10000,)),
        ], b"")
        counters = firmware_pipeline_diagnosis(info)["cores"][0]["draw_counters"]
        self.assertEqual([item["draw_number"] for item in counters], [0, 0, 1])
        self.assertEqual([item["counter_value"] for item in counters], [0, 1, 1])
        checks = firmware_pipeline_diagnosis(info)["cores"][0]["draw_counter_checks"]
        self.assertFalse(checks[0]["aligned"])

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
            PerCoreRegister(0x042C, (0,)),
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
        self.assertEqual(counters["mte"], 0xABB)
        self.assertEqual(counters["vdm-pipe-0"], 0x1233)
        self.assertEqual(counters["isp-output"], 0x45)
        tiles = {item["code"]: item for item in core["tile_progress"]}
        self.assertEqual((tiles["pds"]["tile_x"], tiles["pds"]["tile_y"]), (20, 10))
        self.assertEqual((tiles["isp-output"]["tile_x"], tiles["isp-output"]["tile_y"]), (2, 6))
        self.assertEqual(core["render_bounds"]["width_pixels"], 640)
        self.assertEqual(core["render_bounds"]["height_pixels"], 224)
        self.assertEqual(core["region_header_base"], 0x50042000)
        self.assertEqual(core["zls"]["width_pixels"], 640)
        self.assertEqual(core["zls"]["surfaces"]["depth_load"]["address"], 0x62000040)
        self.assertTrue(result["master"]["transfer_active"])
        self.assertTrue(result["master"]["partial_render_deadlock"])


if __name__ == "__main__":
    unittest.main()
