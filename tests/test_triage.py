import struct
import unittest
from pathlib import Path
from unittest.mock import patch

from psp2_gpucrash_parse.core import Note
from psp2_gpucrash_parse.registers import RegisterDatabase
from psp2_gpucrash_parse.triage import triage_dumps
from tests.test_diagnosis import core_fixture, gpu_fixture


class TriageTests(unittest.TestCase):
    def test_entry_and_directory_faults_remain_separate_after_condition_enum_correction(self):
        cores = {}
        for filename, status in (("entry.psp2dmp", 0x90002), ("directory.psp2dmp", 0xD0002)):
            core = core_fixture()
            core.path = Path(filename)
            data = bytearray(gpu_fixture(fault=True))
            # Fifteen-word header, three master pairs, core status row; core 2.
            struct.pack_into("<I", data, 15 * 4 + 3 * 8 + 4 + 2 * 4, status)
            core.notes.append(Note("GPU_INFO", 0x2000, bytes(data), 0))
            cores[core.path] = core
        with patch("psp2_gpucrash_parse.triage.CoreDump.read", side_effect=cores.__getitem__):
            report = triage_dumps(list(cores), RegisterDatabase.load())
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(len(report["groups"]["failure"]), 2)
        self.assertEqual(len(report["groups"]["failure_pattern"]), 2)
        self.assertEqual([entry["fault_conditions"] for entry in report["entries"]],
                         [["Page not mapped"], ["Page not mapped"]])
        self.assertEqual([entry["faults"][0]["page_access_type"] for entry in report["entries"]],
                         ["Entry access", "Directory access"])


if __name__ == "__main__":
    unittest.main()
