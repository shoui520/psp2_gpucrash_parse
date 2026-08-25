import struct
import unittest

from psp2_gpucrash_parse.core import CoreDump, LoadSegment
from psp2_gpucrash_parse.unwind import execute_compact_bytecode
from tests.test_debugger import debug_core


class UnwindTests(unittest.TestCase):
    def test_compact_ehabi_pop_produces_a_verified_caller_state(self):
        core: CoreDump = debug_core()
        stack_base = 0x83000000
        stack = struct.pack("<2I", 0x44444444, 0x81000081)
        core.loads.append(
            LoadSegment(
                9,
                0,
                stack_base,
                0,
                len(stack),
                len(stack),
                6,
                4,
                stack,
            )
        )
        registers = [0] * 16
        registers[13] = stack_base
        registers[14] = 0xDEADBEEF
        registers[15] = 0x81000041
        # 0xa8 pops r4 and lr; finish copies the restored lr into pc.
        result = execute_compact_bytecode(
            core, registers, stack_base, stack_base + len(stack), [0xA8, 0xB0]
        )
        self.assertTrue(result["finished"])
        self.assertEqual(result["registers"][4], 0x44444444)
        self.assertEqual(result["registers"][13], stack_base + 8)
        self.assertEqual(result["registers"][15], 0x81000081)
        self.assertEqual(len(result["stack_reads"]), 2)


if __name__ == "__main__":
    unittest.main()
