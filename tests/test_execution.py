import struct
import unittest

from psp2_gpucrash_parse.execution import (
    ModuleInfo,
    StackInfo,
    ThreadInfo,
    ThreadRegisterInfo,
)


class ExecutionInfoTests(unittest.TestCase):
    def test_decodes_module_segments_and_locations(self):
        header = bytearray(0x50)
        struct.pack_into("<II", header, 0, 0x50, 0x1234)
        header[0x24:0x2E] = b"TestModule"
        struct.pack_into("<I", header, 0x4C, 2)
        text = struct.pack("<5I", 0, 5, 0x81000000, 0x1000, 0x1000)
        data = struct.pack("<5I", 0, 6, 0x81001000, 0x800, 0x1000)
        payload = struct.pack("<2I", 1, 1) + header + text + data + bytes(0x10)

        modules = ModuleInfo.parse(payload)
        self.assertEqual(modules.modules[0].segments[0].permissions, "r-x")
        self.assertEqual(modules.modules[0].segments[1].permissions, "rw-")
        location = modules.locate(0x81000125)
        self.assertEqual(location.notation, "TestModule@1+0x124")
        self.assertIsNone(modules.locate(0x81001100, executable_only=True))

    def test_decodes_thread_register_and_stack_layouts(self):
        thread = bytearray(0xA0)
        struct.pack_into("<II", thread, 0, len(thread), 0x40010001)
        thread[8:18] = b"RenderTask"
        struct.pack_into("<I", thread, 0x28, 0x10000000)
        struct.pack_into("<I", thread, 0x30, 0x00000008)
        struct.pack_into("<III", thread, 0x34, 0x81000101, 0x83000000, 0x4000)
        struct.pack_into("<iii", thread, 0x4C, 96, 100, 2)
        struct.pack_into("<II", thread, 0x54, 0xF, 0x4)
        struct.pack_into("<i", thread, 0x5C, 2)
        struct.pack_into("<II", thread, 0x60, 4, 0x20010002)
        struct.pack_into("<II", thread, 0x68, 0x55667788, 0x11223344)
        struct.pack_into("<I", thread, 0x9C, 0x81000120)
        threads = ThreadInfo.parse(struct.pack("<2I", 0x12, 1) + thread)
        item = threads.threads[0]
        self.assertEqual(item.status_name, "waiting")
        self.assertEqual(item.current_priority, 100)
        self.assertEqual(item.current_cpu_id, 2)
        self.assertEqual(item.wait_id, 0x20010002)
        self.assertEqual(item.run_clocks, 0x1122334455667788)

        gpr = list(range(16))
        gpr[13:16] = [0x83001000, 0x81000201, 0x81000301]
        register = struct.pack("<2I16II", 0x4C, item.uid, *gpr, 0x600001D3)
        registers = ThreadRegisterInfo.parse(struct.pack("<2I", 0x11, 1) + register)
        self.assertEqual(registers.records[0].sp, 0x83001000)
        self.assertEqual(registers.records[0].lr, 0x81000201)
        self.assertEqual(registers.records[0].pc, 0x81000301)

        stack = StackInfo.parse(
            struct.pack("<2I4I", 4, 1, 1, item.uid, 0x3000, 0x1000)
        )
        self.assertEqual(stack.records[0].peak_bytes, 0x3000)
        self.assertEqual(stack.records[0].current_bytes, 0x1000)


if __name__ == "__main__":
    unittest.main()
