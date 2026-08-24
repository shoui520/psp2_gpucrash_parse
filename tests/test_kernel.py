import struct
import unittest
from pathlib import Path

from psp2_gpucrash_parse.core import CoreDump, Note
from psp2_gpucrash_parse.kernel import KernelObjectRegistry


def core_with_notes(notes):
    return CoreDump(
        path=Path("kernel.psp2dmp"),
        compressed=False,
        raw_file_sha256="",
        image_sha256="",
        image_size=0,
        elf_type=4,
        machine=40,
        flags=0,
        notes=notes,
        loads=[],
    )


class KernelObjectTests(unittest.TestCase):
    def test_decodes_waiters_and_lock_ownership(self):
        semaphore = bytearray(0x44)
        struct.pack_into("<III", semaphore, 0, 0, 0x1001, 0x2001)
        semaphore[0x0C:0x18] = b"WorkReady".ljust(12, b"\0")
        struct.pack_into("<I", semaphore, 0x2C, 0x100)
        struct.pack_into("<IIII", semaphore, 0x30, 0, 1, 0, 1)
        struct.pack_into("<I", semaphore, 0x40, 1)
        semaphore_payload = (
            struct.pack("<2I", 1, 1)
            + semaphore
            + struct.pack("<2I", 0x2001, 0x4001)
        )

        mutex = bytearray(0x48)
        struct.pack_into("<III", mutex, 0, 0, 0x1002, 0x2001)
        mutex[0x0C:0x18] = b"QueueLock".ljust(12, b"\0")
        struct.pack_into("<I", mutex, 0x34, 1)
        struct.pack_into("<II", mutex, 0x38, 1, 0x4002)
        mutex_payload = struct.pack("<2I", 9, 1) + mutex

        core = core_with_notes(
            [
                Note("SEMAPHORE_INFO", 0, semaphore_payload, 0),
                Note("MUTEX_INFO", 0, mutex_payload, 1),
            ]
        )
        registry = KernelObjectRegistry.parse(core)
        sem = registry.resolve(0x1001)
        lock = registry.resolve(0x1002)
        self.assertEqual(sem.waiting_thread_count, 1)
        self.assertEqual(sem.waiters[0].thread_uid, 0x4001)
        self.assertEqual(sem.state["current_count"], 0)
        self.assertEqual(lock.owner_thread_uid, 0x4002)
        self.assertEqual(registry.summary()["active_object_count"], 2)

    def test_decodes_variable_message_pipe_record(self):
        record = bytearray(0x58)
        struct.pack_into("<IIII", record, 0, len(record), 0x3001, 0x2001, 0)
        record[0x0C:0x18] = b"CommandPipe".ljust(12, b"\0")
        struct.pack_into("<III", record, 0x34, 0x1000, 0x400, 1)
        struct.pack_into("<I", record, 0x40, 0)
        struct.pack_into("<2I", record, 0x50, 0x2001, 0x4003)
        payload = struct.pack("<2I", 0x11, 1) + record
        core = core_with_notes([Note("MESG_PIPE_INFO", 0, payload, 0)])

        pipe = KernelObjectRegistry.parse(core).resolve(0x3001)
        self.assertEqual(pipe.state["buffer_size"], 0x1000)
        self.assertEqual(pipe.state["free_size"], 0x400)
        self.assertEqual(pipe.waiters[0].thread_uid, 0x4003)

    def test_decodes_condition_variable_waiters(self):
        cond = bytearray(0x3C)
        struct.pack_into("<III", cond, 0, 0, 0x5001, 0x2001)
        cond[0x0C:0x2C] = b"QueueCondition".ljust(32, b"\0")
        struct.pack_into("<II", cond, 0x34, 0x1002, 1)
        cond_payload = (
            struct.pack("<2I", 1, 1)
            + cond
            + struct.pack("<2I", 0x2001, 0x4004)
        )

        lwcond = bytearray(0x48)
        struct.pack_into("<II", lwcond, 0, 0, 0x5002)
        lwcond[8:0x28] = b"LightCondition".ljust(32, b"\0")
        struct.pack_into("<II", lwcond, 0x30, 0x6001, 1)
        struct.pack_into("<2I", lwcond, 0x38, 0x2001, 0x4005)
        struct.pack_into("<2I", lwcond, 0x40, 0x81000101, 0x81000201)
        lwcond_payload = struct.pack("<2I", 6, 1) + lwcond

        registry = KernelObjectRegistry.parse(
            core_with_notes(
                [
                    Note("CONDVAR_INFO", 0, cond_payload, 0),
                    Note("LWCONDVAR_INFO", 0, lwcond_payload, 1),
                ]
            )
        )
        condition = registry.resolve(0x5001)
        light = registry.resolve(0x5002)
        self.assertEqual(condition.state["associated_mutex_uid"], 0x1002)
        self.assertEqual(condition.waiters[0].thread_uid, 0x4004)
        self.assertEqual(light.state["associated_lwmutex_uid"], 0x6001)
        self.assertEqual(light.waiters[0].thread_uid, 0x4005)


if __name__ == "__main__":
    unittest.main()
