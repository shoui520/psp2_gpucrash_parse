import unittest

from psp2_gpucrash_parse.registers import (
    RegisterDatabase,
    default_register_sources,
)


class RegisterDatabaseTests(unittest.TestCase):
    def test_packaged_gpucrash_register_data_is_complete(self):
        sources = default_register_sources()
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(path.is_file() for path in sources))

        database = RegisterDatabase.load()
        self.assertEqual(len(database.core), 462)
        self.assertEqual(len(database.master), 329)
        self.assertEqual(sum(len(item.fields) for item in database.core.values()), 1304)
        self.assertEqual(sum(len(item.fields) for item in database.master.values()), 979)

        screen = database.definition(0x0248, master=False)
        self.assertEqual(screen.display_name, "EUR_CR_MTE_SCREEN")
        self.assertEqual(
            [(field.name, field.mask, field.shift) for field in screen.fields],
            [
                ("PIXXMAX", 0x00000FFF, 0),
                ("PIXYMAX", 0x00FFF000, 12),
            ],
        )
        partial_render = database.definition(0x4658, master=True)
        self.assertEqual(
            partial_render.display_name, "EUR_CR_MASTER_DPM_PARTIAL_RENDER"
        )
        self.assertEqual(partial_render.fields[0].name, "ENABLE")


if __name__ == "__main__":
    unittest.main()
