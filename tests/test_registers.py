import unittest

from psp2_gpucrash_parse.registers import (
    RegisterDatabase,
    decoded_fields,
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
        self.assertEqual(sum(len(item.fields) for item in database.core.values()), 1149)
        self.assertEqual(sum(len(item.fields) for item in database.master.values()), 836)

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

    def test_numeric_macro_aliases_do_not_replace_register_names(self):
        database = RegisterDatabase.load()
        self.assertEqual(
            database.definition(0x0000, master=False).display_name,
            "EUR_CR_CLKGATECTL",
        )
        self.assertEqual(
            database.definition(0x0010, master=False).display_name,
            "EUR_CR_CLKGATECTLOVR2",
        )
        self.assertEqual(
            database.definition(0x0020, master=False).display_name,
            "EUR_CR_CORE_ID",
        )

    def test_longer_register_prefixes_do_not_leak_fields(self):
        database = RegisterDatabase.load()
        master_core = database.definition(0x4000, master=True)
        self.assertEqual([field.name for field in master_core.fields], ["ENABLE"])

    def test_shifted_address_fields_include_register_aligned_address(self):
        database = RegisterDatabase.load()
        definition = database.definition(0x4238, master=True)
        fields = decoded_fields(definition, 0x70113480)
        address = next(field for field in fields if field["name"] == "ADDR")
        self.assertEqual(address["value"], 0x1C044D20)
        self.assertEqual(address["address"], 0x70113480)


if __name__ == "__main__":
    unittest.main()
