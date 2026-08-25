from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


@dataclass(frozen=True)
class BitField:
    name: str
    mask: int
    shift: int

    def value(self, register_value: int) -> int:
        return (register_value & self.mask) >> self.shift

    @property
    def is_address(self) -> bool:
        if self.shift == 0 or self.name in {"MIN_ADDRESS", "MAX_ADDRESS"}:
            return False
        return (
            self.name in {"ADDR", "ADDRESS"}
            or self.name.endswith("_ADDR")
            or self.name.endswith("_ADDRESS")
            or self.name.endswith("_BASEADDR")
        )

    def address(self, register_value: int) -> int:
        """Return an address field in its register-aligned representation."""
        return register_value & self.mask


@dataclass
class RegisterDefinition:
    offset: int
    names: List[str] = field(default_factory=list)
    fields: List[BitField] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        if not self.names:
            return f"REG_{self.offset:04X}"
        # Header extraction also encounters field enumerants whose numeric
        # value happens to equal a register offset.  Real register names are
        # normally the shortest aliases at that offset; explicit enumerant
        # markers make the distinction unambiguous where present.
        def rank(item: tuple[int, str]) -> tuple[int, int, int]:
            index, name = item
            enumerant = any(
                marker in name
                for marker in (
                    "_VALUE_",
                    "_SIZE_UINT",
                    "_NUM_ENTRIES",
                    "_FAST_SCANDIR_",
                    "_USE_PIPELINE_",
                    "_ZERO_RETURN_",
                )
            )
            return (int(enumerant), len(name), index)

        return min(enumerate(self.names), key=rank)[1]


class RegisterDatabase:
    def __init__(self) -> None:
        self.core: Dict[int, RegisterDefinition] = {}
        self.master: Dict[int, RegisterDefinition] = {}
        self.sources: List[str] = []
        self._install_builtins()

    def _install_builtins(self) -> None:
        core = {
            0x0118: "EUR_CR_EVENT_STATUS2",
            0x012C: "EUR_CR_EVENT_STATUS",
            0x0348: "EUR_CR_VDM_WAIT_FOR_KICK",
            0x0400: "EUR_CR_ISP_STATUS",
            0x0408: "EUR_CR_ISP_RGN_BASE",
            0x0758: "EUR_CR_DPM_DEBUG_STATUS",
            0x0C00: "EUR_CR_BIF_CTRL",
            0x0C04: "EUR_CR_BIF_INT_STAT",
            0x0C08: "EUR_CR_BIF_FAULT",
        }
        master = {
            0x4118: "EUR_CR_MASTER_EVENT_STATUS2",
            0x412C: "EUR_CR_MASTER_EVENT_STATUS",
            0x4348: "EUR_CR_MASTER_VDM_WAIT_FOR_KICK",
            0x4400: "EUR_CR_MASTER_ISP_STATUS",
            0x4408: "EUR_CR_MASTER_ISP_RGN_BASE",
            0x4758: "EUR_CR_MASTER_DPM_DEBUG_STATUS",
            0x4C00: "EUR_CR_MASTER_BIF_CTRL",
            0x4C04: "EUR_CR_MASTER_BIF_INT_STAT",
            0x4C08: "EUR_CR_MASTER_BIF_FAULT",
        }
        directory_offsets = (0x0C84, 0x0C38, 0x0C3C, 0x0C40, 0x0C44, 0x0C48, 0x0C4C, 0x0C50)
        for context_id, offset in enumerate(directory_offsets):
            core[offset] = f"EUR_CR_BIF_DIR_LIST_BASE{context_id}"
            master[offset + 0x4000] = f"EUR_CR_MASTER_BIF_DIR_LIST_BASE{context_id}"
        for offset, name in core.items():
            self.core[offset] = RegisterDefinition(offset, [name])
        for offset, name in master.items():
            self.master[offset] = RegisterDefinition(offset, [name])
        bif_status_fields = [
            BitField("FAULT_REQ", 0x00003FFF, 0),
            BitField("FAULT_TYPE", 0x00070000, 16),
            BitField("FLUSH_COMPLETE", 0x00080000, 19),
        ]
        bif_fault_fields = [
            BitField("CID", 0x0000000F, 0),
            BitField("SB", 0x000001F0, 4),
            BitField("ADDR", 0xFFFFF000, 12),
        ]
        self.core[0x0C04].fields = bif_status_fields
        self.core[0x0C08].fields = bif_fault_fields
        self.master[0x4C04].fields = bif_status_fields
        self.master[0x4C08].fields = bif_fault_fields
        event_fields = [
            BitField("TA_DPM_FAULT", 0x10000000, 28),
            BitField("DPM_OUT_OF_MEMORY_ZLS", 0x02000000, 25),
            BitField("BREAKPOINT", 0x00008000, 15),
            BitField("DPM_OUT_OF_MEMORY_GBL", 0x00000004, 2),
            BitField("DPM_OUT_OF_MEMORY_MT", 0x00000002, 1),
        ]
        self.core[0x012C].fields = list(event_fields)
        self.master[0x412C].fields = list(event_fields) + [
            BitField("ISP_HALT", 0x00020000, 17),
            BitField("ISP_VISIBILITY_FAIL", 0x00010000, 16),
        ]
        status2_fields = [
            BitField("DATA_BREAKPOINT_UNTRAPPED", 0x00000800, 11),
            BitField("DATA_BREAKPOINT_TRAPPED", 0x00000400, 10),
        ]
        self.core[0x0118].fields = list(status2_fields)
        self.master[0x4118].fields = list(status2_fields)
        dpm_debug_fields = [
            BitField("OUTOFMEM_DEADLOCK", 0x00000001, 0),
            BitField("OUTOFMEM_SLEEP", 0x00000002, 1),
            BitField("OUTOFMEM_WAIT", 0x00000004, 2),
        ]
        self.core[0x0758].fields = list(dpm_debug_fields)
        self.master[0x4758].fields = list(dpm_debug_fields)
        page_address_fields = [BitField("ADDR", 0xFFFFF000, 12)]
        region_address_fields = [BitField("ADDR", 0xFFFFFFFC, 2)]
        self.core[0x0408].fields = list(region_address_fields)
        self.master[0x4408].fields = list(region_address_fields)
        for offset in directory_offsets:
            self.core[offset].fields = list(page_address_fields)
            self.master[offset + 0x4000].fields = list(page_address_fields)

    @classmethod
    def load(cls, headers: Iterable[Union[Path, str]] = ()) -> "RegisterDatabase":
        database = cls()
        supplied = [Path(path) for path in headers]
        candidates = supplied or default_register_sources()
        for path in candidates:
            if path.is_file():
                if path.suffix.casefold() == ".json":
                    database.load_register_data(path)
                else:
                    database.load_header(path)
        database._prune_shadowed_fields()
        return database

    def _prune_shadowed_fields(self) -> None:
        """Remove fields borrowed from longer register-name prefixes.

        The original C headers describe both registers and their fields with
        flat preprocessor names.  Without this pass, for example,
        EUR_CR_MASTER_CORE incorrectly acquires all fields belonging to
        EUR_CR_MASTER_CORE_ID and EUR_CR_MASTER_CORE_REVISION.
        """
        for target in (self.core, self.master):
            register_names = {
                name for definition in target.values() for name in definition.names
            }
            for definition in target.values():
                shadow_prefixes = set()
                for base_name in definition.names:
                    prefix = base_name + "_"
                    shadow_prefixes.update(
                        candidate[len(prefix) :]
                        for candidate in register_names
                        if candidate.startswith(prefix)
                    )
                definition.fields = [
                    item
                    for item in definition.fields
                    if not any(
                        item.name == prefix or item.name.startswith(prefix + "_")
                        for prefix in shadow_prefixes
                    )
                ]

    def load_register_data(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError(f"unsupported register-data version in {path}")
        scope = payload.get("scope")
        if scope not in ("core", "master"):
            raise ValueError(f"invalid register-data scope in {path}")
        target = self.core if scope == "core" else self.master
        for record in payload.get("registers", []):
            if not isinstance(record, list) or len(record) != 3:
                raise ValueError(f"invalid register record in {path}")
            offset, names, fields = record
            definition = target.setdefault(offset, RegisterDefinition(offset))
            for name in names:
                if name not in definition.names:
                    definition.names.append(name)
            existing = {field.name for field in definition.fields}
            for name, mask, shift in fields:
                if name not in existing:
                    definition.fields.append(BitField(name, mask, shift))
                    existing.add(name)
        self.sources.append(str(path))

    def load_header(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8", errors="replace")
        macros: Dict[str, int] = {}
        for line in text.splitlines():
            match = re.match(
                r"\s*#define\s+(EUR_CR_[A-Z0-9_]+)\s+"
                r"(0x[0-9A-Fa-f]+|[0-9]+)(?:U|UL|ULL|L)?(?:\s|$)",
                line,
            )
            if match:
                macros[match.group(1)] = int(match.group(2), 0)
        register_names = [
            name
            for name in macros
            if not name.endswith(("_MASK", "_SHIFT", "_SIGNED"))
        ]
        register_names.sort(key=len, reverse=True)
        target = self.master if path.name.lower().startswith("sgxmp") else self.core
        by_name: Dict[str, RegisterDefinition] = {}
        for name in register_names:
            offset = macros[name]
            definition = target.setdefault(offset, RegisterDefinition(offset))
            if name not in definition.names:
                definition.names.append(name)
            by_name[name] = definition
        for register_name in register_names:
            mask_prefix = register_name + "_"
            definition = by_name[register_name]
            existing = {item.name for item in definition.fields}
            for macro_name, mask in macros.items():
                if not macro_name.startswith(mask_prefix) or not macro_name.endswith("_MASK"):
                    continue
                field_name = macro_name[len(mask_prefix) : -len("_MASK")]
                shift = macros.get(mask_prefix + field_name + "_SHIFT")
                if shift is None or field_name in existing:
                    continue
                definition.fields.append(BitField(field_name, mask, shift))
                existing.add(field_name)
        self.sources.append(str(path))

    def definition(self, offset: int, master: bool) -> Optional[RegisterDefinition]:
        return (self.master if master else self.core).get(offset)


def default_register_sources() -> List[Path]:
    base = Path(__file__).resolve().parent / "data"
    return [
        base / "sgx543_gpucrash_registers.json",
        base / "sgxmp_gpucrash_registers.json",
    ]


def default_register_headers() -> List[Path]:
    """Compatibility alias for callers using the former helper name."""
    return default_register_sources()


def decoded_fields(definition: Optional[RegisterDefinition], value: int) -> List[dict]:
    if definition is None:
        return []
    result = []
    for item in definition.fields:
        field_value = item.value(value)
        if field_value == 0:
            continue
        decoded = {
            "name": item.name,
            "value": field_value,
            "mask": item.mask,
            "shift": item.shift,
        }
        if item.is_address:
            decoded["address"] = item.address(value)
        result.append(decoded)
    return result
