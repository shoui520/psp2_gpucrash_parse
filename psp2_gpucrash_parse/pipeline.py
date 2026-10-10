from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .gpu import GpuInfo


# SceGpuCoredump's 3.68 ES4 diagnostic table tests these bits in indexed
# selector 2.  A unit is active when its bit is clear.  Selector values of all
# ones are the idle/unavailable state seen on cores which were not processing.
_ACTIVE_UNIT_BITS = (
    (0x01000000, "pds", "PDS", "shared"),
    (0x20000000, "vdm", "VDM", "vertex"),
    (0x00000002, "vertex-index-fetch", "Vertex Index Fetch", "vertex"),
    (0x04000000, "mte", "MTE", "vertex"),
    (0x08000000, "te", "TE", "vertex"),
    (0x40000000, "isp-parameter-fetch", "ISP Parameter Fetch", "fragment"),
    (0x10000000, "isp", "ISP", "fragment"),
    (0x00100000, "tsp", "TSP", "fragment"),
    (0x00800000, "pbe", "PBE", "fragment"),
)

_DRAW_SELECTORS = (
    (0x26, "mte", "MTE", 0x00001FFF, 0),
    (0x12, "vdm-pipe-0", "VDM pipe 0", 0x1FFF0000, 16),
    (0x13, "vdm-pipe-1", "VDM pipe 1", 0x1FFF0000, 16),
    (0x14, "vdm-pipe-2", "VDM pipe 2", 0x1FFF0000, 16),
    (0x15, "vdm-pipe-3", "VDM pipe 3", 0x1FFF0000, 16),
    (0x2E, "isp-output", "ISP output", 0x1FFF0000, 16),
    (0x31, "tsp-parameter-fetch", "TSP parameter fetch", 0x1FFF0000, 16),
)

# The firmware uses two coordinate formatters. USSE/PDS selectors use the
# swapped formatter: X in bits 7:0, Y in bits 15:8. The ordinary formatter
# uses X in bits 15:8, Y in bits 7:0. A tile is 16 by 16 pixels.
_TILE_SELECTORS = (
    (0x51, "usse-pipe-0", "USSE pipe 0", True),
    (0x52, "usse-pipe-1", "USSE pipe 1", True),
    (0x71, "usse-pipe-2", "USSE pipe 2", True),
    (0x72, "usse-pipe-3", "USSE pipe 3", True),
    (0x11, "pds", "PDS", True),
    (0x2F, "isp-input", "ISP input", False),
    (0x2E, "isp-output", "ISP output", False),
    (0x31, "tsp-parameter-fetch", "TSP parameter fetch", False),
    (0x33, "color-output", "Color output", False),
)

_ZLS_FLAGS = (
    (0x00000001, "z-only-render"),
    (0x00000002, "force-depth-load"),
    (0x00000004, "force-depth-store"),
    (0x00001000, "tiled-load"),
    (0x00002000, "stencil-load"),
    (0x00004000, "depth-load"),
    (0x00008000, "mask-load"),
    (0x00010000, "tiled-store"),
    (0x00020000, "stencil-store"),
    (0x00040000, "depth-store"),
    (0x00080000, "mask-store"),
    (0x00100000, "external-depth-buffer"),
)


def _indexed_table(info: GpuInfo) -> Dict[int, Tuple[int, ...]]:
    return {item.offset: item.values for item in info.indexed_state}


def _indexed_value(
    table: Dict[int, Tuple[int, ...]], selector: int, core_index: int
) -> Optional[int]:
    values = table.get(selector)
    if values is None or core_index >= len(values):
        return None
    return values[core_index]


def _core_value(info: GpuInfo, offset: int, core_index: int) -> Optional[int]:
    values = info.core_values(offset)
    if values is None or core_index >= len(values):
        return None
    return values[core_index]


def _tile(selector: int, code: str, name: str, value: int, swapped: bool) -> dict:
    packed = value & 0xFFFF
    if swapped:
        x = packed & 0xFF
        y = (packed >> 8) & 0xFF
    else:
        x = (packed >> 8) & 0xFF
        y = packed & 0xFF
    return {
        "code": code,
        "name": name,
        "selector": selector,
        "raw_value": value,
        "tile_x": x,
        "tile_y": y,
        "pixel_x": x * 16,
        "pixel_y": y * 16,
    }


def _surface(value: Optional[int], enabled: bool, stencil: bool = False) -> dict:
    return {
        "raw_value": value,
        "address": value & 0xFFFFFFF0 if value is not None else None,
        "enabled": enabled,
        "register_enable": bool(value & 1) if stencil and value is not None else None,
    }


def _zls_state(info: GpuInfo, core_index: int) -> Optional[dict]:
    control = _core_value(info, 0x0480, core_index)
    if control is None:
        return None
    external = bool(control & 0x00100000)
    extent = (control & 0x000007F8) >> 3
    flags = [name for mask, name in _ZLS_FLAGS if control & mask]
    return {
        "control": control,
        "flags": flags,
        "external_depth_buffer": external,
        "extent_field": extent,
        "width_pixels": (extent + 1) * 32 if external else None,
        "depth_load_format": (control >> 21) & 0x7,
        "depth_store_format": (control >> 24) & 0x7,
        "surfaces": {
            "depth_load": _surface(
                _core_value(info, 0x0484, core_index),
                bool(control & (0x00004000 | 0x00000002)),
            ),
            "depth_store": _surface(
                _core_value(info, 0x0488, core_index),
                bool(control & (0x00040000 | 0x00000004)),
            ),
            "stencil_load": _surface(
                _core_value(info, 0x048C, core_index),
                bool(control & 0x00002000),
                stencil=True,
            ),
            "stencil_store": _surface(
                _core_value(info, 0x0490, core_index),
                bool(control & 0x00020000),
                stencil=True,
            ),
        },
    }


def _usse_data_master_usage(info: GpuInfo, core_index: int) -> List[dict]:
    # GpuCoreDumpPrintUssePipe reads sixteen two-bit slots, least significant
    # first. These are data-master tags, not PCs or instruction opcodes.
    names = ("vertex", "fragment", "idle", "service")
    symbols = "VP.S"
    pipes = []
    for pipe, offset in enumerate((0x0AA4, 0x0AA8, 0x0B80, 0x0B84)):
        value = _core_value(info, offset, core_index)
        if value is None:
            continue
        slots = [(value >> (index * 2)) & 3 for index in range(16)]
        pipes.append({
            "pipe": pipe, "register_offset": offset, "raw_value": value,
            "usage": "".join(symbols[slot] for slot in slots),
            "slots": [names[slot] for slot in slots],
            "counts": {name: slots.count(index) for index, name in enumerate(names)},
        })
    return pipes


def _core_pipeline(
    info: GpuInfo, table: Dict[int, Tuple[int, ...]], core_index: int
) -> dict:
    phase_value = _indexed_value(table, 0x00, core_index)
    phases: List[str] = []
    if phase_value is not None:
        if phase_value & 0x1:
            phases.append("vertex")
        if phase_value & 0x2:
            phases.append("fragment")

    unit_value = _indexed_value(table, 0x02, core_index)
    active_units = []
    if unit_value is not None and unit_value != 0xFFFFFFFF:
        active_units = [
            {"code": code, "name": name, "stage": stage, "inactive_mask": mask}
            for mask, code, name, stage in _ACTIVE_UNIT_BITS
            if not unit_value & mask
        ]

    draw_counters = []
    for selector, code, name, mask, shift in _DRAW_SELECTORS:
        value = _indexed_value(table, selector, core_index)
        if value is not None:
            counter = (value & mask) >> shift
            # MTE/VDM use PrintDrawNumber (zero or count - 1); ISP/TSP use
            # PrintUnsigned directly. Preserve both the field and display.
            biased = code == "mte" or code.startswith("vdm-pipe-")
            draw_counters.append(
                {
                    "code": code,
                    "name": name,
                    "selector": selector,
                    "raw_value": value,
                    "counter_value": counter,
                    "draw_number": max(0, counter - 1) if biased else counter,
                    "encoding": "zero-or-draw-plus-one" if biased else "draw-number",
                    "counter_mask": mask >> shift,
                }
            )

    counter_by_code = {
        item["code"]: item["counter_value"] for item in draw_counters
    }
    counter_checks = []
    for code, name, members in (
        (
            "ta-draw-counters",
            "TA MTE/VDM draw counters",
            ("mte", "vdm-pipe-0", "vdm-pipe-1", "vdm-pipe-2", "vdm-pipe-3"),
        ),
        (
            "fragment-draw-counters",
            "3D ISP/TSP draw counters",
            ("isp-output", "tsp-parameter-fetch"),
        ),
    ):
        values = {
            member: counter_by_code[member]
            for member in members
            if member in counter_by_code
        }
        if values:
            counter_checks.append(
                {
                    "code": code,
                    "name": name,
                    "values": values,
                    "basis": "raw counter fields",
                    "aligned": len(set(values.values())) == 1,
                }
            )

    tiles = []
    for selector, code, name, swapped in _TILE_SELECTORS:
        value = _indexed_value(table, selector, core_index)
        if value is not None:
            tiles.append(_tile(selector, code, name, value, swapped))
    usse_tiles = [
        (item["tile_x"], item["tile_y"])
        for item in tiles
        if item["code"].startswith("usse-pipe-")
    ]
    tile_checks = []
    if usse_tiles:
        tile_checks.append(
            {
                "code": "usse-pipe-tiles",
                "name": "USSE pipe tile positions",
                "values": [list(value) for value in usse_tiles],
                "aligned": len(set(usse_tiles)) == 1,
            }
        )

    screen = _core_value(info, 0x0248, core_index)
    render_bounds = None
    if screen is not None:
        maximum_x = screen & 0xFFF
        maximum_y = (screen >> 12) & 0xFFF
        render_bounds = {
            "register_value": screen,
            "maximum_pixel_x": maximum_x,
            "maximum_pixel_y": maximum_y,
            "width_pixels": maximum_x + 1,
            "height_pixels": maximum_y + 1,
        }

    status2 = _core_value(info, 0x04E4, core_index)
    isp_progress = None
    if status2 is not None:
        isp_progress = {
            "register_value": status2,
            "primitive_number": (status2 >> 16) & 0x1F,
            "tile_x": status2 & 0xFF,
            "tile_y": (status2 >> 8) & 0xFF,
            "pixel_x": (status2 & 0xFF) * 16,
            "pixel_y": ((status2 >> 8) & 0xFF) * 16,
        }

    region_base = _core_value(info, 0x0408, core_index)
    return {
        "core": core_index,
        "phase_selector": 0x00,
        "phase_value": phase_value,
        "active_phases": phases,
        "unit_selector": 0x02,
        "unit_value": unit_value,
        "active_units": active_units,
        "usse_data_master_usage": _usse_data_master_usage(info, core_index),
        "draw_counters": draw_counters,
        "draw_counter_checks": counter_checks,
        "tile_progress": tiles,
        "tile_checks": tile_checks,
        "render_bounds": render_bounds,
        "isp_progress": isp_progress,
        "region_header_base_raw": region_base,
        "region_header_base": region_base & 0xFFFFFFFC if region_base is not None else None,
        "zls": _zls_state(info, core_index),
    }


def firmware_pipeline_diagnosis(info: GpuInfo) -> dict:
    """Decode the progress diagnostics used by the 3.68 ES4 coredump module."""

    table = _indexed_table(info)
    cores = [_core_pipeline(info, table, index) for index in range(info.core_count)]
    master_status2 = info.master_value(0x4118)
    transfer_status = info.master_value(0x4178)
    partial_render_status = info.master_value(0x4658)
    active_cores = [item["core"] for item in cores if item["active_phases"]]
    return {
        "decoder": "ES4 firmware diagnostic tables (3.68 reference module)",
        "indexed_selectors_present": sorted(table),
        "active_core_count": len(active_cores),
        "active_cores": active_cores,
        "active_unit_count": sum(len(item["active_units"]) for item in cores),
        "phase_active_unit_count": sum(
            len(item["active_units"]) for item in cores if item["active_phases"]
        ),
        "master": {
            "event_status2": master_status2,
            "transfer_status": transfer_status,
            "transfer_active": bool(transfer_status & 1) if transfer_status is not None else None,
            "partial_render_status": partial_render_status,
            "partial_render_active": bool(partial_render_status & 1)
            if partial_render_status is not None
            else None,
            "partial_render_deadlock": bool(master_status2 & 0x4)
            if master_status2 is not None
            else None,
        },
        "cores": cores,
        "notes": [
            "Draw numbers are retained 13-bit hardware counters and may wrap.",
            "MTE/VDM draw numbers use the firmware's zero-or-counter-minus-one formatter; raw counter fields are retained.",
            "Tile coordinates are hardware progress positions; each tile is 16x16 pixels.",
            "USSE slots are data-master tags, not shader instructions or instruction addresses.",
            "Active-unit bits are low-active in indexed selector 2.",
        ],
    }
