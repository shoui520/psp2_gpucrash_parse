from __future__ import annotations

from typing import Optional


# Selectors consumed by the firmware 3.68 ES4 coredump formatter and native
# register-dump analysis. The GPU producer captures 109 selectors through 0x7a;
# these consumers assign semantics only to this subset. Unlisted selectors
# remain unnamed rather than inheriting speculative SGX debug-register labels.
INDEXED_SELECTOR_METADATA = {
    0x00: {
        "name": "pipeline phase flags",
        "roles": ["activity"],
        "description": "Vertex/fragment phase flags.",
    },
    0x02: {
        "name": "active pipeline units",
        "roles": ["activity"],
        "description": "Active-low PDS, VDM, MTE, TE, ISP, TSP, and PBE unit state.",
    },
    0x11: {
        "name": "PDS tile position",
        "roles": ["tile-position"],
        "description": "PDS tile coordinate.",
    },
    0x12: {
        "name": "VDM pipe 0 draw number",
        "roles": ["draw-counter"],
        "description": "VDM pipe 0 draw counter.",
    },
    0x13: {
        "name": "VDM pipe 1 draw number",
        "roles": ["draw-counter"],
        "description": "VDM pipe 1 draw counter.",
    },
    0x14: {
        "name": "VDM pipe 2 draw number",
        "roles": ["draw-counter"],
        "description": "VDM pipe 2 draw counter.",
    },
    0x15: {
        "name": "VDM pipe 3 draw number",
        "roles": ["draw-counter"],
        "description": "VDM pipe 3 draw counter.",
    },
    0x25: {
        "name": "clipper halt signature",
        "roles": ["halt-signature"],
        "description": "Bit 4 is the firmware's possible clipper-halt signature when no BIF fault is present.",
    },
    0x26: {
        "name": "MTE draw number",
        "roles": ["draw-counter"],
        "description": "MTE draw counter.",
    },
    0x2E: {
        "name": "ISP output progress",
        "roles": ["draw-counter", "tile-position"],
        "description": "ISP output draw counter and tile coordinate.",
    },
    0x2F: {
        "name": "ISP input tile position",
        "roles": ["tile-position"],
        "description": "ISP input tile coordinate.",
    },
    0x31: {
        "name": "TSP parameter fetch progress",
        "roles": ["draw-counter", "tile-position"],
        "description": "TSP parameter-fetch draw counter and tile coordinate.",
    },
    0x33: {
        "name": "color output tile position",
        "roles": ["tile-position"],
        "description": "Color-output tile coordinate.",
    },
    0x51: {
        "name": "USSE pipe 0 tile position",
        "roles": ["tile-position"],
        "description": "USSE pipe 0 tile coordinate.",
    },
    0x52: {
        "name": "USSE pipe 1 tile position",
        "roles": ["tile-position"],
        "description": "USSE pipe 1 tile coordinate.",
    },
    0x71: {
        "name": "USSE pipe 2 tile position",
        "roles": ["tile-position"],
        "description": "USSE pipe 2 tile coordinate.",
    },
    0x72: {
        "name": "USSE pipe 3 tile position",
        "roles": ["tile-position"],
        "description": "USSE pipe 3 tile coordinate.",
    },
}


def indexed_selector_metadata(selector: int) -> Optional[dict]:
    metadata = INDEXED_SELECTOR_METADATA.get(selector)
    return dict(metadata) if metadata is not None else None


def indexed_selector_summary(selector: int, values: tuple[int, ...]) -> dict:
    metadata = indexed_selector_metadata(selector)
    return {
        "index": selector,
        "values": list(values),
        "consumer_interpreted": metadata is not None,
        "name": metadata["name"] if metadata is not None else None,
        "roles": list(metadata["roles"]) if metadata is not None else [],
        "description": metadata["description"] if metadata is not None else None,
    }
