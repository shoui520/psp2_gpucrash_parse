from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from .core import CoreDump, ParseError
from .debugger import load_debug_state
from .registers import RegisterDatabase


def _signature(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _pipeline_components(diagnosis: dict) -> dict:
    pipeline = diagnosis["firmware_pipeline"]
    return {
        "master": pipeline["master"],
        "cores": [
            {
                "core": core["core"],
                "active_phases": sorted(core["active_phases"]),
                "active_units": sorted(item["code"] for item in core["active_units"]),
                "draw_counters": {
                    item["code"]: item["draw_number"]
                    for item in core["draw_counters"]
                    if item["applicability"] == "active"
                },
                "tile_progress": {
                    item["code"]: [item["tile_x"], item["tile_y"]]
                    for item in core["tile_progress"]
                    if item["applicability"] == "active"
                },
            }
            for core in pipeline["cores"]
        ],
    }


def _register_components(info) -> dict:
    return {
        "header": list(info.header_words[:3] + info.header_words[7:]),
        "master": [[item.offset, item.value] for item in info.master_registers],
        "per_core": [
            [item.offset, list(item.values)] for item in info.per_core_registers
        ],
        "indexed": [[item.offset, list(item.values)] for item in info.indexed_state],
    }


def _entry(path: Path, database: RegisterDatabase) -> dict:
    core = CoreDump.read(path)
    info, diagnosis = load_debug_state(core, database)
    application = diagnosis["supporting_context"].get("application") or {}
    faults = [
        {
            "page": item["gpu_virtual_address"],
            "requestors": sorted(item["requestor_names"]),
            "conditions": sorted(item["fault_conditions"]),
            "page_access_type": item["page_access_type"],
            "context_id": item["context_id"],
            "scopes": sorted(item["scopes"]),
        }
        for item in diagnosis["fault_groups"]
    ]
    pattern_faults = [
        {key: value for key, value in item.items() if key != "page"}
        for item in faults
    ]
    phases = sorted({item["code"] for item in diagnosis["lockup_phases"]})
    active_units = sorted(
        {
            unit["code"]
            for core_state in diagnosis["firmware_pipeline"]["cores"]
            for unit in core_state["active_units"]
        }
    )
    owner_slots = sorted(
        {
            item["context_attribution"]["owner"]["slot"]
            for item in diagnosis["fault_groups"]
            if (item.get("context_attribution") or {}).get("owner") is not None
        }
    )
    exact_components = {
        "classification": diagnosis["classification"],
        "faults": faults,
        "lockup_phases": phases,
        "event_codes": sorted(item["code"] for item in diagnosis["event_signals"]),
    }
    pattern_components = {**exact_components, "faults": pattern_faults}
    pipeline_components = _pipeline_components(diagnosis)
    register_components = _register_components(info)
    return {
        "path": str(core.path),
        "sha256": core.raw_file_sha256,
        "application": {
            "title_id": application.get("title_id"),
            "title_name": application.get("title_name"),
        },
        "classification": diagnosis["classification"],
        "analysis_code": diagnosis["analysis_code"],
        "faults": faults,
        "fault_pages": sorted({item["page"] for item in faults}),
        "requestors": sorted(
            {requestor for item in faults for requestor in item["requestors"]}
        ),
        "fault_conditions": sorted(
            {condition for item in faults for condition in item["conditions"]}
        ),
        "context_owner_slots": owner_slots,
        "lockup_phases": phases,
        "active_units": active_units,
        "failure_signature": _signature(exact_components),
        "failure_pattern_signature": _signature(pattern_components),
        "pipeline_signature": _signature(pipeline_components),
        "register_signature": _signature(register_components),
        "signature_components": exact_components,
    }


def _groups(entries: Sequence[dict], key: str) -> list:
    grouped = defaultdict(list)
    for index, entry in enumerate(entries):
        grouped[entry[key]].append(index)
    return [
        {"signature": signature, "count": len(indices), "entry_indices": indices}
        for signature, indices in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]


def _facet(entries: Sequence[dict], key: str, plural: bool = False) -> list:
    counts = Counter()
    for entry in entries:
        values: Iterable = entry[key] if plural else (entry[key],)
        for value in values:
            counts[value if value is not None else "<missing>"] += 1

    def order(item):
        value, count = item
        if isinstance(value, int):
            value_key = (0, value)
        else:
            value_key = (1, str(value))
        return (-count, *value_key)

    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=order)
    ]


def triage_dumps(paths: Sequence[Path], database: RegisterDatabase) -> dict:
    if not paths:
        raise ParseError("triage requires at least one dump")
    entries = []
    errors = []
    for path in paths:
        try:
            entries.append(_entry(Path(path), database))
        except (OSError, ParseError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return {
        "schema_version": 1,
        "command": "triage",
        "requested_count": len(paths),
        "dump_count": len(entries),
        "error_count": len(errors),
        "entries": entries,
        "errors": errors,
        "groups": {
            "failure": _groups(entries, "failure_signature"),
            "failure_pattern": _groups(entries, "failure_pattern_signature"),
            "pipeline": _groups(entries, "pipeline_signature"),
            "register": _groups(entries, "register_signature"),
        },
        "facets": {
            "classification": _facet(entries, "classification"),
            "application": _facet(
                [{"application": item["application"].get("title_id")} for item in entries],
                "application",
            ),
            "fault_page": _facet(entries, "fault_pages", plural=True),
            "requestor": _facet(entries, "requestors", plural=True),
            "fault_condition": _facet(entries, "fault_conditions", plural=True),
            "context_owner_slot": _facet(entries, "context_owner_slots", plural=True),
            "lockup_phase": _facet(entries, "lockup_phases", plural=True),
            "active_unit": _facet(entries, "active_units", plural=True),
        },
        "signature_notes": {
            "failure": "Classification, exact fault pages, requestors, conditions, page-access types, contexts, scopes, lockup phases, and event codes.",
            "failure_pattern": "Failure signature with fault pages omitted for address-independent grouping.",
            "pipeline": "Decoded firmware master/core phase and unit state, with applicable draw/tile progress only.",
            "register": "GPU register and indexed tables; capture timestamp and context-owner trailer are excluded.",
        },
    }
