from __future__ import annotations

import bisect
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .core import CoreDump, ParseError
from .elfimage import ElfImage, _match_image
from .execution import ExecutionContext


SHT_ARM_EXIDX = 0x70000001
EXIDX_CANTUNWIND = 1


def _prel31(place: int, value: int) -> int:
    offset = value & 0x7FFFFFFF
    if offset & 0x40000000:
        offset -= 0x80000000
    return (place + offset) & 0xFFFFFFFF


def _section_data(image: ElfImage, section: dict) -> bytes:
    start = section["file_offset"]
    end = start + section["size"]
    if start < 0 or end > len(image.data):
        raise ParseError(
            f"{image.path} section {section['name'] or section['index']} exceeds image"
        )
    return image.data[start:end]


def _exidx_entries(image: ElfImage) -> List[dict]:
    entries = []
    for section in image.section_headers():
        if section["type"] != SHT_ARM_EXIDX:
            continue
        data = _section_data(image, section)
        entry_size = section["entry_size"] or 8
        if entry_size < 8 or len(data) % entry_size:
            raise ParseError(f"{image.path} has a malformed ARM.exidx section")
        for offset in range(0, len(data), entry_size):
            function_word, unwind_word = struct.unpack_from("<2I", data, offset)
            place = section["address"] + offset
            entries.append(
                {
                    "function_address": _prel31(place, function_word),
                    "entry_address": place,
                    "unwind_word": unwind_word,
                    "section": section,
                }
            )
    entries.sort(key=lambda item: item["function_address"])
    for index, item in enumerate(entries):
        item["function_end"] = (
            entries[index + 1]["function_address"]
            if index + 1 < len(entries)
            else None
        )
    return entries


def _linked_file_bytes(image: ElfImage, address: int, size: int) -> bytes:
    for load in image.load_segments:
        if (
            load.virtual_address <= address
            and address + size <= load.virtual_address + load.file_size
        ):
            offset = load.file_offset + address - load.virtual_address
            return image.data[offset : offset + size]
    raise ParseError(f"ELF address 0x{address:08x} is not file-backed")


def _compact_bytecode(image: ElfImage, entry: dict) -> dict:
    word = entry["unwind_word"]
    if word == EXIDX_CANTUNWIND:
        return {"status": "cantunwind", "personality": None, "bytecode": []}
    source = "inline"
    if not word & 0x80000000:
        extab_address = _prel31(entry["entry_address"] + 4, word)
        word = struct.unpack("<I", _linked_file_bytes(image, extab_address, 4))[0]
        source = "extab"
    else:
        extab_address = None
    if not word & 0x80000000:
        return {
            "status": "unsupported-generic-personality",
            "personality": None,
            "bytecode": [],
            "source": source,
            "extab_address": extab_address,
        }
    personality = (word >> 24) & 0xF
    if personality == 0:
        bytecode = [(word >> shift) & 0xFF for shift in (16, 8, 0)]
    elif personality in (1, 2) and source == "extab":
        additional_words = (word >> 16) & 0xFF
        bytecode = [(word >> 8) & 0xFF, word & 0xFF]
        if additional_words:
            raw = _linked_file_bytes(
                image, extab_address + 4, additional_words * 4
            )
            for value in struct.unpack(f"<{additional_words}I", raw):
                bytecode.extend((value >> shift) & 0xFF for shift in (24, 16, 8, 0))
    else:
        return {
            "status": "unsupported-compact-personality",
            "personality": personality,
            "bytecode": [],
            "source": source,
            "extab_address": extab_address,
        }
    return {
        "status": "available",
        "personality": personality,
        "bytecode": bytecode,
        "source": source,
        "extab_address": extab_address,
    }


def _uleb128(bytecode: Sequence[int], cursor: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while cursor < len(bytecode) and shift <= 28:
        byte = bytecode[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7
    raise ParseError("truncated or oversized EHABI ULEB128 operand")


def execute_compact_bytecode(
    core: CoreDump,
    registers: Sequence[int],
    stack_base: int,
    stack_end: int,
    bytecode: Sequence[int],
) -> dict:
    """Execute the integer/VSP subset of the standard EHABI virtual machine."""
    state = list(registers)
    if len(state) != 16:
        raise ParseError("EHABI register state must contain r0 through r15")
    cursor = 0
    operations = []
    reads = []
    pc_restored = False

    def pop(register_numbers: Sequence[int]) -> None:
        nonlocal pc_restored
        address = state[13]
        restored_sp = None
        values = []
        for register_number in register_numbers:
            if not stack_base <= address or address + 4 > stack_end:
                raise ParseError(
                    f"EHABI pop at 0x{address:08x} leaves the retained thread stack"
                )
            value = struct.unpack("<I", core.read_memory(address, 4))[0]
            reads.append(
                {"address": address, "register": register_number, "value": value}
            )
            values.append((register_number, value))
            address += 4
        for register_number, value in values:
            state[register_number] = value
            if register_number == 13:
                restored_sp = value
            elif register_number == 15:
                pc_restored = True
        state[13] = restored_sp if restored_sp is not None else address

    while cursor < len(bytecode):
        opcode = bytecode[cursor]
        cursor += 1
        if opcode <= 0x3F:
            amount = ((opcode & 0x3F) << 2) + 4
            state[13] += amount
            operations.append(f"vsp += 0x{amount:x}")
        elif opcode <= 0x7F:
            amount = ((opcode & 0x3F) << 2) + 4
            state[13] -= amount
            operations.append(f"vsp -= 0x{amount:x}")
        elif opcode <= 0x8F:
            if cursor >= len(bytecode):
                raise ParseError("truncated EHABI register mask")
            mask = ((opcode & 0xF) << 8) | bytecode[cursor]
            cursor += 1
            if mask == 0:
                raise ParseError("EHABI refused-to-unwind opcode")
            selected = [number for number in range(4, 16) if mask & (1 << (number - 4))]
            pop(selected)
            operations.append("pop {" + ",".join(f"r{x}" for x in selected) + "}")
        elif opcode <= 0x9F:
            register_number = opcode & 0xF
            if register_number in (13, 15):
                raise ParseError(f"reserved EHABI set-vsp opcode 0x{opcode:02x}")
            state[13] = state[register_number]
            operations.append(f"vsp = r{register_number}")
        elif opcode <= 0xA7:
            selected = list(range(4, 5 + (opcode & 7)))
            pop(selected)
            operations.append("pop {" + ",".join(f"r{x}" for x in selected) + "}")
        elif opcode <= 0xAF:
            selected = list(range(4, 5 + (opcode & 7))) + [14]
            pop(selected)
            operations.append("pop {" + ",".join(f"r{x}" for x in selected) + "}")
        elif opcode == 0xB0:
            if not pc_restored:
                state[15] = state[14]
            operations.append("finish")
            return {
                "registers": state,
                "operations": operations,
                "stack_reads": reads,
                "finished": True,
                "pc_restored": pc_restored,
            }
        elif opcode == 0xB1:
            if cursor >= len(bytecode):
                raise ParseError("truncated EHABI low-register mask")
            mask = bytecode[cursor]
            cursor += 1
            if not mask or mask & 0xF0:
                raise ParseError("invalid EHABI low-register mask")
            selected = [number for number in range(4) if mask & (1 << number)]
            pop(selected)
            operations.append("pop {" + ",".join(f"r{x}" for x in selected) + "}")
        elif opcode == 0xB2:
            value, cursor = _uleb128(bytecode, cursor)
            amount = 0x204 + (value << 2)
            state[13] += amount
            operations.append(f"vsp += 0x{amount:x}")
        elif opcode == 0xB3:
            if cursor >= len(bytecode):
                raise ParseError("truncated EHABI VFP-pop operand")
            operand = bytecode[cursor]
            cursor += 1
            count = (operand & 0xF) + 1
            state[13] += count * 8
            operations.append(f"pop {count} VFP registers")
        elif 0xB8 <= opcode <= 0xBF or 0xD0 <= opcode <= 0xD7:
            count = (opcode & 7) + 1
            state[13] += count * 8
            operations.append(f"pop {count} VFP registers")
        elif opcode in (0xC8, 0xC9):
            if cursor >= len(bytecode):
                raise ParseError("truncated EHABI VFP-pop operand")
            operand = bytecode[cursor]
            cursor += 1
            count = (operand & 0xF) + 1
            state[13] += count * 8
            operations.append(f"pop {count} VFP registers")
        else:
            raise ParseError(f"unsupported EHABI opcode 0x{opcode:02x}")
    raise ParseError("EHABI bytecode ended without a finish opcode")


def _frame_location(
    modules,
    images_by_module: Dict[int, List[Tuple[ElfImage, dict]]],
    runtime_pc: int,
    return_address: bool,
) -> dict:
    runtime_location = modules.locate(runtime_pc, executable_only=True)
    if runtime_location is None:
        return {"status": "unmapped-runtime-pc", "runtime_location": None}
    candidates = images_by_module.get(runtime_location.module_index, [])
    if not candidates:
        return {
            "status": "no-matched-image",
            "runtime_location": runtime_location.summary(),
        }
    if len(candidates) != 1:
        return {
            "status": "ambiguous-matched-images",
            "runtime_location": runtime_location.summary(),
        }
    image, association = candidates[0]
    ordinal = runtime_location.segment_number - 1
    if ordinal >= len(image.load_segments):
        return {"status": "segment-mismatch", "runtime_location": runtime_location.summary()}
    load = image.load_segments[ordinal]
    linked_pc = load.virtual_address + (runtime_pc & ~1) - runtime_location.segment_start
    lookup_pc = linked_pc
    if return_address:
        lookup_pc -= 2 if runtime_pc & 1 else 4
    symbol = image.symbol_for(lookup_pc)
    return {
        "status": "mapped",
        "runtime_location": runtime_location.summary(),
        "image": image,
        "association": association,
        "image_path": str(image.path),
        "elf_virtual_address": linked_pc,
        "lookup_elf_address": lookup_pc,
        "symbol": symbol.summary(lookup_pc) if symbol is not None else None,
    }


def unwind_thread(
    core: CoreDump,
    diagnosis: dict,
    thread: dict,
    paths: Sequence[Path],
    max_frames: int = 32,
) -> dict:
    if max_frames < 1 or max_frames > 256:
        raise ParseError("maximum unwind frame count must be between 1 and 256")
    execution = ExecutionContext.parse(core)
    if execution.modules is None:
        raise ParseError("MODULE_INFO is required for EHABI unwinding")
    images_by_module: Dict[int, List[Tuple[ElfImage, dict]]] = {}
    image_reports = []
    for path in paths:
        try:
            image = ElfImage.read(Path(path))
            association = _match_image(image, diagnosis["execution_context"]["modules"])
        except (OSError, ParseError) as exc:
            image_reports.append({"path": str(path), "status": "invalid", "error": str(exc)})
            continue
        report = {
            "path": str(path),
            "status": association["status"],
            "association": association,
            "arm_exidx_sections": sum(
                section["type"] == SHT_ARM_EXIDX for section in image.section_headers()
            ),
        }
        image_reports.append(report)
        if association["status"] == "matched":
            images_by_module.setdefault(association["module_index"], []).append(
                (image, association)
            )
    registers = thread["registers"]
    if registers is None:
        return {
            "status": "registers-not-retained",
            "frames": [],
            "images": image_reports,
            "verified_frame_count": 0,
        }
    if not thread.get("register_context_live", True):
        return {
            "status": "non-live-thread-context",
            "frames": [],
            "images": image_reports,
            "verified_frame_count": 0,
            "stop": {
                "code": "non-live-thread-context",
                "detail": thread.get(
                    "register_context_qualification",
                    "thread register record is not a live execution context",
                ),
            },
            "qualification": (
                "Dormant and dead thread register records are not used as backtrace roots."
            ),
        }
    state = list(registers["gpr"])
    frames = []
    stop = None
    seen = set()
    for frame_index in range(max_frames):
        pc, sp = state[15], state[13]
        key = (pc & ~1, sp)
        if key in seen:
            stop = {"code": "repeated-state", "detail": "PC/SP state repeated"}
            break
        seen.add(key)
        mapped = _frame_location(
            execution.modules, images_by_module, pc, return_address=frame_index > 0
        )
        frame = {
            "index": frame_index,
            "pc": pc,
            "sp": sp,
            "lr": state[14],
            "source": "captured-registers" if frame_index == 0 else "arm-ehabi",
            "runtime_location": mapped.get("runtime_location"),
            "image_path": mapped.get("image_path"),
            "elf_virtual_address": mapped.get("elf_virtual_address"),
            "symbol": mapped.get("symbol"),
        }
        frames.append(frame)
        if mapped["status"] != "mapped":
            stop = {"code": mapped["status"], "frame": frame_index}
            break
        image: ElfImage = mapped["image"]
        try:
            entries = _exidx_entries(image)
        except ParseError as exc:
            stop = {"code": "invalid-arm-exidx", "frame": frame_index, "detail": str(exc)}
            break
        if not entries:
            stop = {"code": "no-arm-exidx", "frame": frame_index}
            break
        starts = [item["function_address"] for item in entries]
        entry_index = bisect.bisect_right(starts, mapped["lookup_elf_address"]) - 1
        if entry_index < 0:
            stop = {"code": "no-exidx-entry", "frame": frame_index}
            break
        entry = entries[entry_index]
        if (
            entry["function_end"] is not None
            and mapped["lookup_elf_address"] >= entry["function_end"]
        ):
            stop = {"code": "no-exidx-entry", "frame": frame_index}
            break
        try:
            program = _compact_bytecode(image, entry)
        except ParseError as exc:
            stop = {"code": "invalid-unwind-program", "frame": frame_index, "detail": str(exc)}
            break
        if program["status"] != "available":
            stop = {"code": program["status"], "frame": frame_index}
            break
        try:
            transition = execute_compact_bytecode(
                core,
                state,
                thread["stack_base"],
                thread["stack_end"],
                program["bytecode"],
            )
        except ParseError as exc:
            stop = {"code": "unwind-step-failed", "frame": frame_index, "detail": str(exc)}
            break
        next_state = transition["registers"]
        if next_state[15] in (0, 0xFFFFFFFF):
            stop = {"code": "end-of-stack", "frame": frame_index}
            break
        if next_state[13] <= sp or next_state[13] > thread["stack_end"]:
            stop = {
                "code": "non-progressing-stack",
                "frame": frame_index,
                "detail": f"SP 0x{sp:08x} -> 0x{next_state[13]:08x}",
            }
            break
        frame["unwind"] = {
            "entry_address": entry["entry_address"],
            "function_address": entry["function_address"],
            "function_end": entry["function_end"],
            "personality": program["personality"],
            "program_source": program["source"],
            "bytecode": program["bytecode"],
            "operations": transition["operations"],
            "stack_reads": transition["stack_reads"],
        }
        state = next_state
    else:
        stop = {"code": "frame-limit", "frame": max_frames - 1}
    return {
        "status": "complete" if stop and stop["code"] == "end-of-stack" else "stopped",
        "frames": frames,
        "images": image_reports,
        "verified_frame_count": max(0, len(frames) - 1),
        "stop": stop,
        "qualification": (
            "Only the captured frame and transitions executed from matched ARM EHABI tables are listed; stack address candidates are not frames."
        ),
    }
