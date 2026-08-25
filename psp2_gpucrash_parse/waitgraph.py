from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .core import CoreDump, ParseError
from .debugger import load_debug_state
from .kernel import KernelObject, KernelObjectRegistry
from .registers import RegisterDatabase


def _thread_node(uid: int) -> str:
    return f"thread:{uid:08x}"


def _object_node(uid: int) -> str:
    return f"object:{uid:08x}"


def _canonical_cycle(nodes: Sequence[str]) -> Tuple[str, ...]:
    cycle = list(nodes[:-1] if len(nodes) > 1 and nodes[0] == nodes[-1] else nodes)
    rotations = [tuple(cycle[index:] + cycle[:index]) for index in range(len(cycle))]
    return min(rotations)


def _cycles(adjacency: Dict[str, List[str]]) -> List[List[str]]:
    found = set()

    def visit(start: str, node: str, path: List[str], present: set) -> None:
        if len(path) > 32:
            return
        for target in adjacency.get(node, []):
            if target == start:
                found.add(_canonical_cycle(path + [start]))
            elif target not in present:
                visit(start, target, path + [target], present | {target})

    for node in sorted(adjacency):
        visit(node, node, [node], {node})
    return [list(item) + [item[0]] for item in sorted(found)]


def _chains(
    thread_nodes: Sequence[str], adjacency: Dict[str, List[str]], limit: int = 32
) -> List[dict]:
    result = []
    for start in thread_nodes:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            targets = adjacency.get(node, [])
            if not targets:
                if len(path) > 1:
                    result.append(
                        {"start": start, "path": path, "termination": "terminal"}
                    )
                continue
            for target in targets:
                if target in path:
                    result.append(
                        {
                            "start": start,
                            "path": path + [target],
                            "termination": "cycle",
                        }
                    )
                elif len(path) >= limit:
                    result.append(
                        {
                            "start": start,
                            "path": path,
                            "termination": "depth-limit",
                        }
                    )
                else:
                    stack.append((target, path + [target]))
    return result


def wait_graph(
    core: CoreDump,
    database: RegisterDatabase,
    diagnosis: Optional[dict] = None,
) -> dict:
    if diagnosis is None:
        _info, diagnosis = load_debug_state(core, database)
    threads = diagnosis["execution_context"]["threads"]
    registry = KernelObjectRegistry.parse(core)
    objects = {item.uid: item for item in registry.objects}
    nodes = []
    for thread in threads:
        nodes.append(
            {
                "id": _thread_node(thread["uid"]),
                "kind": "thread",
                "uid": thread["uid"],
                "index": thread["index"],
                "name": thread["name"],
                "status_name": thread["status_name"],
                "wait_type": thread["wait_type"],
                "wait_id": thread["wait_id"],
            }
        )
    for item in registry.objects:
        nodes.append(
            {
                "id": _object_node(item.uid),
                "kind": "object",
                "uid": item.uid,
                "object_kind": item.kind,
                "name": item.name,
                "owner_thread_uid": item.owner_thread_uid,
                "waiting_thread_count": item.waiting_thread_count,
            }
        )

    edge_map: Dict[Tuple[str, str, str], dict] = {}

    def edge(source: str, target: str, relation: str, evidence: str) -> None:
        key = (source, target, relation)
        current = edge_map.get(key)
        if current is None:
            edge_map[key] = {
                "source": source,
                "target": target,
                "relation": relation,
                "evidence": [evidence],
            }
        elif evidence not in current["evidence"]:
            current["evidence"].append(evidence)

    unresolved_waits = []
    for thread in threads:
        wait_id = thread["wait_id"]
        if wait_id in (0, 0xFFFFFFFF):
            continue
        if wait_id in objects:
            edge(
                _thread_node(thread["uid"]),
                _object_node(wait_id),
                "waits-on",
                "THREAD_INFO wait_id",
            )
        else:
            unresolved_waits.append(
                {
                    "thread_uid": thread["uid"],
                    "thread_index": thread["index"],
                    "thread_name": thread["name"],
                    "wait_type": thread["wait_type"],
                    "wait_id": wait_id,
                }
            )
    thread_uids = {thread["uid"] for thread in threads}
    for item in registry.objects:
        for waiter in item.waiters:
            if waiter.thread_uid in thread_uids:
                edge(
                    _thread_node(waiter.thread_uid),
                    _object_node(item.uid),
                    "waits-on",
                    f"{item.note_name} waiter record",
                )
        if item.owner_thread_uid is not None:
            edge(
                _object_node(item.uid),
                _thread_node(item.owner_thread_uid),
                "owned-by",
                f"{item.note_name} owner field",
            )

    relationships = []
    for item in registry.objects:
        associated = item.state.get("associated_mutex_uid")
        if associated is None:
            associated = item.state.get("associated_lwmutex_uid")
        if associated not in (None, 0, 0xFFFFFFFF):
            relationships.append(
                {
                    "source": _object_node(item.uid),
                    "target": _object_node(associated),
                    "relation": "associated-lock",
                    "target_present": associated in objects,
                }
            )

    edges = sorted(
        edge_map.values(),
        key=lambda item: (item["source"], item["target"], item["relation"]),
    )
    adjacency: Dict[str, List[str]] = {}
    for item in edges:
        adjacency.setdefault(item["source"], []).append(item["target"])
    cycles = _cycles(adjacency)
    waiting_thread_nodes = sorted(
        {item["source"] for item in edges if item["relation"] == "waits-on"}
    )
    chains = _chains(waiting_thread_nodes, adjacency)
    node_by_id = {item["id"]: item for item in nodes}
    cycle_reports = [
        {
            "nodes": cycle,
            "path": [node_by_id.get(node, {"id": node}) for node in cycle],
        }
        for cycle in cycles
    ]
    return {
        "schema_version": 1,
        "command": "waits",
        "path": str(core.path),
        "thread_count": len(threads),
        "object_count": len(objects),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "waiting_thread_count": len(waiting_thread_nodes),
        "unresolved_wait_count": len(unresolved_waits),
        "cycle_count": len(cycle_reports),
        "nodes": nodes,
        "edges": edges,
        "relationships": relationships,
        "chains": chains,
        "cycles": cycle_reports,
        "unresolved_waits": unresolved_waits,
        "qualification": (
            "Cycles are reported only from captured thread wait IDs, object waiter records, and mutex owner fields."
        ),
    }


def object_report(
    core: CoreDump, database: RegisterDatabase, uid: int
) -> dict:
    graph = wait_graph(core, database)
    object_id = _object_node(uid)
    matches = [
        item
        for item in graph["nodes"]
        if item["id"] == object_id and item["kind"] == "object"
    ]
    if not matches:
        raise ParseError(f"kernel object UID 0x{uid:08x} is not retained")
    registry = KernelObjectRegistry.parse(core)
    item: Optional[KernelObject] = registry.resolve(uid)
    if item is None:
        raise ParseError(f"kernel object UID 0x{uid:08x} is ambiguous")
    related_edges = [
        edge
        for edge in graph["edges"]
        if edge["source"] == object_id or edge["target"] == object_id
    ]
    related_relationships = [
        relation
        for relation in graph["relationships"]
        if relation["source"] == object_id or relation["target"] == object_id
    ]
    nodes = {node["id"]: node for node in graph["nodes"]}
    for edge in related_edges:
        edge["source_node"] = nodes.get(edge["source"])
        edge["target_node"] = nodes.get(edge["target"])
    return {
        "schema_version": 1,
        "command": "object",
        "path": str(core.path),
        "object": item.summary(),
        "edges": related_edges,
        "relationships": related_relationships,
        "participating_cycles": [
            cycle for cycle in graph["cycles"] if object_id in cycle["nodes"]
        ],
    }
