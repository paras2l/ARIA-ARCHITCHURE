"""
ARIA — Graph Builder
====================
Loads concept nodes from JSON or raw text files into a live in-memory 3D graph.

Supports:
1. JSON datasets: {"concept_name": [x, y, z], ...}
2. Plain text files (.txt, .md, scripts): Automatically parses vocabulary,
   assigns initial positions, and builds the in-memory graph.
"""

import json
import math
import os
import re
import time
from aria.guard.layer import auto_register, _generate_seed_coordinate

# ---------------------------------------------------------------------------
# In-memory graph store
# ---------------------------------------------------------------------------

_nodes: dict[int, dict] = {}        # id → {id, name, x, y, z}
_name_index: dict[str, int] = {}    # name → id  (fast lookup)
_next_id: int = 0                   # auto-increment counter


def _new_id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


def _distance_from_center(x: float, y: float, z: float) -> float:
    """Euclidean distance from origin (0,0,0)."""
    return math.sqrt(x * x + y * y + z * z)


def _insert_node(name: str, x: float, y: float, z: float) -> int:
    """Create and store a node; return its new ID."""
    node_id = _new_id()
    node = {"id": node_id, "name": name, "x": float(x), "y": float(y), "z": float(z)}
    _nodes[node_id] = node
    _name_index[name] = node_id
    # Also ensure it is registered in Guard Layer
    auto_register(name, x, y, z)
    return node_id


def clear_graph() -> None:
    """Clear all in-memory nodes and reset index."""
    global _nodes, _name_index, _next_id
    _nodes.clear()
    _name_index.clear()
    _next_id = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_graph(filepath: str) -> int:
    """
    Load nodes from either a JSON coordinate file or a raw text file.
    """
    clear_graph()
    print(f"[ARIA] Loading graph from: {filepath}")
    t0 = time.time()

    if not os.path.exists(filepath):
        print(f"[ARIA] ⚠️ File not found: {filepath}")
        return 0

    # Determine file type
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            for concept_name, coords in raw.items():
                if isinstance(coords, (list, tuple)) and len(coords) == 3:
                    try:
                        x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                        if concept_name not in _name_index:
                            _insert_node(concept_name, x, y, z)
                    except (ValueError, TypeError):
                        continue
    else:
        # Raw text file — ingest tokens
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        words = re.findall(r"\b[A-Za-z0-9\-_]{2,}\b", text)
        unique_words = sorted(list(set(words)))

        print(f"[ARIA] Extracted {len(unique_words):,} unique vocabulary tokens from text.")
        for word in unique_words:
            if word not in _name_index:
                gx, gy, gz = _generate_seed_coordinate(word)
                _insert_node(word, gx, gy, gz)

    elapsed_total = time.time() - t0
    total_in_memory = len(_nodes)

    print(f"[ARIA] ✅ Graph ready! Total nodes: {total_in_memory:,} ({elapsed_total:.2f}s)")
    return total_in_memory


def get_node(node_id: int) -> dict | None:
    """Retrieve a single node by ID."""
    return _nodes.get(node_id)


def get_node_by_name(name: str) -> dict | None:
    """Retrieve a node by its name."""
    node_id = _name_index.get(name)
    if node_id is None:
        return None
    return _nodes[node_id]


def get_all_nodes() -> list[dict]:
    """Return all nodes in the graph."""
    return list(_nodes.values())


def add_node(name: str, x: float, y: float, z: float) -> int:
    """Add a new node to the live graph."""
    if name in _name_index:
        return _name_index[name]
    return _insert_node(name, float(x), float(y), float(z))


def graph_stats() -> dict:
    """Summary of current graph state."""
    if not _nodes:
        return {"total_nodes": 0, "max_distance": 0.0, "min_distance": 0.0}

    distances = [
        _distance_from_center(n["x"], n["y"], n["z"])
        for n in _nodes.values()
    ]
    return {
        "total_nodes": len(_nodes),
        "max_distance": round(max(distances), 6),
        "min_distance": round(min(distances), 6),
    }
