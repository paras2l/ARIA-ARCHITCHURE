"""
ARIA — Stigma Manager
=====================
Rolling 3-slot persistence and high-speed binary serialization for ARIA.

Slot files:
    aria_stigma_slot_{N}.json       (metadata + node positions)
    aria_stigma_slot_{N}_conn.bin   (directed synapses binary packed)

Binary format per edge (18 bytes):
    - tgt_binary length (uint16) + tgt_binary (UTF-8)
    - weight (float32)
    - count (uint16)
    - dx (float32)
    - dy (float32)
    - dz (float32)
"""

import json
import os
import struct
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_SLOTS        = {1, 2, 3}
_SLOT_FILENAME      = "aria_stigma_slot_{}.json"
_SLOT_CONN_FILENAME = "aria_stigma_slot_{}_conn.bin"


def _slot_path(slot: int) -> str:
    return _SLOT_FILENAME.format(slot)


def _slot_conn_path(slot: int) -> str:
    return _SLOT_CONN_FILENAME.format(slot)


def _count_connections(connections: dict) -> int:
    return sum(len(v) for v in connections.values())


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


# ---------------------------------------------------------------------------
# Binary Serialization Helpers
# ---------------------------------------------------------------------------

MAGIC_V2 = b"STG2"


def _save_connections_binary(connections: dict, filepath: str, nodes: list | None = None) -> None:
    """
    Save directed connections dict to binary using Compact Roll-Number Indexing (V2).
    Reduces per-edge size from ~65 bytes to 25 bytes flat (10x smaller).
    """
    if nodes is None:
        nodes = []

    bin_to_idx = {n["binary"]: i for i, n in enumerate(nodes)}
    
    # Filter active sources with valid registered indices
    active_sources = [
        (bin_to_idx[src_b], src_b, edges)
        for src_b, edges in connections.items()
        if src_b in bin_to_idx and edges
    ]

    with open(filepath, "wb") as f:
        # Magic Header for V2 Compact Format
        f.write(MAGIC_V2)
        f.write(struct.pack(">I", len(active_sources)))

        for src_idx, src_b, edges in active_sources:
            # Filter valid target edges
            valid_edges = []
            if isinstance(edges, dict):
                # Hash Map support: edges is dict[tgt_b -> edge_data]
                for tgt_b, e in edges.items():
                    if tgt_b in bin_to_idx:
                        valid_edges.append((bin_to_idx[tgt_b], e))
            else:
                # List support
                for e in edges:
                    tgt_b = e.get("to", "")
                    if tgt_b in bin_to_idx:
                        valid_edges.append((bin_to_idx[tgt_b], e))

            f.write(struct.pack(">II", src_idx, len(valid_edges)))
            for tgt_idx, edge in valid_edges:
                weight = float(edge.get("weight", 0.0))
                count = int(edge.get("count", edge.get("co_count", 1)))
                dx = float(edge.get("dx", 0.0))
                dy = float(edge.get("dy", 0.0))
                dz = float(edge.get("dz", 0.0))
                polarity = int(edge.get("polarity", 1))
                # Pack exactly 25 bytes: tgt_idx (I=4), weight (f=4), count (I=4), dx (f=4), dy (f=4), dz (f=4), polarity (b=1)
                f.write(struct.pack(">IfIfffb", tgt_idx, weight, count, dx, dy, dz, polarity))


def _load_connections_binary(filepath: str, nodes: list | None = None) -> dict:
    """
    Load directed connections dict from binary file.
    Supports both V2 Compact Roll-Number format and V1 Legacy String format.
    """
    connections: dict = {}
    if not os.path.isfile(filepath):
        return connections

    idx_to_bin = [n["binary"] for n in nodes] if nodes else []

    with open(filepath, "rb") as f:
        header = f.read(4)
        if len(header) < 4:
            return connections

        # -------------------------------------------------------------------
        # V2 Compact Roll-Number Format (Magic: b"STG2")
        # -------------------------------------------------------------------
        if header == MAGIC_V2 and idx_to_bin:
            total_src_b = f.read(4)
            if len(total_src_b) < 4:
                return connections
            total_src = struct.unpack(">I", total_src_b)[0]
            num_nodes = len(idx_to_bin)

            for _ in range(total_src):
                src_meta = f.read(8)
                if len(src_meta) < 8:
                    break
                src_idx, num_edges = struct.unpack(">II", src_meta)
                if src_idx >= num_nodes:
                    # Skip corrupt index
                    f.seek(num_edges * 25, os.SEEK_CUR)
                    continue

                src_binary = idx_to_bin[src_idx]
                edges = []

                for _ in range(num_edges):
                    edge_data = f.read(25)
                    if len(edge_data) < 25:
                        break
                    tgt_idx, weight, count, dx, dy, dz, polarity = struct.unpack(">IfIfffb", edge_data)
                    if tgt_idx < num_nodes:
                        edges.append({
                            "to": idx_to_bin[tgt_idx],
                            "weight": round(weight, 6),
                            "count": count,
                            "dx": round(dx, 6),
                            "dy": round(dy, 6),
                            "dz": round(dz, 6),
                            "polarity": polarity,
                        })
                connections[src_binary] = edges

            return connections

        # -------------------------------------------------------------------
        # V1 Legacy Format (String to String backward compatibility)
        # -------------------------------------------------------------------
        total_src = struct.unpack(">I", header)[0]
        for _ in range(total_src):
            src_len_b = f.read(2)
            if len(src_len_b) < 2:
                break
            src_len = struct.unpack(">H", src_len_b)[0]
            src_binary = f.read(src_len).decode("utf-8", errors="ignore")

            edges_len_b = f.read(4)
            if len(edges_len_b) < 4:
                break
            num_edges = struct.unpack(">I", edges_len_b)[0]

            edges = []
            for _ in range(num_edges):
                tgt_len_b = f.read(2)
                if len(tgt_len_b) < 2:
                    break
                tgt_len = struct.unpack(">H", tgt_len_b)[0]
                tgt_binary = f.read(tgt_len).decode("utf-8", errors="ignore")

                # Try 21 bytes (uint32 count) or 19 bytes (uint16 count)
                edge_data = f.read(21)
                if len(edge_data) < 21:
                    break
                try:
                    weight, count, dx, dy, dz, polarity = struct.unpack(">fIfffb", edge_data)
                except struct.error:
                    weight, count, dx, dy, dz, polarity = struct.unpack(">fHfffb", edge_data[:19])
                    f.seek(-2, os.SEEK_CUR)

                edges.append({
                    "to": tgt_binary,
                    "weight": round(weight, 6),
                    "count": count,
                    "dx": round(dx, 6),
                    "dy": round(dy, 6),
                    "dz": round(dz, 6),
                    "polarity": polarity,
                })
            connections[src_binary] = edges

    return connections


# ---------------------------------------------------------------------------
# Public API: save, load, list_saves
# ---------------------------------------------------------------------------

def save(slot: int, connections: dict, nodes: list) -> bool:
    """Save ARIA's current state to a slot."""
    if slot not in _VALID_SLOTS:
        print(f"[ARIA:Stigma] ❌ Invalid slot {slot} — must be 1, 2, or 3")
        return False

    total_nodes       = len(nodes)
    total_connections = _count_connections(connections)
    saved_at          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    json_filepath = _slot_path(slot)
    conn_filepath = _slot_conn_path(slot)

    payload = {
        "slot":              slot,
        "saved_at":          saved_at,
        "total_nodes":       total_nodes,
        "total_connections": total_connections,
        "nodes":             nodes,
    }

    try:
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        _save_connections_binary(connections, conn_filepath, nodes=nodes)
    except (OSError, Exception) as e:
        print(f"[ARIA:Stigma] ❌ Failed to write slot {slot}: {e}")
        return False

    json_size = os.path.getsize(json_filepath) if os.path.exists(json_filepath) else 0
    conn_size = os.path.getsize(conn_filepath) if os.path.exists(conn_filepath) else 0

    print(f"[ARIA:Stigma] 💾 Slot {slot} saved — "
          f"{total_nodes:,} nodes, {total_connections:,} directed synapses ({saved_at})\n"
          f"[ARIA:Stigma]    Nodes JSON      : {json_filepath} ({_format_size(json_size)})\n"
          f"[ARIA:Stigma]    Connections BIN : {conn_filepath} ({_format_size(conn_size)})")
    return True


def load(slot: int) -> dict | None:
    """Load a previously saved slot."""
    if slot not in _VALID_SLOTS:
        print(f"[ARIA:Stigma] ❌ Invalid slot {slot} — must be 1, 2, or 3")
        return None

    json_filepath = _slot_path(slot)
    conn_filepath = _slot_conn_path(slot)

    if not os.path.isfile(json_filepath):
        print(f"[ARIA:Stigma] ⚠️  Slot {slot} is empty — file not found: {json_filepath}")
        return None

    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[ARIA:Stigma] ❌ Failed to read slot {slot}: {e}")
        return None

    nodes       = data.get("nodes", [])
    saved_at    = data.get("saved_at", "unknown")
    total_nodes = data.get("total_nodes", len(nodes))

    if os.path.isfile(conn_filepath):
        try:
            connections = _load_connections_binary(conn_filepath, nodes=nodes)
        except Exception as e:
            print(f"[ARIA:Stigma] ⚠️ Error reading binary connections: {e}")
            connections = data.get("connections", {})
    else:
        connections = data.get("connections", {})

    total_conns = _count_connections(connections)

    print(f"[ARIA:Stigma] 📂 Slot {slot} loaded — "
          f"{total_nodes:,} nodes, {total_conns:,} directed synapses (saved: {saved_at})")

    return {
        "connections": connections,
        "nodes":       nodes,
    }


def list_saves() -> list:
    """List all save slots and their metadata."""
    results = []
    print(f"[ARIA:Stigma] ─── Save Slots ────────────────────────────")

    for slot in sorted(_VALID_SLOTS):
        json_filepath = _slot_path(slot)
        conn_filepath = _slot_conn_path(slot)
        entry = {
            "slot":        slot,
            "occupied":    False,
            "saved_at":    None,
            "nodes":       None,
            "connections": None,
            "filepath":    json_filepath,
        }

        if os.path.isfile(json_filepath):
            try:
                with open(json_filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                entry["occupied"]    = True
                entry["saved_at"]    = data.get("saved_at", "unknown")
                entry["nodes"]       = data.get("total_nodes", len(data.get("nodes", [])))
                entry["connections"] = data.get("total_connections", 0)

                json_sz = os.path.getsize(json_filepath)
                conn_sz = os.path.getsize(conn_filepath) if os.path.isfile(conn_filepath) else 0
                size_info = f" [{_format_size(json_sz)} JSON, {_format_size(conn_sz)} BIN]"

                print(f"[ARIA:Stigma]   Slot {slot} ✅  "
                      f"saved={entry['saved_at']}  "
                      f"nodes={entry['nodes']:,}  "
                      f"synapses={entry['connections']:,}{size_info}")
            except Exception:
                entry["occupied"] = False
                print(f"[ARIA:Stigma]   Slot {slot} ⚠️  corrupted")
        else:
            print(f"[ARIA:Stigma]   Slot {slot} — empty")

        results.append(entry)

    print(f"[ARIA:Stigma] ──────────────────────────────────────────")
    return results
