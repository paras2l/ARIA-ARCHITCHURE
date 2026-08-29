"""
ARIA — Data Router (Dynamic Insertion Coordinator)
===================================================
Entry point for new words, concepts, and batches into ARIA.

Coordinates:
1. Unknown Word Handler & Guard Layer (Coordinate assignment + Binary key)
2. Graph Builder (In-memory node registration)
3. Resonance Block (Directed synapse formation with context text)
"""

from aria.graph.builder   import add_node
from aria.guard.layer     import exists, get_binary, auto_register
from aria.frequency.converter import handle_unknown_word
from aria.resonance.block import run_block2

def _count_connections(connections: dict) -> int:
    return sum(len(v) for v in connections.values())


def route_new_word(
    name:        str,
    x:           float | None = None,
    y:           float | None = None,
    z:           float | None = None,
    connections: dict | None  = None,
    all_nodes:   list | None  = None,
    script_text: str          = "",
    threshold:   float        = 0.5,
) -> dict:
    """
    Route a single new word into ARIA.
    If x, y, z are not provided, auto-assigns via Unknown Word Handler.
    """
    name = name.strip()
    if not name:
        return {"status": "invalid", "name": ""}

    if connections is None:
        connections = {}

    # If word not in Guard Layer or coordinates missing, resolve via Unknown Word Handler
    if not exists(name) or x is None or y is None or z is None:
        info = handle_unknown_word(name, context_text=script_text)
        x, y, z = info["x"], info["y"], info["z"]
        binary = info["binary"]
    else:
        binary, _ = auto_register(name, x, y, z)

    # Ensure in Graph Builder
    try:
        add_node(name, float(x), float(y), float(z))
    except Exception:
        pass

    # Node record
    new_node = {
        "binary": binary,
        "x":      float(x),
        "y":      float(y),
        "z":      float(z),
    }

    # Run directed resonance pass if context script provided
    conn_before = _count_connections(connections)
    if script_text:
        run_block2(name, script_text, connections)
    conn_after = _count_connections(connections)
    connections_made = conn_after - conn_before

    if all_nodes is not None:
        # Check if node with this binary already in all_nodes
        if not any(n.get("binary") == binary for n in all_nodes):
            all_nodes.append(new_node)

    return {
        "status":           "added",
        "name":             name,
        "binary":           binary,
        "x":                float(x),
        "y":                float(y),
        "z":                float(z),
        "connections_made": connections_made,
    }


def route_batch(
    words:       list,
    connections: dict,
    all_nodes:   list,
    script_text: str   = "",
    threshold:   float = 0.5,
) -> dict:
    """Route a batch of words into ARIA."""
    total = len(words)
    added = 0
    skipped = 0
    results = []

    print(f"[ARIA:Router] Batch insert starting — {total:,} words")
    for idx, item in enumerate(words, start=1):
        if isinstance(item, str):
            name = item
            x, y, z = None, None, None
        else:
            name = item.get("name", item.get("word", ""))
            x = item.get("x")
            y = item.get("y")
            z = item.get("z")

        if not name:
            skipped += 1
            continue

        res = route_new_word(
            name=name,
            x=x, y=y, z=z,
            connections=connections,
            all_nodes=all_nodes,
            script_text=script_text,
            threshold=threshold,
        )
        results.append(res)
        added += 1

        if idx % 500 == 0:
            print(f"[ARIA:Router]   {idx:,} / {total:,} words routed")

    print(f"[ARIA:Router] ✅ Batch complete — {added:,} added, {skipped:,} skipped")
    return {
        "total":   total,
        "added":   added,
        "skipped": skipped,
        "results": results,
    }
