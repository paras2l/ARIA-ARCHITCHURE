"""
ARIA — Guard Layer
==================
Dynamic entry gate, symbol table, and vocabulary store for ARIA.

Every word/concept that enters ARIA passes through here first.
Each word is encoded to a 7-bit binary signature and stored under that key.

Storage layout (in _store):
    { binary_signature: {"x": float, "y": float, "z": float} }

    Example key : "1100011 1100001 1110100"   (= "cat")
    Example val : {"x": 1.2, "y": -0.5, "z": 3.1}

No word string is stored inside _store — only the binary key.
Use _from_binary() to recover the original word from a key.
"""

import math
import hashlib
import random

# ---------------------------------------------------------------------------
# Internal store
# ---------------------------------------------------------------------------

_store: dict[str, dict] = {}
# binary_signature → {"x": float, "y": float, "z": float}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_binary(word: str) -> str:
    """
    Convert a word to its binary-ASCII signature.
    Each character → ASCII decimal → 7-bit binary string.
    Parts joined by a single space.
    """
    return " ".join(format(ord(ch), "07b") for ch in word)


def _from_binary(binary_sig: str) -> str:
    """
    Decode a binary-ASCII signature back to the original word.
    Reverses _to_binary() exactly.
    """
    try:
        return "".join(chr(int(b, 2)) for b in binary_sig.split())
    except Exception:
        return binary_sig


def _generate_seed_coordinate(word: str) -> tuple[float, float, float]:
    """
    Generate a deterministic, nicely distributed pseudo-random 3D coordinate
    for a new word when no context coordinate is provided.
    Uses SHA-256 hash digest so same word always gets same initial seed.
    """
    h = hashlib.sha256(word.encode("utf-8")).digest()
    # Extract three 32-bit floats scaled between -1.0 and 1.0
    val_x = int.from_bytes(h[0:4], "big") / (2**32 - 1)
    val_y = int.from_bytes(h[4:8], "big") / (2**32 - 1)
    val_z = int.from_bytes(h[8:12], "big") / (2**32 - 1)
    
    # Map to [-1.0, 1.0] with 6 decimal places
    x = round((val_x * 2.0) - 1.0, 6)
    y = round((val_y * 2.0) - 1.0, 6)
    z = round((val_z * 2.0) - 1.0, 6)
    return x, y, z


# ---------------------------------------------------------------------------
# Public API: Registration & Lookups
# ---------------------------------------------------------------------------

def auto_register(
    word: str,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
) -> tuple[str, bool]:
    """
    Auto-register a word in the Guard Layer.
    
    If the word already exists, returns (binary_signature, False).
    If it is new:
        - Uses (x, y, z) if provided.
        - Otherwise generates deterministic seed coordinates.
        - Stores in _store and returns (binary_signature, True).
    """
    binary_key = _to_binary(word)
    if binary_key in _store:
        return binary_key, False

    if x is None or y is None or z is None:
        gx, gy, gz = _generate_seed_coordinate(word)
    else:
        gx, gy, gz = float(x), float(y), float(z)

    _store[binary_key] = {
        "x": gx,
        "y": gy,
        "z": gz,
    }
    return binary_key, True


def add_word(name: str, x: float, y: float, z: float) -> bool:
    """Register a new word in the Guard Layer."""
    binary_key = _to_binary(name)
    if binary_key in _store:
        return False

    _store[binary_key] = {
        "x": float(x),
        "y": float(y),
        "z": float(z),
    }
    return True


def get_frequency(word: str) -> dict | None:
    """Return the (x, y, z) frequency position of a word."""
    entry = _store.get(_to_binary(word))
    if entry is None:
        return None
    return {"x": entry["x"], "y": entry["y"], "z": entry["z"]}


def get_frequency_by_binary(binary: str) -> dict | None:
    """Get x,y,z directly from binary signature."""
    entry = _store.get(binary)
    if entry is None:
        return None
    return {"x": entry["x"], "y": entry["y"], "z": entry["z"]}


def get_binary(word: str) -> str | None:
    """Return the binary-ASCII signature of a registered word."""
    binary_key = _to_binary(word)
    if binary_key not in _store:
        return None
    return binary_key


def exists(word: str) -> bool:
    """Check whether a word is registered in the Guard Layer."""
    return _to_binary(word) in _store


def exists_binary(binary: str) -> bool:
    """Check whether a binary key exists in the Guard Layer."""
    return binary in _store


# ---------------------------------------------------------------------------
# Public API: Position Updates (Dynamic Drift)
# ---------------------------------------------------------------------------

def update_frequency(name: str, x: float, y: float, z: float) -> bool:
    """Update x,y,z position of an existing word by name."""
    binary = _to_binary(name)
    if binary not in _store:
        return False
    _store[binary]["x"] = float(x)
    _store[binary]["y"] = float(y)
    _store[binary]["z"] = float(z)
    return True


def update_word_position(name: str, x: float, y: float, z: float) -> bool:
    """Alias for update_frequency."""
    return update_frequency(name, x, y, z)


def update_frequency_by_binary(binary: str, x: float, y: float, z: float) -> bool:
    """Update position using binary signature directly."""
    if binary not in _store:
        return False
    _store[binary]["x"] = float(x)
    _store[binary]["y"] = float(y)
    _store[binary]["z"] = float(z)
    return True


# ---------------------------------------------------------------------------
# Bulk operations & State Management
# ---------------------------------------------------------------------------

def get_store() -> dict[str, dict]:
    """Return reference to internal _store dictionary."""
    return _store


def clear_store() -> None:
    """Clear all entries in the Guard Layer."""
    _store.clear()


def set_store(new_store: dict[str, dict]) -> None:
    """Replace internal _store dictionary with provided dictionary."""
    global _store
    _store = new_store


def get_all_words() -> list[str]:
    """Return list of all decoded words currently registered."""
    return [_from_binary(b) for b in _store.keys()]


def load_all(graph_nodes: list) -> int:
    """Bulk-load nodes into the Guard Layer."""
    total = len(graph_nodes)
    loaded = 0
    skipped = 0

    for node in graph_nodes:
        name = node.get("name", "")
        if not name:
            skipped += 1
            continue

        binary_key = _to_binary(name)
        if binary_key in _store:
            skipped += 1
            continue

        _store[binary_key] = {
            "x": float(node["x"]),
            "y": float(node["y"]),
            "z": float(node["z"]),
        }
        loaded += 1

    return len(_store)


def find_nearest_branch(x: float, y: float, z: float) -> dict | None:
    """Find the Guard Layer entry whose (x, y, z) is closest to given position."""
    if not _store:
        return None

    best_key: str = ""
    best_dist: float = math.inf

    for binary_key, entry in _store.items():
        dx = entry["x"] - x
        dy = entry["y"] - y
        dz = entry["z"] - z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < best_dist:
            best_dist = dist
            best_key = binary_key

    entry = _store[best_key]
    return {
        "name":     _from_binary(best_key),
        "binary":   best_key,
        "x":        entry["x"],
        "y":        entry["y"],
        "z":        entry["z"],
        "distance": round(best_dist, 8),
    }


def store_size() -> int:
    """Return total number of entries in the Guard Layer."""
    return len(_store)


def guard_stats() -> dict:
    """Summary of Guard Layer state."""
    if not _store:
        return {"total_words": 0, "max_distance": 0.0, "min_distance": 0.0}

    distances = [
        math.sqrt(e["x"] ** 2 + e["y"] ** 2 + e["z"] ** 2)
        for e in _store.values()
    ]
    return {
        "total_words": len(_store),
        "max_distance": round(max(distances), 6),
        "min_distance": round(min(distances), 6),
    }
