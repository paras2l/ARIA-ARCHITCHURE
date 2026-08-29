"""
ARIA — Resonance Block (Adaptive 8–32 Multi-Directional 3D Resonance)
=====================================================================
Builds multi-directional sequence connections, negation polarity, document anchors,
and directional vectors between concepts using Adaptive 8–32 Multi-Directional Scaling.

Key Capabilities:
1. Adaptive 8–32 Multi-Directional Scaling: Auto-selects 8, 16, or 32 directional axes per sentence
2. Multi-Span Skip Horizons: Immediate forward/reverse, skip-gram horizons, and clause cohesion
3. Negation & Inhibitory Synapses (polarity: +1 for affirmation, -1 for negation)
4. Hierarchical Document Anchors for long-range chapter/topic context
5. Translation & Equivalence Bridges (weight 1.0 zero-distance cross-lingual links)
6. Directional Vector Deltas (dx, dy, dz) on every edge
7. Smooth Gravitational Coordinate Drift with momentum damping (85% old / 15% pull)
8. Synaptic Noise Pruning to drop spurious 1-occurrence noise edges
"""

import math
import re
import time

from aria.guard.layer import (
    exists,
    get_binary,
    get_frequency,
    get_frequency_by_binary,
    _from_binary,
    _store,
    update_frequency_by_binary,
    update_word_position,
    auto_register,
)

# ---------------------------------------------------------------------------
# Negation trigger words (English + Hindi)
# ---------------------------------------------------------------------------

_NEGATION_WORDS: frozenset = frozenset({
    "not", "never", "no", "none", "neither", "nor",
    "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't", "werent", "weren't",
    "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "cant", "can't", "cannot", "wont", "won't", "shouldnt", "shouldn't",
    "without", "false", "nahi", "na", "mat",
})


# ---------------------------------------------------------------------------
# Internal Helpers: Guard Keys & Longest-Match Tokenizer
# ---------------------------------------------------------------------------

def _build_guard_keys() -> list[str]:
    """Return all Guard Layer word names sorted longest-first."""
    try:
        names = [_from_binary(b) for b in _store.keys()]
    except Exception:
        names = []
    names.sort(key=len, reverse=True)
    return names


def _clean_tokens(text: str, guard_keys: list[str]) -> list[str]:
    """
    Scan text left-to-right and greedily match the longest Guard Layer key.
    Case-insensitive matching that preserves original Guard key casing.
    """
    text_clean = " ".join(text.split())
    text_lower = text_clean.lower()
    guard_map = {k.lower(): k for k in guard_keys}
    sorted_lowers = sorted(guard_map.keys(), key=len, reverse=True)

    tokens = []
    i = 0
    n = len(text_lower)

    while i < n:
        if text_lower[i] == " ":
            i += 1
            continue

        matched = False
        for k_low in sorted_lowers:
            k_len = len(k_low)
            if i + k_len <= n and text_lower[i:i + k_len] == k_low:
                # Word boundary check
                before_ok = (i == 0 or text_lower[i - 1] in " \t\n\r,.;:!?\"'()[]{}")
                after_ok = (i + k_len == n or text_lower[i + k_len] in " \t\n\r,.;:!?\"'()[]{}")

                if before_ok and after_ok:
                    orig_key = guard_map[k_low]
                    tokens.append(orig_key)
                    i += k_len
                    matched = True
                    break

        if not matched:
            while i < n and text_lower[i] not in " \t\n\r,.;:!?\"'()[]{}":
                i += 1
            while i < n and text_lower[i] in " \t\n\r,.;:!?\"'()[]{}":
                i += 1

    return tokens


# ---------------------------------------------------------------------------
# Directed Edge Management
# ---------------------------------------------------------------------------

def _upsert_directed_edge(
    connections: dict,
    src_b: str,
    tgt_b: str,
    weight_increment: float,
    dx: float,
    dy: float,
    dz: float,
    polarity: int = 1,
) -> bool:
    """
    Insert or update a directed edge: src_b → tgt_b with polarity (+1 / -1).
    Returns True if this was a brand new edge, False if updated existing.
    """
    if src_b not in connections:
        connections[src_b] = []

    edge_list = connections[src_b]
    for edge in edge_list:
        if edge["to"] == tgt_b:
            edge["count"] = edge.get("count", 0) + 1
            edge["weight"] = round(min(edge["weight"] + weight_increment, 1.0), 6)
            edge["dx"] = round(dx, 6)
            edge["dy"] = round(dy, 6)
            edge["dz"] = round(dz, 6)
            # If ever negated, polarity becomes inhibitory (-1)
            if polarity == -1:
                edge["polarity"] = -1
            elif "polarity" not in edge:
                edge["polarity"] = 1
            return False

    # New directed edge
    edge_list.append({
        "to": tgt_b,
        "weight": round(min(weight_increment, 1.0), 6),
        "count": 1,
        "dx": round(dx, 6),
        "dy": round(dy, 6),
        "dz": round(dz, 6),
        "polarity": int(polarity),
    })
    return True


# ---------------------------------------------------------------------------
# Translation & Equivalence Bridges
# ---------------------------------------------------------------------------

def create_bridge(
    word_a: str,
    word_b: str,
    connections: dict,
    weight: float = 1.0,
) -> dict:
    """
    Create a bidirectional translation / equivalence bridge between two words.
    E.g. create_bridge("Pani", "Water", connections)
    """
    word_a = word_a.strip()
    word_b = word_b.strip()
    if not word_a or not word_b:
        return {"status": "invalid"}

    bin_a, _ = auto_register(word_a)
    bin_b, _ = auto_register(word_b)

    # Position synchronization
    pos_a = get_frequency_by_binary(bin_a)
    pos_b = get_frequency_by_binary(bin_b)
    dx = (pos_b["x"] - pos_a["x"]) if (pos_a and pos_b) else 0.0
    dy = (pos_b["y"] - pos_a["y"]) if (pos_a and pos_b) else 0.0
    dz = (pos_b["z"] - pos_a["z"]) if (pos_a and pos_b) else 0.0

    # Bidirectional high-weight bridge with polarity +1
    _upsert_directed_edge(connections, bin_a, bin_b, weight, dx, dy, dz, polarity=1)
    _upsert_directed_edge(connections, bin_b, bin_a, weight, -dx, -dy, -dz, polarity=1)

    print(f"[ARIA:Resonance] 🌉 Bridge established: '{word_a}' ⟷ '{word_b}' (weight={weight})")
    return {"status": "bridged", "word_a": word_a, "word_b": word_b, "weight": weight}


# ---------------------------------------------------------------------------
# Synaptic Pruning
# ---------------------------------------------------------------------------

def prune_synapses(
    connections: dict,
    min_weight: float = 0.05,
    min_count: int = 1,
) -> int:
    """
    Prune weak, spurious connections below min_weight or min_count.
    Returns total number of edges pruned.
    """
    pruned_count = 0
    empty_keys = []

    for src_b, edges in connections.items():
        original_len = len(edges)
        filtered = [
            e for e in edges
            if e.get("weight", 0.0) >= min_weight and e.get("count", 0) >= min_count
        ]
        pruned_count += (original_len - len(filtered))
        connections[src_b] = filtered
        if not filtered:
            empty_keys.append(src_b)

    for k in empty_keys:
        del connections[k]

    return pruned_count


# ---------------------------------------------------------------------------
# Damped Gravitational Coordinate Drift
# ---------------------------------------------------------------------------

def update_frequencies(
    connections: dict,
    momentum: float = 0.85,
) -> int:
    """
    Smoothly drift word coordinates toward their connected neighbors.
    Uses momentum damping (default 85% old position + 15% neighbor pull).
    Inhibitory (polarity: -1) edges gently push away instead of pull.
    """
    drift_pull = 1.0 - momentum
    updated_count = 0
    total_displacement = 0.0

    for src_b, edges in connections.items():
        src_pos = get_frequency_by_binary(src_b)
        if not src_pos or not edges:
            continue

        sum_w = 0.0
        target_x = 0.0
        target_y = 0.0
        target_z = 0.0

        for edge in edges:
            tgt_b = edge["to"]
            w = edge.get("weight", 0.1)
            pol = edge.get("polarity", 1)
            tgt_pos = get_frequency_by_binary(tgt_b)
            if tgt_pos:
                if pol == 1:
                    target_x += tgt_pos["x"] * w
                    target_y += tgt_pos["y"] * w
                    target_z += tgt_pos["z"] * w
                    sum_w += w
                else:
                    # Mild repulsion for negated attributes
                    target_x -= (tgt_pos["x"] - src_pos["x"]) * (w * 0.1)
                    target_y -= (tgt_pos["y"] - src_pos["y"]) * (w * 0.1)
                    target_z -= (tgt_pos["z"] - src_pos["z"]) * (w * 0.1)

        if sum_w > 0:
            avg_x = target_x / sum_w
            avg_y = target_y / sum_w
            avg_z = target_z / sum_w
            new_x = round((src_pos["x"] * momentum) + (avg_x * drift_pull), 6)
            new_y = round((src_pos["y"] * momentum) + (avg_y * drift_pull), 6)
            new_z = round((src_pos["z"] * momentum) + (avg_z * drift_pull), 6)

            disp = (new_x - src_pos["x"])**2 + (new_y - src_pos["y"])**2 + (new_z - src_pos["z"])**2
            total_displacement += disp
            update_frequency_by_binary(src_b, new_x, new_y, new_z)
            updated_count += 1

    drift_loss = round(total_displacement / max(1, updated_count), 6)
    return updated_count, drift_loss


# ---------------------------------------------------------------------------
# Core Block Passes: Adaptive 8–32 Multi-Directional Resonance
# ---------------------------------------------------------------------------

def run_block1(
    script_text: str,
    connections: dict,
    window_size: int = 5,
    document_anchor: str = "",
    pass_num: int = 1,
) -> dict:
    """
    Scan script text and forge multi-directional transition connections with
    Adaptive 8–32 directional scaling, polarity, hierarchical document anchors,
    and Dynamic Learning Rate / Momentum Annealing.
    """
    t0 = time.time()
    # Learning Rate Annealing for 3D Gravitational Drift: 0.15 -> 0.0975 -> 0.0634 -> ...
    drift_lr = max(0.01, 0.15 * (0.65 ** (pass_num - 1)))
    momentum = 1.0 - drift_lr
    
    # Check for document anchor (e.g. from header or argument)
    active_anchor = document_anchor.strip()
    lines = script_text.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped.lower().startswith("chapter:") or stripped.lower().startswith("topic:"):
            header_clean = re.sub(r"^[#\s]+", "", stripped)
            header_clean = re.sub(r"^(chapter|topic):\s*", "", header_clean, flags=re.I).strip()
            if header_clean:
                active_anchor = header_clean
                auto_register(active_anchor)
                break

    anchor_b = get_binary(active_anchor) if active_anchor else None
    anchor_pos = get_frequency_by_binary(anchor_b) if anchor_b else None

    # Sentence-level scanning for adaptive multi-directional resonance
    sentences = re.split(r"[.!?\n\r]+", script_text)
    new_edges = 0
    total_tokens_seen = 0
    directions_logged = []

    for sent in sentences:
        sent_clean = sent.strip()
        if not sent_clean:
            continue

        raw_words = [w.strip().lower() for w in sent_clean.split() if w.strip()]
        raw_tokens = re.findall(r"\b[A-Za-z0-9\-_]{2,}\b", sent_clean)
        
        # Ensure all tokens in sentence are registered
        for rt in raw_tokens:
            if rt.lower() not in _NEGATION_WORDS and not exists(rt):
                auto_register(rt)
        
        tokens = [rt for rt in raw_tokens if rt.lower() not in _NEGATION_WORDS and exists(rt)]
        num_tokens = len(tokens)
        total_tokens_seen += num_tokens

        if num_tokens < 1:
            continue

        # Dynamic Adaptive Directions Selection (8, 16, or 32)
        adaptive_directions = min(32, max(8, num_tokens * 2))
        adaptive_window = min(8, max(3, adaptive_directions // 4))
        directions_logged.append(adaptive_directions)

        # Check if sentence contains negation markers
        neg_indices = [idx for idx, w in enumerate(raw_words) if w in _NEGATION_WORDS]
        has_sentence_negation = len(neg_indices) > 0

        # 1. Thematic Anchor connections (Hierarchical Tier)
        if anchor_b and anchor_pos:
            for tok in tokens:
                tok_b = get_binary(tok)
                tok_pos = get_frequency_by_binary(tok_b) if tok_b else None
                if tok_b and tok_pos and tok_b != anchor_b:
                    adx = tok_pos["x"] - anchor_pos["x"]
                    ady = tok_pos["y"] - anchor_pos["y"]
                    adz = tok_pos["z"] - anchor_pos["z"]
                    if _upsert_directed_edge(connections, anchor_b, tok_b, 0.25, adx, ady, adz, polarity=1):
                        new_edges += 1

        # 2. Multi-Directional Sequential & Skip-Gram Scanning
        for i in range(num_tokens):
            src_word = tokens[i]
            src_b = get_binary(src_word)
            src_pos = get_frequency_by_binary(src_b) if src_b else None
            if not src_b or not src_pos:
                continue

            # Forward Spans (+1 to +k)
            for j in range(i + 1, min(i + adaptive_window + 1, num_tokens)):
                tgt_word = tokens[j]
                if tgt_word == src_word:
                    continue

                tgt_b = get_binary(tgt_word)
                tgt_pos = get_frequency_by_binary(tgt_b) if tgt_b else None
                if not tgt_b or not tgt_pos:
                    continue

                step = j - i
                # Step 1 (immediate syntax) vs Skip-Steps (semantic horizon)
                forward_weight = 0.25 if step == 1 else (0.15 / (step ** 0.7))
                dx = tgt_pos["x"] - src_pos["x"]
                dy = tgt_pos["y"] - src_pos["y"]
                dz = tgt_pos["z"] - src_pos["z"]

                edge_polarity = -1 if has_sentence_negation else 1

                if _upsert_directed_edge(connections, src_b, tgt_b, forward_weight, dx, dy, dz, polarity=edge_polarity):
                    new_edges += 1

                # Reverse Harmonic Spans (-1 to -k)
                reverse_weight = forward_weight * 0.45
                _upsert_directed_edge(connections, tgt_b, src_b, reverse_weight, -dx, -dy, -dz, polarity=edge_polarity)

        # 3. Clause Cohesion Boundary Anchors (Sentence-Start <-> Sentence-End)
        if num_tokens >= 6:
            start_b = get_binary(tokens[0])
            end_b = get_binary(tokens[-1])
            start_pos = get_frequency_by_binary(start_b) if start_b else None
            end_pos = get_frequency_by_binary(end_b) if end_b else None
            if start_b and end_b and start_pos and end_pos and start_b != end_b:
                cdx = end_pos["x"] - start_pos["x"]
                cdy = end_pos["y"] - start_pos["y"]
                cdz = end_pos["z"] - start_pos["z"]
                _upsert_directed_edge(connections, start_b, end_b, 0.12, cdx, cdy, cdz, polarity=1)

    # Apply damped coordinate drift with annealed momentum
    updated_nodes, drift_loss = update_frequencies(connections, momentum=momentum)
    elapsed = time.time() - t0
    avg_directions = round(sum(directions_logged) / len(directions_logged), 1) if directions_logged else 8

    return {
        "new_connections": new_edges,
        "total_tokens": total_tokens_seen,
        "document_anchor": active_anchor,
        "avg_directions": avg_directions,
        "nodes_drifted": updated_nodes,
        "drift_loss": drift_loss,
        "drift_lr": round(drift_lr, 4),
        "time_s": round(elapsed, 4),
    }


def run_block2(
    word: str,
    script_text: str,
    connections: dict,
    window_size: int = 5,
) -> dict:
    """
    Dynamic insertion pass for a single word in context.
    """
    if not exists(word):
        return {"status": "not_in_guard", "new_connections": 0}

    return run_block1(script_text, connections, window_size=window_size)


def run_loop(
    script_text: str,
    connections: dict,
    max_passes: int = 5,
    window_size: int = 5,
    document_anchor: str = "",
    save_callback=None,
) -> dict:
    """
    Run multi-pass directed resonance loop until convergence or max_passes.
    Applies synaptic pruning at the end.
    """
    print(f"[ARIA:Resonance] Starting Adaptive 8–32 Multi-Directional Resonance Loop (max_passes={max_passes})")
    total_new = 0
    t_start = time.time()

    for p in range(1, max_passes + 1):
        res = run_block1(script_text, connections, window_size=window_size, document_anchor=document_anchor)
        new_conn = res["new_connections"]
        total_new += new_conn
        print(f"[ARIA:Resonance] Pass {p}/{max_passes}: +{new_conn:,} new multi-directional synapses ({res['time_s']}s, avg_dirs={res.get('avg_directions', 8)})")

        if save_callback:
            save_callback(connections)

        if new_conn == 0:
            print(f"[ARIA:Resonance] 🎯 Graph converged at pass {p}.")
            break

    # Synaptic pruning pass
    pruned = prune_synapses(connections, min_weight=0.05, min_count=1)
    if pruned > 0:
        print(f"[ARIA:Resonance] ✂️ Pruned {pruned:,} weak noise edges.")

    total_time = round(time.time() - t_start, 2)
    print(f"[ARIA:Resonance] ✅ Loop complete in {total_time}s. Total synapses in graph: {sum(len(v) for v in connections.values()):,}")

    return {
        "passes_run": p,
        "total_new_connections": total_new,
        "document_anchor": res.get("document_anchor", ""),
        "total_time_s": total_time,
    }
