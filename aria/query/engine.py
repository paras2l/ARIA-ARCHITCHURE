"""
ARIA — Query Engine (Autoregressive Chain, Polarity & Working Memory)
=====================================================================
The reasoning and retrieval brain of ARIA.

Key Features:
1. Session Working Memory: Retains multi-turn conversation context
2. Autoregressive Concept Chain Generation: Follows forward directed transitions
3. Polarity Filtering: Separates affirmative (+1) vs. inhibitory/negated (-1) relations
4. Additive multi-token attention blending
5. Emergent geometric midpoint reasoning for sparse subgraphs
"""

import math
from collections import deque

from aria.attention.head import multi_attend, get_traversal_start, multi_head_attend
from aria.guard.layer   import _from_binary, get_frequency_by_binary, exists, get_binary
from aria.resonance.block import _clean_tokens, _build_guard_keys

# ---------------------------------------------------------------------------
# Session Working Memory (Short-Term Memory Buffer)
# ---------------------------------------------------------------------------

_session_memory: deque = deque(maxlen=20)  # Stores recent active binary signatures


def clear_session_memory() -> None:
    """Clear conversational short-term working memory."""
    _session_memory.clear()


def get_session_memory() -> list[str]:
    """Return decoded words currently active in working memory."""
    return [_from_binary(b) for b in _session_memory]


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _euclidean(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)


def _generate_concept_chain(
    start_binary: str,
    connections: dict,
    max_steps: int = 5,
    fallback_candidates: list[dict] | None = None,
) -> list[dict]:
    """
    Autoregressive path rollout: follow highest-confidence forward transitions
    to construct a sequential thought / concept chain.
    If a dead-end is reached, seamlessly falls back to the next best unvisited
    candidate anchor from the attention head.
    """
    chain = []
    current_b = start_binary
    visited = {start_binary}
    
    fallback_pool = [c["binary"] for c in (fallback_candidates or []) if c.get("binary") != start_binary]

    while len(chain) < max_steps:
        edges = connections.get(current_b, [])
        # Pick best unvisited affirmative forward edge
        best_edge = None
        for edge in sorted(edges, key=lambda e: (e.get("weight", 0.0), e.get("count", 0)), reverse=True):
            if edge["to"] not in visited and edge.get("polarity", 1) == 1:
                best_edge = edge
                break

        if best_edge:
            tgt_b = best_edge["to"]
            visited.add(tgt_b)
            pos = get_frequency_by_binary(tgt_b)
            chain.append({
                "word":     _from_binary(tgt_b),
                "binary":   tgt_b,
                "weight":   best_edge.get("weight", 0.0),
                "count":    best_edge.get("count", 1),
                "polarity": best_edge.get("polarity", 1),
                "x":        pos["x"] if pos else 0.0,
                "y":        pos["y"] if pos else 0.0,
                "z":        pos["z"] if pos else 0.0,
            })
            current_b = tgt_b
        else:
            # Fallback to next unvisited anchor candidate
            next_anchor = None
            while fallback_pool:
                cand_b = fallback_pool.pop(0)
                if cand_b not in visited:
                    next_anchor = cand_b
                    break

            if not next_anchor:
                break

            visited.add(next_anchor)
            pos = get_frequency_by_binary(next_anchor)
            chain.append({
                "word":     _from_binary(next_anchor),
                "binary":   next_anchor,
                "weight":   0.5,
                "count":    1,
                "polarity": 1,
                "x":        pos["x"] if pos else 0.0,
                "y":        pos["y"] if pos else 0.0,
                "z":        pos["z"] if pos else 0.0,
            })
            current_b = next_anchor

    return chain


# ---------------------------------------------------------------------------
# Directed BFS Traversal (Polarity-Aware)
# ---------------------------------------------------------------------------

def _bfs(
    start_binary: str,
    connections: dict,
    max_depth: int = 3,
    top_n: int = 10,
    target_polarity: int = 1,
) -> list[dict]:
    """
    Breadth-first search traversing directed resonance edges.
    Respects target_polarity (+1 for normal affirmative facts, -1 for negated facts).
    """
    visited: set[str] = {start_binary}
    weight_map: dict[str, float] = {}
    pos_map: dict[str, dict] = {}
    pol_map: dict[str, int] = {}
    queue: deque = deque([(start_binary, 0)])

    while queue:
        current_b, depth = queue.popleft()
        if depth >= max_depth:
            continue

        raw_edges = connections.get(current_b, [])
        # Sort by weight descending
        top_edges = sorted(raw_edges, key=lambda e: e.get("weight", 0.0), reverse=True)[:top_n * 2]

        for edge in top_edges:
            tgt_b = edge["to"]
            w = edge.get("weight", 0.0)
            pol = edge.get("polarity", 1)

            # Filter by polarity intent
            if pol != target_polarity and target_polarity != 0:
                continue

            # Apply working memory boost if active in recent conversation
            if tgt_b in _session_memory:
                w = min(w + 0.15, 1.0)

            if tgt_b in visited:
                if w > weight_map.get(tgt_b, 0.0):
                    weight_map[tgt_b] = w
                    pol_map[tgt_b] = pol
                continue

            visited.add(tgt_b)
            weight_map[tgt_b] = w
            pol_map[tgt_b] = pol
            pos = get_frequency_by_binary(tgt_b)
            pos_map[tgt_b] = pos or {"x": 0.0, "y": 0.0, "z": 0.0}
            queue.append((tgt_b, depth + 1))

    results = []
    for b, w in weight_map.items():
        if b == start_binary:
            continue
        pos = pos_map.get(b, {"x": 0.0, "y": 0.0, "z": 0.0})
        results.append({
            "word":     _from_binary(b),
            "binary":   b,
            "weight":   round(w, 6),
            "polarity": pol_map.get(b, 1),
            "x":        pos["x"],
            "y":        pos["y"],
            "z":        pos["z"],
        })

    results.sort(key=lambda n: n["weight"], reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Emergent Midpoint Reasoning
# ---------------------------------------------------------------------------

def _emergent_reasoning(
    nodes: list[dict],
    connections: dict,
) -> list[dict]:
    """Find hidden bridge concept between top 2 result nodes in 3D coordinate space."""
    if len(nodes) < 2:
        return []

    a, b = nodes[0], nodes[1]
    mx = (a["x"] + b["x"]) / 2.0
    my = (a["y"] + b["y"]) / 2.0
    mz = (a["z"] + b["z"]) / 2.0

    best_binary = ""
    best_dist = math.inf
    already_in = {n["binary"] for n in nodes}

    for binary in connections:
        if binary in already_in:
            continue
        pos = get_frequency_by_binary(binary)
        if not pos:
            continue
        dist = _euclidean(mx, my, mz, pos["x"], pos["y"], pos["z"])
        if dist < best_dist:
            best_dist = dist
            best_binary = binary

    if not best_binary:
        return []

    pos = get_frequency_by_binary(best_binary)
    avg_weight = round(((a["weight"] + b["weight"]) / 2.0) * 0.85, 6)

    return [{
        "word":     _from_binary(best_binary),
        "binary":   best_binary,
        "weight":   avg_weight,
        "polarity": 1,
        "x":        pos["x"] if pos else 0.0,
        "y":        pos["y"] if pos else 0.0,
        "z":        pos["z"] if pos else 0.0,
        "note":     "emergent midpoint inference",
    }]


# ---------------------------------------------------------------------------
# Public Query Entry Point
# ---------------------------------------------------------------------------

def query(
    text: str,
    connections: dict,
    top_n: int = 10,
    max_depth: int = 3,
) -> dict:
    """
    Execute a sequence-aware query with polarity filtering and concept chain generation.
    """
    print(f"\n[ARIA:Query] ─────────────────────────────────────────")
    print(f"[ARIA:Query] Query: '{text}'")

    # Step 1: Detect query polarity intent (affirmative vs negative query)
    lower_query = text.lower()
    is_neg_query = any(neg in lower_query.split() for neg in ["not", "never", "false", "nahi", "wrong", "untrue", "without"])
    target_polarity = -1 if is_neg_query else 1

    # Step 2: Tokenize using Guard Layer longest-match
    guard_keys = _build_guard_keys()
    words = _clean_tokens(text, guard_keys)
    if not words:
        words = [w.strip() for w in text.split() if w.strip()]

    print(f"[ARIA:Query] Guard matched: {words} (Polarity intent: {'[-] Negation' if is_neg_query else '[+] Affirmation'})")
    if not words:
        return {
            "status": "need_more_data",
            "query": text,
            "words": [],
            "results": [],
            "concept_chain": [],
            "depth_reached": 0,
            "emergent": False,
        }

    # Record matched words into Session Working Memory
    for w in words:
        b = get_binary(w)
        if b and b not in _session_memory:
            _session_memory.append(b)

    # Step 3: Multi-Head Attention Ensemble
    mha = multi_head_attend(words, connections, working_memory=list(_session_memory), top_n=top_n)
    combined = mha.get("ensemble", [])
    active_heads = list({h for c in combined for h in c.get("heads_active", [])})

    # Step 4: Pick Best Traversal Start from Multi-Head Ensemble
    start_node = combined[0] if combined else None
    if not start_node:
        for word in words:
            candidate = get_traversal_start(word, connections)
            if candidate and (start_node is None or candidate["weight"] > start_node["weight"]):
                start_node = candidate

    # Step 5: BFS Traversal & Concept Chain Generation
    if start_node:
        bfs_nodes = _bfs(start_node["binary"], connections, max_depth=max_depth, top_n=top_n, target_polarity=target_polarity)
        concept_chain = _generate_concept_chain(
            start_binary=start_node["binary"],
            connections=connections,
            max_steps=5,
            fallback_candidates=combined,
        )
    else:
        bfs_nodes = []
        concept_chain = []

    # Step 6: Merge and rank results
    seen: set[str] = set()
    merged: list[dict] = []

    for node in bfs_nodes + [c for c in combined if c.get("polarity", 1) == target_polarity]:
        b = node["binary"]
        if b not in seen:
            seen.add(b)
            # Add to working memory
            if b not in _session_memory:
                _session_memory.append(b)
            merged.append(node)

    merged.sort(key=lambda n: n.get("weight", 0.0), reverse=True)
    results = merged[:top_n]

    emergent_fired = False
    if len(results) < 3 and connections and not is_neg_query:
        emergent = _emergent_reasoning(results, connections)
        if emergent:
            results.extend(emergent)
            emergent_fired = True

    status = "ok" if results else "need_more_data"
    print(f"[ARIA:Query] ✅ Status: {status} ({len(results)} nodes retrieved, chain length={len(concept_chain)})")
    if concept_chain:
        chain_words = " -> ".join([f"{c['word']}" for c in concept_chain])
        print(f"[ARIA:Query] 🔗 Concept Chain: {chain_words}")
    print(f"[ARIA:Query] ─────────────────────────────────────────\n")

    return {
        "status":        status,
        "query":         text,
        "words":         words,
        "polarity":      target_polarity,
        "results":       results,
        "concept_chain": concept_chain,
        "depth_reached": max_depth,
        "emergent":      emergent_fired,
    }
