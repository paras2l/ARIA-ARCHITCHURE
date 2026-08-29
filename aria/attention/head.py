"""
ARIA — Multi-Head Attention Engine
===================================
4-Head Parallel Attention for sequence, spatial, context, and bridge reasoning.

Heads:
1. Head 1 (Sequence Head) : Causal forward transitions (A → B) with transition counts
2. Head 2 (Spatial Head)  : 3D geometric vector alignment and proximity (dx, dy, dz)
3. Head 3 (Context Head)  : Working Memory alignment (prioritizes active conversation topics)
4. Head 4 (Bridge Head)   : Cross-lingual / high-weight equivalence bridges (weight >= 0.8)

Ensemble:
- Fuses multi-head projections with consensus confidence scoring
- Hub Node Damping (IDF scaling) prevents generic high-degree words from dominating
"""

import math
from aria.guard.layer import get_binary, get_frequency_by_binary, exists, _from_binary

# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _enrich_connection(conn: dict, head_tag: str = "general") -> dict:
    """Add target position, decoded name, and head tag to a raw connection dict."""
    target_binary = conn["to"]
    pos = get_frequency_by_binary(target_binary)
    word = _from_binary(target_binary)

    return {
        "word":     word,
        "binary":   target_binary,
        "weight":   conn.get("weight", 0.0),
        "count":    conn.get("count", 1),
        "dx":       conn.get("dx", 0.0),
        "dy":       conn.get("dy", 0.0),
        "dz":       conn.get("dz", 0.0),
        "polarity": conn.get("polarity", 1),
        "head":     head_tag,
        "x":        pos["x"] if pos else 0.0,
        "y":        pos["y"] if pos else 0.0,
        "z":        pos["z"] if pos else 0.0,
    }


def _calculate_node_degree(binary: str, connections: dict) -> int:
    """Get outgoing degree of a node as a fast proxy for general word frequency."""
    return len(connections.get(binary, []))


def _hub_idf(binary: str, connections: dict) -> float:
    """Smooth IDF multiplier: lowers influence of massive hub nodes."""
    deg = _calculate_node_degree(binary, connections)
    return 1.0 / math.log2(2 + deg * 0.1) if deg > 10 else 1.0


# ---------------------------------------------------------------------------
# Individual Attention Heads
# ---------------------------------------------------------------------------

def head_sequence(query_word: str, connections: dict, top_n: int = 10) -> list[dict]:
    """Head 1: Causal sequence flow and transition count."""
    if not exists(query_word):
        return []
    q_bin = get_binary(query_word)
    raw = connections.get(q_bin, [])
    if not raw:
        return []

    def _seq_score(c: dict) -> float:
        w = c.get("weight", 0.0)
        cnt = c.get("count", 1)
        pol = c.get("polarity", 1)
        idf = _hub_idf(c["to"], connections)
        return (w * (1.0 + cnt * 0.05) * pol) * idf

    sorted_conns = sorted(raw, key=_seq_score, reverse=True)[:top_n]
    return [_enrich_connection(c, head_tag="sequence") for c in sorted_conns]


def head_spatial(query_word: str, connections: dict, top_n: int = 10) -> list[dict]:
    """Head 2: 3D geometric vector alignment and proximity."""
    if not exists(query_word):
        return []
    q_bin = get_binary(query_word)
    raw = connections.get(q_bin, [])
    if not raw:
        return []

    def _spatial_score(c: dict) -> float:
        dx, dy, dz = c.get("dx", 0.0), c.get("dy", 0.0), c.get("dz", 0.0)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        proximity = 1.0 / (1.0 + dist)
        w = c.get("weight", 0.0)
        idf = _hub_idf(c["to"], connections)
        return (w * 0.5 + proximity * 0.5) * idf

    sorted_conns = sorted(raw, key=_spatial_score, reverse=True)[:top_n]
    return [_enrich_connection(c, head_tag="spatial") for c in sorted_conns]


def head_context(
    query_word: str,
    connections: dict,
    working_memory: list[str] | None = None,
    top_n: int = 10,
) -> list[dict]:
    """Head 3: Working Memory Context alignment (short-term conversation history)."""
    if not exists(query_word):
        return []
    q_bin = get_binary(query_word)
    raw = connections.get(q_bin, [])
    if not raw:
        return []

    mem_set = set(working_memory or [])

    def _context_score(c: dict) -> float:
        tgt_b = c["to"]
        w = c.get("weight", 0.0)
        # Big boost if target intersects conversation working memory
        boost = 1.5 if (tgt_b in mem_set or _from_binary(tgt_b) in mem_set) else 1.0
        idf = _hub_idf(tgt_b, connections)
        return (w * boost) * idf

    sorted_conns = sorted(raw, key=_context_score, reverse=True)[:top_n]
    return [_enrich_connection(c, head_tag="context") for c in sorted_conns]


def head_bridge(query_word: str, connections: dict, top_n: int = 10) -> list[dict]:
    """Head 4: Cross-lingual / high-weight equivalence bridges."""
    if not exists(query_word):
        return []
    q_bin = get_binary(query_word)
    raw = connections.get(q_bin, [])
    if not raw:
        return []

    # Filter for high-weight or bridge connections (weight >= 0.75)
    bridge_conns = [c for c in raw if c.get("weight", 0.0) >= 0.75]
    sorted_conns = sorted(bridge_conns, key=lambda c: c.get("weight", 0.0), reverse=True)[:top_n]
    return [_enrich_connection(c, head_tag="bridge") for c in sorted_conns]


# ---------------------------------------------------------------------------
# Master Multi-Head Attention API
# ---------------------------------------------------------------------------

def multi_head_attend(
    query_words: list[str],
    connections: dict,
    working_memory: list[str] | None = None,
    top_n: int = 10,
) -> dict:
    """
    Run 4 parallel attention heads across query tokens and compute consensus ensemble.
    """
    heads_results: dict[str, list] = {
        "sequence": [],
        "spatial":  [],
        "context":  [],
        "bridge":   [],
    }

    combined_scores: dict[str, dict] = {}

    for word in query_words:
        h1 = head_sequence(word, connections, top_n=top_n)
        h2 = head_spatial(word, connections, top_n=top_n)
        h3 = head_context(word, connections, working_memory=working_memory, top_n=top_n)
        h4 = head_bridge(word, connections, top_n=top_n)

        heads_results["sequence"].extend(h1)
        heads_results["spatial"].extend(h2)
        heads_results["context"].extend(h3)
        heads_results["bridge"].extend(h4)

        # Weighted Ensemble: 40% sequence, 25% spatial, 20% context, 15% bridge
        for node in h1:
            b = node["binary"]
            combined_scores.setdefault(b, {"node": node, "score": 0.0, "heads": set()})
            combined_scores[b]["score"] += node["weight"] * 0.40
            combined_scores[b]["heads"].add("sequence")

        for node in h2:
            b = node["binary"]
            combined_scores.setdefault(b, {"node": node, "score": 0.0, "heads": set()})
            combined_scores[b]["score"] += node["weight"] * 0.25
            combined_scores[b]["heads"].add("spatial")

        for node in h3:
            b = node["binary"]
            combined_scores.setdefault(b, {"node": node, "score": 0.0, "heads": set()})
            combined_scores[b]["score"] += node["weight"] * 0.20
            combined_scores[b]["heads"].add("context")

        for node in h4:
            b = node["binary"]
            combined_scores.setdefault(b, {"node": node, "score": 0.0, "heads": set()})
            combined_scores[b]["score"] += node["weight"] * 0.15
            combined_scores[b]["heads"].add("bridge")

    # Sort ensemble consensus
    ensemble = []
    for b, item in sorted(combined_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n]:
        res_node = dict(item["node"])
        res_node["ensemble_score"] = round(item["score"], 6)
        res_node["heads_active"] = list(item["heads"])
        ensemble.append(res_node)

    return {
        "heads": heads_results,
        "ensemble": ensemble,
        "top_candidates": ensemble,
    }


# ---------------------------------------------------------------------------
# Backward-Compatible Public API
# ---------------------------------------------------------------------------

def attend(
    query_word: str,
    connections: dict,
    top_n: int = 10,
    damp_hubs: bool = True,
) -> list[dict]:
    """Single-word attention using Sequence Head (with hub damping)."""
    return head_sequence(query_word, connections, top_n=top_n)


def get_traversal_start(
    query_word: str,
    connections: dict,
) -> dict | None:
    """Return the single highest-confidence anchor connection."""
    top = head_sequence(query_word, connections, top_n=1)
    return top[0] if top else None


def multi_attend(
    query_words: list[str],
    connections: dict,
    top_n: int = 10,
) -> dict:
    """Multi-word attention using Multi-Head Ensemble."""
    mha = multi_head_attend(query_words, connections, top_n=top_n)
    return {
        "combined": mha["ensemble"],
        "heads": mha["heads"],
    }
