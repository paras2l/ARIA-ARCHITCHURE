"""
ARIA — Core Runtime Engine
==========================
Master orchestrator connecting all ARIA components:
1. Guard Layer & Unknown Word Handler (Symbol store & Dynamic coordinates)
2. Graph Builder (In-memory nodes)
3. Resonance Block (Directed sequence synapses, deltas, damped drift, pruning)
4. Attention Head & Query Engine (Chain generation & working memory)
5. Stigma Manager (Compact binary persistence)
"""

import sys
import os
import traceback

from aria.graph.builder       import load_graph, get_all_nodes, add_node
from aria.guard.layer         import (
    load_all, get_binary, exists, get_store, set_store,
    _from_binary, _to_binary, auto_register, get_frequency_by_binary
)
from aria.resonance.block     import run_loop, run_block1, create_bridge
from aria.frequency.converter import handle_unknown_word, get_rough_position
from aria.router.insert       import route_new_word, route_batch
from aria.query.engine        import query as _engine_query, clear_session_memory, get_session_memory
from aria.stigma.manager      import (
    save as stigma_save,
    load as stigma_load,
    list_saves,
)

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_state: dict = {
    "initialized": False,
    "connections": {},
    "all_nodes":   [],
}


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _phase(name: str, emoji: str = "🔄") -> None:
    print(f"\n[ARIA:Core] {'═' * 48}")
    print(f"[ARIA:Core] {emoji}  {name}")
    print(f"[ARIA:Core] {'═' * 48}")


def _handle_error(phase: str, error: Exception) -> None:
    print(f"\n[ARIA:Core] {'!' * 48}")
    print(f"[ARIA:Core] ❌ ERROR in: {phase}")
    print(f"[ARIA:Core] Type   : {type(error).__name__}")
    print(f"[ARIA:Core] Reason : {str(error)}")
    print(f"[ARIA:Core] {'!' * 48}")
    traceback.print_exc()


def _count_connections(connections: dict) -> int:
    return sum(len(v) for v in connections.values())


def _check_initialized(fn_name: str) -> bool:
    if not _state["initialized"]:
        print(f"[ARIA:Core] ⚠️  {fn_name}() — ARIA not initialised, run aria_init() first")
        return False
    return True


def _rotate_and_save(connections: dict, nodes: list) -> None:
    # slot 2 → slot 3
    try:
        s2 = stigma_load(2)
        if s2 is not None:
            stigma_save(3, s2["connections"], s2["nodes"])
    except Exception:
        pass

    # slot 1 → slot 2
    try:
        s1 = stigma_load(1)
        if s1 is not None:
            stigma_save(2, s1["connections"], s1["nodes"])
    except Exception:
        pass

    # current → slot 1
    try:
        stigma_save(1, connections, nodes)
    except Exception as e:
        _handle_error("Stigma Save (slot 1)", e)


# ---------------------------------------------------------------------------
# 1. aria_init()
# ---------------------------------------------------------------------------

def aria_init(filepath: str = "", script_text: str = "") -> bool:
    """
    Start the ARIA runtime.
    Mode 1: No filepath → Restores from latest Stigma slot (1 -> 2 -> 3).
    Mode 2: Filepath given → Builds graph from raw text or JSON dataset.
    """
    global _state

    print(f"\n[ARIA:Core] ════════════════════════════════════════════════")
    print(f"[ARIA:Core] 🧠  ARIA Startup (Directed Neural Graph)")
    if filepath:
        print(f"[ARIA:Core]     Mode  : Full Build / Training")
        print(f"[ARIA:Core]     Source: {filepath}")
    else:
        print(f"[ARIA:Core]     Mode  : Stigma Slot Restore")
    print(f"[ARIA:Core] ════════════════════════════════════════════════")

    _state["initialized"] = False
    _state["connections"] = {}
    _state["all_nodes"]   = []

    # MODE 1 — Restore from Stigma
    if not filepath:
        _phase("Checking Stigma Saves", "🔍")
        for slot in (1, 2, 3):
            try:
                restored = stigma_load(slot)
                if restored is not None:
                    _state["connections"] = restored["connections"]
                    _state["all_nodes"]   = restored["nodes"]

                    # Hydrate Guard Layer
                    for n in _state["all_nodes"]:
                        auto_register(
                            word=_from_binary(n["binary"]),
                            x=n["x"], y=n["y"], z=n["z"]
                        )

                    _state["initialized"] = True
                    _phase("ARIA READY", "✅")
                    print(f"[ARIA:Core]  Restored from : Stigma slot {slot}")
                    print(f"[ARIA:Core]  Nodes         : {len(_state['all_nodes']):,}")
                    print(f"[ARIA:Core]  Synapses      : {_count_connections(_state['connections']):,}")
                    return True
            except Exception as e:
                _handle_error(f"Stigma Load slot {slot}", e)
                continue

        print("[ARIA:Core] ❌ No valid Stigma save found. Run with --build <file> first.")
        return False

    # MODE 2 — Build from raw text or dataset
    _phase("Graph Ingestion", "📂")
    try:
        load_graph(filepath)
        graph_nodes = get_all_nodes()
        load_all(graph_nodes)

        _state["all_nodes"] = [
            {
                "binary": get_binary(n["name"]),
                "x": float(n["x"]),
                "y": float(n["y"]),
                "z": float(n["z"]),
            }
            for n in graph_nodes
            if get_binary(n["name"]) is not None
        ]
        print(f"[ARIA:Core] Registered {len(_state['all_nodes']):,} nodes in Guard Layer")
    except Exception as e:
        _handle_error("Graph Ingestion", e)
        return False

    # If the source file itself was a text file and no separate script was provided, use it
    if not script_text and not filepath.endswith(".json") and os.path.isfile(filepath):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                script_text = f.read()
        except Exception:
            pass

    # Resonance Loop (Directed text transitions & drift)
    if script_text:
        _phase("Directed Resonance Loop", "🔗")
        try:
            def _interim_save(conns):
                stigma_save(1, conns, _state["all_nodes"])

            run_loop(
                script_text=script_text,
                connections=_state["connections"],
                max_passes=5,
                window_size=5,
                save_callback=_interim_save,
            )
            print(f"[ARIA:Core] {_count_connections(_state['connections']):,} directed synapses established")
        except Exception as e:
            _handle_error("Resonance Loop", e)
            return False

    # Stigma Save
    _phase("Stigma Snapshot Save", "💾")
    try:
        _rotate_and_save(_state["connections"], _state["all_nodes"])
    except Exception as e:
        _handle_error("Stigma Save", e)

    _state["initialized"] = True
    _phase("ARIA READY", "✅")
    print(f"[ARIA:Core]  Nodes         : {len(_state['all_nodes']):,}")
    print(f"[ARIA:Core]  Synapses      : {_count_connections(_state['connections']):,}")
    return True


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------

def aria_query(text: str, top_n: int = 10, max_depth: int = 3) -> dict | None:
    """Run a sequence-aware query with concept chain generation."""
    if not _check_initialized("aria_query"):
        return None
    try:
        return _engine_query(
            text=text,
            connections=_state["connections"],
            top_n=top_n,
            max_depth=max_depth,
        )
    except Exception as e:
        _handle_error("Query Engine", e)
        return None


def aria_add(name: str, context_text: str = "", x: float = None, y: float = None, z: float = None) -> dict | None:
    """Dynamically add a new concept on the fly."""
    if not _check_initialized("aria_add"):
        return None

    _phase(f"Adding '{name}'", "➕")
    try:
        res = route_new_word(
            name=name,
            x=x, y=y, z=z,
            connections=_state["connections"],
            all_nodes=_state["all_nodes"],
            script_text=context_text,
        )
        _rotate_and_save(_state["connections"], _state["all_nodes"])
        return res
    except Exception as e:
        _handle_error("Add Word", e)
        return None


def aria_bridge(word_a: str, word_b: str, weight: float = 1.0) -> dict | None:
    """Create a bidirectional cross-language or equivalence bridge between two words."""
    if not _check_initialized("aria_bridge"):
        return None

    _phase(f"Bridging '{word_a}' ⟷ '{word_b}'", "🌉")
    try:
        res = create_bridge(word_a, word_b, _state["connections"], weight=weight)
        
        # Ensure both nodes are in _state["all_nodes"]
        for w in (word_a, word_b):
            b = get_binary(w)
            pos = get_frequency_by_binary(b) if b else None
            if b and pos and not any(n.get("binary") == b for n in _state["all_nodes"]):
                _state["all_nodes"].append({"binary": b, "x": pos["x"], "y": pos["y"], "z": pos["z"]})
        
        _rotate_and_save(_state["connections"], _state["all_nodes"])
        return res
    except Exception as e:
        _handle_error("Bridge Creation", e)
        return None


def aria_save(slot: int = 1) -> bool:
    """Save to Stigma slot."""
    if not _check_initialized("aria_save"):
        return False
    return stigma_save(slot, _state["connections"], _state["all_nodes"])


def aria_load(slot: int = 1) -> bool:
    """Load from Stigma slot."""
    try:
        res = stigma_load(slot)
        if res is None:
            return False
        _state["connections"] = res["connections"]
        _state["all_nodes"] = res["nodes"]
        for n in _state["all_nodes"]:
            auto_register(_from_binary(n["binary"]), n["x"], n["y"], n["z"])
        _state["initialized"] = True
        return True
    except Exception as e:
        _handle_error("Load Slot", e)
        return False


def aria_status() -> dict:
    """Print and return system status."""
    status = {
        "initialized": _state["initialized"],
        "total_nodes": len(_state["all_nodes"]),
        "total_synapses": _count_connections(_state["connections"]),
        "working_memory": get_session_memory(),
    }
    print(f"\n[ARIA:Core] ─── Status ──────────────────────────────────")
    print(f"[ARIA:Core]  Initialized   : {status['initialized']}")
    print(f"[ARIA:Core]  Nodes         : {status['total_nodes']:,}")
    print(f"[ARIA:Core]  Synapses      : {status['total_synapses']:,}")
    print(f"[ARIA:Core]  Working Memory: {status['working_memory']}")
    print(f"[ARIA:Core] ────────────────────────────────────────────")
    return status


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def _run_cli() -> None:
    print(f"\n[ARIA:Core] Interactive CLI Ready — Type query or command:")
    print(f"[ARIA:Core] Commands: status | memory | clear_memory | save <1-3> | load <1-3> | saves | add <word> | bridge <w1> <w2> | quit\n")

    while True:
        try:
            raw = input("[ARIA] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[ARIA:Core] Goodbye.")
            break

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in ("quit", "exit", "q"):
            print("[ARIA:Core] Goodbye.")
            break
        elif cmd == "status":
            aria_status()
        elif cmd == "memory":
            print(f"[ARIA:Core] Working Memory: {get_session_memory()}")
        elif cmd == "clear_memory":
            clear_session_memory()
            print("[ARIA:Core] Working memory cleared.")
        elif cmd == "saves":
            list_saves()
        elif cmd.startswith("save"):
            parts = raw.split()
            slot = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            aria_save(slot)
        elif cmd.startswith("load"):
            parts = raw.split()
            slot = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            aria_load(slot)
        elif cmd.startswith("bridge "):
            parts = raw.split()
            if len(parts) >= 3:
                w1, w2 = parts[1], parts[2]
                aria_bridge(w1, w2)
            else:
                print("[ARIA:Core] Usage: bridge <word1> <word2>")
        elif cmd.startswith("add "):
            word = raw[4:].strip()
            ctx = input(f"[ARIA] Context text for '{word}': ").strip()
            aria_add(name=word, context_text=ctx)
        else:
            res = aria_query(raw)
            if res:
                print(f"\n[ARIA:Core] Status: {res['status']} (Polarity: {'[-] Negation' if res.get('polarity') == -1 else '[+] Affirmation'})")
                if res.get("concept_chain"):
                    chain_str = " ➔ ".join([c["word"] for c in res["concept_chain"]])
                    print(f"[ARIA:Core] 🔗 Concept Chain: {chain_str}")
                print(f"[ARIA:Core] Top Results ({len(res['results'])} nodes):")
                for i, n in enumerate(res["results"][:10], 1):
                    w = n.get("word", _from_binary(n["binary"]))
                    pol_icon = "[-]" if n.get("polarity") == -1 else "[+]"
                    print(f"   {i:2}. {pol_icon} {w:<23} (weight={n['weight']:.4f})")
                print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARIA Neural Graph Engine")
    parser.add_argument("--build", "-b", default="", help="Path to text or JSON dataset to build/train")
    parser.add_argument("--script", "-s", default="", help="Path to text script for Resonance training")
    args = parser.parse_args()

    script_content = ""
    if args.script and os.path.isfile(args.script):
        with open(args.script, "r", encoding="utf-8", errors="ignore") as f:
            script_content = f.read()

    ok = aria_init(filepath=args.build, script_text=script_content)
    if not ok:
        sys.exit(1)

    _run_cli()
