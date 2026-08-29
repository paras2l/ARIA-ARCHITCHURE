"""
ARIA — 3D Knowledge Graph Visualizer
=====================================
Opens an interactive 3D graph in the browser showing every
word/concept (node) and every resonance connection (edge).

Loads automatically from the latest Stigma save — no file path needed.

Usage
-----
    python aria/scripts/visualize.py
    python aria/scripts/visualize.py --slot 2
    python aria/scripts/visualize.py --min-weight 0.5
    python aria/scripts/visualize.py --max-edges 50000

Requirements
------------
    pip install plotly
"""

import sys
import os
import math
import argparse

# ---------------------------------------------------------------------------
# Path setup — make sure project root is importable
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# ARIA imports
# ---------------------------------------------------------------------------
try:
    from aria.stigma.manager import load as stigma_load, list_saves
    from aria.guard.layer    import _from_binary, get_frequency_by_binary
except ImportError as e:
    print(f"[ARIA:Viz] ❌ Cannot import ARIA modules: {e}")
    print(f"[ARIA:Viz]    Run from project root:  python aria/scripts/visualize.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Plotly import
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
except ImportError:
    print("[ARIA:Viz] ❌ Plotly not installed.")
    print("[ARIA:Viz]    Run:  pip install plotly")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist(x: float, y: float, z: float) -> float:
    """Euclidean distance from origin."""
    return math.sqrt(x * x + y * y + z * z)


def _decode(binary: str) -> str:
    """Decode binary signature → word. Falls back gracefully."""
    try:
        return _from_binary(binary)
    except Exception:
        return binary[:18] + "…"


def _load_stigma(preferred_slot: int = 0) -> dict | None:
    """Try Stigma slots preferred → 3 → 2 → 1. Return first found."""
    order = [preferred_slot] if preferred_slot else []
    order += [s for s in (1, 2, 3) if s not in order]

    for slot in order:
        try:
            data = stigma_load(slot)
            if data is not None:
                print(f"[ARIA:Viz] ✅ Loaded Stigma slot {slot}")
                return data
        except Exception as e:
            print(f"[ARIA:Viz] ⚠️  Slot {slot} error: {e}")
    return None


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------

def build_and_show(
    slot:       int   = 0,
    min_weight: float = 0.5,
    max_edges:  int   = 100_000,
) -> None:
    """Load Stigma, build traces, open browser."""

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"\n[ARIA:Viz] ════════════════════════════════════════")
    print(f"[ARIA:Viz]  ARIA 3D Knowledge Graph Visualizer")
    print(f"[ARIA:Viz] ════════════════════════════════════════")
    print(f"[ARIA:Viz] Loading Stigma …")
    list_saves()

    data = _load_stigma(preferred_slot=slot)
    if data is None:
        print(f"\n[ARIA:Viz] ❌ No valid Stigma save found.")
        print(f"[ARIA:Viz]    First build:  python aria/core.py --build <file>")
        sys.exit(1)

    nodes_raw   = data.get("nodes",       [])
    connections = data.get("connections", {})

    if not nodes_raw:
        print(f"[ARIA:Viz] ❌ Stigma has 0 nodes — nothing to draw.")
        sys.exit(1)

    total_raw_edges = sum(len(v) for v in connections.values())
    print(f"[ARIA:Viz] {len(nodes_raw):,} nodes  ·  "
          f"{total_raw_edges:,} raw connections in save")

    # ── Decode nodes ─────────────────────────────────────────────────────────
    print(f"[ARIA:Viz] Decoding {len(nodes_raw):,} nodes …")

    # Count connections per binary for node sizing
    conn_count: dict[str, int] = {}
    for src_b, targets in connections.items():
        conn_count[src_b] = conn_count.get(src_b, 0) + len(targets)
        for conn in targets:
            tgt_b = conn["to"]   # extract binary string from connection dict
            conn_count[tgt_b] = conn_count.get(tgt_b, 0) + 1

    # Build node info map  binary → {name, x, y, z, dist, n_conn}
    node_map: dict[str, dict] = {}
    for i, node in enumerate(nodes_raw, 1):
        binary = node.get("binary", "")
        if not binary:
            continue

        # Get x,y,z — try node dict first, then Guard Layer
        x = node.get("x")
        y = node.get("y")
        z = node.get("z")

        if x is None or y is None or z is None:
            freq = get_frequency_by_binary(binary)
            if freq:
                x, y, z = freq["x"], freq["y"], freq["z"]
            else:
                x = y = z = 0.0

        x, y, z = float(x), float(y), float(z)
        name    = _decode(binary)
        n_conn  = conn_count.get(binary, 0)

        node_map[binary] = {
            "name":   name,
            "x":      x,
            "y":      y,
            "z":      z,
            "dist":   _dist(x, y, z),
            "n_conn": n_conn,
        }

        if i % 5000 == 0:
            print(f"[ARIA:Viz]   {i:,} / {len(nodes_raw):,} decoded …")

    print(f"[ARIA:Viz] {len(node_map):,} nodes decoded ✅")

    # ── Collect + filter edges ────────────────────────────────────────────────
    print(f"[ARIA:Viz] Building edges "
          f"(min_weight={min_weight}, cap={max_edges:,}) …")

    all_edges: list[tuple] = []   # (src_binary, tgt_binary, weight)
    for src_b, targets in connections.items():
        if src_b not in node_map:
            continue
        for conn in targets:
            tgt_b  = conn["to"]
            weight = conn.get("weight", 0.0)
            if tgt_b not in node_map:
                continue
            w = float(weight)
            if w < min_weight:
                continue
            all_edges.append((src_b, tgt_b, w))

    # Sort by weight descending — draw strongest first
    all_edges.sort(key=lambda e: e[2], reverse=True)
    drawn_edges = all_edges[:max_edges]

    print(f"[ARIA:Viz] {len(drawn_edges):,} edges drawn  "
          f"(of {len(all_edges):,} above min_weight, "
          f"{total_raw_edges:,} total)")

    # ── TRACE 1: Edges ────────────────────────────────────────────────────────
    print(f"[ARIA:Viz] Building edge trace …")

    # Use 4 weight bands with different colours/opacities for clarity
    BANDS = [
        (0.75, 1.01, "rgba(130, 130, 255, 0.90)", "Strong edges  (≥0.75)"),
        (0.50, 0.75, "rgba(100, 100, 200, 0.55)", "Medium edges  (0.50–0.75)"),
        (0.30, 0.50, "rgba( 80,  80, 160, 0.30)", "Weak edges    (0.30–0.50)"),
        (0.00, 0.30, "rgba( 60,  60, 120, 0.15)", "Faint edges   (<0.30)"),
    ]

    edge_traces = []
    for lo, hi, colour, label in BANDS:
        band = [e for e in drawn_edges if lo <= e[2] < hi]
        if not band:
            continue

        xs, ys, zs = [], [], []
        for src_b, tgt_b, _ in band:
            s = node_map[src_b]
            t = node_map[tgt_b]
            # None creates a gap — keeps edges separate in one trace
            xs += [s["x"], t["x"], None]
            ys += [s["y"], t["y"], None]
            zs += [s["z"], t["z"], None]

        edge_traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            name=label,
            hoverinfo="skip",
            line=dict(color=colour, width=1),
        ))

    # ── TRACE 2: Nodes ────────────────────────────────────────────────────────
    print(f"[ARIA:Viz] Building node trace …")

    # Separate Universe node (at origin) from regular nodes
    regular_bins  = []
    universe_bins = []

    for binary, info in node_map.items():
        if info["name"].lower() == "universe" or (
            info["x"] == 0.0 and info["y"] == 0.0 and info["z"] == 0.0
        ):
            universe_bins.append(binary)
        else:
            regular_bins.append(binary)

    # Regular nodes
    xs_n, ys_n, zs_n      = [], [], []
    dists, sizes, hovers   = [], [], []

    for binary in regular_bins:
        info = node_map[binary]
        xs_n.append(info["x"])
        ys_n.append(info["y"])
        zs_n.append(info["z"])
        dists.append(info["dist"])
        # Size: base 3 + bonus per connection, capped at 12
        sizes.append(min(12, 3 + info["n_conn"] * 0.5))
        hovers.append(
            f"<b>{info['name']}</b><br>"
            f"Position : ({info['x']:.4f}, {info['y']:.4f}, {info['z']:.4f})<br>"
            f"Distance : {info['dist']:.4f} from origin<br>"
            f"Connections: {info['n_conn']}"
        )

    node_trace = go.Scatter3d(
        x=xs_n, y=ys_n, z=zs_n,
        mode="markers",
        name=f"Nodes ({len(regular_bins):,})",
        text=hovers,
        hoverinfo="text",
        marker=dict(
            size=sizes,
            color=dists,
            colorscale="Plasma",
            reversescale=True,          # bright/yellow = close to origin
            opacity=0.85,
            colorbar=dict(
                title=dict(text="Distance from Centre", font=dict(color="white")),
                thickness=14,
                len=0.55,
                tickfont=dict(color="white"),
            ),
            line=dict(width=0),
        ),
    )

    # Universe node(s)
    uni_xs = [node_map[b]["x"] for b in universe_bins] or [0.0]
    uni_ys = [node_map[b]["y"] for b in universe_bins] or [0.0]
    uni_zs = [node_map[b]["z"] for b in universe_bins] or [0.0]
    uni_hover = [
        f"<b>Universe</b><br>Position: (0, 0, 0)<br>Centre of ARIA sphere"
    ] * max(len(universe_bins), 1)

    universe_trace = go.Scatter3d(
        x=uni_xs, y=uni_ys, z=uni_zs,
        mode="markers+text",
        name="Universe (origin)",
        text=["Universe"] * max(len(universe_bins), 1),
        textfont=dict(color="white", size=11),
        textposition="top center",
        hovertext=uni_hover,
        hoverinfo="text",
        marker=dict(
            size=10,
            color="white",
            symbol="diamond",
            opacity=1.0,
            line=dict(color="yellow", width=1),
        ),
    )

    # ── Compose figure ────────────────────────────────────────────────────────
    print(f"[ARIA:Viz] Rendering …")

    all_traces = edge_traces + [node_trace, universe_trace]

    axis_style = dict(
        showbackground=True,
        backgroundcolor="rgb(10, 10, 20)",
        gridcolor="rgb(35, 35, 60)",
        zerolinecolor="rgb(60, 60, 100)",
        color="rgb(140, 140, 180)",
    )

    total_nodes = len(node_map)
    total_shown_edges = len(drawn_edges)

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=dict(
            text=(
                f"ARIA — 3D Knowledge Graph<br>"
                f"<sup>{total_nodes:,} nodes · "
                f"{total_shown_edges:,} edges shown · "
                f"min weight ≥ {min_weight}</sup>"
            ),
            x=0.5,
            font=dict(color="white", size=20, family="monospace"),
        ),
        paper_bgcolor="rgb(5, 5, 15)",
        plot_bgcolor="rgb(5, 5, 15)",
        scene=dict(
            bgcolor="rgb(5, 5, 15)",
            xaxis=dict(**axis_style, title="X"),
            yaxis=dict(**axis_style, title="Y"),
            zaxis=dict(**axis_style, title="Z"),
        ),
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(5, 5, 15, 0.8)",
            bordercolor="rgb(60, 60, 100)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="rgb(20, 20, 45)",
            font_size=13,
            font_family="monospace",
            font_color="white",
            bordercolor="rgb(80, 80, 160)",
        ),
        margin=dict(l=0, r=0, t=70, b=0),
    )

    output_html = os.path.join(ROOT, "aria_3d_graph.html")
    fig.write_html(output_html, auto_open=False)
    print(f"[ARIA:Viz] 💾 Interactive 3D Graph saved to: {output_html}")
    print(f"\n[ARIA:Viz] ════════════════════════════════════════")
    print(f"[ARIA:Viz]  Opening 3D visualizer in browser …")
    print(f"[ARIA:Viz]  Hover over any node to see details")
    print(f"[ARIA:Viz]  Drag to rotate · Scroll to zoom")
    print(f"[ARIA:Viz] ════════════════════════════════════════\n")

    try:
        import webbrowser
        webbrowser.open(f"file://{output_html}")
    except Exception:
        fig.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ARIA 3D Knowledge Graph Visualizer"
    )
    parser.add_argument(
        "--slot", "-s",
        type=int, default=0,
        help="Stigma slot to load (default: latest found — tries 3 → 2 → 1)",
    )
    parser.add_argument(
        "--min-weight", "-w",
        type=float, default=0.5,
        help="Minimum edge weight to draw (default: 0.5)",
    )
    parser.add_argument(
        "--max-edges", "-e",
        type=int, default=100_000,
        help="Maximum edges to render (default: 100,000)",
    )
    args = parser.parse_args()

    build_and_show(
        slot=args.slot,
        min_weight=args.min_weight,
        max_edges=args.max_edges,
    )
