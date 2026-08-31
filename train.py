"""
ARIA — Master Brain Trainer CLI (With Epochs, Dynamic LR Annealing, Loss & Convergence)
=======================================================================================
Features:
1. Epoch / Multi-Pass Training: Iterates through datasets with dynamic learning rate decay.
2. Learning Rate Annealing (LR): Anneals 3D coordinate drift from 0.1500 -> 0.0975 -> 0.0634...
3. Gravitational Drift Loss (L_drift): Tracks spatial 3D manifold convergence.
4. Auto-Convergence: Automatically detects when 0 new connections are formed and stops early.
5. Final Architecture Metrics: Displays total nodes, final directed synapses, density, and loss.

Usage:
------
python train.py --all --passes 3
python train.py --all --data-dir /kaggle/input/aria-datasets --out-dir /kaggle/working --passes 3
"""

import sys
import os
import time
import json
import math
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aria.guard.layer import clear_store, auto_register, _store, _from_binary, get_binary
from aria.resonance.block import run_block1, prune_synapses
from aria.stigma.manager import save as stigma_save, load as stigma_load
from aria.query.engine import query, clear_session_memory

DEFAULT_DATASETS_DIR = os.path.join(ROOT, "datasets")
DEFAULT_OUT_DIR = ROOT


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_checkpoint_state_path(out_dir: str, slot: int) -> str:
    return os.path.join(out_dir, f"aria_checkpoint_slot_{slot}.json")


def save_checkpoint(out_dir: str, slot: int, connections: dict, completed_files: list[str], elapsed_s: float) -> None:
    """Save full Stigma slot + checkpoint metadata state."""
    nodes_list = [
        {"binary": b, "x": pos["x"], "y": pos["y"], "z": pos["z"]}
        for b, pos in _store.items()
    ]
    prev_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        stigma_save(slot, connections, nodes_list)
    finally:
        os.chdir(prev_cwd)

    state_path = get_checkpoint_state_path(out_dir, slot)
    total_edges = sum(len(v) for v in connections.values())
    state = {
        "slot": slot,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_files": completed_files,
        "total_nodes": len(nodes_list),
        "total_synapses": total_edges,
        "elapsed_seconds": round(elapsed_s, 2),
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    # Auto-archive checkpoint files into aria_trained_brain.zip for instant 1-click download
    try:
        import zipfile
        zip_path = os.path.join(out_dir, "aria_trained_brain.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            f_json = os.path.join(out_dir, f"aria_stigma_slot_{slot}.json")
            f_bin = os.path.join(out_dir, f"aria_stigma_slot_{slot}_conn.bin")
            f_state = state_path
            if os.path.exists(f_json):
                zf.write(f_json, arcname=os.path.basename(f_json))
            if os.path.exists(f_bin):
                zf.write(f_bin, arcname=os.path.basename(f_bin))
            if os.path.exists(f_state):
                zf.write(f_state, arcname=os.path.basename(f_state))
    except Exception:
        pass

    time_str = _format_time(elapsed_s)
    print(f"\n✅ [SAVED CHECKPOINT] slot={slot}, completed={len(completed_files)} files, nodes={len(nodes_list):,}, synapses={total_edges:,}, elapsed={time_str}")
    print(f"📦 [AUTO-ZIP READY] -> {os.path.join(out_dir, 'aria_trained_brain.zip')}\n")


def load_checkpoint(out_dir: str, slot: int) -> tuple[dict, list[str]]:
    """Attempt to restore graph state and list of completed files."""
    state_path = get_checkpoint_state_path(out_dir, slot)
    completed_files = []
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
                completed_files = state.get("completed_files", [])
        except Exception:
            pass

    prev_cwd = os.getcwd()
    connections = {}
    try:
        os.chdir(out_dir)
        save_data = stigma_load(slot)
        if save_data:
            connections = save_data["connections"]
            for n in save_data["nodes"]:
                auto_register(_from_binary(n["binary"]), n["x"], n["y"], n["z"])
            print(f"✅ RESUMED → slot={slot}, completed={len(completed_files)} files, nodes={len(_store):,}, synapses={sum(len(v) for v in connections.values()):,}")
    finally:
        os.chdir(prev_cwd)

    return connections, completed_files


def train_single_file_with_progress(
    fp: str,
    connections: dict,
    file_idx: int,
    total_files: int,
    passes: int = 3,
    chunk_lines: int = 2500,
):
    fname = os.path.basename(fp)
    fsize_mb = os.path.getsize(fp) / (1024 * 1024)

    # Read file lines
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    total_lines = len(lines)
    if total_lines == 0:
        return 0.0

    print(f"\n[Dataset {file_idx}/{total_files}] {fname} ({fsize_mb:.2f} MB, {total_lines:,} lines)")

    t_file_start = time.time()
    last_drift_loss = 0.0

    for p in range(1, passes + 1):
        pass_start = time.time()
        num_chunks = math.ceil(total_lines / chunk_lines)
        synapses_before_pass = sum(len(v) for v in connections.values())
        new_synapses_in_pass = 0

        for c_idx in range(num_chunks):
            c_start = c_idx * chunk_lines
            c_end = min((c_idx + 1) * chunk_lines, total_lines)
            chunk_text = "".join(lines[c_start:c_end])

            # Run 3D resonance on chunk with pass_num for learning rate decay
            res = run_block1(chunk_text, connections, pass_num=p)
            last_drift_loss = res.get("drift_loss", 0.0)
            drift_lr = res.get("drift_lr", 0.15)
            new_synapses_in_pass += res.get("new_connections", 0)

            # Calculate Progress Metrics
            elapsed = max(0.001, time.time() - pass_start)
            processed_lines = c_end
            pct = (processed_lines / total_lines) * 100
            rate = processed_lines / elapsed

            remaining_lines = total_lines - processed_lines
            eta = remaining_lines / max(1.0, rate)

            time_str = f"[{_format_time(elapsed)}<{_format_time(eta)}, {rate:,.1f} lines/s]"
            nodes_count = len(_store)
            syn_count = sum(len(v) for v in connections.values())

            # Progress Bar Rendering (0-100%)
            bar_len = 20
            filled_len = int(bar_len * processed_lines // total_lines)
            bar = "█" * filled_len + "░" * (bar_len - filled_len)

            status_line = (
                f"\rEpoch {p}/{passes}: {pct:3.0f}%|{bar}| "
                f"{processed_lines:,}/{total_lines:,} {time_str}, "
                f"nodes={nodes_count:,}, syn={syn_count:,} (+{new_synapses_in_pass:,}), "
                f"drift_loss={last_drift_loss:.4f}, lr={drift_lr:.4f}"
            )
            sys.stdout.write(status_line)
            sys.stdout.flush()

        print()  # Newline after epoch

        # Convergence Check: If 0 new connections formed and drift loss is tiny, stop early!
        if p >= 2 and new_synapses_in_pass == 0 and last_drift_loss < 0.005:
            print(f"  🎯 [CONVERGED] Graph reached stable equilibrium at Epoch {p} (0 new synapses).")
            break

    # Prune weak noise
    pruned = prune_synapses(connections, min_weight=0.05, min_count=1)
    if pruned > 0:
        print(f"  ✂️ [PRUNED] {pruned:,} weak noise edges.")

    return round(time.time() - t_file_start, 2)


def train_master(
    filepaths: list[str],
    slot: int = 1,
    max_passes: int = 3,
    out_dir: str = DEFAULT_OUT_DIR,
    fresh: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    is_kaggle = os.path.exists("/kaggle")
    env_name = "Kaggle Cloud" if is_kaggle else "Local PC / Workstation"

    from aria.resonance.accelerator import get_hardware_info
    hw = get_hardware_info()

    print("\n" + "="*70)
    print("🚀 ARIA MASTER BRAIN TRAINING ENGINE")
    print("="*70)
    print(f"ENV            : {env_name}")
    print(f"Device Mode    : {hw['display']}")
    print(f"Output Dir     : {out_dir}")
    print(f"Active Slot    : {slot}")
    print(f"Max Epochs     : {max_passes} Passes (Auto-Stopping on 0 new connections)")
    print(f"Total Datasets : {len(filepaths)} Files")
    print("="*70)

    if fresh:
        clear_store()
        connections = {}
        completed_files = []
        print("🌱 FRESH START: Previous checkpoints cleared.")
    else:
        connections, completed_files = load_checkpoint(out_dir, slot)

    print("\nStarting Training...\n")
    t_global_start = time.time()
    files_processed_count = 0

    for idx, fp in enumerate(filepaths, 1):
        fname = os.path.basename(fp)
        if fname in completed_files:
            print(f"⏩ [Dataset {idx}/{len(filepaths)}] Skipping '{fname}' (Already completed in checkpoint).")
            continue

        if not os.path.isfile(fp):
            print(f"⚠️ File not found: {fp}")
            continue

        train_single_file_with_progress(
            fp=fp,
            connections=connections,
            file_idx=idx,
            total_files=len(filepaths),
            passes=max_passes,
        )

        completed_files.append(fname)
        files_processed_count += 1
        elapsed_total = time.time() - t_global_start
        save_checkpoint(out_dir, slot, connections, completed_files, elapsed_total)

    total_nodes = len(_store)
    total_edges = sum(len(v) for v in connections.values())
    total_time_str = _format_time(time.time() - t_global_start)
    avg_density = round(total_edges / max(1, total_nodes), 2)

    print("="*70)
    print(f"🎉 MASTER TRAINING COMPLETE IN {total_time_str}!")
    print(f"💎 Total Knowledge Concepts (Nodes)  : {total_nodes:,}")
    print(f"🧠 Total Final Directed 3D Synapses  : {total_edges:,}")
    print(f"🌐 Average Synaptic Density          : {avg_density} synapses/concept")
    print(f"📦 Datasets Processed This Run       : {files_processed_count} / {len(filepaths)}")
    print(f"💾 Final Master Brain Saved In Slot  : Slot {slot} ({out_dir})")
    print("="*70 + "\n")


def interactive_cli(slot: int = 1, out_dir: str = DEFAULT_OUT_DIR):
    prev_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        save_data = stigma_load(slot)
    finally:
        os.chdir(prev_cwd)

    if not save_data:
        print("❌ No brain found in slot. Please train first.")
        return

    connections = save_data["connections"]
    for n in save_data["nodes"]:
        auto_register(_from_binary(n["binary"]), n["x"], n["y"], n["z"])

    print("\n[ARIA] ═════════════════════════════════════════════════")
    print(f"[ARIA] Brain Loaded: {len(_store):,} Concepts · {sum(len(v) for v in connections.values()):,} Synapses")
    print("[ARIA] Type your question or prompt (or 'exit' to quit):")
    print("[ARIA] ═════════════════════════════════════════════════\n")

    while True:
        try:
            q = input("[ARIA] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q or q.lower() in ("exit", "quit", "q"):
            break

        res = query(q, connections, top_n=6)
        if res and res.get("results"):
            if res.get("concept_chain"):
                chain_str = " ➔ ".join([c["word"] for c in res["concept_chain"]])
                print(f"\n[ARIA] 🔗 Thought Trajectory: {chain_str}")
            print(f"[ARIA] Top Retrieved Knowledge:")
            for i, r in enumerate(res["results"][:6], 1):
                print(f"   {i}. {r['word']:<22} (confidence={r['weight']:.4f})")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIA Master Brain Trainer with Epochs, LR Decay & Auto-Convergence")
    parser.add_argument("file", nargs="?", default="", help="Path to specific text dataset to train on")
    parser.add_argument("--all", action="store_true", help="Train on all dataset files in data-dir")
    parser.add_argument("--data-dir", default="", help="Path to directory containing dataset .txt files")
    parser.add_argument("--out-dir", default="", help="Path to output directory for saving Stigma slots")
    parser.add_argument("--slot", type=int, default=1, help="Stigma save slot (1, 2, or 3)")
    parser.add_argument("--passes", "--epochs", type=int, default=3, help="Number of passes/epochs per file (default: 3)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh and ignore existing checkpoints")
    parser.add_argument("--chat", action="store_true", help="Enter interactive chat immediately")
    args = parser.parse_args()

    data_directory = args.data_dir if args.data_dir else DEFAULT_DATASETS_DIR
    output_directory = args.out_dir if args.out_dir else DEFAULT_OUT_DIR

    if args.chat:
        interactive_cli(slot=args.slot, out_dir=output_directory)
        sys.exit(0)

    if args.all:
        target_files = []
        if os.path.isdir(data_directory):
            for root, dirs, files in os.walk(data_directory):
                for f in sorted(files):
                    if f.endswith(".txt"):
                        target_files.append(os.path.join(root, f))
        
        if not target_files:
            print(f"❌ No .txt dataset files found in: {data_directory}")
            sys.exit(1)

        train_master(
            target_files,
            slot=args.slot,
            max_passes=args.passes,
            out_dir=output_directory,
            fresh=args.fresh,
        )
    elif args.file:
        train_master(
            [args.file],
            slot=args.slot,
            max_passes=args.passes,
            out_dir=output_directory,
            fresh=args.fresh,
        )
    else:
        print("Usage:")
        print("  python train.py --all --epochs 3 --data-dir <path> --out-dir <path>")
        print("  python train.py <file.txt>")
        print("  python train.py --chat")
