#!/usr/bin/env python3
"""Compile, validate, and rank CUDA wave configurations.

Examples:
  python3 sweep_wave.py --scene small --steps 33
  python3 sweep_wave.py --scene large --steps 129,257 \
      --tiles 32x32,64x32,64x64 --k 1,2,4,8 \
      --blocks 8x8,16x8,32x8,16x16 --output results.csv\n\nPlots are saved automatically (PNG and SVG). Use --plot-only to recreate\nfigures from an existing CSV without requiring CUDA.

K = fused timesteps per launch. --steps = total simulated timesteps.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def dims(text):
    out = []
    for value in text.split(','):
        try:
            x, y = value.lower().split('x')
            pair = int(x), int(y)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid 2D size {value!r}; use 32x16") from exc
        if min(pair) <= 0:
            raise argparse.ArgumentTypeError("Dimensions must be positive")
        out.append(pair)
    return out


def ints(text):
    try:
        values = [int(part) for part in text.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("Values must be positive")
    return values


def write_csv(path, rows):
    fieldnames = ["status", "scene", "steps", "tile_x", "tile_y", "k",
                  "block_x", "block_y", "out_x", "out_y", "shared_bytes",
                  "launches", "time_ms", "ms_per_step", "error_prev", "error_curr",
                  "max_abs_diff", "correct", "message"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(csv_path, plots_dir, top_n, formats):
    """Save plots for passing configurations; never compare different workloads directly."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Works on a remote server without a display.
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Plots require matplotlib and numpy. Install with: "
            "python3 -m pip install matplotlib numpy"
        ) from exc

    with csv_path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    results = []
    for row in raw_rows:
        if row.get("status") != "ok":
            continue
        try:
            record = {
                "scene": row["scene"],
                "steps": int(row["steps"]),
                "tile_x": int(row["tile_x"]),
                "tile_y": int(row["tile_y"]),
                "k": int(row["k"]),
                "block_x": int(row["block_x"]),
                "block_y": int(row["block_y"]),
                "time_ms": float(row["time_ms"]),
            }
        except (KeyError, ValueError, TypeError):
            continue
        if record["time_ms"] <= 0 or not math.isfinite(record["time_ms"]):
            continue
        results.append(record)

    if not results:
        print("No valid, correct benchmark rows found; no plots generated.")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(fig, basename):
        for extension in formats:
            path = plots_dir / f"{basename}.{extension}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)

    def envelope(rows, key):
        """Minimum observed runtime per value of key, optimizing other dimensions."""
        minima = {}
        for row in rows:
            value = key(row)
            minima[value] = min(minima.get(value, math.inf), row["time_ms"])
        return minima

    grouped = {}
    for row in results:
        grouped.setdefault((row["scene"], row["steps"]), []).append(row)

    for (scene, steps), rows in sorted(grouped.items()):
        prefix = f"{safe_filename(scene)}_steps{steps}"
        subtitle = (f"Scene: {scene}  |  Total steps: {steps}  |  "
                    "Correct runs only; lower is better")

        # 1. Actual fastest complete configurations (no parameter aggregation).
        top = sorted(rows, key=lambda row: row["time_ms"])[:top_n]
        labels = [
            (f"Tile {r['tile_x']}x{r['tile_y']}  |  "
             f"K={r['k']}  |  Block {r['block_x']}x{r['block_y']}")
            for r in top
        ]
        times = [r["time_ms"] for r in top]
        fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(top) + 1.5)))
        ax.barh(range(len(top)), times)
        ax.set_yticks(range(len(top)), labels)
        ax.invert_yaxis()
        ax.set_xlim(left=0, right=max(times) * 1.22)
        for i, value in enumerate(times):
            ax.text(value + max(times) * 0.01, i, f"{value:.3f} ms", va="center", fontsize=9)
        ax.set_xlabel("Full simulation time (ms)")
        ax.set_title(f"Fastest configurations\n{subtitle}")
        ax.grid(axis="x", alpha=0.2)
        save(fig, prefix + "_top_configs")

        # 2. Effect of fused timesteps per launch (other parameters tuned).
        by_k = envelope(rows, lambda r: r["k"])
        ks = sorted(by_k)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(ks, [by_k[k] for k in ks], marker="o")
        ax.set_xticks(ks)
        ax.set_xlabel("K (timesteps per kernel launch)")
        ax.set_ylabel("Best observed full simulation time (ms)")
        ax.set_title(f"Runtime vs K\n{subtitle}")
        ax.grid(alpha=0.2)
        ax.text(0.01, -0.19, "Minimum across tested tile and thread-block sizes for each K",
                transform=ax.transAxes, fontsize=9)
        save(fig, prefix + "_by_k")

        # 3. Effect of tile size (other parameters tuned).
        by_tile = envelope(rows, lambda r: (r["tile_x"], r["tile_y"]))
        tile_items = sorted(by_tile.items(), key=lambda item: item[1])
        fig, ax = plt.subplots(figsize=(max(7, len(tile_items) * 0.8 + 2), 4.8))
        ax.bar(range(len(tile_items)), [v for _, v in tile_items])
        ax.set_xticks(range(len(tile_items)), [f"{x}x{y}" for (x, y), _ in tile_items], rotation=30, ha="right")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Best observed full simulation time (ms)")
        ax.set_xlabel("Input tile size")
        ax.set_title(f"Runtime vs input tile size\n{subtitle}")
        ax.grid(axis="y", alpha=0.2)
        ax.text(0.01, -0.25, "Minimum across tested K and thread-block sizes for each tile",
                transform=ax.transAxes, fontsize=9)
        save(fig, prefix + "_by_tile")

        # 4. Effect of CUDA thread-block dimensions (other parameters tuned).
        by_block = envelope(rows, lambda r: (r["block_x"], r["block_y"]))
        block_items = sorted(by_block.items(), key=lambda item: item[1])
        fig, ax = plt.subplots(figsize=(max(7, len(block_items) * 0.8 + 2), 4.8))
        ax.bar(range(len(block_items)), [v for _, v in block_items])
        ax.set_xticks(range(len(block_items)),
                      [f"{x}x{y}\n({x*y})" for (x, y), _ in block_items],
                      rotation=25, ha="right")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Best observed full simulation time (ms)")
        ax.set_xlabel("Thread-block dimensions (total threads)")
        ax.set_title(f"Runtime vs threads per block\n{subtitle}")
        ax.grid(axis="y", alpha=0.2)
        ax.text(0.01, -0.27, "Minimum across tested K and tile sizes for each block size",
                transform=ax.transAxes, fontsize=9)
        save(fig, prefix + "_by_block")

        # 5. K x tile heatmap; each entry is the best observed block configuration.
        tiles = sorted(by_tile)
        matrix = np.full((len(ks), len(tiles)), np.nan)
        best_kt = envelope(rows, lambda r: (r["k"], (r["tile_x"], r["tile_y"])))
        for row_idx, k in enumerate(ks):
            for col_idx, tile in enumerate(tiles):
                if (k, tile) in best_kt:
                    matrix[row_idx, col_idx] = best_kt[(k, tile)]
        fig, ax = plt.subplots(figsize=(max(7, len(tiles) * 0.95 + 2), max(4, len(ks) * 0.7 + 1.5)))
        img = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto")
        fig.colorbar(img, ax=ax, label="Full simulation time (ms)")
        ax.set_xticks(range(len(tiles)), [f"{x}x{y}" for x, y in tiles], rotation=30, ha="right")
        ax.set_yticks(range(len(ks)), ks)
        ax.set_xlabel("Input tile size")
        ax.set_ylabel("K (timesteps per launch)")
        ax.set_title(f"K x tile size: best block size for each pair\n{subtitle}")
        save(fig, prefix + "_k_tile_heatmap")

    # 6. Best full-run times for different simulation lengths, grouped by scene.
    scenes = sorted({r["scene"] for r in results})
    for scene in scenes:
        scene_rows = [r for r in results if r["scene"] == scene]
        by_steps = envelope(scene_rows, lambda r: r["steps"])
        if len(by_steps) <= 1:
            continue
        xs = sorted(by_steps)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(xs, [by_steps[x] for x in xs], marker="o")
        ax.set_xlabel("Total simulation timesteps")
        ax.set_ylabel("Best observed full simulation time (ms)")
        ax.set_title(f"Best runtime vs total simulated steps\nScene: {scene}; minimum across tested configurations")
        ax.grid(alpha=0.2)
        save(fig, safe_filename(scene) + "_compare_total_steps")

    return saved


def safe_filename(value):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("wave.cu"))
    parser.add_argument("--scene", choices=["small", "large"], default="large")
    parser.add_argument("--steps", type=ints, default=[129],
                        help="Total simulation step counts, e.g. 129,257 (default: 129)")
    parser.add_argument("--k", type=ints, default=[1, 2, 4],
                        help="Fused timesteps per kernel launch (default: 1,2,4)")
    parser.add_argument("--tiles", type=dims, default=[(32, 32), (64, 32)],
                        help="Input tile shapes, e.g. 32x32,64x32")
    parser.add_argument("--blocks", type=dims, default=[(8, 8), (16, 8), (32, 8)],
                        help="CUDA thread-block shapes, e.g. 8x8,16x8,32x8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-shared-bytes", type=int, default=49152,
                        help="Conservative default dynamic shared memory limit (48 KiB)")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="", help="CUDA architecture, e.g. sm_90 (optional)")
    parser.add_argument("--nvcc-flags", default="", help="Extra flags, pass as --nvcc-flags=-lineinfo")
    parser.add_argument("--build-dir", type=Path, default=Path(".wave_build"))
    parser.add_argument("--output", type=Path, default=Path("wave_sweep_results.csv"))
    parser.add_argument("--plots-dir", type=Path, default=None,
                        help="Where to save plots (default: <output_stem>_plots next to CSV)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Make plots from an existing --output CSV; no nvcc/GPU needed")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV only")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Configurations to show in each leaderboard (default: 15)")
    parser.add_argument("--plot-formats", default="png,svg",
                        help="Comma-separated plot formats: png,svg,pdf (default: png,svg)")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Useful for a quick smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Print candidates without compiling")
    args = parser.parse_args()

    formats = [f.strip().lower() for f in args.plot_formats.split(",")]
    if not formats or any(f not in ("png", "svg", "pdf") for f in formats):
        parser.error("--plot-formats must use png, svg, and/or pdf")
    formats = list(dict.fromkeys(formats))
    if args.top_n < 1:
        parser.error("--top-n must be >= 1")
    if args.plots_dir is None:
        args.plots_dir = args.output.parent / (args.output.stem + "_plots")
    if args.plot_only:
        if not args.output.is_file():
            parser.error(f"Results CSV does not exist: {args.output}")
        if args.no_plots:
            parser.error("--plot-only cannot be used with --no-plots")
        try:
            paths = make_plots(args.output, args.plots_dir, args.top_n, formats)
        except (RuntimeError, OSError) as exc:
            parser.error(str(exc))
        print(f"Saved {len(paths)} plots to {args.plots_dir}")
        return 0 if paths else 1
    if args.warmup < 0 or args.repeats < 1 or args.max_shared_bytes < 1:
        parser.error("Invalid warmup, repeats, or max-shared-bytes")
    args.source = args.source.resolve()
    if not args.source.exists():
        parser.error(f"Source not found: {args.source}")
    if not args.dry_run and not shutil.which(args.nvcc):
        parser.error("nvcc not found. Run on a machine with the CUDA toolkit, or use --dry-run.")

    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()[:10]
    all_candidates = list(itertools.product(args.tiles, args.k, args.blocks, args.steps))
    if args.max_candidates is not None:
        all_candidates = all_candidates[:args.max_candidates]
    print(f"Candidates requested: {len(all_candidates)}; scene={args.scene}", flush=True)
    rows = []
    compile_status = {}

    for number, ((tx, ty), k, (bx, by), steps) in enumerate(all_candidates, 1):
        base = dict(scene=args.scene, steps=steps, tile_x=tx, tile_y=ty,
                    k=k, block_x=bx, block_y=by,
                    out_x=tx - 2*k, out_y=ty - 2*k,
                    shared_bytes=2*tx*ty*4)
        label = f"[{number}/{len(all_candidates)}] tile={tx}x{ty} K={k} threads={bx}x{by} steps={steps}"
        if tx <= 2*k or ty <= 2*k or base["shared_bytes"] > args.max_shared_bytes or bx*by > 1024:
            rows.append({**base, "status": "skipped", "message": "Invalid tile, shared-memory size, or thread count"})
            print(label + " SKIPPED", flush=True)
            continue

        compile_key = (tx, ty, k)
        flags = shlex.split(args.nvcc_flags)
        if args.arch:
            flags.append(f"-arch={args.arch}")
        fingerprint = hashlib.sha256((source_hash + repr(flags)).encode()).hexdigest()[:10]
        binary = args.build_dir / f"wave_{tx}x{ty}_k{k}_{fingerprint}"
        compile_cmd = [args.nvcc, "-O3", "-std=c++17", *flags,
                       f"-DWAVE_IN_TILE_X={tx}", f"-DWAVE_IN_TILE_Y={ty}",
                       f"-DWAVE_K={k}", str(args.source), "-o", str(binary)]
        bench_cmd = [str(binary.resolve()), "--bench-wave", "--scene", args.scene,
                     "--steps", str(steps), "--block-x", str(bx),
                     "--block-y", str(by), "--warmup", str(args.warmup),
                     "--repeats", str(args.repeats)]
        if args.dry_run:
            print(label)
            print("  COMPILE:", shlex.join(compile_cmd))
            print("  RUN:    ", shlex.join(bench_cmd))
            continue

        if compile_key not in compile_status:
            args.build_dir.mkdir(parents=True, exist_ok=True)
            if not binary.exists():
                try:
                    completed = subprocess.run(compile_cmd, text=True, capture_output=True)
                    compile_status[compile_key] = (completed.returncode == 0, completed.stderr)
                except OSError as exc:
                    compile_status[compile_key] = (False, str(exc))
            else:
                compile_status[compile_key] = (True, "")

        compiled, build_message = compile_status[compile_key]
        if not compiled:
            rows.append({**base, "status": "compile_error", "message": build_message[-1000:]})
            print(label + " COMPILE ERROR", flush=True)
            continue

        try:
            result = subprocess.run(bench_cmd, text=True, capture_output=True)
        except OSError as exc:
            rows.append({**base, "status": "run_error", "message": str(exc)})
            print(label + " RUN ERROR", flush=True)
            continue

        json_lines = [line[len("BENCH_JSON="):] for line in result.stdout.splitlines()
                      if line.startswith("BENCH_JSON=")]
        if not json_lines:
            rows.append({**base, "status": "run_error", "message": (result.stderr + result.stdout)[-1000:]})
            print(label + " RUN ERROR", flush=True)
            continue

        parsed = json.loads(json_lines[-1])
        status = "ok" if result.returncode == 0 and parsed.get("correct") else "incorrect"
        row = {**base, **parsed, "status": status,
               "ms_per_step": parsed["time_ms"] / steps,
               "message": result.stderr[-1000:] if result.returncode else ""}
        rows.append(row)
        print(label + f" {status.upper()} {parsed['time_ms']:.4f} ms"
              + f" (errors {parsed['error_prev']:.2e}, {parsed['error_curr']:.2e})", flush=True)
        write_csv(args.output, rows)

    if args.dry_run:
        return 0
    # Rank only correct configurations; keep failures in CSV for debugging.
    # Compare runs only within the same total step count: raw runtimes for
    # 129 and 257 simulated steps are not directly comparable.
    rows.sort(key=lambda row: (int(row["steps"]), row["status"] != "ok",
                               float(row.get("time_ms", float("inf")))))
    write_csv(args.output, rows)
    passed = [r for r in rows if r["status"] == "ok"]
    print(f"\nPassed: {len(passed)} / {len(all_candidates)}; saved: {args.output}")
    for steps in args.steps:
        group = [r for r in passed if r["steps"] == steps]
        print(f"Fastest correct configurations for n_steps={steps}:")
        for rank, row in enumerate(group[:10], 1):
            print(f" {rank:2d}. {row['time_ms']:9.4f} ms  "
                  f"tile={row['tile_x']}x{row['tile_y']} K={row['k']} "
                  f"threads={row['block_x']}x{row['block_y']}")
    if passed and not args.no_plots:
        try:
            paths = make_plots(args.output, args.plots_dir, args.top_n, formats)
        except (RuntimeError, OSError) as exc:
            print(f"Plot generation failed: {exc}", file=sys.stderr)
            return 1
        print(f"Saved {len(paths)} plots to {args.plots_dir}")
    if not passed:
        print("No correct configurations. Inspect CSV 'message' values.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
