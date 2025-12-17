import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt  # Added for plotting
from Parallel.strategy_1_batch import batch_process_paths
from Parallel.strategy_2_bidirectional import run_bidirectional_parallel
from Parallel.strategy_3_region import run_region_parallel
from Sequential.Astar import find_path

# Configuration
GRID_SIZES = [50, 100, 250, 500, 1000, 2500, 5000]
OBSTACLE_DENSITY = 0.34
NUM_TRIALS = 5
NUM_CORES = mp.cpu_count()
BATCH_SIZE = 32  # For Strategy 1: Simulate 32 concurrent users


def create_random_grid(size, density):
    # Optimization: Use int8 to save memory on large grids
    return (np.random.random((size, size)) < density).astype(np.int8)


def generate_benchmark_plots(results, grid_sizes):

    # Extract data for plotting
    seq_times = [results[size]["Sequential"]['mean'] for size in grid_sizes]
    batch_times = [results[size]["Strategy 1: Batch"]['mean'] for size in grid_sizes]
    bi_times = [results[size]["Strategy 2: Bidirectional"]['mean'] for size in grid_sizes]
    reg_times = [results[size]["Strategy 3: Region-Based"]['mean'] for size in grid_sizes]

    # Calculate Speedups
    batch_speedup = [seq / batch if batch > 0 else 0 for seq, batch in zip(seq_times, batch_times)]
    bi_speedup = [seq / bi if bi > 0 else 0 for seq, bi in zip(seq_times, bi_times)]
    reg_speedup = [seq / reg if reg > 0 else 0 for seq, reg in zip(seq_times, reg_times)]

    # --- PLOT 1: Scalability (Execution Time vs Grid Size) ---
    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, seq_times, marker='o', color='red', label='Sequential', linewidth=2)
    plt.plot(grid_sizes, bi_times, marker='^', color='green', label='Bidirectional', linewidth=2)
    plt.plot(grid_sizes, reg_times, marker='s', color='blue', label='Region-Based', linewidth=2)
    plt.plot(grid_sizes, batch_times, marker='x', color='orange', linestyle='--', label='Batch (Avg/Path)', linewidth=2)

    plt.yscale('log')  # Log scale is essential because Sequential grows exponentially
    plt.title('Scalability Analysis: Execution Time vs Grid Size (Log Scale)')
    plt.xlabel('Grid Size (NxN)')
    plt.ylabel('Time (Seconds)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    # --- PLOT 2: Speedup Factor vs Grid Size ---
    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, bi_speedup, marker='^', color='green', label='Bidirectional Speedup')
    plt.plot(grid_sizes, reg_speedup, marker='s', color='blue', label='Region-Based Speedup')
    plt.plot(grid_sizes, batch_speedup, marker='x', color='orange', label='Batch Speedup')
    plt.axhline(y=1, color='red', linestyle='--', label='Baseline (1x)')

    plt.title('Parallel Speedup Factor vs Grid Size')
    plt.xlabel('Grid Size (NxN)')
    plt.ylabel('Speedup (x Times Faster)')
    plt.grid(True)
    plt.legend()


    # --- PLOT 3: The "Hero" Bar Chart (Largest Grid Only) ---
    max_size = grid_sizes[-1]
    labels = ['Sequential', 'Batch', 'Bidirectional', 'Region-Based']
    times_max = [
        results[max_size]["Sequential"]['mean'],
        results[max_size]["Strategy 1: Batch"]['mean'],
        results[max_size]["Strategy 2: Bidirectional"]['mean'],
        results[max_size]["Strategy 3: Region-Based"]['mean']
    ]
    colors = ['#ff9999', '#ffcc99', '#99ff99', '#66b3ff']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times_max, color=colors, edgecolor='black')

    # Add value labels on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title(f'Performance Comparison on Largest Grid ({max_size}x{max_size})')
    plt.ylabel('Execution Time (Seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.show()


def run_benchmark():
    WIDTH = 130

    print("=" * WIDTH)
    print(f"{'COMPREHENSIVE A* BENCHMARK':^{WIDTH}}")
    print(f"{f'Testing on {NUM_CORES} Cores':^{WIDTH}}")
    print("=" * WIDTH)

    results = {}

    strategies = {
        "Sequential": find_path,
        "Strategy 1: Batch": "batch_mode",
        "Strategy 2: Bidirectional": run_bidirectional_parallel,
        "Strategy 3: Region-Based": run_region_parallel,
    }

    for size in GRID_SIZES:
        print(f"\n{'=' * 120}")
        print(f"GRID SIZE: {size} x {size}")
        print(f"{'=' * 120}")

        results[size] = {}

        # 1. Generate Environment for this Grid Size
        trial_grids = []
        for t in range(NUM_TRIALS):
            np.random.seed(42 + size + t)
            g = create_random_grid(size, OBSTACLE_DENSITY)
            s, e = (0, 0), (size - 1, size - 1)
            g[s], g[e] = 0, 0
            trial_grids.append((g, s, e))

        # 2. Run Strategies
        for name, func in strategies.items():
            print(f"--- Testing: {name} ---")

            times = []
            steps_list = []
            success_count = 0

            for t in range(NUM_TRIALS):
                grid, start, goal = trial_grids[t]

                # --- STRATEGY 1: BATCH PROCESSING ---
                if name == "Strategy 1: Batch":
                    batch_jobs = []
                    current_batch = BATCH_SIZE if size < 2500 else 4

                    for _ in range(current_batch):
                        s_r = (np.random.randint(0, size), np.random.randint(0, size))
                        g_r = (np.random.randint(0, size), np.random.randint(0, size))
                        grid_copy = grid.copy()
                        grid_copy[s_r], grid_copy[g_r] = 0, 0
                        batch_jobs.append((grid_copy, s_r, g_r))

                    t0 = time.time()
                    batch_results = batch_process_paths(batch_jobs)
                    t1 = time.time()

                    total_batch_time = t1 - t0
                    avg_time_per_path = total_batch_time / current_batch
                    times.append(avg_time_per_path)

                    if batch_results:
                        avg_batch_steps = np.mean([len(p) for p in batch_results if p])
                        steps_list.append(avg_batch_steps)
                        success_count += 1
                        symbol = "✓"
                    else:
                        steps_list.append(0)
                        symbol = "✗"

                    print(f"  Trial {t + 1}: {current_batch} Paths solved in {total_batch_time:.4f}s "
                          f"| Avg/Path: {avg_time_per_path:.5f}s Path Found:{symbol}")

                # --- STRATEGY 3: REGION ---
                elif name == "Strategy 3: Region-Based":
                    raw_splits = size // 150
                    splits = min(50, max(2, raw_splits))
                    t0 = time.time()
                    path = func(grid, start, goal, splits=splits)
                    t1 = time.time()
                    times.append(t1 - t0)

                    if path:
                        steps_list.append(len(path))
                        success_count += 1
                        symbol = "✓"
                    else:
                        steps_list.append(0)
                        symbol = "✗"

                    print(f"  Trial {t + 1}: {t1 - t0:.5f}s (Splits: {splits}) Path Found:{symbol}")

                # --- STRATEGY 2 & SEQUENTIAL ---
                else:
                    t0 = time.time()
                    path = func(grid, start, goal)
                    t1 = time.time()
                    times.append(t1 - t0)

                    if path:
                        steps_list.append(len(path))
                        success_count += 1
                        symbol = "✓"
                    else:
                        steps_list.append(0)
                        symbol = "✗"

                    print(f"  Trial {t + 1}: {t1 - t0:.5f}s Path Found:{symbol}")

            results[size][name] = {
                'mean': np.mean(times),
                'mean_steps': np.mean(steps_list) if steps_list else 0,
                'success': (success_count / NUM_TRIALS) * 100
            }

    print("\n" + "=" * 140)
    print(f"{'FINAL BENCHMARK REPORT':^140}")
    print("=" * 140)
    print(
        f"{'Grid':<10} | {'Strategy':<30} | {'Time/Path':<15} | {'Speedup':<12} | {'Efficiency':<12} | {'Avg Steps':<10}")
    print("-" * 140)

    for size in GRID_SIZES:
        baseline = results[size]["Sequential"]['mean']
        baseline_steps = results[size]["Sequential"]['mean_steps']

        print(
            f"{size:<10} | {'Sequential (Baseline)':<30} | {baseline:>13.5f}s | {'1.00x':<12} | {'-':<12} | {baseline_steps:>10.1f}")

        for name in strategies.keys():
            if name == "Sequential":
                continue

            res = results[size][name]
            speedup = baseline / res['mean'] if res['mean'] > 0 else 0
            eff = (speedup / NUM_CORES) * 100

            print(
                f"{'':<10} | {name:<30} | {res['mean']:>13.5f}s | {speedup:>11.2f}x | {eff:>11.1f}% | {res['mean_steps']:>10.1f}")

        print("-" * 140)

    generate_benchmark_plots(results, GRID_SIZES)
    plt.show()


if __name__ == "__main__":
    mp.freeze_support()
    run_benchmark()