import time
import numpy as np
import multiprocessing as mp
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


def run_benchmark():
    WIDTH = 130  # Increased width for the new column

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
            steps_list = []  # Store path lengths
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

                    # Calculate average steps for this batch
                    if batch_results:
                        # batch_results is a list of paths (lists)
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

    # ================= REPORT GENERATION =================
    print("\n" + "=" * 130)
    print(f"{'FINAL BENCHMARK REPORT':^130}")
    print("=" * 130)
    # Added "Avg Steps" column
    print(f"{'Grid':<10} | {'Strategy':<30} | {'Time/Path':<15} | {'Speedup':<10} | {'Efficiency':<12} | {'Avg Steps':<10}")
    print("-" * 130)

    for size in GRID_SIZES:
        baseline = results[size]["Sequential"]['mean']
        baseline_steps = results[size]["Sequential"]['mean_steps']

        # 1. Print Baseline
        print(f"{size:<10} | {'Sequential (Baseline)':<30} | {baseline:.5f}s      | {'1.00x':<10} | {'-':<12} | {baseline_steps:.1f}")

        # 2. Print Others
        for name in strategies.keys():
            if name == "Sequential": continue

            res = results[size][name]
            speedup = baseline / res['mean'] if res['mean'] > 0 else 0
            eff = (speedup / NUM_CORES) * 100

            print(f"{'':<10} | {name:<30} | {res['mean']:.5f}s      | {speedup:<9.2f}x | {eff:<9.1f}%   | {res['mean_steps']:.1f}")

        print("-" * 130)

    print("\nNOTE: For 'Batch Processing', Speedup is calculated based on throughput.")
    print(f"      (Sequential Time per Path vs. Parallel Batch Time per Path)")


if __name__ == "__main__":
    mp.freeze_support()
    run_benchmark()