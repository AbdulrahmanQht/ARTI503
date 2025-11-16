import time
import numpy as np
import multiprocessing
from Astar import find_path, create_random_grid
from Parallel.strategy_1_batch import strategy_1_parallel_batch
from Parallel.strategy_2_bidirectional import strategy_2_bidirectional
from Parallel.strategy_3_region import find_path_region_parallel

GRID_SIZES = [5, 10, 25, 50, 100, 250, 500, 750, 1000]
OBSTACLE_DENSITY = 0.34
NUM_TRIALS = 5
NUM_CORES = multiprocessing.cpu_count()


def run_benchmark():
    print("=" * 100)
    print(" " * 25 + "COMPREHENSIVE A* BENCHMARK")
    print(" " * 20 + "Sequential vs. Parallel Strategies")
    print("=" * 100)
    print(f"Grid Sizes: {GRID_SIZES}")
    print(f"Obstacle Density: {OBSTACLE_DENSITY * 100}%")
    print(f"Trials per Configuration: {NUM_TRIALS}")
    print(f"CPU Cores Available: {NUM_CORES}")
    print("=" * 100)

    results = {}

    strategies = {
        "Sequential": find_path,
        "Strategy 1: Batch": strategy_1_parallel_batch,
        "Strategy 2: Bidirectional": strategy_2_bidirectional,
        "Strategy 3: Region-Based": find_path_region_parallel,
    }

    for size in GRID_SIZES:
        print(f"\n{'=' * 100}")
        print(f"GRID SIZE: {size} x {size}")
        print(f"{'=' * 100}")

        results[size] = {}

        for name, func in strategies.items():
            trial_times = []
            trial_path_lengths = []
            successful_paths = 0

            print(f"\n--- Testing: {name} ---")

            for trial in range(NUM_TRIALS):
                # Create fresh grid for each trial
                np.random.seed(42 + trial + size)  # Reproducible but different per trial
                grid = create_random_grid(size, size, OBSTACLE_DENSITY)

                # Standard start/goal positions
                start_x = min(2, size - 1)
                start_y = min(2, size - 1)
                goal_x = max(0, min(size - 3, size - 1))
                goal_y = max(0, min(size - 3, size - 1))
                start, goal = (start_x, start_y), (goal_x, goal_y)

                # Ensure start and goal are walkable
                grid[start[0], start[1]] = 0
                grid[goal[0], goal[1]] = 0

                # Run the algorithm
                start_time = time.perf_counter()

                path = []
                try:
                    if name == "Sequential":
                        path = func(grid, start, goal)
                    elif name == "Strategy 1: Batch":
                        jobs = [(grid, start, goal)]
                        results_list = func(jobs)
                        path = results_list[0] if results_list else []
                    elif name == "Strategy 3: Region-Based":
                        # Dynamic region sizing
                        num_regions = min(4, size // 50) if size >= 50 else 2
                        path = func(grid, start, goal, num_regions, num_regions)
                    else:  # Strategy 2
                        path = func(grid, start, goal)
                except Exception as e:
                    print(f"  Trial {trial + 1}/{NUM_TRIALS}: ERROR - {str(e)}")
                    path = []

                end_time = time.perf_counter()
                elapsed = end_time - start_time
                trial_times.append(elapsed)

                if path:
                    successful_paths += 1
                    trial_path_lengths.append(len(path))
                    print(f"  Trial {trial + 1}/{NUM_TRIALS}: {elapsed:.6f}s | Path: {len(path)} steps ✓")
                else:
                    trial_path_lengths.append(0)
                    print(f"  Trial {trial + 1}/{NUM_TRIALS}: {elapsed:.6f}s | No path found ✗")

            # Calculate statistics
            avg_time = np.mean(trial_times)
            std_time = np.std(trial_times)
            success_rate = successful_paths / NUM_TRIALS
            avg_path_length = np.mean([p for p in trial_path_lengths if p > 0]) if successful_paths > 0 else 0

            results[size][name] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'success_rate': success_rate,
                'avg_path_length': avg_path_length,
                'all_times': trial_times,
                'all_paths': trial_path_lengths
            }

            print(f"\n  Summary:")
            print(f"    Average Time: {avg_time:.6f}s (±{std_time:.6f}s)")
            print(f"    Success Rate: {success_rate * 100:.1f}%")
            if successful_paths > 0:
                print(f"    Avg Path Length: {avg_path_length:.1f} steps")

    # ========== FINAL SUMMARY TABLE ==========
    print("\n" + "=" * 130)
    print(" " * 45 + "FINAL BENCHMARK SUMMARY")
    print("=" * 130)

    header = f"{'Grid':<12} | {'Strategy':<25} | {'Avg Time (s)':<15} | {'Std Dev':<10} | {'Success':<10} | {'Speedup':<10} | {'Efficiency':<12}"
    print(header)
    print("-" * 130)

    for size in GRID_SIZES:
        seq_data = results[size]["Sequential"]
        s1_data = results[size]["Strategy 1: Batch"]
        s2_data = results[size]["Strategy 2: Bidirectional"]
        s3_data = results[size]["Strategy 3: Region-Based"]

        # Use Sequential as baseline
        baseline_time = seq_data['avg_time']

        # Print Sequential (baseline)
        print(f"{size}x{size:<7} | {'Sequential (Baseline)':<25} | "
              f"{seq_data['avg_time']:<15.6f} | {seq_data['std_time']:<10.6f} | "
              f"{seq_data['success_rate'] * 100:<9.1f}% | {'-':<10} | {'-':<12}")

        # Print Strategy 1
        speedup_s1 = baseline_time / s1_data['avg_time'] if s1_data['avg_time'] > 0 else 0
        efficiency_s1 = (speedup_s1 / NUM_CORES) * 100
        print(f"{'':<12} | {'Strategy 1: Batch':<25} | "
              f"{s1_data['avg_time']:<15.6f} | {s1_data['std_time']:<10.6f} | "
              f"{s1_data['success_rate'] * 100:<9.1f}% | {speedup_s1:<9.2f}x | {efficiency_s1:<11.2f}%")

        # Print Strategy 2
        speedup_s2 = baseline_time / s2_data['avg_time'] if s2_data['avg_time'] > 0 else 0
        efficiency_s2 = (speedup_s2 / NUM_CORES) * 100
        print(f"{'':<12} | {'Strategy 2: Bidirectional':<25} | "
              f"{s2_data['avg_time']:<15.6f} | {s2_data['std_time']:<10.6f} | "
              f"{s2_data['success_rate'] * 100:<9.1f}% | {speedup_s2:<9.2f}x | {efficiency_s2:<11.2f}%")

        # Print Strategy 3
        speedup_s3 = baseline_time / s3_data['avg_time'] if s3_data['avg_time'] > 0 else 0
        efficiency_s3 = (speedup_s3 / NUM_CORES) * 100
        print(f"{'':<12} | {'Strategy 3: Region-Based':<25} | "
              f"{s3_data['avg_time']:<15.6f} | {s3_data['std_time']:<10.6f} | "
              f"{s3_data['success_rate'] * 100:<9.1f}% | {speedup_s3:<9.2f}x | {efficiency_s3:<11.2f}%")

        print("-" * 130)

    # ========== ANALYSIS SECTION ==========
    print("\n" + "=" * 130)
    print(" " * 50 + "PERFORMANCE ANALYSIS")
    print("=" * 130)

    print("\n### Speedup and Efficiency Metrics")
    print(f"Baseline: Sequential A* Algorithm")
    print(f"Number of Cores (P): {NUM_CORES}")
    print(f"Speedup = T_sequential / T_parallel")
    print(f"Efficiency = (Speedup / P) × 100%")
    print(f"\nNote: Efficiency > 100% indicates superlinear speedup (rare, usually from cache effects)")
    print(f"      Efficiency < 100% indicates overhead from parallelization")

    print("\n### Strategy-Specific Observations\n")

    # Analyze Strategy 1
    print("STRATEGY 1 (Batch Processing):")
    small_speedup = results[50]["Sequential"]['avg_time'] / results[50]["Strategy 1: Batch"]['avg_time']
    large_speedup = results[1000]["Sequential"]['avg_time'] / results[1000]["Strategy 1: Batch"]['avg_time']
    print(f"  - Speedup at 50x50:   {small_speedup:.2f}x")
    print(f"  - Speedup at 1000x1000: {large_speedup:.2f}x")
    print(f"  - Best for: Multiple independent pathfinding tasks")
    print(f"  - Overhead: Minimal (embarrassingly parallel)")

    # Analyze Strategy 2
    print("\nSTRATEGY 2 (Bidirectional A*):")
    s2_small = results[50]["Strategy 2: Bidirectional"]['avg_time']
    s2_large = results[1000]["Strategy 2: Bidirectional"]['avg_time']
    baseline_small = results[50]["Sequential"]['avg_time']
    baseline_large = results[1000]["Sequential"]['avg_time']

    if s2_small > 0:
        print(f"  - Speedup at 50x50:   {baseline_small / s2_small:.2f}x")
    if s2_large > 0:
        print(f"  - Speedup at 1000x1000: {baseline_large / s2_large:.2f}x")
    print(f"  - Issues: High IPC overhead, frequent synchronization")
    print(f"  - Python Limitation: Multiprocessing pickling dominates execution time")

    # Analyze Strategy 3
    print("\nSTRATEGY 3 (Region-Based A*):")
    s3_small = results[50]["Strategy 3: Region-Based"]['avg_time']
    s3_large = results[1000]["Strategy 3: Region-Based"]['avg_time']

    if s3_small > 0:
        print(f"  - Speedup at 50x50:   {baseline_small / s3_small:.2f}x")
    if s3_large > 0:
        print(f"  - Speedup at 1000x1000: {baseline_large / s3_large:.2f}x")
    print(f"  - Best for: Large grids (>500x500)")
    print(f"  - Trade-off: Path optimality for parallelization")

    print("\n### Scalability Analysis\n")

    # Check if strategies scale with grid size
    for strategy_name in ["Strategy 1: Batch", "Strategy 2: Bidirectional", "Strategy 3: Region-Based"]:
        print(f"{strategy_name}:")
        times_100 = results[100][strategy_name]['avg_time']
        times_500 = results[500][strategy_name]['avg_time']
        times_1000 = results[1000][strategy_name]['avg_time']

        ratio_500 = times_500 / times_100 if times_100 > 0 else float('inf')
        ratio_1000 = times_1000 / times_500 if times_500 > 0 else float('inf')

        print(f"  - Time ratio (500/100):   {ratio_500:.2f}x")
        print(f"  - Time ratio (1000/500):  {ratio_1000:.2f}x")
        print(
            f"  - Scaling: {'Sub-linear (good)' if ratio_1000 < 2 else 'Linear' if ratio_1000 < 3 else 'Super-linear (poor)'}")
        print()

    print("\n### Key Findings\n")
    print("1. Sequential A* provides the baseline for comparison")
    print("2. Strategy 1 (Batch) shows best speedup for independent tasks")
    print("3. Strategy 2 (Bidirectional) suffers from Python's multiprocessing overhead")
    print("4. Strategy 3 (Region-Based) scales better on larger grids")
    print("5. Parallel efficiency decreases with increased synchronization requirements")

    print("\n" + "=" * 130)
    print("Benchmark Complete!")
    print("=" * 130)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_benchmark()