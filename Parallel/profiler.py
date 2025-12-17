import numpy as np
import time
import multiprocessing
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import tracemalloc
from collections import defaultdict
from Sequential.Astar import create_random_grid
from Parallel.strategy_1_batch import batch_process_paths, solve_single_path
from Parallel.strategy_2_bidirectional import run_bidirectional_parallel
from Parallel.strategy_3_region import run_region_parallel


class ParallelProfiler:
    """
    Profiles parallel A* implementations to identify performance bottlenecks
    and generates visual comparison charts.
    """

    def __init__(self):
        self.results = {}

    def profile_strategy(self, strategy_name: str, strategy_func,
                         grid: np.ndarray, start: Tuple[int, int],
                         goal: Tuple[int, int], **kwargs) -> Dict:
        """Generic profiler for a single run."""
        print(f"\n{'=' * 80}")
        print(f"PROFILING: {strategy_name}")
        print(f"{'=' * 80}")

        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        start_time = time.perf_counter()
        # Execute Strategy
        path = strategy_func(grid, start, goal, **kwargs)
        end_time = time.perf_counter()

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_time = end_time - start_time
        memory_used = (peak_mem - start_mem) / (1024 * 1024)

        result = {
            'strategy': strategy_name,
            'time': elapsed_time,
            'path_length': len(path) if path else 0,
            'memory_mb': memory_used,
            'found_path': len(path) > 0
        }

        print(f"  ✓ Execution Time:  {elapsed_time:.4f}s")
        print(f"  ✓ Path Length:     {len(path) if path else 'NO PATH'}")
        print(f"  ✓ Memory Used:     {memory_used:.2f} MB")

        return result

    def profile_batch_strategy(self, num_jobs: int = 50, grid_size: int = 50) -> Dict:
        """Profiles Batch Processing: Sequential Loop vs Parallel Pool."""
        print(f"\n{'=' * 80}")
        print("PROFILING: Strategy 1 - Batch Processing")
        print(f"{'=' * 80}")

        # Create batch jobs
        batch_grid = create_random_grid(grid_size, grid_size, 0.2)
        jobs = []
        for i in range(num_jobs):
            job_start = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            job_goal = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
            batch_grid[job_start[0], job_start[1]] = 0
            batch_grid[job_goal[0], job_goal[1]] = 0
            jobs.append((batch_grid, job_start, job_goal))

        # Profile sequential batch
        print(f"\n  Running Sequential Batch ({num_jobs} jobs on {grid_size}x{grid_size} grid)...")
        tracemalloc.start()
        start_time = time.perf_counter()
        seq_results = [solve_single_path(j) for j in jobs]
        seq_time = time.perf_counter() - start_time
        seq_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        # Profile parallel batch
        print("  Running Parallel Batch...")
        tracemalloc.start()
        start_time = time.perf_counter()
        para_results = batch_process_paths(jobs)
        para_time = time.perf_counter() - start_time
        para_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        speedup = seq_time / para_time if para_time > 0 else 0

        print(f"\n  Sequential Time:  {seq_time:.4f}s, Memory: {seq_mem:.2f} MB")
        print(f"  Parallel Time:    {para_time:.4f}s, Memory: {para_mem:.2f} MB")
        print(f"  Speedup:          {speedup:.2f}x")

        return {
            'strategy': 'Batch Processing',
            'seq_time': seq_time,
            'para_time': para_time,
            'speedup': speedup,
            'seq_memory_mb': seq_mem,
            'para_memory_mb': para_mem,
            'num_jobs': num_jobs
        }

    # IPC Overhead & Lock Contention Tests
    def measure_ipc_overhead(self, grid: np.ndarray, start: Tuple[int, int],
                             goal: Tuple[int, int], num_batch_jobs: int = 50) -> Dict:
        """Measures Inter-Process Communication overhead."""
        print(f"\n{'=' * 80}")
        print("MEASURING IPC OVERHEAD")
        print(f"{'=' * 80}")

        print("\n[Strategy 1: Batch Processing]")
        # Use small grid to emphasize overhead
        batch_grid = create_random_grid(50, 50, 0.2)
        jobs = []
        for i in range(num_batch_jobs):
            job_start = (np.random.randint(0, 50), np.random.randint(0, 50))
            job_goal = (np.random.randint(0, 50), np.random.randint(0, 50))
            batch_grid[job_start[0], job_start[1]] = 0
            batch_grid[job_goal[0], job_goal[1]] = 0
            jobs.append((batch_grid, job_start, job_goal))

        start_time = time.perf_counter()
        seq_results = [solve_single_path(j) for j in jobs]
        seq_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        para_results = batch_process_paths(jobs)
        para_time = time.perf_counter() - start_time

        print("\n[Strategy 2: Bidirectional]")
        start_time = time.perf_counter()
        path_bi = run_bidirectional_parallel(grid, start, goal)
        total_time_bi = time.perf_counter() - start_time

        print("\n[Strategy 3: Region-Based]")
        grid_size = grid.shape[0]
        splits = max(2, grid_size // 50)
        start_time = time.perf_counter()
        path_reg = run_region_parallel(grid, start, goal, splits=splits)
        total_time_reg = time.perf_counter() - start_time

        result = {
            'batch': {
                'seq_time': seq_time,
                'para_time': para_time,
                'speedup': seq_time / para_time if para_time > 0 else 0,
            },
            'bidirectional': total_time_bi,
            'region_based': total_time_reg
        }

        print(f"\n  Batch Sequential Time:    {seq_time:.4f}s")
        print(f"  Batch Parallel Time:      {para_time:.4f}s (Speedup: {result['batch']['speedup']:.2f}x)")
        print(f"  Bidirectional Total Time: {total_time_bi:.4f}s")
        print(f"  Region-Based Total Time:  {total_time_reg:.4f}s")

        return result

    def profile_lock_contention(self, grid: np.ndarray,
                                start: Tuple[int, int],
                                goal: Tuple[int, int],
                                num_batch_jobs: int = 50) -> Dict:
        """Estimates lock contention (Variance)."""
        print(f"\n{'=' * 80}")
        print("LOCK CONTENTION ANALYSIS")
        print(f"{'=' * 80}")

        num_runs = 5

        # Strategy 1 (Batch)
        print("\n[Strategy 1: Batch Processing]")
        batch_grid = create_random_grid(100, 100, 0.2)
        jobs = []
        for i in range(num_batch_jobs):
            job_start = (0, 0)
            job_goal = (99, 99)
            batch_grid[job_start] = 0
            batch_grid[job_goal] = 0
            jobs.append((batch_grid, job_start, job_goal))

        times_batch = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            batch_process_paths(jobs)
            times_batch.append(time.perf_counter() - start_time)
            print(f"  Run {i + 1}: {times_batch[-1]:.4f}s")

        avg_batch = np.mean(times_batch)
        std_batch = np.std(times_batch)
        variance_batch = (std_batch / avg_batch * 100) if avg_batch > 0 else 0

        # Strategy 2
        print("\n[Strategy 2: Bidirectional]")
        times_bi = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            run_bidirectional_parallel(grid, start, goal)
            times_bi.append(time.perf_counter() - start_time)
            print(f"  Run {i + 1}: {times_bi[-1]:.4f}s")

        avg_bi = np.mean(times_bi)
        std_bi = np.std(times_bi)
        variance_bi = (std_bi / avg_bi * 100) if avg_bi > 0 else 0

        # Strategy 3
        print("\n[Strategy 3: Region-Based]")
        grid_size = grid.shape[0]
        splits = min(50, max(2, grid_size // 150))
        times_reg = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            run_region_parallel(grid, start, goal, splits=splits)
            times_reg.append(time.perf_counter() - start_time)
            print(f"  Run {i + 1}: {times_reg[-1]:.4f}s")

        avg_reg = np.mean(times_reg)
        std_reg = np.std(times_reg)
        variance_reg = (std_reg / avg_reg * 100) if avg_reg > 0 else 0

        print(f"\n  Batch Processing - Avg: {avg_batch:.4f}s, Var: {variance_batch:.2f}%")
        print(f"  Bidirectional    - Avg: {avg_bi:.4f}s, Var: {variance_bi:.2f}%")
        print(f"  Region-Based     - Avg: {avg_reg:.4f}s, Var: {variance_reg:.2f}%")

        return {
            'batch': {'avg': avg_batch, 'variance_pct': variance_batch},
            'bidirectional': {'avg': avg_bi, 'variance_pct': variance_bi},
            'region_based': {'avg': avg_reg, 'variance_pct': variance_reg}
        }

    # Plotting Functions
    def plot_profile_comparison(self, results: Dict):
        """Generates bar charts comparing Time and Memory across strategies."""
        strategies = ['Batch (Para)', 'Bidirectional', 'Region-Based']
        times = [results['batch']['para_time'], results['bi']['time'], results['reg']['time']]
        memory = [results['batch']['para_memory_mb'], results['bi']['memory_mb'], results['reg']['memory_mb']]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Execution Time Chart
        ax1.bar(strategies, times, color=['#4CAF50', '#2196F3', '#FFC107'])
        ax1.set_title('Execution Time Comparison (Lower is Better)')
        ax1.set_ylabel('Time (seconds)')
        for i, v in enumerate(times):
            ax1.text(i, v, f"{v:.4f}s", ha='center', va='bottom', fontweight='bold')

        # 2. Memory Usage Chart
        ax2.bar(strategies, memory, color=['#8BC34A', '#03A9F4', '#FF9800'])
        ax2.set_title('Peak Memory Usage (Lower is Better)')
        ax2.set_ylabel('Memory (MB)')
        for i, v in enumerate(memory):
            ax2.text(i, v, f"{v:.1f} MB", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def plot_scalability_results(self, results: Dict):
        """Generates grouped bar charts for Scalability Analysis."""
        sizes = results['grid_sizes']
        bi_times = results['bidirectional_times']
        reg_times = results['region_times']
        batch_jobs = [x[0] for x in results['batch_speedup']]
        batch_speedups = [x[1] for x in results['batch_speedup']]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Chart 1: Grid Scalability
        x = np.arange(len(sizes))
        width = 0.35
        ax1.bar(x - width / 2, bi_times, width, label='Bidirectional', color='#2196F3')
        ax1.bar(x + width / 2, reg_times, width, label='Region-Based', color='#FFC107')
        ax1.set_xlabel('Grid Size (NxN)')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Scalability: Execution Time vs Grid Size')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sizes)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Chart 2: Batch Speedup
        ax2.plot(batch_jobs, batch_speedups, marker='o', linestyle='-', color='#4CAF50', linewidth=2)
        ax2.bar(batch_jobs, batch_speedups, color='#4CAF50', alpha=0.3, width=20)
        ax2.set_xlabel('Number of Jobs')
        ax2.set_ylabel('Speedup Factor (x)')
        ax2.set_title('Batch Processing Speedup vs Workload')
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    # Scalability Analysis
    def analyze_scalability(self, size_list: List[int] = None) -> Dict:
        """Tests scalability and gathers data for plotting."""
        print(f"\n{'=' * 80}")
        print("SCALABILITY ANALYSIS")
        print(f"{'=' * 80}")

        results = defaultdict(list)

        # TEST 1: Batch Scalability
        print(f"\n--- Strategy 1: Batch Processing (Scaling by Job Count) ---")
        job_counts = [50, 100, 200, 500]
        batch_grid_size = 300
        batch_grid = create_random_grid(batch_grid_size, batch_grid_size, 0.2)

        for num_jobs in job_counts:
            jobs = []
            for i in range(num_jobs):
                job_start = (0, 0)
                job_goal = (batch_grid_size - 1, batch_grid_size - 1)
                batch_grid[job_start] = 0
                batch_grid[job_goal] = 0
                jobs.append((batch_grid, job_start, job_goal))

            start_time = time.perf_counter()
            [solve_single_path(j) for j in jobs]
            time_seq = time.perf_counter() - start_time

            start_time = time.perf_counter()
            batch_process_paths(jobs)
            time_para = time.perf_counter() - start_time

            speedup = time_seq / time_para if time_para > 0 else 0
            results['batch_speedup'].append((num_jobs, speedup))
            print(f"  Jobs: {num_jobs:4d} | Speedup: {speedup:.2f}x")

        # TEST 2: Single Path Scalability
        print(f"\n--- Strategies 2 & 3: Single Path (Scaling by Grid Size) ---")
        if size_list:
            sizes = size_list
        else:
            sizes = [500, 1000, 2500, 5000]

        results['grid_sizes'] = sizes

        for size in sizes:
            print(f"\n  Grid Size: {size}x{size}")
            grid = create_random_grid(size, size, 0.25)
            start, goal = (0, 0), (size - 1, size - 1)
            grid[start], grid[goal] = 0, 0

            # Test Strategy 2
            start_time = time.perf_counter()
            run_bidirectional_parallel(grid, start, goal)
            time_bi = time.perf_counter() - start_time
            results['bidirectional_times'].append(time_bi)
            print(f"    Bidirectional: {time_bi:.4f}s")

            # Test Strategy 3
            raw_splits = size // 150
            splits = min(50, max(2, raw_splits))
            start_time = time.perf_counter()
            run_region_parallel(grid, start, goal, splits=splits)
            time_reg = time.perf_counter() - start_time
            results['region_times'].append(time_reg)
            print(f"    Region-Based:  {time_reg:.4f}s")

        self.plot_scalability_results(dict(results))
        return dict(results)

    def run_comprehensive_profile(self, grid_size: int = 500):
        print("\n" + "=" * 80)
        print(" " * 15 + "COMPREHENSIVE PARALLEL A* PROFILER")
        print("=" * 80)

        grid = create_random_grid(grid_size, grid_size, 0.3)
        start, goal = (0, 0), (grid_size - 1, grid_size - 1)
        grid[start], grid[goal] = 0, 0

        all_results = {}
        all_results['bi'] = self.profile_strategy("Strategy 2: Bidirectional", run_bidirectional_parallel, grid, start,
                                                  goal)

        splits = min(50, max(2, grid_size // 50))
        all_results['reg'] = self.profile_strategy("Strategy 3: Region-Based", run_region_parallel, grid, start, goal,
                                                   splits=splits)

        all_results['batch'] = self.profile_batch_strategy(num_jobs=32, grid_size=300)

        self.measure_ipc_overhead(grid, start, goal)
        self.profile_lock_contention(grid, start, goal)

        self.print_summary_report(all_results)
        self.plot_profile_comparison(all_results)
        return all_results

    def print_summary_report(self, results: Dict):
        print("\n" + "=" * 80)
        print(" " * 25 + "SUMMARY REPORT")
        print("=" * 80)
        print(f"{'Strategy':<25} | {'Time (s)':<12} | {'Memory (MB)':<12} | {'Speedup/Note':<20}")
        print("-" * 80)
        print(f"{'Batch':<25} | {results['batch']['para_time']:<12.4f} | "
              f"{results['batch']['para_memory_mb']:<12.2f} | "
              f"{results['batch']['speedup']:.2f}x Speedup")
        print(f"{'Bidirectional':<25} | {results['bi']['time']:<12.4f} | "
              f"{results['bi']['memory_mb']:<12.2f} | Single path")
        print(f"{'Region-Based':<25} | {results['reg']['time']:<12.4f} | "
              f"{results['reg']['memory_mb']:<12.2f} | Single path")
        print("=" * 80)


import numpy as np
import time
import multiprocessing
import matplotlib.pyplot as plt
from Sequential.Astar import create_random_grid
from Parallel.strategy_1_batch import batch_process_paths
from Parallel.strategy_2_bidirectional import run_bidirectional_parallel
from Parallel.strategy_3_region import run_region_parallel


def benchmark_parallel_scaling():
    # Configuration
    grid_size = 5000
    obstacle_density = 0.34
    num_trials = 5
    core_configs = [1, 2, 4, 8, 12]

    print(
        f"Starting Strong Scaling Benchmark: {grid_size}x{grid_size} | {num_trials} Trials | {obstacle_density * 100}% Density")

    # Static grid for all tests to ensure fairness
    grid = create_random_grid(grid_size, grid_size, obstacle_density)
    start, goal = (0, 0), (grid_size - 1, grid_size - 1)
    grid[start], grid[goal] = 0, 0

    results = {
        'cores': core_configs,
        'Strategy 1: Batch': [],
        'Strategy 2: Bidirectional': [],
        'Strategy 3: Region': []
    }

    for p in core_configs:
        print(f"\n--- Testing with {p} Cores ---")

        trial_times_s1 = []
        trial_times_s2 = []
        trial_times_s3 = []

        for t in range(num_trials):
            print(f"  Trial {t + 1}/{num_trials}...", end="\r")

            # Strategy 1: Batch (Average of 4 paths)
            t_start = time.perf_counter()
            jobs = [(grid, start, goal)] * 4
            # Note: Ensure your batch_process_paths is modified to accept a 'p' parameter if needed
            batch_process_paths(jobs)
            trial_times_s1.append((time.perf_counter() - t_start) / 4)

            # Strategy 2: Bidirectional
            t_start = time.perf_counter()
            run_bidirectional_parallel(grid, start, goal)
            trial_times_s2.append(time.perf_counter() - t_start)

            # Strategy 3: Region
            t_start = time.perf_counter()
            # Splits often scale with core count for better efficiency
            run_region_parallel(grid, start, goal, splits=max(2, p))
            trial_times_s3.append(time.perf_counter() - t_start)

        # Record averages for this core count
        results['Strategy 1: Batch'].append(np.mean(trial_times_s1))
        results['Strategy 2: Bidirectional'].append(np.mean(trial_times_s2))
        results['Strategy 3: Region'].append(np.mean(trial_times_s3))

        print(
            f"  Done. Avg Times -> S1: {results['Strategy 1: Batch'][-1]:.2f}s | S2: {results['Strategy 2: Bidirectional'][-1]:.2f}s | S3: {results['Strategy 3: Region'][-1]:.2f}s")

    # Plotting

    plt.figure(figsize=(10, 6))
    for strategy in ['Strategy 1: Batch', 'Strategy 2: Bidirectional', 'Strategy 3: Region']:
        plt.plot(core_configs, results[strategy], marker='o', linewidth=2, label=strategy)

    plt.title(f'Strong Scaling: Execution Time vs CPU Cores (Fixed {grid_size}x{grid_size})')
    plt.xlabel('Number of CPU Cores')
    plt.ylabel('Average Execution Time (Seconds)')
    plt.xticks(core_configs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def run_parallel_profile_mode():
    profiler = ParallelProfiler()
    print("\nSelect profiling mode:")
    print("  1: Comprehensive Profile")
    print("  2: Quick Profile (Single grid size)")
    print("  3: Scalability Analysis With Problem Size")
    print("  4: Scalability Analysis With CPU Cores")
    print("  5: Lock Contention Analysis")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == '1':
        profiler.run_comprehensive_profile(grid_size=500)
    elif choice == '2':
        grid_size = 500
        grid = create_random_grid(grid_size, grid_size, 0.3)
        start, goal = (0, 0), (grid_size - 1, grid_size - 1)
        grid[start] = 0
        grid[goal] = 0

        results = {}
        # Run Batch
        results['batch'] = profiler.profile_batch_strategy(num_jobs=20, grid_size=300)
        # Run Bidirectional
        results['bi'] = profiler.profile_strategy("Strategy 2: Bidirectional", run_bidirectional_parallel, grid, start,
                                                  goal)
        # Run Region
        results['reg'] = profiler.profile_strategy("Strategy 3: Region-Based", run_region_parallel, grid, start, goal,
                                                   splits=10)

        profiler.print_summary_report(results)
        profiler.plot_profile_comparison(results)  # Generates charts for Quick Profile too!

    elif choice == '3':
        custom_sizes = [500, 1000, 2500, 5000]
        profiler.analyze_scalability(size_list=custom_sizes)
    elif choice == '4':
        benchmark_parallel_scaling()
    elif choice == '5':
        grid_size = 300
        grid = create_random_grid(grid_size, grid_size, 0.3)
        start, goal = (0, 0), (grid_size - 1, grid_size - 1)
        grid[start] = 0
        grid[goal] = 0
        profiler.profile_lock_contention(grid, start, goal, num_batch_jobs=50)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    run_parallel_profile_mode()