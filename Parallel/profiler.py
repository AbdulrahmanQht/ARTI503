# Parallel_Profiler.py
# Profiles parallel A* strategies to identify bottlenecks
# Measures: IPC overhead, lock contention, worker utilization, and memory usage

import numpy as np
import time
import multiprocessing
from typing import Dict, Tuple, List
import tracemalloc
from collections import defaultdict

from Astar import create_random_grid
from Parallel.strategy_1_batch import strategy_1_parallel_batch, find_path_wrapper
from Parallel.strategy_2_bidirectional import strategy_2_bidirectional
from Parallel.strategy_3_region import find_path_region_parallel


class ParallelProfiler:
    """
    Profiles parallel A* implementations to identify performance bottlenecks.
    """

    def __init__(self):
        self.results = {}

    def profile_strategy(self, strategy_name: str, strategy_func,
                         grid: np.ndarray, start: Tuple[int, int],
                         goal: Tuple[int, int], **kwargs) -> Dict:
        """
        Profile a single parallel strategy.

        Returns:
            Dictionary containing timing, memory, and path information
        """
        print(f"\n{'=' * 80}")
        print(f"PROFILING: {strategy_name}")
        print(f"{'=' * 80}")

        # Start memory tracking
        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        # Time the execution
        start_time = time.perf_counter()
        path = strategy_func(grid, start, goal, **kwargs)
        end_time = time.perf_counter()

        # Get memory usage
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_time = end_time - start_time
        memory_used = (peak_mem - start_mem) / (1024 * 1024)  # Convert to MB

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
        """
        Profile Strategy 1 (Batch Processing) specifically.
        """
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
        seq_results = [find_path_wrapper(j) for j in jobs]
        seq_time = time.perf_counter() - start_time
        seq_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        # Profile parallel batch
        print("  Running Parallel Batch...")
        tracemalloc.start()
        start_time = time.perf_counter()
        para_results = strategy_1_parallel_batch(jobs)
        para_time = time.perf_counter() - start_time
        para_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        speedup = seq_time / para_time if para_time > 0 else 0

        print(f"\n  Sequential Time:  {seq_time:.4f}s, Memory: {seq_mem:.2f} MB")
        print(f"  Parallel Time:    {para_time:.4f}s, Memory: {para_mem:.2f} MB")
        print(f"  Speedup:          {speedup:.2f}x")
        print(f"  ✓ Results Match:  {[len(p) for p in seq_results] == [len(p) for p in para_results]}")

        return {
            'strategy': 'Batch Processing',
            'seq_time': seq_time,
            'para_time': para_time,
            'speedup': speedup,
            'seq_memory_mb': seq_mem,
            'para_memory_mb': para_mem,
            'num_jobs': num_jobs
        }

    def measure_ipc_overhead(self, grid: np.ndarray, start: Tuple[int, int],
                             goal: Tuple[int, int], num_batch_jobs: int = 50) -> Dict:
        """
        Measures Inter-Process Communication overhead for all strategies.
        """
        print(f"\n{'=' * 80}")
        print("MEASURING IPC OVERHEAD")
        print(f"{'=' * 80}")

        # Strategy 1 (Batch)
        print("\n[Strategy 1: Batch Processing]")
        batch_grid = create_random_grid(50, 50, 0.2)
        jobs = []
        for i in range(num_batch_jobs):
            job_start = (np.random.randint(0, 50), np.random.randint(0, 50))
            job_goal = (np.random.randint(0, 50), np.random.randint(0, 50))
            batch_grid[job_start[0], job_start[1]] = 0
            batch_grid[job_goal[0], job_goal[1]] = 0
            jobs.append((batch_grid, job_start, job_goal))

        start_time = time.perf_counter()
        seq_results = [find_path_wrapper(j) for j in jobs]
        seq_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        para_results = strategy_1_parallel_batch(jobs)
        para_time = time.perf_counter() - start_time

        # Strategy 2 (Bidirectional)
        print("\n[Strategy 2: Bidirectional]")
        start_time = time.perf_counter()
        path_bi = strategy_2_bidirectional(grid, start, goal)
        total_time_bi = time.perf_counter() - start_time

        # Strategy 3 (Region-Based)
        print("\n[Strategy 3: Region-Based]")
        grid_size = grid.shape[0]
        num_regions = min(4, grid_size // 50) if grid_size >= 50 else 2
        start_time = time.perf_counter()
        path_reg = find_path_region_parallel(grid, start, goal, num_regions, num_regions)
        total_time_reg = time.perf_counter() - start_time

        result = {
            'batch': {
                'seq_time': seq_time,
                'para_time': para_time,
                'speedup': seq_time / para_time if para_time > 0 else 0,
                'total_paths': sum(len(p) for p in para_results)
            },
            'bidirectional': {
                'total_time': total_time_bi,
                'path_length': len(path_bi) if path_bi else 0
            },
            'region_based': {
                'total_time': total_time_reg,
                'path_length': len(path_reg) if path_reg else 0
            }
        }

        print(f"\n  Batch Sequential Time:    {seq_time:.4f}s")
        print(f"  Batch Parallel Time:      {para_time:.4f}s (Speedup: {result['batch']['speedup']:.2f}x)")
        print(f"  Bidirectional Total Time: {total_time_bi:.4f}s")
        print(f"  Region-Based Total Time:  {total_time_reg:.4f}s")

        return result

    def analyze_scalability(self,
                            size_list: List[int] = None,
                            base_size: int = 100,
                            max_size: int = 500,
                            step: int = 100) -> Dict:
        """
        Tests how strategies scale with increasing grid size or job count.
        Strategy 1 scales with number of jobs (not grid size).
        Strategies 2 & 3 scale with grid size for single-path problems.
        """
        print(f"\n{'=' * 80}")
        print("SCALABILITY ANALYSIS")
        print(f"{'=' * 80}")

        results = defaultdict(list)

        # Test Strategy 1: Scale by number of jobs (fixed grid size)
        print(f"\n--- Strategy 1: Batch Processing (Scaling by Job Count) ---")
        job_counts = [5, 10, 25, 50, 100, 250, 500, 750, 1000]
        batch_grid_size = 50
        batch_grid = create_random_grid(batch_grid_size, batch_grid_size, 0.2)

        for num_jobs in job_counts:
            jobs = []
            for i in range(num_jobs):
                job_start = (np.random.randint(0, batch_grid_size), np.random.randint(0, batch_grid_size))
                job_goal = (np.random.randint(0, batch_grid_size), np.random.randint(0, batch_grid_size))
                batch_grid[job_start[0], job_start[1]] = 0
                batch_grid[job_goal[0], job_goal[1]] = 0
                jobs.append((batch_grid, job_start, job_goal))

            # Sequential
            start_time = time.perf_counter()
            seq_results = [find_path_wrapper(j) for j in jobs]
            time_seq = time.perf_counter() - start_time

            # Parallel
            start_time = time.perf_counter()
            para_results = strategy_1_parallel_batch(jobs)
            time_para = time.perf_counter() - start_time

            speedup = time_seq / time_para if time_para > 0 else 0
            results['batch_sequential'].append((num_jobs, time_seq, sum(len(p) for p in seq_results)))
            results['batch_parallel'].append((num_jobs, time_para, sum(len(p) for p in para_results)))
            results['batch_speedup'].append((num_jobs, speedup))

            print(f"  Jobs: {num_jobs:4d} | Seq: {time_seq:.4f}s | Para: {time_para:.4f}s | Speedup: {speedup:.2f}x")

        # Test Strategies 2 & 3: Scale by grid size (single path)
        print(f"\n--- Strategies 2 & 3: Single Path (Scaling by Grid Size) ---")

        if size_list:
            sizes = size_list
            print(f"  Testing custom grid sizes: {sizes}")
        else:
            sizes = range(base_size, max_size + 1, step)
            print(f"  Testing grid sizes from {base_size} to {max_size}...")

        for size in sizes:
            print(f"\n  Grid Size: {size}x{size}")
            grid = create_random_grid(size, size, 0.3)
            start = (2, 2)
            goal = (size - 3, size - 3)
            grid[start[0], start[1]] = 0
            grid[goal[0], goal[1]] = 0

            # Test Strategy 2
            start_time = time.perf_counter()
            path_bi = strategy_2_bidirectional(grid, start, goal)
            time_bi = time.perf_counter() - start_time
            results['bidirectional'].append((size, time_bi, len(path_bi) if path_bi else 0))
            print(f"    Bidirectional: {time_bi:.4f}s")

            # Test Strategy 3
            num_regions = min(4, size // 50) if size >= 50 else 2
            start_time = time.perf_counter()
            path_reg = find_path_region_parallel(grid, start, goal, num_regions, num_regions)
            time_reg = time.perf_counter() - start_time
            results['region_based'].append((size, time_reg, len(path_reg) if path_reg else 0))
            print(f"    Region-Based:  {time_reg:.4f}s (using {num_regions}x{num_regions} regions)")

        return dict(results)

    def profile_lock_contention(self, grid: np.ndarray,
                                start: Tuple[int, int],
                                goal: Tuple[int, int],
                                num_batch_jobs: int = 50) -> Dict:
        """
        Estimates lock contention by running strategies multiple times
        and measuring variance in execution time.
        Includes all strategies.
        """
        print(f"\n{'=' * 80}")
        print("LOCK CONTENTION ANALYSIS")
        print(f"{'=' * 80}")

        num_runs = 5

        # Strategy 1 (Batch)
        print("\n[Strategy 1: Batch Processing]")
        batch_grid = create_random_grid(50, 50, 0.2)
        jobs = []
        for i in range(num_batch_jobs):
            job_start = (np.random.randint(0, 50), np.random.randint(0, 50))
            job_goal = (np.random.randint(0, 50), np.random.randint(0, 50))
            batch_grid[job_start[0], job_start[1]] = 0
            batch_grid[job_goal[0], job_goal[1]] = 0
            jobs.append((batch_grid, job_start, job_goal))

        times_batch = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            strategy_1_parallel_batch(jobs)
            times_batch.append(time.perf_counter() - start_time)
            print(f"  Run {i + 1}: {times_batch[-1]:.4f}s")

        avg_batch = np.mean(times_batch)
        std_batch = np.std(times_batch)
        variance_batch = (std_batch / avg_batch * 100) if avg_batch > 0 else 0

        # Strategy 2 (Bidirectional)
        print("\n[Strategy 2: Bidirectional]")
        times_bi = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            strategy_2_bidirectional(grid, start, goal)
            times_bi.append(time.perf_counter() - start_time)
            print(f"  Run {i + 1}: {times_bi[-1]:.4f}s")

        avg_bi = np.mean(times_bi)
        std_bi = np.std(times_bi)
        variance_bi = (std_bi / avg_bi * 100) if avg_bi > 0 else 0

        # Strategy 3 (Region-Based)
        print("\n[Strategy 3: Region-Based]")
        grid_size = grid.shape[0]
        num_regions = min(4, grid_size // 50) if grid_size >= 50 else 2
        times_reg = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            find_path_region_parallel(grid, start, goal, num_regions, num_regions)
            times_reg.append(time.perf_counter() - start_time)
            print(f"  Run {i + 1}: {times_reg[-1]:.4f}s")

        avg_reg = np.mean(times_reg)
        std_reg = np.std(times_reg)
        variance_reg = (std_reg / avg_reg * 100) if avg_reg > 0 else 0

        print(f"\n  Batch Processing - Avg: {avg_batch:.4f}s, Std: {std_batch:.4f}s, Variance: {variance_batch:.2f}%")
        print(f"  Bidirectional    - Avg: {avg_bi:.4f}s, Std: {std_bi:.4f}s, Variance: {variance_bi:.2f}%")
        print(f"  Region-Based     - Avg: {avg_reg:.4f}s, Std: {std_reg:.4f}s, Variance: {variance_reg:.2f}%")

        return {
            'batch': {
                'times': times_batch,
                'avg': avg_batch,
                'std': std_batch,
                'variance_pct': variance_batch
            },
            'bidirectional': {
                'times': times_bi,
                'avg': avg_bi,
                'std': std_bi,
                'variance_pct': variance_bi
            },
            'region_based': {
                'times': times_reg,
                'avg': avg_reg,
                'std': std_reg,
                'variance_pct': variance_reg
            }
        }

    def run_comprehensive_profile(self, grid_size: int = 250):
        """
        Runs all profiling tests and generates a comprehensive report.
        """
        print("\n" + "=" * 80)
        print(" " * 15 + "COMPREHENSIVE PARALLEL A* PROFILER")
        print("=" * 80)

        # Setup test grid for single-path strategies
        print(f"\nSetting up {grid_size}x{grid_size} grid with 30% obstacles...")
        grid = create_random_grid(grid_size, grid_size, 0.3)
        start = (2, 2)
        goal = (grid_size - 3, grid_size - 3)
        grid[start[0], start[1]] = 0
        grid[goal[0], goal[1]] = 0

        all_results = {}

        # 1. Basic profiling - Single Path Strategies
        all_results['basic_bi'] = self.profile_strategy(
            "Strategy 2: Bidirectional",
            strategy_2_bidirectional,
            grid, start, goal
        )

        num_regions = min(4, grid_size // 50) if grid_size >= 50 else 2
        all_results['basic_reg'] = self.profile_strategy(
            "Strategy 3: Region-Based",
            find_path_region_parallel,
            grid, start, goal,
            num_regions_x=num_regions,
            num_regions_y=num_regions
        )

        # 1b. Basic profiling - Batch Strategy
        all_results['basic_batch'] = self.profile_batch_strategy(num_jobs=50, grid_size=50)

        # 2. IPC Overhead
        all_results['ipc'] = self.measure_ipc_overhead(grid, start, goal)

        # 3. Lock Contention
        all_results['lock_contention'] = self.profile_lock_contention(grid, start, goal)

        # 4. Generate Summary Report
        self.print_summary_report(all_results)

        return all_results

    def print_summary_report(self, results: Dict):
        """
        Prints a comprehensive summary of all profiling results.
        """
        print("\n" + "=" * 80)
        print(" " * 25 + "SUMMARY REPORT")
        print("=" * 80)

        print("\nEXECUTION TIME COMPARISON")
        print("-" * 80)
        print(f"{'Strategy':<25} | {'Time (s)':<12} | {'Memory (MB)':<12} | {'Notes':<20}")
        print("-" * 80)
        print(f"{'Batch (Parallel)':<25} | {results['basic_batch']['para_time']:<12.4f} | "
              f"{results['basic_batch']['para_memory_mb']:<12.2f} | "
              f"Speedup: {results['basic_batch']['speedup']:.2f}x")
        print(f"{'Bidirectional':<25} | {results['basic_bi']['time']:<12.4f} | "
              f"{results['basic_bi']['memory_mb']:<12.2f} | Single path")
        print(f"{'Region-Based':<25} | {results['basic_reg']['time']:<12.4f} | "
              f"{results['basic_reg']['memory_mb']:<12.2f} | Single path")

        print("\nLOCK CONTENTION (Variance indicates contention)")
        print("-" * 80)
        print(f"{'Strategy':<25} | {'Avg Time (s)':<15} | {'Variance %':<12}")
        print("-" * 80)
        batch_lock = results['lock_contention']['batch']
        bi_lock = results['lock_contention']['bidirectional']
        reg_lock = results['lock_contention']['region_based']
        print(f"{'Batch Processing':<25} | {batch_lock['avg']:<15.4f} | {batch_lock['variance_pct']:<12.2f}")
        print(f"{'Bidirectional':<25} | {bi_lock['avg']:<15.4f} | {bi_lock['variance_pct']:<12.2f}")
        print(f"{'Region-Based':<25} | {reg_lock['avg']:<15.4f} | {reg_lock['variance_pct']:<12.2f}")

        print("\nBOTTLENECK ANALYSIS")
        print("-" * 80)

        # Identify bottlenecks
        if batch_lock['variance_pct'] > 10:
            print("  -  Batch Processing: HIGH lock contention detected (>10% variance)")
        else:
            print("  ✓  Batch Processing: Low lock contention (embarrassingly parallel)")

        if bi_lock['variance_pct'] > 10:
            print("  -  Bidirectional: HIGH lock contention detected (>10% variance)")
        else:
            print("  ✓  Bidirectional: Low lock contention")

        if reg_lock['variance_pct'] > 10:
            print("  -  Region-Based: HIGH lock contention detected (>10% variance)")
        else:
            print("  ✓  Region-Based: Low lock contention")

        # Memory analysis
        if results['basic_batch']['para_memory_mb'] > 100:
            print(f"  ⚠  Batch: High memory usage ({results['basic_batch']['para_memory_mb']:.1f} MB)")

        if results['basic_bi']['memory_mb'] > 100:
            print(f"  ⚠  Bidirectional: High memory usage ({results['basic_bi']['memory_mb']:.1f} MB)")

        if results['basic_reg']['memory_mb'] > 100:
            print(f"  ⚠  Region-Based: High memory usage ({results['basic_reg']['memory_mb']:.1f} MB)")

        print("\nRECOMMENDATIONS")
        print("-" * 80)

        # Strategy-specific recommendations
        if results['basic_batch']['speedup'] > 1.5:
            print(f"  ✓ Strategy 1 (Batch) achieves good speedup ({results['basic_batch']['speedup']:.2f}x)")
            print("    → Best for processing multiple independent paths")
        else:
            print("  ⚠ Strategy 1 (Batch) shows limited speedup")
            print("    → Increase number of jobs or job complexity")

        if results['basic_bi']['time'] > 1.0 or results['basic_reg']['time'] > 1.0:
            print("  ⚠ Single-path strategies: IPC overhead dominates for small grids")
            print("    → Python's multiprocessing overhead is significant")
            print("    → Best performance on large grids (>500x500)")

        if bi_lock['variance_pct'] > 10 or reg_lock['variance_pct'] > 10:
            print("  ⚠ Reduce lock granularity or use lock-free data structures")
            print("    → Consider batching updates to shared state")

        if results['basic_bi']['memory_mb'] > 50 or results['basic_reg']['memory_mb'] > 50:
            print("  ⚠ High memory usage due to Manager() overhead")
            print("    → Consider using shared memory arrays for grid data")

        print("=" * 80)


def run_parallel_profile_mode():
    """
    Main entry point for parallel profiling.
    """
    profiler = ParallelProfiler()

    print("\nSelect profiling mode:")
    print("  1: Comprehensive Profile (All 3 Strategies - Recommended)")
    print("  2: Quick Profile (Single grid size, all strategies)")
    print("  3: Scalability Analysis (Multiple grid sizes & job counts)")
    print("  4: Lock Contention Analysis Only (All strategies)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        profiler.run_comprehensive_profile(grid_size=250)

    elif choice == '2':
        # Single path grid
        grid_size = 200
        grid = create_random_grid(grid_size, grid_size, 0.3)
        start = (2, 2)
        goal = (grid_size - 3, grid_size - 3)
        grid[start[0], start[1]] = 0
        grid[goal[0], goal[1]] = 0

        # Profile all strategies
        profiler.profile_batch_strategy(num_jobs=10, grid_size=50)
        profiler.profile_strategy("Bidirectional", strategy_2_bidirectional, grid, start, goal)

        num_regions = min(4, grid_size // 50) if grid_size >= 50 else 2
        profiler.profile_strategy("Region-Based", find_path_region_parallel, grid, start, goal,
                                  num_regions_x=num_regions, num_regions_y=num_regions)

    elif choice == '3':
        custom_sizes = [5, 10, 25, 50, 100, 250, 500, 750, 1000]
        results = profiler.analyze_scalability(size_list=custom_sizes)

        print("\nSCALABILITY RESULTS:")
        print("-" * 80)

        # Batch results
        if 'batch_speedup' in results:
            print("\nSTRATEGY 1 - BATCH PROCESSING (by job count):")
            print(f"{'Jobs':<10} | {'Sequential':<12} | {'Parallel':<12} | {'Speedup':<10}")
            print("-" * 80)
            for i in range(len(results['batch_speedup'])):
                jobs, speedup = results['batch_speedup'][i]
                _, seq_time, _ = results['batch_sequential'][i]
                _, para_time, _ = results['batch_parallel'][i]
                print(f"{jobs:<10} | {seq_time:<12.4f} | {para_time:<12.4f} | {speedup:<10.2f}x")

        # Single-path results
        print("\nSTRATEGIES 2 & 3 - SINGLE PATH (by grid size):")
        for strategy, data in results.items():
            if strategy not in ['batch_sequential', 'batch_parallel', 'batch_speedup']:
                print(f"\n{strategy.upper()}:")
                print(f"{'Grid Size':<12} | {'Time (s)':<12} | {'Path Length':<12}")
                print("-" * 80)
                for size, time_val, path_len in data:
                    print(f"{size}x{size:<7} | {time_val:<12.4f} | {path_len:<12}")

    elif choice == '4':
        grid_size = 200
        grid = create_random_grid(grid_size, grid_size, 0.3)
        start = (2, 2)
        goal = (grid_size - 3, grid_size - 3)
        grid[start[0], start[1]] = 0
        grid[goal[0], goal[1]] = 0

        profiler.profile_lock_contention(grid, start, goal, num_batch_jobs=50)

    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    # Essential for multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    run_parallel_profile_mode()