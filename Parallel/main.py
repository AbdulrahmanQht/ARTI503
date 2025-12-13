import numpy as np
import time
import cProfile
import pstats
import io
import multiprocessing
from Sequential.Astar import (
    visualize_path,
    create_empty_grid,
    create_random_grid,
    add_wall,
    print_grid_info,
    find_path  # Import sequential baseline
)
from Parallel.strategy_1_batch import batch_process_paths
from Parallel.strategy_2_bidirectional import run_bidirectional_parallel
from Parallel.strategy_3_region import run_region_parallel


# HELPER FUNCTIONS
def print_header():
    print("=" * 80)
    print(" " * 20 + "A* PATHFINDING ALGORITHM")
    print(" " * 15 + "Parallel Implementation - 9MS2")
    print(" " * 18 + "ARTI503 - Group 2 - 2025")
    print("=" * 80)


def get_grid_size():
    print("\n GRID SIZE")
    print("-" * 40)
    while True:
        try:
            size = int(input("Enter grid size (e.g., 20 for 20x20 grid): "))
            if size > 1 and size <= 5000:  # Increased limit for parallel testing
                return size
            else:
                print(" Please enter a number > 1 and <= 5000")
        except ValueError:
            print(" Invalid input! Please enter a number.")


def get_grid_type():
    print("\n GRID TYPE")
    print("-" * 40)
    print("1. Empty grid (no obstacles)")
    print("2. Random obstacles")
    print("3. Custom grid with walls")
    while True:
        try:
            choice = int(input("Select grid type (1-3): "))
            if choice in [1, 2, 3]:
                return choice
            else:
                print(" Please enter 1, 2, or 3")
        except ValueError:
            print(" Invalid input! Please enter a number.")


def get_obstacle_density():
    print("\n OBSTACLE DENSITY")
    print("-" * 40)
    while True:
        try:
            density = float(input("Enter obstacle density (0.0 to 1.0, e.g., 0.2 for 20%): "))
            if 0.0 <= density <= 1.0:
                return density
            else:
                print(" Please enter a value between 0.0 and 1.0")
        except ValueError:
            print(" Invalid input! Please enter a decimal number.")


def get_position(prompt: str, grid_size: int):
    while True:
        try:
            pos_input = input(prompt)
            x, y = map(int, pos_input.split(','))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                return (x, y)
            else:
                print(f" Position must be within grid bounds (0 to {grid_size - 1})")
        except (ValueError, TypeError):
            print(" Invalid format! Please enter as: x,y (e.g., 5,10)")


def create_custom_grid(size: int):
    grid = create_empty_grid(size, size)
    print("\n CUSTOM WALLS")
    print("-" * 40)
    print("Add walls to your grid:")
    if size > 15:
        grid = add_wall(grid, (10, 5), (10, 15), 'vertical')
        grid = add_wall(grid, (5, 10), (15, 10), 'horizontal')
        print(" Sample walls added.")
    return grid


def get_strategy_choice():
    print("\nCHOOSE PARALLEL STRATEGY")
    print("-" * 40)
    print("1. Strategy 1: Batch Processing (Task-Parallel)")
    print("2. Strategy 2: Bidirectional A* (Shared Memory/Hybrid)")
    print("3. Strategy 3: Region-Based A* (Decomposition)")

    while True:
        try:
            strategy_choice = int(input("\nSelect strategy (1-3): "))
            if strategy_choice in [1, 2, 3]:
                return strategy_choice
            else:
                print("Please enter 1, 2, or 3")
        except ValueError:
            print("Invalid input! Please enter a number.")


def get_benchmark_choice():
    print("\nCHOOSE BENCHMARK MODE")
    print("-" * 40)
    print("1. Benchmark Strategy 1: Batch Processing")
    print("2. Benchmark Strategy 2: Bidirectional A*")
    print("3. Benchmark Strategy 3: Region-Based A*")
    print("4. Run All Strategies (Comparison)")

    while True:
        try:
            strategy_choice = int(input("\nSelect benchmark (1-4): "))
            if strategy_choice in [1, 2, 3, 4]:
                return strategy_choice
            else:
                print("Please enter 1, 2, 3, or 4")
        except ValueError:
            print("Invalid input! Please enter a number.")


def run_parallel_pathfinding(grid: np.ndarray, start: tuple, goal: tuple, strategy: int):
    """Run A* pathfinding using the chosen parallel strategy."""

    strategy_names = {
        1: "Strategy 1: Batch Processing",
        2: "Strategy 2: Bidirectional A*",
        3: "Strategy 3: Region-Based A*"
    }
    strategy_name = strategy_names.get(strategy, "Unknown Strategy")

    print(f"\n RUNNING A* ALGORITHM ({strategy_name})...")
    print("-" * 40)

    # Run algorithm with timing
    start_time = time.perf_counter()

    path = []
    if strategy == 1:
        print(" (Note: Batch Strategy is optimized for multiple paths, overhead will be high for just 1)")
        jobs = [(grid, start, goal)]
        results = batch_process_paths(jobs)  # UPDATED CALL
        path = results[0] if results else []
    elif strategy == 2:
        path = run_bidirectional_parallel(grid, start, goal)  # UPDATED CALL
    elif strategy == 3:
        # Use simple heuristic for splits based on grid size
        splits = max(2, grid.shape[0] // 50)
        path = run_region_parallel(grid, start, goal, splits=splits)  # UPDATED CALL

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    print("\n RESULTS:")
    print("-" * 40)
    if path:
        print(f" Path found!")
        print(f"  Path length: {len(path)} steps")
        print(f"  Execution time: {execution_time:.6f} seconds")
        print(f"  Start position: {start}")
        print(f"  Goal position: {goal}")

        print("\n Displaying visualization...")
        title = f"{strategy_name}\n{len(path)} steps in {execution_time:.4f}s"
        visualize_path(grid, path, start, goal, title)
    else:
        print("x No path found!")
        print(f"  Execution time: {execution_time:.6f} seconds")
        visualize_path(grid, path, start, goal, f"{strategy_name}: No path found")


def run_comparison_benchmark():
    print("\n BENCHMARK MODE: COMPARING ALL PARALLEL STRATEGIES")
    print("=" * 80)

    # --- CONFIGURE BENCHMARK ---
    grid_sizes = [50, 100, 250, 500,1000,2500,5000]
    obstacle_density = 0.34
    num_trials = 3

    print(f"Testing grid sizes: {grid_sizes}")
    print(f"Obstacle density: {obstacle_density * 100}%")
    print(f"Trials per size: {num_trials}\n")

    benchmark_results = {size: {} for size in grid_sizes}

    # --- RUN BENCHMARK LOOP ---
    for size in grid_sizes:
        print(f"\n{'=' * 80}")
        print(f"Grid Size: {size} x {size}")
        print(f"{'=' * 80}")

        strategies = [
            (1, "Batch Processing", batch_process_paths),
            (2, "Bidirectional", run_bidirectional_parallel),
            (3, "Region-Based", run_region_parallel)
        ]

        for strat_id, strat_name, strat_func in strategies:
            print(f"\n--- Testing: {strat_name} ---")

            trial_times = []
            successful_paths = 0

            for trial in range(num_trials):
                grid = create_random_grid(size, size, obstacle_density)
                s, g = (0, 0), (size - 1, size - 1)
                grid[s], grid[g] = 0, 0  # Ensure clear

                start_time = time.perf_counter()

                path = []
                try:
                    if strat_id == 1:
                        # Testing single path latency
                        jobs = [(grid, s, g)]
                        res = strat_func(jobs)
                        path = res[0] if res else []
                    elif strat_id == 3:
                        splits = max(2, size // 100)
                        path = strat_func(grid, s, g, splits=splits)
                    else:
                        path = strat_func(grid, s, g)
                except Exception as e:
                    print(f"Error: {e}")

                elapsed = time.perf_counter() - start_time
                trial_times.append(elapsed)

                if path:
                    successful_paths += 1
                    print(f"Trial {trial + 1}: {elapsed:.4f}s | Path: {len(path)}")
                else:
                    print(f"Trial {trial + 1}: {elapsed:.4f}s | No path")

            avg_time = np.mean(trial_times)
            success_rate = successful_paths / num_trials

            benchmark_results[size][strat_id] = {
                'name': strat_name,
                'avg_time': avg_time,
                'success_rate': success_rate
            }

    print("\n" + "=" * 80)
    print(f"FINAL BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"{'Grid Size':<10} | {'Strategy':<20} | {'Avg Time (s)':<15} | {'Success':<10}")
    print("-" * 65)

    for size in grid_sizes:
        for strat_id in [1, 2, 3]:
            res = benchmark_results[size][strat_id]
            print(f"{size:<10} | {res['name']:<20} | {res['avg_time']:<15.4f} | {res['success_rate'] * 100:.0f}%")


def run_benchmark_mode():
    """Asks the user which benchmark to run, then executes it."""
    benchmark_choice = get_benchmark_choice()

    if benchmark_choice == 4:
        run_comparison_benchmark()
        return

    # Map choice to function and name
    config = {
        1: (batch_process_paths, "Batch Processing"),
        2: (run_bidirectional_parallel, "Bidirectional A*"),
        3: (run_region_parallel, "Region-Based A*")
    }

    strategy_func, strategy_name = config[benchmark_choice]

    print(f"\n BENCHMARK MODE: {strategy_name.upper()}")
    print("=" * 80)

    grid_sizes = [50, 100, 250, 500, 1000]
    num_trials = 3
    results = []

    for size in grid_sizes:
        print(f"\nTesting Grid Size: {size}x{size}")
        times = []
        success = 0

        for i in range(num_trials):
            grid = create_random_grid(size, size, 0.3)
            s, g = (0, 0), (size - 1, size - 1)
            grid[s], grid[g] = 0, 0

            t0 = time.perf_counter()
            path = []
            if benchmark_choice == 1:
                # Batch of 10 for better profiling
                jobs = [(grid, s, g) for _ in range(10)]
                res = strategy_func(jobs)
                path = res[0] if res else []
            elif benchmark_choice == 3:
                path = strategy_func(grid, s, g, splits=max(2, size // 100))
            else:
                path = strategy_func(grid, s, g)
            t1 = time.perf_counter()

            times.append(t1 - t0)
            if path: success += 1
            print(f" Trial {i + 1}: {t1 - t0:.4f}s")

        results.append({
            'size': size,
            'avg': np.mean(times),
            'success': success / num_trials
        })

    print("\nSUMMARY")
    print(f"{'Size':<10} {'Avg Time':<15} {'Success':<10}")
    for r in results:
        print(f"{r['size']:<10} {r['avg']:<15.4f} {r['success'] * 100:.0f}%")


def main():
    print_header()
    print("\n SELECT MODE")
    print("-" * 40)
    print("1. Interactive Mode (Visualize a single path)")
    print("2. Benchmark Mode (Test strategies)")

    while True:
        try:
            mode = int(input("\nSelect mode (1-2): "))
            if mode in [1, 2]:
                break
        except ValueError:
            pass

    if mode == 2:
        run_benchmark_mode()
        return

    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - PARALLEL PATHFINDING")
    print("=" * 80)

    grid_size = get_grid_size()
    grid_type = get_grid_type()

    if grid_type == 1:
        grid = create_empty_grid(grid_size, grid_size)
    elif grid_type == 2:
        density = get_obstacle_density()
        grid = create_random_grid(grid_size, grid_size, density)
    else:
        grid = create_custom_grid(grid_size)

    start = get_position(f"Enter START (0-{grid_size - 1}): ", grid_size)
    goal = get_position(f"Enter GOAL (0-{grid_size - 1}): ", grid_size)

    # Clean start/goal
    grid[start], grid[goal] = 0, 0

    strategy_choice = get_strategy_choice()
    run_parallel_pathfinding(grid, start, goal, strategy_choice)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()