import numpy as np
import time
import cProfile
import pstats
import io
import matplotlib.pyplot as plt
import multiprocessing

# 1. IMPORT Astar UTILITIES
# (Importing from 'Astar.py' as in your file)
from Astar import (
    find_path,
    visualize_path,
    create_empty_grid,
    create_random_grid,
    add_wall,
    print_grid_info
)

# 2. IMPORT YOUR PARALLEL STRATEGIES
from Parallel.strategy_1_batch import strategy_1_parallel_batch
from Parallel.strategy_2_bidirectional import strategy_2_bidirectional
from Parallel.strategy_3_region import strategy_3_region_based


# 3. HELPER FUNCTIONS (Copied from your sequential main.py)

def print_header():
    print("=" * 80)
    print(" " * 20 + "A* PATHFINDING ALGORITHM")
    print(" " * 15 + "Parallel Implementation - 9MS2")  # <-- Changed
    print(" " * 18 + "ARTI503 - Group 2 - 2025")
    print("=" * 80)


def get_grid_size():
    """Get grid size from user."""
    print("\n GRID SIZE")
    print("-" * 40)
    while True:
        try:
            # Use a larger max size for parallel
            size = int(input("Enter grid size (e.g., 20 for 20x20 grid): "))
            if size > 1 and size <= 3000:
                return size
            else:
                print(" Please enter a number > 1 and <= 3000")
        except ValueError:
            print(" Invalid input! Please enter a number.")


def get_grid_type():
    """Get grid type from user."""
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
    """Get obstacle density from user."""
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
    """Get a position from user."""
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
    """Create a custom grid with walls."""
    grid = create_empty_grid(size, size)
    print("\n CUSTOM WALLS")
    print("-" * 40)
    print("Add walls to your grid:")
    if size > 15:
        grid = add_wall(grid, (10, 5), (10, 15), 'vertical')
        grid = add_wall(grid, (5, 10), (15, 10), 'horizontal')
        print(" Sample walls added.")
    return grid


# 4. MODIFIED FUNCTIONS FOR PARALLEL STRATEGIES

def get_strategy_choice():
    """Asks the user to select a parallel strategy."""
    print("\nCHOOSE PARALLEL STRATEGY")
    print("-" * 40)
    print("1. Strategy 1: Batch Processing (Task-Parallel)")
    print("2. Strategy 2: Bidirectional A* (Shared-State)")
    print("3. Strategy 3: Region-Based A* (Message-Passing)")

    while True:
        try:
            strategy_choice = int(input("\nSelect strategy (1-3): "))
            if strategy_choice in [1, 2, 3]:
                return strategy_choice
            else:
                print("Please enter 1, 2, or 3")
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
        # Strategy 1 expects a list of jobs
        jobs = [(grid, start, goal)]
        results = strategy_1_parallel_batch(jobs)
        path = results[0] if results else []
    elif strategy == 2:
        path = strategy_2_bidirectional(grid, start, goal)
    elif strategy == 3:
        path = strategy_3_region_based(grid, start, goal)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Display results
    print("\n RESULTS:")
    print("-" * 40)

    if path:
        print(f" Path found!")
        print(f"  Path length: {len(path)} steps")
        print(f"  Execution time: {execution_time:.6f} seconds")
        print(f"  Start position: {start}")
        print(f"  Goal position: {goal}")
        print(f"\n  First 5 steps: {path[:5]}")
        if len(path) > 5:
            print(f"  Last 5 steps: {path[-5:]}")

        # Visualize
        print("\n Displaying visualization...")
        title = f"{strategy_name}\n{len(path)} steps in {execution_time:.4f}s"
        visualize_path(grid, path, start, goal, title)
    else:
        print("✗ No path found!")
        print(f"  Execution time: {execution_time:.6f} seconds")
        visualize_path(grid, path, start, goal, f"{strategy_name}: No path found")


def run_benchmark_mode():
    """
    Runs a benchmark on a *single chosen parallel strategy*
    over multiple grid sizes, just like the sequential benchmark.
    """

    # --- 1. CHOOSE STRATEGY ---
    strategy_choice = get_strategy_choice()

    strategy_funcs = {
        1: strategy_1_parallel_batch,
        2: strategy_2_bidirectional,
        3: strategy_3_region_based
    }
    strategy_names = {
        1: "Strategy 1: Batch Processing",
        2: "Strategy 2: Bidirectional A*",
        3: "Strategy 3: Region-Based A*"
    }

    strategy_func = strategy_funcs[strategy_choice]
    strategy_name = strategy_names[strategy_choice]

    print(f"\n BENCHMARK MODE: {strategy_name.upper()}")
    print("=" * 80)

    # --- 2. CONFIGURE BENCHMARK ---
    grid_sizes = [5, 25, 50, 100, 250, 500, 750, 1000, 2500]
    obstacle_density = 0.34
    num_trials = 5

    print(f"Testing grid sizes: {grid_sizes}")
    print(f"Obstacle density: {obstacle_density * 100}%")
    print(f"Trials per size: {num_trials}\n")

    results = []
    profiler = cProfile.Profile()

    # --- 3. RUN BENCHMARK LOOP ---
    for size in grid_sizes:
        print(f"\n{'=' * 80}")
        print(f"Grid Size: {size} x {size}")
        print(f"{'=' * 80}")

        trial_times = []
        successful_paths = 0

        for trial in range(num_trials):
            # Using the same (2,2) and (size-3, size-3) logic from your file
            grid = create_random_grid(size, size, obstacle_density)

            # Handle small grids where (2,2) and (size-3, size-3) might be invalid
            start_x = min(2, size - 1)
            start_y = min(2, size - 1)
            goal_x = min(size - 3, size - 1)
            goal_y = min(size - 3, size - 1)
            if goal_x < 0: goal_x = size - 1
            if goal_y < 0: goal_y = size - 1

            start, goal = (start_x, start_y), (goal_x, goal_y)

            grid[start[0], start[1]] = 0
            grid[goal[0], goal[1]] = 0

            start_time = time.perf_counter()
            profiler.enable()

            # --- Call the chosen strategy ---
            path = []
            if strategy_choice == 1:
                jobs = [(grid, start, goal)]
                results_list = strategy_func(jobs)
                path = results_list[0] if results_list else []
            else:
                # S2 and S3 are called directly
                path = strategy_func(grid, start, goal)

            profiler.disable()
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            trial_times.append(elapsed)

            if path:
                successful_paths += 1
                print(f"Trial {trial + 1}: {elapsed:.6f}s | Path: {len(path)} steps ")
            else:
                print(f"Trial {trial + 1}: {elapsed:.6f}s | No path found ✗")

        avg_time = np.mean(trial_times)
        results.append({
            'size': size,
            'avg_time': avg_time,
            'success_rate': successful_paths / num_trials
        })
        print(f"\nAverage time: {avg_time:.6f} seconds")
        print(f"Success rate: {successful_paths}/{num_trials}")

    # --- 4. PRINT SUMMARY ---
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY ({strategy_name})")
    print("=" * 80)
    print(f"{'Grid Size':<15} {'Avg Time (s)':<20} {'Success Rate':<15}")
    print("-" * 80)
    for result in results:
        print(f"{result['size']}x{result['size']:<10} {result['avg_time']:<20.6f} "
              f"{result['success_rate'] * 100:.1f}%")

    print("\n" + "=" * 80)
    print("CUMULATIVE PERFORMANCE PROFILING RESULTS")
    print("=" * 80)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('tottime')

    print("Most Time-Consuming Functions (in main process):")
    print("-" * 80)
    ps.print_stats(15)
    print(s.getvalue().rstrip())


# 5. MAIN FUNCTION (Adapted for Parallel)

def main():
    print_header()

    # Ask for mode
    print("\n SELECT MODE")
    print("-" * 40)
    print("1. Interactive Mode (Visualize a single path)")
    print("2. Benchmark Mode (Test a single parallel strategy)")

    while True:
        try:
            mode = int(input("\nSelect mode (1-2): "))
            if mode in [1, 2]:
                break
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Invalid input! Please enter a number.")

    if mode == 2:
        run_benchmark_mode()
        return

    # --- Interactive Mode ---
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - PARALLEL PATHFINDING")
    print("=" * 80)

    # Step 1: Get grid size
    grid_size = get_grid_size()

    # Step 2: Get grid type
    grid_type = get_grid_type()

    # Step 3: Create grid
    print("\nCREATING GRID...")
    print("-" * 40)
    if grid_type == 1:
        grid = create_empty_grid(grid_size, grid_size)
    elif grid_type == 2:
        density = get_obstacle_density()
        grid = create_random_grid(grid_size, grid_size, density)
    else:
        grid = create_custom_grid(grid_size)
    print()
    print_grid_info(grid)

    # Step 4: Get start and goal positions
    print("\nPOSITIONS")
    print("-" * 40)
    start = get_position(f"Enter START position (x,y) [0-{grid_size - 1}]: ", grid_size)
    goal = get_position(f"Enter GOAL position (x,y) [0-{grid_size - 1}]: ", grid_size)

    # Clear obstacles at start and goal
    print("\nEnsuring start and goal are accessible...")
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    # Clear a small 3x3 area
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for pos in [start, goal]:
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    grid[nx, ny] = 0
    print("Start and goal positions cleared")
    print(f"\nStart: {start}")
    print(f"Goal: {goal}")

    # --- Step 5: CHOOSE STRATEGY ---
    strategy_choice = get_strategy_choice()

    # Step 6: Run pathfinding
    run_parallel_pathfinding(grid, start, goal, strategy_choice)

    # Ask if user wants to try again
    print("\n" + "=" * 80)
    retry = input("Would you like to try another configuration? (y/n): ")
    if retry.lower() == 'y':
        main()
    else:
        print("\n Thank you for using A* Pathfinding Algorithm!")
        print("=" * 80)


if __name__ == "__main__":
    # --- THIS IS ESSENTIAL for multiprocessing ---
    multiprocessing.freeze_support()
    main()