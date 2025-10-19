import numpy as np
import time
from AStar import (
    find_path,
    visualize_path,
    create_empty_grid,
    create_random_grid,
    add_wall,
    print_grid_info
)


def print_header():
    print("=" * 80)
    print(" " * 20 + "A* PATHFINDING ALGORITHM")
    print(" " * 15 + "Sequential Implementation - 9MS2")
    print(" " * 18 + "ARTI503 - Group 2 - 2025")
    print("=" * 80)


def get_grid_size():
    """Get grid size from user."""
    print("\n GRID SIZE")
    print("-" * 40)
    while True:
        try:
            size = int(input("Enter grid size (e.g., 20 for 20x20 grid): "))
            if size > 0 and size <= 1000:
                return size
            else:
                print(" Please enter a positive number ≤ 1000")
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
                print(f" Position must be within grid bounds (0 to {grid_size-1})")
        except ValueError:
            print(" Invalid format! Please enter as: x,y (e.g., 5,10)")


def create_custom_grid(size: int):
    """Create a custom grid with walls."""
    grid = create_empty_grid(size, size)

    print("\n CUSTOM WALLS")
    print("-" * 40)
    print("Add walls to your grid:")

    # Example: Add vertical wall
    print("\nAdding sample vertical wall at x=10, from y=5 to y=15...")
    if size > 15:
        grid = add_wall(grid, (10, 5), (10, 15), 'vertical')

    # Example: Add horizontal wall
    print("Adding sample horizontal wall at y=10, from x=5 to x=15...")
    if size > 15:
        grid = add_wall(grid, (5, 10), (15, 10), 'horizontal')

    return grid


def run_pathfinding(grid: np.ndarray, start: tuple, goal: tuple):
    """Run A* pathfinding and display results."""
    print("\n RUNNING A* ALGORITHM...")
    print("-" * 40)

    # Validate start and goal positions
    if grid[start[0], start[1]] == 1:
        print(" Error: Start position is an obstacle!")
        return

    if grid[goal[0], goal[1]] == 1:
        print(" Error: Goal position is an obstacle!")
        return

    # Run algorithm with timing
    start_time = time.perf_counter()
    path = find_path(grid, start, goal)
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

        # Show first few steps of the path
        print(f"\n  First 5 steps: {path[:5]}")
        if len(path) > 5:
            print(f"  Last 5 steps: {path[-5:]}")

        # Visualize
        print("\n Displaying visualization...")
        visualize_path(grid, path, start, goal,
                      f"A* Pathfinding: {len(path)} steps in {execution_time:.4f}s")
    else:
        print("✗ No path found!")
        print(f"  Execution time: {execution_time:.6f} seconds")
        print("  The goal may be unreachable due to obstacles.")

        # Still show the grid
        print("\n Displaying grid...")
        visualize_path(grid, path, start, goal, "A* Pathfinding: No path found")


def run_benchmark_mode():
    """Run benchmark mode with multiple grid sizes."""
    print("\n BENCHMARK MODE")
    print("=" * 80)

    grid_sizes = [20, 50, 100, 200]
    obstacle_density = 0.2
    num_trials = 3

    print(f"Testing grid sizes: {grid_sizes}")
    print(f"Obstacle density: {obstacle_density * 100}%")
    print(f"Trials per size: {num_trials}\n")

    results = []

    for size in grid_sizes:
        print(f"\n{'='*80}")
        print(f"Grid Size: {size} x {size}")
        print(f"{'='*80}")

        trial_times = []
        successful_paths = 0

        for trial in range(num_trials):
            # Create random grid
            grid = create_random_grid(size, size, obstacle_density)

            # Set start and goal
            start = (2, 2)
            goal = (size - 3, size - 3)

            # Ensure they're not obstacles
            grid[start[0], start[1]] = 0
            grid[goal[0], goal[1]] = 0

            # Run pathfinding
            start_time = time.perf_counter()
            path = find_path(grid, start, goal)
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

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Grid Size':<15} {'Avg Time (s)':<20} {'Success Rate':<15}")
    print("-" * 80)
    for result in results:
        print(f"{result['size']}x{result['size']:<10} {result['avg_time']:<20.6f} "
              f"{result['success_rate']*100:.1f}%")

    print("\n" + "=" * 80)
    print("TIME-CONSUMING SECTIONS IDENTIFIED:")
    print("=" * 80)
    print("1. Main A* loop - Sequential bottleneck")
    print("2. Neighbor exploration - Can be parallelized")
    print("3. Heuristic calculations - Can be parallelized")
    print("4. Grid operations - Can be parallelized")
    print("=" * 80)


def main():
    print_header()

    # Ask for mode
    print("\n SELECT MODE")
    print("-" * 40)
    print("1. Interactive Mode (visualize single pathfinding)")
    print("2. Benchmark Mode (test multiple grid sizes)")

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

    # Interactive Mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - PATHFINDING VISUALIZATION")
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
        print(" Empty grid created")
    elif grid_type == 2:
        density = get_obstacle_density()
        grid = create_random_grid(grid_size, grid_size, density)
        print(f" Random grid created with {density*100:.1f}% obstacles")
    else:
        grid = create_custom_grid(grid_size)
        print(" Custom grid created with sample walls")

    # Print grid info
    print()
    print_grid_info(grid)

    # Step 4: Get start and goal positions
    print("\nPOSITIONS")
    print("-" * 40)
    start = get_position(f"Enter START position (x,y) [0-{grid_size-1}]: ", grid_size)
    goal = get_position(f"Enter GOAL position (x,y) [0-{grid_size-1}]: ", grid_size)

    # Clear obstacles at start and goal positions (and surrounding area)
    print("\nEnsuring start and goal are accessible...")
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    # Clear a small area around start and goal to ensure accessibility
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            start_x, start_y = start[0] + dx, start[1] + dy
            goal_x, goal_y = goal[0] + dx, goal[1] + dy

            if 0 <= start_x < grid_size and 0 <= start_y < grid_size:
                grid[start_x, start_y] = 0

            if 0 <= goal_x < grid_size and 0 <= goal_y < grid_size:
                grid[goal_x, goal_y] = 0

    print("Start and goal positions cleared")

    print(f"\nStart: {start}")
    print(f"Goal: {goal}")

    # Step 5: Run pathfinding
    run_pathfinding(grid, start, goal)

    # Ask if user wants to try again
    print("\n" + "=" * 80)
    retry = input("Would you like to try another configuration? (y/n): ")
    if retry.lower() == 'y':
        main()
    else:
        print("\n Thank you for using A* Pathfinding Algorithm!")
        print("=" * 80)


if __name__ == "__main__":
    main()