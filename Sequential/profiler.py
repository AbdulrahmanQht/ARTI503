import numpy as np
import time
import heapq
from math import sqrt
import matplotlib.pyplot as plt
import cProfile
import pstats


# ==============================================================================
# A* Algorithm and Grid Functions (from your AStar.py and main.py)
# ==============================================================================

def create_node(position: tuple, g: float = float('inf'),
                h: float = 0.0, parent: dict = None) -> dict:
    # A* is built by Priority Queue and F(n) = G(n) + H(n)
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }


def calculate_heuristic(pos1: tuple, pos2: tuple) -> float:
    x1, y1 = pos1
    x2, y2 = pos2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_valid_neighbors(grid: np.ndarray, position: tuple) -> list[tuple]:
    x, y = position
    rows, cols = grid.shape

    # All possible moves (including diagonals)
    possible_moves = [
        (x + 1, y), (x - 1, y),  # Right, Left
        (x, y + 1), (x, y - 1),  # Up, Down
        (x + 1, y + 1), (x - 1, y - 1),  # Diagonal moves
        (x + 1, y - 1), (x - 1, y + 1)  # Diagonal moves
    ]

    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols  # Within grid bounds
           and grid[nx, ny] == 0  # Not an obstacle
    ]


def reconstruct_path(goal_node: dict) -> list[tuple]:
    path = []
    current = goal_node

    while current is not None:
        path.append(current['position'])
        current = current['parent']

    return path[::-1]  # Reverse to get path from start to goal


def find_path(grid: np.ndarray, start: tuple,
              goal: tuple) -> list[tuple]:
    # Initialize start node
    start_node = create_node(
        position=start,
        g=0,
        h=calculate_heuristic(start, goal)
    )

    # Initialize open and closed sets
    open_list = [(start_node['f'], start)]  # Priority queue
    open_dict = {start: start_node}  # For quick node lookup
    closed_set = set()  # Explored nodes

    while open_list:
        # Get node with lowest f value
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]

        # Check if we reached the goal
        if current_pos == goal:
            return reconstruct_path(current_node)

        closed_set.add(current_pos)

        # NEIGHBOR EXPLORATION: Each neighbor evaluation is independent
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            # Skip if already explored
            if neighbor_pos in closed_set:
                continue

            # Calculate new path cost
            tentative_g = current_node['g'] + calculate_heuristic(current_pos, neighbor_pos)

            # Create or update neighbor
            if neighbor_pos not in open_dict:
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_heuristic(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]['g']:
                # Found a better path to the neighbor
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node

    return []  # No path found


def create_random_grid(rows: int, cols: int, obstacle_density: float = 0.2) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            if np.random.random() < obstacle_density:
                grid[i, j] = 1

    return grid


# ==============================================================================
# Profiler Execution
# ==============================================================================

def run_profile_mode():
    """Runs a single large grid and profiles the execution to find bottlenecks."""
    print("\n PROFILE MODE")
    print("=" * 80)

    # 1. Set up a single, large problem that is slow enough to measure
    size = 2500
    obstacle_density = 0.33
    grid = create_random_grid(size, size, obstacle_density)
    start = (2, 2)
    goal = (size - 3, size - 3)
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 0

    print(f"Profiling `find_path` on a {size}x{size} grid...")
    print("This may take several seconds...")

    # 2. Create a profiler object and run the function
    profiler = cProfile.Profile()
    profiler.enable()

    path = find_path(grid, start, goal)  # The function being profiled

    profiler.disable()

    # 3. Print the results
    print("-" * 80)
    if path:
        print(f"Path found with {len(path)} steps.")
    else:
        print("No path found.")

    print("\nPROFILER RESULTS (Sorted by cumulative time):")
    print("-" * 80)

    # 4. Create statistics from the profiler, sort by cumulative time, and print
    # 'cumulative' is the total time spent in a function, including sub-calls.
    stats = pstats.Stats(profiler).sort_stats('cumulative')

    # Print the top 15 most time-consuming functions
    stats.print_stats(15)


if __name__ == "__main__":
    run_profile_mode()