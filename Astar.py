# Astar_unified.py
# This file contains all common A* functions, grid utilities, and visualization.
# It can be run directly to test the sequential algorithm,
# or imported by parallel scripts.

from typing import List, Tuple, Dict
import numpy as np
import heapq
from math import sqrt
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# ## Core A* Data Structures & Helpers
# (Combined from both files)
# ----------------------------------------------------------------------------

def create_node(position: Tuple[int, int], g: float = float('inf'),
                h: float = 0.0, parent: Dict = None) -> Dict:
    """Creates a node dictionary for the A* algorithm."""
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }


def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculates the Euclidean distance heuristic."""
    x1, y1 = pos1
    x2, y2 = pos2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Gets all valid (within bounds, not obstacle) neighbors, including diagonals."""
    x, y = position
    rows, cols = grid.shape

    # All possible moves (including diagonals)
    possible_moves = [
        (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
        (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1)
    ]

    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0
    ]


def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    """Reconstructs the path from a goal node (dict) by following parents."""
    path = []
    current = goal_node
    while current is not None:
        path.append(current['position'])
        current = current['parent']
    return path[::-1]


def reconstruct_from_parents(parents_map: Dict, goal_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Reconstructs the path from a parents dictionary.

    *** THIS IS THE FIXED, SAFER VERSION ***
    """
    path = []
    current_pos = goal_pos

    # Add a safety limit (e.g., grid size) to prevent infinite loops
    # in case of a corrupted parent map.
    grid_area = len(parents_map)  # A rough estimate
    if grid_area == 0:
        return []

    for _ in range(grid_area):  # Safety break
        path.append(current_pos)

        parent = parents_map.get(current_pos)

        if parent is None:
            # We hit the start node (its parent is None)
            if current_pos == path[-1]:  # Check if start is in map
                break  # Path is complete
            else:
                # This means we found the goal, but the start node (0,0)
                # isn't in the parents_map yet. The path is broken.
                return []  # Return an empty list

        current_pos = parent

        # This check is crucial:
        # If the loop ends *before* we find the 'None' parent,
        # it means the path is corrupt (e.g., a loop) or too long.
        if _ == grid_area - 1:
            return []  # Path is corrupt or too long

    # We found the start. Now, check if the *last* node in the path
    # (which should be the start node) actually has 'None' as its parent.
    start_node = path[-1]
    if parents_map.get(start_node) is None:
        return path[::-1]  # Return the reversed, correct path
    else:
        # The path terminated, but not at a valid start node
        return []


# ----------------------------------------------------------------------------
# ## Main Sequential A* Algorithm
# (Using the more optimized version from Parallel/Astar.py)
# ----------------------------------------------------------------------------

def find_path(grid: np.ndarray, start: Tuple[int, int],
              goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Finds the shortest path using the sequential A* algorithm.
    """

    # Check for invalid grid or start/goal positions
    if grid.shape[0] == 0 or (grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1):
        return []

    # Initialize start node
    start_node = create_node(
        position=start,
        g=0,
        h=calculate_heuristic(start, goal)
    )

    # Initialize open and closed sets
    open_list = [(start_node['f'], start)]  # Priority queue
    open_dict = {start: start_node}  # For quick node lookup by position
    closed_set = set()  # Explored nodes

    while open_list:
        # Get node with lowest f value
        _, current_pos = heapq.heappop(open_list)

        # Avoid re-processing nodes (can happen with heap)
        if current_pos in closed_set:
            continue

        current_node = open_dict[current_pos]

        # Goal check
        if current_pos == goal:
            return reconstruct_path(current_node)

        closed_set.add(current_pos)

        # Neighbor exploration
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            if neighbor_pos in closed_set:
                continue

            # Calculate new path cost
            tentative_g = current_node['g'] + calculate_heuristic(current_pos, neighbor_pos)
            neighbor_node = open_dict.get(neighbor_pos)

            # Check if this is a new node or a better path
            if neighbor_node is None or tentative_g < neighbor_node['g']:
                new_h = calculate_heuristic(neighbor_pos, goal)

                if neighbor_node is None:
                    # New node
                    neighbor_node = create_node(
                        position=neighbor_pos, g=tentative_g, h=new_h, parent=current_node
                    )
                    open_dict[neighbor_pos] = neighbor_node
                else:
                    # Better path to existing node
                    neighbor_node['g'] = tentative_g
                    neighbor_node['f'] = tentative_g + new_h
                    neighbor_node['parent'] = current_node

                # Add to priority queue
                heapq.heappush(open_list, (neighbor_node['f'], neighbor_pos))

    return []  # No path found


# ----------------------------------------------------------------------------
# ## Grid Creation Utilities
# (From Sequential/Astar.py)
# ----------------------------------------------------------------------------

def create_empty_grid(rows: int, cols: int) -> np.ndarray:
    """Creates an empty grid of zeros."""
    return np.zeros((rows, cols), dtype=int)


def create_random_grid(rows: int, cols: int, obstacle_density: float = 0.2) -> np.ndarray:
    """Creates a grid with randomly placed obstacles."""
    grid = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if np.random.random() < obstacle_density:
                grid[i, j] = 1
    return grid


def add_wall(grid: np.ndarray, start_pos: Tuple[int, int],
             end_pos: Tuple[int, int], orientation: str = 'vertical') -> np.ndarray:
    """Adds a line of obstacles (a wall) to the grid."""
    if orientation == 'vertical':
        x = start_pos[0]
        for y in range(start_pos[1], end_pos[1] + 1):
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                grid[x, y] = 1
    else:  # horizontal
        y = start_pos[1]
        for x in range(start_pos[0], end_pos[0] + 1):
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                grid[x, y] = 1
    return grid


def print_grid_info(grid: np.ndarray):
    """Prints statistics about the grid."""
    total_cells = grid.size
    obstacle_cells = np.sum(grid == 1)
    walkable_cells = np.sum(grid == 0)
    obstacle_percentage = (obstacle_cells / total_cells) * 100

    print(f"Grid Size: {grid.shape[0]} x {grid.shape[1]}")
    print(f"Total Cells: {total_cells}")
    print(f"Walkable Cells: {walkable_cells}")
    print(f"Obstacle Cells: {obstacle_cells} ({obstacle_percentage:.1f}%)")


# ----------------------------------------------------------------------------
# ## Visualization Utility
# (From Sequential/Astar.py)
# ----------------------------------------------------------------------------

def visualize_path(grid: np.ndarray, path: List[Tuple[int, int]],
                   start: Tuple[int, int], goal: Tuple[int, int],
                   title: str = "A* Pathfinding Result"):
    """Uses Matplotlib to draw the grid, obstacles, and path."""
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary')  # Black for obstacles (1), white for walkable (0)

    if path:
        path_array = np.array(path)
        plt.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, label='Path')
        plt.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=15, label='Start')
        plt.plot(path_array[-1, 1], path_array[-1, 0], 'ro', markersize=15, label='Goal')
    else:
        # No path found, just show start and goal
        plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
        plt.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal')

    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title(title, fontsize=16)
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# ## Main execution block
# (This code only runs when you execute this file directly)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Create a test grid
    GRID_SIZE = 50
    OBSTACLE_DENSITY = 0.3

    grid = create_random_grid(GRID_SIZE, GRID_SIZE, OBSTACLE_DENSITY)

    # 2. Define Start and Goal
    # Ensure start/goal are not obstacles
    start_pos = (0, 0)
    goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1)
    grid[start_pos] = 0
    grid[goal_pos] = 0

    print("--- Sequential A* Test ---")
    print_grid_info(grid)
    print(f"Start: {start_pos}, Goal: {goal_pos}")

    # 3. Run the pathfinding
    import time

    start_time = time.time()
    path = find_path(grid, start_pos, goal_pos)
    end_time = time.time()

    # 4. Print results
    if path:
        print(f"\nPath found in {end_time - start_time:.4f} seconds.")
        # print(f"Path: {path}") # Uncomment to see the full path
    else:
        print(f"\nNo path found. Time taken: {end_time - start_time:.4f} seconds.")

    # 5. Visualize the result
    visualize_path(grid, path, start_pos, goal_pos,
                   title=f"Sequential A* Test ({GRID_SIZE}x{GRID_SIZE})")