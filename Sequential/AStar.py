from typing import List, Tuple, Dict
import numpy as np
import heapq
from math import sqrt
import matplotlib.pyplot as plt


def create_node(position: Tuple[int, int], g: float = float('inf'),
                h: float = 0.0, parent: Dict = None) -> Dict:
    # A* is built by Priority Queue and F(n) = G(n) + H(n)
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }


def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    x1, y1 = pos1
    x2, y2 = pos2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
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


def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    path = []
    current = goal_node

    while current is not None:
        path.append(current['position'])
        current = current['parent']

    return path[::-1]  # Reverse to get path from start to goal


def find_path(grid: np.ndarray, start: Tuple[int, int],
              goal: Tuple[int, int]) -> List[Tuple[int, int]]:

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


def visualize_path(grid: np.ndarray, path: List[Tuple[int, int]],
                   start: Tuple[int, int], goal: Tuple[int, int],
                   title: str = "Sequential A* Pathfinding Result"):

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary')

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
    plt.title(title)
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.tight_layout()
    plt.show()


def create_empty_grid(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=int)


def create_random_grid(rows: int, cols: int, obstacle_density: float = 0.2) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            if np.random.random() < obstacle_density:
                grid[i, j] = 1

    return grid


def add_wall(grid: np.ndarray, start_pos: Tuple[int, int],
             end_pos: Tuple[int, int], orientation: str = 'vertical') -> np.ndarray:

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
    total_cells = grid.size
    obstacle_cells = np.sum(grid == 1)
    walkable_cells = np.sum(grid == 0)
    obstacle_percentage = (obstacle_cells / total_cells) * 100

    print(f"Grid Size: {grid.shape[0]} x {grid.shape[1]}")
    print(f"Total Cells: {total_cells}")
    print(f"Walkable Cells: {walkable_cells}")
    print(f"Obstacle Cells: {obstacle_cells} ({obstacle_percentage:.1f}%)")