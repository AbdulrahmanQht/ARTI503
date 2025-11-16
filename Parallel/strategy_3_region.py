# strategy_3_region.py
from typing import List, Tuple, Dict, Set
import numpy as np
import heapq
from math import sqrt
import matplotlib.pyplot as plt
import multiprocessing as mp
from dataclasses import dataclass
import time
from Astar import (
    create_node, calculate_heuristic, get_valid_neighbors,
    reconstruct_path, create_random_grid, visualize_path
)

@dataclass
class Region:
    """Represents a rectangular region of the grid."""
    id: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int

    def contains(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within this region."""
        x, y = pos
        return (self.x_start <= x < self.x_end and
                self.y_start <= y < self.y_end)

    def get_border_with_neighbor(self, neighbor: 'Region') -> Set[Tuple[int, int]]:
        """Get the shared border cells between this region and a neighbor."""
        border = set()

        # Check if regions are adjacent
        # Vertical border (left-right neighbors)
        if self.x_start == neighbor.x_end or self.x_end == neighbor.x_start:
            x = self.x_start if self.x_start == neighbor.x_end else self.x_end - 1
            y_start = max(self.y_start, neighbor.y_start)
            y_end = min(self.y_end, neighbor.y_end)
            for y in range(y_start, y_end):
                border.add((x, y))

        # Horizontal border (top-bottom neighbors)
        elif self.y_start == neighbor.y_end or self.y_end == neighbor.y_start:
            y = self.y_start if self.y_start == neighbor.y_end else self.y_end - 1
            x_start = max(self.x_start, neighbor.x_start)
            x_end = min(self.x_end, neighbor.x_end)
            for x in range(x_start, x_end):
                border.add((x, y))

        return border


def divide_grid_into_regions(grid: np.ndarray, num_regions_x: int,
                             num_regions_y: int) -> List[Region]:
    """Divide the grid into rectangular regions."""
    rows, cols = grid.shape
    regions = []

    region_height = rows // num_regions_x
    region_width = cols // num_regions_y

    region_id = 0
    for i in range(num_regions_x):
        for j in range(num_regions_y):
            x_start = i * region_height
            x_end = (i + 1) * region_height if i < num_regions_x - 1 else rows
            y_start = j * region_width
            y_end = (j + 1) * region_width if j < num_regions_y - 1 else cols

            regions.append(Region(region_id, x_start, x_end, y_start, y_end))
            region_id += 1

    return regions


def find_local_path_worker(args) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Worker to find path within a single region from entry to exit point.
    Much more efficient - only one A* search per region in the path.
    """
    grid, region, entry_point, exit_point = args

    def calculate_heuristic(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def create_node(position, g=float('inf'), h=0.0, parent=None):
        return {'position': position, 'g': g, 'h': h, 'f': g + h, 'parent': parent}

    def get_neighbors_in_region(pos):
        x, y = pos
        moves = [
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
            (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1)
        ]
        return [(nx, ny) for nx, ny in moves
                if (region.x_start <= nx < region.x_end and
                    region.y_start <= ny < region.y_end and
                    grid[nx, ny] == 0)]

    # A* within region
    start_node = create_node(entry_point, g=0, h=calculate_heuristic(entry_point, exit_point))
    open_list = [(start_node['f'], entry_point)]
    open_dict = {entry_point: start_node}
    closed_set = set()

    while open_list:
        _, current_pos = heapq.heappop(open_list)

        if current_pos in closed_set:
            continue

        current_node = open_dict[current_pos]

        if current_pos == exit_point:
            # Reconstruct path
            path = []
            node = current_node
            while node:
                path.append(node['position'])
                node = node['parent']
            return region.id, path[::-1]

        closed_set.add(current_pos)

        for neighbor_pos in get_neighbors_in_region(current_pos):
            if neighbor_pos in closed_set:
                continue

            dx = abs(neighbor_pos[0] - current_pos[0])
            dy = abs(neighbor_pos[1] - current_pos[1])
            move_cost = sqrt(2) if (dx == 1 and dy == 1) else 1.0
            tentative_g = current_node['g'] + move_cost
            neighbor_node = open_dict.get(neighbor_pos)

            if neighbor_node is None or tentative_g < neighbor_node['g']:
                new_h = calculate_heuristic(neighbor_pos, exit_point)

                if neighbor_node is None:
                    neighbor_node = create_node(neighbor_pos, tentative_g, new_h, current_node)
                    open_dict[neighbor_pos] = neighbor_node
                else:
                    neighbor_node['g'] = tentative_g
                    neighbor_node['f'] = tentative_g + new_h
                    neighbor_node['parent'] = current_node

                heapq.heappush(open_list, (neighbor_node['f'], neighbor_pos))

    return region.id, []  # No path found in this region


def find_path_region_parallel(grid: np.ndarray, start: Tuple[int, int],
                              goal: Tuple[int, int], num_regions_x: int = 4,
                              num_regions_y: int = 4, num_processes: int = None) -> List[Tuple[int, int]]:
    """
    Efficient region-based parallel A*.
    Uses high-level planning then parallel local searches.
    """

    def find_path_sequential(grid, start, goal):
        """Fallback sequential A*"""
        if grid.shape[0] == 0 or grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
            return []

        def create_node(position, g=float('inf'), h=0.0, parent=None):
            return {'position': position, 'g': g, 'h': h, 'f': g + h, 'parent': parent}

        def calculate_heuristic(pos1, pos2):
            x1, y1 = pos1
            x2, y2 = pos2
            return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        def get_valid_neighbors(grid, position):
            x, y = position
            rows, cols = grid.shape
            possible_moves = [
                (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1)
            ]
            return [(nx, ny) for nx, ny in possible_moves
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0]

        def reconstruct_path(goal_node):
            path = []
            current = goal_node
            while current is not None:
                path.append(current['position'])
                current = current['parent']
            return path[::-1]

        start_node = create_node(position=start, g=0, h=calculate_heuristic(start, goal))
        open_list = [(start_node['f'], start)]
        open_dict = {start: start_node}
        closed_set = set()

        while open_list:
            _, current_pos = heapq.heappop(open_list)

            if current_pos in closed_set:
                continue

            current_node = open_dict[current_pos]

            if current_pos == goal:
                return reconstruct_path(current_node)

            closed_set.add(current_pos)

            for neighbor_pos in get_valid_neighbors(grid, current_pos):
                if neighbor_pos in closed_set:
                    continue

                dx = abs(neighbor_pos[0] - current_pos[0])
                dy = abs(neighbor_pos[1] - current_pos[1])
                move_cost = sqrt(2) if (dx == 1 and dy == 1) else 1.0
                tentative_g = current_node['g'] + move_cost
                neighbor_node = open_dict.get(neighbor_pos)

                if neighbor_node is None or tentative_g < neighbor_node['g']:
                    new_h = calculate_heuristic(neighbor_pos, goal)

                    if neighbor_node is None:
                        neighbor_node = create_node(
                            position=neighbor_pos, g=tentative_g, h=new_h, parent=current_node
                        )
                        open_dict[neighbor_pos] = neighbor_node
                    else:
                        neighbor_node['g'] = tentative_g
                        neighbor_node['f'] = tentative_g + new_h
                        neighbor_node['parent'] = current_node

                    heapq.heappush(open_list, (neighbor_node['f'], neighbor_pos))

        return []

    # Validation
    if grid.shape[0] == 0 or grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        return []

    # For small grids or short distances, use sequential
    dist = sqrt((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2)
    if dist < 50 or grid.shape[0] < 100:
        return find_path_sequential(grid, start, goal)

    # Divide into regions
    regions = divide_grid_into_regions(grid, num_regions_x, num_regions_y)

    # Find which regions contain start and goal
    start_region = next((r for r in regions if r.contains(start)), None)
    goal_region = next((r for r in regions if r.contains(goal)), None)

    if not start_region or not goal_region:
        return find_path_sequential(grid, start, goal)

    # If start and goal in same region, just use sequential
    if start_region.id == goal_region.id:
        return find_path_sequential(grid, start, goal)

    # High-level path: which regions to traverse (simple straight-line heuristic)
    # For simplicity, we'll identify regions along a rough path from start to goal
    current_region = start_region
    region_sequence = [start_region]
    visited_regions = {start_region.id}

    # Simple greedy region selection toward goal
    while current_region.id != goal_region.id:
        # Find neighboring region closest to goal
        best_neighbor = None
        best_dist = float('inf')

        for region in regions:
            if region.id in visited_regions:
                continue

            # Check if adjacent
            is_adjacent = (
                                  (abs(region.x_start - current_region.x_end) <= 1 or
                                   abs(region.x_end - current_region.x_start) <= 1) and
                                  not (region.y_end <= current_region.y_start or
                                       region.y_start >= current_region.y_end)
                          ) or (
                                  (abs(region.y_start - current_region.y_end) <= 1 or
                                   abs(region.y_end - current_region.y_start) <= 1) and
                                  not (region.x_end <= current_region.x_start or
                                       region.x_start >= current_region.x_end)
                          )

            if is_adjacent:
                # Distance from region center to goal
                center_x = (region.x_start + region.x_end) / 2
                center_y = (region.y_start + region.y_end) / 2
                dist_to_goal = sqrt((center_x - goal[0]) ** 2 + (center_y - goal[1]) ** 2)

                if dist_to_goal < best_dist:
                    best_dist = dist_to_goal
                    best_neighbor = region

        if not best_neighbor:
            # Can't find path through regions, use sequential
            return find_path_sequential(grid, start, goal)

        region_sequence.append(best_neighbor)
        visited_regions.add(best_neighbor.id)
        current_region = best_neighbor

        # Safety limit
        if len(region_sequence) > len(regions):
            return find_path_sequential(grid, start, goal)

    # Now find entry/exit points for each region pair
    segment_tasks = []
    for i in range(len(region_sequence) - 1):
        curr_reg = region_sequence[i]
        next_reg = region_sequence[i + 1]

        # Entry point
        if i == 0:
            entry = start
        else:
            # Find best crossing point from previous segment
            border = curr_reg.get_border_with_neighbor(region_sequence[i - 1])
            walkable_border = [p for p in border if grid[p[0], p[1]] == 0]
            if not walkable_border:
                return find_path_sequential(grid, start, goal)
            entry = min(walkable_border,
                        key=lambda p: sqrt((p[0] - goal[0]) ** 2 + (p[1] - goal[1]) ** 2))

        # Exit point
        if i == len(region_sequence) - 2:
            exit_pt = goal
        else:
            border = curr_reg.get_border_with_neighbor(next_reg)
            walkable_border = [p for p in border if grid[p[0], p[1]] == 0]
            if not walkable_border:
                return find_path_sequential(grid, start, goal)
            exit_pt = min(walkable_border,
                          key=lambda p: sqrt((p[0] - goal[0]) ** 2 + (p[1] - goal[1]) ** 2))

        segment_tasks.append((grid, curr_reg, entry, exit_pt))

    # Process segments in parallel
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(segment_tasks))

    try:
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(find_local_path_worker, segment_tasks)
    except Exception as e:
        print(f"Parallel processing failed: {e}")
        return find_path_sequential(grid, start, goal)

    # Combine segment paths
    full_path = []
    for region_id, segment_path in results:
        if not segment_path:
            # Segment failed, use sequential fallback
            return find_path_sequential(grid, start, goal)

        if full_path and segment_path[0] == full_path[-1]:
            full_path.extend(segment_path[1:])
        else:
            full_path.extend(segment_path)

    return full_path if full_path else find_path_sequential(grid, start, goal)


def visualize_with_regions(grid: np.ndarray, path: List[Tuple[int, int]],
                           start: Tuple[int, int], goal: Tuple[int, int],
                           num_regions_x: int, num_regions_y: int,
                           title: str = "Region-Based Parallel A*"):
    """Visualize path with region boundaries."""
    regions = divide_grid_into_regions(grid, num_regions_x, num_regions_y)

    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap='binary', alpha=0.7)

    # Draw region boundaries
    for region in regions:
        rect = plt.Rectangle((region.y_start - 0.5, region.x_start - 0.5),
                             region.y_end - region.y_start,
                             region.x_end - region.x_start,
                             fill=False, edgecolor='blue', linewidth=2, linestyle='--', alpha=0.6)
        plt.gca().add_patch(rect)

    # Draw path
    if path:
        path_array = np.array(path)
        plt.plot(path_array[:, 1], path_array[:, 0], 'g-', linewidth=2, label='Path', alpha=0.8)

    plt.plot(start[1], start[0], 'go', markersize=12, label='Start')
    plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.title(title, fontsize=16)
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    def create_random_grid(rows, cols, obstacle_density=0.2):
        """Creates a grid with randomly placed obstacles."""
        grid = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            for j in range(cols):
                if np.random.random() < obstacle_density:
                    grid[i, j] = 1
        return grid


    # Test configuration
    GRID_SIZE = 1000
    OBSTACLE_DENSITY = 0.34
    NUM_REGIONS_X = 4
    NUM_REGIONS_Y = 4

    print("=" * 60)
    print("EFFICIENT REGION-BASED PARALLEL A* TEST")
    print("=" * 60)
    print()

    # Create test grid
    np.random.seed(42)
    grid = create_random_grid(GRID_SIZE, GRID_SIZE, OBSTACLE_DENSITY)
    start_pos = (5, 5)
    goal_pos = (GRID_SIZE - 6, GRID_SIZE - 6)
    grid[start_pos] = 0
    grid[goal_pos] = 0

    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Obstacle Density: {OBSTACLE_DENSITY * 100}%")
    print(f"Region Division: {NUM_REGIONS_X}x{NUM_REGIONS_Y} = {NUM_REGIONS_X * NUM_REGIONS_Y} regions")
    print(f"Start: {start_pos}, Goal: {goal_pos}")
    print(f"CPU Cores Available: {mp.cpu_count()}")
    print()

    # Test
    print("Running Efficient Region-Based Parallel A*...")
    start_time = time.time()
    path = find_path_region_parallel(grid, start_pos, goal_pos,
                                     NUM_REGIONS_X, NUM_REGIONS_Y)
    elapsed_time = time.time() - start_time

    if path:
        print(f"✓ Path found!")
        print(f"  Steps: {len(path)}")
        path_length = sum(sqrt((path[i + 1][0] - path[i][0]) ** 2 +
                               (path[i + 1][1] - path[i][1]) ** 2)
                          for i in range(len(path) - 1))
        print(f"  Path length: {path_length:.2f}")
        print(f"  Time: {elapsed_time:.4f} seconds")
    else:
        print(f"✗ No path found")
        print(f"  Time: {elapsed_time:.4f} seconds")

    print()
    print("Visualizing result...")
    visualize_with_regions(grid, path, start_pos, goal_pos,
                           NUM_REGIONS_X, NUM_REGIONS_Y,
                           title=f"Efficient Region-Based Parallel A*")