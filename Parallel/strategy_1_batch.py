import time
import heapq
import numpy as np
from math import sqrt
import multiprocessing as mp
from typing import List, Tuple, Dict

def calculate_heuristic(pos1, pos2):
    return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def get_neighbors(grid, pos):
    rows, cols = grid.shape
    x, y = pos
    moves = [
        (x+1, y), (x-1, y), (x, y+1), (x, y-1),
        (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)
    ]
    # Check bounds and obstacles (0 is walkable)
    return [(nx, ny) for nx, ny in moves 
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0]

def solve_single_path(args):
    """
    Standard A* Logic. 
    Args is a tuple: (grid, start, goal)
    """
    grid, start, goal = args
    
    # Safety checks
    if grid[start] == 1 or grid[goal] == 1:
        return []

    priority_queue = [(0, start)]
    g_costs = {start: 0.0}
    parents = {start: None}
    
    while priority_queue:
        current_f, current = heapq.heappop(priority_queue)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return path[::-1]
            
        # Optimization: Lazy deletion from heap
        if current_f > g_costs.get(current, float('inf')) + calculate_heuristic(current, goal):
             continue

        for neighbor in get_neighbors(grid, current):
            dx = abs(neighbor[0] - current[0])
            dy = abs(neighbor[1] - current[1])
            move_cost = 1.414 if dx == 1 and dy == 1 else 1.0
            
            new_g = g_costs[current] + move_cost
            
            if new_g < g_costs.get(neighbor, float('inf')):
                g_costs[neighbor] = new_g
                h = calculate_heuristic(neighbor, goal)
                parents[neighbor] = current
                heapq.heappush(priority_queue, (new_g + h, neighbor))
                
    return []

# --- Parallel Wrapper ---
def batch_process_paths(jobs: List[Tuple[np.ndarray, Tuple, Tuple]]):
    """
    Processes a list of pathfinding jobs in parallel.
    """
    # Use all available cores
    num_workers = mp.cpu_count()
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(solve_single_path, jobs)
    return results

# --- Main Benchmark ---
if __name__ == "__main__":
    # Setup
    GRID_SIZE = 200
    NUM_JOBS = 50  # Simulating 50 units asking for paths simultaneously
    
    print(f"--- Strategy 1: Batch Processing ({NUM_JOBS} paths) ---")
    
    # Create a random grid
    np.random.seed(42)
    master_grid = (np.random.random((GRID_SIZE, GRID_SIZE)) < 0.2).astype(int)
    
    # Generate random start/goal pairs
    jobs = []
    for _ in range(NUM_JOBS):
        s = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        g = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
        # Ensure start/goal are walkable
        master_grid[s] = 0
        master_grid[g] = 0
        jobs.append((master_grid, s, g))
    
    # 1. Sequential Benchmark
    start_time = time.time()
    seq_results = [solve_single_path(job) for job in jobs]
    seq_duration = time.time() - start_time
    print(f"Sequential Time: {seq_duration:.4f}s")
    
    # 2. Parallel Benchmark
    start_time = time.time()
    par_results = batch_process_paths(jobs)
    par_duration = time.time() - start_time
    print(f"Parallel Time:   {par_duration:.4f}s")
    
    # Stats
    speedup = seq_duration / par_duration
    print(f"Speedup:         {speedup:.2f}x")
    
    if speedup > 1.0:
        print("SUCCESS: Batch processing effectively handled high throughput.")
    else:
        print("NOTE: Grid might be too small or overhead too high.")