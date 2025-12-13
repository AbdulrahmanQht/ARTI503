import time
import heapq
import numpy as np
from math import sqrt
import multiprocessing as mp
from typing import List, Tuple

def solve_region_segment(args):
    """
    Finds a path within a specific region from a start point to an end point.
    """
    grid, x_bounds, y_bounds, local_start, local_goal = args
    
    # Local bounds
    r_min, r_max = x_bounds
    c_min, c_max = y_bounds
    
    def get_local_neighbors(pos):
        x, y = pos
        moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1),
                 (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]
        valid = []
        for nx, ny in moves:
            # Stay within REGION bounds
            if r_min <= nx < r_max and c_min <= ny < c_max:
                if grid[nx, ny] == 0:
                    valid.append((nx, ny))
        return valid

    # Standard A*
    pq = [(0, local_start)]
    g_cost = {local_start: 0}
    parents = {local_start: None}
    
    while pq:
        _, curr = heapq.heappop(pq)
        if curr == local_goal:
            path = []
            while curr:
                path.append(curr)
                curr = parents[curr]
            return path[::-1]
            
        for n in get_local_neighbors(curr):
            dx, dy = abs(n[0]-curr[0]), abs(n[1]-curr[1])
            new_g = g_cost[curr] + (1.414 if dx==1 and dy==1 else 1.0)
            
            if new_g < g_cost.get(n, float('inf')):
                g_cost[n] = new_g
                parents[n] = curr
                h = sqrt((n[0]-local_goal[0])**2 + (n[1]-local_goal[1])**2)
                heapq.heappush(pq, (new_g + h, n))
    return [] # No path in this segment

def run_region_parallel(grid, start, goal, splits=4):
    rows, cols = grid.shape
    row_h, col_w = rows // splits, cols // splits
    
    # 1. Identify "Waypoints" at region borders
    # Simplified Logic: Draw a diagonal line of waypoints
    waypoints = [start]
    
    # Calculate intermediate points along the diagonal
    for i in range(1, splits):
        # The center of the shared border between region i-1 and region i
        wp_r = int(start[0] + (goal[0] - start[0]) * (i / splits))
        wp_c = int(start[1] + (goal[1] - start[1]) * (i / splits))
        
        # Ensure waypoint is walkable (search nearby if blocked)
        search_radius = 5
        found = False
        for r in range(max(0, wp_r-search_radius), min(rows, wp_r+search_radius)):
            for c in range(max(0, wp_c-search_radius), min(cols, wp_c+search_radius)):
                if grid[r, c] == 0:
                    waypoints.append((r, c))
                    found = True
                    break
            if found: break
        
        if not found: return [] # Critical failure in planning
        
    waypoints.append(goal)
    
    # 2. Create Parallel Jobs
    tasks = []
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i+1]
        
        # Define region bounds that contain both p1 and p2
        # We give it some buffer so it's not strictly limited to one small box
        r_min = max(0, min(p1[0], p2[0]) - 20)
        r_max = min(rows, max(p1[0], p2[0]) + 20)
        c_min = max(0, min(p1[1], p2[1]) - 20)
        c_max = min(cols, max(p1[1], p2[1]) + 20)
        
        tasks.append((grid, (r_min, r_max), (c_min, c_max), p1, p2))
    
    # 3. Execute Parallel
    with mp.Pool(processes=len(tasks)) as pool:
        segment_paths = pool.map(solve_region_segment, tasks)
    
    # 4. Stitch
    full_path = []
    for p in segment_paths:
        if not p: return [] # A segment failed
        if full_path:
            full_path.extend(p[1:]) # Avoid duplicating the join point
        else:
            full_path.extend(p)
            
    return full_path

if __name__ == "__main__":
    GRID_SIZE = 1000
    print(f"--- Strategy 3: Region Decomposition ({GRID_SIZE}x{GRID_SIZE}) ---")
    
    np.random.seed(42)
    grid = (np.random.random((GRID_SIZE, GRID_SIZE)) < 0.15).astype(int) # Lower density for region validity
    start, goal = (0, 0), (GRID_SIZE-1, GRID_SIZE-1)
    grid[start], grid[goal] = 0, 0
    
    start_t = time.time()
    path = run_region_parallel(grid, start, goal, splits=8) # 8 segments
    duration = time.time() - start_t
    
    print(f"Time Taken: {duration:.4f}s")
    if path:
        print(f"Path Found! Length: {len(path)}")
    else:
        print("No path found (likely blocked waypoint).")