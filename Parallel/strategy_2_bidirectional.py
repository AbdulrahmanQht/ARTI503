import multiprocessing as mp
import numpy as np
import heapq
import time
from math import sqrt
from typing import List, Tuple

def get_neighbors_flat(current_idx, rows, cols, grid_flat):
    r, c = divmod(current_idx, cols)
    neighbors = []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            n_idx = nr * cols + nc
            if grid_flat[n_idx] == 0:
                neighbors.append(n_idx)
    return neighbors


def shared_memory_worker(
        worker_id, grid_shape, grid_shared,
        my_visited_g, other_visited_g, parent_map, other_parents,  # Added other_parents
        start_pos, goal_pos,
        meeting_node_shared, meeting_g_shared, lock
):
    rows, cols = grid_shape
    start_h = sqrt((start_pos[0] - goal_pos[0]) ** 2 + (start_pos[1] - goal_pos[1]) ** 2)

    start_idx = start_pos[0] * cols + start_pos[1]
    open_list = [(start_h, start_idx)]

    # Init start node
    parent_map[start_idx] = -1
    my_visited_g[start_idx] = 0.0

    closed_set_local = set()
    local_best_meeting_cost = float('inf')
    ops_counter = 0

    while open_list:
        ops_counter += 1
        if ops_counter >= 100:
            ops_counter = 0
            if meeting_g_shared.value < local_best_meeting_cost:
                local_best_meeting_cost = meeting_g_shared.value
            if open_list and open_list[0][0] > local_best_meeting_cost:
                break

        f, current_idx = heapq.heappop(open_list)

        if current_idx in closed_set_local: continue
        closed_set_local.add(current_idx)

        current_g = my_visited_g[current_idx]

        # --- ROBUST INTERSECTION CHECK ---
        other_g = other_visited_g[current_idx]

        if other_g != -1.0:
            # INTEGRITY CHECK: Ensure the other thread has actually written the parent
            # If parent is still -2 (unvisited) means we caught a race condition
            # We skip this meeting point for now; we will catch later when the write propagates
            if other_parents[current_idx] != -2:
                total_cost = current_g + other_g
                if total_cost < local_best_meeting_cost:
                    with lock:
                        if total_cost < meeting_g_shared.value:
                            meeting_g_shared.value = total_cost
                            meeting_node_shared[0] = current_idx
                            local_best_meeting_cost = total_cost

        current_r, current_c = divmod(current_idx, cols)

        for n_idx in get_neighbors_flat(current_idx, rows, cols, grid_shared):
            if n_idx in closed_set_local: continue

            nr, nc = divmod(n_idx, cols)
            dist = 1.414 if abs(nr - current_r) + abs(nc - current_c) == 2 else 1.0
            new_g = current_g + dist

            existing_g = my_visited_g[n_idx]

            if existing_g == -1.0 or new_g < existing_g:
                # Update parent first, then cost
                parent_map[n_idx] = current_idx
                my_visited_g[n_idx] = new_g

                h = sqrt((nr - goal_pos[0]) ** 2 + (nc - goal_pos[1]) ** 2)
                heapq.heappush(open_list, (new_g + h, n_idx))


def run_bidirectional_parallel(grid, start, goal):
    if start == goal: return [start]
    rows, cols = grid.shape
    total_cells = rows * cols

    # Shared Arrays
    grid_flat = mp.Array('i', grid.flatten(), lock=False)
    fwd_g = mp.Array('d', [-1.0] * total_cells, lock=False)
    bwd_g = mp.Array('d', [-1.0] * total_cells, lock=False)
    fwd_parents = mp.Array('i', [-2] * total_cells, lock=False)
    bwd_parents = mp.Array('i', [-2] * total_cells, lock=False)

    meet_node = mp.Array('i', [-1])
    meet_cost = mp.Value('d', float('inf'))
    lock = mp.Lock()

    # Pass BOTH parent arrays to BOTH workers so they can verify integrity
    p1 = mp.Process(target=shared_memory_worker, args=(
    1, grid.shape, grid_flat, fwd_g, bwd_g, fwd_parents, bwd_parents, start, goal, meet_node, meet_cost, lock))
    p2 = mp.Process(target=shared_memory_worker, args=(
    2, grid.shape, grid_flat, bwd_g, fwd_g, bwd_parents, fwd_parents, goal, start, meet_node, meet_cost, lock))

    p1.start();
    p2.start()
    p1.join();
    p2.join()

    if meet_node[0] == -1: return []

    # Path Reconstruction
    meet_idx = meet_node[0]
    path_fwd = []
    curr = meet_idx

    # Safety loop limit
    limit = total_cells
    while curr != -1 and curr != -2 and limit > 0:
        path_fwd.append(divmod(curr, cols))
        curr = fwd_parents[curr]
        limit -= 1
    path_fwd.reverse()

    path_bwd = []
    curr = bwd_parents[meet_idx]
    limit = total_cells
    while curr != -1 and curr != -2 and limit > 0:
        path_bwd.append(divmod(curr, cols))
        curr = bwd_parents[curr]
        limit -= 1

    return path_fwd + path_bwd