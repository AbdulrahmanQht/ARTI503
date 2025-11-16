# Parallel/strategy_2_bidirectional.py
import multiprocessing
import heapq
from multiprocessing import Queue, Value, Array
import ctypes

import numpy as np
from typing import List, Tuple, Dict
from Astar import (
    calculate_heuristic, get_valid_neighbors,
    create_node
)


def bidirectional_worker(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        direction: str,
        result_queue: Queue,
        best_cost: Value,
        stop_flag: Value
):
    """
    Worker process for bidirectional search.
    Each worker explores from its start point and reports nodes visited.
    """
    if direction == 'fwd':
        my_start, my_goal = start, goal
    else:
        my_start, my_goal = goal, start

    # Local data structures (no shared memory overhead)
    open_list = [(0.0, my_start)]
    g_costs = {my_start: 0.0}
    parents = {my_start: None}
    closed_set = set()

    start_node = create_node(
        position=my_start,
        g=0,
        h=calculate_heuristic(my_start, my_goal)
    )
    open_dict = {my_start: start_node}

    while open_list and not stop_flag.value:
        current_f, current_pos = heapq.heappop(open_list)

        # Skip if already processed
        if current_pos in closed_set:
            continue

        # Prune if we already have a better solution
        if current_f >= best_cost.value:
            break

        if current_pos not in open_dict:
            continue

        current_node = open_dict.pop(current_pos)
        closed_set.add(current_pos)

        # Send this node to the coordinator
        result_queue.put((direction, current_pos, current_node['g'], parents[current_pos]))

        # Expand neighbors
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            if neighbor_pos in closed_set:
                continue

            tentative_g = current_node['g'] + calculate_heuristic(current_pos, neighbor_pos)

            # Only update if this is a better path
            if neighbor_pos not in g_costs or tentative_g < g_costs[neighbor_pos]:
                g_costs[neighbor_pos] = tentative_g
                parents[neighbor_pos] = current_pos

                h = calculate_heuristic(neighbor_pos, my_goal)
                f = tentative_g + h

                neighbor_node = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=h,
                    parent=current_node
                )

                heapq.heappush(open_list, (f, neighbor_pos))
                open_dict[neighbor_pos] = neighbor_node

    # Signal completion
    result_queue.put((direction, None, None, None))


def strategy_2_bidirectional(grid: np.ndarray, start: Tuple[int, int],
                             goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Bidirectional A* using multiprocessing.
    Two processes search from start and goal simultaneously.
    """

    # Check trivial cases
    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        return []

    if start == goal:
        return [start]

    # Shared memory for coordination
    result_queue = Queue()
    best_cost = Value('d', float('inf'))  # 'd' for double
    stop_flag = Value(ctypes.c_bool, False)

    # Start both workers
    p_fwd = multiprocessing.Process(
        target=bidirectional_worker,
        args=(grid, start, goal, 'fwd', result_queue, best_cost, stop_flag)
    )
    p_bwd = multiprocessing.Process(
        target=bidirectional_worker,
        args=(grid, start, goal, 'bwd', result_queue, best_cost, stop_flag)
    )

    p_fwd.start()
    p_bwd.start()

    # Coordinator: collect results and detect meeting point
    g_fwd: Dict[Tuple[int, int], float] = {}
    g_bwd: Dict[Tuple[int, int], float] = {}
    parents_fwd: Dict[Tuple[int, int], Tuple[int, int]] = {}
    parents_bwd: Dict[Tuple[int, int], Tuple[int, int]] = {}

    best_meeting = None
    completed_workers = 0

    while completed_workers < 2:
        try:
            direction, pos, g_cost, parent = result_queue.get(timeout=0.1)

            # Check for completion signal
            if pos is None:
                completed_workers += 1
                continue

            # Store the node info
            if direction == 'fwd':
                g_fwd[pos] = g_cost
                parents_fwd[pos] = parent

                # Check if backward search has visited this node
                if pos in g_bwd:
                    total_cost = g_cost + g_bwd[pos]
                    if total_cost < best_cost.value:
                        best_cost.value = total_cost
                        best_meeting = pos

            else:  # 'bwd'
                g_bwd[pos] = g_cost
                parents_bwd[pos] = parent

                # Check if forward search has visited this node
                if pos in g_fwd:
                    total_cost = g_cost + g_fwd[pos]
                    if total_cost < best_cost.value:
                        best_cost.value = total_cost
                        best_meeting = pos

        except:
            # Timeout - check if processes are still alive
            if not p_fwd.is_alive() and not p_bwd.is_alive():
                break

    # Signal workers to stop
    stop_flag.value = True

    # Wait for processes to finish
    p_fwd.join(timeout=1)
    p_bwd.join(timeout=1)

    # Terminate if still running
    if p_fwd.is_alive():
        p_fwd.terminate()
    if p_bwd.is_alive():
        p_bwd.terminate()

    # Reconstruct path if meeting point found
    if best_meeting is not None:
        # Forward path: start → meeting_node
        path_fwd = []
        current = best_meeting
        while current is not None:
            path_fwd.append(current)
            current = parents_fwd.get(current)
        path_fwd.reverse()

        # Backward path: meeting_node → goal
        path_bwd = []
        current = parents_bwd.get(best_meeting)  # Skip meeting node itself
        while current is not None:
            path_bwd.append(current)
            current = parents_bwd.get(current)

        return path_fwd + path_bwd

    return []