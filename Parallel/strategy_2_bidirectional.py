# strategy_2_bidirectional.py
import multiprocessing
import heapq
from multiprocessing import Queue, Value
import ctypes
import time
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from Astar import calculate_heuristic, get_valid_neighbors


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
    Worker with FIXED early termination logic.
    """
    if direction == 'fwd':
        my_start, my_goal = start, goal
    else:
        my_start, my_goal = goal, start

    # Local data structures
    open_list = []
    g_costs = {}
    parents = {}
    closed_set = set()

    # Initialize
    start_g = 0.0
    start_h = calculate_heuristic(my_start, my_goal)
    heapq.heappush(open_list, (start_g + start_h, my_start))
    g_costs[my_start] = start_g
    parents[my_start] = None

    # Send updates less frequently
    UPDATE_FREQUENCY = 10000  # Every 10k nodes
    nodes_processed = 0
    last_update = 0
    MAX_NODES = 1000000  # Safety limit

    try:
        while open_list and not stop_flag.value and nodes_processed < MAX_NODES:
            current_f, current_pos = heapq.heappop(open_list)

            if current_pos in closed_set:
                continue

            # FIXED: Only prune if we actually have a valid meeting point
            # Don't terminate early based on best_cost if it's still infinity
            current_best = best_cost.value
            if current_best < float('inf'):
                # We have a meeting point, can be more aggressive
                if current_f >= current_best * 1.5:
                    break
            # If no meeting yet, keep searching

            closed_set.add(current_pos)
            nodes_processed += 1

            # Send periodic updates
            if nodes_processed - last_update >= UPDATE_FREQUENCY:
                # Send sample of recent nodes near middle
                middle_x = (start[0] + goal[0]) // 2
                middle_y = (start[1] + goal[1]) // 2

                middle_nodes = []
                # Sample from recent expansions
                recent_count = min(5000, len(g_costs))
                sample_items = list(g_costs.items())[-recent_count:]

                for pos, g in sample_items:
                    dist_to_middle = abs(pos[0] - middle_x) + abs(pos[1] - middle_y)
                    if dist_to_middle < max(grid.shape) // 3:
                        middle_nodes.append((pos, g, parents[pos]))
                        if len(middle_nodes) >= 200:
                            break

                if middle_nodes:
                    result_queue.put(('update', direction, middle_nodes, nodes_processed))
                last_update = nodes_processed

            # Expand neighbors
            for neighbor_pos in get_valid_neighbors(grid, current_pos):
                if neighbor_pos in closed_set:
                    continue

                dx = abs(neighbor_pos[0] - current_pos[0])
                dy = abs(neighbor_pos[1] - current_pos[1])
                move_cost = 1.4142135623730951 if (dx == 1 and dy == 1) else 1.0

                tentative_g = g_costs[current_pos] + move_cost

                if neighbor_pos not in g_costs or tentative_g < g_costs[neighbor_pos]:
                    g_costs[neighbor_pos] = tentative_g
                    parents[neighbor_pos] = current_pos

                    h = calculate_heuristic(neighbor_pos, my_goal)
                    f = tentative_g + h

                    # Only prune if we have a valid meeting point
                    if current_best < float('inf'):
                        if f >= current_best * 1.5:
                            continue

                    heapq.heappush(open_list, (f, neighbor_pos))

            # Check stop flag less frequently
            if nodes_processed % 5000 == 0 and stop_flag.value:
                break

    except Exception as e:
        result_queue.put(('error', direction, str(e)))
        return

    # Send final complete data
    final_data = {
        'g_costs': dict(g_costs),
        'parents': dict(parents),
        'nodes_processed': nodes_processed
    }

    result_queue.put(('complete', direction, final_data))


def find_meeting_point(
        g_fwd: Dict[Tuple[int, int], float],
        g_bwd: Dict[Tuple[int, int], float],
        current_best: float
) -> Tuple[Optional[Tuple[int, int]], float]:
    """Find best meeting point."""
    best_meeting = None
    best_cost = current_best

    meeting_points = g_fwd.keys() & g_bwd.keys()

    for pos in meeting_points:
        total = g_fwd[pos] + g_bwd[pos]
        if total < best_cost:
            best_cost = total
            best_meeting = pos

    return best_meeting, best_cost


def strategy_2_bidirectional(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Bidirectional A* with fixed early termination.
    """
    # Check trivial cases
    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        print("  Start or goal is blocked!")
        return []

    if start == goal:
        return [start]

    # Quick reachability check
    if not get_valid_neighbors(grid, start):
        print("  Start position has no valid neighbors!")
        return []

    if not get_valid_neighbors(grid, goal):
        print("  Goal position has no valid neighbors!")
        return []

    # Shared memory
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()
    best_cost = Value('d', float('inf'))
    stop_flag = Value(ctypes.c_bool, False)

    # Start workers
    p_fwd = ctx.Process(
        target=bidirectional_worker,
        args=(grid, start, goal, 'fwd', result_queue, best_cost, stop_flag)
    )
    p_bwd = ctx.Process(
        target=bidirectional_worker,
        args=(grid, start, goal, 'bwd', result_queue, best_cost, stop_flag)
    )

    p_fwd.start()
    p_bwd.start()

    # Coordinator
    g_fwd: Dict[Tuple[int, int], float] = {start: 0.0}
    g_bwd: Dict[Tuple[int, int], float] = {goal: 0.0}
    parents_fwd: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    parents_bwd: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {goal: None}

    best_meeting: Optional[Tuple[int, int]] = None
    completed_workers = 0
    start_time = time.time()
    last_print = time.time()

    TIMEOUT = 60.0
    fwd_nodes = 0
    bwd_nodes = 0

    # Main coordination loop
    while completed_workers < 2:
        try:
            msg = result_queue.get(timeout=0.2)

            if msg[0] == 'update':
                _, direction, nodes, processed = msg

                # Update progress
                if direction == 'fwd':
                    fwd_nodes = processed
                else:
                    bwd_nodes = processed

                # Merge update
                if direction == 'fwd':
                    for pos, g, parent in nodes:
                        if pos not in g_fwd or g < g_fwd[pos]:
                            g_fwd[pos] = g
                            parents_fwd[pos] = parent
                else:
                    for pos, g, parent in nodes:
                        if pos not in g_bwd or g < g_bwd[pos]:
                            g_bwd[pos] = g
                            parents_bwd[pos] = parent

                # Check for meeting
                new_meeting, new_cost = find_meeting_point(g_fwd, g_bwd, best_cost.value)
                if new_meeting is not None:
                    if best_meeting is None:
                        best_meeting = new_meeting
                        best_cost.value = new_cost

            elif msg[0] == 'complete':
                _, direction, final_data = msg



                # Merge final data
                if direction == 'fwd':
                    g_fwd.update(final_data['g_costs'])
                    parents_fwd.update(final_data['parents'])
                else:
                    g_bwd.update(final_data['g_costs'])
                    parents_bwd.update(final_data['parents'])

                completed_workers += 1

                # Final meeting check
                new_meeting, new_cost = find_meeting_point(g_fwd, g_bwd, best_cost.value)
                if new_meeting is not None:
                    best_meeting = new_meeting
                    best_cost.value = new_cost

            elif msg[0] == 'error':
                _, direction, error_msg = msg
                print(f"  ERROR in {direction} worker: {error_msg}")
                completed_workers += 1

        except:
            # Timeout - print progress
            current_time = time.time()
            elapsed = current_time - start_time

            if current_time - last_print > 3.0:
                if best_meeting:
                    last_print = current_time

            # Timeout check
            if elapsed > TIMEOUT:
                print(f"  Timeout after {elapsed:.1f}s - stopping search")
                stop_flag.value = True
                break

            # Check if both processes died
            if not p_fwd.is_alive() and not p_bwd.is_alive():
                print("  Both workers stopped")
                completed_workers = 2

    # Signal stop
    stop_flag.value = True

    # Clean up processes
    for process, name in [(p_fwd, 'forward'), (p_bwd, 'backward')]:
        if process.is_alive():
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)



    # Reconstruct path
    if best_meeting is not None:

        # Forward path
        path_fwd = []
        current = best_meeting
        safety = 0
        while current is not None and safety < 1000000:
            path_fwd.append(current)
            current = parents_fwd.get(current)
            safety += 1

        if safety >= 1000000:
            print("  ERROR: Forward path reconstruction failed (cycle detected)")
            return []

        path_fwd.reverse()

        # Backward path
        path_bwd = []
        current = parents_bwd.get(best_meeting)
        safety = 0
        while current is not None and safety < 1000000:
            path_bwd.append(current)
            current = parents_bwd.get(current)
            safety += 1

        if safety >= 1000000:
            print("  ERROR: Backward path reconstruction failed (cycle detected)")
            return []

        final_path = path_fwd + path_bwd
        return final_path

    print("  No path exists between start and goal")
    return []