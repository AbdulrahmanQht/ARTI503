# Parallel/strategy_3_region.py
import multiprocessing
import heapq
import time
import numpy as np
from typing import List, Tuple, Dict, Set
from Astar import (
    calculate_heuristic, get_valid_neighbors,
    create_node
)


def get_region(pos: Tuple[int, int], grid_shape: Tuple[int, int], num_regions: int = 4) -> int:
    """
    Divide grid into regions (2x2 = 4 regions by default).
    Region layout:
    [0][1]
    [2][3]
    """
    x, y = pos
    rows, cols = grid_shape

    # Determine which half (vertical and horizontal)
    row_half = 0 if x < rows / 2 else 1
    col_half = 0 if y < cols / 2 else 1

    return row_half * 2 + col_half


def region_worker(
        worker_id: int,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        inbox: multiprocessing.Queue,
        outboxes: List[multiprocessing.Queue],
        result_queue: multiprocessing.Queue,
        all_done: multiprocessing.Value,
        active_workers: multiprocessing.Value,
        lock: multiprocessing.Lock
):
    """
    Each worker processes nodes in its region.
    Uses local state for speed, communicates via queues.
    """
    NUM_REGIONS = 4
    grid_shape = grid.shape

    # Local state (fast, no IPC overhead)
    open_list = []
    open_dict: Dict[Tuple[int, int], float] = {}  # pos -> f_score
    g_costs: Dict[Tuple[int, int], float] = {}
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed_set: Set[Tuple[int, int]] = set()

    # Track activity
    iterations_without_work = 0
    MAX_IDLE_ITERATIONS = 50

    # If this worker owns the start node, initialize it
    if get_region(start, grid_shape, NUM_REGIONS) == worker_id:
        g_costs[start] = 0.0
        parents[start] = None
        h = calculate_heuristic(start, goal)
        f = h
        heapq.heappush(open_list, (f, start))
        open_dict[start] = f

    is_active = True

    while not all_done.value:
        # Process incoming messages
        messages_received = 0
        try:
            while not inbox.empty() and messages_received < 100:  # Batch process
                msg_type, data = inbox.get_nowait()
                messages_received += 1

                if msg_type == 'node':
                    pos, g, parent_pos = data

                    # Only process if this is a better path
                    if pos not in g_costs or g < g_costs[pos]:
                        g_costs[pos] = g
                        parents[pos] = parent_pos

                        if pos not in closed_set:
                            h = calculate_heuristic(pos, goal)
                            f = g + h

                            # Update or add to open list
                            if pos not in open_dict or f < open_dict[pos]:
                                open_dict[pos] = f
                                heapq.heappush(open_list, (f, pos))

                elif msg_type == 'shutdown':
                    return

        except:
            pass

        # Process local nodes
        processed_local = False

        while open_list and not all_done.value:
            f_score, current_pos = heapq.heappop(open_list)

            # Skip if outdated or already closed
            if current_pos in closed_set:
                continue
            if current_pos in open_dict and open_dict[current_pos] < f_score:
                continue

            # Remove from open dict
            if current_pos in open_dict:
                open_dict.pop(current_pos)

            closed_set.add(current_pos)
            processed_local = True
            iterations_without_work = 0

            # Check if goal reached
            if current_pos == goal:
                # Reconstruct path locally
                path = []
                node = current_pos
                while node is not None:
                    path.append(node)
                    node = parents.get(node)
                path.reverse()

                # Signal completion
                with lock:
                    if not all_done.value:
                        all_done.value = True
                        result_queue.put(path)
                return

            current_g = g_costs[current_pos]

            # Expand neighbors
            for neighbor_pos in get_valid_neighbors(grid, current_pos):
                if neighbor_pos in closed_set:
                    continue

                tentative_g = current_g + calculate_heuristic(current_pos, neighbor_pos)

                # Check if this is a better path
                if neighbor_pos not in g_costs or tentative_g < g_costs[neighbor_pos]:
                    neighbor_region = get_region(neighbor_pos, grid_shape, NUM_REGIONS)

                    if neighbor_region == worker_id:
                        # Process locally
                        g_costs[neighbor_pos] = tentative_g
                        parents[neighbor_pos] = current_pos

                        h = calculate_heuristic(neighbor_pos, goal)
                        f = tentative_g + h

                        if neighbor_pos not in open_dict or f < open_dict[neighbor_pos]:
                            open_dict[neighbor_pos] = f
                            heapq.heappush(open_list, (f, neighbor_pos))
                    else:
                        # Send to owner region
                        try:
                            outboxes[neighbor_region].put(
                                ('node', (neighbor_pos, tentative_g, current_pos)),
                                block=False
                            )
                        except:
                            # Queue full - node will be discovered later
                            pass

            # Don't process too many at once - give time for messages
            break

        # Update activity status
        if not processed_local and messages_received == 0:
            iterations_without_work += 1

            if iterations_without_work >= MAX_IDLE_ITERATIONS:
                if is_active:
                    is_active = False
                    with lock:
                        active_workers.value -= 1

                    # Check if all workers are idle
                    if active_workers.value == 0:
                        with lock:
                            all_done.value = True
                        return

            time.sleep(0.001)  # Brief sleep when idle
        else:
            if not is_active:
                is_active = True
                with lock:
                    active_workers.value += 1
            iterations_without_work = 0


def strategy_3_region_based(grid: np.ndarray, start: Tuple[int, int],
                            goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Region-based parallel A* - divides grid into 4 quadrants.
    Each worker is responsible for exploring nodes in its region.
    """

    # Validate inputs
    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        return []

    if start == goal:
        return [start]

    NUM_PROCESSES = 4

    # Create communication queues (one inbox per worker)
    manager = multiprocessing.Manager()
    inboxes = [manager.Queue(maxsize=5000) for _ in range(NUM_PROCESSES)]
    result_queue = manager.Queue()

    # Shared coordination
    all_done = manager.Value('b', False)
    active_workers = manager.Value('i', NUM_PROCESSES)  # Start with all active
    lock = manager.Lock()

    # Start worker processes
    processes = []
    for i in range(NUM_PROCESSES):
        # Each worker gets its own inbox and can send to all outboxes
        p = multiprocessing.Process(
            target=region_worker,
            args=(i, grid, start, goal, inboxes[i], inboxes,
                  result_queue, all_done, active_workers, lock)
        )
        processes.append(p)
        p.start()

    # Wait for completion or timeout
    start_time = time.time()
    timeout = 30  # 30 second timeout

    for p in processes:
        remaining = timeout - (time.time() - start_time)
        if remaining > 0:
            p.join(timeout=remaining)
        else:
            break

    # Terminate any remaining processes
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)

    # Get result if found
    if not result_queue.empty():
        return result_queue.get()
    else:
        return []