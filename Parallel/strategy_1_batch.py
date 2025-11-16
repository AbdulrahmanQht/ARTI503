# Parallel/strategy_1_batch.py
import multiprocessing
import numpy as np
from typing import List, Tuple
from Astar import find_path

def find_path_wrapper(args) -> List[Tuple[int, int]]:
    grid, start, goal = args
    return find_path(grid, start, goal)

def strategy_1_parallel_batch(jobs: List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    with multiprocessing.Pool() as pool:
        results = pool.map(find_path_wrapper, jobs)
    return results