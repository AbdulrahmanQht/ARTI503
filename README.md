# ARTI 503
Accelerating A* Pathfinding on Multi-Core Architectures Through Parallelism Using Python MultiProcessing Library
## Overview
This project implements and benchmarks three optimized parallel versions of the A* pathfinding algorithm using Python's `multiprocessing` library. By leveraging multi-core architectures, we target specific performance bottlenecks—throughput, latency, and scalability—making the solution suitable for applications ranging from real-time game AI to large-scale robotic navigation.

## Features
- **Multi-Strategy Parallelism**: Implements Data Parallelism, Shared-State Parallelism, and Domain Decomposition to handle different workload types.
- **Optimized Memory Management**: Utilizes Shared Memory (`multiprocessing.Array`) and lock-free reading patterns to minimize overhead.
- **Robust Profiling**: Includes a custom benchmarking suite to measure Speedup, Efficiency, IPC Overhead, and Lock Contention.
- **Scalability**: Capable of handling massive grids (up to 5000x5000) efficiently.

## Algorithms
- Sequential A* Pathfinding Algorithm: Standard implementation using a binary heap priority queue. Serves as the performance baseline.
- Parallel Batch Processing A* Pathfinding Algorithm: (Data Parallelism) Distributes multiple independent pathfinding requests across available CPU cores for high throughput.
- Bidirectional A* Pathfinding Algorithm: (Shared-State Parallelism) Two processes search simultaneously from Start and Goal, utilizing "Lazy Synchronization" to reduce memory contention.
- Region-Based A* Pathfinding Algorithm: (Domain Decomposition) Decomposes the grid into smaller sub-regions to solve local paths in parallel, ideal for extremely large maps.

## Installation
#### To set up the project locally, follow these steps:
1. Clone the repository:
 ```bash
   git clone https://github.com/AbdulrahmanQht/ARTI503
   ```
2. Navigate to the project directory:
```bash
   cd ARTI503
   ```
3. Install the required libraries:
```bash
   pip install -r requirements.txt
   ```