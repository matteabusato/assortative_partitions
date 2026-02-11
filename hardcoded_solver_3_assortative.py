import itertools
import numpy as np
import networkx as nx

def compute_m_vector(partition, N):
    m = [0, 0, 0]
    for i in range(3):
        m[i] = np.sum(partition == i) / N
    return np.array(m)

def is_valid_partition(partition, H):
    group_counts = np.bincount(partition, minlength=3)
    return np.sum(group_counts) >= H

def find_solutions(N, D, H, M):
    counter = 0
    all_partitions = list(itertools.product(range(3), repeat=N))
    print(len(all_partitions))
    for partition in all_partitions:
        partition = np.array(partition)
        if np.allclose(compute_m_vector(partition, N), M) and is_valid_partition(partition, H):
            counter += 1

    return counter

def random_d_regular_adjacency(N: int, D: int):
    if not (0 <= D < N):
        raise ValueError("Need 0 <= D < N.")
    if (N * D) % 2 != 0:
        raise ValueError("Need N*D even (handshaking lemma).")

    G = nx.random_regular_graph(d=D, n=N)
    A = nx.to_numpy_array(G, dtype=np.int8)
    A = (A > 0).astype(np.int8)
    return A

if __name__ == "__main__":
    N = 15  # Number of nodes
    D = 7   # Average degree
    graph = random_d_regular_adjacency(N, D)

    H = 3
    M = np.array([1/3, 1/3, 1/3])
    n_solutions = find_solutions(N, D, H, M)
    print(f"Number of valid partitions: {n_solutions}")
