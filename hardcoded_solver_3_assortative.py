import itertools
import numpy as np

def compute_m_vector(partition, N):
    m = []
    for i in range(3):
        m_i = np.sum(partition == i) / N
        m.append(m_i)
    return np.array(m)

def is_valid_partition(partition, D, H):
    # Check if the partition is valid based on the degree and number of friends
    group_counts = np.bincount(partition, minlength=3)
    return np.all(group_counts <= D) and np.sum(group_counts) == H

def find_solutions(N, D, H, M):
    counter = 0
    all_partitions = list(itertools.product(range(3), repeat=N))
    print(all_partitions)
    for partition in all_partitions:
        if np.allclose(compute_m_vector(partition, N), M) and is_valid_partition(partition, D, H):
            counter += 1

    return counter

def generate_random_graph(N, D):
    # Generate a random graph with N nodes and average degree D, return adjancency matrix
    p = D / (N - 1)  # Probability of edge creation
    graph = np.random.rand(N, N) < p  # Adjacency matrix
    np.fill_diagonal(graph, 0)  # No self-loops
    print(f"Generated random graph with {N} nodes and average degree {D}.")
    print(f"Actual average degree: {graph.sum() / N:.2f}")
    print(f"Adjacency matrix:\n{graph.astype(int)}")
    return graph.astype(int)


if __name__ == "__main__":
    N = 3  # Number of nodes
    D = 2   # Average degree
    graph = generate_random_graph(N, D)

    H = 1
    M = np.array([1/3, 1/3, 1/3])
    n_solutions = find_solutions(N, D, H, M)
