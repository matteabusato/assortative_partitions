import itertools
import numpy as np
import networkx as nx
import math
import wandb

def compute_m_vector(partition, N):
    m = [0, 0, 0]
    for i in range(3):
        m[i] = np.sum(partition == i) / N
    return np.array(m)

def is_valid_partition(partition, N, H, graph):
    for i in range(N):
        counter = 0
        for j in range(N):
            if graph[i, j] == 1 and partition[i] == partition[j]:
                counter += 1
        if counter < H :
            return False
    return True

def find_solutions(K, N, D, H, M, graph):
    counter = 0
    all_partitions = list(itertools.product(range(3), repeat=N))
    total_partitions = int(len(all_partitions) / math.factorial(K)) 
    for partition in all_partitions:
        partition = np.array(partition)
        if np.allclose(compute_m_vector(partition, N), M) and is_valid_partition(partition, N, H, graph):
            counter += 1

    counter /= math.factorial(K) # Normalize by the number of unique partitions
    return int(counter), total_partitions

def random_d_regular_adjacency(N, D):
    if not (0 <= D < N):
        raise ValueError("Need 0 <= D < N.")
    if (N * D) % 2 != 0:
        raise ValueError("Need N*D even (handshaking lemma).")

    G = nx.random_regular_graph(d=D, n=N)
    A = nx.to_numpy_array(G, dtype=np.int8)
    A = (A > 0).astype(np.int8)
    return A

if __name__ == "__main__":
    K = 3  # Number of groups
    alpha = 6
    N = K * alpha  # Number of nodes  (multiple of K!!!!!!)
    D = 15   # Average degreeH
    graph = random_d_regular_adjacency(N, D)

    H = 5   # Minimum number of same-group neighbors. Less than D / K to ensure some valid partitions exist.
    M = np.array([1/3, 1/3, 1/3])

    SEED = np.random.randint(0, 1000000)
    np.random.seed(SEED)

    wandb.init(
        project="hardcoded_assortative",
        name=f"N{N}_D{D}_H{H}_M{M.tolist()}",
        group=f"N{N}_D{D}_H{H}_M{M.tolist()}", 
        config={
            "N": N,
            "D": D,
            "H": H,
            "M": M.tolist(),
            "SEED": SEED
        }
    )

    n_solutions, total_partitions = find_solutions(K, N, D, H, M, graph)

    wandb.log({
        "N": N,
        "D": D,
        "H": H,
        "n_solutions": n_solutions,
        "total_partitions": total_partitions,
    })

    wandb.finish()
