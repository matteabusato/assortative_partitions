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

def is_valid_partition_fast(labels, neigh, H):
    # neigh[i] is array of neighbors of i
    for i in range(len(labels)):
        # count same-group neighbors
        if np.sum(labels[neigh[i]] == labels[i]) < H:
            return False
    return True

def generate_unlabeled_balanced_partitions(N, K=3):
    """
    Balanced partitions with sizes N/K each, but invariant to group label.
    We enforce a canonical labeling to avoid counting permutations:
      - node 0 is always in group 0
      - among remaining nodes, the smallest not in group 0 is forced into group 1
      - group 2 is the rest
    Yields labels array of shape (N,) with values {0,1,2}.
    """
    if K != 3:
        raise NotImplementedError("This canonical generator is written for K=3.")
    if N % 3 != 0:
        raise ValueError("Need N multiple of 3 for M=[1/3,1/3,1/3].")

    s = N // 3
    nodes = list(range(N))

    # Choose group0 of size s, but force node 0 into it to break symmetry
    # so we choose s-1 more from 1..N-1
    for g0_rest in itertools.combinations(range(1, N), s - 1):
        g0 = {0, *g0_rest}
        remaining1 = [v for v in nodes if v not in g0]

        # Force the smallest remaining into group1 to break remaining symmetry
        forced = remaining1[0]
        remaining1_rest = remaining1[1:]

        # choose the rest of group1 (need s-1 more)
        for g1_rest in itertools.combinations(remaining1_rest, s - 1):
            g1 = {forced, *g1_rest}
            # g2 is whatever remains
            g2 = [v for v in remaining1 if v not in g1]

            labels = np.empty(N, dtype=np.int8)
            labels[list(g0)] = 0
            labels[list(g1)] = 1
            labels[g2] = 2
            yield labels

def find_solutions(N, neigh, H):
    counter = 0
    total = 0
    for labels in generate_unlabeled_balanced_partitions(N, K=3):
        total += 1
        if is_valid_partition_fast(labels, neigh, H):
            counter += 1
    return counter, total

def adjacency_lists(A): # list of numpy arrays of neighbors
    return [np.flatnonzero(A[i]) for i in range(A.shape[0])]

def random_d_regular_adjacency(N, D, seed=None):
    if not (0 <= D < N):
        raise ValueError("Need 0 <= D < N.")
    if (N * D) % 2 != 0:
        raise ValueError("Need N*D even (handshaking lemma).")

    G = nx.random_regular_graph(d=D, n=N, seed=seed)
    A = nx.to_numpy_array(G, dtype=np.int8)
    A = (A > 0).astype(np.int8)
    return A

if __name__ == "__main__":
    # K = 3  # Number of groups
    # alpha = 6
    # N = K * alpha  # 18
    # D = 15
    # SEED = np.random.randint(0, 1000000)
    # np.random.seed(SEED)
    # H = 5   # Minimum number of same-group neighbors. Less than D / K to ensure some valid partitions exist.
    # M = np.array([1/3, 1/3, 1/3])

    K = 3  # Number of groups
    alpha = 2
    N = K * alpha  # 6
    D = 4
    SEED = np.random.randint(0, 1000000)
    np.random.seed(SEED)
    H = 1   # Minimum number of same-group neighbors. Less than D / K to ensure some valid partitions exist.
    M = np.array([1/3, 1/3, 1/3])

    graph = random_d_regular_adjacency(N, D, SEED)
    neigh = adjacency_lists(graph)

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

    n_solutions, total_partitions = find_solutions(N, neigh, H)

    wandb.log({
        "N": N,
        "D": D,
        "H": H,
        "n_solutions": n_solutions,
        "total_partitions": total_partitions,
        "fraction": n_solutions / total_partitions if total_partitions > 0 else 0
    })

    wandb.finish()
