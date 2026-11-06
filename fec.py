from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
            self,
            k: int = 10,
            sigma: Optional[float] = None,
            alpha: float = 0.5,
            T: int = 25,
            epsilon: float = 1e-4,
            beta: Optional[float] = None,
    ) -> None:
        # Assign parameters to instance attributes
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels``.
        """
        # Hint: Start with computing centroid = X.mean(axis=0)

        # Get number of data points (n) and dimensions (m)
        n, m = X.shape
        effective_k = min(self.k, n - 1)

        # Compute centroid and distances from each point to the centroid
        centroid = np.zeros(m)
        for j in range(m):
            centroid[j] = np.mean(X[:, j])

        d = np.zeros(n)
        for i in range(n):
            d[i] = np.linalg.norm(X[i] - centroid)

        # Compute distance matrix
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])

        # Determine sigma if None
        if self.sigma is None:
            pairwise_distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairwise_distances.append(distance_matrix[i, j])

            if len(pairwise_distances) > 0:
                sigma = 0.2 * np.median(np.array(pairwise_distances))
            else:
                sigma = 1.0
        else:
            sigma = self.sigma

        # Build k nearest neighbors graph and Gaussian edge weights (w_ij)
        w_ij = np.zeros((n, effective_k))
        neighbor_indices = np.zeros((n, effective_k), dtype=int)

        for i in range(n):
            # Distances to all other nodes
            other_distances = []
            for j in range(n):
                if i != j:
                    other_distances.append((distance_matrix[i, j], j))
                else:
                    other_distances.append((np.inf, j))

            # Sort by distance and pick the k neighbors
            other_distances.sort(key=lambda x: x[0])

            for idx in range(effective_k):
                dist_val, neighbor_node = other_distances[idx]
                neighbor_indices[i, idx] = neighbor_node

                # Calculate Gaussian kernel weight
                dist_sq = dist_val ** 2
                w_ij[i, idx] = np.exp(-dist_sq / (sigma ** 2))

        # Initialize flux
        flux = np.ones(n)
        tau = 0.1 * sigma
        delta_T = np.zeros((n, effective_k))

        # Simulate flow over T iterations
        for t in range(self.T):
            phi = np.ones(n)
            if t > 0:
                if self.beta == 0:
                    phi = np.ones(n)
                elif self.beta is None:
                    phi = np.log(1 + flux)
                else:
                    phi = flux ** self.beta

            # Compute edge weights (w_tilde)
            w_tilde = np.zeros((n, effective_k))
            for i in range(n):
                for idx in range(effective_k):
                    j = neighbor_indices[i, idx]
                    w_tilde[i, idx] = w_ij[i, idx] * phi[j]

            # Distribute flux along edges
            delta = np.zeros((n, effective_k))
            outgoing = np.zeros(n)

            for i in range(n):
                # Find valid downhill neighbors
                valid_weights_sum = 0.0
                valid_flags = np.zeros(effective_k, dtype=bool)

                for idx in range(effective_k):
                    j = neighbor_indices[i, idx]
                    if d[j] <= d[i] + tau:
                        valid_flags[idx] = True
                        valid_weights_sum += w_tilde[i, idx]

                # Send flux if valid downhill neighbors exist
                if valid_weights_sum > 0:
                    for idx in range(effective_k):
                        if valid_flags[idx]:
                            delta[i, idx] = self.alpha * w_tilde[i, idx] * (flux[i] / valid_weights_sum)

                for idx in range(effective_k):
                    outgoing[i] += delta[i, idx]

            if t == self.T - 1:
                delta_T = delta.copy()

            # Update flux for next iteration
            new_flux = np.copy(flux)
            for i in range(n):
                new_flux[i] -= outgoing[i]
                for idx in range(effective_k):
                    j = neighbor_indices[i, idx]
                    new_flux[j] += delta[i, idx]
            flux = new_flux

        # Detect Sinks (nodes with outgoing flux < epsilon)
        outgoing_flux_T = np.zeros(n)
        for i in range(n):
            for idx in range(effective_k):
                outgoing_flux_T[i] += delta_T[i, idx]

        sinks = np.zeros(n, dtype=int)
        for i in range(n):
            if outgoing_flux_T[i] < self.epsilon:
                sinks[i] = 1

        # If no sinks are found, pick the node closest to the centroid
        sink_indices = []
        for i in range(n):
            if sinks[i] == 1:
                sink_indices.append(i)

                if len(sink_indices) == 0:
                    closest_idx = int(np.argmin(d))
                    sinks[closest_idx] = 1
                    sink_indices = [closest_idx]

                # Assign Clusters
                labels = -np.ones(n, dtype=int)
                for cluster_id, sink_node in enumerate(sink_indices):
                    labels[sink_node] = cluster_id

                for i in range(n):
                    if labels[i] != -1:
                        continue

                    current = i
                    path = [current]
                    visited = {current}

                    while labels[current] == -1:
                        best_j = -1
                        max_delta = -1.0

                        for idx in range(effective_k):
                            j = neighbor_indices[current, idx]
                            if d[j] <= d[current] + tau:
                                if delta_T[current, idx] > max_delta:
                                    max_delta = delta_T[current, idx]
                                    best_j = j

                        # If no downhill neighbor or zero delta, break and make new sink/cluster
                        if best_j == -1 or max_delta == 0:
                            new_cluster_id = len(sink_indices)
                            labels[current] = new_cluster_id
                            break

                        # If we visit a node twice, break cycle
                        if best_j in visited:
                            new_cluster_id = len(sink_indices)
                            labels[current] = new_cluster_id
                            break

                        current = best_j
                        path.append(current)
                        visited.add(current)

                        if labels[current] != -1:
                            term_label = labels[current]
                            for node in path:
                                labels[node] = term_label
                            break

                    term_label = labels[current]
                    for node in path:
                        labels[node] = term_label

                # Small cluster merging
                s_min = max(2, n // 20)
                unique_labels = np.unique(labels)

                cluster_sizes = {}
                for lbl in unique_labels:
                    cluster_sizes[lbl] = np.sum(labels == lbl)

                small_clusters = [lbl for lbl, size in cluster_sizes.items() if size < s_min]
                large_clusters = [lbl for lbl, size in cluster_sizes.items() if size >= s_min]

                if len(large_clusters) > 0:
                    for small_lbl in small_clusters:
                        # Compute centroid of the small cluster
                        small_points = X[labels == small_lbl]
                        mu_small = small_points.mean(axis=0)

                        best_large_lbl = large_clusters[0]
                        min_dist = np.inf

                        for large_lbl in large_clusters:
                            large_points = X[labels == large_lbl]
                            mu_large = large_points.mean(axis=0)
                            dist_centroids = np.linalg.norm(mu_small - mu_large)
                            if dist_centroids < min_dist:
                                min_dist = dist_centroids
                                best_large_lbl = large_lbl

                        # Merge small cluster into the nearest large cluster
                        labels[labels == small_lbl] = best_large_lbl

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"
