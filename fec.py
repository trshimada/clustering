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

        # make sure the input is a 2D float array with at least one row
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a 2D array with at least one row")

        n = X.shape[0]
        effective_k = min(self.k, n - 1)

        # centroid and how far each point is from it
        centroid = X.mean(axis=0)

        d = np.zeros(n)
        for i in range(n):
            d[i] = np.linalg.norm(X[i] - centroid)

        # dist[i][j] = distance between points i and j
        # doing a whole row at once
        dist = np.zeros((n, n))
        for i in range(n):
            dist[i] = np.linalg.norm(X - X[i], axis=1)

        # sigma is 0.2 * median pairwise distance unless one was passed in
        if self.sigma is None:
            all_pair_distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    all_pair_distances.append(dist[i][j])

            if len(all_pair_distances) > 0:
                sigma = 0.2 * float(np.median(all_pair_distances))
            else:
                sigma = 1.0

            # if every point is identical sigma ends up 0 and we'd divide by zero
            if sigma <= 0:
                sigma = 1.0
        else:
            sigma = float(self.sigma)

        tau = 0.1 * sigma

        # k nearest neighbors for each point, and the gaussian weight on each edge
        neighbor_indices = np.zeros((n, effective_k), dtype=int)
        w_ij = np.zeros((n, effective_k))

        for i in range(n):
            candidates = []
            for j in range(n):
                if j != i:
                    candidates.append((dist[i][j], j))

            # sort by distance
            candidates.sort(key=lambda pair: pair[0])

            for idx in range(effective_k):
                neighbor_distance, neighbor_point = candidates[idx]
                neighbor_indices[i][idx] = neighbor_point
                w_ij[i][idx] = np.exp(-(neighbor_distance ** 2) / (sigma ** 2))

        # figure out which neighbors are downhill
        is_downhill = np.zeros((n, effective_k), dtype=bool)
        for i in range(n):
            for idx in range(effective_k):
                j = neighbor_indices[i][idx]
                if d[j] <= d[i] + tau:
                    is_downhill[i][idx] = True

        # initialize flux
        flux = np.ones(n)

        # delta[i][idx] = flux i sends to its idx'th neighbor
        # after the loop this is just the last iteration's values
        delta = np.zeros((n, effective_k))

        for t in range(self.T):
            # phi scales each edge by how much flux the neighbor is holding
            # t = 0 and beta = 0 both mean no scaling
            phi = np.ones(n)
            if t > 0:
                if self.beta is None:
                    phi = np.log(1.0 + flux)
                elif self.beta != 0:
                    phi = flux ** self.beta

            delta = np.zeros((n, effective_k))
            new_flux = flux.copy()

            for i in range(n):
                # only counting downhill neighbors
                weight_sum = 0.0
                for idx in range(effective_k):
                    if is_downhill[i][idx]:
                        j = neighbor_indices[i][idx]
                        weight_sum += w_ij[i][idx] * phi[j]

                if weight_sum <= 0:
                    continue

                for idx in range(effective_k):
                    if is_downhill[i][idx]:
                        j = neighbor_indices[i][idx]
                        weight = w_ij[i][idx] * phi[j]
                        amount = self.alpha * weight * flux[i] / weight_sum

                        delta[i][idx] = amount
                        new_flux[i] -= amount
                        new_flux[j] += amount

            flux = new_flux

        # a sink is a node that barely sent anything out on the last iteration
        sinks = np.zeros(n, dtype=bool)
        for i in range(n):
            total_outgoing = delta[i].sum()
            if total_outgoing < self.epsilon:
                sinks[i] = True

        # no sinks, so use whichever point is closest to the centroid
        if not sinks.any():
            sinks[int(np.argmin(d))] = True

        # -1 means this point doesn't have a cluster yet
        labels = -np.ones(n, dtype=int)

        # every sink starts its own cluster
        next_label = 0
        for i in range(n):
            if sinks[i]:
                labels[i] = next_label
                next_label += 1

        # steepest[i] = downhill neighbor that got the most flux from i (-1 if there isn't one)
        steepest = -np.ones(n, dtype=int)
        for i in range(n):
            best_flux = 0.0
            for idx in range(effective_k):
                if is_downhill[i][idx] and delta[i][idx] > best_flux:
                    best_flux = delta[i][idx]
                    steepest[i] = neighbor_indices[i][idx]

        # for each unlabeled point, walk the steepest neighbors until we hit a labeled point, a dead end, or a cycle
        for start in range(n):
            if labels[start] != -1:
                continue

            current = start
            path = [current]
            seen = {current}

            while True:
                next_point = steepest[current]
                # dead end, this node becomes a new sink
                if next_point == -1:
                    labels[current] = next_label
                    final_label = next_label
                    next_label += 1
                    break

                # we'd loop back onto the path, so break the cycle by making this node a sink
                if int(next_point) in seen:
                    labels[current] = next_label
                    final_label = next_label
                    next_label += 1
                    break

                # next point already has a cluster, we're done
                if labels[next_point] != -1:
                    final_label = labels[next_point]
                    break

                current = int(next_point)
                path.append(current)
                seen.add(current)

            # everything on the path gets the same label
            for point in path:
                labels[point] = final_label

        # merge small clusters into the nearest larger cluster
        min_size = max(2, n // 20)

        # count how many points are in each cluster
        cluster_sizes = {}
        for label in labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

        small_clusters = []
        large_clusters = []
        for label in cluster_sizes:
            if cluster_sizes[label] < min_size:
                small_clusters.append(label)
            else:
                large_clusters.append(label)

        # if no cluster is big enough there's nothing to merge into
        if len(large_clusters) > 0:
            for small_label in small_clusters:
                small_centroid = X[labels == small_label].mean(axis=0)

                closest_label = large_clusters[0]
                closest_distance = np.inf
                for large_label in large_clusters:
                    large_centroid = X[labels == large_label].mean(axis=0)
                    gap = np.linalg.norm(small_centroid - large_centroid)
                    if gap < closest_distance:
                        closest_distance = gap
                        closest_label = large_label

                labels[labels == small_label] = closest_label

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
