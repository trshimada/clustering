import pytest
import numpy as np
from fec import FluxEquilibriumClustering


def test_two_clusters():
    """Test FEC on two well-separated clusters."""
    np.random.seed(2)

    n = 16
    k = 6

    # Create two well-separated clusters
    cluster1 = np.random.normal([3, 3], 0.8, (int(n / 2), 2))
    cluster2 = np.random.normal([-3, -3], 0.8, (int(n / 2), 2))
    X = np.vstack([cluster1, cluster2])

    # Test with aggressive parameters
    fec = FluxEquilibriumClustering(k=k, alpha=0.7, T=20, epsilon=1e-5, beta=None)

    labels = fec.fit_predict(X)
    unique_labels, counts = np.unique(labels, return_counts=True)

    assert fec.w_ij.shape == (n, k)
    assert fec.flux.shape == (n,)
    assert fec.sinks.shape == (n,)
    assert fec.labels.shape == (n,)

    assert len(unique_labels) == 2

    expected_flux = np.array([0, 0, 8, 0, 0, 0, 0, 0, 0.0243, 0.0151])
    assert np.all(np.isclose(fec.flux[: len(expected_flux)], expected_flux, atol=0.01))
