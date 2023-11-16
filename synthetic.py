# synthetic.py
"""
This file runs Stratified-NMF on the synthetic dataset.
It additionally keeps track of the mean of the v's in each stratum.
"""

import numpy as np
from tqdm import trange

from save_load import save_experiment
from datasets import get_synthetic_data
from stratified_nmf import update_V, update_W, update_H, loss


def synthetic_experiment(
    *,
    rows: int = 100,
    cols: int = 100,
    rank: int = 5,
    strata: int = 4,
    iterations: int = 10000,
    v_scaling: int = 2,
):
    """Runs Stratified-NMF on the synthetic dataset and keeps track of the mean of the v's.

    Args:
        rows: Rows in each A(i). Defaults to 100.
        cols: Columns in each A(i). Defaults to 100.
        rank: Rank to use for W's and H. Defaults to 5.
        strata: Number of strata. Defaults to 4.
        iterations: Number of iterations to run. Defaults to 10000.
        v_scaling: Number of v updates each iteration. Defaults to 2.
    """

    # Generate random data
    A = get_synthetic_data(strata, rows, cols, rank)

    # Initialize W, H, V
    W = [np.random.rand(rows, rank) / rank**0.5 for _ in range(strata)]
    H = np.random.rand(rank, cols) / rank**0.5
    V = np.random.rand(strata, cols)

    loss_array = np.zeros(iterations)
    v_stats = np.zeros((strata, iterations))

    # Run Stratified NMF
    for i in trange(iterations):

        # Calculate loss
        loss_array[i] = loss(A, V, W, H)
        v_sizes = np.mean(V, axis=1)
        v_stats[:, i] = v_sizes

        # Update V
        for _ in range(v_scaling):
            V = update_V(A, V, W, H)

        # Update W, H
        W, H = update_W(A, V, W, H), update_H(A, V, W, H)

    assert np.all(V >= 0) and np.all(H >= 0)
    for s in range(strata):
        assert np.all(W[s] >= 0)

    print(f"Final loss = {loss_array[-1]}")
    print("Final v means:")
    for s in range(strata):
        print(round(v_sizes[s], 2))

    # Save data
    params_dict = {
        "dataset": "synthetic",
        "rows": [rows for _ in range(strata)],
        "cols": cols,
        "rank": rank,
        "strata": strata,
        "iterations": iterations,
        "v_scaling": v_scaling,
    }
    data_dict = {
        "A_norm_0": sum([np.linalg.norm(A[s]) ** 2 for s in range(strata)]) ** 0.5,
        "loss": loss_array,
        "v_stats": v_stats,
    }
    save_experiment(
        params_dict=params_dict,
        data_dict=data_dict,
        experiment_name="synthetic",
    )


if __name__ == "__main__":
    synthetic_experiment()
