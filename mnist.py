# mnist.py
"""
This file runs Stratified-NMF on the mnist dataset.
"""

import numpy as np

from datasets import get_mnist
from stratified_nmf import stratified_nmf
from save_load import save_experiment


def mnist_experiment(
    strata_dict,
    *,
    rank: int = 5,
    strata_size: int = 100,
    iterations: int = 100,
):
    """Runs stratified NMF on the mnist dataset and saves the result.

    Args:
        strata_dict: Dictionary of strata to use (based on digit label).
            Refer to datasets.get_mnist for more information.
        rank: Rank to use for W's and H. Defaults to 5.
        strata_size: Number of images from each class/strata. Defaults to 100.
            Refer to datasets.get_mnist for more information.
        iterations: Iterations to run Stratified-NMF. Defaults to 100.
    """
    data = get_mnist(strata_dict, strata_size)

    V, W, H, loss_array = stratified_nmf(data, rank, iterations)

    # Save data
    params_dict = {
        "dataset": "mnist",
        "rows": [W[i].shape[0] for i in range(len(W))],
        "cols": H.shape[1],
        "rank": rank,
        "strata": len(W),
        "iterations": iterations,
        "v_scaling": 2,
    }
    data_dict = {
        "A_norm_0": sum([np.linalg.norm(A_s) ** 2 for A_s in data]) ** 0.5,
        "loss": loss_array,
        "V": V,
        "H": H,
    }
    save_experiment(
        params_dict=params_dict,
        data_dict=data_dict,
        experiment_name="mnist",
    )


if __name__ == "__main__":
    # Run the experiment
    mnist_experiment({"12": [1, 2], "23": [2, 3]})
