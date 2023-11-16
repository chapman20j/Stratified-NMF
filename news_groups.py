# news_groups.py
"""
This file runs Stratified-NMF on the 20 news groups dataset.
"""

import os
import timeit

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from scipy.sparse.linalg import norm as sparse_norm

from datasets import get_20_newsgroups
from stratified_nmf import stratified_nmf
from save_load import save_experiment


def top_v_newsgroup_words(
    V: np.ndarray,
    feature_names: np.ndarray,
    num_words: int,
) -> np.ndarray:
    """Gets the words with highest value in V for each stratum

    Args:
        V: List of vectors (strata features)
        feature_names: Array of words
        num_words: Number of words to return

    Returns:
        Two dimensional array of the top words for each stratum
    """
    max_nonzero_in_group = np.max(np.count_nonzero(V, axis=1))
    if num_words > max_nonzero_in_group:
        print("num_words exceeds the maximum of non-zero entries in V(i) for some i.")
        print(f"Using num_words = {max_nonzero_in_group} instead")
    top_indices = np.argsort(V, axis=1)[:, -num_words:]

    top_words = [feature_names[top_indices[i]] for i in range(V.shape[0])]
    return np.array(top_words)


def news_groups_experiment(
    *,
    rank: int = 20,
    iterations: int = 100,
):
    """Runs Stratified-NMF for the 20 news groups dataset and saves the result.
    The function also prints the running time and the top words for each stratum.

    Args:
        rank: Rank of W's and H. Defaults to 20.
        iterations: Iterations to run Stratified-NMF. Defaults to 100.
    """

    data, features = get_20_newsgroups()

    for d in data:
        assert isinstance(d, csr_array)

    start = timeit.default_timer()
    V, W, H, loss_array = stratified_nmf(data, rank, iterations, calculate_loss=False)
    end = timeit.default_timer()
    print("Time: ", end - start)
    print("Done")

    top_words = top_v_newsgroup_words(V, features, num_words=30)

    for i in range(V.shape[0]):
        print(top_words[i])

    # Save data
    params_dict = {
        "dataset": "news_groups",
        "rows": [W[i].shape[0] for i in range(len(W))],
        "cols": H.shape[1],
        "rank": rank,
        "strata": len(W),
        "iterations": iterations,
        "v_scaling": 2,
    }
    data_dict = {
        "A_norm_0": sum([sparse_norm(A_s) ** 2 for A_s in data]) ** 0.5,
        "loss": loss_array,
    }
    folder_name = save_experiment(
        params_dict=params_dict,
        data_dict=data_dict,
        experiment_name="news_groups",
    )

    # Save the list of top_words
    df = pd.DataFrame(top_words)
    df.to_csv(os.path.join(folder_name, "top_words.csv"))


if __name__ == "__main__":
    news_groups_experiment()
