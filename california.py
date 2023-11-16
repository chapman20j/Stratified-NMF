# california.py
"""
This file runs Stratified-NMF on the california housing dataset.
"""

import os

import pandas as pd
import numpy as np

from save_load import save_experiment
from datasets import get_california_housing_dataset
from stratified_nmf import stratified_nmf


def california_experiment(
    *,
    rank: int = 5,
    iterations: int = 100,
):
    """Runs Stratified-NMF on the california housing dataset and saves the result.

    Args:
        rank: Rank to use for W's and H. Defaults to 5.
        iterations: Iterations to run Stratified-NMF. Defaults to 100.
    """
    # Generate random data
    A = get_california_housing_dataset()

    strata: int = len(A)

    # Run Stratified NMF
    V, _, _, loss_array = stratified_nmf(A, rank, iterations)

    # Save data
    params_dict = {
        "dataset": "california",
        "rows": [len(a) for a in A],
        "cols": A[0].shape[1],
        "rank": rank,
        "strata": strata,
        "iterations": iterations,
        "v_scaling": 2,
    }
    data_dict = {
        "A_norm_0": sum([np.linalg.norm(A[s]) ** 2 for s in range(strata)]) ** 0.5,
        "loss": loss_array,
    }
    folder_name = save_experiment(
        params_dict=params_dict,
        data_dict=data_dict,
        experiment_name="california",
    )

    # Create a barchart for the V matrix and save it
    field_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
    ]
    strata_names = ["low", "medium", "high"]

    # We remove the ave_occup because they come out to nearly 0
    barchart_v = V.copy()[:, :-1]
    barchart_v /= np.sum(barchart_v, axis=0)

    df = pd.DataFrame(
        barchart_v,
        index=strata_names,
        columns=field_names,
    )

    df = df.T
    # Save the dataframe
    df.to_csv(os.path.join(folder_name, "barchart.csv"))


if __name__ == "__main__":
    # Run the experiment
    california_experiment()
