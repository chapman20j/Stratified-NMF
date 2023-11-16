# utils.py
"""
This file contains functions for saving and loading experiments.
It also checks that the parameters are valid when saving. 
"""

import os
import json
from typing import Any, Callable, Union

import numpy as np

ReqParamType = dict[str, Callable[[Any], bool]]
DataDictType = dict[str, Union[np.ndarray, np.float64]]
LoadExpType = tuple[dict[str, Any], dict[str, np.ndarray]]


# Dictionary used to check that the parameters are included and are valid
# dict[param_name] = function which checks if the parameter is valid
_REQUIRED_PARAMS: ReqParamType = {
    "dataset": lambda x: x in ["synthetic", "california", "mnist", "news_groups"],
    "rows": lambda x: isinstance(x, list)
    and all([isinstance(i, int) for i in x])
    and all([i > 0 for i in x]),
    "cols": lambda x: isinstance(x, int) and x > 0,
    "rank": lambda x: isinstance(x, int) and x > 0,
    "strata": lambda x: isinstance(x, int) and x > 0,
    "iterations": lambda x: isinstance(x, int) and x > 0,
    "v_scaling": lambda x: isinstance(x, int) and x > 0,
}

_RECOMMENDED_DATA: list[str] = [
    "A_norm_0",  # norm of A
    "loss",  # loss over iterations
]


def save_experiment(
    params_dict: dict[str, Any],
    data_dict: DataDictType,
    experiment_name: str,
) -> str:
    """saves experiment data and parameters to Results/experiment_name folder

    Args:
        params_dict: dictionary of parameter names and their corresponding values
            to be saved in params.json file, enforcing that the required parameters are
            saved. e.g. {'dataset': 'synthetic', 'v_scaling': 2}
        data_dict: dictionary of experiment result name and their corresponding values
            examples include loss and A_norm. eg. {'loss': [5,4,3,2]}
        experiment_name: name of the folder to save params and data to in Results folder
            e.g. 'synthetic_experiment' so that the save folder is
            Results/synthetic_experiment

    Raises:
        KeyError: Missing parameter in params_dict
        ValueError: Invalid parameter in params_dict

    Returns:
        String containing the folder name
    """
    # Create folder
    folder_name = os.path.join("Results", experiment_name)
    if not os.path.exists(folder_name):
        print(f"Making folder: {folder_name}")
        os.mkdir(folder_name)

    # Check that all parameters are valid and exist
    for param, is_valid in _REQUIRED_PARAMS.items():
        if param not in params_dict:
            raise KeyError(f"Missing parameter: {param}")
        if not is_valid(params_dict[param]):
            raise ValueError(f"Invalid parameter: {param} -> {params_dict[param]}")

    # Check the recommended data and suggest it
    for data in _RECOMMENDED_DATA:
        if data not in data_dict:
            print(f"Missing data: {data}. We suggest saving this variable.")

    # Save parameters
    with open(os.path.join(folder_name, "params.json"), "w") as f:
        json.dump(params_dict, f)

    # Save data
    np.savez(os.path.join(folder_name, "data.npz"), **data_dict)

    return folder_name


def load_experiment(folder_name: str) -> LoadExpType:
    """Loads the experiment from the folder name

    Args:
        folder_name: name of the folder to load from

    Returns:
        params_dict: parameters used in the experiment
        data_dict: data produced by the experiment

    """
    # Load parameters
    with open(os.path.join(folder_name, "params.json"), "r") as f:
        params_dict = json.load(f)

    # Load data
    data_dict = np.load(os.path.join(folder_name, "data.npz"))

    return params_dict, data_dict
