# Stratified-NMF

This github repository contains the code used to run the experiments in "Stratified-NMF for Heterogenous Data" by James Chapman, Yotam Yaniv, and Deanna Needell.

## How to use this code

To get started, run

`conda env create -f environment.yml`

in the terminal to create a conda environment and install the dependencies. Activate the conda environment with

`conda activate stratified_nmf`

The experiments can be run by running each of the following files:

- `california.py`
- `mnist.py`
- `news_groups.py`
- `synthetic.py`

Note that this will over-write the data in the `Results/` folder. To view the results, run the `plot.py` file and the figures will be saved in the `Figures/` folder. Some information will be displayed in the terminal.

## Additional Information

- `Results/`: Contains the data from the experiments. Experiment parameters are saved in the `params.json` file. The `data.npz` file contains a bunch of info about the experiment. Some experiments contain csv files.
- `datasets.py`: Contains code for pre-processing and loading datasets into the proper format for Stratified-NMF.
- `plot.py`: Contains code for plotting the experiments in `Results/`. Refer to `plot.py` for more details.
- `save_load.py`: Contains code for saving and loading the results of the experiments.
- `stratified_nmf.py`: Contains code for the multiplicative updates and a function for running Stratified-NMF.
