# plot.py
"""
Code to plot the results of the Stratified-NMF experiments.
Running this file plots the results of each experiment and saves the figures in the Figures folder.
"""

import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

from save_load import load_experiment

orig_news_group_names = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]

pretty_news_group_names = [
    "atheism",
    "graphics",
    "misc computer",
    "pc hardware",
    "mac hardware",
    "windows x",
    "forsale",
    "autos",
    "motorcycles",
    "baseball",
    "hockey",
    "cryptography",
    "electronics",
    "medicine",
    "space",
    "christian",
    "politics guns",
    "politics mideast",
    "politics misc",
    "religion misc",
]


def nice_plot_synthetic(
    exp_name: str,
    *,
    save_fig: bool = True,
    show_fig: bool = False,
):
    """Plots the results of the synthetic experiment.
    Includes plots of:
        Normalized Loss
        Mean of v's per strata in each iteration
        Loglog plot of normalized loss

    Args:
        exp_name: Name of the experiment to plot.
        save_fig: Whether to save figures. Defaults to True.
        show_fig: Whether to show figures. Defaults to False.
    """
    # Constants
    folder_name = os.path.join("Results", exp_name)
    savefig_name = os.path.join("Figures", exp_name)

    # Matplotlib options
    sns.set_style("whitegrid")
    rcParams["axes.edgecolor"] = "0.6"
    rcParams["figure.figsize"] = [6, 4]
    rcParams["font.family"] = "serif"
    rcParams["mathtext.fontset"] = "dejavuserif"
    rcParams["grid.color"] = "0.85"
    rcParams["legend.edgecolor"] = "0.6"
    rcParams["legend.framealpha"] = "1"
    rcParams["legend.frameon"] = True
    rcParams["legend.handlelength"] *= 1.5
    rcParams["xtick.major.pad"] = 3
    rcParams["ytick.major.pad"] = 2

    # Load the data
    params_dict, data = load_experiment(folder_name)
    num_plot_points = params_dict["iterations"]

    plot_x_spacing = -num_plot_points / 100

    # Plot the normalized loss
    normalized_loss = data["loss"] / data["A_norm_0"]
    plt.plot(normalized_loss[:num_plot_points])
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Loss")
    plt.xlim([plot_x_spacing, num_plot_points])
    plt.ylim([0, None])
    plt.tight_layout()
    if save_fig:
        plt.savefig(savefig_name + "_loss.png")
    if show_fig:
        plt.show()
    else:
        plt.clf()

    marker_list = ["D", "x", "o", "^"]
    # Plot the mean of v's per strata in each iteration
    for i in range(data["v_stats"].shape[0]):
        plt.plot(
            data["v_stats"][i, :num_plot_points],
            marker=marker_list[i],
            label=f"Strata {i+1}",
            markevery=num_plot_points // 15,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Mean $v(i)$")
    plt.xlim([plot_x_spacing, num_plot_points])
    plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        mode="expand",
        loc="lower left",
        ncol=4,
    )
    plt.tight_layout()
    if save_fig:
        plt.savefig(savefig_name + "_v_means.png")
    if show_fig:
        plt.show()
    else:
        plt.clf()

    # Log-log plot for normalized loss
    plt.loglog(normalized_loss[:num_plot_points])
    plt.xlabel("Iteration")
    plt.ylabel("Log Normalized Loss")
    plt.tight_layout()
    if save_fig:
        plt.savefig(savefig_name + "_loglog_loss.png")
    if show_fig:
        plt.show()
    else:
        plt.clf()

    # Print out additional details for the paper
    print(f"Final Normalized Loss = {normalized_loss[num_plot_points - 1]}")
    print(f"Final v_means = {data['v_stats'][:,num_plot_points - 1]}")


def nice_plot_california(
    exp_name: str,
    *,
    save_fig: bool = True,
    show_fig: bool = False,
):
    """Plots the results of the california experiment.
    Includes plots of:
        Loss
        Barchart of v's

    Args:
        exp_name: Name of the experiment to plot.
        save_fig: Whether to save figures. Defaults to True.
        show_fig: Whether to show figures. Defaults to False.
    """
    # Constants
    folder_name = os.path.join("Results", exp_name)
    savefig_name = os.path.join("Figures", exp_name)

    # Data
    _, data = load_experiment(folder_name)
    df = pd.read_csv(os.path.join(folder_name, "barchart.csv"), index_col=0)

    # Plot loss
    plt.plot(data["loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.xlim([0, None])
    plt.ylim([0, None])
    plt.tight_layout()
    if save_fig:
        plt.savefig(savefig_name + "_loss.png")
    if show_fig:
        plt.show()
    else:
        plt.clf()

    # Plot barchart
    df.plot.bar()
    plt.ylim(0, 1)
    plt.title("California Housing Dataset $v(i)$'s")
    plt.ylabel("Normalized value")
    plt.tight_layout()
    if save_fig:
        plt.savefig(savefig_name + "_barchart.png")
    if show_fig:
        plt.show()
    else:
        plt.clf()


def nice_plot_mnist(
    exp_name: str,
    *,
    save_fig: bool = True,
    show_fig: bool = False,
):
    """Plots the results of the mnist experiment.
    Includes plots of:
        All learned v's in one figure
        v's in separate images
        All learned h's in one figure
        h's in separate images

    Args:
        exp_name: Name of the experiment to plot.
        save_fig: Whether to save figures. Defaults to True.
        show_fig: Whether to show figures. Defaults to False.
    """
    # Constants
    folder_name = os.path.join("Results", exp_name)
    savefig_name = os.path.join("Figures", exp_name)

    # Data
    _, data = load_experiment(folder_name)

    V = data["V"]
    H = data["H"]

    # Matplotlib options
    cmap = "plasma"

    # Plot the v's on one figure
    _, axs = plt.subplots(1, V.shape[0])
    for s in range(V.shape[0]):
        axs[s].axis("off")
        plt.subplot(1, V.shape[0], s + 1)
        plt.imshow(
            V[s].reshape(28, 28),
            interpolation="nearest",
            cmap=cmap,
        )
    if save_fig:
        plt.savefig(savefig_name + "_V.png")
    if show_fig:
        plt.show()
    else:
        plt.clf()

    # Plot the v's separately
    for s in range(V.shape[0]):
        plt.imshow(
            V[s].reshape(28, 28),
            interpolation="nearest",
            cmap=cmap,
        )
        plt.axis("off")
        plt.tight_layout()

        if save_fig:
            plt.savefig(savefig_name + f"_v({s}).png")
        plt.clf()

    # Plot the h's on one figure
    _, axs = plt.subplots(1, H.shape[0])
    for f in range(H.shape[0]):
        axs[f].axis("off")
        plt.subplot(1, H.shape[0], f + 1)
        plt.imshow(
            H[f].reshape(28, 28),
            interpolation="nearest",
            cmap=cmap,
        )
    if save_fig:
        plt.savefig(savefig_name + "_H.png")
    if show_fig:
        plt.show()
    else:
        plt.clf()

    # Plot the h's separately
    for f in range(H.shape[0]):
        plt.imshow(
            H[f].reshape(28, 28),
            interpolation="nearest",
            cmap=cmap,
        )
        plt.axis("off")
        plt.tight_layout()

        if save_fig:
            plt.savefig(savefig_name + f"_h({f}).png")
        plt.clf()


def nice_plot_news_groups(
    exp_name: str,
    *,
    print_latex: bool = True,
    latex_num_words: int = 3,
):
    """Displays the results of the news groups experiment.
    Prints:
        Top words for each news group
        Latex table of top words for each news group

    Args:
        exp_name: Name of the experiment to display.
        print_latex: Whether to print latex table. Defaults to True.
        latex_num_words: Number of words to show in latex table. Defaults to 3.
    """
    folder_name = os.path.join("Results", exp_name)

    # Data
    # Rows are news group, cols are words
    top_words = pd.read_csv(os.path.join(folder_name, "top_words.csv"), index_col=0)

    print(top_words)

    # get a dataframe with only the rows from groups_overleaf and only first latex_num_words columns
    top_words = top_words.iloc[:, :latex_num_words]

    # print top_words as a latex table
    if print_latex:
        top_words.index = pretty_news_group_names
        print(top_words.style.to_latex())


if __name__ == "__main__":
    # Run all the plotting functions
    nice_plot_synthetic("synthetic")
    nice_plot_california("california")
    nice_plot_mnist("mnist")
    nice_plot_news_groups("news_groups")
