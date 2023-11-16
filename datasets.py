# datasets.py
"""
Code to get datasets and put into the right format for Stratified-NMF.
Running this file just verifies that the output data is non-negative.
It also provides information about the number of words for the 20 newsgroups dataset.
"""

from typing import Any

import numpy as np
import torchvision
from sklearn.datasets import fetch_california_housing, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_array


def get_synthetic_data(
    strata: int,
    rows: int,
    cols: int,
    rank: int,
) -> list[np.ndarray]:
    """Generates synthetic data for stratified NMF.
    Each stratum is created as
        A(i) = v(i) + W(i) H
        v(i) ~ U(i-1, i)
        W(i) ~ U(0, 1)
        H ~ U(0, 1)

    Args:
        strata: Number of strata
        rows: Rows in each stratum
        cols: Number of columns (variables) in each data matrix
        rank: Rank used in the W's and H

    Returns:
        List of matrices with each row shifted by a v(i) drawn from a strata dependent distribution.
    """

    w_true = np.random.rand(strata, rows, rank)
    h_true = np.random.rand(rank, cols)
    v_true = np.random.rand(strata, cols)

    # This gives the shift
    v_true += np.arange(strata).reshape(-1, 1)

    # Construct mat
    mat = [
        np.dot(w_true[i], h_true) + np.outer(np.ones(rows), v_true[i])
        for i in range(strata)
    ]

    return mat


def get_california_housing_dataset() -> list[np.ndarray]:
    """Returns the California housing dataset from sklearn.datasets.
    Stratifies the data into 3 strata based on median income (in units of 10k dollars)
        low income: < 1.5
        medium income: 4.5 < income < 5
        high income: > 10
    The latitude and longitude data is removed because this is non-negative.

    Returns:
        List of matrices with the [low, medium, high] income strata.
    """
    data = fetch_california_housing()

    # Remove latitude and longitude
    data_matrix = data.data[:, :-2]

    low_income = data_matrix[data_matrix[:, 0] < 1.5]
    high_income = data_matrix[data_matrix[:, 0] > 10]
    med_income = data_matrix[(data_matrix[:, 0] > 4.5) & (data_matrix[:, 0] < 5)]
    return [low_income, med_income, high_income]


def get_mnist(
    strata_dict: dict[str, list[Any]],
    strata_size: int,
) -> list[np.ndarray]:
    """Returns a stratified version of a subset of the MNIST dataset from torchvision.
    Where each stratum contains disjoint flattened digits defined by the strata_dict
    and strata_size parameters.

    Args:
        strata_dict: A dictionary with strata name and a list of labels
            ex strata_dict = {"12": [1, 2], "23": [2, 3]}
        strata_size: number of samples in each class that go into a strata
            eg s1 has 3 classes and s2 has 2 classes, then s1 has 300 points and s2 has 200 points

    Returns:
        List of matrices containing the data for each strata.
            Rows of each matrix are flattened images.
    """
    data = torchvision.datasets.MNIST(root="./Datasets", train=True, download=True)

    # First separate by the labels
    class_indices = [np.where(data.targets.numpy() == i)[0] for i in range(10)]

    # Now get the subset indices
    strata_indices = []
    for s_ind, val in enumerate(strata_dict.values()):
        strata_indices.append(
            np.concatenate(
                [
                    class_indices[i][s_ind * strata_size : (s_ind + 1) * strata_size]
                    for i in val
                ]
            )
        )

    return [
        data.data.numpy()[s_inds, ...].reshape(len(s_inds), -1)
        for s_inds in strata_indices
    ]


def get_20_newsgroups() -> tuple[list[np.ndarray], np.ndarray]:
    """Gets the 20 newsgroups dataset from sklearn.datasets.
    Stratifies the data into 20 strata based on the target labels.
    The text is preprocessed by removing headers, footers, and quotes.
    The data is then vectorized using sklearn.feature_extraction.text.TfidfVectorizer
    where the minimum document frequency is 2, meaning that a word must appear in at
    least 2 documents and the maximum document frequency is 0.95 meaning that each word
    must appear in less than 95% of the documents.

    Returns:
        List of matrices containing the data for each strata.
        Numpy array of the string names corresponding to each column of the data matrices.
    """

    data = fetch_20newsgroups(
        subset="all",
        shuffle=False,
        random_state=1,
        remove=("headers", "footers", "quotes"),
    )

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
    # data matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(data.data)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    strata = [csr_array(tfidf_matrix[data.target == i]) for i in range(20)]

    return strata, feature_names


if __name__ == "__main__":
    # Code to check non-negativity of the data

    # Synthetic
    synthetic_data = get_synthetic_data(10, 100, 100, 10)
    for s in synthetic_data:
        assert np.all(s >= 0), "Negative values in synthetic data"

    # California housing
    california_data = get_california_housing_dataset()
    for s in california_data:
        assert np.all(s >= 0), "Negative values in california housing data"

    # MNIST
    mnist_data = get_mnist(strata_dict={"12": [1, 2], "23": [2, 3]}, strata_size=100)
    for s in mnist_data:
        assert np.all(s >= 0), "Negative values in mnist data"

    # 20 newsgroups
    news_data, news_feature_names = get_20_newsgroups()
    for s in news_data:
        assert s.min() >= 0, "Negative values in 20 newsgroups data"

    print(f"The feature names array has shape: {news_feature_names.shape}")
