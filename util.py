"""Util module final

Provides utility functions for dtree, including determining split criterion, information gain and more
"""

from math import inf, log2
import random
from typing import Tuple
import matplotlib.pyplot as plt

import numpy as np
from sting.data import Feature, FeatureType
from typing import List

def determine_split_criterion(
        schema: List[Feature],
        ignore: List,
        X: np.ndarray,
        y: np.ndarray,
        use_gain_ratio: bool):
    '''
    Determines attribute to partition dataset on.
    If continuous attribute is chosen, also returns pivot value

    Args:
        schema: list of features
        ignore: list of features indexes and/or values that
            have already been used higher in the tree
        X: data
        y: labels
        use_gain_ratio: boolean value determining if
            gain ratio should be used, if not: information gain is used
    '''
    max_information_gain = -inf
    max_feature_idx = -1
    feature_pivot = None

    ignore = sorted(ignore)

    for i, feature in enumerate(schema):
        if len(ignore) > 0 and i == ignore[0]:
            del ignore[0]
            continue

        information_gain = None
        pivot = None

        if feature.ftype == FeatureType.CONTINUOUS:
            information_gain, pivot = continuous_information_gain(
                i, X, y, use_gain_ratio)
        elif feature.ftype == FeatureType.NOMINAL or feature.ftype == FeatureType.BINARY:
            information_gain = discrete_information_gain(
                i, X, y, use_gain_ratio)

        else:
            print("Unhandled Feature Type")
            continue

        # Check to see if the best split has positive information gain
        if information_gain == 0:
            continue

        # update max information gain
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            max_feature_idx = i
            feature_pivot = pivot

    return max_feature_idx, feature_pivot

def calc_entropy_val(p: int):
    return 0 if p == 0 else -p * log2(p)

def calc_entropy(total: int, num_1: int):
    '''
    calculates entropy given total number of samples, and number of positive samples (value == 1)

    Args:
        total: total number of entries
        num_1: number of values that equal 1

    Returns:
        entropy
    '''
    if total == 0 or num_1 == 0 or num_1 == total:
        return 0

    # Get probabilities for each class
    p_1 = num_1 / total
    p_0 = 1 - p_1

    # H(Y) = -p1*log2(p1) - p0*log2(p0)
    return -p_1 * log2(p_1) - p_0 * log2(p_0)

def calc_entropy_arr(y: np.ndarray):
    """
    calculates entropy of array y
    """
    return calc_entropy(len(y), (y == 1).sum())

def continuous_information_gain(
        feature_idx: int,
        X: np.ndarray,
        y: np.ndarray,
        use_gain_ratio: bool):
    ''' Returns information gain (or gain ratio) of continuous feature of index [feature_idx] from X and y'''
    # get length of input and entropy of y before partition
    num = len(y)
    h_y = calc_entropy_arr(y)

    # sort X and y by continuous feature x
    idxs = X[:, feature_idx]
    idxs_sorted = idxs.argsort()

    y = y[idxs_sorted]
    X = X[idxs_sorted, feature_idx]

    # find the pivots
    pivot_idxs = np.where(y[:-1] != y[1:])[0] + 1

    if len(pivot_idxs) == 0:
        return 0

    # # H(Y|X) = sum_x(H(Y|X=x) * p(X=x)) --> find information_gain of all of the pivots
    if use_gain_ratio:
        information_gain = [(h_y - (calc_entropy_arr(y[:idx]) * idx + calc_entropy_arr(y[idx:]) * (num - idx)) / num) / calc_entropy(num, idx) for idx in pivot_idxs]
    else:
        information_gain = [h_y - (calc_entropy_arr(y[:idx]) * idx + calc_entropy_arr(y[idx:]) * (num - idx)) / num for idx in pivot_idxs]

    # find index of max information gain
    max_info_gain_idx = np.argmax(information_gain)
    max_info_gain = information_gain[max_info_gain_idx]
    pivot_idx = pivot_idxs[max_info_gain_idx]

    return max_info_gain, (X[pivot_idx - 1] + X[pivot_idx]) / 2

def discrete_information_gain(
        feature_idx: int,
        X: np.ndarray,
        y: np.ndarray,
        use_gain_ratio: bool):
    '''Returns information gain (or gain ratio) of discrete feature of index [feature_idx] from X and y'''
    num = len(y)
    h_y = calc_entropy_arr(y)
    vals, idxs, counts = np.unique(X[:, feature_idx], return_inverse=True, return_counts=True)

    if len(counts) <= 1:
        return 0

    h_yx = sum([calc_entropy_arr(y[idxs == i]) * cnt / num for i, cnt in enumerate(counts)])
    ig = h_y - h_yx
    if use_gain_ratio:
        ig /= sum([calc_entropy_val(cnt / num) for cnt in counts])

    return ig

def cv_split(
    X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Conducts a cross-validation split on the given data.

    Args:
        X: Data of shape (n_examples, n_features)
        y: Labels of shape (n_examples,)
        folds: Number of CV folds
        stratified:

    Returns: A tuple containing the training data, training labels, testing data, and testing labels, respectively
    for each fold.

    For example, 5 fold cross validation would return the following:
    (
        (X_train_1, y_train_1, X_test_1, y_test_1),
        (X_train_2, y_train_2, X_test_2, y_test_2),
        (X_train_3, y_train_3, X_test_3, y_test_3),
        (X_train_4, y_train_4, X_test_4, y_test_4),
        (X_train_5, y_train_5, X_test_5, y_test_5)
    )

    """

    if folds <= 1:
        return (X, y, X, y),

    # Set the RNG seed to 12345 to ensure repeatability
    np.random.seed(12345)
    random.seed(12345)

    idxs = []
    if stratified:
        # Fold 1's and 0's separately
        folds_0 = fold_data(np.where(y == 0)[0], folds)
        folds_1 = fold_data(np.where(y == 1)[0], folds)
        # Generate training/test index sets, combining corresponding 1/0 folds
        idxs = [[np.concatenate(folds_0[0:i] + folds_0[i + 1:] + folds_1[0:i] + folds_1[i + 1:]),
                 np.concatenate((folds_0[i], folds_1[i]))] for i in range(folds)]
    else:
        # Fold all the data together
        data_folds = fold_data(np.arange(y.shape[0]), folds)
        # Generate training/test sets
        idxs = [[np.concatenate(
            data_folds[0:i] + data_folds[i + 1:]), data_folds[i]] for i in range(folds)]

    # Retrieve actual data values
    return [[X[idx[0]], y[idx[0]], X[idx[1]], y[idx[1]]] for idx in idxs]

def fold_data(data: np.ndarray, folds: int):
    '''Returns partitions of data given number of folds'''
    num = data.shape[0]

    # Get a random order of data values
    choice = data[np.random.choice(range(num), size=(num), replace=False)]
    # Compute the size of each fold and the remainder elements
    fold_size = int(num / folds)
    rem = num - fold_size * folds

    # Partition the random data
    return [choice[fold_size * i + min(rem,
                                       i): fold_size * (i + 1) + min(rem,
                                                                     i + 1)] for i in range(folds)]
