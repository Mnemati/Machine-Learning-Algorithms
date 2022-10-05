import random
import warnings
from typing import Tuple, Iterable
import numpy as np

"""
This is where you will implement helper functions and utility code which you will reuse from project to project.
Feel free to edit the parameters if necessary or if it makes it more convenient.
Make sure you read the instruction clearly to know whether you have to implement a function for a specific assignment.
"""


# def get_unique_values(v: np.ndarray) -> set:
#     """
#     Returns the unique values in a vector.
#     Args:
#         v: np.ndarray
#             The vector (one column).
#
#     Returns:
#         Unique values in it.
#
#     """
#     return set(np.unique(v))


def get_unique_values_nominal_enum(schema, feature_index, return_kind="values"):
    """
    It becomes a hassle to get the unique values... :
    """
    if return_kind == "values":
        mapping_list = [{value._name_: value._value_} for value in schema[feature_index].values]
        values = [list(d.values())[0] for d in mapping_list]
        return values

    else:
        raise NotImplementedError()


def count_label_occurrences(y: np.ndarray) -> Tuple[int, int]:
    """
    This is a simple example of a helpful helper method you may decide to implement. Simply takes an array of labels and
    counts the number of positive and negative labels.

    HINT: Maybe a method like this is useful for calculating more complicated things like entropy!

    Args:
        y: Array of binary labels.

    Returns: A tuple containing the number of negative occurrences, and number of positive occurences, respectively.

    """
    n_ones = (y == 1).sum()  # How does this work? What does (y == 1) return?
    n_zeros = y.size - n_ones
    return n_zeros, n_ones



#def cv_split(
#        X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False, n_fold = 5) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
#    """
#    Conducts a cross-validation split on the given data.
#
#    Args:
#        X: Data of shape (n_examples, n_features)
#        y: Labels of shape (n_examples,)
#        folds: Number of CV folds
#        stratified:
#
#    Returns: A tuple containing the training data, training labels, testing data, and testing labels, respectively
#    for each fold.
#
#    For example, 5-fold cross validation would return the following:
#    (
#        (X_train_1, y_train_1, X_test_1, y_test_1),
#        (X_train_2, y_train_2, X_test_2, y_test_2),
#        (X_train_3, y_train_3, X_test_3, y_test_3),
#        (X_train_4, y_train_4, X_test_4, y_test_4),
#        (X_train_5, y_train_5, X_test_5, y_test_5)
#    )
#
#    """
#
#    # Set the RNG seed to 12345 to ensure repeatability
#
#    np.random.seed(12345)
#    random.seed(12345)
#    proportion = np.sum(y)/len(y) # this is the proportion of ones in the target variable
#    fold_size = len(y) / n_fold # this is the size of each fold
#    last_fold_size = len(y) - (n_fold - 1) * fold_size
#
#    for fold in range(1,n_fold,1):
#        print(fold)
#
#
#
#    # HINT!
#    if stratified:
#        n_zeros, n_ones = count_label_occurrences(y)
#
#    warnings.warn('cv_split is not yet implemented. Simply returning the entire dataset as a single fold...')
#
#    return (X, y, X, y),


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Another example of a helper method. Implement the rest yourself!

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Accuracy
    """

    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')

    n = y.size

    return (y == y_hat).sum() / n


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    return np.sum(y * y_hat)/np.sum(y_hat)


def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')
    return np.sum(y * y_hat)/np.sum(y)


def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray) -> Iterable[Tuple[float, float]]:
    """
    Calculate the ROC curve pairs.
    """
    assert len(y) == len(p_y_hat)
    assert np.all((y == 0) | (y == 1))
    assert np.all((p_y_hat >= 0) & (p_y_hat <= 1))

    # sort the values
    c = np.c_[y, p_y_hat]
    c = c[c[:, 1].argsort()[::-1]]

    # calculate the true positive rate and false positive rate
    tpr = np.cumsum(c[:, 0]) / np.sum(c[:, 0])
    fpr = np.cumsum(1 - c[:, 0]) / np.sum(1 - c[:, 0])

    return np.c_[fpr, tpr]


def auc(y: np.ndarray, p_y_hat: np.ndarray) -> float:
    """
    Calculate the ROC AUC.
    """
    assert len(y) == len(p_y_hat)
    assert np.all((y == 0) | (y == 1))
    assert np.all((p_y_hat >= 0) & (p_y_hat <= 1))

    # calculate the roc curve pairs
    roc_curve_pairs_ = roc_curve_pairs(y, p_y_hat)

    # calculate the area under the curve
    auc = 0
    for i in range(1, len(roc_curve_pairs_)):
        auc += (roc_curve_pairs_[i, 0] - roc_curve_pairs_[i - 1, 0]) * roc_curve_pairs_[i, 1]

    return auc


def cv_split(X, y, folds, stratified=True):
    """
    Split the data into k folds while keeping the ratio with respect to the ratio of classes in target variable.
    """
    np.random.seed(12345)  # specifying a seed
    random.seed(12345)
    assert X.shape[0] == y.shape[0]
    assert folds > 1
    if stratified:
        # create a list of indices
        indices = np.arange(X.shape[0])
        # shuffle the indices
        np.random.shuffle(indices)  # shuffling the indices
        # create a list of folds
        folds = np.array_split(indices, folds)
        # create a list of train and test indices
        for fold in folds:
            train_indices = np.concatenate([f for f in folds if f is not fold])
            test_indices = fold
            # create the train and test sets
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            # print out the ratio of 0 and 1 in y_train and y_test and y
            # normalize the values
            # print(np.unique(y_train, return_counts=True)[1] / len(y_train))
            # print(np.unique(y_test, return_counts=True)[1] / len(y_test))
            # print(np.unique(y, return_counts=True)[1] / len(y))
            # yield the train and test sets
            yield X_train, y_train, X_test, y_test

    else:
        n = X.shape[0]

        # shuffle the data
        idx = np.random.permutation(n)
        X = X[idx]
        y = y[idx]

        # split the data
        fold_size = n // folds
        for i in range(folds):
            start = i * fold_size
            end = start + fold_size

            # split the data
            X_train = np.concatenate((X[:start], X[end:]), axis=0)
            y_train = np.concatenate((y[:start], y[end:]), axis=0)

            X_test = X[start:end]
            y_test = y[start:end]

            yield X_train, y_train, X_test, y_test


def choose_less_values(values, kind=None):
    """
    Choose sqrt of length values by default. This function chooses a portion of critical splitting points of continuous variabel
    to find the best splitting criteria instead of considering all the splitting point at each step

    """
    if kind is None:
        kind = "sqrt"

    values = list(values)
    if kind == "sqrt":
        n = int(np.sqrt(len(values)))
    elif kind == "log":
        n = int(np.log2(len(values)))
    elif kind == "loglog":
        n = int(np.log2(np.log2(len(values))))
    elif kind == "half":
        n = int(len(values) / 2)
    elif kind == "all":
        n = len(values)
    else:
        raise ValueError(f"Invalid kind. {kind=}")
    return np.random.choice(values, size=n, replace=False)
