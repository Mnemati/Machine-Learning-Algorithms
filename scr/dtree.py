"""
author: Mo Nemati, Ibrahim Berber
"""

from __future__ import annotations
import argparse
import os.path
import warnings
import scipy
import pandas as pd

from IPython.display import display
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, parse_c45, FeatureType

import util

from logger import get_handler
import logging

# import numba

handler = get_handler()

log = logging.getLogger(__name__)
log.handlers[:] = []
log.addHandler(handler)
log.setLevel(logging.INFO)


def set_log_level(level: int | str) -> None:
    """
    Set the log level.
    """
    if level is None:
        level = logging.DEBUG
    elif level == "DEBUG" or level == 5:
        level = logging.DEBUG
    elif level == "INFO" or level == 4:
        level = logging.INFO
    elif level == "WARNING" or level == 3:
        level = logging.WARNING
    elif level == "ERROR" or level == 2:
        level = logging.ERROR
    elif level == "CRITICAL" or level == 1:
        level = logging.CRITICAL
    elif level == 0:
        level = logging.NOTSET
    else:
        raise ValueError("Invalid level")

    log.setLevel(level)


class Node:
    """
    Node of a decision tree.
    """

    def __init__(self, X=None, y=None, label=None, level=0, is_leaf=False, is_continuous=False, split_point=None):
        log.debug(f"CREATING NODE WITH LEVEL {level}")
        self.X = X
        self.y = y
        self.label = label
        self.children: Dict[Node] = {}
        self.level = level
        self.decision_class = None
        self.is_leaf = is_leaf

        self.split_point = split_point
        self.is_continuous = is_continuous

        self.criteria = None
        self.best_attribute_index = None

    def predict(self):
        return self.decision_class

    def add_child_node(self, child_node: Node, route):
        self.children[route] = child_node

    def get_decision(self):
        pass

    def __repr__(self):
        return f"NODE: " \
               f"  DECISION: {self.decision_class}" \
               f"  LEVEL: {self.level}" \
               f"  IS_LEAF: {self.is_leaf}" \
               f"  IS_CONTINUOUS: {self.is_continuous}" \
               f"  BEST ATTRIBUTE INDEX: {self.best_attribute_index}" \
               f"  CRITERIA: {self.criteria}"

    def set_best_attribute_index(self, best_attribute_index):
        self.best_attribute_index = best_attribute_index

    def set_criteria(self, criteria, ftype):
        self.criteria = criteria
        if len(criteria) == 1:
            [self.split_point] = criteria
            if ftype == FeatureType.CONTINUOUS:
                self.is_continuous = True


# In Python, the convention for class names is CamelCase, just like Java! However, the convention for method and
# variable names is lowercase_separated_by_underscores, unlike Java.
class DecisionTree(Classifier):
    def __init__(
            self,
            schema: List[Feature],
            max_depth=None,
            information_gain=False,
            use_less_critical_values=None,
            verbose=None,
            print_tree=False
    ):
        """
        This is the class where you will implement your decision tree. At the moment, we have provided some dummy code
        where this is simply a majority classifier in order to give you an idea of how the interface works. Don't forget
        to use all the good programming skills you learned in 132 and utilize numpy optimizations wherever possible.
        Good luck!
        """

        set_log_level(verbose)
        log.info('Initializing Decision Tree ..')

        # warnings.warn('The DecisionTree class is currently running dummy Majority Classifier code. ' +
        #               'Once you start implementing your decision tree delete this warning message.')

        self._schema = schema  # For some models (like a decision tree) it makes sense to keep track of the data schema
        self._majority_label = 0  # Protected attributes in Python have an underscore prefix

        self._root = None

        if max_depth == 0:
            self.max_depth = None
        else:
            self.max_depth = max_depth

        self.depth = None

        self._n_zero = None
        self._n_one = None
        self._features = None

        self._n_instances = None
        self._n_features = None
        self._attributes_remaining = None
        self.first_feature = None

        self.n_nodes = None

        self.use_less_critical_values = use_less_critical_values
        self.information_gain = information_gain

        self.display_tree_config()
        self.print_tree = print_tree

    def display_tree_config(self):
        log.info("DECISION TREE CONFIG:")
        log.info(f"MAX DEPTH: {'NO LIMIT' if self.max_depth is None else self.max_depth}")
        log.info(f"DEPTH: {self.depth}")
        log.info(f"INFORMATION GAIN: {'INFO_GAIN' if self.information_gain else 'GAIN_RATIO'}")
        log.info(f"USE LESS CRITICAL VALUES: {self.use_less_critical_values}")

    def display_tree_properties(self):
        log.info("DECISION TREE PROPERTIES:")
        log.info(f"# OF FEATURES: {self._n_features}")
        log.info(f"# OF NODES: {self.n_nodes}")
        log.info(f"DEPTH: {self.max_depth}")
        log.info(f"INFORMATION GAIN: {'INFO_GAIN' if self.information_gain else 'GAIN_RATIO'}")

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """
        This is the method where the training algorithm will run.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example. Will become relevant later in the course, ignore for now.
        """

        # In Java, it is best practice to LBYL (Look Before You Leap), i.e. check to see if code will throw an exception
        # BEFORE running it. In Python, the dominant paradigm is EAFP (Easier to Ask Forgiveness than Permission), where
        # try/except blocks (like try/catch blocks) are commonly used to catch expected exceptions and deal with them.
        # try:
        #     split_criterion = self._determine_split_criterion(X, y)
        # except NotImplementedError:
        #     warnings.warn('This is for demonstration purposes only.')

        log.debug("Fitting the Decision Tree ..")



        n_zero, n_one = util.count_label_occurrences(y)

        if n_one > n_zero:
            self._majority_label = 1
        else:
            self._majority_label = 0

        self._n_instances, self._n_features = X.shape

        self._prepare_attributes_remaining(X, y)

        self._root = Node(X, y, level=0)
        self.depth = 0
        self.n_nodes = 1
        queue = [self._root]
        self._build_decision_tree(queue)
        log.info("FIT COMPLETED.")
        if self.print_tree:
            self.traverse_tree()

    def _prepare_attributes_remaining(self, X, y):
        log.debug("Preparing the attributes remaining..")
        self._attributes_remaining = {}
        for feature_index in range(self._n_features):
            if self._schema[feature_index].ftype == FeatureType.NOMINAL:
                unique_values = util.get_unique_values_nominal_enum(schema=self._schema, feature_index=feature_index)
                self._attributes_remaining[feature_index] = unique_values

            elif self._schema[feature_index].ftype == FeatureType.BINARY:
                raise NotImplementedError  # TODO

            elif self._schema[feature_index].ftype == FeatureType.CONTINUOUS:
                critical_points = self._find_critical_points_for_continuous(X[:, feature_index], y)
                self._attributes_remaining[feature_index] = critical_points

            else:
                raise

    @staticmethod
    def are_all_values_same(y):
        """
        Check if all values in y are same.
        """
        return np.all(y == y[0])

    @staticmethod
    def _are_all_values_1(y):
        # check all values are 1
        return np.all(y == 1)

    @staticmethod
    def _are_all_values_0(y):
        # check all values are 0
        return np.all(y == 0)

    @staticmethod
    def _get_majority(y):
        """
        Get the majority class. If they are equal, randomly choose one.
        """
        ones = np.sum(y)
        zeros = len(y) - ones
        if ones > zeros:
            return 1
        elif zeros > ones:
            return 0
        else:
            return np.random.choice([0, 1])

    def _build_decision_tree(self, queue: List[Node] = None):

        log.info(f"QUEUE HAS {len(queue)} ELEMENTS IN IT. LEVELS: {[n.level for n in queue]}")
        if len(queue) == 0:
            log.info("QUEUE BECOMES EMPTY. TERMINATING..")
            return

        node = queue.pop(0)

        X, y = node.X, node.y

        log.debug(f"[RECURSIVE CALL] Building decision tree is called (from level {node.level - 1})")

        # display(pd.DataFrame(np.c_[X, y]))

        # log.debug(f"ATTRIBUTES REMAINING: {self._attributes_remaining}")

        # If all instances are positive, then return a single-node tree Root with label +
        if self._are_all_values_1(y):
            log.info("All instances are positive, thus terminating the process.")
            node.is_leaf = True
            node.decision_class = 1

        # If all instances are negative, then return a single-node tree Root with label -
        if self._are_all_values_0(y):
            log.info("All instances are negative, thus terminating the process.")
            node.is_leaf = True
            node.decision_class = 0

        # If there is no attributes, then return a single-node tree Root with
        # label = most common value.
        if len(self._attributes_remaining.keys()) == 0:
            log.info("There are no features, returning most common value.")
            node.is_leaf = True
            node.decision_class = self._get_majority(y)
            for node_on_the_queue in queue:
                node_on_the_queue.is_leaf = True
                node_on_the_queue.decision_class = self._get_majority(y)
            return

        if node.is_leaf:
            log.info("* * * become leaf * * * ")

        else:
            # Stopping criteria max_depth :
            if self.max_depth is not None and self.max_depth <= node.level:
                node.is_leaf = True
                node.decision_class = self._get_majority(y)

                for node_on_the_queue in queue:
                    if node_on_the_queue.level >= self.max_depth:
                        node_on_the_queue.is_leaf = True
                        node_on_the_queue.decision_class = self._get_majority(y)

                log.info("Decision tree has reached to the maximum depth limit.")
                return

            best_attribute_index, criteria, ftype = self._find_best_attribute(X, y, self.information_gain)
            node.set_best_attribute_index(best_attribute_index)
            node.set_criteria(criteria, ftype)


            # If NOMINAL or BINARY attribute contains only one type, we should not create any further node.
            if ftype in [FeatureType.NOMINAL or FeatureType.BINARY] and len(criteria) == 1:
                node.is_leaf = True
                log.info("- - ATTRIBUTE REMAINS?? TERMINATION.")
                for node_on_the_queue in queue:
                    node_on_the_queue.is_leaf = True
                    node_on_the_queue.decision_class = self._get_majority(y)

                return

            filtered_datasets = self.filter_data(X, y, criteria, ftype, best_attribute_index)

            for filtered_dataset in filtered_datasets:
                X_filtered, y_filtered, criterion = filtered_dataset
                # Adding child
                child_node = Node(X=X_filtered, y=y_filtered, level=node.level + 1)
                node.add_child_node(child_node, criterion)
                queue.append(child_node)
                self.n_nodes += 1
                self.depth = max(self.depth, node.level + 1)

        return self._build_decision_tree(queue=queue)

    def filter_data(
            self,
            X,
            y,
            criteria,
            ftype,
            best_attribute_index
    ) -> Tuple[List[Tuple[Any, Any, str]], Any] | List[Tuple[Any, Any, Any]]:

        if ftype == FeatureType.CONTINUOUS:
            # unpack a single value contained in 'criteria' (plural).
            [c] = criteria
            c = round(c, 3)
            filtering_mask_1 = X[:, best_attribute_index] <= c
            filtering_mask_2 = X[:, best_attribute_index] > c

            X_left = X[filtering_mask_1]
            y_left = y[filtering_mask_1]

            X_right = X[filtering_mask_2]
            y_right = y[filtering_mask_2]

            return [(X_left, y_left, f"<"), (X_right, y_right, f">")]

        elif ftype in [FeatureType.BINARY, FeatureType.NOMINAL]:
            datasets = []
            for criterion in criteria:
                filtering_mask = X[:, best_attribute_index] == criterion
                X_filtered = X[filtering_mask]
                y_filtered = y[filtering_mask]
                datasets.append((X_filtered, y_filtered, criterion))

            return datasets

        else:
            raise ValueError("........")

    def traverse_tree(self):
        visited = set()
        self._traverse_tree(visited, self._root)

    def _traverse_tree(self, visited, node):  # function for dfs
        if node not in visited:
            indent = node.level
            indent_text = " " * indent * 10 + "|___"
            print(f"{indent_text}{node}")
            visited.add(node)
            for route, child_node in node.children.items():
                self._traverse_tree(visited, child_node)

    def _predict(self, node, entry):
        if node.is_leaf:
            return node.predict()

        else:
            if node.is_continuous:
                val = entry[node.best_attribute_index]
                if val >= node.split_point:
                    return self._predict(node.children[">"], entry)

                else:
                    return self._predict(node.children["<"], entry)

            else:
                return self._predict(node.children[entry[node.best_attribute_index]], entry)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This is the method where the decision tree is evaluated.

        Args:
            X: The testing data of shape (n_examples, n_features).

        Returns: Predictions of shape (n_examples,), either 0 or 1
        """

        # Returns either all 1s or all 0s, depending on _majority_label.
        # return np.ones(X.shape[0], dtype=np.int) * self._majority_label

        y_predicted = []
        for e in X:
            y_predicted.append(self._predict(self._root, e))

        return np.array(y_predicted)

    def _info_gain(self, y, feature):
        """ This function takes in all the featues and compute information gain of
        the each feature"""
        return self._entropy(y) - self._conditional_entropy(feature, y)

    def _gain_ratio(self, y, feature):
        return self._info_gain(y, feature) / self._entropy(feature)

    def _information_selection_mode(self, y, feature, information_gain):
        if information_gain:
            return self._info_gain(y, feature)

        else:
            return self._gain_ratio(y, feature)

    def _entropy(self, feature):
        """ This function takes in a feature and caluclates the entropy
        args:

        returns: Entropy of the target variable
        """
        vc = pd.DataFrame(feature).value_counts(normalize=True)
        return -(vc * np.log(vc) / np.log(2)).sum()

    def _conditional_entropy(self, x: np.ndarray, y):
        '''This calculates the conditional entropy of a specific feature given the target variable'''

        n_samples = len(x)
        unique_cat = np.unique(x)  # this finds the different categories of each feature
        entropy = 0
        for cat in unique_cat:
            mask = x == cat  # indexing the rows that the value of the feature is equal to a specific cateogry
            y_temp = y[mask]  # target variable corresponsing to that specific category
            entropy += (sum(mask) / n_samples) * self._entropy(
                y_temp)  # calculating the entropy related to a specific category
        return entropy

    def _find_best_attribute(self, X, y, information_gain) -> (int, List, FeatureType):
        """

        """
        log.debug("Finding the best attribute..")
        max_ig = float("-inf")
        best_index = -1
        criteria_values = None
        best_ftype = None
        for feature_index in sorted(self._attributes_remaining.keys()):
            log.debug(f"\tChecking Feature index {feature_index}")

            # handling information gain separately for continuous and categorical variables
            if self._schema[feature_index].ftype == FeatureType.NOMINAL:
                ig = self._information_selection_mode(y, X[:, feature_index], information_gain)
                if ig > max_ig:
                    max_ig = ig
                    best_index = feature_index
                    best_ftype = self._schema[feature_index].ftype
                    criteria_values = util.get_unique_values_nominal_enum(schema=self._schema, feature_index=feature_index)
                    log.debug(f"\t\tFeature index {feature_index} becomes best selection ({best_ftype}) "
                              f"with criteria_values of {criteria_values}")

            elif self._schema[feature_index].ftype == FeatureType.BINARY:
                raise NotImplementedError()

            elif self._schema[feature_index].ftype == FeatureType.CONTINUOUS:
                # If it is continuous, we call self.attributes_remaining to access the critical
                # values of the continuous feature

                # creating a list of all possible critical points for split in continuous feature
                critical_values = self._attributes_remaining[feature_index]
                # this for loop is calculating the entropy for each of the splitting points
                if self.use_less_critical_values is not None:
                    critical_values = util.choose_less_values(critical_values, self.use_less_critical_values)

                for value in critical_values:  # vectorize: here
                    # creating a binary temporary feature that represents
                    # all the values less than value by a 1 and 0 otherwise
                    x_temp = X[:, feature_index] <= value
                    ig = self._information_selection_mode(y, x_temp, information_gain)

                    if ig > max_ig:
                        max_ig = ig
                        best_index = feature_index
                        criteria_values = [value]
                        best_ftype = self._schema[feature_index].ftype
                        log.debug(f"\t\tFeature index {feature_index} becomes best selection ({best_ftype}) "
                                  f"with criteria_values of {criteria_values}")

        log.info(f"BEST ATTRIBUTE INDEX: {best_index} ")
        if self.first_feature is None:
            log.info(f"SETTING THE FIRST FEATURE: {best_index}")
            self.first_feature = best_index

        # If best feature type is a NOMINAL or BINARY feature, then we discard that feature index all together,
        # so that we don't use it anymore.
        if best_ftype in [FeatureType.NOMINAL, FeatureType.BINARY]:
            log.info(f"DELETING THE FEATURE INDEX {best_index} FROM `ATTRIBUTES_REMAINING`")
            del self._attributes_remaining[best_index]

        # For CONTINUOUS feature type, we remove ONLY ONE VALUE. That means, this feature is going to be available
        # for us to pick with its remaining values.
        else:
            [value] = criteria_values
            log.info(f"REMOVING {value} for values of FEATURE INDEX {best_index} from `ATTRIBUTES_REMAINING`.")
            self._attributes_remaining[best_index].remove(value)
            # if we ran out of values, then we can cross of this feature index, too.
            if len(self._attributes_remaining[best_index]) == 0:
                log.info(f"We ran out of values for feature index {best_index}, so crossing it off, too.")
                del self._attributes_remaining[best_index]

        return best_index, criteria_values, best_ftype

    # In Python, instead of getters and setters we have properties: docs.python.org/3/library/functions.html#property
    @property
    def schema(self):
        """
        Returns: The dataset schema
        """
        return self._schema

    # It is standard practice to prepend helper methods with an underscore "_" to mark them as protected.
    def _determine_split_criterion(self, X: np.ndarray, y: np.ndarray):
        """
        Determine decision tree split criterion. This is just an example to encourage you to use helper methods.
        Implement this however you like!
        """
        raise NotImplementedError()

    @staticmethod
    # find the values where y changes
    def _find_critical_points_for_continuous(a, b):
        assert a.ndim == 1
        c = np.c_[a, b]
        c = c[c[:, 0].argsort()]

        critical_points = set()
        for i in range(len(c)):
            if c[i - 1, 1] != c[i, 1]:
                critical_points.add((c[i - 1, 0] + c[i, 0]) / 2)

        return critical_points


def evaluate_and_print_metrics(dtree: DecisionTree, X: np.ndarray, y: np.ndarray):
    """
    You will implement this method.
    Given a trained decision tree and labelled dataset, Evaluate the tree and print metrics.
    """

    y_hat = dtree.predict(X)
    acc = util.accuracy(y, y_hat)
    print(f'Accuracy: {acc:.2f}')
    print(f'Size: {dtree.n_nodes}')
    print(f'Maximum (constructed) Depth: {dtree.depth}')
    print(f'First Feature: {dtree.schema[dtree.first_feature].name} (index: {dtree.first_feature})')


def dtree(
        data_path: str,
        tree_depth_limit: int,
        use_cross_validation: bool = True,
        information_gain: bool = True,
        use_less_critical_values=None,
        verbose=None,
        print_tree=False,
):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook. This function has been PARTIALLY implemented for you, but not completely!

    :param data_path: The path to the data.
    :param tree_depth_limit: Depth limit of the decision tree
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
    :return:
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    if schema[0].name == "image_id":
        del schema[0]
        X = X[:, 1:]

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    for X_train, y_train, X_test, y_test in datasets:
        decision_tree = DecisionTree(
            schema,
            max_depth=tree_depth_limit,
            information_gain=information_gain,
            use_less_critical_values=use_less_critical_values,
            verbose=verbose,
            print_tree=print_tree
        )
        decision_tree.fit(X_train, y_train)
        evaluate_and_print_metrics(decision_tree, X_test, y_test)


if __name__ == '__main__':
    """
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here. We have provided argparse code
    for you for this assignment, but in the future you may be responsible for doing this yourself.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('depth_limit', metavar='DEPTH', type=int,
                        help='Depth limit of the tree. Must be a non-negative integer. A value of 0 sets no limit.')
    parser.add_argument('--use-less-critical-values', dest='lc', default=None, type=str,
                        help='Uses less critical splitting values for continuous attributes. Please pass following '
                             'values: all, log, loglog, sqrt, half')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--use-gain-ratio', dest='gain_ratio', action='store_true',
                        help='Use gain ratio as tree split criterion instead of information gain.')
    parser.add_argument('--print-tree', dest='print_tree', default=False, action='store_true',
                        help='Prints the tree by traversing the nodes.')
    parser.add_argument('--verbose', dest='verbose', type=int, default=0,
                        help='Showing the detail steps with levels [0-5]. Lower is the more detailed. Default is 0.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    # If the depth limit is negative throw an exception
    if args.depth_limit < 0:
        raise argparse.ArgumentTypeError('Tree depth limit must be non-negative.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    max_depth = args.depth_limit
    use_cross_validation = args.cv
    use_information_gain = not args.gain_ratio
    use_less_critical_values = args.lc
    print_tree = args.print_tree
    verbose = args.verbose

    dtree(
        data_path=data_path,
        tree_depth_limit=max_depth,
        use_cross_validation=use_cross_validation,
        information_gain=use_information_gain,
        use_less_critical_values=use_less_critical_values,
        verbose=verbose,
        print_tree=print_tree,
    )
