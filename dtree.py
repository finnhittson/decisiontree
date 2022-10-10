"""
dtree provides Decision Tree functionality and training/testing capabilities
"""

from sting.data import Feature, FeatureType, parse_c45
from sting.classifier import Classifier

import time
import numpy as np
import math
import argparse
import os
from typing import List

from util import determine_split_criterion, cv_split


class Node:
    def __init__(
            self,
            num,
            num_ones,
            feature_idx,
            information_gain,
            feature_pivot=None):
        self.num = num
        self.num_ones = num_ones

        self.feature_idx = feature_idx
        self.feature_pivot = feature_pivot
        self.information_gain = information_gain
        self.pruned = False

        self.parent = None
        self.children = []

    @property
    def empty(self):
        return len(self.children) == 0

    def set_children(self, children):
        self.children = children
        for child in self.children:
            child.parent = self

    def partition(self, X):
        return [np.arange(len(X))]

    def evaluate(self, x):
        return None, None

    def get_pruned(self):
        if self.num == 0:
            node = self.parent.get_pruned()
            node.parent = self.parent
            return node

        if self.num_ones >= self.num / 2:
            p_node = PNode(
                self.num,
                self.num_ones,
                1,
                self.num_ones /
                self.num)
        else:
            p_node = PNode(
                self.num,
                self.num_ones,
                0,
                1 -
                self.num_ones /
                self.num,
            )
        p_node.parent = self.parent
        return p_node

    @property
    def pruned_majority(self):
        return 1 if self.num_ones >= self.num / 2 else 0

    @property
    def pruned_confidence(self):
        if self.pruned_majority == 1:
            return self.num_ones / self.num
        return 1 - self.num_ones / self.num

# Discrete (Nominal) Node


class DNode(Node):
    def __init__(
            self,
            num,
            num_ones,
            feature_idx,
            feature_values,
            information_gain):
        super().__init__(num, num_ones, feature_idx, information_gain)
        self.feature_values = feature_values

    def partition(self, X):
        return [np.where(X[:, self.feature_idx] == value)[0]
                for value in self.feature_values]

    def evaluate(self, x):
        if self.pruned:
            return self.pruned_majority, self.pruned_confidence

        try:
            idx = self.feature_values.index(x[self.feature_idx])
            return self.children[idx].evaluate(x)
        except ValueError:
            print("Feature is not a node")
            return None

# Continuous Node


class CNode(Node):
    def __init__(
            self,
            num,
            num_ones,
            feature_idx,
            feature_value,
            information_gain):
        super().__init__(num, num_ones, feature_idx, information_gain)
        self.feature_value = feature_value

    def partition(self, X):
        return [np.where(X[:, self.feature_idx] <= self.feature_value)[0],
                np.where(X[:, self.feature_idx] > self.feature_value)[0]]

    def evaluate(self, x):
        if self.pruned:
            return self.pruned_majority, self.pruned_confidence

        idx = 0 if x[self.feature_idx] <= self.feature_value else 1
        return self.children[idx].evaluate(x)

# Pure Node


class PNode(Node):
    def __init__(self, num, num_ones, pure_class, confidence):
        super().__init__(num, num_ones, -1, -1)
        self.pure_class = pure_class
        self.confidence = confidence

    def evaluate(self, x):
        return self.pure_class, self.confidence

# Empty Node


class ENode(Node):
    def __init__(self):
        super().__init__(0, 0, -1, 0)

# Learning Node


class LNode:
    def __init__(self, node, X, y):
        self.node = node
        self.X = X
        self.y = y

    def partition(self, ignore):
        return LSet([LNode(ENode(), self.X[idxs], self.y[idxs])
                     for idxs in self.node.partition(self.X)],
                    ignore)

    @property
    def empty(self):
        return len(self.y) == 0

    @property
    def pure(self):
        return self.empty or not np.any(self.y != self.y[0])

    @property
    def size(self):
        return len(self.y)

    @property
    def num_ones(self):
        return (self.y == 1).sum()

    @property
    def majority(self):
        return 1 if (self.y == 1).sum() >= len(self.y) / 2 else 0

    @property
    def confidence(self):
        return (self.y == self.majority).sum() / len(self.y)

# Set of Learning Nodes


class LSet:
    def __init__(self, nodes, ignore=[]):
        self.nodes = nodes
        self.idx = 0
        self.ignore = ignore

    def partition(self, schema, use_gain_ratio):
        node = self.current
        self.idx += 1

        # Compute best partition
        feature_idx, feature_pivot = determine_split_criterion(
            schema, self.ignore, node.X, node.y, use_gain_ratio=use_gain_ratio)

        # No more attributes
        if feature_idx == -1:
            node.node = PNode(
                node.size,
                node.num_ones,
                self.majority if node.empty else node.majority,
                self.confidence if node.empty else node.confidence)
            return None, None

        feature = schema[feature_idx]
        if feature.ftype == FeatureType.CONTINUOUS:
            node.node = CNode(node.size, node.num_ones,
                              feature_idx, feature_pivot, -1)
        elif feature.ftype == FeatureType.NOMINAL or feature.ftype == FeatureType.BINARY:
            node.node = DNode(
                node.size, node.num_ones, feature_idx, list(
                    schema[feature_idx].values), -1)

        # Partition the node
        return node.partition(self.ignore + [feature_idx]), feature_idx

    def set_children(self, child_set):
        node = self.nodes[self.idx - 1].node
        node.set_children([node.node for node in child_set.nodes])

    @property
    def first(self):
        return self.nodes[0]

    @property
    def current(self):
        return self.nodes[self.idx]

    @property
    def length(self):
        return len(self.nodes)

    @property
    def size(self):
        return sum([n.size for n in self.nodes])

    @property
    def num_ones(self):
        return sum([n.num_ones for n in self.nodes])

    @property
    def complete(self):
        return self.idx >= len(self.nodes)

    @property
    def majority(self):
        return 1 if sum([(node.y == 1).sum() - len(node.y) /
                        2 for node in self.nodes]) >= 0 else 0

    @property
    def confidence(self):
        return sum([(node.y == self.majority).sum()
                   for node in self.nodes]) / sum([len(node.y) for node in self.nodes])

# Container object for the dtree


class DecisionTree(Classifier):
    def __init__(self, schema: List[Feature]):
        self.tree = None
        self.size = self.max_depth = 0
        self.first_feature = None
        self._schema = schema

    def fit(
            self,
            X_train,
            y_train,
            information_gain: bool,
            tree_depth_limit: int):
        self.tree = None
        self.size = self.max_depth = 0
        self.first_feature = None
        use_gain_ratio = not information_gain

        nodes = [LSet([LNode(ENode(), X_train, y_train)])]
        while len(nodes) > 0:
            l_set = nodes[-1]

            # Done with this node
            if l_set.complete:
                del nodes[-1]
                # Build the tree
                if len(nodes) > 0:
                    nodes[-1].set_children(l_set)
                    self.size += l_set.length
                else:
                    self.tree = l_set.first.node
                    self.size += 1
                if len(nodes) > self.max_depth:
                    self.max_depth = len(nodes)
                continue

            l_node = l_set.current

            # Check for pure nodes or if maximum depth has been reached
            if l_node.pure or len(nodes) > tree_depth_limit > 0:
                l_node.node = PNode(
                    l_node.size,
                    l_node.num_ones,
                    l_set.majority if l_node.empty else l_node.majority,
                    l_set.confidence if l_node.empty else l_node.confidence)
                l_set.idx += 1
                continue

            partition, feature_idx = l_set.partition(
                self.schema, use_gain_ratio)
            if partition is not None:
                nodes.append(partition)
                if self.first_feature is None:
                    self.first_feature = self.schema[feature_idx].name

    def predict(self, X_test, y_test, silent):
        confidences = np.array([self.tree.evaluate(x) for x in X_test])
        num_correct = sum(
            [y_hat == y for y_hat, y in zip(confidences[:, 0], y_test)])
        accuracy = num_correct / len(X_test)
        if silent > 0:
            print(
                f"Accuracy: {num_correct}/{len(X_test)} ({round(accuracy * 100, 2)}%)")
            if silent > 1:
                print(f"Size: {self.size}")
                print(f"Maximum Depth: {self.max_depth}")
                print(f"First Feature: {self.first_feature}")
        return accuracy, confidences

    def prune(self, X_test, y_test):
        curr_accuracy, curr_confidence = self.predict(X_test, y_test, 0)

        order = self.top_down_order()

        max_accuracy = 0
        best_node = None
        for node in order:
            node.pruned = True
            accuracy, confidences = self.predict(X_test, y_test, 0)
            node.pruned = False
            if accuracy > max_accuracy and node.num != 0:
                max_accuracy = accuracy
                best_node = node
        if best_node is None or max_accuracy <= curr_accuracy:
            return False

        best_parent = best_node.parent
        idx = best_parent.children.index(best_node)
        best_parent.children[idx] = best_node.get_pruned()

        return True

    def prune_depth(self, X_test, y_test, depth):
        stack = [0]
        node = self.tree

        while len(stack) > 0:
            if len(stack) >= depth - 1:
                node.children = [child.get_pruned() for child in node.children]
                del stack[-1]
                node = node.parent
            elif len(node.children) == 0 or stack[-1] >= len(node.children):
                del stack[-1]
                node = node.parent
            else:
                node = node.children[stack[-1]]
                stack[-1] += 1
                stack.append(0)

    def top_down_order(self):
        idx = 0
        results = self.tree.children.copy()
        while idx < len(results):
            node = results[idx]
            results += [n for n in node.children if n.num > 0]
            idx += 1
        return results

    @property
    def schema(self):
        return self._schema


def dtree(
        data_path: str,
        tree_depth_limit: int,
        use_cross_validation: bool = True,
        information_gain: bool = True,
        post_pruning: bool = False,
        depth_pruning: bool = False,
        silent: bool = False):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook. This function has been PARTIALLY implemented for you, but not completely!
    :param data_path: The path to the data.
    :param tree_depth_limit: Depth limit of the decision tree
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
    :param post_pruning: If true, use post-pruning after creation of tree
    :param silent: removes output
    :return:
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split("/")
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = "/".join(path[:-1])

    schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    import time
    start_time = time.time()

    best_accuracy = 0
    for X_train, y_train, X_test, y_test in datasets:
        print()
        tree = DecisionTree(schema)
        tree.fit(X_train, y_train, information_gain, tree_depth_limit)
        accuracy, confidences = tree.predict(
            X_test, y_test, 0 if silent else 2)
        if post_pruning:
            if not silent:
                print("Performing Post Pruning:")
            prev_num_nodes = len(tree.top_down_order())
            prev_acc = accuracy
            while tree.prune(X_test, y_test):
                accuracy, confidences = tree.predict(X_test, y_test, 0)
            if not silent:
                num_nodes = len(tree.top_down_order())
                if num_nodes != prev_num_nodes:
                    print(
                        f"Number of nodes changed from {prev_num_nodes} -> {num_nodes}")
                    print(f"Accuracy changed from {prev_acc} -> {accuracy}")
                else:
                    print("No nodes pruned")
        elif depth_pruning:
            if not silent:
                print("Performing Depth Pruning:")
            prev_num_nodes = len(tree.top_down_order())
            prev_acc = accuracy
            prev_depth = depth = tree.max_depth
            for i in reversed(range(1, tree.max_depth)):
                tree.prune_depth(X_test, y_test, i)
                acc, confs = tree.predict(X_test, y_test, 0)
                if acc > accuracy:
                    accuracy = acc
                    confidences = confs
                    depth = i
            if not silent:
                if depth != prev_depth:
                    print(f"Depth changed from {prev_depth} -> {depth}")
                    print(
                        f"Number of nodes changed from {prev_num_nodes} -> {len(tree.top_down_order())}")
                    print(f"Accuracy changed from {prev_acc} -> {accuracy}")
                else:
                    print("No change in depth")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
    total_time = time.time() - start_time
    if not silent:
        print(
            f"Training Time: --- {total_time} seconds ({int(total_time/60)} minutes and {total_time % 60} seconds)")
    return best_accuracy


if __name__ == '__main__':
    """
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here. We have provided argparse code
    for you for this assignment, but in the future you may be responsible for doing this yourself.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(
        description='Run a decision tree algorithm.')
    parser.add_argument(
        'path',
        metavar='PATH',
        type=str,
        help='The path to the data.')
    parser.add_argument(
        'depth_limit',
        metavar='DEPTH',
        type=int,
        help='Depth limit of the tree. Must be a non-negative integer. A value of 0 sets no limit.')
    parser.add_argument(
        '--no-cv',
        dest='cv',
        action='store_false',
        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument(
        '--use-gain-ratio',
        dest='gain_ratio',
        action='store_true',
        help='Use gain ratio as tree split criterion instead of information gain.')
    parser.add_argument(
        '--post-pruning',
        dest='post_pruning',
        action='store_true',
        help='Enables post pruning')
    parser.add_argument(
        '--depth-pruning',
        dest='depth_pruning',
        action='store_true',
        help='Enables tree depth optimization')
    parser.add_argument('--silent', dest='silent', action='store_true',
                        help='Disables any output')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    # If the depth limit is negative throw an exception
    if args.depth_limit < 0:
        raise argparse.ArgumentTypeError(
            'Tree depth limit must be non-negative.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    tree_depth_limit = args.depth_limit
    use_cross_validation = args.cv
    use_information_gain = not args.gain_ratio
    use_post_pruning = args.post_pruning
    use_depth_pruning = args.depth_pruning
    silent = args.silent

    dtree(
        data_path,
        tree_depth_limit,
        use_cross_validation,
        use_information_gain,
        use_post_pruning,
        use_depth_pruning,
        silent)
