import numpy as np
from scipy import stats


class RTLearner(object):

    def __init__(self, leaf_size=5):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.decision_tree = None


    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        def build_tree(data_x, data_y):
            # Base Cases
            # if number of rows left is less than or equal to leaf size, then make a leaf
            if data_x.shape[0] <= self.leaf_size:
                mode = int(stats.mode(data_y)[0][0])
                return np.array([[-1, mode, -1, -1]])
            # if all the rest of the y values are the same
            if np.all(data_y == data_y[0]):
                return np.array([[-1, data_y[0], -1, -1]])
            # -------------
            featureIndex = np.random.randint(data_x.shape[1], size=1)[0]
            median = np.median(data_x[:, featureIndex])
            # Handle case where median is also the maximum value but not all y values are the same
            if median == np.max(data_x[:, featureIndex]):
                splitVal = np.mean(data_x[:, featureIndex])
            else:
                splitVal = median
            # -------------
            leftTree = build_tree(data_x[data_x[:,featureIndex]<=splitVal], data_y[data_x[:,featureIndex]<=splitVal])
            rightTree = build_tree(data_x[data_x[:,featureIndex]>splitVal], data_y[data_x[:,featureIndex]>splitVal])
            root = np.array([featureIndex, splitVal, 1, leftTree.shape[0]+1])
            return np.vstack((root, leftTree, rightTree))

        self.decision_tree = build_tree(data_x, data_y)


    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        def search_tree(index, arr):
            feature = int(self.decision_tree[index][0])
            splitVal = self.decision_tree[index][1]
            left = int(self.decision_tree[index][2])
            right = int(self.decision_tree[index][3])
            if feature == -1:
                return splitVal
            if arr[feature] <= splitVal:
                leaf = search_tree(left+index, arr)
            else:
                leaf = search_tree(right+index, arr)

            return leaf


        if points.ndim == 1:
            return search_tree(0, points)
        predictions = np.array([])
        for point in points:
            predictions = np.append(predictions, search_tree(0, point))
        return predictions
