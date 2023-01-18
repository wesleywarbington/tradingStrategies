import numpy as np
from scipy import stats


class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost=False):
        """
        Constructor method
        """
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.learners = np.array([])


    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        for _ in range(self.bags):
            temp_learner = self.learner(**self.kwargs)
            indexes = np.random.randint(data_x.shape[0], size=data_x.shape[0])
            temp_x_data = np.array(data_x[indexes[0]])
            temp_y_data = np.array(data_y[indexes[0]])
            for i in indexes[1:]:
                temp_x_data = np.vstack((temp_x_data, data_x[i]))
                temp_y_data = np.append(temp_y_data, data_y[i])
            temp_learner.add_evidence(temp_x_data, temp_y_data)
            self.learners = np.append(self.learners, temp_learner)


    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        predictions = np.array([self.learners[0].query(points)])
        for learner in self.learners[1:]:
            predictions = np.vstack((predictions, learner.query(points)))
        mode = stats.mode(predictions, axis=0)[0][0]
        return mode
