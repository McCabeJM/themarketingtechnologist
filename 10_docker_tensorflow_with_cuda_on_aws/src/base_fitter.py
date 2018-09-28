import logging
import numpy as np
from constants import SEED

np.random.seed(SEED)


class BaseEstimatorFitter(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.estimator = None

    def fit_estimator(self):
        self._set_estimator()
        self._fit_estimator_on_full_training_set()
        self._calculate_train_and_test_score()
        return self.estimator

    def _set_estimator(self):
        raise NotImplementedError

    def _fit_estimator_on_full_training_set(self):
        self.estimator.fit(self.x_train, self.y_train)

    def _calculate_train_and_test_score(self):
        train_score = -1 * self.estimator.score(self.x_train, self.y_train)
        test_score = -1 * self.estimator.score(self.x_test, self.y_test)
        logging.info("Estimator train score: %.8f, test score: %.8f" % (train_score, test_score))
