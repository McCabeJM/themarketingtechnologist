import logging
from keras_fitter import KerasModelFitter
from constants import MLP_HYPER_PARAMETER_EPOCHS, MLP_HYPER_PARAMETER_BATCH_SIZE


class MultiLayerPerceptronFitter(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.estimator = None

    def fit_estimator(self):
        self.estimator = KerasModelFitter(self.x_train, self.y_train, self.x_test, self.y_test)
        self._set_hyper_parameters()
        self._fit_estimator_for_set_of_hyper_parameters()
        return self.fitted_estimator

    def _set_hyper_parameters(self):
        self.mlp_hyper_parameters_range = {
            'epochs': MLP_HYPER_PARAMETER_EPOCHS,
            'batch_size': MLP_HYPER_PARAMETER_BATCH_SIZE
        }

    def _fit_estimator_for_set_of_hyper_parameters(self):
        logging.info("Fit estimator...")
        best_params = {}
        for hyper_parameter, value in self.mlp_hyper_parameters_range.items():
            best_params[hyper_parameter] = value
            setattr(self.estimator, hyper_parameter, value)
        logging.info("Best parameters: {}".format(best_params))
        self.fitted_estimator = self.estimator.fit_estimator()
