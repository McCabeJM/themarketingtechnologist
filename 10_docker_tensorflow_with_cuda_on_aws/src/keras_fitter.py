from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from base_fitter import BaseEstimatorFitter
from constants import NUM_CLASSES, LOSS_FUNCTION, OPTIMIZER, NUM_FEATURES


class KerasModelFitter(BaseEstimatorFitter):
    def __init__(self, x_train, y_train, x_test, y_test):
        super(KerasModelFitter, self).__init__(x_train, y_train, x_test, y_test)
        self.batch_size = None
        self.epochs = None

    def _set_estimator(self):
        estimators = list()
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(build_fn=self._get_baseline_model,
                                                 epochs=self.epochs,
                                                 batch_size=self.batch_size,
                                                 verbose=2)))
        self.estimator = Pipeline(estimators)

    def _get_baseline_model(self):
        model = Sequential()

        model.add(Dense(512, activation='relu', input_shape=(NUM_FEATURES,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        model.compile(loss=LOSS_FUNCTION,
                      optimizer=OPTIMIZER,
                      metrics=['accuracy'])

        return model
