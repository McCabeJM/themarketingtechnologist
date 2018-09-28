import logging
from keras.datasets import mnist
from keras.utils import to_categorical
from constants import OUTPUT_FOLDER, NUM_CLASSES, NUM_FEATURES
from data_model import KerasEstimatorMlp, KerasEstimatorPipeline
from mlp_fitter import MultiLayerPerceptronFitter


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class ModelFitterJob:
    def __init__(self):
        self.output_folder = OUTPUT_FOLDER
        self.estimator = None

    def run(self):
        self._load_data()
        self._preprocess_data()
        self._fit_model()
        self._save_estimator()

    def _load_data(self):
        logging.info("Load data...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def _preprocess_data(self):
        self._reshape_x_matrices_to_vectors()
        self._one_hot_encode_y_vectors()

    def _reshape_x_matrices_to_vectors(self):
        self.x_train = self.x_train.reshape(60000, NUM_FEATURES).astype('float32')
        self.x_test = self.x_test.reshape(10000, NUM_FEATURES).astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        logging.info('{} train samples'.format(self.x_train.shape[0]))
        logging.info('{} test samples'.format(self.x_test.shape[0]))

    def _one_hot_encode_y_vectors(self):
        self.y_train = to_categorical(self.y_train, NUM_CLASSES)
        self.y_test = to_categorical(self.y_test, NUM_CLASSES)

    def _fit_model(self):
        logging.info("Fit model...")
        fitter = MultiLayerPerceptronFitter(self.x_train, self.y_train, self.x_test, self.y_test)
        self.estimator = fitter.fit_estimator()

    def _save_estimator(self):
        logging.info("Save estimator...")
        self._save_keras_estimator()

    def _save_keras_estimator(self):
        keras_mlp_model = self.estimator.named_steps['mlp'].model
        KerasEstimatorMlp(keras_mlp_model).save(self.output_folder)

        self.estimator.named_steps['mlp'].model = None
        KerasEstimatorPipeline(self.estimator).save(self.output_folder)


if __name__ == "__main__":
    ModelFitterJob().run()
