from pathlib import Path
from keras.optimizers import RMSprop


BASE_FOLDER = Path(__file__).parents[1]
OUTPUT_FOLDER = Path(BASE_FOLDER, 'output')
NUM_CLASSES = 10
NUM_FEATURES = 784
MLP_HYPER_PARAMETER_BATCH_SIZE = 128
MLP_HYPER_PARAMETER_EPOCHS = 20
LOSS_FUNCTION = 'categorical_crossentropy'
OPTIMIZER = RMSprop()
SEED = 7
