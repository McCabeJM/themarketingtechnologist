from pathlib import Path
from sklearn.externals import joblib


class BaseDataModel(object):
    def __init__(self, data=None, **kwargs):
        self.data = data
        self.file_name = None
        self.save_function = None

    def save(self, target_folder):
        self.save_function(target_folder)

    def _save_object_to_pickle(self, target_folder):
        target_file_path = Path(target_folder, '{}.p'.format(self.file_name))
        joblib.dump(self.data, target_file_path)
        return target_file_path

    def _save_keras_model_to_h5(self, target_folder):
        target_file_path = Path(target_folder, '{}.h5'.format(self.file_name))
        self.data.save(target_file_path)
        return target_file_path


class KerasEstimatorMlp(BaseDataModel):
    def __init__(self, data=None):
        super(BaseDataModel, self).__init__()
        self.data = data
        self.file_name = 'keras_pipeline_mlp'
        self.save_function = self._save_keras_model_to_h5


class KerasEstimatorPipeline(BaseDataModel):
    def __init__(self, data=None):
        super(BaseDataModel, self).__init__()
        self.data = data
        self.file_name = 'keras_pipeline'
        self.save_function = self._save_object_to_pickle
