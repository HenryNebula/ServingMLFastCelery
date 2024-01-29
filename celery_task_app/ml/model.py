import joblib
import os
import pandas as pd

from typing import List, Union

MODEL_PATH = os.environ['MODEL_PATH']
CLASSIFIER_PATH = os.environ['CLASSIFIER_PATH']


class BaseModel:

    """ Wrapper for loading and serving pre-trained model"""

    def __init__(self, model_path):
        self.model = self._load_model_from_path(model_path)

    @staticmethod
    def _load_model_from_path(path):
        model = joblib.load(path)
        return model

    def predict(self, data, return_option='Prob'):
        """
        Make batch prediction on list of preprocessed feature dicts.
        Returns class probabilities if 'return_options' is 'Prob', otherwise returns class membership predictions
        """
        df = pd.DataFrame(data)
        if return_option == 'Prob':
            predictions = self.model.predict_proba(df)
        else:
            predictions = self.model.predict(df)
        return predictions


class ChurnModel(BaseModel):
    def __init__(self):
        super().__init__(MODEL_PATH)


class Classifier(BaseModel):
    def __init__(self):
        super().__init__(CLASSIFIER_PATH)

    def predict(self, data: Union[List[str], str], return_option='Prob'):
        
        if isinstance(data, str):
            data = [data]
        
        if return_option == 'Prob':
            predictions = self.model.predict_proba(data)
        else:
            predictions = self.model.predict(data)
        return predictions
