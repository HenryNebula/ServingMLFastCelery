from pathlib import Path
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=32, verbose=False):
        cache_folder=Path(__file__).parent / ".cache/"
        if not cache_folder.exists():
            cache_folder.mkdir()
        self.transformer = SentenceTransformer(model_name, device="cpu", cache_folder=cache_folder)

        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t_0 = time()
        embedding_size = self.transformer.get_sentence_embedding_dimension()
        X_ = self.transformer.encode(X, batch_size=self.batch_size, show_progress_bar=self.verbose).reshape(-1, embedding_size)
        if self.verbose:
            print(f"Encoding finishes in {time() - t_0: .1f} seconds")
        return X_