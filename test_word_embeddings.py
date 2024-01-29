# %%
%load_ext autoreload
%autoreload 2
# %%
from celery_task_app.model_utils.news_group.preprocessing import load_dataset
from celery_task_app.model_utils.news_group.vectorizer import SentenceTransVectorizer

import numpy as np
# %%
features = np.array(["This is a sentence", "This is another sentence"])
features.shape
# %%
sentence_vectorize = SentenceTransVectorizer(verbose=True,)
sentence_vectorize.fit_transform(features).shape
# %%
X_train, X_test, y_train, y_test, target_names = load_dataset(remove=("headers", "footers", "quotes"))
# %%
X_train[0]
# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
# %%
def benchmark_pipeline(vectorizer, classifier, return_pipeline=False):
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    if return_pipeline:
        return pipeline
    else:
        return accuracy_score(y_test, y_pred)

# %%
benchmark_pipeline(
    TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"), 
    ComplementNB(alpha=0.1)
)
# %%
benchmark_pipeline(
    SentenceTransVectorizer(verbose=True), 
    KNeighborsClassifier(n_neighbors=10, weights='distance', metric='cosine')
)
# %%
production_pipeline = benchmark_pipeline(
    SentenceTransVectorizer(verbose=False), 
    KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine'),
    return_pipeline=True
)
# %%
import joblib
# %%
joblib.dump(production_pipeline, 'news_classifer.joblib')
# %%
