import json
from pathlib import Path

from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import joblib
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as pipe_imb

df = pd.read_csv(Path(__file__).parent / "data/BankChurners.csv")
unused_nb_cols = df.filter(like='Naive_Bayes_Classifier').columns
df = df.drop(columns=unused_nb_cols)
sample = pd.read_json(Path(__file__).parent / "data/sample.json", orient='records')
print(sample)

le = LabelEncoder()
le.classes_ = np.array(['Existing Customer', 'Attrited Customer']) # Override label order so Churn = 1
y = le.transform(df['Attrition_Flag'])
df = df.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1)

CATEGORICAL_FEATURES = ['Gender','Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
NUMERICAL_FEATURES = [col for col in df.columns if col not in CATEGORICAL_FEATURES]

preprocessing_pipeline = ColumnTransformer(transformers=[
    ('num', StandardScaler(), NUMERICAL_FEATURES),
    ('cat', OneHotEncoder(sparse=False), CATEGORICAL_FEATURES)
])

df_new = pd.DataFrame(preprocessing_pipeline.fit_transform(df))

l_transformers = list(preprocessing_pipeline._iter(fitted=True))
column_names = []
for name, trans, column, _ in l_transformers:
  if hasattr(trans, 'get_feature_names'):
    column_names.extend(trans.get_feature_names(column))
  else:
    column_names.extend(column)

X_train, X_test, y_train, y_test = train_test_split(df_new, y, test_size=0.2, random_state=11)

sm = BorderlineSMOTE(sampling_strategy=0.3)
rus = RandomUnderSampler(sampling_strategy=0.6)

lgb = LGBMClassifier(
  objective = 'binary',
  learning_rate=0.03,
  n_estimators=5000,
  max_depth=3,
  subsample=0.9,
  colsample_bytree=0.9,
  min_child_weight=3
)
train_pipeline = pipe_imb([('sm', sm), ('rus', rus), ('clf', lgb)])
train_pipeline.fit(X_train.values, y_train)

final_model = train_pipeline.steps[-1][-1]
final_pipeline = Pipeline([('preprocess', preprocessing_pipeline), ('model', final_model)])
print(final_pipeline.predict(sample))

joblib.dump(final_pipeline, Path(__file__).parent / 'data/model.pkl')