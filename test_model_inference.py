# %%
import joblib
# %%
model = joblib.load('./data/news_classifer.joblib')
# %%
model.predict_proba(["This is a sentence", "This is another sentence"])
# %
# %%
