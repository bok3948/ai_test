import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_log_error
import numpy as np
from datetime import datetime

root = "/mnt/d/data/tabular/bike"
train_root = os.path.join(root, "train.csv")
test_root = os.path.join(root, "test.csv")

train = pd.read_csv(train_root)
Y = train["count"]
Y = Y.to_numpy()

train.drop(["count"], axis=1, inplace=True)
len_train = len(train)
test = pd.read_csv(test_root)
test_datetime = test["datetime"]
df = pd.concat((train, test), ignore_index=True)

df.drop(["casual", "registered", "workingday"], axis=1, inplace=True)
df.info()
print(df.isnull().sum().sort_values(ascending=False))

#category 분류 
cat_feats = []

for col in df.columns:
    if df[col].dtype == "object":
        cat_feats.append(col)
        print(col)
        print(df[col].unique())
    elif len(df[col].unique()) <= 10:
        cat_feats.append(col)
        #print(col)
        #print(df[col].unique())

#datetime processing

def datetime_processor(x):
    
    tem, time = x.split()
    date_obj = datetime.strptime(tem, "%Y-%m-%d")
    weekday = date_obj.weekday()
    hour = time.split(":")[0]
    year, month, day = tem.split("-")
    
    return weekday, year, month, day, hour

df[["weekday", "year", "month", "day", "hour"]] = df["datetime"].apply(datetime_processor).apply(pd.Series)
df[["year", "month", "day", "hour"]] = df[["year", "month", "day", "hour"]].astype(float)

print(set(df["weekday"]))

cat_feats = []
for col in df.columns:
    if df[col].dtype == "object":
        cat_feats.append(col)

df.drop(columns=cat_feats, axis=1, inplace=True)
df.info()

print(df.isnull().sum().sort_values(ascending=False))

x_train = df.iloc[:len_train]
print(len_train)
x_test = df.iloc[len_train:].copy()

x_train, x_test = x_train.to_numpy(), x_test.to_numpy()
print(f"train: {x_train.shape, Y.shape}, X_test: {x_test.shape}")


def rmsle(y, a):
    d = mean_squared_log_error(y, a)
    e = np.sqrt(d)
    return e

rfc = RandomForestRegressor(random_state=1)
#params = {"n_estimators": range(50, 200, 50)}
params = {
    "n_estimators": [150], # range(50, 200, 50),
    "max_depth": [None], #[None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
rfc = GridSearchCV(rfc, param_grid=params, scoring=make_scorer(rmsle, greater_is_better=False), cv=5)
rfc.fit(x_train, Y)
best_params = rfc.best_params_
best_scores = -1 * rfc.best_score_
print(best_params)
results = rfc.cv_results_

#scores = cross_val_score(rfc, x_train, Y, scoring=make_scorer(rmsle))
print(f"RMSLE: {best_scores:.4f}")


def inferance(model, x_train, Y, x_test):
    model.fit(x_train, Y)
    A =  model.predict(x_test)
    A[A < 0] = 0
    bike_sub = pd.DataFrame()
    bike_sub["datetime"] = test_datetime
    bike_sub["count"] = A
    bike_sub.to_csv(os.path.join(root, "submission.csv"), index=False)
    return None

rfc = RandomForestRegressor(**best_params, random_state=1)
inferance(rfc, x_train, Y, x_test)