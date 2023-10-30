import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

root = "/mnt/d/data/tabular/walmart-recruiting-store-sales-forecasting"
train_root = os.path.join(root, "train.csv", "train.csv")
test_root = os.path.join(root, "test.csv", "test.csv")
feature_root = os.path.join(root, "features.csv", "features.csv")
stores_root = os.path.join(root, "stores.csv")
sample_root = os.path.join(root, "sampleSubmission.csv", "sampleSubmission.csv")

train = pd.read_csv(train_root)
test = pd.read_csv(test_root)
feats = pd.read_csv(feature_root)
stores = pd.read_csv(stores_root)
sample = pd.read_csv(sample_root)

#check
def check(df):
    print("-"*50)
    print()
    df.info()
    print(df.head())
    

for df in [train, test, feats, stores, sample]:
    check(df)
Y = train["Weekly_Sales"]
train.drop(columns=["Weekly_Sales"], axis=1, inplace=True)

tem = pd.merge(feats, stores, how='inner', on=["Store"])
train = pd.merge(train, tem, how='inner', on=["Store", "Date", "IsHoliday"]).sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)
test = pd.merge(test, tem, how='inner', on=["Store", "Date", "IsHoliday"]).sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

Ids = []
for i in range(len(test)):
    a = test["Store"][i]
    b = test["Dept"][i]
    c = test["Date"][i]
    id = str(a)+"-"+str(b)+"-"+str(c)
    Ids.append(id)

len_train = len(train)
df = pd.concat((train, test), ignore_index=True)



