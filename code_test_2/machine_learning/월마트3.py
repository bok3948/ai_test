import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestRegreesor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer

#print(dir(sklearn.model_selection))

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

def check(df):
    print("-"* 50)
    print()
    df.info()

for d in [train, test, feats, stores, sample]:
    check(d)

Y = train["Weekly_Sales"]
train.drop(columns=["Weekly_Sales"], axis=1, inplace=True)

#ids 는 병합있는는 나중에

#merge
tem = pd.merge(stores, feats, how='inner', on=["Store"])
train = pd.merge(train, tem, how='inner', on=["Store", "Date", "IsHoliday"]).sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)
test = pd.merge(test, tem, how='inner', on=["Store", "Date", "IsHoliday"]).sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)


ids = test["Store"].astype(str) + "_" + test["Dept"].astype(str) + "_" + test["Date"].astype(str)



len_train = len(train)
df = pd.concat((train, test), ignore_index=True)

def feat_cls(df):
    a, b, c, d = [], [], [], []
    for col in df.columns:
        if (df[col].dtype == "object") and (len(df[col].unique()) > 10):
            a.append(col)
        elif (df[col].dtype == "object") and (len(df[col].unique()) <= 10):
            b.append(col)
        elif not(df[col].dtype == "object") and (len(df[col].unique()) < 10):
            c.append(col)
        elif not(df[col].dtype == "object") and (len(df[col].unique()) >= 10):
            d.append(col)

    print(f"too_many_str: {a} \n category: {b} \n int_cat: {c} \n numers: {d}")

    return a, b, c, d
a, b, c, d = feat_cls(df)

#feature engineering
df["Date"] = pd.to_datetime(df["Date"])
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["week"] = df["Date"].dt.isocalendar().week
df["weekday"] = df["Date"].dt.weekday

df["days_to_christmas"] = (pd.to_datetime(df["year"].astype(str) + "-12-24") - df["Date"]).dt.days.astype(int)
df["days_to_thanksgiving"]=(pd.to_datetime(df["year"].astype(str) + "-11-24") - df["Date"]).dt.days.astype(int)

df["superball"] = df["week"].apply(lambda x: 1 if x==6 else 0)
df["laborday"] = df["week"].apply(lambda x: 1 if x==36 else 0)
df["thanksgiving"] = df["week"].apply(lambda x: 1 if x==47 else 0)
df["christmas"] = df["week"].apply(lambda x: 1 if x==52 else 0)

df["MarkDownSum"] = df["MarkDown1"]+df["MarkDown2"]+df["MarkDown3"]+df["MarkDown4"]+df["MarkDown5"]

df.drop(columns=["Date"], axis=1, inplace=True)

#find to drop
nulls = df.isnull().sum().sort_values(ascending=False)
#print(nulls)

#cat, int_cat
LE = LabelEncoder()
df["Type"] = LE.fit_transform(df["Type"])
df["IsHoliday"] = df["IsHoliday"].astype(int)

a, b, c, d = feat_cls(df)

df["CPI"].fillna(df["CPI"].mean(), inplace=True)
df["Unemployment"].fillna(df["Unemployment"].mean(), inplace=True)
df.fillna(0, inplace=True)

x_train = df[:len_train]
x_test = df[len_train:].copy()

print(f"train: {x_train.shape, Y.shape} test: {x_test.shape}")


rfc = RandomForestRegressor(random_state=0)
#param_grid = {
  #  "n_estimators": [50],#list(range(50, 100, 50))
    #"max_depth": [None, 20, 40]
  #            }
#grid_sea = GridSearchCV(rfc, param_grid, cv=5, scoring="neg_mean_absolute_error")
#grid_sea.fit(x_train, Y)
#scores = cross_val_score(rfc, x_train, Y, cv=5, scoring=make_scorer(mean_absolute_error))
#best_rfc = grid_sea.best_estimator_
#best_param = grid_sea.best_params_
#best_scores = grid_sea.best_score_
rfc.fit(x_train, Y)

def inference(model, x_test):
    #model.fit(x_train, Y)
    A = model.predict(x_test)

    submission = pd.DataFrame()
    submission["Id"] = ids
    submission["Weekly_Sales"] = A
    #print(submission)
    submission.to_csv(os.path.join(root, "submission.csv"), index=False)

inference(rfc, x_test)












