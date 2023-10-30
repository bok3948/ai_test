import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor

root = "/mnt/d/data/tabular/bike"
train_root = os.path.join(root, "train.csv")
test_root = os.path.join(root, "test.csv")
sample_root = os.path.join(root, "sampleSubmission.csv")

train = pd.read_csv(train_root)
test = pd.read_csv(test_root)
sample = pd.read_csv(sample_root)

def check(df):
    print("-"*50)
    print()
    print(df.info())

Y = train["count"]
train.drop(columns=["count"], axis=1, inplace=True)
test_date = test["datetime"]

len_train = len(train)

df = pd.concat((train, test), ignore_index=True)

#too many str 처리
def date_split(x):
    a, b= x.split()
    c = b.split(":")
    return a, int(c[0])

df[["tem", "hour"]] = df["datetime"].apply(date_split).apply(pd.Series)
df["tem"] = pd.to_datetime(df["tem"])
df["year"] = df["tem"].dt.year
df["month"] = df["tem"].dt.month
df["day"] = df["tem"].dt.day
df["weekday"] = df["tem"].dt.weekday

df.drop(columns=["datetime", "tem"], axis=1, inplace=True)

#find to drop
nulls = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
print(nulls)

df.drop(columns=["casual", "registered", "workingday"], axis=1, inplace=True)

x_train = df[:len_train]
x_test = df[len_train:].copy()

check(x_train)

print(f"train: {x_train.shape, Y.shape} test: {x_test.shape}")

rfc = RandomForestRegressor(random_state=0)
rfc.fit(x_train, Y)
A = rfc.predict(x_train)
from sklearn.metrics import accuracy_score
#import sklearn.metrics
#print(dir(sklearn.metrics))
#print(rfc.oob_score_)
score = accuracy_score(Y, A)
print(score)

def inference(model, x_test):
    A = model.predict(x_test)
    submission = pd.DataFrame()
    submission["datetime"] = test_date
    A[A < 0] = 0
    submission["count"] = A
    submission.to_csv(os.path.join(root, "submission.csv"), index=False)
inference(rfc, x_test)


