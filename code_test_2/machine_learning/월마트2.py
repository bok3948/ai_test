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
    return None


#for df in [train, test, feats, stores, sample]:
#    check(df)

Y = train["Weekly_Sales"]
train.drop(columns=["Weekly_Sales"], axis=1, inplace=True)

#check sample 
#for col in sample.columns:
#    print(sample[col][:5])


#merge 

tem = pd.merge(feats, stores, how="inner", on=["Store"])
train = pd.merge(train, tem, how='inner', on=["Store", "Date", "IsHoliday"]).sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)
test = pd.merge(test, tem, how='inner', on=["Store", "Date", "IsHoliday"]).sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

Ids = []
for i in range(len(test)):
    a = test["Store"][i]
    b = test["Dept"][i]
    c = test["Date"][i]
    id = str(a) + "_" + str(b) + "_" + str(c)
    Ids.append(id)


len_train = len(train)
df = pd.concat((train, test), ignore_index=True)

def feat_cls(df):
    a, b, c, d = [], [], [], []
    for col in df.columns:
        if (df[col].dtype == "object") and (len(df[col].unique()) < 10):
            a.append(col)
        elif (df[col].dtype == "object") and (len(df[col].unique()) >= 10):
            b.append(col)

        elif not (df[col].dtype == "object") and (len(df[col].unique()) < 10):
            c.append(col)
        elif not (df[col].dtype == "object") and (len(df[col].unique()) >= 10):
            d.append(col)
    
    print(f"category: {a} \n too_many_category: {b} \n int_category: {c} \n numer: {d}")
    return a, b, c, d
a, b, c, d = feat_cls(df)


#features engineering
df["MarkDownSum"] = df["MarkDown1"] + df["MarkDown2"] + df["MarkDown3"] + df["MarkDown4"] + df["MarkDown5"]

print(df["Date"][:3])
df["Date"] = pd.to_datetime(df["Date"])
df["year"] = df["Date"].dt.year 
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["weekday"] = df["Date"].dt.isocalendar().week

tem = df.copy()
df["Days_to_Thaksgiving"] = (pd.to_datetime(tem["year"].astype(str) +"-11-24") - tem["Date"]).dt.days.astype(int)
df["Days_to_Christmas"] = (pd.to_datetime(tem["year"].astype(str) +"-12-24") - tem["Date"]).dt.days.astype(int)

for col in ["Date", "Days_to_Thaksgiving", "Days_to_Christmas"]:
    print(df[col][:3])

df["SuperBowlWeek"] = df["weekday"].apply(lambda x: 1 if x == 6 else 0)
df["Laborday"] = df["weekday"].apply(lambda x: 1 if x == 36 else 0)
df["Thanksgiving"] = df["weekday"].apply(lambda x: 1 if x == 47 else 0)
df["Cristmas"] = df["weekday"].apply(lambda x: 1 if x == 52 else 0)


df.drop(["Date"], axis=1, inplace=True)


na = df.isnull().sum().sort_values(ascending=False)
print(na)
df["CPI"].fillna(df["CPI"].mean(), inplace=True)
df["Unemployment"].fillna(df["Unemployment"].mean(), inplace=True)
df.fillna(0, inplace=True)

LE = LabelEncoder()
df["Type"] = LE.fit_transform(df["Type"])
#df = pd.get_dummies(df, columns=["Type"])
df["IsHoliday"] =df["IsHoliday"].astype(int)
a, b, c, d = feat_cls(df)

x_train = df[:len_train]
x_test = df[len_train:].copy()

#scaler = StandardScaler()
#x_train[d] = scaler.fit_transform(x_train[d])
#x_test[d] = scaler.transform(x_test[d])

print(f"train: {x_train.shape, Y.shape} test: {x_test.shape}")

#rfr = RandomForestRegressor(random_state=0)

param_grid = {"n_estimators": 60, #list(range(50, 200, 50)),
          "max_depth": 25, #[None, 10, 20],
          "min_samples_split": 3, #[2, 4],
          "min_samples_leaf": 1, #[1, 2]
          } finally


def weight_mae(y, a, weights):
    we = weights.apply(lambda x: 5 if x ==1 else 1)
    ans = np.sum(we*abs(y-a)) / np.sum(we)
    return ans 

#rfr = GridSearchCV(rfr, param_grid=param_grid, cv=5, scoring="neg_mean_absolute_error")
rfr = RandomForestRegressor(**param_grid, random_state=0)

#rfr.fit(x_train, Y)

#score = rfr.best_score_
#best_param = rfr.best_params_
#print(f"5 fold best score: {round(score, 2)}")
#print(f"best param :{best_param}")


#rfr = RandomForestRegressor(**best_param, random_state=0)

def inferance(model, x_train, x_test, Y):
    model.fit(x_train, Y)
    A = model.predict(x_test)

    submission = pd.DataFrame()
    submission["Id"] = Ids
    submission["Weekly_Sales"] = A
    submission.to_csv(os.path.join(root,"submission.csv"), index=False)

inferance(rfr, x_train, x_test, Y)