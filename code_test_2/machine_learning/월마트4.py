import os 
import pandas as  pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

root = "/mnt/d/data/tabular/walmart-recruiting-store-sales-forecasting"
train_root = os.path.join(root, "train.csv", "train.csv")
test_root = os.path.join(root, "test.csv", "test.csv")
feature_root = os.path.join(root, "features.csv", "features.csv")
stores_root = os.path.join(root, "stores.csv")
sample_root = os.path.join(root, "sampleSubmission.csv", "sampleSubmission.csv")

train = pd.read_csv(train_root)
test = pd.read_csv(test_root)
feature = pd.read_csv(feature_root)
stores = pd.read_csv(stores_root)
sample = pd.read_csv(sample_root)

def check(df):
  
    print("-"*50)
    print()
    df.info()
#for sas in [train, test, feature, stores, sample]:
#    check(sas)
#Y
Y = train["Weekly_Sales"]
train.drop(columns=["Weekly_Sales"], axis=1, inplace=True)

#merge
tem = pd.merge(stores, feature, how='inner', on="Store")
train = pd.merge(train, tem, how='inner', on=['Store', 'Date', 'IsHoliday']).sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)
test = pd.merge(test, tem, how='inner', on=['Store', 'Date', 'IsHoliday']).sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

#ids
ids = test["Store"].astype(str) + "_" + test["Dept"].astype(str) + "_" + test["Date"].astype(str)

len_train = len(train)
df = pd.concat((train, test), ignore_index=True)

#check(df)

#feature enginnerring
def feat_cls(df):
    a, b, c, d = [], [], [], []
    for col in df.columns:
        if (df[col].dtype == 'object') and (len(df[col].unique()) > 10):
            a.append(col)
        elif (df[col].dtype == 'object') and (len(df[col].unique()) <= 10):
            b.append(col)
        elif not(df[col].dtype == 'object') and (len(df[col].unique()) > 10):
            c.append(col)
        elif not(df[col].dtype == 'object') and (len(df[col].unique()) <= 10):
            d.append(col)

    print(f"too_many_cat: {a}\n cat: {b}\n numer: {c}\n int_cat: {d}")
    return a,b,c,d
#a,b,c,d = feat_cls(df)

df["Date"] = pd.to_datetime(df["Date"])
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["week"] = df["Date"].dt.isocalendar().week
df["day"] = df["Date"].dt.day

#df["days_to_chrismas"] = (pd.to_datetime(df["year"].astype(str) + "-12-24") - df["Date"]).dt.days.astype(int)
#df["days_to_thanksgiving"] = (pd.to_datetime(df["year"].astype(str) + "-11-24") - df["Date"]).dt.days.astype(int)

#df["superball"] = df["week"].apply(lambda x: 1 if x==6 else 0)
#df["laborday"] = df["week"].apply(lambda x: 1 if x==36  else 0)
#df["thanksgiving"] = df["week"].apply(lambda x: 1 if x==47  else 0)
#df["christmas"] = df["week"].apply(lambda x: 1 if x==52 else 0)

df.drop(columns=["Date"], axis=1, inplace=True)
#a,b,c,d = feat_cls(df)
df["MarkDownSum"] = df["MarkDown1"] + df["MarkDown2"] + df["MarkDown3"] + df["MarkDown4"] + df["MarkDown5"] 

#fill na

df["CPI"].fillna(df["CPI"].mean(), inplace=True)
df["Unemployment"].fillna(df["Unemployment"].mean(), inplace=True)
df.fillna(0, inplace=True)

#nulls = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
#print(nulls)

#cat 
LE = LabelEncoder()
df["Type"] = LE.fit_transform(df["Type"])
df["IsHoliday"] = df["IsHoliday"].astype(int)

a,b,c,d = feat_cls(df)

x_train = df[:len_train]
x_test = df[len_train:].copy()

print(f" train: {x_train.shape, Y.shape} test: {x_test.shape}")

rfc = RandomForestRegressor(random_state=0)
rfc.fit(x_train, Y)

def inference(model, x_test):
    A = model.predict(x_test)
    sub = pd.DataFrame()
    sub["Id"] = ids
    sub["Weekly_Sales"] = A
    sub.to_csv(os.path.join(root, "submission.csv"), index=False)
inference(rfc, x_test)


