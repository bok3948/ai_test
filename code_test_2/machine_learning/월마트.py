import os
import pandas as pd
from sklearn.model_selection import cross_val_score
root = "/mnt/d/data/tabular/walmart-recruiting-store-sales-forecasting"
feature_root = os.path.join(root, "features.csv", "features.csv")
stores_root = os.path.join(root, "stores.csv")

train_root = os.path.join(root, "train.csv", "train.csv")
test_root = os.path.join(root, "test.csv", "test.csv")

train = pd.read_csv(train_root)
test = pd.read_csv(test_root)
feats = pd.read_csv(feature_root)
stores = pd.read_csv(stores_root)

#train.info()
#test.info()

#feats.info()
#stores.info()


ids = []
for i in range(len(test["Store"])):
    store = test["Store"][i]
    dept = test["Dept"][i]
    date = test["Date"][i]
    id = str(store) + "_" + str(dept) + "_" + str(date)
    ids.append(id)

Y = train["Weekly_Sales"]
train.drop(["Weekly_Sales"], axis=1, inplace=True)

len_train = len(train)
#train = pd.concat((train,test), ignore_index=True)

tem = pd.merge(stores, feats, on = ['Store'],how = 'inner')
df_train = pd.merge(train, tem, on=["Store", "Date", "IsHoliday"], how='inner').sort_values(by=["Store","Dept", "Date"]).reset_index(drop=True)

df_test = pd.merge(test, tem, on=["Store", "Date", "IsHoliday"], how='inner').sort_values(by=["Store","Dept", "Date"]).reset_index(drop=True)




# Converting date column to datetime 
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])

# Adding some basic datetime features
df_train['Day'] = df_train['Date'].dt.day
df_train['Week'] = df_train['Date'].dt.isocalendar().week
df_train['Month'] = df_train['Date'].dt.month
df_train['Year'] = df_train['Date'].dt.year

df_test['Day'] = df_test['Date'].dt.day
df_test['Week'] = df_test['Date'].dt.isocalendar().week
df_test['Month'] = df_test['Date'].dt.month
df_test['Year'] = df_test['Date'].dt.year

df_test.info()

null_nums_train = df_train.isnull().sum().sort_values(ascending=False)
null_nums_test = df_test.isnull().sum().sort_values(ascending=False)

#print(null_nums_train)
#print(null_nums_test)
df_train["Marksum"] = df_train["MarkDown1"] + df_train["MarkDown2"] + df_train["MarkDown3"] + df_train["MarkDown4"] +df_train["MarkDown5"]

df_test["Marksum"] = df_test["MarkDown1"] + df_test["MarkDown2"] + df_test["MarkDown3"] + df_test["MarkDown4"] +df_test["MarkDown5"]

data_test = df_test
data_train = df_train

test_df = df_test.copy()
train_df = df_train.copy()

data_train['Days_to_Thansksgiving'] = (pd.to_datetime(train_df["Year"].astype(str)+"-11-24", format="%Y-%m-%d") - pd.to_datetime(train_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)
data_train['Days_to_Christmas'] = (pd.to_datetime(train_df["Year"].astype(str)+"-12-24", format="%Y-%m-%d") - pd.to_datetime(train_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)

data_test['Days_to_Thansksgiving'] = (pd.to_datetime(test_df["Year"].astype(str)+"-11-24", format="%Y-%m-%d") - pd.to_datetime(test_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)
data_test['Days_to_Christmas'] = (pd.to_datetime(test_df["Year"].astype(str)+"-12-24", format="%Y-%m-%d") - pd.to_datetime(test_df["Date"], format="%Y-%m-%d")).dt.days.astype(int)

data_train['SuperBowlWeek'] = train_df['Week'].apply(lambda x: 1 if x == 6 else 0)
data_train['LaborDay'] = train_df['Week'].apply(lambda x: 1 if x == 36 else 0)
data_train['Tranksgiving'] = train_df['Week'].apply(lambda x: 1 if x == 47 else 0)
data_train['Christmas'] = train_df['Week'].apply(lambda x: 1 if x == 52 else 0)

data_test['SuperBowlWeek'] = test_df['Week'].apply(lambda x: 1 if x == 6 else 0)
data_test['LaborDay'] = test_df['Week'].apply(lambda x: 1 if x == 36 else 0)
data_test['Tranksgiving'] = test_df['Week'].apply(lambda x: 1 if x == 47 else 0)
data_test['Christmas'] = test_df['Week'].apply(lambda x: 1 if x == 52 else 0)


cat_list = []
too_many_str_list = []
cat_int_list = []
for col in df_train.columns:
    if df_train[col].dtype == "object" and not(len(df_train[col].unique()) > 10):
        cat_list.append(col)
    elif (df_train[col].dtype == "object") and (len(df_train[col].unique()) > 10):
        too_many_str_list.append(col)
    elif not(df_train[col].dtype == "object" ) and (len(df_train[col].unique()) < 10):
        cat_int_list.append(col)


print(cat_list)
print(too_many_str_list)
print(cat_int_list)

df_train = data_train
df_test = data_test

df_train.fillna(0, inplace=True)
df_test['CPI'].fillna(df_test['CPI'].mean(), inplace = True)
df_test['Unemployment'].fillna(df_test['Unemployment'].mean(), inplace = True)
df_test.fillna(0, inplace=True)

df_train['IsHoliday'] = df_train['IsHoliday'].apply(lambda x: 1 if x == True else 0)
df_test['IsHoliday'] = df_test['IsHoliday'].apply(lambda x: 1 if x == True else 0)

df_train['Type'] = df_train['Type'].apply(lambda x: 1 if x == 'A' else (2 if x == 'B' else 3))
df_test['Type'] = df_test['Type'].apply(lambda x: 1 if x == 'A' else (2 if x == 'B' else 3))

features = [feature for feature in df_train.columns if feature not in ('Date','Weekly_Sales')]
X = df_train[features].copy()
df_test = df_test[features].copy()

print(f"train: {X.shape, Y.shape}, test: {df_test.shape}")
#df_test.info()
#X.info()

for col in df_test.columns:
    if col not in X.columns:
        print(col)