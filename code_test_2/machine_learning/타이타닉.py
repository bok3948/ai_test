import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os
root = "/mnt/d/data/tabular"
train_root = "/mnt/d/data/tabular/train.csv"
test_root = "/mnt/d/data/tabular/test.csv"
sample_root = "/mnt/d/data/tabular/tit_submission.csv"

train = pd.read_csv(train_root)
test = pd.read_csv(test_root)
sample = pd.read_csv(sample_root)

def check(df):
    print("-"*50)
    print()
    df.info()

for de in [train, test, sample]:
    check(de)

Y = train["Survived"]
train.drop(columns=["Survived"], axis=1, inplace=True)

ids = test["PassengerId"]

len_train = len(train)
df = pd.concat((train, test), ignore_index=True)

#feature enginnerring

def feat_cls(df):
    a, b, c, d = [], [], [], []
    for col in df.columns:
        if (df[col].dtype == "object") and (len(df[col].unique()) < 10):
            a.append(col)
        elif (df[col].dtype == "object") and (len(df[col].unique()) >= 10):
            b.append(col)
        elif not(df[col].dtype == "object") and (len(df[col].unique()) < 10):
            c.append(col)
        elif not(df[col].dtype == "object") and (len(df[col].unique()) >= 10):
            d.append(col)
    print(f"cat: {a}\n too_many: {b} \n int_cat: {c}\n numers {d}")
    return a,b,c,d

a,b,c,d = feat_cls(df)

#Name 

def name_processor(x):
    d = x.split(".")[0].split()[1]
    return d
df["Name"] = df["Name"].apply(name_processor).apply(pd.Series)

counts = df["Name"].value_counts().sort_values(ascending=False)
der = set(df["Name"])
for i in counts[:4].index:
    der.remove(i)

df["Name"] = df["Name"].replace(der, "rare")
print(set(df["Name"]))

import re
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names

# Group all non-common titles into one single grouping "Rare"
dataset= df
dataset['Title'] = dataset['Name'].apply(get_title)
dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
df =dataset

df["Family"] = df["SibSp"] + df["Parch"] + 1

df.drop(columns=[ 'Cabin', 'PassengerId', "Name"], axis=1, inplace=True)

#nan
nulls = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
#print(nulls)

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],  inplace=True)
df["Fare"].fillna(df["Fare"].mean(),  inplace=True)

#LE = LabelEncoder()
#df["Sex"] = LE.fit_transform(df["Sex"])
#df["Embarked"] = LE.fit_transform(df["Embarked"])
#df["Title"] = LE.fit_transform(df["Title"])
df["Pclass"] = df["Pclass"].astype(str)
df = pd.get_dummies(df, columns=["Ticket", "Sex", "Embarked", "Pclass", "Title"])

e = feat_cls(df)

x_train = df[:len_train]
x_test = df[len_train:].copy()

print(f" train: {x_train.shape, Y.shape} test: {x_test.shape}")

rfc = RandomForestClassifier(random_state=0)

## Search grid for optimal parameters
rf_param_grid = {
                "n_estimators": list(range(50, 200, 50)),
                "max_depth": [None, 10, 20],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False, True],
}


gsRFC = GridSearchCV(rfc,param_grid = rf_param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(x_train,Y)

RFC_best = gsRFC.best_estimator_



def inference(model, x_test):
    A = model.predict(x_test)
    sub = pd.DataFrame()
    sub["PassengerId"] = ids
    sub["Survived"] = A
    sub.to_csv(os.path.join(root, "submission.csv"), index=False)
inference(RFC_best, x_test)