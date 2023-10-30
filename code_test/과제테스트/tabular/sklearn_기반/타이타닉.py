import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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

test_PassengerId = test["PassengerId"]

#EDA
#print(train["Name"][:5])
def name_processor(x):
    ans = x.split('.')[0].split()[1]
    return ans
tem_train = pd.DataFrame()  
tem_train["title"] = train["Name"].apply(name_processor)
print(Y.groupby(tem_train["title"]).mean())
print(tem_train["title"].value_counts())
title_list = ["Mr", "Miss", "Mrs", "Master"]

len_train = len(train)
df = pd.concat((train, test), ignore_index=True)

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

a,b,c,d = feat_cls(df)

df["title"] = df["Name"].apply(name_processor)

def remove_key(df, name_list):
    a = set(df)
    for dfdf in name_list:
        a.remove(dfdf)
    return a
rare = remove_key(df["title"], ["Mr", "Miss", "Mrs", "Master"])

df["title"] = df["title"].replace(rare, "rare")
df["name_len"] = df["Name"].apply(lambda x: len(x))
df["len_ticket"] = df["Ticket"].apply(lambda x: len(x))
print(df["title"].value_counts())

df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True)

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Fare"].fillna(df["Fare"].mean(), inplace=True)
df["Pclass"] = df["Pclass"].astype(str)

df = pd.get_dummies(df, columns=["Sex", "Embarked", "Pclass", "title"])

a,b,c,d= feat_cls(df)

x_train = df[:len_train]
x_test = df[len_train:].copy()

print(f" train: {x_train.shape, Y.shape} test: {x_test.shape}")

rfc = RandomForestClassifier(random_state=0)

#print(help(RandomForestClassifier))
grid_param = {
    "n_estimators" : list(range(100, 1000, 100)),
    #"max_depth": [None, 10, 20],
    #"min_samples_split": [2, 4],
   #"min_samples_leaf": [1, 3],
}
#grid_rfc = GridSearchCV(rfc, grid_param, cv=5)
grid_rfc = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
grid_rfc.fit(x_train, Y)
best_rfc= grid_rfc
print(best_rfc.oob_score_)
#best_rfc = grid_rfc.best_estimator_
impor = best_rfc.feature_importances_

show_df = pd.DataFrame()
show_df["importance"]  = pd.Series(impor)
show_df["name"] = x_train.columns
show_df = show_df.sort_values(by=["importance"], ascending=False).reset_index(drop=True)
print(show_df.iloc[:10])

def inference(model, x_test):
    A = model.predict(x_test)
    sub = pd.DataFrame()
    sub["PassengerId"] = test_PassengerId
    sub["Survived"] = A > 0.5
    sub.to_csv(os.path.join(root, "submission.csv"), index=False)

inference(best_rfc, x_test)
