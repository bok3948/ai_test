import os 
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder


root = "/mnt/d/data/tabular"
train_root = "/mnt/d/data/tabular/train.csv"
test_root = "/mnt/d/data/tabular/test.csv"
sample_root = "/mnt/d/data/tabular/tit_submission.csv"

train = pd.read_csv(train_root)
test = pd.read_csv(test_root)
sample = pd.read_csv(sample_root)

def check(df):
    print()
    print("-"*50)
    df.info()
    print(df.head())

for df in [train, test, sample]:
    check(df)

Y = train["Survived"]
train.drop(columns=["Survived"], axis=1, inplace=True)
test_ids = test["PassengerId"]

len_train = len(train)
df = pd.concat((train, test), ignore_index=True)

def feat_cls(df):
    a,b,c,d = [], [], [], []
    for col in df.columns:
        if (df[col].dtype == "object") and (len(df[col].unique()) > 10):
            a.append(col)
        elif (df[col].dtype == "object") and (len(df[col].unique()) <= 10):
            b.append(col)
        elif (df[col].dtype != "object") and (len(df[col].unique()) > 10):
            c.append(col)
        elif (df[col].dtype != "object") and (len(df[col].unique()) <= 10):
            d.append(col)

    print(f" too_many cat: {a}\n cat: {b}\n numer: {c}\n cat_int: {d}")
    return a,b,c,d
a,b,c,d = feat_cls(df)

nan = df.isnull().sum().sort_values(ascending=False)
print(nan)

for col in a:
    print(df[col].head())

def name_processor(x):
    title = x.split(".")[0].split()[1]
    a = ["Mr", "Miss", "Mrs", "Master"]
    if title in a:
        pass
    else:
        title = "rare"
    return title

df["title"] = df["Name"].apply(name_processor)
print(df["title"].value_counts(ascending=False))
df.drop(columns=["Ticket", "Cabin", "PassengerId", "Name"], axis=1, inplace=True)

df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Fare"].fillna(df["Fare"].mean(), inplace=True)

print(set(df["Sex"]), set(df["Embarked"]))

le = LabelEncoder()
df["title"] = le.fit_transform(df["title"])
df["Sex"] = df["Sex"].map({'female':0 , 'male': 1})
df["Embarked"] = df["Embarked"].map({'C': 0, 'Q': 1, 'S': 2})

check(df)

#unit variance로 만
#scaler = RobustScaler(centering=False)
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
df = df.astype(float)
a,b,c,d= feat_cls(df)

x_train = df[:len_train]
x_test = df[len_train:].copy()

print(f" train:{x_train.shape, Y.shape} test: {x_test.shape}")

rfc = RandomForestClassifier(random_state=0)

rfc.fit(x_train, Y)

def inference(model, x_test):
    A = model.predict(x_test)
    sub = pd.DataFrame()
    sub["PassengerId"] = test_ids
    sub["Survived"] = A
    sub.to_csv(os.path.join(root, "submission.csv"), index=False)

inference(rfc, x_test)

