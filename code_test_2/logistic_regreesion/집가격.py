import pandas as pd
import os

pd.set_option("display.max_rows", None)
root = "/mnt/d/data/tabular/house-prices-advanced-regression-techniques"
train_root = os.path.join(root, "train.csv")
test_root = os.path.join(root, "test.csv")

train_data = pd.read_csv(train_root)
test_data = pd.read_csv(test_root)

#train_data.info()
#print(train_data.head(5))
dfd = train_data.isna().sum().sort_values(ascending=False)
Y_train = train_data["SalePrice"].to_numpy()
train_data.fillna(None, inplace=True)
drop_list = [
"PoolQC" ,       
"MiscFeature" ,  
"Alley"      ,   
"Fence"        ,
"MasVnrType"     , 
"FireplaceQu"   ,  
"LotFrontage",
"SalePrice",
]
train_data.drop(drop_list, axis=1, inplace=True)
X_train = pd.get_dummies(train_data)
X_train.info()
print(X_train.columns)
