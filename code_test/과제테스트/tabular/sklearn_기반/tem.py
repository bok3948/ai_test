import pandas as pd
root = "/mnt/d/data/tabular/lol/games.csv"

import pandas as pd

DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/Jeju.csv'
DataUrl = "https://raw.githubusercontent.com/Datamanim/pandas/main/chipo.csv"
#df = pd.read_csv(DataUrl, encoding='euc-kr')
DataUrl = "https://raw.githubusercontent.com/Datamanim/pandas/main/AB_NYC_2019.csv"

#df = pd.read_csv(DataUrl)

df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/BankChurnersUp.csv',index_col=0)
Ans =df.shape

print(Ans.head())


import torch

def calculate_f1_score(y_true, y_pred, epsilon=1e-9):
    """
    Calculate F1 Score using PyTorch
    Args:
    y_true: tensor of shape (batch_size,) - ground truth labels
    y_pred: tensor of shape (batch_size,) - predicted labels
    
    Returns:
    f1_score: computed F1 Score
    """
    assert y_true.shape == y_pred.shape, "Shape of y_true and y_pred should be the same"
    
    y_true = y_true.bool()
    y_pred = y_pred.bool()
    
    true_positive = torch.sum(y_true & y_pred).float()
    false_positive = torch.sum(~y_true & y_pred).float()
    false_negative = torch.sum(y_true & ~y_pred).float()
    
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    
    return f1_score.item()

# Example usage
y_true = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.int32)
y_pred = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.int32)

f1_score = calculate_f1_score(y_true, y_pred)
print("F1 Score:", f1_score)

