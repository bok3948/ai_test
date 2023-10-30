import torch
import torch.nn as nn
import util.metric as metric

def train_one_epoch(loader: nn.Module, model, criterion: nn.Module, optimizer, dataset_size, epoch):
    
    log = {"loss": None}
    epoch_loss, pri_freq = 0, 100
    model.train()
    for i, (x, y) in enumerate(loader):
        x, y = x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)

        logits = model(x)
        logits = logits.squeeze()
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % pri_freq == 0:
            print(f"Epoch [{epoch}] [{i}/{len(loader)}] loss: {loss.item():.6f}")

        epoch_loss += loss.item() * x.shape[0]
        
    log["loss"] = epoch_loss  / dataset_size
    return log


def evaluate(loader: nn.Module, model, criterion, dataset_size):

    log = {"loss": None, "acc1": None, "acc5": None}
    epoch_loss, epoch_acc1, epoch_acc5 = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)

            logits = model(x)
            logits = logits.squeeze()
            loss = criterion(logits, y)


            epoch_loss += loss.item() 
            epoch_acc1 += metric.accuracy(logits, y, 1)
            epoch_acc5 += metric.accuracy(logits, y, 5)

    log["acc1"] = epoch_acc1 / len(loader)
    log["acc5"] = epoch_acc5 / len(loader)
    log["loss"] = loss / dataset_size
    return log


