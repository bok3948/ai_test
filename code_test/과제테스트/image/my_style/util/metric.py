import torch

def accuracy(logits, target, topk=1):
    batch_size = logits.shape[0]
    _, pred = logits.topk(topk, dim=1, largest=True, sorted=True)
    print(pred.shape)
    target = target.unsqueeze(dim=1).expand_as(pred)
    print(target.shape)
    result = pred.eq(target)
    print(result.shape)
    return result.float().sum() / batch_size * 100

