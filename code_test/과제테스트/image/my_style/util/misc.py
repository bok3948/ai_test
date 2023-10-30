import torch

def save(to_save: dict, file_name):
    return torch.save(to_save, file_name)