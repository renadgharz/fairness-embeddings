import torch

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    protected_attrs = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return images, labels, protected_attrs