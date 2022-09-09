from torch.utils.data import Dataset
import torch

class DmriDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = torch.tensor(data)
        self.targets = torch.tensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)
