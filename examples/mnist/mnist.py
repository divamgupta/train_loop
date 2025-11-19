import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class MNISTDataset(torch.utils.data.Dataset):
    """
    MNIST dataset that returns dicts with keys 'inp' and 'gt'.
    """
    def __init__(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(
            root='/tmp/dataMN',
            train=train,
            download=True,
            transform=transform
        )
        self.collate_fn = None  # Use default collate function

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        inp, gt = self.dataset[idx]
        return {"inp": inp, "gt_class_id": gt}

class MNISTClassifier(nn.Module):
    def __init__(self, n_hidden=128):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x['inp']
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return {"pred_logits" : x}

# Example usage:
# dataset = MNISTDictDataset(train=True)  # or train=False for test set
# model = MNISTClassifier()
