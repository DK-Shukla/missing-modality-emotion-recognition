# load mosei dataset

import pickle
import torch
from torch.utils.data import Dataset


class MOSEIDataset(Dataset):

    def __init__(self, data, split="train"):

        self.data = data[split]

        self.text = self.data["text"]
        self.audio = self.data["audio"]
        self.vision = self.data["vision"]

        self.labels = self.data["classification_labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return {
            "text": torch.tensor(self.text[idx], dtype=torch.float32),
            "audio": torch.tensor(self.audio[idx], dtype=torch.float32),
            "vision": torch.tensor(self.vision[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }