import pickle

# load mosei dataset
import torch
from torch.utils.data import Dataset


class MOSIDataset(Dataset):

    def __init__(self, data, split="train"):

        self.data = data[split]

        self.text = self.data["text"]
        self.audio = self.data["audio"]
        self.vision = self.data["vision"]

        self.labels = self.data["classification_labels"]

        # store indices of positive and negative samples
        self.valid_indices = []

        for i, label in enumerate(self.labels):

            # ignore neutral samples
            if label != 1:
                self.valid_indices.append(i)

    def __len__(self):

        return len(self.valid_indices)

    def __getitem__(self, idx):

        # get actual index
        real_idx = self.valid_indices[idx]

        label = self.labels[real_idx]

        # convert to binary labels
        # 0 = negative
        # 1 = positive

        if label == 2:
            label = 1
        else:
            label = 0

        return {
            "text": torch.tensor(
                self.text[real_idx],
                dtype=torch.float32
            ),

            "audio": torch.tensor(
                self.audio[real_idx],
                dtype=torch.float32
            ),

            "vision": torch.tensor(
                self.vision[real_idx],
                dtype=torch.float32
            ),

            "label": torch.tensor(
                label,
                dtype=torch.long
            )
        }