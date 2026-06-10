import torch.nn as nn


class AudioEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # convert audio features
        self.linear = nn.Linear(74, 128)

        self.relu = nn.ReLU()

    def forward(self, x):

        # x shape:
        # [batch_size, 50, 74]

        x = self.linear(x)

        x = self.relu(x)

        return x