import torch.nn as nn


class VisionEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # convert visual features
        self.linear = nn.Linear(35, 256)

        self.relu = nn.ReLU()

    def forward(self, x):

        # x shape:
        # [batch_size, 50, 35]

        x = self.linear(x)

        x = self.relu(x)

        return x