import torch.nn as nn


class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # convert 768-dim text features into 256-dim features
        self.linear = nn.Linear(768, 256)

        self.relu = nn.ReLU()

    def forward(self, x):

        # x shape:
        # [batch_size, 50, 768]

        x = self.linear(x)

        x = self.relu(x)

        return x