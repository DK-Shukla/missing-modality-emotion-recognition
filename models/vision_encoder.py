import torch.nn as nn

class VisionEncoder(nn.Module):

    def __init__(self, input_dim=35):

        super().__init__()

        self.linear = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.linear(x)
        x = self.relu(x)

        return x