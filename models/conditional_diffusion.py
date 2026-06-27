import torch
import torch.nn as nn


class ConditionalDiffusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, semantic_features):

        recovered_features = self.network(
            semantic_features
        )

        return recovered_features