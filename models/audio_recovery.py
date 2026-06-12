import torch.nn as nn


class AudioRecovery(nn.Module):

    def __init__(self):

        super().__init__()

        self.recovery = nn.Sequential(

            nn.Linear(256, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 256)

        )

    def forward(
        self,
        semantic_features
    ):

        recovered_audio = self.recovery(
            semantic_features
        )

        return recovered_audio