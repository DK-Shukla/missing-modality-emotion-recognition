import torch.nn as nn


class SentimentClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(256, 128)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):

        # x shape:
        # [batch_size, 50, 256]

        # average over sequence length

        x = x.mean(dim=1)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x