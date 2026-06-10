import torch
import torch.nn as nn


class TransformerFusion(nn.Module):

    def __init__(self):

        super().__init__()

        # transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True
        )

        # stack 2 transformer layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(
        self,
        text_features,
        audio_features,
        vision_features
    ):

        # fuse modalities

        fused = (
            text_features +
            audio_features +
            vision_features
        )

        # learn cross-modal relationships

        output = self.transformer(
            fused
        )

        return output