import torch
import torch.nn as nn


class TransformerFusion(nn.Module):

    def __init__(self):

        super().__init__()

        # make all modalities same dimension

        self.audio_proj = nn.Linear(128, 256)

        self.vision_proj = nn.Linear(128, 256)

        # transformer encoder

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True
        )

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

        audio_features = self.audio_proj(
            audio_features
        )

        vision_features = self.vision_proj(
            vision_features
        )

        # combine all modalities

        fused = (
            text_features +
            audio_features +
            vision_features
        )

        output = self.transformer(
            fused
        )

        return output