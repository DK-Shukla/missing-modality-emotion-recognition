import torch
import torch.nn as nn


class SemanticTransformer(nn.Module):

    def __init__(self):

        super().__init__()

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

        fused = (
          text_features +
          audio_features +
          vision_features
     )

        semantic_features = self.transformer(
            fused
        )

        return semantic_features