import torch
import torch.nn as nn

from models.cross_modal_attention import CrossModalAttention


class TransformerFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.cross_attention = CrossModalAttention()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dropout=0.3,
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

        enhanced_text = self.cross_attention(
            text_features,
            audio_features,
            vision_features
        )

        fused = torch.cat(
            [
                enhanced_text,
                audio_features,
                vision_features
            ],
            dim=1
        )

        output = self.transformer(
            fused
        )

        return output