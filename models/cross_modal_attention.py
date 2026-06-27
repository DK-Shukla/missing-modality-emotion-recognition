import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )

    def forward(
        self,
        text_features,
        audio_features,
        vision_features
    ):

        av_features = (
            audio_features +
            vision_features
        )

        enhanced_text, _ = self.attention(
            text_features,
            av_features,
            av_features
        )

        enhanced_text = (
            enhanced_text +
            text_features
        )

        return enhanced_text