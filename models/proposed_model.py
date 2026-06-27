import torch.nn as nn

from models.text_encoder import TextEncoder
from models.audio_encoder import AudioEncoder
from models.vision_encoder import VisionEncoder
from models.cross_modal_attention import CrossModalAttention
from models.semantic_transformer import SemanticTransformer
from models.audio_recovery import AudioRecovery

from models.classifier import SentimentClassifier


class ProposedModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.text_encoder = TextEncoder()
        self.audio_encoder = AudioEncoder()
        self.vision_encoder = VisionEncoder()
        self.cross_attention = CrossModalAttention()
        self.semantic_transformer = SemanticTransformer()

        self.audio_recovery = AudioRecovery()

        self.classifier = SentimentClassifier()

    def forward(
        self,
        text,
        audio,
        vision
    ):

        text_features = self.text_encoder(text)

        audio_features = self.audio_encoder(audio)

        vision_features = self.vision_encoder(vision)

        enhanced_text = self.cross_attention(
            text_features,
            audio_features,
            vision_features
          )

        semantic_features = self.semantic_transformer(
            text_features,
            audio_features,
            vision_features
        )

        recovered_audio = self.audio_recovery(
            semantic_features
        )

        enhanced_features = (
            semantic_features +
            recovered_audio
        )

        output = self.classifier(
            enhanced_features
        )

        return (
            output,
            recovered_audio,
            audio_features
        )