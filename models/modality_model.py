import torch.nn as nn

from models.text_encoder import TextEncoder
from models.audio_encoder import AudioEncoder
from models.vision_encoder import VisionEncoder

from models.transformer_fusion import TransformerFusion

from models.classifier import SentimentClassifier


class ModalityModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.text_encoder = TextEncoder()

        self.audio_encoder = AudioEncoder()

        self.vision_encoder = VisionEncoder()

        self.fusion = TransformerFusion()

        self.classifier = SentimentClassifier()

    def forward(
        self,
        text,
        audio,
        vision
    ):

        text = self.text_encoder(text)

        audio = self.audio_encoder(audio)

        vision = self.vision_encoder(vision)

        fused = self.fusion(
            text,
            audio,
            vision
        )

        output = self.classifier(
            fused
        )

        return output