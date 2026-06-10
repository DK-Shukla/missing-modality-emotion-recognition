import torch.nn as nn

from models.text_encoder import TextEncoder
from models.audio_encoder import AudioEncoder
from models.vision_encoder import VisionEncoder

from models.transformer_fusion import TransformerFusion
from models.classifier import SentimentClassifier


class ModalityModel(nn.Module):

    def __init__(self):

        super().__init__()

        # modality encoders

        self.text_encoder = TextEncoder()

        self.audio_encoder = AudioEncoder()

        self.vision_encoder = VisionEncoder()

        # fusion module

        self.fusion = TransformerFusion()

        # sentiment classifier

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

        fused_features = self.fusion(
            text_features,
            audio_features,
            vision_features
        )

        output = self.classifier(
            fused_features
        )

        return output