import torch

from models.text_encoder import TextEncoder
from models.audio_encoder import AudioEncoder
from models.vision_encoder import VisionEncoder

from models.transformer_fusion import TransformerFusion


text = torch.randn(
    32,
    50,
    768
)

audio = torch.randn(
    32,
    50,
    74
)

vision = torch.randn(
    32,
    50,
    35
)

text_encoder = TextEncoder()
audio_encoder = AudioEncoder()
vision_encoder = VisionEncoder()

fusion = TransformerFusion()

text_out = text_encoder(text)

audio_out = audio_encoder(audio)

vision_out = vision_encoder(vision)

fused = fusion(
    text_out,
    audio_out,
    vision_out
)

print(fused.shape)