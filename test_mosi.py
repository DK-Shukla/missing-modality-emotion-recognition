import torch

from models.proposed_model import ProposedModel

model = ProposedModel(
    audio_dim=5,
    vision_dim=20
)

text = torch.randn(2, 50, 768)
audio = torch.randn(2, 50, 5)
vision = torch.randn(2, 50, 20)

output, recovered_audio, audio_features = model(
    text,
    audio,
    vision
)

print("Output:", output.shape)
print("Recovered:", recovered_audio.shape)
print("Audio Features:", audio_features.shape)