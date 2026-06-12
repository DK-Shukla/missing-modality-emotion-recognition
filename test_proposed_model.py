import torch

from models.proposed_model import ProposedModel


model = ProposedModel()

text = torch.randn(
    8,
    50,
    768
)

audio = torch.randn(
    8,
    50,
    74
)

vision = torch.randn(
    8,
    50,
    35
)

output, recovered_audio, audio_features = model(
    text,
    audio,
    vision
)

print(output.shape)

print(recovered_audio.shape)

print(audio_features.shape)