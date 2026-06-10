import torch

from models.modality_model import ModalityModel


model = ModalityModel()

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

output = model(
    text,
    audio,
    vision
)

print(output.shape)