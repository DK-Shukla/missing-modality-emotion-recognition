import torch

from models.modality_model import ModalityModel

# create model
model = ModalityModel()

# dummy batch
text = torch.randn(32, 50, 768)
audio = torch.randn(32, 50, 74)
vision = torch.randn(32, 50, 35)

# forward pass
output = model(
    text,
    audio,
    vision
)

print("Output Shape:", output.shape)