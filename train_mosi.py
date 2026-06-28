import torch
import torch.nn as nn

from models.proposed_model import ProposedModel
from utils.dataloader import get_dataloaders
from tqdm import tqdm

# =====================================
# MOSI Dataset Path
# =====================================

DATA_PATH = "data/processed/mosi_aligned_50.pkl"

# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

# =====================================
# Data Loaders
# =====================================

train_loader, valid_loader, _ = get_dataloaders(
    DATA_PATH,
    batch_size=32
)

# =====================================
# Model
# MOSI:
# Audio = 5
# Vision = 20
# =====================================

model = ProposedModel(
    audio_dim=5,
    vision_dim=20
).to(device)

# =====================================
# Loss Functions
# =====================================

criterion_cls = nn.CrossEntropyLoss()

criterion_recon = nn.MSELoss()

# =====================================
# Optimizer
# =====================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =====================================
# Training Config
# =====================================

epochs = 20

best_val_acc = 0

# =====================================
# Training Loop
# =====================================

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for batch in tqdm(train_loader):

        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)

        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs, recovered_audio, audio_features = model(
            text,
            audio,
            vision
        )

        classification_loss = criterion_cls(
            outputs,
            labels
        )

        reconstruction_loss = criterion_recon(
            recovered_audio,
            audio_features
        )

        loss = (
            classification_loss +
            0.1 * reconstruction_loss
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_acc = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Train Acc: {train_acc:.2f}%"
    )

    # =====================================
    # Validation
    # =====================================

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for batch in valid_loader:

            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)

            labels = batch["label"].to(device)

            outputs, _, _ = model(
                text,
                audio,
                vision
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    val_acc = 100 * correct / total

    print(
        f"Validation Acc: {val_acc:.2f}%"
    )

    # =====================================
    # Save Best Model
    # =====================================

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_model_mosi.pt"
        )

        print("Best model saved")

print("\nTraining Complete!")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")