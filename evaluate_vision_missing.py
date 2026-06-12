import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from models.modality_model import ModalityModel
from utils.dataloader import get_dataloaders


DATA_PATH = "/content/drive/MyDrive/Modality/aligned_50.pkl"

MODEL_PATH = "/content/drive/MyDrive/Modality/best_model.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

# load model

model = ModalityModel().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

# load test data

_, _, test_loader = get_dataloaders(
    DATA_PATH,
    batch_size=32
)

all_labels = []
all_predictions = []

with torch.no_grad():

    for batch in test_loader:

        text = batch["text"].to(device)

        audio = batch["audio"].to(device)

        vision = torch.zeros_like(
                 batch["vision"]
                ).to(device)

        labels = batch["label"].to(device)

        outputs = model(
            text,
            audio,
            vision
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

# metrics

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nResults")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)

print("\nConfusion Matrix")

print(cm)