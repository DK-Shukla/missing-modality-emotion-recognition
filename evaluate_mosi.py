import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pickle

from models.proposed_model import ProposedModel
from datasets.mosi_dataset import MOSIDataset

# ==========================
# CONFIG
# ==========================
DATA_PATH = "/content/drive/MyDrive/Modality/mosi_aligned_50.pkl"
MODEL_PATH = "best_model_mosi.pt"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==========================
# LOAD DATA
# ==========================
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

test_dataset = MOSIDataset(data, split="test")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# LOAD MODEL
# ==========================
model = ProposedModel(
    audio_dim=5,
    vision_dim=20
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()

# ==========================
# EVALUATION
# ==========================
all_preds = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["label"].to(device)

        outputs, _, _ = model(
            text,
            audio,
            vision
        )

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

# ==========================
# METRICS
# ==========================
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n========== MOSI RESULTS ==========")
print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        digits=4
    )
)